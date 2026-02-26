"""
TabTransformer training script.

Usage:
  python src/train_baseline.py --data-dir data --sample-size 50000 --max-epochs 30
  python src/train_baseline.py --config configs/default.yaml
"""

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.tab_transformer import TabTransformer
from src.data.data_loader import (
    load_ieee_cis,
    load_generic_csv,
    get_column_info,
    prepare_data,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train TabTransformer baseline")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Path to specific CSV (overrides --data-dir)")
    parser.add_argument("--target-col", type=str, default="isFraud")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="output/phase1")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (overrides CLI args)")
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def compute_class_weights(dataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced data."""
    y = dataset.y.numpy()
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    # Positive class weight = n_neg / n_pos (upweight minority)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    return pos_weight


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        x_cat = batch["x_cat"].to(device)
        x_cont = batch["x_cont"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        out = model(x_cat, x_cont)
        logits = out["logits"].squeeze(-1)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_probs = []
    all_labels = []

    for batch in loader:
        x_cat = batch["x_cat"].to(device)
        x_cont = batch["x_cont"].to(device)
        y = batch["y"].to(device)

        out = model(x_cat, x_cont)
        logits = out["logits"].squeeze(-1)
        loss = criterion(logits, y)

        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)

    return avg_loss, roc_auc, pr_auc


def main():
    args = parse_args()

    # Optional YAML config override
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k.replace("-", "_")):
                setattr(args, k.replace("-", "_"), v)

    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load data
    if args.data_file:
        df = load_generic_csv(args.data_file, args.target_col, args.sample_size)
    else:
        df = load_ieee_cis(args.data_dir, args.sample_size)

    col_info = get_column_info(df, target_col=args.target_col)
    print(f"Categorical columns: {len(col_info['categorical'])}")
    print(f"Continuous columns: {len(col_info['continuous'])}")

    # Prepare datasets
    data = prepare_data(df, col_info)
    train_ds, val_ds, test_ds = data["train"], data["val"], data["test"]
    encoder = data["encoder"]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2)

    # Build model
    model = TabTransformer(
        num_categories_per_col=encoder.num_categories_per_col,
        num_continuous=len(col_info["continuous"]),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    print(f"Model parameters: {model.num_parameters:,}")

    # Loss & optimizer
    pos_weight = compute_class_weights(train_ds).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs
    )

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_auc = 0
    patience_counter = 0

    print(f"\n{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>10} {'Val AUC':>10} {'Val PR-AUC':>12} {'LR':>10}")
    print("-" * 65)

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, val_pr_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        print(f"{epoch:5d} {train_loss:12.4f} {val_loss:10.4f} {val_auc:10.4f} {val_pr_auc:12.4f} {lr:10.2e}  ({elapsed:.1f}s)")

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    # Final evaluation on test set
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"), weights_only=True))
    test_loss, test_auc, test_pr_auc = evaluate(model, test_loader, criterion, device)
    print(f"\n{'='*50}")
    print(f"TEST RESULTS:")
    print(f"  ROC-AUC:  {test_auc:.4f}")
    print(f"  PR-AUC:   {test_pr_auc:.4f}")
    print(f"  Loss:     {test_loss:.4f}")
    print(f"{'='*50}")

    # Log experiment
    log_entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "category": "TabTransformer-Baseline",
        "parent_idea": None,
        "prd_link": "docs/prd/drill_baseline.md",
        "git_branch": "feat/phase1-tabtransformer",
        "idea": "TabTransformer baseline: column-level self-attention on categorical embeddings",
        "parameters": {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "sample_size": args.sample_size,
        },
        "results": {
            "test_roc_auc": test_auc,
            "test_pr_auc": test_pr_auc,
            "best_val_roc_auc": best_val_auc,
            "num_params": model.num_parameters,
        },
        "is_champion": True,
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/experiment_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"\nExperiment logged to artifacts/experiment_log.jsonl")
    print(f"Best model saved to {args.output_dir}/best_model.pt")


if __name__ == "__main__":
    main()
