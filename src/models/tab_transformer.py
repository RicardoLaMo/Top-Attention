"""
TabTransformer: Tabular Data Modeling Using Contextual Embeddings
================================================================
Faithful implementation of Huang et al. (2020) — arXiv:2012.06678

Architecture:
  1. Column Embeddings: Each categorical column gets a learnable embedding
     + a shared column-type embedding for positional context.
  2. Transformer Encoder: Multi-head self-attention across column embeddings
     produces contextual embeddings that capture inter-column dependencies.
  3. Continuous Features: Passed through LayerNorm, concatenated with
     contextual embeddings.
  4. MLP Head: Classification head on the concatenated representation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class ColumnEmbedding(nn.Module):
    """Embeds each categorical column into a shared embedding space.

    Each categorical column has its own embedding table. A learned
    column-type embedding is added so the Transformer knows *which*
    column a token came from (analogous to segment embeddings in BERT).
    """

    def __init__(
        self,
        num_categories_per_col: List[int],
        d_model: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_cols = len(num_categories_per_col)

        # Per-column embedding tables  (+1 for unknown/missing)
        # No padding_idx: the missing-value embedding (index 0) is learnable,
        # matching Huang et al. (2020) Section 2 — improves robustness to missing data.
        self.col_embeddings = nn.ModuleList([
            nn.Embedding(num_cat + 1, d_model)
            for num_cat in num_categories_per_col
        ])

        # Shared column-type positional embedding
        self.col_type_embed = nn.Embedding(self.num_cols, d_model)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_cat: (batch, num_cat_cols) integer-encoded categoricals

        Returns:
            (batch, num_cat_cols, d_model) embedded + positionally encoded
        """
        batch_size = x_cat.size(0)
        col_indices = torch.arange(self.num_cols, device=x_cat.device)
        col_type = self.col_type_embed(col_indices)  # (num_cols, d_model)

        embeddings = []
        for i, emb_layer in enumerate(self.col_embeddings):
            col_emb = emb_layer(x_cat[:, i])          # (batch, d_model)
            embeddings.append(col_emb)

        # Stack → (batch, num_cols, d_model), add column-type PE
        out = torch.stack(embeddings, dim=1)           # (batch, num_cols, d_model)
        out = out + col_type.unsqueeze(0)              # broadcast
        return out


class ContinuousNormalizer(nn.Module):
    """Projects continuous features through LayerNorm + linear projection."""

    def __init__(self, num_continuous: int, d_model: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(num_continuous)
        self.proj = nn.Linear(num_continuous, d_model)

    def forward(self, x_cont: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_cont: (batch, num_cont_cols)

        Returns:
            (batch, d_model)
        """
        return self.proj(self.norm(x_cont))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LayerNorm → MHA → Residual → LayerNorm → FFN → Residual."""

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 8,
        d_ff: int = 128,
        dropout: float = 0.1,
        attn_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            attn_bias: optional (batch*n_heads, seq_len, seq_len) additive bias
                       for future TopAttention extension (Phase 3).

        Returns:
            (output, attention_weights)
        """
        # Pre-norm MHA
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_bias,
            need_weights=True,
            average_attn_weights=True,
        )
        x = residual + attn_out

        # Pre-norm FFN
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x, attn_weights


# ---------------------------------------------------------------------------
# TabTransformer
# ---------------------------------------------------------------------------

class TabTransformer(nn.Module):
    """
    TabTransformer (Huang et al., 2020)

    Applies multi-layer self-attention across categorical column embeddings,
    concatenates with normalized continuous features, and classifies via MLP.

    Designed for extension: the `attn_bias` hook allows Phase 3 (TopAttention)
    to inject GW transport structure into the attention computation.
    """

    def __init__(
        self,
        num_categories_per_col: List[int],
        num_continuous: int,
        d_model: int = 32,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 128,
        dropout: float = 0.1,
        mlp_hidden: List[int] = None,
        num_classes: int = 1,
    ):
        """
        Args:
            num_categories_per_col: List of cardinality per categorical column.
            num_continuous: Number of continuous features.
            d_model: Embedding / Transformer hidden dimension.
            n_heads: Number of attention heads per layer.
            n_layers: Number of Transformer blocks.
            d_ff: Feed-forward hidden dimension.
            dropout: Dropout rate.
            mlp_hidden: Hidden layer sizes for the classification MLP.
            num_classes: 1 for binary classification (sigmoid).
        """
        super().__init__()

        if mlp_hidden is None:
            mlp_hidden = [128, 64]

        self.num_cat_cols = len(num_categories_per_col)
        self.num_continuous = num_continuous
        self.d_model = d_model
        self.n_layers = n_layers

        # Categorical column embeddings
        self.col_embed = ColumnEmbedding(num_categories_per_col, d_model)

        # Continuous feature normalizer
        self.cont_norm = ContinuousNormalizer(num_continuous, d_model) if num_continuous > 0 else None

        # Transformer encoder stack
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final LayerNorm on contextual embeddings
        self.final_norm = nn.LayerNorm(d_model)

        # MLP classification head
        # Input: flattened contextual embeddings + continuous projection
        mlp_input_dim = self.num_cat_cols * d_model
        if num_continuous > 0:
            mlp_input_dim += d_model

        layers = []
        prev_dim = mlp_input_dim
        for h in mlp_hidden:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp_head = nn.Sequential(*layers)

        # Store attention weights for interpretability
        self._attn_weights: List[torch.Tensor] = []

    def forward(
        self,
        x_cat: torch.Tensor,
        x_cont: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_cat: (batch, num_cat_cols) integer-encoded categoricals.
            x_cont: (batch, num_cont_cols) continuous features.
            attn_bias: optional additive attention bias (for TopAttention Phase 3).
            return_embeddings: if True, also return contextual embeddings.

        Returns:
            dict with keys:
                "logits": (batch, num_classes) raw logits
                "embeddings": (batch, num_cat_cols, d_model) if return_embeddings
                "attn_weights": list of (batch, num_cat_cols, num_cat_cols) per layer
        """
        # Embed categorical columns
        h = self.col_embed(x_cat)  # (batch, num_cat_cols, d_model)

        # Pass through Transformer blocks
        self._attn_weights = []
        for block in self.transformer_blocks:
            h, aw = block(h, attn_bias=attn_bias)
            self._attn_weights.append(aw)

        # Final norm
        h = self.final_norm(h)  # (batch, num_cat_cols, d_model)

        # Flatten contextual embeddings
        h_flat = h.reshape(h.size(0), -1)  # (batch, num_cat_cols * d_model)

        # Concatenate with continuous features
        if self.cont_norm is not None and x_cont is not None:
            c = self.cont_norm(x_cont)  # (batch, d_model)
            h_flat = torch.cat([h_flat, c], dim=-1)

        # Classify
        logits = self.mlp_head(h_flat)  # (batch, num_classes)

        result = {"logits": logits, "attn_weights": self._attn_weights}
        if return_embeddings:
            result["embeddings"] = h
        return result

    def get_attention_maps(self) -> List[torch.Tensor]:
        """Return attention weights from the last forward pass."""
        return self._attn_weights

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
