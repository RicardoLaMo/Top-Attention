"""
Smoke tests for TabTransformer model.
Tests forward pass shapes, attention weight extraction, and gradient flow.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from src.models.tab_transformer import TabTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_small():
    """A small TabTransformer for fast testing."""
    return TabTransformer(
        num_categories_per_col=[10, 20, 5, 15, 8],  # 5 categorical columns
        num_continuous=3,
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
        dropout=0.0,
    )


@pytest.fixture
def batch_data():
    """Fake batch of 4 samples, 5 cat cols, 3 cont cols."""
    x_cat = torch.randint(0, 5, (4, 5))
    x_cont = torch.randn(4, 3)
    return x_cat, x_cont


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTabTransformer:
    def test_forward_shape(self, model_small, batch_data):
        """Logits should be (batch, 1) for binary classification."""
        x_cat, x_cont = batch_data
        out = model_small(x_cat, x_cont)
        assert out["logits"].shape == (4, 1)

    def test_forward_no_continuous(self):
        """Model should work with zero continuous features."""
        model = TabTransformer(
            num_categories_per_col=[10, 20],
            num_continuous=0,
            d_model=16, n_heads=4, n_layers=1,
        )
        x_cat = torch.randint(0, 5, (4, 2))
        out = model(x_cat, x_cont=None)
        assert out["logits"].shape == (4, 1)

    def test_embeddings_returned(self, model_small, batch_data):
        """return_embeddings=True should include contextual embeddings."""
        x_cat, x_cont = batch_data
        out = model_small(x_cat, x_cont, return_embeddings=True)
        assert "embeddings" in out
        # (batch, num_cat_cols, d_model)
        assert out["embeddings"].shape == (4, 5, 16)

    def test_attention_weights(self, model_small, batch_data):
        """Attention weights should exist for each layer."""
        x_cat, x_cont = batch_data
        out = model_small(x_cat, x_cont)
        attn_list = out["attn_weights"]
        assert len(attn_list) == 2  # n_layers=2
        for aw in attn_list:
            assert aw.shape == (4, 5, 5)  # (batch, num_cols, num_cols)

    def test_gradient_flow(self, model_small, batch_data):
        """Gradients should flow through the entire model."""
        x_cat, x_cont = batch_data
        out = model_small(x_cat, x_cont)
        loss = out["logits"].sum()
        loss.backward()
        # Check that column embeddings received gradients
        for emb in model_small.col_embed.col_embeddings:
            assert emb.weight.grad is not None
            assert emb.weight.grad.abs().sum() > 0

    def test_attn_bias_hook(self, model_small, batch_data):
        """attn_bias should not crash the forward pass (Phase 3 prep)."""
        x_cat, x_cont = batch_data
        batch_size = x_cat.size(0)  # 4
        n_heads = 4
        n_cols = 5
        # PyTorch MHA expects 3D attn_mask as (batch*n_heads, seq, seq)
        bias = torch.zeros(batch_size * n_heads, n_cols, n_cols)
        out = model_small(x_cat, x_cont, attn_bias=bias)
        assert out["logits"].shape == (4, 1)

    def test_parameter_count(self, model_small):
        """Model should have a reasonable number of parameters."""
        n = model_small.num_parameters
        assert n > 0
        assert n < 1_000_000  # small model should be well under 1M


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
