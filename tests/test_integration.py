#!/usr/bin/env python3
"""
Integration smoke test: one forward + backward pass through the full model
with synthetic data (no pretrained weights needed — uses small random models).
"""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.moe_lora import MoELoRA, CrossModalFusion


class _SmallTextEncoder(nn.Module):
    """Tiny text encoder substitute for integration testing."""
    def __init__(self, vocab_size=100, hidden_dim=64, seq_len=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim)
    
    def forward(self, input_ids, attention_mask=None):
        hidden = self.embed(input_ids)
        pooled = hidden[:, 0, :]
        return hidden, pooled


class _SmallVisionEncoder(nn.Module):
    """Tiny vision encoder substitute for integration testing."""
    def __init__(self, hidden_dim=64, num_patches=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.proj = nn.Linear(3 * 16 * 16, hidden_dim)

    def forward(self, pixel_values):
        bs = pixel_values.shape[0]
        # Flatten and project to hidden_dim patches
        flat = pixel_values.view(bs, 3, -1)[:, :, :16*16].reshape(bs, 3*16*16)
        patch = self.proj(flat).unsqueeze(1).expand(bs, self.num_patches, self.hidden_dim)
        pooled = patch[:, 0, :]
        return patch, pooled


class SmallMultimodalModel(nn.Module):
    """Tiny multimodal model that mirrors the real architecture without pretrained weights."""
    def __init__(self, hidden_dim=64, num_experts=2, num_labels=2, label_smoothing=0.0):
        super().__init__()
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        self.text_encoder = _SmallTextEncoder(hidden_dim=hidden_dim)
        self.vision_encoder = _SmallVisionEncoder(hidden_dim=hidden_dim)
        self.cross_modal_fusion = CrossModalFusion(
            text_dim=hidden_dim, vision_dim=hidden_dim,
            hidden_dim=hidden_dim, num_heads=4, num_layers=1,
        )
        self.moe_lora = MoELoRA(
            hidden_dim=hidden_dim, num_experts=num_experts,
            lora_r=4, lora_alpha=8.0,
        )
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.contrastive_weight = 0.1
        self.diversity_weight = 0.1

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        text_hidden, text_pooled = self.text_encoder(input_ids, attention_mask)
        vision_hidden, vision_pooled = self.vision_encoder(pixel_values)

        fused = self.cross_modal_fusion(text_hidden, vision_hidden, attention_mask)
        moe_out, diversity_loss = self.moe_lora(fused, return_aux_loss=True)

        cls_features = moe_out[:, 0, :]
        logits = self.classifier(cls_features)

        result = {"logits": logits}
        if labels is not None:
            import torch.nn.functional as F
            ce = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
            # Simple contrastive proxy
            text_norm = F.normalize(text_pooled, dim=-1)
            vis_norm = F.normalize(vision_pooled, dim=-1)
            sim = (text_norm * vis_norm).sum(dim=-1).mean()
            result["loss"] = ce + self.diversity_weight * diversity_loss - self.contrastive_weight * sim

        return result


class TestIntegrationSmoke:
    """End-to-end smoke tests."""

    def test_forward_backward(self):
        """Full forward + backward pass with synthetic data."""
        model = SmallMultimodalModel(hidden_dim=64, num_experts=2, num_labels=2)
        
        batch_size = 4
        seq_len = 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        pixel_values = torch.randn(batch_size, 3, 32, 32)
        labels = torch.randint(0, 2, (batch_size,))

        outputs = model(input_ids, attention_mask, pixel_values, labels=labels)

        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (batch_size, 2)
        assert outputs["loss"].ndim == 0
        assert torch.isfinite(outputs["loss"])

        # Backward
        outputs["loss"].backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_forward_without_labels(self):
        """Forward pass without labels should not compute loss."""
        model = SmallMultimodalModel()
        input_ids = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.bool)
        pixel_values = torch.randn(2, 3, 32, 32)

        outputs = model(input_ids, attention_mask, pixel_values)
        assert "logits" in outputs
        assert "loss" not in outputs

    def test_label_smoothing(self):
        """Model with label_smoothing should still produce finite loss."""
        model = SmallMultimodalModel(label_smoothing=0.1)
        input_ids = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.bool)
        pixel_values = torch.randn(2, 3, 32, 32)
        labels = torch.randint(0, 2, (2,))

        outputs = model(input_ids, attention_mask, pixel_values, labels=labels)
        assert torch.isfinite(outputs["loss"])

    def test_multiple_forward_passes(self):
        """Multiple forward passes should all work (no stale state)."""
        model = SmallMultimodalModel()
        for _ in range(3):
            input_ids = torch.randint(0, 100, (2, 16))
            attention_mask = torch.ones(2, 16, dtype=torch.bool)
            pixel_values = torch.randn(2, 3, 32, 32)
            labels = torch.randint(0, 2, (2,))
            outputs = model(input_ids, attention_mask, pixel_values, labels=labels)
            assert torch.isfinite(outputs["loss"])

    def test_training_step(self):
        """Simulate a single training step with optimizer."""
        model = SmallMultimodalModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        input_ids = torch.randint(0, 100, (4, 16))
        attention_mask = torch.ones(4, 16, dtype=torch.bool)
        pixel_values = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 2, (4,))

        # Step 1
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, pixel_values, labels=labels)
        loss1 = outputs["loss"].item()
        outputs["loss"].backward()
        optimizer.step()

        # Step 2 (loss should generally change)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, pixel_values, labels=labels)
        loss2 = outputs["loss"].item()

        # Both losses should be finite
        assert torch.isfinite(torch.tensor(loss1))
        assert torch.isfinite(torch.tensor(loss2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
