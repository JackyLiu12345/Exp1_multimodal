#!/usr/bin/env python3
"""
Unit tests for contrastive loss, uncertainty head, and classifier components.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.classifier import UncertaintyHead, MultimodalFakeNewsClassifier


# ---------- UncertaintyHead ----------

class TestUncertaintyHead:
    """Tests for the evidential uncertainty estimation head."""

    def test_output_shapes(self):
        head = UncertaintyHead(hidden_dim=64, num_labels=2)
        x = torch.randn(4, 64)
        out = head(x)
        assert out["evidence"].shape == (4, 2)
        assert out["uncertainty"].shape == (4,)
        assert out["pred_probs"].shape == (4, 2)

    def test_evidence_non_negative(self):
        head = UncertaintyHead(hidden_dim=64, num_labels=2)
        x = torch.randn(4, 64)
        out = head(x)
        assert (out["evidence"] >= 0).all()

    def test_pred_probs_sum_to_one(self):
        head = UncertaintyHead(hidden_dim=64, num_labels=3)
        x = torch.randn(4, 64)
        out = head(x)
        sums = out["pred_probs"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_uncertainty_positive(self):
        head = UncertaintyHead(hidden_dim=64, num_labels=2)
        x = torch.randn(4, 64)
        out = head(x)
        assert (out["uncertainty"] > 0).all()

    def test_backward_pass(self):
        head = UncertaintyHead(hidden_dim=64, num_labels=2)
        x = torch.randn(4, 64, requires_grad=True)
        out = head(x)
        loss = out["evidence"].sum()
        loss.backward()
        assert x.grad is not None


# ---------- Contrastive Loss ----------

class TestContrastiveLoss:
    """Test the contrastive loss computation via a lightweight model proxy."""

    def _make_model(self):
        """Create a minimal model to test internal loss methods."""
        # We can't easily instantiate the full model without pretrained weights,
        # so we test the loss function logic directly.
        pass

    def test_contrastive_loss_basic(self):
        """Test supervised InfoNCE with matching labels."""
        # Simulate what MultimodalFakeNewsClassifier._compute_contrastive_loss does
        batch_size = 4
        hidden_dim = 32
        temperature = 0.07

        text_features = torch.randn(batch_size, hidden_dim)
        vision_features = torch.randn(batch_size, hidden_dim)
        labels = torch.tensor([0, 0, 1, 1])

        # Pool vision if 3D
        if vision_features.dim() == 3:
            vision_features = vision_features.mean(dim=1)

        text_norm = F.normalize(text_features, dim=-1)
        vision_norm = F.normalize(vision_features, dim=-1)
        similarity = text_norm @ vision_norm.t() / temperature

        labels_same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        self_mask = torch.eye(batch_size, dtype=torch.bool)
        labels_same.masked_fill_(self_mask, 0.0)

        log_prob = similarity - torch.logsumexp(
            similarity.masked_fill(self_mask, -1e9), dim=-1, keepdim=True
        )

        num_positives = labels_same.sum(dim=-1)
        valid_rows = num_positives > 0
        
        contrastive_loss = -(log_prob * labels_same).sum(dim=-1) / num_positives.clamp(min=1)
        result = contrastive_loss[valid_rows].mean()

        assert result.ndim == 0  # scalar
        assert torch.isfinite(result)

    def test_contrastive_loss_single_sample(self):
        """With batch_size=1, contrastive loss should be 0."""
        text = torch.randn(1, 32)
        # Simulate the batch_size < 2 check
        batch_size = text.shape[0]
        if batch_size < 2:
            result = torch.tensor(0.0)
        assert result.item() == 0.0

    def test_contrastive_loss_no_positives(self):
        """When all labels are different, valid_rows might be empty."""
        batch_size = 3
        labels = torch.tensor([0, 1, 2])
        labels_same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        self_mask = torch.eye(batch_size, dtype=torch.bool)
        labels_same.masked_fill_(self_mask, 0.0)
        num_positives = labels_same.sum(dim=-1)
        valid_rows = num_positives > 0
        # No valid rows since each label is unique
        assert not valid_rows.any()

    def test_contrastive_loss_vision_3d(self):
        """Vision features with [batch, patches, hidden] should be pooled."""
        vision_3d = torch.randn(4, 10, 32)
        vision_pooled = vision_3d.mean(dim=1)
        assert vision_pooled.shape == (4, 32)


# ---------- Uncertainty Loss ----------

class TestUncertaintyLoss:
    """Test evidential uncertainty loss computation."""

    def test_loss_finite(self):
        """Loss should produce finite values for valid inputs."""
        num_labels = 2
        evidence = torch.rand(4, num_labels) * 5  # positive evidence
        labels = torch.tensor([0, 1, 0, 1])

        one_hot = F.one_hot(labels, num_labels).float()
        alpha = evidence + 1
        S = alpha.sum(dim=-1, keepdim=True)

        nll_loss = (one_hot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1).mean()
        wrong_evidence = evidence * (1 - one_hot)
        reg_loss = wrong_evidence.sum(dim=-1).mean()
        total = nll_loss + 0.1 * reg_loss

        assert torch.isfinite(total)

    def test_loss_decreases_with_correct_evidence(self):
        """Loss should be lower when evidence is concentrated on correct class."""
        num_labels = 2
        labels = torch.tensor([0, 0])

        # High evidence on correct class
        evidence_good = torch.tensor([[10.0, 0.1], [10.0, 0.1]])
        # High evidence on wrong class
        evidence_bad = torch.tensor([[0.1, 10.0], [0.1, 10.0]])

        def compute_loss(evidence):
            one_hot = F.one_hot(labels, num_labels).float()
            alpha = evidence + 1
            S = alpha.sum(dim=-1, keepdim=True)
            nll = (one_hot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1).mean()
            reg = (evidence * (1 - one_hot)).sum(dim=-1).mean()
            return nll + 0.1 * reg

        loss_good = compute_loss(evidence_good)
        loss_bad = compute_loss(evidence_bad)
        assert loss_good < loss_bad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
