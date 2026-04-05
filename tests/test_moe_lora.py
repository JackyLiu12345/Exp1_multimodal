#!/usr/bin/env python3
"""
Unit tests for LoRA layer, MoE routing, and CrossModalFusion.
"""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.moe_lora import LoRALayer, Expert, GatingNetwork, MoELoRA, CrossModalFusion


# ---------- LoRA Layer ----------

class TestLoRALayer:
    """Tests for the basic LoRA layer."""

    def test_output_shape(self):
        layer = LoRALayer(in_features=64, out_features=64, r=4, alpha=8.0)
        x = torch.randn(2, 10, 64)
        out = layer(x)
        assert out.shape == (2, 10, 64)

    def test_output_shape_with_base_weight(self):
        layer = LoRALayer(in_features=64, out_features=64, r=4, alpha=8.0)
        base_weight = torch.randn(64, 64)
        x = torch.randn(2, 10, 64)
        out = layer(x, base_weight=base_weight)
        assert out.shape == (2, 10, 64)

    def test_zero_init_produces_base_only(self):
        """With B initialized to zeros, LoRA output should be zero initially."""
        layer = LoRALayer(in_features=32, out_features=32, r=4, alpha=8.0, dropout=0.0)
        x = torch.randn(1, 5, 32)
        lora_only = layer(x)  # no base weight
        # B is zeros, so B@A is zeros, so output should be ~0
        assert lora_only.abs().max().item() < 1e-6

    def test_backward_pass(self):
        layer = LoRALayer(in_features=64, out_features=64, r=4, alpha=8.0)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.lora_A.grad is not None
        assert layer.lora_B.grad is not None

    def test_scaling_factor(self):
        layer = LoRALayer(in_features=32, out_features=32, r=4, alpha=16.0)
        assert layer.scaling == 16.0 / 4


# ---------- Expert ----------

class TestExpert:
    """Tests for individual LoRA expert."""

    def test_output_shape(self):
        expert = Expert(expert_id=0, hidden_dim=64, lora_r=4)
        x = torch.randn(2, 10, 64)
        out = expert(x)
        assert out.shape == (2, 10, 64)

    def test_different_experts_differ(self):
        """Two experts should produce different outputs (different random init)."""
        e1 = Expert(expert_id=0, hidden_dim=64, lora_r=4)
        e2 = Expert(expert_id=1, hidden_dim=64, lora_r=4)
        x = torch.randn(1, 5, 64)
        out1 = e1(x)
        out2 = e2(x)
        assert not torch.allclose(out1, out2, atol=1e-5)


# ---------- Gating Network ----------

class TestGatingNetwork:
    """Tests for the gating/routing network."""

    def test_softmax_routing_shapes(self):
        gate = GatingNetwork(hidden_dim=64, num_experts=3, top_k=2, routing_method="softmax")
        x = torch.randn(2, 10, 64)
        weights, indices = gate(x)
        assert weights.shape == (2, 10, 3)
        assert indices.shape == (2, 10, 3)

    def test_topk_routing_shapes(self):
        gate = GatingNetwork(hidden_dim=64, num_experts=4, top_k=2, routing_method="top_k")
        x = torch.randn(2, 10, 64)
        weights, indices = gate(x)
        assert weights.shape == (2, 10, 2)
        assert indices.shape == (2, 10, 2)

    def test_softmax_weights_sum_to_one(self):
        gate = GatingNetwork(hidden_dim=64, num_experts=3, routing_method="softmax")
        x = torch.randn(2, 10, 64)
        weights, _ = gate(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_topk_weights_sum_to_one(self):
        gate = GatingNetwork(hidden_dim=64, num_experts=4, top_k=2, routing_method="top_k")
        x = torch.randn(2, 10, 64)
        weights, _ = gate(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_aux_loss_computed(self):
        gate = GatingNetwork(hidden_dim=64, num_experts=3, routing_method="softmax")
        x = torch.randn(2, 10, 64)
        gate(x)
        assert isinstance(gate.aux_loss, torch.Tensor)
        assert gate.aux_loss.ndim == 0  # scalar

    def test_expert_usage_stored(self):
        gate = GatingNetwork(hidden_dim=64, num_experts=3, routing_method="softmax")
        x = torch.randn(2, 10, 64)
        gate(x)
        assert hasattr(gate, 'expert_usage')
        assert gate.expert_usage.shape == (3,)


# ---------- MoELoRA ----------

class TestMoELoRA:
    """Tests for the full MoE-LoRA module."""

    def test_output_shape(self):
        moe = MoELoRA(hidden_dim=64, num_experts=3, lora_r=4, top_k=2)
        x = torch.randn(2, 10, 64)
        out = moe(x)
        assert out.shape == (2, 10, 64)

    def test_output_with_aux_loss(self):
        moe = MoELoRA(hidden_dim=64, num_experts=3, lora_r=4)
        x = torch.randn(2, 10, 64)
        out, aux = moe(x, return_aux_loss=True)
        assert out.shape == (2, 10, 64)
        assert isinstance(aux, torch.Tensor)
        assert aux.ndim == 0

    def test_residual_connection(self):
        """Output should differ from input by more than just noise (residual + expert)."""
        moe = MoELoRA(hidden_dim=64, num_experts=3, lora_r=4)
        x = torch.randn(2, 10, 64)
        out = moe(x)
        # Due to residual connection, output should not be zero
        assert out.abs().sum() > 0

    def test_backward_pass(self):
        moe = MoELoRA(hidden_dim=64, num_experts=3, lora_r=4)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out, aux = moe(x, return_aux_loss=True)
        loss = out.sum() + aux
        loss.backward()
        assert x.grad is not None

    def test_topk_routing(self):
        moe = MoELoRA(hidden_dim=64, num_experts=4, lora_r=4, top_k=2, routing_method="top_k")
        x = torch.randn(2, 10, 64)
        out, aux = moe(x, return_aux_loss=True)
        assert out.shape == (2, 10, 64)

    def test_expert_usage(self):
        moe = MoELoRA(hidden_dim=64, num_experts=3, lora_r=4)
        x = torch.randn(2, 10, 64)
        moe(x, return_aux_loss=True)
        usage = moe.get_expert_usage()
        assert usage is not None
        assert usage.shape == (3,)


# ---------- CrossModalFusion ----------

class TestCrossModalFusion:
    """Tests for the cross-modal fusion module."""

    def test_output_shape(self):
        fusion = CrossModalFusion(text_dim=64, vision_dim=64, hidden_dim=64, num_heads=4, num_layers=1)
        text = torch.randn(2, 10, 64)
        vision = torch.randn(2, 5, 64)
        out = fusion(text, vision)
        assert out.shape == (2, 10, 64)

    def test_with_attention_mask(self):
        fusion = CrossModalFusion(text_dim=64, vision_dim=64, hidden_dim=64, num_heads=4, num_layers=1)
        text = torch.randn(2, 10, 64)
        vision = torch.randn(2, 5, 64)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[0, 8:] = False  # mask out last 2 tokens for first sample
        out = fusion(text, vision, attention_mask=mask)
        assert out.shape == (2, 10, 64)

    def test_different_dims(self):
        fusion = CrossModalFusion(text_dim=32, vision_dim=64, hidden_dim=48, num_heads=4, num_layers=1)
        text = torch.randn(2, 10, 32)
        vision = torch.randn(2, 5, 64)
        out = fusion(text, vision)
        assert out.shape == (2, 10, 48)

    def test_backward_pass(self):
        fusion = CrossModalFusion(text_dim=64, vision_dim=64, hidden_dim=64, num_heads=4, num_layers=1)
        text = torch.randn(2, 10, 64, requires_grad=True)
        vision = torch.randn(2, 5, 64, requires_grad=True)
        out = fusion(text, vision)
        loss = out.sum()
        loss.backward()
        assert text.grad is not None
        assert vision.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
