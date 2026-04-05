#!/usr/bin/env python3
"""
纯文本虚假新闻分类器 (用于 FakeNewsNet CSV 数据)

无图像输入，仅使用文本

使用方法:
    from models.text_classifier import TextOnlyFakeNewsClassifier
    
    model = TextOnlyFakeNewsClassifier(
        model_name="FacebookAI/roberta-base",
        num_labels=2,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from transformers import AutoModel, AutoTokenizer


class TextOnlyFakeNewsClassifier(nn.Module):
    """
    纯文本虚假新闻分类器
    
    架构:
    文本 → RoBERTa → MoE-LoRA → 分类头 → 预测
                                      ↓
                               不确定性估计
    """
    
    def __init__(
        self,
        model_name: str = "FacebookAI/roberta-base",
        num_labels: int = 2,
        # MoE-LoRA 配置
        moe_num_experts: int = 3,
        moe_lora_r: int = 8,
        moe_lora_alpha: float = 16.0,
        moe_top_k: int = 2,
        # 隐藏层维度
        hidden_dim: int = 768,
        # Dropout
        dropout: float = 0.1,
        # 冻结编码器
        freeze_encoder: bool = False,
        # Label smoothing
        label_smoothing: float = 0.0,
        # Gradient checkpointing
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.label_smoothing = label_smoothing
        
        # 文本编码器
        self.text_encoder = AutoModel.from_pretrained(model_name)
        
        # 冻结参数 (可选)
        if freeze_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Enable gradient checkpointing
        if gradient_checkpointing and hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
            self.text_encoder.gradient_checkpointing_enable()
        
        encoder_hidden = self.text_encoder.config.hidden_size
        
        # MoE-LoRA 层
        from .moe_lora import MoELoRA
        
        self.moe_lora = MoELoRA(
            hidden_dim=encoder_hidden,
            num_experts=moe_num_experts,
            lora_r=moe_lora_r,
            lora_alpha=moe_lora_alpha,
            lora_dropout=dropout,
            target_modules=["query", "value"],
            top_k=moe_top_k,
            routing_method="softmax",
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(encoder_hidden, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )
        
        # 不确定性估计头
        self.uncertainty_head = nn.Sequential(
            nn.Linear(encoder_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels),
            nn.Softplus(),
        )
        
        # 损失权重
        self.diversity_weight = 0.1
        self.uncertainty_weight = 0.2
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size] (可选)
            return_loss: 是否返回损失
        
        Returns:
            {
                "logits": [batch, num_labels],
                "probs": [batch, num_labels],
                "uncertainty": [batch],
                "loss": (如果 return_loss=True),
            }
        """
        # 1. 编码文本
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # 使用 [CLS] token
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        
        # 2. MoE-LoRA 处理
        # 需要扩展为序列格式
        batch_size = cls_output.shape[0]
        seq_output = cls_output.unsqueeze(1)  # [batch, 1, hidden]
        
        moe_output, diversity_loss = self.moe_lora(
            seq_output,
            return_aux_loss=True,
        )
        
        cls_features = moe_output[:, 0, :]  # [batch, hidden]
        
        # 3. 分类
        logits = self.classifier(cls_features)
        probs = F.softmax(logits, dim=-1)
        
        # 4. 不确定性估计
        evidence = self.uncertainty_head(cls_features)
        uncertainty = self.num_labels / (evidence.sum(dim=-1) + 1e-8)
        
        # 5. 计算损失
        loss_dict = {}
        if return_loss and labels is not None:
            # 主分类损失
            classification_loss = F.cross_entropy(
                logits, labels, label_smoothing=self.label_smoothing
            )
            
            # 不确定性损失
            uncertainty_loss = self._compute_uncertainty_loss(evidence, labels)
            
            # 总损失
            total_loss = (
                classification_loss
                + self.diversity_weight * diversity_loss
                + self.uncertainty_weight * uncertainty_loss
            )
            
            loss_dict = {
                "loss": total_loss,
                "classification_loss": classification_loss,
                "diversity_loss": diversity_loss,
                "uncertainty_loss": uncertainty_loss,
            }
        
        return {
            "logits": logits,
            "probs": probs,
            "uncertainty": uncertainty,
            "cls_features": cls_features,
            **loss_dict,
        }
    
    def _compute_uncertainty_loss(
        self,
        evidence: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Evidential Deep Learning loss"""
        one_hot = F.one_hot(labels, self.num_labels).float()
        alpha = evidence + 1
        S = alpha.sum(dim=-1, keepdim=True)
        
        # Type-II maximum likelihood loss
        nll_loss = (one_hot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1).mean()
        
        # Regularization: penalize wrong-class evidence
        wrong_evidence = evidence * (1 - one_hot)
        reg_loss = wrong_evidence.sum(dim=-1).mean()
        
        return nll_loss + 0.1 * reg_loss
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """预测"""
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                return_loss=False,
            )
        
        predictions = outputs["probs"].argmax(dim=-1)
        max_probs = outputs["probs"].max(dim=-1).values
        low_confidence = (
            (outputs["uncertainty"] > threshold) |
            (max_probs < (1 - threshold))
        )
        
        return {
            "predictions": predictions,
            "probs": outputs["probs"],
            "uncertainty": outputs["uncertainty"],
            "low_confidence": low_confidence,
        }
    
    def get_num_params(self) -> Dict[str, int]:
        """参数量统计"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }


if __name__ == "__main__":
    # 测试
    print("测试 TextOnlyFakeNewsClassifier...")
    
    model = TextOnlyFakeNewsClassifier(
        model_name="FacebookAI/roberta-base",
        num_labels=2,
        moe_num_experts=3,
    )
    
    batch_size = 4
    seq_len = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    labels = torch.randint(0, 2, (batch_size,))
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_loss=True,
    )
    
    print(f"\n输入：{input_ids.shape}")
    print(f"输出：logits={outputs['logits'].shape}, loss={outputs['loss'].item():.4f}")
    
    param_stats = model.get_num_params()
    print(f"参数：总计={param_stats['total']:,}, 可训练={param_stats['trainable']:,}")
    
    print("\n✓ 测试通过!")
