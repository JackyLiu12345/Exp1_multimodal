#!/usr/bin/env python3
"""
多模态虚假新闻分类器

整合:
- RoBERTa-base (文本编码器)
- ViT-base (视觉编码器)
- MoE-LoRA (多专家融合)
- 不确定性估计

使用方法:
    from models.classifier import MultimodalFakeNewsClassifier
    
    model = MultimodalFakeNewsClassifier(
        text_model_name="FacebookAI/roberta-base",
        vision_model_name="google/vit-base-patch16-224",
        num_labels=2,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import AutoModel, AutoTokenizer
from .moe_lora import MoELoRA, CrossModalFusion


class TextEncoder(nn.Module):
    """文本编码器 (基于 RoBERTa)"""
    
    def __init__(
        self,
        model_name: str = "FacebookAI/roberta-base",
        freeze: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        
        # 加载预训练模型
        self.roberta = AutoModel.from_pretrained(model_name)
        
        # 冻结参数 (可选)
        if freeze:
            for param in self.roberta.parameters():
                param.requires_grad = False
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 维度
        self.hidden_dim = self.roberta.config.hidden_size
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            (last_hidden_state, pooled_output)
            - last_hidden_state: [batch_size, seq_len, hidden_dim]
            - pooled_output: [batch_size, hidden_dim]
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # 获取 [CLS] token 作为 pooled output
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        
        # Dropout
        last_hidden = self.dropout(outputs.last_hidden_state)
        pooled = self.dropout(pooled_output)
        
        return last_hidden, pooled


class VisionEncoder(nn.Module):
    """视觉编码器 (基于 ViT)"""
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        freeze: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        
        # 加载预训练模型
        self.vit = AutoModel.from_pretrained(model_name)
        
        # 冻结参数 (可选)
        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 维度
        self.hidden_dim = self.vit.config.hidden_size
        self.num_patches = self.vit.config.num_patches + 1  # +1 for [CLS]
    
    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values: [batch_size, 3, 224, 224]
        
        Returns:
            (last_hidden_state, pooled_output)
            - last_hidden_state: [batch_size, num_patches, hidden_dim]
            - pooled_output: [batch_size, hidden_dim]
        """
        outputs = self.vit(
            pixel_values=pixel_values,
            return_dict=True,
        )
        
        # ViT 的第一个 token 是 [CLS]
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        
        # Dropout
        last_hidden = self.dropout(outputs.last_hidden_state)
        pooled = self.dropout(pooled_output)
        
        return last_hidden, pooled


class UncertaintyHead(nn.Module):
    """
    不确定性估计头
    
    使用 Evidential Deep Learning 方法
    输出证据 (evidence) 而非直接概率
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_labels: int,
    ):
        super().__init__()
        
        self.num_labels = num_labels
        
        # 证据网络 (输出非负证据)
        self.evidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels),
            nn.Softplus(),  # 确保非负
        )
    
    def forward(
        self,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch_size, hidden_dim]
        
        Returns:
            {
                "evidence": 证据 [batch, num_labels],
                "uncertainty": 不确定性 [batch],
                "pred_probs": 预测概率 [batch, num_labels],
            }
        """
        # 计算证据
        evidence = self.evidence_net(features)  # [batch, num_labels]
        
        # Dirichlet 分布参数
        alpha = evidence + 1  # [batch, num_labels]
        
        # 预测概率 (Dirichlet 均值)
        pred_probs = alpha / alpha.sum(dim=-1, keepdim=True)  # [batch, num_labels]
        
        # 不确定性 (总证据的倒数)
        total_evidence = evidence.sum(dim=-1)  # [batch]
        uncertainty = self.num_labels / (total_evidence + 1e-8)  # [batch]
        
        return {
            "evidence": evidence,
            "uncertainty": uncertainty,
            "pred_probs": pred_probs,
        }


class MultimodalFakeNewsClassifier(nn.Module):
    """
    多模态虚假新闻分类器
    
    架构:
    文本 → RoBERTa ──┐
                    ├──→ MoE-LoRA 融合 → 分类头 → 预测
    图像 → ViT ─────┘                ↓
                              不确定性估计
    """
    
    def __init__(
        self,
        text_model_name: str = "FacebookAI/roberta-base",
        vision_model_name: str = "google/vit-base-patch16-224",
        num_labels: int = 2,
        # MoE-LoRA 配置
        moe_num_experts: int = 3,
        moe_lora_r: int = 8,
        moe_lora_alpha: float = 16.0,
        moe_top_k: int = 2,
        # 融合配置
        fusion_hidden_dim: int = 768,
        fusion_num_heads: int = 12,
        fusion_num_layers: int = 2,
        # Dropout
        dropout: float = 0.1,
        # 冻结编码器
        freeze_encoders: bool = False,
    ):
        super().__init__()
        
        self.num_labels = num_labels
        
        # 文本编码器
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            freeze=freeze_encoders,
            dropout=dropout,
        )
        
        # 视觉编码器
        self.vision_encoder = VisionEncoder(
            model_name=vision_model_name,
            freeze=freeze_encoders,
            dropout=dropout,
        )
        
        # 跨模态融合 (使用 MoE-LoRA)
        self.cross_modal_fusion = CrossModalFusion(
            text_dim=self.text_encoder.hidden_dim,
            vision_dim=self.vision_encoder.hidden_dim,
            hidden_dim=fusion_hidden_dim,
            num_heads=fusion_num_heads,
            num_layers=fusion_num_layers,
            dropout=dropout,
        )
        
        # MoE-LoRA 层 (在融合后的特征上)
        self.moe_lora = MoELoRA(
            hidden_dim=fusion_hidden_dim,
            num_experts=moe_num_experts,
            lora_r=moe_lora_r,
            lora_alpha=moe_lora_alpha,
            lora_dropout=dropout,
            target_modules=["query", "value"],  # 注意力模块
            top_k=moe_top_k,
            routing_method="softmax",
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_labels),
        )
        
        # 不确定性估计头
        self.uncertainty_head = UncertaintyHead(
            hidden_dim=fusion_hidden_dim,
            num_labels=num_labels,
        )
        
        # 辅助损失权重
        self.contrastive_weight = 0.5
        self.diversity_weight = 0.1
        self.uncertainty_weight = 0.2
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            pixel_values: [batch_size, 3, 224, 224]
            labels: [batch_size] (可选，用于计算损失)
            return_loss: 是否返回损失
        
        Returns:
            {
                "logits": 分类 logits [batch, num_labels],
                "probs": 预测概率 [batch, num_labels],
                "uncertainty": 不确定性 [batch],
                "loss": 总损失 (如果 return_loss=True),
                "contrastive_loss": 对比损失,
                "diversity_loss": 多样性损失,
                "uncertainty_loss": 不确定性损失,
            }
        """
        batch_size = input_ids.shape[0]
        
        # 1. 编码文本
        text_hidden, text_pooled = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )  # text_hidden: [batch, seq_len, hidden]
        
        # 2. 编码图像
        vision_hidden, vision_pooled = self.vision_encoder(
            pixel_values=pixel_values,
        )  # vision_hidden: [batch, num_patches, hidden]
        
        # 3. 跨模态融合
        fused_features = self.cross_modal_fusion(
            text_features=text_hidden,
            vision_features=vision_hidden,
            attention_mask=attention_mask,
        )  # [batch, seq_len, fusion_hidden]
        
        # 4. 使用 [CLS] token 进行分类
        cls_features = fused_features[:, 0, :]  # [batch, fusion_hidden]
        
        # 5. MoE-LoRA 处理
        moe_output, diversity_loss = self.moe_lora(
            fused_features,
            return_aux_loss=True,
        )
        
        # 更新 cls_features
        cls_features = moe_output[:, 0, :]  # [batch, fusion_hidden]
        
        # 6. 分类
        logits = self.classifier(cls_features)  # [batch, num_labels]
        probs = F.softmax(logits, dim=-1)
        
        # 7. 不确定性估计
        uncertainty_output = self.uncertainty_head(cls_features)
        uncertainty = uncertainty_output["uncertainty"]
        
        # 8. 计算损失
        loss_dict = {}
        if return_loss and labels is not None:
            # 主分类损失 (交叉熵)
            classification_loss = F.cross_entropy(logits, labels)
            
            # 对比损失
            contrastive_loss = self._compute_contrastive_loss(
                text_pooled,
                vision_pooled,
                labels,
            )
            
            # 不确定性损失
            uncertainty_loss = self._compute_uncertainty_loss(
                uncertainty_output["evidence"],
                labels,
            )
            
            # 总损失
            total_loss = (
                classification_loss
                + self.contrastive_weight * contrastive_loss
                + self.diversity_weight * diversity_loss
                + self.uncertainty_weight * uncertainty_loss
            )
            
            loss_dict = {
                "loss": total_loss,
                "classification_loss": classification_loss,
                "contrastive_loss": contrastive_loss,
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
    
    def _compute_contrastive_loss(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        跨模态对比损失 (Supervised InfoNCE)
        
        同类样本的图文特征应该相似，异类样本应该相异
        """
        batch_size = text_features.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=text_features.device)
        
        # Ensure both are [batch, hidden] — pool vision if needed
        if vision_features.dim() == 3:
            vision_features = vision_features.mean(dim=1)
        
        # 归一化特征
        text_norm = F.normalize(text_features, dim=-1)
        vision_norm = F.normalize(vision_features, dim=-1)
        
        # 计算相似度矩阵
        similarity = text_norm @ vision_norm.t() / temperature  # [batch, batch]
        
        # 正样本对 (相同标签)
        labels_same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [batch, batch]
        
        # Mask out self-similarity on diagonal
        self_mask = torch.eye(batch_size, device=similarity.device, dtype=torch.bool)
        labels_same.masked_fill_(self_mask, 0.0)
        
        # InfoNCE: log_softmax over each row, weight by positive pairs
        # Use -1e9 instead of -inf to avoid NaN from logsumexp
        log_prob = similarity - torch.logsumexp(similarity.masked_fill(self_mask, -1e9), dim=-1, keepdim=True)
        
        # Average over positive pairs per row
        num_positives = labels_same.sum(dim=-1)
        # Avoid division by zero for rows with no positive pairs
        valid_rows = num_positives > 0
        if not valid_rows.any():
            return torch.tensor(0.0, device=text_features.device)
        
        contrastive_loss = -(log_prob * labels_same).sum(dim=-1) / num_positives.clamp(min=1)
        
        return contrastive_loss[valid_rows].mean()
    
    def _compute_uncertainty_loss(
        self,
        evidence: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evidential Deep Learning loss

        Uses the Dirichlet-based negative log likelihood plus a
        regularizer that minimizes wrong-class evidence.
        """
        # 创建 one-hot 标签
        one_hot = F.one_hot(labels, self.num_labels).float()  # [batch, num_labels]
        
        # Dirichlet 分布参数
        alpha = evidence + 1  # [batch, num_labels]
        S = alpha.sum(dim=-1, keepdim=True)  # Dirichlet strength [batch, 1]
        
        # Type-II maximum likelihood loss (expected cross-entropy under Dirichlet)
        nll_loss = (one_hot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1).mean()
        
        # Regularization: penalize evidence assigned to wrong classes
        wrong_evidence = evidence * (1 - one_hot)
        reg_loss = wrong_evidence.sum(dim=-1).mean()
        
        return nll_loss + 0.1 * reg_loss
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        预测 (包含不确定性)
        
        Returns:
            {
                "predictions": 预测标签 [batch],
                "probs": 预测概率 [batch, num_labels],
                "uncertainty": 不确定性 [batch],
                "low_confidence": 低置信度样本掩码 [batch],
            }
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=None,
                return_loss=False,
            )
        
        predictions = outputs["probs"].argmax(dim=-1)
        max_probs = outputs["probs"].max(dim=-1).values
        
        # 低置信度样本 (高不确定性或低概率)
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
        """获取参数量统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params,
        }


if __name__ == "__main__":
    # 测试代码
    print("测试 MultimodalFakeNewsClassifier...")
    
    # 创建模型 (使用小模型快速测试)
    model = MultimodalFakeNewsClassifier(
        text_model_name="FacebookAI/roberta-base",
        vision_model_name="google/vit-base-patch16-224",
        num_labels=2,
        moe_num_experts=3,
    )
    
    # 测试前向传播
    batch_size = 4
    seq_len = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 2, (batch_size,))
    
    # 前向传播
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
        return_loss=True,
    )
    
    print(f"\n输入:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  labels: {labels.shape}")
    
    print(f"\n输出:")
    print(f"  logits: {outputs['logits'].shape}")
    print(f"  probs: {outputs['probs'].shape}")
    print(f"  uncertainty: {outputs['uncertainty'].shape}")
    print(f"  loss: {outputs['loss'].item():.4f}")
    
    # 参数量统计
    param_stats = model.get_num_params()
    print(f"\n参数量:")
    print(f"  总计：{param_stats['total']:,}")
    print(f"  可训练：{param_stats['trainable']:,}")
    print(f"  冻结：{param_stats['frozen']:,}")
    
    # 测试预测
    pred_outputs = model.predict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
    )
    
    print(f"\n预测:")
    print(f"  predictions: {pred_outputs['predictions'].shape}")
    print(f"  low_confidence: {pred_outputs['low_confidence'].sum().item()} / {batch_size}")
    
    print(f"\n✓ 测试通过!")
