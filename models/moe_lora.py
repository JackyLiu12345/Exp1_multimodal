#!/usr/bin/env python3
"""
MoE-LoRA 核心模块

实现多专家低秩适配器 (Mixture-of-Experts Low-Rank Adaptation)

核心组件:
- LoRALayer: 标准 LoRA 层
- Expert: 单个 LoRA 专家
- MoELoRA: 多专家 MoE 模块
- GatingNetwork: 动态路由网络

参考:
- LoRA: https://arxiv.org/abs/2106.09685
- MoE: https://arxiv.org/abs/1701.06538
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class LoRALayer(nn.Module):
    """
    LoRA 层实现
    
    将权重矩阵分解为: W = W0 + B × A
    其中 W0 冻结，A 和 B 可学习
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        """
        Args:
            in_features: 输入维度
            out_features: 输出维度
            r: LoRA 秩
            alpha: 缩放系数
            dropout: Dropout 率
            bias: 是否使用偏置
        """
        super().__init__()
        
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # 冻结的基础权重 (由外部传入，此处不初始化)
        self.weight = None
        
        # LoRA 参数
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
        self.dropout = nn.Dropout(dropout)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x: torch.Tensor, base_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, in_features]
            base_weight: 基础权重 (可选，如果不提供则只使用 LoRA)
        
        Returns:
            输出张量 [batch_size, seq_len, out_features]
        """
        # LoRA 分支
        lora_output = F.linear(
            self.dropout(x),
            self.lora_B @ self.lora_A,
            None
        ) * self.scaling
        
        # 基础权重分支
        if base_weight is not None:
            base_output = F.linear(x, base_weight, self.bias)
            return base_output + lora_output
        else:
            return lora_output + (self.bias if self.bias is not None else 0)


class Expert(nn.Module):
    """
    单个 LoRA 专家
    
    由多个 LoRA 层组成，处理特定子任务
    """
    
    def __init__(
        self,
        expert_id: int,
        hidden_dim: int,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        target_modules: List[str] = ["query", "value"],
    ):
        super().__init__()
        
        self.expert_id = expert_id
        self.hidden_dim = hidden_dim
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        
        # 为每个目标模块创建 LoRA 层
        self.lora_layers = nn.ModuleDict()
        
        for module_name in target_modules:
            self.lora_layers[f"{module_name}_lora"] = LoRALayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
        
        # 专家特定参数 (可选)
        self.expert_specific = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        base_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_dim]
            base_weights: 基础权重的字典 (可选)
        
        Returns:
            专家输出
        """
        output = hidden_states
        
        # 应用 LoRA 层
        for module_name, lora_layer in self.lora_layers.items():
            key = module_name.replace("_lora", "")
            base_weight = base_weights.get(key, None) if base_weights else None
            output = lora_layer(output, base_weight)
        
        # 专家特定处理
        output = self.expert_specific(output)
        
        return output


class GatingNetwork(nn.Module):
    """
    门控网络 (路由网络)
    
    决定每个 token 应该由哪些专家处理
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        routing_method: str = "softmax",
    ):
        """
        Args:
            hidden_dim: 输入维度
            num_experts: 专家数量
            top_k: 选择的专家数量
            routing_method: 路由方法 (softmax / top_k / gshard)
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.routing_method = routing_method
        
        # 门控网络
        self.gate = nn.Linear(hidden_dim, num_experts)
        
        # 辅助损失 (用于负载均衡)
        self.aux_loss = 0.0
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入 [batch_size, seq_len, hidden_dim]
        
        Returns:
            (expert_weights, expert_indices)
            - expert_weights: 专家权重 [batch_size, seq_len, num_experts]
            - expert_indices: 选择的专家索引 [batch_size, seq_len, top_k]
        """
        # 计算门控分数
        gate_logits = self.gate(x)  # [batch, seq, num_experts]
        
        if self.routing_method == "softmax":
            # 软路由 (所有专家加权)
            expert_weights = F.softmax(gate_logits, dim=-1)
            expert_indices = torch.arange(self.num_experts, device=x.device).expand(
                x.shape[0], x.shape[1], self.num_experts
            )
        
        elif self.routing_method == "top_k":
            # Top-K 路由
            expert_weights, expert_indices = torch.topk(
                F.softmax(gate_logits, dim=-1),
                k=self.top_k,
                dim=-1,
            )
            
            # 归一化权重
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        elif self.routing_method == "gshard":
            # GShard 路由 (带噪声)
            noise = torch.randn_like(gate_logits) * 0.1
            gate_logits = gate_logits + noise
            
            expert_weights, expert_indices = torch.topk(
                F.softmax(gate_logits, dim=-1),
                k=self.top_k,
                dim=-1,
            )
        
        else:
            raise ValueError(f"未知的路由方法：{self.routing_method}")
        
        # 计算辅助损失 (负载均衡)
        self._compute_aux_loss(expert_weights)
        
        return expert_weights, expert_indices
    
    def _compute_aux_loss(self, expert_weights: torch.Tensor):
        """
        计算辅助损失，鼓励专家使用均衡
        
        参考: Switch Transformer (https://arxiv.org/abs/2101.03961)
        """
        # 专家使用率
        expert_usage = expert_weights.mean(dim=[0, 1])  # [num_experts]
        self.expert_usage = expert_usage.detach()
        
        # 鼓励均匀分布
        uniform_usage = torch.ones_like(expert_usage) / self.num_experts
        self.aux_loss = F.kl_div(
            uniform_usage.log(),
            expert_usage,
            reduction="batchmean",
        )


class MoELoRA(nn.Module):
    """
    多专家 LoRA 模块
    
    结合多个 LoRA 专家，通过门控网络动态路由
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        target_modules: List[str] = ["query", "value"],
        top_k: int = 2,
        routing_method: str = "softmax",
    ):
        """
        Args:
            hidden_dim: 隐藏层维度
            num_experts: 专家数量
            lora_r: LoRA 秩
            lora_alpha: LoRA 缩放系数
            lora_dropout: Dropout 率
            target_modules: 目标模块列表
            top_k: 选择的专家数量
            routing_method: 路由方法
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 创建专家
        self.experts = nn.ModuleList([
            Expert(
                expert_id=i,
                hidden_dim=hidden_dim,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
            )
            for i in range(num_experts)
        ])
        
        # 门控网络
        self.gate = GatingNetwork(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            routing_method=routing_method,
        )
        
        # 输出投影 (可选)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        base_weights: Optional[Dict[str, torch.Tensor]] = None,
        return_aux_loss: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, hidden_dim]
            base_weights: 基础权重字典 (可选)
            return_aux_loss: 是否返回辅助损失
        
        Returns:
            输出 [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 获取路由权重
        expert_weights, expert_indices = self.gate(hidden_states)
        
        # 初始化输出
        output = torch.zeros_like(hidden_states)
        
        # 对每个专家进行处理
        if self.gate.routing_method == "softmax":
            # Soft routing: all experts weighted, no masking needed
            for expert_id, expert in enumerate(self.experts):
                expert_output = expert(hidden_states, base_weights)
                weight = expert_weights[:, :, expert_id:expert_id+1]
                output = output + expert_output * weight
        else:
            # Top-K routing: only process tokens assigned to each expert
            for expert_id, expert in enumerate(self.experts):
                # Find tokens assigned to this expert across all top-k slots
                mask = (expert_indices == expert_id).any(dim=-1)  # [batch, seq_len]
                if not mask.any():
                    continue
                
                # Process full batch through expert (unavoidable without scatter)
                expert_output = expert(hidden_states, base_weights)
                
                # Get the weight for this expert from the top-k weights
                # expert_indices: [batch, seq, top_k], expert_weights: [batch, seq, top_k]
                expert_mask = (expert_indices == expert_id).float()  # [batch, seq, top_k]
                weight = (expert_weights * expert_mask).sum(dim=-1, keepdim=True)  # [batch, seq, 1]
                
                output = output + expert_output * weight
        
        # 输出投影
        output = self.output_proj(output)
        
        # 残差连接
        output = output + hidden_states
        
        if return_aux_loss:
            return output, self.gate.aux_loss
        else:
            return output
    
    def get_expert_usage(self) -> torch.Tensor:
        """获取专家使用率统计"""
        return self.gate.expert_usage if hasattr(self.gate, 'expert_usage') else None


class CrossModalFusion(nn.Module):
    """
    跨模态融合模块
    
    使用交叉注意力机制融合文本和视觉特征
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        vision_dim: int = 768,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        
        # 投影层
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        
        # 交叉注意力层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        
        self.fusion_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            text_features: 文本特征 [batch_size, seq_len, text_dim]
            vision_features: 视觉特征 [batch_size, num_patches, vision_dim]
            attention_mask: 文本注意力掩码
        
        Returns:
            融合特征 [batch_size, seq_len, hidden_dim]
        """
        # 投影到统一维度
        text_proj = self.text_proj(text_features)
        vision_proj = self.vision_proj(vision_features)
        
        # 拼接序列
        combined = torch.cat([text_proj, vision_proj], dim=1)  # [batch, seq+patches, hidden]
        
        # 构建注意力掩码
        if attention_mask is not None:
            # TransformerEncoder expects bool padding mask (True = ignore)
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()
            
            # 文本部分的掩码
            text_mask_len = text_proj.shape[1]
            vision_mask_len = vision_proj.shape[1]
            
            # 扩展掩码 (文本部分使用原掩码，视觉部分全为 True — not padded)
            vision_mask = torch.ones(
                attention_mask.shape[0],
                vision_mask_len,
                device=attention_mask.device,
                dtype=torch.bool,
            )
            combined_mask = torch.cat([attention_mask, vision_mask], dim=1)
            # Invert: PyTorch src_key_padding_mask uses True to indicate positions to ignore
            combined_mask = ~combined_mask
        else:
            combined_mask = None
        
        # 交叉注意力融合
        fused = self.fusion_encoder(combined, src_key_padding_mask=combined_mask)
        
        # 提取文本部分
        text_len = text_proj.shape[1]
        fused_text = fused[:, :text_len, :]
        
        # 层归一化 + 残差
        output = self.norm(fused_text + text_proj)
        
        return output


if __name__ == "__main__":
    # 测试代码
    print("测试 MoE-LoRA 模块...")
    
    # 创建 MoE-LoRA
    moe_lora = MoELoRA(
        hidden_dim=768,
        num_experts=3,
        lora_r=8,
        lora_alpha=16,
        top_k=2,
    )
    
    # 测试前向传播
    batch_size = 4
    seq_len = 128
    
    x = torch.randn(batch_size, seq_len, 768)
    
    output, aux_loss = moe_lora(x, return_aux_loss=True)
    
    print(f"输入形状：{x.shape}")
    print(f"输出形状：{output.shape}")
    print(f"辅助损失：{aux_loss.item():.4f}")
    
    # 测试跨模态融合
    print("\n测试跨模态融合...")
    
    fusion = CrossModalFusion(
        text_dim=768,
        vision_dim=768,
        hidden_dim=768,
        num_heads=12,
        num_layers=2,
    )
    
    text_feat = torch.randn(batch_size, seq_len, 768)
    vision_feat = torch.randn(batch_size, 197, 768)  # ViT patch 数量
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    fused = fusion(text_feat, vision_feat, attention_mask)
    
    print(f"文本特征：{text_feat.shape}")
    print(f"视觉特征：{vision_feat.shape}")
    print(f"融合特征：{fused.shape}")
    
    print("\n✓ 测试通过!")
