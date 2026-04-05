# Multimodal Fake News Detection with MoE-LoRA

**核心创新**: 多模态 MoE-LoRA 适配器 + 跨模态对比学习  
**基座模型**: RoBERTa-base (文本) + ViT (视觉)  
**硬件**: NVIDIA A40 30GB

---

## 📁 目录结构

```
Exp1_multimodal/
├── data/                      # 数据集 (see data/README.md for format)
│   ├── raw/                   # 原始数据
│   ├── processed/             # 预处理后数据
│   └── downloads/             # 下载缓存
├── scripts/                   # 脚本
│   ├── collect_data.py        # 数据收集
│   ├── preprocess.py          # 数据预处理
│   ├── train.py               # 训练主脚本
│   └── train_fakenewsnet_quick.py  # FakeNewsNet快速训练
├── models/                    # 模型定义
│   ├── moe_lora.py            # MoE-LoRA 核心 + 跨模态融合
│   ├── classifier.py          # 多模态分类器 (RoBERTa+ViT+MoE-LoRA)
│   ├── text_classifier.py     # 纯文本分类器
│   └── dataset_loader.py      # 数据加载器
├── experiments/               # 实验配置
│   ├── config.yaml            # 多模态训练配置
│   └── config_fakenewsnet_text.yaml  # 纯文本训练配置
├── tests/                     # 测试
│   ├── test_moe_lora.py       # LoRA/MoE/融合单元测试
│   ├── test_classifier.py     # 分类器/损失函数测试
│   ├── test_dataset.py        # 数据加载测试
│   └── test_integration.py    # 端到端集成测试
├── results/                   # 实验结果
├── logs/                      # 训练日志
├── docs/                      # 文档
└── literature/                # 文献
    └── survey.md              # 文献综述
```

---

## 🏗️ 模型架构

```
                    ┌─────────────────────────┐
    文本 ──→ RoBERTa ──┐                      │
                       ├→ CrossModal Fusion ──→ MoE-LoRA ──→ 分类头 ──→ 预测
    图像 ──→ ViT ──────┘                      │              ↓
                    └─────────────────────────┘       不确定性估计
```

**Key components:**
- **MoE-LoRA**: Multiple LoRA expert adapters with learned gating/routing
- **CrossModalFusion**: Transformer encoder layers that align text+vision
- **Supervised InfoNCE**: Contrastive loss for cross-modal alignment
- **Evidential DL**: Dirichlet-based uncertainty estimation

---

## 🚀 快速开始

### 环境安装
```bash
pip install -r requirements.txt
```

### 数据准备
```bash
python scripts/collect_data.py --dataset twitter15
python scripts/preprocess.py --dataset twitter15
```

### 训练
```bash
# 多模态训练
python scripts/train.py --config experiments/config.yaml

# FakeNewsNet 纯文本快速训练
python scripts/train_fakenewsnet_quick.py
```

### 运行测试
```bash
python -m pytest tests/ -v
```

---

## ✅ 实现状态

| 组件 | 状态 | 文件 |
|------|------|------|
| MoE-LoRA 核心 | ✅ 完成 | `models/moe_lora.py` |
| 跨模态融合 | ✅ 完成 | `models/moe_lora.py` |
| 多模态分类器 | ✅ 完成 | `models/classifier.py` |
| 纯文本分类器 | ✅ 完成 | `models/text_classifier.py` |
| 数据加载器 | ✅ 完成 | `models/dataset_loader.py` |
| 训练脚本 | ✅ 完成 | `scripts/train.py` |
| 单元测试 | ✅ 完成 | `tests/` |
| 对比学习损失 | ✅ 完成 | Supervised InfoNCE |
| 不确定性估计 | ✅ 完成 | Evidential Deep Learning |
| 梯度累积/裁剪 | ✅ 完成 | From config |
| Label smoothing | ✅ 完成 | From config |
| Gradient checkpointing | ✅ 完成 | From config |
| 评估脚本 | ⏳ 待实现 | |
| 消融实验 | ⏳ 待实现 | |
| 可视化分析 | ⏳ 待实现 | |

---

## 📊 实验配置

Config supports ablation variants:
- **full_model**: MoE-LoRA + 对比学习 + 不确定性
- **no_moe**: 单 LoRA (num_experts=1)
- **no_contrastive**: 无对比学习
- **no_uncertainty**: 无不确定性估计
- **standard_lora**: 标准 LoRA (无 MoE, 无对比, 无多样性)

See `experiments/config.yaml` for full configuration options.

---

## 📝 数据格式

See [`data/README.md`](data/README.md) for the expected JSON format and examples.
