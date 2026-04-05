# Multimodal Fake News Detection with MoE-LoRA

**核心创新**: 多模态 MoE-LoRA 适配器 + 跨模态对比学习  
**基座模型**: RoBERTa-base (文本) + ViT (视觉)  
**硬件**: NVIDIA A40 30GB

---

## 📁 目录结构

```
Exp1_multimodal/
├── data/                      # 数据集
│   ├── raw/                   # 原始数据
│   ├── processed/             # 预处理后数据
│   └── downloads/             # 下载缓存
├── scripts/                   # 脚本
│   ├── collect_data.py        # 数据收集
│   ├── preprocess.py          # 数据预处理
│   ├── train.py               # 训练主脚本
│   ├── evaluate.py            # 评估脚本
│   └── visualize.py           # 可视化
├── models/                    # 模型定义
│   ├── moe_lora.py            # MoE-LoRA 核心模块
│   ├── cross_modal_fusion.py  # 跨模态融合
│   ├── text_encoder.py        # 文本编码器
│   ├── vision_encoder.py      # 视觉编码器
│   └── classifier.py          # 分类头
├── experiments/               # 实验配置
│   ├── config.yaml            # 主配置文件
│   └── ablation/              # 消融实验配置
├── results/                   # 实验结果
│   ├── metrics/               # 指标文件
│   ├── checkpoints/           # 模型检查点
│   └── figures/               # 图表
├── logs/                      # 训练日志
├── docs/                      # 文档
│   ├── paper_draft.md         # 论文草稿
│   └── experiment_records.md  # 实验记录
└── literature/                # 文献
    ├── survey.md              # 文献综述
    └── papers/                # PDF 文件
```

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
python scripts/train.py --config experiments/config.yaml
```

### 评估
```bash
python scripts/evaluate.py --checkpoint results/checkpoints/best_model.pt
```

---

## 📊 实验记录

| 日期 | 实验 | 配置 | 结果 |
|------|------|------|------|
| | | | |

---

## 📝 待办事项

- [ ] 数据收集脚本
- [ ] MoE-LoRA 模块实现
- [ ] 跨模态融合层
- [ ] 训练循环
- [ ] 评估指标
- [ ] 可视化分析
