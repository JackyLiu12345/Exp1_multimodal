# 多模态虚假信息检测 - 项目启动指南

**项目**: Exp1_multimodal  
**目标**: SCI 一区论文 - MM-MoE-LoRA  
**硬件**: NVIDIA A40 30GB  
**日期**: 2026-04-05

---

## 📋 当前进度

### ✅ 已完成 (2026-04-05)

- [x] 项目目录结构创建
- [x] 文献调研完成 (`literature/survey.md`)
- [x] 数据收集脚本 (`scripts/collect_data.py`)
- [x] 数据预处理脚本 (`scripts/preprocess.py`)
- [x] 数据集加载器 (`models/dataset_loader.py`)
- [x] MoE-LoRA 核心模块 (`models/moe_lora.py`)
- [x] 跨模态融合模块 (`models/moe_lora.py`)
- [x] 多模态分类器 (`models/classifier.py`)
- [x] 纯文本分类器 (`models/text_classifier.py`)
- [x] 训练脚本 (`scripts/train.py`)
- [x] 实验配置文件 (`experiments/config.yaml`)
- [x] 单元测试和集成测试 (`tests/`)
- [x] 数据格式文档 (`data/README.md`)

### 🔄 进行中

- [ ] 数据下载 (需手动)

### ⏳ 待开始

- [ ] 评估脚本
- [ ] 可视化分析
- [ ] 消融实验
- [ ] 论文撰写

---

## 🚀 快速开始

### Step 1: 安装依赖

```bash
cd Exp1_multimodal
pip install -r requirements.txt
```

### Step 2: 下载数据集

**Twitter-15 (推荐起始数据集)**:

```bash
# 1. 手动下载
访问：https://www.cs.columbia.edu/~chenhao/resources/
下载：twitter15.tar.gz

# 2. 移动到下载目录
mv ~/Downloads/twitter15.tar.gz data/downloads/

# 3. 运行收集脚本
python scripts/collect_data.py --dataset twitter15 --output_dir data/raw

# 4. 预处理
python scripts/preprocess.py --dataset twitter15 --output_dir data/processed
```

### Step 3: 验证数据

```bash
# 运行数据集加载器测试
python models/dataset_loader.py
```

预期输出:
```
✓ 加载数据集：XXX 条样本
  检测到标签：['real', 'fake']
✓ 创建 DataLoader: batch_size=16, num_workers=4

样本示例:
  ID: xxx
  文本长度：torch.Size([512])
  图像形状：torch.Size([3, 224, 224])
  有图像：True
  标签：1 (fake)
```

### Step 4: 开始训练

```bash
# 完整多模态训练
python scripts/train.py --config experiments/config.yaml

# FakeNewsNet 纯文本快速训练
python scripts/train_fakenewsnet_quick.py

# 消融实验
python scripts/train.py --config experiments/config.yaml --ablation no_moe
```

---

## 📊 数据集统计

### Twitter-15

| 分割 | 样本数 | 真实 | 虚假 | 有图像 |
|------|--------|------|------|--------|
| Train | TBD | TBD | TBD | TBD |
| Dev | TBD | TBD | TBD | TBD |
| Test | TBD | TBD | TBD | TBD |

**标签说明**:
- `real`: 真实新闻
- `fake`: 虚假新闻

---

## 🏗️ 模型架构

```
输入 → 文本编码器 (RoBERTa-base) ─┐
                                   ├→ MoE-LoRA 融合 → 分类头 → 输出
输入 → 视觉编码器 (ViT-base) ─────┘
                                    ↓
                              不确定性估计
```

### 关键组件

| 组件 | 实现文件 | 状态 |
|------|----------|------|
| MoE-LoRA | `models/moe_lora.py` | ✅ 完成 |
| 跨模态融合 | `models/moe_lora.py` | ✅ 完成 |
| 多模态分类器 | `models/classifier.py` | ✅ 完成 |
| 纯文本分类器 | `models/text_classifier.py` | ✅ 完成 |
| 数据加载 | `models/dataset_loader.py` | ✅ 完成 |
| 训练循环 | `scripts/train.py` | ✅ 完成 |
| 单元测试 | `tests/` | ✅ 完成 |
| 评估 | `scripts/evaluate.py` | ⏳ 待实现 |

---

## 📝 下一步行动

### 本周 (第 1 周)

1. **数据准备** (优先级：高)
   ```bash
   # 下载并处理 Twitter-15
   python scripts/collect_data.py --dataset twitter15
   python scripts/preprocess.py --dataset twitter15
   ```

2. **实现训练脚本** (优先级：高)
   - 创建 `scripts/train.py`
   - 实现训练循环
   - 集成 WandB 日志

3. **实现模型主架构** (优先级：高)
   - 创建 `models/classifier.py`
   - 整合 RoBERTa + ViT + MoE-LoRA

### 下周 (第 2 周)

4. **基线实验**
   - 运行标准 LoRA 基线
   - 记录性能指标

5. **MoE-LoRA 实验**
   - 启用 MoE 模块
   - 对比性能提升

---

## 🔧 常见问题

### Q1: 数据下载失败？

Twitter-15/16 需要手动下载，因为：
- 数据集包含受版权保护的内容
- 需要同意使用条款

**解决方案**: 按照 Step 2 手动下载

### Q2: 显存不足？

A40 30GB 应该足够，但如果遇到 OOM:

```yaml
# 修改 experiments/config.yaml
training:
  batch_size: 4          # 减小批次
  gradient_accumulation_steps: 4  # 增加累积 (有效 batch=16)
  gradient_checkpointing: true    # 启用梯度检查点
```

### Q3: 如何监控训练？

项目集成了两种日志:

1. **WandB** (推荐):
   ```bash
   wandb login
   # 训练时自动上传
   ```

2. **TensorBoard**:
   ```bash
   tensorboard --logdir logs/tensorboard
   ```

---

## 📚 关键文件索引

| 文件 | 用途 |
|------|------|
| `README.md` | 项目总览 |
| `experiments/config.yaml` | 多模态实验配置 |
| `experiments/config_fakenewsnet_text.yaml` | 纯文本实验配置 |
| `scripts/collect_data.py` | 数据收集 |
| `scripts/preprocess.py` | 数据预处理 |
| `scripts/train.py` | 多模态训练脚本 |
| `scripts/train_fakenewsnet_quick.py` | 快速文本训练脚本 |
| `models/dataset_loader.py` | 数据加载 |
| `models/moe_lora.py` | MoE-LoRA 核心 + 跨模态融合 |
| `models/classifier.py` | 多模态分类器 |
| `models/text_classifier.py` | 纯文本分类器 |
| `tests/` | 单元测试和集成测试 |
| `data/README.md` | 数据格式文档 |
| `literature/survey.md` | 文献综述 |

---

## 🎯 里程碑

| 日期 | 里程碑 | 交付物 |
|------|--------|--------|
| 2026-04-12 | 数据准备完成 | 处理好的数据集 |
| 2026-04-19 | 训练脚本完成 | 可运行的训练流程 |
| 2026-04-26 | 基线实验完成 | 初步结果 |
| 2026-05-03 | MoE-LoRA 实验 | 对比分析 |
| 2026-05-10 | 消融实验 | 完整分析 |
| 2026-05-17 | 论文初稿 | 可投稿版本 |

---

**最后更新**: 2026-04-05  
**负责人**: JackyLiu12345
