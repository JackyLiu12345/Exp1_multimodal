# FakeNewsNet CSV 实验配置 (文本-only)

**注意**: 此配置用于 FakeNewsNet CSV 数据（只有文本，没有图像）

---

## 📊 数据集配置

```yaml
dataset:
  name: fakenewsnet_csv
  data_dir: data/processed/fakenewsnet
  max_text_length: 512
  image_size: 224  # 保留但不会使用
```

---

## 🤖 模型配置调整

由于 CSV 数据没有图像，我们需要调整模型：

### 方案 A: 使用纯文本模型 (推荐)

```yaml
model:
  text_encoder:
    name: FacebookAI/roberta-base
    freeze: false
  
  # 视觉编码器禁用
  vision_encoder: null
  
  # 使用纯文本分类器
  classifier:
    type: text_only
    hidden_dim: 768
    num_labels: 2
```

### 方案 B: 使用零图像占位

保留多模态架构，但图像用零填充：
```yaml
model:
  text_encoder:
    name: FacebookAI/roberta-base
  
  vision_encoder:
    name: google/vit-base-patch16-224
  
  # 图像用零填充
  use_zero_image: true
```

---

## 🎯 推荐配置

我为你创建了专门的文本分类配置：

**文件**: `experiments/config_fakenewsnet_text.yaml`

**关键差异**:
- ✅ 只使用 RoBERTa (无 ViT)
- ✅ 更小的模型，更快训练
- ✅ 适合快速验证代码
- ✅ 之后可以轻松切换到多模态

---

## 📈 预期性能

**FakeNewsNet 基线** (文本-only):

| 方法 | Accuracy | F1 |
|------|----------|-----|
| RoBERTa-base | ~0.88 | ~0.88 |
| BERT-base | ~0.87 | ~0.87 |
| LSTM | ~0.82 | ~0.81 |

**我们的目标**: F1 ≥ 0.88

---

## ⏱️ 训练时间估算

**A40 30GB**:
- Batch size: 16
- Epochs: 5
- 预计时间：30-60 分钟

---

## 🚀 快速开始

```bash
# 1. 处理 CSV 数据
python3 scripts/process_fakenewsnet_csv.py --output_dir data/raw/fakenewsnet

# 2. 预处理
python3 scripts/preprocess.py --dataset fakenewsnet --output_dir data/processed

# 3. 训练 (文本-only)
python3 scripts/train.py --config experiments/config_fakenewsnet_text.yaml
```
