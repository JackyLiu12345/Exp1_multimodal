# 🚀 PHEME 数据集下载 - 快速指南

**更新时间**: 2026-04-05 17:25  
**状态**: ⏳ 下载中

---

## ⚠️ 重要提示

**使用 `python3` 而不是 `python`**

```bash
# ❌ 错误
python scripts/xxx.py

# ✅ 正确
python3 scripts/xxx.py
```

---

## 🎯 推荐方案（按优先级）

### 方案 1️⃣: HuggingFace 镜像 (推荐 ⭐⭐⭐⭐⭐)

**使用阿里云镜像，速度快**

```bash
cd /home/admin/openclaw/workspace/Exp1_multimodal

python3 scripts/download_pHEME_hf_mirror.py --output_dir data/raw/pheme
```

**预计时间**: 5-10 分钟  
**优点**: 自动处理，无需手动操作

---

### 方案 2️⃣: Kaggle 手动下载 (备选 ⭐⭐⭐⭐)

如果 HuggingFace 下载失败，使用 Kaggle：

```bash
# 1. 访问 Kaggle
https://www.kaggle.com/datasets/arkaitz/zubiaga-pheme-dataset

# 2. 下载数据集 (需要 Kaggle 账号，免费)

# 3. 移动到项目目录
mv ~/Downloads/pheme-dataset.zip /home/admin/openclaw/workspace/Exp1_multimodal/data/downloads/

# 4. 解压
cd /home/admin/openclaw/workspace/Exp1_multimodal/data/downloads
unzip pheme-dataset.zip

# 5. 运行处理脚本
cd /home/admin/openclaw/workspace/Exp1_multimodal
python3 scripts/collect_data.py --dataset pheme
```

---

### 方案 3️⃣: 使用已有的 FakeNewsNet

**注意**: FakeNewsNet 需要 Twitter API 才能获取完整数据

当前状态:
```
/home/admin/FakeNewsNet/dataset/
├── gossipcop_fake.csv   ✅ (有 CSV)
├── gossipcop_real.csv   ✅ (有 CSV)
├── politifact_fake.csv  ✅ (有 CSV)
└── politifact_real.csv  ✅ (有 CSV)
```

**问题**: 只有 CSV 元数据，缺少推文内容和图像

**解决方案**:
1. 申请 Twitter API (免费，1-2 天)
2. 配置密钥
3. 运行收集脚本

---

## 🔧 网络问题解决方案

### 如果 HuggingFace 下载超时：

**方法 1: 使用镜像**
```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 scripts/download_pHEME_hf_mirror.py --output_dir data/raw/pheme
```

**方法 2: 设置代理**
```bash
export HTTP_PROXY=http://proxy-server:port
export HTTPS_PROXY=http://proxy-server:port
python3 scripts/download_pHEME_hf.py --output_dir data/raw/pheme
```

**方法 3: 使用 Kaggle**
- 访问：https://www.kaggle.com/datasets/arkaitz/zubiaga-pheme-dataset
- 下载速度通常更快

---

## 📊 验证数据集

下载完成后，检查数据：

```bash
# 查看文件
ls -lh data/raw/pheme/

# 查看统计
cat data/raw/pheme/metadata.json

# 应该看到:
# - train.json (约 7,000+ 条)
# - test.json (约 2,000+ 条)
# - metadata.json
```

**正确的内容**:
```json
{
  "dataset": "PHEME",
  "events": ["charliehebdo", "sydneysiege", "ferguson", ...],
  "label_distribution": {"real": 4xxx, "fake": 5xxx}
}
```

**错误的内容** (生物医学数据集):
- ❌ "meningococcal"
- ❌ "vaccine"
- ❌ "Factor H Binding Protein"

---

## 🎯 当前推荐

**立即执行**（使用镜像）:

```bash
cd /home/admin/openclaw/workspace/Exp1_multimodal

python3 scripts/download_pHEME_hf_mirror.py --output_dir data/raw/pheme
```

这个脚本使用阿里云镜像，应该能解决超时问题。

---

## 📞 遇到问题？

请告诉我：

1. **错误信息是什么？**
   - 截图或复制错误日志

2. **网络状况如何？**
   - 能否访问 huggingface.co？
   - 能否访问 kaggle.com？

3. **下载进度？**
   - 卡在哪一步？

我会根据你的反馈提供解决方案！🚀

---

**状态**: ⏳ 等待下载完成  
**推荐命令**: `python3 scripts/download_pHEME_hf_mirror.py`  
**更新时间**: 2026-04-05 17:30
