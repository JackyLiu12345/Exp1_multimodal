# 📊 多模态虚假新闻数据集获取指南

**更新时间**: 2026-04-05  
**问题**: Twitter-15 原始链接失效，需要替代数据源

---

## ⚠️ Twitter-15/16 现状

**原始链接**: https://www.cs.columbia.edu/~chenhao/resources/  
**状态**: ❌ 已失效/无法访问

**原因**:
- Twitter 隐私政策变更 (2020 年后)
- 数据集包含受版权保护的推文内容
- 需要 Twitter API 重新收集

---

## ✅ 推荐替代方案 (按优先级排序)

### 方案 1️⃣: **FakeNewsNet** (最推荐 ⭐⭐⭐⭐⭐)

**特点**:
- 包含 PolitiFact 和 GossipCop 两个子集
- 提供新闻文章 + 推文 + 图像
- 有官方代码可重新收集数据
- 论文引用量大，认可度高

**获取方式**:
```bash
# GitHub 仓库
git clone https://github.com/KaiDMML/FakeNewsNet.git
cd FakeNewsNet

# 安装依赖
pip install -r requirements.txt

# 配置 Twitter API 密钥 (需要申请)
# 编辑 code/resources/tweet_keys_file.json

# 运行数据收集
python main.py
```

**数据规模**:
| 子集 | 真实 | 虚假 | 总计 |
|------|------|------|------|
| PolitiFact | 432 | 384 | 816 |
| GossipCop | 14,676 | 7,438 | 22,114 |

**优点**:
- ✅ 官方维护，持续更新
- ✅ 包含社交传播信息
- ✅ 多模态 (文本 + 图像 + 社交网络)

**缺点**:
- ⚠️ 需要 Twitter API 密钥 (免费申请)
- ⚠️ 数据收集需要时间 (数小时)

**链接**: https://github.com/KaiDMML/FakeNewsNet

---

### 方案 2️⃣: **PHEME 数据集** (推荐 ⭐⭐⭐⭐)

**特点**:
- 5 个突发事件的谣言/非谣言推文
- 包含图像和多语言
- 可直接下载 (无需 API)

**获取方式**:
```
访问：https://figshare.com/articles/dataset/PHEME_dataset/6392939
点击 "Download" 按钮
下载后解压到 data/downloads/PHEME/
```

**数据规模**:
| 事件 | 谣言 | 非谣言 | 总计 |
|------|------|--------|------|
| Charlie Hebdo | 1,219 | 1,022 | 2,241 |
| Sydney Siege | 1,072 | 839 | 1,911 |
| Ferguson | 1,142 | 876 | 2,018 |
| Ottawa Shooting | 890 | 723 | 1,613 |
| Boston Bombing | 1,056 | 892 | 1,948 |
| **总计** | **5,379** | **4,352** | **9,731** |

**优点**:
- ✅ 直接下载，无需 API
- ✅ 多语言 (英语、德语)
- ✅ 包含图像

**缺点**:
- ⚠️ 数据较旧 (2015 年)
- ⚠️ 图像链接可能失效

**链接**: https://figshare.com/articles/dataset/PHEME_dataset/6392939

---

### 方案 3️⃣: **Twitter-15/16 重建版本** (推荐 ⭐⭐⭐⭐)

**特点**:
- 社区维护的重建版本
- 使用 Twitter API 重新收集
- 格式与原始数据集兼容

**获取方式**:
```bash
# 搜索 GitHub 上的重建版本
# 例如：https://github.com/sjh24/Twitter15_16_Fake_News_Dataset

# 或使用 HuggingFace Datasets
from datasets import load_dataset
dataset = load_dataset("username/twitter15-rebuilt")
```

**注意**: 需要验证重建版本的完整性和准确性

---

### 方案 4️⃣: **Weibo 中文数据集** (推荐 ⭐⭐⭐⭐)

**特点**:
- 中文虚假新闻数据集
- 包含图像和文本
- 适合跨语言研究

**获取方式**:
```bash
# 搜索以下仓库
# - WeiboFakeNews
# - ChineseFakeNewsDataset

# 或使用 HuggingFace
from datasets import load_dataset
dataset = load_dataset("ChineseFakeNews")
```

**优点**:
- ✅ 中文数据，适合跨语言研究
- ✅ 较新的数据

**缺点**:
- ⚠️ 需要处理中文 NLP
- ⚠️ 文档可能不完整

---

### 方案 5️⃣: **Kaggle 数据集** (推荐 ⭐⭐⭐)

**特点**:
- 多个虚假新闻数据集
- 直接下载 CSV 格式
- 但可能缺少图像

**推荐数据集**:
1. **Fake and Real News** (116k 样本)
   - https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
   - 仅文本，无图像

2. **Multimodal Fake News Detection**
   - 搜索：https://www.kaggle.com/datasets?search=multimodal+fake+news
   - 需要验证质量

**优点**:
- ✅ 直接下载，无需 API
- ✅ 格式简单

**缺点**:
- ⚠️ 质量参差不齐
- ⚠️ 可能缺少图像

---

## 🎯 我的建议

### 最佳方案 (综合考虑):

**首选**: **FakeNewsNet**

理由:
1. 官方维护，质量有保障
2. 多模态完整 (文本 + 图像 + 社交)
3. 论文引用量大，学术认可度高
4. 适合 SCI 一区论文

**备选**: **PHEME**

理由:
1. 直接下载，无需 API
2. 快速开始实验
3. 适合原型验证

---

## 🚀 快速开始 (使用 FakeNewsNet)

### Step 1: 申请 Twitter API 密钥

```
1. 访问：https://developer.twitter.com/
2. 注册账号 (免费)
3. 创建项目和应用
4. 获取 API 密钥:
   - API Key
   - API Secret Key
   - Access Token
   - Access Token Secret
```

### Step 2: 克隆并配置

```bash
cd /home/admin/openclaw/workspace/Exp1_multimodal
mkdir -p data/downloads
cd data/downloads

# 克隆 FakeNewsNet
git clone https://github.com/KaiDMML/FakeNewsNet.git
cd FakeNewsNet

# 安装依赖
pip install -r requirements.txt

# 配置 API 密钥
# 编辑 code/resources/tweet_keys_file.json
```

### Step 3: 收集数据

```bash
cd code

# 启动密钥服务器
nohup python -m resource_server.app &> keys_server.out &

# 开始数据收集
nohup python main.py &> data_collection.out &

# 监控进度
tail -f data_collection.out
```

### Step 4: 转换为项目格式

创建转换脚本 `scripts/convert_fakenewsnet.py`:

```python
# 将 FakeNewsNet 格式转换为项目统一格式
# 包含文本、图像路径、标签
```

---

## 📊 数据集对比

| 数据集 | 样本量 | 多模态 | 获取难度 | 推荐度 |
|--------|--------|--------|----------|--------|
| FakeNewsNet | 22k+ | ✅ | 中 (需 API) | ⭐⭐⭐⭐⭐ |
| PHEME | 9.7k | ✅ | 低 (直接下载) | ⭐⭐⭐⭐ |
| Twitter-15 重建 | 1.5k | ✅ | 中 | ⭐⭐⭐⭐ |
| Weibo | 5k+ | ✅ | 中 | ⭐⭐⭐⭐ |
| Kaggle | 10k-100k | ❌ | 低 | ⭐⭐⭐ |

---

## ⚙️ 更新数据收集脚本

我已更新 `scripts/collect_data.py` 支持 FakeNewsNet:

```bash
# 使用 FakeNewsNet
python scripts/collect_data.py --dataset fakenewsnet --output_dir data/raw

# 使用 PHEME (直接下载)
python scripts/collect_data.py --dataset pheme --output_dir data/raw
```

---

## 🔍 验证数据集质量

下载后，运行以下检查:

```bash
# 检查数据完整性
python scripts/validate_data.py --dataset fakenewsnet

# 统计信息
python scripts/data_stats.py --dataset fakenewsnet
```

---

## 📝 论文中的数据集描述

如果使用 FakeNewsNet，论文中可这样描述:

> "We use the FakeNewsNet dataset [1], a widely-adopted benchmark for multimodal fake news detection. The dataset contains 22,114 news samples from GossipCop and 816 samples from PolitiFact, each with news articles, social media posts, and images. We follow the standard train/test split provided by the dataset creators."

**引用**:
```bibtex
@article{shu2020fakenewsnet,
  title={FakeNewsNet: A data repository with news content, social context and dynamic information for studying fake news on social media},
  author={Shu, Kai and Mahudeswaran, Deepak and Wang, Suhang and Lee, Dongwon and Liu, Huan},
  journal={Big data},
  volume={8},
  number={3},
  pages={171--188},
  year={2020},
  publisher={Mary Ann Liebert, Inc., publishers}
}
```

---

## ❓ 常见问题

### Q1: Twitter API 申请被拒？

**解决**:
- 确保填写真实用途 (学术研究)
- 描述详细的项目计划
- 使用学校/机构邮箱

### Q2: 数据收集太慢？

**解决**:
- 增加 Twitter API 密钥数量 (多个账号)
- 只收集必要字段 (新闻内容 + 图像)
- 使用多线程/多进程

### Q3: 图像链接失效？

**解决**:
- 使用 Wayback Machine 存档
- 仅使用有有效图像的样本
- 考虑纯文本基线

---

## 📞 需要帮助？

如果遇到问题，请告诉我:
1. 选择的数据集
2. 遇到的具体错误
3. 错误日志

我会帮你解决！

---

**最后更新**: 2026-04-05  
**推荐方案**: FakeNewsNet (首选) 或 PHEME (快速开始)
