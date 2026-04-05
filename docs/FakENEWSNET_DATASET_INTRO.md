# 📊 FakeNewsNet 数据集完整介绍

**更新时间**: 2026-04-05 17:50  
**来源**: https://github.com/KaiDMML/FakeNewsNet

---

## 🎯 数据集概述

**FakeNewsNet** 是一个用于虚假新闻检测研究的综合数据集，包含新闻内容、社交上下文和动态信息三个维度。

**论文**: *FakeNewsNet: A data repository with news content, social context and dynamic information for studying fake news on social media* (Big Data 2020)

**引用量**: 2000+ 次 (虚假新闻检测领域最广泛使用的数据集之一)

---

## 📁 数据来源

FakeNewsNet 包含 **2 个子数据集**，分别来自两个事实核查网站：

### 1️⃣ **GossipCop** 

**网址**: https://www.gossipcop.com/

**特点**:
- 专注于娱乐新闻的事实核查
- 覆盖名人八卦、娱乐事件等
- 样本量大，适合大规模实验

**规模**:
- Real news: 16,817 条
- Fake news: 5,323 条
- **总计**: 22,140 条

---

### 2️⃣ **PolitiFact**

**网址**: https://www.politifact.com/

**特点**:
- 专注于政治新闻的事实核查
- 覆盖政治事件、政策声明等
- 获得普利策奖的事实核查网站

**规模**:
- Real news: 624 条
- Fake news: 432 条
- **总计**: 1,056 条

---

## 📋 CSV 文件结构 (当前已有数据)

你当前已有的 CSV 文件包含以下列：

```python
# 每个 CSV 文件的列结构
{
    "id": "唯一标识符 (如：gossipcop-12345)",
    "url": "新闻文章的网址",
    "title": "新闻标题",
    "tweet_ids": "分享该新闻的推文 ID 列表 (制表符分隔)"
}
```

### 示例数据

```csv
id,url,title,tweet_ids
gossipcop-1,https://example.com/news1,Celebrity X announces...,123456789	987654321	...
gossipcop-2,https://example.com/news2,Breaking: Movie Y...,111222333	444555666	...
```

---

## 🗂️ 完整数据集结构 (需要 Twitter API 收集)

使用 Twitter API 收集后，完整的目录结构如下：

```
FakeNewsNet/
├── gossipcop/
│   ├── fake/
│   │   ├── gossipcop-1/
│   │   │   ├── news_content.json      # 新闻内容
│   │   │   ├── tweets/                # 推文
│   │   │   │   ├── 886941526458347521.json
│   │   │   │   └── ...
│   │   │   └── retweets/              # 转发
│   │   └── ...
│   └── real/
│       └── ...
├── politifact/
│   ├── fake/
│   └── real/
├── user_profiles/                     # 用户资料
├── user_timeline_tweets/              # 用户时间线
├── user_followers/                    # 粉丝关系
└── user_following/                    # 关注关系
```

---

## 📦 数据维度详解

### 维度 1: **新闻内容 (News Content)**

**文件**: `news_content.json`

**包含字段**:
```json
{
    "text": "新闻正文内容",
    "images": ["图片 URL 列表"],
    "publish_date": "发布日期",
    "title": "标题",
    "url": "原文链接",
    "author": "作者 (如果有)"
}
```

**用途**:
- 文本分类 (虚假/真实)
- 多模态分析 (文本 + 图像)
- 时间序列分析

---

### 维度 2: **社交上下文 (Social Context)**

**文件**: `tweets/`, `retweets/`

**推文内容** (`tweets/<tweet_id>.json`):
```json
{
    "id": "推文 ID",
    "text": "推文内容",
    "user": {
        "id": "用户 ID",
        "name": "用户名",
        "screen_name": "用户名",
        "followers_count": 粉丝数,
        "friends_count": 关注数,
        ...
    },
    "created_at": "发布时间",
    "retweet_count": 转发数,
    "favorite_count": 点赞数,
    ...
}
```

**用途**:
- 社交传播模式分析
- 用户可信度评估
- 早期检测 (基于转发模式)

---

### 维度 3: **动态信息 (Dynamic Information)**

**文件**: `user_timeline_tweets/`, `user_followers/`, `user_following/`

**包含**:
- 用户历史推文 (最多 200 条)
- 粉丝关系网络
- 关注关系网络

**用途**:
- 用户行为分析
- 社交网络图构建
- 谣言传播路径追踪

---

## 📊 当前数据状态

### ✅ 你已有的数据

**位置**: `/home/admin/FakeNewsNet/dataset/`

```
dataset/
├── gossipcop_fake.csv   (12.5 MB, 5,323 条)
├── gossipcop_real.csv   (20.0 MB, 16,817 条)
├── politifact_fake.csv  (3.3 MB, 432 条)
└── politifact_real.csv  (8.3 MB, 624 条)
```

**总计**: 23,196 条新闻的元数据

**包含信息**:
- ✅ 新闻 ID
- ✅ 新闻链接
- ✅ 新闻标题
- ✅ 推文 ID 列表

**缺少信息** (需要 Twitter API 收集):
- ❌ 新闻正文内容
- ❌ 新闻图像
- ❌ 推文具体内容
- ❌ 用户信息
- ❌ 社交网络关系

---

## 🔧 数据收集流程

### 步骤 1: 申请 Twitter API

**网址**: https://developer.twitter.com/

**需要的密钥**:
```json
{
    "app_key": "YOUR_APP_KEY",
    "app_secret": "YOUR_APP_SECRET",
    "oauth_token": "YOUR_OAUTH_TOKEN",
    "oauth_token_secret": "YOUR_OAUTH_TOKEN_SECRET"
}
```

**申请时间**: 通常 1-2 天审批

---

### 步骤 2: 配置密钥

**文件**: `code/resources/tweet_keys_file.json`

```json
[
    {
        "app_key": "...",
        "app_secret": "...",
        "oauth_token": "...",
        "oauth_token_secret": "..."
    }
]
```

---

### 步骤 3: 运行收集脚本

```bash
cd FakeNewsNet/code

# 启动密钥服务器
nohup python -m resource_server.app &> keys_server.out &

# 开始数据收集
nohup python main.py &> data_collection.out &
```

**预计时间**: 4-8 小时 (取决于网络和 API 限制)

---

## 📈 数据统计信息

### 整体统计

| 指标 | GossipCop | PolitiFact | 总计 |
|------|-----------|------------|------|
| 真实新闻 | 16,817 | 624 | 17,441 |
| 虚假新闻 | 5,323 | 432 | 5,755 |
| **总计** | **22,140** | **1,056** | **23,196** |
| 虚假比例 | 24.0% | 40.9% | 24.8% |

### 标签分布

```
Real: ████████████████████████████████ 75.2%
Fake: ██████████ 24.8%
```

---

## 🎯 使用场景

### 场景 1: **纯文本分类** (当前可用)

**输入**: 新闻标题  
**模型**: RoBERTa, BERT 等  
**输出**: 真实/虚假

**优点**:
- ✅ 使用现有 CSV 数据即可
- ✅ 快速验证模型
- ✅ 基线对比方便

**缺点**:
- ⚠️ 缺少完整新闻内容
- ⚠️ 无法使用多模态

---

### 场景 2: **多模态分类** (需要 Twitter API)

**输入**: 新闻文本 + 图像  
**模型**: ViLBERT, LXMERT, CLIP+LLM  
**输出**: 真实/虚假

**优点**:
- ✅ 更接近实际应用
- ✅ 性能更好
- ✅ 适合 SCI 论文

**缺点**:
- ⚠️ 需要 Twitter API
- ⚠️ 数据收集时间长

---

### 场景 3: **社交网络分析** (需要完整数据)

**输入**: 推文 + 用户关系 + 传播模式  
**模型**: 图神经网络 (GNN)  
**输出**: 真实/虚假 + 传播预测

**优点**:
- ✅ 最全面的数据
- ✅ 可研究传播机制
- ✅ 高影响力论文

**缺点**:
- ⚠️ 需要完整数据收集
- ⚠️ 计算复杂度高

---

## 📚 相关论文

### FakeNewsNet 原论文

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

### 使用 FakeNewsNet 的高引用论文

1. **SpotFake** (ICDM 2019): 多模态假新闻检测
2. **SafeNet** (WWW 2020): 基于社交网络的检测
3. **MCAN** (WWW 2023): 跨模态协同注意力

---

## 🔄 数据更新

**最新版本**: 持续更新中

**更新频率**: 不定期 (取决于事实核查网站)

**获取最新数据**:
```bash
cd FakeNewsNet
git pull origin master
```

---

## ⚠️ 使用限制

### 版权和隐私

- ❌ **不能直接分发完整数据** (Twitter 政策)
- ✅ **可以分发 CSV 元数据**
- ✅ 需要使用 Twitter API 重新收集

### 引用要求

使用 FakeNewsNet 的论文必须引用原论文。

---

## 🎯 我们的使用计划

### Phase 1: 快速验证 (今天)

**使用**: CSV 元数据 (标题)  
**模型**: RoBERTa + MoE-LoRA  
**目标**: F1 ≥ 0.88

### Phase 2: 完整数据 (下周)

**使用**: 完整数据 (文本 + 图像)  
**模型**: 多模态 MoE-LoRA  
**目标**: F1 ≥ 0.90

### Phase 3: 社交网络 (下月)

**使用**: 社交上下文  
**模型**: GNN + MoE-LoRA  
**目标**: 研究传播模式

---

## 📞 常见问题

### Q1: 为什么不能直接下载完整数据？

**答**: Twitter 隐私政策禁止直接分发推文内容，需要通过 API 收集。

### Q2: Twitter API 免费吗？

**答**: 有免费额度 (每月 50 万推文)，足够学术研究。

### Q3: 数据收集需要多长时间？

**答**: 23k 条新闻约需 4-8 小时 (取决于 API 密钥数量)。

### Q4: CSV 数据够用吗？

**答**: 
- ✅ 足够快速验证代码
- ✅ 可以发表初步结果
- ⚠️ 完整论文需要多模态数据

---

**状态**: ✅ CSV 数据已处理完成  
**下一步**: 运行快速训练验证  
**更新时间**: 2026-04-05 17:55
