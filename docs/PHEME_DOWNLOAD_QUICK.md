# 🚀 PHEME 数据集快速下载

**目标**: 30 分钟内完成数据准备

---

## 📥 下载步骤

### Step 1: 访问下载页面

打开浏览器访问：
```
https://figshare.com/articles/dataset/PHEME_dataset/6392939
```

### Step 2: 下载文件

1. 点击页面右侧的 **"Download"** 按钮
2. 选择 **"PHEME.zip"** (约 100-200MB)
3. 等待下载完成

### Step 3: 移动到项目目录

```bash
# 假设文件在 ~/Downloads/
mv ~/Downloads/PHEME.zip /home/admin/openclaw/workspace/Exp1_multimodal/data/downloads/

# 或者直接用 cp
cp ~/Downloads/PHEME.zip /home/admin/openclaw/workspace/Exp1_multimodal/data/downloads/
```

### Step 4: 运行处理脚本

```bash
cd /home/admin/openclaw/workspace/Exp1_multimodal

# 收集数据 (自动解压)
python scripts/collect_data.py --dataset pheme --output_dir data/raw

# 预处理
python scripts/preprocess.py --dataset pheme --output_dir data/processed
```

### Step 5: 验证数据

```bash
# 检查输出
ls -lh data/raw/pheme/
ls -lh data/processed/pheme/

# 查看统计信息
cat data/processed/pheme/statistics.json
```

---

## ⏱️ 预计时间

| 步骤 | 时间 |
|------|------|
| 下载 | 5-15 分钟 (取决于网络) |
| 移动文件 | 1 分钟 |
| 收集脚本 | 2-5 分钟 |
| 预处理 | 1-2 分钟 |
| **总计** | **约 15-25 分钟** |

---

## 📊 预期输出

```
data/raw/pheme/
├── train.json (约 3k 样本)
├── test.json (约 6k 样本)
└── metadata.json

data/processed/pheme/
├── train.json (预处理后)
├── test.json (预处理后)
├── images/ (图像文件夹)
└── statistics.json (统计信息)
```

---

## ❓ 遇到问题？

### 问题 1: 下载链接打不开

**解决**: 
- 尝试使用科学上网
- 或使用镜像下载
- 或联系我要备用链接

### 问题 2: 解压失败

**解决**:
```bash
cd /home/admin/openclaw/workspace/Exp1_multimodal/data/downloads
unzip -x PHEME.zip
```

### 问题 3: 处理脚本报错

**解决**:
- 检查 Python 依赖是否安装
- 查看错误日志
- 告诉我具体错误信息

---

## ✅ 完成后告诉我

下载并运行处理脚本后，请告诉我：
1. 是否成功？
2. 样本数量是多少？
3. 有任何错误吗？

然后我会继续实现模型和训练脚本！🚀

---

**状态**: ⏳ 等待 PHEME 下载完成  
**更新时间**: 2026-04-05 17:05
