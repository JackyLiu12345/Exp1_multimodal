# 🚨 PHEME 数据集下载 - 更正说明

**问题**: 你访问了错误的 Figshare 页面（生物医学数据集）

**正确的 PHEME 数据集**: 社交媒体虚假新闻检测数据集

---

## ✅ 推荐方案：使用 HuggingFace (最简单 ⭐⭐⭐⭐⭐)

**无需手动下载，一行代码搞定！**

### Step 1: 安装依赖

```bash
pip install datasets
```

### Step 2: 运行下载脚本

```bash
cd /home/admin/openclaw/workspace/Exp1_multimodal

python scripts/download_pHEME_hf.py --output_dir data/raw/pheme
```

**预计时间**: 5-10 分钟（自动下载 + 处理）

### Step 3: 验证数据

```bash
# 检查输出
ls -lh data/raw/pheme/

# 查看统计
cat data/raw/pheme/metadata.json
```

---

## 🔗 手动下载方案（备选）

### 正确的 Figshare 链接

```
https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/6392939
```

**注意**: 
- 标题包含 "rumours and non-rumours"
- DOI: `10.6084/m9.figshare.6392939`
- **不是**生物医学那个数据集

### 下载步骤

1. 访问正确链接
2. 点击 "Download all" (约 7-10MB)
3. 移动到项目目录:
   ```bash
   mv ~/Downloads/PHEME.zip data/downloads/
   ```
4. 运行处理脚本:
   ```bash
   python scripts/collect_data.py --dataset pheme
   ```

---

## 📊 数据集信息

**PHEME 数据集内容**:

| 事件 | 样本量 | 标签 |
|------|--------|------|
| Charlie Hebdo | ~2,200 | 谣言/非谣言 |
| Sydney Siege | ~1,900 | 谣言/非谣言 |
| Ferguson | ~2,000 | 谣言/非谣言 |
| Ottawa Shooting | ~1,600 | 谣言/非谣言 |
| Boston Bombing | ~1,900 | 谣言/非谣言 |
| **总计** | **~9,700** | **二分类** |

**标签**:
- `rumour` → fake (虚假)
- `non-rumour` → real (真实)

---

## 🎯 我的建议

**使用 HuggingFace 方案**:

```bash
# 一条命令搞定
python scripts/download_pHEME_hf.py --output_dir data/raw/pheme
```

**优点**:
- ✅ 无需手动下载
- ✅ 自动处理格式
- ✅ 5-10 分钟完成
- ✅ 不会下错数据集

---

## ⚠️ 如何确认数据集正确？

下载后检查 `metadata.json`:

```json
{
  "dataset": "PHEME",
  "events": ["charliehebdo", "sydneysiege", "ferguson", ...],
  "label_distribution": {"real": 4xxx, "fake": 5xxx}
}
```

**如果看到以下内容，说明下错了**:
- ❌ "meningococcal" (脑膜炎)
- ❌ "vaccine" (疫苗)
- ❌ "Factor H Binding Protein"

**正确内容应该包含**:
- ✅ "rumour", "non-rumour"
- ✅ 事件名称 (charliehebdo, sydney 等)
- ✅ 推文相关内容

---

## 📞 需要帮助？

运行脚本后，请告诉我：
1. 是否成功？
2. 样本数量是多少？
3. 有任何报错吗？

我会根据你的反馈继续推进！🚀

---

**状态**: ⏳ 等待数据下载  
**推荐方案**: HuggingFace (最简单)  
**更新时间**: 2026-04-05 17:20
