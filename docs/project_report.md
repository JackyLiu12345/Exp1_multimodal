# 📊 项目启动报告 - Exp1_multimodal

**日期**: 2026-04-05  
**阶段**: Day 1 - 项目启动完成  
**目标**: SCI 一区论文 - MM-MoE-LoRA 多模态虚假信息检测

---

## ✅ 今日完成工作

### 1. 文献调研与研究方向确定

**核心创新点**:
- ✅ **MoE-LoRA 多模态融合**: 首个将 MoE-LoRA 应用于多模态假新闻检测
- ✅ **跨模态对比学习**: 增强图文一致性建模
- ✅ **不确定性估计**: 支持低置信度样本人工审核

**研究空白识别**:
| 空白 | 我们的方案 |
|------|------------|
| MoE-LoRA 未用于多模态假新闻 | 提出 MM-MoE-LoRA 框架 |
| 缺少不确定性感知 | 添加不确定性估计头 |
| 对比学习 + MoE 未结合 | 联合优化损失函数 |

**文献综述**: `literature/survey.md` (5.3KB)
- 调研 20+ 篇核心论文
- 整理 SOTA 方法对比
- 确定技术路线

---

### 2. 项目架构搭建

**目录结构**:
```
Exp1_multimodal/
├── data/                      # 数据集
├── scripts/                   # 脚本
│   ├── collect_data.py       ✅ 数据收集 (12KB)
│   └── preprocess.py         ✅ 数据预处理 (9.6KB)
├── models/                    # 模型
│   ├── dataset_loader.py     ✅ 数据加载器 (10.6KB)
│   └── moe_lora.py           ✅ MoE-LoRA 核心 (14KB)
├── experiments/
│   └── config.yaml           ✅ 实验配置 (3.4KB)
├── docs/
│   ├── GETTING_STARTED.md    ✅ 启动指南 (3.8KB)
│   ├── experiment_records.md ✅ 实验记录 (2.6KB)
│   └── project_report.md     ✅ 本报告
├── literature/
│   └── survey.md             ✅ 文献综述 (5.3KB)
└── requirements.txt          ✅ 依赖清单
```

**总代码量**: ~59KB (核心模块)

---

### 3. 核心模块实现

#### 📦 数据收集脚本 (`scripts/collect_data.py`)

**功能**:
- 支持 Twitter-15/16, Weibo-20, PHEME 数据集
- 自动化解压和处理
- 提供详细下载指引

**使用示例**:
```bash
python scripts/collect_data.py --dataset twitter15 --output_dir data/raw
```

#### 🔧 数据预处理脚本 (`scripts/preprocess.py`)

**功能**:
- 文本清洗 (移除 URL、提及、特殊字符)
- 图像预处理 (调整大小、标准化)
- 生成统一 JSON 格式
- 自动统计数据集信息

**使用示例**:
```bash
python scripts/preprocess.py --dataset twitter15 --output_dir data/processed
```

#### 📊 数据加载器 (`models/dataset_loader.py`)

**功能**:
- PyTorch Dataset + DataLoader
- 多模态输入 (文本 + 图像)
- 类别不平衡处理 (加权采样)
- 数据增强 (训练集)

**关键类**:
- `MultimodalDataset`: 多模态数据集
- `create_dataloader()`: 创建 DataLoader
- `create_train_val_test_loaders()`: 一键创建三个分割

#### 🧠 MoE-LoRA 核心模块 (`models/moe_lora.py`)

**组件**:
| 类 | 功能 | 代码行数 |
|----|------|----------|
| `LoRALayer` | 标准 LoRA 层 | ~60 行 |
| `Expert` | 单个 LoRA 专家 | ~50 行 |
| `GatingNetwork` | 动态路由网络 | ~80 行 |
| `MoELoRA` | 多专家模块 | ~120 行 |
| `CrossModalFusion` | 跨模态融合 | ~100 行 |

**核心特性**:
- ✅ 支持多种路由方法 (softmax / top-k / GShard)
- ✅ 辅助损失 (负载均衡)
- ✅ 跨模态交叉注意力
- ✅ 可配置专家数量、LoRA 秩

**测试代码**: 内置 `__main__` 测试，可直接运行验证

---

### 4. 实验配置

**配置文件**: `experiments/config.yaml`

**关键参数**:
```yaml
model:
  text_encoder: FacebookAI/roberta-base
  vision_encoder: google/vit-base-patch16-224
  moe_lora:
    num_experts: 3
    lora_r: 8
    lora_alpha: 16

training:
  batch_size: 8
  gradient_accumulation_steps: 2  # 有效 batch=16
  num_epochs: 10

loss:
  classification:
    weight: 1.0
  contrastive:
    weight: 0.5
  diversity:
    weight: 0.1
```

**消融实验配置**:
- `full_model`: 完整模型
- `no_moe`: 单专家
- `no_contrastive`: 无对比学习
- `no_uncertainty`: 无不确定性估计
- `standard_lora`: 标准 LoRA

---

## 📋 下一步行动

### 🔴 高优先级 (本周内)

1. **数据下载** (用户执行)
   ```bash
   # 访问下载页面
   https://www.cs.columbia.edu/~chenhao/resources/
   
   # 下载 twitter15.tar.gz
   # 移动到 data/downloads/
   
   # 运行收集脚本
   python scripts/collect_data.py --dataset twitter15
   
   # 预处理
   python scripts/preprocess.py --dataset twitter15
   ```

2. **模型主架构实现** (AI 协助)
   - 创建 `models/classifier.py`
   - 整合 RoBERTa + ViT + MoE-LoRA
   - 实现不确定性估计头

3. **训练脚本实现** (AI 协助)
   - 创建 `scripts/train.py`
   - 实现训练循环
   - 集成 WandB/TensorBoard 日志

4. **端到端测试**
   - 小数据集验证
   - 显存占用测试
   - 训练速度基准

### 🟡 中优先级 (下周)

5. **基线实验**
   - 标准 LoRA 训练
   - 记录性能指标

6. **MoE-LoRA 实验**
   - 启用 MoE 模块
   - 对比性能提升

### 🟢 低优先级 (第 3 周)

7. **消融实验**
8. **可视化分析**
9. **论文初稿**

---

## 🎯 预期时间线

```
Week 1 (04-05 ~ 04-12): 数据准备 + 训练框架
├── Day 1 (04-05): ✅ 项目启动
├── Day 2 (04-06): 模型架构 + 训练脚本
├── Day 3-4 (04-07~08): 数据下载 + 测试
└── Day 5-7 (04-09~11): 基线实验

Week 2 (04-12 ~ 04-19): 主实验
├── MoE-LoRA 训练
├── 对比学习集成
└── 初步结果分析

Week 3 (04-19 ~ 04-26): 消融实验
├── 5 个变体实验
├── 可视化分析
└── 结果整理

Week 4-5 (04-26 ~ 05-10): 论文撰写
└── 初稿完成

Week 6 (05-10 ~ 05-17): 修改 + 投稿
```

---

## 📊 资源需求

### GPU 资源

| 阶段 | 显存需求 | 预计时间 |
|------|----------|----------|
| 数据预处理 | <5GB | 1-2 小时 |
| 基线训练 | ~15GB | 4-6 小时/epoch |
| MoE-LoRA 训练 | ~20GB | 6-8 小时/epoch |
| 消融实验 | ~20GB | 2-3 天 |

**A40 30GB**: ✅ 满足所有需求

### 存储需求

| 项目 | 大小 |
|------|------|
| Twitter-15 原始数据 | ~500MB |
| 预处理后数据 | ~1GB |
| 模型检查点 | ~2GB/epoch |
| 日志文件 | ~100MB |
| **总计** | ~5-10GB |

---

## ⚠️ 风险提示

### 技术风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 数据下载失败 | 低 | 中 | 使用备用数据集 (Weibo-20) |
| 显存不足 | 低 | 中 | 梯度检查点 + 减小 batch |
| 训练不收敛 | 中 | 高 | 学习率调优 + 检查数据质量 |
| 性能提升不明显 | 中 | 高 | 调整 MoE 配置 + 对比损失权重 |

### 时间风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 实验周期长 | 中 | 中 | 并行运行多个实验 |
| 论文撰写慢 | 低 | 中 | 提前准备图表 + 模板 |

---

## 📈 成功指标

### 技术指标

- [ ] Twitter-15 F1 ≥ 0.85 (基线)
- [ ] MoE-LoRA vs 标准 LoRA 提升 ≥ 2%
- [ ] 训练稳定收敛 (10 epochs 内)
- [ ] 显存占用 < 25GB

### 论文指标

- [ ] 目标期刊：Information Fusion (IF=18.6)
- [ ] 创新点：3 个核心贡献
- [ ] 实验：主实验 + 5 个消融变体
- [ ] 可复现性：代码开源

---

## 📝 关键决策记录

### 决策 1: 基座模型选择

**选项**:
- A: LLM (Qwen2.5-7B)
- B: Encoder (RoBERTa-base)

**决策**: B - RoBERTa-base

**理由**:
- 轻量，适合快速迭代
- A40 30GB 可容纳多模态
- 成熟稳定，调试容易
- 性能足够达到 SOTA

### 决策 2: MoE 专家数量

**选项**:
- A: 2 专家
- B: 3 专家
- C: 4+ 专家

**决策**: B - 3 专家

**理由**:
- 2 专家可能不足
- 4+ 专家计算开销大
- 3 专家平衡性能与效率

### 决策 3: 数据集选择

**顺序**:
1. Twitter-15 (英文，标准基准)
2. Weibo-20 (中文，跨语言验证)
3. PHEME (多事件，泛化性)

---

## 🎓 论文贡献总结

### 理论贡献

1. **首个 MoE-LoRA 多模态假新闻检测框架**
2. **跨模态对比学习增强 MoE 路由机制**
3. **不确定性估计支持人机协同审核**

### 实践贡献

1. **开源代码库**: 完整实现 + 预训练模型
2. **标准化实验**: 在 3 个数据集上验证
3. **可复现性**: 详细配置 + 文档

---

## 📞 联系方式

**项目地址**: `/home/admin/openclaw/workspace/Exp1_multimodal`  
**文档索引**: `docs/GETTING_STARTED.md`  
**实验记录**: `docs/experiment_records.md`  
**文献综述**: `literature/survey.md`

---

**报告生成时间**: 2026-04-05  
**下次汇报**: 2026-04-06 (训练脚本完成后)  
**状态**: 🟢 正常推进
