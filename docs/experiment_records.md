# 实验记录

**项目**: Exp1_multimodal - MM-MoE-LoRA  
**开始日期**: 2026-04-05  
**目标**: SCI 一区论文

---

## 📅 2026-04-05 (Day 1) - 项目启动

### 完成工作

#### 1. 项目架构设计 ✅
- 确定核心创新点：MoE-LoRA + 跨模态对比学习 + 不确定性估计
- 设计整体架构：RoBERTa-base (文本) + ViT (视觉) + MoE 融合
- 规划实验方案：主实验 + 5 个消融变体

#### 2. 文献调研 ✅
- 完成文献综述 (`literature/survey.md`)
- 调研方向:
  - 多模态虚假新闻检测 (EANN, SpotFake, MCAN)
  - PEFT 方法 (LoRA, DoRA, MoE-LoRA)
  - 跨模态融合 (CLIP, ALBEF)
  - 不确定性估计 (MC Dropout, Evidential DL)
- 识别研究空白：MoE-LoRA 未用于多模态假新闻检测

#### 3. 代码实现 ✅
| 模块 | 文件 | 状态 |
|------|------|------|
| 数据收集 | `scripts/collect_data.py` | ✅ 完成 |
| 数据预处理 | `scripts/preprocess.py` | ✅ 完成 |
| 数据加载器 | `models/dataset_loader.py` | ✅ 完成 |
| MoE-LoRA | `models/moe_lora.py` | ✅ 完成 |
| 实验配置 | `experiments/config.yaml` | ✅ 完成 |

#### 4. 文档编写 ✅
- `README.md` - 项目概述
- `docs/GETTING_STARTED.md` - 启动指南
- `requirements.txt` - 依赖清单

### 技术决策

1. **基座模型选择**: RoBERTa-base (而非 LLM)
   - 理由：轻量、成熟、A40 30GB 可容纳多模态
   - 优势：可快速迭代实验

2. **MoE 配置**: 3 专家，top-k=2 路由
   - 平衡性能与计算开销
   - 防止专家坍塌

3. **数据集**: Twitter-15 起始
   - 标准基准，便于对比
   - 后续扩展到 Weibo-20 (中文)

### 遇到的问题

**无** - 所有计划任务已完成

### 明日计划 (2026-04-06)

- [ ] 用户下载 Twitter-15 数据集
- [ ] 实现模型主架构 (`models/classifier.py`)
- [ ] 实现训练脚本 (`scripts/train.py`)
- [ ] 运行数据加载测试

---

## 📅 2026-04-06 (Day 2) - 计划

### 目标
- 完成模型主架构
- 完成训练脚本
- 验证数据加载

### 关键任务
1. 数据下载状态确认
2. `models/classifier.py` 实现
3. `scripts/train.py` 实现
4. 端到端测试

---

## 📊 实验计划

### 主实验

| 实验 ID | 配置 | 预期 F1 | 状态 |
|---------|------|---------|------|
| E1 | 标准 LoRA (基线) | 0.82 | ⏳ 待运行 |
| E2 | MoE-LoRA (3 专家) | 0.85 | ⏳ 待运行 |
| E3 | MoE-LoRA + 对比学习 | 0.87 | ⏳ 待运行 |
| E4 | E3 + 不确定性 | 0.87 | ⏳ 待运行 |

### 消融实验

| 变体 | 修改 | 目的 |
|------|------|------|
| no_moe | 单专家 | 验证 MoE 有效性 |
| no_contrastive | 无对比损失 | 验证对比学习贡献 |
| no_uncertainty | 无不确定性 | 验证不确定性价值 |
| standard_lora | 标准 LoRA | 与 SOTA 对比 |

---

## 📈 性能目标

### Twitter-15 基准

| 方法 | Accuracy | F1 | 来源 |
|------|----------|-----|------|
| EANN (2018) | 0.79 | 0.78 | KDD |
| SpotFake (2019) | 0.82 | 0.81 | ICDM |
| MCAN (2023) | 0.86 | 0.85 | WWW |
| **MM-MoE-LoRA (Ours)** | **0.88+** | **0.87+** | 目标 |

---

## 📝 笔记

### 创新点总结

1. **首个 MoE-LoRA 多模态假新闻检测框架**
2. **跨模态对比学习增强 MoE 路由**
3. **不确定性估计支持人机协同**

### 论文结构规划

1. Introduction (引言)
2. Related Work (相关工作)
3. Method (方法)
   - 3.1 Problem Formulation
   - 3.2 MM-MoE-LoRA Architecture
   - 3.3 Cross-Modal Contrastive Learning
   - 3.4 Uncertainty Estimation
4. Experiments (实验)
   - 4.1 Experimental Setup
   - 4.2 Main Results
   - 4.3 Ablation Studies
   - 4.4 Case Study
5. Conclusion (结论)

---

**最后更新**: 2026-04-05  
**下次更新**: 2026-04-06
