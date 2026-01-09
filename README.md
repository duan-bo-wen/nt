# 图像描述生成项目（Image Captioning）

本项目实现了多个图像描述生成模型，用于服饰图像的中文/英文描述生成。项目包含从简单基线到预训练模型的不同技术路线，并提供了完整的训练、评估和前端界面。

## 📋 目录

- [项目概述](#项目概述)
- [模型列表](#模型列表)
- [快速开始](#快速开始)
- [模型方法对比](#模型方法对比)
- [预期与实际结果分析](#预期与实际结果分析)
- [项目结构](#项目结构)

---

## 项目概述

本项目实现了5个不同的图像描述生成模型，涵盖了多种技术路线：

1. **Original_Model**: 网格特征 + 简单注意力 + GRU（基线模型）
2. **Model1_YellowOrange**: 网格特征 + 加性注意力 + GRU（改进基线）
3. **Model2_Transformer**: 网格特征 + Transformer编码器/解码器
4. **Model3_CNN_GRU**: CNN全局特征 + GRU（简单结构）
5. **Ex1_BLIP**: 基于BLIP的多模态预训练模型微调

所有模型均实现了METEOR、ROUGE-L、CIDEr-D三个评估指标，并提供了统一的训练、评估和推理接口。

---

## 模型列表

### 1. Original_Model
- **结构**: ResNet101网格特征 + 简单加性注意力 + GRU解码器
- **特点**: 最基础的注意力机制实现，作为对照基线
- **详细文档**: [Original_Model/README.md](Original_Model/README.md)

### 2. Model1_YellowOrange
- **结构**: ResNet101网格特征 + 加性注意力 + GRU解码器 + Beam Search
- **特点**: 优化了注意力机制，支持beam search生成，训练稳定
- **详细文档**: [Model1_YellowOrange/README.md](Model1_YellowOrange/README.md)

### 3. Model2_Transformer
- **结构**: ResNet101网格特征 + Transformer编码器 + Transformer解码器 + BERT
- **特点**: 完全基于Transformer架构，支持长依赖建模
- **详细文档**: [Model2_Transformer/README.md](Model2_Transformer/README.md)

### 4. Model3_CNN_GRU
- **结构**: ResNet101全局特征 + GRU解码器
- **特点**: 最简单的结构，无注意力机制，适合快速实验
- **详细文档**: [Model3_CNN_GRU/README.md](Model3_CNN_GRU/README.md)

### 5. Ex1_BLIP
- **结构**: Vision Transformer + Transformer解码器（预训练）
- **特点**: 基于大规模预训练的BLIP模型，支持微调
- **详细文档**: [Ex1_BLIP/README.md](Ex1_BLIP/README.md)

---

## 快速开始

### 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

### 数据预处理

```bash
python main.py preprocess
```

预处理将生成：
- `data/output/vocab.json`: 词表
- `data/output/encoded_captions_*.json`: 编码后的caption
- `data/output/image_paths_*.json`: 图像路径
- `data/output/caplens_*.json`: caption长度

### 训练模型

```bash
# 训练Model1
python main.py train --model model1

# 训练Model2
python main.py train --model model2

# 训练Original
python main.py train --model original

# 训练CNN-GRU
python main.py train --model cnn_gru

# 训练BLIP（微调）
python main.py train --model blip
```

### 评估模型

```bash
# 评估Model1
python main.py eval --model model1

# 评估Model2
python main.py eval --model model2

# 评估Original
python main.py eval --model original

# 评估CNN-GRU
python main.py eval --model cnn_gru

# 评估BLIP
python main.py eval --model blip
```

### 启动Web前端

```bash
python main.py ui
```

前端将在浏览器中打开，支持上传图片并选择不同模型进行描述生成。

---

## 模型方法对比

### 图像编码方式

| 模型 | 图像编码方法 | 特征维度 | 空间信息保留 |
|------|------------|---------|------------|
| Original_Model | ResNet101网格特征 | (B, 196, 2048) | ✅ 保留 |
| Model1_YellowOrange | ResNet101网格特征 | (B, 196, 2048) | ✅ 保留 |
| Model2_Transformer | ResNet101网格特征 + Transformer编码 | (B, 196, 512) | ✅ 保留 |
| Model3_CNN_GRU | ResNet101全局特征 | (B, 2048) | ❌ 不保留 |
| Ex1_BLIP | Vision Transformer (ViT) | 可变序列长度 | ✅ 保留 |

### 文本解码方式

| 模型 | 解码器类型 | 注意力机制 | 生成策略 |
|------|-----------|-----------|---------|
| Original_Model | GRU | 简单加性注意力 | Greedy |
| Model1_YellowOrange | GRU | 加性注意力 | Beam Search (k=3) |
| Model2_Transformer | Transformer解码器 | 多头自注意力+交叉注意力 | Beam Search (k=5) |
| Model3_CNN_GRU | GRU | 无 | Beam Search (k=3) |
| Ex1_BLIP | Transformer解码器 | 多头自注意力+交叉注意力 | Beam Search (k=5) |

### 训练策略

| 模型 | 损失函数 | 优化器 | 学习率 | 正则化 |
|------|---------|--------|--------|--------|
| Original_Model | CrossEntropyLoss | Adam | 1e-4 | 梯度裁剪(max_norm=5.0) |
| Model1_YellowOrange | CrossEntropyLoss | Adam | 1e-4 | 梯度裁剪(max_norm=5.0) |
| Model2_Transformer | CrossEntropyLoss (Label Smoothing=0.1) | Adam | 1e-4 | Warmup + 梯度裁剪 |
| Model3_CNN_GRU | CrossEntropyLoss | Adam | 1e-4 | 梯度裁剪(max_norm=5.0) |
| Ex1_BLIP | CrossEntropyLoss | AdamW | 3e-5 (视觉: 0.3e-5, 文本: 3e-5) | Warmup + 权重衰减 + 梯度裁剪 |

---

## 预期与实际结果分析

### 预期性能排序（基于模型表达能力）

根据模型架构复杂度和表达能力，预期性能排序应为：

1. **BLIP** - 预训练模型，拥有大规模数据先验
2. **Model2_Transformer** - Transformer架构，长依赖建模能力强
3. **Model1_YellowOrange** - 优化的注意力机制，训练稳定
4. **Original_Model** - 基础注意力，简单实现
5. **Model3_CNN_GRU** - 最简单结构，无空间注意力

### 实际结果分析

#### 实际评估得分

| 模型 | METEOR | ROUGE-L | CIDEr-D | 综合表现 |
|------|--------|---------|---------|---------|
| **Original_Model** | **0.5134** | 0.4720 | **0.7089** | ⭐⭐⭐⭐ |
| **Model1_YellowOrange** | 0.4960 | 0.4661 | 0.6956 | ⭐⭐⭐ |
| **Model2_Transformer** | 0.5009 | **0.5154** | **0.7391** | ⭐⭐⭐⭐⭐ |
| **Model3_CNN_GRU** | **0.5437** | 0.4645 | 0.5800 | ⭐⭐⭐ |
| **Ex1_BLIP** | 0.4209 | 0.4478 | 0.6076 | ⭐⭐ |

#### Model2_Transformer（综合表现最佳）

**预期**: 性能优秀，超越Model1  
**实际**: ✅ **综合表现最好**，ROUGE-L和CIDEr-D最高

**分析**:
- ✅ **ROUGE-L最高 (0.5154)**: Transformer的长依赖建模能力在流畅度上表现突出
- ✅ **CIDEr-D最高 (0.7391)**: 描述一致性最好，说明模型能够生成与参考描述风格一致的文本
- ✅ **METEOR良好 (0.5009)**: 语义匹配度也不错
- ✅ Transformer架构在小数据集上也能发挥优势，特别是在描述质量和流畅度上

**成功因素**:
- BERT集成提供了良好的语言模型先验
- Transformer的并行计算和长依赖建模能力
- Beam search (k=5) 提升了生成质量
- 合理的训练策略（label smoothing, warmup等）

#### Original_Model（表现优秀）

**预期**: 略低于Model1  
**实际**: ✅ **表现超出预期**，METEOR最高，CIDEr-D第二高

**分析**:
- ✅ **METEOR最高 (0.5134)**: 语义相似性最好，说明模型能准确理解图像语义
- ✅ **CIDEr-D第二高 (0.7089)**: 描述准确性很好
- ✅ 作为基线模型，表现非常优秀
- ⚠️ 只使用greedy生成，但效果仍然很好

**可能原因**:
- 简单而有效的架构设计
- 注意力机制能够有效关注图像关键区域
- 在小数据集上，简单结构训练稳定，不易过拟合

#### Model3_CNN_GRU（METEOR最高但CIDEr最低）

**预期**: 性能最低  
**实际**: ⚠️ **表现分化**，METEOR最高但CIDEr最低

**分析**:
- ✅ **METEOR最高 (0.5437)**: 意外表现，可能因为生成的描述在词级语义上匹配较好
- ❌ **CIDEr-D最低 (0.5800)**: 描述一致性差，说明生成的描述可能与参考描述在结构上差异较大
- ⚠️ **ROUGE-L较低 (0.4645)**: 流畅度一般
- ⚠️ 无注意力机制导致描述准确性不足，但可能在通用词汇匹配上表现较好

**可能原因**:
- METEOR基于词级匹配，对结构不敏感，可能误判
- CIDEr更关注描述的一致性，更能反映模型真实性能
- 全局特征缺乏空间信息，导致描述细节不足

#### Model1_YellowOrange（表现稳定但略低于预期）

**预期**: 中等偏上，稳定可靠  
**实际**: ⚠️ **表现略低于预期**，各项指标都在中等水平

**分析**:
- ⚠️ 所有指标都处于中等水平（0.46-0.69）
- ⚠️ 不如Original_Model和Model2表现好
- ⚠️ 尽管有beam search优化，但效果不如预期

**可能原因**:
- 虽然添加了beam search，但注意力机制实现可能与Original类似
- 训练可能未充分收敛或超参数需要进一步优化
- 需要更仔细的调优才能发挥其潜力

#### Ex1_BLIP（表现最低）

**预期**: 应该表现最好  
**实际**: ❌ **表现最低**，所有指标都低于其他模型

**关键问题**:
1. **预训练与目标域的差异**
   - BLIP在自然图像上预训练，生成简洁描述
   - 服饰数据集需要详细、结构化的描述
   - 描述风格不匹配导致性能下降

2. **微调策略问题**
   - 视觉编码器解冻层数（6层）可能过多，导致过拟合
   - 学习率（3e-5）可能不适合小数据集
   - 需要更保守的微调策略

3. **数据量问题**
   - 预训练模型通常需要更多数据才能充分微调
   - 小规模数据集可能无法有效利用预训练优势

**建议改进方向**:
- 减少视觉编码器解冻层数（1-2层）
- 使用更小的学习率（1e-5或5e-6）
- 增加正则化（权重衰减、dropout）
- 考虑使用zero-shot或few-shot学习
- 调整生成参数，优化推理质量

---

## 结果总结与原因推测

### 性能排序（实际）

根据综合评估（特别是CIDEr-D，最能反映描述质量）：

1. **Model2_Transformer** - 综合最佳，ROUGE-L和CIDEr-D最高 ⭐⭐⭐⭐⭐
2. **Original_Model** - 表现优秀，METEOR最高，CIDEr-D第二 ⭐⭐⭐⭐
3. **Model1_YellowOrange** - 稳定但略低于预期 ⭐⭐⭐
4. **Model3_CNN_GRU** - METEOR高但CIDEr最低，表现分化 ⭐⭐⭐
5. **Ex1_BLIP** - 表现最低，预训练域与目标域不匹配 ⭐⭐

### 核心发现

1. **Transformer架构在小数据集上也能表现出色**
   - Model2_Transformer综合表现最好，证明了Transformer的强大表达能力
   - 通过合理的训练策略（BERT集成、label smoothing、warmup），即使在小数据集上也能发挥优势
   - 长依赖建模能力在描述流畅度（ROUGE-L）上表现突出

2. **简单模型也能表现优秀**
   - Original_Model作为基线，表现超出预期，METEOR最高
   - 说明简单而有效的架构设计很重要
   - 在小数据集上，结构简单、训练稳定的模型往往能取得好效果

3. **预训练模型需要充分适配**
   - BLIP表现最低，证明了预训练模型不是万能的
   - 需要：
     - 合适的微调策略（层数、学习率）
     - 足够的训练数据
     - **目标域与预训练域的相似性（最关键）**
   - 描述风格不匹配是BLIP表现差的主要原因

4. **注意力机制的重要性**
   - 有注意力的模型（Original, Model1, Model2）综合表现更好
   - Model3虽然METEOR高，但CIDEr最低，说明缺少注意力导致描述准确性不足
   - 对于需要关注细节的图像描述任务，空间注意力至关重要

5. **评估指标需要综合看待**
   - METEOR可能在某些情况下不够准确（如Model3的METEOR最高但CIDEr最低）
   - CIDEr-D更能反映描述质量，是更可靠的指标
   - ROUGE-L反映流畅度，也很重要

### 改进建议

1. **对于Model1/Original**: 
   - 当前表现已经很好，可以尝试增加训练轮数或调整学习率

2. **对于Model2**: 
   - 增加训练数据量
   - 调整Transformer层数和dropout
   - 使用更强的正则化

3. **对于Model3**: 
   - 结构限制明显，建议放弃或仅作为基线

4. **对于BLIP**: 
   - 重新设计微调策略
   - 减少解冻层数，使用更保守的学习率
   - 考虑使用更大的预训练模型或不同的预训练权重

---

## 项目结构

```
nt/
├── README.md                          # 本文件
├── main.py                            # 统一CLI入口
├── preprocess.py                      # 数据预处理
├── metrics.py                         # 评估指标实现
├── utils_data.py                      # 数据加载工具
├── app_gradio.py                      # Gradio Web前端
├── requirements.txt                   # 依赖列表
│
├── Original_Model/                    # 基线模型
│   ├── README.md
│   ├── models.py
│   └── train_eval.py
│
├── Model1_YellowOrange/               # 改进基线模型
│   ├── README.md
│   ├── models.py
│   └── train_eval.py
│
├── Model2_Transformer/                # Transformer模型
│   ├── README.md
│   ├── model.py
│   └── train_eval.py
│
├── Model3_CNN_GRU/                    # CNN+GRU模型
│   ├── README.md
│   ├── models.py
│   └── train_eval.py
│
├── Ex1_BLIP/                          # BLIP预训练模型
│   ├── README.md
│   ├── train_blip.py
│   └── blip_infer.py
│
└── data/                              # 数据目录
    ├── images/                        # 图像文件
    ├── caption.json                   # 原始标注
    ├── train_captions.json            # 训练集标注
    ├── test_captions.json             # 测试集标注
    └── output/                        # 预处理输出和模型权重
        ├── vocab.json
        ├── encoded_captions_*.json
        ├── image_paths_*.json
        ├── caplens_*.json
        └── weights/                   # 模型权重
```

---

## 评估指标说明

项目实现了三个标准评估指标：

- **METEOR**: 基于精确匹配、同义词和词干的指标，考虑语义相似性
- **ROUGE-L**: 基于最长公共子序列的指标，评估流畅度
- **CIDEr-D**: 基于共识的指标，强调描述的一致性

所有模型在相同的数据集和评估设置下进行比较，确保公平对比。

---

## 贡献与扩展

项目支持进一步扩展：

- 添加新的模型架构
- 实现强化学习损失函数（SCST）
- 支持更多评估指标（如SPICE）
- 优化现有模型的超参数
- 增加数据增强策略

---

## 许可证

本项目仅用于学术研究和教育目的。
