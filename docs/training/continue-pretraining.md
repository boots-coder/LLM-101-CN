---
title: "继续预训练"
description: "领域自适应预训练——从通用模型到领域专家的关键步骤"
topics: [continue-pretraining, domain-adaptation, CPT, curriculum-learning, catastrophic-forgetting]
prereqs: [training/pretraining, training/sft]
---
# 继续预训练

> 通用大模型在专业领域的表现往往不尽如人意。继续预训练（Continue Pretraining, CPT）是将通用模型适配到特定领域的关键步骤——用领域语料在预训练好的模型上继续训练，让模型成为"领域专家"。

## 在大模型体系中的位置

```
预训练（通用语料）
   ↓
继续预训练（领域语料） ◄── 你在这里
   ↓
SFT 微调（领域指令数据）
   ↓
RLHF/DPO 对齐
   ↓
部署推理
```

继续预训练处于**预训练**和 **SFT** 之间。它不改变模型架构，只用领域数据延续预训练目标（Next Token Prediction），让模型内化领域知识。

---

## 为什么需要继续预训练

### 通用 LLM 在专业领域的局限

通用大模型（如 Llama、Qwen）在海量互联网语料上训练，具备广泛的语言能力。但在专业领域中常常遇到以下问题：

1. **领域术语覆盖不足**：医学术语（如"心房颤动消融术"）、法律条文编号在预训练语料中出现频率低，模型对这些术语的理解不够准确
2. **领域推理模式缺失**：法律推理的三段论、医学的鉴别诊断流程、代码的类型推导——这些模式需要大量领域文本才能学到
3. **知识时效性**：预训练数据有截止日期，无法覆盖新发布的法规、药物、API 等
4. **分词效率低**：通用 tokenizer 可能将领域术语切分为过多的子词，降低上下文利用效率

### 一个直观的例子

```
通用模型输入: "患者 INR 值为 4.5，目前服用华法林..."
通用模型输出: "建议调整用药"（模糊、缺乏专业性）

领域 CPT 后: "INR 4.5 显著超出治疗范围（通常 2.0-3.0），存在出血风险。
              建议暂停华法林 1-2 剂，48h 后复查 INR，
              同时排查饮食、药物相互作用等因素。"
```

---

## CPT vs SFT vs RAG：三种领域适配方案

| 特性 | 继续预训练 (CPT) | 监督微调 (SFT) | 检索增强生成 (RAG) |
|------|-----------------|---------------|-------------------|
| **数据形式** | 无标注领域文本 | 指令-回复对 | 文档知识库 |
| **数据量需求** | 大（数 B~数十 B tokens） | 中（数千~数万条） | 灵活（按需索引） |
| **改变的是** | 模型的内在知识 | 模型的行为模式 | 模型的输入上下文 |
| **训练成本** | 高 | 中 | 无需训练 |
| **知识更新** | 需要重新训练 | 需要重新训练 | 更新知识库即可 |
| **适用场景** | 领域知识密集型 | 格式和风格调整 | 知识频繁变动 |
| **可组合性** | CPT → SFT → RLHF | SFT 独立使用 | 与任何模型组合 |

::: tip 最佳实践
三者并非互斥。典型的领域大模型流水线为：**CPT**（注入领域知识）→ **SFT**（学会领域对话格式）→ **RAG**（补充实时信息）。
:::

---

## 实现：继续预训练的关键技术

### 数据准备

继续预训练的数据质量直接决定最终效果：

1. **领域语料收集**：学术论文、教材、行业报告、专业论坛等
2. **数据清洗**：去重、去噪、格式标准化
3. **通用语料混合**：为了缓解灾难性遗忘，通常将领域数据与通用数据按比例混合

```python
# 数据混合示例
domain_ratio = 0.7   # 70% 领域数据
general_ratio = 0.3  # 30% 通用数据（如 Wikipedia、Books）
```

**经验法则：** 领域语料与通用语料的比例通常在 7:3 到 9:1 之间。过高的领域比例会加剧遗忘，过低则领域适配效果有限。

### 学习率策略：低 LR Warmup

继续预训练不同于从零开始训练——模型已经有良好的参数分布，过大的学习率会破坏已有知识。

**关键原则：**
- 峰值学习率设为预训练学习率的 **1/10 到 1/5**
- 使用较长的 warmup 阶段（总步数的 5%~10%）
- 采用 Cosine Decay 或线性衰减

$$
\eta_{\text{CPT}} = \frac{\eta_{\text{pretrain}}}{10} \sim \frac{\eta_{\text{pretrain}}}{5}
$$

```python
# 学习率配置示例
pretrain_lr = 3e-4       # 原始预训练学习率
cpt_lr = 3e-5            # CPT 峰值学习率（1/10）
warmup_ratio = 0.05      # 5% 的步数用于 warmup
lr_scheduler = "cosine"  # 余弦衰减
```

### 灾难性遗忘的缓解

**灾难性遗忘（Catastrophic Forgetting）** 是继续预训练面临的最大挑战：模型在学习新领域知识的同时，可能丢失已有的通用能力。

**缓解策略：**

| 策略 | 做法 | 原理 |
|------|------|------|
| **数据混合（Data Replay）** | 训练数据中混入通用语料 | 持续"复习"通用知识 |
| **低学习率** | 使用较小的峰值 LR | 减少对已有参数的扰动 |
| **课程学习（Curriculum）** | 先通用后领域，逐步提高领域比例 | 平滑过渡，避免分布突变 |
| **正则化** | 对关键参数施加 L2 约束 | 限制参数偏移幅度 |
| **弹性权重巩固 (EWC)** | 保护对旧任务重要的参数 | 按 Fisher 信息加权 |

::: warning 灾难性遗忘的检测
在 CPT 过程中，务必**同时监控**领域评测和通用评测（如 MMLU、HellaSwag）。如果通用指标大幅下降，说明遗忘严重，需要调整数据比例或学习率。
:::

---

## 代码示例：基于 HuggingFace Trainer 的 CPT Pipeline

以下是一个完整的继续预训练代码框架：

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, concatenate_datasets

# ============================================================
# 1. 加载预训练模型和 tokenizer
# ============================================================
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # 使用 Flash Attention 加速
)

# 确保 tokenizer 有 pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 2. 准备领域数据 + 通用数据混合
# ============================================================
domain_dataset = load_dataset("json", data_files="domain_corpus.jsonl", split="train")
general_dataset = load_dataset("json", data_files="general_corpus.jsonl", split="train")

# 按 7:3 混合
general_sampled = general_dataset.shuffle(seed=42).select(
    range(int(len(domain_dataset) * 0.3 / 0.7))
)
combined_dataset = concatenate_datasets([domain_dataset, general_sampled]).shuffle(seed=42)

# Tokenize
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding=False,
    )

tokenized_dataset = combined_dataset.map(
    tokenize_fn, batched=True, remove_columns=combined_dataset.column_names
)

# ============================================================
# 3. 配置训练参数
# ============================================================
training_args = TrainingArguments(
    output_dir="./cpt-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,      # 有效 batch size = 4 * 8 = 32
    learning_rate=3e-5,                 # 预训练 LR 的 1/10
    warmup_ratio=0.05,                  # 5% warmup
    lr_scheduler_type="cosine",
    num_train_epochs=1,                 # CPT 通常只训练 1-2 epoch
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    gradient_checkpointing=True,        # 节省显存
    dataloader_num_workers=4,
    report_to="wandb",
)

# ============================================================
# 4. Data Collator：自动构建 Causal LM 的 labels
# ============================================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM，不是 Masked LM
)

# ============================================================
# 5. 启动训练
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./cpt-final")
tokenizer.save_pretrained("./cpt-final")
```

::: details 代码要点说明
- **Flash Attention 2**：`attn_implementation="flash_attention_2"` 可显著加速长序列训练
- **Gradient Checkpointing**：以计算换显存，对大模型 CPT 几乎是必需的
- **DataCollatorForLanguageModeling**：自动将 `input_ids` 右移构造 `labels`，无需手动处理
- **bf16**：BFloat16 混合精度训练，平衡速度和数值稳定性
- **1 epoch**：CPT 通常训练 1-2 个 epoch，多次重复会过拟合领域数据
:::

### 进阶：课程学习策略

更精细的做法是在训练过程中**动态调整领域数据的比例**：

```python
# 课程学习：领域数据比例从 50% 线性增长到 90%
# 阶段 1 (前 30% 步数): domain_ratio = 0.5
# 阶段 2 (30%-70%):      domain_ratio = 0.7
# 阶段 3 (后 30%):       domain_ratio = 0.9
```

这种渐进式方法可以有效缓解灾难性遗忘，让模型平滑过渡到领域知识。

---

## 领域案例

### 医学领域

- **典型语料**：PubMed 论文摘要、临床指南、病历文本（脱敏后）
- **数据规模**：10B~50B tokens
- **代表模型**：Med-PaLM 2、PMC-LLaMA、BioMistral
- **关键挑战**：医学术语的 tokenization 效率、隐私合规

### 法律领域

- **典型语料**：法律条文、判决书、法律评论、合同文本
- **数据规模**：5B~20B tokens
- **代表模型**：ChatLaw、LawGPT
- **关键挑战**：法律推理的严谨性、条文引用的准确性

### 代码领域

- **典型语料**：GitHub 代码、Stack Overflow、技术文档
- **数据规模**：100B+ tokens（代码语料丰富）
- **代表模型**：CodeLlama、StarCoder、DeepSeek-Coder
- **关键挑战**：多语言代码混合、上下文窗口需求大

::: tip 领域 CPT 的通用流程
1. 收集领域语料（10B+ tokens 为佳）
2. 清洗、去重、格式化
3. 与通用语料混合（7:3）
4. 低学习率 + warmup + cosine decay
5. 同时监控领域指标和通用指标
6. CPT 完成后接 SFT 激活对话能力
:::

---

## 苏格拉底时刻

1. **为什么 CPT 使用的学习率要远低于从零预训练？** 提示：模型参数已经处于一个"好的"损失盆地中，过大的学习率会将参数跳出这个盆地。

2. **数据混合比例如何确定？** 如果你有 10B 领域 token 和 100B 通用 token，你会如何设计混合策略？

3. **CPT 和 SFT 能否合并为一步？** 即直接用领域指令数据做微调，跳过 CPT。这样做有什么问题？提示：SFT 数据量通常远小于 CPT 数据量。

4. **灾难性遗忘的根本原因是什么？** 从优化的角度，为什么在新数据上训练会导致旧能力下降？

5. **如何判断 CPT 应该训练多少步？** 你会监控哪些指标来决定何时停止训练？

---

## 常见问题 & 面试考点

| 问题 | 要点 |
|------|------|
| CPT 和 SFT 的核心区别？ | CPT 用无标注文本做 Next Token Prediction，注入知识；SFT 用指令对做有监督训练，调整行为 |
| 为什么不直接用领域数据从零训练？ | 预训练的通用能力（语法、推理、世界知识）是宝贵的基础，从零训练浪费计算且难以恢复 |
| 灾难性遗忘如何缓解？ | 数据混合（replay）、低学习率、课程学习、EWC 正则化 |
| CPT 需要多少数据？ | 经验上至少需要 1B~10B tokens 的领域语料才能见到显著效果 |
| CPT 后还需要 SFT 吗？ | 需要。CPT 只注入知识，不改变模型的对话格式和指令跟随能力 |
| 如何评估 CPT 效果？ | 领域 perplexity、领域下游任务准确率、通用 benchmark 无明显回退 |

---

## 推荐资源

- [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964) — 领域自适应预训练的经典论文 (ACL 2020)
- [PMC-LLaMA: Towards Building Open-source Language Models for Medicine](https://arxiv.org/abs/2304.14454) — 医学领域 CPT 实践
- [CodeLlama: Open Foundation Models for Code](https://arxiv.org/abs/2308.12950) — 代码领域 CPT 的代表性工作
- [Continual Pre-training of Language Models: A Survey](https://arxiv.org/abs/2302.03241) — CPT 综述
- [HuggingFace Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) — Trainer API 官方文档
