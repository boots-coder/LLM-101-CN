---
title: "监督微调"
description: "全参数微调、LoRA 低秩适配、QLoRA 量化微调的原理与实现"
topics: [SFT, LoRA, QLoRA, PEFT, loss-masking, fine-tuning, TRL, SFTTrainer, packing, multi-turn, DeepSpeed, Flash-Attention]
prereqs: [training/pretraining]
---
# 监督微调 (SFT)

> SFT 将预训练模型变成能对话的助手，是 post-training 的第一步

## 在大模型体系中的位置

```
预训练 (Pre-training)          → 学习语言知识和世界知识
    ↓
监督微调 (SFT)  ← 你在这里     → 学习指令跟随能力
    ↓
偏好对齐 (RLHF/DPO/GRPO)      → 学习人类偏好，安全有用
```

预训练模型的目标是 **next-token prediction**，它学会了语言的统计规律，但不知道如何回答问题。SFT 的目标是将预训练模型从"续写文本"转变为"遵循指令"——通过在 `(指令, 回答)` 对上训练，模型学会了按照用户的意图生成有结构的回答，并在适当的时候停止输出。

## 什么是监督微调？

### 预训练模型的"复读机"问题

预训练模型直接用于对话时，往往会出现重复输出或不停止的情况：

```python
# 预训练模型的典型输出（未经 SFT）
input_ids = tokenizer(['请解释什么是机器学习'], return_tensors='pt')['input_ids']
output_ids = model.generate(input_ids, max_new_tokens=128, do_sample=False)
result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# 输出: "请解释什么是机器学习 什么是深度学习 什么是强化学习 什么是神经网络
#        什么是卷积神经网络 什么是循环神经网络 什么是注意力机制 什么是 Transformer..."
# → 不断重复、无法停止，这就是"鹦鹉学舌"现象
```

原因在于预训练数据不是每条都有 `<EOS>` 标记，模型没有学会在适当的时候停止生成（常被称为"复读机"现象）。

### SFT 的核心思路

SFT 仍然使用 **next-token prediction** 的训练目标，但有两个关键差异：

1. **训练数据是结构化的指令-回答对**，而非原始文本
2. **Loss Masking**：只对回答部分计算损失，不学习如何生成指令

$$
\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{response}} \log P_\theta(x_t | x_{<t})
$$

经过 SFT 后，模型能够正确回答问题并预测 `<EOS>` 停止：

```python
# SFT 后的模型输出
# 输出: "机器学习是人工智能的一个分支，它通过算法让计算机从数据中自动学习规律。
#        核心思想是：给模型大量样本数据，让它找到输入和输出之间的映射关系，
#        从而对新数据做出预测或决策。常见方法包括监督学习、无监督学习和强化学习。"
# → 有逻辑、有结构、能正常停止
```

## Full Fine-tuning

全参数微调（Full Fine-tuning）更新模型的**所有参数**，是最直接的微调方式。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载预训练模型，所有参数均可训练
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')

# 全参微调：所有参数都参与梯度更新
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
```

| 优点 | 缺点 |
|------|------|
| 效果最好，模型充分适配下游任务 | 显存占用大：7B 模型全参微调需要 ~60GB 显存 |
| 实现简单，无需额外框架 | 每个任务需要存储完整模型副本 |
| 适合有充足算力的场景 | 容易过拟合，尤其在小数据集上 |

## LoRA：低秩适配

### 核心原理

LoRA (Low-Rank Adaptation) 的核心假设是：**微调过程中权重的更新矩阵是低秩的**。

对于预训练权重 $W_0 \in \mathbb{R}^{d \times d}$，LoRA 将更新分解为两个低秩矩阵的乘积：

$$
W = W_0 + \Delta W = W_0 + \alpha \cdot W_B W_A
$$

其中 $W_A \in \mathbb{R}^{d \times r}$，$W_B \in \mathbb{R}^{r \times d}$，$r \ll d$ 是秩。

**数学直觉**：一个 $4096 \times 4096$ 的矩阵有 1600 万参数，但如果 $r=16$，LoRA 只需 $4096 \times 16 \times 2 = 131072$ 个参数，减少了 **99.2%**。

### PyTorch 实现 LoRA

```python
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """LoRA 线性层：冻结原 Linear，旁路注入 (alpha/r) · B A x"""
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r

        d_out, d_in = base.weight.shape
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x):
        delta = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return self.base(x) + delta
```

### 替换模型中的 Linear 层

```python
def inject_lora(model, target_modules, r=16, alpha=32):
    """递归遍历，把 target_modules 命名的 nn.Linear 替换为 LoRALinear"""
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and name in target_modules:
            setattr(model, name, LoRALinear(child, r=r, alpha=alpha))
        else:
            inject_lora(child, target_modules, r, alpha)
    return model

inject_lora(model, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable: {trainable / total:.2%}")
```

### 关键参数详解

| 参数 | 含义 | 经验值 |
|------|------|--------|
| `r` (rank) | 低秩维度，控制旁路容量 | 8~64，推荐 16 |
| `alpha` | 缩放因子，控制旁路影响强度 | 通常设为 `r` 或 `2*r` |
| `target_modules` | 要加 LoRA 的层 | `["q_proj", "k_proj", "v_proj", "o_proj"]` |
| `lora_dropout` | LoRA 旁路的 Dropout | 0.05~0.1 |

::: tip Raschka 实测要点（数百次 LoRA 实验后总结）
- **LoRA 要加在所有 linear 层，而非只 Q/K**：原始论文为简化只加 Q/K，实测把 Q/K/V/o_proj/MLP/lm_head 全开后 TruthfulQA 从 0.282 涨到 0.302，可训参数 4.2M → 20.3M（5×），显存只多 ~2.4GB。性价比很高，应作为默认起点。
- **α 不一定是 2r**：小 r 时 `α=2r` 是不错的起点，但在 r=256 这种大秩下 `α=128`（即 α/r=0.5）反而更优——等价于把 scaling 调小避免训过头。
- **rsLoRA**：把 LoRA scaling 从 `α/r` 换成 `α/√r`，在大 rank 下更稳，已被 Unsloth 默认启用。
- **可复现性**：同配置跑多次得分差异 < 0.005，可放心用 LoRA 做 ablation 实验。
:::

### 使用 PEFT 库（推荐生产用法）

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,                # 论文推荐 α = 2r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
# 输出: trainable params: 1,310,720 || all params: 630,000,000 || trainable%: 0.21%
```

## QLoRA：量化 + LoRA

QLoRA 在 LoRA 基础上对原始参数进行**量化**，进一步降低显存占用。核心思想是：既然原始参数是冻结的，那就把它压缩存储，计算时再解压。

### 三大技术

1. **4-bit NormalFloat (NF4) 量化**：利用预训练权重近似正态分布的特点，用 4-bit 量化替代 FP16 存储
2. **双重量化 (Double Quantization)**：对量化的缩放因子再做一次量化，进一步节省显存
3. **分页优化器 (Paged Optimizers)**：利用 CPU 内存处理显存不足时的梯度状态

### 实现 QLoRA 核心：量化线性层

```python
import bitsandbytes as bnb

class QuantizedLinear(nn.Module):
    """INT8 量化的线性层，前向时动态去量化"""
    def __init__(self, module, dim_in, dim_out, block_size=64):
        super().__init__()
        self.in_features = dim_in
        self.out_features = dim_out
        self.block_size = block_size

        # 量化原始权重
        w = module.weight.data
        quant_weight, quant_state = bnb.functional.quantize_blockwise(
            w, blocksize=self.block_size
        )
        self.quant_weight = quant_weight    # int8 类型，显存减半
        self.weight_scale = quant_state     # 量化缩放因子
        self.weight = None                  # 删除原始参数节省显存
        self.quantized = True

    def forward(self, x):
        # 计算时动态反量化回 fp16
        dequant_weight = bnb.functional.dequantize_blockwise(
            self.quant_weight, self.weight_scale, blocksize=self.block_size
        )
        return nn.functional.linear(x, dequant_weight, None)
```

### QLoRA = 量化 + LoRA

```python
# 先量化原始 Linear，再注入 LoRA 旁路
quantize_linear_modules(model, QuantizedLinear, block_size=64)
inject_lora(model, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

# 前向时：量化权重无梯度，lora_A / lora_B 有梯度
loss = model(X).loss
loss.backward()
```

### Transformers 快捷调用

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 4-bit 量化
    bnb_4bit_quant_type="nf4",            # NormalFloat4 量化类型
    bnb_4bit_use_double_quant=True,       # 双重量化
    bnb_4bit_compute_dtype=torch.bfloat16 # 计算精度
)
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-0.6B', quantization_config=bnb_config
)
# 1B 模型 fp16 占 ~2GB 显存，4-bit 只占 ~0.5GB
```

::: tip QLoRA 实测代价与单卡基准
- **时间换显存**：相对 16-bit LoRA，QLoRA 显存节省 33%（21.33GB → 14.18GB）但训练时间增加 39%（1.85h → 2.79h）；benchmark 几乎不掉点。在单卡 7B+ 场景里，QLoRA 往往是唯一可行选项。
- **7B 单卡参考配置**：A100 40GB 上 `r=256, α=512`、Q/K/V/o_proj/MLP/lm_head 全开，**17.86GB / 3 小时**跑完 50k Alpaca。
- **优化器**：QLoRA 配 [`adamw_8bit`](https://github.com/TimDettmers/bitsandbytes) 比 32-bit AdamW 显存更省、精度等同，是 Unsloth / Axolotl 的默认选择。
:::

## SFT 数据集准备

### 数据格式对比

SFT 数据通常有三种格式：

| 格式 | 结构 | 示例数据集 |
|------|------|-----------|
| **Alpaca** | `instruction + input + output` | tatsu-lab/alpaca |
| **ShareGPT** | 多轮 `conversations` 列表 | ShareGPT 系列 |
| **OpenAI/ChatML** | `messages` 列表 with `role` | OpenAI API 格式 |

::: warning SFT 的能力边界
SFT **只能"激活"模型已有的知识**，无法注入全新语言或领域知识——硬上会让模型在没把握时编造（幻觉）。补全新知识应该走 [继续预训练](continue-pretraining.md) 而不是 SFT。

高质量 SFT 数据集示例：[mlabonne/FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) 用 [HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) 重过滤 [arcee-ai/The-Tome](https://huggingface.co/datasets/arcee-ai/The-Tome) 得到，含推理、function calling 题型，是单卡 QLoRA 入门的常用起点。
:::

### Chat Template 与 Label Masking（核心思想）

SFT 的工程实现有两条路：

1. **直接用 tokenizer 内置 chat template**：HuggingFace tokenizer 的 `apply_chat_template(messages, tokenize=False)` 会按模型自带模板（ChatML / Llama-3 / Qwen 等）渲染成字符串，再 tokenize 即可得到训练样本。
2. **Loss 只在 assistant 段计算**：构造与 `input_ids` 等长的 `labels`，把 system/user 区段填成 `-100`，配合 `CrossEntropyLoss(ignore_index=-100)` 即可自动跳过。

> 工程上不建议手写 chat template + collate，统一交给下文的 `TRL SFTTrainer`（自动处理模板、label masking、packing），仅在做教学拆解时才手写。下面这一节就是这种"教学拆解"——把 label masking 从概念跑成 loss 曲线，让你亲眼看到训练前后的输出差异。

## 最小可跑 SFT：笔记本本地 < 1 分钟

::: tip 这一节解决什么问题
上面只讲了 label masking 的概念，但**真要把数据喂给模型、看到 loss 下降、对比训练前后输出**，还得跑一遍。下面这段 ~70 行的 PyTorch 代码用 [distilgpt2](https://huggingface.co/distilbert/distilgpt2)（~82M 参数）+ **8 条手写指令对 × 4 = 32 条样本**，CPU 上 30-60 秒、GPU 上 5-10 秒就能看到效果，不依赖 TRL / PEFT，也不需要下载任何数据集。
:::

### Step 1：准备模型与微型数据集

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PAIRS = [
    {"q": "What is 2+3?",                 "a": "2+3 equals 5."},
    {"q": "Capital of France?",           "a": "The capital of France is Paris."},
    {"q": "Define entropy.",              "a": "Entropy measures uncertainty of a random variable."},
    {"q": "What is gradient descent?",    "a": "An optimization algorithm that moves parameters opposite to the gradient."},
    {"q": "Translate 'hello' to French.", "a": "'hello' translates to 'bonjour'."},
    {"q": "What is overfitting?",         "a": "When a model learns training data too well and fails to generalize."},
    {"q": "What is a tensor?",            "a": "A multi-dimensional array used in machine learning."},
    {"q": "What is backpropagation?",     "a": "An algorithm to compute gradients via the chain rule."},
] * 4  # 32 条，足够 80 步训练

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

### Step 2：Tokenize 并对 prompt 段做 label masking

label masking 的核心就是这一行：`labels[:prompt_len] = -100`。HuggingFace causal LM 在内部按 `ignore_index=-100` 跳过这些位置——这正是 SFT 与原始 next-token 预训练的根本分水岭。

```python
TEMPLATE = "Q: {q}\nA: {a}"

def encode(pair, max_len=128):
    prompt = f"Q: {pair['q']}\nA: "
    full = TEMPLATE.format(**pair) + tokenizer.eos_token
    full_ids = tokenizer(full, truncation=True, max_length=max_len)["input_ids"]
    prompt_len = len(tokenizer(prompt)["input_ids"])

    labels = list(full_ids)
    labels[:prompt_len] = [-100] * prompt_len   # 关键：prompt 段不算 loss
    return full_ids, labels

samples = [encode(p) for p in PAIRS]
ids, lbl = samples[0]
print("input_ids[:6] =", ids[:6])
print("labels[:6]    =", lbl[:6])  # 前几位应为 -100，从 A: 之后才是真实 token id
```

### Step 3：Collate + 单步 loss 验证

把变长样本 padding 成同一矩阵，padding 位的 label 也填 `-100`（同一个 ignore_index 同时处理 padding 与 prompt mask 两种情况）。

```python
def collate(batch):
    max_len = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    labels    = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (ids, lbl) in enumerate(batch):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        labels   [i, :len(lbl)] = torch.tensor(lbl)
    return input_ids.to(device), labels.to(device)

ids, lbl = collate(samples[:2])
loss = model(input_ids=ids, labels=lbl).loss
print(f"initial loss = {loss.item():.4f}")  # 通常在 4~6 之间
```

### Step 4：训练循环

```python
import random
random.seed(0)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()

for step in range(80):
    batch = random.sample(samples, k=4)
    ids, lbl = collate(batch)
    loss = model(input_ids=ids, labels=lbl).loss
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 20 == 0:
        print(f"step {step:3d} | loss = {loss.item():.4f}")
```

CPU 每步约 0.4-0.8s，80 步合计 30-60s；GPU 上 80 步 < 10s。loss 通常会从 4-5 降到 1 附近。

### Step 5：训练前后输出对比

为了对比，重新加载一份未训练的 base 模型一起生成：

```python
def generate(m, q, max_new=40):
    inputs = tokenizer(f"Q: {q}\nA:", return_tensors="pt").to(device)
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                         eos_token_id=tokenizer.eos_token_id,
                         pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

base = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device).eval()
for q in ["What is overfitting?", "What is backpropagation?"]:
    print(f"\nQ: {q}")
    print("Base:", generate(base, q))
    print("SFT :", generate(model.eval(), q))
```

期望现象：**Base** 会接着 prompt 一路续写、混入新的"What is..."风格提问、不会停止；**SFT** 训练后能短句直接作答并触发 EOS 提前停止——这就是 label masking 真正起作用的视觉证据。

::: warning 这只是教学样例
80 步、32 条数据、distilgpt2 远不够生产用——它的唯一目的是把"label mask → cross_entropy → backward → 输出风格变化"这条数据流跑通让你看见。真要训出能用的助手，请直接跳到下文 §TRL SFTTrainer 实战，几十万条数据 + LoRA + packing + 多卡才是实际配方。
:::

## 训练参数调优指南

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 学习率 | 1e-5 ~ 5e-5 | 比预训练低 1-2 个数量级 |
| Batch Size | 8~32 | 配合梯度累积使用 |
| Epochs | 1~3 | 过多会过拟合 |
| Warmup | 总步数的 3%~10% | 稳定训练初期 |
| 序列长度 | 512~2048 | 根据数据分布决定 |
| 梯度裁剪 | 1.0 | 防止梯度爆炸 |

**数据质量 > 数据数量**：LIMA 论文证明，1000 条精心标注的高质量数据优于 10 万条低质量数据。

::: tip 经验值背后的实测
- **Epoch 选 1 还是 3？** Raschka 在 Alpaca 上发现 2 epoch 比 1 epoch 在 TruthfulQA 上**反而更差**，1k 条 LIMA 数据集观察到同样退化——疑似过拟合训练集分布。**优先选 1 epoch**，只有 base 模型与目标任务差距很大时才提到 2-3。
- **优化器差异极小**：AdamW / SGD / 是否加 cosine scheduler 在 LoRA 微调中差异都不显著；显存方面，r 较小（如 r=8 时 7B Llama2 只 4.19M 可训参数）时 Adam 比 SGD 只多 ~30MB，r=256 时差距才拉开（17.86GB vs 14.46GB）。
- **Labonne 推荐起手参考配置**（Llama 3.1 8B + FineTome-100k）：`lr=3e-4, scheduler=linear, batch=8 × grad_accum=2, epochs=1, optim=adamw_8bit, weight_decay=0.01, warmup_steps=10, packing=True`。
:::

## TRL SFTTrainer 实战

[TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) 提供了开箱即用的 `SFTTrainer`，是目前最流行的 SFT 训练工具之一。它封装了 Hugging Face Trainer，并内置了 chat template 处理、packing、loss masking 等 SFT 专属功能。

### 基本用法

```python
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir="sft-model",
    max_seq_length=2048,
    packing=True,          # 多条短样本打包为一条长序列
    num_train_epochs=3,
)
trainer = SFTTrainer(
    model="Qwen/Qwen2-0.5B",
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

::: tip 为什么推荐 SFTTrainer？
相比手写训练循环，SFTTrainer 自动处理了 chat template 转换、loss masking、序列打包等繁琐细节，让你专注于数据质量和超参调优。
:::

### Chat Template 处理

TRL 自动调用模型 tokenizer 的 chat template 将对话格式转换为模型输入：

- 数据集格式：`{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
- 自动添加 role tokens（如 `<|im_start|>user`）
- 自动对 assistant 部分做 loss masking，user 和 system 部分不参与损失计算

### 数据格式详解

SFTTrainer 支持三种数据格式：

| 格式 | 数据列 | 适用场景 |
|------|--------|----------|
| Conversational | `messages` 列（标准 chat 格式） | 多轮对话、ChatBot |
| Instruction | `prompt` + `completion` 列 | 单轮指令跟随 |
| Language modeling | 纯 `text` 列 | 续写式微调 |

::: details Conversational 格式示例
```python
dataset = [
    {"messages": [
        {"role": "user", "content": "解释什么是梯度下降"},
        {"role": "assistant", "content": "梯度下降是一种优化算法..."},
    ]}
]
```
:::

### Packing（序列打包）

Packing 是提升 SFT 训练效率的关键技巧：

- 将多条短训练样本打包成一条长序列，**提高 GPU 利用率**
- 使用 attention mask 防止跨样本注意力泄露，保证训练正确性
- 设置 `packing=True` 即可启用
- 特别适合训练数据长度差异大的场景

```
不使用 Packing:
[样本1--padding--padding] [样本2--------padding] [样本3-padding-padding]
  ↑ GPU 大量时间在处理 padding token

使用 Packing:
[样本1--样本2--------样本3] [样本4------样本5------]
  ↑ GPU 利用率大幅提升
```

::: warning
Packing 会改变每个 batch 的有效样本数，可能需要相应调整学习率。
:::

### LoRA 集成

SFTTrainer 原生支持 PEFT，只需传入 `peft_config` 即可启用 LoRA 训练：

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
trainer = SFTTrainer(
    model="meta-llama/Llama-3-8B",
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,  # 直接传入LoRA配置
)
```

这种方式将 SFT 训练和 LoRA 微调无缝结合，无需手动修改模型结构。

## 多轮对话训练

实际场景中，用户与模型的交互往往是多轮对话。多轮对话训练需要特别注意 loss 计算和数据格式。

### 数据格式

```python
# 多轮对话数据格式
dataset = [
    {"messages": [
        {"role": "system", "content": "你是一个有帮助的助手"},
        {"role": "user", "content": "什么是 Transformer？"},
        {"role": "assistant", "content": "Transformer 是一种基于自注意力的..."},
        {"role": "user", "content": "它的核心创新是什么？"},
        {"role": "assistant", "content": "核心创新是自注意力机制..."},
    ]}
]
```

### 训练要点

- **Loss Masking**：每轮只对 assistant 回复计算 loss，system 和 user 部分被 mask 掉
- **Chat Template**：自动添加 role tokens，正确分隔每轮对话
- **Truncation 策略**：长对话可能超过 `max_seq_length`，需要合理截断

::: tip 长对话截断建议
优先保留最后几轮对话（最相关的上下文），截断最早的轮次。也可以将长对话拆分为多条训练样本，每条保留完整的上下文窗口。
:::

```
多轮对话 Loss Masking 示意:

[system] 你是一个助手  [user] 什么是 Transformer？ [assistant] Transformer 是...
 ↑ 不计算 loss          ↑ 不计算 loss              ↑ 计算 loss ✓

[user] 核心创新？ [assistant] 核心创新是自注意力...
 ↑ 不计算 loss     ↑ 计算 loss ✓
```

## 训练加速与显存优化

当模型规模增大或数据量增长时，训练效率和显存管理成为关键瓶颈。以下是常用的优化手段：

### Gradient Checkpointing

以计算换显存——前向传播时不保存中间激活值，反向传播时重新计算：

```python
training_args = SFTConfig(
    gradient_checkpointing=True,
    # ... 其他参数
)
```

通常可节省 **50%-70%** 显存，代价是训练速度下降约 20%-30%。

### Mixed Precision 混合精度训练

使用 bf16 或 fp16 替代 fp32，显存减半、计算加速：

```python
training_args = SFTConfig(
    bf16=True,   # 推荐在 A100/H100 上使用 bf16
    # fp16=True, # 旧款 GPU 可用 fp16
)
```

::: tip bf16 vs fp16
bf16 的动态范围与 fp32 相同，不容易出现溢出，是目前主流选择。fp16 需要配合 loss scaling 使用。
:::

### DeepSpeed ZeRO

ZeRO 将优化器状态、梯度、模型参数分阶段切分到多张 GPU 上：

| 阶段 | 切分内容 | 显存节省 |
|------|----------|----------|
| ZeRO-1 | Optimizer States | ~4x |
| ZeRO-2 | + Gradients | ~8x |
| ZeRO-3 | + Parameters | ~Nx（N 为 GPU 数） |

```bash
# 使用 DeepSpeed ZeRO-2 启动训练
deepspeed --num_gpus=4 train.py --deepspeed ds_config_zero2.json
```

### Gradient Accumulation

小批次累积模拟大 batch，在不增加显存的前提下提升有效 batch size：

```python
training_args = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # 有效 batch size = 2 × 4 = 8
)
```

### Flash Attention

Flash Attention 通过 IO 感知的分块计算，将 attention 的显存复杂度从 $O(n^2)$ 降至 $O(n)$：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    attn_implementation="flash_attention_2",
)
```

::: tip Unsloth：单卡场景的"全家桶式加速"
[Unsloth](https://github.com/unslothai/unsloth) 用自定义 Triton kernel 重写了 attention / MLP / RoPE / Cross-Entropy，在**单卡**上实现 ~2× 加速 + 60% 显存节省（仅支持单卡；多卡走 [TRL](https://github.com/huggingface/trl) 或 [Axolotl](https://github.com/axolotl-ai-cloud/axolotl)，二者已把 Unsloth 作为后端集成）。

QLoRA 训练 100k samples on Llama 3.1 8B 的实测时长基准：A100 40GB ~ **4h45m**，L4 24GB ~ **19h40m**，免费 Colab T4 ~ **47h**。显存吃紧可用 `dataset.select(range(10000))` 取子集快速跑通 pipeline。
:::

### vLLM 集成

TRL 支持使用 vLLM 做生成加速，主要用于需要在线生成的训练算法（如 GRPO、PPO）。vLLM 的 PagedAttention 和 continuous batching 可以大幅加速生成过程。

::: details 各优化技术对比总结
| 技术 | 节省显存 | 速度影响 | 适用场景 |
|------|----------|----------|----------|
| Gradient Checkpointing | ✅ 大幅 | ⬇ 略慢 | 单卡大模型 |
| Mixed Precision (bf16) | ✅ 减半 | ⬆ 加速 | 所有场景 |
| DeepSpeed ZeRO | ✅ 线性扩展 | ➡ 通信开销 | 多卡训练 |
| Gradient Accumulation | ❌ 不变 | ➡ 不变 | 模拟大 batch |
| Flash Attention | ✅ 显著 | ⬆ 加速 | 长序列训练 |
| Packing | ❌ 不变 | ⬆ 加速 | 短样本数据集 |
:::

## 苏格拉底时刻 💡

1. **LoRA 的低秩假设为什么成立？** 微调只是让模型适配新任务，权重变化集中在少数方向上，不需要改变全部参数空间。
2. **SFT 为什么只需 1-3 epochs？** 预训练模型已经学会了语言能力，SFT 只是"教它怎么用"，少量训练即可。过多训练反而会导致过拟合——模型记住了训练数据的模式而非泛化能力。
3. **Loss Masking 为什么重要？** 如果不做 masking，模型会学习生成用户提问的模式，导致输出中混入提问语气的内容。
4. **LoRA 的 rank 选多少？** rank=8 适合简单任务（格式、风格），rank=64 适合复杂任务（新知识、新领域）。过高的 rank 接近全参微调，失去了效率优势。

## 常见问题 & 面试考点

- **Q: LoRA 和 Full Fine-tuning 效果差多少？** 在大多数任务上差距很小（<2%），但在需要大量新知识注入的场景下 Full FT 更优。
- **Q: QLoRA 精度如何？** 量化会引入精度损失，一般比 LoRA 略差。在显存受限时是性价比最高的选择。
- **Q: SFT 数据中的 system prompt 有什么用？** 定义模型的角色和行为边界，是控制模型行为的重要手段。
- **Q: 为什么 SFT 后模型的通用能力可能下降？** 这被称为"对齐税"（Alignment Tax），过度 SFT 会让模型过度拟合特定格式，损失预训练阶段学到的通用能力。

## 推荐资源

### 论文

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by Hu et al. — LoRA 原始论文，提出 ΔW = BA 低秩分解的核心假设。
- [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314) by Dettmers et al. — 4-bit NF4 量化 + Double Quantization + Paged Optimizers，把 65B 微调塞进单卡 48GB。
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — 用 1k 条精选 SFT 数据胜过 50k Alpaca，强调数据质量优先。

### 博客与实证研究

> 这两篇博客的具体结论已经被拆进 §LoRA 关键参数、§QLoRA、§SFT 数据集准备、§训练参数调优指南、§训练加速 几个章节内的 `::: tip` 块，按位嵌入语义最相关的位置。下面只列出原文与可对照的 notebook，需要完整 ablation 数据时请回到原文通读。

- [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) by Sebastian Raschka — 在 Lit-GPT 上跑了数百次 LoRA 微调实验后总结的 FAQ 式经验贴，含 7B Llama2 在不同 r / α / target_modules / 优化器 / epoch 下的完整对比表。
- [Fine-tune Llama 3.1 Ultra-Efficiently with Unsloth](https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html) by Maxime Labonne — 配套 [Colab notebook](https://colab.research.google.com/drive/164cg_O7SV7G8kZr_JXqLd6VC7pd86-1Z?usp=sharing) 的 Llama 3.1 8B QLoRA 端到端实操：[FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) 数据 → ChatML 模板 → rsLoRA + adamw_8bit 训练 → `merged_16bit` / `merged_4bit` 保存 → GGUF 多精度导出（q2_k…q8_0）→ 推 HF Hub，是把本页代码段连接到可部署模型的最短路径。

### 代码参考

- [microsoft/LoRA](https://github.com/microsoft/LoRA)（原始作者实现）：
  - [loralib/layers.py L12-L30](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py#L12-L30) — `LoRALayer` 基类，包含 `r / lora_alpha / lora_dropout / merge_weights` 四个核心字段。
  - [loralib/layers.py L90-L235](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py#L90-L235) — `Linear` 实现：`lora_A` 用 Kaiming 初始化、`lora_B` 用零初始化（保证训练开始时 BA = 0，等价于原模型）；`scaling = lora_alpha / r`；`train()` 时把已合并的权重还原回去、`eval()` 时把 BA 合并进 `weight` 提升推理速度。
  - [loralib/layers.py L32-L88](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py#L32-L88) — `Embedding` 的 LoRA 适配版（embedding 层的低秩注入与 Linear 略有不同，可学习参数顺序是 `lora_A` 在前、`lora_B` 在后）。
- [huggingface/peft](https://github.com/huggingface/peft)（生产级 PEFT 框架）：
  - [src/peft/tuners/lora/layer.py](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py) — 通用 LoRA Layer，支持 `merge / unmerge / multi-adapter / DoRA / Rank-Stabilized LoRA`。
  - [src/peft/tuners/lora/bnb.py](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/bnb.py) — bitsandbytes 4-bit/8-bit + LoRA（即 QLoRA 的 PEFT 实现），对照本页 §QLoRA 阅读最直观。
  - [src/peft/tuners/lora/dora.py](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/dora.py) — DoRA（Weight-Decomposed LoRA）实现，把 ΔW 进一步拆成 magnitude + direction 两个部分。
  - [src/peft/tuners/lora/config.py](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py) — `LoraConfig` 全字段定义，可与本页代码示例的 `LoraConfig(...)` 对照。
- [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — 一站式 SFT/DPO/PPO/LoRA/QLoRA 训练框架，配置文件驱动，适合快速跑出可用模型。详见下文「生产级对照」小节。
- [huggingface/trl](https://github.com/huggingface/trl) — `SFTTrainer` / `DPOTrainer` / `GRPOTrainer` 等 Trainer 工具箱。
- [unslothai/unsloth](https://github.com/unslothai/unsloth) — 单卡 2× 加速 + 60% 显存节省的微调框架，自定义 Triton kernel 重写注意力/MLP/RoPE/Cross Entropy。配套 Colab：[Llama 3.1 8B QLoRA Notebook](https://colab.research.google.com/drive/164cg_O7SV7G8kZr_JXqLd6VC7pd86-1Z?usp=sharing)，是从 0 到"GGUF 模型推到 HF Hub"最短路径。
- [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl) — YAML 配置驱动的多卡微调框架，内置 Unsloth 后端，适合扩到多卡场景。
- [rasbt/LLMs-from-scratch · ch07](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07) — Sebastian Raschka《Build a Large Language Model (From Scratch)》第 7 章配套代码：从一个预训练 GPT-2 出发，亲手实现"指令数据格式化 → mask 掉 prompt 段的 loss → 训练 → 用 Ollama 本地 LLM-as-judge 评测"全流程，是把本页 §SFT 训练流程跟可运行代码对应起来的最干净参考。
  - [ch07/01_main-chapter-code/gpt_instruction_finetuning.py](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/gpt_instruction_finetuning.py)（348 行） — `format_input`(L113-L122) 把样本拼成 Alpaca 模板、`InstructionDataset`(L35-L53) 预先把 instruction+input+response 编码成单条 token 序列、`custom_collate_fn`(L56-L96) 做 padding 并把"除第一个 `<|endoftext|>` 之外的 padding 位"替换成 `ignore_index=-100`，再配 `train_model_simple` 跑完整 SFT 循环。注意：**主章节这份只 mask 了 padding**，并没有对 prompt 段做 loss mask——这就是下面 `exercise_experiments.py` 要补的关键差距。
  - [ch07/01_main-chapter-code/exercise_experiments.py](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/exercise_experiments.py) — `InstructionDatasetWithMasking`(L58-L84) 在 `__init__` 里多记一份 `instruction_lengths`，再由 `custom_collate_with_masking_fn`(L190-L233) 把 prompt 段 label 一刀切成 `-100`，**只在 response token 上算 loss**。这正是本页 §Loss Masking 一节的精确实现，原文（L218-L219）：
    ```python
    # New: Mask all input and instruction tokens in the targets
    targets[:instruction_length-1] = -100
    ```
    这一行是 SFT 与预训练 next-token loss 的根本分水岭——不加它，模型在训练中会被迫学习"如何生成用户提问"，导致输出里混入提问语气；加上它，模型只在"该回答的位置"承担 loss。配合 `pad_token_id` 的 `ignore_index` 处理（L213-L216），构成 ignore_index 的两种使用场景：padding（节省算力）与 prompt mask（决定学习目标）。
  - [ch07/01_main-chapter-code/ollama_evaluate.py](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ollama_evaluate.py)（119 行） — 用本地跑的 Llama 3 当 judge 给每条回复打 0-100 分，是免 OpenAI API 跑 LLM-as-judge 评测的最小可用模板。
  - [ch07/04_preference-tuning-with-dpo](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07/04_preference-tuning-with-dpo) / [05_dataset-generation](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07/05_dataset-generation) — DPO 偏好对齐与合成指令数据生成的小型示例，可作为从 SFT 进入 [alignment.md](alignment.md) 的衔接代码。
- [rasbt/LLMs-from-scratch · appendix-E](https://github.com/rasbt/LLMs-from-scratch/tree/main/appendix-E) — 附录 E 用一份独立 notebook [appendix-E.ipynb](https://github.com/rasbt/LLMs-from-scratch/blob/main/appendix-E/01_main-chapter-code/appendix-E.ipynb) 从零实现 LoRA 并接到第 6 章的分类微调上，没有任何 PEFT / bitsandbytes 依赖，是想看清"LoRA 到底怎么挂到 nn.Linear 上"时最薄的一份代码。

### 生产级对照：LLaMA-Factory（封装层，非手撕）

::: warning 阅读定位
本节**不是**手撕教学对象——LLaMA-Factory 的训练 trainer 直接继承自 HuggingFace `Seq2SeqTrainer` / `trl.PPOTrainer`，本质是**配置 + 编排层**。把它列在这里，是作为"真实工业链路长什么样"的参照系，便于读者从 §TRL SFTTrainer 实战进一步衔接到生产部署。
:::

[hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)（配套论文 [arXiv 2403.13372](https://arxiv.org/abs/2403.13372) "LLaMA Factory: Unified Efficient Fine-Tuning of 100+ Language Models"）是一套**零代码、配置文件驱动**的全家桶，覆盖六种训练范式，目录与本页知识点一一对应：

| 范式 | 仓库目录 | 对应本页/姊妹页 |
|------|---------|----------------|
| Pre-Train | [`src/llamafactory/train/pt/`](https://github.com/hiyouga/LLaMA-Factory/tree/main/src/llamafactory/train/pt) | [pretraining.md](pretraining.md) |
| SFT | [`src/llamafactory/train/sft/`](https://github.com/hiyouga/LLaMA-Factory/tree/main/src/llamafactory/train/sft) | 本页 |
| Reward Modeling | [`src/llamafactory/train/rm/`](https://github.com/hiyouga/LLaMA-Factory/tree/main/src/llamafactory/train/rm) | [alignment.md](alignment.md) |
| DPO | [`src/llamafactory/train/dpo/`](https://github.com/hiyouga/LLaMA-Factory/tree/main/src/llamafactory/train/dpo) | [alignment.md](alignment.md) |
| KTO | [`src/llamafactory/train/kto/`](https://github.com/hiyouga/LLaMA-Factory/tree/main/src/llamafactory/train/kto) | [alignment.md](alignment.md) |
| PPO | [`src/llamafactory/train/ppo/`](https://github.com/hiyouga/LLaMA-Factory/tree/main/src/llamafactory/train/ppo) | [alignment.md](alignment.md) |

**封装方式（必看，避免迷信）**：
[`train/sft/trainer.py L47`](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/train/sft/trainer.py#L47) 真身只有一行：

```python
from transformers import Seq2SeqTrainer
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    """Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""
```

[`train/ppo/trainer.py L35`](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/train/ppo/trainer.py#L35) 同样直接 `from trl import PPOConfig, PPOTrainer` 复用 TRL 实现。所以"读 LLaMA-Factory 学算法"是路径错的——算法都在 transformers / trl 里。

**一行 CLI 跑 LoRA SFT**（[examples/train_lora/qwen3_lora_sft.yaml](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/train_lora/qwen3_lora_sft.yaml)，Qwen3-4B + r=8 + 3 epoch）：

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

CLI 入口由 [`cli.py`](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/cli.py) 暴露 `train / chat / api / export / webchat / webui / env / version` 八个子命令，`webui` 启动的就是 LlamaBoard 可视化训练面板。

**何时该读它**：要把模型推到生产、集成 vLLM 推理后端、上 WebUI 给非工程师同事调参、或一次性比较 SFT/DPO/KTO/PPO 几条线的端到端效果。**何时不该读**：想搞懂 SFT 数据 mask 的 cross-entropy 语义、prompt 段为何要置 -100——那是上面 [LLMs-from-scratch · ch07](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/exercise_experiments.py) 中 `targets[:instruction_length-1] = -100` 一行的事，跟 LLaMA-Factory 无关。
