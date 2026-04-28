---
title: "SFT 训练 Pipeline 填空"
description: "Level 2-3 填空：Chat Template、Loss Masking、数据整理、梯度累积"
topics: [fill-in, SFT, chat-template, loss-masking, gradient-accumulation, data-collation]
---
# SFT 训练 Pipeline 填空 (Level 2-3)

本练习覆盖 SFT（Supervised Fine-Tuning）训练流水线的核心组件：从对话格式化到完整训练循环。逐步实现一个可运行的 SFT pipeline，掌握指令微调的工程要点。

::: info 前置知识
- PyTorch 基础（`nn.Module`、`DataLoader`、优化器）
- Transformer 架构与 Causal LM 原理
- 基本的 tokenizer 使用经验
:::


::: tip SFT 核心思路
SFT 本质上是一个条件语言建模任务：给定指令（prompt），训练模型生成高质量回复（completion）。关键设计是 **只在 assistant 回复部分计算 loss**，prompt 部分作为条件但不贡献梯度。
:::


---

## 练习 1: ChatML 格式化（Level 2）

### 背景

Chat Template 是将多轮对话数据转换为模型可处理的字符串格式。ChatML 是一种广泛使用的模板标准（被 Qwen、Yi 等模型采用），它用特殊标记划分不同角色的内容：

```
<|im_start|>system
你是一个有帮助的助手。<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么可以帮你的吗？<|im_end|>
```

每条消息由 `<|im_start|>角色\n内容<|im_end|>\n` 组成。正确拼接这些标记是 SFT 数据预处理的第一步。

### 任务

请实现将多轮对话数据格式化为 ChatML 模板的函数。

```python
from typing import List, Dict

def format_chatml(messages, add_generation_prompt=False):
    """将对话列表格式化为 ChatML 字符串。"""
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    formatted = ""

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # ===== 填空 1: 拼接单条消息的 ChatML 格式 =====
        # 格式: <|im_start|>role\ncontent<|im_end|>\n
        formatted += _____

    if add_generation_prompt:
        # ===== 填空 2: 添加 assistant 生成起始标记 =====
        # 推理时需要在末尾加上 assistant 的起始标记，让模型从这里开始生成
        formatted += _____

    return formatted


# ====== 测试代码 ======
messages = [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "什么是 SFT？"},
    {"role": "assistant", "content": "SFT 是监督微调（Supervised Fine-Tuning）的缩写。"},
]

result = format_chatml(messages)
print("=== 完整对话 ===")
print(result)

result_gen = format_chatml(messages[:2], add_generation_prompt=True)
print("=== 推理模式（带生成起始标记） ===")
print(result_gen)

# 验证格式正确性
assert result.count("<|im_start|>") == 3
assert result.count("<|im_end|>") == 3
assert result_gen.endswith("<|im_start|>assistant\n")
print("格式验证通过!")
```

### 提示

::: details 提示
- 每条消息的格式是固定的：`f"{IM_START}{role}\n{content}{IM_END}\n"`
- `add_generation_prompt` 时只需要添加 `f"{IM_START}assistant\n"`，不加 `IM_END`
- 可以用 HuggingFace tokenizer 的 `apply_chat_template` 方法对比验证
:::


<details>
<summary>点击查看答案</summary>

```python
# 填空 1
formatted += f"{IM_START}{role}\n{content}{IM_END}\n"

# 填空 2
formatted += f"{IM_START}assistant\n"
```

**解析：**

1. **ChatML 格式**：每条消息被 `<|im_start|>` 和 `<|im_end|>` 包裹，角色名紧跟 start 标记，用换行符分隔角色和内容。
2. **`add_generation_prompt`**：推理时添加 `<|im_start|>assistant\n`（不加 `<|im_end|>`），让模型从这里开始自回归生成。
3. **不同模型的差异**：Llama 用 `[INST]...[/INST]`，ChatGLM 用 `[gMASK]`。建议优先使用 tokenizer 自带的 `apply_chat_template` 保证兼容性。

**验证 -- 与 HuggingFace 对比：**

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
hf_result = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print(f"是否一致: {result.strip() == hf_result.strip()}")
```

</details>

---

## 练习 2: SFT Loss Masking（Level 2）

### 背景

SFT 的核心技巧是 **Loss Masking**：只对 assistant 回复部分计算 Cross Entropy Loss，prompt 部分的 labels 设为 -100。PyTorch 的 `CrossEntropyLoss` 默认 `ignore_index=-100`，这些位置不贡献梯度。

为什么 mask prompt？因为 prompt 是输入条件，如果也计算 loss，模型会浪费容量记忆 prompt 格式。示意：

```
tokens:  [SYS] 你 是 助 手 [USER] 你 好 [ASST] 你 好 ！ [EOS]
labels:  -100 -100 -100 -100 -100 -100 -100 -100 -100  你  好  ！ EOS
```

### 任务

给定 token ids 和每个 token 对应的角色信息，生成正确的 labels。

```python
import torch
import torch.nn as nn

def create_sft_labels(input_ids, role_ids, assistant_role_id=2, ignore_index=-100):
    """根据角色标注生成 labels。role_ids: 0=system, 1=user, 2=assistant"""
    # ===== 填空 1: 创建 assistant 部分的 mask =====
    # assistant_mask 应为 bool tensor，True 表示该位置是 assistant 的 token
    assistant_mask = _____

    # ===== 填空 2: 初始化 labels 为全 ignore_index =====
    labels = _____

    # ===== 填空 3: 将 assistant 部分的 labels 设为对应的 input_ids =====
    # 语言模型的 label 是 "下一个 token"，这里简化处理，直接用当前 token 作为 label
    labels[assistant_mask] = _____

    return labels


# ====== 测试代码 ======
input_ids = torch.tensor([101, 20, 21, 22, 23, 102, 30, 31, 103, 40, 41, 42, 1])
role_ids  = torch.tensor([  0,  0,  0,  0,  0,   1,  1,  1,   2,  2,  2,  2, 2])

labels = create_sft_labels(input_ids, role_ids)
print(f"labels: {labels.tolist()}")
assert (labels[:8] == -100).all(), "prompt 部分应全为 -100"
assert (labels[8:] == input_ids[8:]).all(), "assistant 部分应等于 input_ids"

loss = nn.CrossEntropyLoss(ignore_index=-100)(torch.randn(len(input_ids), 1000), labels)
print(f"Loss: {loss.item():.4f}, 有效 token: {(labels != -100).sum().item()}")
print("验证通过!")
```

### 提示

::: details 提示
- `assistant_mask = (role_ids == assistant_role_id)`
- `torch.full_like(input_ids, ignore_index)` 可以创建与 `input_ids` 同形状、全为 `ignore_index` 的 tensor
- 布尔索引赋值：`labels[mask] = input_ids[mask]`
:::


<details>
<summary>点击查看答案</summary>

```python
# 填空 1: 创建 assistant mask
assistant_mask = (role_ids == assistant_role_id)

# 填空 2: 初始化 labels 为全 -100
labels = torch.full_like(input_ids, ignore_index)

# 填空 3: assistant 部分的 labels 设为对应的 input_ids
labels[assistant_mask] = input_ids[assistant_mask]
```

**解析：**

1. **`ignore_index=-100`**：PyTorch `CrossEntropyLoss` 的默认忽略值，这些位置不产生梯度。
2. **为什么 mask prompt**：SFT 目标是 "根据指令生成回复"，如果 prompt 也参与 loss，有效学习信号被稀释。
3. **实际注意事项**：assistant 起始标记通常也被 mask；EOS token 应参与 loss（否则模型不学会停止生成）；多轮对话中每轮 assistant 都保留 label。
4. **Label shift**：实际训练中 labels 需左移一位（next token prediction），HuggingFace 的 CausalLM 内部已处理。

</details>

---

## 练习 3: SFT DataCollator 实现（Level 3）

### 背景

在 SFT 训练中，一个 batch 内的序列通常长度不同。DataCollator 负责将不等长的样本拼接成统一形状的 batch tensor。核心操作包括：

1. **动态 padding**：将所有序列 pad 到 batch 内的最大长度（而非全局最大长度，节省计算）
2. **attention_mask 生成**：标记哪些位置是真实 token（1），哪些是 padding（0）
3. **labels 对齐**：padding 位置的 label 设为 -100，避免 pad token 贡献 loss

padding 方向：推理时常用 **left padding**（生成的 token 在序列末尾，方便批量解码），训练时 left/right 均可。

### 任务

实现一个完整的 `DataCollatorForSFT` 类。

```python
import torch
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class DataCollatorForSFT:
    """SFT 数据整理器：将不等长样本 pad 成 batch。"""
    pad_token_id: int = 0
    padding_side: str = "right"   # "right" 或 "left"
    ignore_index: int = -100

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # ===== 填空 1: 计算 batch 内的最大序列长度 =====
        max_len = _____

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feat in features:
            input_ids = feat["input_ids"]
            labels = feat["labels"]
            seq_len = len(input_ids)

            # ===== 填空 2: 计算需要 pad 的长度 =====
            pad_len = _____

            # 构造 padding tensor
            pad_ids = torch.full((pad_len,), self.pad_token_id, dtype=input_ids.dtype)
            pad_mask = torch.zeros(pad_len, dtype=torch.long)
            pad_labels = torch.full((pad_len,), self.ignore_index, dtype=labels.dtype)

            # 构造真实部分的 attention mask
            real_mask = torch.ones(seq_len, dtype=torch.long)

            if self.padding_side == "right":
                # ===== 填空 3: right padding 拼接 =====
                input_ids_padded = _____
                attention_mask = _____
                labels_padded = _____
            else:
                # ===== 填空 4: left padding 拼接 =====
                input_ids_padded = _____
                attention_mask = _____
                labels_padded = _____

            batch_input_ids.append(input_ids_padded)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels_padded)

        # ===== 填空 5: 将列表堆叠为 batch tensor =====
        return {
            "input_ids": _____,
            "attention_mask": _____,
            "labels": _____,
        }


# ====== 测试代码 ======
features = [
    {"input_ids": torch.tensor([10, 20, 30, 40, 50]), "labels": torch.tensor([-100, -100, 30, 40, 50])},
    {"input_ids": torch.tensor([11, 21, 31]),          "labels": torch.tensor([-100, 21, 31])},
    {"input_ids": torch.tensor([12, 22, 32, 42]),      "labels": torch.tensor([-100, -100, -100, 42])},
]

collator_right = DataCollatorForSFT(pad_token_id=0, padding_side="right")
batch = collator_right(features)
print(f"input_ids:\n{batch['input_ids']}")
print(f"attention_mask:\n{batch['attention_mask']}")
assert batch["input_ids"].shape == (3, 5)
assert (batch["attention_mask"][1] == torch.tensor([1, 1, 1, 0, 0])).all()
assert (batch["labels"][1, 3:] == -100).all(), "padding label 应为 -100"

collator_left = DataCollatorForSFT(pad_token_id=0, padding_side="left")
batch_left = collator_left(features)
assert (batch_left["attention_mask"][1] == torch.tensor([0, 0, 1, 1, 1])).all()
print("所有测试通过!")
```

### 提示

::: details 提示
- `max_len = max(len(f["input_ids"]) for f in features)`
- `pad_len = max_len - seq_len`
- right padding: `torch.cat([real_tensor, pad_tensor])`
- left padding: `torch.cat([pad_tensor, real_tensor])`
- `torch.stack(list_of_tensors)` 将列表堆叠为 batch tensor
:::


<details>
<summary>点击查看答案</summary>

```python
# 填空 1
max_len = max(len(f["input_ids"]) for f in features)

# 填空 2
pad_len = max_len - seq_len

# 填空 3: right padding
input_ids_padded = torch.cat([input_ids, pad_ids])
attention_mask = torch.cat([real_mask, pad_mask])
labels_padded = torch.cat([labels, pad_labels])

# 填空 4: left padding
input_ids_padded = torch.cat([pad_ids, input_ids])
attention_mask = torch.cat([pad_mask, real_mask])
labels_padded = torch.cat([pad_labels, labels])

# 填空 5
return {
    "input_ids": torch.stack(batch_input_ids),
    "attention_mask": torch.stack(batch_attention_mask),
    "labels": torch.stack(batch_labels),
}
```

**解析：**

1. **动态 vs 固定 padding**：固定 padding（pad to max_length=2048）浪费计算，动态 padding（pad to batch max）显著提升效率。
2. **attention_mask**：padding 位置的 attention score 设为 $-\infty$，防止模型关注 pad token。
3. **Left padding**：推理时新 token 追加在右侧，方便 KV cache 和批量解码。
4. **HuggingFace 对比**：`transformers.DataCollatorForSeq2Seq` 实现类似功能，更通用。

</details>

---

## 练习 4: 梯度累积训练循环（Level 2）

### 背景

受显存限制，SFT 训练时物理 batch size 通常很小。**梯度累积** 通过多步累加梯度模拟大 batch：物理 batch=2，累积步数=8，有效 batch=16。关键：每步 loss 除以累积步数，保证等价性。

$$\nabla_\theta = \sum_{i=1}^{N} \nabla_\theta \frac{L_i}{N} = \frac{1}{N}\sum_{i=1}^{N} \nabla_\theta L_i$$

### 任务

实现带梯度累积的训练循环。

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_with_gradient_accumulation(model, dataloader, optimizer, accumulation_steps=4, num_epochs=1):
    """带梯度累积的训练循环，返回每个 micro-step 的 loss。"""
    model.train()
    losses = []

    for epoch in range(num_epochs):
        for step, (x, y) in enumerate(dataloader):
            # 前向传播
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)

            # ===== 填空 1: 对 loss 进行缩放（除以累积步数） =====
            scaled_loss = _____

            # 反向传播（梯度会自动累加到 .grad 上）
            scaled_loss.backward()

            losses.append(loss.item())

            # ===== 填空 2: 判断是否到了更新参数的时机 =====
            if _____:
                # ===== 填空 3: 更新参数 =====
                _____
                # ===== 填空 4: 清零梯度 =====
                _____

    return losses


# ====== 验证：梯度累积等价性 ======
torch.manual_seed(42)
X = torch.randn(16, 10)
Y = torch.randint(0, 5, (16,))
dataset = TensorDataset(X, Y)

# 方案 A: 大 batch（batch_size=8），直接训练
torch.manual_seed(42)
model_a = nn.Linear(10, 5)
opt_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
loader_a = DataLoader(dataset, batch_size=8, shuffle=False)
model_a.train()
for x, y in loader_a:
    nn.functional.cross_entropy(model_a(x), y).backward()
opt_a.step()

# 方案 B: 小 batch + 梯度累积（batch_size=2, accum=4, 有效 batch=8）
torch.manual_seed(42)
model_b = nn.Linear(10, 5)
opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1)
loader_b = DataLoader(dataset, batch_size=2, shuffle=False)
train_with_gradient_accumulation(model_b, loader_b, opt_b, accumulation_steps=4)

# 比较参数
for (n_a, p_a), (n_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
    diff = (p_a - p_b).abs().max().item()
    print(f"{n_a}: max diff = {diff:.8f}")
    assert diff < 1e-5, f"参数差异过大: {diff}"
print("梯度累积等价性验证通过!")
```

### 提示

::: details 提示
- `scaled_loss = loss / accumulation_steps`
- 判断条件：`(step + 1) % accumulation_steps == 0`
- 更新参数：`optimizer.step()`
- 清零梯度：`optimizer.zero_grad()`
- 注意 loss 缩放是在 `.backward()` 之前，这样反向传播计算出的梯度自动被缩放
:::


<details>
<summary>点击查看答案</summary>

```python
# 填空 1: loss 缩放
scaled_loss = loss / accumulation_steps

# 填空 2: 判断是否到了更新时机
if (step + 1) % accumulation_steps == 0:

    # 填空 3: 更新参数
    optimizer.step()

    # 填空 4: 清零梯度
    optimizer.zero_grad()
```

**解析：**

1. **为什么缩放 loss？** PyTorch `.backward()` 默认累加梯度。不缩放的话，累积 N 步后梯度是大 batch 的 N 倍。除以 N 保证数学等价。
2. **`(step + 1) % accumulation_steps == 0`**：step 从 0 开始，用 `step + 1` 判断累积满 N 个 micro-batch 后更新。
3. **两种等价实现**：缩放 loss（`(loss / N).backward()`）或缩放梯度（`loss.backward()` 后手动 `p.grad /= N`），前者更常用。
4. **尾部处理**：数据总量不能整除时，末尾梯度不会更新，生产代码需额外处理。

</details>

---

## 练习 5: 完整 SFT 训练循环（Level 3）

### 背景

将前面所有组件组装成一个完整的 SFT 训练 pipeline：数据加载、chat template 格式化、loss masking、梯度累积、learning rate scheduler。这是一个端到端的练习，考验对各组件的整合能力。

Learning rate scheduler 采用 cosine with warmup 策略：在 warmup 阶段线性增加学习率，之后按余弦曲线衰减。这是 SFT 训练中最常用的调度策略。

$$\text{lr}(t) = \begin{cases} \text{lr}_{\max} \cdot \frac{t}{T_w} & t < T_w \\ \text{lr}_{\min} + \frac{1}{2}(\text{lr}_{\max} - \text{lr}_{\min})(1 + \cos(\frac{t - T_w}{T - T_w}\pi)) & t \geq T_w \end{cases}$$

其中 $T_w$ 为 warmup 步数，$T$ 为总步数。

### 任务

补全以下完整 SFT 训练脚本的关键部分。

```python
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict


# ---------- 组件 1: Chat Template ----------
def format_chatml(messages):
    """复用练习 1 的实现"""
    formatted = ""
    for msg in messages:
        formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return formatted


# ---------- 组件 2: 简易 Tokenizer & Dataset ----------
class SimpleTokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.pad_token_id = 0

    def encode(self, text):
        return [min(ord(c) % self.vocab_size, self.vocab_size - 1) for c in text]

class SFTDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=64):
        self.data = []
        for conv in conversations:
            ids = tokenizer.encode(format_chatml(conv))[:max_length]
            split = int(len(ids) * 0.6)  # 简化：后 40% 视为 assistant
            self.data.append({
                "input_ids": torch.tensor(ids),
                "labels": torch.tensor([-100] * split + ids[split:]),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ---------- 组件 3: DataCollator（复用练习 3 的思路） ----------
def collate_fn(features, pad_token_id=0):
    max_len = max(len(f["input_ids"]) for f in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        pad_len = max_len - len(f["input_ids"])
        batch["input_ids"].append(torch.cat([f["input_ids"], torch.full((pad_len,), pad_token_id)]))
        batch["attention_mask"].append(torch.cat([torch.ones(len(f["input_ids"])), torch.zeros(pad_len)]).long())
        batch["labels"].append(torch.cat([f["labels"], torch.full((pad_len,), -100)]))
    return {k: torch.stack(v) for k, v in batch.items()}


# ---------- 组件 5: Cosine LR Scheduler ----------
def cosine_lr_with_warmup(optimizer, step, total_steps, warmup_steps, lr_max, lr_min=0.0):
    """
    Cosine annealing with linear warmup.
    直接设置 optimizer 的学习率。
    """
    if step < warmup_steps:
        # ===== 填空 1: warmup 阶段 -- 线性增加学习率 =====
        lr = _____
    else:
        # ===== 填空 2: cosine 衰减阶段 =====
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = _____

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


# ---------- 组件 5: 训练循环 ----------
def sft_train(model, train_dataset, num_epochs=3, batch_size=2, lr=1e-4, accumulation_steps=2, warmup_ratio=0.1):
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dataloader) * num_epochs // accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    global_step, log = 0, []

    model.train()
    for epoch in range(num_epochs):
        for micro_step, batch in enumerate(dataloader):
            input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]

            # ===== 填空 3: 模型前向传播 =====
            logits = _____  # [B, L, V]

            # ===== 填空 4: label shift + reshape 计算 loss =====
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fn(_____, _____)  # [B*(L-1), V] 和 [B*(L-1)]

            (loss / accumulation_steps).backward()

            if (micro_step + 1) % accumulation_steps == 0:
                # ===== 填空 5: 梯度裁剪 =====
                _____
                optimizer.step()
                optimizer.zero_grad()
                # ===== 填空 6: 更新学习率 =====
                current_lr = cosine_lr_with_warmup(_____, _____, _____, _____, _____)
                global_step += 1
                log.append({"step": global_step, "loss": loss.item(), "lr": current_lr})
                print(f"Step {global_step}/{total_steps} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

    return log


# ====== 端到端验证 ======
torch.manual_seed(42)

class TinyLM(nn.Module):
    """单层 Transformer，用于测试"""
    def __init__(self, vocab_size=100, dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=128, batch_first=True)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        if attention_mask is not None:
            x = self.layer(x, src_key_padding_mask=(attention_mask == 0))
        else:
            x = self.layer(x)
        return self.head(x)

conversations = [
    [{"role": "system", "content": "你是助手"}, {"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好呀"}],
    [{"role": "user", "content": "1+1等于几"}, {"role": "assistant", "content": "等于2"}],
    [{"role": "user", "content": "什么是素数"}, {"role": "assistant", "content": "只能被1和自身整除的自然数"}],
    [{"role": "user", "content": "今天天气如何"}, {"role": "assistant", "content": "我无法获取实时天气信息"}],
]

tokenizer = SimpleTokenizer(vocab_size=100)
dataset = SFTDataset(conversations, tokenizer)
model = TinyLM()

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"数据集大小: {len(dataset)}\n")

log = sft_train(model, dataset, num_epochs=3, batch_size=2, lr=1e-3, accumulation_steps=2, warmup_ratio=0.2)

print(f"\n训练完成! 共 {len(log)} 步")
print(f"初始 loss: {log[0]['loss']:.4f}, 最终 loss: {log[-1]['loss']:.4f}")
assert log[-1]["loss"] < log[0]["loss"], "Loss 应该下降!"
print("端到端验证通过!")
```

### 提示

::: details 提示
- 填空 1: `lr = lr_max * step / warmup_steps`
- 填空 2: cosine 公式 `lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))`
- 填空 3: `model(input_ids, attention_mask=attention_mask)`
- 填空 4: `shift_logits.view(-1, shift_logits.size(-1))` 和 `shift_labels.view(-1)`
- 填空 5: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- 填空 6: 传入 `optimizer, global_step, total_steps, warmup_steps, lr`
:::


<details>
<summary>点击查看答案</summary>

```python
# 填空 1: warmup 线性增长
lr = lr_max * step / warmup_steps

# 填空 2: cosine 衰减
progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

# 填空 3: 模型前向传播
logits = model(input_ids, attention_mask=attention_mask)

# 填空 4: reshape 后计算 loss
loss = loss_fn(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
)

# 填空 5: 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 填空 6: 更新学习率
current_lr = cosine_lr_with_warmup(
    optimizer, global_step, total_steps, warmup_steps, lr
)
```

**解析：**

1. **Label shift**：Causal LM 用位置 $t$ 的 logits 预测位置 $t+1$ 的 token。logits 取 `[:, :-1, :]`，labels 取 `[:, 1:]`。HuggingFace 的 `CausalLMOutputWithPast` 内部也是这样处理的。

2. **Reshape**：`CrossEntropyLoss` 期望 `[N, C]` 和 `[N]`。把 `[B, L-1, V]` reshape 为 `[B*(L-1), V]`，labels 从 `[B, L-1]` reshape 为 `[B*(L-1)]`。

3. **梯度裁剪**：`clip_grad_norm_` 计算全局 L2 范数，超过 `max_norm` 则等比缩放，防止梯度爆炸，是 LLM 训练的标配。

4. **Cosine with warmup**：warmup 避免训练初期大学习率破坏预训练权重；cosine 衰减比 step decay 更平滑。

5. **工程扩展**：生产环境还需要混合精度（`torch.cuda.amp`）、分布式训练（`DDP`/`FSDP`）、checkpoint 保存、wandb 日志等。HuggingFace TRL 的 `SFTTrainer` 封装了所有这些。

</details>

---

## MLM 代码训练模式

完成上面的固定填空后，试试随机挖空模式 -- 每次点击「刷新」会随机遮盖不同的代码片段，帮你彻底记住每一行。

### ChatML 格式解析

<CodeMasker title="ChatML 对话模板拼接" :mask-ratio="0.15">
def format_chatml(messages, add_generation_prompt=False):
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"{IM_START}{role}\n{content}{IM_END}\n"
    if add_generation_prompt:
        formatted += f"{IM_START}assistant\n"
    return formatted
</CodeMasker>

### SFT Loss Masking

<CodeMasker title="只对 Assistant Token 计算 Loss" :mask-ratio="0.15">
def create_sft_labels(input_ids, role_ids, assistant_role_id=2, ignore_index=-100):
    assistant_mask = (role_ids == assistant_role_id)
    labels = torch.full_like(input_ids, ignore_index)
    labels[assistant_mask] = input_ids[assistant_mask]
    return labels
</CodeMasker>

### DataCollator 动态 Padding

<CodeMasker title="SFT DataCollator -- 动态 padding + attention mask" :mask-ratio="0.15">
class DataCollatorForSFT:
    pad_token_id: int = 0
    padding_side: str = "right"
    ignore_index: int = -100

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids, batch_attention_mask, batch_labels = [], [], []
        for feat in features:
            input_ids = feat["input_ids"]
            labels = feat["labels"]
            pad_len = max_len - len(input_ids)
            pad_ids = torch.full((pad_len,), self.pad_token_id, dtype=input_ids.dtype)
            pad_mask = torch.zeros(pad_len, dtype=torch.long)
            pad_labels = torch.full((pad_len,), self.ignore_index, dtype=labels.dtype)
            real_mask = torch.ones(len(input_ids), dtype=torch.long)
            if self.padding_side == "right":
                input_ids_padded = torch.cat([input_ids, pad_ids])
                attention_mask = torch.cat([real_mask, pad_mask])
                labels_padded = torch.cat([labels, pad_labels])
            else:
                input_ids_padded = torch.cat([pad_ids, input_ids])
                attention_mask = torch.cat([pad_mask, real_mask])
                labels_padded = torch.cat([pad_labels, labels])
            batch_input_ids.append(input_ids_padded)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels_padded)
        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }
</CodeMasker>

### 梯度累积训练循环

<CodeMasker title="梯度累积 -- loss 缩放 + 定时更新" :mask-ratio="0.15">
def train_with_gradient_accumulation(model, dataloader, optimizer, accumulation_steps=4):
    model.train()
    for step, (x, y) in enumerate(dataloader):
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
</CodeMasker>
