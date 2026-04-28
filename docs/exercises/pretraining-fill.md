---
title: "预训练技术填空"
description: "Level 2-3 填空：数据预处理、CLM 目标函数、学习率调度、训练监控"
topics: [fill-in, pretraining, CLM, data-pipeline, lr-schedule, training-loop]
---
# 预训练技术填空 (Level 2-3)

> **难度:** 中等-困难 | **前置知识:** [预训练](/training/pretraining.md) | **预计时间:** 60-90 分钟

本练习覆盖大模型预训练的核心技术组件：CLM 目标函数、数据预处理、学习率调度、训练监控，最终串联成完整的 mini 预训练流程。每个空白用 `_____` 标记，请填入正确的 PyTorch 代码。

::: info 前置知识
- PyTorch 基础（`nn.Module`、`DataLoader`、优化器）
- Transformer 架构与 Causal LM 原理
:::

---

## 练习 1: Causal Language Modeling 目标函数（Level 2）

### 背景

CLM 是 GPT 系列模型的核心训练目标：给定 $[x_1, ..., x_{t-1}]$，预测 $x_t$。训练时模型对整个序列做前向传播得到 logits，但计算 loss 需要 **shift 操作**：`logits[:-1]` 预测 `labels[1:]`。

这个 shift 体现了自回归模型 "用过去预测未来" 的本质——位置 $i$ 的输出只能看到位置 $0..i$ 的输入，所以它预测的是位置 $i+1$ 的 token。

### 任务

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_clm_loss(logits, labels, ignore_index=-100):
    """
    计算 CLM loss。
    参数:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)
    返回: 标量 loss
    """
    # ===== 填空 1: shift logits 和 labels =====
    # logits 取前 T-1 个位置，labels 取后 T-1 个位置
    shift_logits = _____
    shift_labels = _____

    # ===== 填空 2: CrossEntropyLoss 计算 loss =====
    # 需要 reshape: logits -> (N, vocab_size), labels -> (N,)
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fn(_____, _____)
    return loss

# ====== 测试 ======
torch.manual_seed(42)
batch_size, seq_len, vocab_size = 2, 8, 100
logits = torch.randn(batch_size, seq_len, vocab_size)
labels = torch.randint(0, vocab_size, (batch_size, seq_len))

loss = compute_clm_loss(logits, labels)
print(f"CLM Loss: {loss.item():.4f}")
print(f"期望接近 ln({vocab_size}) = {torch.log(torch.tensor(float(vocab_size))).item():.4f}")

# 测试 ignore_index
labels_masked = labels.clone()
labels_masked[:, :3] = -100
loss_masked = compute_clm_loss(logits, labels_masked)
print(f"Masked Loss: {loss_masked.item():.4f}, 与原 loss 不同: {abs(loss.item() - loss_masked.item()) > 0.01}")
```

### 提示

- 填空 1：`logits[:, :-1, :]` 和 `labels[:, 1:]`
- 填空 2：用 `.reshape(-1, vocab_size)` 和 `.reshape(-1)` 展平

<details>
<summary>参考答案</summary>

```python
# 填空 1
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]

# 填空 2
loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
```

`logits[:, :-1, :]` 取位置 0 到 T-2 的预测，与 `labels[:, 1:]`（位置 1 到 T-1）对齐。`reshape(-1, vocab_size)` 将 batch 和 seq 维度展平，满足 `CrossEntropyLoss` 的输入格式。随机初始化时 loss 接近 $\ln(V)$，因为输出接近均匀分布。

```python
# 验证
assert shift_logits.shape == (batch_size, seq_len - 1, vocab_size)
assert shift_labels.shape == (batch_size, seq_len - 1)
```

</details>

---

## 练习 2: 数据预处理 -- 文档拼接与分块（Level 2）

### 背景

预训练数据由大量长度不一的文档组成。为了高效利用 GPU，需要将文档拼接（packing）成固定长度序列：用 EOS token 连接多个文档，然后按 `max_seq_len` 切分。这样避免了 padding 浪费，几乎 100% 的 token 都是有效的。

拼接后一个序列可能包含多个文档片段。可以通过 document mask 在文档边界处重置 attention，防止跨文档信息泄露。

### 任务

```python
from typing import List

def pack_documents(documents: List[List[int]], max_seq_len: int, eos_token_id: int) -> List[List[int]]:
    """将多个文档拼接并切分为固定长度序列。"""
    # ===== 填空 1: 用 EOS token 拼接所有文档 =====
    all_tokens = []
    for doc in documents:
        _____

    # ===== 填空 2: 按 max_seq_len 切分，丢弃不足的尾部 =====
    chunks = []
    _____
    return chunks

def create_document_masks(chunk: List[int], eos_token_id: int) -> List[int]:
    """为序列创建文档编号 mask，遇到 EOS 则递增文档编号。"""
    # ===== 填空 3: 遍历 chunk，EOS 后递增文档编号 =====
    doc_ids = []
    current_doc = 0
    for token in chunk:
        doc_ids.append(current_doc)
        if token == eos_token_id:
            _____
    return doc_ids

# ====== 测试 ======
documents = [
    [10, 20, 30],
    [40, 50, 60, 70, 80],
    [90, 100],
    [110, 120, 130, 140],
    [150, 160, 170],
]
eos_id = 0
max_seq_len = 8

chunks = pack_documents(documents, max_seq_len, eos_id)
print(f"文档总 token 数（含 EOS）: {sum(len(d) + 1 for d in documents)}")
print(f"切分出 {len(chunks)} 个块，长度 {max_seq_len}")
for i, chunk in enumerate(chunks):
    doc_ids = create_document_masks(chunk, eos_id)
    eos_pos = [j for j, t in enumerate(chunk) if t == eos_id]
    print(f"  块 {i}: tokens={chunk}, doc_ids={doc_ids}, EOS={eos_pos}")
```

### 提示

- 填空 1：`all_tokens.extend(doc)` 加 `all_tokens.append(eos_token_id)`
- 填空 2：`range(0, len(all_tokens) - max_seq_len + 1, max_seq_len)` 做步进切片
- 填空 3：`current_doc += 1`

<details>
<summary>参考答案</summary>

```python
# 填空 1
all_tokens.extend(doc)
all_tokens.append(eos_token_id)

# 填空 2
for i in range(0, len(all_tokens) - max_seq_len + 1, max_seq_len):
    chunks.append(all_tokens[i : i + max_seq_len])

# 填空 3
current_doc += 1
```

每个文档后追加 EOS 作为分隔符。步长为 `max_seq_len` 的非重叠切分确保序列等长，末尾不足部分直接丢弃。`doc_ids` 可用于构建 block-diagonal attention mask，防止跨文档注意力。

```python
# 验证
all_flat = []
for doc in documents:
    all_flat.extend(doc)
    all_flat.append(eos_id)
reconstructed = [t for c in chunks for t in c]
assert reconstructed == all_flat[:len(reconstructed)]
for c in chunks:
    assert len(c) == max_seq_len
```

</details>

---

## 练习 3: 学习率调度 -- Warmup + Cosine Decay（Level 2）

### 背景

几乎所有主流大模型（GPT-3、LLaMA、Qwen）都采用 **Linear Warmup + Cosine Decay** 调度。训练初期梯度不稳定，先用 warmup 逐步增大学习率；达到峰值后用 cosine 曲线平滑衰减到最小值，避免后期震荡。

典型配置：`warmup_steps` 约占总步数 1-5%，`min_lr` 为 `max_lr` 的 1/10。

### 任务

```python
import math

def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    """
    三段式学习率调度:
      1. step < warmup_steps: 线性 warmup (0 -> max_lr)
      2. step > total_steps:  返回 min_lr
      3. 其他: cosine decay (max_lr -> min_lr)
    """
    if step < warmup_steps:
        # ===== 填空 1: 线性 warmup =====
        return _____

    if step > total_steps:
        # ===== 填空 2: 超过总步数 =====
        return _____

    # ===== 填空 3: cosine decay =====
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    # 公式: min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
    lr = _____
    return lr

# ====== 测试 ======
max_lr, min_lr = 3e-4, 3e-5
warmup_steps, total_steps = 200, 2000

print("=== 关键点验证 ===")
tests = [(0, 0.0), (100, 1.5e-4), (200, 3e-4), (2000, 3e-5), (2500, 3e-5)]
for s, expected in tests:
    actual = get_lr(s, max_lr, min_lr, warmup_steps, total_steps)
    print(f"  step={s:5d}: lr={actual:.2e} (期望 {expected:.2e})")

# 纯文本 LR 曲线
print("\n=== LR 曲线 ===")
for s in range(0, total_steps + 1, total_steps // 20):
    lr = get_lr(s, max_lr, min_lr, warmup_steps, total_steps)
    bar = "#" * int(lr / max_lr * 40)
    print(f"  step {s:5d} | {lr:.2e} | {bar}")
```

### 提示

- 填空 1：`max_lr * step / warmup_steps`
- 填空 2：`min_lr`
- 填空 3：直接用公式，`math.cos(math.pi * progress)`

<details>
<summary>参考答案</summary>

```python
# 填空 1
return max_lr * step / warmup_steps

# 填空 2
return min_lr

# 填空 3
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

Warmup 阶段 `step/warmup_steps` 从 0 到 1 线性增长。Cosine decay 中，`progress=0` 时 `cos(0)=1`，`lr=max_lr`；`progress=1` 时 `cos(pi)=-1`，`lr=min_lr`。cosine 曲线保证衰减前慢后快再变慢，比线性衰减更平滑。

```python
# 验证
assert get_lr(0, max_lr, min_lr, warmup_steps, total_steps) == 0.0
assert abs(get_lr(warmup_steps, max_lr, min_lr, warmup_steps, total_steps) - max_lr) < 1e-10
assert abs(get_lr(total_steps, max_lr, min_lr, warmup_steps, total_steps) - min_lr) < 1e-10
```

</details>

---

## 练习 4: 训练指标监控（Level 2-3）

### 背景

大模型预训练持续数天到数周，监控至关重要。核心指标：**Perplexity** = `exp(loss)`，直观表示模型在多少个 token 中 "犹豫"；**Gradient Norm** 反映训练稳定性；**EMA Loss**（指数移动平均）过滤波动展示趋势；**Loss Spike 检测**在 loss 突增时告警。

### 任务

```python
import torch
import torch.nn as nn
import math
from typing import List, Optional

def compute_perplexity(loss: float) -> float:
    """Perplexity = exp(loss)"""
    # ===== 填空 1: 计算 perplexity =====
    return _____

def compute_gradient_norm(model: nn.Module) -> float:
    """计算所有参数梯度的 L2 范数。"""
    # ===== 填空 2: 累加每个参数 grad 的平方范数，最后开根号 =====
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            _____
    return _____

class TrainingMonitor:
    """训练指标监控器。"""
    def __init__(self, ema_decay: float = 0.99, spike_threshold: float = 1.5):
        self.ema_decay = ema_decay
        self.spike_threshold = spike_threshold
        self.ema_loss: Optional[float] = None
        self.history: List[dict] = []

    def update(self, loss: float, lr: float, grad_norm: float) -> dict:
        ppl = compute_perplexity(loss)

        # ===== 填空 3: 更新 EMA loss =====
        # 首次用 loss 初始化；后续 ema = decay * ema + (1-decay) * loss
        if self.ema_loss is None:
            self.ema_loss = _____
        else:
            self.ema_loss = _____

        # ===== 填空 4: 检测 loss spike =====
        is_spike = _____

        metrics = {"loss": loss, "ppl": ppl, "ema_loss": self.ema_loss,
                   "grad_norm": grad_norm, "lr": lr, "is_spike": is_spike}
        self.history.append(metrics)
        return metrics

# ====== 测试 ======
print("=== Perplexity ===")
for l in [4.6, 3.0, 2.0, 1.0]:
    print(f"  loss={l:.1f} -> ppl={compute_perplexity(l):.1f}")

print("\n=== Gradient Norm ===")
model = nn.Linear(10, 5)
x = torch.randn(3, 10)
model(x).sum().backward()
gn = compute_gradient_norm(model)
manual = torch.sqrt(model.weight.grad.norm()**2 + model.bias.grad.norm()**2).item()
print(f"  计算: {gn:.4f}, 手动验证: {manual:.4f}, 匹配: {abs(gn - manual) < 1e-5}")

print("\n=== Monitor ===")
import random
random.seed(42)
monitor = TrainingMonitor(ema_decay=0.9, spike_threshold=1.5)
for i in range(30):
    loss = 5.0 * math.exp(-0.05 * i) + random.gauss(0, 0.1)
    if i == 15:
        loss *= 3.0
    m = monitor.update(loss, 3e-4, 1.0 + random.random())
    if m["is_spike"]:
        print(f"  [SPIKE] step={i}, loss={m['loss']:.4f}, ema={m['ema_loss']:.4f}")
```

### 提示

- 填空 1：`math.exp(loss)`
- 填空 2：`p.grad.data.norm(2).item() ** 2`，返回 `math.sqrt(total_norm_sq)`
- 填空 3：初始化 `loss`，更新 `self.ema_decay * self.ema_loss + (1 - self.ema_decay) * loss`
- 填空 4：`loss > self.ema_loss * self.spike_threshold`

<details>
<summary>参考答案</summary>

```python
# 填空 1
return math.exp(loss)

# 填空 2
total_norm_sq += p.grad.data.norm(2).item() ** 2
return math.sqrt(total_norm_sq)

# 填空 3
self.ema_loss = loss
self.ema_loss = self.ema_decay * self.ema_loss + (1 - self.ema_decay) * loss

# 填空 4
is_spike = loss > self.ema_loss * self.spike_threshold
```

Perplexity = $e^L$，当 loss = $\ln(V)$ 时等于词表大小，表示随机猜。全局梯度范数 $\sqrt{\sum_i \|g_i\|^2}$ 与 `clip_grad_norm_` 内部计算一致。EMA 公式 $\bar{L}_t = \alpha\bar{L}_{t-1} + (1-\alpha)L_t$ 中 $\alpha$ 越大越平滑。

```python
assert abs(compute_perplexity(0.0) - 1.0) < 1e-10
assert abs(compute_perplexity(math.log(100)) - 100.0) < 1e-5
```

</details>

---

## 练习 5: Mini 预训练完整流程（Level 3）

### 背景

将前面的组件串联：CLM loss、数据拼接、LR 调度、梯度裁剪、指标监控。使用 2 层 mini Transformer 在合成数据上训练，规模虽小但完整保留了大模型预训练的所有关键步骤。

实际大模型预训练在此基础上增加的主要是分布式并行和更大规模，核心逻辑无本质区别。

### 任务

```python
import torch
import torch.nn as nn
import math
import random

# ---------- 模型（已提供）---------- #
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, max_seq_len=64):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.0, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        causal_mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()
        memory = torch.zeros(B, 1, x.size(-1), device=input_ids.device)
        x = self.decoder(x, memory, tgt_mask=causal_mask)
        return self.lm_head(x)

# ---------- 工具函数（复用前面的实现）---------- #
def compute_clm_loss(logits, labels):
    s_logits = logits[:, :-1, :].contiguous()
    s_labels = labels[:, 1:].contiguous()
    return nn.CrossEntropyLoss()(s_logits.view(-1, s_logits.size(-1)), s_labels.view(-1))

def get_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    if step < warmup_steps: return max_lr * step / warmup_steps
    if step > total_steps: return min_lr
    p = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))

def compute_gradient_norm(model):
    s = sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)
    return math.sqrt(s)

# ---------- 合成数据 ---------- #
def generate_synthetic_data(vocab_size, num_docs=50, seed=42):
    """生成带有可学习 n-gram 模式的合成文档。"""
    rng = random.Random(seed)
    docs = []
    for _ in range(num_docs):
        pat = [rng.randint(2, vocab_size-1) for _ in range(rng.randint(3, 6))]
        docs.append([pat[i % len(pat)] for i in range(rng.randint(10, 50))])
    return docs

# ---------- 训练循环（填空）---------- #
def pretrain(vocab_size=256, max_seq_len=32, batch_size=4, total_steps=50,
             max_lr=1e-3, min_lr=1e-4, warmup_steps=5, max_grad_norm=1.0,
             eos_token_id=1, log_interval=5, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据准备
    documents = generate_synthetic_data(vocab_size)
    all_tokens = []
    for doc in documents:
        all_tokens.extend(doc)
        all_tokens.append(eos_token_id)
    chunks = [all_tokens[i:i+max_seq_len]
              for i in range(0, len(all_tokens)-max_seq_len+1, max_seq_len)]
    print(f"数据: {len(documents)} 文档, {len(all_tokens)} token, {len(chunks)} 块")

    # ===== 填空 1: 创建模型 =====
    model = _____

    # ===== 填空 2: 创建 AdamW 优化器 =====
    optimizer = _____

    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 60)

    initial_loss = None
    ema_loss = None

    for step in range(total_steps):
        # ===== 填空 3: 设置学习率 =====
        lr = get_lr(step, max_lr, min_lr, warmup_steps, total_steps)
        for pg in optimizer.param_groups:
            _____

        # 采样 batch
        idx = [random.randint(0, len(chunks)-1) for _ in range(batch_size)]
        batch = torch.tensor([chunks[i] for i in idx], dtype=torch.long).to(device)

        # 前向
        logits = model(batch)
        loss = compute_clm_loss(logits, batch)

        # ===== 填空 4: 反向传播 =====
        optimizer.zero_grad()
        _____

        # ===== 填空 5: 梯度裁剪 =====
        _____

        grad_norm = compute_gradient_norm(model)

        # ===== 填空 6: 参数更新 =====
        _____

        # 监控
        lv = loss.item()
        if initial_loss is None: initial_loss = lv
        ema_loss = lv if ema_loss is None else 0.9 * ema_loss + 0.1 * lv
        ppl = math.exp(lv)

        if step % log_interval == 0 or step == total_steps - 1:
            print(f"step {step:3d} | loss {lv:.4f} | ppl {ppl:7.1f} | "
                  f"ema {ema_loss:.4f} | gnorm {grad_norm:.3f} | lr {lr:.2e}")

    final_loss = loss.item()
    print("-" * 60)
    print(f"初始 loss: {initial_loss:.4f} (ppl={math.exp(initial_loss):.1f})")
    print(f"最终 loss: {final_loss:.4f} (ppl={math.exp(final_loss):.1f})")
    assert final_loss < initial_loss, "Loss 应该下降！"
    print("验证通过: loss 成功下降")
    return model

if __name__ == "__main__":
    pretrain()
```

### 提示

- 填空 1：`MiniTransformer(vocab_size, max_seq_len=max_seq_len).to(device)`
- 填空 2：`torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)`
- 填空 3：`pg["lr"] = lr`
- 填空 4：`loss.backward()`
- 填空 5：`torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)`
- 填空 6：`optimizer.step()`

<details>
<summary>参考答案</summary>

```python
# 填空 1
model = MiniTransformer(vocab_size, max_seq_len=max_seq_len).to(device)

# 填空 2
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)

# 填空 3
pg["lr"] = lr

# 填空 4
loss.backward()

# 填空 5
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

# 填空 6
optimizer.step()
```

训练循环的顺序至关重要：`zero_grad` -> `backward` -> `clip_grad_norm_` -> `step`。梯度裁剪必须在 backward 之后、step 之前，否则无法限制梯度爆炸。AdamW 的 `weight_decay=0.01` 提供 L2 正则化，是 GPT-3/LLaMA 的标准配置。手动设置 `param_group["lr"]` 实现自定义 schedule，比 PyTorch 内置 scheduler 更灵活。

在合成数据上 50 步训练，loss 应从约 5.5（ln(256)）明显下降，验证所有组件正确协作。

</details>

---

## MLM 代码训练模式

完成上面的固定填空后，试试随机挖空模式 -- 每次点击「刷新」会随机遮盖不同的代码片段，帮你彻底记住每一行。

### CLM Loss 计算

<CodeMasker title="Causal LM Loss -- shift + cross entropy" :mask-ratio="0.15">
def compute_clm_loss(logits, labels, ignore_index=-100):
    # shift: logits 取前 T-1，labels 取后 T-1
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fn(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1)
    )
    return loss
</CodeMasker>

### 数据拼接与文档 Mask

<CodeMasker title="文档 Packing + Attention Mask" :mask-ratio="0.15">
def pack_documents(documents, max_seq_len, eos_token_id):
    all_tokens = []
    for doc in documents:
        all_tokens.extend(doc)
        all_tokens.append(eos_token_id)
    chunks = []
    for i in range(0, len(all_tokens) - max_seq_len + 1, max_seq_len):
        chunks.append(all_tokens[i : i + max_seq_len])
    return chunks

def create_document_masks(chunk, eos_token_id):
    doc_ids = []
    current_doc = 0
    for token in chunk:
        doc_ids.append(current_doc)
        if token == eos_token_id:
            current_doc += 1
    return doc_ids
</CodeMasker>

### Warmup + Cosine Decay 学习率调度

<CodeMasker title="线性 Warmup + 余弦衰减 LR Schedule" :mask-ratio="0.15">
def get_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
    return lr
</CodeMasker>

### 预训练循环（梯度裁剪 + LR 调度）

<CodeMasker title="训练循环核心：forward → backward → clip → step" :mask-ratio="0.15">
model = MiniTransformer(vocab_size, max_seq_len=max_seq_len).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)

for step in range(total_steps):
    lr = get_lr(step, max_lr, min_lr, warmup_steps, total_steps)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    logits = model(batch)
    loss = compute_clm_loss(logits, batch)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
</CodeMasker>
