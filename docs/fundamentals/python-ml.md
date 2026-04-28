---
title: "Python & 机器学习"
description: "PyTorch 核心操作、Autograd、nn.Module、Hugging Face 生态与调试工具"
topics: [pytorch, autograd, nn-module, dataset, dataloader, huggingface, transformers, wandb]
---
# Python & 机器学习

> **一句话总结:** Python 生态（NumPy、PyTorch、Hugging Face）是大模型开发的通用语言，熟练掌握张量操作和数据管线是高效实验的基础。

## PyTorch 基础

PyTorch 是当前大模型研究和工程的主流框架。本节将介绍张量（Tensor）的创建与操作、自动求导机制（Autograd）、以及如何搭建和训练一个基本模型。重点在于理解计算图和 `.backward()` 的运作方式。

### 张量创建与基础操作

张量是 PyTorch 的核心数据结构，本质上是多维数组。掌握张量操作是一切的基础。

```python
import torch

# === 创建张量 ===
# 从 Python 列表创建
x = torch.tensor([1.0, 2.0, 3.0])

# 常用初始化
zeros = torch.zeros(3, 4)           # 全零 (3×4)
ones = torch.ones(2, 3)             # 全一 (2×3)
rand = torch.randn(2, 3)            # 标准正态分布
arange = torch.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
eye = torch.eye(3)                  # 3×3 单位矩阵

# 指定数据类型（大模型常用 bfloat16）
x_bf16 = torch.randn(2, 3, dtype=torch.bfloat16)
x_fp16 = torch.randn(2, 3, dtype=torch.float16)

# === 索引与切片 ===
x = torch.randn(4, 5)
print(x[0])          # 第 0 行
print(x[:, 1])       # 第 1 列
print(x[1:3, 2:4])   # 子矩阵
print(x[x > 0])      # 布尔索引：所有正数元素

# 高级索引
indices = torch.tensor([0, 2, 3])
print(x[indices])    # 取第 0、2、3 行

# === 形状操作 ===
x = torch.randn(2, 3, 4)
print(x.shape)                      # torch.Size([2, 3, 4])
print(x.view(6, 4).shape)           # 重塑为 (6, 4)，共享内存
print(x.reshape(2, 12).shape)       # 重塑，不保证共享内存
print(x.permute(2, 0, 1).shape)     # 维度置换 → (4, 2, 3)
print(x.unsqueeze(0).shape)         # 增加维度 → (1, 2, 3, 4)
print(x.unsqueeze(0).squeeze(0).shape)  # 去除维度 → (2, 3, 4)
```

### 广播机制（Broadcasting）

广播让不同形状的张量可以进行运算，PyTorch 自动扩展较小张量的维度。理解广播是避免 shape 错误的关键。

```python
# 规则：从最后一个维度开始比较，维度要么相同、要么其中一个为 1
a = torch.randn(3, 4)   # (3, 4)
b = torch.randn(1, 4)   # (1, 4) → 广播为 (3, 4)
c = a + b                # OK: (3, 4)

# 常见用法：对 batch 中每个样本减去均值
batch = torch.randn(32, 768)          # (batch_size, hidden_dim)
mean = batch.mean(dim=0, keepdim=True) # (1, 768)
centered = batch - mean                # 广播 → (32, 768)

# 注意力分数中的广播（mask 应用）
scores = torch.randn(2, 8, 64, 64)   # (batch, heads, seq, seq)
mask = torch.ones(1, 1, 64, 64)      # (1, 1, seq, seq)
masked = scores + mask                 # 广播 → (2, 8, 64, 64)
```

### 设备管理（CPU / GPU）

大模型训练离不开 GPU。确保张量和模型在同一设备上是常见的调试点。

```python
# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 张量移动到 GPU
x = torch.randn(3, 4).to(device)
# 或者直接在 GPU 上创建
y = torch.randn(3, 4, device=device)

# 模型移动到 GPU
model = MyModel().to(device)

# 常见错误：张量在不同设备上
# RuntimeError: Expected all tensors to be on the same device
# 解决：确保输入、标签、模型都在同一设备
inputs = inputs.to(device)
labels = labels.to(device)

# 多 GPU 时指定设备
x = torch.randn(3, 4, device="cuda:0")  # 第 0 块 GPU
y = torch.randn(3, 4, device="cuda:1")  # 第 1 块 GPU
```

## Autograd 原理

PyTorch 的自动求导系统 Autograd 是训练神经网络的基石。它通过动态构建计算图，自动计算任意张量运算的梯度。

### 计算图与 .backward()

```python
# requires_grad=True 告诉 PyTorch 追踪该张量上的所有操作
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2 + 3 * x    # y = x² + 3x
z = y.sum()            # 标量化（backward 需要标量）

# 反向传播：计算 dz/dx
z.backward()
print(x.grad)          # tensor([7., 9.])  即 2x + 3

# 计算图的关键性质：
# 1. 动态构建：每次前向传播都重新构建计算图
# 2. 叶子节点：requires_grad=True 的张量是叶子，只有叶子保留 .grad
# 3. 非叶子节点的 grad_fn 记录了创建该张量的操作
print(y.grad_fn)       # <AddBackward0 object>
```

### 梯度累积与清零

```python
# 重要：梯度默认是累积的！必须手动清零
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in dataloader:
    optimizer.zero_grad()     # 清零梯度（每次迭代必须）
    loss = model(batch)
    loss.backward()           # 计算梯度
    optimizer.step()          # 更新参数

# 梯度累积技巧：在显存不够时模拟大 batch
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps   # 缩放 loss
    loss.backward()                             # 梯度累积
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### detach 与 no_grad

```python
# torch.no_grad()：推理时关闭梯度计算，节省显存和加速
with torch.no_grad():
    outputs = model(inputs)     # 不构建计算图
    # 常用于验证阶段和推理阶段

# detach()：从计算图中分离张量，返回不追踪梯度的新张量
hidden = encoder(x)
hidden_detached = hidden.detach()   # 截断梯度流
output = decoder(hidden_detached)   # decoder 的梯度不会流回 encoder

# 典型场景：目标网络更新（DQN / EMA）
# target_value = target_model(next_state).detach()

# 注意区别：
# .detach() — 返回新张量，共享数据但不追踪梯度
# with torch.no_grad() — 上下文管理器，内部所有操作都不追踪梯度
# .requires_grad_(False) — 原地修改，让张量不再需要梯度
```

## nn.Module 详解

`nn.Module` 是 PyTorch 中所有神经网络模块的基类。理解它的机制对于构建和调试模型至关重要。

### \_\_init\_\_ 与 forward

```python
import torch.nn as nn

class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()  # 必须调用父类 __init__
        # 在 __init__ 中定义所有子模块和参数
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward 定义前向传播逻辑
        # 调用 model(x) 时自动调用 forward（不要直接调用 model.forward(x)）
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + self.dropout(attn_out)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        return x
```

### 参数管理

```python
model = SimpleTransformerBlock(d_model=512, n_heads=8, d_ff=2048)

# 查看所有参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")

# 统计参数量（大模型面试常考）
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params:,}, Trainable: {trainable_params:,}")

# 冻结部分参数（迁移学习 / LoRA 场景）
for name, param in model.named_parameters():
    if "attn" not in name:
        param.requires_grad_(False)

# 保存和加载模型
torch.save(model.state_dict(), "model.pt")                # 只保存参数
model.load_state_dict(torch.load("model.pt"))              # 加载参数
# 注意：save 整个 model 不推荐，因为依赖 pickle 序列化类定义
```

### Hook 机制

Hook 允许你在不修改模型代码的情况下，获取中间层的输入输出或修改梯度。在大模型调试和可解释性研究中非常有用。

```python
# Forward Hook：获取中间层输出
activations = {}

def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# 注册 hook
handle = model.norm1.register_forward_hook(save_activation("norm1"))
output = model(torch.randn(1, 10, 512))
print(activations["norm1"].shape)   # 获取 norm1 的输出

# 用完记得移除 hook
handle.remove()

# Backward Hook：检查或修改梯度
def grad_hook(module, grad_input, grad_output):
    print(f"Gradient norm: {grad_output[0].norm():.4f}")

model.ffn.register_full_backward_hook(grad_hook)
```

## 数据处理

高质量的数据管线直接决定模型性能。本节覆盖从原始文本到训练批次的完整流程。

### 自定义 Dataset

```python
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    """自定义文本分类数据集"""
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            # 注意：这里不做 padding，padding 留给 collate_fn 动态处理
        )
        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "label": torch.tensor(self.labels[idx]),
        }
```

### Collate Function 与动态 Padding

固定长度 padding 浪费计算，动态 padding 只 pad 到当前 batch 的最大长度，效率更高。

```python
def dynamic_padding_collate(batch: list[dict]) -> dict:
    """动态 padding：只 pad 到当前 batch 内最长序列的长度"""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids_list = []
    attention_mask_list = []
    labels = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len
        # 右侧 padding
        input_ids_list.append(
            torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        attention_mask_list.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels.append(item["label"])

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels),
    }

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,           # 多进程加载数据
    collate_fn=dynamic_padding_collate,
    pin_memory=True,         # 加速 CPU → GPU 数据传输
    drop_last=True,          # 丢弃不完整的最后一个 batch
)
```

### DataLoader 的关键参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `batch_size` | 每个批次的样本数 | 受显存限制，尽可能大 |
| `num_workers` | 数据加载子进程数 | 通常设为 CPU 核心数的一半 |
| `pin_memory` | 固定内存以加速传输 | 使用 GPU 时设为 True |
| `prefetch_factor` | 每个 worker 预取的 batch 数 | 默认 2，通常不需改 |
| `persistent_workers` | 保持 worker 进程存活 | 数据集大时设为 True |

## Hugging Face 生态

Hugging Face 是大模型社区的核心基础设施。掌握其工具链可以大幅提升开发效率。

### transformers：AutoModel / AutoTokenizer / Trainer

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# === AutoTokenizer：自动加载对应的分词器 ===
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokens = tokenizer("Hello, world!", return_tensors="pt")
print(tokens["input_ids"])          # tensor([[1, 15043, 29892, 3186, 29991]])
print(tokenizer.decode(tokens["input_ids"][0]))  # "Hello, world!"

# === AutoModel：自动加载对应的模型架构 ===
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,      # 使用 bf16 节省显存
    device_map="auto",                # 自动分配到可用 GPU
    load_in_4bit=True,                # 4-bit 量化加载（需要 bitsandbytes）
)

# 生成文本
inputs = tokenizer("The meaning of life is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, top_p=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# === Trainer：封装训练循环 ===
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,     # 等效 batch_size = 4 * 8 = 32
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    bf16=True,                         # 混合精度训练
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    report_to="wandb",                 # 上报到 wandb
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

### datasets：高效数据加载

```python
from datasets import load_dataset

# 从 Hub 加载数据集
dataset = load_dataset("tatsu-lab/alpaca")

# 流式加载（不下载到本地，适合大数据集）
dataset = load_dataset("HuggingFaceFW/fineweb", streaming=True)
for example in dataset["train"]:
    print(example["text"][:100])
    break

# 数据预处理（map + batched 高效并行）
def preprocess(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
    )

tokenized = dataset.map(preprocess, batched=True, num_proc=8, remove_columns=["text"])
```

### accelerate：多卡训练封装

```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=4,
)

# 一行代码包装模型、优化器、数据加载器
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)      # 替代 loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 启动命令：accelerate launch --num_processes 4 train.py
```

## wandb 实验追踪

Weights & Biases (wandb) 帮助你记录实验过程、比较不同配置、可视化训练曲线。

```python
import wandb

# 初始化实验
wandb.init(
    project="llm-finetune",
    name="llama2-7b-lora-r16",
    config={
        "model": "Llama-2-7b",
        "lora_r": 16,
        "learning_rate": 2e-5,
        "batch_size": 32,
    },
)

# 在训练循环中记录指标
for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    wandb.log({
        "train/loss": loss,
        "train/learning_rate": scheduler.get_last_lr()[0],
        "train/step": step,
    })

# 记录评估结果
wandb.log({"eval/accuracy": accuracy, "eval/loss": eval_loss})

# 结束实验
wandb.finish()
```

## 常用调试工具

### torch.profiler：性能瓶颈分析

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("forward_pass"):
        output = model(inputs)
    with record_function("backward_pass"):
        output.loss.backward()

# 打印最耗时的操作
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# 导出 Chrome trace（可在 chrome://tracing 查看）
prof.export_chrome_trace("trace.json")
```

### 显存监控

```python
# GPU 显存使用情况
print(torch.cuda.memory_summary())           # 详细显存报告
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# 命令行监控
# nvidia-smi                     — 一次性查看 GPU 状态
# watch -n 1 nvidia-smi          — 每秒刷新
# nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 1

# 常见显存 OOM 排查清单：
# 1. 减小 batch_size
# 2. 使用梯度累积代替大 batch
# 3. 开启混合精度训练（bf16/fp16）
# 4. 使用 gradient checkpointing：model.gradient_checkpointing_enable()
# 5. 使用量化（4-bit / 8-bit）
# 6. 检查是否有张量意外保留在 GPU（tensor.detach() / del tensor / torch.cuda.empty_cache()）
```

## 苏格拉底时刻

1. PyTorch 的动态计算图和 TensorFlow 的静态计算图各有什么优劣？为什么大模型社区普遍选择 PyTorch？
2. 当 `DataLoader` 中 `num_workers > 0` 时，数据是如何并行加载的？这与 GPU 计算之间是什么关系？
3. 为什么大模型训练通常使用 `bfloat16` 而非 `float16`？两者的数值范围差异如何影响训练稳定性？
4. Hugging Face 的 `Trainer` 封装了哪些关键步骤？在什么场景下你需要自己写训练循环？
5. `torch.no_grad()` 和 `model.eval()` 的区别是什么？在推理时是否需要同时使用？
6. 梯度累积为什么能模拟大 batch？loss 除以 `accumulation_steps` 的数学原因是什么？

## 推荐资源

- [PyTorch 官方教程](https://pytorch.org/tutorials/) - 从入门到进阶
- [Hugging Face 课程](https://huggingface.co/course) - NLP 全流程实战
- [Practical Deep Learning for Coders](https://course.fast.ai/) - fast.ai 实践课程
- [Dive into Deep Learning](https://d2l.ai/) - 动手学深度学习
- [Hugging Face Accelerate 文档](https://huggingface.co/docs/accelerate) - 多卡训练指南
- [W&B 官方教程](https://docs.wandb.ai/) - 实验追踪最佳实践

### 代码参考

- [MiniMind](https://github.com/jingyaogong/minimind) — **中文社区**最具代表性的**全流程从零训 LLM** 项目，是中文学习者从零训 LLM 的最佳起点。仓库覆盖 Tokenizer 训练、Pretrain、SFT、LoRA、DPO、RLAIF（PPO / GRPO / CISPO）、模型蒸馏与 MoE 全套流程，所有核心算法均用 PyTorch 原生实现，不依赖 `trl` / `peft` 的高层封装。主线模型 `minimind2-small` 仅约 26M 参数，单张 3090 即可在 2 小时内复现 SFT，3 块钱级别的 GPU 租用成本就能跑完一轮完整训练。推荐入门路径：先按 [README](https://github.com/jingyaogong/minimind/blob/master/README.md) 走 quickstart 跑通推理与训练 → 再精读 [`model/model_minimind.py`](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py) 自实现的 Dense + MoE 结构 → 最后顺序对照 [`trainer/train_pretrain.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_pretrain.py)、[`trainer/train_full_sft.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_full_sft.py) 与 [`trainer/train_dpo.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_dpo.py) 三个训练脚本，对应本站「训练」模块的 Pretrain / SFT / Alignment 三阶段。
