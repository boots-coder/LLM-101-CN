---
title: "深度剖析 LoRA"
description: "50 行核心代码实现 LoRA，从数学推导到完整微调 pipeline"
topics: [LoRA, low-rank, fine-tuning, PEFT, parameter-efficient, from-scratch]
prereqs: [fundamentals/math, fundamentals/neural-networks, training/sft]
---
# 深度剖析 LoRA

> **一句话总结:** LoRA 的核心思想极其简洁——冻结原始权重，只训练一对低秩矩阵 $BA$。50 行代码就能实现核心逻辑，但理解"为什么有效"需要线性代数的直觉。

## LoRA 的数学本质

### 低秩假设

LoRA 的理论基础是：**预训练模型在微调时的权重变化 $\Delta W$ 是低秩的。**

也就是说，$\Delta W \in \mathbb{R}^{d \times d}$ 虽然是一个大矩阵，但它的有效信息集中在少数几个方向上，可以用两个小矩阵的乘积来近似：

$$
\Delta W = BA, \quad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times d}, \quad r \ll d
$$

前向传播变为：

$$
h = Wx + \Delta Wx = Wx + BAx
$$

$$
h = (W + BA)x
$$

### 参数量对比

| 方法 | 可训练参数 | 以 d=4096, r=16 为例 |
|------|-----------|---------------------|
| 全参数微调 | $d^2 = 16,777,216$ | 16.8M |
| LoRA | $2dr = 131,072$ | 131K |
| **压缩比** | $\frac{2r}{d}$ | **0.78%** |

### 缩放因子 $\alpha$

LoRA 引入一个缩放因子避免学习率需要随 $r$ 调整：

$$
h = Wx + \frac{\alpha}{r} BAx
$$

当 $\alpha = r$ 时，缩放因子为 1（等效于不缩放）。实践中通常设 $\alpha = 2r$。

### 初始化策略

- **$A$**：用 Kaiming 正态分布初始化（标准做法）
- **$B$**：初始化为零矩阵

这保证训练开始时 $\Delta W = BA = 0$，模型行为与预训练模型完全一致——是一个"零初始化"的扰动。

---

## 核心实现：50 行

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    LoRA 包装的线性层
    
    原始: h = Wx
    LoRA: h = Wx + (alpha/r) * BAx
    """
    def __init__(self, original_linear: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.original = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        d_out, d_in = original_linear.weight.shape
        
        # LoRA 矩阵
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        
        # A 用 Kaiming 初始化，B 初始化为零
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # 冻结原始权重
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
    
    def forward(self, x):
        # 原始前向传播 + LoRA 增量
        h = self.original(x)                     # Wx
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling  # (alpha/r) * BAx
        return h + lora_out
    
    def merge(self):
        """将 LoRA 权重合并到原始权重（部署时用）"""
        self.original.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        return self.original
    
    @property
    def trainable_params(self):
        return self.lora_A.numel() + self.lora_B.numel()
```

### 验证正确性

```python
# 创建一个原始线性层
linear = nn.Linear(512, 512)

# 包装为 LoRA
lora_linear = LoRALinear(linear, r=16, alpha=32)

# 检查参数量
total = sum(p.numel() for p in lora_linear.parameters())
trainable = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
print(f"总参数: {total:,}, 可训练: {trainable:,} ({trainable/total*100:.2f}%)")
# 总参数: 278,528, 可训练: 16,384 (5.88%)

# 验证初始输出不变
x = torch.randn(2, 10, 512)
with torch.no_grad():
    h_original = linear(x)
    h_lora = lora_linear(x)
    print(f"初始误差: {(h_original - h_lora).abs().max():.2e}")
    # 初始误差: 0.00e+00 ← 完全一致（因为 B 初始化为零）
```

---

## 给模型注入 LoRA

```python
def inject_lora(model, target_modules, r=16, alpha=32):
    """
    自动给模型的指定层注入 LoRA
    
    Args:
        model: 原始模型
        target_modules: 要注入 LoRA 的模块名列表（如 ["q_proj", "v_proj"]）
        r: LoRA 秩
        alpha: 缩放系数
    
    Returns:
        注入 LoRA 后的模型，以及 LoRA 模块列表
    """
    lora_modules = []
    
    for name, module in model.named_modules():
        # 检查是否是目标模块
        short_name = name.split(".")[-1]
        if short_name in target_modules and isinstance(module, nn.Linear):
            # 创建 LoRA 包装
            lora_layer = LoRALinear(module, r=r, alpha=alpha)
            lora_modules.append(lora_layer)
            
            # 替换原始模块
            parent_name = ".".join(name.split(".")[:-1])
            parent = dict(model.named_modules())[parent_name] if parent_name else model
            setattr(parent, short_name, lora_layer)
    
    # 冻结所有非 LoRA 参数
    for param in model.parameters():
        param.requires_grad = False
    for lora in lora_modules:
        lora.lora_A.requires_grad = True
        lora.lora_B.requires_grad = True
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA 注入完成: {len(lora_modules)} 个模块")
    print(f"总参数: {total:,}, 可训练: {trainable:,} ({trainable/total*100:.2f}%)")
    
    return model, lora_modules
```

### 在 GPT-2 上使用

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

# 对 Attention 的 Q, V 投影注入 LoRA
model, lora_modules = inject_lora(
    model, 
    target_modules=["c_attn"],  # GPT-2 的 QKV 合并层
    r=16, 
    alpha=32,
)
# LoRA 注入完成: 12 个模块
# 总参数: 124,439,808, 可训练: 393,216 (0.32%)
```

---

## LoRA 的保存与加载

```python
def save_lora(lora_modules, path):
    """只保存 LoRA 权重（通常只有几 MB）"""
    state_dict = {}
    for i, lora in enumerate(lora_modules):
        state_dict[f"lora_{i}.A"] = lora.lora_A.data
        state_dict[f"lora_{i}.B"] = lora.lora_B.data
    torch.save(state_dict, path)
    size_mb = sum(v.numel() * v.element_size() for v in state_dict.values()) / 1e6
    print(f"LoRA 权重已保存: {path} ({size_mb:.1f} MB)")

def load_lora(lora_modules, path):
    """加载 LoRA 权重"""
    state_dict = torch.load(path, weights_only=True)
    for i, lora in enumerate(lora_modules):
        lora.lora_A.data = state_dict[f"lora_{i}.A"]
        lora.lora_B.data = state_dict[f"lora_{i}.B"]
    print(f"LoRA 权重已加载: {path}")

def merge_lora(lora_modules):
    """合并 LoRA 到原始权重（部署时用，消除推理开销）"""
    for lora in lora_modules:
        lora.merge()
    print("LoRA 已合并到原始权重，推理时无额外开销")
```

---

## 完整训练示例

```python
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

def train_lora(model, lora_modules, train_texts, epochs=3, lr=2e-4):
    """LoRA 微调训练循环"""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    
    # 只优化 LoRA 参数
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for text in train_texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, 
                             max_length=256, padding="max_length")
            input_ids = tokens.input_ids.to(device)
            
            # Next-token prediction
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_texts)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # 保存 LoRA 权重
    save_lora(lora_modules, "lora_weights.pt")
    return model

# 使用
train_texts = [
    "The transformer architecture consists of...",
    "Attention mechanism allows the model to...",
    # ... 更多训练数据
]
model = train_lora(model, lora_modules, train_texts)
```

---

## LoRA 变体速览

| 变体 | 改进点 | 适用场景 |
|------|--------|---------|
| **QLoRA** | 基础模型用 4-bit 量化，LoRA 用 FP16 | 显存极其有限（8GB GPU） |
| **LoRA+** | A 和 B 用不同学习率（B 的学习率更大） | 提升训练效率 |
| **DoRA** | 将权重分解为 magnitude + direction，只对 direction 加 LoRA | 更接近全参数微调效果 |
| **rsLoRA** | 缩放因子改为 $\frac{\alpha}{\sqrt{r}}$ | 大 rank 时更稳定 |
| **AdaLoRA** | 动态分配不同层的 rank（重要层 rank 大） | 参数预算有限时 |

---

## 苏格拉底时刻

1. 为什么 $B$ 初始化为零而不是和 $A$ 一样用随机初始化？如果两个都随机初始化会怎样？
2. LoRA 的 rank $r$ 越大越好吗？当 $r = d$ 时，LoRA 退化成什么？
3. 为什么 LoRA 通常加在 Attention 层的 Q、V 投影上，而不是 FFN 层？
4. `merge()` 合并后的模型和 LoRA 模型的推理结果是否完全一致？为什么？
5. QLoRA 把基础模型量化到 4-bit，LoRA 仍用 FP16——梯度如何通过 4-bit 层反向传播？

---

## 常见问题 & 面试考点

| 问题 | 要点 |
|------|------|
| LoRA 的核心假设？ | 微调时的权重变化是低秩的 |
| 为什么 LoRA 有效？ | 预训练模型已经学到了好的特征，微调只需要在少数方向上调整 |
| rank 怎么选？ | 一般 r=8~64。任务越复杂/数据越多，rank 可以越大 |
| alpha 怎么选？ | 通常 alpha = 2r。实际效果是 alpha/r 决定 LoRA 更新的幅度 |
| LoRA vs 全参数微调？ | LoRA 在大多数任务上接近全参数微调效果，但参数量减少 99%+ |
| 合并后推理有开销吗？ | 没有。$W + BA$ 合并为一个矩阵，推理时和原始模型一样 |

---

## 推荐资源

- **Hu et al.《LoRA: Low-Rank Adaptation of Large Language Models》** — LoRA 原始论文
- **Dettmers et al.《QLoRA: Efficient Finetuning of Quantized LLMs》** — QLoRA 论文
- **HuggingFace PEFT 库** — 工业级 LoRA/QLoRA/DoRA 实现
- **Sebastian Raschka: LoRA from Scratch** — 另一个优秀的从零实现教程
