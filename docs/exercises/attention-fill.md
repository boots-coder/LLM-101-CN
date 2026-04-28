---
title: "Attention 代码填空"
description: "Level 2 填空：Scaled Dot-Product Attention 和 Multi-Head Attention 实现"
topics: [fill-in, attention, scaled-dot-product, multi-head]
---
# Attention 代码填空 (Level 2)

> **难度:** 中等 | **前置知识:** [注意力机制](/architecture/attention.md) | **预计时间:** 30-45 分钟

本练习包含 3 个代码填空，覆盖 Scaled Dot-Product Attention 和 Multi-Head Attention 的核心实现。每个空白用 `_____` 标记，你需要填入正确的 PyTorch 代码。

建议在本地 IDE 中完成，填入代码后实际运行验证。

---

## 练习 1：Scaled Dot-Product Attention

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    计算 Scaled Dot-Product Attention
    
    参数:
        Q: Query  (batch_size, n_heads, seq_len, d_k)
        K: Key    (batch_size, n_heads, seq_len, d_k)
        V: Value  (batch_size, n_heads, seq_len, d_v)
        mask: 可选的掩码 (batch_size, 1, 1, seq_len) 或 (batch_size, 1, seq_len, seq_len)
    
    返回:
        output: 注意力输出 (batch_size, n_heads, seq_len, d_v)
        attn_weights: 注意力权重 (batch_size, n_heads, seq_len, seq_len)
    """
    d_k = Q.size(-1)
    
    # 空白1: 计算缩放点积注意力分数
    # 提示: Q 和 K 的转置做矩阵乘法，然后除以缩放因子
    scores = _____
    
    if mask is not None:
        # 空白2: 将 mask 为 0 的位置填充为负无穷
        # 提示: 使用 masked_fill，mask == 0 的地方填 -inf
        scores = _____
    
    # 空白3: 对 scores 的最后一个维度做 softmax，得到注意力权重
    attn_weights = _____
    
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```

**提示:**
- 空白1：回忆公式 $\frac{QK^T}{\sqrt{d_k}}$，注意 K 需要转置最后两个维度
- 空白2：`masked_fill` 的第一个参数是布尔条件，第二个参数是填充值
- 空白3：softmax 的 dim 参数决定沿哪个维度归一化

<details>
<summary>查看答案</summary>

```python
# 空白1
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

# 空白2
scores = scores.masked_fill(mask == 0, float('-inf'))

# 空白3
attn_weights = F.softmax(scores, dim=-1)
```

**解析:**
- `K.transpose(-2, -1)` 将 K 的最后两个维度转置，从 `(batch, heads, seq_len, d_k)` 变为 `(batch, heads, d_k, seq_len)`，这样 Q @ K^T 得到 `(batch, heads, seq_len, seq_len)` 的注意力分数矩阵
- `masked_fill` 将被遮蔽的位置设为 $-\infty$，经过 softmax 后这些位置的权重变为 0
- `dim=-1` 表示对最后一个维度（Key 的序列维度）做归一化，确保每个 Query 对所有 Key 的注意力权重之和为 1

</details>

---

## 练习 2：MultiHeadAttention.__init__

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        """
        参数:
            d_model: 模型隐藏维度 (例如 512)
            n_heads: 注意力头数 (例如 8)
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # 空白1: 计算每个头的维度
        self.d_k = _____
        
        # 空白2: 定义 Q、K、V 的线性投影层
        # 提示: 输入维度是 d_model，输出维度也是 d_model（包含所有头）
        self.W_q = _____
        self.W_k = _____
        self.W_v = _____
        
        # 空白3: 定义输出投影层（将拼接后的多头结果映射回 d_model）
        self.W_o = _____
    
    # forward 方法见练习 3
```

**提示:**
- 空白1：总维度平均分配到每个头
- 空白2：`nn.Linear` 的输入和输出维度分别是什么？
- 空白3：多头拼接后的维度是多少？

<details>
<summary>查看答案</summary>

```python
# 空白1
self.d_k = d_model // n_heads

# 空白2
self.W_q = nn.Linear(d_model, d_model)
self.W_k = nn.Linear(d_model, d_model)
self.W_v = nn.Linear(d_model, d_model)

# 空白3
self.W_o = nn.Linear(d_model, d_model)
```

**解析:**
- `d_k = d_model // n_heads`：如 d_model=512, n_heads=8, 则每个头的维度 d_k=64
- 投影层的输入是 d_model，输出也是 d_model。虽然每个头只用 d_k 维度，但我们把所有头的投影合并到一个线性层中（输出 n_heads * d_k = d_model），之后再 reshape 拆分为多个头
- 输出投影层将多头拼接的结果（维度仍是 d_model）映射回 d_model

</details>

---

## 练习 3：MultiHeadAttention.forward

```python
def forward(self, Q, K, V, mask=None):
    """
    参数:
        Q, K, V: (batch_size, seq_len, d_model)
        mask: 可选掩码
    
    返回:
        output: (batch_size, seq_len, d_model)
    """
    batch_size = Q.size(0)
    
    # 步骤1: 线性投影
    Q = self.W_q(Q)  # (batch_size, seq_len, d_model)
    K = self.W_k(K)
    V = self.W_v(V)
    
    # 步骤2: 拆分为多头
    # 空白1: 将 (batch_size, seq_len, d_model) 变形为 (batch_size, n_heads, seq_len, d_k)
    # 提示: 先 view/reshape 再 transpose
    Q = _____
    K = _____
    V = _____
    
    # 步骤3: 计算注意力（使用练习1的函数）
    attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
    # attn_output: (batch_size, n_heads, seq_len, d_k)
    
    # 步骤4: 拼接多头结果
    # 空白2: 将 (batch_size, n_heads, seq_len, d_k) 变回 (batch_size, seq_len, d_model)
    # 提示: 先 transpose 再 contiguous 再 view/reshape
    attn_output = _____
    
    # 步骤5: 输出投影
    # 空白3: 通过输出投影层
    output = _____
    
    return output
```

**提示:**
- 空白1：`view(batch_size, -1, self.n_heads, self.d_k)` 然后 `transpose(1, 2)`
- 空白2：是空白1的逆操作
- 空白3：一个简单的线性变换

<details>
<summary>查看答案</summary>

```python
# 空白1
Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

# 空白2
attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

# 空白3
output = self.W_o(attn_output)
```

**解析:**

**空白1（拆分多头）的 shape 变化:**
```
(batch, seq_len, d_model)
    → view → (batch, seq_len, n_heads, d_k)
    → transpose(1,2) → (batch, n_heads, seq_len, d_k)
```

**空白2（拼接多头）的 shape 变化:**
```
(batch, n_heads, seq_len, d_k)
    → transpose(1,2) → (batch, seq_len, n_heads, d_k)
    → contiguous → 确保内存连续
    → view → (batch, seq_len, d_model)   # n_heads * d_k = d_model
```

`contiguous()` 是必要的，因为 `transpose` 只改变了 tensor 的 stride 而没有重新排列内存，后续的 `view` 要求内存连续。

**空白3** 就是一个简单的线性投影，将拼接后的多头结果映射到最终的输出空间。

</details>

---

## 验证代码

完成所有填空后，用以下代码验证你的实现：

```python
# 测试
batch_size, seq_len, d_model, n_heads = 2, 10, 512, 8

mha = MultiHeadAttention(d_model, n_heads)
x = torch.randn(batch_size, seq_len, d_model)

# 自注意力：Q=K=V=x
output = mha(x, x, x)
print(f"输入 shape: {x.shape}")
print(f"输出 shape: {output.shape}")
assert output.shape == (batch_size, seq_len, d_model), "Shape 不匹配！"

# 带因果掩码
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
output_masked = mha(x, x, x, mask=causal_mask)
print(f"带掩码输出 shape: {output_masked.shape}")
print("所有测试通过！")
```

---

## MLM 代码训练模式

完成上面的固定填空后，试试随机挖空模式 -- 每次点击「刷新」会随机遮盖不同的代码片段，帮你彻底记住每一行。

### Scaled Dot-Product Attention

<CodeMasker title="Attention 随机训练" :mask-ratio="0.15">
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
</CodeMasker>

### Multi-Head Attention

<CodeMasker title="MultiHeadAttention 随机训练" :mask-ratio="0.15">
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        output = self.W_o(attn_output)
        return output
</CodeMasker>
