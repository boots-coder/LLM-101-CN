---
title: "Llama 实现挑战"
description: "Level 4 完整实现：RoPE、RMSNorm、GQA、SwiGLU、完整 Llama"
topics: [build, Llama, RoPE, RMSNorm, GQA, SwiGLU]
---
# Llama 实现挑战 (Level 3-4)

> 从零构建一个 mini-Llama 模型。先通过热身练习掌握各子模块，再在主挑战中将它们组装为完整的 Decoder 模型。

---

## 热身练习

### 练习 1: RMSNorm 实现（Level 2）

#### 背景

RMSNorm 是 LayerNorm 的简化版，去掉了 re-center（减均值）操作，仅使用 RMS（Root Mean Square）统计量进行缩放。其公式为：

$$
RMS(x) = \sqrt{\frac{1}{N} \sum_i x_i^2}
$$

$$
\text{RMSNorm}(x) = \gamma \cdot \frac{x}{RMS(x)}
$$

RMSNorm 有输入尺度不变性：$\frac{x}{RMS(x)} = \frac{sx}{RMS(sx)}$，且计算比 LayerNorm 更简单（不需要计算均值和方差）。

#### 任务

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-12):
        super().__init__()
        # ===== 填空 1: 定义可学习参数 gamma =====
        self.gamma = _____  # 提示: nn.Parameter, 初始化为全 1
        self.eps = eps

    def forward(self, x):
        # x: [bsz, seq_len, dim]

        # ===== 填空 2: 计算 x^2 在最后一维的均值 =====
        mean_sq = _____  # 提示: (x**2).mean(...)，注意 keepdim

        # ===== 填空 3: 计算 RMS 归一化 =====
        x_normed = _____  # 提示: x / sqrt(mean_sq + eps)

        # ===== 填空 4: 乘以可学习参数 =====
        return _____
```

#### 提示

- `gamma` 是 `dim` 维的向量，初始化为 `torch.ones(dim)`
- 计算均值时 `dim=-1, keepdim=True`，保持维度以便广播
- eps 防止除零

<details>
<summary>参考答案</summary>

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        mean_sq = (x ** 2).mean(-1, keepdim=True)
        x_normed = x / torch.sqrt(mean_sq + self.eps)
        return self.gamma * x_normed
```

**验证:**
```python
norm = RMSNorm(dim=512)
x = torch.randn(2, 6, 512)
y = norm(x)
print(y.shape)       # torch.Size([2, 6, 512])
# 验证 RMS 归一化后统计量为 1
rms = torch.sqrt((y**2).mean(-1))
print(rms.mean())    # 接近 1.0
```

</details>

---

### 练习 2: SwiGLU 实现（Level 2-3）

#### 背景

SwiGLU 将 Swish 激活函数与 GLU（门控线性单元）结合：

$$
h = \text{Swish}(W_{gate}(x)) \odot W_1(x)
$$
$$
y = W_2(h)
$$

其中 Swish 即 `F.silu`（即 `x * sigmoid(x)`），$\odot$ 为逐元素乘法。GLU 门控机制让模型对特征做 token-wise 的细粒度选择。

为保持与标准 FFN 相近的参数量，hidden_dim 设为 `8d/3`（标准 FFN 用 `4d`）。

#### 任务

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # ===== 填空 1: 计算 hidden_dim (保持与 FFN 同等参数量) =====
        hidden_dim = _____  # 提示: 8 * dim // 3

        # ===== 填空 2: 定义三个线性层 (无 bias) =====
        self.w1 = _____      # dim -> hidden_dim
        self.w_gate = _____  # dim -> hidden_dim (门控)
        self.w2 = _____      # hidden_dim -> dim

    def forward(self, x):
        # ===== 填空 3: 门控特征 =====
        gate = _____  # 提示: 对 w_gate(x) 做 silu 激活

        # ===== 填空 4: 上投影特征 =====
        x_up = _____  # 提示: w1(x)

        # ===== 填空 5: GLU 门控乘法 + 下投影 =====
        h = _____  # 提示: gate * x_up
        return _____  # 提示: w2(h)
```

#### 提示

- `F.silu(x)` 等价于 `x * torch.sigmoid(x)`，即 Swish 函数
- 三个 Linear 层均不带 bias（`bias=False`）

<details>
<summary>参考答案</summary>

```python
class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = dim * 8 // 3
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))
        x_up = self.w1(x)
        h = gate * x_up
        return self.w2(h)
```

**验证:**
```python
ffn = SwiGLU(dim=512)
x = torch.randn(2, 3, 512)
print(ffn(x).shape)  # torch.Size([2, 3, 512])

# 对比参数量: FFN(4d) vs SwiGLU(8d/3)
ffn_params = 2 * 512 * 2048  # w1 + w2
swiglu_params = 3 * 512 * 1365  # w1 + w_gate + w2
print(f"FFN: {ffn_params}, SwiGLU: {swiglu_params}")
# 两者参数量接近
```

</details>

---

### 练习 3: GQA 的 KV Repeat（Level 2-3）

#### 背景

GQA（Grouped Query Attention）让多个 Q head 共享同一组 KV head，减少 KV Cache 存储。例如 8 个 Q head 共享 4 个 KV head，每 2 个 Q head 共用 1 个 KV head。

实现时需要将 KV 沿 head 维度复制（repeat），使其与 Q 的 head 数匹配。

#### 任务

```python
import torch

def repeat_kv(
    k: torch.Tensor,   # [bsz, n_kv_heads, seq_len, head_dim]
    v: torch.Tensor,    # [bsz, n_kv_heads, seq_len, head_dim]
    n_rep: int,         # 每个 kv head 需要复制的次数
):
    """
    将 KV 的 head 维度复制 n_rep 次
    例: [bsz, 4, seq_len, head_dim] -> [bsz, 8, seq_len, head_dim] (n_rep=2)
    """
    if n_rep == 1:
        return k, v

    # ===== 填空 1: 使用 repeat_interleave 复制 =====
    k = _____  # 提示: torch.repeat_interleave(k, n_rep, dim=?)
    v = _____  # 提示: 同上，注意 dim 参数

    return k, v


# 验证
bsz, n_kv_heads, seq_len, head_dim = 2, 4, 16, 64
k = torch.randn(bsz, n_kv_heads, seq_len, head_dim)
v = torch.randn(bsz, n_kv_heads, seq_len, head_dim)

k_rep, v_rep = repeat_kv(k, v, n_rep=2)
print(k_rep.shape)  # 应为 torch.Size([2, 8, 16, 64])

# ===== 填空 2: 验证复制正确性 =====
# 第 0 个 kv head 应该等于第 0 和第 1 个 q head 的 kv
assert torch.equal(k_rep[:, 0], _____), "复制不正确"  # 提示: k 的哪个 head?
assert torch.equal(k_rep[:, 1], _____), "复制不正确"  # 提示: 应与 head 0 相同
```

<details>
<summary>参考答案</summary>

```python
# 填空 1
k = torch.repeat_interleave(k, n_rep, dim=1)
v = torch.repeat_interleave(v, n_rep, dim=1)

# 填空 2
assert torch.equal(k_rep[:, 0], k[:, 0])
assert torch.equal(k_rep[:, 1], k[:, 0])  # head 1 也来自原始 kv head 0
```

</details>

---

## 主挑战: 构建完整 Llama Decoder（Level 4）

### 目标

从零实现一个完整的 mini-Llama 模型，包含以下组件：

1. **RMSNorm** -- Pre-Norm 结构
2. **RoPE** -- 旋转位置编码
3. **GQA** -- 分组查询注意力
4. **SwiGLU** -- 门控前馈网络
5. **LlamaDecoderBlock** -- 单个 Decoder 层
6. **LlamaModel** -- 完整模型（Embedding + N 层 Block + LM Head）

### 配置

```python
from dataclasses import dataclass

@dataclass
class LlamaConfig:
    vocab_size: int = 200
    max_len: int = 512
    dim: int = 512
    n_heads: int = 8
    n_kv_heads: int = 4     # GQA: 每 2 个 Q head 共享 1 个 KV head
    num_layers: int = 6
    position_encoding_base: float = 10000.0
    attention_bias: bool = False  # Linear 层不带 bias
```

### 要求

#### Part 1: RoPE（旋转位置编码）

实现两个函数：

```python
def create_rope(max_len, dim, base=10000.0):
    """
    创建 RoPE 的 sin/cos 缓存
    返回: (sin, cos)，形状均为 [max_len, dim]

    思路提示:
    - 频率向量 freqs 形状 [dim/2]：每个分量对应一对维度共享的旋转角速度
    - 用位置 m 和 freqs 做外积，再扩展到全维度即可
    - 详细推导见 [transformer.md 中的 RoPE 章节](/architecture/transformer)
    """
    pass

def apply_rotary_pos_emb(x, cos, sin):
    """
    对 x 施加 RoPE
    x:   [bsz, n_heads, seq_len, head_dim]
    cos: [seq_len, head_dim]
    sin: [seq_len, head_dim]

    思路提示:
    - 把每两维 (x_{2i}, x_{2i+1}) 看作复数，乘以 e^{i m θ_i} 即可完成旋转
    - PyTorch 中可用 view_as_complex / view_as_real 实现，
      也可用实数版本：拆成偶/奇两半，分别做线性组合后再合并

    命名沿用 HuggingFace `modeling_llama.py` 的 `apply_rotary_pos_emb`，
    便于和官方实现对照阅读。
    """
    pass
```

#### Part 2: GroupedQueryAttention

```python
class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力 (与 HuggingFace `LlamaAttention` 同构)

    __init__ 参数（命名参考 HF `modeling_llama.py`）:
    - q_proj: Linear(dim, n_heads * head_dim)         -- Q 投影
    - k_proj: Linear(dim, n_kv_heads * head_dim)      -- K 投影 (KV head 数少于 Q)
    - v_proj: Linear(dim, n_kv_heads * head_dim)      -- V 投影
    - o_proj: Linear(n_heads * head_dim, dim)         -- 输出投影
    - 所有 Linear 不带 bias

    forward(x, mask, cos, sin):
    1. Q, K, V 投影并 reshape 为 multi-head 形式
    2. 对 Q, K 施加 RoPE
    3. K, V 用 repeat_kv 复制到与 Q 相同 head 数 (HF 用 expand+reshape)
    4. Scaled Dot-Product Attention + causal mask
    5. 拼接 + o_proj 输出投影
    """
    pass
```

#### Part 3: LlamaDecoderBlock

```python
class LlamaDecoderBlock(nn.Module):
    """
    Pre-Norm 残差结构:
      X = GQA(RMSNorm(X)) + X
      X = SwiGLU(RMSNorm(X)) + X
    """
    pass
```

#### Part 4: LlamaModel

```python
class LlamaModel(nn.Module):
    """
    完整模型:
    1. Embedding (无位置编码，由 RoPE 在 attention 中引入)
    2. N 个 LlamaDecoderBlock
    3. 最终 RMSNorm
    4. LM Head (Linear, 无 bias)
    5. 缓存 causal mask 和 RoPE sin/cos
    """
    pass
```

### 评估标准

| 项目 | 标准 | 分值 |
|------|------|------|
| RMSNorm | 正确实现，仅有 gamma 参数 | 10 |
| RoPE | create_rope 和 apply_rotary_pos_emb 正确 | 15 |
| GQA | KV repeat + RoPE + Scaled Attention | 25 |
| SwiGLU | Swish 门控 + 参数量对齐 | 10 |
| LlamaDecoderBlock | Pre-Norm 残差结构正确 | 15 |
| LlamaModel | 整体组装、mask、forward 正确 | 15 |
| 代码规范 | 无 bias、head_dim 计算正确 | 10 |
| **总分** | | **100** |

### 测试用例

```python
config = LlamaConfig()
model = LlamaModel(config)

# 测试 1: 基本 forward
input_ids = torch.randint(config.vocab_size, (2, 32))
logits = model(input_ids)
assert logits.shape == (2, 32, config.vocab_size), \
    f"输出形状错误: {logits.shape}"

# 测试 2: 不同序列长度
for seq_len in [1, 16, 64, 128]:
    x = torch.randint(config.vocab_size, (1, seq_len))
    y = model(x)
    assert y.shape == (1, seq_len, config.vocab_size)

# 测试 3: 参数检查 (无 bias)
for name, param in model.named_parameters():
    if 'bias' in name:
        print(f"警告: 发现 bias 参数 {name}")

# 测试 4: RoPE 正确性
sin, cos = create_rope(512, 64)
assert sin.shape == (512, 64)
x = torch.randn(1, 8, 10, 64)
x_rope = apply_rotary_pos_emb(x, cos[:10], sin[:10])
assert x_rope.shape == x.shape

# 测试 5: 可训练
loss = logits.mean()
loss.backward()
print("反向传播成功")

print("所有测试通过!")
```

---

## 参考实现

本练习不提供完整的拼装答案，请独立完成 RoPE / GQA / SwiGLU / LlamaDecoderBlock / LlamaModel 的组装。如需对照官方实现，可参考：

- [meta-llama/llama](https://github.com/meta-llama/llama)（官方 Llama-2 PyTorch 实现）
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- [karpathy/llama2.c](https://github.com/karpathy/llama2.c)（极简 C 实现，便于理解 RoPE 缓存与 KV cache）
- [naklecha/llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch)（92 个 Jupyter cell 单文件从 Meta `consolidated.pth` 起手，**无 `nn.Module`**，按张量级别拆解 Llama-3-8B：cell 28-37 用复数乘法实现 RoPE，cell 39 演示 GQA 的 KV reshape，cell 80 起进入 32 层主循环。和上面"按 module 组装"的练习正好互补——一个看封装，一个看赤裸的张量流）

> 不要参考闭源训练课程的实现。先按上面的子任务一项项跑通，再用 `pytest` 对每个子模块做形状/数值校验。

---

## 进阶思考

完成主挑战后，可以思考以下问题：

1. **KV Cache 集成**: 如何在 GQA 中加入 KV Cache 支持，使其可用于自回归推理？需要修改哪些部分？
2. **RoPE 与 KV Cache 的顺序**: 为什么 RoPE 要在 KV Cache 拼接之前施加？如果在拼接之后施加会怎样？
3. **参数量分析**: 计算模型中 Attention 和 FFN 各占多少参数，理解为什么 MoE 选择扩展 FFN 而非 Attention。
4. **Pre-Norm vs Post-Norm**: Llama 使用 Pre-Norm（先 Norm 再 Attention/FFN），GPT-2 使用 Post-Norm。分析两者在梯度传播上的差异。
5. **Scaling 因子**: 注意力中除以 $\sqrt{d}$ 的作用是什么？如果改为除以 $\sqrt{d_{head}}$ 结果会有什么不同？

---

