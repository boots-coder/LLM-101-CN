---
title: "Flash Attention 填空"
description: "Level 2-3 填空：在线 Softmax、分块计算、内存优化"
topics: [fill-in, flash-attention, online-softmax, tiling, memory-efficient]
---
# Flash Attention 代码填空 (Level 2-3)

> 本练习覆盖 Flash Attention 的核心技术：标准 Attention 的显存瓶颈分析、Safe / 在线 Softmax、分块矩阵计算、Flash Attention Forward 完整实现、反向传播的核心洞察。
> 代码基于纯 PyTorch 实现，用 `_____` 标记需要填写的部分。

---

## 练习 1: 标准 Attention 的内存分析（Level 1-2）

### 背景

标准 Scaled Dot-Product Attention 的计算流程如下：

$$S = QK^T / \sqrt{d} \quad \rightarrow \quad P = \text{softmax}(S) \quad \rightarrow \quad O = PV$$

其中 $Q, K, V \in \mathbb{R}^{N \times d}$，$N$ 是序列长度，$d$ 是每个头的维度。在训练时，中间变量 $S$ 和 $P$ 都需要保留用于反向传播。

以下代码实现了标准 Attention 并计算各中间变量的显存占用：

```python
import torch
import math

def standard_attention(Q, K, V):
    """标准 Attention，返回所有中间变量"""
    d = Q.shape[-1]
    S = Q @ K.T / math.sqrt(d)   # [N, N]
    P = torch.softmax(S, dim=-1)  # [N, N]
    O = P @ V                     # [N, d]
    return O, S, P

# 参数设置
N = 4096       # 序列长度
d = 64         # 每个头的维度
bytes_per_float = 4  # float32 = 4 字节
```

### 任务

计算各中间变量的显存占用（单位：字节），填写下面的表达式：

```python
# ===== 填空 1: Q 的显存占用 =====
mem_Q = _____  # 提示: Q 的 shape 是 [N, d]

# ===== 填空 2: S 矩阵 (QK^T) 的显存占用 =====
mem_S = _____  # 提示: S 的 shape 是 [N, N]

# ===== 填空 3: P 矩阵 (softmax 输出) 的显存占用 =====
mem_P = _____  # 提示: P 的 shape 与 S 相同

# ===== 填空 4: 标准 Attention 中间变量总显存 (S + P) =====
mem_intermediate = _____

# ===== 填空 5: 当 N=4096, d=64, float32 时，中间变量总显存是多少 MB？ =====
mem_intermediate_MB = _____  # 提示: 1 MB = 1024 * 1024 字节

print(f"Q 显存:     {mem_Q / 1024**2:.2f} MB")
print(f"S 显存:     {mem_S / 1024**2:.2f} MB")
print(f"P 显存:     {mem_P / 1024**2:.2f} MB")
print(f"中间变量:   {mem_intermediate / 1024**2:.2f} MB")
print(f"中间变量:   {mem_intermediate_MB:.2f} MB")
print(f"\nQ 显存与 S 显存的比值: 1:{mem_S / mem_Q:.0f}")
print(f"=> 当 N >> d 时，S 和 P 占据绝大部分显存，这就是 O(N^2) 瓶颈")
```

### 提示

- 一个 `[rows, cols]` 的 float32 张量占用 `rows * cols * 4` 字节
- 标准 Attention 需要同时存储 $S$ 和 $P$ 两个 $N \times N$ 的矩阵
- 当 $N = 4096$ 时，单个 $N \times N$ 矩阵就占 64 MB

<details>
<summary>参考答案</summary>

```python
# 填空 1
mem_Q = N * d * bytes_per_float                    # 4096 * 64 * 4 = 1 MB

# 填空 2
mem_S = N * N * bytes_per_float                    # 4096 * 4096 * 4 = 64 MB

# 填空 3
mem_P = N * N * bytes_per_float                    # 与 S 相同 = 64 MB

# 填空 4
mem_intermediate = 2 * N * N * bytes_per_float     # S + P = 128 MB

# 填空 5
mem_intermediate_MB = 2 * N * N * bytes_per_float / (1024 * 1024)  # 128.0
```

**解析:**

- $Q, K, V$ 各占 $N \times d \times 4$ 字节，当 $d=64$ 时仅 1 MB
- $S$ 和 $P$ 各占 $N \times N \times 4$ 字节，当 $N=4096$ 时各 64 MB
- 中间变量总显存 128 MB，是输入的 128 倍 -- 这就是标准 Attention 的 $O(N^2)$ 显存瓶颈
- Flash Attention 的核心目标就是消除对 $S$ 和 $P$ 的完整存储

</details>

---

## 练习 2: Safe Softmax 与在线 Softmax（Level 2）

### 背景

Softmax 的朴素公式 $\text{softmax}(x_i) = e^{x_i} / \sum_j e^{x_j}$ 存在数值溢出问题：当 $x_i$ 很大时 $e^{x_i}$ 会溢出为 `inf`。

**Safe softmax** 通过减去最大值解决：$\text{softmax}(x_i) = e^{x_i - m} / \sum_j e^{x_j - m}$，其中 $m = \max(x)$。

但 safe softmax 需要两遍扫描（第一遍求 max，第二遍算 softmax）。**在线 softmax**（Online Softmax）只需一遍扫描：在遍历过程中同时维护 running max $m$ 和 running sum $l$，当遇到更大的值时对已有的 sum 进行修正。

在线 softmax 的更新规则（逐元素处理 $x_j$）：

$$m_{\text{new}} = \max(m_{\text{old}}, x_j)$$
$$l_{\text{new}} = l_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + e^{x_j - m_{\text{new}}}$$

最终 $\text{softmax}(x_i) = e^{x_i - m} / l$。

### 任务

```python
import torch

def naive_softmax(x):
    """朴素 softmax（可能溢出）"""
    e = torch.exp(x)
    return e / e.sum(dim=-1, keepdim=True)

def safe_softmax(x):
    """数值稳定的 softmax"""
    # ===== 填空 1: 减去最大值后计算 softmax =====
    m = _____                       # 提示: 沿最后一维取 max，保持维度
    e = _____                       # 提示: exp(x - m)
    return e / e.sum(dim=-1, keepdim=True)

def online_softmax(x):
    """
    在线 softmax: 单次遍历完成
    x: [rows, cols]
    返回与 safe_softmax 相同的结果
    """
    rows, cols = x.shape
    output = torch.zeros_like(x)

    for i in range(rows):
        row = x[i]
        m = float('-inf')   # running max
        l = 0.0             # running sum of exp

        # 第一遍: 在线计算 max 和 sum
        for j in range(cols):
            # ===== 填空 2: 在线更新 max 和 sum =====
            m_new = _____                    # 提示: max(m, row[j])
            l = _____                        # 提示: 修正旧的 l 并加上新项
            m = m_new

        # 第二遍: 用最终的 m 和 l 计算输出
        for j in range(cols):
            # ===== 填空 3: 计算最终的 softmax 值 =====
            output[i, j] = _____             # 提示: exp(row[j] - m) / l

    return output

# 验证
torch.manual_seed(42)
x = torch.randn(4, 8) * 10  # 用较大的值测试数值稳定性

ref = torch.softmax(x, dim=-1)
out_safe = safe_softmax(x)
out_online = online_softmax(x)

print(f"safe_softmax   误差: {(out_safe - ref).abs().max().item():.2e}")
print(f"online_softmax 误差: {(out_online - ref).abs().max().item():.2e}")
assert (out_safe - ref).abs().max() < 1e-6, "safe_softmax 结果不正确"
assert (out_online - ref).abs().max() < 1e-5, "online_softmax 结果不正确"
print("验证通过!")
```

### 提示

- 填空 1：`x.max(dim=-1, keepdim=True).values` 获取每行的最大值
- 填空 2：在线更新的关键 -- 当 max 变大时，旧的 sum 需要乘以 $e^{m_{\text{old}} - m_{\text{new}}}$ 进行修正
- 填空 3：标准的 softmax 公式，使用最终的 $m$ 和 $l$

<details>
<summary>参考答案</summary>

```python
# 填空 1
m = x.max(dim=-1, keepdim=True).values
e = torch.exp(x - m)

# 填空 2
m_new = max(m, row[j].item())
l = l * math.exp(m - m_new) + math.exp(row[j].item() - m_new)

# 填空 3
output[i, j] = math.exp(row[j].item() - m) / l
```

需要在文件顶部 `import math`。

**解析:**

- **Safe softmax** 是最基础的数值稳定技巧，PyTorch 的 `torch.softmax` 内部就是这样实现的
- **在线 softmax** 的核心洞察：当 $m$ 更新时，之前累积的 $l$ 需要乘以修正因子 $e^{m_{\text{old}} - m_{\text{new}}}$。因为 $m_{\text{new}} \geq m_{\text{old}}$，所以修正因子 $\leq 1$，不会溢出
- 在线 softmax 是 Flash Attention 的数学基础 -- 它允许我们在分块处理 $K$ 时，逐块更新 softmax 的统计量

</details>

---

## 练习 3: 分块矩阵乘法（Level 2）

### 背景

矩阵乘法可以分块进行：将 $Q \in \mathbb{R}^{N \times d}$ 按行分成若干大小为 $B$ 的块 $Q_1, Q_2, \ldots$，$K$ 和 $V$ 同理。标准 Attention 可以等价地写成分块形式：

$$O_i = \text{softmax}(Q_i K^T / \sqrt{d}) \cdot V$$

更进一步，$K$ 和 $V$ 也可以分块：

$$S_{ij} = Q_i K_j^T / \sqrt{d}$$

但注意，当 $K$ 分块时 softmax 不能直接分块计算（因为 softmax 需要全局的 max 和 sum）。本练习先用"拼接后统一 softmax"的方式验证分块的正确性，下一练习再引入在线 softmax。

### 任务

```python
import torch
import math

def standard_attention(Q, K, V):
    """标准 Attention（作为参考）"""
    d = Q.shape[-1]
    S = Q @ K.T / math.sqrt(d)
    P = torch.softmax(S, dim=-1)
    O = P @ V
    return O

def blocked_attention(Q, K, V, block_size):
    """
    分块 Attention: 将 Q, K, V 按行分块，结果与标准 Attention 等价
    Q, K, V: [N, d]
    block_size: 块大小 B（假设 N 能被 B 整除）
    """
    N, d = Q.shape
    O = torch.zeros(N, d)
    num_blocks = N // block_size

    for i in range(num_blocks):
        # ===== 填空 1: 取出第 i 个 Q 块 =====
        q_i = _____  # 提示: Q 的第 i*block_size 到 (i+1)*block_size 行

        # 收集当前 Q 块与所有 K 块的注意力分数
        scores_row = []
        for j in range(num_blocks):
            # ===== 填空 2: 取出第 j 个 K 块，计算 S_ij =====
            k_j = _____
            s_ij = _____  # 提示: q_i @ k_j^T / sqrt(d)
            scores_row.append(s_ij)

        # ===== 填空 3: 拼接所有 K 块的分数，做完整的 softmax =====
        S_i = _____                  # 提示: torch.cat, 沿列方向拼接
        P_i = _____                  # 提示: softmax, dim=-1

        # ===== 填空 4: 用 P_i 与完整的 V 计算输出 =====
        O[i * block_size:(i + 1) * block_size] = _____  # 提示: P_i @ V

    return O

# 验证
torch.manual_seed(42)
N, d = 128, 64
Q = torch.randn(N, d)
K = torch.randn(N, d)
V = torch.randn(N, d)

ref = standard_attention(Q, K, V)
out = blocked_attention(Q, K, V, block_size=32)

diff = (out - ref).abs().max().item()
print(f"分块 Attention 与标准 Attention 的最大误差: {diff:.2e}")
assert diff < 1e-5, f"误差过大: {diff}"
print("验证通过!")
```

### 提示

- 分块只是把矩阵按行切片：`Q[start:end]`
- 这里的分块只对 $Q$ 分块做外层循环，$K$ 分块后拼接 scores 做统一 softmax，最后乘以完整的 $V$
- 关键理解：只要 softmax 是在完整的一行上做的，结果就和标准 Attention 完全一致

<details>
<summary>参考答案</summary>

```python
# 填空 1
q_i = Q[i * block_size:(i + 1) * block_size]

# 填空 2
k_j = K[j * block_size:(j + 1) * block_size]
s_ij = q_i @ k_j.T / math.sqrt(d)

# 填空 3
S_i = torch.cat(scores_row, dim=-1)         # [block_size, N]
P_i = torch.softmax(S_i, dim=-1)            # [block_size, N]

# 填空 4
O[i * block_size:(i + 1) * block_size] = P_i @ V
```

**解析:**

- 分块的本质是把 $N \times N$ 的大矩阵运算拆成若干 $B \times B$ 的小块
- 本练习中 softmax 仍然是对完整的一行做的（拼接后再 softmax），所以结果精确等价
- Flash Attention 的突破在于：结合在线 softmax，不需要拼接就能逐块更新，从而避免存储完整的 $S$ 矩阵

</details>

---

## 练习 4: Flash Attention Forward（Level 3）

### 背景

Flash Attention 将在线 softmax 与分块计算结合，实现了 $O(N)$ 显存的 Attention 计算。其核心思路是：

1. 外层循环遍历 $K, V$ 的分块（索引 $j$）
2. 内层循环遍历 $Q$ 的分块（索引 $i$）
3. 对每个 $(i, j)$ 块，计算局部注意力分数 $S_{ij}$
4. 用在线 softmax 更新统计量（$m_i$: 行最大值，$l_i$: 行 exp-sum）
5. 用 rescale 技巧逐步累积输出 $O_i$

算法伪代码（来自 Flash Attention 论文 Algorithm 1）：

```
初始化 O = 0, l = 0, m = -inf
for j = 1 to T_k:                    # 遍历 K/V 块
    加载 K_j, V_j
    for i = 1 to T_q:                # 遍历 Q 块
        加载 Q_i, O_i, l_i, m_i
        S_ij = Q_i @ K_j^T / sqrt(d)
        m_ij = rowmax(S_ij)
        m_new = max(m_i, m_ij)
        P_ij = exp(S_ij - m_new)
        l_new = l_i * exp(m_i - m_new) + rowsum(P_ij)
        O_i = O_i * (l_i * exp(m_i - m_new) / l_new) + P_ij @ V_j / l_new
        l_i = l_new
        m_i = m_new
```

### 任务

```python
import torch
import math

def flash_attention_forward(Q, K, V, block_size):
    """
    Flash Attention Forward Pass (单头, 无 mask)
    Q, K, V: [N, d]
    block_size: 块大小 B（假设 N 能被 B 整除）
    返回: O [N, d]
    """
    N, d = Q.shape
    num_blocks = N // block_size

    # 初始化输出和统计量
    O = torch.zeros(N, d)
    l = torch.zeros(N, 1)           # 每行的 exp-sum
    m = torch.full((N, 1), float('-inf'))  # 每行的 max

    # ===== 填空 1: 外层循环遍历 K/V 块, 内层循环遍历 Q 块 =====
    for j in range(_____):              # 提示: 遍历 K/V 块
        # 取出第 j 个 K, V 块
        k_j = K[j * block_size:(j + 1) * block_size]   # [B, d]
        v_j = V[j * block_size:(j + 1) * block_size]   # [B, d]

        for i in range(_____):          # 提示: 遍历 Q 块
            # 取出第 i 个 Q 块及其对应的统计量
            q_start = i * block_size
            q_end = (i + 1) * block_size
            q_i = Q[q_start:q_end]           # [B, d]
            o_i = O[q_start:q_end]           # [B, d]
            l_i = l[q_start:q_end]           # [B, 1]
            m_i = m[q_start:q_end]           # [B, 1]

            # ===== 填空 2: 计算当前块的注意力分数 =====
            s_ij = _____  # 提示: q_i @ k_j^T / sqrt(d), shape [B, B]

            # ===== 填空 3: 在线更新 max 和 sum =====
            m_ij = s_ij.max(dim=-1, keepdim=True).values   # 当前块的行最大值 [B, 1]
            m_new = _____                                   # 提示: 新 max = max(旧 max, 当前块 max)
            p_ij = _____                                    # 提示: exp(s_ij - m_new)
            l_new = _____                                   # 提示: 修正旧 l + 当前块的 rowsum

            # ===== 填空 4: rescale 旧的 O 并累积新的贡献 =====
            # 核心公式: O_new = (l_old * exp(m_old - m_new) * O_old + P_ij @ V_j) / l_new
            alpha = _____                                   # 提示: l_i * exp(m_i - m_new) / l_new
            beta = _____                                    # 提示: 1 / l_new (或 p_ij / l_new)
            o_i = alpha * o_i + beta * (p_ij @ v_j)

            # 写回
            O[q_start:q_end] = o_i
            l[q_start:q_end] = l_new
            m[q_start:q_end] = m_new

    # ===== 填空 5: 最终 normalize（如果上面的实现已经 normalize 了，这里无需操作）=====
    # 如果上面每一步都做了 /l_new，则 O 已经是最终结果
    # 如果上面只累积了未归一化的值，则需要: O = O / l
    return O

# ===== 验证 =====
torch.manual_seed(42)
N, d = 256, 64
Q = torch.randn(N, d)
K = torch.randn(N, d)
V = torch.randn(N, d)

# 标准 Attention
ref = torch.softmax(Q @ K.T / math.sqrt(d), dim=-1) @ V

# Flash Attention
out = flash_attention_forward(Q, K, V, block_size=32)

diff = (out - ref).abs().max().item()
print(f"Flash Attention 与标准 Attention 的最大误差: {diff:.2e}")
assert diff < 1e-4, f"误差过大: {diff}"
print("验证通过! Flash Attention 结果正确")

# 打印显存对比
print(f"\n标准 Attention 中间变量: {2 * N * N * 4 / 1024:.1f} KB (两个 N*N 矩阵)")
print(f"Flash Attention 额外显存: ~{2 * N * 4 / 1024:.1f} KB (仅 l 和 m 向量)")
```

### 提示

- 填空 1：外层 $j$ 遍历 K/V 块，内层 $i$ 遍历 Q 块，总共 `num_blocks` 个块
- 填空 2：标准的缩放点积，注意 `k_j.T` 的转置
- 填空 3：`m_new = torch.max(m_i, m_ij)`；修正旧的 $l$：`l_i * exp(m_i - m_new)`，加上当前块 `p_ij.sum(dim=-1, keepdim=True)`
- 填空 4：`alpha` 是旧输出的缩放因子，`beta` 是新贡献的缩放因子；关键是保证最终 $O$ 被正确归一化
- 填空 5：如果每步都做了 `/ l_new`，则 `O` 已归一化，直接返回即可

<details>
<summary>参考答案</summary>

```python
# 填空 1
for j in range(num_blocks):
    ...
    for i in range(num_blocks):

# 填空 2
s_ij = q_i @ k_j.T / math.sqrt(d)

# 填空 3
m_new = torch.max(m_i, m_ij)
p_ij = torch.exp(s_ij - m_new)
l_new = l_i * torch.exp(m_i - m_new) + p_ij.sum(dim=-1, keepdim=True)

# 填空 4
alpha = l_i * torch.exp(m_i - m_new) / l_new
beta = 1.0 / l_new
o_i = alpha * o_i + beta * (p_ij @ v_j)

# 填空 5
# O 已经在每一步被 normalize（除以 l_new），无需额外操作
return O
```

**解析:**

1. **循环顺序**：外层遍历 K/V 块、内层遍历 Q 块。这样每个 K/V 块只从 HBM 加载一次，被所有 Q 块复用，最大化数据局部性

2. **在线 softmax 更新**：
   - `m_new = max(m_i, m_ij)` -- 全局 max 只可能增大
   - `l_new = l_i * exp(m_i - m_new) + sum(exp(s_ij - m_new))` -- 旧的 sum 需要用修正因子调整
   - 当 `m_new > m_i` 时，修正因子 `exp(m_i - m_new) < 1`，将旧的 sum 缩小

3. **Rescale 技巧**：
   - `alpha = l_i * exp(m_i - m_new) / l_new` 将旧输出从"旧的归一化基"转换到"新的归一化基"
   - `beta = 1.0 / l_new` 将新贡献归一化
   - 这保证了 $O$ 在任何中间状态都是正确归一化的

4. **显存优化**：整个过程只需存储 $O$（$N \times d$）和统计量 $l, m$（各 $N \times 1$），无需存储 $N \times N$ 的 $S$ 或 $P$ 矩阵

</details>

---

## 练习 5: Flash Attention 反向传播的核心洞察（Level 3）

### 背景

Flash Attention 的反向传播面临一个问题：标准反向传播需要存储的 $P$ 矩阵（$N \times N$）在前向传播中被丢弃了（这正是 Flash Attention 节省显存的方式）。

Flash Attention 的解决方案是 **重新计算**（Recomputation）：

- 前向传播时只保存 $O$、$l$（row-sum）和 $m$（row-max）
- 反向传播时利用 $l, m$ 重新计算 $S_{ij}$ 和 $P_{ij}$（分块进行，不需要完整的 $N \times N$）
- 这是一个经典的 **计算换显存** 权衡

反向传播中 $dV$ 的计算公式为：

$$dV_j = P_{ij}^T \cdot dO_i$$

其中 $P_{ij} = \text{diag}(l_i)^{-1} \cdot \exp(S_{ij} - m_i)$，而 $S_{ij} = Q_i K_j^T / \sqrt{d}$。

### 任务

**问题 1（概念）：** 为什么 Flash Attention 反向传播选择重新计算 $S$ 矩阵，而不是在前向传播时存储它？

```
你的回答: _____
```

**问题 2（概念）：** Flash Attention 反向传播的 IO 复杂度（HBM 访问量）是多少？与标准 Attention 反向传播相比如何？

```
你的回答: _____
```

**代码任务：** 利用存储的统计量 $l, m$ 重新计算 $P_{ij}$ 并求 $dV$。

```python
import torch
import math

def compute_dV_block(Q_i, K_j, V_j, dO_i, l_i, m_i):
    """
    计算 Flash Attention 反向传播中第 (i,j) 块对 dV_j 的贡献

    参数:
        Q_i:  [B, d]  第 i 个 Q 块
        K_j:  [B, d]  第 j 个 K 块
        V_j:  [B, d]  第 j 个 V 块
        dO_i: [B, d]  第 i 个输出梯度块
        l_i:  [B, 1]  前向传播存储的第 i 块 row-sum
        m_i:  [B, 1]  前向传播存储的第 i 块 row-max
    返回:
        dV_j_contrib: [B, d]  该块对 dV_j 的贡献
    """
    d = Q_i.shape[-1]

    # ===== 填空 1: 重新计算 S_ij（与前向传播完全相同）=====
    s_ij = _____  # 提示: Q_i @ K_j^T / sqrt(d)

    # ===== 填空 2: 利用存储的 m_i 和 l_i 重新计算 P_ij =====
    # P_ij = exp(S_ij - m_i) / l_i
    p_ij = _____  # 提示: 这就是 softmax，但用的是存储的全局统计量

    # ===== 填空 3: 计算 dV_j 的贡献 =====
    # dV_j += P_ij^T @ dO_i
    dV_j_contrib = _____  # 提示: P 转置后与 dO 相乘

    return dV_j_contrib

# ===== 验证 =====
torch.manual_seed(42)
N, d = 64, 32
Q = torch.randn(N, d, requires_grad=False)
K = torch.randn(N, d, requires_grad=False)
V = torch.randn(N, d, requires_grad=True)

# 标准 Attention forward
S = Q @ K.T / math.sqrt(d)
P = torch.softmax(S, dim=-1)
O = P @ V

# 构造 dO
dO = torch.randn_like(O)

# 标准方式计算 dV
O.backward(dO)
dV_ref = V.grad.clone()

# 用 Flash Attention 方式重新计算 dV
l = P.sum(dim=-1, keepdim=True)  # 这里直接用 P 的行和，实际中从前向传播存储
# 但更准确地：l = sum(exp(S - m)), m = S.max(dim=-1)
m = S.max(dim=-1, keepdim=True).values
l = torch.exp(S - m).sum(dim=-1, keepdim=True)

block_size = 16
num_blocks = N // block_size
dV_flash = torch.zeros_like(V.data)

for j in range(num_blocks):
    js, je = j * block_size, (j + 1) * block_size
    for i in range(num_blocks):
        qs, qe = i * block_size, (i + 1) * block_size
        contrib = compute_dV_block(
            Q[qs:qe], K[js:je], V.data[js:je], dO[qs:qe],
            l[qs:qe], m[qs:qe]
        )
        dV_flash[js:je] += contrib

diff = (dV_flash - dV_ref).abs().max().item()
print(f"Flash dV 与标准 dV 的最大误差: {diff:.2e}")
assert diff < 1e-4, f"误差过大: {diff}"
print("验证通过!")
```

### 提示

- 填空 1：与前向传播的计算完全相同
- 填空 2：注意 $m_i$ 和 $l_i$ 是前向传播中计算好的全局统计量，不是当前块的局部值
- 填空 3：$dV = P^T dO$，注意转置的是 $P$ 而不是 $dO$
- 概念题 1：想想 $S$ 矩阵的 shape 和显存代价
- 概念题 2：Flash Attention 前向和反向的 IO 复杂度相同

<details>
<summary>参考答案</summary>

```python
# 填空 1
s_ij = Q_i @ K_j.T / math.sqrt(d)

# 填空 2
p_ij = torch.exp(s_ij - m_i) / l_i

# 填空 3
dV_j_contrib = p_ij.T @ dO_i
```

**概念题 1 参考答案:**

存储完整的 $S$ 矩阵需要 $O(N^2)$ 显存，这正是 Flash Attention 要避免的瓶颈。而重新计算 $S_{ij}$（每次只计算一个 $B \times B$ 的小块）只需要 $O(B^2)$ 的额外显存。虽然增加了计算量（多了一次矩阵乘法），但在 GPU 上计算速度远快于 HBM 访问速度，因此这个权衡是值得的。在现代 GPU 上，FLOPs 远多于内存带宽，重新计算反而更快。

**概念题 2 参考答案:**

Flash Attention 反向传播的 HBM 访问量（IO 复杂度）为 $O(N^2 d^2 M^{-1})$，其中 $M$ 是 SRAM 大小。这与前向传播的 IO 复杂度相同。

与标准 Attention 反向传播的 $O(N^2 + Nd)$ 的 HBM 读写相比，当 $M$ 足够大（$d^2 \leq M$）时，Flash Attention 可以减少 HBM 访问次数。标准 Attention 需要从 HBM 读取完整的 $P$（$N \times N$），而 Flash Attention 只需要读取 $Q, K, V, O, l, m$（均为 $O(Nd)$），代价是需要重新计算 $S$ 和 $P$。

**解析:**

- **重新计算 vs 存储** 是 Flash Attention 最核心的设计选择。传统观点认为"存下来复用更好"，但 Flash Attention 指出在 IO-bound 的场景下，重新计算反而更快
- 反向传播需要重新计算 $S_{ij} = Q_i K_j^T / \sqrt{d}$，这是一次额外的矩阵乘法。但因为是分块进行（$B \times B$），每块都在 SRAM 中完成，不产生额外的 HBM 访问
- 前向传播只需存储 $O$（$N \times d$）、$l$（$N \times 1$）、$m$（$N \times 1$），反向传播就能利用这些信息完整地恢复 $P_{ij}$

</details>

---

## MLM 代码训练模式

完成上面的固定填空后，试试随机挖空模式 -- 每次点击「刷新」会随机遮盖不同的代码片段，帮你彻底记住每一行。

### 在线 Softmax (m/l 更新)

<CodeMasker title="在线 Softmax — running max 与 running sum 更新" :mask-ratio="0.15">
def online_softmax(x):
    rows, cols = x.shape
    output = torch.zeros_like(x)
    for i in range(rows):
        row = x[i]
        m = float('-inf')
        l = 0.0
        for j in range(cols):
            m_new = max(m, row[j].item())
            l = l * math.exp(m - m_new) + math.exp(row[j].item() - m_new)
            m = m_new
        for j in range(cols):
            output[i, j] = math.exp(row[j].item() - m) / l
    return output
</CodeMasker>

### 分块加载与 Tiling

<CodeMasker title="分块 Attention — Q 分块 + K 分块拼接 softmax" :mask-ratio="0.15">
def blocked_attention(Q, K, V, block_size):
    N, d = Q.shape
    O = torch.zeros(N, d)
    num_blocks = N // block_size
    for i in range(num_blocks):
        q_i = Q[i * block_size:(i + 1) * block_size]
        scores_row = []
        for j in range(num_blocks):
            k_j = K[j * block_size:(j + 1) * block_size]
            s_ij = q_i @ k_j.T / math.sqrt(d)
            scores_row.append(s_ij)
        S_i = torch.cat(scores_row, dim=-1)
        P_i = torch.softmax(S_i, dim=-1)
        O[i * block_size:(i + 1) * block_size] = P_i @ V
    return O
</CodeMasker>

### Flash Attention Forward 核心逻辑

<CodeMasker title="Flash Attention Forward — 在线 softmax + rescale 累积" :mask-ratio="0.15">
O = torch.zeros(N, d)
l = torch.zeros(N, 1)
m = torch.full((N, 1), float('-inf'))

for j in range(num_blocks):
    k_j = K[j * block_size:(j + 1) * block_size]
    v_j = V[j * block_size:(j + 1) * block_size]
    for i in range(num_blocks):
        q_i = Q[q_start:q_end]
        o_i = O[q_start:q_end]
        l_i = l[q_start:q_end]
        m_i = m[q_start:q_end]

        s_ij = q_i @ k_j.T / math.sqrt(d)
        m_ij = s_ij.max(dim=-1, keepdim=True).values
        m_new = torch.max(m_i, m_ij)
        p_ij = torch.exp(s_ij - m_new)
        l_new = l_i * torch.exp(m_i - m_new) + p_ij.sum(dim=-1, keepdim=True)

        alpha = l_i * torch.exp(m_i - m_new) / l_new
        beta = 1.0 / l_new
        o_i = alpha * o_i + beta * (p_ij @ v_j)

        O[q_start:q_end] = o_i
        l[q_start:q_end] = l_new
        m[q_start:q_end] = m_new
return O
</CodeMasker>
