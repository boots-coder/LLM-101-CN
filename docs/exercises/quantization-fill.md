---
title: "量化技术填空"
description: "Level 2-3 填空：Absmax 量化、Zero-Point 量化、GPTQ 核心、AWQ 核心"
topics: [fill-in, quantization, INT8, INT4, absmax, zero-point, GPTQ, AWQ]
---
# 量化技术填空 (Level 2-3)

> 本练习覆盖模型量化的核心技术：从最基础的 Absmax 对称量化，到 Zero-Point 非对称量化，再到 Per-Channel 细粒度量化、GPTQ 逐列误差补偿，以及 SmoothQuant 激活平滑。
> 代码基于实际量化实现，用 `_____` 标记需要填写的部分。

::: info 前置知识
- NumPy / PyTorch 基础
- 线性代数基础（矩阵乘法）
- 浮点数与整数的表示范围
:::

::: tip 量化核心思想
将浮点权重 $x \in \mathbb{R}$ 映射到低精度整数 $q \in \mathbb{Z}$：

$$q = \text{round}\left(\frac{x}{s}\right) + z, \quad \hat{x} = s \cdot (q - z)$$

其中 $s$ 为缩放因子（scale），$z$ 为零点（zero-point）。对称量化时 $z = 0$。
:::

---

## 练习 1: Absmax 对称量化（Level 2）

### 背景

Absmax 量化是最简单的量化方式：用张量中绝对值的最大值计算 scale，将浮点数映射到 $[-127, 127]$ 的 INT8 范围。因为以零为中心对称映射，所以称为"对称量化"。

核心公式：

$$s = \frac{\max(|x|)}{127}, \quad q = \text{round}\left(\frac{x}{s}\right), \quad \hat{x} = q \cdot s$$

### 任务

```python
import numpy as np

def absmax_quantize(x):
    """
    Absmax 对称量化到 INT8 [-127, 127]
    
    参数:
        x: np.ndarray, 浮点权重
    返回:
        q: np.ndarray, int8 量化值
        scale: float, 缩放因子
    """
    # ===== 填空 1: 计算 scale =====
    # scale = 绝对值最大值 / 127
    scale = _____

    # ===== 填空 2: 量化 =====
    # q = round(x / scale), 并裁剪到 [-127, 127]
    q = _____

    return q.astype(np.int8), scale


def absmax_dequantize(q, scale):
    """反量化: x_hat = q * scale"""
    # ===== 填空 3: 反量化 =====
    x_hat = _____
    return x_hat
```

### 提示

- `np.abs(x).max()` 计算绝对值最大值
- `np.round(...)` 做四舍五入
- `np.clip(arr, -127, 127)` 裁剪到范围内
- 反量化只需将整数乘以 scale

<details>
<summary>参考答案</summary>

```python
# 填空 1
scale = np.abs(x).max() / 127

# 填空 2
q = np.clip(np.round(x / scale), -127, 127)

# 填空 3
x_hat = q.astype(np.float32) * scale
```

**验证:**
```python
np.random.seed(42)

# 正态分布数据
x_normal = np.random.randn(1000).astype(np.float32)
q, scale = absmax_quantize(x_normal)
x_hat = absmax_dequantize(q, scale)
mse_normal = np.mean((x_normal - x_hat) ** 2)
print(f"正态分布 - Scale: {scale:.6f}, MSE: {mse_normal:.6f}")

# 含 outlier 的数据
x_outlier = np.random.randn(1000).astype(np.float32)
x_outlier[0] = 100.0  # 一个极端 outlier
q2, scale2 = absmax_quantize(x_outlier)
x_hat2 = absmax_dequantize(q2, scale2)
mse_outlier = np.mean((x_outlier - x_hat2) ** 2)
print(f"含 outlier  - Scale: {scale2:.6f}, MSE: {mse_outlier:.6f}")
print(f"Outlier 导致 MSE 增大 {mse_outlier / mse_normal:.1f} 倍")
# outlier 把 scale 拉大，导致大部分正常值的量化精度下降
```

**解析：**

1. **Scale 计算**：$s = \frac{\max(|x|)}{127}$，将最大绝对值映射到 INT8 的极限 127。
2. **量化**：除以 scale 后四舍五入，再裁剪防止溢出。
3. **反量化**：乘以 scale 恢复近似浮点值。注意 `q` 要先转为 float 再乘。
4. **Outlier 问题**：当数据中有极端值时，scale 被拉大，导致所有正常值的量化精度严重下降。这正是后续 SmoothQuant 要解决的问题。

</details>

---

## 练习 2: Zero-Point 非对称量化（Level 2）

### 背景

Absmax 量化假设数据以零为中心，但实际权重/激活往往分布不对称（如 ReLU 后全为正值）。Zero-Point 量化引入偏移量 $z$，将 $[\min(x), \max(x)]$ 映射到 $[0, 255]$ 的 UINT8 范围。

核心公式：

$$s = \frac{\max(x) - \min(x)}{255}, \quad z = \text{round}\left(-\frac{\min(x)}{s}\right)$$

$$q = \text{round}\left(\frac{x}{s}\right) + z, \quad \hat{x} = s \cdot (q - z)$$

### 任务

```python
import numpy as np

def zeropoint_quantize(x):
    """
    Zero-Point 非对称量化到 UINT8 [0, 255]
    
    参数:
        x: np.ndarray, 浮点权重
    返回:
        q: np.ndarray, uint8 量化值
        scale: float, 缩放因子
        zero_point: int, 零点偏移
    """
    x_min, x_max = x.min(), x.max()

    # ===== 填空 1: 计算 scale =====
    scale = _____

    # ===== 填空 2: 计算 zero_point =====
    zero_point = _____

    # ===== 填空 3: 量化 =====
    # q = round(x / scale) + zero_point, 裁剪到 [0, 255]
    q = _____

    return q.astype(np.uint8), scale, int(zero_point)


def zeropoint_dequantize(q, scale, zero_point):
    """反量化: x_hat = scale * (q - zero_point)"""
    # ===== 填空 4: 反量化 =====
    x_hat = _____
    return x_hat
```

### 提示

- Scale 用的是数据的全范围 `(max - min)` 除以量化范围 255
- Zero-point 的含义：浮点零在量化空间中对应的整数值
- `np.clip(arr, 0, 255)` 裁剪到 UINT8 范围
- 反量化时先减去 zero_point 再乘以 scale

<details>
<summary>参考答案</summary>

```python
# 填空 1
scale = (x_max - x_min) / 255

# 填空 2
zero_point = np.round(-x_min / scale)

# 填空 3
q = np.clip(np.round(x / scale) + zero_point, 0, 255)

# 填空 4
x_hat = scale * (q.astype(np.float32) - zero_point)
```

**验证:**
```python
np.random.seed(42)

# 偏斜分布 (模拟 ReLU 后的激活)
x_skewed = np.abs(np.random.randn(1000).astype(np.float32)) + 2.0

# Absmax 量化
q_abs, s_abs = absmax_quantize(x_skewed)
x_hat_abs = absmax_dequantize(q_abs, s_abs)
mse_abs = np.mean((x_skewed - x_hat_abs) ** 2)

# Zero-Point 量化
q_zp, s_zp, zp = zeropoint_quantize(x_skewed)
x_hat_zp = zeropoint_dequantize(q_zp, s_zp, zp)
mse_zp = np.mean((x_skewed - x_hat_zp) ** 2)

print(f"偏斜分布 x in [{x_skewed.min():.2f}, {x_skewed.max():.2f}]")
print(f"Absmax     MSE: {mse_abs:.6f}")
print(f"Zero-Point MSE: {mse_zp:.6f}")
print(f"Zero-Point 误差降低 {(1 - mse_zp / mse_abs) * 100:.1f}%")
# 偏斜分布下 Zero-Point 显著优于 Absmax
```

**解析：**

1. **Scale**：$s = \frac{x_{\max} - x_{\min}}{255}$，将数据的完整范围映射到 256 个量化级别。
2. **Zero-Point**：$z = \text{round}(-x_{\min}/s)$，确保浮点 0 在量化空间中有精确对应。这对 zero-padding 等操作很重要。
3. **优势**：对于 ReLU 后全正的激活，Absmax 量化会浪费一半的量化范围（负数区间未使用），而 Zero-Point 量化充分利用了 $[0, 255]$ 的全部范围。
4. **代价**：需要额外存储 zero_point，且反量化多一步减法。

</details>

---

## 练习 3: Per-Channel vs Per-Tensor 量化（Level 2-3）

### 背景

Per-Tensor 量化用一个 scale 处理整个权重矩阵，而 Per-Channel 量化为每个输出通道（即权重矩阵的每一行）计算独立的 scale。当权重矩阵的不同通道数值范围差异很大时（存在 outlier channel），Per-Channel 量化可以显著降低误差。

$$\text{Per-Tensor}: \quad s = \frac{\max(|W|)}{127}$$

$$\text{Per-Channel}: \quad s_i = \frac{\max(|W[i, :]|)}{127}, \quad i = 0, 1, \ldots, C_{\text{out}} - 1$$

### 任务

```python
import numpy as np

def pertensor_absmax_quantize(W):
    """
    Per-Tensor Absmax 量化
    W: [out_channels, in_channels]
    """
    scale = np.abs(W).max() / 127
    q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    return q, scale

def pertensor_dequantize(q, scale):
    return q.astype(np.float32) * scale


def perchannel_absmax_quantize(W):
    """
    Per-Channel Absmax 量化
    W: [out_channels, in_channels]
    每一行用独立的 scale
    
    返回:
        q: [out_channels, in_channels], int8
        scales: [out_channels], 每个通道的 scale
    """
    # ===== 填空 1: 对每一行计算 absmax =====
    # 结果形状应为 [out_channels]
    absmax_per_row = _____

    # ===== 填空 2: 计算每行的 scale =====
    # 形状 [out_channels], 注意防止除零
    scales = _____

    # ===== 填空 3: 量化（注意 scale 的广播） =====
    # 需要将 scales 扩展为 [out_channels, 1] 以便广播
    q = _____

    return q.astype(np.int8), scales


def perchannel_dequantize(q, scales):
    """Per-Channel 反量化"""
    # ===== 填空 4: 反量化（注意 scale 的广播） =====
    x_hat = _____
    return x_hat
```

### 提示

- `np.abs(W).max(axis=1)` 对每一行取绝对值最大值，结果形状 `[out_channels]`
- 广播时需要 reshape：`scales[:, np.newaxis]` 变成 `[out_channels, 1]`
- 防止除零可用 `np.maximum(absmax, 1e-8)`

<details>
<summary>参考答案</summary>

```python
# 填空 1
absmax_per_row = np.abs(W).max(axis=1)

# 填空 2
scales = np.maximum(absmax_per_row, 1e-8) / 127

# 填空 3
q = np.clip(np.round(W / scales[:, np.newaxis]), -127, 127)

# 填空 4
x_hat = q.astype(np.float32) * scales[:, np.newaxis]
```

**验证:**
```python
np.random.seed(42)

# 构造含 outlier channel 的权重矩阵
W = np.random.randn(4, 128).astype(np.float32) * 0.1
W[2, :] *= 100  # 第 2 行是 outlier channel, 数值大 100 倍

print("各通道权重范围:")
for i in range(4):
    print(f"  Channel {i}: [{W[i].min():.3f}, {W[i].max():.3f}]")

# Per-Tensor 量化
q_pt, s_pt = pertensor_absmax_quantize(W)
W_hat_pt = pertensor_dequantize(q_pt, s_pt)
mse_pt = np.mean((W - W_hat_pt) ** 2)

# Per-Channel 量化
q_pc, s_pc = perchannel_absmax_quantize(W)
W_hat_pc = perchannel_dequantize(q_pc, s_pc)
mse_pc = np.mean((W - W_hat_pc) ** 2)

print(f"\nPer-Tensor  MSE: {mse_pt:.6f} (scale = {s_pt:.6f})")
print(f"Per-Channel MSE: {mse_pc:.6f}")
print(f"Per-Channel 误差降低 {(1 - mse_pc / mse_pt) * 100:.1f}%")

# 分通道查看误差
print("\n各通道 MSE:")
for i in range(4):
    mse_i_pt = np.mean((W[i] - W_hat_pt[i]) ** 2)
    mse_i_pc = np.mean((W[i] - W_hat_pc[i]) ** 2)
    print(f"  Channel {i}: Per-Tensor={mse_i_pt:.6f}, Per-Channel={mse_i_pc:.6f}")
# 可以看到 outlier channel 把 per-tensor 的 scale 拉大,
# 导致正常通道的量化精度严重下降
```

**解析：**

1. **Per-Tensor 的问题**：一个 outlier channel 会把全局 scale 拉大，所有正常通道的量化精度都会下降。例如 outlier 通道的 absmax 是 10，正常通道的 absmax 是 0.1，Per-Tensor 的 scale = 10/127，正常通道只能用到很小的量化范围。
2. **Per-Channel 的优势**：每个通道独立计算 scale，outlier 通道不影响其他通道。实际量化工具（如 GPTQ、AWQ）都采用 Per-Channel 或更细粒度的量化。
3. **存储开销**：Per-Channel 需要为每个输出通道存储一个 scale（FP16），对于 $4096 \times 4096$ 的矩阵，额外存储 4096 个 FP16 值（8 KB），相对于整个矩阵的存储可以忽略不计。

</details>

---

## 练习 4: GPTQ 核心 -- 逐列量化与误差补偿（Level 3）

### 背景

GPTQ（Accurate Post-Training Quantization for Generative Pre-trained Transformers）的核心思想：逐列量化权重矩阵，每量化一列后，将量化误差补偿到尚未量化的列。这样后续列可以"纠正"前面列引入的误差，使整体量化误差最小化。

完整的 GPTQ 使用 Hessian 矩阵（$H = 2X^TX$）来决定误差分配权重。本练习使用简化版：将误差均匀分配到剩余列。

简化 GPTQ 流程（对权重矩阵 $W$ 的每一列 $j$）：
1. 量化第 $j$ 列：$\hat{W}_{:,j} = \text{quantize}(W_{:,j})$
2. 计算误差：$\delta_j = W_{:,j} - \hat{W}_{:,j}$
3. 将误差均匀分配到剩余列 $k > j$：$W_{:,k} \mathrel{+}= \frac{\delta_j}{n_{\text{remaining}}}$

### 任务

```python
import numpy as np

def quantize_column(col, n_bits=8):
    """对单列做 Absmax 量化并返回量化后的值"""
    qmax = 2 ** (n_bits - 1) - 1  # 127 for 8-bit
    scale = np.maximum(np.abs(col).max(), 1e-8) / qmax
    q = np.clip(np.round(col / scale), -qmax, qmax)
    return q * scale  # 返回反量化后的值


def naive_quantize(W, n_bits=8):
    """朴素量化: 逐列独立量化, 无误差补偿"""
    W_hat = np.zeros_like(W)
    for j in range(W.shape[1]):
        W_hat[:, j] = quantize_column(W[:, j], n_bits)
    return W_hat


def gptq_simplified(W, n_bits=8):
    """
    简化版 GPTQ: 逐列量化 + 均匀误差补偿
    
    W: [out_channels, in_channels]
    返回: W_hat, 量化后的权重矩阵
    """
    W = W.copy()  # 不修改原始矩阵
    n_cols = W.shape[1]
    W_hat = np.zeros_like(W)

    for j in range(n_cols):
        # ===== 填空 1: 量化当前列 =====
        W_hat[:, j] = _____

        # ===== 填空 2: 计算量化误差 =====
        error = _____

        # ===== 填空 3: 将误差均匀补偿到剩余未量化的列 =====
        n_remaining = n_cols - j - 1
        if n_remaining > 0:
            _____

    return W_hat
```

### 提示

- `quantize_column(W[:, j], n_bits)` 对第 $j$ 列量化
- 误差 = 原始值 - 量化值：`W[:, j] - W_hat[:, j]`
- 补偿：`W[:, j+1:] += error[:, np.newaxis] / n_remaining`
- 注意 `error` 形状是 `[out_channels]`，需要扩展维度才能加到 `W[:, j+1:]` 上

<details>
<summary>参考答案</summary>

```python
# 填空 1
W_hat[:, j] = quantize_column(W[:, j], n_bits)

# 填空 2
error = W[:, j] - W_hat[:, j]

# 填空 3
W[:, j+1:] += error[:, np.newaxis] / n_remaining
```

**验证:**
```python
np.random.seed(42)
W = np.random.randn(64, 128).astype(np.float32) * 0.1

# 朴素量化
W_naive = naive_quantize(W, n_bits=4)
mse_naive = np.mean((W - W_naive) ** 2)

# 简化 GPTQ
W_gptq = gptq_simplified(W, n_bits=4)
mse_gptq = np.mean((W - W_gptq) ** 2)

print(f"4-bit 朴素量化 MSE: {mse_naive:.8f}")
print(f"4-bit GPTQ     MSE: {mse_gptq:.8f}")
print(f"GPTQ 误差降低 {(1 - mse_gptq / mse_naive) * 100:.1f}%")

# 8-bit 对比
W_naive_8 = naive_quantize(W, n_bits=8)
W_gptq_8 = gptq_simplified(W, n_bits=8)
print(f"\n8-bit 朴素量化 MSE: {np.mean((W - W_naive_8)**2):.8f}")
print(f"8-bit GPTQ     MSE: {np.mean((W - W_gptq_8)**2):.8f}")
# 低比特下 GPTQ 的误差补偿效果更显著
```

**解析：**

1. **逐列量化**：GPTQ 不是一次性量化整个矩阵，而是逐列处理。这允许后续列根据前面列的误差做出调整。
2. **误差补偿**：核心思想来自 Optimal Brain Surgeon 方法。量化第 $j$ 列产生的误差 $\delta_j$ 会影响最终输出 $Y = WX$，通过调整未量化列的权重，可以部分抵消这个影响。
3. **完整 GPTQ 的改进**：实际 GPTQ 使用 Hessian 逆矩阵 $H^{-1}$ 来决定误差分配的权重，而非均匀分配。Hessian 包含了输入数据的二阶统计信息，能更精确地分配误差。此外，GPTQ 对列进行分组（block-wise）处理以提高数值稳定性。
4. **低比特更有效**：在 4-bit 量化下，单列误差更大，误差补偿的价值更高；8-bit 下单列误差本身就很小，补偿效果相对不明显。

</details>

---

## 练习 5: SmoothQuant -- 激活平滑（Level 3）

### 背景

LLM 推理中，激活（activation）往往包含 outlier（少数通道的值远大于其他通道），这使得激活量化非常困难。SmoothQuant 的核心思想：通过数学等价变换，将激活的量化难度转移到权重上。

对于线性层 $Y = XW$，SmoothQuant 引入 per-channel 的平滑因子 $s$：

$$Y = XW = (X \text{diag}(s)^{-1}) \cdot (\text{diag}(s) W) = \hat{X} \hat{W}$$

其中 $\hat{X} = X / s$（激活除以 $s$，减小 outlier），$\hat{W} = s \cdot W$（权重乘以 $s$，吸收难度）。

平滑因子的计算：

$$s_j = \frac{\max(|X_{:,j}|)^{\alpha}}{\max(|W_{j,:}|)^{1-\alpha}}$$

其中 $\alpha \in [0, 1]$ 控制难度在激活和权重之间的分配比例，通常取 $\alpha = 0.5$。

### 任务

```python
import numpy as np

def compute_smooth_factor(X, W, alpha=0.5):
    """
    计算 SmoothQuant 的平滑因子
    
    X: [n_tokens, dim]  校准数据的激活值
    W: [dim, out_dim]   权重矩阵
    alpha: float, 平滑强度 (0=全部转移到权重, 1=保持激活不变)
    
    返回:
        s: [dim], per-channel 平滑因子
    """
    # ===== 填空 1: 计算激活每个通道的最大绝对值 =====
    # act_scales[j] = max(|X[:, j]|), 形状 [dim]
    act_scales = _____

    # ===== 填空 2: 计算权重每个输入通道的最大绝对值 =====
    # weight_scales[j] = max(|W[j, :]|), 形状 [dim]
    weight_scales = _____

    # ===== 填空 3: 计算平滑因子 =====
    # s = act_scales^alpha / weight_scales^(1-alpha)
    s = _____

    return s


def apply_smoothing(X, W, s):
    """
    应用平滑变换
    
    返回:
        X_smooth: X / s  (按列除)
        W_smooth: diag(s) @ W  (按行乘)
    """
    # ===== 填空 4: 平滑激活 =====
    X_smooth = _____

    # ===== 填空 5: 平滑权重 =====
    W_smooth = _____

    return X_smooth, W_smooth
```

### 提示

- `np.abs(X).max(axis=0)` 对每列取最大绝对值，形状 `[dim]`
- `np.abs(W).max(axis=1)` 对每行取最大绝对值，形状 `[dim]`
- 幂运算：`arr ** alpha`
- 平滑激活：`X / s[np.newaxis, :]`（广播到每行）
- 平滑权重：`s[:, np.newaxis] * W`（广播到每列）

<details>
<summary>参考答案</summary>

```python
# 填空 1
act_scales = np.abs(X).max(axis=0)

# 填空 2
weight_scales = np.abs(W).max(axis=1)

# 填空 3
s = (act_scales ** alpha) / (weight_scales ** (1 - alpha) + 1e-8)

# 填空 4
X_smooth = X / s[np.newaxis, :]

# 填空 5
W_smooth = s[:, np.newaxis] * W
```

**验证:**
```python
np.random.seed(42)

# 构造含 outlier 的激活和正常权重
dim, out_dim, n_tokens = 128, 64, 100
X = np.random.randn(n_tokens, dim).astype(np.float32)
W = np.random.randn(dim, out_dim).astype(np.float32) * 0.1

# 在激活中注入 outlier channels
outlier_channels = [10, 50, 100]
for ch in outlier_channels:
    X[:, ch] *= 50  # 这些通道的值放大 50 倍

print("平滑前激活各通道 max|X|:")
print(f"  正常通道 (ch=0):  {np.abs(X[:, 0]).max():.2f}")
print(f"  Outlier (ch=10):  {np.abs(X[:, 10]).max():.2f}")
print(f"  Outlier (ch=50):  {np.abs(X[:, 50]).max():.2f}")

# 计算平滑因子并应用
s = compute_smooth_factor(X, W, alpha=0.5)
X_smooth, W_smooth = apply_smoothing(X, W, s)

print(f"\n平滑后激活各通道 max|X_smooth|:")
print(f"  正常通道 (ch=0):  {np.abs(X_smooth[:, 0]).max():.2f}")
print(f"  Outlier (ch=10):  {np.abs(X_smooth[:, 10]).max():.2f}")
print(f"  Outlier (ch=50):  {np.abs(X_smooth[:, 50]).max():.2f}")

# 验证数学等价性: Y = XW = X_smooth @ W_smooth
Y_original = X @ W
Y_smooth = X_smooth @ W_smooth
diff = np.abs(Y_original - Y_smooth).max()
print(f"\n数学等价性验证: max|XW - X_smooth @ W_smooth| = {diff:.10f}")

# 量化误差对比
def quantize_and_compute(X, W):
    """量化激活和权重后计算 Y, 返回与真实值的 MSE"""
    q_x, s_x = absmax_quantize(X)
    q_w, s_w = absmax_quantize(W)
    X_hat = absmax_dequantize(q_x, s_x)
    W_hat = absmax_dequantize(q_w, s_w)
    Y_hat = X_hat @ W_hat
    Y_true = X @ W
    return np.mean((Y_true - Y_hat) ** 2)

mse_before = quantize_and_compute(X, W)
mse_after = quantize_and_compute(X_smooth, W_smooth)
print(f"\n平滑前量化 MSE: {mse_before:.6f}")
print(f"平滑后量化 MSE: {mse_after:.6f}")
print(f"平滑后误差降低 {(1 - mse_after / mse_before) * 100:.1f}%")
```

**解析：**

1. **激活 outlier 问题**：LLM 中某些通道的激活值会比其他通道大几十倍甚至上百倍。如果直接量化，这些 outlier 会把 scale 拉大，使大多数通道的精度严重下降。
2. **SmoothQuant 的巧妙之处**：利用 $Y = XW = (X/s)(sW)$ 的等价变换，将激活中的 outlier "转移"到权重上。权重是静态的，可以离线处理；激活是动态的，减小其范围能直接提升量化精度。
3. **$\alpha$ 的作用**：$\alpha = 1$ 时不做平滑，$\alpha = 0$ 时将所有难度转移到权重。实际中 $\alpha = 0.5$ 是较好的平衡点，让激活和权重各承担一半的量化难度。
4. **与其他方法的关系**：SmoothQuant 解决了 W8A8（权重 8-bit + 激活 8-bit）量化的关键障碍，使得 LLM 推理可以使用 INT8 矩阵乘法加速。AWQ 则更进一步，根据激活 outlier 的重要性来保护关键权重通道。

</details>

---

## MLM 代码训练模式

完成上面的固定填空后，试试随机挖空模式 -- 每次点击「刷新」会随机遮盖不同的代码片段，帮你彻底记住每一行。

### Absmax 对称量化/反量化

<CodeMasker title="Absmax 量化：scale 计算、量化与反量化" :mask-ratio="0.15">
def absmax_quantize(x):
    scale = np.abs(x).max() / 127
    q = np.clip(np.round(x / scale), -127, 127)
    return q.astype(np.int8), scale

def absmax_dequantize(q, scale):
    x_hat = q.astype(np.float32) * scale
    return x_hat
</CodeMasker>

### Zero-Point 非对称量化

<CodeMasker title="Zero-Point 量化：scale、零点与反量化" :mask-ratio="0.15">
def zeropoint_quantize(x):
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / 255
    zero_point = np.round(-x_min / scale)
    q = np.clip(np.round(x / scale) + zero_point, 0, 255)
    return q.astype(np.uint8), scale, int(zero_point)

def zeropoint_dequantize(q, scale, zero_point):
    x_hat = scale * (q.astype(np.float32) - zero_point)
    return x_hat
</CodeMasker>

### Per-Channel 量化

<CodeMasker title="Per-Channel Absmax：逐行 scale 与广播量化" :mask-ratio="0.15">
def perchannel_absmax_quantize(W):
    absmax_per_row = np.abs(W).max(axis=1)
    scales = np.maximum(absmax_per_row, 1e-8) / 127
    q = np.clip(np.round(W / scales[:, np.newaxis]), -127, 127)
    return q.astype(np.int8), scales

def perchannel_dequantize(q, scales):
    x_hat = q.astype(np.float32) * scales[:, np.newaxis]
    return x_hat
</CodeMasker>

### SmoothQuant 激活平滑

<CodeMasker title="SmoothQuant：平滑因子计算与激活-权重变换" :mask-ratio="0.15">
def compute_smooth_factor(X, W, alpha=0.5):
    act_scales = np.abs(X).max(axis=0)
    weight_scales = np.abs(W).max(axis=1)
    s = (act_scales ** alpha) / (weight_scales ** (1 - alpha) + 1e-8)
    return s

def apply_smoothing(X, W, s):
    X_smooth = X / s[np.newaxis, :]
    W_smooth = s[:, np.newaxis] * W
    return X_smooth, W_smooth
</CodeMasker>
