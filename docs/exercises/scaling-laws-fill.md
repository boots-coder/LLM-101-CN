---
title: "Scaling Laws 填空"
description: "Level 1-3 填空：参数量计算、FLOPs 估算、Chinchilla 最优配比、Loss 预测"
topics: [fill-in, scaling-laws, FLOPs, Chinchilla, compute-optimal, parameter-counting]
---
# Scaling Laws 填空 (Level 1-3)

> 本练习覆盖大模型 Scaling Laws 的核心计算：从 Transformer 参数量统计，到训练 FLOPs 估算，再到 Chinchilla 最优配比和 Loss 幂律拟合。
> 代码基于 NumPy / SciPy，用 `_____` 标记需要填写的部分。

::: info 前置知识
- Python / NumPy / SciPy 基础
- Transformer 架构基本概念（Attention、FFN、Embedding）
- 基本的数学优化概念（最小二乘法）
:::

::: tip Scaling Laws 核心公式
Kaplan et al. (2020) 和 Hoffmann et al. (2022, Chinchilla) 发现，模型性能遵循幂律：

$$L(N) = a N^{-\alpha} + L_\infty$$

其中 $N$ 为参数量，$L_\infty$ 为不可约损失（数据本身的熵下界）。

训练所需计算量近似为 $C \approx 6ND$（$N$ = 参数量，$D$ = 训练 token 数）。

Chinchilla 最优配比：给定计算预算 $C$，最优的 $N$ 和 $D$ 满足 $D \approx 20N$。
:::

---

## 练习 1: Transformer 参数量计算（Level 1-2）

### 背景

一个标准 decoder-only Transformer 的参数来自三部分：**Embedding 层**（$V \times d$）、**Attention**（Q/K/V/O 四个 $d \times d$ 矩阵）、**FFN**（两层线性变换，中间维度 $4d$）。总参数量近似 $N \approx V d + L \times (12 d^2 + 13d)$。

现代大模型（如 LLaMA）使用 GatedMLP（SwiGLU），FFN 有三个投影矩阵而非两个，参数量需相应调整。

### 任务

```python
import numpy as np

def count_transformer_params(
    vocab_size,      # V: 词表大小
    hidden_size,     # d: 隐藏维度
    num_layers,      # L: Transformer 层数
    num_heads,       # h: 注意力头数
    ffn_mult=4,      # FFN 中间维度倍数（标准 Transformer 为 4）
    use_gated_mlp=False,  # 是否使用 GatedMLP (SwiGLU)
    weight_tying=False    # 输入/输出 embedding 是否共享权重
):
    """
    计算 decoder-only Transformer 的参数量。
    不计 bias（现代大模型通常不用 bias）。
    """
    head_dim = hidden_size // num_heads
    ffn_hidden = hidden_size * ffn_mult

    # ===== 填空 1: Embedding 参数量 =====
    # Token embedding: V * d
    embedding_params = _____

    # ===== 填空 2: 单层 Attention 参数量 =====
    # Q, K, V 各一个投影矩阵: d -> d (即 d * d 每个)
    # Output 投影矩阵: d -> d
    # 共 4 个 d*d 的矩阵
    attn_params_per_layer = _____

    # ===== 填空 3: 单层 FFN 参数量 =====
    # 标准 FFN: up (d -> ffn_hidden) + down (ffn_hidden -> d)
    # GatedMLP: gate (d -> ffn_hidden) + up (d -> ffn_hidden) + down (ffn_hidden -> d)
    if use_gated_mlp:
        ffn_params_per_layer = _____  # 3 个矩阵
    else:
        ffn_params_per_layer = _____  # 2 个矩阵

    # ===== 填空 4: LayerNorm 参数量 =====
    # 每层 2 个 LayerNorm (attention 前 + FFN 前)，每个有 d 个 scale 参数
    # (RMSNorm 没有 bias/shift，只有 scale)
    ln_params_per_layer = _____

    # ===== 填空 5: 总参数量 =====
    # 所有层的参数 + embedding + 最终 LayerNorm
    # 如果 weight_tying=False, 输出 head 额外有 d*V 个参数
    total_per_layer = attn_params_per_layer + ffn_params_per_layer + ln_params_per_layer
    total = num_layers * total_per_layer + embedding_params + hidden_size  # 最终 LN

    if not weight_tying:
        output_head_params = _____
        total += output_head_params

    return {
        "embedding": embedding_params,
        "attn_per_layer": attn_params_per_layer,
        "ffn_per_layer": ffn_params_per_layer,
        "ln_per_layer": ln_params_per_layer,
        "total_per_layer": total_per_layer,
        "total": total,
    }
```

### 提示

- Embedding 参数量就是 $V \times d$
- Attention 的 Q/K/V/O 四个投影矩阵各为 $d \times d$，共 $4d^2$
- 标准 FFN 有 2 个矩阵：$d \times 4d$ 和 $4d \times d$，共 $2 \times d \times \text{ffn\_hidden} = 8d^2$
- GatedMLP 多一个 gate 矩阵，共 3 个 $d \times \text{ffn\_hidden}$ 的矩阵
- 输出 head 与 embedding 形状相同：$d \times V$

<details>
<summary>参考答案</summary>

```python
# 填空 1: Embedding 参数量
embedding_params = vocab_size * hidden_size

# 填空 2: Attention 参数量 (Q + K + V + O)
attn_params_per_layer = 4 * hidden_size * hidden_size

# 填空 3: FFN 参数量
if use_gated_mlp:
    ffn_params_per_layer = 3 * hidden_size * ffn_hidden  # gate + up + down
else:
    ffn_params_per_layer = 2 * hidden_size * ffn_hidden  # up + down

# 填空 4: LayerNorm (RMSNorm) 参数量
ln_params_per_layer = 2 * hidden_size

# 填空 5: 输出 head 参数量
output_head_params = hidden_size * vocab_size
```

**验证:**
```python
gpt2 = count_transformer_params(50257, 768, 12, 12, 4, False, True)
print(f"GPT-2 Small: {gpt2['total']/1e6:.1f}M")  # 预期 ~124M

llama7b = count_transformer_params(32000, 4096, 32, 32, 11008/4096, True, False)
print(f"LLaMA-7B:   {llama7b['total']/1e9:.2f}B")  # 预期 ~6.74B

llama70b = count_transformer_params(32000, 8192, 80, 64, 28672/8192, True, False)
print(f"LLaMA-70B:  {llama70b['total']/1e9:.2f}B")  # 预期 ~65B
```

**解析：** Embedding 参数占比通常不到 5%。LLaMA 使用 GQA，KV 投影更小，实际参数略少于 MHA 近似值。"7B" 是向上取整的营销数字，实际约 6.74B。

</details>

---

## 练习 2: 训练 FLOPs 估算（Level 2）

### 背景

Kaplan et al. (2020) 给出训练 FLOPs 的经验公式：$C \approx 6ND$（$N$ = 参数量，$D$ = token 数）。系数 6 = 2（前向）+ 4（反向约 2 倍前向），误差通常不超过 10%。

结合 GPU 理论算力（如 A100 BF16 = 312 TFLOPS）和实际利用率 MFU（通常 30%-50%），可以估算训练时间。

### 任务

```python
import numpy as np

def estimate_training_flops(num_params, num_tokens):
    """
    估算训练所需的总 FLOPs。
    
    参数:
        num_params: 模型参数量 N
        num_tokens: 训练 token 数 D
    返回:
        total_flops: 总 FLOPs (C ≈ 6ND)
    """
    # ===== 填空 1: 计算总 FLOPs =====
    total_flops = _____
    return total_flops


def estimate_training_time(
    num_params,
    num_tokens,
    num_gpus,
    gpu_peak_tflops,    # GPU 峰值算力 (TFLOPS, 如 A100 BF16 = 312)
    mfu=0.4             # Model FLOPs Utilization (通常 0.3-0.5)
):
    """
    估算训练时间。
    
    返回:
        training_hours: 训练时间（小时）
    """
    total_flops = estimate_training_flops(num_params, num_tokens)

    # ===== 填空 2: 计算实际每秒可用 FLOPs =====
    # 每块 GPU 的实际吞吐 = 峰值 * MFU
    # 总吞吐 = num_gpus * 实际吞吐
    # 注意单位: gpu_peak_tflops 是 TFLOPS (10^12)
    effective_flops_per_sec = _____

    # ===== 填空 3: 计算训练时间（秒 -> 小时） =====
    training_seconds = _____
    training_hours = _____
    return training_hours


def max_model_size(
    total_gpu_hours,
    num_gpus,
    gpu_peak_tflops,
    mfu=0.4,
    tokens_per_param=20   # Chinchilla 最优比例 D ≈ 20N
):
    """
    给定 GPU 算力预算，计算能训练的最大模型。
    
    利用 C = 6ND 和 D = tokens_per_param * N，
    得到 C = 6 * N * tokens_per_param * N = 6 * tokens_per_param * N^2
    """
    # ===== 填空 4: 计算可用总 FLOPs =====
    total_seconds = total_gpu_hours * 3600
    available_flops = _____

    # ===== 填空 5: 从 C = 6 * tokens_per_param * N^2 反推 N =====
    # N = sqrt(C / (6 * tokens_per_param))
    max_n = _____

    max_d = int(tokens_per_param * max_n)
    return int(max_n), max_d
```

### 提示

- $C = 6ND$ 是最核心的公式
- 实际吞吐 = GPU 数量 $\times$ 单卡峰值 $\times$ MFU
- 注意单位换算：TFLOPS = $10^{12}$ FLOPS，1 小时 = 3600 秒
- 反推模型大小时，将 $D = 20N$ 代入 $C = 6ND$ 得到 $C = 120N^2$，解出 $N = \sqrt{C/120}$

<details>
<summary>参考答案</summary>

```python
# 填空 1: 总 FLOPs
total_flops = 6 * num_params * num_tokens

# 填空 2: 实际每秒可用 FLOPs
effective_flops_per_sec = num_gpus * gpu_peak_tflops * 1e12 * mfu

# 填空 3: 训练时间
training_seconds = total_flops / effective_flops_per_sec
training_hours = training_seconds / 3600

# 填空 4: 可用总 FLOPs
available_flops = total_seconds * num_gpus * gpu_peak_tflops * 1e12 * mfu

# 填空 5: 反推最大模型参数量
max_n = np.sqrt(available_flops / (6 * tokens_per_param))
```

**验证:**
```python
N_7b, D_1t = 6.7e9, 1e12
print(f"FLOPs: {estimate_training_flops(N_7b, D_1t):.2e}")  # 预期 4.02e22

hours_8a100 = estimate_training_time(N_7b, D_1t, 8, 312, 0.4)
print(f"8x A100: {hours_8a100:,.0f}h ({hours_8a100/24:,.0f}d)")  # 预期 ~465 天

hours_2048 = estimate_training_time(N_7b, D_1t, 2048, 312, 0.35)
print(f"2048x A100: {hours_2048:,.0f}h ({hours_2048/24:.1f}d)")

max_n, max_d = max_model_size(10000, 1, 312, 0.4)
print(f"10000 GPU-h: N={max_n/1e9:.2f}B, D={max_d/1e9:.1f}B tokens")
```

**解析：** 系数 6 = 2（前向）+ 4（反向约 2 倍前向）。MFU 受通信开销、内存带宽、Pipeline bubble 等影响，大规模并行时通常更低（0.3-0.35）。

</details>

---

## 练习 3: Chinchilla 最优配比（Level 2-3）

### 背景

Chinchilla (Hoffmann et al., 2022) 核心发现：给定计算预算 $C$，$N$ 和 $D$ 应等比例缩放，最优比例约 $D \approx 20N$。70B 的 Chinchilla 在 1.4T token 上训练，性能超过 175B GPT-3（仅 300B token）。

Loss 模型：$L(N, D) = A/N^\alpha + B/D^\beta + L_\infty$，第一项是模型容量不足损失，第二项是数据不足损失。给定 $C = 6ND$，可求使 $L$ 最小的 $N^*$ 和 $D^*$。

### 任务

```python
import numpy as np
from scipy.optimize import minimize_scalar

# Chinchilla 论文中拟合的参数（近似值）
A = 406.4
B = 410.7
alpha = 0.34
beta = 0.28
L_inf = 1.69

def chinchilla_loss(N, D):
    """
    计算 Chinchilla Loss: L(N, D) = A/N^alpha + B/D^beta + L_inf
    """
    # ===== 填空 1: 实现 Chinchilla Loss 公式 =====
    loss = _____
    return loss


def chinchilla_optimal(C):
    """
    给定计算预算 C (FLOPs)，找到最优的 N 和 D。
    约束: C = 6 * N * D
    
    方法: 将 D = C / (6N) 代入 Loss 函数，对 N 求最优。
    """
    def loss_given_N(log_N):
        N = np.exp(log_N)
        # ===== 填空 2: 从 C 和 N 计算 D =====
        D = _____
        if D < 1:
            return 1e10  # 无效区域
        return chinchilla_loss(N, D)

    # 搜索范围: N 从 1M 到 1T
    result = minimize_scalar(
        loss_given_N,
        bounds=(np.log(1e6), np.log(1e12)),
        method='bounded'
    )

    N_opt = np.exp(result.x)
    # ===== 填空 3: 计算对应的最优 D =====
    D_opt = _____

    return int(N_opt), int(D_opt), result.fun


def compare_strategies(C):
    """
    对比三种策略:
    1. 过大模型 (Large N, Small D): 将 80% 计算花在更大的模型上
    2. 小模型多数据 (Small N, Large D): 将 20% 计算花在模型上
    3. Chinchilla 最优
    """
    N_opt, D_opt, loss_opt = chinchilla_optimal(C)

    # ===== 填空 4: 过大模型策略 =====
    # N 是最优值的 4 倍，D 从 C = 6ND 反推
    N_large = 4 * N_opt
    D_large = _____
    loss_large = chinchilla_loss(N_large, D_large)

    # ===== 填空 5: 小模型多数据策略 =====
    # N 是最优值的 1/4，D 从 C = 6ND 反推
    N_small = N_opt // 4
    D_small = _____
    loss_small = chinchilla_loss(N_small, D_small)

    return {
        "过大模型": {"N": N_large, "D": D_large, "loss": loss_large},
        "Chinchilla 最优": {"N": N_opt, "D": D_opt, "loss": loss_opt},
        "小模型多数据": {"N": N_small, "D": D_small, "loss": loss_small},
    }
```

### 提示

- Loss 公式直接对应三项之和：$A / N^\alpha + B / D^\beta + L_\infty$
- 约束条件 $C = 6ND$ 意味着 $D = C / (6N)$
- 过大模型和小模型策略的 $D$ 都通过 $D = C / (6N)$ 从同一总预算 $C$ 推出
- `scipy.optimize.minimize_scalar` 对 $\log N$ 搜索更稳定（避免数值问题）

<details>
<summary>参考答案</summary>

```python
# 填空 1: Chinchilla Loss
loss = A / (N ** alpha) + B / (D ** beta) + L_inf

# 填空 2: 从 C 和 N 计算 D
D = C / (6 * N)

# 填空 3: 最优 D
D_opt = C / (6 * N_opt)

# 填空 4: 过大模型的 D
D_large = C / (6 * N_large)

# 填空 5: 小模型多数据的 D
D_small = C / (6 * N_small)
```

**验证:**
```python
C_budget = 6 * 6.7e9 * 1e12  # ≈ 4e22 FLOPs
N_opt, D_opt, loss_opt = chinchilla_optimal(C_budget)
print(f"Chinchilla 最优: N={N_opt/1e9:.2f}B, D={D_opt/1e9:.0f}B, D/N={D_opt/N_opt:.1f}")

strategies = compare_strategies(C_budget)
for name, s in strategies.items():
    print(f"{name:12s}: N={s['N']/1e9:7.2f}B, D={s['D']/1e9:7.0f}B, Loss={s['loss']:.4f}")
# 预期: Chinchilla 最优的 loss 最低，D/N 约 20
```

**解析：** "过大模型"策略的 loss 偏高（数据不足，$B/D^\beta$ 大），"小模型多数据"策略同理（模型容量不足，$A/N^\alpha$ 大）。实际中，推理成本也是重要考量——如果推理量大，LLaMA 式 over-training（$D/N > 100$）可能更经济。

</details>

---

## 练习 4: Loss 预测与幂律拟合（Level 2-3）

### 背景

Scaling Laws 最实用的应用：**用小模型实验预测大模型性能**。在 3-5 个规模的小模型上测量 loss，拟合 $L(N) = a N^{-\alpha} + L_\infty$ 的三个参数，即可预测更大模型的 loss。

拟合用 `scipy.optimize.curve_fit`（非线性最小二乘法）。注意 $L_\infty$ 的估计对外推影响很大。

### 任务

```python
import numpy as np
from scipy.optimize import curve_fit

# 模拟实验数据 (来自不同规模模型的训练结果)
# 这些数据点大致符合公开的 Scaling Laws 实验结果
experiment_data = {
    "N": np.array([1e7, 5e7, 1e8, 5e8, 1e9, 3e9]),    # 参数量
    "L": np.array([3.20, 2.85, 2.65, 2.35, 2.20, 2.05]) # 验证集 loss
}

def scaling_law_func(N, a, alpha, L_inf):
    """
    幂律公式: L(N) = a * N^(-alpha) + L_inf
    """
    # ===== 填空 1: 实现幂律公式 =====
    return _____


def fit_scaling_law(N_data, L_data):
    """
    拟合 Scaling Law 的三个参数: a, alpha, L_inf
    
    使用 scipy.optimize.curve_fit 做非线性最小二乘拟合。
    """
    # ===== 填空 2: 设置初始猜测值 =====
    # a: 控制曲线整体高度，通常在 1-1000 之间
    # alpha: 控制下降速率，通常在 0.05-1 之间
    # L_inf: 不可约损失，通常在 1-2 之间
    p0 = _____  # [a_init, alpha_init, L_inf_init]

    # ===== 填空 3: 调用 curve_fit 拟合 =====
    # curve_fit 返回 (最优参数, 协方差矩阵)
    popt, pcov = _____

    a, alpha, L_inf = popt
    # 计算参数的标准误差
    perr = np.sqrt(np.diag(pcov))

    return {
        "a": a, "alpha": alpha, "L_inf": L_inf,
        "std_errors": {"a": perr[0], "alpha": perr[1], "L_inf": perr[2]},
    }


def predict_loss(fit_result, N_target):
    """
    用拟合结果预测目标规模模型的 loss。
    """
    a = fit_result["a"]
    alpha = fit_result["alpha"]
    L_inf = fit_result["L_inf"]

    # ===== 填空 4: 预测 loss =====
    predicted_loss = _____

    return predicted_loss


def analyze_extrapolation(fit_result, N_data, L_data):
    """
    分析拟合质量和外推可靠性。
    """
    # ===== 填空 5: 计算训练数据上的 R^2 =====
    L_pred = np.array([predict_loss(fit_result, n) for n in N_data])
    ss_res = np.sum((L_data - L_pred) ** 2)
    ss_tot = _____  # 总变异 = sum((L_data - mean(L_data))^2)
    r_squared = _____

    return {
        "r_squared": r_squared,
        "residuals": L_data - L_pred,
        "L_inf": fit_result["L_inf"],
    }
```

### 提示

- 幂律公式：`a * N ** (-alpha) + L_inf`
- `curve_fit(f, xdata, ydata, p0=初始值)` 做非线性最小二乘拟合
- 初始猜测值不需要很精确，但量级要合理：`p0 = [100, 0.1, 1.8]` 是一个可以尝试的起点
- $R^2 = 1 - SS_{res} / SS_{tot}$，其中 $SS_{tot} = \sum(y_i - \bar{y})^2$

<details>
<summary>参考答案</summary>

```python
# 填空 1: 幂律公式
return a * N ** (-alpha) + L_inf

# 填空 2: 初始猜测值
p0 = [100.0, 0.1, 1.8]

# 填空 3: curve_fit 拟合
popt, pcov = curve_fit(scaling_law_func, N_data, L_data, p0=p0, maxfev=10000)

# 填空 4: 预测 loss
predicted_loss = a * N_target ** (-alpha) + L_inf

# 填空 5: R^2 计算
ss_tot = np.sum((L_data - np.mean(L_data)) ** 2)
r_squared = 1 - ss_res / ss_tot
```

**验证:**
```python
result = fit_scaling_law(experiment_data["N"], experiment_data["L"])
print(f"a={result['a']:.4f}, alpha={result['alpha']:.4f}, L_inf={result['L_inf']:.4f}")

analysis = analyze_extrapolation(result, experiment_data["N"], experiment_data["L"])
print(f"R^2: {analysis['r_squared']:.6f}")

for N_t in [7e9, 13e9, 70e9, 175e9]:
    print(f"  N={N_t/1e9:.0f}B -> Loss={predict_loss(result, N_t):.4f}")
print(f"注意: 175B 是训练数据最大 N 的 {175e9/3e9:.0f}x 外推，可靠性较低")
```

**解析：** 幂律拟合在训练范围内通常 $R^2 > 0.99$，但外推可靠性随距离下降。$L_\infty$ 的估计对外推影响最大。实际中通常只信任 10x 以内的外推。

</details>

---

## 练习 5: 训练预算规划器（Level 3）

### 背景

本练习将前面所有知识整合为实用工具：给定 GPU 资源和预算，自动计算 Chinchilla 最优配置、估算训练时间、预测 loss。

规划器综合考虑：GPU 算力 -> 可用 FLOPs -> 最优 $N$ 和 $D$ -> 预测 loss -> 推荐模型架构（层数、隐藏维度等）。

### 任务

```python
import numpy as np
from scipy.optimize import minimize_scalar

# GPU 规格表
GPU_SPECS = {
    "A100_40G":  {"peak_tflops": 312, "memory_gb": 40, "cost_per_hour": 1.10},
    "A100_80G":  {"peak_tflops": 312, "memory_gb": 80, "cost_per_hour": 1.85},
    "H100_80G":  {"peak_tflops": 989, "memory_gb": 80, "cost_per_hour": 3.50},
    "H200_141G": {"peak_tflops": 989, "memory_gb": 141, "cost_per_hour": 4.50},
}

# Chinchilla 参数
CHINCHILLA = {"A": 406.4, "B": 410.7, "alpha": 0.34, "beta": 0.28, "L_inf": 1.69}


class TrainingBudgetPlanner:
    def __init__(self, gpu_type="A100_80G", num_gpus=8, mfu=0.4):
        assert gpu_type in GPU_SPECS, f"未知 GPU 类型: {gpu_type}"
        self.gpu = GPU_SPECS[gpu_type]
        self.gpu_type = gpu_type
        self.num_gpus = num_gpus
        self.mfu = mfu

    def compute_available_flops(self, budget_hours=None, budget_dollars=None):
        """
        计算给定预算下的可用 FLOPs。
        可以指定 GPU 小时预算或美元预算（二选一）。
        """
        if budget_dollars is not None:
            # ===== 填空 1: 从美元预算计算可用 GPU 小时 =====
            # 总费用 = num_gpus * cost_per_hour * hours
            budget_hours = _____

        # ===== 填空 2: 计算可用 FLOPs =====
        # 可用 FLOPs = GPU 数 * 峰值算力 * MFU * 总训练秒数
        total_seconds = budget_hours * 3600
        available_flops = _____

        return available_flops, budget_hours

    def find_optimal_config(self, available_flops):
        """
        给定可用 FLOPs，找到 Chinchilla 最优的 N 和 D。
        """
        C = available_flops
        p = CHINCHILLA

        def loss_given_log_N(log_N):
            N = np.exp(log_N)
            D = C / (6 * N)
            if D < 1:
                return 1e10
            # ===== 填空 3: 计算 Chinchilla Loss =====
            return _____

        result = minimize_scalar(
            loss_given_log_N,
            bounds=(np.log(1e6), np.log(1e12)),
            method='bounded'
        )

        N_opt = int(np.exp(result.x))
        D_opt = int(C / (6 * N_opt))
        loss_opt = result.fun
        return N_opt, D_opt, loss_opt

    def suggest_architecture(self, target_params):
        """
        根据目标参数量，推荐合理的模型架构。
        
        使用经验法则:
        - hidden_size 选择 128 的倍数（硬件友好）
        - depth-to-width 比例约 layers ≈ 2 * hidden_size / 128
        - FFN 使用 GatedMLP (SwiGLU), 中间维度约 8/3 * hidden_size
        """
        # ===== 填空 4: 估算 hidden_size =====
        # 近似: N ≈ 12 * L * d^2 (忽略 embedding 和小项)
        # 设 L ≈ d / 64 (经验比例), 则 N ≈ 12 * (d/64) * d^2 = 0.1875 * d^3
        # d ≈ (N / 0.1875) ^ (1/3)
        d_raw = _____
        hidden_size = int(round(d_raw / 128)) * 128  # 对齐到 128
        hidden_size = max(hidden_size, 256)

        # ===== 填空 5: 估算层数 =====
        # 层数 ≈ hidden_size / 64 (经验比例)
        num_layers = _____
        num_layers = max(num_layers, 4)

        num_heads = max(hidden_size // 128, 4)  # head_dim = 128
        ffn_hidden = int(hidden_size * 8 / 3)
        ffn_hidden = int(round(ffn_hidden / 256)) * 256  # 对齐到 256

        # 实际参数量估算 (GatedMLP)
        vocab_size = 32000
        params_per_layer = 4 * hidden_size**2 + 3 * hidden_size * ffn_hidden + 2 * hidden_size
        actual_params = num_layers * params_per_layer + 2 * vocab_size * hidden_size + hidden_size

        return {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ffn_hidden": ffn_hidden,
            "vocab_size": vocab_size,
            "actual_params": actual_params,
        }

    def plan(self, budget_hours=None, budget_dollars=None):
        """
        完整规划流程: 预算 -> FLOPs -> 最优 N,D -> 架构 -> 训练时间
        """
        flops, hours = self.compute_available_flops(budget_hours, budget_dollars)
        N_opt, D_opt, loss_pred = self.find_optimal_config(flops)
        arch = self.suggest_architecture(N_opt)
        cost = hours * self.num_gpus * self.gpu["cost_per_hour"]

        return {
            "budget": {
                "gpu_type": self.gpu_type,
                "num_gpus": self.num_gpus,
                "hours": hours,
                "cost_usd": cost,
            },
            "compute": {
                "available_flops": flops,
                "mfu": self.mfu,
            },
            "optimal": {
                "N": N_opt,
                "D": D_opt,
                "D_over_N": D_opt / N_opt,
                "predicted_loss": loss_pred,
            },
            "architecture": arch,
        }
```

### 提示

- 美元预算转 GPU 小时：`hours = budget_dollars / (num_gpus * cost_per_hour)`
- 可用 FLOPs = GPU 数量 $\times$ 峰值 TFLOPS $\times 10^{12} \times$ MFU $\times$ 总秒数
- Chinchilla Loss 就是练习 3 中的公式：$A/N^\alpha + B/D^\beta + L_\infty$
- 从参数量估算 `hidden_size`：利用 $N \approx 0.1875 d^3$ 反推 $d$
- 层数的经验比例：`num_layers = hidden_size // 64`

<details>
<summary>参考答案</summary>

```python
# 填空 1: 从美元预算计算 GPU 小时
budget_hours = budget_dollars / (self.num_gpus * self.gpu["cost_per_hour"])

# 填空 2: 计算可用 FLOPs
available_flops = self.num_gpus * self.gpu["peak_tflops"] * 1e12 * self.mfu * total_seconds

# 填空 3: Chinchilla Loss
return p["A"] / (N ** p["alpha"]) + p["B"] / (D ** p["beta"]) + p["L_inf"]

# 填空 4: 估算 hidden_size
d_raw = (target_params / 0.1875) ** (1/3)

# 填空 5: 估算层数
num_layers = hidden_size // 64
```

**验证:**
```python
def print_plan(label, plan):
    o = plan["optimal"]
    a = plan["architecture"]
    print(f"=== {label} ===")
    print(f"  FLOPs={plan['compute']['available_flops']:.2e}, "
          f"N={o['N']/1e9:.2f}B, D={o['D']/1e9:.0f}B, Loss={o['predicted_loss']:.4f}")
    print(f"  架构: d={a['hidden_size']}, L={a['num_layers']}, ${plan['budget']['cost_usd']:,.0f}")

# 小团队: 8x A100, 1000 小时
print_plan("8x A100, 1000h",
    TrainingBudgetPlanner("A100_80G", 8, 0.4).plan(budget_hours=1000))

# 中等团队: 64x H100, $100K
print_plan("64x H100, $100K",
    TrainingBudgetPlanner("H100_80G", 64, 0.4).plan(budget_dollars=100000))

# 大公司: 2048x H100, 30 天
print_plan("2048x H100, 30d",
    TrainingBudgetPlanner("H100_80G", 2048, 0.35).plan(budget_hours=720))
```

**解析：** `suggest_architecture` 中的经验法则只是粗略估计，实际需考虑显存限制、并行策略（TP/PP/DP）、batch size 等。大集群 MFU 通常更低（0.3-0.35），规划时应保守估计。

</details>

---

## MLM 代码训练模式

完成上面的固定填空后，试试随机挖空模式 -- 每次点击「刷新」会随机遮盖不同的代码片段，帮你彻底记住每一行。

### Transformer 参数量计算公式

<CodeMasker title="参数量统计 — Embedding / Attention / FFN / LayerNorm" :mask-ratio="0.15">
embedding_params = vocab_size * hidden_size

attn_params_per_layer = 4 * hidden_size * hidden_size

if use_gated_mlp:
    ffn_params_per_layer = 3 * hidden_size * ffn_hidden
else:
    ffn_params_per_layer = 2 * hidden_size * ffn_hidden

ln_params_per_layer = 2 * hidden_size

output_head_params = hidden_size * vocab_size

total_per_layer = attn_params_per_layer + ffn_params_per_layer + ln_params_per_layer
total = num_layers * total_per_layer + embedding_params + hidden_size
</CodeMasker>

### FLOPs 估算 (6ND) 与训练时间

<CodeMasker title="训练 FLOPs = 6ND，GPU 时间估算" :mask-ratio="0.15">
total_flops = 6 * num_params * num_tokens

effective_flops_per_sec = num_gpus * gpu_peak_tflops * 1e12 * mfu

training_seconds = total_flops / effective_flops_per_sec
training_hours = training_seconds / 3600

available_flops = total_seconds * num_gpus * gpu_peak_tflops * 1e12 * mfu
max_n = np.sqrt(available_flops / (6 * tokens_per_param))
</CodeMasker>

### Chinchilla 最优配比

<CodeMasker title="Chinchilla Loss 与最优 N/D 搜索" :mask-ratio="0.15">
def chinchilla_loss(N, D):
    loss = A / (N ** alpha) + B / (D ** beta) + L_inf
    return loss

def chinchilla_optimal(C):
    def loss_given_N(log_N):
        N = np.exp(log_N)
        D = C / (6 * N)
        if D < 1:
            return 1e10
        return chinchilla_loss(N, D)

    result = minimize_scalar(
        loss_given_N,
        bounds=(np.log(1e6), np.log(1e12)),
        method='bounded'
    )
    N_opt = np.exp(result.x)
    D_opt = C / (6 * N_opt)
    return int(N_opt), int(D_opt), result.fun
</CodeMasker>

### Loss 幂律拟合

<CodeMasker title="幂律拟合 L(N) = a * N^(-alpha) + L_inf" :mask-ratio="0.15">
def scaling_law_func(N, a, alpha, L_inf):
    return a * N ** (-alpha) + L_inf

p0 = [100.0, 0.1, 1.8]
popt, pcov = curve_fit(scaling_law_func, N_data, L_data, p0=p0, maxfev=10000)

predicted_loss = a * N_target ** (-alpha) + L_inf

ss_res = np.sum((L_data - L_pred) ** 2)
ss_tot = np.sum((L_data - np.mean(L_data)) ** 2)
r_squared = 1 - ss_res / ss_tot
</CodeMasker>
