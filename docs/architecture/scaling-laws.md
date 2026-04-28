---
title: "Scaling Laws"
description: "从 Kaplan 到 Chinchilla，计算最优策略与涌现能力"
topics: [scaling-laws, Chinchilla, compute-optimal, FLOPs, emergent-abilities, power-law]
prereqs: [architecture/transformer, training/pretraining]
---
# Scaling Laws

> **一句话总结:** Scaling Laws 揭示了模型性能与参数量、数据量、计算量之间的幂律关系——它告诉我们"给定预算，应该训练多大的模型、用多少数据"，是大模型时代最重要的实验科学发现之一。

## 在大模型体系中的位置

```
Scaling Laws ◄── 你在这里
  ├── 指导模型大小选择     → 7B vs 13B vs 70B 怎么选
  ├── 指导数据量规划       → 需要多少 token 的训练数据
  ├── 指导计算预算分配     → 买多少 GPU、训多少步
  └── 解释涌现能力         → 为什么大模型会"突然变聪明"
```

在决定训练一个新模型之前，Scaling Laws 是最重要的参考框架。它让模型训练从"凭经验试"变成了"可预测、可规划"。

---

## 核心概念：幂律关系

### 基本形式

Scaling Laws 的核心发现是：模型的测试损失（loss）与参数量 $N$、数据量 $D$、计算量 $C$ 之间存在**幂律关系**：

$$
L(N) \propto N^{-\alpha_N}, \quad L(D) \propto D^{-\alpha_D}, \quad L(C) \propto C^{-\alpha_C}
$$

其中 $\alpha$ 是幂律指数。这意味着：增大参数量/数据量/计算量，loss 会平滑下降，但收益递减。

**关键直觉**：在双对数坐标下（log-log plot），loss 与 N/D/C 的关系近似线性。这使得我们可以用小规模实验**预测**大规模模型的性能。

### 计算量公式

一个 Transformer 模型的前向传播计算量（FLOPs）近似为：

$$
C \approx 6ND
$$

其中 $N$ 是参数量，$D$ 是训练 token 数。系数 6 来自：前向传播约 $2ND$（每个 token 经过每个参数做一次乘加），反向传播约是前向的 2 倍（见 profiling.md），合计 $6ND$。

$$
C_{\text{total}} \approx 6 \times N_{\text{params}} \times D_{\text{tokens}}
$$

**快速估算**：7B 模型训练 2T tokens 的计算量 = $6 \times 7 \times 10^9 \times 2 \times 10^{12} = 8.4 \times 10^{22}$ FLOPs。

---

## Kaplan Scaling Laws (2020)

### 来源

OpenAI 的 Kaplan 等人在论文《Scaling Laws for Neural Language Models》中首次系统化地研究了 Scaling Laws。

### 核心发现

1. **模型大小比数据量更重要**：在固定计算预算下，应该优先增大模型参数量，即使数据训练不够充分（训练步数较少）。
2. **架构细节不太重要**：模型宽度、深度、头数等超参数的影响远小于总参数量 $N$ 的影响。
3. **固定预算的最优策略**：给定计算预算 $C$，应该用一个尽可能大的模型，即使只训练很少的 epoch。

### Kaplan 的经验公式

$$
L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty
$$

其中 $\alpha_N \approx 0.076$，$\alpha_D \approx 0.095$，$L_\infty$ 是理论最小损失。

**Kaplan 的实践建议**：模型参数量增大 10×，数据量只需增大约 1.7×。这意味着 Kaplan 建议"用大模型训少量数据"。

---

## Chinchilla Scaling Laws (2022)

### DeepMind 的挑战

DeepMind 的 Hoffmann 等人在《Training Compute-Optimal Large Language Models》中推翻了 Kaplan 的建议，提出了截然不同的结论。

### 核心发现

**模型参数量和数据量应该同等重要地扩展。** 具体来说：

$$
N_{\text{opt}} \propto C^{0.50}, \quad D_{\text{opt}} \propto C^{0.50}
$$

即计算预算增大 10×，最优模型大小和最优数据量应各增大约 $\sqrt{10} \approx 3.16$×。

### Chinchilla 最优比例

**经验法则：最优训练 token 数 ≈ 20 × 参数量。**

| 参数量 | 最优训练 token 数 | 等效 epoch |
|--------|-------------------|-----------|
| 1B | 20B tokens | - |
| 7B | 140B tokens | - |
| 70B | 1.4T tokens | - |
| 175B | 3.5T tokens | - |

### Chinchilla vs GPT-3

这是 Chinchilla 论文最令人震撼的实验：

| 模型 | 参数量 | 训练 tokens | 性能 |
|------|--------|------------|------|
| GPT-3 | 175B | 300B | 基准 |
| Gopher | 280B | 300B | 比 GPT-3 好一点 |
| **Chinchilla** | **70B** | **1.4T** | **全面超越 GPT-3 和 Gopher** |

Chinchilla 用**不到 GPT-3 一半的参数量**，通过训练更多数据，就实现了更好的性能。更重要的是，更小的模型意味着**更低的推理成本**（推理成本正比于参数量，而非训练量）。

### 为什么 Kaplan 和 Chinchilla 结论不同？

| 差异点 | Kaplan (2020) | Chinchilla (2022) |
|--------|--------------|-------------------|
| 最优分配 | 偏向大模型少数据 | 模型和数据同等扩展 |
| N:D 比例 | 约 1:1.7 | 约 1:20 |
| 实验规模 | 较小（最大 1.5B） | 更大（最大 70B+） |
| 核心原因 | 没有充分探索数据量的影响 | 更系统的实验设计 |

**工业界的选择**：Chinchilla 之后，几乎所有主流模型都遵循"充分训练"策略。Llama 2 (7B) 训了 2T tokens（远超 Chinchilla 最优的 140B），因为推理成本是持续开销，多花训练成本换来更小但更强的模型是值得的。

---

## 后 Chinchilla 时代：推理最优

### 过训练（Over-training）策略

Chinchilla 假设训练成本是唯一关注点。但在实际部署中，**推理成本远大于训练成本**（一次训练 vs 百万次推理）。因此现代模型倾向于"过训练"——用比 Chinchilla 最优更多的数据训练更小的模型。

$$
\text{总成本} = C_{\text{train}} + N_{\text{requests}} \times C_{\text{inference}}(N)
$$

当 $N_{\text{requests}}$ 很大时，减小 $N$（模型更小）带来的推理节省远超多训一些数据的成本。

| 模型 | 参数量 | 训练 tokens | tokens/param | 策略 |
|------|--------|------------|-------------|------|
| Chinchilla 最优 (7B) | 7B | 140B | 20× | Compute-optimal |
| Llama 2 7B | 7B | 2T | 286× | 过训练 14× |
| Llama 3 8B | 8B | 15T | 1875× | 过训练 94× |
| Phi-3 mini | 3.8B | 3.3T | 868× | 过训练 + 高质量数据 |

**趋势**：小模型 + 大数据 + 高质量数据 = 最佳性价比。

### DeepSeek 的 Scaling Laws 研究

DeepSeek 团队进一步探索了 MoE（Mixture of Experts）模型的 Scaling Laws。MoE 模型的总参数量很大，但每个 token 只激活一部分参数（active parameters），使得 FLOPs 远小于同等参数量的稠密模型。

DeepSeek 发现：MoE 的 Scaling Laws 与稠密模型不同，需要分别考虑总参数量和激活参数量。这指导了 DeepSeek-V2/V3 的 MoE 架构设计。

---

## 涌现能力（Emergent Abilities）

### 什么是涌现

涌现能力指的是：某些能力在小模型中完全不存在，但在模型规模超过某个阈值后"突然出现"。

经典例子：
- **数学推理**（如 GSM8K）：8B 以下的模型几乎无法解决，65B+ 的模型开始展现能力
- **思维链推理**（Chain-of-Thought）：小模型使用 CoT 反而变差，大模型使用 CoT 显著提升
- **少样本学习**（Few-shot In-Context Learning）：小模型无法从 context 中学习新模式

### 涌现是真的吗？

2023 年 Stanford 的 Schaeffer 等人在《Are Emergent Abilities of Large Language Models a Mirage?》中提出挑战：

**核心观点**：涌现可能是评估指标的假象而非模型能力的真实跳变。

1. **非线性指标的假象**：如果用 0/1 准确率（全对才算对），看起来像阶跃函数；如果用连续指标（如 BLEU、token-level accuracy），性能是平滑增长的
2. **选择性评估**：只选择那些恰好在某个规模出现跳变的任务来展示"涌现"

**当前共识**：
- 在连续指标下，大多数"涌现"消失，性能是平滑的幂律增长
- 但部分能力（如复杂推理）的确在小模型中极弱、大模型中显著增强，这不完全是指标假象
- 实用意义不变：某些任务确实需要足够大的模型才能做好

---

## 实用意义

### 如何估算训练成本

```python
def estimate_training_cost(
    n_params_b: float,        # 参数量（B）
    n_tokens_t: float,        # 训练 token 数（T）
    gpu_flops_tflops: float = 312,  # A100 BF16 峰值 TFLOPS
    gpu_utilization: float = 0.4,   # 实际利用率（MFU）
    gpu_cost_per_hour: float = 2.0, # 单卡每小时成本（美元）
):
    """估算训练成本"""
    # 总计算量
    total_flops = 6 * n_params_b * 1e9 * n_tokens_t * 1e12
    
    # 单卡每秒有效 FLOPS
    effective_flops = gpu_flops_tflops * 1e12 * gpu_utilization
    
    # 需要的 GPU 小时数
    gpu_hours = total_flops / effective_flops / 3600
    
    # 成本
    cost = gpu_hours * gpu_cost_per_hour
    
    print(f"=== Training Cost Estimate ===")
    print(f"Model: {n_params_b}B params, {n_tokens_t}T tokens")
    print(f"Total FLOPs: {total_flops:.2e}")
    print(f"GPU hours (single A100): {gpu_hours:,.0f}")
    print(f"With 1000 GPUs: {gpu_hours/1000:.1f} hours = {gpu_hours/1000/24:.1f} days")
    print(f"Estimated cost: ${cost:,.0f}")
    return cost

# 估算几个经典模型的训练成本
estimate_training_cost(7, 2)      # Llama 2 7B
# Total FLOPs: 8.40e+22, GPU hours: 187K, ~8 days on 1000 GPUs, ~$374K

estimate_training_cost(70, 2)     # Llama 2 70B
# Total FLOPs: 8.40e+23, GPU hours: 1.87M, ~78 days on 1000 GPUs, ~$3.7M

estimate_training_cost(405, 15)   # Llama 3 405B 量级
# Total FLOPs: 3.65e+25, GPU hours: 81M, ~3375 days on 1000 GPUs
```

### 用小模型预测大模型性能

Scaling Laws 最实用的价值：**在小规模实验中预测大规模结果**。

```python
import numpy as np
from scipy.optimize import curve_fit

def power_law(x, a, b, c):
    """幂律函数：L = a * x^(-b) + c"""
    return a * np.power(x, -b) + c

# 假设你在 125M, 350M, 1.3B, 2.7B 规模上做了实验
model_sizes = np.array([125e6, 350e6, 1.3e9, 2.7e9])
losses = np.array([3.42, 3.10, 2.85, 2.72])

# 拟合幂律
params, _ = curve_fit(power_law, model_sizes, losses, p0=[10, 0.1, 2.5])

# 预测 7B 和 13B 的 loss
for size in [7e9, 13e9, 70e9]:
    predicted_loss = power_law(size, *params)
    print(f"{size/1e9:.0f}B model predicted loss: {predicted_loss:.3f}")
```

---

## 苏格拉底时刻

1. 为什么 Chinchilla 的结论和 Kaplan 不同？哪个更适合当前的工业场景？
2. Llama 3 用 15T tokens 训练 8B 模型（1875× 过训练），这违反了 Chinchilla 最优吗？为什么这反而是更好的策略？
3. 如果你有 $100 万预算训练一个对话模型，你会选择训练 7B 模型还是 70B 模型？各需要多少数据？
4. "涌现能力是评估指标的假象"——这个观点如果成立，对模型选型有什么影响？
5. MoE 模型的 Scaling Laws 和稠密模型有什么不同？这如何指导 MoE 架构设计？

---

## 常见问题 & 面试考点

| 问题 | 要点 |
|------|------|
| Chinchilla 的核心结论？ | 计算最优 = 模型和数据同等扩展，token/param ≈ 20:1 |
| 为什么现代模型都"过训练"？ | 推理成本 >> 训练成本，更小的模型推理更便宜 |
| 6ND 公式什么含义？ | 训练 FLOPs ≈ 6 × 参数量 × 训练 tokens（前向2 + 反向4） |
| 涌现能力是真的吗？ | 连续指标下大多消失，但复杂推理确实需要足够大的模型 |
| 如何做小模型实验预测大模型？ | 在 log-log 图上拟合幂律，外推到目标规模 |

---

## 推荐资源

- **Kaplan et al.《Scaling Laws for Neural Language Models》** — 首篇系统性 Scaling Laws 论文
- **Hoffmann et al.《Training Compute-Optimal Large Language Models》** — Chinchilla 论文
- **Schaeffer et al.《Are Emergent Abilities a Mirage?》** — 对涌现能力的质疑
- **DeepSeek 技术报告** — MoE Scaling Laws 的实践探索
- **Sardana & Frankle《Beyond Chinchilla-Optimal》** — 推理最优视角的 Scaling Laws
