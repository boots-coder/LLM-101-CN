---
title: "预训练"
description: "Next Token Prediction、Scaling Laws（Kaplan/Chinchilla）、训练成本估算"
topics: [pretraining, next-token-prediction, scaling-laws, Chinchilla, AdamW, mixed-precision, compute-optimal]
prereqs: [architecture/gpt]
---
# 预训练

> 预训练是让模型从海量文本中学会语言的过程，是 LLM 所有能力的基础。模型在这一阶段通过预测下一个 token，从万亿级语料中压缩出语言结构、世界知识和推理模式。

## 在大模型体系中的位置

```
预训练（本章）──> SFT 微调 ──> RLHF/DPO 对齐 ──> 部署推理
   │                │              │
   │                │              └─ 偏好数据（chosen/rejected）
   │                └─ 指令数据（instruction-response）
   └─ 海量无标注文本（万亿 token）
```

预训练是整个流水线中**计算量最大、耗时最长、成本最高**的阶段。以 Llama 3 70B 为例，预训练在 15T token 上进行，消耗约 6.4M GPU-hours（H100）。但正是这一阶段赋予模型所有的基础能力——后续的 SFT 和 RLHF 只是在预训练的基础上"激活"和"对齐"。

---

## 预训练的本质

### Next Token Prediction 目标

预训练的核心目标极其简单：**给定前面的所有 token，预测下一个 token**。

给定序列 $x_1, x_2, \dots, x_T$，模型学习条件概率分布：

$$P(x_t | x_1, x_2, \dots, x_{t-1}; \theta)$$

训练目标是最大化整个序列的对数似然（等价于最小化交叉熵损失）：

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)$$

**交叉熵损失的直觉**：当模型预测正确类别的概率越高，$-\log P$ 越小（损失越低）。

```python
# 交叉熵的本质
import torch
import torch.nn.functional as F

# 给定两个概率分布，计算交叉熵
q = torch.tensor([0.05, 0.9, 0.05])  # 模型预测（预测较准确）
p = torch.tensor([0.0, 1.0, 0.0])    # 目标分布（真实标签是第2类）

entropy = -p * torch.log(q)
print(entropy.sum())  # tensor(0.1054) —— 预测准确时损失低

q_bad = torch.tensor([0.3, 0.5, 0.2])  # 模型预测（不够准确）
entropy_bad = -p * torch.log(q_bad)
print(entropy_bad.sum())  # tensor(0.6931) —— 预测不准时损失高
```

在实际分类问题中，由于目标分布是 one-hot 的（只有正确类别概率为 1），交叉熵可以简化为：

$$\mathcal{L} = -\log q_{y}$$

其中 $y$ 是正确类别的索引。这就是 PyTorch 中 `nn.CrossEntropyLoss` 的实现方式：

```python
# 手动实现批量交叉熵
def manual_ce_loss(targets, logits):
    """
    targets: [batch]       — 每个样本的正确类别索引
    logits:  [batch, num_cls] — 模型输出的未归一化分数
    """
    batch, _ = logits.shape
    probs = F.softmax(logits, dim=-1)
    row_idx = torch.arange(batch)
    # 只取正确类别的 log 概率
    log_probs = probs[row_idx, targets].log()
    loss = -log_probs.mean()
    return loss

# 验证与 PyTorch 实现一致
import torch.nn as nn
loss_fn = nn.CrossEntropyLoss()
logits = torch.randn(6, 8)
targets = torch.randint(high=8, size=(6,))
print(manual_ce_loss(targets, logits))  # tensor(2.xxxx)
print(loss_fn(logits, targets))          # tensor(2.xxxx) —— 结果一致
```

**交叉熵梯度的优美性质**：CE + Softmax 对 logits 的梯度有一个极简的形式：

$$\frac{\partial \mathcal{L}}{\partial z_i} = q_i - p_i$$

即**预测概率减去目标概率**。这意味着：正确类别的梯度方向是增大其概率（负梯度），错误类别的梯度方向是减小其概率。

```python
# 验证梯度公式：CE + Softmax 对 logits 的梯度
z = torch.tensor([1.0, 2.0, 3.0])
p = torch.tensor([0.0, 1.0, 0.0])  # 目标分布（one-hot）
q = F.softmax(z, dim=0)             # softmax 输出

# 通过链式法则手动计算 dL/dz
dL_dp = -(p / q)                              # dL/dq
J_softmax = torch.diag(q) - torch.outer(q, q) # softmax 雅可比矩阵
dL_dz = dL_dp @ J_softmax                     # 链式法则
print(dL_dz)    # tensor([ 0.0900, -0.7553,  0.6652])
print(q - p)     # tensor([ 0.0900, -0.7553,  0.6652]) —— 完全一致
```

### Causal Language Modeling vs Masked Language Modeling

| 特性 | Causal LM (GPT 系列) | Masked LM (BERT 系列) |
|------|----------------------|----------------------|
| 预测方向 | 从左到右，预测下一个 token | 双向，预测被 mask 的 token |
| 注意力模式 | Causal mask（下三角） | 全注意力 |
| 生成能力 | 天然支持自回归生成 | 不擅长开放式生成 |
| 代表模型 | GPT, Llama, Qwen | BERT, RoBERTa |
| 当前主流 | 几乎所有 LLM 都采用 | 主要用于理解任务 |

现代 LLM 几乎全部采用 Causal LM。原因：**生成能力是 LLM 最核心的能力**，而 Causal LM 的训练目标天然与自回归生成对齐。

### 预训练赋予模型什么能力？

预训练并非简单的"记忆"，而是通过预测下一个 token 这一目标，迫使模型学会：

1. **语法和语言结构**：主谓宾搭配、时态一致性、代词指代
2. **世界知识**："北京是中国的首都"、"水在 100 度沸腾"
3. **推理模式**：因果推理、类比推理、数学逻辑
4. **代码理解**：编程语言的语法、API 使用模式、算法结构
5. **多语言能力**：如果训练数据包含多种语言，模型会学到跨语言的表示

---

## Scaling Laws

### Kaplan et al. (2020) 的发现

OpenAI 在 2020 年发现了语言模型性能与三个关键变量之间的幂律关系：

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

其中：
- $N$ = 模型参数量（非嵌入层参数）
- $D$ = 训练数据量（token 数）
- $C$ = 计算量（FLOPs）
- $L$ = 测试集交叉熵损失

核心发现：**性能与这三个变量分别呈幂律关系，且在很大范围内保持平滑**。这意味着可以用小实验预测大模型的性能。

### Chinchilla 法则

DeepMind 的 Chinchilla 论文（Hoffmann et al., 2022）修正了 Kaplan 的结论，提出了**计算最优**的训练策略：

> 给定固定的计算预算 $C$，模型参数量 $N$ 和训练数据量 $D$ 应当等比例增长。

具体而言：

$$N_{opt} \propto C^{0.50}, \quad D_{opt} \propto C^{0.50}$$

经验法则：**每个参数对应约 20 个训练 token**。即：

| 模型参数量 | 最优训练 token 数 | 计算量 (FLOPs) |
|-----------|-----------------|---------------|
| 400M | 8B | 1.9e19 |
| 1B | 20B | 1.2e20 |
| 7B | 140B | 5.8e21 |
| 13B | 260B | 2.0e22 |
| 70B | 1.4T | 5.9e23 |
| 175B | 3.5T | 3.7e24 |

### 对实践的指导意义

1. **不要只加大模型**：Chinchilla (70B, 1.4T tokens) 打败了 Gopher (280B, 300B tokens)，因为后者严重"过大欠训练"
2. **数据是瓶颈**：对于大模型，高质量数据的需求量巨大。Llama 3 在 15T token 上训练 8B 和 70B 模型，远超 Chinchilla 最优比例，说明**过度训练（over-training）在推理成本敏感场景下是合理的**
3. **预测训练成本**：$C \approx 6ND$（近似公式），其中 $C$ 是 FLOPs，$N$ 是参数量，$D$ 是 token 数

> **实际案例**：训练 Llama 3 8B 在 15T tokens 上，计算量约 $6 \times 8 \times 10^9 \times 15 \times 10^{12} = 7.2 \times 10^{23}$ FLOPs。以 H100 (989 TFLOPS BF16) 和 40% MFU 计算，需要约 $7.2 \times 10^{23} / (989 \times 10^{12} \times 0.4 \times 3600) \approx 505{,}000$ GPU-hours。

### Scaling Laws 深度：从理论到实践

#### Kaplan Scaling Laws（OpenAI 2020）详解

Kaplan 等人在 2020 年系统地研究了语言模型的 loss 与三个核心变量之间的幂律关系。这些关系在数个数量级上保持惊人的平滑：

**单变量幂律**：固定其中两个变量，只增长第三个：

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N} \approx \left(\frac{8.8 \times 10^{13}}{N}\right)^{0.076}$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D} \approx \left(\frac{5.4 \times 10^{13}}{D}\right)^{0.095}$$

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C} \approx \left(\frac{3.1 \times 10^8}{C}\right)^{0.050}$$

**联合幂律**：当同时优化 $N$ 和 $D$ 时：

$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}$$

**Kaplan 的关键结论**：

1. **模型大小比数据量重要**：$\alpha_N = 0.076 > \alpha_D = 0.095$（注意：$\alpha$ 越小意味着对 loss 的边际收益越大）。Kaplan 认为在计算预算有限时，应该优先加大模型
2. **最优分配**：给定计算预算 $C$，$N_{opt} \propto C^{0.73}$，$D_{opt} \propto C^{0.27}$——这意味着大部分预算应该给模型参数
3. **架构细节次要**：在合理范围内，层数/宽度/头数的具体选择对 scaling 影响不大

#### Chinchilla Scaling Laws（DeepMind 2022）的修正

Hoffmann 等人（DeepMind）在 2022 年发表了 Chinchilla 论文，通过更严格的实验设计修正了 Kaplan 的结论。

**实验方法**：他们用三种不同方法估计最优分配：
1. 固定计算预算，训练大量不同 $N$ 和 $D$ 的模型
2. 固定模型大小，变化训练 token 数
3. 参数化拟合 $L(N, D)$

**Chinchilla 的幂律**：

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E$$

其中 $\alpha \approx 0.34$，$\beta \approx 0.28$，$E \approx 1.69$（不可约损失）。

**compute-optimal 训练的核心结论**：

$$N_{opt} \propto C^{0.50}, \quad D_{opt} \propto C^{0.50}$$

这与 Kaplan 的结论（$N \propto C^{0.73}$）**截然不同**！Chinchilla 认为模型参数和数据量应该**等比例增长**。

**为什么 Kaplan 和 Chinchilla 结论不同？**

1. Kaplan 的实验中，大模型没有训练到收敛，导致高估了"增大模型"的收益
2. Kaplan 对计算量 $C \approx 6ND$ 的近似中忽略了 embedding 层
3. Chinchilla 使用了更大范围的模型-数据组合（400 多个实验点）

**Chinchilla 对工业界的深远影响**：

在 Chinchilla 之前，业界的信条是"bigger is better"——GPT-3 175B 只用了 300B tokens 训练。Chinchilla 70B 用 1.4T tokens 训练后，性能**超过了** Gopher 280B（300B tokens），仅用约 $1/4$ 的推理成本。

这直接改变了后续所有模型的训练策略：

| 模型 | 参数量 | 训练 Token 数 | Token/参数比 | 策略 |
|------|-------|--------------|-------------|------|
| GPT-3 | 175B | 300B | 1.7 | 欠训练 |
| Gopher | 280B | 300B | 1.1 | 严重欠训练 |
| Chinchilla | 70B | 1.4T | 20 | compute-optimal |
| Llama 2 70B | 70B | 2T | 29 | 略超 Chinchilla |
| Llama 3 8B | 8B | 15T | 1875 | 大幅过训练 |
| Llama 3 70B | 70B | 15T | 214 | 大幅过训练 |

**为什么 Llama 3 大幅超过 Chinchilla 比例？** 因为 Chinchilla 优化的是**训练计算量**，而实际部署时还要考虑**推理成本**。小模型训练更久，推理更快、更省钱。对于部署量大的模型，多花训练成本换来的推理节省远超投入。

#### 推理 Scaling Laws（OpenAI 2024）

2024 年，随着 OpenAI o1 等"推理模型"的出现，一种新的 Scaling Law 被提出——**test-time compute scaling**：

$$L(c_{\text{test}}) \propto c_{\text{test}}^{-\gamma}$$

其中 $c_{\text{test}}$ 是推理时使用的计算量（如 Chain-of-Thought 的 token 数、搜索树的大小）。

**核心发现**：

1. **推理时的计算量也遵循幂律**：让模型"思考更久"（生成更多推理步骤），可以持续提升准确率
2. **训练计算和推理计算可以互换**：在某些任务上，一个小模型 + 大量推理计算，可以匹敌大模型 + 少量推理计算
3. **最优分配**：给定总预算（训练 + 推理），应该在两者之间寻找最优分配

```
传统 Scaling:       推理 Scaling:
  更大模型 → 更好      更多推理步骤 → 更好
  1B → 7B → 70B       1 step → 10 steps → 100 steps
  训练时确定能力        推理时释放能力
```

这意味着 Scaling Laws 的维度从"预训练三角"（$N, D, C_{\text{train}}$）扩展到了四维：$N, D, C_{\text{train}}, C_{\text{test}}$。

#### Scaling Laws 的实用价值

**1. 训练成本预估**

在启动大规模训练前，先用小模型验证 scaling trend：

```python
import numpy as np
from scipy.optimize import curve_fit

def power_law(x, a, b, c):
    """幂律函数: L = a / x^b + c"""
    return a / np.power(x, b) + c

# 用小规模实验数据拟合 Scaling Law
# 模型参数量（单位：百万）
N_values = np.array([25, 50, 100, 200, 400, 800])
# 对应的验证集 loss
loss_values = np.array([3.85, 3.52, 3.25, 3.05, 2.88, 2.75])

# 拟合幂律参数
params, _ = curve_fit(power_law, N_values, loss_values,
                      p0=[10, 0.1, 2.0], maxfev=10000)

a, b, c = params
print(f"拟合结果: L(N) = {a:.2f} / N^{b:.4f} + {c:.4f}")
print(f"不可约损失 (irreducible loss): {c:.4f}")

# 预测更大模型的性能
for target_N in [7000, 13000, 70000]:  # 7B, 13B, 70B
    predicted_loss = power_law(target_N, *params)
    print(f"预测 {target_N}M 参数模型 loss: {predicted_loss:.4f}")
```

**2. 最优模型大小选择**

给定计算预算 $C$（FLOPs），按 Chinchilla 法则：

```python
def chinchilla_optimal(compute_budget_flops):
    """
    给定计算预算，计算 Chinchilla-optimal 的模型大小和数据量
    基于 C ≈ 6ND 和 N_opt ∝ C^0.5, D_opt ∝ C^0.5
    经验关系: D_opt ≈ 20 * N_opt
    """
    # 从 C = 6ND 和 D = 20N 得: C = 120 * N^2
    N_opt = np.sqrt(compute_budget_flops / 120)
    D_opt = 20 * N_opt

    return {
        'params': N_opt,
        'tokens': D_opt,
        'params_B': N_opt / 1e9,
        'tokens_T': D_opt / 1e12,
        'compute_flops': compute_budget_flops,
    }

# 不同计算预算下的最优配置
budgets = {
    '小型实验': 1e19,
    '7B 级别': 6e21,
    '70B 级别': 6e23,
    '175B 级别': 4e24,
}

for name, budget in budgets.items():
    result = chinchilla_optimal(budget)
    print(f"{name} (C={budget:.0e}):")
    print(f"  最优参数量: {result['params_B']:.1f}B")
    print(f"  最优数据量: {result['tokens_T']:.2f}T tokens")
    print()

# 输出:
# 小型实验 (C=1e+19): 最优参数量: 0.3B, 最优数据量: 0.01T tokens
# 7B 级别 (C=6e+21): 最优参数量: 7.1B, 最优数据量: 0.14T tokens
# 70B 级别 (C=6e+23): 最优参数量: 70.7B, 最优数据量: 1.41T tokens
# 175B 级别 (C=4e+24): 最优参数量: 182.6B, 最优数据量: 3.65T tokens
```

**3. 数据需求规划**

```python
def estimate_data_needs(target_params_B, strategy='chinchilla'):
    """估算不同训练策略下的数据需求"""
    N = target_params_B * 1e9

    if strategy == 'chinchilla':
        # Chinchilla-optimal: 20 tokens/param
        tokens = 20 * N
    elif strategy == 'llama3':
        # Llama 3 风格: 过度训练，优化推理成本
        # 经验值: 小模型 ~2000 tokens/param, 大模型 ~200 tokens/param
        tokens_per_param = 2000 if target_params_B < 20 else 200
        tokens = tokens_per_param * N
    elif strategy == 'balanced':
        # 折中: 100 tokens/param
        tokens = 100 * N

    compute = 6 * N * tokens  # FLOPs

    return {
        'tokens_T': tokens / 1e12,
        'compute_flops': compute,
        'h100_hours': compute / (989e12 * 0.4 * 3600),
    }

for model_size in [1, 7, 13, 70]:
    print(f"\n=== {model_size}B 模型 ===")
    for strategy in ['chinchilla', 'llama3', 'balanced']:
        result = estimate_data_needs(model_size, strategy)
        print(f"  {strategy:12s}: {result['tokens_T']:8.1f}T tokens, "
              f"{result['h100_hours']:10,.0f} H100-hours")

# 输出示例:
# === 7B 模型 ===
#   chinchilla  :      0.1T tokens,      2,960 H100-hours
#   llama3      :     14.0T tokens,    414,444 H100-hours
#   balanced    :      0.7T tokens,     20,723 H100-hours
```

**4. 拟合自己的 Scaling Law 曲线**

```python
import matplotlib.pyplot as plt

def fit_and_plot_scaling_law(N_values, loss_values, title="Scaling Law"):
    """
    拟合并可视化 Scaling Law 曲线
    N_values: 模型参数量（列表）
    loss_values: 对应的验证 loss（列表）
    """
    N = np.array(N_values, dtype=np.float64)
    L = np.array(loss_values, dtype=np.float64)

    # 拟合 L(N) = a / N^b + c
    params, cov = curve_fit(power_law, N, L,
                            p0=[10, 0.1, 2.0], maxfev=10000)
    a, b, c = params

    # 绘制拟合曲线
    N_range = np.logspace(np.log10(N.min() * 0.5),
                          np.log10(N.max() * 100), 200)
    L_predicted = power_law(N_range, *params)

    plt.figure(figsize=(10, 6))
    plt.scatter(N, L, s=100, c='red', zorder=5, label='实验数据')
    plt.plot(N_range, L_predicted, 'b--', label=f'拟合: L = {a:.2f}/N^{b:.4f} + {c:.4f}')
    plt.axhline(y=c, color='gray', linestyle=':', label=f'不可约损失 = {c:.4f}')
    plt.xscale('log')
    plt.xlabel('模型参数量 (M)')
    plt.ylabel('验证集 Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('scaling_law.png', dpi=150)
    print(f"Scaling Law 拟合完成, 图表已保存")

    return params

# 使用示例
N_experiment = [25, 50, 100, 200, 400, 800, 1500]
loss_experiment = [3.85, 3.52, 3.25, 3.05, 2.88, 2.75, 2.65]

params = fit_and_plot_scaling_law(N_experiment, loss_experiment,
                                  title="Validation Loss Scaling Law")
```

> **面试考点：Scaling Laws 的意义是什么？** Scaling Laws 使得大模型训练从"黑盒炼丹"变为"工程可预测"。通过小规模实验拟合幂律参数，可以在花费数百万美元训练前，预估目标模型的大致性能、所需数据量和训练时间。这是大模型训练从艺术走向科学的关键一步。

---

## 数据准备

### 数据来源

现代 LLM 的预训练数据通常来自以下几个大类：

| 数据源 | 规模 | 特点 |
|-------|------|------|
| Common Crawl | 数万亿 token | 覆盖面广但质量参差不齐 |
| Wikipedia | ~40 亿 token（英文） | 高质量、结构化 |
| GitHub 代码 | 数千亿 token | 提升模型代码能力 |
| 书籍语料 | 数百亿 token | 长文本、高质量 |
| 学术论文 (ArXiv) | 数百亿 token | 数学和科学推理 |
| StackOverflow/论坛 | 数百亿 token | 问答格式、实践知识 |

工程上推荐用 `datasets.load_dataset(..., streaming=True)` 直接加载分片化语料；如果是 JSONL 中间产物，每行一个 `{"text": ...}`。预训练样本的标准流程是：**文档级原文 → tokenize → 拼成长 token 序列 → 按 `max_seq_len` 切块 → 每块末尾插入 `<eos>` 作为文档边界**。具体加载与切块的实现请参考 [training/datasets](/training/datasets#数据格式)。

### 数据清洗流水线

原始网页数据极其嘈杂，需要多层清洗：

```
原始 Common Crawl 快照
    │
    ├── 1. URL 过滤：移除成人站点、广告站点、已知低质量域名
    │
    ├── 2. 语言识别：使用 fastText 分类器，保留目标语言（如英文 > 0.65）
    │
    ├── 3. 质量过滤：
    │       ├── 基于规则：行长度、特殊字符比例、重复行比例
    │       ├── 基于困惑度：用 KenLM 计算 perplexity，过滤高困惑度文档
    │       └── 基于分类器：训练质量分类器（以 Wikipedia 为正例）
    │
    ├── 4. 去重：
    │       ├── 精确去重：文档级 SHA-256 哈希
    │       └── 模糊去重：MinHash + LSH（下文详述）
    │
    └── 5. 去污染：移除与评测集（MMLU, HumanEval 等）重叠的内容

    最终保留约 10-15% 的原始数据
```

### FineWeb 与 RedPajama

**FineWeb**（HuggingFace, 2024）是目前最大的开源英文网页数据集之一（15T token）。其关键创新在于：
- 使用更严格的质量过滤器，基于教育内容评分（educational score）
- FineWeb-Edu 子集在教育类 benchmark 上显著优于全量数据
- 证明了**数据质量筛选比简单扩大规模更有效**

**RedPajama**（Together AI）旨在复现 Llama 的训练数据：
- RedPajama v1: 1.2T token，复现 Llama 1 的数据配比
- RedPajama v2: 30T token 的原始数据 + 质量信号标注，支持用户自定义过滤

### 数据配比

Llama 3 的数据配比策略（Meta, 2024）：

| 数据类型 | 占比 | 说明 |
|---------|------|------|
| 网页文本 | ~50% | 经过严格质量过滤的 Common Crawl |
| 代码 | ~17% | GitHub 代码 |
| 数学 | ~4.5% | 数学推理相关内容 |
| 书籍 | ~4.5% | 长文本、连贯叙述 |
| 学术论文 | ~4.5% | 科学推理 |
| 多语言 | ~20% | 非英语数据 |

关键策略：
- **数据上采样**（upsampling）：对高质量数据（代码、数学）进行多次重复使用
- **动态调整**：训练后期增加高质量数据比例（即 Data Curriculum）
- **去重比重复更重要**：Llama 3 对数据进行了 4 轮去重

---

## 优化器

### AdamW 详解

几乎所有现代 LLM 的预训练都使用 AdamW 优化器。它是 Adam 的改进版本，核心修复了 weight decay 的实现方式。

**Adam 算法推导**：

给定参数 $\theta$ 和梯度 $g_t = \nabla_\theta \mathcal{L}_t$，Adam 维护两个指数移动平均：

1. **一阶矩估计**（梯度的均值）：$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
2. **二阶矩估计**（梯度平方的均值）：$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$

由于初始化为零，需要偏差校正：

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

参数更新：

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Weight Decay vs L2 正则化**：

在标准 SGD 中，L2 正则化与 weight decay 是等价的。但在 Adam 中两者不等价：
- L2 正则化将正则项加入损失函数，梯度会经过 Adam 的自适应缩放
- Weight Decay 直接在参数更新时减去 $\lambda \theta_t$，不经过自适应缩放

AdamW 选择后者（解耦 weight decay），在实践中表现更好。

下面用一个最小可跑示例来把这套数学公式落到代码上。优化器内部直接持有被更新的张量（与 `torch.optim` 的 in-place 风格一致），调用 `step()` 时只接收当前梯度，省去了把权重在外部反复传进传出的样板代码。

```python
import torch

# ---- AdamW：解耦权重衰减的从零实现 ----
class AdamWFromScratch:
    """
    用 PyTorch 官方 torch.optim.AdamW 的命名风格手写一遍，
    便于把数学推导与工程实现一一对应。
    """
    def __init__(self, parameter, lr=3e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=1e-2):
        self.parameter = parameter           # 直接持有引用
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        # 与 PyTorch 内部状态字典对齐的命名
        self.exp_avg = torch.zeros_like(parameter)
        self.exp_avg_sq = torch.zeros_like(parameter)
        self.update_step = 0

    @torch.no_grad()
    def step(self, gradients):
        self.update_step += 1
        s = self.update_step

        # 一阶 / 二阶矩的指数滑动平均
        self.exp_avg.mul_(self.b1).add_(gradients, alpha=1 - self.b1)
        self.exp_avg_sq.mul_(self.b2).addcmul_(gradients, gradients,
                                               value=1 - self.b2)

        # 偏差校正
        bias_c1 = 1 - self.b1 ** s
        bias_c2 = 1 - self.b2 ** s
        denom = (self.exp_avg_sq / bias_c2).sqrt().add_(self.eps)

        # 解耦的 weight decay：先把参数自身按比例收缩，再做自适应步长更新
        if self.weight_decay != 0:
            self.parameter.mul_(1 - self.lr * self.weight_decay)
        self.parameter.addcdiv_(self.exp_avg, denom,
                                value=-self.lr / bias_c1)


# ---- 玩具任务：三分类 softmax 回归 ----
def fit_demo(num_steps=600, n_features=4, n_classes=3, n_samples=64, seed=0):
    torch.manual_seed(seed)

    # 人造可分二/多分类数据：每类一个高斯簇
    centers = torch.randn(n_classes, n_features) * 2.5
    labels = torch.arange(n_classes).repeat_interleave(n_samples // n_classes)
    features = centers[labels] + 0.6 * torch.randn(labels.numel(), n_features)

    # 待学习的权重矩阵（softmax 回归，无 bias 简化）
    weights = torch.zeros(n_features, n_classes, requires_grad=True)
    optimizer = AdamWFromScratch(weights, lr=5e-2, weight_decay=1e-3)

    for step in range(num_steps):
        logits = features @ weights
        loss = torch.nn.functional.cross_entropy(logits, labels)

        weights.grad = None
        loss.backward()
        optimizer.step(weights.grad)

        if step % 100 == 0 or step == num_steps - 1:
            acc = (logits.argmax(dim=-1) == labels).float().mean().item()
            print(f"step {step:4d} | loss={loss.item():.4f} | acc={acc:.3f}")

    return weights


if __name__ == "__main__":
    fit_demo()
# 典型输出：loss 从 ~1.10 平滑下降到 0.1 以下，准确率收敛到 1.0
```

**算法来源 / 工业实现 / 教学实现 三处对照阅读**：

- 论文原文：Kingma & Ba, *Adam: A Method for Stochastic Optimization*，[arXiv:1412.6980](https://arxiv.org/abs/1412.6980)；AdamW 解耦修正参考 Loshchilov & Hutter, [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)。
- PyTorch 官方实现（工业级，包含 foreach / fused kernel 等优化路径）：[pytorch/torch/optim/adamw.py](https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py)，状态字段命名 `exp_avg` / `exp_avg_sq` 与本文保持一致。
- 真实 LLM 训练里如何挂上 AdamW（教学实现）：[rasbt/LLMs-from-scratch ch05/gpt_train.py](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/gpt_train.py)，直接调用 `torch.optim.AdamW(model.parameters(), lr=..., weight_decay=0.1)` 训练 GPT-124M。

**典型超参数**（Llama 3）：$\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-8}, \lambda = 0.1$

### 学习率调度

学习率调度对训练稳定性至关重要。

**Warmup 的必要性**：
- 训练初期，Adam 的二阶矩估计 $v_t$ 尚不准确（接近 0），导致更新步长过大
- Warmup 阶段线性增加学习率，给 $v_t$ 积累时间
- 典型 warmup 步数：2000 步

**Cosine Annealing（余弦退火）**：

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t - T_w}{T - T_w}\pi\right)\right)$$

其中 $T_w$ 是 warmup 步数，$T$ 是总训练步数。直觉：学习率从峰值平滑下降到接近 0，前期下降慢（探索），后期下降快（收敛）。

**WSD 调度器（Warmup-Stable-Decay）**：

MiniCPM 和部分新模型采用的三阶段策略：

```
Learning Rate
  │     ┌──────────────┐
  │    /│   Stable      │\
  │   / │               │ \
  │  /  │               │  \
  │ /   │               │   \
  │/    │               │    \
  └─────┴───────────────┴─────── Training Steps
  Warmup     Stable        Decay
  (~2000)   (most steps)   (~last 10%)
```

WSD 的优势：Stable 阶段可以随时决定何时开始 Decay，方便灵活调整训练长度。

```python
# 学习率调度实现
import math

def cosine_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0):
    """Cosine Annealing with Warmup"""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def wsd_schedule(step, total_steps, warmup_steps, decay_steps, max_lr, min_lr=0):
    """Warmup-Stable-Decay 调度器"""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    elif step < total_steps - decay_steps:
        return max_lr
    else:
        decay_progress = (step - (total_steps - decay_steps)) / decay_steps
        return min_lr + (max_lr - min_lr) * (1 - decay_progress)
```

---

## Mixed Precision Training

### FP32, FP16, BF16 的精度对比

| 格式 | 符号位 | 指数位 | 尾数位 | 数值范围 | 精度 |
|------|--------|--------|--------|---------|------|
| FP32 | 1 | 8 | 23 | ~3.4e38 | 高 |
| FP16 | 1 | 5 | 10 | ~6.5e4 | 中 |
| BF16 | 1 | 8 | 7 | ~3.4e38 | 低（但范围大） |

### Loss Scaling 技巧

FP16 的数值范围有限（最大 65504），训练中容易出现梯度下溢（很小的梯度变为 0）。Loss Scaling 的做法：

1. 将 loss 乘以一个大常数（如 1024）
2. 反向传播得到放大后的梯度
3. 更新参数前将梯度除以同一个常数

动态 Loss Scaling 会自动调整缩放因子：如果出现 NaN/Inf，减半缩放因子并跳过当前步。

### 为什么 BF16 比 FP16 更适合训练

- **BF16 的指数位与 FP32 相同（8位）**，数值范围一致，几乎不会溢出
- 不需要 Loss Scaling，训练流程更简单
- 代价是精度略低（7 位尾数 vs FP16 的 10 位），但实践证明对训练影响很小
- **Llama 3, Qwen 2.5 等主流模型全部使用 BF16 训练**

混合精度训练的典型配置：
- 模型参数和梯度：BF16
- 优化器状态（$m_t, v_t$）：FP32（需要高精度累加）
- 损失计算：FP32

---

## Gradient Checkpointing

### 内存 vs 计算的 trade-off

标准训练需要保存所有中间激活值用于反向传播。对于一个 $L$ 层的 Transformer：
- 不用 checkpointing：内存 $O(L)$，计算 $1\times$
- 全量 checkpointing：内存 $O(\sqrt{L})$，计算约 $1.33\times$（多做一次前向）

### 实现原理

```
标准训练:
  前向: 保存 layer1_out, layer2_out, ..., layerL_out  （内存占用大）
  反向: 用保存的激活值计算梯度

Gradient Checkpointing:
  前向: 只保存 layer1_out, layer_k_out, layer_2k_out...  （每隔 k 层保存）
  反向: 遇到未保存的激活值时，从最近的 checkpoint 重新前向计算
```

典型配置：对每个 Transformer block 做 checkpointing，即每层保存输入，层内的中间激活不保存。代价是约 33% 的额外计算时间，但可以显著减少显存占用。

---

## 训练监控

### 关键指标：loss, grad norm, learning rate

| 指标 | 正常范围 | 异常信号 |
|------|---------|---------|
| Training loss | 持续平滑下降 | 突然飙升（loss spike） |
| Gradient norm | 稳定在 0.1-10 | 突然变大（梯度爆炸）或趋近 0 |
| Learning rate | 按照 schedule 变化 | 应当与 loss 曲线配合 |
| Tokens/sec | 基本恒定 | 突然下降说明有硬件问题 |

```python
# 训练循环中的监控
self.model.train()
self.optimizer.zero_grad()

logits = self.model(input_tensor)
loss = self.criterion(
    logits.view(-1, logits.size(-1)),
    label_tensor.view(-1)
)
loss.backward()

# 梯度裁剪：防止梯度爆炸
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

self.optimizer.step()
self.loss_history.append(loss.item())

# 定期打印统计信息
avg_loss = np.mean(self.loss_history[-100:])
```

### Loss Spike 的处理

Loss spike（损失突然飙升）在大规模训练中很常见。处理策略：

1. **梯度裁剪**（gradient clipping）：限制梯度范数，通常 `max_norm=1.0`
2. **回滚到之前的 checkpoint**：如果 spike 持续不恢复
3. **降低学习率**：从 spike 前的 checkpoint 恢复，学习率减半
4. **数据检查**：某些低质量数据 batch 可能导致 spike
5. **跳过异常步**：如果梯度出现 NaN/Inf，跳过当前更新

### MFU (Model FLOPs Utilization) 的计算

MFU 衡量 GPU 的实际利用率：

$$\text{MFU} = \frac{\text{实际每秒 FLOPs}}{\text{GPU 理论峰值 FLOPs}}$$

对于 Transformer 模型，每个 token 的前向 + 反向 FLOPs 约为 $6N$（$N$ 为参数量）：

$$\text{实际 FLOPs/s} = \frac{6 \times N \times \text{batch\_size} \times \text{seq\_len}}{\text{每步耗时(秒)}}$$

典型 MFU：30-50%。达到 50% 以上说明系统优化较好。

---

## 预训练实战

### 完整训练配置示例

以训练一个 7B 参数模型为例：

```python
# 模型配置
model_config = {
    "hidden_size": 4096,
    "num_layers": 32,
    "num_heads": 32,
    "vocab_size": 32000,
    "max_seq_len": 4096,
    "intermediate_size": 11008,  # FFN 中间层
}

# 训练配置
train_config = {
    "total_tokens": 2_000_000_000_000,  # 2T tokens
    "batch_size": 4_000_000,             # ~4M tokens per batch (global)
    "micro_batch_size": 4,               # 每个 GPU 的 micro batch
    "seq_len": 4096,
    "learning_rate": 3e-4,
    "min_lr": 3e-5,
    "warmup_steps": 2000,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "optimizer": "AdamW",
    "betas": (0.9, 0.95),
    "precision": "bf16",
    "gradient_checkpointing": True,
}

```

> 数据加载实际工程上推荐用 HuggingFace `datasets` + 流式读取，配 `DataCollatorForLanguageModeling`，不必手写 Dataset。

### 最小可跑预训练：笔记本本地 < 1 分钟

::: tip 这一节解决什么问题
上面的 7B 配置只是"声明性纸面配方"。下面这段 ~60 行代码用 **GPT-2 的 50K 参数 toy 版本**（`n_embd=64, n_layer=2`）+ 字符级分词 + 5 句话语料，CPU 上 30-60 秒看到 loss 从 `ln(V)≈4.85` 降到 1.x、采样从乱码逐渐变成有词形结构的字符串。这正是预训练的**最小完整数据流**：tokenize → batch → next-token cross-entropy → backward → step → 生成对照。
:::

#### Step 1：准备语料 + 字符级分词

```python
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

CORPUS = """The quick brown fox jumps over the lazy dog.
A neural network learns by adjusting its parameters.
Gradient descent moves opposite to the gradient.
The cross entropy loss measures prediction accuracy.
Tokens are the basic units of language modeling.
""" * 8

vocab = sorted(set(CORPUS))
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}
V = len(vocab)
data = torch.tensor([stoi[c] for c in CORPUS], dtype=torch.long)
print(f"vocab = {V} chars,  corpus = {len(data)} tokens")
```

#### Step 2：从零初始化一个 toy GPT

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
config = GPT2Config(
    vocab_size=V, n_positions=64,
    n_embd=64, n_layer=2, n_head=4,
)
model = GPT2LMHeadModel(config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"model params  = {n_params/1e3:.1f}K")
print(f"random baseline loss ≈ ln(V) = {torch.log(torch.tensor(float(V))).item():.3f}")
```

随机初始化时模型对每个 token 等概率猜测，cross-entropy 期望值就是 $\ln V$——这是判断"模型有没有开始学到东西"的基准线。

#### Step 3：causal-LM 切片 + 单步 loss 验证

next-token 预测的精髓：让模型在长度 T 的窗口里看到前 T-1 个 token、预测第 2 到第 T 个 token。HF GPT2 内部会自动把 `labels` 左移一位对齐——所以 `input_ids` 和 `labels` 直接传同一个 tensor 即可。

```python
T = 32
def get_batch(B=16):
    ix = torch.randint(0, len(data) - T, (B,))
    return torch.stack([data[i:i+T] for i in ix]).to(device)

x = get_batch()
loss = model(input_ids=x, labels=x).loss     # HF 内部 shift 1 位
print(f"initial loss = {loss.item():.3f}")    # 应接近上面 ln(V) 基线
```

#### Step 4：训练循环

```python
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
model.train()

for step in range(300):
    x = get_batch(B=16)
    loss = model(input_ids=x, labels=x).loss
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 50 == 0:
        print(f"step {step:3d} | loss = {loss.item():.3f}")
```

CPU 上 300 步约 30-60s。loss 通常会从 `ln(V)≈4.85` 一路降到 1.0~1.5。

#### Step 5：随机初始化 vs 训练后采样对比

```python
@torch.no_grad()
def sample(m, prompt, n=80, temperature=0.8):
    m.eval()
    ids = torch.tensor([[stoi[c] for c in prompt]], device=device)
    for _ in range(n):
        logits = m(ids[:, -64:]).logits[0, -1] / temperature
        next_id = torch.multinomial(F.softmax(logits, -1), 1)
        ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
    return "".join(itos[int(i)] for i in ids[0])

fresh = GPT2LMHeadModel(config).to(device)         # 同 config 重新初始化
print("--- random init ---")
print(sample(fresh, "The "))
print("\n--- after 300 steps ---")
print(sample(model, "The "))
```

期望现象：**random** 输出是按字符均匀分布的乱码；**after 300 steps** 开始出现 `the`、`is`、`gradient`、空格-单词节奏等真实语言局部模式——loss 数字下降的同时，能直接"读"到模型在学什么。

::: warning 这只是教学样例
50K 参数、字符级分词、5 句话语料，远不到任何"通用语言建模"的门槛——它只是把 §Next Token Prediction 目标 一节里讲过的公式跑成可见的数据流。**真正的预训练**对应上面 §完整训练配置示例 的 7B / 2T tokens / 4M batch，按 §训练成本估算 大概 19 天 × 128×H100 ≈ \$177k。
:::

### 训练成本估算

**问题：训练一个 7B 模型，2T tokens，需要多少 GPU-hours？**

计算步骤：
1. 总 FLOPs = $6 \times N \times D = 6 \times 7 \times 10^9 \times 2 \times 10^{12} = 8.4 \times 10^{22}$
2. H100 BF16 理论峰值 = 989 TFLOPS
3. 假设 MFU = 40%，有效算力 = $989 \times 10^{12} \times 0.4 = 3.96 \times 10^{14}$ FLOPS
4. 总 GPU 秒数 = $8.4 \times 10^{22} / 3.96 \times 10^{14} \approx 2.12 \times 10^{8}$ 秒
5. **总 GPU-hours $\approx$ 59,000 H100-hours**

如果使用 128 张 H100：
- 训练时间 $\approx$ 59000 / 128 $\approx$ 461 小时 $\approx$ **19 天**
- 按 H100 云价格 $3/GPU-hour 计算：**约 $177,000**

---

## 苏格拉底时刻

1. **预训练的 loss 从 10+ 降到 2-3 的过程中，模型分别在学什么？** Loss 从 10 到 5 的阶段，模型主要在学习词频分布和简单的语法模式；从 5 到 3 的阶段，开始学习语义关联和世界知识；从 3 到 2.5 以下，开始涌现推理能力。

2. **Chinchilla 法则说每个参数需要 20 个 token，但 Llama 3 用了远超此比例的数据（8B 模型训练 15T token），为什么？** 因为 Chinchilla 法则优化的是训练计算量，但推理成本也很重要。小模型训练更久，推理时更省算力——这在部署阶段的收益远大于额外的训练成本。

3. **交叉熵损失的梯度为 $q - p$，这个简洁的形式意味着什么？** 意味着每次更新，模型在"向正确答案靠近"的同时"远离错误答案"，而且调整的幅度与当前预测的误差成正比。这是一个"自我校正"的过程。

4. **为什么预训练数据中包含大量低质量网页内容，模型仍能学到有用知识？** 因为有用的语言模式在高质量和低质量文本中都存在（语法、常见搭配）。但数据质量过低会导致模型学到错误知识和有毒内容。这就是为什么数据清洗如此重要。

5. **如果将所有预训练数据重复训练多个 epoch，效果会变差吗？** 会。实验表明数据重复 4 次以上，模型开始"记忆"而非"泛化"。这也是为什么高质量去重数据如此珍贵。

---

## 常见问题 & 面试考点

| 问题 | 要点 |
|------|------|
| 预训练的损失函数是什么？ | 交叉熵损失，等价于最大化下一个 token 的对数似然 |
| 解释 Chinchilla Scaling Law | 给定计算预算，模型参数和数据量应等比增长，每参数约 20 tokens |
| AdamW 和 Adam 的区别 | AdamW 将 weight decay 与自适应学习率解耦 |
| 为什么用 BF16 而不是 FP16？ | BF16 指数位与 FP32 相同，不易溢出，无需 loss scaling |
| Gradient checkpointing 的代价 | 约 33% 额外计算时间，换取显著内存节省 |
| 什么是 MFU？好的 MFU 是多少？ | 实际 FLOPs / 理论峰值 FLOPs，40-50% 是较好的水平 |
| Warmup 的作用 | 让 Adam 的二阶矩估计充分积累，避免初始阶段步长过大 |
| 预训练数据去重为什么重要？ | 重复数据导致过拟合、降低多样性、浪费计算资源 |

---

## 推荐资源

### 论文

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — OpenAI Scaling Laws 论文。
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla 论文，提出 token/参数 ≈ 20:1 的最优配比。
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — AdamW 原始论文。
- [MiniCPM 技术报告](https://arxiv.org/abs/2404.06395) — WSD（Warmup-Stable-Decay）学习率调度策略。
- [Llama 3 技术报告](https://arxiv.org/abs/2407.21783) — Meta 最新训练实践，含数据/训练/对齐全流程。

### 数据集

- [FineWeb 数据集](https://huggingface.co/datasets/HuggingFaceFW/fineweb) 与 [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — 开源高质量预训练数据，详细的清洗流程见 [training/datasets.md](datasets.md#推荐资源)。

### 代码参考

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy 的极简 GPT 训练代码，适合先跑通再读源码。
- [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) — GPT-2 (124M) 端到端复现仓库（MIT 许可，521 行单文件训练脚本），配套 [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) 视频。本页讲到的工程要点都能在它里面找到对应实现：
  - [fineweb.py](https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py) — 拉取 FineWeb-Edu `sample-10BT` 子集，用 `tiktoken` GPT-2 分词后切成 100M tokens / shard 的 `.npy`，是从原始数据到训练就绪格式的最小可用流水线。
  - [train_gpt2.py L207-L252](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L207-L252) — `DataLoaderLite`：跨 shard 顺序读取，按 DDP rank 切分，没有 shuffle —— 体现"web 数据本身已足够混洗，不必 in-memory shuffle"的工程取舍。
  - [train_gpt2.py L290-L347](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L290-L347) — DDP 启动 + autocast(bfloat16) + `torch.compile`：单卡/多卡兼容的标准模板，用 `torchrun --standalone --nproc_per_node=8 train_gpt2.py` 一行起 8 卡。
  - [train_gpt2.py L324-L331](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L324-L331) — **梯度累积**：`total_batch_size = 524288`（=2¹⁹ tokens ≈ 0.5M，与 GPT-3 论文一致），按 `B*T*world_size` 自动算 `grad_accum_steps`，把"想要的全局 batch"和"GPU 显存能装下的 micro-batch"解耦。
  - [train_gpt2.py L349-L370](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L349-L370) — **cosine LR with linear warmup**：`max_lr=6e-4, min_lr=6e-5, warmup_steps=715, max_steps=19073`（10B tokens / 0.5M batch ≈ 1 epoch），是 GPT 系列的经典调度。
  - [train_gpt2.py L155-L205](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L155-L205) — `configure_optimizers`：把"参数维度 ≥ 2"的（embedding/Linear weight）放进 weight decay 组，bias 与 LayerNorm 不衰减；自动检测 PyTorch 的 fused AdamW 加速。
  - [train_gpt2.py L426-L475](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L426-L475) — 训练主循环：`grad_accum_steps` 内累加 loss/grad、`clip_grad_norm_(1.0)`、记录 tokens/sec 与 step time，是判断训练是否健康的最小可用监控。
  - [train_gpt2.py L376-L424](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L376-L424) + [hellaswag.py](https://github.com/karpathy/build-nanogpt/blob/master/hellaswag.py) — 训练中每 250 步交替跑 val loss + HellaSwag eval，正好对应 [FineWeb 论文](datasets.md#推荐资源)里强调的"小模型 early-signal benchmark"评测范式。
- [rasbt/LLMs-from-scratch · ch05](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05) — Sebastian Raschka《Build a Large Language Model (From Scratch)》第 5 章配套代码，主线是"在小数据集上从零预训练，再加载 OpenAI GPT-2 权重做对照"，把训练循环 / 采样 / 权重装载这三件事拆得极细。
  - [ch05/01_main-chapter-code/gpt_train.py](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/gpt_train.py)（242 行） — `calc_loss_batch`(L28-L32) / `calc_loss_loader`(L35-L49) / `evaluate_model`(L52-L58) / `train_model_simple`(L75-L109) 四个函数构成最小训练循环，无 DDP / 无混合精度，是理解"loss 怎么算、eval 怎么穿插"的最干净版本。补足了本页 §训练目标 一节里被一笔带过的工程细节：交叉熵在 `(B*T, vocab)` 上 flatten 之后再算、`eval_freq` 步穿插一次 `model.eval()` + `torch.no_grad()` 验证、`tokens_seen` 累加用于 x 轴双轴绘图。最关键的一段是 loss 计算（L28-L32，原文）：
    ```python
    def calc_loss_batch(input_batch, target_batch, model, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss
    ```
    `train_model_simple` 主循环（L86-L102，原文）则把"前向→反向→优化器→可选评估"四步串起来，是对照 nanoGPT 那份带 grad accumulation / autocast / DDP 的工业版之前最值得先读的玩具版本。
  - [ch05/01_main-chapter-code/gpt_generate.py](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/gpt_generate.py)（299 行） — 把 `generate` 拆成 temperature scaling、top-k 截断、EOS 提前停三步，配合 [gpt_download.py](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/gpt_download.py) 把 OpenAI TF checkpoint 转成 PyTorch state_dict，对照本页 §权重加载与采样最直观。
  - [ch05/04_learning_rate_schedulers](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/04_learning_rate_schedulers) — warmup + cosine 调度的最小实现，与 nanoGPT 的写法略不同但更易读。
  - [ch05/03_bonus_pretraining_on_gutenberg](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/03_bonus_pretraining_on_gutenberg) — 在 Project Gutenberg 数据集上做端到端预训练的小型工程示例。
  - [ch05/07_gpt_to_llama](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/07_gpt_to_llama) / [11_qwen3](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3) / [12_gemma3](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/12_gemma3) — 把同一套预训练框架替换成 Llama 3.2 / Qwen3 / Gemma 3 的 from-scratch 复现，可作为本页讲到的"模型架构选型"的代码版对照。
