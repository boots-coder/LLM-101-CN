---
title: "Flow Matching"
description: "从 Diffusion 到 Flow Matching，生成式建模的新范式"
topics: [flow-matching, diffusion, ODE, generative-model, rectified-flow, conditional-flow-matching]
prereqs: [fundamentals/math]
---
# Flow Matching

::: info 一句话总结
Flow Matching 是一种比 Diffusion 更简洁的生成式建模方法——不再需要定义复杂的噪声 schedule 和去噪过程，而是**直接学习一个向量场（velocity field）**，让噪声沿着 ODE 轨迹流向目标分布。训练目标简单到一行公式：$\mathcal{L} = \|v_\theta(x_t, t) - (x_1 - x_0)\|^2$。
:::

## 在大模型体系中的位置

```
┌────────────────────────────────────────────────────────────┐
│              Generative Models for AI                       │
│                                                             │
│  VAE ──→ GAN ──→ Diffusion ──→ [Flow Matching] ──→ ???     │
│                      ↑               ↑                      │
│               加噪+去噪         直接学 ODE 向量场            │
│              (复杂 schedule)    (线性插值，更简洁)            │
│                                                             │
│  应用: 图像生成 (Stable Diffusion 3)                         │
│        语音合成 (Voicebox)                                   │
│        蛋白质结构 (FrameFlow)                                │
│        LLM 文本生成 (MDLM, Flow Matching for Discrete Data) │
└────────────────────────────────────────────────────────────┘
```

Flow Matching 正在成为 Diffusion 的下一代替代方案。Stable Diffusion 3、Meta 的 Voicebox 等都已采用 Flow Matching。更重要的是，它正在被探索用于**离散 token 的生成**，可能为 LLM 带来全新的生成范式。

---

## 1. 从 Diffusion 到 Flow Matching

### 1.1 Diffusion 回顾

Diffusion Model 的核心思路：

```
前向过程（加噪）:  x_0 → x_1 → x_2 → ... → x_T ≈ N(0, I)
                   数据      逐步加噪           纯噪声

反向过程（去噪）:  x_T → x_{T-1} → ... → x_0
                   噪声     逐步去噪         生成数据
```

**前向过程**（固定，不可学习）：

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) \mathbf{I})
$$

**反向过程**（需要学习）：

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 \mathbf{I})
$$

**训练目标**：预测噪声 $\epsilon$

$$
\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{x_0, \epsilon, t} \left[ \|\epsilon_\theta(x_t, t) - \epsilon\|^2 \right]
$$

::: warning Diffusion 的不足
1. 噪声 schedule（$\beta_t$ 或 $\bar{\alpha}_t$）需要精心设计，不同 schedule 影响巨大
2. 前向/反向过程依赖 SDE（随机微分方程），数学推导复杂
3. 采样需要数百到上千步，速度慢
4. 理论框架和实际训练目标之间有 gap
:::

### 1.2 Flow Matching 的核心思想

Flow Matching 的思路极其简洁：

> **直接学习一个向量场 $v_\theta(x, t)$，使得数据点沿着这个向量场从噪声分布"流"到目标分布。**

```
Flow Matching:

t=0                                              t=1
噪声 x_0 ─────────── 沿向量场流动 ──────────────→ 数据 x_1
  ∼ N(0,I)         dx/dt = v_θ(x_t, t)           ∼ p_data
```

用 ODE（常微分方程）描述：

$$
\frac{dx_t}{dt} = v_\theta(x_t, t), \quad t \in [0, 1]
$$

给定 $x_0 \sim \mathcal{N}(0, I)$，沿着 $v_\theta$ 积分到 $t=1$，就得到生成的样本。

### 1.3 关键区别对比

| 特性 | Diffusion | Flow Matching |
|------|-----------|---------------|
| 核心方程 | SDE（随机微分方程） | ODE（常微分方程） |
| 学习目标 | 预测噪声 $\epsilon$ 或 score $\nabla \log p$ | 预测速度场 $v(x_t, t)$ |
| 路径 | 由噪声 schedule 决定 | 可以自由设计（如线性） |
| 训练目标 | $\|\epsilon_\theta - \epsilon\|^2$ | $\|v_\theta - (x_1 - x_0)\|^2$ |
| 采样 | 反向 SDE 求解（需要很多步） | ODE 求解（可以更少步） |
| 噪声 schedule | 需要精心设计 | 不需要 |

---

## 2. 数学基础

### 2.1 常微分方程（ODE）

给定向量场 $v: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$，ODE 定义了一条从 $x_0$ 到 $x_1$ 的路径：

$$
\frac{dx_t}{dt} = v(x_t, t), \quad x_0 \sim p_0
$$

这条路径定义了一个**流（flow）** $\phi_t$：$\phi_t(x_0) = x_t$。

流的关键性质：$\phi_t$ 是一个可逆映射（diffeomorphism），意味着：
- 每个 $x_0$ 有且只有一条轨迹
- 概率密度沿轨迹变化可以用**连续性方程**精确描述

### 2.2 概率路径

我们希望构造一个时间连续的概率分布族 $p_t$：

$$
p_0 = \mathcal{N}(0, I) \quad \text{（噪声分布）}
$$
$$
p_1 = p_{\text{data}} \quad \text{（目标数据分布）}
$$

中间的 $p_t$（$0 < t < 1$）描述了从噪声到数据的**过渡分布**。

**连续性方程**将概率路径 $p_t$ 和向量场 $v_t$ 联系起来：

$$
\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t v_t) = 0
$$

如果我们能找到一个 $v_t$ 使得连续性方程成立，那么沿着这个向量场采样就能从 $p_0$ 生成 $p_1$。

### 2.3 速度场（Velocity Field）

速度场 $v_t(x)$ 告诉我们在时刻 $t$、位置 $x$ 处，数据点应该往哪个方向、以多快的速度移动。

对于最简单的**线性插值路径**：

$$
x_t = (1 - t) x_0 + t x_1
$$

对应的速度场为：

$$
v_t = \frac{dx_t}{dt} = x_1 - x_0
$$

这就是为什么 Flow Matching 的训练目标如此简洁——目标速度就是 $x_1 - x_0$（终点减起点）。

---

## 3. Conditional Flow Matching (CFM)

### 3.1 为什么需要 "Conditional"？

直接优化 Flow Matching 目标需要知道边际向量场 $u_t(x)$，这通常是 intractable 的。

**核心技巧**：条件化（conditioning）。我们不直接学边际向量场，而是学**条件向量场** $u_t(x | x_1)$——给定目标样本 $x_1$，从 $x_0$ 到 $x_1$ 的向量场。

### 3.2 CFM 目标函数

**Conditional Flow Matching (CFM) 损失**：

$$
\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t \sim \mathcal{U}[0,1], \, x_1 \sim p_{\text{data}}, \, x_0 \sim \mathcal{N}(0,I)} \left[ \| v_\theta(x_t, t) - u_t(x_t | x_1) \|^2 \right]
$$

对于线性条件路径 $x_t = (1-t)x_0 + tx_1$：

$$
u_t(x_t | x_1) = x_1 - x_0
$$

因此 CFM 损失简化为：

$$
\boxed{\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta((1-t)x_0 + tx_1, \; t) - (x_1 - x_0) \|^2 \right]}
$$

::: tip 为什么比 Diffusion 训练更简单？
1. **无需噪声 schedule**：$x_t$ 就是线性插值，没有 $\alpha_t, \beta_t, \bar{\alpha}_t$ 这些超参数
2. **目标直观**：学习速度 $x_1 - x_0$（从噪声到数据的位移），比预测噪声 $\epsilon$ 更直观
3. **数学等价**：Lipman et al. (2023) 证明优化 CFM 损失等价于优化真正的 Flow Matching 目标
:::

### 3.3 训练算法

```
算法: Conditional Flow Matching 训练

Repeat:
    1. 从数据集采样 x_1 ~ p_data
    2. 从噪声分布采样 x_0 ~ N(0, I)
    3. 均匀采样时间 t ~ U[0, 1]
    4. 计算插值点 x_t = (1-t) * x_0 + t * x_1
    5. 计算目标速度 v_target = x_1 - x_0
    6. 预测速度 v_pred = v_θ(x_t, t)
    7. 计算损失 L = ||v_pred - v_target||²
    8. 反向传播更新 θ
```

### 3.4 采样算法

```
算法: Flow Matching 采样（欧拉法）

1. 采样 x_0 ~ N(0, I)
2. 设置步长 Δt = 1/N
3. For t = 0, Δt, 2Δt, ..., 1-Δt:
       v = v_θ(x_t, t)          # 预测速度
       x_{t+Δt} = x_t + v * Δt  # 欧拉更新
4. 返回 x_1 作为生成样本
```

---

## 4. Rectified Flow

### 4.1 直线路径的优势

Rectified Flow（Liu et al., 2023）进一步强调了**直线路径**的优越性。

核心观察：如果向量场 $v_\theta$ 学到的足够好，使得每条轨迹都接近直线，那么：
- ODE 求解只需要**很少的步数**（极端情况下一步就够）
- 采样速度大幅提升

```
一般 Flow Matching:          Rectified Flow:

x_0 ~~~曲线~~~→ x_1          x_0 ──直线──→ x_1
    需要很多步                     可能只需 1 步
```

### 4.2 Reflow 操作

Rectified Flow 的关键技巧——**Reflow**：

1. 训练一个初始 flow model $v_\theta$
2. 用 $v_\theta$ 从噪声 $x_0$ 生成样本 $\hat{x}_1 = \text{ODE}(x_0, v_\theta)$
3. 用 $(x_0, \hat{x}_1)$ 配对重新训练一个新的 flow model

每次 Reflow 都会让轨迹更接近直线。经过 2-3 次 Reflow，即使用 1-2 步欧拉法也能得到高质量样本。

### 4.3 一步生成的可能性

$$
\text{一步生成}: \quad x_1 = x_0 + v_\theta(x_0, 0) \cdot 1 = x_0 + v_\theta(x_0, 0)
$$

如果轨迹是完美直线，一步欧拉法就是精确解。这使得 Flow Matching + Rectified Flow 成为**最快的 diffusion 类生成方法之一**。

::: tip Stable Diffusion 3 的选择
Stability AI 在 Stable Diffusion 3 中采用了 Rectified Flow 作为采样方法，相比 SD 1/2 的 DDPM/DDIM 采样，在相同步数下获得了更好的图像质量。
:::

---

## 5. 代码实现

### 5.1 基础 Flow Matching（2D 演示）

用一个简单的 2D 例子理解 Flow Matching 的全过程——让噪声点"流向"正弦曲线：

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ===== 超参数 =====
dim = 2             # 2D 数据
num_samples = 200
lr = 1e-2
epochs = 100000
num_steps = 50      # ODE 求解步数

# ===== 1. 准备数据 =====
# 目标分布：正弦曲线上的点
x1_samples = torch.rand(num_samples, 1) * 4 * torch.pi  # [0, 4π]
y1_samples = torch.sin(x1_samples)
target_data = torch.cat([x1_samples, y1_samples], dim=1)  # (200, 2)

# 噪声分布：标准高斯
noise_data = torch.randn(num_samples, dim) * 2  # (200, 2)

# ===== 2. 定义向量场网络 =====
class VectorField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64),  # 输入: x(2) + t(1) = 3
            nn.ReLU(),
            nn.Linear(64, dim)       # 输出: 速度向量(2)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

# ===== 3. 训练 =====
model = VectorField()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    idx = torch.randperm(num_samples)
    x0 = noise_data[idx]   # 起点：噪声
    x1 = target_data[idx]  # 终点：正弦曲线

    t = torch.rand(x0.size(0), 1)     # 随机时间 t ~ U[0,1]
    xt = (1 - t) * x0 + t * x1        # 线性插值
    vt_pred = model(xt, t)             # 预测速度
    vt_target = x1 - x0               # 目标速度

    loss = torch.mean((vt_pred - vt_target) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ===== 4. 采样（ODE 欧拉法） =====
x = noise_data[0:1]            # 一个噪声起点
trajectory = [x.detach().numpy()]

t = 0
delta_t = 1 / num_steps
with torch.no_grad():
    for i in range(num_steps):
        vt = model(x, torch.tensor([[t]]))
        x = x + vt * delta_t           # 欧拉更新
        t += delta_t
        trajectory.append(x.detach().numpy())

# trajectory 记录了从噪声到正弦曲线的完整轨迹
```

**核心只有 4 行**：
1. `xt = (1 - t) * x0 + t * x1` — 线性插值
2. `vt_pred = model(xt, t)` — 预测速度
3. `vt_target = x1 - x0` — 目标速度
4. `loss = ||vt_pred - vt_target||^2` — MSE 损失

### 5.2 Conditional Flow Matching（带条件）

加入条件信息 `tag`，可以控制生成特定区域的样本：

```python
class ConditionalVectorField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 2, 64),  # 输入: x(2) + t(1) + tag(1) = 4
            nn.ReLU(),
            nn.Linear(64, dim)
        )

    def forward(self, x, t, tag):
        return self.net(torch.cat([x, t, tag], dim=1))

# 训练时加入条件
model = ConditionalVectorField()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 为每个样本分配条件标签（如按 x 坐标分区）
tags = (x1_samples / (4 * torch.pi / 10.0)).int()

for epoch in range(5000):
    idx = torch.randperm(num_samples)
    x0 = noise_data[idx]
    x1 = target_data[idx]
    t = torch.rand(x0.size(0), 1)

    xt = (1 - t) * x0 + t * x1
    vt_pred = model(xt, t, tags[idx].float())  # 传入条件
    vt_target = x1 - x0

    loss = torch.mean((vt_pred - vt_target) ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 采样时指定条件
tag_num = 3  # 生成 tag=3 区域的样本
x = torch.randn(1, dim)
t = 0
delta_t = 1 / 50
with torch.no_grad():
    for i in range(50):
        vt = model(x, torch.tensor([[t]]),
                   torch.tensor([[tag_num]], dtype=torch.float))
        x = x + vt * delta_t
        t += delta_t
# x 现在应该落在 tag=3 对应的正弦曲线区域
```

::: tip Conditional Flow Matching 的意义
条件信息可以是任何东西：类别标签、文本 embedding、图像特征等。这就是为什么 CFM 能用于 text-to-image（文本条件）、text-to-speech（文本条件）等任务。
:::

---

## 6. 在 LLM 中的应用

### 6.1 从连续到离散：离散 Flow Matching

传统 Flow Matching 处理连续数据（图像像素、音频波形）。但 LLM 的输入是**离散 token**。如何把 Flow Matching 用于离散数据？

几种主要方向：

**方向一：嵌入空间 Flow Matching**

```
离散 token → Embedding 空间（连续）→ Flow Matching → 连续向量 → 最近邻解码 → token
```

代表工作：**MDLM (Masked Diffusion Language Model)**

**方向二：离散概率 Flow Matching**

在 simplex（概率单纯形）上定义 flow：

$$
x_t \in \Delta^V = \{p \in \mathbb{R}^V : p_i \geq 0, \sum p_i = 1\}
$$

$x_0$ 是均匀分布（最大不确定性），$x_1$ 是 one-hot（确定的 token）。

**方向三：CTMC（连续时间马尔可夫链）**

将离散扩散重新表述为连续时间框架下的转移速率矩阵，与 flow matching 的 ODE 形式对偶。

### 6.2 与自回归 LLM 的对比

| 特性 | 自回归 LLM | Flow Matching LLM |
|------|-----------|-------------------|
| 生成方式 | 逐 token 生成 | 所有 token 并行精炼 |
| 生成速度 | O(n) 步 | O(T) 步（T 为 flow 步数，与序列长度无关） |
| 左到右依赖 | 强制从左到右 | 无方向限制，可并行 |
| 编辑能力 | 只能 append | 可以修改任意位置 |
| 训练目标 | next-token prediction | flow matching loss |

### 6.3 未来展望

Flow Matching 在 LLM 领域仍处于早期探索阶段，但有几个值得关注的方向：

1. **非自回归文本生成**：打破"从左到右"的限制，一次性生成整个句子
2. **文本编辑与补全**：利用 flow 的可逆性，对现有文本进行局部修改
3. **多模态统一**：图像和文本在同一个 flow 框架下生成
4. **加速采样**：Rectified Flow + 蒸馏，用 1-4 步生成完整文本

---

## 苏格拉底时刻

在继续之前，尝试回答这些问题：

1. **核心区别**：Flow Matching 和 Diffusion 的最本质区别是什么？（提示：SDE vs ODE，随机 vs 确定性）

2. **为什么线性插值 work**：$x_t = (1-t)x_0 + tx_1$ 这么简单的路径就够了吗？凭什么不需要 Diffusion 那种精心设计的噪声 schedule？

3. **CFM 的巧妙之处**：为什么"条件化"能让 intractable 的边际向量场变成 tractable 的训练目标？（提示：类比 VAE 的 ELBO）

4. **一步生成**：Rectified Flow 号称可以"一步生成"，但实际一步生成的质量如何？瓶颈在哪里？

5. **离散化挑战**：将 Flow Matching 应用于离散 token 的核心困难是什么？（提示：梯度、连续空间与离散空间的映射）

::: details 参考思路
1. Diffusion 的反向过程是 SDE（有随机噪声项），Flow Matching 用的是 ODE（纯确定性）。ODE 的轨迹确定且不交叉，使理论分析和加速采样更容易。
2. 线性插值定义了最简单的"条件概率路径"。CFM 的理论保证了：只要条件路径合理，优化条件损失就等价于优化真正的 Flow Matching 目标。噪声 schedule 是 Diffusion 为了保证数学性质而"设计"出来的，Flow Matching 从构造上就绕过了这个需求。
3. 边际向量场是对所有可能的 $(x_0, x_1)$ 配对求期望，无法直接计算。但 CFM 证明了条件损失的梯度与边际损失的梯度相等（类似 score matching 的 Stein's identity 技巧），所以优化条件版本就等于优化边际版本。
4. 一步生成需要轨迹是完美直线。实际中轨迹总有弯曲，Reflow 可以减少弯曲但不能完全消除。实际一步生成的质量低于多步，通常 4-8 步是质量和速度的 sweet spot。
5. 离散空间没有连续梯度，不能直接定义 ODE。需要将离散 token 映射到连续空间（embedding/simplex），在连续空间做 flow，再映射回离散空间。这个"连续-离散"转换会引入误差。
:::

---

## 推荐资源

| 资源 | 链接 | 说明 |
|------|------|------|
| Flow Matching 原论文 (Lipman et al.) | [arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747) | CFM 的理论基础 |
| Rectified Flow 原论文 (Liu et al.) | [arxiv.org/abs/2209.03003](https://arxiv.org/abs/2209.03003) | 直线路径 + Reflow 技巧 |
| Flow Matching Guide | [arxiv.org/abs/2412.06264](https://arxiv.org/abs/2412.06264) | Meta 的综合教程 |
| Stable Diffusion 3 技术报告 | [arxiv.org/abs/2403.03206](https://arxiv.org/abs/2403.03206) | Rectified Flow 在工业界的应用 |
| MDLM 论文 | [arxiv.org/abs/2406.07524](https://arxiv.org/abs/2406.07524) | 离散 Flow Matching 用于语言模型 |
| Voicebox (Meta) | [arxiv.org/abs/2306.15687](https://arxiv.org/abs/2306.15687) | CFM 用于语音合成 |
| 知乎 Flow Matching 入门 | [zhuanlan.zhihu.com/p/28731517852](https://zhuanlan.zhihu.com/p/28731517852) | 中文入门教程 |
