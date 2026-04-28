---
title: "神经网络基础"
description: "从感知机到 MLP，激活函数、损失函数、反向传播与优化器"
topics: [MLP, activation, ReLU, GELU, SwiGLU, cross-entropy, backpropagation, Adam, AdamW, Muon, LayerNorm, dropout]
prereqs: [fundamentals/math]
---
# 神经网络

> 理解前馈网络和反向传播，是掌握 Transformer 的必经之路。Transformer 中的 FFN 层就是两层 MLP，注意力层的输出也要经过线性投影——神经网络的基本功贯穿始终。

## 在大模型体系中的位置

```
神经网络 ◄── 你在这里
  ├── MLP          → Transformer FFN 层的核心组件
  ├── 激活函数      → GELU (GPT), SwiGLU (Llama) 的选择依据
  ├── 反向传播      → 模型训练的核心算法
  ├── 损失函数      → 交叉熵驱动 next-token prediction
  ├── 优化器        → AdamW 是大模型训练的标配
  └── 正则化        → LayerNorm, Dropout 的取舍
```

---

## 感知机到多层网络

### 单个神经元

一个神经元执行两步操作：**线性变换** + **激活函数**：

$$
y = \sigma(w^T x + b)
$$

其中 $w$ 是权重向量，$b$ 是偏置，$\sigma$ 是非线性激活函数。

没有激活函数的话，多层网络等价于单层（线性函数的组合仍是线性函数）。**激活函数引入非线性，是网络能拟合复杂函数的关键。**

### 多层感知机 (MLP)

多层感知机由多个全连接层堆叠而成。下面用一个两层线性网络演示前向与反向传播的完整流程：

```python
import torch

batch_size = 4       # 样本数
in_features = 8      # 输入维度
hidden_features = 16 # 隐藏层维度
out_features = 3     # 输出维度

x = torch.randn(batch_size, in_features)
W1 = torch.randn(in_features, hidden_features, requires_grad=True)
W2 = torch.randn(hidden_features, out_features)
target = torch.randn(batch_size, out_features)

# 前向传播
h = x @ W1            # 隐藏层: [batch, in] @ [in, hidden] -> [batch, hidden]
h.retain_grad()
logits = h @ W2        # 输出层: [batch, hidden] @ [hidden, out] -> [batch, out]
logits.retain_grad()
loss = ((target - logits) ** 2).mean()

# 反向传播
loss.backward()
```

**手动验证梯度计算**——理解反向传播的核心逻辑：

```python
# 输出层梯度：手动推导 vs PyTorch autograd
N = batch_size * out_features
grad_out = 2.0 * (logits - target) / N     # d(loss)/d(logits)
W2_grad_manual = h.t() @ grad_out           # d(loss)/d(W2) = h^T @ grad_out
h_grad_manual = grad_out @ W2.t()           # d(loss)/d(h)  = grad_out @ W2^T

print(torch.allclose(W2_grad_manual, W2.grad))  # True
print(torch.allclose(h_grad_manual, h.grad))    # True

# 隐藏层梯度
W1_grad_manual = x.t() @ h_grad_manual          # d(loss)/d(W1) = x^T @ d(loss)/d(h)
print(torch.allclose(W1_grad_manual, W1.grad))  # True
```

**关键结论：梯度计算是前向计算的两倍计算量。** 单层网络中两者计算量相同（不需要对输入 $x$ 求导）；多层网络中，每层需要计算对权重和对输入的两个梯度，所以是两倍。

### 万能逼近定理

**Universal Approximation Theorem：** 一个具有足够宽度的单隐藏层前馈网络，可以以任意精度逼近任何连续函数。

但"足够宽"可能意味着指数级的神经元数量。实践中我们用**更深的网络**（更多层、更少宽度）来高效逼近复杂函数。

---

## 激活函数

### ReLU 及其变体

$$
\text{ReLU}(x) = \max(0, x)
$$

优点：计算简单、缓解梯度消失。缺点：负值区域梯度为 0（"死神经元"问题）。

**Leaky ReLU** 给负值区域一个小斜率：$f(x) = \max(\alpha x, x)$，$\alpha = 0.01$。

### Sigmoid 与 Tanh

$$
\sigma(x) = \frac{1}{1 + e^{-x}}, \quad \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

两者的输出都是有界的（sigmoid: $(0,1)$，tanh: $(-1,1)$），在深层网络中容易梯度消失。现代网络中较少作为隐藏层激活函数，但 sigmoid 常用于门控机制（如 LSTM 的门、SwiGLU 的门）。

### GELU (GPT 使用)

$$
\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)
$$

GELU 是 ReLU 的平滑版本，可以看作对输入的"软门控"——根据输入值的大小，以一定概率保留或丢弃。GPT 系列模型使用 GELU。

### SwiGLU (Llama 使用)

$$
\text{SwiGLU}(x, W_1, W_2, W_3) = (\text{SiLU}(xW_1) \odot xW_2) W_3
$$

其中 $\text{SiLU}(x) = x \cdot \sigma(x)$（也叫 Swish），$\odot$ 是逐元素乘法。

SwiGLU 引入了**门控机制**（Gate），用一个分支控制另一个分支的信息流。Llama、Mistral 等模型的 FFN 层使用 SwiGLU，需要三个权重矩阵（$W_1, W_2, W_3$），因此隐藏层维度通常设为 $\frac{8d}{3}$ 的整数倍以保持参数量。

---

## 损失函数

### MSE (均方误差)

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

适用于**回归任务**。在分类任务中，MSE 的梯度在预测接近 0 或 1 时非常小（梯度消失），训练效率低。

### 交叉熵损失

交叉熵是分类任务和语言模型的标准损失函数：

$$
\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{C} p_i \log q_i
$$

在分类问题中，真实分布 $p$ 是 one-hot 向量，简化为 $\mathcal{L} = -\log q_{\text{label}}$。

完整的实现流程：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

vocab = 50        # 特征维度
num_classes = 5   # 分类数
batch = 8         # batch size

x = torch.randn(batch, vocab)
label = torch.randint(high=num_classes, size=(batch,))
proj = nn.Linear(vocab, num_classes)
logits = proj(x)

# 方式 1：手动实现——从 softmax 概率出发
def manual_cross_entropy(labels, logits):
    n = logits.size(0)
    probs = F.softmax(logits, dim=-1)
    # 只取正确类别的概率，再求负对数均值
    correct_probs = probs[torch.arange(n), labels]
    return -correct_probs.log().mean()

# 方式 2：用 log_softmax 避免数值溢出（PyTorch 内部做法）
def ce_from_logits(labels, logits):
    n = logits.size(0)
    log_probs = F.log_softmax(logits, dim=-1)
    return -log_probs[torch.arange(n), labels].mean()

# 方式 3：PyTorch 官方接口
loss_fn = nn.CrossEntropyLoss()

print(manual_cross_entropy(label, logits))
print(ce_from_logits(label, logits))
print(loss_fn(logits, label))      # 三者结果一致
```

### 交叉熵的梯度推导

下面给出完整推导。**结论先行：**

$$
\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_i} = q_i - p_i
$$

即 **Softmax + 交叉熵** 对 logits 的梯度 = 预测概率 - 真实概率。极其优雅简洁。

**推导过程：**

1. **交叉熵对 softmax 输出 $q$ 的梯度：**

$$
\frac{\partial \mathcal{L}}{\partial q_i} = -\frac{p_i}{q_i}
$$

2. **Softmax 对 logits $z$ 的雅可比矩阵：**

$$
\frac{\partial q_i}{\partial z_j} = \begin{cases} q_i(1 - q_i) & \text{if } i = j \\ -q_i q_j & \text{if } i \neq j \end{cases}
$$

矩阵形式：$\frac{\partial q}{\partial z} = \text{diag}(q) - q^T q$

3. **链式法则组合（以第 $k$ 个 logit 为例）：**

$$
\frac{\partial \mathcal{L}}{\partial z_k} = \sum_i \frac{\partial \mathcal{L}}{\partial q_i} \cdot \frac{\partial q_i}{\partial z_k} = \sum_i \left(-\frac{p_i}{q_i}\right) \cdot (-q_i q_k) + \left(-\frac{p_k}{q_k}\right) \cdot q_k(1-q_k)
$$

$$
= q_k \sum_i p_i - p_k = q_k \cdot 1 - p_k = q_k - p_k
$$

代码验证：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

C = 7  # 类别数
z = torch.randn(1, C, requires_grad=True)
y = torch.randint(high=C, size=(1,))

# 1. PyTorch autograd
criterion = nn.CrossEntropyLoss()
loss = criterion(z, y)
loss.backward()
print("autograd 梯度:", z.grad)

# 2. 用 q - p 直接验证
one_hot = torch.zeros(1, C)
one_hot[0, y] = 1.0
q = F.softmax(z, dim=1)
grad_check = q - one_hot
print("q - p 梯度:   ", grad_check.detach())
# 两者完全一致
```

**实践意义：** 这就是为什么 `nn.CrossEntropyLoss` 接收 logits 而非 softmax 后的概率——直接用 $q - p$ 计算梯度，避免分别对 CE 和 softmax 求导带来的额外计算量。

---

## 反向传播

### 计算图

神经网络的每一步运算构成一个**有向无环图**（DAG）。前向传播沿着图从输入到输出计算结果；反向传播沿着图从输出到输入计算梯度。

### 链式法则在计算图上的应用

下面手动实现一个微型自动微分引擎，思路参考 Karpathy 的 micrograd，但变量命名和实现细节做了重新组织：

```python
import math

class Scalar:
    """标量计算图节点，支持自动反向传播"""
    def __init__(self, data, _inputs=(), _operator='', label=''):
        self.data = data
        self.grad = 0.0
        self._grad_fn = lambda: None
        self._inputs = set(_inputs)
        self._operator = _operator
        self.label = label

    def __add__(self, other):
        result = Scalar(self.data + other.data, (self, other), '+')
        def _grad_fn():
            self.grad += 1.0 * result.grad    # d(a+b)/da = 1
            other.grad += 1.0 * result.grad   # d(a+b)/db = 1
        result._grad_fn = _grad_fn
        return result

    def __mul__(self, other):
        result = Scalar(self.data * other.data, (self, other), '*')
        def _grad_fn():
            self.grad += other.data * result.grad   # d(a*b)/da = b
            other.grad += self.data * result.grad   # d(a*b)/db = a
        result._grad_fn = _grad_fn
        return result

    def tanh(self):
        t = math.tanh(self.data)
        result = Scalar(t, (self,), 'tanh')
        def _grad_fn():
            self.grad += (1.0 - t ** 2) * result.grad   # d(tanh)/dx = 1 - tanh^2
        result._grad_fn = _grad_fn
        return result

    def backward(self):
        # 拓扑排序，确保上游节点先计算梯度
        order = []
        seen = set()
        def _topological_sort(node):
            if node not in seen:
                seen.add(node)
                for inp in node._inputs:
                    _topological_sort(inp)
                order.append(node)
        _topological_sort(self)
        self.grad = 1.0
        for node in reversed(order):
            node._grad_fn()
```

使用这个引擎模拟一个简单的神经元：

```python
# 构建计算图: out = tanh(x1*w1 + x2*w2 + b)
x1 = Scalar(3.0, label='x1')
x2 = Scalar(-1.0, label='x2')
w1 = Scalar(0.5, label='w1')
w2 = Scalar(-1.5, label='w2')
b = Scalar(2.0, label='b')

h1 = x1 * w1           # 1.5
h2 = x2 * w2           # 1.5
pre_act = h1 + h2 + b  # 5.0
out = pre_act.tanh()    # ≈ 0.9999

# 反向传播
out.backward()
# 现在每个节点的 .grad 都已计算完毕
```

### PyTorch 的自动微分

PyTorch 的 `autograd` 就是上述思想的工业级实现。设置 `requires_grad=True` 的张量会自动构建计算图，调用 `.backward()` 自动计算梯度。

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1   # y = x^2 + 3x + 1
y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 7.0
```

---

## 优化器

### SGD 与动量

**随机梯度下降（SGD）：**
$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}
$$

**带动量的 SGD：** 引入"惯性"，平滑梯度更新方向：
$$
v_t = \gamma v_{t-1} + \eta \nabla_\theta \mathcal{L}, \quad \theta_{t+1} = \theta_t - v_t
$$

### Adam 的数学推导

Adam 同时使用**一阶矩**（梯度的均值）和**二阶矩**（梯度的方差）来自适应调整每个参数的学习率：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(一阶矩估计)}
$$
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(二阶矩估计)}
$$

**偏差校正**（训练初期 $m_t, v_t$ 偏向零）：
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**参数更新：**
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

常用超参数：$\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$。

**直觉：** 一阶矩提供动量（平滑方向），二阶矩提供自适应学习率（梯度大的参数步长小，梯度小的参数步长大）。

### AdamW (解耦权重衰减)

标准 Adam 中，权重衰减（L2 正则化）和自适应学习率耦合在一起，效果不理想。**AdamW** 将权重衰减从梯度更新中解耦出来：

$$
\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)
$$

**AdamW 是目前大模型预训练的标配优化器。**

### 学习率调度

大模型通常使用 **Warmup + Cosine Decay** 策略：

1. **Warmup 阶段：** 学习率从 0 线性增长到峰值（如 1000 步），避免训练初期梯度不稳定
2. **Cosine Decay 阶段：** 学习率按余弦曲线从峰值衰减到接近 0

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{t \pi}{T}\right)
$$

---

## 正则化

### Dropout

训练时随机将一部分神经元输出置零（概率 $p$），推理时关闭。

$$
\text{Dropout}(x_i) = \begin{cases} 0 & \text{概率 } p \\ \frac{x_i}{1-p} & \text{概率 } 1-p \end{cases}
$$

除以 $1-p$ 是为了保持期望值不变。

**大模型的实践：** GPT-3 等大模型在预训练阶段通常**不使用 Dropout**，因为训练数据足够大，过拟合风险低。但在微调阶段，尤其是数据量少时，会加入 Dropout。

### Layer Normalization

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

其中 $\mu, \sigma^2$ 是在**特征维度**上计算的均值和方差，$\gamma, \beta$ 是可学习参数。

**为什么 Transformer 选择 LayerNorm 而非 BatchNorm？**
- BatchNorm 在 batch 维度上归一化，依赖 batch size，在序列长度不一致时不适用
- LayerNorm 在特征维度上归一化，与 batch size 和序列长度无关

**RMSNorm（Llama 使用）：** LayerNorm 的简化版，去掉了均值中心化：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma
$$

计算更快，效果相当。

### Weight Decay

权重衰减等价于对参数施加 L2 正则化，鼓励参数值保持较小，防止过拟合：

$$
\mathcal{L}_{\text{total}} = \mathcal{L} + \frac{\lambda}{2} \|\theta\|^2
$$

在 AdamW 中以解耦形式实现（见优化器部分）。

---

## 前沿优化器：Muon

### 核心思想

AdamW 是当前大模型训练的标配，但它是 **element-wise** 的优化器——每个参数独立维护动量和方差。**Muon（Matrix-based Update Optimization via Newton-schulz）** 则利用了权重矩阵的**整体结构信息**，通过矩阵符号函数 (msign) 对梯度进行正交化处理，实现更高效的参数更新。

**Muon 算法的核心更新规则：**

$$
\boldsymbol{M}_t = \beta \boldsymbol{M}_{t-1} + \boldsymbol{G}_t
$$

$$
\boldsymbol{W}_t = \boldsymbol{W}_{t-1} - \eta_t [\text{msign}(\boldsymbol{M}_t) + \lambda \boldsymbol{W}_{t-1}]
$$

其中 $\text{msign}(\boldsymbol{M})$ 是矩阵符号函数，定义为：

$$
\text{msign}(\boldsymbol{M}) = \boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^{-1/2} = \boldsymbol{U}\boldsymbol{V}^{\top}
$$

（$\boldsymbol{M} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$ 为 SVD 分解）

**直觉理解：** msign 是 sign 函数从标量到矩阵的推广。标量 sign(x) 只保留符号、丢弃幅度；msign(M) 保留矩阵的"方向"（正交结构 $UV^{\top}$）、丢弃奇异值的缩放。这相当于在 2 范数约束下做梯度下降。

### Newton-Schulz 迭代：高效计算 msign

直接对大矩阵做 SVD 计算量为 $O(nm^2)$，成本过高。Muon 使用 **Newton-Schulz 迭代**来近似 msign，只需要矩阵乘法，非常 GPU 友好：

$$
\boldsymbol{X}_{t+1} = a\boldsymbol{X}_t + b\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t) + c\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t)^2
$$

Muon 官方使用的系数为 $(a, b, c) = (3.4445, -4.7750, 2.0315)$，经 5 次迭代即可收敛。

### 简化版 Muon 实现

以下是简化版 Muon 优化器实现：

```python
import torch

class SimpleMuon:
    """简化版 Muon 优化器"""
    def __init__(self, params, lr=1e-3, momentum=0.95, ns_steps=5):
        self.lr = lr
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.state = {id(p): torch.zeros_like(p) for p in params}

    def _newton_schulz(self, M, steps=5):
        """Newton-Schulz 迭代求 msign"""
        a, b, c = 3.4445, -4.7750, 2.0315
        X = M / (M.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X
        return X

    @torch.no_grad()
    def step(self, params, weight_decay=1e-2):
        for p in params:
            if p.grad is None:
                continue
            buf = self.state[id(p)]
            # 动量累积
            buf.mul_(self.momentum).add_(p.grad, alpha=1 - self.momentum)
            # Nesterov 动量
            M = p.grad * (1 - self.momentum) + buf * self.momentum
            # msign 变换（仅对 2D 权重矩阵生效）
            dW = self._newton_schulz(M, self.ns_steps) if M.dim() == 2 else M
            # 权重衰减 + 更新
            p.data.mul_(1 - self.lr * weight_decay).add_(dW, alpha=-self.lr)
```

### Muon vs AdamW 对比

| 特性 | AdamW | Muon |
|------|-------|------|
| **更新粒度** | 逐元素 (element-wise) | 矩阵级 (matrix-wise) |
| **状态变量** | 2 组（一阶矩 + 二阶矩） | 1 组（动量） |
| **显存开销** | 参数量 x2 | 参数量 x1，更低 |
| **核心操作** | 逐元素除法 | 矩阵乘法（Newton-Schulz） |
| **适用参数** | 所有参数 | 2D 权重矩阵（Attention, FFN） |
| **收敛速度** | 基准 | 部分场景更快 |

::: tip 混合训练策略
实践中，Muon 通常与 Adam **混合使用**：Attention 和 FFN 的权重矩阵用 Muon，而 Embedding、RMSNorm、LM Head 等非矩阵参数仍用 Adam。
:::

---

## 苏格拉底时刻

1. 如果所有激活函数都换成线性函数，多层网络和单层网络有什么区别？这说明了激活函数的什么作用？
2. 手动推导：对于 $y = Wx$，$\frac{\partial L}{\partial W}$ 和 $\frac{\partial L}{\partial x}$ 分别是什么？（提示：$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} x^T$）
3. 为什么交叉熵 + Softmax 的梯度公式如此简洁（$q - p$）？这是巧合还是设计？
4. Adam 优化器同时使用一阶矩和二阶矩估计，这比普通 SGD 带来了什么优势？又引入了什么额外开销？
5. 为什么大模型预训练通常不使用 Dropout，但在微调阶段有时会加入？
6. Layer Normalization 和 Batch Normalization 的核心区别是什么？为什么 Transformer 选择了前者？
7. 梯度的计算量为什么是前向传播的两倍？（提示：每层需要对权重和输入分别求导）

---

## 常见问题 & 面试考点

| 问题 | 要点 |
|------|------|
| 反向传播的本质是什么？ | 链式法则在计算图上的应用，从输出到输入逐节点计算梯度 |
| 为什么 ReLU 比 Sigmoid 好？ | 梯度不饱和、计算简单、缓解梯度消失 |
| Adam vs SGD 选哪个？ | Adam 收敛快但可能泛化差；SGD+动量泛化好但调参难。大模型用 AdamW |
| CrossEntropyLoss 为什么接收 logits？ | 内部用 log_softmax 保证数值稳定，且梯度公式 $q-p$ 更高效 |
| 什么是梯度裁剪？ | 限制梯度范数不超过阈值，防止梯度爆炸。大模型常设 max_norm=1.0 |
| Pre-LN vs Post-LN？ | Pre-LN（先 norm 再 attention/FFN）训练更稳定，是现代大模型的标准选择 |

---

## 推荐资源

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Michael Nielsen 经典教程
- [Andrej Karpathy: Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) - 从零实现 micrograd 和 GPT
- [CS231n: Convolutional Neural Networks](https://cs231n.stanford.edu/) - Stanford 视觉识别课程（基础部分通用）
- [Deep Learning Book](https://www.deeplearningbook.org/) - Goodfellow 等人的权威教材
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) - Transformer 逐行注释实现
