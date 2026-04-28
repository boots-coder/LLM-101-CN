---
title: "数学基础"
description: "线性代数、微积分、概率论与信息论——大模型的数学根基"
topics: [linear-algebra, SVD, calculus, gradient, cross-entropy, KL-divergence, softmax, MLE]
---
# 数学基础

> 掌握线性代数、微积分和概率论的核心概念，是理解大模型原理的基石。没有数学直觉，你只能把模型当黑盒；有了数学直觉，你才能理解为什么这样设计、如何改进。

## 在大模型体系中的位置

```
数学基础 ◄── 你在这里
  ├── 线性代数 → Attention 的矩阵运算、LoRA 低秩分解
  ├── 微积分   → 反向传播、梯度下降、优化器设计
  └── 概率论   → Softmax、交叉熵损失、采样策略、KL 散度
```

数学是贯穿整个 LLM 学习路径的底层语言。Transformer 里的每一步计算——Query/Key 内积、Softmax 归一化、残差连接、LayerNorm——都可以用这三个数学分支来解释。

---

## 线性代数

### 向量与矩阵运算

神经网络的本质操作是**矩阵乘法**。一个全连接层 $y = Wx + b$ 就是一次线性变换。

**几何直觉：矩阵乘法 = 线性变换**

矩阵 $W$ 将输入向量 $x$ 映射到新的空间。旋转、缩放、投影都是线性变换的特例。Transformer 中的 $W_Q, W_K, W_V$ 矩阵就是把输入投影到 Query、Key、Value 三个不同的子空间。

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

**维度对齐的重要性：** 矩阵 $A \in \mathbb{R}^{m \times n}$ 和 $B \in \mathbb{R}^{n \times p}$ 相乘，要求 $A$ 的列数等于 $B$ 的行数，结果为 $\mathbb{R}^{m \times p}$。写代码时 shape 不匹配是最常见的 bug。

### 特征值与特征向量

对于方阵 $A$，如果存在非零向量 $v$ 和标量 $\lambda$ 使得：

$$
Av = \lambda v
$$

则 $v$ 是特征向量，$\lambda$ 是特征值。特征向量是矩阵变换中"方向不变"的向量，特征值表示在该方向上的缩放因子。

**SVD 分解：** 任意矩阵 $M$ 可以分解为：

$$
M = U \Sigma V^T
$$

其中 $U, V$ 是正交矩阵，$\Sigma$ 是对角矩阵，对角元素为奇异值（按降序排列）。

### 矩阵的秩与 LoRA 的关系

矩阵的**秩**（rank）是其线性无关行（或列）的最大数量。一个 $m \times n$ 的矩阵，秩最大为 $\min(m, n)$。

**LoRA 的核心思想：** 预训练模型的权重更新矩阵 $\Delta W$ 是低秩的。与其更新完整的 $d \times d$ 矩阵，不如将其分解为两个小矩阵的乘积：

$$
\Delta W = BA, \quad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times d}, \quad r \ll d
$$

参数量从 $d^2$ 降低到 $2dr$。当 $d=4096, r=16$ 时，参数量减少到原来的 $\frac{2 \times 16}{4096} \approx 0.78\%$。

### 广播机制

NumPy/PyTorch 的广播（broadcasting）规则在实际编程中无处不在：

**规则：** 从最右维度开始对齐，每个维度要么相等，要么其中一个为 1。

```python
import torch

# shape: [2, 3] + [3] -> [2, 3] + [1, 3] -> [2, 3]
A = torch.randn(2, 3)
b = torch.randn(3)
result = A + b  # b 自动广播到 [2, 3]

# Attention 中的 mask 广播
# scores: [batch, heads, seq_len, seq_len]
# mask:   [1, 1, seq_len, seq_len]  -> 广播到 4 维
```

---

## 微积分

### 导数与梯度

标量函数 $f(x)$ 的导数 $f'(x)$ 表示函数在 $x$ 处的变化率。对于多元函数 $f(x_1, x_2, \ldots, x_n)$，**梯度**是所有偏导数构成的向量：

$$
\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)
$$

梯度指向函数值增长最快的方向。梯度下降的核心思想：沿着梯度的**反方向**更新参数，使损失减小。

### 链式法则

复合函数 $f(g(x))$ 的导数：

$$
\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

这是**反向传播**的数学基础。神经网络是多层复合函数的嵌套，链式法则让我们能从输出层逐层计算梯度回传到输入层。

### 雅可比矩阵

当函数的输入和输出都是向量时，导数变成**雅可比矩阵**（Jacobian Matrix）：

$$
J = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

Softmax 的雅可比矩阵就是后面推导交叉熵梯度的关键。

### 梯度消失与梯度爆炸

**用具体数字说明：** 假设一个 50 层的网络，每层的梯度乘以一个系数 $\alpha$：

- 若 $\alpha = 0.9$：经过 50 层后梯度变为 $0.9^{50} \approx 0.005$（**消失**）
- 若 $\alpha = 1.1$：经过 50 层后梯度变为 $1.1^{50} \approx 117$（**爆炸**）

这就是深层网络训练困难的根源。解决方案包括：
- **残差连接**（ResNet / Transformer）：梯度可以跳过层直接回传
- **梯度裁剪**：限制梯度的最大范数
- **适当的初始化**：Xavier / He 初始化

---

## 概率与信息论

### 概率分布

**离散概率分布：** 骰子的每个面出现的概率 $P(X=i) = \frac{1}{6}$

**连续概率分布：** 高斯分布 $P(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

在 LLM 中，模型输出的 logits 经过 Softmax 后变成**离散概率分布**——每个 token 被选中的概率：

$$
P(\text{token}_i | \text{context}) = \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

### 信息量与熵

**自信息（Self-information）：** 一个事件的信息量与其概率成反比：

$$
I(x) = -\log P(x)
$$

直觉：越罕见的事件，信息量越大。"太阳从东方升起"信息量低；"太阳从西方升起"信息量极高。

**熵（Entropy）：** 一个分布的平均信息量，衡量不确定性：

$$
H(P) = -\sum_i P(x_i) \log P(x_i)
$$

均匀分布的熵最大（最不确定）；确定性分布的熵为 0。

### 交叉熵

交叉熵衡量用分布 $Q$ 来编码真实分布 $P$ 的平均信息量：

$$
H(P, Q) = -\sum_i P(x_i) \log Q(x_i)
$$

**为什么交叉熵是 LLM 训练的损失函数？** 在分类任务中，真实分布 $P$ 是 one-hot 向量（只有正确类别为 1），$Q$ 是模型预测的概率分布。此时交叉熵简化为：

$$
H(P, Q) = -\log Q(\text{正确类别})
$$

这就是**负对数似然**（Negative Log-Likelihood）。

以下代码演示了交叉熵的直觉和计算：

```python
import torch
import torch.nn.functional as F

# 从零实现交叉熵
def cross_entropy(p, q):
    """p 和 q 都是相同维度的概率分布"""
    return (-p * torch.log(q)).sum()

# 预测不准确时，交叉熵较大
q = torch.tensor([0.3, 0.5, 0.2])  # 预测分布
p = torch.tensor([0.0, 1.0, 0.0])  # 真实分布（类别 1）
print(cross_entropy(p, q))  # tensor(0.6931)

# 预测较准确时，交叉熵较小
q = torch.tensor([0.05, 0.9, 0.05])
print(cross_entropy(p, q))  # tensor(0.1054)
```

在分类任务中，可以进一步简化——只需要计算正确类别的负对数概率：

```python
# 批量版本，仿 PyTorch 实现
def cross_entropy_with_batch_logits(label, logits):
    """
    label size : [bs]
    logits size: [bs, classes]
    """
    bs, _ = logits.shape
    prob = F.softmax(logits, dim=-1)
    idx = torch.arange(0, bs)
    logprob = prob[idx, label].log()
    CE_loss = -logprob.mean()
    return CE_loss

# 验证与 PyTorch 一致
loss_fn = torch.nn.CrossEntropyLoss()
logits = torch.randn(4, 10)
label = torch.randint(high=10, size=(4,))
print(cross_entropy_with_batch_logits(label, logits))
print(loss_fn(logits, label))  # 结果一致
```

### KL 散度

KL 散度衡量两个分布之间的"距离"（非对称的）：

$$
D_{KL}(P \| Q) = \sum_i P(x_i) \log \frac{P(x_i)}{Q(x_i)} = H(P, Q) - H(P)
$$

**在 RLHF/GRPO 中的应用：** KL 散度被用作惩罚项，防止 RL 微调后的策略模型 $\pi_\theta$ 偏离参考模型 $\pi_{\text{ref}}$ 太远：

$$
\mathcal{L} = \mathcal{L}_{\text{reward}} + \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

**注意 KL 散度的不对称性：** $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$。前向 KL 倾向于覆盖 $P$ 的所有模式（mode-covering），反向 KL 倾向于集中在 $P$ 的某个模式（mode-seeking）。DPO 中选择哪种方向的 KL 会影响训练效果。

### 最大似然估计

给定数据集 $\{x_1, x_2, \ldots, x_N\}$，最大似然估计（MLE）寻找使数据出现概率最大的参数 $\theta$：

$$
\theta^* = \arg\max_\theta \prod_{i=1}^{N} P(x_i | \theta) = \arg\max_\theta \sum_{i=1}^{N} \log P(x_i | \theta)
$$

**LLM 的预训练目标——Next-Token Prediction——本质就是最大似然估计：**

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_1, \ldots, x_{t-1}; \theta)
$$

最大化每个位置预测正确 token 的概率之和，等价于最小化交叉熵损失。

---

## Softmax 函数

### 数学定义与直觉

Softmax 将任意实数向量（logits）转换为概率分布：

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**直觉：** 指数函数将大的 logit 值放大、小的压缩，然后归一化使得所有输出之和为 1。温度参数 $T$ 控制分布的"尖锐度"：

$$
\text{softmax}(z_i; T) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

- $T \to 0$：分布趋近 one-hot（贪心采样）
- $T \to \infty$：分布趋近均匀分布（完全随机）

### 数值稳定性问题

**Overflow 问题：** 当 logit 值很大时（如 10000），$e^{10000}$ 会溢出为 `inf`。

**Safe Softmax 解决方案：** 减去最大值不改变结果：

$$
\text{softmax}(z_i) = \frac{e^{z_i - c}}{\sum_j e^{z_j - c}}, \quad c = \max_j z_j
$$

以下是数值稳定的 Softmax 实现：

```python
import torch

def SoftMax(logits):
    """Safe Softmax 实现"""
    logits_max, _ = logits.max(dim=-1)
    logits = logits - logits_max.unsqueeze(1)  # 减去最大值
    logits = logits.exp()
    logits_sum = logits.sum(-1, keepdim=True)
    prob = logits / logits_sum
    return prob

# 正常情况下没有问题
logits = torch.randn(8, 10)
prob = SoftMax(logits)
print(prob[0].sum())  # tensor(1.)
```

**LogSoftmax 解决 log(softmax) 的溢出：** 当我们需要 $\log P$（最大似然估计中必须的），先 softmax 再 log 可能产生 `-inf`：

```python
# softmax 后再 log，数值不稳定
logits = torch.tensor([[10, 2, 10000, 4]], dtype=torch.float32)
prob = SoftMax(logits)
print(prob.log())  # tensor([[-inf, -inf, 0., -inf]])  ← 出现 -inf！

# LogSoftmax 直接计算，数值稳定
# log_softmax(x_i) = x_i - c - log(sum(exp(x_j - c)))
print(torch.nn.functional.log_softmax(logits, dim=-1))
# tensor([[-9990., -9998., 0., -9996.]])  ← 正确结果
```

```python
def LogSoftMax(logits):
    """数值稳定的 LogSoftmax"""
    logits_max, _ = logits.max(dim=-1)
    safe_logits = logits - logits_max.unsqueeze(1)
    safe_logits_exp = safe_logits.exp()
    safe_logits_sum = safe_logits_exp.sum(-1, keepdim=True)
    log_logits_sum = safe_logits_sum.log()
    log_probs = safe_logits - log_logits_sum
    return log_probs
```

$$
\text{log\_softmax}(x_i) = x_i - c - \log \sum_{j=1}^{d} e^{x_j - c}
$$

**这就是为什么 PyTorch 的 `CrossEntropyLoss` 接收 logits 而不是概率**——内部使用 `log_softmax` 直接计算，避免数值溢出，同时减少计算量。

---

## 苏格拉底时刻

在继续之前，试着回答以下问题来检验你的理解深度：

1. 为什么 Transformer 中 Q、K、V 的计算本质上是线性变换？如果去掉这些投影矩阵会发生什么？
2. 链式法则在反向传播中具体是如何工作的？当网络层数很深时，梯度可能出现什么问题？
3. 交叉熵损失函数的概率意义是什么？为什么语言模型不用均方误差（MSE）作为损失函数？
4. KL 散度不满足对称性，这在实际训练（如 RLHF 中的 KL 惩罚）中意味着什么？
5. SVD 与 LoRA 低秩适配之间有什么联系？
6. 为什么 `CrossEntropyLoss` 要求传入 logits 而非概率？（提示：数值稳定性 + 计算效率）
7. Temperature 参数如何影响 Softmax 的输出分布？$T=0.1$ 和 $T=10$ 分别产生什么效果？

---

## 推荐资源

- [3Blue1Brown 线性代数的本质](https://www.3blue1brown.com/topics/linear-algebra) - 直观可视化，强烈推荐从这里开始
- [Mathematics for Machine Learning](https://mml-book.github.io/) - 系统化教材
- [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528) - 深度学习专用微积分
- [StatQuest](https://www.youtube.com/c/joshstarmer) - 概率统计可视化讲解
- [Information Theory, Inference, and Learning Algorithms](https://www.inference.org.uk/itprnn/book.pdf) - David MacKay 经典信息论教材
