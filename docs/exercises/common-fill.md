---
title: "基础组件实现"
description: "Level 2-3 填空：Softmax、Cross Entropy、AdamW、LayerNorm、BPE"
topics: [fill-in, softmax, cross-entropy, AdamW, LayerNorm, BPE]
---
# 基础组件实现 (Level 2-3)

本练习覆盖深度学习中最基础的核心组件：Softmax、Cross Entropy、AdamW、LayerNorm 和 BPE 分词器。这些组件是理解大模型训练的基石。

::: info 前置知识
- 线性代数基础（向量运算、矩阵运算）
- 微积分基础（导数、梯度）
- PyTorch 张量操作
:::


---

## 练习 1：数值稳定的 Softmax 实现（Level 2）

直接计算 $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$ 会有数值溢出问题（当 $x_i$ 很大时 $e^{x_i} \to \infty$）。请实现数值稳定版本（Safe Softmax）：

$$\text{softmax}(x_i) = \frac{e^{x_i - c}}{\sum_j e^{x_j - c}}, \quad c = \max(x)$$

```python
import torch

def safe_softmax(logits):
    """
    数值稳定的 Softmax 实现。
    
    参数:
        logits: [batch_size, dim]
    返回:
        prob: [batch_size, dim]，每行和为 1
    """
    # TODO 1: 找到每行的最大值（用于数值稳定）
    logits_max, _ = _____

    # TODO 2: 减去最大值（广播操作，注意维度）
    logits = _____

    # TODO 3: 计算 exp
    logits = _____

    # TODO 4: 求和并归一化
    logits_sum = _____
    prob = _____
    return prob


# ====== 测试 ======
# 正常数据
logits = torch.randn(2, 5)
prob = safe_softmax(logits)
print(f"Softmax 结果: {prob[0]}")
print(f"每行和: {prob.sum(dim=-1)}")  # 应为 [1.0, 1.0]

# 极端数据（不安全的 softmax 会溢出）
logits_extreme = torch.tensor([[10.0, 2.0, 10000.0, 4.0]])
prob_extreme = safe_softmax(logits_extreme)
print(f"极端输入结果: {prob_extreme}")  # 第3个元素应接近 1.0

# 与 PyTorch 对比
prob_torch = torch.nn.functional.softmax(logits, dim=-1)
assert torch.allclose(prob, prob_torch, atol=1e-6), "与 PyTorch 结果不一致!"
print("与 PyTorch 实现一致!")
```

::: details 提示
- `logits.max(dim=-1)` 返回 (values, indices) 元组
- 减最大值时需要 `unsqueeze(1)` 或 `keepdim=True` 使维度匹配
- 求和时用 `keepdim=True` 保持维度以便广播
:::


<details>
<summary>点击查看答案</summary>

```python
def safe_softmax(logits):
    # TODO 1: 每行最大值
    logits_max, _ = logits.max(dim=-1)

    # TODO 2: 减去最大值（数值稳定化）
    logits = logits - logits_max.unsqueeze(1)

    # TODO 3: exp
    logits = logits.exp()

    # TODO 4: 归一化
    logits_sum = logits.sum(-1, keepdim=True)
    prob = logits / logits_sum
    return prob
```

**解析：**

减去最大值的数学证明（等价性）：

$$\frac{e^{x_i - c}}{\sum_j e^{x_j - c}} = \frac{e^{x_i} \cdot e^{-c}}{\sum_j e^{x_j} \cdot e^{-c}} = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

减去最大值后，最大的指数变为 $e^0 = 1$，其他指数都在 $(0, 1]$ 范围内，消除了溢出风险。这就是 "Safe Softmax" 或 "Stable Softmax" 的核心思想。

PyTorch 内部的 `F.softmax` 也使用了类似的技巧。在实际应用中，推荐使用 `F.log_softmax`（LogSoftmax），它将 softmax 和 log 合并计算，避免先算 softmax 再取 log 时可能出现的 $\log(0) = -\infty$ 问题。

</details>

---

## 练习 2：Cross Entropy 实现（Level 2）

实现分类任务中的交叉熵损失，接口与 PyTorch 的 `nn.CrossEntropyLoss` 对齐。输入是原始 logits（未经 softmax），标签是类别索引。

```python
import torch
import torch.nn.functional as F

def cross_entropy_loss(logits, labels):
    """
    实现交叉熵损失（从 logits 开始）。
    
    参数:
        logits: [batch_size, num_classes]，模型原始输出
        labels: [batch_size]，每个样本的正确类别索引
    
    返回:
        loss: 标量，batch 平均的交叉熵
    
    公式:
        CE = -log(softmax(logits)[label])
           = -(logits[label] - log(sum(exp(logits))))
    """
    bs, _ = logits.shape

    # TODO 1: 使用 log_softmax 获取 log probability（数值稳定）
    logprob = _____

    # TODO 2: 用 labels 索引取出正确类别的 log prob
    # 提示: 用 torch.arange(bs) 生成行索引
    idx = torch.arange(bs)
    target_logprob = _____

    # TODO 3: 取负号并求 batch 平均
    loss = _____
    return loss


# ====== 测试 ======
torch.manual_seed(42)
bs, num_classes = 4, 10
logits = torch.randn(bs, num_classes)
labels = torch.randint(0, num_classes, (bs,))

# 从零实现
my_loss = cross_entropy_loss(logits, labels)

# PyTorch 实现
loss_fn = torch.nn.CrossEntropyLoss()
torch_loss = loss_fn(logits, labels)

print(f"自实现 CE Loss: {my_loss.item():.4f}")
print(f"PyTorch CE Loss: {torch_loss.item():.4f}")
assert torch.allclose(my_loss, torch_loss, atol=1e-5), "结果不一致!"
print("与 PyTorch 实现一致!")
```

::: details 提示
- `F.log_softmax(logits, dim=-1)` 一步获得数值稳定的 log probability
- `logprob[idx, labels]` 使用高级索引取出每个样本对应标签的 log prob
- 交叉熵 = $-\text{logprob}$ 的均值
:::


<details>
<summary>点击查看答案</summary>

```python
def cross_entropy_loss(logits, labels):
    bs, _ = logits.shape

    # TODO 1: log softmax
    logprob = F.log_softmax(logits, dim=-1)

    # TODO 2: 取出正确类别的 log prob
    idx = torch.arange(bs)
    target_logprob = logprob[idx, labels]

    # TODO 3: 取负平均
    loss = -target_logprob.mean()
    return loss
```

**解析：**

交叉熵的计算可以理解为三步：

1. **Log Softmax**：将原始 logits 转为 log probability。使用 `F.log_softmax` 而非先 softmax 再 log，原因是：
   - 数值稳定：避免 softmax 输出极小值（如 $10^{-38}$）再取 log 变成 $-\infty$
   - 计算高效：$\log\text{softmax}(x_i) = x_i - c - \log\sum_j e^{x_j - c}$，一步到位

2. **索引取值**：由于标签是 one-hot 的，交叉熵 $-\sum_c p_c \log q_c$ 中只有正确类别的 $p_c = 1$，其余为 0。所以只需取出正确类别对应的 log prob，即 `logprob[idx, labels]`。

3. **取负平均**：负对数似然的 batch 平均。

这正是 PyTorch 的 `nn.CrossEntropyLoss` 的内部实现逻辑。

</details>

---

## 练习 3：AdamW 优化器（Level 2）

实现 AdamW 优化器的核心更新步骤，包括一阶矩估计、二阶矩估计、偏差校正和权重衰减。

```python
import torch

class AdamW:
    """
    AdamW 优化器实现。
    
    与 Adam 的区别: weight decay 是解耦的（直接在权重上衰减，
    而非加到梯度中），这使得正则化效果更好。
    
    更新公式:
        m = beta1 * m + (1 - beta1) * grad          # 一阶矩
        v = beta2 * v + (1 - beta2) * grad^2         # 二阶矩
        m_hat = m / (1 - beta1^t)                     # 偏差校正
        v_hat = v / (1 - beta2^t)                     # 偏差校正
        w = w - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.w = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

        self.m = torch.zeros_like(params)  # 一阶矩
        self.v = torch.zeros_like(params)  # 二阶矩
        self.t = 0                          # 时间步

    def step(self, w, grad, weight_decay=1e-2):
        self.t += 1

        # TODO 1: 更新一阶矩估计（梯度的指数移动平均）
        self.m = _____

        # TODO 2: 更新二阶矩估计（梯度平方的指数移动平均）
        self.v = _____

        # TODO 3: 偏差校正
        m_hat = _____
        v_hat = _____

        # TODO 4: AdamW 更新（注意 weight_decay 是解耦的）
        if weight_decay is not None:
            return _____

        # 无 weight decay 时退化为 Adam
        return w - self.lr * m_hat / (v_hat.sqrt() + self.eps)


# ====== 测试 ======
torch.manual_seed(42)
w = torch.randn(10, 1)
optimizer = AdamW(w, lr=1e-3)
input_data = torch.randn(8, 10)
target = torch.randn(8, 1)

for epoch in range(1000):
    output = input_data @ w
    # 手动计算 MSE 的梯度: d/dw (0.5 * ||Xw - y||^2) = X^T(Xw - y)
    grad = input_data.T @ (output - target)

    if epoch % 200 == 0:
        loss = (0.5 / 8) * ((output - target) ** 2).sum()
        print(f"Epoch {epoch:4d}, Loss: {loss.item():.4f}")

    w = optimizer.step(w, grad, weight_decay=1e-2)

print(f"最终 Loss: {(0.5/8 * ((input_data @ w - target)**2).sum()).item():.4f}")
```

::: details 提示
- 一阶矩: `beta1 * m + (1 - beta1) * grad`
- 二阶矩: `beta2 * v + (1 - beta2) * grad.pow(2)` 或 `grad ** 2`
- 偏差校正: `m / (1 - beta1 ** t)` 和 `v / (1 - beta2 ** t)`
- AdamW 更新: `w - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w)`
:::


<details>
<summary>点击查看答案</summary>

```python
# TODO 1: 一阶矩
self.m = self.beta1 * self.m + (1 - self.beta1) * grad

# TODO 2: 二阶矩
self.v = self.beta2 * self.v + (1 - self.beta2) * grad.pow(2)

# TODO 3: 偏差校正
m_hat = self.m / (1 - self.beta1 ** self.t)
v_hat = self.v / (1 - self.beta2 ** self.t)

# TODO 4: AdamW 更新
if weight_decay is not None:
    return w - self.lr * (m_hat / (v_hat.sqrt() + self.eps) + weight_decay * w)
```

**解析：**

AdamW 的各组件作用：

1. **一阶矩 $m$（动量）**：梯度的指数移动平均，起到平滑梯度的作用，帮助跳出局部最优。$\beta_1 = 0.9$ 意味着当前梯度占 10%，历史梯度占 90%。

2. **二阶矩 $v$（自适应学习率）**：梯度平方的指数移动平均，衡量每个参数的梯度"振幅"。振幅大的参数用较小的学习率，振幅小的用较大的学习率。

3. **偏差校正**：初始化 $m=0, v=0$，前几步会严重偏向零。校正因子 $\frac{1}{1-\beta^t}$ 消除这个偏差。例如 $t=1$ 时，$m_{\text{hat}} = m / 0.1 = 10m$，放大了第一步的梯度。

4. **Weight Decay（权重衰减）**：Adam 将 L2 正则化加到梯度中（`grad + wd * w`），但这会被自适应学习率"吞掉"。AdamW 将权重衰减解耦出来（`w - lr * wd * w`），使正则化效果与学习率无关。

</details>

---

## 练习 4：LayerNorm 实现（Level 3）

实现 Layer Normalization，这是 Transformer 中的关键归一化层。与 BatchNorm 不同，LayerNorm 在特征维度上归一化，不依赖 batch 统计量。

```python
import torch
import torch.nn as nn

class MyLayerNorm(nn.Module):
    """
    Layer Normalization 实现。
    
    公式:
        y = (x - mean) / sqrt(var + eps) * gamma + beta
    
    其中 mean 和 var 沿最后一个维度（特征维度）计算。
    gamma 和 beta 是可学习参数。
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        # TODO 1: 定义可学习参数 gamma (初始化为 1) 和 beta (初始化为 0)
        self.gamma = _____
        self.beta = _____

    def forward(self, x):
        """
        参数:
            x: [..., normalized_shape]，最后一维是特征维度
        返回:
            y: 与 x 同形状
        """
        # TODO 2: 计算特征维度上的均值
        mean = _____

        # TODO 3: 计算特征维度上的方差
        var = _____

        # TODO 4: 归一化
        x_norm = _____

        # TODO 5: 仿射变换
        y = _____
        return y


# ====== 测试 ======
torch.manual_seed(42)
batch_size, seq_len, dim = 2, 5, 64

x = torch.randn(batch_size, seq_len, dim)

# 从零实现
my_ln = MyLayerNorm(dim)
y_my = my_ln(x)

# PyTorch 实现
torch_ln = nn.LayerNorm(dim)
# 使参数一致
torch_ln.weight.data.fill_(1.0)
torch_ln.bias.data.fill_(0.0)
y_torch = torch_ln(x)

print(f"输出形状: {y_my.shape}")
print(f"输出均值 (应接近0): {y_my.mean(dim=-1)[0]}")
print(f"输出方差 (应接近1): {y_my.var(dim=-1, unbiased=False)[0]}")

assert torch.allclose(y_my, y_torch, atol=1e-5), "与 PyTorch 结果不一致!"
print("与 PyTorch 实现一致!")
```

::: details 提示
- gamma: `nn.Parameter(torch.ones(normalized_shape))`
- beta: `nn.Parameter(torch.zeros(normalized_shape))`
- mean: `x.mean(dim=-1, keepdim=True)`
- var: `x.var(dim=-1, keepdim=True, unbiased=False)`（注意用无偏=False）
- 归一化: `(x - mean) / torch.sqrt(var + eps)`
- 仿射: `x_norm * gamma + beta`
:::


<details>
<summary>点击查看答案</summary>

```python
class MyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        # TODO 1
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # TODO 2: 均值
        mean = x.mean(dim=-1, keepdim=True)

        # TODO 3: 方差 (无偏估计=False，与 PyTorch LayerNorm 一致)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # TODO 4: 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # TODO 5: 仿射变换
        y = x_norm * self.gamma + self.beta
        return y
```

**解析：**

LayerNorm 的关键点：

1. **归一化维度**：沿最后一维（特征维度）计算 mean 和 var。对于 `[batch, seq, dim]` 的输入，每个 token 的特征向量独立归一化，不受 batch 中其他样本的影响。

2. **无偏方差**：PyTorch 的 `nn.LayerNorm` 使用 `unbiased=False`（即除以 $N$ 而非 $N-1$）。这是个容易踩的坑。

3. **keepdim=True**：保持维度以便广播减法和除法。

4. **epsilon**：防止方差为零时除零。通常取 $10^{-5}$。

5. **仿射变换**：$\gamma$ 和 $\beta$ 是可学习参数。$\gamma=1, \beta=0$ 时是纯归一化；训练中模型可以学习最优的缩放和偏移。

**LayerNorm vs BatchNorm：**

- BatchNorm 沿 batch 维度归一化，推理时需要全局统计量
- LayerNorm 沿特征维度归一化，每个样本独立，适合序列模型
- Transformer 使用 LayerNorm 而非 BatchNorm

</details>

---

## 练习 5：BPE 分词器核心逻辑（Level 3）

实现 Byte-Pair Encoding (BPE) 分词器的核心合并逻辑：统计相邻 token 对的频率，找到最频繁的 pair 进行合并。

```python
from collections import Counter

def get_pair_counts(token_sequences):
    """
    统计所有相邻 token 对的出现频率。
    
    参数:
        token_sequences: list[list[str]]，每个元素是一个词的 token 序列
            例如: [['l', 'o', 'w'], ['l', 'o', 'w', 'e', 'r']]
    
    返回:
        pair_counts: Counter，key 为 (token_a, token_b)，value 为频率
    """
    pair_counts = Counter()
    # TODO 1: 遍历每个 token 序列，统计所有相邻 pair 的频率
    for tokens in token_sequences:
        for i in range(len(tokens) - 1):
            pair = _____
            pair_counts[pair] += 1
    return pair_counts


def merge_pair(token_sequences, pair):
    """
    将所有出现的 pair 合并为一个新 token。
    
    参数:
        token_sequences: list[list[str]]
        pair: tuple[str, str]，要合并的 token 对
    
    返回:
        new_sequences: list[list[str]]，合并后的新序列
    """
    new_sequences = []
    merged_token = pair[0] + pair[1]  # 合并后的新 token

    for tokens in token_sequences:
        new_tokens = []
        i = 0
        while i < len(tokens):
            # TODO 2: 如果当前位置匹配 pair，合并；否则保留原 token
            if i < len(tokens) - 1 and _____:
                _____
                i += 2  # 跳过已合并的两个 token
            else:
                _____
                i += 1
        new_sequences.append(new_tokens)
    return new_sequences


def bpe_train(corpus, num_merges):
    """
    BPE 训练：重复 "统计 pair 频率 -> 合并最频繁 pair" 的过程。
    
    参数:
        corpus: list[str]，训练语料（词列表）
        num_merges: int，合并次数
    
    返回:
        token_sequences: 最终的 token 序列
        merge_rules: 合并规则列表
    """
    # 初始化：每个词拆成字符
    token_sequences = [list(word) for word in corpus]
    merge_rules = []

    for step in range(num_merges):
        # TODO 3: 统计 pair 频率
        pair_counts = _____

        if not pair_counts:
            break

        # TODO 4: 找到最频繁的 pair
        best_pair = _____

        # TODO 5: 执行合并
        token_sequences = _____
        merge_rules.append(best_pair)

        print(f"Step {step+1}: 合并 {best_pair} -> "
              f"'{best_pair[0]+best_pair[1]}', 频率: {pair_counts[best_pair]}")

    return token_sequences, merge_rules


# ====== 测试 ======
corpus = ['low', 'low', 'low', 'lower', 'newest', 'widest']

print("初始 token 序列:")
for word, tokens in zip(corpus, [list(w) for w in corpus]):
    print(f"  {word} -> {tokens}")

print("\nBPE 训练过程:")
final_tokens, rules = bpe_train(corpus, num_merges=5)

print(f"\n最终 token 序列:")
for word, tokens in zip(corpus, final_tokens):
    print(f"  {word} -> {tokens}")

print(f"\n合并规则: {rules}")
```

::: details 提示
- pair 统计: `pair = (tokens[i], tokens[i+1])`
- 合并匹配条件: `tokens[i] == pair[0] and tokens[i+1] == pair[1]`
- 合并操作: `new_tokens.append(merged_token)`
- 否则: `new_tokens.append(tokens[i])`
- 最频繁 pair: `pair_counts.most_common(1)[0][0]`
:::


<details>
<summary>点击查看答案</summary>

```python
def get_pair_counts(token_sequences):
    pair_counts = Counter()
    for tokens in token_sequences:
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += 1
    return pair_counts


def merge_pair(token_sequences, pair):
    new_sequences = []
    merged_token = pair[0] + pair[1]
    for tokens in token_sequences:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        new_sequences.append(new_tokens)
    return new_sequences


def bpe_train(corpus, num_merges):
    token_sequences = [list(word) for word in corpus]
    merge_rules = []
    for step in range(num_merges):
        # TODO 3
        pair_counts = get_pair_counts(token_sequences)
        if not pair_counts:
            break
        # TODO 4
        best_pair = pair_counts.most_common(1)[0][0]
        # TODO 5
        token_sequences = merge_pair(token_sequences, best_pair)
        merge_rules.append(best_pair)
        print(f"Step {step+1}: 合并 {best_pair} -> "
              f"'{best_pair[0]+best_pair[1]}', 频率: {pair_counts[best_pair]}")
    return token_sequences, merge_rules
```

**解析：**

BPE 分词器的训练过程：

1. **初始化**：将每个词拆分成单个字符，例如 `"low" -> ['l', 'o', 'w']`。
2. **统计 pair 频率**：遍历所有 token 序列，统计相邻 pair 的出现次数。例如 `('l', 'o')` 在 "low" 出现 3 次（因为 "low" 出现了 3 次），在 "lower" 中出现 1 次。
3. **合并最频繁的 pair**：例如 `('l', 'o')` 频率最高，则合并为新 token `"lo"`。`['l', 'o', 'w'] -> ['lo', 'w']`。
4. **重复**：在新的 token 序列上继续统计和合并，直到达到指定的合并次数。

**关键特性：**

- BPE 是一种 subword 分词方法，平衡了字符级（词汇小、序列长）和词级（词汇大、OOV 问题）的优缺点。
- 高频词会被合并为一个 token（如 "the"），低频词会被拆成子词（如 "unbelievable" -> "un" + "believ" + "able"）。
- GPT 系列和大部分现代 LLM 使用 BPE 或其变体（如 SentencePiece 的 Unigram）作为分词器。

</details>

---

## MLM 代码训练模式

完成上面的固定填空后，试试随机挖空模式 -- 每次点击「刷新」会随机遮盖不同的代码片段，帮你彻底记住每一行。

### LayerNorm 前向传播

<CodeMasker title="LayerNorm forward 完整实现" :mask-ratio="0.15">
class MyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        y = x_norm * self.gamma + self.beta
        return y
</CodeMasker>

### Safe Softmax

<CodeMasker title="数值稳定 Softmax 实现" :mask-ratio="0.15">
def safe_softmax(logits):
    logits_max, _ = logits.max(dim=-1)
    logits = logits - logits_max.unsqueeze(1)
    logits = logits.exp()
    logits_sum = logits.sum(-1, keepdim=True)
    prob = logits / logits_sum
    return prob
</CodeMasker>

### AdamW 优化器更新

<CodeMasker title="AdamW 矩估计与权重衰减" :mask-ratio="0.15">
def step(self, w, grad, weight_decay=1e-2):
    self.t += 1
    self.m = self.beta1 * self.m + (1 - self.beta1) * grad
    self.v = self.beta2 * self.v + (1 - self.beta2) * grad.pow(2)
    m_hat = self.m / (1 - self.beta1 ** self.t)
    v_hat = self.v / (1 - self.beta2 ** self.t)
    if weight_decay is not None:
        return w - self.lr * (m_hat / (v_hat.sqrt() + self.eps) + weight_decay * w)
    return w - self.lr * m_hat / (v_hat.sqrt() + self.eps)
</CodeMasker>

### Cross Entropy Loss

<CodeMasker title="交叉熵损失从 logits 到 loss" :mask-ratio="0.15">
def cross_entropy_loss(logits, labels):
    bs, _ = logits.shape
    logprob = F.log_softmax(logits, dim=-1)
    idx = torch.arange(bs)
    target_logprob = logprob[idx, labels]
    loss = -target_logprob.mean()
    return loss
</CodeMasker>
