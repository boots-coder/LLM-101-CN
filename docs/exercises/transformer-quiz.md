---
title: "Transformer 概念题"
description: "Level 1 选择题：注意力机制、位置编码、Decoder-Only 设计"
topics: [quiz, transformer, attention, positional-encoding]
---
# Transformer 概念测验 (Level 1)

> **难度:** 入门 | **前置知识:** [Transformer 架构](/architecture/transformer.md)、[注意力机制](/architecture/attention.md) | **预计时间:** 15-20 分钟

请独立完成以下 6 道选择题。每道题只有一个正确答案。先选出你的答案，再展开查看解析。

---

## 第 1 题：为什么除以 $\sqrt{d_k}$？

在 Scaled Dot-Product Attention 的公式 $\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$ 中，为什么要除以 $\sqrt{d_k}$？

**A.** 为了让注意力权重的和等于 1

**B.** 为了防止点积值过大导致 softmax 进入梯度饱和区

**C.** 为了让 Q 和 K 的维度匹配

**D.** 为了节省计算量，降低浮点运算精度要求

<details>
<summary>查看答案与解析</summary>

**正确答案：B**

当 $d_k$ 较大时，$Q$ 和 $K$ 的点积结果的方差也会变大（假设 Q、K 的每个元素独立且均值为 0、方差为 1，则点积的方差为 $d_k$）。点积值过大时，softmax 的输入值之间差异悬殊，输出会趋近于 one-hot 分布，此时梯度几乎为零（梯度饱和），导致训练困难。

除以 $\sqrt{d_k}$ 后，点积的方差恢复为 1，softmax 的输入值保持在合理范围内。

- A 错误：注意力权重的和等于 1 是 softmax 本身保证的，与缩放无关
- C 错误：Q 和 K 的维度匹配是矩阵乘法的前提，缩放不改变维度
- D 错误：除法并不降低精度，只是改变数值大小

</details>

---

## 第 2 题：Multi-Head vs Single-Head 的优势

Multi-Head Attention 相比 Single-Head Attention 的核心优势是什么？

**A.** 计算量更小，因为每个头的维度更低

**B.** 允许模型同时关注不同子空间中的不同位置的信息

**C.** 能处理更长的序列

**D.** 不需要位置编码就能捕捉位置信息

<details>
<summary>查看答案与解析</summary>

**正确答案：B**

Multi-Head Attention 将隐藏维度拆分为多个头，每个头独立学习注意力模式。不同的头可以关注不同类型的信息 — 例如某些头关注局部语法关系，某些头关注长距离依赖，某些头关注语义相似性。

这等价于在不同的"表示子空间"中并行地做注意力计算，然后将结果拼接融合。

- A 错误：虽然每个头的维度更低，但头的数量增加，总计算量与 Single-Head 基本相同（$h \times d_{head} = d_{model}$）
- C 错误：序列长度由注意力的 $O(n^2)$ 复杂度限制，与头数无关
- D 错误：Multi-Head 与位置编码是独立的机制

</details>

---

## 第 3 题：位置编码的必要性

为什么 Transformer 必须使用位置编码？

**A.** 为了让模型区分不同的 token

**B.** 为了减少注意力计算的复杂度

**C.** 因为 Self-Attention 对输入序列的顺序是置换不变的

**D.** 为了让模型能够处理变长序列

<details>
<summary>查看答案与解析</summary>

**正确答案：C**

Self-Attention 的计算本质上是集合操作 — 如果你打乱输入 token 的顺序，每个 token 与其他 token 的注意力权重只是换了个位置，但数值不变。换句话说，Self-Attention 本身不知道"顺序"这个概念。

这对于语言来说是致命的，因为"我吃鱼"和"鱼吃我"包含完全相同的 token，但含义截然不同。位置编码通过为每个位置注入唯一的位置信号，打破了这种置换不变性。

- A 错误：不同 token 已经有不同的 token embedding，位置编码解决的不是 token 区分问题
- B 错误：位置编码不影响计算复杂度
- D 错误：Transformer 处理变长序列是通过 padding + mask 实现的

</details>

---

## 第 4 题：残差连接的作用

Transformer 中的残差连接（Residual Connection）主要解决什么问题？

**A.** 减少模型的参数量

**B.** 提高模型的推理速度

**C.** 缓解深层网络中的梯度消失问题，使梯度能够直接回传

**D.** 让模型能够学习非线性变换

<details>
<summary>查看答案与解析</summary>

**正确答案：C**

残差连接的核心公式是 $\text{output} = x + F(x)$，其中 $F(x)$ 是子层（注意力或 FFN）的输出。在反向传播时，梯度可以通过"+"号直接从输出传回输入，不经过 $F(x)$ 的任何非线性变换。

这意味着即使 $F(x)$ 的梯度接近零，梯度仍然可以通过恒等映射（identity shortcut）传回更浅的层。对于 Transformer 这样动辄几十层的深层网络，残差连接是训练成功的关键。

此外，残差连接还有一个好处：模型只需要学习 $F(x) = \text{output} - x$，即输入和输出之间的"残差"，这比直接学习 $\text{output}$ 更容易优化。

- A 错误：残差连接不减少参数（它是直连，没有参数）
- B 错误：残差连接增加了一次加法，对速度几乎没影响
- D 错误：非线性变换由激活函数提供，与残差连接无关

</details>

---

## 第 5 题：Encoder vs Decoder 的关键区别

在原始 Transformer 中，Encoder 和 Decoder 最关键的结构区别是什么？

**A.** Encoder 使用 Multi-Head Attention，Decoder 使用 Single-Head Attention

**B.** Decoder 中的 Self-Attention 使用因果掩码（Causal Mask），防止看到未来位置

**C.** Encoder 有残差连接但 Decoder 没有

**D.** Encoder 处理固定长度的输入，Decoder 处理可变长度的输出

<details>
<summary>查看答案与解析</summary>

**正确答案：B**

Encoder 和 Decoder 的核心区别在于注意力掩码：

- **Encoder**：使用双向注意力（每个 token 可以看到所有其他 token），因为编码阶段可以利用完整的输入信息
- **Decoder**：使用因果掩码（Causal Mask），第 $i$ 个位置只能看到位置 $1$ 到 $i$ 的信息，不能"偷看"未来要生成的内容

此外，Decoder 还多了一层 Cross-Attention，用于关注 Encoder 的输出。但因果掩码才是最根本的区别，它决定了 Decoder 的自回归生成能力。

- A 错误：两者都使用 Multi-Head Attention
- C 错误：两者都有残差连接
- D 错误：两者都可以处理变长序列

</details>

---

## 第 6 题：LayerNorm vs BatchNorm

Transformer 为什么使用 LayerNorm 而非 BatchNorm？

**A.** LayerNorm 的计算速度更快

**B.** LayerNorm 在每个样本内部独立归一化，不依赖 batch 中其他样本，更适合变长序列

**C.** BatchNorm 无法应用于 Transformer 的结构

**D.** LayerNorm 不需要可学习的参数

<details>
<summary>查看答案与解析</summary>

**正确答案：B**

BatchNorm 沿 batch 维度计算均值和方差 — 它依赖于 batch 中所有样本的统计量。这在 NLP 中有两个问题：

1. **变长序列**：不同样本的序列长度不同，不同位置上的 batch 统计量含义不一致
2. **batch 依赖**：推理时需要使用训练时积累的 running mean/var，当 batch size 为 1 时，或者数据分布发生变化时，效果不稳定

LayerNorm 在每个样本的特征维度上独立归一化，与 batch 中的其他样本完全无关。这使得它天然适合变长序列和自回归生成（每次生成一个 token）。

- A 错误：两者计算复杂度相当
- C 错误：BatchNorm 在技术上可以应用于 Transformer，只是效果不好
- D 错误：LayerNorm 同样有可学习的 scale（$\gamma$）和 shift（$\beta$）参数

</details>

---

## 自评标准

| 正确数 | 水平 | 建议 |
|--------|------|------|
| 6/6 | 概念扎实 | 直接进入 Level 2 代码填空 |
| 4-5/6 | 基础良好 | 复习做错的知识点后进入 Level 2 |
| 2-3/6 | 需要巩固 | 建议重新学习 [Transformer 架构](/architecture/transformer.md) |
| 0-1/6 | 需要从基础开始 | 建议先完成 [基础知识](/fundamentals/) 章节 |
