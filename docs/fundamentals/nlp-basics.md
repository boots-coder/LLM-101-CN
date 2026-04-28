---
title: "NLP 基础概念"
description: "从 One-Hot 到 Word2Vec，RNN/LSTM，Seq2Seq 与注意力的起源"
topics: [embedding, Word2Vec, tokenization, RNN, LSTM, GRU, seq2seq, Bahdanau-attention, perplexity]
prereqs: [fundamentals/neural-networks]
---
# NLP 基础

> 从文本表示到序列建模，NLP 基础是理解 LLM 的前置知识。了解 RNN 的局限才能体会 Transformer 的革命性，了解 Word2Vec 才能理解 Embedding 的本质。

## 在大模型体系中的位置

```
NLP 基础 ◄── 你在这里
  ├── 文本表示     → Embedding 层是 Transformer 的入口
  ├── 分词         → Tokenizer 决定模型看到什么
  ├── 序列建模     → RNN/LSTM 的局限催生了 Transformer
  ├── Seq2Seq      → Encoder-Decoder 架构和注意力机制的起源
  └── 语言模型     → Next-token prediction 的理论框架
```

---

## 文本表示

### One-Hot 编码的局限

对于词表中包含 5 个 token 的字典 `{'i': 0, 'love': 1, 'cat': 2, 'you': 3, '!': 4}`，每个词用一个只有一个 1 的向量表示：

$$
\text{i} = [1, 0, 0, 0, 0], \quad \text{love} = [0, 1, 0, 0, 0], \quad \text{cat} = [0, 0, 1, 0, 0]
$$

**三个致命问题：**

1. **维度灾难：** 词表大小为 $V$，则每个向量是 $V$ 维的。GPT 的词表有 ~50000 个 token，one-hot 向量就是 50000 维
2. **语义缺失：** 任意两个 one-hot 向量正交，$\cos(\text{cat}, \text{dog}) = 0$，无法表达语义相似性
3. **稀疏浪费：** 几乎所有维度都是 0，信息密度极低

```python
import torch
import torch.nn.functional as F

vocab_size = 7
seq_len = 4
x = torch.randint(0, vocab_size, (1, seq_len))
one_hot = F.one_hot(x, num_classes=vocab_size)
print(x)        # tensor([[4, 1, 6, 3]])
print(one_hot)  # tensor([[[0, 0, 0, 0, 1, 0, 0],
                #          [0, 1, 0, 0, 0, 0, 0],
                #          [0, 0, 0, 0, 0, 0, 1],
                #          [0, 0, 0, 1, 0, 0, 0]]])
```

### 词嵌入的直觉

**核心思想：** 将高维离散的 one-hot 向量"嵌入"到低维连续的向量空间中，让语义相近的词在空间中靠近。

经典例子——向量空间中的语义运算：

$$
\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}
$$

这说明词嵌入捕捉到了"性别"这个语义维度。

**Embedding 的本质是一个查找表（Lookup Table）：** 存储一个 $V \times d$ 的矩阵 $E$，每行是一个 token 的 $d$ 维向量表示。给定 token id，取对应行向量即可。

以下是词嵌入的基本实现：

```python
import torch
import torch.nn as nn

vocab_size = 7
dim = 5

# 手动实现 embedding：本质就是矩阵的行索引
E = torch.randn(vocab_size, dim)  # 随机初始化 embedding 矩阵
x = torch.randint(0, vocab_size, (1, 4))  # 输入 token ids
input_embd = E[x[0, :], :]  # 取对应行向量
print(input_embd.shape)  # torch.Size([4, 5])

# PyTorch 实现：nn.Embedding 本质相同，但参数可学习
embedding_layer = nn.Embedding(vocab_size, dim)
embedding_layer.weight.data = E.clone()  # 用手动的矩阵初始化
print(embedding_layer(x))  # 与手动实现结果一致
```

**关键认识：** 随机初始化的 embedding 没有语义——语义是通过训练任务学出来的。Embedding 层就是模型的一个输入层，随着梯度反向传播自然调整参数，表征了什么完全由训练任务决定。

### Word2Vec

Word2Vec 是一种无监督学习词向量的方法，核心假设：**相邻的词之间有关联。**

**两种架构：**

- **CBOW (Continuous Bag of Words):** 用周围词预测中心词
- **Skip-gram:** 用中心词预测周围词

```
    上下文 [我, 唱, 有, 2]  →  预测中心词 "跳"    (CBOW)
    中心词 "跳"            →  预测上下文 [我, 唱, 有, 2]  (Skip-gram)
```

简化的 Word2Vec 训练代码：

```python
import torch
import torch.nn.functional as F

vocab_size = 7
dim = 5
E = torch.randn(vocab_size, dim, requires_grad=True)

# Skip-gram: 中心词预测上下文
def train(center_words, target_words, model):
    # 前向传播: 计算中心词与所有词的相似度
    logits = model[center_words, :] @ model.t()  # [n, vocab_size]
    # 计算交叉熵损失
    label = torch.tensor(target_words, dtype=torch.long)
    loss = F.cross_entropy(logits, label)
    return loss

# 训练循环
lr = 0.01
for epoch in range(100):
    E.requires_grad = True
    loss = train(center_ids, target_ids, E)
    loss.backward()
    with torch.no_grad():
        E -= lr * E.grad
    E.grad = None
```

训练后，语义相近的词在向量空间中距离更近，$E[\text{cat}] \cdot E[\text{dog}]$ 的余弦相似度会比 $E[\text{cat}] \cdot E[\text{car}]$ 更高。

---

## 分词基础

### 字符级 vs 词级 vs 子词级

| 分词方式 | 示例（"unhappiness"） | 词表大小 | 优缺点 |
|---------|---------------------|---------|--------|
| 字符级 | u, n, h, a, p, p, i, n, e, s, s | ~256 | 词表小但序列长，难学语义 |
| 词级 | unhappiness | ~100K+ | 序列短但 OOV 严重 |
| 子词级 | un, happi, ness | ~30K-50K | 平衡词表大小和表达能力 |

**子词分词（BPE/SentencePiece）是现代 LLM 的标准选择。**

### 分词的实现

以下代码展示了基础的分词规则：

```python
import re
import string

zh_symbols = '，。！？；：""''【】（）《》、'
en_symbols = re.escape(string.punctuation)
all_symbols = zh_symbols + en_symbols + ' '

special_tokens = ['<SOS>', '<EOS>', '<PAD>', '<UNK>']

# 分词正则规则
pattern = (
    r'(?:' + '|'.join(special_tokens) + ')'  # 匹配特殊 token
    r'|[' + re.escape(all_symbols) + ']'      # 匹配标点符号
    r'|\d'                                      # 匹配单个数字
    r'|[\u4e00-\u9fa5]'                         # 匹配单个中文字符
    r'|[^\s' + re.escape(all_symbols)           # 匹配连续英文单词
    + r'\d\u4e00-\u9fa5]+'
)

text = "我唱跳和rap有 2 年半。"
tokens = re.findall(pattern, text)
print(tokens)
# ['我', '唱', '跳', '和', 'rap', '有', ' ', '2', ' ', '年', '半', '。']
```

**构建词表和 Encode/Decode：**

```python
from typing import Dict

def build_vocab(token_list) -> Dict[str, int]:
    """从 token 列表构建词表"""
    vocab = {}
    idx = 0
    for token in token_list:
        if token not in vocab:
            vocab[token] = idx
            idx += 1
    return vocab

def encode(vocab, pattern, text):
    """文本 → token id 列表"""
    tokens = re.findall(pattern, text)
    token_ids = []
    for token in tokens:
        if token in vocab:
            token_ids.append(vocab[token])
        else:
            token_ids.append(vocab.get('<UNK>', -1))
    return token_ids

def decode(vocab_reverse, token_ids):
    """token id 列表 → 文本"""
    return [vocab_reverse[idx] for idx in token_ids]
```

**完整的 Tokenizer 流程：**
```
原始文本 → 预处理(清洗/规范化) → 分词(pattern matching)
→ 查词表(encode) → token_ids → 添加特殊 token(<SOS>/<EOS>)
→ Padding/Truncation → 输入模型
```

---

## 序列建模

### RNN 的基本结构

循环神经网络通过**隐状态** $h_t$ 在时间步之间传递信息：

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$
$$
y_t = W_{hy} h_t + b_y
$$

隐状态 $h_t$ 理论上编码了从 $x_1$ 到 $x_t$ 的所有历史信息。

**问题：** RNN 是串行计算的——$h_t$ 依赖 $h_{t-1}$，无法并行化。序列越长，训练越慢。

### 梯度消失问题

对于 RNN，$\frac{\partial h_T}{\partial h_1}$ 涉及连续的矩阵乘法：

$$
\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=2}^{T} W_{hh}^T \cdot \text{diag}(\tanh'(\cdot))
$$

当 $T$ 很大时，这个连乘要么趋向 0（梯度消失），要么趋向无穷（梯度爆炸），使得 RNN 难以学习**长距离依赖**。

### LSTM 的门控机制

LSTM 通过三个"门"来控制信息流，缓解梯度消失：

$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(遗忘门: 丢弃多少旧信息)}
$$
$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(输入门: 接收多少新信息)}
$$
$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(输出门: 输出多少信息)}
$$

**细胞状态**更新——核心"高速公路"：

$$
\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C) \quad \text{(候选信息)}
$$
$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(细胞状态更新)}
$$
$$
h_t = o_t \odot \tanh(C_t) \quad \text{(隐状态输出)}
$$

**直觉：** 遗忘门可以设为接近 1，让梯度通过 $C_t$ 的"高速公路"几乎无损地传播到很远的过去。这就是 LSTM 能捕捉长距离依赖的关键。

### GRU

GRU 是 LSTM 的简化版，合并了遗忘门和输入门为一个"更新门"，参数更少、计算更快：

$$
z_t = \sigma(W_z [h_{t-1}, x_t]) \quad \text{(更新门)}
$$
$$
r_t = \sigma(W_r [h_{t-1}, x_t]) \quad \text{(重置门)}
$$
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tanh(W_h [r_t \odot h_{t-1}, x_t])
$$

---

## Seq2Seq 与注意力

### Encoder-Decoder 架构

Seq2Seq 模型处理**输入序列长度 $\neq$ 输出序列长度**的任务（如翻译）：

```
Encoder: "我爱你" → h1, h2, h3 → context_vector (最后一个隐状态)
Decoder: context_vector → "I" → "love" → "you" → <EOS>
```

**瓶颈问题：** 整个输入序列被压缩到一个固定长度的 context_vector 中。对于长句子，这个向量无法承载所有信息。

### 注意力机制的起源 (Bahdanau Attention)

Bahdanau（2014）的解决方案：让 Decoder 在每一步都能"回头看" Encoder 的所有隐状态，根据需要**选择性关注**不同位置：

$$
\alpha_{t,s} = \frac{\exp(e_{t,s})}{\sum_{s'} \exp(e_{t,s'})}, \quad e_{t,s} = \text{score}(h_t^{\text{dec}}, h_s^{\text{enc}})
$$
$$
c_t = \sum_s \alpha_{t,s} h_s^{\text{enc}}
$$

这里 $\alpha_{t,s}$ 就是注意力权重——Decoder 在时间步 $t$ 对 Encoder 位置 $s$ 的关注程度。$c_t$ 是加权组合的上下文向量。

### 从 RNN+Attention 到 Transformer

RNN+Attention 仍然受限于 RNN 的串行计算。Transformer 的关键突破：

1. **去掉 RNN：** 完全基于注意力机制，不再依赖隐状态的递归传递
2. **自注意力（Self-Attention）：** 序列内部的每个位置都能直接关注所有其他位置
3. **完全并行化：** 所有位置的注意力计算可以同时进行

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

这就是 "Attention Is All You Need" 的核心含义。

序列特征组合方式对比：

| 方法 | 思路 | 特点 |
|------|------|------|
| 归并（均值池化） | $S = \frac{1}{N}\sum_j X_j$ | 一视同仁，丢失位置信息 |
| 加权组合 | $S_i = \sum_j w_{ij} X_j$ | 软选择 → 注意力机制的核心 |
| 循环神经网络 | $h_t = f(h_{t-1}, x_t)$ | 递增学习，但串行计算 |

---

## 语言模型

### 统计语言模型 (N-gram)

语言模型的目标：估计一个句子（token 序列）的概率：

$$
P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^{T} P(w_t | w_1, \ldots, w_{t-1})
$$

N-gram 用马尔可夫假设简化——只看前 $n-1$ 个词：

$$
P(w_t | w_1, \ldots, w_{t-1}) \approx P(w_t | w_{t-n+1}, \ldots, w_{t-1})
$$

**局限：** 数据稀疏（高阶 n-gram 在语料中出现频率极低）、无法捕捉长距离依赖、不具备泛化能力。

### 神经语言模型

用神经网络参数化条件概率：

$$
P(w_t | w_1, \ldots, w_{t-1}) = \text{softmax}(f_\theta(w_1, \ldots, w_{t-1}))
$$

$f_\theta$ 可以是 RNN、LSTM、或 Transformer。现代 LLM（GPT 系列）就是基于 Transformer Decoder 的超大规模神经语言模型。

### Perplexity 评估指标

困惑度（Perplexity）是语言模型最常用的评估指标：

$$
\text{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log P(w_t | w_1, \ldots, w_{t-1})\right) = \exp(\mathcal{L}_{\text{CE}})
$$

**直觉：** PPL 可以理解为模型在每个位置平均需要从多少个等概率的候选 token 中选择。PPL = 10 意味着模型平均"犹豫"在 10 个候选之间。

- PPL 越低，模型对数据的拟合越好
- PPL 与交叉熵损失是指数关系：$\text{PPL} = e^{\text{CE}}$

---

## 苏格拉底时刻

1. Word2Vec 能捕捉到 "bank" 在 "河岸" 和 "银行" 两个语境下的不同含义吗？这个问题后来是如何被解决的？（提示：静态嵌入 vs 上下文嵌入）
2. LSTM 的遗忘门在什么情况下会完全"遗忘"？这对建模长文本有什么影响？
3. Seq2Seq 模型用一个固定长度的向量来表示整个输入序列，这有什么根本性的缺陷？注意力机制是如何解决这个问题的？
4. 为什么 Transformer 论文的标题是 "Attention Is All You Need"？RNN 被完全去掉后，位置信息是如何保留的？
5. 子词分词（如 BPE）相比词级分词有什么优势？它是如何平衡词表大小和表达能力的？
6. 为什么 Embedding 矩阵的参数可以通过反向传播学习？它的梯度是什么形式？
7. Perplexity 为 1 意味着什么？为 100 又意味着什么？

---

## 推荐资源

- [CS224n: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) - Stanford NLP 经典课程
- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) - Jurafsky & Martin 教材
- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) - 可视化讲解词向量
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Colah 经典博客
- [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) - 从零实现 GPT
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 可视化讲解 Transformer
