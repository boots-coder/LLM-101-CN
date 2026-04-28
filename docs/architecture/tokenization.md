---
title: "分词器"
description: "BPE、WordPiece、SentencePiece 算法原理与中文分词挑战"
topics: [tokenizer, BPE, WordPiece, SentencePiece, Unigram, vocabulary]
prereqs: [fundamentals/nlp-basics]
---
# 分词器（Tokenization）

> **一句话总结:** 分词器是大语言模型的"眼睛"——它决定了模型如何将原始文本切分为可计算的基本单元（Token），分词策略的好坏直接影响模型的词表大小、训练效率和多语言能力。

## 在大模型体系中的位置

分词器位于整个 LLM 流水线的最前端和最末端：输入时将文本编码为 Token ID 序列，输出时将模型预测的 Token ID 解码回文本。它不参与模型的核心计算，但它的设计深刻影响着模型能力的上限。一个糟糕的分词器会让模型"看不清"输入，而一个优秀的分词器则能让模型以更少的 Token 表达更丰富的语义。

```
原始文本 → [分词器编码] → Token ID 序列 → [Embedding 层] → 向量序列 → [Transformer] → 输出向量 → [LM Head] → Token ID → [分词器解码] → 生成文本
```

## 核心概念

### 为什么需要分词

神经网络只能处理数值，不能直接理解字符串。我们需要一种方式将文本转换为离散的整数 ID，再通过 Embedding 层映射为连续向量。这个"文本到整数序列"的过程就是分词。

分词的核心挑战在于**粒度选择**：粒度太细（如单字符）会导致序列过长、语义稀薄；粒度太粗（如整词）会导致词表爆炸、无法处理未登录词（OOV）。

### 分词粒度：字符、词、子词

| 粒度 | 示例（"unhappiness"） | 优点 | 缺点 |
|------|----------------------|------|------|
| **字符级** | `u, n, h, a, p, p, i, n, e, s, s` | 词表极小，无 OOV 问题 | 序列过长，单个字符缺乏语义 |
| **词级** | `unhappiness` | 语义完整 | 词表巨大，无法处理新词和拼写错误 |
| **子词级** | `un, happi, ness` | 词表适中，能泛化到新词 | 需要学习合适的切分策略 |

现代 LLM 几乎全部采用**子词级分词（Subword Tokenization）**，它在词表大小和序列长度之间取得了最佳平衡。核心思想是：高频词保留为完整 Token，低频词拆分为更小的有意义片段。

### BPE 算法（Byte Pair Encoding）

BPE 是目前最主流的子词分词算法，被 GPT 系列、Llama、DeepSeek 等模型广泛采用。它的核心思想简洁优雅：从最小单元（字节或字符）出发，反复合并出现频率最高的相邻对。

**详细步骤：**

**第 1 步：初始化词表。** 将训练语料中的所有文本拆分为最小单元（通常是 UTF-8 字节，共 256 个），这就是初始词表。

**第 2 步：统计相邻对频率。** 遍历语料，统计所有相邻 Token 对出现的次数。例如在语料 `["l o w", "l o w e r", "n e w"]` 中，`(l, o)` 出现 2 次，`(o, w)` 出现 2 次，`(e, r)` 出现 1 次……

**第 3 步：合并最高频对。** 找到出现频率最高的相邻对（如 `(l, o)`），将语料中所有该相邻对合并为一个新 Token `lo`，并将其加入词表。

**第 4 步：重复直到达到目标词表大小。** 不断重复第 2-3 步。每次合并都会产生一个新 Token，词表大小加 1。当词表达到预设大小（如 32K、64K、128K）时停止。

```python
# BPE 训练的简化伪代码
vocab = set(all_bytes)           # 256 个初始 Token
merges = []                       # 合并规则列表

while len(vocab) < target_vocab_size:
    # 统计所有相邻对的频率
    pair_counts = count_adjacent_pairs(corpus)
    # 找到最高频的对
    best_pair = max(pair_counts, key=pair_counts.get)
    # 合并：将语料中所有 best_pair 替换为新 Token
    corpus = merge_pair(corpus, best_pair)
    # 记录合并规则
    merges.append(best_pair)
    vocab.add(best_pair[0] + best_pair[1])
```

**编码（推理时）：** 对新文本执行与训练时相同顺序的合并规则，依次应用所有 merge 操作，即可得到分词结果。

**BPE 的关键特性：**

- 频率驱动：高频子串自然被合并为整体，低频词被拆分为常见子片段
- 确定性：相同的 merge 规则对相同文本总是产生相同的分词结果
- 无需语言学知识：纯数据驱动，天然适用于多语言场景

### WordPiece

WordPiece 由 Google 提出，被 BERT 系列模型采用。它与 BPE 的主要区别在于**合并策略**：

- **BPE**：合并出现频率最高的相邻对
- **WordPiece**：合并使语言模型似然提升最大的相邻对，即选择 $\frac{P(ab)}{P(a) \cdot P(b)}$ 最大的对

这意味着 WordPiece 倾向于合并那些"共同出现远多于独立出现"的对，从信息论角度来看更加合理。但实际效果与 BPE 差距不大，且 BPE 实现更简单，因此 GPT 之后的主流模型大多选择 BPE。

WordPiece 的另一个特征是使用 `##` 前缀标记非首子词，例如 `playing` → `play ##ing`，这让人类容易辨认子词边界。

### SentencePiece

SentencePiece 并不是一种新的分词算法，而是 Google 开发的一个**分词工具库**。它的核心贡献是：

1. **将空格也视为普通字符**（用 `▁` 替换空格），从而实现"语言无关"的分词——不需要预先按空格分词，天然支持中文、日文等无空格语言
2. **支持多种算法**：可选 BPE 或 Unigram 作为底层算法
3. **端到端训练**：直接从原始文本训练分词模型，无需预处理

Llama、T5 等模型均使用 SentencePiece 训练分词器。

**Unigram 算法（SentencePiece 的另一种模式）：** 与 BPE 的自底向上合并策略相反，Unigram 采用自顶向下的剪枝策略——从一个很大的候选词表出发，反复移除对整体似然影响最小的 Token，直到词表缩小到目标大小。

### 中文分词的特殊挑战

中文没有天然的词边界（空格），且每个汉字本身就承载较完整的语义。现代 LLM 的分词器（如 GPT-4、Llama 3）通常：

- 使用 UTF-8 字节级 BPE，汉字自然被编码为多字节序列
- 通过大量中文语料训练，让常见汉字和词组被合并为单个 Token
- 词表中包含足够多的中文 Token，避免中文文本被过度拆分（Token 效率低下）

一个实用的衡量指标是**压缩率**：同样一段中文文本，好的分词器用更少的 Token 就能表达。

## 代码实战

本节从零实现一个完整的 BPE 分词器。代码分三部分：训练（学习 merge 规则）、编码（文本→Token ID）、解码（Token ID→文本）。

> BPE 实现参考了 Sennrich et al. (2016) 原始论文的算法描述。

### 第一步：统计相邻对频率

```python
def get_pair_counts(token_ids):
    """统计所有相邻 Token 对的出现次数"""
    counts = {}
    for i in range(len(token_ids) - 1):
        pair = (token_ids[i], token_ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts
```

### 第二步：执行合并

```python
def apply_merge(ids, target_pair, merged_id):
    """将 ids 中所有出现的 target_pair 替换为 merged_id"""
    out = []
    pos = 0
    while pos < len(ids):
        # 如果当前位置匹配目标对，合并为新 ID
        if pos < len(ids) - 1 and (ids[pos], ids[pos + 1]) == target_pair:
            out.append(merged_id)
            pos += 2
        else:
            out.append(ids[pos])
            pos += 1
    return out
```

### 第三步：BPE 训练

```python
def train_bpe(text, vocab_size):
    """
    从原始文本训练 BPE 分词器
    
    Args:
        text: 训练语料（字符串）
        vocab_size: 目标词表大小（必须 > 256）
    
    Returns:
        merges: 合并规则列表 [(pair, new_id), ...]
        vocab: 完整词表 {id: bytes}
    """
    assert vocab_size > 256, "词表大小必须大于 256（初始字节数）"
    
    # 初始化：将文本转为 UTF-8 字节序列
    token_ids = list(text.encode("utf-8"))
    
    # 初始词表：256 个字节
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    
    num_merges = vocab_size - 256
    for step in range(num_merges):
        # 统计相邻对频率
        pair_counts = get_pair_counts(token_ids)
        if not pair_counts:
            break  # 序列已无法继续合并
        
        # 找到最高频的对
        best_pair = max(pair_counts, key=pair_counts.get)
        
        # 分配新 ID 并合并
        new_id = 256 + step
        token_ids = apply_merge(token_ids, best_pair, new_id)
        
        # 记录合并规则和新词表项
        merges.append((best_pair, new_id))
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        
        if step < 5:  # 打印前 5 步观察训练过程
            pair_str = vocab[best_pair[0]] + vocab[best_pair[1]]
            print(f"merge {step}: {best_pair} -> {new_id} "
                  f"('{pair_str.decode('utf-8', errors='replace')}') "
                  f"freq={pair_counts[best_pair]}")
    
    return merges, vocab
```

### 第四步：编码（文本 → Token ID 序列）

```python
def encode(text, merges):
    """用训练好的 merge 规则将文本编码为 Token ID 序列"""
    token_ids = list(text.encode("utf-8"))
    
    # 按训练时的顺序依次应用每条 merge 规则
    for pair, new_id in merges:
        token_ids = apply_merge(token_ids, pair, new_id)
    
    return token_ids
```

### 第五步：解码（Token ID 序列 → 文本）

```python
def decode(token_ids, vocab):
    """将 Token ID 序列解码回文本"""
    byte_sequence = b"".join(vocab[tid] for tid in token_ids)
    return byte_sequence.decode("utf-8", errors="replace")
```

### 完整运行示例

```python
# 训练语料
corpus = """
The quick brown fox jumps over the lazy dog.
The quick brown fox is very quick and the dog is very lazy.
A quick movement of the enemy will jeopardize five gunboats.
""" * 10  # 重复 10 次增加频率信号

# 训练：词表大小 = 280（256 字节 + 24 次合并）
merges, vocab = train_bpe(corpus, vocab_size=280)
# merge 0: (32, 116) -> 256 (' t') freq=50
# merge 1: (256, 104) -> 257 (' th') freq=40
# merge 2: (101, 32) -> 258 ('e ') freq=38
# ...

# 编码
text = "The quick brown fox"
ids = encode(text, merges)
print(f"原文: '{text}'")
print(f"Token IDs: {ids}")
print(f"Token 数量: {len(ids)}（原始字节数: {len(text.encode('utf-8'))}）")

# 解码验证：必须能完美还原
reconstructed = decode(ids, vocab)
assert reconstructed == text, "解码结果与原文不一致！"
print(f"解码还原: '{reconstructed}' ✓")
```

::: tip 动手实验
试着修改 `vocab_size` 参数（如 300、400、512），观察：
1. 词表越大，同一段文本的 Token 数量如何变化？
2. 查看 `vocab` 中新增的 Token，它们是否对应有意义的子词？
3. 对中文文本运行同样的代码，观察 UTF-8 字节如何被合并为汉字 Token。
:::

## 苏格拉底时刻

请停下来思考以下问题，不急于查看答案：

1. 如果词表大小从 32K 增加到 128K，对模型有什么影响？词表越大越好吗？（提示：考虑 Embedding 层参数量、序列长度、稀有 Token 的训练充分性）
2. BPE 是贪心算法——每次只合并当前最高频的对。这是全局最优的吗？能否举一个 BPE 得到次优分词结果的例子？
3. 为什么现代 LLM 大多使用字节级（Byte-level）BPE 而非字符级 BPE？这对多语言支持有什么意义？
4. 同一段代码（如 Python 代码）经过不同的分词器，Token 数量可能差异很大——这说明了什么？代码分词有哪些特殊考虑？
5. 分词器一旦训练完成就固定不变，但语言在持续演化（新词、网络用语等）。这个矛盾如何缓解？

## 常见问题 & 面试考点

- **Q: BPE 和 WordPiece 的核心区别是什么？** BPE 按频率合并，WordPiece 按互信息（似然提升）合并。实际效果差异不大，BPE 更主流。
- **Q: 为什么分词器需要单独训练，而不是和模型一起端到端学习？** 分词器将连续文本离散化为 Token ID，这个过程不可微分，无法用梯度反向传播优化。
- **Q: 词表大小如何选择？** 通常在 32K-128K 之间。太小导致序列过长（增加计算成本），太大导致 Embedding 参数量过大且稀有 Token 学不充分。GPT-2 用 50K，Llama 2 用 32K，Llama 3 用 128K。
- **Q: 什么是 Token 效率？** 表达同样内容所需的 Token 数量。Token 效率越高，模型能在固定上下文窗口内处理更多内容，推理成本也更低。

## 推荐资源

- **Andrej Karpathy "Let's build the GPT Tokenizer"** — 从零手写 BPE 分词器的视频教程，极其推荐
- **Sennrich et al.《Neural Machine Translation of Rare Words with Subword Units》** — BPE 应用于 NLP 的原始论文
- **Kudo & Richardson《SentencePiece: A simple and language independent subword tokenizer》** — SentencePiece 工具论文
- **HuggingFace Tokenizers 文档** — 工业级分词器实现，支持 BPE、WordPiece、Unigram
- **OpenAI tiktoken 库** — GPT 系列模型使用的分词器实现，可以直接体验不同模型的分词效果

### 代码参考

- **LLMs-from-scratch / ch02**（Sebastian Raschka，Manning 出版书《Build a Large Language Model (From Scratch)》ch02 BPE 番外）：[bpe-from-scratch.ipynb](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/05_bpe-from-scratch/bpe-from-scratch.ipynb) 把上面的伪代码扩展成一个可工作的 `BPETokenizerSimple` 类，补齐了本页代码实战没展开的几个工业级细节：

  - `train()` 用 `Counter` + `find_freq_pair` 在每轮迭代里全局统计相邻对，再调 `replace_pair`（基于 `collections.deque`）做就地合并，`self.bpe_merges` 记录完整 merge 序列；
  - `pretokenize_text()` 用 `Ġ` 前缀（U+0120）标记词首空格，并显式处理 `\r\n / \r / \n`，对齐 GPT-2 的预切分约定；
  - `load_vocab_and_merges_from_openai()` 直接读 OpenAI 发布的 `encoder.json` + `vocab.bpe`，并启用 `bpe_ranks`（按学习顺序定的优先级）做 GPT-2 风格的 rank-based 合并——同一份 `encode()` 因此可以无缝切到工业 tokenizer。

  ```python
  # 节选自 BPETokenizerSimple.train —— 主循环就是"找最高频对 → 替换 → 记 merge"
  for new_id in range(len(self.vocab), vocab_size):
      pair_id = self.find_freq_pair(token_id_sequences, mode="most")
      if pair_id is None:
          break
      token_id_sequences = self.replace_pair(token_id_sequences, pair_id, new_id)
      self.bpe_merges[pair_id] = new_id
  ```

  把本页的 `train_bpe` 和 notebook 里的 `BPETokenizerSimple` 对照阅读，可以清楚看到从"教学版"走到"能加载 GPT-2 词表"还差哪几步：预切分规则、特殊 token passthrough、GPT-2 ranks。
