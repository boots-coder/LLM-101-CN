---
title: RAG 系统实现挑战
description: Level 4 端到端实现：分块策略、Embedding 检索、Hybrid Search、Reranker、Query Rewriting、生成
topics: [build, RAG, retrieval, embedding, vector-store, BM25, hybrid-search, reranker, chunking]
---
# RAG 系统实现挑战 (Level 4)

> **难度:** 困难 | **前置知识:** [RAG 基础](/applications/rag.md)、Embedding 模型、Python NumPy 熟练 | **预计时间:** 5-8 小时

## 挑战目标

从零搭建一个端到端可运行的 RAG 系统，覆盖完整链路：

```
document loader → chunker → embedder → vector store → retriever → reranker → generator
```

不依赖 LangChain / LlamaIndex 等高层框架。所有核心组件手写，最终要在一个小知识库（~50 条 FAQ）上跑通"问 → 检索 → 引用回答"全流程。

完成后你将拥有：

- 三种分块策略（固定窗口 / 语义 / 父子文档）
- 一个 in-memory 向量存储（NumPy 实现）
- 三种检索器（Dense / BM25 / Hybrid via RRF）
- Cross-encoder reranker
- Query rewriting（HyDE + multi-query）
- 带 inline 引用的生成模块

::: tip 与 [GPT 实现挑战](gpt-build.md) 的差异
GPT 挑战是"一个模型从头训"，RAG 挑战是"多个模块拼系统"。这里几乎不写矩阵乘法，但更考验**信息流设计**和**召回-排序的工程权衡**。
:::

---

## 系统架构总览

```
                    ┌─────────────────────────────────────────┐
                    │          离线索引阶段（Indexing）        │
                    └─────────────────────────────────────────┘
   raw docs                                                        
      │                                                            
      ▼                                                            
 ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌──────────────┐
 │ Loader  │───▶│ Chunker │───▶│ Embedder │───▶│ Vector Store │
 └─────────┘    └─────────┘    └──────────┘    └──────────────┘
                     │              │                  │
                     ▼              │                  │
                ┌─────────┐         │                  │
                │ BM25    │◀────────┘                  │
                │ Index   │                            │
                └─────────┘                            │
                                                       │
                    ┌─────────────────────────────────────────┐
                    │          在线查询阶段（Querying）        │
                    └─────────────────────────────────────────┘
   query                                                          │
      │                                                            │
      ▼                                                            │
 ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────┐    │
 │ Rewriter │──▶│  Retriever   │◀──┤  Stores  │◀──┘          │    │
 │ (HyDE/   │   │ Dense+Sparse │   │  (Dense  │              │    │
 │  Multi-Q)│   │   via RRF    │   │  + BM25) │              │    │
 └──────────┘   └──────────────┘   └──────────┘              │    │
                       │                                     │    │
                       ▼                                     │    │
                 ┌──────────┐    ┌─────────┐    ┌────────┐  │    │
                 │ Reranker │───▶│ Context │───▶│ LLM    │──┘    │
                 │ (Cross-  │    │ Builder │    │ Gen +  │       │
                 │ Encoder) │    │         │    │ Cite   │       │
                 └──────────┘    └─────────┘    └────────┘       │
                                                       │         │
                                                       ▼         │
                                                  answer +       │
                                                  [doc_id]       │
```

::: warning 实现顺序建议
**先做能跑通的最简版本**（阶段 1 → 2 → 3.1 → 5.1），再回头补 BM25、Hybrid、Reranker、HyDE。一上来就做 Hybrid + Rerank + HyDE 容易卡死在调参上。
:::

### 环境准备

```bash
pip install sentence-transformers numpy scipy jieba
# 可选（用于真实 LLM 调用，本练习也提供 mock 替代方案）
pip install openai
```

本挑战默认用以下开源模型（HuggingFace 公开权重）：

| 角色 | 模型 | 备选 |
|------|------|------|
| Embedder | `BAAI/bge-small-zh-v1.5` (512 维) | `BAAI/bge-m3` |
| Reranker | `BAAI/bge-reranker-base` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generator | OpenAI / Qwen / DeepSeek API | 或用 mock LLM |

---

## 阶段 1：文档处理与分块

### 任务 1.1：固定窗口分块（带 overlap）

最朴素的策略：把长文本按字符数滑动切，相邻 chunk 重叠 overlap 个字符以保留边界上下文。

```python
from typing import List
from dataclasses import dataclass, field

@dataclass
class Chunk:
    """分块的标准数据结构（贯穿全练习）"""
    chunk_id: str           # 全局唯一 id
    doc_id: str             # 来源文档 id
    text: str               # chunk 内容
    metadata: dict = field(default_factory=dict)
    parent_id: str = None   # 阶段 1.3 用


def fixed_window_chunk(
    text: str,
    doc_id: str,
    chunk_size: int = 256,
    overlap: int = 64,
) -> List[Chunk]:
    """
    固定窗口分块。

    返回: List[Chunk]，每个 chunk 长度 ≤ chunk_size，
          相邻 chunk 共享 overlap 个字符。

    提示:
        - step = chunk_size - overlap
        - chunk_id 建议格式: f"{doc_id}::chunk_{idx:04d}"
        - 跳过纯空白 chunk
    """
    assert overlap < chunk_size, "overlap 必须 < chunk_size，否则永远走不动"
    # TODO: 你的实现
    pass


# 测试
text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 20  # 520 字符
chunks = fixed_window_chunk(text, "doc_test", chunk_size=100, overlap=20)
# 预期: 步长 80，约 7 个 chunk
print(f"分块数: {len(chunks)}")
for i in range(len(chunks) - 1):
    # 验证 overlap：当前 chunk 末尾 = 下一 chunk 开头
    tail = chunks[i].text[-20:]
    head = chunks[i + 1].text[:20]
    assert tail == head, f"chunk {i} overlap 校验失败"
print("固定窗口分块 ok")
```

::: tip chunk_size 的经验取值
- 中文 FAQ / 客服：256-512 字符
- 长文档（论文、合同）：800-1500 字符
- code chunk：建议按函数 / class 切，而不是按字符
- overlap：通常取 chunk_size 的 10%-25%
:::

### 任务 1.2：语义分块（句子边界 + 长度阈值）

固定窗口的问题：可能把一个完整的句子切成两半。语义分块按句号切，再合并到目标长度。

```python
import re

def split_sentences_zh(text: str) -> List[str]:
    """
    按中文句末标点切句。
    提示: 保留分隔符可以用 re.split(r'(?<=[。！？\n])', text)
          注意 (?<=...) 是 lookbehind，分隔符跟在前一句尾部
    """
    # TODO
    pass


def semantic_chunk(
    text: str,
    doc_id: str,
    target_size: int = 256,
    min_size: int = 64,
) -> List[Chunk]:
    """
    语义分块策略：
        1. 先切成句子
        2. 贪心累加到 target_size 附近
        3. 不在句子中间切；最后一个 chunk 太短就并入前一个

    返回: List[Chunk]
    """
    # TODO:
    #   sentences = split_sentences_zh(text)
    #   buf = []
    #   for s in sentences:
    #       if 当前 buf 长度 + s 长度 > target_size and buf 非空:
    #           落盘成 chunk
    #           buf = []
    #       buf.append(s)
    #   处理最后剩余 buf；如果太短并入前一个 chunk
    pass


# 测试
text_zh = "RAG 是检索增强生成。它可以减少幻觉。但是检索质量决定上限。" \
          "因此召回与排序都很重要。让我们继续讨论。" * 5
chunks = semantic_chunk(text_zh, "doc_zh", target_size=80)
for c in chunks:
    print(f"[{len(c.text):3d} 字] {c.text[:40]}...")
    # 断言：chunk 不会以非句末字符结尾（除最后一个）
```

::: warning 中文 vs 英文分句
英文用 `nltk.sent_tokenize` 或正则 `(?<=[.!?])\s+`；中文要用 `[。！？]` 加换行。
代码 / 表格 / Markdown 标题需要单独处理（至少不切散）。
:::

### 任务 1.3：父子文档结构

**核心动机：** 检索时用小 chunk（精度高），喂给 LLM 时用大 chunk（上下文完整）。这是工业 RAG 系统的事实标准做法之一（LlamaIndex 称为 `ParentChildSplitter`，LangChain 称为 `ParentDocumentRetriever`）。

```python
def parent_child_chunk(
    text: str,
    doc_id: str,
    parent_size: int = 1024,
    child_size: int = 256,
    child_overlap: int = 32,
) -> tuple[List[Chunk], List[Chunk]]:
    """
    返回 (parents, children)：
        - parents: 大 chunk 列表，作为 LLM 的上下文
        - children: 小 chunk 列表，参与检索
        - 每个 child.parent_id 指向所属 parent.chunk_id

    实现思路:
        1. 先用 semantic_chunk 切大 chunk（target_size=parent_size）作为 parents
        2. 对每个 parent.text 用 fixed_window_chunk 切小 chunk
        3. 设置 child.parent_id = parent.chunk_id
    """
    # TODO
    pass


# 测试
parents, children = parent_child_chunk(text_zh * 3, "doc_pc")
print(f"父 chunk: {len(parents)}, 子 chunk: {len(children)}")
# 断言: 每个 child 都能找到父
parent_ids = {p.chunk_id for p in parents}
for c in children:
    assert c.parent_id in parent_ids
```

<details>
<summary>参考实现（请先独立完成）</summary>

```python
def fixed_window_chunk(text, doc_id, chunk_size=256, overlap=64):
    step = chunk_size - overlap
    chunks = []
    for idx, start in enumerate(range(0, len(text), step)):
        piece = text[start:start + chunk_size]
        if piece.strip():
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::chunk_{idx:04d}",
                doc_id=doc_id,
                text=piece,
                metadata={"start": start, "end": start + len(piece)},
            ))
        if start + chunk_size >= len(text):
            break
    return chunks


def split_sentences_zh(text):
    parts = re.split(r'(?<=[。！？\n])', text)
    return [p.strip() for p in parts if p.strip()]


def semantic_chunk(text, doc_id, target_size=256, min_size=64):
    sentences = split_sentences_zh(text)
    chunks, buf, cur_len = [], [], 0
    for s in sentences:
        if cur_len + len(s) > target_size and buf:
            chunks.append("".join(buf))
            buf, cur_len = [], 0
        buf.append(s)
        cur_len += len(s)
    if buf:
        chunks.append("".join(buf))
    # 末尾过短，并入前一个
    if len(chunks) >= 2 and len(chunks[-1]) < min_size:
        chunks[-2] += chunks[-1]
        chunks.pop()
    return [
        Chunk(chunk_id=f"{doc_id}::sem_{i:04d}", doc_id=doc_id, text=t)
        for i, t in enumerate(chunks)
    ]
```

</details>

---

## 阶段 2：Embedding 与向量存储

### 任务 2.1：用 sentence-transformers 做 embedding

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
        # TODO: 加载模型
        # 提示: SentenceTransformer(model_name, device=device)
        # bge 模型推荐 normalize_embeddings=True，这样余弦 = 点积
        self.model = ...
        self.dim = ...  # 提示: model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        返回: shape (N, dim) 的 float32 矩阵
        提示:
            - self.model.encode(texts, batch_size=..., normalize_embeddings=normalize)
            - convert_to_numpy=True
            - 大批量时显式分 batch 避免 OOM（任务 2.3）
        """
        # TODO
        pass


# 测试
embedder = Embedder()
embs = embedder.encode(["你好世界", "RAG 检索增强生成", "今天天气不错"])
assert embs.shape[0] == 3
assert embs.shape[1] == embedder.dim
# normalize 后每行 L2 范数 ≈ 1
np.testing.assert_allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-5)
print(f"Embedding shape: {embs.shape}, dim={embedder.dim}")
```

::: tip 为什么 BGE 默认 normalize？
归一化后，**余弦相似度退化为点积**：`cos(a, b) = a·b / (|a||b|) = a·b（当 |a|=|b|=1）`。
这意味着检索时只需要矩阵乘法 `Q @ X.T`，比维护两个范数再除快得多。
:::

### 任务 2.2：手写 in-memory 向量存储

不依赖 FAISS / Chroma / Milvus，用 NumPy 实现一个最小可用的向量库。

```python
class InMemoryVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.embeddings: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self.chunks: List[Chunk] = []
        # 索引: chunk_id → 行号
        self._id2idx: dict = {}

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        增量插入。
        提示:
            - assert embeddings.shape == (len(chunks), self.dim)
            - np.vstack 拼接（生产系统会预分配避免反复拷贝）
            - 同步维护 _id2idx
        """
        # TODO
        pass

    def search(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
    ) -> List[tuple[Chunk, float]]:
        """
        返回 top-k (chunk, score) 列表，按 score 降序。

        提示:
            - query_emb shape: (dim,) 或 (1, dim)
            - 假设所有 embedding 已 L2-normalize → 点积 = cosine
            - scores = self.embeddings @ query_emb  # shape (N,)
            - 用 np.argpartition 取 top-k 比 argsort 快（不要求严格全排序）
        """
        # TODO
        pass

    def __len__(self):
        return len(self.chunks)


# 测试
store = InMemoryVectorStore(dim=embedder.dim)
chunks = [
    Chunk(chunk_id=f"c{i}", doc_id="d0", text=t)
    for i, t in enumerate([
        "RAG 通过检索外部知识来增强生成",
        "今天股市大跌，投资者损失惨重",
        "向量数据库是 RAG 的核心基础设施",
    ])
]
embs = embedder.encode([c.text for c in chunks])
store.add(chunks, embs)

q_emb = embedder.encode(["什么是 RAG？"])[0]
results = store.search(q_emb, top_k=2)
for c, s in results:
    print(f"{s:.4f}  {c.text}")
# 第一名应该是 "RAG 通过检索..."
assert "RAG" in results[0][0].text
```

### 任务 2.3：批处理 embedding

文档库很大时，一次性 encode 会 OOM。需要显式分批：

```python
def encode_in_batches(
    embedder: Embedder,
    texts: List[str],
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    分批 encode，避免显存 / 内存爆炸。

    提示:
        - 用 tqdm 显示进度
        - 每批 encode 后立即 .astype(np.float32) 节省内存
        - np.concatenate 拼最终结果
    """
    # TODO
    pass
```

::: warning 余弦 vs 点积
- **未归一化向量**：必须用余弦 `(a·b) / (|a| |b|)`，否则长向量会"虚高"
- **已归一化向量**：直接点积即可，速度快且数值稳定
- **不要混用**：如果 index 时归一化了，query 时也必须归一化，否则相似度无意义
:::

---

## 阶段 3：检索（Dense + Sparse + Hybrid）

### 任务 3.1：Dense 检索

最简单的封装——把 embedder + vector store 包成一个 `DenseRetriever`：

```python
class DenseRetriever:
    def __init__(self, embedder: Embedder, store: InMemoryVectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 5) -> List[tuple[Chunk, float]]:
        # TODO: encode query → store.search
        pass
```

### 任务 3.2：手写 BM25

BM25 是经典稀疏检索算法，对**关键词精确匹配**特别强（dense 模型可能把 "BGE-M3" 和 "M3 高速公路" 混淆，BM25 不会）。

**核心公式：**

$$
\text{BM25}(q, d) = \sum_{q_i \in q} \text{IDF}(q_i) \cdot \frac{f(q_i, d) \cdot (k_1 + 1)}{f(q_i, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

其中：

- $f(q_i, d)$：term $q_i$ 在文档 $d$ 中的出现次数
- $|d|$：文档 $d$ 的长度，$\text{avgdl}$：所有文档平均长度
- $k_1 \in [1.2, 2.0]$（默认 1.5），控制 term 频率饱和
- $b \in [0, 1]$（默认 0.75），控制文档长度归一化强度
- $\text{IDF}(q_i) = \ln\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)$（Lucene 变体，保证 ≥ 0）

```python
import math
from collections import Counter
import jieba

def tokenize_zh(text: str) -> List[str]:
    """中文分词：jieba 精确模式，过滤空白"""
    return [t for t in jieba.cut(text) if t.strip()]


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_freqs: List[Counter] = []   # 每篇文档的 term → freq
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.idf: dict = {}                  # term → idf 值
        self.N: int = 0

    def index(self, chunks: List[Chunk]):
        """
        建索引。

        提示:
            1. 对每个 chunk.text 分词，存 doc_tokens / doc_freqs / doc_len
            2. avgdl = sum(doc_len) / N
            3. 计算 IDF: 对每个出现过的 term，
               df = 包含它的文档数
               idf = ln((N - df + 0.5) / (df + 0.5) + 1)
        """
        # TODO
        pass

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        """
        计算 query 对单篇文档的 BM25 分数。

        提示:
            f = doc_freqs[doc_idx].get(qi, 0)
            len_d = doc_len[doc_idx]
            numerator = f * (k1 + 1)
            denom = f + k1 * (1 - b + b * len_d / avgdl)
            score += idf[qi] * numerator / denom
            注意 qi 不在 idf 中时跳过
        """
        # TODO
        pass

    def retrieve(self, query: str, top_k: int = 5) -> List[tuple[Chunk, float]]:
        # TODO: tokenize query → 对所有文档算 score → 取 top-k
        pass


# 测试
bm25 = BM25Retriever()
bm25.index(chunks)  # 复用阶段 2 的 chunks
results = bm25.retrieve("RAG 是什么", top_k=3)
for c, s in results:
    print(f"BM25={s:.4f}  {c.text}")
```

::: tip BM25 vs TF-IDF
TF-IDF 是 `tf * idf`，term 频率线性增长。BM25 引入 $k_1$ 让 tf **饱和**（一个词出现 100 次和 10 次差不多），并用 $b$ 惩罚长文档"刷词"。这两个改进让 BM25 在传统检索任务上几十年来都打不破。
:::

### 任务 3.3：Hybrid 检索（RRF）

Dense 擅长**语义相似**（"汽车" ≈ "轿车"），BM25 擅长**关键词精确**（"BGE-M3" 必须命中）。Hybrid 合并两者结果。

最简单也最鲁棒的融合方法是 **RRF（Reciprocal Rank Fusion）**：

$$
\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)}
$$

其中 $k$ 默认 60（Cormack et al. 2009 的经验值）。注意 RRF **不依赖原始分数的 scale**——dense 是 cosine ∈ [0,1]，BM25 可能是 [0, 30]，直接相加会被 BM25 主导。RRF 只看 rank，天然抗 scale。

```python
def rrf_fuse(
    rankings: List[List[tuple[Chunk, float]]],
    k: int = 60,
    top_k: int = 5,
) -> List[tuple[Chunk, float]]:
    """
    Reciprocal Rank Fusion.

    参数:
        rankings: 多个检索器的结果，每个是 [(chunk, score), ...] 已按 score 排序
        k: RRF 平滑常数

    返回: 按 RRF 分数降序的 top-k

    提示:
        - 用 chunk_id 去重
        - rank 从 1 开始（不是 0）
        - 维护 chunk_id → 累计 rrf_score 的字典
    """
    # TODO
    pass


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, sparse: BM25Retriever, k: int = 60):
        self.dense = dense
        self.sparse = sparse
        self.k = k

    def retrieve(self, query: str, top_k: int = 5) -> List[tuple[Chunk, float]]:
        # 提示: 各取 top_k * 2 候选，再 RRF 融合，最后取 top_k
        # TODO
        pass


# 测试: 一个 dense / sparse 各有偏好的查询
hybrid = HybridRetriever(dense_retriever, bm25)
for q in ["BGE-M3 是什么", "如何减少幻觉"]:
    print(f"\n[Q] {q}")
    for c, s in hybrid.retrieve(q, top_k=3):
        print(f"  {s:.4f}  {c.text[:50]}")
```

<details>
<summary>RRF 参考实现</summary>

```python
def rrf_fuse(rankings, k=60, top_k=5):
    scores = {}
    chunk_map = {}
    for ranking in rankings:
        for rank, (chunk, _) in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_map[chunk.chunk_id] = chunk
    sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [(chunk_map[cid], scores[cid]) for cid in sorted_ids]
```

</details>

::: warning RRF 也不是银弹
- 如果两路检索结果**几乎完全不重叠**，RRF 退化成简单 union；这通常说明你的 dense 模型或 BM25 之一坏了，先排查
- $k$ 越大越偏向"温和"融合（对 rank 不敏感），越小越偏向"赢者通吃"
- 进阶融合：可学习的 fusion（用 cross-encoder 或 LightGBM 做 learning-to-rank）
:::

---

## 阶段 4：重排与查询改写

### 任务 4.1：Cross-Encoder Reranker

**为什么需要 reranker？**

- Dense / BM25 是 **bi-encoder**：query 和 doc 独立编码，计算 `sim(q_emb, d_emb)`，**速度快但精度有上限**（query 和 doc 没有交叉信息）
- Cross-encoder：把 `[CLS] query [SEP] doc [SEP]` 拼起来一起进 BERT，输出一个相关性分数。**精度高但慢**（不能预计算 doc embedding，每次都要跑模型）

工业级 RAG 几乎都是两阶段：先用便宜的 retriever 召回 top-100，再用 reranker 精选 top-5。

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        # TODO: 加载 CrossEncoder
        self.model = ...

    def rerank(
        self,
        query: str,
        candidates: List[tuple[Chunk, float]],
        top_k: int = 5,
    ) -> List[tuple[Chunk, float]]:
        """
        提示:
            - pairs = [(query, c.text) for c, _ in candidates]
            - scores = self.model.predict(pairs)  # shape (N,)
            - 按 scores 降序排序后取 top_k
            - 返回 (chunk, rerank_score) —— 注意分数变了，不再是原 retriever 的分
        """
        # TODO
        pass


# 测试: 召回 10 → rerank 取 3
candidates = hybrid.retrieve("RAG 怎么减少幻觉", top_k=10)
reranker = Reranker()
final = reranker.rerank("RAG 怎么减少幻觉", candidates, top_k=3)
for c, s in final:
    print(f"rerank={s:.4f}  {c.text[:60]}")
```

::: tip 召回 vs 排序的资源分配
经验法则：retriever 输出 50-200 个候选，reranker 处理这批，最终给 LLM 喂 3-10 个 chunk。
- 召回数太小：top-1 错了无法挽救
- 召回数太大：reranker 算力浪费，且高 rank 噪声会污染最终结果
:::

### 任务 4.2：Query Rewriting

#### 4.2.1 HyDE（Hypothetical Document Embeddings）

**核心思想：** 用户的 query 通常是问题（"RAG 怎么减少幻觉？"），但知识库里是陈述句（"RAG 通过……减少幻觉"）。**问题和答案的 embedding 在向量空间中可能离得远**。

HyDE：先让 LLM 生成一个**假设答案**，再用假设答案的 embedding 去检索。

```python
def hyde_rewrite(query: str, llm_call) -> str:
    """
    HyDE: 让 LLM 写一段假设性答案，用它去检索。
    
    参数:
        query: 用户原始问题
        llm_call: callable, llm_call(prompt: str) -> str
    
    返回: 假设答案文本
    """
    prompt = f"""请针对以下问题，写一段简短的假设性回答（2-3 句话）。
不需要保证事实正确，只要在语义上接近真实答案即可。

问题: {query}

假设性回答:"""
    # TODO: hypothetical = llm_call(prompt).strip()
    # 注意: 实际系统中通常把 hypothetical 拼上原 query 一起 encode
    #       例如 return f"{query}\n{hypothetical}"
    pass
```

#### 4.2.2 Multi-Query

让 LLM 把一个 query 改写成多个不同表述，分别检索，最后用 RRF 融合。

```python
def multi_query_rewrite(query: str, llm_call, n: int = 3) -> List[str]:
    """
    生成 n 个语义等价但表述不同的 query 变体。
    
    提示:
        - prompt 中要求 LLM 输出 JSON list 或编号列表
        - 解析失败时 fallback 到原 query
        - 一定要把原 query 也保留（n 个改写 + 1 个原始）
    """
    prompt = f"""请把下面的问题改写成 {n} 个不同的表达方式，覆盖：
1. 同义词替换
2. 更具体的提问
3. 更宽泛的提问

每行一个，不要编号。

原问题: {query}

改写:"""
    # TODO
    pass


class RewriteRetriever:
    """组合：multi-query → 各自 hybrid retrieve → RRF 融合"""
    def __init__(self, base: HybridRetriever, llm_call, n_rewrites: int = 3):
        self.base = base
        self.llm_call = llm_call
        self.n = n_rewrites

    def retrieve(self, query: str, top_k: int = 5) -> List[tuple[Chunk, float]]:
        # TODO:
        #   queries = [query] + multi_query_rewrite(query, self.llm_call, self.n)
        #   rankings = [self.base.retrieve(q, top_k=top_k * 2) for q in queries]
        #   return rrf_fuse(rankings, top_k=top_k)
        pass
```

::: warning HyDE 的成本
HyDE 每次 query 都要调 LLM 生成假设答案，**延迟和费用翻倍以上**。生产系统通常只对"低召回置信度"的 query 启用 HyDE（比如 dense top-1 分数 < 阈值时才触发）。
:::

---

## 阶段 5：生成与引用

### 任务 5.1：Context 拼接 + 生成

把 retrieved chunks 拼成 prompt，让 LLM 基于上下文回答。

```python
def build_rag_prompt(query: str, contexts: List[Chunk]) -> str:
    """
    构造 RAG prompt。

    设计要点:
        - 每个 chunk 前加显式编号 [doc_id]，方便引用
        - 明确告诉 LLM "只基于上下文回答，找不到就说不知道"
        - 不要塞太多 chunk（5-10 个，超过会稀释）
    """
    blocks = []
    for c in contexts:
        # 用 chunk_id 或 doc_id 作为引用标识
        ref = c.metadata.get("ref", c.chunk_id)
        blocks.append(f"[{ref}]\n{c.text}")
    context_str = "\n\n".join(blocks)

    return f"""你是一个严谨的问答助手。请基于下面提供的【参考资料】回答用户问题。

要求:
1. 只使用参考资料中的信息，不要编造
2. 在回答中用 [ref_id] 标注每条信息的来源
3. 如果参考资料不足以回答，请明确说"根据现有资料无法回答"

【参考资料】
{context_str}

【用户问题】
{query}

【回答】"""


def rag_answer(
    query: str,
    retriever,
    reranker,
    llm_call,
    retrieve_k: int = 20,
    final_k: int = 5,
) -> tuple[str, List[Chunk]]:
    """完整 RAG 流程: retrieve → rerank → generate"""
    # TODO:
    #   candidates = retriever.retrieve(query, top_k=retrieve_k)
    #   if reranker: candidates = reranker.rerank(query, candidates, top_k=final_k)
    #   else: candidates = candidates[:final_k]
    #   contexts = [c for c, _ in candidates]
    #   prompt = build_rag_prompt(query, contexts)
    #   answer = llm_call(prompt)
    #   return answer, contexts
    pass
```

### 任务 5.2：Inline 引用

让 LLM 输出形如 `RAG 通过检索外部知识来减少幻觉 [doc_001]，但召回质量决定上限 [doc_007]。` 的回答。

```python
def parse_citations(answer: str) -> tuple[str, List[str]]:
    """
    从答案中提取所有 [xxx] 引用。

    提示: 用 re.findall(r'\[([^\]]+)\]', answer)
    返回: (清理后的文本可选, 引用 id 列表 in order)
    """
    # TODO
    pass


def validate_citations(answer: str, contexts: List[Chunk]) -> dict:
    """
    校验答案中的引用是否都来自实际的 contexts。
    
    返回: {
        "cited": [ref_id, ...],          # 答案中实际出现的引用
        "valid": [ref_id, ...],           # 在 contexts 中找得到的
        "hallucinated": [ref_id, ...],    # 编造的（不在 contexts 中）
        "unused": [ref_id, ...],          # contexts 中存在但答案没引用
    }
    """
    valid_refs = {c.metadata.get("ref", c.chunk_id) for c in contexts}
    # TODO
    pass
```

::: tip 引用准确率是 RAG 的关键 KPI
即使答案文字正确，如果引用错乱（说"根据 [doc_3]……"但 doc_3 实际没这个信息），用户的信任会崩塌。生产系统会做 **citation grounding 校验**：把每句话和它引用的 chunk 一起喂给一个 NLI 模型，检查 entailment。
:::

### 任务 5.3（可选）：Streaming + 边输出边引用

```python
def rag_answer_streaming(query, retriever, reranker, llm_stream_call):
    """
    生成时边输出边匹配引用。
    
    实现思路:
        1. retrieve + rerank 拿到 contexts
        2. llm_stream_call 返回 generator 逐 token 输出
        3. 用 buffer 累积输出，一旦 buffer 中出现完整 [xxx]，
           检查是否是 contexts 中的有效 ref，
           然后 yield 一个事件 {type: "citation", ref: xxx, chunk: ...}
        4. 同时正常 yield 文本 token
    """
    # TODO（选做）
    pass
```

---

## 端到端验证

构造一个 toy 知识库（50 条 FAQ），跑通完整流程并对比有 / 无 RAG 的回答。

```python
# 1. 准备 toy 数据
toy_faqs = [
    {"id": "faq_001", "title": "什么是 RAG", "text": "RAG 即 Retrieval-Augmented Generation，通过检索外部知识库来增强 LLM 的生成。"},
    {"id": "faq_002", "title": "RAG 的优势", "text": "RAG 可以减少幻觉、引用来源、动态更新知识，无需重新训练模型。"},
    {"id": "faq_003", "title": "什么是 chunking", "text": "Chunking 把长文档切成小块以便 embedding 和检索，常见策略有固定窗口、语义分块、父子文档。"},
    # ... 凑到 50 条，覆盖 RAG / Embedding / 检索 / Reranker / Agent 等主题
]

# 2. 构建索引
all_chunks = []
for faq in toy_faqs:
    cs = semantic_chunk(faq["text"], doc_id=faq["id"], target_size=200)
    for c in cs:
        c.metadata["ref"] = faq["id"]
        c.metadata["title"] = faq["title"]
    all_chunks.extend(cs)

embedder = Embedder()
store = InMemoryVectorStore(dim=embedder.dim)
embs = encode_in_batches(embedder, [c.text for c in all_chunks])
store.add(all_chunks, embs)

dense = DenseRetriever(embedder, store)
bm25 = BM25Retriever(); bm25.index(all_chunks)
hybrid = HybridRetriever(dense, bm25)
reranker = Reranker()

# 3. 跑 5 个测试问题
test_queries = [
    "RAG 是什么？",
    "为什么要做 chunking？",
    "BM25 和向量检索哪个好？",
    "Reranker 用在 RAG 的哪个阶段？",
    "怎么减少 LLM 的幻觉？",
]

for q in test_queries:
    print(f"\n{'='*60}\n[Q] {q}")

    # 无 RAG（对照组）
    plain = llm_call(f"请回答：{q}")
    print(f"\n[无 RAG]\n{plain[:200]}")

    # 有 RAG
    answer, ctx = rag_answer(q, hybrid, reranker, llm_call)
    print(f"\n[RAG]\n{answer[:300]}")
    print(f"\n引用 chunk: {[c.metadata['ref'] for c in ctx]}")

    # 校验引用
    audit = validate_citations(answer, ctx)
    print(f"引用校验: {audit}")
```

::: tip 对比观察点
- **事实准确性**：无 RAG 容易在细节上编造（"RAG 由 X 公司在 Y 年提出"，往往年份编错）
- **可追溯性**：RAG 回答能给出 [faq_xxx]，无 RAG 不行
- **拒答能力**：问知识库外的问题，RAG 应该能说"不知道"，无 RAG 倾向于硬答
- **召回失败 case**：故意问知识库覆盖不到的问题，看 retriever 返回什么、reranker 是否过滤掉、LLM 是否拒答
:::

---

## 评分标准

### 基础要求（必须达成）

| 检查项 | 要求 |
|--------|------|
| 三种分块都能跑 | 固定 / 语义 / 父子，断言全过 |
| 向量存储可增量插入 | `add` 调用多次后总数正确 |
| Dense 检索正确 | top-1 命中预期 chunk |
| BM25 公式正确 | 与 `rank_bm25` 库（如果安装）输出 cosine 相关性 > 0.95 |
| RRF 融合 | 对一个 dense / sparse 各偏好的 query 都能正确召回两边的好结果 |
| 引用校验 | `validate_citations` 能识别幻觉引用 |
| 端到端跑通 | 5 个测试问题都能输出 answer + ctx |

### 进阶要求（挑战自我）

| 检查项 | 要求 |
|--------|------|
| Reranker 接入 | 召回 20 → rerank top 5，可观察到排序变化 |
| HyDE 实现 | 至少实现一个 query 上召回质量提升 |
| Multi-Query + RRF | 改写 3 个 query 后整体召回率提升 |
| Streaming 引用 | 输出过程中能事件式 emit citation |
| 评估指标 | 实现 Hit@k / MRR / nDCG 中至少一个 |

### 高阶挑战（选做）

- 替换 in-memory store 为 FAISS（IVF-PQ）支持百万级文档
- 用 BGE-M3 同时输出 dense + sparse + multi-vector，单模型替代当前两路检索
- 实现 **contextual chunking**（让 LLM 给每个 chunk 生成一句"上下文摘要"再 embedding，Anthropic 2024 提出）
- 实现 **Self-RAG**：让 LLM 自己判断"需不需要检索"
- 集成 evaluation harness（Ragas / TruLens）

---

## 进阶：GraphRAG / Agentic RAG

完成基础挑战后，可以进一步探索两个方向：

### GraphRAG（实体图增强）

传统 RAG 是"扁平"的——每个 chunk 独立检索。GraphRAG 在索引阶段抽取实体和关系建图，查询时同时检索 chunk 和相关子图。

**核心改动：**

```
chunker → 【新增】实体抽取（用 LLM 提取 entity / relation） → 建图
retriever → 【新增】图上做 community detection / multi-hop walk → 子图作为额外 context
```

参考开源实现：[`refrences-projects/graphrag/`](https://github.com/microsoft/graphrag)。**注意：参考思路即可，禁止照抄代码**。

适合场景：实体密集型问答（"X 和 Y 之间的关系"、"沿着 A → B → C 的因果链回答")。
不适合场景：FAQ / 单跳事实问答（建图开销远超收益）。

### Agentic RAG

把检索本身做成 **tool**，让 Agent 自己决定：

- 要不要检索？（简单问候不需要）
- 检索几次？（复杂问题需要多轮 query 改写）
- 哪些 chunk 不够再追加检索？
- 多个 sub-question 分别检索后合并？

伪代码：

```python
agent_loop:
    while not done:
        action = LLM.plan(query, history, retrieved_so_far)
        if action == "search":
            chunks = retriever.retrieve(action.query)
            history.append(chunks)
        elif action == "answer":
            answer = LLM.generate_with_context(query, retrieved_so_far)
            done = True
```

参考 [Agent 框架](/applications/agent-frameworks.md) 一节。

---

## 常见陷阱

1. **embedding 没归一化但用了点积**：搜出来的全是长向量，跟语义无关。要么 normalize，要么显式用 cosine
2. **BM25 的 IDF 出现负数**：用 Lucene 的 `+1` 平滑公式（见任务 3.2），不要用最朴素的 `log(N/df)`
3. **RRF 的 k 调成 0**：rank=1 直接吃满分，rank>1 几乎为 0，失去融合意义
4. **chunk 太大导致 reranker OOM**：cross-encoder 的输入长度通常 512 token，chunk 超过会截断，建议 reranker 阶段的 chunk ≤ 300 字
5. **引用 id 漂移**：检索时用 chunk_id，喂 LLM 时用 faq_id，最后校验时不一致——统一在 metadata 里加 `ref` 字段并贯穿全流程
6. **测试集数据泄露**：用同一批数据建索引又测试，效果会虚高。验证时至少留出 20% 不入库的 query

## 参考时间分配

| 阶段 | 内容 | 建议时间 |
|------|------|---------|
| 1 | 三种分块 | 60 分钟 |
| 2 | Embedder + 向量存储 | 60 分钟 |
| 3.1 + 3.2 | Dense + BM25 | 90 分钟 |
| 3.3 | Hybrid + RRF | 30 分钟 |
| 4.1 | Reranker | 30 分钟 |
| 4.2 | HyDE + Multi-Query | 60 分钟 |
| 5 | 生成 + 引用 | 60 分钟 |
| 端到端 | toy 数据集 + 5 个测试问题 | 60 分钟 |
| 进阶 | 选做 | 60+ 分钟 |

---

## 推荐资源

| 资源 | 链接 | 说明 |
|------|------|------|
| sentence-transformers | [GitHub](https://github.com/UKPLab/sentence-transformers) | bi-encoder + cross-encoder 一站式（Apache 2.0） |
| BGE 系列 | [GitHub](https://github.com/FlagOpen/FlagEmbedding) | 中文 embedding / reranker 标杆开源模型 |
| BM25 实现参考 | [rank_bm25](https://github.com/dorianbrown/rank_bm25) | 仅作对照，挑战要求自己写（Apache 2.0） |
| HyDE 论文 | [arXiv 2212.10496](https://arxiv.org/abs/2212.10496) | "Precise Zero-Shot Dense Retrieval without Relevance Labels" |
| RRF 原始论文 | [SIGIR 2009](https://dl.acm.org/doi/10.1145/1571941.1572114) | Cormack et al. "Reciprocal Rank Fusion outperforms Condorcet…" |
| GraphRAG | [GitHub](https://github.com/microsoft/graphrag) | 微软开源 GraphRAG，参考思路勿照抄（MIT） |
| RAGFlow | [GitHub](https://github.com/infiniflow/ragflow) | 工业级 RAG 引擎（Apache 2.0） |
| Ragas | [GitHub](https://github.com/explodinggradients/ragas) | RAG 自动评估框架 |

完成本挑战后，你可以回到 [RAG 主线](/applications/rag.md) 巩固理论，或挑战 [Agent 实现](/applications/agents.md) 把 RAG 嵌入到 ReAct loop 中。

---

## MLM 代码训练模式

完成上面的固定填空后，试试随机挖空模式 -- 每次点击「刷新」会随机遮盖不同的代码片段，帮你彻底记住每一行。

### BM25 评分核心

<CodeMasker title="BM25 公式（k1, b, IDF, 长度归一化）" :mask-ratio="0.18">
def score(self, query_tokens, doc_idx):
    score = 0.0
    f_dict = self.doc_freqs[doc_idx]
    len_d = self.doc_len[doc_idx]
    for qi in query_tokens:
        if qi not in self.idf:
            continue
        f = f_dict.get(qi, 0)
        if f == 0:
            continue
        numerator = f * (self.k1 + 1)
        denom = f + self.k1 * (1 - self.b + self.b * len_d / self.avgdl)
        score += self.idf[qi] * numerator / denom
    return score
</CodeMasker>

### RRF 融合

<CodeMasker title="Reciprocal Rank Fusion" :mask-ratio="0.18">
def rrf_fuse(rankings, k=60, top_k=5):
    scores = {}
    chunk_map = {}
    for ranking in rankings:
        for rank, (chunk, _) in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_map[chunk.chunk_id] = chunk
    sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [(chunk_map[cid], scores[cid]) for cid in sorted_ids]
</CodeMasker>

### Dense 检索 top-k

<CodeMasker title="向量存储的余弦检索" :mask-ratio="0.15">
def search(self, query_emb, top_k=5):
    if query_emb.ndim == 2:
        query_emb = query_emb[0]
    scores = self.embeddings @ query_emb
    if top_k >= len(scores):
        top_idx = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, top_k)[:top_k]
        top_idx = part[np.argsort(-scores[part])]
    return [(self.chunks[i], float(scores[i])) for i in top_idx]
</CodeMasker>

### 端到端 RAG Pipeline

<CodeMasker title="retrieve → rerank → generate" :mask-ratio="0.15">
def rag_answer(query, retriever, reranker, llm_call, retrieve_k=20, final_k=5):
    candidates = retriever.retrieve(query, top_k=retrieve_k)
    if reranker is not None:
        candidates = reranker.rerank(query, candidates, top_k=final_k)
    else:
        candidates = candidates[:final_k]
    contexts = [c for c, _ in candidates]
    prompt = build_rag_prompt(query, contexts)
    answer = llm_call(prompt)
    return answer, contexts
</CodeMasker>
