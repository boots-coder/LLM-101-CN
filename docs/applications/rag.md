---
title: "RAG 检索增强生成"
description: "RAG 架构、Chunking、Embedding、向量数据库、混合检索、GraphRAG、评估"
topics: [RAG, chunking, embedding, vector-database, hybrid-search, GraphRAG, Ragas, reranking]
---
# 检索增强生成 (RAG)

::: info 一句话总结
RAG 通过在生成前检索相关文档，让 LLM 能够基于外部知识回答问题，既减少幻觉又保持知识的时效性。
:::


## 在大模型体系中的位置

```
大模型应用层
├── Prompt Engineering（提示工程）
├── RAG（检索增强生成）◄── 你在这里
├── Agent（智能体）
├── Fine-tuning（微调）
└── 评估与对齐
```

RAG 处于应用层，是连接"静态模型"与"动态知识"的桥梁。它不修改模型参数，而是通过**检索外部信息**来增强模型的生成能力——这使得 RAG 成为企业落地最广泛的大模型应用模式。

## 为什么需要 RAG？

### LLM 的知识边界

大模型的知识"冻结"在训练截止日期：

- **时效性问题**：无法回答训练数据之后发生的事件
- **幻觉问题**：对冷门知识缺乏了解时，会编造看似合理的答案
- **私域知识**：无法访问企业内部文档、数据库等私有信息
- **可追溯性**：无法说明答案来源，难以验证正确性

### Fine-tuning vs RAG 的对比

| 维度 | RAG | Fine-tuning |
|------|-----|-------------|
| **知识更新** | 替换/新增文档即可，秒级 | 需要重新训练，小时/天级 |
| **可追溯性** | 可引用来源文档和段落 | 无法追溯知识来源 |
| **成本** | 无需 GPU 训练 | 需要训练资源和数据标注 |
| **幻觉控制** | 基于检索证据生成，可验证 | 仍可能产生幻觉 |
| **适用场景** | 知识密集型问答、客服、文档检索 | 风格迁移、领域适配、指令遵循 |
| **局限** | 依赖检索质量，不擅长复杂推理 | 灾难性遗忘，数据需求大 |

> 实践中 RAG 和 Fine-tuning 并非互斥——很多生产系统同时使用：先微调模型以适配领域风格，再用 RAG 注入实时知识。

## RAG 基础架构

### Index → Retrieve → Generate

```
                        Offline Stage (Index)
┌──────────────────────────────────────────────────────────┐
│  Documents → Chunking → Embedding Model → Vector Store   │
└──────────────────────────────────────────────────────────┘

                      Online Stage (Retrieve + Generate)
┌──────────────────────────────────────────────────────────────┐
│  User Query → Embedding → Vector Search → Top-K Chunks       │
│                                             ↓                │
│                          [Query + Retrieved Chunks] → LLM → Answer │
└──────────────────────────────────────────────────────────────┘
```

三个核心步骤：

1. **索引（Indexing）**：将文档切分为 chunk，通过 Embedding 模型转换为向量，存入向量数据库
2. **检索（Retrieval）**：将用户问题转换为向量，在向量数据库中找到最相似的 chunk
3. **生成（Generation）**：将检索到的 chunk 与问题一起送入 LLM，生成基于证据的回答

## 文本切分 (Chunking)

Chunking 是 RAG 中被严重低估的环节——切分质量直接决定检索效果。

### 固定大小切分

最简单的方案：按固定 token 数（如 512）切分，相邻 chunk 保留一定重叠。

```python
def fixed_size_chunking(text: str, chunk_size: int = 512, overlap: int = 50):
    """固定大小切分，带重叠区域"""
    tokens = text.split()  # 简化为按空格分词，实际应使用 tokenizer
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # 向前滑动，保留 overlap 个 token 的重叠
    return chunks
```

**优点：** 简单、chunk 大小均匀，便于批处理
**缺点：** 可能在句子/段落中间截断，破坏语义完整性

### 语义切分

基于 Embedding 相似度自动检测语义边界：

```python
import numpy as np
from sentence_transformers import SentenceTransformer

def semantic_chunking(sentences: list, model, threshold: float = 0.5):
    """
    语义切分：当相邻句子的语义相似度低于阈值时，在此处断开
    """
    embeddings = model.encode(sentences)
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        # 计算相邻句子的余弦相似度
        similarity = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
        )
        if similarity < threshold:
            # 语义跳变，开启新 chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
```

### 递归切分

LangChain 默认方式——按层级分隔符递归切分：先按 `\n\n`（段落），再按 `\n`（行），再按 `.`（句子），直到 chunk 小于目标大小。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,          # 目标 chunk 大小（字符数）
    chunk_overlap=50,        # 重叠区域
    separators=["\n\n", "\n", "。", ".", " ", ""]  # 分隔符优先级
)
chunks = splitter.split_text(document_text)
```

### Chunk 大小对检索质量的影响

| Chunk 大小 | 检索效果 | 生成效果 | 适用场景 |
|-----------|---------|---------|---------|
| 小 (100-200) | 精确匹配，召回率低 | 上下文不足 | FAQ、定义查询 |
| 中 (300-500) | 平衡 | 平衡 | 通用问答 |
| 大 (500-1000) | 模糊匹配，召回率高 | 上下文充足但可能引入噪声 | 长文档分析 |

> 经验法则：chunk_size = 512, overlap = 50 是一个安全的起点，然后根据评估结果调优。

## 嵌入模型 (Embedding)

Embedding 模型将文本映射到稠密向量空间，使得**语义相近的文本在向量空间中距离也近**。这是 RAG 检索能力的基础。

### 什么是文本嵌入

```python
from sentence_transformers import SentenceTransformer

# 加载嵌入模型
model = SentenceTransformer("BAAI/bge-base-zh-v1.5")

# 将文本转换为向量
text = "北京是中国的首都"
embedding = model.encode(text)   # 返回一个 768 维的浮点向量
print(f"向量维度: {embedding.shape}")   # (768,)
print(f"向量范数: {np.linalg.norm(embedding):.4f}")  # 通常归一化为 1
```

### 主流嵌入模型对比

| 模型 | 维度 | 中文支持 | 特点 |
|------|------|---------|------|
| **BGE** (BAAI/智源) | 768/1024 | 优秀 | 中文 MTEB 排行榜领先 |
| **GTE** (阿里) | 768/1024 | 优秀 | 多语言，长文本支持 |
| **E5-Mistral** | 4096 | 中等 | 基于 LLM，精度高但慢 |
| **Jina Embeddings** | 768 | 良好 | 支持 8192 token 长文本 |
| **text-embedding-3-large** (OpenAI) | 3072 | 良好 | 商用标杆，API 调用 |

### 嵌入质量的重要性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-zh-v1.5")

def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 好的嵌入模型应该做到：
q = model.encode("北京是中国的首都")
d1 = model.encode("中华人民共和国的首都是北京")  # 语义相同
d2 = model.encode("上海是中国最大的城市")          # 语义相关
d3 = model.encode("今天天气真好")                  # 语义无关

print(f"语义相同: {cosine_similarity(q, d1):.4f}")  # 应该很高 ~0.95
print(f"语义相关: {cosine_similarity(q, d2):.4f}")  # 中等 ~0.70
print(f"语义无关: {cosine_similarity(q, d3):.4f}")  # 应该很低 ~0.30
```

选择 Embedding 模型时，建议在 [MTEB 排行榜](https://huggingface.co/spaces/mteb/leaderboard) 上查看目标语言和任务类型的评测结果。

## 向量数据库

### FAISS (Facebook AI Similarity Search)

Meta 开源的向量检索库，是学术界和工业界最广泛使用的底层引擎：

```python
import faiss
import numpy as np

# 构建索引
dimension = 768                      # 向量维度
index = faiss.IndexFlatL2(dimension) # 暴力搜索（精确最近邻）

# 添加向量
vectors = np.random.random((10000, dimension)).astype('float32')
index.add(vectors)

# 查询
query = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query, k=5)  # 返回最近的 5 个向量
print(f"最近邻索引: {indices[0]}, 距离: {distances[0]}")
```

### Milvus

分布式向量数据库，适合大规模生产环境，支持十亿级向量。

### Chroma

轻量级，Python 原生，适合快速原型开发：

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

# 添加文档（自动调用 Embedding 模型）
collection.add(
    documents=["北京是中国的首都", "上海是金融中心", "深圳是科技之城"],
    ids=["doc1", "doc2", "doc3"]
)

# 查询
results = collection.query(query_texts=["中国最重要的城市"], n_results=2)
print(results["documents"])  # [['北京是中国的首都', '上海是金融中心']]
```

### 索引类型

| 索引类型 | 原理 | 时间复杂度 | 适用场景 |
|---------|------|-----------|---------|
| **Flat** | 暴力搜索，逐一比较 | O(n) | 数据量 < 10 万 |
| **IVF** | 将向量空间聚类为 Voronoi 单元，只搜索最近的 nprobe 个单元 | O(n/k) | 数据量 10 万 - 1000 万 |
| **HNSW** | 构建多层跳表式图结构，贪心搜索 | O(log n) | 高召回率要求 |
| **PQ** | 将向量切分为子向量，各自量化 | O(n) 但常数小 | 内存受限场景 |

> 实践推荐：数据量小用 Flat；中等规模用 IVF-PQ；高召回率要求用 HNSW。

## 检索策略

### Dense Retrieval（向量检索）

基于 Embedding 的语义相似度检索，是 RAG 最核心的检索方式：

```python
def dense_retrieval(query: str, model, index, documents, top_k: int = 5):
    """稠密向量检索"""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [(documents[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results
```

**优点：** 理解语义，"北京是首都" 可以匹配 "中国的首都城市"
**缺点：** 对专业术语、实体名称、编号等精确匹配较弱

### Sparse Retrieval（BM25）

经典的关键词检索算法，基于词频和逆文档频率：

```python
from rank_bm25 import BM25Okapi
import jieba

# 中文分词 + BM25
corpus = ["北京是中国的首都", "上海是金融中心", "深圳被称为科技之城"]
tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

query = "中国首都"
tokenized_query = list(jieba.cut(query))
scores = bm25.get_scores(tokenized_query)
print(f"BM25 得分: {scores}")  # 第一个文档得分最高
```

**优点：** 对精确关键词匹配（人名、产品编号）效果好
**缺点：** 无法理解同义词和语义关系

### Hybrid Search（混合检索）

将语义检索和关键词检索的结果融合，取长补短：

```python
def hybrid_search(query, dense_results, sparse_results, alpha=0.7):
    """
    混合检索：加权融合稠密检索和稀疏检索的分数
    alpha: 稠密检索的权重（0-1），1-alpha 为稀疏检索权重
    """
    combined = {}
    for doc_id, score in dense_results:
        combined[doc_id] = alpha * normalize(score)
    for doc_id, score in sparse_results:
        combined[doc_id] = combined.get(doc_id, 0) + (1 - alpha) * normalize(score)
    # 按融合分数排序
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)
```

### Re-ranking

检索返回的 top-k 结果并非都相关。用 Cross-Encoder 重排序可以显著提升精度：

```
粗检索 top-50 → Cross-Encoder 精排 → 取 top-5 → 送入 LLM
```

```python
from sentence_transformers import CrossEncoder

# Cross-Encoder 逐对打分（query, document），精度高但速度慢
reranker = CrossEncoder("BAAI/bge-reranker-base")
pairs = [(query, doc) for doc in candidate_documents]
scores = reranker.predict(pairs)
# 按分数重排，取 top-5 送入 LLM
```

## Advanced RAG

Naive RAG（简单的检索-生成管道）存在诸多瓶颈。Advanced RAG 从检索前、检索中、检索后三个阶段进行优化。

### Query Rewriting / HyDE

用户的原始问题往往不适合直接检索：

**HyDE（假设文档嵌入）：** 先让 LLM 生成一个"假设答案"，用这个假设答案去检索——因为假设答案与真实文档的词汇更接近。

```python
def hyde_retrieval(query: str, llm, retriever):
    """HyDE：用假设答案替代原始问题进行检索"""
    # 第一步：让 LLM 生成假设答案（可能不准确，但词汇接近真实文档）
    hypothetical_answer = llm.generate(
        f"请回答以下问题（尽管你可能不完全确定）：{query}"
    )
    # 第二步：用假设答案去检索
    results = retriever.search(hypothetical_answer)
    return results
```

**多查询（Multi-Query）：** 将一个问题从多个角度改写，分别检索后合并去重。

### Self-RAG（自我反思检索）

让模型自己判断是否需要检索以及检索结果是否有用：

```
1. 模型判断：这个问题需要检索外部信息吗？
   - 如果不需要 → 直接生成
   - 如果需要 → 执行检索
2. 模型判断：检索到的文档与问题相关吗？
   - 如果不相关 → 忽略该文档，尝试重新检索
   - 如果相关 → 基于文档生成答案
3. 模型判断：生成的答案有文档支撑吗？
   - 如果有 → 输出答案
   - 如果没有 → 标注不确定或重新生成
```

### Graph RAG

将文档中的实体和关系抽取为知识图谱，实现结构化推理：

```
文档 → 实体抽取 → 关系抽取 → 知识图谱
                                  ↓
用户问题 → 图检索(实体+关系) + 向量检索 → 合并结果 → LLM
```

Graph RAG 特别适合需要**多跳推理**的问题，例如："张三的导师在哪个大学任教？"需要先找到张三的导师是谁，再找到该导师的大学。

### Agentic RAG

将 RAG 与 Agent 结合，让 Agent 自主决定检索策略：

```python
def agentic_rag(query: str, agent, tools):
    """
    Agent 驱动的 RAG：不再是简单的检索-生成管道，
    而是让 Agent 自主决定何时检索、检索什么、是否需要多轮检索
    """
    # Agent 可能的行动：
    # 1. 分解问题为多个子问题
    # 2. 针对每个子问题选择不同的数据源检索
    # 3. 评估检索结果质量，决定是否追加检索
    # 4. 综合所有信息生成最终答案
    return agent.run(query, tools=tools)
```

### 完整 RAG Pipeline 代码

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SimpleRAGPipeline:
    """一个完整但简洁的 RAG 系统实现"""
    
    def __init__(self, embedding_model_name="BAAI/bge-base-zh-v1.5"):
        self.embed_model = SentenceTransformer(embedding_model_name)
        self.documents = []
        self.index = None
    
    def index_documents(self, documents: list):
        """离线索引：将文档列表编码为向量并构建 FAISS 索引"""
        self.documents = documents
        embeddings = self.embed_model.encode(documents, normalize_embeddings=True)
        dimension = embeddings.shape[1]
        
        # 使用内积索引（归一化向量的内积 = 余弦相似度）
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        print(f"索引完成：{len(documents)} 个文档，{dimension} 维向量")
    
    def retrieve(self, query: str, top_k: int = 3):
        """在线检索：根据用户问题检索最相关的文档片段"""
        query_embedding = self.embed_model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "document": self.documents[idx],
                "score": float(scores[0][i])
            })
        return results
    
    def generate_prompt(self, query: str, contexts: list):
        """构造 RAG Prompt：将检索到的上下文与问题组合"""
        context_str = "\n\n".join([f"[文档{i+1}] {c['document']}" 
                                   for i, c in enumerate(contexts)])
        prompt = f"""基于以下参考文档回答用户的问题。如果文档中没有相关信息，请说明。

参考文档：
{context_str}

用户问题：{query}

回答："""
        return prompt
    
    def query(self, question: str, top_k: int = 3):
        """端到端查询：检索 + 构造 Prompt"""
        contexts = self.retrieve(question, top_k)
        prompt = self.generate_prompt(question, contexts)
        # 在实际系统中，将 prompt 发送给 LLM 获取最终回答
        return prompt, contexts

# 使用示例
rag = SimpleRAGPipeline()
docs = [
    "Transformer 架构由 Vaswani 等人在 2017 年提出，核心是自注意力机制。",
    "BERT 是一个双向编码器模型，适合理解任务如分类和问答。",
    "GPT 系列采用仅解码器架构，通过自回归方式生成文本。",
    "LoRA 通过低秩分解来高效微调大模型，只需训练少量参数。",
    "RAG 在生成前检索相关文档，减少大模型的幻觉问题。",
]
rag.index_documents(docs)
prompt, contexts = rag.query("什么是 Transformer？")
print(f"检索到 {len(contexts)} 个相关文档")
print(f"最相关: {contexts[0]['document']} (分数: {contexts[0]['score']:.4f})")
```

## RAG 评估

### 检索质量评估

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| **Recall@K** | Top-K 结果中包含正确答案的比例 | 正确文档数 / 总正确文档数 |
| **MRR** | 第一个正确结果的排名倒数的均值 | $\frac{1}{|Q|}\sum\frac{1}{\text{rank}_i}$ |
| **NDCG@K** | 考虑排名位置的检索质量 | 综合相关性和排名位置 |

### 生成质量评估 (RAGAS)

RAGAS 框架从四个维度评估 RAG 系统：

| 维度 | 评估内容 | 无需标注 |
|------|---------|---------|
| **Faithfulness** | 答案是否忠实于检索到的文档 | 是 |
| **Answer Relevancy** | 答案是否切题 | 是 |
| **Context Precision** | 检索到的文档中有多少是有用的 | 是 |
| **Context Recall** | 回答所需的信息是否都被检索到了 | 需要标注 |

### 端到端评估

```python
# 简化的 RAG 评估流程
def evaluate_rag(rag_system, test_cases):
    """
    test_cases: [{"question": "...", "expected_docs": [...], "expected_answer": "..."}]
    """
    metrics = {"recall": [], "faithfulness": []}
    
    for case in test_cases:
        # 评估检索质量
        retrieved = rag_system.retrieve(case["question"])
        retrieved_ids = set(r["id"] for r in retrieved)
        expected_ids = set(case["expected_docs"])
        recall = len(retrieved_ids & expected_ids) / len(expected_ids)
        metrics["recall"].append(recall)
        
        # 评估生成质量（需要 LLM 判断）
        # answer = rag_system.generate(case["question"])
        # faithfulness = llm_judge(answer, retrieved)
    
    return {k: sum(v)/len(v) for k, v in metrics.items()}
```

## 苏格拉底时刻

停下来思考以下问题，不急于查看答案：

::: details 1. Chunk 策略如何影响 RAG 的效果？如果文档包含大量表格和代码怎么办？
Chunk 太小会丢失上下文，太大会引入噪声。表格和代码有特殊结构——按行切分表格会让每个 chunk 失去表头信息，切断代码函数会让 chunk 失去可理解性。解决方案包括：对表格保留完整表头、对代码按函数/类切分、使用多模态 Embedding 处理混合内容。最根本的思路是：**chunk 应该是一个自包含的、可独立理解的信息单元**。
:::


::: details 2. RAG 的检索和生成是解耦的——这带来什么问题？如何缓解？
解耦意味着检索器不知道 LLM 需要什么信息，LLM 也无法指导检索器去找什么。如果检索到的文档包含答案但不够直接，LLM 可能仍然无法正确回答。缓解方案包括：Query Rewriting（让检索器理解真实意图）、Re-ranking（用更强模型精排）、Agentic RAG（让 Agent 迭代式检索）、Self-RAG（让模型自我判断检索质量）。
:::


::: details 3. HyDE 先让 LLM 生成假设答案再检索——如果 LLM 本身就会产生幻觉呢？
HyDE 的有效性建立在一个假设上：即使 LLM 的假设答案不准确，其**词汇和风格**仍然接近真实文档，从而提升检索效果。但如果 LLM 对某个领域完全不了解（比如生成了错误的专业术语），HyDE 反而会误导检索。HyDE 最适合的场景是：LLM 对领域有基本了解但缺乏具体事实的情况。
:::


::: details 4. 当问题需要跨多个文档推理时（如'Q3 营收同比增长多少'），纯 RAG 能解决吗？
纯 RAG 的单次检索-生成管道难以处理这类问题：它需要同时找到 Q3 和去年 Q3 的数据，还要做计算。解决方案有：(1) Agentic RAG——Agent 分步检索后计算；(2) 多跳检索——先检索 Q3 数据，从中提取线索再检索去年数据；(3) 结构化存储——将财务数据存入关系型数据库，用 Text-to-SQL 替代向量检索。
:::


## GraphRAG

### 传统 RAG 的局限

传统 RAG 基于"查询 → 检索相似片段 → 生成"的管道，在以下场景表现乏力：

- **全局性问题**：如"这篇论文的核心贡献是什么？"——答案分散在多个 chunk 中，任何单个 chunk 都不完整
- **跨文档推理**：如"张三和李四有什么关联？"——需要从不同文档中提取实体关系并推理
- **主题概述**：如"这个代码库的架构设计是怎样的？"——需要全局理解，而非局部片段

根本原因：向量检索基于**局部语义相似度**，缺乏对文档集合的**全局结构性理解**。

### GraphRAG 核心流程

GraphRAG（由 Microsoft Research 提出）通过构建知识图谱，将文档的全局结构显式化。

```
Documents
   │
   ▼
┌──────────────────────────────────────────────────────┐
│ 1. Entity Extraction     → Extract named entities    │
│ 2. Relationship Extraction → Extract relations       │
│ 3. Graph Construction    → Build knowledge graph     │
│ 4. Community Detection   → Discover communities      │
│ 5. Community Summary     → Generate summaries        │
└──────────────────────────────────────────────────────┘
   │
   ▼
Query: Local Search or Global Search
```

#### 1. Entity Extraction（命名实体抽取）

用 LLM 从每个文本 chunk 中抽取命名实体（人物、组织、概念、事件等）。

```python
def extract_entities(chunk: str, llm) -> list:
    """用 LLM 从文本中抽取命名实体"""
    prompt = f"""请从以下文本中抽取所有命名实体。
对每个实体，输出 JSON 格式：{{"name": "实体名", "type": "类型", "description": "简要描述"}}

文本：{chunk}

请输出 JSON 数组："""
    
    response = llm.generate(prompt)
    entities = json.loads(response)
    return entities

# 示例输出：
# [
#   {"name": "Transformer", "type": "技术", "description": "基于自注意力的神经网络架构"},
#   {"name": "Vaswani", "type": "人物", "description": "Transformer 论文第一作者"},
#   {"name": "Google", "type": "组织", "description": "Transformer 的提出机构"}
# ]
```

#### 2. Relationship Extraction（关系抽取）

从同一个 chunk 中抽取实体之间的关系。

```python
def extract_relationships(chunk: str, entities: list, llm) -> list:
    """抽取实体之间的关系"""
    entity_names = [e["name"] for e in entities]
    prompt = f"""已知以下实体：{entity_names}
请从文本中抽取这些实体之间的关系。
输出 JSON 格式：{{"source": "实体A", "target": "实体B", "relationship": "关系描述", "weight": 权重(1-10)}}

文本：{chunk}

请输出 JSON 数组："""
    
    response = llm.generate(prompt)
    return json.loads(response)

# 示例输出：
# [
#   {"source": "Vaswani", "target": "Transformer", "relationship": "提出了", "weight": 9},
#   {"source": "Transformer", "target": "Google", "relationship": "诞生于", "weight": 7}
# ]
```

#### 3. Graph Construction（知识图谱构建）

将所有 chunk 抽取的实体和关系合并为一张图。相同实体需要去重和合并。

```python
import networkx as nx

def build_knowledge_graph(all_entities: list, all_relationships: list) -> nx.Graph:
    """构建知识图谱"""
    G = nx.Graph()
    
    # 添加节点（去重合并同名实体）
    entity_map = {}
    for entity in all_entities:
        name = entity["name"]
        if name in entity_map:
            # 合并描述
            entity_map[name]["description"] += "; " + entity["description"]
        else:
            entity_map[name] = entity
    
    for name, attrs in entity_map.items():
        G.add_node(name, **attrs)
    
    # 添加边（合并重复关系，权重累加）
    for rel in all_relationships:
        if G.has_edge(rel["source"], rel["target"]):
            G[rel["source"]][rel["target"]]["weight"] += rel["weight"]
        else:
            G.add_edge(rel["source"], rel["target"],
                      relationship=rel["relationship"],
                      weight=rel["weight"])
    
    return G
```

#### 4. Community Detection（社区发现）

使用 Leiden 算法（Louvain 的改进版）将图划分为多个社区。同一社区内的实体关联紧密，代表一个"主题簇"。

```python
import leidenalg
import igraph as ig

def detect_communities(nx_graph: nx.Graph) -> dict:
    """使用 Leiden 算法发现社区结构"""
    # NetworkX → igraph 转换
    ig_graph = ig.Graph.from_networkx(nx_graph)
    
    # 运行 Leiden 算法（支持多层级分辨率）
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=1.0  # 调整社区粒度
    )
    
    # 构建社区映射：{community_id: [node_names]}
    communities = {}
    for node_idx, comm_id in enumerate(partition.membership):
        node_name = ig_graph.vs[node_idx]["_nx_name"]
        communities.setdefault(comm_id, []).append(node_name)
    
    return communities  
    # 示例：{0: ["Transformer", "Attention", "Vaswani"], 1: ["BERT", "MLM", "NSP"], ...}
```

#### 5. Community Summarization（社区摘要生成）

为每个社区生成一段总结，描述该社区涵盖的主题和关键信息。

```python
def summarize_community(community_nodes: list, graph: nx.Graph, llm) -> str:
    """为一个社区生成摘要"""
    # 收集社区内所有实体和关系的信息
    info_parts = []
    for node in community_nodes:
        attrs = graph.nodes[node]
        info_parts.append(f"实体: {node} ({attrs.get('type', '未知')}): {attrs.get('description', '')}")
        for neighbor in graph.neighbors(node):
            if neighbor in community_nodes:
                edge_data = graph[node][neighbor]
                info_parts.append(f"  关系: {node} --[{edge_data.get('relationship', '')}]--> {neighbor}")
    
    prompt = f"""以下是一组紧密相关的实体和关系。请生成一段 200 字以内的摘要，概括这组信息的核心主题。

{chr(10).join(info_parts)}

摘要："""
    return llm.generate(prompt)
```

#### 6. Query：Local Search vs Global Search

GraphRAG 提供两种检索模式：

| 模式 | 原理 | 适用问题 |
|------|------|---------|
| **Local Search** | 从查询实体出发，在图中遍历邻近节点和关系 | "Transformer 的作者是谁？"（具体事实） |
| **Global Search** | 遍历所有社区摘要，汇总全局信息 | "这些论文的主要研究方向有哪些？"（全局概述） |

```python
def local_search(query: str, graph: nx.Graph, llm, top_k=5):
    """局部搜索：从查询实体出发，收集邻域信息"""
    # 1. 从查询中抽取实体
    query_entities = extract_entities(query, llm)
    
    # 2. 在图中找到匹配的节点，收集 k 跳邻域的信息
    context_parts = []
    for entity in query_entities:
        if entity["name"] in graph:
            neighbors = nx.single_source_shortest_path_length(graph, entity["name"], cutoff=2)
            for neighbor, distance in neighbors.items():
                node_info = graph.nodes[neighbor]
                context_parts.append(f"{neighbor}: {node_info.get('description', '')}")
    
    # 3. 用收集到的图上下文 + 传统向量检索结果，一起送入 LLM
    return context_parts[:top_k]

def global_search(query: str, community_summaries: list, llm):
    """全局搜索：遍历所有社区摘要，map-reduce 生成答案"""
    # Map：每个社区摘要独立回答
    partial_answers = []
    for summary in community_summaries:
        answer = llm.generate(f"基于以下信息回答问题（如无相关信息请回复'无'）：\n信息：{summary}\n问题：{query}")
        if answer.strip() != "无":
            partial_answers.append(answer)
    
    # Reduce：汇总所有局部答案
    final_prompt = f"问题：{query}\n\n各部分的回答：\n" + "\n---\n".join(partial_answers) + "\n\n请综合以上信息给出完整回答。"
    return llm.generate(final_prompt)
```

### GraphRAG vs 传统 RAG 对比

| 维度 | 传统 RAG | GraphRAG |
|------|---------|----------|
| **索引方式** | 向量化 chunk | 向量 + 知识图谱 + 社区摘要 |
| **检索方式** | 语义相似度 | 图遍历 + 社区摘要 |
| **全局问题** | 差（只能检索局部片段） | 好（社区摘要提供全局视角） |
| **多跳推理** | 差（单次检索） | 好（图上多跳遍历） |
| **索引成本** | 低（Embedding 计算） | 高（大量 LLM 调用做实体抽取） |
| **索引时间** | 快 | 慢（10x-100x） |
| **适用场景** | 事实性问答 | 全局分析、跨文档推理、主题概述 |

### Microsoft GraphRAG 实践要点

- **成本考量**：索引阶段需要对每个 chunk 调用 LLM 做实体/关系抽取，数千页文档可能消耗数十美元 API 费用
- **分辨率调优**：Leiden 算法的 `resolution_parameter` 控制社区粒度，值越大社区越细
- **增量更新**：新增文档需要重新抽取实体、合并图、重新聚类——目前增量更新成本较高
- **混合使用**：实践中常将 GraphRAG 的 Global Search 与传统 RAG 的 Local Search 结合

## RAG 评估体系

### 检索评估指标

检索质量是 RAG 系统的基石。以下是核心评估指标：

#### Context Precision（上下文精确率）

检索到的 K 个文档中，有多少是真正相关的？

$$\text{Context Precision} = \frac{\text{检索到的相关文档数}}{\text{检索到的总文档数 K}}$$

#### Context Recall（上下文召回率）

回答问题所需的所有信息，有多少被检索到了？

$$\text{Context Recall} = \frac{\text{检索到的相关信息条目数}}{\text{标注答案中的总信息条目数}}$$

#### MRR（Mean Reciprocal Rank）

第一个正确结果出现在排名的多高位置？

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

如果第一个相关文档排在第 1 位，得分 1；排在第 3 位，得分 1/3。MRR 反映的是"用户不需要翻几页就能找到答案"。

#### NDCG@K（Normalized Discounted Cumulative Gain）

同时考虑相关性和排名位置的综合指标：

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

其中 IDCG 是理想排列下的 DCG。NDCG 惩罚将高相关文档排在低位的情况。

### 生成评估指标

#### Faithfulness（忠实度）

生成的答案是否忠实于检索到的上下文？是否存在"编造"了上下文中没有的信息？

```
评估方法：
1. 将答案拆分为多个独立的事实性陈述（claims）
2. 逐一判断每个 claim 是否能在检索到的上下文中找到支撑
3. Faithfulness = 有支撑的 claims 数 / 总 claims 数
```

#### Answer Relevancy（答案相关性）

生成的答案是否切中了用户的问题？

```
评估方法（Ragas 实现）：
1. 从答案出发，让 LLM 逆向生成 N 个可能对应的问题
2. 计算这 N 个生成问题与原始问题的平均余弦相似度
3. 相似度越高，说明答案与问题越相关
```

#### Hallucination Rate（幻觉率）

答案中有多少内容是检索上下文中不存在的"编造"信息？

$$\text{Hallucination Rate} = 1 - \text{Faithfulness}$$

### 端到端评估：Ragas 框架

[Ragas](https://github.com/explodinggradients/ragas) 是目前最流行的 RAG 评估框架，支持无需人工标注的自动化评估。

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# 准备评估数据
eval_data = {
    "question": [
        "什么是 Transformer？",
        "BERT 和 GPT 有什么区别？"
    ],
    "answer": [
        "Transformer 是一种基于自注意力机制的神经网络架构，由 Vaswani 等人在 2017 年提出。",
        "BERT 使用双向编码器，适合理解任务；GPT 使用单向解码器，适合生成任务。"
    ],
    "contexts": [
        ["Transformer 架构由 Vaswani 等人在 2017 年提出，核心创新是自注意力机制。"],
        ["BERT 是双向编码器模型。", "GPT 采用自回归解码器架构。"]
    ],
    "ground_truth": [
        "Transformer 是 Vaswani 等人 2017 年提出的基于自注意力机制的架构。",
        "BERT 是双向编码器用于理解，GPT 是自回归解码器用于生成。"
    ]
}

dataset = Dataset.from_dict(eval_data)

# 执行评估
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print(results)
# {'faithfulness': 0.95, 'answer_relevancy': 0.88, 
#  'context_precision': 0.90, 'context_recall': 0.85}
```

### DeepEval 评估框架

[DeepEval](https://github.com/confident-ai/deepeval) 提供了更丰富的评估指标和更友好的测试体验：

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    HallucinationMetric
)

# 定义测试用例
test_case = LLMTestCase(
    input="LoRA 的核心思想是什么？",
    actual_output="LoRA 通过低秩矩阵分解来微调大模型，只需训练少量参数。",
    retrieval_context=[
        "LoRA 的核心思想是冻结预训练权重，仅训练低秩分解矩阵 A 和 B。",
        "LoRA 可以将可训练参数量减少到原始模型的 0.1% 以下。"
    ]
)

# 定义指标
faithfulness = FaithfulnessMetric(threshold=0.7)
relevancy = AnswerRelevancyMetric(threshold=0.7)

# 评估
faithfulness.measure(test_case)
print(f"Faithfulness: {faithfulness.score}")  # 0.92
print(f"Reason: {faithfulness.reason}")       # 详细的评估解释
```

### 评估指标选择指南

| 你关心的问题 | 使用指标 | 需要标注数据？ |
|-------------|---------|-------------|
| 检索到的文档准不准？ | Context Precision | 否（LLM 判断） |
| 该检索的信息都找到了吗？ | Context Recall | 是（需要 ground truth） |
| 答案是否忠实于检索结果？ | Faithfulness | 否 |
| 答案是否切中问题？ | Answer Relevancy | 否 |
| 答案有没有编造信息？ | Hallucination Rate | 否 |
| 检索排序质量如何？ | MRR, NDCG | 是（需要相关性标注） |

## RAG 常见问题与优化

### PDF 解析难题

PDF 是企业文档最常见的格式，但也是 RAG 最头疼的数据源。

**核心挑战：**
- **复杂布局**：多栏排版、页眉页脚、脚注、侧边栏
- **表格识别**：跨页表格、合并单元格、嵌套表格
- **图片和公式**：扫描件 PDF（需要 OCR）、LaTeX 公式
- **元数据丢失**：标题层级、列表结构在 PDF 中不是语义标签

**解决方案分级：**

| 方案 | 工具 | 适用场景 | 精度 |
|------|------|---------|------|
| **基础文本提取** | PyPDF2, pdfplumber | 文字型 PDF，简单布局 | 中 |
| **布局分析** | Unstructured, DocLayNet | 复杂布局，需要保留结构 | 较高 |
| **OCR + 布局** | PaddleOCR, Tesseract | 扫描件 PDF | 中 |
| **多模态解析** | GPT-4V, 通义千问-VL | 图表、公式混排 | 高（但贵） |

```python
# 使用 Unstructured 进行结构化 PDF 解析
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="report.pdf",
    strategy="hi_res",            # 高精度模式（使用布局检测模型）
    infer_table_structure=True,   # 推断表格结构
    languages=["chi_sim", "eng"] # 中英文
)

# elements 包含带有类型标签的结构化元素
for el in elements:
    print(f"[{el.category}] {el.text[:50]}...")
    # [Title] 2024年度财务报告...
    # [NarrativeText] 本报告期内，公司实现营业收入...
    # [Table] | 项目 | Q1 | Q2 | Q3 | Q4 |...
```

### 长文本切分策略选择

不同类型的文档需要不同的切分策略：

| 文档类型 | 推荐策略 | 关键考量 |
|---------|---------|---------|
| **技术文档** | 按标题层级（Markdown Header）切分 | 保持章节完整性 |
| **法律合同** | 按条款切分 | 条款是最小独立语义单元 |
| **代码仓库** | 按函数/类切分 | 保持代码块完整 |
| **对话记录** | 按对话轮次切分 | 保持问答配对 |
| **学术论文** | 按章节 + 段落切分 | 摘要、引言、方法分别处理 |
| **表格数据** | 行级切分 + 保留表头 | 每个 chunk 必须包含列名 |

```python
# 针对 Markdown 文档的层级切分
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)
chunks = splitter.split_text(markdown_text)
# 每个 chunk 都带有层级元数据，如 {"h1": "引言", "h2": "背景"}
```

### 负样本挖掘提升检索精度

当检索结果中混入大量"看起来相关但实际无关"的文档时，可以通过负样本挖掘（Hard Negative Mining）来微调 Embedding 模型。

```python
def mine_hard_negatives(query: str, positive_doc: str, all_docs: list, 
                         model, top_k=10):
    """
    挖掘困难负样本：找到与 query 相似度高但实际不相关的文档
    这些"容易混淆"的样本对训练 Embedding 模型最有价值
    """
    query_emb = model.encode(query)
    all_embs = model.encode(all_docs)
    
    # 计算相似度并排序
    similarities = np.dot(all_embs, query_emb)
    ranked_indices = np.argsort(similarities)[::-1]
    
    hard_negatives = []
    for idx in ranked_indices[:top_k]:
        if all_docs[idx] != positive_doc:  # 排除正样本
            hard_negatives.append(all_docs[idx])
    
    return hard_negatives

# 训练三元组：(query, positive_doc, hard_negative_doc)
# 目标：让模型学会区分"看起来像但不是"的文档
```

### RAG-Fusion（多查询 + RRF 排序）

RAG-Fusion 通过生成多个查询变体，分别检索后用 RRF（Reciprocal Rank Fusion）合并排序，显著提升召回率。

```python
def rag_fusion(original_query: str, llm, retriever, num_variants=4):
    """
    RAG-Fusion 完整流程：
    1. 从原始查询生成多个变体
    2. 每个变体独立检索
    3. 用 RRF 合并排序
    """
    # 第一步：生成查询变体
    variant_prompt = f"""请将以下问题从不同角度重新表述为 {num_variants} 个查询：
原始问题：{original_query}
请输出 JSON 数组格式的查询列表："""
    
    variants = json.loads(llm.generate(variant_prompt))
    all_queries = [original_query] + variants
    
    # 第二步：每个查询独立检索
    all_results = {}  # {doc_id: {query_idx: rank}}
    for q_idx, query in enumerate(all_queries):
        results = retriever.search(query, top_k=20)
        for rank, doc in enumerate(results):
            if doc.id not in all_results:
                all_results[doc.id] = {}
            all_results[doc.id][q_idx] = rank + 1
    
    # 第三步：RRF 融合排序
    k = 60  # RRF 常数，通常取 60
    rrf_scores = {}
    for doc_id, ranks in all_results.items():
        rrf_scores[doc_id] = sum(1.0 / (k + r) for r in ranks.values())
    
    # 按 RRF 分数排序，取 top 结果
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:10]
```

**为什么 RRF 有效？**
- 如果一个文档在多个查询变体的检索中都排名靠前，它极有可能是真正相关的
- RRF 公式 $\frac{1}{k + rank}$ 平滑了不同检索器分数量纲的差异
- 实验证明 RRF 在大多数场景下优于简单的分数加权融合

### 多模态 RAG（图片 + 文本混合检索）

当文档包含图表、流程图、产品图片等视觉信息时，纯文本 RAG 会丢失关键信息。

```
多模态 RAG 架构：

文档集合
├── 文本内容 → Text Embedding → 文本向量库
├── 图片/图表 → 两种策略：
│   ├── 策略 A：VLM 生成图片描述 → Text Embedding → 文本向量库
│   └── 策略 B：CLIP Embedding → 图片向量库
└── 表格 → 结构化提取 → Text Embedding → 文本向量库

查询时：
用户问题 → Text Embedding → 检索文本 + 检索图片描述/图片 → 多模态 LLM 生成答案
```

```python
class MultimodalRAG:
    """支持图文混合检索的 RAG 系统"""
    
    def __init__(self, text_model, vision_model, vlm):
        self.text_model = text_model    # 文本 Embedding 模型
        self.vision_model = vision_model # CLIP 等视觉模型
        self.vlm = vlm                  # 视觉语言模型（GPT-4V 等）
        self.text_store = []
        self.image_store = []
    
    def index_document(self, text_chunks: list, images: list):
        """索引文本和图片"""
        # 文本正常索引
        for chunk in text_chunks:
            emb = self.text_model.encode(chunk)
            self.text_store.append({"text": chunk, "embedding": emb})
        
        # 图片：用 VLM 生成描述，再索引描述文本
        for img in images:
            description = self.vlm.describe(img)  # "这是一张展示 Transformer 架构的图..."
            emb = self.text_model.encode(description)
            self.image_store.append({
                "image": img, 
                "description": description, 
                "embedding": emb
            })
    
    def retrieve(self, query: str, top_k=5):
        """混合检索文本和图片"""
        query_emb = self.text_model.encode(query)
        
        # 检索文本
        text_results = self._search(self.text_store, query_emb, top_k)
        # 检索图片（通过描述文本）
        image_results = self._search(self.image_store, query_emb, top_k)
        
        # 合并结果
        return {"texts": text_results, "images": image_results}
```

**实践建议：**
- 图片描述策略（策略 A）实现简单，但依赖 VLM 的描述质量
- CLIP 直接检索（策略 B）不丢信息，但跨模态对齐质量不如文本-文本检索
- 生产系统通常两种策略并用，互为补充

## 常见问题 & 面试考点

::: tip 面试高频问题

**Q: RAG 中 Embedding 模型和生成模型的关系是什么？**
A: 两者完全独立。Embedding 模型（如 BGE）负责将文本编码为向量用于检索；生成模型（如 GPT-4）负责基于检索结果生成答案。它们可以分别替换和升级，这是 RAG 架构灵活性的来源。

**Q: 为什么向量检索用余弦相似度而不是欧氏距离？**
A: 余弦相似度衡量方向（语义方向），忽略长度（与文本长度无关）。两段语义相同但长度不同的文本，欧氏距离可能很大，但余弦相似度仍然很高。如果向量已归一化，余弦相似度等价于内积，计算更快。

**Q: RAG 的知识如何更新？**
A: 增量更新——新文档编码后加入向量数据库即可。无需重新训练模型，无需重建整个索引（大多数向量数据库支持增量插入）。这是 RAG 相对 Fine-tuning 的核心优势。

**Q: 如何处理 RAG 中的"检索到了但 LLM 没用上"的问题？**
A: 这通常是因为检索结果排列在 prompt 末尾被 LLM 忽视（"lost in the middle" 现象）。解决方案：(1) 减少检索数量（top-3 而非 top-10）；(2) 对检索结果做摘要压缩；(3) 将最相关的文档放在 prompt 开头和结尾。
:::


## 推荐资源

- **Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"** — RAG 的原始论文
- **Gao et al. "Retrieval-Augmented Generation for Large Language Models: A Survey"** — 最全面的 RAG 综述
- **Asai et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"** — 自反思 RAG
- **Edge et al. "From Local to Global: A Graph RAG Approach"** — 微软 Graph RAG
- **LangChain / LlamaIndex 官方文档** — RAG 框架的最佳实践
- **[MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)** — Embedding 模型评测排行榜
- **[RAGAS 框架](https://github.com/explodinggradients/ragas)** — RAG 系统评估工具
