---
title: "端到端实战项目"
description: "从零搭建 RAG 应用和 LoRA 微调 pipeline，完整可运行代码"
topics: [RAG, LoRA, fine-tuning, hands-on, project, end-to-end, FAISS, embedding]
prereqs: [applications/rag, training/sft]
---
# 端到端实战项目

> **一句话总结:** 看再多理论不如跑一遍完整 pipeline——本章提供两个从零开始的实战项目（RAG 应用 + LoRA 微调），每个都能在单卡 GPU 上运行，带你走完"从数据到部署"的全流程。

## 项目一：从零搭建 RAG 应用

### 目标

构建一个基于本地文档的问答系统：用户提出问题 → 检索相关文档片段 → LLM 基于检索结果生成回答。整个 pipeline 约 150 行 Python，不依赖 LangChain 等框架。

### 整体架构

```
          Offline Stage                         Online Stage
┌───────────────────────────┐    ┌──────────────────────────────────┐
│ Documents                 │    │ User Query                       │
│   ↓                       │    │   ↓                              │
│ Chunk (split by paragraph)│    │ Embed query                      │
│   ↓                       │    │   ↓                              │
│ Embed (sentence-transformers) │ │ FAISS similarity search          │
│   ↓                       │    │   ↓                              │
│ Store in FAISS index      │    │ Top-k relevant chunks            │
└───────────────────────────┘    │   ↓                              │
                                 │ Prompt = query + context chunks  │
                                 │   ↓                              │
                                 │ LLM generates answer             │
                                 └──────────────────────────────────┘
```

### 第一步：文档分块

```python
def chunk_documents(documents: list[str], chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    将文档列表切分为固定大小的文本块
    
    Args:
        documents: 原始文档列表
        chunk_size: 每个 chunk 的最大字符数
        overlap: 相邻 chunk 之间的重叠字符数（保证语义连续性）
    
    Returns:
        切分后的 chunk 列表
    """
    chunks = []
    for doc in documents:
        # 先按段落切分
        paragraphs = doc.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # 带 overlap：保留上一个 chunk 的末尾
                if overlap > 0 and current_chunk:
                    current_chunk = current_chunk[-overlap:] + para + "\n\n"
                else:
                    current_chunk = para + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
    return chunks

# 示例：加载文档
documents = [
    open(f).read() for f in ["doc1.txt", "doc2.txt", "doc3.txt"]
]
chunks = chunk_documents(documents, chunk_size=500, overlap=50)
print(f"文档数: {len(documents)}, 切分后 chunk 数: {len(chunks)}")
```

::: tip 为什么需要 overlap？
如果一个关键信息恰好被切分到两个 chunk 的边界，没有 overlap 就可能丢失上下文。50-100 字符的 overlap 是常见做法。
:::

### 第二步：向量化与索引

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 加载 Embedding 模型
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")  # 中文推荐
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")      # 英文推荐

def build_index(chunks: list[str], embed_model) -> tuple[faiss.Index, np.ndarray]:
    """
    将文本块向量化并构建 FAISS 索引
    
    Returns:
        index: FAISS 索引对象
        embeddings: 向量矩阵 [n_chunks, dim]
    """
    # 批量编码（比逐条编码快很多）
    embeddings = embed_model.encode(
        chunks, 
        batch_size=32, 
        show_progress_bar=True,
        normalize_embeddings=True  # L2 归一化，使余弦相似度 = 内积
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # 构建 FAISS 索引（内积 = 余弦相似度，因为已归一化）
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP = Inner Product
    index.add(embeddings)
    
    print(f"索引构建完成: {index.ntotal} vectors, dim={dim}")
    return index, embeddings

index, embeddings = build_index(chunks, embed_model)
```

### 第三步：检索

```python
def retrieve(query: str, index: faiss.Index, chunks: list[str], 
             embed_model, top_k: int = 3) -> list[tuple[str, float]]:
    """
    检索与 query 最相关的 top_k 个文档块
    
    Returns:
        [(chunk_text, similarity_score), ...]
    """
    # 编码查询
    query_vec = embed_model.encode(
        [query], normalize_embeddings=True
    ).astype(np.float32)
    
    # FAISS 检索
    scores, indices = index.search(query_vec, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append((chunks[idx], float(score)))
    
    return results

# 测试检索
results = retrieve("什么是 Attention 机制？", index, chunks, embed_model, top_k=3)
for i, (chunk, score) in enumerate(results):
    print(f"[{i+1}] Score: {score:.4f}")
    print(f"    {chunk[:100]}...")
```

### 第四步：生成回答

```python
from openai import OpenAI

# 可以用任何兼容 OpenAI API 的服务（本地 vLLM、Ollama 等）
client = OpenAI(
    base_url="http://localhost:8000/v1",  # 本地 vLLM 服务
    api_key="not-needed"
)

def generate_answer(query: str, context_chunks: list[tuple[str, float]], 
                    model: str = "Qwen/Qwen2.5-7B-Instruct") -> str:
    """基于检索结果生成回答"""
    
    # 构建 prompt
    context = "\n\n---\n\n".join([chunk for chunk, _ in context_chunks])
    
    prompt = f"""请基于以下参考资料回答用户的问题。如果参考资料中没有相关信息，请如实说明。

## 参考资料

{context}

## 用户问题

{query}

## 回答"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个准确、简洁的问答助手。只基于提供的参考资料回答问题。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # 低温度保证事实性
        max_tokens=512,
    )
    
    return response.choices[0].message.content

# 完整 pipeline
query = "Flash Attention 的核心优化思想是什么？"
results = retrieve(query, index, chunks, embed_model, top_k=3)
answer = generate_answer(query, results)
print(f"Q: {query}")
print(f"A: {answer}")
```

### 第五步：整合为完整类

```python
class SimpleRAG:
    """极简 RAG 系统，约 100 行核心代码"""
    
    def __init__(self, embed_model_name="BAAI/bge-small-zh-v1.5",
                 llm_base_url="http://localhost:8000/v1",
                 llm_model="Qwen/Qwen2.5-7B-Instruct"):
        self.embed_model = SentenceTransformer(embed_model_name)
        self.client = OpenAI(base_url=llm_base_url, api_key="not-needed")
        self.llm_model = llm_model
        self.chunks = []
        self.index = None
    
    def add_documents(self, documents: list[str], chunk_size=500, overlap=50):
        """添加文档到知识库"""
        self.chunks = chunk_documents(documents, chunk_size, overlap)
        self.index, _ = build_index(self.chunks, self.embed_model)
        print(f"知识库已更新: {len(self.chunks)} chunks")
    
    def query(self, question: str, top_k: int = 3) -> str:
        """提问并获取回答"""
        results = retrieve(question, self.index, self.chunks, 
                          self.embed_model, top_k)
        answer = generate_answer(question, results, self.llm_model)
        return answer

# 使用
rag = SimpleRAG()
rag.add_documents([open("my_notes.txt").read()])
print(rag.query("什么是 KV Cache？"))
```

::: tip 延伸挑战
1. **添加重排序**：检索 top_20 后用 Cross-Encoder 重排，取 top_3 送给 LLM
2. **添加引用溯源**：在回答中标注每句话引用了哪个 chunk
3. **持久化索引**：用 `faiss.write_index()` 保存索引，避免每次重建
4. **对比实验**：换不同的 Embedding 模型和 chunk_size，观察回答质量变化
:::

---

## 项目二：LoRA 微调实战

### 目标

用 LoRA 微调一个 Qwen2.5-1.5B 模型完成特定任务（如中文指令遵循），在单张 16GB GPU 上即可运行。代码基于 HuggingFace PEFT + TRL，约 80 行核心代码。

### LoRA 原理回顾

LoRA 冻结原始模型权重 $W$，只训练低秩增量 $\Delta W = BA$：

$$
h = Wx + \Delta Wx = Wx + BAx, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}
$$

其中 $r \ll d$（如 $r=16, d=2048$），可训练参数量仅为原始的 $\frac{2r}{d} \approx 1.6\%$。

### 第一步：准备环境和数据

```bash
pip install torch transformers peft trl datasets accelerate
```

```python
from datasets import load_dataset

# 加载开源指令数据集（中文）
dataset = load_dataset("shibing624/alpaca-zh", split="train")

# 查看数据格式
print(dataset[0])
# {'instruction': '保持健康的三个提示。', 
#  'input': '', 
#  'output': '1. 保持良好的饮食习惯...'}

# 格式化为 Chat Template
def format_example(example):
    """将数据转换为对话格式"""
    if example["input"]:
        user_msg = f"{example['instruction']}\n\n{example['input']}"
    else:
        user_msg = example["instruction"]
    
    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": example["output"]}
        ]
    }

dataset = dataset.map(format_example)
# 取子集用于快速实验
dataset = dataset.select(range(min(5000, len(dataset))))
print(f"训练样本数: {len(dataset)}")
```

### 第二步：加载模型和 LoRA 配置

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                      # 秩（越大容量越强，显存越大）
    lora_alpha=32,             # 缩放系数（通常设为 2*r）
    lora_dropout=0.05,         # Dropout 防过拟合
    target_modules=[           # 要插入 LoRA 的层
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 12,582,912 || all params: 1,555,467,264 || trainable%: 0.81%
```

::: tip target_modules 选择
- **最小配置**：只对 `q_proj, v_proj` 加 LoRA（参数量最少）
- **标准配置**：对 QKV + O 全加（上面的配置）
- **最大配置**：连 FFN 的 `gate_proj, up_proj, down_proj` 也加（效果最好但参数最多）

一般推荐标准配置，兼顾效果和效率。
:::

### 第三步：训练

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,    # 等效 batch_size = 4 * 4 = 16
    learning_rate=2e-4,               # LoRA 学习率通常比全参数微调高
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,                        # BF16 混合精度
    logging_steps=10,
    save_strategy="epoch",
    max_seq_length=1024,
    gradient_checkpointing=True,      # 省显存
    report_to="none",                 # 不上报到 wandb
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# 开始训练
trainer.train()

# 保存 LoRA 权重（只有几十 MB）
trainer.save_model("./lora-output/final")
print("训练完成！LoRA 权重已保存。")
```

### 第四步：推理验证

```python
from peft import PeftModel

# 方法 1：直接用训练好的模型推理
model.eval()

messages = [{"role": "user", "content": "请解释什么是注意力机制？"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print(f"Q: 请解释什么是注意力机制？")
print(f"A: {response}")

# 方法 2：从保存的权重加载
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
lora_model = PeftModel.from_pretrained(base_model, "./lora-output/final")
lora_model.eval()

# 方法 3：合并 LoRA 权重到基础模型（部署时推荐，无额外推理开销）
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
```

### 第五步：效果评估

```python
# 简单的人工评估框架
test_prompts = [
    "用简单的语言解释什么是深度学习。",
    "写一个 Python 函数计算斐波那契数列。",
    "总结一下 Transformer 架构的核心创新点。",
    "如何评估一个语言模型的质量？",
]

print("=== 微调前后对比 ===\n")
for prompt in test_prompts:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"Q: {prompt}")
    print(f"A: {response[:200]}...")
    print("-" * 60)
```

::: tip 延伸挑战
1. **对比不同 rank**：分别用 r=4, 16, 64 训练，比较效果和显存占用
2. **QLoRA**：加上 4-bit 量化（`load_in_4bit=True`），在 8GB GPU 上也能跑
3. **DPO 对齐**：在 LoRA 基础上加 DPO 训练，用 TRL 的 `DPOTrainer`
4. **自定义数据**：用自己的领域数据（如法律、医学）微调，观察领域适配效果
:::

---

## 环境要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| RAG 应用 | 8GB RAM, CPU | 16GB RAM, GPU（加速 Embedding） |
| LoRA 微调 | 16GB VRAM (如 T4) | 24GB VRAM (如 A10/4090) |

```bash
# 安装所有依赖
pip install torch transformers peft trl datasets accelerate \
    sentence-transformers faiss-cpu openai
```

---

## 苏格拉底时刻

1. RAG 中 chunk_size 设得太大或太小分别有什么问题？如何找到最优值？
2. 如果检索到的 chunk 和问题完全不相关，LLM 会怎么做？如何让系统在"不知道"时正确拒绝回答？
3. LoRA 的 rank r 越大效果越好吗？它和全参数微调的关系是什么？
4. 为什么 LoRA 的学习率（2e-4）通常比全参数微调的学习率（2e-5）高一个数量级？
5. `merge_and_unload()` 合并后的模型和原始 LoRA 模型的推理结果完全一致吗？为什么？

---

## 推荐资源

- **LlamaIndex / LangChain** — 工业级 RAG 框架，功能更完整但复杂度更高
- **PEFT 官方文档** — LoRA、QLoRA、Prefix Tuning 等参数高效微调方法
- **TRL 库** — HuggingFace 的训练库，支持 SFT、DPO、PPO、GRPO
- **FAISS 官方文档** — Facebook 的向量检索库，支持十亿级向量的高效搜索
- **MTEB Leaderboard** — Embedding 模型排行榜，帮助选择最适合的 Embedding 模型
