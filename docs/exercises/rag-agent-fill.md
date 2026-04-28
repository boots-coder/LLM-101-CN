---
title: "RAG 与 Agent 填空"
description: "Level 2-3 填空：文本分块、向量检索、ReAct Agent、Tool Calling"
topics: [fill-in, RAG, agent, chunking, embedding, retrieval, ReAct, tool-calling]
---
# RAG 与 Agent 代码填空 (Level 2-3)

> 本练习覆盖 RAG 与 Agent 核心技术：文本分块、余弦相似度检索、RAG Pipeline、ReAct Agent、Tool Calling。
> 纯 Python 实现，用 mock 替代真实 LLM API，可完全离线运行。

---

## 练习 1: 文本分块策略（Level 2）

### 背景

RAG 第一步是将长文档切分成适合 embedding 的 chunk。chunk 太大引入噪声，太小丢失上下文。本题实现固定大小分块（带 overlap）和按 Markdown 标题的语义分块。

### 任务

```python
import re
from typing import List

def fixed_size_chunk(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """固定大小分块，相邻 chunk 有 overlap 个字符重叠。"""
    assert overlap < chunk_size
    chunks = []
    # ===== 填空 1: 计算滑动窗口步长 =====
    step = _____  # 提示: chunk_size - overlap

    # ===== 填空 2: 滑动窗口遍历 =====
    for start in _____:  # 提示: range(0, len(text), step)
        chunk = text[start:start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def semantic_chunk_by_heading(text: str) -> List[dict]:
    """按 Markdown 标题（## / ###）切分，返回 [{"heading": ..., "content": ...}]"""
    chunks = []
    # ===== 填空 3: 按标题行切分，保留分隔符 =====
    parts = re.split(r'(^#{2,3}\s+.+)$', text, flags=_____) # 提示: re.MULTILINE

    current_heading = "(无标题)"
    current_parts = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # ===== 填空 4: 判断是否是标题行 =====
        if re.match(r'_____', part):  # 提示: ^#{2,3}\s+
            if current_parts:
                chunks.append({"heading": current_heading,
                                "content": "\n".join(current_parts)})
                current_parts = []
            current_heading = part
        else:
            current_parts.append(part)
    if current_parts:
        chunks.append({"heading": current_heading,
                        "content": "\n".join(current_parts)})
    return chunks
```

### 提示

- 步长 = `chunk_size - overlap`，保证尾部和下一 chunk 头部重叠 `overlap` 字符
- `re.split` 加捕获组 `()` 保留分隔符，`re.MULTILINE` 让 `^` 匹配每行开头

<details>
<summary>参考答案</summary>

```python
# 填空 1
step = chunk_size - overlap
# 填空 2
for start in range(0, len(text), step):
# 填空 3
parts = re.split(r'(^#{2,3}\s+.+)$', text, flags=re.MULTILINE)
# 填空 4
if re.match(r'^#{2,3}\s+', part):
```

**验证:**
```python
text = "A" * 500
chunks = fixed_size_chunk(text, chunk_size=200, overlap=50)
for i in range(len(chunks) - 1):
    assert chunks[i][-50:] == chunks[i+1][:50], f"chunk {i} overlap 不正确"
print(f"固定分块: {len(chunks)} chunks, overlap 验证通过")

md = "## 第一章\n内容一\n### 1.1 小节\n内容二\n## 第二章\n内容三"
sc = semantic_chunk_by_heading(md)
print(f"语义分块: {len(sc)} chunks")
for c in sc:
    print(f"  [{c['heading']}] {c['content'][:30]}")
```

</details>

---

## 练习 2: 余弦相似度与向量检索（Level 2）

### 背景

给定 query embedding 和 document embeddings，用余弦相似度检索 top-k 文档：$\text{sim}(a,b) = \frac{a \cdot b}{\|a\| \|b\|}$

### 任务

```python
import math
from typing import List, Tuple

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算余弦相似度"""
    # ===== 填空 1: 点积 =====
    dot = _____   # 提示: sum(x*y for x,y in zip(a,b))
    # ===== 填空 2: L2 范数 =====
    na = _____    # 提示: math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    if na == 0 or nb == 0:
        return 0.0
    # ===== 填空 3: 余弦相似度 =====
    return _____  # 提示: dot / (na * nb)


def topk_retrieval(query_emb, doc_embs, doc_texts, k=3):
    """检索 top-k 最相似文档，返回 [(index, score, text), ...]"""
    scores = []
    for i, emb in enumerate(doc_embs):
        # ===== 填空 4: 计算相似度 =====
        sim = _____  # 提示: cosine_similarity(query_emb, emb)
        scores.append((i, sim, doc_texts[i]))
    # ===== 填空 5: 按相似度降序取 top-k =====
    scores.sort(key=lambda x: _____, reverse=True)  # 提示: x[1]
    return scores[:k]


class SimpleInvertedIndex:
    """倒排索引，用于向量检索前的粗筛"""
    def __init__(self):
        self.index = {}  # word -> set of doc_ids

    def add_document(self, doc_id: int, text: str):
        for word in text.lower().split():
            # ===== 填空 6: 建立倒排映射 =====
            if word not in self.index:
                self.index[word] = _____  # 提示: set()
            _____  # 提示: self.index[word].add(doc_id)

    def search(self, query: str) -> List[int]:
        # ===== 填空 7: 合并匹配文档集合 =====
        result = _____  # 提示: set()
        for w in query.lower().split():
            if w in self.index:
                result = _____  # 提示: result | self.index[w]
        return sorted(result)
```

### 提示

- 余弦相似度范围 [-1, 1]，注意零向量边界
- 倒排索引在向量检索前做粗筛可大幅减少计算量

<details>
<summary>参考答案</summary>

```python
# 填空 1-3
dot = sum(x * y for x, y in zip(a, b))
na = math.sqrt(sum(x ** 2 for x in a))
return dot / (na * nb)
# 填空 4-5
sim = cosine_similarity(query_emb, emb)
scores.sort(key=lambda x: x[1], reverse=True)
# 填空 6-7
self.index[word] = set()
self.index[word].add(doc_id)
result = set()
result = result | self.index[w]
```

**验证:**
```python
import random; random.seed(42)
dim = 64
docs = ["Transformer 注意力机制", "BERT 预训练模型", "CNN 图像识别",
        "RNN 时序数据", "注意力机制与序列建模"]
embs = [[random.gauss(0,1) for _ in range(dim)] for _ in docs]
query = [x + random.gauss(0,0.1) for x in embs[0]]  # 接近 doc 0

results = topk_retrieval(query, embs, docs, k=3)
print("Top-3:", [(i, f"{s:.3f}", t) for i,s,t in results])
assert results[0][0] == 0, "排序错误"
print(f"自身相似度: {cosine_similarity(embs[0], embs[0]):.6f}")

idx = SimpleInvertedIndex()
for i, t in enumerate(docs):
    idx.add_document(i, t)
print(f"倒排索引 '注意力': {idx.search('注意力')}")
```

</details>

---

## 练习 3: RAG Pipeline 端到端（Level 3）

### 背景

完整 RAG 流程：分块 -> embedding -> 检索 -> prompt 组装 -> 生成。本题将前两练的组件串联成 pipeline，用 mock LLM 替代 API。

### 任务

```python
import random, math
from typing import List, Dict

def mock_embedding(text: str, dim: int = 32) -> List[float]:
    """用哈希生成伪 embedding，相同文本返回相同向量"""
    random.seed(hash(text) % (2**31))
    return [random.gauss(0, 1) for _ in range(dim)]

def mock_llm(prompt: str) -> str:
    if "[上下文]" in prompt:
        s = prompt.index("[上下文]") + 6
        e = prompt.index("[问题]") if "[问题]" in prompt else len(prompt)
        return f"根据资料，{prompt[s:e].strip()[:80]}..."
    return "无法回答。"

class SimpleRAG:
    def __init__(self, chunk_size=200, overlap=50, top_k=3,
                 max_context_length=500):
        self.chunk_size, self.overlap = chunk_size, overlap
        self.top_k = top_k
        self.max_context_length = max_context_length
        self.chunks, self.embeddings, self.sources = [], [], []

    def add_document(self, text: str, source: str = "unknown"):
        step = self.chunk_size - self.overlap
        for start in range(0, len(text), step):
            chunk = text[start:start + self.chunk_size]
            if chunk.strip():
                self.chunks.append(chunk)
                self.embeddings.append(mock_embedding(chunk))
                self.sources.append(source)

    def retrieve(self, query: str) -> List[Dict]:
        qe = mock_embedding(query)
        scored = []
        for i, emb in enumerate(self.embeddings):
            dot = sum(a*b for a,b in zip(qe, emb))
            na = math.sqrt(sum(a**2 for a in qe))
            nb = math.sqrt(sum(b**2 for b in emb))
            sim = dot/(na*nb) if na and nb else 0
            scored.append({"index": i, "score": sim,
                           "text": self.chunks[i], "source": self.sources[i]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:self.top_k]

    def build_prompt(self, query: str, retrieved: List[Dict]) -> str:
        ctx_parts, total = [], 0
        for item in retrieved:
            # ===== 填空 1: 添加来源标注 =====
            piece = f"[来源: {_____}]\n{item['text']}"  # 提示: item["source"]
            # ===== 填空 2: max_context_length 截断 =====
            if total + len(piece) > _____:  # 提示: self.max_context_length
                break
            ctx_parts.append(piece)
            total += len(piece)
        context = "\n\n".join(ctx_parts)
        # ===== 填空 3: 组装 prompt =====
        prompt = _____
        # 提示: f"请根据上下文回答。\n\n[上下文]\n{context}\n\n[问题]\n{query}"
        return prompt

    def query(self, question: str) -> Dict:
        retrieved = self.retrieve(question)
        prompt = self.build_prompt(question, retrieved)
        return {"answer": mock_llm(prompt), "sources": [r["source"] for r in retrieved],
                "prompt": prompt}
```

### 提示

- 来源标注让用户追溯答案出处，是 RAG 的关键特性
- `max_context_length` 防止超出 LLM 上下文窗口

<details>
<summary>参考答案</summary>

```python
# 填空 1
piece = f"[来源: {item['source']}]\n{item['text']}"
# 填空 2
if total + len(piece) > self.max_context_length:
# 填空 3
prompt = f"请根据上下文回答。\n\n[上下文]\n{context}\n\n[问题]\n{query}"
```

**验证:**
```python
rag = SimpleRAG(chunk_size=100, overlap=20, top_k=3, max_context_length=400)
rag.add_document("Transformer 由 Vaswani 在 2017 年提出，基于自注意力机制。"
                 "它抛弃了 RNN 和 CNN，大幅提升训练效率。", source="transformer")
rag.add_document("BERT 由 Google 在 2018 年推出，使用 MLM 和 NSP 预训练。", source="bert")

r = rag.query("什么是 Transformer?")
print(f"回答: {r['answer']}")
print(f"来源: {r['sources']}")
assert "[上下文]" in r["prompt"] and "[问题]" in r["prompt"]
assert "来源:" in r["prompt"]
print("RAG Pipeline 验证通过")
```

</details>

---

## 练习 4: ReAct Agent 循环（Level 2-3）

### 背景

ReAct = Reason + Act，Agent 在 Thought -> Action -> Observation 循环中解决问题，直到输出 Final Answer。

### 任务

```python
import re
from typing import Dict, Callable

# 三个 mock 工具
def calculator(expr: str) -> str:
    try:
        return str(eval(expr))  # 生产环境应用 ast.literal_eval
    except Exception as e:
        return f"错误: {e}"

def search(query: str) -> str:
    db = {"中国人口": "约 14.1 亿", "地球月球距离": "约 384400 公里"}
    for k, v in db.items():
        if k in query:
            return v
    return f"未找到 '{query}' 的结果"

def get_weather(city: str) -> str:
    db = {"北京": "晴 25度", "上海": "多云 22度", "深圳": "小雨 28度"}
    return db.get(city, f"无 {city} 天气数据")

# Mock LLM: 预设多步 ReAct 输出
def mock_react_llm(history: str, step: int) -> str:
    if step == 0:
        return ("Thought: 先查天气。\n"
                "Action: get_weather\nAction Input: 北京")
    elif step == 1:
        return ("Thought: 天气已知，算数学题。\n"
                "Action: calculator\nAction Input: 25 * 3 + 10")
    else:
        return ("Thought: 信息齐全。\n"
                "Final Answer: 北京晴天 25 度，25*3+10=85。")


class ReActAgent:
    def __init__(self, tools: Dict[str, Callable], max_steps: int = 10):
        self.tools = tools
        self.max_steps = max_steps

    def parse_action(self, llm_output: str) -> dict:
        # ===== 填空 1: 检测 Final Answer =====
        if _____ in llm_output:  # 提示: "Final Answer:"
            ans = llm_output.split("Final Answer:")[-1].strip()
            return {"type": "final_answer", "answer": ans}

        # ===== 填空 2: 解析 Action / Action Input =====
        a = re.search(r'Action:\s*(.+)', llm_output)
        ai = re.search(r'Action Input:\s*(.+)', llm_output)
        if a and ai:
            return {"type": "action",
                    "action": _____,        # 提示: a.group(1).strip()
                    "action_input": _____}   # 提示: ai.group(1).strip()
        return {"type": "error"}

    def execute_tool(self, action: str, action_input: str) -> str:
        # ===== 填空 3: 调用对应工具 =====
        if action in _____:  # 提示: self.tools
            return _____     # 提示: self.tools[action](action_input)
        return f"未知工具: {action}"

    def run(self, question: str) -> dict:
        history = f"Question: {question}\n"
        steps = []
        for step in range(self.max_steps):
            out = mock_react_llm(history, step)
            parsed = self.parse_action(out)

            # ===== 填空 4: 终止条件 =====
            if parsed["type"] == "final_answer":
                steps.append({"step": step, "output": out})
                return {"answer": _____,  # 提示: parsed["answer"]
                        "steps": steps, "num_steps": step + 1}

            if parsed["type"] == "action":
                obs = self.execute_tool(parsed["action"], parsed["action_input"])
                steps.append({"step": step, "action": parsed["action"],
                              "input": parsed["action_input"], "observation": obs})
                # ===== 填空 5: 追加到 history =====
                history += f"\n{out}\nObservation: {_____}\n"  # 提示: obs

        return {"answer": "达到最大步数", "steps": steps,
                "num_steps": self.max_steps}
```

### 提示

- `Final Answer:` 是终止信号，检测到立即返回
- history 累积保证后续步骤能看到之前的推理链
- 正则 `r'Action:\s*(.+)'` 可灵活匹配不同格式

<details>
<summary>参考答案</summary>

```python
# 填空 1
if "Final Answer:" in llm_output:
# 填空 2
"action": a.group(1).strip(),
"action_input": ai.group(1).strip()
# 填空 3
if action in self.tools:
    return self.tools[action](action_input)
# 填空 4
return {"answer": parsed["answer"], "steps": steps, "num_steps": step + 1}
# 填空 5
history += f"\n{out}\nObservation: {obs}\n"
```

**验证:**
```python
tools = {"calculator": calculator, "search": search, "get_weather": get_weather}
agent = ReActAgent(tools=tools)
result = agent.run("北京天气如何？另外算 25*3+10。")
print(f"回答: {result['answer']}")
print(f"步数: {result['num_steps']}")
for s in result["steps"]:
    if "action" in s:
        print(f"  Step {s['step']}: {s['action']}({s['input']}) -> {s['observation']}")
assert result["num_steps"] == 3
assert "85" in result["answer"]
print("ReAct Agent 验证通过")
```

</details>

---

## 练习 5: 结构化 Tool Calling（Level 3）

### 背景

现代 LLM 支持结构化 function calling / tool_use，比 ReAct 文本解析更可靠。核心：用 JSON Schema 定义工具 -> LLM 返回 tool_calls JSON -> 解析并调用 -> 多轮交互。

### 任务

```python
import json, inspect
from typing import Callable, Dict, List, get_type_hints

TYPE_MAP = {int: "integer", float: "number", str: "string",
            bool: "boolean", list: "array", dict: "object"}

def function_to_tool_schema(func: Callable) -> dict:
    """将 Python 函数自动转换为 JSON Schema 格式的 tool 定义"""
    # ===== 填空 1: 提取函数名和描述 =====
    name = _____  # 提示: func.__name__
    desc = (func.__doc__ or "").strip().split("\n")[0]

    sig = inspect.signature(func)
    hints = get_type_hints(func)
    properties, required = {}, []
    for pname, param in sig.parameters.items():
        ptype = hints.get(pname, str)
        properties[pname] = {"type": TYPE_MAP.get(ptype, "string"),
                             "description": f"参数 {pname}"}
        # ===== 填空 2: 无默认值则加入 required =====
        if param.default is _____:  # 提示: inspect.Parameter.empty
            required.append(pname)

    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": properties,
                        "required": required}}}


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.schemas: List[dict] = []

    def register(self, func: Callable):
        self.tools[func.__name__] = func
        self.schemas.append(function_to_tool_schema(func))
        return func

    def execute_tool_call(self, tool_call: dict) -> dict:
        cid = tool_call["id"]
        fname = tool_call["function"]["name"]
        # ===== 填空 3: 解析 arguments =====
        args = _____  # 提示: json.loads(tool_call["function"]["arguments"])
        if fname not in self.tools:
            return {"id": cid, "error": f"未知工具: {fname}"}
        # ===== 填空 4: 调用函数 =====
        func = _____  # 提示: self.tools[fname]
        try:
            return {"id": cid, "result": str(_____)}  # 提示: func(**args)
        except Exception as e:
            return {"id": cid, "error": str(e)}


# Mock LLM: 模拟多轮 tool calling
def mock_tc_llm(messages, schemas, step):
    if step == 0:
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "function": {"name": "get_population",
             "arguments": json.dumps({"country": "中国"})}},
            {"id": "c2", "function": {"name": "get_area",
             "arguments": json.dumps({"country": "中国"})}}]}
    elif step == 1:
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c3", "function": {"name": "compute_density",
             "arguments": json.dumps({"population": 1410000000,
                                       "area_km2": 9600000})}}]}
    return {"role": "assistant", "tool_calls": [],
            "content": "中国人口约 14.1 亿，面积约 960 万 km2，密度约 146.88 人/km2。"}


def run_tool_calling_loop(question: str, registry: ToolRegistry,
                          max_rounds: int = 5) -> dict:
    messages = [{"role": "user", "content": question}]
    all_results = []

    for step in range(max_rounds):
        resp = mock_tc_llm(messages, registry.schemas, step)
        messages.append(resp)
        tc = resp.get("tool_calls", [])

        # ===== 填空 5: 无 tool_calls 则返回最终回答 =====
        if not _____:  # 提示: tc (空列表为 falsy)
            return {"answer": resp["content"],
                    "tool_calls_made": all_results, "num_rounds": step + 1}

        # ===== 填空 6: 执行所有 tool calls =====
        for t in tc:
            res = _____  # 提示: registry.execute_tool_call(t)
            all_results.append(res)
            # ===== 填空 7: 工具结果追加到消息历史 =====
            messages.append({"role": "tool",
                             "tool_call_id": _____,  # 提示: t["id"]
                             "content": json.dumps(res, ensure_ascii=False)})

    return {"answer": "达到最大轮数", "tool_calls_made": all_results,
            "num_rounds": max_rounds}
```

### 提示

- `inspect.Parameter.empty` 表示参数无默认值
- `json.loads` 将 JSON 字符串解析为 dict，`func(**args)` 解包为关键字参数
- 每个 tool result 必须带 `tool_call_id` 与对应请求关联

<details>
<summary>参考答案</summary>

```python
# 填空 1: name = func.__name__
# 填空 2: if param.default is inspect.Parameter.empty:
# 填空 3: args = json.loads(tool_call["function"]["arguments"])
# 填空 4: func = self.tools[fname]  /  str(func(**args))
# 填空 5: if not tc:
# 填空 6: res = registry.execute_tool_call(t)
# 填空 7: "tool_call_id": t["id"],
```

**验证:**
```python
registry = ToolRegistry()

@registry.register
def get_population(country: str) -> int:
    """查询国家人口"""
    return {"中国": 1410000000, "美国": 331000000}.get(country, 0)

@registry.register
def get_area(country: str) -> float:
    """查询国家面积（km2）"""
    return {"中国": 9600000, "美国": 9834000}.get(country, 0)

@registry.register
def compute_density(population: int, area_km2: float) -> float:
    """计算人口密度"""
    return round(population / area_km2, 2) if area_km2 else 0

# 验证 schema 生成
for s in registry.schemas:
    f = s["function"]
    print(f"工具: {f['name']} | 必填: {f['parameters']['required']}")

# 运行多轮 tool calling
result = run_tool_calling_loop("中国的人口密度?", registry)
print(f"\n回答: {result['answer']}")
print(f"轮数: {result['num_rounds']}, 调用次数: {len(result['tool_calls_made'])}")
assert result["num_rounds"] == 3
assert len(result["tool_calls_made"]) == 3
print("Tool Calling 验证通过")
```

</details>

---

## 总结

| 练习 | 难度 | 核心知识点 |
|------|------|-----------|
| 文本分块策略 | Level 2 | 固定窗口 + overlap，按标题语义切分 |
| 余弦相似度与向量检索 | Level 2 | 手写 cosine similarity，top-k 排序，倒排索引 |
| RAG Pipeline 端到端 | Level 3 | 分块 + 检索 + prompt 组装 + 来源标注 |
| ReAct Agent 循环 | Level 2-3 | Thought-Action-Observation 循环，终止检测 |
| 结构化 Tool Calling | Level 3 | schema 自动生成，JSON 解析，多轮工具调用 |

### 延伸思考

1. **分块评估**: 如何量化不同分块策略对检索质量的影响？可以用什么指标？
2. **Embedding 选择**: 通用 embedding 模型（BGE、GTE）为什么不一定适合特定领域？
3. **ReAct vs Tool Calling**: 自由文本解析和结构化 JSON 各有什么优缺点？适用场景？
4. **Agent 安全**: 如何防止 prompt 注入导致 Agent 越权操作？
5. **多 Agent 协作**: 如何拆分复杂任务给多个 Agent？通信与协调机制？

---

## MLM 代码训练模式

完成上面的固定填空后，试试随机挖空模式 -- 每次点击「刷新」会随机遮盖不同的代码片段，帮你彻底记住每一行。

### 文本分块与重叠

<CodeMasker title="固定大小分块 (带 Overlap)" :mask-ratio="0.15">
def fixed_size_chunk(text, chunk_size=200, overlap=50):
    assert overlap < chunk_size
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks
</CodeMasker>

### 余弦相似度检索

<CodeMasker title="余弦相似度与 Top-K 检索" :mask-ratio="0.15">
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x ** 2 for x in a))
    nb = math.sqrt(sum(x ** 2 for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def topk_retrieval(query_emb, doc_embs, doc_texts, k=3):
    scores = []
    for i, emb in enumerate(doc_embs):
        sim = cosine_similarity(query_emb, emb)
        scores.append((i, sim, doc_texts[i]))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]
</CodeMasker>

### RAG Pipeline: 检索-增强-生成

<CodeMasker title="RAG: Retrieve → Augment Prompt → Generate" :mask-ratio="0.15">
def build_prompt(self, query, retrieved):
    ctx_parts, total = [], 0
    for item in retrieved:
        piece = f"[来源: {item['source']}]\n{item['text']}"
        if total + len(piece) > self.max_context_length:
            break
        ctx_parts.append(piece)
        total += len(piece)
    context = "\n\n".join(ctx_parts)
    return f"请根据上下文回答。\n\n[上下文]\n{context}\n\n[问题]\n{query}"

def query(self, question):
    retrieved = self.retrieve(question)
    prompt = self.build_prompt(question, retrieved)
    return {"answer": mock_llm(prompt), "sources": [r["source"] for r in retrieved]}
</CodeMasker>

### ReAct Agent 循环

<CodeMasker title="ReAct: Thought-Action-Observation 循环" :mask-ratio="0.15">
def parse_action(self, llm_output):
    if "Final Answer:" in llm_output:
        ans = llm_output.split("Final Answer:")[-1].strip()
        return {"type": "final_answer", "answer": ans}
    a = re.search(r'Action:\s*(.+)', llm_output)
    ai = re.search(r'Action Input:\s*(.+)', llm_output)
    if a and ai:
        return {"type": "action",
                "action": a.group(1).strip(),
                "action_input": ai.group(1).strip()}
    return {"type": "error"}

def execute_tool(self, action, action_input):
    if action in self.tools:
        return self.tools[action](action_input)
    return f"未知工具: {action}"

def run(self, question):
    history = f"Question: {question}\n"
    steps = []
    for step in range(self.max_steps):
        out = mock_react_llm(history, step)
        parsed = self.parse_action(out)
        if parsed["type"] == "final_answer":
            steps.append({"step": step, "output": out})
            return {"answer": parsed["answer"], "steps": steps, "num_steps": step + 1}
        if parsed["type"] == "action":
            obs = self.execute_tool(parsed["action"], parsed["action_input"])
            history += f"\n{out}\nObservation: {obs}\n"
</CodeMasker>

### Tool Calling 调度器

<CodeMasker title="Tool Schema 生成与多轮调用循环" :mask-ratio="0.15">
def function_to_tool_schema(func):
    name = func.__name__
    desc = (func.__doc__ or "").strip().split("\n")[0]
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    properties, required = {}, []
    for pname, param in sig.parameters.items():
        ptype = hints.get(pname, str)
        properties[pname] = {"type": TYPE_MAP.get(ptype, "string")}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": properties, "required": required}}}

def execute_tool_call(self, tool_call):
    fname = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])
    func = self.tools[fname]
    return {"id": tool_call["id"], "result": str(func(**args))}

def run_tool_calling_loop(question, registry, max_rounds=5):
    messages = [{"role": "user", "content": question}]
    for step in range(max_rounds):
        resp = mock_tc_llm(messages, registry.schemas, step)
        messages.append(resp)
        tc = resp.get("tool_calls", [])
        if not tc:
            return {"answer": resp["content"], "num_rounds": step + 1}
        for t in tc:
            res = registry.execute_tool_call(t)
            messages.append({"role": "tool", "tool_call_id": t["id"],
                             "content": json.dumps(res, ensure_ascii=False)})
</CodeMasker>
