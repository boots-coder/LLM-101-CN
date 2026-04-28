---
title: "Prompt Engineering"
description: "Zero-shot/Few-shot/CoT/ToT/ReAct/Self-Consistency、DSPy、结构化输出"
topics: [prompt-engineering, zero-shot, few-shot, chain-of-thought, tree-of-thoughts, ReAct, DSPy, in-context-learning]
---
# Prompt Engineering 系统化指南

::: info 一句话总结
Prompt Engineering 是与大模型对话的"编程语言"——通过精心设计输入指令的结构、示例和约束，引导模型产生高质量、可控、可靠的输出。
:::


## 在大模型体系中的位置

```
大模型应用层
├── Prompt Engineering（提示工程）◄── 你在这里
├── RAG（检索增强生成）
├── Agent（智能体）
├── Fine-tuning（微调）
└── 评估与对齐
```

Prompt Engineering 是大模型应用的**最底层能力**。无论是 RAG、Agent 还是 Fine-tuning，最终都要通过精心设计的 Prompt 与模型交互。掌握 Prompt Engineering，就是掌握了与 LLM "对话"的语法和语义。

与 Fine-tuning 相比，Prompt Engineering 不修改模型参数，而是在**推理时**通过输入文本引导模型行为——成本几乎为零，却能显著改变输出质量。

## Prompt 的本质

### 四大组成要素

一个完整的 Prompt 可以拆解为四个正交维度：

```
┌───────────────────────────────────────────────────┐
│                Prompt Structure                    │
│                                                   │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐    │
│  │Instruction│  │  Context  │  │  Examples  │    │
│  │ What to do│  │ Background│  │ How to do  │    │
│  └───────────┘  └───────────┘  └───────────┘    │
│                                                   │
│  ┌──────────────────────────────────────────┐    │
│  │           Constraints                     │    │
│  │  Format / Length / Role / Restrictions     │    │
│  └──────────────────────────────────────────┘    │
└───────────────────────────────────────────────────┘
```

| 要素 | 作用 | 示例 |
|------|------|------|
| **指令 (Instruction)** | 告诉模型要做什么任务 | "翻译以下文本为英文" |
| **上下文 (Context)** | 提供背景信息或参考资料 | "以下是一份财报摘要：..." |
| **示例 (Examples)** | 通过输入-输出对展示期望行为 | "正面 → positive，负面 → negative" |
| **约束 (Constraints)** | 限定输出的格式、风格、边界 | "用 JSON 格式输出，不超过 100 字" |

### 从信息论角度理解 Prompt

Prompt 本质上是在**减少模型输出的熵**。未加约束时，模型面对一个 token 有数万种可能的延续；Prompt 的每一个组成部分都在缩小这个搜索空间：

$$
H(Y|X_{\text{prompt}}) \ll H(Y)
$$

其中 $Y$ 是模型输出，$X_{\text{prompt}}$ 是我们设计的 Prompt。指令约束了**任务类型**，上下文约束了**知识范围**，示例约束了**输出格式**，显式约束则直接**裁剪输出空间**。

## 基础技巧

### Zero-shot Prompting

直接给出指令，不提供示例。依赖模型在预训练和对齐阶段学到的能力。

```python
def zero_shot_classify(text: str, client) -> str:
    """Zero-shot 情感分类"""
    prompt = f"""请判断以下文本的情感倾向，只输出"正面"、"负面"或"中性"。

文本：{text}

情感倾向："""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# 示例
# zero_shot_classify("这家餐厅的菜品非常美味，服务也很周到") → "正面"
```

Zero-shot 适合**简单、明确、模型训练数据中常见**的任务。但对于复杂任务或需要特定输出格式的场景，效果往往不够稳定。

### Few-shot Prompting

提供几个输入-输出示例，让模型"学习"任务模式。这是 GPT-3 论文的核心发现——大模型可以通过**上下文中的示例**来适应新任务，而不需要梯度更新。

```python
def few_shot_classify(text: str, client) -> str:
    """Few-shot 情感分类——通过示例引导模型"""
    prompt = """请判断文本的情感倾向。

文本：今天天气真好，心情愉快！
情感：正面

文本：快递又丢了，客服态度还差。
情感：负面

文本：今天周三，明天周四。
情感：中性

文本：{text}
情感：""".format(text=text)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()
```

**Few-shot 的关键设计原则：**

1. **示例数量**：通常 3-5 个即可，过多反而可能引入噪声
2. **示例多样性**：覆盖不同类别和边界情况
3. **示例顺序**：最近的示例（recency bias）影响更大
4. **示例格式**：保持一致的格式，模型会模仿格式

### Chain-of-Thought (CoT)

CoT 是 2022 年 Google Brain 提出的里程碑式技巧——让模型**逐步推理**，而不是直接给出答案。

```python
def cot_math_reasoning(question: str, client) -> str:
    """Chain-of-Thought 数学推理"""
    prompt = f"""请一步一步地思考，然后给出最终答案。

问题：{question}

让我们一步一步来思考："""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

# 问题："一个商店有 23 个苹果，卖出 17 个后又进货 12 个，问现在有多少个苹果？"
# CoT 输出：
# 1. 初始苹果数量：23 个
# 2. 卖出 17 个后：23 - 17 = 6 个
# 3. 又进货 12 个：6 + 12 = 18 个
# 最终答案：18 个苹果
```

CoT 的两种激活方式：

| 方式 | 说明 | Prompt |
|------|------|--------|
| **Zero-shot CoT** | 只加一句话即可激活 | "Let's think step by step" |
| **Few-shot CoT** | 在示例中展示推理过程 | 提供带推理步骤的 QA 示例 |

**为什么 CoT 有效？** 从计算理论的角度看，Transformer 直接映射输入到输出相当于一个固定深度的电路；而 CoT 将中间推理步骤显式化，相当于增加了"计算步数"，让模型可以解决需要更多串行计算步骤的问题。

## 高级技巧

### Self-Consistency（自洽性解码）

核心思想：对同一个问题**多次采样**（设置较高 temperature），然后对最终答案进行**多数投票**。

```python
import collections

def self_consistency(question: str, client, n_samples: int = 5) -> str:
    """Self-Consistency: 多次采样 + 多数投票"""
    prompt = f"""请一步一步思考，然后在最后一行给出最终答案，格式为"答案：XXX"。

问题：{question}

让我们一步一步来思考："""

    answers = []
    for _ in range(n_samples):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # 较高温度以产生多样化推理路径
            max_tokens=1024
        )
        text = response.choices[0].message.content
        # 提取最终答案
        for line in text.strip().split("\n")[::-1]:
            if "答案" in line:
                answer = line.split("：")[-1].strip()
                answers.append(answer)
                break

    # 多数投票
    counter = collections.Counter(answers)
    best_answer, count = counter.most_common(1)[0]
    confidence = count / len(answers)

    return f"{best_answer} (置信度: {confidence:.0%}, 共 {len(answers)} 次采样)"
```

Self-Consistency 的直觉：正确的推理路径有多种，但它们会**收敛到同一个答案**；错误的推理路径则各有各的错误。所以多数投票能有效过滤掉错误。

### Tree-of-Thoughts (ToT)

CoT 是一条链，ToT 是一棵树——允许模型在推理过程中**探索多条路径**、**回溯**、**评估**。

```python
def tree_of_thoughts(problem: str, client, breadth: int = 3, depth: int = 3) -> str:
    """Tree-of-Thoughts: 树状搜索推理"""

    def generate_thoughts(state: str, step: int) -> list[str]:
        """为当前状态生成多个可能的下一步思考"""
        prompt = f"""问题：{problem}

当前推理状态：
{state}

请生成 {breadth} 个不同的下一步思考方向，用 [1] [2] [3] 分隔。
每个方向用 1-2 句话描述。"""

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        text = resp.choices[0].message.content
        thoughts = []
        for i in range(1, breadth + 1):
            marker = f"[{i}]"
            next_marker = f"[{i+1}]"
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index(next_marker) if next_marker in text else len(text)
                thoughts.append(text[start:end].strip())
        return thoughts if thoughts else [text]

    def evaluate_thought(state: str, thought: str) -> float:
        """评估某个思考方向的前景"""
        prompt = f"""问题：{problem}

推理路径：
{state}
→ {thought}

请评估这条推理路径的前景（1-10 分），只输出一个数字。"""

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        try:
            return float(resp.choices[0].message.content.strip())
        except ValueError:
            return 5.0

    # BFS 搜索
    current_states = [""]
    for step in range(depth):
        candidates = []
        for state in current_states:
            thoughts = generate_thoughts(state, step)
            for thought in thoughts:
                score = evaluate_thought(state, thought)
                new_state = f"{state}\n步骤 {step+1}: {thought}"
                candidates.append((new_state, score))

        # 保留得分最高的 breadth 个状态
        candidates.sort(key=lambda x: x[1], reverse=True)
        current_states = [c[0] for c in candidates[:breadth]]

    return current_states[0] if current_states else "无法求解"
```

### ReAct（Reasoning + Acting）

ReAct 将**推理**和**工具调用**交织在一起，是 Agent 架构的基石（详见 [Agent 章节](./agents.md)）。

```python
REACT_SYSTEM_PROMPT = """你是一个智能助手，可以使用以下工具：

1. search(query) - 搜索互联网获取信息
2. calculator(expression) - 计算数学表达式
3. lookup(term) - 在百科中查找术语

请按照以下格式回答：
Thought: 分析当前情况，决定下一步
Action: 工具名(参数)
Observation: [工具返回结果]
...（重复上述步骤直到可以回答）
Thought: 我现在可以回答了
Answer: 最终答案
"""

def react_loop(question: str, tools: dict, client, max_steps: int = 5) -> str:
    """ReAct 推理-行动循环"""
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            stop=["Observation:"]  # 在 Observation 前停下，等待工具执行
        )
        assistant_text = response.choices[0].message.content

        # 检查是否已经给出最终答案
        if "Answer:" in assistant_text:
            return assistant_text.split("Answer:")[-1].strip()

        # 解析 Action
        if "Action:" in assistant_text:
            action_line = [l for l in assistant_text.split("\n") if "Action:" in l][-1]
            action_str = action_line.split("Action:")[-1].strip()
            # 解析工具名和参数
            tool_name = action_str.split("(")[0].strip()
            tool_arg = action_str.split("(")[1].rstrip(")")

            # 执行工具
            if tool_name in tools:
                observation = tools[tool_name](tool_arg)
            else:
                observation = f"错误：未知工具 {tool_name}"

            # 将结果加入对话
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "达到最大步数限制，无法得出结论"
```

### Automatic Prompt Optimization (APO)

手工设计 Prompt 效率低，APO 的思路是让 LLM **自动优化 Prompt**：

```python
def auto_prompt_optimize(
    task_description: str,
    eval_examples: list[dict],  # [{"input": ..., "expected": ...}]
    client,
    n_iterations: int = 5
) -> str:
    """自动 Prompt 优化：用 LLM 迭代改进 Prompt"""

    # 初始 Prompt
    current_prompt = f"请完成以下任务：{task_description}\n\n输入：{{input}}\n输出："

    for iteration in range(n_iterations):
        # 在验证集上评估当前 Prompt
        results = []
        for ex in eval_examples:
            filled = current_prompt.format(input=ex["input"])
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": filled}],
                temperature=0
            )
            output = resp.choices[0].message.content.strip()
            correct = output == ex["expected"]
            results.append({
                "input": ex["input"],
                "expected": ex["expected"],
                "actual": output,
                "correct": correct
            })

        accuracy = sum(r["correct"] for r in results) / len(results)
        errors = [r for r in results if not r["correct"]]

        if accuracy == 1.0:
            break  # 完美！

        # 让 LLM 分析错误并改进 Prompt
        error_report = "\n".join(
            f"输入: {e['input']} | 期望: {e['expected']} | 实际: {e['actual']}"
            for e in errors[:5]
        )

        meta_prompt = f"""当前 Prompt：
{current_prompt}

当前准确率：{accuracy:.0%}

错误案例：
{error_report}

请分析错误原因，然后生成一个改进后的 Prompt。
只输出改进后的完整 Prompt，不要其他解释。"""

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": meta_prompt}],
            temperature=0.3
        )
        current_prompt = resp.choices[0].message.content.strip()
        print(f"迭代 {iteration+1}: 准确率 {accuracy:.0%}")

    return current_prompt
```

## 结构化输出引导

### JSON Mode

现代 LLM API 大多支持强制 JSON 输出：

```python
import json

def structured_extraction(text: str, client) -> dict:
    """结构化信息抽取——使用 JSON Mode"""
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},  # 强制 JSON 输出
        messages=[
            {"role": "system", "content": """从文本中抽取实体信息，以 JSON 格式输出：
{
    "persons": [{"name": "...", "role": "..."}],
    "organizations": [{"name": "...", "type": "..."}],
    "events": [{"description": "...", "date": "..."}]
}"""},
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    return json.loads(response.choices[0].message.content)
```

### Outlines 库：基于 FSM 的受限生成

Outlines 是一个开源库，通过**有限状态机 (FSM)** 在解码时约束 token 采样，确保输出严格符合指定的 JSON Schema 或正则表达式。

```python
# pip install outlines
import outlines
from pydantic import BaseModel

class Sentiment(BaseModel):
    text: str
    label: str      # "positive" | "negative" | "neutral"
    confidence: float

# 从 HuggingFace 加载模型
model = outlines.models.transformers("Qwen/Qwen2.5-7B-Instruct")

# 基于 Pydantic Schema 的受限生成
generator = outlines.generate.json(model, Sentiment)
result = generator("分析这段文本的情感：今天的会议效率很高。")
# result 一定是 Sentiment 类型，字段完整，类型正确
print(result)  # Sentiment(text="...", label="positive", confidence=0.92)
```

### LMQL：LLM 的查询语言

LMQL 提供类 SQL 的语法来控制 LLM 生成：

```python
# LMQL 示例（伪代码风格）
"""
argmax
    "请对以下评论进行分类：{review}\n"
    "类别：[CATEGORY]"
    "理由：[REASON]"
from
    "gpt-4o"
where
    CATEGORY in ["正面", "负面", "中性"]
    and len(REASON) < 100
"""
```

## Prompt 模板设计模式

### Chat Template：System / User / Assistant

现代 Chat 模型使用角色化的消息格式：

```python
def build_chat_messages(
    system_prompt: str,
    user_query: str,
    few_shot_examples: list[dict] = None
) -> list[dict]:
    """构建标准的 Chat 消息序列"""
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # 注入 Few-shot 示例（通过 User/Assistant 交替模拟）
    if few_shot_examples:
        for ex in few_shot_examples:
            messages.append({"role": "user", "content": ex["input"]})
            messages.append({"role": "assistant", "content": ex["output"]})

    messages.append({"role": "user", "content": user_query})
    return messages

# 使用
messages = build_chat_messages(
    system_prompt="你是一个专业的代码审查专家。请审查代码并指出问题。",
    user_query="请审查以下 Python 代码：\n```python\ndef add(a, b): return a+b\n```",
    few_shot_examples=[
        {
            "input": "审查：`x = eval(input())`",
            "output": "严重安全问题：eval() 可执行任意代码，应使用 ast.literal_eval() 或特定类型转换。"
        }
    ]
)
```

### Prompt 模板工厂

```python
from string import Template
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    """可复用的 Prompt 模板"""
    name: str
    system: str
    user_template: str  # 使用 {variable} 占位符
    few_shot: list[dict] = None

    def render(self, **kwargs) -> list[dict]:
        messages = [{"role": "system", "content": self.system}]
        if self.few_shot:
            for ex in self.few_shot:
                messages.append({"role": "user", "content": ex["input"]})
                messages.append({"role": "assistant", "content": ex["output"]})
        messages.append({
            "role": "user",
            "content": self.user_template.format(**kwargs)
        })
        return messages

# 预定义模板
SUMMARIZE_TEMPLATE = PromptTemplate(
    name="summarize",
    system="你是一个专业的文本摘要专家。请用简洁准确的语言进行摘要。",
    user_template="请将以下文本摘要为 {max_words} 字以内：\n\n{text}"
)

TRANSLATE_TEMPLATE = PromptTemplate(
    name="translate",
    system="你是一个专业翻译，擅长中英互译。保持原文的语气和风格。",
    user_template="请将以下{source_lang}文本翻译为{target_lang}：\n\n{text}"
)

# 使用
msgs = SUMMARIZE_TEMPLATE.render(text="一篇很长的文章...", max_words=100)
```

## In-Context Learning 的原理与局限

### ICL 是什么？

In-Context Learning (ICL) 是指模型**仅通过 Prompt 中的示例**就能学习新任务，而不需要更新权重。这是大模型最令人惊讶的涌现能力之一。

### ICL 的两种假说

**假说 1：贝叶斯推断视角**

模型在预训练时学到了一个隐式的贝叶斯学习算法。给定 Prompt 中的示例 $(x_1, y_1), ..., (x_k, y_k)$，模型在做：

$$
P(y|x, \text{examples}) \approx \sum_{\theta} P(y|x,\theta) \cdot P(\theta|\text{examples})
$$

即模型在示例上"后验推断"出了任务参数 $\theta$。

**假说 2：任务向量视角**

Prompt 中的示例在模型内部激活了特定的"任务向量"——注意力头学会了从示例中提取 input→output 的映射模式，并将这个模式应用到新输入上。

### ICL 的局限

| 局限 | 说明 |
|------|------|
| **上下文窗口限制** | 示例数量受 context length 约束 |
| **位置偏差** | 模型对示例的顺序和位置敏感 |
| **不稳定性** | 示例的微小变化可能导致输出大幅波动 |
| **无法学习全新概念** | ICL 更像是"唤醒"已有能力，而非真正学习新知识 |
| **浅层模式匹配** | 对复杂逻辑推理，ICL 不如 Fine-tuning 可靠 |

## 常见陷阱

### 位置偏差 (Positional Bias)

模型对 Prompt 中信息的位置高度敏感。在选择题场景中，模型可能偏好选择 A 或最后一个选项：

```python
def mitigate_position_bias(question: str, options: list[str], client, n_shuffles: int = 3) -> str:
    """通过多次打乱选项顺序来缓解位置偏差"""
    import random
    from collections import Counter

    votes = Counter()
    for _ in range(n_shuffles):
        shuffled = options.copy()
        random.shuffle(shuffled)
        option_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(shuffled))

        prompt = f"""{question}

{option_text}

请只输出正确选项的内容（不要输出字母编号）。"""

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer_content = resp.choices[0].message.content.strip()
        votes[answer_content] += 1

    return votes.most_common(1)[0][0]
```

### Lost in the Middle

Liu et al. (2023) 发现：当相关信息放在长 Prompt 的**中间位置**时，模型表现显著下降。模型对开头和结尾的信息更敏感。

```
性能
 ▲
 │  ●                              ●
 │    ●                          ●
 │      ●                      ●
 │        ●                  ●
 │          ●    ●    ●    ●
 │            ●    ●    ●
 └──────────────────────────────→ 关键信息位置
   开头                        结尾
```

**缓解策略：**
- 将最重要的信息放在 Prompt 开头或结尾
- 使用分隔符明确标记关键段落
- 对长文档做摘要后再放入 Prompt

### 长上下文退化

即使模型声称支持 128K token，实际性能随上下文长度增加而衰减：

```python
def chunk_and_process(long_text: str, question: str, client, chunk_size: int = 4000) -> str:
    """对长文本分块处理，避免长上下文退化"""
    # 1. 将长文本切分为块
    words = long_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    # 2. 对每个块提取与问题相关的信息
    relevant_parts = []
    for i, chunk in enumerate(chunks):
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "从文本中提取与问题相关的关键信息。如果不相关，输出'无相关信息'。"},
                {"role": "user", "content": f"问题：{question}\n\n文本块 {i+1}：\n{chunk}"}
            ],
            temperature=0
        )
        answer = resp.choices[0].message.content.strip()
        if answer != "无相关信息":
            relevant_parts.append(answer)

    # 3. 综合所有相关信息回答
    combined = "\n\n".join(relevant_parts)
    final_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "基于以下整理后的信息回答用户问题。"},
            {"role": "user", "content": f"问题：{question}\n\n相关信息：\n{combined}"}
        ],
        temperature=0
    )
    return final_resp.choices[0].message.content
```

## DSPy 框架简介

DSPy 将 Prompt Engineering 从"手工文本调参"提升为"可编程、可优化"的范式。

### 核心概念

```
┌────────────────────────────────────────────┐
│                  DSPy                      │
│                                            │
│  Signature  →  Define input/output fields  │
│  Module     →  Composable prompt units     │
│  Optimizer  →  Auto-optimize prompts       │
│                                            │
│  "The PyTorch of Prompt Engineering"       │
└────────────────────────────────────────────┘
```

### 代码示例

```python
import dspy

# 配置 LLM
lm = dspy.LM("openai/gpt-4o", temperature=0)
dspy.configure(lm=lm)

# --- Signature：声明式定义任务 ---
class SentimentAnalysis(dspy.Signature):
    """判断文本的情感倾向"""
    text: str = dspy.InputField(desc="待分析的文本")
    sentiment: str = dspy.OutputField(desc="情感：positive/negative/neutral")
    reason: str = dspy.OutputField(desc="判断理由")

# --- Module：使用 Signature ---
class SentimentClassifier(dspy.Module):
    def __init__(self):
        self.predict = dspy.ChainOfThought(SentimentAnalysis)

    def forward(self, text: str):
        return self.predict(text=text)

# 使用
classifier = SentimentClassifier()
result = classifier(text="这家店的服务太差了，再也不来了")
print(result.sentiment)  # "negative"
print(result.reason)     # "文本表达了对服务的不满和抵制态度"

# --- Optimizer：自动优化 ---
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

# 定义评估指标
def metric(example, prediction, trace=None):
    return example.sentiment == prediction.sentiment

# 用训练数据自动优化
optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(SentimentClassifier(), trainset=train_examples)
```

DSPy 的革命性在于：你不再手写 Prompt 文本，而是**声明任务接口 (Signature)**，让框架自动搜索最优的 Prompt 组合——包括指令、示例数量、示例选择、推理步骤等。

## 代码实战：多步 CoT 推理 Pipeline

以下实现一个完整的多步推理系统，结合 CoT、Self-Consistency 和结构化输出：

```python
"""
多步 CoT 推理 Pipeline
任务：复杂数学应用题求解
"""
import json
import re
from collections import Counter
from dataclasses import dataclass

@dataclass
class ReasoningStep:
    step_number: int
    description: str
    calculation: str
    result: str

@dataclass
class Solution:
    question: str
    steps: list[ReasoningStep]
    final_answer: str
    confidence: float

COT_SYSTEM = """你是一个严谨的数学推理专家。
请按照以下 JSON 格式一步一步解题：

{
    "steps": [
        {
            "step_number": 1,
            "description": "这一步做什么",
            "calculation": "具体计算过程",
            "result": "这一步的结果"
        }
    ],
    "final_answer": "最终数值答案"
}

要求：
1. 每一步只做一个操作
2. 明确写出计算过程
3. 最终答案只包含数值和单位"""

def solve_with_cot(question: str, client) -> dict:
    """单次 CoT 求解"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": question}
        ],
        temperature=0.7
    )
    return json.loads(resp.choices[0].message.content)

def extract_numeric(answer: str) -> float:
    """从答案字符串中提取数值"""
    numbers = re.findall(r"[-+]?\d*\.?\d+", answer)
    return float(numbers[0]) if numbers else 0.0

def cot_pipeline(question: str, client, n_samples: int = 5) -> Solution:
    """
    完整的 CoT 推理 Pipeline：
    1. 多次采样（Self-Consistency）
    2. 提取数值答案
    3. 多数投票
    4. 返回最佳推理路径
    """
    all_solutions = []
    numeric_answers = []

    for i in range(n_samples):
        try:
            sol = solve_with_cot(question, client)
            answer = sol.get("final_answer", "")
            numeric = extract_numeric(answer)
            all_solutions.append(sol)
            numeric_answers.append(numeric)
        except (json.JSONDecodeError, KeyError):
            continue  # 跳过格式错误的输出

    if not numeric_answers:
        raise ValueError("所有采样均失败")

    # 多数投票（对数值取最接近的聚类）
    counter = Counter(numeric_answers)
    best_numeric, vote_count = counter.most_common(1)[0]
    confidence = vote_count / len(numeric_answers)

    # 找到对应最佳答案的推理路径
    best_idx = numeric_answers.index(best_numeric)
    best_sol = all_solutions[best_idx]

    steps = [
        ReasoningStep(
            step_number=s["step_number"],
            description=s["description"],
            calculation=s["calculation"],
            result=s["result"]
        )
        for s in best_sol.get("steps", [])
    ]

    return Solution(
        question=question,
        steps=steps,
        final_answer=best_sol["final_answer"],
        confidence=confidence
    )

# --- 使用示例 ---
if __name__ == "__main__":
    from openai import OpenAI
    client = OpenAI()

    question = """
    一个水果店进了 3 箱苹果和 5 箱橙子。每箱苹果有 24 个，每箱橙子有 30 个。
    苹果每个卖 3 元，橙子每个卖 2.5 元。如果全部卖出，总收入是多少？
    """

    solution = cot_pipeline(question, client, n_samples=5)

    print(f"问题：{solution.question.strip()}")
    print(f"\n推理步骤：")
    for step in solution.steps:
        print(f"  步骤 {step.step_number}: {step.description}")
        print(f"    计算: {step.calculation}")
        print(f"    结果: {step.result}")
    print(f"\n最终答案：{solution.final_answer}")
    print(f"置信度：{solution.confidence:.0%}")
```

## 苏格拉底时刻

> **Q1：为什么 Few-shot 有时比 Zero-shot 差？**
> 提示：考虑示例选择不当、格式干扰、以及模型"过度模仿"示例中的错误模式。

> **Q2：CoT 对所有任务都有效吗？**
> 提示：考虑任务复杂度——对简单的事实查询，CoT 反而会"想多了"；只有需要多步推理的任务才能从 CoT 中获益。

> **Q3：Self-Consistency 为什么比单次 CoT 更好？**
> 提示：从统计学角度——多次独立采样的众数是最大似然估计的近似；从信息论角度——多条推理路径提供了互补信息。

> **Q4：ICL 和 Fine-tuning 的本质区别是什么？**
> 提示：ICL 不修改权重，是在激活空间中做"临时适配"；Fine-tuning 通过梯度更新永久改变了模型参数。ICL 受限于上下文窗口，Fine-tuning 不受此限制。

> **Q5：为什么长 Prompt 中间的信息容易被忽略？**
> 提示：与注意力机制的分布有关——位置编码使得注意力对开头和结尾的 token 权重更高，形成"U 形"注意力曲线。

## 常见问题 & 面试考点

### 面试高频问题

**Q：为什么 Few-shot 有效？**

A：大模型在预训练时接触了海量的"输入-输出"模式（如 QA 对、翻译对、代码-注释对），Few-shot 示例实际上是在**激活模型已有的任务理解能力**，而不是真正"学习"新技能。更准确地说，Few-shot 帮助模型定位到正确的"任务子空间"。

**Q：ICL 和 Fine-tuning 怎么选？**

| 场景 | 选择 | 理由 |
|------|------|------|
| 快速原型验证 | ICL | 零成本，即时生效 |
| 少量标注数据 | ICL + CoT | 充分利用每个示例 |
| 大量标注数据 + 高精度需求 | Fine-tuning | 深度适配任务 |
| 需要特定输出风格 | Fine-tuning | 风格需要参数级别的学习 |
| 知识密集型任务 | ICL + RAG | RAG 提供知识，ICL 提供格式 |

**Q：Prompt 注入攻击如何防御？**

Prompt Injection 是指用户在输入中嵌入"覆盖指令"来劫持模型行为。防御方法：

1. **输入净化**：过滤或转义特殊指令标记
2. **角色隔离**：严格区分 system prompt 和 user input
3. **输出验证**：对模型输出做后处理检查
4. **多层防御**：用另一个 LLM 检测 Prompt 注入

**Q：temperature 怎么选？**

- `0`：确定性输出，适合事实问答、分类、结构化抽取
- `0.3-0.5`：轻微随机性，适合摘要、翻译
- `0.7-1.0`：较高随机性，适合创意写作、Self-Consistency 采样
- `>1.0`：极高随机性，几乎不用于生产场景

## 推荐资源

| 资源 | 类型 | 说明 |
|------|------|------|
| [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) | 官方文档 | OpenAI 官方最佳实践 |
| [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering) | 官方文档 | Claude 提示工程指南 |
| [DSPy 文档](https://dspy-docs.vercel.app/) | 框架文档 | 可编程 Prompt 优化框架 |
| [Chain-of-Thought 原始论文](https://arxiv.org/abs/2201.11903) | 论文 | Wei et al., 2022 |
| [Tree of Thoughts](https://arxiv.org/abs/2305.10601) | 论文 | Yao et al., 2023 |
| [Self-Consistency](https://arxiv.org/abs/2203.11171) | 论文 | Wang et al., 2022 |
| [Lost in the Middle](https://arxiv.org/abs/2307.03172) | 论文 | Liu et al., 2023 |
| [LMQL](https://lmql.ai/) | 工具 | LLM 查询语言 |
| [Outlines](https://github.com/outlines-dev/outlines) | 工具 | 受限生成库 |

---

> 下一步：掌握 Prompt Engineering 后，进入 [RAG](./rag.md) 学习如何为模型注入外部知识，或进入 [Agent](./agents.md) 学习如何让模型自主行动。
