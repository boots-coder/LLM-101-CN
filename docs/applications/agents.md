---
title: "Agent 智能体"
description: "ReAct 框架、Function Calling、Human-in-the-loop、Multi-Agent 架构"
topics: [agent, ReAct, function-calling, tool-use, human-in-the-loop, multi-agent, MCP, A2A]
---
# Agent 智能体

::: info 一句话总结
Agent 让 LLM 从被动回答者变为主动行动者——通过感知环境、使用工具、自主决策的循环，完成复杂的多步任务。
:::


## 在大模型体系中的位置

```
大模型应用层
├── Prompt Engineering（提示工程）
├── RAG（检索增强生成）
├── Agent（智能体）◄── 你在这里
├── Fine-tuning（微调）
└── 评估与对齐
```

如果说 RAG 让 LLM 获得了"查阅资料"的能力，那 Agent 让 LLM 获得了"动手做事"的能力。Agent 是大模型应用的最高层抽象——它将 LLM 作为"大脑"，配合记忆、工具和规划能力，形成一个自主的智能系统。

## Agent 的核心概念

### 什么是 LLM Agent

传统的 LLM 交互是单轮的：用户提问，模型回答。Agent 打破了这个模式——它让 LLM 能够**观察环境、制定计划、调用工具、获取反馈、调整策略**，形成一个自主的决策循环。

### Agent = LLM + Memory + Tools + Planning

```
              ┌───────────────────────────┐
              │        LLM Brain          │
              │  (Reasoning / Planning)   │
              └────────┬──────────────────┘
                       │
         ┌─────────────┼──────────────────┐
         │             │                  │
    ┌────▼─────┐  ┌────▼─────┐    ┌──────▼───────┐
    │  Memory  │  │  Tools   │    │   Planning   │
    │ Short +  │  │ Search/  │    │  CoT / ToT   │
    │ Long-term│  │ Code/API │    │  Task Decomp │
    └──────────┘  └──────────┘    └──────────────┘
```

四个核心组件：

| 组件 | 作用 | 类比 |
|------|------|------|
| **LLM** | 推理、决策、语言理解 | 大脑 |
| **Memory** | 存储历史交互和知识 | 记忆 |
| **Tools** | 执行具体操作（搜索、计算、API） | 手和脚 |
| **Planning** | 任务分解和策略制定 | 前额叶 |

## ReAct 框架

### Reasoning + Acting

ReAct 是最经典的 Agent 框架，核心思想来自一个简单的观察：人类解决问题时，**推理**（思考下一步做什么）和**行动**（实际去做）是交替进行的。

```
Thought: 我需要查找2024年中国GDP数据
Action: search("2024年中国GDP")
Observation: 2024年中国GDP约为134.9万亿元...
Thought: 我已经找到了数据，现在需要计算增长率
Action: calculator("(134.9 - 126.1) / 126.1 * 100")
Observation: 6.98%
Thought: 我现在可以回答用户的问题了
Answer: 2024年中国GDP约134.9万亿元，同比增长约7.0%。
```

### 思考-行动-观察循环

ReAct 的每一轮包含三个步骤：

1. **Thought（思考）**：LLM 显式输出推理过程——分析当前状态，决定下一步做什么
2. **Action（行动）**：调用一个具体的工具，传入参数
3. **Observation（观察）**：工具返回的结果，作为新的上下文加入对话

这个循环持续进行，直到 LLM 认为已经收集了足够的信息来回答问题。

> ReAct 比纯 Chain-of-Thought 强在哪里？CoT 只能推理，无法获取新信息；ReAct 可以在推理过程中随时"查阅资料"，将**内部推理**和**外部信息获取**统一在一个框架中。

### ReAct 的完整实现

```python
import json
import re

class ReActAgent:
    """
    ReAct Agent 的完整实现
    实现 Thought → Action → Observation 的循环推理
    """
    
    def __init__(self, llm, tools: dict, max_steps: int = 10):
        self.llm = llm           # LLM 接口（如 OpenAI API）
        self.tools = tools       # 可用工具字典 {"tool_name": tool_function}
        self.max_steps = max_steps
    
    def _build_system_prompt(self):
        """构建 ReAct 系统提示词，包含可用工具的描述"""
        tool_descriptions = "\n".join([
            f"- {name}: {func.__doc__}" for name, func in self.tools.items()
        ])
        return f"""你是一个具有工具使用能力的 AI 助手。你可以使用以下工具：

{tool_descriptions}

请按照以下格式回答问题：

Thought: 分析当前状态，思考下一步应该做什么
Action: tool_name(参数)
Observation: [工具返回的结果，由系统填充]
... (可以重复多轮 Thought/Action/Observation)
Thought: 我已经获得足够信息来回答了
Answer: 最终答案
"""
    
    def _parse_action(self, text: str):
        """从 LLM 输出中解析 Action 调用"""
        # 匹配 Action: tool_name(args) 格式
        match = re.search(r'Action:\s*(\w+)\((.+?)\)', text)
        if match:
            tool_name = match.group(1)
            tool_args = match.group(2).strip('"\'')
            return tool_name, tool_args
        return None, None
    
    def _execute_tool(self, tool_name: str, tool_args: str):
        """执行工具调用并返回结果"""
        if tool_name not in self.tools:
            return f"错误：工具 '{tool_name}' 不存在。可用工具：{list(self.tools.keys())}"
        try:
            result = self.tools[tool_name](tool_args)
            return str(result)
        except Exception as e:
            return f"工具执行错误：{str(e)}"
    
    def run(self, question: str) -> str:
        """运行 ReAct 循环直到得到最终答案"""
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": question}
        ]
        
        for step in range(self.max_steps):
            # 让 LLM 生成 Thought + Action
            response = self.llm.chat(messages)
            
            # 检查是否已经得到最终答案
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
                return answer
            
            # 解析并执行 Action
            tool_name, tool_args = self._parse_action(response)
            if tool_name is None:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "请按照格式输出 Action。"})
                continue
            
            # 执行工具，获得 Observation
            observation = self._execute_tool(tool_name, tool_args)
            
            # 将整个过程加入对话历史
            full_response = f"{response}\nObservation: {observation}"
            messages.append({"role": "assistant", "content": full_response})
            messages.append({"role": "user", "content": "请继续推理。"})
        
        return "超过最大推理步数，无法得出答案。"


# ===== 使用示例 =====

def search(query: str) -> str:
    """在搜索引擎中搜索信息，返回相关结果摘要"""
    # 实际应用中调用搜索 API
    return f"搜索 '{query}' 的结果：..."

def calculator(expression: str) -> str:
    """计算数学表达式，返回计算结果"""
    try:
        return str(eval(expression))
    except:
        return "计算错误"

def lookup(term: str) -> str:
    """在知识库中查找特定术语的定义"""
    knowledge = {
        "Transformer": "一种基于自注意力机制的神经网络架构",
        "RLHF": "基于人类反馈的强化学习，用于对齐大模型",
    }
    return knowledge.get(term, f"未找到 '{term}' 的定义")

# 注册工具并运行
tools = {"search": search, "calculator": calculator, "lookup": lookup}
agent = ReActAgent(llm=my_llm, tools=tools, max_steps=8)
answer = agent.run("Transformer 是什么？它最早发表在哪一年？")
```

### ReAct 的局限性

- **推理步数限制**：context window 有限，长链推理可能丢失前文
- **错误传播**：早期的错误 Thought 会影响后续所有推理
- **工具选择错误**：模型可能选择错误的工具或传入错误的参数
- **"过度自信"**：模型可能跳过检索，直接基于（可能错误的）参数知识回答

## Function Calling

### OpenAI Function Calling 协议

Function Calling 是让 LLM **结构化**调用工具的标准机制。与 ReAct 中自然语言描述的工具调用不同，Function Calling 输出严格的 JSON，更容易解析和执行。

```
1. 系统定义可用函数的 schema（名称、参数、描述）
2. 用户发送请求
3. LLM 判断是否需要调用函数
4. 如需要，输出函数名 + JSON 参数
5. 应用层执行函数，将结果返回 LLM
6. LLM 基于函数结果生成最终回答
```

### Tool 定义格式 (JSON Schema)

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 '北京'、'上海'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位，默认摄氏度"
                    }
                },
                "required": ["city"]  # city 是必填参数
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "在知识库中搜索与查询相关的文档",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询词"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回的文档数量，默认 5"
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

### 模型如何学会调用工具

模型并不是"理解"了工具——它是通过训练学会了**根据 JSON Schema 描述和用户意图，生成符合格式的函数调用**。关键因素：

1. **工具描述质量**：清晰的 description 直接影响模型选择准确率
2. **参数约束**：enum、required 等约束帮助模型生成合法参数
3. **Few-shot 示例**：在 system prompt 中提供调用示例可提升准确率

```python
from openai import OpenAI
client = OpenAI()

# 完整的 Function Calling 流程
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,           # 传入工具定义
    tool_choice="auto"     # 让模型自动决定是否调用工具
)

# 模型输出结构化的函数调用
tool_call = response.choices[0].message.tool_calls[0]
print(f"函数名: {tool_call.function.name}")           # "get_weather"
print(f"参数: {tool_call.function.arguments}")          # '{"city": "北京"}'

# 执行函数并将结果返回给模型
weather_result = get_weather(city="北京")
follow_up = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "北京今天天气怎么样？"},
        response.choices[0].message,  # 包含 tool_call 的消息
        {"role": "tool", "tool_call_id": tool_call.id, "content": weather_result}
    ]
)
print(follow_up.choices[0].message.content)  # "北京今天晴，气温28度..."
```

### 并行 Function Calling

当模型判断需要同时调用多个工具时（如用户问"北京和上海的天气"），它可以在一次响应中返回多个 tool_calls：

```python
# 模型可能返回两个 tool_calls：
# tool_calls[0]: get_weather(city="北京")
# tool_calls[1]: get_weather(city="上海")
# 应用层可以并行执行这两个调用，提升响应速度
```

## 工具使用 (Tool Use)

### 常见工具类型

| 工具类型 | 示例 | 能力扩展 |
|---------|------|---------|
| **信息检索** | 搜索引擎、数据库查询、RAG | 获取实时/私域知识 |
| **代码执行** | Python 解释器、Shell 命令 | 精确计算、数据分析、绘图 |
| **API 调用** | 天气、地图、支付、邮件 | 连接外部服务 |
| **文件操作** | 读写文件、解析 PDF/Excel | 处理非文本数据 |
| **浏览器** | 网页浏览、表单填写 | 与 Web 应用交互 |

### 工具选择策略

当可用工具很多（10+）时，模型需要准确判断使用哪个工具。工具描述的质量直接影响选择准确率：

```python
# 差的工具描述：
{"name": "tool1", "description": "处理数据"}  # 太模糊

# 好的工具描述：
{
    "name": "sql_query",
    "description": "在企业数据库中执行 SQL 查询。适用于需要精确数字统计的问题，"
                   "如营收、用户量、订单数等。不适用于模糊的语义搜索。"
                   "返回格式为 JSON 数组。",
}
```

好的工具描述应该清晰说明：**做什么、不做什么、输入格式、返回格式、适用场景**。

### 搜索工具实现示例

```python
import requests

def web_search(query: str, num_results: int = 5) -> str:
    """
    调用搜索引擎 API 获取网页搜索结果
    返回 top-k 结果的标题和摘要
    """
    # 实际应用中使用 Serper/Bing/Google API
    results = search_api.search(query, num=num_results)
    formatted = []
    for i, r in enumerate(results):
        formatted.append(f"[{i+1}] {r['title']}\n{r['snippet']}\nURL: {r['url']}")
    return "\n\n".join(formatted)

def code_executor(code: str) -> str:
    """
    在沙箱环境中执行 Python 代码并返回输出
    支持数学计算、数据分析、绘图等
    安全提示：代码在隔离环境中运行，无法访问网络和文件系统
    """
    import subprocess
    result = subprocess.run(
        ["python", "-c", code],
        capture_output=True, text=True, timeout=30
    )
    return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
```

## Planning 能力

### Chain-of-Thought (CoT)

让模型逐步推理，而非直接给出答案：

```
问：如果一家公司去年营收 100 亿，今年增长 20%，其中海外业务占 40%，海外营收是多少？

# 无 CoT：48亿（可能直接答错）

# 有 CoT：
# 步骤1：今年总营收 = 100 × 1.2 = 120 亿
# 步骤2：海外营收 = 120 × 0.4 = 48 亿
# 答案：48 亿
```

### Tree-of-Thought (ToT)

对于需要探索多条推理路径的问题，ToT 让模型生成多个可能的思路，评估每条路径的前景，选择最优路径继续：

```
             问题
          /   |   \
       思路A  思路B  思路C
       /  \    |     |
      A1  A2   B1   C1  ← 评估每个节点
      |        |
     深入     深入   ← 只展开有前景的分支
      |
    最终答案
```

### Plan-and-Solve

先制定整体计划，再逐步执行——特别适合多步骤任务：

```python
def plan_and_solve(question: str, llm, tools):
    """
    Plan-and-Solve 策略：
    1. 让 LLM 先制定计划（分解为子任务）
    2. 按顺序执行每个子任务
    3. 综合所有子任务的结果生成最终答案
    """
    # 第一步：规划
    plan_prompt = f"""请将以下问题分解为可执行的子任务：
    问题：{question}
    输出格式：
    1. 子任务1
    2. 子任务2
    ...
    """
    plan = llm.generate(plan_prompt)
    
    # 第二步：逐步执行
    results = []
    for subtask in parse_plan(plan):
        result = execute_subtask(subtask, llm, tools)
        results.append(result)
    
    # 第三步：综合
    synthesis_prompt = f"原始问题：{question}\n子任务结果：{results}\n请综合以上结果给出最终答案。"
    return llm.generate(synthesis_prompt)
```

## Memory 机制

### 短期记忆 (Context Window)

模型的上下文窗口就是它的"工作记忆"——当前对话的所有内容都在这里。

| 模型 | Context Window | 等效字数 |
|------|---------------|---------|
| GPT-4 | 128K tokens | ~10 万字 |
| Claude 3 | 200K tokens | ~15 万字 |
| Llama 3 | 128K tokens | ~10 万字 |

**问题：** 即使窗口足够大，模型对长上下文中间部分的关注度也会下降（"lost in the middle" 现象）。

### 长期记忆 (向量数据库)

将历史对话和重要信息编码为向量存入向量数据库，需要时检索：

```python
class AgentMemory:
    """Agent 的长期记忆系统"""
    
    def __init__(self, embed_model, vector_store):
        self.embed_model = embed_model
        self.vector_store = vector_store
    
    def store(self, text: str, metadata: dict = None):
        """将一条信息存入长期记忆"""
        embedding = self.embed_model.encode(text)
        self.vector_store.add(embedding, text, metadata)
    
    def recall(self, query: str, top_k: int = 5):
        """根据当前问题检索相关的历史记忆"""
        query_embedding = self.embed_model.encode(query)
        results = self.vector_store.search(query_embedding, top_k)
        return results
    
    def summarize_and_store(self, conversation: list, llm):
        """将长对话压缩为摘要后存储，节省检索空间"""
        summary = llm.generate(f"请用 3 句话总结以下对话的关键信息：\n{conversation}")
        self.store(summary, metadata={"type": "summary"})
```

### 对话摘要

当对话超过上下文窗口时，用摘要压缩历史：

```
对话轮次 1-20 → 摘要："用户讨论了 RAG 的检索策略，重点关注混合检索..."
对话轮次 21-30 → 保留完整内容
```

这样既保留了早期对话的关键信息，又腾出了窗口空间给最新的交互。

## Multi-Agent 系统

### 多智能体协作模式

当任务足够复杂时，单个 Agent 可能力不从心。Multi-Agent 通过**分工协作**来解决复杂问题。

#### 主从架构（Orchestrator-Worker）

```
         ┌───────────────────┐
         │ Orchestrator Agent│
         │ (Task Assignment) │
         └────────┬──────────┘
        ┌─────────┼──────────┐
        ▼         ▼          ▼
   ┌─────────┐ ┌─────────┐ ┌──────────┐
   │ Search  │ │  Code   │ │ Analysis │
   │  Agent  │ │  Agent  │ │  Agent   │
   └─────────┘ └─────────┘ └──────────┘
```

#### 辩论架构（Debate）

多个 Agent 对同一问题给出不同观点，通过辩论达成共识：

```
Agent A: 我认为答案是 X，理由是...
Agent B: 我不同意，X 忽略了...，答案应该是 Y
Agent A: 你说的有道理，但考虑到...
Judge:   综合双方观点，最终答案是...
```

#### 流水线架构（Pipeline）

```
需求分析 Agent → 代码生成 Agent → 代码审查 Agent → 测试 Agent
```

### 角色分工

Multi-Agent 系统中，每个 Agent 有明确的角色和职责：

```python
# CrewAI 风格的角色定义
agents = {
    "researcher": {
        "role": "研究员",
        "goal": "搜索和整理与任务相关的信息",
        "backstory": "你是一位经验丰富的研究员，擅长从海量信息中提取关键洞察。",
        "tools": ["web_search", "document_reader"]
    },
    "writer": {
        "role": "撰稿人",
        "goal": "基于研究结果撰写高质量的报告",
        "backstory": "你是一位专业的技术写作者，擅长将复杂概念转化为清晰的文字。",
        "tools": ["text_editor"]
    },
    "reviewer": {
        "role": "审稿人",
        "goal": "检查报告的准确性、完整性和可读性",
        "backstory": "你是一位严谨的审稿人，对事实准确性有极高要求。",
        "tools": ["fact_checker"]
    }
}
```

### 通信协议

Agent 之间如何交换信息是 Multi-Agent 系统的关键设计决策：

- **消息传递**：Agent 通过结构化消息通信（AutoGen 采用此方式）
- **共享记忆**：所有 Agent 读写同一个知识库（黑板架构）
- **工作流图**：预定义的 DAG 结构控制信息流（LangGraph 采用此方式）

### 代表性框架

| 框架 | 特点 | 适用场景 |
|------|------|---------|
| **AutoGen** (Microsoft) | 对话驱动，灵活的 Agent 交互模式 | 通用多 Agent 对话 |
| **CrewAI** | 角色扮演，类似团队协作 | 创意写作、研究报告 |
| **LangGraph** | 基于图的工作流定义，状态机 | 复杂业务流程 |
| **MetaGPT** | 模拟软件公司的多角色协作 | 软件开发自动化 |

## Agent 评估与安全

### Agent Bench

评估 Agent 能力的维度：

| 评估维度 | 指标 | 说明 |
|---------|------|------|
| **任务完成率** | Success Rate | 最终是否正确完成任务 |
| **效率** | Steps to Complete | 使用了多少步达成目标 |
| **工具使用** | Tool Accuracy | 是否选择了正确的工具和参数 |
| **鲁棒性** | Error Recovery | 工具调用失败后能否恢复 |
| **安全性** | Harmful Actions | 是否执行了危险操作 |

### 工具调用安全

Agent 能调用外部工具——这意味着它有可能执行危险操作：

```python
# 安全设计原则
class SafeToolExecutor:
    """带安全检查的工具执行器"""
    
    def __init__(self, tools, allowed_actions=None):
        self.tools = tools
        self.allowed_actions = allowed_actions or set()
    
    def execute(self, tool_name: str, args: dict) -> str:
        # 1. 检查工具是否在允许列表中
        if tool_name not in self.allowed_actions:
            return f"安全拒绝：工具 '{tool_name}' 未被授权"
        
        # 2. 参数合法性检查
        if not self._validate_args(tool_name, args):
            return "安全拒绝：参数不合法"
        
        # 3. 高危操作需要人工确认
        if self._is_high_risk(tool_name, args):
            approval = self._request_human_approval(tool_name, args)
            if not approval:
                return "操作已被人工拒绝"
        
        # 4. 在沙箱中执行
        return self._sandboxed_execute(tool_name, args)
```

核心原则：

- **最小权限**：Agent 只能访问完成任务所需的最少工具
- **沙箱执行**：代码执行在隔离环境中，不能访问宿主系统
- **人工审批**：高危操作（删除文件、发送邮件、执行支付）需要人工确认
- **操作日志**：记录所有工具调用，便于审计和回溯

### 幻觉与错误传播

Agent 系统中，幻觉问题更为严重——因为错误不仅停留在文本层面，还可能导致**错误的行动**：

```
Thought: 用户的订单号应该是 #12345（实际是幻觉）
Action: cancel_order(order_id="12345")  ← 取消了错误的订单！
Observation: 订单 #12345 已取消
```

缓解策略：

1. **确认机制**：关键操作前要求用户确认
2. **可逆设计**：所有操作应可回滚
3. **多 Agent 交叉验证**：一个 Agent 执行，另一个 Agent 验证
4. **Grounding**：将 Agent 的推理锚定在检索到的真实数据上

## 苏格拉底时刻

停下来思考以下问题，不急于查看答案：

::: details 1. ReAct 中如果模型'过度自信'，跳过检索直接回答怎么办？
这是 Agent 系统中的经典问题。可以通过以下机制缓解：(1) 在 system prompt 中强调"对于事实性问题，必须先搜索再回答"；(2) 设计"强制检索"机制——对特定类型的问题自动触发搜索，不依赖模型判断；(3) 使用 Self-RAG 的思路，在生成答案后让模型自我评估"这个答案有证据支持吗？"；(4) 结合置信度估计，当模型不确定时自动触发检索。
:::


::: details 2. Function Calling 输出了语法错误的 JSON 怎么办？
这在实践中是真实存在的问题，尤其是较小的模型。处理策略包括：(1) **重试**——最简单，让模型重新生成，通常第二次会正确；(2) **修复**——用 JSON 修复库（如 `json-repair`）尝试自动修复常见错误（缺失引号、多余逗号）；(3) **约束解码**——在生成阶段用有限状态机约束输出必须符合 JSON 语法（如 vLLM 的 guided generation）；(4) **降级**——放弃工具调用，用自然语言回答。
:::


::: details 3. Multi-Agent 系统中协调者 Agent 理解有偏差怎么办？
"单点失败"是主从架构的固有风险。缓解方案：(1) 让协调者输出计划后，各 Worker Agent 可以"反驳"——如果子任务分配不合理，Worker 可以提出修改建议；(2) 使用辩论架构替代主从架构，避免单点依赖；(3) 在协调者和 Worker 之间增加"计划审查"环节；(4) 限制协调者的权限，它只负责分工，不参与具体执行。
:::


::: details 4. 如果底层模型从 GPT-4 换成 7B 小模型，Agent 能力如何退化？
从实践来看，退化是渐进但显著的：**工具选择准确率**最先下降（小模型难以理解复杂的 JSON Schema）；其次是**多步推理能力**（在第 3-4 步后开始"跑偏"）；最后是**指令遵循**（无法严格按 ReAct 格式输出）。缓解方案：减少可用工具数量、简化工具描述、减少推理步数上限、使用更严格的 prompt 模板。本质上，Agent 能力是 LLM 基础能力的放大器——基座弱，Agent 也弱。
:::


## Function Calling 深度实战

### Function Calling 的本质

Function Calling 的核心思想是：**让 LLM 不直接执行操作，而是结构化地输出工具调用参数**，由应用层负责真正的执行。

```
传统方式：用户 → LLM 生成自然语言描述的操作 → 正则解析（脆弱）→ 执行
Function Calling：用户 → LLM 生成结构化 JSON → 直接解析 → 执行
```

本质上，Function Calling 把 LLM 变成了一个**意图识别 + 参数提取**引擎。模型通过 SFT（有监督微调）学会了根据 JSON Schema 描述和用户意图，输出符合格式的函数调用。这不是"理解"了工具，而是学会了一种**结构化输出模式**。

### OpenAI Function Calling API 完整示例

```python
import json
from openai import OpenAI

client = OpenAI()

# ===== 第一步：定义工具（JSON Schema） =====
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "获取指定股票的当前价格和涨跌幅。适用于用户询问股价、行情等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "股票代码，如 'AAPL'、'600519.SH'"
                    },
                    "market": {
                        "type": "string",
                        "enum": ["US", "CN", "HK"],
                        "description": "市场，US=美股，CN=A股，HK=港股"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_reminder",
            "description": "为用户创建一个定时提醒。",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "提醒内容"},
                    "time": {"type": "string", "description": "提醒时间，ISO 8601 格式"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "优先级，默认 medium"
                    }
                },
                "required": ["message", "time"]
            }
        }
    }
]

# ===== 第二步：发送请求，让模型决定是否调用函数 =====
messages = [{"role": "user", "content": "帮我查一下苹果公司的股价"}]

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice="auto"  # "auto" | "none" | {"type": "function", "function": {"name": "xxx"}}
)

assistant_message = response.choices[0].message

# ===== 第三步：检查模型是否要调用函数 =====
if assistant_message.tool_calls:
    # 模型决定调用函数
    tool_call = assistant_message.tool_calls[0]
    func_name = tool_call.function.name          # "get_stock_price"
    func_args = json.loads(tool_call.function.arguments)  # {"symbol": "AAPL", "market": "US"}
    
    print(f"模型要调用: {func_name}({func_args})")
    
    # ===== 第四步：应用层执行函数 =====
    # 这里是真正的业务逻辑，模型不参与执行
    def get_stock_price(symbol, market="US"):
        # 实际调用股票 API
        return json.dumps({"symbol": symbol, "price": 178.52, "change": "+1.2%"})
    
    result = get_stock_price(**func_args)
    
    # ===== 第五步：将结果返回给模型，生成最终回答 =====
    messages.append(assistant_message)  # 包含 tool_calls 的 assistant 消息
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,  # 必须匹配 tool_call 的 id
        "content": result
    })
    
    final_response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools
    )
    print(final_response.choices[0].message.content)
    # 输出："苹果公司（AAPL）当前股价为 178.52 美元，今日涨幅 +1.2%。"
else:
    # 模型认为不需要调用函数，直接回答
    print(assistant_message.content)
```

### 并行函数调用（Parallel Function Calling）

当用户的请求涉及多个独立的工具调用时，模型会在一次响应中返回多个 `tool_calls`，应用层可以并行执行以降低延迟。

```python
import asyncio
import json

# 用户问："北京和东京的天气怎么样？顺便帮我查下美元兑日元汇率"
# 模型返回 3 个 tool_calls：
#   tool_calls[0]: get_weather(city="北京")
#   tool_calls[1]: get_weather(city="东京")
#   tool_calls[2]: get_exchange_rate(from="USD", to="JPY")

async def handle_parallel_tool_calls(assistant_message, tool_executors):
    """并行执行多个工具调用"""
    tool_calls = assistant_message.tool_calls
    
    async def execute_one(tc):
        func_name = tc.function.name
        func_args = json.loads(tc.function.arguments)
        executor = tool_executors[func_name]
        result = await executor(**func_args)
        return {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result, ensure_ascii=False)
        }
    
    # 并行执行所有工具调用
    tool_messages = await asyncio.gather(*[execute_one(tc) for tc in tool_calls])
    return list(tool_messages)

# 关键点：
# 1. 所有 tool 消息的 tool_call_id 必须与对应的 tool_call 匹配
# 2. 所有 tool 消息必须在同一轮中全部返回，不能只返回部分
# 3. tool 消息的顺序不影响结果，但建议与 tool_calls 顺序一致
```

### 多步骤函数调用链

复杂任务往往需要多轮函数调用，每一轮的结果决定下一轮调用什么。

```python
def multi_step_function_calling(user_query, client, tools, max_rounds=5):
    """
    多步骤函数调用：循环执行直到模型不再需要调用工具
    
    典型场景：
    用户："帮我订今晚北京到上海的高铁，最便宜的那趟"
    第1轮：search_trains(from="北京", to="上海", date="今天") → 返回车次列表
    第2轮：sort_by_price(trains=...) → 返回最便宜的车次
    第3轮：book_ticket(train_id="G123", ...) → 返回订单信息
    """
    messages = [{"role": "user", "content": user_query}]
    
    for round_idx in range(max_rounds):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        assistant_msg = response.choices[0].message
        messages.append(assistant_msg)
        
        # 如果模型不再调用工具，返回最终回答
        if not assistant_msg.tool_calls:
            return assistant_msg.content
        
        # 执行本轮所有工具调用
        for tool_call in assistant_msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            
            result = execute_function(func_name, func_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })
    
    return "达到最大调用轮数，任务未完成。"
```

### 错误处理和重试策略

生产环境中，函数调用可能因为多种原因失败。健壮的错误处理至关重要。

```python
import json
import time
from json_repair import repair_json  # pip install json-repair

class RobustFunctionCaller:
    """带错误处理和重试的 Function Calling 执行器"""
    
    def __init__(self, client, tools, max_retries=3):
        self.client = client
        self.tools = tools
        self.max_retries = max_retries
    
    def _safe_parse_arguments(self, raw_arguments: str) -> dict:
        """安全解析函数参数 JSON，处理常见的格式错误"""
        try:
            return json.loads(raw_arguments)
        except json.JSONDecodeError:
            # 尝试自动修复 JSON（缺失引号、尾逗号等）
            try:
                repaired = repair_json(raw_arguments)
                return json.loads(repaired)
            except Exception:
                raise ValueError(f"无法解析函数参数: {raw_arguments}")
    
    def _execute_with_retry(self, func_name, func_args, retries=3):
        """带重试的工具执行"""
        last_error = None
        for attempt in range(retries):
            try:
                result = self.tool_registry[func_name](**func_args)
                return {"status": "success", "data": result}
            except Exception as e:
                last_error = str(e)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
        return {"status": "error", "error": last_error}
    
    def call(self, messages):
        """执行一轮函数调用，包含完整的错误处理"""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )
        
        assistant_msg = response.choices[0].message
        
        if not assistant_msg.tool_calls:
            return assistant_msg.content, messages
        
        messages.append(assistant_msg)
        
        for tool_call in assistant_msg.tool_calls:
            try:
                func_args = self._safe_parse_arguments(tool_call.function.arguments)
                result = self._execute_with_retry(tool_call.function.name, func_args)
            except ValueError as e:
                # JSON 解析彻底失败，告知模型重新生成
                result = {"status": "error", "error": str(e)}
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })
        
        return None, messages  # 返回 None 表示需要继续循环
```

### tool_choice 参数详解

`tool_choice` 控制模型是否调用函数以及调用哪个函数：

| 值 | 行为 | 适用场景 |
|---|------|---------|
| `"auto"` | 模型自行决定是否调用 | 通用场景，推荐默认值 |
| `"none"` | 禁止调用任何函数 | 需要模型纯文本回答时 |
| `"required"` | 必须调用某个函数 | 确定需要工具但不指定哪个 |
| `{"type": "function", "function": {"name": "xxx"}}` | 强制调用指定函数 | 明确知道该用哪个工具时 |

## Human-in-the-loop 模式

### 为什么需要人机协作

Agent 的自主性是一把双刃剑——越自主，风险越大。以下场景必须引入人类参与：

- **安全关键操作**：删除数据、转账支付、发送邮件——一旦执行无法撤回
- **高不确定性决策**：模型置信度低、多个候选方案难以取舍
- **合规要求**：金融、医疗等行业要求关键决策有人类审批记录
- **成本敏感操作**：调用付费 API、消耗大量计算资源

```
全自主 Agent（危险）          人机协作 Agent（安全）
用户 → Agent → 执行          用户 → Agent → 提交计划 → 人类审批 → 执行
     无人监管                          ↑                      │
                                       └── 如被拒绝，修改计划 ←┘
```

### 审批模式（Approval Pattern）

Agent 生成行动计划后，暂停执行，等待人类审批。

```python
import enum
from dataclasses import dataclass, field
from typing import Any

class ApprovalStatus(enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"

@dataclass
class ActionProposal:
    """Agent 提交的行动提案"""
    action_type: str              # 要调用的工具名
    parameters: dict              # 工具参数
    reasoning: str                # Agent 的推理过程
    risk_level: str = "low"       # 风险等级：low / medium / high
    status: ApprovalStatus = ApprovalStatus.PENDING

class HumanInTheLoopAgent:
    """带人类审批的 Agent"""
    
    def __init__(self, llm, tools, risk_threshold="medium"):
        self.llm = llm
        self.tools = tools
        self.risk_threshold = risk_threshold  # 超过此风险等级需要审批
        self.action_history = []
    
    def _assess_risk(self, action_type: str, parameters: dict) -> str:
        """评估操作的风险等级"""
        high_risk_actions = {"delete_file", "send_email", "execute_payment", "drop_table"}
        medium_risk_actions = {"update_record", "create_user", "modify_config"}
        
        if action_type in high_risk_actions:
            return "high"
        elif action_type in medium_risk_actions:
            return "medium"
        return "low"
    
    def _needs_approval(self, risk_level: str) -> bool:
        """判断是否需要人类审批"""
        levels = {"low": 0, "medium": 1, "high": 2}
        return levels[risk_level] >= levels[self.risk_threshold]
    
    def propose_action(self, action_type, parameters, reasoning):
        """提交行动提案"""
        risk = self._assess_risk(action_type, parameters)
        proposal = ActionProposal(
            action_type=action_type,
            parameters=parameters,
            reasoning=reasoning,
            risk_level=risk
        )
        
        if self._needs_approval(risk):
            # 暂停，等待人类审批
            print(f"\n⚠️ 需要审批 [{risk}风险]")
            print(f"  操作: {action_type}({parameters})")
            print(f"  理由: {reasoning}")
            return proposal  # 返回提案，等待外部调用 approve/reject
        else:
            # 低风险，自动执行
            proposal.status = ApprovalStatus.APPROVED
            return self._execute(proposal)
    
    def approve(self, proposal: ActionProposal, modifications=None):
        """人类审批通过（可附带修改）"""
        if modifications:
            proposal.parameters.update(modifications)
            proposal.status = ApprovalStatus.MODIFIED
        else:
            proposal.status = ApprovalStatus.APPROVED
        return self._execute(proposal)
    
    def reject(self, proposal: ActionProposal, feedback: str = ""):
        """人类拒绝操作"""
        proposal.status = ApprovalStatus.REJECTED
        self.action_history.append(proposal)
        # 将拒绝反馈传回 Agent，让它调整策略
        return self._replan(proposal, feedback)
    
    def _execute(self, proposal):
        """执行已批准的操作"""
        tool = self.tools[proposal.action_type]
        result = tool(**proposal.parameters)
        self.action_history.append(proposal)
        return result
    
    def _replan(self, rejected_proposal, feedback):
        """根据人类反馈重新规划"""
        prompt = f"""你之前的操作计划被拒绝了。
被拒绝的操作: {rejected_proposal.action_type}({rejected_proposal.parameters})
拒绝理由: {feedback}
请提出一个新的方案。"""
        return self.llm.generate(prompt)
```

### 中断/恢复机制

长时间运行的 Agent 任务需要支持中断和恢复——人类可能需要离开、可能需要等待外部审批。

```python
import json
import uuid
from datetime import datetime

class CheckpointableAgent:
    """支持中断/恢复的 Agent"""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.session_id = str(uuid.uuid4())
    
    def save_checkpoint(self, state: dict, filepath: str):
        """保存当前执行状态"""
        checkpoint = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "messages": state["messages"],
            "current_step": state["step"],
            "pending_approvals": state.get("pending_approvals", []),
            "completed_actions": state.get("completed_actions", [])
        }
        with open(filepath, "w") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    def load_checkpoint(self, filepath: str) -> dict:
        """从检查点恢复执行状态"""
        with open(filepath, "r") as f:
            checkpoint = json.load(f)
        self.session_id = checkpoint["session_id"]
        return {
            "messages": checkpoint["messages"],
            "step": checkpoint["current_step"],
            "pending_approvals": checkpoint["pending_approvals"],
            "completed_actions": checkpoint["completed_actions"]
        }
    
    def run_with_checkpoints(self, query, checkpoint_dir="./checkpoints"):
        """运行任务，在每个需要审批的节点自动保存检查点"""
        state = {"messages": [{"role": "user", "content": query}], "step": 0,
                 "pending_approvals": [], "completed_actions": []}
        
        while state["step"] < 20:
            response = self.llm.chat(state["messages"])
            
            if self._needs_human_input(response):
                # 保存检查点，暂停执行
                self.save_checkpoint(state, f"{checkpoint_dir}/{self.session_id}.json")
                return {"status": "paused", "reason": "需要人类审批", "session_id": self.session_id}
            
            # 继续执行...
            state["step"] += 1
        
        return {"status": "completed", "result": state}
```

### 反馈循环：人类纠正与 Agent 学习

人类的纠正不仅影响当前任务，还可以形成长期记忆，提升 Agent 在未来类似场景中的表现。

```python
class FeedbackLoop:
    """收集人类反馈，形成 Agent 的长期经验"""
    
    def __init__(self):
        self.feedback_store = []  # 生产中应使用持久化存储
    
    def record_correction(self, agent_action, human_correction, context):
        """记录人类纠正"""
        self.feedback_store.append({
            "context": context,
            "agent_proposed": agent_action,
            "human_corrected": human_correction,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_relevant_feedback(self, current_context, top_k=3):
        """检索与当前场景相似的历史纠正（可用向量检索）"""
        # 简化实现：实际应用中用 Embedding 相似度匹配
        return self.feedback_store[-top_k:]
    
    def build_guidance_prompt(self, current_context):
        """将历史反馈转化为 system prompt 中的指导规则"""
        relevant = self.get_relevant_feedback(current_context)
        if not relevant:
            return ""
        
        rules = []
        for fb in relevant:
            rules.append(
                f"- 在类似 '{fb['context']}' 的场景中，"
                f"不要 {fb['agent_proposed']}，而应该 {fb['human_corrected']}"
            )
        return "基于历史经验，请注意以下规则：\n" + "\n".join(rules)
```

### 实际应用场景

| 场景 | 人类角色 | 审批粒度 |
|------|---------|---------|
| **客服工单处理** | 审批退款/赔偿方案 | 超过金额阈值时审批 |
| **代码部署** | 审批 CI/CD 流水线操作 | 生产环境部署必须审批 |
| **数据分析** | 确认分析方向和假设 | SQL 查询包含 DELETE/UPDATE 时审批 |
| **内容生成** | 审核生成的营销文案 | 所有对外发布内容必须审批 |
| **医疗辅助** | 医生确认诊断建议 | 所有诊断和处方建议必须审批 |

## Multi-Agent 架构深度

### Supervisor 模式（中心协调者）

一个中心 Agent 负责理解任务、分配子任务、汇总结果，其他 Agent 作为专业 Worker 执行具体工作。

```
                   ┌──────────────────────┐
                   │   Supervisor Agent   │
                   │  Parse → Assign →   │
                   │  Aggregate → Reply  │
                   └────────┬─────────────┘
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       ┌──────────────┐ ┌───────────┐ ┌───────────┐
       │Research Agent│ │Code Agent │ │Data Agent │
       │ (Info Search)│ │(Write Code)│ │(Analysis) │
       └──────────────┘ └───────────┘ └───────────┘
```

**优点：** 控制流清晰，易于调试和监控，Supervisor 可以做全局优化

**缺点：** Supervisor 是单点故障，其理解偏差会影响所有 Worker；不适合需要 Worker 之间频繁交互的场景

```python
class SupervisorAgent:
    """Supervisor 模式：中心协调者管理多个 Worker Agent"""
    
    def __init__(self, llm, workers: dict):
        self.llm = llm
        self.workers = workers  # {"name": WorkerAgent}
    
    def plan(self, task: str) -> list:
        """将任务分解为子任务并分配给 Worker"""
        worker_descriptions = "\n".join([
            f"- {name}: {w.description}" for name, w in self.workers.items()
        ])
        prompt = f"""你是一个任务协调者。根据以下可用的工作者，将任务分解为子任务。

可用工作者：
{worker_descriptions}

任务：{task}

请输出 JSON 格式的子任务列表：
[{{"worker": "worker_name", "subtask": "具体子任务描述", "depends_on": []}}]
"""
        plan = self.llm.generate(prompt)
        return json.loads(plan)
    
    def execute(self, task: str) -> str:
        """执行完整流程：规划 → 分配 → 汇总"""
        subtasks = self.plan(task)
        results = {}
        
        # 按依赖关系排序执行
        for subtask in self._topological_sort(subtasks):
            worker = self.workers[subtask["worker"]]
            # 注入依赖任务的结果作为上下文
            context = {dep: results[dep] for dep in subtask.get("depends_on", []) if dep in results}
            result = worker.execute(subtask["subtask"], context)
            results[subtask["subtask"]] = result
        
        # 汇总所有结果
        summary_prompt = f"原始任务：{task}\n\n各子任务结果：\n{json.dumps(results, ensure_ascii=False, indent=2)}\n\n请综合以上结果，给出最终回答。"
        return self.llm.generate(summary_prompt)
```

### Hierarchical 模式（分层管理）

类似企业组织架构，分为多个管理层级。顶层 Agent 管理中层 Agent，中层 Agent 再管理底层执行 Agent。

```
                    ┌───────────────────┐
                    │    CEO Agent      │
                    │ Strategy/Planning │
                    └────────┬──────────┘
              ┌──────────────┼───────────────┐
              ▼              ▼               ▼
      ┌──────────────┐ ┌───────────┐ ┌────────────┐
      │  Dev Manager │ │ Marketing │ │    Ops     │
      │  (Dev Tasks) │ │ (Campaigns)│ │(Infra Mgmt)│
      └──────┬───────┘ └────┬──────┘ └─────┬──────┘
        ┌────┼────┐     ┌───┼───┐     ┌────┼────┐
        ▼    ▼    ▼     ▼   ▼   ▼     ▼    ▼    ▼
      Front Back  QA  Copy Design SEO Monitor Deploy Sec
```

**适用场景：** 大型复杂项目，如自动化软件开发（MetaGPT 就采用类似架构）

**关键设计：** 每一层只需要和直接上下级通信，降低通信复杂度（从 O(n²) 降到 O(n)）

### Peer-to-Peer 模式（平等协作）

没有中心协调者，每个 Agent 地位平等，通过共享消息总线或直接通信来协作。

```
      ┌──────────┐  Messages  ┌──────────┐
      │ Agent A  │◄──────────►│ Agent B  │
      │ (Search) │            │(Analysis)│
      └─────┬────┘            └─────┬────┘
            │      ┌──────────┐     │
            └─────►│ Agent C  │◄────┘
                   │ (Writer) │
                   └──────────┘
```

**典型实现：** 辩论模式——多个 Agent 对同一问题发表观点，互相质疑和补充。

```python
class PeerDebateSystem:
    """Peer-to-Peer 辩论系统"""
    
    def __init__(self, agents: list, judge_llm, max_rounds=3):
        self.agents = agents      # 参与辩论的 Agent 列表
        self.judge_llm = judge_llm
        self.max_rounds = max_rounds
    
    def debate(self, question: str) -> str:
        debate_history = []
        
        for round_idx in range(self.max_rounds):
            round_responses = []
            for agent in self.agents:
                # 每个 Agent 基于辩论历史发表观点
                response = agent.respond(question, debate_history)
                round_responses.append({
                    "agent": agent.name,
                    "position": response
                })
            debate_history.extend(round_responses)
        
        # 裁判综合所有观点给出最终结论
        judge_prompt = f"""问题：{question}

辩论记录：
{json.dumps(debate_history, ensure_ascii=False, indent=2)}

请综合各方观点，给出最终结论。"""
        return self.judge_llm.generate(judge_prompt)
```

### 三种模式对比

| 维度 | Supervisor | Hierarchical | Peer-to-Peer |
|------|-----------|-------------|-------------|
| **控制方式** | 集中式 | 分层式 | 分布式 |
| **通信复杂度** | O(n) | O(n) | O(n²) |
| **容错性** | 低（单点故障） | 中 | 高 |
| **适用规模** | 3-8 个 Agent | 10+ 个 Agent | 2-5 个 Agent |
| **典型框架** | LangGraph | MetaGPT | AutoGen |

### 通信协议

Agent 之间的通信协议是 Multi-Agent 系统的基础设施。

#### MCP（Model Context Protocol）

MCP 是 Anthropic 提出的开放协议，旨在标准化 LLM 与外部工具/数据源的连接方式。

```
┌──────────────┐    MCP Protocol     ┌──────────────┐
│  LLM App     │ ◄──────────────────►│  MCP Server  │
│ (MCP Client) │   JSON-RPC over     │ (Tools/Data) │
│              │   stdio / SSE       │              │
└──────────────┘                     └──────────────┘
```

MCP 的核心价值：
- **标准化接口**：一个 MCP Server 可以被任何支持 MCP 的 LLM 应用调用
- **三种原语**：Tools（工具调用）、Resources（数据资源）、Prompts（提示模板）
- **双向通信**：Server 可以主动推送通知给 Client

#### A2A（Agent-to-Agent Protocol）

Google 提出的 A2A 协议，专注于 Agent 之间的互操作：

```
┌────────────┐  A2A Protocol  ┌────────────┐
│  Agent A   │ ◄─────────────►│  Agent B   │
│ (Client)   │  HTTP + JSON   │ (Server)   │
└────────────┘                └────────────┘
```

A2A 定义了：
- **Agent Card**：描述 Agent 的能力、端点、认证方式（类似 API 说明文档）
- **Task 生命周期**：submitted → working → input-required → completed / failed
- **Streaming**：支持 SSE 实时推送任务进度

#### MCP vs A2A

| 维度 | MCP | A2A |
|------|-----|-----|
| **连接对象** | LLM ↔ 工具/数据 | Agent ↔ Agent |
| **类比** | USB 接口（连接外设） | HTTP 协议（连接服务） |
| **关系** | 互补，非竞争 | 互补，非竞争 |
| **典型场景** | Agent 调用数据库、文件系统 | 不同厂商的 Agent 协作 |

### 代码示例：简单的双 Agent 协作系统

```python
import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentMessage:
    """Agent 之间的通信消息"""
    sender: str
    receiver: str
    content: str
    msg_type: str = "request"  # request / response / info

class BaseAgent:
    """Agent 基类"""
    
    def __init__(self, name: str, description: str, llm):
        self.name = name
        self.description = description
        self.llm = llm
        self.inbox = []
    
    def receive(self, message: AgentMessage):
        self.inbox.append(message)
    
    def process(self) -> Optional[AgentMessage]:
        raise NotImplementedError

class ResearchAgent(BaseAgent):
    """研究 Agent：负责信息收集和分析"""
    
    def __init__(self, llm, search_tool):
        super().__init__("researcher", "负责搜索和分析信息", llm)
        self.search_tool = search_tool
    
    def process(self) -> Optional[AgentMessage]:
        if not self.inbox:
            return None
        msg = self.inbox.pop(0)
        
        # 搜索相关信息
        search_results = self.search_tool(msg.content)
        
        # 让 LLM 分析搜索结果
        analysis = self.llm.generate(
            f"请分析以下搜索结果并提取关键信息：\n\n"
            f"主题：{msg.content}\n搜索结果：{search_results}"
        )
        
        return AgentMessage(
            sender=self.name,
            receiver=msg.sender,
            content=analysis,
            msg_type="response"
        )

class WriterAgent(BaseAgent):
    """写作 Agent：负责根据研究结果撰写内容"""
    
    def __init__(self, llm):
        super().__init__("writer", "负责撰写结构化的报告", llm)
    
    def process(self) -> Optional[AgentMessage]:
        if not self.inbox:
            return None
        msg = self.inbox.pop(0)
        
        report = self.llm.generate(
            f"请基于以下研究材料撰写一份简洁的报告：\n\n{msg.content}"
        )
        
        return AgentMessage(
            sender=self.name,
            receiver="user",
            content=report,
            msg_type="response"
        )

class DualAgentSystem:
    """双 Agent 协作系统：研究员搜集信息 → 写手撰写报告"""
    
    def __init__(self, researcher: ResearchAgent, writer: WriterAgent):
        self.researcher = researcher
        self.writer = writer
    
    def run(self, user_query: str) -> str:
        # 第一步：向研究员发送任务
        self.researcher.receive(AgentMessage(
            sender="writer",
            receiver="researcher",
            content=user_query,
            msg_type="request"
        ))
        
        # 第二步：研究员处理并返回分析结果
        research_result = self.researcher.process()
        
        # 第三步：将研究结果发送给写手
        self.writer.receive(research_result)
        
        # 第四步：写手生成最终报告
        final_report = self.writer.process()
        
        return final_report.content

# ===== 使用示例 =====
# researcher = ResearchAgent(llm=my_llm, search_tool=web_search)
# writer = WriterAgent(llm=my_llm)
# system = DualAgentSystem(researcher, writer)
# report = system.run("2024年大模型技术的主要进展有哪些？")
# print(report)
```

## 常见问题 & 面试考点

::: tip 面试高频问题

**Q: ReAct 和 Function Calling 的区别是什么？**
A: ReAct 是一种 Agent 推理框架（Thought-Action-Observation 循环），工具调用用自然语言描述。Function Calling 是一种具体的工具调用协议，输出结构化 JSON。两者不互斥——可以在 ReAct 框架中使用 Function Calling 来执行 Action 步骤，获得更可靠的工具调用。

**Q: Agent 和 RAG 的关系是什么？**
A: RAG 可以看作 Agent 的一个子能力——Agent 可以将"检索知识库"作为一个工具来调用。但 Agent 的能力远不止检索：它还能执行代码、调用 API、管理文件等。当 RAG 无法满足复杂问题时（如需要多步推理、跨数据源查询），Agentic RAG 用 Agent 框架来编排多次检索。

**Q: 如何评估一个 Agent 系统的好坏？**
A: 四个维度：(1) **任务完成率**——最终结果是否正确；(2) **效率**——用了多少步/多少 token；(3) **鲁棒性**——面对工具失败、模糊问题时能否恢复；(4) **安全性**——是否执行了不该执行的操作。不能只看最终结果，过程质量同样重要。

**Q: 为什么 Agent 系统在生产中落地困难？**
A: 主要障碍：(1) 可靠性——LLM 的非确定性输出导致 Agent 行为不可预测；(2) 延迟——多轮 LLM 调用 + 工具执行的累积延迟；(3) 成本——大量 token 消耗；(4) 调试困难——链式推理的错误难以定位。当前的趋势是将"全自主 Agent"降级为"人机协作的半自主 Agent"。
:::


## 推荐资源

- **Yao et al. "ReAct: Synergizing Reasoning and Acting in Language Models"** — ReAct 原始论文
- **Shinn et al. "Reflexion: Language Agents with Verbal Reinforcement Learning"** — Agent 自我反思机制
- **Yao et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"** — 树状推理
- **Wang et al. "A Survey on Large Language Model based Autonomous Agents"** — LLM Agent 综述
- **OpenAI Function Calling 文档** — Function Calling 最佳实践
- **AutoGen / CrewAI / LangGraph 官方文档** — Multi-Agent 框架入门
- **[AgentBench](https://github.com/THUDM/AgentBench)** — Agent 评估基准
