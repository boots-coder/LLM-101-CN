---
title: "Agent 框架实战"
description: "LangGraph/AutoGen/CrewAI 框架对比、GraphRAG、Function Calling API"
topics: [LangGraph, AutoGen, CrewAI, agent-framework, GraphRAG, function-calling, LangSmith]
---
# Agent 开发框架实战

::: info 一句话总结
Agent 框架将 LLM 的推理能力与工具调用、状态管理、多智能体协作整合为可编程的工作流，LangGraph 是当前最成熟的状态机式 Agent 编排框架。
:::


## 在大模型体系中的位置

```
大模型应用层
├── Prompt Engineering（提示工程）
├── RAG（检索增强生成）
├── Agent（智能体）
│   ├── Agent 基础概念 ← 见 agents.md
│   └── Agent 开发框架 ◄── 你在这里
├── Fine-tuning（微调）
└── 评估与对齐
```

在 [Agent 智能体](./agents.md) 一章中我们理解了 Agent 的核心概念——ReAct 循环、工具调用、记忆与规划。本章聚焦**工程实践**：如何用成熟的框架高效构建生产级 Agent 系统。

## Agent 框架全景

### 四大主流框架对比

```
                    抽象层级
          低 ◄──────────────────► 高
          │                        │
    LangGraph                  CrewAI
    (状态机编排)             (角色扮演)
          │                        │
    LlamaIndex Agents          AutoGen
    (数据增强)              (对话驱动)
```

| 框架 | 核心理念 | 优势 | 适用场景 |
|------|----------|------|----------|
| **LangGraph** | 图状态机编排 Agent 工作流 | 精细控制流程、可回溯、可中断 | 复杂多步工作流、生产系统 |
| **AutoGen** | 多 Agent 对话协作 | 多智能体交互简单、代码执行 | 多角色协作、代码生成 |
| **CrewAI** | 角色扮演式任务分配 | 上手快、角色定义直观 | 快速原型、内容生成流水线 |
| **LlamaIndex Agents** | 以数据为中心的 Agent | 与数据索引深度整合 | RAG + Agent 混合场景 |

## LangGraph 深度解析

LangGraph 是 LangChain 团队推出的 Agent 编排框架，核心思想是用**有向图 (Directed Graph)** 描述 Agent 的工作流：节点是计算步骤，边是状态转移。

### 核心概念

```
┌──────────────────────────────────────────────────┐
│            LangGraph Core Concepts               │
│                                                  │
│  StateGraph   - Stateful directed graph          │
│  State        - Shared state across all nodes    │
│  Node         - Function that performs compute   │
│  Edge         - Connection between nodes         │
│  Conditional  - Dynamic routing based on state   │
│    Edge                                          │
│  Checkpointer - Persistence (memory + resume)    │
│                                                  │
└──────────────────────────────────────────────────┘
```

### State 状态模式设计

LangGraph 的 State 是一个 TypedDict，所有节点共享并可以修改它：

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """Agent 的共享状态"""
    # add_messages 是一个 reducer：新消息追加到列表，而非覆盖
    messages: Annotated[list, add_messages]
    # 自定义状态字段
    current_tool: str
    iteration_count: int
    final_answer: str
```

`Annotated[list, add_messages]` 中的 `add_messages` 是一个 **reducer 函数**——它定义了当多个节点向同一个字段写入时如何合并。默认行为是覆盖，`add_messages` 改为追加。

### 最小 LangGraph 示例

```python
from langgraph.graph import StateGraph, START, END

class SimpleState(TypedDict):
    input: str
    output: str

def process(state: SimpleState) -> dict:
    """处理节点：将输入转为大写"""
    return {"output": state["input"].upper()}

# 构建图
graph = StateGraph(SimpleState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

# 编译并运行
app = graph.compile()
result = app.invoke({"input": "hello langgraph"})
print(result["output"])  # "HELLO LANGGRAPH"
```

### 条件边 (Conditional Edge)

条件边是 LangGraph 的核心能力——根据当前状态**动态决定**下一个节点：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

class RouterState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str

def classifier(state: RouterState) -> dict:
    """分类节点：判断用户意图"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    resp = llm.invoke([
        {"role": "system", "content": "判断用户意图，只输出: search / calculate / chitchat"},
        *state["messages"]
    ])
    return {"route": resp.content.strip()}

def search_handler(state: RouterState) -> dict:
    return {"messages": [AIMessage(content="[搜索结果] 这是搜索到的信息...")]}

def calc_handler(state: RouterState) -> dict:
    return {"messages": [AIMessage(content="[计算结果] 42")]}

def chitchat_handler(state: RouterState) -> dict:
    return {"messages": [AIMessage(content="你好！有什么可以帮你的吗？")]}

def route_decision(state: RouterState) -> str:
    """条件路由函数：返回下一个节点名"""
    route = state.get("route", "chitchat")
    if route == "search":
        return "search"
    elif route == "calculate":
        return "calculate"
    else:
        return "chitchat"

# 构建图
graph = StateGraph(RouterState)
graph.add_node("classifier", classifier)
graph.add_node("search", search_handler)
graph.add_node("calculate", calc_handler)
graph.add_node("chitchat", chitchat_handler)

graph.add_edge(START, "classifier")
graph.add_conditional_edges(
    "classifier",
    route_decision,
    {"search": "search", "calculate": "calculate", "chitchat": "chitchat"}
)
graph.add_edge("search", END)
graph.add_edge("calculate", END)
graph.add_edge("chitchat", END)

app = graph.compile()
```

### 构建 ReAct Agent

LangGraph 内置了构建 ReAct Agent 的工具绑定模式：

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 定义工具
@tool
def search_web(query: str) -> str:
    """搜索互联网获取最新信息"""
    # 实际项目中接入搜索 API
    return f"搜索 '{query}' 的结果：这是一段相关信息..."

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)  # 生产环境应使用安全的表达式解析
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    return f"{city}今天晴，气温 25°C，适合出行。"

# 创建 ReAct Agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_react_agent(
    model=llm,
    tools=[search_web, calculator, get_weather],
    state_modifier="你是一个全能助手，可以搜索、计算和查天气。请先思考再行动。"
)

# 运行
result = agent.invoke({
    "messages": [HumanMessage(content="北京今天天气怎么样？如果气温超过 20 度，帮我算一下 23 * 17")]
})
for msg in result["messages"]:
    print(f"[{msg.type}] {msg.content[:100]}")
```

### Memory 机制

#### 短期记忆：Checkpointer

Checkpointer 将每一步的状态持久化，支持**断点续跑**和**对话记忆**：

```python
from langgraph.checkpoint.memory import MemorySaver

# 内存版 Checkpointer（开发用）
memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[search_web, calculator],
    checkpointer=memory  # 启用状态持久化
)

# 第一轮对话
config = {"configurable": {"thread_id": "user-123"}}
result1 = agent.invoke(
    {"messages": [HumanMessage(content="我叫张三，我在北京")]},
    config=config
)

# 第二轮对话——Agent 记得之前的对话
result2 = agent.invoke(
    {"messages": [HumanMessage(content="我叫什么名字？我在哪里？")]},
    config=config
)
# Agent 会回答："你叫张三，你在北京。"
```

#### 长期记忆：外部存储

对于跨会话的长期记忆，需要结合外部存储（向量数据库 / KV 存储）：

```python
from langchain_core.messages import SystemMessage

class LongTermMemory:
    """简单的长期记忆实现"""

    def __init__(self):
        self.user_profiles = {}  # user_id -> {facts}

    def store(self, user_id: str, fact: str):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = []
        self.user_profiles[user_id].append(fact)

    def retrieve(self, user_id: str) -> str:
        facts = self.user_profiles.get(user_id, [])
        if not facts:
            return "暂无该用户的历史信息。"
        return "已知用户信息：\n" + "\n".join(f"- {f}" for f in facts)

    def inject_into_state(self, user_id: str) -> SystemMessage:
        """将长期记忆注入为系统消息"""
        memory_text = self.retrieve(user_id)
        return SystemMessage(content=f"[长期记忆]\n{memory_text}")
```

### Human-in-the-loop 模式

LangGraph 支持在图执行过程中**暂停等待人类审批**：

```python
from langgraph.graph import StateGraph, START, END

class ApprovalState(TypedDict):
    messages: Annotated[list, add_messages]
    action_plan: str
    approved: bool

def plan_action(state: ApprovalState) -> dict:
    """AI 制定行动计划"""
    return {"action_plan": "我打算发送一封邮件给客户，内容是：..."}

def execute_action(state: ApprovalState) -> dict:
    """执行已审批的操作"""
    if state.get("approved"):
        return {"messages": [AIMessage(content=f"已执行：{state['action_plan']}")]}
    return {"messages": [AIMessage(content="操作已被人类拒绝。")]}

def check_approval(state: ApprovalState) -> str:
    return "execute" if state.get("approved") else END

graph = StateGraph(ApprovalState)
graph.add_node("plan", plan_action)
graph.add_node("execute", execute_action)

graph.add_edge(START, "plan")
# interrupt_after="plan" —— 在 plan 节点执行后暂停，等待人类输入
graph.add_conditional_edges("plan", check_approval, {"execute": "execute", END: END})
graph.add_edge("execute", END)

app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_after=["plan"]  # 关键：在 plan 节点后中断
)

# 第一次运行——执行到 plan 后暂停
config = {"configurable": {"thread_id": "approval-1"}}
result = app.invoke({"messages": [HumanMessage(content="帮我联系客户")]}, config)
print(f"待审批计划：{result['action_plan']}")

# 人类审批后继续
app.update_state(config, {"approved": True})
result = app.invoke(None, config)  # 从中断点继续执行
```

### Multi-Agent 架构

#### Supervisor 模式（主管调度）

一个 Supervisor Agent 负责分配任务给 Worker Agent：

```python
from langchain_core.messages import HumanMessage, AIMessage

class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    task_complete: bool

def supervisor(state: MultiAgentState) -> dict:
    """主管 Agent：决定下一步由谁执行"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    resp = llm.invoke([
        {"role": "system", "content": """你是项目主管。根据对话历史决定下一步：
- 如果需要搜索信息，输出 "researcher"
- 如果需要撰写内容，输出 "writer"
- 如果任务已完成，输出 "FINISH"
只输出一个词。"""},
        *state["messages"]
    ])
    next_agent = resp.content.strip().lower()
    return {
        "next_agent": next_agent,
        "task_complete": next_agent == "finish"
    }

def researcher(state: MultiAgentState) -> dict:
    """研究员 Agent：负责信息搜索和分析"""
    llm = ChatOpenAI(model="gpt-4o")
    resp = llm.invoke([
        {"role": "system", "content": "你是研究员，负责搜索和整理信息。基于对话历史进行研究。"},
        *state["messages"]
    ])
    return {"messages": [AIMessage(content=f"[研究员] {resp.content}")]}

def writer(state: MultiAgentState) -> dict:
    """撰稿人 Agent：负责内容创作"""
    llm = ChatOpenAI(model="gpt-4o")
    resp = llm.invoke([
        {"role": "system", "content": "你是撰稿人，负责将研究成果写成文章。"},
        *state["messages"]
    ])
    return {"messages": [AIMessage(content=f"[撰稿人] {resp.content}")]}

def route_to_agent(state: MultiAgentState) -> str:
    if state.get("task_complete"):
        return END
    return state.get("next_agent", "researcher")

# 构建 Multi-Agent 图
graph = StateGraph(MultiAgentState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {"researcher": "researcher", "writer": "writer", END: END}
)
# Worker 完成后回到 Supervisor 重新调度
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")

multi_agent = graph.compile()
```

#### 三种 Multi-Agent 模式对比

```
Supervisor（主管模式）         Hierarchical（层级模式）
┌──────────┐                  ┌──────────┐
│ Supervisor│                  │  Leader   │
└────┬──┬──┘                  └────┬──┬──┘
     │  │                          │  │
  ┌──▼┐ ┌▼──┐                 ┌───▼┐ ┌▼───┐
  │ R │ │ W │                 │Sup1│ │Sup2│
  └───┘ └───┘                 └─┬──┘ └──┬─┘
                               ┌▼┐ ┌▼┐ ┌▼┐ ┌▼┐
                               │A│ │B│ │C│ │D│
                               └─┘ └─┘ └─┘ └─┘

Peer-to-Peer（对等模式）
  ┌───┐ ←→ ┌───┐
  │ A │    │ B │
  └─┬─┘    └─┬─┘
    │    ↕    │
  ┌─▼────────▼─┐
  │     C      │
  └────────────┘
```

### LangSmith 调试与监控

LangSmith 是 LangGraph 的配套可观测性平台：

```python
import os

# 配置 LangSmith（设置环境变量即可自动追踪）
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "my-agent-project"

# 之后所有 LangGraph 调用都会自动上报到 LangSmith
# 可以在 smith.langchain.com 查看：
# - 每个节点的输入/输出
# - Token 使用量和延迟
# - 完整的执行轨迹（trace）
# - 错误定位和回放
```

## AutoGen 简介

AutoGen (微软) 采用**多 Agent 对话**的范式——Agent 之间通过消息传递来协作。

### Agent 对话模式

```python
from autogen import AssistantAgent, UserProxyAgent

# 创建 AI 助手
assistant = AssistantAgent(
    name="coding_assistant",
    llm_config={"model": "gpt-4o", "temperature": 0},
    system_message="你是一个 Python 编程专家，擅长数据分析。"
)

# 创建用户代理（可自动执行代码）
user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",          # 不需要人类介入
    code_execution_config={
        "work_dir": "workspace",
        "use_docker": False
    },
    max_consecutive_auto_reply=5
)

# 发起对话——两个 Agent 自动来回交互
user_proxy.initiate_chat(
    assistant,
    message="请用 Python 分析 iris 数据集，画出散点图矩阵并计算各特征的相关系数。"
)
# assistant 会生成代码 → user_proxy 自动执行 → 看结果 → 反馈 → 继续改进
```

### GroupChat 多代理协作

```python
from autogen import GroupChat, GroupChatManager

# 定义多个专业 Agent
planner = AssistantAgent(
    name="planner",
    system_message="你是项目经理，负责分解任务和协调团队。"
)
coder = AssistantAgent(
    name="coder",
    system_message="你是高级程序员，负责编写高质量代码。"
)
reviewer = AssistantAgent(
    name="reviewer",
    system_message="你是代码审查员，负责检查代码质量和安全性。"
)

# 创建群聊
group_chat = GroupChat(
    agents=[user_proxy, planner, coder, reviewer],
    messages=[],
    max_round=12,
    speaker_selection_method="auto"  # LLM 自动选择下一个发言者
)
manager = GroupChatManager(groupchat=group_chat, llm_config={"model": "gpt-4o"})

# 启动群聊
user_proxy.initiate_chat(
    manager,
    message="开发一个 Python 命令行工具，用于批量压缩 PNG 图片。"
)
```

## GraphRAG

GraphRAG 将**知识图谱**与 LLM 检索结合，解决传统 RAG 在全局问题（如"数据集的主要主题是什么？"）上的短板。

### 与传统 RAG 的对比

| 维度 | 传统 RAG | GraphRAG |
|------|----------|----------|
| **索引结构** | 向量数据库（flat） | 知识图谱（hierarchical） |
| **检索方式** | 语义相似度 | 图遍历 + 社区摘要 |
| **擅长问题** | 局部事实查询 | 全局性/聚合性问题 |
| **幻觉控制** | 靠检索片段 | 靠结构化知识 + 来源追踪 |
| **构建成本** | 低 | 高（需要实体抽取 + 图构建） |

### GraphRAG Pipeline

```
文档 → Entity Extraction → Graph Construction → Community Detection → Query
         (实体抽取)         (图构建)             (社区发现)          (查询)
```

```python
"""GraphRAG 简化实现"""
import json
from dataclasses import dataclass, field

@dataclass
class Entity:
    name: str
    type: str  # person, org, concept, event
    description: str

@dataclass
class Relationship:
    source: str
    target: str
    relation: str
    description: str

@dataclass
class KnowledgeGraph:
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)

def extract_entities(text: str, client) -> tuple[list[Entity], list[Relationship]]:
    """Step 1: 从文本中抽取实体和关系"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": """从文本中抽取实体和关系，JSON 格式：
{
    "entities": [{"name": "...", "type": "person|org|concept|event", "description": "..."}],
    "relationships": [{"source": "...", "target": "...", "relation": "...", "description": "..."}]
}"""},
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    data = json.loads(resp.choices[0].message.content)
    entities = [Entity(**e) for e in data.get("entities", [])]
    relationships = [Relationship(**r) for r in data.get("relationships", [])]
    return entities, relationships

def build_graph(documents: list[str], client) -> KnowledgeGraph:
    """Step 2: 从多个文档构建知识图谱"""
    kg = KnowledgeGraph()
    for doc in documents:
        entities, rels = extract_entities(doc, client)
        kg.entities.extend(entities)
        kg.relationships.extend(rels)
    # 去重（按名称合并同一实体）
    seen = {}
    for e in kg.entities:
        if e.name not in seen:
            seen[e.name] = e
    kg.entities = list(seen.values())
    return kg

def community_summarize(kg: KnowledgeGraph, client) -> list[str]:
    """Step 3: 社区发现 + 摘要（简化版）"""
    # 实际应使用 Leiden 等社区发现算法
    # 这里简化为对所有实体分组摘要
    entity_descriptions = "\n".join(
        f"- {e.name} ({e.type}): {e.description}" for e in kg.entities
    )
    rel_descriptions = "\n".join(
        f"- {r.source} --[{r.relation}]--> {r.target}" for r in kg.relationships
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "基于知识图谱生成 3-5 个主题社区摘要。"},
            {"role": "user", "content": f"实体：\n{entity_descriptions}\n\n关系：\n{rel_descriptions}"}
        ],
        temperature=0
    )
    return resp.choices[0].message.content.split("\n\n")

def graph_rag_query(question: str, kg: KnowledgeGraph, summaries: list[str], client) -> str:
    """Step 4: 基于图谱和社区摘要回答问题"""
    context = "\n\n".join(summaries)
    entities_ctx = "\n".join(f"- {e.name}: {e.description}" for e in kg.entities[:20])

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "基于以下知识图谱信息回答问题。引用具体实体作为证据。"},
            {"role": "user", "content": f"社区摘要：\n{context}\n\n关键实体：\n{entities_ctx}\n\n问题：{question}"}
        ],
        temperature=0
    )
    return resp.choices[0].message.content
```

## Function Calling 深度实战

### OpenAI / Claude / GLM 的 API 对比

三大平台的 Function Calling 遵循类似的模式，但 API 细节有差异：

```python
"""三大平台 Function Calling 对比"""

# ============ OpenAI ============
def openai_function_calling(client, query: str):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询指定城市的天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}],
        tools=tools,
        tool_choice="auto"  # auto / required / none / {"type":"function","function":{"name":"..."}}
    )
    return response

# ============ Claude (Anthropic) ============
def claude_function_calling(client, query: str):
    tools = [
        {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "input_schema": {  # 注意：Claude 用 input_schema 而非 parameters
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    ]
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": query}]
    )
    return response

# ============ GLM (智谱) ============
def glm_function_calling(client, query: str):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询指定城市的天气",
                "parameters": {  # GLM 与 OpenAI 格式一致
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    response = client.chat.completions.create(
        model="glm-4",
        messages=[{"role": "user", "content": query}],
        tools=tools
    )
    return response
```

### Tool 定义规范（JSON Schema）

```python
"""完整的 Tool 定义示例——带嵌套对象和数组"""

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_calendar_event",
        "description": "创建日历事件",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "事件标题"
                },
                "start_time": {
                    "type": "string",
                    "description": "开始时间，ISO 8601 格式，如 2024-03-15T14:00:00"
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "持续时间（分钟）"
                },
                "attendees": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string", "format": "email"}
                        },
                        "required": ["email"]
                    },
                    "description": "参会人列表"
                },
                "recurrence": {
                    "type": "object",
                    "properties": {
                        "frequency": {"type": "string", "enum": ["daily", "weekly", "monthly"]},
                        "count": {"type": "integer", "description": "重复次数"}
                    },
                    "description": "重复规则（可选）"
                }
            },
            "required": ["title", "start_time", "duration_minutes"]
        }
    }
}
```

### 并行调用 / 多步调用

```python
import json

def handle_tool_calls(response, tool_registry: dict, client) -> str:
    """
    处理 Function Calling 的完整流程，支持并行和多步调用
    """
    messages = [{"role": "user", "content": "原始问题..."}]
    msg = response.choices[0].message
    messages.append(msg)

    while msg.tool_calls:
        # 并行执行所有 tool_calls
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            # 调用实际函数
            if func_name in tool_registry:
                result = tool_registry[func_name](**func_args)
            else:
                result = f"未知工具: {func_name}"

            # 将工具结果加入消息
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

        # 让模型处理工具结果——可能触发更多 tool_calls（多步调用）
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=list(tool_registry.values())
        )
        msg = response.choices[0].message
        messages.append(msg)

    return msg.content
```

## 代码实战：带记忆的多工具 Agent

以下用 LangGraph 构建一个完整的生产级 Agent，支持多工具、记忆和流式输出：

```python
"""
LangGraph 多工具 Agent——带短期记忆和长期记忆
"""
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import json
import datetime

# ========== 1. 定义工具 ==========

@tool
def search_knowledge(query: str) -> str:
    """在知识库中搜索信息。用于回答事实性问题。"""
    # 模拟知识库搜索
    knowledge = {
        "python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。",
        "transformer": "Transformer 是 2017 年由 Google 提出的深度学习架构，基于自注意力机制。",
        "langgraph": "LangGraph 是 LangChain 团队推出的 Agent 编排框架，基于有向图实现工作流。"
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return f"未找到关于 '{query}' 的信息。"

@tool
def run_python_code(code: str) -> str:
    """执行 Python 代码并返回结果。用于计算、数据处理等。"""
    try:
        local_vars = {}
        exec(code, {"__builtins__": {}}, local_vars)
        if "result" in local_vars:
            return str(local_vars["result"])
        return "代码执行完成（无 result 变量）"
    except Exception as e:
        return f"执行错误: {type(e).__name__}: {e}"

@tool
def save_note(title: str, content: str) -> str:
    """保存一条笔记到用户的笔记本。"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"已保存笔记 [{title}] ({timestamp})：{content[:50]}..."

@tool
def get_current_time() -> str:
    """获取当前日期和时间。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ========== 2. 定义 State ==========

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_profile: str  # 长期记忆：用户画像

# ========== 3. 定义节点 ==========

TOOLS = [search_knowledge, run_python_code, save_note, get_current_time]
LLM = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(TOOLS)

def agent_node(state: AgentState) -> dict:
    """Agent 主节点：调用 LLM 进行推理"""
    system_msg = SystemMessage(content=f"""你是一个智能助手，拥有以下能力：
1. 搜索知识库回答问题
2. 执行 Python 代码进行计算
3. 保存和管理笔记
4. 查询当前时间

用户画像：{state.get('user_profile', '新用户，暂无信息')}

请根据需要使用工具。如果不需要工具，直接回答。""")

    messages = [system_msg] + state["messages"]
    response = LLM.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """判断是否需要继续调用工具"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

def update_profile(state: AgentState) -> dict:
    """从对话中提取用户信息更新长期记忆（简化版）"""
    messages = state["messages"]
    user_msgs = [m.content for m in messages if isinstance(m, HumanMessage)]
    # 实际项目中应用 LLM 来提取用户偏好
    profile = state.get("user_profile", "")
    new_info = f" | 最近话题: {user_msgs[-1][:30]}" if user_msgs else ""
    return {"user_profile": profile + new_info}

# ========== 4. 构建图 ==========

tool_node = ToolNode(TOOLS)

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("update_profile", update_profile)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: "update_profile"})
graph.add_edge("tools", "agent")  # 工具执行后回到 Agent
graph.add_edge("update_profile", END)

# 编译（带记忆）
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# ========== 5. 运行 ==========

def chat(user_input: str, thread_id: str = "default"):
    """对话接口"""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    # 返回最后一条 AI 消息
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "（无回复）"

# --- 示例对话 ---
if __name__ == "__main__":
    thread = "demo-user-001"

    print("=" * 60)
    print("LangGraph 多工具 Agent Demo")
    print("=" * 60)

    # 多轮对话演示
    queries = [
        "你好！我是张三，我在学习大模型。",
        "帮我搜一下 Transformer 是什么？",
        "用 Python 算一下 2 的 10 次方是多少",
        "帮我保存一条笔记：今天学习了 LangGraph 的基本用法",
        "还记得我是谁吗？我在学什么？"
    ]

    for q in queries:
        print(f"\n用户: {q}")
        answer = chat(q, thread)
        print(f"Agent: {answer}")
```

执行流程可视化：

```
用户消息
    │
    ▼
┌─────────┐   has tool_calls   ┌────────┐
│  Agent  │ ─────────────────►│ Tools  │
│  (LLM)  │ ◄─────────────── │(Execute)│
└────┬────┘   tool results    └────────┘
     │
     │ no tool_calls
     ▼
┌───────────────┐
│Update Profile │ ── update long-term memory
└──────┬────────┘
       │
       ▼
      END
```

## 苏格拉底时刻

> **Q1：LangGraph 的 StateGraph 相比普通的 Python 函数调用链有什么本质优势？**
> 提示：考虑断点续跑、状态回溯、条件分支、并发执行——这些在普通函数链中很难实现。

> **Q2：Supervisor 模式和 Peer-to-Peer 模式各有什么优缺点？**
> 提示：Supervisor 有单点瓶颈但易于控制；P2P 更灵活但可能陷入无限对话循环。

> **Q3：GraphRAG 为什么能回答"全局性"问题而传统 RAG 不行？**
> 提示：传统 RAG 只检索局部相似片段；GraphRAG 通过社区摘要捕获了**跨文档**的全局结构。

> **Q4：Function Calling 和 ReAct 是什么关系？**
> 提示：Function Calling 是 API 层面的工具调用协议；ReAct 是 Prompt 层面的推理框架。Function Calling 让 ReAct 从"文本解析"进化为"结构化调用"。

> **Q5：为什么 Agent 需要短期记忆和长期记忆两套机制？**
> 提示：短期记忆是当前对话的上下文窗口，受 token 限制；长期记忆跨会话持久化，需要压缩和索引。

## 常见问题 & 面试考点

### 面试高频问题

**Q：LangGraph 和 LangChain 的 AgentExecutor 有什么区别？**

A：AgentExecutor 是一个黑盒循环——你只能控制工具和 Prompt，无法自定义流程。LangGraph 是白盒——你可以定义任意的节点、边、条件分支，支持：
- 并行节点执行
- 条件路由
- Human-in-the-loop 中断
- 状态持久化和断点续跑
- 子图嵌套

**Q：Multi-Agent 系统中如何防止 Agent 之间的无限循环？**

A：常见策略：
1. **最大轮数限制**：设置 `max_iterations`
2. **Supervisor 判断**：让主管 Agent 判断是否完成
3. **状态收敛检测**：如果连续 N 轮状态无变化则终止
4. **Token 预算**：设置总 token 上限

**Q：Function Calling 的 JSON Schema 设计有什么最佳实践？**

A：
- 每个参数都要有清晰的 `description`
- 使用 `enum` 约束离散值
- 合理使用 `required` 标记必填字段
- 避免过深的嵌套（模型对深层嵌套的理解能力下降）
- 工具数量控制在 10-20 个以内（过多会降低选择准确率）

**Q：GraphRAG 的构建成本高，什么时候值得用？**

A：当满足以下条件时值得考虑：
- 文档之间存在大量**实体交叉引用**
- 用户问题偏向**全局性/聚合性**（如"主要趋势是什么"）
- 对**答案可追溯性**要求高
- 文档集相对稳定，不会频繁全量更新

## 推荐资源

| 资源 | 类型 | 说明 |
|------|------|------|
| [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/) | 官方文档 | 最权威的参考，含教程和 API |
| [LangGraph Academy](https://academy.langchain.com/) | 课程 | 7 个模块的系统化课程 |
| [AutoGen 文档](https://microsoft.github.io/autogen/) | 官方文档 | 微软多 Agent 框架 |
| [GraphRAG 论文](https://arxiv.org/abs/2404.16130) | 论文 | Microsoft Research, 2024 |
| [CrewAI 文档](https://docs.crewai.com/) | 官方文档 | 角色扮演式 Agent 框架 |
| [Function Calling Guide (OpenAI)](https://platform.openai.com/docs/guides/function-calling) | 官方文档 | OpenAI 工具调用指南 |
| [Tool Use (Anthropic)](https://docs.anthropic.com/claude/docs/tool-use) | 官方文档 | Claude 工具调用指南 |
| [ReAct 论文](https://arxiv.org/abs/2210.03629) | 论文 | Yao et al., 2022 |
| [LangSmith](https://smith.langchain.com/) | 平台 | Agent 可观测性和调试 |

---

> 下一步：回顾 [Agent 智能体基础](./agents.md) 深入理解 ReAct 原理，或查看 [RAG](./rag.md) 了解如何为 Agent 配备知识检索能力。
