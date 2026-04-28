---
title: "Harness 工程"
description: "从 Prompt Engineering 到 Harness Engineering——Agent 运行环境、约束、反馈回路与生产级治理"
topics: [harness-engineering, agent-harness, context-engineering, feedback-loop, scaffolding, progressive-disclosure, linter, guardrails]
prereqs: [applications/agents, applications/agent-frameworks]
---

# Harness 工程

::: info 一句话总结
Harness 工程不设计 Agent 本身的能力，而是设计 Agent **赖以运行的整个环境**——让它能看到正确信息、自动发现错误、被约束拦住、产出被验证、垃圾被清理。核心公式：**Agent = Model + Harness**。
:::

## 在大模型体系中的位置

```
提示词工程 → RAG 工程 → Agent 工程 → Harness 工程
   (单次调用)   (知识注入)   (自主行动)   (生产环境治理)
```

Harness 工程是 Agent 工程的上层抽象。Agent 工程关心"如何完成一个任务"，Harness 工程关心"如何让 Agent 在**长期、大规模**运行中不跑偏"。

## 核心概念

### 什么是 Harness？

"Harness"直译是"挽具"——套在马身上让马能拉车的装置。马有力量（Agent 有能力），但没有挽具它无法有效做功。

> Phil Schmid (Hugging Face) 的计算机类比：模型是 CPU，上下文窗口是 RAM，外部数据库是磁盘，工具是设备驱动，**Harness 是操作系统**。

**Harness 的正式定义**：为 Agent 构建的完整运行环境和支撑体系，包含模型推理本身之外的一切——工具编排、记忆、状态持久化、上下文管理、错误恢复、验证循环和安全执行。

### 三阶段演进

| 阶段 | 时期 | 核心关注 |
|------|------|---------|
| **Prompt Engineering** | 2022–2024 | 单轮指令的精心设计 |
| **Context Engineering** | 2025 | 模型"看到什么、何时看到、如何看到" |
| **Harness Engineering** | 2026+ | 完整系统基础设施——工具调度、运行时编排、跨会话状态、反馈循环、崩溃恢复 |

### 四个层次的工程对比

| 维度 | 提示词工程 | RAG 工程 | Agent 工程 | Harness 工程 |
|------|-----------|---------|-----------|-------------|
| **关注尺度** | 单次调用 | 检索+单次调用 | 多步任务 | 持续运行的生产系统 |
| **核心问题** | "怎么问" | "带什么知识问" | "怎么规划和行动" | "怎么让 Agent 可靠持续工作" |
| **时间跨度** | 毫秒级 | 秒级 | 分钟到小时 | 天到月 |
| **工程师角色** | 编写提示 | 构建检索管道 | 设计工具和推理流程 | 设计环境、定义约束、构建反馈回路 |
| **类比** | 给马下口头指令 | 给马看地图 | 教马自己找路 | 修好道路、装好挽具、设好围栏 |

## Harness 的六大核心组件

综合 Anthropic、OpenAI、LangChain 的实践，生产级 Agent Harness 通常包含六大子系统：

### 1. 记忆与上下文管理

模型看到什么、何时看到、如何看到。包括上下文裁剪、压缩、RAG 和外部状态存储。

**关键原则：给地图不给说明书**

OpenAI Codex 团队最初把所有指导信息塞进一个巨大的 AGENTS.md——相当于给 Agent 一本 1000 页的说明书。结果失败了：

- 巨大指令文件挤占 token 空间
- 所有内容都标记"重要"时，Agent 无法区分优先级
- 庞大文件很快过时
- 单一大文件无法被自动化工具校验

**解决方案：两层信息结构**

```
第一层（索引/目录）── 始终在上下文中
    ├── "架构请看 docs/DESIGN.md"
    ├── "前端规范请看 docs/FRONTEND.md"
    └── "安全请看 docs/SECURITY.md"

第二层（详细内容）── 按需加载
    └── Agent 需要某领域知识时，按图索骥去读
```

这就是**渐进式披露**（Progressive Disclosure）：Agent 启动时只加载最小量核心信息，然后根据任务需要逐步加载更深层文档。

::: tip 实际案例：Claude Code 的 CLAUDE.md
Claude Code 正是渐进式披露的典型实现：
- 自动读取 `CLAUDE.md`（~100 行规则索引）
- 需要时再读 `CONTENT_MAP.md`（详细内容地图）
- 最后用行号精准跳转到具体章节
:::

### 2. 工具编排

连接模型到外部 API、代码执行环境、数据库和文件系统，含访问控制策略。

::: warning Vercel 的反直觉发现
Vercel 的 v0 产品发现**移除 80% 可用工具**反而产生了更好的结果——更少的工具混淆带来更优的 Agent 表现。工具不是越多越好。
:::

### 3. 状态持久化

会话存储、文件系统即状态模式、基于 git 的进度跟踪、checkpoint/resume。

```
Anthropic 的推荐模式：
初始化 Agent → 生成 feature list → 写入 progress file
每次新 session → 读 progress file + git log → 恢复上下文
```

### 4. 验证与反馈循环

自动测试、LLM 可读错误信息的 linter、自愈循环。

**最小可行 Harness = 一个反馈回路：**

```
人类意图 → Agent 执行 → 自动验证 → 通过则完成 / 不通过则重试
```

三要素：
1. **一份清晰的指令**（地图，非说明书）
2. **一个可执行的验证手段**（测试/lint/类型检查——能机械判断"对不对"）
3. **一个重试机制**（失败信息反馈给 Agent，让它再来一次）

::: details OpenAI Codex 的 linter 实践
OpenAI 团队的自定义 linter 不仅检查违规，错误信息里还**直接嵌入修复指令**。这样 Agent 看到 lint 报错时能直接知道该怎么修，不需要人类介入。这就是把"品味"编码为机械化约束。
:::

Boris Cherny 的实验表明：有效的验证方法可提升 Agent 输出质量 **2–3 倍**。

### 5. 安全与护栏

输入过滤、输出验证、权限控制、沙箱隔离、不可逆操作的人工审批门。

**核心原则：约束用机械手段强制执行，不依赖 Agent 自觉**

```python
# ❌ 依赖 Agent 自觉（不可靠）
system_prompt = "请不要删除生产数据库中的数据"

# ✅ 机械化约束（可靠）
if action.type == "database_write" and action.target == "production":
    require_human_approval(action)
```

### 6. 生命周期管理

会话初始化、优雅终止、崩溃恢复、成本追踪、预算执行。

## 五个通用设计原则

从 OpenAI、Anthropic、LangChain 的实践中提炼出的通用原则，适用于任何领域的 Harness 构建：

### 原则一：情境必须对 Agent 可见且可发现

Agent 看不到的东西等于不存在。关键问题：
- Agent 完成任务需要哪些知识？
- 这些知识现在存储在哪里？
- Agent 能访问到吗？

**做法**：把隐性知识显式化，分散信息集中化，非结构化内容结构化。

> OpenAI 团队的教训：团队曾在 Slack 里讨论并达成某个架构共识，但 Codex Agent 无法访问 Slack——如果决策没被写进仓库，Agent 就完全不知道。

### 原则二：约束用机械手段强制执行

问自己："这条规则能不能写成一个返回 true/false 的函数？"
- **能** → 编码为自动检查
- **不能** → 继续拆解直到可以，或接受需要人类判断

### 原则三：给地图不给说明书

构建两层信息结构：
- 第一层：索引/目录，始终在上下文中
- 第二层：详细内容，Agent 按需加载

### 原则四：验证必须独立于 Agent 的生成过程

如果验证本身依赖 Agent 的主观判断，反馈回路就是自我循环，没有纠错能力。

验证应该是**外部的、确定性的、可重复的**：
- 写代码 → 测试
- 写文档 → 格式校验 + 交叉引用检查
- 数据分析 → 已知基准值对比
- 找不到确定性验证手段的地方 → 需要人类介入

### 原则五：持续清理，不让熵积累

Agent 会复现环境中的模式，**包括坏模式**。如果文档模板有问题，Agent 会用这个有问题的模板生成更多文档。

**做法**：定义"什么是好的"（黄金原则），设置定期扫描任务检测偏差，自动修复。频率要高于退化速度。

> OpenAI 团队将此比作**垃圾回收（GC）**：与其让垃圾堆积到不得不大规模清理（stop-the-world GC），不如持续小额清理（incremental GC）。

## Harness 架构分类

Eric Gerl 提出五类 Harness 架构：

| 架构类型 | 描述 | 代表 |
|---------|------|------|
| **Thin Loop Harness** | 模型思考，Harness 仅做"笨循环" | Claude Code |
| **图式 Harness** | 显式控制流（状态机/DAG） | LangGraph |
| **角色多 Agent Harness** | 多角色分工协作 | CrewAI |
| **自配置 Harness** | Agent 调整自身配置 | — |
| **自修改 Harness** | Agent 改写自身 Harness 代码 | — |

ThoughtWorks 的 Birgitta Böckeler 提供了另一种分解：

```
Harness 机制
├── 前馈 Guides（行动前施加的约束）
│   ├── 计算性的（确定性）── lint、类型检查
│   └── 推理性的（LLM 驱动）── 规划 Agent
│
└── 反馈 Sensors（观察后施加的修正）
    ├── 计算性的（确定性）── 测试结果
    └── 推理性的（LLM 驱动）── 审查 Agent
```

## 术语辨析

| 概念 | 定义 | 核心问题 |
|------|------|---------|
| **Scaffolding** | 首次 prompt 之前的预构建——系统提示、工具 schema、子 Agent 注册 | "Agent 是怎么组装的？" |
| **Harness** | 首次 prompt 之后的运行时编排——工具调度、上下文压缩、安全不变量 | "Agent 实际怎么运行？" |
| **Orchestration** | Harness 内部管理多 Agent 协调和工作流 | "多个 Agent 如何协作？" |
| **Framework** | 提供构建 Agent 的库和抽象 | "用什么工具构建？" |

> Salesforce 精辟概括：Framework 提供构建 Agent 的库，Harness 是**治理** Agent 在真实环境中行为的运行时系统。

## 从最小回路到生产系统

所有复杂的 Harness 都是最小反馈回路的多维扩展：

| 最小 Harness | 扩展后 |
|-------------|--------|
| 测试用例验证正确性 | 加上 linter 验证风格、结构测试验证架构 |
| 人类写测试 | Agent 自己写测试，再由另一个 Agent 审查 |
| 单次任务 | 长期运行，需要上下文管理和渐进式披露 |
| 一个 Agent | 多个 Agent 协作（子代理、Agent 审查 Agent） |
| 人类手动检查文档 | doc-gardening Agent 自动维护 |
| 人类手动清理坏模式 | 垃圾回收 Agent 定期扫描修复 |

每一层扩展都是因为某个瓶颈出现了（人类时间不够、上下文窗口不够、坏模式在扩散），然后针对瓶颈把人类的判断编码为自动化约束。

## 跨领域 Harness 设计清单

构建新领域 Harness 时的通用检查清单：

::: details 展开查看完整清单

**关于可见性**
- Agent 完成任务需要哪些知识？
- 这些知识 Agent 现在能访问到吗？
- 如果不能，如何让它可访问？

**关于约束**
- 哪些规则是必须遵守的？
- 其中哪些可以编码为自动检查？
- 错误信息是否包含足够的修复指引？

**关于情境管理**
- 信息总量是否超出上下文窗口？
- 是否建立了索引层和详情层的两级结构？

**关于验证**
- 有没有独立于 Agent 的验证手段？
- 这些验证是确定性的还是概率性的？
- 概率性的部分由谁做最终判断？

**关于持续维护**
- Agent 运行会产生哪些退化？
- 如何检测？如何修复？频率够不够？
:::

## 行业关键里程碑

| 时间 | 事件 | 意义 |
|------|------|------|
| 2025.11 | **Anthropic** 发布 "Effective Harnesses for Long-Running Agents" | 最早正式使用该术语的机构之一 |
| 2026.02.05 | **Mitchell Hashimoto** 发表 "My AI Adoption Journey" | 首次提出 Harness 概念 |
| 2026.02.11 | **OpenAI** 发表 Codex Harness Engineering 案例 | 3–7 人团队通过 1,500 PR 产出 100 万行代码，零手写 |
| 2026.02 | **LangChain** 提出 "Agent = Model + Harness" | 推出 DeepAgents 产品 |
| 2026.03 | 多篇学术论文发表（NLAHs、Meta-Harness、AutoHarness） | Harness 成为正式学术研究对象 |
| 2026.04 | **字节 DeerFlow 2.0** 开源，31.6K+ Stars | 国内首个大规模 Harness 开源项目 |

### 厚 Harness vs 薄 Harness 之争

- **Anthropic**：倡导厚 Harness——丰富的基础设施、广泛的约束和反馈循环
- **OpenAI Codex**：主张 Harness 应随模型改进而逐步变薄
- **Vercel v0**：移除 80% 工具反而效果更好

**核心实证发现**：Harness 质量现在比模型质量更重要。某模型仅通过更换 Harness 就从 6.7% 提升至 68.3% 的准确率。LangChain 仅更换 Harness（不换模型）就从 TerminalBench 排名 30+ 跃升至第 5。

## Harness 与 Agent 的本质区别

```
Agent 工程  → 让 Agent 能做事   → "如何完成一个任务？"
Harness 工程 → 让 Agent 持续做对事 → "如何保证长期不跑偏？"
```

Agent 工程设计的是 Agent 的**能力**——给它什么工具、怎么规划、怎么调用 API。

Harness 工程设计的是 Agent 的**环境**——让它能看到正确信息、自动发现做错了、做错时被拦住、结果被验证、垃圾被清理。

::: tip 辩证看待
Harness 工程并非全新技术，而是一个**新的统一抽象**——将此前分散的实践（工具编排、上下文管理、护栏、记忆、错误恢复）整合在一个概念清晰的术语下。类比：DevOps 也不是新技术，运维和 CI/CD 早已存在，但 DevOps 将它们提升为主要工作、重新定义了优先级。

**模型是商品，Harness 才是护城河。**
:::

## 开源 Harness 生态

| 项目 | 机构 | 特点 |
|------|------|------|
| **DeepAgents** | LangChain | 规划 + 文件系统 + 子 Agent 的生产 Harness |
| **OpenHarness** | 香港大学 HKUDS | 轻量级实现，43 工具 + 54 命令，MIT 许可 |
| **DeerFlow 2.0** | 字节跳动 | Super Agent Harness，31.6K+ Stars |
| **AutoHarness** | aiming-lab | 3 层管线治理，YAML 宪法，基于 trace 诊断 |
| **Claude Code** | Anthropic | Thin Loop 架构的典型代表 |

## 苏格拉底时刻

::: warning 停下来思考
1. **反馈回路设计**：你的 Agent 项目中，验证手段是独立于 Agent 的吗？还是依赖 Agent 自我评判？如果是后者，如何引入外部验证？

2. **信息架构**：你给 Agent 的是"说明书"还是"地图"？Token 是稀缺资源——你的信息架构是否支持按需加载？

3. **约束编码**：你希望 Agent 遵守的规则中，有多少已经编码为自动检查？有多少还停留在自然语言指令阶段？

4. **熵控制**：Agent 运行一段时间后，环境是变好了还是变差了？有没有自动检测退化的机制？

5. **厚薄之争**：对你的场景来说，应该用厚 Harness（更多约束和基础设施）还是薄 Harness（让模型承担更多）？你的判断依据是什么？
:::

## 面试考点

::: details 常见问题

**Q: Harness Engineering 和 Prompt Engineering 的本质区别是什么？**

A: Prompt Engineering 关注单次调用的输入输出质量，Harness Engineering 关注让 Agent 在持续运行的生产系统中可靠工作。前者解决"怎么问"，后者解决"怎么让 Agent 持续做对事"。关键区别在于时间尺度（毫秒 vs 天/月）和关注层面（模型层 vs 系统层）。

**Q: 什么是最小可行的 Harness？**

A: 一个反馈回路：清晰指令 + 可执行验证 + 重试机制。例如先写好测试用例，让 Agent 写函数，自动运行测试，失败则把错误信息喂回 Agent 重试。

**Q: Scaffolding 和 Harness 有什么区别？**

A: Scaffolding 是首次 prompt 之前的预构建（系统提示、工具 schema），Harness 是首次 prompt 之后的运行时编排（工具调度、上下文压缩、安全不变量、状态持久化）。

**Q: 为什么说"Harness 质量比模型质量更重要"？**

A: 实证发现：同一模型仅更换 Harness 就能从 6.7% 提升至 68.3%；LangChain 不换模型只换 Harness 就从排名 30+ 跃升至第 5。这说明在模型能力已经足够的前提下，环境设计才是瓶颈。

**Q: "给地图不给说明书"原则的具体实现是什么？**

A: 构建两层信息结构。第一层是始终在上下文中的索引/目录（~100 行），告诉 Agent 有哪些信息在哪里。第二层是详细内容，Agent 按需加载。代表实现：Claude Code 的 CLAUDE.md + CONTENT_MAP.md。
:::

## 推荐资源

- [OpenAI — Harness Engineering: Leveraging Codex](https://openai.com/zh-Hans-CN/index/harness-engineering/) — 里程碑案例研究
- [Anthropic — Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) — 最早的正式术语使用
- [LangChain — The Anatomy of an Agent Harness](https://blog.langchain.com/the-anatomy-of-an-agent-harness/) — "Agent = Model + Harness" 的提出
- [Mitchell Hashimoto — My AI Adoption Journey](https://mitchellh.com/writing/my-ai-adoption-journey) — 概念的起源
- [Birgitta Böckeler — Harness Engineering](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html) — Guides/Sensors 分解框架
- [arXiv:2603.25723 — Natural-Language Agent Harnesses](https://arxiv.org/abs/2603.25723) — 首篇正式学术论文
- [arXiv:2603.05344 — Building AI Coding Agents for the Terminal](https://arxiv.org/html/2603.05344v1) — Scaffolding vs Harness 辨析
