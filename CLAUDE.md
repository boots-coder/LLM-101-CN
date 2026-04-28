# CLAUDE.md — LLM 维护协议

## 项目概述

LLM-101-CN 是一个中文大模型体系化教程站点，基于 VitePress 构建。

- **技术栈**：VitePress + GitHub Pages + Mermaid + MathJax
- **内容规模**：~50 个 markdown 文件，25,000+ 行
- **仓库**：git@github.com:boots-coder/LLM-101-CN.git

## 项目结构

```
llm-101-cn/
├── CLAUDE.md              ← 你正在读的文件（维护规则）
├── CONTENT_MAP.md         ← 全局内容索引（自动生成，LLM 入口点）
├── package.json           ← npm scripts（docs:dev / docs:build / map）
├── scripts/
│   └── generate-content-map.js  ← 索引生成脚本
└── docs/
    ├── .vitepress/
    │   ├── config.ts      ← 导航栏 + 侧边栏配置（新增页面必须修改）
    │   └── theme/         ← 自定义主题（批注功能组件）
    ├── index.md           ← 首页
    ├── roadmap.md         ← 学习路线
    ├── annotations.md     ← 我的笔记页面
    ├── fundamentals/      ← 基础知识（math, python-ml, neural-networks, nlp-basics）
    ├── architecture/      ← 模型架构（transformer, attention, tokenization, decoding, gpt, llama, deepseek）
    ├── training/          ← 训练（pretraining, datasets, sft, alignment, reasoning, distillation）
    ├── engineering/       ← 工程化（inference, quantization, deployment, merging, safety, distributed, evaluation）
    ├── applications/      ← 应用（prompt-engineering, rag, agents, agent-frameworks, multimodal）
    ├── deep-dives/        ← 深度剖析（持续更新）
    └── exercises/         ← 练习系统（11 套，4 级难度）
```

## 搜索协议（必读）

**永远不要全量读取所有文件。** 遵循以下搜索流程：

1. **先读 `CONTENT_MAP.md`** — 它包含每个文件的每个 `##` 标题和行号
2. **用行号精准跳转** — `Read file_path offset=<行号> limit=80` 直达目标章节
3. **按关键词搜索** — grep `CONTENT_MAP.md` 中的 `topics:` 行定位相关文件
4. **了解模块结构** — 读对应模块的 `index.md`（如 `docs/architecture/index.md`）
5. **了解侧边栏顺序** — 读 `docs/.vitepress/config.ts` 的 sidebar 配置

### 搜索示例

```
想找 Flash Attention 相关内容？
1. grep CONTENT_MAP.md 找到 "flash-attention" → architecture/attention.md
2. 看到 "Flash Attention 深度实现 (L429)" → Read attention.md offset=429 limit=100
3. 精准定位，无需读完 715 行全文
```

## 维护规则（强制）

### 规则 1：内容变更后必须更新索引

| 操作 | 必须执行 |
|------|---------|
| 增/删/改任何 `##` 标题 | `npm run map` |
| 新增 .md 文件 | 加 YAML frontmatter + 加 config.ts sidebar + `npm run map` |
| 改变文件的话题覆盖范围 | 更新该文件的 frontmatter `topics` 字段 + `npm run map` |
| 删除 .md 文件 | 从 config.ts sidebar 移除 + `npm run map` |
| 新增模块目录 | 创建 index.md + 加 config.ts nav/sidebar + `npm run map` |

### 规则 2：YAML Frontmatter 规范

每个内容 .md 文件（非 index.md）必须有：

```yaml
---
title: <页面标题>
description: <一行摘要，50-100 字符>
topics: [<关键词1>, <关键词2>, ...]
prereqs: [<前置文件路径>]  # 可选
---
```

### 规则 3：内容风格规范

- 每个内容文件以 `# 标题` 开头，紧跟一句话总结（`::: info` 或 `> 引用`）
- 章节顺序：体系定位 → 核心内容 → 苏格拉底时刻 → 面试考点 → 推荐资源
- 中文正文，英文技术术语，中英文之间加空格
- 代码块标注语言（python, typescript, bash 等）
- 使用 VitePress 容器语法（`::: tip` / `::: warning` / `::: details`），不用 MkDocs 语法（`!!!` / `???`）

### 规则 4：文件命名规范

- 全小写，连字符分隔：`flash-attention.md`，不用 `FlashAttention.md`
- 模块目录用单数名词：`training/`，不用 `trainings/`
- 练习文件模式：`<主题>-<类型>.md`（如 `attention-fill.md`, `gpt-build.md`）

### 规则 5：提交规范

- commit 信息用中文，格式：`<type>: <描述>`（feat/fix/docs/refactor）

## 常用命令

```bash
npm run docs:dev      # 本地开发预览
npm run docs:build    # 构建（验证无错误）
npm run docs:preview  # 预览构建结果
npm run map           # 重新生成 CONTENT_MAP.md（内容变更后必须运行）
npm run lint          # 校验 frontmatter / sidebar 一致性 / 死链接 / 索引新鲜度
```

> **提示**：内容变更后建议先 `npm run map`，再 `npm run lint` 自检。

## 知识来源

教程内容参考了以下开源项目与资料：

| 资源 | 链接 | 内容 |
|------|------|------|
| llm-course | [GitHub](https://github.com/mlabonne/llm-course) | Maxime Labonne 的 LLM 课程（英文参考） |
| AngelSlim | [GitHub](https://github.com/Tencent/AngelSlim) | 投机采样（EAGLE-3/SpecExit）+ 量化工具包 |
| TRL | [GitHub](https://github.com/huggingface/trl) | HuggingFace 训练库（DPO/GRPO/PPO/SFT Trainer） |
| OpenAI Harness Engineering | [Blog](https://openai.com/zh-Hans-CN/index/harness-engineering/) | Agent Harness 工程实践 |
| Anthropic Agent Harnesses | [Blog](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) | 长时间运行 Agent 的 Harness 设计 |
