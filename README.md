# LLM 101 中文站

[![Deploy](https://github.com/boots-coder/LLM-101-CN/actions/workflows/deploy.yml/badge.svg)](https://boots-coder.github.io/LLM-101-CN/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 一份系统整理的大语言模型（LLM）中文学习资料，基于 VitePress 构建，在线阅读：[boots-coder.github.io/LLM-101-CN](https://boots-coder.github.io/LLM-101-CN/)

## 这是什么

把大模型领域散落在论文、博客、开源项目里的知识，按照从基础到前沿的顺序整理成一份**可检索、可跳转、可做题**的在线手册。

内容使用 AI 辅助搜集与整理，并参考了 [mlabonne/llm-course](https://github.com/mlabonne/llm-course) 等优秀开源项目的结构设计。本仓库的价值不在于原创发明，而在于**体系化地梳理大模型领域的核心知识和学习路径**。

## 适合谁

| 角色 | 怎么用 |
|------|--------|
| **刚入门的同学** | 当教学字典——从数学基础到 RLHF，按顺序读一遍建立全局认知 |
| **准备面试的同学** | 当面试宝典——每章末尾有面试考点，练习系统有四级难度的实现题 |
| **已经在工作的同学** | 当复习笔记——遇到具体问题时全站搜索，快速定位到对应章节 |
| **想系统复习的同学** | 当知识地图——首页有 2 周速通、2 月完整、3 天突击三条学习路线 |

## 内容覆盖

```
基础知识        数学 · PyTorch · 神经网络 · NLP 基础
模型架构        Transformer · 注意力 · 分词器 · 解码策略 · GPT · Llama · DeepSeek-V3 · MoE · Flash Attention · Flow Matching · Scaling Laws
训练            预训练 · 继续预训练 · 数据集 · SFT/LoRA · RLHF/PPO/DPO/GRPO/KTO · 高级对齐 · RL 基础 · 推理(o1/R1) · Agent-RL(rule-based/RLVR/RLVE) · 蒸馏
工程化          推理优化(vLLM/KV Cache) · 量化 · 分布式训练 · 部署 · 模型合并 · 评估 · 安全(越狱/abliteration) · 性能剖析 · Ray 框架
应用            Prompt Engineering · RAG · Agent · Agent 框架 · Harness 工程 · 多模态(ViT/CLIP/LLaVA/原生多模态/视频/音频/Vision Agent/多模态 DPO)
深度剖析        GPT/LoRA/RLHF/vLLM/R1 复现 · minimind 端到端 · Kimi K2 · DeepSeek V4 · 数据流水线(datatrove/DCLM/去重) · 评估(MMLU/HumanEval/LLM Judge) · 安全(RLHF Safety/Red Team/GCG) · 手撕 nano Agent-RL · 手撕 LLaVA
练习系统        L1 选择题 → L2/L3 代码填空 → L4 完整实现（GPT / Llama / MoE / RLHF Pipeline / RAG 系统五大挑战）
MLM 代码训练    类似 BERT 随机挖空，每次刷新随机遮盖不同代码片段，刷到能默写为止
```

## 项目结构

```
llm-101-cn/
├── CLAUDE.md                  # 维护规则（搜索协议、frontmatter 规范、风格规范）
├── CONTENT_MAP.md             # 自动生成的全站索引（85 文件 / 2300+ 标题 / 行号）
├── docs/
│   ├── .vitepress/config.ts   # 导航 / 侧边栏配置
│   ├── fundamentals/          # 4 篇：数学 · Python+ML · 神经网络 · NLP 基础
│   ├── architecture/          # 9 篇：Transformer / Attention / Tokenizer / GPT / Llama / DeepSeek / MoE …
│   ├── training/              # 11 篇：Pretraining / SFT / RLHF / 对齐 / 推理模型 / Agent-RL / 蒸馏 …
│   ├── engineering/           # 11 篇：vLLM / 量化 / 分布式 / 部署 / 评估 / 安全 / Profiling …
│   ├── applications/          # 7 篇：Prompt / RAG / Agent / 多模态 / Harness …
│   ├── deep-dives/            # 19 篇手撕剖析：nano-vLLM · nano-GPT · nano-RLHF · nano-Agent-RL · LLaVA-from-scratch · R1 复现 · Kimi K2 内部 …
│   └── exercises/             # 21 套练习：L1 quiz · L2/L3 fill · L4 build（5 大完整实现挑战）
└── scripts/                   # 索引生成 / lint / 体例校验
```

> 全部内容由 `CONTENT_MAP.md` 索引，新增/改动文件后跑 `npm run map && npm run lint` 即可自检 frontmatter / sidebar 一致性 / 死链接。

## 特色功能

**AI 助教（两阶段上下文检索）** — 页面右下角内置 AI 对话。采用"LLM-as-Retriever"架构：构建时自动生成全站章节索引（`content-index.json`，包含每个文件的标题、话题标签、章节树），提问时先让 LLM 从当前模块的索引中筛选最相关的 2-3 个章节，再从 DOM 精准提取对应段落作为上下文送入回答模型。短页面（<3000 字）直接全文送入，跳过检索阶段。这种方式天然适合结构化文档站 — 章节标题本身就是高质量的 chunk 边界，不需要额外做 embedding 或向量数据库。

**划词批注** — 选中任意文字即可添加个人笔记，批注保存在浏览器本地，方便复习时快速回忆当时的理解。

**MLM 代码训练** — 练习页底部的随机挖空模式，灵感来自 BERT 的 Masked Language Model：每次刷新随机遮盖不同的代码片段，可调节挖空率（10%~50%），支持验证、显示答案、重新作答，反复训练直到能默写核心实现。

**结构化内容索引** — 全站内容自动生成 `CONTENT_MAP.md`，记录每个文件的章节标题与行号，同时构建 `content-index.json` 供 AI 助教检索使用。开发者可通过 `npm run map` 一键更新索引。

## 快速开始

```bash
git clone https://github.com/boots-coder/LLM-101-CN.git
cd LLM-101-CN
npm install
npm run docs:dev
```

浏览器打开 `http://localhost:5173` 即可。

## 致谢

特别感谢 **Deng Hang 老师**（GitHub: [@dhcode-cpp](https://github.com/dhcode-cpp)）在我学习大模型这条路上给予的耐心指点和系统性的启发，让我对 RLHF / 训练全流程有了更具象的理解。

本项目的知识内容参考了以下优秀的开源项目和资料：

**教学型课程与手撕代码**

- [mlabonne/llm-course](https://github.com/mlabonne/llm-course) — Maxime Labonne 的 LLM 课程，本项目的结构设计主要参考于此
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) + [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) — 预训练章节的主线代码与训练循环讲法
- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) — 第 5 章预训练手撕实现的对照参考
- [naklecha/llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch) — Llama 推理流程的逐步分解

**训练框架与 RLHF**

- [microsoft/LoRA](https://github.com/microsoft/LoRA) — LoRA 原始实现，本项目剖析章节最主要的代码来源
- [huggingface/peft](https://github.com/huggingface/peft) — LoRA / QLoRA 的工程化实现
- [huggingface/transformers](https://github.com/huggingface/transformers) — 模型架构与 tokenizer 的事实标准
- [huggingface/trl](https://github.com/huggingface/trl) — HuggingFace 的 RLHF 训练库（PPO / GRPO 教学代码以其 API 习惯为蓝本重写）
- [volcengine/verl](https://github.com/volcengine/verl) — RL 训练框架，RLHF 工程实现的对照参考
- [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — RLHF 全流程的另一条工程路线
- [THUDM/slime](https://github.com/THUDM/slime) — RL 训练框架剖析参考
- [dhcode-cpp/X-R1](https://github.com/dhcode-cpp/X-R1) — Apache 2.0，R1-Zero 低成本复现路径与 reward 函数设计参考自此项目

**推理引擎与量化**

- [vllm-project/vllm](https://github.com/vllm-project/vllm) — 高性能推理引擎
- [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — vLLM 的精简版本，便于阅读与剖析
- [Tencent/AngelSlim](https://github.com/Tencent/AngelSlim) — 投机采样（EAGLE-3 / SpecExit）与量化工具包

**旗舰模型与数据**

- [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) — MoE / MLA 架构的原始实现
- [meta-llama/llama3](https://github.com/meta-llama/llama3) — Llama 系列官方代码
- [huggingface/datatrove](https://github.com/huggingface/datatrove) — 预训练数据流水线（深度剖析章节使用）

**多模态（ViT / CLIP / LLaVA / 原生多模态 / 视频 / 音频 / Vision Agent）**

- [openai/CLIP](https://github.com/openai/CLIP) — 对比学习视觉-语言预训练原始实现
- [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) — Visual Instruction Tuning，多模态主线与「手撕 LLaVA」的核心参考
- [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL) — 6B 视觉编码器与渐进对齐策略
- [QwenLM/Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) — 动态分辨率 + M-RoPE 的开源代表
- [OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) — 端侧多模态，强调高分辨率与小模型
- [PKU-YuanGroup/Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) — 图像 / 视频统一对齐空间
- [QwenLM/Qwen-Audio](https://github.com/QwenLM/Qwen-Audio) — Whisper-style 音频编码 + LLM
- [microsoft/OmniParser](https://github.com/microsoft/OmniParser) — 屏幕解析与 GUI Agent 基础设施

**MoE 与高效架构**

- [XueFuzhao/OpenMoE](https://github.com/XueFuzhao/OpenMoE) — 完全开源的 MoE 实现
- [mistralai/mistral-inference](https://github.com/mistralai/mistral-inference) — Mixtral 风格的 sparse MoE 推理参考

**RAG 与检索**

- [microsoft/graphrag](https://github.com/microsoft/graphrag) — Graph-aware RAG，知识图谱与检索结合
- [infiniflow/ragflow](https://github.com/infiniflow/ragflow) — 工程级 RAG 系统，文档解析与检索流水线参考

**Agent-RL / R1 复现**

- [hkust-nlp/simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason) — 极简 R1-Zero 复现，nano Agent-RL 章节的最小目标
- [huggingface/open-r1](https://github.com/huggingface/open-r1) — R1 复现的 reward 函数集（accuracy / format / cosine）

**论文**

- Flash Attention (Dao et al., 2022 / 2023)、DPO (Rafailov et al., 2023)、GRPO (Shao et al., 2024) 等原始论文

内容整理过程中使用了 AI 工具辅助搜集和编写。

## 许可证

[MIT](LICENSE)
