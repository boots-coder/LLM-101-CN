# 练习系统

本章提供一套渐进式的练习体系，帮助你从概念理解到代码实现，逐步巩固大模型的核心知识。

## 四级渐进体系

我们将练习分为四个难度级别，每个级别考察不同层次的能力：

| 级别 | 类型 | 考察能力 | 预计时间 |
|------|------|---------|---------|
| **Level 1** | 选择题 | 概念理解、直觉判断 | 15-30 分钟 |
| **Level 2** | 代码填空 | 关键代码的精确理解 | 30-60 分钟 |
| **Level 3** | 模块实现 | 核心模块的独立实现 | 1-2 小时 |
| **Level 4** | 实现挑战 | 从零实现完整系统 | 3-5 小时 |

### Level 1：选择题

适合刚学完理论的阶段。通过精心设计的选项，暴露常见的理解误区。每道题的干扰选项都来自真实的错误认知。

### Level 2：代码填空

给出完整代码框架，隐去关键部分（用 `_____` 标记）。你需要填入正确的代码使程序运行。考察的是"看懂全貌后能否补全细节"的能力。

### Level 3：模块实现

给出接口定义和测试用例，你需要独立实现一个完整的模块。不再是填空，而是从函数签名开始的完整实现。

### Level 4：实现挑战

从零实现一个完整的系统（如 mini-GPT）。只给需求描述和类骨架，所有实现细节需要自己决定。这是对理解深度和工程能力的综合考验。

---

## 完整练习列表

### 基础架构篇

| 练习 | 级别 | 涉及知识点 | 预计时间 | 难度 |
|------|------|-----------|---------|------|
| [Transformer 概念测验](transformer-quiz.md) | Level 1 | Transformer 架构、注意力机制 | 20 分钟 | :star: 入门 |
| [Attention 代码填空](attention-fill.md) | Level 2 | 缩放点积注意力、多头注意力 | 45 分钟 | :star::star: 中等 |
| [Tokenization 分词填空](tokenization-fill.md) -- BPE 训练 | Level 2 | pair 统计、合并规则 | 30 分钟 | :star::star: 中等 |
| [Tokenization 分词填空](tokenization-fill.md) -- WordPiece 对比 | Level 2-3 | 似然增益、合并策略差异 | 30 分钟 | :star::star::star: 中高 |
| [Tokenization 分词填空](tokenization-fill.md) -- 完整 Tokenizer | Level 3 | train/encode/decode 端到端 | 50 分钟 | :star::star::star: 中高 |
| [GPT 实现挑战](gpt-build.md) | Level 4 | Transformer 全部内容 | 3-5 小时 | :star::star::star::star: 困难 |
| [Llama 实现挑战](llama-build.md) | Level 4 | RoPE、RMSNorm、GQA、SwiGLU | 3-5 小时 | :star::star::star::star: 困难 |
| [Flash Attention 填空](flash-attn-fill.md) -- 内存分析 | Level 1-2 | 显存占用计算、O(N^2) 瓶颈 | 15 分钟 | :star: 入门 |
| [Flash Attention 填空](flash-attn-fill.md) -- 在线 Softmax | Level 2 | Safe Softmax、one-pass 在线算法 | 30 分钟 | :star::star: 中等 |
| [Flash Attention 填空](flash-attn-fill.md) -- Flash Forward | Level 3 | 分块计算、在线更新、rescale | 60 分钟 | :star::star::star: 中高 |
| [MoE 代码填空](moe-fill.md) -- Top-K 路由 | Level 2 | 门控网络、稀疏选择 | 25 分钟 | :star::star: 中等 |
| [MoE 代码填空](moe-fill.md) -- 稀疏 Forward | Level 3 | dispatch-compute-combine | 40 分钟 | :star::star::star: 中高 |
| [MoE 代码填空](moe-fill.md) -- 负载均衡 | Level 3 | 辅助损失、专家利用率 | 30 分钟 | :star::star::star: 中高 |
| [DeepSeek 架构填空](deepseek-arch-fill.md) -- MLA KV 压缩 | Level 2 | 低秩投影、KV Cache 压缩 | 30 分钟 | :star::star: 中等 |
| [DeepSeek 架构填空](deepseek-arch-fill.md) -- MLA RoPE 解耦 | Level 2-3 | nope/rope 拆分、双路注意力 | 35 分钟 | :star::star::star: 中高 |
| [DeepSeek 架构填空](deepseek-arch-fill.md) -- DeepSeekMoE | Level 2-3 | 共享专家 + 路由专家 | 35 分钟 | :star::star::star: 中高 |
| [DeepSeek 架构填空](deepseek-arch-fill.md) -- 辅助损失 Free | Level 3 | bias 动态负载均衡 | 40 分钟 | :star::star::star: 中高 |
| [Scaling Laws 填空](scaling-laws-fill.md) -- 参数量计算 | Level 1-2 | embedding/attention/FFN 参数 | 20 分钟 | :star: 入门 |
| [Scaling Laws 填空](scaling-laws-fill.md) -- FLOPs 估算 | Level 2 | C=6ND、训练时间 | 25 分钟 | :star::star: 中等 |
| [Scaling Laws 填空](scaling-laws-fill.md) -- Chinchilla 配比 | Level 2-3 | 最优 N/D、幂律拟合 | 40 分钟 | :star::star::star: 中高 |
| [Scaling Laws 填空](scaling-laws-fill.md) -- 预算规划器 | Level 3 | 综合规划、模型配置推荐 | 45 分钟 | :star::star::star: 中高 |

### 基础组件篇

| 练习 | 级别 | 涉及知识点 | 预计时间 | 难度 |
|------|------|-----------|---------|------|
| [基础组件实现](common-fill.md) -- Softmax | Level 2 | 数值稳定、Safe Softmax | 20 分钟 | :star::star: 中等 |
| [基础组件实现](common-fill.md) -- Cross Entropy | Level 2 | 交叉熵、log_softmax、NLL | 25 分钟 | :star::star: 中等 |
| [基础组件实现](common-fill.md) -- AdamW | Level 2 | 一阶矩、二阶矩、偏差校正、权重衰减 | 30 分钟 | :star::star: 中等 |
| [基础组件实现](common-fill.md) -- LayerNorm | Level 3 | 归一化、仿射变换 | 25 分钟 | :star::star::star: 中高 |
| [基础组件实现](common-fill.md) -- BPE 分词器 | Level 3 | BPE 合并规则、子词分词 | 40 分钟 | :star::star::star: 中高 |

### 预训练篇

| 练习 | 级别 | 涉及知识点 | 预计时间 | 难度 |
|------|------|-----------|---------|------|
| [预训练技术填空](pretraining-fill.md) -- CLM 目标函数 | Level 2 | next-token prediction、shift labels | 25 分钟 | :star::star: 中等 |
| [预训练技术填空](pretraining-fill.md) -- 文档拼接与分块 | Level 2 | packing、EOS 分隔、attention mask | 30 分钟 | :star::star: 中等 |
| [预训练技术填空](pretraining-fill.md) -- LR 调度 | Level 2 | warmup + cosine decay | 20 分钟 | :star::star: 中等 |
| [预训练技术填空](pretraining-fill.md) -- 训练监控 | Level 2-3 | perplexity、gradient norm、EMA、spike 检测 | 35 分钟 | :star::star::star: 中高 |
| [预训练技术填空](pretraining-fill.md) -- Mini 预训练流程 | Level 3 | 端到端训练循环 | 50 分钟 | :star::star::star: 中高 |

### 训练与微调篇

| 练习 | 级别 | 涉及知识点 | 预计时间 | 难度 |
|------|------|-----------|---------|------|
| [SFT 训练 Pipeline](sft-training-fill.md) -- ChatML 格式化 | Level 2 | Chat Template、消息拼接、EOS 处理 | 25 分钟 | :star::star: 中等 |
| [SFT 训练 Pipeline](sft-training-fill.md) -- Loss Masking | Level 2 | label masking、ignore_index=-100 | 30 分钟 | :star::star: 中等 |
| [SFT 训练 Pipeline](sft-training-fill.md) -- DataCollator | Level 3 | 动态 padding、attention mask、batch 拼接 | 40 分钟 | :star::star::star: 中高 |
| [SFT 训练 Pipeline](sft-training-fill.md) -- 梯度累积 | Level 2 | loss 缩放、等效 batch size | 20 分钟 | :star::star: 中等 |
| [SFT 训练 Pipeline](sft-training-fill.md) -- 完整训练循环 | Level 3 | cosine warmup、端到端训练 | 50 分钟 | :star::star::star: 中高 |
| [LoRA 代码填空](lora-fill.md) -- 矩阵初始化 | Level 2 | Kaiming 初始化、零初始化、旁路设计 | 20 分钟 | :star::star: 中等 |
| [LoRA 代码填空](lora-fill.md) -- merge/unmerge | Level 2 | 权重合并、推理优化 | 20 分钟 | :star::star: 中等 |
| [LoRA 代码填空](lora-fill.md) -- 完整模块 | Level 3 | LoRALinear 完整实现 | 40 分钟 | :star::star::star: 中高 |
| [LoRA 代码填空](lora-fill.md) -- 模型注入 | Level 3 | 递归替换、模型改造 | 30 分钟 | :star::star::star: 中高 |

### 对齐训练篇

| 练习 | 级别 | 涉及知识点 | 预计时间 | 难度 |
|------|------|-----------|---------|------|
| [DPO/GRPO 代码填空](dpo-grpo-fill.md) -- 概念题 | Level 1 | Bradley-Terry、DPO 原理、GRPO 原理 | 15 分钟 | :star: 入门 |
| [DPO/GRPO 代码填空](dpo-grpo-fill.md) -- DPO Loss | Level 2 | log ratio、sigmoid loss | 30 分钟 | :star::star: 中等 |
| [DPO/GRPO 代码填空](dpo-grpo-fill.md) -- GRPO Advantage | Level 2 | group 归一化、z-score | 20 分钟 | :star::star: 中等 |
| [DPO/GRPO 代码填空](dpo-grpo-fill.md) -- GRPO Loss | Level 3 | clipped ratio、KL penalty、策略梯度 | 45 分钟 | :star::star::star: 中高 |
| [DPO/GRPO 代码填空](dpo-grpo-fill.md) -- Reward Model | Level 3 | Value head、偏好建模 | 35 分钟 | :star::star::star: 中高 |

### 工程优化篇

| 练习 | 级别 | 涉及知识点 | 预计时间 | 难度 |
|------|------|-----------|---------|------|
| [推理优化填空](inference-fill.md) -- KV Cache | Level 2 | 缓存更新、增量解码 | 30 分钟 | :star::star: 中等 |
| [推理优化填空](inference-fill.md) -- PagedAttention | Level 2 | 块表管理、虚拟内存 | 35 分钟 | :star::star: 中等 |
| [推理优化填空](inference-fill.md) -- Continuous Batching | Level 3 | 动态调度、请求管理 | 45 分钟 | :star::star::star: 中高 |
| [推理优化填空](inference-fill.md) -- 投机采样 | Level 3 | draft-verify、接受率 | 40 分钟 | :star::star::star: 中高 |
| [量化技术填空](quantization-fill.md) -- Absmax 量化 | Level 2 | 对称量化、scale 计算 | 20 分钟 | :star::star: 中等 |
| [量化技术填空](quantization-fill.md) -- Zero-Point 量化 | Level 2 | 非对称量化、偏斜分布 | 25 分钟 | :star::star: 中等 |
| [量化技术填空](quantization-fill.md) -- Per-Channel 量化 | Level 2-3 | outlier channel、逐通道 | 30 分钟 | :star::star::star: 中高 |
| [量化技术填空](quantization-fill.md) -- GPTQ 核心 | Level 3 | 逐列量化、误差补偿 | 45 分钟 | :star::star::star: 中高 |
| [量化技术填空](quantization-fill.md) -- SmoothQuant | Level 3 | 激活平滑、难度转移 | 40 分钟 | :star::star::star: 中高 |
| [分布式训练填空](distributed-fill.md) -- Ring AllReduce | Level 2 | scatter-reduce、allgather | 30 分钟 | :star::star: 中等 |
| [分布式训练填空](distributed-fill.md) -- DDP 训练 | Level 2-3 | 梯度同步、DistributedSampler | 40 分钟 | :star::star::star: 中高 |
| [分布式训练填空](distributed-fill.md) -- 列并行 | Level 3 | Megatron-style、权重切分 | 35 分钟 | :star::star::star: 中高 |
| [分布式训练填空](distributed-fill.md) -- 行并行 | Level 3 | AllReduce sum、FFN 配对 | 35 分钟 | :star::star::star: 中高 |
| [分布式训练填空](distributed-fill.md) -- Pipeline 调度 | Level 3 | GPipe、bubble ratio | 40 分钟 | :star::star::star: 中高 |

### 应用与评估篇

| 练习 | 级别 | 涉及知识点 | 预计时间 | 难度 |
|------|------|-----------|---------|------|
| [Prompt 与评估填空](prompt-eval-fill.md) -- Few-Shot 构造器 | Level 2 | 示例选择、格式化 | 25 分钟 | :star::star: 中等 |
| [Prompt 与评估填空](prompt-eval-fill.md) -- Chain-of-Thought | Level 1-2 | zero-shot/few-shot CoT | 20 分钟 | :star: 入门 |
| [Prompt 与评估填空](prompt-eval-fill.md) -- Prompt 注入防御 | Level 2-3 | 输入清洗、sandwich defense | 35 分钟 | :star::star::star: 中高 |
| [Prompt 与评估填空](prompt-eval-fill.md) -- BLEU 分数 | Level 2 | n-gram precision、brevity penalty | 30 分钟 | :star::star: 中等 |
| [Prompt 与评估填空](prompt-eval-fill.md) -- ROUGE-L | Level 2-3 | LCS 动态规划、F1 | 30 分钟 | :star::star::star: 中高 |
| [Prompt 与评估填空](prompt-eval-fill.md) -- LLM-as-Judge | Level 3 | 评分 prompt、pairwise comparison | 40 分钟 | :star::star::star: 中高 |
| [RAG 与 Agent 填空](rag-agent-fill.md) -- 文本分块 | Level 2 | 滑动窗口、语义分块 | 25 分钟 | :star::star: 中等 |
| [RAG 与 Agent 填空](rag-agent-fill.md) -- 向量检索 | Level 2 | 余弦相似度、top-k | 25 分钟 | :star::star: 中等 |
| [RAG 与 Agent 填空](rag-agent-fill.md) -- RAG Pipeline | Level 3 | 端到端检索增强生成 | 40 分钟 | :star::star::star: 中高 |
| [RAG 与 Agent 填空](rag-agent-fill.md) -- ReAct Agent | Level 2-3 | Thought-Action-Observation 循环 | 35 分钟 | :star::star::star: 中高 |
| [RAG 与 Agent 填空](rag-agent-fill.md) -- Tool Calling | Level 3 | JSON Schema、函数动态调用 | 45 分钟 | :star::star::star: 中高 |

---

## 推荐学习路径

### 路径一：从零开始（适合初学者）

1. Transformer 概念测验 (Level 1) -- 建立直觉
2. 基础组件实现: Softmax + Cross Entropy (Level 2) -- 夯实基础
3. Attention 代码填空 (Level 2) -- 理解核心机制
4. RoPE 代码填空 (Level 2-3) -- 掌握位置编码
5. GPT 实现挑战 (Level 4) -- 综合实战
6. SFT 训练 Pipeline: ChatML + Loss Masking (Level 2) -- 走通训练

### 路径二：聚焦训练（适合有基础的学习者）

1. SFT 训练 Pipeline: 全部 5 个练习 (Level 2-3) -- 掌握训练全流程
2. LoRA 代码填空: 全部 4 个练习 (Level 2-3)
3. DPO/GRPO 概念题 + 代码填空: 全部练习 (Level 1-3)
4. 量化技术: Absmax + GPTQ (Level 2-3) -- 理解模型压缩

### 路径三：工程进阶（适合想成为大佬的开发者）

1. Flash Attention 全部练习 (Level 1-3) -- 理解高性能计算
2. 分布式训练: AllReduce → DDP → Tensor Parallel (Level 2-3)
3. 推理优化: KV Cache → PagedAttention → 投机采样 (Level 2-3)
4. 量化技术: 从 Absmax 到 SmoothQuant (Level 2-3)

### 路径四：全栈应用（适合做产品的开发者）

1. RAG 与 Agent: 文本分块 + 向量检索 + RAG Pipeline (Level 2-3)
2. RAG 与 Agent: ReAct Agent + Tool Calling (Level 2-3)
3. SFT 训练 Pipeline: 完整训练循环 (Level 3)
4. 量化技术: SmoothQuant (Level 3) -- 部署优化

### 路径五：速通挑战（适合有经验的开发者）

1. 直接挑战 GPT 实现 (Level 4)
2. Flash Attention Forward (Level 3)
3. GRPO Loss 完整实现 (Level 3)
4. 分布式训练: Tensor Parallel + Pipeline 调度 (Level 3)
5. GPTQ 核心 (Level 3)

---

## 使用建议

1. **先学后练**：每个练习都标注了前置知识，确保你已学完对应章节再开始
2. **先想后看**：答案都用 `<details>` 折叠，先独立思考，实在想不出再展开
3. **动手跑代码**：Level 2 以上的练习，一定要在本地跑通，不要只"看懂"
4. **记录错误**：做错的题往往暴露了知识盲点，这比做对更有价值
5. **循序渐进**：同一文件内的练习难度递增，按顺序做效果最好

## 详细内容索引
<!-- AUTO-GENERATED-CONTENT-INDEX-START -->

### agent-rl-fill.md — Agent-RL 代码填空 (767 lines)
- 练习 1：格式奖励（Format Reward, Level 2） (L30)
- 练习 2：可验证奖励（Verifiable Reward, Level 2） (L98)
- 练习 3：多轮轨迹的 Loss Mask（Level 2-3） (L194)
- 练习 4：异步 Rollout 队列（Level 3） (L297)
- 练习 5：Composite Reward 组装（Level 3） (L399)
- 练习 6：Trajectory → Token Reward 分配（Level 3） (L507)
- MLM 代码训练模式 (L629)
  - 格式奖励 (L633)
  - 可验证奖励 (L644)
  - 多轮轨迹 Loss Mask (L671)
  - 异步 Rollout Buffer (L686)
  - Composite Reward 工厂 (L710)
  - Trajectory→Token Reward 分配 (L723)
- 苏格拉底时刻 (L752)
- 推荐资源 (L759)

### attention-fill.md — Attention 代码填空 (300 lines)
- 练习 1：Scaled Dot-Product Attention (L16)
- 练习 2：MultiHeadAttention.__init__ (L83)
- 练习 3：MultiHeadAttention.forward (L147)
- 验证代码 (L235)
- MLM 代码训练模式 (L261)
  - Scaled Dot-Product Attention (L265)
  - Multi-Head Attention (L278)

### common-fill.md — 基础组件实现 (704 lines)
- 练习 1：数值稳定的 Softmax 实现（Level 2） (L19)
- 练习 2：Cross Entropy 实现（Level 2） (L111)
- 练习 3：AdamW 优化器（Level 2） (L212)
- 练习 4：LayerNorm 实现（Level 3） (L328)
- 练习 5：BPE 分词器核心逻辑（Level 3） (L460)
- MLM 代码训练模式 (L645)
  - LayerNorm 前向传播 (L649)
  - Safe Softmax (L667)
  - AdamW 优化器更新 (L679)
  - Cross Entropy Loss (L693)

### deepseek-arch-fill.md — DeepSeek 架构填空 (749 lines)
- 练习 1: Multi-Latent Attention (MLA) -- KV 压缩（Level 2） (L13)
  - 背景 (L15)
  - 任务 (L24)
  - 提示 (L88)
- 练习 2: MLA -- RoPE 解耦（Level 2-3） (L129)
  - 背景 (L131)
  - 任务 (L141)
  - 提示 (L203)
- 练习 3: DeepSeekMoE -- 共享专家 + 路由专家（Level 2-3） (L253)
  - 背景 (L255)
  - 任务 (L263)
  - 提示 (L343)
- 练习 4: 辅助损失 Free 的负载均衡（Level 3） (L390)
  - 背景 (L392)
  - 任务 (L396)
  - 提示 (L485)
- 练习 5: KV Cache 大小计算综合题（Level 1-2） (L536)
  - 背景 (L538)
  - 任务 (L554)
  - 提示 (L595)
- 总结 (L634)
  - 延伸思考 (L644)
- MLM 代码训练模式 (L652)
  - MLA 低秩 KV 压缩 (L656)
  - RoPE 解耦: nope/rope 分离 (L684)
  - DeepSeekMoE 共享专家 + 路由专家 (L704)
  - 辅助损失 Free 负载均衡 (L730)

### distributed-fill.md — 分布式训练填空 (799 lines)
- 练习 1: Ring AllReduce 模拟（Level 2） (L13)
  - 背景 (L15)
  - 任务 (L24)
  - 提示 (L103)
- 练习 2: 手写 DDP 训练循环（Level 2-3） (L153)
  - 背景 (L155)
  - 任务 (L165)
  - 提示 (L261)
- 练习 3: Tensor Parallelism -- 列并行线性层（Level 3） (L313)
  - 背景 (L315)
  - 任务 (L326)
  - 提示 (L387)
- 练习 4: Tensor Parallelism -- 行并行线性层（Level 3） (L425)
  - 背景 (L427)
  - 任务 (L440)
  - 提示 (L504)
- 练习 5: Pipeline Parallelism 微批次调度（Level 3） (L552)
  - 背景 (L554)
  - 任务 (L565)
  - 提示 (L648)
- MLM 代码训练模式 (L703)
  - Ring AllReduce 核心循环 (L707)
  - DDP 训练循环关键步骤 (L732)
  - Tensor Parallelism: 列并行与行并行 (L752)
  - Pipeline Parallelism 微批次调度 (L775)

### dpo-grpo-fill.md — DPO/GRPO 填空 (783 lines)
- 练习 1：Bradley-Terry 模型概念题（Level 1） (L28)
- 练习 2：DPO Loss 实现（Level 2） (L96)
- 练习 3：GRPO Advantage 计算（Level 2） (L188)
- 练习 4：GRPO Loss 完整实现（Level 3） (L271)
- 练习 5：Reward Model 的 forward（Level 3） (L415)
- 练习 6：端到端 toy GRPO 训练（Level 3） (L574)
- MLM 代码训练模式 (L723)
  - DPO Loss 计算 (L727)
  - GRPO Advantage 计算 (L745)
  - GRPO Policy Loss（含 KL Penalty） (L755)

### flash-attn-fill.md — Flash Attention 填空 (727 lines)
- 练习 1: 标准 Attention 的内存分析（Level 1-2） (L13)
  - 背景 (L15)
  - 任务 (L43)
  - 提示 (L72)
- 练习 2: Safe Softmax 与在线 Softmax（Level 2） (L109)
  - 背景 (L111)
  - 任务 (L126)
  - 提示 (L186)
- 练习 3: 分块矩阵乘法（Level 2） (L220)
  - 背景 (L222)
  - 任务 (L234)
  - 提示 (L295)
- 练习 4: Flash Attention Forward（Level 3） (L330)
  - 背景 (L332)
  - 任务 (L360)
  - 提示 (L444)
- 练习 5: Flash Attention 反向传播的核心洞察（Level 3） (L499)
  - 背景 (L501)
  - 任务 (L517)
  - 提示 (L611)
- MLM 代码训练模式 (L653)
  - 在线 Softmax (m/l 更新) (L657)
  - 分块加载与 Tiling (L676)
  - Flash Attention Forward 核心逻辑 (L696)

### gpt-build.md — GPT 实现挑战 (678 lines)
- 挑战目标 (L10)
- 热身练习 (L18)
  - 热身 1：手写 GELU 激活函数 (L22)
  - 热身 2：构造因果掩码 (L51)
  - 热身 3：实现 KV Cache 更新 (L91)
- 需求规格 (L148)
  - 模型配置 (L150)
  - 必须实现的组件 (L164)
- 类骨架 (L174)
- 训练脚本骨架 (L343)
- 评估标准 (L376)
  - 基础要求（必须达成） (L378)
  - 进阶要求（挑战自我） (L389)
  - 高阶挑战（选做） (L401)
- 常见陷阱 (L409)
- 参考时间分配 (L419)
- 参考实现 (L433)
- MLM 代码训练模式 (L605)
  - 因果自注意力（合并 QKV 投影 + 多头拆分 + 缩放点积） (L609)
  - GPT Block（Pre-Norm 残差结构） (L631)
  - GPT Forward + Generate (L648)

### inference-fill.md — 推理优化填空 (710 lines)
- 练习 1：KV Cache 更新（Level 2） (L13)
  - 背景 (L15)
  - 任务 (L19)
  - 提示 (L69)
- 练习 2：PagedAttention 块表管理（Level 2） (L91)
  - 背景 (L93)
  - 任务 (L97)
  - 提示 (L162)
- 练习 3：Continuous Batching 调度器（Level 3） (L201)
  - 背景 (L203)
  - 任务 (L210)
  - 提示 (L304)
- 练习 4：投机采样验证（Level 3） (L349)
  - 背景 (L351)
  - 任务 (L359)
  - 提示 (L412)
- 练习 5：Chunked Prefill 调度（Level 3） (L458)
  - 背景 (L460)
  - 任务 (L464)
  - 提示 (L560)
- 总结 (L616)
- MLM 代码训练模式 (L628)
  - KV Cache 实现 (L632)
  - 投机采样验证 (L652)
  - Continuous Batching 调度 (L683)

### llama-build.md — Llama 实现挑战 (440 lines)
- 热身练习 (L12)
  - 练习 1: RMSNorm 实现（Level 2） (L14)
  - 练习 2: SwiGLU 实现（Level 2-3） (L93)
  - 练习 3: GQA 的 KV Repeat（Level 2-3） (L181)
- 主挑战: 构建完整 Llama Decoder（Level 4） (L244)
  - 目标 (L246)
  - 配置 (L257)
  - 要求 (L274)
  - 评估标准 (L362)
  - 测试用例 (L375)
- 参考实现 (L415)
- 进阶思考 (L428)

### lora-fill.md — LoRA 代码填空 (304 lines)
- 练习 1：LoRA 旁路前向（Level 2） (L26)
- 练习 2：LoRA 的 merge 与 unmerge（Level 2） (L94)
- 练习 3：完整 LoRALinear 模块（Level 3） (L169)
- 练习 4：给预训练模型注入 LoRA（Level 3） (L236)

### moe-build.md — MoE 实现挑战 (749 lines)
- 挑战目标 (L10)
- 热身练习 (L22)
  - 热身 1：Top-K Router（softmax + topk + 归一化） (L26)
  - 热身 2：Load Balance Loss（auxiliary loss） (L108)
  - 热身 3：Expert FFN（一个 SwiGLU MLP） (L187)
  - 热身 4（可选）：容量因子 + Token Dropping (L227)
- 完整挑战：实现 MoE Transformer (L298)
  - 模型配置 (L300)
  - 子任务 1：MoE Layer（router + N experts + dispatch + combine） (L326)
  - 子任务 2：MoE Transformer Block (L413)
  - 子任务 3：完整 MoE 模型 + 训练 loop (L468)
  - 训练脚本骨架 (L521)
- 测试与验证 (L556)
  - 测试 1：Shape 测试（必过） (L558)
  - 测试 2：Load Balance 数值测试（关键） (L581)
  - 测试 3：训练 loss 应正常下降 (L627)
  - 测试 4：Aux loss 在均衡时应趋近理论值 (L651)
- 进阶：DeepSeek MoE 风格的 fine-grained + shared expert (L663)
- 评分标准 (L689)
- 常见陷阱 (L706)
- 参考时间分配 (L718)
- 推荐资源 (L731)

### moe-fill.md — MoE 代码填空 (423 lines)
- 练习 1: Top-K 路由（Level 2） (L13)
  - 背景 (L15)
  - 任务 (L25)
  - 提示 (L70)
- 练习 2: 稀疏 MoE Forward — dispatch-compute-combine（Level 3） (L107)
  - 背景 (L109)
  - 任务 (L119)
  - 提示 (L179)
- 练习 3: 负载均衡损失（Level 3） (L218)
  - 背景 (L220)
  - 任务 (L231)
  - 提示 (L271)
- 总结 (L333)
  - 延伸思考 (L341)
- MLM 代码训练模式 (L350)
  - Top-K 路由门控 (L354)
  - 稀疏 MoE Forward（dispatch-compute-combine） (L365)
  - 负载均衡损失 (L406)

### multimodal-fill.md — 多模态代码填空 (888 lines)
- 练习 1：Patch Embedding（Level 2） (L30)
- 练习 2：CLIP InfoNCE Loss（Level 2） (L133)
- 练习 3：LLaVA Visual Projector 工厂（Level 2） (L262)
- 练习 4：Video Frame Sampler（Level 3） (L403)
- 练习 5：Vision Tool Router（Level 3） (L560)
- MLM 风格巩固 (L732)
  - Patch Embedding (L736)
  - CLIP InfoNCE Loss (L758)
  - LLaVA Visual Projector 工厂 (L787)
  - Video Frame Sampler (L814)
  - Vision Tool Router (L842)
- 苏格拉底时刻 (L870)
- 推荐资源 (L878)

### ppo-fill.md — PPO 代码填空 (639 lines)
- 练习 1：Importance Ratio（Level 2） (L28)
- 练习 2：GAE 倒序递推（Level 2） (L95)
- 练习 3：Clipped Surrogate Policy Loss（Level 2） (L187)
- 练习 4：Value Clipped Loss（Level 3） (L277)
- 练习 5：KL 散度的三种估计（Level 2） (L365)
- 练习 6：把 KL 揉进 Token-Level Reward（Level 3） (L458)
- MLM 代码训练模式 (L560)
  - Importance Ratio (L564)
  - GAE 倒序递推 (L574)
  - Clipped Surrogate Loss (L593)
  - Value Clipped Loss (L604)
  - KL Estimators (L617)
  - Token Reward Shaping (L628)

### pretraining-fill.md — 预训练技术填空 (657 lines)
- 练习 1: Causal Language Modeling 目标函数（Level 2） (L19)
  - 背景 (L21)
  - 任务 (L27)
  - 提示 (L70)
- 练习 2: 数据预处理 -- 文档拼接与分块（Level 2） (L99)
  - 背景 (L101)
  - 任务 (L107)
  - 提示 (L155)
- 练习 3: 学习率调度 -- Warmup + Cosine Decay（Level 2） (L195)
  - 背景 (L197)
  - 任务 (L203)
  - 提示 (L247)
- 练习 4: 训练指标监控（Level 2-3） (L280)
  - 背景 (L282)
  - 任务 (L286)
  - 提示 (L360)
- 练习 5: Mini 预训练完整流程（Level 3） (L397)
  - 背景 (L399)
  - 任务 (L405)
  - 提示 (L542)
- MLM 代码训练模式 (L582)
  - CLM Loss 计算 (L586)
  - 数据拼接与文档 Mask (L602)
  - Warmup + Cosine Decay 学习率调度 (L625)
  - 预训练循环（梯度裁剪 + LR 调度） (L638)

### prompt-eval-fill.md — Prompt Engineering 与评估填空 (899 lines)
- 练习 1: Few-Shot Prompt 构造器（Level 2） (L14)
  - 背景 (L16)
  - 任务 (L22)
  - 提示 (L80)
- 练习 2: Chain-of-Thought 模板（Level 1-2） (L124)
  - 背景 (L126)
  - 任务 (L132)
  - 提示 (L174)
- 练习 3: Prompt 注入防御（Level 2-3） (L224)
  - 背景 (L226)
  - 任务 (L232)
  - 提示 (L311)
- 练习 4: BLEU 分数计算（Level 2） (L363)
  - 背景 (L365)
  - 任务 (L371)
  - 提示 (L437)
- 练习 5: ROUGE-L 分数计算（Level 2-3） (L485)
  - 背景 (L487)
  - 任务 (L493)
  - 提示 (L535)
- 练习 6: LLM-as-Judge 评估框架（Level 3） (L589)
  - 背景 (L591)
  - 任务 (L597)
  - 提示 (L701)
- MLM 代码训练模式 (L750)
  - Few-Shot Prompt 构造器 (L754)
  - Chain-of-Thought 模板 (L783)
  - BLEU 分数计算 (L801)
  - ROUGE-L (LCS) 计算 (L835)
  - LLM-as-Judge 评估 Prompt (L860)

### quantization-fill.md — 量化技术填空 (696 lines)
- 练习 1: Absmax 对称量化（Level 2） (L27)
  - 背景 (L29)
  - 任务 (L37)
  - 提示 (L70)
- 练习 2: Zero-Point 非对称量化（Level 2） (L124)
  - 背景 (L126)
  - 任务 (L136)
  - 提示 (L174)
- 练习 3: Per-Channel vs Per-Tensor 量化（Level 2-3） (L233)
  - 背景 (L235)
  - 任务 (L243)
  - 提示 (L293)
- 练习 4: GPTQ 核心 -- 逐列量化与误差补偿（Level 3） (L362)
  - 背景 (L364)
  - 任务 (L375)
  - 提示 (L422)
- 练习 5: SmoothQuant -- 激活平滑（Level 3） (L479)
  - 背景 (L481)
  - 任务 (L497)
  - 提示 (L545)
- MLM 代码训练模式 (L636)
  - Absmax 对称量化/反量化 (L640)
  - Zero-Point 非对称量化 (L653)
  - Per-Channel 量化 (L668)
  - SmoothQuant 激活平滑 (L682)

### rag-agent-fill.md — RAG 与 Agent 填空 (772 lines)
- 练习 1: 文本分块策略（Level 2） (L13)
  - 背景 (L15)
  - 任务 (L19)
  - 提示 (L67)
- 练习 2: 余弦相似度与向量检索（Level 2） (L105)
  - 背景 (L107)
  - 任务 (L111)
  - 提示 (L163)
- 练习 3: RAG Pipeline 端到端（Level 3） (L210)
  - 背景 (L212)
  - 任务 (L216)
  - 提示 (L287)
- 练习 4: ReAct Agent 循环（Level 2-3） (L323)
  - 背景 (L325)
  - 任务 (L329)
  - 提示 (L416)
- 练习 5: 结构化 Tool Calling（Level 3） (L459)
  - 背景 (L461)
  - 任务 (L465)
  - 提示 (L567)
- 总结 (L623)
  - 延伸思考 (L633)
- MLM 代码训练模式 (L643)
  - 文本分块与重叠 (L647)
  - 余弦相似度检索 (L661)
  - RAG Pipeline: 检索-增强-生成 (L681)
  - ReAct Agent 循环 (L701)
  - Tool Calling 调度器 (L735)

### rag-build.md — RAG 系统实现挑战 (1137 lines)
- 挑战目标 (L10)
- 系统架构总览 (L35)
  - 环境准备 (L82)
- 阶段 1：文档处理与分块 (L100)
  - 任务 1.1：固定窗口分块（带 overlap） (L102)
  - 任务 1.2：语义分块（句子边界 + 长度阈值） (L162)
  - 任务 1.3：父子文档结构 (L219)
- 阶段 2：Embedding 与向量存储 (L306)
  - 任务 2.1：用 sentence-transformers 做 embedding (L308)
  - 任务 2.2：手写 in-memory 向量存储 (L354)
  - 任务 2.3：批处理 embedding (L420)
- 阶段 3：检索（Dense + Sparse + Hybrid） (L451)
  - 任务 3.1：Dense 检索 (L453)
  - 任务 3.2：手写 BM25 (L468)
  - 任务 3.3：Hybrid 检索（RRF） (L554)
- 阶段 4：重排与查询改写 (L635)
  - 任务 4.1：Cross-Encoder Reranker (L637)
  - 任务 4.2：Query Rewriting (L685)
- 阶段 5：生成与引用 (L765)
  - 任务 5.1：Context 拼接 + 生成 (L767)
  - 任务 5.2：Inline 引用 (L824)
  - 任务 5.3（可选）：Streaming + 边输出边引用 (L860)
- 端到端验证 (L881)
- 评分标准 (L948)
  - 基础要求（必须达成） (L950)
  - 进阶要求（挑战自我） (L962)
  - 高阶挑战（选做） (L972)
- 进阶：GraphRAG / Agentic RAG (L982)
  - GraphRAG（实体图增强） (L986)
  - Agentic RAG (L1002)
- 常见陷阱 (L1029)
- 参考时间分配 (L1038)
- 推荐资源 (L1054)
- MLM 代码训练模式 (L1071)
  - BM25 评分核心 (L1075)
  - RRF 融合 (L1094)
  - Dense 检索 top-k (L1108)
  - 端到端 RAG Pipeline (L1123)

### rlhf-build.md — RLHF Pipeline 实现挑战 (1253 lines)
- 挑战目标 (L10)
- 三阶段总览 (L26)
- 环境与依赖 (L73)
- 阶段 1：SFT 微调 (L100)
  - 1.1 任务说明 (L102)
  - 1.2 数据准备 (L114)
  - 1.3 训练循环 (L179)
  - 1.4 验证 (L213)
- 阶段 2：Reward Model 训练 (L240)
  - 2.1 任务说明 (L242)
  - 2.2 RewardModel 实现 (L262)
  - 2.3 数据 + 损失 (L306)
  - 2.4 训练循环 (L367)
  - 2.5 验证 (L392)
- 阶段 3：PPO 优化 (L429)
  - 3.1 子任务：实现 Value Head（共享 backbone + 新 head） (L433)
  - 3.2 子任务：计算 token-level log probability (L491)
  - 3.3 子任务：实现 GAE (L528)
  - 3.4 子任务：实现 Clipped Surrogate + Value Loss (L568)
  - 3.5 子任务：完整 PPO Loop (L624)
- 端到端验证 (L779)
  - 端到端验证标准 (L846)
- 评分标准 (L867)
- 进阶：把 PPO 替换成 GRPO / DPO (L883)
  - 进阶 A：替换为 DPO（去掉 RM 和 PPO，全部用 SFT-style 训练） (L887)
  - 进阶 B：替换为 GRPO（去掉 critic，用 group-baseline 估计 advantage） (L904)
- 常见陷阱总结 (L923)
- 时间分配建议 (L955)
- 参考实现 (L972)
- 推荐资源 (L1131)
- MLM 代码训练模式 (L1160)
  - Reward Model（取最后一个 valid token 的 hidden） (L1164)
  - Bradley-Terry Loss (L1182)
  - Policy with Value Head（共享 backbone） (L1191)
  - Gather Token-Level Log Probabilities (L1215)
  - PPO 总损失（policy clipped + value clipped） (L1225)
  - Reward Shaping（KL + 末位 RM） (L1242)

### scaling-laws-fill.md — Scaling Laws 填空 (828 lines)
- 练习 1: Transformer 参数量计算（Level 1-2） (L31)
  - 背景 (L33)
  - 任务 (L39)
  - 提示 (L103)
- 练习 2: 训练 FLOPs 估算（Level 2） (L152)
  - 背景 (L154)
  - 任务 (L160)
  - 提示 (L232)
- 练习 3: Chinchilla 最优配比（Level 2-3） (L281)
  - 背景 (L283)
  - 任务 (L289)
  - 提示 (L368)
- 练习 4: Loss 预测与幂律拟合（Level 2-3） (L413)
  - 背景 (L415)
  - 任务 (L421)
  - 提示 (L499)
- 练习 5: 训练预算规划器（Level 3） (L546)
  - 背景 (L548)
  - 任务 (L554)
  - 提示 (L693)
- MLM 代码训练模式 (L750)
  - Transformer 参数量计算公式 (L754)
  - FLOPs 估算 (6ND) 与训练时间 (L774)
  - Chinchilla 最优配比 (L788)
  - Loss 幂律拟合 (L813)

### sft-training-fill.md — SFT 训练 Pipeline 填空 (784 lines)
- 练习 1: ChatML 格式化（Level 2） (L24)
  - 背景 (L26)
  - 任务 (L41)
  - 提示 (L91)
- 练习 2: SFT Loss Masking（Level 2） (L130)
  - 背景 (L132)
  - 任务 (L143)
  - 提示 (L181)
- 练习 3: SFT DataCollator 实现（Level 3） (L215)
  - 背景 (L217)
  - 任务 (L227)
  - 提示 (L311)
- 练习 4: 梯度累积训练循环（Level 2） (L361)
  - 背景 (L363)
  - 任务 (L369)
  - 提示 (L438)
- 练习 5: 完整 SFT 训练循环（Level 3） (L477)
  - 背景 (L479)
  - 任务 (L489)
  - 提示 (L646)
- MLM 代码训练模式 (L703)
  - ChatML 格式解析 (L707)
  - SFT Loss Masking (L723)
  - DataCollator 动态 Padding (L733)
  - 梯度累积训练循环 (L770)

### tokenization-fill.md — Tokenization 分词填空 (717 lines)
- 练习 1: BPE 训练算法（Level 2） (L19)
  - 背景 (L21)
  - 任务 (L27)
  - 提示 (L78)
- 练习 2: BPE 编码（分词）（Level 2） (L130)
  - 背景 (L132)
  - 任务 (L136)
  - 提示 (L169)
- 练习 3: WordPiece vs BPE 对比（Level 2-3） (L218)
  - 背景 (L220)
  - 任务 (L228)
  - 提示 (L275)
- 练习 4: 中文分词的特殊挑战（Level 2） (L322)
  - 背景 (L324)
  - 任务 (L330)
  - 提示 (L389)
- 练习 5: 完整 Tokenizer 实现（Level 3） (L440)
  - 背景 (L442)
  - 任务 (L446)
  - 提示 (L575)
- MLM 代码训练模式 (L642)
  - BPE 训练核心循环 (L646)
  - BPE 编码（分词） (L684)
  - WordPiece 似然增益得分 (L703)

### transformer-quiz.md — Transformer 概念题 (204 lines)
- 第 1 题：为什么除以 $\sqrt{d_k}$？ (L14)
- 第 2 题：Multi-Head vs Single-Head 的优势 (L43)
- 第 3 题：位置编码的必要性 (L72)
- 第 4 题：残差连接的作用 (L101)
- 第 5 题：Encoder vs Decoder 的关键区别 (L132)
- 第 6 题：LayerNorm vs BatchNorm (L164)
- 自评标准 (L196)

<!-- AUTO-GENERATED-CONTENT-INDEX-END -->
