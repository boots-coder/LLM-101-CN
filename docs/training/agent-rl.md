---
title: "Agent 强化学习"
description: "Agent-RL 训练范式：环境交互、奖励设计、GRPO/PPO 训练、异步 Rollout 与生产级框架"
topics: [agent-rl, reinforcement-learning, GRPO, PPO, rollout, reward-design, slime, agentic-training]
prereqs: [training/alignment, applications/agents]
---
# Agent 强化学习

> Agent-RL 是大模型训练的新前沿——让模型在真实环境中通过交互学习完成任务，而非仅仅模仿人类偏好

## 在大模型体系中的位置

```
预训练 (Pre-training)           → 学习语言知识和世界知识
    ↓
监督微调 (SFT)                  → 学习指令跟随能力
    ↓
偏好对齐 (RLHF / DPO / GRPO)   → 学习人类偏好，安全有用
    ↓
Agent-RL  ← 你在这里             → 在环境中学习使用工具、规划、纠错
  ├── 代码执行环境               → 写代码并运行，根据结果学习
  ├── Web 浏览环境               → 搜索、点击、填表，完成网页任务
  ├── 工具调用环境               → 学习何时、如何调用 API/工具
  └── 多 Agent 对抗环境          → 通过竞争博弈提升能力
```

偏好对齐（如 DPO / GRPO）让模型学会"说什么更好"，但 Agent-RL 更进一步——让模型学会"怎么做事"。这里的"做事"意味着模型需要在真实环境中执行动作、观察结果、调整策略，最终完成任务。

::: tip 一个直观的类比
偏好对齐像是让一个人读了很多"好回答 vs 坏回答"的例子，Agent-RL 则像是让一个人实际去编程、上网搜索、使用工具——只有在实战中才能学会这些技能。
:::

## 为什么需要 Agent-RL？

### 静态对齐的局限

传统对齐方法（DPO / GRPO on text）在以下场景遇到瓶颈：

| 能力 | 静态对齐能否解决 | 为什么 |
|------|:---:|--------|
| 多轮工具调用 | ❌ | 没有环境反馈，模型无法学习工具的真实行为 |
| 错误恢复 | ❌ | 偏好数据只有"好/坏"，没有"出错后怎么补救" |
| 长期规划 | ❌ | 单轮生成无法训练跨越多步的策略 |
| 代码调试 | ❌ | 不执行代码就无法知道是否正确 |
| 探索式搜索 | ❌ | 静态数据无法覆盖所有可能的搜索路径 |

### Agent-RL 的核心洞察

Agent-RL 的关键思想可以概括为三点：

1. **环境即老师**：奖励信号来自任务完成，而非人类偏好评分。代码能不能跑、网页任务能不能完成、答案对不对——这些都是客观的、可验证的信号。

2. **交互即数据**：训练数据不是预先收集的偏好对，而是模型与环境实时交互产生的轨迹 (trajectory)。每条轨迹包含多步动作和观察。

3. **在线即进化**：模型在训练过程中不断生成新的轨迹，用最新策略与环境交互，避免了离线数据的分布偏移问题。

```python
# 传统对齐：静态偏好对
data = {"prompt": "...", "chosen": "好回答", "rejected": "差回答"}

# Agent-RL：动态交互轨迹
trajectory = [
    {"role": "user", "content": "帮我写一个排序算法并测试"},
    {"role": "assistant", "content": "```python\ndef sort(arr): ...```\n我来执行这段代码"},
    {"role": "tool", "content": "执行结果: [1, 2, 3, 5, 8] ✓"},   # 环境反馈
    {"role": "assistant", "content": "排序正确，让我再测试边界情况..."},
    {"role": "tool", "content": "执行结果: [] ✓"},                  # 环境反馈
    # reward = 1.0 (所有测试通过)
]
```

## Agent-RL 核心架构

### 经典 Agent-RL 循环

Agent-RL 的训练循环与传统 RL 本质相同，但策略 (policy) 是一个大语言模型，动作空间 (action space) 是自然语言：

```
┌──────────────────────────────────────────────────────────┐
│                  Agent-RL Training Loop                   │
│                                                          │
│   ┌──────────┐  Action (text/tool call) ┌────────────┐  │
│   │  Policy  │ ───────────────────────→ │Environment │  │
│   │  (LLM)   │                          │(Sandbox/   │  │
│   │          │ ←─────────────────────── │ Web/Tools) │  │
│   └──────────┘  Observation (result)    └────────────┘  │
│        ↑                                      │         │
│        │            ┌──────────┐              │         │
│        └─────────── │  Reward  │ ←────────────┘         │
│     Policy Update   │  Signal  │                        │
│                     └──────────┘                        │
└──────────────────────────────────────────────────────────┘
```

### 三大核心组件

**1. 策略模型 (Policy)**

策略就是正在训练的 LLM。它接收当前的对话历史（包括之前的动作和环境观察），输出下一步动作。动作可以是：
- 纯文本回答（推理、思考）
- 工具调用（函数调用、代码执行）
- 结构化动作（API 请求、网页操作）

**2. 环境 (Environment)**

环境负责执行 Agent 的动作并返回观察。典型的环境包括：

| 环境类型 | 输入 | 输出 | 示例 |
|---------|------|------|------|
| 代码沙箱 | Python/C++ 代码 | 执行结果/报错 | Jupyter、Docker |
| Web 浏览器 | 点击/输入/导航 | 页面状态 | Playwright、Selenium |
| 工具 API | 函数调用 JSON | API 返回结果 | MCP Server、OpenAPI |
| 数据库 | SQL 查询 | 查询结果 | SQLite、PostgreSQL |
| 竞技场 | 博弈动作 | 对手响应 | 棋盘游戏、辩论 |

**3. 奖励 (Reward)**

奖励信号告诉模型任务完成得如何。Agent-RL 的奖励通常是客观可验证的：

```python
# 代码任务：测试用例是否通过
def code_reward(trajectory) -> float:
    test_results = run_tests(trajectory.final_code)
    return sum(test_results) / len(test_results)  # 通过率

# 数学任务：答案是否正确
def math_reward(trajectory) -> float:
    return 1.0 if trajectory.answer == ground_truth else 0.0

# Web 任务：是否完成目标操作
def web_reward(trajectory) -> float:
    return 1.0 if check_goal_achieved(trajectory.final_state) else 0.0
```

## 训练算法

Agent-RL 的核心训练算法仍然是 GRPO 和 PPO——与偏好对齐中使用的相同算法，但应用场景从单轮文本生成扩展到了多步环境交互。

### GRPO for Agents

GRPO (Group Relative Policy Optimization) 是 DeepSeek 提出并在 DeepSeek-R1 中验证的算法，其核心思想是用 **组内相对排名** 来估计优势，省去 Critic 网络。

在 Agent-RL 场景下，GRPO 的工作流程：

1. 对每个 prompt，生成 $G$ 条完整轨迹（每条轨迹包含多轮工具调用）
2. 每条轨迹获得一个任务完成奖励 $r_i$
3. 计算组内优势：$A_i = \frac{r_i - \text{mean}(r_1, ..., r_G)}{\text{std}(r_1, ..., r_G)}$
4. 用 clipped policy gradient 更新策略

$$
\mathcal{L}_{\text{GRPO}} = -\frac{1}{G} \sum_{i=1}^{G} \min\left(\rho_i A_i, \ \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i\right)
$$

其中 $\rho_i = \frac{\pi_\theta(o_i | q)}{\pi_{\theta_{\text{old}}}(o_i | q)}$ 是重要性采样比。

::: details slime 中的 GRPO 实现细节
slime 在 `ppo_utils.py` 中实现了 GRPO 的 advantage 计算和 policy loss：

```python
# 1. GRPO returns: 将标量奖励广播到每个 token
def get_grpo_returns(rewards, kl):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns

# 2. Policy loss: clipped surrogate objective
def compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None):
    ratio = (-ppo_kl).exp()
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses = torch.maximum(pg_losses1, pg_losses2)
    # 支持 Dual-clip PPO (eps_clip_c)
    if eps_clip_c is not None:
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses = torch.where(
            advantages < 0,
            torch.min(pg_losses3, clip_pg_losses),
            clip_pg_losses,
        )
    return clip_pg_losses, clipfrac
```

关键配置参数（`arguments.py`）：
- `--advantage-estimator grpo` — 使用 GRPO 估计优势
- `--eps-clip 0.2` — PPO clip 范围
- `--n-samples-per-prompt 8` — 每个 prompt 采样的轨迹数
- `--normalize-advantages` — 是否归一化优势
- `--disable-grpo-std-normalization` — 是否禁用标准差归一化（Dr.GRPO 变体）
:::

### PPO for Agents

经典 PPO 使用 Value Network（Critic）来估计每个状态的价值，通过 GAE (Generalized Advantage Estimation) 计算优势：

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

::: details slime 中的 GAE 实现
slime 提供了两种 GAE 计算方式：

```python
# 1. 朴素 GAE: O(T) 时间复杂度
def vanilla_gae(rewards, values, gamma, lambd):
    B, T = rewards.shape
    lastgaelam = torch.zeros(B, device=device, dtype=dtype)
    adv_rev = []
    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        adv_rev.append(lastgaelam)
    return torch.stack(adv_rev[::-1], dim=1), ...

# 2. Chunked GAE: 利用 parallel prefix scan 加速
#    将序列分块，块内并行计算，块间递推传播
#    时间复杂度从 O(T) 降到 O(T / chunk_size)
def chunked_gae(rewards, values, gamma, lambd, chunk_size=128):
    # 构造块内并行扫描核 M[i,j] = w^(j-i) if j >= i
    # S_local = Δ @ M (块内并行)
    # S_global = S_local + s_prev * pow_vec (块间递推)
    ...
```

Chunked GAE 灵感来自 FlashLinearAttention，是 slime 的性能优化亮点之一。
:::

### GRPO vs PPO：Agent-RL 场景下的选择

| 维度 | GRPO | PPO |
|------|------|-----|
| Critic 网络 | 不需要 | 需要（额外显存和计算） |
| 优势估计 | 组内相对排名 | GAE（需要 Value Function） |
| 多样性要求 | 需要组内有差异 | 不需要组采样 |
| 信用分配 | 粗粒度（整条轨迹同一奖励） | 细粒度（每步有 value 估计） |
| 实现复杂度 | 较低 | 较高 |
| 适用场景 | 结果可验证的任务 | 需要密集反馈的长轨迹任务 |

::: tip 实践建议
目前大多数 Agent-RL 工作使用 **GRPO**，因为 Agent 任务通常有明确的完成/失败判断，组内相对排名已经足够。PPO 在需要密集步级奖励的场景更有优势。slime 默认使用 `--advantage-estimator grpo`。
:::

### Agent-RL 与文本 GRPO 的关键差异

虽然算法公式相同，但 Agent-RL 场景下存在几个关键差异：

1. **轨迹长度不同**：文本 GRPO 的 rollout 通常是几百个 token 的单轮生成；Agent-RL 的轨迹可能包含数十轮工具调用，总长度达到数千甚至上万 token。

2. **奖励稀疏性**：文本 GRPO 在生成结束后即可打分；Agent-RL 通常只在整条轨迹结束时才获得奖励（例如所有测试用例是否通过），中间步骤没有奖励信号。

3. **环境交互开销**：Agent-RL 的 rollout 不仅需要模型推理，还需要实际执行环境操作（运行代码、请求 API），这大大增加了 rollout 时间。

4. **生成不确定性**：环境返回的结果是不可控的（网络延迟、API 错误、超时），需要更健壮的 rollout 管理。

## Rollout 生成

Rollout 是 Agent-RL 训练的核心瓶颈。一次 rollout 需要：模型生成动作 → 环境执行 → 模型观察结果 → 生成下一步动作 → ... → 任务结束 → 计算奖励。

### 同步 Rollout

最直接的方式：训练和 rollout 交替进行。

```
时间线:
[----Rollout----][--Train--][----Rollout----][--Train--]

Rollout 阶段: 用当前策略生成 B 条轨迹，计算奖励
Train 阶段:   用生成的轨迹更新策略参数
```

优点是实现简单，on-policy 数据质量高；缺点是 GPU 利用率低——训练时推理引擎空闲，推理时训练引擎空闲。

### 异步 Rollout

slime 的核心创新之一是 **异步训练-Rollout 流水线**：Rollout 生成和训练同时进行，互不等待。

```
时间线 (异步):
Rollout:  [--Rollout 1--][--Rollout 2--][--Rollout 3--][--Rollout 4--]
Train:           [--Train 1--][--Train 2--][--Train 3--][--Train 4--]
                 ↑            ↑
                 使用 Rollout 1    使用 Rollout 2
                 的数据训练        的数据训练
```

::: details slime 的全异步 Rollout 实现
slime 在 `examples/fully_async/` 中提供了全异步 rollout 的参考实现。核心思路：

```python
class AsyncRolloutWorker:
    """持续运行的异步 rollout worker"""

    async def continuous_worker_loop(self):
        """不断从 data_buffer 取数据、启动生成任务"""
        active_tasks = set()
        while self.running:
            # 清理已完成任务
            done_tasks = {t for t in active_tasks if t.done()}
            active_tasks -= done_tasks

            # 取新数据，启动异步生成
            while len(active_tasks) < max_concurrent_tasks:
                samples = self.data_buffer.get_samples(1)
                task = asyncio.create_task(
                    generate_and_rm_group(self.args, group, ...)
                )
                task.add_done_callback(lambda t: self.output_queue.put(t.result()))
                active_tasks.add(task)

            await asyncio.sleep(1)
```

使用方式只需两步配置：
1. 使用异步训练驱动：`train_async.py`
2. 指定 rollout 函数：`--rollout-function-path fully_async_rollout.generate_rollout_fully_async`

训练端只需从 output queue 中取出已完成的轨迹，无需等待 rollout 完成。
:::

### SGLang 高吞吐推理

slime 使用 [SGLang](https://github.com/sgl-project/sglang) 作为推理引擎，通过 Router 分发请求到多个 SGLang worker：

```
                    ┌──── SGLang Worker 0 (GPU 0)
                    │
Request ──→ Router ─┼──── SGLang Worker 1 (GPU 1)
                    │
                    └──── SGLang Worker 2 (GPU 2)
```

关键优化：
- **Continuous batching**：新请求可以动态加入正在处理的 batch
- **RadixAttention**：共享前缀的 KV cache 复用，减少重复计算
- **DP rank balancing**：在多 data-parallel 副本间均衡负载

```python
# slime 中的 SGLang 请求构造（sglang_rollout.py）
payload = {
    "sampling_params": {
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
        "max_new_tokens": args.rollout_max_response_len,
        "stop": args.rollout_stop,
        "stop_token_ids": args.rollout_stop_token_ids,
    },
    "input_ids": prompt_ids,
    "return_logprob": True,   # 返回 log-prob 用于 PPO/GRPO 计算
}
output = await post(router_url, payload)
```

### 多轮轨迹：Agent Rollout 的特殊结构

Agent 的一次 rollout 不是简单的 "输入 → 输出"，而是多轮交互：

```
Prompt:      "请帮我分析这个 CSV 文件的数据分布"
   ↓
Action 1:    model.generate() → "我先读取文件看看结构\n```python\nimport pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())\n```"
   ↓
Observation 1: execute_code() → "   name  age  score\n0  Alice   25     89\n..."
   ↓
Action 2:    model.generate() → "数据有 3 列，我来画分布图\n```python\ndf['score'].hist()\nplt.savefig('dist.png')\n```"
   ↓
Observation 2: execute_code() → "图片已保存到 dist.png"
   ↓
Action 3:    model.generate() → "从分布图来看，成绩呈正态分布..."
   ↓
Reward:      reward_fn(trajectory) → 0.8  (正确分析了分布)
```

在 slime 中，这样的多轮交互通过 **自定义生成函数** (`--custom-generate-function-path`) 实现。每个 `Sample` 对象会累积多轮的 tokens、response 和 log_probs：

```python
# 每次 generate 调用后，Sample 累积新内容
sample.tokens = sample.tokens + new_response_tokens
sample.response_length += len(new_response_tokens)
sample.response += output["text"]
sample.rollout_log_probs += new_response_log_probs
```

## 奖励设计

奖励设计是 Agent-RL 中最关键的工程问题之一。好的奖励函数需要准确反映任务目标，同时易于规模化获取。

### Rule-based Rewards（规则奖励）

最简单也最可靠的方式——用确定性规则判断任务完成度：

```python
# 代码任务：单测通过率
def code_reward(sample) -> float:
    code = extract_code(sample.response)
    results = run_test_suite(code, sample.label["test_cases"])
    return sum(results) / len(results)

# 数学任务：答案匹配
def math_reward(sample) -> float:
    answer = extract_answer(sample.response)
    return 1.0 if normalize(answer) == normalize(sample.label) else 0.0

# 格式奖励：输出是否符合格式要求
def format_reward(sample) -> float:
    import re
    if re.search(r"<think>.*?</think>.*<answer>.*?</answer>", sample.response, re.DOTALL):
        return 0.2  # 格式正确加分
    return 0.0
```

::: tip slime 的奖励函数接口
通过 `--custom-rm-path` 指定自定义奖励函数。slime 支持同步和异步奖励计算：

```python
# 同步奖励
async def custom_rm(args, sample):
    return compute_reward(sample)

# 批量奖励（适合需要组内比较的场景）
async def batched_custom_rm(args, samples):
    return [compute_reward(s) for s in samples]
```
:::

### Verifiable Environments (RLVE)

RLVE (Reinforcement Learning with Verifiable Environments) 是一种更系统化的奖励获取方式：环境能够 **程序化地生成问题** 并 **自动验证答案**。

```
┌──────────────────────────────────────────────────┐
│            Verifiable Environment                │
│                                                  │
│  generate_problem()  → Generate new problems     │
│  verify_answer()     → Verify answer correctness │
│  adaptive_difficulty → Adjust by model ability   │
│                                                  │
└──────────────────────────────────────────────────┘
```

RLVE 的核心优势：
- **无限数据**：环境可以程序化生成无穷多的训练问题
- **零标注成本**：答案由环境自动验证，不需要人工标注
- **自适应难度**：根据模型当前能力动态调整题目难度

::: warning 奖励黑客 (Reward Hacking) 风险
Agent 可能找到奖励函数的漏洞——比如在代码任务中直接硬编码测试用例的答案，而不是写通用算法。防御策略：
1. 使用足够多样化的测试用例
2. 定期检查生成的轨迹
3. 添加格式/风格约束作为辅助奖励
4. 使用 hidden test cases（模型看不到的测试用例）
:::

### Process Rewards vs Outcome Rewards

| 维度 | Outcome Reward（结果奖励） | Process Reward（过程奖励） |
|------|:---:|:---:|
| 评判时机 | 轨迹结束后 | 每一步 |
| 信号密度 | 稀疏（一个标量） | 密集（每步一个分数） |
| 信用分配 | 困难（哪一步导致了失败？） | 容易（直接标出错误步骤） |
| 实现复杂度 | 低 | 高（需要步级评判能力） |
| 典型场景 | 代码测试、答案匹配 | 数学推理、多步规划 |

实践中，大多数 Agent-RL 工作使用 **outcome reward**，因为它最容易实现且信号最可靠。Process reward 在需要精确信用分配的场景（如数学逐步推理）更有价值。

## 生产级训练框架：slime 架构

slime 是 GLM-4.5 / 4.6 / 4.7 / 5 背后的 RL 训练框架，由清华大学 THUDM 团队开发。它的设计目标是为 RL scaling 提供高性能、灵活的 post-training 基础设施。

### 整体架构

```
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────────┐
│                  │      │                  │      │                      │
│    Training      │ ←─── │   Data Buffer    │ ←─── │      Rollout         │
│   (Megatron)     │      │                  │      │  (SGLang + Router)   │
│                  │ ───→ │                  │      │                      │
│  · Grad Update   │      │  · Prompt Mgmt   │      │  · Trajectory Gen    │
│  · Param Sync    │      │  · Data Buffer   │      │  · Reward Compute    │
│  · Checkpoint    │      │  · Custom Data   │      │  · Dynamic Filter    │
│                  │      │                  │      │                      │
└────────┬─────────┘      └──────────────────┘      └──────────┬───────────┘
         │                                                     ↑
         └──────────────── Parameter Sync ─────────────────────┘
```

### 三大模块详解

**1. Training 模块 (Megatron)**

基于 Megatron-LM 构建，负责：
- 从 Data Buffer 读取 rollout 生成的数据
- 计算 PPO / GRPO loss 并执行梯度更新
- 训练完成后将最新参数同步到 Rollout 模块的 SGLang 引擎

支持的分布式策略：
- Tensor Parallelism (TP)
- Pipeline Parallelism (PP)
- Context Parallelism (CP)
- Data Parallelism (DP)

**2. Rollout 模块 (SGLang + Router)**

负责生成训练数据（轨迹 + 奖励）：
- 使用 SGLang 作为高吞吐推理引擎
- Router 在多个 SGLang worker 之间分发请求
- 支持自定义生成函数和奖励函数
- 支持动态采样过滤 (dynamic sampling filter)

```python
# Rollout 的核心流程（简化版）
async def generate_rollout_async(args, rollout_id, data_source):
    state = GenerateState(args)
    data = []
    while len(data) < args.rollout_batch_size:
        # 1. 从 buffer 取 prompt
        samples = data_source(args.over_sampling_batch_size)
        # 2. 提交异步生成任务
        state.submit_generate_tasks(samples)
        # 3. 等待完成并收集结果
        done, state.pendings = await asyncio.wait(
            state.pendings, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            group = task.result()
            # 4. 动态过滤（可选）
            if dynamic_filter(group).keep:
                data.append(group)
    return data
```

**3. Data Buffer**

Training 和 Rollout 之间的桥梁：
- 管理 prompt 数据集的初始化和分发
- 缓存 Rollout 生成的轨迹数据
- 支持 partial rollout（未完成的轨迹可以回收继续）
- 支持 over-sampling（生成多于训练需要的数据，过滤后使用）

### 参数同步与 Colocate 模式

slime 支持两种部署模式：

**分离模式 (Separate)**：训练和推理使用不同的 GPU
```bash
# 4 GPU 训练 + 4 GPU 推理
--actor-num-gpus-per-node 4 --rollout-num-gpus 4
```

**共享模式 (Colocate)**：训练和推理共享同一组 GPU
```bash
# 8 GPU 共享，交替使用
--colocate --actor-num-gpus-per-node 8
```

Colocate 模式下，训练时推理引擎 offload 到 CPU，推理时训练引擎 offload：
```
Colocate 时间线:
[Rollout (GPU)] → [Offload Rollout→CPU] → [Train (GPU)] → [Offload Train→CPU] → [Rollout (GPU)]
```

### 支持的模型

slime 支持主流开源模型系列：
- **GLM 系列**：GLM-4.5 / 4.6 / 4.7 / 5
- **Qwen3 系列**：Qwen3, Qwen3MoE, Qwen3Next, Qwen2.5
- **DeepSeek 系列**：DeepSeek V3, V3.1, DeepSeek R1
- **Llama 系列**：Llama 3

## Agentic Training 的特殊挑战

Agent-RL 相比文本 RL 面临一系列独特的工程和算法挑战：

### 1. 长轨迹问题

Agent 的一次任务可能涉及数十轮工具调用，轨迹长度可达数万 token。这导致：
- **显存压力**：KV cache 随轨迹长度线性增长
- **梯度消失/爆发**：反向传播经过的步数太多
- **训练效率低**：一条轨迹占据大量计算资源

应对策略：
- Context Parallelism (CP)：将长序列切分到多个 GPU
- 截断轨迹：设置最大轨迹长度 (`--rollout-max-response-len`)
- Partial Rollout：未完成的轨迹可以保存并在下一轮继续

### 2. 稀疏奖励与信用分配

Agent 通常只在任务结束时获得奖励（成功/失败），中间的十几步动作没有奖励信号。模型很难判断"哪一步动作导致了成功或失败"。

```
动作 1: 搜索文档       → 无奖励
动作 2: 提取关键信息    → 无奖励
动作 3: 写代码         → 无奖励
动作 4: 运行代码 (报错)  → 无奖励
动作 5: 修复 bug       → 无奖励
动作 6: 重新运行 (通过)  → reward = 1.0  ← 唯一的奖励信号
```

应对策略：
- GRPO 的组内对比：同一 prompt 的不同轨迹对比，间接实现信用分配
- REINFORCE++ baseline：用 KL 惩罚作为逐 token 的奖励塑形

::: details slime 中的 REINFORCE++ 实现
```python
def get_reinforce_plus_plus_returns(rewards, kl, loss_masks, ...):
    """REINFORCE++ 的 discounted returns 计算"""
    for i in range(len(rewards)):
        # 逐 token 的奖励 = -kl_coef * KL_penalty
        token_level_rewards = -kl_coef * masked_kl
        # 最后一个 token 加上任务奖励
        token_level_rewards[last_idx] += rewards[i]
        # 反向递推计算 discounted return
        for t in reversed(range(token_level_rewards.size(0))):
            running_return = token_level_rewards[t] + gamma * running_return
            returns_for_seq[t] = running_return
    return returns
```
:::

### 3. 环境多样性

不同任务需要不同的环境（代码沙箱、Web 浏览器、API 服务），每种环境的延迟和可靠性都不同。

slime 通过以下机制应对：
- **自定义生成函数** (`--custom-generate-function-path`)：每种环境实现自己的 `generate` 函数
- **自定义奖励函数** (`--custom-rm-path`)：每种环境实现自己的奖励逻辑
- **Per-sample 配置**：每个 sample 可以携带自己的 `generate_function_path`

### 4. 异步解耦

Training 和 Rollout 的速度可能差异很大——一条复杂的 Agent 轨迹可能需要几分钟生成，而训练一个 batch 只需要几秒。异步解耦让两者以各自的最优速度运行。

::: warning Off-Policy 数据风险
异步模式下，Rollout 使用的模型参数可能已经过时（训练已经更新了参数但还没同步到 Rollout）。这导致生成的数据是 **off-policy** 的。slime 提供了几种缓解机制：
- **OPSM (Off-Policy Sequence Masking)**：检测序列级 KL 散度，masking 掉偏离太大的样本
- **Partial Rollout + Loss Mask**：标记 off-policy 部分的 token，在 loss 计算时屏蔽
- **KL 惩罚**：通过 `--kl-coef` 或 `--kl-loss-coef` 限制策略偏移幅度
:::

### 5. 训练加速

Agent-RL 的 rollout 阶段通常占据 90% 以上的训练时间。slime 生态中的 APRIL 项目专门解决这个问题：

- **Active Partial Rollouts**：智能地过量提交生成请求
- **Early Termination**：当已收集足够有效样本时，主动 abort 剩余请求
- **Over-sampling + Filter**：生成更多样本，过滤后取最优子集

## 实际应用案例

slime 已被多个研究项目和生产系统使用，以下是几个代表性案例：

### P1：物理奥赛推理

[P1](https://prime-rl.github.io/P1/) 是一系列完全通过 RL 训练的物理推理模型。核心创新：
- **多阶段渐进训练**：从简单题到难题，逐步提升推理能力
- **自适应可学习性调整**：根据模型当前水平动态选择合适难度的训练数据
- **稳定化机制**：防止 RL 训练中的性能崩塌

### RLVE：400 个可验证环境

[RLVE](https://github.com/Zhiyuan-Zeng/RLVE) 构建了 400 个可验证环境，涵盖数学、逻辑、编程等领域。关键特性：
- 每个环境可以程序化生成无穷多的问题
- 答案由算法自动验证（不需要人工标注）
- 环境难度根据模型能力自适应调整
- 多环境联合训练，跨领域能力迁移

### TritonForge：训练模型写 GPU Kernel

[TritonForge](https://github.com/RLsys-Foundation/TritonForge) 用 Agent-RL 训练 LLM 自动生成优化的 GPU kernel：
- **阶段一 (SFT)**：在 Triton kernel 数据集上做 SFT
- **阶段二 (RL)**：用多轮编译反馈做 RL 训练——模型写代码 → 编译 → 收到编译结果 → 修改代码 → 再编译
- 奖励信号：编译是否成功 + kernel 性能（延迟、吞吐）

### OpenClaw-RL：个性化对话 Agent

[OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL) 训练个性化的对话 Agent：
- 使用 GRPO + 二元反馈（从后续对话状态推断用户满意度）
- slime 的异步 RL 架构让训练不干扰 API serving
- 支持 on-policy 蒸馏：从后续反馈中提取 hindsight hints

### qqr (ArenaRL)：开放式 Agent 进化

[qqr](https://github.com/Alibaba-NLP/qqr) 实现了基于竞技场的 Agent 进化：
- **ArenaRL 算法**：通过锦标赛排名（循环赛、淘汰赛）获取相对奖励
- **MCP 集成**：使用 Model Context Protocol 标准化工具环境
- 解决了开放式任务中"判别退化"(discriminative collapse) 的问题

## Agent-RL vs 传统对齐

| 维度 | 传统对齐 (DPO/GRPO-text) | Agent-RL |
|------|:---:|:---:|
| **数据来源** | 静态偏好对 | 动态环境交互轨迹 |
| **奖励信号** | 人类偏好 / 奖励模型 | 任务完成信号（可验证） |
| **轨迹结构** | 单轮生成 | 多轮工具调用 |
| **环境** | 无 | 代码沙箱 / Web / 工具 API |
| **训练速度** | 快（纯文本生成） | 慢（需要环境交互） |
| **可扩展瓶颈** | 数据规模和质量 | 环境多样性和吞吐 |
| **奖励可靠性** | 依赖标注质量 | 客观可验证 |
| **探索能力** | 有限（受数据分布约束） | 强（可在环境中自由探索） |

::: tip 两者不是替代关系
Agent-RL 不是要取代传统对齐，而是建立在对齐基础上。典型的训练流程是：SFT → DPO/GRPO (文本对齐) → Agent-RL (环境交互)。文本对齐让模型学会基本的沟通和推理能力，Agent-RL 在此基础上训练模型完成实际任务。
:::

## 苏格拉底时刻

1. **Agent-RL 的奖励信号真的比人类偏好更好吗？** 可验证的任务完成信号是客观的、无噪声的，但它只能覆盖"能自动判断对错"的任务。对于开放式创意任务、情感交互等，仍然需要人类偏好。Agent-RL 和传统对齐各有适用范围。

2. **为什么 GRPO 在 Agent-RL 中如此流行，而不是 PPO？** PPO 需要一个 Value Function 来估计每个状态的价值，但 Agent 轨迹的状态空间极大（包含整个对话历史 + 环境状态），训练一个准确的 Value Function 非常困难。GRPO 绕开了这个问题，只需要对比同一 prompt 的不同轨迹。

3. **异步 Rollout 引入的 off-policy 偏差真的可以接受吗？** 这是一个工程与效果的 trade-off。实践中，适度的 off-policy（1-2 步延迟）对最终效果影响很小，但带来的训练效率提升是巨大的。slime 的 OPSM 机制可以自动过滤偏离太大的样本。

4. **如果 Agent 学会了"作弊"——比如硬编码测试用例的答案——怎么办？** 这就是 reward hacking。核心对策是：让环境不可预测（随机测试用例）、使用 hidden test、添加代码风格等辅助奖励。RLVE 的"自适应难度"也有助于让模型持续面对新挑战。

5. **Agent-RL 的规模化瓶颈在哪里？** 不在算法，而在环境。算法可以用更多 GPU 加速，但环境的吞吐量是有限的（代码执行、API 调用都有延迟）。这就是为什么 slime 的异步架构如此重要——它让环境交互的延迟不阻塞训练。

## 面试考点

**Q: Agent-RL 和传统 RLHF 的本质区别是什么？**

传统 RLHF 的奖励来自人类偏好（或偏好代理模型），数据是静态的偏好对。Agent-RL 的奖励来自环境交互（任务是否完成），数据是模型与环境实时交互的轨迹。Agent-RL 的轨迹是多步的（多轮工具调用），而传统 RLHF 通常是单步的（一次性生成完整回答）。

**Q: 为什么 Agent-RL 的 rollout 比文本 RL 慢得多？**

因为 Agent-RL 的每次 rollout 不仅需要模型推理，还需要实际执行环境操作（运行代码、调用 API、渲染网页等）。一条轨迹可能包含 10-50 次"生成-执行"循环，每次环境执行都有额外延迟。这就是为什么异步 rollout 和高吞吐推理引擎（如 SGLang）在 Agent-RL 中至关重要。

**Q: slime 架构中 Training、Rollout、Data Buffer 三个模块各自的职责是什么？**

- **Training (Megatron)**：从 Data Buffer 读取 rollout 数据，计算 loss（PPO/GRPO），执行梯度更新，训练完成后同步参数到 Rollout 模块
- **Rollout (SGLang + Router)**：用当前策略生成轨迹，计算奖励，将结果存入 Data Buffer
- **Data Buffer**：桥梁模块，管理 prompt 的分发、rollout 数据的缓存、partial rollout 的回收

**Q: GRPO 中的 "Group" 在 Agent-RL 场景下具体指什么？**

对每个 prompt（任务描述），采样 $G$ 条完整轨迹。每条轨迹代表 Agent 解决同一任务的不同尝试（不同的工具调用序列、不同的推理路径）。Group 就是这 $G$ 条轨迹的集合，它们共享同一个 prompt 但有不同的执行路径和最终奖励。

**Q: 什么是 RLVE？它解决了什么问题？**

RLVE (Reinforcement Learning with Verifiable Environments) 使用可验证环境来生成训练数据。每个环境可以程序化生成问题并自动验证答案，解决了 Agent-RL 中训练数据获取困难和标注成本高的问题。通过 400 个环境的联合训练，模型可以获得跨领域的泛化能力。

**Q: 异步 Rollout 的 off-policy 问题如何缓解？**

slime 提供了多种机制：(1) OPSM (Off-Policy Sequence Masking) 检测并 mask 偏离太大的序列；(2) Partial Rollout + Loss Mask 标记 off-policy token 并在 loss 中屏蔽；(3) KL 惩罚限制策略偏移幅度。实践中适度的 off-policy 对效果影响很小，但显著提升训练效率。

## 推荐资源

- [slime 项目](https://github.com/THUDM/slime) — GLM-5 背后的 RL 训练框架
- [slime 博客: An SGLang-Native Post-Training Framework for RL Scaling](https://lmsys.org/blog/2025-07-09-slime/) — 架构设计与实现细节
- [Agent-Oriented Design: An Asynchronous and Decoupled Framework for Agentic RL](https://www.notion.so/Agent-Oriented-Design-An-Asynchronous-and-Decoupled-Framework-for-Agentic-RL-2278e692d081802cbdd5d37cef76a547) — Agent-RL 的异步解耦设计思想
- [RLVE: Scaling LM RL with Adaptive Verifiable Environments](https://github.com/Zhiyuan-Zeng/RLVE) — 400 个可验证环境联合训练
- [P1: Mastering Physics Olympiads with RL](https://prime-rl.github.io/P1/) — 纯 RL 训练的物理推理模型
- [TritonForge: Agentic RL for Kernel Generation](https://github.com/RLsys-Foundation/TritonForge) — 用 Agent-RL 训练 GPU kernel 生成
- [APRIL: Accelerating RL Training with Active Partial Rollouts](https://github.com/RLsys-Foundation/APRIL) — Rollout 加速优化
- [qqr (ArenaRL): Open-Ended Agent Evolution](https://github.com/Alibaba-NLP/qqr) — 竞技场式 Agent 进化
- [DeepSeek-R1: Incentivizing Reasoning via RL](https://arxiv.org/abs/2501.12948) — GRPO 算法详解
- [SGLang](https://github.com/sgl-project/sglang) — 高吞吐 LLM 推理引擎
