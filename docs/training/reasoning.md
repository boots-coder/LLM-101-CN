---
title: "推理模型"
description: "O1/R1 范式、PRM 过程奖励模型、MCTS、GRPO for Reasoning、推理蒸馏"
topics: [O1, R1, reasoning, PRM, MCTS, test-time-compute, chain-of-thought, aha-moment]
prereqs: [training/alignment]
---
# 推理模型（O1/R1 范式）

> 推理模型通过在推理时投入更多计算来提升回答质量，是大模型从"快思考"走向"慢思考"的关键范式

## 在大模型体系中的位置

```
预训练 (Pre-training)          → 学习语言知识和世界知识
    ↓
监督微调 (SFT)                 → 学习指令跟随能力
    ↓
偏好对齐 (RLHF/DPO/GRPO)      → 学习人类偏好，安全有用
    ↓
推理增强  ← 你在这里            → 在推理时花更多计算换更好结果
  ├── PRM + Best-of-N          → 过程奖励引导搜索
  ├── MCTS + LLM               → 树搜索式推理
  ├── GRPO for Reasoning        → DeepSeek-R1 方案
  └── 推理蒸馏                  → 大模型 CoT → 小模型
```

传统大模型在生成每个 token 时花费的计算量是固定的。推理模型的核心洞察是：**困难问题需要更多的思考时间**。这与人类的 System 2 Thinking 类似——面对复杂数学题，我们不会瞬间给出答案，而是会一步步推导。

## 从 System 2 Thinking 到 Test-time Compute Scaling

Daniel Kahneman 在《Thinking, Fast and Slow》中将人类思维分为两个系统：

| 思维系统 | 特点 | LLM 对应 |
|----------|------|----------|
| **System 1** | 快速、自动、直觉 | 标准 LLM 的单次前向传播 |
| **System 2** | 慢速、刻意、逻辑 | 推理模型的多步推理过程 |

**Test-time Compute Scaling** 的核心思想：与其在训练时投入所有计算（更大模型、更多数据），不如在推理时动态分配计算资源——简单问题快速回答，困难问题深入思考。

$$
\text{Performance} \propto f(\text{Train-time Compute}, \text{Test-time Compute})
$$

OpenAI 的 O1 模型首次验证了这一思路：通过在推理时生成长链式思考（Chain-of-Thought），模型在数学、编程、科学推理等任务上取得了巨大提升。

## Reasoning Token 与 Chain-of-Thought

### 什么是 Reasoning Token？

O1 模型在生成最终答案之前，会先生成一段 **Reasoning Token**（推理 token），这些 token 不展示给用户，但对提升回答质量至关重要：

```
用户问题: "求解方程 3x + 5 = 20"

[Reasoning Tokens - 用户不可见]
<think>
我需要求解 3x + 5 = 20。
第一步：两边减去 5，得到 3x = 15。
第二步：两边除以 3，得到 x = 5。
让我验证：3 * 5 + 5 = 15 + 5 = 20，正确。
</think>

[最终回答 - 用户可见]
x = 5
```

关键区别在于：传统 CoT 需要 prompt 诱导（"Let's think step by step"），只是浅层模仿推理格式。而 O1 的推理能力是**通过 RL 训练出来的**，模型自主决定何时深入思考、何时快速回答，并能发现错误后回溯纠正。

## Process Reward Model (PRM)

### ORM vs PRM

传统的 **Outcome Reward Model (ORM)** 只对最终结果打分——答案对了给高分，错了给低分。这存在一个问题：**无法区分"方法正确但计算出错"和"方法完全错误"**。

**Process Reward Model (PRM)** 对推理的**每一步**进行打分：

```
问题: "一个矩形，长是宽的2倍，周长为24，求面积。"

Step 1: 设宽为 x，则长为 2x        → PRM 评分: 0.95 ✓
Step 2: 周长 = 2(x + 2x) = 24      → PRM 评分: 0.92 ✓
Step 3: 6x = 24, x = 4             → PRM 评分: 0.90 ✓
Step 4: 面积 = 4 × 8 = 32          → PRM 评分: 0.93 ✓

对比错误推理:
Step 1: 设宽为 x，则长为 2x        → PRM 评分: 0.95 ✓
Step 2: 周长 = x + 2x = 24         → PRM 评分: 0.15 ✗  ← PRM 在这里就能发现错误！
Step 3: 3x = 24, x = 8             → PRM 评分: 0.20 ✗
Step 4: 面积 = 8 × 16 = 128        → PRM 评分: 0.10 ✗
```

### PRM 的数学形式

给定问题 $x$ 和推理过程 $s = (s_1, s_2, \dots, s_n)$，PRM 为每一步计算一个正确性分数：

$$
\text{PRM}_\theta(x, s_1, \dots, s_k) = P(\text{step } s_k \text{ is correct} \mid x, s_1, \dots, s_{k-1})
$$

整个推理过程的分数可以用多种聚合方式：

$$
\text{Score}_{\min}(x, s) = \min_{k=1}^{n} \text{PRM}_\theta(x, s_1, \dots, s_k)
$$

$$
\text{Score}_{\prod}(x, s) = \prod_{k=1}^{n} \text{PRM}_\theta(x, s_1, \dots, s_k)
$$

$$
\text{Score}_{\text{last}}(x, s) = \text{PRM}_\theta(x, s_1, \dots, s_n)
$$

实践中 $\text{Score}_{\min}$ 效果最好——因为一个错误步骤足以导致整个推理失败。

### PRM 训练数据构建（PRM800K）

OpenAI 发布的 PRM800K 数据集包含约 80 万条**人工逐步标注**的数学推理步骤，每一步标注为 positive / negative / neutral。由于人工标注成本极高，后续工作（如 Math-Shepherd）提出了**Monte Carlo 自动标注**——对每一步，多次采样后续推理并检查最终答案，用正确率作为该步的分数：

```python
def auto_label_step(model, question, steps_so_far, correct_answer, n_samples=64):
    """Monte Carlo 估计某一步的正确概率"""
    correct_count = 0
    for _ in range(n_samples):
        completion = model.generate(
            prompt=question + "\n".join(steps_so_far),
            temperature=0.8, max_tokens=512)
        if extract_answer(completion) == correct_answer:
            correct_count += 1
    return correct_count / n_samples
```

### PRM 模型架构

PRM 的架构与奖励模型类似（参见 [alignment.md](alignment.md)），但在**步骤分隔符位置**输出分数：

```python
import torch
import torch.nn as nn

class ProcessRewardModel(nn.Module):
    """过程奖励模型：在步骤分隔符位置输出正确性分数"""
    def __init__(self, base_model, step_token_id):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)
        self.step_token_id = step_token_id

    def forward(self, input_ids, attention_mask=None):
        hidden = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True).hidden_states[-1]
        # 提取步骤分隔符位置的隐藏状态并打分
        positions = (input_ids == self.step_token_id).nonzero(as_tuple=True)
        step_hidden = hidden[positions[0], positions[1]]
        return torch.sigmoid(self.reward_head(step_hidden).squeeze(-1))
```

训练使用 BCE Loss，只在步骤位置计算损失。完整训练代码见下方代码实战。

## Monte Carlo Tree Search (MCTS)

### MCTS 核心思想

MCTS 是一种在大搜索空间中寻找最优策略的树搜索算法。在 LLM 推理中，我们将**生成过程看作树搜索**：

```
Root = Question
          │
    ┌─────┼─────────┐
    ↓     ↓         ↓
  Step1a  Step1b  Step1c     ← Different first reasoning steps
    │       │       │
  ┌─┼─┐   ┌┼┐     ┌┼┐
  ↓ ↓ ↓   ↓ ↓     ↓ ↓
 ...       ...     ...       ← Different follow-up reasoning
```

### MCTS 四步循环

MCTS 通过四步循环不断改进搜索策略：

**1. Selection（选择）**：从根节点出发，用 UCB 公式选择最有潜力的子节点：

$$
\text{UCB}(s) = \frac{Q(s)}{N(s)} + c \cdot \sqrt{\frac{\ln N(\text{parent})}{N(s)}}
$$

其中 $Q(s)$ 是累计奖励，$N(s)$ 是访问次数，$c$ 是探索系数。

**2. Expansion（扩展）**：到达叶节点后，用 LLM 生成新的推理步骤作为子节点。

**3. Simulation（模拟）**：从新节点出发，用 LLM 快速生成完整解答，检查结果是否正确。

**4. Backpropagation（回传）**：将模拟结果沿路径回传，更新所有祖先节点的统计信息。

### 代码示例：简化版 MCTS for LLM Reasoning

> MCTS 搜索实现参考了 AlphaGo 论文 (Silver et al., 2016) 的 UCB 选择策略，并适配了 LLM 推理场景。

```python
import math, random

class MCTSNode:
    """MCTS 节点：代表推理过程中的一个状态"""
    def __init__(self, state, parent=None, step_text=""):
        self.state = state            # 到当前步的完整推理文本
        self.parent = parent
        self.step_text = step_text
        self.children = []
        self.visits = 0               # 访问次数 N(s)
        self.total_reward = 0.0       # 累计奖励 Q(s)

    def ucb_score(self, c=1.414):
        if self.visits == 0:
            return float('inf')
        return (self.total_reward / self.visits +
                c * math.sqrt(math.log(self.parent.visits) / self.visits))


class MCTSReasoner:
    """将 MCTS 应用于 LLM 推理"""
    def __init__(self, llm, prm=None, max_steps=6, n_candidates=3):
        self.llm = llm
        self.prm = prm
        self.max_steps = max_steps
        self.n_candidates = n_candidates

    def search(self, question, n_iterations=50):
        root = MCTSNode(state=question)
        for _ in range(n_iterations):
            node = self._select(root)                      # 1. Selection
            if node.visits > 0 and len(node.children) == 0:
                node = self._expand(node)                  # 2. Expansion
            reward = self._simulate(node)                  # 3. Simulation
            self._backpropagate(node, reward)               # 4. Backpropagation
        return self._best_path(root)

    def _select(self, node):
        while node.children:
            node = max(node.children, key=lambda n: n.ucb_score())
        return node

    def _expand(self, node):
        for _ in range(self.n_candidates):
            next_step = self.llm.generate_step(node.state, temperature=0.7)
            child = MCTSNode(node.state + "\n" + next_step, node, next_step)
            node.children.append(child)
        return random.choice(node.children)

    def _simulate(self, node):
        completion = self.llm.complete_reasoning(node.state, self.max_steps)
        return self.prm.score(completion) if self.prm else self._check_answer(completion)

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def _best_path(self, root):
        path, node = [], root
        while node.children:
            node = max(node.children, key=lambda n: n.visits)
            path.append(node.step_text)
        return path

    def _check_answer(self, completion):
        return 1.0 if "正确答案" in completion else 0.0
```

### MCTS 在推理模型中的应用

MCTS + LLM 的核心优势在于可以系统地探索推理空间：

| 方法 | 搜索策略 | 优点 | 缺点 |
|------|---------|------|------|
| Greedy Decoding | 每步选概率最高的 | 速度快 | 可能错过更好的路径 |
| Beam Search | 保留 top-k 路径 | 比贪心好 | 路径之间缺乏多样性 |
| Best-of-N | 独立采样 N 个 | 简单有效 | 无法利用部分推理的信息 |
| **MCTS** | 动态树搜索 | 智能分配计算资源 | 计算成本高 |

## GRPO for Reasoning（DeepSeek-R1 方案）

### 无需 Reward Model 的 RL 训练

DeepSeek-R1 提出了一个令人兴奋的发现：**不需要复杂的 Reward Model，仅用简单的规则奖励 + GRPO 就能训练出强大的推理模型**。

训练流程：

```
基座模型 (DeepSeek-V3-Base)
    ↓
冷启动 SFT（少量高质量 CoT 数据）
    ↓
GRPO + 规则奖励（大规模 RL 训练）        ← 核心阶段
    ↓
拒绝采样 + SFT（格式规范化）
    ↓
再次 GRPO（进一步提升 + 对齐）
    ↓
DeepSeek-R1
```

### R1 多阶段训练的真实细节（论文 §3.2）

上面那张流程图是简化版。把 R1 论文 page 6–9 里真正起决定性作用的几个超参和奖励组合摊开看，就能理解为什么"多阶段"不是凑数：

**Stage-2 的奖励是三项之和**（论文 page 8，公式 8–10）：

$$
\text{Reward} = \text{Reward}_{\text{reasoning}} + \text{Reward}_{\text{general}} + \text{Reward}_{\text{language}}
$$

其中 `Reward_reasoning = Reward_rule`，`Reward_general = Reward_reward_model + Reward_format`。换句话说，第二阶段把"规则可验证的硬奖励"和"模型偏好的软奖励"在同一个 batch 里混着用，靠多种 prompt 分布来分摊 reward hacking 的风险。

**Language Consistency Reward**（page 7，公式 7），论文为了缓解 CoT 里中英混杂而引入：

$$
\text{Reward}_{\text{language}} = \frac{\text{Num}(\text{Words}_{\text{target}})}{\text{Num}(\text{Words})}
$$

也就是 CoT 中 target language 词数占比。论文坦白这个 reward 在 ablation（Supplementary B.6）里**轻微降低了推理表现**，但提升了可读性，最终还是保留了——可读性和最优精度之间不是免费午餐。

**反直觉超参**：Stage-1 RL 的 GRPO **clip ratio ε = 10**（page 8，§3.2.1），远高于 PPO/GRPO 默认的 0.2。论文解释，ε 太低会"截断大量 token 的梯度"，对 long-CoT 训练反而不稳；ε 拉到 10 之后梯度才传得动。这是 R1 训练能跑通的关键开关之一，社区复现时经常踩坑。

**异构 Reward Model**（page 7）：

| RM | 训练方式 | 数据规模 | 输入 |
|----|----------|----------|------|
| Helpful RM | Bradley-Terry pairwise | 66K 偏好对（DeepSeek-V3 当裁判 4 次取均值） | $(\text{Response}_A, \text{Response}_B)$ |
| Safety RM | Pointwise 二分类 | 106K 标注（safe/unsafe） | 单条 Response |

Query 按类别路由到不同 RM——helpfulness 走 pairwise（捕捉相对偏好），safety 走 pointwise（绝对底线）。一刀切的统一 RM 在这两个目标下会互相拖累。

**Dev1 / Dev2 / Dev3 ablation**（page 9，Table 3）是多阶段必要性的铁证：

| Benchmark | R1-Zero | R1-Dev1（Cold-Start SFT） | R1-Dev2（+ Reasoning RL） | R1-Dev3（+ 混合 SFT） | R1（+ General RL） |
|-----------|---------|---------------------------|---------------------------|------------------------|---------------------|
| AIME 2024 (Pass@1) | 77.9 | **59.0** ↓ | 74.0 | 78.1 | 79.8 |
| MATH-500 (Pass@1) | 95.9 | 94.2 | 95.9 | 95.4 | 97.3 |
| IF-Eval (Prompt Strict) | 46.6 | 71.7 | 72.0 | 78.1 | 83.3 |
| ArenaHard | 53.6 | 77.0 | 73.2 | 75.6 | 92.3 |

看 AIME 那一列：仅做 Cold-Start SFT 反而把 R1-Zero 的 77.9 砸到 **59.0**——SFT 在指令遵循上涨点（IF-Eval 46.6→71.7），但**会损伤已经被纯 RL 激活的推理能力**。Dev2 的 Reasoning RL 把推理能力救回 74.0，Dev3 的混合 SFT 维持推理同时补齐写作，最终 General RL 阶段把 ArenaHard 从 75.6 推到 92.3。任何一阶段拿掉都会留下明显窟窿。

::: warning R1 论文自己列出的局限（page 10–11，§6）
- **Tool Use 缺失**：R1 不能调用搜索引擎、计算器，结构化输出也偏弱——论文承认下版本会补
- **Token efficiency / overthinking**：简单题也会狂吐 token，"过度思考"是 long-CoT 的副作用
- **Language Mixing**：仅对中英文优化，处理其他语言时仍会切回英文/中文 reasoning
- **Few-shot 反而降分**：与传统经验相反，论文建议**用 zero-shot + 直接描述问题**取得最佳效果
- **Software Engineering RL 未规模化**：评测耗时长拖慢 RL，软件工程 benchmark 相对 V3 提升有限
:::

> 引用：DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", arXiv:2501.12948.

### 规则奖励函数设计

DeepSeek-R1 的奖励函数由两部分组成：

```python
def deepseek_r1_reward(response, ground_truth):
    """DeepSeek-R1 的规则奖励函数"""
    reward = 0.0

    # 1. 格式奖励：检查是否使用了 <think>...</think> 格式
    has_think_start = "<think>" in response
    has_think_end = "</think>" in response
    if has_think_start and has_think_end:
        think_content = response.split("<think>")[1].split("</think>")[0]
        if len(think_content.strip()) > 0:
            reward += 1.0  # 格式正确 +1

    # 2. 正确性奖励：检查最终答案是否正确
    predicted_answer = extract_boxed_answer(response)
    if predicted_answer == ground_truth:
        reward += 1.0  # 答案正确 +1

    return reward


def extract_boxed_answer(response):
    """提取 \\boxed{...} 中的答案"""
    import re
    match = re.search(r'\\boxed\{(.+?)\}', response)
    return match.group(1).strip() if match else ""
```

### "Aha Moment" 涌现

DeepSeek-R1 在训练过程中观察到了一个惊人的现象——**模型自发学会了反思和自我纠错**，被称为 "Aha Moment"：

```
问题: "计算 28 × 15"

[训练早期的输出]
<think>
28 × 15 = 28 × 10 + 28 × 5 = 280 + 140 = 410
</think>
答案是 410。    ← 错误，没有反思

[训练后期的输出 - Aha Moment]
<think>
28 × 15 = 28 × 10 + 28 × 5 = 280 + 140 = 410
等一下，让我重新验证一下。
28 × 15 = 30 × 15 - 2 × 15 = 450 - 30 = 420     ← 自发重新验证！
两种方法结果不同，我再算一次：
28 × 5 = 140，这是对的。
28 × 10 = 280，这也是对的。
280 + 140 = 420，不是 410！之前算错了。        ← 发现并纠正错误！
</think>
答案是 \boxed{420}。
```

这种自我反思能力不是通过标注数据教会的，而是在 RL 训练过程中**自然涌现**的——模型发现"检查自己的推理"能提高获得正确答案的概率，从而被奖励强化。

### GRPO 推理训练核心逻辑

```python
import torch

def grpo_reasoning_step(model, ref_model, tokenizer, question, answer,
                         group_size=8, beta=0.04, epsilon=0.2):
    """GRPO 推理训练的核心步骤（简化版，完整实现见 alignment.md）"""
    # 1. 采样 G 个回答
    responses, log_probs_old = [], []
    for _ in range(group_size):
        with torch.no_grad():
            output = model.generate(
                tokenizer(question, return_tensors='pt').input_ids,
                max_new_tokens=512, temperature=0.7, do_sample=True,
                return_dict_in_generate=True, output_scores=True)
            responses.append(tokenizer.decode(output.sequences[0]))
            log_probs_old.append(compute_log_prob(output))

    # 2. 规则奖励
    rewards = torch.tensor([deepseek_r1_reward(r, answer) for r in responses])

    # 3. 组内相对优势（全对/全错时跳过）
    if rewards.std() < 1e-6:
        return torch.tensor(0.0)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # 4. PPO-clip + KL 正则化
    total_loss = 0
    for i, resp in enumerate(responses):
        ids = tokenizer(resp, return_tensors='pt').input_ids
        log_p_new = compute_token_log_probs(model(ids).logits, ids)
        with torch.no_grad():
            log_p_ref = compute_token_log_probs(ref_model(ids).logits, ids)
        ratio = torch.exp(log_p_new - log_probs_old[i])
        clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        pg = -torch.min(ratio * advantages[i], clipped * advantages[i]).mean()
        kl = (log_p_ref.exp() / log_p_new.exp() - (log_p_ref - log_p_new) - 1).mean()
        total_loss += pg + beta * kl
    return total_loss / group_size
```

::: tip 想看"工程级最简 GRPO 复现"?
[hkust-nlp/simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)（[arXiv 2503.18892](https://arxiv.org/abs/2503.18892)）是港科大 NLP 组开源的 zero-RL 教学项目——**只用规则奖励 + 8K 数学题**，在 Llama3 / Mistral / DeepSeekMath / Qwen2.5 0.5B-32B / Qwen2.5-Math-7B 共 10 个 base 模型上都跑通 R1-Zero。代码量比 verl / open-r1 少一个数量级，是建立 GRPO 工程直觉的最佳起点。完整复现指南见 [R1 复现](/deep-dives/r1-reproduction#技术方案选择-verl-vs-slime)。
:::

### Reward 设计模式：6 个值得收藏的奖励函数

DeepSeek-R1 论文里用的 reward 看似只有"格式 + 答案"两条，但真要训出能稳定推理的模型，社区在实践中沉淀出了 **6 个常用的 reward 设计模式**——它们可以叠加使用，覆盖正确性、格式、推理深度、长度控制和重复抑制等多个目标。下面这套模式参考了 [dhcode-cpp/X-R1](https://github.com/dhcode-cpp/X-R1)（Apache 2.0，源码改编自 HuggingFace open-r1）的 `rewards.py`，按 TRL `GRPOTrainer(reward_funcs=[...])` 的调用习惯整理。

::: tip Reward 接口约定
所有 reward 函数都遵守 TRL 的统一签名：
- 输入 `completions`：`list[list[dict]]`，每个 completion 是一段对话（`[{"role": "assistant", "content": "..."}]`）
- 可选输入 `solution`：`list[str]`，对应每个样本的标准答案
- 返回值：`list[float]`，每条 completion 的奖励分数

这样多个 reward 可以直接相加：`reward_funcs=[acc, fmt, steps, length]` → GRPO 会按位相加得到组内最终 reward。
:::

#### 1. accuracy_reward — 答案对就是对

```python
from math_verify import parse, verify, LatexExtractionConfig

def accuracy_reward(completions, solution, **kwargs):
    """用 math-verify 解析 LaTeX 答案，对就 1.0，错就 0.0。"""
    rewards = []
    for completion, sol in zip(completions, solution):
        gold = parse(sol, extraction_config=[LatexExtractionConfig()])
        pred = parse(completion[0]["content"],
                     extraction_config=[LatexExtractionConfig()])
        rewards.append(float(verify(pred, gold)) if gold else 0.0)
    return rewards
```

**为什么用 math-verify 而不是字符串相等？** 数学答案有大量等价表达：`\frac{1}{2}` vs `0.5` vs `\frac12`，`x^2 + 2x + 1` vs `(x+1)^2`。`math-verify` 会把两边都解析成 SymPy 表达式做符号比较，避免"答对了但匹配失败"。

#### 2. format_reward — 强制 `<think>...</think><answer>...</answer>` 结构

```python
import re

def format_reward(completions, **kwargs):
    """检查 completion 是否严格遵循 <think>...</think><answer>...</answer> 模板。"""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return [
        1.0 if re.match(pattern, c[0]["content"], re.DOTALL) else 0.0
        for c in completions
    ]
```

格式 reward 的作用是**给模型一个稳定的"推理腔"**，让它学会"先思考再回答"的两段式输出。在 R1-Zero 训练初期，没有这条 reward 模型很容易直接输出答案而跳过推理。

#### 3. reasoning_steps_reward — 鼓励显式分步

```python
def reasoning_steps_reward(completions, **kwargs):
    """统计推理步骤数量，鼓励 ≥ 3 步的显式推理。"""
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    return [
        min(1.0, len(re.findall(pattern, c[0]["content"])) / 3)
        for c in completions
    ]
```

**为什么是 3？** 是经验数：低于 3 步往往是"跳步"或"直接给答案"；3 步及以上能保证至少有"分析 → 计算 → 验证"三个动作。`min(1.0, count/3)` 既给部分分，又设了上限——避免模型为了刷分故意把步骤拆得过碎。

#### 4. len_reward — Kimi-1.5 的长度平衡奖励

```python
def len_reward(completions, solutions, **kwargs):
    """Kimi-1.5 风格长度奖励：短的对的多奖、长的错的少惩。

    Reference: Kimi 1.5 Tech Report (arXiv:2501.12599)
    """
    contents = [c[0]["content"] for c in completions]
    correctness = [is_answer_correct(c, s) for c, s in zip(contents, solutions)]
    lengths = [len(c) for c in contents]
    min_len, max_len = min(lengths), max(lengths)
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        # 越短的 lambda 越大，正确时直接拿到，错误时只能拿到 ≤ 0 的部分
        lam = 0.5 - (length - min_len) / (max_len - min_len)
        rewards.append(float(lam if is_correct else min(0, lam)))
    return rewards
```

**核心思路：**
- 答案对的：在同一组里**越短越香**（lambda ∈ [-0.5, 0.5] 的正区间）
- 答案错的：**越长越不亏**（lambda 截到 0 以下，鼓励错的也别冗长地错）

这条 reward 是对 R1-Zero 早期"为了推理而推理"导致 completion 越来越长的直接补丁。

#### 5. cosine_scaled_reward — 长度的余弦调度

```python
import math

def get_cosine_scaled_reward(
    min_value_wrong=-1.0, max_value_wrong=-0.5,
    min_value_correct=0.5, max_value_correct=1.0,
    max_len=1000,
):
    """工厂函数：返回一个按长度做余弦插值的 reward。"""
    def cosine_scaled_reward(completions, solution, **kwargs):
        rewards = []
        for c, sol in zip(completions, solution):
            content = c[0]["content"]
            is_correct = check_answer(content, sol)
            progress = len(content) / max_len      # 长度归一化到 [0, 1]
            cosine = math.cos(progress * math.pi)  # cos: 1 → -1
            if is_correct:
                lo, hi = min_value_correct, max_value_correct
            else:
                lo, hi = max_value_wrong, min_value_wrong  # 注意这里是反的
            rewards.append(lo + 0.5 * (hi - lo) * (1.0 + cosine))
        return rewards
    return cosine_scaled_reward
```

**与 `len_reward` 的区别：**
- `len_reward` 用**组内 min/max** 做归一化，组内对比；
- `cosine_scaled_reward` 用**全局 max_len** 做归一化，绝对长度对比；同时把长度→分数的关系从线性换成余弦曲线，前半段惩罚轻、后半段陡降。

适合训练后期 reward 已经接近饱和、需要更细颗粒度地"拧螺丝"的场景。

#### 6. repetition_penalty_reward — N-gram 反复读机检测

```python
def get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0):
    """N-gram 多样性惩罚，专治 RL 训出来的"复读机"。

    Reference: Demystify Long CoT (arXiv:2502.03373) Appendix C.2
    """
    assert max_penalty <= 0, "max_penalty 必须 ≤ 0（这是惩罚不是奖励）"

    def reward(completions, **kwargs):
        rewards = []
        for c in completions:
            words = c[0]["content"].lower().split()
            if len(words) < ngram_size:
                rewards.append(0.0)
                continue
            ngrams = list(zip(*[words[i:] for i in range(ngram_size)]))
            unique_ratio = len(set(ngrams)) / len(ngrams)
            # 重复越多 → 1 - unique_ratio 越大 → 惩罚越深
            rewards.append((1 - unique_ratio) * max_penalty)
        return rewards
    return reward
```

RL 训练中很常见的失败模式是模型陷入"看似在推理实则在重复"的局部最优——同一个推理片段反复说三遍换汤不换药。这条 reward 在 token 级别用 trigram 多样性兜底，效果立竿见影。

#### 在 TRL `GRPOTrainer` 里组合使用

```python
from trl import GRPOTrainer, GRPOConfig

reward_funcs = [
    accuracy_reward,                                 # 1.0：答案正确性
    format_reward,                                   # 1.0：模板合规
    reasoning_steps_reward,                          # 1.0：分步推理
    len_reward,                                      # ±0.5：长度平衡
    get_cosine_scaled_reward(max_len=1024),          # ±1.0：精细长度调度
    get_repetition_penalty_reward(ngram_size=3,
                                   max_penalty=-1.0),# ≤0：反复读
]

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B",
    reward_funcs=reward_funcs,                       # 多 reward 自动相加
    args=GRPOConfig(num_generations=12, ...),
    train_dataset=dataset,
)
trainer.train()
```

::: tip 设计 Reward 组合的两条经验
- **正负 reward 平衡**：上面这 6 个加起来，最大可能是 `1+1+1+0.5+1+0 = 4.5`，最小可能是 `0+0+0-0.5-1-1 = -2.5`。组内 advantage 标准化后差异要足够区分"好坏 completion"，否则 GRPO 学不动。
- **从简到繁**：先只用 `accuracy + format`（R1-Zero 原始配方），跑通后再叠加 `reasoning_steps`，最后才加长度类 reward。一次性堆 6 个会让 reward 信号互相干扰，难以归因。
:::

> 实现参考：[dhcode-cpp/X-R1](https://github.com/dhcode-cpp/X-R1) `src/x_r1/rewards.py`（Apache-2.0，改编自 HuggingFace open-r1）

## Qwen3 Hybrid Thinking：思考/直答合一与思考预算

R1 走的是"专门训一个推理模型"的路线，Qwen3（arXiv:2505.09388）的设计哲学相反：**同一个模型同时具备 thinking mode 和 non-thinking mode，由用户在 chat template 里动态切换**——不再需要在"对话模型"和"推理模型"之间二选一（page 1，Abstract）。

**Chat template flag**（page 11，Table 9）。Qwen3 在 user 消息里加一个 flag 来选模式：

```text
<|im_start|>user
{query} /think<|im_end|>            ← thinking mode：会输出 <think>...</think> 推理块
<|im_start|>user
{query} /no_think<|im_end|>         ← non-thinking mode：<think></think> 留空，直接答
```

HuggingFace tokenizer 的 chat template 内置了 `enable_thinking=False` 参数等价禁用 thinking。多轮对话里可以在不同 user turn 各自标 `/think` 或 `/no_think`，模型按"最后一次出现的 flag"来切换。

**Thinking Budget**（page 11，§4.3 末尾）。用户设置一个 thinking token 上限，**当 CoT 长度达到阈值时系统注入一句 stop-thinking 指令**：

```text
"Considering the limited time by the user, I have to give the
solution based on the thinking directly now.\n</think>.\n\n"
```

模型随后基于"已经积累的不完整推理"直接给出最终答案。Qwen3 论文特别强调这个能力**不是显式训练出来的，而是 Thinking Mode Fusion 阶段自然涌现**——一旦模型同时学会了 thinking 和 non-thinking，它就具备处理"半截思考"的中间态的能力。

**后训练四阶段**（page 9，Figure 1）。旗舰模型 Qwen3-235B-A22B 与 Qwen3-32B 走完整流水线：

```
Base Model
   ↓ Stage 1：Long-CoT Cold Start         （在数学/代码/STEM 上做 long-CoT SFT 打底）
   ↓ Stage 2：Reasoning RL                （GRPO，仅 ~3995 query-verifier 对）
   ↓ Stage 3：Thinking Mode Fusion        （继续 SFT，融合 thinking + non-thinking 数据）
   ↓ Stage 4：General RL                  （20+ 任务的 reward 系统：指令、格式、Agent、RAG…）
Qwen3 旗舰
```

论文披露 Stage 2 的 RL 把 Qwen3-235B-A22B 在 AIME'24 上从 70.1 拉到 85.1，整个 RL 只跑了 170 步（page 11）。

**Strong-to-Weak Distillation**（page 9 + 12，§4.5）。剩下 6 个轻量模型（30B-A3B / 14B / 8B / 4B / 1.7B / 0.6B）**不再单独跑四阶段**，而是用旗舰模型当 teacher 做两阶段蒸馏：

1. **Off-policy Distillation**：teacher 在 `/think` 和 `/no_think` 两种模式下生成响应，student 直接学 token 序列
2. **On-policy Distillation**：student 自己采样，再用 teacher 的 logits 对齐（KL 散度），关键的"模式切换"能力靠这一步固化

论文给出的对比是：相比让每个小模型独立跑四阶段，蒸馏路径只用 **~1/10 GPU hours**，并且在 Pass@1 / Pass@64 上反而更高——Pass@64 提升说明蒸馏不仅传授了正确答案，也传授了"探索能力"（page 10）。

> 引用：Qwen Team, "Qwen3 Technical Report", arXiv:2505.09388.

## 推理蒸馏

### 用大推理模型的 CoT 教小模型

推理蒸馏的核心思路：**用大推理模型（如 DeepSeek-R1、O1）生成包含详细推理过程的数据，然后用这些数据对小模型进行 SFT**。

```
大推理模型（Teacher）                  小模型（Student）
         │                                 │
    生成高质量 CoT 推理数据 ──────→  用 CoT 数据进行 SFT
         │                                 │
  "设宽为x，长为2x，              "设宽为x，长为2x，
   周长=2(x+2x)=6x=24，            周长=2(x+2x)=6x=24，
   x=4，面积=4×8=32"               x=4，面积=4×8=32"
```

DeepSeek-R1 的实验表明：**用 R1 生成的 80 万条推理数据微调 Qwen-2.5-32B，效果接近甚至超过用 RL 直接训练**。

流程：(1) Teacher 对每个问题多次采样，选出答案正确的 CoT 推理；(2) 用这些 `(问题, <think>推理</think>答案)` 数据对 Student 做 SFT。详见 [distillation.md](distillation.md)。

### 蒸馏 vs RL 的效果对比

| 方法 | 适合场景 | 优点 | 缺点 |
|------|---------|------|------|
| **RL 训练** | 有算力、追求极致效果 | 能涌现新能力 | 训练不稳定，成本极高 |
| **推理蒸馏** | 快速部署、成本敏感 | 简单稳定，效果好 | 受限于 Teacher 的能力上限 |

> DeepSeek-R1 论文指出：蒸馏是"站在巨人的肩膀上"，而 RL 是"自己成为巨人"。对于大多数实际场景，蒸馏是更务实的选择。

## Scaling Law for Test-time Compute

OpenAI 和 DeepSeek 的实验都表明，推理时计算量与性能之间存在类似 Scaling Law 的关系：

$$
\text{Performance}(C_{\text{test}}) \approx a \cdot \log(C_{\text{test}}) + b
$$

增加 test-time compute 的方式：

| 方式 | 具体做法 | 效率 |
|------|---------|------|
| 生成更多 token | 允许模型思考更长 | 边际递减 |
| Best-of-N 采样 | 采样 N 个回答，选最好的 | $O(N)$ |
| 树搜索（MCTS） | 系统性搜索推理空间 | 最高效 |
| 集成（Majority Vote） | 多次采样取多数投票 | 简单有效 |

**关键发现**：在某些困难推理任务上，小模型 + 大量 test-time compute 可以匹敌大模型 + 少量 test-time compute。这意味着我们可以用更小的模型实现同等的推理能力，只要在推理时给予足够的计算资源。

## 代码实战：PRM 训练 + MCTS 搜索 + Best-of-N 采样

::: tip 实战目标
将前面的 PRM 理论和 MCTS 搜索算法落地为可运行的代码。包含四部分：
1. **PRM 训练** — 步级 Reward Model 的数据格式与训练流程
2. **MCTS 搜索实现** — 完整的 Selection → Expansion → Simulation → Backpropagation 循环
3. **PRM 引导的逐步搜索** — 生成一步、验证一步、拒绝重采样
4. **端到端示例** — 用 MCTS + PRM 解一道 GSM8K 数学题
:::

### Part 1：PRM 训练代码实战

PRM 的核心思路：**复用 LLM 的 next-token prediction head，在步骤分隔符位置预测 positive / negative 两个特殊 token 的概率**。这比额外加一个 reward head 更高效，因为可以直接用 LoRA 微调。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ============================================
# 1. 数据格式：PRM 训练数据长什么样
# ============================================
# 每条数据 = prompt + 若干 step，每个 step 以 SEP_TOKEN 结尾
# label 标注该 step 是 positive 还是 negative

SEP_TOKEN = "<|reserved_special_token_1|>"   # 步骤分隔符
POSITIVE_TOKEN = "<|reserved_special_token_2|>"  # 正确标签 token
NEGATIVE_TOKEN = "<|reserved_special_token_3|>"  # 错误标签 token

prm_train_example = {
    "prompt": "计算 (7+3) × (12-8) 的值",
    "steps": [
        "第一步：计算括号内，7+3=10",      # ← positive
        "第二步：计算括号内，12-8=4",       # ← positive
        "第三步：计算乘法，10×4=40",        # ← positive
    ],
    "labels": [True, True, True],  # 每步是否正确
}

prm_train_negative = {
    "prompt": "计算 (7+3) × (12-8) 的值",
    "steps": [
        "第一步：计算括号内，7+3=10",      # ← positive
        "第二步：直接算 10×12=120",         # ← negative（跳过了减法）
        "第三步：120-8=112",               # ← negative（后续全错）
    ],
    "labels": [True, False, False],
}

def format_prm_training_text(example, tokenizer):
    """将一条 PRM 数据转换为训练文本 + label 序列"""
    text = example["prompt"]
    label_ids = []  # 只在 SEP 位置有 label，其余位置 = -100（忽略）
    for step, is_correct in zip(example["steps"], example["labels"]):
        text += "\n" + step + SEP_TOKEN
        target_token = POSITIVE_TOKEN if is_correct else NEGATIVE_TOKEN
        label_ids.append(tokenizer.convert_tokens_to_ids(target_token))
    return text, label_ids

# ============================================
# 2. PRM 模型：复用 LM head，在 SEP 位置做二分类
# ============================================

class ProcessRewardTrainer:
    """PRM 训练器：在步骤分隔符位置，用 LM head 预测 positive/negative"""
    def __init__(self, model, tokenizer, lr=1e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.sep_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        self.pos_id = tokenizer.convert_tokens_to_ids(POSITIVE_TOKEN)
        self.neg_id = tokenizer.convert_tokens_to_ids(NEGATIVE_TOKEN)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_step(self, examples):
        """一个 batch 的训练步骤"""
        self.model.train()
        total_loss = 0.0
        for ex in examples:
            self.optimizer.zero_grad()
            text, step_labels = format_prm_training_text(ex, self.tokenizer)
            enc = self.tokenizer(text, return_tensors="pt")
            input_ids = enc["input_ids"]

            # 前向传播，拿到所有位置的 logits
            logits = self.model(input_ids).logits  # (1, seq_len, vocab_size)

            # 找到所有 SEP_TOKEN 的位置
            sep_positions = (input_ids[0] == self.sep_id).nonzero(as_tuple=True)[0]

            # 只取 SEP 位置前一个 token 的 logits（next-token prediction）
            # 因为 model 在位置 t 预测的是 t+1 的 token
            pred_positions = sep_positions - 1
            sep_logits = logits[0, pred_positions, :]  # (n_steps, vocab_size)

            # 只看 positive / negative 两个 token 的 logits
            binary_logits = sep_logits[:, [self.neg_id, self.pos_id]]  # (n_steps, 2)
            targets = torch.tensor(
                [1 if l == self.pos_id else 0 for l in step_labels],
                dtype=torch.long
            )

            loss = F.cross_entropy(binary_logits, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(examples)

# 使用示例
# trainer = ProcessRewardTrainer(model, tokenizer)
# for epoch in range(3):
#     loss = trainer.train_step(train_data)
#     print(f"Epoch {epoch+1}, PRM Loss: {loss:.4f}")
```

::: details PRM 训练的工程细节
- **LoRA 微调**：实践中通常用 QLoRA 4-bit 量化 + LoRA 微调 LM head 附近的层，显存占用约为全量微调的 1/4
- **数据来源**：可以用 [PRM800K](https://github.com/openai/prm800k) 人工标注数据，也可以用 Monte Carlo 自动标注（见上文 `auto_label_step` 函数）
- **SEP token 选择**：不同模型有不同的 reserved token，Llama 3 用 `<|reserved_special_token_N|>`，Qwen 可以自定义添加
:::

### Part 2：MCTS 搜索实现——完整四步循环

将前面的简化版 MCTS 扩展为**可与真实 LLM 配合**的完整实现，核心改进包括：PRM 打分替代随机 rollout、KV Cache 复用、步级粒度搜索。

```python
import math
import random
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ============================================
# MCTS 节点定义
# ============================================

@dataclass
class ReasoningNode:
    """MCTS 节点：代表推理过程中一个步骤的状态"""
    state_text: str              # 从根到当前节点的完整推理文本
    step_text: str = ""          # 当前节点对应的推理步骤文本
    parent: Optional['ReasoningNode'] = None
    children: List['ReasoningNode'] = field(default_factory=list)
    visits: int = 0              # N(s) — 访问次数
    total_reward: float = 0.0    # Q(s) — 累计奖励
    prm_score: float = 0.0       # PRM 对当前步骤的打分
    is_terminal: bool = False    # 是否到达终止状态（得出最终答案）

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def ucb_score(self, c: float = 1.414) -> float:
        """UCB1 公式：平衡探索与利用"""
        if self.visits == 0:
            return float('inf')  # 未访问的节点优先探索
        exploitation = self.total_reward / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration


# ============================================
# MCTS 搜索引擎
# ============================================

class LLMReasoningMCTS:
    """
    MCTS + LLM + PRM 的推理搜索引擎
    - Selection:  UCB 选择最有潜力的叶节点
    - Expansion:  用 LLM 生成多个候选推理步骤
    - Simulation: 用 PRM 打分（替代传统的随机 rollout）
    - Backprop:   回传奖励，更新路径上所有节点的统计量
    """
    def __init__(self, llm, prm, tokenizer,
                 n_candidates=3, max_depth=8, c_explore=1.414):
        self.llm = llm
        self.prm = prm
        self.tokenizer = tokenizer
        self.n_candidates = n_candidates  # Expansion 时每个节点生成几个子节点
        self.max_depth = max_depth
        self.c_explore = c_explore

    # ---------- 1. Selection ----------
    def _select(self, node: ReasoningNode) -> ReasoningNode:
        """从根递归选择 UCB 最高的子节点，直到叶节点"""
        while node.children and not node.is_terminal:
            node = max(node.children, key=lambda n: n.ucb_score(self.c_explore))
        return node

    # ---------- 2. Expansion ----------
    def _expand(self, node: ReasoningNode) -> ReasoningNode:
        """用 LLM 采样生成多个候选下一步，作为子节点"""
        if node.is_terminal:
            return node

        for _ in range(self.n_candidates):
            # 用 temperature sampling 生成不同的下一步推理
            next_step = self._generate_one_step(node.state_text, temperature=0.7)
            child_state = node.state_text + "\n" + next_step
            child = ReasoningNode(
                state_text=child_state,
                step_text=next_step,
                parent=node,
                is_terminal=self._is_answer_complete(next_step),
            )
            node.children.append(child)

        # 返回一个随机子节点进行 simulation
        return random.choice(node.children)

    # ---------- 3. Simulation ----------
    def _simulate(self, node: ReasoningNode) -> float:
        """
        用 PRM 给当前推理路径打分（替代传统的随机 rollout）
        如果路径未完成，先用 LLM 快速补全到终止，再整体打分
        """
        if node.is_terminal:
            return self._prm_score_path(node.state_text)

        # 快速补全：greedy decoding 到终止
        completion = self._fast_complete(node.state_text)
        full_path = node.state_text + "\n" + completion
        return self._prm_score_path(full_path)

    # ---------- 4. Backpropagation ----------
    def _backpropagate(self, node: ReasoningNode, reward: float):
        """将 reward 沿路径回传到根节点"""
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    # ---------- 主搜索循环 ----------
    def search(self, question: str, n_iterations: int = 50) -> Tuple[str, float]:
        """执行 MCTS 搜索，返回最佳推理路径和得分"""
        root = ReasoningNode(state_text=question)

        for i in range(n_iterations):
            # 1. Selection — 找到最有潜力的叶节点
            leaf = self._select(root)

            # 2. Expansion — 如果叶节点已被访问过，展开生成子节点
            if leaf.visits > 0 and not leaf.is_terminal:
                leaf = self._expand(leaf)

            # 3. Simulation — 用 PRM 评估当前路径
            reward = self._simulate(leaf)

            # 4. Backpropagation — 回传奖励
            self._backpropagate(leaf, reward)

        # 搜索结束，沿 visits 最多的路径提取最终答案
        return self._extract_best_path(root)

    # ---------- 辅助方法 ----------
    def _generate_one_step(self, context: str, temperature: float = 0.7) -> str:
        """用 LLM 生成一个推理步骤（到换行或 SEP token 为止）"""
        ids = self.tokenizer(context, return_tensors="pt").input_ids
        with torch.no_grad():
            out = self.llm.generate(
                ids, max_new_tokens=64, temperature=temperature,
                do_sample=True, eos_token_id=self.tokenizer.encode("\n")[0])
        new_tokens = out[0, ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _fast_complete(self, context: str) -> str:
        """Greedy decoding 快速补全推理到终止"""
        ids = self.tokenizer(context, return_tensors="pt").input_ids
        with torch.no_grad():
            out = self.llm.generate(ids, max_new_tokens=256, do_sample=False)
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def _prm_score_path(self, full_text: str) -> float:
        """用 PRM 对整条推理路径打分，返回 min-step-score"""
        ids = self.tokenizer(full_text, return_tensors="pt").input_ids
        sep_id = self.tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        with torch.no_grad():
            logits = self.prm(ids).logits[0]
        sep_positions = (ids[0] == sep_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) == 0:
            return 0.0
        pos_id = self.tokenizer.convert_tokens_to_ids(POSITIVE_TOKEN)
        neg_id = self.tokenizer.convert_tokens_to_ids(NEGATIVE_TOKEN)
        scores = []
        for pos in sep_positions:
            step_logits = logits[pos, [neg_id, pos_id]]
            prob_positive = torch.softmax(step_logits, dim=0)[1].item()
            scores.append(prob_positive)
        return min(scores)  # min-score 聚合

    def _is_answer_complete(self, step_text: str) -> bool:
        """判断是否已得出最终答案"""
        return "答案" in step_text or "\\boxed" in step_text or "因此" in step_text

    def _extract_best_path(self, root: ReasoningNode) -> Tuple[str, float]:
        """沿访问次数最多的路径，提取最终推理结果"""
        path_steps = []
        node = root
        while node.children:
            node = max(node.children, key=lambda n: n.visits)
            path_steps.append(node.step_text)
        score = node.avg_reward
        return "\n".join(path_steps), score
```

::: warning MCTS 搜索的计算开销
假设 `n_iterations=50`、`n_candidates=3`，每次 iteration 需要 1 次 LLM 推理（expansion）+ 1 次 LLM 补全（simulation）+ 1 次 PRM 前向传播。总计约 **100-150 次模型调用**，因此 MCTS 主要用于对准确率要求极高的场景（数学竞赛、代码生成）。对于一般问答任务，Best-of-N 更实用。
:::

### Part 3：PRM 引导的逐步验证搜索

与 MCTS 的树搜索不同，这是一种更轻量的**逐步生成 + 逐步验证**策略：每生成一步就用 PRM 打分，如果当前步骤被判定为错误，则拒绝并重新采样。这个思路来自 O1 风格的 PRM Search。

```python
def prm_guided_step_search(
    llm, prm_model, tokenizer, question,
    max_steps=10, max_retries_per_step=5, temperature=0.9
):
    """
    逐步生成 + PRM 验证的推理搜索
    - 每一步：LLM 生成候选 step → PRM 打分 → 通过则继续，否则重采样
    - 如果某一步多次重采样都不通过，则强制接受（避免死循环）
    """
    sep_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
    pos_id = tokenizer.convert_tokens_to_ids(POSITIVE_TOKEN)
    neg_id = tokenizer.convert_tokens_to_ids(NEGATIVE_TOKEN)

    current_ids = tokenizer(question, return_tensors="pt").input_ids
    accepted_steps = []

    for step_i in range(max_steps):
        # 检查是否已生成 EOS
        if current_ids[0, -1] == tokenizer.eos_token_id:
            break

        best_step, best_prob = None, -1.0
        for retry in range(max_retries_per_step):
            # 1. 生成一个推理步骤（do_sample 保证多样性）
            with torch.no_grad():
                step_out = llm.generate(
                    current_ids, max_new_tokens=64,
                    temperature=temperature, do_sample=True,
                    eos_token_id=sep_id)
            step_ids = step_out[:, current_ids.shape[1]:]

            # 2. 拼接后用 PRM 验证
            candidate_ids = torch.cat([current_ids, step_ids], dim=1)
            with torch.no_grad():
                logits = prm_model(candidate_ids).logits[0]

            # 在最后一个 SEP 位置取 positive 概率
            sep_positions = (candidate_ids[0] == sep_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 0:
                last_sep = sep_positions[-1]
                step_logits = logits[last_sep, [neg_id, pos_id]]
                p_positive = torch.softmax(step_logits, dim=0)[1].item()
            else:
                p_positive = 0.5

            # 记录最好的候选
            if p_positive > best_prob:
                best_prob = p_positive
                best_step = step_ids

            # 3. 如果 PRM 判定为正确，接受该步骤
            if p_positive > 0.5:
                break

        # 接受最好的步骤（即使没通过阈值也选概率最高的）
        current_ids = torch.cat([current_ids, best_step], dim=1)
        step_text = tokenizer.decode(best_step[0], skip_special_tokens=False)
        accepted_steps.append({
            "step": step_text.strip(),
            "prm_positive_prob": best_prob,
            "retries": retry + 1
        })
        print(f"Step {step_i+1}: P(correct)={best_prob:.3f}, "
              f"retries={retry+1}, text={step_text.strip()[:60]}")

    full_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
    return full_text, accepted_steps
```

::: details PRM Search vs MCTS vs Best-of-N 对比
| 维度 | Best-of-N | PRM Step Search | MCTS |
|------|-----------|-----------------|------|
| 搜索策略 | 独立采样 N 条完整路径 | 逐步生成+验证 | 树搜索+回溯 |
| LLM 调用次数 | N 次 | max_steps × avg_retries | iterations × 2 |
| 能否利用部分正确推理 | 不能 | 能（逐步保留） | 能（树结构复用） |
| 实现复杂度 | 低 | 中 | 高 |
| 推荐场景 | N ≤ 64，通用任务 | 步骤可验证的数学/逻辑题 | 竞赛级别难题 |
:::

### Part 4：端到端示例——用 MCTS + PRM 解 GSM8K 数学题

将上面三个组件串联起来，展示一个完整的推理搜索流程：

```python
import torch
import re

# ============================================
# GSM8K 端到端示例
# ============================================

def solve_gsm8k_with_mcts(question, ground_truth_answer,
                           llm, prm_model, tokenizer):
    """
    用 MCTS + PRM 解一道 GSM8K 数学题
    1. MCTS 搜索最佳推理路径
    2. Best-of-N 做 baseline 对比
    3. PRM 逐步搜索做对比
    """
    print("=" * 60)
    print(f"问题: {question}")
    print(f"标准答案: {ground_truth_answer}")
    print("=" * 60)

    # --- 方法 1：直接 greedy decoding ---
    ids = tokenizer(question, return_tensors="pt").input_ids
    with torch.no_grad():
        greedy_out = llm.generate(ids, max_new_tokens=256, do_sample=False)
    greedy_answer = tokenizer.decode(greedy_out[0], skip_special_tokens=True)
    greedy_result = extract_number(greedy_answer)
    print(f"\n[Greedy] 答案: {greedy_result}")

    # --- 方法 2：Best-of-N + PRM 打分 ---
    best_answer, best_score = best_of_n_with_prm(
        llm, prm_model, tokenizer, question, n=8, temperature=0.7)
    bon_result = extract_number(best_answer)
    print(f"[Best-of-8] 答案: {bon_result}, PRM score: {best_score:.3f}")

    # --- 方法 3：PRM 逐步搜索 ---
    step_answer, steps = prm_guided_step_search(
        llm, prm_model, tokenizer, question,
        max_steps=6, max_retries_per_step=3)
    step_result = extract_number(step_answer)
    print(f"[PRM Search] 答案: {step_result}")

    # --- 方法 4：MCTS 搜索 ---
    mcts = LLMReasoningMCTS(llm, prm_model, tokenizer,
                             n_candidates=3, max_depth=6)
    mcts_path, mcts_score = mcts.search(question, n_iterations=30)
    mcts_result = extract_number(mcts_path)
    print(f"[MCTS] 答案: {mcts_result}, score: {mcts_score:.3f}")
    print(f"  推理路径:\n  {mcts_path.replace(chr(10), chr(10) + '  ')}")

    # --- 对比结果 ---
    print("\n" + "-" * 40)
    for name, result in [("Greedy", greedy_result), ("Best-of-8", bon_result),
                          ("PRM Search", step_result), ("MCTS", mcts_result)]:
        correct = "correct" if str(result) == str(ground_truth_answer) else "wrong"
        print(f"  {name:12s}: {result:>8s}  [{correct}]")


def best_of_n_with_prm(llm, prm_model, tokenizer, question,
                        n=8, temperature=0.7):
    """生成 N 条完整推理，用 PRM min-score 选最佳"""
    sep_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
    pos_id = tokenizer.convert_tokens_to_ids(POSITIVE_TOKEN)
    neg_id = tokenizer.convert_tokens_to_ids(NEGATIVE_TOKEN)
    candidates, scores = [], []

    for _ in range(n):
        ids = tokenizer(question, return_tensors="pt").input_ids
        with torch.no_grad():
            out = llm.generate(ids, max_new_tokens=256,
                               temperature=temperature, do_sample=True)
        candidates.append(tokenizer.decode(out[0], skip_special_tokens=True))

    for cand in candidates:
        cand_ids = tokenizer(cand, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = prm_model(cand_ids).logits[0]
        sep_positions = (cand_ids[0] == sep_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) == 0:
            scores.append(0.0)
            continue
        step_scores = []
        for pos in sep_positions:
            sl = logits[pos, [neg_id, pos_id]]
            step_scores.append(torch.softmax(sl, dim=0)[1].item())
        scores.append(min(step_scores))  # min-score 聚合

    best_idx = max(range(n), key=lambda i: scores[i])
    return candidates[best_idx], scores[best_idx]


def extract_number(text):
    """从推理文本中提取最终数值答案"""
    # 优先从 \\boxed{} 中提取
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    # 否则取最后出现的数字
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    return numbers[-1] if numbers else ""


# ============================================
# 运行示例（需要加载实际模型和 PRM）
# ============================================

# gsm8k_question = (
#     "Janet's ducks lay 16 eggs per day. She eats three for breakfast "
#     "every morning and bakes muffins for her friends every day with four. "
#     "She sells the remainder at the farmers' market daily for $2 per "
#     "fresh duck egg. How much in dollars does she make every day at "
#     "the farmers' market?"
# )
# solve_gsm8k_with_mcts(gsm8k_question, "18", llm, prm_model, tokenizer)
```

::: tip 预期输出效果
在 GSM8K 基准测试上，不同搜索策略的典型效果（以 Llama-3-8B 为例）：
- **Greedy**: ~50% 准确率
- **Best-of-8**: ~62% 准确率（+12%）
- **PRM Step Search**: ~66% 准确率（+16%）
- **MCTS (30 iterations)**: ~70% 准确率（+20%）

随着搜索预算增加（更多 iterations / 更大 N），准确率持续提升，这就是 **Test-time Compute Scaling** 的直观体现。
:::

## 苏格拉底时刻

1. **为什么 PRM 比 ORM 更有效？** ORM 只提供"对/错"的稀疏信号，而 PRM 在每一步都给出反馈。这类似于老师批改数学作业——只看最终答案对错（ORM）远不如逐步批改（PRM）有指导价值。
2. **MCTS 的 exploration-exploitation tradeoff 如何理解？** UCB 公式中，第一项是利用（选择历史奖励高的路径），第二项是探索（选择访问少的路径）。$c$ 值越大越倾向探索。推理任务中，适度探索能避免陷入"看似正确实则错误"的推理路径。
3. **DeepSeek-R1 的 "Aha Moment" 为什么能涌现？** 因为自我纠错能提高最终答案的正确率，而正确率直接对应奖励。RL 的优化压力使模型自然学会了这种策略。这说明足够简单的奖励信号也能催生复杂的行为。
4. **推理蒸馏的上限在哪里？** 蒸馏模型最多只能达到 Teacher 的水平，因为它学习的是 Teacher 的推理模式。RL 训练则有可能超越 Teacher（因为 RL 是自我探索）。但在实践中，蒸馏的效率优势往往使其成为更好的选择。
5. **小模型 + 大量 test-time compute 能否替代大模型？** 理论上可以在某些任务上实现，但前提是任务有明确的验证信号（如数学题可以验算）。对于开放式生成任务（创意写作、综合分析），test-time compute scaling 的收益有限。

## 常见问题 & 面试考点

- **Q: O1 和 R1 的核心区别是什么？** O1 是闭源产品，具体技术未公开。R1 公开了技术细节：使用 GRPO + 规则奖励进行 RL 训练，并通过推理蒸馏扩展到不同规模的模型。
- **Q: PRM 训练数据如何获取？** 三种方式：(1) 人工标注（成本高但质量好，如 PRM800K）；(2) Monte Carlo 自动标注（用采样估计每步正确率）；(3) 模型自动生成标注（成本最低但质量不稳定）。
- **Q: Best-of-N 和 MCTS 如何选择？** Best-of-N 简单高效，适合 N 较小的场景（8~64）。MCTS 更智能但计算开销大，适合需要深入推理的难题。实践中建议先用 Best-of-N，不够再上 MCTS。
- **Q: GRPO 在推理任务上为什么比 PPO 更受欢迎？** (1) 不需要训练额外的 Critic 网络，节省一半显存；(2) 规则奖励消除了 Reward Model 的噪声；(3) 组内相对优势在推理任务上信号更清晰。
- **Q: Reasoning Token 增加了推理成本，如何平衡？** 可以通过训练"think budget"——让模型学习根据问题难度决定思考的深度。简单问题少思考，难题多思考，实现计算资源的动态分配。

## 推荐资源

### 论文与博客

- [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/) - OpenAI O1 技术博客
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2501.12948) - DeepSeek-R1 论文
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) - PRM / PRM800K 论文
- [Scaling LLM Test-Time Compute](https://arxiv.org/abs/2408.03314) - Test-time compute scaling 研究
- [Math-Shepherd: Verify and Reinforce LLMs Step-by-step](https://arxiv.org/abs/2312.08935) - 自动化 PRM 标注
- [Thinking, Fast and Slow](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) - Kahneman 的双系统理论
- [Monte Carlo Tree Search in AlphaGo](https://www.nature.com/articles/nature16961) - MCTS 的经典应用

### 代码参考

- [HuggingFace open-r1](https://github.com/huggingface/open-r1) — 官方公开的 DeepSeek-R1 三阶段复现工程：Step 1（蒸馏）已通过 `src/open_r1/sft.py` + `recipes/OpenR1-Distill-7B/sft/config_distill.yaml` 在 [Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) (350k R1 traces) 上完成，复刻出对标 DeepSeek-R1-Distill-Qwen-7B 的 [OpenR1-Distill-7B](https://huggingface.co/open-r1/OpenR1-Distill-7B)；Step 2 的 R1-Zero 风格纯 RL 路径走 `src/open_r1/grpo.py` + `recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml`，底层基于 TRL `GRPOTrainer` + vLLM (colocate / server 双模式) + accelerate ZeRO-3。`src/open_r1/rewards.py` 把推理任务的 reward 设计模式（accuracy / format / cosine-length / repetition-penalty）写得很完整，可以直接照抄进自己的 GRPO 训练；`scripts/generate_reasoning.py` 演示了如何用 [Distilabel](https://github.com/argilla-io/distilabel) 从 R1 蒸馏推理数据，是想把"R1-Zero / SFT distillation / 评测"链路打通时最值得对照的一手实现
