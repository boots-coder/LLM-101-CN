---
title: "偏好对齐"
description: "RLHF 三阶段、Reward Model、PPO、DPO、GRPO、KTO、ORPO、CPO、RLOO、PRM、TRL 实战"
topics: [RLHF, reward-model, PPO, DPO, GRPO, KTO, ORPO, CPO, SimPO, RLOO, Online-DPO, PRM, TRL, Bradley-Terry, preference-alignment]
prereqs: [training/sft]
---
# 偏好对齐

> 偏好对齐让模型的输出符合人类期望，是 LLM 安全和有用的关键

## 在大模型体系中的位置

```
预训练 (Pre-training)          → 学习语言知识和世界知识
    ↓
监督微调 (SFT)                 → 学习指令跟随能力
    ↓
偏好对齐  ← 你在这里            → 学习人类偏好，安全有用
  ├── RLHF (PPO)               → 经典方案：奖励模型 + 强化学习
  ├── DPO                      → 无需奖励模型，直接偏好优化
  ├── GRPO                     → DeepSeek 方案：组内相对排名
  └── KTO                      → 无需成对数据，前景理论启发
```

SFT 后的模型能遵循指令，但可能生成有害、不准确或低质量的回答。偏好对齐通过人类反馈引导模型学习"什么样的回答更好"，使模型既 **有用 (helpful)** 又 **安全 (harmless)**。

## RLHF 概述

RLHF (Reinforcement Learning from Human Feedback) 是 ChatGPT 成功的关键技术，完整 pipeline 包括三个阶段：

```
阶段一：SFT 训练       → 得到 SFT 模型 (π_sft)
    ↓
阶段二：训练 Reward Model → 学习人类偏好打分
    ↓
阶段三：PPO 优化        → 用 RL 最大化奖励，同时不偏离 SFT 模型太远
```

### Bradley-Terry 偏好模型

RLHF 的数学基础是 **Bradley-Terry 模型**——给定提示 $x$，人类偏好回答 $y_w$（chosen）胜过 $y_l$（rejected）的概率为：

$$
P(y_w \succ y_l | x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))
$$

其中 $\sigma$ 是 sigmoid 函数，$r_\theta$ 是奖励模型。训练目标是最大化这个概率：

$$
\mathcal{L}_{\text{RM}} = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))
$$

## Reward Model 训练

### 模型架构

奖励模型的架构与预训练模型相同，但将 next-token prediction 的分类头替换为**输出标量奖励的回归头**：

```python
from transformers import LlamaForCausalLM, LlamaForSequenceClassification, LlamaConfig

config = LlamaConfig(
    vocab_size=100, hidden_size=256,
    intermediate_size=512, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=4,
)

# 从预训练模型初始化奖励模型
model = LlamaForCausalLM(config)
model.save_pretrained('./lm_pretrained')

# 关键：num_labels=1，输出标量奖励
rm_model = LlamaForSequenceClassification.from_pretrained(
    './lm_pretrained', num_labels=1
)
# 原始的 lm_head 被替换为: score = Linear(hidden_size, 1)
```

### 训练（含 Margin Loss）

LLaMA 2 使用了带 margin 的损失函数，margin 由偏好评级决定：

$$
\mathcal{L} = -\log \sigma(r_\theta(x, y_c) - r_\theta(x, y_r) - m(r))
$$

```python
# 奖励模型训练：让 chosen 的得分高于 rejected
X_chosen = torch.randint(0, 100, (1, 10))
X_rejected = torch.randint(0, 100, (1, 10))
margin = 3.0  # Margin 越大，表示 chosen 明显优于 rejected

rm_chosen = rm_model(input_ids=X_chosen).logits     # chosen 的奖励分数
rm_rejected = rm_model(input_ids=X_rejected).logits  # rejected 的奖励分数

# 标准 loss
loss = -torch.sigmoid(rm_chosen - rm_rejected).log()
# 带 margin 的 loss（LLaMA 2 方案）
loss_with_margin = -torch.sigmoid(rm_chosen - rm_rejected - margin).log()
```

::: tip Reward Model 的实证规律
- **用排序而不是直接打分**：人类直接打分会因不同标注员尺度不一污染数据，所以 RLHF 论文都用 pairwise 排序，再用 Bradley-Terry / Elo 把排序转成标量奖励。
- **6B-10B 是 RM sweet spot**：HuggingFace illustrated RLHF 综合 OpenAI / Anthropic / DeepMind 的发现指出 RM 在 6B-10B 范围效果最佳；过大反而加剧 reward hacking——policy 学会钻 RM 漏洞而非满足真实偏好。
- **数据量参考**：OpenAI 50k、Anthropic [hh-rlhf](https://github.com/anthropics/hh-rlhf) 169k pairs，是当前主流开源 preference 数据集；后续 PPO 阶段必须配 KL 项约束策略不偏离 SFT，原因正是"RM 不是真函数，是从排序数据拟合出的近似"，policy 一脱离 RM 训练分布就开始胡来。
:::

### LLaMA 2 的双奖励选择

LLaMA 2 同时训练**安全性奖励模型**与**有用性奖励模型**，在 RLHF 阶段按 prompt 性质与安全分数动态选择哪一路作为最终奖励：

$$
R_c(g | p) = \begin{cases} R_s(g|p) & \text{if is\_safety}(p) \text{ or } R_s(g|p) < 0.15 \\ R_h(g|p) & \text{otherwise} \end{cases}
$$

下面给出该公式的教学版 Python 实现。要点有三：

1. **三条分支显式分离**——`is_safety_prompt` / `safety_score` 低于阈值 / 否则走 `helpfulness_score`，避免单行 ternary 把 `is_safety(p)` 那一支吞掉。
2. **阈值参数化**——论文给的 0.15 只是经验值，开放给读者调；
3. **docstring 带可执行 assert**——直接看到三种路径的输出。

```python
def gated_dual_reward(
    safety_score: float,
    helpfulness_score: float,
    is_safety_prompt: bool = False,
    safety_threshold: float = 0.15,
) -> float:
    """按 Llama 2 论文 Section 3.2.2 组合安全/有用性两路奖励。

    对应公式 R_c(g|p)：
      - 当 prompt 被分类为安全敏感（is_safety_prompt=True）→ 取安全奖励
      - 或安全奖励低于阈值（默认 0.15，来自论文经验值）→ 同样取安全奖励
      - 其余情况下使用有用性奖励

    参考：Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models",
          arXiv:2307.09288, Section 3.2.2 / 3.2.4。

    >>> gated_dual_reward(0.05, 0.9)            # 安全分数过低 → 走安全
    0.05
    >>> gated_dual_reward(0.8, 0.9, is_safety_prompt=True)  # 显式安全 prompt
    0.8
    >>> gated_dual_reward(0.8, 0.9)             # 普通 prompt 且安全足够 → 走有用性
    0.9
    """
    if is_safety_prompt:
        return safety_score
    elif safety_score < safety_threshold:
        return safety_score
    else:
        return helpfulness_score


# 简单自检
assert gated_dual_reward(0.05, 0.9) == 0.05
assert gated_dual_reward(0.8, 0.9, is_safety_prompt=True) == 0.8
assert gated_dual_reward(0.8, 0.9) == 0.9
```

::: tip 工程级实现对照
本节的 `gated_dual_reward` 只是公式的逐行还原。真正在 PPO 训练中如何把奖励标量"注入"到逐 token 的 advantage 计算里，可参考仓库已收录的两个开源工程：

- **DeepSpeed-Chat** 的 reward 处理：[microsoft/DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/dschat/rlhf) 中的 `compute_rewards`，展示了 KL 惩罚 + 末位 token 注入 RM 分数的标准做法。
- **trlx** 的 reward filtering：[CarperAI/trlx](https://github.com/CarperAI/trlx)，对应 PPO trainer 中 reward 的归一化与裁剪流程。
:::

::: details 参考文献
- Touvron et al., 2023, **Llama 2: Open Foundation and Fine-Tuned Chat Models**, [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)。
  - Section 3.2.2 "Helpfulness and Safety Reward Models"——双奖励模型的训练目标与解耦动机。
  - Section 3.2.4 提到的 **Margin Loss**（与本页前一节"训练（含 Margin Loss）"对应）以及把两路 RM 组合进 PPO 的工程细节。
- DeepSpeed-Chat RLHF 模块（`compute_rewards`）：[microsoft/DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/dschat/rlhf)。
- trlx PPO Trainer reward 流水线：[CarperAI/trlx](https://github.com/CarperAI/trlx)。
:::

### 奖励后处理：Whiten + KL 惩罚

```python
# 1. 逆 Sigmoid（Logit 函数）：稳定数值
def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

# 2. Whiten：标准化奖励分布
def whiten(values, shift_mean=True):
    mean, var = torch.mean(values), torch.var(values)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    return whitened

# 3. KL 惩罚：防止模型偏离 SFT 策略太远
# R(g|p) = R_hat(g|p) - beta * D_KL(pi_theta || pi_0)
beta = 0.01
kl = logprobs_new - logprobs_ref
reward_final = rm_score - beta * kl
```

## PPO (Proximal Policy Optimization)

PPO 是 RLHF 中最常用的强化学习算法。它通过 **clip 机制** 限制策略更新幅度，确保训练稳定。

### RLHF-PPO 需要四个模型

PPO 阶段同时维护四个模型：

| 模型 | 是否更新 | 作用 |
|------|---------|------|
| Policy（actor） | ✅ 训练 | 被优化的 LLM，输出 token 概率 |
| Reference | ❌ 冻结 | SFT 后的副本，用于 KL 约束 |
| Reward Model | ❌ 冻结 | 给完整 response 打标量分 |
| Value Model（critic） | ✅ 训练 | 逐 token 估计未来回报 |

实践中 Policy 和 Value 通常共享 backbone，再分别接一个语言建模头和一个标量回归头——这就是 TRL 的 `AutoModelForCausalLMWithValueHead` 在做的事情。

```python
import torch
from torch import nn

class LMWithValueHead(nn.Module):
    """语言模型 + 价值头：对齐 TRL 的 AutoModelForCausalLMWithValueHead 设计"""

    def __init__(self, base_model: nn.Module, hidden_dim: int):
        super().__init__()
        self.base = base_model
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids, attention_mask=None):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        last_hidden = out.hidden_states[-1]                  # [B, T, H]
        values = self.value_head(last_hidden).squeeze(-1)    # [B, T]
        return out.logits, values
```

### PPO 训练流程

```python
from dataclasses import dataclass

@dataclass
class PPOHParams:
    """与 TRL PPOConfig 同名的核心字段，便于将来无缝切换到 trl.PPOConfig"""
    ppo_epochs: int = 4         # 同一批样本上的 PPO 优化轮数
    mini_batch_size: int = 1
    learning_rate: float = 1e-5
    gamma: float = 1.0          # 语言任务通常不打折
    lam: float = 0.95           # GAE λ
    clip_range: float = 0.2     # 策略 ratio 的 clip 阈值
    clip_range_value: float = 0.2  # value 的 clip 阈值
    kl_coef: float = 0.05       # 每 token KL 惩罚系数
    vf_coef: float = 0.1        # value loss 在总 loss 中的权重


def run_ppo(policy, ref_policy, reward_model, prompts, hparams: PPOHParams):
    """PPO 主循环（教学版骨架）"""
    optimizer = torch.optim.AdamW(policy.parameters(), lr=hparams.learning_rate)

    for prompt_batch in prompts:
        # 1. roll out：当前策略生成 response
        responses, old_logprobs, ref_logprobs, values = rollout(
            policy, ref_policy, prompt_batch
        )

        # 2. 打分：reward model 给每条完整 response 一个标量
        scores = reward_model(prompt_batch, responses)

        # 3. 多轮小步更新
        for _ in range(hparams.ppo_epochs):
            loss = ppo_update_step(
                policy, prompt_batch, responses,
                old_logprobs, ref_logprobs, values, scores, hparams,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### 把 KL 惩罚揉进 token 级 reward

PPO 把"奖励信号"和"KL 约束"统一到一条 token 级序列里：每个生成 token 减去 KL 项，最后一个 token 再加上 RM 给的标量分。

```python
def shape_token_rewards(rm_score, logprobs, ref_logprobs, response_mask, kl_coef):
    """
    Args:
        rm_score:       [B]      reward model 对整条 response 的打分
        logprobs:       [B, T]   policy 当前 token log-probs
        ref_logprobs:   [B, T]   reference model 的 token log-probs
        response_mask:  [B, T]   1 表示 response 部分
        kl_coef:        float
    Returns:
        token_rewards:  [B, T]   作为 PPO 优势估计的输入
    """
    kl = logprobs - ref_logprobs                        # 每 token 的 KL
    token_rewards = -kl_coef * kl                       # 全程的 KL 惩罚
    last_idx = response_mask.sum(dim=-1).long() - 1     # 每条样本最后一个 response token
    token_rewards[torch.arange(rm_score.size(0)), last_idx] += rm_score
    return token_rewards * response_mask
```

::: tip 为什么 RM 分数只加在最后一个 token？
RM 是 sequence-level 评分，没有 token 级监督。把它放在最后一个 token 上，让 GAE 反向传播时把这份信用顺着 critic 的估计自然分摊到前面的 token——这是 InstructGPT / TRL 沿用至今的做法。
:::

### GAE 优势估计

广义优势估计 (Generalized Advantage Estimation, GAE) 用 TD 残差的指数加权平均同时控制偏差和方差。完整推导见 [Schulman et al. 2015](https://arxiv.org/abs/1506.02438)。

```python
def compute_gae(token_rewards, values, response_mask, gamma, lam):
    """
    GAE 反向递推：A_t = δ_t + γλ A_{t+1}

    Args:
        token_rewards: [B, T]
        values:        [B, T]   critic 对每个 token 的估计
        response_mask: [B, T]
        gamma, lam:    float
    Returns:
        advantages, returns: [B, T]
    """
    B, T = token_rewards.shape
    advantages = torch.zeros_like(token_rewards)
    gae = torch.zeros(B, device=token_rewards.device)

    # 倒序：T-1 -> 0
    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t + 1 < T else torch.zeros(B)
        delta = token_rewards[:, t] + gamma * next_value - values[:, t]
        gae = delta + gamma * lam * gae
        advantages[:, t] = gae

    advantages = advantages * response_mask
    returns = advantages + values                      # critic 的回归目标
    return advantages.detach(), returns.detach()
```

### PPO 的两个 loss

PPO 的总目标 = 策略目标（clip 形式的 surrogate）+ value 损失（clipped MSE）。下面的实现严格对照 TRL 的 [`PPOTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py) 的命名习惯。

```python
def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1)


def policy_surrogate_loss(new_logprobs, old_logprobs, advantages, mask, clip_range):
    """PPO-clip：max(unclipped, clipped) 取悲观估计"""
    log_ratio = new_logprobs - old_logprobs
    ratio = log_ratio.exp()

    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

    return masked_mean(torch.maximum(unclipped, clipped), mask)


def value_clipped_loss(new_values, old_values, returns, mask, clip_range_value):
    """value 也要 clip，避免一次性把 critic 拉飞"""
    v_clipped = old_values + torch.clamp(
        new_values - old_values, -clip_range_value, clip_range_value,
    )
    loss_unclipped = (new_values - returns) ** 2
    loss_clipped = (v_clipped - returns) ** 2
    return 0.5 * masked_mean(torch.maximum(loss_unclipped, loss_clipped), mask)


def ppo_objective(new_logprobs, old_logprobs, new_values, old_values,
                  advantages, returns, response_mask, hparams: PPOHParams):
    pg_loss = policy_surrogate_loss(
        new_logprobs, old_logprobs, advantages, response_mask, hparams.clip_range,
    )
    vf_loss = value_clipped_loss(
        new_values, old_values, returns, response_mask, hparams.clip_range_value,
    )
    return pg_loss + hparams.vf_coef * vf_loss, pg_loss, vf_loss
```

> 实现参考：[HuggingFace TRL · PPOTrainer](https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py)（Apache-2.0）。本节代码做了教学化简化——去掉了 accelerator / dataclass config / logging 等工程细节，只保留算法骨架。

### 最小可跑 PPO 一步：演示 ratio → clip → GAE → loss 计算链路

::: tip 这一节解决什么问题
PPO 完整 rollout 涉及"policy + ref + RM + critic 四个模型 + 多轮 inner update + accelerate 启动"，对纸笔推导极不友好。下面这段 ~50 行代码用**纯合成数据**走完一次 PPO 计算链——不依赖任何模型 generate，几秒跑完。看清四件事：
1. `ratio = exp(new - old)` 在 token 维度怎么算
2. GAE 怎么倒序累加
3. `clipped surrogate` 为什么取 `max(unclipped, clipped)` 而不是 `min`
4. value loss 也要 clip
:::

#### Step 1：构造合成 rollout 张量（替代真 generate）

```python
import torch
torch.manual_seed(42)

B, T = 2, 8                                          # 2 条样本，每条 8 个 response token
old_logprobs = torch.randn(B, T) * 0.1               # 假装 old policy 给的 log prob
new_logprobs = old_logprobs + torch.randn(B, T) * 0.1  # 假装 inner update 后 logprob 漂了一点
mask = torch.ones(B, T)
values = torch.randn(B, T) * 0.5                     # critic 的旧估计
rewards = torch.zeros(B, T)
rewards[:, -1] = torch.tensor([1.5, -0.5])           # RM 标量分只加在最后一位 (见 §把 KL 惩罚揉进 token 级 reward)
```

#### Step 2：GAE 倒序递推

```python
def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.size(0))
    for t in reversed(range(rewards.size(1))):
        next_v = values[:, t + 1] if t + 1 < rewards.size(1) else torch.zeros_like(values[:, 0])
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        gae = delta + gamma * lam * gae
        advantages[:, t] = gae
    return advantages * mask, advantages * mask + values

advantages, returns = compute_gae(rewards, values, mask)
print(f"advantages last 3 cols:\n{advantages[:, -3:]}")
print(f"returns    last 3 cols:\n{returns[:, -3:]}")
```

注意倒序的关键：当前 token 的 advantage 依赖于**未来**所有 token 的 TD 残差，所以必须从 T-1 反向走。

#### Step 3：Clipped surrogate（policy loss）

`max(unclipped, clipped)` 取的是**悲观估计**——选两者中"损失更大"（对应 `-adv·ratio` 更大）的一项作为 loss，等价于在策略改进上"宁可保守"。

```python
def policy_loss(new_lp, old_lp, adv, mask, clip=0.2):
    ratio = (new_lp - old_lp).exp()
    unclipped = -adv * ratio
    clipped   = -adv * torch.clamp(ratio, 1 - clip, 1 + clip)
    return ((torch.maximum(unclipped, clipped) * mask).sum() / mask.sum())

ratio = (new_logprobs - old_logprobs).exp()
print(f"ratio range: [{ratio.min():.3f}, {ratio.max():.3f}]")
pg = policy_loss(new_logprobs, old_logprobs, advantages, mask)
print(f"policy loss = {pg.item():+.4f}")
```

#### Step 4：Value clipped loss + 总目标

value 也要 clip——避免一次 inner update 把 critic 拉飞，这是 PPO 论文里常被忽略的细节。

```python
def value_loss(new_v, old_v, returns, mask, clip=0.2):
    v_clipped = old_v + torch.clamp(new_v - old_v, -clip, clip)
    l_un = (new_v - returns) ** 2
    l_cl = (v_clipped - returns) ** 2
    return 0.5 * ((torch.maximum(l_un, l_cl) * mask).sum() / mask.sum())

new_values = values + torch.randn_like(values) * 0.1   # 假装 critic 也更新了一点
vf = value_loss(new_values, values, returns, mask)
total = pg + 0.1 * vf
print(f"value  loss = {vf.item():+.4f}")
print(f"total  loss = {total.item():+.4f}  (= pg + 0.1 * vf)")
```

#### Step 5：验证 clip 在 ratio 越界时如何"踢入"

```python
# 故意构造一组 ratio 越界的样本
old_lp  = torch.zeros(1, 4)
new_lp  = torch.tensor([[0.0, 0.5, -0.5, 1.0]])   # ratio 将分别是 [1.00, 1.65, 0.61, 2.72]
adv     = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
mask_e  = torch.ones(1, 4)
ratio_e = (new_lp - old_lp).exp()
unclip  = -adv * ratio_e
clip    = -adv * torch.clamp(ratio_e, 0.8, 1.2)
print(f"ratio:     {ratio_e[0].tolist()}")
print(f"unclipped: {unclip[0].tolist()}")
print(f"clipped:   {clip[0].tolist()}")
print(f"max (=loss term): {torch.maximum(unclip, clip)[0].tolist()}")
# 第 4 列 ratio=2.72 越界 → max 选 -1.20 而不是 -2.72，等价于"看到大幅 ratio 也只用 1.20 信任更新"
```

::: warning 这只是计算链路演示
没有真 rollout、没有 RM 打分、没有 backward——目的是把 PPO 的"4 个 loss 张量"在数值上跑一遍让你看见。**真完整的 PPO** 见上方 §PPO 训练流程 的 `run_ppo` 骨架，再到 [TRL · `PPOTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py) / [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) 的工业实现。
:::

### 代码参考：DS-Chat 与 trlx

把 PPO 数学公式映射到真实工程代码，最值得对照阅读的有两个开源实现：DeepSpeed-Chat 把 RLHF 三阶段（SFT → RM → PPO）做成了端到端最简模板；trlx 则在注释里直接给出 PPO 损失到原始论文的逐项出处，是把"公式 → 代码"翻译得最透的实现之一。

**DeepSpeed-Chat（[arXiv 2308.01320](https://arxiv.org/abs/2308.01320)）**——`actor_loss_fn` 与 GAE 倒序累加都极其紧凑：

```python
# dschat/rlhf/ppo_trainer.py L282-L291
def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
    ## policy gradient loss
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                         1.0 + self.cliprange)
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
    return pg_loss

# L308-L320  GAE 倒序累加：lastgaelam = δ + γ·λ·lastgaelam
def get_advantages_and_returns(self, values, rewards, start):
    lastgaelam = 0
    advantages_reversed = []
    length = rewards.size()[-1]
    for t in reversed(range(start, length)):
        nextvalues = values[:, t + 1] if t < length - 1 else 0.0
        delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
        lastgaelam = delta + self.gamma * self.lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values[:, start:]
    return advantages.detach(), returns
```

> 永久链接：[`actor_loss_fn` L282-L291](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py#L282-L291) · [`get_advantages_and_returns` L308-L320](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py#L308-L320)。同文件 `compute_rewards` (L181) 把 reward score 与 KL penalty 注入 token 级奖励——回应了上一节"把 KL 揉进 reward"的工程做法。

**trlx（[CarperAI/trlx](https://github.com/CarperAI/trlx)，archived 但教学价值仍高）**——`AdaptiveKLController` 类注释里直接写出对应论文与代码出处：

```python
# trlx/models/modeling_ppo.py 头部
class AdaptiveKLController:
    """Adaptive KL Controller as described in Ziegler et al. "Fine-Tuning Language Models from Human Preferences"
    Reference: Section 2.2 https://arxiv.org/pdf/1909.08593.pdf#page=2
    Source: https://github.com/openai/lm-human-preferences/blob/master/lm_human_preferences/train_policy.py
    """

# L189-L215 PPO 损失：value clip + ratio clip + 无偏 KL 估计 (k3)
values_clipped = torch.clamp(values, old_values - self.cliprange_value,
                                     old_values + self.cliprange_value)
vf_loss1 = (values - returns) ** 2
vf_loss2 = (values_clipped - returns) ** 2
vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n

log_ratio = (logprobs - old_logprobs) * mask
ratio = torch.exp(log_ratio)
# Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
with torch.no_grad():
    approx_kl = torch.mean((ratio - 1) - log_ratio)

pg_loss1 = -advantages * ratio
pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                            1.0 + self.cliprange)
pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
loss = pg_loss + self.vf_coef * vf_loss
```

> 永久链接：[`modeling_ppo.py#L189-L215`](https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L189-L215)。`get_advantages_and_returns` 在 L171 还有一行 `advantages = whiten(advantages)`——对应"奖励 whiten"那一节；同文件 `cliprange` / `cliprange_value` / `cliprange_reward` 三处 clip 把 PPO 的所有数值不稳定都拍平。`approx_kl` 用的是 [http://joschu.net/blog/kl-approx.html](http://joschu.net/blog/kl-approx.html) 的 k3 估计——与本文后续"GRPO 中的 KL 散度分析"用的是同一族公式。

::: tip 阅读顺序建议
先看 trlx（注释信息密度最高，能看清"公式从哪来"），再看 DS-Chat（更接近真实工程，包含 ZeRO/Hybrid Engine 的并行调度），最后回到 TRL 的 `PPOTrainer` 看现代封装。
:::

## DPO (Direct Preference Optimization)

DPO 的核心突破：**消除了 Reward Model 和 RL 训练**，将偏好对齐简化为一个分类损失。

### 从 RLHF 到 DPO 的数学推导

RLHF 的优化目标是：

$$
\max_{\pi_\theta} \mathbb{E}_{x, y \sim \pi_\theta}[r(x,y)] - \beta D_{KL}[\pi_\theta(y|x) || \pi_{\text{ref}}(y|x)]
$$

这个 KL 约束的优化问题有闭合解：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)
$$

将奖励函数反解出来：

$$
r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

代入 Bradley-Terry 模型（$Z(x)$ 被消掉），得到 **DPO 损失**：

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

### DPO 损失的直觉

- **增大** $\pi_\theta(y_w|x)$：让模型更可能生成 chosen 回答
- **减小** $\pi_\theta(y_l|x)$：让模型更不可能生成 rejected 回答
- $\beta$ 控制偏离参考策略的程度：$\beta$ 越大，优化越保守

::: tip 为什么必须有 ref 模型？SFT 的隐藏副作用
SFT 在拟合 chosen 时会**无意提高 rejected 的生成概率**——因为 rejected 也来自相似分布。结果是 SFT 后 chosen 的概率高，但与 rejected 的差距未必拉开。

这正是 DPO 用 $\log\pi/\pi_{\text{ref}}$ 而不是 $\log\pi$ 作为优势项的原因：只奖励**相对参考模型的提升**，避免把"原本就高"误读为"学会了偏好"。等价地说，ref 模型扮演了一个 anchor，DPO 学的是"相对 anchor 的偏好方向"。

ORPO 用 odds ratio 把 SFT 与偏好对齐合并到一阶段，则是另一种解决思路（详见 §ORPO）。
:::

### PyTorch 实现 DPO

```python
import torch.nn.functional as F

def compute_log_probs(model_logits, token_ids):
    """从模型输出的 logits 中提取每个位置对应 token 的 log 概率
    
    Args:
        model_logits: 模型输出  [batch, seq_len, vocab_size]
        token_ids:    目标 token [batch, seq_len]
    Returns:
        per_token_logps: 每个 token 的 log 概率 [batch, seq_len]
    """
    log_dist = F.log_softmax(model_logits, dim=-1)            # 归一化为 log 概率分布
    per_token_logps = log_dist.gather(2, token_ids.unsqueeze(2)).squeeze(2)
    return per_token_logps

# DPO 前向：需要当前模型和参考模型分别对 chosen/rejected 计算 log prob
with torch.no_grad():
    ref_chosen_logits = ref_model(input_ids=x_chosen).logits
    ref_rejected_logits = ref_model(input_ids=x_rejected).logits
policy_chosen_logits = model(input_ids=x_chosen).logits
policy_rejected_logits = model(input_ids=x_rejected).logits

# 计算 log 概率
logp_chosen_ref = compute_log_probs(ref_chosen_logits, x_chosen)
logp_chosen = compute_log_probs(policy_chosen_logits, x_chosen)
logp_rejected_ref = compute_log_probs(ref_rejected_logits, x_rejected)
logp_rejected = compute_log_probs(policy_rejected_logits, x_rejected)

# DPO Loss 计算
beta = 0.1
policy_diff = logp_chosen - logp_rejected             # log(pi(yw)/pi(yl))
baseline_diff = logp_chosen_ref - logp_rejected_ref    # log(pi_ref(yw)/pi_ref(yl))
dpo_logits = policy_diff - baseline_diff               # DPO logits
losses = -F.logsigmoid(beta * dpo_logits)              # DPO loss
loss = losses.mean()
```

### 主流框架的 DPO loss 入口

上面的"序列级标量 + sigmoid 对比"模式不是 TRL 独有——四大主流 RLHF 框架在 DPO loss 计算上**完全收敛**到同一个写法。下表给出每家"可以一行翻到"的入口，对照阅读能解掉很多工程疑惑（concatenated forward、padding mask、SimPO/IPO/ORPO 变体怎么共享同一套基础设施等）：

| 框架 | 入口文件:行号 | 关键代码片段 |
|---|---|---|
| **HuggingFace TRL** | [`trl/trainer/dpo_trainer.py:1195`](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py) | `chosen_logratios = chosen_logps - ref_chosen_logps` |
| **PKU safe-rlhf** | [`safe_rlhf/algorithms/dpo/trainer.py:163`](https://github.com/PKU-Alignment/safe-rlhf/blob/main/safe_rlhf/algorithms/dpo/trainer.py) | `better_log_prob = better_sequence_log_probs[i, slice].sum(dim=-1)` |
| **OpenRLHF** | [`openrlhf/trainer/dpo_trainer.py:386-387`](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/dpo_trainer.py) | `logprobs_sums = (per_token_logps * loss_masks).sum(-1)` （同时也提供 `logprobs_means` 长度归一化变体，对应 SimPO 思路） |
| **LLaMA-Factory** | [`llamafactory/train/dpo/trainer.py:189`](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/train/dpo/trainer.py) | 函数签名 `policy_chosen_logps: torch.Tensor`（标量级），DPO/IPO/ORPO/SimPO/BCO 共用此接口 |

::: tip 共识与差异
**共识**：4 家全部走"先在 completion 区间把 token log p 求和→得到序列级标量→在标量上做 sigmoid 对比"。任何"token-level mask 后求平均再过 sigmoid"的写法都不在主流路径上。

**差异**：
- TRL / LLaMA-Factory **必须显式传入 ref model 的 logp**（或 forward 时一起算）；OpenRLHF 在 [L325](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/dpo_trainer.py) 把 `chosen_logps = all_logps_sum[: chosen_ids.shape[0]]` 与 ref 用同一个 `concatenated_forward` 一次算完，节省一次前向。
- LLaMA-Factory 在同一个文件里用 `if-else` 复用同一份 logp 计算给 5 种 loss（DPO/IPO/ORPO/SimPO/BCO），是工程化共享基础设施的范本。
:::

### 最小可跑 DPO：笔记本本地 < 1 分钟

::: tip 这一节解决什么问题
上面的 `compute_log_probs` 只是公式落地；下面是一个**真能跑、能看到 reward margin 上升**的最小例子。用 [distilgpt2](https://huggingface.co/distilbert/distilgpt2) + 4 条手写偏好对（chosen=礼貌助手风、rejected=粗鲁风），CPU 上 30-60 秒就能看到训练前后 `logp(chosen) - logp(rejected)` 显著拉开。
:::

#### Step 1：模型 + 偏好数据

```python
import copy, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

PAIRS = [
    {"prompt": "User asks for help.\nAssistant:",   "chosen": " Sure, I'd be happy to help. What do you need?",         "rejected": " Figure it out yourself, I'm busy."},
    {"prompt": "User says thanks.\nAssistant:",     "chosen": " You're welcome! Let me know if anything else comes up.", "rejected": " Whatever, don't bother me again."},
    {"prompt": "User asks a question.\nAssistant:", "chosen": " That's a great question. Here is a clear answer.",        "rejected": " Stop asking dumb questions."},
    {"prompt": "User reports a bug.\nAssistant:",   "chosen": " Thanks for reporting, I'll look into it right away.",     "rejected": " Not my problem, deal with it."},
]

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
policy = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
ref = copy.deepcopy(policy).eval()                    # ref 冻结
for p in ref.parameters(): p.requires_grad_(False)
```

#### Step 2：只对 response 段累计 log prob

DPO 的核心约束是"只看 response 部分的概率变化"——prompt 段是输入条件，不参与偏好比较。

```python
def encode_pair(prompt, response, max_len=128):
    full_ids = tokenizer(prompt + response, return_tensors="pt",
                         truncation=True, max_length=max_len).input_ids[0]
    prompt_len = len(tokenizer(prompt).input_ids)
    return full_ids, prompt_len

def response_logp(model, full_ids, prompt_len):
    """response 段所有 token log prob 之和（左移一位对齐 next-token）"""
    ids = full_ids.unsqueeze(0).to(device)
    logits = model(ids).logits[0, :-1]                                    # [T-1, V]
    targets = ids[0, 1:]
    logp = F.log_softmax(logits, -1).gather(1, targets.unsqueeze(1)).squeeze(1)
    return logp[prompt_len-1:].sum()                                      # 只累 response
```

#### Step 3：DPO loss（与公式一一对应）

```python
def dpo_loss(policy, ref, pair, beta=0.1):
    pw_ids, pw_len = encode_pair(pair["prompt"], pair["chosen"])
    pl_ids, pl_len = encode_pair(pair["prompt"], pair["rejected"])
    logp_pol_w = response_logp(policy, pw_ids, pw_len)
    logp_pol_l = response_logp(policy, pl_ids, pl_len)
    with torch.no_grad():
        logp_ref_w = response_logp(ref, pw_ids, pw_len)
        logp_ref_l = response_logp(ref, pl_ids, pl_len)
    diff = (logp_pol_w - logp_ref_w) - (logp_pol_l - logp_ref_l)          # 关键量
    loss = -F.logsigmoid(beta * diff)
    return loss, (beta * diff).item()                                     # margin 用于监控
```

`diff` 就是公式里的 $\log\frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)} - \log\frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)}$，乘上 β 后过 `-logsigmoid` 即得 DPO loss。

#### Step 4：训练 + 监控 reward margin

```python
import random
random.seed(0)
opt = torch.optim.AdamW(policy.parameters(), lr=1e-5)
margins = []
for step in range(60):
    pair = random.choice(PAIRS)
    loss, margin = dpo_loss(policy, ref, pair, beta=0.1)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    margins.append(margin)
    if step % 15 == 0:
        print(f"step {step:2d} | loss={loss.item():.4f} | margin={margin:+.3f}")
print(f"\nmargin first10 avg = {sum(margins[:10])/10:+.3f}")
print(f"margin last10  avg = {sum(margins[-10:])/10:+.3f}")
```

reward margin = $\beta \cdot \text{diff}$。**期望现象**：随训练推进 `margin` 从接近 0 上升至显著正值（通常 +0.05 ~ +0.3），说明策略在拉开 chosen 与 rejected 的概率差。

#### Step 5：训练前后 chosen / rejected 概率差对比

```python
def logp_gap(model, pair):
    pw_ids, pw_len = encode_pair(pair["prompt"], pair["chosen"])
    pl_ids, pl_len = encode_pair(pair["prompt"], pair["rejected"])
    with torch.no_grad():
        return (response_logp(model, pw_ids, pw_len)
                - response_logp(model, pl_ids, pl_len)).item()

for pair in PAIRS:
    print(f"prompt: {pair['prompt'].splitlines()[0]}")
    print(f"  ref    Δlogp(chosen-rejected) = {logp_gap(ref, pair):+.3f}")
    print(f"  policy Δlogp(chosen-rejected) = {logp_gap(policy, pair):+.3f}")
```

**期望**：`policy` 的 Δlogp 比 `ref` 大若干个单位。如果两者一样大，说明训练没生效；如果 `policy` 反而比 `ref` 小，说明 β 设得太大或学习率太低。

::: warning 这只是教学样例
4 条偏好对、60 步、distilgpt2 远不够生产用——它的目的是把"DPO loss 在做什么、reward margin 怎么被拉开"这条数据流跑通让你看见。真要训出对齐模型，请直接跳到下文 §TRL 实战 · DPO，几万到几十万条 preference 数据 + LoRA + 多卡才是实际配方。
:::

### DPO 的过拟合问题与 IPO

DPO 存在过拟合风险：$\pi_\theta(y_l)$ 可能被压到 0，导致 BT 模型得分趋向 $+\infty$。IPO (Identity Preference Optimization) 通过将损失改为二次形式来避免：

$$
\mathcal{L}_{\text{IPO}} = \mathbb{E} \left( h_\pi(y_w, y_l, x) - \frac{\tau^{-1}}{2} \right)^2
$$

```python
# IPO loss：回归损失而非分类损失
constant = 1.0 / (beta * 2.0)
losses_ipo = torch.square(dpo_logits - constant) * label
```

## GRPO (Group Relative Policy Optimization)

GRPO 是 DeepSeek 提出的方法，创新在于：**不需要 Reward Model 和 Critic 网络**，通过组内相对排名计算优势函数。

### GRPO 损失函数

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\left[\pi_\theta(o_{i,t} | q, o_{i,<t})\right]_{\text{no\_grad}}} \hat{A}_{i,t} - \beta D_{KL}\left[\pi_\theta || \pi_{\text{ref}}\right] \right]
$$

### 步骤 1：Group Sampling（组采样）

对同一个 prompt，让 policy 在采样模式下生成 $G$ 条 response。这里直接用 DeepSeek-R1 的 think/answer 标签格式，与开源生态保持一致。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

R1_SYSTEM_PROMPT = (
    "You are a helpful assistant. The reasoning process and answer are "
    "enclosed within <think> </think> and <answer> </answer> tags."
)

def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": R1_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)


def sample_group(policy, tokenizer, question: str, num_generations: int = 8,
                  max_new_tokens: int = 512, temperature: float = 0.9):
    """对同一个 prompt 一次性采样 G 条 response（GRPOTrainer 的 num_generations 字段）"""
    prompt = build_prompt(tokenizer, question)
    enc = tokenizer([prompt] * num_generations, return_tensors="pt").to(policy.device)

    out = policy.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        return_dict_in_generate=True,
    )
    return enc.input_ids, out.sequences[:, enc.input_ids.shape[1]:]   # prompt_ids, completion_ids
```

### 步骤 2：可验证奖励（Verifiable Reward）

GRPO 在数学 / 代码 / 推理任务上的核心优势是 **奖励可以由规则给出**——不需要训练 Reward Model。下面两个函数对应 DeepSeek-R1 论文中的 *accuracy reward* 和 *format reward*。

```python
import re

ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
FORMAT_TAG = re.compile(r"^.*?<think>.+?</think>\s*<answer>.+?</answer>\s*$", re.DOTALL)


def accuracy_reward(completion: str, gold: str) -> float:
    """从 <answer>…</answer> 中抽出最终答案，跟标准答案做字符串比对"""
    match = ANSWER_TAG.search(completion)
    if match is None:
        return 0.0
    return 1.0 if match.group(1).strip() == gold.strip() else 0.0


def format_reward(completion: str) -> float:
    """要求输出严格遵守 <think>…</think><answer>…</answer> 的形态"""
    return 1.0 if FORMAT_TAG.match(completion) else 0.0


def total_reward(completion: str, gold: str) -> float:
    return accuracy_reward(completion, gold) + 0.2 * format_reward(completion)
```

### 步骤 3：组内相对优势（Group-Relative Advantage）

GRPO 不要 critic，把组内其他样本的奖励当作 baseline，做 z-score 归一化：

$$
\hat{A}_i = \frac{r_i - \operatorname{mean}(\mathbf{r})}{\operatorname{std}(\mathbf{r}) + \epsilon}
$$

```python
def group_relative_advantage(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """rewards: [G]，返回同形状的标准化优势"""
    return (rewards - rewards.mean()) / (rewards.std() + eps)


# 直观示例：同一组里只有一条对，对的那条优势最大
print(group_relative_advantage(torch.tensor([1., 0., 0., 0., 0., 0.])))
# tensor([ 2.2361, -0.4472, -0.4472, -0.4472, -0.4472, -0.4472])

print(group_relative_advantage(torch.tensor([1., 1., 1., 1., 1., 1.])))
# tensor([0., 0., 0., 0., 0., 0.])  ← 全对，无梯度信号

print(group_relative_advantage(torch.tensor([0., 0., 0., 0., 0., 0.])))
# tensor([0., 0., 0., 0., 0., 0.])  ← 全错，无梯度信号
```

> **关键洞察**：全对或全错时 std=0，组内优势退化为 0；GRPO 的有效学习信号来自"组内有难有易"。这也是为什么 R1-Zero 训练时要刻意控制题目难度分布。

### 步骤 4：GRPO Loss

GRPO 的 loss 长得像 PPO（也有 ratio + clip），但有两个差异：① 优势直接来自步骤 3 的标准化 reward；② KL 项放在 loss 内（per-token），而不是塞进 reward。

```python
def low_variance_kl(new_logprobs, ref_logprobs):
    """K3 估计：exp(log r) - log r - 1，恒非负且方差小（Schulman approximation）"""
    log_ratio = ref_logprobs - new_logprobs
    return log_ratio.exp() - log_ratio - 1


def grpo_loss(new_logprobs, old_logprobs, ref_logprobs, advantages,
              completion_mask, clip_range: float = 0.2, kl_coef: float = 0.04):
    """
    Args:
        new_logprobs:    [G, T_resp]   当前 policy 的逐 token log-prob
        old_logprobs:    [G, T_resp]   采样时刻的 log-prob（detached）
        ref_logprobs:    [G, T_resp]   reference 模型的 log-prob
        advantages:      [G]           来自 group_relative_advantage
        completion_mask: [G, T_resp]   1 表示该 token 在 response 内
    """
    # PPO-clip surrogate（注意 advantages 在 token 维上是常数）
    ratio = (new_logprobs - old_logprobs).exp()
    adv = advantages.unsqueeze(-1)
    pg = torch.minimum(ratio * adv, ratio.clamp(1 - clip_range, 1 + clip_range) * adv)

    # KL 正则
    kl = low_variance_kl(new_logprobs, ref_logprobs)

    # GRPO 论文的 loss 形式：先在 token 维取均值，再在组维取均值
    per_token = -(pg - kl_coef * kl) * completion_mask
    per_sample = per_token.sum(-1) / completion_mask.sum(-1).clamp(min=1)
    return per_sample.mean()
```

> 实现参考：[HuggingFace TRL · GRPOTrainer](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)（Apache-2.0）和 [DeepSeek-R1 论文](https://arxiv.org/abs/2501.12948)。本节代码是教学化简版，省略了 importance sampling reweighting、length normalization 的若干 ablation 选项。

### 工程实现参考

GRPO 在工业界已有多个权威开源实现，对照阅读能补全本节简化省略的 importance sampling reweighting、length normalization、KL 控制器等细节：

| 仓库 | 定位 | 看哪里 |
|---|---|---|
| **[huggingface/trl](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)** | 单机 GRPO 教学级标杆，每个 buffer/loss 项对照论文公式 | `GRPOTrainer.compute_loss` 里的 importance sampling 与 token-level KL |
| **[volcengine/verl](https://github.com/volcengine/verl)** | 字节跳动，**当前 GRPO 工业首选**，FSDP/Megatron + Ray 大规模训练 | [`verl/trainer/ppo/core_algos.py`](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py) 的 GAE/GRPO advantage + `examples/grpo_trainer/` 的 R1 复现 yaml |
| **[OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)** | Ray 多机 PPO/GRPO/DPO/REINFORCE++ 一体，模块化清晰 | [`openrlhf/trainer/ppo_trainer.py`](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_trainer.py) 把 advantage 估计、KL 控制器、Critic warmup 拆成独立模块 |
| **[THUDM/slime](https://github.com/THUDM/slime)** | 智谱，**全异步 GRPO**——rollout 与 train 解耦 | Megatron 后端 + 训推分离的工程范本 |
| **[huggingface/open-r1](https://github.com/huggingface/open-r1)** | R1 完整复现 pipeline，端到端最完整 | `src/open_r1/grpo.py` + `recipes/` 下数学/代码 reward 组合 |

::: tip 不同视角怎么读
- **公式 → 代码翻译**：先读 `trl`，最易理解
- **R1 怎么跑通**：`open-r1` 有可直接复制的 yaml + 数据脚本
- **大规模 / 长序列 / 异步**：`verl`（同步）与 `slime`（异步）代表两种工程哲学，参见 [R1 复现指南](/deep-dives/r1-reproduction#技术方案选择-verl-vs-slime)
:::

#### Advantage 归一化的 epsilon：三家约定差异

GRPO 的 advantage 归一化都是 `(r - mean) / (std + eps)`，但 `eps` 取值在工业实现里并不统一——这个看似不起眼的常数直接影响**奖励方差极小（如组内全对/全错）时的数值稳定性**：

| 仓库 | epsilon | 看哪里 | 设计取舍 |
|---|---|---|---|
| **TRL `GRPOTrainer`** | `1e-4` | [`grpo_trainer.py:2253/2260/2264`](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py) | 偏保守：当 std → 0（全对/全错组）时直接把 advantage 压向 0，避免梯度爆炸 |
| **verl** | `1e-6` | [`verl/trainer/ppo/core_algos.py:272/339/366/476`](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py) | 折中：保留极小方差组的信号，但仍兜底；同时显式区分 `epsilon_low` / `epsilon_high` 做不对称 clip |
| **OpenRLHF** | `1e-9` | [`openrlhf/trainer/ppo_utils/experience_maker.py:270`](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_utils/experience_maker.py) | 几乎不平滑：信任上游的 reward shaping 与组采样，让数值最接近"纯净" z-score |

::: warning 这个常数为什么重要
当一组 G 个 rollout 的 reward 完全相同（例如全 0 或全 1），`std = 0`。
- `eps=1e-4`：`adv ≈ (r - mean) / 1e-4`，但因为 `r = mean`，分子也是 0，结果稳定为 0——**该组对 loss 无贡献，安全跳过**
- `eps=1e-9`：分子是 0 时仍是 0；但若 reward 微小抖动（如全是 0.999±1e-7），`adv` 会被放大 10⁹ 倍——**全对组也能反向传播一个超大梯度**

→ 实战推荐：**reward 离散（accuracy / format）就跟 TRL 用 `1e-4`；reward 连续平滑（reward model）才考虑调小到 `1e-6` 以下**。
:::

### 最小可跑 GRPO：笔记本本地 < 1 分钟

::: tip 这一节解决什么问题
上面的 `sample_group / accuracy_reward / group_relative_advantage / grpo_loss` 都是离散模块。下面用一个**小到极致的 toy 任务**——让 tiny GPT 学会在 `<bos>` 后输出指定字符 `"a"`——把这些模块串成一次端到端可跑的 GRPO 循环，CPU 30-60 秒看到 reward 从 ~0.1（随机）爬到 ~1.0。任务足够简单，但 rollout → 可验证奖励 → 组内 advantage → clipped surrogate + K3 KL → policy update → old ← new 这条核心数据流和真实 DeepSeek-R1 训练一模一样。
:::

#### Step 1：设置 tiny 任务、tiny 模型、reward 函数

```python
import copy, torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

VOCAB = list("abcdefghi") + ["<bos>"]
V = len(VOCAB)                                       # 10 tokens
stoi = {c: i for i, c in enumerate(VOCAB)}
TARGET = stoi["a"]                                   # 让模型学会输出 'a'

device = "cuda" if torch.cuda.is_available() else "cpu"
config = GPT2Config(vocab_size=V, n_positions=4, n_embd=16, n_layer=1, n_head=2)
policy = GPT2LMHeadModel(config).to(device)
ref    = copy.deepcopy(policy).eval()                # KL 惩罚的 anchor，永不更新
old    = copy.deepcopy(policy).eval()                # rollout 时的 frozen 副本
for m in (ref, old):
    for p in m.parameters(): p.requires_grad_(False)

prompt_id = torch.tensor([[stoi["<bos>"]]], device=device)  # 长度 1 的 prompt
def reward_fn(token_id): return 1.0 if token_id == TARGET else 0.0
```

#### Step 2：Group sampling — 用 old policy 采 G 条 single-token completion

```python
@torch.no_grad()
def sample_group(old_policy, G=32):
    logits = old_policy(prompt_id).logits[0, -1]                 # [V]
    probs = F.softmax(logits, -1)
    samples = torch.multinomial(probs, G, replacement=True)      # [G]
    old_lp = F.log_softmax(logits, -1)[samples]                  # [G]
    rewards = torch.tensor([reward_fn(s.item()) for s in samples],
                           device=device, dtype=torch.float)
    return samples, old_lp, rewards

samples, old_lp, rewards = sample_group(old, G=32)
print(f"random init: reward mean = {rewards.mean():.3f}  (≈ 1/V = {1.0/V:.3f})")
```

#### Step 3：Group-relative advantage + GRPO step（含 clip + K3 KL）

```python
def grpo_advantage(rewards, eps=1e-4):
    # 组内 z-score：DeepSeekMath / TRL GRPOTrainer 的标准做法（eps 与 TRL 上游一致）
    return (rewards - rewards.mean()) / (rewards.std() + eps)

def grpo_step(policy, ref, samples, old_lp, advantages, clip=0.2, kl_coef=0.04):
    logits = policy(prompt_id).logits[0, -1]
    new_lp = F.log_softmax(logits, -1)[samples]
    with torch.no_grad():
        ref_lp = F.log_softmax(ref(prompt_id).logits[0, -1], -1)[samples]
    ratio = (new_lp - old_lp).exp()
    pg = torch.minimum(ratio * advantages,
                       ratio.clamp(1 - clip, 1 + clip) * advantages)
    log_ratio = ref_lp - new_lp                         # K3：exp(lr) - lr - 1，恒 ≥ 0
    kl = log_ratio.exp() - log_ratio - 1
    return -(pg - kl_coef * kl).mean()                  # 注意是 -(...) 因为想最大化 pg
```

#### Step 4：训练主循环（rollout → K 次 inner update → old ← new）

```python
opt = torch.optim.AdamW(policy.parameters(), lr=1e-2)
history = []

for outer in range(40):
    samples, old_lp, rewards = sample_group(old, G=32)
    advantages = grpo_advantage(rewards)
    history.append(rewards.mean().item())
    for _ in range(2):                                  # GRPO 论文常用 K=2~4 inner update
        loss = grpo_step(policy, ref, samples, old_lp, advantages)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    old.load_state_dict(policy.state_dict())            # rollout 副本同步
    if outer % 10 == 0:
        print(f"outer {outer:2d} | reward = {rewards.mean():.3f} | loss = {loss.item():+.4f}")

print(f"\nfirst 5 outer reward avg = {sum(history[:5])/5:.3f}")
print(f"last  5 outer reward avg = {sum(history[-5:])/5:.3f}")
```

#### Step 5：检验最终策略分布

```python
@torch.no_grad()
def show_dist(model, name):
    probs = F.softmax(model(prompt_id).logits[0, -1], -1)
    print(f"--- {name} P(token | <bos>) ---")
    for i, c in enumerate(VOCAB):
        bar = "█" * int(probs[i].item() * 40)
        mark = " ← TARGET" if i == TARGET else ""
        print(f"  {c:>5}: {probs[i].item():.3f} {bar}{mark}")

show_dist(ref, "ref (frozen, untrained)")
show_dist(policy, "policy (after GRPO)")
```

期望现象：**ref** 上各 token 概率接近均匀（约 0.1）；**policy** 上 `a` 的概率拉到 0.7+，其他 token 被压低——这就是 GRPO 用"组内相对奖励 + clipped surrogate"驱动策略偏移到目标分布的视觉证据。整段 ~70 行代码就是 R1-Zero 训练循环的最小同构。

::: warning 这只是教学样例
40 个 outer step、单 token completion、10 个字符的 vocab 远不够生产用——目的只是把"GRPO 的算法骨架"跑成可见的 reward 曲线。**真实的 GRPO**（如 DeepSeek-R1 / open-r1）每步需要几百 token 的 rollout、数百题的难度分布、ref/old 双副本带来的显存压力、vLLM 加速的 colocate 部署。把这段 toy 代码理解透后，再去看 [TRL · `GRPOTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py) 与 [open-r1](https://github.com/huggingface/open-r1) 的产线实现，会顺很多。
:::

### 代码参考：minimind 的 GRPO + CISPO 实现

[jingyaogong/minimind](https://github.com/jingyaogong/minimind) 是 26M 参数的中文社区"完整训练栈"项目——pretrain / SFT / LoRA / DPO / GRPO / 蒸馏 / Agent 全部纯 PyTorch 手撕，单 GPU 2 小时即可跑通。其 `trainer/train_grpo.py` 把 group-relative advantage、K3 KL、CISPO 三个工程要点压缩在 23 行内，是上面教学版代码很好的工程化对照。

特别值得抄读的是 [`train_grpo.py` L119-L141](https://github.com/jingyaogong/minimind/blob/master/trainer/train_grpo.py#L119-L141) 中的 **CISPO**（Clipped Importance Sampling Policy Optimization）分支：与标准 PPO 把 ratio 直接送进 `min(unclipped, clipped)` 不同，CISPO 把 `clamp` 后的 ratio 先 `detach()`，仅作为加权系数，让梯度直接走到 `per_token_logps`——数值更稳定、避免极端 ratio 把梯度拉飞，是上文 [`grpo_loss`](#步骤-4-grpo-loss) 章节没展开的工程细节。

```python
grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_gen]

kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1  # K3 估计
ratio = torch.exp(per_token_logps - old_per_token_logps)
if args.loss_type == "cispo":
    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
    per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
else:
    clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
    per_token_loss1 = ratio * advantages.unsqueeze(1)
    per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
    per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
```

注意 CISPO 分支里 `clamped_ratio` 被 `.detach()` 后只做权重——梯度只对 `per_token_logps` 求导，而不再对 ratio 自身求导，这正是它区别于标准 PPO clip 的核心。

## GRPO 中的 KL 散度分析

GRPO 使用 KL 散度约束策略不偏离参考模型太远。不同的 KL 估计方式在偏差和方差上有显著差异，直接影响训练稳定性。

### 三种 KL 估计方式

设 $r = \frac{p(x)}{q(x)}$，从 $q$ 分布采样时，有三种常用的 KL 估计：

| 估计方式 | 公式 | 特点 |
|---------|------|------|
| **K1** (log-ratio) | $-\log r$ | 无偏但有负值，方差最大 |
| **K2** (squared log) | $\frac{(\log r)^2}{2}$ | 恒正，方差较小，但有偏（偏大） |
| **K3** (Schulman 近似) | $r - 1 - \log r$ | 恒正，方差较小，近似无偏 |

参考：[John Schulman - Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)

### 为什么 K3 恒为正？

K3 的形式为 $f(r) = r - 1 - \log r$。由于 $\log r \leq r - 1$（对数不等式），所以 $r - 1 - \log r \geq 0$，等号当且仅当 $r = 1$（即 $p = q$）时成立。

```python
import torch

# 验证三种 KL 估计
p = torch.distributions.Normal(0, 1)
q = torch.distributions.Normal(0.5, 1)
x = q.sample((100_000,))

true_kl = torch.distributions.kl_divergence(p, q)  # 解析解 = 0.125
logr = p.log_prob(x) - q.log_prob(x)

k1 = -logr                          # 无偏，但方差 ~4x true_kl
k2 = logr ** 2 / 2                  # 有偏（偏大 ~5%），方差 ~1.5x
k3 = logr.exp() - 1 - logr          # 近似无偏，方差 ~1.5x

print(f"True KL: {true_kl:.4f}")
print(f"K1: mean={k1.mean():.4f}, std={k1.std():.4f}")
print(f"K2: mean={k2.mean():.4f}, std={k2.std():.4f}")
print(f"K3: mean={k3.mean():.4f}, std={k3.std():.4f}")
# K3 均值最接近真实 KL，且方差远小于 K1
```

### GRPO 使用 K3 的原因

GRPO 选择 K3（即 `exp(logr) - logr - 1`）作为 KL penalty——也就是上一节 [`low_variance_kl`](#步骤-4-grpo-loss) 的同款实现：

```python
def low_variance_kl(new_logprobs, ref_logprobs):
    """K3 估计：恒非负、方差小，是 GRPO/PPO 实战中的默认选择"""
    log_ratio = ref_logprobs - new_logprobs           # log r = log(π_ref / π_θ)
    return log_ratio.exp() - log_ratio - 1
```

相比 K1（`log π_θ - log π_ref`），K3 不会产生负值，训练更稳定；相比 K2，K3 的偏差更小。TRL 的 `GRPOTrainer` 默认就走这条路径。

::: tip KL 估计方式的选择原则
- **K1**：数学上最优（无偏），但方差太大，实际训练中波动剧烈
- **K2**：简单高效，但系统性偏大，会过度惩罚策略偏移
- **K3**：兼顾低偏差和低方差，是 GRPO / PPO 实践中的首选
:::

## KTO (Kahneman-Tversky Optimization)

KTO 基于行为经济学的 **前景理论 (Prospect Theory)**，核心优势：**不需要成对的偏好数据**，只需要独立标注"好的回答"和"差的回答"。

### 动机：为什么不需要成对数据？

DPO 要求每条训练样本包含 `(prompt, chosen, rejected)` 三元组——同一个 prompt 下的一对好/坏回答。但在实际标注中：

- 成对标注成本高（标注员需要同时阅读两个回答并比较）
- 很多场景下只有独立的质量判断（"这个回答好" / "这个回答差"）
- 不同 prompt 下的好/坏回答也包含偏好信息

KTO 的突破在于：**每条样本只需要一个 binary 标签（好/坏），不需要成对比较。**

### 前景理论 (Prospect Theory) 的启发

KTO 的损失函数设计源自 Kahneman 和 Tversky 的 **前景理论**（2002 年诺贝尔经济学奖）：

1. **损失厌恶 (Loss Aversion)**：人对损失的敏感度高于对同等收益的敏感度。丢 100 元的痛苦 > 捡到 100 元的快乐
2. **参考点依赖 (Reference Dependence)**：人评估收益/损失时依赖一个参考点，而非绝对值

映射到 LLM 对齐：
- **参考点** = 参考模型 $\pi_{\text{ref}}$ 的 KL 散度
- **收益** = chosen 回答相对参考点的提升
- **损失** = rejected 回答相对参考点的下降
- **损失厌恶** = 对差回答施加更大的惩罚权重

### 数学公式

KTO 的损失函数：

$$
\mathcal{L}_{\text{KTO}}(\pi_\theta, \pi_{\text{ref}}) = \mathbb{E}_{x,y \sim \mathcal{D}} \left[ w(y) \left(1 - v_{\text{KTO}}(x, y; \beta) \right) \right]
$$

其中：

$$
r_{\text{KTO}}(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}, \quad z_{\text{ref}} = \mathbb{E}_{x'} \left[ \beta \, \text{KL}(\pi_\theta \| \pi_{\text{ref}} | x') \right]
$$

$$
v_{\text{KTO}}(x, y; \beta) = \begin{cases} \sigma(r_{\text{KTO}} - z_{\text{ref}}) & \text{if } y \sim y_{\text{desirable}} \\ \sigma(z_{\text{ref}} - r_{\text{KTO}}) & \text{if } y \sim y_{\text{undesirable}} \end{cases}
$$

$$
w(y) = \begin{cases} \lambda_D & \text{if } y \sim y_{\text{desirable}} \\ \lambda_U & \text{if } y \sim y_{\text{undesirable}} \end{cases}
$$

- $\lambda_D / \lambda_U$ 是非对称权重，体现损失厌恶（通常 $\lambda_D > \lambda_U$）
- $z_{\text{ref}}$ 是参考点（KL 散度的期望值）

### 数据格式

```python
# KTO 数据不需要成对，每条独立标注一个 binary label
kto_dataset = {
    "prompt": [
        "请用一句话解释什么是注意力机制",
        "Transformer 中 LayerNorm 起什么作用",
        "为什么大模型推理需要 KV cache",
        "什么是混合专家模型 (MoE)",
    ],
    "completion": [
        "注意力让模型按相关性给序列里的不同 token 加权聚合，权重由 query 与 key 的点积归一化得到。",
        "稳定训练。",
        "因为不缓存就慢。",
        "MoE 把 token 路由给若干专家网络，只激活其中少数几个，从而以低算力放大模型容量。",
    ],
    "label": [True, False, False, True],  # True=desirable, False=undesirable
}
```

### KTO 损失函数实现

> 参考：Ethayarajh et al. (2024) 原始论文的损失定义、[TRL `KTOTrainer`](https://huggingface.co/docs/trl/main/en/kto_trainer) 的工程实现。

```python
import torch
import torch.nn.functional as F

def kto_loss(policy_logits, ref_logits, completion_ids, completion_mask, labels,
             beta=0.1, lambda_d=1.0, lambda_u=1.0):
    """
    KTO 损失：基于前景理论的非对称偏好优化（TRL KTOTrainer 风格）。

    Args:
        policy_logits:   [batch, seq_len, vocab]  当前策略 logits
        ref_logits:      [batch, seq_len, vocab]  参考策略 logits
        completion_ids:  [batch, seq_len]         与 logits 对齐的 next-token id
        completion_mask: [batch, seq_len]         completion 区间为 1，prompt/pad 为 0
        labels:          [batch]  bool, True=desirable, False=undesirable
        beta:            温度
        lambda_d/lambda_u: 损失厌恶权重，常取 lambda_d > lambda_u
    """
    # 1. completion 区间的序列级 log p（求和，不做长度归一化）
    def seq_logp(lgt):
        log_probs = lgt.log_softmax(dim=-1)
        per_tok = torch.gather(
            log_probs, dim=-1, index=completion_ids.unsqueeze(-1)
        ).squeeze(-1)
        return (per_tok * completion_mask).sum(dim=-1)            # [batch]

    policy_logp = seq_logp(policy_logits)
    ref_logp    = seq_logp(ref_logits)

    # 2. 隐式 reward = beta * log(π_θ / π_ref)
    implicit_r = beta * (policy_logp - ref_logp)

    # 3. 参考点 z_ref：整 batch 平均 KL，detach 不参与梯度
    z_ref = (beta * (policy_logp - ref_logp)).mean().detach()

    # 4. 前景理论的非对称损失
    desirable   = labels
    undesirable = ~labels

    loss_desirable   = 1 - torch.sigmoid(implicit_r[desirable]   - z_ref)
    loss_undesirable = 1 - torch.sigmoid(z_ref - implicit_r[undesirable])

    loss = torch.cat([
        lambda_d * loss_desirable,
        lambda_u * loss_undesirable,
    ]).mean()
    return loss
```

### KTO vs DPO 对比

| 对比项 | DPO | KTO |
|--------|-----|-----|
| 数据格式 | 成对偏好 (chosen, rejected) | 独立 binary (好/坏) |
| 标注成本 | 高（需要比较两个回答） | 低（只需判断好坏） |
| 理论基础 | Bradley-Terry 偏好模型 | Kahneman-Tversky 前景理论 |
| 参考点 | 隐含在 log-ratio 中 | 显式的 KL 参考点 $z_{\text{ref}}$ |
| 损失对称性 | 对称（chosen 和 rejected 等权） | **非对称**（损失厌恶） |
| 效果 | 成对数据充足时更优 | 数据稀缺或标注预算有限时更实用 |

::: tip 什么时候选 KTO？
1. **数据只有 thumbs up/down**：用户反馈通常是"赞/踩"，天然适合 KTO
2. **标注预算有限**：成对标注成本约为独立标注的 2-3 倍
3. **数据来源异构**：不同标注员标注的样本难以配对，KTO 直接使用
:::

> 论文：[KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)

## 各方法对比

| 方法 | 需要的模型 | 数据要求 | 计算成本 | 适用场景 |
|------|-----------|---------|---------|---------|
| **PPO** | Actor + Ref + RM + Critic (4个) | 成对偏好数据 + RM | 最高 | 通用对齐，效果最稳定 |
| **DPO** | Policy + Ref (2个) | 成对偏好数据 | 低 | 资源受限，快速对齐 |
| **GRPO** | Policy + Ref (2个) | 规则可判定的任务 | 中等 | 数学/代码/推理任务 |
| **KTO** | Policy + Ref (2个) | 非成对标注数据 | 低 | 数据收集成本高的场景 |

**选型建议**：

- 有充足算力和高质量成对数据 → **PPO**
- 有成对数据但算力有限 → **DPO**
- 推理/数学/代码任务 → **GRPO**（DeepSeek-R1 的成功验证）
- 只有好/差的独立标注 → **KTO**

## 更多对齐算法

上面介绍的 PPO、DPO、GRPO、KTO 是最主流的四种方法。随着研究推进，社区涌现了更多变体，各有侧重。

### ORPO (Odds Ratio Preference Optimization)

ORPO 的核心创新：**把 SFT 和对齐合并成一个阶段**，通过在语言建模损失上叠加一个 odds ratio 惩罚项来实现偏好对齐。

$$L_{\text{ORPO}} = L_{\text{SFT}} + \lambda \cdot \log \frac{\text{odds}(y_w | x)}{\text{odds}(y_l | x)}$$

其中 $\text{odds}(y|x) = \frac{P(y|x)}{1 - P(y|x)}$，$y_w$ 为 chosen，$y_l$ 为 rejected。

::: tip 优势
- **不需要参考模型**：比 DPO 更简单，省掉 reference model 的显存开销
- **一步到位**：SFT + 对齐同时完成，省去多阶段训练的复杂度
:::

> 论文：[ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)

::: tip ORPO 实操参数与数据集（Llama-3-8B 实测）
Maxime Labonne 在 Llama-3-8B 上的起手参考值：

- 学习率 **8e-6**——远低于 SFT 的 1e-5、DPO 的 5e-6（原论文消融定值，作者建议生产环境进一步降到 1e-6）
- `beta=0.1`（即论文里的 λ）、`max_length=1024 / max_prompt_length=512`、`paged_adamw_8bit`、`grad_accum=4`
- QLoRA 4-bit (nf4 + double quant) + LoRA `r=16/α=32/dropout=0.05`，全 7 个 linear 层
- 数据集 [mlabonne/orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k) —— 聚合 distilabel-capybara / orca / ultrafeedback / math + truthy/toxic 共 ~40k 偏好对
- L4 单卡 1k 样本约 2 小时；OrpoLlama-3-8B 在 Nous benchmark 套件上 avg **46.76**（base Llama-3 8B = 45.42）

ORPO 已被 [TRL](https://github.com/huggingface/trl)、[Axolotl](https://github.com/axolotl-ai-cloud/axolotl)、[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 主流微调库收录。
:::

### CPO (Contrastive Preference Optimization)

CPO 同样移除了对参考模型的依赖，使用 **对比损失 (contrastive loss)** 直接在偏好对上优化。

::: details SimPO — CPO 的实用变体
SimPO (Simple Preference Optimization) 在 CPO 基础上增加了两个改进：
1. **长度归一化奖励**：用序列平均 log-probability 作为隐式奖励，避免模型偏好长回答
2. **目标奖励间距 (target reward margin)**：在 chosen 和 rejected 之间强制一个最小间距 $\gamma$

$$L_{\text{SimPO}} = -\log \sigma\left(\frac{\beta}{|y_w|} \log \pi(y_w|x) - \frac{\beta}{|y_l|} \log \pi(y_l|x) - \gamma\right)$$
:::

### RLOO (REINFORCE Leave-One-Out)

RLOO 是一种 **无需 Critic 网络** 的在线 RL 方法，使用 leave-one-out baseline 来估计优势。

**核心思路**：对每个 prompt 采样 $K$ 个 response，每个 response 的 baseline = 其他 $K-1$ 个 response 的平均奖励。

$$A_i = r_i - \frac{1}{K-1} \sum_{j \neq i} r_j$$

::: tip 与 GRPO 的关系
RLOO 和 GRPO 思路相似——都用组内其他样本来估计 baseline，不需要额外的 value network。区别在于 GRPO 用组内均值和标准差做归一化（advantage normalization），而 RLOO 直接用 leave-one-out 均值。RLOO 的方差更低，理论性质更好。
:::

### Online DPO

标准 DPO 是 **离线 (offline)** 的——在固定的偏好数据集上训练。Online DPO 在训练过程中 **实时生成** response 并排序：

1. 当前策略对 prompt 生成多个 response
2. 用 judge 模型 / reward model 对生成的 response 打分排序
3. 构造新的偏好对 (chosen, rejected) 进行 DPO 更新

::: warning 为什么需要 Online？
离线 DPO 的偏好数据来自其他模型（如 GPT-4），与当前策略的分布存在 **分布偏移 (distribution shift)**。Online DPO 用自身生成的数据训练，弥合了这一差距，效果通常优于离线版本。
:::

### PRM (Process Reward Model)

传统的 Reward Model 是 **结果奖励模型 (ORM)**：只对最终回答打分。PRM 提供 **步级监督 (step-level supervision)**：对推理过程中的每一步都给出奖励。

| | ORM (结果奖励) | PRM (过程奖励) |
|---|---|---|
| 监督粒度 | 整个回答 | 每个推理步骤 |
| 训练数据 | 好/坏回答对 | 每步标注 correct/incorrect |
| 适用场景 | 通用对话 | 数学、代码、逻辑推理 |
| 优势 | 标注成本低 | 定位错误更精确，信用分配更清晰 |

```
prompt: "求解 2x + 3 = 7"

Step 1: 2x + 3 = 7        ✅ (PRM: +1)
Step 2: 2x = 4            ✅ (PRM: +1)  
Step 3: x = 2             ✅ (PRM: +1)

vs.

Step 1: 2x + 3 = 7        ✅ (PRM: +1)
Step 2: 2x = 10           ❌ (PRM: -1)  ← PRM 能精确定位这一步出错
Step 3: x = 5             ❌ (PRM: -1)
```

::: tip PRM 与 RL 的结合
PRM 可以作为 PPO / GRPO 等 RL 算法的奖励信号源，用步级奖励替代结果级奖励，让策略梯度的信用分配更准确。OpenAI 的 [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) 是这一方向的代表工作。
:::

## Constitutional AI / RLAIF

一句话定位：**CAI 把"对齐目标"从大量人工偏好标注换成了"少量原则文本（宪法） + AI 自我批判"**——只用人类标注 helpfulness，harmlessness 完全交给 AI 反馈。来源：Bai et al. 2022, Anthropic, [arXiv 2212.08073](https://arxiv.org/abs/2212.08073)。

论文摘要（p.1）即点明动机："train a harmless AI assistant through self-improvement, **without any human labels identifying harmful outputs**. The only human oversight is provided through a list of rules or principles, and so we refer to the method as 'Constitutional AI'."

### 两阶段流程

CAI 把 RLHF 的标准管线拆成两段（见论文 §1.2，p.5）：

**Stage 1 — SL-CAI：Critique → Revision → Supervised Learning**

1. 用一个 helpful-only RLHF 模型对红队提示生成有害的初始回答；
2. 让同一个模型按宪法里的某条原则**批判（critique）**自己的回答；
3. 让它再**修订（revise）**原回答以消除批判中指出的问题；
4. 在迭代多轮后的修订答案上做 SFT，得到 SL-CAI 模型。

> 论文原话："We then ask the model to critique its response according to a principle in the constitution, and then revise the original response in light of the critique. We revise responses repeatedly in a sequence, where we randomly draw principles from the constitution at each step." (§1.2)

**Stage 2 — RL-AIF：AI Comparison → Preference Model → RL**

1. 用 SL-CAI 模型对同一个 prompt 采两个回答；
2. 把"prompt + 两个回答 + 一条宪法原则"喂回 helpful-only 模型，让它做多选题，选出更符合该原则的那个，得到 AI-generated 偏好对；
3. 把这批 AI 偏好（harmlessness）与人类偏好（helpfulness）混合，训练 PM；
4. 以 PM 为 reward signal 对 SL-CAI 做标准 PPO，得到 RL-CAI（即 RLAIF）。

论文称其为 **RLAIF（RL from AI Feedback）**，相比 RLHF "replaces reinforcement learning from human feedback ... without any human feedback labels for harms"（p.2）。

### 宪法原则示例

CAI 论文里的"宪法"是一组短小的自然语言原则。Anthropic 后续在 [Claude's Constitution](https://www.anthropic.com/news/claudes-constitution) 公开了更新后的、面向产品 Claude 的版本。从该页面摘录两条具体表述：

> "Claude should never provide significant uplift to a bioweapons attack."

> "Create cyberweapons or malicious code that could cause significant damage if deployed." —— 列在 **hard constraints**（绝对禁止行为）之中。

这些原则被用作 SL-CAI 阶段批判/修订的 prompt 模板，也是 RL-AIF 阶段让 AI 对比两个回答时的判定标准。

### 与 RLHF 的对比

| 维度 | 标准 RLHF | Constitutional AI / RLAIF |
|------|-----------|---------------------------|
| harmlessness 监督来源 | 数万条人工 (chosen, rejected) | 一组（≈10 条量级）自然语言原则 + AI 自我批判 |
| 标注成本 | 高（每个样本都需人审） | 低（写宪法一次性，标注由 AI 完成） |
| 可解释性 | 偏好藏在权重里，难以审计 | 训练目标即"这条原则" → 行为可追溯 |
| 可扩展性 (scaling supervision) | 受限于人类审核速度 | AI 能力越强，监督质量越高（论文 §1.1 "Scaling Supervision"） |
| 拒答倾向 | 易把 harmlessness 训成"evasive" | 显式鼓励**解释拒绝原因**而非简单回避（论文 §1：reduce evasiveness） |
| 主流实现 | TRL `PPOTrainer` / DS-Chat / trlx | Anthropic 内部；开源近似：[`anthropic/ConstitutionalHarmlessnessPaper`](https://github.com/anthropics/ConstitutionalHarmlessnessPaper) few-shot prompts |

::: tip 在自己项目里复现 CAI 的轻量做法
不必从零做 RLAIF。常见的简化路径是：(1) 写 5-10 条领域宪法；(2) 用一个强 helpful 模型做 critique-revise，把修订结果当 SFT 数据；(3) 用任意 judge LLM 做成对偏好打分，再走标准 DPO。这条路绕开了 PM + PPO 的复杂工程，被许多后续 RLAIF 论文沿用。
:::

## TRL 实战：对齐训练实现

[TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) 是 Hugging Face 提供的对齐训练框架，支持 DPO、GRPO、PPO、KTO 等主流算法。下面通过代码展示核心用法。

### DPO 训练

```python
from trl import DPOConfig, DPOTrainer

training_args = DPOConfig(
    output_dir="dpo-model",
    beta=0.1,            # KL 惩罚强度，越大越保守
    loss_type="sigmoid",  # 可选 "sigmoid", "hinge", "ipo"
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # 需要 "prompt", "chosen", "rejected" 三列
)
trainer.train()
```

::: details DPO 数据格式示例
```python
dataset = {
    "prompt": ["请解释量子计算", "写一首关于秋天的诗"],
    "chosen": ["量子计算利用量子力学原理...", "秋风起，落叶知..."],
    "rejected": ["不知道", "春天来了..."],
}
```
:::

### GRPO 训练

```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    output_dir="grpo-model",
    num_generations=8,      # 每个 prompt 生成 G 个 completion
    beta=0.0,               # KL 惩罚（0 = 不使用，当前主流实践）
    scale_rewards=True,     # 按 std(rewards) 归一化
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=accuracy_reward,  # 可以是函数或奖励模型
    train_dataset=dataset,
)
trainer.train()
```

**GRPO 关键参数解读**：

- `num_generations`：每个 prompt 生成 $G$ 个 completion，组内相对排名计算 advantage。$G$ 越大估计越准但显存开销越高，通常取 4~16
- `reward_funcs`：可以是 **基于规则的函数**（如数学题答案对错判断），也可以是 **奖励模型**
- `scale_rewards=True`：将奖励除以组内标准差进行归一化。设为 `False` 可避免难度偏差——简单题组内方差小，归一化后梯度被放大（来自 [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783)）
- `beta=0.0`：不使用 KL 约束，这是 DeepSeek-R1 等工作验证的主流实践

::: warning beta=0 的风险
不使用 KL 约束时，模型可能产生 **格式退化**（输出变得不可读）或 **reward hacking**。实践中需要配合格式奖励（format reward）来约束输出格式。
:::

### PPO 训练

PPO 是最复杂的对齐方法，需要 **4 个模型**同时在显存中：

```python
from trl import PPOConfig, PPOTrainer

# PPO 需要的 4 个模型：
# 1. policy model      — 被优化的策略
# 2. reference model   — 冻结的 SFT 模型（用于 KL 约束）
# 3. reward model      — 打分器
# 4. value model       — Critic，估计状态价值

config = PPOConfig(
    output_dir="ppo-model",
    kl_coef=0.05,           # KL 惩罚系数
    cliprange=0.2,          # PPO clip 范围
)

trainer = PPOTrainer(
    model=model,              # policy model
    reward_model=reward_model,
    value_model=value_model,
    args=config,
    train_dataset=dataset,
)
trainer.train()
```

::: tip 显存优化
PPO 的 4 个模型非常吃显存。常见做法：
- **LoRA**：policy 和 value model 只训练 adapter
- **模型共享**：policy 和 value model 共享 backbone，分别加不同的 head
- **DeepSpeed ZeRO**：多卡分片
:::

### 奖励函数设计模式

```python
# 模式 1：基于规则的奖励（适合数学/代码等有明确答案的任务）
def accuracy_reward(completions, ground_truth):
    """检查回答是否包含正确答案"""
    rewards = []
    for completion, answer in zip(completions, ground_truth):
        if answer in completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

# 模式 2：格式奖励（约束输出格式）
def format_reward(completions):
    """检查输出是否符合指定格式"""
    import re
    rewards = []
    for completion in completions:
        # 例：要求输出包含 <think>...</think><answer>...</answer>
        if re.search(r"<think>.*?</think>.*<answer>.*?</answer>", completion, re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

# 模式 3：多奖励组合
def combined_reward(completions, ground_truth):
    """组合多个奖励信号"""
    acc = accuracy_reward(completions, ground_truth)
    fmt = format_reward(completions)
    return [0.8 * a + 0.2 * f for a, f in zip(acc, fmt)]
```

### 补充：Rubric as Reward（RaR）—— RLVR 在不可验证领域的推广

> 本节是对前文「步骤 2：可验证奖励」的**补充**：当任务没有 math/code 那种 ground truth（如医学、科学问答、开放写作）时，如何继续保留"可验证"风味的稳定信号？

[Rubric as Reward (RaR, arXiv 2507.17746, Scale AI)](https://arxiv.org/abs/2507.17746) 给出的回答是：**为每条 prompt 生成一份 instance-specific rubric（评分清单）**，每条 rubric item 是一个二元判据 + 权重，由 LLM judge 打 0/1，再聚合成标量奖励驱动 GRPO（论文 Figure 1，page 2）。

**Problem Formulation（page 2 §2.1）**：每个 prompt $x$ 关联一组 $k$ 条 rubric $\{(w_j, c_j)\}_{j=1}^{k}$，其中 $w_j \in \mathbb{R}$ 是权重，$c_j: (x, \hat{y}) \mapsto \{0,1\}$ 是该条标准是否满足的二元判据。

**两种聚合方式**（page 3 公式 1、2）：

$$
r_{\text{explicit}}(x, \hat{y}) = \frac{\sum_{j=1}^{k} w_j \cdot c_j(x, \hat{y})}{\sum_{j=1}^{k} w_j} \quad (1)
$$

$$
r_{\text{implicit}}(x, \hat{y}) = f_\phi\bigl(x, \hat{y}, \{d_j\}_{j=1}^{k}\bigr) \quad (2)
$$

- **Explicit**：每条 rubric 单独喂 judge 拿 0/1，再加权平均（权重归一化使不同 prompt 间可比）。
- **Implicit**：把整张 rubric 列表作为上下文喂 judge，让它直接给出一个整体分，省去手调权重。

**Remark 1：RaR 包含 RLVR**（page 3 公式 3）。当 $k=1$、$w_1=1$、$c_1(x, \hat{y}) = \text{match}(y, \hat{y})$ 时，公式 (1) 退化为：

$$
r_{\text{RLVR}}(x, \hat{y}) = \text{match}(y, \hat{y})
$$

也就是说，**RLVR 是 RaR 在"单条 essential 判据 + 精确匹配"这一极端设定下的特例**——RaR 把"可验证奖励"从 math/code 推广到任何能写出 checklist 的领域。

**四条 Rubric 设计原则**（page 3-4 §3.1 Desiderata）：

| 原则 | 含义 |
|------|------|
| Grounded in Expert Guidance | 反映领域专家知识（关键事实、推理步骤、结论） |
| Comprehensive Coverage | 覆盖事实、逻辑、完备性、风格、安全等多维度，含 negative pitfalls |
| Criterion Importance | 不同标准重要性不同，需赋权（数值或类别标签） |
| Self-Contained Evaluation | 每条都可独立判断，无需外部上下文或专业背景 |

**四类标签 + 默认权重**（page 5 §4.4，RaR-Explicit 设定）：

```python
CATEGORY_WEIGHTS = {
    "Essential":  1.0,   # 必要事实，缺则直接错
    "Important":  0.7,   # 重要但非致命
    "Optional":   0.3,   # 加分项（风格、清晰度）
    "Pitfall":    0.9,   # 负向项，以正面陈述（"避免误诊"），未满足则扣分
}
```

> Pitfall 在原文以正面形式表达（如 "Response avoids misinformation"），满足则正向加分（page 5 脚注 3）。

**关键洞察：让小 judge 也能用**。论文核心 takeaway 之一（abstract + page 2 贡献 (iv)）：rubric 提供的结构化指引让小尺寸 judge 也能稳定对齐人类偏好——即"rubric-based rewards provide stable supervision across judge sizes, helping smaller models align effectively with human preferences"。最佳 RaR 变体在 HealthBench 上相对提升最高 31%、GPQA-Diamond 上 7%（abstract）。这意味着**用更便宜的 judge 也能拿到 大 judge 的奖励质量**，显著降低 on-policy RL 的奖励计算成本。

### 补充：RLVER 案例 —— 把"对话效果"操作化为可验证情感分

> 这是把"可验证奖励"思想搬到**多轮对话**这种非 math/code 领域的另一条路径，与 RaR 互补。

[RLVER (arXiv 2507.03112, 腾讯混元)](https://arxiv.org/abs/2507.03112) 的核心 trick：用一个**自洽的情感模拟用户**作为 RL 环境，把它在对话过程中的情感打分作为可验证奖励，端到端训练共情能力。Qwen2.5-7B 在 Sentient-Bench 上从 **13.3 → 79.2**（page 1 Abstract / page 2 末段）。

**Verifiable Emotion Reward（page 3 §2.1）**：构建在 SAGE（Sentient Agent as a Judge）框架之上。Sentient Agent 在每轮 LLM 回复后做两件事：

- $f_{\text{emo}}$：根据回复更新自身情感分数 $e_t \in [0, 100]$，并产出可解释的"内心想法"作为依据；
- $f_{\text{reply}}$：基于新情感、人设、目标，写出下一句用户回复延续对话。

终态情感分被归一化作为整段对话的奖励：

$$
r_\phi(x, y) = \frac{e_T}{100}, \quad \text{where } e_T = \mathcal{S}_{\text{emotion}}(h_T)
$$

其中 $h_T = \{x_0, y_0, x_1, \dots, y_T, x_T\}$ 为终止时的完整对话历史（page 3-4）。**关键点**：$e_t$ 由 user simulator 内部按规则**确定性提取**，而非用 learned RM——这是论文为缓解大规模 RL 中 reward hacking 设计的核心 trick（page 3 中段）。

**Heart-in-the-Loop 多轮闭环**（page 4 §2.2）：

```
seed s_i ──► [agent π_θ] y_t ──► [Sentient Agent S]
                ▲                       │
                │                       ├── e_t  （可验证情感分）
                │                       └── x_t  （用户下一句）
                └────── h_{t-1} ◄───────┘
```

每步 $t$ agent 观察历史 $h_{t-1}$ 采样 $y_t \sim \pi_\theta(\cdot \mid h_{t-1})$，simulator 输出 $(e_t, x_t)$。**若 $e_t \le 0$ 则提前终止，视为 social alignment failure**（page 4），最终 $e_T$ 作为 RL 奖励。

**Think-Then-Say 模板 + format reward**（page 5-6 §2.3）：每条回复必须先输出 `<think>...</think>` 段做内心规划，再写用户可见回复。**违反该格式的输出被 format reward 直接置零**（page 6 顶部："Outputs violating this syntactic specification are penalized with zero reward"）——和 R1 系的 format reward 同一思路。论文消融显示带 think 模板的模型更快收敛、表达更多样、更稳健地探索高共情策略。

**PPO vs GRPO 经验**（page 2 末段贡献 (iii) + page 4 末段）：在情感场景下作者得到的经验性结论是 **GRPO 更稳定、收益更平衡**（"GRPO consistently delivers stable and balanced improvements"），而 **PPO 偶尔能把特定能力推到更高上限**（"PPO can occasionally push the upper bounds of specific capabilities"）。这与一般推理任务上的偏好略有差异，提示算法选择需要结合任务方差与 reward 信号特性。

::: tip RaR 与 RLVER 的共同启示
两篇工作都是 **RLVR 哲学的延伸**：拒绝把开放任务直接交给 learned RM（容易 hacking），而是把"好回答"显式拆解成可机器验证的子信号——RaR 拆成 rubric checklist，RLVER 拆成 user simulator 的确定性情感分。**保留"可验证"骨架，是 reward hacking 时代下相对鲁棒的奖励设计范式**。
:::

## 对齐算法选择指南

| 场景 | 推荐算法 | 原因 |
|------|---------|------|
| 有成对偏好数据，追求简单 | **DPO** | 无需 RM，训练稳定 |
| 有成对数据，想一步完成 SFT + 对齐 | **ORPO** | 省掉 SFT 阶段，流程最简 |
| 数学/推理任务，可定义规则奖励 | **GRPO** | DeepSeek 验证，支持规则奖励 |
| 只有好/坏标签（非成对） | **KTO** | 不需要成对数据 |
| 追求最高质量，有充足资源 | **PPO** | 最灵活但最复杂 |
| 需要过程监督（数学/推理） | **PRM + RL** | 步级奖励更精确 |

```
选择决策流程：

有成对偏好数据？
├── 是 → 需要同时做 SFT？
│       ├── 是 → ORPO
│       └── 否 → 有在线生成能力？
│               ├── 是 → Online DPO / PPO
│               └── 否 → DPO
└── 否 → 有规则可定义的奖励？
        ├── 是 → GRPO
        └── 否 → 有好/坏标签？
                ├── 是 → KTO
                └── 否 → 先收集数据 😅
```

::: tip 实践建议
1. **起步用 DPO**：最简单，效果不差，适合快速迭代
2. **数学/代码用 GRPO**：规则奖励天然适合，DeepSeek-R1 已充分验证
3. **PPO 是上限最高的选择**，但工程复杂度也最高，建议有经验后再尝试
4. **关注 Online 方法**：Online DPO / GRPO 等在线方法通常优于离线版本
:::

## 苏格拉底时刻 💡

1. **DPO 真的不需要 Reward Model 吗？** DPO 隐含了一个奖励模型 $r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$，只是不需要显式训练。当偏好数据分布与模型能力差距较大时，DPO 可能不如 RLHF。
2. **GRPO 为什么在推理任务上特别好用？** 因为推理任务有明确的正确答案，可以用规则奖励（答案对错）取代人工标注，而开放式对话难以定义规则。
3. **KL 惩罚为什么必不可少？** 没有 KL 约束，模型会 reward hacking——找到奖励模型的漏洞，生成得分高但质量低的输出。
4. **全对/全错时 GRPO 的优势为 0，怎么办？** 需要调整采样温度和采样数量，确保组内有足够的多样性。如果某个难度的题目模型总是全对或全错，说明这个难度不适合当前阶段的 RL 训练。
5. **偏好对齐是否限制模型能力？** "对齐税"确实存在，但通过精心设计训练策略（如较小的 $\beta$、高质量数据）可以将其最小化。

## 常见问题 & 面试考点

- **Q: PPO 和 DPO 哪个效果更好？** 取决于场景。PPO 在 online 设定下更强（可以探索），DPO 在 offline 设定下更稳定。
- **Q: DPO 的 $\beta$ 如何选择？** $\beta$ 越大越保守（更接近 SFT 模型），$\beta$ 越小优化越激进。通常在 0.1~0.5 之间搜索。
- **Q: GRPO 和 PPO 的本质区别是什么？** GRPO 用组内相对排名代替了 Critic 网络来估计优势，省去了一半的模型。
- **Q: 为什么 KTO 不需要成对数据？** KTO 分别处理好/差回答，用 KL 散度作为"锚点"来衡量偏离程度，不需要直接比较两个回答。

## 推荐资源

### 论文

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT / RLHF 奠基论文
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — PPO 原始论文
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO 论文
- [A General Theoretical Paradigm to Understand Learning from Human Feedback](https://arxiv.org/abs/2310.12036) — IPO 论文
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) — KTO 论文
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2501.12948) — GRPO / DeepSeek-R1
- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) — ORPO 论文
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — OpenAI 的 PRM 代表作

### 博客与可视化

> 这两篇博客的具体结论已经被拆进 §Reward Model 训练、§DPO 损失的直觉、§ORPO 几个章节内的 `::: tip` 块，按位嵌入语义最相关的位置。下面只列出原文与可对照的 notebook，需要完整 ablation 数据/上下文时回到原文通读。

- [Fine-tune Llama 3 with ORPO](https://mlabonne.github.io/blog/posts/2024-04-19_Fine_tune_Llama_3_with_ORPO.html) by Maxime Labonne — Llama-3-8B + [orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k) 上的 QLoRA + ORPO 端到端教程，给出从 SFT 副作用 → ORPO 单阶段方案 → L4 单卡训练 → Nous benchmark 验证的完整路径。
- [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf) by Lambert et al. — HuggingFace 上的 RLHF 入门长文，按"预训练 LM → 训练 RM → PPO 微调 + KL 惩罚"三段式拆解，并对比 [TRL](https://github.com/huggingface/trl) / [TRLX](https://github.com/CarperAI/trlx) / [RL4LMs](https://github.com/allenai/RL4LMs) 三个主流开源框架的定位。
- [DPO 推导讲解](https://huggingface.co/blog/pref-tuning) — HuggingFace 偏好微调系列博客
- [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) — John Schulman 解释 K1/K2/K3 三种 KL 估计

### 代码参考

- [HuggingFace TRL · `PPOTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py) — Apache-2.0，PPO 全套：rollout / GAE / clip surrogate / value clipped loss
- [HuggingFace TRL · `DPOTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py) — DPO/IPO/Hinge 多种 loss type 切换
- [HuggingFace TRL · `GRPOTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py) — GRPO 组采样 + group-relative advantage + K3 KL
- [HuggingFace TRL · `KTOTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/kto_trainer.py) — 非成对数据 + 非对称权重
- [HuggingFace TRL · `ORPOTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/orpo_trainer.py) — SFT + 偏好对齐合并到单 stage，无需 reference model
- [CarperAI TRLX](https://github.com/CarperAI/trlx) — 大规模在线 PPO/ILQL，支持 20B+ 参数
- [AllenAI RL4LMs](https://github.com/allenai/RL4LMs) — NLP 任务库 + 多算法对照
- [HuggingFace open-r1 · GRPO](https://github.com/huggingface/open-r1) — 复现 DeepSeek-R1 的开源实现，`src/open_r1/grpo.py` 是 TRL `GRPOTrainer` 的最小可用入口，配合 `recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml` 和 `recipes/accelerate_configs/zero3.yaml` 可以一行命令在 8×H100 上跑通 GRPO（`vllm_mode=colocate` 时单节点即可启动），`src/open_r1/rewards.py` 给出可直接复用的 accuracy / format / cosine-length / repetition-penalty 等规则奖励组合，是把 alignment 训练落地到 R1-style 数学/代码任务的一手参考
