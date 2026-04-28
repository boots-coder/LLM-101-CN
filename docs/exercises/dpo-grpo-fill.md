---
title: "DPO/GRPO 填空"
description: "Level 2-3 填空：Bradley-Terry、DPO Loss、GRPO 优势函数"
topics: [fill-in, DPO, GRPO, Bradley-Terry, reward-model]
---
# DPO/GRPO 代码填空 (Level 2-3)

本练习覆盖偏好对齐训练的两大核心算法：DPO（Direct Preference Optimization）和 GRPO（Group Relative Policy Optimization）。从概念理解到完整实现，掌握 RLHF 的关键技术。

::: info 前置知识
- 强化学习基础（策略、奖励、KL 散度）
- Bradley-Terry 偏好模型
- PyTorch 基础（log_softmax、gather、sigmoid）
:::


::: tip 核心公式
**DPO Loss:**
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

**GRPO Loss:**
$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\min\left(\frac{\pi_\theta}{\pi_{\text{old}}} A, \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1-\epsilon, 1+\epsilon\right) A\right) - \beta \cdot D_{\text{KL}}\right]$$
:::


---

## 练习 1：Bradley-Terry 模型概念题（Level 1）

**题目 1**：在 Bradley-Terry 偏好模型中，给定两个回答 $y_w$（优选）和 $y_l$（劣选），人类偏好 $y_w$ 的概率建模为：

$$P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$$

其中 $\sigma$ 是 sigmoid 函数。如果 $r(y_w) = 2.0$，$r(y_l) = -1.0$，那么 $P(y_w \succ y_l)$ 最接近：

- A) 0.50
- B) 0.73
- C) 0.95
- D) 0.99

<details>
<summary>点击查看答案</summary>

**答案：C**

$\sigma(2.0 - (-1.0)) = \sigma(3.0) = \frac{1}{1 + e^{-3}} \approx 0.953$

Bradley-Terry 模型的核心思想是：两个选项被偏好的概率由它们 reward 之差的 sigmoid 决定。差距越大，偏好越确定。

</details>

---

**题目 2**：DPO 的核心创新在于：

- A) 使用一个单独训练的 Reward Model 来打分
- B) 将 reward 隐式表达为策略与参考策略的 log ratio，跳过了 Reward Model 的显式训练
- C) 使用 PPO 算法来优化策略
- D) 对 reward 进行 group 内归一化

<details>
<summary>点击查看答案</summary>

**答案：B**

DPO 的关键洞察是从 RLHF 的目标函数出发，推导出最优策略满足：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

将此关系代入 Bradley-Terry 模型，可以直接用策略网络的 log probability 替代 reward，从而跳过 Reward Model 的训练，将 RLHF 简化为一个分类问题。

</details>

---

**题目 3**：GRPO 与 PPO 的最大区别是：

- A) GRPO 使用了 clipping 机制
- B) GRPO 不需要 Critic（Value Function），而是用 group 内的 reward 归一化来估计 advantage
- C) GRPO 使用了 KL penalty
- D) GRPO 的学习率更大

<details>
<summary>点击查看答案</summary>

**答案：B**

GRPO 的核心创新是去掉了 PPO 中的 Critic 网络（价值函数）。传统 PPO 需要一个额外的 Value Function 来估计 advantage，而 GRPO 对同一个 prompt 采样一组回答（group），用 group 内 reward 的均值和标准差来归一化，得到 advantage 估计。这大幅减少了训练开销。

A 和 C 都是 GRPO 和 PPO 共有的特性，不是区别。

</details>

---

## 练习 2：DPO Loss 实现（Level 2）

实现 TRL `DPOTrainer` 风格的**序列级** DPO loss。输入是策略与参考模型对 chosen/rejected 序列分别求和后的 log probability。

> 实现风格参考：[TRL DPOTrainer.dpo_loss](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py)（sigmoid 损失类型，TRL 默认设置）。

```python
import torch
import torch.nn.functional as F


def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp,    ref_rejected_logp,
             beta=0.1):
    """
    序列级 DPO loss（sigmoid 损失类型）。

    参数（每个都是 [batch_size] 的标量序列级 log p）:
        policy_chosen_logp:    log π_θ(y_w | x)
        policy_rejected_logp:  log π_θ(y_l | x)
        ref_chosen_logp:       log π_ref(y_w | x)
        ref_rejected_logp:     log π_ref(y_l | x)
        beta:                  温度，典型 0.1
    返回:
        loss: 标量，batch 平均 DPO loss
    """
    # TODO 1: chosen 序列上的隐式 reward = π_θ 与 π_ref 的 log 差
    chosen_implicit_r = _____

    # TODO 2: rejected 序列上的隐式 reward
    rejected_implicit_r = _____

    # TODO 3: 用 beta 缩放 (chosen reward − rejected reward)，得到对比 margin
    margin = _____

    # TODO 4: 对 margin 取负 log-sigmoid，batch 内求平均
    loss = _____
    return loss


# ====== 测试 ======
# 直接给出已经在 completion 区间求和后的序列级 log p（数值就是 log π(y|x) 的真实量级）
policy_chosen_logp   = torch.tensor([-23.41,  -9.85, -17.62, -12.07])
policy_rejected_logp = torch.tensor([-25.30,  -8.71, -16.94, -13.85])
ref_chosen_logp      = torch.tensor([-22.96, -10.12, -17.40, -12.45])
ref_rejected_logp    = torch.tensor([-23.88,  -9.04, -16.71, -12.30])

loss = dpo_loss(policy_chosen_logp, policy_rejected_logp,
                ref_chosen_logp,    ref_rejected_logp, beta=0.1)
print(f"DPO Loss: {loss.item():.4f}")
```

::: details 提示
- chosen 隐式 reward = `policy_chosen_logp - ref_chosen_logp`
- rejected 隐式 reward = `policy_rejected_logp - ref_rejected_logp`
- margin = `beta * (chosen_implicit_r - rejected_implicit_r)`
- loss = `-F.logsigmoid(margin).mean()`，比 `-torch.log(torch.sigmoid(...))` 数值稳定
:::


<details>
<summary>点击查看答案</summary>

```python
# TODO 1
chosen_implicit_r = policy_chosen_logp - ref_chosen_logp

# TODO 2
rejected_implicit_r = policy_rejected_logp - ref_rejected_logp

# TODO 3
margin = beta * (chosen_implicit_r - rejected_implicit_r)

# TODO 4
loss = -F.logsigmoid(margin).mean()
```

**解析：**

DPO 把 RLHF 重写成一个"序列级二分类"问题，每条样本独立贡献一个标量 loss：

1. **隐式 reward**：DPO 论文证明，RLHF 最优策略下 $r(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \text{const}(x)$。所以 `policy_logp - ref_logp` 就是模型对 $(x,y)$ 的隐式打分，把 reward model 完全消掉。
2. **对比 margin**：把 chosen 与 rejected 的隐式 reward 作差再乘 $\beta$，得到 Bradley-Terry 模型的 logits。$\beta$ 越小、margin 越平、KL 约束越强。
3. **Sigmoid 损失**：$-\log\sigma(\text{margin})$ 把 "chosen 应当被偏好" 的概率最大化。margin 很正时 loss 趋近 0；margin 为负时 loss 显著增大并产生强梯度。
4. **数值稳定**：用 `F.logsigmoid` 而不是 `log(sigmoid(...))`，避免大负数下 sigmoid 下溢成 0 后再取 log 得到 −∞。

注意输入是**序列级** log probability —— 调用方需要先把 completion 区间内每个 token 的 log p 求和（prompt 区间不算），这一步通常封装在 `concatenated_forward` 这类工具函数里。

</details>

---

## 练习 3：GRPO Advantage 计算（Level 2）

GRPO 对同一 prompt 采一组（group）候选回答，用组内 reward 的 z-score 当 advantage，从而省掉 PPO 里的 critic 网络。

> 参考：[TRL GRPOTrainer 文档](https://huggingface.co/docs/trl/main/en/grpo_trainer)、DeepSeekMath / DeepSeek-R1 论文。

```python
import torch


def compute_grpo_advantages(group_rewards, eps=1e-4):
    """
    GRPO group-relative advantage（组内 z-score 归一化）。

    参数:
        group_rewards: 1D tensor 或 list，同一 prompt 下采到的 G 个 reward
                       reward 可以是连续值（RM 打分）或 0/1（规则可判）。
        eps:           防止全组 reward 相同时除零

    返回:
        advantages: tensor，与输入同形状

    公式:
        A_i = (r_i − mean(r_group)) / (std(r_group) + eps)
    """
    group_rewards = torch.as_tensor(group_rewards, dtype=torch.float)

    # TODO: 用组内均值与标准差对 reward 做 z-score 归一化
    advantages = _____
    return advantages


# ====== 测试用例 ======
# Case 1: 连续 reward（典型 RM 打分），advantage 应有正有负
A = compute_grpo_advantages([0.83, 0.41, 0.95, 0.27, 0.62, 0.09, 0.51, 0.78])
print(f"连续 reward (G=8): {A}")

# Case 2: 全组 reward 相同 -> advantage 全 ≈ 0，该组无训练信号
A = compute_grpo_advantages([0.5] * 8)
print(f"全相同: {A}")

# Case 3: 单个高分异常值，稀有正例 advantage 显著大于其他
A = compute_grpo_advantages([0.10, 0.12, 0.85, 0.08, 0.15, 0.11, 0.09, 0.13])
print(f"稀有正例 (G=8): {A}")

# Case 4: 数学题 pass/fail 风格，G=128 中 3 道做对
rewards_128 = torch.zeros(128)
rewards_128[[7, 53, 91]] = 1.0
A = compute_grpo_advantages(rewards_128)
print(f"3/128 通过, 通过样本 advantage = {A[7].item():.2f}")
print(f"3/128 通过, 失败样本 advantage = {A[0].item():.2f}")
```

::: details 提示
- 标准 z-score：`(x - x.mean()) / (x.std() + eps)`
- `group_rewards.mean()` / `group_rewards.std()` 直接调用即可
- `eps=1e-4` 与 [TRL `GRPOTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py) 上游一致（几乎同分组时防 advantage 爆炸）
- PyTorch `std` 默认 `unbiased=True`（贝塞尔修正），如要与 numpy 默认对齐用 `unbiased=False`
:::


<details>
<summary>点击查看答案</summary>

```python
advantages = (group_rewards - group_rewards.mean()) / (group_rewards.std() + eps)
```

**解析：**

GRPO advantage 本质就是组内 z-score 归一化，但放在 RLHF 框架下有几层语义：

1. **去 baseline**：减去组内均值等于把"整组都一样好"的部分梯度抵消，只有"高于平均"才被鼓励。这正好替代 PPO 里 critic 估计的 value baseline。
2. **方差归一化**：不同 prompt 下 reward 尺度可能差很大（简单题分数普遍高、难题普遍低），除以 std 之后所有 prompt 给优化器的梯度量级一致。
3. **eps 兜底**：同组 reward 完全一致时 std=0，加 eps 让 advantage 直接退化成 0，该 group 不产生梯度，等价于"跳过"。
4. **采样规模**：G 越大估计越准。DeepSeekMath / DeepSeek-R1 报告里典型 G=8~64，推理类任务也有 G=128 的设置。

观察测试输出：Case 1 连续 reward 下 advantage 有正有负；Case 3 中第 3 个稀有正例的 advantage 远高于其他；Case 4 中 3/128 通过率下，通过样本 advantage 是巨大正值，失败样本是小负值 —— 这就是稀疏奖励下 GRPO 仍然能学习的原因。

</details>

---

## 练习 4：GRPO Loss 完整实现（Level 3）

实现完整的 GRPO loss，包括 clipped policy ratio、advantage 加权和 KL penalty。

```python
import torch
import torch.nn.functional as F

def approx_kl(log_p, log_q):
    """
    计算近似 KL 散度（非对称）。
    
    公式: KL = exp(q/p) - (q - p) - 1
    即: q/p - log(q/p) - 1
    """
    return log_q.exp() / log_p.exp() \
           - (log_q - log_p) - 1


def grpo_loss(pi_logprob, pi_old_logprob, pi_ref_logprob,
              advantage, input_len):
    """
    计算 GRPO loss。
    
    参数:
        pi_logprob:     当前策略的 token log prob, [group_size, seq_len]
        pi_old_logprob: 采样时策略的 token log prob, [group_size, seq_len]
        pi_ref_logprob: 参考策略的 token log prob, [group_size, seq_len]
        advantage:      group-relative advantage, [group_size]
        input_len:      prompt 长度（只在回答部分计算 loss）
    
    返回:
        loss: 标量
    """
    epsilon = 0.2  # clipping 范围
    beta = 0.01    # KL penalty 系数

    bs, seq_len = pi_logprob.shape

    # TODO 1: 将 advantage 扩展维度以便广播 [group_size] -> [group_size, 1]
    advantage = _____

    # TODO 2: 计算 importance sampling ratio: exp(log_pi - log_pi_old)
    ratio = _____

    # TODO 3: clip ratio 到 [1-epsilon, 1+epsilon]
    ratio_clip = _____

    # TODO 4: 计算 clipped policy gradient（取 min，保守更新）
    policy_gradient = _____

    # TODO 5: 计算 KL penalty
    kl = _____

    # 只在回答部分（跳过 prompt）计算 loss
    response_loss = (policy_gradient - beta * kl)[:, input_len:]
    loss = -response_loss.mean()
    return loss


# ====== 测试代码 ======
torch.manual_seed(42)
group_size = 4
seq_len = 8
vocab_size = 50

# 模拟 logits
pi_logits = torch.randn(group_size, seq_len, vocab_size)
pi_old_logits = torch.randn(group_size, seq_len, vocab_size)
pi_ref_logits = torch.randn(group_size, seq_len, vocab_size)

# 模拟 token ids
token_ids = torch.randint(0, vocab_size, (group_size, seq_len))

# 计算 log prob
pi_logprob = torch.gather(
    F.log_softmax(pi_logits, dim=-1), dim=-1,
    index=token_ids.unsqueeze(-1)
).squeeze(-1)

pi_old_logprob = torch.gather(
    F.log_softmax(pi_old_logits, dim=-1), dim=-1,
    index=token_ids.unsqueeze(-1)
).squeeze(-1)

pi_ref_logprob = torch.gather(
    F.log_softmax(pi_ref_logits, dim=-1), dim=-1,
    index=token_ids.unsqueeze(-1)
).squeeze(-1)

# 模拟 advantage（沿用练习 3 的 z-score 函数）
advantage = compute_grpo_advantages([0.7, 0.2, 0.4, 0.85])
input_len = 3

loss = grpo_loss(pi_logprob, pi_old_logprob, pi_ref_logprob,
                 advantage, input_len)
print(f"GRPO Loss: {loss.item():.4f}")
```

::: details 提示
- `advantage.unsqueeze(dim=1)` 将形状从 `[G]` 变为 `[G, 1]`，便于与 `[G, seq_len]` 广播
- ratio = `torch.exp(pi_logprob - pi_old_logprob)`
- clipping: `torch.clamp(ratio, 1 - epsilon, 1 + epsilon)`
- policy gradient: `torch.minimum(ratio * advantage, ratio_clip * advantage)`
- KL: 直接调用 `approx_kl(pi_logprob, pi_ref_logprob)`
:::


<details>
<summary>点击查看答案</summary>

```python
# TODO 1: 扩展 advantage 维度
advantage = advantage.unsqueeze(dim=1)  # [G] -> [G, 1]

# TODO 2: importance sampling ratio
ratio = torch.exp(pi_logprob - pi_old_logprob)

# TODO 3: clip ratio
ratio_clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

# TODO 4: clipped policy gradient (PPO-style)
policy_gradient = torch.minimum(ratio * advantage, ratio_clip * advantage)

# TODO 5: KL penalty
kl = approx_kl(pi_logprob, pi_ref_logprob)
```

**解析：**

GRPO loss 由三个核心组件构成：

1. **Importance Sampling Ratio**：$\frac{\pi_\theta(y|x)}{\pi_{\text{old}}(y|x)}$。在 log 空间中计算差值再 exp，避免数值问题。ratio=1 表示策略没有变化。

2. **Clipping**：将 ratio 限制在 $[1-\epsilon, 1+\epsilon]$ 范围内，防止策略更新过大。取 `minimum` 确保保守更新 -- 当 advantage > 0 时，限制 ratio 不超过 $1+\epsilon$；当 advantage < 0 时，限制 ratio 不低于 $1-\epsilon$。

3. **KL Penalty**：$D_{\text{KL}}(\pi_{\text{ref}} || \pi_\theta) = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$。这是一个非对称 KL 散度，惩罚当前策略偏离参考策略过远，防止 reward hacking。

最终 loss = $-(\text{policy\_gradient} - \beta \cdot \text{KL})$，取负号是因为我们要最大化 policy gradient，最小化 KL。

</details>

---

## 练习 5：Reward Model 的 forward（Level 3）

实现一个简单的 Reward Model，它在预训练语言模型的基础上加一个 value head，输出标量 reward。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """
    Reward Model: 在语言模型基础上添加 value head，
    对每个序列输出一个标量 reward。
    
    结构:
        input -> backbone(LM) -> hidden_states -> value_head -> scalar reward
    
    训练目标:
        对于 (chosen, rejected) 对，最大化 r(chosen) - r(rejected) 的概率
    """
    def __init__(self, backbone, hidden_dim):
        super().__init__()
        # TODO 1: 保存 backbone 并冻结其参数
        self.backbone = _____
        for param in self.backbone.parameters():
            _____

        # TODO 2: 定义 value head (Linear: hidden_dim -> 1)
        self.value_head = _____

    def forward(self, input_ids):
        """
        参数:
            input_ids: [batch_size, seq_len]
        返回:
            rewards: [batch_size]，每个序列的标量 reward
        """
        # TODO 3: 获取 backbone 的 hidden states（取最后一层）
        with torch.no_grad():
            hidden_states = _____  # [bs, seq_len, hidden_dim]

        # TODO 4: 取最后一个 token 的 hidden state 作为序列表示
        last_hidden = _____  # [bs, hidden_dim]

        # TODO 5: 通过 value head 得到标量 reward
        reward = _____  # [bs, 1] -> [bs]
        return reward

    def compute_preference_loss(self, reward_chosen, reward_rejected):
        """
        Bradley-Terry 偏好 loss:
            loss = -log(sigmoid(r_chosen - r_rejected))
        
        参数:
            reward_chosen:   [batch_size]
            reward_rejected: [batch_size]
        返回:
            loss: 标量
        """
        # TODO 6: 计算 Bradley-Terry loss
        loss = _____
        return loss


# ====== 测试代码 ======
class SimpleBackbone(nn.Module):
    """模拟一个简单的语言模型 backbone"""
    def __init__(self, vocab_size, hidden_dim, seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        h = F.gelu(self.linear(h))
        return h

torch.manual_seed(42)
vocab_size, hidden_dim, seq_len = 100, 64, 10

backbone = SimpleBackbone(vocab_size, hidden_dim, seq_len)
reward_model = RewardModel(backbone, hidden_dim)

chosen_ids = torch.randint(0, vocab_size, (4, seq_len))
rejected_ids = torch.randint(0, vocab_size, (4, seq_len))

r_chosen = reward_model(chosen_ids)
r_rejected = reward_model(rejected_ids)

print(f"Chosen rewards: {r_chosen}")
print(f"Rejected rewards: {r_rejected}")

loss = reward_model.compute_preference_loss(r_chosen, r_rejected)
print(f"Preference loss: {loss.item():.4f}")

# 验证 value_head 可训练，backbone 冻结
trainable = sum(p.numel() for p in reward_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in reward_model.parameters())
print(f"可训练参数: {trainable}, 总参数: {total}")
```

::: details 提示
- backbone 赋值后用 `param.requires_grad_(False)` 冻结
- value_head: `nn.Linear(hidden_dim, 1)`
- hidden_states: `self.backbone(input_ids)`
- 最后一个 token: `hidden_states[:, -1, :]`
- reward: `self.value_head(last_hidden).squeeze(-1)` 去掉最后一维
- preference loss: `-F.logsigmoid(reward_chosen - reward_rejected).mean()`
:::


<details>
<summary>点击查看答案</summary>

```python
class RewardModel(nn.Module):
    def __init__(self, backbone, hidden_dim):
        super().__init__()
        # TODO 1
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        # TODO 2
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        # TODO 3
        with torch.no_grad():
            hidden_states = self.backbone(input_ids)

        # TODO 4
        last_hidden = hidden_states[:, -1, :]

        # TODO 5
        reward = self.value_head(last_hidden).squeeze(-1)
        return reward

    def compute_preference_loss(self, reward_chosen, reward_rejected):
        # TODO 6
        loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
        return loss
```

**解析：**

Reward Model 的设计要点：

1. **Backbone 冻结**：Reward Model 通常基于预训练 LM 初始化，训练时只更新 value head，保留语言理解能力。
2. **序列表示**：取最后一个 token 的 hidden state 作为整个序列的表示。在 causal LM 中，最后一个 token 经过了所有上文的注意力聚合，信息最丰富。
3. **Value Head**：一个简单的 Linear 层，将 hidden_dim 维映射到标量 reward。
4. **Bradley-Terry Loss**：$-\log\sigma(r_w - r_l)$，直接优化"chosen 的 reward 高于 rejected"的概率。这与 DPO 中的隐式 reward 对应：$r(y|x) = \beta \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$。

在实际训练中，Reward Model 的质量直接决定了 RLHF 的上限。常见问题包括 reward hacking（策略找到 RM 的漏洞而非真正改善质量）和 overoptimization。DPO 绕过了显式 RM 的训练，但 GRPO 仍然需要 reward function（可以是规则或 RM）。

</details>

---

## 练习 6：端到端 toy GRPO 训练（Level 3）

把前几题的零件拼起来，跑一个真能收敛的最小 GRPO。任务：在 10 token 词表上，让 policy 学会"从 `<bos>` 出发只输出 `a`"。reward 由规则定义，模型用 `n_layer=1, n_embd=16` 的 tiny GPT-2，CPU 上 30 秒收敛。

```python
import copy, torch, torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

torch.manual_seed(0)
VOCAB = list("abcdefghi") + ["<bos>"]
V = len(VOCAB)
stoi = {c: i for i, c in enumerate(VOCAB)}
TARGET = stoi["a"]
prompt_id = torch.tensor([[stoi["<bos>"]]])

cfg = GPT2Config(vocab_size=V, n_positions=4,
                 n_embd=16, n_layer=1, n_head=2)
policy = GPT2LMHeadModel(cfg)
ref = copy.deepcopy(policy).eval()
old = copy.deepcopy(policy).eval()
for p in ref.parameters(): p.requires_grad_(False)

def reward_fn(token_id: int) -> float:
    return 1.0 if token_id == TARGET else 0.0


def sample_group(model, G=32):
    """从 prompt 出发，用 old policy 各自采 1 个 token，返回 (samples, old_lp, rewards)."""
    with torch.no_grad():
        # TODO 1: 取 prompt 最后一个位置的 logits, [1, V]
        logits = _____
        probs = logits.softmax(-1)
        # TODO 2: 从 probs 多次采样 G 个 token, 形状 [G]
        samples = _____
        # TODO 3: 计算这些 token 在 old policy 下的 log prob, [G]
        old_lp = _____
    rewards = torch.tensor([reward_fn(t.item()) for t in samples])
    return samples, old_lp, rewards


def grpo_advantage(rewards):
    """组内 z-score 归一化（GRPO 的核心）。"""
    # TODO 4: 用 mean / (std + 1e-8) 做 z-score
    return _____


def grpo_step(samples, old_lp, advantages, clip=0.2, kl_coef=0.04):
    """clipped surrogate + K3 KL penalty。"""
    logits = policy(prompt_id).logits[0, -1]      # [V]
    new_lp_full = F.log_softmax(logits, dim=-1)   # [V]
    new_lp = new_lp_full[samples]                 # [G]

    with torch.no_grad():
        ref_lp_full = F.log_softmax(
            ref(prompt_id).logits[0, -1], dim=-1)
    ref_lp = ref_lp_full[samples]                 # [G]

    # TODO 5: importance ratio = exp(new_lp - old_lp)
    ratio = _____
    # TODO 6: clipped surrogate（注意 PG 是最大化，loss 取负号）
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
    pg = _____  # 提示：min(unclipped, clipped)

    # TODO 7: K3 KL: exp(log_r) - log_r - 1，其中 log_r = ref_lp - new_lp
    log_r = ref_lp - new_lp
    kl = _____

    return -(pg - kl_coef * kl).mean()


# ====== 训练循环 ======
opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
for it in range(40):
    samples, old_lp, rewards = sample_group(old, G=32)
    adv = grpo_advantage(rewards)
    for _ in range(2):  # inner update
        loss = grpo_step(samples, old_lp, adv)
        opt.zero_grad(); loss.backward(); opt.step()
    # TODO 8: 把更新后的 policy 同步到 old（GRPO 的 on-policy 边界）
    _____
    if it % 10 == 0:
        with torch.no_grad():
            p_a = policy(prompt_id).logits[0, -1].softmax(-1)[TARGET]
        print(f"iter {it:>2}  loss={loss.item():+.3f}  P(a)={p_a:.3f}")
```

::: details 提示
- TODO 1：`old(prompt_id).logits[0, -1]`，形状 `[V]`
- TODO 2：`torch.multinomial(probs, num_samples=G, replacement=True).squeeze(0)`
- TODO 3：`F.log_softmax(logits, dim=-1)[samples]`
- TODO 4：`(rewards - rewards.mean()) / (rewards.std() + 1e-8)`
- TODO 5：`(new_lp - old_lp).exp()`
- TODO 6：`torch.minimum(unclipped, clipped)`
- TODO 7：`log_r.exp() - log_r - 1`（Schulman K3 估计，无偏且非负）
- TODO 8：`old.load_state_dict(policy.state_dict())`
:::


<details>
<summary>点击查看答案</summary>

```python
# TODO 1: prompt 最后一位的 logits
logits = old(prompt_id).logits[0, -1]                            # [V]

# TODO 2: 多次采样
samples = torch.multinomial(probs, num_samples=G,
                            replacement=True).squeeze(0)         # [G]

# TODO 3: old policy 下的 log prob
old_lp = F.log_softmax(logits, dim=-1)[samples]                  # [G]

# TODO 4: 组内 z-score
return (rewards - rewards.mean()) / (rewards.std() + 1e-8)

# TODO 5: importance ratio
ratio = (new_lp - old_lp).exp()

# TODO 6: clipped surrogate
pg = torch.minimum(unclipped, clipped)

# TODO 7: K3 KL（log_r = ref_lp - new_lp）
kl = log_r.exp() - log_r - 1

# TODO 8: 把 policy 拷贝给 old
old.load_state_dict(policy.state_dict())
```

**解析：**

这道题把 GRPO 的所有零件拼起来端到端跑通，是练习 3 / 4 的"组装版"：

1. **TODO 1-3 — `sample_group`**：GRPO 的"采样阶段"。用 `old policy` 在 `<bos>` 后做 G 次独立采样组成一组，同时记录采样时的 log prob 作为 `old_lp`。注意 `@torch.no_grad`：采样不参与梯度。`old_lp` 在后续被当成"分母"（重要性采样的旧分布），所以它必须来自 old，而不是当前 policy。

2. **TODO 4 — `grpo_advantage`**：GRPO 区别于 PPO 的核心 -- **没有 value model**，advantage 直接用组内 reward 的 z-score。这等价于"比组里平均好多少"。`std + 1e-8` 防止全组 reward 相同时除零。

3. **TODO 5-6 — clipped surrogate**：和练习 4 一致，`min(ratio * A, clip(ratio) * A)` 同时处理 A>0 和 A<0 两种情况，防止策略一步更新过大。

4. **TODO 7 — K3 KL**：Schulman 提出的 unbiased、非负的 KL 估计：$\hat k_3 = e^{\log r} - \log r - 1$，其中 $\log r = \log\pi_{\text{ref}} - \log\pi_\theta$。它比 $-\log r$（K1）方差更小，比 $\frac{1}{2}(\log r)^2$（K2）有更好的非负性保证。

5. **TODO 8 — `old ← policy`**：GRPO 是 on-policy 算法，每个外层迭代结束都要把 `old` 同步到最新的 `policy`，否则 ratio 会越偏越离谱、clip 会一直生效、训练崩溃。这是练习里最容易漏掉的一行。

跑起来你应该看到 `P(a)` 从 0.10（均匀）逐步升到 0.70+，loss 振荡但整体下降。这就是 minimal-GRPO 的全部精髓。

</details>

---

## MLM 代码训练模式

完成上面的固定填空后，试试随机挖空模式 -- 每次点击「刷新」会随机遮盖不同的代码片段，帮你彻底记住每一行。

### DPO Loss 计算

<CodeMasker title="DPO Loss：序列级隐式 reward 与 sigmoid 对比" :mask-ratio="0.15">
def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp,    ref_rejected_logp,
             beta=0.1):
    # 隐式 reward = π_θ 与 π_ref 的 log 差
    chosen_implicit_r   = policy_chosen_logp   - ref_chosen_logp
    rejected_implicit_r = policy_rejected_logp - ref_rejected_logp

    # 对比 margin = beta * (chosen − rejected)
    margin = beta * (chosen_implicit_r - rejected_implicit_r)

    # 负 log-sigmoid 损失，batch 内求平均
    loss = -F.logsigmoid(margin).mean()
    return loss
</CodeMasker>

### GRPO Advantage 计算

<CodeMasker title="GRPO：Group Relative Advantage（z-score 归一化）" :mask-ratio="0.15">
def compute_grpo_advantages(group_rewards, eps=1e-4):
    group_rewards = torch.as_tensor(group_rewards, dtype=torch.float)
    # 组内 z-score：减均值、除标准差，eps 兜住全相同的退化情况
    advantages = (group_rewards - group_rewards.mean()) / (group_rewards.std() + eps)
    return advantages
</CodeMasker>

### GRPO Policy Loss（含 KL Penalty）

<CodeMasker title="GRPO Loss：clipped ratio + advantage + KL penalty" :mask-ratio="0.15">
def grpo_loss(pi_logprob, pi_old_logprob, pi_ref_logprob,
              advantage, input_len):
    epsilon = 0.2
    beta = 0.01

    # 扩展 advantage 维度以便广播
    advantage = advantage.unsqueeze(dim=1)

    # importance sampling ratio
    ratio = torch.exp(pi_logprob - pi_old_logprob)

    # clip ratio 到 [1-epsilon, 1+epsilon]
    ratio_clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    # clipped policy gradient（取 min，保守更新）
    policy_gradient = torch.minimum(ratio * advantage, ratio_clip * advantage)

    # KL penalty
    kl = approx_kl(pi_logprob, pi_ref_logprob)

    # 只在回答部分计算 loss
    response_loss = (policy_gradient - beta * kl)[:, input_len:]
    loss = -response_loss.mean()
    return loss
</CodeMasker>
