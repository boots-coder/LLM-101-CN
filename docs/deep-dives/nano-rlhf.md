---
title: "深度剖析 RLHF Pipeline"
description: "纯 PyTorch 实现完整 RLHF pipeline：Bradley-Terry → Reward Model → PPO → DPO → GRPO"
topics: [RLHF, PPO, DPO, GRPO, reward-model, Bradley-Terry, GAE, from-scratch, alignment]
prereqs: [training/alignment, deep-dives/nano-gpt]
---
# 深度剖析 RLHF Pipeline

> **一句话总结:** 不依赖 TRL 库，用纯 PyTorch 从零实现完整 RLHF pipeline（Reward Model → PPO → DPO → GRPO），200 行核心代码搞懂偏好对齐的每一个梯度。

## 为什么要从零实现 RLHF

::: tip 调库 vs 从零实现
[alignment 章节](/training/alignment) 用 TRL 库快速跑通了 RLHF/DPO/GRPO 的完整流程。但 TRL 封装了太多细节——你知道 PPO 的 loss 由哪几部分组成吗？GAE 是怎么从后往前递推的？DPO 为什么能绕过 Reward Model？

自己实现一遍，这些问题自然就有答案了。
:::

**本文的目标：**

| 模块 | 核心代码量 | 你会搞懂的问题 |
|------|-----------|---------------|
| Bradley-Terry 模型 | ~20 行 | 偏好建模的数学基础 |
| PPO 核心 | ~80 行 | GAE、Clipped Loss、KL 惩罚 |
| DPO 核心 | ~30 行 | 为什么不需要 Reward Model |
| GRPO 核心 | ~40 行 | 组内相对优势、无需 Value Network |

**前置知识：** 建议先读完 [偏好对齐](/training/alignment) 了解算法原理，本文聚焦代码实现。

## Bradley-Terry 偏好模型

### 数学回顾

Bradley-Terry (BT) 模型是 RLHF 的数学基础。核心思想：**通过成对比较来估计个体的潜在分数。**

给定两个选项 $i, j$，$i$ 胜过 $j$ 的概率为：

$$
P(i \succ j) = \frac{\exp(s_i)}{\exp(s_i) + \exp(s_j)} = \sigma(s_i - s_j)
$$

这里 $s_i$ 是选项 $i$ 的潜在分数（即 reward），$\sigma$ 是 sigmoid 函数。

### 从零实现

我们用 PyTorch 手写一个最小的 BT 模型。思路：给每个选项分配一个可学习分数，
用 sigmoid(分数差) 预测胜率，再用 BCE 损失反向传播。

```python
import torch
import torch.nn as nn

class PairwisePreferenceModel(nn.Module):
    """最小 BT 实现：每个候选项一个标量分数，成对比较产生偏好概率"""
    def __init__(self, num_candidates: int):
        super().__init__()
        self.scores = nn.Parameter(torch.zeros(num_candidates))

    def predict_preference(self, winner_idx: int, loser_idx: int) -> torch.Tensor:
        """返回 P(winner > loser) = σ(s_w − s_l)"""
        return torch.sigmoid(self.scores[winner_idx] - self.scores[loser_idx])

# ---------- 构造比赛数据 ----------
# 5 支队伍的循环赛片段（编号 0-4）
matches = [
    (0, 2),  # 队伍 0 胜 队伍 2
    (1, 4),  # 队伍 1 胜 队伍 4
    (0, 3),  # 队伍 0 胜 队伍 3
    (2, 4),  # 队伍 2 胜 队伍 4
    (3, 1),  # 队伍 3 胜 队伍 1
]

model = PairwisePreferenceModel(num_candidates=5)
opt = torch.optim.Adam(model.parameters(), lr=0.05)
bce = nn.BCELoss()

for step in range(200):
    epoch_loss = 0.0
    for w, l in matches:
        opt.zero_grad()
        prob = model.predict_preference(w, l)
        loss = bce(prob, torch.ones(1))  # 标签恒为 1（w 确实胜了）
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    if step % 50 == 0:
        print(f"step {step:>3d}  loss={epoch_loss:.4f}")

# 查看学到的排名
ranking = model.scores.data
team_order = ranking.argsort(descending=True)
print(f"\n学到的分数: {ranking.tolist()}")
print(f"排名（强→弱）: {team_order.tolist()}")
# 队伍 0 分数最高（两胜零负），队伍 4 最低（零胜两负）
```

::: details 为什么 BT 模型对 RLHF 如此重要？
BT 模型把"人类偏好"转化为"分数差的 sigmoid"。这意味着：
1. **Reward Model** 的训练目标就是 BT 损失：$-\log \sigma(r(x, y_w) - r(x, y_l))$
2. **DPO** 的核心推导也基于 BT 假设，只是用策略的对数概率比替代了显式 reward
3. 理解了 BT，就理解了整个偏好对齐的数学框架
:::

## PPO 从零实现

PPO (Proximal Policy Optimization) 是 RLHF 中最经典的强化学习算法。我们分四步实现。

### 第一步：PPO 需要哪些模型

```python
from transformers import LlamaForCausalLM, LlamaConfig

class ValueHead(torch.nn.Module):
    """Value Network：在语言模型基础上加一个标量输出头"""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        hidden_dim = backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens):
        out = self.backbone(**tokens, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]          # [B, T, D]
        return self.head(last_hidden).squeeze(-1)    # [B, T]

class RLHFModelBundle:
    """PPO 四件套"""
    def __init__(self, policy, reference, reward_fn, value_net):
        self.policy = policy        # 策略模型（要训练的）
        self.reference = reference  # 参考模型（冻结，计算 KL）
        self.reward_fn = reward_fn  # 奖励模型（冻结，打分）
        self.value_net = value_net  # 价值模型（要训练的）
```

::: warning 四个模型的关系
```
SFT 模型 ──→ policy（可训练）      用于生成 + 策略更新
         ├──→ reference（冻结）   用于计算 KL 散度
         └──→ reward_fn（冻结）   用于给生成结果打分
RM 模型  ──→ value_net（可训练）  用于估计状态价值 V(s)
```
这就是为什么 PPO-RLHF 需要大量显存——至少要加载 4 个模型。
:::

### 第二步：GAE 优势估计

**GAE (Generalized Advantage Estimation)** 是 PPO 的核心创新之一，用于平衡偏差和方差。

$$
\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

```python
def generalized_advantage(rewards, values, mask, discount=0.99, gae_lambda=0.95):
    """
    GAE 优势估计（先算 TD 残差，再反向累积）

    Args:
        rewards:    [B, T] — KL 惩罚后的 token 级奖励
        values:     [B, T] — Value Network 的估计
        mask:       [B, T] — attention mask
        discount:   折扣因子
        gae_lambda: GAE λ 参数（0=高偏差低方差, 1=低偏差高方差）
    """
    v = values * mask
    r = rewards * mask
    T = r.size(1)

    # 1) 先计算每步 TD 残差 δ_t = r_t + γ·V_{t+1} − V_t
    next_v = torch.cat([v[:, 1:], torch.zeros_like(v[:, :1])], dim=1)
    td_err = r + discount * next_v - v                      # [B, T]

    # 2) 从末尾向前累积  A_t = δ_t + γλ·A_{t+1}
    adv = torch.zeros_like(td_err)
    running = torch.zeros(r.size(0), device=r.device)
    for step in range(T - 1, -1, -1):
        running = td_err[:, step] + discount * gae_lambda * running
        adv[:, step] = running

    return adv
```

::: tip 直觉理解 GAE
- 当 $\lambda = 0$ 时，$\hat{A}_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$（单步 TD，高偏差）
- 当 $\lambda = 1$ 时，$\hat{A}_t = \sum_{l=0}^{T-t} \gamma^l r_{t+l} - V(s_t)$（Monte Carlo，高方差）
- $\lambda \in (0,1)$ 在两者之间权衡，常用 0.95
:::

### 第三步：KL 惩罚 + Reward 计算

PPO-RLHF 的奖励不只是 RM 的打分，还要减去 KL 惩罚，防止策略偏离参考模型太远：

$$
r_t = -\beta \cdot \text{KL}_t, \quad r_T = -\beta \cdot \text{KL}_T + r_{\text{RM}}
$$

```python
def apply_kl_penalty(rm_scores, ref_lp, policy_lp, kl_weight=0.01):
    """
    将 RM 分数 + KL 惩罚合并为 token 级奖励

    关键：KL 惩罚分布在每个 token 上，RM 分数只加在最后一个 token
    """
    # token 级 KL：log π_policy(a|s) − log π_ref(a|s)
    per_token_kl = policy_lp - ref_lp
    shaped_reward = -kl_weight * per_token_kl

    # RM 分数只加在序列末尾
    shaped_reward[:, -1] += rm_scores[:, 0]

    return shaped_reward


def token_log_probs(logits, token_ids):
    """从 logits 提取对应 token 的 log probability"""
    log_p = torch.nn.functional.log_softmax(logits, dim=-1)
    selected = log_p.gather(dim=2, index=token_ids.unsqueeze(-1)).squeeze(-1)
    return selected, log_p
```

### 第四步：PPO 三大 Loss

这是 PPO 算法的核心——三个损失函数的组合。

```python
def mean_with_mask(tensor, mask, dim=None):
    """只对 mask=1 的位置求均值"""
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)

# ─── 1. Policy Loss（Clipped Surrogate Objective）───
def ppo_surrogate_loss(new_lp, old_lp, adv, mask, clip_eps=0.2):
    """
    L_CLIP = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]

    其中 r_t = π_θ(a|s) / π_θ_old(a|s) = exp(logπ_new - logπ_old)
    """
    ratio = torch.exp(new_lp - old_lp)
    surr1 = -adv * ratio
    surr2 = -adv * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    return mean_with_mask(torch.max(surr1, surr2), mask)

# ─── 2. Value Loss（Clipped Value Loss）───
def clipped_value_loss(adv, v_pred, v_old, mask, clip_eps=0.2):
    """
    L_VF = 0.5 * E[max((V-V_target)^2, (V_clip-V_target)^2)]

    V_target = advantages + V_old（GAE + 旧的价值估计 = 回报估计）
    """
    v_target = adv + v_old
    v_clipped = torch.clamp(v_pred, v_old - clip_eps, v_old + clip_eps)
    loss_raw = (v_pred - v_target) ** 2
    loss_clipped = (v_clipped - v_target) ** 2
    return 0.5 * mean_with_mask(torch.max(loss_raw, loss_clipped), mask)

# ─── 3. 组合损失 ───
def combined_ppo_loss(batch, cfg):
    """PPO 总损失 = Policy Loss + c1 * Value Loss"""
    adv = generalized_advantage(
        batch['shaped_rewards'], batch['values'],
        batch['mask'], cfg.discount, cfg.gae_lambda
    )
    l_policy = ppo_surrogate_loss(
        batch['new_lp'], batch['old_lp'],
        adv, batch['mask'], cfg.clip_eps
    )
    l_value = clipped_value_loss(
        adv, batch['v_pred'], batch['v_old'],
        batch['mask'], cfg.clip_eps
    )
    return l_policy + cfg.value_coef * l_value, l_policy, l_value
```

::: details PPO Clipped Loss 的直觉
为什么要 clip？考虑两种情况：
- **优势 A > 0**（好动作）：我们想增大 ratio，但 clip 限制了最大增幅为 $1+\epsilon$
- **优势 A < 0**（坏动作）：我们想减小 ratio，但 clip 限制了最小减幅为 $1-\epsilon$

效果：每次更新只允许策略"小步移动"，避免一次更新太多导致策略崩溃。
:::

## RLHF 完整 Pipeline

把上面的模块串联起来，就是完整的 PPO-RLHF 训练流程：

```python
class TrainingArgs:
    def __init__(self):
        self.inner_epochs = 3      # 每批数据重复训练的轮数
        self.micro_bs = 1
        self.num_rounds = 2
        self.kl_weight = 0.01      # KL 惩罚系数
        self.value_coef = 0.01     # Value Loss 权重
        self.gae_lambda = 0.95     # GAE λ
        self.discount = 0.99       # 折扣因子
        self.clip_eps = 0.2        # Policy / Value clip 范围


def run_rlhf(bundle, args, prompts, gen_len=64):
    """完整 RLHF 训练循环"""
    for rnd in range(args.num_rounds):
        # ── Phase 1: 生成 ──
        with torch.no_grad():
            responses = bundle.policy.generate(prompts, max_new_tokens=gen_len)
            rm_scores = bundle.reward_fn(responses)          # RM 打分

        # ── Phase 2: 收集经验 ──
        with torch.no_grad():
            ref_lp, _ = token_log_probs(
                bundle.reference(responses).logits, responses
            )
        policy_lp, _ = token_log_probs(
            bundle.policy(responses).logits, responses
        )
        v_old = bundle.value_net(responses)
        shaped_rewards = apply_kl_penalty(
            rm_scores, ref_lp, policy_lp, args.kl_weight
        )

        # ── Phase 3: PPO 更新 ──
        batch = {
            'old_lp': policy_lp.detach(),
            'v_old': v_old.detach(),
            'shaped_rewards': shaped_rewards,
            'mask': (responses != pad_token_id).float(),
            'tokens': responses,
        }
        for _ in range(args.inner_epochs):
            # 重新前向传播获取当前策略的 logprobs 和 values
            new_lp, _ = token_log_probs(
                bundle.policy(batch['tokens']).logits, batch['tokens']
            )
            v_pred = bundle.value_net(batch['tokens'])
            batch['new_lp'] = new_lp
            batch['v_pred'] = v_pred

            loss, l_pol, l_val = combined_ppo_loss(batch, args)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

::: warning PPO-RLHF 的训练流程总结
```
循环开始
  │
  ├─ 1. policy 生成回答            → responses
  ├─ 2. reward_fn 给回答打分       → rm_scores
  ├─ 3. reference 算参考 logprobs  → ref_lp
  ├─ 4. 计算 KL 惩罚后的奖励       → shaped_rewards
  ├─ 5. value_net 估计状态价值     → v_old
  │
  └─ PPO 更新（重复 K 次）
       ├─ 重新前向传播 policy + value_net
       ├─ 计算 GAE 优势估计
       ├─ 计算 Policy Loss（clipped）
       ├─ 计算 Value Loss（clipped）
       └─ 反向传播更新 policy + value_net
循环结束
```
:::

## DPO 从零实现

DPO (Direct Preference Optimization) 的核心洞察：**可以跳过 Reward Model，直接从偏好数据优化策略。**

### DPO Loss 推导直觉

从 RLHF 的最优策略解出发：

$$
r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

代入 BT 模型，$Z(x)$ 消掉，得到 DPO 损失：

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right]\right)
$$

### 核心实现

> DPO 实现参考了 Rafailov et al. (2023) 原始论文的损失函数推导。

```python
import torch.nn.functional as F

def gather_log_probs(logits, ids):
    """从 logits 提取每个 token 的 log probability"""
    log_p = F.log_softmax(logits, dim=-1)           # [B, T, V]
    return log_p[torch.arange(log_p.size(0)).unsqueeze(1),
                 torch.arange(log_p.size(1)).unsqueeze(0),
                 ids]                                # [B, T]


def direct_preference_loss(policy, ref, chosen_tok, rejected_tok, beta=0.1):
    """
    DPO 损失函数：不需要 Reward Model！

    Args:
        policy:       当前策略 π_θ
        ref:          参考策略 π_ref（冻结）
        chosen_tok:   chosen 样本的 token ids
        rejected_tok: rejected 样本的 token ids
        beta:         温度参数（控制偏离 ref 的程度）
    """
    # 1. 前向传播，获取四组 logits
    with torch.no_grad():
        ref_w_logits = ref(**chosen_tok).logits
        ref_l_logits = ref(**rejected_tok).logits
    pol_w_logits = policy(**chosen_tok).logits
    pol_l_logits = policy(**rejected_tok).logits

    # 2. 提取 log probabilities
    lp_w     = gather_log_probs(pol_w_logits,   chosen_tok['input_ids'])
    lp_l     = gather_log_probs(pol_l_logits,   rejected_tok['input_ids'])
    lp_w_ref = gather_log_probs(ref_w_logits,   chosen_tok['input_ids'])
    lp_l_ref = gather_log_probs(ref_l_logits,   rejected_tok['input_ids'])

    # 3. 计算 log ratio
    policy_ratio = lp_w - lp_l              # π_θ 的偏好
    ref_ratio    = lp_w_ref - lp_l_ref      # π_ref 的偏好
    margin       = policy_ratio - ref_ratio  # 相对偏好变化

    # 4. DPO Loss = -log σ(β * margin)
    return -F.logsigmoid(beta * margin).mean()


# ─── IPO 变体：只需改一行 ───
def ipo_loss(margin, beta=0.1):
    """IPO Loss = (margin - 1/(2β))^2"""
    target = 1.0 / (2.0 * beta)
    return (margin - target).square().mean()
```

::: tip DPO vs PPO：一张表说清
| 对比项 | PPO | DPO |
|--------|-----|-----|
| 需要 Reward Model | 是 | **否** |
| 需要 Value Network | 是 | **否** |
| 需要在线采样 | 是（actor 生成） | **否**（离线数据） |
| 模型数量 | 4 个 | **2 个**（policy + ref） |
| 训练稳定性 | 需要仔细调参 | **更稳定** |
| 核心代码量 | ~80 行 | **~30 行** |
| 表达能力 | 更强（在线探索） | 受限于离线数据质量 |
:::

## GRPO 从零实现

GRPO (Group Relative Policy Optimization) 是 DeepSeek 提出的方案，核心创新：**用组内相对排名替代 Value Network。**

### GRPO 的三个关键组件

#### 1. 组内优势估计（替代 GAE）

对同一个 prompt 采样 $G$ 个回答，用组内 reward 的均值和标准差归一化：

$$
\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r}) + \epsilon}
$$

```python
def standardize_rewards(raw_rewards, eps=1e-4):
    """
    GRPO 优势估计：组内 z-score 归一化（不需要 Value Network）。
    eps 与 TRL GRPOTrainer 上游一致（几乎同分组时防 advantage 爆炸）。
    """
    r = torch.as_tensor(raw_rewards, dtype=torch.float)
    mu, sigma = r.mean(), r.std()
    return (r - mu) / (sigma + eps)

# 示例：连续 RM 打分，G=8
raw_rewards = [0.83, 0.41, 0.95, 0.27, 0.62, 0.09, 0.51, 0.78]
print(standardize_rewards(raw_rewards))
# 高分样本 advantage > 0（鼓励），低分样本 < 0（抑制）
```

::: details GRPO 优势估计的关键洞察
1. **全对或全错时 advantage ≈ 0**：无法学到东西，这种 batch 可以跳过
2. **正例越少，其 advantage 越大**：稀缺正例获得更大梯度信号
3. **采样越多，估计越准确**：DeepSeek-R1 用 G=64 个采样
:::

#### 2. KL 散度（约束策略偏离）

GRPO 使用的 KL 散度形式与 PPO 略有不同：

$$
D_{\text{KL}}[\pi_\theta || \pi_{\text{ref}}] = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log \frac{\pi_{\text{ref}}}{\pi_\theta} - 1
$$

```python
def approx_kl_divergence(log_p, log_q):
    """GRPO 的 KL 散度（非对称形式）: D_KL[π||ref] ≈ exp(log_q - log_p) - (log_q - log_p) - 1"""
    diff = log_q - log_p              # log(π_ref / π_θ)
    return torch.exp(diff) - diff - 1
```

#### 3. GRPO Loss

$$
\mathcal{L}_{\text{GRPO}} = -\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[\min\left(\frac{\pi_\theta}{\pi_{\text{old}}} \hat{A}_i,\; \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1{-}\epsilon, 1{+}\epsilon\right) \hat{A}_i\right) - \beta D_{\text{KL}}\right]
$$

```python
def group_policy_loss(cur_lp, old_lp, ref_lp,
                      adv, prompt_len, clip_eps=0.2, kl_coef=0.01):
    """
    GRPO 损失函数

    Args:
        cur_lp:     当前策略 log π_θ        [G, T]
        old_lp:     采样时策略 log π_old     [G, T]
        ref_lp:     参考策略 log π_ref       [G, T]
        adv:        组内归一化优势            [G]
        prompt_len: 输入 prompt 长度（只对 response 算 loss）
    """
    G, T = cur_lp.shape
    adv = adv.view(-1, 1)  # [G, 1] 广播到每个 token

    # Clipped policy gradient（和 PPO 一样）
    r = torch.exp(cur_lp - old_lp)
    r_clip = r.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    pg = torch.min(r * adv, r_clip * adv)

    # KL 惩罚
    kl = approx_kl_divergence(cur_lp, ref_lp)

    # 只对 response 部分计算 loss（构造 mask）
    resp_mask = torch.zeros(G, T)
    resp_mask[:, prompt_len:] = 1.0
    resp_len = T - prompt_len

    per_token = (pg - kl_coef * kl) * resp_mask
    return -(per_token.sum() / (G * resp_len))
```

::: tip GRPO vs PPO：核心区别
| 特性 | PPO | GRPO |
|------|-----|------|
| 优势估计 | GAE（需要 Value Network） | **组内归一化**（无需 Value Network） |
| 奖励来源 | Reward Model | **规则奖励**（如答案正确性） |
| 采样方式 | 单次采样 | **多次采样**（G 个回答） |
| 模型数量 | 4 个 | **2 个**（policy + ref） |
| 适用场景 | 通用 | 有明确正确答案的任务（数学、代码） |

GRPO 最大的工程优势：**省掉了 Critic 模型，显存需求几乎减半。**
:::

## 从教学版到生产版

教学版本帮助理解原理，但生产级实现需要额外的工程优化。以下是关键差异：

### 工程优化清单

| 优化项 | 教学版 | 生产版 |
|--------|--------|--------|
| 多卡训练 | 单卡 | DeepSpeed / FSDP |
| 参数效率 | 全参数 | **QLoRA + Multi-Adapter** |
| 梯度累积 | 无 | 有（等效更大 batch size） |
| 混合精度 | FP32 | BF16 / FP16 |
| 生成优化 | 朴素采样 | vLLM 加速 / 投机采样 |
| KV Cache | 无 | 有（加速生成） |

### Multi-Adapter LoRA 架构

生产级 PPO 的一个巧妙设计是用 **Multi-Adapter LoRA** 共享基座模型：

```
基座模型 (冻结)
  ├── LoRA-Policy   → actor（可训练）
  ├── LoRA-Ref      → ref（冻结快照）
  └── LoRA-Value    → critic（可训练）
  
RM 单独加载或同样用 LoRA Adapter
```

这样四个模型共享基座权重，只训练少量 LoRA 参数，显存占用大幅降低。

::: details 生产级训练脚本的关键配置
```python
# PPO 训练的关键超参数（生产环境参考值）
config = {
    "learning_rate": 1e-5,
    "batch_size": 64,
    "mini_batch_size": 8,
    "gradient_accumulation_steps": 8,     # 等效 batch_size = 64
    "num_ppo_epochs": 4,                  # 每批数据重复训练次数
    "max_grad_norm": 1.0,                 # 梯度裁剪，防止 NaN
    "kl_coef": 0.01,                      # KL 惩罚系数
    "cliprange": 0.2,                     # PPO clip 范围
}
```
:::

## LoRA 参数复用：单卡 PPO 的工程实践

### 传统 PPO 的显存瓶颈

从前面的实现可以看到，PPO-RLHF 需要同时加载 **4 个模型**：Actor、Critic、Reference、Reward。以 LLaMA-7B 为例：

| 模型 | 参数量 | FP16 显存 | 角色 |
|------|--------|----------|------|
| Actor | 7B | ~14 GB | 生成 + 策略更新 |
| Critic | 7B | ~14 GB | 价值估计 |
| Reference | 7B | ~14 GB | KL 约束 |
| Reward | 7B | ~14 GB | 打分 |
| **合计** | **28B** | **~56 GB** | — |

即使用 BF16，一张 A100 80GB 也很勉强。**但仔细想想：这 4 个模型的主干参数几乎相同，真正不同的只是微调部分。** 能不能只加载一份主干？

### 核心思路：一份基座 + 多组 LoRA 权重

关键洞察：LoRA 微调只改变极少参数（通常 < 0.5%），四个角色可以共享同一个量化基座：

```
                    ┌────────────────┐
                    │ 量化基座 (4-bit) │
                    │    ~3.5 GB      │
                    └───────┬────────┘
                            │ 共享
         ┌──────────┬───────┼────────┬────────────┐
         ▼          ▼       ▼        ▼            │
   ┌──────────┐ ┌───────┐ ┌──────┐               │
   │ LoRA-π   │ │ Value │ │LoRA-r│  Reference    │
   │ (policy) │ │ Head  │ │(reward)│ = 基座本身   │
   │  ~20 MB  │ │ ~4 KB │ │~20 MB│  (冻结 LoRA) │
   └──────────┘ └───────┘ └──────┘               │
```

- **Policy**：可训练的 LoRA adapter
- **Value Head**：一个线性层 $\mathbb{R}^{d} \to \mathbb{R}$
- **Reward**：预训练好的 RM LoRA，推理时加载、冻结
- **Reference**：就是基座本身——`set_adapter` / `disable_adapter` 切换即可，零额外显存

### 显存对比

| 方案 | 加载的模型份数 | 7B 显存估算 | 最低硬件 |
|------|-------------|------------|---------|
| 传统 PPO (FP16) | 4 份完整模型 | ~56 GB | 4x A100 40GB |
| 传统 PPO (BF16) | 4 份完整模型 | ~56 GB | 1x A100 80GB (勉强) |
| **LoRA 复用 (QLoRA)** | **1 份基座 + LoRA 权重** | **~4 GB** | **1x RTX 3090 24GB** |

显存节省比：$\frac{4 \times 14\text{GB}}{3.5\text{GB} + 3 \times 0.02\text{GB}} \approx 16\times$

### 实现：基于 PEFT 的角色切换

这个思路在 TRL 库中已有原生支持。下面基于 `peft` 的 `set_adapter` API 写一个教学版：

```python
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch, torch.nn as nn

class LoRARoleSwitcher:
    """
    教学实现：用 PEFT adapter 管理 PPO 的多个角色。
    实际工程中推荐直接用 TRL 的 PPOTrainer + peft_config 参数。
    """
    def __init__(self, base_model_id: str, reward_lora_path: str):
        # ---- 1. 加载量化基座 ----
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb
        )

        # ---- 2. 挂载 policy adapter（可训练）----
        policy_lora = LoraConfig(r=32, lora_alpha=16, task_type="CAUSAL_LM")
        self.base = get_peft_model(self.base, policy_lora, adapter_name="policy")

        # ---- 3. 加载 reward adapter（冻结）----
        self.base.load_adapter(reward_lora_path, adapter_name="reward")
        for p in self.base.get_adapter("reward").parameters():
            p.requires_grad_(False)

        # ---- 4. Value Head：一个可训练的线性层 ----
        hidden = self.base.config.hidden_size
        self.v_head = nn.Linear(hidden, 1)

    # ---------- 角色切换 ----------

    def policy_forward(self, input_ids):
        """激活 policy adapter，返回 logits"""
        self.base.set_adapter("policy")
        return self.base(input_ids=input_ids).logits

    def reference_logprobs(self, input_ids):
        """禁用所有 adapter → 纯基座 = reference model"""
        with self.base.disable_adapter():
            with torch.no_grad():
                logits = self.base(input_ids=input_ids).logits
        return torch.log_softmax(logits, dim=-1)

    def reward_score(self, input_ids):
        """切换到 reward adapter，取最后 token 隐状态过 v_head"""
        self.base.set_adapter("reward")
        with torch.no_grad():
            h = self.base(input_ids=input_ids, output_hidden_states=True)
            last_hidden = h.hidden_states[-1][:, -1, :]
        return self.v_head(last_hidden).squeeze(-1)
```

::: details 与直接用 TRL PPOTrainer 的区别？
上面的 `LoRARoleSwitcher` 是纯教学代码，手动管理 adapter 切换。
实际项目中，TRL 的 `PPOTrainer` 已内置了这些逻辑：

```python
from trl import PPOConfig, PPOTrainer
# 传入 peft_config 即可自动处理 ref model 和 adapter 切换
trainer = PPOTrainer(
    config=PPOConfig(batch_size=16, mini_batch_size=4),
    model=model,
    tokenizer=tokenizer,
    peft_config=LoraConfig(r=32, lora_alpha=16),
)
```

TRL 会自动用 `disable_adapter()` 生成 reference logprobs，无需手动管理。
:::

### 训练成本估算

| 配置 | 硬件 | 7B 模型训练时间 | 成本估算 |
|------|------|---------------|---------|
| 传统 PPO (FP16) | 4x A100 80GB | ~10 小时 | ~$120 (云) |
| LoRA 复用 (QLoRA) | 1x A100 40GB | ~8 小时 | ~$16 (云) |
| LoRA 复用 (QLoRA) | 1x RTX 3090 | ~15 小时 | 消费级可行 |

::: tip 工程启示
LoRA 参数复用的核心思路：**微调后的模型 = 基座 + 低秩增量**，不同角色只是不同的增量。
这让 PPO 从"4 个完整模型"降级为"1 份基座 + 几组 LoRA 权重"，彻底打破了显存壁垒。
这一思路不仅适用于 PPO，任何需要多模型协作的场景（self-play、judge+generator 等）都可以复用。
:::

## 苏格拉底时刻

::: details Q1: PPO 的 clip 和 KL 惩罚是否冗余？
不完全冗余，它们从不同角度约束策略更新：
- **KL 惩罚**在 reward 层面约束——让偏离参考模型的生成获得更低奖励
- **Clip** 在梯度层面约束——限制单次更新的步长

实践中两者结合效果最好。有些实现（如 TRL）还支持 Adaptive KL，动态调整 KL 系数。
:::

::: details Q2: DPO 的 β 参数如何理解？
$\beta$ 控制策略偏离参考模型的程度：
- **β 大**（如 0.5）：策略紧贴 ref，学到的偏好较弱但更稳定
- **β 小**（如 0.01）：策略可以大幅偏离 ref，学到的偏好更强但可能过拟合

直觉：$\beta$ 就是信任度——你多大程度上信任偏好数据而非原始 SFT 模型？
:::

::: details Q3: GRPO 为什么不需要 Value Network？
PPO 用 Value Network 估计 $V(s)$ 是为了计算优势 $A = Q - V$。GRPO 的巧妙之处在于：

**同一个 prompt 采样多个回答，用组内统计量替代 Value 估计。**

- 组均值 $\bar{r}$ ≈ $V(s)$（同一状态的期望回报）
- $r_i - \bar{r}$ ≈ $A_i$（相对于平均水平的优势）

代价是需要更多采样（G=8~64），但省掉了 Critic 模型的训练和显存。
:::

::: details Q4: 如果偏好数据质量很差，DPO 和 PPO 谁更鲁棒？
PPO 更鲁棒，因为：
1. PPO 通过在线采样不断生成新数据，不完全依赖固定数据集
2. RM 的打分可以过滤掉明显错误的偏好标注
3. DPO 直接在离线数据上训练，数据中的噪声会直接影响策略

这也是为什么 Online DPO（在线采样 + DPO loss）越来越流行。
:::

## 面试考点

::: warning 高频面试题
1. **PPO 的三个 loss 分别是什么？** Policy Loss（clipped surrogate）、Value Loss（clipped value）、Entropy Loss（鼓励探索）
2. **GAE 的 λ 参数如何影响训练？** λ→0 高偏差低方差（类似 TD），λ→1 低偏差高方差（类似 MC）
3. **DPO 相比 PPO 的优缺点？** 优：简单、稳定、省显存；缺：依赖离线数据质量、无在线探索
4. **GRPO 如何省掉 Value Network？** 同 prompt 多次采样，用组内 reward 统计量替代 V(s)
5. **RLHF 中 KL 惩罚的作用？** 防止策略偏离 SFT 模型太远，避免 reward hacking
6. **Bradley-Terry 模型和 DPO 的关系？** DPO 的推导基于 BT 假设，将显式 reward 替换为隐式的策略 log-ratio
:::

## 推荐资源

- [Proximal Policy Optimization (OpenAI)](https://arxiv.org/abs/1707.06347) — PPO 原论文
- [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) — RLHF 应用于 LLM 的开创性工作
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) — DPO 原论文
- [DeepSeekMath: GRPO](https://arxiv.org/abs/2402.03300) — GRPO 的提出
- [TRL 文档](https://huggingface.co/docs/trl) — 生产级实现参考
- [The N Implementation Details of RLHF](https://huggingface.co/blog/the_n_implementation_details_of_RLHF_with_PPO) — PPO-RLHF 的工程细节
