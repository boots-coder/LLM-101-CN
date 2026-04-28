---
title: "PPO 代码填空"
description: "Level 2-3 填空：Importance Ratio、GAE、Clipped Surrogate、Value Clipped、KL K1/K2/K3、Token Reward Shaping"
topics: [fill-in, PPO, GAE, clipped-surrogate, KL-estimation, importance-ratio, value-loss, RLHF]
---
# PPO 代码填空 (Level 2-3)

本练习覆盖 RLHF-PPO 训练中的 6 个核心算子：importance ratio、GAE 倒序递推、clipped surrogate policy loss、clipped value loss、KL 三种估计、token-level reward shaping。把这 6 块拼起来就是 [TRL `PPOTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py) 的算法核心。

::: info 前置知识
- 强化学习基础（policy gradient、advantage、value function）
- KL 散度的三种近似估计（详见 [training/alignment.md](../training/alignment.md#grpo-中的-kl-散度分析)）
- PyTorch 基础（`torch.clamp` / `torch.maximum` / `torch.minimum`）
:::

::: tip 核心公式
**Importance Ratio:** $\rho_t = \exp(\log\pi_\theta(a_t|s_t) - \log\pi_{\text{old}}(a_t|s_t))$

**GAE:** $\hat{A}_t = \delta_t + \gamma\lambda\hat{A}_{t+1}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

**Clipped Surrogate (策略 loss):** $\mathcal{L}^{\text{CLIP}} = \mathbb{E}\left[\max\left(-\rho_t \hat{A}_t,\ -\text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$

**Value Clipped Loss:** $\mathcal{L}^{V} = \frac{1}{2}\mathbb{E}\left[\max\left((V_\theta - R)^2,\ (V_{\text{clip}} - R)^2\right)\right]$
:::

---

## 练习 1：Importance Ratio（Level 2）

PPO 的核心是用 `old policy` 采样 + 多轮 `inner update` 复用样本。每次 inner update 时，用 importance ratio 修正"采样分布与当前策略的差异"。

```python
import torch

def importance_ratio(new_logprobs, old_logprobs, mask):
    """
    计算 token 级 importance ratio。

    参数:
        new_logprobs: 当前 policy 的逐 token log prob, [B, T]
        old_logprobs: 采样时刻 frozen old policy 的 log prob, [B, T]
        mask:         [B, T]，1 表示该 token 在 response 内
    返回:
        ratio:        [B, T]，masked 后的 importance ratio
    """
    # TODO 1: 计算 log ratio = new - old
    log_ratio = _____

    # TODO 2: ratio = exp(log_ratio)，并对 padding 位置置 1（exp(0)=1，不影响后续 mean）
    ratio = _____
    ratio = ratio * mask + (1 - mask)
    return ratio


# ====== 测试 ======
torch.manual_seed(0)
B, T = 2, 6
old = torch.randn(B, T) * 0.1
new = old + torch.randn(B, T) * 0.05
mask = torch.tensor([[1, 1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0, 0]], dtype=torch.float)
ratio = importance_ratio(new, old, mask)
print(f"ratio range: [{ratio.min():.4f}, {ratio.max():.4f}]")  # 通常都接近 1.0
```

::: details 提示
- log_ratio 是逐 token 的差，对应数学定义 $\log\frac{\pi_\theta}{\pi_{\text{old}}}$。
- `ratio = log_ratio.exp()` 即可。
- 对 mask=0 的位置补 1.0 是工程约定，避免随后 `(ratio * adv).mean()` 把 0/0 噪声放大。
:::

<details>
<summary>点击查看答案</summary>

```python
# TODO 1
log_ratio = new_logprobs - old_logprobs

# TODO 2
ratio = log_ratio.exp()
```

**解析：**

importance ratio 的本质是"在新策略下采到这个 token 的概率 / 在采样策略下采到这个 token 的概率"。在 token 维度计算时直接用 `log_softmax → gather` 拿到的 log prob 相减，再 exp 即可。

**为什么先 log_ratio 再 exp，而不是直接 `pi/pi_old`？**

数值稳定。直接除法在 prob 极小时会下溢；log 域计算先做减法再 exp，配合 PyTorch 内部的 `log_softmax` fused kernel 几乎不会丢精度。

</details>

---

## 练习 2：GAE 倒序递推（Level 2）

GAE 的递推关系：$\hat{A}_t = \delta_t + \gamma\lambda\hat{A}_{t+1}$，必须**从最后一个 token 反向**累加。这是 PPO 实现里最容易写反的地方。

```python
import torch

def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """
    GAE 倒序递推。

    参数:
        rewards: token 级 reward, [B, T]（KL 惩罚已揉进，最后一位还有 RM 标量）
        values:  critic 估计, [B, T]
        mask:    [B, T]
        gamma:   折扣因子（语言任务通常取 1.0）
        lam:     GAE λ
    返回:
        advantages, returns: 都是 [B, T]
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B)

    # TODO 1: 倒序遍历 t（从 T-1 到 0）
    for t in _____:
        # TODO 2: next_value：t+1 越界时为 0，否则取 values[:, t+1]
        next_value = _____

        # TODO 3: TD 残差 delta_t = r_t + gamma * V(t+1) - V(t)
        delta = _____

        # TODO 4: GAE 递推 A_t = delta + gamma * lam * A_{t+1}
        gae = _____

        advantages[:, t] = gae

    advantages = advantages * mask
    returns = advantages + values                              # critic 的回归目标
    return advantages.detach(), returns.detach()


# ====== 测试 ======
B, T = 2, 5
rewards = torch.zeros(B, T)
rewards[:, -1] = torch.tensor([1.0, -0.5])                     # RM 分加在最后一位
values = torch.zeros(B, T) + 0.1
mask = torch.ones(B, T)
adv, ret = compute_gae(rewards, values, mask)
print(f"advantages:\n{adv}")
print(f"returns:\n{ret}")
```

::: details 提示
- `for t in reversed(range(T)):` —— Python 标准倒序写法。
- `next_value = values[:, t+1] if t + 1 < T else torch.zeros(B)` —— 边界保护。
- delta 严格按 TD(0) 残差定义：`rewards[:, t] + gamma * next_value - values[:, t]`。
- gae 用上一轮（更晚 t）的累计值递推：`delta + gamma * lam * gae`。
:::

<details>
<summary>点击查看答案</summary>

```python
# TODO 1
for t in reversed(range(T)):

    # TODO 2
    next_value = values[:, t + 1] if t + 1 < T else torch.zeros(B)

    # TODO 3
    delta = rewards[:, t] + gamma * next_value - values[:, t]

    # TODO 4
    gae = delta + gamma * lam * gae
```

**解析：**

GAE 的核心是把 **bias-variance trade-off** 通过 λ 调节：
- λ=0 退化为 1-step TD（高 bias，低 variance）
- λ=1 退化为 Monte Carlo return（低 bias，高 variance）
- λ=0.95 是 PPO 论文经验值，跨任务效果稳定

倒序的原因：当前 token 的 advantage 等于"未来所有 TD 残差的几何加权和"，所以必须从未来反推到当前。这等价于把 BPTT 沿 time 轴展开。

**`gae` 累积变量的初始化为 0** 暗含一个假设：T 时刻之后的 advantage 为 0（episode 终止）。

</details>

---

## 练习 3：Clipped Surrogate Policy Loss（Level 2）

PPO 用 `max(unclipped, clipped)` 作为 loss——这里的"max"是 PPO 论文里最容易记错的细节。

```python
import torch

def policy_surrogate_loss(new_logprobs, old_logprobs, advantages, mask, clip_range=0.2):
    """
    PPO clipped surrogate loss。

    参数:
        new_logprobs: [B, T]
        old_logprobs: [B, T]
        advantages:   [B, T]（来自 compute_gae）
        mask:         [B, T]
        clip_range:   ε，论文常用 0.2
    返回:
        scalar loss（已经过 mask 平均）
    """
    # TODO 1: 计算 ratio = exp(new - old)
    ratio = _____

    # TODO 2: unclipped 项 = -advantages * ratio
    unclipped = _____

    # TODO 3: clipped 项 = -advantages * clamp(ratio, 1 - eps, 1 + eps)
    clipped = _____

    # TODO 4: 取 max（悲观估计：选两者中 loss 更大的一项）
    loss_per_tok = _____

    # mask 平均
    return (loss_per_tok * mask).sum() / mask.sum().clamp(min=1)


# ====== 测试 ======
torch.manual_seed(1)
B, T = 2, 4
old = torch.zeros(B, T)
new = torch.tensor([[0.0, 0.5, -0.5, 1.0],
                    [-0.3, 0.0, 0.3, -0.1]])                  # ratio 越界样本
adv = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                    [-1.0, -1.0, -1.0, -1.0]])
mask = torch.ones(B, T)
loss = policy_surrogate_loss(new, old, adv, mask, clip_range=0.2)
print(f"policy loss = {loss.item():+.4f}")
```

::: details 提示
- `ratio = (new_logprobs - old_logprobs).exp()`
- 注意 unclipped / clipped 项前都有**负号**（policy gradient 上升 = loss 下降）。
- `torch.clamp(ratio, 1 - clip_range, 1 + clip_range)` 把 ratio 限制在 `[0.8, 1.2]`（ε=0.2 时）。
- 用 `torch.maximum(unclipped, clipped)` 取逐元素最大值。
:::

<details>
<summary>点击查看答案</summary>

```python
# TODO 1
ratio = (new_logprobs - old_logprobs).exp()

# TODO 2
unclipped = -advantages * ratio

# TODO 3
clipped = -advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)

# TODO 4
loss_per_tok = torch.maximum(unclipped, clipped)
```

**解析：**

为什么是 `max` 而不是 `min`？因为 unclipped/clipped 项前都有负号，"loss 更大"对应"对策略改进更悲观"。具体在 4 种情形下行为：

| advantage 符号 | ratio 状态 | clip 行为 |
|----|----|----|
| A > 0 | ρ > 1+ε | clip 限制收益（最大收益 = (1+ε)·A） |
| A > 0 | ρ < 1-ε | unclipped 项更小（更负），max 取 unclipped——**继续给信号** |
| A < 0 | ρ > 1+ε | unclipped 项更小（更负），max 取 unclipped——**继续给信号** |
| A < 0 | ρ < 1-ε | clip 限制损失（最大下降 = (1-ε)·A） |

**核心直觉**：clip 只在"沿好方向走太远"时踢入，"想往坏方向走"时不限制——逼着策略只做小步、保守的改进，与 trust region 的思路一致。

</details>

---

## 练习 4：Value Clipped Loss（Level 3）

PPO 的 value loss 不是简单的 MSE——critic 也要 clip，避免一次 inner update 把 `V_θ` 拉飞。这是 PPO 论文 Appendix 里经常被忽略的细节。

```python
import torch

def value_clipped_loss(new_values, old_values, returns, mask, clip_range_value=0.2):
    """
    PPO clipped value loss。

    参数:
        new_values: 当前 critic 输出, [B, T]
        old_values: rollout 时刻冻结的 critic 输出, [B, T]
        returns:    GAE 返回的 returns（advantages + values_old）, [B, T]
        mask:       [B, T]
        clip_range_value: critic 更新的 clip 范围
    返回:
        scalar loss
    """
    # TODO 1: 把 new_values 限制在 old_values ± clip_range_value 内
    v_clipped = _____

    # TODO 2: unclipped 损失 = (new_values - returns) ** 2
    loss_unclipped = _____

    # TODO 3: clipped 损失 = (v_clipped - returns) ** 2
    loss_clipped = _____

    # TODO 4: 取 max（同样是悲观估计）后乘 0.5
    loss_per_tok = _____

    return (loss_per_tok * mask).sum() / mask.sum().clamp(min=1)


# ====== 测试 ======
torch.manual_seed(2)
B, T = 2, 5
old_v = torch.randn(B, T) * 0.5
new_v = old_v + torch.randn(B, T) * 0.3                       # 引入了一些"漂移"
returns = old_v + torch.randn(B, T) * 0.4
mask = torch.ones(B, T)
loss = value_clipped_loss(new_v, old_v, returns, mask, clip_range_value=0.2)
print(f"value loss = {loss.item():.4f}")
```

::: details 提示
- `v_clipped = old_values + torch.clamp(new_values - old_values, -clip_range_value, clip_range_value)`
- 两个 squared error 项都用 `(... - returns) ** 2`。
- max 用 `torch.maximum(loss_unclipped, loss_clipped)` 然后乘 0.5（半 MSE 是 PG 论文标准约定）。
:::

<details>
<summary>点击查看答案</summary>

```python
# TODO 1
v_clipped = old_values + torch.clamp(
    new_values - old_values, -clip_range_value, clip_range_value,
)

# TODO 2
loss_unclipped = (new_values - returns) ** 2

# TODO 3
loss_clipped = (v_clipped - returns) ** 2

# TODO 4
loss_per_tok = 0.5 * torch.maximum(loss_unclipped, loss_clipped)
```

**解析：**

PPO 的 value clip 与 policy clip **逻辑对称**：

| 项 | clip 对象 | 直觉 |
|----|----------|------|
| Policy | `ratio = π/π_old` | 限制策略偏移幅度 |
| Value | `V_new - V_old` | 限制 critic 更新幅度 |

为什么必须 clip critic？多轮 inner update（`ppo_epochs > 1`）共用一份 rollout 数据。如果不 clip，critic 在第 K 次 inner update 时可能已经偏离 rollout 时刻太远，导致后续 advantage 计算错位。

**TRL 的实现**完全一致，可对照 [`PPOTrainer.training_step`](https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py)。

</details>

---

## 练习 5：KL 散度的三种估计（Level 2）

K1（朴素）/ K2（无偏低方差）/ K3（Schulman 推荐）三种 KL 估计——RLHF 中 PPO 用 K1（揉进 reward），GRPO 用 K3（直接进 loss）。详见 [John Schulman 的博客](http://joschu.net/blog/kl-approx.html)。

```python
import torch

def kl_estimators(p_logprobs, q_logprobs):
    """
    给定从 q 采样的 token 序列在 p 和 q 下的 log prob，给出 KL(q || p) 的三种估计。

    定义 r = p / q（比值）；以下表达式都是从 q 采样估计 E_q[KL]：
        K1 = -log r
        K2 = (1/2) * (log r) ** 2
        K3 = (r - 1) - log r

    参数:
        p_logprobs: [B, T]，目标分布 p 在采样 token 上的 log prob
        q_logprobs: [B, T]，采样分布 q 在采样 token 上的 log prob
    返回:
        k1, k2, k3: 三个估计，形状均为 [B, T]
    """
    # TODO 1: log r = p_logprobs - q_logprobs
    log_r = _____

    # TODO 2: K1 = -log_r
    k1 = _____

    # TODO 3: K2 = 0.5 * log_r ** 2
    k2 = _____

    # TODO 4: K3 = exp(log_r) - 1 - log_r （等价于 r - 1 - log r）
    k3 = _____

    return k1, k2, k3


# ====== 测试 ======
torch.manual_seed(3)
B, T = 2, 8
q = torch.randn(B, T) * 0.5
p = q + torch.randn(B, T) * 0.05                              # p 与 q 接近
k1, k2, k3 = kl_estimators(p, q)
print(f"K1 mean = {k1.mean():+.4f}  (could be negative — high variance)")
print(f"K2 mean = {k2.mean():+.4f}  (always >= 0)")
print(f"K3 mean = {k3.mean():+.4f}  (always >= 0, low variance)")
```

::: details 提示
- log_r 就是 `p - q`（注意是 KL(q||p)，所以 log r = log(p/q)）。
- K1 是无偏但方差大；K2 永远非负但有偏；K3 既无偏也非负，是 Schulman 推荐用法。
- K3 的恒非负性来自不等式 $e^x - 1 \geq x$，等号在 $x=0$ 时取得。
:::

<details>
<summary>点击查看答案</summary>

```python
# TODO 1
log_r = p_logprobs - q_logprobs

# TODO 2
k1 = -log_r

# TODO 3
k2 = 0.5 * log_r ** 2

# TODO 4
k3 = log_r.exp() - 1 - log_r
```

**解析：**

| 估计 | 表达式 | 无偏性 | 方差 | 非负 |
|------|--------|--------|------|------|
| K1 | $-\log r$ | ✅ | 高 | ❌ |
| K2 | $\frac{1}{2}(\log r)^2$ | ❌ | 中 | ✅ |
| K3 | $r - 1 - \log r$ | ✅ | 低 | ✅ |

**为什么 K3 最好？**

- **无偏**：$E_q[r - 1 - \log r] = E_q[r] - 1 - E_q[\log r] = 1 - 1 + \text{KL}(q\|p) = \text{KL}(q\|p)$
- **非负**：$e^x \geq 1 + x$ → $r \geq 1 + \log r$ → $r - 1 - \log r \geq 0$
- **低方差**：相比 K1（线性 log），K3 的二阶展开附近更"平"，采样估计更稳

**RLHF 中的工程选择**：
- **PPO**：把 K1 揉进 token-level reward（每 token 减 `kl_coef * (logprobs - ref_logprobs)`），然后让 GAE 自动分摊。
- **GRPO**：直接把 K3 加进 per-token loss，避免 reward 形状失真——这也是 R1 论文的选择。

</details>

---

## 练习 6：把 KL 揉进 Token-Level Reward（Level 3）

PPO 把 RM 的"序列级标量分"和 KL 惩罚的"token 级序列"统一到一条 token reward 张量里——这是 PPO 训练流程里最容易写错的拼装步骤。

```python
import torch

def shape_token_rewards(rm_score, logprobs, ref_logprobs, response_mask, kl_coef=0.05):
    """
    把 RM 序列级分数 + KL 惩罚拼成 token 级 reward。

    参数:
        rm_score:      [B]      RM 给整条 response 的标量分
        logprobs:      [B, T]   policy 当前 token log-prob
        ref_logprobs:  [B, T]   reference 模型的 token log-prob
        response_mask: [B, T]   1 表示该 token 在 response 内
        kl_coef:       float    KL 惩罚系数
    返回:
        token_rewards: [B, T]
    """
    # TODO 1: 每 token 的 KL（K1 估计）= logprobs - ref_logprobs
    kl = _____

    # TODO 2: token_rewards 全程减 kl_coef * kl（每 token 都受 KL 约束）
    token_rewards = _____

    # TODO 3: 找到每条样本的"最后一个 response token"位置
    last_idx = response_mask.sum(dim=-1).long() - 1

    # TODO 4: 把 RM 标量分加到最后一位（仅最后一个 response token）
    batch_idx = torch.arange(rm_score.size(0))
    token_rewards[batch_idx, last_idx] += _____

    # 应用 mask（response 之外置 0）
    return token_rewards * response_mask


# ====== 测试 ======
torch.manual_seed(4)
B, T = 2, 6
rm_score = torch.tensor([1.5, -0.3])                          # 第 1 条好，第 2 条差
logprobs = torch.randn(B, T) * 0.1
ref_logprobs = logprobs - torch.randn(B, T) * 0.05            # 假设 policy 比 ref 略微偏高
response_mask = torch.tensor([[0, 0, 1, 1, 1, 1],
                              [0, 1, 1, 1, 1, 0]], dtype=torch.float)
rewards = shape_token_rewards(rm_score, logprobs, ref_logprobs, response_mask, kl_coef=0.05)
print(f"token_rewards:\n{rewards}")
print(f"\n第 1 条最后一个 response token (位置 5) 应包含 RM 分: {rewards[0, 5]:.3f}  (≈ 1.5 + KL 项)")
print(f"第 2 条最后一个 response token (位置 4) 应包含 RM 分: {rewards[1, 4]:.3f}  (≈ -0.3 + KL 项)")
```

::: details 提示
- `kl = logprobs - ref_logprobs` —— 这是 K1 估计（PPO 用 K1 揉进 reward）。
- `token_rewards = -kl_coef * kl` —— 全程的 KL 惩罚（per-token，从 response 第 1 个 token 开始）。
- `last_idx = response_mask.sum(-1).long() - 1` —— `response_mask` 累加得到 response 长度，再 -1 得到最后一个 response token 的索引。
- 把 `rm_score` 加到 `[batch_idx, last_idx]` 位置即可，注意 `batch_idx` 是 `torch.arange(B)`。
:::

<details>
<summary>点击查看答案</summary>

```python
# TODO 1
kl = logprobs - ref_logprobs

# TODO 2
token_rewards = -kl_coef * kl

# TODO 4
token_rewards[batch_idx, last_idx] += rm_score
```

**解析：**

PPO 的 reward shaping 是 "**整段 KL 惩罚 + 末位 RM 标量分**" 的合成：

```
position:    0     1     2     3     4     5
token_r:    -κk  -κk  -κk  -κk  -κk  -κk + R
            └─────────KL 全程──────────┘  └─末位 RM─┘
```

为什么 RM 分**只加在最后一位**？因为 RM 是 sequence-level 评分——它只对完整 response 给分，没有 token-level 监督信号。GAE 倒序累加时，会自动把这份"信用"通过 critic 的 V 估计向前分摊。

**KL 用 K1 而不是 K3**：在 PPO 里 KL 是作为"reward 的负贡献"出现的，K1 = `-log r` 是无偏估计，与 reward 的"有正有负"语义自然对应。GRPO 把 KL 直接塞 loss，需要恒非负，所以选 K3。

**完整 PPO 拼装链路（对照本练习 1-6）**：
```
rollout → shape_token_rewards (Ex.6)
       → compute_gae (Ex.2)
       → ppo_epochs 内：
           importance_ratio (Ex.1)
         + kl_estimators (Ex.5)
         → policy_surrogate_loss (Ex.3)
         + value_clipped_loss (Ex.4)
         → backward
```

</details>

---

## MLM 代码训练模式

完成上面的固定填空后，试试随机挖空模式 -- 每次点击「刷新」会随机遮盖不同的代码片段，帮你彻底记住每一行。

### Importance Ratio

<CodeMasker title="PPO Importance Ratio：log domain → exp" :mask-ratio="0.15">
def importance_ratio(new_logprobs, old_logprobs, mask):
    log_ratio = new_logprobs - old_logprobs
    ratio = log_ratio.exp()
    ratio = ratio * mask + (1 - mask)
    return ratio
</CodeMasker>

### GAE 倒序递推

<CodeMasker title="GAE：倒序累加 TD 残差" :mask-ratio="0.15">
def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B)

    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t + 1 < T else torch.zeros(B)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        gae = delta + gamma * lam * gae
        advantages[:, t] = gae

    advantages = advantages * mask
    returns = advantages + values
    return advantages.detach(), returns.detach()
</CodeMasker>

### Clipped Surrogate Loss

<CodeMasker title="PPO 策略目标：max(unclipped, clipped) 取悲观估计" :mask-ratio="0.15">
def policy_surrogate_loss(new_logprobs, old_logprobs, advantages, mask, clip_range=0.2):
    ratio = (new_logprobs - old_logprobs).exp()
    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    loss_per_tok = torch.maximum(unclipped, clipped)
    return (loss_per_tok * mask).sum() / mask.sum().clamp(min=1)
</CodeMasker>

### Value Clipped Loss

<CodeMasker title="PPO Value Loss：critic 也要 clip" :mask-ratio="0.15">
def value_clipped_loss(new_values, old_values, returns, mask, clip_range_value=0.2):
    v_clipped = old_values + torch.clamp(
        new_values - old_values, -clip_range_value, clip_range_value,
    )
    loss_unclipped = (new_values - returns) ** 2
    loss_clipped = (v_clipped - returns) ** 2
    loss_per_tok = 0.5 * torch.maximum(loss_unclipped, loss_clipped)
    return (loss_per_tok * mask).sum() / mask.sum().clamp(min=1)
</CodeMasker>

### KL Estimators

<CodeMasker title="K1 / K2 / K3：三种 KL 估计" :mask-ratio="0.15">
def kl_estimators(p_logprobs, q_logprobs):
    log_r = p_logprobs - q_logprobs
    k1 = -log_r
    k2 = 0.5 * log_r ** 2
    k3 = log_r.exp() - 1 - log_r
    return k1, k2, k3
</CodeMasker>

### Token Reward Shaping

<CodeMasker title="PPO Reward Shaping：整段 KL + 末位 RM 分" :mask-ratio="0.15">
def shape_token_rewards(rm_score, logprobs, ref_logprobs, response_mask, kl_coef=0.05):
    kl = logprobs - ref_logprobs
    token_rewards = -kl_coef * kl
    last_idx = response_mask.sum(dim=-1).long() - 1
    batch_idx = torch.arange(rm_score.size(0))
    token_rewards[batch_idx, last_idx] += rm_score
    return token_rewards * response_mask
</CodeMasker>
