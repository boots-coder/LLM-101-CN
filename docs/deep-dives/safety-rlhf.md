---
title: 深度剖析 Safe RLHF——用 PPO-Lagrangian 把有用与安全解耦
description: 北大开源工作 PKU-Alignment/safe-rlhf：双 preference + 双模型（reward + cost）+ 拉格朗日约束优化
topics: [safety, rlhf, ppo-lagrangian, cost-model, beavertails, dual-preference, constrained-rl]
prereqs: [/training/alignment, /engineering/safety, /deep-dives/nano-rlhf]
---

# 深度剖析 Safe RLHF——用 PPO-Lagrangian 把有用与安全解耦

::: info 一句话总结
Safe RLHF 把"对齐"重新表达成一个**带约束的强化学习问题**：在 reward 模型衡量的"有用"上做最大化，同时把 cost 模型衡量的"伤害"作为不等式约束，用 PPO-Lagrangian 自适应地学到一个权衡系数 lambda，让"有用"与"安全"在训练过程中真正解耦。
:::

::: tip 来源声明
本文基于北京大学对齐组（PKU-Alignment）开源项目 [PKU-Alignment/safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf)（Apache 2.0），论文为 Dai et al. 2023 [Safe RLHF: Safe Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2310.12773)。所有源码引用均链接到 GitHub 仓库 main 分支。
:::

## 体系定位：从"加权"到"约束"的范式跃迁

[偏好对齐章节](/training/alignment) 已经介绍了两种主流的"安全 + 有用"组合方式，本文要介绍的 Safe RLHF 是第三种。先把三家放到同一张谱系图上：

| 范式 | 代表工作 | 安全建模方式 | 切换/加权 |
|------|---------|------------|----------|
| **单 RM 隐式混合** | Anthropic HH-RLHF (2022) | helpfulness 与 harmlessness 标注混在一个 RM 里训练 | 由数据比例隐式决定，无法事后调权 |
| **双 RM 静态切换** | LLaMA 2 Chat (2023) | 训两个独立 RM，按 `safety_score < 0.15` 切换 | 阈值是经验值（0.15），需要重新跑实验调 |
| **双 RM 动态约束** | Safe RLHF (2023) | reward + cost 两个独立模型，约束优化 | lambda 由训练动力学自学习，不需要手调阈值 |

**Safe RLHF 的核心主张**：把"安全"从优化目标里拆出来，变成约束。形式化：

$$
\max_\pi \; \mathbb{E}_{(s,a) \sim \pi}[R(s,a)] \quad \text{s.t.} \quad \mathbb{E}_{(s,a) \sim \pi}[C(s,a)] \le d
$$

其中 $R$ 是 reward 模型（衡量有用），$C$ 是 cost 模型（衡量伤害），$d$ 是可接受的安全阈值（论文取 0）。这与 LLaMA 2 的"硬切换"形成鲜明对比：

- **LLaMA 2**：决策逻辑写死在公式里。一旦数据分布变化、阈值 0.15 失效，需要重新调参 + 重新训。
- **Safe RLHF**：决策逻辑变成训练动力学的一部分。当 cost 突破约束，lambda 上升让安全主导；当 cost 充裕，lambda 下降让有用主导。**这是一个反馈控制系统，不需要手动调阈值**。

::: details 为什么"加权 R - lambda·C"比单 RM 强
你可能会想：那我直接训一个 RM = R - 0.5·C 不就行了？区别在于：单 RM 训练时安全权重是**先验固定**的（由训练数据比例隐式决定），而 PPO-Lagrangian 的 lambda 是**后验自适应**的——它根据当前 policy 在 cost 约束上的违反情况实时调整。换句话说，Safe RLHF 把"加多少安全权重才合适"这个超参，从人手工调成自动学。
:::

## 核心内容

### 1. 双 preference 标注：BeaverTails 数据集

传统 RLHF 数据集（如 Anthropic HH）只标注一次偏好：标注员选 chosen vs rejected。Safe RLHF 要求标注员对同一对 `(response_A, response_B)` 做**两次独立判断**：

1. **Helpfulness 偏好**：哪个回复更有用？→ `better_response_id`
2. **Harmlessness 偏好**：哪个回复更安全？→ `safer_response_id`
3. **绝对安全标签**：每个回复单独标 `is_response_X_safe`（boolean）

这个 schema 在 [`safe_rlhf/datasets/raw/safe_rlhf.py:42-52`](https://github.com/PKU-Alignment/safe-rlhf/blob/main/safe_rlhf/datasets/raw/safe_rlhf.py#L42-L52) 里清晰可见——一条样本同时供给 reward 模型（用 `better`）和 cost 模型（用 `safer` + `is_safe`）。

**数据感**：

- 公开的 [PKU-SafeRLHF 数据集](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) 包含约 30 万条 dual preference 样本（30K 是早期版本），覆盖 14 个伤害类别。
- 标注一条 dual preference 在 BeaverTails 论文（[arXiv 2307.04657](https://arxiv.org/abs/2307.04657)）中报告的成本约为 2-3 美元，因为标注员需要做两次独立判断 + 一次绝对安全分类，是普通 HH 标注的 1.5 倍。
- 关键洞察：**helpful 偏好与 safer 偏好经常不一致**——一个有用的越狱回复可能是 better 但 unsafer。这种"冲突样本"恰恰是 Safe RLHF 想抓住的训练信号。

### 2. 双值函数：Reward 与 Cost 模型的非对称性

reward 模型与 cost 模型都基于 `AutoModelForScore`（一个挂在 LM 头上的 1 维 score head），但它们的损失函数**不对称**：

| 模型 | Loss 形式 | 含义 |
|------|----------|------|
| Reward (BT) | `-logsigmoid(R_better - R_worse)` | 只学相对排序 |
| Cost (BT + sign) | `-logsigmoid(C_unsafer - C_safer)` **`- logsigmoid(±sign · C)`** | 既学相对排序，又把绝对零点钉在"安全/不安全"的边界 |

cost 模型多出来的 sign 项，让"安全样本的 cost < 0、不安全样本的 cost > 0"成为一个**可学习的硬约束**——这是后续 lambda 更新的关键，因为 PPO-Lagrangian 需要 cost 的零点有物理含义（`d = 0` 才能解释为"恰好临界"）。

具体见 [`safe_rlhf/values/cost/trainer.py:260-272`](https://github.com/PKU-Alignment/safe-rlhf/blob/main/safe_rlhf/values/cost/trainer.py#L260-L272) 的 `sequence-wise` 分支。

### 3. PPO-Lagrangian：约束优化的拉格朗日松弛

约束优化的拉格朗日松弛把硬约束放进目标：

$$
\mathcal{L}(\pi, \lambda) = \mathbb{E}[R(s,a)] - \lambda \cdot \big(\mathbb{E}[C(s,a)] - d\big), \quad \lambda \ge 0
$$

这是一个 minmax 问题：policy 最大化 $\mathcal{L}$，乘子 $\lambda$ 最小化它（即最大化约束违反惩罚）。**Dual ascent** 给出 $\lambda$ 的更新规则：

$$
\lambda \leftarrow \max\big(0, \; \lambda + \eta_\lambda \cdot (\bar{C} - d)\big)
$$

直觉：

- 当 $\bar{C} > d$（违反约束，回复太不安全）→ $\lambda$ 增大 → 安全在 actor loss 里权重上升 → policy 被推向更安全。
- 当 $\bar{C} < d$（约束充裕）→ $\lambda$ 减小 → 不浪费"安全预算"，让 policy 更关注 reward。

Safe RLHF 在工程上做了一个数值稳定改进：**不直接更新 lambda，而是更新 log_lambda**，再 `lambda = exp(log_lambda)` 回去。这保证 lambda > 0 不需要手动 clip，且梯度尺度更稳定。这一点直接借鉴自约束 RL 的经典实现 PPO-Lagrangian（Stooke et al. 2020 [arXiv 2007.03964](https://arxiv.org/abs/2007.03964)）。

### 4. Actor Loss 的双 advantage 组合

Safe RLHF 跑两个 critic（reward critic + cost critic），分别用 GAE 算出 `reward_advantages` 和 `cost_advantages`，然后在 actor loss 里组合：

$$
A_{\text{combined}} = \frac{A_R - \lambda \cdot A_C}{1 + \lambda}
$$

注意分母 $(1 + \lambda)$——这是一个**归一化技巧**，防止 lambda 增大时整体 advantage 量级爆炸（否则 PPO 的 clip ratio 会失效）。从信息论角度看，这相当于把 advantage 投影到一个"reward-cost 单纯形"上。

### 5. 训练动力学与超参经验

来自 [`scripts/ppo-lag.sh:181-186`](https://github.com/PKU-Alignment/safe-rlhf/blob/main/scripts/ppo-lag.sh#L181-L186) 的默认配置：

| 超参 | 默认值 | 作用 |
|------|--------|------|
| `threshold` | `0.0` | $d$，cost 阈值（cost model 的零点已经标定为安全/不安全边界） |
| `lambda_init` | `1.0` | lambda 初值（reward 与 cost 等权重起步） |
| `lambda_lr` | `0.1` | lambda 的学习率（远大于 actor 的 1e-5，因为 lambda 需要快速响应） |
| `lambda_max` | `5.0` | lambda 上界，防止训练后期 reward 完全坍缩 |
| `lambda_update_delay_steps` | `0` | 延迟更新 lambda 的 step 数（warmup 期可以让 RM/CM 先稳定） |
| `episode_cost_window_size` | `128` | 滑动平均窗口，平滑 cost 信号（单 step cost 噪声大） |
| `kl_coeff` | `0.01` | KL 惩罚系数（与标准 PPO 相同） |

**单卡训练数据感**：根据论文实验，在 8×A100 (80G) 上跑 Alpaca-7B 一轮 PPO-Lagrangian 大约 **3-4 小时**（以 30K 数据子集计算）。完整 300K 数据约需 1-1.5 天。相比标准 PPO，多了一个 cost critic 模型，显存占用增加约 30%，因此实际部署常用 ZeRO-3 + offload。

### 6. KL 约束与 cost 的双向耦合

普通 PPO 用 KL 散度惩罚把 reward 拉回参考策略（防止 hacking），Safe RLHF 把这个机制做了**双向**——KL 惩罚同时**加到 reward** 和**减到 cost**：

$$
R_{\text{eff}}[t_{\text{end}}] = R(s,a) - \beta \cdot \text{KL}(\pi || \pi_{\text{ref}}), \quad C_{\text{eff}}[t_{\text{end}}] = C(s,a) + \beta \cdot \text{KL}(\pi || \pi_{\text{ref}})
$$

直觉：reward 维度的 KL 是"奖励 hacking 的惩罚"，cost 维度的 KL 是"安全 hacking 的反向惩罚"——避免 policy 通过偏离 ref 来骗 cost model 给出低分。这一点在 [`safe_rlhf/algorithms/ppo_lag/trainer.py:258-287`](https://github.com/PKU-Alignment/safe-rlhf/blob/main/safe_rlhf/algorithms/ppo_lag/trainer.py#L258-L287) 的 `add_kl_divergence_regularization` 函数里实现：注意 cost 项前的负号被 `torch.scatter_add(-kl_penalty_rewards, ...)` 翻转。

### 7. 与 LLaMA 2 双 RM 加权法的工程对比

| 维度 | LLaMA 2 (静态切换) | Safe RLHF (动态约束) |
|------|------------------|---------------------|
| **决策时机** | 推理 + 训练时根据 `safety_score` 阈值切换 | 训练时通过 lambda 自动权衡，推理时只用 actor |
| **超参敏感度** | 0.15 阈值需要重新调实验 | lambda 自适应，threshold 来自 cost model 的物理零点 |
| **冷启动** | 不需要 cost model，只要两个 RM | 需要先训 cost model（多一步训练 + 多一份偏好标注） |
| **样本利用** | 安全/有用样本各占一半训练 RM | dual preference 一条样本同时贡献 reward + cost 信号 |
| **失败模式** | 阈值卡死时 reward shaping 无法恢复 | lambda 失稳（震荡或单调上升）时整个训练会跑偏 |

工程口径的 take-away：**LLaMA 2 路径更适合数据/算力受限的中小团队**（少训一个模型，少一种偏好标注），**Safe RLHF 更适合追求精细化对齐控制的研究/产品团队**（拿到 lambda 这个可观测信号，可以诊断"安全预算是否被打爆"）。

### 8. 端到端 Pipeline 的 5 个阶段

把 Safe RLHF 完整跑一遍要经过五步训练，每一步都依赖上一步的产出：

| 阶段 | 输入 | 产出 | 数据需求 |
|------|------|------|---------|
| **Stage 0** Pretraining | 通用语料 | base LM | 万亿 token（已有） |
| **Stage 1** SFT | instruction 数据 | SFT model（actor 起点） | ~5 万条 instruction |
| **Stage 2** Reward Model | (chosen, rejected) helpful 偏好 | RM | ~30 万条 dual preference 中的 helpful 信号 |
| **Stage 3** Cost Model | (safer, unsafer) + is_safe 标签 | CM | 同上 dataset，复用 safer 信号 + 绝对安全标签 |
| **Stage 4** PPO-Lagrangian | SFT + RM + CM | aligned actor | prompt-only 数据 |

**复用关键**：Stage 2 和 Stage 3 用同一份 dual preference 数据，不需要额外标注。这就是 dual preference 标注 1.5 倍成本能换到 2 个独立模型的根本原因。

仓库里这五步对应的脚本依次是 [`scripts/sft.sh`](https://github.com/PKU-Alignment/safe-rlhf/blob/main/scripts/sft.sh)、[`scripts/reward-model.sh`](https://github.com/PKU-Alignment/safe-rlhf/blob/main/scripts/reward-model.sh)、[`scripts/cost-model.sh`](https://github.com/PKU-Alignment/safe-rlhf/blob/main/scripts/cost-model.sh)、[`scripts/ppo-lag.sh`](https://github.com/PKU-Alignment/safe-rlhf/blob/main/scripts/ppo-lag.sh)。

## 手撕源码

### 片段 1：cost loss 中的 sign 项（让 cost 零点有物理含义）

来源：[`safe_rlhf/values/cost/trainer.py:260-272`](https://github.com/PKU-Alignment/safe-rlhf/blob/main/safe_rlhf/values/cost/trainer.py#L260-L272)

```python
# sequence-wise 分支：cost loss = BT 排序项 + 安全符号项
# WHY: 普通 RM 只关心相对排序（chosen 比 rejected 高），
#      cost model 还要把"安全/不安全"的绝对零点钉死，
#      否则后面 PPO-Lag 的 threshold=0 没有物理意义。
elif self.args.loss_type == 'sequence-wise':
    loss = (
        # 第一项：unsafer cost > safer cost（与 BT 一致）
        -F.logsigmoid(higher_end_cost - lower_end_cost)
        # 第二项：safe 样本（sign=-1）的 cost 应为负
        - F.logsigmoid(lower_cost_sign * lower_end_cost)
        # 第三项：unsafe 样本（sign=+1）的 cost 应为正
        - F.logsigmoid(higher_cost_sign * higher_end_cost)
    ).mean()
```

**为什么不能像 reward 那样只用 BT margin loss**：reward 学的是 helpfulness 的相对排序，零点是常数无关紧要。但 cost 的零点是 PPO-Lagrangian 的 `threshold=0`，必须钉在"恰好临界安全"。少了 sign 两项，cost 模型可能学到"全负"或"全正"的退化解，让 lambda 更新失去方向。

### 片段 2：lambda 的对数空间更新（dual ascent）

来源：[`safe_rlhf/algorithms/ppo_lag/trainer.py:58-67`](https://github.com/PKU-Alignment/safe-rlhf/blob/main/safe_rlhf/algorithms/ppo_lag/trainer.py#L58-L67) 与 [`safe_rlhf/algorithms/ppo_lag/trainer.py:312-326`](https://github.com/PKU-Alignment/safe-rlhf/blob/main/safe_rlhf/algorithms/ppo_lag/trainer.py#L312-L326)

```python
# 初始化：log 空间存 lambda，保证 lambda = exp(log_lambda) > 0
# WHY: 直接更新 lambda 容易跨过 0 变负，PPO-Lag 对负 lambda 没有定义
self.log_lambda = torch.nn.Parameter(
    torch.tensor(np.log(self.args.lambda_init), device=self.args.device),
    requires_grad=True,
)
self.log_lambda_optimizer = torch.optim.SGD(
    [self.log_lambda], lr=self.args.lambda_lr,  # SGD 而非 Adam，避免动量污染
)

# 训练步：dual ascent 更新 log_lambda
# 注意 loss = -(C - d) * exp(log_lambda)，对 log_lambda 求梯度后 SGD 一步
if is_main_process() and self.global_step >= self.lambda_update_delay_steps:
    lambda_loss = -(episode_cost - self.threshold) * self.log_lambda.exp()
    self.log_lambda_optimizer.zero_grad()
    lambda_loss.backward()      # cost 高 → loss 大 → log_lambda 增大
    self.log_lambda_optimizer.step()
    if self.log_lambda_max is not None:
        with torch.no_grad():
            self.log_lambda.clamp_(max=self.log_lambda_max)  # 上界防爆炸
```

**关键工程细节**：lambda 只在 `main_process` 上更新，然后通过 `dist.broadcast` 广播到所有 GPU——避免每张卡用各自的 `episode_cost` 估计独立更新（会导致 lambda 在不同 rank 上发散）。

### 片段 3：actor loss 中的双 advantage 组合

来源：[`safe_rlhf/algorithms/ppo_lag/trainer.py:289-309`](https://github.com/PKU-Alignment/safe-rlhf/blob/main/safe_rlhf/algorithms/ppo_lag/trainer.py#L289-L309)

```python
def actor_loss_fn(self, log_probs, old_log_probs,
                  reward_advantages, cost_advantages, mask):
    # 把 log_lambda 取出当前值（detach，不让 actor 更新乘子）
    multiplier = self.log_lambda.exp().item()

    # 核心：双 advantage 组合（注意分母 1+lambda 的归一化）
    # WHY: 没有分母时，lambda 增大会让 advantages 整体放大，
    #      触发 PPO 的 clip 机制把 surrogate 截断，反而抑制学习
    advantages = (reward_advantages - multiplier * cost_advantages) / (1.0 + multiplier)

    # 标准 PPO clip ratio
    ratios = torch.exp(log_probs - old_log_probs)
    surrogate1 = advantages * ratios
    surrogate2 = advantages * torch.clamp(
        ratios, 1.0 - self.clip_range_ratio, 1.0 + self.clip_range_ratio,
    )
    surrogate = torch.minimum(surrogate1, surrogate2)
    return -masked_mean(surrogate, mask)
```

**为什么是减号而不是加号**：cost advantage 是"伤害"维度的优势（cost 越大越糟），所以减去它意味着"鼓励降低 cost"。这与 reward advantage 的方向相反，因此用一个减号显式表达。

### 片段 4：rollout 阶段同时取 reward 和 cost

来源：[`safe_rlhf/algorithms/ppo_lag/trainer.py:181-209`](https://github.com/PKU-Alignment/safe-rlhf/blob/main/safe_rlhf/algorithms/ppo_lag/trainer.py#L181-L209)

```python
# rollout 完成后，同时 forward 四个模型：actor / ref / reward / cost
# WHY: 这是 Safe RLHF 比标准 PPO 多出来的开销——多一次 cost 前向 + cost critic 前向
logits = self.actor_model(sequence, attention_mask=attention_mask).logits
ref_logits = self.actor_reference_model(sequence, attention_mask=attention_mask).logits

# reward 和 cost 都取序列末尾的 end_score（生成完整回复的总评）
reward = self.reward_model(reward_seq, attention_mask=reward_attention_mask).end_scores
cost = self.cost_model(cost_seq, attention_mask=cost_attention_mask).end_scores
# 两个 critic 给每个 token 一个 value 预测（用于 GAE）
reward_values = self.reward_critic_model(sequence, attention_mask=attention_mask).scores
cost_values = self.cost_critic_model(sequence, attention_mask=attention_mask).scores

# 维护一个滑动窗口的 episode_costs，给 lambda 更新用
# WHY: 单 step cost 噪声大，用 deque(maxlen=128) 平滑后才适合做 dual ascent
self.episode_costs.extend(cost.tolist())

return {
    'reward': reward, 'cost': cost,                    # 标量评分
    'reward_values': reward_values, 'cost_values': cost_values,  # token 级 value
    # ...（log_probs / ref_log_probs / input_ids 等）
}
```

**显存口径**：actor + ref + reward + cost + reward_critic + cost_critic = 6 份 7B 模型权重。其中 ref / reward / cost 是 frozen（只 forward 不反传），剩下三个 trainable。这就是为什么 Safe RLHF 默认配 ZeRO-3 + offload 跑——单纯比标准 PPO 多 30-40% 显存。

### 片段 5：Safety Preference 的双视角 collator

来源：[`safe_rlhf/datasets/safety_preference.py:107-137`](https://github.com/PKU-Alignment/safe-rlhf/blob/main/safe_rlhf/datasets/safety_preference.py#L107-L137)

```python
class SafetyPreferenceCollator(CollatorBase):
    def __call__(self, samples):
        # WHY: 同一个 batch 里要同时跑 safer 和 unsafer 两条样本
        #      把它们拼成 2*B，一次 forward 算两个 cost，再 chunk 回去
        input_ids = [s['safer_input_ids'] for s in samples] + \
                    [s['unsafer_input_ids'] for s in samples]
        # safety_sign: +1 safe / -1 unsafe，喂给 cost loss 的 sign 项
        safety_sign = [s['safer_sign'] for s in samples] + \
                      [s['unsafer_sign'] for s in samples]
        # 右 padding，cost head 取最后非 pad token 的得分
        input_ids = right_padding(input_ids, padding_value=self.pad_token_id)
        safety_sign = torch.tensor(safety_sign, dtype=torch.long)

        # 拆回 (B, L) 两组：safer / unsafer
        safer_input_ids, unsafer_input_ids = input_ids.chunk(2, dim=0)
        safer_safety_sign, unsafer_safety_sign = safety_sign.chunk(2, dim=0)
        return {
            'safer_input_ids': safer_input_ids,
            'unsafer_input_ids': unsafer_input_ids,
            'safer_safety_sign': safer_safety_sign,
            'unsafer_safety_sign': unsafer_safety_sign,
            # ...（attention_mask 类似处理，省略）
        }
```

**与普通 PreferenceCollator 的区别**：传 `safety_sign` 而非简单的 `chosen/rejected` 标签——cost loss 需要绝对安全标签做 sign 项。这就是为什么 Safe RLHF 必须有 `is_response_X_safe` 这个独立标注字段。

### 9. 训练日志该看哪些信号

PPO-Lagrangian 训练时关键的 7 条曲线（来自 `rl_step` 返回的字典，最终落到 wandb 上）：

| 指标 | 健康趋势 | 异常诊断 |
|------|---------|---------|
| `train/lambda` | 在 1.0 附近震荡，幅度 0.5-3.0 | 单调上升顶到 `lambda_max` → cost model 偏置或 policy 真的太不安全 |
| `train/episode_cost` | 接近 `threshold`（默认 0） | 持续 > 0 → policy 没在学安全；持续 << 0 → cost 信号失真 |
| `train/reward` | 缓慢上升 | 持续下降 + lambda 高 → 安全过度抑制了有用 |
| `train/cost` | 接近 0 | 与 episode_cost 不一致 → 滑窗太大或太小 |
| `train/reward_advantage` | 接近 0 (GAE 已消方差) | 持续大幅偏移 → critic 没收敛 |
| `train/cost_advantage` | 接近 0 | 同上，但要分别看 cost critic loss |
| `train/kl_divergence` | 缓慢上升但有界 | 失控上升 → kl_coeff 太小或 reward hacking |

健康的 PPO-Lag 训练曲线大致长这样：lambda 在前 200 step 有一次小爆发（policy 探索不安全行为），随后 cost 被压下来，lambda 回落到 1 附近，reward 缓慢上升。

### 10. 与 DPO 的兼容性：Safe DPO 是否可行？

直觉上 DPO 不需要显式的 reward model，那能不能也做 Safe DPO？仓库里 [`safe_rlhf/algorithms/dpo`](https://github.com/PKU-Alignment/safe-rlhf/tree/main/safe_rlhf/algorithms/dpo) 给的实现仍然只用单一 helpful 偏好——因为 DPO 的封闭解形式难以直接套约束优化（lambda 没有 dual ascent 的目标）。学界后续工作（如 SafeDPO、C-DPO）通过把约束写成偏好数据的重加权来近似，但效果离 PPO-Lag 仍有差距。这反过来说明：**约束 RL 不是免费午餐**——在线乘子更新这件事需要 RL 训练循环本身做载体。



### 1. 拉格朗日乘子 lambda 不收敛会怎样？

如果 cost 信号有偏（cost model 系统性高估某类 prompt 的 cost），lambda 会持续单调上升 → reward 信号被完全压制 → policy 退化为"什么都不答"（拒答倾向）。**线索**：训练日志里 `train/lambda` 上界顶到 `lambda_max` 不下来，同时 `train/reward` 持续下降。**修复**：

- 检查 cost model 在 SFT 模型采样上的 cost 分布是否大致以 0 为中心；
- 如果偏正，可以临时调高 `threshold`（论文默认 0）让 cost 预算更宽松；
- 或者降低 `lambda_lr` / 增大 `episode_cost_window_size` 让信号更平滑。

### 2. 双 preference 标注成本翻倍，真的值得吗？

值不值得取决于**冲突样本占比**。如果你的场景里 helpful 与 safe 偏好高度一致（比如纯客服问答），单 RM 就够了。但在通用助手场景，论文报告 BeaverTails 上有 **~25% 的样本两种偏好不一致**——这部分样本恰好是单 RM 学不到的"权衡知识"。**判断标准**：先用少量样本（~1000 条）做双标注实验，看冲突率，再决定是否全量上 dual preference。

### 3. cost model 失效了会怎样？怎么发现？

cost model 失效有两种典型方式：

- **校准失效**：cost 零点漂移，导致大量"明显安全"的回复被打成正 cost。表现：训练初期 lambda 就快速上升。
- **泛化失效**：cost model 在 OOD prompt 上几乎随机。表现：`train/cost_advantage` 接近 0，方差也低。

**诊断**：每隔 N 步在固定的 sanity prompt 集（一半安全、一半不安全）上算 cost，看分布是否仍以 0 为分界。如果偏离严重，要么停下重训 cost model，要么用 `episode_cost` 滑动平均做事后校准。

### 4. 为什么不直接 multi-objective RL（Pareto 前沿）？

理论上更"对"，但工程上太贵——Pareto 前沿优化要么需要标量化（回到加权法），要么需要多次训练扫超参，单次成本是 PPO-Lag 的 N 倍。Safe RLHF 用拉格朗日松弛在"灵活性 vs 成本"之间取了一个甜点：用单次训练近似一个 Pareto 点，由 `threshold` 参数控制你想取哪一个。

### 5. 把 cost 当作"约束"而不是"负 reward"，本质上买到了什么？

买到的是**对 reward shaping 的解耦**。如果你直接用 `R - alpha·C` 训单 RM，alpha 是 ad-hoc 超参；改 alpha 要重新训 RM。Safe RLHF 把这个超参换成了"在线学到的 lambda"，并且这个 lambda 与人能理解的物理量（"当前 batch cost 是否超过阈值"）直接挂钩——**可调试性**比纯黑箱加权强一个量级。

## 面试考点

### 1. 为什么 cost loss 要用 sigmoid 加 sign 项，而不是像 reward 那样只用 BT margin loss？

reward 是相对量，零点无意义；cost 的零点必须代表"恰好临界安全"，因为 PPO-Lagrangian 的 `threshold=0` 依赖这个物理含义。`-logsigmoid(sign · C)` 把"safe→C<0、unsafe→C>0"作为额外监督信号，钉死了零点。少了它，cost model 可以学出"全部样本 cost 都 > 0"的退化解，lambda 永远上升，训练崩溃。

### 2. lambda 在数值上不稳定怎么办？工程上有哪几招？

四招组合拳：

1. **对数空间存 lambda**（`log_lambda` 而不是 `lambda`），天然保证正值；
2. **lambda_max 上界**，防止 lambda 爆炸吞掉 reward 信号；
3. **lambda_update_delay_steps**，让 actor / critic 先 warmup 再更新乘子；
4. **滑动平均 cost**（`episode_cost_window_size=128`），平滑单 step 噪声；
5. **lambda 只在 rank0 更新 + broadcast**，避免分布式训练下 lambda 在不同卡发散。

### 3. PPO-Lagrangian 与 LLaMA 2 双 RM 加权法相比，多/少了什么？

**多**：cost model（一次额外训练 + dual preference 标注）+ cost critic（PPO 阶段多一个 value head + 多一份显存）。
**少**：手调阈值的实验循环（`safety_score < 0.15` 那个 0.15 不再需要），以及"切换式 reward 在边界处不可导"的工程麻烦。
**核心 trade-off**：用一次性的工程投入（多训一个模型），换长期的可调性（lambda 自适应）。

### 4. 双 advantage 组合公式 `(A_R - lambda·A_C) / (1 + lambda)` 的分母是干嘛的？

归一化 advantage 量级。如果只写 `A_R - lambda·A_C`，当 lambda 从 1 涨到 5，advantage 数值放大 ~5 倍，PPO 的 clip ratio（0.2）相对就变小了，等价于变相缩小学习率。除以 `(1 + lambda)` 把 advantage 维持在与原始 reward advantage 同一量级，让 PPO 的超参（clip_range_ratio、actor_lr）在不同 lambda 下保持一致行为。

### 5. 如果只有一份 helpfulness 标注（没有 safer 标注），还能用 PPO-Lagrangian 吗？

不能直接用，因为 cost model 训练需要 `is_safe` 绝对标签和 `safer` 偏好。**变通方案**：

- 用 LLM-as-judge 自动给已有 (chosen, rejected) 数据补打安全标签（精度有限但可启动）；
- 走 [Constitutional AI 路径](/training/alignment#constitutional-ai)，用宪法 + AI 自我批判生成 harmlessness 偏好；
- 或者退一步用 LLaMA 2 风格的双 RM 静态切换，安全数据规模要求更小。

## 推荐资源

| 资源 | 链接 | 内容 |
|------|------|------|
| **源仓库** | [PKU-Alignment/safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf) | PPO-Lag / DPO / Reward + Cost Model 的完整实现，Apache 2.0 |
| **Safe RLHF 论文** | [arXiv 2310.12773](https://arxiv.org/abs/2310.12773) | Dai et al. 2023，PPO-Lagrangian 应用于 LLM 对齐的原始论文 |
| **BeaverTails 论文** | [arXiv 2307.04657](https://arxiv.org/abs/2307.04657) | dual-preference 数据集的标注 schema 与统计分析 |
| **PPO-Lagrangian 原论文** | [arXiv 2007.03964](https://arxiv.org/abs/2007.03964) | Stooke et al. 2020，约束 RL 中 lambda 对数空间更新的来源 |
| **PKU-SafeRLHF 数据集** | [HuggingFace](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) | 30 万条 dual preference，14 个伤害类别 |
| **本站相关文章** | [偏好对齐](/training/alignment) · [安全工程](/engineering/safety) · [手搓 RLHF](/deep-dives/nano-rlhf) | 宏观 RLHF 谱系 + 安全对齐工程化 + 纯 PyTorch 实现 |

::: tip 学习路径建议
1. 先读 [偏好对齐](/training/alignment) 了解 RLHF / DPO / GRPO 与 LLaMA 2 双 RM 加权法；
2. 跑一遍 [手搓 RLHF Pipeline](/deep-dives/nano-rlhf)，把 PPO 的 GAE 与 clipped loss 在脑子里跑通；
3. 再读本文 + Safe RLHF 论文，把"约束"维度叠加上去；
4. 最后到 [安全工程](/engineering/safety) 把训练时对齐与推理时防御拼成完整图。
:::
