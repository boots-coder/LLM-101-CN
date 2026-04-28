---
title: RLHF Pipeline 实现挑战
description: Level 4 端到端实现：SFT 微调、Reward Model 训练、PPO 优化（含 KL、value head、GAE）
topics: [build, RLHF, PPO, reward-model, SFT, value-head, GAE, KL-penalty, end-to-end]
---
# RLHF Pipeline 实现挑战 (Level 4)

> **难度:** 困难（本系列最高难度之一） | **前置知识:** Transformer / GPT 训练熟练、PPO 公式（建议先做完 [ppo-fill.md](./ppo-fill.md)） | **预计时间:** 6-10 小时

## 挑战目标

从零搭一条**端到端可跑**的 minimal RLHF 流水线：SFT → Reward Model → PPO。完成后你应该能在一个 toy 任务（让模型生成「积极情绪」的句子续写）上观察到：

- SFT 阶段：模型 loss 收敛，生成结果跟随指令
- RM 阶段：在 chosen / rejected 偏好对上 accuracy > 70%
- PPO 阶段：reward 单调上升，KL 维持在合理范围（不崩坏）

不允许调用 `trl.PPOTrainer` / `trl.SFTTrainer` 等高层封装，所有核心逻辑（loss、value head、GAE、PPO update）必须自己写。底层模型可以用 HuggingFace 的 `GPT2LMHeadModel` 加载预训练权重作为 backbone。

::: warning 难度提示
RLHF 实现的痛点不在「公式」（看起来都很简单），而在**三个阶段之间的数据流**：什么是 frozen 的、什么是要更新的、token 索引怎么对齐 prompt / response 边界、reward / advantage / return 在哪里计算……这些细节错一个就训不动。建议**逐阶段验证 + 频繁打印 shape**。
:::

---

## 三阶段总览

```
                      ┌─────────────────────────────────────────────┐
                      │         RLHF Pipeline (本练习)               │
                      └─────────────────────────────────────────────┘

阶段 1 SFT                阶段 2 RM                  阶段 3 PPO
─────────────────         ────────────────────       ────────────────────────────
 prompt + response         (prompt, chosen,           prompt-only batch
        │                  rejected)                       │
        ▼                       │                          ▼ rollout (sample)
 GPT-2 small                    ▼                   ┌──────────────────┐
        │                  GPT-2 + scalar head      │  policy (π_θ)    │ ←── update
   masked CE loss          ──────────────────       │  ref policy (π_0)│ ←── frozen
   (只算 response 部分)    BT loss:                 │  value head V_φ  │ ←── update
        │                  -log σ(r_c − r_r)        │  reward model RM │ ←── frozen
        ▼                       │                   └──────────────────┘
   π_SFT （初始策略）            ▼                          │
        │                  RM_φ                            ▼
        │                       │                    rollout 张量
        │                       │                          │
        └────────────────►──────┴──────────►───────────────▼
                                                  reward shaping (KL+RM)
                                                          │
                                                          ▼
                                                     compute GAE
                                                          │
                                                          ▼
                                              PPO inner update (K epochs)
                                                  ratio / clip / value
                                                          │
                                                          ▼
                                                  π_θ ← updated
```

整条 pipeline 涉及四个模型实例（同一份 backbone 的不同副本）：

| 角色 | 是否更新 | 来源 | 用途 |
|------|---------|------|------|
| `policy`（π_θ） | ✅ | SFT 输出 + value head | rollout 采样 + PPO 优化 |
| `ref_policy`（π_0） | ❌ | SFT 输出（frozen） | 计算 KL 惩罚 |
| `reward_model`（RM_φ） | ❌ | RM 训练输出（frozen） | 给 response 打分 |
| `value`（V_φ） | ✅ | 与 policy 共享 backbone + scalar head | 估计状态价值 |

---

## 环境与依赖

```python
# requirements.txt（建议）
torch>=2.0
transformers>=4.40
datasets>=2.14
numpy
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import math
import copy
```

::: tip Backbone 选择
本练习用 `gpt2`（124M 参数）作为 backbone。如果你的显存极小（< 8GB），可以换成 `sshleifer/tiny-gpt2`（toy 玩具，但能跑通流程）。挑战的目的是**理解 pipeline**，不是刷分。
:::

---

## 阶段 1：SFT 微调

### 1.1 任务说明

输入是 **指令 + 响应** 拼接成一条文本，训练时**只对 response 部分算 loss**——prompt 部分的 token 不参与梯度（mask = 0）。这是 SFT 与裸预训练唯一的本质区别。

```
input_ids:  [BOS, prompt_t1, ..., prompt_tn, response_t1, ..., response_tm, EOS]
labels:     [-100, -100,    ...,  -100,     response_t1, ..., response_tm, EOS]
                                            ↑ 只有这一段进 cross_entropy
```

PyTorch 约定 `label = -100` 时 `F.cross_entropy(..., ignore_index=-100)` 自动跳过这些位置。

### 1.2 数据准备

为了让你能在本机跑通，提供一个 toy 「积极情绪续写」数据集（也可以替换成 `Anthropic/hh-rlhf` / `Dahoas/rm-static` 等公开数据）：

```python
TOY_SFT_DATA = [
    {"prompt": "今天的天气", "response": "真好，阳光明媚让人心情愉悦。"},
    {"prompt": "这部电影", "response": "情节扣人心弦，演员表演令人惊艳。"},
    {"prompt": "新发布的手机", "response": "性能强劲，拍照效果非常出色。"},
    {"prompt": "周末和朋友", "response": "一起去公园野餐，度过了愉快的一天。"},
    {"prompt": "刚学完的课程", "response": "内容充实，让我收获了许多新知识。"},
    # ... 实际练习时建议扩充到 100+ 条
]
```

```python
class SFTDataset(Dataset):
    """
    每条样本返回：
        input_ids: LongTensor [T]
        labels:    LongTensor [T]，prompt 部分填 -100
    """

    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tok = tokenizer
        self.max_len = max_len
        # 注意：GPT-2 默认没有 pad_token，借用 eos
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data[idx]["prompt"]
        response = self.data[idx]["response"]

        # TODO 1: 分别 tokenize prompt 和 response（不加 special token）
        # 提示: self.tok(prompt, add_special_tokens=False).input_ids
        prompt_ids = ...          # type: list[int]
        response_ids = ...        # type: list[int]

        # TODO 2: 拼接 input_ids = prompt + response + [eos]
        eos_id = self.tok.eos_token_id
        input_ids = ...

        # TODO 3: 构造 labels：prompt 段填 -100，response + eos 保留
        # 提示: labels = [-100] * len(prompt_ids) + response_ids + [eos_id]
        labels = ...

        # TODO 4: 截断 / pad 到 max_len
        # 注意：pad 部分的 label 也要置 -100，避免污染 loss
        ...

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1 if x != self.tok.pad_token_id else 0 for x in input_ids],
                dtype=torch.long,
            ),
        }
```

### 1.3 训练循环

```python
def train_sft(model, dataloader, n_epochs=3, lr=5e-5, device="cuda"):
    """
    SFT 训练循环。
    关键点：用 model 自带的 loss（HF GPT2LMHeadModel forward 时传 labels 会自动算 CE）。
    """
    model.to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for step, batch in enumerate(dataloader):
            # TODO 1: 把 batch 上 device
            input_ids = ...
            labels = ...
            attn_mask = ...

            # TODO 2: forward，HF 会自动用 ignore_index=-100 跳过 prompt 段
            # 提示: out = model(input_ids=..., attention_mask=..., labels=...)
            #       loss = out.loss
            out = ...
            loss = out.loss

            # TODO 3: backward + clip + step + zero_grad
            ...

            if step % 10 == 0:
                print(f"[SFT] epoch={epoch} step={step} loss={loss.item():.4f}")

    return model
```

### 1.4 验证

```python
def sanity_check_sft(model, tokenizer, prompt, device="cuda"):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=20, do_sample=True, top_p=0.9, temperature=0.8,
        )
    print(f"prompt: {prompt}")
    print(f"output: {tokenizer.decode(out[0], skip_special_tokens=True)}")


# 用例
sanity_check_sft(sft_model, tokenizer, "新发布的手机")
# 期望输出大致是「正面、流畅」的续写，比如「性能不错，外观也很好看」
```

::: tip 本阶段评分
- ✅ loss 在 3 个 epoch 内从 ~5 下降到 ~2 以下
- ✅ 生成结果**遵循指令**（不只是预训练时的随机续写）
- ✅ prompt 段的 label 确实是 -100（自己 print 一条样本检查）
:::

---

## 阶段 2：Reward Model 训练

### 2.1 任务说明

**输入**：一对 `(prompt, chosen, rejected)`，其中 chosen 是人类偏好的回答，rejected 是较差的回答。

**模型**：在 GPT-2 backbone 上加一个 **scalar head**：

```
hidden_states: [B, T, hidden]
                ↓ 取最后一个有效 token 的 hidden
reward:        [B]
```

**Loss**（Bradley-Terry）：

$$
\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_c, y_r)\sim D}\left[\log\sigma\left(r_\phi(x, y_c) - r_\phi(x, y_r)\right)\right]
$$

直觉：让 chosen 的分数尽可能高于 rejected 的分数，差距越大 loss 越小。

### 2.2 RewardModel 实现

```python
class RewardModel(nn.Module):
    """
    GPT-2 backbone + scalar head。
    forward 返回 [B] 的标量 reward（取最后一个非 pad token 的 hidden 投影）。
    """

    def __init__(self, base_model_name="gpt2"):
        super().__init__()
        # TODO 1: 加载 backbone（注意只要 transformer 部分，不要 LM head）
        # 提示: from transformers import GPT2Model
        #       self.backbone = GPT2Model.from_pretrained(base_model_name)
        self.backbone = ...

        # TODO 2: 定义 scalar head
        # 提示: nn.Linear(hidden_size, 1)，hidden_size = self.backbone.config.n_embd
        self.v_head = ...

    def forward(self, input_ids, attention_mask):
        """
        参数:
            input_ids:      [B, T]
            attention_mask: [B, T]，1=valid, 0=pad
        返回:
            reward: [B]
        """
        # TODO 3: 走 backbone 得到 last_hidden_state [B, T, H]
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state              # [B, T, H]

        # TODO 4: scalar head 投影 → [B, T, 1] → squeeze → [B, T]
        rewards_per_token = ...                     # [B, T]

        # TODO 5: 取每条样本「最后一个 valid token」的 reward
        # 提示: last_idx = attention_mask.sum(-1) - 1     # [B]
        #       batch_idx = torch.arange(B)
        #       reward = rewards_per_token[batch_idx, last_idx]
        last_idx = ...
        reward = ...                                # [B]
        return reward
```

### 2.3 数据 + 损失

```python
TOY_RM_DATA = [
    {
        "prompt": "今天的天气",
        "chosen": "真好，阳光明媚让人心情愉悦。",
        "rejected": "糟糕透顶，让人想立刻回家。",
    },
    {
        "prompt": "这部电影",
        "chosen": "情节扣人心弦，演员表演令人惊艳。",
        "rejected": "毫无亮点，看完后只想退票。",
    },
    # ... 实际建议 200+ 条
]


class RMDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tok = tokenizer
        self.max_len = max_len
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __getitem__(self, idx):
        item = self.data[idx]
        # TODO: 分别拼接 chosen 序列 = prompt + chosen，rejected 序列 = prompt + rejected
        # 然后 tokenize + pad 到 max_len
        chosen_ids, chosen_mask = ...
        rejected_ids, rejected_mask = ...

        return {
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "chosen_mask": torch.tensor(chosen_mask, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "rejected_mask": torch.tensor(rejected_mask, dtype=torch.long),
        }

    def __len__(self):
        return len(self.data)


def rm_loss(r_chosen, r_rejected):
    """
    Bradley-Terry loss = -log_sigmoid(r_chosen - r_rejected)
    返回:
        loss:     scalar
        accuracy: scalar (chosen > rejected 的比例，用于监控)
    """
    # TODO 1: 用 F.logsigmoid 计算逐样本 loss
    # 提示: -F.logsigmoid(r_chosen - r_rejected)
    loss = ...

    # TODO 2: 计算 accuracy = (r_chosen > r_rejected).float().mean()
    accuracy = ...

    return loss.mean(), accuracy
```

### 2.4 训练循环

```python
def train_rm(rm_model, dataloader, n_epochs=2, lr=1e-5, device="cuda"):
    rm_model.to(device).train()
    optim = torch.optim.AdamW(rm_model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for step, batch in enumerate(dataloader):
            # TODO 1: 分别 forward chosen 和 rejected，得到 r_chosen / r_rejected [B]
            r_chosen = ...
            r_rejected = ...

            # TODO 2: 计算 BT loss + accuracy
            loss, acc = rm_loss(r_chosen, r_rejected)

            # TODO 3: backward + clip + step + zero_grad
            ...

            if step % 10 == 0:
                print(f"[RM] epoch={epoch} step={step} loss={loss.item():.4f} acc={acc.item():.3f}")

    return rm_model
```

### 2.5 验证

```python
@torch.no_grad()
def sanity_check_rm(rm_model, tokenizer, prompt, good, bad, device="cuda"):
    rm_model.eval()
    enc_good = tokenizer(prompt + good, return_tensors="pt", padding=True).to(device)
    enc_bad = tokenizer(prompt + bad, return_tensors="pt", padding=True).to(device)
    r_good = rm_model(**enc_good)
    r_bad = rm_model(**enc_bad)
    print(f"prompt: {prompt}")
    print(f"  good: '{good}' → r={r_good.item():+.3f}")
    print(f"  bad : '{bad}'  → r={r_bad.item():+.3f}")
    assert r_good.item() > r_bad.item(), "RM 给好回答的分数应该更高！"
    print("RM sanity check 通过")


sanity_check_rm(
    rm_model, tokenizer,
    prompt="今天的天气",
    good="真好，阳光明媚让人心情愉悦。",
    bad="糟糕透顶，让人想立刻回家。",
)
```

::: tip 本阶段评分
- ✅ RM 训练 accuracy > 70%（toy 数据上更高，公开数据集 65-75% 是正常的）
- ✅ sanity check 中 chosen 分数高于 rejected
- ✅ 你能解释**为什么取「最后一个 valid token」的 hidden 而不是均值池化**（提示：自回归模型的因果性）
:::

::: warning 替代方案
如果手动训 RM 太慢，**端到端验证阶段**可以用一个预训练的 sentiment classifier（如 `distilbert-base-uncased-finetuned-sst-2-english`）当 RM 替代——把「预测正面情绪的概率」作为 reward。这是 RLHF 论文里也常用的 toy setup。本练习强烈建议第一遍先用这个替身把 PPO 流程跑通，再回头训自己的 RM。
:::

---

## 阶段 3：PPO 优化

这是本练习最复杂的部分。我们把它拆成 5 个子任务。

### 3.1 子任务：实现 Value Head（共享 backbone + 新 head）

PPO 需要一个 critic（价值网络）。最常见的设计是 **policy 和 value 共享 backbone**，再各自接一个独立的 head：

```
input_ids
    ↓
GPT-2 backbone (shared)
    ↓ hidden [B, T, H]
    ├──→ lm_head    → logits [B, T, V]   (policy)
    └──→ value_head → values [B, T]      (critic)
```

```python
class PolicyWithValueHead(nn.Module):
    """
    backbone 共享的 actor-critic 架构。
    policy_forward 返回 logits，value_forward 返回 V(s)。
    """

    def __init__(self, base_model_name="gpt2"):
        super().__init__()
        # TODO 1: 加载完整的 GPT2LMHeadModel（含 lm_head）
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)

        # TODO 2: 定义 value_head: hidden_size → 1
        # 提示: nn.Linear(self.lm.config.n_embd, 1)
        # 进阶: 加一层 MLP（GELU 中间层）通常更稳
        self.value_head = ...

    def forward(self, input_ids, attention_mask=None):
        """
        一次 forward 同时拿到 logits 和 values（避免重复跑 backbone）。
        返回:
            logits: [B, T, V]
            values: [B, T]
        """
        # TODO 3: 拿到 backbone 的 hidden states
        # 提示: 用 output_hidden_states=True，hidden_states[-1] 是最后一层
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = out.logits                              # [B, T, V]
        last_hidden = out.hidden_states[-1]              # [B, T, H]

        # TODO 4: value_head(last_hidden).squeeze(-1) → [B, T]
        values = ...

        return logits, values

    def load_from_sft(self, sft_model_state_dict):
        """从 SFT 输出加载 backbone+lm_head 权重，value_head 随机初始化。"""
        # 注意：load_state_dict(strict=False) 允许 value_head 不匹配
        self.lm.load_state_dict(sft_model_state_dict, strict=True)
```

### 3.2 子任务：计算 token-level log probability

PPO 的核心张量是「每个生成 token 在 policy 下的 log prob」——通过 `gather + log_softmax` 计算。

```python
def gather_logprobs(logits, input_ids):
    """
    给定 logits 和实际 token，返回每个位置上「生成该 token 的 log prob」。

    参数:
        logits:    [B, T, V]   模型 forward 输出
        input_ids: [B, T]      实际的 token id

    返回:
        logprobs:  [B, T-1]    第 t 位置 = log P(token_{t+1} | token_{0..t})
                               注意 shape 少了 1，因为最后一个位置没有 next token 监督

    """
    # TODO 1: shift 让 logits[:, t, :] 预测 input_ids[:, t+1]
    # 提示: logits[:, :-1, :] 对齐 input_ids[:, 1:]
    shift_logits = ...                               # [B, T-1, V]
    shift_ids = ...                                  # [B, T-1]

    # TODO 2: log_softmax 沿 vocab 维
    log_probs = ...                                  # [B, T-1, V]

    # TODO 3: 用 gather 取出实际 token 的 log prob
    # 提示: log_probs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
    selected = ...                                   # [B, T-1]

    return selected
```

::: tip Shape 对齐口诀
PPO 中所有「per-token」量（logprobs / values / advantages / rewards）都应该是**同一个 shape**——通常是 `[B, response_len]`，对应于 response 段的每个 token。Prompt 段的 token 不参与梯度（既不是动作也不是状态价值），rollout 时直接 mask 掉。
:::

### 3.3 子任务：实现 GAE

复用 [ppo-fill.md 练习 2](./ppo-fill.md#练习-2-gae-倒序递推-level-2) 的逻辑，但要注意**这里的 rewards / values shape 是 `[B, response_len]`**（只覆盖 response 段）。

```python
def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """
    GAE 倒序递推。
    参数:
        rewards: [B, T]   token 级 reward（已经做过 KL shaping）
        values:  [B, T]   critic 估计
        mask:    [B, T]   1 表示该 token 在 response 内
        gamma:   折扣因子（语言任务通常 1.0）
        lam:     GAE λ

    返回:
        advantages, returns: [B, T]，detach 过
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, device=rewards.device)

    # TODO 1: 倒序循环 t in [T-1, T-2, ..., 0]
    for t in reversed(range(T)):
        # TODO 2: next_value（边界为 0）
        next_value = values[:, t + 1] if t + 1 < T else torch.zeros(B, device=values.device)

        # TODO 3: TD 残差 δ_t = r_t + γ V_{t+1} - V_t
        delta = ...

        # TODO 4: GAE 递推 A_t = δ_t + γλ A_{t+1}
        gae = ...

        advantages[:, t] = gae

    advantages = advantages * mask
    returns = advantages + values
    return advantages.detach(), returns.detach()
```

### 3.4 子任务：实现 Clipped Surrogate + Value Loss

```python
def ppo_losses(
    new_logprobs, old_logprobs, advantages,
    new_values, old_values, returns,
    mask, clip_range=0.2, clip_range_value=0.2, vf_coef=0.5,
):
    """
    PPO 总损失（policy clipped + value clipped）。

    参数:
        new_logprobs, old_logprobs: [B, T]
        advantages, returns:        [B, T]
        new_values, old_values:     [B, T]
        mask:                       [B, T]

    返回:
        total_loss: scalar
        info:       dict（policy_loss / value_loss / kl / clipfrac）
    """
    # ---- Policy loss ----
    # TODO 1: ratio = exp(new - old)，padding 位置补 1
    ratio = ...

    # TODO 2: unclipped = -A * ratio；clipped = -A * clamp(ratio, 1-ε, 1+ε)
    unclipped = ...
    clipped = ...

    # TODO 3: policy_loss_per_tok = max(unclipped, clipped)
    policy_loss_per_tok = ...
    policy_loss = (policy_loss_per_tok * mask).sum() / mask.sum().clamp(min=1)

    # ---- Value loss (clipped) ----
    # TODO 4: v_clipped = old + clamp(new - old, -ε_v, +ε_v)
    v_clipped = ...

    # TODO 5: max((new_v - returns)^2, (v_clipped - returns)^2) * 0.5
    v_loss_per_tok = ...
    value_loss = (v_loss_per_tok * mask).sum() / mask.sum().clamp(min=1)

    # ---- 监控 ----
    with torch.no_grad():
        approx_kl = ((new_logprobs - old_logprobs) * mask).sum() / mask.sum().clamp(min=1)
        clipfrac = (((ratio - 1.0).abs() > clip_range).float() * mask).sum() / mask.sum().clamp(min=1)

    total_loss = policy_loss + vf_coef * value_loss

    return total_loss, {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "approx_kl": approx_kl.item(),
        "clipfrac": clipfrac.item(),
    }
```

### 3.5 子任务：完整 PPO Loop

把以上组件拼成主循环：

```
┌──────────────────────────── PPO 一个 outer step ─────────────────────────────┐
│                                                                              │
│  1. ROLLOUT（policy 是 frozen 的 old policy）                                 │
│     for prompt in batch:                                                     │
│         response = policy.generate(prompt, do_sample=True)                   │
│         old_logprobs, old_values = forward_with_old_policy(prompt+response)  │
│         ref_logprobs              = forward_with_ref_policy(prompt+response) │
│         rm_score                  = reward_model(prompt+response)            │
│                                                                              │
│  2. REWARD SHAPING                                                           │
│     token_rewards = -kl_coef * (old_logprobs - ref_logprobs)                 │
│     token_rewards[last_idx] += rm_score                                      │
│                                                                              │
│  3. ADVANTAGE                                                                │
│     advantages, returns = compute_gae(token_rewards, old_values, mask)       │
│     # 标准化 advantage（PPO 标准 trick）                                      │
│     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)│
│                                                                              │
│  4. INNER UPDATE（K 个 epoch，复用同一份 rollout）                            │
│     for _ in range(ppo_epochs):                                              │
│         new_logprobs, new_values = policy.forward(prompt + response)         │
│         loss = ppo_losses(...)                                               │
│         loss.backward(); clip_grad; step; zero_grad                          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

```python
@torch.no_grad()
def rollout(policy, ref_policy, reward_model, tokenizer, prompts,
            max_new_tokens=20, device="cuda"):
    """
    采样一批 response，并返回 PPO 所需的所有 frozen 张量。
    """
    policy.eval()

    # TODO 1: 用 policy.lm.generate(...) 采样 response
    # 提示: tokenizer batch encode prompts，然后 generate(do_sample=True)
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    prompt_len = enc["input_ids"].size(1)
    full_ids = policy.lm.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=True,
        top_p=0.9, temperature=1.0, pad_token_id=tokenizer.pad_token_id,
    )                                                # [B, prompt_len + new_len]
    full_mask = (full_ids != tokenizer.pad_token_id).long()

    # TODO 2: forward 一次 policy 拿 old_logprobs / old_values
    logits, values = policy(full_ids, attention_mask=full_mask)
    old_logprobs = gather_logprobs(logits, full_ids)        # [B, T-1]

    # values 也需要对齐到「response 段」的 shape
    # 注意：values 是 [B, T]，对应每个位置的 V(s_t)；用 [:, prompt_len-1:-1] 截取 response 段的 V
    # ...细节根据你的实现取索引

    # TODO 3: forward 一次 ref_policy（frozen）拿 ref_logprobs
    ref_logits, _ = ref_policy(full_ids, attention_mask=full_mask)
    ref_logprobs = gather_logprobs(ref_logits, full_ids)

    # TODO 4: reward_model 给整条 response 打分
    rm_score = reward_model(full_ids, attention_mask=full_mask)  # [B]

    # TODO 5: 构造 response_mask（prompt 段为 0，response 段为 1，pad 为 0）
    response_mask = ...                                          # [B, T-1]

    return {
        "full_ids": full_ids,
        "full_mask": full_mask,
        "old_logprobs": old_logprobs,
        "ref_logprobs": ref_logprobs,
        "old_values": values,              # 注意切片对齐
        "rm_score": rm_score,
        "response_mask": response_mask,
    }


def shape_token_rewards(rm_score, old_logprobs, ref_logprobs, response_mask, kl_coef=0.05):
    """
    把 RM 序列分 + KL 揉成 token reward。
    （和 ppo-fill.md 练习 6 一致）
    """
    kl = old_logprobs - ref_logprobs                       # K1 估计
    token_rewards = -kl_coef * kl

    last_idx = response_mask.sum(dim=-1).long() - 1
    batch_idx = torch.arange(rm_score.size(0), device=rm_score.device)
    token_rewards[batch_idx, last_idx] += rm_score

    return token_rewards * response_mask


def ppo_step(policy, optim, rollout_data, ppo_epochs=4, clip_range=0.2,
             clip_range_value=0.2, kl_coef=0.05, vf_coef=0.5, max_grad_norm=1.0):
    """
    一次 PPO outer step：用一份 rollout 数据跑 ppo_epochs 个 inner update。
    """
    policy.train()

    # 1. reward shaping
    token_rewards = shape_token_rewards(
        rollout_data["rm_score"],
        rollout_data["old_logprobs"],
        rollout_data["ref_logprobs"],
        rollout_data["response_mask"],
        kl_coef=kl_coef,
    )

    # 2. GAE
    advantages, returns = compute_gae(
        token_rewards, rollout_data["old_values"], rollout_data["response_mask"],
    )

    # 3. advantage normalization
    adv_mean = (advantages * rollout_data["response_mask"]).sum() / rollout_data["response_mask"].sum().clamp(min=1)
    adv_var = (((advantages - adv_mean) ** 2) * rollout_data["response_mask"]).sum() / rollout_data["response_mask"].sum().clamp(min=1)
    advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8)

    # 4. inner update
    info_log = []
    for inner in range(ppo_epochs):
        # TODO: forward 一遍 policy（这次是 train mode）→ new_logprobs, new_values
        logits, new_values = policy(rollout_data["full_ids"], rollout_data["full_mask"])
        new_logprobs = gather_logprobs(logits, rollout_data["full_ids"])

        loss, info = ppo_losses(
            new_logprobs, rollout_data["old_logprobs"], advantages,
            new_values, rollout_data["old_values"], returns,
            mask=rollout_data["response_mask"],
            clip_range=clip_range, clip_range_value=clip_range_value, vf_coef=vf_coef,
        )

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optim.step()

        info["mean_reward"] = rollout_data["rm_score"].mean().item()
        info_log.append(info)

    return info_log[-1]
```

::: warning 容易踩的坑
1. **values 的索引对齐**：`values` 是 `[B, T]` 对应每个位置的 V(s_t)，但 GAE 需要的是「response 段每个 token 之后的状态价值」——切片时要错位一格。建议你 print `values.shape` / `old_logprobs.shape` / `response_mask.shape` 确认三者一致。
2. **inner update 时 ref_logprobs / rm_score / advantages 必须是 frozen**：它们都是 rollout 时计算的，和 inner update 解耦。代码里要 `.detach()` 或在 `torch.no_grad()` 下生成。
3. **第一次 inner update 时 `new_logprobs == old_logprobs`**：所以 `ratio=1`、`policy_loss=0` 是正常的——只有从第 2 次 inner update 开始才有非零 ratio。
4. **KL 爆炸**：如果你看到 `approx_kl > 0.1` 持续上涨，要么 lr 太大、要么 `clip_range` 太宽，要么 `kl_coef` 太小。论文典型值：`lr=1e-6 ~ 1e-5`、`clip=0.2`、`kl_coef=0.05`。
:::

---

## 端到端验证

最快 sanity check：用 `distilbert-base-uncased-finetuned-sst-2-english` 这种公开 sentiment classifier 做 RM 替身（替代你自己训的 RM），目标是让 GPT-2 续写**积极情绪的句子**。

```python
def sentiment_reward_model(texts, classifier, tokenizer):
    """
    用预训练的 sentiment classifier 计算「正面情绪概率」作为 reward。
    返回: [B] 标量分（softmax 后的正面类概率）
    """
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(classifier.device)
    with torch.no_grad():
        logits = classifier(**enc).logits           # [B, 2]
        probs = F.softmax(logits, dim=-1)
        positive_prob = probs[:, 1]                 # SST-2 中 1 = positive
    return positive_prob


def end_to_end_demo():
    """
    一个 minimal demo：
    - prompts = ["Today the weather", "This movie is", "The new phone"]
    - 跑 50 个 PPO outer step
    - 比较训练前后 reward 均值 + KL
    """
    # 1. 加载 SFT 后的模型
    policy = PolicyWithValueHead("gpt2").to("cuda")
    policy.load_from_sft(sft_model.state_dict())

    ref_policy = copy.deepcopy(policy).eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    # 2. 加载 sentiment classifier 当 RM
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    rm_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    rm_cls = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    ).to("cuda").eval()

    # 3. PPO 主循环
    optim = torch.optim.AdamW(policy.parameters(), lr=1e-6)
    prompts_pool = ["Today the weather", "This movie", "The new phone"]

    rewards_history = []
    kl_history = []

    for step in range(50):
        prompts = np.random.choice(prompts_pool, 4).tolist()

        # rollout（注意 reward 函数被替换成 sentiment classifier）
        # ... 你需要把 rollout 中的 reward_model(...) 调用改成
        # sentiment_reward_model(decoded_responses, rm_cls, rm_tok)

        rollout_data = rollout(policy, ref_policy, ..., tokenizer, prompts, ...)
        info = ppo_step(policy, optim, rollout_data)

        rewards_history.append(info["mean_reward"])
        kl_history.append(info["approx_kl"])

        if step % 5 == 0:
            print(f"[PPO] step={step} reward={info['mean_reward']:.3f} "
                  f"kl={info['approx_kl']:.4f} pi_loss={info['policy_loss']:.4f}")

    return rewards_history, kl_history
```

### 端到端验证标准

跑完 50 步后，应该观察到：

| 指标 | 训练前 | 训练后 | 是否达标 |
|------|--------|--------|----------|
| 平均 reward（正面概率） | ~0.5 ± 0.2 | > 0.75 | ✅ 显著提升 |
| approx_kl | n/a | < 0.05 | ✅ 没有崩坏 |
| 生成示例 | 中性 / 负面混杂 | 大多正面 | ✅ 主观可见 |

如果 reward 不上升或 KL 爆炸，参考下面的 debug checklist。

::: warning 训练崩了怎么办
- **reward 不上升 + KL ≈ 0**：lr 太小，或 advantage 没有归一化 → 让 advantage 标准化生效
- **reward 短暂上升后断崖下跌**：经典 mode collapse，policy 学到了「reward hacking 的 shortcut」（比如不断重复同一个正面词）→ 增大 `kl_coef`、缩小 `clip_range`
- **value loss 比 policy loss 大几个数量级**：value head 还没收敛 → 跑几步纯 value warmup（policy lr=0），或减小 `vf_coef`
- **生成内容全是 pad / eos**：tokenizer 的 `pad_token_id` 设置或 `attention_mask` 错位 → 检查 rollout 里的 mask 构造
:::

---

## 评分标准

完成挑战后，对照以下 checklist 自评。能勾选 5/7 算通过，7/7 满分。

| # | 检查项 | 通过标准 |
|---|--------|----------|
| 1 | SFT 收敛 | 3 epoch 内 loss < 2.0，生成结果跟随指令 |
| 2 | RM 训练 | accuracy > 70%，sanity check 中 chosen > rejected |
| 3 | Value head 实现 | 与 policy 共享 backbone，单次 forward 同时拿 logits/values |
| 4 | GAE 倒序正确 | 单步测试中 advantage 形状正确、数值非异常 |
| 5 | PPO 损失正确 | clip 行为符合预期，approx_kl 监控正常 |
| 6 | KL 防偏离 | 端到端训练中 approx_kl 维持在 [0, 0.1] 之间 |
| 7 | 端到端见效 | 50 步后 reward 显著上升，生成主观可见正向变化 |

---

## 进阶：把 PPO 替换成 GRPO / DPO

完成 PPO 版本后，尝试以下进阶替换（每个都比 PPO 简单一个数量级）：

### 进阶 A：替换为 DPO（去掉 RM 和 PPO，全部用 SFT-style 训练）

DPO 的核心 insight：把 RM + PPO 两步合并成一个 closed-form loss，**不需要 critic、不需要 rollout**。

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta\log\frac{\pi_\theta(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right)\right]
$$

实现路径：

1. 复用阶段 1 的 SFT 模型作为 `π_ref`（frozen）
2. 复用阶段 2 的 `(prompt, chosen, rejected)` 数据（**不需要训 RM**）
3. 计算 chosen / rejected 的 log probability 之差（policy vs ref）
4. 套 BT loss 直接训练 policy

详见 [dpo-grpo-fill.md](./dpo-grpo-fill.md) 中的 DPO 填空。代码量约为 PPO 的 1/5。

### 进阶 B：替换为 GRPO（去掉 critic，用 group-baseline 估计 advantage）

GRPO 的 insight：对每个 prompt 采样 G 条 response，**用组内 reward 的均值 / 标准差当 baseline**，省掉 value head。

$$
\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}
$$

实现路径：

1. 在阶段 3 的 rollout 里改成「每个 prompt 采 G=4 条 response」
2. 用 group-normalize 替代 GAE（去掉 value head）
3. 把 KL 直接加在 loss 上（用 K3 估计），不再揉进 reward
4. 其他（importance ratio、clip）保持不变

详见 [dpo-grpo-fill.md](./dpo-grpo-fill.md) 的 GRPO 填空。这是 DeepSeek-R1 用的算法。

---

## 常见陷阱总结

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 陷阱 1：SFT 阶段没把 prompt 段 mask 掉                                    │
│   症状: 模型只会重复 prompt，生成质量差                                   │
│   修复: 检查 labels 中 prompt 段是否为 -100                               │
├──────────────────────────────────────────────────────────────────────────┤
│ 陷阱 2：RM 取的不是「最后一个 valid token」的 hidden                       │
│   症状: r_chosen / r_rejected 非常接近，accuracy 不上升                   │
│   修复: 用 attention_mask.sum(-1) - 1 拿到正确索引                        │
├──────────────────────────────────────────────────────────────────────────┤
│ 陷阱 3：PPO inner update 用了 stale advantages                            │
│   症状: 第 1 次 inner update 后 loss 异常                                 │
│   修复: 确保 advantages / returns 在 outer step 开头算一次，inner 内复用  │
├──────────────────────────────────────────────────────────────────────────┤
│ 陷阱 4：response_mask 的 shape 与 logprobs 错位                           │
│   症状: 训练 loss 看似正常但 reward 不上升                                │
│   修复: 所有 per-token 张量 shape 必须严格对齐 [B, T-1] 或 [B, T]         │
├──────────────────────────────────────────────────────────────────────────┤
│ 陷阱 5：忘记 freeze ref_policy / reward_model                            │
│   症状: 显存暴涨，reward 不稳定                                           │
│   修复: for p in ref_policy.parameters(): p.requires_grad_(False)        │
├──────────────────────────────────────────────────────────────────────────┤
│ 陷阱 6：advantage 没有归一化                                              │
│   症状: KL 暴涨或 reward 不上升                                           │
│   修复: 在 GAE 之后做 (A - mean) / (std + 1e-8)，仅在 mask 内计算         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 时间分配建议

| 阶段 | 内容 | 建议时间 |
|------|------|---------|
| 0 | 通读 [ppo-fill.md](./ppo-fill.md) + [alignment.md](../training/alignment.md) | 30 分钟 |
| 1 | SFT Dataset + 训练 + 验证 | 60-90 分钟 |
| 2 | RewardModel + BT loss + 训练 + 验证 | 60-90 分钟 |
| 3.1 | Value head + 共享 backbone | 30 分钟 |
| 3.2 | gather_logprobs（容易写错的地方） | 30 分钟 |
| 3.3 | GAE 倒序 | 20 分钟 |
| 3.4 | ppo_losses（policy + value clipped） | 30 分钟 |
| 3.5 | 完整 PPO loop（rollout + step） | 90-120 分钟 |
| 4 | 端到端 sanity check + 调参 | 60-90 分钟 |
| 5 | 进阶（DPO 或 GRPO 替换） | 60-90 分钟（选做） |

---

## 参考实现

::: details 完成挑战后点击查看参考实现（请先独立完成！）

```python
# ====================================================================
# Reference implementation — minimal RLHF pipeline
# 命名风格参考 huggingface/trl 与 OpenRLHF（均为开源项目）：
#   - PolicyWithValueHead = AutoModelForCausalLMWithValueHead
#   - rollout / ppo_step  = PPOTrainer.step 的拆分
# 完整工业级实现参考 https://github.com/huggingface/trl
# ====================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Model
import copy


# ---------- Stage 1: SFT ----------
class SFTDataset(Dataset):
    def __getitem__(self, idx):
        prompt = self.data[idx]["prompt"]
        response = self.data[idx]["response"]
        prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
        response_ids = self.tok(response, add_special_tokens=False).input_ids
        eos = self.tok.eos_token_id
        input_ids = prompt_ids + response_ids + [eos]
        labels = [-100] * len(prompt_ids) + response_ids + [eos]

        # truncate / pad
        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
            labels = labels[: self.max_len]
        pad_id = self.tok.pad_token_id
        pad_len = self.max_len - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len
        attn_mask = [1] * (self.max_len - pad_len) + [0] * pad_len
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn_mask),
        }


# ---------- Stage 2: RM ----------
class RewardModel(nn.Module):
    def __init__(self, base_model_name="gpt2"):
        super().__init__()
        self.backbone = GPT2Model.from_pretrained(base_model_name)
        H = self.backbone.config.n_embd
        self.v_head = nn.Linear(H, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        r_per_tok = self.v_head(h).squeeze(-1)              # [B, T]
        last_idx = attention_mask.sum(-1).long() - 1
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        return r_per_tok[batch_idx, last_idx]               # [B]


def rm_loss(r_chosen, r_rejected):
    loss = -F.logsigmoid(r_chosen - r_rejected)
    acc = (r_chosen > r_rejected).float().mean()
    return loss.mean(), acc


# ---------- Stage 3: PPO ----------
class PolicyWithValueHead(nn.Module):
    def __init__(self, base_model_name="gpt2"):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        H = self.lm.config.n_embd
        self.value_head = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.Linear(H, 1),
        )

    def forward(self, input_ids, attention_mask=None):
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = out.logits
        last_hidden = out.hidden_states[-1]
        values = self.value_head(last_hidden).squeeze(-1)
        return logits, values


def gather_logprobs(logits, input_ids):
    shift_logits = logits[:, :-1, :]
    shift_ids = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    return log_probs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t + 1 < T else torch.zeros(B, device=values.device)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        gae = delta + gamma * lam * gae
        advantages[:, t] = gae
    advantages = advantages * mask
    returns = advantages + values
    return advantages.detach(), returns.detach()


def ppo_losses(new_lp, old_lp, adv, new_v, old_v, ret, mask,
               clip=0.2, clip_v=0.2, vf_coef=0.5):
    ratio = (new_lp - old_lp).exp()
    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1 - clip, 1 + clip)
    policy_loss_per_tok = torch.maximum(unclipped, clipped)
    policy_loss = (policy_loss_per_tok * mask).sum() / mask.sum().clamp(min=1)

    v_clipped = old_v + torch.clamp(new_v - old_v, -clip_v, clip_v)
    v_loss_per_tok = 0.5 * torch.maximum(
        (new_v - ret) ** 2,
        (v_clipped - ret) ** 2,
    )
    value_loss = (v_loss_per_tok * mask).sum() / mask.sum().clamp(min=1)

    with torch.no_grad():
        approx_kl = ((new_lp - old_lp) * mask).sum() / mask.sum().clamp(min=1)
        clipfrac = (((ratio - 1.0).abs() > clip).float() * mask).sum() / mask.sum().clamp(min=1)

    total = policy_loss + vf_coef * value_loss
    return total, {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "approx_kl": approx_kl.item(),
        "clipfrac": clipfrac.item(),
    }


def shape_token_rewards(rm_score, old_lp, ref_lp, mask, kl_coef=0.05):
    kl = old_lp - ref_lp
    rewards = -kl_coef * kl
    last_idx = mask.sum(dim=-1).long() - 1
    batch_idx = torch.arange(rm_score.size(0), device=rm_score.device)
    rewards[batch_idx, last_idx] += rm_score
    return rewards * mask
```

完整训练循环 + 数据 pipeline 可参考开源项目（按依赖顺序）：

- [karpathy/minGPT](https://github.com/karpathy/minGPT)（MIT）— SFT 风格的训练循环范式
- [huggingface/trl](https://github.com/huggingface/trl)（Apache-2.0）— `PPOTrainer` / `RewardTrainer` / `DPOTrainer` 的工业级实现
- [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)（Apache-2.0）— Ray-based 大规模 RLHF 框架
- [PKU-Alignment/safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf)（Apache-2.0）— 多目标（安全 + 有用性）RLHF 实现

:::

---

## 推荐资源

::: tip 论文
- **InstructGPT (2022)**: [Training language models to follow instructions](https://arxiv.org/abs/2203.02155) — RLHF 三阶段范式的奠基论文
- **PPO (2017)**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — clipped surrogate / GAE 的源头
- **DPO (2023)**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — 把 RLHF 从两步合成一步
- **DeepSeek-R1 (2025)**: [DeepSeek-R1: Incentivizing Reasoning Capability via RL](https://arxiv.org/abs/2501.12948) — GRPO 在 reasoning 任务上的实践
:::

::: tip 开源代码（仅参考思路，不要逐行复制）
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — 最 minimal 的 GPT 训练（SFT 阶段直接套用）
- [huggingface/trl](https://github.com/huggingface/trl) — `PPOTrainer.step` 是本练习 PPO loop 的工业级版本
- [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — 看「rollout / advantage / loss」三块怎么解耦
- [eric-mitchell/direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization) — DPO 官方实现
:::

::: tip 教程与博客
- [Schulman: Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) — K1/K2/K3 的源头
- [HuggingFace TRL 文档](https://huggingface.co/docs/trl) — PPOTrainer 用法
- 本仓库的 [training/alignment.md](../training/alignment.md) — RLHF 主线知识
- 本仓库的 [deep-dives/nano-rlhf.md](../deep-dives/nano-rlhf.md) — RLHF 深度手撕（互补阅读）
:::

完成这个挑战后，你应该对**整条 alignment 流水线**的每个数据流向都了如指掌。RLHF 的难度不在公式，而在「四个模型 × 五种 per-token 张量 × 两个时间点（rollout vs inner update）」的组合复杂度——能把它从零跑通的人，工业 RLHF 框架的代码就能完全看懂了。

祝你训练顺利！遇到 `approx_kl` 跳变 / `reward` 不上升的崩溃局面是正常的——那才是真正学到东西的开始。

---

## MLM 代码训练模式

完成上面的固定填空后，试试随机挖空模式 -- 每次点击「刷新」会随机遮盖不同的代码片段，帮你彻底记住每一行。

### Reward Model（取最后一个 valid token 的 hidden）

<CodeMasker title="Reward Model：scalar head + last-valid-token pooling" :mask-ratio="0.15">
class RewardModel(nn.Module):
    def __init__(self, base_model_name="gpt2"):
        super().__init__()
        self.backbone = GPT2Model.from_pretrained(base_model_name)
        H = self.backbone.config.n_embd
        self.v_head = nn.Linear(H, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        r_per_tok = self.v_head(h).squeeze(-1)
        last_idx = attention_mask.sum(-1).long() - 1
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        return r_per_tok[batch_idx, last_idx]
</CodeMasker>

### Bradley-Terry Loss

<CodeMasker title="RM 训练：Bradley-Terry pairwise loss" :mask-ratio="0.15">
def rm_loss(r_chosen, r_rejected):
    loss = -F.logsigmoid(r_chosen - r_rejected)
    acc = (r_chosen > r_rejected).float().mean()
    return loss.mean(), acc
</CodeMasker>

### Policy with Value Head（共享 backbone）

<CodeMasker title="Actor-Critic：共享 backbone + 双 head" :mask-ratio="0.15">
class PolicyWithValueHead(nn.Module):
    def __init__(self, base_model_name="gpt2"):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        H = self.lm.config.n_embd
        self.value_head = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.Linear(H, 1),
        )

    def forward(self, input_ids, attention_mask=None):
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = out.logits
        last_hidden = out.hidden_states[-1]
        values = self.value_head(last_hidden).squeeze(-1)
        return logits, values
</CodeMasker>

### Gather Token-Level Log Probabilities

<CodeMasker title="Token Logprobs：shift + log_softmax + gather" :mask-ratio="0.15">
def gather_logprobs(logits, input_ids):
    shift_logits = logits[:, :-1, :]
    shift_ids = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    return log_probs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
</CodeMasker>

### PPO 总损失（policy clipped + value clipped）

<CodeMasker title="PPO Total Loss：policy + value 双 clip" :mask-ratio="0.15">
def ppo_losses(new_lp, old_lp, adv, new_v, old_v, ret, mask,
               clip=0.2, clip_v=0.2, vf_coef=0.5):
    ratio = (new_lp - old_lp).exp()
    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1 - clip, 1 + clip)
    policy_loss = (torch.maximum(unclipped, clipped) * mask).sum() / mask.sum().clamp(min=1)

    v_clipped = old_v + torch.clamp(new_v - old_v, -clip_v, clip_v)
    v_loss = 0.5 * torch.maximum((new_v - ret) ** 2, (v_clipped - ret) ** 2)
    value_loss = (v_loss * mask).sum() / mask.sum().clamp(min=1)

    return policy_loss + vf_coef * value_loss
</CodeMasker>

### Reward Shaping（KL + 末位 RM）

<CodeMasker title="Reward Shaping：全程 KL + 最后一位 RM 标量" :mask-ratio="0.15">
def shape_token_rewards(rm_score, old_lp, ref_lp, mask, kl_coef=0.05):
    kl = old_lp - ref_lp
    rewards = -kl_coef * kl
    last_idx = mask.sum(dim=-1).long() - 1
    batch_idx = torch.arange(rm_score.size(0), device=rm_score.device)
    rewards[batch_idx, last_idx] += rm_score
    return rewards * mask
</CodeMasker>
