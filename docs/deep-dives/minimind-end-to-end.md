---
title: minimind 端到端复现
description: 26M 参数中文社区从零 LLM 训练栈：tokenizer → pretrain → SFT → DPO → GRPO/CISPO → 蒸馏 → 服务化
topics: [minimind, end-to-end, pretrain, sft, dpo, grpo, cispo, distillation, lora, moe, openai-api]
prereqs: [training/pretraining, training/sft, training/alignment]
---
# minimind 端到端复现

> **一句话总结：** [minimind](https://github.com/jingyaogong/minimind) 用 ~26M 参数、单 GPU 2 小时把"tokenizer → 预训练 → SFT → LoRA → DPO → GRPO/CISPO → 蒸馏 → OpenAI 兼容服务"全栈跑通，是中文社区少见的"完整训练栈"参考实现。

## 为什么要读 minimind

::: tip 与其他 deep-dive 的差异
| 项目 | 关注点 | 训练栈完整度 |
|------|--------|-------------|
| build-nanogpt / nano-gpt | GPT-2 模型本身、预训练 loss 曲线 | 仅预训练 |
| nano-rlhf | 用纯 PyTorch 实现 RM/PPO/DPO/GRPO 公式 | 仅对齐算法 |
| **minimind** | 一个真实可跑的小模型从零到上线的全流程 | tokenizer + pretrain + SFT + LoRA + DPO + GRPO/CISPO + 蒸馏 + 服务化 |

minimind 不是"最教学化"的实现（它已经用了 `transformers.PreTrainedModel`、`F.scaled_dot_product_attention`、`AutoTokenizer`），但它是"最 end-to-end"的：你能在自己的卡上把一个会聊天的小模型从零训出来，再用 OpenAI 兼容 API 让前端直接接入。
:::

**前置知识：** [预训练](/training/pretraining) → [SFT](/training/sft) → [偏好对齐](/training/alignment)。读完那三节再来看 minimind 的代码会很顺。

## 1. 项目定位与硬件门槛

minimind 仓库（[jingyaogong/minimind](https://github.com/jingyaogong/minimind)）以 768 维 / 8 层 / GQA 4-KV-head 的默认配置训练出 ~26M 参数的中文 base 模型，作者声称 2~3 小时单卡（3090 级别）可跑通预训练 + SFT。所有训练脚本都在 `trainer/` 下，每个脚本就是一个独立可执行入口（`python trainer/train_pretrain.py --batch_size 32`），没有 trainer 抽象层包装，方便逐行读。

仓库目录：

```
minimind/
├── model/
│   ├── model_minimind.py    # 模型主体（Config + RMSNorm + RoPE+YaRN + GQA + MoE）
│   └── model_lora.py        # LoRA monkey-patch
├── trainer/
│   ├── train_pretrain.py    # 从零预训练
│   ├── train_full_sft.py    # 全参 SFT
│   ├── train_lora.py        # LoRA 微调
│   ├── train_dpo.py         # DPO 偏好对齐
│   ├── train_grpo.py        # GRPO + CISPO（推理对齐）
│   ├── train_ppo.py         # 经典 PPO
│   ├── train_distillation.py  # 白盒蒸馏
│   ├── train_agent.py       # Agent 阶段微调
│   └── rollout_engine.py    # GRPO 用的可插拔 rollout 引擎
└── scripts/
    └── serve_openai_api.py  # FastAPI OpenAI 兼容层
```

::: details 怎么读这个仓库
1. **先扫架构** — `model/model_minimind.py` 全文 ~286 行，半小时能读完
2. **再选一条主线** — 比如"预训练 → SFT → LoRA → DPO"四件套，每个 trainer 脚本都是一个 `train_epoch` + 一个 `__main__`
3. **对比训练循环** — 你会发现 SFT、LoRA、DPO 的 `train_epoch` 框架几乎一样，差异都集中在 **loss 计算** 这十几行上
:::

## 2. 模型架构亮点

### 2.1 Config：一处控制所有

[`model/model_minimind.py#L10-L45`](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L10-L45) 把模型所有可调旋钮压在一个 `MiniMindConfig` 里：

```python
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)  # GQA
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.intermediate_size = kwargs.get(
            "intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64
        )
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
```

几个观察点：

- **vocab_size = 6400**：自己训的中文 BPE，词表极小，所以 26M 参数里 embedding 只占 `6400 × 768 ≈ 4.9M`
- **GQA 默认 8 Q-head / 4 KV-head**：训练阶段就用 GQA，KV-cache 直接减半
- **`intermediate_size = ⌈hidden·π/64⌉·64`**：用 π 做 FFN 中间维比例，并对齐到 64
- **`tie_word_embeddings=True`**：embedding 和 lm_head 共享权重，再省一份矩阵

### 2.2 RMSNorm 与 RoPE + YaRN

RMSNorm 实现走 fp32 计算后 cast 回原 dtype（[L50-L60](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L50-L60)）：

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
```

真正有意思的是 RoPE 频率预计算把 **YaRN 长上下文动态缩放** 直接缝进了 `precompute_freqs_cis`（[L62-L78](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L62-L78)）：

```python
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024),
                         rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (
        torch.arange(0, dim, 2)[: (dim // 2)].float() / dim
    )), 1.0
    if rope_scaling is not None:  # YaRN: f'(i) = f(i)((1-γ) + γ/s), γ∈[0,1] 线性 ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) \
                                / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), \
                        min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin
```

**关键点：** YaRN 的核心思路是按频率分段缩放——高频维（`beta_fast=32` 以上）保持不变，低频维（`beta_slow=1` 以下）按 factor 缩放，中间用 `ramp` 做线性插值。代码里 `inv_dim(b)` 把"波长 → 维度索引"换算出来，再用 `clamp((i-low)/(high-low), 0, 1)` 给每个维度算一个 0~1 的 ramp 系数。这样训练时只用 2048 上下文，推理时打开 `inference_rope_scaling` 就能外推到 32k 而不爆掉。

### 2.3 GQA Attention：Flash + 手写双路径

[`Attention.forward`（L111-L134）](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L111-L134) 同时保留 flash 路径和回退路径：

```python
def forward(self, x, position_embeddings, past_key_value=None,
            use_cache=False, attention_mask=None):
    bsz, seq_len, _ = x.shape
    xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    xq = xq.view(bsz, seq_len, self.n_local_heads,    self.head_dim)
    xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
    xq, xk = self.q_norm(xq), self.k_norm(xk)            # QK-Norm
    cos, sin = position_embeddings
    xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
    if past_key_value is not None:
        xk = torch.cat([past_key_value[0], xk], dim=1)
        xv = torch.cat([past_key_value[1], xv], dim=1)
    past_kv = (xk, xv) if use_cache else None
    xq, xk, xv = (xq.transpose(1, 2),
                  repeat_kv(xk, self.n_rep).transpose(1, 2),
                  repeat_kv(xv, self.n_rep).transpose(1, 2))
    if self.flash and (seq_len > 1) and ...:
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal)
    else:
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # ... 手写 softmax + causal mask
```

值得注意的细节：

- **QK-Norm**：`self.q_norm = RMSNorm(head_dim)` / `self.k_norm = RMSNorm(head_dim)`（[L104-L105](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L104-L105)），在 RoPE 之前做 head 维度的 RMSNorm，对小模型训练稳定性帮助很大
- **`repeat_kv`** 用 `expand+reshape` 而不是 `repeat_interleave`，零拷贝（[L86-L89](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L86-L89)）
- **flash 与手写双路径**：当 `seq_len==1`（解码阶段）或带显式 mask 时回退到手写 attention，否则走 SDPA

### 2.4 MoE：单文件实现，aux_loss 直接累加

[`MOEFeedForward`（L148-L176）](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L148-L176) 用 `nn.ModuleList` + `index_add_` 实现 top-k 路由：

```python
class MOEFeedForward(nn.Module):
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(
            scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx,
                             (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())  # 防止 unused param 报错
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() \
                            * self.config.num_experts \
                            * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)
```

aux_loss 在 `MiniMindModel.forward` 里被沿层累加（[L231](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L231)）：

```python
aux_loss = sum([l.mlp.aux_loss for l in self.layers
                if isinstance(l.mlp, MOEFeedForward)],
               hidden_states.new_zeros(1).squeeze())
return hidden_states, presents, aux_loss
```

然后在 `MiniMindForCausalLM` 里用 `MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, ...)` 把它一路冒泡出来——这就解释了为什么所有 trainer 脚本里都有 `loss = res.loss + res.aux_loss` 这一行（即使在非 MoE 时 `aux_loss` 是 0 的标量）。

## 3. 预训练栈

[`trainer/train_pretrain.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_pretrain.py) 的训练循环非常薄（[L23-L79](https://github.com/jingyaogong/minimind/blob/master/trainer/train_pretrain.py#L23-L79)）：

```python
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
```

把它对照 nano-gpt 的训练循环看，差异基本就两点：

1. **loss 已经在模型里算好** — `model(input_ids, labels=labels)` 直接返回 `MoeCausalLMOutputWithPast`，外部只负责 backward
2. **`res.aux_loss` 始终加上** — 这样 MoE 和 dense 共用同一段代码

默认参数（[L82-L106](https://github.com/jingyaogong/minimind/blob/master/trainer/train_pretrain.py#L82-L106)）：`epochs=2, batch_size=32, lr=5e-4, accumulation_steps=8, grad_clip=1.0, max_seq_len=340`，作者直接在帮助文档里写了"中文 1 token ≈ 1.5~1.7 字符"，这种朴素的工程注释在大型 trainer 库里反而很少见。

::: details 为什么 max_seq_len 只有 340
预训练阶段用极短上下文是 minimind 的关键工程取舍：26M 模型短窗口下能塞更大 batch、更快迭代，且 RoPE 在小窗口下学到的频率分布会被推理阶段的 YaRN 自动外推到 32k。
:::

## 4. SFT + LoRA：训练循环复用，loss 不变

### 4.1 Full SFT 与 Pretrain 几乎一致

[`train_full_sft.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_full_sft.py) 的 `train_epoch`（[L23-L80](https://github.com/jingyaogong/minimind/blob/master/trainer/train_full_sft.py#L23-L80)）和预训练版本几乎逐行相同。差异：

- 数据集换成 `SFTDataset`（多轮对话数据，labels 在 prompt 部分填 -100）
- 默认 `lr=1e-5, batch_size=16, accumulation_steps=1, max_seq_len=768`
- `--from_weight` 默认 `pretrain`，自动从预训练 ckpt 加载

[L102](https://github.com/jingyaogong/minimind/blob/master/trainer/train_full_sft.py#L102) 这行就是把 SFT 与预训练串起来的关键：

```python
parser.add_argument('--from_weight', default='pretrain', type=str,
                    help="基于哪个权重训练，为none则不基于任何权重训练")
```

### 4.2 LoRA monkey-patch：54 行实现一个 PEFT

[`model/model_lora.py`](https://github.com/jingyaogong/minimind/blob/master/model/model_lora.py) 全文只有 65 行，没有继承任何 PEFT 库：

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)  # A 高斯初始化
        self.B.weight.data.zero_()                       # B 零初始化（关键：开始时 ΔW=0）

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)
            module.forward = forward_with_lora
```

**注意 `module.weight.shape[0] == module.weight.shape[1]` 这条筛选**：只给方阵 Linear（即 `q_proj/k_proj/v_proj/o_proj` 中维度匹配的那些）注入 LoRA，FFN 的非方阵 Linear 跳过。这是个比"全部 Linear 都加"更省的策略，但读者需要意识到它是个工程妥协，不一定是最优。

### 4.3 LoRA 训练循环：冻结 + 收集

[`train_lora.py#L138-L151`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_lora.py#L138-L151) 是模板写法：

```python
# 冻结非LoRA参数，收集LoRA参数
lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True
        lora_params.append(param)
    else:
        param.requires_grad = False

train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
```

需要警惕的兼容性提醒（[L162-L165](https://github.com/jingyaogong/minimind/blob/master/trainer/train_lora.py#L162-L165)）：

```python
if args.use_compile == 1:
    args.use_compile = 0
    Logger('[LoRA] monkey-patch forward 与 torch.compile 不兼容，use_compile 已自动关闭')
```

monkey-patch `module.forward` 会破坏 `torch.compile` 的 dispatch graph，作者直接强制关掉——这是很多自己写 LoRA 的人会踩的坑。

## 5. 对齐三件套：DPO / GRPO+CISPO / PPO

### 5.1 DPO：50 行核心实现

[`train_dpo.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_dpo.py) 把 DPO loss 拆成两个独立函数（[L24-L49](https://github.com/jingyaogong/minimind/blob/master/trainer/train_dpo.py#L24-L49)）：

```python
def logits_to_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(
        log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    ref_log_probs    = (ref_log_probs    * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs    = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs    = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    pi_logratios  = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs    - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

**实现技巧：** chosen 和 rejected 不分两次 forward——而是在 `train_epoch` 里把它们 `torch.cat` 到 batch 维（[L64-L66](https://github.com/jingyaogong/minimind/blob/master/trainer/train_dpo.py#L64-L66)）：

```python
x    = torch.cat([x_chosen,    x_rejected],    dim=0)
y    = torch.cat([y_chosen,    y_rejected],    dim=0)
mask = torch.cat([mask_chosen, mask_rejected], dim=0)
```

这样 policy 和 ref 各自只前向一次，前一半 batch 是 chosen、后一半是 rejected——`batch_size // 2` 切片就还原回成对比较。

### 5.2 GRPO + CISPO：一行 if 切换两种 loss

GRPO 是本仓库工程上最有意思的部分。优势计算（[`train_grpo.py#L119-L122`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_grpo.py#L119-L122)）是标准的组内归一化：

```python
grouped_rewards = rewards.view(-1, args.num_generations)         # [B, num_gen]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
std_r  = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
advantages = (rewards - mean_r) / (std_r + 1e-4)
```

但真正的看点在 [L130-L141](https://github.com/jingyaogong/minimind/blob/master/trainer/train_grpo.py#L130-L141) —— **CISPO（MiniMax 论文里的"Clipped IS-weight Policy Optimization"）和 GRPO 共用一个 `if`**：

```python
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1                    # [B*num_gen, R]
ratio = torch.exp(per_token_logps - old_per_token_logps)         # [B*num_gen, R]
if args.loss_type == "cispo":
    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
    per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps
                       - args.beta * per_token_kl)
else:
    clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
    per_token_loss1 = ratio         * advantages.unsqueeze(1)
    per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
    per_token_loss = -(torch.min(per_token_loss1, per_token_loss2)
                       - args.beta * per_token_kl)
policy_loss = ((per_token_loss * completion_mask).sum(dim=1)
               / completion_mask.sum(dim=1).clamp(min=1)).mean()
```

::: tip CISPO vs GRPO 的关键差异
- **GRPO**：用 `min(ratio·A, clip(ratio,1±ε)·A)` 经典 PPO clip
- **CISPO**：把 ratio `clamp` 到上界后 **detach** 当作重要性权重，再乘到 `per_token_logps` 上——梯度只通过 logp 流，重要性权重不参与求导
- 默认 `epsilon=0.2`（GRPO），`epsilon_high=5.0`（CISPO 上界放宽到 5 倍），见 [L227-L228](https://github.com/jingyaogong/minimind/blob/master/trainer/train_grpo.py#L227-L228)
- KL 用 `exp(kl) - kl - 1` 这个非负无偏估计，不是直接的 `kl`
:::

### 5.3 Reward 设计：规则 + 奖励模型

[`calculate_rewards`（L36-L67）](https://github.com/jingyaogong/minimind/blob/master/trainer/train_grpo.py#L36-L67) 把"规则奖励 + reward model 奖励"叠加：

```python
def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)
    with torch.no_grad():
        reward_model_scores = []
        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                # 1) 长度奖励
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                # 2) thinking 段奖励
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                # 3) n-gram 重复惩罚
                rewards[response_idx] -= rep_penalty(answer)
                # 4) reward model 分数
                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)
        rewards += torch.tensor(reward_model_scores, device=args.device)
```

四类奖励叠加是 minimind 推理对齐的核心工程套路，规则部分是 R1 风格的（长度区间 + thinking 段长度 + 标签数量），加上一个独立的 InternLM2 reward model 当连续奖励。

### 5.4 Rollout 引擎：可插拔架构

[`trainer/rollout_engine.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/rollout_engine.py) 抽象出一个 `RolloutEngine` 基类，并提供 `TorchRolloutEngine`（PyTorch 原生 generate）和 `SGLangRolloutEngine` 两种实现（[L51-L91](https://github.com/jingyaogong/minimind/blob/master/trainer/rollout_engine.py#L51-L91)）：

```python
class RolloutEngine(ABC):
    tokenizer = None
    @abstractmethod
    def rollout(self, prompt_ids, attention_mask, num_generations,
                max_new_tokens, temperature=0.8) -> RolloutResult: ...
    @abstractmethod
    def update_policy(self, model: torch.nn.Module): ...


class TorchRolloutEngine(RolloutEngine):
    def rollout(self, prompt_ids, attention_mask, num_generations,
                max_new_tokens, temperature=0.8):
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) \
                else self.policy_model
        with torch.no_grad(), (self.autocast_ctx or nullcontext()):
            output_ids = model.generate(
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
```

`update_policy(model)` 是关键设计——SGLang 后端在每次 policy 更新后需要把权重 dump 到共享路径再重启 server，这个抽象让 GRPO 主循环不用关心后端到底是 PyTorch 还是 SGLang。

## 6. 白盒蒸馏：温度平方 + α 混合

[`train_distillation.py#L24-L92`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_distillation.py#L24-L92) 的核心就两段：

```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return (temperature ** 2) * kl
```

```python
# 1) Ground-Truth CE Loss
shift_labels = labels[..., 1:].contiguous()
loss_mask_flat = loss_mask.view(-1)
ce_loss = F.cross_entropy(
    student_logits.view(-1, student_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
    reduction='none')
ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
if lm_config_student.use_moe: ce_loss = ce_loss_raw + res.aux_loss
else: ce_loss = ce_loss_raw

# 2) Distillation Loss
if teacher_model is not None:
    distill_loss = distillation_loss(
        student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
        teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
        temperature=temperature)
else:
    distill_loss = torch.tensor(0.0, device=args.device)

# 3) 总损失 = alpha * CE + (1-alpha) * Distill
loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps
```

::: details 为什么乘 `temperature ** 2`
经典 Hinton 蒸馏公式：用温度 T 软化 logits 后，KL 散度的梯度大小约为 1/T²，所以乘 T² 把梯度量级拉回与 T=1 时同一尺度，让 distill_loss 和 ce_loss 可以直接相加而不用重新调 lr。
:::

另一个值得注意的细节是 [L64-L65](https://github.com/jingyaogong/minimind/blob/master/trainer/train_distillation.py#L64-L65)：

```python
vocab_size_student = student_logits.size(-1)
teacher_logits = teacher_logits[..., :vocab_size_student]
```

直接对齐 vocab 维度——这是教师/学生用不同 tokenizer 时的"穷人版"对齐方案，前提是学生词表是教师词表的前缀子集。

## 7. 服务化：FastAPI 流式 + tool calling

[`scripts/serve_openai_api.py`](https://github.com/jingyaogong/minimind/blob/master/scripts/serve_openai_api.py) 用 230 行实现了一个最小可用的 OpenAI 兼容服务：

### 7.1 流式生成：`TextStreamer + Queue + Thread`

`CustomStreamer`（[L71-L80](https://github.com/jingyaogong/minimind/blob/master/scripts/serve_openai_api.py#L71-L80)）继承 `transformers.TextStreamer`，把每段 decode 后的文本扔进 `queue.Queue`：

```python
class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)
```

主线程 yield SSE，生成在子线程里（[L113-L126](https://github.com/jingyaogong/minimind/blob/master/scripts/serve_openai_api.py#L113-L126)）：

```python
def _generate():
    model.generate(
        inputs.input_ids, max_new_tokens=max_tokens, do_sample=True,
        temperature=temperature, top_p=top_p,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
Thread(target=_generate).start()
```

### 7.2 thinking 段与 tool_calls 的解析

`parse_response`（[L83-L102](https://github.com/jingyaogong/minimind/blob/master/scripts/serve_openai_api.py#L83-L102)）是个朴素的正则解析器：

```python
def parse_response(text):
    reasoning_content = None
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        try:
            call = json.loads(m.strip())
            tool_calls.append({
                "id": f"call_{int(time.time())}_{i}",
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
                },
            })
        except Exception:
            pass
    if tool_calls:
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    return text.strip(), reasoning_content, tool_calls or None
```

流式版（[L132-L165](https://github.com/jingyaogong/minimind/blob/master/scripts/serve_openai_api.py#L132-L165)）做的事更精细：边 yield 边检测 `</think>`，命中前的文本作为 `reasoning_content` 字段返回，命中后的作为 `content` 字段——这就实现了 OpenAI 新的 `reasoning_content` 协议。

整个服务对外暴露的就是 `POST /v1/chat/completions`（[L171-L227](https://github.com/jingyaogong/minimind/blob/master/scripts/serve_openai_api.py#L171-L227)），前端可以直接用任何 OpenAI SDK 接入。

## 8. 苏格拉底时刻

读完上面这些代码，自己问自己：

1. **YaRN 为什么是"分频段"缩放，而不是简单地把 `rope_theta` 调大？**  
   提示：`precompute_freqs_cis` 里 `low/high` 的物理含义是什么？高频维度（小波长）和低频维度（大波长）的外推效果有什么区别？

2. **DPO 把 chosen 和 rejected 拼到一个 batch 是为了节省什么？相比"分两次 forward"省了多少显存？**

3. **CISPO 用 `clamp(ratio, max=ε_high).detach()` 而不是 `min(ratio·A, clip(ratio)·A)`，从梯度路径看，两者本质差异在哪里？为什么 CISPO 把 `ε_high` 默认设到 5.0？**

4. **LoRA monkey-patch 为什么会和 `torch.compile` 冲突？如果想用 compile，应该怎么改写 `apply_lora`？**

5. **MoE 的 `aux_loss` 直接乘以 `num_experts × router_aux_loss_coef`，但 dense 模型 `aux_loss=0`——为什么 trainer 里还要无脑写 `loss = res.loss + res.aux_loss`，不会引入额外开销吗？**

6. **蒸馏里 `vocab_size_student = student_logits.size(-1); teacher_logits = teacher_logits[..., :vocab_size_student]` 这种"截断对齐"成立的前提是什么？如果两个 tokenizer 完全无关，应该怎么改？**

## 9. 学习路径建议

把 minimind 当作"中文 LLM 训练栈的索引"来读：

| 阶段 | 入口文件 | 配套阅读 |
|------|---------|---------|
| 模型架构 | `model/model_minimind.py` | [Llama 架构](/architecture/llama)、[注意力机制](/architecture/attention) |
| 预训练 | `trainer/train_pretrain.py` | [预训练](/training/pretraining)、[深度剖析 GPT-2](/deep-dives/nano-gpt) |
| SFT + LoRA | `trainer/train_full_sft.py` + `trainer/train_lora.py` + `model/model_lora.py` | [SFT](/training/sft)、[深度剖析 LoRA](/deep-dives/lora-from-scratch) |
| DPO/GRPO/CISPO | `trainer/train_dpo.py` + `trainer/train_grpo.py` | [偏好对齐](/training/alignment)、[深度剖析 RLHF Pipeline](/deep-dives/nano-rlhf) |
| 推理对齐 | `trainer/rollout_engine.py` + `train_grpo.py` | [R1 推理模型复现](/deep-dives/r1-reproduction) |
| 蒸馏 | `trainer/train_distillation.py` | [知识蒸馏](/training/distillation) |
| 服务化 | `scripts/serve_openai_api.py` | [模型部署](/engineering/deployment) |

::: tip 一个建议的复现顺序
1. 先 `train_pretrain.py` 跑通一个会续写中文的 base 模型（2~3 小时）
2. `train_full_sft.py` 把它 SFT 成会聊天的模型（1 小时）
3. 用 `serve_openai_api.py` 起服务，用 ChatGPT 风格前端验证
4. **回头**再读 DPO/GRPO 这种"改进 loss"——你已经有 base+SFT 模型了，对齐才有意义
5. 最后试 LoRA 和蒸馏，理解 PEFT 与模型压缩的工程取舍
:::

## 推荐资源

- 仓库主页：<https://github.com/jingyaogong/minimind>
- 中文 README：详细的训练步骤和数据集说明
- 作者技术讨论区：<https://github.com/jingyaogong/minimind/discussions>（generate 实现细节、训练曲线分享）

> 与本站其他 deep-dive 配合阅读：[深度剖析 GPT-2](/deep-dives/nano-gpt) 看预训练循环、[深度剖析 LoRA](/deep-dives/lora-from-scratch) 看 PEFT 数学、[深度剖析 RLHF Pipeline](/deep-dives/nano-rlhf) 看对齐算法公式、[R1 推理模型复现](/deep-dives/r1-reproduction) 看推理对齐工程化。
