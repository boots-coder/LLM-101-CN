---
title: "DeepSeek 架构填空"
description: "Level 2-3 填空：MLA 低秩注意力、DeepSeekMoE 路由、辅助损失 Free"
topics: [fill-in, DeepSeek, MLA, multi-latent-attention, DeepSeekMoE, auxiliary-loss-free]
---
# DeepSeek 架构填空 (Level 2-3)

> 本练习覆盖 DeepSeek-V2/V3 的核心架构创新：Multi-Latent Attention (MLA) 低秩压缩、RoPE 解耦、DeepSeekMoE 共享专家路由、辅助损失 Free 负载均衡，以及 KV Cache 显存计算。
> 代码基于实际架构设计，用 `_____` 标记需要填写的部分。

---

## 练习 1: Multi-Latent Attention (MLA) -- KV 压缩（Level 2）

### 背景

MHA 推理时需缓存完整的 K、V，KV Cache 与 $n_\text{heads} \times d_\text{head}$ 成正比。GQA 通过共享 KV head 缓解，但压缩比有限。DeepSeek-V2 提出 MLA：用低秩投影将 KV 压缩到 latent 向量 $\mathbf{c}$。

MLA 核心流程：
1. **下投影**: $\mathbf{c} = W^{DKV} \mathbf{x}$，将 $d_\text{model}$ 压缩到 $d_c$ 维
2. **上投影**: $\mathbf{K} = W^{UK} \mathbf{c}$，$\mathbf{V} = W^{UV} \mathbf{c}$，从 latent 恢复 K/V
3. 推理时只缓存 $\mathbf{c}$（$d_c$ 维），而非完整 K+V（$2 \times n_\text{heads} \times d_\text{head}$ 维）

### 任务

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLAttention(nn.Module):
    """Multi-Latent Attention: KV 低秩压缩"""
    def __init__(
        self,
        d_model=2048,
        n_heads=16,
        d_head=128,
        d_c=512,       # latent 维度 (远小于 n_heads * d_head = 2048)
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_c = d_c

        # Q 投影 (标准)
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)

        # ===== 填空 1: KV 下投影 — 将 d_model 压缩到 d_c =====
        self.W_dkv = _____  # 提示: nn.Linear(d_model, d_c)

        # ===== 填空 2: K 上投影 — 从 d_c 恢复到 n_heads * d_head =====
        self.W_uk = _____   # 提示: nn.Linear(d_c, n_heads * d_head)

        # ===== 填空 3: V 上投影 — 从 d_c 恢复到 n_heads * d_head =====
        self.W_uv = _____   # 提示: nn.Linear(d_c, n_heads * d_head)

        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x):
        """
        x: [bsz, seq_len, d_model]
        返回: [bsz, seq_len, d_model]
        """
        bsz, seq_len, _ = x.shape

        # Q: 标准投影
        Q = self.W_q(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # ===== 填空 4: KV 下投影得到 latent 向量 c =====
        c = _____  # 提示: self.W_dkv(x), shape [bsz, seq_len, d_c]

        # ===== 填空 5: 从 c 上投影恢复 K 和 V =====
        K = _____  # 提示: self.W_uk(c), 然后 view + transpose 成多头格式
        K = K.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = _____  # 提示: self.W_uv(c), 然后 view + transpose
        V = V.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # 标准 Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.W_o(out), c  # 返回 c 用于缓存
```

### 提示

- `W_dkv` 将 `d_model` 压缩到 `d_c`，推理时只缓存 `c`
- 压缩比 = $\frac{n_\text{heads} \times d_\text{head}}{d_c}$，本例 $\frac{16 \times 128}{512} = 4\times$

<details>
<summary>参考答案</summary>

```python
# 填空 1: KV 下投影
self.W_dkv = nn.Linear(d_model, d_c, bias=False)

# 填空 2: K 上投影
self.W_uk = nn.Linear(d_c, n_heads * d_head, bias=False)

# 填空 3: V 上投影
self.W_uv = nn.Linear(d_c, n_heads * d_head, bias=False)

# 填空 4: 下投影得到 latent 向量
c = self.W_dkv(x)  # [bsz, seq_len, d_c]

# 填空 5: 上投影恢复 K 和 V
K = self.W_uk(c)  # [bsz, seq_len, n_heads * d_head]
V = self.W_uv(c)  # [bsz, seq_len, n_heads * d_head]
```

**验证:**
```python
torch.manual_seed(42)
mla = MLAttention(d_model=2048, n_heads=16, d_head=128, d_c=512)
x = torch.randn(2, 8, 2048)
out, c = mla(x)
print(f"输出: {out.shape}")    # [2, 8, 2048]
print(f"Latent c: {c.shape}")  # [2, 8, 512]
# KV Cache 对比: MHA 缓存 2*16*128=4096 维, MLA 只缓存 512 维 -> 8x 压缩
```

</details>

---

## 练习 2: MLA -- RoPE 解耦（Level 2-3）

### 背景

RoPE 通过旋转 Q/K 注入位置信息。但在 MLA 中，若对压缩前的 K 施加 RoPE，$\mathbf{c}$ 会包含位置信息，破坏压缩的位置无关性。

DeepSeek-V2 的解决方案是 **RoPE 解耦**：将 Q/K 各拆为两部分：
- **nope 部分**: 不带位置编码，走压缩/恢复流程
- **rope 部分**: 单独投影并施加 RoPE，K 的 rope 部分不经过压缩

最终 attention score = $\mathbf{q}_\text{nope} \mathbf{k}_\text{nope}^T + \mathbf{q}_\text{rope} \mathbf{k}_\text{rope}^T$

### 任务

```python
def apply_rotary_pos_emb(x, cos, sin):
    """
    对 x 的最后一维施加 RoPE 旋转
    命名沿用 HuggingFace `transformers/models/llama/modeling_llama.py` 的 `apply_rotary_pos_emb`
    """
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

class MLAWithRoPE(nn.Module):
    def __init__(self, d_model=2048, n_heads=16, d_head=128,
                 d_head_rope=64, d_c=512):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_head_rope = d_head_rope
        self.d_head_nope = d_head - d_head_rope

        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_dkv = nn.Linear(d_model, d_c, bias=False)
        self.W_uk = nn.Linear(d_c, n_heads * self.d_head_nope, bias=False)
        self.W_uv = nn.Linear(d_c, n_heads * d_head, bias=False)
        self.W_kr = nn.Linear(d_model, n_heads * d_head_rope, bias=False)  # rope 部分不压缩
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x, cos, sin):
        """x: [bsz, seq_len, d_model], cos/sin: [1, 1, seq_len, d_rope//2]"""
        bsz, seq_len, _ = x.shape
        nh, d_nope, d_rope = self.n_heads, self.d_head_nope, self.d_head_rope

        Q = self.W_q(x).view(bsz, seq_len, nh, self.d_head).transpose(1, 2)

        # ===== 填空 1: 将 Q 拆分为 nope 和 rope 两部分 =====
        q_nope = _____  # 提示: Q[..., :d_nope]
        q_rope = _____  # 提示: Q[..., d_nope:]
        q_rope = apply_rotary_pos_emb(q_rope, cos, sin)

        # K 的 nope 部分: 从压缩向量恢复
        c = self.W_dkv(x)
        k_nope = self.W_uk(c).view(bsz, seq_len, nh, d_nope).transpose(1, 2)

        # ===== 填空 2: K 的 rope 部分 — 单独投影, 不经过压缩 =====
        k_rope = _____  # 提示: self.W_kr(x)
        k_rope = k_rope.view(bsz, seq_len, nh, d_rope).transpose(1, 2)
        k_rope = apply_rotary_pos_emb(k_rope, cos, sin)

        V = self.W_uv(c).view(bsz, seq_len, nh, self.d_head).transpose(1, 2)

        # ===== 填空 3: attention score = nope 部分 + rope 部分 =====
        score_nope = _____  # 提示: torch.matmul(q_nope, k_nope.transpose(-2, -1))
        score_rope = _____  # 提示: torch.matmul(q_rope, k_rope.transpose(-2, -1))
        scores = _____ / math.sqrt(self.d_head)  # 提示: 两部分相加

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.W_o(out), c, k_rope
```

### 提示

- Q 前 `d_nope` 维不带 RoPE，后 `d_rope` 维带 RoPE；K 的 rope 部分直接从 `x` 投影，不经过压缩
- 推理缓存 `c`（$d_c$）+ `k_rope`（$n_\text{heads} \times d_\text{rope}$），仍远小于 MHA

<details>
<summary>参考答案</summary>

```python
# 填空 1: Q 拆分
q_nope = Q[..., :d_nope]   # [bsz, nh, seq_len, d_nope]
q_rope = Q[..., d_nope:]   # [bsz, nh, seq_len, d_rope]

# 填空 2: K 的 rope 部分
k_rope = self.W_kr(x)  # [bsz, seq_len, nh * d_rope]

# 填空 3: attention score
score_nope = torch.matmul(q_nope, k_nope.transpose(-2, -1))
score_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))
scores = (score_nope + score_rope) / math.sqrt(self.d_head)
```

**验证:**
```python
torch.manual_seed(42)
d_model, n_heads, d_head, d_rope, d_c = 2048, 16, 128, 64, 512
bsz, seq_len = 2, 8

mla = MLAWithRoPE(d_model=d_model, n_heads=n_heads,
                  d_head=d_head, d_head_rope=d_rope, d_c=d_c)
x = torch.randn(bsz, seq_len, d_model)

# 预计算 RoPE
freqs = 1.0 / (10000 ** (torch.arange(0, d_rope, 2).float() / d_rope))
t = torch.arange(seq_len).float()
angles = torch.outer(t, freqs)
cos = angles.cos().view(1, 1, seq_len, d_rope // 2)
sin = angles.sin().view(1, 1, seq_len, d_rope // 2)

out, c, k_r = mla(x, cos, sin)
print(f"输出: {out.shape}")          # [2, 8, 2048]
print(f"Latent c: {c.shape}")        # [2, 8, 512]
print(f"K rope cache: {k_r.shape}")  # [2, 16, 8, 64]
# 推理缓存: c(512) + k_rope(16*64=1024) = 1536 维, MHA 需 4096 维 -> 2.67x 压缩
```

</details>

---

## 练习 3: DeepSeekMoE -- 共享专家 + 路由专家（Level 2-3）

### 背景

标准 MoE 中所有专家都通过路由竞争，但许多 token 需要通用知识（语法、常见实体），导致路由专家间存在冗余。DeepSeekMoE 将专家分为两类：
- **共享专家 (Shared Experts)**: 每个 token 都经过，捕捉通用知识
- **路由专家 (Routed Experts)**: Top-K 门控选择，捕捉专业化知识

最终输出 = 共享专家输出 + 路由专家加权输出。

### 任务

```python
class Expert(nn.Module):
    """单个专家: 标准 FFN"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class DeepSeekMoE(nn.Module):
    def __init__(
        self,
        dim=1024,
        n_shared=2,         # 共享专家数
        n_routed=64,        # 路由专家数
        topk=6,             # 每 token 激活的路由专家数
        expert_hidden=2048, # 每个专家的隐藏层维度 (细粒度, 较小)
    ):
        super().__init__()
        self.n_shared = n_shared
        self.n_routed = n_routed
        self.topk = topk

        # 共享专家
        self.shared_experts = nn.ModuleList(
            [Expert(dim, expert_hidden) for _ in range(n_shared)]
        )
        # 路由专家
        self.routed_experts = nn.ModuleList(
            [Expert(dim, expert_hidden) for _ in range(n_routed)]
        )
        # 门控网络: 只对路由专家做路由
        self.gate = nn.Linear(dim, n_routed, bias=False)

    def forward(self, x):
        """
        x: [bsz, seq_len, dim]
        返回: [bsz, seq_len, dim]
        """
        bsz, seq_len, dim = x.shape
        N = bsz * seq_len
        x_flat = x.view(N, dim)

        # ===== 填空 1: 所有 token 过共享专家, 结果相加 =====
        shared_out = _____  # 提示: 初始化为 torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out = _____  # 提示: 累加每个共享专家的输出

        # 路由: 对路由专家做 Top-K 选择
        gate_logits = self.gate(x_flat)               # [N, n_routed]
        gate_probs = F.softmax(gate_logits, dim=-1)
        topk_weight, topk_idx = torch.topk(gate_probs, k=self.topk, dim=-1)
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        # ===== 填空 2: 路由专家的 dispatch-compute-combine =====
        routed_out = torch.zeros_like(x_flat)
        for i in range(self.n_routed):
            # 找出选择了第 i 个路由专家的 token
            mask = (topk_idx == i)                     # [N, topk] bool
            token_ids, topk_pos = torch.where(mask)
            if len(token_ids) == 0:
                continue
            # ===== 填空: 取出 token, 过专家, 加权累加 =====
            expert_input = _____    # 提示: x_flat[token_ids]
            expert_output = _____   # 提示: self.routed_experts[i](expert_input)
            cur_weight = _____      # 提示: topk_weight[token_ids, topk_pos]
            routed_out[token_ids] += _____  # 提示: cur_weight.unsqueeze(-1) * expert_output

        # ===== 填空 3: 合并共享专家和路由专家的输出 =====
        output = _____  # 提示: shared_out + routed_out

        return output.view(bsz, seq_len, dim)
```

### 提示

- 共享专家直接对所有 token 计算并求和，路由专家走标准 dispatch-compute-combine
- DeepSeek 设计：多路由专家（64）+ 小隐藏层 + 大 Top-K（6），提升专业化程度

<details>
<summary>参考答案</summary>

```python
# 填空 1: 共享专家
shared_out = torch.zeros_like(x_flat)
for expert in self.shared_experts:
    shared_out = shared_out + expert(x_flat)

# 填空 2: 路由专家
expert_input = x_flat[token_ids]
expert_output = self.routed_experts[i](expert_input)
cur_weight = topk_weight[token_ids, topk_pos]
routed_out[token_ids] += cur_weight.unsqueeze(-1) * expert_output

# 填空 3: 合并
output = shared_out + routed_out
```

**验证:**
```python
torch.manual_seed(42)
moe = DeepSeekMoE(dim=1024, n_shared=2, n_routed=16, topk=4, expert_hidden=2048)
x = torch.randn(2, 4, 1024)
y = moe(x)
print(f"输出: {y.shape}")   # [2, 4, 1024]

# 共享专家贡献分析
with torch.no_grad():
    x_flat = x.view(-1, 1024)
    shared = sum(e(x_flat) for e in moe.shared_experts)
    print(f"共享专家 / 总输出 范数比: {shared.norm() / y.view(-1,1024).norm():.1%}")

loss = y.mean()
loss.backward()
print("反向传播成功, 共享+路由专家均有梯度")
```

</details>

---

## 练习 4: 辅助损失 Free 的负载均衡（Level 3）

### 背景

传统 MoE 用辅助损失强制 token 均匀分配给各专家，但损失权重难调节：太大影响性能，太小无法防止专家坍塌。DeepSeek-V3 提出 **辅助损失 Free** 方案：为每个专家维护 bias $b_i$，加到 gate logits 上影响路由选择，然后根据负载动态更新 bias。关键：bias 只影响 Top-K 选择，不参与最终权重计算，因此不影响梯度回传。

### 任务

```python
class AuxLossFreeMoE(nn.Module):
    def __init__(
        self,
        dim=1024,
        num_experts=64,
        topk=6,
        bias_update_speed=0.001,  # bias 更新步长
    ):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.bias_update_speed = bias_update_speed

        self.experts = nn.ModuleList(
            [Expert(dim) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(dim, num_experts, bias=False)

        # ===== 填空 1: 为每个专家维护一个 bias, 不参与梯度 =====
        # 提示: 使用 register_buffer, 初始化为全零
        _____  # self.register_buffer('expert_bias', ...)

    def route_with_bias(self, x):
        """
        x: [N, dim]
        返回: topk_weight, topk_idx (权重用原始概率, 选择用加 bias 后的分数)
        """
        gate_logits = self.gate(x)  # [N, num_experts]

        # 原始概率 (用于最终加权)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # ===== 填空 2: 在 gate logits 上加 bias, 用于 Top-K 选择 =====
        biased_logits = _____  # 提示: gate_logits + self.expert_bias

        # 用加了 bias 的 logits 做 Top-K 选择
        _, topk_idx = torch.topk(biased_logits, k=self.topk, dim=-1)

        # 但权重使用原始概率 (不含 bias)
        topk_weight = gate_probs.gather(1, topk_idx)
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        return topk_weight, topk_idx

    @torch.no_grad()
    def update_bias(self, topk_idx, N):
        """
        根据实际负载动态更新 bias
        topk_idx: [N, topk]
        N: 总 token 数
        """
        # 统计每个专家被选中的次数
        expert_counts = torch.zeros(self.num_experts, device=topk_idx.device)
        for k in range(self.topk):
            expert_counts.scatter_add_(
                0, topk_idx[:, k],
                torch.ones(N, device=topk_idx.device)
            )

        # 理想的均匀负载
        ideal_count = N * self.topk / self.num_experts

        # ===== 填空 3: 根据负载偏差更新 bias =====
        # 负载过高 -> 降低 bias, 负载过低 -> 提高 bias
        load_diff = _____  # 提示: expert_counts - ideal_count
        self.expert_bias -= _____  # 提示: self.bias_update_speed * load_diff

    def forward(self, x):
        bsz, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        N = x_flat.shape[0]

        topk_weight, topk_idx = self.route_with_bias(x_flat)

        # dispatch-compute-combine (同练习 3, 此处省略细节)
        y = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            tok, pos = torch.where(topk_idx == i)
            if len(tok) == 0: continue
            y[tok] += topk_weight[tok, pos].unsqueeze(-1) * self.experts[i](x_flat[tok])

        if self.training:
            self.update_bias(topk_idx, N)
        return y.view(bsz, seq_len, dim)
```

### 提示

- `register_buffer` 使 bias 随模型保存但不参与梯度
- 更新规则：`bias -= speed * (actual - ideal)`，简单负反馈控制

<details>
<summary>参考答案</summary>

```python
# 填空 1: 注册 bias buffer
self.register_buffer('expert_bias', torch.zeros(num_experts))

# 填空 2: 加 bias 影响路由
biased_logits = gate_logits + self.expert_bias

# 填空 3: 根据负载更新 bias
load_diff = expert_counts - ideal_count
self.expert_bias -= self.bias_update_speed * load_diff
```

**验证:**
```python
torch.manual_seed(42)
dim, num_experts, topk = 256, 16, 4

moe = AuxLossFreeMoE(dim=dim, num_experts=num_experts, topk=topk)
moe.train()
print("初始 bias:", moe.expert_bias[:8].tolist())  # 全零

# 模拟多步训练
for step in range(10):
    x = torch.randn(4, 8, dim)
    y = moe(x)

print(f"10 步后 bias std: {moe.expert_bias.std().item():.4f}")
print(f"bias 前 8 个: {[f'{b:.4f}' for b in moe.expert_bias[:8].tolist()]}")

# 对比有/无 bias 的负载均衡
with torch.no_grad():
    x = torch.randn(4, 32, dim).view(-1, dim)
    _, idx = moe.route_with_bias(x)
    counts = torch.zeros(num_experts)
    for k in range(topk):
        counts.scatter_add_(0, idx[:, k].cpu(), torch.ones(idx.shape[0]))
    print(f"有 bias 负载 std: {counts.std():.2f}")
```

</details>

---

## 练习 5: KV Cache 大小计算综合题（Level 1-2）

### 背景

KV Cache 是 LLM 推理的核心显存消耗，直接决定最大 batch size 和序列长度。考虑以下三种配置（均为单层）：

| 参数 | 值 |
|------|-----|
| $d_\text{model}$ | 4096 |
| $n_\text{heads}$ (Q heads) | 32 |
| $d_\text{head}$ | 128 |
| 数据类型 | float16 (2 bytes) |

三种注意力方案：
- **MHA**: 每个 head 独立存储 K 和 V
- **GQA**: $n_\text{kv\_heads} = 8$，即每 4 个 Q head 共享 1 组 KV
- **MLA**: $d_c = 512$（latent 维度），额外有 $d_\text{rope} = 64$（每头 rope 维度，共 $n_\text{heads}$ 头的 k_rope）

### 任务

```python
def kv_cache_analysis():
    # 模型参数
    n_heads, d_head = 32, 128
    n_kv_heads_gqa = 8       # GQA 的 KV head 数
    d_c, d_rope = 512, 64    # MLA latent 维度 / 每头 rope 维度
    bpw = 2                  # bytes per weight (float16)
    n_layers = 60

    # ===== 填空 1: 每 token 每层 KV Cache 维度 =====
    dims_mha = _____  # 提示: n_heads * d_head * 2
    dims_gqa = _____  # 提示: n_kv_heads_gqa * d_head * 2
    dims_mla = _____  # 提示: d_c + n_heads * d_rope

    for name, d in [("MHA", dims_mha), ("GQA", dims_gqa), ("MLA", dims_mla)]:
        print(f"{name}: {d} dims = {d * bpw} bytes/token/layer")

    # ===== 填空 2: 总 KV Cache (batch=32, seq=4096) =====
    bs, seq = 32, 4096
    cache_mha = _____  # 提示: bs * seq * n_layers * dims_mha * bpw
    cache_gqa = _____
    cache_mla = _____
    GB = 1024 ** 3
    for name, c in [("MHA", cache_mha), ("GQA", cache_gqa), ("MLA", cache_mla)]:
        print(f"{name}: {c / GB:.2f} GB")

    # ===== 填空 3: 压缩比 =====
    ratio_gqa = _____  # 提示: dims_mha / dims_gqa
    ratio_mla = _____  # 提示: dims_mha / dims_mla
    print(f"GQA vs MHA: {ratio_gqa:.1f}x,  MLA vs MHA: {ratio_mla:.1f}x")

    # 40GB 显存预算下最大序列长度 (bs=1)
    budget = 40 * GB
    for name, d in [("MHA", dims_mha), ("GQA", dims_gqa), ("MLA", dims_mla)]:
        print(f"{name} max_seq (bs=1): {budget // (n_layers * d * bpw):,}")

kv_cache_analysis()
```

### 提示

- MHA: $n_\text{heads} \times d_\text{head} \times 2$; GQA: $n_\text{kv\_heads} \times d_\text{head} \times 2$; MLA: $d_c + n_\text{heads} \times d_\text{rope}$
- 总显存 = batch $\times$ seq $\times$ layers $\times$ dims $\times$ bytes

<details>
<summary>参考答案</summary>

```python
# 填空 1: 每 token 每层 KV Cache 维度
dims_mha = n_heads * d_head * 2              # 32 * 128 * 2 = 8192
dims_gqa = n_kv_heads_gqa * d_head * 2       # 8 * 128 * 2  = 2048
dims_mla = d_c + n_heads * d_rope            # 512 + 32 * 64 = 2560

# 填空 2: 总 KV Cache (bytes)
cache_mha = batch_size * seq_len * n_layers * dims_mha * bytes_per_param
cache_gqa = batch_size * seq_len * n_layers * dims_gqa * bytes_per_param
cache_mla = batch_size * seq_len * n_layers * dims_mla * bytes_per_param

# 填空 3: 压缩比
ratio_gqa = dims_mha / dims_gqa   # 8192 / 2048 = 4.0x
ratio_mla = dims_mha / dims_mla   # 8192 / 2560 = 3.2x
```

**验证:**
```python
kv_cache_analysis()
# 预期: MHA 120.00 GB, GQA 30.00 GB, MLA 37.50 GB
# 压缩比: GQA 4.0x, MLA 3.2x
# MLA 压缩比看似不如 GQA, 但优势在于:
# 1. d_c 可灵活调整, GQA 只能取整数比
# 2. MLA 表达能力等价 MHA (低秩 != 低能力)
# 3. 上投影可吸收到 Q 投影, 推理无额外计算
```

</details>

---

## 总结

| 练习 | 难度 | 核心知识点 |
|------|------|-----------|
| MLA KV 压缩 | Level 2 | 低秩投影 down/up-projection，KV Cache 压缩 |
| MLA RoPE 解耦 | Level 2-3 | 位置编码与压缩的冲突，nope/rope 分离 |
| DeepSeekMoE | Level 2-3 | 共享专家 + 路由专家，细粒度专家设计 |
| 辅助损失 Free | Level 3 | bias-based 动态负载均衡，无辅助损失 |
| KV Cache 计算 | Level 1-2 | MHA/GQA/MLA 显存对比，长序列推理分析 |

### 延伸思考

1. **MLA 上投影吸收**: 推理时 $W^{UK}$ 可与 $W^Q$ 合并为 $\tilde{W}^Q = W^Q (W^{UK})^T$，从而直接用 $\mathbf{c}$ 计算 attention，请推导。
2. **细粒度 vs 粗粒度专家**: DeepSeek 64 小专家 + Top-6 vs Mixtral 8 大专家 + Top-2，总计算量相当，为何细粒度更优？
3. **bias 更新稳定性**: 若 `bias_update_speed` 过大会怎样？如何设计更稳定的策略（如 EMA）？

---

## MLM 代码训练模式

完成上面的固定填空后，试试随机挖空模式 -- 每次点击「刷新」会随机遮盖不同的代码片段，帮你彻底记住每一行。

### MLA 低秩 KV 压缩

<CodeMasker title="MLA 下投影 + 上投影核心流程" :mask-ratio="0.15">
class MLAttention(nn.Module):
    def __init__(self, d_model=2048, n_heads=16, d_head=128, d_c=512):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_c = d_c
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_dkv = nn.Linear(d_model, d_c, bias=False)
        self.W_uk = nn.Linear(d_c, n_heads * d_head, bias=False)
        self.W_uv = nn.Linear(d_c, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        Q = self.W_q(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        c = self.W_dkv(x)
        K = self.W_uk(c).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_uv(c).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.W_o(out), c
</CodeMasker>

### RoPE 解耦: nope/rope 分离

<CodeMasker title="MLA RoPE 解耦 — Q/K 拆分与 score 合并" :mask-ratio="0.15">
Q = self.W_q(x).view(bsz, seq_len, nh, self.d_head).transpose(1, 2)
q_nope = Q[..., :d_nope]
q_rope = Q[..., d_nope:]
q_rope = apply_rotary_pos_emb(q_rope, cos, sin)

c = self.W_dkv(x)
k_nope = self.W_uk(c).view(bsz, seq_len, nh, d_nope).transpose(1, 2)

k_rope = self.W_kr(x)
k_rope = k_rope.view(bsz, seq_len, nh, d_rope).transpose(1, 2)
k_rope = apply_rotary_pos_emb(k_rope, cos, sin)

score_nope = torch.matmul(q_nope, k_nope.transpose(-2, -1))
score_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))
scores = (score_nope + score_rope) / math.sqrt(self.d_head)
</CodeMasker>

### DeepSeekMoE 共享专家 + 路由专家

<CodeMasker title="DeepSeekMoE dispatch-compute-combine" :mask-ratio="0.15">
shared_out = torch.zeros_like(x_flat)
for expert in self.shared_experts:
    shared_out = shared_out + expert(x_flat)

gate_logits = self.gate(x_flat)
gate_probs = F.softmax(gate_logits, dim=-1)
topk_weight, topk_idx = torch.topk(gate_probs, k=self.topk, dim=-1)
topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

routed_out = torch.zeros_like(x_flat)
for i in range(self.n_routed):
    mask = (topk_idx == i)
    token_ids, topk_pos = torch.where(mask)
    if len(token_ids) == 0:
        continue
    expert_input = x_flat[token_ids]
    expert_output = self.routed_experts[i](expert_input)
    cur_weight = topk_weight[token_ids, topk_pos]
    routed_out[token_ids] += cur_weight.unsqueeze(-1) * expert_output

output = shared_out + routed_out
</CodeMasker>

### 辅助损失 Free 负载均衡

<CodeMasker title="bias 路由 + 动态负载更新" :mask-ratio="0.15">
self.register_buffer('expert_bias', torch.zeros(num_experts))

def route_with_bias(self, x):
    gate_logits = self.gate(x)
    gate_probs = F.softmax(gate_logits, dim=-1)
    biased_logits = gate_logits + self.expert_bias
    _, topk_idx = torch.topk(biased_logits, k=self.topk, dim=-1)
    topk_weight = gate_probs.gather(1, topk_idx)
    topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    return topk_weight, topk_idx

def update_bias(self, topk_idx, N):
    ideal_count = N * self.topk / self.num_experts
    load_diff = expert_counts - ideal_count
    self.expert_bias -= self.bias_update_speed * load_diff
</CodeMasker>
