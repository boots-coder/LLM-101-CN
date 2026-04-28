---
title: "DeepSeek-V3 架构"
description: "MoE 混合专家、MLA 多头潜注意力、Multi-Token Prediction"
topics: [DeepSeek-V3, MoE, mixture-of-experts, MLA, multi-token-prediction, load-balancing]
prereqs: [architecture/llama]
---
# DeepSeek-V3 架构

> **一句话总结:** DeepSeek-V3 在 Llama-style 架构基础上引入了 MoE 混合专家实现高效扩容、MLA 多头潜在注意力实现 KV Cache 极致压缩、以及多 Token 预测提升训练效率，代表了 2024 年大模型架构的前沿水平。

## 在大模型体系中的位置

DeepSeek-V3 由 DeepSeek 团队于 2024 年发布，是一个 671B 总参数（37B 激活参数）的 MoE 模型。它在 Llama 确立的架构基础上做了三项重大创新，每一项都解决了大模型规模扩展中的核心瓶颈：

```
Llama 架构                       DeepSeek-V3 架构创新
─────────                        ─────────────────────
Dense（所有参数都参与计算）   →    MoE（只激活部分专家，以少量计算撬动大参数量）
GQA（减少 KV 头数量）        →    MLA（低秩压缩 KV Cache，压缩率更极致）
单 Token 预测               →    多 Token 预测（一次预测多个 Token，训练更高效）
```

## 核心概念

### MoE 混合专家（Mixture of Experts）

MoE 的核心思想是：**不让所有参数参与每个 Token 的计算，而是为每个 Token 动态选择一小部分"专家"来处理**。在 Transformer 中，每层的 FFN 被替换为多个并行的 FFN（"专家"），加上一个路由网络（Gate）决定每个 Token 使用哪些专家。

**路由机制与稀疏门控：** 路由网络将 Token 的隐藏状态映射为各专家的得分，选择 top-k 个专家：

$$G(x) = \text{Softmax}(\text{KeepTopK}(W_r \cdot x, k))$$

$$\text{MoE}(x) = \sum_{i \in \text{TopK}} g_i(x) \cdot E_i(x)$$

**单个专家的最小可运行实现：** 在 V3 官方实现里，每个专家就是一个 SwiGLU MLP（`gate_proj → SiLU` 与 `up_proj` 逐元素相乘后再走 `down_proj`），与 Llama 的 FFN 结构同源、只是宽度更小（routed expert 的 `intermediate_size` 比 dense FFN 小一个量级，由路由稀疏性弥补容量）。

::: tip 开源参考
- DeepSeek-V3 官方仓库：<https://github.com/deepseek-ai/DeepSeek-V3>（`inference/model.py` 中的 `MLP` 与 `Expert`）
- HuggingFace `modeling_deepseek_v3.py`：<https://huggingface.co/deepseek-ai/DeepSeek-V3>（`DeepseekV3MLP` 即下方等价结构）
:::

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """SwiGLU FFN — DeepSeek-V3 中每个 routed/shared expert 的最小骨架。"""
    def __init__(self, dim: int, intermediate_dim: int | None = None):
        super().__init__()
        # routed expert 的 intermediate 通常 ≈ 2~4×dim 而非 dense FFN 的 ~8×dim
        intermediate_dim = intermediate_dim or 2 * dim
        self.gate_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.up_proj   = nn.Linear(dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

**稀疏 MoE 的三段式实现（dispatch-compute-combine）：**

```python
class SparseExpertLayer(nn.Module):
    def __init__(self, hidden_dim=512, n_experts=8, top_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [Expert(dim=self.hidden_dim) for _ in range(self.n_experts)]
        )
        self.router = nn.Linear(self.hidden_dim, self.n_experts)

    def forward(self, x):
        bsz, seq_len, d = x.shape
        num_tokens = bsz * seq_len
        flat_x = x.view(num_tokens, d)  # 展平为 token 列表

        # 0. 路由打分
        scores = F.softmax(self.router(flat_x), dim=-1)
        topk_weights, topk_ids = torch.topk(scores, dim=-1, k=self.top_k)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # 1. dispatch: 为每个专家收集被分配的 token
        assignments = [None] * self.n_experts
        for eid in range(self.n_experts):
            matched = torch.where(topk_ids == eid)
            if matched[0].numel() > 0:
                assignments[eid] = matched

        # 2. compute: 各专家批量前向
        expert_out = [None] * self.n_experts
        for eid in range(self.n_experts):
            if assignments[eid] is not None:
                sel_tokens = assignments[eid][0]
                expert_out[eid] = self.experts[eid](flat_x[sel_tokens])

        # 3. combine: 按权重聚合各专家输出
        combined = torch.zeros_like(flat_x)
        for eid in range(self.n_experts):
            if expert_out[eid] is not None:
                tok_idx, slot_idx = assignments[eid]
                w = topk_weights[tok_idx, slot_idx]
                combined[tok_idx] += w.unsqueeze(-1) * expert_out[eid]

        return combined.reshape(bsz, seq_len, d)
```

**DeepSeek-V3 的 MoE 特点：** 使用 256 个路由专家 + 1 个共享专家，每个 Token 选择 8 个路由专家。共享专家始终参与计算，捕获通用知识。另外 V3 使用 **sigmoid 代替 softmax** 作为门控函数：

```python
class V3MoELayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.n_routed = cfg.n_routed_experts
        self.top_k = cfg.top_k
        self.route_experts = nn.ModuleList(
            [Expert(self.dim) for _ in range(self.n_routed)]
        )
        self.gate_proj = nn.Linear(self.dim, self.n_routed)
        # 共享专家：每个 token 都经过，学习通用知识
        self.shared_experts = nn.ModuleList(
            [Expert(self.dim) for _ in range(cfg.n_shared_experts)]
        )

    def route_forward(self, x):
        scores = F.sigmoid(self.gate_proj(x))  # V3 用 sigmoid 代替 softmax
        topk_w, topk_idx = torch.topk(scores, dim=-1, k=self.top_k)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        # dispatch-compute-combine 逻辑...
        return routed_out, scores, topk_idx

    def shared_forward(self, x):
        out = torch.zeros_like(x)
        for expert in self.shared_experts:
            out = out + expert(x)  # 共享专家无门控权重
        return out

    def forward(self, x):
        y_routed, gate_scores, sel_idx = self.route_forward(x)
        y_shared = self.shared_forward(x)
        return x + y_routed + y_shared  # 残差连接
```

### 负载均衡（Load Balance）

sMoE 训练中，如果路由网络总是偏好少数专家，会导致其他专家被"饿死"。负载均衡机制保证各专家均衡参与训练。

**Switch Transformer 负载均衡 Loss：** 综合考虑每个专家处理的 token 比例 $f_i$ 和累积权重 $P_i$：

$$\mathcal{L}_\text{balance} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

$$f_i = \frac{1}{T}\sum_{x \in \mathcal{B}} \mathbb{1}\{\text{topk}(p(x)) = i\}, \quad P_i = \frac{1}{T}\sum_{x \in \mathcal{B}} p_i(x)$$

```python
def switch_balance_loss(sel_idx, sel_weights, n_experts):
    """Switch Transformer 风格的负载均衡损失"""
    num_tokens, k = sel_idx.shape
    # 构建专家级别的分配矩阵和权重矩阵
    dispatch_mask = torch.zeros(num_tokens, n_experts)
    weight_map = torch.zeros(num_tokens, n_experts)
    dispatch_mask.scatter_(-1, sel_idx, 1.0)
    weight_map.scatter_(-1, sel_idx, sel_weights)

    token_frac = dispatch_mask.mean(dim=0)   # 每个专家处理的 token 比例
    weight_frac = weight_map.mean(dim=0)     # 每个专家的平均权重

    return n_experts * (token_frac * weight_frac).sum()
```

**DeepSeek-V3 的无辅助损失策略：** 传统辅助 loss 会与主训练目标冲突。V3 为每个专家维护一个动态偏置项（bias），在路由时加上该偏置。过载专家减小偏置，欠载专家增大偏置——不干扰主损失函数。

### MLA 多头潜在注意力（Multi-head Latent Attention）

MLA 通过**低秩联合压缩**将 KV Cache 压缩到极小尺寸。核心思想：不直接缓存 K/V，而是缓存一个低维压缩向量 $c_t$，推理时从 $c_t$ 恢复 K/V。

$$c_t = W_\text{DKV} \cdot x_t \quad (d_\text{model} \to d_c, \; d_c \ll d_\text{model})$$

$$K_t = W_\text{UK} \cdot c_t, \quad V_t = W_\text{UV} \cdot c_t$$

**MLA 实现代码：**

```python
class LatentAttention(nn.Module):
    """MLA: 低秩压缩 KV Cache 的多头注意力"""
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.dim = cfg.dim
        self.head_dim = cfg.dim // cfg.n_heads
        self.kv_compress_dim = cfg.kv_compress_dim   # KV 压缩维度，远小于 dim
        self.q_compress_dim = cfg.q_compress_dim     # Q 压缩维度

        # Q 低秩分解：降维 → 升维
        self.q_proj_down = nn.Linear(self.dim, self.q_compress_dim, bias=False)
        self.q_proj_up = nn.Linear(self.q_compress_dim, self.dim, bias=False)

        # KV 共享压缩：同一个 latent 向量分别映射为 K 和 V
        self.kv_proj_down = nn.Linear(self.dim, self.kv_compress_dim, bias=False)
        self.k_proj_up = nn.Linear(self.kv_compress_dim, self.dim, bias=False)
        self.v_proj_up = nn.Linear(self.kv_compress_dim, self.dim, bias=False)

        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)

        # RoPE 解耦：位置编码独立参数
        self.q_rope_proj = nn.Linear(self.q_compress_dim, self.dim, bias=False)
        self.k_rope_proj = nn.Linear(self.dim, self.head_dim, bias=False)
```

**训练阶段的前向计算：**

```python
# 训练时：降维再升维
latent_q = self.q_proj_down(X)       # [bs, seq, q_compress_dim] — Q 的低秩表示
Q_full = self.q_proj_up(latent_q)    # [bs, seq, dim] — 恢复满维度 Q

latent_kv = self.kv_proj_down(X)     # [bs, seq, kv_compress_dim] — KV 共享压缩
K_full = self.k_proj_up(latent_kv)   # [bs, seq, dim] — 恢复 K
V_full = self.v_proj_up(latent_kv)   # [bs, seq, dim] — 恢复 V

# RoPE 解耦：位置编码单独计算
Q_pe = self.q_rope_proj(latent_q)    # 多头位置编码
K_pe = self.k_rope_proj(X)           # 单头位置编码（节省 cache）

# 将内容部分与位置编码部分拼接后做注意力
Q = torch.cat((Q_nope_heads, RoPE(Q_pe)), dim=-1)  # 内容 + 位置
K = torch.cat((K_nope_heads, RoPE(K_pe)), dim=-1)  # 内容 + 位置
```

**矩阵吸收技巧（推理优化）：** 训练完成后，可以将低秩矩阵合并为满矩阵，减少推理时的矩阵运算次数：

```python
# 训练时: Q = q_proj_up(q_proj_down(h))，两次矩阵乘
# 推理时: 合并为一次矩阵乘
W_q_merged = (attn.q_proj_up.weight.data @ attn.q_proj_down.weight.data).t()
Q = X @ W_q_merged  # 单次矩阵乘，精度完全一致

# 同理，V 的升维矩阵吸收到输出投影中：
# 原本需要 [d x d] + [c x d]，吸收后只需 [c x d]
W_vo_fused = W_v_up @ W_out  # 参数量显著减少
```

**KV Cache 对比：**

| 方案 | 缓存内容 | 每层每 Token 缓存量 | 相对大小 |
|------|---------|-------------------|---------|
| MHA | 完整 K、V | $2 \times n_h \times d_h$ | 1.0 |
| GQA（8 组） | 分组 K、V | $2 \times n_g \times d_h$ | $\sim$1/4 |
| MLA | $c_t$ + rope_k | $d_c + d_h^R$ | 可低至 1/16 |

**KV Cache 存什么？** MLA 缓存的是 `latent_kv = kv_proj_down(h)`（维度 $d_c$）加上解耦的 rope key（维度 $d_h^R$），而非完整的 K/V。推理时按需恢复 K/V，用计算时间换存储空间。

### Multi-Token Prediction（多 Token 预测）

标准自回归模型每步只预测下一个 Token。DeepSeek-V3 引入 MTP 作为辅助训练目标，每个位置同时预测多个未来 Token。

**MTP 数据构造：** 对于输入序列，构建多个偏移的 label：

```python
PAD_LABEL = -100

class NextKTokenDataset(Dataset):
    """为 MTP 构造 k 个偏移的 label 序列"""
    def __init__(self, token_ids, k_ahead=5):
        super().__init__()
        self.k_ahead = k_ahead
        self.token_ids = token_ids.clone()
        batch, length = token_ids.shape
        # 每个偏移量生成一组 label，尾部用 PAD_LABEL 填充
        self.targets = torch.full((k_ahead, batch, length), PAD_LABEL, dtype=torch.long)
        for offset in range(k_ahead):
            self.targets[offset, :, :length - offset - 1] = token_ids[:, offset + 1:]
# 示例: 输入 [83, 74, 53, 6, 3, 2, 29, 44, 93, 32]
# targets[0]: [74, 53, 6, 3, 2, 29, 44, 93, 32, -100]  (next-1)
# targets[1]: [53, 6, 3, 2, 29, 44, 93, 32, -100, -100] (next-2)
```

**DeepSeek-V3 MTP 模块：** V3 的 MTP 设计与 basic MTP 不同——各 MTP 头**共享主体模型的 lm_head**，通过递归传递 latent 特征：

```python
class MultiTokenPredictor(nn.Module):
    """单个 MTP 头：融合上一级隐状态与当前 embedding，递归预测下一个 Token"""
    def __init__(self, dim):
        super().__init__()
        self.norm_prev = nn.Linear(dim, dim)      # 归一化上一级特征
        self.norm_emb = nn.Linear(dim, dim)       # 归一化当前嵌入
        self.fuse_proj = nn.Linear(dim * 2, dim)  # 融合后降维
        self.tfm_block = nn.Linear(dim, dim)      # 简化的 Transformer block

    def forward(self, cur_emb, prev_hidden):
        h_emb = self.norm_emb(cur_emb)
        h_prev = self.norm_prev(prev_hidden)
        fused = torch.cat((h_emb, h_prev), dim=-1)  # 拼接嵌入与上级特征
        out = self.fuse_proj(fused)
        out = self.tfm_block(out)
        return out


class DeepSeekV3WithMTP(nn.Module):
    def __init__(self, dim, vocab_size, n_mtp_heads=5):
        super().__init__()
        self.n_mtp_heads = n_mtp_heads
        self.tok_emb = nn.Embedding(vocab_size, dim)     # 共享嵌入层
        self.backbone = nn.Linear(dim, dim)               # 主体模型（简化）
        self.lm_head = nn.Linear(dim, vocab_size)         # 共享输出头
        self.predictors = nn.ModuleList(
            [MultiTokenPredictor(dim) for _ in range(n_mtp_heads)]
        )

    def forward(self, x):
        bsz, seq_len = x.shape
        emb = self.tok_emb(x)
        h = self.backbone(emb[:, -self.n_mtp_heads:, :])
        main_logits = self.lm_head(h)

        # 递归调用各预测头：每级接收上级 hidden + 偏移位置的 embedding
        for i in range(self.n_mtp_heads):
            emb_shifted = emb[:, i+1:i+1+seq_len-self.n_mtp_heads, :]
            h_i = self.predictors[i](
                emb_shifted, h.detach()  # detach 截断梯度回传
            )
            pred_logits_i = self.lm_head(h_i)  # 复用同一个 lm_head
            h = h_i

        return main_logits, pred_logits
```

**MTP 损失函数：** 主体 NTP 损失 + 各 MTP 头损失的加权和：

$$\mathcal{L} = \mathcal{L}_\text{lm\_head} + \lambda \cdot \frac{1}{N}\sum_{k=1}^{N} \mathcal{L}_{\text{mtp}_k}$$

```python
def compute_mtp_loss(main_logits, pred_logits, labels, weight=0.1):
    n_heads, bsz, seq_len, vocab = pred_logits.shape
    ce = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)

    # 主体语言模型损失
    main_loss = ce(main_logits.reshape(bsz * seq_len, vocab),
                   labels[:, :-n_heads].reshape(bsz * seq_len))

    # 各 MTP 预测头的损失
    head_losses = torch.zeros(n_heads)
    for k in range(n_heads):
        head_losses[k] = ce(
            pred_logits[k].reshape(bsz * seq_len, vocab),
            labels[:, k:k - n_heads].reshape(bsz * seq_len)
        )

    total_loss = main_loss + weight * head_losses.mean()
    return total_loss
```

**MTP 推理（Speculative Decoding）：** MTP 头可用于投机解码——主体预测 $t_5$，MTP 头递归预测 $t_6, t_7, ...$，一次生成多个候选 token：

```python
# Inference: 主体模型预测 next-token，MTP 头递归预测后续 token
tok = torch.argmax(main_logits[:, -1, :], dim=-1)  # 主体预测 t5
seq = torch.cat((seq, tok.unsqueeze(1)), dim=1)

for k in range(n_mtp_heads):
    emb_k = model.tok_emb(seq[:, k+1:])
    h_k = model.predictors[k](emb_k, h)
    logits_k = model.lm_head(h_k)
    tok = torch.argmax(logits_k[:, -1, :], dim=-1)
    seq = torch.cat((seq, tok.unsqueeze(1)), dim=1)
    h = h_k
# 结果: 一次生成 1 + n_mtp_heads 个 token
```

## 完整 DeepSeek-V3 Block

将 MLA、MoE 和 RMSNorm 组装成完整的 DeepSeek-V3 解码块：

```python
class V3DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.attn = LatentAttention(cfg)   # MLA 替代 GQA
        self.moe = V3MoELayer(cfg)         # MoE 替代 Dense FFN

    def forward(self, h, mask=None, sin=None, cos=None):
        # Pre-Norm + MLA 注意力 + 残差
        h = h + self.attn(self.attn_norm(h), sin, cos)
        # Pre-Norm + MoE FFN + 残差
        moe_out, gate_scores, sel_idx = self.moe(self.ffn_norm(h))
        h = h + moe_out
        return h, gate_scores, sel_idx
```

完整 DeepSeek-V3 模型将多个 Block 堆叠，顶层加上 MTP 训练头：

```python
class V3LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(
            [V3DecoderBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size)
        # MTP 模块（训练时使用，推理时可去除）
        self.predictors = nn.ModuleList(
            [MultiTokenPredictor(cfg) for _ in range(cfg.n_mtp)]
        )
        # RoPE 缓存
        self.rope_cos, self.rope_sin = create_rope(cfg.max_seq_len, ...)

    def forward(self, x, mask):
        h = self.tok_emb(x)
        for layer in self.layers:
            h, _, _ = layer(h, mask, self.rope_sin, self.rope_cos)
        h = self.final_norm(h)
        logits = self.lm_head(h)
        return logits  # + MTP logits (训练时)
```

## 苏格拉底时刻

请停下来思考以下问题，不急于查看答案：

1. MoE 让模型拥有 671B 参数但只激活 37B——那些"未被选中"的专家参数是否浪费了？它们在什么时候发挥作用？
2. MLA 将 KV Cache 压缩到极小维度——低秩近似的误差会不会影响模型质量？训练时的低秩分解和推理时的矩阵吸收本质区别是什么？
3. MTP 中每个头做的仍然是 NTP（next-token-prediction），为什么这种递归式 NTP 比真正的 NNTP（隔空预测）更有效？
4. MoE 的路由是 token-level 的——同一句话中相邻 Token 可能被路由到完全不同的专家。这是问题还是特性？
5. DeepSeek-V3 的三大创新（MoE、MLA、MTP）之间存在什么协同效应？例如 MoE 增加了参数量但 MLA 减少了显存占用，二者如何互补？

## 常见问题 & 面试考点

- **Q: DeepSeek-V3 的 671B 参数和 Llama 70B 比，推理成本差多少？** 每个 Token 只激活约 37B 参数，推理 FLOPs 与 Llama 70B 同一数量级。但模型参数总量大，显存占用高。
- **Q: MoE 的专家数量越多越好吗？** 不一定。专家越多容量越大，但通信开销和路由不稳定风险也增加。
- **Q: MLA 和 GQA 能共存吗？** 没有必要——MLA 是 GQA 思想的极端推广，已经实现更极致的 KV Cache 压缩。
- **Q: 什么是"共享专家"？** 每个 Token 都会经过的专家，学习通用知识，与路由专家互补。
- **Q: 辅助损失负载均衡和 V3 的无辅助损失策略有什么区别？** 传统方法在损失中加均衡项，与主目标冲突。V3 通过动态偏置实现均衡，不干扰主损失。
- **Q: MTP 训练完后推理时怎么用？** V3 发布时砍掉 MTP 头，只用标准 NTP 推理。也可保留 MTP 头做投机解码加速。

## 推荐资源

- **DeepSeek-AI《DeepSeek-V3 Technical Report》** — DeepSeek-V3 技术报告
- **DeepSeek-AI《DeepSeek-V2》** — MLA 首次提出的论文
- **Shazeer et al.《Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer》** — MoE 经典论文
- **Fedus et al.《Switch Transformers》** — 简化版 MoE 与负载均衡 Loss
- **Gloeckle et al.《Better & Faster Large Language Models via Multi-token Prediction》** — 多 Token 预测方法
- **DeepSeek-V3 开源代码** — 官方开源实现
