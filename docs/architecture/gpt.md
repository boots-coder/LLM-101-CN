---
title: "GPT 架构"
description: "Decoder-Only 架构、GELU、Pre-Norm、KV Cache 与 GPT-2 完整实现"
topics: [GPT, decoder-only, GELU, pre-norm, KV-cache, causal-mask]
prereqs: [architecture/transformer]
---
# GPT 架构

> **一句话总结:** GPT 确立了 Decoder-Only 的自回归范式，通过 Pre-Normalization 和 GELU 激活函数等关键改进，将 Transformer 从序列到序列的翻译模型进化为通用的语言生成引擎。

## 在大模型体系中的位置

GPT（Generative Pre-trained Transformer）是从原始 Transformer 到现代 LLM 的关键跳板。GPT-1（2018）首次证明了"大规模预训练 + 下游微调"的范式，GPT-2 展示了 zero-shot 能力，GPT-3 开启了 in-context learning 时代。此后几乎所有主流 LLM（Llama、DeepSeek、Qwen 等）都沿用了 GPT 建立的 Decoder-Only 架构，并在此基础上进行改良。

```
Transformer (2017)          GPT-1 (2018)              GPT-2/3 (2019/2020)
Encoder-Decoder        →   Decoder-Only          →   规模扩展 + 涌现能力
Post-Norm                   Pre-Norm                   In-Context Learning
ReLU                        GELU                       Few-Shot / Zero-Shot
```

## 核心概念

### Decoder-Only vs Encoder-Decoder

原始 Transformer 采用 Encoder-Decoder 结构：Encoder 双向看到全部输入，Decoder 通过交叉注意力读取 Encoder 输出并自回归生成。GPT 做了一个关键简化：**去掉 Encoder 和交叉注意力，只保留 Decoder**。

**为什么 Decoder-Only 胜出？**

1. **架构统一：** 输入和输出使用同一套参数处理，不需要区分"理解"和"生成"两个阶段
2. **训练效率：** 每个 Token 都可以作为预测目标（Causal LM），训练信号更密集
3. **灵活性：** 通过 prompt 可以统一处理分类、翻译、问答、生成等各种任务，无需为每种任务设计不同的输入输出格式
4. **规模效应：** 实践证明 Decoder-Only 在大规模下的 scaling 表现最好

**因果掩码（Causal Mask）：** Decoder-Only 的核心约束——每个 Token 只能看到它前面（包括自身）的 Token，不能看到未来的 Token。这通过在注意力矩阵上施加一个下三角掩码实现：

$$\text{Mask}_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

```python
import torch
import torch.nn.functional as F

# 因果掩码的构造与效果
L = 4
causal_mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
scores = torch.zeros(L, L)
scores = scores.masked_fill(causal_mask, float('-inf'))
print(scores)
# tensor([[0., -inf, -inf, -inf],
#         [0.,  0., -inf, -inf],
#         [0.,  0.,  0., -inf],
#         [0.,  0.,  0.,  0.]])

# softmax 后：未来位置的注意力权重变为 0
p = F.softmax(scores, dim=-1)
# tensor([[1.0000, 0.0000, 0.0000, 0.0000],   # 位置0只看自己
#         [0.5000, 0.5000, 0.0000, 0.0000],   # 位置1看0和1
#         [0.3333, 0.3333, 0.3333, 0.0000],   # 位置2看0,1,2
#         [0.2500, 0.2500, 0.2500, 0.2500]])  # 位置3看全部
```

### GELU 激活函数

原始 Transformer 的 FFN 使用 ReLU 激活：$\text{ReLU}(x) = \max(0, x)$。GPT 将其替换为 GELU（Gaussian Error Linear Unit）：

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数（CDF）。

**GELU vs ReLU：**

- ReLU 对负值硬截断为 0，存在"死神经元"问题
- GELU 是平滑的近似门控——对正值近似恒等，对负值给予一个随幅度衰减的小概率通过
- GELU 在实践中带来更好的训练表现，成为 LLM 的标配激活函数

```python
import torch
import torch.nn.functional as F
import math

# 精确实现：基于误差函数 erf
def gelu_exact(x):
    """GELU 激活函数 - 精确版本（erf 公式）"""
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

# PyTorch 内置实现（推荐在实际项目中使用）
def gelu_builtin(x):
    """直接调用 F.gelu，内部已高度优化"""
    return F.gelu(x)

# 近似实现：基于 sigmoid 的快速近似
def gelu_fast(x):
    """GELU sigmoid 近似，速度快但精度略低"""
    return x * torch.sigmoid(1.702 * x)

# ReLU 对比
def relu(x):
    return torch.clamp(x, min=0)

# 验证各实现的一致性
x = torch.linspace(-3, 3, 7)
print("x:         ", x.tolist())
print("ReLU:      ", relu(x).tolist())
print("GELU_exact:", gelu_exact(x).tolist())
print("GELU_F:    ", gelu_builtin(x).tolist())
```

后续模型进一步发展出 SwiGLU 等门控激活函数（详见 [Llama 架构](llama.md)）。

### Pre-Normalization（预归一化）

原始 Transformer 使用 Post-Norm：先做子层计算，再做残差连接，最后做 LayerNorm。

```
Post-Norm:  y = Norm(F(x) + x)     先计算再归一化
Pre-Norm:   y = x + F(Norm(x))     先归一化再计算
```

GPT-2 开始改用 Pre-Norm，将 LayerNorm 移到子层**之前**。这一看似微小的改动带来了显著的训练稳定性提升：

- **梯度流更顺畅：** 残差连接不再被 LayerNorm 阻断，梯度可以从最后一层直接流回第一层
- **训练更稳定：** 深层网络（数十到上百层）的训练不再需要精细的学习率调整和 warmup
- **代价：** Pre-Norm 在最终输出前通常需要额外加一个 LayerNorm（Final LN），否则残差路径上的数值可能没有被归一化

**Post-Norm vs Pre-Norm 代码对比：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PostNormBlock(nn.Module):
    """Post-Norm：y = Norm(F(x) + x)"""
    def __init__(self, dim=512):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)   # 直接使用 PyTorch 内置 LayerNorm

    def forward(self, x):
        h = F.relu(self.linear1(x))
        h = self.linear2(h)
        return self.norm(h + x)         # 残差 + 归一化


class PreNormBlock(nn.Module):
    """Pre-Norm：y = x + F(Norm(x))"""
    def __init__(self, dim=512):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.norm(x)                # 先归一化
        h = F.relu(self.linear1(h))
        h = self.linear2(h)
        return x + h                    # 残差连接（不经过 Norm）
```

**关键区别：** Post-Norm 的输出方差始终为 1（被 Norm 约束），而 Pre-Norm 的输出方差随深度逐渐增大。因此 Pre-Norm 模型最后一层通常需要额外的 LayerNorm：

```python
# Pre-Norm 模型的最后一层必须加 LayerNorm
class PreNormModel(nn.Module):
    def __init__(self, dim=512, num_layers=6):
        super().__init__()
        self.blocks = nn.ModuleList(
            [PreNormBlock(dim) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(dim)  # 关键：最终归一化

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)  # 不加这层，方差会偏离
        return x
```

此后几乎所有 LLM 都采用 Pre-Norm 方案。

### KV Cache 原理与实现

KV Cache 是 Decoder-Only 模型推理时最重要的优化技术。理解它需要先理解自回归生成的过程。

**问题：自回归生成的重复计算**

在自回归生成中，每生成一个新 Token 都需要对整个序列做一次注意力计算。但注意到：已生成的 Token 的 Key 和 Value 在后续步骤中不会改变（因为因果掩码使得前面的 Token "看不到"后面的 Token）。

核心观察：next token prediction 预测的本质是第 $t=n$ 时的 $q_n$ 与 $k_{1:n}, v_{1:n}$ 做注意力计算。没有 KV Cache 时实际计算的是 $q_{1:n}$ 与 $k_{1:n}, v_{1:n}$，其中 $q_{1:n-1}$ 的计算全部是冗余的。

**无 KV Cache 的 Decoder（可以观察冗余计算）：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, mask):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # q, k, v 的 shape 都是 [bs, seq_len, dim]
        # 每一步都重新计算全部 token 的 q, k, v — 存在冗余
        attn = q @ k.transpose(-1, -2) / math.sqrt(self.dim)
        attn = attn.masked_fill(mask[:x.size(1), :x.size(1)], float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        return self.out_proj(out)
```

**有 KV Cache 的 Decoder：**

```python
class CachedAttention(nn.Module):
    """带 KV Cache 的注意力层"""
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.cache_k = None   # 缓存历史 Key
        self.cache_v = None   # 缓存历史 Value

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 首次调用（prefill）：初始化缓存
        if self.cache_k is None:
            self.cache_k = k
            self.cache_v = v
        else:
            # 后续调用（decode）：将新的 k, v 追加到缓存
            self.cache_k = torch.cat([self.cache_k, k], dim=1)
            self.cache_v = torch.cat([self.cache_v, v], dim=1)

        # q 与完整的 KV Cache 做注意力
        seq_len = self.cache_k.size(1)
        attn = q @ self.cache_k.transpose(-1, -2) / math.sqrt(self.dim)

        # 构造因果掩码：只屏蔽当前 q 不应看到的未来位置
        causal = torch.triu(torch.ones(q.size(1), seq_len, device=q.device), diagonal=seq_len - q.size(1) + 1).bool()
        attn = attn.masked_fill(causal.unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = attn @ self.cache_v
        return self.out_proj(out)
```

**生成过程的变化：**

```python
# 无 KV Cache：input_ids 不断累增
input_ids = torch.cat([input_ids, next_token_idx], dim=-1)
# 每次 forward 都传入完整序列 → q, k, v 全部重算

# 有 KV Cache：只传入新 token
input_ids = next_token_idx  # 只传 1 个 token
# forward 只计算新 token 的 q, k, v → k, v 追加到缓存
```

**两个阶段：**

- **Prefill（预填充）：** 将整个 prompt 一次性输入模型，并行计算所有 Token 的 K、V 并缓存。这个阶段是 compute-bound（计算密集型）
- **Decode（解码）：** 逐个生成新 Token，每次只计算一个 Token 的 Q、K、V。这个阶段是 memory-bound（内存带宽密集型），需要频繁加载投影权重和 KV Cache

**KV Cache 的内存开销：**

$$\text{KV Cache 大小} = \text{batch} \times \text{seq\_len} \times \text{dim} \times \text{num\_layers} \times 2 \times \text{dtype\_bytes}$$

例如，一个 70B 模型、序列长度 4096、batch size 32，KV Cache 可占数十 GB 显存。这也是为什么后续架构（GQA、MLA）持续优化 KV Cache 大小的原因。

## 代码实战：完整 GPT-2 模型

以下是基于 GPT-2 架构的完整实现，包含所有核心组件：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class GPTConfig:
    """GPT-2 模型配置"""
    vocab_size: int = 100       # 词表大小
    max_len: int = 512          # 最大序列长度
    d_model: int = 512          # 隐藏维度
    n_head: int = 8             # 注意力头数
    n_layer: int = 6            # Decoder 层数


class GPTAttention(nn.Module):
    """多头因果自注意力（合并 QKV 投影）"""
    def __init__(self, d_model, n_head=8):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal_mask=None):
        bs, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # 拆分多头: [bs, seq_len, d_model] → [bs, n_head, seq_len, head_dim]
        q = q.view(bs, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bs, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bs, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask[None, None, :seq_len, :seq_len], float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        context = weights @ v

        # 合并多头: [bs, n_head, seq_len, head_dim] → [bs, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)
        return self.out_proj(context)


class GPTFeedForward(nn.Module):
    """前馈网络：d_model → 4*d_model → d_model"""
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)     # 升维
        self.fc2 = nn.Linear(4 * d_model, d_model)      # 降维

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class GPTBlock(nn.Module):
    """GPT-2 Decoder 块（Pre-Norm 架构）"""
    def __init__(self, d_model=512, n_head=8):
        super().__init__()
        self.attn = GPTAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = GPTFeedForward(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, causal_mask=None):
        """Pre-Norm: y = x + F(Norm(x))"""
        x = x + self.attn(self.norm1(x), causal_mask=causal_mask)   # 第一个残差连接
        x = x + self.ffn(self.norm2(x))                              # 第二个残差连接
        return x


class GPT(nn.Module):
    """完整的 GPT-2 模型"""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Decoder 层堆叠
        self.blocks = nn.ModuleList(
            [GPTBlock(cfg.d_model, cfg.n_head) for _ in range(cfg.n_layer)]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)   # 最终 LayerNorm（Pre-Norm 必需）
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # 预计算因果掩码（上三角为 True 的 bool 矩阵）
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(cfg.max_len, cfg.max_len), diagonal=1).bool()
        )

    def forward(self, input_ids):
        bs, seq_len = input_ids.shape
        h = self.token_emb(input_ids)
        for block in self.blocks:
            h = block(h, causal_mask=self.causal_mask)
        h = self.final_norm(h)             # Pre-Norm 架构需要最终 LayerNorm
        logits = self.lm_head(h)           # [bs, seq_len, vocab_size]
        return logits


# 使用示例
cfg = GPTConfig(vocab_size=100, d_model=512, n_head=8, n_layer=6)
model = GPT(cfg)
x = torch.randint(100, [2, 16])
logits = model(x)
print(logits.shape)  # torch.Size([2, 16, 100])
```

### Perplexity（困惑度）计算

困惑度（Perplexity, PPL）是评价语言模型拟合程度的核心指标。PPL 越低，模型越好。

$$\text{PPL} = \exp\left(-\frac{1}{L}\sum_{t=1}^{L}\log p_\theta(x_t|x_{<t})\right) = \exp(\text{CrossEntropyLoss})$$

```python
import torch
import torch.nn as nn

IGNORE_INDEX = -100
loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

def compute_perplexity(model, input_ids, vocab_size):
    """计算模型在给定数据上的困惑度"""
    bs, seq_len = input_ids.shape

    # 构造 labels：将 input_ids 左移一位（预测下一个 token）
    labels = torch.zeros_like(input_ids, dtype=torch.long)
    labels[:, 0:seq_len-1] = input_ids[:, 1:seq_len]
    labels[:, seq_len-1] = IGNORE_INDEX  # 最后一个位置无目标

    with torch.no_grad():
        logits = model(input_ids)
    loss = loss_fn(
        logits.view(bs * seq_len, vocab_size),
        labels.view(bs * seq_len)
    )
    ppl = loss.exp()
    return ppl.item()

# 使用示例
# ppl = compute_perplexity(model, input_ids, vocab_size=100)
# 未训练模型的 PPL 约等于 vocab_size（随机猜测）
```

## 苏格拉底时刻

请停下来思考以下问题，不急于查看答案：

1. Encoder-Decoder 模型（如 T5）也能做通用生成任务，为什么 Decoder-Only 最终成为主流？这是必然还是偶然？是否存在 Encoder-Decoder 更适合的场景？
2. KV Cache 让推理快了很多，但它的内存占用与序列长度成正比——当上下文窗口扩展到 100K+ 时，KV Cache 会成为瓶颈吗？有哪些解决思路？
3. Pre-Norm 比 Post-Norm 训练更稳定，但有研究指出 Post-Norm 在收敛后模型质量略优。为什么会有这种差异？
4. 自回归生成是逐 Token 串行的，无法并行化——这是一个根本性限制吗？有哪些方法可以加速？（提示：考虑 Speculative Decoding、多 Token 预测等）

## 常见问题 & 面试考点

- **Q: GPT 的"预训练目标"是什么？** Causal Language Modeling（因果语言建模），即预测下一个 Token。损失函数是交叉熵 $\mathcal{L} = -\sum_t \log P(x_t | x_{<t})$。
- **Q: KV Cache 能节省多少计算量？** 将总计算复杂度从 $O(n^3)$ 降到 $O(n^2)$（$n$ 为序列长度）。单步推理从 $O(n^2)$ 降到 $O(n)$。
- **Q: 什么是 Prefill 和 Decode 阶段？** Prefill 是处理 prompt 阶段（计算密集），Decode 是逐 Token 生成阶段（内存带宽密集）。二者对硬件的需求不同。
- **Q: GPT-1/2/3 之间的核心区别是什么？** 主要是规模差异。GPT-1 = 1.17 亿参数，GPT-2 = 15 亿参数（Pre-Norm），GPT-3 = 1750 亿参数（In-Context Learning 涌现）。架构变化不大，核心发现是 scaling law。

## 推荐资源

### 论文

- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) by Radford et al. — GPT-1 原始论文。
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) by Radford et al. — GPT-2 论文，提出 zero-shot 任务迁移。
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) by Brown et al. — GPT-3 论文，In-Context Learning 的开山之作。

### 博客与可视化

- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) by Jay Alammar — GPT-2 架构的可视化讲解。
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy — 从零实现 GPT 的经典视频教程，配套代码 [nanoGPT](https://github.com/karpathy/nanoGPT)。

### 代码参考

- [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) — Karpathy 的 GPT-2 (124M) 单文件复现仓库（MIT 许可），配套 [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) 视频。重点对照本页 §核心实现 与下列文件：
  - [train_gpt2.py L12-L40](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L12-L40) — `CausalSelfAttention`：QKV 合并投影 + `F.scaled_dot_product_attention(is_causal=True)` 直接拿到 Flash Attention。
  - [train_gpt2.py L42-L55](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L42-L55) — `MLP`：4× hidden + `GELU(approximate='tanh')`（GPT-2 原版用的就是 tanh 近似）。
  - [train_gpt2.py L57-L69](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L57-L69) — `Block`：Pre-Norm 结构（`x = x + attn(ln_1(x))`），与 GPT-1 的 Post-Norm 形成鲜明对比。
  - [train_gpt2.py L79-L145](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L79-L145) — `GPT` 主类：`wte/wpe` 嵌入、堆叠 Block、**weight tying**（`transformer.wte.weight = lm_head.weight`，省下 ~38M 参数）、按 `(2 * n_layer)^(-0.5)` 对残差分支做的初始化缩放。
  - [train_gpt2.py L147-L200](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L147-L200) — `from_pretrained`：从 HuggingFace 装载 GPT-2 权重并处理 Conv1D ↔ Linear 的转置兼容（OpenAI 原版用 TF Conv1D，HF 用 Linear，权重需要 `.t()`）。
- [rasbt/LLMs-from-scratch · ch04](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04) — Sebastian Raschka《Build a Large Language Model (From Scratch)》第 4 章配套代码，从零搭一个 124M GPT-2。和 nanoGPT 是教学法上的互补关系：nanoGPT 追求"最少代码量"，这里追求"每一步对应书里一张图"。
  - [ch04/01_main-chapter-code/gpt.py](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/gpt.py)（277 行） — 单文件实现 `LayerNorm` / `GELU` / `FeedForward` / `MultiHeadAttention` / `TransformerBlock` / `GPTModel` 与文末的 `generate_text_simple`，每个类都是独立组件，逐层堆出来一个可跑通的 GPT。
  - [ch04/01_main-chapter-code/ch04.ipynb](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/ch04.ipynb) — 同样的内容拆成 jupyter cell，配合书里的图谱讲"为什么要 Pre-Norm / 为什么残差里要除 √(2·n_layer)"等设计取舍。
  - [ch04/03_kv-cache](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/03_kv-cache) / [04_gqa](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/04_gqa) / [05_mla](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/05_mla) / [06_swa](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/06_swa) / [07_moe](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/07_moe) — 在主章节 GPT 之上的注意力变体与 MoE 改造，可与本仓 [attention.md](../architecture/attention.md) 对照阅读。
