---
title: "深度剖析 GPT-2"
description: "300 行 PyTorch 实现完整的 GPT-2，从 Attention 到训练循环"
topics: [GPT-2, transformer, implementation, from-scratch, causal-attention, training-loop]
prereqs: [architecture/transformer, architecture/attention, fundamentals/neural-networks]
---
# 深度剖析 GPT-2

> **一句话总结:** 用不到 300 行 PyTorch 代码实现一个完整的 GPT-2 模型——覆盖 Causal Self-Attention、FFN、LayerNorm、位置编码、训练循环和文本生成，每一行都有数学对应。

## 为什么要从零实现 GPT-2

1. **GPT-2 是所有现代 LLM 的原型**：Llama、DeepSeek、Qwen 都是 GPT 架构的变体
2. **结构简洁**：只有 Decoder（没有 Encoder），核心组件只有 Attention + FFN + LayerNorm
3. **可以在笔记本上训练**：117M 参数的 GPT-2 small 可以在单张消费级 GPU 上训练
4. **每个组件都有明确的数学公式**：从代码到公式的对应清晰直接

---

## 模型架构总览

```
Input Token IDs: [batch_size, seq_len]
       │
       ▼
┌─────────────────┐
│ Token Embedding  │  wte: [vocab_size, d_model]
│ + Pos Embedding  │  wpe: [max_seq_len, d_model]
└────────┬────────┘
         │
    ┌────▼────┐
    │ Block 1 │ ─── LayerNorm → CausalSelfAttn → LayerNorm → FFN
    ├─────────┤
    │ Block 2 │
    ├─────────┤
    │  ...    │
    ├─────────┤
    │ Block N │
    └────┬────┘
         │
    ┌────▼────────┐
    │ LayerNorm   │
    │ LM Head     │  [d_model → vocab_size]（与 wte 共享权重）
    └─────────────┘
         │
         ▼
  logits: [batch_size, seq_len, vocab_size]
```

---

## 第一步：配置

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257    # GPT-2 的 BPE 词表大小
    max_seq_len: int = 1024    # 最大序列长度
    n_layer: int = 12          # Transformer Block 数量
    n_head: int = 12           # 注意力头数
    d_model: int = 768         # 隐藏层维度
    dropout: float = 0.1       # Dropout 率

    @property
    def d_head(self):
        """每个注意力头的维度"""
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head
```

GPT-2 的各版本配置：

| 版本 | n_layer | n_head | d_model | 参数量 |
|------|---------|--------|---------|--------|
| Small | 12 | 12 | 768 | 117M |
| Medium | 24 | 16 | 1024 | 345M |
| Large | 36 | 20 | 1280 | 762M |
| XL | 48 | 25 | 1600 | 1.5B |

---

## 第二步：Causal Self-Attention

因果自注意力 = Scaled Dot-Product Attention + 因果掩码（下三角）。

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V
$$

其中 $M$ 是因果掩码矩阵：$M_{ij} = 0$ 当 $i \geq j$，$M_{ij} = -\infty$ 当 $i < j$。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        
        # Q, K, V 合并为一个线性层（效率更高）
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        # 输出投影
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # 因果掩码：下三角矩阵
        # register_buffer 不参与梯度计算，但会随模型保存/加载
        self.register_buffer("mask", torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len)
        ).view(1, 1, config.max_seq_len, config.max_seq_len))
    
    def forward(self, x):
        B, T, C = x.shape  # batch, seq_len, d_model
        
        # 计算 Q, K, V
        qkv = self.c_attn(x)                           # [B, T, 3*C]
        q, k, v = qkv.split(self.d_model, dim=2)       # 各 [B, T, C]
        
        # 分多头: [B, T, C] → [B, n_head, T, d_head]
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, nh, T, T]
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和
        out = attn @ v                                   # [B, nh, T, d_head]
        
        # 合并多头: [B, nh, T, d_head] → [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # 输出投影
        out = self.c_proj(out)
        out = self.dropout(out)
        return out
```

::: tip 为什么 Q, K, V 合并成一个线性层？
分开写是 `q = W_q(x); k = W_k(x); v = W_v(x)` 三次矩阵乘法。合并后只需一次 `qkv = W_qkv(x)` 再 split，GPU 对大矩阵乘法更高效。
:::

---

## 第三步：FFN（前馈网络）

GPT-2 的 FFN 是两层 MLP + GELU 激活：

$$
\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
$$

隐藏层维度 = 4 × d_model（这是经验值）。

```python
class FFN(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model)
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)         # GPT-2 用 GELU；Llama 用 SwiGLU
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

---

## 第四步：Transformer Block

每个 Block = Attention + FFN，各自带 LayerNorm 和残差连接。

GPT-2 使用 **Pre-LN** 架构（先 Norm 再 Attention/FFN）：

$$
x = x + \text{Attention}(\text{LN}(x))
$$
$$
x = x + \text{FFN}(\text{LN}(x))
$$

```python
class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # 残差 + Attention
        x = x + self.ffn(self.ln_2(x))    # 残差 + FFN
        return x
```

---

## 第五步：完整 GPT 模型

```python
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Embedding 层
        self.wte = nn.Embedding(config.vocab_size, config.d_model)   # Token Embedding
        self.wpe = nn.Embedding(config.max_seq_len, config.d_model)  # Position Embedding
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # 最终 LayerNorm
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # LM Head（与 wte 共享权重）
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # 权重共享！
        
        # 初始化
        self.apply(self._init_weights)
        print(f"GPT model with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def _init_weights(self, module):
        """GPT-2 风格的权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"序列长度 {T} 超过最大值 {self.config.max_seq_len}"
        
        # Embedding: Token + Position
        pos = torch.arange(0, T, device=input_ids.device)
        tok_emb = self.wte(input_ids)     # [B, T, d_model]
        pos_emb = self.wpe(pos)           # [T, d_model]
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x)
        
        # LM Head
        x = self.ln_f(x)
        logits = self.lm_head(x)          # [B, T, vocab_size]
        
        # 计算 Loss（如果提供了 targets）
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # [B*T, vocab_size]
                targets.view(-1)                    # [B*T]
            )
        
        return logits, loss
```

::: tip 权重共享（Weight Tying）
`self.lm_head.weight = self.wte.weight` 让输入 Embedding 和输出投影共享同一组权重。直觉：Token Embedding 学到的语义空间应该和预测下一个 Token 的空间一致。这节省了 `vocab_size × d_model` 的参数（GPT-2: 50257 × 768 ≈ 38M 参数）。
:::

---

## 第六步：文本生成

```python
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, 
             temperature=0.8, top_k=40):
    """自回归文本生成"""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(next(model.parameters()).device)
    
    for _ in range(max_new_tokens):
        # 截断到 max_seq_len
        idx = input_ids if input_ids.size(1) <= model.config.max_seq_len \
              else input_ids[:, -model.config.max_seq_len:]
        
        # 前向传播
        logits, _ = model(idx)
        logits = logits[:, -1, :] / temperature  # 只取最后一个位置
        
        # Top-k 过滤
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # 采样
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        
        input_ids = torch.cat([input_ids, next_id], dim=1)
        
        # 遇到 EOS 停止
        if next_id.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
```

---

## 第七步：训练循环

```python
from transformers import GPT2Tokenizer

def train():
    # 配置（用小一点的模型方便训练）
    config = GPTConfig(
        vocab_size=50257,
        max_seq_len=256,    # 缩短序列方便实验
        n_layer=6,          # 6 层（GPT-2 small 是 12 层）
        n_head=6,
        d_model=384,
        dropout=0.1,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # 准备训练数据（这里用简单文本演示）
    text = open("train.txt").read()
    tokens = tokenizer.encode(text)
    data = torch.tensor(tokens, dtype=torch.long)
    
    # 训练
    model.train()
    batch_size = 8
    seq_len = config.max_seq_len
    
    for step in range(1000):
        # 随机采样 batch
        ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
        
        # 前向 + 反向
        logits, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
            # 生成样本
            sample = generate(model, tokenizer, "The ", max_new_tokens=50)
            print(f"  Sample: {sample}")
    
    return model

model = train()
```

---

## 参数量验证

手动计算 GPT-2 Small (d=768, L=12, V=50257) 的参数量：

```python
def count_params(config):
    d = config.d_model
    V = config.vocab_size
    L = config.n_layer
    S = config.max_seq_len
    
    # Embedding
    wte = V * d                          # 38,597,376
    wpe = S * d                          # 786,432
    
    # 每个 Block
    ln = 2 * (d + d)                     # LayerNorm (2 个，各有 weight + bias)
    attn_qkv = d * 3 * d + 3 * d        # c_attn (QKV 合并)
    attn_proj = d * d + d                # c_proj
    ffn_fc = d * 4 * d + 4 * d          # c_fc
    ffn_proj = 4 * d * d + d            # c_proj
    block = ln + attn_qkv + attn_proj + ffn_fc + ffn_proj
    
    # 最终 LayerNorm
    ln_f = d + d
    
    # LM Head 与 wte 共享，不额外计算
    total = wte + wpe + L * block + ln_f
    
    print(f"Token Embedding: {wte:>12,}")
    print(f"Position Embedding: {wpe:>9,}")
    print(f"Per Block: {block:>15,} × {L} = {L*block:,}")
    print(f"Final LayerNorm: {ln_f:>10,}")
    print(f"Total: {total:>18,}")
    return total

count_params(GPTConfig())
# Token Embedding:   38,597,376
# Position Embedding:   786,432
# Per Block:       7,087,872 × 12 = 85,054,464
# Final LayerNorm:      1,536
# Total:          124,439,808  ≈ 124M（与官方 GPT-2 Small 一致）
```

---

## 从 GPT-2 到现代 LLM

GPT-2 是起点，现代 LLM 在其基础上做了这些改进：

| 组件 | GPT-2 | 现代 LLM (Llama 3) |
|------|-------|-------------------|
| 位置编码 | 学习式 Absolute PE | RoPE (旋转位置编码) |
| Norm | LayerNorm | RMSNorm |
| 激活函数 | GELU | SwiGLU |
| 注意力 | MHA | GQA (Grouped Query Attention) |
| FFN 维度 | 4d | 8d/3（SwiGLU 需要三个矩阵） |
| 词表大小 | 50K | 128K |
| Bias | 有 | 无（去掉所有 bias） |

理解了 GPT-2 的每一行代码，再看 Llama、DeepSeek 的改进就是"哪里换了什么"的问题。

---

## 苏格拉底时刻

1. 因果掩码为什么是下三角矩阵？如果不加掩码，模型训练会怎样？
2. 为什么 `lm_head` 和 `wte` 共享权重？如果不共享会怎样？
3. GPT-2 的参数量主要集中在哪个组件？Embedding 占比多少？
4. 如果把 `max_seq_len` 从 1024 改成 4096，哪些地方需要修改？哪些不需要？
5. 这个实现和 HuggingFace 的 GPT2Model 有什么区别？缺少了哪些工程优化？

---

## 推荐资源

- **Andrej Karpathy: nanoGPT** — 本章代码的灵感来源，约 300 行实现 GPT-2
- **Andrej Karpathy: "Let's build GPT"** — 2 小时视频从零实现 GPT，极其推荐
- **Radford et al. "Language Models are Unsupervised Multitask Learners"** — GPT-2 原始论文
- **The Annotated Transformer** — Transformer 的逐行注释实现
