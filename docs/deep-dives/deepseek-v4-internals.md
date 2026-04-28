---
title: DeepSeek-V4 内部机制
description: Hybrid Attention（CSA + HCA）、mHC、Muon、On-Policy Distillation、TileLang/MegaMoE——把 1M 上下文 27% FLOPs / 10% KV 的"超长序列效率跃迁"拆开看
topics: [deepseek-v4, csa, hca, hybrid-attention, mhc, hyper-connections, muon, on-policy-distillation, anticipatory-routing, tilelang, megamoe, fp4-qat]
prereqs: [architecture/deepseek, architecture/attention, training/pretraining, training/alignment]
---

# DeepSeek-V4 内部机制

> **一句话总结：** DeepSeek-V4 把超长序列效率推到了开源 LLM 的新边界——靠 **Hybrid Attention (CSA + HCA)** 把 1M token 单步推理 FLOPs 压到 V3.2 的 27%、KV cache 压到 10%；靠 **Manifold-Constrained Hyper-Connections (mHC)** 让残差流的谱半径被 Sinkhorn-Knopp 投影到 ≤1，从而支撑深堆栈；靠 **Muon + Hybrid Newton-Schulz** 跑出更快的收敛与更稳的训练；最后用 **On-Policy Distillation** 把十几个领域专家融合进单一统一模型——这份 deep-dive 把 V4 技术报告（DeepSeek-AI 2026）拆成可复用的工程拼图。

::: info 阅读提示
本文聚焦 V4 区别于 V3 / V3.2 的"独特机制"。如果你想看通用的 MoE / MLA / FP8 / DualPipe 等基础，请先读 [DeepSeek-V3 架构](/architecture/deepseek)、[Attention 深度实现](/architecture/attention)、[预训练](/training/pretraining)、[对齐](/training/alignment)。源码以 V4-Pro 官方推理仓库（开源 MIT）为准：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference>。
:::

## V4 一图概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                       DeepSeek-V4 系列（2026-04 preview）           │
│                                                                     │
│  V4-Flash ── 284B / 13B 激活 / 43 层 / d=4096 / 32T tokens          │
│           │  256 routed + 1 shared expert / Top-6 / inter=2048      │
│           │                                                         │
│  V4-Pro ─── 1.6T / 49B 激活 / 61 层 / d=7168 / 33T tokens           │
│           │  384 routed + 1 shared expert / Top-6 / inter=3072      │
│           │                                                         │
│  Attention  Hybrid（CSA + HCA 交替） + 前 2 层 SWA + n_win=128       │
│  Residual   mHC，扩展系数 n_hc=4，Sinkhorn-Knopp t_max=20            │
│  Optimizer  Muon（主体）+ AdamW（embedding / head / RMSNorm / mHC）  │
│  Stability  Anticipatory Routing + SwiGLU Clamping([-10,10] / cap10) │
│  Training   FP8 主体 + FP4 QAT（MoE 权重 + CSA 索引器 QK 路径）     │
│  Post-Train Specialist (SFT + GRPO + GRM) → On-Policy Distillation   │
└─────────────────────────────────────────────────────────────────────┘
```

| 维度 | V3.2 | V4-Flash | V4-Pro | 出处 |
|------|------|---------|--------|------|
| 总参 / 激活 | 671B / 37B | 284B / 13B | 1.6T / 49B | V4 报告 §4.2.1 |
| 1M 单 token FLOPs（相对 V3.2） | 100% | 10% | 27% | §1, Figure 1 |
| 1M KV cache（相对 V3.2） | 100% | 7% | 10% | §1 |
| 注意力 | MLA | Hybrid CSA+HCA | Hybrid CSA+HCA | §2.3 |
| 优化器 | AdamW | Muon (主体) | Muon (主体) | §2.4 |

::: tip 你应该带走的"心理图"
V4 不是 V3 的扩参数版——它把"超长序列效率"作为第一性约束反推架构：先在 attention 端用 **两种压缩 + 一个稀疏 selector** 把 KV 拍扁，再在残差端用 **几何约束的 hyper-connection** 抵消由此带来的信号传播不稳定，最后在优化器端换成 **基于 Newton-Schulz 正交化的 Muon** 来吃满低 FLOPs/Byte。下面按这条逻辑链一节一节展开。
:::

## 1. Hybrid Attention：CSA + HCA 的双轨压缩

### 1.1 为什么需要"混合"

朴素 attention 的二次复杂度在 1M token 这种尺度下完全不可承受；V3 系的 MLA + V3.2 引入的 **DSA（DeepSeek Sparse Attention）** 已经把 KV cache 压到了开源最低水平之一，但仍然不够：CSA 把"压缩"和"稀疏"叠加，HCA 把"压缩率"再推高一阶，两者**层间交错**部署——一些层做"细粒度稀疏看局部 + 长程"，一些层做"重压缩看全局摘要"——总体把 1M 上下文的 KV 压到 BF16 GQA8 head=128 baseline 的 ~2%（V4 报告 §2.3.4）。

V4 的 transformer block（§2，Figure 2）整体结构：

```
Embedding → [Pre-Block Mixing → CSA/HCA → Post-Block Mixing → ⊕ residual
            → Pre-Block Mixing → DeepSeekMoE → Post-Block Mixing → ⊕] × L
            → Prediction Head + MTP Modules
```

**Mixing** 即 mHC 的 A_l / B_l / C_l 三个映射；**⊕** 是 residual 加法。下面分别展开两条 attention 路径。

### 1.2 CSA：先压缩再稀疏选 top-k

CSA（Compressed Sparse Attention，§2.3.1）做两件事：把每 m 个 KV token 压成一条"摘要 entry"，然后让 query 通过 **Lightning Indexer** 做 top-k 稀疏选择。给定隐状态序列 $H \in \mathbb{R}^{n \times d}$：

**Step 1 — 双路压缩。** 投影出两套 KV entry 与对应的"压缩权重"：

$$
C^a = H \cdot W^{aKV},\quad C^b = H \cdot W^{bKV}, \qquad Z^a = H \cdot W^{aZ},\quad Z^b = H \cdot W^{bZ}
$$

每 $m$ 个 KV entry 被合并成 1 条压缩 entry $C_i^{\text{Comp}}$。注意 $C^a$ 与 $C^b$ 的索引**有重叠**（错开半窗），从而 $C_i^{\text{Comp}}$ 实质上吸收了 $2m$ 条原始 KV 的信息，但序列长度仍被压到原来的 $1/m$：

$$
[S^a_{m i:m(i+1)-1};\; S^b_{m(i-1):m i-1}] = \mathrm{Softmax}_{\text{row}}\!\big( [Z^a_{m i:m(i+1)-1} + B^a;\; Z^b_{m(i-1):m i-1} + B^b] \big)
$$

$$
C_i^{\text{Comp}} = \sum_{j} S^a_j \odot C^a_j + \sum_{j} S^b_j \odot C^b_j
$$

**Step 2 — Lightning Indexer 选 top-k。** 对压缩后的 entry 再走一次同样的"压缩"得到 indexer key $K^{I\text{Comp}} \in \mathbb{R}^{(n/m)\times c^I}$；query 端经低秩投影 $\mathbf{c}^Q_t = h_t \cdot W^{DQ}$ 拿到 indexer queries（多头），再 ReLU + 加权求得 index score：

$$
I_{t,s} = \sum_{h=1}^{n^I_h} w^I_{t,h} \cdot \mathrm{ReLU}\!\big(\mathbf{q}^I_{t,h} \cdot K^{I\text{Comp}}_s\big)
$$

按 $I_{t,:}$ 的 top-k 选出 $C_t^{\text{SprsComp}}$ 喂给 core attention；core attention 是 **Shared-KV MQA**——压缩 entry 同时充当 K 和 V，head 共享同一份 K/V 但有独立 Q。

**Step 3 — Grouped Output Projection.** $cn_h$ 维度太大，直接投回 $d$ 太贵。把 $n_h$ 个 head 输出分成 $g$ 组，每组先压到中间维 $d_g < c\cdot n_h/g$，再拼接投回 $d$（V4-Pro: $g=16$, $d_g=1024$）。

### 1.3 HCA：更狠的压缩 + 不做稀疏

HCA（Heavily Compressed Attention，§2.3.2）和 CSA 的差别只在两点：

1. **压缩率拉到 $m' \gg m$**（V4 取 $m=4$, $m'=128$，即每 128 个 token 压成 1 条）。
2. **不做 top-k 选择**——压完之后直接对全部压缩 entry 做 dense MQA。

```
HCA 工作流：
  H ──(W^KV)──→ C ∈ R^{n×c}
  H ──(W^Z) ──→ Z ∈ R^{n×c}
                │
                ▼
  每 m' 条 C 在 Z 的 softmax 权重下合并 ──→ C^Comp ∈ R^{(n/m')×c}
                │
                ▼
  q_t ──→ c^Q_t (低秩) ──→ {q_{t,1}, ..., q_{t,n_h}}
                │
                ▼
  CoreAttn(query=q_{t,i}, key=C^Comp, value=C^Comp)  // 不再 top-k
                │
                ▼
  Grouped Output Projection ──→ d 维输出
```

直觉：**CSA 看"近 + 选出来的远"，HCA 看"远的摘要全景"**。两者交错叠层，互补长短程。

### 1.4 三个补丁：让 hybrid attention 真的能用

§2.3.3 列出了 4 个生产级细节：

| 细节 | 机制 | 解决的问题 |
|------|------|----------|
| **Q/KV Entry Normalization** | core attention 之前对 query 每个 head + 唯一一个压缩 KV head 做 RMSNorm | 防止 logit 爆炸（这也是 V4 不需要 QK-Clip 的原因，§2.4） |
| **Partial RoPE** | 仅对 q / KV / 输出 $\mathbf{o}_{t,i}$ 的最后 64 维做 RoPE，输出端用位置 $-i$ | 压缩 KV 同时做 K 和 V 时，朴素输出会带绝对位置；负位置 RoPE 让输出携带相对位置 |
| **Sliding-Window 旁路** | 每个 query 额外加一个 $n_{\text{win}}$（=128）的未压缩 KV 切片 | 严格因果性下，query 看不到自己所在压缩块的同 batch 内信息——SWA 弥补"近邻盲区" |
| **Attention Sink** | 每个 head 加一组可学习 logit $\{z'_h\}$ 进入 softmax 分母 | 允许 query 把"总注意力质量"调到 < 1 甚至 ≈0（不强制每个 token 都 attend 到东西） |

$$
s_{h,i,j} = \frac{\exp(z_{h,i,j})}{\sum_k \exp(z_{h,i,k}) + \exp(z'_h)}
$$

### 1.5 效率账（§2.3.4）

KV 用混合精度存：RoPE 维度 BF16、其余维度 FP8——KV cache 比纯 BF16 直接砍半。CSA Lightning Indexer 内部走 FP4。1M 上下文配置下：

| 模型 | 单 token FLOPs（vs V3.2） | KV cache（vs V3.2） |
|------|------------------------|--------------------|
| V4-Pro | 27% | 10% |
| V4-Flash | 10% | 7% |

vs **BF16 GQA8 head=128** 这条工业基线：V4 KV cache 在 1M 设置下 **≈ 2%**。

::: details 报告里没明说但值得想一遍：CSA 比 V3.2 DSA 强在哪？
V3.2 DSA 是**直接在原始 KV 上做 top-k**，等价于 sequence_length × top_k 的注意力图；CSA 在 top-k 之前先压了一道 $1/m$ 的"压缩 entry"，所以 selector 在搜索空间小一个量级的 entry 池里挑——既减小 indexer 计算开销，又让 sparse pattern 更"语义化"。HCA 则承担了 V3.2 DSA 完全没承担的"全局摘要"角色。
:::

## 2. mHC：把残差映射约束在 Birkhoff 多面体

### 2.1 标准 Hyper-Connections 与"撑不住深"的问题

普通 HC（Hyper-Connections，Zhu et al. 2025）把 residual stream 从 $\mathbb{R}^d$ 扩到 $\mathbb{R}^{n_{\text{hc}} \times d}$，每层引入三组线性映射 $A_l \in \mathbb{R}^{1\times n_{\text{hc}}}$、$B_l \in \mathbb{R}^{n_{\text{hc}} \times n_{\text{hc}}}$、$C_l \in \mathbb{R}^{n_{\text{hc}} \times 1}$：

$$
X_{l+1} = B_l X_l + C_l \,\mathcal{F}_l(A_l X_l)
$$

V4 报告（§2.2）指出：HC 在大量层堆叠时**频繁数值不稳定**——这是 V4 不能直接用 HC 的根本原因。

### 2.2 mHC 的核心：把 $B_l$ 投到 doubly stochastic 矩阵流形

mHC（Manifold-Constrained Hyper-Connections，Xie et al. 2026）把 $B_l$ 限制在 **Birkhoff 多面体** $\mathcal{M}$（双随机矩阵集合）：

$$
B_l \in \mathcal{M} := \{ M \in \mathbb{R}^{n \times n} \mid M \mathbf{1}_n = \mathbf{1}_n,\; \mathbf{1}_n^T M = \mathbf{1}_n^T,\; M \geq 0 \}
$$

这一约束的物理意义：**$\|B_l\|_2 \leq 1$**——映射非膨胀，因此前向反向都不会因层数堆叠而爆炸/塌缩；而且 $\mathcal{M}$ 在矩阵乘法下封闭，深堆栈天然稳定。$A_l$ 与 $C_l$ 则用 Sigmoid 卡到非负有界，避免相互抵消。

### 2.3 工程实现：动态 + 静态 + Sinkhorn-Knopp

参数被拆成"输入相关动态分量"加"输入无关静态分量"——先 RMSNorm-flatten 输入 $\hat{X}_l = \mathrm{RMSNorm}(\mathrm{vec}(X_l))$，再经三组可学习权重 + 静态偏置生成原始 $\tilde{A}_l, \tilde{B}_l, \tilde{C}_l$（§2.2 公式 3-5），然后施加约束：

$$
A_l = \sigma(\tilde{A}_l), \qquad C_l = 2\sigma(\tilde{C}_l)
$$

$$
B_l = M^{(t_{\max})}, \quad M^{(0)} = \exp(\tilde{B}_l), \quad M^{(t)} = \mathcal{T}_r(\mathcal{T}_c(M^{(t-1)}))
$$

其中 $\mathcal{T}_r / \mathcal{T}_c$ 是行/列归一化（Sinkhorn-Knopp 迭代），V4 取 $t_{\max}=20$；$n_{\text{hc}}=4$。

::: tip 工程视角的"为什么是 4"
$n_{\text{hc}}$ 是新增的"通道维度"，与 $d$ 解耦——当 $n_{\text{hc}} \ll d$（V4: 4 vs 7168）时，扩展成本几乎可忽略，但表达力（每层有 $n_{\text{hc}}^2$ 个独立 residual 路径）显著上升。代价是 $\mathrm{Mat}(\hat{X}_l W^{\text{res}}_l)$ 这步矩阵乘法的输出维只有 $n_{\text{hc}}^2 = 16$；§3.4 专门提到这是少数必须用 split-k GEMM 的场景。
:::

### 2.4 训练稳定性的代价怎么吃下来

§3.5.2 给出的优化菜单：

1. **mHC 专用融合内核**（训练 + 推理）。
2. **Selective recomputation**：只 recompute 大多数 hidden states 与所有归一化输入，跳过算力密集 op。
3. **DualPipe 1F1B 重排**——把 mHC 增大的 pipeline 通信塞进现有重叠窗口。

最终：mHC 在 overlapped 1F1B stage 上的 wall-clock 开销控制在 **6.7%**。

## 3. Muon Optimizer：基于 Newton-Schulz 正交化

### 3.1 谁用 Muon、谁用 AdamW

V4 是 DeepSeek 第一次大规模在 LLM 主体上换掉 AdamW（§2.4 + §4.2.2）。**保留 AdamW** 的模块：embedding、prediction head、所有 RMSNorm 权重、所有静态 bias / gating factor、**整个 mHC 模块**。其余所有可学习矩阵走 Muon。原因是 Muon 的更新方向需要 SVD 意义下的"正交化"，对小矩阵 / 标量参数收益不大，对 mHC 这种带几何约束的更会"打架"。

### 3.2 Muon 的核心步骤（V4 报告 Algorithm 1）

```
对每个训练 step t、每个逻辑独立权重 W ∈ R^{n×m}：
  G_t  = ∇_W L_t(W_{t-1})                       # 梯度
  M_t  = μ M_{t-1} + G_t                        # Nesterov 风格动量
  O'_t = HybridNewtonSchulz(μ M_t + G_t)        # 关键：把更新矩阵正交化
  O_t  = O'_t · √max(n,m) · γ                   # 复用 AdamW 学习率的 RMS 缩放
  W_t  = W_{t-1} · (1 - η λ) − η O_t            # 解耦 weight decay + 更新
```

直观理解：AdamW 的 update 方向由二阶矩归一化每个坐标轴；Muon 的 update 方向由 Newton-Schulz 把整个矩阵 update 推向"$UV^T$"——即把奇异值整体拉到 1，等于对 update 矩阵做了 spectral 归一化。

### 3.3 Hybrid Newton-Schulz：两阶段系数（§2.4）

Newton-Schulz 迭代：

$$
M_k = a M_{k-1} + b (M_{k-1} M_{k-1}^T) M_{k-1} + c (M_{k-1} M_{k-1}^T)^2 M_{k-1}
$$

V4 跑 10 步、分两段：

| 阶段 | 步数 | $(a, b, c)$ | 目标 |
|------|------|-------------|------|
| 快收敛段 | 8 | $(3.4445, -4.7750, 2.0315)$ | 把奇异值快速拉到 1 附近 |
| 稳定段 | 2 | $(2, -1.5, 0.5)$ | 把奇异值精确钉在 1 |

为什么 V4 不再需要 QK-Clip（参考 [Kimi K2 deep-dive](/deep-dives/kimi-k2-internals)）：因为 §2.3.3 已经在 attention query / KV entry 上加了 RMSNorm，logit 不再爆炸。

## 4. MoE：与 V3 / V3.2 的精确差异

V4 仍然用 DeepSeekMoE（细粒度 routed + 共享专家）+ 无辅助损失负载均衡，但有四处具体改动（§2.1）：

1. **激活函数：Sigmoid → Sqrt(Softplus)** 计算 affinity score——避免 sigmoid 在两端饱和导致 routing 信号弱化。
2. **保留 sequence-wise balance loss**（权重 0.0001），但放开 V3 中"每个 token 路由的目标节点数 ≤ N"的硬约束。
3. **前 3 层 MoE 用 Hash 路由**——直接由 token id 经哈希函数选 expert，免去 routing 的训练不稳定。
4. **MTP 完全沿用 V3 配置**，不动。

具体配置（§4.2.1）：

| 配置项 | V4-Flash | V4-Pro |
|--------|---------|--------|
| Transformer 层数 | 43 | 61 |
| hidden $d$ | 4096 | 7168 |
| Routed expert 数 | 256 | 384 |
| Shared expert 数 | 1 | 1 |
| 每 token 激活 | 6 routed + 1 shared | 6 routed + 1 shared |
| Routed expert intermediate | 2048 | 3072 |
| Hybrid attention 起点 | 第 3 层（前 2 层纯 SWA） | 第 2 层（前 1 层 HCA） |
| Sliding-window $n_{\text{win}}$ | 128 | 128 |
| MTP depth | 1 | 1 |

## 5. 训练稳定性：Anticipatory Routing + SwiGLU Clamping

§4.2.3 直白承认：1.6T MoE 训练遇到了不少 loss spike。两个被验证有效的工程招式——

### 5.1 Anticipatory Routing：异步 routing 索引

观察：spike 几乎总是和 MoE routing 在 outlier token 处的"突变"相关。修复思路是**解耦 routing 网络与主干网络的同步更新**——

```
正常步（无 spike）：
  step t: 用 θ_t 算 forward + backward + routing 索引

Spike-detection 模式（启动 Anticipatory Routing）：
  step t-Δt: 多读一份数据，用 θ_{t-Δt} 预先算出 step t 的 routing 索引并缓存
  step t:    forward / backward 仍然用 θ_t，但 routing 索引来自 t-Δt 时刻
```

延迟一个 step 的 routing 索引让"突变 token"先被压在历史 routing 上，避免同一步既爆 loss 又爆 routing。spike 缓解后会自动回到标准训练。整体额外 wall-clock ≈ 20%，但只在检测到 spike 后才启用。

### 5.2 SwiGLU Clamping

把 SwiGLU 线性分量 clamp 到 $[-10, 10]$，gate 上界 cap 到 10——经验上有效压制 outlier，作者承认"理论解释还不充分"，open question。

## 6. 基础设施亮点（节选）

### 6.1 TileLang + MegaMoE 融合内核（§3.1, §3.2）

V4 的 MoE EP（Expert Parallelism）调度叫 **wave-based scheduling**——把同一层的 routed expert 拆成 wave，wave A 的 dispatch 完就立刻开始 wave A 的 GEMM，同时 wave B 的 dispatch 继续在 NIC 上跑。理论加速 1.92×（V4-Flash 配置），实测 1.50–1.73×。CUDA mega-kernel 实现叫 **MegaMoE**，已合并到 DeepGEMM PR #304。

报告还提了一个对硬件设计的要求：

$$
\frac{C}{B} \leq 2d = 6144 \text{ FLOPs/Byte}
$$

即每 GBps 的互联带宽够覆盖 6.1 TFLOPs 计算时通信就完全被掩盖。**V4 把 LLM 训练对带宽的实际需求量化了**——这对未来芯片设计的指导意义远超 V4 本身。

### 6.2 FP4 QAT（§3.4）

V4 在两个地方做 FP4 量化感知训练：

- **MoE expert 权重**：FP32 master weight → 量化到 FP4 → 反量化回 FP8 计算（FP4 用 E2M1，FP8 用 E4M3，多两个 exponent bit 做缓冲）。**关键观察**：FP4 sub-block (1×32 tile) 与 FP8 quantization block (128×128 tile) 的 max/min scale ratio 不超过某阈值时，FP8 的 dynamic range 足以无损吸收 FP4 scale——所以 V4 直接复用现有 FP8 训练框架，**0 修改**。
- **CSA Lightning Indexer 的 QK 路径**：QK 全程 FP4 加载 / 计算；index score $I_{t,:}$ 进一步从 FP32 量化到 BF16；top-k selector 2× 加速、KV recall 99.7%。

### 6.3 Batch Invariance + Determinism（§3.3）

为了 train/inference bit-wise 对齐：

- **Attention**：放弃 split-KV（破坏 batch invariance），改为**双 kernel 解码策略**——kernel 1 把整条序列放进单 SM，kernel 2 用 distributed shared memory 跨 SM、保证累加顺序与 kernel 1 一致。
- **MatMul**：放弃 cuBLAS，全部走 DeepGEMM；mHC 那个 24 维输出的小 GEMM 必须 split-k，作者把 split 部分**单独输出再 deterministic reduce**。
- **MoE Backward**：在 receiving rank 上做 token order 预处理 + buffer isolation，保证 EP 反传顺序确定。

::: tip 为什么"batch invariance"对 RL 这么重要
RL rollout / training / evaluation 跑在同一组权重上，但 batch size、layout、SM 分配都不同——任何一处累加序违反位级一致都会让 advantage 估计偏。V4 报告把这点单独成节是因为 RL 阶段（OPD）牵扯十几个 teacher 模型的同步。
:::

## 7. 异构 KV Cache + On-Disk Storage（§3.6）

V4 的 KV 同时来自 4 个源：CSA 主 KV、CSA indexer KV、HCA KV、Sliding-Window KV——尺寸、压缩比、淘汰策略全都不同。Paged-Attention 的同构假设直接失效。V4 的方案：

- **State Cache**（SWA + 未达压缩阈值的 tail 状态）：固定大小池，按 sequence 动态分配。
- **KV Cache**（CSA / HCA）：经典 paged，但 block 内 token 数取 **lcm($m, m'$)**——这样不管哪一层用 CSA 还是 HCA，单 block 都能整数分裂出整齐的压缩 entry，sparse-attention kernel 能假设固定 token-per-block。

**On-disk KV cache** 是 V4 落地的一个"工程加分项"：共享 prefix 的 CSA/HCA 压缩 entry 全量落盘，命中后直接读；SWA 因为体量是压缩版的 8×，提供三种策略让你按场景挑：

| 策略 | 存储 | 计算冗余 | 适用场景 |
|------|------|--------|---------|
| Full SWA Caching | 全存 | 0 | 读密集、命中率高 |
| Periodic Checkpointing | 每 $p$ 个 token 存一次最近 $n_{\text{win}}$ | 取决于 $p$ | 平衡型 |
| Zero SWA Caching | 不存 | 重算最近 $n_{\text{win}} \cdot L$ token | 写密集 / 容量极紧 |

## 8. Post-Training：Specialist + On-Policy Distillation

V4 完全替换了 V3.2 的"混合 RL"阶段——改成：**先训领域专家，再用 OPD 合一**（§5.1）。

### 8.1 Specialist Training（§5.1.1）

每个领域（数学 / 代码 / 智能体 / 指令跟随）独立训一个专家：

```
Base ─→ 领域 SFT ─→ 领域 GRPO RL（带 GRM 评估） ─→ 专家模型 π_E_i
```

两个亮点：

- **三档思考模式**：Non-think / Think High / Think Max，用 `<think></think>` 划分。Think Max 在 system prompt 前段注入一段固定模板（"Reasoning Effort: Absolute maximum..."，§Table 3），让模型自己进入"穷尽搜索"模式。
- **Generative Reward Model (GRM)**：抛弃传统 scalar RM——actor **本身就是 GRM**。"评判能力"和"生成能力"在同一组参数里联合优化，靠 RL 直接对 GRM 输出做 RL；少量人类标注就能 work，理由是 actor 用自己的推理能力做评判时泛化更好。

### 8.2 On-Policy Distillation（§5.1.2）

把 N 个 specialist 合一进单一学生 $\pi_\theta$：

$$
\mathcal{L}_{\text{OPD}}(\theta) = \sum_{i=1}^{N} w_i \cdot D_{\text{KL}}\!\big(\pi_\theta \,\|\, \pi_{E_i}\big)
$$

注意是 **reverse KL**（学生在前），且**轨迹 $y$ 来自学生自己的 rollout**（on-policy）——这是 OPD 区别于普通蒸馏的核心，在 [Thinking Machines Lab 的 On-Policy Distillation 博客](https://thinkingmachines.ai/blog/on-policy-distillation/) 已有系统讨论；项目里也已经在 [蒸馏](/training/distillation) 一节集成了它的入门版。

V4 的工程难点是 **full-vocabulary OPD**——KL 必须在完整 |V|>100k 的词表上算才稳定，但十几个 1.6T 参数的 teacher 同时把 full-logits 物化进 GPU 是不可能的。V4 的解法（§5.2.2）：

1. **缓存 teacher 最后一层 hidden state，而非 logits**——hidden 维度 $d$ 远小于 $|V|$。
2. **训练时再用 prediction head 把 hidden 还原成 logits**，重算开销忽略不计。
3. **按 teacher 索引重排 mini-batch**——保证每个 mini-batch 至多只有 1 份 teacher prediction head 在 GPU 上。
4. **TileLang 写专门的 KL kernel**——student/teacher logits 都不物化，直接在 kernel 里算完 KL 走人。

### 8.3 Tool-Call Schema：`<|DSML|>` XML

V4 把 tool-call 格式从 V3 的 JSON 改成了 XML 风格（§Table 4）：

```
<|DSML|tool_calls>
  <|DSML|invoke name="$TOOL_NAME">
    <|DSML|parameter name="$PARAM" string="true|false">$VALUE</|DSML|parameter>
  </|DSML|invoke>
</|DSML|tool_calls>
```

`string="true"` 时参数原样传，否则按 JSON 解析。报告说 XML 比 JSON 转义错误更少、tool-call 错误率明显下降。

### 8.4 Quick Instruction：复用 KV 的辅助任务

聊天场景有一堆"是否触发搜索 / 意图识别 / 生成会话标题"等小任务。传统做法是另起小模型，但每次重新 prefill 太贵。V4 把这些任务编码成 7 个特殊 token（§Table 5），直接拼到输入末尾、复用主模型 KV cache：

| Token | 作用 |
|-------|------|
| `<\|action\|>` | 判断是否需要调用搜索 |
| `<\|title\|>` | 生成会话标题 |
| `<\|query\|>` | 生成搜索 query |
| `<\|authority\|>` | 判断对来源权威性的需求 |
| `<\|domain\|>` | 识别提问的领域 |
| `<\|extracted_url\|>` `<\|read_url\|>` | URL 抽取与是否抓取 |

代价：训练阶段把这些 token 连同主任务一起 SFT。收益：用户感知 TTFT 大幅下降 + 不再维护小模型。

## 9. 评估硬数据（节选）

V4-Pro-Base vs V3.2-Base（§4.3, Table 1）：MMLU 87.8 → 90.1；GSM8K 91.1 → 92.6；MATH 60.5 → 64.5；LongBench-V2 40.2 → 51.5。

V4-Pro-Max vs 闭源旗舰（§5.3, Table 6）：

| 任务 | DS-V4-Pro-Max | Opus-4.6 Max | GPT-5.4 xHigh | Gemini-3.1-Pro High |
|------|--------------|--------------|---------------|---------------------|
| MMLU-Pro | 87.5 | 89.1 | 87.5 | **91.0** |
| GPQA-Diamond | 90.1 | 91.3 | **93.0** | 94.3 |
| LiveCodeBench | **93.5** | 88.8 | – | 91.7 |
| Codeforces (rating) | **3206** | – | 3168 | 3052 |
| Apex Pass@1 | 38.3 | 34.5 | 54.1 | **60.9** |
| Apex Shortlist | **90.2** | 85.9 | 78.1 | 89.1 |
| MRCR 1M | **83.5** | – | – | 76.3 |

报告自评："V4 在编码竞赛 / 长上下文检索类基准上首次让开源对齐到闭源旗舰；在 PhD-级 reasoning（HLE / GPQA）上离 Gemini-3.1-Pro 还有 3-6 个月的差距"（§5.3.2）。

## 10. 苏格拉底时刻

::: details Q1：CSA 已经做了"压缩 + 稀疏"，为什么还要再叠一个 HCA？
压缩比 $1/m=1/4$ 在 1M token 上仍然意味着 250K 条压缩 entry——top-k 选个 512 出来还是 dense attention 的 1/500，对长程"全局摘要"覆盖不足。HCA 用 $1/m'=1/128$ 把 1M 压成 ~8K 条 entry 后做 dense MQA，等于在每隔几层给模型一个"鸟瞰图"。读 §2.3 figure 2 时注意 CSA 与 HCA **是层间交错的**，不是同层并联。
:::

::: details Q2：mHC 把 $B_l$ 限制成 doubly stochastic 矩阵，但 $A_l$ 与 $C_l$ 只用 Sigmoid 卡到非负有界——为什么 $B_l$ 需要更强的几何约束？
$A_l$ 与 $C_l$ 是 1×n / n×1 的"边界"映射，不在反复迭代里出现；真正反复进入"$B_l \cdot B_{l-1} \cdot \ldots$"乘法链的是 $B_l$。Birkhoff 多面体在乘法下封闭 + 谱半径 ≤ 1，所以无论叠多少层，乘积仍然非膨胀；如果用 Sigmoid 只压到 [0,1]，单步看起来稳，但乘起来会发散。
:::

::: details Q3：V4 用 Muon 而不再需要 QK-Clip，但 Kimi K2 用 Muon 必须配 QK-Clip——差别在哪？
K2 把 Q/K 直接拼进 MLA 后做 attention，logit 由内积爆炸；K2 又不能用 QK-Norm（与 MLA 推理不兼容）。V4 在 attention 入口前对 query / KV entry 做了 **RMSNorm**（§2.3.3 第一条），把 logit 在算分数之前就摁住，自然不需要 QK-Clip。这也说明：稳定大模型训练有两条路——优化器侧 clip，或架构侧归一化。两选一即可，不必都做。
:::

::: details Q4：On-Policy Distillation 用 reverse KL 而不是 forward KL，会不会"模式坍缩"？
Reverse KL 的确有 mode-seeking 倾向，但 V4 用了 N 个 teacher（覆盖不同领域）做加权和，每个 teacher 自己已经是某领域的"窄专家"——所以 reverse KL 在每个领域上 mode-seek 是想要的行为（学生应该精确复刻某一领域的最优分布）。如果只用一个 teacher 做 reverse KL，模式坍缩才会真发生。
:::

## 11. 面试考点

1. **CSA 的压缩为什么用两套权重 $C^a/C^b$ 且索引重叠？** —— 让单条压缩 entry 实际吸收 $2m$ 条原始 KV 的信息，但序列长度仍然 $1/m$，等于"半窗滑动 + 拼接"——后续 indexer 选 top-k 时不会因为压缩边界把信息切碎。
2. **Lightning Indexer 为什么用 ReLU 而不用 softmax 做 score 加权？** —— ReLU 输出非负，等价于"投票求和"——多个 head 都觉得相关的位置被加强；softmax 会在 head 间分配概率质量，反而稀释强信号。
3. **mHC 的 Sinkhorn-Knopp 为什么固定 $t_{\max}=20$ 而不是迭代到收敛？** —— $t_{\max}=20$ 在 V4 配置下经验上离严格收敛 < $10^{-4}$ 已经足够，再多就是浪费——而且 fp16/bf16 的舍入误差让"严格收敛"本身没意义。
4. **Muon Hybrid Newton-Schulz 两段系数是怎么调出来的？** —— 第一段 $(3.4445, -4.7750, 2.0315)$ 优化"把奇异值快速从 $[0,2]$ 区间拉到 1 附近"的收敛速度；第二段 $(2, -1.5, 0.5)$ 是经典 NS 系数，在 1 附近是稳定吸引子。组合等价于"先用激进系数收敛、再用温柔系数稳定"。
5. **Anticipatory Routing 为什么不一直开着？** —— 它要多预读一份数据 + 跑两次 routing forward，wall-clock 大约 +20%。spike 是稀有事件，常态打开就是恒定 20% 浪费——所以做成"自动检测、按需启用、几步后回归"。
6. **Full-vocabulary OPD 为什么不能直接缓存 teacher logits？** —— logits 维度是 $|V|$（V4 词表 128K），而 hidden 维度只有 $d$（≈ 4K-7K）；缓存 hidden 比缓存 logits 省一个量级，且重算 prediction head 几乎免费。

## 12. 推荐资源

**论文 / 报告**
- DeepSeek-V4 Technical Report (DeepSeek-AI, 2026): <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf>（本文主要参考来源；本仓库已收录于 `refrences-projects/papers/13_deepseek-v4_pro.pdf`）
- DeepSeek-V3 Technical Report (arXiv:2412.19437)：先读 V3 才能看清 V4 改了什么
- Hyper-Connections (Zhu et al., 2025) + mHC (Xie et al., 2026)：mHC 的母论文
- Muon Optimizer (Jordan et al., 2024) + (Liu et al., 2025)：Muon 与 Newton-Schulz 系数推导

**官方开源**
- DeepSeek-V4-Pro 推理仓库（MIT）：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference>
- DeepGEMM PR #304（MegaMoE 融合内核）：<https://github.com/deepseek-ai/DeepGEMM/pull/304>
- 模型权重集合：<https://huggingface.co/collections/deepseek-ai/deepseek-v4>

**站内交叉**
- [DeepSeek-V3 架构](/architecture/deepseek)：MoE / MLA / MTP 的基础
- [Attention 深度实现](/architecture/attention)：Flash Attention v1/v2 + Online Softmax
- [Kimi K2 内部机制](/deep-dives/kimi-k2-internals)：Muon + QK-Clip 的另一种解法
- [蒸馏](/training/distillation)：On-Policy Distillation 入门
