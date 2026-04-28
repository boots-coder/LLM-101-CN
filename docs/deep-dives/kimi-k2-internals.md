---
title: Kimi K2 内部机制
description: MuonClip 优化器、QK-Clip、稀疏 Scaling Law、Agentic 数据合成、Checkpoint Engine——1T 参数 15.5T token 零 loss spike 的关键
topics: [kimi-k2, muonclip, qk-clip, muon-optimizer, sparsity-scaling, agentic-data, self-critique-rubric, checkpoint-engine]
prereqs: [training/pretraining, training/alignment, training/agent-rl]
---

# Kimi K2 内部机制

> **一句话总结：** Kimi K2 用 MuonClip 把 token-efficient 的 Muon 优化器扩到了 1T 参数 + 15.5T token——全程零 loss spike；用稀疏 Scaling Law 把 expert 数推到 384、attention head 砍到 64；用 3000+ MCP 工具 + 20000+ 合成工具构造 agentic 数据；并以 Self-Critique Rubric Reward + K1.5 风格 RL + 30 秒全集群 checkpoint 广播完成 RL 阶段——这份 deep-dive 把 K2 技术报告 (arXiv:2507.20534) 拆成可复用的工程拼图。

::: info 阅读提示
本文专注 K2 区别于 DeepSeek-V3 / Llama / 普通 RLHF 的"独特机制"。如果你想看 MoE / MLA / SFT / RL 的通用基础，请先读 [DeepSeek-V3 架构](/architecture/deepseek)、[偏好对齐](/training/alignment)、[Agent 强化学习](/training/agent-rl)。
:::

## K2 一图概览

```
┌────────────────────────────────────────────────────────────────────┐
│                           Kimi K2 (2025-07)                        │
│                                                                    │
│  Pre-train ─── 1.04T 参数 / 32.6B 激活 / 384 experts (8 active)    │
│             │  64 attention heads / MLA / 61 layers                │
│             │  15.5T tokens / 4k→32k→128k (YaRN)                   │
│             │  MuonClip (Muon + QK-Clip) → 零 loss spike           │
│             │                                                      │
│  Post-train ── SFT (Muon 微调) + Agentic Data Synthesis            │
│             │  3000+ MCP tools + 20000+ synthetic tools            │
│             │                                                      │
│  RL ────────── K1.5-style + Self-Critique Rubric Reward            │
│             │  Budget Control + PTX Loss + Temperature Decay       │
│             │  Checkpoint Engine：1T 参数全集群广播 < 30s          │
└────────────────────────────────────────────────────────────────────┘
```

| 维度 | K2 数值 | 出处 |
|------|--------|------|
| 总参数 / 激活 | 1.04T / 32.6B | K2 报告 p.6, Table 2 |
| 专家数 / 激活专家 | 384 / 8 | p.6 |
| Sparsity | 48 (= 384 / 8) | p.7 |
| Attention heads | 64（V3 是 128） | p.6, p.7 |
| 预训练 token | 15.5T | p.2 |
| 预训练 loss spike | 0 | p.4, Figure 3 |
| Checkpoint 广播延迟 | < 30s（1T 参数全集群） | p.14 |

::: tip 这一节读完你应该带走的"图"
K2 不是 V3 加大版——它砍了 attention head、加倍了 expert、换了优化器、并且把"agentic"渗到了 SFT/RL 的每个环节。下面 9 节按从 optimizer 到 RL 基础设施的顺序，逐层展开。
:::

## MuonClip 优化器：QK-Clip 是怎么把 logits 摁住的

### 为什么不能直接用 Muon

Moonshot 自家的 Moonlight 工作 (arXiv:2502.16982) 已经证明：在相同 compute 预算下，Muon 比 AdamW 显著省 token。但 K2 把规模继续推大时，遇到了一个新问题——**attention logits 爆炸 (exploding attention logits)**：在 Muon 训练里比 AdamW 更频繁，是其特有的不稳定性 (K2 报告 p.3)。

已有缓解手段在 K2 这种 MLA + 大规模场景下都不顶用：

- **Logit soft-cap**（Gemma 系）只 clip 最终 attention logits，但 query 与 key 的内积仍然在 cap 之前疯狂增长，等价于"截了梢、没断根"。(p.3)
- **QK-Norm** 需要把 Key 矩阵全实例化，这与 MLA 推理时**不完全物化 K** 的设计不兼容——MLA 把 K 折叠进了 latent，QK-Norm 没法插。(p.3)

K2 的解法是：**post-update weight clip**——每个 optimizer step 之后，按 per-head max-logit 反着把 W_q, W_k 缩回去。

### 核心数学

对于注意力头 h：

$$
\mathbf{Q}^h = \mathbf{X}\mathbf{W}_q^h,\quad \mathbf{K}^h = \mathbf{X}\mathbf{W}_k^h,\quad \mathbf{V}^h = \mathbf{X}\mathbf{W}_v^h
$$

$$
\mathbf{O}^h = \mathrm{softmax}\!\left(\frac{1}{\sqrt{d}}\mathbf{Q}^h\mathbf{K}^{h\top}\right)\mathbf{V}^h
$$

定义**per-head max-logit**（在一个训练 batch B 里、跨所有 token 对的最大点积）：

$$
S_{\max}^h = \frac{1}{\sqrt{d}}\,\max_{\mathbf{X}\in B}\,\max_{i,j}\,\mathbf{Q}_i^h\mathbf{K}_j^{h\top}
$$

QK-Clip 的核心 idea：当 $S_{\max}^h$ 超过阈值 $\tau$ 时，rescale W_q^h, W_k^h，强行把 logit 拉回 $\tau$ 以内。**这一步发生在 forward/backward 之外**——它不改本步的梯度，只在 optimizer step 之后修正权重。(p.3)

朴素实现（所有 head 一起 clip）：

$$
\mathbf{W}_q^h \leftarrow \gamma^\alpha\,\mathbf{W}_q^h,\quad \mathbf{W}_k^h \leftarrow \gamma^{1-\alpha}\,\mathbf{W}_k^h
$$

其中 $\gamma = \min(1, \tau/S_{\max})$，$S_{\max} = \max_h S_{\max}^h$，$\alpha$ 通常取 0.5（query/key 各承担一半缩放）。

但实测中**只有少数 head 在爆**——K2 改用 **per-head 缩放因子**：

$$
\boxed{\gamma_h = \min\!\left(1,\; \tau / S_{\max}^h\right)}
$$

只对超阈值的那个 head 操作，最大限度减少对训练动态的干预。(K2 报告 p.3)

### MLA 的特殊处理

MLA 的 q/k 含**共享 rotary 分量**，对它做缩放会跨 head 互相影响。K2 的策略 (p.3)：

| 分量 | 缩放因子 | 说明 |
|------|---------|------|
| $\mathbf{q}^C$（head-specific component） | $\sqrt{\gamma_h}$ | 非共享，每个 head 独立 |
| $\mathbf{k}^C$（head-specific component） | $\sqrt{\gamma_h}$ | 非共享，每个 head 独立 |
| $\mathbf{q}^R$（head-specific rotary） | $\gamma_h$ | head 独有 rotary，全量缩 |
| $\mathbf{k}^R$（**shared rotary**） | **不动** | 共享，碰一下会跨 head 串味 |

直觉记忆：**只 clip 非共享分量**——共享 rotary 的 K 部分被故意放过，因为它跨 head 共用，缩它会污染没爆的 head。

### Algorithm 1（完整伪代码）

下面是从 K2 报告 p.4 摘出的完整 Algorithm 1：

```
Algorithm 1  MuonClip Optimizer
────────────────────────────────────────────────────────────────────
 1: for each training step t do
 2:     // 1. Muon optimizer step
 3:     for each weight W ∈ ℝ^{n×m} do
 4:         M_t = μ · M_{t-1} + G_t          ▷ M_0 = 0, G_t is grad of W_t, μ is momentum
 5:         O_t = Newton-Schulz(M_t) · √(max(n,m)) · 0.2   ▷ Match Adam RMS
 6:         W_t = W_{t-1} - η · (O_t + λ · W_{t-1})        ▷ learning rate η, weight decay λ
 7:     end for
 8:     // 2. QK-Clip
 9:     for each attention head h in every attention layer of the model do
10:         Obtain S_max^h already computed during forward
11:         if S_max^h > τ then
12:             γ ← τ / S_max^h
13:             W_qc^h ← W_qc^h · √γ
14:             W_kc^h ← W_kc^h · √γ
15:             W_qr^h ← W_qr^h · γ
16:         end if
17:     end for
18: end for
```

(K2 报告 p.4, Algorithm 1)

读这段伪代码的几个关键点：

1. **Newton-Schulz 投影 + RMS match**：第 5 行的 $\sqrt{\max(n,m)}\cdot 0.2$ 是 Moonlight 提出的 RMS 匹配因子——让 Muon 的 update RMS 与 AdamW 等价，从而沿用 AdamW 的 lr/wd 经验值。(详见 [Moonlight 论文](https://arxiv.org/abs/2502.16982))
2. **$S_{\max}^h$ 复用 forward 计算**：第 10 行明确说"already computed during forward"——只是顺手记录每个 head 的 max logit，不引入额外 forward。
3. **没有 W_kr 缩放**：注意伪代码里**只更新 W_qc, W_kc, W_qr**，**W_kr (shared rotary key)** 不动——和上面 MLA 表格对得上。

### 实验曲线：从"必爆"到"平稳"

K2 报告 Figure 2 (p.4) 给了对照实验：

- **Vanilla Muon**（左图）：在 9B 激活 / 53B 总参数的中等规模 MoE 上，max logit 在训练过程中**指数级冲到 1000+**，意味着 softmax 后变成几乎 one-hot 的尖峰，loss 经常 spike，有时直接发散。
- **MuonClip with τ=100**（右图）：max logit 上来就被 cap 在 100 附近；训练约 **30%** 后开始自然回落，最终稳定在 cap 之下——说明 QK-Clip 不需要持续干预，只在前期"扶一把"。

更重要的是 K2 完整的 15.5T token 预训练曲线 (Figure 3, p.5)：**整条 loss 曲线没有 smoothing，也没有 sub-sampling，全程零 spike**。(K2 报告 p.5)

::: warning QK-Clip 不是降学习率
有人会问：为什么不直接降低 learning rate？降 lr 会**全局拖慢**学习；QK-Clip 只在**问题 head + 问题 step** 介入，其他时间 Muon 全速跑。这是 K2 在 token efficiency 上能继续吃 Muon 红利的关键。
:::

（K2 报告 p.3-4）

## 稀疏 Scaling Law：为什么 384 expert + 64 head

### 固定激活量，加 expert 持续降 loss

K2 团队针对 MoE + Muon 重新跑了一组 scaling law 实验：**固定 activated experts = 8、shared experts = 1，只变化 total experts**，得到 sparsity ∈ {8, 16, 32, 48, 64} 五组曲线。(p.7, Figure 5)

结论极简：**在 iso-FLOPs 下，提高 sparsity 持续降低 validation loss**。具体而言，**为了达到同一 validation loss = 1.5**，相比 sparsity = 8 / 16 / 32，sparsity = 48 分别少用 **1.69× / 1.39× / 1.15× FLOPs**。(p.7)

K2 选择 **sparsity = 48**（384 total / 8 active）作为性能与基础设施复杂度的折中——再往上推 sparsity 边际收益小但 EP 通讯/路由开销陡增。

### Attention head 砍一半的代价收益分析

更反直觉的决定是：**heads 从 V3 的 128 砍到 K2 的 64**。

按 DeepSeek-V3 的设计直觉，head 数 ≈ 2× layers 利于 memory bandwidth 利用。但 K2 在 Figure 6 (p.7) 上做了对照：

- 把 head 数从 = layers 增加到 = 2× layers，validation loss 只降 **0.5%~1.2%**
- 在 **128k 上下文**下，128 heads 相比 64 heads 推理 FLOPs **+83%**（attention 是 head 数线性的，长上下文更明显）

也就是说：**为 1% 的 loss 优势承担 83% 的推理 FLOPs**——对一个明确要做 long-context agentic 工作的模型来说，这笔账不划算。K2 选 64 heads 把推理成本压下去。(p.7)

::: tip 64 heads 是 agentic-first 的设计选择
K2 不是"普通 chat 模型"——它要服务于 tool-use / 代码 / 长任务这类 128k+ 上下文场景。**架构选择反映目标场景**：head 数砍半是为了让长上下文推理成本可承受，不是简单的"省钱"。
:::

（K2 报告 p.7, Figure 5-6）

## Knowledge & Math Data Rephrasing：1× rephrase + 10 epoch 打败 raw 10 epoch

高质量预训练 token 已是稀缺资源，简单地多 epoch 重复会过拟合。K2 在 K1.5 之上引入了**合成数据改写流水线**：

### Knowledge Rephrasing 三件套 (p.5)

1. **Style- and perspective-diverse prompting**（受 WRAP 启发）：用一组精心设计的 prompt 引导 LLM 用不同风格和视角改写同一段原文。
2. **Chunk-wise autoregressive generation**（Figure 4, p.6）：4096 token 的长文档**按 256 token 切块、依次自回归改写、再拼回完整段落**——避免 LLM 重写长文时的"隐式长度上限"，并用前文做 context 保持全局一致。
3. **Fidelity verification**：把每段改写与原文做语义对齐校验，作为训练前的初筛。

### Table 1：1× rephrase 就把 SimpleQA 从 23.76 提到 27.39

K2 用早期 checkpoint 做了对照实验 (p.5, Table 1)：

| # Rephrasings | # Epochs | SimpleQA Accuracy |
|---|---|---|
| 0 (raw wiki-text) | 10 | 23.76 |
| 1 | 10 | **27.39** |
| 10 | 1 | **28.94** |

**改写一次 + 训 10 个 epoch 比原文训 10 个 epoch 高 3.6 个点**——而总 token 看到次数是一样的。这说明 reword 提供的"语义同义但表面不同"的样本，有效降低了过拟合压力。

K2 把这套同样应用到 large-scale knowledge corpora，**每个 corpus 最多 rephrase 2 次**——边际收益和合成数据成本的取舍。

### Math Data Rephrasing：改写成"learning-note"风格

数学语料另开了一条路 (p.6)：把高质量数学文档**重写成"学习笔记"风格**（参考 SwallowMath），并把其它语种的高质量数学材料**翻译成英语**扩大语料多样性。

::: warning 合成数据的 caveat
报告 p.6 也明确写了：合成数据 scaling 仍是 active research，关键挑战包括跨域泛化、保事实准确、避免幻觉/毒性、可扩展性。**K2 的 1× / 2× rephrase 是个保守上限**——不是越多越好。
:::

（K2 报告 p.4-6, Figure 4 + Table 1）

## K2 vs DeepSeek-V3 架构对比

K2 报告 p.6 的 Table 2 直接给了关键差异：

| 维度 | DeepSeek-V3 | Kimi K2 | Δ |
|------|------------|---------|---|
| #Layers | 61 | 61 | = |
| Total Parameters | 671B | **1.04T** | ↑ 54% |
| Activated Parameters | 37B | 32.6B | ↓ 13% |
| Experts (total) | 256 | **384** | ↑ 50% |
| Experts Active per Token | 8 | 8 | = |
| Shared Experts | 1 | 1 | = |
| **Attention Heads** | 128 | **64** | **↓ 50%** |
| Number of Dense Layers | 3 | 1 | ↓ 67% |
| Expert Grouping | Yes | **No** | - |

::: tip 一句话读懂这张表
K2 = "**更稀疏 + 更窄 attention + 不分组的 expert 池**" 的 V3。Sparsity 从 32 提到 48、attention head 砍半、放弃 expert grouping——三个改变都指向同一个目标：**iso-FLOPs 下 token 效率最大化 + agentic 长上下文推理可负担**。
:::

放弃 expert grouping 的代价是更复杂的 EP 通讯调度，但换来更细粒度的路由——每个 token 可以从 384 个候选里选 8 个，组合空间远大于 V3 的"先选组、再选组内 top-k"。

（K2 报告 p.6, Table 2）

## Agentic Data Synthesis Pipeline：3000+ MCP + 20000+ synthetic tools

K2 后训练阶段最大的工程量在 agentic 数据合成。Figure 8 (p.10) 展示了三阶段流水线：

```
        ┌───────────────┐
        │   Domains     │ (financial trading / software / robot control / ...)
        └───────┬───────┘
                ↓
        ┌───────────────────────────────┐
        │   Tool Repository             │
        │   ┌────────────┐ ┌──────────┐ │
        │   │ MCP tools  │ │synthetic │ │
        │   │  (3000+)   │ │  (20000+)│ │
        │   └────────────┘ └──────────┘ │
        └───────┬───────────────────────┘
                ↓
        ┌──────────────┐    ┌──────────────┐
        │   Agents     │───▶│ Tasks (rubrics)│
        └──────┬───────┘    └──────┬───────┘
               ↓                   ↓
        ┌──────────────────────────────┐
        │ Multi-turn Trajectory Gen    │
        │  User Sim ↔ Agent ↔ Tools    │
        │           ↓                  │
        │     Judge Agent (rubric)     │
        │           ↓                  │
        │      Filtered Data           │
        └──────────────────────────────┘
```

### 三个阶段 (p.9-10)

1. **Tool spec generation**
   - 实拉 **3000+ 个 MCP (Model Context Protocol) tools** from GitHub。
   - 通过分层域生成（先选 key category 如金融/软件/机器人控制，再演化出多个具体应用域）合成 **20000+ synthetic tools**。
   - Figure 9 用 t-SNE 展示了真实 MCP tools 与合成 tools 的覆盖互补——合成工具按预定义域分布更均匀，MCP 按真实来源聚成自然簇。
2. **Agent and task generation**
   - 从 tool 仓库采样 toolset → 生成"持有这套 tool 的 agent"（不同 system prompt + 专长）。
   - 为每个 agent + toolset 生成任务，每个任务**配套显式 rubric**（成功标准 / 期望工具调用模式 / 评估检查点）。
3. **Trajectory generation**
   - **User Simulation**：LLM 扮演不同沟通风格 / 偏好的用户与 agent 多轮交互。
   - **Tool Simulator** + **真实 sandbox 混合**：通用任务用模拟器；**软件工程任务**则跑在 **Kubernetes 驱动的真实沙箱**（**支持 >10000 并发实例**），把 GitHub 上拉来的 PR/issue + 单元测试当作可执行环境 (p.12 Coding & SWE)。
   - **Judge Agent** 按 rubric 给轨迹打分，过滤出训练用的高质量轨迹。

::: tip 为什么要"模拟 + 真实"混合
纯模拟器便宜但 reward 不真实（容易 reward hacking）；纯真实环境信号准但成本爆炸。K2 的折中：**通用 tool 用模拟、SWE 用真实 sandbox**——把"需要执行结果验证"的部分放到真环境，把"对话 + 一般 API 调用"的部分放到模拟。
:::

（K2 报告 p.9-10, Figure 8 + p.12）

## Self-Critique Rubric Reward：让模型评判自己

可验证奖励 (RLVR) 在数学/代码上很好用，但**开放式任务**（创意写作、复杂推理、细微的 helpfulness）没有 ground truth。K2 的解法是 **Self-Critique Rubric Reward** (p.12)：

### 核心循环

> K2 actor 对 prompt 生成多个 response → **K2 critic 用 pairwise rubric ranking** 给所有 response 排序 → 排序作为 RL 奖励。

Critic 在 SFT 阶段就用混合的开源 + 内部偏好数据**预热**好评判能力。

### 三类 rubric (p.12)

| Rubric 类型 | 来源 | 作用 |
|---|---|---|
| **Core rubrics** (Appendix F.1) | K2 团队设定 | 体现助手核心价值观（保持人设连贯） |
| **Prescriptive rubrics** (Appendix F.2) | 反 reward-hacking 设计 | 显式禁掉常见的"骗分"模式 |
| **Human-annotated rubrics** | 数据团队为特定指令场景定制 | 局部任务的强约束 |

某些 rubric 可标记为 mandatory（必须满足），其它的允许 critic 用自己的 prior 加权——这种"软优先级"使得 critic 在保持身份一致的同时能根据 prompt 灵活调整。

### Closed-Loop Critic Refinement（关键创新）

只有 critic 不进步是不够的。K2 的核心 trick：**用 RLVR 反向校准 critic** (p.12)。

具体做法：在 RL 训练中，**用 verifiable-reward 的 on-policy rollouts**（比如数学题，有明确对错）持续更新 critic——把"客观信号"distill 进 critic 的判断里，让 critic 的主观打分**锚定在可验证数据上**。

::: tip 这步为什么重要
传统 reward model 训完就静态使用 → 容易被 policy 绕过 (reward hacking)。K2 让 critic 持续吃 verifiable data → critic **跟着 policy 一起进化**，缩小"policy 能做到 vs critic 能识别"的 gap。这是闭环对齐 (closed-loop alignment) 的工程化实现。
:::

（K2 报告 p.12）

## K1.5-style RL：均方损失 + Budget Control + PTX + 温度衰减

### RL Loss 是均方形式，不是 PPO ratio

K2 沿用 K1.5 的策略优化目标 (p.13)：对每个问题 x，从 $\pi_{\text{old}}$ 采样 K 个 response $\{y_1, \ldots, y_K\}$，优化：

$$
\boxed{L_{\mathrm{RL}}(\theta) = \mathbb{E}_{x\sim\mathcal{D}}\!\left[\frac{1}{K}\sum_{i=1}^{K}\left(\left(r(x, y_i) - \bar{r}(x)\right) - \tau\log\frac{\pi_\theta(y_i\mid x)}{\pi_{\text{old}}(y_i\mid x)}\right)^{\!2}\right]}
$$

其中 $\bar{r}(x) = \frac{1}{K}\sum_{i=1}^K r(x, y_i)$ 是组内平均奖励，$\tau > 0$ 是稳定学习的正则参数。

::: warning 这不是 PPO 也不是标准 GRPO
PPO/GRPO 都用 clipped importance ratio 形式（min(ratio·A, clip(ratio)·A)）。K2 这里是 **均方形式**——把"advantage 与 policy log-ratio 的差"当作残差去最小化，本质上是把 RL 写成了一个回归问题。

注意：SFT 同样用 Muon 优化器（p.9）—— K2 团队的经验是 "**Muon 预训的 checkpoint，用 Muon 微调效果最佳**"，这一原则贯穿 SFT/RL。
:::

### Budget Control：抑制"越答越长"

RL 经常让模型回答越来越长。K2 在每个 prompt 上**强制 per-sample maximum token budget**（按 task type 设定），超出 budget 的 response 被截断且打惩罚 (p.13)。

效果：在非推理任务上明显抑制了 verbosity；推理任务的 budget 给得更宽，保留 test-time compute 的空间。

### PTX Loss：防止灾难性遗忘

为了不在 RL 阶段忘掉 SFT 里学的高质量数据，K2 维护一份精选数据集，作为 **auxiliary PTX loss** 加入 RL objective（语出 InstructGPT 的 PTX 思路）(p.13)。

效果：扩展模型在 RL 训练任务之外的泛化能力，避免"为了奖励指标牺牲基本能力"。

### Temperature Decay：先探索后收敛

- **早期 RL**：高温采样，鼓励生成多样化、有创造性的 response，找到有效策略。
- **后期 RL**：温度按 schedule 衰减——降低随机性，提升输出一致性和可靠性。

直觉：**先 explore 后 exploit** 的工程化实现。(p.13)

（K2 报告 p.13）

## Checkpoint Engine：1T 参数全集群广播 < 30s

### 为什么 RL 阶段需要"快速广播"

K2 的 RL 用 **colocated 训推架构** (p.13)：训练 engine 与 inference engine 跑在同一批 worker 上，轮流占用 GPU。每次 iteration：

```
inference 生成 rollouts → 训练 engine 用 rollouts 更新 → 把新参数推给 inference
                                                         ↑
                                                  这一步是瓶颈
```

对 1T 参数模型，常见方案有两类，都不顺手：

1. **共享 NFS 网络文件系统**：bandwidth 需要数 PB/s 才能不拖慢，根本做不到。
2. **Transfer-what-you-need**：训练侧分片直接推给推理侧对应分片——理论最优、**实现极复杂**（每改一次推理 sharding 就要改一次广播逻辑）。

### K2 的工程取舍：分布式 Checkpoint Engine

K2 在每个训练节点 colocate 一个 **checkpoint engine worker**，三步完成参数同步 (Figure 10, p.14)：

```
┌──────────────────────── pod ────────────────────────┐
│  ┌──────────┐    ┌────────────┐    ┌────────────┐  │
│  │  train   │───▶│  ckpt      │───▶│ inference  │  │
│  │  engine  │    │  engine    │    │  engine    │  │
│  └──────────┘    └─────┬──────┘    └────────────┘  │
│                        │                            │
│                        ▼                            │
│                   broadcast                         │
│                   (across pod)                      │
└─────────────────────────────────────────────────────┘
```

1. 每个 ckpt worker 从同 pod 的训练 engine **拉一份参数 local copy**。
2. ckpt worker 之间**全集群广播完整参数**。
3. inference engine 从同 pod 的 ckpt worker **按需拉自己分片**。

为 1T 模型，这套流程**逐参数 pipelined**（详见报告 Appendix G）以最小化峰值显存。**完整一轮参数更新 < 30 秒**。(p.14)

### 这个取舍的妙处

> K2 团队明说：**这比 transfer-what-you-need 多传几次数据**——但**完全解耦训练 engine 与推理 engine 的 sharding scheme**，整个系统的可维护性大幅提升。30 秒对一个 RL iteration 来说是**可忽略的开销**，团队选择用 minor overhead 换 simpler design。(p.14)

代码已开源：[github.com/MoonshotAI/checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine)。

::: tip 工程哲学：选简单胜过选最优
"理论最优" (transfer-what-you-need) 在 1T 规模下变成维护噩梦——每次改 sharding 都要重写广播。K2 选了一个"传得多但模块解耦"的方案——这是大模型工程里反复出现的取舍：**当系统复杂度本身成为瓶颈时，简单胜于最优**。
:::

（K2 报告 p.14, Figure 10）

## 苏格拉底时刻

::: tip 停下来思考
1. **为什么 QK-Clip 是 post-update 缩放权重，而不是直接降低学习率或加 logit cap？** 提示：lr 是全局的，QK-Clip 只针对**单个出问题的 head**；logit soft-cap 治标不治本，QK 的内积可以在 cap 之前已经膨胀失控。post-update clip 让每个 head 在自己越界时才被"按一下"，对其他 head 训练动态零干扰。

2. **为什么 K2 选了"1× rephrase + 10 epoch"而不是"10× rephrase + 1 epoch"，即使后者 SimpleQA 更高？** 提示：看 Table 1 的差距 (27.39 vs 28.94) 与合成成本——10× rephrase 要调用 10× 推理 token；K2 的官方做法是"每 corpus 至多 rephrase 2 次"，这是质量边际收益 vs 合成 cost 的工程平衡，不是单点指标最优。

3. **K2 把 attention head 从 128 砍到 64，为什么这个决定要等到"明确为 agentic 设计"才合理？** 提示：iso-token 训练下 64 vs 128 只差 0.5%-1.2% loss，但 128k context 推理 FLOPs 差 83%。一个普通 chat 模型可能选 128（追极致 loss），但一个目标场景就是长上下文 + 多轮 tool-use 的模型必须算推理账——**架构选择反映目标场景**。

4. **Self-Critique Rubric Reward 的"Closed-Loop Critic Refinement"为什么用 RLVR 数据反向校准 critic？** 提示：如果 critic 静态不变，policy 会逐渐学会"用 critic 喜欢但实际无用的方式骗分"（reward hacking）。把 verifiable 奖励的 on-policy 信号 distill 进 critic，让 critic 与 policy 同步进化——critic 始终被"客观可验证"的信号锚定。

5. **为什么 Checkpoint Engine 选择"全集群广播完整参数"这种看似浪费的方案，而不是 transfer-what-you-need？** 提示：理论最优的方案要求训练侧分片与推理侧分片**严格协调**——每改一次推理 sharding 就要改广播逻辑，1T 参数 + 多种推理后端下维护成本爆炸。30 秒一次广播对 RL iteration 来说可忽略，**模块解耦的工程价值远大于这点 bandwidth**。

6. **K2 的 RL Loss 用的是均方形式 $(r - \bar{r} - \tau \log\pi_\theta/\pi_{\text{old}})^2$，而不是 PPO 的 clipped ratio。这种形式有什么数值稳定性优势？** 提示：均方形式把 RL 写成回归问题——梯度大小由残差决定，自然有界；不需要 PPO 的 clip 操作（它本质上是个非光滑近似）。代价是它假设 $\tau$ 选得合理，否则可能太保守。
:::

## 面试考点

::: details Q: MuonClip 与 QK-Norm / Logit Soft-Cap 的本质区别？
**A:** 三者都解决 attention logits 失控，但介入点完全不同：
- **QK-Norm**：在 forward 里**对 Q, K 做 LayerNorm**，必须把 K 全部物化——和 MLA（K 折叠进 latent，推理时不完全实例化）冲突，无法用。
- **Logit Soft-Cap**：在 softmax **之前**给 logits 套 tanh 一类有界函数——治表不治里，QK 内积可以在 cap 之前已经膨胀，梯度也会被这个非线性扭曲。
- **MuonClip (QK-Clip)**：**post-optimizer-step** 直接缩 W_q, W_k 的权重——只在某 head 的 max-logit 越过 τ 时才介入；不影响 forward/backward；MLA 友好（只缩 head-specific 分量，shared rotary key 不动）。
:::

::: details Q: K2 为什么放弃 DeepSeek-V3 的 expert grouping？代价是什么？
**A:** Expert grouping 的优势是减少路由通讯（先选组、再选组内 top-k）。但 K2 把 expert 数推到 384、sparsity = 48 时，**分组路由的表达能力受限**——每个 token 在更细粒度上选 8 个 expert 可以解锁更多组合空间。代价是 EP 通讯更复杂、负载均衡更难。K2 团队选择"路由质量 > 通讯简便性"，这与 K2 强调 token 效率的整体哲学一致。
:::

::: details Q: K2 报告里 SFT 也用 Muon 优化器，为什么不像很多 paper 那样 SFT 用 AdamW？
**A:** K2 团队（基于 Moonlight 的经验）观察到：**Muon 预训练的 checkpoint 用 Muon 继续微调效果最佳** (p.9)。直觉：预训练阶段 Muon 把权重塑造到了"对 Muon update 友好"的几何上；切到 AdamW 等价于换了一个不同曲率假设的优化器——可能引入轻微的不一致。这不是大问题，但既然 Muon 已经在 SFT 上 work，就保持一致。
:::

::: details Q: 为什么 K2 的 Self-Critique 用 pairwise ranking 而不是 pointwise scoring？
**A:** Pairwise ranking 比 pointwise scoring **更稳定、校准要求更低**——critic 不需要给出绝对分数（如 "7.5/10"），只需要判断 A 与 B 哪个更好。LLM-as-judge 的研究反复证明：模型在 pointwise 上的打分尺度漂移严重（同一回答今天给 7、明天给 8），但相对偏好稳定得多。K2 的 RL 只需要**相对偏好序**作为奖励信号，pairwise ranking 的鲁棒性正合适。
:::

::: details Q: Checkpoint Engine 是 30 秒完成 1T 参数广播——这个数字背后的硬件假设是什么？
**A:** K2 训练集群是 H800，每节点 8 GPU + NVLink/NVSwitch + **8×400 Gbps RoCE** 跨节点互联 (p.7)。1T 参数 ≈ 2TB (bf16)，全集群广播需要 cross-node bandwidth 充足；H800 的 8×400Gbps 提供约 400GB/s 节点级聚合带宽。30 秒对应 ~67GB/s 平均吞吐，在这种网络拓扑下完全可达；逐参数 pipelined 设计还能把峰值显存控制在合理范围。换到普通 IB 集群同样的设计可能更慢，但思路（broadcast + decoupled sharding）依然适用。
:::

## 推荐资源

| 资源 | 链接 | 说明 |
|------|------|------|
| Kimi K2 技术报告 | [arXiv:2507.20534](https://arxiv.org/abs/2507.20534) | 本文的全部公式与表格出处 |
| Moonlight (Muon at Scale) | [arXiv:2502.16982](https://arxiv.org/abs/2502.16982) | Muon + Newton-Schulz + RMS match 的前作 |
| Muon Optimizer | [Keller Jordan 博客](https://kellerjordan.github.io/posts/muon/) | Muon 原作，理解 K2 优化器的起点 |
| Kimi K1.5 报告 | [arXiv:2501.12599](https://arxiv.org/abs/2501.12599) | K2 RL 算法 / 数据流水线的前身 |
| DeepSeek-V3 报告 | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) | K2 架构的对照基准（Table 2 的对手） |
| WRAP (Web Rephrase) | [arXiv:2401.16380](https://arxiv.org/abs/2401.16380) | K2 知识改写流水线的灵感来源 |
| Checkpoint Engine | [GitHub: MoonshotAI/checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine) | K2 RL 广播组件开源实现 |
| YaRN (128k 扩展) | [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) | K2 把 4k→128k 的方法 |
