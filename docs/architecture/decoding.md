---
title: "解码策略"
description: "Greedy、Beam Search、Temperature、Top-k、Top-p 采样策略"
topics: [greedy-search, beam-search, temperature, top-k, top-p, nucleus-sampling]
prereqs: [architecture/transformer]
---
# 解码策略（Decoding Strategies）

> **一句话总结:** 解码策略决定了模型"如何从概率分布中选择下一个 Token"——同一个模型搭配不同的解码策略，可以生成从严谨精确到天马行空的截然不同的文本。

## 在大模型体系中的位置

语言模型的前向传播输出的是下一个 Token 在整个词表上的概率分布（logits → softmax → 概率）。解码策略作用于这个概率分布之上，决定最终选哪个 Token。它不影响模型参数，却深刻影响生成质量——选择合适的解码策略是 LLM 应用落地的关键环节。

```
输入 Token 序列 → [Transformer 前向传播] → logits (词表大小的向量)
    → [Temperature 缩放] → [Top-k / Top-p 过滤] → [采样或取 argmax] → 下一个 Token
```

## 核心概念

### Greedy Search（贪心搜索）

最简单的解码策略：每一步都选择概率最高的 Token。

$$x_t = \arg\max_{x} P(x | x_{<t})$$

**优点：** 速度快，确定性输出，适合有唯一正确答案的任务（如分类、提取）。

**缺点：** 容易陷入局部最优。因为每一步都只看当前最优，可能错过整体更优的序列。例如，当前概率最高的词可能把后续生成引入一条"死胡同"，导致整体序列质量下降。

```python
# Greedy Search 实现
def greedy_decode(model, input_ids, max_length):
    for _ in range(max_length):
        logits = model(input_ids)          # [batch, seq_len, vocab_size]
        next_token = logits[:, -1, :].argmax(dim=-1)  # 取概率最高的
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
        if next_token == eos_token_id:
            break
    return input_ids
```

### Beam Search（束搜索）

Beam Search 是 Greedy Search 的扩展：不只保留一条路径，而是同时维护 $k$ 条（beam width = $k$）最优候选路径，最终从中选择总概率最高的序列。

**工作流程：**

1. 从起始 Token 出发，生成 Top-k 个候选
2. 对每条候选路径，分别扩展下一步的 Top-k 个 Token
3. 在所有 $k \times k$ 个候选中，保留总概率最高的 $k$ 条
4. 重复直到所有路径都生成了结束符或达到最大长度

**核心公式：** 每条路径的得分是各步 log 概率之和：

$$\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i | y_{<i})$$

通常还需要做**长度归一化**（除以序列长度），否则 Beam Search 会倾向于生成更短的序列（因为每多一步，log 概率之和就更小）。

**优点：** 比 Greedy 能找到更优的序列，适合翻译、摘要等需要精确输出的任务。

**缺点：** 生成的文本缺乏多样性，容易重复；计算量随 beam width 线性增长。在开放式生成（如对话、创意写作）中效果不佳。

### Temperature（温度参数）

Temperature 通过缩放 logits 来控制概率分布的"尖锐程度"：

$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

其中 $T$ 是温度参数，$z_i$ 是 logits。

| Temperature | 效果 | 适用场景 |
|------------|------|---------|
| $T \to 0$ | 分布趋近 one-hot（等价于 Greedy） | 事实性问答、代码生成 |
| $T = 1$ | 保持模型原始分布 | 通用场景 |
| $T > 1$ | 分布更平坦，低概率 Token 获得更多机会 | 创意写作、头脑风暴 |

**直觉理解：** Temperature 低 = 模型更"自信"更保守；Temperature 高 = 模型更"随机"更有创意。

### Top-k Sampling（Top-k 采样）

在采样前，只保留概率最高的 $k$ 个 Token，将其余 Token 的概率置零，然后在这 $k$ 个候选中按概率采样。

```python
# Top-k Sampling
def top_k_sampling(logits, k):
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_logits, dim=-1)
    next_token_index = torch.multinomial(probs, num_samples=1)
    return top_k_indices[next_token_index]
```

**问题：** $k$ 是固定的，但不同位置的概率分布差异很大。有时模型非常确定（前 2 个 Token 占 99% 概率），$k=50$ 就引入了过多噪声；有时模型很不确定（概率分散在数百个 Token 上），$k=50$ 又过度截断。

### Top-p / Nucleus Sampling（核采样）

Top-p 采样解决了 Top-k 中 $k$ 固定的问题：不是固定候选数量，而是**动态选择累积概率达到阈值 $p$ 的最小 Token 集合**。

$$\text{Top-p}(P, p) = \min \{ V' \subseteq V : \sum_{x \in V'} P(x) \geq p \}$$

其中 Token 按概率从高到低排序后依次加入，直到累积概率 $\geq p$。

```python
# Top-p (Nucleus) Sampling
def top_p_sampling(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # 找到累积概率超过 p 的位置，移除之后的 Token
    mask = cumulative_probs - sorted_probs > p
    sorted_logits[mask] = -float('inf')
    probs = F.softmax(sorted_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return sorted_indices.gather(-1, next_token)
```

**优点：** 自适应候选集大小。当模型很确定时，可能只保留 2-3 个 Token；当模型不确定时，可能保留数百个 Token。

**实践中的组合策略：** 现代 LLM 推理通常同时使用 Temperature + Top-p（有时再加 Top-k）。例如 OpenAI API 的默认配置是 `temperature=1.0, top_p=1.0`，用户可根据任务调整。

### 各策略对比总结

| 策略 | 确定性 | 多样性 | 典型应用 |
|------|--------|--------|---------|
| Greedy | 完全确定 | 无 | 分类、提取、简单问答 |
| Beam Search | 近似确定 | 低 | 机器翻译、摘要 |
| Top-k Sampling | 随机 | 中 | 通用生成 |
| Top-p Sampling | 随机 | 中高 | 通用生成（更自适应） |
| Temperature < 1 | 偏确定 | 低 | 事实性任务、代码 |
| Temperature > 1 | 偏随机 | 高 | 创意写作、多样化输出 |

### 进阶话题：重复惩罚与停止条件

实际部署中还需要考虑：

- **重复惩罚（Repetition Penalty）：** 降低已生成 Token 的概率，避免模型陷入重复循环
- **频率惩罚 / 存在惩罚（Frequency / Presence Penalty）：** OpenAI API 提供的两种不同粒度的重复控制
- **停止条件：** 生成 EOS Token、达到最大长度、或匹配到指定的停止字符串

## 代码实战

本节从零实现四种解码策略，并用同一个 prompt 对比生成效果。所有代码基于 PyTorch，可直接在 HuggingFace 模型上运行。

### 统一的采样框架

```python
import torch
import torch.nn.functional as F

def sample_next_token(logits, strategy="greedy", temperature=1.0, top_k=50, top_p=0.9):
    """
    统一的 Token 采样函数，支持多种策略组合
    
    Args:
        logits: 模型输出的 logits，shape [vocab_size]
        strategy: "greedy" | "sample"（采样时可叠加 temperature/top_k/top_p）
        temperature: 温度参数，越低越确定
        top_k: Top-k 截断，0 表示不使用
        top_p: Top-p 核采样阈值，1.0 表示不使用
    
    Returns:
        选中的 token id（标量）
    """
    if strategy == "greedy":
        return logits.argmax(dim=-1)
    
    # Step 1: Temperature 缩放
    if temperature != 1.0:
        logits = logits / temperature
    
    # Step 2: Top-k 过滤
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_value = logits.topk(top_k).values[-1]
        logits = logits.where(logits >= kth_value, torch.tensor(float('-inf')))
    
    # Step 3: Top-p 过滤
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 移除累积概率超过 p 的 Token（保留第一个超过的）
        mask = cumulative_probs - sorted_probs > top_p
        sorted_logits[mask] = float('-inf')
        
        # 还原到原始顺序
        logits = sorted_logits.scatter(-1, sorted_indices.argsort(-1), sorted_logits)
    
    # Step 4: 采样
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

### Beam Search 从零实现

```python
def beam_search(model, input_ids, max_length, beam_width=4, length_penalty=0.6):
    """
    Beam Search 实现
    
    Args:
        model: 语言模型（输入 token ids，输出 logits）
        input_ids: 初始 token 序列，shape [1, seq_len]
        max_length: 最大生成长度
        beam_width: 束宽度
        length_penalty: 长度惩罚系数（>0 鼓励更长序列）
    
    Returns:
        最优序列的 token ids
    """
    device = input_ids.device
    
    # 每条 beam: (累积 log 概率, token 序列)
    beams = [(0.0, input_ids.squeeze(0).tolist())]
    completed = []
    
    for step in range(max_length):
        all_candidates = []
        
        for score, seq in beams:
            # 如果这条 beam 已结束，直接保留
            if seq[-1] == model.config.eos_token_id:
                completed.append((score, seq))
                continue
            
            # 前向传播获取下一步 logits
            ids = torch.tensor([seq], device=device)
            with torch.no_grad():
                logits = model(ids).logits[0, -1, :]  # [vocab_size]
            
            log_probs = F.log_softmax(logits, dim=-1)
            
            # 取 Top-k 个候选
            topk_log_probs, topk_ids = log_probs.topk(beam_width)
            
            for i in range(beam_width):
                new_score = score + topk_log_probs[i].item()
                new_seq = seq + [topk_ids[i].item()]
                all_candidates.append((new_score, new_seq))
        
        if not all_candidates:
            break
        
        # 按长度归一化得分排序，保留 Top beam_width 条
        def normalized_score(item):
            score, seq = item
            return score / (len(seq) ** length_penalty)
        
        all_candidates.sort(key=normalized_score, reverse=True)
        beams = all_candidates[:beam_width]
    
    # 从 completed + beams 中选最优
    all_results = completed + beams
    best = max(all_results, key=normalized_score)
    return best[1]
```

### 完整运行示例：对比四种策略

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型（用小模型便于本地实验）
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

prompt = "The future of artificial intelligence is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

def generate(input_ids, max_new_tokens=30, **kwargs):
    """通用生成函数"""
    ids = input_ids.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :]
        next_id = sample_next_token(logits, **kwargs)
        ids = torch.cat([ids, next_id.unsqueeze(0).unsqueeze(0)], dim=-1)
        if next_id == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids[0], skip_special_tokens=True)

# 对比不同策略
strategies = {
    "Greedy":               dict(strategy="greedy"),
    "Sample (T=0.7)":       dict(strategy="sample", temperature=0.7, top_p=1.0),
    "Top-k (k=50)":         dict(strategy="sample", top_k=50, top_p=1.0),
    "Top-p (p=0.9)":        dict(strategy="sample", top_k=0, top_p=0.9),
    "Top-p + Low T (T=0.3)":dict(strategy="sample", temperature=0.3, top_p=0.9),
}

for name, params in strategies.items():
    result = generate(input_ids, max_new_tokens=30, **params)
    print(f"[{name}]")
    print(f"  {result}\n")
```

::: tip 动手实验
1. 把 `temperature` 分别设为 0.1、0.5、1.0、2.0，观察输出的确定性和多样性变化
2. 把 `top_p` 从 0.1 逐步增加到 1.0，对比生成文本的质量
3. 对同一 prompt 多次运行采样策略，观察输出的方差——这就是为什么 ChatGPT 每次回答不同
4. 尝试用 Beam Search 生成，对比它与采样策略在"事实性问题"vs"创意写作"上的表现差异
:::

## 实战复现：可视化解码树

> 本节复现 Maxime Labonne 的 [Decoding Strategies in LLMs](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html) notebook（19 cell，~3 MB，含运行截图），用 **GPT-2 + networkx + matplotlib** 把每一种解码策略画成可观察的搜索树。这是上面"代码实战"的可视化伴侣——同样的算法，不同的目的：上面追求"对齐生产框架的接口形态"，这里追求"把每一步选择画给眼睛看"。

### 环境准备（Cell 1）

```bash
sudo apt-get install graphviz graphviz-dev   # macOS: brew install graphviz
pip install transformers pygraphviz networkx matplotlib
```

### 共享设置（Cell 3）

整个 notebook 用同一段 prompt `"I have a dream"`，每个策略续写 5 个 token，便于横向对比。

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()

text = "I have a dream"
input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
```

### Greedy Search 的递归实现（Cell 5）

把每一步 argmax 当作**一棵深度 5 的链表**，节点上记 token + log-prob，最后用 networkx 一把渲染。

```python
def get_log_prob(logits, token_id):
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    return torch.log(probabilities)[token_id].item()

def greedy_search(input_ids, node, length=5):
    if length == 0: return input_ids
    logits = model(input_ids).logits[0, -1, :]
    token_id = torch.argmax(logits).unsqueeze(0)
    token_score = get_log_prob(logits, token_id)
    new_input_ids = torch.cat([input_ids, token_id.unsqueeze(0)], dim=-1)

    current_node = list(graph.successors(node))[0]
    graph.nodes[current_node]['tokenscore'] = np.exp(token_score) * 100
    graph.nodes[current_node]['token'] = tokenizer.decode(token_id) + f"_{length}"
    return greedy_search(new_input_ids, current_node, length - 1)
```

::: tip 关键观察
节点存的是 `np.exp(log_prob) * 100`——这是把概率拉回 0-100 的可读尺度，作为 graphviz 节点的着色依据。颜色越深 = 概率越接近 1 = 模型越"确定"。
:::

### Beam Search 的累积评分（Cell 8-9）

beam search 的核心是**长度归一化的累积 log-prob**：

$$
\text{score}(y_1, \ldots, y_t) = \frac{1}{t} \sum_{i=1}^{t} \log P(y_i \mid y_{<i})
$$

```python
def beam_search(input_ids, node, bar, length, beams, sampling, temperature=0.1):
    if length == 0: return None
    logits = model(input_ids).logits[0, -1, :]

    if sampling == 'greedy':   top_token_ids = torch.topk(logits, beams).indices
    elif sampling == 'top_k':  top_token_ids = top_k_sampling(logits, temperature, 20, beams)
    elif sampling == 'nucleus':top_token_ids = nucleus_sampling(logits, temperature, 0.5, beams)

    for j, token_id in enumerate(top_token_ids):
        token_score = get_log_prob(logits, token_id)
        cumulative_score = graph.nodes[node]['cumscore'] + token_score
        new_input_ids = torch.cat([input_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=-1)

        current_node = list(graph.successors(node))[j]
        graph.nodes[current_node]['cumscore'] = cumulative_score
        graph.nodes[current_node]['sequencescore'] = cumulative_score / len(new_input_ids.squeeze())
        beam_search(new_input_ids, current_node, bar, length - 1, beams, sampling, 1)


def get_best_sequence(G):
    leaf_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
    max_score_node = max(leaf_nodes, key=lambda n: G.nodes[n]['sequencescore'])
    path = nx.shortest_path(G, source=0, target=max_score_node)
    return ''.join([G.nodes[n]['token'].split('_')[0] for n in path]), G.nodes[max_score_node]['sequencescore']
```

::: warning 这就是 Beam Search 的"全栈"
`beam_search` 是个统一接口——把 `sampling` 参数从 `'greedy'` 切到 `'top_k'`/`'nucleus'`，递归内部从"挑 top-K"换成"按概率采样 K 个"，立刻获得不同算法。这是 notebook 设计上最值得抄的地方。
:::

### Top-k：硬截断 + 温度缩放（Cell 12）

```python
def top_k_sampling(logits, temperature, top_k, beams, plot=True):
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    new_logits = torch.clone(logits)
    new_logits[indices_to_remove] = float('-inf')
    probabilities = torch.nn.functional.softmax(new_logits / temperature, dim=-1)
    return torch.multinomial(probabilities, beams)
```

`indices_to_remove` 这一行是教学重点：先取 top-k 的最小值作阈值，所有低于它的 logit 设为 `-inf`，再走 softmax/temperature——`-inf` 经 exp 变 0，自然排除。

### Nucleus：动态截断（Cell 16）

```python
def nucleus_sampling(logits, temperature, p, beams, plot=True):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probabilities = torch.nn.functional.softmax(sorted_logits / temperature, dim=-1)
    cumulative_probabilities = torch.cumsum(probabilities, dim=-1)

    mask = cumulative_probabilities < p
    if mask.sum() > beams:
        top_p_index_to_keep = torch.where(mask)[0][-1].detach().cpu().tolist()
    else:
        top_p_index_to_keep = beams

    indices_to_remove = sorted_indices[top_p_index_to_keep:]
    sorted_logits[indices_to_remove] = float('-inf')
    probabilities = torch.nn.functional.softmax(sorted_logits / temperature, dim=-1)
    return torch.multinomial(probabilities, beams)
```

::: tip top-k vs nucleus 的根本差异
top-k 是**固定 K 个候选**（不论概率分布平坦还是尖锐）；nucleus 是**直到累积概率超过 p 才停**——分布尖锐时只保留 1-2 个、平坦时保留几十个。这正是为什么 nucleus 在开放生成里比 top-k 更稳。
:::

### 跑通建议

1. **本机即可**：GPT-2 124M 在 CPU 上 5 步生成 ~10 秒；Colab 免费 T4 ~1 秒
2. **graphviz 是硬依赖**：少装一个 `graphviz-dev` 头文件 `pygraphviz` 编译就会失败，macOS 用 `brew install graphviz` 比 apt 快
3. **可视化输出**：每跑完一个策略会在工作目录生成 `top_k_<timestamp>.png` / `nucleus_<timestamp>.png`，建议跑完一段 `mv *.png decoding_runs/<策略名>/`，不然全堆在根目录
4. **改 `length=10, beams=4`**：默认参数太小看不出差异，把搜索深度和分支度拉一点，beam search 树的"幸存者偏差"会一目了然——这才是这份 notebook 真正的教学价值

## 苏格拉底时刻

请停下来思考以下问题，不急于查看答案：

1. Beam Search 的 beam width 越大结果越好吗？为什么在开放式对话中 Beam Search 表现不如采样方法？
2. Temperature 和 Top-p 都能控制生成的"随机性"——它们的作用机制有什么本质区别？能否只用其中一个？
3. 如果你需要让 LLM 生成 JSON 格式的结构化输出，应该如何选择解码策略？为什么？
4. 为什么 Greedy Decoding 不能保证找到全局最优序列？能否构造一个具体的反例？
5. "采样导致的随机性"和"模型能力的不确定性"是一回事吗？如何区分一个错误答案是因为解码策略不好还是因为模型本身能力不足？

## 常见问题 & 面试考点

- **Q: Top-k 和 Top-p 可以同时使用吗？** 可以，先做 Top-k 截断，再在 Top-k 结果中做 Top-p 过滤。很多推理框架支持同时设置。
- **Q: Temperature = 0 和 Greedy 完全等价吗？** 数学极限上等价（$T \to 0$ 时 softmax 退化为 argmax）。实现中 `temperature=0` 通常直接走 argmax 逻辑。
- **Q: 为什么 ChatGPT 同一个问题每次回答不同？** 因为使用了采样策略（Temperature > 0），每次从概率分布中随机采样的结果不同。
- **Q: Speculative Decoding 是什么？** 用一个小模型快速生成候选 Token，再用大模型并行验证，从而加速推理。这不改变生成分布，只加速推理过程。

## 推荐资源

### 论文

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) by Holtzman et al. — 提出 Top-p / Nucleus Sampling 的论文，揭示了纯 likelihood maximization 会导致重复退化的现象。
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) by Leviathan et al. — Speculative Decoding 原始论文，用小模型起草 + 大模型验证，无损加速生成。
- [Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319) by Welleck et al. — 从训练目标角度分析与缓解重复生成问题。

### 博客与可视化

- [How to generate text: using different decoding methods](https://huggingface.co/blog/how-to-generate) by HuggingFace — 各种解码策略的交互式教程，配合 `transformers` 的 `generate()` 接口示例。
- [Decoding Strategies in Large Language Models](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html) by Maxime Labonne — 用 GPT-2 续写 "I have a dream" 这一例子贯穿全文，从 logits → softmax → token 选择的最底层视角逐步可视化讲解 greedy search、beam search（带长度归一化的累积 log 概率）、top-k sampling（叠加 temperature `softmax(x_i/T)` 控制分布锐度）、nucleus / top-p sampling（按累积概率动态截断）四种策略；用决策树和概率分布柱状图清晰对比"确定性但易陷入局部最优"与"随机采样带来的多样性"，是理解采样族策略最直观的入门读物。
- [OpenAI API Reference — Sampling Parameters](https://platform.openai.com/docs/api-reference/chat/create) — 理解 temperature、top_p、frequency_penalty、presence_penalty 在生产环境中的实际效果。

### 代码参考

- [rasbt/LLMs-from-scratch · ch04/03_kv-cache](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/03_kv-cache) — Sebastian Raschka《Build a Large Language Model (From Scratch)》第 4 章配套代码。本页 §可视化解码树 偏重"如何选 token"，这份代码补足的是"自回归阶段的 KV cache 工程"——为什么大模型推理时第二个 token 之后不再重新算前缀的 K/V。
  - [gpt_with_kv_cache.py](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/03_kv-cache/gpt_with_kv_cache.py)（376 行） — 在 `MultiHeadAttention`(L14-L109) 里加两个 `register_buffer("cache_k"/"cache_v", None)` 和 `ptr_current_pos`，`forward` 中按 `use_cache` 分两路：首次推理把整段 prompt 的 K/V 写入 cache（L56-L62）、之后每步只算新 token 的 K/V 然后 `torch.cat` 追加（L60-L61）。`GPTModel.forward`(L212-L241) 维护 `current_pos` 让 position embedding 对上号，`reset_kv_cache`(L245-L249) 在新一轮生成前清空。`generate_text_simple_cached`(L280-L304) 是对照基准——首调传完整 prompt、之后每次只喂 `next_idx` 单 token，对比 `generate_text_simple`(L252-L275) 每步重算整段，实测 GPT-2 124M 在 CPU 上 200 token 生成可观察到几倍加速。
- [rasbt/LLMs-from-scratch · ch04/04_gqa](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/04_gqa) — 同书的 GQA（Grouped-Query Attention）单文件实现，与上面 KV cache 版本是配套对照。
  - [gpt_with_kv_gqa.py](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/04_gqa/gpt_with_kv_gqa.py)（358 行） — `GroupedQueryAttention`(L20-L123) 把 `W_key / W_value` 的输出维度从 `num_heads * head_dim` 缩成 `num_kv_groups * head_dim`（L32-L33），forward 里靠 `keys.repeat_interleave(self.group_size, dim=1)`(L73-L74) 把少数 KV 头广播到所有 Q 头共享。这一改动让 KV cache 显存按 `num_heads / num_kv_groups` 倍缩小——Llama 3 / Mistral / Qwen 长上下文推理省显存的核心机制就是这一行。把它跟上面那份 MHA + KV cache 并排读，能直接看清 MHA → MQA → GQA 的渐变路径。
