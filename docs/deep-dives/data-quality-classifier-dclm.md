---
title: 深度剖析 DCLM——用 fastText 质量分类器主导预训练数据筛选
description: DataComp-LM 的关键洞见：与其堆 heuristic 规则，不如训一个 fastText 二分类器，让模型自己定义"好数据"
topics: [data-engineering, pretraining-data, dclm, fasttext-classifier, data-quality, common-crawl]
prereqs: [/training/datasets, /deep-dives/data-pipeline-datatrove]
---

# 深度剖析 DCLM——用 fastText 质量分类器主导预训练数据筛选

::: info 一句话总结
DCLM 把"数据质量"当成一个**可学习的二分类问题**：用 OpenHermes-2.5 + ELI5 当 positive、随机 CommonCrawl 当 negative，训一个十几兆的 fastText 模型，然后用它从 240T token 的原始 CC 中筛出 4T 的 DCLM-Baseline——这套数据训出的 7B 模型 CORE 分数超过 LLaMA-2 7B、与 Mistral 7B 同档。
:::

> **来源声明**：本文基于 [mlfoundations/dclm](https://github.com/mlfoundations/dclm)（MIT License），论文 Li et al. 2024 *DataComp-LM: In search of the next generation of training sets for language models*（[arXiv:2406.11794](https://arxiv.org/abs/2406.11794)）。所有代码引用均给出 GitHub 永久链接，方便读者按行号对照原文。

---

## 一、体系定位：从 heuristic filter 到 learned filter

在数据工程的语境里，过滤一篇网页可以走两条路：

- **Heuristic filter（启发式过滤）**：写规则。词数 < 50 丢、符号比例 > 10% 丢、停用词 < 2 丢、重复 n-gram 比例 > 30% 丢……这是 Gopher / RefinedWeb / FineWeb / datatrove 的主流派。规则可解释、可审计、零模型推理开销，缺点是**规则的天花板取决于人类先验**。
- **Learned filter（模型化过滤）**：训一个分类器告诉你"这页是不是好数据"。早期的 GPT-3 用 logistic regression（WebText vs CC）、CCNet 用 KenLM perplexity（Wikipedia 当参考分布）。**DCLM 是这一派的现代代表**，把分类器从"辅助打分"提升到了"主筛选器"的地位。

DCLM 还顺手给整个数据社区贡献了一个 benchmark 范式：

| 维度 | DataComp-LM 怎么做 |
|------|---------------------|
| **训练代码** | 固定（基于 [open_lm](https://github.com/mlfoundations/open_lm)） |
| **模型架构 / 超参** | 固定（每个 scale 给定） |
| **算力 / token 预算** | 固定（400M-1x、1B-1x、7B-1x、7B-2x 等档位） |
| **下游评测** | 固定（53 个任务，CORE / EXTENDED / MMLU） |
| **可变量** | **只有数据集本身** |

也就是说，DCLM 把"数据质量"的玄学问题压缩成了一个**可重复的 leaderboard 比赛**——你拿同样的 240T raw pool，用任意 pipeline 筛出训练集，谁筛出的数据最终模型分数高，谁就赢。这种"控制变量 = 训练 recipe + 评测，开放变量 = 数据"的方式，在 Vision 领域有 [DataComp](https://www.datacomp.ai/) 的先例，DCLM 把它平移到了 LLM 预训练。

> 本系列已有 [`/training/datasets`](/training/datasets) 做宏观介绍（预训练 / SFT / 偏好数据三层结构），以及（后续会有）一篇 datatrove deep-dive 介绍 heuristic 流派。本文聚焦 DCLM 这条 learned filter 路线最关键的一环：**fastText 质量分类器**。

### 1.1 一段简史：learned filter 的演化

| 时间 | 系统 | 分类器形态 | positive 是什么 |
|------|------|-----------|-----------------|
| 2019 | GPT-2 / WebText | 隐式：用 Reddit outbound link 当人工"质量信号" | Reddit 高赞外链 |
| 2020 | GPT-3 | logistic regression on bigram features | WebText（被 GPT-2 间接背书） |
| 2020 | CCNet | KenLM 5-gram 困惑度分桶 | Wikipedia |
| 2023 | RedPajama-v1 | fastText `cc` vs `wikipedia` | Wikipedia |
| 2024 | **DCLM-Baseline** | **fastText `hq` vs `cc`** | **OpenHermes-2.5 + Reddit ELI5** |
| 2024 | FineWeb-Edu | BERT-style 分类器（蒸馏自 Llama-3-70B） | Llama-3 给的"教育性"打分 0–5 |

可以看到一条清晰的演化线：**positive 信号从"百科式参考分布"逐步迁移到"目标对话行为分布"**。DCLM 是这条线上的关键一跳——它第一次把"模型最终要会做的事（指令遵循）"当成了筛选预训练数据的标准。FineWeb-Edu 则是另一种方向，把"高质量"具象化为"教育性"，并用大模型当 oracle。

---

## 二、核心内容

### 2.1 三个池子：DCLM-Pool / DCLM-RefinedWeb / DCLM-Baseline

DCLM 把数据资产分成三档，理解这三个池子的关系是看懂全局的关键：

```
CommonCrawl WARC（PB 级）
        │ resiliparse 抽文本
        ▼
DCLM-Pool（240T token，未过滤）  ◀── 比赛的 raw input pool
        │ RefinedWeb pipeline
        │ （URL 黑名单 + 语言过滤 + 启发式 + BFF 去重）
        ▼
DCLM-RefinedWeb（约 100T token）  ◀── "已经做完 heuristic 但还没用 fastText"
        │ fastText OH2.5+ELI5 分类器（top 10% prob）
        ▼
DCLM-Baseline（约 4T token）     ◀── 论文的最强 baseline
```

注意两点：

1. **DCLM-Baseline 不是单纯的 fastText 过滤结果**，而是 RefinedWeb-style heuristic + fastText 的串联。即"先用规则筛掉显然垃圾，再用模型筛掉不够好的"。
2. **DCLM-RefinedWeb 单独保留**，是为了让参赛者能"只替换 fastText 这一步"，不必重复跑前面昂贵的 heuristic 阶段。这是 DCLM 留给社区的研究接口。

数据感：240T → 4T，**整体保留率约 1.7%**。其中 fastText 这一步本身（从 RefinedWeb 到 Baseline）保留 top ~10%。这意味着 fastText 分类器的阈值卡得相当激进。

### 2.2 为什么是 fastText 而不是 BERT / LLM

DCLM 团队在论文 §4 / §A 里讨论过分类器选择，本质是 **inference cost vs quality 的 Pareto 选择**。在动手实验之前，先快速回忆一下 fastText 的内部结构——它和"深度学习"几乎没关系：

```
输入文本 "deep learning tutorial"
   │ ① 切词 + char n-gram（"dee","eep","ep ", ...）
   │ ② 每个 token / n-gram 哈希到 [0, bucket_size) 的桶
   │   bucket_size 默认 2M，hash 冲突被有意接受（增加泛化）
   ▼
查表：每个 bucket id → dim 维向量
   │ ③ 所有向量做 mean pooling，得到一条 dim 维 doc 表征
   ▼
线性层 W ∈ R^{dim × num_labels} → softmax
   ▼
P(__label__hq | text) = 0.834
```

整个网络**没有非线性激活、没有 attention、没有 RNN**，纯加和 + 一次矩阵乘。这就是为什么它能 CPU 单核 30s/GB——所有 cost 都花在 hash 和向量加法上。这也是为什么它能在 200k 样本上 5 分钟跑完一个 epoch。

回到分类器选型对比：

| 分类器 | 单 GB 文本推理时间（CPU） | 训练成本 | 召回质量 | 是否可端到端 dump CC |
|--------|---------------------------|----------|----------|----------------------|
| fastText（bag of n-gram + 浅层 hash） | **~30-60 秒** | 单 CPU 数小时 | 中等 | ✅ |
| BERT-base 微调 | ~分钟级 / GB（GPU） | GPU 数小时 | 高 | 勉强可行 |
| LLM-as-judge（如 Llama-3-70B） | ~小时级 / GB | 不需训练，只需推理 | 很高 | ❌ 跑全 CC 不现实 |

DCLM 的关键洞察是：**当你要扫的是 240T token 的 CC 时，分类器必须是 CPU-only、单文档亚毫秒级的**。fastText 正好踩在这个甜区——参数量 ~100MB，纯 hash + 线性层，没有矩阵乘也没有 attention。

后续的 [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 走了一条混合路线：先用 Llama-3-70B 给 ~50 万样本打分（教育性 0–5 分），再蒸馏到一个**更小的 BERT-style 分类器**去扫全量 CC。这是对 DCLM 思路的一个很自然的延伸——**用 LLM 当 oracle、用小分类器当 worker**。

### 2.3 反直觉的训练配方：用 OpenHermes-2.5 + ELI5 当 positive

这是 DCLM 最受讨论的设计选择。你可能会想，预训练数据的"好"应该对标 Wikipedia / 学术论文 / 高质量书籍——但 DCLM 选的 positive 是：

- **OpenHermes-2.5**：~1M 条 instruction-following 对话数据（GPT-4 蒸馏 + Capybara + Airoboros 等混合）。
- **ELI5（Reddit r/explainlikeimfive）**：高赞解释性问答。

negative 则是从 RefinedWeb 里**随机采样**的 CC 文档。模型容量极小（fastText 浅层网络），训练目标就是一个二分类：`__label__hq` vs `__label__cc`。

为什么这反直觉？因为 instruction tuning 数据严格说是**post-training data**——你拿它给一个 base model 做 SFT 才合理，怎么能拿来定义"预训练数据的好坏"？

DCLM 的暗含假设是：

> **"模型最终要 align 到的形态"反过来定义了"预训练阶段应该长什么样"**。如果 base model 最终要会回答指令、解释概念、组织段落，那预训练时多看长得像"清晰自然语言解释"的 CC 网页就是有益的。

这其实和 GPT-4 / Phi-3 系列的 textbook-style 数据哲学异曲同工：**数据的"质量"不是绝对的，而是相对于你想要的最终模型行为定义的**。论文 Table 7 报告，相比 Wikipedia-as-positive 这种经典选择，OH2.5 + ELI5 在 CORE 上有约 1–2 个点的稳定优势。

::: warning 风险点
这个选择也意味着 DCLM-Baseline 在 base model 阶段就**轻微 SFT-bias** 了——在 RLHF / SFT 之前，模型已经"提前接触了"长得像 instruction 数据的网页。对纯 base model 评测（perplexity on raw text）可能不那么友好；对下游 RLHF 后的可用性反而更友好。
:::

### 2.4 fastText 的训练流程

正负样本各取 ~20 万条、按行写入 `train.txt`，每行用 `__label__hq` / `__label__cc` 前缀，调用一次 `fasttext.train_supervised`，几个小时单 CPU 就训完。DCLM 公开的官方 checkpoint 是 [`mlfoundations/fasttext-oh-eli5`](https://huggingface.co/mlfoundations/fasttext-oh-eli5)，文件名 `openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin`，可以直接 [下载使用](https://huggingface.co/mlfoundations/fasttext-oh-eli5/resolve/main/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin)。

关键超参在 `train_fasttext_classifier.py` 里都有：

- `dim=100`（词向量维度）
- `wordNgrams=2`（命令行实际用的；脚本默认 1）
- `epoch=5`、`lr=0.1`、`minCount=1`

注意 fastText 的"训练"本质是 **subword n-gram hash 表 + 一个线性 softmax**，没有反传到深层网络，所以单 CPU 几小时就能搞定 200k 样本。

### 2.5 推理与过滤的两阶段范式

DCLM 把"打分"和"卡阈值"拆成两个独立的 mapper：

1. **Enricher 阶段**：`classify_fasttext_hq_prob_enricher` 加载 fastText 模型，给每个 page 注入一个 `fasttext_oh_eli5_vs_rw_v2_prob` 字段。这一步**不丢 page**，只加 metadata。
2. **Filter 阶段**：`quality_filter` 读取上面那个字段，对比 `threshold=0.018112`，低于阈值的丢掉。

这种解耦带来一个工程上的好处——**昂贵的模型推理只跑一次**，之后想调阈值只需重跑廉价的 filter，不用重新做推理。论文里也提到他们扫了多档 threshold 才选定 0.018112 这个看起来很怪的数字（实际就是 top ~10% 分位点）。

### 2.5.1 一组关键的消融数据

DCLM 论文 §4 / Table 6–7 报告了几组重要对比，整理如下（数据基于 1B-1x scale，CORE 平均）：

| 配置 | CORE 平均 | 备注 |
|------|----------|------|
| RefinedWeb pipeline only（heuristic + dedup） | 36.2 | 不带任何 learned filter 的"纯启发式"基线 |
| + fastText（positive = Wikipedia） | 36.9 | RedPajama-v1 风格，几乎没提升 |
| + fastText（positive = OpenWebText） | 37.4 | 用 GPT-2 时代 Reddit 链接当 positive |
| + fastText（positive = OpenHermes-2.5） | 38.6 | 单用 OH2.5 就比 Wikipedia 强 1.7 个点 |
| **+ fastText（positive = OH2.5 + ELI5）** | **39.4** | **DCLM-Baseline 实际选用** |
| + DSIR（domain importance resampling） | 37.8 | 另一种 learned filter 思路，不如 fastText |

两个观察：

1. positive 分布的选择**对最终模型分数的影响 > 分类器架构的选择**。换言之"用什么定义好数据"比"用什么模型分类"更关键。
2. 仅靠这一步 fastText 过滤，CORE 就涨了 ~3 个点（36.2 → 39.4）。考虑到下游评测是 53 个任务的平均，3 个点是非常显著的差距——相当于把模型能力等价地推进了好几个月的训练算力。

### 2.6 fastText 在 pipeline 中的位置：在 BFF 去重之前还是之后？

看 `dclm_baseline_refinedweb.yaml` 和 README §2 的描述，DCLM-Baseline 的实际顺序是：

```
DCLM-Pool
  │ ① 文本抽取（resiliparse）
  │ ② RefinedWeb-style heuristic（URL 黑名单、长度、符号比、重复 n-gram、massive_web_repetition_filters 等）
  │ ③ BFF（Bloom Filter Fuzzy）模糊去重 ←── Rust 实现，单机
  │ ④ fastText OH2.5+ELI5 质量分类 + threshold 过滤
  │ ⑤ tokenize-shuffle
  ▼
DCLM-Baseline
```

**fastText 在去重之后**。这个顺序是经过对比实验选定的：先去重再过滤，能避免分类器把同一篇 viral spam 反复打高分（或反复打低分）影响阈值分布。BFF 用的是 Rust 实现，在 [`dedup/bff`](https://github.com/mlfoundations/dclm/tree/main/dedup/bff) 目录，与 Python ray pipeline 解耦，因为 BFF 本质需要全局视野（Bloom 集合），不适合 per-shard 并行。

::: details 为什么不能反过来：先 fastText 再去重？
反向顺序的问题在于 fastText 的"误判方差"会被放大。假设某条 spam 模板被复制了 1000 次，fastText 对单条的判断带 1–2% 抖动——先过滤的话，可能 600 条被留下、400 条被丢。但如果先去重，1000 条压成 1 条，fastText 只判一次，结果稳定。

更关键的是：fastText 推理是**整个 pipeline 中最贵的一步**（CPU 时长占比超 50%）。先做廉价的 BFF 去重把数据量砍一半，能直接省掉一半的 fastText 推理算力。
:::

### 2.6.1 阈值是怎么定的：threshold = 0.018112 的来历

这个数字在论文 §A.7 / 配置文件里写死。它的产生过程：

1. **跑一遍全量 enricher**：在 RefinedWeb 全量上跑 fastText，每个 page 拿到一个 `hq_prob` 分数。
2. **画分数分布直方图**：DCLM 论文 Figure 8 给了这张图——分布大致是双峰的，一峰在 0.001 附近（明显 spam）、另一峰在 0.05+（明显高质）。
3. **扫多档 threshold**：在 1B-1x scale 上把 threshold 扫 [0.005, 0.01, 0.02, 0.05] 几档，分别训模型评测。
4. **选 CORE 最高的那档**：实测 0.018112 对应 ~top 10% 分位、CORE 最优。

为什么不是 round number 像 0.02？因为 0.018112 就是某次实验里 ~10% 分位点的具体浮点值，作者没有刻意 round。这也提示我们：**这个数字是经验最优、对池子敏感**。换一个底层池（比如 SlimPajama），需要重新扫一遍。

### 2.6.2 与 FineWeb-Edu 的对比：两条 learned filter 的路线之争

FineWeb 团队（HuggingFace）在 DCLM 之后发布的 [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 是 DCLM 思路的一个变体，也是目前公开数据集里和 DCLM-Baseline 直接对位的方案。两者的差异值得专门对比：

| 维度 | DCLM-Baseline | FineWeb-Edu |
|------|---------------|-------------|
| 底层池 | DCLM-Pool（240T，CC-only） | FineWeb（15T，CC + heuristic） |
| 分类器架构 | fastText（n-gram + 线性） | BERT-style 小模型（Snowflake-arctic-embed） |
| positive 信号来源 | 现成数据集（OH2.5 + ELI5） | **Llama-3-70B 给打分** |
| positive 标注规模 | ~20 万样本（直接抓） | ~50 万样本（LLM 标注成本：~$10k） |
| 标签形态 | 二分类（hq / cc） | 5 档回归（教育性 0–5 分） |
| 推理 throughput | ~30s/GB（CPU） | ~分钟/GB（GPU） |
| 公开数据规模 | 4T token | 1.3T token（Edu 子集） |
| 7B 模型 CORE v2 | 56.0（2.6T tok） | 较低（见 leaderboard） |

可以看到这是两条**不同质的学习信号**：

- DCLM 走"**用现成的好数据当 positive**"——便宜、零标注成本、但 positive 分布的"好"是静态、间接定义的。
- FineWeb-Edu 走"**让大模型现场判断什么是好**"——昂贵、但能动态、显式定义"好"的语义（"教育价值"）。

谁更优？在 1.3T 这个 token budget 下，DCLM-Baseline 仍然占优；但 FineWeb-Edu 的策略更可扩展（你可以让 LLM 给"代码可读性"、"事实准确性"、"叙事连贯性"等任意维度打分），是更通用的范式。可以预见后续的数据 pipeline 会越来越多走 LLM-as-oracle 蒸馏小分类器这条路。

### 2.7 数据感：分类器的吞吐与训练成本

按 README 与论文 §A：

- **fastText 训练**：200k 训练样本、单 CPU、几小时内完成，模型大小 ~1.5GB（含完整 hash 表）。
- **fastText 推理**：单 CPU 单核 30–60 秒/GB 文本。EC2 `i4i.4xlarge` 上配 `--ray_num_cpus 2` 是推荐配置（因为模型常驻内存几 GB，不能给每个 worker 都加载一份）。
- **过滤掉的比例**：threshold=0.018112 对应保留 top ~10%，从 ~100T 的 RefinedWeb 缩到 ~4T 的 Baseline。
- **完整 pipeline**：论文里报告的全量 240T → 4T pipeline 在数百节点 ray cluster 上跑了几天到一两周量级（具体数字可参考论文 §A）。

对比维度上看 leaderboard：在 7B / 2.6T token 这个档位，**DCLM-Baseline CORE v2 = 56.0、MMLU = 63.7**，超过同档 token 数的 RefinedWeb（Falcon 7B/1T，CORE v2 = 41.6）、OLMo-1.7（CORE v2 = 45.8）、MAP-Neo（CORE v2 = 49.0），与 Mistral-0.3 7B（CORE v2 = 55.7）持平。这个差距很多就来自 fastText 分类器这一步。

---

## 三、手撕源码

下面三段代码涵盖了 fastText 分类器的**训练 → 推理 → 配置接入**全链路。每段都尽量短、把 WHY 写在注释里。

### 3.1 fastText 训练入口

入口很薄，本质就是 `fasttext.train_supervised` 的一层 argparse 包装。关键是默认超参的选择：

```python
# 来自 baselines/train_fasttext_classifier.py:33-53
# 默认超参 lr=0.1 / dim=100 / epoch=5 / wordNgrams=1
# 但 README 推荐用 wordNgrams=2，让 bigram 也进 hash 表
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--dim", type=int, default=100)        # 词向量维度，100 维已经足够
parser.add_argument("--ws", type=int, default=5)           # context window，对二分类影响不大
parser.add_argument("--epoch", type=int, default=5)        # epoch 多了反而过拟合，200k 数据 5 轮即可
parser.add_argument("--minCount", type=int, default=1)     # 不过滤低频词，因为 CC 长尾词对区分很有用
parser.add_argument("--wordNgrams", type=int, default=1)   # README 用 2，命令行覆盖

# train_supervised 本质：subword hash → 线性 softmax → CE loss
hyperparams = {'lr': args.lr, 'dim': args.dim, 'ws': args.ws,
               'epoch': args.epoch, 'minCount': args.minCount,
               'wordNgrams': args.wordNgrams}
model = fasttext.train_supervised(input=train_input, **hyperparams)
```

> 引用：[train_fasttext_classifier.py:33-70](https://github.com/mlfoundations/dclm/blob/main/baselines/train_fasttext_classifier.py#L33-L70)

注意 `minCount=1`——这和文本分类的常识（过滤低频词降噪）反着来。原因是 CC 上的"垃圾标识词"（比如 SEO 关键词堆叠、特定垃圾站点的固定词组）天然是长尾，过滤掉反而会丢失分辨力。

### 3.2 fastText 推理：把概率写回 page

```python
# 来自 baselines/mappers/enrichers/quality_prediction_enrichers_calc_fasttext.py:33-64
def classify_fasttext_hq_prob(model, content: str) -> dict:
    output = {}

    # WHY: fastText 不会原生处理换行，把多行 page 压成一行
    # 注意这里没做截断 —— 长 page 会被 fastText 内部按 max_line_size 截断
    text = " ".join(content.strip().splitlines())

    pred = model.predict(text)              # 返回 (label_tuple, prob_tuple)
    (pred_label, pred_prob) = pred
    pred_label = pred_label[0]              # 只取 top-1 标签
    hq_prob = pred_prob[0]

    # WHY: fastText 输出的是 top-1 标签的概率
    # 如果 top-1 是 "cc"（low-quality），我们要的"高质量概率"得用 1 - p 翻一下
    # 这一步如果忘了，threshold 会卡在错误一边、过滤完全反向
    if pred_label == "__label__cc":
        hq_prob = 1 - hq_prob

    return hq_prob
```

> 引用：[quality_prediction_enrichers_calc_fasttext.py:33-64](https://github.com/mlfoundations/dclm/blob/main/baselines/mappers/enrichers/quality_prediction_enrichers_calc_fasttext.py#L33-L64)

这段 20 行代码是整个 DCLM-Baseline 数据筛选的"最热路径"——240T token 每一篇都会过它一次。`" ".join(splitlines())` 这种朴素操作里，藏着对 fastText 输入格式的深刻理解（fastText 把换行视作样本分隔符）。

外层用 factory 模式做了一次模型懒加载——每个 worker 进程只 `load_model` 一次，避免每页都重新读 1.5GB 二进制：

```python
# 来自 baselines/mappers/enrichers/quality_prediction_enrichers_calc_fasttext.py:67-89
@factory_function
def classify_fasttext_hq_prob_enricher(model_filename=RPJ_MODEL_FILENAME,
                                       key: str = "fasttext_hq_prob",
                                       overwrite: bool = False):
    # WHY: 模型加载放在闭包外面 —— factory_function 装饰器保证整个 worker 生命周期只跑一次
    # 否则每页 reload 1.5GB 模型，IO 直接打爆
    model = load_fasttext_model(model_filename)

    def enrich(page: Dict) -> List[Dict]:
        # WHY: 不允许覆盖已有 key，避免下游误用了上一轮的旧分数
        assert overwrite or key not in page, f"cannot overwrite an existing key {key}"
        page[key] = classify_fasttext_hq_prob(model, page[CONTENT])
        return [page]   # enricher 永远返回长度 1 的 list（不丢 page）

    return enrich
```

> 引用：[quality_prediction_enrichers_calc_fasttext.py:67-89](https://github.com/mlfoundations/dclm/blob/main/baselines/mappers/enrichers/quality_prediction_enrichers_calc_fasttext.py#L67-L89)

### 3.3 quality_filter：拿概率卡阈值

`enrich` 只是给 page 加字段，真正决定"这条留不留"的是下面这段：

```python
# 来自 baselines/mappers/filters/metadata_filters.py:52-76
def quality_filter(page, key='fasttext_hq_prob', threshold=0.0,
                   lower_better=False, key_must_exist=True):
    # WHY: 强制要求 enricher 已经跑过 —— 没有分数就过不了 filter
    # 这是"快速失败"原则：宁可炸掉，也不要静默地把所有页放过
    if key_must_exist:
        assert key in page, f'The input JSON object does not have a {key} field'
        quality_score = page[key]
    else:
        # WHY: 如果允许缺失，缺失值要给一个"必然被过滤掉"的极端分数
        missing_score = float('inf') if lower_better else -float('inf')
        quality_score = page.get(key, missing_score)

    # WHY: lower_better 是为了同一个 filter 复用到 perplexity 这种"越低越好"的指标
    # 对 fastText hq_prob 永远走 else 分支
    if lower_better:
        return [page] if quality_score <= threshold else []
    else:
        return [page] if quality_score >= threshold else []
```

> 引用：[metadata_filters.py:52-76](https://github.com/mlfoundations/dclm/blob/main/baselines/mappers/filters/metadata_filters.py#L52-L76)

`return [page]` / `return []` 是 DCLM mapper 的标准约定——**返回 list 而不是 page**，从而让"过滤"和"复制 / 拆分"用同一套接口表达。一个 mapper 想保留就 `[page]`，想丢就 `[]`，想拆成多页就 `[page1, page2]`。

### 3.4 把上面三件事串起来：YAML 配置

```yaml
# 来自 baselines/baselines_configs/fasttext_filter.yaml:1-9
- source: cc
  steps:
    - func: classify_fasttext_hq_prob_enricher          # ① enricher：注入分数
      model_filename: fasttext_oh_eli5.bin              # 由 setup.py 从 HuggingFace 下载
      key: fasttext_oh_eli5_vs_rw_v2_prob               # 自定义字段名
    - func: quality_filter                              # ② filter：卡阈值
      key: fasttext_oh_eli5_vs_rw_v2_prob               # 必须和上面 enricher 的 key 一致
      threshold: 0.018112                               # top ~10% 分位
```

> 引用：[fasttext_filter.yaml:1-9](https://github.com/mlfoundations/dclm/blob/main/baselines/baselines_configs/fasttext_filter.yaml#L1-L9)

这就是 DCLM 整个数据筛选哲学最凝练的体现——**两段 YAML、九行代码，决定了 4T token 训练集长什么样**。模型下载逻辑在 [`setup.py:107-127`](https://github.com/mlfoundations/dclm/blob/main/setup.py#L107-L127)，会从 [`huggingface.co/mlfoundations/fasttext-oh-eli5`](https://huggingface.co/mlfoundations/fasttext-oh-eli5) 拉取 `.bin` 到本地的 `quality_prediction_enrichment_models/` 目录。

---

## 四、苏格拉底时刻

::: tip 1. fastText 在代码 / 数学等 OOD 内容上失效会怎样？
positive 数据是 OH2.5（自然语言对话）+ ELI5（自然语言解释），里面几乎没有代码块和 LaTeX 公式。一个 GitHub README 含大量 ` ``` ` 代码段、或一篇 arXiv 论文含很多 `$\sum$`，会被分类器打成什么分？
你猜会偏向 `__label__cc`（低质）。如果是的话，DCLM-Baseline 的代码 / 数学覆盖会不会被压低？这能解释 DCLM 在 GSM8K 之类任务上不如 LLaMA-3 吗？怎么验证？
:::

::: tip 2. positive 用 OpenHermes 是不是给 base model 偷偷"提前 SFT"？
OH2.5 是 GPT-4 蒸馏的对话数据，长得就是 "User: ... Assistant: ..." 格式。当我们用它当 fastText 的 positive，分类器会偏好"长得像对话"的 CC 网页（论坛 QA、Stack Overflow 回答、教程文章）。
这意味着 DCLM-Baseline 训出的 base model 已经**在预训练时就接触了大量 instruction-style 文本**——这究竟是 feature 还是 bug？对纯 perplexity 评测、对后续 RLHF、对 in-context learning 各自的影响是什么？
:::

::: tip 3. 为什么是 fastText 而不是更强的 BERT？
DCLM 团队也做过 BERT 分类器对比（论文 §A）。结论是 BERT 召回质量略高、但推理成本上去后整体 cost-quality Pareto 不如 fastText。
那么如果你只有 1T token 要筛（不是 240T），是不是 BERT 反而更划算？FineWeb-Edu 用 Llama-3 当 oracle 蒸馏 BERT 的策略，是不是说明"分类器选型"本质是"待筛 token 量"的函数？
:::

::: tip 4. threshold = 0.018112 这个数字怎么定的？
看起来很怪——既不是 0.5（朴素二分类阈值），也不是 0.1（top 10% 直觉阈值）。它实际是 OH2.5+ELI5 模型在 RefinedWeb 全量上的 ~10% 分位点经验值。
如果你换一个 base 池子（比如换成 SlimPajama），同一个 fastText 模型的 threshold 还能直接复用吗？为什么不在 pipeline 里做 dynamic percentile thresholding 而要写死一个浮点数？
:::

::: tip 5. learned filter 是否会把数据多样性"卷死"？
当 fastText 模型偏好某种文体（比如教程式解释），它会系统性地丢掉小众但有价值的内容（诗歌、口述历史、方言论坛）。这种"多样性损失"在小规模评测上看不出来（CORE 任务都是知识 + 推理），但可能影响模型的长尾创造力。
你怎么设计一个评测来量化"learned filter 的多样性损害"？为什么主流 leaderboard 都不报告这个？
:::

---

## 五、面试考点

1. **手画 DCLM 数据 pipeline 全图**：DCLM-Pool（240T）→ heuristic + BFF dedup → DCLM-RefinedWeb（~100T）→ fastText OH2.5+ELI5 → DCLM-Baseline（~4T）。能说清每一步丢掉的比例与原因。
2. **fastText 分类器训练数据是怎么构造的**：positive = OpenHermes-2.5 + Reddit ELI5 各取 ~10 万；negative = RefinedWeb 随机采样 ~20 万；标签格式 `__label__hq` / `__label__cc`；超参 `wordNgrams=2, dim=100, epoch=5`。
3. **为什么不用 LLM-as-judge**：240T token 的扫描成本不可承受。fastText CPU 单核 30–60s/GB，BERT 分钟级/GB（GPU），LLM 小时级/GB——只有 fastText 能 end-to-end 扫完整个 CC。
4. **enrich-then-filter 两段式设计的工程意义**：模型推理（贵，跑一次写到 metadata）和阈值过滤（廉价，可反复重跑）解耦，方便扫 threshold 找最优分位点。
5. **DCLM-Baseline 与 RefinedWeb / FineWeb-Edu 的关系**：RefinedWeb 是纯 heuristic（FineWeb 同源）；DCLM-Baseline = RefinedWeb-style heuristic + BFF + fastText；FineWeb-Edu = 用 Llama-3 当 judge 蒸馏 BERT 分类器，思路是 DCLM 的"LLM 蒸馏分类器"变体。能说清这条技术演进线。

---

## 六、推荐资源

| 资源 | 链接 | 内容 |
|------|------|------|
| DCLM 源仓库 | [github.com/mlfoundations/dclm](https://github.com/mlfoundations/dclm) | 全套 pipeline + 配置 + 训练代码（MIT） |
| DCLM 论文 | [arXiv:2406.11794](https://arxiv.org/abs/2406.11794) | Li et al. 2024，含 53 任务评测细节与消融实验 |
| 官方 fastText checkpoint | [huggingface.co/mlfoundations/fasttext-oh-eli5](https://huggingface.co/mlfoundations/fasttext-oh-eli5) | 直接可下载的 `.bin`，~1.5GB |
| OpenHermes-2.5 数据集 | [huggingface.co/datasets/teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | fastText 的 positive 来源之一 |
| FineWeb-Edu | [huggingface.co/datasets/HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | DCLM 思路的 LLM-as-judge 版本，1.3T token 教育性筛选 |
| open_lm 训练框架 | [github.com/mlfoundations/open_lm](https://github.com/mlfoundations/open_lm) | DCLM benchmark 用的标准训练代码 |
| fastText 官方文档 | [fasttext.cc/docs/en/supervised-tutorial.html](https://fasttext.cc/docs/en/supervised-tutorial.html) | `train_supervised` 超参与训练数据格式 |
| DCLM Leaderboard | [datacomp.ai/dclm/leaderboard](https://datacomp.ai/dclm/leaderboard) | 看社区在 filtering / mixing track 上的最新提交 |

---

::: info 一句话回顾
DCLM 用一个 1.5GB 的 fastText 模型替代了"人写一百条规则"的传统范式——它不是在告诉你"什么是好数据"，而是在让数据**自我标注**：你想要的最终模型行为（OH2.5 对话）就是它自己的好数据定义。这是 learned filter 流派给数据工程的最大启示。
:::
