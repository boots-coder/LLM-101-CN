---
title: 深度剖析 datatrove——从 CommonCrawl 到 FineWeb 的工业流水线
description: HuggingFace datatrove 的 PipelineStep + Executor 抽象、Gopher/C4 过滤规则、4 阶段 MinHash 近似去重
topics: [data-engineering, pretraining-data, fineweb, datatrove, minhash-dedup, quality-filter, common-crawl]
prereqs: [/training/datasets, /training/pretraining]
---

# 深度剖析 datatrove——从 CommonCrawl 到 FineWeb 的工业流水线

::: info 一句话总结
datatrove 把「90 TB CommonCrawl dump → 0.3 TB 高质量 token」拆成可独立 sharding 的 PipelineStep 算子，再用 Executor 抽象同时支持 1 节点本地调试和 1000+ slurm 任务并发；FineWeb 15T token 就是这套流水线在 96 个 dump 上重复 96 次的产物。
:::

::: tip 来源声明
本文基于 [huggingface/datatrove](https://github.com/huggingface/datatrove)（Apache 2.0），它是 [FineWeb 数据集](https://huggingface.co/datasets/HuggingFaceFW/fineweb) 的同源工程实现。所有源码引用均链接到上游仓库 `main` 分支，本文仅做工程化解读，不复制连续大段源码。前置阅读：[预训练数据生态](/training/datasets) 与 [预训练](/training/pretraining)。
:::

## 体系定位：为什么需要又一套数据流水线

预训练数据这条赛道的主要里程碑可以粗略串成一条线：

| 数据集 | 年份 | 规模 | 关键工程贡献 |
|--------|------|------|--------------|
| C4 | 2019 | ~750 GB | 一次性 TF Datasets 脚本 + 启发式行过滤 |
| The Pile | 2020 | ~825 GB | 22 个来源拼接，基本无大规模去重 |
| RedPajama-1T | 2023 | ~1.2 TB | 复刻 LLaMA-1 配方，spark + 单机脚本混合 |
| RefinedWeb | 2023 | ~5 TB（公开 600 GB） | 提出「精挑 web 也能打 The Pile」 |
| SlimPajama | 2023 | 627 B token | 全量 MinHash 去重 RedPajama |
| **FineWeb / FineWeb-Edu** | 2024 | **15 T / 1.3 T token** | datatrove 流水线 + 96 个 dump 端到端 |
| DCLM-Pool | 2024 | ~240 T token | 与 datatrove 互补，主推 fastText 质量分类器 |

datatrove 与早期一次性 spark 脚本的差异不在算法新颖度，而在**工程抽象选择**：

- **可重入**：每个算子的输入输出都是磁盘上的 jsonl/parquet 分片，单步崩了重跑就行，不用从 WARC 头来过。
- **可移植**：本地 4 核迭代规则用 `LocalPipelineExecutor`，上 100 节点就换 `SlurmPipelineExecutor`，pipeline 列表一字不改。
- **可观测**：每个 PipelineStep 自带 `Stats` 计数器，过滤掉的文档可被 `exclusion_writer` 完整 dump 出来事后检查（这是调阈值的命根子）。

::: warning 30T token 的工程逻辑
当数据规模来到 15T~30T token 这一档，「跑 30 天发现某个 Gopher 阈值订错」的代价是**整个 dump 的算力打水漂**。所以工程抽象不是 nice-to-have，而是项目能不能成立的硬约束。下文每一个设计选择都要带着这个视角看。
:::

## 核心内容

### 一、PipelineStep 抽象：每步都是一个独立算子

datatrove 的核心抽象只有两个类：`PipelineStep`（算子）和 `PipelineExecutor`（调度器）。先看算子。

每个算子继承 `PipelineStep`，强制实现一个生成器风格的 `run` 方法，输入是 `Document` 流，输出也是 `Document` 流。这意味着：

- **过滤器**就是「输入 1 个 doc，要么 yield 它要么不 yield」
- **去重器**就是「读 sharded 签名文件，跨 shard 比对后 yield 保留的 doc」
- **写出器**就是「把流写到 jsonl/parquet 后再 yield 一遍（或不 yield，做终端）」

```python
# 简化版：base.py 的核心契约
class PipelineStep(ABC):
    name: str = None
    type: str = None

    def __init__(self):
        super().__init__()
        # 每步自带 Stats 计数器：total / forwarded / dropped
        self.stats = Stats(str(self))

    @abstractmethod
    def run(self, data: DocumentsPipeline,
            rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        # data 是上一步的生成器；rank/world_size 用于 shard 数据切分
        # 任何一步都可以根据 rank 决定自己读哪一片输入，互不重叠
        if data:
            yield from data
```

源码：[base.py:L9-L114](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/base.py#L9-L114)。`rank` 和 `world_size` 是整套体系能横向扩展的根本——同一份代码，1 个进程时 `rank=0, world_size=1`，1000 个 slurm 任务时 `rank∈[0,999]`，每个进程根据 rank 自动挑自己那一份输入文件，无需中心调度器。

::: details 为什么是生成器而不是 list
WARC 单文件解压后常常 5~10 GB，一个 dump 90 TB。如果用 list 收集再传给下一步，光是中间结果就会撑爆内存。生成器把流水线压成「逐文档拉模型」，常驻内存只有 O(1) 个文档加上各算子的状态字典。
:::

### 二、Executor 抽象：local/slurm 同形对偶

`PipelineExecutor` 抽象出三件事：`run()`（怎么调度任务）、`world_size`（总任务数）、`get_distributed_env()`（分布式环境变量）。

```python
# 简化版：base.py + local.py 的核心调度
class PipelineExecutor(ABC):
    @abstractmethod
    def run(self): ...

    def _run_for_rank(self, rank: int, ...) -> PipelineStats:
        # 1. 跳过已完成 rank（断点续跑的关键）
        # 2. 把 pipeline 列表串成生成器链：data = step(data, rank, world_size)
        # 3. 写 stats.json + completions/<rank> 标记完成
        ...

class LocalPipelineExecutor(PipelineExecutor):
    def run(self):
        # workers 个进程的 multiprocess.Pool，imap_unordered 跑所有 rank
        with ctx.Pool(self.workers) as pool:
            stats = list(pool.imap_unordered(self._launch_run_for_rank, ranks_to_run))
```

源码：[executor/base.py:L37-L120](https://github.com/huggingface/datatrove/blob/main/src/datatrove/executor/base.py#L37-L120) 与 [executor/local.py:L84-L154](https://github.com/huggingface/datatrove/blob/main/src/datatrove/executor/local.py#L84-L154)。Slurm 版的差异只在 `run()`：它把同样一段 `pipeline` 序列化进 sbatch 脚本，提交 `tasks` 个 array job，每个 job 通过 `SLURM_ARRAY_TASK_ID` 拿到自己的 `rank`，然后调一样的 `_run_for_rank`。

**这就是「同形对偶」**：你在 MacBook 上跑 `LocalPipelineExecutor(pipeline=[...], tasks=4)` 调通的 pipeline 列表，原封不动塞进 `SlurmPipelineExecutor(pipeline=[...], tasks=8000)`，就是 FineWeb 那 8000 个并发 cpu 任务的真实写法。`examples/fineweb.py` 的第 33-73 行就是这么干的：

```python
# 来源:examples/fineweb.py（节选，节略 paths）
main_processing_executor = SlurmPipelineExecutor(
    job_name=f"cc_{DUMP_TO_PROCESS}",
    pipeline=[
        WarcReader(f"s3://commoncrawl/crawl-data/{DUMP_TO_PROCESS}/segments/", ...),
        URLFilter(exclusion_writer=JsonlWriter(...)),       # 1. URL 黑名单
        Trafilatura(favour_precision=True),                  # 2. HTML→text
        LanguageFilter(exclusion_writer=...),                # 3. fastText 语言识别
        GopherRepetitionFilter(exclusion_writer=...),        # 4. Gopher 重复性
        GopherQualityFilter(exclusion_writer=...),           # 5. Gopher 质量
        C4QualityFilter(filter_no_terminal_punct=False, ...),# 6. C4 行级规则
        FineWebQualityFilter(exclusion_writer=...),          # 7. FineWeb 自家加项
        JsonlWriter(...),
    ],
    tasks=8000, time="10:00:00", mem_per_cpu_gb=2,
)
```

完整脚本：[examples/fineweb.py:L33-L73](https://github.com/huggingface/datatrove/blob/main/examples/fineweb.py#L33-L73)。请注意过滤层级是**按计算成本递增**排列的：URL 黑名单是 O(1) 查 hash，Trafilatura 要解析 HTML，LanguageFilter 要跑 fastText 模型，最贵的 GopherRepetition / FineWebQuality 才会在末尾跑——这样 80% 的垃圾在最便宜的几步就被掐掉了。

### 三、过滤层级：按计算成本递增的漏斗

| # | 算子 | 单 doc 成本 | 典型淘汰率 | 关键依赖 |
|---|------|------------|------------|----------|
| 1 | `URLFilter` | O(1)，aho-corasick | 5%-20% | UT1 黑名单 + 自定义关键词 |
| 2 | `Trafilatura` | ~10ms（HTML→text） | n/a（提取不是过滤） | trafilatura |
| 3 | `LanguageFilter` | ~2ms / fastText | ~40%（保留 en） | fasttext + ft176/glotlid |
| 4 | `GopherRepetitionFilter` | O(L)，n-gram 计数 | 10%-25% | 纯 Python |
| 5 | `GopherQualityFilter` | O(L)，10+ 启发式 | 15%-30% | 纯 Python |
| 6 | `C4QualityFilter` | O(L)，行级规则 | 5%-15% | 句子分词 |
| 7 | `FineWebQualityFilter` | O(L) | 5%-10% | 自家加项 |
| 8 | `MinhashDedup`（4 阶段） | ★ 重头戏 | 60%-90% | numpy + fsspec |
| 9 | `PIIFormatter` | O(L) | n/a（替换） | 正则 |

数据感（FineWeb-paper 数量级）：

- CommonCrawl 单 dump 原始 WARC：**~90 TB（压缩）**
- Trafilatura 提取后：~6 TB
- 语言过滤 + Gopher + C4 + FineWeb：**~1-3 TB**
- 整 dump MinHash 去重之后：**~0.3-1 TB**
- 96 个 dump 全部跑完拼起来：**FineWeb 15T token**

**漏斗收敛比 ≈ 100×～300×**——这数据是非常残酷的，意味着你在 CommonCrawl 里 99% 以上的字节都是噪声。

### 四、Gopher 质量过滤：十几条启发式规则的集合体

Gopher 论文（[arxiv 2112.11446](https://arxiv.org/abs/2112.11446)）把「web 上的废话」总结成了一组启发式特征。datatrove 几乎逐字翻译了论文里的阈值表：

```python
# 来源:gopher_quality_filter.py 的核心判定逻辑（重写注释版）
def filter(self, doc):
    words = split_into_words(doc.text, self.language)
    n_words = len(words)
    # 去掉纯标点的 token，避免 ascii art / 代码片段误判
    non_symbol_words = [w for w in words if any(ch not in PUNCTUATION_SET for ch in w)]

    # 规则 1: 词数下界（< 50 词的多半是导航条 / 短评）
    if n_non_symbol_words_words < self.min_doc_words:
        return False, "gopher_short_doc"
    # 规则 2: 平均词长 ∈ [3, 10]（< 3 多是聊天缩写 / 验证码；> 10 是URL拼接）
    avg_n_words = np.mean([len(w) for w in non_symbol_words])
    if avg_n_words < self.min_avg_word_length: return False, "gopher_below_avg_threshold"
    # 规则 3: # 号 / 省略号占比（导航页和被截断爬虫常见）
    if text.count("#") / n_words > self.max_symbol_word_ratio: return False, "gopher_too_many_hashes"
    # 规则 4: 项目符号开头 > 90%（菜单页）；规则 5: 行尾省略号 > 30%（截断列表页）
    # 规则 6: 含字母词 < 80%（数字表 / 代码 dump）
    # 规则 7: 必含 ≥2 个停用词 the/be/to/of/and/that/have/with（保证是英文连贯文本）
    if len(self.stop_words.intersection(set(words))) < self.min_stop_words:
        return False, "gopher_enough_stop_words"
    return True
```

源码：[gopher_quality_filter.py:L13-L125](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/gopher_quality_filter.py#L13-L125)。

::: warning 阈值是工程艺术，不是科学
`min_avg_word_length=3` 看起来像凭空捏造的数字，背后其实是一种「英文常识」：英语的平均词长 ~4.7，给 3-10 留余量。但这条规则放到中文/日文 web 上会**全军覆没**——中文 token 化后单字平均长度就是 1，整个语料会被 99% 干掉。这就是为什么 LanguageFilter 必须放在 Gopher 之前：先确认是英语，再用英语规则。
:::

GopherRepetitionFilter 是配套的另一半，专治 SEO 农场和爬虫复制粘贴。它检查 `top 2~4-gram` 字符占比和 `5~10-gram` 重复占比（参考 Gopher 论文 Table A1）：

```python
# 来源:gopher_repetition_filter.py 的判定（节略）
# top 2-gram 字符占比上限 0.20、3-gram 0.18、4-gram 0.16
# duplicate 5-gram 字符占比上限 0.15、6-gram 0.14、…、10-gram 0.10
# n 越大阈值越紧——长 n-gram 重复说明在「整段照抄」
for n, n_frac in self.dup_n_grams:
    if find_all_duplicate(words, n) / len(text) > n_frac:
        return False, f"duplicated_{n}_n_grams"
```

源码：[gopher_repetition_filter.py:L73-L142](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/gopher_repetition_filter.py#L73-L142)。

C4QualityFilter 的逻辑层级更细：先按行 split，丢掉 `Javascript`、`{` 出现的行，丢掉句号问号叹号都没有的行，再保证整篇至少 5 句。源码：[c4_filters.py:L27-L60](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/c4_filters.py#L27-L60)。

### 五、近似去重的 4 阶段 MinHash：FineWeb 的真正核心

精确去重（hash 整篇文档）只能干掉「字符级一模一样」的副本，但 web 上更多是「同一篇新闻被 30 个网站转载、各自加了点 banner」。**MinHash + LSH 是工业上唯一在 30T token 规模可行的近似去重方案**。

#### 5.1 为什么不能一步到位

朴素思路是把所有文档两两算 Jaccard，但 N=10^10 量级下这是 N² ~ 10^20 次比较，物理上不可能。datatrove 把它拆成 4 阶段，每阶段单独 sharding：

| 阶段 | 名字 | 输入 | 输出 | 瓶颈 |
|------|------|------|------|------|
| 1 | `MinhashDedupSignature` | jsonl 文档 | `bucket_xxx/rank_yyy.minhash.sig`（B 个 bucket × N 个 rank） | CPU（n-gram 哈希） |
| 2 | `MinhashDedupBuckets` | 阶段 1 全部 sig 文件 | `<bucket>_<worker>.dups`（重复对） | 磁盘 IO + 内存（堆排序） |
| 3 | `MinhashDedupCluster` | 阶段 2 所有 .dups | `<rank>.remove`（要删的 doc id） | 内存（并查集，必须单机） |
| 4 | `MinhashDedupFilter` | 原文 + .remove | 去重后 jsonl | CPU + 磁盘 IO |

::: tip 为什么不能合并阶段
- **1↔2 不能合并**：阶段 1 必须严格按 `(rank,doc_id)` 顺序写，阶段 2 才能在每个 bucket 内做归并排序。如果一边算 sig 一边比对，bucket 这维不能并行。
- **2↔3 不能合并**：阶段 2 输出的「重复对」是局部的（A↔B、B↔C），要算「连通分量」必须把所有 bucket 的重复对汇总到一台机器跑并查集。这一步**必须 `tasks=1`**。
- **3↔4 不能合并**：阶段 3 输出的是「全局要删的 doc id」，阶段 4 要按原 shard 重读文档把它们标删。3 是单机 cluster，4 是 `TOTAL_TASKS` 个并发 filter。
:::

#### 5.2 阶段 1：算签名

每篇文档先做归一化（小写、去标点）→ 切 `n_grams=5` 个词的 shingle → 哈希成 uint64 → 用 `B*R = 14*8 = 112` 组 (a,b) 系数生成 112 个 permuted hash → 取每组的最小值 → 14 个 bucket，每个 bucket 8 个 hash。

```python
# 来源:dedup/minhash.py 的签名生成（注释重写版）
def get_signature(self, shingles):
    # a, b 是预计算的 (1, 112) 维系数；shingles 是 (N, 1) 的 uint64
    a, b = self.parameters
    # 对每个 shingle 做 112 次「(s*a + b) % p」的 universal hashing
    phv = (shingles * a + b) % _mersenne_prime
    # 沿 N 这维取 min（这就是 MinHash 的「Min」），得到 (1, 112) 签名
    # 再 split 成 14 个 bucket × 8 个 hash，每个 bucket 后续单独做 LSH
    return [x.tolist() for x in np.split(np.min(phv, axis=0), self.config.num_buckets)]
```

源码：[dedup/minhash.py:L172-L188](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/dedup/minhash.py#L172-L188)。每篇文档输出 14 个 bucket，每个 bucket 写一行 `(8 个 hash + doc_idx)` 进 `bucket_BBB/rank_RRR.minhash.sig`。**关键设计**：bucket 维和 rank 维都在路径里，下一阶段就能按 bucket 切任务、按 rank 切 worker。

#### 5.3 阶段 2：bucket 内找重复

LSH 的核心命题：如果两个文档的某一个 bucket 的 8 个 hash 完全相同，那它们 Jaccard 相似度大概率 ≥ 阈值。所以这一步只要在每个 bucket 内找「8 个 hash 一模一样」的文档对。

datatrove 的实现是**多路归并 + 优先队列**：每个 sig 文件已按 hash 排好序（阶段 1 末尾的 `records.sort`），多路归并就能在 O(N log K) 时间扫一遍，相同 sig 的相邻项就是重复对，每对写一行 `(file1, doc1, file2, doc2)` 进 `BBB_WW.dups`。

```python
# 来源:dedup/minhash.py 的归并比对（注释重写版）
pq = [next(r) for r in sig_readers]; heapq.heapify(pq)
last = None
while pq:
    v = heapq.heappop(pq)
    # 关键判定：上一个签名和当前签名的 8 个 hash 完全相同 → 同一 LSH bucket
    # 注意我们在每个 bucket 内分别判断；只要 14 个 bucket 中任意 1 个命中就算重复
    if last is not None and last.sig == v.sig:
        out_f.write(struct.pack("<4I", int(last.file_stem), last.doc_id,
                                       int(v.file_stem), v.doc_id))
    last = v
    nxt = next(sig_readers[v.reader_id], None)
    if nxt: heapq.heappush(pq, nxt)
```

源码：[dedup/minhash.py:L454-L497](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/dedup/minhash.py#L454-L497)。**14 个 bucket 是天然并行的**——FineWeb 用了 `num_buckets * 50 = 700` 个 slurm task 并发跑这一阶段。

#### 5.4 (B, R) 参数与 0.85 阈值的几何含义

LSH 命中概率公式：两个文档 Jaccard 相似度为 $s$，那么至少有 1 个 bucket 完全命中的概率是

$$P(\text{flag duplicate} \mid s) = 1 - (1 - s^R)^B$$

代入 FineWeb 默认的 $B=14, R=8$：

| $s$（真实 Jaccard） | 被判重复概率 |
|----|----|
| 0.50 | 5.4% |
| 0.70 | 47% |
| 0.80 | 92.4% |
| 0.85 | **99.5%** |
| 0.90 | 99.99% |

阈值大致落在 $(1/B)^{1/R} = (1/14)^{1/8} \approx 0.72$ 附近开始急剧上升，到 0.85 几乎必然命中。**这就是「FineWeb 用 0.85 阈值近似去重」的工程含义**：不是 hard 阈值，而是「s≥0.85 几乎必删，s≤0.5 几乎必留」。注释里 datatrove 自己写了这套算式，见 [dedup/minhash.py:L30-L36](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/dedup/minhash.py#L30-L36)。

#### 5.5 阶段 3：并查集求连通分量

```python
# 来源:dedup/minhash.py 的并查集（节略，注释重写）
def parent(x):
    if x not in union_set or union_set[x] == x:
        union_set[x] = x; return x
    union_set[x] = parent(union_set[x])  # 路径压缩
    return union_set[x]

def union(a, b):
    ra, rb = parent(a), parent(b)
    if ra != rb:
        # 按 size 合并（小集合并到大集合，保持树浅）
        union_set[rb] = ra
        set_size[ra] = set_size.get(ra, 1) + set_size.get(rb, 1)
```

源码：[dedup/minhash.py:L500-L596](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/dedup/minhash.py#L500-L596)。这一步必须 `tasks=1, mem_per_cpu_gb=25`——单 dump 几亿条重复对全部进内存做并查集。FineWeb 这步官方给的是 30 小时上限。每个连通分量保留 1 个代表，其它都进 `<rank>.remove`。

#### 5.6 阶段 4：按 remove 列表过滤原文

```python
# 来源:dedup/minhash.py 的最终过滤（节略，注释重写）
next_removal = get_next()  # 读 .remove 文件里的下一个 doc 索引
for idx, doc in enumerate(data):
    if next_removal == idx:
        # 命中：丢弃 + 写到 exclusion_writer 留证
        if self.exclusion_writer: exc_writer.write(doc, rank)
        next_removal = get_next()
        continue
    yield doc  # 保留
```

源码：[dedup/minhash.py:L599-L688](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/dedup/minhash.py#L599-L688)。这一阶段的设计精妙之处：`.remove` 文件按 doc 索引升序，原文 jsonl 也是按写入顺序排的，所以是**单次顺序扫描 O(N)**，磁盘 IO 完全顺序读。

### 六、数据感：实际跑起来什么样

按 [FineWeb 论文](https://arxiv.org/abs/2406.17557) 给的数据（datatrove README 也有引用）：

- **单 dump 全量**：~90 TB WARC → ~3 TB 过滤后 → ~1 TB 去重后
- **96 个 dump（2013–2024）**：累计 ~15 T token
- **slurm 资源占用**：base_processing 8000 个 task × 10 小时 ≈ 8 万 cpu 小时；MinHash 4 阶段合计 ~5-10 万 cpu 小时
- **整个 FineWeb pipeline 一次完整跑**：以 1000 个 cpu 节点估算约 **7-14 天**

::: warning 工程经验
不要被「8000 任务并发」误导成「线性提速 8000 倍」。瓶颈往往是 **S3 list 请求限流**（datatrove 用 `randomize_start_duration=180` 给每个任务加随机启动延迟正是为此）和 **MinHash 阶段 3 的单机内存**（30 GB 起跳）。这两点是真实复现 FineWeb 时最常踩的坑。
:::

### 七、Reader、PII、Sentence Dedup：被忽视的配套件

#### 7.1 WarcReader：从 90 TB WARC 流到 Document 流

CommonCrawl 给的是 WARC（Web ARChive）格式，每个 record 包含 HTTP header + 原始 HTML。`WarcReader` 在 `read_file` 里调 `warcio.ArchiveIterator` 边解压边迭代，**永远不会把整个 WARC 文件加载进内存**：

```python
# 来源:readers/warc.py（节略，注释重写版）
def read_file(self, filepath: str):
    from warcio.archiveiterator import ArchiveIterator
    # 关键：用流式 open（fsspec 支持 s3://、gs://、本地等多种 backend）
    # 然后 ArchiveIterator 是 pull-based 的，每次 next() 才解压下一个 record
    with self.data_folder.open(filepath, "rb", compression=self.compression) as f:
        for ri, record in enumerate(ArchiveIterator(f)):
            extracted_data = process_record(record)
            # 提取 url、HTTP body、Content-Type 等放进 Document.metadata
            # 这一步还**没**做 HTML 提取，纯文本是后面 Trafilatura 的活
```

源码：[readers/warc.py:L72-L100](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/readers/warc.py#L72-L100)。`BaseDiskReader` 的 sharding 逻辑是按 `rank/world_size` 切文件列表——这就是为什么 8000 个 slurm 任务能完美并行：每个任务读 `total_files / 8000` 个 WARC 文件，互不重叠。

`JsonlReader` 是更轻量的兄弟，主要用在 MinHash 阶段 4 重新读已经过滤过的 jsonl，行为完全一致，只是 `read_file` 改成了 jsonl 逐行解析。

#### 7.2 PIIFormatter：邮箱和 IP 的最后一道闸

预训练数据里直接出现真实邮箱和 IP 是法律合规风险（GDPR、CCPA）。datatrove 在 MinHash 阶段 4 后挂了个 `PIIFormatter`，用正则把所有 `\w+@\w+\.\w+` 替换成 `email@example.com`，把 IPv4/IPv6 替换成保留地址 `22.214.171.124` / `2001:db8::1`。这一步**不是过滤**而是**改写**：算子 yield 同一篇 doc，只是 `doc.text` 被改了。

::: tip 为什么 PII 放在去重之后
反过来想：如果先 PII 改写再去重，「同一封邮件被两个网站转载、邮箱被各自打码成不同 placeholder」就会变成两篇不同的文档逃过去重。所以顺序必须是「**去重 → PII**」，FineWeb 流水线最后两步就是这么排的。
:::

#### 7.3 Sentence Dedup：MinHash 抓不到的「短篇剽窃」

MinHash 是文档级去重，但「同一段菜谱被切碎放进 10 篇不同博文里」这种**篇内片段重复**它抓不到。datatrove 提供了 `SentenceDedup` 作为补充——把每个文档切成 3 句一组的滑窗，对每组算 hash，跨文档比对找重复 span，在文档内删掉重复 span（如果删完不到 3 句就整篇丢弃）。

源码入口：[dedup/sentence_dedup.py:L41-L60](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/dedup/sentence_dedup.py#L41-L60)。这一步在 FineWeb 主线流水线里**没用**——FineWeb 的判断是「篇内片段重复」由 GopherRepetition 兜底就够了。但在 SlimPajama / Dolma 等数据集里它是核心步骤，值得知道何时该开。

### 八、与 dclm 的差异（剧透）

datatrove 走的是 **heuristic + 大规模近似去重** 路线：用十几条 Gopher/C4 规则 + MinHash 砸出干净数据。
[mlfoundations/dclm](https://github.com/mlfoundations/dclm) 走的是 **fastText 质量分类器** 路线：训一个二分类器把「OpenHermes 风格 / 高质量」从一坨 web 里捞出来。

| 维度 | datatrove (FineWeb) | dclm (DCLM-Pool) |
|------|---------------------|------------------|
| 主过滤手段 | 启发式规则 + MinHash | fastText 分类器（OpenHermes vs RW-v2） |
| 阈值可解释性 | 高（每条规则一句话） | 低（score 阈值是回归出来的） |
| 训练分类器需求 | 无 | 需要 ~400k 正负样本 |
| 对长尾领域的偏见 | 中（取决于规则） | 高（分类器有 OpenHermes 偏见） |
| 规模上限 | 30T+ token | 240T token |
| 复现成本 | CPU only，约 10 万 cpu·小时 / 96 dump | CPU + 分类器训练（GPU） |

- datatrove 优点：规则透明、debug 容易、阈值可解释；缺点：阈值需要人工拍脑袋，对非英文语言适配差。
- dclm 优点：质量分类器能捕捉「语义级好坏」；缺点：分类器本身的偏见会被放大，少数派内容（如学术论文、代码、非西方文化文本）容易被压低分数。

二者实际上是互补的——dclm 论文也用 datatrove 做了基础过滤再上分类器（先漏斗去 99% 噪声，再用分类器挑「金子」）。下一篇深度对比文章会专门拆 dclm 的 fastText 训练数据来源、score 阈值的选择，以及 DCLM-Baseline 在哪些任务上确实超过 FineWeb。

### 九、实战工程经验：拍脑袋之前先看的 Checklist

下面是从 datatrove 提交记录、issue 和 FineWeb 论文 ablation 里能挖出来的几条「真血泪」工程经验，对自己复现 web 数据流水线很有用：

1. **永远开 `exclusion_writer`**。每个过滤器都接受一个 `exclusion_writer=JsonlWriter(...)` 把被丢的文档写出来。一次 dump 跑完，看每个 reason（`gopher_short_doc` / `dup_para_frac` / ...）丢了多少、抽样看几条丢的对不对——这是唯一能让你判断阈值有没有订错的手段。
2. **`randomize_start_duration` 不是装饰**。8000 个 slurm 任务同一秒启动，对 S3 list 接口就是 8000 QPS 的雷击，AWS 限流上限通常 5500 QPS / prefix，会直接 503 雪崩。给 180 秒随机抖动后稳定到 ~44 QPS。
3. **MinHash 阶段 1 的 `skip_existing_sigs`**。这个开关让阶段 1 在重启时检测已经写好的 `.sig` 文件 + 校验大小一致就跳过——很关键，因为阶段 1 平均跑 5 小时，崩了重跑代价巨大。源码：[dedup/minhash.py:L212-L237](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/dedup/minhash.py#L212-L237)。
4. **阶段 3 的 OOM 是常态**。30 GB / cpu × 8 cpu = 240 GB 内存看起来夸张，实际单 dump 几亿条重复对是真的能撑爆。FineWeb 论文里专门提到他们试过把 `<rank>.remove` 文件分 chunk 写避免一次性持有太多 doc id 集合。
5. **PII 之前 dump 一份原始 token count**。`TokensCounter()` 算子在 MinHash 阶段 4 入口处统计去重前 token，比对去重后 token，得到「dedup ratio」。如果某个 dump 这个比例和邻居 dump 偏离太多（比如别的 dump 都是 30%，这个突然 70%），说明这个 dump 和其它有大量交叉，需要起 cross-dump dedup（FineWeb 论文 §4.4）。
6. **不要尝试自己实现 base reader 的 sharding**。`BaseDiskReader` 的实现保证了「同一个文件永远只被一个 rank 处理」（按文件名排序后取模），而且 `paths_file` 模式下还能按文件大小做 weighted balance。绕过它直接 glob 文件几乎必然导致重复或漏读。

::: warning 一次完整复现 FineWeb 单 dump 的硬件账
按 hopper-cpu 节点（96 cpu / 节点）估算：
- base_processing：8000 任务 × 10 小时 / 96 cpu = ~833 节点·小时
- MinHash 阶段 1：1000 任务 × 5 小时 = ~52 节点·小时
- MinHash 阶段 2：700 任务 × 2 小时 × 3 cpu = ~44 节点·小时
- MinHash 阶段 3：1 任务 × 30 小时 × 8 cpu = ~2.5 节点·小时
- MinHash 阶段 4：1000 任务 × 5 小时 = ~52 节点·小时

合计约 **~990 节点·小时 / 单 dump**，96 个 dump 全跑 ≈ **~9.5 万节点·小时**。这就是「FineWeb 不是个人项目」的工程含金量。
:::

### 十、把这套思路用到中文语料

datatrove 默认是英文优先的，但其实 90% 的代码是语言无关的，把它套到中文（CCI、Wudao、SkyPile）需要换的只有：

- **`LanguageFilter`**：把 `languages=["en"]` 改成 `languages=["zh"]`，并把 `backend` 切到 `glotlid`（`ft176` 对中文方言区分较粗）。
- **`GopherQualityFilter`**：`min_avg_word_length=3, max_avg_word_length=10` 完全不适用——中文 jieba 分词后平均词长 ~1.6，必须重设阈值（业界经验：min=1, max=4），并且 `STOP_WORDS` 必须换成中文停用词表（「的、是、在、和、了」）。
- **`GopherRepetitionFilter`**：n-gram 阈值不变，但 `split_into_words` 必须换成 jieba 而不是空格 split，否则整段中文会被当成单 token。
- **`MinhashDedup` 的 `simplify_text`**：默认会去标点小写，对中文还需要加全角→半角、繁→简归一化才能保证「同一篇文章繁简两版」被正确识别为重复。

::: tip 为什么这条信息重要
经常看到中文项目「直接套 datatrove 默认配置」然后过滤掉 70% 数据还以为成功了——其实是把所有非垃圾的中文都当 Gopher 异常扔了。建议第一次跑中文语料时把 `min_doc_words` 拉到 1（事实上不过滤），用 `exclusion_writer` 看看每条规则丢了什么再调阈值。
:::

## 苏格拉底时刻：5 个开放问题

1. **为什么近似去重在工程上比精确去重更可行？** 提示：精确去重要全局 hash 表（内存 O(N)），MinHash 只要每个 bucket 的局部归并（每 bucket 内存 O(N/B)）。
2. **如果你把 GopherQualityFilter 的 `min_avg_word_length` 从 3 调到 5，下游训练 loss 会怎样？** 提示：英文平均词长 4.7，调到 5 会把大量正常英文当垃圾扔掉，token 数锐减但 loss 收敛曲线不一定变好——少量更干净 vs. 多量稍噪，是 scaling law 上的真问题。
3. **fastText 语言识别在「中英混合代码注释」上为什么不可靠？** 提示：fastText 训练语料按整篇文档打标签，遇到一篇 50% 英文 + 50% 中文的文档时，predict 出的概率会非常分散（常常 en=0.45, zh=0.40），都低于 0.65 阈值被一刀切掉，但这种文档对训练 code-LLM 反而是宝贝。
4. **MinHash 的 (B=14, R=8) 阈值是 0.85，如果改成 (B=20, R=5) 会发生什么？** 提示：阈值降到 $(1/20)^{1/5} \approx 0.55$，召回率拉满但精确率崩盘——「相似但不重复」的文档（同一作者写的两篇文章）会被误删。
5. **阶段 3 的并查集为什么不能用 spark 之类的分布式实现？** 提示：union-find 的核心是路径压缩 + 按秩合并，分布式做这两步要全局通信，反而比单机一次扫慢。FineWeb 的实测：30 小时单机 vs. 估算 ~120 小时分布式（含通信开销）。

## 面试考点

1. **PipelineStep 的 `rank/world_size` 是干什么用的？** 答：每个进程的全局任务编号 + 总任务数，让算子能根据 rank 自己决定读哪一片输入文件，无需中心调度。
2. **MinHash 4 阶段为什么不能合并？** 答：1→2 受 bucket 排序约束，2→3 必须单机做并查集，3→4 是 N 个并发 filter，三处的并行度模型完全不同。
3. **LSH 的 (B, R) 参数怎么选？** 答：阈值约为 $(1/B)^{1/R}$，FineWeb 默认 $(14,8)$ 给出 0.72 起跳、0.85 必中的曲线；要更严就增大 R（曲线变陡），要更宽就增大 B（曲线左移）。
4. **过滤层级为什么要按计算成本递增排？** 答：URL 黑名单 O(1) 几乎零成本，先把垃圾干掉，最贵的 GopherRepetition / FineWebQuality 只跑剩下的几十万条文档而非几亿条。
5. **断点续跑怎么实现？** 答：每个 rank 完成后写 `completions/<rank>` 标记文件，`get_incomplete_ranks` 启动时跳过已完成的 rank。这是 30T 规模能跑下来的工程命门。

## 推荐资源

- **源码**：[huggingface/datatrove](https://github.com/huggingface/datatrove)（Apache 2.0），重点看 `src/datatrove/pipeline/base.py`、`src/datatrove/pipeline/dedup/minhash.py`、`examples/fineweb.py`
- **数据集**：[HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)、[HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- **FineWeb 论文**：[The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale (arxiv 2406.17557)](https://arxiv.org/abs/2406.17557)
- **Gopher 论文**：[Scaling Language Models: Methods, Analysis & Insights (arxiv 2112.11446)](https://arxiv.org/abs/2112.11446)
- **MinHash 综述**：[A Survey of LSH and MinHash for Web-Scale Deduplication (arxiv 2107.06499)](https://arxiv.org/abs/2107.06499)
- **C4 论文**：[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (arxiv 1910.10683)](https://arxiv.org/abs/1910.10683)
- **DCLM 论文**：[DataComp-LM: In search of the next generation of training sets (arxiv 2406.11794)](https://arxiv.org/abs/2406.11794)
- **本站前置**：[预训练数据生态宏观篇](/training/datasets) 给出从 C4 → RedPajama → FineWeb 的版本谱系；[预训练](/training/pretraining) 解释了 token 数与 scaling law 的关系，能帮你理解「为什么去重比例对 loss 影响这么大」。
- **延伸阅读**：HuggingFace 官方 [FineWeb 数据集卡片](https://huggingface.co/datasets/HuggingFaceFW/fineweb)（含 ablation 详细数字）、[FineWeb-Edu 卡片](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)（在 FineWeb 之上加了一层教育内容分类器）。
- **下一篇**：本站会出一篇 [datatrove vs dclm 深度对比]——拆 dclm 的 fastText 分类器训练流程、score 阈值的 ablation、以及为什么 DCLM-Baseline-7B 在 MMLU 上能反超 LLaMA-2-7B。
