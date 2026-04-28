---
title: 深度剖析 text-dedup——MinHash / SimHash / Suffix Array 三种去重算法
description: 同样是去重，MinHash 抓"段重复"、SimHash 抓"逐字微改"、Suffix Array 抓"长子串重复"——三种算法的本质差异与实现对比
topics: [data-engineering, deduplication, minhash, simhash, suffix-array, lsh, jaccard-similarity]
prereqs: [/training/datasets, /deep-dives/data-pipeline-datatrove]
---

# 深度剖析 text-dedup——MinHash / SimHash / Suffix Array 三种去重算法

::: info 一句话总结
预训练数据去重不是一个算法能搞定的事情：MinHash + LSH 抓「段落级近似重复」，SimHash 抓「逐字微改的同质内容」，Suffix Array 抓「跨文档共享的长样板子串」——三者用的都是「指纹 + 候选 + 验证」三段式，但指纹定义和候选生成方式完全不同，对应不同 noise 类型。
:::

> **来源声明**：本文基于 [ChenghaoMou/text-dedup](https://github.com/ChenghaoMou/text-dedup)（Apache 2.0），它是 The Pile / SlimPajama / RedPajama 等开源数据集广泛参考的去重实现，作者 Chenghao Mou 也是 BigCode 近重复去重 Spaces 的贡献者。本文阅读所有源码均来自该仓库 `main` 分支。

---

## 体系定位：去重在预训练 pipeline 中放在哪里

一条标准的预训练数据流大致是：

```
原始抓取 (CommonCrawl / 自有语料)
   ↓
语言识别 + 编码修复  ← 噪声最大、最便宜
   ↓
URL/MD5 精确去重     ← 第一道闸：把完全相同的 HTML 删掉
   ↓
质量过滤 (perplexity / classifier)
   ↓
近似去重 (MinHash / SimHash)   ← 本文重点：段落级
   ↓
Suffix Array 子串去重           ← 本文重点：跨文档长子串
   ↓
(可选) 语义去重 (embedding cluster)
   ↓
最终训练语料
```

为什么去重要放在质量过滤**之后**？因为质量过滤会扔掉大量低质文档，先过滤再去重可以把 MinHash / SimHash 的候选对数量减少一个数量级，省下的 IO 远超过滤本身的开销。但 URL / MD5 这种 O(N) 精确去重要放在最前面，因为它几乎免费。

**与姊妹文章的关系**：

- [/deep-dives/data-pipeline-datatrove](/deep-dives/data-pipeline-datatrove) 讲的是 HuggingFace `datatrove` 在百亿文档级别上**怎么把 MinHash 切成 4 个 stage 落到磁盘**——它关心的是工业落地、shuffle、partition、disk-efficient。
- 本文聚焦的是**算法层**：MinHash / SimHash / Suffix Array 各自抓什么 noise、为什么 LSH 的 `(B, R)` 参数那么调、SimHash 在中文上为什么不灵、Suffix Array 跟前两者根本是两种生物。

如果你已经读过 datatrove 那篇但被 LSH banding 的几何含义、SimHash bit-flip 与海明距离的关系、或 Suffix Array 怎么做到 5x 内存还能跑通 200 GB 语料卡住——那本文就是为你准备的。

---

## Part A——MinHash + LSH：抓段落级 Jaccard 近似

MinHash 想解决的问题：**两个文档 A 和 B 共享 80% 的 5-gram，能不能用一个固定大小的指纹快速判定？**

### A.1 从 Jaccard 到 MinHash 的直觉

把每个文档表示成一个 5-gram 集合 $S_A$。两文档相似度定义为 Jaccard：

$$
J(A, B) = \frac{|S_A \cap S_B|}{|S_A \cup S_B|}
$$

直接算 Jaccard 要求两两比较所有文档对，O(N²)，不可能。

**关键观察**（Broder 1997）：选一个把所有可能 n-gram 映射到大整数的随机哈希 h，定义 $\text{minhash}(S) = \min_{x \in S} h(x)$。则有：

$$
\Pr[\text{minhash}(S_A) = \text{minhash}(S_B)] = J(A, B)
$$

直觉理解：把所有 n-gram 排成一个由 h 决定的随机顺序，全集 $S_A \cup S_B$ 里第一个被取到的 n-gram，落在交集里的概率正好是 $|S_A \cap S_B| / |S_A \cup S_B|$。

实践上我们用 K 个独立哈希得到一个长度 K 的「签名向量」，两签名相同位置的命中比例就是 Jaccard 的无偏估计。在 text-dedup 里 K 默认设为 `num_perm = 256`，相当于每个文档用一个 64KB 的 signature 替代它的整个 n-gram 集合。

### A.2 LSH banding——把 N² 比较降成期望 O(N)

光有签名还不够：N=10⁸ 文档两两比较签名也算不动。LSH 的做法是把 256 维签名切成 `B` 个 band，每个 band 包含 `R` 行，签名两两在**同一个 band 上完全相等**就被判为「候选对」。

候选对的概率公式：

$$
P(s) = 1 - (1 - s^R)^B
$$

这是一个 S 形曲线，绕「阈值 t」处陡峭跳变。对 `num_perm=256, threshold=0.85` 求最优解大致得到 `(B, R) = (14, 8)`：

| s (真 Jaccard) | 0.5 | 0.7 | 0.8 | 0.85 | 0.9 |
|---|---|---|---|---|---|
| 候选对概率 | ~0.5% | ~7% | ~30% | ~60% | ~92% |

直观：把签名当成一组「侦测员」，每个 band 上 8 行全部完全一致才报警，14 个 band 任意一个报警就当作候选——「全一致是高门槛、多 band 是宽容差」。这就是 LSH 的几何本质：**用 OR over AND 把陡峭的阶跃函数拟合成 S 曲线**。

text-dedup 用 `datasketch` 同款的优化器在 `false_positive_weight` 与 `false_negative_weight` 之间求最优 `(B, R)`。

### A.3 数据感

| 量 | 量级 |
|---|---|
| 单文档 signature 大小 | `num_perm × hash_bits / 8` = 256 × 8 = **2048 bytes**（64-bit）；32-bit 配置下 1024 bytes |
| 1 亿文档 signature 总量 | ~200 GB（64-bit）/ ~100 GB（32-bit） |
| 单文档 band key 数 | B = 14（每 band 一个 64-byte 字符串） |
| 候选对期望（threshold=0.85） | 全语料 ~0.1%~0.5% 的文档对 |
| Union-Find 节点数 | 1 亿（int → int 的 dict，约 6~8 GB） |

> 这就是为什么 100 亿文档级别的 MinHash 一定要落盘做 sort-shuffle（datatrove 的做法），不能纯内存。

### A.4 候选 → 等价类：Union-Find

LSH 给出的是「文档 i 和 文档 j 是候选对」的二元关系。要去重，需要把所有连通的候选对合并成等价类，每类只留一个代表（一般留最小 index）。这就是 Union-Find（DSU）的活儿，复杂度近似 O(α(N)·E) ≈ O(E)。

text-dedup 在 MinHash 主流程里没有直接调 `UnionFind` 类，而是把候选对用 polars 的 `super_merger` 在 DataFrame 上做并查集（[minhash.py:L48-L79](https://github.com/ChenghaoMou/text-dedup/blob/main/src/text_dedup/minhash.py#L48-L79)）。原因很简单：MinHash 的候选对动辄上亿，纯 Python 的 dict-based UnionFind 在合并阶段会爆内存；polars 在原生 Arrow buffer 上做 group/join 几乎是 5-10x 的提速。SimHash 实现里候选对少一两个数量级，才用 Python `UnionFind`。

### A.5 后置验证：MinHash 的 false positive 怎么办

LSH 是有损的：一对真 Jaccard 只有 0.6 的文档也可能因为某个 band 巧合命中而进候选。text-dedup 的 `check_false_positives` 阶段（见 [minhash.py:L101-L170](https://github.com/ChenghaoMou/text-dedup/blob/main/src/text_dedup/minhash.py#L101-L170)）会对每对候选**真算一次 Jaccard**，低于 `threshold` 的丢弃。代价是要把候选对原文加载回来——所以这是个可选项 (`check_false_positive: bool = False`)，工业上有时候宁可多去 5% 文档也不愿付这个 IO。

> 实务建议：第一次跑全量数据时打开验证、抽样核对；之后定期跑 incremental 增量去重时关掉，省时间。

---

## Part B——SimHash：抓逐字微改的同质内容

### B.1 与 MinHash 的本质差异

**MinHash 衡量集合 Jaccard**——它把文档抽象成「无序的 n-gram 集合」，对于「整段抄过来再删几句」类型的 plagiarism 非常敏感。

**SimHash 衡量加权向量的余弦相似**——把每个 token / n-gram 看成一个向量贡献，最后投影到 64 bit fingerprint 的符号位上。它对**「逐字微改、词频几乎一致」**的同质内容（典型如 SEO 农场、模板化新闻稿）特别敏感。

直觉举例：
- 文档 A：`"how to lose weight fast"`
- 文档 B：`"how to lose weight fast and easy"`

二者 Jaccard 可能只有 0.5（对 5-gram 来说更低），MinHash 可能放过；但 SimHash 因为绝大多数 token 重复且权重相同，fingerprint 的海明距离可能只有 1~2 bit，会被 SimHash 抓住。

### B.2 SimHash 算法本身

1. 把文档拆成 n-gram（默认 `ngram_size = 3`，比 MinHash 短，这是为了让 token 频次的权重信号更密集）。
2. 对每个 n-gram 用 xxh3 算一个 f-bit 哈希（默认 f = 64）。
3. 把这个 f-bit 哈希展开成长度 f 的 ±1 向量（bit=1 → +1，bit=0 → -1），可加上权重 w（频次或 IDF）。
4. 把所有 n-gram 的向量逐位累加得到一个 f 维实数向量 v。
5. 取 `sign(v)`：`v[i] > 0 → bit=1，否则 bit=0`。这就是 64-bit SimHash fingerprint。

判断两文档相似：海明距离 `popcount(sig_A XOR sig_B) ≤ k`（默认 `bit_diff = 3`，相当于允许 64 bit 中差异不超过 ~5%）。

### B.3 LSH on SimHash——Permutation-based 桶

SimHash 想把「海明距离 ≤ k」的对从 N² 暴搜降下来，用的是另一种 LSH：把 64 bit 切成 `num_bucket = 4` 个 block，每次让 `(num_bucket - bit_diff)` 个 block 固定不动作为前缀 key，候选对里**至少有一组排列下前缀完全一致**。

- 一共生成 $\binom{4}{1} = 4$ 种排列（默认 `num_bucket=4, bit_diff=3`）。
- 每个文档在每种排列下都进一个 bucket。
- 同一 bucket 内两两比海明距离，≤ `bit_diff` 就 union。

这是经典的「鸽笼原理 LSH」——4 块切，差异最多 3 块，必然有 1 块完全一致。

### B.4 SimHash 在中文上为什么效果一般

中文没有自然词边界，`ngram_size = 3` 在 token 层面（也就是 3 个**词**）很难落在被切对的位置上。如果直接退化到 3 个**汉字**层面，token 总数膨胀十倍但权重信号变薄，反而拖慢 fingerprint 的稳定度。一种常见 workaround 是：先做分词（jieba / pkuseg）再做 SimHash；或者改用 MinHash + 字符级 5-gram，对中文更友好。

具体一点说：英文 SimHash 之所以稳定，是因为 stop word 占整篇文档 30~40% 的 token，它们在 fingerprint 上贡献了一个稳定的「语种 / 文体偏置」基底，真正区分内容的词反而是少数翻转 bit。中文同质内容里这个基底常常是「的、是、在、了」等高频词的 char，但 char 级 3-gram 切下来的「的是在」「是在了」之类几乎在任意中文里都出现，反而会让 fingerprint 趋同（高 collision），把 SimHash 退化成「所有中文文档海明距离都很近」。

---

## Part C——Suffix Array：抓跨文档的长公共子串

### C.1 与前两者根本不同

MinHash / SimHash 都是**文档级**去重——决定整篇要不要删。但有一种很常见的 noise 它们都抓不住：

> 「100 篇文档各自只有 200 字符是 GPL License 模板，其余全部是各异的内容。MinHash 觉得 Jaccard < 0.05 不重复，SimHash 海明距离很大也不重复，但这 100 篇里那段 200 字符的 License 在训练时被模型见过 100 次。」

这种「**长公共子串污染**」要靠 Suffix Array。

### C.2 算法骨架

1. 把整个语料（所有文档拼接起来，字符之间加分隔符）当成一个字节串 T，长度 N。
2. 构造后缀数组 `SA[0..N-1]`：T 所有后缀按字典序排序后的起始下标列表。构造算法 SA-IS / DC3 时间 O(N)，但常数大，实际用 Google 的 `deduplicate-text-datasets` （Rust 实现）。
3. 排序后**字典序相邻的两个后缀必然有最长 LCP**（最长公共前缀）。计算每对相邻后缀的 LCP 长度。
4. 所有 LCP ≥ K（默认 `length_threshold = 100` 字节）的位置，标记两份后缀**前 K 字节是公共子串**。
5. 把所有这种公共子串的字节区间映射回原文档（每个文档一个 byte slice list），从原文档里**切掉**这些区间。

text-dedup 的实现会把这一步外包给 Google 的 Rust 工具（`scripts/make_suffix_array.py` + `cargo run self-similar`），主程序只负责前后处理：把文档写进一个大文件、记录每个文档的 byte offset，最后根据返回的 duplicate slice 列表把每个文档对应区间挖掉。

### C.3 数据感与代价

| 量 | 量级 |
|---|---|
| 后缀数组本体 | 每个 byte 占用 ~5 byte（int40 位置 + 辅助），**5x 内存膨胀** |
| 200 GB 语料的 SA 构造 | 需要 ~1 TB 内存或外存版本 |
| 时间 | 单机 SA-IS 大约几小时；Google 版多线程 4-8 小时 |
| 阈值 K | 默认 100 字节（约 25-30 个英文 token），LLaMA / Lee et al. 论文也用 50 字节 |

正是因为代价大，Suffix Array 通常**只在 MinHash 之后跑一次**：MinHash 把文档量减掉 50% 之后再做 SA，省 50% 内存。

### C.4 哪些训练数据用过它

- **LLaMA / LLaMA-2** 训练数据 pipeline 公开提到过 Suffix Array 子串去重。
- **GPT-4 技术报告** 里有一句「we deduplicated training data with substring-level deduplication」。
- Lee et al. 2022 ([arxiv 2107.06499](https://arxiv.org/abs/2107.06499)) 是这条路线的奠基之作，附带的 Rust 工具就是 text-dedup 调用的那个。

### C.5 为什么 Suffix Array 抓住的 noise 特别值钱

这种「跨文档长样板」是模型记忆化（memorization）和 prompt injection 的高发区：模型在训练时看见同一段「Terms of Service」200 次，几乎就背下来了，evaluation 时只要给前 20 个词就能逐字续写出后面。Lee et al. 2022 把这种 substring-level 去重的影响量化过：训练数据里同一 50-byte 子串出现 ≥ 10 次的样本删除后，模型在 valid set 上的 token-level memorization rate 直接降一个数量级，下游任务无显著损伤甚至有微小提升。

换言之，前两种近重复主要解决「**多样性**」问题，Suffix Array 解决的是「**安全 + 泛化**」问题——三者解决的根本就是不同 layer 的事。

---

## 苏格拉底时刻

1. **如果 Jaccard 阈值从 0.85 降到 0.7，候选对数量会变成多少倍？训练 loss 会改善还是恶化？**
   提示：候选对数量随 threshold 下降近似指数膨胀；但低 Jaccard 的候选大量是「同主题但内容不同」，删了反而损伤多样性。

2. **MinHash 的 `(B, R)` 优化器目标是 fp_weight × FP_area + fn_weight × FN_area，它假设了 P(s) 在 [0, 1] 上均匀分布。如果你的语料 Jaccard 直方图集中在 0.95，最优 `(B, R)` 应该往哪边偏？**

3. **SimHash 的 `bit_diff` 从 3 提到 6 会发生什么？为什么？**
   提示：海明距离 ≤ 6 在 64 bit 上意味着允许 ~10% 比特翻转，候选对暴增；同时 LSH 鸽笼条件要求 `num_bucket > bit_diff`，要重新切桶。

4. **能不能把 MinHash + SimHash + Suffix Array 串起来用？顺序应该怎么排？**
   提示：先 MinHash 删段落级近似（去掉 30~50% 文档）→ 再 SimHash 抓微改（再去 5~10%）→ 最后 Suffix Array 抹长样板（不删文档但切片段）。前两步都靠近 O(N)，最后一步必须 O(N log N) 但语料已经小了。

5. **为什么 MinHash 的 `num_perm` 通常取 128 或 256，不取 1024？**
   提示：signature 大小、per-band 字符串长度、内存与精度的边际收益。

6. **如果你的语料是中英文混杂的工程教程（代码块 + 自然语言），三种方法分别需要做什么改造？**

---

## 面试考点

1. **MinHash 期望与 Jaccard 相等的证明（直觉版能讲清楚就够）**：「全集随机排序、第一个被取到的元素落在交集的概率」。

2. **`num_perm = 128 / 256` 而不是更高的几何理由**：MinHash 估计 Jaccard 的方差是 $J(1-J)/K$；K 从 256 翻到 1024 只让标准差从 √(0.85×0.15/256)≈0.022 降到 0.011，但 signature / 内存 / IO / per-band-key-长度全部翻 4 倍。**收益亚线性，成本线性**。

3. **`(B, R)` 调参的几何理解**：S 曲线在 $s = (1/B)^{1/R}$ 处过 0.5。固定 num_perm=B×R 时，B 大让曲线平缓（更宽容、更多候选），R 大让曲线陡峭（更严格、更少候选）。**threshold 高就让 R 大；threshold 低就让 B 大**。

4. **SimHash 在工业搜索（Google News 去重）vs 在 LLM 数据去重中的差异**：搜索场景文档短、词频信号干净，bit_diff = 3 / 64 已足；LLM 数据文档长、噪声多，更多人选 MinHash + 字符级 n-gram。

5. **Suffix Array 为什么必须用外存或专门的 Rust 实现**：5x 内存膨胀 + LCP 数组本身，纯 Python 在 GB 级就崩。Google 的 Rust 实现做了块化构造与磁盘分片。

---

## 推荐资源

- **源仓库**：[ChenghaoMou/text-dedup](https://github.com/ChenghaoMou/text-dedup) — Apache 2.0，本文所有代码引用都来自这里。
- **MinHash 原始论文**：Andrei Broder, *On the Resemblance and Containment of Documents*, 1997。SEQUENCES'97 会议论文。
- **SimHash 原始论文**：Moses Charikar, *Similarity Estimation Techniques from Rounding Algorithms*, STOC 2002。
- **Suffix Array LLM 去重奠基论文**：Lee et al., *Deduplicating Training Data Makes Language Models Better*, ACL 2022。[arxiv.org/abs/2107.06499](https://arxiv.org/abs/2107.06499)
- **BigCode 近重复实战 Spaces**：[huggingface.co/spaces/bigcode/near-deduplication](https://huggingface.co/spaces/bigcode/near-deduplication)（含 `(B, R)` 交互式 demo）。
- **Google 官方 Rust SA 工具**：text-dedup 调用的就是它的 self-similar 命令。

---

## 手撕源码

### 片段 1——MinHash signature 的核心计算

下面这段是 MinHash 算法的「心脏」：把 n-gram 集合的初始哈希通过 `(a·x + b) mod p` 这种 universal hashing 派生出 `num_perm` 个独立伪哈希，然后逐列取 min。

```python
# 引自 minhash.py 配置类的 get_embed_func
# 见: https://github.com/ChenghaoMou/text-dedup/blob/main/src/text_dedup/config/algorithms/minhash.py#L200-L238

# tokens 是一个文档去重后的 n-gram 字节集合
# hash_func 默认 xxh3，先把每个 n-gram 哈希成一个 base 哈希
hashvalues = np.array([hash_func(t) for t in tokens], dtype=dtype).reshape(len(tokens), 1)

# 关键：universal hashing 派生 num_perm 个伪独立哈希
# a, b 是 num_perm 维向量，p 是 mersenne prime
# 一行就把 K 个 hash_func 替代了——这是 MinHash 工程实现的标志
hashvalues = (hashvalues * a + b) % modulo_prime & max_hash

# masks 全部填 max_hash 作为「无穷大」初始值
masks = np.full(shape=num_perm, dtype=dtype, fill_value=max_hash)
# vstack + min(axis=0) 沿 num_perm 维取列最小——这才是 minhash 名字的由来
hashvalues = np.vstack([hashvalues, masks]).min(axis=0)
# hashvalues 现在是一条长度 num_perm 的 signature 向量
```

为什么这样写而不是循环 `num_perm` 次重新哈希？因为 xxh3 调用 256 次比 numpy 在 ndarray 上做一次 `(a·x + b)` 慢一个数量级以上，universal hashing 的本质就是「**用一次硬哈希 + K 次廉价仿射变换** 替代 K 次硬哈希」。

### 片段 2——LSH banding：把 signature 切成 (B, R) 桶

```python
# 引自 minhash.py 配置类
# 见: https://github.com/ChenghaoMou/text-dedup/blob/main/src/text_dedup/config/algorithms/minhash.py#L148-L153
# 以及 https://github.com/ChenghaoMou/text-dedup/blob/main/src/text_dedup/config/algorithms/minhash.py#L232-L236

# bands * rows = num_perm
# 比如 num_perm=256, bands=14, rows=8 → 14 个 band，每 band 8 个连续 hash
hash_ranges = [(i * rows, (i + 1) * rows) for i in range(bands)]

# 把 signature 按 band 切片，每个 band 转成一个 bytes 作为 key
# 同一 band 上 bytes 完全相等的两个文档就是 LSH 候选对
return {
    "__band_idx__": list(range(len(hash_ranges))),
    # byteswap 只是历史 backward-compat，正确性不依赖它
    "__band_val__": [bytes(sig[s:e].byteswap().data) for (s, e) in hash_ranges],
    "id": [idx] * len(hash_ranges),
}

# 上层用 polars 的 group_by(['__band_idx__', '__band_val__']).agg(id) 找候选对
# 效果等价于「同一 band 内 bytes 完全相同」的文档归为一个候选 cluster
```

这种「(band_idx, band_val) → list of doc_id」的 group-by 策略是 text-dedup 的工业关键：用一次 polars groupby 替代了所有手写 hash table，可以直接 spill 到磁盘。

### 片段 3——SimHash fingerprint：±1 投票后取符号

```python
# 引自 simhash.py 的 compute()
# 见: https://github.com/ChenghaoMou/text-dedup/blob/main/src/text_dedup/config/algorithms/simhash.py#L222-L253

def compute(hashes: list[bitarray]) -> bitarray:
    # hashes 是文档所有 n-gram 各自的 f-bit hash 列表
    # 把每个 hash 展开成 0/1 向量
    sigs = np.asarray([h.tolist() for h in hashes], dtype=int)
    # 关键：2*sigs - 1 把 0/1 转 -1/+1（这就是 SimHash 的「投票」）
    # 沿 axis=0 累加所有 n-gram 的投票，最后取符号
    sig = np.where(np.sum(2 * sigs - 1, axis=0) > 0, True, False)
    res = bitarray()
    res.extend(sig.tolist())
    return res

# 物理含义：
# - 每个 n-gram 在 64 个 bit 位上各投一票（+1 或 -1）
# - 整篇文档加权累加得到 64 维实数向量
# - 取符号位 → 64-bit fingerprint
# - 海明距离衡量「多少个 bit 位投票翻向」≈ 余弦相似度的离散版本
```

注意作者在注释里特意提了「Cython / numpy operator 都试过没快」——这是 SimHash 的实现瓶颈：每个 n-gram 都要展开 64 个 bit 然后做 column sum，CPython 的 `bitarray.tolist()` 已经是合理上限。

### 片段 4——Union-Find：把候选对压成等价类

```python
# 引自 utils/union_find.py
# 见: https://github.com/ChenghaoMou/text-dedup/blob/main/src/text_dedup/utils/union_find.py#L17-L65

def find(self, x: int) -> int:
    # 路径压缩：让 x 一步走到根
    if x not in self.parent:
        self.parent[x] = x
        self.rank[x] = 0
        return x
    if self.parent[x] != x:
        self.parent[x] = self.find(self.parent[x])  # 压缩
    return self.parent[x]

def union(self, x: int, y: int) -> None:
    rx, ry = self.find(x), self.find(y)
    if rx == ry:
        return
    # 按 rank 合并：低树挂高树下面，避免链化
    if self.rank[rx] < self.rank[ry]:
        self.parent[rx] = ry
    elif self.rank[rx] > self.rank[ry]:
        self.parent[ry] = rx
    else:
        self.parent[ry] = rx
        self.rank[rx] += 1
```

工业上 100M 文档的 Union-Find 不会放内存——MinHash 实现里 text-dedup 改用 polars 的 `super_merger` 在 DataFrame 上做等价类合并（见 [minhash.py:L48-L79](https://github.com/ChenghaoMou/text-dedup/blob/main/src/text_dedup/minhash.py#L48-L79)）；这里这份纯 Python UnionFind 是 SimHash 用的，因为 SimHash 候选对量级一般小一两个数量级。**算法选实现，永远跟着数据规模走**。

---

## 三套配置怎么读：数值背后的 trade-off

text-dedup 的 README 给的三套默认配置，回头看其实把每个参数的「位置」交代得很清楚：

**MinHash 默认**（[README 配置](https://github.com/ChenghaoMou/text-dedup/blob/main/README.md)）：

```
num_perm = 240, threshold = 0.7, ngram_size = 5
hash_bits = 64, check_false_positive = true
```

- `num_perm = 240` 而不是 256，是为了让 `(B, R)` 的整除关系在常见 threshold 下都能落到整数；`240 = 2^4 × 3 × 5` 因子多，优化器可枚举的 `(B, R)` 组合更密。
- `threshold = 0.7` 偏激进，配合 `check_false_positive = true` 走二次验证——「**先放大候选池，再用真 Jaccard 过滤**」是工业标配。
- `ngram_size = 5` 是字符级 5（先经 `\W` 切 token），抓得住短语级抄袭。

**SimHash 默认**：

```
hash_bits = 64, ngram_size = 3, bit_diff = 3, num_bucket = 4
```

`num_bucket - bit_diff = 1` 卡在最小可行值——再降一档（比如 num_bucket=4, bit_diff=4）鸽笼条件就破了，算法层面崩。这意味着 SimHash 的容差几乎已经摸到了 LSH 桶能承受的上限。如果你的语料确实需要更宽容的相似度（比如 bit_diff=6），必须把 `num_bucket` 提到 8 或更高，相应地排列数 / 候选量爆炸。

**Suffix Array 默认**：

```
length_threshold = 100 (bytes), merge_strategy = "longest"
```

`length_threshold = 100 byte` 大致对应 25-30 个英文 token 或 30-50 个汉字，这是 Lee et al. 2022 经验值。`merge_strategy = "longest"` 表示遇到嵌套区间时只保留最长那个、不强行合并——它对「License + URL + License」这种模板化内容更友好（不会把中间散文也连带切掉）。换 `"overlapping"` 适合 boilerplate 极重的网页。

---

## 一图收尾：三种算法的对照表

| 维度 | MinHash + LSH | SimHash | Suffix Array |
|---|---|---|---|
| 衡量什么 | 集合 Jaccard | 加权向量余弦 | 精确公共子串 |
| 抓哪种 noise | 段落级近似复制 | 逐字微改、模板化内容 | 跨文档长样板（License、引文、boilerplate） |
| 文档级 / 子串级 | 文档级 | 文档级 | **子串级**（不删文档，只删片段） |
| 时间复杂度 | 期望 ~O(N)（带 LSH） | 期望 ~O(N) | O(N log N)，常数巨大 |
| 内存代价 | signature 2 KB / 文档 | fingerprint 8 byte / 文档 | **5x 语料字节** |
| 推荐 n-gram | 5（字符级 5）or 5 词 | 3 词 | 不需要 n-gram，按 byte |
| 关键超参 | `num_perm`, `(B, R)`, `threshold` | `bit_diff`, `num_bucket` | `length_threshold` (字节) |
| 工业代表 | The Pile, RedPajama, SlimPajama | Google News 去重 | LLaMA, GPT-4 训练数据 |

把它们当成「不同 noise 的对应工具」而不是「谁更强」——这是这篇文章最想传达的一点。
