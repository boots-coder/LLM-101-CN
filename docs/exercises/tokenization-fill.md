---
title: "Tokenization 分词填空"
description: "Level 1-3 填空：BPE 训练、WordPiece、SentencePiece、中文分词挑战"
topics: [fill-in, tokenization, BPE, WordPiece, SentencePiece, vocabulary]
---
# Tokenization 分词填空 (Level 2-3)

> 本练习覆盖 Tokenization 核心技术：BPE 训练与编码、WordPiece 对比、中文分词挑战、完整 Tokenizer 组装。
> 代码基于纯 Python 实现，用 `_____` 标记需要填写的部分。

::: info 前置知识
- Python 基础（字典、字符串操作、排序）
- 了解 UTF-8 编码原理
- 阅读过 [Tokenization](/architecture/tokenization.md)
:::

---

## 练习 1: BPE 训练算法（Level 2）

### 背景

BPE (Byte Pair Encoding) 是最主流的子词分词算法，被 GPT、LLaMA 等模型采用。训练过程：从字符级词表开始，反复统计相邻 pair 频率，合并最高频 pair 为新 token，直到词表达到目标大小。

本练习用小语料模拟 BPE 训练。每个单词预先拆为字符序列，末尾加 `</w>` 标记词尾。

### 任务

```python
def get_pair_freqs(vocab):
    """统计词表中所有相邻 pair 的频率
    vocab: dict, 键为 token 元组, 值为出现频率
    返回: pair_freqs dict, 键为 pair 元组, 值为频率之和
    """
    pair_freqs = {}
    for tokens, freq in vocab.items():
        for i in range(len(tokens) - 1):
            # ===== 填空 1: 构造相邻 pair 并累加频率 =====
            pair = _____
            pair_freqs[pair] = pair_freqs.get(pair, 0) + _____
    return pair_freqs


def merge_pair(vocab, pair):
    """在词表中执行一次合并操作"""
    new_vocab = {}
    for tokens, freq in vocab.items():
        new_tokens = []
        i = 0
        while i < len(tokens):
            # ===== 填空 2: 匹配 pair 则合并，否则保留原 token =====
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(_____)
                i += _____
            else:
                new_tokens.append(tokens[i])
                i += 1
        new_vocab[tuple(new_tokens)] = freq
    return new_vocab


def bpe_train(vocab, num_merges):
    """BPE 训练主循环，返回 (vocab, merges)"""
    merges = []
    for step in range(num_merges):
        pair_freqs = get_pair_freqs(vocab)
        if not pair_freqs:
            break
        # ===== 填空 3: 找到频率最高的 pair =====
        best_pair = _____
        # ===== 填空 4: 执行合并并记录规则 =====
        vocab = _____
        merges.append(best_pair)
        print(f"Step {step + 1}: 合并 {best_pair} -> '{''.join(best_pair)}'")
    return vocab, merges
```

### 提示

- 填空 1：`(tokens[i], tokens[i+1])`，累加 `freq`
- 填空 2：拼接 `pair[0] + pair[1]`，跳过 2 个位置
- 填空 3：`max(pair_freqs, key=pair_freqs.get)`
- 填空 4：`merge_pair(vocab, best_pair)`

<details>
<summary>参考答案</summary>

```python
# 填空 1
pair = (tokens[i], tokens[i + 1])
pair_freqs[pair] = pair_freqs.get(pair, 0) + freq

# 填空 2
new_tokens.append(pair[0] + pair[1])
i += 2

# 填空 3
best_pair = max(pair_freqs, key=pair_freqs.get)

# 填空 4
vocab = merge_pair(vocab, best_pair)
```

**验证:**
```python
corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
vocab = {tuple(list(w) + ["</w>"]): f for w, f in corpus.items()}

print("初始词表:")
for tokens, freq in vocab.items():
    print(f"  {' '.join(tokens)} : {freq}")

vocab, merges = bpe_train(vocab, num_merges=10)

print(f"\n合并规则 ({len(merges)} 条):")
for i, pair in enumerate(merges):
    print(f"  {i+1}. {pair[0]} + {pair[1]} -> {''.join(pair)}")

# ('e','s') 频率 = 6+3 = 9，应为第一步合并
assert merges[0] == ('e', 's'), f"第一步应合并 ('e','s'), 实际: {merges[0]}"
print("\n验证通过!")
```

**解析：** Pair 频率 = 所有包含该 pair 的单词频率之和。`('e','s')` 出现在 newest(6) 和 widest(3) 中，总频率 9 最高，因此第一步合并。BPE 的核心就是贪心选频率最高的 pair。

</details>

---

## 练习 2: BPE 编码（分词）（Level 2）

### 背景

训练完成后得到有序的合并规则。编码新文本时：将输入拆为字符序列，按训练时的合并顺序依次检查并合并匹配的 pair。训练时越早合并的 pair，编码时优先级越高。

### 任务

```python
def bpe_encode(text, merges):
    """用 BPE 合并规则对单个单词编码，返回 token 列表"""
    # ===== 填空 1: 将文本拆为字符列表，末尾加 '</w>' =====
    tokens = _____

    for pair in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            # ===== 填空 2: 匹配 pair 则合并，否则保留 =====
            if i < len(tokens) - 1 and _____:
                new_tokens.append(_____)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens


def bpe_tokenize(text, merges):
    """对句子分词：按空格拆分为单词，分别编码"""
    all_tokens = []
    # ===== 填空 3: 按空格拆分，对每个单词调用 bpe_encode =====
    for word in _____:
        word_tokens = _____
        all_tokens.extend(word_tokens)
    return all_tokens
```

### 提示

- 填空 1：`list(text) + ["</w>"]`
- 填空 2：条件 `tokens[i] == pair[0] and tokens[i+1] == pair[1]`，合并结果 `pair[0] + pair[1]`
- 填空 3：`text.split()` 拆分，`bpe_encode(word, merges)` 编码

<details>
<summary>参考答案</summary>

```python
# 填空 1
tokens = list(text) + ["</w>"]

# 填空 2
if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
    new_tokens.append(pair[0] + pair[1])

# 填空 3
for word in text.split():
    word_tokens = bpe_encode(word, merges)
```

**验证:**
```python
test_merges = [
    ('e', 's'), ('es', 't'), ('est', '</w>'),
    ('l', 'o'), ('lo', 'w'),
    ('n', 'e'), ('ne', 'w'), ('new', 'est</w>'),
]

print(f"'lowest' -> {bpe_encode('lowest', test_merges)}")
# 预期: ['low', 'est</w>']

print(f"'newer'  -> {bpe_encode('newer', test_merges)}")
# 预期: ['new', 'e', 'r', '</w>']

print(f"'tax'    -> {bpe_encode('tax', test_merges)}")
# 预期: ['t', 'a', 'x', '</w>']（无匹配规则，回退字符级）

print(f"'low newest' -> {bpe_tokenize('low newest', test_merges)}")
print("\n验证通过!")
```

**解析：** BPE 不存在真正的 OOV -- 最坏情况下回退到字符级。`</w>` 标记区分独立词和前缀（"low" vs "lower" 中的 "low"）。合并规则的顺序决定了编码结果。

</details>

---

## 练习 3: WordPiece vs BPE 对比（Level 2-3）

### 背景

WordPiece (BERT 使用) 与 BPE 的区别在于合并策略：BPE 选频率最高的 pair，WordPiece 选"似然增益"最大的 pair：

$$\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \times \text{freq}(b)}$$

直觉上，WordPiece 偏好"共现概率远高于独立出现概率"的 pair，类似 PMI (Pointwise Mutual Information)。即使 `(t, h)` 频率高，但 `t` 和 `h` 各自也高频，score 可能不如低频但强关联的 pair。

### 任务

```python
def get_token_freqs(vocab):
    """统计每个 token 的独立出现频率"""
    token_freqs = {}
    for tokens, freq in vocab.items():
        for token in tokens:
            # ===== 填空 1: 累加频率 =====
            token_freqs[token] = _____
    return token_freqs


def wordpiece_score(pair, pair_freq, token_freqs):
    """计算 WordPiece 合并得分"""
    # ===== 填空 2: score = freq(pair) / (freq(a) * freq(b)) =====
    score = _____
    return score


def compare_bpe_wordpiece(vocab):
    """对比 BPE 和 WordPiece 选择的第一个合并 pair"""
    pair_freqs = get_pair_freqs(vocab)  # 复用练习 1
    token_freqs = get_token_freqs(vocab)

    bpe_ranking = sorted(pair_freqs.items(), key=lambda x: -x[1])

    wp_scores = {}
    for pair, freq in pair_freqs.items():
        # ===== 填空 3: 计算 WordPiece 得分 =====
        wp_scores[pair] = _____
    wp_ranking = sorted(wp_scores.items(), key=lambda x: -x[1])

    print("BPE Top-5 (按频率):")
    for pair, freq in bpe_ranking[:5]:
        print(f"  {pair}: freq={freq}")
    print("\nWordPiece Top-5 (按似然增益):")
    for pair, score in wp_ranking[:5]:
        print(f"  {pair}: score={score:.4f} (freq={pair_freqs[pair]})")

    # ===== 填空 4: 提取最优 pair =====
    bpe_best = _____
    wp_best = _____
    print(f"\nBPE: {bpe_best}, WordPiece: {wp_best}, 一致: {bpe_best == wp_best}")
    return bpe_best, wp_best
```

### 提示

- 填空 1：`token_freqs.get(token, 0) + freq`
- 填空 2：`pair_freq / (token_freqs[pair[0]] * token_freqs[pair[1]])`
- 填空 3：`wordpiece_score(pair, freq, token_freqs)`
- 填空 4：`bpe_ranking[0][0]` 和 `wp_ranking[0][0]`

<details>
<summary>参考答案</summary>

```python
# 填空 1
token_freqs[token] = token_freqs.get(token, 0) + freq

# 填空 2
score = pair_freq / (token_freqs[pair[0]] * token_freqs[pair[1]])

# 填空 3
wp_scores[pair] = wordpiece_score(pair, freq, token_freqs)

# 填空 4
bpe_best = bpe_ranking[0][0]
wp_best = wp_ranking[0][0]
```

**验证:**
```python
corpus = {"the": 10, "there": 3, "that": 5, "thin": 2, "cat": 7, "car": 4}
vocab = {tuple(list(w) + ["</w>"]): f for w, f in corpus.items()}

bpe_best, wp_best = compare_bpe_wordpiece(vocab)

# 分析 ('t','h') 的差异
tf = get_token_freqs(vocab)
pf = get_pair_freqs(vocab)
th = pf.get(('t', 'h'), 0)
print(f"\n('t','h'): pair_freq={th}, t_freq={tf['t']}, h_freq={tf['h']}")
print(f"  BPE score = {th}, WordPiece score = {th/(tf['t']*tf['h']):.4f}")
print("\n验证通过!")
```

**解析：** BPE 偏好高频 pair；WordPiece 偏好"惊喜度"高的 pair（共现远超随机预期）。实际效果上，WordPiece 倾向合并强绑定组合（如英语中 "qu" 几乎总共现），BPE 可能先合并常见但松散的组合。

</details>

---

## 练习 4: 中文分词的特殊挑战（Level 2）

### 背景

中文没有空格分隔，GPT 系列使用 byte-level BPE 在 UTF-8 字节层面操作。一个中文字符占 3 个 UTF-8 字节，ASCII 字符仅 1 字节。这导致中文需要显著更多 token，带来更高推理成本和更短有效上下文窗口。

ChatGLM 和 Qwen 等中文优化模型通过增加中文训练语料比例、在词表中预置高频汉字和词组来解决这一问题。

### 任务

```python
def text_to_utf8_bytes(text):
    """将文本转为 UTF-8 字节序列 (int 列表)"""
    # ===== 填空 1: 编码为 UTF-8 =====
    byte_sequence = _____
    # ===== 填空 2: 转为 int 列表 =====
    byte_list = _____
    return byte_list


def analyze_byte_distribution(text):
    """分析每个字符的 UTF-8 字节数"""
    char_bytes = []
    for char in text:
        # ===== 填空 3: 计算单个字符的字节数 =====
        n_bytes = _____
        char_bytes.append((char, n_bytes))
    return char_bytes


def compare_token_efficiency(texts, labels):
    """对比不同语言文本的 byte-level token 效率"""
    print(f"{'标签':<12} {'字符数':<8} {'字节数':<8} {'字节/字符':<10}")
    print("-" * 42)
    for text, label in zip(texts, labels):
        n_chars = len(text)
        # ===== 填空 4: 计算字节数 =====
        n_bytes = _____
        ratio = n_bytes / n_chars if n_chars > 0 else 0
        print(f"{label:<12} {n_chars:<8} {n_bytes:<8} {ratio:<10.2f}")


def simulate_byte_bpe_vocab(chinese_chars, target_vocab_size=500):
    """模拟在中文上构建 byte-level 词表"""
    vocab = set(range(256))  # 基础 byte 词表

    char_byte_map = {}
    for char in set(chinese_chars):
        byte_seq = tuple(char.encode("utf-8"))
        # ===== 填空 5: 存入映射 =====
        char_byte_map[char] = _____

    from collections import Counter
    char_freq = Counter(chinese_chars)
    sorted_chars = sorted(char_freq.items(), key=lambda x: -x[1])

    added = 0
    for char, freq in sorted_chars:
        if len(vocab) >= target_vocab_size:
            break
        vocab.add(char_byte_map[char])
        added += 1

    print(f"基础 byte 词表: 256, 添加中文: {added}, 总计: {len(vocab)}")
    return vocab
```

### 提示

- 填空 1：`text.encode("utf-8")`
- 填空 2：`list(byte_sequence)`
- 填空 3：`len(char.encode("utf-8"))`
- 填空 4：`len(text.encode("utf-8"))`
- 填空 5：`byte_seq`

<details>
<summary>参考答案</summary>

```python
# 填空 1
byte_sequence = text.encode("utf-8")
# 填空 2
byte_list = list(byte_sequence)
# 填空 3
n_bytes = len(char.encode("utf-8"))
# 填空 4
n_bytes = len(text.encode("utf-8"))
# 填空 5
char_byte_map[char] = byte_seq
```

**验证:**
```python
# UTF-8 字节分析
test_text = "Hello你好"
print(f"'{test_text}' -> {text_to_utf8_bytes(test_text)}")
for char, n in analyze_byte_distribution(test_text):
    print(f"  '{char}': {n} 字节")

# 效率对比
compare_token_efficiency(
    ["The large language model is powerful", "大型语言模型非常强大", "LLM 大型语言模型"],
    ["English", "Chinese", "Mixed"],
)

# 词表模拟
simulate_byte_bpe_vocab("大型语言模型是人工智能领域的重要突破", 300)

assert len(text_to_utf8_bytes("A")) == 1 and len(text_to_utf8_bytes("中")) == 3
print("\n验证通过!")
```

**解析：** ASCII 1 字节 vs 中文 3 字节，导致中文在 GPT tokenizer 下通常需要 1.5-2 倍于英文的 token 数。Qwen 词表含 151,643 个 token（大量中文 token），有效降低了中文 token 消耗。

</details>

---

## 练习 5: 完整 Tokenizer 实现（Level 3）

### 背景

将 BPE 训练、编码、解码和特殊 token 处理组装为完整 Tokenizer 类。关键要求：(1) 特殊 token 不参与 BPE 合并，直接映射固定 ID；(2) encode/decode 满足往返一致性 -- `decode(encode(text)) == text`。

### 任务

```python
class BPETokenizer:
    def __init__(self, special_tokens=None):
        self.merges = []
        self.vocab = {}            # token -> id
        self.inverse_vocab = {}    # id -> token
        self.special_tokens = special_tokens or []

    def _build_vocab(self):
        """根据合并规则构建词表"""
        self.vocab = {}
        idx = 0
        # ===== 填空 1: 特殊 token 优先加入词表 =====
        for token in self.special_tokens:
            self.vocab[token] = _____
            idx += 1

        # 基础字符 (可打印 ASCII + </w>)
        for ch in [chr(i) for i in range(32, 127)] + ["</w>"]:
            if ch not in self.vocab:
                self.vocab[ch] = idx
                idx += 1

        # ===== 填空 2: 合并规则产生的新 token 加入词表 =====
        for pair in self.merges:
            merged_token = _____
            if merged_token not in self.vocab:
                self.vocab[merged_token] = idx
                idx += 1

        # ===== 填空 3: 构建反向映射 =====
        self.inverse_vocab = _____

    def train(self, corpus, num_merges):
        """在语料上训练 BPE。corpus: {word: freq}"""
        word_vocab = {tuple(list(w) + ["</w>"]): f for w, f in corpus.items()}

        self.merges = []
        for step in range(num_merges):
            pair_freqs = {}
            for tokens, freq in word_vocab.items():
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
            if not pair_freqs:
                break

            # ===== 填空 4: 找最优 pair =====
            best_pair = _____

            new_word_vocab = {}
            for tokens, freq in word_vocab.items():
                new_tokens, i = [], 0
                while i < len(tokens):
                    if (i < len(tokens) - 1
                        and tokens[i] == best_pair[0]
                        and tokens[i + 1] == best_pair[1]):
                        new_tokens.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                new_word_vocab[tuple(new_tokens)] = freq
            word_vocab = new_word_vocab
            self.merges.append(best_pair)

        self._build_vocab()
        print(f"训练完成: {len(self.merges)} 条规则, 词表 {len(self.vocab)}")

    def encode(self, text):
        """将文本编码为 token ID 列表"""
        # ===== 填空 5: 处理特殊 token（分割 + 递归编码） =====
        for sp_token in self.special_tokens:
            if sp_token in text:
                parts = text.split(sp_token)
                ids = []
                for i, part in enumerate(parts):
                    if part:
                        ids.extend(self.encode(part))
                    if i < len(parts) - 1:
                        ids.append(_____)
                return ids

        all_ids = []
        for word in text.split():
            tokens = list(word) + ["</w>"]
            for pair in self.merges:
                new_tokens, i = [], 0
                while i < len(tokens):
                    if (i < len(tokens) - 1
                        and tokens[i] == pair[0]
                        and tokens[i + 1] == pair[1]):
                        new_tokens.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            # ===== 填空 6: token 转 ID =====
            for token in tokens:
                if token in self.vocab:
                    all_ids.append(_____)
                else:
                    all_ids.append(self.vocab.get("<unk>", -1))
        return all_ids

    def decode(self, ids):
        """将 token ID 列表解码为文本"""
        tokens = []
        for id_ in ids:
            # ===== 填空 7: ID 转 token =====
            token = _____
            tokens.append(token)

        text = ""
        for token in tokens:
            if token in self.special_tokens:
                text += token
            # ===== 填空 8: 处理 </w> 词尾标记 =====
            elif token.endswith("</w>"):
                text += _____ + " "
            else:
                text += token
        return text.strip()
```

### 提示

- 填空 1：`idx`（当前索引）
- 填空 2：`pair[0] + pair[1]`
- 填空 3：`{v: k for k, v in self.vocab.items()}`
- 填空 4：`max(pair_freqs, key=pair_freqs.get)`
- 填空 5：`self.vocab[sp_token]`
- 填空 6：`self.vocab[token]`
- 填空 7：`self.inverse_vocab.get(id_, "<unk>")`
- 填空 8：`token[:-len("</w>")]`（去掉词尾标记）

<details>
<summary>参考答案</summary>

```python
# 填空 1
self.vocab[token] = idx
# 填空 2
merged_token = pair[0] + pair[1]
# 填空 3
self.inverse_vocab = {v: k for k, v in self.vocab.items()}
# 填空 4
best_pair = max(pair_freqs, key=pair_freqs.get)
# 填空 5
ids.append(self.vocab[sp_token])
# 填空 6
all_ids.append(self.vocab[token])
# 填空 7
token = self.inverse_vocab.get(id_, "<unk>")
# 填空 8
text += token[:-len("</w>")] + " "
```

**验证:**
```python
corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3, "new": 4}
tokenizer = BPETokenizer(special_tokens=["<|endoftext|>", "<pad>", "<unk>"])
tokenizer.train(corpus, num_merges=10)

# 往返一致性测试
for text in ["low", "newest", "lower"]:
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    print(f"'{text}' -> {ids} -> '{decoded}'")
    assert decoded == text, f"往返不一致! '{text}' -> '{decoded}'"

# 特殊 token 测试
sp_text = "low<|endoftext|>new"
ids = tokenizer.encode(sp_text)
decoded = tokenizer.decode(ids)
print(f"'{sp_text}' -> {ids} -> '{decoded}'")
assert "<|endoftext|>" in decoded

assert tokenizer.vocab["<pad>"] == 1
print("\n所有验证通过! encode -> decode 往返一致性成立。")
```

**解析：**

1. **词表构建顺序**：特殊 token (ID 0,1,2...) -> 基础字符 -> 合并子词，保证 ID 分配确定性。
2. **往返一致性**：编码追加 `</w>` 标记词尾，解码时将 `</w>` 替换为空格恢复词间分隔。
3. **与生产级的差距**：tiktoken / HuggingFace tokenizers 还需处理 Unicode 规范化、预分词正则、字节级回退等，本练习只涉及核心逻辑。

</details>

---

## MLM 代码训练模式

完成上面的固定填空后，试试随机挖空模式 -- 每次点击「刷新」会随机遮盖不同的代码片段，帮你彻底记住每一行。

### BPE 训练核心循环

<CodeMasker title="BPE pair 统计与合并" :mask-ratio="0.15">
def get_pair_freqs(vocab):
    pair_freqs = {}
    for tokens, freq in vocab.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs

def merge_pair(vocab, pair):
    new_vocab = {}
    for tokens, freq in vocab.items():
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        new_vocab[tuple(new_tokens)] = freq
    return new_vocab

def bpe_train(vocab, num_merges):
    merges = []
    for step in range(num_merges):
        pair_freqs = get_pair_freqs(vocab)
        if not pair_freqs:
            break
        best_pair = max(pair_freqs, key=pair_freqs.get)
        vocab = merge_pair(vocab, best_pair)
        merges.append(best_pair)
    return vocab, merges
</CodeMasker>

### BPE 编码（分词）

<CodeMasker title="BPE encode 按合并规则编码" :mask-ratio="0.15">
def bpe_encode(text, merges):
    tokens = list(text) + ["</w>"]
    for pair in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
</CodeMasker>

### WordPiece 似然增益得分

<CodeMasker title="WordPiece score 计算" :mask-ratio="0.15">
def wordpiece_score(pair, pair_freq, token_freqs):
    score = pair_freq / (token_freqs[pair[0]] * token_freqs[pair[1]])
    return score

def get_token_freqs(vocab):
    token_freqs = {}
    for tokens, freq in vocab.items():
        for token in tokens:
            token_freqs[token] = token_freqs.get(token, 0) + freq
    return token_freqs
</CodeMasker>
