---
title: "数据集构建"
description: "预训练数据清洗、SFT 数据构建、偏好数据构建、MinHash 去重"
topics: [dataset, data-cleaning, MinHash, deduplication, chat-template, synthetic-data, quality-filtering]
---
# 数据集构建

> 高质量数据集是模型能力的上限，模型只能学到数据中包含的知识。数据工程往往是 LLM 项目中投入产出比最高的环节。

## 在大模型体系中的位置

```
数据集构建（本章）
    │
    ├── 预训练数据 ──> 预训练阶段（万亿 token 无标注文本）
    │
    ├── SFT 数据 ──> 监督微调阶段（万级~十万级指令-回答对）
    │
    └── 偏好数据 ──> RLHF/DPO 对齐阶段（万级 chosen/rejected 对）
```

数据贯穿整个 LLM 训练流水线。每个阶段对数据的规模、格式和质量要求截然不同。

---

## 预训练数据 vs 后训练数据

| 维度 | 预训练数据 | SFT 数据 | 偏好数据 |
|------|-----------|---------|---------|
| **规模** | 1T-15T tokens | 1万-100万条 | 1万-50万条 |
| **格式** | 纯文本（文档级） | 指令-回答对 | chosen/rejected 对 |
| **质量标准** | 宁缺毋滥，去重很关键 | 高质量 >> 大规模 | 对比差异要有意义 |
| **来源** | 网页爬虫、书籍、代码 | 人工标注 + 合成数据 | 人工标注 + 模型生成 |
| **成本** | 主要是清洗和计算成本 | 人工标注成本高 | 对比标注更耗时 |
| **关键挑战** | 去重、质量过滤、配比 | 多样性和难度覆盖 | 确保对比差异有意义 |

一个直觉：**预训练数据决定模型"知道什么"，SFT 数据决定模型"怎么说话"，偏好数据决定模型"说得多好"。**

---

## 预训练数据集构建

### 数据来源与规模

| 数据源 | 典型规模（tokens） | 特点 | 代表数据集 |
|--------|-------------------|------|-----------|
| Common Crawl | 数万亿 | 覆盖广、噪声大 | FineWeb, RedPajama |
| Wikipedia | 英文 ~40 亿 / 中文 ~10 亿 | 高质量、结构化 | 20+ 语言版本 |
| GitHub 代码 | 数千亿 | 提升推理和代码能力 | The Stack v2 |
| 书籍语料 | 数百亿 | 长文本、高质量叙述 | Books3, Gutenberg |
| 学术论文 | ArXiv ~300 亿 | 数学和科学推理 | RedPajama-ArXiv |
| StackOverflow | ~100 亿 | 问答格式、实践知识 | Stack Exchange dump |
| 新闻 | 数百亿 | 时事知识 | CC-News |

推荐直接用 `datasets` 库一行加载流式语料（如 `load_dataset("HuggingFaceFW/fineweb", streaming=True)`），再调 `tokenizer` 切块；JSONL 是最常见的中间产物格式，每行一个 `{"text": ...}` 即可。文档级流程是：**逐行解析 → 拼接为长串 → 按 `max_seq_len` 切块 → 在每块末尾插入 `<eos>`**。

### 数据清洗流水线

#### URL 过滤

第一道防线：基于 URL 规则过滤明显的低质量来源。

- 黑名单域名：成人站点、赌博站点、已知 spam 域名
- 白名单域名（高权重）：edu, gov, 知名媒体、Wikipedia
- URL 模式过滤：移除过长 URL、含有广告参数的 URL

#### 语言识别 (fastText)

```python
# 使用 fastText 进行语言检测
import fasttext

model = fasttext.load_model('lid.176.bin')

def detect_language(text, threshold=0.65):
    """检测文本语言，返回语言代码和置信度"""
    predictions = model.predict(text.replace('\n', ' ')[:500])
    lang = predictions[0][0].replace('__label__', '')
    score = predictions[1][0]
    return lang, score

# 过滤：只保留目标语言且置信度 > 0.65 的文档
lang, score = detect_language("This is an English text.")
# lang='en', score=0.99
```

#### 质量过滤 (perplexity, n-gram, heuristic rules)

**基于规则的启发式过滤**（FineWeb 使用的部分规则）：

| 规则 | 阈值 | 说明 |
|------|------|------|
| 平均行长度 | > 10 字符 | 过滤列表/目录页 |
| 最长行长度 | < 100,000 字符 | 过滤异常文件 |
| 字母数字比例 | > 0.6 | 过滤编码乱码 |
| 重复行比例 | < 0.3 | 过滤模板页面 |
| 特殊字符比例 | < 0.2 | 过滤代码注释/日志 |
| 包含 "lorem ipsum" | 移除 | 占位符文本 |
| 停用词覆盖率 | > 一定比例 | 确保是自然语言 |

**基于困惑度的过滤**：

用预训练好的 n-gram 语言模型（如 KenLM）计算每篇文档的困惑度。高困惑度意味着文本不像自然语言（可能是乱码、机器生成的 spam）。

**基于分类器的过滤**（FineWeb-Edu 方法）：

训练一个质量分类器，以 Wikipedia 和教科书为正例、随机网页为负例，对每篇文档打分。FineWeb-Edu 使用 Llama 3 对 50 万网页打教育质量分（0-5 分），然后训练 fastText 分类器在全量数据上快速推断。

#### MinHash 去重

去重是预训练数据清洗中最关键的步骤之一。文档级去重通常使用 MinHash + LSH（局部敏感哈希）。

**MinHash 签名的计算原理**：

1. 将文档转换为 n-gram 集合（如 5-gram）
2. 选择 $k$ 个不同的哈希函数 $h_1, h_2, \dots, h_k$
3. 对每个哈希函数，计算所有 n-gram 的哈希值，取最小值作为签名的一个分量
4. 两篇文档的 MinHash 签名碰撞概率 = Jaccard 相似度

$$P(\text{MinHash}(A) = \text{MinHash}(B)) = J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

**LSH（局部敏感哈希）加速**：

将 $k$ 个签名分成 $b$ 个 band，每个 band 包含 $r$ 个签名（$k = b \times r$）。两篇文档在任一 band 完全匹配就被视为候选近似重复对。

- $b$ 越大：越容易匹配（高召回，可能误判）
- $r$ 越大：越难匹配（高精度，可能遗漏）

典型配置：$k=128$, $b=9$, $r \approx 14$，对应 Jaccard > 0.8 的文档对。

#### 去污染 (Decontamination)

确保评测集的内容不出现在训练集中，否则评测分数会虚高。

方法：
1. 收集所有常用评测集（MMLU, HumanEval, GSM8K, HellaSwag 等）
2. 将评测集内容转换为 n-gram 集合
3. 在训练数据中搜索高度重叠的段落
4. 移除或替换包含评测集内容的训练文档

Llama 3 使用 10-gram 重叠度来判定数据污染，移除了与评测集有显著重叠的训练样本。

---

## SFT 数据格式

### Alpaca 格式

最早由 Stanford Alpaca 提出的简洁格式：

```json
{
    "instruction": "将以下段落翻译成英文",
    "input": "人工智能正在改变世界。",
    "output": "Artificial intelligence is changing the world."
}
```

当 `input` 为空时，只有 `instruction` 和 `output`。简单直接，适合单轮任务。

### ShareGPT 格式

来自用户分享的 ChatGPT 对话，天然支持多轮：

```json
{
    "conversations": [
        {"from": "system", "value": "你是一个有帮助的助手。"},
        {"from": "human", "value": "什么是机器学习？"},
        {"from": "gpt", "value": "机器学习是人工智能的一个分支..."},
        {"from": "human", "value": "能举个例子吗？"},
        {"from": "gpt", "value": "比如垃圾邮件分类..."}
    ]
}
```

### OpenAI 格式

OpenAI API 风格的 messages 格式，已成为事实标准：

```json
{
    "messages": [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "什么是机器学习？"},
        {"role": "assistant", "content": "机器学习是人工智能的一个分支..."}
    ]
}
```

### Chat Template

不同模型使用不同的 chat template 来标记角色边界。tokenizer 需要正确处理这些特殊标记。

**ChatML 格式**（Qwen 系列使用）：

```
<|im_start|>system
你是xiaoming智能体,请安全详细回答用户的问题<|im_end|>
<|im_start|>user
$sin^2x+cos^2x=?<|im_end|>
<|im_start|>assistant
结果为 $\boxed{1}$<|im_end|>
```

**Llama 格式**：

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>
```

```python
# Chat Template 的实际使用
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)

messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': '用一句话解释梯度下降。'},
    {'role': 'assistant', 'content': '梯度下降是一种迭代优化算法，沿损失函数负梯度方向更新参数以最小化损失。'},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print(prompt)
```

**SFT 训练的关键**：只在 assistant 段计算损失。构造与 `input_ids` 等长的 `labels`，把 system / user 段全部填 `-100`，配合 `CrossEntropyLoss(ignore_index=-100)` 即可自动跳过：

```text
input:  <system>...<user>...<assistant> 回答内容 <eos>
labels: [-100]    [-100]   [-100]       回答 token  eos
                                        ↑ 只有这一段参与损失计算
```

---

## 合成数据生成

### Self-Instruct

Wang et al. (2022) 提出的方法，用模型自身生成指令数据：

```
流程:
1. 准备 175 条人工编写的 seed instructions
2. 从已有 task pool 中随机抽取 6 条作为 few-shot 示例
3. 让模型生成新的 instruction
4. 让模型为新 instruction 生成 input 和 output
5. 质量过滤：去重 + ROUGE-L 过滤相似指令 + 关键词过滤
6. 将通过过滤的样本加入 task pool
7. 重复步骤 2-6
```

Self-Instruct 生成的 52K 数据训练出的模型（Alpaca）在开放指令上已经接近 text-davinci-003 的 performance。

### Evol-Instruct (WizardLM)

Xu et al. (2023) 提出的指令进化方法。核心思想：用 LLM 对已有指令进行"进化"，增加复杂度和多样性。

**两种进化方向**：

| 类型 | 策略 | 示例 |
|------|------|------|
| **深度进化** | 增加约束、增加推理步骤、换具体场景 | "排序数组" → "用快速排序算法对包含重复元素的整数数组进行排序，要求空间复杂度 O(1)" |
| **广度进化** | 变换主题、变换任务类型 | "排序数组" → "设计一个数据库索引策略" |

```
进化 Prompt 示例（深度）:
"I want you to act as a Prompt Rewriter.
Your objective is to rewrite a given prompt into a more complex version.
The rewritten prompt must be reasonable, understood by humans, and answerable.

#Given Prompt#:
写一个 Python 函数来计算列表的平均值。

#Rewritten Prompt#:
（让 LLM 生成更复杂的版本）"
```

### 使用 GPT-4 / Claude 生成数据的最佳实践

1. **Prompt 设计要具体**：明确角色、输出格式、长度要求、质量标准
2. **temperature 调高以增加多样性**：通常 0.7-1.0
3. **批量生成 + 后过滤**：生成 5x 的量，过滤掉低质量的
4. **避免模式坍缩**：定期检查生成内容的多样性，用不同的 seed prompt
5. **标注质量校验**：随机抽样 5-10% 人工检查

### Seed Task 设计

好的 seed task 应当覆盖：

| 维度 | 示例 |
|------|------|
| 任务类型 | 分类、生成、翻译、总结、推理、代码、数学 |
| 难度层级 | 基础知识 → 分析应用 → 创造性推理 |
| 输出格式 | 短回答、长文、代码、JSON、表格、列表 |
| 语言风格 | 正式、口语化、技术文档、对话 |

### 质量把控

合成数据的常见问题及对策：

| 问题 | 检测方法 | 对策 |
|------|---------|------|
| 幻觉/事实错误 | 对比知识库、人工抽检 | 增加事实验证步骤 |
| 格式不一致 | 正则表达式检查 | Prompt 中明确格式要求 |
| 多样性不足 | n-gram diversity 统计 | 增加 seed 多样性、升高 temperature |
| 难度分布不均 | 人工抽样评估 | 使用 Evol-Instruct 增加难题 |
| 含有 AI 痕迹 | 检测"As an AI"等套话 | 过滤或 prompt 中禁止 |

---

## 数据增强技术

### Rejection Sampling

让模型对同一问题生成多个回答，用 reward model 打分，只保留高分回答。

```
Input: "解释量子纠缠"
├── 回答 1 (reward=0.85) ✓ 保留
├── 回答 2 (reward=0.32) ✗ 丢弃
├── 回答 3 (reward=0.91) ✓ 保留
└── 回答 4 (reward=0.45) ✗ 丢弃
```

Llama 3 在 SFT 阶段就大量使用 rejection sampling 来筛选训练数据。

### Chain-of-Thought 扩展

对已有的问答数据，补充推理链（chain-of-thought），提升模型推理能力：

```json
{
    "instruction": "一个水池有两个水管，A管每小时注水3吨，B管每小时放水1吨。水池中有10吨水，同时开两管，几小时后水池有22吨水？",
    "output_without_cot": "6小时",
    "output_with_cot": "让我们一步步思考：\n1. A管注水速度：3吨/小时\n2. B管放水速度：1吨/小时\n3. 净注水速度：3-1=2吨/小时\n4. 需要增加的水量：22-10=12吨\n5. 所需时间：12÷2=6小时\n\n答案是 6 小时。"
}
```

### 多样性增强 (Persona-driven)

让模型扮演不同角色来回答同一问题，增加回答的多样性：

```
同一问题："解释什么是递归"

Persona 1（大学教授）：递归是一种函数调用自身的编程技术...（学术风格）
Persona 2（少儿编程老师）：想象你站在两面镜子中间...（比喻风格）
Persona 3（面试官）：递归需要满足两个条件：基准情形和递归步骤...（面试风格）
```

### 难度递增 (Auto-Evol)

在 Evol-Instruct 基础上，自动化地逐步增加难度：

```
Level 1: 写一个函数计算斐波那契数列第 n 项
Level 2: 写一个函数计算斐波那契数列第 n 项，要求时间复杂度 O(log n)
Level 3: 实现广义斐波那契数列的矩阵快速幂解法，支持自定义初始值和递推系数
```

---

## 偏好数据构建

### Chosen/Rejected pair 的收集

偏好数据的核心形式：对于同一个 prompt，提供一个"好的"回答（chosen）和一个"差的"回答（rejected）。

```json
{
    "prompt": "什么是梯度下降？",
    "chosen": "梯度下降是一种优化算法。它通过计算损失函数对参数的梯度，沿梯度反方向更新参数，逐步找到损失函数的（局部）最小值。学习率控制每步更新的幅度。",
    "rejected": "梯度下降就是让损失变小的方法。"
}
```

好的偏好数据要求 chosen 和 rejected 之间有**有意义的质量差异**，而不仅仅是长度差异。

### 人工标注 vs 模型标注

| 方法 | 优势 | 劣势 |
|------|------|------|
| **人工标注** | 质量高、反映真实人类偏好 | 成本高、速度慢、标注者之间一致性差 |
| **模型标注** (AI Feedback) | 成本低、速度快、一致性高 | 可能引入模型偏见、难以超越标注模型能力 |
| **混合方法** | 平衡质量和效率 | 需要设计好的质量控制流程 |

实践建议：用人工标注建立"金标准"子集（~1000 条），用于校准模型标注的质量；剩余大量数据用模型标注 + 人工抽检。

### Bradley-Terry 模型的数据需求

RLHF 中的 reward model 通常使用 Bradley-Terry 模型训练：

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

数据需求：
- **最低规模**：~10K 对偏好数据可以训练出有意义的 reward model
- **推荐规模**：50K-200K 对
- **质量要求**：标注者间一致性（inter-annotator agreement）应 > 70%
- **覆盖要求**：prompt 应覆盖多种任务类型和难度级别

---

## 数据质量评估

### 自动评估指标

| 指标 | 适用场景 | 工具 |
|------|---------|------|
| 困惑度 (Perplexity) | 预训练数据质量 | KenLM, GPT-2 |
| ROUGE / BLEU | 生成任务的参考对比 | nlg-eval |
| BERTScore | 语义相似度评估 | bert_score 包 |
| 多样性 (Distinct n-gram) | 检测数据单调性 | 自定义脚本 |
| 平均长度 & 长度分布 | 检测长度偏差 | 统计分析 |

### Reward Model 打分

用训练好的 reward model 对 SFT 数据进行评分，是一种高效的质量筛选方法：

```python
# 伪代码：使用 reward model 筛选数据
for sample in sft_dataset:
    score = reward_model.score(sample['prompt'], sample['response'])
    if score > threshold:
        filtered_dataset.append(sample)

# 典型做法：保留 top 50-70% 的数据
```

### 去重与去污染验证

**SFT 数据去重**：

```python
# 基于编辑距离的去重
from difflib import SequenceMatcher

def is_near_duplicate(text1, text2, threshold=0.85):
    ratio = SequenceMatcher(None, text1, text2).ratio()
    return ratio > threshold

# 或者使用 embedding 相似度
# cosine_sim = cos(embed(text1), embed(text2))
```

**去污染验证清单**：
- [ ] 检查训练数据与 MMLU 的 n-gram 重叠
- [ ] 检查训练数据与 HumanEval 的代码重叠
- [ ] 检查训练数据与 GSM8K 的数学题重叠
- [ ] 检查训练数据与 HellaSwag, ARC 等的重叠
- [ ] 使用 13-gram 或更严格的标准匹配

---

## 苏格拉底时刻

1. **用 GPT-4 生成的合成数据训练小模型，是否构成"模型坍缩"（Model Collapse）的风险？** 是的。如果多代模型都用前代生成的合成数据训练，分布会逐渐退化。解决方案：每一代都混入真实人工数据，保持数据分布的多样性。

2. **数据去重为什么如此重要？** 如果某段文本重复出现 100 次，模型会对它过拟合（记忆而非理解），同时挤占其他数据的学习机会。实验表明去重可以让模型在相同计算量下获得更好的泛化能力。

3. **SFT 数据的数量并不需要很大（通常几万条），为什么少量高质量数据反而比大量低质量数据效果更好？** 因为 SFT 不是在教模型新知识（那是预训练的工作），而是在"激活"模型已有的能力，教它以正确的格式输出。这更像"调音"而非"教学"，因此质量远比数量重要。

4. **在构建偏好数据时，如何确保 chosen 和 rejected 的差异是有意义的？** 差异应该体现在事实准确性、完整性、逻辑性等维度，而不是仅仅是长度或格式差异。一个好的做法是让标注者写下"为什么 chosen 更好"的理由。

5. **Chat Template 的特殊 token 如果处理不当会怎样？** 模型会无法正确区分用户输入和自己的输出，导致在生成时"角色混乱"。这就是为什么 tokenizer 的 `apply_chat_template` 方法如此重要。

---

## 常见问题 & 面试考点

| 问题 | 要点 |
|------|------|
| 预训练数据和 SFT 数据的核心区别？ | 预训练用无标注文本学习知识，SFT 用指令数据激活能力 |
| MinHash 去重的原理？ | 用 n-gram 集合的最小哈希值近似 Jaccard 相似度，LSH 加速候选对检索 |
| 解释 Self-Instruct | 用模型自身从 seed tasks 出发生成指令-回答对，迭代扩充 |
| Evol-Instruct 的两个进化方向？ | 深度进化（增加复杂度）和广度进化（扩展话题） |
| SFT 训练时只在 assistant 部分计算损失，为什么？ | 因为 system 和 user 部分是"输入条件"，不是模型需要学习生成的内容 |
| 偏好数据的 chosen/rejected 差异太小会怎样？ | Reward model 无法学到有意义的偏好信号，RLHF 效果会很差 |
| ChatML 和 Llama chat template 有什么区别？ | ChatML 用 `<\|im_start\|>/<\|im_end\|>`，Llama 用 `<\|start_header_id\|>/<\|eot_id\|>` |
| 数据去污染为什么重要？ | 防止训练数据泄露评测集内容，导致评测分数虚高 |

---

## 推荐资源

### 论文

- [Self-Instruct: Aligning LMs with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) — 合成数据的开山之作。
- [WizardLM: Empowering LLMs to Follow Complex Instructions](https://arxiv.org/abs/2304.12244) — Evol-Instruct 方法。
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) — 系统论证了去重对预训练性能的提升与对记忆化的抑制。
- [The RefinedWeb Dataset](https://arxiv.org/abs/2306.01116) — Falcon 团队提出的"只用 Web 数据也能匹敌精选语料"的 500B 数据集。
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — 仅用 1,000 条精选 SFT 数据即可激活基础模型的对齐能力。
- [Llama 3 技术报告](https://arxiv.org/abs/2407.21783) — 数据工程最佳实践，含分类器驱动的质量过滤与去污染流程。

### 博客与技术报告

- [RedPajama](https://github.com/togethercomputer/RedPajama-Data) — Llama1 训练数据的开源复现，覆盖 CommonCrawl/C4/GitHub/Books/ArXiv/Wikipedia/StackExchange 七大来源。
- [FineWeb: decanting the web for the finest text data at scale](https://huggingface.co/spaces/HuggingFaceFW/blog-fineweb-v1) by Penedo et al. (HuggingFace) — 当前最权威的"如何把 96 个 CommonCrawl dump 蒸馏成 15T tokens 高质量预训练语料"的实战长文。完整披露了 datatrove 流水线的每一步：URL 黑名单 → trafilatura 从 WARC 直接抽文（**不要用 CC 自带的 WET，会多保留 25% 但都是 boilerplate**）→ fastText 英文分类（≥ 0.65）→ Gopher 质量与重复过滤 → **每个 dump 单独 MinHash 去重**（112 个 hash / 14 桶 × 8）→ 选择性 C4 过滤（保留除 terminal_punct 之外的全部，因为它会砍掉 30% token）→ 三个新增统计过滤器（行末标点比、重复行字符比、短行占比）→ PII 移除。最反直觉的发现是**"全局跨 dump 去重反而损害性能"**：因为只在一个 dump 出现的"独有内容"质量更高，跨 dump 重复保留下来的反而是经过多次抓取的低质内容；老 dump 中 90% 被全局 dedup 移除的数据，单独训练时反而比"留下来的 10%"更好。配套发布的 FineWeb-Edu（用 Llama-3-70B 给 50 万样本打 0–5 教育分→蒸馏成 Snowflake-arctic-embed 分类器→以阈值 ≥3 过滤得到 1.3T，阈值 ≥2 得到 5.4T）在 MMLU/ARC/OpenBookQA 上显著超过 FineWeb 本身，10× 更少 token 即可匹配 C4/Dolma 的 MMLU 成绩。同时开源了 [datatrove](https://github.com/huggingface/datatrove)（处理）、[nanotron](https://github.com/huggingface/nanotron)（1.82B Llama-arch 消融模型）、[lighteval](https://github.com/huggingface/lighteval)（评测）三件套与全部 ablation checkpoint。是理解"高质量预训练数据集到底是怎么做出来的"最系统的公开材料，强烈推荐通读全文配合后文 [pretraining.md](pretraining.md) 的训练流程一起看。
- [HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) — FineWeb-Edu 用的教育质量分类器（含训练/推理代码），可直接套用到自己的语料上做"教育性"过滤。
- [Building High-Quality Datasets with distilabel and Prometheus 2](https://huggingface.co/blog/burtenshaw/distilabel-prometheus-2) by Ben Burtenshaw (HuggingFace, 2024-06-03) — 全流程演示如何用 [distilabel](https://github.com/argilla-io/distilabel) 流水线 + [Prometheus 2](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0) 7B 评估模型同时蒸馏 SFT 和 DPO 数据集。**核心动机**：以前要做高质量 AI Feedback (AIF) 必须用 GPT-4，又贵又不透明；Prometheus 2 是把 GPT-4 评估能力蒸馏到 Mistral-7B 上的开源替代，通过权重合并支持两种模式——**absolute**（对单个回答打 1–5 分）和 **relative**（成对比较哪个更好）——刚好覆盖 SFT 过滤和 DPO 偏好对构建两类需求；rubric 可指定 `factual-validity / helpfulness / honesty / harmlessness` 等多维度。**Pipeline 1：SFT 蒸馏**——`LoadHubDataset(openbmb/UltraInteract_sft)` → `PrometheusEval(mode="absolute", rubric="factual-validity")` 用 vLLM 高吞吐打分 → 过滤低分样本，得到精炼 SFT 集。**Pipeline 2：SFT → DPO**——同一 prompt 让 `Llama-3-70B-Instruct` 和 `Llama-3-8B-Instruct` 通过 HuggingFace Inference Endpoints 各生成一份 response → `CombineColumns` 把多模型输出聚合 → `PrometheusEval(mode="relative")` 成对比较 → `KeepColumns` 保留 `instruction/generations/feedback/result/model_name` → 可选 `DPOToArgilla` 推到 Argilla 由人审复核。**值得复用的工程模式**：① 评估模型与生成模型解耦；② vLLM 放本地、生成模型放 Inference Endpoints，节省显存预算；③ `num_examples=3/5` 先小批跑通再放量；④ 把 Argilla 接到管线尾巴上做 human-in-the-loop。这套思路是当下"开源闭环数据生成"的代表实践，与 FineTome-100k（[sft.md](sft.md) 中提到的 arcee-ai/The-Tome 用 fineweb-edu-classifier 过滤）形成 *预训练用分类器 / 微调用 LLM-as-judge* 的两条清晰路线。

### 代码参考

- [karpathy/build-nanogpt — fineweb.py](https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py) — 80 行可读完的最小可用流水线：`load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT")` → `tiktoken` GPT-2 分词 → 多进程 `Pool.imap` 加速 → 切成 100M tokens 一片的 `.npy` shard，正好对接 [pretraining.md](pretraining.md#代码参考) 里 `DataLoaderLite` 的读取格式。配 [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) 视频里数据准备的一段一起看。
- [argilla-io/distilabel](https://github.com/argilla-io/distilabel) — Apache-2.0 的合成数据生成流水线框架，内置 `LoadHubDataset / TextGeneration / PrometheusEval / CombineColumns / KeepColumns / DPOToArgilla` 等可拼装节点；底层支持 vLLM、HuggingFace Inference Endpoints、OpenAI/Anthropic API 等多种 LLM 后端。
- [prometheus-eval/prometheus-eval](https://github.com/prometheus-eval/prometheus-eval) — Prometheus 2 评估模型的官方仓库，含 absolute / relative 两种模式的 prompt 模板和多种 rubric（factual-validity / helpfulness / honesty / harmlessness 等），可直接挂到 distilabel 的 `PrometheusEval` 节点。
- [argilla-io/argilla](https://github.com/argilla-io/argilla) — 开源数据标注平台，与 distilabel 无缝衔接，把 LLM 生成的 chosen/rejected 对推入后由人工二次审核，闭环 human-in-the-loop。
