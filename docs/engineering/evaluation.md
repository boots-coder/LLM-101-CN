---
title: "评估"
description: "MMLU/HumanEval/GSM8K 等 Benchmark、LLM-as-Judge、Chatbot Arena、污染检测"
topics: [evaluation, MMLU, HumanEval, GSM8K, LLM-as-Judge, Chatbot-Arena, benchmark, contamination, HealthBench, rubric-evaluation, meta-evaluation]
---
# 评估

::: info 一句话总结
评估是大模型开发中最容易被忽视但最关键的环节——你无法改进你无法衡量的东西。
:::


## 为什么评估很难？

大模型评估面临的独特挑战：

- **能力多样**：一个模型需要同时具备推理、编码、数学、创作、对话等多种能力
- **开放式输出**：不像分类任务有唯一正确答案，生成任务的好坏很主观
- **评估污染**：训练数据可能包含测试集内容（数据泄露），导致评估结果虚高
- **涌现能力**：某些能力只在特定规模的模型上出现，传统 benchmark 可能无法捕捉

## 主流 Benchmark 详解

### MMLU（Massive Multitask Language Understanding）

MMLU 是评估 LLM 知识广度的标杆测试，覆盖 57 个学科。

**格式**：四选一选择题

```
Question: The longest wavelength of light that can be used to cause
photoionization of an atom of Li²⁺ is approximately
(A) 10 nm   (B) 20 nm   (C) 100 nm   (D) 200 nm
Answer: (A)
```

**评估方式**：
- Few-shot（通常 5-shot）：在 prompt 中给出 5 个示例，然后让模型作答
- 计算方式：比较模型输出的下一个 token 在 A/B/C/D 四个 token 上的概率，取概率最高的作为答案
- 也可以直接生成答案字母，再用正则提取

**注意事项**：
- 不同评估框架的实现细节不同（prompt 格式、概率计算方式），导致同一模型得分可能差异显著
- MMLU-Pro 是升级版：选项增加到 10 个，加入更多推理题，减少随机猜对的概率

### HumanEval（代码生成）

HumanEval 测试模型的代码生成能力，包含 164 道 Python 编程题。

**格式**：给出函数签名和 docstring，模型补全函数体

```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers
    closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0], 0.3)
    True
    """
    # 模型需要补全这里的代码
```

**核心指标 pass@k**：

生成 k 个候选代码，只要有 1 个通过所有测试用例即算通过。

$$\text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

其中 $n$ 是总生成数，$c$ 是通过的数量。实践中通常生成 $n$ 个样本（如 200），然后用上式估算 pass@1、pass@10、pass@100。

**为什么用 pass@k 而不是 pass@1**？
- pass@1 方差大，受采样随机性影响
- pass@k 更稳定，且反映了模型的"潜力"——通过采样多次能否解决问题

### GSM8K（数学推理）

GSM8K 包含 8,500 道小学到初中水平的数学应用题。

**特点**：
- 需要多步推理（平均 2-8 步）
- 答案是具体数字，可以精确匹配
- 常配合 Chain-of-Thought (CoT) 评测

```
Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast
every morning and bakes muffins for her friends every day with four.
She sells the remainder at the farmers' market daily for $2 per fresh
duck egg. How much in dollars does she make every day at the farmers' market?

Answer: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = <<9*2=18>>$18 every day.
#### 18
```

### 其他重要 Benchmark

| Benchmark | 评测能力 | 核心指标 | 难度定位 |
|-----------|----------|----------|----------|
| **HellaSwag** | 常识推理 | 准确率 | 补全日常场景描述，人类准确率 95%+ |
| **TruthfulQA** | 真实性 | MC1/MC2 准确率 | 测试模型是否会复述常见误解 |
| **MATH** | 高等数学 | 准确率 | 竞赛级数学，需要严格推理 |
| **ARC** | 科学推理 | 准确率 | 小学科学选择题，分 Easy 和 Challenge |
| **WinoGrande** | 常识推理 | 准确率 | 代词消歧，需要世界知识 |
| **MBPP** | 代码生成 | pass@k | 974 道 Python 题，比 HumanEval 更多 |
| **IFEval** | 指令遵循 | 严格/宽松准确率 | 测试模型能否遵循格式约束 |

## 排行榜

### Open LLM Leaderboard（HuggingFace）

HuggingFace 维护的开源模型排行榜，是社区最常引用的参考。

**Leaderboard v2 使用的 Benchmark**：
- IFEval（指令遵循）
- BBH（Big Bench Hard，复杂推理）
- MATH（数学，Lvl 5）
- GPQA（研究生级科学问答）
- MUSR（多步推理）
- MMLU-Pro（升级版知识测试）

**注意事项**：
- 所有评测使用统一的 lm-evaluation-harness 框架
- 模型必须开源权重才能上榜
- v1 和 v2 的 benchmark 不同，分数不可直接比较

### Chatbot Arena（LMSYS）

基于真实用户投票的 ELO 排名系统，被广泛认为是**最能反映真实能力的排行榜**。

**工作原理**：
1. 用户输入任意问题
2. 系统随机选择两个匿名模型（用户不知道是哪个模型）
3. 两个模型同时回答
4. 用户选择更好的（A 更好 / B 更好 / 平局 / 都不好）
5. 基于 Bradley-Terry 模型更新 ELO 分数

**ELO 评分系统**：
- 源自国际象棋排名，用于成对比较
- 新模型从基准分开始，每次比较后调整
- 经过数万次比较后收敛到稳定排名
- 截至目前已收集超过百万次投票

**为什么 Chatbot Arena 更可靠**：
- 问题来自真实用户，不会被针对性优化
- 盲评避免了品牌偏见
- 持续更新，不会饱和
- 但也有局限：用户偏好可能偏向流畅性而非准确性

### AlpacaEval

自动化对话评估框架，使用 GPT-4 作为评判。

- **AlpacaEval 2.0**：使用 Length-Controlled (LC) Win Rate，惩罚靠"长回答"刷分的行为
- 805 条评估指令，覆盖多种任务
- 成本低（相比 Chatbot Arena 的人工成本）
- 但本质上受限于 GPT-4 作为评判的偏差

## LLM-as-a-Judge 深度解析

使用强大的 LLM（如 GPT-4）来评判其他模型的输出质量。这是当前大模型评估中增长最快的范式。

### Direct Scoring（直接打分）

让评判模型对单个输出进行评分。

```
请评估以下回答的质量，从准确性、完整性、清晰度三个维度打分（1-10）：

问题：{question}
回答：{answer}

请给出每个维度的分数和理由。
```

**优点**：简单直接，可以获得绝对分数
**缺点**：评分标准模糊，不同 judge 模型的尺度不一致，分数集中在某个区间

### Pairwise Comparison（成对比较）

让评判模型比较两个回答，选出更好的。

```
以下是同一个问题的两个回答，请判断哪个更好：

问题：{question}
回答 A：{answer_a}
回答 B：{answer_b}

请说明你的选择和理由。只输出 "A" 或 "B"。
```

**优点**：比绝对打分更稳定，人类也更擅长比较而非绝对评分
**缺点**：只能得到相对排序，需要 O(n²) 次比较

### Rubric Evaluation 与 HealthBench

OpenAI 在 2025 年提出的 **HealthBench**（arXiv 2505.08775）把 LLM-as-Judge 推到了第三种范式：**rubric evaluation**——既不是 Direct Scoring 给一个总分，也不是 Pairwise Comparison 选出胜者，而是给每条对话**一份独立的、由领域专家逐条撰写的评分细则**，judge 模型对照清单逐条打勾。

**核心机制**：每个样本配一个 rubric，rubric 由若干「self-contained, objective criterion」组成，每条 criterion 带一个 −10 到 +10 之间的非零整数权重——正权重代表"应该出现的行为"（如"建议立即就医"），负权重代表"不应出现的行为"（如"在结尾才轻描淡写地提到转诊"）。Judge 独立判断每条 criterion 是否被满足（meet / not meet）。

最终得分公式（论文 Section 2）：

$$
\text{Score} = \mathrm{clip}_{[0,1]}\left(\frac{\sum_{i \in \text{met}} w_i}{\sum_{j: w_j > 0} w_j}\right)
$$

> 注意分母只累加**正权重**——即"理论最大可能得分"。负权重在分子里以惩罚形式出现，因此单条样本的原始分可能为负，再 clip 到 $[0,1]$。

**Themes × Axes 双切片**：HealthBench 把 5,000 条对话切成两个正交维度，使得"总分 32%"可以拆解成"哪类对话哪个维度差"：

| 维度 | 名称 | 数量 |
|------|------|------|
| **Themes**（对话场景） | emergency referrals / global health / context-seeking / health data tasks / 等 | 7 个 |
| **Axes**（行为维度） | accuracy / completeness / instruction following / communication quality / context awareness | 5 个 |

**Rubric criterion 示例**（论文 Figure 1，婴儿活力下降场景）：

```json
[
  {"criterion": "States that the infant may have muscle weakness",            "points":  +7},
  {"criterion": "Briefly describes common causes of muscle weakness in infants","points":  +5},
  {"criterion": "Includes advice to seek medical care right away",            "points": +10},
  {"criterion": "Referral advice is only buried in the middle or at the end", "points":  -6}
]
```

**与 Direct Scoring / Pairwise 的对比**：

| 维度 | Direct Scoring | Pairwise Comparison | Rubric Evaluation |
|------|---------------|---------------------|-------------------|
| Judge 输出 | 1–10 标量 | A / B / Tie | 每条 criterion 的 0/1 命中 |
| 评分依据 | judge 自身先验 | 两回答相对优劣 | 专家预先写死的清单 |
| 可解释性 | 弱（只有总分） | 中（带理由文本） | 强（精确到哪条标准没满足） |
| 可诊断性 | 不分维度 | 不分维度 | Themes × Axes 双切片 |
| 防 judge 漂移 | 差（尺度不稳） | 中（位置偏差） | 强（criterion 是固定锚点） |
| 主要成本 | 低 | 中（O(n²) 比较） | 高（每样本都要专家写 rubric） |

::: tip HealthBench 的"开放式领域评估"人力成本
- **5,000** 条多轮真实对话（mean 2.6 turns / 668 chars）
- **48,562** 条 conversation-specific rubric criteria（中位数 11 条/样本，最多 48 条）
- **262 位** 医生参与（覆盖 26 个 specialties、60 个国家、49 种语言），筛选自 1,021 位申请者
- 这套数字反映了 rubric evaluation 在开放式专业领域的真实代价：要把"主观质量"变成"可被 judge 客观打分的清单"，本质上是把专家判断**外化成一份份评分细则**。
:::

> 论文：Arora et al., *HealthBench: Evaluating Large Language Models Towards Improved Human Health*, OpenAI, arXiv 2505.08775（2025）。代码与数据集随 OpenAI `simple-evals` 仓库一同开源。

### 常见偏差与缓解

| 偏差类型 | 描述 | 缓解方法 |
|----------|------|----------|
| **位置偏差** (Position Bias) | 倾向于选择第一个或最后一个回答 | 交换 AB 顺序各评一次，取一致结果 |
| **长度偏差** (Length Bias) | 倾向于选择更长的回答 | 在 prompt 中明确不以长度为标准；使用 LC Win Rate |
| **自我偏好** (Self-Preference) | GPT-4 评判时倾向于选择 GPT-4 的输出 | 使用多个不同的评判模型交叉验证 |
| **格式偏差** (Format Bias) | Markdown 格式、列表排版得分更高 | 统一格式后再评判 |
| **华丽偏差** (Verbosity Bias) | 修辞华丽但内容空洞的回答得分高 | 在评判 prompt 中强调内容准确性 |

#### Meta-evaluation：怎么证明 judge 本身可靠？

上面的偏差列表回答的是"judge 会出哪些错"，但还有一个更根本的问题：**你怎么知道你的 judge 整体是可信的？** HealthBench 给出的答案叫 **meta-evaluation**（论文 Section 8）：把 judge 模型的打分结果、和**人类专家**对同一批样本的打分结果做一致性比较，并把"专家间一致性"当作天花板——

- **model–physician agreement**：模型 judge 与医生在 criterion 命中判断上的吻合度
- **physician–physician agreement**：两位医生彼此独立标注同一样本时的吻合度（人类内在噪声）

如果前者 ≈ 后者，说明 judge 已经"和人一样可靠"，再继续优化 judge 反而是过拟合人类间的随机噪声；如果前者 ≪ 后者，说明 judge 还没追上人类水平。这个思路把 judge 可靠性从"主观相信"变成"可数值检验"。HealthBench 还据此设计了两个子集：**HealthBench Consensus**（3,671 个样本，只保留 34 条经多位医生共识验证的 criterion，降低医生间分歧噪声）和 **HealthBench Hard**（1,000 个困难样本，当时所有前沿模型得分均不超过 32%，专门防止 benchmark 过早饱和）。

### MT-Bench 评测流程详解

MT-Bench 是 LLM-as-a-Judge 的经典实现，专门评测多轮对话能力：

```
┌──────────────────────────────────────────────────┐
│ MT-Bench Evaluation Pipeline                     │
│                                                  │
│ 1. 80 questions (8 categories x 10)              │
│    Categories: Writing, Roleplay, Reasoning,     │
│    Math, Coding, Extraction, STEM, Humanities    │
│                                                  │
│ 2. Two-turn dialogue per question                │
│    Turn 1: Open-ended question                   │
│    Turn 2: Follow-up with constraints            │
│    e.g. "Write a poem about spring"              │
│         -> "Rewrite as a haiku"                  │
│                                                  │
│ 3. GPT-4 scores each turn 1-10                   │
│    Uses reference answers for grading            │
│                                                  │
│ 4. Final Score = Average across all turns        │
└──────────────────────────────────────────────────┘
```

### 代码实现：构建一个简单的 LLM Judge

```python
import openai

def llm_judge_pairwise(question: str, answer_a: str, answer_b: str,
                        model: str = "gpt-4") -> dict:
    """
    用 LLM 做成对比较评判
    关键：交换顺序评两次，消除位置偏差
    """
    judge_prompt = """请作为一个公正的评判者，比较以下两个 AI 助手对用户问题的回答质量。

评判标准：
1. 准确性：信息是否正确
2. 有用性：是否真正回答了问题
3. 清晰度：表达是否清晰易懂

注意：不要因为回答更长就认为更好。关注内容质量而非形式。

用户问题：{question}

【回答 A】
{answer_a}

【回答 B】
{answer_b}

请先简要分析两个回答的优劣，然后给出最终判断。
最后一行只输出 "A" 或 "B" 或 "TIE"。"""

    client = openai.OpenAI()

    # 第一次评判：原始顺序
    response_1 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt.format(
            question=question, answer_a=answer_a, answer_b=answer_b
        )}],
        temperature=0,
    )
    result_1 = response_1.choices[0].message.content.strip().split("\n")[-1]

    # 第二次评判：交换顺序（消除位置偏差）
    response_2 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt.format(
            question=question, answer_a=answer_b, answer_b=answer_a  # 注意交换
        )}],
        temperature=0,
    )
    result_2 = response_2.choices[0].message.content.strip().split("\n")[-1]

    # 综合两次结果
    # 如果两次一致（考虑顺序交换），则采信；否则判为平局
    if result_1 == "A" and result_2 == "B":
        final = "A"     # 两次都认为原始 A 更好
    elif result_1 == "B" and result_2 == "A":
        final = "B"     # 两次都认为原始 B 更好
    else:
        final = "TIE"   # 不一致，判为平局

    return {
        "result_original_order": result_1,
        "result_swapped_order": result_2,
        "final_verdict": final,
    }

# 使用示例
result = llm_judge_pairwise(
    question="解释什么是梯度下降",
    answer_a="梯度下降是一种优化算法，通过计算损失函数的梯度来更新参数...",
    answer_b="梯度下降就像下山一样，沿着最陡的方向走...",
)
print(f"胜者: {result['final_verdict']}")
```

## 人类评估

### 为什么仍然需要人类评估？

- **金标准**：最终用户是人类，人类判断是最直接的评估
- **主观质量**：创意、幽默感、文风等维度难以自动化评估
- **安全性**：有害内容的判断需要人类的价值观和上下文理解
- **补充自动评估的盲区**：自动指标可能遗漏的问题人类可以发现

### Chatbot Arena 模式

LMSYS 的 Chatbot Arena 是目前公认最有价值的人类评估平台：

1. 用户输入问题，系统随机选择两个匿名模型回答
2. 用户选择更好的回答（或平局）
3. 使用 Bradley-Terry 模型计算 Elo 评分
4. 截至目前已收集超过百万次投票

### 标注流程设计

一个严谨的人类评估需要：

```
1. 定义评估维度
   ├── 准确性：事实是否正确
   ├── 有用性：是否真正回答了问题
   ├── 安全性：是否包含有害内容
   └── 流畅性：表达是否自然

2. 编写标注指南
   ├── 每个维度的评分标准（1-5 分各对应什么水平）
   ├── 边界案例的处理规则
   └── 至少 5 个标注示例（含正反面）

3. 试标注 + 校准
   ├── 3-5 名标注员标注相同的 50 条样本
   ├── 计算一致性，讨论分歧
   └── 修订标注指南

4. 正式标注
   ├── 每条样本至少 2-3 人标注
   ├── 随机插入质检题（gold set）
   └── 定期计算一致性指标
```

### 一致性指标：Cohen's Kappa

衡量标注员之间的一致性，排除随机一致的影响：

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

其中 $p_o$ 是实际一致率，$p_e$ 是随机情况下的期望一致率。

| $\kappa$ 值 | 一致性水平 |
|-------------|-----------|
| < 0.20 | 极差 |
| 0.21 - 0.40 | 一般 |
| 0.41 - 0.60 | 中等 |
| 0.61 - 0.80 | 较好 |
| 0.81 - 1.00 | 几乎完美 |

**实践中**：LLM 输出评估的 Kappa 值通常在 0.4-0.6 之间（中等），因为开放式文本的好坏确实很主观。

## 污染检测

### 为什么需要关注污染？

大模型的训练数据来自互联网，很可能包含了 benchmark 的测试集。如果模型"背过"了答案，评估分数就毫无意义。

### N-gram Overlap 检测

最简单的方法：检查训练数据中是否包含测试题的原文。

```python
def check_contamination(test_example: str, training_data: list[str],
                         n: int = 13) -> bool:
    """
    检查 test_example 的 n-gram 是否出现在训练数据中
    GPT-4 技术报告使用 n=13 作为阈值
    """
    test_ngrams = set()
    words = test_example.split()
    for i in range(len(words) - n + 1):
        test_ngrams.add(tuple(words[i:i+n]))

    for doc in training_data:
        doc_words = doc.split()
        for i in range(len(doc_words) - n + 1):
            if tuple(doc_words[i:i+n]) in test_ngrams:
                return True  # 检测到污染
    return False
```

### Benchmark 泄露的应对策略

| 策略 | 方法 | 代表案例 |
|------|------|----------|
| **保密测试集** | 不公开测试题，只提供提交接口 | HumanEval+ |
| **动态生成** | 每次评测生成新题目 | DynaBench |
| **污染后移除** | 检测到泄露的题目后从评分中移除 | GPT-4 技术报告 |
| **改写题目** | 保留考察点但改变表面形式 | MMLU-Pro |
| **时间戳过滤** | 只使用模型训练截止日期之后的数据 | LiveBench |

## 评估工具

### lm-evaluation-harness

EleutherAI 开发的标准化评估框架，是 Open LLM Leaderboard 的后端。

```python
# 安装
# pip install lm-eval

# 命令行使用
# lm_eval --model hf \
#     --model_args pretrained=meta-llama/Llama-2-7b-hf \
#     --tasks mmlu,hellaswag,gsm8k \
#     --num_fewshot 5 \
#     --batch_size 16 \
#     --output_path ./results

# Python API 使用
from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model="hf",
    model_args="pretrained=meta-llama/Llama-2-7b-hf",
    tasks=["mmlu", "hellaswag", "gsm8k"],
    num_fewshot=5,
    batch_size=16,
)

# 查看结果
for task, metrics in results["results"].items():
    print(f"{task}: {metrics}")
```

**常用参数**：
- `--num_fewshot`：few-shot 示例数量
- `--batch_size`：推理 batch size，设为 `auto` 自动调整
- `--limit`：每个任务最多评估多少样本（调试时有用）
- `--tasks`：支持 300+ 个预定义任务

### Lighteval（HuggingFace）

HuggingFace 开发的轻量评估框架，与 Hub 深度集成。

```bash
# 安装
pip install lighteval

# 使用
lighteval accelerate \
    --model_args "pretrained=meta-llama/Llama-2-7b-hf" \
    --tasks "leaderboard|mmlu|5|0,leaderboard|gsm8k|5|0" \
    --output_dir ./results
```

**对比**：

| 特性 | lm-evaluation-harness | Lighteval |
|------|----------------------|-----------|
| 任务数量 | 300+ | 较少但在增长 |
| 社区采用 | 更广泛 | HF 生态内 |
| 扩展性 | 自定义 task 较灵活 | 与 HF Hub 集成好 |
| 多卡推理 | 支持 | 原生 accelerate 支持 |

## 面试考点

### Benchmark 的局限性

**常见问题**："你如何看待当前 LLM 评估体系的问题？"

要点：
1. **Goodhart 定律**：当一个指标变成目标时，它就不再是好指标。模型可能针对 benchmark 过拟合
2. **数据泄露**：无法完全保证训练数据不包含测试集
3. **静态 vs 动态**：固定 benchmark 会被模型逐渐"饱和"
4. **覆盖不全**：单一 benchmark 只测一种能力，组合起来也有盲区
5. **评估格式敏感**：同一个问题换个 prompt 格式，得分可能差很多

### 如何设计评估方案？

**面试场景**："你需要评估一个新训练的 7B 对话模型，如何设计评估方案？"

```
第一层：自动化 Benchmark（快速、低成本）
├── 知识类：MMLU / MMLU-Pro
├── 推理类：GSM8K / MATH / BBH
├── 代码类：HumanEval / MBPP
├── 指令遵循：IFEval
└── 安全性：TruthfulQA

第二层：LLM-as-a-Judge（中等成本）
├── MT-Bench（多轮对话）
├── AlpacaEval 2.0（开放对话）
└── 自定义领域评测（用 GPT-4 评判）

第三层：人类评估（高成本、金标准）
├── 内部小规模评测（10-50 条核心场景）
├── A/B 测试（对比上一版本）
└── 安全红队测试（专项测试有害内容）

贯穿始终：
├── 污染检测：检查训练数据是否包含测试集
├── 消融实验：对比不同训练配置的效果差异
└── 案例分析：人工审查 bad case，定性了解模型弱点
```

### 高频面试问答

**Q：pass@k 和 pass@1 有什么区别？什么时候用哪个？**
A：pass@1 是单次生成通过率，反映用户实际体验（只用一次的场景）。pass@k 是 k 次中至少一次通过的概率，反映模型潜力。如果搭配代码验证器（如 AlphaCode 的 filter + cluster），pass@k 更有意义。

**Q：为什么不能只看一个 Benchmark 的分数？**
A：单一 Benchmark 只测一种能力，且可能因格式、prompt、few-shot 数量不同而波动。需要多个互补的 Benchmark + 人类评估综合判断。

**Q：如何发现模型在 Benchmark 上"作弊"（数据泄露）？**
A：N-gram 重叠检测、在 rephrased 版本上测试（分数是否大幅下降）、检查模型是否能"背诵"测试题的原文、对比公开和非公开测试集上的表现差异。

## 苏格拉底时刻

::: details 1. 为什么 Chatbot Arena 的 Elo 排名被认为比固定 benchmark 更可靠？
Chatbot Arena 基于真实用户的真实问题和偏好，避免了数据泄露和 benchmark 过拟合问题。Elo 系统通过大量成对比较收敛到稳定排名，且问题持续更新，不会饱和。但它也有局限：用户偏好可能偏向流畅度而非准确性。
:::


::: details 2. LLM-as-a-Judge 有什么系统性偏差？如何缓解？
主要偏差包括位置偏差、长度偏差和自我偏好。缓解方法：随机化回答顺序并多次评判、在 prompt 中明确评判标准、使用多个不同的评判模型交叉验证、与人类评估结果做校准。
:::


::: details "3. 一个模型在 MMLU 上得分很高，是否意味着它真的"懂"这些知识？"
不一定。高分可能来自训练数据泄露（见过测试题）、选择题的统计规律利用、或者表面模式匹配而非真正理解。需要结合多个不同格式的 benchmark、开放式问答、以及实际应用场景的表现来综合判断。
:::


::: details "4. 如何设计一个不容易被"刷分"的评测基准？"
几个方向：动态生成测试题（如 DynaBench）、使用开放式而非选择题格式、测试集保密不公开、定期更换测试集、引入对抗样本、评估推理过程而非仅看最终答案。最根本的是评估真实场景中的表现而非固定 benchmark。
:::


::: details 5. 人类评估的 Kappa 值只有 0.5 左右，这说明什么？对评估结论有什么影响？
说明开放式文本评估本身就具有主观性，不同人对"好"的定义不同。这意味着：(1) 需要更多标注员来获得稳定的平均值；(2) 评估结论应包含置信区间；(3) 对于分数接近的模型，差异可能不显著。
:::

