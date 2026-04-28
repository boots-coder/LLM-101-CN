---
title: 深度剖析 LLM-as-Judge——MT-Bench 与 AlpacaEval 工程实现
description: 把 GPT-4 当评分器要解决的工程问题：position bias、length bias、judge prompt 设计与 win-rate 计算
topics: [evaluation, mt-bench, alpaca-eval, llm-as-judge, position-bias, length-bias]
prereqs: [/engineering/evaluation, /training/alignment]
---

# 深度剖析 LLM-as-Judge——MT-Bench 与 AlpacaEval 工程实现

::: info 一句话总结
当 SFT/RLHF 之后的对话模型再也无法用 perplexity 或选择题分数刻画时，业界把 GPT-4 推上了"考官"的位置；而真正难做的，不是让它打分，是让分数稳定、无偏、可复现。
:::

::: tip 双来源声明
本文逐行参考两个 Apache 2.0 开源项目，所有引用均给出确切行号。
- **FastChat / MT-Bench**：[github.com/lm-sys/FastChat](https://github.com/lm-sys/FastChat)，Apache 2.0
- **AlpacaEval**：[github.com/tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval)，Apache 2.0

本地路径：`refrences-projects/FastChat`、`refrences-projects/alpaca_eval`。文中代码片段均控制在 ≤ 25 行，仅做学习注解；如需复用，请回到原仓库并保留 LICENSE。
:::

## 一、体系定位：为什么需要 LLM 当裁判

在 GPT-3 之前，NLP 评估基本是"客观题三件套"：

1. **Likelihood-based**：MMLU / HellaSwag / ARC——给一道单选题，比较模型对每个选项的 logits 谁更高。这只测"知不知道"，不测"会不会写"。
2. **Code-execution**：HumanEval / MBPP——直接跑代码，看 pass@1。这是仅有的"绝对客观"赛道，但只覆盖编程。
3. **N-gram match**：BLEU / ROUGE——靠词面重合，长得不像参考答案的好回答会被判低分。

而 SFT + RLHF 之后的"对话能力"是这些都覆盖不了的：写作、角色扮演、多轮指令理解、安全拒答……答案不是唯一的、没法机械比较的、没有 reference text 的。MT-Bench 论文（[arXiv:2306.05685](https://arxiv.org/abs/2306.05685)）给出了第三条路：让一个能力远强于被测模型的 LLM 当 judge，输入问题与回答，输出分数或胜负。

LLM-as-Judge 的现实定位：

| 方法 | 适用场景 | 自动化 | 与人类一致性 |
|------|---------|------|--------------|
| Likelihood | 选择题、知识 | 高 | 中 |
| Code-exec | 代码 | 高 | 高（但范围窄） |
| Human eval | 任何任务 | 低（贵、慢） | —— |
| **LLM-as-Judge** | 对话、写作、推理 | 高 | 与人类相关性 ≥ 0.8（强 judge）|

代价也很现实：MT-Bench 默认 80 题 × 2 轮 × 2 模型（pairwise）≈ 320 次 GPT-4 调用，按 2024 年 GPT-4-Turbo 价位单次约 0.05 美金，跑一次榜单 16 美金；AlpacaEval 2 单次完整跑 805 题接近 10 美金。便宜，但不免费——尤其是"重跑一次看 variance"这件事会被预算限制。

::: details 为什么传统 benchmark 在对齐之后失效
SFT/RLHF 模型的输出分布与 base model 完全不同：
- **风格化**：会主动加 emoji、加 markdown、加"As an AI assistant ..."；这些会让 BLEU 分崩溃。
- **拒答**：对敏感问题主动说 "I cannot help with that"，但选择题任务会被判为"答错"。
- **多轮一致性**：第二轮对第一轮的引用、否定、改写，靠 perplexity 完全测不到。

这就是为什么 MT-Bench / AlpacaEval / Arena-Hard / WildBench 这一波 LLM-as-Judge benchmark 在 2023-2024 年集体兴起——不是流行，是不得不。
:::

### 关键术语对齐

| 术语 | 含义 | 在哪用 |
|------|------|--------|
| Judge / Annotator | 当裁判的 LLM（通常是 GPT-4 系） | FastChat 叫 judge，AlpacaEval 叫 annotator |
| Pairwise | 同时给 judge 看 A/B 两个回答 | MT-Bench、AlpacaEval 默认 |
| Single grading | 只看一个回答打 1-10 分 | MT-Bench 备用模式 |
| Win rate | 被测模型 vs baseline 的胜率 | 两个项目都用 |
| LC-WR | length-controlled win rate | AlpacaEval 2.0 主指标 |
| Position bias | A/B 顺序影响判决 | 两个项目都要解决 |
| Length bias | 长回答更容易赢 | LC-WR 专门解决 |

### LLM-as-Judge 的核心工作流

```mermaid
flowchart LR
    Q[问题集<br/>80 / 805 题] --> M1[被测模型<br/>生成回答 A]
    Q --> M2[Baseline 模型<br/>生成回答 B]
    M1 --> P[组装 judge prompt<br/>system + question + A + B]
    M2 --> P
    P --> J[Judge 模型<br/>GPT-4 / GPT-4-Turbo]
    J -->|MT-Bench 路径| T1[文本 [[A]]/[[B]]/[[C]]]
    J -->|AlpacaEval 路径| T2[token logprob<br/>m / M]
    T1 --> R1[正则解析]
    T2 --> R2[softmax → 概率]
    R1 --> W1[win_rate_adjusted]
    R2 --> W2[win_rate + LC-WR<br/>GLM 校正]
```

两个项目共享前半段——区别在 judge 输出形式、解析方式和最终统计指标。下文 Part A 走文本路径，Part B 走 logprob 路径。

## 二、Part A：MT-Bench 怎么用一个 judge 跑出榜单

### 2.1 80 道题、8 个类别、两轮对话

MT-Bench 的题目结构非常简单——一个 JSONL 文件 [question.jsonl:L1-80](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/mt_bench/question.jsonl#L1)，每行一题，两轮人类提问（`turns[0]` 和 `turns[1]`），第二轮通常是基于第一轮回答的追问/改写：

```json
{"question_id": 81, "category": "writing",
 "turns": ["Compose an engaging travel blog post...",
           "Rewrite your previous response. Start every sentence with the letter A."]}
```

8 个类别的设计意图是覆盖"对话式 LLM 该会的事"：writing / roleplay / extraction / reasoning / math / coding / stem / humanities。其中 **math / reasoning / coding** 三类被定义为"需要参考答案"，因为 GPT-4 自己也容易在这些题上误判。

### 2.2 三种 judge 模式

FastChat 的 judge 分三种，对应不同评估目标：

- **single（绝对打分 1-10）**：每个回答独立打分，最后取均值。便于上 leaderboard。
- **pairwise-baseline**：被测模型 vs 一个固定 baseline（默认 `gpt-3.5-turbo`），每题输出 A/B/tie。
- **pairwise-all**：N 个模型两两 PK，输出胜率矩阵。

模式选择由命令行 `--mode` 决定，主流程见 [gen_judgment.py:L234-253](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_judgment.py#L234)。注意 single 与 pairwise 在 prompt、输出格式、解析逻辑上是**两套独立实现**，不是参数化共享。

### 2.3 NEED_REF_CATS：参考答案是给 GPT-4 看的

打开 [common.py:L31](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/common.py#L31)，会看到一个看似不起眼的常量：

```python
# 来自 FastChat / Apache 2.0
# 这四类必须配 reference answer，因为 GPT-4 自己算数也会错
NEED_REF_CATS = ["math", "reasoning", "coding", "arena-hard-200"]
```

为什么数学/推理/代码必须给参考答案？因为如果 judge 模型自己都没把题做对，它对被测模型的判断就是错的。把 reference 喂进 judge prompt，相当于"开卷批改"。`gen_judgment.py` 在分发任务时会按 category 分流，default 题走默认 prompt，math 题走 math prompt（自动注入 reference），见 [gen_judgment.py:L257-272](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_judgment.py#L257)。

### 2.4 Judge prompt 的设计：抗 bias 是写在 system prompt 里的

打开 [judge_prompts.jsonl:L1](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/judge_prompts.jsonl#L1) 看 `pair-v2` 的 system prompt（节选，原文 Apache 2.0）：

> Please act as an impartial judge ... **Avoid any position biases** and ensure that the order in which the responses were presented does not influence your decision. **Do not allow the length of the responses to influence your evaluation.** Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.

三件事被显式写进 prompt：

1. **要求先解释再判决**：CoT 在 judge 任务上是有效的，能减少草率决定。
2. **明确警告 position bias 与 length bias**：虽然不能完全消除，但能压一点。
3. **强约束输出格式 `[[A]]`**：方便正则解析（见 2.6 节）。

### 2.5 多轮对话怎么塞进 judge prompt

第二轮才是真正考验上下文理解的题。FastChat 的做法是把整段对话拼接进 prompt，用 `<|The Start of Assistant A's Conversation with User|>` 这样的伪标签包起来。看 `pair-v2-multi-turn` 模板（[judge_prompts.jsonl:L2](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/judge_prompts.jsonl#L2)）：

```text
<|The Start of Assistant A's Conversation with User|>
### User: {question_1}
### Assistant A: {answer_a_1}
### User: {question_2}
### Assistant A: {answer_a_2}
<|The End of Assistant A's Conversation with User|>
```

并且 system prompt 里专门加一句 *"You should focus on who provides a better answer to the second user question."*——把 judge 的注意力强制聚焦到第二轮，否则它会被第一轮的高质量答案带偏。

### 2.6 解析 judge 输出：正则不是装饰，是工程必需品

Judge 偶尔会"忘记格式"，吐出 `[[A]]` 或 `Rating: [[7]]` 之外的东西。FastChat 的兜底是双层正则：

```python
# [common.py:L33-37] 来自 FastChat / Apache 2.0
# pairwise 双数：[[8.5, 7.0]] 这种格式
two_score_pattern = re.compile("\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]")
two_score_pattern_backup = re.compile("\[(\d+\.?\d*),\s?(\d+\.?\d*)\]")  # 退一步：少一层括号也接受
# single 单数：[[7]] 这种格式
one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
```

解析失败的样本会打 `score = -1`，在 [show_result.py:L20](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/show_result.py#L20) 被显式过滤掉。**这点很重要**：如果 judge 失败率在某个模型上特别高（比如 5%+），意味着那个模型的输出形式非常规，会系统性地损害分数。

### 2.7 win-rate 怎么算：tie 当半个胜利

打开 [show_result.py:L77-92](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/show_result.py#L77)，会看到 MT-Bench 的 pairwise 胜率公式：

```python
# 来自 FastChat / Apache 2.0
# 朴素胜率：把平局也算进分母，但不算赢
df["win_rate"]  = df["win"] / (df["win"] + df["loss"] + df["tie"])
df["loss_rate"] = df["loss"] / (df["win"] + df["loss"] + df["tie"])
# 调整胜率：tie 算半场胜利（国际象棋/Elo 系统的通用做法）
df["win_rate_adjusted"] = (df["win"] + 0.5 * df["tie"]) / (df["win"] + df["loss"] + df["tie"])
```

`win_rate_adjusted` 才是榜单实际使用的指标。`0.5 * tie` 的好处是：当模型质量接近时，judge 输出大量平局而不会让两边都"看起来很差"。

### 2.8 决定性的细节：temperature=0 和 swap 评估

[common.py:L167](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/common.py#L167) 写得明明白白：judge 调用 OpenAI 时强制 `temperature=0, max_tokens=2048`。这是为了**降低同一题多次跑出不同结果的概率**。但 temperature=0 也不能消除随机性（OpenAI 后端不保证完全 deterministic），所以 MT-Bench 的标准做法是每对 (model_1, model_2) **跑两次，A/B 顺序对调一次**——这就是 [show_result.py:L49](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/show_result.py#L49) 的 `g1_winner` 和 `g2_winner`。如果两次结论不一致，就判 tie（[show_result.py:L64](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/show_result.py#L64)）。这是工程上对 position bias 最朴素也最有效的对冲。

### 2.9 被测模型生成时的 temperature：按类别分配

容易被忽视的一个细节：**被测模型生成回答时**的 temperature 也是按类别配置的，不是统一 0。看 [common.py:L40-50](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/common.py#L40)：

```python
# 来自 FastChat / Apache 2.0
temperature_config = {
    "writing": 0.7, "roleplay": 0.7,        # 创意类：高温度，鼓励多样性
    "extraction": 0.0, "math": 0.0,          # 抽取/数学：必须确定
    "coding": 0.0, "reasoning": 0.0,         # 代码/推理：必须确定
    "stem": 0.1, "humanities": 0.1,          # 知识类：略带温度
    "arena-hard-200": 0.0,
}
```

**为什么这么设计**：写作题如果都用 temperature=0，模型会输出非常套路化的回答，judge 给的分数差异化不出来；而 math/coding 用 0.7 会引入大量随机错误，掩盖了模型真实能力。这背后的隐含假设是"创意题靠均值，确定题靠最优"——但也意味着每次跑 MT-Bench 的写作部分会有不可忽略的 variance。

### 2.10 完整成本估算

把上面的所有细节加起来，跑一次完整 MT-Bench pairwise-baseline（被测模型 vs gpt-3.5-turbo）的开销：

- 80 道题 × 2 轮 = 160 次"被测模型"生成
- 80 道题 × 2 轮 × 2 个 judge prompt（default + math）= 实际只走 4 类共 ~160 次 judge 调用
- 每对 (model, baseline) 跑 swap 双向 → judge 调用翻倍 ~320 次
- 单次 GPT-4 judge 输入 ~1500 tokens、输出 ~300 tokens，按 GPT-4 单价折合 0.04-0.06 美金
- 总成本约 **15-20 美金 / 模型 / 次**

如果你做 N=10 次 ablation 实验，这就是 200 美金。所以"小成本可重跑"本身就是 LLM-as-Judge benchmark 的隐性约束。

## 三、Part B：AlpacaEval 怎么把 judge 做得"更像人"

### 3.1 与 MT-Bench 的差异

| 维度 | MT-Bench | AlpacaEval |
|------|----------|------------|
| 题目数 | 80 | 805 |
| 轮数 | 2 轮 | 1 轮 |
| 题型 | 命题式（写一篇游记） | 真实指令（怎么修我的简历） |
| 默认 judge | GPT-4 | gpt-4-1106-preview（"weighted"）|
| 输出格式 | `[[A]]`/`Rating` 文本 | **token logprob**（M/m）|
| 主指标 | 1-10 平均分 + win_rate_adjusted | win_rate + length-controlled win rate |

AlpacaEval 的 805 道题来自真实用户对话日志，平均长度更短、更"日常"，因此 single-turn 就够。它的真正贡献是**两件事**：用 logprob 做软投票，以及用 GLM 校正长度偏好。

### 3.2 默认 annotator 是 logprob 软投票，不是 hard A/B

打开 [constants.py:L40-42](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/constants.py#L40)：

```python
# 来自 alpaca_eval / Apache 2.0
ANNOTATOR_CONFIG_AE1 = "alpaca_eval_gpt4"             # AlpacaEval 1.0：硬选
ANNOTATOR_CONFIG_AE2 = "weighted_alpaca_eval_gpt4_turbo"  # AlpacaEval 2.0：软投票
DEFAULT_ANNOTATOR_CONFIG = ANNOTATOR_CONFIG_AE2 if IS_ALPACA_EVAL_2 else ANNOTATOR_CONFIG_AE1
```

而 `weighted_alpaca_eval_gpt4_turbo` 的核心配置是：

```yaml
# [weighted_alpaca_eval_gpt4_turbo/configs.yaml:L1-13] 来自 alpaca_eval / Apache 2.0
completions_kwargs:
  model_name: "gpt-4-1106-preview"
  max_tokens: 1            # 只生成 1 个 token：m 或 M
  logprobs: true           # 拿到 top-5 候选 token 的 logprob
  top_logprobs: 5
fn_completion_parser: "logprob_parser"
completion_parser_kwargs:
  numerator_token: "m"     # m 代表偏好 output_1
  denominator_tokens: ["m", "M"]
  is_binarize: false       # 关键：不二值化，直接用概率
```

也就是说，judge 不是被强迫"非黑即白"地选 A 或 B，而是**返回一个 [0, 1] 的偏好概率**，由 token 的 softmax 概率算出。这等价于"软投票"——judge 觉得 A 略好但不确定时，输出 0.6 而不是 1。最终 win rate 是这些软概率的均值。

### 3.3 logprob_parser 的实现：softmax over {m, M}

[completion_parsers.py:L275-305](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/completion_parsers.py#L275)：

```python
# 来自 alpaca_eval / Apache 2.0
def single_logprob_parser(top_logprobs):
    # 只保留 {numerator_token} ∪ {denominator_tokens} 中的 token
    map_tokens_to_logprobs = {
        t["token"]: t["logprob"] for t in top_logprobs
        if t["token"] in denominator_tokens + [numerator_token]
    }
    # 分子：偏好 output_1 的 logprob
    baseline_logprob = map_tokens_to_logprobs.get(numerator_token, float("-inf"))
    # 分母：所有候选 token 的 logsumexp = log(P(m) + P(M))
    denominator_logprob = logsumexp([map_tokens_to_logprobs.get(t, -inf) for t in denominator_tokens])
    # softmax 概率：P(m | {m, M})
    out_logprob = baseline_logprob - denominator_logprob
    probability = np.exp(out_logprob)
    # 用 [1, 2] 编码偏好：1 = 完全偏好 output_1，2 = 完全偏好 output_2
    return 2 - probability
```

软投票的好处：在 GPT-4 真的"摇摆"的题上（logprob(m) ≈ logprob(M)），分数会自然平滑到 1.5；而在它非常确定时，会接近 1 或 2。比纯硬投票降低了 variance。

### 3.4 Position bias：随机交换 A/B 顺序

GPT-4 系列对 prompt 中先出现的回答有系统性偏好（论文里 GPT-4 偏 A 的概率约 60%）。AlpacaEval 用 `RandomSwitchTwoColumnsProcessor` 在每条样本上独立做随机翻转：

```python
# [processors.py:L122-130] 来自 alpaca_eval / Apache 2.0
df_to_annotate[self._switch_column] = df_to_annotate.apply(
    # 用 instruction 内容做 seed，保证可复现且 swap 决策不会全相关
    lambda x: utils.random_seeded_choice(
        seed=self._switch_column + "".join(x[self.random_seed_columns]) + str(self.seed),
        choices=[False, True],
    ),
    axis=1,
)
return self._switch_or_unswitch(df_to_annotate, is_switch=True)  # 翻转 output_1 / output_2
```

后处理时再翻回来（[processors.py:L132-137](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/processors.py#L132)），保证 `preference` 字段的语义始终对应"原始的 output_1 vs output_2"。注意：FastChat 用"双跑互换"，AlpacaEval 用"单跑随机"——前者贵两倍，后者依靠样本量（805 道）做平均消除偏差。

### 3.5 Length bias：GPT-4 偏爱长回答怎么办

LC-WR 论文（[arXiv:2404.04475](https://arxiv.org/abs/2404.04475)）测出来：GPT-4 的偏好与回答长度的相关系数约 0.68。换句话说，让模型多说几句废话就能涨 win rate——这显然不是我们想要的"对齐质量"。AlpacaEval 2.0 的解法是**逻辑回归校正**。

核心公式定义在 [glm_winrate.py:L21-37](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/metrics/glm_winrate.py#L21)：

```python
# 来自 alpaca_eval / Apache 2.0
GLM_INFO = {
    "length_controlled_v1": {
        # 三个特征：长度差（tanh 压缩）+ 题目难度 + 模型固有质量
        "formula": "np.tanh(std_delta_len) + instruction_difficulty + not_gamed_baseline.astype(float) - 1",
        "regularize_to_baseline_lambda": 0.2,
        "kwargs": {"n_splits": 5},
    },
}
```

特征拆解：

- **`std_delta_len`**：标准化的"output_1 长度 − output_2 长度"，再过 tanh 压到 [−1, 1]，避免极端长度主导回归。
- **`instruction_difficulty`**：每道题的固有难度，预先在 `df_gamed.csv` 里算好（见 [glm_winrate.py:L218-228](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/metrics/glm_winrate.py#L218)）。
- **`not_gamed_baseline`**：模型自身的"非长度因素质量"，这是要拟合的目标系数。

### 3.6 LC-WR 的真正含义：把长度差扣到 0 重新预测

回归拟合完之后，预测胜率时**把 `std_delta_len` 强制设为 0**（[glm_winrate.py:L242-243](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/metrics/glm_winrate.py#L242)）：

```python
# 来自 alpaca_eval / Apache 2.0
df_test = df[["instruction_difficulty", "not_gamed_baseline"]].copy()
df_test["std_delta_len"] = 0   # 反事实：假设两边长度一样，会怎么样
```

再代回 logistic 模型，得到的就是 length-controlled win rate。**直观理解**：LC-WR 回答的不是"模型 X 实际胜率多少"，而是"如果模型 X 写得跟 baseline 一样长，胜率会是多少"。这把"靠堆字数"的策略钉死在地。

最终输出（[glm_winrate.py:L117-118](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/metrics/glm_winrate.py#L117)）：

```python
# 来自 alpaca_eval / Apache 2.0
metrics["length_controlled_winrate"] = predicted_preferences.mean() * 100
metrics["lc_standard_error"] = pd.Series(predicted_preferences).sem() * 100
```

LC-WR 论文中实测：与 ChatBot Arena 人类偏好的 Spearman 相关系数从朴素 win rate 的 0.93 提升到 0.98。这是个"工程小改动撬动榜单可信度"的经典案例。

### 3.7 GLM 训练机制：为什么用 LogisticRegressionCV

很多人会以为 LC-WR 就是"用线性回归把长度因素消掉"。但实际上 [glm_winrate.py:L303-360](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/metrics/glm_winrate.py#L303) 用的是**带 L1 正则的 LogisticRegressionCV + 5-fold CV**，主要原因有三：

1. **preference 是 [0, 1] 软标签**（不是 0/1 硬标签），需要把 sklearn 的 logloss 改造成"y 表示 P(label=1)"。代码里通过把同一行复制成两份（一份 y=1 权重为 P，一份 y=0 权重为 1-P）的 trick 实现。
2. **L1 正则 + liblinear**：让 instruction_difficulty 这种高维稀疏特征自动选择。
3. **GroupKFold by 样本 index**：保证一个 instruction 的两份"复制行"不被分到不同 fold，否则会数据泄露。

这套实现细节决定了：**直接用 numpy 自己写一个简化版 LC-WR 是错的**——少了正则就过拟合（805 题相对于特征维度其实不算大），少了软标签就丢失 logprob 信号。

### 3.8 极端变化告警：GLM 没拟合好的兜底

[glm_winrate.py:L186-193](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/metrics/glm_winrate.py#L186) 还有个有趣的 sanity check：

```python
# 来自 alpaca_eval / Apache 2.0
def get_is_extreme_changes(prev_winrate, new_winrate, abs_diff=10, rel_diff=4, ...):
    # 如果 LC-WR 与 raw WR 差超过 10 分或 4 倍，警告：很可能 GLM 没拟合好
    too_small = new_winrate < min(prev_winrate - (prev_winrate / rel_diff), prev_winrate - abs_diff)
    too_large = new_winrate > max(prev_winrate + ((100 - prev_winrate) / rel_diff), prev_winrate + abs_diff)
    return (too_small and min_warn) or (too_large and max_warn)
```

实际效果：当某个模型样本量过少、或长度分布过于极端（比如总输出几千字），GLM 会算出离谱的 LC-WR，告警就触发。**给做对齐的同学的提醒**：如果你训练出来的模型 LC-WR 比 raw 高了 30 分，先怀疑数据，再怀疑 judge。

### 3.9 Human agreement：怎么验证 judge 与人对得上

LLM-as-Judge 最朴素的质疑就是："凭什么相信 GPT-4 的判决和人一致？"AlpacaEval 论文给出的答案是**公开数据集 + 相关性测试**：

1. **构造数据集**：让人类标注员对同一批 (prompt, output_1, output_2) 做偏好判决，作为 ground truth。
2. **跑 judge**：用 GPT-4-Turbo annotator 对同一批样本输出 preference。
3. **计算一致性**：统计 judge 与人类多数投票一致的比例（agreement rate），以及与人类胜率的 Spearman 相关系数。

LC-WR 论文实测的关键数字：

- 人类标注员之间互相同意率约 **65%**（也就是说 35% 的题人和人都吵不出结论）。
- GPT-4 judge 与人类多数投票的同意率约 **63%**——已经接近"另一个人"的水平。
- 在 AlpacaEval 2 + LC-WR 加持下，与 ChatBot Arena Elo 的 Spearman 相关 **0.98**，是当前自动 benchmark 的天花板之一。

**这意味着什么**：单条样本上 GPT-4 的判决很可能与你不同（37% 的概率），但**在 805 题平均后**，整体排名与人类共识高度一致。所以 LLM-as-Judge 的可信度建立在 **大数定律 + 多 bias 校正** 上，而非"judge 永远比你聪明"。

### 3.10 自定义 annotator：换一个更便宜的 judge

如果不想烧 GPT-4 钱，AlpacaEval 支持任何符合接口的 annotator。看 [evaluators_configs/](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs) 目录就能发现 30+ 个开箱即用配置：`weighted_alpaca_eval_vllm_llama3_70b`、`claude_3_opus_ranking`、`gpt-3.5-turbo-1106_ranking` 等等。每个配置都是一个 yaml + 可能的 prompt template。

替换 judge 的代价：

- **便宜的 judge（Claude Haiku、Llama-3 70B）**：成本降 5-10 倍，但 human agreement 通常掉 5-10 个点，对 7B vs 13B 的细粒度比较已经分辨不出。
- **同等价位但不同家族（Claude 3 Opus vs GPT-4）**：用于做 cross-family 验证，避免 self-bias。
- **小模型 judge（Llama-3 8B）**：仅在简单题上勉强能用，复杂题 < 50% 一致性，基本不可用。

经验法则：如果你做严肃 ablation，至少跑两个不同家族的 judge 取交集；如果只是看 PR 是不是变差了，便宜 judge + LC-WR 的 CI 信号就够了。

### 3.11 与 MT-Bench 的对比小结

| 设计点 | MT-Bench | AlpacaEval 2.0 | 谁更合适？ |
|--------|----------|---------------|------------|
| Position bias | swap 双跑取交集 | 单跑随机翻转 | 题少→swap，题多→random |
| Judge 输出 | 文本 `[[A]]` + 正则 | logprob 软投票 | logprob 信噪比更高 |
| Length bias | 仅在 prompt 里要求 | LC-WR 回归校正 | LC-WR 严格更好 |
| 多轮 | 是（设计目标） | 否 | MT-Bench 独占 |
| 样本量 | 80 | 805 | AE2 更稳 |
| 算分 | 1-10 平均 + win rate | win rate + LC-WR | 互补，不替代 |

实践建议：**两个都跑**。MT-Bench 抓多轮和细分类别能力，AlpacaEval 抓样本量稳定性和长度公平性。如果只能选一个用于自动化 CI，选 AlpacaEval 2 + LC-WR。

## 四、苏格拉底时刻

> **Q1**：用 GPT-4 当 judge 评 GPT-4 自己有什么风险？AlpacaEval 默认 judge 是 GPT-4-Turbo，被测模型也常常是 GPT-4 family，这种"自评"会不会高估自己？

> **Q2**：如果你只有 7B 的开源 judge（比如 Llama-3-70B-Instruct），它能在哪些题上替代 GPT-4？哪些题上一定会失败？提示：看 [evaluators_configs/](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs) 里的 `weighted_alpaca_eval_vllm_llama3_70b`。

> **Q3**：FastChat 的 pairwise 模式跑 swap 双向比较（贵 2 倍），AlpacaEval 单跑随机翻转（便宜但更依赖样本量）。如果你只有 200 道题预算，会怎么权衡？

> **Q4**：为什么 LC-WR 比朴素 win rate 更可信？设想你的对齐数据里塞了大量啰嗦的回答，朴素 win rate 上涨 5 个点，你会高兴吗？

> **Q5**：MT-Bench 用 1-10 单数评分，AlpacaEval 用 token logprob 软投票。如果 judge 模型给同样的回答打 9 分和 9.1 分有没有意义？再细的分辨率值不值得？

> **Q6**：判 tie 是工程兜底还是真实信号？看 [common.py:L28](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/common.py#L28) 的 `TIE_DELTA = 0.1`——这意味着两个 1-10 分的回答只要差距 ≤ 0.1 就算平。这个阈值怎么选才合理？

### 实战清单：对齐迭代时该看哪些指标

把 LLM-as-Judge 用到日常对齐迭代里时，下面这些信号比"win rate 一个数字"更值得每天盯：

1. **win + tie + loss 三件套独立看**：tie 涨说明区分度下降，loss 涨说明真的退化。
2. **judge 失败率**：解析失败（`score = -1` / `winner = "error"`）超过 2% 要排查输出格式。
3. **LC-WR 与 raw WR 的差**：差距大 = 模型靠长度刷分，需要在数据侧约束 token budget。
4. **single 模式下 8 类别雷达图**：MT-Bench 单类雷达比平均分更能定位"是哪类能力退化"。
5. **同一 PR 跑 3 次取标准差**：variance > 1 分（10 分制）说明评估本身不可信，先稳住 judge 再调模型。
6. **swap 不一致率**：FastChat 双跑结论翻转的比例 > 15% 说明被测模型质量与 baseline 接近，需要更难的 benchmark。

## 五、面试考点

1. **Self-bias / Self-enhancement bias**：judge 模型对"自己风格"的回答有偏好。GPT-4 倾向于给 GPT 风格的回答更高分。MT-Bench 论文做过实验，模型 vs 自己当 judge 的对手时，胜率虚高 ~5 分。**对齐时的隐患**：如果你用 GPT-4 当 SFT 老师 + 当 judge，整个训练-评估闭环都被同一个 model 的 prior 偏置。

2. **Verbosity bias / Length bias**：长回答更容易被打高分。原因是 judge 在 attention 上更容易"找到亮点"。MT-Bench 的对策是 prompt 里写一句"do not allow length to influence"，AlpacaEval 的对策是 LC-WR 回归。

3. **Position bias**：在 pairwise prompt 里先出现的回答更容易赢。FastChat 用 swap 双跑取交集（[show_result.py:L64](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/show_result.py#L64)），AlpacaEval 用随机翻转 + 大样本平均。

4. **Majority bias / Sycophancy**：judge 在某些问题上倾向于"两边都说得对"，输出 tie。这不是真的 tie，是判断不出来。看到 tie 率超过 30% 就要警惕——可能是题目区分度太低，或两个模型都没能产生足够的差异化输出。

5. **Judge 不稳定（Variance）**：温度=0 也不保证完全 deterministic，OpenAI 后端会引入数值层面的不确定性。**实践折衷**：要么 majority vote（同一对样本跑 3 次取多数），要么把 score 改成 logprob 软投票（AlpacaEval 2 的做法），后者用同样 1 次调用得到了更平滑的信号。

6. **Refusal bias**：judge 模型自己有安全策略，会在拒答类问题上系统性偏好"礼貌拒答"的一方，即便另一方提供了真实有用的信息。如果你的对齐目标是"helpful first"，在涉敏类目上 LLM judge 与人类共识会显著背离。

7. **Format bias**：带 markdown 列表/标题的回答平均比纯文本得分高 ~3-5 分。这与 length bias 相关但不完全重叠，LC-WR 不能完全消除。如果做格式化对齐（让模型多用 markdown），要警惕这部分涨分的真实性。

## 六、推荐资源

- **源仓库**：
  - [lm-sys/FastChat](https://github.com/lm-sys/FastChat)（Apache 2.0，本文 Part A 全部代码引用来源）
  - [tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval)（Apache 2.0，本文 Part B 全部代码引用来源）
- **论文**：
  - MT-Bench / Chatbot Arena：[Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
  - AlpacaEval / AlpacaFarm：[AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback](https://arxiv.org/abs/2305.14387)
  - Length-Controlled WR：[Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators](https://arxiv.org/abs/2404.04475)
- **延伸**：
  - 站内 [/engineering/evaluation](/engineering/evaluation)（评估全景）
  - 站内 [/training/alignment](/training/alignment)（RLHF / DPO 怎么消费这些榜单信号）
  - Chatbot Arena 实时榜单：[lmarena.ai](https://lmarena.ai/)，与 LLM-as-Judge 互为参照系
