---
title: 深度剖析 lm-evaluation-harness——MMLU 的 likelihood 评估机制
description: 为什么 MMLU 不让模型生成 A/B/C/D，而是计算 4 个选项的对数似然——从源码看评估的工程契约
topics: [evaluation, mmlu, lm-eval-harness, loglikelihood, multiple-choice, few-shot]
prereqs: [/engineering/evaluation, /architecture/decoding]
---

# 深度剖析 lm-evaluation-harness——MMLU 的 likelihood 评估机制

::: info 一句话总结
MMLU 不让模型「生成」答案，而是把题面 + 4 个选项各拼一次，比较 4 条 `log P(choice | question)`，谁最大就是模型的「选择」——这是一种把分类任务伪装成语言建模任务、用 1 次前向打 4 次「假生成」的工程契约。
:::

::: tip 来源声明
本文基于 [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)（MIT License）的 `main` 分支源码，所有引用均为重写过的关键骨架，请以原仓库为准。配套论文：[A framework for few-shot language model evaluation](https://arxiv.org/abs/2405.14782)。
:::

---

## 在大模型体系中的位置

[engineering/evaluation.md](/engineering/evaluation) 已经在概念层讲过 MMLU 是什么——57 个学科 × 14042 道四选一选择题，号称「大模型高考」。但读完那一篇，多数人还有一个隐隐的疑惑：

> **既然是选择题，为什么不直接让模型生成「A」「B」「C」「D」中的一个字符，再比对答案？**

答案藏在工程层。lm-evaluation-harness（下称 lm-eval-harness 或 harness）把所有评估任务划成两个二分世界：

| 世界 | OutputType | 代表任务 | 模型调用 |
|---|---|---|---|
| **likelihood-based** | `loglikelihood` / `multiple_choice` / `loglikelihood_rolling` | MMLU、HellaSwag、ARC、PIQA、Winogrande、Lambada | 1 次 forward 算 logits，求 sum(log p)，**不解码** |
| **generation-based** | `generate_until` | GSM8K、HumanEval、MATH、TriviaQA（开放问答） | 自回归 decode 直到 stop token |

MMLU 走的是 `multiple_choice` 这一路。这条路有三个性质：
1. **比 generation 快 5-20 倍**——MMLU 选项平均才 1-3 个 token，4 选项加起来不过 ~10 token，全部并行算 logit；而生成模式即使只生成 1 个字也要走整套自回归 KV-cache 路径。
2. **比 generation 公平**——不依赖模型「肯不肯听话输出 A/B/C/D」。基础模型（pretrain-only）几乎不会规范输出 `A`，但它的内部概率分布完全可以反映对四个选项的偏好。
3. **天然消除「解码温度」**——T、top-p、top-k 完全无关，结果**完全确定**。

::: warning 一个常见误解
很多人以为 `OUTPUT_TYPE = "multiple_choice"` 是单独一条评估通路。**不是**。它会被 `construct_requests` 拆成 N 条 `request_type="loglikelihood"` 的 `Instance`，最终走的还是 LM 的 `loglikelihood()` 接口。`multiple_choice` 只是 task 层的语义标签，不是 model 层的能力维度。证据见下文 [task.py:1374](#L1374-L1453) 的源码骨架。
:::

---

## 核心内容

### 1. Task / Instance / LM —— 三层抽象的工程契约

harness 的整套架构靠三个抽象类咬合：

```
Task        ↔   把数据集（doc）翻译成「请求」（Instance）
Instance    ↔   一个标准化的 LM 调用单元（带 request_type 标签）
LM          ↔   只懂三件事：loglikelihood / loglikelihood_rolling / generate_until
```

`Instance` 是契约的**数据格式**——所有任务最终都化为它：

::: details Instance 数据类骨架（重写自原文件）
```python
# /lm_eval/api/instance.py
OutputType = Literal[
    "loglikelihood", "loglikelihood_rolling",
    "generate_until", "multiple_choice",
]

@dataclass
class Instance:
    request_type: OutputType   # 决定下游派发到 LM 的哪个方法
    doc: dict                  # 原始样本（debug / process_results 用）
    arguments: tuple           # 真正喂给 LM 的参数：(context, continuation) 或 (context, gen_kwargs)
    idx: int                   # 在「同一题」内的下标（MMLU 一题 4 个 Instance, idx=0..3）
    resps: list                # LM 调用后回填的原始返回
```
:::

引用自 [api/instance.py:5-29](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/instance.py#L5-L29)。

注意 **`idx` 字段**——它就是为 MMLU 这种「一道题拆 4 条 request」准备的。后面计分时，会用 `idx` 把 4 个 logprob 重新对齐回同一道题。

`LM` 抽象侧的契约就更精炼了：

::: details LM 抽象方法（重写关键签名 + 中文注释）
```python
# /lm_eval/api/model.py
class LM(abc.ABC):
    @abc.abstractmethod
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """
        每个 Instance.args = (context, continuation)
        返回 (sum_logprob_of_continuation, is_greedy)
          - sum_logprob: 给定 context 时, continuation 每个 token 的 log-prob 求和
          - is_greedy:   每个位置 argmax 是否恰好等于 continuation 的真实 token
        关键：这条接口不调用 model.generate()！只算 forward 一次。
        """

    @abc.abstractmethod
    def generate_until(self, requests: list[Instance]) -> list[str]:
        """args = (context, gen_kwargs); 真正自回归生成 + stopping criteria"""
```
:::

引用自 [api/model.py:40-110](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/model.py#L40-L110)。

读到这里你应该意识到：**MMLU 的「四选一」会被压成 4 个 (context, continuation) 的二元组**——context 是题面，continuation 分别是 `" A"` / `" B"` / `" C"` / `" D"`（或选项原文，取决于 `doc_to_choice` 的写法）。

### 2. MMLU 的 YAML 怎么落到代码

harness 的 task 是声明式的，绝大多数任务只有一个 yaml。MMLU `default` 子集的「主模板」如下（完整文件见 [tasks/mmlu/default/_default_template_yaml](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_default_template_yaml#L1-L15)，仅 15 行）：

```yaml
dataset_path: cais/mmlu
test_split: test
fewshot_split: dev
fewshot_config:
  sampler: first_n        # 不打乱, 直接取 dev 集前 N 个
output_type: multiple_choice
doc_to_text: "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:"
doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: answer     # answer 字段是 0/1/2/3 的整数
metric_list:
  - metric: acc
```

每一个学科的 yaml（如 [mmlu_abstract_algebra.yaml](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/mmlu_abstract_algebra.yaml#L1-L7)）只是 `include` 这个模板，再覆盖 `dataset_name` 和 `description`：

```yaml
"include": "_default_template_yaml"
"dataset_name": "abstract_algebra"
"task": "mmlu_abstract_algebra"
"description": "The following are multiple choice questions ..."
```

57 个学科共 57 个 7 行 yaml，由 `_generate_configs.py` 一键生成（`SUBJECTS` 常量在 [_generate_configs.py:17-75](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/_generate_configs.py#L17-L75) 列出，对应 stem / humanities / social_sciences / other 四大类）。

::: tip 落到代码：四个核心字段对应什么
- `doc_to_text(doc) -> str`：把一条样本渲染成 prompt 模板。Jinja 表达式由 ConfigurableTask 解析。
- `doc_to_choice(doc) -> list[str]`：返回候选项。MMLU 这里写死成 `["A","B","C","D"]`，这意味着**模型实际被打分的 continuation 是字母 A/B/C/D，不是选项原文**——这一点埋了重要差异，下文会展开。
- `doc_to_target(doc) -> int|str`：金标。返回整数（choices 的下标）或字符串（必须命中 choices 之一）。
- `output_type: multiple_choice`：通知 framework 走 likelihood 流程。
:::

### 3. construct_requests —— 把一道题炸成 4 条 request

这是整个 likelihood 评估机制的核心代码。下面的骨架重写自 [api/task.py:1362-1453](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/task.py#L1362-L1453)，只保留 `multiple_choice` 分支：

```python
def construct_requests(self, doc, ctx, **kwargs):
    # ctx: 已经渲染好的 prompt 字符串（含 few-shot 前缀 + 当前题目 + "Answer:"）
    if self.OUTPUT_TYPE == "multiple_choice":
        choices = self.doc_to_choice(doc)               # ["A","B","C","D"]
        target_delimiter = self.config.target_delimiter  # 默认 " "（一个空格）

        # 关键：每个选项都是一条独立的 (context, continuation) request
        # arguments[i] = (ctx, " A") / (ctx, " B") / (ctx, " C") / (ctx, " D")
        arguments = [
            (ctx, f"{target_delimiter}{cont}") for cont in choices
        ]

    # 然后把 N 个 arguments 封装成 N 个 Instance, request_type 改写为 "loglikelihood"
    # 注意! request_type 不是 "multiple_choice"——multiple_choice 只是 Task 的标签，
    # 真正进入 LM 的请求都被「降级」为最基础的 loglikelihood 调用
    return [
        Instance(request_type="loglikelihood", doc=doc, arguments=arg, idx=i, **kwargs)
        for i, arg in enumerate(arguments)
    ]
```

::: warning target_delimiter 的细节
context 是 `"...Answer:"`，continuation 是 `" A"`（前导空格！）——这是因为 GPT 系 BPE tokenizer 把 `"Answer: A"` 编码成的 token 序列，和 `"Answer:" + " A"` 拼接后再编码出的序列必须一致。harness 通过 `_encode_pair` 处理「context 末尾空格」的边界，详见 [api/model.py:368-406](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/model.py#L368-L406)。**自己复现评估时，前导空格踩坑率 90%。**
:::

### 4. process_results —— 4 个 logprob 怎么变成「acc」和「acc_norm」

LM 跑完后，每条 Instance.resps 里多了一个 `(logprob, is_greedy)`。打分流程（重写自 [task.py:1489-1576](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/task.py#L1489-L1576)）：

```python
def process_results(self, doc, results):
    # results 是按 idx 排好序的 4 条 (logprob, is_greedy)
    lls, is_greedy = zip(*results)        # lls = [-2.31, -3.05, -1.87, -4.12]
    choices = self.doc_to_choice(doc)     # ["A","B","C","D"]

    completion_len = np.array([len(c) for c in choices])  # 字符长度: [1,1,1,1]
    byte_length    = np.array([len(c.encode("utf-8")) for c in choices])

    # 三种「最大似然选项」
    pred       = np.argmax(lls)                      # 原始 logprob 最大
    pred_norm  = np.argmax(lls / completion_len)     # 按字符数归一化
    pred_byte  = np.argmax(lls / byte_length)        # 按字节数归一化

    gold = self.doc_to_target(doc)                   # 0/1/2/3
    return {
        "acc":      1.0 if pred == gold else 0.0,
        "acc_norm": 1.0 if pred_norm == gold else 0.0,
    }
```

::: tip acc vs acc_norm 的本质
- `acc`：直接比 4 条 sum(log p)。当所有选项 token 数一样（MMLU 的 A/B/C/D 都是 1 个 token），acc 和 acc_norm 几乎等价。
- `acc_norm`：把 sum(log p) 除以**字符长度**，得到「平均每字符的 log-prob」。**只在选项原文长度差异大时有意义**，例如 HellaSwag 的 4 个续写句长度从 30 到 100 字符不等——不归一化的话长答案天然吃亏（更多 token 相乘 ⇒ logprob 更小）。

所以你会看到很多论文 MMLU 报 `acc`，HellaSwag/ARC 报 `acc_norm`。这是有工程原因的，不是随机选的。

**坑**：如果 task 用了 `doc_to_choice: ["A","B","C","D"]`（像 MMLU），acc_norm 等于在用 4 个 1 字符做归一化分母——除一个常数没意义。**真正的 MMLU 评估应该看 `acc`。**
:::

### 5. evaluator 的批量派发

`simple_evaluate` 把所有 task 的 instances 拍扁后按 `request_type` 分桶、批量派发——见 [evaluator.py:550-590](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/evaluator.py#L550-L590)：

```python
# 收集所有 task 的所有 instances，按 request_type 聚合
for instance in task.instances:
    requests[instance.request_type].append(instance)
    # MMLU 的 instance.request_type 全是 "loglikelihood"

# 按桶批量调用
for reqtype, reqs in requests.items():
    # 这一行就是 harness 的「魔法」：用字符串反射调用 LM 的对应方法
    resps = getattr(lm, reqtype)(reqs)   # = lm.loglikelihood(reqs)
    for x, req in zip(resps, reqs):
        req.resps.append(x)
```

这就是为什么 harness 接入新模型只需要实现三个方法——所有 task 都通过 `request_type` 字符串路由到固定的三条接口之一。

### 6. 为什么 likelihood 比 generation 快这么多

::: details 一道 MMLU 题的算力对比
**Likelihood 版本（4 个选项，A/B/C/D 各 1 token）**

- 准备 4 条 input：`prompt + " A"`, `prompt + " B"`, ...
- 1 次 batched forward，输出 logits
- 取最后一个位置（也就是 A/B/C/D 那个 token 位）的 logits, gather 出对应 token 的 log-prob
- **总 FLOP ≈ 1 次 prefill**（batch=4，但 KV 共享前缀的实现可以让 prompt 只算一次，详见 vLLM/SGLang）

**Generation 版本（让模型生成 "A"）**

- 1 次 prefill 走 prompt
- 1 次 decode 步生成首 token
- 还得做 stopping criteria 判断、tokenizer 后处理、字符串匹配
- 如果模型不老实，输出 `"The answer is A."`，还要正则提取
- **总 FLOP ≈ 1 次 prefill + 1 次 decode + 后处理**

对短选项，两者算力接近；但 likelihood 模式有 3 个工程优势：
1. **完全确定**——同一模型同一权重必然得出相同结果，方便 CI、回归测试。
2. **batch 友好**——所有 4N 条 request 同型 padding，一发打完。
3. **base model 友好**——pretrain-only 模型几乎不会乖乖输出 "A"，但它的内部分布完全可用。
:::

::: tip 数据感
- MMLU `test_split` 共 14042 题 × 4 候选 = 56168 条 loglikelihood request
- 5-shot（默认）：每条 prompt 长度约 1500-3000 tokens，主要被 few-shot 撑大
- A100 80G 跑 7B 模型 + 5-shot MMLU：**约 30-60 分钟**（依赖 batch_size 与 KV cache 命中率）
- 同模型如果切到 generation 模式（`mmlu_generative`，存在于 [tasks/mmlu/generative/](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/generative)）：约 **2-4 倍**时间，因为还得做自回归 + 字符匹配。
:::

### 7. few-shot 是怎么组装到 ctx 里的

MMLU 默认是 **5-shot**——从 `dev` split 取 5 个示例 + 它们的标准答案，拼到 test 题目前面。`fewshot_config.sampler: first_n` 表示**直接取 dev 集前 5 条，不打乱、不随机**。这是有意为之：

::: warning few-shot 顺序的影响
论文与社区已多次证明：few-shot 示例的顺序、构造、平衡性对最终 acc 有 **±2~5%** 的波动。`first_n` 的好处是**完全可复现**——同一份 dataset 切片 + 同一个模板，永远算出同一份 prompt。社区共识是：MMLU 的 5-shot 一定要用 first_n，否则跨论文不可比。

但**坑也大**：dev 集如果质量不均匀（比如前 5 条都是同一答案 "A"），会引入位置偏差。harness 的处理是 trust the dataset——dev split 是 dataset 作者排好的，不背锅。
:::

few-shot 拼装的相关代码骨架（重写自 [task.py:980-1010](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/task.py#L980-L1010)）：

```python
# fewshot_split=dev 的样本
fewshot_docs = self.dataset[self.config.fewshot_split]

# 用 first_n sampler 取 N 个
for fs_doc in self.sampler.sample(n=num_fewshot, eval_doc=doc):
    q = self.doc_to_text(fs_doc, self.fewshot_cfg.doc_to_text)
    a = self.doc_to_target(fs_doc, self.fewshot_cfg.doc_to_target)
    # 注意 a 是金标的「显式答案」: "A" / "B" / ...
    messages += build_qa_turn(q, c=None, a=a, ...)

# 最后拼上当前 doc 的 question, 末尾停在 "Answer:"
# 这就是 ctx，传给 construct_requests
```

数学上 ctx 长这样：

$$
\text{ctx} = \underbrace{\text{description}}_{\text{『以下是关于...的多选题』}} + \underbrace{\sum_{i=1}^{5} (Q_i + \text{Answer: } A_i)}_{\text{5-shot examples}} + \underbrace{Q_{\text{test}} + \text{Answer:}}_{\text{当前题，停在冒号}}
$$

然后 4 条 continuation `" A"/" B"/" C"/" D"` 各自接到 ctx 后面打分。

### 8. registry——为什么写一个新 task 不用改 framework

harness 用装饰器注册一切。新增一个模型只需要：

```python
# /lm_eval/api/registry.py 重写要点
@register_model("my_model")
class MyModel(TemplateLM):
    def loglikelihood(self, requests): ...
    def generate_until(self, requests): ...
```

之后命令行 `--model my_model` 就能用。task 也类似——把一个 yaml 扔进 `lm_eval/tasks/<name>/`，TaskManager 启动时扫描全部 yaml 即注册。这是典型的「**约定优于配置**」工程范式，让贡献新 benchmark 的门槛低到只写一个 15 行 yaml。

参考 [api/registry.py:465+](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/registry.py#L465) 的 `register_model` 装饰器实现。

---

## 苏格拉底时刻

::: details Q1：如果 doc_to_choice 写成 `[choices[0], choices[1], ...]`（选项原文）而不是 `["A","B","C","D"]`，会发生什么？
两者**评估结果完全不一样**。`["A","B","C","D"]` 评估的是「模型在 Answer: 后该选哪个字母」——本质上还是分类。换成原文后，评估的是「模型对四段语义内容各自的语言模型概率」——更接近 HellaSwag 那种「续写哪条最自然」的范式。

更重要的是：**原文长度差异巨大**时，`acc` 会严重偏向短答案，必须切到 `acc_norm`。MMLU 选 A/B/C/D 字母，本质是把这道题彻底变成「字母分类」，规避了 `acc_norm` 的复杂性。
:::

::: details Q2：为什么有些任务（比如 ARC-c）`acc_norm` 反而比 `acc` 低？
`acc_norm` 除以字符长度，相当于在 `log P(choice | ctx)` 上加一个长度先验。如果模型本身就**偏好长答案**（很多 RLHF 后的模型有这个倾向），那除以长度反而把它「校正」过头了，正确率下降。

延伸：harness 还提供 `acc_bytes`（按字节归一化）、`acc_mutual_info`（用互信息归一化）。后者会**追加**一组 unconditional request `("", " A")` 来估 P(A)——开销加倍但更稳健。代码见 [task.py:1393-1405](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/task.py#L1393-L1405)。
:::

::: details Q3：如果模型只接受 chat template（如 Llama-3-Instruct），还能算 likelihood 吗？
能算，但**结果不再可比**。问题是：

1. chat template 把 prompt 包成 `<|user|>...<|assistant|>` 结构，token 序列被"系统消息"撑大；
2. 续写位置不再是裸的 "Answer:"，而是 `<|assistant|>` 之后；
3. 不同 chat template 改变了概率几何，跨模型直接比较 acc 不公平。

harness 的解：`--apply_chat_template` 开关 + `gen_prefix` 配置，强制让续写出现在某个固定 prefix 后。但社区**通常**约定：MMLU 报 base model 的非 chat 结果（更纯粹反映 pretrain 知识），instruct 模型用 generative 版的 MMLU 单独评。
:::

::: details Q4：MMLU 这种「打 4 个 logprob 取 argmax」是不是绕过了 RLHF 让模型「拒绝回答」的能力？
是的。likelihood 评估**强制**模型在 4 个选项中选一个——它无法表达「我不知道」。这是 MMLU 高分但模型仍然胡说八道的根因之一。OpenAI 的 SimpleQA、Hendrycks 等人后来的 MMLU-Pro 引入「以上都不是」选项，部分缓解了该问题。
:::

::: details Q5：4 条 logprob 本身的「绝对差距」有意义吗？
有，但 MMLU 不用。`brier_score`、`likelihood`（保留 lls 整向量）等 metric 才会用绝对值——可以用来评校准（calibration）。`acc` 只看 argmax，等于扔掉了大量信息。这也是 MMLU 不能直接用作 RM 训练信号的原因——它太"硬"了。
:::

---

## 面试考点

1. **lm-evaluation-harness 的三层抽象是什么？**——Task 把数据集翻译成 Instance；Instance 用 `request_type` 字段标识 LM 调用类型；LM 只暴露 `loglikelihood / loglikelihood_rolling / generate_until` 三个方法。新增模型只需实现这三个抽象方法。

2. **MMLU 的 OutputType 是 `multiple_choice`，但实际 LM 接口被调用的是什么？**——`loglikelihood`。`multiple_choice` 是 task 层的语义标签，`construct_requests` 会把一道题拆成 N 条 `request_type="loglikelihood"` 的 Instance。**这是高频陷阱题。**

3. **acc 和 acc_norm 的差别是什么？什么时候用哪个？**——acc 直接比原始 sum(log p)；acc_norm 除以字符长度做归一化。**只有当候选项长度差异显著时 acc_norm 才有意义**。MMLU 的选项是 A/B/C/D 字母（长度全为 1），所以 acc_norm 等于 acc，社区都报 acc。HellaSwag/ARC 选项是变长句子，社区报 acc_norm。

4. **常见踩坑**：自己复现 MMLU 时把 continuation 写成 `"A"` 而不是 `" A"`（少了前导空格）→ BPE tokenizer 切出来的 token 不一致 → 模型概率全错。修复：用 harness 的 `_encode_pair`，让它替你处理空格边界。

5. **few-shot 的 `first_n` sampler 为什么是默认？**——可复现性。MMLU 跨模型/跨论文比较的前提是同 prompt。随机采样会让方差吃掉模型差距（±2~5% acc 抖动很常见）。代价是：dev 集的样本顺序如果有偏（比如前 5 条都答 "A"），会引入系统性 position bias，但这不背锅给 harness。

6. **为什么 likelihood 比 generation 快？**——一次 forward vs 自回归 decode + 后处理。MMLU 选项长度仅 1 token 时，4 个选项一次 batch 前向就完事，不走 KV cache 的 decode 循环；generation 还要带 stopping criteria 和字符串匹配。

---

## 推荐资源

- 源仓库：[EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)（MIT）——本文所有代码骨架的来源
- 配套论文：[A framework for few-shot language model evaluation](https://arxiv.org/abs/2405.14782)（Biderman et al., 2024）
- MMLU 原始论文：[Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)（Hendrycks et al., ICLR 2021）
- 关键阅读路径建议：
  1. `lm_eval/api/instance.py` — 38 行，先看数据契约
  2. `lm_eval/api/model.py` 的 `LM` 抽象类 — 看三个方法的 docstring
  3. `lm_eval/tasks/mmlu/default/_default_template_yaml` — 15 行 yaml 落到一个完整 task
  4. `lm_eval/api/task.py` 的 `construct_requests` 与 `process_results` — 看 multiple_choice 怎么炸成 loglikelihood
  5. `lm_eval/evaluator.py` 的 `simple_evaluate` — 看主流程如何把请求按 request_type 派发
- 进一步阅读：[engineering/evaluation.md](/engineering/evaluation) 的「污染检测」「Chatbot Arena」「LLM-as-Judge」章节，本文是其工程视角补充。
