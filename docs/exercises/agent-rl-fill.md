---
title: Agent-RL 代码填空
description: Level 2-3 填空：格式奖励、可验证奖励、多轮轨迹 Mask、异步 Rollout、Composite Reward、Trajectory→Token 分配
topics: [fill-in, agent-rl, GRPO, reward-design, rule-based-reward, async-rollout, multi-turn, RLVR, RLVE, trajectory-mask]
prereqs: [/training/agent-rl, /exercises/dpo-grpo-fill, /exercises/ppo-fill]
---

# Agent-RL 代码填空

> 主线参考：[Agent-RL 训练范式](/training/agent-rl)。本章把 Agent-RL 区别于普通 RLHF 的几个工程零件挖出来填——格式奖励、多轮 mask、异步队列、composite reward。

::: tip 推荐做题顺序
1. 练习 1-2（rule-based reward 基础）→ 练习 5（composite 组装）
2. 练习 3（多轮 mask）→ 练习 6（trajectory→token 分配）
3. 练习 4（异步 rollout，最难，工程最重）

做完后建议跑一遍 [手撕 nano Agent-RL](/deep-dives/nano-agent-rl)。
:::

参考开源项目（外部链接）：

- [HuggingFace open-r1](https://github.com/huggingface/open-r1) — R1 复现的奖励函数集
- [hkust-nlp/simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason) — 极简 R1-Zero 复现
- [THUDM/slime](https://github.com/THUDM/slime) — 异步 Rollout 框架
- [volcengine/verl](https://github.com/volcengine/verl) — Single-Controller PPO/GRPO
- [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — Async PPO Trainer

---

## 练习 1：格式奖励（Format Reward, Level 2）

R1-Zero 的核心 trick 之一：用一个**纯 0/1 奖励**强制模型必须输出 `<think>...</think><answer>...</answer>` 这样的结构。结构对了拿 1 分，错了拿 0 分。完全不打分内容质量——但模型为了拿到这 1 分，会自己学会把推理过程写到 `<think>` 里。

```python
import re

def format_reward(completion: str) -> float:
    """
    检查 completion 是否严格符合 R1 的格式要求：
    - 以 <think> 开头（允许前面有空白）
    - 包含一对完整的 <think>...</think>
    - 紧跟一对 <answer>...</answer>
    - 中间允许任意空白
    
    返回 1.0（合格）或 0.0（不合格）。
    """
    # TODO 1: 写出匹配 <think>X</think>...<answer>Y</answer> 的正则
    #         X, Y 都允许任意字符（含换行），用非贪婪 .*?
    pattern = _____

    # TODO 2: 用 re.match（不是 search）+ re.DOTALL 来检查
    #         为什么用 match 不用 search？因为只允许开头匹配
    match = _____

    return 1.0 if match else 0.0


# ====== 测试 ======
ok = "<think>Let me compute 2+3</think><answer>5</answer>"
bad1 = "The answer is 5"                             # 完全没标签
bad2 = "<answer>5</answer><think>oops</think>"      # 顺序反了
bad3 = "<think>halfway"                              # 缺闭合标签
bad4 = "  <think>x</think>\n\n<answer>y</answer>"   # 前后空白：应该接受？

for c in [ok, bad1, bad2, bad3, bad4]:
    print(f"{format_reward(c):.1f}  | {c[:40]!r}")
```

::: details 提示
- 正则：`r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"`
- `re.DOTALL` 让 `.` 也能匹配换行符
- `re.match` 从字符串开头开始尝试匹配（隐式 `^`），但加 `$` 才能确保到结尾
- bad4 应该接受（1.0），因为我们允许首尾空白
:::

<details>
<summary>点击查看答案</summary>

```python
pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
match = re.match(pattern, completion, re.DOTALL)
```

**解析：**

格式奖励看似简单，是 R1-Zero 论文里最被低估的设计：

1. **0/1 奖励的"塑形作用"**：在 GRPO 里，`reward = 1` 比 `reward = 0` 在组内做 z-score 后会有正的 advantage，模型会被推向"产生格式正确的回答"。
2. **为什么不打分内容**：纯格式奖励让模型**先学会结构**，内容质量交给后续的 accuracy reward。如果一开始就用复合奖励，模型会因 reward sparsity 学不到任何东西。
3. **正则的非贪婪 `.*?`**：避免吞掉第二对 `<think>` 标签——如果模型重复输出多组标签，只匹配第一组。
4. **`re.DOTALL`**：思考过程经常跨行，缺这个 flag 会让所有多行 completion 都拿 0 分。
5. **首尾 `\s*` + `$`**：实际数据里 chat template 经常会带前导空格或末尾换行，必须容忍。

</details>

---

## 练习 2：可验证奖励（Verifiable Reward, Level 2）

数学题的奖励是 R1 / RLVR 范式的另一支柱：给定标准答案，从模型输出里抠出最后那个数，做**符号等价**判断（`1/2 == 0.5`、`\frac{1}{2} == 0.5`）。这就是 RLVR（Reinforcement Learning with Verifiable Reward）的 V。

```python
import re
from fractions import Fraction

def extract_boxed_answer(text: str) -> str | None:
    """从形如 '...所以答案是 \\boxed{42}' 的文本里抠出 '42'。"""
    # TODO 1: 找最后一个 \boxed{...}，返回大括号内的字符串
    #         为什么要"最后一个"？因为模型可能在思考过程中也写过 \boxed
    matches = _____
    return matches[-1] if matches else None


def normalize_to_fraction(s: str) -> Fraction | None:
    """把 '1/2'、'0.5'、'\\frac{1}{2}' 统一成 Fraction(1, 2)。"""
    s = s.strip().replace(r"\frac{", "").replace("}{", "/").replace("}", "")
    try:
        # TODO 2: 优先尝试 Fraction(s)（处理 "1/2"），失败再尝试 float
        return _____
    except (ValueError, ZeroDivisionError):
        return None


def accuracy_reward(completion: str, gold: str) -> float:
    """
    返回 1.0 / 0.0：
    - 1.0：completion 中最后一个 \boxed{} 的值在数值上等于 gold
    - 0.0：抠不出来 / 不相等
    """
    pred_str = extract_boxed_answer(completion)
    if pred_str is None:
        return 0.0
    pred = normalize_to_fraction(pred_str)
    gold_v = normalize_to_fraction(gold)

    # TODO 3: 都成功才比较；任意一个失败返回 0
    if _____:
        return 0.0
    return 1.0 if pred == gold_v else 0.0


# ====== 测试 ======
cases = [
    ("思考...所以\\boxed{0.5}", "1/2",     1.0),  # 跨表示等价
    ("\\boxed{\\frac{1}{2}}",   "0.5",     1.0),  # latex frac vs decimal
    ("\\boxed{2}\n\\boxed{3}",  "3",       1.0),  # 取最后一个
    ("没有答案",                "5",       0.0),  # 抠不出
    ("\\boxed{abc}",           "5",       0.0),  # 抠到了但不是数
]
for comp, gold, expected in cases:
    r = accuracy_reward(comp, gold)
    print(f"{r:.1f}  expected={expected:.1f}  | {comp[:30]!r}")
```

::: details 提示
- `re.findall(r"\\boxed\{([^{}]*)\}", text)` 抓所有 `\boxed{...}` 的内容
- `Fraction("1/2") == Fraction("0.5")`（True，因为 Fraction 把 0.5 转成 1/2）
- 失败链：先 `Fraction(s)` 试 `1/2` 这种，再 `Fraction(float(s))` 试 `0.5` 这种
- TODO 3：`pred is None or gold_v is None`
:::

<details>
<summary>点击查看答案</summary>

```python
# TODO 1
matches = re.findall(r"\\boxed\{([^{}]*)\}", text)

# TODO 2
try:
    return Fraction(s)
except ValueError:
    return Fraction(float(s))  # "0.5" → Fraction(1, 2)

# TODO 3
if pred is None or gold_v is None:
    return 0.0
```

**解析：**

生产级的 verifier（如 [open-r1](https://github.com/huggingface/open-r1) 的 `accuracy_reward` 用 `math_verify` + `latex2sympy2_extended`）做的事比这复杂得多——支持方程、单位、boxed 嵌套、LaTeX 各种乱七八糟的标记。但**核心三步永远是一样的**：

1. **Extract**：从 free-form completion 里抠出"答案 token"。`\boxed{}` 是数学题的事实标准，因为 LaTeX 渲染时它就是答案框。
2. **Normalize**：把不同表示法（分数 / 小数 / latex frac / 百分号）映射到同一个规范形式。`Fraction` 是最简单的"对数值做 canonical form"。
3. **Compare**：在规范形式下做精确等于（不是字符串等于）。

为什么 RLVR 是对齐范式的范式跃迁？因为 reward 的方差为 0、bias 为 0、cost 为 0。这三个性质 reward model 一个都满足不了。

</details>

---

## 练习 3：多轮轨迹的 Loss Mask（Level 2-3）

Agent rollout 与单轮聊天的最大不同：**轨迹包含交错的 user / assistant / tool 角色**。loss 只能算在 assistant 生成的 token 上——user / tool / system 的 token 是"observation"，模型不该被它们的内容惩罚或鼓励。

数据结构（一条轨迹）：
```python
trajectory = [
    {"role": "system",    "content": "You are a math agent."},
    {"role": "user",      "content": "What is 23 * 17?"},
    {"role": "assistant", "content": "Let me use the calculator. <call>mul(23,17)</call>"},
    {"role": "tool",      "content": "391"},
    {"role": "assistant", "content": "23 * 17 = 391"},
]
```

```python
def build_loss_mask(token_ids: list[int],
                   role_spans: list[tuple[str, int, int]]) -> list[int]:
    """
    根据 role_spans (role, start, end) 列表构造 loss mask。
    
    role_spans: 每条 (role, start, end) 表示 token_ids[start:end] 属于 role。
                role ∈ {"system", "user", "assistant", "tool"}
    
    返回长度等于 len(token_ids) 的 0/1 列表：
    - 1：该 token 计入 loss（assistant 生成的 token）
    - 0：mask 掉（system / user / tool 提供的 observation）
    """
    mask = [0] * len(token_ids)
    for role, start, end in role_spans:
        # TODO 1: 只有 assistant 段需要被训练
        if _____:
            for i in range(start, end):
                mask[i] = 1
    return mask


def shift_mask_for_clm(mask: list[int]) -> list[int]:
    """
    Causal LM 的 loss 形式是 logits[t] 预测 token[t+1]。
    所以 token[t] 是否计入 loss，取决于 mask[t+1]。
    
    返回 shifted_mask，长度 len(mask) - 1。
    """
    # TODO 2: shift left by 1（丢掉最后一个，因为没有"下一个 token"可预测）
    return _____


# ====== 测试 ======
# 一条轨迹被 tokenize 成 12 个 token：
# [sys=0..2] [usr=2..5] [asst=5..8] [tool=8..10] [asst=10..12]
spans = [
    ("system",    0, 2),
    ("user",      2, 5),
    ("assistant", 5, 8),
    ("tool",      8, 10),
    ("assistant", 10, 12),
]
ids = list(range(12))
m = build_loss_mask(ids, spans)
print("raw mask:    ", m)
# 期望: [0,0, 0,0,0, 1,1,1, 0,0, 1,1]

shifted = shift_mask_for_clm(m)
print("shifted mask:", shifted)
# 期望: [0, 0,0,0, 1,1,1, 0,0, 1,1] (长度 11)
# 解读：第 4 个 token (index 4, 最后一个 user token) 的 logits 预测第 5 个 (第一个 asst)，
#       这个 loss 要算！所以 shifted[4] = mask[5] = 1
print("loss tokens:", sum(shifted), "/ 11")
```

::: details 提示
- TODO 1：`role == "assistant"`
- TODO 2：`mask[1:]`（从下标 1 开始，丢掉 mask[0]，长度自动减 1）
:::

<details>
<summary>点击查看答案</summary>

```python
# TODO 1
if role == "assistant":
    for i in range(start, end):
        mask[i] = 1

# TODO 2
return mask[1:]
```

**解析：**

Agent rollout 训练里有三个非常容易踩的坑，这道题各覆盖一个：

1. **Tool 输出绝对不能算 loss**。tool 是环境返回的"事实"，模型如果被训练去预测 tool 的内容，等于在预测**未来的环境 stochasticity**——这会让 policy 变成"模仿环境"，而不是"决定如何调环境"。
2. **System / user 不算 loss**。理由类似：你给模型的 prompt 不是它的输出。
3. **Shift by 1 的方向**：Causal LM 是 `logits[t] → token[t+1]`。所以**当前 token 是否计入 loss，看的是它要预测的下一个 token 的 mask**。`shifted_mask[t] = mask[t+1]`。

工程上 [verl](https://github.com/volcengine/verl) 与 [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 还会在 mask 里再排除掉 padding token、BOS/EOS（视场景）、特殊 tool tag（避免模型学会输出 tool tag 但不输出真正的 tool name）。这些都是同一个 mask 框架的进一步精化。

</details>

---

## 练习 4：异步 Rollout 队列（Level 3）

Agent-RL 的最大工程瓶颈：**rollout 慢**。一次 rollout 要跑多轮，每轮要等 LLM 生成 + 等环境响应。如果 rollout 与 training 串行（rollout 完一组才训一步），GPU 利用率会跌到 30%。

解决方案：**异步队列**。Rollout worker 不停产生轨迹塞进队列；Trainer 从队列里捞一个 batch 就训一步。两者解耦，GPU 满载。

```python
import asyncio
import random
from collections import deque

class AsyncRolloutBuffer:
    """
    最小异步 rollout buffer。
    - rollout worker 调 put() 持续塞轨迹
    - trainer 调 get_batch() 阻塞等到攒够 batch_size 条
    - 用 max_age 防止 trainer 拿到过老的（off-policy 严重的）轨迹
    """
    def __init__(self, batch_size: int, max_capacity: int):
        self.batch_size = batch_size
        # TODO 1: 用 deque 加 maxlen=max_capacity，自动丢弃最老的轨迹
        self.buf = _____
        self.cond = asyncio.Condition()

    async def put(self, traj: dict):
        """rollout worker 调用：放进一条轨迹，唤醒 trainer。"""
        async with self.cond:
            self.buf.append(traj)
            # TODO 2: 唤醒所有在 wait 的 trainer 协程
            _____

    async def get_batch(self) -> list[dict]:
        """trainer 调用：阻塞直到有 batch_size 条数据可用。"""
        async with self.cond:
            # TODO 3: 等待条件满足。注意 wait_for 比裸 wait 更安全（防虚假唤醒）
            await self.cond.wait_for(lambda: _____)
            # 取出 batch_size 条最新的
            batch = [self.buf.popleft() for _ in range(self.batch_size)]
            return batch


# ====== 测试：模拟 rollout 与 trainer 协程并发 ======
async def rollout_worker(buf: AsyncRolloutBuffer, n: int):
    for i in range(n):
        await asyncio.sleep(random.uniform(0.01, 0.05))  # 模拟生成耗时
        await buf.put({"traj_id": i, "reward": random.random()})
        print(f"  [rollout] put traj {i}")

async def trainer_loop(buf: AsyncRolloutBuffer, n_steps: int):
    for step in range(n_steps):
        batch = await buf.get_batch()
        ids = [t["traj_id"] for t in batch]
        print(f"[trainer] step {step}: trained on {ids}")

async def main():
    buf = AsyncRolloutBuffer(batch_size=4, max_capacity=16)
    await asyncio.gather(
        rollout_worker(buf, n=12),
        trainer_loop(buf, n_steps=3),
    )

# asyncio.run(main())
```

::: details 提示
- TODO 1：`deque(maxlen=max_capacity)` —— 满了之后再 append 会自动丢最老的
- TODO 2：`self.cond.notify_all()`
- TODO 3：`len(self.buf) >= self.batch_size`
:::

<details>
<summary>点击查看答案</summary>

```python
# TODO 1
self.buf = deque(maxlen=max_capacity)

# TODO 2
self.cond.notify_all()

# TODO 3
await self.cond.wait_for(lambda: len(self.buf) >= self.batch_size)
```

**解析：**

这是 [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 的 `ppo_trainer_async.py` 与 [slime](https://github.com/THUDM/slime) 的 rollout buffer 的核心抽象，去掉了分布式 / 多进程的复杂性。三个关键设计点：

1. **`deque(maxlen=...)` 自动丢弃最老的**：如果 rollout 比 trainer 快太多，最老的轨迹会逐渐变得 off-policy（采样它的 policy 已经被更新过几次了）。直接丢掉比留着用更安全。生产框架会进一步**给每条 traj 标 policy version**，trainer 根据版本差决定要不要用。
2. **`notify_all` + `wait_for`**：不是用裸 `wait`，因为 asyncio 也有"虚假唤醒"风险（其实主要是为了防 bug）。`wait_for` 内部会循环检查条件，确保被唤醒后真的满足条件才返回。
3. **`Condition` 不是 `Queue`**：`Queue.get()` 是一条一条拿，凑 batch 要 N 次切换。`Condition + deque` 一次性拿一个 batch，**少 N 次协程上下文切换**。在 batch=128 时差异显著。

进阶问题（生产级要解决的）：
- **多 rollout worker**：用 `asyncio.gather` 启多个 worker 协程；用 `asyncio.Lock` 保护写。
- **跨进程**：用 `Ray.remote Actor` 或 `multiprocessing.Queue` 替代。
- **GPU 上的 model 同步**：worker 持有的 policy 落后于 trainer 多少版本？超过阈值要 `weight_sync()`。
- **Rollout 失败**：环境超时、tool 异常如何回收？参考 verl 的 `dispatch_mode`。

</details>

---

## 练习 5：Composite Reward 组装（Level 3）

实战里没有人只用一个奖励。生产 reward 通常长这样：

$$
r(x, y) = w_{\text{fmt}} \cdot r_{\text{fmt}} + w_{\text{acc}} \cdot r_{\text{acc}} + w_{\text{len}} \cdot r_{\text{len}} - w_{\text{rep}} \cdot r_{\text{rep}}
$$

每个 component 是独立函数（练习 1-2 是其中两个），主训练脚本负责加权组装。这道题填一个 reward composer。

```python
from typing import Callable

def cosine_length_reward(completion: str, gold: str, max_len: int = 512) -> float:
    """
    长度 cosine scaling：答错时短的扣分多，答对时短的得分多。
    L = len(completion); ratio = L / max_len ∈ [0, 1]
    
    答对 (acc=1.0): r = 1.0 - 0.5 * (1 - cos(π * ratio))   长 → 1.0,  短 → 0.0
                                                           （惩罚没思考就答对——可能是猜的）
    答错 (acc=0.0): r = -1.0 + 0.5 * (1 - cos(π * ratio))  长 → -1.0, 短 → 0.0
                                                           （惩罚长篇错答——浪费 token）
    """
    import math
    acc = accuracy_reward(completion, gold)  # 复用练习 2
    ratio = min(len(completion) / max_len, 1.0)
    cos_factor = 0.5 * (1 - math.cos(math.pi * ratio))

    # TODO 1: 按 acc 0/1 走两条公式
    if acc > 0.5:
        return _____
    else:
        return _____


def make_reward_fn(weights: dict[str, float],
                   funcs: dict[str, Callable]) -> Callable:
    """
    工厂函数：把多个奖励按权重组合成一个。
    
    weights: 名字 → 权重
    funcs:   名字 → 单个 reward function (返回标量)
    
    返回一个新函数 R(completion, gold) → 加权总分。
    """
    # TODO 2: 检查 weights 和 funcs 的 key 完全对应
    assert _____, "weights and funcs must have identical keys"

    def composite(completion: str, gold: str) -> dict:
        scores = {name: f(completion, gold) for name, f in funcs.items()}
        # TODO 3: 加权求和（注意这里返回 dict，方便日志记录每个分量）
        scores["total"] = _____
        return scores
    return composite


# ====== 测试 ======
funcs = {
    "fmt":     lambda c, g: format_reward(c),       # 练习 1
    "acc":     lambda c, g: accuracy_reward(c, g),  # 练习 2
    "cos_len": cosine_length_reward,
}
weights = {"fmt": 0.1, "acc": 1.0, "cos_len": 0.2}

R = make_reward_fn(weights, funcs)

c = "<think>23*17 = 23*17 = 391</think><answer>\\boxed{391}</answer>"
print(R(c, "391"))
# 期望：fmt=1.0, acc=1.0, cos_len≈0 (短文本对答错才扣，对答对接近 0), total≈1.1
```

::: details 提示
- TODO 1：答对 `1.0 - cos_factor`，答错 `-1.0 + cos_factor`
- TODO 2：`set(weights) == set(funcs)`
- TODO 3：`sum(weights[k] * scores[k] for k in weights)`
:::

<details>
<summary>点击查看答案</summary>

```python
# TODO 1
if acc > 0.5:
    return 1.0 - cos_factor
else:
    return -1.0 + cos_factor

# TODO 2
assert set(weights) == set(funcs)

# TODO 3
scores["total"] = sum(weights[k] * scores[k] for k in weights)
```

**解析：**

为什么要 cosine length scaling？这是 [open-r1](https://github.com/huggingface/open-r1)（`get_cosine_scaled_reward`）和 DAPO（[slime](https://github.com/THUDM/slime) 用的）的关键技巧——

- **不加长度信号**：模型容易**长篇大论但答错**（长 → 探索更多 → 更可能蒙到 reward signal），collapse 到长答案。
- **简单负长度**：模型学会**短而精**——但同时也学会**短而错**（短答案的方差太大）。
- **Cosine scaling**：让"短 + 对"成为局部最优，"长 + 错"是最大惩罚。这是一个**与正确性耦合的长度奖励**，比独立加项稳定得多。

返回 `dict` 而不是单个标量是工程上的小聪明：训练时只用 `total`，但 wandb 日志要记录每个分量——出问题时（比如 acc 一直 0），看分量曲线能瞬间定位是 reward 设计的锅还是 policy 没学会的锅。

</details>

---

## 练习 6：Trajectory → Token Reward 分配（Level 3）

Agent rollout 的轨迹奖励是**序列级别**的（"答对了 +1，答错了 0"）。但 PPO/GRPO 需要**每个 token 都有 advantage**才能算 loss。如何把一个标量 reward 分配到 T 个 token 上？

四种主流策略：

1. **Last-token only**：reward 全加到最后一个 assistant token。简单、稀疏、信号弱。
2. **Distribute uniform**：均摊到所有 assistant token。容易让模型刷长度。
3. **Distribute by KL**：按 ref / policy 的 KL 分布权重。SimPO/某些 R1 变体使用。
4. **Token-level critic**：单独训一个 value model。这就是经典 PPO，重。

```python
import torch

def assign_reward_last_token(seq_reward: float,
                              loss_mask: torch.Tensor) -> torch.Tensor:
    """
    策略 1：把 reward 全加到 mask=1 的最后一个位置。
    
    loss_mask: shape [T], 0/1 tensor
    返回:      shape [T], reward tensor
    """
    out = torch.zeros_like(loss_mask, dtype=torch.float32)
    # TODO 1: 找到 mask=1 的最后一个位置
    #         提示：mask.nonzero() 返回所有非零位置的下标
    nz = _____
    if len(nz) > 0:
        last_idx = nz[-1].item()
        out[last_idx] = seq_reward
    return out


def assign_reward_uniform(seq_reward: float,
                          loss_mask: torch.Tensor) -> torch.Tensor:
    """
    策略 2：均摊到所有 mask=1 的位置。
    """
    out = torch.zeros_like(loss_mask, dtype=torch.float32)
    n = loss_mask.sum().item()
    if n == 0:
        return out
    # TODO 2: 给所有 mask=1 的位置赋值 seq_reward / n
    out[_____] = seq_reward / n
    return out


def assign_reward_kl_weighted(seq_reward: float,
                               loss_mask: torch.Tensor,
                               kl_per_token: torch.Tensor) -> torch.Tensor:
    """
    策略 3：按 KL 加权分配。
    KL 大的 token = 模型自信偏离 ref policy，应该承担更多 reward (or punishment)。
    
    kl_per_token: shape [T], 每个 token 的 |KL(policy || ref)|
    """
    out = torch.zeros_like(loss_mask, dtype=torch.float32)
    # TODO 3: 取 mask=1 位置的 KL，归一化成权重，再乘以 seq_reward
    masked_kl = kl_per_token * loss_mask
    total = masked_kl.sum()
    if total < 1e-8:
        # KL 太小退化为 uniform
        return assign_reward_uniform(seq_reward, loss_mask)
    weights = _____
    return seq_reward * weights


# ====== 测试 ======
mask = torch.tensor([0, 0, 1, 1, 0, 1, 1, 1, 0])  # 5 个 assistant token
kl   = torch.tensor([0., 0., 0.1, 0.5, 0., 2.0, 0.1, 0.3, 0.])

print("last:    ", assign_reward_last_token(1.0, mask))
# 期望: [0,0,0,0,0,0,0,1,0]

print("uniform: ", assign_reward_uniform(1.0, mask))
# 期望: 5 个 mask=1 位置各为 0.2

print("kl_weighted:", assign_reward_kl_weighted(1.0, mask, kl))
# 期望: 第 5 位（kl=2.0）拿大头，约 0.667
```

::: details 提示
- TODO 1：`loss_mask.nonzero(as_tuple=False).flatten()` 或 `(loss_mask > 0).nonzero().flatten()`
- TODO 2：`out[loss_mask.bool()] = seq_reward / n`
- TODO 3：`weights = masked_kl / total`（这样 weights 在 mask=1 的位置上和为 1）
:::

<details>
<summary>点击查看答案</summary>

```python
# TODO 1
nz = (loss_mask > 0).nonzero().flatten()

# TODO 2
out[loss_mask.bool()] = seq_reward / n

# TODO 3
weights = masked_kl / total
```

**解析：**

这道题的本质是 **credit assignment**：一个轨迹的成功/失败到底要"归功"于哪些 token？四种策略的 trade-off：

| 策略 | 优点 | 缺点 | 用在哪 |
|------|------|------|--------|
| Last-token | 信号干净，没有歧义 | 极稀疏，长轨迹梯度方差大 | R1-Zero（GRPO 配 group 平均能容忍稀疏） |
| Uniform | 信号密集 | 鼓励长度 | 早期 PPO，少用 |
| KL-weighted | 把信用分给"敢做决定"的 token | 需要 ref model 算 KL，多花一份显存 | RLOO 的某些变体 |
| Token critic | 理论最优 | 训 value model 难、不稳 | 经典 PPO |

[GRPO 论文](https://arxiv.org/abs/2402.03300) 默认用 **last-token only**，靠 group-level baseline 的 z-score 来降稀疏 reward 的方差。这是 GRPO 比 PPO 简单的另一个原因（不用 value model + 不用复杂 credit assignment）。

工程注意：
- `seq_reward` 在 group 内做完 z-score 之后才能 assign，否则 advantage 失去意义。
- KL-weighted 模式需要在 rollout 阶段就把 `kl_per_token` 算出来存下，trainer 再读——增加一次 ref forward。
- Last-token 模式下，**最后一个 mask=1 的位置不一定是 EOS**——可能是模型没生成完就 truncate 了，要确保 loss_mask 的语义是"assistant 写的所有 token"。

</details>

---

## MLM 代码训练模式

完成上面的固定填空后，试试随机挖空模式 -- 每次点击「刷新」会随机遮盖不同的代码片段，帮你彻底记住每一行。

### 格式奖励

<CodeMasker title="Format Reward：R1 风格 think+answer 标签" :mask-ratio="0.15">
import re

def format_reward(completion: str) -> float:
    pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    match = re.match(pattern, completion, re.DOTALL)
    return 1.0 if match else 0.0
</CodeMasker>

### 可验证奖励

<CodeMasker title="Accuracy Reward：从 boxed 抠数 + 跨表示等价" :mask-ratio="0.15">
import re
from fractions import Fraction

def extract_boxed(text):
    matches = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return matches[-1] if matches else None

def normalize(s):
    s = s.strip().replace(r"\frac{", "").replace("}{", "/").replace("}", "")
    try:
        return Fraction(s)
    except ValueError:
        return Fraction(float(s))

def accuracy_reward(completion, gold):
    pred = extract_boxed(completion)
    if pred is None:
        return 0.0
    p, g = normalize(pred), normalize(gold)
    if p is None or g is None:
        return 0.0
    return 1.0 if p == g else 0.0
</CodeMasker>

### 多轮轨迹 Loss Mask

<CodeMasker title="Loss Mask：只在 assistant token 上算 loss + shift for CLM" :mask-ratio="0.15">
def build_loss_mask(token_ids, role_spans):
    mask = [0] * len(token_ids)
    for role, start, end in role_spans:
        if role == "assistant":
            for i in range(start, end):
                mask[i] = 1
    return mask

def shift_mask_for_clm(mask):
    return mask[1:]
</CodeMasker>

### 异步 Rollout Buffer

<CodeMasker title="Async Buffer：deque + Condition，rollout/trainer 解耦" :mask-ratio="0.15">
import asyncio
from collections import deque

class AsyncRolloutBuffer:
    def __init__(self, batch_size, max_capacity):
        self.batch_size = batch_size
        self.buf = deque(maxlen=max_capacity)
        self.cond = asyncio.Condition()

    async def put(self, traj):
        async with self.cond:
            self.buf.append(traj)
            self.cond.notify_all()

    async def get_batch(self):
        async with self.cond:
            await self.cond.wait_for(
                lambda: len(self.buf) >= self.batch_size)
            return [self.buf.popleft() for _ in range(self.batch_size)]
</CodeMasker>

### Composite Reward 工厂

<CodeMasker title="Composite Reward：加权组合 + 分量日志" :mask-ratio="0.15">
def make_reward_fn(weights, funcs):
    assert set(weights) == set(funcs)

    def composite(completion, gold):
        scores = {name: f(completion, gold) for name, f in funcs.items()}
        scores["total"] = sum(weights[k] * scores[k] for k in weights)
        return scores
    return composite
</CodeMasker>

### Trajectory→Token Reward 分配

<CodeMasker title="Last-token / Uniform / KL-weighted 三种分配" :mask-ratio="0.15">
import torch

def assign_last(seq_r, mask):
    out = torch.zeros_like(mask, dtype=torch.float32)
    nz = (mask > 0).nonzero().flatten()
    if len(nz) > 0:
        out[nz[-1].item()] = seq_r
    return out

def assign_uniform(seq_r, mask):
    out = torch.zeros_like(mask, dtype=torch.float32)
    n = mask.sum().item()
    if n > 0:
        out[mask.bool()] = seq_r / n
    return out

def assign_kl_weighted(seq_r, mask, kl):
    masked_kl = kl * mask
    total = masked_kl.sum()
    if total < 1e-8:
        return assign_uniform(seq_r, mask)
    return seq_r * (masked_kl / total)
</CodeMasker>

---

## 苏格拉底时刻

1. 练习 1 的格式奖励为什么是 0/1 而不是连续打分（比如标签缺一个扣 0.5）？连续奖励会引入什么问题？
2. 练习 4 的 `deque(maxlen=...)` 自动丢老轨迹——但你怎么知道丢哪些是"安全"的？引入 policy version 后，丢弃策略应该如何调整？
3. 练习 5 的 cosine length reward 在 acc=0.5 附近会发生什么？（提示：accuracy 真的是严格 0/1 吗？如果用 reward model 给的连续分数会怎样？）
4. 练习 6 的 last-token 策略，为什么在 GRPO 下能 work，但在 PPO 下经常炸？group-level baseline 起了什么作用？

## 推荐资源

- [open-r1/rewards.py](https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py) — 练习 1, 2, 5 的生产级版本（accuracy / format / cosine_scaled / repetition penalty / code）
- [verl trainer/main_ppo.py](https://github.com/volcengine/verl/blob/main/verl/trainer/main_ppo.py) — 练习 3, 4 的工业实现（多轮 mask + Single-Controller dispatch）
- [OpenRLHF ppo_trainer_async.py](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_trainer_async.py) — 练习 4 的 async PPO 完整版
- [THUDM/slime](https://github.com/THUDM/slime) — 异步 Rollout 框架，重点看 `slime/rollout/`
- [hkust-nlp/simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason) — 极简 R1-Zero 复现，适合通读
- [DeepSeek-R1 论文](https://arxiv.org/abs/2501.12948) — Section 2 描述了 rule-based reward 与多阶段训练
