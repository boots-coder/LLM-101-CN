---
title: 手撕 nano Agent-RL
description: 从零拼一个"会用工具的 LLM Agent" + 用 GRPO 训练，~500 行内可在 CPU 跑通的最小可教学版
topics: [agent-rl, GRPO, multi-turn-rollout, tool-use, RLVR, calculator-agent, from-scratch]
prereqs: [/training/agent-rl, /training/alignment, /deep-dives/nano-rlhf, /exercises/agent-rl-fill]
---

# 手撕 nano Agent-RL

> 把 Agent-RL 的所有架构零件——env / rollout / reward / GRPO trainer——拆到最简，用一个"两步加法计算器"做端到端 demo。本文不是为了训出一个有用的模型，而是为了让你**从代码层面看清楚每条数据的流动**。

::: info 配套资源
- 主线：[Agent-RL 训练范式](/training/agent-rl)（架构与算法）
- 算法：[手撕 RLHF Pipeline](/deep-dives/nano-rlhf)（PPO/DPO/GRPO 的 from-scratch）
- 框架：[R1 复现](/deep-dives/r1-reproduction)（verl / slime / X-R1 实战）
- 练习：[Agent-RL 代码填空](/exercises/agent-rl-fill)
:::

## 为什么再写一个 nano

[nano-rlhf.md](/deep-dives/nano-rlhf) 已经从零实现过 PPO/DPO/GRPO，但它的 rollout 是**单步生成 → 单步打分**（典型的 RLHF 偏好对齐）。

Agent-RL 的本质区别在于 rollout 的形状：

| 维度 | RLHF (nano-rlhf) | Agent-RL (本文) |
|------|------------------|-----------------|
| rollout 步数 | 1 | T（多轮，T 不定长） |
| 奖励来源 | reward model | 环境 / verifier |
| advantage 单位 | 整段回答 | 整条轨迹（含 tool obs） |
| loss mask | 跳过 prompt | 跳过 prompt + tool obs + system |
| 终止条件 | EOS | EOS / 答完 / 步数上限 / 工具失败 |

本文聚焦后者。代码全部是从零实现的中文教学版，生产级实现请看 [verl](https://github.com/volcengine/verl) / [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) / [slime](https://github.com/THUDM/slime)。

---

## 一图看清整体架构

```text
┌─────────────────────────────────────────────────────────────┐
│                       AgentRLTrainer                        │
│                                                             │
│   ┌────────────┐   ┌─────────────┐   ┌────────────────┐    │
│   │  Policy    │──▶│  Rollout    │──▶│  Reward + Mask │    │
│   │  (πθ)      │   │  (multi-    │   │  Construction  │    │
│   │            │   │   turn)     │   │                │    │
│   └─────▲──────┘   └──────┬──────┘   └────────┬───────┘    │
│         │                 │                   │            │
│         │          ┌──────▼──────┐            │            │
│         │          │  ToolEnv    │            │            │
│         │          │ (calculator)│            │            │
│         │          └─────────────┘            │            │
│         │                                     │            │
│         │     ┌───────────────────────────────▼─────┐      │
│         └─────│  GRPO Update                        │      │
│               │  (group z-score → clipped surrogate)│      │
│               └─────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

四个组件，对应下面四节。

---

## 第一步：定义环境（ToolEnv）

环境是 Agent-RL 的"事实来源"。我们做最简的：一个支持 `add` / `mul` 两个工具的内存计算器。

```python
import re
from dataclasses import dataclass

@dataclass
class StepResult:
    obs: str            # 观测（要拼回 trajectory 的 tool 段）
    done: bool          # 这一步后是否终止
    info: dict          # 调试信息

class CalculatorEnv:
    """
    一个极小的工具环境。
    
    协议：
    - assistant 输出 <call>op(a,b)</call> 时调用工具
    - assistant 输出 <answer>\\boxed{N}</answer> 时终止 episode
    - 每条 episode 最多 max_steps 次工具调用
    """
    CALL_RE = re.compile(r"<call>(\w+)\(([^)]+)\)</call>")
    ANSWER_RE = re.compile(r"<answer>\\boxed\{(-?\d+)\}</answer>")

    def __init__(self, max_steps: int = 4):
        self.max_steps = max_steps

    def reset(self, problem: dict):
        """problem = {'expr': '(3+4)*5', 'answer': 35}"""
        self.problem = problem
        self.steps = 0
        self.history: list[str] = []   # tool obs 历史（仅用于调试）
        return f"问题：计算 {problem['expr']}\n"

    def step(self, completion: str) -> StepResult:
        """
        给 assistant 的最新一段输出，返回环境反馈。
        - 优先看 <answer>，看到就终止（不管对错）
        - 没 <answer> 就找 <call>，调用工具
        - 都没有：当作格式失败，给 invalid obs，不终止
        """
        # 检查终止
        ans = self.ANSWER_RE.search(completion)
        if ans:
            return StepResult(obs="", done=True,
                              info={"final": int(ans.group(1))})
        # 检查工具调用
        call = self.CALL_RE.search(completion)
        if call:
            op, args = call.group(1), call.group(2)
            try:
                a, b = [int(x) for x in args.split(",")]
                if op == "add":
                    r = a + b
                elif op == "mul":
                    r = a * b
                else:
                    return StepResult(
                        obs=f"<obs>error: unknown op {op}</obs>",
                        done=False, info={})
                self.history.append(f"{op}({a},{b})={r}")
            except (ValueError, ZeroDivisionError):
                return StepResult(
                    obs=f"<obs>error: invalid args {args}</obs>",
                    done=False, info={})
            self.steps += 1
            done = self.steps >= self.max_steps
            return StepResult(obs=f"<obs>{r}</obs>",
                              done=done, info={"tool_result": r})
        # 啥都没有
        return StepResult(obs="<obs>error: no call or answer</obs>",
                          done=False, info={})
```

**为什么把 env 设计成可中断 + 阶段性返回 obs？** 因为 Agent rollout 的本质就是"模型说一段 → 环境插一段 → 模型再说一段"的拼接。环境必须能 inspect 模型的中间输出（不能等整个 trajectory 结束才打分）。这是 Agent-RL 与 vanilla RLHF 的根本区别。

---

## 第二步：多轮 Rollout

Rollout 的核心是**让模型边生成边停**：每生成完一段（遇到 `</call>` 或 `</answer>`），暂停，把环境的反馈 append 回 prompt，继续生成。

```python
import torch

STOP_TAGS = ("</call>", "</answer>")

def rollout_one(model, tokenizer, env, problem,
                max_new_tokens: int = 80) -> dict:
    """
    跑完一条 trajectory。返回:
        token_ids: list[int], 整条 traj 的 token 序列
        role_spans: list[(role, start, end)]
        seq_reward: float（轨迹级 reward，由外部 reward_fn 计算）
    """
    prompt = env.reset(problem)
    # role_spans 用于后续构造 loss mask
    role_spans = []
    cur_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
    role_spans.append(("user", 0, len(cur_ids)))

    while True:
        # 生成一段直到 stop tag 或上限
        new_text = generate_until_stop(
            model, tokenizer, cur_ids, STOP_TAGS, max_new_tokens)
        new_ids = tokenizer.encode(new_text, return_tensors="pt")[0]
        start = len(cur_ids)
        cur_ids = torch.cat([cur_ids, new_ids])
        role_spans.append(("assistant", start, len(cur_ids)))

        # 让 env 处理这一段
        result = env.step(new_text)
        if result.done:
            break

        # 把 obs 拼回去
        obs_ids = tokenizer.encode(result.obs, return_tensors="pt")[0]
        start = len(cur_ids)
        cur_ids = torch.cat([cur_ids, obs_ids])
        role_spans.append(("tool", start, len(cur_ids)))

        if len(cur_ids) > 512:  # 安全阀
            break

    return {
        "token_ids": cur_ids.tolist(),
        "role_spans": role_spans,
        "completion_text": tokenizer.decode(cur_ids),
    }


def generate_until_stop(model, tokenizer, prompt_ids,
                        stop_tags, max_new_tokens) -> str:
    """简化版生成：每生成一个 token 就解码一次，发现 stop tag 立即停。"""
    out_ids = prompt_ids.tolist()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            inp = torch.tensor([out_ids])
            logits = model(inp).logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        out_ids.append(next_id)
        # 解码后 N 个字符检查是否含 stop（N 取个保守值）
        tail = tokenizer.decode(out_ids[len(prompt_ids):])
        if any(tag in tail for tag in stop_tags):
            return tail
    return tokenizer.decode(out_ids[len(prompt_ids):])
```

**两个工程要点：**

1. **Stop sequence 检测要在解码后做，不能在 token-id 上做。** 因为 `</call>` 经常被 tokenizer 切成多个 token，不同 BPE 切法不一样。直接在文本上 `endswith` 是最稳的。生产框架（[vLLM stop sequences](https://docs.vllm.ai/en/latest/api/inference_params.html)、[SGLang](https://github.com/sgl-project/sglang)）的内部实现也是这么做的，只是用了更高效的滑动窗口 KMP。

2. **role_spans 是 loss mask 的依据。** Rollout 阶段就要把每段 token 的来源记下来，否则 trainer 算 loss 时分不清谁是 assistant 谁是 tool。这是 [verl](https://github.com/volcengine/verl) 的 `data_proto.py` 里 `loss_mask` / `response_mask` 的本质。

---

## 第三步：Reward 计算 + Loss Mask

Reward 直接复用 [Agent-RL 代码填空](/exercises/agent-rl-fill) 练习 1 + 2 的 `format_reward` + `accuracy_reward`。Mask 同练习 3。

```python
def trajectory_reward(traj_text: str, gold_answer: int,
                      w_fmt: float = 0.1) -> float:
    """轨迹级 reward = 格式 + 答对。"""
    fmt = format_reward_extended(traj_text)         # 见下
    acc = accuracy_reward_int(traj_text, gold_answer)
    return w_fmt * fmt + acc

def format_reward_extended(text: str) -> float:
    """允许中间有任意多次 <call>...</call><obs>...</obs>，最后必须 <answer>。"""
    return 1.0 if re.search(
        r"<answer>\\boxed\{-?\d+\}</answer>", text) else 0.0

def accuracy_reward_int(text: str, gold: int) -> float:
    m = CalculatorEnv.ANSWER_RE.search(text)
    return 1.0 if (m and int(m.group(1)) == gold) else 0.0


def build_loss_mask(token_ids, role_spans):
    mask = [0] * len(token_ids)
    for role, start, end in role_spans:
        if role == "assistant":
            for i in range(start, end):
                mask[i] = 1
    return mask
```

**为什么权重 `w_fmt` 远小于 acc？** 因为 R1-Zero 论文里发现："格式分太大 → 模型快速学会输出格式但不学会算 → reward 上限被锁死"。先用小权重让格式信号活着，主信号还是 accuracy。

---

## 第四步：GRPO 更新

最后把零件拼成训练循环。Group 大小 G=8（每个题目采 8 条轨迹），用 group 内 z-score 算 advantage。

```python
import copy
import torch.nn.functional as F

def grpo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """组内 z-score。rewards: [G]，返回 [G]。"""
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)

def grpo_loss(policy, ref_policy, batch,
              clip: float = 0.2, kl_coef: float = 0.04) -> torch.Tensor:
    """
    batch 里每条数据有: token_ids (T,), loss_mask (T,), advantage (scalar)
    
    流程: forward 一次拿到 logp (T, V) → gather 拿到每个 token 的 logp(token) →
          算 ratio = exp(new_lp - old_lp), clipped surrogate, K3 KL，最后 mask 求和。
    """
    losses = []
    for item in batch:
        ids = torch.tensor(item["token_ids"])
        mask = torch.tensor(item["loss_mask"], dtype=torch.float32)
        adv = item["advantage"]

        # forward + gather
        logits = policy(ids.unsqueeze(0)).logits[0]            # [T, V]
        logp_full = F.log_softmax(logits, dim=-1)
        # 每个位置的 token 是 ids[t+1]，要 shift
        # 简化：用 logp[:-1] 预测 ids[1:]
        new_lp = logp_full[:-1].gather(
            1, ids[1:].unsqueeze(-1)).squeeze(-1)              # [T-1]
        # ref policy 同位置
        with torch.no_grad():
            ref_logits = ref_policy(ids.unsqueeze(0)).logits[0]
            ref_lp = F.log_softmax(ref_logits, dim=-1)[:-1]\
                       .gather(1, ids[1:].unsqueeze(-1)).squeeze(-1)
        # old policy 用 rollout 阶段冻结的 new_lp（这里简化为同一份）
        old_lp = new_lp.detach()

        # 对应 mask 也要 shift
        m = mask[1:]                                            # [T-1]

        # ratio + clipped surrogate
        ratio = (new_lp - old_lp).exp()
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
        pg = torch.minimum(unclipped, clipped)

        # K3 KL
        log_r = ref_lp - new_lp
        kl = log_r.exp() - log_r - 1

        # masked mean
        token_loss = -(pg - kl_coef * kl)
        loss = (token_loss * m).sum() / m.sum().clamp(min=1)
        losses.append(loss)

    return torch.stack(losses).mean()
```

**为什么这里 `old_lp = new_lp.detach()` 而不是真的存 rollout 阶段的 logp？**

教学版简化。生产实现：
1. **Rollout 阶段**用一个**冻结的 `policy_old`** 跑生成，把 `old_logp` 记下来存进 buffer。
2. **Train 阶段**`policy` 已经被前几次更新移动过，`new_logp` 与 `old_logp` 不再相等，ratio ≠ 1，clipped surrogate 才真正起作用。

具体怎么写见 [练习 6 的 toy GRPO](/exercises/dpo-grpo-fill#练习-6端到端-toy-grpo-训练level-3) 与 [OpenRLHF/openrlhf/trainer/ppo_trainer.py](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_trainer.py)。

---

## 第五步：拼完整训练循环

```python
def train_one_iteration(policy, ref, opt, env, problems, G=8):
    """一个外层迭代：对每个题目采 G 条轨迹，组成一个 batch 做一次 GRPO step。"""
    batch = []
    for prob in problems:
        # 1. 采 G 条轨迹
        rewards, masks, ids_list = [], [], []
        for _ in range(G):
            traj = rollout_one(policy, tokenizer, env, prob)
            r = trajectory_reward(traj["completion_text"], prob["answer"])
            rewards.append(r)
            ids_list.append(traj["token_ids"])
            masks.append(build_loss_mask(traj["token_ids"], traj["role_spans"]))

        # 2. group z-score
        rewards_t = torch.tensor(rewards)
        adv = grpo_advantage(rewards_t)

        # 3. 装 batch（这里把每个 traj 当一条 sample）
        for ids, m, a in zip(ids_list, masks, adv.tolist()):
            batch.append({
                "token_ids": ids,
                "loss_mask": m,
                "advantage": a,
            })

    # 4. 几次 inner update（GRPO 标配 K=1~4）
    for _ in range(2):
        loss = grpo_loss(policy, ref, batch)
        opt.zero_grad(); loss.backward(); opt.step()

    return rewards_t.mean().item()


# ====== 主循环骨架 ======
PROBLEMS = [
    {"expr": "3+4", "answer": 7},
    {"expr": "5+2", "answer": 7},
    {"expr": "(3+4)+1", "answer": 8},
]  # 真实场景下从数据集采样

for it in range(50):
    avg_r = train_one_iteration(policy, ref, opt, env, PROBLEMS, G=8)
    print(f"iter {it}: avg reward = {avg_r:.3f}")
```

**预期收敛曲线：**
- iter 0-5：reward ≈ 0（模型纯随机，连 `<answer>` tag 都不会输出）
- iter 5-15：reward 跳到 0.1-0.2（学会输出 `<answer>` 但答案瞎填）
- iter 15-30：reward 跳到 0.5+（学会调用 add/mul 工具）
- iter 30+：reward 接近 1.0

**可能不收敛的常见原因：**
1. **Group size 太小**：G < 4 时 z-score 方差过大，advantage 噪声主导。
2. **Reward 全 0 或全 1**：组内方差为 0，无信号，policy 不更新。混合不同难度的题缓解。
3. **`old_logp` 没冻结**（如本文教学版）：理论上 ratio 永远是 1，clip 没用。短期能 work，长期会出问题。
4. **格式 reward 权重太大**：模型只学格式不学算。把 `w_fmt` 从 0.1 调到 0.05 试试。

---

## 这个 nano 比生产框架少了什么

| 维度 | nano（本文） | 生产（[verl](https://github.com/volcengine/verl) / [slime](https://github.com/THUDM/slime)） |
|------|--------------|--------------------------------------------------------------------------------------------|
| Rollout backend | HuggingFace forward + 手写采样 | vLLM / SGLang，吞吐高 100×+ |
| Old policy | 同当前 policy（教学简化） | 独立冻结副本 + buffer |
| Policy version | 隐式 = 当前迭代 | 显式 version + on/off-policy 检查 |
| 多 rollout worker | 串行 | Ray Actor Group 并发 |
| Weight sync | 不需要 | NCCL Broadcast 或 CUDA IPC |
| 分布式训练 | 单卡 | DP × TP × PP × ZeRO 任意组合 |
| 数据加载 | 内存列表 | datasets streaming + parquet |
| 监控 | print | wandb + grad norm + KL 曲线 + acc/fmt 分量 |
| 失败恢复 | 无 | rollout 超时 / OOM / NaN 自动重试 |

**记住一件事：所有这些复杂性都是工程，不是算法。** 算法部分就是上面 200 行——env / rollout / mask / GRPO。看懂这 200 行，再看 verl 的几万行就只是"把它们做对、做快、做大"。

---

## 怎么把它扩到一个真实任务

如果想跑通真正的 R1-Zero 复现，最小改动清单（≈3 天工作量）：

1. **换数据**：把 `PROBLEMS` 换成 [GSM8K](https://huggingface.co/datasets/openai/gsm8k) 或 [MATH](https://huggingface.co/datasets/lighteval/MATH-Hard) 的 `(question, answer)` 对。
2. **换模型**：从 tiny GPT-2 换成 `Qwen2.5-0.5B-Instruct` 或 `Qwen2.5-1.5B`（CPU 训不动，需要单卡 24G）。
3. **换 rollout backend**：把 `generate_until_stop` 换成 `vllm.LLM.generate` 加 `stop=["</call>", "</answer>"]`。吞吐瞬间提升 50×。
4. **加 `old_policy`**：每次 rollout 用 `policy.eval()` 的当前快照，存 `old_logp`。
5. **多任务批量**：每次外层迭代采 32-64 个题目，G=8，总 batch ≈ 256-512。
6. **加 monitor**：wandb 记录 `reward / fmt / acc / kl_div / grad_norm / response_length`。

到这一步你就基本复现了 [hkust-nlp/simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)。

继续做：
- 替换 GRPO → DAPO（[slime](https://github.com/THUDM/slime) 的默认）—— 加 token-level 长度惩罚
- 切换到 verl 的 Single-Controller 架构 —— 跨节点
- 多模态化 —— 把 prompt 换成图片 + 问题，参考 [open-r1/multimodal](https://github.com/huggingface/open-r1)

---

## 苏格拉底时刻

1. 第二步的 `generate_until_stop` 在生成每个 token 后都解码整段尾部检查 stop。如果 stop tag 跨 token 边界（比如 `</call>` 被切成 `</`, `call`, `>`），逐 token 检查会不会漏？要怎么改？
2. 第三步只用了 last-token reward？还是 trajectory reward？如果改成 [Token Reward Shaping（练习 6）](/exercises/agent-rl-fill#练习-6trajectory--token-reward-分配level-3) 的 KL-weighted 分配，loss 公式要怎么改？
3. 第四步的 `old_lp = new_lp.detach()` 是个简化。**如果一直这样写，训练会发生什么？** 提示：考虑 ratio = 1 ⇒ clipped surrogate 的退化形式 ⇒ 与 vanilla policy gradient 的关系。
4. 我们的环境是确定性的（`add(3,4)` 永远 = 7）。如果换成**有随机性**的环境（搜索引擎、人类反馈），rollout 阶段记 `old_logp` 还有意义吗？credit assignment 又会变成什么样？

## 推荐资源

- **从零实现参考**：
  - [hkust-nlp/simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason) — R1-Zero 极简复现（约 1500 行）
  - [HuggingFace open-r1](https://github.com/huggingface/open-r1) — `src/open_r1/grpo.py` + `rewards.py` 是本文的"工业升级版"

- **生产级框架**（推荐看的顺序）：
  - [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — 单卡可跑，代码好读
  - [volcengine/verl](https://github.com/volcengine/verl) — Single-Controller 架构典范
  - [THUDM/slime](https://github.com/THUDM/slime) — 全异步 + Hybrid Engine，最新设计

- **论文**：
  - [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — Section 2 描述 GRPO + rule-based reward
  - [DAPO](https://arxiv.org/abs/2503.14476) — slime 默认算法的论文
  - [GRPO 原论文（DeepSeekMath）](https://arxiv.org/abs/2402.03300)

- **本项目相关**：
  - 主线：[Agent-RL](/training/agent-rl) / [对齐](/training/alignment)
  - 算法：[手撕 RLHF](/deep-dives/nano-rlhf)
  - 复现：[R1 复现实战](/deep-dives/r1-reproduction)
  - 练习：[Agent-RL Fill-in](/exercises/agent-rl-fill) / [DPO/GRPO Fill-in](/exercises/dpo-grpo-fill) / [PPO Fill-in](/exercises/ppo-fill)
