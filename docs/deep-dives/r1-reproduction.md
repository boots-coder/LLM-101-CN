---
title: "R1 风格推理模型训练复现"
description: "用开源工具复现 R1 推理能力——从数据准备到 RL 训练到评估的完整指南"
topics: [R1, GRPO, PPO, reasoning, RL, verl, slime, GSM8K, Qwen3, reproduction]
prereqs: [training/reasoning, training/alignment]
---
# R1 风格推理模型训练复现

> **一句话总结:** 用开源工具复现 DeepSeek-R1 的推理能力——从数据准备到 GRPO 训练到评估的完整指南，覆盖 verl 和 slime 两套框架，支持从 0.6B 到 8B 的 Scaling 实验。

## R1 是什么

### DeepSeek-R1 的核心创新

DeepSeek-R1 是 2025 年初最具影响力的开源推理模型，其核心创新在于：**不依赖人工标注的 CoT 数据，仅通过 RL 训练就能让模型涌现出 Chain-of-Thought 推理能力。**

传统做法是先收集人工标注的推理过程（SFT 阶段），再做偏好对齐。R1 证明了一条更优路径：

```
Base Model (无推理能力)
    ↓  RL 训练 (GRPO + 规则奖励)
R1-Zero (涌现 CoT，但格式混乱)
    ↓  少量 SFT 数据做格式规范
R1 (结构化推理 + 高质量输出)
```

### "Aha Moment"——推理能力的涌现

R1 训练过程中最令人兴奋的发现是 **Aha Moment**：模型在 RL 训练到一定步数后，会突然开始在 `<think>` 标签中生成自我反思和验证步骤，例如：

```
<think>
Wait, let me reconsider this approach...
I made an error in step 2. Let me redo the calculation.
Actually, 3 * 15 = 45, not 35. So the correct answer is...
</think>
```

这种自我纠错能力不是通过监督学习"教"出来的，而是在 RL 训练中自发涌现的。

### 我们要复现什么

本指南复现的是 **R1-Zero** 阶段——纯 RL 训练：

| 目标 | 具体内容 |
|------|---------|
| 基座模型 | Qwen3-0.6B-Base / Qwen3-8B-Base |
| 训练方法 | GRPO（Group Relative Policy Optimization） |
| 训练数据 | 数学推理数据集（GSM8K / 自定义数据） |
| 奖励信号 | 规则奖励（答案正确性验证） |
| 评估指标 | GSM8K / MATH test set accuracy |

## 技术方案选择：verl vs slime

复现 R1 有两套主流开源框架可选：

| 对比维度 | verl | slime |
|----------|------|-------|
| **开发方** | 字节跳动 | 社区开源 |
| **训练范式** | 同步 GRPO/PPO | 全异步 GRPO |
| **推理引擎** | sglang / vLLM | sglang |
| **训练后端** | FSDP / Megatron | Megatron |
| **资源调度** | Ray + ResourcePool | Ray + 独立 GPU 分组 |
| **配置方式** | Hydra YAML | 命令行参数 |
| **适合场景** | 通用 RL 训练，稳定性优先 | 高吞吐异步训练，效率优先 |
| **上手难度** | 中等（配置较多） | 中等（需理解异步架构） |

::: tip 如何选择？
- **初学者**：推荐 verl，文档完善，社区活跃，配置化程度高
- **追求效率**：推荐 slime 全异步模式，训推分离，GPU 利用率更高
- **资源有限**：verl 的 Megatron 后端支持 TP 并行，2 张 3090 即可跑 0.6B
- **极致省钱 / 个人 PC 复现**：参见下方 [X-R1 路径](#x-r1-4-3090-低成本复现-r1-zero)，0.5B 模型 + 4×3090，1 小时跑通；单卡 + LoRA 也能玩
:::

::: info 第三条路径：simpleRL-reason —— 教学级最简 GRPO 复现
[hkust-nlp/simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)（港科大 NLP 组，[arXiv 2503.18892](https://arxiv.org/abs/2503.18892)）把 R1-Zero 配方压到极简：**只用规则奖励 + GSM8K/MATH 8K 题**，用同一份代码在 Llama3 8B / Mistral 7B/24B / DeepSeekMath 7B / Qwen2.5 0.5B-32B / Qwen2.5-Math-7B 共 10 个 base 模型上都复现出准确率提升 10~20 个绝对点。

- **定位**：介于 X-R1（4×3090，trl GRPOTrainer）与 verl/slime（8×A100，工业级）之间——比 X-R1 更适合"读代码 → 学清楚 GRPO 怎么把 base 模型从 0 训出推理能力"，比 verl/slime 门槛低得多
- **历史**：v0 基于 OpenRLHF + PPO（[GitHub tag v0](https://github.com/hkust-nlp/simpleRL-reason/tree/v0)），当前 main 分支已迁到 Ray + GRPO
- **附带产出**：[SimpleRL-Zoo 模型集合](https://huggingface.co/collections/hkust-nlp/simplerl-zoo-67e0fd24c185423c1e3452d1) 与 10 个 base 模型的 zero-RL 训练曲线，是研究"哪些 base 模型适合 zero-RL"的稀缺数据
:::

## X-R1: 4×3090 低成本复现 R1-Zero

verl 与 slime 都是工业级框架，Megatron + Ray 的栈对个人学习者并不友好。**[X-R1](https://github.com/dhcode-cpp/X-R1)（Apache 2.0）** 是社区里"最小可跑通的 R1-Zero 配方"——基于 HuggingFace TRL 的 `GRPOTrainer`，用 Qwen2.5 0.5B / 1.5B / 3B 三档基座，配上 750 / 1500 / 7500 三档数学数据，在 4×3090 (24G) 上一个小时内就能复现"Aha Moment"。

::: info 为什么单独列一节
verl / slime 的最小复现门槛通常是 8×A100，且需要熟悉 Megatron + Ray。X-R1 把门槛压到 **4×3090 / 1 小时 / ~7 美元算力费**，是想"亲眼看一眼 GRPO 怎么跑起来"的最低成本路径，特别适合在自家工作站或便宜云 GPU 上验证想法。
:::

### 整体配方一览

| 维度 | X-R1 默认 0.5B 配方 | 说明 |
|------|---------------------|------|
| 基座模型 | `Qwen/Qwen2.5-0.5B`（Base，非 Instruct） | 验证 R1-Zero 假设：Base 模型直接 RL |
| 训练数据 | `xiaodongguaAIGC/X-R1-750`（数学推理 750 条） | 也提供 1500 / 7500 三档可选 |
| 算法 | TRL `GRPOTrainer` + 4 个 reward 组合 | accuracy + format + reasoning_steps + len |
| 分布式 | accelerate + DeepSpeed ZeRO-3 + vLLM rollout | 3 卡训练 + 1 卡 vLLM 推理 |
| 显存占用 | ~22GB / 3090 | 留 2GB 余量给 OS |
| 单 step 时长 | ~30s | 总计 ~1 小时收敛 |
| Aha Moment | 约 step 37 出现 | 模型开始在 `<think>` 中自我验证 |

### 关键 yaml 超参拆解

X-R1 用一份精简到约 50 行的 yaml 控制全部训练逻辑，下面拆开 0.5B 配方中**最值得理解的那几个旋钮**：

```yaml
# === 模型与数据 ===
model_name_or_path: Qwen/Qwen2.5-0.5B           # Base，不是 Instruct
attn_implementation: flash_attention_2          # 必开，否则 1024 长度会爆显存
torch_dtype: bfloat16
dataset_name: xiaodongguaAIGC/X-R1-750          # 数学题库

# === GRPO 核心 ===
num_generations: 12          # 每个 prompt 采 12 个 completion，组成 GRPO 的 G
max_prompt_length: 256       # prompt 截断长度
max_completion_length: 1024  # 给推理过程留够 token，太短就出不了 Aha Moment

# === 训练循环 ===
learning_rate: 3.0e-06       # RLHF 经典学习率，过大会让策略崩塌
warmup_ratio: 0.1            # 前 10% step 线性 warmup
lr_scheduler_type: cosine
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 8   # 等效 batch = 4×8×3 卡 = 96 个 completion / step
gradient_checkpointing: true     # 用计算换显存

# === vLLM 推理后端 ===
use_vllm: true
vllm_device: auto                     # vLLM 自动占用最后一张可用卡
vllm_gpu_memory_utilization: 0.7      # vLLM 用 70% 显存，剩 30% 给 KV cache
```

::: warning 调参的 3 个直觉
- **`num_generations` 是 GRPO 的灵魂**：组太小（<4）advantage 估计噪声大；组太大（>16）显存爆炸。12 是 24G 卡上的甜点。
- **`max_completion_length` 千万别砍**：教学时容易为了省显存改成 256，但这会直接掐死 Aha Moment——模型还没来得及"等等让我重新算一下"就被截断。
- **`vllm_gpu_memory_utilization=0.7`**：vLLM 单独占一张卡跑 rollout，0.7 是给 prompt + completion 的 KV cache 留出空间。如果显存不够，就降到 0.5 但要接受吞吐变慢。
:::

### 启动脚本

```bash
# 3 卡训练 + 1 卡 vLLM rollout
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/zero3.yaml \
  --num_processes=3 \
  src/x_r1/grpo.py \
  --config recipes/X_R1_zero_0dot5B_config.yaml \
  > ./output/x_r1_0dot5B_sampling.log 2>&1
```

`accelerate` 的 ZeRO-3 配置会把 0.5B 的优化器状态切成 3 份摊到 3 张卡上，剩下的 1 张卡留给 vLLM 做生成。日志里搜 `aha` 或观察 reward 曲线是否在 30~40 step 处出现明显跳变，就能复现"Aha Moment"。

### 单卡 LoRA 变体（1×3090，~8h）

如果只有一张 24G 卡，X-R1 还提供了 LoRA 版本——把 1.5B 模型的 q/k/v_proj 和 embed_tokens 用 LoRA 包起来训练：

```yaml
# 单卡 LoRA 关键配置
model_name_or_path: Qwen/Qwen2.5-1.5B
use_vllm: false                                       # 单卡装不下 vLLM，退回 HF generate
num_generations: 6                                    # 组数砍半省显存
max_completion_length: 128                            # 长度也大幅压缩，仅做教学验证
gradient_accumulation_steps: 8

# LoRA 配置
lora_r: 32
lora_alpha: 8
lora_target_modules: ["q_proj", "v_proj", "k_proj", "embed_tokens"]
use_peft: true
```

::: tip 单卡 LoRA 的取舍
单卡 LoRA 主要是为了"在自己笔记本/单卡工作站上把流水线跑通看效果"，因为 `max_completion_length=128` 几乎不可能涌现真正的 Aha Moment。**想看到自我反思请优先选 4×3090 的 0.5B 全参方案**，单卡 LoRA 仅作为代码走查与超参实验。
:::

### 与 verl / slime 的定位差异

| 框架 | 抽象层 | 适合谁 | 上手成本 |
|------|--------|--------|---------|
| **X-R1** | TRL Trainer + accelerate | 个人学习者 / 教学复现 | ⭐ 30 分钟读完 yaml |
| **verl** | Ray + Worker 抽象 | 实验室 / 公司复现 | ⭐⭐⭐ 需熟悉 Hydra + Ray |
| **slime** | 全异步 Megatron | 追求训练吞吐的团队 | ⭐⭐⭐⭐ 需理解异步 RL |

::: info 教学路径建议
1. 先用 X-R1 跑通 0.5B 配方，亲眼看一次 reward 曲线和 Aha Moment
2. 再切到 verl 学习 Ray + 模块化 Worker 的工程组织方式
3. 最后用 slime 体会全异步 RL 的吞吐优势
:::

## 数据准备

### GSM8K 数据集适配

GSM8K 是 OpenAI 发布的小学数学推理数据集，包含约 7,500 条训练数据和 1,319 条测试数据。verl 框架要求数据以 **parquet 格式**存储，且包含特定字段。

核心处理逻辑：

```python
import re
import datasets

def parse_answer(raw_answer: str) -> str:
    """从 GSM8K 的 answer 字段中解析出最终数值"""
    # GSM8K 答案格式：推导过程后以 #### 开头给出最终数字
    hits = re.findall(r"####\s*(-?[\d,\.]+)", raw_answer)
    assert len(hits) > 0, f"无法从答案中解析数值: {raw_answer[:80]}"
    return hits[-1].replace(",", "")   # 取最后一个匹配，去掉千位逗号

# 加载数据集
dataset = datasets.load_dataset("openai/gsm8k", "main")

STEP_BY_STEP_SUFFIX = (
    "Please reason through this problem step by step, "
    'then write your final numeric answer after the "####" marker.'
)

def build_formatter(split_name: str):
    """返回一个 map 函数，将原始样本转换为 verl 所需的 parquet 字段格式"""
    def _transform(sample, row_idx):
        prompt_text = sample["question"] + " " + STEP_BY_STEP_SUFFIX
        answer_num = parse_answer(sample["answer"])
        return {
            "source": "openai/gsm8k",
            "prompt": [{"role": "user", "content": prompt_text}],
            "task_type": "math",
            "reward_model": {
                "style": "rule",           # 使用规则奖励
                "ground_truth": answer_num, # 标准答案
            },
            "metadata": {
                "split": split_name,
                "index": row_idx,
            },
        }
    return _transform

train_dataset = dataset["train"].map(build_formatter("train"), with_indices=True)
test_dataset = dataset["test"].map(build_formatter("test"), with_indices=True)

# 保存为 parquet
train_dataset.to_parquet("~/data/gsm8k/train.parquet")
test_dataset.to_parquet("~/data/gsm8k/test.parquet")
```

### verl 数据格式规范

verl 框架对数据有严格的字段要求：

```json
{
  "source": "openai/gsm8k",
  "prompt": [{"role": "user", "content": "问题文本..."}],
  "task_type": "math",
  "reward_model": {
    "style": "rule",
    "ground_truth": "42"
  },
  "metadata": {"split": "train", "index": 0}
}
```

::: warning 关键字段说明
- `source`：决定使用哪个 reward function。设为 `"openai/gsm8k"` 时触发 GSM8K 规则匹配，设为 `"lighteval/MATH"` 时触发 LaTeX 数学式匹配
- `reward_model.style`：`"rule"` 表示使用规则奖励，verl 内置了多种数学答案匹配规则
- `prompt`：必须是 chat format 的 message list
:::

### 自定义数据集适配

对于 MATH 等使用 `\boxed{}` 格式答案的数据集，需要修改答案提取逻辑：

```python
def parse_boxed_answer(text):
    """提取 LaTeX \\boxed{} 中的答案"""
    hits = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', text)
    return hits[-1] if hits else None
```

::: details 几何数据集 (Geometry3K) 的多模态适配
几何推理需要图片输入，数据格式中需要额外的 `images` 字段，并使用显式的 `<think>` 标签提示：

```python
GEOMETRY_PROMPT = (
    "First, reason about the problem internally as a step-by-step thought process. "
    "Then give your final answer. "
    r"Wrap your reasoning inside <think> </think> tags. "
    r"Put the final answer inside \boxed{}."
)

data = {
    "source": "hiyouga/geometry3k",
    "prompt": [{"role": "user", "content": prompt}],
    "images": images,  # 几何图片
    "task_type": "math",
    "reward_model": {"style": "rule", "ground_truth": answer},
}
```
:::

## Reward 设计

### 规则奖励 vs 模型奖励

R1 复现中最简洁高效的方式是**规则奖励**（Rule-based Reward）：

| 奖励类型 | 实现方式 | 优缺点 |
|----------|---------|--------|
| **规则奖励** | 答案字符串匹配 | 简单可靠，无需额外模型；但只适用于有标准答案的任务 |
| **PRM 模型** | 过程奖励模型打分 | 能评估推理过程质量；但需要训练额外模型 |
| **ORM 模型** | 结果奖励模型打分 | 适用于开放式任务；但信号较稀疏 |

数学推理任务天然适合规则奖励——答案要么对要么错，无需主观判断。

### verl 中的奖励实现

verl 通过 `reward_model` 配置自动路由奖励函数：

```python
# verl 内部根据 data_source 和 style 字段选择奖励函数
reward_fn = load_reward_manager(
    config, tokenizer, num_examine=0,
    **config.reward_model.get("reward_kwargs", {})
)
```

当 `reward_model.style == "rule"` 时，verl 会：
1. 从模型生成中提取答案（支持 `####` 和 `\boxed{}` 两种格式）
2. 与 `ground_truth` 做字符串匹配（支持数值等价判断）
3. 返回 0 或 1 的二值奖励

### slime 中的 DAPO 奖励

slime 框架使用 DAPO（Dynamic Advantage Policy Optimization）风格的奖励：

```bash
--rm-type dapo        # 使用 DAPO 风格奖励
--reward-key score    # 奖励值的字段名
```

DAPO 在标准 GRPO 基础上增加了动态采样和过滤机制，能更有效地利用训练信号。

## 训练配置

### 环境搭建

```bash
# 创建环境
conda create -n llm python=3.11
conda activate llm

# 安装核心依赖（verl 方案）
pip install verl==0.7.0 torch==2.8.0 vllm==0.11.0

# 安装 flash-attention（需从 GitHub Release 下载对应版本 whl）
pip install flash_attn-2.8.3+cu12torch2.8-cp312-cp312-linux_x86_64.whl
```

### DeepSpeed 配置

对于 SFT 等非 RL 训练阶段，使用 DeepSpeed ZeRO-1 即可。具体 YAML 由 `accelerate config` 交互式生成，关键字段为 `zero_stage: 1` + `mixed_precision: bf16` + `num_processes` = 实际 GPU 数。完整模板请参考 [accelerate 官方文档](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed)。

### 模型选择策略

| 模型 | 参数量 | 最低显存 | 训练时间(1 epoch) | 预期效果 |
|------|--------|---------|-------------------|---------|
| Qwen3-0.6B-Base | 0.6B | 2x 3090 (24GB) | ~2h | GSM8K ~57% |
| Qwen3-8B-Base | 8B | 4x 4090 (48GB) | ~2h | GSM8K ~92% |

::: tip 为什么选 Base 模型？
R1 的核心假设是：**推理能力可以从 Base 模型通过 RL 直接涌现**，不需要先做 SFT。使用 Base 模型而非 Chat 模型，更能验证这一假设。
:::

## verl 训练实战

### 架构概览

verl 的 PPO/GRPO 训练流程基于 Ray 分布式框架：

```
┌─────────────────────────────────────────┐
│              RayPPOTrainer              │
│                                         │
│  ┌──────────┐  ┌──────────┐  ┌───────┐ │
│  │  Actor    │  │ Rollout  │  │  Ref  │ │
│  │ (训练)   │  │ (推理)   │  │ Policy│ │
│  └────┬─────┘  └────┬─────┘  └───┬───┘ │
│       │              │            │      │
│       └──────────────┼────────────┘      │
│                      ↓                   │
│              ResourcePool               │
│           (GPU 资源共享管理)              │
└─────────────────────────────────────────┘
```

### main_ppo.py 核心流程

verl 的训练入口 `main_ppo.py` 使用 Hydra 配置管理，核心流程如下：

```python
@hydra.main(config_path="config", config_name="ppo_trainer")
def main(config):
    run_ppo(config)

def run_ppo(config):
    # 1. 初始化 Ray 集群
    ray.init(**ray_init_kwargs)

    # 2. 创建 TaskRunner 并在 Ray 上执行
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))
```

`TaskRunner.run()` 是训练的核心，四个阶段依次执行：**注册 Worker**（Actor/Rollout/Critic/RefPolicy）-> **加载 tokenizer 和 reward function** -> **创建训练/验证数据集** -> **初始化 RayPPOTrainer 并调用 `trainer.fit()` 启动训练循环**。

verl 的 Worker 支持 `fsdp`/`fsdp2`（PyTorch 原生分布式）和 `megatron`（Megatron-LM 后端，适合大模型 TP/PP 并行）两种策略，通过配置切换。

### 训练脚本要点

完整启动脚本请参考 [verl 官方示例](https://github.com/volcengine/verl/tree/main/examples)。本节只解释几个 R1 复现里关键的参数语义，不复制具体脚本。

::: warning 关键参数解读
- **`algorithm.adv_estimator=grpo`**：使用 GRPO 而非 PPO。GRPO 不需要 Critic 模型，用组内相对优势估计
- **`actor_rollout_ref.rollout.n`**：每个 prompt 采样多少回复，这是 GRPO 的 "Group" —— 用组内对比计算优势
- **`actor_rollout_ref.actor.kl_loss_type=low_var_kl`**：低方差 KL 散度估计，比标准 KL 更稳定
- **`actor_rollout_ref.rollout.gpu_memory_utilization`**：推理引擎占用的显存比例，剩下给训练
- **`actor_rollout_ref.actor.kl_loss_coef`**：KL 惩罚系数，过大会限制探索，常见取值 1e-3 量级
:::

### Qwen3-8B 的 Scaling 配置差异

从 0.6B 到 8B，关键的工程差异在显存策略：开启 Megatron 的 `optimizer_offload`（把 Adam 优化器状态卸载到 CPU），同时把 `data.max_prompt_length` 适度缩短（比如 256）来腾出激活显存。参数和梯度由于参与前/反向计算，通常仍保留在 GPU 上。

::: tip Offload 策略
8B 模型在 4x 4090 上需要开启 `optimizer_offload`，将 Adam 优化器状态（每参数 8 字节）卸载到 CPU 内存。仅卸载优化器对训练速度影响较小（~10%），但能显著降低显存需求。
:::

## slime 异步训练

### 全异步架构

slime 框架的核心优势是 **训推完全分离**：

```
┌──────────────────────────────────────────┐
│              Ray Cluster                 │
│                                          │
│  ┌────────────────┐  ┌────────────────┐  │
│  │  Training GPU  │  │  Rollout GPU   │  │
│  │  (Megatron)    │  │  (sglang)      │  │
│  │                │  │                │  │
│  │  GPU 0,1       │  │  GPU 2,3       │  │
│  │  TP=2          │  │  TP=2          │  │
│  └───────┬────────┘  └───────┬────────┘  │
│          │                   │           │
│          └───────┬───────────┘           │
│                  ↓                       │
│          Async Data Buffer               │
│      (训练和推理异步交换数据)              │
└──────────────────────────────────────────┘
```

与 verl 的同步模式相比，slime 的异步架构有几个显著优势：

1. **无 GPU 空闲等待**：训练和推理在不同 GPU 上并行执行
2. **动态 batch**：使用 `--use-dynamic-batch-size` 根据序列长度动态调整
3. **更高吞吐**：在同等硬件上，异步训练的样本吞吐量通常更高

### slime 训练脚本要点

具体启动脚本请参考 [slime 官方仓库](https://github.com/THUDM/slime) 的 `examples/`。值得理解的是它的几类参数分组：

- **Rollout 参数**：rollout 函数路径、prompt 数据、奖励类型（如规则奖励）、采样数 / batch / max-response-len、是否 balance-data
- **GRPO 参数**：advantage estimator、KL loss 系数（slime 中常设 0，由 importance ratio clip 兜底）、eps-clip 上下界（DAPO 引入双侧 clip）、是否启用 Truncated Importance Sampling
- **优化器参数**：Adam、常数学习率（1e-6 量级）、weight decay、CPU offload、精度感知优化器

### Retool 模式：训推共享 GPU

slime 支持 Retool（colocate）模式：`--colocate` 启用后，所有 GPU 同时承担训练和推理任务，配合 `--sglang-mem-fraction-static` 控制推理引擎占用比例。

::: tip Retool vs Fully Async
- **Retool**：适合 GPU 数量少但单卡显存大的场景（如 4x A100）
- **Fully Async**：适合 GPU 数量多但单卡显存有限的场景（如 8x 3090）
:::

## 评估与分析

### GSM8K 准确率评估

verl 内置了定期评估机制，通过 `trainer.test_freq` 控制评估频率：

```bash
trainer.test_freq=10    # 每 10 个训练步评估一次
```

评估时使用验证集数据，计算模型生成答案的正确率。典型的训练曲线如下：

```
训练步数    0.6B 准确率    8B 准确率
Step 0      ~5%            ~15%
Step 50     ~25%           ~50%
Step 100    ~40%           ~75%
Step 200    ~50%           ~85%
Step 300    ~55%           ~90%
Final       ~57%           ~92%
```

### WandB 监控

两套框架都支持 WandB 日志：

```bash
# verl
trainer.logger='["console","wandb"]'
trainer.project_name='verl_grpo'

# slime
--use-wandb
--wandb-project fully-async-0.6B-2gpu
```

关注的核心指标：
- **reward/mean**：平均奖励值（应持续上升）
- **reward/accuracy**：正确率
- **kl_divergence**：KL 散度（不应过大）
- **entropy**：策略熵（应缓慢下降但不坍缩）

## 从 0.6B 到 8B：Scaling 训练的注意事项

### 显存管理

| 技术 | 0.6B 是否需要 | 8B 是否需要 | 显存节省 |
|------|:------------:|:-----------:|---------|
| Gradient Checkpointing | 推荐 | 必须 | ~40% |
| Optimizer Offload | 不需要 | 必须 | ~30% |
| Tensor Parallelism | TP=2 | TP=4 | 线性 |
| Param Offload | 不需要 | 可选 | ~20%（影响速度） |

```bash
# 8B 模型必须开启的选项
++actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.actor.megatron.optimizer_offload=True
```

### 超参调整建议

| 超参 | 0.6B 推荐值 | 8B 推荐值 | 说明 |
|------|:-----------:|:---------:|------|
| 学习率 | 1e-6 | 1e-6 | 两个规模通用 |
| KL 系数 | 0.001 | 0.001 | 过大会限制探索 |
| Batch size | 8 | 8 | 受显存限制 |
| Micro batch | 2 | 2 | 梯度累积步数 = batch/micro |
| 采样数 (n) | 8 | 8 | GRPO 的组大小 |
| Max response len | 1024 | 1024 | 更长会显著增加显存 |

### 常见问题排查

::: details 训练初期 reward 长时间不涨
- 检查数据格式：确认 `reward_model.ground_truth` 字段有值
- 检查 `source` 是否匹配正确的规则奖励函数
- 降低学习率到 5e-7 试试
:::

::: details OOM（显存不足）
- 开启 gradient checkpointing
- 减小 `max_response_length`（从 1024 降到 512）
- 开启 optimizer offload
- 减小 `rollout.gpu_memory_utilization`（从 0.5 降到 0.4）
:::

::: details KL 散度爆炸
- 增大 `kl_loss_coef`（从 0.001 到 0.01）
- 检查学习率是否过大
- 确认参考策略正确加载
:::

## 苏格拉底时刻

::: tip 停下来思考
1. **R1-Zero 为什么能"无中生有"地涌现推理能力？** 提示：Base 模型预训练阶段已经见过数学推导文本，RL 训练做的是"激活"而非"教授"。

2. **GRPO 相比 PPO 省掉了什么？** 提示：PPO 需要一个 Critic 模型估计 Value Function，GRPO 用组内采样的相对奖励替代了 Value baseline。

3. **为什么规则奖励在数学任务上足够好？** 提示：数学答案的验证比生成容易得多（P vs NP 的直觉），二值奖励虽然稀疏但信号准确。

4. **异步训练 vs 同步训练的 trade-off 是什么？** 提示：异步训练用的是"过时"的 rollout 数据（off-policy），但 GPU 利用率更高。GRPO 的 importance sampling ratio clip 正是为了缓解这个问题。
:::

## 面试考点

::: details Q: DeepSeek-R1 和 OpenAI O1 的技术路线有什么本质区别？
**A:** O1 的技术细节未公开，但从公开信息推测使用了大量人工标注的 CoT 数据做 SFT + RL。R1 的突破在于证明了 **纯 RL 路线**（R1-Zero）也能涌现推理能力，无需人工 CoT 标注。R1 的完整流程是 R1-Zero(RL) → 少量 SFT 规范格式 → 继续 RL，大幅降低了数据标注成本。
:::

::: details Q: GRPO 的数学原理是什么？相比 PPO 有什么优势？
**A:** GRPO（Group Relative Policy Optimization）对每个 prompt 采样一组回复 $\{o_1, ..., o_G\}$，将每个回复的 reward 减去组内均值再除以标准差作为 advantage 估计：$A_i = (r_i - \mu_G) / \sigma_G$。相比 PPO 省掉了 Critic 模型（节省约 50% 显存），且 advantage 估计更稳定（基于同一 prompt 的多个采样，减少了 prompt 间方差的干扰）。
:::

::: details Q: 训练 R1 时为什么用 Base 模型而不是 Chat 模型？
**A:** 使用 Base 模型有两个原因：(1) 验证"推理能力可以从 RL 直接涌现"这一核心假设；(2) Chat 模型已经被 SFT 过的回答模式"锁定"了，RL 训练需要更大的 KL 散度才能跳出已有模式，反而更难训练。Base 模型的策略空间更大，RL 有更多的探索自由度。
:::

::: details Q: 如何判断 R1 训练是否出现了"Aha Moment"？
**A:** 监控以下信号：(1) reward 曲线出现跳变而非渐进上升；(2) 模型输出中开始出现自我纠错模式（"Wait, let me reconsider..."）；(3) 推理过程变长但准确率同步提升；(4) 模型开始对不确定的步骤进行验证（"Let me verify..."）。这些通常发生在训练中期，是 RL 探索到了高效推理策略的标志。
:::

## 推荐资源

| 资源 | 链接 | 说明 |
|------|------|------|
| DeepSeek-R1 论文 | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) | R1 原始论文 |
| verl 框架 | [GitHub](https://github.com/volcengine/verl) | 字节开源的 RL 训练框架 |
| slime 框架 | [GitHub](https://github.com/THUDM/slime) | 全异步 RL 训练框架 |
| DAPO 论文 | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) | 动态优势策略优化 |
| Open-R1 项目 | [GitHub](https://github.com/huggingface/open-r1) | HuggingFace 的 R1 复现 |
| GSM8K 数据集 | [HuggingFace](https://huggingface.co/datasets/openai/gsm8k) | 小学数学推理基准 |
