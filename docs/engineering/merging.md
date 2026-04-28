---
title: "模型合并"
description: "Linear/SLERP/TIES/DARE 合并方法、mergekit 工具、MoE 合并"
topics: [model-merging, SLERP, TIES, DARE, task-arithmetic, mergekit, frankenmerge]
---
# 模型合并

::: info 一句话总结
模型合并（Model Merging）不需要任何训练，仅通过对多个模型的权重进行数学运算（线性插值、SLERP、TIES、DARE 等），就能将不同模型的能力融合到一个模型中。
:::


## 在大模型体系中的位置

```
预训练 → SFT/RLHF → 多个专长模型
                       ├── 模型 A（擅长代码）
                       ├── 模型 B（擅长数学）
                       └── 模型 C（擅长对话）
                              ↓
                       【模型合并】 ◄── 你在这里
                              ↓
                       合并模型（多能力融合）
                              ↓
                       部署上线
```

模型合并是一种**零训练成本**的能力增强技术。当你有多个微调模型各自擅长不同任务时，合并可以让你"鱼和熊掌兼得"——无需 GPU 训练，只需要能加载模型权重的 CPU 内存。

## 为什么要合并模型？

| 方法 | 训练成本 | 数据需求 | 效果 |
|------|---------|---------|------|
| 从头训练多任务模型 | 极高 | 需要所有任务的数据 | 最优但成本最高 |
| 多任务微调 | 高 | 需要所有任务的数据 | 好，但有数据混合问题 |
| **模型合并** | **零** | **不需要训练数据** | 通常接近多任务微调 |
| MoE 路由 | 推理成本高 | 需要路由数据 | 好，但模型变大 |

核心优势：

1. **零训练成本**：不需要 GPU，纯 CPU 运算
2. **无需数据**：不需要获取各个任务的训练数据
3. **组合灵活**：可以自由组合社区发布的各种微调模型
4. **快速迭代**：分钟级完成，可以快速实验

---

## 合并方法详解

### 1. Linear（线性插值）

最简单直接的方法：对两个模型的权重做加权平均。

$$\theta_{merged} = \alpha \cdot \theta_A + (1 - \alpha) \cdot \theta_B$$

其中 $\alpha \in [0, 1]$ 控制两个模型的混合比例。

**直觉**：想象两个模型是空间中的两个点，线性插值就是在它们之间的连线上找一个点。

```python
import torch
from collections import OrderedDict

def linear_merge(
    state_dict_a: OrderedDict,
    state_dict_b: OrderedDict,
    alpha: float = 0.5,
) -> OrderedDict:
    """线性插值合并两个模型
    
    Args:
        state_dict_a: 模型 A 的权重字典
        state_dict_b: 模型 B 的权重字典
        alpha: 模型 A 的权重比例，模型 B 的比例为 (1 - alpha)
    
    Returns:
        合并后的权重字典
    """
    merged = OrderedDict()
    
    for key in state_dict_a:
        if key in state_dict_b:
            merged[key] = alpha * state_dict_a[key] + (1 - alpha) * state_dict_b[key]
        else:
            merged[key] = state_dict_a[key]
    
    # 处理 B 中有但 A 中没有的键
    for key in state_dict_b:
        if key not in state_dict_a:
            merged[key] = state_dict_b[key]
    
    return merged

# 使用示例
# model_a = AutoModelForCausalLM.from_pretrained("model_a")
# model_b = AutoModelForCausalLM.from_pretrained("model_b")
# merged_state = linear_merge(model_a.state_dict(), model_b.state_dict(), alpha=0.6)
# model_a.load_state_dict(merged_state)
# model_a.save_pretrained("merged_model")
```

**局限性**：线性插值假设两个模型的权重空间是"对齐"的，但不同微调产生的权重可能在不同的"盆地"中，简单平均可能落在两个盆地之间的"高地"上，导致性能下降。

### 2. SLERP（球面线性插值）

SLERP（Spherical Linear Interpolation）不是在欧氏空间中做线性插值，而是在**超球面**上沿大圆弧做插值。

$$\theta_{merged} = \frac{\sin((1 - t) \cdot \Omega)}{\sin \Omega} \cdot \theta_A + \frac{\sin(t \cdot \Omega)}{\sin \Omega} \cdot \theta_B$$

其中 $\Omega = \arccos\left(\frac{\theta_A \cdot \theta_B}{|\theta_A| \cdot |\theta_B|}\right)$ 是两个向量之间的角度。

**直觉**：如果把模型权重想象成超球面上的方向向量，SLERP 保证了插值路径上每个点的"能量"（向量模长）不变。而线性插值会导致中间点的模长变小（想象从北极到赤道走直线 vs 沿球面走）。

```python
import torch
import numpy as np
from collections import OrderedDict

def slerp(
    t: float,
    v0: torch.Tensor,
    v1: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """球面线性插值（Spherical Linear Interpolation）
    
    Args:
        t: 插值系数，0.0 返回 v0，1.0 返回 v1
        v0: 起始向量
        v1: 终止向量
        eps: 数值稳定性的小量
    
    Returns:
        插值结果向量
    """
    # 展平为 1D
    orig_shape = v0.shape
    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()
    
    # 计算夹角
    v0_norm = v0_flat / (v0_flat.norm() + eps)
    v1_norm = v1_flat / (v1_flat.norm() + eps)
    
    # 余弦相似度 → 夹角
    cos_omega = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
    omega = torch.acos(cos_omega)
    
    # 如果夹角太小，退化为线性插值（避免除以 0）
    if omega.abs() < eps:
        result = (1 - t) * v0_flat + t * v1_flat
    else:
        sin_omega = torch.sin(omega)
        result = (torch.sin((1 - t) * omega) / sin_omega) * v0_flat + \
                 (torch.sin(t * omega) / sin_omega) * v1_flat
    
    return result.reshape(orig_shape).to(v0.dtype)


def slerp_merge(
    state_dict_a: OrderedDict,
    state_dict_b: OrderedDict,
    t: float = 0.5,
) -> OrderedDict:
    """SLERP 合并两个模型"""
    merged = OrderedDict()
    
    for key in state_dict_a:
        if key in state_dict_b:
            merged[key] = slerp(t, state_dict_a[key], state_dict_b[key])
        else:
            merged[key] = state_dict_a[key]
    
    for key in state_dict_b:
        if key not in state_dict_a:
            merged[key] = state_dict_b[key]
    
    return merged


# 验证 SLERP vs Linear
v0 = torch.randn(1000)
v1 = torch.randn(1000)

for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    linear_result = (1 - t) * v0 + t * v1
    slerp_result = slerp(t, v0, v1)
    print(f"t={t:.2f} | Linear 模长: {linear_result.norm():.4f} | "
          f"SLERP 模长: {slerp_result.norm():.4f}")
# 你会观察到：SLERP 的中间点模长更加平滑，而 Linear 在 t=0.5 时模长会变小
```

### 3. Task Arithmetic（任务算术）

核心思想：微调模型与基座模型的权重差（task vector）编码了特定任务的能力。这些 task vector 可以像向量一样进行加减运算。

$$\tau_A = \theta_A - \theta_{base}$$
$$\tau_B = \theta_B - \theta_{base}$$
$$\theta_{merged} = \theta_{base} + \lambda_A \cdot \tau_A + \lambda_B \cdot \tau_B$$

**直觉**：把基座模型想象成一个"技能树"的原点，每个微调模型都是从原点出发在某个方向上的移动。Task Arithmetic 就是把这些方向的移动叠加起来。

```python
def task_arithmetic_merge(
    base_state_dict: OrderedDict,
    expert_state_dicts: list[OrderedDict],
    weights: list[float],
) -> OrderedDict:
    """Task Arithmetic 合并：基座 + 加权任务向量之和
    
    Args:
        base_state_dict: 基座模型权重
        expert_state_dicts: 各个微调模型的权重列表
        weights: 每个任务向量的缩放系数
    """
    assert len(expert_state_dicts) == len(weights)
    
    merged = OrderedDict()
    
    for key in base_state_dict:
        # 从基座开始
        merged_param = base_state_dict[key].clone().float()
        
        # 累加每个任务向量
        for expert_sd, w in zip(expert_state_dicts, weights):
            if key in expert_sd:
                task_vector = expert_sd[key].float() - base_state_dict[key].float()
                merged_param += w * task_vector
        
        merged[key] = merged_param.to(base_state_dict[key].dtype)
    
    return merged

# 使用示例
# base = load("llama-3.1-8b")
# code_expert = load("llama-3.1-8b-code")
# math_expert = load("llama-3.1-8b-math")
# merged = task_arithmetic_merge(base, [code_expert, math_expert], [0.8, 0.6])
```

### 4. TIES（Trim, Elect Sign & Merge）

Task Arithmetic 的问题：不同任务向量之间可能存在**冲突**（同一个参数被拉向相反方向）和**噪声**（许多微小变化实际上是噪声）。TIES 通过三个步骤解决这些问题。

**步骤**：

1. **Trim（修剪）**：将 task vector 中绝对值小于阈值的参数归零，只保留变化最大的参数
2. **Elect Sign（选举符号）**：对每个参数位置，选择多数 task vector 一致的符号方向
3. **Merge（合并）**：只合并与选举符号一致的 task vector

$$\tau_A^{trimmed} = \text{TopK}(|\tau_A|) \odot \text{sign}(\tau_A)$$
$$\text{sign}_{elected} = \text{sign}\left(\sum_i \tau_i^{trimmed}\right)$$
$$\theta_{merged} = \theta_{base} + \lambda \sum_i \tau_i^{trimmed} \odot \mathbb{1}[\text{sign}(\tau_i^{trimmed}) = \text{sign}_{elected}]$$

```python
def ties_merge(
    base_state_dict: OrderedDict,
    expert_state_dicts: list[OrderedDict],
    weights: list[float],
    density: float = 0.5,  # 保留 top-k% 的参数
) -> OrderedDict:
    """TIES 合并：Trim + Elect Sign + Merge
    
    Args:
        base_state_dict: 基座模型权重
        expert_state_dicts: 微调模型权重列表
        weights: 每个任务向量的权重
        density: 保留参数的比例（0-1），越小越稀疏
    """
    merged = OrderedDict()
    
    for key in base_state_dict:
        base_param = base_state_dict[key].float()
        
        # Step 0: 计算所有任务向量
        task_vectors = []
        for expert_sd, w in zip(expert_state_dicts, weights):
            if key in expert_sd:
                tv = (expert_sd[key].float() - base_param) * w
                task_vectors.append(tv)
        
        if not task_vectors:
            merged[key] = base_state_dict[key]
            continue
        
        # Step 1: Trim（修剪）—— 保留绝对值最大的 top-k%
        trimmed_vectors = []
        for tv in task_vectors:
            threshold = torch.quantile(tv.abs().float(), 1 - density)
            mask = tv.abs() >= threshold
            trimmed_vectors.append(tv * mask)
        
        # Step 2: Elect Sign（选举符号）
        # 对每个参数位置，统计正负投票
        stacked = torch.stack(trimmed_vectors, dim=0)
        sign_sum = stacked.sum(dim=0)
        elected_sign = torch.sign(sign_sum)
        
        # Step 3: Merge（合并）—— 只保留与选举符号一致的分量
        merged_tv = torch.zeros_like(base_param)
        count = torch.zeros_like(base_param)
        
        for tv in trimmed_vectors:
            # 只累加与选举符号一致的参数
            agree_mask = (torch.sign(tv) == elected_sign) & (tv != 0)
            merged_tv += tv * agree_mask
            count += agree_mask.float()
        
        # 取平均（避免除以 0）
        count = torch.clamp(count, min=1)
        merged_tv = merged_tv / count
        
        merged[key] = (base_param + merged_tv).to(base_state_dict[key].dtype)
    
    return merged
```

### 5. DARE（Drop And REscale）

DARE 的核心思想更加激进：**随机丢弃** task vector 中大部分参数（类似 Dropout），然后**重新缩放**剩余参数以补偿丢弃的部分。

$$m \sim \text{Bernoulli}(p)$$
$$\tilde{\tau} = \frac{\tau \odot m}{p}$$
$$\theta_{merged} = \theta_{base} + \sum_i \lambda_i \cdot \tilde{\tau}_i$$

其中 $p$ 是保留概率（通常 0.1-0.3），$\frac{1}{p}$ 是缩放因子。

**直觉**：微调产生的大部分参数变化是冗余的（与 Lottery Ticket Hypothesis 类似的思想）。随机保留一小部分变化足以保留任务能力，而大幅减少参数冲突。

```python
def dare_merge(
    base_state_dict: OrderedDict,
    expert_state_dicts: list[OrderedDict],
    weights: list[float],
    drop_rate: float = 0.8,  # 丢弃比例
    seed: int = 42,
) -> OrderedDict:
    """DARE 合并：随机丢弃 + 重缩放
    
    Args:
        base_state_dict: 基座模型权重
        expert_state_dicts: 微调模型权重列表
        weights: 每个任务向量的权重
        drop_rate: 丢弃比例（0.8 表示只保留 20% 的参数变化）
        seed: 随机种子
    """
    torch.manual_seed(seed)
    keep_rate = 1.0 - drop_rate
    merged = OrderedDict()
    
    for key in base_state_dict:
        base_param = base_state_dict[key].float()
        merged_param = base_param.clone()
        
        for expert_sd, w in zip(expert_state_dicts, weights):
            if key in expert_sd:
                task_vector = expert_sd[key].float() - base_param
                
                # 随机丢弃
                mask = torch.bernoulli(torch.full_like(task_vector, keep_rate))
                
                # 重缩放：除以保留概率以保持期望不变
                dropped_tv = task_vector * mask / keep_rate
                
                merged_param += w * dropped_tv
        
        merged[key] = merged_param.to(base_state_dict[key].dtype)
    
    return merged
```

### 方法对比总结

| 方法 | 核心思想 | 需要基座？ | 超参数 | 适合场景 |
|------|---------|-----------|--------|---------|
| **Linear** | 加权平均 | 否 | $\alpha$ | 两个相似模型 |
| **SLERP** | 球面插值 | 否 | $t$ | 两个模型，保持能量 |
| **Task Arithmetic** | 任务向量相加 | 是 | $\lambda_i$ | 多个微调模型 |
| **TIES** | 修剪+选举+合并 | 是 | $\lambda$, density | 多模型，减少冲突 |
| **DARE** | 随机丢弃+缩放 | 是 | $\lambda$, drop_rate | 多模型，减少冲突 |

---

## mergekit 工具使用

[mergekit](https://github.com/arcee-ai/mergekit) 是目前最流行的模型合并工具，支持上述所有合并方法。

### 安装

```bash
pip install mergekit

# 或从源码安装（获取最新功能）
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit
pip install -e .
```

### 配置文件格式

mergekit 通过 YAML 配置文件定义合并策略。

```yaml
# === 示例 1: SLERP 合并两个模型 ===
# config_slerp.yml

merge_method: slerp
base_model: meta-llama/Llama-3.1-8B-Instruct  # 仅 SLERP 的方向参考
dtype: float16
parameters:
  t:
    - filter: self_attn     # attention 层的插值系数
      value: 0.6
    - filter: mlp           # MLP 层的插值系数
      value: 0.4
    - value: 0.5            # 其他层的默认值
models:
  - model: model_a_path
  - model: model_b_path
```

```yaml
# === 示例 2: TIES 合并多个模型 ===
# config_ties.yml

merge_method: ties
base_model: meta-llama/Llama-3.1-8B
dtype: float16
parameters:
  density: 0.5
  normalize: true
  int_space: true
models:
  - model: code-expert-8b
    parameters:
      weight: 0.6
      density: 0.6       # 代码模型保留更多参数
  - model: math-expert-8b
    parameters:
      weight: 0.4
      density: 0.4
```

```yaml
# === 示例 3: DARE + TIES ===
# config_dare_ties.yml

merge_method: dare_ties
base_model: meta-llama/Llama-3.1-8B
dtype: float16
parameters:
  density: 0.3       # 只保留 30% 的参数变化
  normalize: true
models:
  - model: expert-chat-8b
    parameters:
      weight: 1.0
  - model: expert-code-8b
    parameters:
      weight: 0.8
  - model: expert-math-8b
    parameters:
      weight: 0.6
```

### 运行合并

```bash
# 基本合并
mergekit-yaml config_slerp.yml ./merged_model --cuda

# 指定设备
mergekit-yaml config_ties.yml ./merged_model --cuda --device cuda:0

# CPU 合并（不需要 GPU，但更慢）
mergekit-yaml config_ties.yml ./merged_model

# 使用 lazy unpickle 减少内存占用
mergekit-yaml config_ties.yml ./merged_model --lazy-unpickle

# 合并后直接上传到 HuggingFace
mergekit-yaml config_ties.yml ./merged_model --cuda
huggingface-cli upload my-username/my-merged-model ./merged_model
```

---

## 实战：合并 LoRA 适配器

在实际场景中，常常需要合并多个 LoRA 微调适配器。

### 方法 1：先合并回基座再合并

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict

def merge_lora_models(
    base_model_id: str,
    lora_adapters: list[str],
    weights: list[float],
    merge_method: str = "task_arithmetic",
) -> AutoModelForCausalLM:
    """合并多个 LoRA 适配器
    
    流程：
    1. 分别将每个 LoRA 合并回基座，得到完整模型
    2. 用指定方法合并这些完整模型
    """
    # 加载基座模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype="auto"
    )
    base_sd = base_model.state_dict()
    
    # 获取每个 LoRA 合并后的完整模型权重
    expert_sds = []
    for adapter_path in lora_adapters:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype="auto"
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # 将 LoRA 合并回基座
        expert_sds.append(model.state_dict())
        del model
    
    # 使用 Task Arithmetic 合并
    if merge_method == "task_arithmetic":
        merged_sd = task_arithmetic_merge(base_sd, expert_sds, weights)
    elif merge_method == "ties":
        merged_sd = ties_merge(base_sd, expert_sds, weights, density=0.5)
    elif merge_method == "dare":
        merged_sd = dare_merge(base_sd, expert_sds, weights, drop_rate=0.8)
    else:
        raise ValueError(f"Unknown method: {merge_method}")
    
    base_model.load_state_dict(merged_sd)
    return base_model


# 使用示例
# merged = merge_lora_models(
#     base_model_id="meta-llama/Llama-3.1-8B",
#     lora_adapters=["./lora-code", "./lora-math", "./lora-chat"],
#     weights=[0.7, 0.5, 0.8],
#     merge_method="ties",
# )
# merged.save_pretrained("./merged-model")
```

### 方法 2：直接合并 LoRA 权重

如果多个 LoRA 共享相同的基座和配置，可以直接在 LoRA 权重空间中合并，更省内存。

```python
import torch
from safetensors.torch import load_file, save_file

def merge_lora_weights_directly(
    lora_paths: list[str],
    weights: list[float],
    output_path: str,
):
    """直接合并 LoRA 权重（不需要加载完整基座模型）
    
    LoRA 的权重结构：每个层有 lora_A 和 lora_B 两个矩阵
    ΔW = B @ A（低秩分解）
    
    合并策略：加权平均 A 和 B 矩阵
    注意：这在数学上不完全等价于合并完整权重，但实践中效果不错
    """
    # 加载所有 LoRA 权重
    all_lora_sds = []
    for path in lora_paths:
        sd = load_file(f"{path}/adapter_model.safetensors")
        all_lora_sds.append(sd)
    
    # 加权平均
    merged_sd = {}
    for key in all_lora_sds[0]:
        merged_sd[key] = sum(
            w * sd[key].float() for w, sd in zip(weights, all_lora_sds)
        ) / sum(weights)
        merged_sd[key] = merged_sd[key].to(all_lora_sds[0][key].dtype)
    
    # 保存（复制第一个 adapter 的配置）
    import shutil, json
    shutil.copy(f"{lora_paths[0]}/adapter_config.json", f"{output_path}/adapter_config.json")
    save_file(merged_sd, f"{output_path}/adapter_model.safetensors")
    
    print(f"合并了 {len(lora_paths)} 个 LoRA 适配器 → {output_path}")
```

---

## Frankenmerge（跨架构层拼接）

Frankenmerge 是一种更激进的合并方式：从不同模型中**挑选特定层**拼接成一个新模型。这类似于弗兰肯斯坦——从不同"身体"上取零件组装。

```yaml
# mergekit Frankenmerge 配置示例
# 从模型 A 取前 16 层，从模型 B 取后 16 层

merge_method: passthrough
slices:
  - sources:
      - model: model-a
        layer_range: [0, 16]    # 模型 A 的第 0-15 层
  - sources:
      - model: model-b
        layer_range: [16, 32]   # 模型 B 的第 16-31 层
dtype: float16
```

```yaml
# 更复杂的拼接：交错层
merge_method: passthrough
slices:
  - sources:
      - model: model-a
        layer_range: [0, 8]
  - sources:
      - model: model-b
        layer_range: [8, 16]
  - sources:
      - model: model-a
        layer_range: [16, 24]
  - sources:
      - model: model-b
        layer_range: [24, 32]
dtype: float16
```

> **注意**：Frankenmerge 通常用于相同架构的模型。不同架构（如 Llama 和 Mistral）的层维度可能不同，无法直接拼接。拼接后模型可能需要少量微调来"磨合"接口处的层。

---

## MoE 合并

mergekit 支持将多个专家模型合并为一个 Mixture of Experts（MoE）模型。这不是简单的权重平均，而是让每个专家保持完整，通过路由网络动态选择。

```yaml
# mergekit MoE 配置
# config_moe.yml

base_model: meta-llama/Llama-3.1-8B-Instruct
gate_mode: hidden            # 路由方式：hidden（基于隐状态）
dtype: float16
experts_per_token: 2         # 每个 token 激活几个专家
experts:
  - source_model: code-expert-8b
    positive_prompts:
      - "Write a Python function"
      - "Debug this code"
      - "Implement an algorithm"
  - source_model: math-expert-8b
    positive_prompts:
      - "Solve this math problem"
      - "Calculate the derivative"
      - "Prove that"
  - source_model: chat-expert-8b
    positive_prompts:
      - "Tell me about"
      - "How do I"
      - "Explain the concept"
```

```bash
# 运行 MoE 合并
mergekit-moe config_moe.yml ./moe_model --cuda

# 生成的模型使用 Mixtral 架构
# 可以直接用 vLLM 或 transformers 加载
```

**MoE 合并的优劣**：

| 优点 | 缺点 |
|------|------|
| 每个专家能力完全保留 | 模型参数量变大（N 倍） |
| 路由机制自动选择专家 | 推理时需要更多显存 |
| 理论上是最优的合并方式 | 路由训练依赖 positive_prompts 质量 |

---

## 合并后评估策略

合并完成后，必须系统评估合并效果。

```python
"""合并模型评估框架"""

class MergeEvaluator:
    """评估合并模型在各个任务上的表现
    
    核心思路：
    1. 在合并前记录各个专家模型的单项成绩
    2. 合并后与各个专家对比
    3. 理想结果：合并模型在所有任务上接近或超过各自的专家
    """
    
    def __init__(self, tasks: dict):
        """
        tasks: {"task_name": evaluate_fn}
        evaluate_fn 接收模型并返回分数
        """
        self.tasks = tasks
        self.results = {}
    
    def evaluate_model(self, model_name: str, model, tokenizer):
        """评估单个模型在所有任务上的表现"""
        self.results[model_name] = {}
        for task_name, eval_fn in self.tasks.items():
            score = eval_fn(model, tokenizer)
            self.results[model_name][task_name] = score
            print(f"  {model_name} on {task_name}: {score:.4f}")
    
    def compare(self):
        """打印对比表格"""
        import pandas as pd
        df = pd.DataFrame(self.results).T
        print("\n=== 合并效果对比 ===")
        print(df.to_string())
        
        # 计算合并模型相对各专家的保留率
        if "merged" in self.results:
            print("\n=== 能力保留率 ===")
            for task in self.tasks:
                best_expert = max(
                    (name for name in self.results if name != "merged"),
                    key=lambda n: self.results[n].get(task, 0),
                )
                expert_score = self.results[best_expert][task]
                merged_score = self.results["merged"][task]
                retention = merged_score / expert_score * 100 if expert_score > 0 else 0
                print(f"  {task}: {retention:.1f}% (vs {best_expert}: {expert_score:.4f})")


# 常用评估基准
# - MMLU: 多任务语言理解
# - HumanEval: 代码生成
# - GSM8K: 数学推理
# - MT-Bench: 对话质量
# - 使用 lm-evaluation-harness 运行：
#   lm_eval --model hf --model_args pretrained=./merged_model --tasks mmlu,gsm8k
```

### 使用 lm-evaluation-harness

```bash
# 安装
pip install lm-eval

# 评估原始专家模型
lm_eval --model hf \
    --model_args pretrained=code-expert-8b \
    --tasks humaneval,mbpp \
    --batch_size 8

# 评估合并模型
lm_eval --model hf \
    --model_args pretrained=./merged_model \
    --tasks humaneval,mbpp,gsm8k,mmlu \
    --batch_size 8

# 输出包含详细的每个任务分数，方便对比
```

---

## 实战：完整合并流程

```python
"""
完整示例：合并一个代码专家和一个数学专家

假设你已经有：
- 基座模型：meta-llama/Llama-3.1-8B
- 代码专家：./models/code-expert（SFT on code data）
- 数学专家：./models/math-expert（SFT on math data）
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict

def full_merge_pipeline(
    base_id: str,
    expert_paths: list[str],
    expert_weights: list[float],
    output_path: str,
    method: str = "ties",
):
    print(f"[1/4] 加载基座模型: {base_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.float16, device_map="cpu",
    )
    base_sd = base_model.state_dict()
    
    print(f"[2/4] 加载 {len(expert_paths)} 个专家模型")
    expert_sds = []
    for path in expert_paths:
        m = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map="cpu",
        )
        expert_sds.append(m.state_dict())
        del m
        torch.cuda.empty_cache()
    
    print(f"[3/4] 使用 {method} 方法合并")
    if method == "linear":
        # 只支持两个模型
        merged_sd = linear_merge(expert_sds[0], expert_sds[1], alpha=expert_weights[0])
    elif method == "slerp":
        merged_sd = slerp_merge(expert_sds[0], expert_sds[1], t=expert_weights[0])
    elif method == "task_arithmetic":
        merged_sd = task_arithmetic_merge(base_sd, expert_sds, expert_weights)
    elif method == "ties":
        merged_sd = ties_merge(base_sd, expert_sds, expert_weights, density=0.5)
    elif method == "dare":
        merged_sd = dare_merge(base_sd, expert_sds, expert_weights, drop_rate=0.8)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"[4/4] 保存合并模型到 {output_path}")
    base_model.load_state_dict(merged_sd)
    base_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("完成！")
    return base_model, tokenizer


# 调用示例
# model, tok = full_merge_pipeline(
#     base_id="meta-llama/Llama-3.1-8B",
#     expert_paths=["./models/code-expert", "./models/math-expert"],
#     expert_weights=[0.7, 0.5],
#     output_path="./models/merged-code-math",
#     method="ties",
# )
#
# # 快速验证
# inputs = tok("def fibonacci(n):", return_tensors="pt")
# output = model.generate(**inputs, max_new_tokens=100)
# print(tok.decode(output[0]))
```

---

## 苏格拉底时刻

1. **线性插值在两个模型之间取中间值可能落在"损失高地"上。为什么微调后的模型权重空间不一定是凸的？这与 Loss Landscape 的几何结构有什么关系？**

2. **DARE 随机丢弃 80% 的参数变化居然还能保持性能，这说明了什么？与 Lottery Ticket Hypothesis 有什么联系？如果大部分参数变化是冗余的，我们是否可以用更小的 LoRA rank 来微调？**

3. **模型合并不需要训练数据就能获得新能力，这是否意味着"能力"是以某种可分解的方式编码在权重中的？如果是，为什么我们不能精确提取和操作特定能力？**

4. **Frankenmerge 从不同模型取层拼接，为什么这有时能工作？不同层之间的"接口"（hidden state 的分布）不匹配怎么办？**

5. **MoE 合并让每个专家保持完整但模型变大了 N 倍。有没有一种方法既能保持专家完整性又不增加参数量？这在数学上是否可能？**

---

## 常见问题 & 面试考点

**Q: SLERP 和线性插值的本质区别是什么？什么时候该用 SLERP？**

A: 线性插值在欧氏空间中沿直线移动，中间点的向量模长会减小（对于非平行向量）。SLERP 在超球面上沿大圆弧移动，保持模长不变。当权重的方向（而非大小）更重要时，SLERP 更合适。实践中，对于相似度较高的两个模型，两者差异不大；对于差异较大的模型，SLERP 通常更稳定。

**Q: 为什么 TIES 和 DARE 比简单的 Task Arithmetic 效果好？**

A: 因为它们解决了任务向量的两个核心问题：(1) 冲突——同一参数被不同任务拉向相反方向，TIES 通过投票选举解决，DARE 通过随机稀疏化降低冲突概率；(2) 噪声——微调产生的大量微小参数变化可能是噪声而非有意义的信号，Trim/Drop 操作过滤了这些噪声。

**Q: 模型合并的上限在哪里？什么情况下合并不如多任务训练？**

A: 当任务之间的权重变化高度冲突（如中文→英文翻译 vs 英文→中文翻译，可能在 embedding 层有对抗性变化），合并效果会显著下降。此外，合并无法产生"新能力"——它只能组合已有能力。如果目标任务需要两个专家都不具备的能力，合并无法解决。

**Q: mergekit 的 YAML 配置中，不同层使用不同的合并系数有什么讲究？**

A: 通常的经验法则：(1) Attention 层编码了"关注什么"的模式，可能更任务特定；(2) MLP 层存储了更多的知识；(3) 底层（接近输入）通常更通用，高层（接近输出）更任务特定。因此可以对高层使用更大的专家权重，低层使用更均匀的混合。

---

## 推荐资源

- [mergekit GitHub](https://github.com/arcee-ai/mergekit) — 最流行的模型合并工具
- [Editing Models with Task Arithmetic (ICLR 2023)](https://arxiv.org/abs/2212.04089) — Task Arithmetic 原始论文
- [TIES-Merging (NeurIPS 2023)](https://arxiv.org/abs/2306.01708) — TIES 方法论文
- [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE)](https://arxiv.org/abs/2311.03099) — DARE 方法论文
- [SLERP 球面线性插值数学推导](https://en.wikipedia.org/wiki/Slerp) — 理解 SLERP 的几何直觉
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) — 许多排行榜前列的模型都使用了模型合并
- [Charles Goddard 的 mergekit 教程](https://huggingface.co/blog/mlabonne/merge-models) — mergekit 作者的实战教程
