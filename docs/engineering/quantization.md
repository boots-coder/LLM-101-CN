---
title: "量化"
description: "INT8/INT4/NF4 量化、PTQ/QAT、GPTQ、AWQ、GGUF 格式"
topics: [quantization, INT8, INT4, NF4, PTQ, QAT, GPTQ, AWQ, GGUF]
prereqs: [architecture/transformer]
---
# 量化

::: info 一句话总结
量化通过降低数值精度（FP32 → FP16 → INT8 → INT4）来压缩模型，是大模型从实验室走向生产部署的关键技术。
:::


## 在大模型体系中的位置

```
预训练 → SFT/RLHF → 部署优化
                      ├── 量化 ◄── 你在这里
                      ├── 蒸馏
                      ├── 剪枝
                      └── 推理框架 (vLLM, TensorRT-LLM)
```

量化属于**模型压缩**技术，处于训练完成之后、上线部署之前的关键环节。它与推理框架（vLLM、TensorRT-LLM）配合使用，共同解决"大模型怎么跑起来"的问题。

## 为什么需要量化？

### 显存计算

一个参数在不同精度下占用的字节数：

| 数据类型 | 每参数字节 | 说明 |
|---------|-----------|------|
| FP32 | 4 bytes | 单精度浮点，训练默认精度 |
| FP16/BF16 | 2 bytes | 半精度，推理常用基线 |
| INT8 | 1 byte | 8-bit 整数量化 |
| INT4/NF4 | 0.5 byte | 4-bit 量化，消费级显卡可用 |

用这个公式估算显存需求：**显存 ≈ 参数量 × 每参数字节**

| 模型规模 | FP32 | FP16 | INT8 | INT4 |
|---------|------|------|------|------|
| **7B** | 28 GB | 14 GB | 7 GB | **3.5 GB** |
| **13B** | 52 GB | 26 GB | 13 GB | **6.5 GB** |
| **70B** | 280 GB | 140 GB | 70 GB | **35 GB** |

> 这意味着：Llama-2 70B 在 FP16 下需要 2 张 A100 80GB，但 INT4 量化后仅需 1 张 A100 40GB 或一张消费级 RTX 4090 (24GB) + CPU offload。

### 推理延迟分析

LLM 推理通常是 **memory-bound**（显存带宽瓶颈），而非 compute-bound。原因是 Transformer 的 decode 阶段每次只生成一个 token，计算量小但需要读取全部模型权重。

```
推理延迟 ≈ 模型大小 / 显存带宽
```

| 场景 | FP16 模型大小 | INT4 模型大小 | 带宽节省 |
|------|-------------|-------------|---------|
| Llama-7B 单次前向 | 14 GB 读取 | 3.5 GB 读取 | **4x** |

因此量化不仅省显存，还能通过减少内存读取量来**直接加速推理**，这一点常被初学者忽略。

## 量化基础

### 均匀量化

量化的核心操作是将连续的浮点数映射到离散的低位整数。

**对称量化（Symmetric Quantization）：**

$$x_{int} = \text{round}\left(\frac{x_{float}}{s}\right), \quad s = \frac{\max(|x|)}{2^{b-1} - 1}$$

反量化：$x_{float} \approx s \cdot x_{int}$

**非对称量化（Asymmetric Quantization）：**

$$x_{int} = \text{round}\left(\frac{x_{float}}{s}\right) + z, \quad s = \frac{x_{max} - x_{min}}{2^b - 1}, \quad z = -\text{round}\left(\frac{x_{min}}{s}\right)$$

反量化：$x_{float} \approx s \cdot (x_{int} - z)$

```python
import torch

def symmetric_quantize(x: torch.Tensor, bits: int = 8):
    """对称量化：将浮点张量映射到 [-2^(b-1)+1, 2^(b-1)-1] 的整数范围"""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().max() / qmax                # 计算缩放因子
    x_int = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    return x_int, scale

def symmetric_dequantize(x_int: torch.Tensor, scale: float):
    """反量化：将整数还原为近似的浮点数"""
    return x_int.float() * scale

# 示例
w = torch.randn(4, 4)  # 模拟一层权重
w_int, scale = symmetric_quantize(w, bits=8)
w_approx = symmetric_dequantize(w_int, scale)
print(f"量化误差: {(w - w_approx).abs().mean():.6f}")  # 通常在 0.001 级别
```

**对称 vs 非对称的选择：** 权重分布通常以 0 为中心，适合对称量化；激活值分布可能偏移（如 ReLU 后全为正），适合非对称量化。

### 量化粒度

量化粒度决定了多少个数值共享同一个 scale/zero_point：

| 粒度 | 说明 | 精度 | 开销 |
|------|------|------|------|
| **Per-tensor** | 整个张量共享一个 scale | 低 | 最小 |
| **Per-channel** | 每个输出通道一个 scale | 中 | 中等 |
| **Per-group** | 每 g 个元素一个 scale（常用 g=128） | 高 | 较大 |

> 实践中 INT4 量化几乎都使用 **per-group** 粒度（group_size=128），因为 4-bit 的表达能力太有限，需要更细的粒度来控制误差。

### 量化误差分析

量化引入的误差可以分解为：

1. **舍入误差**：round 操作的固有误差，均匀分布在 $[-s/2, s/2]$
2. **截断误差**：超出量化范围的值被 clamp，离群值受影响最大
3. **累积误差**：逐层量化时，前一层的误差会传播到后续层

大模型对量化误差具有天然的鲁棒性——大量冗余参数意味着微小扰动不会显著改变输出分布。这是量化能够成功的理论基础。

## INT8 量化

### LLM.int8()（bitsandbytes）

LLM.int8() 的关键发现：**激活值中存在离群值（outliers）**。在大模型（≥6.7B）中，某些隐藏维度的激活值绝对值远大于其他维度（可达 100 倍以上），且这些维度在所有 token 上都是一致的。

如果直接做 INT8 量化，离群值会"吃掉"整个量化范围，导致大量正常值被压缩到 0 或 1。

**解决方案 —— 混合精度分解（Mixed-precision Decomposition）：**

```
矩阵乘法 XW:
1. 检测激活 X 中的离群维度（|x| > 6.0 的维度）
2. 将 X 和 W 按维度拆分为：
   - 离群部分 → FP16 计算（约占 0.1% 的维度）
   - 正常部分 → INT8 计算（约占 99.9% 的维度）
3. 两部分结果相加
```

```python
from transformers import AutoModelForCausalLM

# 使用 bitsandbytes 加载 INT8 量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,          # 启用 LLM.int8() 量化
    device_map="auto"           # 自动分配到可用 GPU
)
# 显存需求从 14GB 降至约 7-8GB（额外的 FP16 离群值开销）
```

### SmoothQuant

SmoothQuant 从另一个角度解决激活值离群值问题：**既然激活值难量化、权重好量化，那就把量化难度从激活"搬"到权重上。**

数学上等价的变换：

$$Y = X W = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X} \hat{W}$$

其中 $s$ 是每个通道的平滑因子：$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$，$\alpha$ 是迁移强度（通常 0.5）。

变换后 $\hat{X}$ 的分布更均匀（离群值被压缩），$\hat{W}$ 稍微变大但仍可接受，两者都可以用 INT8 量化——实现了真正的 **W8A8**（权重和激活都是 INT8），推理速度优于 LLM.int8() 的混合精度方案。

## INT4 量化

INT4 量化将精度进一步压缩，每个参数仅用 4 bit 存储。4 bit 只能表示 16 个不同的值，因此需要更精巧的量化策略来控制精度损失。

### NormalFloat (NF4)

NF4 是 QLoRA 论文中提出的 **信息论最优的 4-bit 数据类型**。

核心洞察：预训练模型的权重分布近似正态分布 $\mathcal{N}(0, \sigma^2)$。既然分布已知，就可以设计一种数据类型，使得 16 个量化级别在正态分布的 CDF 上等距分布——每个量化 bin 包含相同数量的值，从而最小化信息损失。

```python
# NF4 的 16 个量化值（归一化后）
# 注意它们在数轴上并非等距，而是在概率密度高的区域（接近0）更密集
NF4_VALUES = [
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
     0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230, 1.0
]
```

NF4 相比普通 INT4（均匀量化）在正态分布数据上的量化误差更小，这就是 QLoRA 选择 NF4 的原因。

### bitsandbytes FP4 实现

以下代码来自 bitsandbytes 库的 FP4 数据类型实现，展示了如何用浮点格式的思路构建 4-bit 量化码本：

```python
import torch
import itertools

def create_fp8_map(signed=True, exponent_bits=5, precision_bits=2, total_bits=8):
    """
    创建浮点量化码本（适用于 FP4/FP8 等不同位宽）
    通过枚举所有可能的 bit pattern，按照 IEEE 754 风格计算每个模式对应的浮点值
    
    参数：
        signed: 是否有符号
        exponent_bits: 指数位数（FP4 通常为 2）
        precision_bits: 尾数位数（FP4 通常为 1）
        total_bits: 总位数
    """
    e = exponent_bits
    p = precision_bits
    has_sign = 1 if signed else 0
    assert e + p == total_bits - has_sign

    evalues = []
    pvalues = []
    for i, val in enumerate(range(-(2 ** (exponent_bits - has_sign)),
                                   2 ** (exponent_bits - has_sign), 1)):
        evalues.append(2 ** val)

    values = []
    lst = list(itertools.product([0, 1], repeat=precision_bits))  # 所有尾数的 bit 组合
    bias = 2 ** (exponent_bits - 1)  # 指数偏置

    for evalue in range(2 ** exponent_bits):
        for bit_pattern in lst:
            # 根据指数是否为 0 区分 normal 和 subnormal
            value = 1 if evalue != 0 else 0  # normal 数有隐含的前导 1
            for i, pval in enumerate(list(bit_pattern)):
                value += pval * (2 ** -(i + 1))  # 累加尾数位的贡献
            if evalue == 0:
                # subnormal 数：指数固定为 -bias
                value = value * 2 ** -(bias)
            else:
                # normal 数：指数 = evalue - bias - 1
                value = value * 2 ** -(evalue - bias - 1)
            values.append(value)
            if signed:
                values.append(-value)

    assert len(values) == 2 ** total_bits
    values.sort()
    if total_bits < 8:
        gap = 256 - len(values)
        for i in range(gap):
            values.append(0)
    values.sort()
    code = torch.Tensor(values)
    code /= code.max()  # 归一化到 [-1, 1]

    return code

# 生成 FP4 码本：2-bit 指数 + 1-bit 尾数 + 1-bit 符号 = 4 bit
fp4_full = create_fp8_map(signed=True, exponent_bits=2, precision_bits=1, total_bits=4)

# FP4 只有 16 个有效值，但函数返回 256 个（为了对齐 FP8 接口），取首尾各 8 个
fp4 = torch.zeros(16)
fp4[:8] = fp4_full[:8]
fp4[-8:] = fp4_full[-8:]

print("FP4 码本（16个量化级别）:")
print(fp4)
```

> 这段代码展示了 FP4 如何用类似 IEEE 754 的浮点编码来设计 4-bit 码本——与 NF4 的"正态分布等概率"思路不同，FP4 保留了浮点数的 subnormal/normal 结构。

### GPTQ

GPTQ（GPT Quantization）是一种基于**二阶信息**的训练后量化（PTQ）方法：

**核心思想：** 量化的目标是最小化量化前后每层输出的差异。使用 Hessian 矩阵（二阶导数）来衡量每个权重对输出的影响，优先精确量化对输出影响大的权重，将误差补偿到其他未量化的权重上。

**算法流程：**

1. 使用少量校准数据（128 条）前向传播，收集每层的输入激活
2. 逐层量化：对每层权重矩阵，逐列处理
3. 每量化一列，利用 Hessian 逆矩阵将量化误差分散到后续列
4. 这是 OBQ（Optimal Brain Quantization）的高效近似

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 配置 GPTQ 量化：4-bit，使用 c4 数据集校准
quantization_config = GPTQConfig(
    bits=4,                      # 量化位数
    dataset="c4",                # 校准数据集
    tokenizer=tokenizer,         # 分词器
    group_size=128,              # 每 128 个权重共享一个 scale
)

# 加载并量化模型（约需 4 分钟 for 7B 模型）
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### AWQ（Activation-aware Weight Quantization）

AWQ 的核心洞察：**不是所有权重都同等重要**。

通过观察校准数据上的激活值分布，AWQ 发现：仅 1% 的"显著权重通道"（salient channels）对模型输出有决定性影响。如果这 1% 的权重被精确量化，整体精度几乎不受损。

**方法：**

1. 统计每个权重通道对激活值的影响（$\text{saliency}_j = ||W_j|| \cdot ||X_j||$）
2. 对重要通道乘以一个缩放因子 $s$，等效于提升它们的量化精度
3. 搜索最优的 $s$ 使量化误差最小

**AWQ vs GPTQ：**

- GPTQ 依赖特定校准数据的 Hessian 信息 → 在对应分布上精度高，但可能过拟合
- AWQ 基于通道重要性的统计规律 → 不依赖特定数据分布，泛化性更好

## GGUF 格式

### 为什么需要 GGUF？

PyTorch 格式和 Safetensors 格式的模型文件需要 Python 生态和 GPU。GGUF（GPT-Generated Unified Format）为 **llama.cpp** 设计，目标是：

- **纯 CPU 推理**：在没有 GPU 的电脑上也能运行大模型
- **自包含**：模型权重 + 分词器 + 架构参数全部打包在一个文件里
- **灵活量化**：支持从 2-bit 到 8-bit 的多种量化级别

### 量化级别

| 量化方法 | 每参数比特 | 7B 模型大小 | 质量评估 |
|---------|-----------|------------|---------|
| Q2_K | 2.5 | ~2.7 GB | 质量损失明显 |
| Q3_K_S | 3.4 | ~3.0 GB | 可用但有损 |
| Q4_0 | 4.0 | ~3.8 GB | 基础 4-bit，速度快 |
| **Q4_K_M** | **4.8** | **~4.1 GB** | **推荐：质量/大小平衡** |
| Q5_K_S | 5.4 | ~4.8 GB | 较高质量 |
| Q5_K_M | 5.7 | ~5.1 GB | 高质量 |
| Q6_K | 6.6 | ~5.9 GB | 接近 FP16 |
| Q8_0 | 8.0 | ~7.2 GB | 几乎无损 |

> 命名规则：`Q{bits}_K_{S/M/L}` 中 K 表示使用 k-quant 方法，S/M/L 表示小/中/大的不同权重分组策略。

### llama.cpp 量化流程

```bash
# 1. 从 HuggingFace 下载模型（Safetensors 格式）
# 2. 转换为 GGUF 格式
python convert_hf_to_gguf.py ./Llama-2-7B-Chat/ --outfile llama2-7b-chat-f16.gguf

# 3. 量化到目标精度
./llama-quantize llama2-7b-chat-f16.gguf llama2-7b-chat-Q4_K_M.gguf Q4_K_M

# 4. 运行推理
./llama-cli -m llama2-7b-chat-Q4_K_M.gguf -p "你好" -n 128

# 或使用 ollama（更简单的方式）
ollama run llama2:7b-q4_K_M
```

## PTQ vs QAT

### Post-Training Quantization (PTQ)

训练后量化：在已训练好的模型上直接量化，**无需重新训练**。

- 优势：快速（分钟级）、不需要训练数据和 GPU 训练资源
- 劣势：高压缩比（如 2-bit）下精度损失显著
- 代表方法：GPTQ、AWQ、bitsandbytes

### Quantization-Aware Training (QAT)

量化感知训练：在训练过程中模拟量化误差，让模型学会"适应"低精度。

- 优势：在极低比特（2-3 bit）下仍能保持较好精度
- 劣势：需要训练资源，耗时更长
- 代表方法：LLM-QAT、QLoRA（部分 QAT 特性）

### 选择指南

| 场景 | 推荐方案 |
|------|---------|
| 快速部署，4-bit 精度可接受 | PTQ (AWQ/GPTQ) |
| 需要极低比特量化 | QAT |
| 在消费级 GPU 上微调 | QLoRA (NF4 + LoRA) |
| CPU / 边缘设备部署 | GGUF (llama.cpp) |

## 各方案对比表格

| 方法 | 精度 | 量化速度 | GPU 推理 | CPU 推理 | 生态 | 核心特点 |
|------|------|---------|----------|----------|------|---------|
| **GPTQ** | 高 | 快 (分钟级) | 是 | 否 | exllama, AutoGPTQ | Hessian 信息，特定数据精度高 |
| **AWQ** | 高 | 快 | 是 | 否 | vLLM 原生支持 | 激活感知，泛化性好 |
| **GGUF** | 中-高 | 中 | 是 | **是** | llama.cpp, ollama | CPU 友好，格式自包含 |
| **bitsandbytes** | 中 | 即时 | 是 | 否 | HuggingFace 集成 | 使用最简单，支持 QLoRA |

> 实际选择建议：如果用 vLLM 部署 → AWQ；如果需要 CPU 运行 → GGUF；如果做微调 → bitsandbytes + QLoRA；如果追求特定任务精度 → GPTQ。

## 实战复现：手撕量化

> 本节复现 Maxime Labonne 的 [Introduction to Quantization](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html) notebook（14 cell，中文翻译版）。它从零实现 **absmax** 与 **zero-point** 两种 INT8 量化，再调用 `bitsandbytes` 体验 **LLM.int8()** 的离群值处理——是把上文公式落到 PyTorch 代码上的最短路径。

### Absmax 量化：对称量化的最简版本

把张量按 $\max(|x|)$ 缩放到 INT8 的 $[-127, 127]$ 区间，scale 是单一标量，反量化时直接除回去——这就是对称量化的"最小可运行实现"。

```python
import torch

def absmax_quantize(X):
    # 计算 scale：用张量的绝对值最大值把 [-max, max] 映射到 [-127, 127]
    scale = 127 / torch.max(torch.abs(X))

    # 量化：先放缩再四舍五入
    X_quant = (scale * X).round()

    # 反量化：除回 scale，得到近似的浮点值
    X_dequant = X_quant / scale

    return X_quant.to(torch.int8), X_dequant
```

::: tip 为什么是 127 而不是 128
INT8 的真实范围是 $[-128, 127]$，但对称量化故意丢掉 -128 这一格，让正负两侧严格对称、scale 只有一个标量。多丢一个码字换简洁——量化里很常见的 trade-off。
:::

### Zero-point 量化：非对称量化的标准实现

权重大致以 0 为中心，但激活值（尤其 ReLU 后）会偏移。zero-point 引入一个整数偏置 $z$，把任意区间 $[\min, \max]$ 平移对齐到 INT8 的全 256 个格点。

```python
def zeropoint_quantize(X):
    # 计算数值范围（作为分母）
    x_range = torch.max(X) - torch.min(X)
    x_range = 1 if x_range == 0 else x_range

    # 计算 scale：把整个动态范围映射到 INT8 的 256 个格点
    scale = 255 / x_range

    # 计算 zero-point：使最小值对齐到 -128
    zeropoint = (-scale * torch.min(X) - 128).round()

    # 放缩 + 平移 + 四舍五入 + 截断到 [-128, 127]
    X_quant = torch.clip((X * scale + zeropoint).round(), -128, 127)

    # 反量化
    X_dequant = (X_quant - zeropoint) / scale

    return X_quant.to(torch.int8), X_dequant
```

对比 absmax：分母从 `2 * |x|_max` 换成 `max - min`、出现 `zeropoint` 平移项、`clip` 显式生效。本节后面会看到，在 GPT-2 的权重上两者 perplexity 几乎相同——但激活值上区别会立刻拉开。

### 把整个模型量化掉：deepcopy + 原地替换

理解单层之后，对 `model.parameters()` 逐一调用、用反量化结果原地覆盖 `param.data`，就得到一个"权重已经过 INT8 round-trip"的 GPT-2 副本。这一步是把"量化算子"升级到"量化模型"的桥梁。

```python
import numpy as np
from copy import deepcopy

# 备份原始权重
weights = [param.data.clone() for param in model.parameters()]

# 创建一份副本用于 absmax 量化
model_abs = deepcopy(model)

# 对全部参数做 absmax 量化（用反量化后的权重原地替换）
weights_abs = []
for param in model_abs.parameters():
    _, dequantized = absmax_quantize(param.data)
    param.data = dequantized
    weights_abs.append(dequantized)
```

::: warning 这是模拟量化（fake quant），不是真 INT8 推理
权重虽然经过了 INT8 round-trip，但 `param.data` 仍然是 FP32 张量、矩阵乘法仍在 FP32 上跑——节省不了显存也加速不了。这种写法的目的是**研究量化误差对输出的影响**，不是部署。真正的 INT8 推理需要专用 kernel（bitsandbytes / cutlass / TensorRT）。
:::

### 用困惑度量化「量化的代价」

光看权重直方图还不够，要回答"量化后模型还能不能用"——困惑度（perplexity）是最直接的指标。teacher forcing 下用同一段文本算 NLL，再 `exp` 一下：

```python
def calculate_perplexity(model, text):
    encodings = tokenizer(text, return_tensors='pt').to(device)
    input_ids = encodings.input_ids
    target_ids = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

    # 取 loss（即平均 negative log likelihood），perplexity = exp(NLL)
    neg_log_likelihood = outputs.loss
    ppl = torch.exp(neg_log_likelihood)
    return ppl

ppl     = calculate_perplexity(model, original_text)
ppl_abs = calculate_perplexity(model_abs, original_text)
ppl_zp  = calculate_perplexity(model_zp, original_text)
```

GPT-2 上原作者得到的典型数值：原始 ≈ 15.5、absmax ≈ 17、zero-point ≈ 17——量化后 perplexity 仅上升 ~10%，这就是大模型对量化误差具有鲁棒性的实证证据。

### LLM.int8()：交给 bitsandbytes 处理离群值

最后一步从"自己手撕"切换到"调用工业级实现"——`load_in_8bit=True` 一行触发 bitsandbytes 的混合精度分解：99.9% 的正常维度走 INT8、0.1% 的离群维度仍走 FP16，自动接管整张图。

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 通过 bitsandbytes 加载 LLM.int8() 量化模型（需要 GPU）
model_int8 = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map='auto',
                                             load_in_8bit=True,
                                             )
print(f"Model size: {model_int8.get_memory_footprint():,} bytes")

# 直接生成 + 算 perplexity，对比之前的手撕版本
text_int8 = generate_text(model_int8, "I have a dream")
ppl = calculate_perplexity(model_int8, text_int8)
print(f"Perplexity (LLM.int8()): {ppl.item():.2f}")
```

::: tip 为什么手撕 absmax/zp 在 GPT-2 上够用、到了 7B 就崩
GPT-2 的 124M 参数还没出现明显的离群维度，所以 naive 的 absmax/zp 也能保住 perplexity。一旦模型 ≥ 6.7B，少数维度的激活值会大上百倍，直接 INT8 量化会把绝大多数正常值压成 0/1——这就是上文 LLM.int8() 章节解释的「混合精度分解」要解决的问题。**手撕 → bitsandbytes** 的对照实验，正是为了让你亲眼看见这条「scale 上的鸿沟」。
:::

### 跑通建议

1. **前 9 个 cell（absmax / zero-point / 权重直方图 / 文本生成 / perplexity）纯 CPU 即可**，GPT-2 124M 在 MacBook 上几分钟就能跑完
2. **第 10-13 个 cell 需要 GPU**——`load_in_8bit=True` 依赖 bitsandbytes 的 CUDA kernel，CPU/MPS 上会报 `RuntimeError`；Colab 免费 T4 / Kaggle P100 都够用
3. **依赖**：`transformers >= 4.30`、`accelerate`、`bitsandbytes >= 0.39`；macOS 装 bitsandbytes 推荐用 `bitsandbytes-foundation` fork 或直接跳过 INT8 部分
4. **显存**：GPT-2 + LLM.int8() 模型本身只占 ~100MB，但生成 + perplexity 计算会再吃几百 MB，总占用 < 2GB
5. **画图慢**：cell 7 / 11 用了 `dpi=300`，渲染 1.5M 个权重的直方图在 CPU 上可能要 10-20 秒，不是卡死

## 苏格拉底时刻

停下来思考以下问题，不急于查看答案：

::: details 1. 量化为什么能减少显存但几乎不损失模型性能？
大模型的权重分布近似正态分布，大量参数值集中在 0 附近——这些参数之间的差异在量化后仍可保留。更重要的是，神经网络对权重的小扰动具有鲁棒性：量化引入的误差类似于训练时的正则化噪声，不会显著改变模型的输出分布。此外，通过 group-wise 量化，每一小组权重都有独立的 scale，进一步保留了局部精度。
:::


::: details 2. 为什么激活值比权重更难量化？
权重在推理时固定，分布稳定且近似正态，容易找到合适的量化参数。激活值则依赖于输入，分布变化大。更关键的是，大模型的激活值中存在**离群维度**——少数维度的数值可以达到其他维度的 100 倍以上。如果强行用统一的 scale 量化，大量正常值会被压缩到 0 或 1，信息严重丢失。LLM.int8() 和 SmoothQuant 分别用混合精度分解和平滑变换来解决这个问题。
:::


::: details 3. GPTQ 和 AWQ 的核心区别是什么？
GPTQ 用 Hessian 矩阵（二阶导数）来衡量每个权重的"灵敏度"，将量化误差从高灵敏度权重补偿到低灵敏度权重——它依赖校准数据来估计 Hessian，因此结果与校准数据分布相关。AWQ 则从激活值角度出发，通过统计每个权重通道的重要性来决定保护策略——它关注的是通道级的统计规律而非逐权重的梯度信息，因此泛化性更强。简言之：GPTQ 优化的是"量化误差的数学最优解"，AWQ 优化的是"对模型输出影响最大的权重"。
:::


::: details 4. NF4 为什么比普通 INT4 更适合量化大模型？
普通 INT4 的 16 个量化级别在数轴上等距分布，但模型权重集中在 0 附近（正态分布）。NF4 的设计让 16 个量化级别在正态分布的 CDF 上等距——即每个量化 bin 包含等概率的权重值。在 0 附近（权重密集区域）量化级别更密，在尾部（权重稀疏区域）更疏。这符合信息论中**最优量化器**的设计原则：在数据密集区分配更多码字。
:::


## 常见问题 & 面试考点

::: tip 面试高频问题

**Q: 量化的本质是什么？**
A: 量化是一种有损压缩：用更少的 bit 来近似表示模型参数，通过牺牲微小精度换取显存和速度的显著提升。核心挑战是如何设计量化策略使信息损失最小。

**Q: 为什么推理用 INT4 但训练不能用 INT4？**
A: 训练需要梯度更新，梯度通常很小（1e-4 量级），INT4 的精度（只有 16 个值）完全无法表示。此外，训练的前向/反向传播需要较高精度来避免累积误差。推理只需前向传播且不更新权重，对精度要求低得多。

**Q: 量化模型能否继续微调？**
A: 直接微调困难（梯度精度不够），但 QLoRA 巧妙地解决了这个问题：基础模型冻结在 NF4，只训练 FP16/BF16 的 LoRA 适配器。梯度只需通过低秩适配器回传，绕开了量化权重的精度限制。
:::


## 推荐资源

- **Dettmers et al. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"** — 离群值分解的开创性工作
- **Dettmers et al. "QLoRA: Efficient Finetuning of Quantized Language Models"** — NF4 + 双重量化 + 分页优化器
- **Frantar et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"** — 基于 Hessian 的逐层量化
- **Lin et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"** — 激活感知的权重保护
- **Xiao et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"** — 平滑量化
- **[MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)** — 各量化模型的评测对比
- **[llama.cpp Wiki](https://github.com/ggerganov/llama.cpp/wiki)** — GGUF 格式的完整文档
