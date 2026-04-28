---
title: "知识蒸馏"
description: "经典蒸馏（Hinton 2015）、LLM 黑盒/白盒蒸馏、推理蒸馏"
topics: [distillation, knowledge-distillation, soft-labels, temperature-scaling, Alpaca, Vicuna]
prereqs: [training/sft]
---
# 知识蒸馏

> 知识蒸馏将大模型的能力迁移到小模型，是大模型落地部署的关键技术

## 在大模型体系中的位置

```
预训练 (Pre-training)          → 学习语言知识和世界知识
    ↓
监督微调 (SFT)                 → 学习指令跟随能力
    ↓
偏好对齐 (RLHF/DPO/GRPO)      → 学习人类偏好，安全有用
    ↓
模型压缩  ← 你在这里            → 让模型更小、更快、更省
  ├── 知识蒸馏                  → 大模型教小模型
  ├── 模型量化                  → 降低数值精度
  ├── 模型剪枝                  → 去掉冗余参数
  └── LoRA/适配器               → 参数高效微调
```

大模型效果好但推理成本高昂。知识蒸馏的核心思想是：**让一个小模型（Student）学习大模型（Teacher）的行为模式，从而在更低的计算成本下获得接近的效果**。

## 蒸馏的动机

### 为什么需要蒸馏？

| 模型 | 参数量 | 推理速度 | 部署成本 |
|------|--------|---------|---------|
| GPT-4 | ~1.8T (传闻) | 慢 | 极高（多卡集群） |
| LLaMA-70B | 70B | 较慢 | 高（多卡） |
| LLaMA-7B | 7B | 中等 | 中（单卡） |
| **蒸馏后的 3B 模型** | 3B | 快 | **低（消费级 GPU）** |

一个 70B 模型蒸馏到 7B，推理速度提升约 10 倍，显存占用降低约 10 倍，而效果下降可能只有 5-15%。在很多实际场景中，这是一个极有吸引力的权衡。

### 蒸馏的本质：暗知识（Dark Knowledge）

Hinton 的核心洞察：**模型的"错误"输出也包含有价值的信息**。

```
问题: "苹果是什么？"

硬标签 (One-hot):
  水果: 1.0, 公司: 0.0, 颜色: 0.0, 动物: 0.0

Teacher 的软标签:
  水果: 0.85, 公司: 0.12, 颜色: 0.02, 动物: 0.01
```

软标签中 "公司: 0.12" 这个信息告诉 Student：苹果和公司概念之间存在某种关联（Apple 公司）。这种隐含在概率分布中的信息就是 **Dark Knowledge**——它无法从硬标签中获得。

## 经典知识蒸馏（Hinton 2015）

### 核心框架

Hinton 等人在 2015 年提出了知识蒸馏的经典框架：

```
Teacher Model (Large)
    │
    +-- Hard Labels ──→ Hard Loss (Standard CE Loss)
    │                          │
    +-- Soft Labels ──→ Soft Loss (KL Divergence)
         (T>1)                 │
                        ┌──────┴──────┐
                        │             │
              alpha x Soft Loss + (1-alpha) x Hard Loss = Total Loss
                        │
              Student Model (Small)
```

### Temperature Scaling

Temperature 是蒸馏中最关键的超参数。它控制概率分布的"软化"程度：

$$
q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

其中 $z_i$ 是 logit（softmax 前的值），$T$ 是温度。

```python
import torch
import torch.nn.functional as F

def softmax_with_temperature(logits, temperature):
    """带温度的 Softmax"""
    return F.softmax(logits / temperature, dim=-1)

# 示例：不同温度下的概率分布
logits = torch.tensor([5.0, 3.0, 1.0, 0.5])

print("T=1 (标准):", softmax_with_temperature(logits, T=1.0))
# tensor([0.8360, 0.1131, 0.0153, 0.0093])   ← 尖锐分布，几乎只关注最大值

print("T=3 (中等):", softmax_with_temperature(logits, T=3.0))
# tensor([0.4764, 0.2384, 0.1193, 0.0978])   ← 更平滑，各类别差异可见

print("T=10 (高温):", softmax_with_temperature(logits, T=10.0))
# tensor([0.3072, 0.2653, 0.2290, 0.2193])   ← 接近均匀分布

print("T→∞:", softmax_with_temperature(logits, T=100.0))
# tensor([0.2574, 0.2524, 0.2476, 0.2463])   ← 几乎均匀
```

**温度的直觉理解**：

- $T=1$：标准 softmax，峰值尖锐，只能看到"最可能的答案"
- $T>1$：分布变平滑，暴露出类别之间的相对关系（Dark Knowledge）
- $T \to \infty$：趋向均匀分布，所有信息消失

实践中 $T \in [2, 20]$ 效果较好，$T=4$ 是常见的起点。

### 完整数学推导

**蒸馏损失的推导**：

给定 Teacher 的 logits $z^T$ 和 Student 的 logits $z^S$，在温度 $T$ 下的软化概率为：

$$
p_i^T = \frac{\exp(z_i^T / T)}{\sum_j \exp(z_j^T / T)}, \quad
q_i^S = \frac{\exp(z_i^S / T)}{\sum_j \exp(z_j^S / T)}
$$

**Soft Loss** 使用 KL 散度衡量 Student 和 Teacher 软分布的差异：

$$
\mathcal{L}_{\text{soft}} = T^2 \cdot \text{KL}(p^T \| q^S) = T^2 \sum_i p_i^T \log \frac{p_i^T}{q_i^S}
$$

> **为什么要乘以 $T^2$？** 当 $T > 1$ 时，softmax 后的梯度量级与 $1/T^2$ 成正比。乘以 $T^2$ 确保在不同温度下梯度量级一致，使得 $\alpha$ 的选择不受 $T$ 影响。

**推导过程**：对 $\mathcal{L}_{\text{soft}}$ 关于 Student logit $z_i^S$ 求导：

$$
\frac{\partial \mathcal{L}_{\text{soft}}}{\partial z_i^S} = T^2 \cdot \frac{1}{T}(q_i^S - p_i^T) = T(q_i^S - p_i^T)
$$

当 $T$ 较大时，$q_i^S \approx \frac{1}{N} + \frac{z_i^S}{NT}$，$p_i^T \approx \frac{1}{N} + \frac{z_i^T}{NT}$，代入得：

$$
\frac{\partial \mathcal{L}_{\text{soft}}}{\partial z_i^S} \approx T \cdot \frac{z_i^S - z_i^T}{NT} = \frac{z_i^S - z_i^T}{N}
$$

这说明在高温极限下，蒸馏损失近似于 **Student 和 Teacher logits 的 MSE**。

**Hard Loss** 是标准的交叉熵损失（使用真实标签 $y$）：

$$
\mathcal{L}_{\text{hard}} = -\sum_i y_i \log q_i^S \quad (\text{其中 } q_i^S \text{ 使用 } T=1)
$$

**总蒸馏损失**：

$$
\mathcal{L}_{\text{distill}} = \alpha \cdot \mathcal{L}_{\text{soft}} + (1 - \alpha) \cdot \mathcal{L}_{\text{hard}}
$$

其中 $\alpha$ 控制两部分的权重。Hinton 推荐 $\alpha$ 较大（如 0.7~0.9），因为 Teacher 的软标签包含更多信息。

### PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """经典知识蒸馏损失"""
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        """
        Args:
            student_logits: Student 模型的 logits, (batch, num_classes)
            teacher_logits: Teacher 模型的 logits, (batch, num_classes)
            labels: 真实标签, (batch,)
        """
        # Soft Loss: KL(Teacher_soft || Student_soft) * T^2
        soft_teacher = F.softmax(teacher_logits / self.T, dim=-1)
        soft_student = F.log_softmax(student_logits / self.T, dim=-1)
        soft_loss = F.kl_div(
            soft_student, soft_teacher,
            reduction='batchmean'
        ) * (self.T ** 2)

        # Hard Loss: 标准交叉熵
        hard_loss = self.ce_loss(student_logits, labels)

        # 总损失
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return loss


# 使用示例
teacher_logits = torch.tensor([[5.0, 3.0, 1.0], [1.0, 5.0, 2.0]])
student_logits = torch.tensor([[4.0, 2.5, 1.5], [1.5, 4.5, 2.0]])
labels = torch.tensor([0, 1])

criterion = DistillationLoss(temperature=4.0, alpha=0.7)
loss = criterion(student_logits, teacher_logits, labels)
print(f"Distillation Loss: {loss.item():.4f}")
```

训练流程的核心是 Teacher 冻结（`eval()` + `torch.no_grad()`），只更新 Student 的参数。完整实现见下方代码实战部分。

## LLM 时代的蒸馏方法

传统蒸馏是在分类任务上对齐 Teacher 和 Student 的输出分布。在 LLM 时代，蒸馏方法更加多样化。

### 黑盒蒸馏：数据驱动

**核心思路**：无法访问 Teacher 的权重和 logits，只能通过 API 获取 Teacher 的文本输出，然后用这些输出作为训练数据对 Student 进行 SFT。

```
Teacher (闭源大模型，如 GPT-4)
    │
    └── API 调用 → 生成高质量回答
                        │
                        ↓
              收集 (prompt, response) 数据
                        │
                        ↓
              Student (开源小模型) 做 SFT
```

```python
def black_box_distillation(teacher_api, seed_tasks):
    """黑盒蒸馏：用 Teacher API 生成数据 → 过滤 → Student SFT"""
    distill_data = []
    for task in seed_tasks:
        response = teacher_api.generate(prompt=task, temperature=0.7, max_tokens=1024)
        distill_data.append({"instruction": task, "output": response})

    # 质量过滤：去掉太短/太长/拒绝回答的样本
    filtered = [d for d in distill_data
                if 50 < len(d['output']) < 2048
                and not any(kw in d['output'] for kw in ["我无法", "作为AI"])]
    return filtered  # 用过滤后的数据做 SFT
```

**代表项目**：
- **Alpaca**：用 GPT-3.5 生成 52K 条指令数据，微调 LLaMA-7B
- **Vicuna**：用 ShareGPT 对话数据（来自 ChatGPT）微调 LLaMA-13B
- **WizardLM**：用 Evol-Instruct 方法逐步增加指令难度

### 白盒蒸馏：知识深度对齐

**核心思路**：可以访问 Teacher 的权重，直接对齐 Teacher 和 Student 的内部表示。

#### Logit 层蒸馏

最直接的方式，对齐 Teacher 和 Student 在每个 token 位置的输出分布：

$$
\mathcal{L}_{\text{logit}} = \sum_{t=1}^{T} \text{KL}\left( p_{\text{teacher}}(\cdot | x_{<t}) \| p_{\text{student}}(\cdot | x_{<t}) \right)
$$

```python
def logit_distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Logit 层蒸馏：对齐每个 token 位置的输出分布

    Args:
        student_logits: (batch, seq_len, vocab_size)
        teacher_logits: (batch, seq_len, vocab_size)
    """
    batch_size, seq_len, vocab_size = student_logits.shape

    # 在 token 维度展平
    student_flat = student_logits.view(-1, vocab_size)
    teacher_flat = teacher_logits.view(-1, vocab_size)

    # KL 散度 + Temperature Scaling
    soft_student = F.log_softmax(student_flat / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_flat / temperature, dim=-1)

    loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
    loss = loss * (temperature ** 2)
    return loss
```

#### 隐藏层蒸馏

不仅对齐输出，还对齐中间隐藏层的表示（如 TinyBERT）。由于 Teacher 和 Student 的隐藏层维度不同，需要投影层：

$$
\mathcal{L}_{\text{hidden}} = \sum_{l \in \mathcal{L}} \text{MSE}\left( W_l \cdot h_{\text{student}}^{l_s}, h_{\text{teacher}}^{l_t} \right)
$$

实现要点：定义层映射（如 Student 第 2 层对齐 Teacher 第 6 层），用 `nn.Linear` 投影 Student 隐藏状态到 Teacher 维度，然后计算 MSE。

### 推理蒸馏：蒸馏 CoT 能力

推理蒸馏是 LLM 时代最具价值的蒸馏方式——将大推理模型（如 O1、DeepSeek-R1）的 Chain-of-Thought 能力迁移到小模型：

流程：(1) Teacher 对每个问题采样 8 个推理过程；(2) 筛选答案正确的推理路径，选最短的（更简洁更好学）；(3) 用筛选后的 `(问题, <think>推理</think>答案)` 数据对 Student 做 SFT。详见 [reasoning.md](reasoning.md) 的推理蒸馏部分。

**DeepSeek 的推理蒸馏实验结果**：

| Student 模型 | 蒸馏数据量 | MATH 500 | AIME 2024 |
|-------------|-----------|----------|-----------|
| Qwen-2.5-1.5B (无蒸馏) | 0 | 36.6% | 3.3% |
| Qwen-2.5-1.5B (蒸馏) | 800K | 83.9% | 28.6% |
| Qwen-2.5-7B (蒸馏) | 800K | 92.8% | 55.5% |
| Qwen-2.5-32B (蒸馏) | 800K | 94.3% | 72.6% |

1.5B 模型经过蒸馏后在 MATH 上从 36.6% 跃升到 83.9%，效果惊人。

### On-Policy Distillation：让学生在自己的 rollout 上被打分

上面的黑盒/白盒/推理蒸馏都属于 **off-policy 蒸馏**——训练数据来自 Teacher（或 Teacher 采样的固定语料），学生只是模仿这些"理想轨迹"。Thinking Machines Lab 在 2025 年 10 月的博客 [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)（Kevin Lu 等）系统化提出了第三种范式：**让学生自己 rollout，再用 Teacher 对学生轨迹的每个 token 打 logp**。

| 方法 | 采样来源 | 监督密度 |
|------|----------|----------|
| SFT / 白盒蒸馏 | off-policy（Teacher 数据） | dense（每 token） |
| RL（RLHF/GRPO） | on-policy（Student 自采） | sparse（每条轨迹一个 reward） |
| **On-Policy Distillation** | **on-policy（Student 自采）** | **dense（每 token reverse KL）** |

**为什么 on-policy 比 off-policy 好**：off-policy 的学生只学会了在 Teacher 常去的状态分布上模仿，一旦自己早期推错一步，就会进入 Teacher 训练轨迹里没有出现过的"陌生状态"，错误持续累积。on-policy 蒸馏让学生在**自己将来真会遇到的状态**上学习，并由 Teacher 在每个 token 上给出 reverse KL 作为 per-token 奖励，因此可以视为"GRPO 但 reward 换成 $-\text{KL}(\pi_\theta \| \pi_\text{teacher})$"——一行代码就能在任何 RL 训练栈上接入。

> 博客原文："The core idea of on-policy distillation is to sample trajectories from the *student* model and use a high-performing teacher to grade *each token* of each trajectory." 在 Qwen3-8B 的数学推理实验上，on-policy 蒸馏以 1,800 GPU·小时把 AIME'24 从 60% 推到 74.4%，而同等水平的 RL 需要 17,920 GPU·小时（约 9–10× 加速，FLOPs 口径下 50–100×）。

## 实用蒸馏流水线

### 数据生成流水线

高质量的蒸馏数据是成功的关键。完整的数据生成流水线包括：

```
Seed Tasks（种子任务）
    ↓
指令进化（Evol-Instruct）
    ↓
Teacher 生成回答
    ↓
质量过滤 & 去重
    ↓
蒸馏训练数据
```

```python
def evol_instruct(seed_instruction, teacher_api, depth=3):
    """Evol-Instruct：逐步进化指令难度（WizardLM 方案）"""
    instruction = seed_instruction
    evolution_prompts = [
        "请将以下指令变得更加复杂和具体：\n{instruction}",
        "请为以下指令添加约束条件和特殊要求：\n{instruction}",
        "请将以下指令扩展为需要多步推理的任务：\n{instruction}",
    ]
    for d in range(depth):
        prompt = evolution_prompts[d % len(evolution_prompts)]
        instruction = teacher_api.generate(
            prompt.format(instruction=instruction), temperature=0.7)
    return instruction

def build_distillation_dataset(seed_tasks, teacher_api, target_size=50000):
    """构建蒸馏数据集：种子进化 → Teacher 生成 → 过滤去重"""
    dataset = []
    for seed in seed_tasks:
        # 指令进化 + Teacher 生成回答
        tasks = [seed] + [evol_instruct(seed, teacher_api, d+1) for d in range(3)]
        for task in tasks:
            response = teacher_api.generate(task, temperature=0.7)
            dataset.append({"instruction": task, "output": response})
        if len(dataset) >= target_size:
            break

    dataset = quality_filter(dataset)
    # 去重：基于 n-gram 相似度，阈值 0.85
    dataset = deduplicate_by_ngram(dataset, threshold=0.85)
    return dataset
```

### 训练策略

#### 渐进式蒸馏

不是一步到位蒸馏到最小模型，而是逐步缩小，每一步的 Student 成为下一步的 Teacher：

```
Teacher (70B) → 蒸馏 → 中间模型 (30B) → 蒸馏 → Student (7B) → 蒸馏 → 更小模型 (1.5B)
```

优势：每一步压缩比小，信息损失更少。实验表明 70B→7B 直接蒸馏的效果不如 70B→30B→7B 两步蒸馏。

#### 多教师蒸馏

用多个 Teacher 的加权软标签来训练 Student，融合不同模型的优势：

$$
\mathcal{L}_{\text{multi}} = \sum_{k=1}^{K} w_k \cdot T^2 \cdot \text{KL}(p_k^T \| q^S) + (1-\alpha) \cdot \mathcal{L}_{\text{hard}}
$$

例如用一个擅长数学的 Teacher 和一个擅长代码的 Teacher，按领域加权融合。

## Alpaca / Vicuna 等项目的蒸馏实践

### Alpaca：指令数据蒸馏的先驱

```
175 条人工编写的种子指令
    ↓
GPT-3.5 (text-davinci-003) 生成 52K 条指令-回答对
    ↓
微调 LLaMA-7B
    ↓
Stanford Alpaca（效果接近 text-davinci-003 的 7B 模型）
```

Alpaca 的成本分析：
- 数据生成：~$500（调用 GPT-3.5 API）
- 训练：~$100（4 张 A100 训练 3 小时）
- **总计 ~$600 得到一个接近 GPT-3.5 水平的 7B 模型**

### Vicuna：对话数据蒸馏

```
ShareGPT 平台上用户与 ChatGPT 的真实对话（70K 条）
    ↓
数据清洗 & 格式化
    ↓
微调 LLaMA-13B
    ↓
Vicuna-13B（GPT-4 评测达到 ChatGPT 的 92% 水平）
```

### 蒸馏的法律和伦理问题

大多数闭源模型（GPT-4、Claude）的使用条款**禁止使用其输出训练竞品模型**。蒸馏实践需要关注：

| 方面 | 注意事项 |
|------|---------|
| 使用条款 | 检查 Teacher 模型的 ToS 是否允许蒸馏 |
| 数据许可 | 确认蒸馏数据的使用许可 |
| 模型许可 | 蒸馏模型是否继承 Teacher 的许可限制 |
| 学术研究 | 多数有 research-only 豁免 |

## 蒸馏 vs LoRA vs 量化

| 维度 | 知识蒸馏 | LoRA | 量化 |
|------|---------|------|------|
| **目标** | 训练一个更小的模型 | 高效微调现有模型 | 降低数值精度 |
| **模型大小** | 大幅缩小（如 70B→7B） | 不变（仅添加少量参数） | 不变（但权重占用减半/四分之一） |
| **推理速度** | 线性提升（参数少） | 不变或略慢 | 提升（低精度计算更快） |
| **效果损失** | 5-15%（取决于压缩比） | 1-3% | 1-5%（取决于量化位数） |
| **训练成本** | 高（需要 Teacher 生成数据） | 低（只训练少量参数） | 几乎为零（后训练量化） |
| **适用场景** | 大规模部署、边缘设备 | 快速适配新任务 | 降低推理成本 |
| **可组合性** | 可与 LoRA、量化组合使用 | 可与量化组合使用 | 可与蒸馏、LoRA 组合 |

**实践建议**：

```
需要在手机/嵌入式设备上运行？
    → 蒸馏到小模型 + 量化

需要快速适配新任务但不改变部署？
    → LoRA 微调

需要降低服务器推理成本？
    → 量化（INT8/INT4）

追求极致效果的小模型？
    → 蒸馏 + LoRA 微调 + 量化（三者组合）
```

## 代码实战：用 Temperature Scaling 实现简单蒸馏

以下是端到端的蒸馏实战，使用 PyTorch 在 MNIST 上对比蒸馏与直接训练：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 模型定义
class TeacherNet(nn.Module):
    """大模型：3 层 MLP，~538K 参数"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 10))
    def forward(self, x): return self.net(x)

class StudentNet(nn.Module):
    """小模型：1 层 MLP，~51K 参数（压缩 10x）"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))
    def forward(self, x): return self.net(x)

# 蒸馏训练核心
def train_with_distillation(teacher, student, loader, T=4.0, alpha=0.7, epochs=5):
    teacher.eval()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    for epoch in range(epochs):
        student.train()
        correct, total = 0, 0
        for images, labels in loader:
            optimizer.zero_grad()
            with torch.no_grad():
                t_logits = teacher(images)
            s_logits = student(images)

            # 蒸馏损失 = α * T² * KL(teacher_soft || student_soft) + (1-α) * CE
            soft_loss = F.kl_div(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction='batchmean') * (T ** 2)
            hard_loss = F.cross_entropy(s_logits, labels)
            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            loss.backward()
            optimizer.step()
            correct += (s_logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch+1}: Acc={100*correct/total:.1f}%")

# 运行实验
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True,
                          transform=transform), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=256)

# 1. 训练 Teacher
teacher = TeacherNet()
optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3)
for epoch in range(5):
    teacher.train()
    for img, lbl in train_loader:
        optimizer.zero_grad()
        F.cross_entropy(teacher(img), lbl).backward()
        optimizer.step()

# 2. 蒸馏训练 Student vs 直接训练 Student
student_distill = StudentNet()
train_with_distillation(teacher, student_distill, train_loader, T=4.0, alpha=0.7)

student_baseline = StudentNet()
opt = torch.optim.Adam(student_baseline.parameters(), lr=1e-3)
for _ in range(5):
    student_baseline.train()
    for img, lbl in train_loader:
        opt.zero_grad()
        F.cross_entropy(student_baseline(img), lbl).backward()
        opt.step()

# 3. 评估对比
def evaluate(model):
    model.eval()
    correct = sum((model(img).argmax(1)==lbl).sum().item() for img,lbl in test_loader)
    return 100 * correct / len(test_loader.dataset)

print(f"Teacher:           {evaluate(teacher):.1f}%  (538K params)")
print(f"Student (蒸馏):     {evaluate(student_distill):.1f}%  (51K params)")
print(f"Student (直接训练):  {evaluate(student_baseline):.1f}%  (51K params)")
```

**预期输出**：

```
Teacher:           98.3%  (538K params)
Student (蒸馏):     97.6%  (51K params)   ← 蒸馏效果，10x 压缩仅损失 0.7%
Student (直接训练):  97.0%  (51K params)   ← 基线
```

在复杂的 NLP 任务上，蒸馏的提升通常在 2-5% 甚至更多。

## 苏格拉底时刻

1. **为什么蒸馏比直接训练小模型效果好？** 软标签提供了比硬标签更丰富的监督信号。一个 one-hot 标签只有 1 bit 信息（类别），而一个 softmax 分布包含了所有类别之间的相对关系。小模型从这些额外信息中受益。
2. **Temperature 为什么不能太高也不能太低？** 太低（T 接近 1）分布太尖锐，暗知识被淹没；太高（T 远大于 10）分布趋向均匀，信号消失。最优 T 取决于任务复杂度和 Teacher 的置信度。
3. **黑盒蒸馏是否违反了 Scaling Law？** 表面上看，小模型不应该达到大模型的效果。但蒸馏不是凭空创造能力，而是用 Teacher 的知识提供了更高效的训练信号。Student 的上限仍然受其参数量限制。
4. **为什么 Alpaca/Vicuna 能用如此少的数据取得好效果？** 因为基座模型（LLaMA）已经通过预训练获得了强大的语言能力，蒸馏数据只需要"激活"和"对齐"这些能力。这也是为什么数据质量比数量更重要。
5. **蒸馏和迁移学习有什么区别？** 迁移学习通常是用预训练模型的参数作为初始化；蒸馏是用 Teacher 的行为（输出分布）作为训练信号。两者可以组合：用预训练参数初始化 Student，再用 Teacher 的输出进行蒸馏。

## 常见问题 & 面试考点

- **Q: 蒸馏损失中 $T^2$ 的作用是什么？** 补偿高温 softmax 导致的梯度缩小。没有 $T^2$，温度越高梯度越小，soft loss 的贡献会被 hard loss 淹没。
- **Q: $\alpha$ 如何选择？** 当 Teacher 很强时（远优于 Student），增大 $\alpha$ 更多依赖软标签；当 Teacher 不太可靠时，减小 $\alpha$ 更多依赖硬标签。通常从 0.5 开始搜索。
- **Q: 白盒蒸馏和黑盒蒸馏哪个效果更好？** 白盒蒸馏通常更好，因为可以对齐更细粒度的信息（logit 分布、隐藏层表示）。但黑盒蒸馏的优势是可以利用闭源模型（如 GPT-4）。
- **Q: 蒸馏模型能否超过 Teacher？** 在同一任务上通常不能。但在特定子集上可以——如果 Student 在某些类型的输入上过拟合 Teacher 的模式，可能在这些输入上表现更好。此外，多 Teacher 蒸馏可能超过任何单个 Teacher。
- **Q: LLM 蒸馏和传统蒸馏有什么本质区别？** 传统蒸馏对齐的是分类分布（几十到几千类），LLM 蒸馏对齐的是 token 级别的生成分布（词表大小通常 32K~128K），且需要处理序列生成的自回归特性。

## 推荐资源

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) - Hinton 2015，知识蒸馏开山之作
- [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351) - 隐藏层蒸馏
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) - 自我指令生成
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) - 指令蒸馏实践
- [Vicuna: An Open-Source Chatbot](https://lmsys.org/blog/2023-03-30-vicuna/) - 对话蒸馏
- [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244) - Evol-Instruct
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - 推理蒸馏的最新实践
- [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) - Thinking Machines Lab (Kevin Lu, 2025-10-27)：学生自己 rollout、Teacher 逐 token 打 reverse KL 的蒸馏新范式，配套 [Tinker cookbook 实现](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation)
- [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) - Agarwal et al. 2023，on-policy 蒸馏的早期理论

### 代码参考：minimind 的白盒蒸馏

[jingyaogong/minimind](https://github.com/jingyaogong/minimind) 是 26M 参数的中文社区"从零搭 LLM"完整训练栈，pretrain / SFT / LoRA / DPO / GRPO / 蒸馏全部纯 PyTorch 手撕、单 GPU 2 小时跑通。它的 `trainer/train_distillation.py` 把上文 Hinton 公式直接落地为可跑代码：[L24-L35 的 `distillation_loss`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_distillation.py#L24-L35) 严格按照 Hinton 标准做法对 KL 乘 `temperature**2` 缩放，[L67-L91 的 α 混合](https://github.com/jingyaogong/minimind/blob/master/trainer/train_distillation.py#L67-L91) 则展示了一个 LLM 白盒蒸馏里容易踩的细节——**用 `loss_mask_flat` 过滤 padding 后再算 KL**，并把学生词表大小作为 teacher logits 的截断索引（`teacher_logits[..., :vocab_size_student]`），这两点本文上方"Logit 层蒸馏"一节没有展开。

```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return (temperature ** 2) * kl

# ... 训练循环里：
ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
distill_loss = distillation_loss(
    student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
    teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
    temperature=temperature,
)
loss = alpha * distill_loss + (1 - alpha) * ce_loss
```

`alpha` 控制 soft/hard 两路损失的混合比例，与上文 Hinton 公式一一对应；teacher 在 `torch.no_grad()` 下前向、`.detach()` 概率分布，是显存与梯度安全的工程惯例。
