---
title: "多模态大模型"
description: "ViT、CLIP、LLaVA、原生多模态、视频/音频、Vision Agent、多模态 RM/DPO"
topics: [multimodal, ViT, CLIP, LLaVA, vision-transformer, contrastive-learning, Qwen-VL, InternVL, native-multimodal, GPT-4o, Gemini, video-LLM, audio-LLM, vision-agent, OmniParser, multimodal-DPO, multimodal-RM]
prereqs: [architecture/transformer, training/alignment]
---
# 多模态大模型

> **一句话总结:** 多模态大模型通过将视觉编码器与语言模型对齐，让 LLM 不仅能理解文字，还能"看懂"图像，实现跨模态的理解与推理。

## 为什么需要多模态？

人类认知世界是多模态的 — 我们同时通过视觉、听觉、触觉来理解环境。纯文本 LLM 只能处理语言，这严重限制了它的应用场景。多模态大模型的目标是：

- 图文理解：看图回答问题、描述图片内容
- 文档解析：理解包含表格、图表的文档
- 视觉推理：基于图像内容进行逻辑推理
- 图像生成：根据文字描述创作图像

## ViT（Vision Transformer）深度解析

ViT 是将 Transformer 应用于视觉领域的里程碑工作。核心思想出奇简单：**把图像当作"句子"来处理**。

### ViT 的工作流程

```
输入图像 (224×224×3)
    ↓
切分为 Patch (16×16 = 196个patch)
    ↓
每个 Patch 线性投影为向量 (patch embedding, dim=768)
    ↓
加上位置编码 + [CLS] token
    ↓
送入标准 Transformer Encoder (L=12, H=12)
    ↓
[CLS] token 的输出 → 分类头
```

### Patch Embedding 实现

Patch Embedding 的本质是把每个 16x16x3 的图像块通过一个线性层映射为一个 D 维向量。实际实现中常用 `Conv2d` 来高效完成这个操作：

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """将图像切分为 patch 并投影为嵌入向量"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # 196
        # 用 Conv2d 替代 "展平 + Linear"，等价但更高效
        # kernel_size = stride = patch_size，恰好无重叠地切分图像
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224)
        x = self.proj(x)          # (B, 768, 14, 14)
        x = x.flatten(2)          # (B, 768, 196)
        x = x.transpose(1, 2)     # (B, 196, 768)  — 每个 patch 一个 token
        return x
```

### Position Embedding 与 [CLS] Token

```python
class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768,
                 depth=12, num_heads=12, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        # [CLS] token：可学习的分类标记，放在序列最前面
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置编码：可学习的绝对位置编码（不是 sinusoidal 的）
        # 长度为 num_patches + 1（包含 [CLS]）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)                                    # (B, 196, 768)
        cls_tokens = self.cls_token.expand(B, -1, -1)              # (B, 1, 768)
        x = torch.cat([cls_tokens, x], dim=1)                      # (B, 197, 768)
        x = x + self.pos_embed                                      # 加位置编码
        x = self.encoder(x)                                         # Transformer
        x = self.norm(x[:, 0])                                      # 取 [CLS] 输出
        return self.head(x)                                          # 分类
```

### ViT 的关键设计总结

| 设计点 | 实现方式 | 说明 |
|--------|----------|------|
| Patch Embedding | Conv2d(kernel=stride=16) | 等价于 "展平 + Linear"，效率更高 |
| 位置编码 | 可学习参数 | 不同于原始 Transformer 的 sinusoidal |
| [CLS] Token | 可学习参数 | 聚合全局信息用于分类 |
| 归纳偏置 | 几乎没有 | 不假设局部性、平移不变性，需要大量数据 |

### ViT 的启示

ViT 证明了一个重要观点：**Transformer 的注意力机制本身足够通用**，不需要卷积等视觉特有的归纳偏置。但代价是需要大量数据（ViT 在 JFT-300M 上预训练才超越 CNN）。后续 DeiT 通过蒸馏 + 数据增强证明在 ImageNet 规模也能训练好 ViT。

## CLIP（对比语言-图像预训练）深度解析

CLIP（Contrastive Language-Image Pre-training）是连接视觉和语言的桥梁。它学习一个**共享的嵌入空间**，使得匹配的图文对距离近，不匹配的距离远。

### 双编码器架构

```
图像 ──→ [图像编码器(ViT-L/14)] ──→ 图像特征 ──→ [线性投影] ──→ 图像向量 (d=512) ──┐
                                                                                  ├→ 余弦相似度矩阵
文本 ──→ [文本编码器(Transformer)] ──→ [EOS]特征 ──→ [线性投影] ──→ 文本向量 (d=512) ──┘
```

关键设计点：
- **图像编码器**：ViT-L/14（L 层 Transformer，patch size 14）或 ResNet
- **文本编码器**：12 层 Transformer，最大 77 token
- **投影层**：将两个编码器的输出投影到同一维度空间
- **归一化**：投影后做 L2 归一化，余弦相似度等价于内积

### 对比学习损失 InfoNCE

给定一个 batch 的 N 个图文对，CLIP 计算 N×N 的相似度矩阵，对角线是正样本对：

```
        文本1   文本2   文本3   文本4
图像1   [0.9]   0.1    0.05   0.02    ← 对角线是正样本
图像2    0.05  [0.85]  0.08   0.03
图像3    0.02   0.06  [0.88]  0.04
图像4    0.03   0.04   0.02  [0.92]
```

损失函数是对称的 InfoNCE（Info Noise Contrastive Estimation）：

```python
def clip_loss(image_features, text_features, temperature):
    """
    InfoNCE 对比学习损失
    Args:
        image_features: (N, D) L2 归一化的图像嵌入
        text_features:  (N, D) L2 归一化的文本嵌入
        temperature:    可学习的温度参数（标量）
    """
    # 计算相似度矩阵 (N, N)
    logits = (image_features @ text_features.T) / temperature

    # 标签：对角线为正样本，即 labels = [0, 1, 2, ..., N-1]
    labels = torch.arange(len(logits), device=logits.device)

    # 对称损失：图像→文本 + 文本→图像
    loss_i2t = F.cross_entropy(logits, labels)       # 每行做 softmax
    loss_t2i = F.cross_entropy(logits.T, labels)     # 每列做 softmax
    return (loss_i2t + loss_t2i) / 2
```

### Temperature 参数的作用

Temperature 是 CLIP 中一个关键的可学习参数：

- **Temperature 小**（如 0.01）：softmax 分布更尖锐，模型更自信，正负样本分离更明确
- **Temperature 大**（如 1.0）：softmax 分布更平滑，梯度更均匀
- CLIP 中初始化为 `temperature = nn.Parameter(torch.tensor(1/0.07))`，即约 14.3
- 训练过程中模型自动学习最优温度

### Zero-Shot 分类原理

CLIP 最惊艳的能力是**零样本分类**：无需额外训练就能识别新类别。其原理本质是将分类问题转化为图文匹配问题：

```python
import clip

# 加载模型
model, preprocess = clip.load("ViT-B/32", device="cuda")

# 1. 构造文本提示（prompt engineering 很重要）
class_names = ["cat", "dog", "bird", "car", "airplane"]
text_prompts = [f"a photo of a {c}" for c in class_names]

# 2. 编码文本（可以离线计算一次，缓存起来）
text_tokens = clip.tokenize(text_prompts).to("cuda")
with torch.no_grad():
    text_features = model.encode_text(text_tokens)       # (5, 512)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 3. 编码图像
image = preprocess(PIL.Image.open("cat.jpg")).unsqueeze(0).to("cuda")
with torch.no_grad():
    image_features = model.encode_image(image)            # (1, 512)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

# 4. 计算相似度 → 分类
similarities = (image_features @ text_features.T).squeeze(0)  # (5,)
probs = similarities.softmax(dim=0)
# → tensor([0.92, 0.03, 0.02, 0.02, 0.01])  → 预测为 cat
```

**Prompt Engineering 对 CLIP 零样本性能影响巨大**：
- `"cat"` → 准确率较低
- `"a photo of a cat"` → 显著提升
- `"a photo of a cat, a type of pet"` → 进一步提升
- CLIP 论文中使用了 80 个 prompt 模板做 ensemble

### 简化 CLIP 对比学习代码实战

以下是一个可运行的简化 CLIP 训练代码：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCLIP(nn.Module):
    """简化版 CLIP：理解对比学习的核心机制"""
    def __init__(self, image_encoder, text_encoder, embed_dim=256):
        super().__init__()
        self.image_encoder = image_encoder   # 任意图像编码器
        self.text_encoder = text_encoder     # 任意文本编码器
        # 投影头：将编码器输出映射到共享空间
        self.image_proj = nn.Linear(image_encoder.output_dim, embed_dim)
        self.text_proj = nn.Linear(text_encoder.output_dim, embed_dim)
        # 可学习温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0 / 0.07).log())

    def forward(self, images, input_ids, attention_mask):
        # 编码
        image_feat = self.image_encoder(images)                      # (B, img_dim)
        text_feat = self.text_encoder(input_ids, attention_mask)     # (B, txt_dim)

        # 投影 + L2 归一化
        image_embed = F.normalize(self.image_proj(image_feat), dim=-1)  # (B, embed_dim)
        text_embed = F.normalize(self.text_proj(text_feat), dim=-1)     # (B, embed_dim)

        # 计算对比损失
        temperature = self.temperature.exp()                     # 确保正数
        logits = (image_embed @ text_embed.T) * temperature      # (B, B)
        labels = torch.arange(len(logits), device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2

# 训练循环示例
model = SimpleCLIP(image_encoder, text_encoder, embed_dim=256).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

for images, input_ids, attention_mask in dataloader:
    loss = model(images.cuda(), input_ids.cuda(), attention_mask.cuda())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## LLaVA（大语言与视觉助手）深度解析

LLaVA（Large Language and Vision Assistant）是多模态大模型的代表性工作，展示了一种简洁而有效的架构：**视觉编码器 + 投影层 + LLM**。

### LLaVA 架构详解

```
输入图像 (336×336)
    ↓
[CLIP ViT-L/14 @336px 视觉编码器]  ← 冻结，不参与训练
    ↓
视觉特征 (576 tokens, 每个 1024-dim)    ← 24×24 = 576 个 patch
    ↓
[MLP 投影层: Linear(1024, 4096) → GELU → Linear(4096, 4096)]  ← LLaVA-1.5 使用 2 层 MLP
    ↓
视觉 Token (576 个, 每个 4096-dim)      ← 与 LLM 的 hidden_dim 对齐
    ↓
[拼接] ← 系统提示 tokens + 视觉 tokens + 用户指令 tokens
    ↓
[LLM (Vicuna-7B / Llama-2)]           ← 阶段一冻结，阶段二微调
    ↓
文本回答（自回归生成）
```

**投影层的演进**：
- LLaVA v1：单层 Linear → 效果已不错，证明了简单方案的可行性
- LLaVA-1.5：2 层 MLP (Linear → GELU → Linear) → 性能显著提升
- 更复杂的方案（如 Q-Former、Perceiver Resampler）不一定更好

### Visual Instruction Tuning：两阶段训练

**阶段一：特征对齐预训练**

```
目标：让投影层学会将视觉特征"翻译"为 LLM 能理解的表示
冻结：视觉编码器 ✓, LLM ✓
训练：仅投影层
数据：CC3M 筛选的 595K 图文对 → 生成 caption 任务
     格式：<image> Describe this image in detail. → "A cat sitting on..."
训练时长：约 4 小时 (8× A100)
```

**阶段二：端到端指令微调**

```
目标：让模型学会遵循指令进行多种视觉任务
冻结：视觉编码器 ✓
训练：投影层 + LLM 全部参数
数据：158K 多模态指令数据（GPT-4/GPT-4V 辅助生成）
     包括：详细描述、复杂推理、多轮对话
训练时长：约 10 小时 (8× A100)
```

**指令数据构建的巧妙之处**：

LLaVA 的核心创新之一是用 GPT-4 生成高质量指令数据。流程为：
1. 将图像的 caption 和 bounding box 描述输入 GPT-4（纯文本）
2. GPT-4 基于这些描述生成三类问答：
   - **对话**：自然语言问答（"图片里有什么？"）
   - **详细描述**：长篇细致描述
   - **复杂推理**：需要多步逻辑的问题

### LLaVA 的设计洞察

1. **投影层是关键**：一个简单的 MLP 就能有效连接视觉和语言模态，无需复杂的跨模态注意力
2. **数据质量 > 数量**：用 GPT-4 生成的高质量指令数据，比大量低质量数据更有效
3. **冻结视觉编码器**：利用 CLIP 已学到的强大视觉表示，避免灾难性遗忘
4. **视觉 token 作为"外语"**：把视觉 token 直接拼接到文本序列中，LLM 将它们当作一种新的"语言"来理解

## 国产多模态模型

### Qwen-VL（通义千问视觉）

```
架构：ViT-G/14 (1.9B) + Qwen-7B
关键创新：
├── 多分辨率输入：支持 448×448 高分辨率
├── 多语言支持：中英文双语能力强
├── 细粒度定位：支持 bounding box 输出
└── 多图理解：支持多张图片输入
训练数据：1.4B 图文对（含大量中文数据）
```

Qwen-VL 的一大特色是支持**区域理解**——用户可以在图中框选区域提问，模型也可以输出 bounding box 定位目标。

### InternVL

```
架构：InternViT-6B + InternLM-20B（可换）
关键创新：
├── 超大视觉编码器：6B 参数的 ViT，远超 CLIP ViT-L 的 300M
├── 动态分辨率：将图像拆为多个 448×448 的子图
├── Progressive Alignment：渐进式对齐策略
└── 多种 LLM 适配：可接入不同的语言模型
```

InternVL 的思路是：**如果视觉编码器更强大，多模态理解就能更好**。通过扩大视觉编码器到 6B 参数并在大规模图文数据上训练，获得了更强的视觉表示。

### 多模态模型的演进

| 模型 | 时间 | 视觉编码器 | LLM | 关键创新 |
|------|------|-----------|-----|---------|
| LLaVA | 2023.4 | CLIP ViT-L | Vicuna-7B | 简洁的视觉-语言连接 |
| LLaVA-1.5 | 2023.10 | CLIP ViT-L@336 | Vicuna-7B/13B | MLP 投影 + 高分辨率 |
| Qwen-VL | 2023.8 | ViT-G/14 | Qwen-7B | 多分辨率 + 多语言 + 定位 |
| InternVL | 2023.12 | InternViT-6B | InternLM | 超大视觉编码器 |
| GPT-4V | 2023.9 | 未公开 | GPT-4 | 商用多模态标杆 |
| GPT-4o | 2024.5 | 原生多模态 | GPT-4 | 原生多模态，端到端训练 |
| Qwen2-VL | 2024.8 | ViT + NaViT | Qwen2 | 动态分辨率，视频理解 |
| InternVL2.5 | 2024.12 | InternViT-6B | InternLM2.5 | 性能逼近 GPT-4o |

## 多模态训练的关键挑战

### 模态对齐（Modality Alignment）

视觉和语言是两种截然不同的信息形态。如何让 LLM 真正"理解"视觉信息，而不只是学到浅层的统计对应关系？

**核心问题**：
- 视觉特征空间和语言特征空间的分布不同
- 投影层的能力瓶颈：简单的线性/MLP 投影是否足够？
- 视觉信息的粒度：全局特征 vs 局部特征 vs 区域特征

**常见解决方案**：
- 分阶段训练：先对齐，再微调
- Q-Former（BLIP-2）：用交叉注意力提取视觉信息
- Perceiver Resampler：将可变长度的视觉特征压缩为固定数量的 token

### 多模态数据构建

高质量的多模态数据是稀缺资源：

| 数据类型 | 规模 | 质量 | 用途 |
|----------|------|------|------|
| 图文对 (LAION) | 数十亿 | 低（网页抓取） | 预训练对齐 |
| 高质量 caption | 数百万 | 中 | 特征对齐 |
| 指令数据 (GPT-4 生成) | 数十万 | 高 | 指令微调 |
| 人工标注 VQA | 数万 | 最高 | 评估/精调 |

**数据构建的实践经验**：
- 预训练阶段用量取胜，微调阶段用质取胜
- GPT-4/4V 辅助生成指令数据是目前最高效的方法
- 混合不同类型的数据（caption、VQA、推理、对话）训练效果最好

### 多模态幻觉（Hallucination）

多模态幻觉是当前最突出的问题：模型生成了图像中不存在的内容。

**幻觉的类型**：
- **物体幻觉**：声称看到了图中不存在的物体（"图中有一只猫"，但实际没有）
- **属性幻觉**：错误描述物体的属性（颜色、大小、位置）
- **关系幻觉**：错误描述物体之间的关系（"猫在桌子上"，实际在地上）

**幻觉的原因**：
- 语言先验过强：LLM 倾向于生成"合理"的描述，即使与图像不符
- 视觉信息损失：投影过程中丢失了细节
- 训练数据偏差：某些物体共现频率高导致的虚假关联

**缓解方法**：
- RLHF/DPO 对齐：用人类偏好惩罚幻觉回答
- 视觉 grounding：让模型输出的内容可以追溯到图像区域
- 对比解码：比较有图和无图时的输出差异

## 原生多模态（Native Multimodal）：从「拼接」到「同源」

LLaVA 风格的「视觉编码器 + 投影层 + LLM」可以看作 **晚融合（late fusion）**：图像与文本各自有独立的编码路径，仅在 LLM 输入处做 token 级拼接。GPT-4o、Gemini-1.5、Qwen2-VL 等新一代模型则推动 **原生多模态（natively multimodal）** 的范式 —— 一个 Transformer 主干同时承载所有模态，输入端只是不同模态各自的 token 化器。

### 晚融合 vs 原生（早融合）

```
晚融合 (LLaVA)                          原生多模态 (GPT-4o / Gemini)
────────────────────────             ──────────────────────────────
图像 → ViT → MLP ┐                    图像 → patch tokenizer ┐
                ├→ LLM (only text)                          ├→ Unified Transformer
文本 → tokenize ─┘                    文本 → BPE tokenizer  ┘   (text + vision + audio)
                                       音频 → mel + tokenizer ┘

特点：                                 特点：
- 视觉编码器冻结，LLM 不需重训          - 全部模态共享主干，端到端训练
- 工程上简单，开源生态成熟              - 模态间能在每一层交互
- 视觉 token 数固定（如 576）           - 支持任意模态混合输入输出
- 跨模态融合发生在「输入接口」          - 跨模态融合贯穿整个网络
```

::: tip 原生多模态的核心承诺
**模态间在每一层都能注意彼此**。晚融合方案下，视觉信息只在 LLM 第一层的拼接处进入；越往后，文本占据的注意力比重越大，视觉信息容易被「稀释」。原生方案让视觉/语音 token 与文本 token 在所有层共享 KV，从根上避免这个稀释问题。
:::

### Qwen2-VL 的「动态分辨率 + M-RoPE」

[Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) 的两个关键设计：

**1. NaViT 风格的动态分辨率（Native Resolution）**

传统 ViT 把输入 resize 到固定尺寸（如 224×224），高分辨率图像被强行下采样会损失细节。NaViT/Qwen2-VL 的做法是：

```
原图 1024×768 → patch_size=14 → 73×54 = 3942 个 patch
原图 336×336  → patch_size=14 → 24×24 = 576 个 patch

→ 每张图的 visual token 数随分辨率自适应，再用 2×2 merger 压缩
→ 长边超过阈值（如 1280）时按比例缩放，避免序列过长
```

**2. M-RoPE（多模态旋转位置编码）**

文本 RoPE 只编码 1 维位置（token index），视觉需要 (h, w) 两维，视频还需要 t 维。M-RoPE 把旋转角度按 (t, h, w) 三个维度切分：

```python
# 伪代码：M-RoPE 的核心是把 head_dim 分成三段，每段编码一个维度
def m_rope(q, k, t_pos, h_pos, w_pos):
    # head_dim = D，按 (D/4, D/4, D/2) 分给 (t, h, w) 或 (D/3, D/3, D/3)
    cos_t, sin_t = rope_angles(t_pos, dim=D // 4)
    cos_h, sin_h = rope_angles(h_pos, dim=D // 4)
    cos_w, sin_w = rope_angles(w_pos, dim=D // 2)
    q = apply_rotary(q, [cos_t, cos_h, cos_w], [sin_t, sin_h, sin_w])
    k = apply_rotary(k, [cos_t, cos_h, cos_w], [sin_t, sin_h, sin_w])
    return q, k
```

文本 token 的 (t, h, w) 都设为同一个全局递增数；视觉 token 各自带空间坐标；视频帧带时间索引。这样一套 RoPE 就能同时刻画一维语言、二维图像、三维视频。

### GPT-4o 的「同源 token」推测

GPT-4o 没有公开权重，但社区基于公开线索的推测：

- **图像端**：用类似 [VQ-VAE](https://arxiv.org/abs/1711.00937) 或 [MAGVIT-v2](https://arxiv.org/abs/2310.05737) 的离散 tokenizer，把 224×224 图像编码为 ~256 个离散 token
- **音频端**：用 [SoundStream](https://arxiv.org/abs/2107.03312) / [Encodec](https://github.com/facebookresearch/encodec) 的残差矢量量化，把 24 kHz 音频编码为 ~75 token/秒
- **统一序列**：所有模态都被映射成同一个词表中的 token，主干就是一个超大的 next-token 预测模型

这种思路在开源社区的对应实现包括 [Chameleon](https://github.com/facebookresearch/chameleon) 和 [AnyGPT](https://github.com/OpenMOSS/AnyGPT)。

### 路线对比一览

| 维度 | 晚融合（LLaVA） | 原生 / 早融合（GPT-4o） |
|------|----------------|-----------------------|
| 视觉表示 | 连续向量（投影后拼接） | 离散 token 或共享嵌入空间 |
| 训练成本 | 低（仅训投影 + 微调 LLM） | 极高（端到端预训练） |
| 模态切换延迟 | 高（不同模型推理） | 低（同一前向） |
| 跨模态推理深度 | 浅层（仅输入处） | 深层（每一层） |
| 多模态生成 | 一般只生成文本 | 可生成图像/音频 token |
| 适合场景 | 学术、工程实践、轻量微调 | 端到端原生多模态助手 |

## 视频理解：从单帧到时序

视频比图像多了 **时间维度**。一个 30 秒、24 fps 的视频就有 720 帧，如果每帧用 LLaVA 的 576 token，总长度 720×576 = 414,720 —— 远远超出现实可用的 context。视频 LLM 的核心问题就是：**怎么把时序信息压缩进可承受的 token 预算**。

### 帧采样策略

```
策略             描述                              适用
─────────────────────────────────────────────────────────
均匀采样          每 N 帧取一帧（如 8/16 帧）         短视频，简单基线
关键帧采样        基于场景变化检测                   长视频，避免冗余
密集 + 稀疏        前几秒密集，后段稀疏               动作/事件识别
查询驱动采样      用 query 引导找最相关帧             长视频问答
Token Compression KV 缓存阶段做 Top-K 视觉 token      Video-LLaMA-2/Long-VA
```

::: tip 帧数预算的工程经验
- LLaVA 风格：每帧 576 token，常用 8-32 帧（即 4.6k-18.4k visual token）
- 长视频（10+ 分钟）：通常做 token 压缩到 ≤4 token/帧（Video-LLaMA-2、LongVA）
- 视频问答：先用 CLIP 算 query-frame 相关度，挑 Top-K 帧再送 LLM
:::

### Video-LLaVA 的「统一编码」思路

[Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) 的核心观察：图像和视频应该共享同一个对齐空间，否则模型在切换模态时会出现「割裂」。它使用 [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind) 做联合编码，把图像和视频都映射到与语言对齐的特征空间，然后接同一套投影层与 LLM。

```
图像 ──┐
       ├→ LanguageBind Encoder ─→ 统一特征 ─→ MLP Projector ─→ LLM
视频 ──┘     (image / video / audio)
```

这种设计让模型在同一次训练中能处理图像问答、视频问答，避免了「先训图像 → 再训视频」常见的灾难性遗忘。

### 时序建模：Token Merging 与 Temporal Attention

简单的帧采样会丢失时序连贯性，常见加强方案：

**1. 时序 Token Merging（如 Video-LLaMA-2）**

```python
def temporal_token_merge(frame_tokens, merge_ratio=0.5):
    # frame_tokens: (T, N, D) 即 T 帧、每帧 N 个 token、D 维
    # 相邻帧间余弦相似度高的 token 对做平均合并
    T, N, D = frame_tokens.shape
    merged = []
    for t in range(T - 1):
        sim = F.cosine_similarity(
            frame_tokens[t].unsqueeze(1),    # (N, 1, D)
            frame_tokens[t + 1].unsqueeze(0), # (1, N, D)
            dim=-1
        )  # (N, N)
        # 选 Top-K 最相似的 token 对做平均
        ...
    return merged
```

**2. Temporal Attention（双流架构，如早期 VideoBERT）**

把空间注意力（每帧内部）和时间注意力（跨帧同一位置）解耦，分别计算，类似 [TimeSformer](https://arxiv.org/abs/2102.05095) 的 divided space-time attention。

### 长视频的「分层摘要」思路

对于 1+ 小时的长视频（如电影、监控、教学视频），单次喂入显然不可能。常用 **Hierarchical Caption + Retrieval**：

```
长视频 (60 min)
    ↓ 切片为 30s 片段
[片段 1] [片段 2] ... [片段 N]   ← 每个片段用 Video-LLM 生成 caption
    ↓
片段 captions（文本）+ 片段 visual embedding（向量）
    ↓ 建立时序索引
        ↓
用户问题 → 检索最相关 K 个片段 → 把片段 visual + caption 送给最终 LLM 回答
```

这个结构跟 RAG 几乎同构，只是「文档块」换成了「视频片段」。

## 音频与语音：从 Whisper 到 Qwen-Audio

音频在多模态体系中长期被低估，但它是 **Agent 与人类交互的核心通道**（语音助手、会议纪要、播客理解）。处理音频的两个关键决策：

1. **离散 token 还是连续嵌入？** 离散适合统一 next-token 预测（GPT-4o 推测路线），连续适合接 LLM 做理解任务（Qwen-Audio 路线）
2. **是否端到端做 ASR？** 还是先用 Whisper 转写，再交给纯文本 LLM？

### Qwen-Audio 的「Whisper + LLM」结构

[Qwen-Audio](https://github.com/QwenLM/Qwen-Audio) 借用了 [Whisper](https://github.com/openai/whisper) 的音频编码器作为「视觉编码器的等价物」，然后接 Qwen LLM：

```
音频波形 (16 kHz)
    ↓
Mel Spectrogram (80 mel bins, 100Hz)         ← 标准 Whisper 预处理
    ↓
[Whisper Encoder]                              ← 冻结，提取音频特征
    ↓
音频特征 (T, 1280)                             ← T = 帧数
    ↓
[Adapter / 下采样 + Linear]                    ← 类似 LLaVA 的投影层
    ↓
音频 Token (T', 4096)                          ← 与 LLM hidden 对齐
    ↓
[Qwen LLM] ← 拼接「文本指令 + 音频 token」
    ↓
文本回答（ASR / 翻译 / 音频问答 / 情绪识别）
```

Qwen-Audio 的多任务设计：训练时用 **task tag** 显式告知模型当前任务（如 `<|asr|>`、`<|translation|>`、`<|sound|>`），共用同一套权重处理 30+ 种音频任务。

### 端到端语音对话：GPT-4o 路线

GPT-4o 之所以能做到 **<300ms 端到端语音对话**，关键是绕过了「ASR → LLM → TTS」三段式管线：

```
传统三段式：               原生语音：
─────────────────         ─────────────────
音频 → ASR (Whisper)       音频 → audio tokenizer
   → 文本                       → audio token
   → LLM 推理                   ↓
   → 文本                  Unified Transformer
   → TTS                        (text + audio token 共享词表)
   → 音频                       ↓
                          → audio token
延迟：~2 秒                     → audio detokenizer (Encodec)
                          延迟：~300ms
```

开源对应实现：[Mini-Omni](https://github.com/gpt-omni/mini-omni)、[LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni)、[Moshi](https://github.com/kyutai-labs/moshi) 等。

## Vision Agent：让 LLM 看屏幕、点按钮

GUI Agent 是 2024-2026 最热门的 Agent 方向 —— 它需要 **视觉感知 + 工具调用 + 长程规划** 三种能力的结合。其核心难题是把「屏幕截图」翻译成模型能消化的结构化表示。

### OmniParser 的屏幕解析管线

[OmniParser](https://github.com/microsoft/OmniParser) 的设计是 **不依赖 OS-level 可访问性 API**，纯靠视觉解析任意屏幕：

```
屏幕截图 (任意分辨率)
    ↓
[1. 可交互区域检测]           ← YOLOv8 微调，识别按钮/输入框/链接的 bbox
    ↓
[2. 图标语义标签]             ← BLIP-2 / Florence 给每个 bbox 生成功能描述
    ↓
[3. OCR 文本提取]             ← PaddleOCR 提取屏幕上的所有文字
    ↓
结构化 DOM-like 表示：
  [{"id": 1, "type": "button", "bbox": [x, y, w, h],
    "label": "Send Email", "ocr_text": "Send"},
   {"id": 2, "type": "input", "bbox": [...],
    "label": "Search box", "ocr_text": ""}]
    ↓
LLM 接收这个结构化列表 + 用户指令，输出 "click on element id=1"
```

::: tip 为什么不直接给 LLM 看截图？
- **分辨率诅咒**：1080p 截图按 14×14 patch 切分会产生 ~6k visual token，且小图标在低分辨率下不可识别
- **Grounding 误差**：LLM 直接输出像素坐标的精度不足，容易点偏
- **结构化更稳健**：先做 detection + caption，再让 LLM 在已识别元素中挑选 id —— 这是工程上更可靠的范式
:::

### Vision Agent 的 Reward 设计

训练 Vision Agent 时，奖励函数比纯文本 Agent 更复杂：

| 信号 | 含义 | 计算 |
|------|------|------|
| 任务完成 | 最终页面状态匹配目标 | 终态检测器（文本/视觉匹配） |
| 步骤正确 | 每步点击/输入是否合理 | 子目标判别器（基于轨迹回放） |
| 屏幕变化 | 操作后屏幕确实改变 | 截图前后 SSIM / 像素差异 |
| 元素命中 | 点击坐标落在目标元素内 | bbox 内点检验 |
| 副作用惩罚 | 触发了不该触发的操作 | 黑名单元素列表 |

参考 [SeeClick](https://github.com/njucckevin/SeeClick)、[OS-Atlas](https://github.com/OS-Copilot/OS-Atlas) 等开源 GUI 数据集。

## 多模态对齐：RM、DPO、Hallucination 缓解

多模态模型的对齐与纯文本对齐的本质区别在于：**奖励信号需要把「视觉证据」纳入考量**。仅基于文本的 RM 容易奖励「读起来流畅但脱离图像」的回答 —— 这正是多模态幻觉的训练源头。

### 多模态 RM 的两条路线

**路线 1：视觉条件 RM（Visual-Conditioned RM）**

直接训练一个多模态 RM：输入是 (图像, 文本回答)，输出是标量分数。

```python
class VisualRM(nn.Module):
    def __init__(self, base_mllm):
        super().__init__()
        self.mllm = base_mllm                    # LLaVA-style backbone
        self.value_head = nn.Linear(self.mllm.hidden_size, 1)

    def forward(self, image, response_ids, attention_mask):
        # 把图像 token 与回答 token 拼接，取最后一个 hidden state
        hidden = self.mllm.forward_with_image(image, response_ids, attention_mask)
        last_hidden = hidden[:, -1, :]            # (B, H)
        reward = self.value_head(last_hidden)     # (B, 1)
        return reward.squeeze(-1)
```

**路线 2：拆分式 RM（Decomposed RM）**

用多个专门的判别器：
- **Faithfulness RM**：判断回答是否忠于图像（是否有幻觉）
- **Helpfulness RM**：判断回答是否有用、流畅
- 最终奖励 = α · faithfulness + β · helpfulness

参考开源实现：[RLHF-V](https://github.com/RLHF-V/RLHF-V)、[Silkie](https://github.com/vlf-silkie/VLFeedback)、[POVID](https://github.com/YiyangZhou/POVID)。

### 多模态 DPO

DPO 损失形式与文本完全相同，关键在于 **偏好数据如何构造**：

```python
def multimodal_dpo_loss(policy, ref, batch, beta=0.1):
    # batch: image, prompt, chosen, rejected
    img = batch["image"]

    # chosen / rejected 都是同一张图配的两个回答
    pi_chosen = policy.logp_with_image(img, batch["prompt"], batch["chosen"])
    pi_reject = policy.logp_with_image(img, batch["prompt"], batch["rejected"])
    ref_chosen = ref.logp_with_image(img, batch["prompt"], batch["chosen"])
    ref_reject = ref.logp_with_image(img, batch["prompt"], batch["rejected"])

    log_ratio_chosen = pi_chosen - ref_chosen
    log_ratio_reject = pi_reject - ref_reject
    margin = beta * (log_ratio_chosen - log_ratio_reject)

    return -F.logsigmoid(margin).mean()
```

::: tip 多模态 DPO 的偏好数据构造
- **幻觉对比对**：chosen = 严格基于图像的回答；rejected = 同一张图但加入虚构物体的回答（用 GPT-4V 改写得到）
- **细节对比对**：chosen = 包含图中具体细节；rejected = 笼统但正确
- **图像扰动对**：同一回答 + 不同图像 → 不匹配的图像应得低分（隐式 grounding）

参考数据集：[RLHF-V Dataset](https://huggingface.co/datasets/openbmb/RLHF-V-Dataset)、[VLFeedback](https://huggingface.co/datasets/MMInstruction/VLFeedback)。
:::

### 缓解多模态幻觉的工程手段

除了 RLHF/DPO，还有几种推理时与训练时的低成本手段：

**1. Visual Contrastive Decoding (VCD)**

```python
# 同时对原图和「加噪图」做前向，相减得到「真正依赖视觉的 logits」
def vcd_decode(model, image, prompt, alpha=1.0):
    noisy_image = add_diffusion_noise(image, sigma=0.5)
    logits_clean = model(image=image, prompt=prompt).logits[:, -1]
    logits_noisy = model(image=noisy_image, prompt=prompt).logits[:, -1]
    # contrastive logits：放大「依赖原图才会高」的部分
    logits_final = (1 + alpha) * logits_clean - alpha * logits_noisy
    return logits_final.softmax(-1).argmax(-1)
```

直觉：如果一个 token 在原图和噪图下的概率差不多，那它就是来自语言先验而非图像 —— 应该被压制。

**2. POVID（Image-Conditioned Preference Optimization）**

把 chosen 设为「原图配的回答」，rejected 设为「噪声图/无图配的同一回答」，强迫模型学到「输出必须依赖于真实图像」。这是图像层面的隐式 grounding 约束。

**3. Object Halluciation Benchmark：CHAIR / POPE**

评估幻觉的标准指标：
- **CHAIR**：caption 中提到的物体里有多少不在 ground-truth 列表中
- **POPE**：用 yes/no 问题（"图中有 cat 吗？"）测试存在性回答的准确率

任何对齐手段都应该能稳定降低这两个指标，否则可能只是「让回答听起来更顺」而没真的减少幻觉。

## 苏格拉底时刻

请停下来思考以下问题，不急于查看答案：

1. ViT 将图像切成 16x16 的 patch — 如果一个关键物体恰好被切分到了两个 patch 中，模型能正确识别它吗？注意力机制如何弥补这个问题？
2. CLIP 通过对比学习对齐图文空间，但它只学习了全局的图文匹配关系。如果你需要模型理解"图片左边的红色球"这样的细粒度空间关系，CLIP 能做到吗？为什么？
3. LLaVA 用一个简单的线性投影层连接视觉编码器和 LLM — 这似乎太简单了。为什么这能工作？如果换成一个更复杂的跨模态注意力模块，效果一定会更好吗？
4. 当前的多模态模型大多是"看图说话"模式 — 图像理解是单向的。如果你想让模型能够"指着图中某个区域提问"（视觉定位），架构需要如何改变？
5. CLIP 的 InfoNCE 损失中，batch size 越大效果越好（OpenAI 用了 32768）。为什么 batch size 对对比学习如此重要？batch 太小会有什么问题？
6. 多模态幻觉和纯文本 LLM 的幻觉有什么本质区别？为什么多模态幻觉更难解决？
7. 原生多模态相比晚融合方案的训练成本要高几个数量级，但 LLaVA 至今仍是开源界的主流。如果你给一个创业团队做技术选型，2026 年还有哪些场景是「晚融合 + 大量中文图文指令数据」就够用、不必上原生方案的？
8. Qwen2-VL 的 M-RoPE 把旋转编码按 (t, h, w) 切分。如果一个用户输入是「先看一张图、再听一段语音、再看一张图」，按你的理解应该如何分配 (t, h, w) 三个维度的位置编码？纯文本 token 又该怎么编码？
9. Vision Agent 把屏幕截图先做 OCR + bbox detection 再交给 LLM，看似比「直接给 LLM 看截图」更繁琐 —— 但工程上更稳健。在什么样的场景下，反而应该让模型直接读截图（比如游戏、绘图软件）？为什么？
10. 多模态 DPO 的「rejected = 加入虚构物体的回答」是如何造出来的？如果让你设计一条用于 grounding 的偏好数据 pipeline，你会用什么模型来「造负例」、又如何防止造出来的负例本身有偏？

## 推荐资源

- **Dosovitskiy et al. "An Image is Worth 16x16 Words"** — ViT 原始论文
- **Radford et al. "Learning Transferable Visual Models From Natural Language Supervision"** — CLIP 论文
- **Liu et al. "Visual Instruction Tuning"** — LLaVA 论文
- **Liu et al. "Improved Baselines with Visual Instruction Tuning"** — LLaVA-1.5 论文
- **Bai et al. "Qwen-VL: A Versatile Vision-Language Model"** — Qwen-VL 论文
- **Chen et al. "InternVL: Scaling up Vision Foundation Models"** — InternVL 论文
- **Li et al. "BLIP-2: Bootstrapping Language-Image Pre-training"** — Q-Former 架构
- [OpenAI CLIP GitHub](https://github.com/openai/CLIP) — 官方实现与预训练权重
- [LLaVA 项目主页](https://llava-vl.github.io/) — Demo 与代码

### 原生多模态 / 视频 / 音频

- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) — 动态分辨率 + M-RoPE 的开源代表
- [InternVL](https://github.com/OpenGVLab/InternVL) — 6B 视觉编码器与渐进对齐策略
- [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) — 端侧多模态，强调高分辨率与小模型
- [Chameleon](https://github.com/facebookresearch/chameleon) — 早融合、统一离散 token 的开源方案
- [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) — 图像 / 视频统一对齐空间
- [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA) — 长视频 token 压缩策略
- [Qwen-Audio](https://github.com/QwenLM/Qwen-Audio) — Whisper-style 音频编码 + LLM
- [Mini-Omni](https://github.com/gpt-omni/mini-omni) / [LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni) — 端到端语音对话开源实现
- [Encodec](https://github.com/facebookresearch/encodec) — 神经音频编解码器，原生多模态的音频 tokenizer

### Vision Agent

- [OmniParser](https://github.com/microsoft/OmniParser) — 屏幕解析与 GUI Agent 基础设施
- [SeeClick](https://github.com/njucckevin/SeeClick) — GUI grounding 数据集与模型
- [OS-Atlas](https://github.com/OS-Copilot/OS-Atlas) — 跨平台 GUI Agent 基础模型

### 多模态对齐 / RM / DPO

- [RLHF-V](https://github.com/RLHF-V/RLHF-V) — 多模态 RLHF 与幻觉缓解
- [VLFeedback / Silkie](https://github.com/vlf-silkie/VLFeedback) — 多模态偏好数据集
- [POVID](https://github.com/YiyangZhou/POVID) — 图像条件偏好优化（image-DPO）
- [POPE](https://github.com/RUCAIBox/POPE) / [CHAIR](https://github.com/LisaAnne/Hallucination) — 多模态幻觉评估基准
