---
title: 多模态代码填空
description: Level 2-3 填空：Patch Embed、CLIP InfoNCE、LLaVA Projector、视频采样、Vision 工具路由
topics: [fill-in, multimodal, ViT, CLIP, LLaVA, vision-projector, video-sampling, vision-agent]
prereqs: [/applications/multimodal]
---

# 多模态代码填空

> 主线参考：[多模态大模型](/applications/multimodal)。本章把多模态系统中最容易"看懂但写不出"的几个零件挖出来填——从最底层的 patch embedding、CLIP 训练目标，到 LLaVA 的视觉投影器，再到视频采样与 Vision Agent 的工具路由。

::: tip 推荐做题顺序
1. 练习 1（Patch Embed）→ 练习 2（CLIP InfoNCE）：从图像 token 化到对比学习的两个经典基石
2. 练习 3（LLaVA Projector）：理解视觉特征如何"翻译"成 LLM 能吃的 token
3. 练习 4（视频采样）→ 练习 5（Vision Agent 路由）：往应用层走，工程感更强

做完建议跑一遍 [多模态主线](/applications/multimodal) 的 ViT / CLIP / LLaVA 三段。
:::

参考开源项目（外部链接）：

- [openai/CLIP](https://github.com/openai/CLIP) — 原版 CLIP 实现，对比 InfoNCE
- [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) — Visual Instruction Tuning 的 reference
- [PKU-YuanGroup/Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) — 视频多模态范式
- [microsoft/OmniParser](https://github.com/microsoft/OmniParser) — Screen Parser，给 Vision Agent 喂结构化输入
- [QwenLM/Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) — 动态分辨率的开源 VL 模型

---

## 练习 1：Patch Embedding（Level 2）

ViT 把图像切成 patch、再线性投影成 token——这是所有视觉 Transformer（ViT / CLIP / DINO / SAM 的 image encoder）的入口。一个反直觉的事实：**这一步并不需要写双重 for 循环切 patch**，一个 `Conv2d(kernel_size=patch_size, stride=patch_size)` 就把"切 patch + 线性投影"两步合成了一个算子，效率最高。

输入图像 224×224、patch_size=16、embed_dim=768，应当产生 14×14=196 个 token，每个 token 维度 768。

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """ViT 风格的 patch embedding：把 [B, C, H, W] 切 patch + 线性投影成 [B, N, D]。"""

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # TODO 1: 计算 patch 数 N = (H/P) * (W/P)
        self.num_patches = _____

        # TODO 2: 用一个 Conv2d 同时完成"切 patch"与"线性投影"
        #         kernel_size = stride = patch_size 是关键：
        #         每个 patch 内部的像素被一个 (patch_size, patch_size) 的卷积核线性组合，
        #         相邻 patch 之间没有重叠（stride = kernel）。
        self.proj = _____

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        返回: [B, N, D]，N=num_patches，D=embed_dim
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"input size mismatch: got {H}x{W}, expected {self.img_size}"

        # TODO 3: Conv2d 输出形如 [B, D, H/P, W/P]
        #         flatten 后两个维度，再交换成 [B, N, D]
        x = self.proj(x)               # [B, D, H/P, W/P]
        x = x.flatten(2)               # [B, D, N]
        x = _____                      # [B, N, D]
        return x


# ====== 测试 ======
patch_embed = PatchEmbed(img_size=224, patch_size=16, embed_dim=768)
x = torch.randn(2, 3, 224, 224)
y = patch_embed(x)
print(y.shape)             # 期望: torch.Size([2, 196, 768])
print(patch_embed.num_patches)  # 期望: 196

# 参数量验算：Conv2d(3, 768, k=16, s=16) 的权重 = 3*768*16*16 + 768 = 590,592
n_params = sum(p.numel() for p in patch_embed.parameters())
print("params:", n_params)  # 期望: 590592
```

::: details 提示
- TODO 1：`(img_size // patch_size) ** 2`
- TODO 2：`nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)`
- TODO 3：`x.transpose(1, 2)` 把 [B, D, N] 变成 [B, N, D]
:::

<details>
<summary>点击查看答案</summary>

```python
# TODO 1
self.num_patches = (img_size // patch_size) ** 2

# TODO 2
self.proj = nn.Conv2d(in_channels, embed_dim,
                      kernel_size=patch_size, stride=patch_size)

# TODO 3
x = x.transpose(1, 2)   # [B, D, N] -> [B, N, D]
```

**解析：**

为什么不写"先切 patch 再 Linear"？两条等价路径：

```python
# 路径 A：朴素切 patch + Linear
patches = x.unfold(2, P, P).unfold(3, P, P)        # [B, C, H/P, W/P, P, P]
patches = patches.permute(0, 2, 3, 1, 4, 5)         # [B, H/P, W/P, C, P, P]
patches = patches.reshape(B, N, C*P*P)              # [B, N, C*P*P]
tokens = nn.Linear(C*P*P, D)(patches)               # [B, N, D]

# 路径 B：单个 Conv2d
tokens = nn.Conv2d(C, D, P, stride=P)(x).flatten(2).transpose(1, 2)
```

两者**数学上等价**：Conv2d 的每个输出位置就是把对应 patch 的 `C*P*P` 个像素做线性组合。但 Conv2d 内部做了 `im2col + GEMM` 的高度优化，比手写 unfold + Linear 快很多。

工程小细节：
1. **PE（Position Embedding）独立加在外面**：PatchEmbed 只负责 token 化，位置编码由后面的 ViT block 在第一层加上。
2. **CLS token 的位置**：通常 ViT 会在 PatchEmbed 之后 prepend 一个可学习的 [CLS] token，使序列变成 `1 + N` 个 token。这一步不放在 PatchEmbed 里。
3. **动态分辨率的扩展**：[Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) 把 patch 数变成动态的，PE 改用 RoPE-2D。但底层的 Conv2d patch_size=14 思路不变。

</details>

---

## 练习 2：CLIP InfoNCE Loss（Level 2）

CLIP 训练的核心是 **N×N 的对比矩阵**：一个 batch 里有 N 张图、N 段文本（一一配对），用它们的 embedding 算 cosine similarity 得到一个 N×N 矩阵，**对角线是正样本（图 i 配文 i），非对角线全是负样本**。然后在两个方向上各算一次 cross-entropy（image-to-text、text-to-image），取平均。

三个关键点：
- L2 normalize：让点积变成 cosine
- learnable temperature：`logit_scale = exp(τ)`，τ 初始化为 `ln(1/0.07)`，clip 上限避免数值爆炸
- 双向损失：避免单方向坍塌

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    """CLIP 的对称 InfoNCE 损失。"""

    def __init__(self, init_temperature: float = 0.07):
        super().__init__()
        # TODO 1: 把 temperature 做成 log 空间的可学习参数
        #         为什么用 log？避免 τ 直接被优化成负数（temperature 必须 > 0）
        #         初始值: ln(1/0.07)
        import math
        self.logit_scale = nn.Parameter(_____)

    def forward(self, image_features: torch.Tensor,
                text_features: torch.Tensor) -> dict:
        """
        image_features: [N, D]
        text_features:  [N, D]
        返回 dict：包含 loss / loss_i2t / loss_t2i / accuracy
        """
        # TODO 2: 在最后一维做 L2 normalize
        image_features = _____
        text_features = _____

        # 把 logit_scale 转回线性 + clip 上限避免数值不稳定
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # TODO 3: 算 logits 矩阵 [N, N]
        #         logits_i2t[i, j] = image_i 与 text_j 的相似度（乘以 scale）
        logits_i2t = _____
        logits_t2i = logits_i2t.t()

        # TODO 4: ground-truth label：对角线即正样本对
        N = image_features.size(0)
        labels = _____   # [0, 1, 2, ..., N-1]

        # TODO 5: 双向 cross-entropy + 平均
        loss_i2t = F.cross_entropy(logits_i2t, labels)
        loss_t2i = F.cross_entropy(logits_t2i, labels)
        loss = _____

        # 评估：top-1 accuracy（图找对文本）
        with torch.no_grad():
            acc = (logits_i2t.argmax(dim=-1) == labels).float().mean()

        return {"loss": loss, "loss_i2t": loss_i2t,
                "loss_t2i": loss_t2i, "accuracy": acc}


# ====== 测试 ======
torch.manual_seed(42)
loss_fn = CLIPLoss()

# 8 张图、8 段文本，假设 image 与 text encoder 都是 512 维
N, D = 8, 512
img = torch.randn(N, D, requires_grad=True)
txt = torch.randn(N, D, requires_grad=True)

out = loss_fn(img, txt)
print(f"loss     = {out['loss'].item():.4f}")     # 随机初始化下 ~ ln(8) = 2.08
print(f"loss_i2t = {out['loss_i2t'].item():.4f}")
print(f"loss_t2i = {out['loss_t2i'].item():.4f}")
print(f"acc      = {out['accuracy'].item():.4f}")  # 随机情况下 ~ 1/N = 0.125

out["loss"].backward()
print("logit_scale.grad =", loss_fn.logit_scale.grad.item())  # 应该非零
```

::: details 提示
- TODO 1：`torch.tensor(math.log(1.0 / init_temperature))` 或 `torch.ones([]) * math.log(1/0.07)`
- TODO 2：`F.normalize(image_features, dim=-1)`
- TODO 3：`logit_scale * image_features @ text_features.t()`
- TODO 4：`torch.arange(N, device=image_features.device)`
- TODO 5：`(loss_i2t + loss_t2i) / 2`
:::

<details>
<summary>点击查看答案</summary>

```python
# TODO 1
import math
self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))

# TODO 2
image_features = F.normalize(image_features, dim=-1)
text_features = F.normalize(text_features, dim=-1)

# TODO 3
logits_i2t = logit_scale * image_features @ text_features.t()  # [N, N]

# TODO 4
labels = torch.arange(N, device=image_features.device)

# TODO 5
loss = (loss_i2t + loss_t2i) / 2
```

**解析：**

CLIP 的 loss 看似简单，但每个细节都有"为什么这么写"：

1. **L2 normalize 把点积变 cosine**：归一化后 `x · y = cos(θ)`，范围 [-1, 1]。如果不归一化，向量 norm 会被吞进 logits，相似度大小被向量长度污染。
2. **可学习 temperature 的 log 参数化**：直接把 τ 设成参数，优化中可能变成负数或零；用 `log_τ` 参数化保证 τ > 0 永远成立，且梯度更平缓。初始值 `ln(1/0.07) ≈ 2.66`。
3. **clamp(max=100) 是保命措施**：训练后期 logit_scale 会持续增大（让正样本与负样本拉开），但太大会导致 softmax 数值溢出。CLIP 论文里把它 clip 到 100。
4. **labels = arange(N)** 的妙处：因为我们**人为安排了 batch 里第 i 张图配第 i 段文**，所以 ground-truth label 永远是 [0, 1, 2, ..., N-1]——不需要额外标注。这就是 CLIP "免费的对齐信号"的来源。
5. **双向 loss 的必要性**：只算 i2t 会让 image encoder 学得很积极，text encoder 偷懒；只算 t2i 反过来。对称 loss 强制两边都学会区分。

工程上，[openai/CLIP](https://github.com/openai/CLIP) 的实际实现还有：
- **大 batch 是关键**：负样本数量 = batch_size - 1。OpenAI 用 batch=32K，OpenCLIP 把 batch 切到多卡再 all-gather embedding 凑大 batch。
- **混合精度**：fp16 要小心 logit_scale.exp() 溢出，logit_scale 通常保持 fp32。
- **gather 时的梯度**：`all_gather` 默认不传梯度，需要 `torch.distributed.nn.all_gather` 或自定义 autograd。

</details>

---

## 练习 3：LLaVA Visual Projector 工厂（Level 2）

LLaVA 的核心 trick：**冻结 vision encoder（CLIP-ViT）+ 冻结 LLM，只训中间一个小的 projector**，把 vision feature 投到 LLM 的 hidden_size。这个 projector 只占总参数的 < 1%，但决定了视觉信号能不能被 LLM 吃下去。

主流配置三选一：
- `linear`：单个 `nn.Linear(mm_hidden_size, hidden_size)`。LLaVA-v1 默认。
- `mlp2x_gelu`：`Linear → GELU → Linear`。LLaVA-1.5 升级版，多一层非线性。
- `mlp3x_gelu`：`Linear → GELU → Linear → GELU → Linear`。LLaVA-1.6/Next 用过。

要求：写一个**字符串驱动的工厂函数**，输入 `proj_type: str`，输出对应 `nn.Module`。`mlpNx_gelu` 的 N 要从字符串里抠出来，支持任意正整数。

```python
import re
import torch
import torch.nn as nn

def build_visual_projector(proj_type: str,
                           vision_dim: int,
                           llm_dim: int) -> nn.Module:
    """
    根据字符串构建 visual projector。

    proj_type: 
      - "linear":     [vision_dim] → [llm_dim]
      - "identity":   恒等映射（要求 vision_dim == llm_dim）
      - "mlpNx_gelu": N 个 Linear，中间夹 GELU
                      第一层 vision_dim → llm_dim，之后 llm_dim → llm_dim
    vision_dim: vision encoder 输出维度（如 CLIP ViT-L/14 是 1024）
    llm_dim:    LLM 的 hidden_size（如 LLaMA-7B 是 4096）

    返回一个 nn.Module。
    """
    # ---------- 分支 1: linear ----------
    if proj_type == "linear":
        # TODO 1: 单个 Linear
        return _____

    # ---------- 分支 2: identity ----------
    if proj_type == "identity":
        if vision_dim != llm_dim:
            raise ValueError(
                f"identity projector requires vision_dim == llm_dim, "
                f"got {vision_dim} vs {llm_dim}"
            )
        return nn.Identity()

    # ---------- 分支 3: mlpNx_gelu ----------
    # TODO 2: 用正则抽取 N，例如 "mlp2x_gelu" → 2，"mlp3x_gelu" → 3
    #         匹配模式: ^mlp(\d+)x_gelu$
    match = _____
    if match:
        depth = int(match.group(1))
        if depth < 1:
            raise ValueError(f"mlp depth must be >= 1, got {depth}")

        # TODO 3: 拼一个 [Linear, GELU, Linear, GELU, ..., Linear] 序列
        #         第 1 层: vision_dim → llm_dim
        #         之后:   llm_dim → llm_dim，前面加 GELU
        layers = [nn.Linear(vision_dim, llm_dim)]
        for _ in range(1, depth):
            layers.append(nn.GELU())
            layers.append(_____)
        return nn.Sequential(*layers)

    # ---------- 兜底 ----------
    raise ValueError(f"Unknown projector type: {proj_type!r}")


# ====== 测试 ======
# 模拟 CLIP ViT-L/14 → LLaMA-7B
V, L = 1024, 4096
batch_n_tokens = (2, 256)  # 2 个样本，每个 256 个 vision token

x = torch.randn(*batch_n_tokens, V)

for proj_type in ["linear", "mlp2x_gelu", "mlp3x_gelu"]:
    proj = build_visual_projector(proj_type, V, L)
    y = proj(x)
    n_params = sum(p.numel() for p in proj.parameters())
    print(f"{proj_type:12s} | out={tuple(y.shape)} | params={n_params:,}")

# 期望:
# linear       | out=(2, 256, 4096) | params=4,198,400
# mlp2x_gelu   | out=(2, 256, 4096) | params=21,000,192
# mlp3x_gelu   | out=(2, 256, 4096) | params=37,801,984

# 边界测试
identity_proj = build_visual_projector("identity", 4096, 4096)
print("identity:", identity_proj(torch.randn(1, 4, 4096)).shape)

try:
    build_visual_projector("identity", 1024, 4096)
except ValueError as e:
    print("expected error:", e)

try:
    build_visual_projector("mlp0x_gelu", V, L)
except ValueError as e:
    print("expected error:", e)
```

::: details 提示
- TODO 1：`nn.Linear(vision_dim, llm_dim)`
- TODO 2：`re.match(r"^mlp(\d+)x_gelu$", proj_type)`
- TODO 3：`nn.Linear(llm_dim, llm_dim)`
- 注意 `mlp1x_gelu` 应当退化为单 Linear（depth=1，循环不执行）
:::

<details>
<summary>点击查看答案</summary>

```python
# TODO 1
return nn.Linear(vision_dim, llm_dim)

# TODO 2
match = re.match(r"^mlp(\d+)x_gelu$", proj_type)

# TODO 3
layers.append(nn.Linear(llm_dim, llm_dim))
```

**解析：**

为什么 LLaVA 的 projector 长这样而不是别的？这里有几个被论文低估的设计选择：

1. **第一层就升维**（vision_dim → llm_dim），后续保持在 llm_dim 上做非线性。如果反过来"先在低维做几层 MLP 再升维"，参数更省但表达力受限——视觉信号需要尽快进入 LLM 的语义空间。
2. **GELU 而非 ReLU**：LLaMA 系列用 SwiGLU，但 projector 沿用 ViT 的 GELU 习惯，简化设计。GELU 在小模型 / 小数据下表现稳定。
3. **没有 LayerNorm / Dropout**：projector 训练数据极少（通常 558K image-text pair 做 pretrain），加 LN 反而损害收敛。
4. **字符串驱动的工厂模式**：训练脚本通过 config 字段（如 `mm_projector_type: "mlp2x_gelu"`）切换结构，不改代码。这种"depth as string"的设计在 LLaVA 系列后续被广泛抄袭——MoE-LLaVA、ShareGPT4V 等都沿用。
5. **identity projector 的存在**：当 vision encoder 已经被 align 到 LLM 维度（如 BLIP-2 的 Q-Former 输出 128*768），可以直接用 identity 跳过投影。

参考 [LLaVA 的 builder.py](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py)（开源链接）——但**这道题的答案是独立写的**，不要直接照搬。生产代码还有几个增强：
- 加 `IdentityMap` 类带 `.config` 属性，方便 from_pretrained 时恢复
- 加 `SimpleResBlock`：`x + MLP(LN(x))`，残差更稳但 LLaVA 默认没用
- 支持 `delay_load`：DeepSpeed ZeRO-3 下按需加载

</details>

---

## 练习 4：Video Frame Sampler（Level 3）

视频比图像贵 30~120 倍——不可能把每一帧都喂给 vision encoder。Video-LLaVA / VideoChat / LLaVA-Video 等都依赖一个**采样器**：从 T 帧视频里挑 K 帧（K << T）送进模型。三种主流采样策略：

1. **均匀采样（Uniform）**：每隔 T/K 取一帧。最简单、最常用、最容易漏关键帧。
2. **关键帧采样（Keyframe）**：基于相邻帧差选变化大的帧——动作 / 场景切换处。
3. **Query-driven**：给定文本 query，用 CLIP 算每帧与 query 的相似度，取 Top-K。检索类任务最强。

```python
import torch
import torch.nn.functional as F

def sample_uniform(num_frames: int, k: int) -> list[int]:
    """
    均匀采样：从 [0, num_frames) 里取 k 个等距下标。
    要求 k >= 1。如果 k >= num_frames，返回所有帧。
    """
    if k >= num_frames:
        return list(range(num_frames))
    # TODO 1: 用 linspace 在 [0, num_frames-1] 上取 k 个浮点点，再四舍五入到 int
    #         注意去重（极端情况下相邻 index 可能撞）
    idxs = _____
    idxs = sorted(set(idxs))
    return idxs


def sample_keyframe(frames: torch.Tensor, k: int) -> list[int]:
    """
    关键帧采样：基于"相邻帧像素差"挑变化大的帧。
    
    frames: [T, C, H, W]，已归一化到 [0, 1]
    返回 k 个 frame index（升序）
    """
    T = frames.size(0)
    if k >= T:
        return list(range(T))

    # TODO 2: 算相邻帧的 L1 距离: diffs[t] = mean |frames[t+1] - frames[t]|
    #         得到 [T-1] 长度的差分序列
    diffs = _____

    # 第 0 帧总是保留（视频开头），剩下从 diffs 里挑 top-(k-1) 大的
    # diffs[t] 大表示 frames[t+1] 是关键帧（变化点）
    # TODO 3: 用 topk 选 k-1 个最大的 diff 下标，对应的关键帧是 idx+1
    top_diff = _____
    keyframe_idx = (top_diff + 1).tolist()

    return sorted(set([0] + keyframe_idx))[:k]


def sample_query_driven(frame_features: torch.Tensor,
                        query_feature: torch.Tensor,
                        k: int) -> list[int]:
    """
    Query 驱动采样：用预计算好的 CLIP feature 算相似度，取 Top-K。
    
    frame_features: [T, D]，每帧的 CLIP image embedding（已 L2 normalized）
    query_feature:  [D]，query 文本的 CLIP text embedding（已 L2 normalized）
    返回 k 个 frame index（按相似度从高到低）
    """
    T = frame_features.size(0)
    if k >= T:
        return list(range(T))

    # TODO 4: cosine similarity（已经 normalize 过，直接点积）
    sims = _____   # [T]

    # TODO 5: topk 取 k 个相似度最高的帧
    top = _____
    return top.indices.tolist()


# ====== 测试 ======
torch.manual_seed(0)

# 模拟一段 30 帧视频
T, C, H, W = 30, 3, 64, 64
frames = torch.rand(T, C, H, W)

# 故意在第 10 / 20 帧造一个剧烈变化（场景切换）
frames[10:] += 0.5
frames[20:] -= 0.8
frames = frames.clamp(0, 1)

print("uniform   :", sample_uniform(T, 5))
# 期望: [0, 7, 14, 21, 29] 这种等距分布

print("keyframe  :", sample_keyframe(frames, 5))
# 期望: 包含 0, 10, 20 附近的下标（因为我们故意造了变化）

# Query-driven：模拟 CLIP feature
D = 512
frame_feats = F.normalize(torch.randn(T, D), dim=-1)
query = F.normalize(torch.randn(D), dim=-1)

# 故意把第 7 / 15 / 25 帧造成与 query 相似度最高
for i in [7, 15, 25]:
    frame_feats[i] = F.normalize(query + 0.1 * torch.randn(D), dim=-1)

print("query-top3:", sample_query_driven(frame_feats, query, 3))
# 期望: [7, 15, 25]（顺序可能不同，但应该是这三个）
```

::: details 提示
- TODO 1：`torch.linspace(0, num_frames - 1, k).round().long().tolist()`
- TODO 2：`(frames[1:] - frames[:-1]).abs().mean(dim=(1, 2, 3))` 在 C/H/W 上求平均
- TODO 3：`diffs.topk(k - 1).indices`
- TODO 4：`frame_features @ query_feature`
- TODO 5：`sims.topk(k)`
:::

<details>
<summary>点击查看答案</summary>

```python
# TODO 1
idxs = torch.linspace(0, num_frames - 1, k).round().long().tolist()

# TODO 2
diffs = (frames[1:] - frames[:-1]).abs().mean(dim=(1, 2, 3))   # [T-1]

# TODO 3
top_diff = diffs.topk(k - 1).indices

# TODO 4
sims = frame_features @ query_feature   # [T]

# TODO 5
top = sims.topk(k)
```

**解析：**

视频采样里**没有银弹**——三种策略各有死角：

| 策略 | 优点 | 死角 | 用在哪 |
|------|------|------|--------|
| Uniform | 实现简单、覆盖全片、与 query 无关（可预先缓存） | 漏关键事件（如 30 秒视频里第 5 秒的爆炸） | Video-LLaVA / VideoChatGPT 默认 |
| Keyframe | 抓住变化点 | 单调长镜头里几乎没差分信号；摄像机晃动会被误判为关键帧 | 监控、运动视频 |
| Query-driven | 与任务对齐，召回率高 | 需要预计算所有帧的 CLIP feature；query 偏 / 错时整体崩 | VideoQA、视频检索 |

工程上的真实做法是**混合采样**：
1. **uniform 兜底**（保 N/2 帧），覆盖时序结构
2. **keyframe 加成**（保 N/4 帧），抓事件切换
3. **query top-k**（保 N/4 帧），召回任务相关帧

[Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) 默认 8 帧 uniform；LLaVA-NeXT-Video 用了"slow-fast"路线，密集小分辨率 + 稀疏高分辨率混合喂；商用 VL 模型（Gemini-Video / Qwen2-VL-Video）支持上百帧并用动态分辨率压缩。

进阶问题（生产上要解决）：
- **SSIM vs L1 diff 谁更好**？SSIM 对亮度变化更鲁棒；L1 实现简单。在监控视频中 SSIM 优于 L1。
- **第一帧总保留是好主意吗**？给 LLM 时序锚点很重要——但首帧如果是黑屏（fade-in）反而误导。可以加首帧"非黑"检测。
- **Top-K 之后要不要按时序排序**？要。LLM 吃乱序帧会丢失时序推理能力——`sort(top.indices)` 是常见 fix。

</details>

---

## 练习 5：Vision Tool Router（Level 3）

Vision Agent（如 OmniParser + GPT-4V、Claude Computer Use）的常见架构：**LLM 做 planner，调用一组视觉工具完成原子任务**。每次输入 (screenshot, user_query)，agent 需要决定调哪个工具：

- `OCR`：抽取屏幕上的文字（"截图里写了什么"）
- `detection`：定位 UI 元素或物体（"找到登录按钮"）
- `caption`：整体图像描述（"这张图在干嘛"）
- `VQA`：开放问答（"图里有几个人"）

朴素做法：让 LLM 看 query 自己决定。但 **LLM 调用一次成本高**——很多场景下用纯关键词路由器就能覆盖 80% 的请求，剩下 20% 再回退给 LLM。这个练习实现一个 **keyword + heuristic 路由器**。

```python
import re
from dataclasses import dataclass, field

@dataclass
class RouteDecision:
    tool: str
    confidence: float       # [0, 1]，1 = 非常确定
    reason: str             # 给 log / debug 用

@dataclass
class VisionToolRouter:
    """关键词驱动的 Vision tool 路由器。"""

    # 每个工具的触发关键词（可被外部 override）
    ocr_keywords: list[str] = field(default_factory=lambda: [
        "文字", "文本", "text", "字", "ocr", "读出", "识别字",
        "写的", "写着", "看清", "看不清",
    ])
    detection_keywords: list[str] = field(default_factory=lambda: [
        "在哪", "位置", "找到", "定位", "click", "按钮", "图标",
        "where", "locate", "find", "bounding box", "bbox",
    ])
    caption_keywords: list[str] = field(default_factory=lambda: [
        "描述", "总结", "概括", "describe", "caption", "summary",
        "整体", "大致", "在干什么", "在做什么",
    ])

    def route(self, user_query: str,
              screenshot_meta: dict | None = None) -> RouteDecision:
        """
        screenshot_meta: 可选，包含 image 的 metadata 帮助决策：
            - "has_text_region": bool   是否检出大段文字（OCR pre-detect）
            - "ui_elements": int        UI 元素数量（高 → 大概率是网页/app）
            - "image_size": (W, H)
        """
        q = user_query.lower().strip()

        # ---------- 规则 1: OCR 关键词命中 ----------
        # TODO 1: 命中 ocr_keywords 任何一个 → 返回 ocr，confidence=0.9
        if _____:
            return RouteDecision(
                tool="ocr", confidence=0.9,
                reason=f"matched ocr keyword in: {q!r}"
            )

        # ---------- 规则 2: detection 关键词 ----------
        if any(kw in q for kw in self.detection_keywords):
            return RouteDecision(
                tool="detection", confidence=0.9,
                reason=f"matched detection keyword in: {q!r}"
            )

        # ---------- 规则 3: caption 关键词 ----------
        if any(kw in q for kw in self.caption_keywords):
            return RouteDecision(
                tool="caption", confidence=0.85,
                reason=f"matched caption keyword in: {q!r}"
            )

        # ---------- 规则 4: 启发式（按 metadata） ----------
        if screenshot_meta:
            # TODO 2: UI 元素 > 5 且 query 含"?"或"什么/哪个/几个" → VQA on UI
            if (screenshot_meta.get("ui_elements", 0) > 5
                    and _____):
                return RouteDecision(
                    tool="vqa", confidence=0.7,
                    reason="ui-rich screen + interrogative query"
                )

            # TODO 3: query 极短（< 5 个字符） + 检出文字区域 → 大概率是想 OCR
            if (len(q) < 5
                    and screenshot_meta.get("has_text_region", False)):
                return RouteDecision(
                    tool="ocr", confidence=0.6,
                    reason="ultra-short query + text-rich screen"
                )

        # ---------- 规则 5: 兜底用 VQA（最通用） ----------
        # TODO 4: VQA 是兜底——所有疑问句都可以回答
        return RouteDecision(
            tool=_____, confidence=0.5,
            reason="fallback to general VQA"
        )


# ====== 测试 ======
router = VisionToolRouter()

cases = [
    ("截图里的文字是什么？", None,                        "ocr"),
    ("登录按钮在哪？",       None,                        "detection"),
    ("总结一下这张图",       None,                        "caption"),
    ("这张图里有几个人？",   None,                        "vqa"),       # 兜底
    ("?",                    {"ui_elements": 10,
                              "has_text_region": True},   "vqa"),       # 启发式 2
    ("好",                   {"has_text_region": True},   "ocr"),       # 启发式 3
    ("这张图怎么样",         None,                        "vqa"),       # 兜底
]

for query, meta, expected in cases:
    d = router.route(query, meta)
    ok = "OK" if d.tool == expected else "MISS"
    print(f"[{ok}] {query!r:35s} → {d.tool:10s} "
          f"(conf={d.confidence:.2f}) | {d.reason}")
```

::: details 提示
- TODO 1：`any(kw in q for kw in self.ocr_keywords)`
- TODO 2：`any(c in q for c in ["?", "？", "什么", "哪个", "几个"])`
- TODO 3：见参考答案
- TODO 4：`"vqa"`
:::

<details>
<summary>点击查看答案</summary>

```python
# TODO 1
if any(kw in q for kw in self.ocr_keywords):
    return RouteDecision(
        tool="ocr", confidence=0.9,
        reason=f"matched ocr keyword in: {q!r}"
    )

# TODO 2
if (screenshot_meta.get("ui_elements", 0) > 5
        and any(c in q for c in ["?", "？", "什么", "哪个", "几个"])):

# TODO 3 略（参见原代码）

# TODO 4
return RouteDecision(
    tool="vqa", confidence=0.5,
    reason="fallback to general VQA"
)
```

**解析：**

为什么要写一个"前置路由器"而不是直接让 LLM 决定调哪个工具？

1. **延迟成本**：让 LLM 看一遍 query 至少 200ms + 100 tokens，pure-rule 路由 < 1ms。在 Computer Use 这种 1 秒内要响应的场景里差异巨大。
2. **可解释性**：rule-based 路由的每次决策都能 log 出"匹配到了哪个关键词"，出错时一眼定位；LLM-based 路由是黑盒。
3. **Confidence 阈值 + 回退链**：高 confidence 直接走规则；低 confidence（如 < 0.6）回退给 LLM。这是 [OmniParser](https://github.com/microsoft/OmniParser) + GPT-4V 这类系统的常见模式。

工程化要点：
- **screenshot_meta 是关键**：纯文本 query + 没有图像信号会有大量歧义（"这是什么"既可能是 caption 也可能是 VQA）。一个轻量 pre-detect（"这屏幕上是不是有大段文字"）可以让路由器一下子准很多。
- **关键词列表的扩展**：实际部署时 keyword 集合会通过用户日志离线挖掘——把"被路由错了"的样本聚类，加进 keyword 列表。
- **多语言**：中文 query 与英文 query 的 keyword 集合不同。生产上会做语言检测后选择对应词表。

进阶——为什么有的 Agent 框架不需要这层路由器？
- **Anthropic Computer Use**：直接给 Claude 看屏幕，让它自己决定调什么工具。但这是因为 Claude 本身已经是端到端的"decide + execute"，且只暴露了 `screenshot / click / type` 三个原子动作——decision 空间小到可以让 LLM 直接判断。
- **OmniParser 风格**：先把屏幕变成结构化文本（element list），再让 LLM 在文本上推理。这层路由器变成了"什么时候要调 OmniParser"。

参考开源系统：[OmniParser](https://github.com/microsoft/OmniParser) 的 element parsing pipeline，[ShowUI](https://github.com/showlab/ShowUI) 的 GUI grounding，都是这道题的"重型版本"。

</details>

---

## MLM 风格巩固

完成上面的固定填空后，试试随机挖空模式 —— 每次点击「刷新」会随机遮盖不同的代码片段，帮你彻底记住每一行。

### Patch Embedding

<CodeMasker title="Patch Embed：Conv2d 一步切 patch + 投影" :mask-ratio="0.15">
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)         # [B, D, H/P, W/P]
        x = x.flatten(2)         # [B, D, N]
        x = x.transpose(1, 2)    # [B, N, D]
        return x
</CodeMasker>

### CLIP InfoNCE Loss

<CodeMasker title="CLIP Loss：双向 InfoNCE + learnable temperature" :mask-ratio="0.15">
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self, init_temperature=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / init_temperature)))

    def forward(self, image_features, text_features):
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_i2t = logit_scale * image_features @ text_features.t()
        logits_t2i = logits_i2t.t()

        N = image_features.size(0)
        labels = torch.arange(N, device=image_features.device)
        loss_i2t = F.cross_entropy(logits_i2t, labels)
        loss_t2i = F.cross_entropy(logits_t2i, labels)
        return (loss_i2t + loss_t2i) / 2
</CodeMasker>

### LLaVA Visual Projector 工厂

<CodeMasker title="Projector Factory：linear / mlpNx_gelu / identity" :mask-ratio="0.15">
import re
import torch.nn as nn

def build_visual_projector(proj_type, vision_dim, llm_dim):
    if proj_type == "linear":
        return nn.Linear(vision_dim, llm_dim)

    if proj_type == "identity":
        if vision_dim != llm_dim:
            raise ValueError("identity needs equal dims")
        return nn.Identity()

    match = re.match(r"^mlp(\d+)x_gelu$", proj_type)
    if match:
        depth = int(match.group(1))
        layers = [nn.Linear(vision_dim, llm_dim)]
        for _ in range(1, depth):
            layers.append(nn.GELU())
            layers.append(nn.Linear(llm_dim, llm_dim))
        return nn.Sequential(*layers)

    raise ValueError(f"Unknown projector: {proj_type!r}")
</CodeMasker>

### Video Frame Sampler

<CodeMasker title="Video Sampler：uniform / keyframe / query-driven" :mask-ratio="0.15">
import torch

def sample_uniform(num_frames, k):
    if k >= num_frames:
        return list(range(num_frames))
    idxs = torch.linspace(0, num_frames - 1, k).round().long().tolist()
    return sorted(set(idxs))

def sample_keyframe(frames, k):
    T = frames.size(0)
    if k >= T:
        return list(range(T))
    diffs = (frames[1:] - frames[:-1]).abs().mean(dim=(1, 2, 3))
    top_diff = diffs.topk(k - 1).indices
    keyframe_idx = (top_diff + 1).tolist()
    return sorted(set([0] + keyframe_idx))[:k]

def sample_query_driven(frame_features, query_feature, k):
    T = frame_features.size(0)
    if k >= T:
        return list(range(T))
    sims = frame_features @ query_feature
    return sims.topk(k).indices.tolist()
</CodeMasker>

### Vision Tool Router

<CodeMasker title="Vision Router：keyword + heuristic 路由" :mask-ratio="0.15">
def route(user_query, screenshot_meta=None):
    q = user_query.lower().strip()

    if any(kw in q for kw in OCR_KEYWORDS):
        return ("ocr", 0.9)

    if any(kw in q for kw in DETECTION_KEYWORDS):
        return ("detection", 0.9)

    if any(kw in q for kw in CAPTION_KEYWORDS):
        return ("caption", 0.85)

    if screenshot_meta:
        if (screenshot_meta.get("ui_elements", 0) > 5
                and any(c in q for c in ["?", "？", "什么", "哪个", "几个"])):
            return ("vqa", 0.7)
        if (len(q) < 5
                and screenshot_meta.get("has_text_region", False)):
            return ("ocr", 0.6)

    return ("vqa", 0.5)
</CodeMasker>

---

## 苏格拉底时刻

1. 练习 1 的 PatchEmbed 用 `Conv2d(k=P, s=P)` 与"unfold + Linear"在数学上等价——但梯度计算路径完全一样吗？混合精度训练下，哪种更稳？
2. 练习 2 的 logit_scale 训练后期通常会被 clamp 在 100，这意味着什么？如果不 clamp 会发生什么数值灾难？同时，为什么 CLIP 一定要双向 loss——只算 i2t 会让谁"摆烂"？
3. 练习 3 的 LLaVA projector 第一层就把 vision_dim 升到 llm_dim。如果反过来"先在低维 MLP 几层再升到 llm_dim"，参数更少，但效果通常更差——为什么？这与 information bottleneck 假设是什么关系？
4. 练习 4 的三种采样策略中，哪一种对"长视频"扩展性最差？如果视频长达 1 小时（5400 帧），你会怎么混合？keyframe 检测的"相邻帧 L1 差"在摄像机晃动场景下会失效——你怎么修？
5. 练习 5 的 router 把绝大多数 query 路由到了关键词分支。如果用户查询都是模糊指代（"那个东西"、"它"），关键词命中率会暴跌——你会引入什么 fallback 机制（除了 LLM）？

## 推荐资源

- [openai/CLIP](https://github.com/openai/CLIP) — 练习 1, 2 的 reference 实现，重点看 `model.py` 的 `VisionTransformer` 与 `loss` 函数
- [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) — 练习 3 的工业版 projector 与 instruction tuning pipeline
- [PKU-YuanGroup/Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) — 练习 4 的视频处理 reference，重点看 `videollava/model/multimodal_encoder/`
- [microsoft/OmniParser](https://github.com/microsoft/OmniParser) — 练习 5 的 screen parsing pipeline，给 Vision Agent 喂结构化 element 列表
- [QwenLM/Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) — 动态分辨率与原生多模态架构，练习 1 的"无固定 patch 数"扩展
- [ViT 论文 (An Image is Worth 16x16 Words)](https://arxiv.org/abs/2010.11929) — 练习 1 的源头
- [CLIP 论文](https://arxiv.org/abs/2103.00020) — 练习 2 的源头
- [LLaVA 论文](https://arxiv.org/abs/2304.08485) / [LLaVA-1.5 论文](https://arxiv.org/abs/2310.03744) — 练习 3 的设计思路
