---
title: 手撕 LLaVA：从零实现多模态助手
description: 用 ~500 行实现 LLaVA-style 视觉-语言模型，覆盖 projector 设计、两阶段训练、与 production 对比
topics: [deep-dive, multimodal, LLaVA, vision-projector, visual-instruction-tuning, from-scratch, CLIP-ViT, GPT-2]
prereqs: [/applications/multimodal, /architecture/transformer, /architecture/gpt]
---

# 手撕 LLaVA：从零实现多模态助手

> **一句话总结：** 把多模态大模型最重要的"视觉特征 → 文本 token 序列"那道桥拆到最简——视觉编码器冻结、一个 MLP projector、`<image>` 占位符替换、两阶段训练——用 GPT-2 small + CLIP-ViT-B/32 在单卡上跑通端到端。

::: info 配套资源
- 主线：[多模态大模型](/applications/multimodal)（ViT / CLIP / LLaVA / 原生多模态）
- 架构：[Transformer](/architecture/transformer) / [GPT 系列](/architecture/gpt)
- 同系列：[手撕 GPT](/deep-dives/nano-gpt) / [手撕 RLHF](/deep-dives/nano-rlhf) / [手撕 Agent-RL](/deep-dives/nano-agent-rl)
:::

## 体系定位

[多模态主线](/applications/multimodal) 已经讲过 LLaVA 的高层范式：**冻结的视觉编码器 + 轻量 projector + 文本 LLM**。但是高层范式背后，仍然有一些到了实现层才会清楚的细节：

- 视觉特征是 `[B, 256, 1024]`，文本 embedding 是 `[B, T, 768]`，它们是怎么"对齐"到一个序列里去的？
- 训练数据的 `<image>` 占位符在哪一步被替换成 256 个视觉 token？怎么保证 loss mask 不把图片 token 也算进去？
- 阶段 1 (pretrain projector) 和阶段 2 (visual instruction tuning) 的 freeze / unfreeze 是怎么写的？为什么不能合成一阶段？
- 推理时的 `generate` 比纯文本 LLM 多了哪一步？

本文不打算训练一个有用的 LLaVA。目标是把这些问题的代码答案放在一起，让你看完之后再去读 [LLaVA](https://github.com/haotian-liu/LLaVA) / [InternVL](https://github.com/OpenGVLab/InternVL) / [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) 时，能直接定位到它们的工程加成。

::: warning 本文不是生产实现
代码以单卡 toy 跑通为目标。生产级要点（高分辨率切图、视频帧采样、AnyRes、动态 patch、长 context 优化、多机训练）都会在最后一节列出，本文不展开。
:::

---

## 架构总览

LLaVA 的整体结构比"transformer + 一堆训练 trick"要清爽得多。**三块串联**就完事了：

```text
        ┌─────────────────┐
图片 ──▶│  视觉编码器     │  patch features  [B, N_v, D_v]
        │  CLIP-ViT-B/32  │  D_v = 768
        │  (冻结)         │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  MLP Projector  │  visual tokens   [B, N_v, D_l]
        │  Linear→GELU→   │  D_l = 768 (GPT-2)
        │  Linear         │
        └────────┬────────┘
                 │
                 ▼   插入到 <image> 占位符所在的位置
        ┌─────────────────────────────────────────┐
        │  text embed: [BOS] What is in <image> ? │
        │            ↓ 替换 <image> 为视觉 tokens │
        │  full seq:  [BOS] What is in [v1..vN] ? │
        └────────────────────┬────────────────────┘
                             ▼
        ┌─────────────────────────────────────────┐
        │  LLM (GPT-2 small)                      │
        │  阶段 1：冻结，只训 projector           │
        │  阶段 2：解冻，与 projector 一起训      │
        └────────────────────┬────────────────────┘
                             ▼
                       next-token loss
                  (mask 掉 prompt 与视觉 token)
```

三个组件、两个阶段、一道关键拼接。下面逐步实现。

---

## Step 1：视觉编码器接入

视觉编码器在整个 LLaVA 流水里只做一件事：**给定图片，输出一串 patch feature**。它在两个训练阶段都是冻结的（LLaVA-1.5 之后才解冻视觉部分）。

我们用 HuggingFace 的 `CLIPVisionModel`，因为它和 LLaVA 系列实际用的视觉塔同构。CLIP-ViT-B/32 输入 224×224，patch=32，所以 patch 数 = (224/32)² = 49，加上 [CLS] 共 50 个 token。生产 LLaVA 用的是 ViT-L/14 336×336，patch 数 = (336/14)² = 576。

```python
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

class FrozenVisionTower(nn.Module):
    """
    封装一个冻结的 CLIP-ViT，用作 LLaVA 的"视觉编码器"。

    设计要点：
    1. 只取 penultimate 层（倒数第二层）作为视觉特征。
       原因：CLIP 的最后一层是为了对齐文本-图像 contrastive loss 设计的，
       它强调全局语义而损失了局部细节；倒数第二层保留更多视觉细节，
       更适合给 LLM 做"看图说话"。
    2. 丢掉 [CLS] token，只保留 patch tokens。
       原因：LLM 不需要全局摘要——它自己会做。我们让 projector 接管 patch
       级别的表示。
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        # output_hidden_states=True 让我们能取任意层
        self.vision = CLIPVisionModel.from_pretrained(
            model_name, output_hidden_states=True
        )
        # 冻结所有参数（阶段 1 和阶段 2 都冻结）
        for p in self.vision.parameters():
            p.requires_grad = False
        self.vision.eval()
        self.hidden_size = self.vision.config.hidden_size  # 768 for B/32

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        输入  pixel_values: [B, 3, 224, 224]
        输出  patch_features: [B, N_patch, D_v]
              其中 N_patch = 49（B/32），D_v = 768
        """
        out = self.vision(pixel_values=pixel_values)
        # hidden_states 是 (embed_layer, layer_1, ..., layer_N) 共 N+1 个
        # 倒数第二层 → 索引 -2
        hs = out.hidden_states[-2]      # [B, 1+N_patch, D_v]
        # 第 0 个是 [CLS]，丢掉
        return hs[:, 1:, :]              # [B, N_patch, D_v]
```

**实现 Tip：** 视觉塔的 `forward` 加 `@torch.no_grad()` 装饰器是个工程小动作但很重要——它保证视觉部分**完全不进入 autograd 图**，省下大量显存。如果你忘记加，PyTorch 仍然会为冻结参数构建 graph（只是不计算梯度），冻结的视觉 ViT 会让显存占用翻倍。

---

## Step 2：MLP Projector 设计

Projector 是 LLaVA 全家最便宜也最关键的模块——它就是连接视觉和语言的"那道桥"。LLaVA-1.0 用的是单层 Linear；LLaVA-1.5 把它换成两层 MLP（`Linear → GELU → Linear`），并且**这一个改动把 benchmark 抬高了几个点**。

我们实现两种 projector，方便对比：

```python
class LinearProjector(nn.Module):
    """LLaVA-1.0 风格：单层线性映射，最简但表达力有限。"""
    def __init__(self, d_vision: int, d_llm: int):
        super().__init__()
        self.proj = nn.Linear(d_vision, d_llm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N_v, D_v] → [B, N_v, D_l]
        return self.proj(x)


class MLP2xGeluProjector(nn.Module):
    """
    LLaVA-1.5 风格：两层 MLP，中间一个 GELU。
    
    为什么是两层：
    - 视觉空间和语言空间的"距离"不止是一次仿射变换能拉近的，
      多一个非线性能学到更复杂的对齐函数。
    - 但太深也没意义——视觉特征已经很抽象了，再叠 4 层基本无收益，
      只会增加阶段 1 的训练时间。两层是个 sweet spot。
    
    为什么不加 LayerNorm：
    - 视觉特征本身已经被 ViT 内的 LN 归一化过；
    - LLM 入口已经会做 LN；
    - 中间这层 LN 经验上反而拉低 benchmark（LLaVA-1.5 ablation）。
    """
    def __init__(self, d_vision: int, d_llm: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_vision, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

::: details 为什么不能用 cross-attention？（Flamingo / BLIP-2 路线对比）
[Flamingo](https://arxiv.org/abs/2204.14198) 和 [BLIP-2](https://arxiv.org/abs/2301.12597) 走的是另一条路线：在 LLM 的若干层之间插入 **gated cross-attention** 或 **Q-Former**，让 LLM 通过 attention 机制"查询"视觉特征。

LLaVA 走的是更简单的路线：把视觉特征**当成 token 序列直接拼到文本前面**，让 LLM 用自身的 self-attention 处理两种模态。

两条路线的工程取舍：
- **LLaVA 路线**：projector 极轻（百万参数级）；LLM 不动结构；缺点是视觉 token 占住 context window 导致长度成本高。
- **Flamingo 路线**：cross-attention 模块更重；LLM 结构被改；优点是视觉特征不占 LLM 的 context window。

LLaVA 后来"赢了"主要是因为：(a) 它能直接吃任何现成的预训练 LLM，(b) 训练数据合成简单（直接走 instruction tuning），(c) 视觉 token 占 context 的问题可以靠 token compression（如 Honeybee 的 Resampler）解决。
:::

把 projector 写成 builder 函数：

```python
def build_projector(kind: str, d_vision: int, d_llm: int) -> nn.Module:
    if kind == "linear":
        return LinearProjector(d_vision, d_llm)
    if kind == "mlp2x_gelu":
        return MLP2xGeluProjector(d_vision, d_llm)
    raise ValueError(f"unknown projector type: {kind}")
```

---

## Step 3：视觉-文本拼接

这是整个 LLaVA 的"魔法点"，也是从零实现时最容易写错的地方。流程是：

1. 把训练样本里的图片占位符 `<image>` 加入 tokenizer 词表（特殊 token）。
2. 文本走标准 tokenize，拿到 input_ids。
3. **在 input embedding 阶段**——也就是过 `wte` 之后但还没进 transformer 之前——找到 `<image>` 那个位置，把那一个 token 的 embedding 替换成 N 个视觉 token 的 embedding。
4. 同步生成 `attention_mask` 和 `labels`（labels 中视觉 token 位置标 -100，避免算 loss）。

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100   # cross_entropy 默认会忽略的 label 值


class LlavaForCausalLM(nn.Module):
    """nano LLaVA：CLIP 视觉塔 + MLP projector + GPT-2 LLM。"""

    def __init__(self,
                 llm_name: str = "gpt2",
                 vision_name: str = "openai/clip-vit-base-patch32",
                 projector_kind: str = "mlp2x_gelu"):
        super().__init__()
        # 1. LLM
        self.tokenizer = GPT2Tokenizer.from_pretrained(llm_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # 加 <image> 特殊 token
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [IMAGE_TOKEN]})
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        self.llm = GPT2LMHeadModel.from_pretrained(llm_name)
        # 词表加了一个 token，必须 resize embedding 矩阵
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.d_llm = self.llm.config.hidden_size  # 768 for gpt2 small

        # 2. 视觉塔
        self.vision = FrozenVisionTower(vision_name)
        self.d_vision = self.vision.hidden_size

        # 3. Projector
        self.projector = build_projector(
            projector_kind, self.d_vision, self.d_llm)

    # ------------------------------------------------------------
    # 关键函数：把视觉 features 嵌入到文本 embedding 序列中
    # ------------------------------------------------------------
    def merge_visual_into_text(self,
                               input_ids: torch.Tensor,
                               attention_mask: torch.Tensor,
                               labels: torch.Tensor,
                               pixel_values: torch.Tensor):
        """
        输入:
          input_ids:    [B, T_text]    含 <image> 占位符的 token id
          attention_mask: [B, T_text]
          labels:       [B, T_text]    与 input_ids 同形状，prompt 段为 -100
          pixel_values: [B, 3, H, W]
        
        输出:
          inputs_embeds:   [B, T_full, D_l]
          attention_mask:  [B, T_full]
          labels:          [B, T_full]   视觉 token 位置全部为 -100
        """
        B = input_ids.size(0)
        # ---- 1. 算视觉 token ----
        vis_feats = self.vision(pixel_values)         # [B, N_v, D_v]
        vis_tokens = self.projector(vis_feats)        # [B, N_v, D_l]
        N_v = vis_tokens.size(1)

        # ---- 2. 文本走 wte 拿到 embedding ----
        text_embeds = self.llm.transformer.wte(input_ids)  # [B, T_text, D_l]

        # ---- 3. 在每条样本里替换 <image> 为 N_v 个视觉 token ----
        new_embeds, new_masks, new_labels = [], [], []
        for b in range(B):
            ids = input_ids[b]
            # 假设每条样本恰好一个 <image>（多图情况见末尾扩展）
            pos = (ids == self.image_token_id).nonzero(as_tuple=True)[0]
            assert len(pos) == 1, "本 demo 只支持单张图片"
            p = pos.item()

            # 文本左段 / 右段 embedding
            left  = text_embeds[b, :p]                  # [p, D_l]
            right = text_embeds[b, p+1:]                # [T_text-p-1, D_l]
            merged = torch.cat([left, vis_tokens[b], right], dim=0)
            # 同步 mask 和 labels
            mask_left  = attention_mask[b, :p]
            mask_vis   = torch.ones(N_v, dtype=attention_mask.dtype,
                                    device=attention_mask.device)
            mask_right = attention_mask[b, p+1:]
            merged_mask = torch.cat([mask_left, mask_vis, mask_right])

            lbl_left  = labels[b, :p]
            lbl_vis   = torch.full((N_v,), IGNORE_INDEX,
                                   dtype=labels.dtype, device=labels.device)
            lbl_right = labels[b, p+1:]
            merged_lbl = torch.cat([lbl_left, lbl_vis, lbl_right])

            new_embeds.append(merged)
            new_masks.append(merged_mask)
            new_labels.append(merged_lbl)

        # 因为每条都从 1 个占位符变成 N_v 个 token，长度变化一致，可以直接 stack
        return (torch.stack(new_embeds),
                torch.stack(new_masks),
                torch.stack(new_labels))

    def forward(self, input_ids, attention_mask, labels, pixel_values):
        embeds, mask, lbl = self.merge_visual_into_text(
            input_ids, attention_mask, labels, pixel_values)
        # GPT-2 的 forward 同时支持 input_ids 和 inputs_embeds，二选一
        out = self.llm(inputs_embeds=embeds,
                       attention_mask=mask,
                       labels=lbl)
        return out  # out.loss 已经按 -100 mask 算好了
```

::: warning 三个最常踩的坑

1. **`labels` 必须把视觉 token 位置标成 -100。** 否则 LLM 会在视觉 token 上算 cross_entropy，目标是 vocab 里的某个 token id——但你的视觉 embedding 从来不在 vocab 里，loss 会爆炸。
2. **`attention_mask` 视觉 token 位置必须为 1。** 否则 LLM 会以为这部分是 padding，attention 会 mask 掉视觉信息，模型永远学不到。
3. **不要在视觉 token 位置加 position_id 偏移。** GPT-2 的位置编码是 `0..T-1`，我们的视觉 token 自然占据 position（紧接前面文本 token），不需要任何偏移——LLM 会把它们当成"普通的位置 N..N+N_v-1 的 token"。

:::

---

## Step 4：阶段 1 训练（特征对齐）

阶段 1 的目标是让 projector 学会"把 CLIP 的视觉空间投影到 GPT-2 的语言空间"。这一步**只训 projector**，视觉塔和 LLM 都冻结。

数据用 image-caption 对（本 demo 直接造 5 条假数据；真实实验用 [LAION-CC-SBU-558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) 或 [CC3M](https://github.com/google-research-datasets/conceptual-captions)）。Prompt 格式：

```
USER: <image>
描述这张图片。
ASSISTANT: <caption>
```

```python
def freeze_for_stage1(model: LlavaForCausalLM):
    """阶段 1：只训 projector。"""
    for p in model.llm.parameters():
        p.requires_grad = False
    for p in model.vision.parameters():
        p.requires_grad = False
    for p in model.projector.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[stage1] trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M "
          f"({100*trainable/total:.2f}%)")


def build_stage1_sample(tokenizer, caption: str, n_image_tokens_placeholder=1):
    """
    构造一条阶段 1 样本（单图 captioning）。
    注意：这里 input_ids 里只放 1 个 <image> 占位符，
    在 forward 时由 merge_visual_into_text 替换成 N_v 个真视觉 token。
    """
    prompt = f"USER: {IMAGE_TOKEN}\n描述这张图片。\nASSISTANT:"
    full   = prompt + " " + caption + tokenizer.eos_token

    ids_full   = tokenizer(full,   return_tensors="pt").input_ids[0]
    ids_prompt = tokenizer(prompt, return_tensors="pt").input_ids[0]
    L_p = len(ids_prompt)

    labels = ids_full.clone()
    labels[:L_p] = IGNORE_INDEX            # 不在 prompt 上算 loss
    attention_mask = torch.ones_like(ids_full)
    return ids_full, attention_mask, labels


def train_stage1(model, image_caption_pairs, image_processor,
                 lr=1e-3, epochs=3):
    freeze_for_stage1(model)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    for ep in range(epochs):
        total_loss = 0.0
        for img, caption in image_caption_pairs:
            ids, mask, lbl = build_stage1_sample(model.tokenizer, caption)
            pix = image_processor(images=img,
                                  return_tensors="pt").pixel_values

            out = model(
                input_ids=ids.unsqueeze(0),
                attention_mask=mask.unsqueeze(0),
                labels=lbl.unsqueeze(0),
                pixel_values=pix,
            )
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            total_loss += out.loss.item()
        print(f"[stage1] epoch {ep}: loss = "
              f"{total_loss/len(image_caption_pairs):.3f}")
```

::: tip 阶段 1 的学习率为什么可以调到 1e-3？
因为只训一个轻量 projector（百万参数量级）、且不动 LLM 和 ViT，learning rate 可以远高于全参微调（通常 2e-5）。LLaVA-1.5 阶段 1 用的是 **1e-3**（全模型 SFT 是 2e-5），差了两个数量级。这是"小模块独立预训练"的典型操作。
:::

---

## Step 5：阶段 2 训练（指令微调）

阶段 2 解冻 LLM，**LLM + projector 一起训**，视觉塔仍冻结（LLaVA-1.5 的设定；LLaVA-NeXT 之后才解冻视觉）。数据从 captioning 切到 visual instruction tuning（多轮对话 + 视觉问答），格式：

```
USER: <image>
这张图里有什么？
ASSISTANT: 一只橘色的猫。
USER: 它在做什么？
ASSISTANT: 它趴在沙发上睡觉。
```

```python
def freeze_for_stage2(model: LlavaForCausalLM):
    """阶段 2：解冻 LLM + projector，仍然冻结视觉。"""
    for p in model.vision.parameters():
        p.requires_grad = False
    for p in model.llm.parameters():
        p.requires_grad = True
    for p in model.projector.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[stage2] trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M "
          f"({100*trainable/total:.2f}%)")


def build_stage2_sample(tokenizer, conversations: list[tuple[str, str]]):
    """
    conversations 是 [(user_msg, assistant_msg), ...]，
    第一轮 user_msg 必须以 <image> 开头（多图扩展见末尾）。
    
    Loss 只算 ASSISTANT 段，USER 段全部 mask 成 -100。
    """
    text_pieces = []
    label_mask = []   # 与 text_pieces 一一对应：True 表示算 loss
    for u, a in conversations:
        text_pieces.append(f"USER: {u}\nASSISTANT:")
        label_mask.append(False)
        text_pieces.append(f" {a}{tokenizer.eos_token}\n")
        label_mask.append(True)

    ids_list, lbl_list = [], []
    for txt, do_loss in zip(text_pieces, label_mask):
        piece_ids = tokenizer(txt, return_tensors="pt").input_ids[0]
        ids_list.append(piece_ids)
        if do_loss:
            lbl_list.append(piece_ids.clone())
        else:
            lbl_list.append(torch.full_like(piece_ids, IGNORE_INDEX))

    input_ids = torch.cat(ids_list)
    labels    = torch.cat(lbl_list)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, labels


def train_stage2(model, conv_dataset, image_processor,
                 lr=2e-5, epochs=3):
    freeze_for_stage2(model)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    for ep in range(epochs):
        total_loss = 0.0
        for img, conv in conv_dataset:
            ids, mask, lbl = build_stage2_sample(model.tokenizer, conv)
            pix = image_processor(images=img,
                                  return_tensors="pt").pixel_values
            out = model(
                input_ids=ids.unsqueeze(0),
                attention_mask=mask.unsqueeze(0),
                labels=lbl.unsqueeze(0),
                pixel_values=pix,
            )
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            total_loss += out.loss.item()
        print(f"[stage2] epoch {ep}: loss = "
              f"{total_loss/len(conv_dataset):.3f}")
```

**为什么必须分两阶段，不能合一**：

| 假想：合一阶段 | 现实：两阶段 |
|----------------|-------------|
| projector 随机初始化，LLM 立刻面对一堆"乱码视觉 token" | 阶段 1 把 projector 训到位，LLM 接到的视觉 token 已经像"伪文本" |
| LLM 一边受到大量噪声视觉 token 影响、一边要做 next-token 预测，loss 巨大且不稳定 | 阶段 2 接管时 loss 已经是个合理水平 |
| 容易出现"忘记原本语言能力"的 catastrophic forgetting | 阶段 1 LLM 被冻结，语言能力毫发无损 |

LLaVA 论文里 ablate 过：直接合一阶段会导致 benchmark 下降 5-10 个点。

::: details 阶段 1 / 阶段 2 数据规模为什么差这么多？
LLaVA-1.5：
- 阶段 1：558K image-caption（弱标注，互联网抓取）
- 阶段 2：665K instruction（GPT-4V 合成 + 学术 VQA 数据）

阶段 1 数据"廉价但量大"，目标是让 projector 在视觉空间到语言空间之间建立一个粗对齐。阶段 2 数据"昂贵但量小"，目标是教 LLM "怎么用视觉信息做问答"。这种"廉价对齐 + 昂贵微调"的两段式设计已经成为多模态训练的事实标准（[Qwen2-VL](https://arxiv.org/abs/2409.12191)、[InternVL-2](https://github.com/OpenGVLab/InternVL)、[MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) 都是这套）。
:::

---

## Step 6：推理与生成

推理比训练简单——除了多了一步"先把图片走 vision + projector"。

```python
@torch.no_grad()
def generate(model: LlavaForCausalLM,
             image,
             question: str,
             image_processor,
             max_new_tokens: int = 64,
             temperature: float = 0.7) -> str:
    """
    单图问答推理。
    
    流程：
    1. 把 prompt 拼成 "USER: <image>\n{question}\nASSISTANT:"
    2. tokenize → input_ids
    3. 用 merge_visual_into_text 把 <image> 替换成视觉 token，得到 inputs_embeds
    4. 走 LLM.generate，但用 inputs_embeds 而非 input_ids
    """
    model.eval()
    prompt = f"USER: {IMAGE_TOKEN}\n{question}\nASSISTANT:"
    ids = model.tokenizer(prompt, return_tensors="pt").input_ids       # [1, T]
    mask = torch.ones_like(ids)
    lbl  = torch.full_like(ids, IGNORE_INDEX)  # 推理用不到 labels，占位
    pix  = image_processor(images=image, return_tensors="pt").pixel_values

    embeds, mask, _ = model.merge_visual_into_text(ids, mask, lbl, pix)

    # 核心：直接喂 inputs_embeds 给 generate
    out_ids = model.llm.generate(
        inputs_embeds=embeds,
        attention_mask=mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=model.tokenizer.eos_token_id,
    )
    # generate 返回的是新生成的 token ids（不含 prompt——因为 prompt 是 embeds 给的）
    return model.tokenizer.decode(out_ids[0], skip_special_tokens=True)
```

::: warning `inputs_embeds` 与 `input_ids` 不要混传
HuggingFace 的 `generate` 默认接收 `input_ids` 然后查 `wte` 转 embedding。但我们的视觉 token 不在 vocab 里，没有 token id，所以必须直接从 embedding 层下手。

注意：HF 的 `generate` 用 `inputs_embeds` 时返回值的语义会变——它**只返回新生成的 token id 序列**，不再把 prompt 包在前面。这是版本相关的小坑，写测试时容易翻车。
:::

---

## 端到端 Demo：5 行跑起来

把上面所有零件拼起来：

```python
from PIL import Image
from transformers import CLIPImageProcessor

# ---------- 1. 装配模型 ----------
model = LlavaForCausalLM(
    llm_name="gpt2",
    vision_name="openai/clip-vit-base-patch32",
    projector_kind="mlp2x_gelu",
)
image_processor = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-base-patch32")

# ---------- 2. 造 toy 数据 ----------
def fake_image(color):
    """造一张 224×224 纯色图（演示用）。"""
    return Image.new("RGB", (224, 224), color=color)

stage1_pairs = [
    (fake_image("red"),    "一片红色。"),
    (fake_image("green"),  "一片绿色。"),
    (fake_image("blue"),   "一片蓝色。"),
    (fake_image("yellow"), "一片黄色。"),
    (fake_image("white"),  "一片白色。"),
]

stage2_data = [
    (fake_image("red"),
     [("这张图什么颜色？", "红色。"),
      ("你确定吗？", "我很确定，整张图都是红色。")]),
    (fake_image("blue"),
     [("这张图什么颜色？", "蓝色。")]),
]

# ---------- 3. 跑两阶段训练 ----------
train_stage1(model, stage1_pairs, image_processor, lr=1e-3, epochs=20)
train_stage2(model, stage2_data,  image_processor, lr=2e-5, epochs=10)

# ---------- 4. 推理 ----------
ans = generate(model, fake_image("green"), "这张图什么颜色？",
               image_processor, max_new_tokens=20)
print("模型输出:", ans)
```

**预期表现**：

| 阶段 | 输出 |
|------|------|
| 训练前 | "the the the the..."（GPT-2 base 看到陌生 embedding 直接乱码） |
| 阶段 1 后 | 形式像句子但内容与图无关："一片白色。"（projector 学到"输出 caption 风格"，但还分不清颜色——因为 LLM 冻结，没学过颜色） |
| 阶段 2 后 | "绿色。"（LLM 在 visual instruction tuning 数据上学会了把视觉特征对应到颜色词） |

注意：纯色图 + GPT-2 small + 5 张训练图当然学不出真正的多模态能力——这只是为了**让流水线跑通**。真正想看到能识图，至少要 LLaVA-1.5 量级的数据 + 模型。

---

## nano vs production：差在哪？

| 维度 | nano-llava（本文） | LLaVA-1.5 (production) |
|------|--------------------|------------------------|
| 视觉编码器 | CLIP-ViT-B/32 (224, 49 patch) | CLIP-ViT-L/14 (336, 576 patch) |
| LLM | GPT-2 small (124M) | Vicuna-7B / 13B |
| Projector | Linear / MLP2xGELU (~0.5M 参数) | MLP2xGELU (~20M 参数) |
| 阶段 1 数据 | 5 张假图 | LAION-CC-SBU 558K image-caption |
| 阶段 2 数据 | 2 段假对话 | 665K visual instruction (含 GPT-4V 合成) |
| 训练硬件 | 单卡 CPU 即可 | 8× A100 80G |
| 训练时长 | 2 分钟 | 阶段 1: 几小时；阶段 2: 一天 |
| Batch size | 1 | 256 (global) |
| Learning rate | 1e-3 / 2e-5 | 1e-3 / 2e-5（一致） |
| 高分辨率支持 | 无 | LLaVA-NeXT 加 AnyRes，分块切图 |
| Benchmark | 玩具不评测 | MMBench 67.7, ScienceQA 71.6, MMMU 36.4 |

代码量对比：本文核心代码 ~250 行，LLaVA 仓库 `llava/model/` 约 1500 行，外加 `llava/train/`、`llava/serve/`、`llava/eval/` 总计 1 万+ 行。

**核心算法部分（projector + 拼接 + 两阶段）的代码占比：5%。**

剩下的 95% 都是工程：DeepSpeed 集成、AnyRes 切图、LoRA 训练支持、多模态 batch 拼接（不同图片 patch 数不同，要 pad）、流式推理、Web UI、各种 benchmark eval 脚本……

---

## 扩展路线：从 nano 到生产

如果想把这套代码扩展到能在 MMBench 上拿到 50+ 分的水平，最小改动清单：

**第一档（一周可达）：**
1. **换数据**：阶段 1 用 [LLaVA-Pretrain-558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)，阶段 2 用 [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)。
2. **换 LLM**：从 GPT-2 换成 `Qwen2.5-1.5B-Instruct`，conv format 用 ChatML 而非 USER/ASSISTANT。
3. **换 ViT**：CLIP-ViT-B/32 换成 [SigLIP-So400m/14](https://huggingface.co/google/siglip-so400m-patch14-384)，分辨率上到 384。
4. **加 batch 训练**：本文的 batch=1 是为了拼接逻辑简单。生产化要做"按 patch 数分桶 + pad 到相同长度"。

**第二档（一个月可达）：**
5. **AnyRes 高分辨率**：参考 [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) 的 grid splitting，把 1024×768 切成 4 块各跑 ViT，patch 数从 576 涨到 ~2880，benchmark 显著上涨。
6. **加 LoRA 训练**：阶段 2 用 LoRA 而非全参，省 4× 显存。
7. **加 multi-image 与 video 支持**：tokenizer 加 `<video>` 占位符，多帧直接展开为多个 image。
8. **Token compression**：参考 Honeybee 的 Resampler 或 InternVL 的 PixelShuffle，把 N_v 从 576 压到 144，省 4× LLM 推理 cost。

**第三档（产品级）：**
9. **解冻视觉塔**：参考 LLaVA-NeXT 的 stage-2.5（视觉部分以 5e-6 极小 lr 联训）。
10. **多机训练 + DeepSpeed Zero-3**：模型上到 13B/34B 规模。
11. **多模态偏好对齐**：用 [RLHF-V](https://github.com/RLHF-V/RLHF-V) / 多模态 DPO 减幻觉。
12. **原生多模态**：从"接 projector"切换到"重训 native multimodal"——参考 [Chameleon](https://arxiv.org/abs/2405.09818)、[Gemini](https://deepmind.google/technologies/gemini/) 的路线。这就是另一篇文章了。

---

## 苏格拉底时刻

1. 本文的 `merge_visual_into_text` 假设每条样本恰好一个 `<image>`。如果数据里出现多图（比如"对比图 A 和图 B 的差异"），代码要怎么改？拼接顺序对模型理解多图关系有影响吗？

2. 我们把 CLIP 倒数第二层的 patch features 喂给 projector。为什么不用最后一层？为什么不用所有层的 concat？为什么不丢 [CLS] 也行（其实 LLaVA-1.5 也确实丢了）？这三个选择背后各自的权衡是什么？

3. 阶段 1 lr=1e-3，阶段 2 lr=2e-5，差了 50 倍。如果阶段 2 想"额外多训一个新加的 projector 层"，给那层单独配 lr=1e-3，剩下的 LLM 用 2e-5——这种"分组学习率"在 PyTorch 里怎么实现？这是真实的 LLaVA-NeXT 训练 trick。

4. 推理时的 KV cache 是怎么和视觉 token 兼容的？第一次 forward 喂 `inputs_embeds=[text + visual + text]`，cache 后第二次 forward 喂的是 `inputs_embeds=[new_token]`——视觉 token 还在 cache 里吗？什么时候会失效？

5. 假设你接到一个任务："把 LLaVA 改成既能看图又能听音"。最少需要改几个组件？哪些组件可以零改动？projector 还是同一个还是要分两个？这个问题的答案就是 [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215) / [Step-Audio](https://github.com/stepfun-ai/Step-Audio) 之类原生多模态系统的架构核心。

---

## 推荐资源

**从零实现 / 简化版参考：**
- [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) — 官方实现，阅读 `llava/model/llava_arch.py` 与 `llava/model/multimodal_projector/builder.py` 即可看懂全部架构
- [LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) — AnyRes、视频、多图扩展
- [tinyllava-factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) — 教学向"小模型 LLaVA"，比 nano 真实，比官方好读

**生产级多模态 LLM（推荐阅读顺序）：**
- [OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) — 端侧多模态最强，3B 体量做到 GPT-4V 接近水平
- [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL) — 全开源 SOTA，PixelShuffle token 压缩值得读
- [QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — 动态分辨率 + M-RoPE
- [DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2) — 高效混合专家多模态

**论文：**
- [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) — LLaVA 原论文，必读
- [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744) — LLaVA-1.5，MLP projector 的来源
- [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/) — AnyRes / 高分辨率 blog
- [Flamingo](https://arxiv.org/abs/2204.14198) — cross-attention 路线对照
- [BLIP-2](https://arxiv.org/abs/2301.12597) — Q-Former 路线对照

**本项目相关：**
- 主线：[多模态大模型](/applications/multimodal)
- 架构基础：[ViT 与 Patch Embedding](/applications/multimodal#vit-vision-transformer-深度解析) / [Transformer](/architecture/transformer)
- 训练范式：[SFT](/training/sft) / [偏好对齐](/training/alignment)
- 同系列手撕：[nano-gpt](/deep-dives/nano-gpt) / [nano-rlhf](/deep-dives/nano-rlhf) / [nano-agent-rl](/deep-dives/nano-agent-rl) / [lora-from-scratch](/deep-dives/lora-from-scratch)
