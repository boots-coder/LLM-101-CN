---
title: "LoRA 代码填空"
description: "Level 2-3 填空：LoRA 初始化、权重合并、完整模块、模型注入"
topics: [fill-in, LoRA, low-rank-adaptation]
---
# LoRA 代码填空 (Level 2-3)

本练习基于 LoRA（Low-Rank Adaptation）论文的核心思想，从低秩矩阵初始化到完整模块实现，逐步掌握参数高效微调技术。

::: info 前置知识
- 线性代数基础（矩阵乘法、低秩分解）
- PyTorch 基础（`nn.Module`、`nn.Linear`、`nn.Parameter`）
- Transformer 架构中的线性层
:::

::: tip LoRA 核心公式（LoRA 论文 Hu et al., 2021）
$$h = Wx + \frac{\alpha}{r}\,B A\,x$$

其中 $A \in \mathbb{R}^{r \times d_{\text{in}}}$（用 Kaiming 初始化），$B \in \mathbb{R}^{d_{\text{out}} \times r}$（零初始化），$r \ll d$ 为秩，$\alpha/r$ 为缩放系数（论文采用与 r 解耦的缩放）。
:::

> 完整可运行的从零实现见 [深度剖析 LoRA](/deep-dives/lora-from-scratch)；本页是配套的填空练习。

---

## 练习 1：LoRA 旁路前向（Level 2）

请补全 `__init__` 中的矩阵初始化和 `forward` 中的前向计算。

```python
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """LoRA: h = base(x) + (alpha / r) · B A x"""
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        d_out, d_in = base.weight.shape
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.empty(d_out, r))

        # 冻结原权重
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # TODO: 初始化 lora_A（Kaiming uniform，a=sqrt(5)）和 lora_B（全零）
        # 目的：训练初始时 Δ W = B A = 0，不改变预训练行为
        _____  # lora_A: Kaiming uniform 初始化
        _____  # lora_B: 零初始化

    def forward(self, x):
        # TODO: 在 base(x) 上叠加 (alpha / r) · B A x
        h = self.base(x)
        delta = _____    # 提示：x @ lora_A.T @ lora_B.T，再乘 self.scaling
        return h + delta
```

::: details 提示
- `nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))` 与 `nn.Linear` 默认初始化一致
- `nn.init.zeros_(self.lora_B)` 让 $\Delta W = 0$
- LoRA 旁路：输入先过 $A$（降到 $r$ 维），再过 $B$（升回 $d_{\text{out}}$），按 `scaling` 缩放
:::


<details>
<summary>点击查看答案</summary>

```python
# 初始化部分
nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
nn.init.zeros_(self.lora_B)

# forward 部分
delta = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
```

**解析：**

1. **A 用 Kaiming**：保证降维投影的方差稳定，训练初期能产生有意义的梯度。
2. **B 用零**：$\Delta W = B A$，当 $B = 0$ 时旁路输出为 0，模型在训练开始时与预训练模型行为一致。
3. **forward**：`x @ A^T` 把输入降到 $r$ 维，`@ B^T` 升回 $d_{\text{out}}$，按 $\alpha / r$ 缩放后加到原输出上。

</details>

---

## 练习 2：LoRA 的 merge 与 unmerge（Level 2）

推理阶段可将 LoRA 合并回原始权重，避免额外乘法开销。请补全 `merge` 和 `unmerge`。

```python
class LoRALinear(nn.Module):
    def __init__(self, base, r=16, alpha=32):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False

        d_out, d_in = base.weight.shape
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def merge(self):
        """将 LoRA 增量合并到原始权重"""
        if not self.merged:
            # TODO: 计算 ΔW = (alpha / r) · B A，并加到 base.weight 上
            # 注意 nn.Linear.weight 形状为 [d_out, d_in]
            delta_W = _____
            self.base.weight.data += _____
            self.merged = True

    def unmerge(self):
        """撤销 merge"""
        if self.merged:
            delta_W = _____
            self.base.weight.data -= _____
            self.merged = False

    def forward(self, x):
        if self.merged:
            return self.base(x)
        h = self.base(x)
        delta = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return h + delta
```


<details>
<summary>点击查看答案</summary>

```python
def merge(self):
    if not self.merged:
        delta_W = self.lora_B @ self.lora_A          # [d_out, d_in]
        self.base.weight.data += self.scaling * delta_W
        self.merged = True

def unmerge(self):
    if self.merged:
        delta_W = self.lora_B @ self.lora_A
        self.base.weight.data -= self.scaling * delta_W
        self.merged = False
```

**解析：**

1. **merge**：$W' = W + (\alpha/r) \cdot B A$。合并后推理仅需一次矩阵乘法，零额外开销。
2. **unmerge**：merge 的逆操作，便于在训练 / 切换 adapter 时恢复原始权重。
3. **形状对齐**：`B @ A` 形状是 `[d_out, r] @ [r, d_in] = [d_out, d_in]`，与 `nn.Linear.weight` 一致。

</details>

---

## 练习 3：完整 LoRALinear 模块（Level 3）

请独立实现一个完整的 `LoRALinear`：冻结原权重、Kaiming + 零初始化、forward 含旁路、支持 merge/unmerge。

```python
class LoRALinear(nn.Module):
    """
    base 行为:
      未合并: h = base(x) + (alpha/r) · B A x
      已合并: h = (W + (alpha/r) · B A)x + bias
    """
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        # TODO: 初始化、冻结、注册 lora_A / lora_B
        _____

    def forward(self, x):
        # TODO
        _____

    def merge(self):
        # TODO
        _____

    def unmerge(self):
        # TODO
        _____


# ====== 测试 ======
torch.manual_seed(42)
linear = nn.Linear(512, 256)
x = torch.randn(2, 10, 512)

lora = LoRALinear(linear, r=8, alpha=16)

with torch.no_grad():
    y_ref = linear(x)
    y_lora = lora(x)
    assert torch.allclose(y_ref, y_lora, atol=1e-5)   # B=0，旁路输出为 0

lora.merge()
with torch.no_grad():
    assert torch.allclose(y_ref, lora(x), atol=1e-5)  # 初始化时 ΔW = 0

lora.unmerge()
with torch.no_grad():
    assert torch.allclose(y_ref, lora(x), atol=1e-5)
print("通过")
```

::: details 提示
- 参考练习 1、2
- 处理 `base.bias` 为 `None` 的情况
- 用 `self.merged` 标志位驱动 forward 的两条路径
:::


<details>
<summary>点击查看答案</summary>

参考实现见 [深度剖析 LoRA](/deep-dives/lora-from-scratch#核心实现-50-行)，采用 `lora_A`、`lora_B`、`scaling = alpha/r` 的标准 LoRA 论文风格。

</details>

---

## 练习 4：给预训练模型注入 LoRA（Level 3）

实现 `inject_lora`：递归遍历模型，把 `target_modules` 指定名称的 `nn.Linear` 替换为 `LoRALinear`。

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )


def inject_lora(model, target_modules, r=16, alpha=32):
    """
    递归遍历 model，把 target_modules 中的 nn.Linear 替换为 LoRALinear。

    target_modules: list[str]，如 ["q_proj", "k_proj", "v_proj", "o_proj"]
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and _____:
            # TODO: 用 LoRALinear 包装并替换
            _____
        else:
            # TODO: 递归
            _____
    return model
```


<details>
<summary>点击查看答案</summary>

```python
def inject_lora(model, target_modules, r=16, alpha=32):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in target_modules:
            setattr(model, name, LoRALinear(module, r=r, alpha=alpha))
        else:
            inject_lora(module, target_modules, r, alpha)
    return model
```

**对比 HuggingFace PEFT：**

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,                                # 论文默认 α = 2r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
```

PEFT 内部做的事和上面的 `inject_lora` 一致，只是多了 dropout、状态序列化、多 adapter 切换等工程化能力。

</details>
