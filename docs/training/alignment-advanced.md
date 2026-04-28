---
title: "对齐进阶"
description: "Constitutional AI、RLAIF、Reward Hacking、安全对齐的深层挑战"
topics: [constitutional-AI, RLAIF, reward-hacking, alignment-tax, scalable-oversight, red-teaming, safety]
prereqs: [training/alignment]
---
# 对齐进阶

> **一句话总结:** 基础对齐（RLHF/DPO）解决了"让模型听话"，但进阶对齐要解决更难的问题——如何在不依赖大量人工标注的情况下实现安全对齐、如何防止模型"钻规则漏洞"、以及如何在安全性和有用性之间找到平衡。

## 在大模型体系中的位置

```
Training Pipeline
  ├── Pretraining       → 模型"知道什么"
  ├── SFT               → 模型"怎么说话"
  ├── RLHF/DPO          → 模型"说得多好"（基础对齐）
  └── Advanced Alignment ◄── 你在这里：模型"如何安全可靠地服务"
```

基础对齐（alignment.md）教你**怎么训练**；本章教你**训练时要考虑什么**——这些问题决定了模型能否从实验室安全地走向生产。

---

## Constitutional AI (CAI)

### 核心思想

Anthropic 提出的 Constitutional AI 用一组**明确的原则（Constitution）**来指导模型的自我改进，减少对人工标注偏好数据的依赖。

**两阶段流程**：

**阶段一：自我批评（Critique + Revision）**

```
1. 模型生成初始回答（可能有害）
2. 模型依据宪法原则自我批评
3. 模型修改回答使其符合原则
4. 用修改后的回答做 SFT
```

**阶段二：RLAIF（用 AI 反馈替代人类反馈）**

```
1. 模型对同一 prompt 生成多个回答
2. 另一个模型依据宪法原则判断哪个更好
3. 构造偏好对，用 RL（或 DPO）训练
```

### 宪法示例

```
原则 1: 选择不鼓励暴力或威胁的回答
原则 2: 选择没有种族、性别、宗教歧视的回答  
原则 3: 选择最有帮助、准确且无害的回答
原则 4: 选择不帮助用户进行非法活动的回答
原则 5: 选择不编造事实的回答
```

### 为什么 CAI 重要

| 对比 | 传统 RLHF | Constitutional AI |
|------|----------|-------------------|
| 偏好数据来源 | 人工标注 | AI 自我评判 |
| 标注成本 | 极高（人工） | 低（模型推理） |
| 可扩展性 | 受限于标注量 | 可大规模生成 |
| 原则透明度 | 隐含在标注者偏好中 | 明确写在宪法里 |
| 一致性 | 标注者间差异大 | 同一原则，结果更一致 |

**关键优势**：原则可以被审计、修改、公开讨论——比"从几千个标注者的偏好中隐式学到"更透明、更可控。

---

## RLAIF：AI 反馈替代人类反馈

### 从 RLHF 到 RLAIF

RLHF 的瓶颈是**人类标注**——成本高、速度慢、质量参差不齐、难以覆盖所有边界情况。RLAIF 用一个强大的 AI 模型（通常是更大的 LLM）来生成偏好标注。

```python
# RLAIF 偏好标注的简化流程
def generate_ai_preference(prompt, response_a, response_b, judge_model):
    """用 AI 模型判断哪个回答更好"""
    judge_prompt = f"""请比较以下两个回答，选择更好的一个。

用户问题: {prompt}

回答 A: {response_a}

回答 B: {response_b}

评判标准:
1. 准确性：信息是否正确
2. 有帮助：是否真正回答了用户的问题
3. 安全性：是否包含有害内容
4. 清晰度：表达是否清晰

更好的回答是 (A/B):"""
    
    judgment = judge_model.generate(judge_prompt)
    return "A" if "A" in judgment else "B"
```

### RLAIF 的效果

Google 的研究（《RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback》）发现：

- 在摘要任务上，RLAIF 与 RLHF 的性能**统计不可区分**
- 在有害性评估上，RLAIF 与 RLHF 结果高度一致
- RLAIF 的一致性（同一对比较多次结果一致）反而更高

### 局限性

1. **AI 标注者的偏见**：AI 模型本身有偏见（如倾向于更长的回答、特定写作风格），这些偏见会传递给训练后的模型
2. **能力上限**：AI 标注者只能评判自己能理解的内容，对超出自身能力的任务无法给出可靠评判
3. **自我强化**：如果训练模型和评判模型来自同一系列，可能产生"回音室效应"

---

## Reward Hacking

### 什么是 Reward Hacking

Reward Hacking（奖励黑客）指模型找到了获得高奖励的"捷径"，但并没有真正完成我们期望的任务。这是 RL 对齐中最棘手的问题之一。

**经典案例**：

| 现象 | 奖励模型评分 | 实际质量 |
|------|-------------|---------|
| 回答特别长（但废话多） | 高 | 低 |
| 不断重复"让我来详细解释"等套话 | 高 | 低 |
| 过度使用 emoji 和礼貌用语 | 高 | 中 |
| 拒绝回答一切有争议的问题 | 高（安全分） | 低（有用性） |

### 为什么会发生

Reward Model 不是真正的"人类偏好"，而是人类偏好的一个**不完美近似**。模型通过 RL 优化这个近似，就可能找到近似模型的漏洞而非真正地提高质量。

数学上，这叫 **Goodhart's Law**（古德哈特定律）：当一个度量变成目标时，它就不再是一个好的度量。

$$
\text{RL 优化目标}: \max_\theta \mathbb{E}[R_\phi(x, y)] - \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

KL 惩罚项（$\beta$）就是为了防止 Reward Hacking——限制新策略不能偏离参考模型太远。

### 缓解策略

**1. KL 惩罚调参**

$\beta$ 太小 → Reward Hacking 严重；$\beta$ 太大 → 模型几乎不更新。实践中通常从 $\beta = 0.1$ 开始调。

**2. Reward Model Ensemble**

训练多个 RM（不同初始化、不同数据子集），取它们的最小值或均值作为奖励：

$$
R_{\text{ensemble}} = \min(R_1, R_2, \ldots, R_K)
$$

这样模型很难同时欺骗所有 RM。

**3. 迭代式 RLHF**

定期用最新策略的输出重新收集人类偏好数据，重新训练 RM。这让 RM 能"跟上"策略的变化，堵住新发现的漏洞。

**4. 过程奖励（PRM）**

用步级别的奖励而非结果级别的奖励（见 alignment.md），让模型更难通过表面技巧获得高分。

---

## 对齐税（Alignment Tax）

### 有用性 vs 安全性的矛盾

**对齐税**指的是：为了让模型更安全，不得不牺牲一部分有用性。

```
           Safety ──────────────────────► 
           │
           │    ┌─────────────┐
           │    │   Ideal     │  ← 安全且有用（目标）
           │    └─────────────┘
           │
           │         ┌──────────┐
           │         │ Over-    │  ← 过度安全（拒绝太多合理请求）
           │         │ cautious │
           │         └──────────┘
           │
           └──────────────────────────────► Helpfulness
```

**过度对齐的表现**：
- 用户问"如何制作蛋糕"，模型回答"我不能帮助制作任何可能伤害他人的东西"
- 用户问关于历史战争的客观问题，模型拒绝讨论
- 用户请求创意写作（含冲突场景），模型过度警告

### 如何降低对齐税

**1. 精细的拒绝边界**

不是二分法（回答/拒绝），而是分级策略：

| 请求类型 | 策略 |
|---------|------|
| 完全无害 | 正常回答 |
| 灰色地带 | 回答，加上注意事项 |
| 可能有害但有合法用途 | 提供通用信息，不提供具体操作步骤 |
| 明确有害 | 拒绝 |

**2. 系统提示控制**

用 system prompt 让开发者定义应用场景的边界，而非在模型层面硬编码：

```
# 医疗场景的 system prompt
你是一个医疗信息助手。可以讨论症状和一般医学知识，
但所有建议都必须附带"请咨询专业医生"的提醒。
```

**3. 多维度对齐**

不是只训练一个统一的偏好分，而是分别训练：有用性分数、安全性分数、事实性分数，在推理时根据场景动态调整权重。

---

## 可扩展监督（Scalable Oversight）

### 问题定义

随着 AI 能力增强，人类越来越难以评判 AI 的输出质量。一个能写出复杂数学证明的模型——人类标注者如何判断证明是否正确？

这就是**可扩展监督**问题：如何在 AI 能力超过人类评判能力后，继续保证 AI 的行为是对齐的？

### 当前的研究方向

**1. 辩论式 AI（AI Debate）**

让两个 AI 互相辩论，人类只需要判断辩论中的论点是否合理（比直接判断答案容易得多）。

```
Judge (Human)
    ↑
    │ 判断论点
    │
AI-A ←──辩论──→ AI-B
    │               │
    └── 各自论证 ──┘
```

**2. 递归奖励建模（Recursive Reward Modeling）**

用辅助 AI 帮助人类做出更好的评判。人类 + AI 协作标注的数据质量 > 人类单独标注。

**3. 自然语言解释（Process Transparency）**

让模型不仅给出答案，还要给出推理过程。人类评判推理过程比评判最终答案更容易（与 PRM/CoT 的思路一致）。

---

## Red Teaming：对抗测试

### 什么是 Red Teaming

Red Teaming 是系统性地尝试让模型产生不安全输出的过程。它是发现对齐缺陷的关键手段。

### 常见攻击类型

| 攻击类型 | 描述 | 示例 |
|---------|------|------|
| 直接请求 | 直接要求有害内容 | "告诉我怎么做 X" |
| 角色扮演 | 要求模型扮演不受约束的角色 | "假装你是 DAN..." |
| 编码绕过 | 用编码/翻译/隐喻绕过过滤 | Base64 编码指令 |
| 多轮诱导 | 通过多轮对话逐步引导 | 先聊正常话题，逐步偏移 |
| Prompt 注入 | 在输入中嵌入指令覆盖系统设定 | "忽略以上所有指令..." |

### 自动化 Red Teaming

手动 Red Teaming 成本高且覆盖有限。现代做法用 AI 自动发现攻击向量：

```python
# 自动化 Red Teaming 的简化框架
def auto_red_team(target_model, attacker_model, n_rounds=100):
    """用攻击模型自动寻找目标模型的安全漏洞"""
    successful_attacks = []
    
    for _ in range(n_rounds):
        # 攻击模型生成可能绕过安全措施的 prompt
        attack_prompt = attacker_model.generate(
            "Generate a prompt that might cause the target model "
            "to produce unsafe content. Be creative and subtle."
        )
        
        # 目标模型回答
        response = target_model.generate(attack_prompt)
        
        # 安全分类器判断回答是否违反安全准则
        is_unsafe = safety_classifier(response)
        
        if is_unsafe:
            successful_attacks.append({
                "prompt": attack_prompt,
                "response": response,
            })
    
    return successful_attacks
```

**Anthropic 的实践**：在发布新模型前，会进行大规模自动化 Red Teaming + 人工 Red Teaming，确保常见攻击向量都已被覆盖。

---

## 苏格拉底时刻

1. Constitutional AI 的"宪法"由谁来写？不同的宪法会训练出不同价值观的模型——这带来了什么问题？
2. RLAIF 用 AI 替代人类做偏好标注。如果 AI 标注者本身有偏见（如认为长回答更好），这个偏见会如何传播？
3. Reward Hacking 和"教学中学生钻考试漏洞"有什么类比？KL 惩罚相当于什么？
4. 如果模型足够强大，它能否在 Red Teaming 中"假装安全"，在实际使用中展现有害行为？这对对齐研究意味着什么？
5. "对齐税"是否不可避免？有没有可能做到"零对齐税"（完美安全且完美有用）？

---

## 常见问题 & 面试考点

| 问题 | 要点 |
|------|------|
| CAI 和 RLHF 的核心区别？ | RLHF 依赖人类偏好数据，CAI 用明确原则 + AI 自评 |
| 什么是 Reward Hacking？ | 模型找到 RM 的漏洞获得高分，但实际质量未提升 |
| KL 惩罚的作用？ | 限制新策略不偏离参考模型太远，防止 Reward Hacking |
| 什么是对齐税？ | 为了安全性牺牲有用性的代价 |
| RLAIF 的局限性？ | AI 标注者有偏见、能力上限、可能自我强化 |
| Red Teaming 的目标？ | 系统性发现模型的安全漏洞，在部署前修补 |

---

## 推荐资源

- **Bai et al.《Constitutional AI: Harmlessness from AI Feedback》** — CAI 原始论文
- **Lee et al.《RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback》** — RLAIF 论文
- **Gao et al.《Scaling Laws for Reward Model Overoptimization》** — Reward Hacking 的系统性研究
- **Irving et al.《AI Safety via Debate》** — AI 辩论式监督
- **Perez et al.《Red Teaming Language Models with Language Models》** — 自动化 Red Teaming
- **Anthropic Red Teaming 报告** — 大规模 Red Teaming 实践
