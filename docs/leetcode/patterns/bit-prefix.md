---
title: 位运算与前缀和 · LeetCode Pattern
description: 异或消除自身、累计和差分
topics: [leetcode, bit-prefix, pattern]
prereqs: [leetcode/]
---

# 位运算与前缀和 · Bit / Prefix Sum

> **一句话骨架**：① 位运算抓两个本质——异或「消除成对元素」、`x & (x-1)` 「消掉最低位 1」；② 前缀和把「区间和 = 前缀差」配合**哈希表**实现 O(n)「子数组和等于 K」。

## 两类必背骨架

```python
# 1. 前缀和 + 哈希：subarray sum = k
prefix = 0; cnt = {0: 1}; ans = 0
for x in nums:
    prefix += x
    ans += cnt.get(prefix - k, 0)   # 之前出现过几次 prefix-k，就有几个子数组
    cnt[prefix] = cnt.get(prefix, 0) + 1

# 2. XOR 消同类
ans = 0
for x in nums: ans ^= x
return ans   # 出现两次的全消掉，剩下出现一次的

# 3. 前缀积 + 后缀积（不允许除法时的产品代替方案）
n = len(nums); ans = [1]*n; left = 1
for i in range(n): ans[i] = left; left *= nums[i]
right = 1
for i in range(n-1, -1, -1): ans[i] *= right; right *= nums[i]
```

## 关卡 1 · 只出现一次的数字

::: info 学习目标
背下 XOR 三性质（`a^a=0, a^0=a, 交换律`）；理解题目硬约束 O(n) 时间 + O(1) 空间是怎么逼出 XOR 的。
:::

<LcLesson problem-id="single-number" />

## 关卡 2 · 和为 K 的子数组

::: info 学习目标
掌握「前缀和 + 哈希」模板；记住 `cnt = {0: 1}` 处理整段和等于 k 的边界。
:::

<LcLesson problem-id="subarray-sum-equals-k" />

## 关卡 3 · 除自身以外数组的乘积

::: info 学习目标
理解「不能用除法」的现实约束；用「前缀积 + 后缀积」两遍扫描共用一份 answer 实现 O(1) 额外空间。
:::

<LcLesson problem-id="product-except-self" />

## 同 pattern 索引（hot100 同类题）

| 题号 | 题名 | 难度 | 关键变形 |
|------|------|------|----------|
| 169 | 多数元素 | 简单 | Boyer-Moore 摩尔投票：候选 + 计数对消（与 XOR「成对消除」同源思想） |
| 75 | 颜色分类 | 中等 | 三指针（low / mid / high）一次扫——前缀和的「三段划分」变体 |
| 287 | 寻找重复数 | 中等 | 视为「下标 → 值」的隐式链表，Floyd 判环找重复元素 |
| 31 | 下一个排列 | 中等 | 从右往左找第一个降点 + 右段反转——和前/后缀扫的双向思想类似 |

## 面试常踩

- **XOR 起步用 `0`**：0 是 XOR 的单位元（a^0=a），从其它值起步会污染答案。
- **`cnt = {0: 1}`** 的初始化不能省——否则「整段和恰好 = k」漏解。
- **`prefix += x; ans += ...; cnt[prefix] = ...`**：先查后存的纪律和 two-sum 一脉相承。k=0 时尤其暴露这种 bug。
- **238 不能用除法**——nums 含 0 时除零不可避免；必须用前缀积/后缀积两遍扫描。
- **238 O(1) 空间**：第一遍把左积写进 answer、第二遍滚动变量 right 乘进去。这是「输出不算额外空间」的经典空间优化。
