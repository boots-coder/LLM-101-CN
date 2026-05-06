---
title: 一维 DP · LeetCode Pattern
description: 线性状态转移，dp[i] 由 dp[i-1] 等推出
topics: [leetcode, dp-1d, pattern]
prereqs: [leetcode/]
---

# 一维 DP · DP 1D

> **一句话骨架**：定义 `dp[i] =`「以 i 结尾 / 到达 i 时」的某个最优值，写出**转移方程**与**初始值**——能不能滚动数组优化空间是面试加分项。

## 解题四步

1. **状态定义**：`dp[i]` 表示「以 i 结尾」/「前 i 个」的什么？
2. **转移方程**：`dp[i] = f(dp[i-1], dp[i-2], ...)` 怎么推？
3. **初始值与边界**：`dp[0]`、`dp[1]` 如何初始化，循环从哪开始？
4. **空间优化**：只依赖前 1-2 个状态时用变量代替数组。

## 通用模板

```python
# 经典「只依赖前两个」的滚动写法
prev2, prev1 = base0, base1
for i in range(start, n):
    cur = transition(prev1, prev2, nums[i])  # 转移方程
    prev2, prev1 = prev1, cur  # 同行赋值,避免覆盖
return prev1
```

## 关卡 1 · 爬楼梯

::: info 学习目标
掌握「加法原理 → 二阶递推 → 斐波那契」的链路，理解为什么 `dp[0]=1`（而非 0）才让转移自洽。

:::

<LcLesson problem-id="climbing-stairs" />

## 关卡 2 · 打家劫舍

::: info 学习目标
学会"相邻互斥型"DP：`dp[i] = max(dp[i-1], dp[i-2] + nums[i])` 的两支转移分别对应"不偷 / 偷"——这套模板可推广到环形（213）、二叉树版（337）。
:::

<LcLesson problem-id="house-robber" />

## 关卡 3 · 最长递增子序列

::: info 学习目标
区分朴素 O(n²) 的「以 i 结尾」状态与 O(n log n) 贪心 + 二分中 `tails[k]` 的"长度档位末尾最小值"语义；理清 `bisect_left` vs `bisect_right` 在严格/非严格递增时的取舍。
:::

<LcLesson problem-id="longest-increasing-subsequence" />

## 关卡 4 · 单词拆分

::: info 学习目标
建立「前缀 DP」：`dp[i]` = `s[:i]` 是否可拆。学会用 set 把字典查询压到 O(1)，并用「最长单词长度」剪枝内层循环。
:::

<LcLesson problem-id="word-break" />

## 同 pattern 索引（hot100 同类题）

| 题号 | 题名 | 难度 | 关键变形 |
|------|------|------|----------|
| 279 | 完全平方数 | 中等 | `dp[i] = min(dp[i-k²]) + 1`，对每个 i 枚举可减的平方数 |
| 322 | 零钱兑换 | 中等 | `dp[i] = min(dp[i-c]) + 1`，无解返回 -1（注意初始化为 inf） |
| 118 | 杨辉三角 | 简单 | 一维生成第 i 行 = 上一行错位相加 |
| 152 | 乘积最大子数组 | 中等 | 双 dp：同时维护以 i 结尾的 max 与 min（负负得正） |
| 32 | 最长有效括号 | 困难 | 栈记录左括号下标 / 或 dp[i] 表示以 i 结尾的有效长度 |
| 121 | 买卖股票的最佳时机 | 简单 | 边扫边维护 min_price 与 max_profit，本质是 dp |
| 55 | 跳跃游戏 | 中等 | 贪心维护 max_reach，可视作 dp[i] 是否可达的常数空间版 |
| 45 | 跳跃游戏 II | 中等 | 贪心 BFS 思想分层扩展，O(n) 跳跃次数 |

## 面试常踩

1. **`dp[0]` 选 0 还是 1**——取决于状态定义。爬楼梯里 dp[0]=1（"空走法"算一种）才能让 `dp[2]=dp[1]+dp[0]=2` 自洽。
2. **滚动赋值方向**——Python 同行赋值 `prev2, prev1 = prev1, cur` 是右侧元组先求值再解包，避免顺序敏感。
3. **`bisect_left` vs `bisect_right`**——严格递增 LIS 用 `bisect_left`（等值时替换），非严格用 `bisect_right`（等值时延长）。这是一字之差导致 WA 的经典坑。
4. **DP 不是贪心**——单词拆分、打家劫舍等都不能"每次选最大/最长"，必须枚举切割点 / 决策点。
