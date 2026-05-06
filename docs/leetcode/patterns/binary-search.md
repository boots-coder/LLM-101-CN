---
title: 二分查找 · LeetCode Pattern
description: 单调即可二分，关键是边界判定与 mid 是否纳入下一轮
topics: [leetcode, binary-search, pattern]
prereqs: [leetcode/]
---

# 二分查找 · Binary Search

> **一句话骨架**：单调即可二分，关键是边界判定与 mid 是否纳入下一轮。

## 何时用

- 数组**已排序**或目标具有**单调性**（答案越大越容易满足 / 越小越容易满足）
- 在「找下标 / 找峰值 / 找第 K 个」这类需要 O(log n) 的场景
- 可以"猜答案 + 验证"的优化问题（如最小化最大值）

## 通用模板

```python
left, right = 0, len(nums) - 1   # 闭区间 [left, right]
while left <= right:
    mid = (left + right) // 2
    if check(mid):
        right = mid - 1          # 答案在左半（含 mid 不满足时也走这分支）
    else:
        left = mid + 1
return left  # 第一个满足 check 的下标
```

::: warning 边界三原则
1. 区间是「闭」还是「左闭右开」要从头到尾保持一致；
2. `left = mid + 1` / `right = mid - 1` 必须保证**搜索区间真的缩小**，否则死循环；
3. 返回 `left` 还是 `right` 取决于「找第一个 / 最后一个满足条件」。
:::

## 关卡 1 · 搜索旋转排序数组

::: info 学习目标
掌握"切开后至少有一半有序"这个 invariant——把不严格单调的数组也压回 O(log n)。这是二分思维从"前提单调"到"局部单调"的跃升。
:::

<LcLesson problem-id="search-rotated-array" />

## 关卡 2 · 寻找峰值

::: info 学习目标
理解"二分不一定要排序"——只要能保证"砍掉的那半一定不是答案"就能二分。配合 `right = mid` 写法时循环条件必须用 `<` 防死循环，是写二分的常见陷阱。
:::

<LcLesson problem-id="find-peak-element" />

## 关卡 3 · 寻找两个正序数组的中位数

::: info 学习目标
学会把"求中位数"重新建模成"切两刀 + 满足两不等式"的二分问题。这是 hard 二分题的通用思路：当朴素方法是 O(m+n) 时，先想"能不能转成切点二分"。
:::

<LcLesson problem-id="median-two-sorted-arrays" />

## 同 pattern 索引（hot100 同类题）

| 题号 | 题名 | 难度 | 关键变形 |
|------|------|------|----------|
| 35 | 搜索插入位置 | 简单 | 标准二分模板，找第一个 ≥ target 的下标 |
| 34 | 在排序数组中查找元素的第一个和最后一个位置 | 中等 | 两次二分：lower_bound 和 upper_bound |
| 153 | 寻找旋转排序数组中的最小值 | 中等 | 与 33 同骨架，只是不找 target 而是找拐点 |
| 74 | 搜索二维矩阵 | 中等 | 把 m×n 矩阵看成长度 m·n 的有序数组，下标 ↔ (行,列) 互转 |
| 240 | 搜索二维矩阵 II | 中等 | 从右上角出发的"阶梯二分"，每步排除一行或一列 |

## 面试常踩

1. **`right = mid` 必配 `while left < right`**——任何写 `right = mid`（不严格收缩）的版本，循环条件都不能是 `<=`，否则当 left==right==mid 时陷入死循环。
2. **闭区间 vs 左闭右开**要选一种用到底——混用最容易写出"差一格"的 bug。建议入门期固定用闭区间 `[left, right]` + `while left <= right` + `right=mid-1` / `left=mid+1`。
3. **mid 计算**用 `left + (right - left) // 2`——Python 不溢出但跨语言通用；写这个版本是面试加分项。
4. **链式比较 `a < b < c`**——Python 特性等价于 `a < b and b < c`，且 b 只求值一次。33 题里 `nums[left] <= target < nums[mid]` 写得地道。
