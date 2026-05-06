---
title: 双指针 · LeetCode Pattern
description: 对撞或同向收缩，把 O(n²) 的两层循环压到 O(n)
topics: [leetcode, two-pointer, pattern]
prereqs: [leetcode/]
---

# 双指针 · Two Pointers

> **一句话骨架**：在**有序**或**对撞**的场景下，左右指针向中间收缩；移动的关键不是"轮流走"，而是**根据当前情况选择该移动的一侧**——这一选择决定了能否把 O(n²) 压到 O(n)。

## 何时用

- 数组**有序** → 对撞双指针（盛水容器、三数之和）
- 字符串/数组中维护一段"区间"且能单调收缩 → 同向双指针（移动零、回文判断）
- 两个串/链表对齐扫描（合并两个有序链表、相交链表）

## 通用模板

```python
left, right = 0, len(arr) - 1
while left < right:
    # 1. 计算当前状态
    cur = f(arr[left], arr[right])
    # 2. 更新最优
    best = max(best, cur)
    # 3. 决定移动哪一侧（核心难点）
    if 应该缩左:
        left += 1
    else:
        right -= 1
```

## 关卡 1 · 盛最多水的容器

<LcLesson problem-id="container-with-most-water" />

## 关卡 2 · 三数之和

> 排序 + 固定 i + 内层对撞双指针。**三处去重**是细节决定成败的地方——很多人栽在"找到解后只移一格"上。

<LcLesson problem-id="three-sum" />

## 同 pattern 索引

| 题号 | 题名 | 难度 | 关键变形 |
|------|------|------|----------|
| 42 | 接雨水 | 困难 | 双指针维护 left_max / right_max，移较矮一侧 |
| 283 | 移动零 | 简单 | 同向双指针，slow 指向"下一个非零落脚点" |
| 88 | 合并两个有序数组 | 简单 | 倒序双指针避免覆盖 |

## 面试常踩

1. **越界检查放在 while 条件里**——例如三数之和内层去重 `while left < right and nums[left] == nums[left+1]`，少了 `left < right` 会越界。
2. **找到解后必须跳过相同元素**——只移一格会重复。
3. **对撞双指针的"贪心移动"要会讲清楚**：移动较高侧时，新的 min 不会增、宽度还在减，面积只会更糟。这是面试时区分"会写"和"懂为什么"的分水岭。
