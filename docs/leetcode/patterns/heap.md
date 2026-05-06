---
title: 堆 · LeetCode Pattern
description: Top-K、流式中位数、合并 K 个有序
topics: [leetcode, heap, pattern]
prereqs: [leetcode/]
---

# 堆 · Heap / Priority Queue

> **一句话骨架**：堆是「能在 O(log n) 内拿到当前最值」的结构——Top-K 用「大小为 K 的最小堆」，中位数用「双堆对顶」，合并 K 路用「k 元素的最小堆」。

## Python 速查

`heapq` 默认是**最小堆**：

```python
import heapq
h = []
heapq.heappush(h, x)        # 推入
small = heapq.heappop(h)    # 弹出最小
heapq.heappushpop(h, x)     # 先 push 再 pop，O(log n) 一次完成
heapq.nlargest(k, nums)     # 直接拿前 K 大
heapq.heapify(arr)          # 把任意 list 原地变成堆，O(n)
```

要做**最大堆**？把值取负后再入堆：`heappush(h, -x)` / `-heappop(h)`。

## 关卡 1 · 数组中的第 K 个最大元素

::: info 学习目标
真正理解「找第 K 大用大小为 K 的最小堆」这一反直觉模板——堆顶就是答案；同时与排序法 / quickselect 三种解法做时间复杂度对比。
:::

<LcLesson problem-id="kth-largest-element" />

## 关卡 2 · 前 K 个高频元素

::: info 学习目标
学会把「频次」作为堆的比较 key——元组 `(freq, num)` 入堆是 Python 的标准技巧；并对照桶排序解法理解 O(n log k) 与 O(n) 的取舍。
:::

<LcLesson problem-id="top-k-frequent" />

## 关卡 3 · 数据流的中位数

::: info 学习目标
掌握「双堆对顶」的不变量与 addNum 的中转写法——理解为什么不能直接 push 进对应一侧，以及 Python 没有原生大根堆时如何用「值取负」绕开。
:::

<LcLesson problem-id="find-median-data-stream" />

## 同 pattern 索引（hot100 同类题）

| 题号 | 题名 | 难度 | 关键变形 |
|------|------|------|----------|
| 23 | 合并 K 个升序链表 | 困难 | k 元素的最小堆，每次弹出最小并把它的 next 入堆 |
| 239 | 滑动窗口最大值 | 困难 | 主流解法是单调队列（deque），堆解法需「延迟删除」过期元素 |
| 480 | 滑动窗口的中位数 | 困难 | 双堆 + 延迟删除；或 SortedList——是 295 的滑窗版 |

## 面试常踩

1. **Top-K 的方向反着用**——找「前 K 大」用最小堆、找「前 K 小」用最大堆。背反了会写出错误且丑陋的代码。
2. **Python 没有原生大根堆**——只能值取负或用 `(-key, value)` 元组；用完别忘了取负还原。
3. **元组比较时 key 要放第一位**——`(freq, num)` 比较先看 freq；如果反过来 `(num, freq)` 就成了按 num 排序。
4. **`heappush + heappop` vs `heappushpop`**——后者是单次 sift，常数小一倍。Top-K 模板里非常常用。
5. **`heapify` 是 O(n)，但循环里反复 heapify 会变成 O(n²)**——增量插入用 heappush，一次性建堆用 heapify。
