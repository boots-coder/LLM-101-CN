---
title: 哈希 · LeetCode Pattern
description: 空间换时间，用哈希表把"判断是否见过 / 找配对"压到 O(1)
topics: [leetcode, hashmap, hash-table, pattern]
prereqs: [leetcode/]
---

# 哈希表 · Hash Map

> **一句话骨架**：扫一遍数组，每个元素先查"另一半在不在哈希表里"，没在就把自己存进去。先查后存，永远不会自配对。

## 何时用

- 需要 O(1) 查询"是否见过"或"配对值在哪"
- 需要按 key 分组（字母异位词 → sort 后的字符串作 key）
- 题目要求返回**索引**（用 dict.value 存索引），而非值本身（值用 set 即可）

## 关卡 1 · 两数之和（Two Sum）

::: info 学习目标
通过这一关，你应该能从空白纸默写出 8 行解法——而不需要"背"，是从结构里**推**出来的。
:::

<LcLesson problem-id="two-sum" />

## 同 pattern 索引

下面的题用同一骨架就能解，时间充裕时回过头扫一遍：

| 题号 | 题名 | 难度 | 关键变形 |
|------|------|------|----------|
| 49 | 字母异位词分组 | 中等 | dict 的 key 用 `''.join(sorted(s))` 或字符计数 tuple |
| 128 | 最长连续序列 | 中等 | 先 set 化，从"序列起点"（n-1 不在集合）开始向上数 |
| 41 | 缺失的第一个正数 | 困难 | 用数组本身做哈希（原地交换 nums[i] 到 i+1 位置） |

## 面试常踩

1. **`{}` 是 dict 不是 set**——空集合写 `set()`。
2. **`seen[num] = i` 必须在判断之后**——否则可能自配对。
3. **`dict.keys()` vs `dict`**：用 `key in d` 而非 `key in d.keys()`，前者更地道（虽然都是 O(1)）。
4. **不要为了去重就 `list(set(arr))`**——会丢失顺序；如要"保序去重"用 `dict.fromkeys(arr)`。
