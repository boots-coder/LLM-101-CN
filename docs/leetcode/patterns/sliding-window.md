---
title: 滑动窗口 · LeetCode Pattern
description: 维护一个区间，单向扩张和收缩，O(n) 解决子串/子数组类问题
topics: [leetcode, sliding-window, pattern]
prereqs: [leetcode/]
---

# 滑动窗口 · Sliding Window

> **一句话骨架**：右指针扩张吃元素，左指针在「窗口违规」时收缩——左右指针都只走 n 步，所以 O(n)。

## 何时用

- "连续子串 / 子数组"且需要某种约束（无重复 / 字符分布 / 长度固定 / 覆盖目标）
- 暴力枚举所有区间是 O(n²) 或更高，但相邻区间状态可**增量更新**
- 字符串/数组上的"覆盖"、"恰好包含"、"最多 K 种"等问句

## 通用模板

```python
left = 0
window = {}
best = 0
for right, ch in enumerate(s):
    # 1. 把 ch 加入 window
    window[ch] = window.get(ch, 0) + 1
    # 2. 当窗口违规时收缩
    while violates(window):
        window[s[left]] -= 1
        if window[s[left]] == 0: del window[s[left]]
        left += 1
    # 3. 更新答案
    best = max(best, right - left + 1)
return best
```

## 关卡 1 · 无重复字符的最长子串

::: info 学习目标
理解"位置法"与"计数法"两种滑窗写法的差别——前者一次跳跃 O(n)，后者更通用可改 K-种限制题。
:::

<LcLesson problem-id="longest-substring-without-repeating" />

## 关卡 2 · 找到字符串中所有字母异位词

::: info 学习目标
学会**定长滑窗**：窗口大小恒为 len(p)，进新出旧同步更新两张计数表，相等时记录起点。这是滑窗里最规整的一种写法。
:::

<LcLesson problem-id="find-anagrams" />

## 关卡 3 · 最小覆盖子串

::: info 学习目标
掌握**变长滑窗**的精髓：用 formed/required 两个量把"是否覆盖"压成 O(1) 判断，配合"扩张外循环 + 收缩内循环"双层结构。这是滑窗 hard 题的统一范式。
:::

<LcLesson problem-id="minimum-window-substring" />

## 同 pattern 索引（hot100 同类题）

| 题号 | 题名 | 难度 | 关键变形 |
|------|------|------|----------|
| 239 | 滑动窗口最大值 | 困难 | 单调递减双端队列维护窗口最大值 |
| 567 | 字符串的排列 | 中等 | 与 438 双胞胎，固定长度滑窗判窗口字符分布 == p |
| 209 | 长度最小的子数组 | 中等 | 数值版变长滑窗，sum ≥ target 时收缩 left |
| 424 | 替换后的最长重复字符 | 中等 | 维护"窗口长度 - 最高频字符次数 ≤ k"作为不变式 |

## 面试常踩

1. **变长 vs 定长**——开题先判：窗口长度由谁决定？长度未知就是变长，要"扩张外 + 收缩内"双循环；长度等于某给定值就是定长，单循环 + `if right >= m: 弹左`。
2. **位置法 vs 计数法**——位置法（last[ch] = right）只解"无重复"类，常数小；计数法可解"最多 K 种"等通用变形，常数稍大。看题选写法。
3. **`have == need` 的常数代价**——438 用定长 26 数组比较是 O(Σ)，Σ 视为常数；如果换成 dict 比较，常数会显著放大。这是数据结构选择影响速度的典型例子。
4. **formed 的 invariant 维护**——只有"由不达标变为达标"那一步 +1，"由达标变为不达标"那一步 -1。多写几次最小覆盖子串就能形成肌肉记忆。
