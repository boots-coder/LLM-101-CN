---
title: 回溯 · LeetCode Pattern
description: 选择 → 递归 → 撤销，画决策树
topics: [leetcode, backtracking, pattern]
prereqs: [leetcode/]
---

# 回溯 · Backtracking

> **一句话骨架**：在决策树上做 DFS——「做选择 → 递归到下一层 → 撤销选择」三段式雷打不动。回溯的难点不是写代码，而是画清楚「这一层在选什么」。

## 通用模板

```python
def backtrack(path, choices):
    if 满足结束条件:
        res.append(path[:])     # 拷贝！否则后续 pop 会破坏快照
        return
    for c in choices:
        if 不合法: continue
        path.append(c)          # 做选择
        backtrack(path, 新的选择列表)
        path.pop()              # 撤销
```

## 关卡 1 · 全排列

::: info 学习目标
掌握「used 数组 + path[:] 拷贝」两件事；分清排列（used）vs 组合（start）的本质差异。
:::

<LcLesson problem-id="permutations" />

## 关卡 2 · 组合总和

::: info 学习目标
体会「`start` 参数防顺序重复」与「下一层从 i 开始 → 同元素可重选」两条招牌细节。
:::

<LcLesson problem-id="combination-sum" />

## 关卡 3 · N 皇后

::: info 学习目标
学会用三个 set（cols / row-col / row+col）把冲突检查从 O(n) 压到 O(1)。
:::

<LcLesson problem-id="n-queens" />

## 关卡 4 · 单词搜索

::: info 学习目标
掌握网格回溯三段式：「占位 board[i][j]='#' → 四方向递归 → 出 dfs 还原」。
:::

<LcLesson problem-id="word-search" />

## 同 pattern 索引（hot100 同类题）

| 题号 | 题名 | 难度 | 关键变形 |
|------|------|------|----------|
| 78 | 子集 | 中等 | 每个元素「选 / 不选」两枝；或固定 start 累加 path（叶子无终止条件） |
| 17 | 电话号码的字母组合 | 中等 | 每层选项是「当前数字对应的若干字母」，而非数组下标 |
| 22 | 括号生成 | 中等 | 用「left 还能放几个、right 还能放几个」剪枝，无需事后校验 |
| 131 | 分割回文串 | 中等 | 每层枚举切割点；递归 (s[i:]) 处理剩余串 |

## 面试常踩

- **`path[:]` 拷贝**：忘记拷贝就把同一引用塞进 res，最终 res 里全是空 list。等价写法 `list(path)`。
- **排列用 `used`、组合用 `start`**：搞错了不是写不出来——是会少解或多解（[1,2] 与 [2,1] 算一个还是两个）。
- **进 dfs 改、出 dfs 还原**——网格回溯（79、212）的占位法必须配对，少一边就污染棋盘。
- **N 皇后的对角线身份证**：主对角线 row-col 相等，副对角线 row+col 相等——背下来。
