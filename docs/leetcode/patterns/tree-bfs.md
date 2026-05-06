---
title: 树 BFS · LeetCode Pattern
description: 队列 + 按层处理
topics: [leetcode, tree-bfs, pattern]
prereqs: [leetcode/]
---

# 树 BFS · Tree BFS

> **一句话骨架**：BFS 用 `collections.deque` 当队列，进入循环时**先记下当前层的大小**，把这一层全部弹出后再处理下一层。这「层级感」就是 BFS 的灵魂。

## 通用模板

```python
from collections import deque
q = deque([root])
while q:
    size = len(q)            # 关键：当前层节点数
    level = []
    for _ in range(size):
        node = q.popleft()
        level.append(node.val)
        if node.left:  q.append(node.left)
        if node.right: q.append(node.right)
    res.append(level)
```

::: tip 为什么用 deque 不用 list
`list.pop(0)` 是 O(n)（要左移所有元素）；`deque.popleft()` 是 O(1)。BFS 涉及大量「从前端取」，用错容器直接 O(n²)。
:::

## 关卡 1 · 二叉树的层序遍历

::: info 学习目标
内化「先记 size 再 for size 次」的层级感模板；能讲清 deque vs list 的 O(1) vs O(n)。
:::

<LcLesson problem-id="level-order-traversal" />

## 关卡 2 · 二叉树的右视图

::: info 学习目标
能用 BFS 取每层最后一个 + DFS 先右后左两种思路解决，并说清「右视图 ≠ 最右链」。
:::

<LcLesson problem-id="right-side-view" />

## 关卡 3 · 二叉树展开为链表

::: info 学习目标
掌握 O(1) 额外空间的「左子树最右节点接右子树」技巧；理解 BFS / DFS / 迭代为何在本题殊途同归。
:::

<LcLesson problem-id="flatten-tree-to-list" />

## 同 pattern 索引（hot100 同类题）

| 题号 | 题名 | 难度 | 关键变形 |
|------|------|------|----------|
| 104 | 二叉树的最大深度 | 简单 | 既可 DFS 归纳，也可 BFS 数层数——双解题 |
| 222 | 完全二叉树的节点个数 | 中等 | 不是裸 BFS——利用「完全二叉树」性质 + 二分树高，O(log² n) |
| 116 | 填充每个节点的下一个右侧节点指针 | 中等 | BFS 按层连 next；进阶版 O(1) 空间用「父层的 next 指引子层」 |

## 面试常踩

- **`size = len(q)` 漏写**：队列里同时混着当前层和下一层，会把分层弄成扁平输出。
- **None 进队**：忘写 `if node.left:` 判断会让 None 入队，下一轮 `node.val` 抛 AttributeError。
- **入队顺序决定层内方向**：先左后右 → 层内从左到右；想要「左视图/右视图」改判断条件 `i == 0` 或 `i == size - 1` 即可。
- **BFS vs DFS 空间偏好**：BFS 是 O(w)（最宽层），DFS 是 O(h)（树高）；瘦高树 DFS 省空间、矮宽树 DFS 也省（满二叉树最后一层 ≈ n/2）。
