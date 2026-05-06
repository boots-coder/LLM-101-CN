---
title: 图搜索 · LeetCode Pattern
description: DFS / BFS / 拓扑排序 / 并查集
topics: [leetcode, graph, pattern]
prereqs: [leetcode/]
---

# 图搜索 · Graph Search

> **一句话骨架**：① 网格题 → 「四方向 DFS / BFS + visited」；② 依赖题 → 「Kahn 拓扑排序」；③ 连通分量题 → 「并查集」。识别出题型，模板套上去就能 80% 解决。

## 三个最常用骨架

```python
# 1. 网格 BFS（多源也是同一个写法）
from collections import deque
q = deque([起点])
while q:
    x, y = q.popleft()
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        nx, ny = x+dx, y+dy
        if 合法 and 未访问:
            visited[nx][ny] = True
            q.append((nx, ny))

# 2. Kahn 拓扑排序
indeg = [0]*n
g = [[] for _ in range(n)]
for u, v in edges: g[u].append(v); indeg[v] += 1
q = deque([i for i in range(n) if indeg[i]==0])
order = []
while q:
    u = q.popleft(); order.append(u)
    for v in g[u]:
        indeg[v] -= 1
        if indeg[v] == 0: q.append(v)
return len(order) == n   # 是否无环

# 3. 网格 DFS「沉岛」
def dfs(i, j):
    if i<0 or i>=m or j<0 or j>=n or grid[i][j]!='1': return
    grid[i][j] = '0'        # 沉岛 = 标记 visited
    for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
        dfs(i+di, j+dj)
```

## 关卡 1 · 课程表

::: info 学习目标
把「能否完成所有课程」翻译成「图是否无环」，掌握 Kahn BFS 模板。
:::

<LcLesson problem-id="course-schedule" />

## 关卡 2 · 岛屿数量

::: info 学习目标
学会「扫到 1 就 cnt++ 然后 DFS 沉岛」的连通分量计数模板；体会 grid 自身充当 visited 的省空间技巧。
:::

<LcLesson problem-id="number-of-islands" />

## 关卡 3 · 腐烂的橘子

::: info 学习目标
掌握**多源 BFS**——所有腐烂橘子作为「第 0 层」一次性入队，按层扩张计时间。
:::

<LcLesson problem-id="rotting-oranges" />

## 同 pattern 索引（hot100 同类题）

| 题号 | 题名 | 难度 | 关键变形 |
|------|------|------|----------|
| 210 | 课程表 II | 中等 | 在 Kahn BFS 中按弹出顺序记录 order 即拓扑序 |
| 695 | 岛屿的最大面积 | 中等 | dfs 返回 1 + 四邻 dfs 之和；外层取 max |
| 130 | 被围绕的区域 | 中等 | 从边界出发反向 DFS 标记「不被围绕」的 O，剩下的 O 就翻 X |
| 417 | 太平洋大西洋水流 | 中等 | 从两个海洋逆向 DFS / BFS，求两片可达区域的交集 |

## 面试常踩

- **`from collections import deque`**：BFS 必备。`list.pop(0)` 是 O(n)，会让算法退化。
- **dirs 常量**：`((1,0),(-1,0),(0,1),(0,-1))` 写在循环外、循环里 `for di, dj in dirs` 可读性最好；想扩 8 方向只要加四角。
- **越界检查在前**：`0 <= ni < m and 0 <= nj < n and grid[ni][nj] == ...`——靠 Python 的短路求值避免 IndexError。
- **多源 BFS vs 普通 BFS**：从 m 个源同时扩张，等价于一个虚拟超级源——一次性入队全部源点是关键。
- **拓扑判环 = 看是否所有点都能被弹出**：剩下的就是环上节点。
