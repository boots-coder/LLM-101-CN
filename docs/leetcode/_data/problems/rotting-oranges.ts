import type { Problem } from '../types'

const code = `from collections import deque

def orangesRotting(grid: list[list[int]]) -> int:
    m, n = len(grid), len(grid[0])
    q = deque()
    fresh = 0
    # 1. 收集所有「初始就腐烂」的橘子作为多源 BFS 起点；同时统计新鲜橘子数
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 2:
                q.append((i, j))
            elif grid[i][j] == 1:
                fresh += 1

    minutes = 0
    dirs = ((1,0),(-1,0),(0,1),(0,-1))
    # 2. 一层一层扩展，每扩张一层 minutes += 1
    while q and fresh > 0:
        for _ in range(len(q)):              # 把当前层一次性吐完
            i, j = q.popleft()
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1:
                    grid[ni][nj] = 2
                    fresh -= 1
                    q.append((ni, nj))
        minutes += 1

    return minutes if fresh == 0 else -1`

export const problem: Problem = {
  id: 'rotting-oranges',
  leetcodeNo: 994,
  title: { zh: '腐烂的橘子', en: 'Rotting Oranges' },
  difficulty: 'medium',
  pattern: 'graph',
  tags: ['matrix', 'bfs', 'graph'],
  statement:
    '在给定的 `m x n` 网格 `grid` 中，每个单元格可以有以下三个值之一：\n\n- `0` 代表**空单元格**；\n- `1` 代表**新鲜橘子**；\n- `2` 代表**腐烂的橘子**。\n\n每分钟，**腐烂的橘子周围 4 个方向上相邻的新鲜橘子**都会腐烂。\n\n返回直到单元格中没有新鲜橘子为止所必须经过的**最小分钟数**。如果不可能，返回 `-1`。',
  examples: [
    { input: 'grid = [[2,1,1],[1,1,0],[0,1,1]]', output: '4' },
    { input: 'grid = [[2,1,1],[0,1,1],[1,0,1]]', output: '-1', note: '左下角的 1 永远无法被腐烂——它没有任何相邻的腐烂橘子' },
    { input: 'grid = [[0,2]]', output: '0', note: '一开始就没有新鲜橘子' },
  ],
  constraints: [
    'm == grid.length, n == grid[i].length',
    '1 ≤ m, n ≤ 10',
    'grid[i][j] 仅为 0、1、2',
  ],
  intuition:
    '「最少分钟数」= BFS 的层数。把所有初始腐烂橘子一次性入队作为「第 0 层」，每层让相邻新鲜橘子腐烂并入队，扩展一层 minutes++。最终若 fresh > 0 表示有橘子永远到不了 → -1。这是**多源 BFS** 的经典应用。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(m·n) — 每个格子最多入队/出队一次', space: 'O(m·n) — 队列最坏装下整张图' },
  microQuestions: [
    {
      id: 'rotting-oranges.q1',
      prompt: '本题最自然的算法是？',
      options: [
        { id: 'a', text: '多源 BFS——把所有初始腐烂橘子作为「第 0 层」一次性入队' },
        { id: 'b', text: 'DFS' },
        { id: 'c', text: '动态规划' },
        { id: 'd', text: '贪心选择最近邻' },
      ],
      answer: 'a',
      explain:
        '「最少时间 / 最短路径」+「网格四方向」≈ 几乎一定是 BFS。多个起点要同时扩张时，用「多源 BFS」一次性入队。',
      tags: ['data-structure'],
    },
    {
      id: 'rotting-oranges.q2',
      prompt: '为什么必须先把**所有**腐烂橘子一次性入队？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: '它们同时在第 0 分钟向四周扩散——按时间层来分，所有起点等价于「第 0 层」' },
        { id: 'b', text: '只入队第一个就行' },
        { id: 'c', text: '为了节省内存' },
        { id: 'd', text: '没意义' },
      ],
      answer: 'a',
      explain:
        '只入队一个就跑普通 BFS 会高估时间。多源 BFS 等价于把多个 source 拼成一个「虚拟超级源」，一齐扩张才是真实物理过程。',
      tags: ['invariant'],
    },
    {
      id: 'rotting-oranges.q3',
      prompt: '「按层扩张」的写法核心是？',
      codeContext: code,
      highlightLine: 18,
      options: [
        { id: 'a', text: '`for _ in range(len(q))` —— 每轮只处理「当前层」的所有节点' },
        { id: 'b', text: '一直 popleft 直到队列空' },
        { id: 'c', text: '使用栈代替' },
        { id: 'd', text: '随机选取' },
      ],
      answer: 'a',
      explain:
        '关键习惯：进 while 后先冻结 `len(q)` 作为本层节点数，循环结束 minutes += 1。这是「BFS 按层」的标准模板，比「在节点里塞 (i,j,depth)」更直观。',
      tags: ['data-structure', 'pythonism'],
    },
    {
      id: 'rotting-oranges.q4',
      prompt: '`fresh` 计数器有两个作用，是哪两个？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: '① 提前终止 BFS（fresh 到 0 即结束）；② 最终判断「是否有橘子永远没腐烂」' },
        { id: 'b', text: '只是为了打印' },
        { id: 'c', text: '记录腐烂橘子数' },
        { id: 'd', text: '记录递归深度' },
      ],
      answer: 'a',
      explain:
        '不维护 fresh 也能跑（`while q`），但要在最后再扫一遍 grid 找剩余 1。维护 fresh 让两件事一气呵成、还能当 BFS 提前终止条件。',
      tags: ['invariant'],
    },
    {
      id: 'rotting-oranges.q5',
      prompt: '`while q and fresh > 0` 中 `fresh > 0` 这个条件的意义？',
      codeContext: code,
      highlightLine: 17,
      options: [
        { id: 'a', text: '剪枝——如果 fresh 已经为 0 就不需要再扩张一层（避免 minutes 多 +1）' },
        { id: 'b', text: '没用' },
        { id: 'c', text: '为了让循环结束' },
        { id: 'd', text: '用 q 即可，fresh 是噪音' },
      ],
      answer: 'a',
      explain:
        '如果只写 `while q`，最后一层节点出队后 minutes 还会被 +1 一次。加上 `fresh > 0` 让「最后一个新鲜橘子被腐蚀的瞬间」就停下来。',
      tags: ['boundary'],
    },
    {
      id: 'rotting-oranges.q6',
      prompt: '若初始网格中根本没有新鲜橘子（全 0 和 2），应该返回？',
      codeContext: code,
      highlightLine: 30,
      options: [
        { id: 'a', text: '0 — 无需等待，直接达成目标状态' },
        { id: 'b', text: '-1' },
        { id: 'c', text: 'm·n' },
        { id: 'd', text: '抛异常' },
      ],
      answer: 'a',
      explain:
        '`fresh==0` 时 while 循环根本不会进，minutes 保持 0；最后 `if fresh == 0` 直接 return 0。这正是「fresh 计数器同时管两件事」的优雅之处。',
      tags: ['boundary'],
    },
    {
      id: 'rotting-oranges.q7',
      prompt: '`grid[ni][nj] = 2` 这一步做什么？',
      codeContext: code,
      highlightLine: 24,
      options: [
        { id: 'a', text: '把新鲜橘子标为腐烂——既改变状态又起到 visited 作用，避免重复入队' },
        { id: 'b', text: '永久销毁' },
        { id: 'c', text: '调试用' },
        { id: 'd', text: '没用' },
      ],
      answer: 'a',
      explain:
        '把 grid 自身当 visited 是这类网格 BFS 的标配，省一份 m×n 的 visited 数组。代价是污染输入。',
      tags: ['data-structure'],
    },
    {
      id: 'rotting-oranges.q8',
      prompt: '为什么用 deque 而不用 list 做队列？',
      options: [
        { id: 'a', text: 'deque 的 popleft 是 O(1)；list.pop(0) 是 O(n)，会让 BFS 退化' },
        { id: 'b', text: '没区别' },
        { id: 'c', text: 'deque 是栈' },
        { id: 'd', text: 'list 不能 BFS' },
      ],
      answer: 'a',
      explain:
        'BFS 本质要求「先进先出」，deque 是 Python 的标准选择。`from collections import deque` 是 BFS 题的肌肉记忆。',
      tags: ['pythonism', 'complexity'],
    },
    {
      id: 'rotting-oranges.q9',
      prompt: '若 BFS 结束后还剩 fresh > 0，意味着？',
      options: [
        { id: 'a', text: '有新鲜橘子永远到不了任何腐烂源——它们与所有源不连通' },
        { id: 'b', text: '题目数据错' },
        { id: 'c', text: 'BFS 写错了' },
        { id: 'd', text: '答案是 fresh' },
      ],
      answer: 'a',
      explain:
        'BFS 已经把所有可达点都腐蚀了，剩下的就是不连通分量上的新鲜橘子，物理上永远不会腐烂——返回 -1。',
      tags: ['invariant'],
    },
    {
      id: 'rotting-oranges.q10',
      prompt: '本题时间复杂度?',
      options: [
        { id: 'a', text: 'O(m·n) — 每格最多入队/出队一次' },
        { id: 'b', text: 'O((m·n)²)' },
        { id: 'c', text: 'O(m+n)' },
        { id: 'd', text: 'O(m·n·log(m·n))' },
      ],
      answer: 'a',
      explain:
        'BFS 的时间复杂度 = O(V+E) — 网格图里 V=m·n、E=4·m·n，合起来 O(m·n)。',
      tags: ['complexity'],
    },
    {
      id: 'rotting-oranges.q11',
      prompt: '空间复杂度?',
      options: [
        { id: 'a', text: 'O(m·n) — 队列最坏装下整张图' },
        { id: 'b', text: 'O(1)' },
        { id: 'c', text: 'O(min(m,n))' },
        { id: 'd', text: 'O(m+n)' },
      ],
      answer: 'a',
      explain:
        '所有格子都可能同时在某一层入队（极端情况下）；BFS 队列宽度可达 m·n。',
      tags: ['complexity'],
    },
    {
      id: 'rotting-oranges.q12',
      prompt: '如果题目变体「时间不是 1 分钟而是任意权重」（→ Dijkstra 范畴），骨架要变吗？',
      options: [
        { id: 'a', text: '需要——BFS 只对「等权重」最短路；带权要 Dijkstra（堆优先队列）' },
        { id: 'b', text: '不变' },
        { id: 'c', text: 'BFS 永远适用' },
        { id: 'd', text: 'DFS 即可' },
      ],
      answer: 'a',
      explain:
        '这是 BFS / Dijkstra 的分水岭：等权重 → BFS；非负权重 → Dijkstra；可负权 → Bellman-Ford。腐烂橘子是等权 BFS 的典型。',
      tags: ['data-structure'],
    },
  ],
}

export default problem
