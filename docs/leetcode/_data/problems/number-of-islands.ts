import type { Problem } from '../types'

const code = `def numIslands(grid: list[list[str]]) -> int:
    if not grid or not grid[0]:
        return 0
    m, n = len(grid), len(grid[0])

    def dfs(i: int, j: int) -> None:
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
            return
        grid[i][j] = '0'                 # 「沉岛」——把当前陆地变成水，避免重复访问
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            dfs(i + di, j + dj)

    cnt = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                cnt += 1
                dfs(i, j)
    return cnt`

export const problem: Problem = {
  id: 'number-of-islands',
  leetcodeNo: 200,
  title: { zh: '岛屿数量', en: 'Number of Islands' },
  difficulty: 'medium',
  pattern: 'graph',
  tags: ['matrix', 'dfs', 'bfs', 'union-find', 'graph'],
  statement:
    '给你一个由 `\'1\'`（陆地）和 `\'0\'`（水）组成的二维网格，请你计算网格中**岛屿**的数量。\n\n岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上**相邻**的陆地连接形成。\n\n此外，你可以假设该网格的四条边均被水包围。',
  examples: [
    {
      input: 'grid = [\n  ["1","1","1","1","0"],\n  ["1","1","0","1","0"],\n  ["1","1","0","0","0"],\n  ["0","0","0","0","0"]\n]',
      output: '1',
    },
    {
      input: 'grid = [\n  ["1","1","0","0","0"],\n  ["1","1","0","0","0"],\n  ["0","0","1","0","0"],\n  ["0","0","0","1","1"]\n]',
      output: '3',
    },
  ],
  constraints: [
    'm == grid.length, n == grid[i].length',
    '1 ≤ m, n ≤ 300',
    'grid[i][j] 为 "0" 或 "1"',
  ],
  intuition:
    '扫一遍每个格子。遇到 "1" 就 cnt++ 并启动一次 DFS 把整片相连的 "1" 全部「沉岛」（改成 "0"），保证它们不再被外层循环计数。BFS 同样可行——把递归换成 deque。也可以用并查集（更适合「动态加入岛」的变体）。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(m·n) — 每个格子最多被访问一次', space: 'O(m·n) 最坏递归栈深度（整张图都是岛）' },
  microQuestions: [
    {
      id: 'number-of-islands.q1',
      prompt: '本题「数岛屿」的关键技巧是？',
      options: [
        { id: 'a', text: '扫到一个 "1" 就 cnt++，然后把整片相连的陆地一次性「沉岛」（变 "0"）以避免重复计数' },
        { id: 'b', text: '排序后二分' },
        { id: 'c', text: '逐行扫一次，每行有几个 1 加几次' },
        { id: 'd', text: '动态规划 dp[i][j]' },
      ],
      answer: 'a',
      explain:
        '「沉岛 + 计数」是网格连通分量的标准模板：发现新连通块就 +1，再 DFS/BFS 把这块标记掉。',
      tags: ['data-structure'],
    },
    {
      id: 'number-of-islands.q2',
      prompt: '`grid[i][j] = "0"` 这一步在做什么？',
      codeContext: code,
      highlightLine: 9,
      options: [
        { id: 'a', text: '原地把陆地改成水，达到「visited」效果——省去额外 visited 数组' },
        { id: 'b', text: '永久销毁数据' },
        { id: 'c', text: '调试用' },
        { id: 'd', text: '把它换成水后好打印' },
      ],
      answer: 'a',
      explain:
        '这个「沉岛」技巧用 grid 自己当 visited，O(1) 额外空间。代价是污染输入——若题目要求保留输入则需另开 visited 数组。',
      tags: ['data-structure'],
    },
    {
      id: 'number-of-islands.q3',
      prompt: '为什么我们不需要在 dfs 出口处「还原」grid？',
      options: [
        { id: 'a', text: '本题只数岛屿数量、之后再也不需要这个格子；不像 word search 那种「同格不能在同条路径里反复」的回溯' },
        { id: 'b', text: '需要还原，作者写错了' },
        { id: 'c', text: 'Python 自动还原' },
        { id: 'd', text: '随便' },
      ],
      answer: 'a',
      explain:
        '注意区分：回溯题（path 复用）出 dfs 要还原；连通分量题（一次性把这片标记掉）不还原。两者用途不同，不要混淆。',
      tags: ['invariant'],
    },
    {
      id: 'number-of-islands.q4',
      prompt: '`for di, dj in ((1,0),(-1,0),(0,1),(0,-1))` 的作用？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: '四方向偏移量——上下左右' },
        { id: 'b', text: '随便选的常量' },
        { id: 'c', text: 'Python 内置' },
        { id: 'd', text: '初始化 visited' },
      ],
      answer: 'a',
      explain:
        '`dirs` 是网格题的招牌写法：写在循环外作为常量、循环里 `for di, dj in dirs` 一气呵成。比 4 个独立 dfs 调用更整洁、也方便扩展为 8 方向。',
      tags: ['pythonism'],
    },
    {
      id: 'number-of-islands.q5',
      prompt: 'dfs 入口的越界 + 类型检查 `i<0 or i>=m or j<0 or j>=n or grid[i][j] != "1"` 这一行可以拆成两行写吗？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: '可以但不必，必须保证「越界检查在前」否则 grid[i][j] 会 IndexError' },
        { id: 'b', text: '不行' },
        { id: 'c', text: '必须拆' },
        { id: 'd', text: 'Python 求值顺序无所谓' },
      ],
      answer: 'a',
      explain:
        'Python 的 `or` 短路求值——前面任何一个为真后续就不会算 grid[i][j]，避免越界。这也是把越界检查放在 grid 检查之前的经典原因。',
      tags: ['boundary', 'pythonism'],
    },
    {
      id: 'number-of-islands.q6',
      prompt: 'BFS 版与 DFS 版的核心差别？',
      options: [
        { id: 'a', text: '把递归改为 deque：发现 "1" 时入队 + 标记，循环 popleft 扩展四方向' },
        { id: 'b', text: 'BFS 慢一倍' },
        { id: 'c', text: 'BFS 不能解' },
        { id: 'd', text: 'BFS 必须用并查集' },
      ],
      answer: 'a',
      explain:
        'BFS 与 DFS 在本题等价、复杂度相同。BFS 的好处：不依赖递归栈、网格非常大时不会 RecursionError。',
      tags: ['data-structure'],
    },
    {
      id: 'number-of-islands.q7',
      prompt: '并查集（Union-Find）解法的思路是？',
      options: [
        { id: 'a', text: '初始每个 "1" 是一个独立分量；扫描时把相邻的 "1" union 起来；最后数有几个根' },
        { id: 'b', text: '每个格子都单独是一棵树' },
        { id: 'c', text: '与 DFS 相同' },
        { id: 'd', text: '不可能用并查集' },
      ],
      answer: 'a',
      explain:
        '并查集对「动态加入边/格子」更友好（如 LC305 岛屿数量 II）；纯静态本题用 DFS 更直观。',
      tags: ['data-structure'],
    },
    {
      id: 'number-of-islands.q8',
      prompt: '本题为何不需要单独的 visited 数组（即使不允许污染输入）？',
      options: [
        { id: 'a', text: '允许污染时直接「沉岛」即可；不允许污染则必须新开 m×n 的 visited' },
        { id: 'b', text: '永远不需要 visited' },
        { id: 'c', text: 'Python 内置 visited' },
        { id: 'd', text: 'visited 与 grid 等价' },
      ],
      answer: 'a',
      explain:
        '这是一个常见面试问题的两个版本：默认实现污染 grid 节省空间；如果面试官追问「不许改 grid」，要会写 visited 二维数组版本。',
      tags: ['boundary', 'data-structure'],
    },
    {
      id: 'number-of-islands.q9',
      prompt: '时间复杂度?',
      options: [
        { id: 'a', text: 'O(m+n)' },
        { id: 'b', text: 'O(m·n) — 每个格子最多被访问一次' },
        { id: 'c', text: 'O((m·n)²)' },
        { id: 'd', text: 'O(m·n·log(m·n))' },
      ],
      answer: 'b',
      explain:
        '所有格子被外层循环访问 1 次，被 DFS 访问最多 1 次（沉岛后再次扫到也立刻 return），总 O(m·n)。',
      tags: ['complexity'],
    },
    {
      id: 'number-of-islands.q10',
      prompt: '空间复杂度?',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(min(m,n)) — BFS 队列最大宽度' },
        { id: 'c', text: 'O(m·n) — 最坏全是岛时 DFS 栈深可达 m·n；BFS 也可能整片入队' },
        { id: 'd', text: 'O(m+n)' },
      ],
      answer: 'c',
      explain:
        '严格上界 O(m·n)。教学性表述：DFS 栈深与「最长一连串相邻陆地」相关；BFS 队列宽度也由这一连块决定。最坏整张图都是 1，所以 O(m·n)。',
      tags: ['complexity'],
    },
    {
      id: 'number-of-islands.q11',
      prompt: '若题目改为「数岛屿面积之和的最大值」（LC695），最小改动？',
      options: [
        { id: 'a', text: 'dfs 返回 1 + 四方向 dfs 之和；外层取 max' },
        { id: 'b', text: '完全换算法' },
        { id: 'c', text: '不可能解' },
        { id: 'd', text: '加排序' },
      ],
      answer: 'a',
      explain:
        'LC695 把 dfs 的返回值从「无」变成「这片岛的面积」即可，骨架完全相同。理解了 200 几乎能秒解 695。',
      tags: ['data-structure'],
    },
  ],
}

export default problem
