import type { Problem } from '../types'

const code = `def solveNQueens(n: int) -> list[list[str]]:
    res = []
    cols = set()      # 已占的列
    diag1 = set()     # 已占的「主对角线」：row - col 相同
    diag2 = set()     # 已占的「副对角线」：row + col 相同
    queens = [-1] * n # queens[row] = col

    def backtrack(row: int) -> None:
        if row == n:
            board = []
            for r in range(n):
                line = ['.'] * n
                line[queens[r]] = 'Q'
                board.append(''.join(line))
            res.append(board)
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            cols.add(col); diag1.add(row - col); diag2.add(row + col)
            queens[row] = col
            backtrack(row + 1)
            cols.remove(col); diag1.remove(row - col); diag2.remove(row + col)

    backtrack(0)
    return res`

export const problem: Problem = {
  id: 'n-queens',
  leetcodeNo: 51,
  title: { zh: 'N 皇后', en: 'N-Queens' },
  difficulty: 'hard',
  pattern: 'backtracking',
  tags: ['array', 'backtracking', 'recursion'],
  statement:
    '按照国际象棋的规则，皇后可以攻击与之处在**同一行**或**同一列**或**同一斜线**上的棋子。\n\n**N 皇后问题**研究的是如何将 `n` 个皇后放置在 `n × n` 的棋盘上，并且使皇后彼此之间**不能相互攻击**。\n\n给你一个整数 `n`，返回所有不同的 **N 皇后问题**的解决方案。每一种解法包含一个不同的 N 皇后问题的棋子放置方案，该方案中 `Q` 和 `.` 分别代表了皇后和空位。',
  examples: [
    {
      input: 'n = 4',
      output: '[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]',
    },
    { input: 'n = 1', output: '[["Q"]]' },
  ],
  constraints: ['1 ≤ n ≤ 9'],
  intuition:
    '按行枚举（同行天然不能再放）→ 每行选一列。三种冲突一次 O(1) 检查：① cols 记录列；② diag1 记录 row-col 主对角线；③ diag2 记录 row+col 副对角线。同一对角线上的格子 row-col（或 row+col）相等。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n!) — 第一行 n 种、第二行至多 n-2 种…', space: 'O(n) — 三个 set 各 ≤ n、递归栈 ≤ n' },
  microQuestions: [
    {
      id: 'n-queens.q1',
      prompt: '为什么我们不需要「row 是否被占」的检查？',
      options: [
        { id: 'a', text: '按行枚举——每行只放一个，row 自动不冲突' },
        { id: 'b', text: '题目保证不冲突' },
        { id: 'c', text: '需要，作者忘了写' },
        { id: 'd', text: 'row 永远 = col' },
      ],
      answer: 'a',
      explain:
        '回溯按行展开，每层 row 只放一个皇后，`backtrack(row+1)` 就到下一行——row 维度的「冲突」从决策结构上就不可能发生。这是 N 皇后最优雅的设计。',
      tags: ['invariant'],
    },
    {
      id: 'n-queens.q2',
      prompt: '同一条「主对角线」（左上→右下）上格子的什么值相等？',
      codeContext: code,
      highlightLine: 4,
      options: [
        { id: 'a', text: 'row + col' },
        { id: 'b', text: 'row - col' },
        { id: 'c', text: 'row * col' },
        { id: 'd', text: 'row 与 col 无关系' },
      ],
      answer: 'b',
      explain:
        '主对角线上 (0,0)、(1,1)、(2,2) 的 row-col 都是 0；另一条主对角线 (0,1)、(1,2)、(2,3) 的 row-col 都是 -1。所以 `row-col` 是主对角线的「身份证」。',
      tags: ['invariant'],
    },
    {
      id: 'n-queens.q3',
      prompt: '同一条「副对角线」（右上→左下）上格子的什么值相等？',
      codeContext: code,
      highlightLine: 5,
      options: [
        { id: 'a', text: 'row + col' },
        { id: 'b', text: 'row - col' },
        { id: 'c', text: 'col - row' },
        { id: 'd', text: 'col / row' },
      ],
      answer: 'a',
      explain:
        '副对角线 (0,3)、(1,2)、(2,1)、(3,0) 的 row+col 都是 3。和主对角线对应——`row+col` 是副对角线的「身份证」。',
      tags: ['invariant'],
    },
    {
      id: 'n-queens.q4',
      prompt: '`if col in cols or (row - col) in diag1 or (row + col) in diag2` 这一行检查的是？',
      codeContext: code,
      highlightLine: 17,
      options: [
        { id: 'a', text: '当前位置是否被三种攻击线之一占据' },
        { id: 'b', text: '当前位置是否在棋盘内' },
        { id: 'c', text: 'row 是否合法' },
        { id: 'd', text: 'col 是否为质数' },
      ],
      answer: 'a',
      explain:
        '皇后冲突只有列、主对角线、副对角线三条（行已经被「按行枚举」结构性排除）。这一行 O(1) 完成所有冲突检查——这是 N 皇后从 O(n²) 检查降到 O(1) 的关键。',
      tags: ['invariant', 'data-structure'],
    },
    {
      id: 'n-queens.q5',
      prompt: '为什么用三个 set 比「每次扫已放皇后逐个比对」快？',
      options: [
        { id: 'a', text: '逐个比对 O(row) 每行；三 set O(1)；整体从 O(n²) 降到 O(n)' },
        { id: 'b', text: '没区别' },
        { id: 'c', text: 'set 在 Python 慢' },
        { id: 'd', text: '三 set 答案错' },
      ],
      answer: 'a',
      explain:
        '把「冲突检查」预先分类记录在 hash set 里，是把 O(row) 的扫描换成 O(1) 哈希查询的典型空间换时间。',
      tags: ['complexity'],
    },
    {
      id: 'n-queens.q6',
      prompt: '回溯撤销时为什么三个 set 都要 remove？',
      codeContext: code,
      highlightLine: 22,
      options: [
        { id: 'a', text: '三段式标准撤销——和前面三个 add 一一配对' },
        { id: 'b', text: '只 remove cols 即可' },
        { id: 'c', text: 'Python 自动撤销' },
        { id: 'd', text: '可以省略' },
      ],
      answer: 'a',
      explain:
        '回溯三段式：「做选择 → 递归 → 撤销选择」必须严格对称。少撤一项就会污染其它分支，搜索结果错。',
      tags: ['invariant'],
    },
    {
      id: 'n-queens.q7',
      prompt: '`queens = [-1]*n` 的作用是？',
      codeContext: code,
      highlightLine: 6,
      options: [
        { id: 'a', text: '记录每行的皇后所在列，便于到达叶子节点时 O(n²) 还原棋盘字符串' },
        { id: 'b', text: '随便写的占位' },
        { id: 'c', text: '存放结果' },
        { id: 'd', text: '记录递归深度' },
      ],
      answer: 'a',
      explain:
        '用一个 list 表达「行→列」的映射比维护完整 m×n 棋盘更省、更快；命中叶子时再一次性渲染成字符行 `["....", ...]`。',
      tags: ['data-structure'],
    },
    {
      id: 'n-queens.q8',
      prompt: '把字符串行 `["...Q"]` 拼起来时常用的写法是？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: 'line = ["."]*n; line[col] = "Q"; "".join(line)' },
        { id: 'b', text: '"."*col + "Q" + "."*(n-col-1)（也对，但容易写错索引）' },
        { id: 'c', text: '上述两种都常用' },
        { id: 'd', text: 'Python 不支持字符串' },
      ],
      answer: 'c',
      explain:
        '两种风格都很常见：list 拼接 + join 更不易写错；字符串切片更简洁。建议熟悉两种写法都能口算。',
      tags: ['pythonism'],
    },
    {
      id: 'n-queens.q9',
      prompt: 'n=4 一共有几个解？',
      options: [
        { id: 'a', text: '1' },
        { id: 'b', text: '2' },
        { id: 'c', text: '4' },
        { id: 'd', text: '6' },
      ],
      answer: 'b',
      explain:
        'n=4 的两个经典解互为镜像。常用来调试代码——若你的程序在 n=4 上不返回 2，说明剪枝/对角线索引一定写错了。',
      tags: ['boundary'],
    },
    {
      id: 'n-queens.q10',
      prompt: '本题最坏时间复杂度的合理表达？',
      options: [
        { id: 'a', text: 'O(n²)' },
        { id: 'b', text: 'O(2ⁿ)' },
        { id: 'c', text: 'O(n!) — 第一行 n 种选择、第二行 ≤ n-2、第三行 ≤ n-4 …' },
        { id: 'd', text: 'O(n^n)' },
      ],
      answer: 'c',
      explain:
        '冲突检查后每行有效选择递减，整体接近 O(n!)。用 O(n^n) 也是合法上界但太松，O(n!) 更准确。',
      tags: ['complexity'],
    },
    {
      id: 'n-queens.q11',
      prompt: '若题目改成「LC52 只问解的个数」，最易优化是？',
      options: [
        { id: 'a', text: '不构造棋盘字符串、不维护 queens——只在叶子处 cnt += 1；用位运算（mask 三个 int）替换三个 set 是经典极致优化' },
        { id: 'b', text: '改用动态规划' },
        { id: 'c', text: 'BFS' },
        { id: 'd', text: '没区别' },
      ],
      answer: 'a',
      explain:
        'LC52 不需要返回棋盘，可以省下 O(n²) 的字符串构造；进一步用位运算 mask 把三个 set 压成三个 int，常数缩到极致。',
      tags: ['data-structure', 'pythonism'],
    },
    {
      id: 'n-queens.q12',
      prompt: '空间复杂度（不计输出）是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(n) — 三个 set 加 queens 加递归栈' },
        { id: 'c', text: 'O(n²)' },
        { id: 'd', text: 'O(n!)' },
      ],
      answer: 'b',
      explain:
        '三个 set 各 ≤ n、queens 数组 = n、递归栈深度 = n，合起来 O(n)。',
      tags: ['complexity'],
    },
  ],
}

export default problem
