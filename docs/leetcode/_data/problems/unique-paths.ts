import type { Problem } from '../types'

const code = `def uniquePaths(m: int, n: int) -> int:
    # 一维滚动:dp[j] 表示「到达当前行第 j 列」的路径数
    dp = [1] * n  # 第一行全部为 1
    for i in range(1, m):
        # dp[0] 始终是 1(第一列只能从上面下来)
        for j in range(1, n):
            dp[j] = dp[j] + dp[j - 1]  # 上方 + 左方
    return dp[n - 1]`

export const problem: Problem = {
  id: 'unique-paths',
  leetcodeNo: 62,
  title: { zh: '不同路径', en: 'Unique Paths' },
  difficulty: 'medium',
  pattern: 'dp-2d',
  tags: ['dp', 'grid', 'rolling-array', 'combinatorics'],
  statement:
    '一个机器人位于一个 `m x n` 网格的**左上角**（起始点在下图中标记为 「Start」）。\n\n机器人**每次只能向下或者向右**移动一步。机器人试图达到网格的右下角（标记为 「Finish」）。\n\n问总共有多少条不同的路径？',
  examples: [
    { input: 'm = 3, n = 7', output: '28' },
    { input: 'm = 3, n = 2', output: '3' },
    { input: 'm = 7, n = 3', output: '28', note: '行列对换答案不变（组合数对称）' },
  ],
  constraints: [
    '1 ≤ m, n ≤ 100',
    '题目保证答案小于等于 2 × 10⁹',
  ],
  intuition:
    '到达 (i, j) 只能来自正上方 (i-1, j) 或正左方 (i, j-1)，所以 dp[i][j] = dp[i-1][j] + dp[i][j-1]。第一行只能从左来 → 全 1；第一列只能从上来 → 全 1。由于 dp[i][j] 只依赖"上方"和"左方"，可以用一维滚动数组：从左往右遍历时，`dp[j]`（更新前）= 上方旧值，`dp[j-1]`（已更新）= 左方新值。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(m·n)', space: 'O(n)（一维滚动）或 O(1) 用组合数公式' },
  microQuestions: [
    {
      id: 'unique-paths.q1',
      prompt: '`dp[i][j]` 的状态定义是？',
      options: [
        { id: 'a', text: '从 (0,0) 到 (i,j) 的不同路径数' },
        { id: 'b', text: '到达 (i,j) 已走过的步数' },
        { id: 'c', text: '从 (i,j) 到 (m-1,n-1) 的路径数' },
        { id: 'd', text: '是否能到达 (i,j)（布尔）' },
      ],
      answer: 'a',
      explain:
        '路径计数题的标准状态：dp[i][j] = 到达此格的方案数。c 也对（反向 DP），但传统从左上向右下推更直观。',
      tags: ['naming', 'invariant'],
    },
    {
      id: 'unique-paths.q2',
      prompt: '转移方程是？',
      options: [
        { id: 'a', text: 'dp[i][j] = dp[i-1][j] + dp[i][j-1]' },
        { id: 'b', text: 'dp[i][j] = dp[i-1][j] * dp[i][j-1]' },
        { id: 'c', text: 'dp[i][j] = max(dp[i-1][j], dp[i][j-1])' },
        { id: 'd', text: 'dp[i][j] = dp[i-1][j-1] + 1' },
      ],
      answer: 'a',
      explain:
        '加法原理：到达 (i,j) 的最后一步要么从上方来（dp[i-1][j] 条路）要么从左方来（dp[i][j-1] 条路），二者互不相交，求和。',
      tags: ['invariant'],
    },
    {
      id: 'unique-paths.q3',
      prompt: '边界——第一行 dp[0][j] 与第一列 dp[i][0] 应当填什么？',
      options: [
        { id: 'a', text: '都填 1，因为只能直走一条路' },
        { id: 'b', text: '都填 0' },
        { id: 'c', text: '第一行 1，第一列 0' },
        { id: 'd', text: '随便' },
      ],
      answer: 'a',
      explain:
        '第一行只能一直向右一步步走（一条路径），第一列只能一直向下（一条路径）——所以全 1。这是网格类 DP 最常见的边界。如果填 0 转移方程会全程为 0。',
      tags: ['boundary'],
    },
    {
      id: 'unique-paths.q4',
      prompt: '一维滚动数组中 `dp[j] = dp[j] + dp[j-1]` 的含义是？',
      codeContext: code,
      highlightLine: 7,
      options: [
        { id: 'a', text: '右边的 dp[j] 是"上方旧值"，dp[j-1] 是"左方刚更新值"，求和后写回' },
        { id: 'b', text: 'dp[j] 累加 dp[j-1]' },
        { id: 'c', text: 'dp 数组不变' },
        { id: 'd', text: '从右往左更新' },
      ],
      answer: 'a',
      explain:
        '理解一维滚动的关键：循环顺序决定 dp[j] 和 dp[j-1] 各自的"时态"。**从左往右**遍历 j：dp[j] 还没更新 → 是上一行的值（即"上方"）；dp[j-1] 已更新 → 是当前行的值（即"左方"）。所以右式恰好等于"上方+左方"。',
      tags: ['invariant'],
    },
    {
      id: 'unique-paths.q5',
      prompt: '一维滚动若改为**从右往左**遍历 j，会出什么问题？',
      options: [
        { id: 'a', text: 'dp[j-1] 还没被本行更新——它是"左方旧值"而不是"左方新值"，转移含义错误' },
        { id: 'b', text: '没有问题' },
        { id: 'c', text: '更快' },
        { id: 'd', text: '导致越界' },
      ],
      answer: 'a',
      explain:
        '"循环方向决定一维 DP 的正确性"是 0-1 背包 vs 完全背包的核心差异，本题方向必须从左往右。学会从这里推断方向：哪个被更新的值需要"新"还是"旧"，决定了循环顺序。',
      tags: ['invariant', 'boundary'],
    },
    {
      id: 'unique-paths.q6',
      prompt: '为什么内层从 `j=1` 而不是 `j=0` 开始？',
      codeContext: code,
      highlightLine: 6,
      options: [
        { id: 'a', text: 'dp[0]（第一列）始终为 1，不需要更新' },
        { id: 'b', text: '避免越界 dp[-1]' },
        { id: 'c', text: 'a 与 b 都对' },
        { id: 'd', text: '随便' },
      ],
      answer: 'c',
      explain:
        '两个理由都成立：dp[0] 表示第一列，永远是 1；从 j=1 开始也避免了 dp[j-1] 访问 dp[-1]（Python 中 dp[-1] 不报错但语义错）。',
      tags: ['boundary'],
    },
    {
      id: 'unique-paths.q7',
      prompt: '二维版 dp 数组初始化：`dp = [[1]*n for _ in range(m)]` 与 `dp = [[1]*n]*m` 的区别？',
      options: [
        { id: 'a', text: '完全等价' },
        { id: 'b', text: '后者所有"行"是同一个 list 对象的引用——改一行所有行都变，是经典 Python 陷阱' },
        { id: 'c', text: '前者更慢' },
        { id: 'd', text: '后者会报错' },
      ],
      answer: 'b',
      explain:
        '`[[1]*n]*m` 的 `*m` 只是把同一个 list 的引用复制 m 次。给 dp[1][2]=5 你会看到 dp[0][2] 也变成 5。务必用 list comprehension 来真正生成 m 个独立行。这是 Python 二维数组初始化的头号坑。',
      tags: ['pythonism', 'boundary'],
    },
    {
      id: 'unique-paths.q8',
      prompt: '本题的纯组合数学解是？',
      options: [
        { id: 'a', text: 'C(m+n-2, m-1) —— 总共要走 m-1 次下 + n-1 次右,从中选哪 m-1 步是下' },
        { id: 'b', text: 'm * n' },
        { id: 'c', text: '2^(m+n)' },
        { id: 'd', text: 'm! * n!' },
      ],
      answer: 'a',
      explain:
        '到 (m-1, n-1) 必须恰好走 m-1 步下、n-1 步右，总步数 m+n-2，"在哪些步走下"完全决定路径——所以是组合数。Python 用 `math.comb(m+n-2, m-1)` 直接 O(1) 空间 O(min(m,n)) 时间。',
      tags: ['invariant', 'pythonism'],
    },
    {
      id: 'unique-paths.q9',
      prompt: '时间复杂度是？',
      options: [
        { id: 'a', text: 'O(m + n)' },
        { id: 'b', text: 'O(m · n)' },
        { id: 'c', text: 'O(m²)' },
        { id: 'd', text: 'O(2^(m+n))' },
      ],
      answer: 'b',
      explain:
        '每个格子常数时间更新一次，总 m·n 个格子。组合数法可以 O(min(m,n))。',
      tags: ['complexity'],
    },
    {
      id: 'unique-paths.q10',
      prompt: '一维滚动后的空间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(n)（取较小那一维）' },
        { id: 'c', text: 'O(m·n)' },
        { id: 'd', text: 'O(log n)' },
      ],
      answer: 'b',
      explain:
        '只保留一行 dp（n 个）。还可以选 min(m, n) 那一维进一步省常数。组合数法是真正的 O(1)。',
      tags: ['complexity'],
    },
    {
      id: 'unique-paths.q11',
      prompt: '若网格存在障碍物（LC 63 不同路径 II），DP 应如何修改？',
      options: [
        { id: 'a', text: '障碍格 dp[i][j] = 0；其他保持转移；第一行/列遇到障碍后**之后所有格**也归 0' },
        { id: 'b', text: '直接跳过障碍格' },
        { id: 'c', text: '把障碍格当起点' },
        { id: 'd', text: '用 BFS 替代 DP' },
      ],
      answer: 'a',
      explain:
        '障碍 = 不可达 = 0。第一行/列的边界要小心：遇到第一个障碍后，后面格子失去"从前面继承的 1"——必须显式置 0。这是 LC 63 的常见 bug。',
      tags: ['boundary'],
    },
    {
      id: 'unique-paths.q12',
      prompt: '若返回值"很大"题目要求取模 10⁹+7，是否影响 DP 结构？',
      options: [
        { id: 'a', text: '不影响——每次 dp[j] = (dp[j] + dp[j-1]) % MOD 即可' },
        { id: 'b', text: '需要换算法' },
        { id: 'c', text: '不能取模' },
        { id: 'd', text: '需要矩阵快速幂' },
      ],
      answer: 'a',
      explain:
        '"加法 + 取模"在每步都做 % MOD 即可——这是计数 DP 加模数的通用招式。本题不要求取模（题目保证 ≤ 2e9，64 位整数足够），但 LC 1639 这类计数题就必须取模。',
      tags: ['boundary'],
    },
  ],
}

export default problem
