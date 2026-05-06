import type { Problem } from '../types'

const code = `def climbStairs(n: int) -> int:
    if n <= 2:
        return n
    # dp[i] 表示「到达第 i 阶」的方法数
    prev2, prev1 = 1, 2  # dp[1]=1, dp[2]=2
    for i in range(3, n + 1):
        cur = prev1 + prev2
        prev2, prev1 = prev1, cur
    return prev1`

export const problem: Problem = {
  id: 'climbing-stairs',
  leetcodeNo: 70,
  title: { zh: '爬楼梯', en: 'Climbing Stairs' },
  difficulty: 'easy',
  pattern: 'dp-1d',
  tags: ['dp', 'fibonacci', 'rolling-array'],
  statement:
    '假设你正在爬楼梯。需要 `n` 阶你才能到达楼顶。每次你可以爬 `1` 或 `2` 个台阶。你有多少种**不同的方法**可以爬到楼顶呢？',
  examples: [
    { input: 'n = 2', output: '2', note: '1+1 / 2' },
    { input: 'n = 3', output: '3', note: '1+1+1 / 1+2 / 2+1' },
    { input: 'n = 5', output: '8', note: '斐波那契数列：1,2,3,5,8' },
  ],
  constraints: [
    '1 ≤ n ≤ 45',
  ],
  intuition:
    '到达第 i 阶要么从 i-1 阶迈 1 步上来、要么从 i-2 阶迈 2 步上来，方案数互不相交，所以 dp[i] = dp[i-1] + dp[i-2]——这就是斐波那契。只依赖前两个状态，可以用两个滚动变量把空间压到 O(1)。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(1)' },
  microQuestions: [
    {
      id: 'climbing-stairs.q1',
      prompt: '`dp[i]` 最自然的状态定义是？',
      options: [
        { id: 'a', text: '到达第 i 阶的不同方法数' },
        { id: 'b', text: '前 i 步走过的总台阶数' },
        { id: 'c', text: '是否能到达第 i 阶（布尔）' },
        { id: 'd', text: '走第 i 步时的累计代价' },
      ],
      answer: 'a',
      explain:
        '本题问"方法数"，状态就是"到达第 i 阶的方法数"。状态定义对了，转移方程几乎自动浮现：到达 i 的最后一步只可能来自 i-1（迈 1 步）或 i-2（迈 2 步）。',
      tags: ['naming', 'invariant'],
    },
    {
      id: 'climbing-stairs.q2',
      prompt: '转移方程应当是？',
      codeContext: code,
      highlightLine: 7,
      options: [
        { id: 'a', text: 'dp[i] = dp[i-1] * dp[i-2]' },
        { id: 'b', text: 'dp[i] = dp[i-1] + dp[i-2]' },
        { id: 'c', text: 'dp[i] = max(dp[i-1], dp[i-2]) + 1' },
        { id: 'd', text: 'dp[i] = dp[i-1] + 1' },
      ],
      answer: 'b',
      explain:
        '"加法原理"：到达 i 的所有方法 = (到 i-1 后再走 1 步的方法) + (到 i-2 后再走 2 步的方法)，两类互不相交，所以是和不是积。',
      tags: ['invariant'],
    },
    {
      id: 'climbing-stairs.q3',
      prompt: '若以 `dp[i]` = 到达第 i 阶的方法数定义状态，初始值最稳的写法是？',
      options: [
        { id: 'a', text: 'dp[0] = 0, dp[1] = 1' },
        { id: 'b', text: 'dp[1] = 1, dp[2] = 2（从 i=3 开始递推）' },
        { id: 'c', text: 'dp[0] = 1, dp[1] = 1（把 dp[0] 当"空走法"）' },
        { id: 'd', text: 'b 与 c 都对' },
      ],
      answer: 'd',
      explain:
        '两种约定都常见：① dp[1]=1, dp[2]=2 直接来自题意；② dp[0]=1（空集只有一种"什么都不做"的走法）让 dp[2]=dp[1]+dp[0]=1+1=2 也成立。关键是**保持自洽**——选了一种就不能中途换。',
      tags: ['boundary'],
    },
    {
      id: 'climbing-stairs.q4',
      prompt: '为什么不能把 `dp[0]` 设为 0？',
      options: [
        { id: 'a', text: '因为 0 阶就是起点，方法数是 0 听起来合理但破坏 dp[2]=dp[1]+dp[0]=1+0=1 与题意 dp[2]=2 矛盾' },
        { id: 'b', text: '0 阶不是合法状态，必须从 1 开始' },
        { id: 'c', text: 'Python 不允许下标为 0' },
        { id: 'd', text: '会触发斐波那契的边界异常' },
      ],
      answer: 'a',
      explain:
        '关键是状态定义的一致性。如果用 dp[0]=1（"空走法"算 1 种），那 dp[2]=dp[1]+dp[0]=1+1=2 与直观一致；若用 dp[0]=0 则 dp[2]=1 错。所以选 dp[0]=1 或干脆从 dp[1], dp[2] 起步。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'climbing-stairs.q5',
      prompt: '此题与斐波那契数列的关系？',
      options: [
        { id: 'a', text: '完全无关，只是巧合' },
        { id: 'b', text: '递推式相同 F(i)=F(i-1)+F(i-2)，但起点不同：本题 dp[1]=1, dp[2]=2，相当于 F(n+1)' },
        { id: 'c', text: '是斐波那契的平方' },
        { id: 'd', text: '是斐波那契的反向' },
      ],
      answer: 'b',
      explain:
        '经典斐波那契 1,1,2,3,5,8...；本题 1,2,3,5,8...——同一递推式，只是起点错开一位。理解到这一步说明你抓到了"问题本质 = 加法原理产生的二阶递推"。',
      tags: ['invariant'],
    },
    {
      id: 'climbing-stairs.q6',
      prompt: '空间优化的核心观察是？',
      codeContext: code,
      highlightLine: 5,
      options: [
        { id: 'a', text: 'dp[i] 只依赖 dp[i-1] 和 dp[i-2]，前面的状态都不再需要' },
        { id: 'b', text: 'dp 数组可以反着填' },
        { id: 'c', text: '可以并行计算' },
        { id: 'd', text: '使用更小的整数类型' },
      ],
      answer: 'a',
      explain:
        '"只依赖最近 k 个状态"是一切滚动数组优化的入口。本题 k=2，所以只需 prev1, prev2 两个变量，空间从 O(n) 降到 O(1)。',
      tags: ['complexity'],
    },
    {
      id: 'climbing-stairs.q7',
      prompt: '滚动更新 `prev2, prev1 = prev1, cur` 这种 Python 同行赋值的关键是？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: '右侧表达式先全部求值，再统一赋值——不会有覆盖问题' },
        { id: 'b', text: '先算左 prev2 再算 prev1，顺序敏感' },
        { id: 'c', text: 'Python 内部用临时元组实现，性能差' },
        { id: 'd', text: '需要先 `import operator`' },
      ],
      answer: 'a',
      explain:
        '元组同时赋值是 Python 滚动数组的最佳实践。等价的 C 风格写法要先 tmp = prev1; prev1 = cur; prev2 = tmp。同行赋值更短也更不易错。',
      tags: ['pythonism', 'syntax'],
    },
    {
      id: 'climbing-stairs.q8',
      prompt: '若不做空间优化、用一维 dp 数组，下面循环范围正确的是？',
      options: [
        { id: 'a', text: 'for i in range(n): ...' },
        { id: 'b', text: 'for i in range(1, n): ...' },
        { id: 'c', text: 'for i in range(2, n+1): ...（dp[0]=1, dp[1]=1）或 for i in range(3, n+1):（dp[1]=1, dp[2]=2）' },
        { id: 'd', text: 'for i in range(0, n+2): ...' },
      ],
      answer: 'c',
      explain:
        '循环起点取决于初值约定：若初始化了 dp[0], dp[1]，循环从 i=2 起；若初始化了 dp[1], dp[2]，循环从 i=3 起。终点要 +1 才包含 n（Python range 右开区间）。',
      tags: ['boundary', 'syntax'],
    },
    {
      id: 'climbing-stairs.q9',
      prompt: '时间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(log n)' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(2^n)（递归暴力）' },
      ],
      answer: 'c',
      explain:
        '一次循环 n-2 步，每步常数操作。注意 d 是**未优化的递归暴力**，每次分裂成两支，重复计算大量子问题，正是引入 DP 的初衷。',
      tags: ['complexity'],
    },
    {
      id: 'climbing-stairs.q10',
      prompt: '空间复杂度（用滚动变量版）是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(n)' },
        { id: 'c', text: 'O(log n)' },
        { id: 'd', text: 'O(n²)' },
      ],
      answer: 'a',
      explain:
        '只用 prev1, prev2, cur 三个变量。一维数组版是 O(n)，矩阵快速幂版还能做到 O(log n) 时间——但工程上滚动 O(1) 已经最实用。',
      tags: ['complexity'],
    },
    {
      id: 'climbing-stairs.q11',
      prompt: '若题目变成"每次能爬 1, 2, 或 3 阶"，转移方程会变成？',
      options: [
        { id: 'a', text: 'dp[i] = dp[i-1] + dp[i-2]' },
        { id: 'b', text: 'dp[i] = dp[i-1] + dp[i-2] + dp[i-3]' },
        { id: 'c', text: 'dp[i] = dp[i-1] * dp[i-2] * dp[i-3]' },
        { id: 'd', text: '题目无解' },
      ],
      answer: 'b',
      explain:
        '推广到"每次可走 k 中之一"：dp[i] = sum(dp[i - kj])。本题是 k=2 的特例。这种"加法原理 + 几个上一状态"的结构是一维 DP 的典型形态。',
      tags: ['invariant'],
    },
    {
      id: 'climbing-stairs.q12',
      prompt: '关于"为什么用 DP 而不是直接递归"，最准确的说法是？',
      options: [
        { id: 'a', text: '递归一定比 DP 慢' },
        { id: 'b', text: '朴素递归 O(2^n) 因子问题重复求解；DP 用记忆化或自底向上把它降到 O(n)' },
        { id: 'c', text: '递归不能算斐波那契' },
        { id: 'd', text: 'Python 不允许深度超过 100 的递归' },
      ],
      answer: 'b',
      explain:
        '朴素 climb(n) = climb(n-1) + climb(n-2) 会重复算 climb(n-3), climb(n-4)... 形成指数膨胀。DP 的本质是"记忆化 + 拓扑序求解"，把指数压成线性。这是 DP 与暴力递归最重要的分水岭。',
      tags: ['complexity'],
    },
  ],
}

export default problem
