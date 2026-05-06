import type { Problem } from '../types'

const code = `def rob(nums: list[int]) -> int:
    if not nums:
        return 0
    n = len(nums)
    if n == 1:
        return nums[0]
    # dp[i] = 偷到第 i 间为止能拿到的最大金额（i 偷不偷都行）
    prev2, prev1 = nums[0], max(nums[0], nums[1])
    for i in range(2, n):
        cur = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, cur
    return prev1`

export const problem: Problem = {
  id: 'house-robber',
  leetcodeNo: 198,
  title: { zh: '打家劫舍', en: 'House Robber' },
  difficulty: 'medium',
  pattern: 'dp-1d',
  tags: ['dp', 'rolling-array'],
  statement:
    '你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是**相邻的房屋装有相互连通的防盗系统**，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。\n\n给定一个代表每个房屋存放金额的非负整数数组，计算你**不触动警报装置**的情况下，一夜之内能够偷窃到的**最高金额**。',
  examples: [
    { input: 'nums = [1,2,3,1]', output: '4', note: '偷 nums[0]=1 + nums[2]=3 = 4' },
    { input: 'nums = [2,7,9,3,1]', output: '12', note: '偷 nums[0]+nums[2]+nums[4] = 2+9+1 = 12' },
    { input: 'nums = [2,1,1,2]', output: '4', note: '偷 nums[0]+nums[3] = 2+2 = 4' },
  ],
  constraints: [
    '1 ≤ nums.length ≤ 100',
    '0 ≤ nums[i] ≤ 400',
  ],
  intuition:
    '到第 i 间房有两种选择：① 不偷它 → 收益 = dp[i-1]；② 偷它 → 必须放弃 i-1，收益 = dp[i-2] + nums[i]。两者取大。这是「相邻互斥型」DP 的母题。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(1)' },
  microQuestions: [
    {
      id: 'house-robber.q1',
      prompt: '`dp[i]` 的状态定义最准确的是？',
      options: [
        { id: 'a', text: '前 i+1 间房中能偷到的最大金额（i 偷不偷由内部最优决定）' },
        { id: 'b', text: '"必须偷 i"时的最大金额' },
        { id: 'c', text: '到第 i 间为止已偷的房屋数量' },
        { id: 'd', text: '从 i 开始往后能偷到的金额' },
      ],
      answer: 'a',
      explain:
        '状态定义有两种流派——本题用的是 a：dp[i] = 「考虑到 i 为止」的全局最优，i 偷或不偷都被两支转移涵盖。b 也能做但需要二维状态 dp[i][0/1]。',
      tags: ['naming', 'invariant'],
    },
    {
      id: 'house-robber.q2',
      prompt: '转移方程的两支分别表示什么？',
      codeContext: code,
      highlightLine: 10,
      options: [
        { id: 'a', text: 'dp[i-1]：不偷第 i 间；dp[i-2] + nums[i]：偷第 i 间' },
        { id: 'b', text: 'dp[i-1]：偷第 i 间；dp[i-2] + nums[i]：不偷' },
        { id: 'c', text: '都是"偷"的情况，只是用不同前缀' },
        { id: 'd', text: '都是"不偷"的情况' },
      ],
      answer: 'a',
      explain:
        '关键约束："偷 i 必须放弃 i-1"。不偷 i 时上一步随便（直接继承 dp[i-1]）；偷 i 时必须从 dp[i-2] 接力，加上 nums[i]。',
      tags: ['invariant'],
    },
    {
      id: 'house-robber.q3',
      prompt: '初始值 `dp[0] = nums[0]`、`dp[1] = max(nums[0], nums[1])` 这样写的理由？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: '只有一间时只能偷它；只有两间时由于相邻只能选大的那间' },
        { id: 'b', text: '随便填即可' },
        { id: 'c', text: '是题目硬编码的边界' },
        { id: 'd', text: '为了对齐 nums 索引' },
      ],
      answer: 'a',
      explain:
        'dp[0] 直观就是 nums[0]。dp[1] 是"前两间"的最优：因为相邻不能同偷，必须二选一，所以 max(nums[0], nums[1])。这两个初值正好为 i≥2 的转移提供 dp[i-1] 与 dp[i-2]。',
      tags: ['boundary'],
    },
    {
      id: 'house-robber.q4',
      prompt: '若把 `dp[1] = max(nums[0], nums[1])` 错写成 `dp[1] = nums[1]`，会出什么问题？',
      options: [
        { id: 'a', text: 'nums = [3, 1] 时返回 1 而非 3，丢解' },
        { id: 'b', text: '没有问题' },
        { id: 'c', text: '只在 nums[0] = 0 时出错' },
        { id: 'd', text: '运行时异常' },
      ],
      answer: 'a',
      explain:
        '反例：nums=[3,1] 应返回 3（偷第一间）。如果 dp[1]=1，那么 i=2 之后的 dp 全部在 1 之上扩展，永远丢掉 nums[0]=3 的可能。这就是为什么 dp[1] 必须取 max，让"只偷 0 不偷 1"也保留在前缀最优中。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'house-robber.q5',
      prompt: '`prev2, prev1 = prev1, cur` 在 Python 里能正确滚动的原因？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: '右边先求值成 (prev1, cur) 元组，再一次性解包赋给左边' },
        { id: 'b', text: 'Python 严格从左到右赋值' },
        { id: 'c', text: '需要 `nonlocal` 关键字' },
        { id: 'd', text: '需要 `global` 关键字' },
      ],
      answer: 'a',
      explain:
        '理解这一点能避免 90% 的滚动数组错误。如果用 C 风格 `prev2 = prev1; prev1 = cur` 也对；但写反顺序 `prev1 = cur; prev2 = prev1` 会出错——所以同行赋值更安全。',
      tags: ['pythonism', 'syntax'],
    },
    {
      id: 'house-robber.q6',
      prompt: '若 nums 全是 0，`max(prev1, prev2 + nums[i])` 还正确吗？',
      options: [
        { id: 'a', text: '正确，全程 0，最终返回 0' },
        { id: 'b', text: '不正确，需要特判' },
        { id: 'c', text: '会陷入死循环' },
        { id: 'd', text: '只在 n>10 时正确' },
      ],
      answer: 'a',
      explain:
        '初值都是 0，每步 max(0, 0+0)=0，正确返回 0。DP 的优雅之处：转移方程对边界自然兼容，不需要为 0 数组写特例。',
      tags: ['boundary'],
    },
    {
      id: 'house-robber.q7',
      prompt: '为什么不能用贪心"每次偷最大值"？',
      options: [
        { id: 'a', text: '反例 [2,1,1,2]：贪心先偷 2 再偷 2 没问题；但 [4,1,2,7,5,3,1] 中贪心从 7 开始可能阻断更优解' },
        { id: 'b', text: 'Python 不支持贪心' },
        { id: 'c', text: '贪心在该题等价于 DP' },
        { id: 'd', text: '贪心解需要排序' },
      ],
      answer: 'a',
      explain:
        '本题缺乏"局部最优 → 全局最优"的贪心选择性质。例如 [2,7,9,3,1]：贪偷 9 之后阻塞 7 与 3，但最优是 2+9+1=12 包含 9——巧合而已。换 [3,1,1,3,1] 贪心可能选两端 3+3=6，DP 同样得 6 但要靠枚举证明。所以稳定的方法是 DP。',
      tags: ['invariant'],
    },
    {
      id: 'house-robber.q8',
      prompt: '此题的"环形版"（LeetCode 213，首尾相接）正确思路是？',
      options: [
        { id: 'a', text: '把数组拼接后跑同一个 DP' },
        { id: 'b', text: '分两次 DP：① nums[0..n-2]（不偷最后一间）；② nums[1..n-1]（不偷第一间）；取 max' },
        { id: 'c', text: '直接调本题代码即可' },
        { id: 'd', text: '需要二维 DP' },
      ],
      answer: 'b',
      explain:
        '环形的核心矛盾："首和尾不能同时偷"。把它拆成两条线性 DP：要么放弃首，要么放弃尾，剩下都是普通打家劫舍。两次 max 即环形最优。这种"打破环依赖"是面试常考技巧。',
      tags: ['invariant'],
    },
    {
      id: 'house-robber.q9',
      prompt: '时间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(n)' },
        { id: 'c', text: 'O(n²)' },
        { id: 'd', text: 'O(n log n)' },
      ],
      answer: 'b',
      explain:
        '一次遍历 n 个房屋，每步常数次比较和加法。',
      tags: ['complexity'],
    },
    {
      id: 'house-robber.q10',
      prompt: '空间复杂度（滚动变量版）是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(n)' },
        { id: 'c', text: 'O(log n)' },
        { id: 'd', text: 'O(n²)' },
      ],
      answer: 'a',
      explain:
        '只保留 prev1, prev2, cur 三个变量。一维数组版是 O(n)。',
      tags: ['complexity'],
    },
    {
      id: 'house-robber.q11',
      prompt: '若 nums 含负数（题目变体，允许"罚钱"），上面代码哪一步可能错？',
      options: [
        { id: 'a', text: '初始化 dp[0]=nums[0] 仍对，但转移 max(prev1, prev2+nums[i]) 也对——其实没错' },
        { id: 'b', text: '需要把 max 改成 min' },
        { id: 'c', text: '需要禁止偷负值，强制 prev2+nums[i] 取 max(0, ...)' },
        { id: 'd', text: '需要二维 DP' },
      ],
      answer: 'c',
      explain:
        '原题保证 nums[i] ≥ 0 所以"偷一定不亏"。允许负值时，"什么都不偷"也是合法选择，应让 dp[i] 与 0 比较——比如换成 dp[i] = max(0, dp[i-1], dp[i-2] + nums[i])。这考察对题目隐含约束的敏感度。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'house-robber.q12',
      prompt: '若把状态扩展为 `dp[i][0/1]`（0=不偷 i, 1=偷 i），转移应是？',
      options: [
        { id: 'a', text: 'dp[i][0] = max(dp[i-1][0], dp[i-1][1]); dp[i][1] = dp[i-1][0] + nums[i]' },
        { id: 'b', text: 'dp[i][0] = dp[i-1][1]; dp[i][1] = dp[i-1][0]' },
        { id: 'c', text: 'dp[i][0] = dp[i-1][0]; dp[i][1] = dp[i-1][1] + nums[i]' },
        { id: 'd', text: '不能这样建模' },
      ],
      answer: 'a',
      explain:
        '"不偷 i" → 上一间偷不偷都行，取 max；"偷 i" → 上一间必须不偷。等价于一维版本，只是显式拆开两条状态线。这种"显式状态机"思维在股票系列（121/122/123/188）里更重要。',
      tags: ['invariant'],
    },
  ],
}

export default problem
