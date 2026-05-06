import type { Problem } from '../types'

const code = `def minDistance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    # dp[i][j] = 把 word1[:i] 变成 word2[:j] 的最少操作数
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # 第一列:word2 是空,word1[:i] 全删 → i 次
    for i in range(m + 1):
        dp[i][0] = i
    # 第一行:word1 是空,要插入 j 次得到 word2[:j]
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 末位相同,不需要操作
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # 替换 word1[i-1] 为 word2[j-1]
                    dp[i - 1][j],      # 删除 word1[i-1]
                    dp[i][j - 1],      # 在 word1 末尾插入 word2[j-1]
                )
    return dp[m][n]`

export const problem: Problem = {
  id: 'edit-distance',
  leetcodeNo: 72,
  title: { zh: '编辑距离', en: 'Edit Distance' },
  difficulty: 'medium',
  pattern: 'dp-2d',
  tags: ['dp', 'string', 'levenshtein'],
  statement:
    '给你两个单词 `word1` 和 `word2`，请返回将 `word1` 转换成 `word2` 所使用的**最少操作数**。\n\n你可以对一个单词进行如下三种操作：\n- 插入一个字符\n- 删除一个字符\n- 替换一个字符',
  examples: [
    { input: 'word1 = "horse", word2 = "ros"', output: '3', note: 'horse → rorse(替换 h→r) → rose(删 r) → ros(删 e)' },
    { input: 'word1 = "intention", word2 = "execution"', output: '5' },
    { input: 'word1 = "", word2 = "abc"', output: '3', note: '插入 3 次' },
  ],
  constraints: [
    '0 ≤ word1.length, word2.length ≤ 500',
    'word1 与 word2 由小写英文字母组成',
  ],
  intuition:
    '`dp[i][j]` = 把 `word1[:i]` 变成 `word2[:j]` 的最少操作。考虑末位 word1[i-1] vs word2[j-1]：① 相同 → 不用操作，dp[i][j] = dp[i-1][j-1]。② 不同 → 取三种操作中最少的：替换 dp[i-1][j-1]+1、删除 word1 末位 dp[i-1][j]+1、给 word1 末尾插入 word2[j-1] 即 dp[i][j-1]+1。第一行第一列分别对应"全插"和"全删"。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(m·n)', space: 'O(m·n)（可压成 O(min(m,n))）' },
  microQuestions: [
    {
      id: 'edit-distance.q1',
      prompt: '`dp[i][j]` 状态定义最准确的是？',
      options: [
        { id: 'a', text: '把 word1[:i] 转成 word2[:j] 的最少操作次数' },
        { id: 'b', text: 'word1 的前 i 个与 word2 的前 j 个是否相等' },
        { id: 'c', text: 'word1[:i] 与 word2[:j] 共享的字符数' },
        { id: 'd', text: 'word1 长度减去 word2 长度' },
      ],
      answer: 'a',
      explain:
        '最少操作数 = 编辑距离。最终答案是 dp[m][n]——整段 word1 转成整段 word2。注意是"操作次数"不是"差异字符数"。',
      tags: ['naming', 'invariant'],
    },
    {
      id: 'edit-distance.q2',
      prompt: '当 word1[i-1] == word2[j-1] 时，转移是？',
      codeContext: code,
      highlightLine: 14,
      options: [
        { id: 'a', text: 'dp[i][j] = dp[i-1][j-1]（不需要操作）' },
        { id: 'b', text: 'dp[i][j] = dp[i-1][j-1] + 1' },
        { id: 'c', text: 'dp[i][j] = min(dp[i-1][j], dp[i][j-1])' },
        { id: 'd', text: 'dp[i][j] = 0' },
      ],
      answer: 'a',
      explain:
        '末位相同 → 这一对字符不需要任何操作，前缀的最优方案直接继承。注意**没有 +1**——这是与 LCS 转移的对应（LCS 是 +1 因为在数长度，编辑距离是不变因为在数操作）。',
      tags: ['invariant'],
    },
    {
      id: 'edit-distance.q3',
      prompt: '末位不同时三个候选 `dp[i-1][j-1]`, `dp[i-1][j]`, `dp[i][j-1]` 分别对应？',
      codeContext: code,
      highlightLine: 17,
      options: [
        { id: 'a', text: '替换、删除、插入' },
        { id: 'b', text: '插入、删除、替换' },
        { id: 'c', text: '都对应替换' },
        { id: 'd', text: '都对应删除' },
      ],
      answer: 'a',
      explain:
        '记忆口诀："三角对应三操作"。dp[i-1][j-1] = 已经把 word1[:i-1] 变成了 word2[:j-1]，再把第 i 个字符**替换**为 word2[j-1]；dp[i-1][j] = 已经把 word1[:i-1] 变成了 word2[:j]，再**删除** word1[i-1]；dp[i][j-1] = 已经把 word1[:i] 变成了 word2[:j-1]，再**插入** word2[j-1]。',
      tags: ['invariant'],
    },
    {
      id: 'edit-distance.q4',
      prompt: '为什么"在 word1 末尾插入 word2[j-1]"对应 `dp[i][j-1]` 而不是 `dp[i-1][j]`？',
      options: [
        { id: 'a', text: '插入后 word1 长度变为 i+1，最末位是新插入的 word2[j-1] 与 word2[j-1] 自然匹配；剩余子问题是 word1[:i] vs word2[:j-1]' },
        { id: 'b', text: '题目限制' },
        { id: 'c', text: '随便对应' },
        { id: 'd', text: '插入和删除等价' },
      ],
      answer: 'a',
      explain:
        '插入操作"消耗"了 word2 的一个字符（j-1），但 word1 仍要全部用上（保持 i 不变）。理解这种"哪一边的下标退一格 = 哪一边消化了一字符"是双串 DP 的关键直觉。',
      tags: ['invariant'],
    },
    {
      id: 'edit-distance.q5',
      prompt: '边界 `dp[i][0] = i` 含义？',
      codeContext: code,
      highlightLine: 7,
      options: [
        { id: 'a', text: 'word2 为空，要把 word1[:i] 全删,操作数 = i' },
        { id: 'b', text: 'word1 全部已匹配' },
        { id: 'c', text: 'word2 为空时不需要操作' },
        { id: 'd', text: '题目要求' },
      ],
      answer: 'a',
      explain:
        '把任何串变成空串就是删完。这是边界的物理意义。如果你写成 dp[i][0] = 0 会得到错误答案——会认为变成空串免费。',
      tags: ['boundary'],
    },
    {
      id: 'edit-distance.q6',
      prompt: '边界 `dp[0][j] = j` 含义？',
      codeContext: code,
      highlightLine: 10,
      options: [
        { id: 'a', text: 'word1 为空，要插入 j 次得到 word2[:j]' },
        { id: 'b', text: 'word1 完全匹配' },
        { id: 'c', text: 'word2 为空' },
        { id: 'd', text: '随便' },
      ],
      answer: 'a',
      explain:
        '把空串变成长度 j 的串就是插 j 次。和上一题对偶：第一列对应"全删"、第一行对应"全插"。这两条边界是编辑距离 DP 的"骨架"。',
      tags: ['boundary'],
    },
    {
      id: 'edit-distance.q7',
      prompt: '若 `word1 == word2`，DP 表的对角线 dp[i][i] 全部应该是？',
      options: [
        { id: 'a', text: '0（前缀相同→不需要操作）' },
        { id: 'b', text: 'i' },
        { id: 'c', text: '随机' },
        { id: 'd', text: 'len(word1)' },
      ],
      answer: 'a',
      explain:
        '"前缀完全相同"对应 dp[i][i] = dp[i-1][i-1] = ... = dp[0][0] = 0。可以用这个性质来调试：若你的 dp[i][i] 不是全 0，那一定哪里写错了。',
      tags: ['invariant', 'boundary'],
    },
    {
      id: 'edit-distance.q8',
      prompt: '为什么"末位相同"分支不需要把"删/插"也加入 min 比较？',
      options: [
        { id: 'a', text: '因为 dp[i-1][j-1] 已经 ≤ dp[i-1][j]+1, dp[i][j-1]+1（单调性）所以无需重复' },
        { id: 'b', text: '因为题目限制' },
        { id: 'c', text: '严格说应该加，否则会错' },
        { id: 'd', text: '加了会出错' },
      ],
      answer: 'a',
      explain:
        '虽然写成 `dp[i][j] = min(dp[i-1][j-1], dp[i-1][j]+1, dp[i][j-1]+1)` 也对，但 dp[i-1][j-1] 永远 ≤ 后两者（每多一个字符差至多需要一次操作）。简洁写法更清晰。',
      tags: ['invariant', 'complexity'],
    },
    {
      id: 'edit-distance.q9',
      prompt: '编辑距离最少为 `|m - n|`，因为？',
      options: [
        { id: 'a', text: '长度差必须靠插入或删除补上,每步只能改变长度 1' },
        { id: 'b', text: '题目要求' },
        { id: 'c', text: '替换可以同时改长度' },
        { id: 'd', text: 'Python 字符串特性' },
      ],
      answer: 'a',
      explain:
        '插入和删除各改变长度 1，替换不改长度。所以 dp[m][n] ≥ |m-n|。这个下界经常用于剪枝（如 K-编辑距离判定）。',
      tags: ['invariant'],
    },
    {
      id: 'edit-distance.q10',
      prompt: '时间复杂度是？',
      options: [
        { id: 'a', text: 'O(m + n)' },
        { id: 'b', text: 'O(m · n)' },
        { id: 'c', text: 'O(min(m,n)·k)（k 是允许的最大编辑距离）' },
        { id: 'd', text: 'O(2^(m+n))' },
      ],
      answer: 'b',
      explain:
        '标准 DP 是 O(mn)。c 是 Ukkonen 改进算法（仅当 k 较小时优）；d 是朴素递归不带记忆化。题目数据规模 500×500 = 25 万 OK。',
      tags: ['complexity'],
    },
    {
      id: 'edit-distance.q11',
      prompt: '空间能否压缩？',
      options: [
        { id: 'a', text: '能压成 O(min(m,n)) ——只保留两行(或一行+一个prev变量)' },
        { id: 'b', text: '不能' },
        { id: 'c', text: '能压成 O(1)' },
        { id: 'd', text: '只能压成 O(m+n)' },
      ],
      answer: 'a',
      explain:
        '转移只依赖 dp[i-1][..] 和 dp[i][..]——两行足够。和 LCS 一样，若需要"还原编辑序列"就不能压。',
      tags: ['complexity'],
    },
    {
      id: 'edit-distance.q12',
      prompt: '若题目改成"只允许插入和删除"（不允许替换），与 LCS 的关系是？',
      options: [
        { id: 'a', text: '答案 = m + n - 2*LCS（共享部分免改,各自独有部分一插一删）' },
        { id: 'b', text: '答案 = m + n' },
        { id: 'c', text: '答案 = max(m,n)' },
        { id: 'd', text: '答案 = LCS' },
      ],
      answer: 'a',
      explain:
        '面试常见关联题（LC 583 两个字符串的删除操作）。无替换的编辑距离正好是 m + n - 2·LCS——LCS 部分两边都保留，剩下各自的字符要么删要么插。从 LCS 直接推。',
      tags: ['invariant'],
    },
    {
      id: 'edit-distance.q13',
      prompt: '若三种操作的"代价不同"（如删除 cost=1、插入 cost=2、替换 cost=3），DP 框架要怎么改？',
      options: [
        { id: 'a', text: '把 +1 换成各自代价,框架不变' },
        { id: 'b', text: '需要换算法' },
        { id: 'c', text: '不能改' },
        { id: 'd', text: '需要图论 BFS' },
      ],
      answer: 'a',
      explain:
        '这正是 DP 比较"通用"的好处——加权版本只需把 +1 换成相应代价。这种推广在生物序列比对（带权 Levenshtein, Needleman-Wunsch）中常用。',
      tags: ['invariant'],
    },
  ],
}

export default problem
