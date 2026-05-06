import type { Problem } from '../types'

const code = `def longestCommonSubsequence(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    # dp[i][j] 表示 text1[:i] 与 text2[:j] 的 LCS 长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                # 末位字符相同 → 必属于 LCS,前缀各退一格
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # 末位不同 → 至少一方不能用,取较优者
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]`

export const problem: Problem = {
  id: 'longest-common-subsequence',
  leetcodeNo: 1143,
  title: { zh: '最长公共子序列', en: 'Longest Common Subsequence' },
  difficulty: 'medium',
  pattern: 'dp-2d',
  tags: ['dp', 'string'],
  statement:
    '给定两个字符串 `text1` 和 `text2`，返回这两个字符串的**最长公共子序列**的长度。如果不存在**公共子序列**，返回 `0`。\n\n一个字符串的**子序列**是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。\n\n两个字符串的**公共子序列**是这两个字符串所共同拥有的子序列。',
  examples: [
    { input: 'text1 = "abcde", text2 = "ace"', output: '3', note: 'LCS 为 "ace"' },
    { input: 'text1 = "abc", text2 = "abc"', output: '3' },
    { input: 'text1 = "abc", text2 = "def"', output: '0' },
  ],
  constraints: [
    '1 ≤ text1.length, text2.length ≤ 1000',
    'text1 与 text2 仅由小写英文字符组成',
  ],
  intuition:
    '`dp[i][j]` = `text1[:i]` 与 `text2[:j]` 的 LCS 长度。考察第 i 个与第 j 个字符：① 相同 → 一定能进入 LCS，dp[i][j] = dp[i-1][j-1] + 1；② 不同 → 至少有一方放弃自己的末尾，dp[i][j] = max(dp[i-1][j], dp[i][j-1])。第一行第一列全 0（任何串与空串的 LCS 是空）。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(m·n)', space: 'O(m·n)（可压成 O(min(m,n))）' },
  microQuestions: [
    {
      id: 'lcs.q1',
      prompt: '`dp[i][j]` 的状态定义最准确的是？',
      options: [
        { id: 'a', text: 'text1[:i] 与 text2[:j] 的 LCS 长度' },
        { id: 'b', text: 'text1 第 i 字符与 text2 第 j 字符是否相等' },
        { id: 'c', text: 'text1[i:] 与 text2[j:] 的 LCS' },
        { id: 'd', text: 'text1[:i] 是否是 text2[:j] 的子串' },
      ],
      answer: 'a',
      explain:
        '前缀视角是双串 DP 的标准：dp[i][j] 关心两个**前缀**的子结构最优值。最终答案在 dp[m][n] —— 完整两串的 LCS。',
      tags: ['naming', 'invariant'],
    },
    {
      id: 'lcs.q2',
      prompt: '当 text1[i-1] == text2[j-1] 时（注意索引偏移）转移为？',
      codeContext: code,
      highlightLine: 9,
      options: [
        { id: 'a', text: 'dp[i][j] = dp[i-1][j-1] + 1' },
        { id: 'b', text: 'dp[i][j] = dp[i-1][j] + 1' },
        { id: 'c', text: 'dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + 1' },
        { id: 'd', text: 'dp[i][j] = dp[i-1][j-1]' },
      ],
      answer: 'a',
      explain:
        '可证明：当末尾字符相同时，**最优 LCS 一定包含这对字符**（若不包含则可以加进去得到更长 LCS，矛盾）。所以两边各退一格再 +1。这是双串 DP 最重要的"匹配则齐退"模式。',
      tags: ['invariant'],
    },
    {
      id: 'lcs.q3',
      prompt: '当末尾不同时，转移 `max(dp[i-1][j], dp[i][j-1])` 的含义？',
      codeContext: code,
      highlightLine: 12,
      options: [
        { id: 'a', text: '末尾不同时至少一方末位不能进入 LCS——枚举"放弃哪一方的末位"，取较好' },
        { id: 'b', text: '随便选一边' },
        { id: 'c', text: '需要再加 1' },
        { id: 'd', text: 'min 而非 max' },
      ],
      answer: 'a',
      explain:
        '末位不同 ⇒ 不可能两个末位都在 LCS 末尾。要么放弃 text1[i-1]（→ dp[i-1][j]），要么放弃 text2[j-1]（→ dp[i][j-1]），取较好者。注意**没有 +1** —— 这一对没匹配上。',
      tags: ['invariant'],
    },
    {
      id: 'lcs.q4',
      prompt: '为什么不需要考虑 `dp[i-1][j-1]` 这一项作为 max 候选？',
      options: [
        { id: 'a', text: '因为 dp[i-1][j-1] ≤ dp[i-1][j] 且 ≤ dp[i][j-1]，已被覆盖' },
        { id: 'b', text: '因为题目限制' },
        { id: 'c', text: '因为会出错' },
        { id: 'd', text: '因为 dp[i-1][j-1] 是不合法状态' },
      ],
      answer: 'a',
      explain:
        '由 dp 单调性（前缀更长 LCS 不会更短），dp[i-1][j-1] ≤ dp[i-1][j]，所以加它进 max 不会改变结果。但**写上不会错**，只是冗余——经典面试追问点。',
      tags: ['invariant'],
    },
    {
      id: 'lcs.q5',
      prompt: '初始化 dp 大小为 `(m+1) × (n+1)`、第一行第一列全 0 的理由？',
      codeContext: code,
      highlightLine: 4,
      options: [
        { id: 'a', text: '让"空串"作为合法状态，避免单独写 i=0 / j=0 的边界判断' },
        { id: 'b', text: '为了多消耗内存' },
        { id: 'c', text: '题目要求' },
        { id: 'd', text: '为了对齐 1-indexed 数组' },
      ],
      answer: 'a',
      explain:
        '空串与任何串的 LCS = 0 是天然边界。把状态从 1 编号到 m/n，dp[i-1][j-1] 这种回退**总是**合法（不会越界）。这是 DP 的"哨兵行/列"技巧。',
      tags: ['boundary'],
    },
    {
      id: 'lcs.q6',
      prompt: '注意索引：`text1[i-1]` 而不是 `text1[i]`，原因？',
      codeContext: code,
      highlightLine: 7,
      options: [
        { id: 'a', text: 'dp 用 1-based（含哨兵），text1 是 0-based，对应字符是 text1[i-1]' },
        { id: 'b', text: 'Python 的字符串索引规则' },
        { id: 'c', text: '随便取' },
        { id: 'd', text: '为了越界' },
      ],
      answer: 'a',
      explain:
        '哨兵行/列让 dp 的索引比字符串多 1。当 i=1 对应 text1 的第 0 个字符——即 text1[0]，所以是 text1[i-1]。这种"DP 1-indexed + 字符串 0-indexed"的对齐 bug 是双串题最常见错误。',
      tags: ['boundary', 'syntax'],
    },
    {
      id: 'lcs.q7',
      prompt: '若需要**还原 LCS 字符串**（不只是长度），思路是？',
      options: [
        { id: 'a', text: '从 dp[m][n] 反推：相同则取该字符并向 (i-1,j-1) 回溯;否则向 dp 较大的一侧回溯' },
        { id: 'b', text: '不可能还原' },
        { id: 'c', text: '需要重新跑一遍 DP' },
        { id: 'd', text: '只能存一份 LCS' },
      ],
      answer: 'a',
      explain:
        '回溯算法：从 (m,n) 向 (0,0) 走，若当前字符匹配则把字符前置加入答案、回 (i-1,j-1)；否则朝 dp 值更大的一边走。注意结果要反转（从尾到头收集）。多解时这种回溯只给出一条 LCS。',
      tags: ['data-structure'],
    },
    {
      id: 'lcs.q8',
      prompt: '`dp = [[0]*(n+1) for _ in range(m+1)]` 与 `dp = [[0]*(n+1)]*(m+1)` 的差别？',
      options: [
        { id: 'a', text: '后者所有行是同一引用——改一处全变，经典 Python 陷阱' },
        { id: 'b', text: '前者更慢' },
        { id: 'c', text: '完全等价' },
        { id: 'd', text: '后者会报错' },
      ],
      answer: 'a',
      explain:
        '`[x] * m` 复制引用而非对象。二维数组必须用列表推导式生成 m 个独立行。这条规则在所有二维 DP 题里都同样重要。',
      tags: ['pythonism', 'boundary'],
    },
    {
      id: 'lcs.q9',
      prompt: '时间复杂度是？',
      options: [
        { id: 'a', text: 'O(m + n)' },
        { id: 'b', text: 'O(m · n)' },
        { id: 'c', text: 'O(m² · n²)' },
        { id: 'd', text: 'O(2^(m+n))' },
      ],
      answer: 'b',
      explain:
        '每个 dp[i][j] 常数时间，总 m·n 个状态。注意没有更优的通用算法（Hunt-Szymanski 仅在某些极端稀疏匹配下更快）。',
      tags: ['complexity'],
    },
    {
      id: 'lcs.q10',
      prompt: '空间能压缩到？',
      options: [
        { id: 'a', text: 'O(min(m, n)) —— 只保留两行（或一行 + 一个 prev 变量）' },
        { id: 'b', text: 'O(1)' },
        { id: 'c', text: 'O(log n)' },
        { id: 'd', text: '不能压缩' },
      ],
      answer: 'a',
      explain:
        '转移只依赖 dp[i-1][..] 与 dp[i][..]——两行即可。再用 prev 变量保存 dp[i-1][j-1] 可压到一行。但若要**还原 LCS**则不能压（需保留全表）。',
      tags: ['complexity'],
    },
    {
      id: 'lcs.q11',
      prompt: 'LCS 与 "最长公共子串"（连续）的关键区别是？',
      options: [
        { id: 'a', text: '子序列允许跳过中间字符;子串必须连续' },
        { id: 'b', text: '完全相同' },
        { id: 'c', text: '子串问题不能用 DP' },
        { id: 'd', text: 'Python 的实现完全一样' },
      ],
      answer: 'a',
      explain:
        '"连续"约束让转移变成：相同时 dp[i-1][j-1]+1，**不同时直接置 0**（断开了）。最长公共子串答案是 max(全表)，而非 dp[m][n]。这两题状态相同但转移和答案位置都不同。',
      tags: ['invariant'],
    },
    {
      id: 'lcs.q12',
      prompt: 'LCS 与"最短公共超序列长度"（SCS, LC 1092）的关系？',
      options: [
        { id: 'a', text: 'SCS = m + n - LCS（LCS 部分共享，其他各贡献一次）' },
        { id: 'b', text: '它们无关' },
        { id: 'c', text: 'SCS = LCS' },
        { id: 'd', text: 'SCS = m * n / LCS' },
      ],
      answer: 'a',
      explain:
        '面试经典对偶关系：让 SCS 最短就是让公共部分最长（LCS）。具体：SCS 长度 = m + n - LCS。这表明 LCS 是双串 DP 的"母题"，SCS、编辑距离都是其变体。',
      tags: ['invariant'],
    },
  ],
}

export default problem
