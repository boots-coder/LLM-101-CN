import type { Problem } from '../types'

const code = `def wordBreak(s: str, wordDict: list[str]) -> bool:
    word_set = set(wordDict)  # O(1) 查询
    max_len = max((len(w) for w in word_set), default=0)
    n = len(s)
    # dp[i] 表示 s[:i] 是否可被拆分;dp[0] = True 是「空串」边界
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        # 只看最近 max_len 个起点 j 即可——再往左 s[j:i] 不可能在字典里
        for j in range(max(0, i - max_len), i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[n]`

export const problem: Problem = {
  id: 'word-break',
  leetcodeNo: 139,
  title: { zh: '单词拆分', en: 'Word Break' },
  difficulty: 'medium',
  pattern: 'dp-1d',
  tags: ['dp', 'string', 'hash-set'],
  statement:
    '给你一个字符串 `s` 和一个字符串列表 `wordDict` 作为字典。请你判断**是否可以**利用字典中出现的单词拼接出 `s`。\n\n**注意**：不要求字典中出现的单词全部都使用，并且字典中的单词可以**重复使用**。',
  examples: [
    { input: 's = "leetcode", wordDict = ["leet","code"]', output: 'true', note: '"leet" + "code"' },
    { input: 's = "applepenapple", wordDict = ["apple","pen"]', output: 'true', note: '可以重复使用 apple' },
    { input: 's = "catsandog", wordDict = ["cats","dog","sand","and","cat"]', output: 'false' },
  ],
  constraints: [
    '1 ≤ s.length ≤ 300',
    '1 ≤ wordDict.length ≤ 1000',
    '1 ≤ wordDict[i].length ≤ 20',
    'wordDict 内单词互不相同',
  ],
  intuition:
    '`dp[i]` = s[:i] 是否可拆。要让 dp[i]=True，存在某个 j (0≤j<i) 使得 dp[j]=True 且 s[j:i] 在字典中——即「前 j 个可拆 + 后缀 s[j:i] 是单词」。把 wordDict 转成 set 让查询变 O(1)；再用「最长单词长度」剪枝，避免无效内层。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n · L)', space: 'O(n)' },
  microQuestions: [
    {
      id: 'word-break.q1',
      prompt: '`dp[i]` 最准确的状态定义是？',
      options: [
        { id: 'a', text: 's[:i]（前 i 个字符）能否被拆分为字典单词序列' },
        { id: 'b', text: 's[i] 是否是某个单词的首字母' },
        { id: 'c', text: '从 i 开始能否拆' },
        { id: 'd', text: '到 i 已用了多少个单词' },
      ],
      answer: 'a',
      explain:
        '"前缀是否可拆"是字符串 DP 的经典定义。注意是 s[:i]（不含 s[i]，长度恰为 i），所以 dp 数组大小是 n+1，最终答案是 dp[n]。',
      tags: ['naming', 'invariant'],
    },
    {
      id: 'word-break.q2',
      prompt: '边界 `dp[0] = True` 的含义？',
      codeContext: code,
      highlightLine: 7,
      options: [
        { id: 'a', text: '空串可"拆"——任何拆分都从空串开始' },
        { id: 'b', text: 'Python 默认值' },
        { id: 'c', text: '为了避免 IndexError' },
        { id: 'd', text: '题目要求' },
      ],
      answer: 'a',
      explain:
        '"空串视为可拆"是数学约定，类似空集是任何集合的子集。它让 dp[i] = (dp[0] and s[0:i] in word_set) 这种"整段就是一个单词"的情况自然成立——否则你需要单独 if 判断 i==某词长度。',
      tags: ['boundary'],
    },
    {
      id: 'word-break.q3',
      prompt: '为什么把 `wordDict` 转成 `set`？',
      codeContext: code,
      highlightLine: 2,
      options: [
        { id: 'a', text: 'list 查询 O(n)，set 查询 O(1)，否则总复杂度退化' },
        { id: 'b', text: 'set 内存更小' },
        { id: 'c', text: 'set 自动去重' },
        { id: 'd', text: 'a 与 c 都是原因' },
      ],
      answer: 'd',
      explain:
        '查询 O(1) 是主要原因；自动去重虽不影响正确性但能少一些重复字符串。如果直接 `s[j:i] in wordDict`（list），每次内层 m 个比较，总复杂度 O(n²·m·L)，慢一倍以上。',
      tags: ['data-structure', 'complexity'],
    },
    {
      id: 'word-break.q4',
      prompt: '内层循环 `for j in range(0, i)` 的语义是？',
      options: [
        { id: 'a', text: '枚举切割点 j：把 s[:i] 切成 s[:j] + s[j:i]' },
        { id: 'b', text: '枚举字典单词' },
        { id: 'c', text: '从字符串末尾向前扫' },
        { id: 'd', text: '枚举哈希值' },
      ],
      answer: 'a',
      explain:
        'j 把 s[:i] 一分为二：左半 s[:j]（用 dp[j] 判定能否拆）+ 右半 s[j:i]（直接查字典）。j 取 0 时左半是空串（dp[0]=True），意味着"整段 s[:i] 必须是一个单词"。',
      tags: ['invariant'],
    },
    {
      id: 'word-break.q5',
      prompt: '判断条件 `dp[j] and s[j:i] in word_set` 的两部分顺序能不能交换？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: '能交换但不该——`dp[j]` 是 O(1) 布尔，先短路掉大部分情况，性能更好' },
        { id: 'b', text: '不能交换，会错' },
        { id: 'c', text: '必须用 or 不能用 and' },
        { id: 'd', text: 'Python 中 and 总是从右往左求值' },
      ],
      answer: 'a',
      explain:
        'Python `and` 是短路求值——左边 False 就不算右边。把 O(1) 的 dp[j] 放左边，能跳过 O(L) 的字符串切片+哈希查询。这种"轻条件在前"的写法是性能微优化。',
      tags: ['pythonism', 'complexity'],
    },
    {
      id: 'word-break.q6',
      prompt: 'dp[i] 一旦置 True 后立即 `break` 的好处？',
      codeContext: code,
      highlightLine: 13,
      options: [
        { id: 'a', text: '只关心"能否拆"——不需要枚举所有切割方案' },
        { id: 'b', text: '会出错' },
        { id: 'c', text: '没有任何区别' },
        { id: 'd', text: '会丢解' },
      ],
      answer: 'a',
      explain:
        '题目只问"能否"——找到一种拆法就足够。如果题目改为"返回所有拆分方案"（LC 140 word break II），就不能 break，需要回溯/记忆化 DFS。',
      tags: ['invariant'],
    },
    {
      id: 'word-break.q7',
      prompt: '剪枝 `range(max(0, i - max_len), i)` 的依据是？',
      codeContext: code,
      highlightLine: 10,
      options: [
        { id: 'a', text: 's[j:i] 长度若超过字典最长单词，必然不在字典里——这种 j 没必要枚举' },
        { id: 'b', text: '让循环写起来更短' },
        { id: 'c', text: 'Python 性能优化' },
        { id: 'd', text: '为了避免 IndexError' },
      ],
      answer: 'a',
      explain:
        '把内层从 O(i) 降到 O(L)（L 是最长单词长度），总复杂度从 O(n²·L) 降到 O(n·L²) 或 O(n·L)（取决于 set 切片成本）。当 n 远大于 L 时收益显著。',
      tags: ['complexity'],
    },
    {
      id: 'word-break.q8',
      prompt: '`s[j:i]` 在 Python 里的复杂度是？',
      options: [
        { id: 'a', text: 'O(1)，是视图' },
        { id: 'b', text: 'O(i - j)，每次切片都拷贝字符' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(log n)' },
      ],
      answer: 'b',
      explain:
        'Python 的 str 切片**不是**视图，会复制底层字符到新对象。这也是为什么剪枝重要——避免对长片段的无效切片。如果想完全 O(1) 比较，可以预处理子串哈希（Rabin-Karp）。',
      tags: ['complexity', 'pythonism'],
    },
    {
      id: 'word-break.q9',
      prompt: '若不剪枝（内层 j 从 0 到 i-1），最坏时间复杂度是？',
      options: [
        { id: 'a', text: 'O(n)' },
        { id: 'b', text: 'O(n²)' },
        { id: 'c', text: 'O(n³)（含切片成本）' },
        { id: 'd', text: 'O(n·m)' },
      ],
      answer: 'c',
      explain:
        '外层 n × 内层 n × 切片+哈希 O(n) ≈ O(n³)。剪枝把它压回 O(n·L)（L 通常远小于 n）。这是面试常被追问的"为什么这么剪"。',
      tags: ['complexity'],
    },
    {
      id: 'word-break.q10',
      prompt: '为什么不能用贪心"从左到右每次切下最长可识别前缀"？',
      options: [
        { id: 'a', text: '反例 s="aaaaab", wordDict=["a","aa","aaaa","aaaaa","b"]：贪心切 "aaaaa"+"b" 没问题，但 s="ab" 而 dict=["a","ab"] 时贪心选"a"导致剩"b"无解，正解应选"ab"' },
        { id: 'b', text: 'Python 的贪心实现复杂' },
        { id: 'c', text: '贪心其实可以' },
        { id: 'd', text: '需要先排序字典' },
      ],
      answer: 'a',
      explain:
        '"最长前缀"或"最短前缀"贪心都有反例。原因：本题没有最优子结构的局部决定性——必须靠 DP 枚举所有切点。这也是为什么 BFS / DFS + 记忆化都行，但贪心不行。',
      tags: ['invariant'],
    },
    {
      id: 'word-break.q11',
      prompt: '记忆化递归（DFS+cache）和本 DP 的关系是？',
      options: [
        { id: 'a', text: '等价——本质都在算同一组子问题，只是迭代/递归方向不同' },
        { id: 'b', text: '记忆化递归更慢' },
        { id: 'c', text: '记忆化递归会出错' },
        { id: 'd', text: '本题不能用记忆化' },
      ],
      answer: 'a',
      explain:
        '从前往后 DP 是"自底向上"，从后往前递归 + lru_cache 是"自顶向下"。两者复杂度相同。Python 写记忆化更简洁但有递归深度风险（n=300 的话边界）；DP 更稳。',
      tags: ['data-structure'],
    },
    {
      id: 'word-break.q12',
      prompt: '空间复杂度是？',
      options: [
        { id: 'a', text: 'O(n + W) —— dp 数组 O(n) + word_set O(W)（W=字典总字符数）' },
        { id: 'b', text: 'O(1)' },
        { id: 'c', text: 'O(n²)' },
        { id: 'd', text: 'O(n!)' },
      ],
      answer: 'a',
      explain:
        'dp 占 n+1 个布尔，word_set 占 O(字典总规模)。算法主体只用 O(n)，通常题目让你只关心 dp 的 O(n)。',
      tags: ['complexity'],
    },
    {
      id: 'word-break.q13',
      prompt: '若题目改成 "返回所有可能的拆分方案"（LC 140），代码主体应改为？',
      options: [
        { id: 'a', text: '保持本 DP 找出可达性，再用回溯还原所有方案' },
        { id: 'b', text: '直接 break 那行去掉' },
        { id: 'c', text: '完全不需要 DP' },
        { id: 'd', text: '把布尔 dp 改成 list of strings，每步存所有可能' },
      ],
      answer: 'a',
      explain:
        '常规打法：先用本 DP 把"可拆性"算出来作为剪枝；再 DFS 从后往前回溯（仅当 dp[j]=True 才递归），列出所有拆法。直接拼字符串列表的 DP 在最坏情况下方案数指数爆炸，内存吃紧。',
      tags: ['data-structure'],
    },
  ],
}

export default problem
