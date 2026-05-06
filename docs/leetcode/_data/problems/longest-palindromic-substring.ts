import type { Problem } from '../types'

const code = `def longestPalindrome(s: str) -> str:
    n = len(s)
    if n == 0:
        return s  # 空串直接返回
    start, max_len = 0, 1

    def expand(left: int, right: int) -> None:
        nonlocal start, max_len
        # 两侧扩展直到不再回文或越界
        while left >= 0 and right < n and s[left] == s[right]:
            if right - left + 1 > max_len:
                start, max_len = left, right - left + 1
            left -= 1
            right += 1

    for i in range(n):
        expand(i, i)      # 奇数中心(以 i 为中心)
        expand(i, i + 1)  # 偶数中心(以 i,i+1 为中心)
    return s[start:start + max_len]`

export const problem: Problem = {
  id: 'longest-palindromic-substring',
  leetcodeNo: 5,
  title: { zh: '最长回文子串', en: 'Longest Palindromic Substring' },
  difficulty: 'medium',
  pattern: 'dp-2d',
  tags: ['dp', 'string', 'two-pointer', 'expand-around-center'],
  statement:
    '给你一个字符串 `s`，找到 `s` 中**最长的回文子串**。\n\n（子串是连续的；与子序列不同。）',
  examples: [
    { input: 's = "babad"', output: '"bab"', note: '"aba" 也是合法答案' },
    { input: 's = "cbbd"', output: '"bb"', note: '偶数中心' },
    { input: 's = "a"', output: '"a"' },
  ],
  constraints: [
    '1 ≤ s.length ≤ 1000',
    's 仅由数字和英文字母组成',
  ],
  intuition:
    '两套主流思路：① **DP** —— `dp[i][j]` = `s[i..j]` 是否回文，转移 `dp[i][j] = dp[i+1][j-1] and s[i]==s[j]`；填表必须**按区间长度从小到大**才能保证 dp[i+1][j-1] 已就绪。② **中心扩展** —— 枚举每个可能的回文中心（n 个奇中心 + n-1 个偶中心），向两边扩展直到失败，记录最长。两者都是 O(n²) 时间，但中心扩展空间 O(1) 更优；最优算法是 Manacher 的 O(n)。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n²)', space: 'O(1)（中心扩展）' },
  microQuestions: [
    {
      id: 'palindrome.q1',
      prompt: '中心扩展法为什么要分"奇中心"和"偶中心"两种？',
      codeContext: code,
      highlightLine: 17,
      options: [
        { id: 'a', text: '回文长度可奇可偶——奇数有单一字符为中心(如 "aba"),偶数中心位于两字符之间(如 "abba")' },
        { id: 'b', text: '只是为了写起来更对称' },
        { id: 'c', text: 'Python 字符串不支持单字符中心' },
        { id: 'd', text: '为了避免越界' },
      ],
      answer: 'a',
      explain:
        '回文有两种几何形态。"aba" 中心在 b（一个字符），"abba" 中心在两 b 之间（两字符）。漏掉任意一种都会丢解。例如只考虑奇中心则永远找不到 "bb"。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'palindrome.q2',
      prompt: '`expand(i, i)` 和 `expand(i, i+1)` 的初始状态分别表示？',
      codeContext: code,
      highlightLine: 18,
      options: [
        { id: 'a', text: '前者:已知中心字符 s[i] 是回文(长度 1);后者:需要先验证 s[i]==s[i+1]' },
        { id: 'b', text: '没区别' },
        { id: 'c', text: '前者错' },
        { id: 'd', text: '后者错' },
      ],
      answer: 'a',
      explain:
        '奇中心从单字符出发"已经回文"；偶中心要先看 s[i] == s[i+1] 才能开始算回文（while 条件天然处理这点）。这两个调用互不重叠地枚举了所有 2n-1 个中心。',
      tags: ['invariant'],
    },
    {
      id: 'palindrome.q3',
      prompt: 'while 条件 `left >= 0 and right < n and s[left] == s[right]` 的三个判断顺序敏感吗？',
      codeContext: code,
      highlightLine: 10,
      options: [
        { id: 'a', text: '敏感——必须先做边界检查再访问 s[left]/s[right],否则可能 IndexError' },
        { id: 'b', text: '不敏感' },
        { id: 'c', text: 'Python 不会越界' },
        { id: 'd', text: '需要用 try/except' },
      ],
      answer: 'a',
      explain:
        'Python 的 `and` 短路求值——前两个条件失败则不会求第三个。如果把 s[left]==s[right] 写在前面，当 left=-1 时 s[-1] 在 Python 中是合法的（取尾部），不会报错但语义错。这是 Python 字符串的特殊陷阱。',
      tags: ['boundary', 'pythonism'],
    },
    {
      id: 'palindrome.q4',
      prompt: 'DP 解法 `dp[i][j]` 的状态定义是？',
      options: [
        { id: 'a', text: 's[i..j] 是否是回文(布尔)' },
        { id: 'b', text: 's[i..j] 中最长回文长度' },
        { id: 'c', text: 's[i..j] 包含的回文数量' },
        { id: 'd', text: 's[i..j] 是否是子序列' },
      ],
      answer: 'a',
      explain:
        '区间 DP 的经典布尔状态：dp[i][j] = s[i..j] 是不是回文。最终答案不是 dp[0][n-1] 而是"找出所有 dp[i][j]==True 中最长那段"。',
      tags: ['naming', 'invariant'],
    },
    {
      id: 'palindrome.q5',
      prompt: 'DP 转移 `dp[i][j] = dp[i+1][j-1] and s[i]==s[j]` 的依赖方向是？',
      options: [
        { id: 'a', text: '依赖更短的区间(i+1,j-1)——所以填表必须按区间长度从小到大' },
        { id: 'b', text: '依赖更长的区间' },
        { id: 'c', text: '依赖左侧或上侧' },
        { id: 'd', text: '没有依赖' },
      ],
      answer: 'a',
      explain:
        '"区间 DP 按长度填表"是核心技巧。常见两种循环写法：① 外层 length 从 2 到 n、内层 i 从 0 到 n-length；② 外层 i 从大到小、内层 j 从 i 到大。两者都保证 dp[i+1][j-1] 已计算。**直接 i 从 0 到 n、j 从 0 到 n 顺序遍历会错**。',
      tags: ['invariant', 'boundary'],
    },
    {
      id: 'palindrome.q6',
      prompt: 'DP 边界——长度 1 与长度 2 的初始化分别是？',
      options: [
        { id: 'a', text: '长度 1: dp[i][i] = True;长度 2: dp[i][i+1] = (s[i] == s[i+1])' },
        { id: 'b', text: '都是 True' },
        { id: 'c', text: '都是 False' },
        { id: 'd', text: '只需要长度 1' },
      ],
      answer: 'a',
      explain:
        '单字符必回文；两字符则看相等。这两个边界是因为转移 dp[i+1][j-1] 在长度=1 时变成 dp[i+1][i]（i+1>i 即"空区间"，需视为回文 True），所以**单独**处理可避免越界讨论。',
      tags: ['boundary'],
    },
    {
      id: 'palindrome.q7',
      prompt: 'DP 与中心扩展的复杂度对比？',
      options: [
        { id: 'a', text: 'DP: O(n²) 时间 + O(n²) 空间;中心扩展: O(n²) 时间 + O(1) 空间' },
        { id: 'b', text: 'DP 永远更快' },
        { id: 'c', text: '两者完全等价' },
        { id: 'd', text: '中心扩展更慢' },
      ],
      answer: 'a',
      explain:
        '时间相同，但中心扩展把空间压到 O(1)。所以面试 LC5 推荐写中心扩展。DP 的好处是状态明显，便于扩展（如 LC 647 回文子串计数 用 dp 即可一行求解）。',
      tags: ['complexity'],
    },
    {
      id: 'palindrome.q8',
      prompt: '`nonlocal start, max_len` 在内嵌函数中的作用？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: '让内层函数能修改外层闭包变量;否则赋值会创建本地新变量' },
        { id: 'b', text: '让变量全局可见' },
        { id: 'c', text: '提升性能' },
        { id: 'd', text: '类似 C 语言的指针' },
      ],
      answer: 'a',
      explain:
        'Python 中：读取闭包变量不需要 nonlocal；但**赋值**时若没有 nonlocal，会被当作创建一个新的局部变量遮蔽外层。这是 Python 闭包的反直觉规则之一。',
      tags: ['pythonism', 'syntax'],
    },
    {
      id: 'palindrome.q9',
      prompt: '为什么对每个 i 要调用 `expand(i, i)` 和 `expand(i, i+1)` 共两次？',
      options: [
        { id: 'a', text: '一次找以 i 为中心的奇回文,一次找以 i,i+1 为中心的偶回文——两类穷尽全部回文中心' },
        { id: 'b', text: '为了对称' },
        { id: 'c', text: '为了 O(n²) 复杂度' },
        { id: 'd', text: '随便' },
      ],
      answer: 'a',
      explain:
        '总共 2n-1 个潜在中心。每个 i 贡献一个奇中心(i,i) 和一个偶中心(i,i+1)，最后一个偶中心 (n-1, n) 因为 right>=n 直接退出 while。所以 expand 调用 2n 次但实际只 2n-1 次有效。',
      tags: ['invariant'],
    },
    {
      id: 'palindrome.q10',
      prompt: '时间复杂度是？',
      options: [
        { id: 'a', text: 'O(n)' },
        { id: 'b', text: 'O(n²)' },
        { id: 'c', text: 'O(n log n)' },
        { id: 'd', text: 'O(n³)' },
      ],
      answer: 'b',
      explain:
        '2n-1 个中心 × 每个最多扩展 n/2 步 = O(n²)。最优算法 Manacher 是 O(n)，但代码远复杂；面试中通常掌握中心扩展即可。朴素枚举所有子串再判定才是 O(n³)。',
      tags: ['complexity'],
    },
    {
      id: 'palindrome.q11',
      prompt: '空间复杂度（中心扩展版）是？',
      options: [
        { id: 'a', text: 'O(1)（不计输出切片）' },
        { id: 'b', text: 'O(n)' },
        { id: 'c', text: 'O(n²)' },
        { id: 'd', text: 'O(log n)' },
      ],
      answer: 'a',
      explain:
        '只用了几个指针。返回切片 s[start:start+max_len] 严格说占 O(max_len)，但这是必要输出不计入算法空间。DP 版需要 O(n²) 表格。',
      tags: ['complexity'],
    },
    {
      id: 'palindrome.q12',
      prompt: '若题目改成"统计回文子串的数量"（LC 647），思路差别？',
      options: [
        { id: 'a', text: '把 expand 里"是否更新最长"换成"计数 +1",同样 O(n²)' },
        { id: 'b', text: '需要换算法' },
        { id: 'c', text: '不能用中心扩展' },
        { id: 'd', text: '需要 Manacher' },
      ],
      answer: 'a',
      explain:
        '中心扩展每多扩成功一次 = 多发现一个回文。所以稍微改 expand 内的 if 即可：每次扩展成功 count += 1，不再追踪最长。这种"母题加结构小变"是面试常考衍生。',
      tags: ['data-structure'],
    },
    {
      id: 'palindrome.q13',
      prompt: '若题目改成"最长回文子序列"（LC 516），上面代码还能用吗？',
      options: [
        { id: 'a', text: '不能——子序列允许跳字符,需要不同的区间 DP: dp[i][j] = max(dp[i+1][j], dp[i][j-1], dp[i+1][j-1] + 2*(s[i]==s[j]))' },
        { id: 'b', text: '能,改一行就行' },
        { id: 'c', text: '完全相同' },
        { id: 'd', text: '只能用回溯' },
      ],
      answer: 'a',
      explain:
        '子串和子序列是两题——子串必须连续，子序列可以跳。LC 516 的转移类似 LCS：相等时 dp[i+1][j-1]+2，不等时 max(dp[i+1][j], dp[i][j-1])。识别"子串 vs 子序列"是 DP 题型分类的关键。',
      tags: ['invariant'],
    },
  ],
}

export default problem
