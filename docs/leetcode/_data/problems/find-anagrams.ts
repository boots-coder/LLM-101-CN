import type { Problem } from '../types'

const code = `def findAnagrams(s: str, p: str) -> list[int]:
    n, m = len(s), len(p)
    if n < m:
        return []
    need = [0] * 26
    have = [0] * 26
    for ch in p:
        need[ord(ch) - 97] += 1
    res = []
    for right in range(n):
        have[ord(s[right]) - 97] += 1
        if right >= m:
            have[ord(s[right - m]) - 97] -= 1
        if have == need:
            res.append(right - m + 1)
    return res`

export const problem: Problem = {
  id: 'find-anagrams',
  leetcodeNo: 438,
  title: { zh: '找到字符串中所有字母异位词', en: 'Find All Anagrams in a String' },
  difficulty: 'medium',
  pattern: 'sliding-window',
  tags: ['string', 'sliding-window', 'hash-table'],
  statement:
    '给定两个字符串 `s` 和 `p`，找到 `s` 中所有 `p` 的**异位词**的子串，返回这些子串的起始索引。不考虑答案输出的顺序。\n\n**异位词**指由相同字母重排列形成的字符串（包括相同的字符串本身）。',
  examples: [
    { input: 's = "cbaebabacd", p = "abc"', output: '[0, 6]', note: '起始 0 的子串 "cba"、起始 6 的子串 "bac" 都是 "abc" 的异位词' },
    { input: 's = "abab", p = "ab"', output: '[0, 1, 2]' },
  ],
  constraints: [
    '1 ≤ s.length, p.length ≤ 3 × 10⁴',
    's 和 p 仅包含小写字母',
  ],
  intuition:
    '固定长度滑窗。窗口长度恒为 len(p)。维护两张 26 长度的字符计数表 need / have：右进左出（进窗口时 have[ch]+=1，超长时 have[s[left]]-=1），每步比较两表是否相等；相等就把窗口起点 right - m + 1 入答案。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n × Σ)', space: 'O(Σ)' },
  pythonRefIds: ['py-list-eq'],
  microQuestions: [
    {
      id: 'find-anagrams.q1',
      prompt: '本题最适合的窗口形态是？',
      options: [
        { id: 'a', text: '变长滑窗——根据违规收缩' },
        { id: 'b', text: '固定长度滑窗——窗口大小恒为 len(p)' },
        { id: 'c', text: '双指针对撞' },
        { id: 'd', text: '前缀和' },
      ],
      answer: 'b',
      explain:
        '"异位词"等价于"相同字符的同长度子串"，所以窗口长度天然固定为 len(p)，只需检查每个长度为 m 的窗口字符分布是否等于 p。',
      tags: ['data-structure'],
    },
    {
      id: 'find-anagrams.q2',
      prompt: 'need / have 用 26 长度数组而非 dict 的好处是？',
      codeContext: code,
      highlightLine: 5,
      options: [
        { id: 'a', text: '更易读' },
        { id: 'b', text: '比较两个 26-list 是 O(1) 常数（字符集固定），且 list == list 由 C 实现非常快' },
        { id: 'c', text: 'dict 不能用 == 比较' },
        { id: 'd', text: '题目要求' },
      ],
      answer: 'b',
      explain:
        '题目限定小写字母（Σ=26），用定长数组让"两表相等"成为 O(26) ≈ O(1) 的常数操作；dict 则要遍历 keys 比较且涉及 hash，常数大很多。',
      tags: ['data-structure', 'complexity'],
    },
    {
      id: 'find-anagrams.q3',
      prompt: '`ord(ch) - 97` 这个写法的含义？',
      options: [
        { id: 'a', text: '把字符转成 0-25 的下标（97 是 "a" 的 ASCII 码）' },
        { id: 'b', text: '把字符转成 ASCII 码值' },
        { id: 'c', text: '用于哈希' },
        { id: 'd', text: '随机化' },
      ],
      answer: 'a',
      explain:
        '`ord("a") == 97`。减去 97 让 a..z 落在 0..25，恰好对应 26 长度数组下标。等价写法 `ord(ch) - ord("a")`，只是慢一点点。',
      tags: ['pythonism'],
    },
    {
      id: 'find-anagrams.q4',
      prompt: '判断"窗口已满"的条件应该是？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: 'right == m' },
        { id: 'b', text: 'right >= m' },
        { id: 'c', text: 'right > m' },
        { id: 'd', text: 'right >= m - 1' },
      ],
      answer: 'b',
      explain:
        '当 right 加完 s[right] 后，窗口里有 right + 1 个字符；要保持长度恰为 m，需要从 right == m 起把 s[right - m] 弹出。等价表达：进窗口前窗口长度 = right；当 right >= m 时该长度已经 >= m，需要弹一个。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'find-anagrams.q5',
      prompt: '弹出旧字符时正确的索引是？',
      codeContext: code,
      highlightLine: 12,
      options: [
        { id: 'a', text: 's[right - m]' },
        { id: 'b', text: 's[right - m + 1]' },
        { id: 'c', text: 's[right - m - 1]' },
        { id: 'd', text: 's[left]，需要额外维护 left' },
      ],
      answer: 'a',
      explain:
        '当前窗口（加入 s[right] 之后）覆盖 [right - m + 1, right]，要把它缩回 m 长度需要弹出 left 上一格 s[right - m]。配合 a 写法不需要单独维护 left，是固定窗口的简洁写法。',
      tags: ['boundary'],
    },
    {
      id: 'find-anagrams.q6',
      prompt: '何时记录答案？',
      codeContext: code,
      highlightLine: 13,
      options: [
        { id: 'a', text: '只要 have[s[right]] >= need[s[right]] 就记' },
        { id: 'b', text: 'have == need 时记' },
        { id: 'c', text: '窗口扩到 m 时记' },
        { id: 'd', text: '循环结束后统一记' },
      ],
      answer: 'b',
      explain:
        '"异位词"等价于字符分布完全相同，所以两表完全相等才命中。Python 的 list 相等比较是逐元素比较，写法直观。',
      tags: ['invariant'],
    },
    {
      id: 'find-anagrams.q7',
      prompt: '匹配窗口的起点应该是？',
      codeContext: code,
      highlightLine: 14,
      options: [
        { id: 'a', text: 'right' },
        { id: 'b', text: 'right - m' },
        { id: 'c', text: 'right - m + 1' },
        { id: 'd', text: 'right + 1' },
      ],
      answer: 'c',
      explain:
        '当前满 m 长的窗口为 [right - m + 1, right]，起点是 right - m + 1。错位 1 是本题最常见的 off-by-one。',
      tags: ['boundary'],
    },
    {
      id: 'find-anagrams.q8',
      prompt: '若不预判 `if n < m: return []`，会发生什么？',
      codeContext: code,
      highlightLine: 3,
      options: [
        { id: 'a', text: '抛 IndexError' },
        { id: 'b', text: '答案错误地包含 [0]' },
        { id: 'c', text: '行为正确——for 循环里永远满足不了 right >= m，have != need，最终 res = []' },
        { id: 'd', text: '死循环' },
      ],
      answer: 'c',
      explain:
        '严格说不预判也对——只是提前 return 更显式、节省构造数组的开销。把它当成"工程性早返"而不是"必要正确性"。',
      tags: ['boundary'],
    },
    {
      id: 'find-anagrams.q9',
      prompt: '时间复杂度是？',
      options: [
        { id: 'a', text: 'O(n × m)' },
        { id: 'b', text: 'O(n × Σ)，Σ=26' },
        { id: 'c', text: 'O(n × log m)' },
        { id: 'd', text: 'O(n + m)' },
      ],
      answer: 'b',
      explain:
        '外层 n 次循环，每次 list == list 比较 26 个元素，总 O(26n)。Σ 视为常数时也可写作 O(n)。',
      tags: ['complexity'],
    },
    {
      id: 'find-anagrams.q10',
      prompt: '空间复杂度是？',
      options: [
        { id: 'a', text: 'O(n)' },
        { id: 'b', text: 'O(m)' },
        { id: 'c', text: 'O(Σ) = O(26) ≈ O(1)' },
        { id: 'd', text: 'O(n × m)' },
      ],
      answer: 'c',
      explain:
        '只用两张定长 26 数组（不计输出 res），所以是 O(Σ) 即 O(1)。',
      tags: ['complexity'],
    },
    {
      id: 'find-anagrams.q11',
      prompt: '能否优化为"每步不全比较 26 个，只维护一个匹配数 matched"？',
      options: [
        { id: 'a', text: '不能' },
        { id: 'b', text: '能——进窗口/出窗口时按"是否使该字符的 have 命中 need" 增减 matched，省掉 26 比较' },
        { id: 'c', text: '会丢解' },
        { id: 'd', text: 'Python 里不允许' },
      ],
      answer: 'b',
      explain:
        '经典优化：用 valid 计数。进窗口时 have[ch] 增到等于 need[ch] 则 valid+=1；超过则不再加。出窗口同理。当 valid == 不同字符种数时即命中。常数更小但代码更长，本题 Σ=26 时直接 list 比较已足够快，优化通常用于面试加分项。',
      tags: ['data-structure', 'complexity'],
    },
    {
      id: 'find-anagrams.q12',
      prompt: '若题目改成"包含 p 的某种排列作为子串"（即 LeetCode 567 字符串的排列），写法变化？',
      options: [
        { id: 'a', text: '完全不同算法' },
        { id: 'b', text: '同一套滑窗，只是返回 bool 而非起点列表（命中即 return True）' },
        { id: 'c', text: '需要改成回溯' },
        { id: 'd', text: '只能用 KMP' },
      ],
      answer: 'b',
      explain:
        '438 和 567 是双胞胎题：判断窗口是否是 p 的排列就是判断字符分布相等。骨架完全一样，区别只在收集所有起点 vs 命中即返。',
      tags: ['pythonism'],
    },
  ],
}

export default problem
