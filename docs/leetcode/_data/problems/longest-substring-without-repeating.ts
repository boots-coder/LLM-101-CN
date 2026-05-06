import type { Problem } from '../types'

const code = `def lengthOfLongestSubstring(s: str) -> int:
    last = {}            # 字符 -> 上次出现位置
    left = 0
    best = 0
    for right, ch in enumerate(s):
        if ch in last and last[ch] >= left:
            left = last[ch] + 1
        last[ch] = right
        best = max(best, right - left + 1)
    return best`

export const problem: Problem = {
  id: 'longest-substring-without-repeating',
  leetcodeNo: 3,
  title: { zh: '无重复字符的最长子串', en: 'Longest Substring Without Repeating Characters' },
  difficulty: 'medium',
  pattern: 'sliding-window',
  tags: ['string', 'sliding-window', 'hash-table'],
  statement:
    '给定一个字符串 `s`，请你找出其中不含有**重复字符**的**最长子串**的长度。\n\n子串是字符串中连续的字符序列。',
  examples: [
    { input: 's = "abcabcbb"', output: '3', note: '最长子串为 "abc"，长度 3' },
    { input: 's = "bbbbb"', output: '1', note: '最长子串为 "b"' },
    { input: 's = "pwwkew"', output: '3', note: '最长子串为 "wke"；注意 "pwke" 是子序列而非子串' },
  ],
  constraints: [
    '0 ≤ s.length ≤ 5 × 10⁴',
    's 由英文字母、数字、符号和空格组成',
  ],
  intuition:
    '滑动窗口 + 哈希记位置。右指针不断扩张吞入新字符；当新字符在窗口内已出现过（last[ch] >= left）就把 left 跳到该字符上次位置 +1，跳过后窗口里再次没有重复。每步用 right - left + 1 更新答案。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(min(n, Σ))' },
  pythonRefIds: ['py-enumerate', 'py-dict-in'],
  microQuestions: [
    {
      id: 'longest-substring-without-repeating.q1',
      prompt: '本题最适合的算法骨架是？',
      options: [
        { id: 'a', text: '暴力枚举所有子串 O(n³)' },
        { id: 'b', text: '滑动窗口 + 哈希记录字符位置或计数' },
        { id: 'c', text: '排序后二分查找' },
        { id: 'd', text: '动态规划 dp[i] = dp[i-1] + 1' },
      ],
      answer: 'b',
      explain:
        '"连续子串 + 不重复"几乎是滑动窗口的招牌信号：右指针扩张吃字符，违规时左指针收缩；左右指针各只走 n 步，O(n)。',
      tags: ['data-structure'],
    },
    {
      id: 'longest-substring-without-repeating.q2',
      prompt: '`last` 字典里 key 和 value 的语义最该是？',
      codeContext: code,
      highlightLine: 2,
      options: [
        { id: 'a', text: 'key = 字符，value = 出现次数' },
        { id: 'b', text: 'key = 字符，value = 最后一次出现的下标' },
        { id: 'c', text: 'key = 下标，value = 字符' },
        { id: 'd', text: 'key = 字符串前缀，value = 长度' },
      ],
      answer: 'b',
      explain:
        '"位置"比"次数"信息更丰富——一旦遇到重复字符，可以直接把 left 跳到 last[ch]+1，省去逐步收缩。计数法（a 选项）也能解，但要循环弹出，常数更差。',
      tags: ['data-structure', 'invariant'],
    },
    {
      id: 'longest-substring-without-repeating.q3',
      prompt: '遇到重复字符 ch 时，为什么判断条件是 `last[ch] >= left` 而不是 `ch in last`？',
      codeContext: code,
      highlightLine: 6,
      options: [
        { id: 'a', text: '为了节省一次哈希查询' },
        { id: 'b', text: '`ch in last` 只能说明历史上见过；只有上次位置仍在当前窗口内（>= left）才需要收缩' },
        { id: 'c', text: '两者完全等价' },
        { id: 'd', text: 'Python 里 `>=` 比 `in` 快' },
      ],
      answer: 'b',
      explain:
        '关键 invariant：last 里可能残留着早已被滑出窗口的旧位置。例如 s = "abba"，处理到第二个 a 时 last[a]=0 但 left 已经是 2——此时不需要再收缩。',
      tags: ['invariant', 'boundary'],
    },
    {
      id: 'longest-substring-without-repeating.q4',
      prompt: '收缩 left 时为什么是 `left = last[ch] + 1`，而不是 `left += 1`？',
      codeContext: code,
      highlightLine: 7,
      options: [
        { id: 'a', text: '`left += 1` 也能 AC，只是慢一点' },
        { id: 'b', text: '可以一步跳到重复字符之后，省去一格一格收缩' },
        { id: 'c', text: '`left += 1` 会越界' },
        { id: 'd', text: 'Python 不支持 += 运算' },
      ],
      answer: 'b',
      explain:
        '这是位置法相对计数法的最大优势：一次跳跃，常数次操作。计数法每次只能 left += 1 直到该字符计数回到 1，最坏会扫一整段窗口。',
      tags: ['complexity', 'pythonism'],
    },
    {
      id: 'longest-substring-without-repeating.q5',
      prompt: '`last[ch] = right` 这一句应该放在哪里？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: '在 if 收缩之前——抢先记录' },
        { id: 'b', text: '在 if 收缩之后——保证先用旧值跳左指针，再覆盖为新位置' },
        { id: 'c', text: '在循环外——只记一次' },
        { id: 'd', text: '可以省掉，不影响正确性' },
      ],
      answer: 'b',
      explain:
        '必须先用旧的 last[ch] 来调整 left，再写入新值。如果先覆盖，last[ch] 就成了 right，跳跃逻辑全错。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'longest-substring-without-repeating.q6',
      prompt: '当前窗口长度的正确表达式是？',
      codeContext: code,
      highlightLine: 9,
      options: [
        { id: 'a', text: 'right - left' },
        { id: 'b', text: 'right - left + 1' },
        { id: 'c', text: 'right + left' },
        { id: 'd', text: 'len(last)' },
      ],
      answer: 'b',
      explain:
        '区间 [left, right] 是闭区间，含两端，所以长度 = right - left + 1。`len(last)` 也巧合可用（位置法时 last 装了所有字符），但语义不直观且依赖实现。',
      tags: ['boundary'],
    },
    {
      id: 'longest-substring-without-repeating.q7',
      prompt: 'left 在循环过程中是否可能减小（回退）？',
      options: [
        { id: 'a', text: '会，遇到重复字符时回退' },
        { id: 'b', text: '不会，left 单调不减——这是 O(n) 的关键' },
        { id: 'c', text: '取决于字符集' },
        { id: 'd', text: '只有空串时不会' },
      ],
      answer: 'b',
      explain:
        'left 单调非减是滑窗 O(n) 的核心保证。`last[ch] + 1` 在 last[ch] >= left 时一定 > left，所以左指针只会前进，不会后退。',
      tags: ['invariant', 'complexity'],
    },
    {
      id: 'longest-substring-without-repeating.q8',
      prompt: '时间复杂度是？',
      options: [
        { id: 'a', text: 'O(n²)' },
        { id: 'b', text: 'O(n log n)' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(n × Σ)，Σ 为字符集大小' },
      ],
      answer: 'c',
      explain:
        '右指针走 n 步、左指针累计也最多走 n 步、dict 操作均摊 O(1)，总 O(n)。',
      tags: ['complexity'],
    },
    {
      id: 'longest-substring-without-repeating.q9',
      prompt: '空间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(min(n, Σ))，Σ 为字符集大小' },
        { id: 'c', text: 'O(n²)' },
        { id: 'd', text: 'O(log n)' },
      ],
      answer: 'b',
      explain:
        'last 至多装 min(n, Σ) 个键——n 是字符串长度、Σ 是字符种类上限。例如 ASCII 限定时可看作 O(1)，更严谨写法是 O(min(n, Σ))。',
      tags: ['complexity'],
    },
    {
      id: 'longest-substring-without-repeating.q10',
      prompt: '若改用"窗口里维护字符计数 + while 收缩"的写法，与位置法的区别？',
      options: [
        { id: 'a', text: '完全等价，性能一样' },
        { id: 'b', text: '都是 O(n)，但计数法常数更大；好处是模板更通用，可改为"最多 K 种"等变形' },
        { id: 'c', text: '计数法会错' },
        { id: 'd', text: '计数法是 O(n²)' },
      ],
      answer: 'b',
      explain:
        '两种都对都是 O(n)。位置法对本题更精简、常数小；计数法（while 收缩）更通用，"最多 K 种字符" "至多 K 个不同元素" 等变形几乎只能用计数法。',
      tags: ['data-structure', 'pythonism'],
    },
    {
      id: 'longest-substring-without-repeating.q11',
      prompt: '空串 `s = ""` 时该解法的行为？',
      options: [
        { id: 'a', text: '会抛 IndexError' },
        { id: 'b', text: '直接返回 0（for 循环根本不进，best 保持初值 0）' },
        { id: 'c', text: '返回 -1' },
        { id: 'd', text: '需要特判 `if not s: return 0`' },
      ],
      answer: 'b',
      explain:
        '空串 enumerate 不产生任何元素，循环体不执行，直接返回 best = 0。这是"用 best=0 初始化"自然支持空串的好处，免去特判。',
      tags: ['boundary'],
    },
    {
      id: 'longest-substring-without-repeating.q12',
      prompt: '题目问"最长子串"，下面哪一项与本题无关？',
      options: [
        { id: 'a', text: '子串必须连续' },
        { id: 'b', text: '子序列可以不连续——本题不允许' },
        { id: 'c', text: '答案是长度而非子串本身' },
        { id: 'd', text: '需要返回所有最长子串' },
      ],
      answer: 'd',
      explain:
        '题目只要长度，不要枚举所有解。陷阱选项 b 是常见混淆点：例 `pwwkew`，子序列 `pwke` 长度 4 不算，子串 `wke` 才算。',
      tags: ['boundary'],
    },
  ],
}

export default problem
