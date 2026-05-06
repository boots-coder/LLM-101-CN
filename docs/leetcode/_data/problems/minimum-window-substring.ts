import type { Problem } from '../types'

const code = `def minWindow(s: str, t: str) -> str:
    if not s or not t or len(s) < len(t):
        return 「」
    need = {}
    for ch in t:
        need[ch] = need.get(ch, 0) + 1
    required = len(need)        # 不同字符种类数
    have = {}
    formed = 0                  # 已经「凑齐数量」的字符种数
    left = 0
    best_len = float(「inf」)
    best_l = 0
    for right, ch in enumerate(s):
        have[ch] = have.get(ch, 0) + 1
        if ch in need and have[ch] == need[ch]:
            formed += 1
        while formed == required:
            if right - left + 1 < best_len:
                best_len = right - left + 1
                best_l = left
            lc = s[left]
            have[lc] -= 1
            if lc in need and have[lc] < need[lc]:
                formed -= 1
            left += 1
    return 「」 if best_len == float(「inf」) else s[best_l: best_l + best_len]`

export const problem: Problem = {
  id: 'minimum-window-substring',
  leetcodeNo: 76,
  title: { zh: '最小覆盖子串', en: 'Minimum Window Substring' },
  difficulty: 'hard',
  pattern: 'sliding-window',
  tags: ['string', 'sliding-window', 'hash-table'],
  statement:
    '给你一个字符串 `s`、一个字符串 `t`。返回 `s` 中**涵盖 `t` 所有字符**的**最小子串**。如果 `s` 中不存在涵盖 `t` 所有字符的子串，则返回空字符串 `""`。\n\n注意：\n- 对于 `t` 中重复字符，我们寻找的子串中该字符数量必须**不少于** `t` 中该字符数量。\n- 如果 `s` 中存在这样的子串，我们保证它是唯一的答案。',
  examples: [
    { input: 's = "ADOBECODEBANC", t = "ABC"', output: '"BANC"', note: '"BANC" 是唯一覆盖 A、B、C 的最小子串' },
    { input: 's = "a", t = "a"', output: '"a"' },
    { input: 's = "a", t = "aa"', output: '""', note: 't 要两个 a，s 只有一个，无解' },
  ],
  constraints: [
    'm == s.length, n == t.length',
    '1 ≤ m, n ≤ 10⁵',
    's 和 t 由英文字母组成',
  ],
  intuition:
    '变长滑窗。用 need 记录 t 的字符需求；窗口里 have 计数 + formed 跟踪"已经凑齐数量的字符种数"。右指针扩张，formed == required 时进入收缩内循环：每弹出一个左字符，更新答案 + 必要时降低 formed 退出收缩。formed 这个量把"是否覆盖"压成 O(1) 判断，避免每步比较两张表。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(m + n)', space: 'O(Σ)' },
  pythonRefIds: ['py-dict-get', 'py-float-inf'],
  microQuestions: [
    {
      id: 'minimum-window-substring.q1',
      prompt: '本题与"找异位词"相比，关键的形态差别？',
      options: [
        { id: 'a', text: '没区别' },
        { id: 'b', text: '本题是变长滑窗（窗口长度未知，需收缩寻最优），异位词是定长滑窗' },
        { id: 'c', text: '本题用动态规划' },
        { id: 'd', text: '本题用前缀和' },
      ],
      answer: 'b',
      explain:
        '"覆盖 t 全部字符"长度未定——找到一次覆盖后必须尝试收缩寻找更小窗。这是变长滑窗的标志。',
      tags: ['data-structure'],
    },
    {
      id: 'minimum-window-substring.q2',
      prompt: '`required = len(need)` 的语义是？',
      codeContext: code,
      highlightLine: 6,
      options: [
        { id: 'a', text: 't 的总字符数' },
        { id: 'b', text: 't 中不同字符的种类数' },
        { id: 'c', text: 's 的长度' },
        { id: 'd', text: '一个常量' },
      ],
      answer: 'b',
      explain:
        '`need` 是字符->需求计数的字典，`len(need)` 即不同 key 数。formed 与 required 比较时，比的就是"种数"——这是把"是否覆盖"O(1) 化的关键。',
      tags: ['naming', 'invariant'],
    },
    {
      id: 'minimum-window-substring.q3',
      prompt: '`formed += 1` 应在何时触发？',
      codeContext: code,
      highlightLine: 14,
      options: [
        { id: 'a', text: '每次进窗口都 +1' },
        { id: 'b', text: '只有 ch 在 need 中且 have[ch] 恰好达到 need[ch] 时' },
        { id: 'c', text: 'have[ch] 超过 need[ch] 时' },
        { id: 'd', text: '窗口长度等于 len(t) 时' },
      ],
      answer: 'b',
      explain:
        '只有"刚好达标"的瞬间一次 +1。已经达标后再多加同字符不再 +1（不会重复计数），保证 formed 始终等于"已凑齐种数"。',
      tags: ['invariant', 'boundary'],
    },
    {
      id: 'minimum-window-substring.q4',
      prompt: '何时应该尝试收缩 left？',
      codeContext: code,
      highlightLine: 15,
      options: [
        { id: 'a', text: '窗口长度 > len(t) 时' },
        { id: 'b', text: 'formed == required 时——已经覆盖，可以试着收缩寻最小' },
        { id: 'c', text: 'have == need 时' },
        { id: 'd', text: '每步都收缩' },
      ],
      answer: 'b',
      explain:
        '只有覆盖了才有"答案候选"，才需要在内层循环里反复缩 left 试探最小值；一旦缩到不再覆盖（formed 降下来）就跳出收缩、回到外层继续扩张。',
      tags: ['invariant'],
    },
    {
      id: 'minimum-window-substring.q5',
      prompt: '收缩时 `formed -= 1` 应在什么条件下触发？',
      codeContext: code,
      highlightLine: 21,
      options: [
        { id: 'a', text: 'lc 在 need 中且 have[lc] 减完后 < need[lc]' },
        { id: 'b', text: 'lc 在 need 中即可' },
        { id: 'c', text: 'have[lc] 减到 0' },
        { id: 'd', text: 'lc 不在 need 中' },
      ],
      answer: 'a',
      explain:
        '镜像于 q3：只有"由满足变为不满足"那一步才 -=1。这两个对偶判断维持 invariant：formed 始终 = 已达标种数。',
      tags: ['invariant', 'boundary'],
    },
    {
      id: 'minimum-window-substring.q6',
      prompt: '更新答案应该在收缩 left 之前还是之后？',
      codeContext: code,
      highlightLine: 16,
      options: [
        { id: 'a', text: '之前——当前窗口本身就是合法候选' },
        { id: 'b', text: '之后——更精确' },
        { id: 'c', text: '都行' },
        { id: 'd', text: '只能在 formed != required 时' },
      ],
      answer: 'a',
      explain:
        '进入 while 体的瞬间窗口已合法（formed == required），先用它更新答案；然后弹掉左字符，下一轮再判 formed 是否还达标。次序错了会漏解。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'minimum-window-substring.q7',
      prompt: '为什么 best_len 用 `float("inf")` 初始化？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: '让任何长度都比它小，不用特判第一次更新' },
        { id: 'b', text: 'Python 没有整数最大值' },
        { id: 'c', text: '为了报错' },
        { id: 'd', text: '随便写的' },
      ],
      answer: 'a',
      explain:
        '哨兵值技巧：任何整数 < inf，第一次发现合法窗口必然进入更新分支；最后再用 `best_len == inf` 判断"是否一次都没更新过"以返回空串。',
      tags: ['pythonism'],
    },
    {
      id: 'minimum-window-substring.q8',
      prompt: 'have 是否需要排除 t 之外的字符？例如 s 里的"杂质"',
      options: [
        { id: 'a', text: '需要——杂质会污染计数' },
        { id: 'b', text: '不需要——have 可以记所有字符，只是 formed 触发条件用了 `ch in need` 过滤' },
        { id: 'c', text: '会出错' },
        { id: 'd', text: 'Python 不能这么写' },
      ],
      answer: 'b',
      explain:
        '记不记杂质都对，因为 formed 的更新条件已经过滤掉非 need 字符。让 have 装所有字符是为了写法统一，对正确性无影响。',
      tags: ['invariant', 'data-structure'],
    },
    {
      id: 'minimum-window-substring.q9',
      prompt: '为什么 t 中有重复字符（如 t = "aab"）也能被这套写法正确处理？',
      options: [
        { id: 'a', text: 'need 是 dict，重复字符的 key 计数会累加；formed 比的是"达到数量"而不是"出现一次"' },
        { id: 'b', text: '只能处理无重复的 t' },
        { id: 'c', text: '需要先去重 t' },
        { id: 'd', text: '需要换算法' },
      ],
      answer: 'a',
      explain:
        '`need[a] = 2, need[b] = 1`；`formed += 1` 当 have[a] 恰好等于 2 时触发。这正是为什么必须比较 `have[ch] == need[ch]` 而非 `>= 1`。',
      tags: ['data-structure', 'invariant'],
    },
    {
      id: 'minimum-window-substring.q10',
      prompt: '时间复杂度是？',
      options: [
        { id: 'a', text: 'O(m × n)' },
        { id: 'b', text: 'O((m + n) × Σ)' },
        { id: 'c', text: 'O(m + n)，m=len(s), n=len(t)' },
        { id: 'd', text: 'O(m log n)' },
      ],
      answer: 'c',
      explain:
        'left 和 right 都最多走 m 步，每步操作 O(1)（dict 均摊）。预处理 t 是 O(n)。总 O(m + n)。',
      tags: ['complexity'],
    },
    {
      id: 'minimum-window-substring.q11',
      prompt: '空间复杂度是？',
      options: [
        { id: 'a', text: 'O(m)' },
        { id: 'b', text: 'O(Σ)，Σ 为字符集大小' },
        { id: 'c', text: 'O(m × n)' },
        { id: 'd', text: 'O(log n)' },
      ],
      answer: 'b',
      explain:
        'need 和 have 至多装 Σ 个不同字符。题面字母时为 O(52) ≈ O(1)。',
      tags: ['complexity'],
    },
    {
      id: 'minimum-window-substring.q12',
      prompt: '如果删除 formed 这个量，改成"每步比较 have 是否覆盖 need"，会怎样？',
      options: [
        { id: 'a', text: '更优' },
        { id: 'b', text: '正确但每步多一次 O(Σ) 比较，整体 O((m+n)·Σ)；formed 是为了把"是否覆盖"压到 O(1)' },
        { id: 'c', text: '一定 TLE' },
        { id: 'd', text: '需要改成定长滑窗' },
      ],
      answer: 'b',
      explain:
        'formed 是 invariant 维护的经典例子——把"判断覆盖"这个查询从每步 O(Σ) 摊到每次进/出窗口的 O(1)。这是哈希滑窗的"高级感"所在。',
      tags: ['complexity', 'invariant'],
    },
    {
      id: 'minimum-window-substring.q13',
      prompt: '收尾返回时为何要判断 `best_len == float("inf")`？',
      codeContext: code,
      highlightLine: 25,
      options: [
        { id: 'a', text: '区分"无解"——一次也没找到合法窗口时返回 ""' },
        { id: 'b', text: 'Python 浮点比较容易出错' },
        { id: 'c', text: '为了节省内存' },
        { id: 'd', text: '没必要写' },
      ],
      answer: 'a',
      explain:
        '若整个扫描过程都没出现 formed == required（s 不能覆盖 t），best_len 维持 inf，必须返回 "" 而不是 s[0:0]——虽然这两个一样，但显式判断让"无解"语义清晰。',
      tags: ['boundary'],
    },
  ],
}

export default problem
