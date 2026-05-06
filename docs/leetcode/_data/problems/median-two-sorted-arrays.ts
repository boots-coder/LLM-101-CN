import type { Problem } from '../types'

const code = `def findMedianSortedArrays(nums1: list[int], nums2: list[int]) -> float:
    # 始终对较短的数组做二分，保证复杂度是 O(log min(m, n))
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    m, n = len(nums1), len(nums2)
    total_left = (m + n + 1) // 2   # 左半总长（奇数时多 1 在左）

    lo, hi = 0, m                   # 在 nums1 上切的位置 i ∈ [0, m]
    INF = float(「inf」)
    while lo <= hi:
        i = (lo + hi) // 2
        j = total_left - i

        l1 = -INF if i == 0 else nums1[i - 1]
        r1 = INF if i == m else nums1[i]
        l2 = -INF if j == 0 else nums2[j - 1]
        r2 = INF if j == n else nums2[j]

        if l1 <= r2 and l2 <= r1:
            # 切点合法：左半最大 ≤ 右半最小
            if (m + n) % 2 == 1:
                return float(max(l1, l2))
            return (max(l1, l2) + min(r1, r2)) / 2
        elif l1 > r2:
            hi = i - 1              # 在 nums1 中切得太靠右
        else:
            lo = i + 1              # 在 nums1 中切得太靠左
    return 0.0                       # 题目保证有解，永远到不了`

export const problem: Problem = {
  id: 'median-two-sorted-arrays',
  leetcodeNo: 4,
  title: { zh: '寻找两个正序数组的中位数', en: 'Median of Two Sorted Arrays' },
  difficulty: 'hard',
  pattern: 'binary-search',
  tags: ['array', 'binary-search', 'divide-and-conquer'],
  statement:
    '给定两个大小分别为 `m` 和 `n` 的**正序**（从小到大）数组 `nums1` 和 `nums2`。请你找出并返回这两个正序数组的**中位数**。\n\n算法的时间复杂度应该为 `O(log (m + n))`。',
  examples: [
    { input: 'nums1 = [1, 3], nums2 = [2]', output: '2.00000', note: '合并后 [1,2,3]，中位数 2' },
    { input: 'nums1 = [1, 2], nums2 = [3, 4]', output: '2.50000', note: '合并后 [1,2,3,4]，中位数 (2+3)/2' },
    { input: 'nums1 = [], nums2 = [1]', output: '1.00000' },
  ],
  constraints: [
    'nums1.length == m, nums2.length == n',
    '0 ≤ m, n ≤ 1000',
    '1 ≤ m + n ≤ 2000',
    '-10⁶ ≤ nums1[i], nums2[i] ≤ 10⁶',
  ],
  intuition:
    '把"找中位数"转化为"在两个数组里各切一刀，使左半总长 = (m+n+1)//2、且左半最大 ≤ 右半最小"。一旦在较短数组上确定切点 i，另一数组切点 j = total_left - i 由长度约束唯一决定。对 i 做二分：l1 > r2 说明 nums1 切多了要左移，l2 > r1 说明切少了要右移。在较短数组上二分把复杂度压到 O(log min(m, n))。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(log min(m, n))', space: 'O(1)' },
  pythonRefIds: ['py-float-inf', 'py-tuple-swap'],
  microQuestions: [
    {
      id: 'median-two-sorted-arrays.q1',
      prompt: '本题为什么不能直接合并两数组找中位数？',
      options: [
        { id: 'a', text: '可以，效率也够' },
        { id: 'b', text: '能 AC 但复杂度 O(m + n)，题面要求 O(log(m + n))' },
        { id: 'c', text: '会出错' },
        { id: 'd', text: 'Python 内置不支持' },
      ],
      answer: 'b',
      explain:
        'O(m+n) 合并是大多数人的第一反应、也确实能过；但本题 hard 的灵魂在 O(log) 解——它要求把"找中位数"重新建模为"二分切点"。',
      tags: ['complexity', 'invariant'],
    },
    {
      id: 'median-two-sorted-arrays.q2',
      prompt: '"切两刀使左半总长为 (m+n+1)//2"的设计，对奇偶情形如何统一？',
      options: [
        { id: 'a', text: '只能处理偶数' },
        { id: 'b', text: '+1 的妙用：奇数时左半比右半多 1，中位数就是左半最大值；偶数时左右等长，中位数 = (左半最大 + 右半最小) / 2' },
        { id: 'c', text: '只能处理偶数，奇数要特殊处理' },
        { id: 'd', text: '没意义' },
      ],
      answer: 'b',
      explain:
        '`(m+n+1)//2` 让奇偶可以统一：奇数 m+n=5 → 左半 3 右半 2，中位数 = max(l1,l2)；偶数 m+n=4 → 左右各 2，中位数 = avg。这一行设计是本解法的精髓。',
      tags: ['invariant', 'pythonism'],
    },
    {
      id: 'median-two-sorted-arrays.q3',
      prompt: '为什么要先 swap 让 nums1 是较短的？',
      codeContext: code,
      highlightLine: 3,
      options: [
        { id: 'a', text: '可读性' },
        { id: 'b', text: '在较短数组上二分，把复杂度压到 O(log min(m, n))；同时避免 j = total_left - i 出现负数（短数组的 i 最大才 m，j 才不会越界负）' },
        { id: 'c', text: 'Python 习惯' },
        { id: 'd', text: '不需要' },
      ],
      answer: 'b',
      explain:
        '两个原因：复杂度（log min 优于 log max）+ 正确性（j 不可能为负，因为 m ≤ n 时 total_left ≥ m ≥ i）。许多 WA 都来自忘了这一行 swap。',
      tags: ['boundary', 'complexity'],
    },
    {
      id: 'median-two-sorted-arrays.q4',
      prompt: '`l1 = -inf if i == 0 else nums1[i-1]` 这行哨兵的作用是？',
      codeContext: code,
      highlightLine: 12,
      options: [
        { id: 'a', text: '让索引不越界' },
        { id: 'b', text: '让"切点在数组最左/最右"时合法性比较 l1 <= r2 / l2 <= r1 自动成立（任何数 ≥ -inf、≤ inf）' },
        { id: 'c', text: '装饰性' },
        { id: 'd', text: '为了输出格式' },
      ],
      answer: 'b',
      explain:
        '哨兵无穷把"切到最边"的特殊情况吞进了一般逻辑，避免一堆 `if i == 0` 分支。这是数值算法里常见的"吸收边界"技巧。',
      tags: ['pythonism', 'boundary'],
    },
    {
      id: 'median-two-sorted-arrays.q5',
      prompt: '切点合法的判定条件应该是？',
      codeContext: code,
      highlightLine: 17,
      options: [
        { id: 'a', text: 'l1 == l2 and r1 == r2' },
        { id: 'b', text: 'l1 <= r2 and l2 <= r1' },
        { id: 'c', text: 'l1 <= r1 and l2 <= r2' },
        { id: 'd', text: 'l1 + l2 == r1 + r2' },
      ],
      answer: 'b',
      explain:
        '"左半最大 ≤ 右半最小"的两两交叉比较：nums1 的左 ≤ nums2 的右、nums2 的左 ≤ nums1 的右。这是把"全局排好序"压成两个不等式的关键洞察。',
      tags: ['invariant'],
    },
    {
      id: 'median-two-sorted-arrays.q6',
      prompt: '当 l1 > r2 时该怎么调整？',
      codeContext: code,
      highlightLine: 22,
      options: [
        { id: 'a', text: 'lo = i + 1' },
        { id: 'b', text: 'hi = i - 1（在 nums1 切得太靠右，需要左移切点 i）' },
        { id: 'c', text: '交换 nums1 和 nums2' },
        { id: 'd', text: '退出循环' },
      ],
      answer: 'b',
      explain:
        'l1 > r2 说明 nums1 左半的最大值已经超过了 nums2 右半的最小值——nums1 切得太多了。要把 i 缩小，所以 hi = i - 1。',
      tags: ['invariant', 'boundary'],
    },
    {
      id: 'median-two-sorted-arrays.q7',
      prompt: '当 l2 > r1 时该怎么调整？',
      codeContext: code,
      highlightLine: 24,
      options: [
        { id: 'a', text: 'hi = i - 1' },
        { id: 'b', text: 'lo = i + 1（nums1 切得太少了，i 要右移）' },
        { id: 'c', text: '交换数组' },
        { id: 'd', text: '设 i = 0' },
      ],
      answer: 'b',
      explain:
        '镜像情形：nums1 留在左半的太少了——nums2 那边为了凑足左半数量塞进了过大的 l2，需要让 nums1 多切一点点，i 右移。',
      tags: ['invariant'],
    },
    {
      id: 'median-two-sorted-arrays.q8',
      prompt: '二分 i 的上界 hi 应该初始化为 m 还是 m - 1？',
      codeContext: code,
      highlightLine: 9,
      options: [
        { id: 'a', text: 'm - 1' },
        { id: 'b', text: 'm（i 表示"左半含 nums1 的前 i 个元素"，i 可以等于 m，意为 nums1 全部归到左半）' },
        { id: 'c', text: 'len(nums1) + 1' },
        { id: 'd', text: 'len(nums1) // 2' },
      ],
      answer: 'b',
      explain:
        '注意 i 是切点不是下标——i ∈ [0, m]，i = 0 表示 nums1 全部归右半，i = m 表示全部归左半。这是"切点二分"和"下标二分"的细微差别。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'median-two-sorted-arrays.q9',
      prompt: '奇数情形 (m+n) 为奇时，中位数应取？',
      codeContext: code,
      highlightLine: 19,
      options: [
        { id: 'a', text: 'min(r1, r2)' },
        { id: 'b', text: 'max(l1, l2)（左半多一格，那一格就是中位数）' },
        { id: 'c', text: '(max(l1, l2) + min(r1, r2)) / 2' },
        { id: 'd', text: '(l1 + l2 + r1 + r2) / 4' },
      ],
      answer: 'b',
      explain:
        '`(m+n+1)//2` 让左半比右半多 1（奇数情形）；这一格就是合并后排序的中间元素，即 max(l1, l2)。',
      tags: ['invariant'],
    },
    {
      id: 'median-two-sorted-arrays.q10',
      prompt: '若 nums1 为空（m == 0），算法应该返回什么？',
      options: [
        { id: 'a', text: '抛异常' },
        { id: 'b', text: '0' },
        { id: 'c', text: 'nums2 的中位数；上面写法天然支持：i 只能取 0，j = total_left，l1 = -inf，r1 = +inf，自然命中合法切点' },
        { id: 'd', text: '需要特判' },
      ],
      answer: 'c',
      explain:
        '这是哨兵 + 切点二分的优雅之处：极端情形被自然吸收。验证设计的好习惯：把 m=0 代入跑一遍。',
      tags: ['boundary'],
    },
    {
      id: 'median-two-sorted-arrays.q11',
      prompt: '时间复杂度是？',
      options: [
        { id: 'a', text: 'O(m + n)' },
        { id: 'b', text: 'O(log(m + n))' },
        { id: 'c', text: 'O(log min(m, n))' },
        { id: 'd', text: 'O(log max(m, n))' },
      ],
      answer: 'c',
      explain:
        '在较短数组上二分，所以 O(log min(m, n))，比题面要求的 O(log(m+n)) 更优（因为 log min ≤ log(m+n)）。',
      tags: ['complexity'],
    },
    {
      id: 'median-two-sorted-arrays.q12',
      prompt: '空间复杂度是？',
      options: [
        { id: 'a', text: 'O(m + n)' },
        { id: 'b', text: 'O(log n)' },
        { id: 'c', text: 'O(1)' },
        { id: 'd', text: 'O(min(m, n))' },
      ],
      answer: 'c',
      explain:
        '只用了若干标量（lo, hi, i, j, l1/r1/l2/r2）。不递归、不构造新数组（swap 仅交换引用）。',
      tags: ['complexity'],
    },
    {
      id: 'median-two-sorted-arrays.q13',
      prompt: 'Python 里 `nums1, nums2 = nums2, nums1` 这种写法的本质是？',
      options: [
        { id: 'a', text: '元组打包 + 解包，无需中间变量' },
        { id: 'b', text: '位运算 swap' },
        { id: 'c', text: '只能用于数字' },
        { id: 'd', text: '会复制整个列表' },
      ],
      answer: 'a',
      explain:
        'Python 先把右边求值成一个临时元组 `(nums2, nums1)`，再解包到左边变量。只复制引用、不复制列表内容，所以 O(1)。',
      tags: ['pythonism'],
    },
  ],
}

export default problem
