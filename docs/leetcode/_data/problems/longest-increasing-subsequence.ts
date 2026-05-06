import type { Problem } from '../types'

const code = `from bisect import bisect_left

def lengthOfLIS(nums: list[int]) -> int:
    # tails[k] 表示「长度为 k+1 的递增子序列」当前可能的末尾最小值
    tails: list[int] = []
    for x in nums:
        # 严格递增 → bisect_left；如果允许「非严格」改用 bisect_right
        idx = bisect_left(tails, x)
        if idx == len(tails):
            tails.append(x)  # x 比所有末尾都大,延长 LIS
        else:
            tails[idx] = x  # 把长度为 idx+1 的「末尾候选」收紧
    return len(tails)`

export const problem: Problem = {
  id: 'longest-increasing-subsequence',
  leetcodeNo: 300,
  title: { zh: '最长递增子序列', en: 'Longest Increasing Subsequence' },
  difficulty: 'medium',
  pattern: 'dp-1d',
  tags: ['dp', 'binary-search', 'patience-sort'],
  statement:
    '给你一个整数数组 `nums`，找到其中**最长严格递增子序列**的长度。\n\n**子序列**是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3,6,2,7]` 是数组 `[0,3,1,6,2,2,7]` 的子序列。\n\n请你设计时间复杂度为 `O(n log(n))` 的算法解决此问题。',
  examples: [
    { input: 'nums = [10,9,2,5,3,7,101,18]', output: '4', note: 'LIS 为 [2,3,7,101] 或 [2,3,7,18]' },
    { input: 'nums = [0,1,0,3,2,3]', output: '4' },
    { input: 'nums = [7,7,7,7,7,7,7]', output: '1', note: '严格递增——重复元素只能选一次' },
  ],
  constraints: [
    '1 ≤ nums.length ≤ 2500',
    '-10⁴ ≤ nums[i] ≤ 10⁴',
  ],
  intuition:
    '两套思路：① 朴素 DP O(n²)：dp[i] = 以 nums[i] 结尾的 LIS 长度，dp[i] = max(dp[j])+1 (j<i, nums[j]<nums[i])。② 贪心 + 二分 O(n log n)：维护 tails，tails[k] 是「长度 k+1 的 LIS 当前可能的最小末尾」。每来一个数二分找到它该替换的位置；若大于所有末尾就 append。注意 tails 本身**不一定是某条真实 LIS**，但它的长度等于 LIS 长度。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n log n)', space: 'O(n)' },
  microQuestions: [
    {
      id: 'lis.q1',
      prompt: '朴素 O(n²) 版本中，`dp[i]` 最自然的状态定义是？',
      options: [
        { id: 'a', text: '前 i 个元素中 LIS 的长度' },
        { id: 'b', text: '以 nums[i] 结尾的 LIS 长度' },
        { id: 'c', text: '不超过 nums[i] 的元素个数' },
        { id: 'd', text: '从 i 开始的 LIS 长度' },
      ],
      answer: 'b',
      explain:
        '"以 i 结尾"是关键——这样最后要 max(dp) 而不是返回 dp[n-1]。a 看似自然但难以转移：你不知道这个 LIS 是不是包含 nums[i]，导致下一步无法判断能不能续。"以 i 结尾"提供了明确的拼接锚点。',
      tags: ['naming', 'invariant'],
    },
    {
      id: 'lis.q2',
      prompt: '朴素 O(n²) 转移方程是？',
      options: [
        { id: 'a', text: 'dp[i] = dp[i-1] + 1' },
        { id: 'b', text: 'dp[i] = max(dp[j] + 1 for j < i if nums[j] < nums[i]), 默认为 1' },
        { id: 'c', text: 'dp[i] = max(dp[j]) for j < i' },
        { id: 'd', text: 'dp[i] = sum(dp[j]) for j < i' },
      ],
      answer: 'b',
      explain:
        '枚举所有合法前驱 j（nums[j] < nums[i]），取它们的 dp 最大值再 +1（接上 nums[i]）。若没有合法 j，dp[i] = 1（自身一个数）。最终答案是 max(dp)。',
      tags: ['invariant'],
    },
    {
      id: 'lis.q3',
      prompt: 'O(n log n) 版本中 tails 数组的语义是？',
      codeContext: code,
      highlightLine: 4,
      options: [
        { id: 'a', text: '某条具体 LIS 的内容' },
        { id: 'b', text: 'tails[k] = 长度为 k+1 的递增子序列**末尾的最小可能值**' },
        { id: 'c', text: '排序后的 nums' },
        { id: 'd', text: '前缀最大值数组' },
      ],
      answer: 'b',
      explain:
        '这是本算法最容易误解的地方。tails 本身**不是 LIS**，它只是"每个长度档位的末尾守门员"——把末尾压到最小，给后续元素更多机会接上。因此返回 len(tails)，但不能直接返回 tails 内容当作 LIS。',
      tags: ['invariant', 'naming'],
    },
    {
      id: 'lis.q4',
      prompt: '为什么 tails 始终是**严格递增**的？',
      options: [
        { id: 'a', text: '是题目约束，不需要证明' },
        { id: 'b', text: '数学归纳：每次插入要么追加（>所有现有元素）要么替换为不大于原值的数，相邻档位仍保持单调' },
        { id: 'c', text: '因为 nums 是有序的' },
        { id: 'd', text: '因为用了二分查找' },
      ],
      answer: 'b',
      explain:
        '正因为 tails 始终单调，才能用二分查找——这是 O(n log n) 的关键。证明大意：若长度 k 的最小末尾是 a、长度 k+1 的最小末尾是 b，必有 a < b（否则可以从那条 k+1 序列里去掉 b，剩下一条长度 k 末尾 < a 矛盾）。',
      tags: ['invariant'],
    },
    {
      id: 'lis.q5',
      prompt: '严格递增 LIS 应当用 `bisect_left` 还是 `bisect_right`？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: 'bisect_left——遇到等值时插到左边，把"等值末尾"也当作要替换的对象' },
        { id: 'b', text: 'bisect_right——遇到等值时插到右边，可延长 LIS' },
        { id: 'c', text: '都行，结果一样' },
        { id: 'd', text: '都不行' },
      ],
      answer: 'a',
      explain:
        '严格递增 = 不允许等值。用 bisect_left 把 x 替换在第一个 ≥x 的位置上：若 tails[k]==x，则我们应该替换掉它（保持长度不变）而不是 append（错误延长）。换成 bisect_right 等价于"非严格递增"，会把 [7,7,7] 算成长度 3。',
      tags: ['boundary', 'pythonism'],
    },
    {
      id: 'lis.q6',
      prompt: '若题目改为"非严格"递增（允许等值），二分函数应改为？',
      options: [
        { id: 'a', text: 'bisect_left' },
        { id: 'b', text: 'bisect_right' },
        { id: 'c', text: 'bisect_left + 1' },
        { id: 'd', text: '不变' },
      ],
      answer: 'b',
      explain:
        '非严格递增允许 x 紧跟在等值之后。bisect_right 在等值时返回右端点，使 idx==len(tails) 触发 append——把 LIS 延长。这是 LeetCode 上"递增"和"非递减"题型最常考的细节。',
      tags: ['boundary'],
    },
    {
      id: 'lis.q7',
      prompt: '`if idx == len(tails): tails.append(x)` 这个分支何时触发？',
      codeContext: code,
      highlightLine: 9,
      options: [
        { id: 'a', text: 'x 比 tails 中所有元素都大——能延长当前 LIS' },
        { id: 'b', text: 'x 等于 tails 末尾元素' },
        { id: 'c', text: 'tails 为空时' },
        { id: 'd', text: '永远不会触发' },
      ],
      answer: 'a',
      explain:
        '当 x 大于 tails[-1]，bisect_left 返回 len(tails)，意味着没有现有"末尾候选"可被替换，应该开辟新长度档位 → append。这就是"长度增长"的唯一时机。',
      tags: ['invariant'],
    },
    {
      id: 'lis.q8',
      prompt: '`tails[idx] = x` 这一步的意义是？',
      codeContext: code,
      highlightLine: 12,
      options: [
        { id: 'a', text: '把长度 idx+1 的 LIS 末尾收紧为更小的 x，给后续更多机会接上' },
        { id: 'b', text: '增加 LIS 长度' },
        { id: 'c', text: '删除某条 LIS' },
        { id: 'd', text: '排序 tails' },
      ],
      answer: 'a',
      explain:
        '替换不增长但改善"末尾"。例如 tails=[2,5,7]，来个 x=3：替换为 [2,3,7]——长度未变但 3 比 5 小，下次来 4 就能扩展成长度 4。这就是"贪心保持末尾最小"的核心动机。',
      tags: ['invariant'],
    },
    {
      id: 'lis.q9',
      prompt: '为什么 O(n log n) 版本的 `tails` 不一定就是某条真实 LIS？',
      options: [
        { id: 'a', text: '它来自不同时间点的"末尾候选"，相对顺序在原数组里可能不存在' },
        { id: 'b', text: '是 bug，应该修复' },
        { id: 'c', text: '只在 nums 有重复时不真实' },
        { id: 'd', text: 'tails 总是一条真实 LIS' },
      ],
      answer: 'a',
      explain:
        '反例：nums=[0,1,0,3,2,3]，过程结束 tails=[0,1,2,3]，但原数组里 0(第三个)→1→2→3 顺序不存在。要还原真实 LIS 需要额外记录每个元素的"长度档位"以及前驱链。',
      tags: ['invariant'],
    },
    {
      id: 'lis.q10',
      prompt: '`bisect_left(tails, x)` 的复杂度是？',
      options: [
        { id: 'a', text: 'O(log n)' },
        { id: 'b', text: 'O(n)' },
        { id: 'c', text: 'O(1)' },
        { id: 'd', text: 'O(n log n)' },
      ],
      answer: 'a',
      explain:
        'bisect 在有序序列上做二分查找。整体算法 n 个元素 × 每次二分 O(log n) = O(n log n)。',
      tags: ['complexity'],
    },
    {
      id: 'lis.q11',
      prompt: '能否用堆 / 平衡 BST 替代 bisect？',
      options: [
        { id: 'a', text: '可以但更复杂；Python 用 list + bisect 已经达到 O(n log n) 是最简洁选择' },
        { id: 'b', text: '不行，必须用堆' },
        { id: 'c', text: '不行，必须用红黑树' },
        { id: 'd', text: '不行，本算法只能用列表' },
      ],
      answer: 'a',
      explain:
        '复杂度上等价。Python 标准库 bisect 已经够用；C++ 里用 lower_bound 同理。用红黑树（如 SortedList）反而引入额外常数。算法竞赛偏 bisect。',
      tags: ['data-structure'],
    },
    {
      id: 'lis.q12',
      prompt: '空间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(log n)' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(n²)' },
      ],
      answer: 'c',
      explain:
        'tails 最坏装 n 个元素（输入本就严格递增的极端情况）。',
      tags: ['complexity'],
    },
    {
      id: 'lis.q13',
      prompt: '关于"两种解法的取舍"，最准确的说法是？',
      options: [
        { id: 'a', text: 'O(n²) 版本直观、易于扩展（如要还原 LIS、要计数 LIS 数量）；O(n log n) 仅算长度时更优' },
        { id: 'b', text: 'O(n log n) 永远更好' },
        { id: 'c', text: '两者完全等价' },
        { id: 'd', text: 'O(n²) 是错的' },
      ],
      answer: 'a',
      explain:
        '面试要会说出取舍。本题问"长度"用贪心+二分；如果问"输出 LIS 内容"或"LIS 数量"通常回到 O(n²) DP 或它的扩展，因为 tails 不携带前驱链信息。',
      tags: ['complexity', 'data-structure'],
    },
  ],
}

export default problem
