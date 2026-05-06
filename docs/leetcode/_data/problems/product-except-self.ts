import type { Problem } from '../types'

const code = `def productExceptSelf(nums: list[int]) -> list[int]:
    n = len(nums)
    answer = [1] * n

    # 第一遍：answer[i] = 「i 左侧所有元素的乘积」
    left = 1
    for i in range(n):
        answer[i] = left
        left *= nums[i]

    # 第二遍：从右往左，把「i 右侧所有元素的乘积」乘进去
    right = 1
    for i in range(n - 1, -1, -1):
        answer[i] *= right
        right *= nums[i]

    return answer`

export const problem: Problem = {
  id: 'product-except-self',
  leetcodeNo: 238,
  title: { zh: '除自身以外数组的乘积', en: 'Product of Array Except Self' },
  difficulty: 'medium',
  pattern: 'bit-prefix',
  tags: ['array', 'prefix-product'],
  statement:
    '给你一个整数数组 `nums`，返回**数组** `answer`，其中 `answer[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积。\n\n题目数据**保证**数组 `nums` 之中任意元素的全部前缀元素和后缀的乘积都在 **32 位整数**范围内。\n\n请**不要使用除法**，且在 `O(n)` 时间复杂度内完成此题。',
  examples: [
    { input: 'nums = [1,2,3,4]', output: '[24,12,8,6]' },
    { input: 'nums = [-1,1,0,-3,3]', output: '[0,0,9,0,0]', note: '含 0 时尤其能体现「不能用除法」的必要性' },
  ],
  constraints: [
    '2 ≤ nums.length ≤ 10⁵',
    '-30 ≤ nums[i] ≤ 30',
    '所有前缀积、后缀积都在 32 位整数范围内',
  ],
  intuition:
    '`answer[i] = (左侧乘积) * (右侧乘积)`。两遍扫描：第一遍从左到右把「i 左侧所有元素的乘积」存到 `answer[i]`；第二遍从右到左用一个滚动变量 right 把「i 右侧所有元素的乘积」乘进去。共用 answer 数组让额外空间降到 O(1)（不计输出）。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(1) 不计输出 answer 数组' },
  microQuestions: [
    {
      id: 'product-except-self.q1',
      prompt: '题目明确禁止「除法」，最大原因是？',
      options: [
        { id: 'a', text: 'nums 含 0 时 total / nums[i] 会除零；含一个 0 与多个 0 的处理也很特殊' },
        { id: 'b', text: '除法慢' },
        { id: 'c', text: 'Python 不支持除法' },
        { id: 'd', text: '为了凑题' },
      ],
      answer: 'a',
      explain:
        '「不用除法」是这道题的灵魂。0 让 `total / nums[i]` 在 0 处除零，且需要分类讨论 0 的个数。强制禁止除法 → 逼出前缀积/后缀积技巧。',
      tags: ['boundary'],
    },
    {
      id: 'product-except-self.q2',
      prompt: '`answer[i]` 的核心分解是？',
      options: [
        { id: 'a', text: '(i 左侧所有元素之积) × (i 右侧所有元素之积)' },
        { id: 'b', text: '∑ 而非 ×' },
        { id: 'c', text: '总积 / nums[i]' },
        { id: 'd', text: '上一项 × nums[i]' },
      ],
      answer: 'a',
      explain:
        '这是整道题的支柱：把「除自身」翻译成「左积 × 右积」。两个量各扫一遍 O(n)、相乘也是 O(n) → 总 O(n)。',
      tags: ['invariant'],
    },
    {
      id: 'product-except-self.q3',
      prompt: '第一遍循环结束后，`answer[i]` 的语义是？',
      codeContext: code,
      highlightLine: 7,
      options: [
        { id: 'a', text: 'i 左侧所有元素的乘积（不含 nums[i]）' },
        { id: 'b', text: '前 i 个元素的乘积（含 nums[i]）' },
        { id: 'c', text: '右侧乘积' },
        { id: 'd', text: '总乘积' },
      ],
      answer: 'a',
      explain:
        '注意「不含 nums[i]」这一点：循环里先 `answer[i] = left` 再 `left *= nums[i]`——left 在写入时还没把 nums[i] 自己算进去，所以恰好是「左侧之积」。',
      tags: ['invariant'],
    },
    {
      id: 'product-except-self.q4',
      prompt: '`answer[i] = left; left *= nums[i]` 这两行的顺序为何不能调换？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: '先把「不含自己的左积」写入，再把自己乘到 left 里——颠倒后 answer[i] 会包含 nums[i]，结果错' },
        { id: 'b', text: '随便' },
        { id: 'c', text: 'Python 强制顺序' },
        { id: 'd', text: '为了节省内存' },
      ],
      answer: 'a',
      explain:
        '严格来说 left 是个滚动状态，「先用旧值写 answer，再更新 left」是这种「左侧累积」模式的招牌写法。和 prefix sum 完全同构。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'product-except-self.q5',
      prompt: '第二遍从右到左写入时，为什么用 `*=` 而不是 `=`？',
      codeContext: code,
      highlightLine: 13,
      options: [
        { id: 'a', text: '因为第一遍 answer[i] 已经是「左积」，第二遍要把「右积」**乘进去**——`*=` 表示「在已有基础上乘」' },
        { id: 'b', text: '随便' },
        { id: 'c', text: '`=` 也对' },
        { id: 'd', text: 'Python 限制' },
      ],
      answer: 'a',
      explain:
        '这是「双共用一份 answer」的关键——第一遍写左积、第二遍乘右积。两遍合一就直接是答案，不需要第三个数组。',
      tags: ['invariant', 'pythonism'],
    },
    {
      id: 'product-except-self.q6',
      prompt: '`right` 这个滚动变量在第二遍循环中的语义是？',
      codeContext: code,
      highlightLine: 12,
      options: [
        { id: 'a', text: 'i 右侧所有元素的乘积（不含 nums[i]）' },
        { id: 'b', text: 'nums[i:] 的乘积（含 nums[i]）' },
        { id: 'c', text: '总乘积' },
        { id: 'd', text: '随便' },
      ],
      answer: 'a',
      explain:
        '与第一遍 left 完全对称：`answer[i] *= right` 用旧 right；之后 `right *= nums[i]` 才把 nums[i] 自己加进去。',
      tags: ['invariant'],
    },
    {
      id: 'product-except-self.q7',
      prompt: '`for i in range(n - 1, -1, -1)` 这是从哪里到哪里的循环？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: '从 n-1 倒序到 0（含两端）' },
        { id: 'b', text: '从 n-1 到 1' },
        { id: 'c', text: '从 0 到 n-1' },
        { id: 'd', text: '空循环' },
      ],
      answer: 'a',
      explain:
        '`range(start, stop, step)` 三参形式——`range(n-1, -1, -1)` 即 `[n-1, n-2, ..., 1, 0]`。也可以写 `reversed(range(n))`，效果相同。',
      tags: ['pythonism', 'syntax'],
    },
    {
      id: 'product-except-self.q8',
      prompt: 'O(1) 额外空间的关键是什么？',
      options: [
        { id: 'a', text: '把「左积数组」直接写在输出 answer 里，第二遍用一个滚动变量 right 当「右积」即可——题目允许 answer 不计入额外空间' },
        { id: 'b', text: '原地修改 nums' },
        { id: 'c', text: '不可能 O(1)' },
        { id: 'd', text: '用一个 set' },
      ],
      answer: 'a',
      explain:
        '题面默认「输出数组不算额外空间」。我们就利用这点把 answer 当成左积数组的载体，再用一个变量代替整条「右积数组」——经典空间优化。',
      tags: ['complexity'],
    },
    {
      id: 'product-except-self.q9',
      prompt: '`answer = [1]*n` 这个初始值有什么用？',
      codeContext: code,
      highlightLine: 3,
      options: [
        { id: 'a', text: '初值 1 是乘法的「单位元」——左边没有元素时左积就是 1（answer[0] = 1）' },
        { id: 'b', text: '随便填' },
        { id: 'c', text: '填 0 也行' },
        { id: 'd', text: '只是为了开数组' },
      ],
      answer: 'a',
      explain:
        '与 XOR 题用 0 起步同理：乘法的单位元是 1。answer[0] = left（初值 1）= 「索引 0 左侧没有元素，乘积就是 1」。',
      tags: ['invariant'],
    },
    {
      id: 'product-except-self.q10',
      prompt: '示例 nums=[1,2,3,4]：第一遍跑完 answer 是？',
      options: [
        { id: 'a', text: '[1, 1, 2, 6]' },
        { id: 'b', text: '[24,12,8,6]' },
        { id: 'c', text: '[1,2,6,24]' },
        { id: 'd', text: '[24,8,6,1]' },
      ],
      answer: 'a',
      explain:
        'answer[0]=1 (左侧空)、answer[1]=1 (=1)、answer[2]=1×2=2、answer[3]=1×2×3=6。第二遍倒着乘：答案 [24,12,8,6]。',
      tags: ['invariant'],
    },
    {
      id: 'product-except-self.q11',
      prompt: '若题目允许除法、且无 0，最朴素的解是？',
      options: [
        { id: 'a', text: 'total = ∏ nums; answer[i] = total // nums[i]' },
        { id: 'b', text: '排序' },
        { id: 'c', text: '哈希表' },
        { id: 'd', text: '双指针' },
      ],
      answer: 'a',
      explain:
        '理解这个反面解法很重要——它正是题面禁止的写法。一旦 nums 有 0 就需要分情况讨论 0 的个数（0 个 / 1 个 / ≥2 个），实际上比前缀积更繁琐。',
      tags: ['boundary'],
    },
    {
      id: 'product-except-self.q12',
      prompt: '本题最常见的扩展场景？',
      options: [
        { id: 'a', text: '「左/右扫两遍」是相当多题的通用骨架——接雨水（双指针 max 版）、最大子数组拆分等' },
        { id: 'b', text: '只能用于本题' },
        { id: 'c', text: '是排序题的简化' },
        { id: 'd', text: '与 BFS 等价' },
      ],
      answer: 'a',
      explain:
        '「双向扫 + 共用一个数组」是面试题的高频模式。理解了这道题，你会在很多看似不相关的题里看到它的影子。',
      tags: ['data-structure'],
    },
  ],
}

export default problem
