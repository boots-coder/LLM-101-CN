import type { Problem } from '../types'

const code = `def largestRectangleArea(heights: list[int]) -> int:
    # 两端各加一个高度为 0 的哨兵，省去边界判断
    heights = [0] + heights + [0]
    stack = []  # 单调递增栈，存下标
    best = 0
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            top = stack.pop()
            # 「弹出 top」⇒ 以 heights[top] 为高的矩形已确定边界
            # 左边界：栈中新栈顶（左侧第一个更矮）
            # 右边界：当前 i（右侧第一个更矮）
            width = i - stack[-1] - 1
            best = max(best, heights[top] * width)
        stack.append(i)
    return best`

export const problem: Problem = {
  id: 'largest-rectangle-histogram',
  leetcodeNo: 84,
  title: { zh: '柱状图中最大的矩形', en: 'Largest Rectangle in Histogram' },
  difficulty: 'hard',
  pattern: 'monotonic-stack',
  tags: ['array', 'stack', 'monotonic-stack'],
  statement:
    '给定 `n` 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且**宽度为 1**。\n\n求在该柱状图中，能够勾勒出来的矩形的**最大面积**。',
  examples: [
    { input: 'heights = [2,1,5,6,2,3]', output: '10', note: '高 5 与 高 6 形成 5×2 = 10' },
    { input: 'heights = [2,4]', output: '4' },
    { input: 'heights = [1,1,1,1]', output: '4', note: '所有等高时整段都是矩形' },
  ],
  constraints: [
    '1 ≤ heights.length ≤ 10⁵',
    '0 ≤ heights[i] ≤ 10⁴',
  ],
  intuition:
    '换个视角：枚举「以哪根柱子为高」的最大矩形——左右各扩张到「第一根更矮的柱子」前一格。单调递增栈恰好维护「左侧第一个更矮」的下标；当出现更矮的新柱子时，弹出栈顶就同时拿到了左右两侧边界。两端加 0 哨兵省去边界判断。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(n)' },
  microQuestions: [
    {
      id: 'largest-rectangle-histogram.q1',
      prompt: '这道题的核心枚举视角是？',
      options: [
        { id: 'a', text: '枚举矩形的左右边界 (i, j) 组合' },
        { id: 'b', text: '枚举「以哪根柱子的高度为矩形高」' },
        { id: 'c', text: '枚举所有可能的高度值' },
        { id: 'd', text: '从最高的柱子开始 DP' },
      ],
      answer: 'b',
      explain:
        '关键认知：最优矩形一定有「最矮的那根柱子作为高」。所以只要对每根柱子 k 求出「以 heights[k] 为高时能向左右扩到多远」，所有候选取 max 即为答案。视角变了，复杂度才有可能从 O(n²) 降到 O(n)。',
      tags: ['invariant'],
    },
    {
      id: 'largest-rectangle-histogram.q2',
      prompt: '应该维护「单调递增栈」还是「单调递减栈」？',
      options: [
        { id: 'a', text: '单调递增栈（栈底到栈顶递增）' },
        { id: 'b', text: '单调递减栈' },
        { id: 'c', text: '都可以' },
        { id: 'd', text: '同时维护两个栈' },
      ],
      answer: 'a',
      explain:
        '我们要找的是「左右第一根比当前更矮的柱子」。维护递增栈意味着栈顶元素左侧紧邻就是「比它更矮」的；遇到更矮的新柱子来时，正好提供了「右侧更矮」——左右边界一次性确定。',
      tags: ['invariant', 'data-structure'],
    },
    {
      id: 'largest-rectangle-histogram.q3',
      prompt: '为什么要在两端加 `0` 哨兵？',
      codeContext: code,
      highlightLine: 3,
      options: [
        { id: 'a', text: '加快算法' },
        { id: 'b', text: '末尾的 0 强制把栈中所有元素清算掉；首部 0 让「左侧无更矮」时仍有合法栈顶可减' },
        { id: 'c', text: '让数组对称' },
        { id: 'd', text: '题目要求' },
      ],
      answer: 'b',
      explain:
        '若不加末尾 0，遍历结束后栈里还可能堆着一摞递增的下标没结算；加 0 让它们必然被弹出。首部 0 让 `stack[-1]` 在弹出时永不为空，省掉 `if not stack: width = i else width = i - stack[-1] - 1` 的分支。两个哨兵都是经典写法。',
      tags: ['boundary', 'pythonism'],
    },
    {
      id: 'largest-rectangle-histogram.q4',
      prompt: 'while 循环条件中比较谓词应是 `>` 还是 `>=`？',
      codeContext: code,
      highlightLine: 7,
      options: [
        { id: 'a', text: '`>`：严格大于才弹' },
        { id: 'b', text: '`>=`：相等也弹' },
        { id: 'c', text: '都可以，结果相同' },
        { id: 'd', text: '取决于是否有重复值' },
      ],
      answer: 'c',
      explain:
        '两种写法都能 AC。用 `>` 时相同高度的柱子会一起暂留在栈里，最后被哨兵 0 一并清算；用 `>=` 时每根等高柱都会立刻清算前一根（算出宽度更窄但同高的矩形）——最终 max 值一致。但 `>` 略快，且不会重复结算。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'largest-rectangle-histogram.q5',
      prompt: '弹出栈顶 `top` 后，矩形的宽度应该怎么算？',
      codeContext: code,
      highlightLine: 12,
      options: [
        { id: 'a', text: 'i - top' },
        { id: 'b', text: 'i - top + 1' },
        { id: 'c', text: 'i - stack[-1] - 1' },
        { id: 'd', text: 'top - stack[-1]' },
      ],
      answer: 'c',
      explain:
        '弹出 top 后栈顶变成「左侧第一根更矮」的下标 stack[-1]，i 是「右侧第一根更矮」的下标。可用区间是 (stack[-1], i) 开区间，宽度 = i - stack[-1] - 1。这是该题最易写错的一行。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'largest-rectangle-histogram.q6',
      prompt: '如果不加首部 0 哨兵，弹出 top 时栈可能为空，正确的宽度处理是？',
      options: [
        { id: 'a', text: '宽度仍是 i - stack[-1] - 1，会 IndexError' },
        { id: 'b', text: '若栈空则宽度 = i（向左可以一直扩到 0）' },
        { id: 'c', text: '若栈空则跳过这次结算' },
        { id: 'd', text: '若栈空则宽度 = i - 1' },
      ],
      answer: 'b',
      explain:
        '栈空意味着 top 是「迄今为止最矮」的柱，向左可以扩到下标 0，向右扩到 i-1，宽度 = i。加首部 0 哨兵正是为了把这种情况统一进 `i - stack[-1] - 1` 的公式（此时 stack[-1] = -1 哨兵下标）。',
      tags: ['boundary'],
    },
    {
      id: 'largest-rectangle-histogram.q7',
      prompt: '相同高度的连续柱子（如 [3,3,3]）会被算成几个候选矩形？',
      options: [
        { id: 'a', text: '只算 1 个最大的' },
        { id: 'b', text: '算 3 个，但 max 取的最终是同一个值' },
        { id: 'c', text: '会重复计入答案，导致结果偏大' },
        { id: 'd', text: '会漏算，导致结果偏小' },
      ],
      answer: 'b',
      explain:
        '用 `>` 的写法下，三个 3 会一起堆在栈里直到末尾哨兵；末尾 0 把它们逐个弹出，得到 1×3、2×3、3×3 三个候选。用 best=max(...) 取最大值即可，结果不会偏大也不会漏。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'largest-rectangle-histogram.q8',
      prompt: '时间复杂度是？',
      options: [
        { id: 'a', text: 'O(n²)' },
        { id: 'b', text: 'O(n log n)' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(n³)' },
      ],
      answer: 'c',
      explain:
        '每个下标最多入栈一次、出栈一次。总操作数线性，整体 O(n)。这是单调栈摊还分析的标准结论。',
      tags: ['complexity'],
    },
    {
      id: 'largest-rectangle-histogram.q9',
      prompt: '空间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(log n)' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(n²)' },
      ],
      answer: 'c',
      explain:
        '严格递增的输入下，所有下标都会停在栈里；加上扩展数组本身也是 O(n)。',
      tags: ['complexity'],
    },
    {
      id: 'largest-rectangle-histogram.q10',
      prompt: '为什么暴力枚举矩形左右边界 (i, j) 是 O(n²) 甚至 O(n³)？',
      options: [
        { id: 'a', text: '需要 O(n) 找区间最小值' },
        { id: 'b', text: '需要 O(n²) 遍历所有 (i, j) 对' },
        { id: 'c', text: '两者结合：枚举 O(n²) × 求最小 O(n) = O(n³)。用稀疏表能降到 O(n² log n)' },
        { id: 'd', text: '暴力本身就是 O(n)' },
      ],
      answer: 'c',
      explain:
        '直观写法：双层循环枚举 (i, j)，每对再扫一遍取 min height，总 O(n³)。单调栈把视角换成「以谁为高」就只剩一遍扫描，是该题精彩之处。',
      tags: ['complexity'],
    },
    {
      id: 'largest-rectangle-histogram.q11',
      prompt: '若不加哨兵且循环结束后栈里还有元素，需要怎么处理？',
      options: [
        { id: 'a', text: '丢弃，对答案无影响' },
        { id: 'b', text: '再做一次「假装来了一个 0」的清算循环' },
        { id: 'c', text: '把它们的面积按全宽度计算' },
        { id: 'd', text: '重新跑一次算法' },
      ],
      answer: 'b',
      explain:
        '不加末尾 0 时，栈里剩下的递增序列都是「右侧没遇到更矮」的——必须再走一遍弹栈，宽度按 n - stack[-1] - 1 计算。加 0 哨兵就是把这步内嵌进主循环，写法更简洁。',
      tags: ['boundary'],
    },
    {
      id: 'largest-rectangle-histogram.q12',
      prompt: '这道题的解法对「最大子矩阵（01 矩阵）」有什么启发？',
      options: [
        { id: 'a', text: '完全无关' },
        { id: 'b', text: '把每一行视作直方图（向上累加连续 1 的高度），逐行调用本题解法即可' },
        { id: 'c', text: '可以把矩阵展平成 1D 数组' },
        { id: 'd', text: '需要全新的二维 DP' },
      ],
      answer: 'b',
      explain:
        '85 题「最大矩形」就是 84 题的二维版：维护「以当前行为底、向上连续 1 的高度数组 height[]」，每行调用一次本算法，O(m·n)。这是单调栈的常见复用，面试常考。',
      tags: ['data-structure', 'invariant'],
    },
  ],
}

export default problem
