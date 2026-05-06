import type { Problem } from '../types'

const code = `def maxArea(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    best = 0
    while left < right:
        h = min(height[left], height[right])
        best = max(best, h * (right - left))
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return best`

export const problem: Problem = {
  id: 'container-with-most-water',
  leetcodeNo: 11,
  title: { zh: '盛最多水的容器', en: 'Container With Most Water' },
  difficulty: 'medium',
  pattern: 'two-pointer',
  tags: ['array', 'two-pointers', 'greedy'],
  statement:
    '给定一个长度为 `n` 的整数数组 `height`。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])`。\n\n找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳**最多的水**。返回容器可以储存的最大水量。\n\n说明：你不能倾斜容器。',
  examples: [
    { input: 'height = [1,8,6,2,5,4,8,3,7]', output: '49', note: '取 i=1(高8) 与 i=8(高7)，宽度 7，min(8,7)*7 = 49' },
    { input: 'height = [1,1]', output: '1' },
  ],
  constraints: [
    'n == height.length',
    '2 ≤ n ≤ 10⁵',
    '0 ≤ height[i] ≤ 10⁴',
  ],
  intuition:
    '对撞双指针。面积 = min(左高, 右高) × 宽度。每次移动较矮的那一侧——因为如果移动较高的，下一次 min 不会变大，宽度还在变小，面积只会更糟。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(1)' },
  pythonRefIds: ['py-min-max'],
  microQuestions: [
    {
      id: 'cwm.q1',
      prompt: '双指针应该如何初始化？',
      codeContext: code,
      highlightLine: 2,
      options: [
        { id: 'a', text: 'left=0, right=0' },
        { id: 'b', text: 'left=0, right=len(height)-1' },
        { id: 'c', text: 'left=0, right=len(height)' },
        { id: 'd', text: 'left=1, right=len(height)' },
      ],
      answer: 'b',
      explain:
        '对撞双指针的标准初始化是"两端"。`right=len-1` 是合法索引；`right=len` 会越界。',
      tags: ['boundary'],
    },
    {
      id: 'cwm.q2',
      prompt: '循环条件应该是？',
      codeContext: code,
      highlightLine: 4,
      options: [
        { id: 'a', text: 'while left < right' },
        { id: 'b', text: 'while left <= right' },
        { id: 'c', text: 'while left != right' },
        { id: 'd', text: 'while right > 0' },
      ],
      answer: 'a',
      explain:
        '当 left == right 时宽度为 0，没有面积可计算；用 `<` 既正确又能避免无效迭代。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'cwm.q3',
      prompt: '当前容器的高度由谁决定？',
      options: [
        { id: 'a', text: 'max(height[left], height[right])' },
        { id: 'b', text: 'min(height[left], height[right])' },
        { id: 'c', text: '(height[left] + height[right]) / 2' },
        { id: 'd', text: 'height[left] * height[right]' },
      ],
      answer: 'b',
      explain: '木桶效应——水位由较矮的一侧决定。',
      tags: ['invariant'],
    },
    {
      id: 'cwm.q4',
      prompt: '宽度应该怎么算？',
      options: [
        { id: 'a', text: 'right - left' },
        { id: 'b', text: 'right - left + 1' },
        { id: 'c', text: 'right + left' },
        { id: 'd', text: 'len(height)' },
      ],
      answer: 'a',
      explain:
        '索引差就是宽度。注意区别：如果题目问"区间长度（含两端）"才用 `+1`；这里求的是水平距离。',
      tags: ['boundary'],
    },
    {
      id: 'cwm.q5',
      prompt: '指针移动的核心策略：每次应该移动哪一侧？',
      options: [
        { id: 'a', text: '总是移动 left' },
        { id: 'b', text: '移动较矮的那一侧' },
        { id: 'c', text: '移动较高的那一侧' },
        { id: 'd', text: '随机选一侧' },
      ],
      answer: 'b',
      explain:
        '关键洞察：移动较矮一侧才有可能让 min 变大；移动较高一侧时新的 min 至少不会变大、而宽度又变小，面积必降。这就是这道题的灵魂。',
      tags: ['invariant'],
    },
    {
      id: 'cwm.q6',
      prompt: '当 height[left] == height[right] 时移动谁？',
      options: [
        { id: 'a', text: '一定要先移动 left' },
        { id: 'b', text: '一定要先移动 right' },
        { id: 'c', text: '移动谁都行——两侧都不会带来更大的 min' },
        { id: 'd', text: '同时各移动一步' },
      ],
      answer: 'c',
      explain:
        '相等时，两侧都不能带来更大的 min（移谁都至多保持不变），所以移谁都不丢解；常见写法是"<= 时移动 left"或"< 时移动 left"，结果相同。',
      tags: ['boundary'],
    },
    {
      id: 'cwm.q7',
      prompt: '为什么这个贪心策略不会错过最优解？',
      options: [
        { id: 'a', text: '因为我们尝试了所有 (i,j) 组合' },
        { id: 'b', text: '因为被丢弃的组合面积一定不大于已考察过的' },
        { id: 'c', text: '因为数组单调' },
        { id: 'd', text: '因为题目保证有唯一解' },
      ],
      answer: 'b',
      explain:
        '当移动较矮侧时，被"放弃"的所有 (短侧, 中间任何位置) 组合，宽度更小且 min 受限于短侧，面积都 ≤ 当前。所以不可能错过更大解。',
      tags: ['invariant'],
    },
    {
      id: 'cwm.q8',
      prompt: '时间复杂度是？',
      options: [
        { id: 'a', text: 'O(n²)' },
        { id: 'b', text: 'O(n log n)' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(log n)' },
      ],
      answer: 'c',
      explain: '两个指针总共最多移动 n 步（左+右指针的位移之和 ≤ n-1），所以 O(n)。',
      tags: ['complexity'],
    },
    {
      id: 'cwm.q9',
      prompt: '空间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(n)' },
        { id: 'c', text: 'O(log n)' },
        { id: 'd', text: 'O(n²)' },
      ],
      answer: 'a',
      explain: '只用了几个标量变量，没有额外结构。',
      tags: ['complexity'],
    },
    {
      id: 'cwm.q10',
      prompt: '把暴力 O(n²) 优化成 O(n) 的关键认知是？',
      options: [
        { id: 'a', text: '加缓存' },
        { id: 'b', text: '排序后二分' },
        { id: 'c', text: '识别出"较高一侧不动比较矮一侧不动更优"，从而每步排除一整列组合' },
        { id: 'd', text: '位运算技巧' },
      ],
      answer: 'c',
      explain:
        '面试官想看到的就是这一句话——它把搜索空间从 n² 压到 n。',
      tags: ['invariant', 'complexity'],
    },
  ],
}

export default problem
