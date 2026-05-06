import type { Problem } from '../types'

const code = `def trap(height: list[int]) -> int:
    stack = []  # 单调递减栈，存下标
    total = 0
    for i, h in enumerate(height):
        while stack and height[stack[-1]] < h:
            bottom = stack.pop()  # 凹底
            if not stack:
                break  # 没有左墙，接不住水
            left = stack[-1]
            width = i - left - 1
            water_height = min(height[left], h) - height[bottom]
            total += width * water_height
        stack.append(i)
    return total`

export const problem: Problem = {
  id: 'trapping-rain-water',
  leetcodeNo: 42,
  title: { zh: '接雨水', en: 'Trapping Rain Water' },
  difficulty: 'hard',
  pattern: 'monotonic-stack',
  tags: ['array', 'stack', 'monotonic-stack', 'two-pointers'],
  statement:
    '给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，**下雨之后能接多少雨水**。',
  examples: [
    {
      input: 'height = [0,1,0,2,1,0,1,3,2,1,2,1]',
      output: '6',
      note: '黑柱与水分别看图可拼出 6 个单位',
    },
    { input: 'height = [4,2,0,3,2,5]', output: '9' },
    { input: 'height = [3,0,2]', output: '2' },
  ],
  constraints: [
    'n == height.length',
    '1 ≤ n ≤ 2·10⁴',
    '0 ≤ height[i] ≤ 10⁵',
  ],
  intuition:
    '单调递减栈按「层」累加雨水。维护一个高度递减的下标栈；每来一根更高的柱子，就把栈顶（凹底）弹出，结合新栈顶（左墙）和当前柱子（右墙）算一层水：宽度 × (min(左墙,右墙) - 凹底)。每根柱子只入出栈各一次，总 O(n)。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(n)' },
  microQuestions: [
    {
      id: 'trapping-rain-water.q1',
      prompt: '单调栈解法的核心几何意象是？',
      options: [
        { id: 'a', text: '把每个位置的水量算清楚再求和' },
        { id: 'b', text: '把雨水分成「层」，每弹一次栈结算一层' },
        { id: 'c', text: '把整个图划分成若干矩形' },
        { id: 'd', text: '从最高柱往两边扩散' },
      ],
      answer: 'b',
      explain:
        '与「按列计算」不同，单调栈是「按层（横切）计算」：栈里维护递减序列，遇到更高的柱子时，弹出的栈顶就是一层水的「凹底」，左墙=新栈顶，右墙=当前柱。这种几何视角是该解法的灵魂。',
      tags: ['invariant'],
    },
    {
      id: 'trapping-rain-water.q2',
      prompt: '应该维护什么样的单调栈？',
      options: [
        { id: 'a', text: '严格递增（栈底到栈顶）' },
        { id: 'b', text: '严格递减（栈底到栈顶）' },
        { id: 'c', text: '非递增（允许等高）' },
        { id: 'd', text: '非递减（允许等高）' },
      ],
      answer: 'c',
      explain:
        '我们要凹型——左侧高、底部低、右侧高。维护「栈底到栈顶非递增」（即遇到严格更高就弹），等高时入栈不会出错（结算时凹高为 0，水量为 0）。一些题解写「单调递减」是口语简称。',
      tags: ['invariant', 'data-structure'],
    },
    {
      id: 'trapping-rain-water.q3',
      prompt: '弹出栈顶 `bottom` 后，若栈为空应该？',
      codeContext: code,
      highlightLine: 7,
      options: [
        { id: 'a', text: '继续算，宽度按 i 算' },
        { id: 'b', text: 'break——没有左墙就接不住水' },
        { id: 'c', text: '把 bottom 重新压回栈' },
        { id: 'd', text: '把 i 当作左墙' },
      ],
      answer: 'b',
      explain:
        '没有左墙时水会从左侧流走——这一层接不住。break 即可（外层 stack.append(i) 会把 i 入栈继续后面的扫描）。也可以用 if/else 跳过本次结算然后继续 while。',
      tags: ['boundary'],
    },
    {
      id: 'trapping-rain-water.q4',
      prompt: '一层水的宽度公式是？',
      codeContext: code,
      highlightLine: 9,
      options: [
        { id: 'a', text: 'i - bottom' },
        { id: 'b', text: 'i - left - 1' },
        { id: 'c', text: 'i - left' },
        { id: 'd', text: 'left - bottom' },
      ],
      answer: 'b',
      explain:
        '左墙下标 left，右墙下标 i，中间凹陷的开区间宽度 = i - left - 1。注意 left 与 i 本身是「墙」不算水的列；区分「闭区间」与「开区间」是新手最容易写错的点。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'trapping-rain-water.q5',
      prompt: '一层水的高度公式是？',
      codeContext: code,
      highlightLine: 10,
      options: [
        { id: 'a', text: 'min(height[left], height[i])' },
        { id: 'b', text: 'min(height[left], height[i]) - height[bottom]' },
        { id: 'c', text: 'max(height[left], height[i]) - height[bottom]' },
        { id: 'd', text: 'height[i] - height[bottom]' },
      ],
      answer: 'b',
      explain:
        '木桶效应：水位由较矮的墙决定，所以是 min(左墙, 右墙)；再减去凹底高度才是这一层雨水的厚度。漏掉减 bottom 是常见 bug——相当于把「凹底以下的实心部分」也算成了水。',
      tags: ['invariant'],
    },
    {
      id: 'trapping-rain-water.q6',
      prompt: '如果 height = [3, 3, 3]，单调栈解法的结果是？',
      options: [
        { id: 'a', text: '0' },
        { id: 'b', text: '3' },
        { id: 'c', text: '会出错' },
        { id: 'd', text: '取决于是否严格 <' },
      ],
      answer: 'a',
      explain:
        '等高柱不构成凹陷，水量自然是 0。代码里 while 用严格 `<`，等高时不弹栈；即使改成 `<=`，结算时 min(左,右) - bottom = 3 - 3 = 0，水量也是 0——这正是为什么允许等高入栈不会出 bug。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'trapping-rain-water.q7',
      prompt: 'while 循环条件中，比较谓词应是 `<` 还是 `<=`？',
      codeContext: code,
      highlightLine: 5,
      options: [
        { id: 'a', text: '`<`：严格更高才弹' },
        { id: 'b', text: '`<=`：等高也弹' },
        { id: 'c', text: '都可以，结果相同' },
        { id: 'd', text: '取决于初始栈是否为空' },
      ],
      answer: 'c',
      explain:
        '两种写法都能 AC：用 `<` 时等高柱并存于栈，最后一起被结算（每层贡献 0）；用 `<=` 时立即弹，每次贡献 0。最终累加值一样。但 `<` 略快、思路更顺。',
      tags: ['boundary'],
    },
    {
      id: 'trapping-rain-water.q8',
      prompt: '本题还可以用「双指针 + 左右最大值」解，相比单调栈的优势是？',
      options: [
        { id: 'a', text: '更快' },
        { id: 'b', text: '空间 O(1)（不需要栈）' },
        { id: 'c', text: '能处理负数高度' },
        { id: 'd', text: '更易写' },
      ],
      answer: 'b',
      explain:
        '双指针法用 left/right 两个指针 + left_max/right_max 两个变量即可，空间 O(1)；时间一样是 O(n)。两种解法都要会，面试官常追问「有没有更省空间的写法」。',
      tags: ['complexity', 'data-structure'],
    },
    {
      id: 'trapping-rain-water.q9',
      prompt: '时间复杂度是？',
      options: [
        { id: 'a', text: 'O(n²)' },
        { id: 'b', text: 'O(n log n)' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(n³)' },
      ],
      answer: 'c',
      explain:
        '每个下标只入出栈各一次，外层 for 走 n 次，内层 while 总弹出次数 ≤ n。线性。',
      tags: ['complexity'],
    },
    {
      id: 'trapping-rain-water.q10',
      prompt: '空间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(log n)' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(n²)' },
      ],
      answer: 'c',
      explain:
        '严格递减的输入会把所有下标堆在栈里，最坏 O(n)。换成双指针法可降至 O(1)。',
      tags: ['complexity'],
    },
    {
      id: 'trapping-rain-water.q11',
      prompt: '若改用「按列计算法」（每列单独算 min(左最高, 右最高) - 自身），暴力实现的复杂度是？',
      options: [
        { id: 'a', text: 'O(n)' },
        { id: 'b', text: 'O(n²)——每列向左右各扫一遍取最大' },
        { id: 'c', text: 'O(n log n)' },
        { id: 'd', text: 'O(n³)' },
      ],
      answer: 'b',
      explain:
        '暴力按列法是 O(n²)。预处理 left_max[]/right_max[] 数组后可降到 O(n) 时间 + O(n) 空间——这就是「DP 解」。三种解法（DP / 单调栈 / 双指针）都常见。',
      tags: ['complexity'],
    },
    {
      id: 'trapping-rain-water.q12',
      prompt: '把 `bottom = stack.pop()` 与 `left = stack[-1]` 顺序写反会怎样？',
      options: [
        { id: 'a', text: '逻辑等价' },
        { id: 'b', text: '会拿错左墙——left 应是 bottom 弹出后的新栈顶' },
        { id: 'c', text: '语法错误' },
        { id: 'd', text: '会越界' },
      ],
      answer: 'b',
      explain:
        '左墙是 bottom 在栈中的「左邻居」，必须先 pop 掉 bottom 才能看到。如果先取 left = stack[-1]，那其实拿的就是 bottom 自己。这是该题写法的关键顺序。',
      tags: ['boundary', 'invariant'],
    },
  ],
}

export default problem
