import type { Problem } from '../types'

const code = `def dailyTemperatures(temperatures: list[int]) -> list[int]:
    n = len(temperatures)
    answer = [0] * n
    stack = []  # 栈里存「还在等更高温」的下标
    for i, t in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < t:
            j = stack.pop()
            answer[j] = i - j
        stack.append(i)
    return answer`

export const problem: Problem = {
  id: 'daily-temperatures',
  leetcodeNo: 739,
  title: { zh: '每日温度', en: 'Daily Temperatures' },
  difficulty: 'medium',
  pattern: 'monotonic-stack',
  tags: ['array', 'stack', 'monotonic-stack'],
  statement:
    '给定一个整数数组 `temperatures`，表示每天的温度，返回一个数组 `answer`，其中 `answer[i]` 是指对于第 `i` 天，下一个更高温度出现在几天后。\n\n如果气温在这之后都**不会升高**，请在该位置用 `0` 来代替。',
  examples: [
    {
      input: 'temperatures = [73,74,75,71,69,72,76,73]',
      output: '[1,1,4,2,1,1,0,0]',
      note: '第 0 天 73，第 1 天 74 就更高，所以 answer[0]=1',
    },
    { input: 'temperatures = [30,40,50,60]', output: '[1,1,1,0]' },
    { input: 'temperatures = [30,60,90]', output: '[1,1,0]' },
  ],
  constraints: [
    '1 ≤ temperatures.length ≤ 10⁵',
    '30 ≤ temperatures[i] ≤ 100',
  ],
  intuition:
    '单调递减栈存「还在等更高温」的下标。今天比栈顶热 → 栈顶找到了答案：弹出并填写 i - j；否则把今天压入栈继续等。每个下标至多入栈、出栈各一次，整体 O(n)。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(n)' },
  microQuestions: [
    {
      id: 'daily-temperatures.q1',
      prompt: '栈里应该存「温度值」还是「下标」？',
      codeContext: code,
      highlightLine: 4,
      options: [
        { id: 'a', text: '存温度值——比较时直接拿' },
        { id: 'b', text: '存下标——既能 O(1) 取到温度，又能算出「几天后」' },
        { id: 'c', text: '存 (温度, 下标) 元组' },
        { id: 'd', text: '存温度值，下标用 enumerate 实时维护' },
      ],
      answer: 'b',
      explain:
        '答案要求的是「天数差」，所以必须有下标；而通过 `temperatures[stack[-1]]` 也能立刻取到温度。存 tuple 是冗余的；只存值就丢了位置。',
      tags: ['data-structure', 'invariant'],
    },
    {
      id: 'daily-temperatures.q2',
      prompt: '`answer = [0] * n` 这一行的作用是？',
      codeContext: code,
      highlightLine: 3,
      options: [
        { id: 'a', text: '只是初始化，可以省略' },
        { id: 'b', text: '默认填 0，正好对应「之后没有更高温」的语义' },
        { id: 'c', text: '为了让数组长度等于 n' },
        { id: 'd', text: 'b 和 c 都是关键' },
      ],
      answer: 'd',
      explain:
        '初值 0 与题目要求「没有更高温就用 0」恰好吻合——循环结束后还留在栈里的下标，根本不会被赋值，自然保持 0。这是一个借「默认值」省一段代码的小技巧。',
      tags: ['invariant', 'pythonism'],
    },
    {
      id: 'daily-temperatures.q3',
      prompt: '应该维护「单调递增栈」还是「单调递减栈」？',
      options: [
        { id: 'a', text: '单调递增栈（栈底到栈顶递增）' },
        { id: 'b', text: '单调递减栈（栈底到栈顶递减）' },
        { id: 'c', text: '只要单调就行' },
        { id: 'd', text: '看输入决定' },
      ],
      answer: 'b',
      explain:
        '我们要找「下一个更高温」，所以遇到比栈顶高的就弹（清算）；那栈里剩下的就是「还没找到更高的」，自顶向底必然非递减——即从栈底到栈顶是递减栈。',
      tags: ['invariant', 'data-structure'],
    },
    {
      id: 'daily-temperatures.q4',
      prompt: 'while 循环条件应该是？',
      codeContext: code,
      highlightLine: 6,
      options: [
        { id: 'a', text: 'while stack and temperatures[stack[-1]] < t' },
        { id: 'b', text: 'while stack and temperatures[stack[-1]] <= t' },
        { id: 'c', text: 'while stack and temperatures[stack[-1]] > t' },
        { id: 'd', text: 'while temperatures[stack[-1]] < t' },
      ],
      answer: 'a',
      explain:
        '严格 `<`：题目说「更高」，相等不算；用 `<=` 会把等温的天也清算，结果偏大。`stack and` 不可省，否则空栈时 `stack[-1]` 会 IndexError。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'daily-temperatures.q5',
      prompt: '弹出栈顶 `j` 后，要给 `answer[j]` 写入什么？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: 'i' },
        { id: 'b', text: 't - temperatures[j]' },
        { id: 'c', text: 'i - j' },
        { id: 'd', text: 'j - i' },
      ],
      answer: 'c',
      explain:
        '题目要的是「几天后」，即两个下标的差。`i - j` 永远为正（因为 j 早于 i 入栈）。',
      tags: ['invariant'],
    },
    {
      id: 'daily-temperatures.q6',
      prompt: '`stack.append(i)` 应放在 while 之前还是之后？',
      codeContext: code,
      highlightLine: 9,
      options: [
        { id: 'a', text: '放在 while 之前——先进栈再清算' },
        { id: 'b', text: '放在 while 之后——清算完再入栈' },
        { id: 'c', text: '都行' },
        { id: 'd', text: '只在没人需要清算时才入栈' },
      ],
      answer: 'b',
      explain:
        '若放前面，当前下标会立刻参与与自己的比较；正确顺序是：今天 i 先把所有「等到更高温」的旧下标结算掉，再让自己进入「等待队列」。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'daily-temperatures.q7',
      prompt: '循环结束后栈里还可能剩下一些下标，它们对应的 answer 值会是？',
      options: [
        { id: 'a', text: '需要再写一段循环置 0' },
        { id: 'b', text: '已经在初始化时是 0，不用再处理' },
        { id: 'c', text: '会保持上一次循环的值' },
        { id: 'd', text: '会是 None' },
      ],
      answer: 'b',
      explain:
        '`answer = [0]*n` 已经把没找到更高温的位置预填为 0；栈里剩下的下标永远没人弹它们，answer 自然保留初始 0。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'daily-temperatures.q8',
      prompt: '时间复杂度是？',
      options: [
        { id: 'a', text: 'O(n²)' },
        { id: 'b', text: 'O(n log n)' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(n³)' },
      ],
      answer: 'c',
      explain:
        '虽然有 while 嵌套在 for 里，但每个下标最多入栈一次、出栈一次。总操作数 ≤ 2n → 摊还 O(n)。这是单调栈分析的关键一步。',
      tags: ['complexity'],
    },
    {
      id: 'daily-temperatures.q9',
      prompt: '空间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(log n)' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(n²)' },
      ],
      answer: 'c',
      explain:
        '极端情况（严格递减温度数组）下，所有下标都会留在栈里——栈最大尺寸为 n。answer 本身也是 O(n) 但通常视作输出。',
      tags: ['complexity'],
    },
    {
      id: 'daily-temperatures.q10',
      prompt: '若改成「找下一个温度 ≥ 今天的天数」，需要改哪一处？',
      options: [
        { id: 'a', text: '把 `<` 改成 `<=`' },
        { id: 'b', text: '把栈改成单调递增栈' },
        { id: 'c', text: '把 answer 初值改成 -1' },
        { id: 'd', text: '加一个二分查找' },
      ],
      answer: 'a',
      explain:
        '题目从「严格更高」变成「不更低」，比较谓词放宽即可：`temperatures[stack[-1]] <= t` 时就清算。其他逻辑全不变。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'daily-temperatures.q11',
      prompt: '若用「暴力法」每个 i 向后扫到第一个更高的，复杂度是？',
      options: [
        { id: 'a', text: 'O(n)' },
        { id: 'b', text: 'O(n log n)' },
        { id: 'c', text: 'O(n²)' },
        { id: 'd', text: 'O(n³)' },
      ],
      answer: 'c',
      explain:
        '最坏情况（如严格递减数组）每个 i 都扫到末尾，总 O(n²)。单调栈正是把「重复扫」用「记忆化栈」省掉，是 n² → n 的经典优化。',
      tags: ['complexity'],
    },
    {
      id: 'daily-temperatures.q12',
      prompt: 'Python 用 `list` 当栈，`pop()` 与 `append()` 的复杂度是？',
      options: [
        { id: 'a', text: '两者都是 O(n)' },
        { id: 'b', text: '`pop()` 是 O(n)，`append()` 是 O(1)' },
        { id: 'c', text: '两者都是 O(1) 摊还' },
        { id: 'd', text: '取决于元素类型' },
      ],
      answer: 'c',
      explain:
        '`list.append` 与 `list.pop()`（不带下标，从尾部弹出）都是均摊 O(1)。`list.pop(0)` 才是 O(n)；如果一定要从队首弹用 `collections.deque`。',
      tags: ['pythonism', 'complexity'],
    },
  ],
}

export default problem
