import type { Problem } from '../types'

const code = `def threeSum(nums: list[int]) -> list[list[int]]:
    nums.sort()
    n = len(nums)
    res = []
    for i in range(n - 2):
        if nums[i] > 0:
            break
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, n - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s == 0:
                res.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1
    return res`

export const problem: Problem = {
  id: 'three-sum',
  leetcodeNo: 15,
  title: { zh: '三数之和', en: '3Sum' },
  difficulty: 'medium',
  pattern: 'two-pointer',
  tags: ['array', 'two-pointers', 'sorting'],
  statement:
    '给你一个整数数组 `nums`，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k` 且 `j != k`，同时还满足 `nums[i] + nums[j] + nums[k] == 0`。\n\n请你返回所有和为 0 且**不重复**的三元组。\n\n注意：答案中不可以包含重复的三元组。',
  examples: [
    { input: 'nums = [-1,0,1,2,-1,-4]', output: '[[-1,-1,2],[-1,0,1]]' },
    { input: 'nums = [0,1,1]', output: '[]' },
    { input: 'nums = [0,0,0]', output: '[[0,0,0]]', note: '同值但来自不同位置算合法' },
  ],
  constraints: [
    '3 ≤ nums.length ≤ 3000',
    '-10⁵ ≤ nums[i] ≤ 10⁵',
  ],
  intuition:
    '排序 + 固定一个数 + 双指针。固定 nums[i]，在 i+1..n-1 上用对撞双指针找两数之和 = -nums[i]。去重三处：i 跳过相同、找到解后 left/right 各自跳过相同。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n²)', space: 'O(1) 不计排序' },
  pythonRefIds: ['py-list-sort'],
  microQuestions: [
    {
      id: '3sum.q1',
      prompt: '为什么第一步要 `nums.sort()`？',
      codeContext: code,
      highlightLine: 2,
      options: [
        { id: 'a', text: '让结果更易读' },
        { id: 'b', text: '排序后才能用双指针、且方便去重' },
        { id: 'c', text: '排序使时间复杂度降到 O(n log n)' },
        { id: 'd', text: '题目要求结果按字典序输出' },
      ],
      answer: 'b',
      explain:
        '双指针的前提是单调；同时排序后相同元素相邻，去重只需"跳过和上一个相同"。',
      tags: ['invariant'],
    },
    {
      id: '3sum.q2',
      prompt: 'Python 里 `nums.sort()` 与 `sorted(nums)` 的区别？',
      options: [
        { id: 'a', text: '完全等价' },
        { id: 'b', text: '`.sort()` 原地修改返回 None；`sorted()` 返回新列表' },
        { id: 'c', text: '`.sort()` 是稳定的，`sorted()` 不稳定' },
        { id: 'd', text: '`.sort()` 只能用于数字' },
      ],
      answer: 'b',
      explain:
        '这是 Python 高频考点。这里用 `.sort()` 因为我们要原地排序、不需要保留原顺序；它返回 None，所以不能写 `nums = nums.sort()`（那样 nums 会变 None）。',
      pythonRefIds: ['py-list-sort'],
      tags: ['pythonism'],
    },
    {
      id: '3sum.q3',
      prompt: '`for i in range(n - 2)` 为什么是 `n - 2` 而不是 `n`？',
      codeContext: code,
      highlightLine: 5,
      options: [
        { id: 'a', text: '随便，`range(n)` 也行' },
        { id: 'b', text: '需要给 left=i+1, right=n-1 留出至少两个位置' },
        { id: 'c', text: '为了 O(n²) 的复杂度' },
        { id: 'd', text: '避免越界访问 nums[i+2]' },
      ],
      answer: 'b',
      explain:
        '当 i = n-2 时 left = n-1, right = n-1，`while left < right` 直接退出；当 i = n-3 才有意义。所以 `range(n-2)` 排除最后两个无效起点。',
      tags: ['boundary'],
    },
    {
      id: '3sum.q4',
      prompt: '`if nums[i] > 0: break` 这个剪枝的依据是？',
      codeContext: code,
      highlightLine: 6,
      options: [
        { id: 'a', text: 'nums[i] > 0 时不可能再有三数和为 0（已排序，后面只会更大）' },
        { id: 'b', text: '题目保证答案非空' },
        { id: 'c', text: '防止溢出' },
        { id: 'd', text: '随便加的，没必要' },
      ],
      answer: 'a',
      explain:
        '排序后 nums[i] > 0 ⇒ left 与 right 也都 > 0，三个正数加不出 0。直接 break 提前结束循环。',
      tags: ['invariant'],
    },
    {
      id: '3sum.q5',
      prompt: '外层去重 `if i > 0 and nums[i] == nums[i-1]: continue` 的关键点？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: '`i > 0` 是为了保护 i-1 不越界' },
        { id: 'b', text: '只有 i > 0 才去重——i=0 时没有"上一个"可比' },
        { id: 'c', text: '`continue` 跳过整个三元组的搜索，不丢解' },
        { id: 'd', text: '以上都对' },
      ],
      answer: 'd',
      explain:
        '三个细节缺一不可：① 边界保护；② 第一个数没有前一个所以必须 i>0 才查；③ 用 continue 而非 break 因为后面 i 还能产生新解。',
      tags: ['boundary', 'pythonism'],
    },
    {
      id: '3sum.q6',
      prompt: '内层双指针的初始化应该是？',
      codeContext: code,
      highlightLine: 9,
      options: [
        { id: 'a', text: 'left, right = 0, n - 1' },
        { id: 'b', text: 'left, right = i, n - 1' },
        { id: 'c', text: 'left, right = i + 1, n - 1' },
        { id: 'd', text: 'left, right = i + 1, n' },
      ],
      answer: 'c',
      explain:
        'left 必须从 i+1 开始，避免重复使用 nums[i]；right 是合法索引 n-1。',
      tags: ['boundary'],
    },
    {
      id: '3sum.q7',
      prompt: '当 s = 0（找到一组解）后，立刻只做 `left += 1; right -= 1` 是否够？',
      options: [
        { id: 'a', text: '够，每次只跳一格也能找到所有解' },
        { id: 'b', text: '不够，会产生重复三元组——必须先跳过和当前 left/right 相同的所有元素' },
        { id: 'c', text: '不够，应该 break 退出内循环' },
        { id: 'd', text: '不够，应该跳到下一个 i' },
      ],
      answer: 'b',
      explain:
        '例如 [-2,-2,0,0,2,2]，固定 i=0 (-2) 后会找到 (-2,0,2)；如果只移一格，下一次 (left=2,right=4) 还会找到 (-2,0,2)。所以解后要 while 跳过相同。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: '3sum.q8',
      prompt: 's < 0 时应该移动谁？',
      codeContext: code,
      highlightLine: 19,
      options: [
        { id: 'a', text: 'left += 1（让和变大）' },
        { id: 'b', text: 'right -= 1（让和变小）' },
        { id: 'c', text: '两边同时移动' },
        { id: 'd', text: '退出循环' },
      ],
      answer: 'a',
      explain:
        '排序后右移 left 会让 nums[left] 不减，所以 s 不减——朝目标 0 靠近。',
      tags: ['invariant'],
    },
    {
      id: '3sum.q9',
      prompt: '时间复杂度是？',
      options: [
        { id: 'a', text: 'O(n)' },
        { id: 'b', text: 'O(n log n)' },
        { id: 'c', text: 'O(n²)' },
        { id: 'd', text: 'O(n³)' },
      ],
      answer: 'c',
      explain:
        '排序 O(n log n) + 外层 n 次 × 内层双指针 O(n) = O(n²)。比暴力 O(n³) 优。',
      tags: ['complexity'],
    },
    {
      id: '3sum.q10',
      prompt: '空间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)（不计排序栈和输出）' },
        { id: 'b', text: 'O(n)' },
        { id: 'c', text: 'O(n²)' },
        { id: 'd', text: 'O(log n)' },
      ],
      answer: 'a',
      explain:
        '通常约定不计排序额外栈空间（Python Timsort 平均 O(n)）和输出 res 本身。算法本身只用常数额外指针。',
      tags: ['complexity'],
    },
    {
      id: '3sum.q11',
      prompt: '能否改为"用 set 去重最后输出"代替三处去重？',
      options: [
        { id: 'a', text: '可以但效率低且依赖 tuple/frozenset 转换' },
        { id: 'b', text: '不行，结果会错' },
        { id: 'c', text: '完全等价，没区别' },
        { id: 'd', text: 'Python 的 set 不支持 list 元素' },
      ],
      answer: 'a',
      explain:
        '"暴力收集 + 最后 set 去重"在功能上能 AC，但既要把每个解转成 tuple、又拖慢速度——面试时显得不熟练。三处 while 跳过是更"会写"的标志。',
      tags: ['pythonism'],
    },
  ],
}

export default problem
