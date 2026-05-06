import type { Problem } from '../types'

const code = `def twoSum(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []`

export const problem: Problem = {
  id: 'two-sum',
  leetcodeNo: 1,
  title: { zh: '两数之和', en: 'Two Sum' },
  difficulty: 'easy',
  pattern: 'hashmap',
  tags: ['array', 'hash-table'],
  statement:
    '给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出**和为目标值** `target` 的那两个整数，并返回它们的数组下标。\n\n你可以假设每种输入只会对应一个答案，并且**同一个元素在数组中不能使用两次**。你可以按任意顺序返回答案。',
  examples: [
    { input: 'nums = [2, 7, 11, 15], target = 9', output: '[0, 1]', note: 'nums[0] + nums[1] == 9' },
    { input: 'nums = [3, 2, 4], target = 6', output: '[1, 2]' },
    { input: 'nums = [3, 3], target = 6', output: '[0, 1]', note: '相同值时返回的是两个不同位置的索引' },
  ],
  constraints: [
    '2 ≤ nums.length ≤ 10⁴',
    '-10⁹ ≤ nums[i], target ≤ 10⁹',
    '只存在一个有效答案',
  ],
  intuition:
    '一遍遍历 + 哈希表。对每个 num，先查"target - num"是否已经见过；若见过就直接配对，否则把当前 num 记入哈希表（key=值，value=索引）。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(n)' },
  pythonRefIds: ['py-enumerate', 'py-dict-in'],
  microQuestions: [
    {
      id: 'two-sum.q1',
      prompt: '题目要求"返回索引"，那么数据结构 `seen` 应该是哪种最合适？',
      options: [
        { id: 'a', text: 'set —— 只需要"见过"这个布尔信息' },
        { id: 'b', text: 'dict —— key 是值，value 是索引' },
        { id: 'c', text: 'list —— 顺序追加，找配对再遍历' },
        { id: 'd', text: '排序后双指针，不需要额外结构' },
      ],
      answer: 'b',
      explain:
        'set 只能告诉你"配对值在不在"，但找不回索引；list 又退化为 O(n)；排序会丢失原始索引。dict 能在 O(1) 内同时回答"在不在"和"索引是几"。',
      tags: ['data-structure'],
    },
    {
      id: 'two-sum.q2',
      prompt: '初始化 `seen = ?`，应当填什么？',
      options: [
        { id: 'a', text: '{}' },
        { id: 'b', text: '[]' },
        { id: 'c', text: 'set()' },
        { id: 'd', text: 'None' },
      ],
      answer: 'a',
      explain: '`{}` 是空字典字面量；`set()` 才是空集合。Python 里 `{}` 永远是 dict 不是 set。',
      tags: ['syntax', 'pythonism'],
    },
    {
      id: 'two-sum.q3',
      prompt: '遍历 `nums` 同时拿到 `index` 与 `value`，最 Pythonic 的写法是？',
      codeContext: code,
      highlightLine: 3,
      options: [
        { id: 'a', text: 'for i in range(len(nums)): num = nums[i]' },
        { id: 'b', text: 'for i, num in enumerate(nums):' },
        { id: 'c', text: 'for num in nums: i = nums.index(num)' },
        { id: 'd', text: 'while i < len(nums): ...' },
      ],
      answer: 'b',
      explain:
        '`enumerate` 一步拿到 (i, num)，可读性最好。注意陷阱 c：`nums.index(num)` 是 O(n) 而且重复值时返回首次索引，会出 bug。',
      pythonRefIds: ['py-enumerate'],
      tags: ['pythonism'],
    },
    {
      id: 'two-sum.q4',
      prompt: '`enumerate(nums)` 默认从哪个索引开始？',
      options: [
        { id: 'a', text: '0' },
        { id: 'b', text: '1' },
        { id: 'c', text: '取决于 Python 版本' },
        { id: 'd', text: '需要显式传 start 参数' },
      ],
      answer: 'a',
      explain: '默认从 0 开始；可以用 `enumerate(nums, start=1)` 改起点。',
      pythonRefIds: ['py-enumerate'],
      tags: ['pythonism'],
    },
    {
      id: 'two-sum.q5',
      prompt: '`complement` 应该是 target 与 num 的什么关系？',
      codeContext: code,
      highlightLine: 4,
      options: [
        { id: 'a', text: 'target + num' },
        { id: 'b', text: 'target - num' },
        { id: 'c', text: 'num - target' },
        { id: 'd', text: 'abs(target - num)' },
      ],
      answer: 'b',
      explain:
        '我们要找的是与当前 num 配对、加起来等于 target 的那个值，所以 complement = target - num。',
      tags: ['invariant'],
    },
    {
      id: 'two-sum.q6',
      prompt: '检查 complement 是否已经存在于 seen 中，最快的写法是？',
      codeContext: code,
      highlightLine: 5,
      options: [
        { id: 'a', text: 'if complement in seen:' },
        { id: 'b', text: 'if seen.get(complement) != None:' },
        { id: 'c', text: 'if complement in seen.keys():' },
        { id: 'd', text: 'if complement in list(seen):' },
      ],
      answer: 'a',
      explain:
        '`in` 对 dict 直接走哈希查 key，O(1)。`seen.keys()` 在 Python 3 中是视图也是 O(1)，但多一次属性调用；`list(seen)` 会构造列表再线性查找，退化为 O(n)。',
      pythonRefIds: ['py-dict-in'],
      tags: ['complexity', 'pythonism'],
    },
    {
      id: 'two-sum.q7',
      prompt: '找到配对时，应当 `return ?`',
      codeContext: code,
      highlightLine: 6,
      options: [
        { id: 'a', text: '[i, seen[complement]]' },
        { id: 'b', text: '[seen[complement], i]' },
        { id: 'c', text: '[complement, num]' },
        { id: 'd', text: '[seen[num], seen[complement]]' },
      ],
      answer: 'b',
      explain:
        '题目要求两个索引（不是值）。seen[complement] 是更早遍历到的索引、i 是当前的，按惯例小索引在前更易读；选项 b 与 a 在某些题解里都被接受，但严格按"先入先出"语义 b 更稳。',
      tags: ['naming', 'invariant'],
    },
    {
      id: 'two-sum.q8',
      prompt: '`seen[num] = i` 这一句应当放在哪里？',
      codeContext: code,
      highlightLine: 7,
      options: [
        { id: 'a', text: '在 `if complement in seen:` 之前——先记录再查' },
        { id: 'b', text: '在 `if` 之后但和 return 同分支——只在没找到时记' },
        { id: 'c', text: '完全不需要——一开始就把整个 nums 都灌进 dict' },
        { id: 'd', text: '在循环结束后——一次性写入' },
      ],
      answer: 'b',
      explain:
        '必须"先查后存"。如果先存，自身就会被当成配对（例如 nums=[3,2,4], target=6 时，3 会和自己配对）。同时只在未命中分支存，命中分支已经 return 了。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'two-sum.q9',
      prompt: '为什么不能"先一次性把 nums 全部放入 dict，再遍历查 complement"？',
      options: [
        { id: 'a', text: '会超时' },
        { id: 'b', text: '存在重复值时后写的索引会覆盖前面的，导致丢解' },
        { id: 'c', text: 'Python 字典不支持整数 key' },
        { id: 'd', text: '空间复杂度会变成 O(n²)' },
      ],
      answer: 'b',
      explain:
        '例如 nums=[3,3], target=6：先全部入 dict 时 {3:1}（覆盖了 {3:0}），找 complement=3 时只能拿到自身索引 1，无法构造 [0,1]。',
      tags: ['boundary'],
    },
    {
      id: 'two-sum.q10',
      prompt: '该解法的时间复杂度是？',
      options: [
        { id: 'a', text: 'O(n²)' },
        { id: 'b', text: 'O(n log n)' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(1)' },
      ],
      answer: 'c',
      explain:
        '一次遍历 n 个元素，每一步 dict in / dict 写入都是均摊 O(1)，总 O(n)。',
      tags: ['complexity'],
    },
    {
      id: 'two-sum.q11',
      prompt: '该解法的空间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(log n)' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(n²)' },
      ],
      answer: 'c',
      explain:
        '最坏情况下 dict 装入近 n 个元素（直到最后一对才匹配），所以 O(n)。',
      tags: ['complexity'],
    },
    {
      id: 'two-sum.q12',
      prompt:
        '如果题目改成"返回两个数本身（而非索引）"且数组允许排序，最优解会变成？',
      options: [
        { id: 'a', text: '依然 dict 一遍法，无变化' },
        { id: 'b', text: '排序 + 双指针，空间 O(1)（不计排序栈）' },
        { id: 'c', text: '暴力 O(n²)' },
        { id: 'd', text: '二分查找' },
      ],
      answer: 'b',
      explain:
        '一旦不在乎索引、可以破坏顺序，就能用经典"对撞双指针"，时间 O(n log n)（排序），空间 O(1)。这正是为什么本题要求返回索引——逼你用 hash 而不是排序。',
      tags: ['pythonism', 'data-structure'],
    },
    {
      id: 'two-sum.q13',
      prompt:
        '题目"假设每种输入只会对应一个答案，且不能重复使用同一个元素"——这句话最关键的是？',
      options: [
        { id: 'a', text: '允许我们提前 return，不必收集所有解' },
        { id: 'b', text: '不能 return [i, i]——同一索引不能用两次' },
        { id: 'c', text: '不要求返回顺序' },
        { id: 'd', text: 'a 和 b 都是关键' },
      ],
      answer: 'd',
      explain:
        '"唯一答案"让我们一找到就能 return；"不能重复元素"是 q8 的核心理由：seen 必须只包含**当前位置之前**的元素，所以要先查后存。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'two-sum.q14',
      prompt: '若全部循环结束都没找到，应当返回？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: 'return []' },
        { id: 'b', text: 'return None' },
        { id: 'c', text: 'raise ValueError' },
        { id: 'd', text: '题目保证有解，不需要写' },
      ],
      answer: 'a',
      explain:
        '题目"保证有唯一解"，理论上不会执行到这里。但工程实践中保留 `return []` 让函数签名一致（永远返回 list），方便调用方。面试时也可以直接说"题目保证有解，可以省略"——两种都对。',
      tags: ['boundary'],
    },
  ],
}

export default problem
