import type { Problem } from '../types'

const code = `def singleNumber(nums: list[int]) -> int:
    ans = 0
    for x in nums:
        ans ^= x      # 出现两次的会互相抵消（a ^ a = 0），剩下的就是只出现一次的
    return ans`

export const problem: Problem = {
  id: 'single-number',
  leetcodeNo: 136,
  title: { zh: '只出现一次的数字', en: 'Single Number' },
  difficulty: 'easy',
  pattern: 'bit-prefix',
  tags: ['array', 'bit-manipulation', 'xor'],
  statement:
    '给你一个**非空**整数数组 `nums`，除了某个元素**只出现一次**以外，其余每个元素均出现**两次**。找出那个只出现了一次的元素。\n\n你必须设计并实现**线性时间复杂度**的算法来解决此问题，且该算法只使用**常量额外空间**。',
  examples: [
    { input: 'nums = [2,2,1]', output: '1' },
    { input: 'nums = [4,1,2,1,2]', output: '4' },
    { input: 'nums = [1]', output: '1' },
  ],
  constraints: [
    '1 ≤ nums.length ≤ 3 × 10⁴',
    '-3 × 10⁴ ≤ nums[i] ≤ 3 × 10⁴',
    '除了某个元素只出现一次以外，数组中每个元素都恰好出现两次',
  ],
  intuition:
    'XOR 三个性质：① `a ^ a = 0`；② `a ^ 0 = a`；③ 满足交换律和结合律。把所有元素 XOR 起来，相同的两两抵消，剩下的就是只出现一次的那个。O(n) 时间、O(1) 空间，比哈希表更优雅。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(1)' },
  microQuestions: [
    {
      id: 'single-number.q1',
      prompt: '题目要求「线性时间 + 常量空间」直接排除了哪个朴素解？',
      options: [
        { id: 'a', text: '哈希表计数（O(n) 空间）' },
        { id: 'b', text: '排序后找相邻不等（O(n log n) 时间）' },
        { id: 'c', text: '上述两个都被排除' },
        { id: 'd', text: '只有暴力 O(n²) 被排除' },
      ],
      answer: 'c',
      explain:
        '硬性约束：哈希表违反 O(1) 空间、排序违反 O(n) 时间。题目其实是在「逼你用位运算」——这是面试出 XOR 题的常用引导方式。',
      tags: ['complexity'],
    },
    {
      id: 'single-number.q2',
      prompt: '`a ^ a` 的结果是？',
      options: [
        { id: 'a', text: '2a' },
        { id: 'b', text: '0' },
        { id: 'c', text: 'a' },
        { id: 'd', text: '取决于 a 的值' },
      ],
      answer: 'b',
      explain:
        'XOR 的核心定义：相同位异或为 0、不同位异或为 1。同一个数和自己 XOR 每一位都相同 → 结果为 0。这是「成对消除」的基石。',
      tags: ['invariant'],
    },
    {
      id: 'single-number.q3',
      prompt: '`a ^ 0` 的结果是？',
      options: [
        { id: 'a', text: '0' },
        { id: 'b', text: 'a' },
        { id: 'c', text: '取决于 a' },
        { id: 'd', text: '1' },
      ],
      answer: 'b',
      explain:
        '0 是 XOR 的「单位元」。`a ^ 0 = a` 让我们可以用 `ans = 0` 起步、逐个 XOR 累积——和 `ans = 1`、`ans *= x` 这种乘累积写法对应。',
      tags: ['invariant'],
    },
    {
      id: 'single-number.q4',
      prompt: 'XOR 是否满足交换律 `a ^ b = b ^ a` 和结合律 `(a ^ b) ^ c = a ^ (b ^ c)`？',
      options: [
        { id: 'a', text: '满足' },
        { id: 'b', text: '只满足交换律' },
        { id: 'c', text: '只满足结合律' },
        { id: 'd', text: '都不满足' },
      ],
      answer: 'a',
      explain:
        '交换律 + 结合律是 XOR 解题的关键——它意味着「无论 nums 的顺序如何、相同的元素都会两两抵消」，无需排序。',
      tags: ['invariant'],
    },
    {
      id: 'single-number.q5',
      prompt: '`ans = 0` 起步是否必须？',
      codeContext: code,
      highlightLine: 2,
      options: [
        { id: 'a', text: '必须——0 是 XOR 的单位元，从其它值起步会污染答案' },
        { id: 'b', text: '随便，1 也行' },
        { id: 'c', text: '不需要起步' },
        { id: 'd', text: '必须 ans = nums[0]' },
      ],
      answer: 'a',
      explain:
        '`a ^ 0 = a`，从 0 起步等于「干干净净从空集合开始 XOR」。如果非要从 nums[0] 起步，循环就要从 nums[1:] 开始——更易写错。',
      tags: ['boundary'],
    },
    {
      id: 'single-number.q6',
      prompt: '`ans ^= x` 是哪种操作的简写？',
      codeContext: code,
      highlightLine: 4,
      options: [
        { id: 'a', text: 'ans = ans ^ x' },
        { id: 'b', text: 'ans = ans + x' },
        { id: 'c', text: 'ans = ans * x' },
        { id: 'd', text: 'ans 与 x 求并集' },
      ],
      answer: 'a',
      explain:
        '复合赋值运算符 `^=` 是 `ans = ans ^ x` 的简写。`+=`/`*=` 也类似。Python 里这种写法很常见。',
      tags: ['syntax', 'pythonism'],
    },
    {
      id: 'single-number.q7',
      prompt: '`functools.reduce(operator.xor, nums)` 与本题循环版的关系？',
      options: [
        { id: 'a', text: '一行函数式写法，等价于 ans = 0; for x in nums: ans ^= x' },
        { id: 'b', text: '更慢' },
        { id: 'c', text: '只能用于浮点数' },
        { id: 'd', text: '不存在这个函数' },
      ],
      answer: 'a',
      explain:
        '`reduce(op, iter, init)` 把二元操作迭代地折叠到序列上。XOR 题写成 `reduce(xor, nums, 0)` 是 Python 风的极简表达——理解原理即可，面试时手写 for 循环更稳。',
      tags: ['pythonism'],
    },
    {
      id: 'single-number.q8',
      prompt: '若题目改为「除某元素出现一次外、其余出现 3 次」（LC137），XOR 还能直接用吗？',
      options: [
        { id: 'a', text: '不能直接用——`a ^ a ^ a = a` 不会自然消掉；需要按位计数取 mod 3，或两个 mask 巧合' },
        { id: 'b', text: '能，写法完全相同' },
        { id: 'c', text: '只能用哈希表' },
        { id: 'd', text: '一定要排序' },
      ],
      answer: 'a',
      explain:
        'XOR 等价于「对每个二进制位算 mod 2」——只对成对消除有效。LC137 需要「mod 3」逻辑，可以按 32 位分别计数 mod 3 还原；或用「ones / twos」两个状态变量的更高级技巧。',
      tags: ['invariant'],
    },
    {
      id: 'single-number.q9',
      prompt: '若数组改为「**两个**数只出现一次，其余出现两次」（LC260），思路？',
      options: [
        { id: 'a', text: '先全 XOR 得到 a^b；找一个为 1 的位把数组分两组分别 XOR' },
        { id: 'b', text: '直接 XOR 即可' },
        { id: 'c', text: '哈希表' },
        { id: 'd', text: '排序' },
      ],
      answer: 'a',
      explain:
        '经典进阶：`a ^ b` 必有某位为 1（说明 a、b 在该位不同）；按这一位是否为 1 把数组分两组，每组只剩一个孤数，分别 XOR 即可。',
      tags: ['invariant'],
    },
    {
      id: 'single-number.q10',
      prompt: '本题对负数也有效吗？',
      options: [
        { id: 'a', text: '有效——XOR 是按位运算，对补码表示的负数同样成立' },
        { id: 'b', text: '只对正数' },
        { id: 'c', text: '只对非零' },
        { id: 'd', text: 'Python 里负数不支持 XOR' },
      ],
      answer: 'a',
      explain:
        'XOR 在补码（two\'s complement）上也满足 `a ^ a = 0`。Python 的整数是任意精度但负数运算用补码语义，`(-3) ^ (-3) == 0` 成立。',
      tags: ['boundary'],
    },
    {
      id: 'single-number.q11',
      prompt: '空间复杂度是？',
      options: [
        { id: 'a', text: 'O(1) — 只有一个累积变量 ans' },
        { id: 'b', text: 'O(n)' },
        { id: 'c', text: 'O(log n)' },
        { id: 'd', text: 'O(n²)' },
      ],
      answer: 'a',
      explain:
        '只用一个 int 变量做 XOR 累积，符合题目「常量额外空间」的硬约束。',
      tags: ['complexity'],
    },
  ],
}

export default problem
