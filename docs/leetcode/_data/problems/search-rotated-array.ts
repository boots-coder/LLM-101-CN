import type { Problem } from '../types'

const code = `def search(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        # 判断左半 [left, mid] 是否有序
        if nums[left] <= nums[mid]:
            # 左半有序
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            # 右半 [mid, right] 有序
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1`

export const problem: Problem = {
  id: 'search-rotated-array',
  leetcodeNo: 33,
  title: { zh: '搜索旋转排序数组', en: 'Search in Rotated Sorted Array' },
  difficulty: 'medium',
  pattern: 'binary-search',
  tags: ['array', 'binary-search'],
  statement:
    '整数数组 `nums` 按升序排列，数组中的值**互不相同**。在传递给函数之前，`nums` 在**预先未知的某个下标 `k`**（`0 <= k < nums.length`）上进行了**旋转**，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`（下标从 0 开始计数）。\n\n给你**旋转后**的数组 `nums` 和一个整数 `target`，如果 `nums` 中存在这个目标值 `target`，则返回它的下标，否则返回 `-1`。\n\n你必须设计一个时间复杂度为 `O(log n)` 的算法解决此问题。',
  examples: [
    { input: 'nums = [4,5,6,7,0,1,2], target = 0', output: '4' },
    { input: 'nums = [4,5,6,7,0,1,2], target = 3', output: '-1' },
    { input: 'nums = [1], target = 0', output: '-1' },
  ],
  constraints: [
    '1 ≤ nums.length ≤ 5000',
    '-10⁴ ≤ nums[i], target ≤ 10⁴',
    'nums 中的每个值都独一无二',
    'nums 在某个未知下标处旋转',
  ],
  intuition:
    '旋转数组虽然整体不有序，但每次取 mid 后，左半 [left, mid] 与右半 [mid, right] 中**至少有一半是有序的**。先判出哪一半有序，再判 target 是否落在该有序段的端点之间——是就走那一半（O(log n) 二分），否则走另一半。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(log n)', space: 'O(1)' },
  pythonRefIds: ['py-chained-compare'],
  microQuestions: [
    {
      id: 'search-rotated-array.q1',
      prompt: '为什么不能直接对旋转后的 nums 跑普通二分？',
      options: [
        { id: 'a', text: '可以，没问题' },
        { id: 'b', text: '整体不再单调——nums[mid] 与 target 的大小关系无法决定走左还是走右' },
        { id: 'c', text: '会越界' },
        { id: 'd', text: 'Python 实现限制' },
      ],
      answer: 'b',
      explain:
        '普通二分依赖单调性："target < nums[mid] 一定在左侧"。旋转破坏了这个 invariant，所以需要先判出哪一半有序、再用单调性。',
      tags: ['invariant'],
    },
    {
      id: 'search-rotated-array.q2',
      prompt: 'mid 切开后，关于"哪一半有序"的关键 invariant 是？',
      options: [
        { id: 'a', text: '两半都有序' },
        { id: 'b', text: '至少有一半是有序的（值互不相同时严格成立）' },
        { id: 'c', text: '随机有序' },
        { id: 'd', text: '只有右半有序' },
      ],
      answer: 'b',
      explain:
        '旋转点最多落在某一半内；另一半被截在旋转点之外，必定是连续递增的。这是本题二分能 O(log n) 的根本。',
      tags: ['invariant'],
    },
    {
      id: 'search-rotated-array.q3',
      prompt: '判断"左半有序"的条件应该是？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: 'nums[left] < nums[mid]' },
        { id: 'b', text: 'nums[left] <= nums[mid]' },
        { id: 'c', text: 'nums[mid] < nums[right]' },
        { id: 'd', text: 'nums[mid] > target' },
      ],
      answer: 'b',
      explain:
        '当 left == mid 时（区间只剩两个元素），nums[left] == nums[mid] 也算"左半有序"（实际只一个元素，平凡有序）。所以用 `<=` 兼容这种边界。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'search-rotated-array.q4',
      prompt: '左半有序时，target 在左半的判定区间应该是？',
      codeContext: code,
      highlightLine: 10,
      options: [
        { id: 'a', text: 'nums[left] <= target <= nums[mid]' },
        { id: 'b', text: 'nums[left] <= target < nums[mid]' },
        { id: 'c', text: 'nums[left] < target < nums[mid]' },
        { id: 'd', text: 'nums[left] < target <= nums[mid]' },
      ],
      answer: 'b',
      explain:
        '上界用 `<` 是因为 target == nums[mid] 已经在循环开头被独立返回。包含 nums[left] 是因为 target 可能恰好等于左端点。',
      tags: ['boundary'],
    },
    {
      id: 'search-rotated-array.q5',
      prompt: '右半有序时，target 在右半的判定区间应该是？',
      codeContext: code,
      highlightLine: 15,
      options: [
        { id: 'a', text: 'nums[mid] <= target <= nums[right]' },
        { id: 'b', text: 'nums[mid] < target <= nums[right]' },
        { id: 'c', text: 'nums[mid] < target < nums[right]' },
        { id: 'd', text: 'nums[mid] <= target < nums[right]' },
      ],
      answer: 'b',
      explain:
        '下界用 `<` 是因为 target == nums[mid] 也已被独立处理；上界包含 nums[right]。注意与 q4 是"对偶但不完全镜像"的——开闭关系要扣细。',
      tags: ['boundary'],
    },
    {
      id: 'search-rotated-array.q6',
      prompt: 'Python 里 `nums[left] <= target < nums[mid]` 这种写法叫什么？',
      options: [
        { id: 'a', text: 'Python 链式比较' },
        { id: 'b', text: '会被解析为 (nums[left] <= target) < nums[mid]，先得 bool' },
        { id: 'c', text: '语法错误' },
        { id: 'd', text: '只能用 and 连接' },
      ],
      answer: 'a',
      explain:
        '这是 Python 特性：`a <= b < c` 等价于 `a <= b and b < c`，且 `b` 只求值一次。比 `nums[left] <= target and target < nums[mid]` 更简洁。',
      tags: ['pythonism', 'syntax'],
    },
    {
      id: 'search-rotated-array.q7',
      prompt: 'mid 用 `(left + right) // 2` 在大整数语言里要担心溢出，Python 里呢？',
      codeContext: code,
      highlightLine: 4,
      options: [
        { id: 'a', text: '同样会溢出' },
        { id: 'b', text: 'Python int 是任意精度的，不溢出；但写 `left + (right - left) // 2` 是面试加分项（跨语言习惯）' },
        { id: 'c', text: 'Python 一定要写 `left + (right - left) // 2`' },
        { id: 'd', text: '会精度丢失' },
      ],
      answer: 'b',
      explain:
        'Python 没有 32/64 位 int 限制；但写 `left + (right - left) // 2` 在 Java/C++ 中是必写惯例，面试时随手这么写显得专业。',
      tags: ['pythonism', 'boundary'],
    },
    {
      id: 'search-rotated-array.q8',
      prompt: '循环条件 `while left <= right` 与 `while left < right` 哪个对？',
      codeContext: code,
      highlightLine: 3,
      options: [
        { id: 'a', text: '只能 `<=`' },
        { id: 'b', text: '只能 `<`' },
        { id: 'c', text: '取决于区间是闭区间还是左闭右开——闭区间用 `<=`，本题用闭区间' },
        { id: 'd', text: '都能 AC' },
      ],
      answer: 'c',
      explain:
        '本题用闭区间 [left, right]，`right = len-1`。当 left == right 时还有一个元素未检查，所以条件必须是 `<=`。`<` 会漏单元素情形。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'search-rotated-array.q9',
      prompt: '若数组未旋转（k=0），算法是否仍然正确？',
      options: [
        { id: 'a', text: '不正确，需要特判' },
        { id: 'b', text: '正确——`nums[left] <= nums[mid]` 始终成立，永远走"左半有序"分支，等价于普通二分' },
        { id: 'c', text: '会死循环' },
        { id: 'd', text: '取决于 target' },
      ],
      answer: 'b',
      explain:
        '这是判断算法是否优雅的细节：边界（无旋转、整段旋转）应自然退化为普通二分而无需特判。',
      tags: ['boundary'],
    },
    {
      id: 'search-rotated-array.q10',
      prompt: '时间复杂度是？',
      options: [
        { id: 'a', text: 'O(n)' },
        { id: 'b', text: 'O(log n)' },
        { id: 'c', text: 'O(n log n)' },
        { id: 'd', text: 'O(√n)' },
      ],
      answer: 'b',
      explain:
        '每步把搜索区间砍半，最多 log n 步。题面也明确要求 O(log n)。',
      tags: ['complexity'],
    },
    {
      id: 'search-rotated-array.q11',
      prompt: '若题目改为"允许重复值"（LeetCode 81），核心难点是？',
      options: [
        { id: 'a', text: '完全相同算法，无变化' },
        { id: 'b', text: '当 nums[left] == nums[mid] == nums[right] 时无法判断哪一半有序，必须 left += 1; right -= 1 退化为线性，最坏 O(n)' },
        { id: 'c', text: '需要换成线性扫' },
        { id: 'd', text: '需要排序' },
      ],
      answer: 'b',
      explain:
        '重复让 invariant"至少一半严格有序"失效——最经典的反例是 [1,1,1,...,1,2,1,1]。最坏退化为 O(n)，但最好仍 O(log n)。',
      tags: ['invariant', 'complexity'],
    },
    {
      id: 'search-rotated-array.q12',
      prompt: '为什么本题不直接"先二分找旋转点 k，再在两段内分别二分 target"？',
      options: [
        { id: 'a', text: '不能这样做' },
        { id: 'b', text: '可以，且也是 O(log n)；只是单趟二分一次完成更精简' },
        { id: 'c', text: '会出错' },
        { id: 'd', text: '会超时' },
      ],
      answer: 'b',
      explain:
        '"先找 k 再二分"思路完全正确，复杂度同样 O(log n)，只是写法分两段。一次到位的版本（本题解）只需要一趟二分，代码更短，是更受青睐的写法。',
      tags: ['data-structure'],
    },
  ],
}

export default problem
