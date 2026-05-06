import type { Problem } from '../types'

const code = `def findPeakElement(nums: list[int]) -> int:
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            # 下降段，峰在 [left, mid]
            right = mid
        else:
            # 上升段，峰在 [mid + 1, right]
            left = mid + 1
    return left`

export const problem: Problem = {
  id: 'find-peak-element',
  leetcodeNo: 162,
  title: { zh: '寻找峰值', en: 'Find Peak Element' },
  difficulty: 'medium',
  pattern: 'binary-search',
  tags: ['array', 'binary-search'],
  statement:
    '峰值元素是指其值**严格大于左右相邻值**的元素。给你一个整数数组 `nums`，找到峰值元素并返回其索引。数组可能包含**多个峰值**，在这种情况下，返回**任何一个**峰值所在位置即可。\n\n你可以假设 `nums[-1] = nums[n] = -∞`，也就是说边界两侧默认是负无穷。\n\n你必须实现时间复杂度为 `O(log n)` 的算法。',
  examples: [
    { input: 'nums = [1, 2, 3, 1]', output: '2', note: 'nums[2]=3 是峰值' },
    { input: 'nums = [1, 2, 1, 3, 5, 6, 4]', output: '5', note: '索引 1 和 5 都是峰值，返回任一个' },
    { input: 'nums = [1]', output: '0', note: '单元素也算峰值（左右是 -∞）' },
  ],
  constraints: [
    '1 ≤ nums.length ≤ 1000',
    '-2³¹ ≤ nums[i] ≤ 2³¹ - 1',
    '对所有有效的 i，nums[i] != nums[i + 1]',
  ],
  intuition:
    '比较 nums[mid] 与 nums[mid+1]：上升段的右侧一定存在峰（沿上升一直走必撞顶），下降段的左侧一定存在峰。每次砍掉一半，O(log n)。题目"相邻不等 + 边界 -∞"两个条件保证整段不可能完全单调，必有峰存在。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(log n)', space: 'O(1)' },
  pythonRefIds: [],
  microQuestions: [
    {
      id: 'find-peak-element.q1',
      prompt: '本题为什么能用二分而不是必须 O(n) 扫描？',
      options: [
        { id: 'a', text: '因为数组排序了' },
        { id: 'b', text: '"任何一个峰"+"边界看作 -∞"+"相邻不等"三件事让"上升段右侧必存在峰"成立，搜索可以单方向砍半' },
        { id: 'c', text: '题目允许返回 -1' },
        { id: 'd', text: '随机化' },
      ],
      answer: 'b',
      explain:
        '这是面试时最该解释清楚的一句话——把"二分"的必要前提"问题具有可砍半性质"讲透。',
      tags: ['invariant'],
    },
    {
      id: 'find-peak-element.q2',
      prompt: '比较 nums[mid] 与 nums[mid+1] 时，nums[mid] > nums[mid+1] 意味着？',
      codeContext: code,
      highlightLine: 5,
      options: [
        { id: 'a', text: 'mid 一定是峰' },
        { id: 'b', text: '当前在下降段，左侧（含 mid）必存在峰' },
        { id: 'c', text: 'mid+1 是峰' },
        { id: 'd', text: '无法判断' },
      ],
      answer: 'b',
      explain:
        '从 mid 开始向左：如果一直下降，左边界 -∞，所以 nums[0] 就是峰；否则中途某处会上升后再下降，那个转折就是峰。所以左半（含 mid）必有峰。',
      tags: ['invariant'],
    },
    {
      id: 'find-peak-element.q3',
      prompt: 'nums[mid] < nums[mid+1] 时应该把 left 设为？',
      codeContext: code,
      highlightLine: 9,
      options: [
        { id: 'a', text: 'left = mid' },
        { id: 'b', text: 'left = mid + 1' },
        { id: 'c', text: 'left = mid - 1' },
        { id: 'd', text: 'left = right' },
      ],
      answer: 'b',
      explain:
        '上升段说明 mid 不是峰（mid+1 比它大），可以排除 mid。`left = mid + 1` 也保证区间真的缩小（避免死循环）。',
      tags: ['boundary'],
    },
    {
      id: 'find-peak-element.q4',
      prompt: '当 nums[mid] > nums[mid+1] 时为什么用 `right = mid` 而非 `right = mid - 1`？',
      codeContext: code,
      highlightLine: 6,
      options: [
        { id: 'a', text: '随便写' },
        { id: 'b', text: '此时 mid 本身可能就是峰（左邻不知道大小但已知 > 右邻），所以不能排除' },
        { id: 'c', text: '为了避免越界' },
        { id: 'd', text: 'Python 习惯' },
      ],
      answer: 'b',
      explain:
        '这是本题的关键细节：`right = mid - 1` 会丢解（恰好 mid 是峰时）。所以右收缩用 `mid`、左推进用 `mid + 1`，组合刚好让区间严格缩小。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'find-peak-element.q5',
      prompt: '循环条件应该用 `<` 还是 `<=`？',
      codeContext: code,
      highlightLine: 3,
      options: [
        { id: 'a', text: '`<=` ——闭区间标配' },
        { id: 'b', text: '`<` ——配合 `right = mid` 的写法，否则会死循环' },
        { id: 'c', text: '都行' },
        { id: 'd', text: '取决于数组长度' },
      ],
      answer: 'b',
      explain:
        '`right = mid` 不严格收缩；如果 `<=` 且 left == right == mid，会反复回到自己，死循环。改用 `<` 让 left == right 时退出，返回 left（也等于 right）即为答案。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'find-peak-element.q6',
      prompt: '循环退出时应该返回什么？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: 'left' },
        { id: 'b', text: 'right' },
        { id: 'c', text: 'mid' },
        { id: 'd', text: '都行（left == right）' },
      ],
      answer: 'd',
      explain:
        '退出时 left == right，区间收敛到唯一索引，返回任一即可。习惯写 `return left`。',
      tags: ['boundary'],
    },
    {
      id: 'find-peak-element.q7',
      prompt: '访问 `nums[mid + 1]` 是否会越界？',
      options: [
        { id: 'a', text: '会，需要特判' },
        { id: 'b', text: '不会——循环条件 left < right 保证 mid < right，所以 mid + 1 ≤ right ≤ n-1' },
        { id: 'c', text: '取决于数组长度' },
        { id: 'd', text: 'Python 自动处理负索引' },
      ],
      answer: 'b',
      explain:
        '`mid = (left + right) // 2` 在 left < right 时严格 mid < right，所以 mid + 1 一定是合法下标。这是循环条件用 `<` 而非 `<=` 的额外好处之一。',
      tags: ['boundary'],
    },
    {
      id: 'find-peak-element.q8',
      prompt: '若数组整体单调递增（如 [1,2,3,4,5]），算法返回什么？',
      options: [
        { id: 'a', text: '0' },
        { id: 'b', text: 'len(nums) - 1（即最后一个元素，因为右边界 -∞ 让它成为峰）' },
        { id: 'c', text: '-1' },
        { id: 'd', text: '抛异常' },
      ],
      answer: 'b',
      explain:
        '一直走"上升段"分支，left 不断右推，最终收敛到 n-1。这正是题面"边界 -∞"假设的妙用：让单调情形也有合法峰。',
      tags: ['boundary'],
    },
    {
      id: 'find-peak-element.q9',
      prompt: '题目"返回任意峰"的设计对算法选择的影响？',
      options: [
        { id: 'a', text: '没影响' },
        { id: 'b', text: '正因为不要求"全部"或"特定"峰，二分才能砍掉一半（否则可能错过另一半的峰）' },
        { id: 'c', text: '允许我们随机' },
        { id: 'd', text: '允许返回 -1' },
      ],
      answer: 'b',
      explain:
        '这是题目设计与算法的耦合：如果要返回"所有峰"或"最大峰"，O(log n) 不够，必须 O(n)。',
      tags: ['invariant'],
    },
    {
      id: 'find-peak-element.q10',
      prompt: '时间复杂度是？',
      options: [
        { id: 'a', text: 'O(n)' },
        { id: 'b', text: 'O(log n)' },
        { id: 'c', text: 'O(n log n)' },
        { id: 'd', text: 'O(1)' },
      ],
      answer: 'b',
      explain:
        '每步把区间砍半，至多 log n 步。题面要求即 O(log n)。',
      tags: ['complexity'],
    },
    {
      id: 'find-peak-element.q11',
      prompt: '空间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(log n)' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(n²)' },
      ],
      answer: 'a',
      explain:
        '只用三个标量索引，不用递归不用额外数据结构。',
      tags: ['complexity'],
    },
    {
      id: 'find-peak-element.q12',
      prompt: '把"`nums[mid] > nums[mid+1]`"换成"`nums[mid] > nums[mid-1]`"是否同样可行？',
      options: [
        { id: 'a', text: '可以，但需要相应处理 mid=0 的越界，整体复杂度不变' },
        { id: 'b', text: '不行，会出错' },
        { id: 'c', text: '会变成 O(n)' },
        { id: 'd', text: '完全等价无差' },
      ],
      answer: 'a',
      explain:
        '"看右邻"和"看左邻"在对称意义上都对，关键还是"判出在上升段还是下降段、相应保留可能含峰那一侧"。看右邻配 right=mid 简洁；看左邻则要配 left=mid 并处理 mid=0 边界，写起来更绕。',
      tags: ['invariant', 'pythonism'],
    },
  ],
}

export default problem
