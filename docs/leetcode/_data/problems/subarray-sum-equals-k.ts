import type { Problem } from '../types'

const code = `def subarraySum(nums: list[int], k: int) -> int:
    cnt = {0: 1}        # 前缀和 0 「出现过 1 次」——处理「整段和恰好 = k」的边界
    prefix = 0
    ans = 0
    for x in nums:
        prefix += x
        ans += cnt.get(prefix - k, 0)   # 之前出现过几次 prefix - k，就有几个以当前结尾的子数组和为 k
        cnt[prefix] = cnt.get(prefix, 0) + 1
    return ans`

export const problem: Problem = {
  id: 'subarray-sum-equals-k',
  leetcodeNo: 560,
  title: { zh: '和为 K 的子数组', en: 'Subarray Sum Equals K' },
  difficulty: 'medium',
  pattern: 'bit-prefix',
  tags: ['array', 'prefix-sum', 'hash-table'],
  statement:
    '给你一个整数数组 `nums` 和一个整数 `k`，请你统计并返回该数组中**和为 `k` 的连续子数组**的**个数**。\n\n子数组是数组中元素的连续非空序列。',
  examples: [
    { input: 'nums = [1,1,1], k = 2', output: '2', note: '[1,1] 出现两次（位置 0-1 和 1-2）' },
    { input: 'nums = [1,2,3], k = 3', output: '2', note: '[1,2] 与 [3]' },
    { input: 'nums = [1,-1,0], k = 0', output: '3', note: '[1,-1]、[-1,0...]、[0]' },
  ],
  constraints: [
    '1 ≤ nums.length ≤ 2 × 10⁴',
    '-1000 ≤ nums[i] ≤ 1000',
    '-10⁷ ≤ k ≤ 10⁷',
  ],
  intuition:
    '前缀和 + 哈希表的招牌组合。子数组 `(i,j]` 的和 = `prefix[j] - prefix[i]`；要找「和为 k」就找「prefix[j] - prefix[i] = k」即 prefix[i] = prefix[j] - k。一边累计 prefix、一边查哈希表中 `prefix - k` 出现过几次（直接累加进答案），再把当前 prefix 写入哈希表。`cnt = {0: 1}` 是处理「从 nums[0] 开始整段和正好等于 k」的边界初始化。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(n)' },
  microQuestions: [
    {
      id: 'subarray-sum-equals-k.q1',
      prompt: '本题为什么用前缀和而不是滑动窗口？',
      options: [
        { id: 'a', text: '因为 nums 可能含负数 —— 滑窗的「单调收缩」前提（添加正数和增、删除正数和减）不再成立' },
        { id: 'b', text: '前缀和更短' },
        { id: 'c', text: '滑窗在 Python 里慢' },
        { id: 'd', text: '滑窗只能用于字符串' },
      ],
      answer: 'a',
      explain:
        '滑窗的「窗口扩张和单调增 / 收缩和单调减」依赖正数。本题允许负数和 0，必须用前缀和 + 哈希。这是判断「滑窗 vs 前缀和」的核心信号。',
      tags: ['invariant'],
    },
    {
      id: 'subarray-sum-equals-k.q2',
      prompt: '子数组 `(i, j]` 的和等于 `prefix[j] - prefix[i]`——这里 (i, j] 是闭还是开？',
      options: [
        { id: 'a', text: '左开右闭——i 是「之前」位置，j 是当前位置' },
        { id: 'b', text: '闭区间' },
        { id: 'c', text: '左闭右开' },
        { id: 'd', text: '开区间' },
      ],
      answer: 'a',
      explain:
        '约定 prefix[k] = nums[0] + ... + nums[k-1]（前 k 个的和），则 `nums[i..j-1]` 的和 = prefix[j] - prefix[i]。本题里我们就是按这种语义查 `prefix - k`。',
      tags: ['invariant', 'boundary'],
    },
    {
      id: 'subarray-sum-equals-k.q3',
      prompt: '哈希表里 key 与 value 的语义是？',
      codeContext: code,
      highlightLine: 2,
      options: [
        { id: 'a', text: 'key = 某个前缀和，value = 这个前缀和**到目前为止出现过的次数**' },
        { id: 'b', text: 'key = 索引，value = 前缀和' },
        { id: 'c', text: 'key = 子数组，value = 和' },
        { id: 'd', text: 'key = nums[i]，value = i' },
      ],
      answer: 'a',
      explain:
        '同一个前缀和可能出现多次（含负数时尤其常见）——所以 value 要存「次数」而不是「最近一次的位置」。两个不同的 i₁ < i₂ 都让 prefix[i₁] = prefix[i₂] = p 的话，j 找到 p+k 时贡献是 2。',
      tags: ['data-structure', 'invariant'],
    },
    {
      id: 'subarray-sum-equals-k.q4',
      prompt: '`cnt = {0: 1}` 这个初始化的关键作用是？',
      codeContext: code,
      highlightLine: 2,
      options: [
        { id: 'a', text: '处理「从 nums[0] 开始的整段和正好等于 k」的边界——此时 prefix-k = 0 必须能命中一次' },
        { id: 'b', text: '随便加的' },
        { id: 'c', text: '为了让代码更整洁' },
        { id: 'd', text: '防止哈希冲突' },
      ],
      answer: 'a',
      explain:
        '想象 nums=[3], k=3：prefix=3, prefix-k=0；如果不初始化 {0:1} 就漏掉这个解。0 代表「空前缀」——它确实出现过 1 次。',
      tags: ['boundary'],
    },
    {
      id: 'subarray-sum-equals-k.q5',
      prompt: '`ans += cnt.get(prefix - k, 0)` 与 `cnt[prefix] = cnt.get(prefix, 0) + 1`，哪个先做？',
      codeContext: code,
      highlightLine: 7,
      options: [
        { id: 'a', text: '先 ans += 再写 cnt——避免「自己」也被算成配对（k=0 时尤其关键）' },
        { id: 'b', text: '先写 cnt 再 ans' },
        { id: 'c', text: '顺序无所谓' },
        { id: 'd', text: '一行写完' },
      ],
      answer: 'a',
      explain:
        '与 LC1（two-sum）的「先查后存」一脉相承——先查 cnt 里的旧记录，再把当前 prefix 加进去。否则 k=0 时每个 prefix 都会和自己配对、答案膨胀一倍。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'subarray-sum-equals-k.q6',
      prompt: '`cnt.get(prefix - k, 0)` 这个 `0` 默认值的意义？',
      codeContext: code,
      highlightLine: 7,
      options: [
        { id: 'a', text: 'key 不存在时返回 0，即「此前没出现过这个前缀，贡献 0 个子数组」' },
        { id: 'b', text: '魔法值' },
        { id: 'c', text: '可以省，会自动返回 None' },
        { id: 'd', text: '为了让结果非负' },
      ],
      answer: 'a',
      explain:
        '`dict.get(k, default)` 是处理「key 不存在」的优雅写法，避免 KeyError 也免去显式 if 检查。Pythonic 标配。',
      tags: ['pythonism'],
    },
    {
      id: 'subarray-sum-equals-k.q7',
      prompt: '为什么暴力 O(n²) 超时但本题 O(n) 能过？',
      options: [
        { id: 'a', text: 'n=2×10⁴ 时 O(n²)=4×10⁸ 接近超时上限；O(n) 才稳' },
        { id: 'b', text: '常数因素' },
        { id: 'c', text: '内存问题' },
        { id: 'd', text: 'O(n²) 也能过' },
      ],
      answer: 'a',
      explain:
        '约束 n ≤ 2×10⁴ 暗示了大概率不能 O(n²)。哈希 + 前缀和把每个 j 的 O(n) 找 i 操作压成 O(1)，整体降到 O(n)。',
      tags: ['complexity'],
    },
    {
      id: 'subarray-sum-equals-k.q8',
      prompt: '本题与 two-sum 在思维上的最大共性是？',
      options: [
        { id: 'a', text: '都是「找配对」——两数之和找 target-x；前缀和找 prefix-k；都用 dict O(1) 查询' },
        { id: 'b', text: '没有共性' },
        { id: 'c', text: '都用 deque' },
        { id: 'd', text: '都用 sort' },
      ],
      answer: 'a',
      explain:
        '掌握这一招，整个 hashmap pattern 都能横扫：把「枚举两端 O(n²)」转化为「枚举一端 + dict 查另一端 O(n)」。',
      tags: ['data-structure'],
    },
    {
      id: 'subarray-sum-equals-k.q9',
      prompt: '若 nums 全为非负数，能否改用滑动窗口？',
      options: [
        { id: 'a', text: '可以——非负数时窗口和单调，扩张就增、收缩就减，是滑窗的前提' },
        { id: 'b', text: '不能' },
        { id: 'c', text: '滑窗永远不能解此类题' },
        { id: 'd', text: '滑窗只能解字符串' },
      ],
      answer: 'a',
      explain:
        '本题的难度其实就来自「允许负数」。若约束变为全非负，滑窗 O(n) 与前缀和 O(n) 等价，常数还更小。这是面试常见追问。',
      tags: ['invariant'],
    },
    {
      id: 'subarray-sum-equals-k.q10',
      prompt: '空间复杂度?',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(n) — 哈希表最坏装 n+1 个不同前缀和' },
        { id: 'c', text: 'O(log n)' },
        { id: 'd', text: 'O(n²)' },
      ],
      answer: 'b',
      explain:
        '没有重复前缀和时哈希表会装下 n+1 个键。这是「空间换时间」的代价，符合 LC560 的标准约束。',
      tags: ['complexity'],
    },
    {
      id: 'subarray-sum-equals-k.q11',
      prompt: '若题目改为「找最长和为 k 的子数组」（LC325），需要怎么改？',
      options: [
        { id: 'a', text: '把 cnt 改成「key=prefix，value=最早出现的索引」；遇到 prefix-k 时用 j - cnt[prefix-k] 更新最长长度；只在 prefix 第一次出现时记录' },
        { id: 'b', text: '完全换算法' },
        { id: 'c', text: '不可能解' },
        { id: 'd', text: '加 sort' },
      ],
      answer: 'a',
      explain:
        '「计数」与「最长长度」的差别就是 dict value 存什么——次数 or 最早索引。二者都是前缀和 + 哈希的同一套骨架。',
      tags: ['data-structure'],
    },
    {
      id: 'subarray-sum-equals-k.q12',
      prompt: '`prefix += x; ans += ...; cnt[prefix] = ...` 这三步顺序写错的常见 bug？',
      options: [
        { id: 'a', text: '把 cnt 写入放在 ans 累加之前——k=0 时把当前 prefix 自己也算成配对，答案翻倍' },
        { id: 'b', text: '把 prefix += x 放在循环外' },
        { id: 'c', text: '没有 bug' },
        { id: 'd', text: 'cnt 必须在 prefix 之前更新' },
      ],
      answer: 'a',
      explain:
        '「先查后存」的纪律和 two-sum 一脉相承。k=0 时这种 bug 最容易暴露——要把这个 case 当成单元测试用例。',
      tags: ['boundary'],
    },
  ],
}

export default problem
