import type { Problem } from '../types'

const code = `import heapq

def findKthLargest(nums: list[int], k: int) -> int:
    # 维护一个大小为 k 的最小堆——堆顶就是「目前为止第 k 大」
    heap = []
    for x in nums:
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:
            heapq.heappushpop(heap, x)
    return heap[0]`

export const problem: Problem = {
  id: 'kth-largest-element',
  leetcodeNo: 215,
  title: { zh: '数组中的第 K 个最大元素', en: 'Kth Largest Element in an Array' },
  difficulty: 'medium',
  pattern: 'heap',
  tags: ['array', 'heap', 'top-k', 'quickselect'],
  statement:
    '给定整数数组 `nums` 和整数 `k`，请返回数组中**第 `k` 个最大**的元素。\n\n请注意，你需要找的是数组**排序后**的第 `k` 个最大的元素，而不是第 `k` 个不同的元素。\n\n你必须设计并实现时间复杂度为 `O(n)` 的算法解决此问题。',
  examples: [
    { input: 'nums = [3,2,1,5,6,4], k = 2', output: '5' },
    { input: 'nums = [3,2,3,1,2,4,5,5,6], k = 4', output: '4', note: '第 4 大允许重复值各占一名' },
    { input: 'nums = [1], k = 1', output: '1' },
  ],
  constraints: [
    '1 ≤ k ≤ nums.length ≤ 10⁵',
    '-10⁴ ≤ nums[i] ≤ 10⁴',
  ],
  intuition:
    '维护「大小为 k 的最小堆」：堆里始终保存「迄今为止最大的 k 个元素」，堆顶就是它们中的最小——也就是「第 k 大」。新元素若大于堆顶，就 heappushpop 把堆顶挤掉。时间 O(n log k)，比排序 O(n log n) 优。要严格 O(n) 需用 quickselect。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n log k)', space: 'O(k)' },
  microQuestions: [
    {
      id: 'kth-largest-element.q1',
      prompt: '要找「第 k 大」，应该用大根堆还是小根堆？',
      options: [
        { id: 'a', text: '大根堆，size = n' },
        { id: 'b', text: '大根堆，size = k' },
        { id: 'c', text: '小根堆，size = k' },
        { id: 'd', text: '小根堆，size = n - k' },
      ],
      answer: 'c',
      explain:
        '反直觉但正确：用「大小为 k 的最小堆」。堆里维护「最大的 k 个元素」，堆顶=这 k 个中的最小=第 k 大。这样空间 O(k) 而非 O(n)，且新元素只需与堆顶比较。如果用大根堆 size=n 则要 pop k-1 次才能拿到答案，浪费空间。',
      tags: ['data-structure', 'invariant'],
    },
    {
      id: 'kth-largest-element.q2',
      prompt: 'Python `heapq` 默认是？',
      options: [
        { id: 'a', text: '最大堆，pop 出最大值' },
        { id: 'b', text: '最小堆，pop 出最小值' },
        { id: 'c', text: '需要参数指定' },
        { id: 'd', text: '随机' },
      ],
      answer: 'b',
      explain:
        '`heapq` 是最小堆。要做最大堆有两条路：① 把值取负后入堆；② 用 `heapq._heapify_max` 这种私有 API（不推荐）。本题正好需要最小堆，无需取负。',
      tags: ['pythonism', 'syntax'],
    },
    {
      id: 'kth-largest-element.q3',
      prompt: '为什么用 `heappushpop(heap, x)` 而不是先 push 再 pop？',
      codeContext: code,
      highlightLine: 9,
      options: [
        { id: 'a', text: '没区别，都是 O(log k)' },
        { id: 'b', text: '`heappushpop` 是单次 sift 操作，常数更小且语义清晰' },
        { id: 'c', text: '先 push 再 pop 会出错' },
        { id: 'd', text: '`heappushpop` 不是 heapq 的 API' },
      ],
      answer: 'b',
      explain:
        '`heappushpop(h, x)` 比 `heappush(h, x); heappop(h)` 少一次 sift——若 x 比堆顶小，直接返回 x（堆不变）；否则替换堆顶后 sift_down 一次。语义也更贴切：「比较并替换」就是 push-pop 的本意。',
      tags: ['pythonism', 'complexity'],
    },
    {
      id: 'kth-largest-element.q4',
      prompt: '为什么进入 elif 分支前要先判断 `x > heap[0]`？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: '可以省略，没影响' },
        { id: 'b', text: '剪枝：x 比堆顶小直接跳过，省一次 log k 的 sift' },
        { id: 'c', text: '为了防止越界' },
        { id: 'd', text: '保证堆是稳定的' },
      ],
      answer: 'b',
      explain:
        '若 x ≤ heap[0]，新元素「肯定挤不进 top-k」——直接 continue 即可，省掉 heappushpop 的 O(log k)。这种「先看堆顶再决定动不动堆」是 top-k 模板的标准优化。',
      tags: ['complexity', 'invariant'],
    },
    {
      id: 'kth-largest-element.q5',
      prompt: '初始填充阶段（heap 还没满）应该怎么处理？',
      codeContext: code,
      highlightLine: 6,
      options: [
        { id: 'a', text: '先空跑 k 步，把前 k 个直接复制' },
        { id: 'b', text: '逐个 heappush，自然形成堆' },
        { id: 'c', text: '先 heapify(nums[:k])' },
        { id: 'd', text: 'b 或 c 都行' },
      ],
      answer: 'd',
      explain:
        '两种写法都对：本题选了 b（统一进 for 循环更简洁）；c 的好处是 heapify 是 O(k) 而非 O(k log k)，但只在 k 大时才有可见差异。面试随便选一个能说清就行。',
      tags: ['pythonism', 'complexity'],
    },
    {
      id: 'kth-largest-element.q6',
      prompt: '循环结束后，第 k 大是堆里的哪个元素？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: 'heap[-1]，最后入堆的元素' },
        { id: 'b', text: 'heap[0]，最小堆的堆顶' },
        { id: 'c', text: 'max(heap)' },
        { id: 'd', text: 'heap[k-1]' },
      ],
      answer: 'b',
      explain:
        '堆里恰好是「最大的 k 个元素」，最小堆的堆顶就是它们中的最小——也就是第 k 大。注意 heap[k-1] 在堆数组里没有「k 大」的语义，因为堆数组不是排序数组。',
      tags: ['invariant', 'data-structure'],
    },
    {
      id: 'kth-largest-element.q7',
      prompt: '本解法的时间复杂度是？',
      options: [
        { id: 'a', text: 'O(n)' },
        { id: 'b', text: 'O(n log k)' },
        { id: 'c', text: 'O(n log n)' },
        { id: 'd', text: 'O(k log n)' },
      ],
      answer: 'b',
      explain:
        'n 个元素每个最多触发一次 heappushpop，每次 O(log k)。整体 O(n log k)，当 k « n 时显著优于排序的 O(n log n)。',
      tags: ['complexity'],
    },
    {
      id: 'kth-largest-element.q8',
      prompt: '空间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(k)' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(log n)' },
      ],
      answer: 'b',
      explain:
        '堆始终保持大小为 k，所以是 O(k)。用 `heapq.nlargest(k, nums)[-1]` 等价、空间也 O(k)。',
      tags: ['complexity'],
    },
    {
      id: 'kth-largest-element.q9',
      prompt: '题目要求 O(n)。能用堆达到 O(n) 吗？该用什么算法？',
      options: [
        { id: 'a', text: '能，调小 k 就行' },
        { id: 'b', text: '不能；O(n) 平均得用 quickselect（基于 partition 的快速选择）' },
        { id: 'c', text: '不能；只有 O(n log n) 是最优' },
        { id: 'd', text: '能，用 heapify(nums) 一次 O(n)' },
      ],
      answer: 'b',
      explain:
        'quickselect 是基于快排 partition 的「找第 k 小/大」算法，平均 O(n)、最坏 O(n²)（用三向切分或随机 pivot 可减小最坏概率）。堆解法严格 O(n log k)，常数小、实现简单，面试时通常先讲堆再讲 quickselect。',
      tags: ['complexity', 'data-structure'],
    },
    {
      id: 'kth-largest-element.q10',
      prompt: '若直接 `sorted(nums, reverse=True)[k-1]`，复杂度是？',
      options: [
        { id: 'a', text: 'O(k)' },
        { id: 'b', text: 'O(n log k)' },
        { id: 'c', text: 'O(n log n)' },
        { id: 'd', text: 'O(n²)' },
      ],
      answer: 'c',
      explain:
        '排序整个数组是 O(n log n)。能 AC，但浪费——你只关心前 k 个，剩下 n-k 个的相对顺序对答案没用。堆 / quickselect 都是利用「不需要全排序」的优化。',
      tags: ['complexity'],
    },
    {
      id: 'kth-largest-element.q11',
      prompt: '`heapq.nlargest(k, nums)` 与本写法的关系？',
      options: [
        { id: 'a', text: '完全等价，都是 O(n log k)' },
        { id: 'b', text: '`nlargest` 实际就是用大小为 k 的最小堆实现的——本写法是它的「展开版」' },
        { id: 'c', text: '`nlargest` 更慢，是 O(n log n)' },
        { id: 'd', text: '都对，b 更准确' },
      ],
      answer: 'd',
      explain:
        '`heapq.nlargest` 的源码就是这个算法。日常一行 `heapq.nlargest(k, nums)[-1]` 就够了；面试时手写循环展示理解。两者复杂度都是 O(n log k)。',
      tags: ['pythonism', 'complexity'],
    },
    {
      id: 'kth-largest-element.q12',
      prompt: '若题目改成「找第 k 小」，最少要改几处？',
      options: [
        { id: 'a', text: '把 nums 取负' },
        { id: 'b', text: '把比较换成 `x < heap[0]` 并把堆改成大根堆（值取负实现）' },
        { id: 'c', text: '把 `>` 换成 `<`，堆继续用最小堆' },
        { id: 'd', text: '用 sorted 排序后取 nums[k-1]' },
      ],
      answer: 'b',
      explain:
        '对称翻转：第 k 小用「大小为 k 的最大堆」，堆顶是「迄今最小 k 个里的最大」=第 k 小。Python 没有原生大根堆——常见做法是把值取负后照样用 heapq。或者直接 `heapq.nsmallest(k, nums)[-1]`。',
      tags: ['data-structure', 'invariant'],
    },
  ],
}

export default problem
