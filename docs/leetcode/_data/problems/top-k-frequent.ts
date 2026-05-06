import type { Problem } from '../types'

const code = `import heapq
from collections import Counter

def topKFrequent(nums: list[int], k: int) -> list[int]:
    count = Counter(nums)  # {值: 出现次数}
    # 用「大小为 k 的最小堆」按频次维护——堆顶=当前最不热门的
    heap = []  # 元素形如 (freq, num)
    for num, freq in count.items():
        if len(heap) < k:
            heapq.heappush(heap, (freq, num))
        elif freq > heap[0][0]:
            heapq.heappushpop(heap, (freq, num))
    return [num for freq, num in heap]`

export const problem: Problem = {
  id: 'top-k-frequent',
  leetcodeNo: 347,
  title: { zh: '前 K 个高频元素', en: 'Top K Frequent Elements' },
  difficulty: 'medium',
  pattern: 'heap',
  tags: ['array', 'heap', 'hash-table', 'top-k', 'bucket-sort'],
  statement:
    '给你一个整数数组 `nums` 和一个整数 `k`，请你返回其中**出现频率前 `k` 高**的元素。\n\n你可以按**任意顺序**返回答案。',
  examples: [
    { input: 'nums = [1,1,1,2,2,3], k = 2', output: '[1,2]', note: '1 出现 3 次，2 出现 2 次' },
    { input: 'nums = [1], k = 1', output: '[1]' },
    { input: 'nums = [4,1,-1,2,-1,2,3], k = 2', output: '[-1, 2]', note: '负数也合法' },
  ],
  constraints: [
    '1 ≤ nums.length ≤ 10⁵',
    '-10⁴ ≤ nums[i] ≤ 10⁴',
    'k 在合法范围内（结果是唯一的）',
  ],
  intuition:
    '两步：① `Counter` 统计每个值的频次；② 用大小为 k 的最小堆按频次维护「迄今最热门的 k 个」——堆顶是它们里频次最低的。新元素若频次大于堆顶就 heappushpop。返回时把堆里的 num 提取出来即可（顺序不重要）。时间 O(n log k)。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n log k)', space: 'O(n)' },
  microQuestions: [
    {
      id: 'top-k-frequent.q1',
      prompt: '第一步统计频次，应该用什么数据结构？',
      codeContext: code,
      highlightLine: 5,
      options: [
        { id: 'a', text: 'list，append 时手动计数' },
        { id: 'b', text: 'set，存出现过的值' },
        { id: 'c', text: 'dict 或 collections.Counter——key=值, value=频次' },
        { id: 'd', text: '用 sorted(nums) 后扫一遍' },
      ],
      answer: 'c',
      explain:
        'Counter 是 dict 的子类，专为「统计可哈希对象的出现次数」设计：`Counter(nums)` 一行搞定。比手写 `d[x] = d.get(x, 0) + 1` 简洁，且自带 `most_common(k)` 等方法。',
      tags: ['data-structure', 'pythonism'],
    },
    {
      id: 'top-k-frequent.q2',
      prompt: '`Counter(nums).most_common(k)` 一行就能得到答案吗？',
      options: [
        { id: 'a', text: '完全可以，且速度最快' },
        { id: 'b', text: '可以但内部是排序，复杂度 O(n log n)；用堆能优到 O(n log k)' },
        { id: 'c', text: '不行，most_common 不能限制 k' },
        { id: 'd', text: '不行，most_common 返回顺序错' },
      ],
      answer: 'b',
      explain:
        '`most_common(k)` 内部用的是 `heapq.nlargest(k, ...)`——也就是 O(n log k)。所以面试时一行 `Counter(nums).most_common(k)` 是能 AC 且常被接受的；手写循环只是为了展示理解。注意旧文档里有说 most_common 用排序，那是 Python 3.0 时的事。',
      tags: ['pythonism', 'complexity'],
    },
    {
      id: 'top-k-frequent.q3',
      prompt: '维护 top-k 高频，应该用大根堆还是小根堆？',
      options: [
        { id: 'a', text: '大根堆，size = unique 值的总数' },
        { id: 'b', text: '大根堆，size = k' },
        { id: 'c', text: '小根堆，size = k' },
        { id: 'd', text: '小根堆，size = unique 值的总数' },
      ],
      answer: 'c',
      explain:
        '套用通用 top-k 模板：「找前 k 大」用「大小为 k 的最小堆」。堆里维护「迄今频次最高的 k 个值」，堆顶是它们里频次最低的——一旦新元素频次更高就替换堆顶。',
      tags: ['data-structure', 'invariant'],
    },
    {
      id: 'top-k-frequent.q4',
      prompt: '堆里每个元素应当存什么？',
      codeContext: code,
      highlightLine: 7,
      options: [
        { id: 'a', text: '只存 num' },
        { id: 'b', text: '只存 freq' },
        { id: 'c', text: '元组 (freq, num)' },
        { id: 'd', text: '元组 (num, freq)' },
      ],
      answer: 'c',
      explain:
        '需要「按频次比较」并「最终拿到 num」。Python 比较元组时按字典序，第一项定胜负——所以频次必须放第一位，否则会按 num 大小比较。',
      tags: ['data-structure', 'invariant', 'pythonism'],
    },
    {
      id: 'top-k-frequent.q5',
      prompt: '若两个元素频次相同，元组 (freq, num) 怎么比？',
      options: [
        { id: 'a', text: '会报错——相同 key 不能比' },
        { id: 'b', text: '按 num 比较——freq 相同时 num 较小的视作较小' },
        { id: 'c', text: 'Python 会随机选' },
        { id: 'd', text: '需要自定义 __lt__' },
      ],
      answer: 'b',
      explain:
        '元组按字典序比较，第一项相同就比第二项。所以频次相同时 num 较小的元素「更小」会先被弹出——这在题目允许任意顺序时无所谓，但若题目规定「频次相同按 num 升序」就要小心。如果 num 是不可比对象（如自定义类），需要写 `(freq, idx, obj)` 用唯一索引兜底。',
      tags: ['pythonism', 'invariant'],
    },
    {
      id: 'top-k-frequent.q6',
      prompt: '`heap[0][0]` 取的是？',
      codeContext: code,
      highlightLine: 9,
      options: [
        { id: 'a', text: '堆顶元素的 num' },
        { id: 'b', text: '堆顶元素的 freq' },
        { id: 'c', text: '堆里最大的频次' },
        { id: 'd', text: '堆的长度' },
      ],
      answer: 'b',
      explain:
        'heap[0] 是堆顶 (freq, num) 元组，[0] 取第一项即频次。这是「先看堆顶再决定动不动堆」的剪枝判断。',
      tags: ['syntax', 'pythonism'],
    },
    {
      id: 'top-k-frequent.q7',
      prompt: '最后返回结果时为什么用列表推导 `[num for freq, num in heap]`？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: '题目要求只返回值，不返回频次' },
        { id: 'b', text: '让结果按频次排序' },
        { id: 'c', text: '解构每个元组，丢弃 freq 只保留 num' },
        { id: 'd', text: 'a 和 c' },
      ],
      answer: 'd',
      explain:
        '题目只要值列表，所以丢弃频次。Python 元组解包 `for freq, num in heap` 同时拆出两项，比 `[t[1] for t in heap]` 可读性更好。题目还说「任意顺序」所以不需排序。',
      tags: ['pythonism', 'syntax'],
    },
    {
      id: 'top-k-frequent.q8',
      prompt: '本解法时间复杂度是？',
      options: [
        { id: 'a', text: 'O(n)' },
        { id: 'b', text: 'O(n log k)' },
        { id: 'c', text: 'O(n log n)' },
        { id: 'd', text: 'O(k log n)' },
      ],
      answer: 'b',
      explain:
        'Counter 统计 O(n)；遍历 unique 值（最多 n 个）每个 O(log k) 入堆，整体 O(n log k)。比排序解法 O(n log n) 优，当 k « n 时差距明显。',
      tags: ['complexity'],
    },
    {
      id: 'top-k-frequent.q9',
      prompt: '空间复杂度是？',
      options: [
        { id: 'a', text: 'O(k)' },
        { id: 'b', text: 'O(n)' },
        { id: 'c', text: 'O(log n)' },
        { id: 'd', text: 'O(1)' },
      ],
      answer: 'b',
      explain:
        '堆本身 O(k)，但 Counter 字典最多有 n 个 unique key——主导项是 O(n)。',
      tags: ['complexity'],
    },
    {
      id: 'top-k-frequent.q10',
      prompt: '能否用「桶排序」做到 O(n)？',
      options: [
        { id: 'a', text: '不能，下界就是 O(n log k)' },
        { id: 'b', text: '能：开 n+1 个桶（频次最大可能为 n），把每个 num 扔进对应频次桶，从高频桶倒序收集 k 个' },
        { id: 'c', text: '能但要 O(n²) 空间' },
        { id: 'd', text: '能但需要排序' },
      ],
      answer: 'b',
      explain:
        '桶排序解法：buckets[i] 存所有「频次为 i」的值。扫桶（i 从 n 到 1）收集前 k 个即可，O(n) 时间 + O(n) 空间。这是该题的另一个经典解法，面试时一并讲能加分。',
      tags: ['data-structure', 'complexity'],
    },
    {
      id: 'top-k-frequent.q11',
      prompt: '`heappush(heap, (freq, num))` 与 `heap.append((freq, num)); heapq.heapify(heap)` 的差别？',
      options: [
        { id: 'a', text: '完全等价' },
        { id: 'b', text: 'heappush 是 O(log n)；heapify 是 O(n)。多次 push 应该一个个 push，不要每次 heapify' },
        { id: 'c', text: 'heappush 不维护堆性质' },
        { id: 'd', text: 'heapify 是稳定排序' },
      ],
      answer: 'b',
      explain:
        '`heapify` 一次 O(n) 把任意 list 变成堆，适合「一次性建堆」；`heappush` 是 O(log n) 每次维护堆性质，适合增量插入。错误示范：循环里每次 append + heapify，会变成 O(n²) 的性能灾难。',
      tags: ['pythonism', 'complexity'],
    },
    {
      id: 'top-k-frequent.q12',
      prompt: '若 k = 数组的 unique 值数量，本算法会怎样？',
      options: [
        { id: 'a', text: '会报错' },
        { id: 'b', text: '退化成「全部返回」，复杂度 O(n log n)' },
        { id: 'c', text: '不动堆顶，性能不受影响' },
        { id: 'd', text: '会死循环' },
      ],
      answer: 'b',
      explain:
        'k 等于 unique 数时堆永远在「填充」分支，所有元素都会被 heappush。这等价于 O(n log n) 的堆排序场景。题目通常保证 k « n，所以一般无需特殊处理。',
      tags: ['boundary', 'complexity'],
    },
  ],
}

export default problem
