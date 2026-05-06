import type { Problem } from '../types'

const code = `import heapq

class MedianFinder:
    def __init__(self):
        # low: 大根堆（存较小的一半），用「负数」实现大根堆
        self.low = []
        # high: 小根堆（存较大的一半）
        self.high = []

    def addNum(self, num: int) -> None:
        # 默认让 low 比 high 多一个；偶数时两者等长
        if len(self.low) == len(self.high):
            # 先入 high 中转，再把 high 顶给 low——保证 low 的最大 ≤ high 的最小
            heapq.heappush(self.high, num)
            heapq.heappush(self.low, -heapq.heappop(self.high))
        else:
            heapq.heappush(self.low, -num)
            heapq.heappush(self.high, -heapq.heappop(self.low))

    def findMedian(self) -> float:
        if len(self.low) > len(self.high):
            return -self.low[0]  # 注意取负还原
        return (-self.low[0] + self.high[0]) / 2`

export const problem: Problem = {
  id: 'find-median-data-stream',
  leetcodeNo: 295,
  title: { zh: '数据流的中位数', en: 'Find Median from Data Stream' },
  difficulty: 'hard',
  pattern: 'heap',
  tags: ['heap', 'design', 'two-heap', 'data-stream'],
  statement:
    '中位数是有序整数列表中**最中间**的数。如果列表的长度是偶数，中位数则是中间两个数的平均值。\n\n请设计一个支持以下两种操作的数据结构：\n- `addNum(num)`：将整数 `num` 加入数据结构\n- `findMedian()`：返回当前所有元素的中位数\n\n要求 `addNum` 与 `findMedian` 的复杂度都尽可能低。',
  examples: [
    {
      input: 'addNum(1); addNum(2); findMedian(); addNum(3); findMedian();',
      output: '1.5, 2.0',
      note: '加入 1, 2 → 中位数 1.5；再加入 3 → [1,2,3] → 中位数 2',
    },
    {
      input: 'addNum(-1); addNum(-2); findMedian(); addNum(-3); findMedian(); addNum(-4); findMedian();',
      output: '-1.5, -2.0, -2.5',
    },
  ],
  constraints: [
    '-10⁵ ≤ num ≤ 10⁵',
    '至少在 findMedian 之前会调用一次 addNum',
    '至多调用 5·10⁴ 次',
  ],
  intuition:
    '「双堆对顶」：大根堆 low 装较小的一半，小根堆 high 装较大的一半。维持不变量 `len(low) - len(high) ∈ {0, 1}`。中位数永远在堆顶——奇数时取 low 顶，偶数时取两堆顶平均。Python 没有原生大根堆，把值取负塞进 heapq 即可。每次 addNum 是 O(log n)，findMedian 是 O(1)。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'addNum O(log n), findMedian O(1)', space: 'O(n)' },
  microQuestions: [
    {
      id: 'find-median-data-stream.q1',
      prompt: '为什么要用「双堆」而不是「一个有序列表 + 二分插入」？',
      options: [
        { id: 'a', text: '双堆代码更简短' },
        { id: 'b', text: '有序列表二分查找位置 O(log n)，但插入要平移 O(n)；双堆把 add 压到 O(log n)' },
        { id: 'c', text: '双堆的空间更小' },
        { id: 'd', text: '有序列表无法处理重复值' },
      ],
      answer: 'b',
      explain:
        'Python 的 list 即使二分定位 O(log n)，物理插入仍是 O(n)（平移元素）。双堆刚好把「插入 + 维持中位数」都压到 O(log n)。如果用 SortedList（sortedcontainers 库）能做到 O(log n) 插入，但面试一般默认只用标准库。',
      tags: ['data-structure', 'complexity'],
    },
    {
      id: 'find-median-data-stream.q2',
      prompt: '`low` 和 `high` 应该分别是？',
      codeContext: code,
      highlightLine: 6,
      options: [
        { id: 'a', text: 'low 小根堆装小一半；high 大根堆装大一半' },
        { id: 'b', text: 'low 大根堆装小一半；high 小根堆装大一半' },
        { id: 'c', text: '都是小根堆' },
        { id: 'd', text: '都是大根堆' },
      ],
      answer: 'b',
      explain:
        '中位数永远在「两堆交界」处。要 O(1) 拿到这个值，low 必须能 O(1) 给出「小一半的最大」（=大根堆顶），high 必须能 O(1) 给出「大一半的最小」（=小根堆顶）。这样中位数就是两堆顶之一或它们的平均。',
      tags: ['data-structure', 'invariant'],
    },
    {
      id: 'find-median-data-stream.q3',
      prompt: 'Python 的 `heapq` 是最小堆。要做大根堆怎么办？',
      codeContext: code,
      highlightLine: 18,
      options: [
        { id: 'a', text: 'heapq 有 max_heap 模式参数' },
        { id: 'b', text: '入堆时取负，出堆时再取负还原' },
        { id: 'c', text: '继承 heapq 重写比较函数' },
        { id: 'd', text: '只能自己手写' },
      ],
      answer: 'b',
      explain:
        '惯用技巧：`heappush(low, -x)` / `-low[0]`。负数让大小关系翻转，最小堆里取到的「最小负数」对应原来的「最大正数」。注意不要忘记还原。',
      tags: ['pythonism', 'syntax'],
    },
    {
      id: 'find-median-data-stream.q4',
      prompt: '本实现的不变量是？',
      options: [
        { id: 'a', text: 'len(low) == len(high)' },
        { id: 'b', text: 'len(low) - len(high) ∈ {0, 1}' },
        { id: 'c', text: 'len(low) - len(high) ∈ {-1, 0, 1}' },
        { id: 'd', text: 'low 与 high 元素无重叠' },
      ],
      answer: 'b',
      explain:
        '约定让 low 永远不少于 high（最多多一个）。这样总数为奇时中位数=low 顶；偶数时=两堆顶平均。也有题解约定相反——只要奇数时谁多就取谁的堆顶，无所谓。',
      tags: ['invariant'],
    },
    {
      id: 'find-median-data-stream.q5',
      prompt: '当 `len(low) == len(high)` 时来一个新数，为什么不能直接 push 进 low？',
      codeContext: code,
      highlightLine: 12,
      options: [
        { id: 'a', text: '可以，效果一样' },
        { id: 'b', text: 'num 可能比 high 顶还大，直接进 low 会破坏「low 全部 ≤ high 全部」的不变量' },
        { id: 'c', text: 'low 已满' },
        { id: 'd', text: '会越界' },
      ],
      answer: 'b',
      explain:
        '关键不变量：low 的所有元素 ≤ high 的所有元素。直接 push num 进 low 可能让大数混进小堆。正确做法：先 push 进 high「过滤」一下，high 顶（即 small among bigger half + new num 中的最小者）才是真正属于「小一半」的——再把它转入 low。',
      tags: ['invariant', 'boundary'],
    },
    {
      id: 'find-median-data-stream.q6',
      prompt: 'addNum 的「中转」做法（先入对面堆再把对面堆顶转过来）目的是？',
      options: [
        { id: 'a', text: '让代码更短' },
        { id: 'b', text: '保证两个堆「分界正确」：跨堆元素一定属于另一边' },
        { id: 'c', text: '让 heapq 工作正常' },
        { id: 'd', text: '避免 IndexError' },
      ],
      answer: 'b',
      explain:
        '中转的本质是「在合并后的有序集里找到正确位置」。新数 + high 的最小一起在 high 中比一次，最小那个属于 low。等价于「插入有序集 + 重新切分」，但只用了堆的 O(log n) 操作。',
      tags: ['invariant'],
    },
    {
      id: 'find-median-data-stream.q7',
      prompt: 'findMedian 在偶数总数下的返回值是？',
      codeContext: code,
      highlightLine: 24,
      options: [
        { id: 'a', text: 'low[0]（大根堆顶，记得取负）' },
        { id: 'b', text: 'high[0]（小根堆顶）' },
        { id: 'c', text: '(-low[0] + high[0]) / 2' },
        { id: 'd', text: '(low[0] + high[0]) / 2' },
      ],
      answer: 'c',
      explain:
        '偶数时两堆等长，中位数=两个中间数的均值。`-low[0]` 是 low 的真实最大（小一半中最大的），`high[0]` 是大一半中最小的——它们就是中间两个。注意一定要除以 2.0（Python 3 中 `/` 默认浮点除法）。',
      tags: ['invariant', 'syntax'],
    },
    {
      id: 'find-median-data-stream.q8',
      prompt: 'findMedian 在奇数总数下应该返回？',
      codeContext: code,
      highlightLine: 23,
      options: [
        { id: 'a', text: 'low[0]' },
        { id: 'b', text: '-low[0]' },
        { id: 'c', text: 'high[0]' },
        { id: 'd', text: '(-low[0] + high[0]) / 2' },
      ],
      answer: 'b',
      explain:
        '依不变量约定 low 多一个，所以奇数时中位数 = low 的最大 = -low[0]（取负还原）。如果约定相反则取 high[0]。',
      tags: ['invariant', 'syntax'],
    },
    {
      id: 'find-median-data-stream.q9',
      prompt: 'addNum 的时间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(log n)' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(n log n)' },
      ],
      answer: 'b',
      explain:
        '每次 addNum 做两次堆操作（push + pop + push），每次 O(log n)。整体 O(log n)。findMedian 只读两个堆顶，是 O(1)。',
      tags: ['complexity'],
    },
    {
      id: 'find-median-data-stream.q10',
      prompt: '空间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(log n)' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(n²)' },
      ],
      answer: 'c',
      explain:
        '所有插入过的数据都要保留——两堆共存 n 个元素，O(n)。',
      tags: ['complexity'],
    },
    {
      id: 'find-median-data-stream.q11',
      prompt: '若题目改为 `99% 的输入在 [0, 100]`，能怎么优化？',
      options: [
        { id: 'a', text: '没有更好的办法' },
        { id: 'b', text: '用桶/计数数组维护频次，addNum O(1)，findMedian O(101)' },
        { id: 'c', text: '改用排序数组' },
        { id: 'd', text: '用 set' },
      ],
      answer: 'b',
      explain:
        '当值域很小（101 个桶），可以用「计数数组 / 频次桶」：addNum 只是 cnt[num] += 1（O(1)）；findMedian 从前往后累加频次到中点（O(值域)）。这是面试官可能追问的优化方向，体现「先理解题目数据特征再选数据结构」。',
      tags: ['data-structure', 'complexity'],
    },
    {
      id: 'find-median-data-stream.q12',
      prompt: '若改成「滑动窗口的中位数」（最近 k 个数），双堆还够用吗？',
      options: [
        { id: 'a', text: '完全够，加一个 k 限制即可' },
        { id: 'b', text: '不够，因为 heapq 不支持「删除堆中任意元素」O(log n)；通常用 SortedList 或「双堆 + 延迟删除」' },
        { id: 'c', text: '够，但要每次重建堆' },
        { id: 'd', text: '够，但是要 O(n) 删除' },
      ],
      answer: 'b',
      explain:
        '滑窗中位数要从两堆里删除「过期」元素——而 heapq 没有原生 O(log n) 的任意删除。常见做法：① 改用 sortedcontainers.SortedList；② 双堆 + 延迟删除（用字典记录待删元素，下一次堆顶碰到再真正弹）。这是 LC 480 的难点。',
      tags: ['data-structure', 'invariant'],
    },
  ],
}

export default problem
