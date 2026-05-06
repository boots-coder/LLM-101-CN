import type { Problem } from '../types'

const code = `import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def mergeKLists(lists: list[ListNode | None]) -> ListNode | None:
    # 用堆维护当前每条链表的「最前未取节点」；总共 N 节点、K 条链
    heap = []
    for i, node in enumerate(lists):
        if node:
            # 元组里第二位放 i 是「打破值相等时的比较僵局」——
            # 因为 ListNode 没有定义 __lt__，直接比较两个节点会 TypeError
            heapq.heappush(heap, (node.val, i, node))
    dummy = ListNode(0)
    tail = dummy
    while heap:
        val, i, node = heapq.heappop(heap)
        tail.next = node
        tail = node
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next`

export const problem: Problem = {
  id: 'merge-k-sorted-lists',
  leetcodeNo: 23,
  title: { zh: '合并 K 个升序链表', en: 'Merge k Sorted Lists' },
  difficulty: 'hard',
  pattern: 'linked-list',
  tags: ['linked-list', 'heap', 'divide-and-conquer'],
  statement:
    '给你一个链表数组，每个链表都已经按升序排列。\n\n请你将所有链表合并到一个升序链表中，返回合并后的链表。',
  examples: [
    { input: 'lists = [[1,4,5],[1,3,4],[2,6]]', output: '[1,1,2,3,4,4,5,6]' },
    { input: 'lists = []', output: '[]' },
    { input: 'lists = [[]]', output: '[]', note: '只有空链表也合法' },
  ],
  constraints: [
    'k == lists.length',
    '0 ≤ k ≤ 10⁴',
    '0 ≤ lists[i].length ≤ 500',
    'lists[i][j] 升序排列且 -10⁴ ≤ lists[i][j] ≤ 10⁴',
  ],
  intuition:
    '两条主流：① **最小堆**——把每条链表的当前头入堆，每次弹最小、把它的 next 入堆，O(N log K)；② **分治两两合并**，类似归并排序合并，O(N log K)。两者复杂度同，堆胜在思路直观。注意 Python 堆需要解决「ListNode 不可比较」问题——元组里塞个唯一索引 i 当 tiebreaker。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(N log K)', space: 'O(K)' },
  microQuestions: [
    {
      id: 'merge-k-sorted-lists.q1',
      prompt: '为什么堆解法的时间复杂度是 O(N log K) 而非 O(N log N)？（N 总节点数，K 链表数）',
      options: [
        { id: 'a', text: '堆里同时只有 K 个元素，每次 push/pop 是 O(log K)；N 次操作合计 O(N log K)' },
        { id: 'b', text: '堆是 O(1) 的' },
        { id: 'c', text: 'log K 是排序常数' },
        { id: 'd', text: '题目数据小到看不出区别' },
      ],
      answer: 'a',
      explain:
        '堆里始终只装「每条链表当前最前」的节点——最多 K 个。每弹一个、再补一个，堆大小不变。N 次操作 × O(log K) = O(N log K)。这就是为什么它比「全部入堆再排序」更优。',
      tags: ['complexity', 'data-structure'],
    },
    {
      id: 'merge-k-sorted-lists.q2',
      prompt: '为什么 heappush 的元组里要放 `i`（链表索引）作为第二位？',
      codeContext: code,
      highlightLine: 14,
      options: [
        { id: 'a', text: '为了找回是哪条链表' },
        { id: 'b', text: '因为 ListNode 没定义 __lt__，值相等时第二位用来打破平手——避免 TypeError' },
        { id: 'c', text: '只是日志方便' },
        { id: 'd', text: '随便加的' },
      ],
      answer: 'b',
      explain:
        'heapq 比较元组是按位置依次比。两个 (val, ?, node) 在 val 相等时会比第二位；如果第二位也相等才会去比 node——但 ListNode 没有 __lt__ 会抛 TypeError。i 唯一就保证不会比到第三位。',
      tags: ['pythonism', 'data-structure'],
    },
    {
      id: 'merge-k-sorted-lists.q3',
      prompt: 'dummy 哨兵节点在这道题里的作用？',
      codeContext: code,
      highlightLine: 16,
      options: [
        { id: 'a', text: '让「拼接第一个节点」与「拼接后续节点」逻辑一致——避免单独处理头' },
        { id: 'b', text: '加快堆操作' },
        { id: 'c', text: '保证有序' },
        { id: 'd', text: '防止 OOM' },
      ],
      answer: 'a',
      explain:
        '没 dummy 时第一次 tail.next = node 会失败（tail 没有指向）。dummy 让所有节点都能用 tail.next = ... 的统一动作处理，最后 return dummy.next 跳过哨兵。',
      tags: ['data-structure'],
    },
    {
      id: 'merge-k-sorted-lists.q4',
      prompt: '弹出堆顶后，下一步关键操作是？',
      codeContext: code,
      highlightLine: 21,
      options: [
        { id: 'a', text: '把弹出节点的 next 入堆——保持「每条链有且仅有一个候选在堆里」' },
        { id: 'b', text: '把整条剩余链表入堆' },
        { id: 'c', text: '什么都不做' },
        { id: 'd', text: '把弹出节点重新入堆' },
      ],
      answer: 'a',
      explain:
        '这是堆解法的精髓——堆永远只装 K 个候选（每条链表当前最前未取的）。弹一个就补它后面那个，保持不变量；若 next 是 None 就不补，堆自然缩小。',
      tags: ['invariant', 'data-structure'],
    },
    {
      id: 'merge-k-sorted-lists.q5',
      prompt: '初始入堆时为什么要 `if node:` 判断？',
      codeContext: code,
      highlightLine: 12,
      options: [
        { id: 'a', text: '题目允许 lists 中含空链表（None）' },
        { id: 'b', text: '加速运行' },
        { id: 'c', text: '避免 ListNode 自身 __bool__ 异常' },
        { id: 'd', text: 'Python 要求' },
      ],
      answer: 'a',
      explain:
        'lists 可能含 None（如 [[1,2], None, [3,4]]）。直接读 node.val 会 AttributeError。这一行是「先跳过空链表」的边界保护。',
      tags: ['boundary'],
    },
    {
      id: 'merge-k-sorted-lists.q6',
      prompt: 'Python heapq 是哪种堆？',
      options: [
        { id: 'a', text: '最大堆' },
        { id: 'b', text: '最小堆' },
        { id: 'c', text: '可以切换的双端堆' },
        { id: 'd', text: '取决于数据' },
      ],
      answer: 'b',
      explain:
        'heapq 永远是最小堆——堆顶是最小元素。要最大堆请把 val 取负或用 (-val, ...) 入堆。',
      tags: ['pythonism', 'data-structure'],
    },
    {
      id: 'merge-k-sorted-lists.q7',
      prompt: '分治版「两两合并」的时间复杂度也是 O(N log K) 而非 O(NK)，关键原因是？',
      options: [
        { id: 'a', text: '每一轮合并涉及全部 N 节点，但轮数只有 log K' },
        { id: 'b', text: '使用了 Tim 排序优化' },
        { id: 'c', text: '因为分治随机化' },
        { id: 'd', text: '不可能是 O(N log K)' },
      ],
      answer: 'a',
      explain:
        '把 K 条链表两两合并：第 1 轮 K/2 次合并，每次平均 2N/K 节点，总 O(N)；共 log K 轮 → O(N log K)。如果像「用一条链反复 merge 其他 K-1 条」，会变成 O(NK)——这是常见错误。',
      tags: ['complexity'],
    },
    {
      id: 'merge-k-sorted-lists.q8',
      prompt: '空间复杂度 O(K) 来自？',
      options: [
        { id: 'a', text: '堆中最多 K 个节点元组' },
        { id: 'b', text: '输出链表本身 O(N)' },
        { id: 'c', text: '递归栈' },
        { id: 'd', text: '复制了所有节点' },
      ],
      answer: 'a',
      explain:
        '堆同时只持有 K 个元组（每条链一个候选）。输出链表是「重新接线」原节点，没有新建——按惯例不计入。',
      tags: ['complexity'],
    },
    {
      id: 'merge-k-sorted-lists.q9',
      prompt: '为什么不能直接 `heapq.heappush(heap, node)`（只放节点不放元组）？',
      options: [
        { id: 'a', text: 'Python 会按 ListNode 默认比较——但 ListNode 未定义 __lt__，抛 TypeError' },
        { id: 'b', text: '语法错误' },
        { id: 'c', text: '性能太差' },
        { id: 'd', text: '完全可以' },
      ],
      answer: 'a',
      explain:
        '堆需要「比较两个元素谁更小」。原生类型有 __lt__，自定义类没有，需要手动包装。两条路：包元组 (val, i, node)，或给 ListNode 加 __lt__。',
      tags: ['pythonism'],
    },
    {
      id: 'merge-k-sorted-lists.q10',
      prompt: '把所有节点先全部入堆再依次弹（不动态补）的复杂度？',
      options: [
        { id: 'a', text: 'O(N log N) —— 比 O(N log K) 差' },
        { id: 'b', text: '相同' },
        { id: 'c', text: 'O(N)' },
        { id: 'd', text: 'O(K log N)' },
      ],
      answer: 'a',
      explain:
        '堆同时持有 N 个元素 → 每次操作 O(log N)。这就是为什么「滑动维护 K 大小堆」是亮点：把 log N 压到 log K。当 K << N 时差距明显。',
      tags: ['complexity'],
    },
    {
      id: 'merge-k-sorted-lists.q11',
      prompt: '`heapq.heappop(heap)` 返回什么？',
      options: [
        { id: 'a', text: '堆中最小元素' },
        { id: 'b', text: '堆中最大元素' },
        { id: 'c', text: '随机一个元素' },
        { id: 'd', text: '抛异常' },
      ],
      answer: 'a',
      explain:
        'heapq 是最小堆，heappop 弹出并返回最小值（O(log n)）；空堆 pop 会抛 IndexError。',
      tags: ['pythonism', 'data-structure'],
    },
    {
      id: 'merge-k-sorted-lists.q12',
      prompt: '相比堆解法，分治两两合并的优势是？',
      options: [
        { id: 'a', text: '不依赖额外数据结构（只要会 merge two lists）；常数也更小' },
        { id: 'b', text: '复杂度更低' },
        { id: 'c', text: '空间 O(1)' },
        { id: 'd', text: '没有优势' },
      ],
      answer: 'a',
      explain:
        '复杂度同 O(N log K)。分治版只用了 LeetCode #21 的 mergeTwoLists 作为子函数，不需要堆，常数项更小。面试时若已经写过 #21，分治更显「会复用」。',
      tags: ['complexity'],
    },
    {
      id: 'merge-k-sorted-lists.q13',
      prompt: '如果 lists 全是 None（每条链都是空），代码会？',
      options: [
        { id: 'a', text: '正确返回 None（因为 dummy.next 一直没被赋值，初始就是 None）' },
        { id: 'b', text: '抛异常' },
        { id: 'c', text: '死循环' },
        { id: 'd', text: '返回 dummy 节点' },
      ],
      answer: 'a',
      explain:
        '初始入堆时 if node: 全部跳过，堆是空的；while heap 不进入；返回 dummy.next 即 None。这就是 dummy 的另一个隐形价值——天然处理空输出。',
      tags: ['boundary'],
    },
  ],
}

export default problem
