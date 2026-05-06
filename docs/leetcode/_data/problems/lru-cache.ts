import type { Problem } from '../types'

const code = `class Node:
    def __init__(self, key=0, val=0):
        self.key, self.val = key, val
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache: dict[int, Node] = {}     # key → 节点指针
        # 双向链表：头部「最近使用」，尾部「最久未用」
        self.head, self.tail = Node(), Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node: Node) -> None:
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add_to_head(node)              # 命中即「挪到表头」
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self._remove(node)
            self._add_to_head(node)
            return
        if len(self.cache) >= self.cap:      # 超容量 → 淘汰表尾
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
        node = Node(key, value)
        self.cache[key] = node
        self._add_to_head(node)`

export const problem: Problem = {
  id: 'lru-cache',
  leetcodeNo: 146,
  title: { zh: 'LRU 缓存', en: 'LRU Cache' },
  difficulty: 'medium',
  pattern: 'linked-list',
  tags: ['design', 'hash-table', 'doubly-linked-list'],
  statement:
    '请你设计并实现一个满足 LRU (最近最少使用) 缓存约束的数据结构。\n\n实现 `LRUCache` 类：\n- `LRUCache(int capacity)` 以**正整数**作为容量初始化 LRU 缓存\n- `int get(int key)` 如果关键字 `key` 存在于缓存中，则返回关键字的值，否则返回 `-1`\n- `void put(int key, int value)` 如果关键字 `key` 已经存在，则变更其数据值；如果不存在，则向缓存中插入该组 `key-value`。如果插入操作导致关键字数量超过容量 `capacity`，则应该**逐出**最久未使用的关键字\n\n函数 `get` 和 `put` 必须以 **O(1)** 的平均时间复杂度运行。',
  examples: [
    {
      input: 'LRUCache(2); put(1,1); put(2,2); get(1); put(3,3); get(2); put(4,4); get(1); get(3); get(4)',
      output: '1, -1, -1, 3, 4',
      note: '容量 2，put(3) 时淘汰 key=2；之后 put(4) 淘汰 key=1',
    },
  ],
  constraints: [
    '1 ≤ capacity ≤ 3000',
    '0 ≤ key ≤ 10⁴',
    '0 ≤ value ≤ 10⁵',
    '调用次数最多 2 × 10⁵ 次',
  ],
  intuition:
    '哈希表 + 双向链表是 O(1) 的「黄金组合」：哈希表把 key 映射到「节点指针」（O(1) 定位）；双向链表维护使用顺序（O(1) 摘除/接入）。head 端永远是「最新」，tail 端永远是「最久」。Python 也有偷懒解：`collections.OrderedDict` + `move_to_end`。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(1) 每次操作', space: 'O(capacity)' },
  microQuestions: [
    {
      id: 'lru-cache.q1',
      prompt: '为什么必须是**双向**链表，而不是单向？',
      options: [
        { id: 'a', text: '单向链表删除节点需要先找前驱（O(n)）；双向链表节点自己持有 prev，O(1) 摘除' },
        { id: 'b', text: '双向才能正确表示顺序' },
        { id: 'c', text: 'Python 不支持单向链表' },
        { id: 'd', text: '为了节省空间' },
      ],
      answer: 'a',
      explain:
        'LRU 的关键是「O(1) 把命中节点挪到表头」——必须能从节点直接断开自己。单向链表要从头扫到前驱才能删，退化为 O(n)。',
      tags: ['data-structure', 'complexity'],
    },
    {
      id: 'lru-cache.q2',
      prompt: '哈希表里存的是「key → 值」还是「key → 节点指针」？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: 'key → 值（int）' },
        { id: 'b', text: 'key → 节点指针' },
        { id: 'c', text: 'key → 节点的索引号' },
        { id: 'd', text: '都可以' },
      ],
      answer: 'b',
      explain:
        '存值的话访问命中后无法 O(1) 找到链表里的对应节点去挪位置；存节点指针后既能直接拿到 .val（满足 get），又能立刻摘除并接到表头。这是 LRU 设计的灵魂。',
      tags: ['data-structure', 'invariant'],
    },
    {
      id: 'lru-cache.q3',
      prompt: 'head 和 tail 两个**哨兵**节点的作用？',
      codeContext: code,
      highlightLine: 13,
      options: [
        { id: 'a', text: '让「在头插入」「在尾删除」与一般操作同质——避免空链表特判' },
        { id: 'b', text: '存储数据' },
        { id: 'c', text: '加速哈希' },
        { id: 'd', text: '没有特别作用' },
      ],
      answer: 'a',
      explain:
        '没哨兵时，向空链表插入第一个节点要单独处理（head/tail 都得改）；删除最后一个节点也要单独处理。两个 dummy 让 _add_to_head 和 _remove 永远只面对「中间节点」的情况。',
      tags: ['data-structure'],
    },
    {
      id: 'lru-cache.q4',
      prompt: '`get(key)` 命中后必须做什么才符合 LRU 语义？',
      codeContext: code,
      highlightLine: 27,
      options: [
        { id: 'a', text: '什么都不做，直接返回值' },
        { id: 'b', text: '把节点摘下并挪到链表头部——更新「最近使用」状态' },
        { id: 'c', text: '把节点的 val 设为 0' },
        { id: 'd', text: '从哈希表删除' },
      ],
      answer: 'b',
      explain:
        'LRU 的 R 是 Recently Used。每次 get 命中都算一次「使用」，所以要把它移到 head。漏掉这一步，LRU 退化为先入先出（FIFO），淘汰逻辑就错了。',
      tags: ['invariant'],
    },
    {
      id: 'lru-cache.q5',
      prompt: '`put` 时如果 key 已存在，正确的处理是？',
      codeContext: code,
      highlightLine: 35,
      options: [
        { id: 'a', text: '直接覆盖 val 即可' },
        { id: 'b', text: '更新 val + 把节点挪到 head（也算一次使用）' },
        { id: 'c', text: '先删再插一个新节点' },
        { id: 'd', text: '抛异常' },
      ],
      answer: 'b',
      explain:
        'put 同样应被视为「最近使用」。只更新值不挪位置会导致它仍可能被淘汰，逻辑错。复用旧节点（不新建）保留了哈希表里指针不变的好处。',
      tags: ['invariant'],
    },
    {
      id: 'lru-cache.q6',
      prompt: '超容量时淘汰哪一端？',
      codeContext: code,
      highlightLine: 42,
      options: [
        { id: 'a', text: '链表头部（最新）' },
        { id: 'b', text: '链表尾部（最久未用）' },
        { id: 'c', text: '中间任意一个' },
        { id: 'd', text: '哈希表里第一个 key' },
      ],
      answer: 'b',
      explain:
        '约定 head 端最新、tail 端最久。淘汰时摘掉 self.tail.prev（哨兵 tail 之前那个真节点）。同时记得 del self.cache[lru.key]，否则哈希表泄漏。',
      tags: ['invariant'],
    },
    {
      id: 'lru-cache.q7',
      prompt: '为什么节点要存 `key`（而不是只存 value）？',
      options: [
        { id: 'a', text: '淘汰节点时需要回头从哈希表里 del 这个 key——必须能从节点拿到 key' },
        { id: 'b', text: '为了打印方便' },
        { id: 'c', text: '加密用' },
        { id: 'd', text: '没有必要' },
      ],
      answer: 'a',
      explain:
        '淘汰流程是「摘链表尾节点 + 从哈希删 key」。如果节点只存 value，哈希就回不去了——哈希里会有死指针指向已被摘的节点。',
      tags: ['invariant', 'data-structure'],
    },
    {
      id: 'lru-cache.q8',
      prompt: '`_remove(node)` 的两行赋值顺序能交换吗？',
      codeContext: code,
      highlightLine: 17,
      options: [
        { id: 'a', text: '可以——两次赋值互不依赖' },
        { id: 'b', text: '不可以——会丢失指针' },
        { id: 'c', text: '只能在双向链表里换' },
        { id: 'd', text: '换了会内存泄漏' },
      ],
      answer: 'a',
      explain:
        '`node.prev.next = node.next` 和 `node.next.prev = node.prev` 各自只读取 node 的属性、写邻居的属性，互不影响。',
      tags: ['boundary'],
    },
    {
      id: 'lru-cache.q9',
      prompt: 'Python 的偷懒写法是什么？',
      options: [
        { id: 'a', text: '`collections.OrderedDict` + `move_to_end` + `popitem(last=False)`' },
        { id: 'b', text: '`functools.lru_cache`' },
        { id: 'c', text: '`heapq` 堆' },
        { id: 'd', text: '`dict.fromkeys`' },
      ],
      answer: 'a',
      explain:
        'OrderedDict 内部就是「dict + 双向链表」的工业实现。`move_to_end(key)` 挪到末（最新），`popitem(last=False)` 弹首（最旧）。面试时先说手撕、有时间再补一句「生产代码会用 OrderedDict」。注意 functools.lru_cache 是缓存函数返回值，方向反了。',
      tags: ['pythonism', 'data-structure'],
    },
    {
      id: 'lru-cache.q10',
      prompt: '`get` 和 `put` 都做到 O(1) 的关键是？',
      options: [
        { id: 'a', text: '哈希 O(1) 定位 + 链表 O(1) 摘接，二者配合' },
        { id: 'b', text: '链表自带索引' },
        { id: 'c', text: 'Python 优化了' },
        { id: 'd', text: '只能近似 O(1)' },
      ],
      answer: 'a',
      explain:
        '只有哈希：能 O(1) 定位但维护不了顺序；只有链表：维护顺序但找节点要 O(n)。「哈希存指针 + 双向链表」让两者各取所长——这就是面试官期待你说出的话。',
      tags: ['complexity', 'data-structure'],
    },
    {
      id: 'lru-cache.q11',
      prompt: '内存上看，capacity = 3000 时大约占用？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(capacity) —— 最多 capacity 个节点 + 哈希表项' },
        { id: 'c', text: 'O(n²)' },
        { id: 'd', text: 'O(log n)' },
      ],
      answer: 'b',
      explain:
        '由 capacity 上限保护——多出的会被淘汰，所以总占用 O(capacity)。',
      tags: ['complexity'],
    },
    {
      id: 'lru-cache.q12',
      prompt: '_add_to_head 中四行赋值的顺序敏感吗？',
      codeContext: code,
      highlightLine: 21,
      options: [
        { id: 'a', text: '是 —— 必须先暂存 head.next 再修改，否则会丢链' },
        { id: 'b', text: '不敏感' },
        { id: 'c', text: '只在多线程时敏感' },
        { id: 'd', text: 'Python GIL 保护了' },
      ],
      answer: 'a',
      explain:
        '原代码的顺序：① 让新节点的 next 指原先的第一个；② 让新节点的 prev 指 head；③ 让原先第一个的 prev 指新节点；④ head.next 指向新节点。如果先做 ④，第 ③ 步就丢了原 head.next。',
      tags: ['boundary', 'invariant'],
    },
  ],
}

export default problem
