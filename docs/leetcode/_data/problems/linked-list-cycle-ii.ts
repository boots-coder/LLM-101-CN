import type { Problem } from '../types'

const code = `class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def detectCycle(head: ListNode | None) -> ListNode | None:
    slow = fast = head
    # 阶段一：快慢指针在环内相遇
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            break
    else:
        return None                # 走到尽头说明无环
    if not (fast and fast.next):
        return None                # 兼容 break 后再判断的情况

    # 阶段二：把任一指针放回 head，同速前进，再次相遇即入口
    slow = head
    while slow is not fast:
        slow = slow.next
        fast = fast.next
    return slow                    # 数学结论：a = c + (n-1) * L`

export const problem: Problem = {
  id: 'linked-list-cycle-ii',
  leetcodeNo: 142,
  title: { zh: '环形链表 II', en: 'Linked List Cycle II' },
  difficulty: 'medium',
  pattern: 'linked-list',
  tags: ['linked-list', 'two-pointers', 'floyd'],
  statement:
    '给定一个链表的头节点 `head`，返回链表开始入环的**第一个节点**。如果链表无环，则返回 `null`。\n\n如果链表中存在环，则返回环入口节点；不允许修改链表，要求空间复杂度 O(1)。',
  examples: [
    { input: 'head = [3,2,0,-4], pos = 1', output: '返回索引 1 的节点（值为 2）', note: 'pos 表示尾节点连接到链表中的哪个位置' },
    { input: 'head = [1,2], pos = 0', output: '返回索引 0 的节点（值为 1）' },
    { input: 'head = [1], pos = -1', output: 'null', note: '无环' },
  ],
  constraints: [
    '链表中节点的数目范围在 [0, 10⁴]',
    '-10⁵ ≤ Node.val ≤ 10⁵',
    'pos ∈ [-1, n-1]，-1 表示无环',
  ],
  intuition:
    'Floyd 判圈两阶段：① 快慢指针 fast 走 2 步、slow 走 1 步，若有环必相遇；② 把一个指针放回 head，两者**同速**前进，再次相遇就是入口。数学推导：设头到入口距离 a，入口到相遇点距离 b，相遇点回到入口距离 c，环长 L=b+c。fast 走 2(a+b) = a+b+nL（n 是 fast 在环内多绕的圈数）⇒ a = c + (n-1)L。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(1)' },
  microQuestions: [
    {
      id: 'linked-list-cycle-ii.q1',
      prompt: '为什么 fast 走 2 步而 slow 走 1 步，而不是 3 步和 1 步？',
      options: [
        { id: 'a', text: '速度差为 1 时，每一步差距增加 1，进入环后必然相遇' },
        { id: 'b', text: '速度差更大也行，但「同速 → 入口」的数学结论只对差 1 成立' },
        { id: 'c', text: 'a 和 b 都对' },
        { id: 'd', text: '只是惯例没有特别原因' },
      ],
      answer: 'c',
      explain:
        '差为 1 保证「最坏情况下绕一圈必相遇」；同时 a = c + (n-1)L 这个公式是 fast 步频是 slow 两倍才推出的。换成 3:1 仍能判环，但第二阶段的「同速回头」结论就失效了。',
      tags: ['invariant'],
    },
    {
      id: 'linked-list-cycle-ii.q2',
      prompt: '`while fast and fast.next:` 这两个条件分别防止什么？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: 'fast 防空链表，fast.next 防尾节点（取 .next.next 会越界）' },
        { id: 'b', text: '都是冗余检查' },
        { id: 'c', text: 'fast 防自环' },
        { id: 'd', text: '只是为了可读性' },
      ],
      answer: 'a',
      explain:
        'fast = fast.next.next 需要 fast 和 fast.next 都不为 None。如果链表无环，fast 在某一步一定会到 None 或某个 .next 是 None——这就是判定无环的标志。',
      tags: ['boundary'],
    },
    {
      id: 'linked-list-cycle-ii.q3',
      prompt: 'Floyd 第二阶段「a = c + (n-1)L」中 a、c、L 分别代表？',
      options: [
        { id: 'a', text: 'a=头→入口，c=相遇点→入口，L=环长' },
        { id: 'b', text: 'a=头→相遇点，c=环长，L=入口→相遇点' },
        { id: 'c', text: 'a=链表总长，c=环长，L=圈数' },
        { id: 'd', text: '都不对' },
      ],
      answer: 'a',
      explain:
        'fast 走 2(a+b)，slow 走 a+b。两者差是 nL（n 圈），所以 a+b = nL ⇒ a = nL - b = (n-1)L + (L-b) = (n-1)L + c。从 head 走 a 步、从相遇点走 c 步，都到入口。',
      tags: ['invariant'],
    },
    {
      id: 'linked-list-cycle-ii.q4',
      prompt: '为什么第二阶段两个指针**同速**前进就能在入口相遇？',
      options: [
        { id: 'a', text: '走 a 步与走 c+(n-1)L 步终点相同——都落在入口' },
        { id: 'b', text: '随便选的，凑巧成立' },
        { id: 'c', text: '因为环对称' },
        { id: 'd', text: '需要 fast 仍然走 2 步' },
      ],
      answer: 'a',
      explain:
        '从 head 走 a 步必到入口；从相遇点开始，走 c 步到入口，再走 (n-1)L 等于绕回入口（绕几圈无所谓）。两者同时到入口就一定在入口相遇。',
      tags: ['invariant'],
    },
    {
      id: 'linked-list-cycle-ii.q5',
      prompt: '判断「相遇」用 `slow is fast` 还是 `slow == fast` 更严谨？',
      options: [
        { id: 'a', text: 'is —— 比较对象身份（同一个节点）' },
        { id: 'b', text: '== —— 比较值' },
        { id: 'c', text: '完全等价' },
        { id: 'd', text: 'is 更慢' },
      ],
      answer: 'a',
      explain:
        '两个不同节点可能 val 相同（题目允许重复值）。`is` 比较内存地址（同一对象），永远正确；`==` 默认对自定义类是 is，但若用户重写 __eq__ 就出 bug。链表题用 is 是肌肉记忆。',
      tags: ['pythonism'],
    },
    {
      id: 'linked-list-cycle-ii.q6',
      prompt: '若链表无环，第一阶段 while 循环如何退出？',
      codeContext: code,
      highlightLine: 13,
      options: [
        { id: 'a', text: 'fast 走到 None 或 fast.next 是 None，进入 else 分支返回 None' },
        { id: 'b', text: 'slow 追上 fast' },
        { id: 'c', text: '手动 break' },
        { id: 'd', text: '永远不会退出' },
      ],
      answer: 'a',
      explain:
        'Python 的 while-else 在「自然结束」（条件假）时进入 else，break 不会进入。无环时 fast 终会撞到 None 让条件假——else 触发。',
      tags: ['boundary', 'pythonism'],
    },
    {
      id: 'linked-list-cycle-ii.q7',
      prompt: '能不能用「哈希表存访问过的节点」替代 Floyd？',
      options: [
        { id: 'a', text: '能，但空间 O(n)；Floyd 是 O(1) 更优' },
        { id: 'b', text: '不能，会丢解' },
        { id: 'c', text: '只能用哈希表' },
        { id: 'd', text: '哈希表不支持节点对象' },
      ],
      answer: 'a',
      explain:
        'set 存节点指针，遇到已存在的就是入口——思路简单但空间 O(n)。Floyd 的精髓正是空间 O(1)，这是本题面试时强调「O(1) 空间」的原因。',
      tags: ['complexity', 'data-structure'],
    },
    {
      id: 'linked-list-cycle-ii.q8',
      prompt: '`slow = fast = head` 这种链式赋值的语义？',
      codeContext: code,
      highlightLine: 6,
      options: [
        { id: 'a', text: 'slow 和 fast 指向同一个节点（同一对象）' },
        { id: 'b', text: '把 fast 复制一份给 slow' },
        { id: 'c', text: '只给 fast 赋值' },
        { id: 'd', text: '语法错误' },
      ],
      answer: 'a',
      explain:
        '从右到左求值：head → fast → slow。三者引用同一节点。修改 slow.next 不会影响 fast.next 是因为它们是变量重绑定后才不同——但初始指向同一对象。',
      tags: ['pythonism', 'syntax'],
    },
    {
      id: 'linked-list-cycle-ii.q9',
      prompt: '当环极小（L=1，单节点自环）时这份代码会？',
      options: [
        { id: 'a', text: '抛异常' },
        { id: 'b', text: '正确返回那个自环节点' },
        { id: 'c', text: '死循环' },
        { id: 'd', text: '返回 None' },
      ],
      answer: 'b',
      explain:
        '第一步 slow 和 fast 都从 head 出发；slow = head.next = head（自环）；fast = head.next.next = head；立刻相遇 → 第二阶段 slow=head, fast=head 已经相等，循环不进入，直接返回 head。',
      tags: ['boundary'],
    },
    {
      id: 'linked-list-cycle-ii.q10',
      prompt: '本算法的时间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(log n)' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(n²)' },
      ],
      answer: 'c',
      explain:
        '阶段一最多 n + L 步，阶段二最多 a 步，加起来 O(n)。',
      tags: ['complexity'],
    },
    {
      id: 'linked-list-cycle-ii.q11',
      prompt: '如果把第二阶段「slow 回到 head」改成「fast 回到 head」，结果如何？',
      options: [
        { id: 'a', text: '结果一样——两者对称' },
        { id: 'b', text: '会出错' },
        { id: 'c', text: 'fast 必须保持原速' },
        { id: 'd', text: '需要重新推导' },
      ],
      answer: 'a',
      explain:
        '两者完全对称——只要其中一个回 head，另一个留在相遇点，都同速前进。变量名只是约定。',
      tags: ['invariant'],
    },
    {
      id: 'linked-list-cycle-ii.q12',
      prompt: 'Python 的 `while ... else` 在哪些时机进入 else？',
      options: [
        { id: 'a', text: 'while 条件正常假退出时进入；break 跳出时**不**进入' },
        { id: 'b', text: '永远进入' },
        { id: 'c', text: 'break 跳出时进入' },
        { id: 'd', text: 'try-except 中才有用' },
      ],
      answer: 'a',
      explain:
        'Python 特有的 while-else / for-else 用于「正常完成搜索 vs 中途找到」的区分。本题里：自然走完 = 无环（else 返回 None）；break = 找到环。',
      tags: ['pythonism', 'syntax'],
    },
  ],
}

export default problem
