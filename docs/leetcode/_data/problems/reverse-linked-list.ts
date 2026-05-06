import type { Problem } from '../types'

const code = `class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def reverseList(head: ListNode | None) -> ListNode | None:
    prev, curr = None, head
    while curr:
        nxt = curr.next       # 1. 暂存下一个，避免掉头后丢失
        curr.next = prev      # 2. 掉头：当前节点指向前一个
        prev = curr           # 3. prev 推进
        curr = nxt            # 4. curr 推进
    return prev               # prev 最终指向原尾节点 = 新头`

export const problem: Problem = {
  id: 'reverse-linked-list',
  leetcodeNo: 206,
  title: { zh: '反转链表', en: 'Reverse Linked List' },
  difficulty: 'easy',
  pattern: 'linked-list',
  tags: ['linked-list', 'recursion'],
  statement:
    '给你单链表的头节点 `head`，请你反转链表，并返回**反转后的链表**。\n\n要求尽可能用迭代或递归两种方式解决，并理解二者的区别。',
  examples: [
    { input: 'head = [1,2,3,4,5]', output: '[5,4,3,2,1]' },
    { input: 'head = [1,2]', output: '[2,1]' },
    { input: 'head = []', output: '[]', note: '空链表合法' },
  ],
  constraints: [
    '链表中节点数目范围 [0, 5000]',
    '-5000 ≤ Node.val ≤ 5000',
  ],
  intuition:
    '迭代版需要三个指针：prev / curr / nxt。核心动作是「先暂存 next，再掉头当前指针，最后整体推进」——四步缺一不可。递归版本更短：reverse(head.next) 之后，让 head.next.next = head 完成反转。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(1)' },
  microQuestions: [
    {
      id: 'reverse-linked-list.q1',
      prompt: '迭代反转最少需要几个指针变量？',
      options: [
        { id: 'a', text: '1 个：curr' },
        { id: 'b', text: '2 个：prev、curr' },
        { id: 'c', text: '3 个：prev、curr、nxt' },
        { id: 'd', text: '4 个：dummy、prev、curr、nxt' },
      ],
      answer: 'c',
      explain:
        '只用 prev 和 curr 时，一旦执行 curr.next = prev 就把原来的 curr.next 覆盖了，丢失对后续节点的引用。所以必须先用 nxt 暂存。dummy 在「反转」里不需要，因为头节点没有变更需要占位。',
      tags: ['data-structure', 'invariant'],
    },
    {
      id: 'reverse-linked-list.q2',
      prompt: '`prev` 的初始值应该是？',
      codeContext: code,
      highlightLine: 6,
      options: [
        { id: 'a', text: 'head' },
        { id: 'b', text: 'head.next' },
        { id: 'c', text: 'None' },
        { id: 'd', text: 'ListNode(0)' },
      ],
      answer: 'c',
      explain:
        '反转后，原来的头节点会变成尾节点，它的 next 必须是 None。让 prev 从 None 开始，第一次循环时 head.next 就被赋值为 None，自然形成新尾。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'reverse-linked-list.q3',
      prompt: '循环体内四步操作的正确顺序是？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: 'curr.next = prev → nxt = curr.next → prev = curr → curr = nxt' },
        { id: 'b', text: 'nxt = curr.next → curr.next = prev → prev = curr → curr = nxt' },
        { id: 'c', text: 'prev = curr → curr = nxt → nxt = curr.next → curr.next = prev' },
        { id: 'd', text: 'nxt = curr.next → prev = curr → curr.next = prev → curr = nxt' },
      ],
      answer: 'b',
      explain:
        '必须先 nxt = curr.next 暂存，否则一旦 curr.next = prev 之后再去取 curr.next 拿到的就是 prev 而非原来的下一个。顺序错了等于把链表折成自环或丢节点。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'reverse-linked-list.q4',
      prompt: '循环结束时 `prev` 指向什么？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: '原链表的头节点' },
        { id: 'b', text: '原链表的尾节点（也就是新链表的头）' },
        { id: 'c', text: 'None' },
        { id: 'd', text: '中间节点' },
      ],
      answer: 'b',
      explain:
        '退出条件是 `curr is None`，此时 curr 走到了尾节点之后；而 prev 恰好停留在原尾节点上，正是反转后的新头。',
      tags: ['invariant'],
    },
    {
      id: 'reverse-linked-list.q5',
      prompt: '空链表（head 为 None）这份代码会发生什么？',
      options: [
        { id: 'a', text: 'AttributeError，因为 head.next 不存在' },
        { id: 'b', text: '直接进入 while 但不进入循环体，返回 None' },
        { id: 'c', text: '死循环' },
        { id: 'd', text: '需要在开头单独 if head is None 判断' },
      ],
      answer: 'b',
      explain:
        '`while curr` 在 curr 为 None 时不进入循环体，prev 保持初始 None 被返回。所以这份代码不需要为空链表写额外分支——这就是好设计。',
      tags: ['boundary'],
    },
    {
      id: 'reverse-linked-list.q6',
      prompt: '递归版反转的核心一行 `head.next.next = head` 含义是？',
      options: [
        { id: 'a', text: '让 head 的下一个节点的下一个指向自己——形成反向链接' },
        { id: 'b', text: '让 head 跳过一个节点' },
        { id: 'c', text: '把链表展开为右链表' },
        { id: 'd', text: '一种 Python 语法糖' },
      ],
      answer: 'a',
      explain:
        '递归先翻转 head.next 之后，head.next 还指向原来的「下一个节点」（因为还没改）。让那个节点的 next 反指 head 就完成了一对反转。再把 head.next 设为 None 防自环。',
      tags: ['invariant'],
    },
    {
      id: 'reverse-linked-list.q7',
      prompt: '递归版的空间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(log n)' },
        { id: 'c', text: 'O(n) —— 调用栈深度' },
        { id: 'd', text: 'O(n²)' },
      ],
      answer: 'c',
      explain:
        '递归会一直深入到尾节点才开始返回，调用栈深 n 层。这就是为什么数据量大时建议迭代版——Python 还有默认 1000 层递归上限。',
      tags: ['complexity'],
    },
    {
      id: 'reverse-linked-list.q8',
      prompt: '为什么这道题不需要 dummy 哨兵节点？',
      options: [
        { id: 'a', text: '题目禁止使用 dummy' },
        { id: 'b', text: '不会出现「头节点被删除」「在头前插入」这类操作；prev 已经足够替代' },
        { id: 'c', text: '链表必非空' },
        { id: 'd', text: 'dummy 会增加常数时间' },
      ],
      answer: 'b',
      explain:
        'dummy 的核心价值是：让头节点和中间节点行为一致（删头 = 删中间）。本题没有删除/插入头节点，prev 在每一步都扮演了「前驱」的角色，足够。',
      tags: ['data-structure', 'pythonism'],
    },
    {
      id: 'reverse-linked-list.q9',
      prompt: '如果把 `prev = None` 改成 `prev = ListNode(0)`，结果会怎样？',
      options: [
        { id: 'a', text: '完全等价' },
        { id: 'b', text: '反转后新链表会多出一个值为 0 的尾节点' },
        { id: 'c', text: '抛出异常' },
        { id: 'd', text: '更安全，因为可以避免空指针' },
      ],
      answer: 'b',
      explain:
        '原来的头节点会指向这个 0 节点，所以反转后链表末尾多了一个 0。这就是为什么 prev 必须是 None——它代表「新尾的下一个不存在」。',
      tags: ['boundary'],
    },
    {
      id: 'reverse-linked-list.q10',
      prompt: '迭代版的时间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(log n)' },
        { id: 'c', text: 'O(n)' },
        { id: 'd', text: 'O(n²)' },
      ],
      answer: 'c',
      explain:
        '每个节点恰好被访问一次（curr 推进一次，操作四步常数）。',
      tags: ['complexity'],
    },
    {
      id: 'reverse-linked-list.q11',
      prompt: 'Python 里 `prev, curr = None, head` 这种写法叫什么？',
      options: [
        { id: 'a', text: '元组解包（tuple unpacking）' },
        { id: 'b', text: '链式赋值' },
        { id: 'c', text: '类型注解' },
        { id: 'd', text: '解构（destructuring）' },
      ],
      answer: 'a',
      explain:
        'Python 把 `None, head` 视作一个 tuple，再左侧解包到两个变量。链式赋值是 `a = b = 0`，两者不同。这种写法在交换变量时尤其方便：`a, b = b, a`。',
      tags: ['pythonism', 'syntax'],
    },
    {
      id: 'reverse-linked-list.q12',
      prompt: '若题目改成「反转链表的前 K 个节点」，最直接的改动是？',
      options: [
        { id: 'a', text: '把 while curr 改为 while curr and 计数 < K，并把剩下的接回来' },
        { id: 'b', text: '换成递归' },
        { id: 'c', text: '需要重新设计算法' },
        { id: 'd', text: '不可能在 O(n) 完成' },
      ],
      answer: 'a',
      explain:
        '保留迭代框架，加一个计数器；循环结束后让原 head（已变尾）的 next 指向 curr（剩下未反转的开头）。这就是 LeetCode #25 K 个一组的思路雏形。',
      tags: ['boundary'],
    },
  ],
}

export default problem
