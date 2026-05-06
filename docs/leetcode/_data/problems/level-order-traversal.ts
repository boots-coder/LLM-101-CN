import type { Problem } from '../types'

const code = `from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def levelOrder(root: TreeNode | None) -> list[list[int]]:
    if not root:
        return []
    res = []
    q = deque([root])
    while q:
        size = len(q)            # 关键：先记下当前层节点数
        level = []
        for _ in range(size):    # 只处理当前层这么多个
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(level)
    return res`

export const problem: Problem = {
  id: 'level-order-traversal',
  leetcodeNo: 102,
  title: { zh: '二叉树的层序遍历', en: 'Binary Tree Level Order Traversal' },
  difficulty: 'medium',
  pattern: 'tree-bfs',
  tags: ['tree', 'bfs', 'queue'],
  statement:
    '给你二叉树的根节点 `root`，返回其节点值的**层序遍历**（即逐层地，从左到右访问所有节点）。\n\n输出形式是一个二维列表，每个内层列表是一层的节点值。',
  examples: [
    { input: 'root = [3,9,20,null,null,15,7]', output: '[[3],[9,20],[15,7]]' },
    { input: 'root = [1]', output: '[[1]]' },
    { input: 'root = []', output: '[]' },
  ],
  constraints: [
    '树中节点数目在范围 [0, 2000]',
    '-1000 ≤ Node.val ≤ 1000',
  ],
  intuition:
    'BFS 模板的直接应用：用 deque 当队列，进入每轮循环时**先记下当前层的大小 `size`**，然后 for size 次只处理当前层的节点（同时把它们的子节点入队作为下一层）。这一行 `size = len(q)` 是层级感的灵魂——把「混在一起的下一层」与「正在处理的当前层」隔开。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(n)' },
  microQuestions: [
    {
      id: 'level-order-traversal.q1',
      prompt: 'BFS 树为什么用 `deque` 而不是 list？',
      codeContext: code,
      highlightLine: 12,
      options: [
        { id: 'a', text: 'list.pop(0) 是 O(n)；deque.popleft 是 O(1)。BFS 涉及大量「从前端取」' },
        { id: 'b', text: 'deque 自动按层' },
        { id: 'c', text: 'list 不能存 TreeNode' },
        { id: 'd', text: '只是约定' },
      ],
      answer: 'a',
      explain:
        'BFS 每个节点都要 popleft 一次。list 实现是动态数组，从前端弹要左移所有元素，每次 O(n)；n 次 O(n²)。deque 双端 O(1)。',
      tags: ['pythonism', 'complexity'],
    },
    {
      id: 'level-order-traversal.q2',
      prompt: '`size = len(q)` 这一行的意义是什么？',
      codeContext: code,
      highlightLine: 14,
      options: [
        { id: 'a', text: '记下「当前层有多少节点」——这一轮只处理 size 个，其他都是下一层（已混入队列）' },
        { id: 'b', text: '只是为了避免 len 多次计算' },
        { id: 'c', text: '校验队列大小' },
        { id: 'd', text: '没特别意义' },
      ],
      answer: 'a',
      explain:
        '处理当前层的节点时，会把它们的子节点（即下一层）追加到队尾。如果不先记 size，`while q` 会把当前层和下一层混在一起。「先记 size，再 for size 次」是 BFS 分层的灵魂。',
      tags: ['invariant'],
    },
    {
      id: 'level-order-traversal.q3',
      prompt: 'for 循环里能不能写 `for _ in range(len(q))` 替代 `size = len(q); for _ in range(size)`？',
      options: [
        { id: 'a', text: '不能 —— Python 的 range 在创建时就用当前 len(q) 作为上限固定下来吗？答案是：固定的，所以两种写法等价' },
        { id: 'b', text: '不能 —— len(q) 每次循环都重算，会包含新入队节点' },
        { id: 'c', text: '完全不能等价' },
        { id: 'd', text: '会抛错' },
      ],
      answer: 'a',
      explain:
        '`range(len(q))` 在迭代开始时就生成 range 对象（上限固定）。两种写法都等价。但显式 `size = len(q)` 更清楚地表达意图，可读性强。',
      tags: ['pythonism'],
    },
    {
      id: 'level-order-traversal.q4',
      prompt: '`if not root: return []` 必须放在最前面，原因是？',
      codeContext: code,
      highlightLine: 9,
      options: [
        { id: 'a', text: '否则 deque([None]) 会把 None 当成节点入队，进入循环时 node.val 抛 AttributeError' },
        { id: 'b', text: '只是为了快速 fail' },
        { id: 'c', text: '题目要求' },
        { id: 'd', text: '没必要' },
      ],
      answer: 'a',
      explain:
        '如果不判断，`deque([root])` 会装一个 None，循环里 `node.val` 立即报错。或者改成「入队前判 if root: q.append」也行——总之要避免 None 进队。',
      tags: ['boundary'],
    },
    {
      id: 'level-order-traversal.q5',
      prompt: '入队子节点时为什么要写 `if node.left:` 判断？',
      codeContext: code,
      highlightLine: 19,
      options: [
        { id: 'a', text: '避免 None 进队（与 q4 相同的不变量）' },
        { id: 'b', text: '加速' },
        { id: 'c', text: 'Python 不允许 append None' },
        { id: 'd', text: '没必要' },
      ],
      answer: 'a',
      explain:
        '保持「队列里全是真实节点」的不变量，是后续无脑访问 .val/.left/.right 的前提。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'level-order-traversal.q6',
      prompt: '本题用 BFS 的好处相比 DFS 是？',
      options: [
        { id: 'a', text: '天然按层组织节点；DFS 也能做但需要带 depth 参数才能分层' },
        { id: 'b', text: 'BFS 一定更快' },
        { id: 'c', text: 'DFS 不能解' },
        { id: 'd', text: 'BFS 空间更省' },
      ],
      answer: 'a',
      explain:
        'DFS 写法：dfs(node, depth)，res.setdefault(depth, []).append(node.val)，最后转 list。BFS 直接「按层弹出」，无需 depth。复杂度同 O(n)，空间 BFS 是「最宽一层」、DFS 是「树高」。',
      tags: ['data-structure'],
    },
    {
      id: 'level-order-traversal.q7',
      prompt: '本题空间复杂度 O(n) 主要来自？',
      options: [
        { id: 'a', text: '队列最多容纳「最宽一层」的节点；最坏（满二叉树最后一层）约 n/2' },
        { id: 'b', text: '递归栈' },
        { id: 'c', text: '哈希表' },
        { id: 'd', text: '输出 res' },
      ],
      answer: 'a',
      explain:
        '满二叉树时最后一层占 n/2，是队列的峰值；这就是 BFS 的空间特征——「最宽一层」决定空间上限。',
      tags: ['complexity'],
    },
    {
      id: 'level-order-traversal.q8',
      prompt: '若题目改成「锯齿形层序」（奇数层从右到左），最少改动是？',
      options: [
        { id: 'a', text: '维护一个 flag，奇数层翻转 level 列表；或用 deque 双端 append' },
        { id: 'b', text: '完全重写' },
        { id: 'c', text: '不可能' },
        { id: 'd', text: '需要 DFS' },
      ],
      answer: 'a',
      explain:
        '`level.reverse()` 在奇层翻转、或用 collections.deque 让奇层 appendleft——两种思路都是面试上的小变形。',
      tags: ['data-structure'],
    },
    {
      id: 'level-order-traversal.q9',
      prompt: '`deque([root])` 这种初始化把整个 list 一次入队，等价于？',
      options: [
        { id: 'a', text: 'q = deque(); q.append(root)' },
        { id: 'b', text: 'q = deque(root)' },
        { id: 'c', text: 'q = list([root])' },
        { id: 'd', text: '都不对' },
      ],
      answer: 'a',
      explain:
        '`deque(iterable)` 把 iterable 中的每个元素作为初始节点。`deque(root)` 会把 root 当 iterable 调用——TreeNode 不可迭代会报错。',
      tags: ['pythonism'],
    },
    {
      id: 'level-order-traversal.q10',
      prompt: '能不能用「记每个节点的 depth」一次循环输出，而不分两层 for？',
      options: [
        { id: 'a', text: '能：q 里存 (node, depth) 元组，根据 depth 把 val 推入对应子列表' },
        { id: 'b', text: '不能' },
        { id: 'c', text: '会出错' },
        { id: 'd', text: '只能 DFS' },
      ],
      answer: 'a',
      explain:
        '另一种风格——一层 while 循环，用 depth 索引到 res[depth]。代码更短但「层级感」弱。两种风格都常见。',
      tags: ['data-structure'],
    },
    {
      id: 'level-order-traversal.q11',
      prompt: '本题时间复杂度是？',
      options: [
        { id: 'a', text: 'O(n) —— 每个节点 popleft 一次、入队一次，O(1) 操作' },
        { id: 'b', text: 'O(n log n)' },
        { id: 'c', text: 'O(n²)' },
        { id: 'd', text: '取决于树形' },
      ],
      answer: 'a',
      explain:
        '每个节点恰好被处理一次。append/popleft 都是 O(1)。',
      tags: ['complexity'],
    },
    {
      id: 'level-order-traversal.q12',
      prompt: 'BFS 与 DFS 在「树高 h」与「最宽层 w」上的空间偏好？',
      options: [
        { id: 'a', text: 'DFS 空间 O(h)；BFS 空间 O(w)。瘦高树用 DFS，矮宽树用 DFS 也可——通常 BFS 空间更大' },
        { id: 'b', text: 'BFS 永远更省' },
        { id: 'c', text: 'DFS 永远更省' },
        { id: 'd', text: '一样' },
      ],
      answer: 'a',
      explain:
        '完全二叉树 h=log n、w=n/2，BFS 占用大；链状 h=n、w=1，DFS 占用大。这是工程选择题——大多数 leetcode 树题用 BFS 是因为 h 通常不至于退化。',
      tags: ['complexity'],
    },
  ],
}

export default problem
