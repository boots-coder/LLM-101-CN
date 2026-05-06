import type { Problem } from '../types'

const code = `class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

# 解法一：递归
def inorderTraversal(root: TreeNode | None) -> list[int]:
    res = []
    def dfs(node: TreeNode | None) -> None:
        if not node:
            return
        dfs(node.left)
        res.append(node.val)
        dfs(node.right)
    dfs(root)
    return res

# 解法二：迭代（栈 + 一路向左）
def inorderTraversalIter(root: TreeNode | None) -> list[int]:
    res, stack = [], []
    curr = root
    while curr or stack:
        # 1. 一路向左压栈
        while curr:
            stack.append(curr)
            curr = curr.left
        # 2. 弹一个并处理
        node = stack.pop()
        res.append(node.val)
        # 3. 转到右子树（继续走 1）
        curr = node.right
    return res`

export const problem: Problem = {
  id: 'inorder-traversal',
  leetcodeNo: 94,
  title: { zh: '二叉树的中序遍历', en: 'Binary Tree Inorder Traversal' },
  difficulty: 'easy',
  pattern: 'tree-dfs',
  tags: ['tree', 'dfs', 'stack', 'recursion'],
  statement:
    '给定一个二叉树的根节点 `root`，返回它的**中序遍历**。\n\n递归实现简单，但请尝试用**迭代**实现（用栈模拟递归）——这是 BST 类题目的基础工具。',
  examples: [
    { input: 'root = [1,null,2,3]', output: '[1,3,2]' },
    { input: 'root = []', output: '[]' },
    { input: 'root = [1]', output: '[1]' },
  ],
  constraints: [
    '树中节点数目在范围 [0, 100]',
    '-100 ≤ Node.val ≤ 100',
  ],
  intuition:
    '中序 = 左 → 根 → 右。递归就是这三行的字面翻译。迭代版用栈：「一路向左压到底，弹一个就处理它，再转去右子树继续向左压」——这个动作用栈精确刻画了递归的隐式调用栈。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(h) h 为树高' },
  microQuestions: [
    {
      id: 'inorder-traversal.q1',
      prompt: '中序遍历的访问顺序是？',
      options: [
        { id: 'a', text: '根 → 左 → 右（前序）' },
        { id: 'b', text: '左 → 根 → 右（中序）' },
        { id: 'c', text: '左 → 右 → 根（后序）' },
        { id: 'd', text: '层序' },
      ],
      answer: 'b',
      explain:
        '中序的「中」指根节点访问位置在中间。在 BST 上，中序遍历输出的就是有序序列——这是大量 BST 题的核心利用点。',
      tags: ['naming'],
    },
    {
      id: 'inorder-traversal.q2',
      prompt: '递归终止条件 `if not node: return` 缺了会怎样？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: 'AttributeError —— None 没有 .left' },
        { id: 'b', text: '死循环' },
        { id: 'c', text: '返回错的结果但不报错' },
        { id: 'd', text: '没有影响' },
      ],
      answer: 'a',
      explain:
        '一旦 node 是 None 就尝试 node.left → AttributeError。每个 dfs 函数的「第一行」就是终止条件，肌肉记忆。',
      tags: ['boundary'],
    },
    {
      id: 'inorder-traversal.q3',
      prompt: '迭代版中「一路向左压栈」的动作模拟了递归的什么？',
      codeContext: code,
      highlightLine: 21,
      options: [
        { id: 'a', text: '函数返回过程' },
        { id: 'b', text: '函数调用入栈过程——把还没访问的祖先全部记住' },
        { id: 'c', text: '尾递归优化' },
        { id: 'd', text: '没有对应' },
      ],
      answer: 'b',
      explain:
        '递归隐式用调用栈保存「未处理的左祖先」；迭代版用显式栈做同一件事。压到最左叶子时，栈顶就是「下一个该处理的节点」。',
      tags: ['data-structure', 'invariant'],
    },
    {
      id: 'inorder-traversal.q4',
      prompt: '迭代版退出循环的条件是什么？',
      codeContext: code,
      highlightLine: 19,
      options: [
        { id: 'a', text: 'curr is None 即可' },
        { id: 'b', text: '栈为空即可' },
        { id: 'c', text: 'curr 为 None **且** 栈为空' },
        { id: 'd', text: 'curr 为 None **或** 栈为空' },
      ],
      answer: 'c',
      explain:
        '只看 curr 不够：curr 在「弹完节点转到右子树」时可能是 None，但栈里还有未处理的祖先。只看栈也不够：栈空时仍可能有右子树没遍历完（curr 不为 None）。两者都空才结束。',
      tags: ['boundary'],
    },
    {
      id: 'inorder-traversal.q5',
      prompt: '弹出节点 `node` 后，`curr = node.right` 这一步意义？',
      codeContext: code,
      highlightLine: 27,
      options: [
        { id: 'a', text: '准备进入「右子树」继续按相同流程「一路向左」' },
        { id: 'b', text: '加快遍历' },
        { id: 'c', text: '为了递归剪枝' },
        { id: 'd', text: '可以省略' },
      ],
      answer: 'a',
      explain:
        '左子树和当前已处理；接下来按中序定义「转到右子树」。但右子树本身又是一棵树——继续走外层循环的「一路向左」逻辑即可。',
      tags: ['invariant'],
    },
    {
      id: 'inorder-traversal.q6',
      prompt: '空间复杂度 O(h) 中 h 是什么？最坏情况下 h 可能多大？',
      options: [
        { id: 'a', text: 'h = 节点数 n（链状树时退化）' },
        { id: 'b', text: 'h = log n（永远）' },
        { id: 'c', text: 'h = 2' },
        { id: 'd', text: 'h = O(1)' },
      ],
      answer: 'a',
      explain:
        'h 是树高。平衡树 h ≈ log n；最坏情况（链状）h = n。所以本题最坏空间 O(n)，但不写 O(n)是因为大多数时候 O(h) 更准确。',
      tags: ['complexity'],
    },
    {
      id: 'inorder-traversal.q7',
      prompt: '在 BST 上做中序遍历，输出会有什么特别？',
      options: [
        { id: 'a', text: '没有特别' },
        { id: 'b', text: '输出**严格递增**——可作为「验证 BST」「BST 第 k 小」等题的金钥匙' },
        { id: 'c', text: '输出递减' },
        { id: 'd', text: '取决于插入顺序' },
      ],
      answer: 'b',
      explain:
        'BST 性质 + 左→根→右遍历 ⇒ 输出有序。LeetCode #98、#230、#108 都基于此。背下「BST 中序 = 有序」可以一招通杀几道题。',
      tags: ['invariant'],
    },
    {
      id: 'inorder-traversal.q8',
      prompt: '递归版 `res` 用闭包变量比起作为参数传递有什么优势？',
      options: [
        { id: 'a', text: '不需要在每次递归调用里传一个长长的列表参数；调用更简洁' },
        { id: 'b', text: '速度更快' },
        { id: 'c', text: '内存更省' },
        { id: 'd', text: 'Python 不允许参数传 list' },
      ],
      answer: 'a',
      explain:
        'Python 的列表是引用，参数传递不会复制——所以两种写法本质等价。但闭包写法签名更短，dfs 只关心「访问当前节点」这个语义。',
      tags: ['pythonism'],
    },
    {
      id: 'inorder-traversal.q9',
      prompt: '为什么递归版用 `res.append(node.val)` 而不是 `return res + ...`？',
      options: [
        { id: 'a', text: 'append 是 O(1)；列表拼接 O(n)，多次拼接退化为 O(n²)' },
        { id: 'b', text: 'return 写法语法错' },
        { id: 'c', text: 'append 更短' },
        { id: 'd', text: '没区别' },
      ],
      answer: 'a',
      explain:
        '`return inorder(left) + [node.val] + inorder(right)` 也能 AC 但每次拼接复制整段列表，最坏 O(n²)。append 模式是「一次性收集」，O(n)。',
      tags: ['complexity', 'pythonism'],
    },
    {
      id: 'inorder-traversal.q10',
      prompt: 'Python 默认递归深度大约多少？',
      options: [
        { id: 'a', text: '1000' },
        { id: 'b', text: '10000' },
        { id: 'c', text: '无限' },
        { id: 'd', text: '由树高决定' },
      ],
      answer: 'a',
      explain:
        '默认 sys.getrecursionlimit() ≈ 1000。链状树超过 1000 个节点会 RecursionError。可用 `sys.setrecursionlimit(10**6)` 提升，但栈空间也有限——这是为什么大数据时偏好迭代。',
      tags: ['pythonism', 'boundary'],
    },
    {
      id: 'inorder-traversal.q11',
      prompt: '把「中序」改成「前序」，递归版的改动是？',
      options: [
        { id: 'a', text: '把 res.append(node.val) 移到 dfs(node.left) 之前' },
        { id: 'b', text: '把递归改成迭代' },
        { id: 'c', text: '完全重写' },
        { id: 'd', text: '不可能改' },
      ],
      answer: 'a',
      explain:
        '前/中/后序的递归版只差「访问根」这一行的位置——前：在递归两子树之前；中：在中间；后：在两子树之后。这就是为什么三种遍历的递归代码看起来几乎一模一样。',
      tags: ['pythonism'],
    },
    {
      id: 'inorder-traversal.q12',
      prompt: '想把空间压到 O(1)（不计输出），可以用什么技巧？',
      options: [
        { id: 'a', text: 'Morris 遍历 —— 利用空指针建立临时线索' },
        { id: 'b', text: '迭代版改写' },
        { id: 'c', text: '递归 + 记忆化' },
        { id: 'd', text: '不可能' },
      ],
      answer: 'a',
      explain:
        'Morris 遍历是面试加分项：找当前节点左子树的最右节点，把它的 right 指向当前节点（线索），遍历到时再恢复。空间 O(1)，时间仍 O(n)。代码很 tricky，写出来面试官会眼前一亮。',
      tags: ['data-structure'],
    },
  ],
}

export default problem
