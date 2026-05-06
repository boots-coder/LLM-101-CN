import type { Problem } from '../types'

const code = `class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def lowestCommonAncestor(
    root: TreeNode | None,
    p: TreeNode,
    q: TreeNode,
) -> TreeNode | None:
    # 终止条件：空 / 撞到 p / 撞到 q —— 把当前节点上抛
    if root is None or root is p or root is q:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    # 左右都「找到了」——当前节点就是 LCA
    if left and right:
        return root
    # 只有一边找到——把那一边的结果上抛（可能是 p、q 之一，也可能是子树里的 LCA）
    return left if left else right`

export const problem: Problem = {
  id: 'lowest-common-ancestor',
  leetcodeNo: 236,
  title: { zh: '二叉树的最近公共祖先', en: 'Lowest Common Ancestor of a Binary Tree' },
  difficulty: 'medium',
  pattern: 'tree-dfs',
  tags: ['tree', 'dfs', 'recursion'],
  statement:
    '给定一个二叉树，找到该树中**两个指定节点的最近公共祖先（LCA）**。\n\n最近公共祖先的定义为：「对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。」',
  examples: [
    { input: 'root = [3,5,1,6,2,0,8,null,null,7,4], p=5, q=1', output: '3' },
    { input: 'root = [3,5,1,6,2,0,8,null,null,7,4], p=5, q=4', output: '5', note: '5 是自己的祖先' },
    { input: 'root = [1,2], p=1, q=2', output: '1' },
  ],
  constraints: [
    '树中节点数目范围在 [2, 10⁵]',
    '-10⁹ ≤ Node.val ≤ 10⁹',
    'p ≠ q，p 和 q 均存在于树中',
  ],
  intuition:
    '后序回溯。对每个节点问：「左子树能找到 p 或 q 吗？右子树呢？」——三种情况：① 两边都找到 → 当前节点就是 LCA；② 只一边找到 → 把那一边的结果上抛（要么是 p/q 本体，要么是更深处的 LCA）；③ 都没找到 → 返回 None。终止条件「撞到 p 或 q 立刻返回」是关键剪枝。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(h)' },
  microQuestions: [
    {
      id: 'lowest-common-ancestor.q1',
      prompt: '终止条件 `root is p or root is q` 撞到立刻返回 root，背后的语义是？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: '一个节点是它自己的祖先；同时把「找到了」这个信号上抛给父亲' },
        { id: 'b', text: '剪枝优化' },
        { id: 'c', text: '随便写' },
        { id: 'd', text: '处理空节点' },
      ],
      answer: 'a',
      explain:
        '题目明确「一个节点也是自己的祖先」。撞到 p 时不需要再深入找 q——如果 q 在 p 子树里，p 就是答案；如果 q 不在，answer 也会在更高层组合时被发现。两种情况通过把 p 上抛都能正确处理。',
      tags: ['invariant', 'boundary'],
    },
    {
      id: 'lowest-common-ancestor.q2',
      prompt: '左右递归都返回非空时，为什么当前 root 就是 LCA？',
      codeContext: code,
      highlightLine: 16,
      options: [
        { id: 'a', text: '说明 p 和 q 分别在左右两棵子树里——能同时覆盖二者的最浅节点就是 root' },
        { id: 'b', text: '随机取一个' },
        { id: 'c', text: '取 left 或 right 都行' },
        { id: 'd', text: 'root 永远是 LCA' },
      ],
      answer: 'a',
      explain:
        '一边找到 p、另一边找到 q（或者各自找到的是「子树 LCA」但因为 p≠q 必须分两侧），它们的最近公共祖先就是当前 root——再上去就经过祖父，距离更远。',
      tags: ['invariant'],
    },
    {
      id: 'lowest-common-ancestor.q3',
      prompt: '只有 left 非空时，应该上抛什么？',
      codeContext: code,
      highlightLine: 19,
      options: [
        { id: 'a', text: 'left —— 它要么是 p/q 本体，要么是子树里已经找好的 LCA' },
        { id: 'b', text: 'root' },
        { id: 'c', text: 'None' },
        { id: 'd', text: 'right' },
      ],
      answer: 'a',
      explain:
        '右子树没找到任何东西，意味着 p 和 q 都在左子树（或只有 p/q 一个节点在树里）。把 left 直接上抛——内部已经决定了是 LCA 还是 p/q 之一，外层不必关心。',
      tags: ['invariant'],
    },
    {
      id: 'lowest-common-ancestor.q4',
      prompt: '什么是「后序回溯」式递归？',
      options: [
        { id: 'a', text: '先递归左右子树，再用子树结果决定当前节点的返回值（自底向上）' },
        { id: 'b', text: '先访问根再访问子树' },
        { id: 'c', text: '只用迭代' },
        { id: 'd', text: '层序遍历的一种' },
      ],
      answer: 'a',
      explain:
        'LCA、最大路径和、二叉树的高度等都用这个模板：递归先深入获得左右结果，回来再合并——典型的「分治后组合」。',
      tags: ['data-structure'],
    },
    {
      id: 'lowest-common-ancestor.q5',
      prompt: '为什么用 `is` 而不是 `==`？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: '`is` 比较「是不是同一个对象」——题目给的是节点引用，不能用值比较' },
        { id: 'b', text: '速度更快' },
        { id: 'c', text: 'Python 规定' },
        { id: 'd', text: '== 也对' },
      ],
      answer: 'a',
      explain:
        '不同节点可能 val 相等（题目允许）。`==` 默认对自定义类是 is，但万一有人重写 __eq__ 就出事。链表/树题里比较节点身份永远用 `is`。',
      tags: ['pythonism'],
    },
    {
      id: 'lowest-common-ancestor.q6',
      prompt: '若 p 在 q 的子树里，递归的执行轨迹是？',
      options: [
        { id: 'a', text: '到 q 时立刻返回 q（终止条件触发）；q 上面的节点在「只 left 非空」分支里把 q 一路上抛——最终结果就是 q' },
        { id: 'b', text: '需要再深入找 p' },
        { id: 'c', text: '会出错' },
        { id: 'd', text: '返回 None' },
      ],
      answer: 'a',
      explain:
        '撞到 q 后剪枝、不再深入；q 子树里有没有 p 此时不重要——q 本身已经是 p 和 q 的 LCA。「不深入」恰好是正确的剪枝。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'lowest-common-ancestor.q7',
      prompt: '本算法时间复杂度是？',
      options: [
        { id: 'a', text: 'O(n)' },
        { id: 'b', text: 'O(n log n)' },
        { id: 'c', text: 'O(n²)' },
        { id: 'd', text: 'O(h) 仅与树高有关' },
      ],
      answer: 'a',
      explain:
        '最坏情况遍历整棵树（如 p、q 一个在最左叶、一个在最右叶）。每个节点工作 O(1)。',
      tags: ['complexity'],
    },
    {
      id: 'lowest-common-ancestor.q8',
      prompt: '空间复杂度 O(h) 来自？',
      options: [
        { id: 'a', text: '递归栈深度 = 树高' },
        { id: 'b', text: '哈希表' },
        { id: 'c', text: '输出列表' },
        { id: 'd', text: '常数' },
      ],
      answer: 'a',
      explain:
        '只有递归栈消耗。链状树最坏 O(n)，平衡树 O(log n)。',
      tags: ['complexity'],
    },
    {
      id: 'lowest-common-ancestor.q9',
      prompt: '若题目改成「BST 的 LCA」，能不能写出更优解？',
      options: [
        { id: 'a', text: '能：利用 BST 性质，从 root 开始按 val 比较走一边即可——O(h) 时间，O(1) 空间（迭代）' },
        { id: 'b', text: '一样' },
        { id: 'c', text: '更慢' },
        { id: 'd', text: '不可能' },
      ],
      answer: 'a',
      explain:
        'LeetCode #235：当 p.val 和 q.val 都比 root 小，去左；都比 root 大，去右；分叉时 root 就是 LCA。这是利用 BST 序的经典优化。',
      tags: ['data-structure'],
    },
    {
      id: 'lowest-common-ancestor.q10',
      prompt: '若 p 不在树里，本代码会怎样？',
      options: [
        { id: 'a', text: '返回 q —— 但题目保证 p、q 都在树中，所以不需要担心' },
        { id: 'b', text: '抛异常' },
        { id: 'c', text: '死循环' },
        { id: 'd', text: '返回 None' },
      ],
      answer: 'a',
      explain:
        '没有撞到 p 时左右递归只有 q 一边非空，会一路上抛 q——「假 LCA」。题目给的约束「p、q 都存在」让这个边界不需处理。如果题目放宽，需要先确认两个节点都找到。',
      tags: ['boundary'],
    },
    {
      id: 'lowest-common-ancestor.q11',
      prompt: '`return left if left else right` 这种写法叫？',
      options: [
        { id: 'a', text: '条件表达式（三元运算符）' },
        { id: 'b', text: 'lambda 表达式' },
        { id: 'c', text: 'walrus 运算符' },
        { id: 'd', text: '解构赋值' },
      ],
      answer: 'a',
      explain:
        'Python 的 `x if cond else y`。也可以等价写 `return left or right` 因为 None 是 falsy——更 Pythonic 但可读性略差。',
      tags: ['pythonism', 'syntax'],
    },
    {
      id: 'lowest-common-ancestor.q12',
      prompt: '面试时如果让你不用递归怎么做？',
      options: [
        { id: 'a', text: '记录每个节点的父指针（一次 BFS/DFS），然后从 p 一路向上做集合，再从 q 向上找第一个碰到的——O(n) 时间 O(n) 空间' },
        { id: 'b', text: '不可能不用递归' },
        { id: 'c', text: '只能 BFS' },
        { id: 'd', text: '用堆' },
      ],
      answer: 'a',
      explain:
        '迭代写法：第一遍建 parent map，第二遍从 p 一路把祖先加进 set，第三遍从 q 沿 parent 找第一个在 set 里的节点。空间换递归栈。面试时能给两种解法是加分项。',
      tags: ['data-structure'],
    },
  ],
}

export default problem
