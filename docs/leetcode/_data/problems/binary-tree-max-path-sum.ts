import type { Problem } from '../types'

const code = `class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

class Solution:
    def maxPathSum(self, root: TreeNode | None) -> int:
        self.ans = float('-inf')

        def gain(node: TreeNode | None) -> int:
            # 返回：以 node 为端点向下走的「最大单边和」
            if not node:
                return 0
            # 子树负贡献则直接舍弃（取 0）——不向下延伸更优
            left = max(gain(node.left), 0)
            right = max(gain(node.right), 0)
            # 「以 node 为顶点跨左右」的路径和——拿来更新全局答案
            self.ans = max(self.ans, node.val + left + right)
            # 但**返回**只能选一边——因为父节点要把当前节点串入它的路径
            return node.val + max(left, right)

        gain(root)
        return self.ans`

export const problem: Problem = {
  id: 'binary-tree-max-path-sum',
  leetcodeNo: 124,
  title: { zh: '二叉树中的最大路径和', en: 'Binary Tree Maximum Path Sum' },
  difficulty: 'hard',
  pattern: 'tree-dfs',
  tags: ['tree', 'dfs', 'recursion'],
  statement:
    '二叉树中的**路径**被定义为一条从树中**任意**节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中**至多出现一次**。该路径**至少包含一个**节点，且不一定经过根节点。\n\n**路径和**是路径中各节点值的总和。\n\n给你一个二叉树的根节点 `root`，返回其**最大路径和**。',
  examples: [
    { input: 'root = [1,2,3]', output: '6', note: '路径 2 → 1 → 3，和为 6' },
    { input: 'root = [-10,9,20,null,null,15,7]', output: '42', note: '路径 15 → 20 → 7' },
    { input: 'root = [-3]', output: '-3', note: '单节点也是合法路径' },
  ],
  constraints: [
    '树中节点数目范围是 [1, 3 × 10⁴]',
    '-1000 ≤ Node.val ≤ 1000',
  ],
  intuition:
    '关键洞察：「DFS 返回值」与「全局最大答案」**不是同一件事**。返回值是「以本节点为端点向下走的最大单边和」（父节点要拿来续路径，只能选一边）；全局答案是「以本节点为顶点跨左右的最大路径和」（不会再向上延伸）。负贡献的子树直接 max(..., 0) 舍弃。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(h)' },
  microQuestions: [
    {
      id: 'binary-tree-max-path-sum.q1',
      prompt: '本题最易错的概念是什么？',
      options: [
        { id: 'a', text: 'DFS 返回值就是答案' },
        { id: 'b', text: '「返回给父节点的值」与「全局最大答案」**不是**一回事' },
        { id: 'c', text: '路径必须经过根节点' },
        { id: 'd', text: '只能向下走' },
      ],
      answer: 'b',
      explain:
        '父节点要把当前节点串入它的路径——只能选「左单边」或「右单边」之一（路径不能分叉再合并）；但全局答案允许「跨左右」（以本节点为最高点）。两个量必须分开维护，否则要么答案小、要么递归出错。',
      tags: ['invariant'],
    },
    {
      id: 'binary-tree-max-path-sum.q2',
      prompt: '`max(gain(node.left), 0)` 这一行的意思？',
      codeContext: code,
      highlightLine: 14,
      options: [
        { id: 'a', text: '左子树贡献为负则舍弃（不延伸到左侧）' },
        { id: 'b', text: '取绝对值' },
        { id: 'c', text: '处理空节点' },
        { id: 'd', text: '随便加的' },
      ],
      answer: 'a',
      explain:
        '路径可以「在当前节点就停」。如果左子树最优单边和是负的，加进来反而拉低总和——干脆当成 0。这是非常 Pythonic 的「负贡献剪枝」。',
      tags: ['invariant'],
    },
    {
      id: 'binary-tree-max-path-sum.q3',
      prompt: '`self.ans = max(self.ans, node.val + left + right)` 中为什么是 `+ left + right`？',
      codeContext: code,
      highlightLine: 17,
      options: [
        { id: 'a', text: '在「以 node 为顶点」的视角里，路径可以同时向左和向右——这才是跨左右的最大值' },
        { id: 'b', text: '应该是 max(left, right)' },
        { id: 'c', text: '随便写的' },
        { id: 'd', text: 'left * right' },
      ],
      answer: 'a',
      explain:
        '全局答案考虑的是「这棵树里最大路径」。如果路径以 node 为最高点，可以伸两边。这是更新全局答案的时刻；返回给父亲时则只能选一边。',
      tags: ['invariant'],
    },
    {
      id: 'binary-tree-max-path-sum.q4',
      prompt: '为什么返回时是 `node.val + max(left, right)`？',
      codeContext: code,
      highlightLine: 19,
      options: [
        { id: 'a', text: '父节点接当前节点后，路径只能向单边延伸（不能分叉再合并）' },
        { id: 'b', text: '为了减少计算' },
        { id: 'c', text: 'Python 限制' },
        { id: 'd', text: 'left + right 会超时' },
      ],
      answer: 'a',
      explain:
        '路径是简单路径——每个节点最多被穿过一次。父节点拿到当前节点后只能继续走到它一侧子孙；选 max 的那一边即可。',
      tags: ['invariant'],
    },
    {
      id: 'binary-tree-max-path-sum.q5',
      prompt: '`self.ans = float("-inf")` 而不是 `0` 的原因？',
      codeContext: code,
      highlightLine: 7,
      options: [
        { id: 'a', text: '节点值可以是负数（如 [-3]），初始 0 会得到错误答案 0' },
        { id: 'b', text: '为了避免类型转换' },
        { id: 'c', text: '是 Python 习惯' },
        { id: 'd', text: '没差' },
      ],
      answer: 'a',
      explain:
        '`-1000 ≤ val ≤ 1000`，且路径至少 1 个节点。若全是负数，正确答案应为最大那个负数（如 -3），不是 0。`-inf` 是「比任何 int 都小」的安全初值。',
      tags: ['boundary'],
    },
    {
      id: 'binary-tree-max-path-sum.q6',
      prompt: '为什么用 `self.ans` 而不是 `nonlocal ans`？',
      options: [
        { id: 'a', text: '挂在实例上方便外部访问/调试；nonlocal 也行，但语法稍长' },
        { id: 'b', text: 'Python 不支持 nonlocal' },
        { id: 'c', text: 'self.ans 更快' },
        { id: 'd', text: '必须挂实例上' },
      ],
      answer: 'a',
      explain:
        '两种都对。leetcode 解法常见 self.ans 是因为本身就在 class Solution 里。注意 Python 闭包对**不可变变量**默认是只读，要改值必须 `nonlocal`，否则会创建同名局部变量——这是新手最易踩坑。',
      tags: ['pythonism'],
    },
    {
      id: 'binary-tree-max-path-sum.q7',
      prompt: '空节点 dfs 应返回什么？',
      codeContext: code,
      highlightLine: 12,
      options: [
        { id: 'a', text: '0 —— 「空子树贡献为 0」是个干净的不变量' },
        { id: 'b', text: '负无穷' },
        { id: 'c', text: '抛异常' },
        { id: 'd', text: 'None' },
      ],
      answer: 'a',
      explain:
        '空子树没节点贡献。返回 0 时再外层用 max(..., 0) 不会引入多余惩罚——逻辑自洽。返回 -inf 在外层 max(..., 0) 时变 0 也对，但一致性差。',
      tags: ['boundary'],
    },
    {
      id: 'binary-tree-max-path-sum.q8',
      prompt: '本题的递归形态属于「DFS 三态」中的哪一种？',
      options: [
        { id: 'a', text: '归纳（自底向上聚合子树结果）' },
        { id: 'b', text: '带状态下传（参数累加）' },
        { id: 'c', text: '分治后组合' },
        { id: 'd', text: 'a 和 c 的混合：每个节点用左右子树结果做局部更新（归纳），同时分治计算单边和' },
      ],
      answer: 'd',
      explain:
        '「拿到左右单边和」是分治；「用 self.ans 在当前层更新全局」是归纳；本题没有「带状态下传」（不需要把累加和往下传给子树）。三态可以组合出现。',
      tags: ['invariant'],
    },
    {
      id: 'binary-tree-max-path-sum.q9',
      prompt: '时间复杂度 O(n)，关键在于？',
      options: [
        { id: 'a', text: '每个节点恰好被访问一次；每次操作 O(1)' },
        { id: 'b', text: 'O(log n) 树高' },
        { id: 'c', text: '排序的 log 项' },
        { id: 'd', text: '动态规划展开' },
      ],
      answer: 'a',
      explain:
        '一次 DFS 遍历整棵树。每个节点的工作就是 max 几次、加一次——O(1)。',
      tags: ['complexity'],
    },
    {
      id: 'binary-tree-max-path-sum.q10',
      prompt: '空间复杂度 O(h) 来自？',
      options: [
        { id: 'a', text: '递归调用栈深度 = 树高' },
        { id: 'b', text: 'res 列表' },
        { id: 'c', text: '哈希表' },
        { id: 'd', text: 'self.ans 一个 float' },
      ],
      answer: 'a',
      explain:
        '没有任何辅助数据结构（self.ans 是常数），唯一的空间开销就是递归栈。最坏链状树时退化为 O(n)。',
      tags: ['complexity'],
    },
    {
      id: 'binary-tree-max-path-sum.q11',
      prompt: '若把 `self.ans = max(self.ans, node.val + left + right)` 改成 `node.val + max(left, right)` 会怎样？',
      options: [
        { id: 'a', text: '会丢解——错过「跨左右」的最优路径' },
        { id: 'b', text: '完全等价' },
        { id: 'c', text: '只是慢一点' },
        { id: 'd', text: '会死循环' },
      ],
      answer: 'a',
      explain:
        '比如 [1,2,3] 的最优是 2+1+3=6（跨左右）；如果只取单边最大就只有 1+3=4。这是核心错误模式。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'binary-tree-max-path-sum.q12',
      prompt: '把 `self.ans` 改成局部变量 `ans = -inf` 然后在 dfs 里赋值，结果如何？',
      options: [
        { id: 'a', text: 'Python 对外层不可变变量默认只读，dfs 内的 ans = ... 会创建新局部变量——外层 ans 永远是 -inf；必须用 nonlocal' },
        { id: 'b', text: '完全没问题' },
        { id: 'c', text: '语法错误' },
        { id: 'd', text: '速度更快' },
      ],
      answer: 'a',
      explain:
        '这是新手最易掉进去的 Python 闭包陷阱。可变对象（list/dict）用 .append 等方法不需要 nonlocal；但「重新赋值不可变变量」必须 nonlocal 或挂在 self 上。',
      tags: ['pythonism'],
    },
  ],
}

export default problem
