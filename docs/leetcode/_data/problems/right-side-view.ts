import type { Problem } from '../types'

const code = `from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

# 解法一：BFS 取每层最后一个
def rightSideView(root: TreeNode | None) -> list[int]:
    if not root:
        return []
    res = []
    q = deque([root])
    while q:
        size = len(q)
        for i in range(size):
            node = q.popleft()
            if i == size - 1:           # 当前层最后一个 = 右视图能看到的
                res.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
    return res

# 解法二：DFS 先右后左，按 depth 第一次见到才记录
def rightSideViewDFS(root: TreeNode | None) -> list[int]:
    res = []
    def dfs(node: TreeNode | None, depth: int) -> None:
        if not node:
            return
        if depth == len(res):           # 这一层第一次被访问到
            res.append(node.val)
        dfs(node.right, depth + 1)      # 关键：先右后左
        dfs(node.left, depth + 1)
    dfs(root, 0)
    return res`

export const problem: Problem = {
  id: 'right-side-view',
  leetcodeNo: 199,
  title: { zh: '二叉树的右视图', en: 'Binary Tree Right Side View' },
  difficulty: 'medium',
  pattern: 'tree-bfs',
  tags: ['tree', 'bfs', 'dfs'],
  statement:
    '给定一个二叉树的根节点 `root`，想象自己站在它的**右侧**，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。\n\n注意：是「每层最右侧能被看见的那一个」，不是「整棵树的最右链」——因为左子树深处的节点也可能比右子树短的部分更深。',
  examples: [
    { input: 'root = [1,2,3,null,5,null,4]', output: '[1,3,4]' },
    { input: 'root = [1,null,3]', output: '[1,3]' },
    { input: 'root = []', output: '[]' },
  ],
  constraints: [
    '二叉树的节点个数的范围是 [0, 100]',
    '-100 ≤ Node.val ≤ 100',
  ],
  intuition:
    '两条主流：① BFS 按层处理，每层取最后一个节点（i == size - 1）；② DFS 先右后左，每个 depth 第一次被访问到时就是右视图的那个节点。两种写法都 O(n)。注意「最右链」不是右视图——若左子树比右子树深，最深那层的右视图来自左子树。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(n)' },
  microQuestions: [
    {
      id: 'right-side-view.q1',
      prompt: '为什么 BFS 解法取「每层最后一个」就是右视图？',
      codeContext: code,
      highlightLine: 16,
      options: [
        { id: 'a', text: '入队顺序「先左后右」，所以每层从左到右——最后一个就是最右' },
        { id: 'b', text: '随便取一个就行' },
        { id: 'c', text: '取第一个' },
        { id: 'd', text: '需要排序' },
      ],
      answer: 'a',
      explain:
        '关键不变量是「入队顺序决定层内顺序」。先左后右入队 → popleft 时从左到右 → 第 size-1 个就是这一层最右节点（站在右侧能看到的那个）。',
      tags: ['invariant'],
    },
    {
      id: 'right-side-view.q2',
      prompt: '「右视图」是不是「树的最右一条链」？',
      options: [
        { id: 'a', text: '不一定 —— 若右子树比左子树短，左子树深处的节点会成为该层的右视图' },
        { id: 'b', text: '永远是' },
        { id: 'c', text: '当 BST 时是' },
        { id: 'd', text: '看树的结构' },
      ],
      answer: 'a',
      explain:
        '示例 [1,2,3,null,5,null,4]：根 1，左子树 2(右子5)，右子树 3(右子4)。第三层只有 5（左子树里），所以右视图是 [1,3,4]——不是「最右链」[1,3,4] 那种刚好的情况。再举 [1,2,3,4]（左子树更深）右视图是 [1,3,4]，4 来自左子树。',
      tags: ['invariant', 'boundary'],
    },
    {
      id: 'right-side-view.q3',
      prompt: 'DFS 解法为什么要「先右后左」？',
      codeContext: code,
      highlightLine: 32,
      options: [
        { id: 'a', text: '让每层第一次访问到的节点就是「最右那个」——配合 depth==len(res) 判断收集' },
        { id: 'b', text: '为了节省栈空间' },
        { id: 'c', text: '随便' },
        { id: 'd', text: 'Python 默认' },
      ],
      answer: 'a',
      explain:
        'DFS 的「先序」性质：在每一层，第一个被深入到的节点会被率先「收下」。先右递归保证我们从右往左探，每层第一个见到的就是最右节点；之后左子树访问到同一层时被 depth==len(res) 拦下。',
      tags: ['invariant'],
    },
    {
      id: 'right-side-view.q4',
      prompt: 'DFS 解法用 `depth == len(res)` 判断这一层是否第一次出现，原理是？',
      codeContext: code,
      highlightLine: 30,
      options: [
        { id: 'a', text: 'res 长度等于已收集的层数；当我们第一次到达 depth 层时 len(res) 恰好等于 depth，可以 append' },
        { id: 'b', text: '巧合' },
        { id: 'c', text: '哈希性质' },
        { id: 'd', text: 'Python 自动' },
      ],
      answer: 'a',
      explain:
        '这是「同步增长」技巧：res 永远是「已收集的最右值」按层排列；下一层第一次到达时 depth = len(res)，append 后两者同时增长一格。后续访问同层节点时 depth < len(res) 跳过。',
      tags: ['invariant', 'pythonism'],
    },
    {
      id: 'right-side-view.q5',
      prompt: '若把 BFS 解法的入队顺序改成「先右后左」，要怎么改才能仍正确？',
      options: [
        { id: 'a', text: '取每层第一个（i == 0）而不是最后一个' },
        { id: 'b', text: '不需要改' },
        { id: 'c', text: '会自动正确' },
        { id: 'd', text: '不能改' },
      ],
      answer: 'a',
      explain:
        '入队顺序决定层内顺序。先右后左入队 → 每层 popleft 出来从右到左 → 第一个就是最右。逻辑要相应翻转，不然会得到「左视图」。',
      tags: ['invariant'],
    },
    {
      id: 'right-side-view.q6',
      prompt: '`if not root: return []` 缺失会发生什么？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: 'deque([None]) 入队后循环里 node.val 抛 AttributeError' },
        { id: 'b', text: '返回 None 而不是 []' },
        { id: 'c', text: '没影响' },
        { id: 'd', text: '死循环' },
      ],
      answer: 'a',
      explain:
        '边界保护与 #102 同源：必须保证队列永远只装真实节点。',
      tags: ['boundary'],
    },
    {
      id: 'right-side-view.q7',
      prompt: 'BFS 和 DFS 两种解法时间复杂度是？',
      options: [
        { id: 'a', text: '都是 O(n)' },
        { id: 'b', text: 'BFS O(n)，DFS O(n²)' },
        { id: 'c', text: '都是 O(n log n)' },
        { id: 'd', text: '取决于树高' },
      ],
      answer: 'a',
      explain:
        '两种都遍历每个节点恰好一次。差别在常数和空间——BFS 队列 O(w)、DFS 栈 O(h)。',
      tags: ['complexity'],
    },
    {
      id: 'right-side-view.q8',
      prompt: '若题目变成「左视图」，BFS 解法最简单的改动？',
      options: [
        { id: 'a', text: '把 `i == size - 1` 改成 `i == 0`，取每层第一个' },
        { id: 'b', text: '完全重写' },
        { id: 'c', text: '调换入队顺序' },
        { id: 'd', text: '不可能' },
      ],
      answer: 'a',
      explain:
        '入队仍是「先左后右」，每层第一个就是最左——左视图。这是「BFS + size 控层」模板的优势：改一个判断条件就能切换需求。',
      tags: ['data-structure'],
    },
    {
      id: 'right-side-view.q9',
      prompt: 'DFS 解法的空间复杂度（不计输出）？',
      options: [
        { id: 'a', text: 'O(h)，h 为树高（递归栈）' },
        { id: 'b', text: 'O(w)，w 为最宽层' },
        { id: 'c', text: 'O(1)' },
        { id: 'd', text: 'O(n²)' },
      ],
      answer: 'a',
      explain:
        '只有递归栈消耗，深度 = 树高。BFS 反之是 O(w)。瘦高树用 DFS、矮宽树用 BFS，是空间选型的常识。',
      tags: ['complexity'],
    },
    {
      id: 'right-side-view.q10',
      prompt: '为什么 BFS 解法在 `if i == size - 1` 时才 append，而不是每个节点都 append？',
      options: [
        { id: 'a', text: '只有当前层最右那个值得记录；其他节点会被「右视图」遮挡' },
        { id: 'b', text: '加速' },
        { id: 'c', text: 'append 会抛错' },
        { id: 'd', text: '没区别' },
      ],
      answer: 'a',
      explain:
        '题目语义：站在右侧看到的是每层「最右」一个；其它会被它遮住。如果都收集就成了层序遍历（#102）。',
      tags: ['naming', 'invariant'],
    },
    {
      id: 'right-side-view.q11',
      prompt: '把 DFS 解法的「先右后左」改成「先左后右」会怎样？',
      options: [
        { id: 'a', text: '会得到「左视图」——每层第一次访问到的是最左节点' },
        { id: 'b', text: '不变' },
        { id: 'c', text: '抛错' },
        { id: 'd', text: '更慢' },
      ],
      answer: 'a',
      explain:
        '与 BFS 的入队顺序对偶——先访问哪边，「第一次见到」就是哪边。这是 DFS 的「方向感」是通过递归顺序传递的。',
      tags: ['invariant'],
    },
    {
      id: 'right-side-view.q12',
      prompt: '本题最坏的「时间 × 空间」复杂度组合？',
      options: [
        { id: 'a', text: '时间 O(n)，空间最坏 O(n)（满二叉树最后一层 ≈ n/2 节点）' },
        { id: 'b', text: 'O(n log n) / O(log n)' },
        { id: 'c', text: 'O(n²) / O(1)' },
        { id: 'd', text: 'O(n) / O(1)' },
      ],
      answer: 'a',
      explain:
        '时间永远 O(n)；空间 BFS 是 O(w)、DFS 是 O(h)。两者最坏都是 O(n)（一种是宽树、一种是链状）。',
      tags: ['complexity'],
    },
  ],
}

export default problem
