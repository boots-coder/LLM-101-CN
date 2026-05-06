import type { Problem } from '../types'

const code = `class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

# 解法一：O(1) 额外空间——「左子树最右节点接右子树」技巧
def flatten(root: TreeNode | None) -> None:
    curr = root
    while curr:
        if curr.left:
            # 找当前节点左子树的「最右节点」
            rightmost = curr.left
            while rightmost.right:
                rightmost = rightmost.right
            # 把原右子树挂到「最右节点」的右边
            rightmost.right = curr.right
            # 把左子树整体搬到右
            curr.right = curr.left
            curr.left = None
        curr = curr.right          # 进入下一节点（已经是展平后的下一个）

# 解法二：递归（前序）
def flattenRec(root: TreeNode | None) -> None:
    def dfs(node: TreeNode | None) -> TreeNode | None:
        # 返回展平后的「尾节点」
        if not node:
            return None
        left_tail = dfs(node.left)
        right_tail = dfs(node.right)
        if left_tail:
            left_tail.right = node.right
            node.right = node.left
            node.left = None
        return right_tail or left_tail or node`

export const problem: Problem = {
  id: 'flatten-tree-to-list',
  leetcodeNo: 114,
  title: { zh: '二叉树展开为链表', en: 'Flatten Binary Tree to Linked List' },
  difficulty: 'medium',
  pattern: 'tree-bfs',
  tags: ['tree', 'dfs', 'in-place'],
  statement:
    '给你二叉树的根节点 `root`，请你将它**展开为一个单链表**：\n\n- 展开后的单链表应该同样使用 `TreeNode`，其中 `right` 子指针指向链表中下一个结点，而左子指针始终为 `null`\n- 展开后的单链表应该与二叉树的**前序遍历**顺序相同\n\n要求**原地**完成，进阶：用 O(1) 额外空间。',
  examples: [
    { input: 'root = [1,2,5,3,4,null,6]', output: '[1,null,2,null,3,null,4,null,5,null,6]' },
    { input: 'root = []', output: '[]' },
    { input: 'root = [0]', output: '[0]' },
  ],
  constraints: [
    '树中结点数在范围 [0, 2000]',
    '-100 ≤ Node.val ≤ 100',
  ],
  intuition:
    '主流 O(1) 空间技巧：对每个有左子树的节点，把它的**右子树整体**挂到**左子树的最右节点**之下，然后把左子树搬到右、左置空。结果整棵树就按前序「拉直」。这其实是 Morris 遍历的同款思路。也可以用前序 DFS 写，但需要 O(h) 栈空间。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(1) 迭代版' },
  microQuestions: [
    {
      id: 'flatten-tree-to-list.q1',
      prompt: '为什么这道题挂在 tree-bfs 关卡——它和 BFS 有什么关系？',
      options: [
        { id: 'a', text: '严格说不是 BFS，但「层级感 + 重排指针」的思维和 BFS 共通；同时也可以用 BFS/前序队列解' },
        { id: 'b', text: '必须用 BFS' },
        { id: 'c', text: '是经典 BFS 题' },
        { id: 'd', text: '题目要求层序' },
      ],
      answer: 'a',
      explain:
        'pattern 归类是教学上的——本题与「重新组织链接」「按前序输出」有关，BFS/DFS/迭代多解殊途同归。掌握 O(1) 空间的「左最右节点接右子」是亮点。',
      tags: ['data-structure'],
    },
    {
      id: 'flatten-tree-to-list.q2',
      prompt: '「左子树最右节点接右子树」这一招的核心思想？',
      codeContext: code,
      highlightLine: 13,
      options: [
        { id: 'a', text: '前序顺序里，左子树最后一个被访问的节点（最右）的「下一个」恰好是当前节点的右子树根——直接接上即可' },
        { id: 'b', text: '为了保持平衡' },
        { id: 'c', text: '巧合' },
        { id: 'd', text: '随便选' },
      ],
      answer: 'a',
      explain:
        '前序遍历访问当前 → 整个左子树（最后访问的是左子树最右节点）→ 整个右子树。所以把右子树拼到「左子树最右」之后正好顺延前序。',
      tags: ['invariant'],
    },
    {
      id: 'flatten-tree-to-list.q3',
      prompt: '迭代版的时间复杂度是？分析关键？',
      options: [
        { id: 'a', text: 'O(n) —— 每条边最多访问 2 次（一次找最右、一次推进 curr）' },
        { id: 'b', text: 'O(n²) —— 每个节点都要找最右' },
        { id: 'c', text: 'O(n log n)' },
        { id: 'd', text: 'O(h)' },
      ],
      answer: 'a',
      explain:
        '直觉以为「每个节点都找一次最右」会 O(n²)；其实每个节点作为「最右」只会被找到一次（之后它就被并入主链）。摊销下来每条边总共 2 次访问，O(n)。',
      tags: ['complexity'],
    },
    {
      id: 'flatten-tree-to-list.q4',
      prompt: '迭代版空间复杂度真的是 O(1) 吗？',
      options: [
        { id: 'a', text: '是 —— 只用了 curr 和 rightmost 两个常数指针，递归版才是 O(h)' },
        { id: 'b', text: '不是 O(1)' },
        { id: 'c', text: '取决于树高' },
        { id: 'd', text: '需要哈希' },
      ],
      answer: 'a',
      explain:
        '迭代版的精髓就是 O(1) 空间——这就是它能称为「最优解」的原因。递归版虽简单但栈空间 O(h)。',
      tags: ['complexity'],
    },
    {
      id: 'flatten-tree-to-list.q5',
      prompt: '展平后所有节点的 `left` 应该是？',
      codeContext: code,
      highlightLine: 18,
      options: [
        { id: 'a', text: 'None —— 题目要求左子针置空，否则不算「单链表」' },
        { id: 'b', text: '指向自己' },
        { id: 'c', text: '不变' },
        { id: 'd', text: '指向父节点' },
      ],
      answer: 'a',
      explain:
        '题目明确「左子指针始终为 null」。漏写 `curr.left = None` 会留下旧的左子指针——leetcode 测试会因此失败。',
      tags: ['boundary', 'naming'],
    },
    {
      id: 'flatten-tree-to-list.q6',
      prompt: 'curr 推进到 `curr.right` 的语义是？',
      codeContext: code,
      highlightLine: 19,
      options: [
        { id: 'a', text: '此时 curr.right 已经是「展平后的下一个节点」（因为左子树已经接到右）；继续推进就好' },
        { id: 'b', text: '回到上一层' },
        { id: 'c', text: '需要回溯' },
        { id: 'd', text: '随机走' },
      ],
      answer: 'a',
      explain:
        '本步骤之后 curr 的 right 链已经包含了前序里它后面的所有节点；推进到 right 等于「移到下一节点」。这是迭代写法不需要栈/递归的关键。',
      tags: ['invariant'],
    },
    {
      id: 'flatten-tree-to-list.q7',
      prompt: '若 curr 没有左子树，会怎样？',
      codeContext: code,
      highlightLine: 9,
      options: [
        { id: 'a', text: '跳过 if 块直接 curr = curr.right —— 已经是前序顺序，不需要处理' },
        { id: 'b', text: '需要特判' },
        { id: 'c', text: '抛错' },
        { id: 'd', text: '退出' },
      ],
      answer: 'a',
      explain:
        '左子树为空时 curr → curr.right 本身就是前序的下一个节点，无需操作。这是「if curr.left:」让代码自然处理的边界。',
      tags: ['boundary'],
    },
    {
      id: 'flatten-tree-to-list.q8',
      prompt: '另一种递归思路是「逆前序（右-左-根）」——它怎么帮我们构造？',
      options: [
        { id: 'a', text: '逆前序 = 后序的反——访问到 root 时右子树已展开；维护一个「上一次访问的节点」prev，把 root.right=prev、root.left=None 即可' },
        { id: 'b', text: '没用' },
        { id: 'c', text: '会出错' },
        { id: 'd', text: '只能 BFS' },
      ],
      answer: 'a',
      explain:
        '这是 O(n) 时间、O(h) 空间的另一种漂亮解：以「右-左-根」顺序递归，`self.prev` 总是「展平后的下一节点」；当前节点 `right=prev; left=None; prev=self`——五行解决。',
      tags: ['data-structure'],
    },
    {
      id: 'flatten-tree-to-list.q9',
      prompt: '若用「前序 BFS 把所有节点收进 list 再串起来」，会怎样？',
      options: [
        { id: 'a', text: '能 AC，O(n) 时间 O(n) 空间——但失去「原地 O(1) 空间」的题意亮点' },
        { id: 'b', text: '不能' },
        { id: 'c', text: '只是慢' },
        { id: 'd', text: '抛异常' },
      ],
      answer: 'a',
      explain:
        '直观但不优雅。题目进阶版要求 O(1) 空间，写出迭代版「左最右接右子」是面试加分。',
      tags: ['complexity'],
    },
    {
      id: 'flatten-tree-to-list.q10',
      prompt: '`while rightmost.right:` 这一行循环找的是？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: 'curr 的左子树里**最右**那个节点（即左子树前序中最后一个）' },
        { id: 'b', text: '最深节点' },
        { id: 'c', text: '随机一个' },
        { id: 'd', text: '左子树根' },
      ],
      answer: 'a',
      explain:
        '一直沿 right 走到 None 为止——找到的就是「左子树前序里最后访问的节点」。',
      tags: ['invariant'],
    },
    {
      id: 'flatten-tree-to-list.q11',
      prompt: '本题不变量是？（每次循环结束时）',
      options: [
        { id: 'a', text: 'curr 之前所有节点（沿 right 链）已经按前序展平；curr 之后的还原样' },
        { id: 'b', text: '所有节点都展平' },
        { id: 'c', text: '左子树清空' },
        { id: 'd', text: '没特别不变量' },
      ],
      answer: 'a',
      explain:
        '这就是迭代版「正确性」的归纳论证——每一步保持「前面已展平，后面待处理」，最后到 None 时整棵树展平。',
      tags: ['invariant'],
    },
    {
      id: 'flatten-tree-to-list.q12',
      prompt: '为什么这种「修改原指针」的做法在 Python 中安全？',
      options: [
        { id: 'a', text: 'Python 中 TreeNode 是引用类型，重赋值 .left/.right 仅改指针不复制对象——任何变量对节点的引用都能看见变化' },
        { id: 'b', text: '是因为 GIL' },
        { id: 'c', text: 'Python 不允许指针' },
        { id: 'd', text: '需要锁' },
      ],
      answer: 'a',
      explain:
        '理解「Python 变量是引用」是面试关键。`a.left = b` 不复制 b；外部其他指向 a 的变量也能看到 a.left 的变化。链表/树的「原地修改」都依赖这点。',
      tags: ['pythonism'],
    },
    {
      id: 'flatten-tree-to-list.q13',
      prompt: '展开顺序与哪种遍历一致？',
      options: [
        { id: 'a', text: '前序（preorder）' },
        { id: 'b', text: '中序' },
        { id: 'c', text: '后序' },
        { id: 'd', text: '层序' },
      ],
      answer: 'a',
      explain:
        '题目原文：「展开后的单链表应该与二叉树的前序遍历顺序相同」。这是为什么用「左最右接右子」会正确——前序定义就是「根→左子树→右子树」。',
      tags: ['naming'],
    },
  ],
}

export default problem
