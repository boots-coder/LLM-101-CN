import type { Problem } from '../types'

const code = `from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

class Codec:
    def serialize(self, root: TreeNode | None) -> str:
        # 前序 DFS：空节点显式记成 「#」
        parts = []
        def dfs(node: TreeNode | None) -> None:
            if not node:
                parts.append('#')
                return
            parts.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        return ','.join(parts)

    def deserialize(self, data: str) -> TreeNode | None:
        # 用 deque 让 popleft 是 O(1)；前序消费序列
        tokens = deque(data.split(','))
        def build() -> TreeNode | None:
            tok = tokens.popleft()
            if tok == '#':
                return None
            node = TreeNode(int(tok))
            node.left = build()
            node.right = build()
            return node
        return build()`

export const problem: Problem = {
  id: 'serialize-deserialize-tree',
  leetcodeNo: 297,
  title: { zh: '二叉树的序列化与反序列化', en: 'Serialize and Deserialize Binary Tree' },
  difficulty: 'hard',
  pattern: 'tree-dfs',
  tags: ['tree', 'dfs', 'design', 'serialization'],
  statement:
    '序列化是将一个数据结构或对象转换为连续的比特位的操作，进而可以将转换后的数据存储在一个文件或者内存中，同时也可以通过网络传输到另一个计算机环境，采取相反方式重构得到原数据。\n\n请设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。',
  examples: [
    { input: 'root = [1,2,3,null,null,4,5]', output: '[1,2,3,null,null,4,5]', note: '前序：1,2,#,#,3,4,#,#,5,#,#' },
    { input: 'root = []', output: '[]' },
    { input: 'root = [1]', output: '[1]' },
  ],
  constraints: [
    '树中节点数目范围是 [0, 10⁴]',
    '-1000 ≤ Node.val ≤ 1000',
  ],
  intuition:
    '前序遍历 + 把空节点显式写成 「#」 即可双射：前序天然定义了「先根、再左、再右」的顺序，加上空哨兵后任何一棵二叉树都唯一对应一个字符串。反序列化时按同样顺序消费 tokens，遇到 「#」 返回 None，否则建节点 + 递归建左右。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n)', space: 'O(n)' },
  microQuestions: [
    {
      id: 'serialize-deserialize-tree.q1',
      prompt: '为什么单独的「前序」或「中序」无法反序列化二叉树，但加上 「#」 哨兵就可以？',
      options: [
        { id: 'a', text: '前序+中序需要两份才能唯一确定（且都不含重复值）；前序+「#」 把空位置显式记下，等价于前序+「下一节点是左还是右」' },
        { id: 'b', text: '前序本身就够' },
        { id: 'c', text: '中序就够' },
        { id: 'd', text: '只能用层序' },
      ],
      answer: 'a',
      explain:
        '裸前序 [1,2,3] 既可能是「1 的左 2，2 的左 3」也可能是「1 的左 2，1 的右 3」——歧义。「#」 显式标空位置后每个分支都能确定走完了，无歧义。',
      tags: ['invariant'],
    },
    {
      id: 'serialize-deserialize-tree.q2',
      prompt: '`deserialize` 用 `deque` 而非 `list` 的原因？',
      codeContext: code,
      highlightLine: 24,
      options: [
        { id: 'a', text: 'list.pop(0) 是 O(n)；deque.popleft 是 O(1)' },
        { id: 'b', text: 'deque 支持双端' },
        { id: 'c', text: '只是习惯' },
        { id: 'd', text: 'list 不能存字符串' },
      ],
      answer: 'a',
      explain:
        '从前端取消费时 list.pop(0) 每次都要把后面元素左移一位——n 次 O(n²)。deque 是双向链表实现的双端队列，两端 O(1)。这是 Python 性能题里的常见考点。',
      tags: ['pythonism', 'complexity'],
    },
    {
      id: 'serialize-deserialize-tree.q3',
      prompt: '`build` 函数为什么不需要传 index 参数也能正确处理？',
      codeContext: code,
      highlightLine: 26,
      options: [
        { id: 'a', text: '通过闭包共享 deque——每次 popleft 都消费下一个 token，递归隐式同步了「位置」' },
        { id: 'b', text: '其实需要传' },
        { id: 'c', text: '靠全局变量' },
        { id: 'd', text: 'Python 自动维护' },
      ],
      answer: 'a',
      explain:
        'tokens 是闭包外层的引用，所有递归调用共享同一个 deque。每次 popleft 把游标向前推一格——这就是「迭代器/可消费队列」的优雅之处。如果用 list[index]，要把 index 用 nonlocal 或 list 包起来才能跨递归层共享。',
      tags: ['pythonism', 'data-structure'],
    },
    {
      id: 'serialize-deserialize-tree.q4',
      prompt: '为什么不能用 None 直接转字符串而要用 「#」？',
      options: [
        { id: 'a', text: 'str(None) = "None"，会和值为 -None 之类无关；用 「#」 是明确的「不是数字也不是合法名字」的哨兵，更安全' },
        { id: 'b', text: '语法错误' },
        { id: 'c', text: 'None 不能 split' },
        { id: 'd', text: '没区别' },
      ],
      answer: 'a',
      explain:
        '原则上 str(None) → "None" 也能用，反序列化时判断字符串等于 「None」 即可。用 「#」 是约定俗成的「占位符」，不容易和合法 val 冲突；同时长度更短。',
      tags: ['naming'],
    },
    {
      id: 'serialize-deserialize-tree.q5',
      prompt: '`serialize` 和 `deserialize` 用「同样的遍历顺序」是关键吗？',
      options: [
        { id: 'a', text: '是 —— 序列化用前序，反序列化也必须先建当前节点再建左、再建右；顺序不一致会建错' },
        { id: 'b', text: '不重要' },
        { id: 'c', text: '只要节点数对就行' },
        { id: 'd', text: '反序列化必须用 BFS' },
      ],
      answer: 'a',
      explain:
        '序列化定义了 token 的顺序；反序列化必须按同样语义消费——前序 → 「先取一个建本节点，再递归取左、再递归取右」。改顺序会把「左」当成「右」，建出来的树形态完全不同。',
      tags: ['invariant'],
    },
    {
      id: 'serialize-deserialize-tree.q6',
      prompt: '空树 root 是 None 时，serialize 输出是？',
      options: [
        { id: 'a', text: '"#"' },
        { id: 'b', text: '""' },
        { id: 'c', text: '"None"' },
        { id: 'd', text: 'null' },
      ],
      answer: 'a',
      explain:
        'dfs(None) 直接 append("#")，join 后就是 "#"。这是边界自洽的好处：不需要为空树写特判。',
      tags: ['boundary'],
    },
    {
      id: 'serialize-deserialize-tree.q7',
      prompt: 'BFS（层序）+ 显式 「#」 也能实现序列化吗？',
      options: [
        { id: 'a', text: '能 —— 这就是 LeetCode 自己显示树用的格式（[1,2,3,null,...]）' },
        { id: 'b', text: '不能' },
        { id: 'c', text: '只能 DFS' },
        { id: 'd', text: '会丢节点' },
      ],
      answer: 'a',
      explain:
        '两种方式各有优势：DFS 实现简短（递归就 5 行），BFS 与 LeetCode 输入格式一致。两种方式得到的字符串不同但都能往返。',
      tags: ['data-structure'],
    },
    {
      id: 'serialize-deserialize-tree.q8',
      prompt: '`tokens.popleft()` 在 `build` 里抛 IndexError 意味着？',
      options: [
        { id: 'a', text: '输入字符串与序列化格式不匹配——少了 token' },
        { id: 'b', text: '正常情况' },
        { id: 'c', text: '内存不足' },
        { id: 'd', text: '需要 try-except' },
      ],
      answer: 'a',
      explain:
        '如果输入是合法序列化结果，popleft 不会少。生产代码可加 try/except 处理脏输入；leetcode 上输入合法时不会发生。',
      tags: ['boundary'],
    },
    {
      id: 'serialize-deserialize-tree.q9',
      prompt: '本题时间复杂度是？',
      options: [
        { id: 'a', text: 'O(n) 每个节点访问一次（包括空哨兵 ≤ 2n+1 个 token）' },
        { id: 'b', text: 'O(n log n)' },
        { id: 'c', text: 'O(n²)' },
        { id: 'd', text: '取决于树形' },
      ],
      answer: 'a',
      explain:
        '空哨兵数量 = 空指针数 = n+1（满二叉树性质）。所以 token 总数 ≤ 2n+1，扫一遍 O(n)。',
      tags: ['complexity'],
    },
    {
      id: 'serialize-deserialize-tree.q10',
      prompt: '空间复杂度 O(n) 来自？',
      options: [
        { id: 'a', text: '输出字符串 O(n) + 递归栈 O(h)，整体 O(n)' },
        { id: 'b', text: '常数' },
        { id: 'c', text: 'O(log n)' },
        { id: 'd', text: '只是哈希表' },
      ],
      answer: 'a',
      explain:
        '字符串的字符数与节点数同级；递归栈最坏 O(n)（链状）。两者加起来 O(n)。',
      tags: ['complexity'],
    },
    {
      id: 'serialize-deserialize-tree.q11',
      prompt: '若 val 包含逗号或负号会出问题吗？',
      options: [
        { id: 'a', text: '会 —— 用 "," 做分隔符时若 val 本身含 "," 会拆错；负号 "-" 不会冲突。安全做法是换分隔符或对 val 做转义' },
        { id: 'b', text: '不会' },
        { id: 'c', text: 'Python 自动处理' },
        { id: 'd', text: '改用 JSON' },
      ],
      answer: 'a',
      explain:
        'leetcode 题目里 val 是 int，无逗号问题。但生产场景设计序列化协议时这是常见坑——可以用 JSON、自定义转义或长度前缀。',
      tags: ['boundary'],
    },
    {
      id: 'serialize-deserialize-tree.q12',
      prompt: 'Python 里 `\',\'.join(parts)` 这种链式操作的优势？',
      codeContext: code,
      highlightLine: 19,
      options: [
        { id: 'a', text: '比 `for + +=` 拼接字符串快一个量级（避免不可变字符串反复创建）' },
        { id: 'b', text: '只是写法短' },
        { id: 'c', text: '需要 import' },
        { id: 'd', text: '只能用 list' },
      ],
      answer: 'a',
      explain:
        'Python 字符串不可变，`s += part` 每次都新建对象——n 次 O(n²)。`.join` 只分配一次，O(n)。这是 Python 性能的肌肉记忆。',
      tags: ['pythonism', 'complexity'],
    },
  ],
}

export default problem
