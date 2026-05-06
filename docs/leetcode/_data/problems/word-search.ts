import type { Problem } from '../types'

const code = `def exist(board: list[list[str]], word: str) -> bool:
    m, n = len(board), len(board[0])

    def dfs(i: int, j: int, k: int) -> bool:
        if k == len(word):
            return True
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[k]:
            return False
        tmp = board[i][j]
        board[i][j] = '#'                # 占位，防止本路径再次走回来
        found = (dfs(i+1, j, k+1) or dfs(i-1, j, k+1)
                 or dfs(i, j+1, k+1) or dfs(i, j-1, k+1))
        board[i][j] = tmp                # 回溯还原
        return found

    for i in range(m):
        for j in range(n):
            if dfs(i, j, 0):
                return True
    return False`

export const problem: Problem = {
  id: 'word-search',
  leetcodeNo: 79,
  title: { zh: '单词搜索', en: 'Word Search' },
  difficulty: 'medium',
  pattern: 'backtracking',
  tags: ['matrix', 'backtracking', 'array'],
  statement:
    '给定一个 `m x n` 二维字符网格 `board` 和一个字符串单词 `word`。如果 `word` 存在于网格中，返回 `true`；否则，返回 `false`。\n\n单词必须按照字母顺序，通过**相邻的单元格**内的字母构成，其中「相邻」单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母**不允许被重复使用**。',
  examples: [
    {
      input: 'board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]\nword = "ABCCED"',
      output: 'true',
    },
    {
      input: 'board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]\nword = "SEE"',
      output: 'true',
    },
    {
      input: 'board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]\nword = "ABCB"',
      output: 'false',
    },
  ],
  constraints: [
    'm == board.length, n == board[i].length',
    '1 ≤ m, n ≤ 6',
    '1 ≤ word.length ≤ 15',
    'board 与 word 仅由大小写英文字母组成',
  ],
  intuition:
    '从每个格子出发 DFS：当前字符等于 word[k] 才继续往四方向走，递归 word[k+1]。「占位 + 还原」模板：进 dfs 把 board[i][j] 改成 #、出 dfs 还原原字符——既能防止重复使用同格、又不需要额外 visited 数组。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(m·n·4^L)，L 为 word 长度', space: 'O(L) 递归栈' },
  microQuestions: [
    {
      id: 'word-search.q1',
      prompt: '从每个格子尝试一次 DFS，外层循环的目的？',
      codeContext: code,
      highlightLine: 17,
      options: [
        { id: 'a', text: '把每个格子都当作 word 的可能起点' },
        { id: 'b', text: '初始化 visited' },
        { id: 'c', text: '排序' },
        { id: 'd', text: '只是为了打印' },
      ],
      answer: 'a',
      explain:
        'word 可以从任何位置开始，所以外层 m·n 个起点逐个尝试。一旦命中立刻 return True 跳出。',
      tags: ['invariant'],
    },
    {
      id: 'word-search.q2',
      prompt: '`board[i][j] = "#"` 这一句的本质是？',
      codeContext: code,
      highlightLine: 9,
      options: [
        { id: 'a', text: '占位 / 临时标记本格已用，回溯到这里时再还原原字符' },
        { id: 'b', text: '永久销毁该格' },
        { id: 'c', text: '调试用，可以省' },
        { id: 'd', text: '把它换成其他字符的随机值' },
      ],
      answer: 'a',
      explain:
        '这是网格回溯的标准技巧——用一个不可能在 word 里出现的占位符（`#`）暂时占住格子，递归出来再用 `tmp` 还原。比 visited 二维数组更省。',
      tags: ['invariant'],
    },
    {
      id: 'word-search.q3',
      prompt: '为什么必须把 `tmp = board[i][j]` 提前保存？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: '占位前先存原字符，回溯时才能恢复' },
        { id: 'b', text: '为了打印' },
        { id: 'c', text: '没必要存，固定用 "A" 还原即可' },
        { id: 'd', text: 'Python 强制要求' },
      ],
      answer: 'a',
      explain:
        '不同格子原本是不同字母，必须用临时变量保存一下；不能假设全是 "A"。这是一个非常常见的写错点。',
      tags: ['boundary'],
    },
    {
      id: 'word-search.q4',
      prompt: '`if k == len(word): return True` 这行作用？',
      codeContext: code,
      highlightLine: 5,
      options: [
        { id: 'a', text: '匹配完 word 全部字符——成功' },
        { id: 'b', text: '失败终止' },
        { id: 'c', text: '占位检查' },
        { id: 'd', text: '没用' },
      ],
      answer: 'a',
      explain:
        'k 是「当前应当匹配 word 的第几位」。当 k == len(word) 表示 word[0..L-1] 都已匹配过去，return True 即胜。',
      tags: ['boundary'],
    },
    {
      id: 'word-search.q5',
      prompt: '`board[i][j] != word[k]` 时立即 return False 是什么思想？',
      codeContext: code,
      highlightLine: 7,
      options: [
        { id: 'a', text: '前缀剪枝——当前字符不匹配，整条 DFS 路径不可能拼出 word' },
        { id: 'b', text: '宽搜' },
        { id: 'c', text: '随便加的' },
        { id: 'd', text: '为了打印' },
      ],
      answer: 'a',
      explain:
        '回溯里最重要的剪枝就是「不合法立即停」。这一行让 DFS 在错误起点上瞬间返回，不浪费 4 次递归调用。',
      tags: ['complexity', 'invariant'],
    },
    {
      id: 'word-search.q6',
      prompt: '四方向扩展 `(i+1,j),(i-1,j),(i,j+1),(i,j-1)` 也常被写成什么 Pythonic 形式？',
      options: [
        { id: 'a', text: 'for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):' },
        { id: 'b', text: 'while True 循环' },
        { id: 'c', text: 'set 内推导' },
        { id: 'd', text: '只有一种写法' },
      ],
      answer: 'a',
      explain:
        '`dirs = [(1,0),(-1,0),(0,1),(0,-1)]` 是网格题的标配。可读性更好、且方便改 8 方向（加四角）。',
      tags: ['pythonism'],
    },
    {
      id: 'word-search.q7',
      prompt: '出 dfs 时的 `board[i][j] = tmp` 若被忘记，会发生什么？',
      codeContext: code,
      highlightLine: 12,
      options: [
        { id: 'a', text: '当前路径回退后这个格子永远是 #；下一条起点再扫到这里就匹配不上 → 漏解' },
        { id: 'b', text: '没问题' },
        { id: 'c', text: '会 IndexError' },
        { id: 'd', text: '会无限递归' },
      ],
      answer: 'a',
      explain:
        '回溯三段式必须配对：进 dfs 改、出 dfs 还原。少一边就会污染整片棋盘，下一条搜索路径全错。',
      tags: ['invariant', 'boundary'],
    },
    {
      id: 'word-search.q8',
      prompt: 'word="SEE" 在示例 board 中能命中——大致路径是？',
      options: [
        { id: 'a', text: '从 (1,3) 的 S 起 → (2,3) E → (2,2) E（回溯过程中需占位避免回头）' },
        { id: 'b', text: '一定从 (0,0) 开始' },
        { id: 'c', text: '不可能命中' },
        { id: 'd', text: '需要排序' },
      ],
      answer: 'a',
      explain:
        '画路径是 word search 类题的硬功夫——能在脑子里走出 SEE 的轨迹，就基本理解了占位回溯。',
      tags: ['invariant'],
    },
    {
      id: 'word-search.q9',
      prompt: '本题最坏时间复杂度?',
      options: [
        { id: 'a', text: 'O(m·n)' },
        { id: 'b', text: 'O(m·n·4^L)，L 为 word 长度' },
        { id: 'c', text: 'O(L²)' },
        { id: 'd', text: 'O(m·n·L)' },
      ],
      answer: 'b',
      explain:
        '每个格子作为起点 m·n；每层 DFS 至多 4 个方向、深度上限 L。第一步固定字符后实际是 3 个方向，但渐近写 4^L 即可。',
      tags: ['complexity'],
    },
    {
      id: 'word-search.q10',
      prompt: '空间复杂度是？',
      options: [
        { id: 'a', text: 'O(L)，主要是递归栈深度上限 L（占位法不需要额外 visited）' },
        { id: 'b', text: 'O(m·n)' },
        { id: 'c', text: 'O(L²)' },
        { id: 'd', text: 'O(1)' },
      ],
      answer: 'a',
      explain:
        '占位法的好处之一就是省一份 m×n 的 visited 数组；递归栈深度由 word 长度 L 决定。',
      tags: ['complexity'],
    },
    {
      id: 'word-search.q11',
      prompt: '若同时要搜大量 word 列表（→ LC212），怎么改进效率？',
      options: [
        { id: 'a', text: '把 words 灌进 Trie，从每个格子出发 DFS——前缀共享，剪枝威力倍增' },
        { id: 'b', text: '对每个 word 各跑一遍 LC79' },
        { id: 'c', text: '用 BFS' },
        { id: 'd', text: '不可能优化' },
      ],
      answer: 'a',
      explain:
        '逐题搜会重复扫格子 O(W·m·n·4^L)；Trie + DFS 让所有公共前缀只走一次，是 LC212 的核心思想。',
      tags: ['data-structure'],
    },
  ],
}

export default problem
