import type { Problem } from '../types'

const code = `def findWords(board: list[list[str]], words: list[str]) -> list[str]:
    # 1. 把所有 words 灌进 Trie，叶子节点存「完整单词」便于命中时直接收集
    root = {}
    for w in words:
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node['$'] = w   # 用 '$' 标记一个完整单词，值就是单词本身

    m, n = len(board), len(board[0])
    res = []

    def dfs(i: int, j: int, node: dict) -> None:
        ch = board[i][j]
        if ch not in node:
            return
        nxt = node[ch]
        if '$' in nxt:
            res.append(nxt['$'])
            del nxt['$']        # 已收集，避免重复加入
        board[i][j] = '#'       # 占位，防止本路径再次走回来
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ni, nj = i+di, j+dj
            if 0 <= ni < m and 0 <= nj < n and board[ni][nj] != '#':
                dfs(ni, nj, nxt)
        board[i][j] = ch        # 回溯还原
        # 可选剪枝：如果 nxt 已经空了，从父节点裁掉这条死路
        if not nxt:
            node.pop(ch, None)

    for i in range(m):
        for j in range(n):
            dfs(i, j, root)
    return res`

export const problem: Problem = {
  id: 'word-search-ii',
  leetcodeNo: 212,
  title: { zh: '单词搜索 II', en: 'Word Search II' },
  difficulty: 'hard',
  pattern: 'trie',
  tags: ['trie', 'backtracking', 'matrix', 'array'],
  statement:
    '给定一个 `m x n` 二维字符网格 `board` 和一个单词（字符串）列表 `words`，**返回所有二维网格上的单词**。\n\n单词必须按照字母顺序，通过**相邻的单元格**内的字母构成，其中「相邻」单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母在一个单词中**不允许被重复使用**。',
  examples: [
    {
      input: 'board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]\nwords = ["oath","pea","eat","rain"]',
      output: '["eat","oath"]',
    },
    {
      input: 'board = [["a","b"],["c","d"]]\nwords = ["abcb"]',
      output: '[]',
    },
  ],
  constraints: [
    'm == board.length, n == board[i].length',
    '1 ≤ m, n ≤ 12',
    '1 ≤ words.length ≤ 3 × 10⁴',
    '1 ≤ words[i].length ≤ 10',
  ],
  intuition:
    '一题一搜会重复扫格子。把所有 words 灌进 Trie，再从每个格子出发 DFS：当前字符必须在 Trie 当前节点的 children 里，否则立刻回头。命中 `$` 标记就把单词收入答案。「占位 # + 出 dfs 还原」是网格回溯三段式的标配。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(m·n·4^L)，L 为最长单词长度（Trie 大幅剪枝实际远小于此）', space: 'O(总字符数) 用于 Trie，递归栈 O(L)' },
  microQuestions: [
    {
      id: 'word-search-ii.q1',
      prompt: '为什么把所有 words 灌进 Trie，比逐个 word 调一遍 LC79 快？',
      options: [
        { id: 'a', text: '因为 Trie 共享前缀——多个单词的公共前缀只搜一次' },
        { id: 'b', text: '只是写起来短，复杂度其实一样' },
        { id: 'c', text: 'Trie 自动并行' },
        { id: 'd', text: 'Trie 是 O(1) 的算法' },
      ],
      answer: 'a',
      explain:
        '逐题搜：对每个 word，从每个格子重新跑一次 DFS，前缀完全不复用。Trie：所有 word 的公共前缀只在 DFS 中走一次，「前缀走得通就继续，走不通立即剪枝」。',
      tags: ['complexity'],
    },
    {
      id: 'word-search-ii.q2',
      prompt: '在 Trie 末端用什么方式标记「这里恰好是某个完整单词」最好？',
      codeContext: code,
      highlightLine: 7,
      options: [
        { id: 'a', text: '用 `is_end=True` 然后再回头拼路径' },
        { id: 'b', text: '直接在叶子节点存完整单词字符串（如 `node["$"] = w`），命中时一步取出' },
        { id: 'c', text: '完全不需要标记' },
        { id: 'd', text: '用一个全局 set 记录所有单词' },
      ],
      answer: 'b',
      explain:
        '标记 + 存字符串这一招省掉「DFS 时还要传一个 path 字符串」，常数小、代码短；命中时直接 `res.append(node["$"])` 即可。',
      tags: ['data-structure'],
    },
    {
      id: 'word-search-ii.q3',
      prompt: '收集到一个单词后，为什么要 `del nxt["$"]`？',
      codeContext: code,
      highlightLine: 18,
      options: [
        { id: 'a', text: '为了节省空间' },
        { id: 'b', text: '避免同一个单词在不同路径下被加进答案多次' },
        { id: 'c', text: '让 Trie 自动萎缩' },
        { id: 'd', text: '没有意义' },
      ],
      answer: 'b',
      explain:
        '若不删除，例如 board 中有两条路径都能拼出 "eat"，那么 res 会出现两次 "eat"。题目要求不重复，所以一旦命中立即抹除标记。',
      tags: ['boundary'],
    },
    {
      id: 'word-search-ii.q4',
      prompt: '`board[i][j] = "#"` 这一步在做什么？',
      codeContext: code,
      highlightLine: 19,
      options: [
        { id: 'a', text: '占位 / 临时标记本格已用，递归回来后再还原' },
        { id: 'b', text: '永久删除该格' },
        { id: 'c', text: '把它换成 Trie 的根节点' },
        { id: 'd', text: '调试用，可以省掉' },
      ],
      answer: 'a',
      explain:
        '回溯三段式：进 dfs 占位 → 递归 → 出 dfs 还原。这样既能 O(1) 检查「是否已被本路径用过」，又不需要额外 visited 数组。出 dfs 必须配套 `board[i][j] = ch`。',
      tags: ['invariant'],
    },
    {
      id: 'word-search-ii.q5',
      prompt: '在 dfs 出口处 `board[i][j] = ch` 这一句的作用？',
      codeContext: code,
      highlightLine: 24,
      options: [
        { id: 'a', text: '把临时占位 # 还原成原字符——没有它，外层迭代会污染棋盘' },
        { id: 'b', text: '调试日志' },
        { id: 'c', text: '没用，可以删' },
        { id: 'd', text: '把节点标记为已访问' },
      ],
      answer: 'a',
      explain:
        '这是回溯模板的「撤销选择」步。少了它，下一条路径在跑到原本是字母的格子时只能看到 #，搜索全错。',
      tags: ['invariant'],
    },
    {
      id: 'word-search-ii.q6',
      prompt: '走完一个完整单词（即命中 `$`）后，是否需要 return（停止本路径）？',
      options: [
        { id: 'a', text: '需要——走到单词末尾就停下来，免得再深入' },
        { id: 'b', text: '不需要——这条 DFS 路径还可能延伸出更长的另一个单词（如 "app" 是 "apple" 的前缀）' },
        { id: 'c', text: '取决于题目' },
        { id: 'd', text: '一定不能继续，否则结果错' },
      ],
      answer: 'b',
      explain:
        '关键陷阱：words 里可能同时有 "app" 与 "apple"，DFS 不能在 "app" 处提前收手。只把单词收进答案、`$` 抹掉，然后继续往下走。',
      tags: ['boundary', 'invariant'],
    },
    {
      id: 'word-search-ii.q7',
      prompt: '`if not nxt: node.pop(ch, None)` 这条剪枝意味着什么？',
      codeContext: code,
      highlightLine: 27,
      options: [
        { id: 'a', text: 'nxt 已经被掏空——这条 Trie 分支再也不会命中任何单词，从父节点裁掉以加速后续搜索' },
        { id: 'b', text: '回溯失败' },
        { id: 'c', text: '内存泄漏' },
        { id: 'd', text: '把单词从 res 中删除' },
      ],
      answer: 'a',
      explain:
        '随着 `$` 不断被删，分支会变成空 dict——保留它没意义但每次还要查。从父节点 pop 掉是显著的常数加速；省掉这一步不会错，只是慢。',
      tags: ['complexity'],
    },
    {
      id: 'word-search-ii.q8',
      prompt: '检查 `if ch not in node: return` 放在 dfs 入口，对应 Trie 的什么思想？',
      codeContext: code,
      highlightLine: 14,
      options: [
        { id: 'a', text: '前缀剪枝——只要当前字符不是任何剩余单词的下一个字符，立即停' },
        { id: 'b', text: '占位回溯' },
        { id: 'c', text: '懒加载' },
        { id: 'd', text: '随便写的' },
      ],
      answer: 'a',
      explain:
        '这一步的剪枝威力巨大：board 上 12×12=144 个起点，普通 DFS 要走到死胡同才会停；Trie 让我们「字符不在字典中就立刻停」，常数缩小 N 倍。',
      tags: ['complexity'],
    },
    {
      id: 'word-search-ii.q9',
      prompt: '本题最坏时间复杂度的常见表达是？',
      options: [
        { id: 'a', text: 'O(m·n)' },
        { id: 'b', text: 'O(m·n·4^L)，L 为最长单词长度（Trie 大幅剪枝下实际远小于此）' },
        { id: 'c', text: 'O(words.length²)' },
        { id: 'd', text: 'O(2^(m·n))' },
      ],
      answer: 'b',
      explain:
        '从每个格子（m·n）出发做四方向 DFS，深度上限 L，最坏每层 4 选 1 → 4^L。Trie 剪枝让真实情况一般远小于此理论上界。',
      tags: ['complexity'],
    },
    {
      id: 'word-search-ii.q10',
      prompt: 'Trie 用普通 class 写还是用嵌套 dict 写？',
      options: [
        { id: 'a', text: '都行——dict 更短更快、class 更易扩展（如加 is_end、计数等字段）' },
        { id: 'b', text: '只能用 class' },
        { id: 'c', text: '只能用 dict' },
        { id: 'd', text: 'dict 在 Python 里慢' },
      ],
      answer: 'a',
      explain:
        '本题是高频「速写」场景，嵌套 dict 行数最少；面试时若有人问「假如要扩展 freq、删除单词」，再升级为 class 即可。',
      tags: ['pythonism'],
    },
    {
      id: 'word-search-ii.q11',
      prompt: '为什么用 `board[i][j] = "#"` 而不是 visited 二维数组？',
      options: [
        { id: 'a', text: '占位法 O(1) 空间复用、不用额外开 m×n 数组；缺点是改了输入（题目允许）' },
        { id: 'b', text: 'visited 是错的' },
        { id: 'c', text: 'Python 不支持二维数组' },
        { id: 'd', text: '占位法更慢' },
      ],
      answer: 'a',
      explain:
        '两种都能 AC。占位法省一份 visited 数组、常数更小，缺点是污染输入；面试时建议补一句「最后还原 board」。',
      tags: ['data-structure'],
    },
  ],
}

export default problem
