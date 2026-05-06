import type { Problem } from '../types'

const code = `class Trie:
    def __init__(self):
        self.children = {}     # 当前节点的子节点：「字符」 -> 「子 Trie」
        self.is_end = False    # 当前节点是否是某个完整单词的结尾

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            node = node.children.setdefault(ch, Trie())
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self._walk(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        return self._walk(prefix) is not None

    def _walk(self, s: str):
        node = self
        for ch in s:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node`

export const problem: Problem = {
  id: 'implement-trie',
  leetcodeNo: 208,
  title: { zh: '实现 Trie（前缀树）', en: 'Implement Trie (Prefix Tree)' },
  difficulty: 'medium',
  pattern: 'trie',
  tags: ['trie', 'design', 'hash-table', 'string'],
  statement:
    'Trie（发音类似 "try"），又称**前缀树**，是一种树形数据结构，用于高效存储和检索字符串数据集中的键。它有许多应用，例如自动补全和拼写检查。\n\n请你实现 `Trie` 类：\n\n- `Trie()` 初始化前缀树对象。\n- `void insert(String word)` 向前缀树中插入字符串 `word`。\n- `boolean search(String word)` 如果字符串 `word` 在前缀树中，返回 `true`，否则返回 `false`。\n- `boolean startsWith(String prefix)` 如果之前已经插入的字符串 `word` 的前缀之一为 `prefix`，返回 `true`，否则返回 `false`。',
  examples: [
    {
      input: 'Trie trie = new Trie();\ntrie.insert("apple");\ntrie.search("apple");   // True\ntrie.search("app");     // False\ntrie.startsWith("app"); // True\ntrie.insert("app");\ntrie.search("app");     // True',
      output: '[null, null, true, false, true, null, true]',
    },
  ],
  constraints: [
    '1 ≤ word.length, prefix.length ≤ 2000',
    'word 和 prefix 仅由小写英文字母组成',
    'insert、search、startsWith 调用次数总计不超过 3 × 10⁴ 次',
  ],
  intuition:
    '把每个字符当作一条边，节点记录「是否到此为止恰好是一个完整单词」。insert 沿路新建缺失的子节点；search 走完后还要校验 is_end；startsWith 只要走得通即可。三种操作都是 O(L)，与字典里词条数无关。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'insert/search/startsWith 均为 O(L)，L 为字符串长度', space: 'O(Σ N L)，N 为单词数、Σ 为字符集大小（最坏每条边都是新节点）' },
  microQuestions: [
    {
      id: 'implement-trie.q1',
      prompt: 'Trie 节点最常见的两种「子节点容器」实现是？',
      options: [
        { id: 'a', text: 'list 顺序追加 + 线性查找' },
        { id: 'b', text: 'dict（字符 → 子节点）或固定大小的 26 元素数组' },
        { id: 'c', text: '红黑树' },
        { id: 'd', text: 'set，只存「字符」' },
      ],
      answer: 'b',
      explain:
        'dict 写起来最通用、字符集大或稀疏时省空间；固定 26 数组（小写英文字母场景）查询略快、缓存友好。两者各有取舍，是面试常见对比点。',
      tags: ['data-structure'],
    },
    {
      id: 'implement-trie.q2',
      prompt: 'children 用 dict 还是 26 数组，主要权衡是？',
      options: [
        { id: 'a', text: 'dict 更慢但更省空间；数组更快但稀疏时浪费内存' },
        { id: 'b', text: 'dict 更慢更费空间；数组完胜' },
        { id: 'c', text: 'dict 更快更省空间；数组完败' },
        { id: 'd', text: '完全没有区别' },
      ],
      answer: 'a',
      explain:
        'dict 按需开槽，对中文/大字符集/稀疏树更省空间；26 数组连每个空格也要预留，但下标查找无哈希开销。本题字符集只有小写英文 26 字母，两者都常见。',
      tags: ['complexity', 'data-structure'],
    },
    {
      id: 'implement-trie.q3',
      prompt: '`node.children.setdefault(ch, Trie())` 等价于哪段手写代码？',
      codeContext: code,
      highlightLine: 9,
      options: [
        { id: 'a', text: 'if ch in node.children: del node.children[ch]\nnode.children[ch] = Trie()' },
        { id: 'b', text: 'if ch not in node.children: node.children[ch] = Trie()\nnode = node.children[ch]' },
        { id: 'c', text: 'node.children[ch] = Trie()' },
        { id: 'd', text: 'node = Trie()' },
      ],
      answer: 'b',
      explain:
        '`setdefault(k, default)` 的语义是「k 不在则放入 default，然后返回 d[k]」。它把「按需创建子节点」这种常见模式压成一行；选项 c 会粗暴覆盖已有子树。',
      tags: ['pythonism'],
    },
    {
      id: 'implement-trie.q4',
      prompt: '为什么必须有 `is_end` 这个字段，而不能仅靠「节点存在」就判断 search？',
      codeContext: code,
      highlightLine: 4,
      options: [
        { id: 'a', text: '为了支持中文字符' },
        { id: 'b', text: '区分「prefix 走得通」与「正好是一个完整单词」——比如 insert("apple") 后 search("app") 必须 False' },
        { id: 'c', text: '`is_end` 只是为了打印好看，可以省' },
        { id: 'd', text: 'Python 里没有它会报错' },
      ],
      answer: 'b',
      explain:
        '没有 is_end 就无法区分「app 是某个更长单词的前缀」与「app 自己就是一个单词」。这正是 search 与 startsWith 的本质差异。',
      tags: ['invariant'],
    },
    {
      id: 'implement-trie.q5',
      prompt: 'search 与 startsWith 的差异在哪一行体现？',
      codeContext: code,
      highlightLine: 13,
      options: [
        { id: 'a', text: 'search 多了 `node is not None and node.is_end` 校验' },
        { id: 'b', text: '没有差异，两者实现完全相同' },
        { id: 'c', text: 'startsWith 还要再走一次回溯' },
        { id: 'd', text: '差异在 insert 那里' },
      ],
      answer: 'a',
      explain:
        '两者共用「能不能走通」的 _walk 助手；search 在走通之后再额外要求 is_end == True，startsWith 走通即可。把公共逻辑抽到 _walk 是面试加分写法。',
      tags: ['invariant'],
    },
    {
      id: 'implement-trie.q6',
      prompt: 'insert("apple") 之后内部结构最接近？',
      options: [
        { id: 'a', text: '一条 a→p→p→l→e 的链，末尾 is_end=True，其余节点 is_end=False' },
        { id: 'b', text: '一颗扁平 dict，key 是 "apple"' },
        { id: 'c', text: '一个长度 5 的 list' },
        { id: 'd', text: '只有一个根节点，is_end=True' },
      ],
      answer: 'a',
      explain:
        '每个字符是一条边、对应一个新建的 Trie 节点。is_end 只在最后一个节点为 True，沿途节点都是 False（除非另有更短的单词正好结束在那里）。',
      tags: ['data-structure', 'invariant'],
    },
    {
      id: 'implement-trie.q7',
      prompt: 'insert 一个长度为 L 的单词，时间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(L)' },
        { id: 'c', text: 'O(N) — 与字典里已有词数 N 相关' },
        { id: 'd', text: 'O(NL)' },
      ],
      answer: 'b',
      explain:
        'Trie 的核心优势：操作成本只与单词长度 L 有关，与字典里已有多少词无关。这就是它在自动补全场景比直接哈希所有前缀更聪明的地方。',
      tags: ['complexity'],
    },
    {
      id: 'implement-trie.q8',
      prompt: '_walk 在中途遇到 `ch not in node.children` 时返回？',
      codeContext: code,
      highlightLine: 19,
      options: [
        { id: 'a', text: '抛 KeyError' },
        { id: 'b', text: '返回 None，让上层决定 search/startsWith 都失败' },
        { id: 'c', text: '返回当前 node' },
        { id: 'd', text: '继续递归' },
      ],
      answer: 'b',
      explain:
        '走不通就返回 None，相当于「这一段前缀根本不存在」。search 和 startsWith 在外层用 `node is not None` 判断这种情况——两个公共操作复用一个 helper。',
      tags: ['boundary'],
    },
    {
      id: 'implement-trie.q9',
      prompt: '若把 `self.children = {}` 换成 `self.children = [None]*26`，怎么改 insert 才正确？',
      options: [
        { id: 'a', text: '不用改' },
        { id: 'b', text: '把 ch 换算成下标 `idx = ord(ch) - ord("a")`，然后 `if children[idx] is None: children[idx] = Trie()`' },
        { id: 'c', text: '直接 `children[ch] = Trie()`' },
        { id: 'd', text: '改用 set' },
      ],
      answer: 'b',
      explain:
        '数组实现的关键就是字符到下标的换算 `ord(ch) - ord("a")`。这个换算只在「字符集固定且小」的场景才划算；中文等大字符集仍以 dict 为准。',
      tags: ['pythonism', 'data-structure'],
    },
    {
      id: 'implement-trie.q10',
      prompt: 'Trie 与「直接把所有前缀塞进 set」相比，主要优势是？',
      options: [
        { id: 'a', text: '更省空间——共享前缀只存一份' },
        { id: 'b', text: '前缀查询更快（其实都是 O(L)）' },
        { id: 'c', text: '更容易写' },
        { id: 'd', text: '没有任何优势' },
      ],
      answer: 'a',
      explain:
        '把每个前缀都单独塞 set，N 个长度 L 的单词最坏要 O(NL²) 个字符串副本；Trie 共用前缀，最坏 O(NL) 个节点。当 N 大时差距巨大。',
      tags: ['complexity', 'data-structure'],
    },
    {
      id: 'implement-trie.q11',
      prompt: '若题目要求支持「带通配符 . 的 search」（LeetCode 211），最自然的扩展是？',
      options: [
        { id: 'a', text: '把 _walk 改成 DFS——遇到 . 就遍历当前节点的所有 children' },
        { id: 'b', text: '改用哈希表存所有前缀' },
        { id: 'c', text: 'Trie 不支持通配符，必须换算法' },
        { id: 'd', text: '用 BFS 层层扩展' },
      ],
      answer: 'a',
      explain:
        '通配符 . 意味着「这一位任意字符都行」——把线性的 _walk 改成「遇 . 时枚举所有 children 递归」即可。这正是 LC211 与 LC208 的核心差异。',
      tags: ['data-structure'],
    },
    {
      id: 'implement-trie.q12',
      prompt: '插入 N 个长度均为 L 的单词，最坏空间复杂度是？',
      options: [
        { id: 'a', text: 'O(L)' },
        { id: 'b', text: 'O(N)' },
        { id: 'c', text: 'O(N L Σ)，Σ 为字符集大小（dict/数组实现下每节点的子槽位）' },
        { id: 'd', text: 'O(N L)' },
      ],
      answer: 'c',
      explain:
        '严格上界：N 个单词最坏没有共享前缀 → N·L 个节点；每个节点的 children 容器（数组实现）占 Σ。dict 实现下因为按需开槽通常远低于此上界。',
      tags: ['complexity'],
    },
  ],
}

export default problem
