import type { PatternMeta } from './types'

export const patterns: PatternMeta[] = [
  {
    slug: 'hashmap',
    name: { zh: '哈希', en: 'Hash Map' },
    oneLiner: '空间换时间，O(1) 查询是否见过 / 找配对',
    day: 1,
    problemIds: ['two-sum', 'group-anagrams', 'longest-consecutive-sequence'],
    bonusList: [
      { no: 283, title: '移动零', difficulty: 'easy' },
      { no: 41, title: '缺失的第一个正数', difficulty: 'hard' },
    ],
  },
  {
    slug: 'two-pointer',
    name: { zh: '双指针', en: 'Two Pointers' },
    oneLiner: '有序或对撞，从两端向中间收缩',
    day: 1,
    problemIds: ['three-sum', 'container-with-most-water', 'trapping-rain-water'],
    bonusList: [
      { no: 283, title: '移动零', difficulty: 'easy' },
    ],
  },
  {
    slug: 'sliding-window',
    name: { zh: '滑动窗口', en: 'Sliding Window' },
    oneLiner: '维护一个区间，单向扩张/收缩',
    day: 1,
    problemIds: ['longest-substring-without-repeating', 'find-anagrams', 'min-window-substring'],
    bonusList: [
      { no: 560, title: '和为 K 的子数组', difficulty: 'medium' },
      { no: 239, title: '滑动窗口最大值', difficulty: 'hard' },
    ],
  },
  {
    slug: 'binary-search',
    name: { zh: '二分查找', en: 'Binary Search' },
    oneLiner: '单调即可二分，关键是边界',
    day: 2,
    problemIds: ['search-rotated-array', 'find-peak-element', 'median-two-sorted-arrays'],
    bonusList: [],
  },
  {
    slug: 'monotonic-stack',
    name: { zh: '单调栈', en: 'Monotonic Stack' },
    oneLiner: '维持单调求"下一个更大/更小"',
    day: 2,
    problemIds: ['daily-temperatures', 'largest-rectangle-histogram', 'trapping-rain-water'],
    bonusList: [],
  },
  {
    slug: 'heap',
    name: { zh: '堆', en: 'Heap / Priority Queue' },
    oneLiner: 'Top-K 与流式中位数',
    day: 2,
    problemIds: ['kth-largest', 'top-k-frequent', 'find-median-stream'],
    bonusList: [],
  },
  {
    slug: 'linked-list',
    name: { zh: '链表', en: 'Linked List' },
    oneLiner: '哨兵节点 + 双指针',
    day: 3,
    problemIds: ['reverse-linked-list', 'linked-list-cycle-ii', 'merge-k-lists', 'lru-cache'],
    bonusList: [],
  },
  {
    slug: 'tree-dfs',
    name: { zh: '树 DFS', en: 'Tree DFS' },
    oneLiner: '递归三态：归纳 / 分治 / 带状态',
    day: 3,
    problemIds: ['inorder-traversal', 'max-path-sum', 'lowest-common-ancestor'],
    bonusList: [],
  },
  {
    slug: 'tree-bfs',
    name: { zh: '树 BFS', en: 'Tree BFS' },
    oneLiner: '层序遍历模板',
    day: 3,
    problemIds: ['level-order', 'right-side-view', 'flatten-tree-to-list'],
    bonusList: [],
  },
  {
    slug: 'trie',
    name: { zh: 'Trie 前缀树', en: 'Trie' },
    oneLiner: '字符索引的多叉树',
    day: 4,
    problemIds: ['implement-trie', 'word-search-ii'],
    bonusList: [],
  },
  {
    slug: 'backtracking',
    name: { zh: '回溯', en: 'Backtracking' },
    oneLiner: '选择 → 递归 → 撤销',
    day: 4,
    problemIds: ['permutations', 'combination-sum', 'n-queens'],
    bonusList: [],
  },
  {
    slug: 'graph',
    name: { zh: '图搜索', en: 'Graph Search' },
    oneLiner: 'DFS / BFS / 拓扑排序',
    day: 4,
    problemIds: ['course-schedule', 'number-of-islands', 'rotting-oranges'],
    bonusList: [],
  },
  {
    slug: 'dp-1d',
    name: { zh: '一维 DP', en: 'DP 1D' },
    oneLiner: '线性状态转移',
    day: 5,
    problemIds: ['climbing-stairs', 'house-robber', 'longest-increasing-subsequence', 'word-break'],
    bonusList: [],
  },
  {
    slug: 'dp-2d',
    name: { zh: '二维 DP', en: 'DP 2D' },
    oneLiner: '网格 / 双串',
    day: 5,
    problemIds: ['unique-paths', 'longest-common-subsequence', 'edit-distance', 'longest-palindromic-substring'],
    bonusList: [],
  },
  {
    slug: 'bit-prefix',
    name: { zh: '位运算与前缀和', en: 'Bit / Prefix Sum' },
    oneLiner: '异或消除 / 累计和',
    day: 5,
    problemIds: ['single-number', 'subarray-sum-equals-k', 'product-except-self'],
    bonusList: [],
  },
]

export function getPattern(slug: string): PatternMeta | undefined {
  return patterns.find(p => p.slug === slug)
}

export function patternsByDay(day: 1 | 2 | 3 | 4 | 5): PatternMeta[] {
  return patterns.filter(p => p.day === day)
}
