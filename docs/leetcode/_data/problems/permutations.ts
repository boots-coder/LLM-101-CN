import type { Problem } from '../types'

const code = `def permute(nums: list[int]) -> list[list[int]]:
    n = len(nums)
    res, path = [], []
    used = [False] * n

    def backtrack():
        if len(path) == n:
            res.append(path[:])      # 必须拷贝快照，否则后续 pop 会改写
            return
        for i in range(n):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack()
            path.pop()               # 撤销
            used[i] = False

    backtrack()
    return res`

export const problem: Problem = {
  id: 'permutations',
  leetcodeNo: 46,
  title: { zh: '全排列', en: 'Permutations' },
  difficulty: 'medium',
  pattern: 'backtracking',
  tags: ['array', 'backtracking', 'recursion'],
  statement:
    '给定一个不含重复数字的数组 `nums`，返回其**所有可能的全排列**。你可以**按任意顺序**返回答案。',
  examples: [
    { input: 'nums = [1,2,3]', output: '[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]' },
    { input: 'nums = [0,1]', output: '[[0,1],[1,0]]' },
    { input: 'nums = [1]', output: '[[1]]' },
  ],
  constraints: [
    '1 ≤ nums.length ≤ 6',
    '-10 ≤ nums[i] ≤ 10',
    'nums 中的所有整数互不相同',
  ],
  intuition:
    '在决策树上 DFS。每一层选择一个「未被使用过」的数加入 path；递归到 len(path) == n 就把 path 拷贝进 res；回溯前 pop 并把 used 标记还原。「used 数组」是排列题与组合题最大的写法分水岭——排列允许任意顺序所以靠 used，组合靠 start 防顺序重复。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(n · n!)，每个排列长度 n、共 n! 个', space: 'O(n) 递归栈与 path / used，不计输出' },
  microQuestions: [
    {
      id: 'permutations.q1',
      prompt: '回溯模板的三步是？',
      options: [
        { id: 'a', text: '排序 → 二分 → 拼接' },
        { id: 'b', text: '做选择 → 递归到下一层 → 撤销选择' },
        { id: 'c', text: '初始化 → 调用 → 返回' },
        { id: 'd', text: '插入 → 删除 → 修改' },
      ],
      answer: 'b',
      explain:
        '三段式雷打不动：选择 → 递归 → 撤销。代码上对应 `path.append → backtrack() → path.pop()` 这一组对称操作。',
      tags: ['invariant'],
    },
    {
      id: 'permutations.q2',
      prompt: '`res.append(path[:])` 中的 `path[:]` 是在做什么？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: '语法噪音，可以省成 `res.append(path)`' },
        { id: 'b', text: '浅拷贝出一个新 list——否则后续 pop 会改写已收集的快照' },
        { id: 'c', text: '深拷贝' },
        { id: 'd', text: '把 path 转 tuple' },
      ],
      answer: 'b',
      explain:
        '`path` 是同一个 list 对象在递归过程中反复修改；不拷贝就把同一个引用塞进 res，最后所有元素都指向同一个空 list。等价写法：`list(path)`。',
      tags: ['pythonism', 'invariant'],
    },
    {
      id: 'permutations.q3',
      prompt: '`used = [False]*n` 的作用是？',
      codeContext: code,
      highlightLine: 4,
      options: [
        { id: 'a', text: '记录每个下标当前是否已经在 path 中——避免「同位置元素被用两次」' },
        { id: 'b', text: '存放结果' },
        { id: 'c', text: '记录当前层数' },
        { id: 'd', text: '存放排序后的下标' },
      ],
      answer: 'a',
      explain:
        '排列题允许任意顺序，所以不能像组合题那样靠 start 排除回头。必须显式记录「这个位置我用过没」，回溯时再翻回 False。',
      tags: ['data-structure', 'invariant'],
    },
    {
      id: 'permutations.q4',
      prompt: '为什么排列题用 `used`，而组合题（如 LC39）用 `start` 参数？',
      options: [
        { id: 'a', text: '排列要求所有顺序都是不同结果，所以每层都从 0 开始挑「没用过的」；组合不在意顺序，只需「下一层从 i 开始」即可天然避免顺序重复' },
        { id: 'b', text: '只是风格不同，可以互换' },
        { id: 'c', text: '组合也必须用 used' },
        { id: 'd', text: '排列也必须用 start' },
      ],
      answer: 'a',
      explain:
        '这是排列 vs 组合的核心思维差异。[1,2] 与 [2,1] 在排列里是两个不同结果，所以每层都要全扫；在组合里同一组合，必须靠 start 锁定挑选顺序。',
      tags: ['invariant'],
    },
    {
      id: 'permutations.q5',
      prompt: '回溯到 `len(path) == n` 时为什么要 return？',
      codeContext: code,
      highlightLine: 9,
      options: [
        { id: 'a', text: '叶子节点收集后必须及时返回，否则 for 循环还会继续 append 越界' },
        { id: 'b', text: '只是为了好看' },
        { id: 'c', text: 'Python 强制要求' },
        { id: 'd', text: '会抛 StackOverflow' },
      ],
      answer: 'a',
      explain:
        '收齐 n 个数后路径已满，再继续会越过 n 个元素的限制（虽然 used 也会拦住，但语义上「叶子节点收集 + return」是回溯模板的标志）。',
      tags: ['boundary'],
    },
    {
      id: 'permutations.q6',
      prompt: '`path.pop()` 与 `used[i] = False` 的顺序是否必须严格？',
      codeContext: code,
      highlightLine: 16,
      options: [
        { id: 'a', text: '本题里 path 与 used 互不依赖，先后均可；但「都要撤销」缺一不可' },
        { id: 'b', text: '必须 used 先撤' },
        { id: 'c', text: '必须 pop 先撤' },
        { id: 'd', text: '只撤其一就行' },
      ],
      answer: 'a',
      explain:
        '关键是「都撤销」，而不是「顺序」。少撤任何一边，下一轮 i 的搜索都会读到「污染」状态。',
      tags: ['invariant'],
    },
    {
      id: 'permutations.q7',
      prompt: '为什么不写 `path.append(nums[i]); used[i] = True; backtrack(); used[i] = False; path.pop()`，把 used 移到第二步可不可以？',
      options: [
        { id: 'a', text: '这两行（path.append 与 used[i]=True）顺序无所谓，只要都在 backtrack 之前；面试常写法是先标 used 再 append 以「先占位再记录路径」' },
        { id: 'b', text: '不行，必须先 path.append' },
        { id: 'c', text: '不行，必须先 used' },
        { id: 'd', text: '会出 bug' },
      ],
      answer: 'a',
      explain:
        '两行只要都在 `backtrack()` 之前、都在出 dfs 处对称撤销即可。常见风格略有不同，重要的是保持配对。',
      tags: ['naming'],
    },
    {
      id: 'permutations.q8',
      prompt: 'permute([1,2,3]) 共有多少种排列？',
      options: [
        { id: 'a', text: '3' },
        { id: 'b', text: '6 — 3!' },
        { id: 'c', text: '8 — 2³' },
        { id: 'd', text: '9' },
      ],
      answer: 'b',
      explain:
        'n 个互不相同的数全排列个数为 n!。3! = 6 是肉眼可数的小例子，调试时常用。',
      tags: ['complexity'],
    },
    {
      id: 'permutations.q9',
      prompt: '时间复杂度的「严格」表达是？',
      options: [
        { id: 'a', text: 'O(n²)' },
        { id: 'b', text: 'O(n!)' },
        { id: 'c', text: 'O(n · n!) — 共 n! 个排列，每个长度 n 拷贝进 res 也要 O(n)' },
        { id: 'd', text: 'O(2ⁿ)' },
      ],
      answer: 'c',
      explain:
        '一般习惯说「O(n!)」，但严格写时要算上叶子节点把长度 n 的 path 拷贝进 res 的成本——O(n · n!)。',
      tags: ['complexity'],
    },
    {
      id: 'permutations.q10',
      prompt: '若数组里有重复（LC47），最常见的去重技巧是？',
      options: [
        { id: 'a', text: '排序 + `if i>0 and nums[i]==nums[i-1] and not used[i-1]: continue`——同一层跳过相同值的「后兄弟」' },
        { id: 'b', text: '把结果转 set 去重' },
        { id: 'c', text: '不可能去重' },
        { id: 'd', text: '改用 BFS' },
      ],
      answer: 'a',
      explain:
        'LC47 的标准技巧：先排序让相同值相邻，然后在选择时只允许「相同值的最左未用」进入，跳过其它。`not used[i-1]` 是细节关键，写错就会少解。',
      tags: ['boundary'],
    },
    {
      id: 'permutations.q11',
      prompt: '空间复杂度（不计输出 res）是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(n) — 递归栈深度 n + path/used 各占 n' },
        { id: 'c', text: 'O(n²)' },
        { id: 'd', text: 'O(n!)' },
      ],
      answer: 'b',
      explain:
        '递归深度上限 n（每层 path 长度 +1）；path 与 used 都是 O(n) 的辅助空间。res 本身按惯例不计。',
      tags: ['complexity'],
    },
    {
      id: 'permutations.q12',
      prompt: '如果用「在数组里 swap 两端」的写法（无 used、就地交换）相比 used 版的差别？',
      options: [
        { id: 'a', text: 'swap 版省 used 数组、空间更小，但破坏 nums 顺序、写不熟容易错；used 版可读性更好' },
        { id: 'b', text: '完全等价' },
        { id: 'c', text: 'swap 版更慢' },
        { id: 'd', text: 'swap 版结果错' },
      ],
      answer: 'a',
      explain:
        '「swap 法」是经典优化，O(1) 额外空间；但写法绕、易错。面试推荐 used 版打底，能讲清 swap 法是加分项。',
      tags: ['data-structure'],
    },
  ],
}

export default problem
