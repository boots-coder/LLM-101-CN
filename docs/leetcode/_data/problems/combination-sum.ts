import type { Problem } from '../types'

const code = `def combinationSum(candidates: list[int], target: int) -> list[list[int]]:
    candidates.sort()                  # 排序后可以提前剪枝（剩余 target 不够时 break）
    res, path = [], []

    def backtrack(start: int, remain: int) -> None:
        if remain == 0:
            res.append(path[:])
            return
        for i in range(start, len(candidates)):
            x = candidates[i]
            if x > remain:             # 排序后越往后越大——直接 break 而非 continue
                break
            path.append(x)
            backtrack(i, remain - x)   # 注意是 i 而不是 i+1：同一个数允许重复使用
            path.pop()

    backtrack(0, target)
    return res`

export const problem: Problem = {
  id: 'combination-sum',
  leetcodeNo: 39,
  title: { zh: '组合总和', en: 'Combination Sum' },
  difficulty: 'medium',
  pattern: 'backtracking',
  tags: ['array', 'backtracking', 'recursion'],
  statement:
    '给你一个**无重复元素**的整数数组 `candidates` 和一个目标整数 `target`，找出 `candidates` 中可以使数字和为目标数 `target` 的**所有不同组合**，并以列表形式返回。你可以按**任意顺序**返回这些组合。\n\n`candidates` 中的同一个数字可以**无限制**重复被选取。如果至少一个数字的被选数量不同，则两种组合是不同的。\n\n对于给定的输入，保证和为 `target` 的不同组合数少于 `150` 个。',
  examples: [
    { input: 'candidates = [2,3,6,7], target = 7', output: '[[2,2,3],[7]]' },
    { input: 'candidates = [2,3,5], target = 8', output: '[[2,2,2,2],[2,3,3],[3,5]]' },
    { input: 'candidates = [2], target = 1', output: '[]' },
  ],
  constraints: [
    '1 ≤ candidates.length ≤ 30',
    '2 ≤ candidates[i] ≤ 40',
    'candidates 的所有元素互不相同',
    '1 ≤ target ≤ 40',
  ],
  intuition:
    '回溯 + start 参数防顺序重复。核心两点：① 用 `start` 参数（每层从 i 开始而非 i+1）允许同元素重复；② 排序 + `if x > remain: break` 把越往后越大的尾巴一刀切掉。如果不要求同元素重复，下一层应当是 i+1。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(N^(T/M))，N 为 candidates 数、T 为 target、M 为最小值（决策树高度上界 T/M）', space: 'O(T/M) 递归栈与 path' },
  microQuestions: [
    {
      id: 'combination-sum.q1',
      prompt: '为什么递归调用是 `backtrack(i, ...)` 而不是 `backtrack(i+1, ...)`？',
      codeContext: code,
      highlightLine: 13,
      options: [
        { id: 'a', text: '题目允许同一个数字无限次重复被选——所以下一层从 i 开始（不是 i+1）' },
        { id: 'b', text: '只是写错了，i+1 也对' },
        { id: 'c', text: '为了保持顺序' },
        { id: 'd', text: '为了避免越界' },
      ],
      answer: 'a',
      explain:
        '这是组合总和的招牌细节：`i` vs `i+1` 决定了「这个数能否重复用」。LC40（组合总和 II）就只能用一次 → 下一层 i+1；LC39 允许重复 → 下一层 i。',
      tags: ['invariant'],
    },
    {
      id: 'combination-sum.q2',
      prompt: '为什么用 `start` 参数而不是 `used` 数组？',
      options: [
        { id: 'a', text: '组合不在意顺序——靠 start 锁定挑选顺序天然避免 [2,3] 与 [3,2] 都进结果' },
        { id: 'b', text: 'start 是错误写法' },
        { id: 'c', text: 'used 写不出来' },
        { id: 'd', text: 'start 更省空间' },
      ],
      answer: 'a',
      explain:
        '排列 vs 组合的灵魂分水岭：排列要求每个顺序都算一组解 → used；组合不在意顺序 → start。理解这一点，几乎所有回溯题都能秒辨用哪种。',
      tags: ['invariant'],
    },
    {
      id: 'combination-sum.q3',
      prompt: '`candidates.sort()` + `if x > remain: break` 的剪枝意义？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: '排序后从小到大遍历——一旦当前数已经大于剩余 target，后面只会更大，直接 break 收尾' },
        { id: 'b', text: '不需要排序也行' },
        { id: 'c', text: '只是为了输出有序' },
        { id: 'd', text: '保证答案非空' },
      ],
      answer: 'a',
      explain:
        '不排序就只能写 continue（不能 break），剪枝威力大幅缩水。排序成本 O(N log N) 一次性付出，远小于剪枝带来的搜索量节省。',
      tags: ['complexity'],
    },
    {
      id: 'combination-sum.q4',
      prompt: '终止条件应当是？',
      codeContext: code,
      highlightLine: 6,
      options: [
        { id: 'a', text: 'remain == 0：恰好凑齐 target' },
        { id: 'b', text: 'remain < 0：超出 target' },
        { id: 'c', text: 'len(path) == target' },
        { id: 'd', text: '没有终止条件' },
      ],
      answer: 'a',
      explain:
        '`remain == 0` 表示路径上数字之和恰好等于 target，收集 path[:] 进 res。`remain < 0` 已经被剪枝（x > remain 时 break）阻止了。',
      tags: ['boundary'],
    },
    {
      id: 'combination-sum.q5',
      prompt: '`res.append(path[:])` 的核心意义？',
      codeContext: code,
      highlightLine: 7,
      options: [
        { id: 'a', text: '拷贝当前 path 的快照——后续 pop 不影响已收集的结果' },
        { id: 'b', text: 'path[:] 是 path 的索引片段' },
        { id: 'c', text: '随便写的，没必要' },
        { id: 'd', text: '深拷贝 path' },
      ],
      answer: 'a',
      explain:
        '回溯过程中 path 被反复 append/pop，是同一个引用。不拷贝就把这个会变的引用塞进 res，最后所有元素都指向最终空 list。',
      tags: ['pythonism'],
    },
    {
      id: 'combination-sum.q6',
      prompt: '`for i in range(start, len(candidates))` 的 start 决定了什么？',
      codeContext: code,
      highlightLine: 9,
      options: [
        { id: 'a', text: '本层只能挑「之前选过的下标 ≥ 上一次的位置」——避免组合重排相同元素' },
        { id: 'b', text: 'start 没用' },
        { id: 'c', text: 'start 控制递归深度' },
        { id: 'd', text: '加快 Python 执行速度' },
      ],
      answer: 'a',
      explain:
        '没有 start，组合 [2,3] 和 [3,2] 都会被算出来，结果重复。start 是「单调挑选」的体现：每一层只能从上次的位置或之后开始挑。',
      tags: ['invariant'],
    },
    {
      id: 'combination-sum.q7',
      prompt: '若 candidates 中存在重复元素（即变成 LC40），需要做哪两点改动？',
      options: [
        { id: 'a', text: '① 排序后在同层 for 里跳过 `i > start and candidates[i] == candidates[i-1]`；② 下一层递归改为 `i+1`（每个元素只用一次）' },
        { id: 'b', text: '只需把答案转 set 去重' },
        { id: 'c', text: '不可能解' },
        { id: 'd', text: '把 candidates.sort() 删掉' },
      ],
      answer: 'a',
      explain:
        'LC40 的标准修改两件事：同层去重（同值的兄弟跳过）+ 递归 i+1（每个位置只用一次）。两个 case 容易混淆：「同层」与「跨层」的差别要分清。',
      tags: ['boundary'],
    },
    {
      id: 'combination-sum.q8',
      prompt: 'combinationSum([2,3,6,7], 7) 的两个解中，[2,2,3] 是怎么搜出来的？',
      options: [
        { id: 'a', text: '从下标 0 开始 → 2 → remain=5；同样下标 0 → 2 → remain=3；下标 1 → 3 → remain=0 → 收解' },
        { id: 'b', text: '一次性把 [2,2,3] 拼起来' },
        { id: 'c', text: '用动态规划' },
        { id: 'd', text: '排序后二分' },
      ],
      answer: 'a',
      explain:
        '画出这条决策路径就能看到 backtrack(0,7)→backtrack(0,5)→backtrack(0,3)→backtrack(1,1)→走 3 越过 break。理解这条路径就懂了「为什么下一层是 i 不是 i+1」。',
      tags: ['invariant'],
    },
    {
      id: 'combination-sum.q9',
      prompt: '本题的复杂度上界以哪个因素为主？',
      options: [
        { id: 'a', text: '决策树深度上界 T/M（M 为最小元素），节点数最坏 N^(T/M)' },
        { id: 'b', text: 'O(N!)' },
        { id: 'c', text: 'O(2^N)' },
        { id: 'd', text: 'O(N²)' },
      ],
      answer: 'a',
      explain:
        '决策树深度由「最小元素能凑多少次」决定 ≈ T/M；每层最多 N 个分支。题目保证组合数 < 150 所以实际很小，但分析时要会写这个上界。',
      tags: ['complexity'],
    },
    {
      id: 'combination-sum.q10',
      prompt: '`if x > remain: break` 写成 `continue` 是否仍然正确？',
      codeContext: code,
      highlightLine: 10,
      options: [
        { id: 'a', text: '正确但慢——排序之后后面的元素只会更大，可以 break 提前结束' },
        { id: 'b', text: '错——会少解' },
        { id: 'c', text: '错——会多解' },
        { id: 'd', text: '完全等价' },
      ],
      answer: 'a',
      explain:
        '不影响正确性，只影响效率。排序后从小到大扫，后面元素只会更大，所以一旦超出剩余预算就可以一刀切。如果没排序，必须 continue。',
      tags: ['complexity'],
    },
    {
      id: 'combination-sum.q11',
      prompt: '空间复杂度（不计 res）是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(T/M) — 递归栈与 path 长度都不超过这个值' },
        { id: 'c', text: 'O(N²)' },
        { id: 'd', text: 'O(N!)' },
      ],
      answer: 'b',
      explain:
        '递归栈深度上界就是路径长度 ≈ T/M（最小元素能凑多少次）。res 输出按惯例不计。',
      tags: ['complexity'],
    },
  ],
}

export default problem
