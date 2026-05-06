import type { Problem } from '../types'

const code = `from collections import deque

def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    g = [[] for _ in range(numCourses)]
    indeg = [0] * numCourses
    for a, b in prerequisites:
        # b -> a：要学 a 必须先学 b
        g[b].append(a)
        indeg[a] += 1

    q = deque(i for i in range(numCourses) if indeg[i] == 0)
    finished = 0
    while q:
        u = q.popleft()
        finished += 1
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return finished == numCourses`

export const problem: Problem = {
  id: 'course-schedule',
  leetcodeNo: 207,
  title: { zh: '课程表', en: 'Course Schedule' },
  difficulty: 'medium',
  pattern: 'graph',
  tags: ['graph', 'topological-sort', 'bfs', 'dfs'],
  statement:
    '你这个学期必须选修 `numCourses` 门课程，记为 `0` 到 `numCourses - 1`。\n\n在选修某些课程之前需要一些先修课程。先修课程按数组 `prerequisites` 给出，其中 `prerequisites[i] = [ai, bi]`，表示如果要学习课程 `ai` 则**必须**先学习课程 `bi`。\n\n请你判断**是否可能**完成所有课程的学习？如果可以，返回 `true`；否则，返回 `false`。',
  examples: [
    { input: 'numCourses = 2, prerequisites = [[1,0]]', output: 'true', note: '先修 0 再学 1，无环' },
    { input: 'numCourses = 2, prerequisites = [[1,0],[0,1]]', output: 'false', note: '0 和 1 互为先修——成环，不可能完成' },
  ],
  constraints: [
    '1 ≤ numCourses ≤ 2000',
    '0 ≤ prerequisites.length ≤ 5000',
    'prerequisites[i].length == 2',
    '0 ≤ ai, bi < numCourses',
    'ai != bi',
    '所有 [ai, bi] 互不相同',
  ],
  intuition:
    '把题目翻译成图：边 b → a 代表「a 依赖 b」。问题等价于「这张有向图是否无环」。Kahn BFS 拓扑排序：把所有入度为 0 的点入队（无依赖的课）；每次取一个点 u，对它指向的 v 入度 -1，如果 v 入度变 0 就入队。最终若所有点都被处理过则无环。',
  language: 'python',
  solutionCode: code,
  complexity: { time: 'O(V+E)', space: 'O(V+E)' },
  microQuestions: [
    {
      id: 'course-schedule.q1',
      prompt: '本题最自然的算法骨架是？',
      options: [
        { id: 'a', text: '动态规划' },
        { id: 'b', text: '拓扑排序（Kahn BFS 或 DFS 三色标记）——「是否能完成」等价于「图是否无环」' },
        { id: 'c', text: '二分查找' },
        { id: 'd', text: '滑动窗口' },
      ],
      answer: 'b',
      explain:
        '「先修关系」就是有向边、「能否完成」等价于「图是否无环」——这是标准的拓扑排序题型。Kahn BFS 与 DFS 三色都能解。',
      tags: ['data-structure'],
    },
    {
      id: 'course-schedule.q2',
      prompt: '`prerequisites = [[a, b]]` 应该建哪个方向的边？',
      codeContext: code,
      highlightLine: 7,
      options: [
        { id: 'a', text: 'a → b' },
        { id: 'b', text: 'b → a：要学 a 必须先学 b，所以 b 是 a 的「上游」' },
        { id: 'c', text: '无所谓' },
        { id: 'd', text: '双向' },
      ],
      answer: 'b',
      explain:
        '约定：拓扑排序里边 u → v 表示「u 必须在 v 之前」，即依赖方向。本题 b 是 a 的先修课，所以 b → a。建错方向就跑出反向拓扑序，结果错。',
      tags: ['invariant'],
    },
    {
      id: 'course-schedule.q3',
      prompt: '入度数组 `indeg[a] += 1` 表达的语义是？',
      codeContext: code,
      highlightLine: 8,
      options: [
        { id: 'a', text: 'a 还有几门没修完的先修课' },
        { id: 'b', text: 'a 已经完成' },
        { id: 'c', text: 'a 是入口课程' },
        { id: 'd', text: 'a 的下游课程数' },
      ],
      answer: 'a',
      explain:
        '入度 = 还有几条上游边没消化。入度归零意味着「先修全部满足」可以学了。',
      tags: ['invariant'],
    },
    {
      id: 'course-schedule.q4',
      prompt: '初始队列为什么是「所有入度为 0 的点」？',
      codeContext: code,
      highlightLine: 11,
      options: [
        { id: 'a', text: '它们没有任何先修——可以直接修，是拓扑序的起点' },
        { id: 'b', text: '它们是叶子' },
        { id: 'c', text: '随便选' },
        { id: 'd', text: '为了让队列非空' },
      ],
      answer: 'a',
      explain:
        '没有上游依赖的课就是「现在就能学」的课。这一队列在迭代过程中会不断补入「先修刚被满足」的新点。',
      tags: ['invariant'],
    },
    {
      id: 'course-schedule.q5',
      prompt: '当我们处理完节点 u，对它的下游 v 做什么？',
      codeContext: code,
      highlightLine: 17,
      options: [
        { id: 'a', text: 'indeg[v] -= 1；若变 0 入队（一个先修课已修完，v 还差几门也减一）' },
        { id: 'b', text: '直接入队 v' },
        { id: 'c', text: '把 v 删除' },
        { id: 'd', text: '什么都不做' },
      ],
      answer: 'a',
      explain:
        '消化一条边 u → v 就让 v 的入度 -1，等价于「v 的某门先修课刚被满足」。只有当 v 全部先修都满足（入度归 0）才轮到它。',
      tags: ['invariant'],
    },
    {
      id: 'course-schedule.q6',
      prompt: '为什么用 `deque` 而不是 list？',
      codeContext: code,
      highlightLine: 1,
      options: [
        { id: 'a', text: 'deque 的 popleft 是 O(1)；list.pop(0) 是 O(n)' },
        { id: 'b', text: '一样' },
        { id: 'c', text: 'deque 不能 append' },
        { id: 'd', text: 'list 不能用作 BFS' },
      ],
      answer: 'a',
      explain:
        '这是 Python BFS 的硬常识。`from collections import deque` + `popleft()` 是 BFS 的标配；用 list.pop(0) 会让算法退化成 O(VE)。',
      tags: ['pythonism', 'complexity'],
    },
    {
      id: 'course-schedule.q7',
      prompt: '若 `finished < numCourses`，意味着？',
      codeContext: code,
      highlightLine: 20,
      options: [
        { id: 'a', text: '剩下的节点必然在某个有向环上——没有任何节点能让它的入度归零' },
        { id: 'b', text: '题目数据有误' },
        { id: 'c', text: 'BFS 写错了' },
        { id: 'd', text: '有重复边' },
      ],
      answer: 'a',
      explain:
        '环上的每个点都至少有 1 条来自环内的入边，永远不会归零，所以不会被弹出。「未处理完的节点 = 环上节点」是拓扑判环的核心结论。',
      tags: ['invariant'],
    },
    {
      id: 'course-schedule.q8',
      prompt: 'DFS 三色标记法判环的「三色」分别是？',
      options: [
        { id: 'a', text: '白（未访问）、灰（在当前 DFS 路径上）、黑（已彻底完成）；遇到灰边 → 有环' },
        { id: 'b', text: '红、绿、蓝' },
        { id: 'c', text: '0、1、2 一样的概念' },
        { id: 'd', text: '只用两色就够了' },
      ],
      answer: 'a',
      explain:
        '三色法是 DFS 判环的经典写法。「灰」代表当前递归路径上的节点；如果 DFS 又走回到了灰节点，说明形成了一个返回当前路径的环。两色（visited/unvisited）无法区分「跨子树访问」与「真正的环」。',
      tags: ['data-structure', 'invariant'],
    },
    {
      id: 'course-schedule.q9',
      prompt: 'Kahn BFS 与 DFS 三色法在功能上的区别？',
      options: [
        { id: 'a', text: '都是 O(V+E)、都能判环；Kahn 自然给出拓扑序、并行思考更直观；DFS 实现更短但需要小心栈深' },
        { id: 'b', text: 'Kahn 更慢' },
        { id: 'c', text: '只有 Kahn 能判环' },
        { id: 'd', text: '只有 DFS 能给拓扑序' },
      ],
      answer: 'a',
      explain:
        '两者复杂度相同，差异在表达：Kahn 直观、能边判环边出拓扑序；DFS 三色更紧凑但栈深较深时需 sys.setrecursionlimit。',
      tags: ['complexity'],
    },
    {
      id: 'course-schedule.q10',
      prompt: '本题时间复杂度是？',
      options: [
        { id: 'a', text: 'O(V+E)' },
        { id: 'b', text: 'O(V·E)' },
        { id: 'c', text: 'O(V²)' },
        { id: 'd', text: 'O(E²)' },
      ],
      answer: 'a',
      explain:
        '每个点入队/出队各 1 次（V）、每条边被检查 1 次（E），合起来 O(V+E)。',
      tags: ['complexity'],
    },
    {
      id: 'course-schedule.q11',
      prompt: '空间复杂度是？',
      options: [
        { id: 'a', text: 'O(1)' },
        { id: 'b', text: 'O(V+E) — 邻接表 + 入度数组 + 队列' },
        { id: 'c', text: 'O(V²)' },
        { id: 'd', text: 'O(E²)' },
      ],
      answer: 'b',
      explain:
        '邻接表 O(V+E)、入度数组 O(V)、队列 O(V)，合起来 O(V+E)。',
      tags: ['complexity'],
    },
    {
      id: 'course-schedule.q12',
      prompt: '若题目改为「LC210 返回拓扑序本身」，需要怎么改？',
      options: [
        { id: 'a', text: '把每次 popleft 出的 u 追加到 order 列表；最后若 len(order) == numCourses 返回 order，否则返回 []' },
        { id: 'b', text: '完全换算法' },
        { id: 'c', text: '不可能扩展' },
        { id: 'd', text: '改用 DFS' },
      ],
      answer: 'a',
      explain:
        'Kahn BFS 自然就是按拓扑序弹出节点的——记下弹出顺序就是拓扑序。这就是 LC207 → LC210 的最小改动。',
      tags: ['data-structure'],
    },
  ],
}

export default problem
