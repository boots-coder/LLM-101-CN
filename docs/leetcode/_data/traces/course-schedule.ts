import type { AlgoTrace, TraceStep } from './types'

// vizKind = 'array-board'
// 主数组 values：固定展示课程编号 0..n-1（labels）
// secondary：动态展示「当前 indeg 数组」
// kv：邻接表 g（u → 下游 v 列表）
// cellState：状态着色（in-queue / done / 高亮）
function generateSteps(numCourses: number, prerequisites: number[][]): TraceStep[] {
  const steps: TraceStep[] = []
  const g: number[][] = Array.from({ length: numCourses }, () => [])
  const indeg: number[] = Array(numCourses).fill(0)
  const queue: number[] = []
  let finished = 0
  // status: 'pending' | 'in-queue' | 'done'
  const status: string[] = Array(numCourses).fill('pending')

  function snap(line: number, note: string, opts: any = {}): TraceStep {
    const cellState: Record<number, string> = {}
    const secCellCls: Record<number, string> = {}
    for (let i = 0; i < numCourses; i++) {
      if (status[i] === 'done') {
        cellState[i] = 'visited'
        secCellCls[i] = 'visited'
      } else if (status[i] === 'in-queue') {
        cellState[i] = 'in-window'
        secCellCls[i] = 'in-window'
      }
    }
    if (opts.highlight !== undefined) {
      cellState[opts.highlight] = 'found'
      secCellCls[opts.highlight] = 'found'
    }
    const pointers: any[] = []
    if (opts.u !== undefined) pointers.push({ name: 'u', idx: opts.u, color: '#22c55e' })
    if (opts.v !== undefined) pointers.push({ name: 'v', idx: opts.v, color: '#3b82f6' })

    return {
      line,
      note,
      state: {
        pointers,
        scalars: [
          { label: 'numCourses', value: String(numCourses) },
          { label: 'finished', value: String(finished) },
          { label: 'queue', value: queue.length ? `[${queue.join(',')}]` : '[]' },
        ],
        secondary: {
          label: 'indeg（入度数组）',
          values: [...indeg],
          cellCls: secCellCls,
        },
        kv: {
          title: '邻接表 g（课程 → 下游课程列表）',
          emptyLabel: '空',
          entries: g.map((adj, i) => ({
            key: i,
            value: adj.length ? `[${adj.join(',')}]` : '[]',
            highlight: opts.u === i,
          })),
        },
        cellState,
      },
    }
  }

  steps.push(snap(4, '初始化邻接表 g = [[],[],[],[]]，入度数组 indeg = [0,0,0,0]'))

  // 建图
  for (const [a, b] of prerequisites) {
    g[b].push(a)
    indeg[a] += 1
    steps.push(
      snap(
        7,
        `读边 [${a},${b}]：要学 ${a} 必须先学 ${b}，加边 ${b}→${a}，indeg[${a}] = ${indeg[a]}`,
        { highlight: a },
      ),
    )
  }

  // 初始化队列
  for (let i = 0; i < numCourses; i++) {
    if (indeg[i] === 0) {
      queue.push(i)
      status[i] = 'in-queue'
    }
  }
  steps.push(
    snap(
      11,
      `把所有 indeg=0 的课入队 → queue=[${queue.join(',')}]（这些课无先修，可直接学）`,
    ),
  )

  steps.push(snap(12, '初始化 finished = 0'))

  // BFS
  while (queue.length) {
    steps.push(snap(13, `while：queue 非空（${queue.length} 个），继续`))
    const u = queue.shift()!
    status[u] = 'done'
    steps.push(snap(14, `u = q.popleft() → u = ${u}（标记为已完成）`, { u }))
    finished += 1
    steps.push(snap(15, `finished += 1 → finished = ${finished}`, { u }))

    if (g[u].length === 0) {
      steps.push(snap(16, `g[${u}] = []，没有下游课程，跳过 for`, { u }))
    } else {
      for (const v of g[u]) {
        steps.push(snap(16, `遍历 ${u} 的下游：v = ${v}`, { u, v }))
        indeg[v] -= 1
        steps.push(
          snap(
            17,
            `indeg[${v}] -= 1 → indeg[${v}] = ${indeg[v]}（${v} 的一门先修被满足）`,
            { u, v, highlight: v },
          ),
        )
        if (indeg[v] === 0) {
          queue.push(v)
          status[v] = 'in-queue'
          steps.push(
            snap(
              19,
              `indeg[${v}] 归零！把 ${v} 入队 → queue=[${queue.join(',')}]`,
              { u, v, highlight: v },
            ),
          )
        } else {
          steps.push(
            snap(18, `indeg[${v}] = ${indeg[v]} ≠ 0，不入队`, { u, v }),
          )
        }
      }
    }
  }

  steps.push(
    snap(
      20,
      `while 退出（queue 空）。finished=${finished}, numCourses=${numCourses} → return ${finished === numCourses}`,
    ),
  )
  return steps
}

const numCourses = 4
const prerequisites: number[][] = [
  [1, 0],
  [2, 0],
  [3, 1],
  [3, 2],
]

const trace: AlgoTrace = {
  title: `示例：numCourses = ${numCourses}, prerequisites = ${JSON.stringify(prerequisites)}`,
  inputLabel: `numCourses=${numCourses}, prereq=${JSON.stringify(prerequisites)}`,
  vizKind: 'array-board',
  vizConfig: {
    values: Array.from({ length: numCourses }, (_, i) => `课${i}`),
    label: '课程编号',
  },
  steps: generateSteps(numCourses, prerequisites),
}

export default [trace]
