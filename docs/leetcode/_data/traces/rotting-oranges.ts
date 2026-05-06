import type { AlgoTrace, TraceStep } from './types'

// vizKind = 'grid'
// 单元格语义：0=空（water）、1=新鲜（fresh）、2=腐烂（rotten）；当前出队的格子=current
function generateSteps(input: number[][]): TraceStep[] {
  const grid = input.map((row) => [...row])
  const m = grid.length
  const n = grid[0].length
  const steps: TraceStep[] = []
  const queue: [number, number][] = []
  let fresh = 0
  let minutes = 0
  const dirs: [number, number][] = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
  ]

  function emoji(v: number): string {
    if (v === 0) return '·'
    if (v === 1) return '🍊'
    if (v === 2) return '💀'
    return String(v)
  }

  function snap(line: number, note: string, opts: any = {}): TraceStep {
    const cellState: Record<string, string> = {}
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        if (grid[i][j] === 0) cellState[`${i},${j}`] = 'water'
        else if (grid[i][j] === 1) cellState[`${i},${j}`] = 'fresh'
        else if (grid[i][j] === 2) cellState[`${i},${j}`] = 'rotten'
      }
    }
    return {
      line,
      note,
      state: {
        grid: grid.map((row) => row.map(emoji)),
        scalars: [
          { label: 'minutes', value: String(minutes) },
          { label: 'fresh', value: String(fresh) },
          { label: 'queue.size', value: String(queue.length) },
        ],
        cellState,
        current: opts.current,
        queue: queue.map(([i, j]) => `(${i},${j})`),
      },
    }
  }

  steps.push(snap(4, `初始化 m=${m}, n=${n}, q = deque(), fresh = 0`))

  // 阶段 1：收集腐烂源 + 统计 fresh
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      if (grid[i][j] === 2) {
        queue.push([i, j])
        steps.push(
          snap(
            12,
            `(${i},${j}) 是腐烂橘子 → 入队作为多源 BFS 起点`,
            { current: [i, j] },
          ),
        )
      } else if (grid[i][j] === 1) {
        fresh += 1
        steps.push(
          snap(
            14,
            `(${i},${j}) 是新鲜橘子 → fresh += 1（fresh=${fresh}）`,
            { current: [i, j] },
          ),
        )
      }
    }
  }

  steps.push(
    snap(
      15,
      `初始扫描完成：q=[${queue.map((p) => `(${p[0]},${p[1]})`).join(',')}], fresh=${fresh}, minutes=0`,
    ),
  )

  // 阶段 2：按层扩张
  while (queue.length && fresh > 0) {
    steps.push(
      snap(
        18,
        `进入 while：q.size=${queue.length}, fresh=${fresh}（>0 才继续扩张）`,
      ),
    )
    const layerSize = queue.length
    steps.push(
      snap(19, `冻结当前层节点数 = ${layerSize}（按层扩张的 BFS 模板）`),
    )

    for (let _k = 0; _k < layerSize; _k++) {
      const [i, j] = queue.shift()!
      steps.push(
        snap(20, `(${i},${j}) = q.popleft()`, { current: [i, j] }),
      )

      for (const [di, dj] of dirs) {
        const ni = i + di
        const nj = j + dj
        if (ni < 0 || ni >= m || nj < 0 || nj >= n) {
          steps.push(
            snap(
              22,
              `邻居 (${ni},${nj})：越界，跳过`,
              { current: [i, j] },
            ),
          )
          continue
        }
        if (grid[ni][nj] !== 1) {
          const tag =
            grid[ni][nj] === 0 ? '空格' : grid[ni][nj] === 2 ? '已腐烂' : '?'
          steps.push(
            snap(
              22,
              `邻居 (${ni},${nj})：值 ${grid[ni][nj]}（${tag}），跳过`,
              { current: [i, j] },
            ),
          )
          continue
        }
        // 是新鲜 → 腐蚀它
        grid[ni][nj] = 2
        fresh -= 1
        queue.push([ni, nj])
        steps.push(
          snap(
            24,
            `邻居 (${ni},${nj}) 是新鲜！标为腐烂(=2)，fresh=${fresh}，入队`,
            { current: [ni, nj] },
          ),
        )
      }
    }

    minutes += 1
    steps.push(
      snap(27, `本层处理完，minutes += 1 → minutes = ${minutes}`),
    )
  }

  if (fresh === 0) {
    steps.push(
      snap(
        29,
        `循环结束：fresh=0 → 全部新鲜橘子已被腐蚀，return minutes = ${minutes}`,
      ),
    )
  } else {
    steps.push(
      snap(
        29,
        `循环结束：fresh=${fresh} > 0 → 有橘子永远到不了源，return -1`,
      ),
    )
  }
  return steps
}

const grid: number[][] = [
  [2, 1, 1],
  [1, 1, 0],
  [0, 1, 1],
]

const trace: AlgoTrace = {
  title: `示例：grid = ${JSON.stringify(grid)}（应返回 4）`,
  inputLabel: `grid=${JSON.stringify(grid)}`,
  vizKind: 'grid',
  vizConfig: {},
  steps: generateSteps(grid),
}

export default [trace]
