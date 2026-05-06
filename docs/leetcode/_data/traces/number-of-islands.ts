import type { AlgoTrace, TraceStep } from './types'

function generateSteps(input: string[][]): TraceStep[] {
  const grid = input.map((row) => [...row])
  const m = grid.length, n = grid[0].length
  const steps: TraceStep[] = []
  let cnt = 0

  function snapshot(): string[][] {
    return grid.map((row) =>
      row.map((c) => (c === '1' ? '🟩' : c === '0' ? '🟦' : c)),
    )
  }

  const snap = (line: number, note: string, opts: any = {}): TraceStep => {
    const cellState: Record<string, string> = {}
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        if (grid[i][j] === '1') cellState[`${i},${j}`] = 'land'
        else cellState[`${i},${j}`] = 'water'
      }
    }
    if (opts.current) cellState[`${opts.current[0]},${opts.current[1]}`] = 'current'
    return {
      line,
      note,
      state: {
        grid: snapshot(),
        scalars: [
          { label: 'cnt', value: String(cnt) },
          ...(opts.i !== undefined ? [{ label: '(i,j)', value: `(${opts.i},${opts.j})` }] : []),
        ],
        cellState,
        current: opts.current,
      },
    }
  }

  function dfs(i: number, j: number) {
    steps.push(snap(7, `dfs(${i},${j})：判断越界 / 是否陆地`, { current: [i, j], i, j }))
    if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] !== '1') {
      steps.push(snap(8, `不是 '1'（越界或已沉），return`, { current: [i, j], i, j }))
      return
    }
    grid[i][j] = '0'
    steps.push(snap(9, `沉岛：grid[${i}][${j}] = '0'`, { current: [i, j], i, j }))
    for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as [number, number][]) {
      dfs(i + di, j + dj)
    }
  }

  steps.push(snap(13, '初始化 cnt = 0；从 (0,0) 开始扫描'))
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      steps.push(snap(14, `外层扫到 (${i},${j})，值 = '${grid[i][j]}'`, { current: [i, j], i, j }))
      if (grid[i][j] === '1') {
        cnt++
        steps.push(snap(17, `发现新岛屿！cnt = ${cnt}`, { current: [i, j], i, j }))
        dfs(i, j)
      }
    }
  }
  steps.push(snap(19, `扫描完毕，return cnt = ${cnt}`))
  return steps
}

const grid = [
  ['1', '1', '0', '0', '0'],
  ['1', '1', '0', '0', '0'],
  ['0', '0', '1', '0', '0'],
  ['0', '0', '0', '1', '1'],
]

const trace: AlgoTrace = {
  title: '示例：4×5 grid（应返回 3 个岛）',
  inputLabel: '4×5 grid',
  vizKind: 'grid',
  vizConfig: {},
  steps: generateSteps(grid),
}

export default [trace]
