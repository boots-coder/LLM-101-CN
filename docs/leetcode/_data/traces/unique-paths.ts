import type { AlgoTrace, TraceStep } from './types'

function generateSteps(m: number, n: number): TraceStep[] {
  // 用「真正的二维」grid 来 viz——每次更新一行的某列
  const grid: (number | null)[][] = Array.from({ length: m }, () => Array(n).fill(null))
  for (let j = 0; j < n; j++) grid[0][j] = 1
  for (let i = 1; i < m; i++) grid[i][0] = 1
  const steps: TraceStep[] = []

  const snap = (line: number, note: string, opts: any = {}): TraceStep => ({
    line,
    note,
    state: {
      grid: grid.map((row) => row.map((v) => (v === null ? '·' : v))),
      scalars: [
        { label: 'm', value: String(m) },
        { label: 'n', value: String(n) },
        ...(opts.i !== undefined ? [{ label: 'i', value: String(opts.i) }] : []),
        ...(opts.j !== undefined ? [{ label: 'j', value: String(opts.j) }] : []),
      ],
      current: opts.current,
      sources: opts.sources ?? [],
      transition: opts.transition,
      cellState: opts.cellState ?? {},
    },
  })

  steps.push(snap(3, `初始化第一行全为 1（边界）`, {
    cellState: Object.fromEntries(grid[0].map((_, j) => [`0,${j}`, 'base'])),
  }))
  steps.push(snap(3, `第一列也全为 1（边界）`, {
    cellState: Object.fromEntries(
      grid.map((_, i) => [`${i},0`, 'base'] as [string, string]),
    ),
  }))
  for (let i = 1; i < m; i++) {
    for (let j = 1; j < n; j++) {
      steps.push(snap(6, `进入 (i=${i}, j=${j})`, {
        i, j, current: [i, j],
        sources: [[i - 1, j], [i, j - 1]],
        transition: `dp[${j}] = dp[${j}] + dp[${j - 1}]（上 + 左）`,
      }))
      const above = grid[i - 1][j] as number
      const leftVal = grid[i][j - 1] as number
      grid[i][j] = above + leftVal
      steps.push(snap(7, `dp[${j}] = ${above} + ${leftVal} = ${grid[i][j]}`, {
        i, j, current: [i, j],
        cellState: { [`${i},${j}`]: 'filled' },
      }))
    }
  }
  steps.push(snap(8, `return dp[n-1] = ${grid[m - 1][n - 1]}`, {
    cellState: { [`${m - 1},${n - 1}`]: 'answer' },
  }))
  return steps
}

const m = 3, n = 3

const trace: AlgoTrace = {
  title: `示例：m = ${m}, n = ${n}`,
  inputLabel: `m=${m}, n=${n}`,
  vizKind: 'dp-grid',
  vizConfig: {
    rows: m,
    cols: n,
    rowLabels: Array.from({ length: m }, (_, i) => `i=${i}`),
    colLabels: Array.from({ length: n }, (_, j) => `j=${j}`),
  },
  steps: generateSteps(m, n),
}

export default [trace]
