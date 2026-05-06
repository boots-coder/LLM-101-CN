import type { AlgoTrace, TraceStep } from './types'

// 二维 DP：dp[i][j] = text1[0..i-1] 与 text2[0..j-1] 的 LCS 长度
// 维数 (m+1) x (n+1)，第 0 行/列为 0（空串边界）
function generateSteps(text1: string, text2: string): TraceStep[] {
  const m = text1.length, n = text2.length
  const dp: (number | null)[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(null))
  // 边界：第 0 行 / 第 0 列全为 0
  for (let j = 0; j <= n; j++) dp[0][j] = 0
  for (let i = 0; i <= m; i++) dp[i][0] = 0

  const steps: TraceStep[] = []

  const snap = (line: number, note: string, opts: any = {}): TraceStep => ({
    line,
    note,
    state: {
      grid: dp.map((row) => row.map((v) => (v === null ? '·' : v))),
      scalars: [
        { label: 'm', value: String(m) },
        { label: 'n', value: String(n) },
        ...(opts.i !== undefined ? [{ label: 'i', value: String(opts.i) }] : []),
        ...(opts.j !== undefined ? [{ label: 'j', value: String(opts.j) }] : []),
        ...(opts.match !== undefined
          ? [{ label: 'match?', value: opts.match ? `${text1[opts.i - 1]}=${text2[opts.j - 1]}` : `${text1[opts.i - 1]}≠${text2[opts.j - 1]}` }]
          : []),
      ],
      current: opts.current,
      sources: opts.sources ?? [],
      transition: opts.transition,
      cellState: opts.cellState ?? {},
    },
  })

  // 边界初始化
  const baseCells: Record<string, string> = {}
  for (let j = 0; j <= n; j++) baseCells[`0,${j}`] = 'base'
  for (let i = 0; i <= m; i++) baseCells[`${i},0`] = 'base'
  steps.push(snap(3, `初始化 dp 为 (m+1)×(n+1) 全 0；第 0 行/列对应空串，LCS 长度=0`, {
    cellState: baseCells,
  }))

  // 主循环
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const c1 = text1[i - 1], c2 = text2[j - 1]
      if (c1 === c2) {
        steps.push(snap(6, `(i=${i}, j=${j}) text1[${i - 1}]='${c1}' == text2[${j - 1}]='${c2}'，匹配`, {
          i, j, current: [i, j], match: true,
          sources: [[i - 1, j - 1]],
          transition: `dp[${i}][${j}] = dp[${i - 1}][${j - 1}] + 1`,
        }))
        dp[i][j] = (dp[i - 1][j - 1] as number) + 1
        steps.push(snap(7, `dp[${i}][${j}] = ${dp[i - 1][j - 1]} + 1 = ${dp[i][j]}`, {
          i, j, current: [i, j],
          cellState: { [`${i},${j}`]: 'filled' },
        }))
      } else {
        steps.push(snap(8, `(i=${i}, j=${j}) '${c1}' ≠ '${c2}'，取上方/左方较大值`, {
          i, j, current: [i, j], match: false,
          sources: [[i - 1, j], [i, j - 1]],
          transition: `dp[${i}][${j}] = max(dp[${i - 1}][${j}], dp[${i}][${j - 1}])`,
        }))
        const up = dp[i - 1][j] as number
        const left = dp[i][j - 1] as number
        dp[i][j] = Math.max(up, left)
        steps.push(snap(9, `dp[${i}][${j}] = max(${up}, ${left}) = ${dp[i][j]}`, {
          i, j, current: [i, j],
          cellState: { [`${i},${j}`]: 'filled' },
        }))
      }
    }
  }

  steps.push(snap(11, `回填完毕，返回右下角 dp[${m}][${n}] = ${dp[m][n]}`, {
    cellState: { [`${m},${n}`]: 'answer' },
  }))
  return steps
}

const text1 = 'abcde'
const text2 = 'ace'

const trace: AlgoTrace = {
  title: `示例：text1 = "${text1}", text2 = "${text2}"`,
  inputLabel: `text1="${text1}", text2="${text2}"`,
  vizKind: 'dp-grid',
  vizConfig: {
    rows: text1.length + 1,
    cols: text2.length + 1,
    rowLabels: ['ε', ...text1.split('').map((c, i) => `${i}:${c}`)],
    colLabels: ['ε', ...text2.split('').map((c, j) => `${j}:${c}`)],
  },
  steps: generateSteps(text1, text2),
}

export default [trace]
