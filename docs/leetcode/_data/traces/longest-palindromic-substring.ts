import type { AlgoTrace, TraceStep } from './types'

// 二维 DP：dp[i][j] = s[i..j] 是否回文（i <= j）
// 按子串长度 L 从 1 到 n 由小到大填，保证依赖 dp[i+1][j-1] 已就绪
function generateSteps(s: string): TraceStep[] {
  const n = s.length
  // 用 0 / 1 表示 false / true，null 表示尚未填
  const dp: (number | null)[][] = Array.from({ length: n }, () => Array(n).fill(null))
  // 下三角（i > j）置为 0（不存在的子串），便于可视化
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) dp[i][j] = 0
  }

  const steps: TraceStep[] = []
  let bestStart = 0, bestLen = 1

  const display = () =>
    dp.map((row, i) =>
      row.map((v, j) => {
        if (j < i) return ''
        if (v === null) return '·'
        return v === 1 ? 'T' : 'F'
      }),
    )

  const snap = (line: number, note: string, opts: any = {}): TraceStep => ({
    line,
    note,
    state: {
      grid: display(),
      scalars: [
        { label: 'n', value: String(n) },
        ...(opts.L !== undefined ? [{ label: 'L', value: String(opts.L) }] : []),
        ...(opts.i !== undefined ? [{ label: 'i', value: String(opts.i) }] : []),
        ...(opts.j !== undefined ? [{ label: 'j', value: String(opts.j) }] : []),
        { label: 'best', value: `s[${bestStart}..${bestStart + bestLen - 1}]="${s.slice(bestStart, bestStart + bestLen)}"` },
      ],
      current: opts.current,
      sources: opts.sources ?? [],
      transition: opts.transition,
      cellState: opts.cellState ?? {},
    },
  })

  steps.push(snap(2, `输入 s = "${s}"，长度 n = ${n}；目标：dp[i][j] 表示 s[i..j] 是否回文`, {}))

  // L = 1：单字符天然回文
  for (let i = 0; i < n; i++) dp[i][i] = 1
  const baseCells1: Record<string, string> = {}
  for (let i = 0; i < n; i++) baseCells1[`${i},${i}`] = 'base'
  steps.push(snap(4, `长度 L=1：所有 dp[i][i] = T（单字符回文，对角线）`, {
    L: 1, cellState: baseCells1,
  }))

  // L = 2：相邻字符相等才回文
  if (n >= 2) {
    steps.push(snap(5, `开始填长度 L=2 的子串`, { L: 2 }))
    for (let i = 0; i + 1 < n; i++) {
      const j = i + 1
      const eq = s[i] === s[j]
      steps.push(snap(6, `(i=${i}, j=${j}) 检查 s[${i}]='${s[i]}' ?= s[${j}]='${s[j]}'`, {
        L: 2, i, j, current: [i, j],
        transition: `dp[${i}][${j}] = (s[${i}] == s[${j}])`,
      }))
      dp[i][j] = eq ? 1 : 0
      if (eq && 2 > bestLen) { bestStart = i; bestLen = 2 }
      steps.push(snap(7, `dp[${i}][${j}] = ${eq ? 'T' : 'F'}${eq ? `；更新 best 候选 "${s.slice(i, j + 1)}"` : ''}`, {
        L: 2, i, j, current: [i, j],
        cellState: { [`${i},${j}`]: eq ? 'filled' : 'base' },
      }))
    }
  }

  // L >= 3：两端相等且去掉两端后仍回文
  for (let L = 3; L <= n; L++) {
    steps.push(snap(9, `进入长度 L=${L} 的子串`, { L }))
    for (let i = 0; i + L - 1 < n; i++) {
      const j = i + L - 1
      steps.push(snap(10, `(i=${i}, j=${j}) 检查两端 s[${i}]='${s[i]}' ?= s[${j}]='${s[j]}' 且 dp[${i + 1}][${j - 1}]`, {
        L, i, j, current: [i, j],
        sources: [[i + 1, j - 1]],
        transition: `dp[${i}][${j}] = (s[${i}]==s[${j}]) && dp[${i + 1}][${j - 1}]`,
      }))
      const inner = dp[i + 1][j - 1] as number
      const ok = s[i] === s[j] && inner === 1
      dp[i][j] = ok ? 1 : 0
      if (ok && L > bestLen) { bestStart = i; bestLen = L }
      steps.push(snap(11, `dp[${i}][${j}] = ${ok ? 'T' : 'F'}${ok ? `；更新 best 候选 "${s.slice(i, j + 1)}"` : ''}`, {
        L, i, j, current: [i, j],
        cellState: { [`${i},${j}`]: ok ? 'filled' : 'base' },
      }))
    }
  }

  // 答案 cell 高亮
  const ansCells: Record<string, string> = {}
  ansCells[`${bestStart},${bestStart + bestLen - 1}`] = 'answer'
  steps.push(snap(14, `填表完毕，最长回文子串 = "${s.slice(bestStart, bestStart + bestLen)}"（s[${bestStart}..${bestStart + bestLen - 1}]，长度 ${bestLen}）`, {
    cellState: ansCells,
  }))

  return steps
}

const s = 'babad'

const trace: AlgoTrace = {
  title: `示例：s = "${s}"`,
  inputLabel: `s="${s}"`,
  vizKind: 'dp-grid',
  vizConfig: {
    rows: s.length,
    cols: s.length,
    rowLabels: s.split('').map((c, i) => `i=${i}:${c}`),
    colLabels: s.split('').map((c, j) => `j=${j}:${c}`),
  },
  steps: generateSteps(s),
}

export default [trace]
