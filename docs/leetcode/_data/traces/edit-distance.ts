import type { AlgoTrace, TraceStep } from './types'

function generateSteps(word1: string, word2: string): TraceStep[] {
  const m = word1.length
  const n = word2.length
  // dp[i][j] = 把 word1[:i] 转成 word2[:j] 的最少操作数；尺寸 (m+1) × (n+1)
  const dp: (number | null)[][] = Array.from({ length: m + 1 }, () =>
    Array(n + 1).fill(null),
  )
  const steps: TraceStep[] = []

  const snap = (line: number, note: string, opts: any = {}): TraceStep => ({
    line,
    note,
    state: {
      grid: dp.map((row) => row.map((v) => (v === null ? '·' : v))),
      scalars: [
        { label: 'word1', value: `"${word1}"` },
        { label: 'word2', value: `"${word2}"` },
        { label: 'm', value: String(m) },
        { label: 'n', value: String(n) },
        ...(opts.i !== undefined ? [{ label: 'i', value: String(opts.i) }] : []),
        ...(opts.j !== undefined ? [{ label: 'j', value: String(opts.j) }] : []),
      ],
      current: opts.current,
      sources: opts.sources ?? [],
      cellState: opts.cellState ?? {},
      transition: opts.transition,
    },
  })

  steps.push(snap(4, `初始化 (${m + 1})×(${n + 1}) DP 表`))

  // 第一列：word2 空，全删
  for (let i = 0; i <= m; i++) dp[i][0] = i
  steps.push(snap(7, `第一列：dp[i][0] = i（把 word1[:i] 全删变空串）`, {
    cellState: Object.fromEntries(
      Array.from({ length: m + 1 }, (_, i) => [`${i},0`, 'base']),
    ),
  }))

  // 第一行：word1 空，全插
  for (let j = 0; j <= n; j++) dp[0][j] = j
  steps.push(snap(10, `第一行：dp[0][j] = j（空串插 j 次得到 word2[:j]）`, {
    cellState: Object.fromEntries(
      Array.from({ length: n + 1 }, (_, j) => [`0,${j}`, 'base']),
    ),
  }))

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const c1 = word1[i - 1]
      const c2 = word2[j - 1]
      steps.push(snap(11, `进入 (i=${i}, j=${j})：word1[${i - 1}]='${c1}' vs word2[${j - 1}]='${c2}'`, {
        i, j,
        current: [i, j],
      }))

      if (c1 === c2) {
        const v = dp[i - 1][j - 1] as number
        dp[i][j] = v
        steps.push(snap(13, `'${c1}' == '${c2}' → 末位相同，无需操作，dp[${i}][${j}] = dp[${i - 1}][${j - 1}] = ${v}`, {
          i, j,
          current: [i, j],
          sources: [[i - 1, j - 1]],
          cellState: { [`${i},${j}`]: 'filled' },
          transition: `dp[${i}][${j}] ← dp[${i - 1}][${j - 1}] = ${v}`,
        }))
      } else {
        const rep = dp[i - 1][j - 1] as number
        const del = dp[i - 1][j] as number
        const ins = dp[i][j - 1] as number
        const best = Math.min(rep, del, ins)
        steps.push(snap(16, `'${c1}' ≠ '${c2}' → 三选一：替换 ${rep} / 删除 ${del} / 插入 ${ins}`, {
          i, j,
          current: [i, j],
          sources: [[i - 1, j - 1], [i - 1, j], [i, j - 1]],
          transition: `dp[${i}][${j}] = 1 + min(${rep}, ${del}, ${ins}) = 1 + ${best}`,
        }))
        dp[i][j] = best + 1
        const which = best === rep ? '替换' : best === del ? '删除' : '插入'
        steps.push(snap(17, `选 ${which}（最小=${best}），dp[${i}][${j}] = ${best + 1}`, {
          i, j,
          current: [i, j],
          cellState: { [`${i},${j}`]: 'filled' },
          transition: `dp[${i}][${j}] ← ${best + 1}`,
        }))
      }
    }
  }

  steps.push(snap(22, `return dp[${m}][${n}] = ${dp[m][n]}（编辑距离）`, {
    cellState: { [`${m},${n}`]: 'answer' },
  }))

  return steps
}

const word1 = 'horse'
const word2 = 'ros'

const trace: AlgoTrace = {
  title: `示例：word1 = "${word1}", word2 = "${word2}"`,
  inputLabel: `word1="${word1}", word2="${word2}"`,
  vizKind: 'dp-grid',
  vizConfig: {
    rows: word1.length + 1,
    cols: word2.length + 1,
    rowLabels: ['ε', ...word1.split('')],
    colLabels: ['ε', ...word2.split('')],
  },
  steps: generateSteps(word1, word2),
}

export default [trace]
