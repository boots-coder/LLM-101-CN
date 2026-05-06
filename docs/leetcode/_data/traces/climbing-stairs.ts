import type { AlgoTrace, TraceStep } from './types'

function generateSteps(n: number): TraceStep[] {
  // 用 dp[i] 表示「到达第 i 阶的方法数」，下标 1..n（dp[0] 占位为 1，本可视化只展示 1..n）
  // grid 长度 n，列对应 i=1..n
  const dp: (number | null)[] = Array(n).fill(null)
  const steps: TraceStep[] = []

  let prev2 = 1 // dp[1]=1
  let prev1 = 2 // dp[2]=2

  const snap = (line: number, note: string, opts: any = {}): TraceStep => {
    const cs: Record<string, string> = {}
    if (opts.prev2Idx !== undefined) cs[`0,${opts.prev2Idx}`] = 'source'
    if (opts.prev1Idx !== undefined) cs[`0,${opts.prev1Idx}`] = 'source'
    if (opts.curIdx !== undefined) cs[`0,${opts.curIdx}`] = 'current'
    if (opts.answerIdx !== undefined) cs[`0,${opts.answerIdx}`] = 'answer'
    return {
      line,
      note,
      state: {
        grid: [dp.map((v) => (v === null ? '·' : String(v)))],
        scalars: [
          { label: 'n', value: String(n) },
          { label: 'prev2', value: String(prev2) },
          { label: 'prev1', value: String(prev1) },
          ...(opts.cur !== undefined ? [{ label: 'cur', value: String(opts.cur) }] : []),
          ...(opts.i !== undefined ? [{ label: 'i', value: String(opts.i) }] : []),
        ],
        cellState: cs,
        transition: opts.transition,
      },
    }
  }

  steps.push(snap(2, `检查 n=${n}，n>2 进入 DP 路径`))

  if (n >= 1) dp[0] = 1
  steps.push(snap(5, `初始化 dp[1] = prev2 = 1（只有 1 种走法：1）`, { prev2Idx: 0 }))

  if (n >= 2) dp[1] = 2
  steps.push(snap(5, `初始化 dp[2] = prev1 = 2（两种：1+1 / 2）`, { prev1Idx: 1 }))

  for (let i = 3; i <= n; i++) {
    const curIdx = i - 1
    steps.push(snap(6, `进入 i=${i}（在 grid 上第 ${i} 列）`, {
      i,
      prev2Idx: i - 3,
      prev1Idx: i - 2,
      curIdx,
      transition: `dp[${i}] = prev1 + prev2 = ${prev1} + ${prev2}`,
    }))

    const cur = prev1 + prev2
    dp[curIdx] = cur
    steps.push(snap(7, `cur = ${prev1} + ${prev2} = ${cur}，写入 dp[${i}]`, {
      i,
      curIdx,
      cur,
      transition: `dp[${i}] ← ${cur}`,
    }))

    prev2 = prev1
    prev1 = cur
    steps.push(snap(8, `滚动：prev2=${prev2}, prev1=${prev1}（释放 dp[${i - 2}]）`, {
      i,
      curIdx,
    }))
  }

  steps.push(snap(9, `循环结束，return prev1 = ${prev1} = dp[${n}]`, {
    answerIdx: n - 1,
  }))

  return steps
}

const n = 12

const trace: AlgoTrace = {
  title: `示例：n = ${n}`,
  inputLabel: `n=${n}`,
  vizKind: 'dp-grid',
  vizConfig: {
    rows: 1,
    cols: n,
    rowLabels: ['dp'],
    colLabels: Array.from({ length: n }, (_, i) => `i=${i + 1}`),
  },
  steps: generateSteps(n),
}

export default [trace]
