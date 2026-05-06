import type { AlgoTrace, TraceStep } from './types'

function generateSteps(nums: number[]): TraceStep[] {
  const n = nums.length
  const dp: (number | null)[] = Array(n).fill(null)
  let prev2 = nums[0], prev1 = Math.max(nums[0], nums[1])
  dp[0] = prev2
  dp[1] = prev1
  const steps: TraceStep[] = []

  const snap = (line: number, note: string, opts: any = {}): TraceStep => {
    const cs: Record<string, string> = {}
    if (opts.prev2Idx !== undefined) cs[`0,${opts.prev2Idx}`] = 'source'
    if (opts.prev1Idx !== undefined) cs[`0,${opts.prev1Idx}`] = 'source'
    if (opts.numIdx !== undefined) cs[`0,${opts.numIdx}`] = 'source'
    if (opts.curIdx !== undefined) cs[`0,${opts.curIdx}`] = 'current'
    return {
      line,
      note,
      state: {
        grid: [dp.map((v) => (v === null ? '·' : String(v)))],
        scalars: [
          { label: 'prev2', value: String(prev2) },
          { label: 'prev1', value: String(prev1) },
          ...(opts.cur !== undefined ? [{ label: 'cur', value: String(opts.cur) }] : []),
        ],
        cellState: cs,
        transition: opts.transition,
      },
    }
  }

  steps.push(snap(2, '检查输入非空', {}))
  steps.push(snap(8, `初始化 prev2=nums[0]=${prev2}, prev1=max(nums[0],nums[1])=${prev1}`, {
    prev2Idx: 0, prev1Idx: 1,
  }))
  for (let i = 2; i < n; i++) {
    steps.push(snap(9, `进入 i=${i}, nums[i]=${nums[i]}`, {
      prev2Idx: i - 2, prev1Idx: i - 1, numIdx: i,
      transition: `dp[${i}] = max(prev1, prev2 + nums[${i}]) = max(${prev1}, ${prev2}+${nums[i]})`,
    }))
    const cur = Math.max(prev1, prev2 + nums[i])
    dp[i] = cur
    steps.push(snap(10, `cur = max(${prev1}, ${prev2 + nums[i]}) = ${cur}`, {
      curIdx: i, cur,
      transition: `选 ${cur === prev1 ? '不偷' : '偷'} 第 ${i} 间`,
    }))
    prev2 = prev1
    prev1 = cur
    steps.push(snap(11, `滚动：prev2=${prev2}, prev1=${prev1}`, { curIdx: i }))
  }
  steps.push(snap(12, `所有 i 处理完，return prev1 = ${prev1}`, { curIdx: n - 1 }))
  return steps
}

const nums = [2, 7, 9, 3, 1]

const trace: AlgoTrace = {
  title: `示例：nums = [${nums.join(', ')}]`,
  inputLabel: `nums=[${nums.join(',')}]`,
  vizKind: 'dp-grid',
  vizConfig: {
    rows: 1,
    cols: nums.length,
    rowLabels: ['dp'],
    colLabels: nums.map((_, i) => `i=${i}`),
  },
  steps: generateSteps(nums),
}

export default [trace]
