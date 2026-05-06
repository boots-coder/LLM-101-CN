import type { AlgoTrace, TraceStep } from './types'

function generateSteps(nums: number[]): TraceStep[] {
  let ans = 0
  const steps: TraceStep[] = []
  const seenCount: Record<number, number> = {}

  const toBin = (v: number, width = 4) => {
    if (v < 0) return v.toString(2)
    return v.toString(2).padStart(width, '0')
  }

  const snap = (line: number, note: string, opts: any = {}): TraceStep => {
    const ptrs: any[] = []
    if (opts.i !== undefined) ptrs.push({ name: 'i', idx: opts.i, color: '#22c55e' })
    return {
      line,
      note,
      state: {
        pointers: ptrs,
        scalars: [
          { label: 'ans', value: String(ans) },
          { label: 'ans(bin)', value: toBin(ans) },
          ...(opts.x !== undefined
            ? [{ label: 'x', value: String(opts.x) }, { label: 'x(bin)', value: toBin(opts.x) }]
            : []),
          ...(opts.newAns !== undefined
            ? [{ label: 'ans^x', value: String(opts.newAns), cls: 'ok' }]
            : []),
        ],
        kv: {
          title: '已 XOR 进 ans 的元素出现次数',
          emptyLabel: '空',
          entries: Object.entries(seenCount).map(([k, v]) => ({
            key: k,
            value: v,
            highlight: v % 2 === 1,
          })),
        },
        cellState: opts.cellState ?? {},
        result: opts.done ? ans : undefined,
      },
    }
  }

  steps.push(snap(2, '初始化 ans = 0（XOR 的单位元）'))
  for (let i = 0; i < nums.length; i++) {
    const x = nums[i]
    steps.push(snap(3, `进入第 ${i} 轮：x = nums[${i}] = ${x}`, { i, x }))
    const newAns = ans ^ x
    steps.push(
      snap(4, `ans ^= x → ${ans} ^ ${x} = ${newAns}（${toBin(ans)} ^ ${toBin(x)} = ${toBin(newAns)}）`, {
        i, x, newAns,
      }),
    )
    ans = newAns
    seenCount[x] = (seenCount[x] ?? 0) + 1
    steps.push(
      snap(4, `ans 更新为 ${ans}；${seenCount[x] % 2 === 0 ? `${x} 已成对，从 ans 中抵消` : `${x} 当前为单数次`}`, {
        i,
        cellState: { [i]: 'visited' },
      }),
    )
  }
  steps.push(snap(5, `扫描结束，return ans = ${ans}（成对元素全抵消，剩下的就是答案）`, { done: true }))
  return steps
}

const nums = [4, 1, 2, 1, 2]

const trace: AlgoTrace = {
  title: `示例：nums = [${nums.join(', ')}]`,
  inputLabel: `nums=[${nums.join(',')}]`,
  vizKind: 'array-board',
  vizConfig: { values: nums, label: 'nums' },
  steps: generateSteps(nums),
}

export default [trace]
