import type { AlgoTrace, TraceStep } from './types'

function generateSteps(nums: number[], k: number): TraceStep[] {
  const cnt: Record<number, number> = { 0: 1 }
  let prefix = 0, ans = 0
  const steps: TraceStep[] = []

  const snap = (line: number, note: string, opts: any = {}): TraceStep => ({
    line,
    note,
    state: {
      pointers: opts.i !== undefined
        ? [{ name: 'i', idx: opts.i, color: '#22c55e' }]
        : [],
      scalars: [
        { label: 'k', value: String(k) },
        { label: 'prefix', value: String(prefix) },
        { label: 'ans', value: String(ans) },
        ...(opts.need !== undefined
          ? [{ label: 'prefix-k', value: `${prefix} - ${k} = ${opts.need}` }]
          : []),
      ],
      kv: {
        title: 'cnt（前缀和 → 出现次数）',
        emptyLabel: '空',
        entries: Object.entries(cnt).map(([key, v]) => ({
          key, value: v, highlight: opts.hitKey !== undefined && Number(key) === opts.hitKey,
        })),
      },
      cellState: opts.cellState ?? {},
    },
  })

  steps.push(snap(2, '初始化 cnt = {0: 1}（处理整段和=k 的边界）'))
  for (let i = 0; i < nums.length; i++) {
    const x = nums[i]
    steps.push(snap(5, `进入 i=${i}, x=${x}`, { i }))
    prefix += x
    steps.push(snap(6, `prefix += ${x} → prefix = ${prefix}`, { i }))
    const need = prefix - k
    const hits = cnt[need] ?? 0
    steps.push(
      snap(7, `查 cnt[prefix-k] = cnt[${need}] = ${hits} → ans += ${hits}`, {
        i, need, hitKey: need in cnt ? need : undefined,
      }),
    )
    ans += hits
    cnt[prefix] = (cnt[prefix] ?? 0) + 1
    steps.push(snap(8, `把 prefix=${prefix} 计入 cnt（cnt[${prefix}] = ${cnt[prefix]}）`, { i }))
  }
  steps.push(snap(9, `循环完，return ans = ${ans}`))
  return steps
}

const nums = [1, 1, 1]
const k = 2

const trace: AlgoTrace = {
  title: `示例：nums = [${nums.join(', ')}], k = ${k}`,
  inputLabel: `nums=[${nums.join(',')}], k=${k}`,
  vizKind: 'array-board',
  vizConfig: { values: nums, label: 'nums' },
  steps: generateSteps(nums, k),
}

export default [trace]
