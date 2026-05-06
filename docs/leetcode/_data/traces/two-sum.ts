import type { AlgoTrace, TraceStep } from './types'

function generateSteps(nums: number[], target: number): TraceStep[] {
  const seen: Record<number, number> = {}
  const steps: TraceStep[] = []
  let result: number[] | undefined

  const snap = (line: number, note: string, opts: any = {}): TraceStep => ({
    line,
    note,
    state: {
      pointers: opts.i !== undefined
        ? [{ name: 'i', idx: opts.i, color: '#22c55e' }]
        : [],
      scalars: [
        { label: 'target', value: String(target) },
        ...(opts.complement !== undefined
          ? [{ label: 'complement', value: `target - num = ${opts.complement}` }]
          : []),
      ],
      kv: {
        title: 'seen（值 → 下标）',
        emptyLabel: '空',
        entries: Object.entries(seen).map(([k, v]) => ({
          key: k,
          value: v,
          highlight: opts.hitKey !== undefined && Number(k) === opts.hitKey,
        })),
      },
      cellState: opts.cellState ?? {},
      result,
    },
  })

  steps.push(snap(2, '初始化哈希表 seen = {}'))
  for (let i = 0; i < nums.length; i++) {
    const num = nums[i]
    steps.push(snap(3, `开始第 ${i} 轮：i=${i}, num=${num}`, { i }))
    const complement = target - num
    steps.push(snap(4, `计算 complement = ${target} - ${num} = ${complement}`, { i, complement }))
    if (complement in seen) {
      steps.push(
        snap(5, `complement=${complement} 在 seen 里！配对成功 ↩`, {
          i, complement, hitKey: complement,
        }),
      )
      result = [seen[complement], i]
      steps.push(
        snap(6, `return [${seen[complement]}, ${i}]`, {
          i, complement, hitKey: complement,
          cellState: { [seen[complement]]: 'found', [i]: 'found' },
        }),
      )
      return steps
    }
    steps.push(snap(5, `complement=${complement} 不在 seen，继续`, { i, complement }))
    seen[num] = i
    steps.push(snap(7, `存入 seen[${num}] = ${i}`, { i }))
  }
  steps.push(snap(8, '循环跑完仍未配对，return []'))
  return steps
}

const nums = [2, 7, 11, 15]
const target = 9

const trace: AlgoTrace = {
  title: `示例：nums = [${nums.join(', ')}], target = ${target}`,
  inputLabel: `nums=[${nums.join(',')}], target=${target}`,
  vizKind: 'array-board',
  vizConfig: { values: nums, label: 'nums' },
  steps: generateSteps(nums, target),
}

export default [trace]
