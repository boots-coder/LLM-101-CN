import type { AlgoTrace, TraceStep } from './types'

function generateSteps(input: number[]): TraceStep[] {
  const nums = [...input].sort((a, b) => a - b)
  const n = nums.length
  const res: number[][] = []
  const steps: TraceStep[] = []

  const snap = (line: number, note: string, opts: any = {}): TraceStep => {
    const ptrs = []
    if (opts.i !== undefined) ptrs.push({ name: 'i', idx: opts.i, color: '#22c55e' })
    if (opts.left !== undefined) ptrs.push({ name: 'L', idx: opts.left, color: '#3b82f6' })
    if (opts.right !== undefined) ptrs.push({ name: 'R', idx: opts.right, color: '#a855f7' })
    return {
      line,
      note,
      state: {
        pointers: ptrs,
        scalars: [
          ...(opts.s !== undefined ? [{ label: 'sum', value: String(opts.s) }] : []),
          { label: 'res.size', value: String(res.length) },
        ],
        cellState: opts.cellState ?? {},
        result: res.map((r) => `[${r.join(',')}]`).join(' '),
      },
    }
  }

  steps.push(snap(2, `排序后：nums = [${nums.join(', ')}]`))
  for (let i = 0; i < n - 2; i++) {
    steps.push(snap(5, `外层 i = ${i}, nums[i] = ${nums[i]}`, { i }))
    if (nums[i] > 0) {
      steps.push(snap(6, `nums[i] = ${nums[i]} > 0，后续不可能凑 0，break`, { i }))
      break
    }
    if (i > 0 && nums[i] === nums[i - 1]) {
      steps.push(snap(8, `nums[${i}] == nums[${i - 1}]，跳过重复 i`, { i }))
      continue
    }
    let left = i + 1, right = n - 1
    steps.push(snap(10, `left=${left}, right=${right}`, { i, left, right }))
    while (left < right) {
      const s = nums[i] + nums[left] + nums[right]
      steps.push(
        snap(12, `s = ${nums[i]} + ${nums[left]} + ${nums[right]} = ${s}`, { i, left, right, s }),
      )
      if (s === 0) {
        res.push([nums[i], nums[left], nums[right]])
        steps.push(
          snap(14, `s == 0 命中！收 [${nums[i]},${nums[left]},${nums[right]}]`, {
            i, left, right, s,
            cellState: { [i]: 'found', [left]: 'found', [right]: 'found' },
          }),
        )
        while (left < right && nums[left] === nums[left + 1]) {
          steps.push(snap(15, `nums[L]=${nums[left]} 与 nums[L+1] 重复，L++ 跳过`, { i, left, right }))
          left++
        }
        while (left < right && nums[right] === nums[right - 1]) {
          steps.push(snap(17, `nums[R]=${nums[right]} 与 nums[R-1] 重复，R-- 跳过`, { i, left, right }))
          right--
        }
        left++
        right--
        steps.push(snap(19, `双指针都向中间收一步`, { i, left, right }))
      } else if (s < 0) {
        left++
        steps.push(snap(22, `s < 0，左指针右移 → 增大和`, { i, left, right }))
      } else {
        right--
        steps.push(snap(24, `s > 0，右指针左移 → 减小和`, { i, left, right }))
      }
    }
    steps.push(snap(11, `内层结束（L=${left} ≥ R=${right}）`, { i }))
  }
  steps.push(snap(25, `搜索完毕，return res（${res.length} 组）`))
  return steps
}

const trace: AlgoTrace = {
  title: '示例：nums = [-1, 0, 1, 2, -1, -4]',
  inputLabel: 'nums=[-1,0,1,2,-1,-4]',
  vizKind: 'array-board',
  vizConfig: { values: [-4, -1, -1, 0, 1, 2], label: 'nums（已排序）' },
  steps: generateSteps([-1, 0, 1, 2, -1, -4]),
}

export default [trace]
