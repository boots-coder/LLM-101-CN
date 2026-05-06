import type { AlgoTrace, TraceStep } from './types'

function generateSteps(nums: number[]): TraceStep[] {
  let left = 0, right = nums.length - 1
  const steps: TraceStep[] = []
  let result: number | undefined

  const snap = (line: number, note: string, mid?: number, half?: 'L' | 'R'): TraceStep => {
    const ptrs: any[] = [
      { name: 'L', idx: left, color: '#3b82f6' },
      { name: 'R', idx: right, color: '#a855f7' },
    ]
    if (mid !== undefined) ptrs.push({ name: 'M', idx: mid, color: '#f59e0b' })
    const cellState: Record<number, string> = {}
    if (mid !== undefined) {
      if (half === 'L') {
        for (let k = left; k <= mid; k++) cellState[k] = 'left-half'
      } else if (half === 'R') {
        for (let k = mid + 1; k <= right; k++) cellState[k] = 'right-half'
      }
    }
    if (result !== undefined) cellState[result] = 'found'
    return {
      line,
      note,
      state: {
        pointers: ptrs,
        scalars: [
          { label: 'L', value: String(left) },
          { label: 'R', value: String(right) },
          ...(mid !== undefined ? [{ label: 'M', value: String(mid) }] : []),
          ...(mid !== undefined
            ? [{ label: 'nums[M] / nums[M+1]', value: `${nums[mid]} / ${mid + 1 <= nums.length - 1 ? nums[mid + 1] : '-∞'}` }]
            : []),
        ],
        cellState,
        result: result !== undefined ? String(result) : undefined,
      },
    }
  }

  steps.push(snap(2, `初始化 left=0, right=${right}（边界两侧默认 -∞）`))
  while (left < right) {
    steps.push(snap(3, `进入循环：left=${left} < right=${right}`))
    const mid = Math.floor((left + right) / 2)
    steps.push(
      snap(4, `mid = (${left}+${right}) // 2 = ${mid}, nums[mid]=${nums[mid]}, nums[mid+1]=${nums[mid + 1]}`, mid),
    )
    if (nums[mid] > nums[mid + 1]) {
      steps.push(
        snap(5, `nums[M]=${nums[mid]} > nums[M+1]=${nums[mid + 1]}，下降段，峰在 [L, M]`, mid, 'L'),
      )
      right = mid
      steps.push(snap(7, `right = mid = ${right}（保留 mid，可能它就是峰）`, mid, 'L'))
    } else {
      steps.push(
        snap(5, `nums[M]=${nums[mid]} ≤ nums[M+1]=${nums[mid + 1]}，上升段，峰在 [M+1, R]`, mid, 'R'),
      )
      left = mid + 1
      steps.push(snap(10, `left = mid + 1 = ${left}（mid 不是峰，排除）`, mid, 'R'))
    }
  }
  result = left
  steps.push(snap(11, `循环退出：left == right == ${left}，return ${left}（nums[${left}]=${nums[left]} 是峰）`))
  return steps
}

const nums = [1, 2, 1, 3, 5, 6, 4]

const trace: AlgoTrace = {
  title: `示例：nums = [${nums.join(', ')}]`,
  inputLabel: `nums=[${nums.join(',')}]`,
  vizKind: 'array-board',
  vizConfig: { values: nums, label: 'nums' },
  steps: generateSteps(nums),
}

export default [trace]
