import type { AlgoTrace, TraceStep } from './types'

function generateSteps(nums: number[], target: number): TraceStep[] {
  let left = 0, right = nums.length - 1
  const steps: TraceStep[] = []
  let result: number | undefined

  const snap = (line: number, note: string, mid?: number, half?: 'L' | 'R'): TraceStep => {
    const ptrs = [
      { name: 'L', idx: left, color: '#3b82f6' },
      { name: 'R', idx: right, color: '#a855f7' },
    ]
    if (mid !== undefined) ptrs.push({ name: 'M', idx: mid, color: '#f59e0b' })
    const cellState: Record<number, string> = {}
    if (left <= right) {
      const m = mid !== undefined ? mid : Math.floor((left + right) / 2)
      if (half === 'L') {
        for (let k = left; k <= m; k++) cellState[k] = 'left-half'
      } else if (half === 'R') {
        for (let k = m; k <= right; k++) cellState[k] = 'right-half'
      }
    }
    if (result !== undefined && result >= 0) cellState[result] = 'found'
    return {
      line,
      note,
      state: {
        pointers: ptrs,
        scalars: [
          { label: 'target', value: String(target) },
          { label: 'L', value: String(left) },
          { label: 'R', value: String(right) },
          ...(mid !== undefined ? [{ label: 'M', value: String(mid) }] : []),
        ],
        cellState,
        result: result !== undefined ? String(result) : undefined,
      },
    }
  }

  steps.push(snap(2, `初始化 left=0, right=${right}`))
  while (left <= right) {
    const mid = Math.floor((left + right) / 2)
    steps.push(snap(4, `mid = (${left}+${right}) // 2 = ${mid}, nums[mid]=${nums[mid]}`, mid))
    if (nums[mid] === target) {
      result = mid
      steps.push(snap(6, `nums[mid]==target，命中！return ${mid}`, mid))
      return steps
    }
    if (nums[left] <= nums[mid]) {
      steps.push(snap(8, `nums[L]=${nums[left]} ≤ nums[M]=${nums[mid]}，左半 [L, M] 有序`, mid, 'L'))
      if (nums[left] <= target && target < nums[mid]) {
        right = mid - 1
        steps.push(snap(11, `target=${target} 在左半范围内，right = mid-1 = ${right}`, mid, 'L'))
      } else {
        left = mid + 1
        steps.push(snap(13, `target=${target} 不在左半，去右半找：left = mid+1 = ${left}`, mid, 'L'))
      }
    } else {
      steps.push(snap(14, `nums[L]=${nums[left]} > nums[M]=${nums[mid]}，右半 [M, R] 有序`, mid, 'R'))
      if (nums[mid] < target && target <= nums[right]) {
        left = mid + 1
        steps.push(snap(17, `target=${target} 在右半范围内，left = mid+1 = ${left}`, mid, 'R'))
      } else {
        right = mid - 1
        steps.push(snap(19, `target=${target} 不在右半，去左半找：right = mid-1 = ${right}`, mid, 'R'))
      }
    }
  }
  result = -1
  steps.push(snap(20, '搜索完毕未找到，return -1'))
  return steps
}

const nums = [4, 5, 6, 7, 0, 1, 2]
const target = 0

const trace: AlgoTrace = {
  title: `示例：nums = [${nums.join(', ')}], target = ${target}`,
  inputLabel: `nums=[${nums.join(',')}], target=${target}`,
  vizKind: 'array-board',
  vizConfig: { values: nums, label: 'nums' },
  steps: generateSteps(nums, target),
}

export default [trace]
