import type { AlgoTrace, TraceStep } from './types'

// 行号对应 problems/product-except-self.ts 里的 solutionCode：
//   1: def productExceptSelf(nums: list[int]) -> list[int]:
//   2:     n = len(nums)
//   3:     answer = [1] * n
//   4:     # 第一遍：answer[i] = i 左侧所有元素的乘积
//   5:     left = 1
//   6:     for i in range(n):
//   7:         answer[i] = left
//   8:         left *= nums[i]
//   9:     # 第二遍：从右往左乘上右侧乘积
//  10:     right = 1
//  11:     for i in range(n - 1, -1, -1):
//  12:         answer[i] *= right
//  13:         right *= nums[i]
//  14:     return answer

function generateSteps(nums: number[]): TraceStep[] {
  const steps: TraceStep[] = []
  const n = nums.length
  const answer: number[] = Array(n).fill(1)
  let left = 1
  let right = 1

  const snap = (
    line: number,
    note: string,
    opts: {
      i?: number
      cellState?: Record<number, string>
      pass?: 1 | 2
    } = {},
  ): TraceStep => {
    const ptrs: any[] = []
    if (opts.i !== undefined && opts.i >= 0 && opts.i < n) {
      const color = opts.pass === 2 ? '#a855f7' : '#22c55e'
      const name = opts.pass === 2 ? 'i←' : 'i→'
      ptrs.push({ name, idx: opts.i, color })
    }
    return {
      line,
      note,
      state: {
        pointers: ptrs,
        scalars: [
          { label: 'left', value: String(left) },
          { label: 'right', value: String(right) },
          ...(opts.pass !== undefined ? [{ label: 'pass', value: String(opts.pass) }] : []),
        ],
        cellState: opts.cellState ?? {},
        secondary: { label: 'answer', values: [...answer], cellCls: {} },
        result: [...answer],
      },
    }
  }

  steps.push(snap(1, `函数入口：productExceptSelf(nums)，nums = [${nums.join(', ')}]`))
  steps.push(snap(2, `n = len(nums) = ${n}`))
  steps.push(snap(3, `初始化 answer = [1]*${n} = [${answer.join(', ')}]`))
  steps.push(snap(5, '第一遍开始：left = 1（乘法单位元，索引 0 左侧没有元素）'))

  for (let i = 0; i < n; i++) {
    steps.push(snap(6, `进入第一遍循环：i=${i}`, { i, pass: 1 }))
    answer[i] = left
    steps.push(
      snap(7, `answer[${i}] = left = ${left}（写入"i 左侧所有元素的乘积"）`, {
        i,
        pass: 1,
        cellState: { [i]: 'left-half' },
      }),
    )
    const oldLeft = left
    left *= nums[i]
    steps.push(
      snap(8, `left *= nums[${i}] → left = ${oldLeft} × ${nums[i]} = ${left}`, {
        i,
        pass: 1,
        cellState: { [i]: 'visited' },
      }),
    )
  }

  steps.push(snap(10, '第二遍开始：right = 1（索引 n-1 右侧没有元素）'))

  for (let i = n - 1; i >= 0; i--) {
    steps.push(snap(11, `进入第二遍循环：i=${i}（倒序）`, { i, pass: 2 }))
    const before = answer[i]
    answer[i] *= right
    steps.push(
      snap(
        12,
        `answer[${i}] *= right → ${before} × ${right} = ${answer[i]}（左积 × 右积）`,
        {
          i,
          pass: 2,
          cellState: { [i]: 'found' },
        },
      ),
    )
    const oldRight = right
    right *= nums[i]
    steps.push(
      snap(13, `right *= nums[${i}] → right = ${oldRight} × ${nums[i]} = ${right}`, {
        i,
        pass: 2,
        cellState: { [i]: 'visited' },
      }),
    )
  }

  steps.push(snap(14, `return answer = [${answer.join(', ')}]`))
  return steps
}

const nums = [1, 2, 3, 4]

const trace: AlgoTrace = {
  title: `示例：nums = [${nums.join(', ')}]`,
  inputLabel: `nums=[${nums.join(',')}]`,
  vizKind: 'array-board',
  vizConfig: { values: nums, label: 'nums' },
  steps: generateSteps(nums),
}

export default [trace]
