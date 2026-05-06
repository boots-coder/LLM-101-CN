import type { AlgoTrace, TraceStep } from './types'

// 注意：本 trace 演示「双指针 + leftMax / rightMax」O(1) 空间解法，
// 与 problems 文件里的单调栈解法是同题的两种写法，可对照学习。
// 行号对应下面这段参考代码：
//   1: def trap(height: list[int]) -> int:
//   2:     l, r = 0, len(height) - 1
//   3:     left_max = right_max = 0
//   4:     total = 0
//   5:     while l < r:
//   6:         if height[l] < height[r]:
//   7:             if height[l] >= left_max:
//   8:                 left_max = height[l]
//   9:             else:
//  10:                 total += left_max - height[l]
//  11:             l += 1
//  12:         else:
//  13:             if height[r] >= right_max:
//  14:                 right_max = height[r]
//  15:             else:
//  16:                 total += right_max - height[r]
//  17:             r -= 1
//  18:     return total

function generateSteps(height: number[]): TraceStep[] {
  const steps: TraceStep[] = []
  const n = height.length
  let l = 0
  let r = n - 1
  let leftMax = 0
  let rightMax = 0
  let total = 0

  const snap = (line: number, note: string, opts: any = {}): TraceStep => {
    const ptrs: any[] = []
    if (l <= r) {
      ptrs.push({ name: 'L', idx: l, color: '#3b82f6' })
      ptrs.push({ name: 'R', idx: r, color: '#a855f7' })
    } else {
      // 已收敛，仍把指针停在最后位置便于阅读
      ptrs.push({ name: 'L', idx: Math.min(l, n - 1), color: '#3b82f6' })
      ptrs.push({ name: 'R', idx: Math.max(r, 0), color: '#a855f7' })
    }
    return {
      line,
      note,
      state: {
        pointers: ptrs,
        scalars: [
          { label: 'leftMax', value: String(leftMax) },
          { label: 'rightMax', value: String(rightMax) },
          { label: 'total', value: String(total) },
          ...(opts.extra ?? []),
        ],
        cellState: opts.cellState ?? {},
        result: total,
      },
    }
  }

  steps.push(snap(2, `初始化 L=${l}, R=${r}`))
  steps.push(snap(3, '初始化 leftMax = rightMax = 0'))
  steps.push(snap(4, '初始化 total = 0'))

  while (l < r) {
    steps.push(snap(5, `进入循环：L=${l} < R=${r}`))
    const hl = height[l]
    const hr = height[r]
    if (hl < hr) {
      steps.push(
        snap(6, `height[L]=${hl} < height[R]=${hr} → 处理较矮的左侧（右墙更高，水位由左侧 leftMax 决定）`),
      )
      if (hl >= leftMax) {
        leftMax = hl
        steps.push(snap(8, `height[L]=${hl} ≥ leftMax → 更新 leftMax = ${leftMax}`))
      } else {
        const add = leftMax - hl
        total += add
        steps.push(
          snap(10, `height[L]=${hl} < leftMax=${leftMax} → total += ${leftMax} - ${hl} = ${add}`, {
            cellState: { [l]: 'found' },
          }),
        )
      }
      l++
      steps.push(snap(11, `L 右移 → L=${l}`))
    } else {
      steps.push(
        snap(12, `height[L]=${hl} ≥ height[R]=${hr} → 处理较矮的右侧（左墙更高，水位由 rightMax 决定）`),
      )
      if (hr >= rightMax) {
        rightMax = hr
        steps.push(snap(14, `height[R]=${hr} ≥ rightMax → 更新 rightMax = ${rightMax}`))
      } else {
        const add = rightMax - hr
        total += add
        steps.push(
          snap(16, `height[R]=${hr} < rightMax=${rightMax} → total += ${rightMax} - ${hr} = ${add}`, {
            cellState: { [r]: 'found' },
          }),
        )
      }
      r--
      steps.push(snap(17, `R 左移 → R=${r}`))
    }
  }

  steps.push(snap(18, `L=${l} ≥ R=${r}，循环结束，return total = ${total}`))
  return steps
}

const height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]

const trace: AlgoTrace = {
  title: `示例：height = [${height.join(', ')}]（双指针解法）`,
  inputLabel: `height=[${height.join(',')}]`,
  vizKind: 'array-board',
  vizConfig: { values: height, label: 'height' },
  steps: generateSteps(height),
}

export default [trace]
