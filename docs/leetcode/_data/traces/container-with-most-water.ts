import type { AlgoTrace, TraceStep } from './types'

function generateSteps(height: number[]): TraceStep[] {
  let left = 0, right = height.length - 1
  let best = 0
  const steps: TraceStep[] = []

  const snap = (line: number, note: string, opts: any = {}): TraceStep => {
    const ptrs = [
      { name: 'L', idx: left, color: '#3b82f6' },
      { name: 'R', idx: right, color: '#a855f7' },
    ]
    const cellState: Record<number, string> = {}
    if (opts.foundPair) {
      cellState[left] = 'found'
      cellState[right] = 'found'
    }
    return {
      line,
      note,
      state: {
        pointers: ptrs,
        scalars: [
          { label: 'L', value: String(left) },
          { label: 'R', value: String(right) },
          ...(opts.h !== undefined ? [{ label: 'h(min)', value: String(opts.h) }] : []),
          ...(opts.w !== undefined ? [{ label: 'w', value: String(opts.w) }] : []),
          ...(opts.area !== undefined
            ? [{ label: 'area', value: String(opts.area), cls: opts.area > best ? 'ok' : undefined }]
            : []),
          { label: 'best', value: String(best) },
        ],
        cellState,
        result: best,
      },
    }
  }

  steps.push(snap(2, `初始化对撞双指针：left=0, right=${right}`))
  steps.push(snap(3, '初始化 best = 0'))
  while (left < right) {
    steps.push(snap(4, `进入循环：left=${left} < right=${right}`))
    const hL = height[left], hR = height[right]
    const h = Math.min(hL, hR)
    const w = right - left
    const area = h * w
    steps.push(
      snap(5, `h = min(height[${left}]=${hL}, height[${right}]=${hR}) = ${h}`, { h }),
    )
    steps.push(
      snap(6, `area = h × (R - L) = ${h} × ${w} = ${area}；尝试更新 best`, {
        h, w, area,
      }),
    )
    if (area > best) {
      best = area
      steps.push(
        snap(6, `area=${area} > 旧 best，best = ${best}`, {
          h, w, area, foundPair: true,
        }),
      )
    } else {
      steps.push(
        snap(6, `area=${area} ≤ best=${best}，best 不变`, { h, w, area }),
      )
    }
    if (hL < hR) {
      steps.push(
        snap(7, `height[L]=${hL} < height[R]=${hR}，移动较矮的一侧 → left++`, { h }),
      )
      left++
      steps.push(snap(8, `left = ${left}`))
    } else {
      steps.push(
        snap(9, `height[L]=${hL} ≥ height[R]=${hR}，移动较矮的一侧 → right--`, { h }),
      )
      right--
      steps.push(snap(10, `right = ${right}`))
    }
  }
  steps.push(snap(11, `双指针相遇，return best = ${best}`))
  return steps
}

const height = [1, 8, 6, 2, 5, 4, 8, 3, 7]

const trace: AlgoTrace = {
  title: `示例：height = [${height.join(', ')}]`,
  inputLabel: `height=[${height.join(',')}]`,
  vizKind: 'array-board',
  vizConfig: { values: height, label: 'height' },
  steps: generateSteps(height),
}

export default [trace]
