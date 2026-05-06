import type { AlgoTrace, TraceStep } from './types'

// 行号对应 problems/largest-rectangle-histogram.ts 里的 solutionCode：
//   1: def largestRectangleArea(heights: list[int]) -> int:
//   2:     # 两端各加一个高度为 0 的哨兵，省去边界判断
//   3:     heights = [0] + heights + [0]
//   4:     stack = []  # 单调递增栈，存下标
//   5:     best = 0
//   6:     for i, h in enumerate(heights):
//   7:         while stack and heights[stack[-1]] > h:
//   8:             top = stack.pop()
//   9:             # 左边界：栈顶；右边界：i
//  10:             width = i - stack[-1] - 1
//  11:             best = max(best, heights[top] * width)
//  12:         stack.append(i)
//  13:     return best

function generateSteps(rawHeights: number[]): TraceStep[] {
  const steps: TraceStep[] = []
  const heights = [0, ...rawHeights, 0] // 加哨兵后的数组（也用作可视化主数组）
  const n = heights.length
  const stack: number[] = []
  let best = 0

  const snap = (
    line: number,
    note: string,
    opts: {
      current?: number
      popped?: number
      poppingTop?: boolean
      width?: number
      area?: number
    } = {},
  ): TraceStep => {
    const cs: Record<number, string> = {}
    for (const idx of stack) cs[idx] = 'in-stack'
    if (opts.current !== undefined) cs[opts.current] = 'current'
    if (opts.popped !== undefined) cs[opts.popped] = 'popped'
    return {
      line,
      note,
      state: {
        pointers:
          opts.current !== undefined
            ? [{ name: 'i', idx: opts.current, color: '#22c55e' }]
            : [],
        scalars: [
          { label: 'best', value: String(best) },
          ...(opts.current !== undefined
            ? [{ label: 'h', value: String(heights[opts.current]) }]
            : []),
          ...(opts.width !== undefined ? [{ label: 'width', value: String(opts.width) }] : []),
          ...(opts.area !== undefined ? [{ label: 'area', value: String(opts.area) }] : []),
          { label: 'stack', value: `[${stack.join(', ')}]` },
        ],
        cellState: cs,
        stackLabel: '单调递增栈（存下标，顶=最近入栈）',
        stack: stack.map((idx) => `${idx}(h=${heights[idx]})`),
        poppingTop: opts.poppingTop ?? false,
        result: best,
      },
    }
  }

  steps.push(snap(3, `两端加 0 哨兵，heights = [${heights.join(', ')}]`))
  steps.push(snap(4, '初始化 stack = []（单调递增）'))
  steps.push(snap(5, '初始化 best = 0'))

  for (let i = 0; i < n; i++) {
    const h = heights[i]
    steps.push(snap(6, `进入循环 i=${i}, h=heights[${i}]=${h}`, { current: i }))

    while (stack.length > 0 && heights[stack[stack.length - 1]] > h) {
      const topIdx = stack[stack.length - 1]
      steps.push(
        snap(
          7,
          `栈顶 ${topIdx}, heights[${topIdx}]=${heights[topIdx]} > h=${h} → 进入弹栈结算`,
          { current: i },
        ),
      )
      stack.pop()
      steps.push(
        snap(8, `top = stack.pop() = ${topIdx}（高=${heights[topIdx]}）`, {
          current: i,
          popped: topIdx,
          poppingTop: true,
        }),
      )
      const leftIdx = stack.length === 0 ? -1 : stack[stack.length - 1]
      const width = i - leftIdx - 1
      const area = heights[topIdx] * width
      steps.push(
        snap(
          10,
          `width = i - stack[-1] - 1 = ${i} - ${leftIdx} - 1 = ${width}` +
            (stack.length === 0 ? '（左侧无更矮，靠首部 0 哨兵兜底）' : ''),
          { current: i, popped: topIdx, width },
        ),
      )
      const newBest = Math.max(best, area)
      steps.push(
        snap(
          11,
          `area = heights[${topIdx}] × width = ${heights[topIdx]} × ${width} = ${area}` +
            (newBest > best ? ` → best 更新为 ${newBest}` : `（best 仍为 ${best}）`),
          { current: i, popped: topIdx, width, area },
        ),
      )
      best = newBest
    }
    stack.push(i)
    steps.push(snap(12, `stack.push(${i})（无更矮可弹，入栈维持递增）`, { current: i }))
  }

  steps.push(snap(13, `循环结束，return best = ${best}`))
  return steps
}

const heights = [2, 1, 5, 6, 2, 3]

const trace: AlgoTrace = {
  title: `示例：heights = [${heights.join(', ')}]（含首尾 0 哨兵）`,
  inputLabel: `heights=[${heights.join(',')}]`,
  vizKind: 'stack-board',
  // 主数组直接展示加了哨兵的版本，便于和 stack 里的下标对齐
  vizConfig: { values: [0, ...heights, 0], label: 'heights（已加首尾 0 哨兵）' },
  steps: generateSteps(heights),
}

export default [trace]
