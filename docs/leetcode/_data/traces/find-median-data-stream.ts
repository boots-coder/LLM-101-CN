import type { AlgoTrace, TraceStep } from './types'

// 极简最小堆（与 kth-largest-element.ts 一致风格）
function heapPush(h: number[], x: number) {
  h.push(x)
  let i = h.length - 1
  while (i > 0) {
    const p = (i - 1) >> 1
    if (h[p] <= h[i]) break
    ;[h[p], h[i]] = [h[i], h[p]]
    i = p
  }
}
function heapPop(h: number[]): number {
  const top = h[0]
  const last = h.pop() as number
  if (h.length) {
    h[0] = last
    let i = 0
    const n = h.length
    while (true) {
      const l = 2 * i + 1, r = 2 * i + 2
      let s = i
      if (l < n && h[l] < h[s]) s = l
      if (r < n && h[r] < h[s]) s = r
      if (s === i) break
      ;[h[s], h[i]] = [h[i], h[s]]
      i = s
    }
  }
  return top
}

function generateSteps(stream: number[]): TraceStep[] {
  // low: 大根堆（用负数装较小一半）；high: 小根堆（装较大一半）
  const low: number[] = []
  const high: number[] = []
  const steps: TraceStep[] = []

  // 把 low 还原成「真实数值」展示（取负），并按 low 顶在前的顺序
  const lowReal = (): number[] => low.map((v) => -v)

  const findMedian = (): number => {
    if (low.length > high.length) return -low[0]
    return (-low[0] + high[0]) / 2
  }

  const snap = (line: number, note: string, opts: any = {}): TraceStep => {
    const total = low.length + high.length
    const med = total > 0 ? findMedian() : null
    return {
      line,
      note,
      state: {
        // 顶部数组：把数据流可视化（已加入的元素 + 待加入指针）
        pointers: opts.i !== undefined
          ? [{ name: 'i', idx: opts.i, color: '#22c55e' }]
          : [],
        cellState: opts.cellState ?? {},
        scalars: [
          { label: 'low.size', value: String(low.length) },
          { label: 'high.size', value: String(high.length) },
          ...(low.length > 0 ? [{ label: '-low[0]', value: String(-low[0]) }] : []),
          ...(high.length > 0 ? [{ label: 'high[0]', value: String(high[0]) }] : []),
          ...(med !== null ? [{ label: 'median', value: String(med) }] : []),
        ],
        // 用栈区一次显示两个堆：上面 high（小根堆，顶=最小），下面 low（大根堆，顶=最大）
        stackLabel: '双堆 [high 小根堆 | low 大根堆]',
        stack: [
          ...high.map((v) => `H: ${v}`),
          '── 中位线 ──',
          ...lowReal().map((v) => `L: ${v}`),
        ],
      },
    }
  }

  steps.push(snap(4, '初始化：low 大根堆（存负数实现）、high 小根堆，全空'))

  for (let i = 0; i < stream.length; i++) {
    const num = stream[i]
    steps.push(snap(10, `addNum(${num}) 入参，开始处理第 ${i + 1} 个数`, { i }))

    if (low.length === high.length) {
      steps.push(snap(12, `两堆等长（${low.length}=${high.length}），走 if 分支：先入 high 中转`, { i }))
      heapPush(high, num)
      steps.push(snap(14, `heappush(high, ${num})，high 顶=${high[0]}（这是 high∪{${num}} 的最小者）`, { i }))
      const mv = heapPop(high)
      heapPush(low, -mv)
      steps.push(snap(15, `heappop(high)=${mv}，再 heappush(low, -${mv})——把"小一半的最大"塞进 low`, { i }))
    } else {
      steps.push(snap(16, `低堆多 1（${low.length}>${high.length}），走 else 分支：先入 low 中转`, { i }))
      heapPush(low, -num)
      steps.push(snap(17, `heappush(low, -${num})，low 真实顶=${-low[0]}（low∪{${num}} 的最大者）`, { i }))
      const mv = -heapPop(low)
      heapPush(high, mv)
      steps.push(snap(18, `heappop(low)=${-(-mv)}（还原=${mv}），heappush(high, ${mv})——把"大一半的最小"塞进 high`, { i }))
    }

    const total = low.length + high.length
    const med = findMedian()
    steps.push(snap(23, `findMedian() → ${total % 2 === 1 ? `low 多一个，return -low[0] = ${med}` : `两堆等长，return (${-low[0]}+${high[0]})/2 = ${med}`}`, {
      i,
    }))
  }

  steps.push(snap(23, `所有元素加入完毕，最终中位数 = ${findMedian()}`))

  return steps
}

const stream = [1, 2, 3, 4, 5, 6, 7]

const trace: AlgoTrace = {
  title: `示例：流入 [${stream.join(', ')}]`,
  inputLabel: `stream=[${stream.join(',')}]`,
  vizKind: 'stack-board',
  vizConfig: { values: stream, label: '数据流（按顺序 addNum）' },
  steps: generateSteps(stream),
}

export default [trace]
