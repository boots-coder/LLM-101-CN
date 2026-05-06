import type { AlgoTrace, TraceStep } from './types'

// 极简最小堆实现（仅 push / pop / peek / pushpop），不依赖外部库
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
function heapPushPop(h: number[], x: number): number {
  if (h.length && h[0] < x) {
    const r = h[0]
    h[0] = x
    let i = 0
    const n = h.length
    while (true) {
      const l = 2 * i + 1, rr = 2 * i + 2
      let s = i
      if (l < n && h[l] < h[s]) s = l
      if (rr < n && h[rr] < h[s]) s = rr
      if (s === i) break
      ;[h[s], h[i]] = [h[i], h[s]]
      i = s
    }
    return r
  }
  return x
}

function generateSteps(nums: number[], k: number): TraceStep[] {
  const heap: number[] = []
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
        { label: 'heap.size', value: String(heap.length) },
        ...(heap.length > 0 ? [{ label: 'heap[0]', value: String(heap[0]) }] : []),
      ],
      cellState: opts.cellState ?? {},
      stackLabel: '最小堆（顶=最小）',
      stack: [...heap],
    },
  })

  steps.push(snap(5, '初始化空堆 heap = []'))
  for (let i = 0; i < nums.length; i++) {
    const x = nums[i]
    steps.push(snap(6, `进入 i=${i}, x=${x}`, { i }))
    if (heap.length < k) {
      heapPush(heap, x)
      steps.push(snap(8, `堆未满（${heap.length}/${k}），heappush(${x})`, { i }))
    } else if (x > heap[0]) {
      const out = heapPushPop(heap, x)
      steps.push(snap(10, `x=${x} > 堆顶 ${out}，pushpop：踢掉 ${out}, 推入 ${x}`, { i }))
    } else {
      steps.push(snap(9, `x=${x} ≤ 堆顶 ${heap[0]}，跳过`, { i, cellState: { [i]: 'skipped' } }))
    }
  }
  steps.push(snap(11, `return heap[0] = ${heap[0]}（即第 ${k} 大）`))
  return steps
}

const nums = [3, 2, 1, 5, 6, 4]
const k = 2

const trace: AlgoTrace = {
  title: `示例：nums = [${nums.join(', ')}], k = ${k}`,
  inputLabel: `nums=[${nums.join(',')}], k=${k}`,
  vizKind: 'stack-board',
  vizConfig: { values: nums, label: 'nums' },
  steps: generateSteps(nums, k),
}

export default [trace]
