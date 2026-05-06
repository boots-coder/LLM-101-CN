import type { AlgoTrace, TraceStep } from './types'

// 行号对应 problems/top-k-frequent.ts 里的 solutionCode：
//   1: import heapq
//   2: from collections import Counter
//   3:
//   4: def topKFrequent(nums, k) -> list[int]:
//   5:     count = Counter(nums)
//   6:     heap = []
//   7:     for num, freq in count.items():
//   8:         if len(heap) < k:
//   9:             heappush(heap, (freq, num))
//  10:         elif freq > heap[0][0]:
//  11:             heappushpop(heap, (freq, num))
//  12:     return [num for freq, num in heap]

// 极简最小堆，堆元素是 [freq, num] 元组，按字典序比较（先 freq 后 num）
type Item = [number, number]

function lt(a: Item, b: Item): boolean {
  if (a[0] !== b[0]) return a[0] < b[0]
  return a[1] < b[1]
}
function heapPush(h: Item[], x: Item) {
  h.push(x)
  let i = h.length - 1
  while (i > 0) {
    const p = (i - 1) >> 1
    if (!lt(h[i], h[p])) break
    ;[h[p], h[i]] = [h[i], h[p]]
    i = p
  }
}
function siftDown(h: Item[]) {
  let i = 0
  const n = h.length
  while (true) {
    const l = 2 * i + 1
    const r = 2 * i + 2
    let s = i
    if (l < n && lt(h[l], h[s])) s = l
    if (r < n && lt(h[r], h[s])) s = r
    if (s === i) break
    ;[h[s], h[i]] = [h[i], h[s]]
    i = s
  }
}
function heapPushPop(h: Item[], x: Item): Item {
  if (h.length && lt(h[0], x)) {
    const r = h[0]
    h[0] = x
    siftDown(h)
    return r
  }
  return x
}

function fmtItem(x: Item): string {
  return `(f=${x[0]}, n=${x[1]})`
}

function generateSteps(nums: number[], k: number): TraceStep[] {
  const steps: TraceStep[] = []
  const heap: Item[] = []
  const count = new Map<number, number>()

  const snap = (
    line: number,
    note: string,
    opts: {
      i?: number
      cellState?: Record<number, string>
      poppingTop?: boolean
      currentItem?: Item
    } = {},
  ): TraceStep => {
    const ptrs: any[] = []
    if (opts.i !== undefined && opts.i >= 0 && opts.i < nums.length) {
      ptrs.push({ name: 'i', idx: opts.i, color: '#22c55e' })
    }
    const top = heap[0]
    return {
      line,
      note,
      state: {
        pointers: ptrs,
        scalars: [
          { label: 'k', value: String(k) },
          { label: 'heap.size', value: String(heap.length) },
          ...(top ? [{ label: 'heap.top', value: fmtItem(top) }] : []),
          ...(opts.currentItem ? [{ label: 'cur', value: fmtItem(opts.currentItem) }] : []),
          {
            label: 'count',
            value:
              '{' +
              [...count.entries()].map(([n, f]) => `${n}:${f}`).join(', ') +
              '}',
          },
        ],
        cellState: opts.cellState ?? {},
        stackLabel: '最小堆（顶=最不热门，元素=(freq,num)）',
        stack: heap.map(fmtItem),
        poppingTop: opts.poppingTop ?? false,
      },
    }
  }

  steps.push(snap(4, '函数入口：topKFrequent(nums, k)'))
  steps.push(snap(5, '步骤一：用 Counter(nums) 统计每个值的频次'))
  steps.push(snap(5, '初始化 count = {}（空字典）'))
  // 演示 Counter 的扫描过程
  for (let i = 0; i < nums.length; i++) {
    const x = nums[i]
    count.set(x, (count.get(x) ?? 0) + 1)
    steps.push(
      snap(5, `扫到 nums[${i}]=${x} → count[${x}] = ${count.get(x)}`, {
        i,
        cellState: { [i]: 'current' },
      }),
    )
  }
  steps.push(snap(5, `统计完毕：count = {${[...count.entries()].map(([n, f]) => `${n}:${f}`).join(', ')}}`))
  steps.push(snap(5, `unique 值有 ${count.size} 个；接下来用堆筛出 freq 最大的 k=${k} 个`))

  const entries: [number, number][] = [] // [num, freq]
  for (const [num, freq] of count.entries()) entries.push([num, freq])

  steps.push(snap(6, '步骤二：用大小为 k 的最小堆维护「迄今最热门的 k 个值」'))
  steps.push(snap(6, '初始化空堆 heap = []（按 (freq, num) 元组字典序比较，堆顶=最不热门）'))

  for (let idx = 0; idx < entries.length; idx++) {
    const [num, freq] = entries[idx]
    // 高亮 nums 中第一次出现该 num 的下标，便于关联
    const firstIdx = nums.indexOf(num)
    const item: Item = [freq, num]

    steps.push(
      snap(7, `--- 第 ${idx + 1}/${entries.length} 项 --- 从 count 取 (num=${num}, freq=${freq})`, {
        i: firstIdx,
        currentItem: item,
        cellState: { [firstIdx]: 'current' },
      }),
    )
    steps.push(
      snap(7, `构造堆元素 item = (freq, num) = ${fmtItem(item)}（freq 在前，便于按频次比较）`, {
        i: firstIdx,
        currentItem: item,
        cellState: { [firstIdx]: 'current' },
      }),
    )

    if (heap.length < k) {
      steps.push(
        snap(8, `判断 len(heap)=${heap.length} < k=${k}：✓ 堆未满`, {
          i: firstIdx,
          currentItem: item,
          cellState: { [firstIdx]: 'current' },
        }),
      )
      heapPush(heap, item)
      steps.push(
        snap(9, `heappush ${fmtItem(item)}，堆现在大小=${heap.length}`, {
          i: firstIdx,
          currentItem: item,
          cellState: { [firstIdx]: 'in-stack' },
        }),
      )
    } else {
      steps.push(
        snap(8, `判断 len(heap)=${heap.length} < k=${k}：✗ 堆已满，看是否替换`, {
          i: firstIdx,
          currentItem: item,
          cellState: { [firstIdx]: 'current' },
        }),
      )
      if (freq > heap[0][0]) {
        const oldTop = heap[0]
        steps.push(
          snap(
            10,
            `freq=${freq} > 堆顶 freq=${oldTop[0]} → 应该踢掉堆顶`,
            {
              i: firstIdx,
              currentItem: item,
              cellState: { [firstIdx]: 'current' },
              poppingTop: true,
            },
          ),
        )
        const out = heapPushPop(heap, item)
        steps.push(
          snap(
            11,
            `heappushpop：弹出 ${fmtItem(out)}，推入 ${fmtItem(item)}`,
            {
              i: firstIdx,
              currentItem: item,
              cellState: { [firstIdx]: 'in-stack' },
            },
          ),
        )
      } else {
        steps.push(
          snap(10, `freq=${freq} ≤ 堆顶 freq=${heap[0][0]} → 跳过（不进堆）`, {
            i: firstIdx,
            currentItem: item,
            cellState: { [firstIdx]: 'popped' },
          }),
        )
      }
    }
  }

  steps.push(snap(7, `count 遍历结束，堆中保留了 freq 最大的 ${heap.length} 个值`))
  steps.push(snap(12, '步骤三：从堆里提取每个元组的 num（题目允许任意顺序）'))
  const result: number[] = []
  for (let i = 0; i < heap.length; i++) {
    result.push(heap[i][1])
    steps.push({
      line: 12,
      note: `从堆里取出 ${fmtItem(heap[i])}，result.push(${heap[i][1]}) → result = [${result.join(', ')}]`,
      state: {
        scalars: [
          { label: 'k', value: String(k) },
          { label: 'heap.size', value: String(heap.length) },
          { label: 'result', value: `[${result.join(', ')}]` },
        ],
        cellState: {},
        stackLabel: '最小堆（顶=最不热门，元素=(freq,num)）',
        stack: heap.map(fmtItem),
      },
    })
  }
  steps.push({
    line: 12,
    note: `return [${result.join(', ')}]`,
    state: {
      scalars: [
        { label: 'k', value: String(k) },
        { label: 'result', value: `[${result.join(', ')}]` },
      ],
      cellState: {},
      stackLabel: '最小堆',
      stack: heap.map(fmtItem),
    },
  })
  return steps
}

const nums = [1, 1, 1, 2, 2, 3]
const k = 2

const trace: AlgoTrace = {
  title: `示例：nums = [${nums.join(', ')}], k = ${k}`,
  inputLabel: `nums=[${nums.join(',')}], k=${k}`,
  vizKind: 'stack-board',
  vizConfig: { values: nums, label: 'nums' },
  steps: generateSteps(nums, k),
}

export default [trace]
