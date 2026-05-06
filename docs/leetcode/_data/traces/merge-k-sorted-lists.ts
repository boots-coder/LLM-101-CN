import type { AlgoTrace, TraceStep } from './types'

// vizKind = 'linked-list' 多 chain
// chains[0..K-1] 是输入的 K 条链表（已取出的节点用 cls='detached'）
// chains[K]   是「正在拼接的合并结果」
// scalars 显示堆内容（按值排序的 (val, i) 元组列表）
function generateSteps(lists: number[][]): TraceStep[] {
  const steps: TraceStep[] = []
  const K = lists.length

  // 每条链表的「头指针」当前位置（idx 表示第 idx 个节点尚未取走；超过长度即耗尽）
  const heads: number[] = lists.map(() => 0)
  // 输出
  const merged: { listIdx: number; val: number }[] = []
  // 堆：保存 { val, listIdx } —— listIdx 同时充当 tiebreaker
  let heap: { val: number; listIdx: number }[] = []

  function pushHeap(item: { val: number; listIdx: number }) {
    heap.push(item)
    heap.sort((a, b) =>
      a.val !== b.val ? a.val - b.val : a.listIdx - b.listIdx,
    )
  }

  function popHeap(): { val: number; listIdx: number } {
    return heap.shift()!
  }

  type Cls = 'detached' | 'head' | ''
  type ChainObj = {
    label: string
    nodes: { val: any; cls: Cls }[]
    tailArrow: boolean
    tailLabel: string
  }

  function snap(line: number, note: string, opts: any = {}): TraceStep {
    // 输入链表 chains
    const chains: ChainObj[] = lists.map((arr, i) => {
      const nodes = arr.map((v, idx) => {
        let cls: Cls = ''
        if (idx < heads[i]) cls = 'detached' // 已取走
        else if (idx === heads[i] && heads[i] < arr.length) cls = 'head' // 当前候选
        return { val: v as any, cls }
      })
      return {
        label: `lists[${i}]（剩余: ${arr.length - heads[i]}）`,
        nodes,
        tailArrow: false,
        tailLabel: heads[i] >= arr.length ? '✓ 已耗尽' : '',
      }
    })

    // 合并结果 chain
    const mergedNodes: { val: any; cls: Cls }[] =
      merged.length === 0
        ? [{ val: 'dummy', cls: 'head' }]
        : [
            { val: 'dummy', cls: '' },
            ...merged.map((m, idx) => ({
              val: m.val as any,
              cls: (idx === merged.length - 1 ? 'head' : '') as Cls,
            })),
          ]
    chains.push({
      label: '合并结果（dummy → tail）',
      nodes: mergedNodes,
      tailArrow: false,
      tailLabel: '',
    })

    // 指针：在每条输入链表的当前候选节点上标 lists[i]
    const pointers: any[] = []
    lists.forEach((arr, i) => {
      if (heads[i] < arr.length) {
        pointers.push({
          name: `L${i}`,
          chain: i,
          idx: heads[i],
          color: ['#22c55e', '#3b82f6', '#a855f7', '#f59e0b', '#06b6d4'][i % 5],
        })
      }
    })
    // tail 指针在合并 chain 上
    const mergedChainIdx = K
    const tailNodeIdx = merged.length // dummy=0; nodes 长度 = merged.length+1（含 dummy）
    pointers.push({
      name: 'tail',
      chain: mergedChainIdx,
      idx: tailNodeIdx,
      color: '#ef4444',
    })

    return {
      line,
      note,
      state: {
        scalars: [
          { label: 'K', value: String(K) },
          { label: 'heap.size', value: String(heap.length) },
          {
            label: 'heap',
            value:
              heap.length === 0
                ? '空'
                : `[${heap.map((h) => `(${h.val},i=${h.listIdx})`).join(', ')}]`,
          },
          {
            label: 'merged',
            value:
              merged.length === 0
                ? '空'
                : `[${merged.map((m) => m.val).join(',')}]`,
          },
        ],
        chains,
        pointers,
      },
    }
  }

  steps.push(snap(9, `初始化 heap = [], dummy 哨兵, tail = dummy`))

  // 初始入堆
  for (let i = 0; i < K; i++) {
    if (lists[i].length > 0) {
      pushHeap({ val: lists[i][0], listIdx: i })
      steps.push(
        snap(
          14,
          `lists[${i}] 头节点 (val=${lists[i][0]}) 入堆 → heap.size=${heap.length}`,
        ),
      )
    } else {
      steps.push(snap(11, `lists[${i}] 是空，跳过 if node`))
    }
  }

  steps.push(snap(15, `初始入堆完成；进入主循环`))

  // 主循环
  while (heap.length > 0) {
    steps.push(snap(17, `检查 while：heap 非空（${heap.length} 个候选），继续`))
    const top = popHeap()
    const i = top.listIdx
    const v = top.val
    steps.push(
      snap(
        18,
        `heappop → 最小元素 (val=${v}, i=${i}) 来自 lists[${i}]`,
      ),
    )

    // tail.next = node; tail = node;
    merged.push({ listIdx: i, val: v })
    heads[i] += 1
    steps.push(
      snap(
        20,
        `把 val=${v} 接到合并结果尾部，tail 推进到该节点`,
      ),
    )

    // 把 lists[i] 的下一节点入堆
    if (heads[i] < lists[i].length) {
      const nv = lists[i][heads[i]]
      pushHeap({ val: nv, listIdx: i })
      steps.push(
        snap(
          22,
          `lists[${i}] 还有下一个 (val=${nv})，heappush → heap.size=${heap.length}`,
        ),
      )
    } else {
      steps.push(
        snap(
          21,
          `lists[${i}] 已耗尽（node.next is None），不再入堆 → heap.size=${heap.length}`,
        ),
      )
    }
  }

  steps.push(
    snap(
      23,
      `heap 空，循环结束。return dummy.next → 合并结果 [${merged
        .map((m) => m.val)
        .join(',')}]`,
    ),
  )
  return steps
}

const lists: number[][] = [
  [1, 4, 5],
  [1, 3, 4],
  [2, 6],
]

const trace: AlgoTrace = {
  title: `示例：lists = ${JSON.stringify(lists)}`,
  inputLabel: `lists=${JSON.stringify(lists)}`,
  vizKind: 'linked-list',
  vizConfig: {},
  steps: generateSteps(lists),
}

export default [trace]
