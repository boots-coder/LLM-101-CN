import type { AlgoTrace, TraceStep } from './types'

// vizKind = 'linked-list' 单 chain
// 节点按原始下标 0..n-1 排列；环用 cls='cycle' 高亮环上节点；tailLabel 标注「→ 节点 X」表示尾连回
function generateSteps(values: number[], pos: number): TraceStep[] {
  const steps: TraceStep[] = []
  const n = values.length
  // next[i] = 下一节点的下标，-1 = null。pos = -1 表示无环；否则末尾连回 pos
  const next: number[] = values.map((_, i) => (i === n - 1 ? -1 : i + 1))
  if (pos >= 0) next[n - 1] = pos

  const cycleSet = new Set<number>()
  if (pos >= 0) {
    let p = pos
    do {
      cycleSet.add(p)
      p = next[p]
    } while (p !== -1 && p !== pos)
  }

  let slow = 0
  let fast = 0
  let phase: 1 | 2 = 1
  let entry: number = -1

  function snap(line: number, note: string, opts: any = {}): TraceStep {
    const nodes = values.map((v, i) => ({
      val: v,
      cls: cycleSet.has(i) ? 'cycle' as const : '' as const,
    }))
    const pointers: any[] = []
    if (slow >= 0) pointers.push({ name: 'slow', chain: 0, idx: slow, color: '#22c55e' })
    if (fast >= 0) pointers.push({ name: 'fast', chain: 0, idx: fast, color: '#ef4444' })
    if (opts.entryMark && entry >= 0) {
      // 用 head cls 标记入口
      nodes[entry] = { val: values[entry], cls: 'head' as const }
    }

    return {
      line,
      note,
      state: {
        scalars: [
          { label: 'phase', value: phase === 1 ? '① 找相遇点' : '② 找入口' },
          {
            label: 'slow',
            value: slow === -1 ? 'None' : `idx=${slow}, val=${values[slow]}`,
          },
          {
            label: 'fast',
            value: fast === -1 ? 'None' : `idx=${fast}, val=${values[fast]}`,
          },
          ...(entry >= 0
            ? [{ label: 'entry', value: `idx=${entry}, val=${values[entry]}` }]
            : []),
        ],
        chains: [
          {
            label:
              pos >= 0
                ? `节点（按原始下标；尾节点 idx=${n - 1} 连回 idx=${pos}）`
                : '节点（无环）',
            nodes,
            tailArrow: pos >= 0,
            tailLabel: pos >= 0 ? `↺ 回到 idx=${pos}` : 'None',
          },
        ],
        pointers,
      },
    }
  }

  steps.push(snap(6, `初始化 slow = fast = head（idx=0, val=${values[0]}）`))

  // 阶段一
  let met = false
  while (true) {
    // 检查 while 条件
    if (fast === -1 || next[fast] === -1) {
      steps.push(
        snap(
          8,
          `检查 while：fast ${fast === -1 ? '是 None' : `idx=${fast}`}${
            fast !== -1 && next[fast] === -1 ? '，fast.next 是 None' : ''
          } → 条件假，退出`,
        ),
      )
      steps.push(snap(13, 'else 分支：链表无环，return None'))
      return steps
    }
    steps.push(
      snap(8, `检查 while：fast 与 fast.next 都非 None，继续`),
    )
    slow = next[slow]
    steps.push(snap(9, `slow = slow.next → idx=${slow}, val=${values[slow]}`))
    fast = next[next[fast]]
    steps.push(
      snap(
        10,
        `fast = fast.next.next → ${
          fast === -1 ? '到 None' : `idx=${fast}, val=${values[fast]}`
        }`,
      ),
    )
    if (slow === fast) {
      steps.push(
        snap(
          11,
          `slow is fast！(都在 idx=${slow}, val=${values[slow]}) → break`,
        ),
      )
      met = true
      break
    } else {
      steps.push(snap(11, `slow ≠ fast，继续循环`))
    }
  }

  if (!met) return steps

  // 阶段二
  phase = 2
  steps.push(snap(19, `进入阶段 ② —— slow 回到 head（idx=0, val=${values[0]}）`))
  slow = 0
  steps.push(snap(19, `slow = head → idx=0, val=${values[0]}`))

  while (slow !== fast) {
    steps.push(
      snap(
        20,
        `检查 while：slow(idx=${slow}) ≠ fast(idx=${fast})，继续同速前进`,
      ),
    )
    slow = next[slow]
    steps.push(snap(21, `slow = slow.next → idx=${slow}, val=${values[slow]}`))
    fast = next[fast]
    steps.push(snap(22, `fast = fast.next → idx=${fast}, val=${values[fast]}`))
  }

  entry = slow
  steps.push(
    snap(
      20,
      `slow is fast！都在 idx=${slow}（val=${values[slow]}）→ 这就是环入口`,
      { entryMark: true },
    ),
  )
  steps.push(
    snap(
      23,
      `return slow → 环入口节点 idx=${entry}, val=${values[entry]}`,
      { entryMark: true },
    ),
  )
  return steps
}

const values = [3, 2, 0, -4]
const pos = 1

const trace: AlgoTrace = {
  title: `示例：head = [${values.join(',')}], pos = ${pos}（尾节点 ${values[values.length - 1]} 连回 idx=${pos} 即值 ${values[pos]}）`,
  inputLabel: `head=[${values.join(',')}], pos=${pos}`,
  vizKind: 'linked-list',
  vizConfig: {},
  steps: generateSteps(values, pos),
}

export default [trace]
