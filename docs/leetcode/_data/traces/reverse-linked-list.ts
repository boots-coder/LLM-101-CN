import type { AlgoTrace, TraceStep } from './types'

// chains[0] 是「原始链表 + prev/curr/nxt 指针」
// chains[1] 是「已反转部分」（视觉上方便辨认）
function generateSteps(values: number[]): TraceStep[] {
  const steps: TraceStep[] = []
  const n = values.length
  // links[i] 表示节点 i 当前指向哪个节点的下标，-1 = null
  const links: number[] = values.map((_, i) => (i === n - 1 ? -1 : i + 1))
  let prev = -1
  let curr = 0
  let nxt = -1

  function chain() {
    // 还原"按链接顺序"的节点排列，便于 viz：从原始下标 0..n-1 直接显示
    const nodes = values.map((v, i) => ({
      val: v,
      cls: i === prev ? 'head' : (i < prev || (prev !== -1 && i === prev) ? 'detached' : ''),
    }))
    return { label: '节点（按原始下标）', nodes, tailArrow: false, tailLabel: '' }
  }

  const snap = (line: number, note: string): TraceStep => ({
    line,
    note,
    state: {
      scalars: [
        { label: 'prev', value: prev === -1 ? 'None' : String(values[prev]) },
        { label: 'curr', value: curr === -1 ? 'None' : String(values[curr]) },
        { label: 'nxt',  value: nxt === -1 ? 'None'  : String(values[nxt]) },
        { label: 'links', value:
          values.map((v, i) => `${v}→${links[i] === -1 ? '∅' : values[links[i]]}`).join('  ') },
      ],
      chains: [
        {
          label: '链表节点',
          nodes: chain().nodes,
          tailArrow: false,
        },
      ],
      pointers: [
        ...(prev !== -1 ? [{ name: 'prev', chain: 0, idx: prev, color: '#a855f7' }] : []),
        ...(curr !== -1 ? [{ name: 'curr', chain: 0, idx: curr, color: '#22c55e' }] : []),
        ...(nxt !== -1  ? [{ name: 'nxt',  chain: 0, idx: nxt,  color: '#3b82f6' }] : []),
      ],
    },
  })

  steps.push(snap(6, `初始化 prev=None, curr=head（值 ${values[0]}）`))
  while (curr !== -1) {
    nxt = links[curr]
    steps.push(snap(8, `1. nxt = curr.next（值 ${nxt === -1 ? 'None' : values[nxt]}）`))
    links[curr] = prev
    steps.push(snap(9, `2. curr.next = prev → 节点 ${values[curr]} 现在指向 ${prev === -1 ? 'None' : values[prev]}`))
    prev = curr
    steps.push(snap(10, `3. prev = curr → prev 移到 ${values[prev]}`))
    curr = nxt
    nxt = -1
    steps.push(snap(11, `4. curr = nxt → curr ${curr === -1 ? '到 None，循环将退出' : '移到 ' + values[curr]}`))
  }
  steps.push(snap(12, `循环结束，return prev（新头节点 ${values[prev]}）`))
  return steps
}

const values = [1, 2, 3, 4, 5]

const trace: AlgoTrace = {
  title: `示例：head = ${values.join(' → ')} → None`,
  inputLabel: `head = ${values.join('→')}`,
  vizKind: 'linked-list',
  vizConfig: {},
  steps: generateSteps(values),
}

export default [trace]
