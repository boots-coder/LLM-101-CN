import type { AlgoTrace, TraceStep } from './types'

interface Node { id: string; val: number; left?: Node | null; right?: Node | null }

function buildTree(): Node {
  // 示例：     3
  //          / \
  //         9   20
  //            /  \
  //          15    7
  return {
    id: 'r', val: 3,
    left:  { id: 'l', val: 9, left: null, right: null },
    right: {
      id: 'rr', val: 20,
      left:  { id: 'rl', val: 15, left: null, right: null },
      right: { id: 'rrr', val: 7, left: null, right: null },
    },
  }
}

function generateSteps(root: Node): TraceStep[] {
  const res: number[][] = []
  const q: Node[] = [root]
  const steps: TraceStep[] = []
  const visited: string[] = []
  let currentId: string | null = null

  const snap = (line: number, note: string, opts: any = {}): TraceStep => ({
    line,
    note,
    state: {
      currentId,
      visitedIds: [...visited],
      queue: q.map((n) => String(n.val)),
      scalars: [
        { label: '|queue|', value: String(q.length) },
        ...(opts.size !== undefined ? [{ label: 'size', value: String(opts.size) }] : []),
        ...(opts.level ? [{ label: '当前层', value: '[' + opts.level.join(', ') + ']' }] : []),
      ],
      output: res.map((r) => '[' + r.join(',') + ']').join(' '),
    },
  })

  steps.push(snap(11, `初始化 q = deque([root=${root.val}])`))
  while (q.length > 0) {
    const size = q.length
    const level: number[] = []
    steps.push(snap(13, `进入新一层：size = ${size}`, { size }))
    for (let i = 0; i < size; i++) {
      const node = q.shift() as Node
      currentId = node.id
      steps.push(snap(16, `q.popleft() → ${node.val}`, { size, level: [...level] }))
      level.push(node.val)
      visited.push(node.id)
      steps.push(snap(17, `level.append(${node.val})`, { size, level: [...level] }))
      if (node.left) {
        q.push(node.left)
        steps.push(snap(19, `左孩子 ${node.left.val} 入队`, { size, level: [...level] }))
      }
      if (node.right) {
        q.push(node.right)
        steps.push(snap(21, `右孩子 ${node.right.val} 入队`, { size, level: [...level] }))
      }
    }
    res.push(level)
    steps.push(snap(22, `本层结束：res.append([${level.join(', ')}])`, { size, level }))
    currentId = null
  }
  steps.push(snap(23, `q 空，return res = [${res.map((r) => '[' + r.join(',') + ']').join(', ')}]`))
  return steps
}

const root = buildTree()

const trace: AlgoTrace = {
  title: '示例：root = [3, 9, 20, null, null, 15, 7]',
  inputLabel: 'tree=[3,9,20,#,#,15,7]',
  vizKind: 'tree',
  vizConfig: { tree: root },
  steps: generateSteps(root),
}

export default [trace]
