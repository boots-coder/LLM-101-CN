import type { AlgoTrace, TraceStep } from './types'

interface Node { id: string; val: number; left?: Node | null; right?: Node | null }

function buildTree(): Node {
  // 示例：     1
  //          / \
  //         2   3
  //          \   \
  //           5   4
  // 右视图：  [1, 3, 4]
  return {
    id: 'r', val: 1,
    left: {
      id: 'l', val: 2,
      left: null,
      right: { id: 'lr', val: 5, left: null, right: null },
    },
    right: {
      id: 'rr', val: 3,
      left: null,
      right: { id: 'rrr', val: 4, left: null, right: null },
    },
  }
}

function generateSteps(root: Node): TraceStep[] {
  const steps: TraceStep[] = []
  const visited: string[] = []
  const view: number[] = []
  const viewIds: string[] = []
  const q: Node[] = [root]
  let currentId: string | null = null

  const snap = (line: number, note: string, opts: any = {}): TraceStep => ({
    line,
    note,
    state: {
      currentId,
      // 右视图节点 + 已弹出的全部节点都算 visited（已成定局）
      visitedIds: [...visited],
      queue: q.map((n) => String(n.val)),
      output: '[' + view.join(', ') + ']',
      scalars: [
        { label: '|queue|', value: String(q.length) },
        ...(opts.size !== undefined ? [{ label: 'size', value: String(opts.size) }] : []),
        ...(opts.i !== undefined ? [{ label: 'i', value: `${opts.i}/${opts.size - 1}` }] : []),
        { label: 'view', value: '[' + view.join(',') + ']' },
        { label: '右视图节点', value: viewIds.length ? viewIds.join(',') : '-' },
      ],
    },
  })

  steps.push(snap(6, `初始化 q = [root=${root.val}]，view = []`))
  while (q.length > 0) {
    const size = q.length
    steps.push(snap(8, `进入新一层，size = ${size}`, { size }))
    for (let i = 0; i < size; i++) {
      const node = q.shift() as Node
      currentId = node.id
      steps.push(snap(10, `q.popleft() → ${node.val}（本层第 ${i} 个）`, { size, i }))
      if (i === size - 1) {
        view.push(node.val)
        viewIds.push(node.id)
        steps.push(snap(11, `i == size-1，是本层最后一个 → view.append(${node.val})`, { size, i }))
      } else {
        steps.push(snap(11, `i = ${i} ≠ size-1，不记录`, { size, i }))
      }
      if (node.left) {
        q.push(node.left)
        steps.push(snap(13, `左孩子 ${node.left.val} 入队`, { size, i }))
      } else {
        steps.push(snap(13, `无左孩子`, { size, i }))
      }
      if (node.right) {
        q.push(node.right)
        steps.push(snap(15, `右孩子 ${node.right.val} 入队`, { size, i }))
      } else {
        steps.push(snap(15, `无右孩子`, { size, i }))
      }
      visited.push(node.id)
    }
    currentId = null
    steps.push(snap(16, `本层结束：当前 view = [${view.join(', ')}]`))
  }
  steps.push(snap(17, `q 空，return view = [${view.join(', ')}]`))
  return steps
}

const root = buildTree()

const trace: AlgoTrace = {
  title: '示例：root = [1, 2, 3, null, 5, null, 4]（右视图 = [1,3,4]）',
  inputLabel: 'tree=[1,2,3,#,5,#,4]',
  vizKind: 'tree',
  vizConfig: { tree: root },
  steps: generateSteps(root),
}

export default [trace]
