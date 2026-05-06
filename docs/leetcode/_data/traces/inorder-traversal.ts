import type { AlgoTrace, TraceStep } from './types'

// 树用 id 字符串作为唯一标识，便于 TreeViz 标记当前 / 路径 / 已访问
interface Node {
  id: string
  val: number
  left?: Node | null
  right?: Node | null
}

function buildTree(): Node {
  // 经典示例：     1
  //              / \
  //            2    3
  //           / \
  //          4   5
  return {
    id: 'r', val: 1,
    left: {
      id: 'l', val: 2,
      left:  { id: 'll', val: 4, left: null, right: null },
      right: { id: 'lr', val: 5, left: null, right: null },
    },
    right: { id: 'rr', val: 3, left: null, right: null },
  }
}

function generateSteps(root: Node): TraceStep[] {
  const res: number[] = []
  const stack: Node[] = []
  let curr: Node | null = root
  const steps: TraceStep[] = []

  const snap = (line: number, note: string, opts: any = {}): TraceStep => ({
    line,
    note,
    state: {
      currentId: curr ? curr.id : null,
      stackIds: stack.map((s) => s.id),
      visitedIds: opts.visitedIds ?? [],
      stack: stack.map((s) => String(s.val)),
      output: [...res],
      scalars: [
        { label: 'curr', value: curr ? String(curr.val) : 'None' },
        { label: 'stack.size', value: String(stack.length) },
        { label: 'res', value: '[' + res.join(',') + ']' },
      ],
    },
  })

  steps.push(snap(20, '初始化 curr=root, stack=[]'))
  const visited: string[] = []
  while (curr !== null || stack.length > 0) {
    steps.push(snap(21, `循环：curr=${curr ? curr.val : 'None'}, |stack|=${stack.length}`, {
      visitedIds: [...visited],
    }))
    while (curr !== null) {
      stack.push(curr)
      steps.push(snap(24, `① 一路向左：stack.push(${curr.val})`, { visitedIds: [...visited] }))
      curr = curr.left ?? null
      steps.push(snap(25, `curr = curr.left → ${curr ? curr.val : 'None'}`, { visitedIds: [...visited] }))
    }
    const node = stack.pop() as Node
    steps.push(snap(27, `② stack.pop() → 取出 ${node.val}`, { visitedIds: [...visited] }))
    res.push(node.val)
    visited.push(node.id)
    curr = node
    steps.push(snap(28, `res.append(${node.val}) → res = [${res.join(', ')}]`, {
      visitedIds: [...visited],
    }))
    curr = node.right ?? null
    steps.push(snap(30, `③ 转向右子树：curr = ${curr ? curr.val : 'None'}`, {
      visitedIds: [...visited],
    }))
  }
  curr = null
  steps.push(snap(31, `curr 与 stack 都空，return res = [${res.join(', ')}]`, {
    visitedIds: [...visited],
  }))
  return steps
}

const root = buildTree()

const trace: AlgoTrace = {
  title: '示例：root = [1, 2, 3, 4, 5]（解法二·迭代）',
  inputLabel: 'tree=[1,2,3,4,5]',
  vizKind: 'tree',
  vizConfig: { tree: root },
  steps: generateSteps(root),
}

export default [trace]
