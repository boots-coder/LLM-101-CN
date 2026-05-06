import type { AlgoTrace, TraceStep } from './types'

interface Node { id: string; val: number; left?: Node | null; right?: Node | null }

function buildTree(): Node {
  // 示例：       3
  //           /     \
  //          5       1
  //         / \     / \
  //        6   2   0   8
  //           / \
  //          7   4
  // p = 5, q = 1, LCA = 3
  return {
    id: 'r', val: 3,
    left: {
      id: 'l', val: 5,
      left:  { id: 'll', val: 6, left: null, right: null },
      right: {
        id: 'lr', val: 2,
        left:  { id: 'lrl', val: 7, left: null, right: null },
        right: { id: 'lrr', val: 4, left: null, right: null },
      },
    },
    right: {
      id: 'rr', val: 1,
      left:  { id: 'rrl', val: 0, left: null, right: null },
      right: { id: 'rrr', val: 8, left: null, right: null },
    },
  }
}

function generateSteps(root: Node, pVal: number, qVal: number): TraceStep[] {
  const steps: TraceStep[] = []
  const visited: string[] = []
  const stackIds: string[] = []
  let currentId: string | null = null
  let ancestorId: string | null = null
  let ancestorVal: number | null = null

  const snap = (line: number, note: string, opts: any = {}): TraceStep => ({
    line,
    note,
    state: {
      currentId,
      stackIds: [...stackIds],
      visitedIds: [...visited],
      stack: stackIds.slice(),
      output: ancestorVal !== null ? `LCA=${ancestorVal}` : 'LCA=?',
      scalars: [
        { label: 'curr', value: opts.currVal !== undefined ? String(opts.currVal) : 'None' },
        { label: 'p', value: String(pVal) },
        { label: 'q', value: String(qVal) },
        { label: 'left?', value: opts.left !== undefined ? String(opts.left) : '-' },
        { label: 'right?', value: opts.right !== undefined ? String(opts.right) : '-' },
        { label: 'self?', value: opts.self !== undefined ? String(opts.self) : '-' },
        { label: 'ret', value: opts.ret !== undefined ? String(opts.ret) : '-' },
        { label: 'LCA', value: ancestorVal !== null ? String(ancestorVal) : '?' },
      ],
    },
  })

  function dfs(node: Node | null | undefined): boolean {
    if (!node) {
      steps.push(snap(7, '空节点：返回 false', { ret: false }))
      return false
    }
    currentId = node.id
    stackIds.push(node.id)
    steps.push(snap(8, `进入 dfs(${node.val})：检查自身/左/右是否含 p 或 q`, { currVal: node.val }))

    const leftHas = dfs(node.left)
    currentId = node.id
    steps.push(snap(9, `dfs(${node.val})：左子树包含 p 或 q ? → ${leftHas}`, {
      currVal: node.val, left: leftHas,
    }))

    const rightHas = dfs(node.right)
    currentId = node.id
    steps.push(snap(10, `dfs(${node.val})：右子树包含 p 或 q ? → ${rightHas}`, {
      currVal: node.val, left: leftHas, right: rightHas,
    }))

    const selfHit = node.val === pVal || node.val === qVal
    steps.push(snap(11, `dfs(${node.val})：自身命中 p 或 q ? → ${selfHit}`, {
      currVal: node.val, left: leftHas, right: rightHas, self: selfHit,
    }))

    // 三选二命中：当前节点是 LCA
    const twoHit =
      (leftHas && rightHas) ||
      (selfHit && (leftHas || rightHas))
    if (twoHit && ancestorId === null) {
      ancestorId = node.id
      ancestorVal = node.val
      steps.push(snap(12, `两条线路命中 → 当前 ${node.val} 即 LCA`, {
        currVal: node.val, left: leftHas, right: rightHas, self: selfHit,
      }))
    }

    const ret = leftHas || rightHas || selfHit
    visited.push(node.id)
    stackIds.pop()
    steps.push(snap(13, `dfs(${node.val}) 返回 ${ret}（左 ∨ 右 ∨ 自身）`, {
      currVal: node.val, left: leftHas, right: rightHas, self: selfHit, ret,
    }))
    return ret
  }

  steps.push(snap(4, `初始化 ancestor = null，p=${pVal}, q=${qVal}，开始 dfs(${root.val})`))
  dfs(root)
  currentId = null
  steps.push(snap(15, `dfs 结束，LCA = ${ancestorVal}`))
  return steps
}

const root = buildTree()

const trace: AlgoTrace = {
  title: '示例：root = [3,5,1,6,2,0,8,#,#,7,4], p = 5, q = 1（LCA = 3）',
  inputLabel: 'tree=[3,5,1,6,2,0,8,#,#,7,4], p=5, q=1',
  vizKind: 'tree',
  vizConfig: { tree: root },
  steps: generateSteps(root, 5, 1),
}

export default [trace]
