import type { AlgoTrace, TraceStep } from './types'

interface Node { id: string; val: number; left?: Node | null; right?: Node | null }

function buildTree(): Node {
  // 示例：     1
  //          / \
  //         2   3
  //            / \
  //           4   5
  return {
    id: 'r', val: 1,
    left:  { id: 'l', val: 2, left: null, right: null },
    right: {
      id: 'rr', val: 3,
      left:  { id: 'rl', val: 4, left: null, right: null },
      right: { id: 'rr2', val: 5, left: null, right: null },
    },
  }
}

function generateSteps(root: Node): TraceStep[] {
  const steps: TraceStep[] = []
  const visited: string[] = []
  const stackIds: string[] = []
  let currentId: string | null = null
  let phase: 'serialize' | 'deserialize' = 'serialize'
  const tokens: string[] = []
  let cursor = 0

  const snap = (line: number, note: string, opts: any = {}): TraceStep => ({
    line,
    note,
    state: {
      currentId,
      stackIds: [...stackIds],
      visitedIds: [...visited],
      stack: stackIds.slice(),
      output: tokens.length ? tokens.join(',') : '(空)',
      scalars: [
        { label: 'phase', value: phase },
        { label: 'curr', value: opts.currVal !== undefined ? String(opts.currVal) : 'None' },
        { label: 'tokens', value: '[' + tokens.join(',') + ']' },
        { label: 'cursor', value: phase === 'deserialize' ? `${cursor}/${tokens.length}` : '-' },
        ...(opts.token !== undefined ? [{ label: 'token', value: String(opts.token) }] : []),
      ],
    },
  })

  // ---------- Phase 1: serialize（前序 DFS）----------
  function ser(node: Node | null | undefined) {
    if (!node) {
      tokens.push('null')
      steps.push(snap(7, `空节点：tokens.append("null")`, { token: 'null' }))
      return
    }
    currentId = node.id
    stackIds.push(node.id)
    tokens.push(String(node.val))
    steps.push(snap(8, `前序：tokens.append("${node.val}")`, { currVal: node.val, token: node.val }))
    ser(node.left)
    currentId = node.id
    steps.push(snap(9, `回到 ${node.val}：递归右子树`, { currVal: node.val }))
    ser(node.right)
    currentId = node.id
    visited.push(node.id)
    stackIds.pop()
    steps.push(snap(10, `节点 ${node.val} 处理完毕`, { currVal: node.val }))
  }

  steps.push(snap(4, '阶段 1：serialize（前序 DFS，空节点写 "null"）'))
  ser(root)
  steps.push(snap(11, `serialize 结果：${tokens.join(',')}`))

  // ---------- Phase 2: deserialize（按序消费 tokens）----------
  phase = 'deserialize'
  visited.length = 0
  stackIds.length = 0
  currentId = null
  cursor = 0

  // 为反序列化重新分配 id
  let idCnt = 0
  function deser(): Node | null {
    const tk = tokens[cursor]
    steps.push(snap(15, `读取 tokens[${cursor}] = "${tk}"`, { token: tk }))
    cursor++
    if (tk === 'null') {
      steps.push(snap(16, `是 "null"，返回空指针`, { token: tk }))
      return null
    }
    const id = 'd' + idCnt++
    const node: Node = { id, val: Number(tk), left: null, right: null }
    currentId = id
    stackIds.push(id)
    steps.push(snap(17, `创建节点 val=${tk}，递归构造左子树`, { currVal: Number(tk), token: tk }))
    node.left = deser()
    currentId = id
    steps.push(snap(18, `节点 ${tk} 的左子树完成，开始构造右子树`, { currVal: Number(tk) }))
    node.right = deser()
    currentId = id
    visited.push(id)
    stackIds.pop()
    steps.push(snap(19, `节点 ${tk} 的左右子树都构造完毕，返回上层`, { currVal: Number(tk) }))
    return node
  }

  steps.push(snap(13, '阶段 2：deserialize（按 tokens 顺序消费，cursor 从 0 开始）'))
  deser()
  steps.push(snap(20, `deserialize 完成，cursor=${cursor}/${tokens.length}`))
  return steps
}

const root = buildTree()

const trace: AlgoTrace = {
  title: '示例：root = [1, 2, 3, null, null, 4, 5]（前序 + null 序列化）',
  inputLabel: 'tree=[1,2,3,#,#,4,5]',
  vizKind: 'tree',
  vizConfig: { tree: root },
  steps: generateSteps(root),
}

export default [trace]
