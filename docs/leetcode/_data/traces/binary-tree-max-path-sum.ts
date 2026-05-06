import type { AlgoTrace, TraceStep } from './types'

interface Node { id: string; val: number; left?: Node | null; right?: Node | null }

function buildTree(): Node {
  // 示例：     -10
  //           /   \
  //          9    20
  //              /  \
  //            15    7
  // 最大路径：15 -> 20 -> 7  =>  42
  return {
    id: 'r', val: -10,
    left:  { id: 'l', val: 9, left: null, right: null },
    right: {
      id: 'rr', val: 20,
      left:  { id: 'rl', val: 15, left: null, right: null },
      right: { id: 'rrr', val: 7, left: null, right: null },
    },
  }
}

function generateSteps(root: Node): TraceStep[] {
  const steps: TraceStep[] = []
  const visited: string[] = []
  const stackIds: string[] = []
  let currentId: string | null = null
  let maxSum = -Infinity

  const snap = (line: number, note: string, opts: any = {}): TraceStep => ({
    line,
    note,
    state: {
      currentId,
      stackIds: [...stackIds],
      visitedIds: [...visited],
      stack: opts.stackVals ?? stackIds.slice(),
      output: maxSum === -Infinity ? '-∞' : String(maxSum),
      scalars: [
        { label: 'curr', value: opts.currVal !== undefined ? String(opts.currVal) : 'None' },
        { label: 'leftGain', value: opts.leftGain !== undefined ? String(opts.leftGain) : '-' },
        { label: 'rightGain', value: opts.rightGain !== undefined ? String(opts.rightGain) : '-' },
        { label: 'newPath', value: opts.newPath !== undefined ? String(opts.newPath) : '-' },
        { label: 'maxSum', value: maxSum === -Infinity ? '-∞' : String(maxSum) },
        { label: 'return', value: opts.ret !== undefined ? String(opts.ret) : '-' },
      ],
    },
  })

  function dfs(node: Node | null | undefined): number {
    if (!node) {
      steps.push(snap(7, '空节点：返回 0', { ret: 0 }))
      return 0
    }
    currentId = node.id
    stackIds.push(node.id)
    steps.push(snap(8, `进入 dfs(${node.val})：先递归左子树`, { currVal: node.val }))

    const leftGain = Math.max(dfs(node.left), 0)
    currentId = node.id
    steps.push(snap(9, `dfs(${node.val})：左子树贡献 leftGain = max(?,0) = ${leftGain}`, {
      currVal: node.val, leftGain,
    }))

    const rightGain = Math.max(dfs(node.right), 0)
    currentId = node.id
    steps.push(snap(10, `dfs(${node.val})：右子树贡献 rightGain = max(?,0) = ${rightGain}`, {
      currVal: node.val, leftGain, rightGain,
    }))

    const newPath = node.val + leftGain + rightGain
    steps.push(snap(11, `穿过 ${node.val} 的完整路径 = ${node.val} + ${leftGain} + ${rightGain} = ${newPath}`, {
      currVal: node.val, leftGain, rightGain, newPath,
    }))

    if (newPath > maxSum) {
      maxSum = newPath
      steps.push(snap(12, `更新全局最大值 maxSum = ${maxSum}`, {
        currVal: node.val, leftGain, rightGain, newPath,
      }))
    } else {
      steps.push(snap(12, `newPath=${newPath} 不超过 maxSum=${maxSum}，保持不变`, {
        currVal: node.val, leftGain, rightGain, newPath,
      }))
    }

    const ret = node.val + Math.max(leftGain, rightGain)
    visited.push(node.id)
    stackIds.pop()
    steps.push(snap(13, `向上返回单边路径：${node.val} + max(${leftGain},${rightGain}) = ${ret}`, {
      currVal: node.val, leftGain, rightGain, newPath, ret,
    }))
    return ret
  }

  steps.push(snap(4, `初始化 maxSum = -∞，开始 dfs(${root.val})`))
  dfs(root)
  currentId = null
  steps.push(snap(15, `dfs 结束，return maxSum = ${maxSum}`))
  return steps
}

const root = buildTree()

const trace: AlgoTrace = {
  title: '示例：root = [-10, 9, 20, null, null, 15, 7]（最大路径和 = 42）',
  inputLabel: 'tree=[-10,9,20,#,#,15,7]',
  vizKind: 'tree',
  vizConfig: { tree: root },
  steps: generateSteps(root),
}

export default [trace]
