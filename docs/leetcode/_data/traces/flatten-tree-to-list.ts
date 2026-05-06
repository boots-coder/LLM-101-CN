import type { AlgoTrace, TraceStep } from './types'

interface Node { id: string; val: number; left?: Node | null; right?: Node | null }

function buildTree(): Node {
  // 示例：     1
  //          / \
  //         2   5
  //        / \   \
  //       3   4   6
  return {
    id: 'r', val: 1,
    left: {
      id: 'l', val: 2,
      left:  { id: 'll', val: 3, left: null, right: null },
      right: { id: 'lr', val: 4, left: null, right: null },
    },
    right: {
      id: 'rr', val: 5,
      left: null,
      right: { id: 'rrr', val: 6, left: null, right: null },
    },
  }
}

// 把 Morris-like / 经典 O(1) 解法可视化
// 算法：从 root 开始，对每个 curr，若 curr.left 存在：
//   1) 找到左子树最右节点 pre
//   2) pre.right = curr.right
//   3) curr.right = curr.left, curr.left = null
//   curr = curr.right
function generateSteps(root: Node): TraceStep[] {
  const steps: TraceStep[] = []
  const visited: string[] = []
  let currentId: string | null = null
  let preId: string | null = null

  // 用于把当前树状态拍平成快照（深拷贝结构）
  function clone(n: Node | null | undefined): Node | null {
    if (!n) return null
    return { id: n.id, val: n.val, left: clone(n.left), right: clone(n.right) }
  }

  // 当前的"动态 root"快照（拍平进行中树会被改）
  let liveRoot: Node = root

  const snap = (line: number, note: string, opts: any = {}): TraceStep => ({
    line,
    note,
    state: {
      currentId,
      visitedIds: [...visited],
      // 给一个动态 tree 让 viz 能跟着结构变化重绘
      tree: clone(liveRoot),
      output: opts.chain ?? '',
      scalars: [
        { label: 'curr', value: opts.currVal !== undefined ? String(opts.currVal) : 'None' },
        { label: 'pre', value: opts.preVal !== undefined ? String(opts.preVal) : '-' },
        { label: '阶段', value: opts.phase ?? '-' },
      ],
    },
  })

  function chainStr(start: Node | null | undefined): string {
    const arr: number[] = []
    let n: Node | null | undefined = start
    while (n) { arr.push(n.val); n = n.right }
    return arr.join(' -> ')
  }

  let curr: Node | null = liveRoot
  steps.push(snap(4, '初始化 curr = root', { currVal: curr.val, phase: '初始化', chain: chainStr(liveRoot) }))

  while (curr) {
    currentId = curr.id
    steps.push(snap(5, `主循环：检查当前节点 curr=${curr.val}`, {
      currVal: curr.val, phase: '主循环', chain: chainStr(liveRoot),
    }))
    if (curr.left) {
      steps.push(snap(6, `curr=${curr.val} 有左子树，需要把右子树挂到左子树最右`, {
        currVal: curr.val, phase: '检查左子树', chain: chainStr(liveRoot),
      }))
      let pre: Node = curr.left
      preId = pre.id
      steps.push(snap(7, `pre = curr.left = ${pre.val}`, {
        currVal: curr.val, preVal: pre.val, phase: '寻找最右', chain: chainStr(liveRoot),
      }))
      while (pre.right) {
        pre = pre.right
        preId = pre.id
        steps.push(snap(8, `pre = pre.right → ${pre.val}`, {
          currVal: curr.val, preVal: pre.val, phase: '寻找最右', chain: chainStr(liveRoot),
        }))
      }
      steps.push(snap(8, `循环结束：pre = ${pre.val} 即左子树最右节点`, {
        currVal: curr.val, preVal: pre.val, phase: '找到最右', chain: chainStr(liveRoot),
      }))
      const oldRightVal = curr.right ? curr.right.val : 'null'
      pre.right = curr.right ?? null
      steps.push(snap(9, `pre(${pre.val}).right = curr.right(=${oldRightVal})：把原右子树挂到左子树最右下面`, {
        currVal: curr.val, preVal: pre.val, phase: '接上右子树', chain: chainStr(liveRoot),
      }))
      const leftVal = curr.left.val
      curr.right = curr.left
      steps.push(snap(10, `curr(${curr.val}).right = curr.left(=${leftVal})：把左子树移到右边`, {
        currVal: curr.val, preVal: pre.val, phase: '左→右', chain: chainStr(liveRoot),
      }))
      curr.left = null
      steps.push(snap(10, `curr(${curr.val}).left = null：清空左指针`, {
        currVal: curr.val, preVal: pre.val, phase: '清空左', chain: chainStr(liveRoot),
      }))
    } else {
      steps.push(snap(11, `curr=${curr.val} 无左子树，直接前进`, {
        currVal: curr.val, phase: '跳过', chain: chainStr(liveRoot),
      }))
    }
    visited.push(curr.id)
    curr = curr.right ?? null
    if (curr) {
      currentId = curr.id
      steps.push(snap(12, `curr = curr.right → ${curr.val}`, {
        currVal: curr.val, phase: '前进', chain: chainStr(liveRoot),
      }))
    }
  }
  currentId = null
  steps.push(snap(13, `curr=null，扁平化完成：${chainStr(liveRoot)}`, {
    phase: '完成', chain: chainStr(liveRoot),
  }))
  return steps
}

const root = buildTree()

const trace: AlgoTrace = {
  title: '示例：root = [1, 2, 5, 3, 4, null, 6]（O(1) 空间扁平化）',
  inputLabel: 'tree=[1,2,5,3,4,#,6]',
  vizKind: 'tree',
  vizConfig: { tree: root },
  steps: generateSteps(root),
}

export default [trace]
