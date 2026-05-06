import type { AlgoTrace, BacktrackState, BacktrackTreeNode, TraceStep } from './types'

// 静态预构建完整决策树（供 viz 居中布局，运行时不会变形）
function buildTree(nums: number[]): BacktrackTreeNode {
  function rec(used: boolean[], pickedIdx: number[]): BacktrackTreeNode {
    const key = pickedIdx.join('.')
    const label = '[' + pickedIdx.map((i) => nums[i]).join(',') + ']'
    const isLeaf = pickedIdx.length === nums.length
    const lastPick = pickedIdx.length > 0 ? pickedIdx[pickedIdx.length - 1] : -1
    const children: BacktrackTreeNode[] = []
    if (!isLeaf) {
      for (let i = 0; i < nums.length; i++) {
        if (used[i]) continue
        used[i] = true
        children.push(rec(used, [...pickedIdx, i]))
        used[i] = false
      }
    }
    return { key, label, pickIdx: lastPick, children, isLeaf }
  }
  return rec(Array(nums.length).fill(false), [])
}

function generateSteps(nums: number[]): TraceStep[] {
  const n = nums.length
  const res: number[][] = []
  const path: number[] = []
  const pathIdx: number[] = []
  const used: boolean[] = Array(n).fill(false)
  const collectedLeafKeys: string[] = []
  const steps: TraceStep[] = []

  const snapshot = (
    line: number,
    note: string,
    action: BacktrackState['action'],
  ): TraceStep => ({
    line,
    note,
    state: {
      path: [...path],
      pathIdx: [...pathIdx],
      used: [...used],
      res: res.map((r) => [...r]),
      collectedLeafKeys: [...collectedLeafKeys],
      action,
    },
  })

  steps.push(snapshot(2, `读取输入：n = ${n}`, 'init'))
  steps.push(snapshot(3, '初始化 res = [], path = []', 'init'))
  steps.push(snapshot(4, `初始化 used = [${used.map(() => 'F').join(', ')}]`, 'init'))
  steps.push(snapshot(19, '从根节点 [] 进入 backtrack()', 'enter'))

  function backtrack(depth: number) {
    steps.push(snapshot(7, `进入 backtrack：判断 len(path)=${path.length} 是否 == ${n}`, 'enter'))
    if (path.length === n) {
      const key = pathIdx.join('.')
      collectedLeafKeys.push(key)
      res.push([...path])
      steps.push(
        snapshot(8, `叶子！把 path 的副本 [${path.join(',')}] 推入 res（res 现 ${res.length} 个）`, 'leaf'),
      )
      steps.push(snapshot(9, '叶子收集完毕，return 回上一层', 'return'))
      return
    }
    for (let i = 0; i < n; i++) {
      steps.push(snapshot(10, `for i = ${i}（path 当前 = [${path.join(',')}]）`, 'enter'))
      if (used[i]) {
        steps.push(snapshot(11, `used[${i}] 为 True，跳过`, 'skip'))
        continue
      }
      used[i] = true
      steps.push(snapshot(13, `选 i=${i}：标记 used[${i}] = True`, 'enter'))
      path.push(nums[i])
      pathIdx.push(i)
      steps.push(snapshot(14, `path.append(nums[${i}]=${nums[i]}) → path = [${path.join(',')}]`, 'enter'))
      steps.push(snapshot(15, `递归进入下一层 backtrack()`, 'enter'))
      backtrack(depth + 1)
      path.pop()
      pathIdx.pop()
      steps.push(snapshot(16, `撤销：path.pop() → path = [${path.join(',')}]`, 'unchoose'))
      used[i] = false
      steps.push(snapshot(17, `撤销：used[${i}] = False`, 'unchoose'))
    }
    steps.push(snapshot(10, `for 循环跑完，return 回上一层`, 'return'))
  }

  backtrack(0)
  steps.push(snapshot(20, `所有分支搜索完毕，返回 res（共 ${res.length} 个排列）`, 'done'))

  return steps
}

const nums = [1, 2, 3]

const trace: AlgoTrace = {
  title: '可视化运行：nums = [1, 2, 3]',
  inputLabel: 'nums = [1, 2, 3]',
  vizKind: 'backtrack-tree',
  vizConfig: {
    nums,
    tree: buildTree(nums),
  },
  steps: generateSteps(nums),
}

export default [trace]
