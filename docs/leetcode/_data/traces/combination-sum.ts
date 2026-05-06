import type { AlgoTrace, BacktrackState, BacktrackTreeNode, TraceStep } from './types'

// 组合总和：candidates 元素可重复使用 → 用 start 参数避免重复组合
// 节点 key 用 pickIdx.join('.')；pickIdx 是「依次选了哪个下标」（允许重复）

function buildTree(candidates: number[], target: number): BacktrackTreeNode {
  function rec(pickedIdx: number[], start: number, sum: number): BacktrackTreeNode {
    const key = pickedIdx.join('.')
    const label = '[' + pickedIdx.map((i) => candidates[i]).join(',') + ']' + ` sum=${sum}`
    const lastPick = pickedIdx.length > 0 ? pickedIdx[pickedIdx.length - 1] : -1
    let isLeaf = false
    const children: BacktrackTreeNode[] = []
    if (sum === target) {
      isLeaf = true // 命中目标的合法叶子
    } else if (sum < target) {
      // 还有展开空间
      for (let i = start; i < candidates.length; i++) {
        const next = sum + candidates[i]
        if (next > target) {
          // 即使越界也保留为「死胡同叶子」，让决策树可视化更直观
          const deadKey = [...pickedIdx, i].join('.')
          const deadLabel = '[' + [...pickedIdx, i].map((k) => candidates[k]).join(',') + ']' + ` sum=${next}✗`
          children.push({ key: deadKey, label: deadLabel, pickIdx: i, children: [], isLeaf: false })
        } else {
          // 可重复选 → 下一层 start = i
          children.push(rec([...pickedIdx, i], i, next))
        }
      }
    }
    return { key, label, pickIdx: lastPick, children, isLeaf }
  }
  return rec([], 0, 0)
}

function generateSteps(candidates: number[], target: number): TraceStep[] {
  const n = candidates.length
  const res: number[][] = []
  const path: number[] = []
  const pathIdx: number[] = []
  const used: boolean[] = Array(n).fill(false) // 仅作 viz 占位（组合题不需要 used）
  const collectedLeafKeys: string[] = []
  const steps: TraceStep[] = []

  const snapshot = (line: number, note: string, action: BacktrackState['action']): TraceStep => ({
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

  steps.push(snapshot(2, `读取输入：candidates = [${candidates.join(',')}], target = ${target}`, 'init'))
  steps.push(snapshot(3, '初始化 res = [], path = []', 'init'))
  steps.push(snapshot(4, '从根节点 [] 进入 backtrack(start=0, sum=0)', 'enter'))

  function backtrack(start: number, sum: number) {
    steps.push(snapshot(7, `进入 backtrack(start=${start}, sum=${sum})；path = [${path.join(',')}]`, 'enter'))
    if (sum === target) {
      const key = pathIdx.join('.')
      collectedLeafKeys.push(key)
      res.push([...path])
      steps.push(snapshot(8, `命中！sum == target，把 [${path.join(',')}] 推入 res（res 现 ${res.length} 个）`, 'leaf'))
      steps.push(snapshot(9, '收集完毕，return 回上一层', 'return'))
      return
    }
    if (sum > target) {
      // 实际代码里靠 if (next > target) 提前剪枝；这里不会真触达
      steps.push(snapshot(10, `sum > target，剪枝 return`, 'return'))
      return
    }

    for (let i = start; i < n; i++) {
      const next = sum + candidates[i]
      steps.push(snapshot(12, `for i=${i}（candidates[${i}]=${candidates[i]}）；试探 sum + ${candidates[i]} = ${next}`, 'enter'))
      if (next > target) {
        // 推一个「越界」中间快照，让决策树可视化对应到死胡同子节点
        path.push(candidates[i])
        pathIdx.push(i)
        steps.push(snapshot(13, `${next} > ${target}，越界剪枝（path 中含 [${path.join(',')}]）`, 'skip'))
        path.pop()
        pathIdx.pop()
        steps.push(snapshot(14, `撤销越界试探，回到 path = [${path.join(',')}]`, 'unchoose'))
        continue
      }
      path.push(candidates[i])
      pathIdx.push(i)
      steps.push(snapshot(15, `选 i=${i}：path.append(${candidates[i]}) → path = [${path.join(',')}]，sum=${next}`, 'enter'))
      steps.push(snapshot(16, `递归进入 backtrack(start=${i}, sum=${next})（可重选 → start=i）`, 'enter'))
      backtrack(i, next)
      path.pop()
      pathIdx.pop()
      steps.push(snapshot(17, `撤销：path.pop() → path = [${path.join(',')}]`, 'unchoose'))
    }
    steps.push(snapshot(12, `for 循环跑完（start=${start}），return 回上一层`, 'return'))
  }

  backtrack(0, 0)
  steps.push(snapshot(20, `所有分支搜索完毕，res 共 ${res.length} 个组合：${JSON.stringify(res)}`, 'done'))

  return steps
}

const candidates = [2, 3, 6, 7]
const target = 7

const trace: AlgoTrace = {
  title: `可视化运行：candidates = [${candidates.join(', ')}], target = ${target}`,
  inputLabel: `candidates=[${candidates.join(',')}], target=${target}`,
  vizKind: 'backtrack-tree',
  vizConfig: {
    nums: candidates,
    tree: buildTree(candidates, target),
  },
  steps: generateSteps(candidates, target),
}

export default [trace]
