import type { AlgoTrace, BacktrackState, BacktrackTreeNode, TraceStep } from './types'

// N 皇后（n=4）：第 row 行尝试每一列；用三个 set 检查 col / diag1 (row-col) / diag2 (row+col)
// 决策树：每个节点表示「已经决定了前 k 行的列」；叶子 = 第 n 行成功放完
// pickIdx 路径表示「第 i 行选第 pickIdx[i] 列」

function buildTree(n: number): BacktrackTreeNode {
  function rec(rowsCols: number[]): BacktrackTreeNode {
    const key = rowsCols.join('.')
    const label = rowsCols.length === 0
      ? '[]'
      : '[' + rowsCols.map((c, r) => `r${r}c${c}`).join(',') + ']'
    const lastPick = rowsCols.length > 0 ? rowsCols[rowsCols.length - 1] : -1
    let isLeaf = false
    const children: BacktrackTreeNode[] = []
    if (rowsCols.length === n) {
      isLeaf = true
      return { key, label, pickIdx: lastPick, children, isLeaf }
    }
    const row = rowsCols.length
    // 先算冲突集合
    const cols = new Set(rowsCols)
    const d1 = new Set(rowsCols.map((c, r) => r - c))   // ↘ 同对角
    const d2 = new Set(rowsCols.map((c, r) => r + c))   // ↙ 反对角
    for (let c = 0; c < n; c++) {
      const conflict = cols.has(c) || d1.has(row - c) || d2.has(row + c)
      if (conflict) {
        // 冲突分支挂一个死胡同子节点（不再展开）
        const deadKey = [...rowsCols, c].join('.')
        const deadLabel = '[' + [...rowsCols, c].map((cc, r) => `r${r}c${cc}`).join(',') + ']✗'
        children.push({ key: deadKey, label: deadLabel, pickIdx: c, children: [], isLeaf: false })
      } else {
        children.push(rec([...rowsCols, c]))
      }
    }
    return { key, label, pickIdx: lastPick, children, isLeaf }
  }
  return rec([])
}

function generateSteps(n: number): TraceStep[] {
  const res: number[][] = [] // 每个解记录每一行的列号
  const path: number[] = []  // 当前行选的列号
  const pathIdx: number[] = [] // 与 path 同步，作为决策树定位 key
  const used: boolean[] = Array(n).fill(false) // viz 复用 used 表示「列是否被占用」
  const cols = new Set<number>()
  const d1 = new Set<number>() // row - col
  const d2 = new Set<number>() // row + col
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

  steps.push(snapshot(2, `n = ${n}；目标：每行放一个皇后，列 / 主对角 / 副对角都不冲突`, 'init'))
  steps.push(snapshot(3, '初始化 res = [], path = [], cols = ∅, d1 = ∅, d2 = ∅', 'init'))
  steps.push(snapshot(4, `从 row=0 开始 backtrack`, 'enter'))

  function backtrack(row: number) {
    steps.push(snapshot(7, `backtrack(row=${row})；path = [${path.join(',')}]`, 'enter'))
    if (row === n) {
      const key = pathIdx.join('.')
      collectedLeafKeys.push(key)
      res.push([...path])
      steps.push(snapshot(8, `row == n，找到解 [${path.join(',')}]！res 现 ${res.length} 个`, 'leaf'))
      steps.push(snapshot(9, '记录完毕，return 回上一层', 'return'))
      return
    }
    for (let c = 0; c < n; c++) {
      const conflict = cols.has(c) || d1.has(row - c) || d2.has(row + c)
      steps.push(snapshot(12, `行 ${row} 试列 c=${c}：冲突? cols=${cols.has(c)}, d1=${d1.has(row - c)}, d2=${d2.has(row + c)}`, 'enter'))
      if (conflict) {
        // 推一个「冲突」中间快照，对应决策树死胡同子节点
        path.push(c)
        pathIdx.push(c)
        used[c] = true
        steps.push(snapshot(13, `冲突！跳过列 ${c}（path 显示中含 r${row}c${c}）`, 'skip'))
        path.pop()
        pathIdx.pop()
        used[c] = false
        steps.push(snapshot(14, `撤销冲突试探，回到 path = [${path.join(',')}]`, 'unchoose'))
        continue
      }
      // 放置皇后
      path.push(c)
      pathIdx.push(c)
      used[c] = true
      cols.add(c); d1.add(row - c); d2.add(row + c)
      steps.push(snapshot(15, `放 (${row},${c})：cols/d1/d2 三 set 加入；path = [${path.join(',')}]`, 'enter'))
      steps.push(snapshot(16, `递归 backtrack(row=${row + 1})`, 'enter'))
      backtrack(row + 1)
      path.pop()
      pathIdx.pop()
      used[c] = false
      cols.delete(c); d1.delete(row - c); d2.delete(row + c)
      steps.push(snapshot(17, `撤销：移除 (${row},${c}) 的列与对角占用`, 'unchoose'))
    }
    steps.push(snapshot(12, `行 ${row} 全部列试完，return`, 'return'))
  }

  backtrack(0)
  steps.push(snapshot(20, `搜索完毕，res 共 ${res.length} 个解：${JSON.stringify(res)}`, 'done'))

  return steps
}

const n = 4

// nums 字段被 BacktrackTreeViz 用来显示 used；这里把列号 0..n-1 当作可被占用的「列资源」
const colNums = Array.from({ length: n }, (_, i) => i)

const trace: AlgoTrace = {
  title: `可视化运行：n = ${n}（4×4 棋盘）`,
  inputLabel: `n=${n}`,
  vizKind: 'backtrack-tree',
  vizConfig: {
    nums: colNums,   // viz 中的「used 数组」表示列是否被占用
    tree: buildTree(n),
  },
  steps: generateSteps(n),
}

export default [trace]
