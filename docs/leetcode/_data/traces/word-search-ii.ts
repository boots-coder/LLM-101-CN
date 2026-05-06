import type { AlgoTrace, TraceStep } from './types'

// Word Search II：先把 words 建成 trie，再从网格每个格子出发回溯，
// 沿 trie 同步走，碰到 isEnd 就把对应 word 收入 result。
// 这里的 trace 只用 GridViz 展示：
//   - cellState: visited / current / matched
//   - scalars: 当前 trie 路径前缀、已收集 words
// 不去重渲染 trie 本身（trie 在 scalars 里以"前缀字符串"出现）。
//
// 教学性优先：本 trace 演示两个起点
//   (1) 从 (1,0)='e' 起步成功匹配 "eat"
//   (2) 从 (0,0)='o' 起步成功匹配 "oath"
// 其他起点的失败分支只挑代表性 1-2 步，避免 step 爆炸。

type Cell = [number, number]

function generateSteps(): TraceStep[] {
  const board: string[][] = [
    ['o', 'a', 'a', 'n'],
    ['e', 't', 'a', 'e'],
    ['i', 'h', 'k', 'r'],
    ['i', 'f', 'l', 'v'],
  ]
  const m = board.length, n = board[0].length
  const steps: TraceStep[] = []

  const visited = Array.from({ length: m }, () => Array(n).fill(false))
  const found: string[] = []
  let prefix = ''
  let trieRoute = '' // 形如 "root → e → a → t [end:eat]"

  const snap = (line: number, note: string, opts: any = {}): TraceStep => {
    const grid: string[][] = board.map((row) => [...row])
    const cellState: Record<string, string> = {}
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        if (visited[i][j]) cellState[`${i},${j}`] = 'visited'
        else cellState[`${i},${j}`] = 'normal'
      }
    }
    if (opts.matched) {
      for (const [r, c] of opts.matched as Cell[]) {
        cellState[`${r},${c}`] = 'matched'
      }
    }
    if (opts.current) {
      cellState[`${opts.current[0]},${opts.current[1]}`] = 'current'
    }
    return {
      line,
      note,
      state: {
        grid,
        cellState,
        current: opts.current,
        scalars: [
          { label: 'prefix', value: prefix || '(空)' },
          { label: 'trie 路径', value: trieRoute || '(root)' },
          { label: '已收集 words', value: found.length ? '[' + found.join(',') + ']' : '[]' },
          ...(opts.cell ? [{ label: '当前格', value: `(${opts.cell[0]},${opts.cell[1]})='${opts.ch}'` }] : []),
          ...(opts.action ? [{ label: '动作', value: opts.action }] : []),
        ],
      },
    }
  }

  // ============ 阶段 1：建 trie（仅用一步示意） ============
  steps.push(snap(2, '阶段 1：把 words = ["oath","pea","eat","rain"] 建成 trie（root → 各分支）。下一步开始网格回溯。', {
    action: '建 trie',
  }))

  // ============ 阶段 2：网格回溯 ============
  // ---- 失败示例：从 (0,0)='o' 走 "oath" 之外的方向先碰壁，最后走通 "oath" ----
  // 演示用脚本化路径而非真递归，控制 step 数。

  // 起点 (1,0) = 'e'，trie 走 root→e
  function visit(r: number, c: number, addPrefix: string, addRoute: string, action: string, line: number, note: string) {
    visited[r][c] = true
    prefix += addPrefix
    trieRoute = addRoute
    steps.push(snap(line, note, { current: [r, c], cell: [r, c], ch: board[r][c], action }))
  }
  function leave(r: number, c: number, line: number, note: string) {
    visited[r][c] = false
    prefix = prefix.slice(0, -1)
    // trieRoute 退化：去掉末尾 " → x" 段
    const idx = trieRoute.lastIndexOf(' → ')
    if (idx > 0) trieRoute = trieRoute.slice(0, idx)
    else trieRoute = ''
    steps.push(snap(line, note, { current: [r, c], cell: [r, c], ch: board[r][c], action: '回溯' }))
  }

  steps.push(snap(10, '外层扫描：从 (0,0) 开始尝试每个起点。', {}))

  // ---- 演示 1：从 (1,0)='e' 出发匹配 "eat" ----
  steps.push(snap(11, "→ 起点 (1,0)='e'：trie 的 root 有子节点 'e'，下钻。", {
    current: [1, 0], cell: [1, 0], ch: 'e', action: 'dfs 进入',
  }))
  visit(1, 0, 'e', 'root → e', 'dfs(1,0)', 12, "标记 visited[1][0]=true，prefix='e'。")

  // 尝试上 (0,0)='o'：trie 节点 e 没有子节点 'o'，剪枝
  steps.push(snap(13, "向上 (0,0)='o'：trie 节点 'e' 没有子节点 'o'，剪枝（不进入）。", {
    current: [0, 0], cell: [0, 0], ch: 'o', matched: [[1, 0]], action: '剪枝',
  }))

  // 尝试右 (1,1)='t'：trie 节点 e 没有 't'，剪枝
  steps.push(snap(13, "向右 (1,1)='t'：trie 节点 'e' 没有子节点 't'，剪枝。", {
    current: [1, 1], cell: [1, 1], ch: 't', matched: [[1, 0]], action: '剪枝',
  }))

  // 尝试下 (2,0)='i'：trie 节点 e 没有 'i'，剪枝
  steps.push(snap(13, "向下 (2,0)='i'：trie 节点 'e' 没有子节点 'i'，剪枝。", {
    current: [2, 0], cell: [2, 0], ch: 'i', matched: [[1, 0]], action: '剪枝',
  }))

  // 此处发现：board 上 'e' 的右上是 (0,0)='o'，没用；
  // 重新换一个 'e'：(1,3)='e'，因为 trie 'e' → 'a' 存在
  // 教学起见：直接说"换起点"
  leave(1, 0, 14, "(1,0) 四个方向都剪枝，回溯：visited[1][0]=false，prefix=''。")

  // ---- 起点 (1,3)='e' 匹配 "eat" ----
  steps.push(snap(11, "→ 换起点 (1,3)='e'：trie root 有 'e'，下钻。", {
    current: [1, 3], cell: [1, 3], ch: 'e', action: 'dfs 进入',
  }))
  visit(1, 3, 'e', 'root → e', 'dfs(1,3)', 12, "visited[1][3]=true，prefix='e'。")

  // 向左 (1,2)='a'：trie 节点 'e' 有 'a'！
  steps.push(snap(13, "向左 (1,2)='a'：trie 节点 'e' 有子节点 'a'，进入。", {
    current: [1, 2], cell: [1, 2], ch: 'a', matched: [[1, 3]], action: 'dfs 进入',
  }))
  visit(1, 2, 'a', 'root → e → a', 'dfs(1,2)', 12, "visited[1][2]=true，prefix='ea'。")

  // 向左 (1,1)='t'：trie 节点 'a' 有 't'！且 't' 是 isEnd（"eat" 完成）
  steps.push(snap(13, "向左 (1,1)='t'：trie 节点 'a' 有子节点 't'，进入。", {
    current: [1, 1], cell: [1, 1], ch: 't', matched: [[1, 3], [1, 2]], action: 'dfs 进入',
  }))
  visit(1, 1, 't', 'root → e → a → t [end:eat]', 'dfs(1,1)', 12, "visited[1][1]=true，prefix='eat'。")
  found.push('eat')
  steps.push(snap(15, "命中 trie 节点的 isEnd='eat'：found.push('eat')。继续 dfs，看是否还能延伸。", {
    current: [1, 1], cell: [1, 1], ch: 't', matched: [[1, 3], [1, 2], [1, 1]], action: '收集 word',
  }))
  // 't' 节点没有更多子节点 → 回溯
  steps.push(snap(13, "trie 节点 't' 无更多子节点，dfs 自然收束，开始回溯。", {
    current: [1, 1], cell: [1, 1], ch: 't', matched: [[1, 3], [1, 2], [1, 1]], action: '回溯',
  }))
  leave(1, 1, 14, "回溯：visited[1][1]=false，prefix='ea'。")
  leave(1, 2, 14, "回溯：visited[1][2]=false，prefix='e'。")
  leave(1, 3, 14, "回溯：visited[1][3]=false，prefix=''。")

  // ---- 起点 (0,0)='o' 匹配 "oath" ----
  steps.push(snap(11, "→ 起点 (0,0)='o'：trie root 有 'o'，下钻。", {
    current: [0, 0], cell: [0, 0], ch: 'o', action: 'dfs 进入',
  }))
  visit(0, 0, 'o', 'root → o', 'dfs(0,0)', 12, "visited[0][0]=true，prefix='o'。")

  // 向右 (0,1)='a'：trie 'o' 有 'a'
  steps.push(snap(13, "向右 (0,1)='a'：trie 节点 'o' 有子节点 'a'，进入。", {
    current: [0, 1], cell: [0, 1], ch: 'a', matched: [[0, 0]], action: 'dfs 进入',
  }))
  visit(0, 1, 'a', 'root → o → a', 'dfs(0,1)', 12, "visited[0][1]=true，prefix='oa'。")

  // 向下 (1,1)='t'：trie 'a' 有 't'
  steps.push(snap(13, "向下 (1,1)='t'：trie 节点 'a' 有子节点 't'，进入。", {
    current: [1, 1], cell: [1, 1], ch: 't', matched: [[0, 0], [0, 1]], action: 'dfs 进入',
  }))
  visit(1, 1, 't', 'root → o → a → t', 'dfs(1,1)', 12, "visited[1][1]=true，prefix='oat'。")

  // 向下 (2,1)='h'：trie 't' 有 'h'，且 'h' 是 isEnd（"oath" 完成）
  steps.push(snap(13, "向下 (2,1)='h'：trie 节点 't' 有子节点 'h'，进入。", {
    current: [2, 1], cell: [2, 1], ch: 'h', matched: [[0, 0], [0, 1], [1, 1]], action: 'dfs 进入',
  }))
  visit(2, 1, 'h', 'root → o → a → t → h [end:oath]', 'dfs(2,1)', 12, "visited[2][1]=true，prefix='oath'。")
  found.push('oath')
  steps.push(snap(15, "命中 isEnd='oath'：found.push('oath')。", {
    current: [2, 1], cell: [2, 1], ch: 'h', matched: [[0, 0], [0, 1], [1, 1], [2, 1]], action: '收集 word',
  }))
  steps.push(snap(13, "trie 节点 'h' 无更多子节点，开始回溯。", {
    current: [2, 1], cell: [2, 1], ch: 'h', matched: [[0, 0], [0, 1], [1, 1], [2, 1]], action: '回溯',
  }))
  leave(2, 1, 14, "回溯：visited[2][1]=false，prefix='oat'。")
  leave(1, 1, 14, "回溯：visited[1][1]=false，prefix='oa'。")
  leave(0, 1, 14, "回溯：visited[0][1]=false，prefix='o'。")
  leave(0, 0, 14, "回溯：visited[0][0]=false，prefix=''。")

  // ---- 失败示例：起点 (3,3)='v'：trie root 无 'v'，剪枝 ----
  steps.push(snap(11, "→ 起点 (3,3)='v'：trie root 没有子节点 'v'，整支剪枝（节省大量搜索）。", {
    current: [3, 3], cell: [3, 3], ch: 'v', action: '剪枝',
  }))

  // ---- 失败示例：起点 (0,3)='n'，trie 有 'n'？只有 "rain"，'n' 不是 root 子节点，剪枝 ----
  steps.push(snap(11, "→ 起点 (0,3)='n'：trie root 没有子节点 'n'（'rain' 起首是 'r'），剪枝。", {
    current: [0, 3], cell: [0, 3], ch: 'n', action: '剪枝',
  }))

  // 结束
  steps.push(snap(20, `所有起点尝试完毕，return found = [${found.join(', ')}]`, {
    action: '完成',
  }))
  return steps
}

const trace: AlgoTrace = {
  title: '示例：board=4×4, words=["oath","pea","eat","rain"]（应返回 ["eat","oath"]）',
  inputLabel: 'board=4×4, words=["oath","pea","eat","rain"]',
  vizKind: 'grid',
  vizConfig: {},
  steps: generateSteps(),
}

export default [trace]
