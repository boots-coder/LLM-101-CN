import type { AlgoTrace, TraceStep } from './types'

// 单词搜索：网格回溯。匹配过的格子临时改为 '#' 占位防止重复使用，回溯时再还原。
// vizKind: 'grid'。cellState：'visited'（'#' 占位中）、'current'。
// 提前终止：找到第一个解就 return true。

function generateSteps(boardInput: string[][], word: string): TraceStep[] {
  const board = boardInput.map((row) => [...row])
  const m = board.length, n = board[0].length
  const steps: TraceStep[] = []
  let found = false
  let result = false

  const snap = (line: number, note: string, opts: any = {}): TraceStep => {
    const cellState: Record<string, string> = {}
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        if (board[i][j] === '#') cellState[`${i},${j}`] = 'visited'
      }
    }
    if (opts.current) cellState[`${opts.current[0]},${opts.current[1]}`] = 'current'
    return {
      line,
      note,
      state: {
        grid: board.map((row) => [...row]),
        scalars: [
          { label: 'word', value: word },
          { label: 'k', value: opts.k !== undefined ? `${opts.k}/${word.length}` : '·' },
          ...(opts.current ? [{ label: '(i,j)', value: `(${opts.current[0]},${opts.current[1]})` }] : []),
          { label: 'found', value: found ? 'true' : 'false' },
        ],
        cellState,
        current: opts.current,
      },
    }
  }

  // dfs 返回是否找到；使用 found 标志短路所有上层
  function dfs(i: number, j: number, k: number): boolean {
    if (found) return true
    steps.push(snap(7, `dfs(${i},${j}, k=${k})：检查越界 / 是否匹配 word[${k}]='${word[k]}'`, {
      current: [i, j], k,
    }))
    if (i < 0 || i >= m || j < 0 || j >= n || board[i][j] !== word[k]) {
      steps.push(snap(8, `不匹配（越界或 board[${i}][${j}]='${board[i] && board[i][j]}' ≠ '${word[k]}'），return false`, {
        current: i >= 0 && i < m && j >= 0 && j < n ? [i, j] : undefined, k,
      }))
      return false
    }
    if (k === word.length - 1) {
      steps.push(snap(9, `已匹配到最后一个字符 '${word[k]}'，整词命中！return true`, {
        current: [i, j], k,
      }))
      found = true
      return true
    }
    // 占位
    const saved = board[i][j]
    board[i][j] = '#'
    steps.push(snap(10, `占位：board[${i}][${j}] = '#' 防重复使用`, { current: [i, j], k }))
    const dirs: [number, number, string][] = [
      [-1, 0, '上'],
      [1, 0, '下'],
      [0, -1, '左'],
      [0, 1, '右'],
    ]
    for (const [di, dj, name] of dirs) {
      if (found) break
      const ni = i + di, nj = j + dj
      steps.push(snap(11, `从 (${i},${j}) 向${name}走 → (${ni},${nj})`, { current: [i, j], k }))
      if (dfs(ni, nj, k + 1)) {
        return true
      }
    }
    // 还原
    board[i][j] = saved
    steps.push(snap(13, `四方向均失败，还原：board[${i}][${j}] = '${saved}'，return false`, {
      current: [i, j], k,
    }))
    return false
  }

  steps.push(snap(2, `输入：${m}×${n} 网格，word = "${word}"`, {}))
  steps.push(snap(3, '从每个格子作为起点，尝试 dfs(i, j, 0)', {}))

  outer: for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      steps.push(snap(15, `外层尝试起点 (${i},${j})，board[${i}][${j}]='${board[i][j]}'`, {
        current: [i, j], k: 0,
      }))
      if (board[i][j] !== word[0]) {
        steps.push(snap(16, `首字符不匹配（'${board[i][j]}' ≠ '${word[0]}'），跳过`, {
          current: [i, j], k: 0,
        }))
        continue
      }
      if (dfs(i, j, 0)) {
        result = true
        break outer
      }
    }
  }

  steps.push(snap(18, `搜索结束，return ${result}`, {}))
  return steps
}

const board = [
  ['A', 'B', 'C', 'E'],
  ['S', 'F', 'C', 'S'],
  ['A', 'D', 'E', 'E'],
]
const word = 'ABCCED'

const trace: AlgoTrace = {
  title: `示例：3×4 网格，word = "${word}"`,
  inputLabel: `board(3×4), word="${word}"`,
  vizKind: 'grid',
  vizConfig: {},
  steps: generateSteps(board, word),
}

export default [trace]
