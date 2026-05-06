// 通用算法 trace 数据结构。
// 不同 vizKind 共用 steps[] 容器；具体可视化组件解释 step.state 的字段。

export type VizKind =
  | 'backtrack-tree'
  | 'array-board'
  | 'dp-grid'
  | 'tree'
  | 'linked-list'
  | 'grid'
  | 'trie'
  | 'stack-board'

export interface TraceStep {
  /** solutionCode 中的 1-indexed 行号 */
  line: number
  /** 用人话讲这一步在做什么 */
  note: string
  /** 由具体 viz 解释的状态对象 */
  state: Record<string, any>
}

export interface AlgoTrace {
  title: string
  /** 用于 header 展示输入 */
  inputLabel: string
  vizKind: VizKind
  /** viz 组件需要的静态配置（如预构建的决策树） */
  vizConfig: any
  steps: TraceStep[]
}

// ============ backtrack-tree 专用类型 ============

export interface BacktrackTreeNode {
  /** 路径 key，如 "" / "0" / "0.1" / "0.1.2" */
  key: string
  /** 显示用的标签，如 "[]" / "[1]" / "[1,2,3]" */
  label: string
  /** 此节点对应被选中的下标（即上一层的 i） */
  pickIdx: number
  children: BacktrackTreeNode[]
  isLeaf: boolean
}

export interface BacktrackState {
  path: any[]
  pathIdx: number[]
  used: boolean[]
  res: any[][]
  /** 已收集的叶子节点 key */
  collectedLeafKeys: string[]
  action: 'init' | 'enter' | 'leaf' | 'unchoose' | 'skip' | 'return' | 'done'
}
