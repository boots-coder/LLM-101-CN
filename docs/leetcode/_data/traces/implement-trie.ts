import type { AlgoTrace, TraceStep } from './types'

interface TrieNode {
  id: string
  label: string
  children: Record<string, TrieNode>
  isEnd: boolean
}

function newNode(id: string, label: string): TrieNode {
  return { id, label, children: {}, isEnd: false }
}

// 深拷贝当前 trie 状态供每一步快照
function cloneTrie(node: TrieNode): TrieNode {
  const c: TrieNode = { id: node.id, label: node.label, children: {}, isEnd: node.isEnd }
  for (const k of Object.keys(node.children)) c.children[k] = cloneTrie(node.children[k])
  return c
}

function generateSteps(): TraceStep[] {
  const root = newNode('', '')
  const steps: TraceStep[] = []
  let pathIds: string[] = []
  let currentId: string | null = null

  const snap = (
    line: number,
    note: string,
    op: string,
    extra: any = {},
  ): TraceStep => ({
    line,
    note,
    state: {
      root: cloneTrie(root),
      pathIds: [...pathIds],
      currentId,
      scalars: [{ label: '操作', value: op }, ...(extra.scalars ?? [])],
      result: extra.result,
    },
  })

  function insert(word: string) {
    pathIds = []
    currentId = root.id
    let node = root
    pathIds.push(node.id)
    steps.push(snap(7, `insert("${word}")：从根开始，node=root`, `insert("${word}")`))
    for (const ch of word) {
      if (!(ch in node.children)) {
        const childId = node.id + '/' + ch
        node.children[ch] = newNode(childId, ch)
        steps.push(
          snap(9, `'${ch}' 不在 node.children → 创建新节点`, `insert("${word}")`),
        )
      } else {
        steps.push(
          snap(9, `'${ch}' 已存在，复用`, `insert("${word}")`),
        )
      }
      node = node.children[ch]
      currentId = node.id
      pathIds.push(node.id)
      steps.push(snap(9, `node 走到 '${ch}'`, `insert("${word}")`))
    }
    node.isEnd = true
    steps.push(snap(10, `node.is_end = True（标记 "${word}" 为完整词）`, `insert("${word}")`))
  }

  function search(word: string) {
    pathIds = []
    currentId = root.id
    let node: TrieNode | null = root
    pathIds.push(root.id)
    steps.push(snap(13, `search("${word}")：调用 _walk`, `search("${word}")`))
    for (const ch of word) {
      if (!node || !(ch in node.children)) {
        steps.push(snap(23, `'${ch}' 不在 node.children，return None`, `search("${word}")`, {
          result: 'False',
        }))
        return
      }
      node = node.children[ch]
      currentId = node.id
      pathIds.push(node.id)
      steps.push(snap(24, `走到 '${ch}'`, `search("${word}")`))
    }
    const ok = !!node && node.isEnd
    steps.push(
      snap(14, `_walk 返回非空 node；is_end = ${node.isEnd}`, `search("${word}")`, {
        result: ok ? 'True ✓' : 'False（前缀但非完整词）',
      }),
    )
  }

  function startsWith(prefix: string) {
    pathIds = []
    currentId = root.id
    let node: TrieNode | null = root
    pathIds.push(root.id)
    steps.push(snap(17, `startsWith("${prefix}")：调用 _walk`, `startsWith("${prefix}")`))
    for (const ch of prefix) {
      if (!node || !(ch in node.children)) {
        steps.push(snap(23, `'${ch}' 不在 children，return None → False`, `startsWith("${prefix}")`, {
          result: 'False',
        }))
        return
      }
      node = node.children[ch]
      currentId = node.id
      pathIds.push(node.id)
      steps.push(snap(24, `走到 '${ch}'`, `startsWith("${prefix}")`))
    }
    steps.push(
      snap(17, `_walk 返回非空 node → True`, `startsWith("${prefix}")`, { result: 'True ✓' }),
    )
  }

  insert('apple')
  insert('app')
  search('app')
  search('appl')
  startsWith('appl')
  return steps
}

const trace: AlgoTrace = {
  title: '示例：insert("apple"), insert("app"), search("app"/"appl"), startsWith("appl")',
  inputLabel: 'apple/app/...',
  vizKind: 'trie',
  vizConfig: {},
  steps: generateSteps(),
}

export default [trace]
