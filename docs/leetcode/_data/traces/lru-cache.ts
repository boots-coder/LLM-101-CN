import type { AlgoTrace, TraceStep } from './types'

// vizKind = 'linked-list'
// chain[0]：双向链表（含 head/tail 哨兵）「最新 ← 头侧；最旧 ← 尾侧」
// scalars 显示 cap、cache、最近返回值
type Op =
  | { kind: 'init'; cap: number }
  | { kind: 'put'; key: number; val: number }
  | { kind: 'get'; key: number }

interface DListNode {
  key: number | null // null = 哨兵
  val: number | null
  isHead?: boolean
  isTail?: boolean
}

function generateSteps(ops: Op[]): TraceStep[] {
  const steps: TraceStep[] = []
  let cap = 0
  // 用数组顺序模拟双向链表（数组下标越小越靠近 head 端）
  // 始终：list[0] = head 哨兵, list[last] = tail 哨兵
  let list: DListNode[] = []
  const cache = new Map<number, number>() // key → 当前 list 中下标（动态维护）
  const cacheVal = new Map<number, number>() // key → val（仅用于显示）
  let lastReturn: number | string = ''

  function rebuildIdxMap() {
    cache.clear()
    list.forEach((n, i) => {
      if (n.key !== null) cache.set(n.key, i)
    })
  }

  function snap(
    line: number,
    note: string,
    opts: any = {},
  ): TraceStep {
    const nodes = list.map((n) => {
      let label: string
      let cls: '' | 'head' | 'cycle' | 'detached' = ''
      if (n.isHead) {
        label = 'H'
        cls = 'head'
      } else if (n.isTail) {
        label = 'T'
        cls = 'head'
      } else {
        label = `${n.key}:${n.val}`
        if (opts.highlightKey !== undefined && n.key === opts.highlightKey) {
          cls = 'cycle'
        }
      }
      return { val: label, cls }
    })

    const pointers: any[] = []
    if (opts.pointer !== undefined) {
      pointers.push({
        name: opts.pointer.name,
        chain: 0,
        idx: opts.pointer.idx,
        color: opts.pointer.color ?? '#22c55e',
      })
    }

    return {
      line,
      note,
      state: {
        scalars: [
          { label: 'cap', value: String(cap) },
          { label: 'size', value: String(cacheVal.size) },
          {
            label: 'cache',
            value:
              cacheVal.size === 0
                ? '空'
                : `{${[...cacheVal.entries()]
                    .map(([k, v]) => `${k}:${v}`)
                    .join(', ')}}`,
          },
          ...(lastReturn !== ''
            ? [{ label: 'last return', value: String(lastReturn) }]
            : []),
        ],
        chains: [
          {
            label: '双向链表（H = head 哨兵，T = tail 哨兵；H 端=最新，T 端=最旧）',
            nodes,
            tailArrow: false,
            tailLabel: '',
          },
        ],
        pointers,
      },
    }
  }

  // 哨兵索引帮手
  function headIdx() {
    return 0
  }
  function tailIdx() {
    return list.length - 1
  }

  function removeAt(idx: number) {
    list.splice(idx, 1)
    rebuildIdxMap()
  }

  function addToHead(node: DListNode) {
    list.splice(1, 0, node) // 紧跟 head 哨兵后
    rebuildIdxMap()
  }

  for (const op of ops) {
    if (op.kind === 'init') {
      cap = op.cap
      list = [
        { key: null, val: null, isHead: true },
        { key: null, val: null, isTail: true },
      ]
      cacheVal.clear()
      cache.clear()
      lastReturn = ''
      steps.push(
        snap(
          7,
          `LRUCache(capacity=${op.cap})：cap=${op.cap}, cache={}`,
        ),
      )
      steps.push(
        snap(
          11,
          `创建 head/tail 哨兵并互相串起来：H ↔ T（_add_to_head 与 _remove 永远只面对中间节点）`,
        ),
      )
    } else if (op.kind === 'put') {
      const { key, val } = op
      steps.push(
        snap(33, `put(key=${key}, value=${val})：先查 cache 是否已含 ${key}`),
      )
      if (cache.has(key)) {
        const idx = cache.get(key)!
        list[idx].val = val
        cacheVal.set(key, val)
        steps.push(
          snap(
            34,
            `key=${key} 已在 cache → 直接更新 val=${val}（节点指针不动）`,
            { highlightKey: key, pointer: { name: 'node', idx, color: '#22c55e' } },
          ),
        )
        // _remove + _add_to_head
        const node = list[idx]
        removeAt(idx)
        steps.push(
          snap(
            38,
            `_remove(node)：把节点从当前位置摘下（node.prev.next = node.next; node.next.prev = node.prev）`,
            { highlightKey: key },
          ),
        )
        addToHead(node)
        const newIdx = cache.get(key)!
        steps.push(
          snap(
            39,
            `_add_to_head(node)：挪到 H 端表示「刚刚被使用」`,
            {
              highlightKey: key,
              pointer: { name: 'node', idx: newIdx, color: '#22c55e' },
            },
          ),
        )
      } else {
        steps.push(
          snap(34, `key=${key} 不在 cache，进入新增分支`),
        )
        if (cacheVal.size >= cap) {
          // 淘汰
          const lruIdx = tailIdx() - 1 // tail 哨兵之前那个真节点
          const lruNode = list[lruIdx]
          steps.push(
            snap(
              40,
              `len(cache)=${cacheVal.size} ≥ cap=${cap}，需要淘汰：lru = tail.prev = 节点 ${lruNode.key}:${lruNode.val}`,
              {
                highlightKey: lruNode.key!,
                pointer: { name: 'lru', idx: lruIdx, color: '#ef4444' },
              },
            ),
          )
          removeAt(lruIdx)
          cacheVal.delete(lruNode.key!)
          steps.push(
            snap(
              42,
              `_remove(lru) + del cache[${lruNode.key}]（哈希表里的死指针也要清，否则泄漏）`,
            ),
          )
        }
        // 创建新节点 + 加到 head
        addToHead({ key, val })
        cacheVal.set(key, val)
        const newIdx = cache.get(key)!
        steps.push(
          snap(
            44,
            `创建新节点 (${key}:${val}) → cache[${key}] = node → _add_to_head`,
            {
              highlightKey: key,
              pointer: { name: 'node', idx: newIdx, color: '#22c55e' },
            },
          ),
        )
      }
    } else if (op.kind === 'get') {
      const { key } = op
      steps.push(snap(25, `get(key=${key})：查 cache 是否含 ${key}`))
      if (!cache.has(key)) {
        lastReturn = -1
        steps.push(snap(27, `${key} 不在 cache → return -1`))
      } else {
        const idx = cache.get(key)!
        const node = list[idx];
        const v = node.val
        lastReturn = v ?? -1
        steps.push(
          snap(28, `命中：node = cache[${key}]，val = ${v}`, {
            highlightKey: key,
            pointer: { name: 'node', idx, color: '#22c55e' },
          }),
        )
        removeAt(idx)
        steps.push(
          snap(
            29,
            `_remove(node)：把节点从当前位置摘下`,
            { highlightKey: key },
          ),
        )
        addToHead(node)
        const newIdx = cache.get(key)!
        steps.push(
          snap(
            30,
            `_add_to_head(node)：挪到 H 端，更新「最近使用」状态`,
            {
              highlightKey: key,
              pointer: { name: 'node', idx: newIdx, color: '#22c55e' },
            },
          ),
        )
        steps.push(snap(31, `return node.val = ${v}`, { highlightKey: key }))
      }
    }
  }

  return steps
}

const ops: Op[] = [
  { kind: 'init', cap: 2 },
  { kind: 'put', key: 1, val: 1 },
  { kind: 'put', key: 2, val: 2 },
  { kind: 'get', key: 1 },
  { kind: 'put', key: 3, val: 3 }, // 触发淘汰
  { kind: 'get', key: 2 }, // 应返回 -1
]

const trace: AlgoTrace = {
  title:
    '示例：LRUCache(2); put(1,1); put(2,2); get(1)→1; put(3,3) 淘汰 key=2; get(2)→-1',
  inputLabel: 'cap=2; put(1,1) put(2,2) get(1) put(3,3) get(2)',
  vizKind: 'linked-list',
  vizConfig: {},
  steps: generateSteps(ops),
}

export default [trace]
