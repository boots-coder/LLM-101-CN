import type { AlgoTrace, TraceStep } from './types'

function generateSteps(s: string, wordDict: string[]): TraceStep[] {
  const wordSet = new Set(wordDict)
  const maxLen = wordDict.reduce((a, w) => Math.max(a, w.length), 0)
  const n = s.length
  // dp[i] 表示 s[:i] 是否可拆；尺寸 n+1（i=0..n）
  const dp: (string | null)[] = Array(n + 1).fill(null)
  const steps: TraceStep[] = []

  const snap = (line: number, note: string, opts: any = {}): TraceStep => {
    const cs: Record<string, string> = { ...(opts.cellState ?? {}) }
    return {
      line,
      note,
      state: {
        grid: [dp.map((v) => (v === null ? '·' : v))],
        scalars: [
          { label: 's', value: `"${s}"` },
          { label: 'dict', value: `[${wordDict.map((w) => `"${w}"`).join(',')}]` },
          { label: 'maxLen', value: String(maxLen) },
          ...(opts.i !== undefined ? [{ label: 'i', value: String(opts.i) }] : []),
          ...(opts.j !== undefined ? [{ label: 'j', value: String(opts.j) }] : []),
          ...(opts.sub !== undefined ? [{ label: 's[j:i]', value: `"${opts.sub}"` }] : []),
        ],
        cellState: cs,
        current: opts.curIdx !== undefined ? [0, opts.curIdx] : undefined,
        sources: opts.srcIdx !== undefined ? [[0, opts.srcIdx]] : [],
        transition: opts.transition,
      },
    }
  }

  steps.push(snap(2, `把 wordDict 转 set 加速查询，maxLen = ${maxLen}`))
  steps.push(snap(6, `dp 长度 ${n + 1}，全部初始化为 False`))
  for (let i = 0; i <= n; i++) dp[i] = 'F'

  dp[0] = 'T'
  steps.push(snap(7, `边界 dp[0] = True（空串可拆）`, {
    cellState: { '0,0': 'base' },
  }))

  for (let i = 1; i <= n; i++) {
    const lo = Math.max(0, i - maxLen)
    steps.push(snap(8, `进入 i=${i}（s[:${i}]="${s.slice(0, i)}"），j 从 ${lo} 到 ${i - 1}`, {
      i,
      curIdx: i,
      transition: `寻找切点 j，使 dp[j]=True 且 s[j:${i}] 在字典中`,
    }))

    let hit = false
    for (let j = lo; j < i; j++) {
      const sub = s.slice(j, i)
      if (dp[j] !== 'T') {
        steps.push(snap(11, `j=${j}: dp[${j}]=False，短路跳过 s[j:i]="${sub}"`, {
          i, j, sub, curIdx: i, srcIdx: j,
        }))
        continue
      }
      const inDict = wordSet.has(sub)
      steps.push(snap(11, `j=${j}: dp[${j}]=True ✓，检查 s[${j}:${i}]="${sub}" ${inDict ? '∈' : '∉'} dict`, {
        i, j, sub, curIdx: i, srcIdx: j,
      }))
      if (inDict) {
        dp[i] = 'T'
        hit = true
        steps.push(snap(12, `命中！dp[${i}] ← True，break`, {
          i, j, sub, curIdx: i, srcIdx: j,
          cellState: { [`0,${i}`]: 'filled', [`0,${j}`]: 'source' },
          transition: `dp[${i}] = dp[${j}] ∧ ("${sub}" ∈ dict) = True`,
        }))
        break
      }
    }

    if (!hit) {
      steps.push(snap(8, `所有 j 都不行，dp[${i}] 保持 False`, {
        i, curIdx: i,
      }))
    }
  }

  steps.push(snap(14, `return dp[${n}] = ${dp[n] === 'T' ? 'True' : 'False'}`, {
    cellState: { [`0,${n}`]: 'answer' },
  }))

  return steps
}

const s = 'leetcode'
const wordDict = ['leet', 'code']

const trace: AlgoTrace = {
  title: `示例：s = "${s}", wordDict = [${wordDict.map((w) => `"${w}"`).join(', ')}]`,
  inputLabel: `s="${s}", dict=[${wordDict.map((w) => `"${w}"`).join(',')}]`,
  vizKind: 'dp-grid',
  vizConfig: {
    rows: 1,
    cols: s.length + 1,
    rowLabels: ['dp'],
    colLabels: Array.from({ length: s.length + 1 }, (_, i) => `i=${i}`),
  },
  steps: generateSteps(s, wordDict),
}

export default [trace]
