import type { AlgoTrace, TraceStep } from './types'

function generateSteps(s: string, p: string): TraceStep[] {
  const n = s.length, m = p.length
  const need: number[] = new Array(26).fill(0)
  const have: number[] = new Array(26).fill(0)
  const res: number[] = []
  const steps: TraceStep[] = []

  const idx = (ch: string) => ch.charCodeAt(0) - 97
  const arrEq = (a: number[], b: number[]) => a.every((v, i) => v === b[i])
  const compactKv = (arr: number[]) => {
    const entries: { key: string; value: number }[] = []
    for (let i = 0; i < 26; i++) {
      if (arr[i] !== 0) entries.push({ key: String.fromCharCode(97 + i), value: arr[i] })
    }
    return entries
  }

  const snap = (line: number, note: string, opts: any = {}): TraceStep => {
    const ptrs: any[] = []
    if (opts.right !== undefined) ptrs.push({ name: 'R', idx: opts.right, color: '#22c55e' })
    if (opts.leftOut !== undefined) ptrs.push({ name: 'out', idx: opts.leftOut, color: '#f59e0b' })
    const winL = opts.winL
    const winR = opts.winR
    return {
      line,
      note,
      state: {
        pointers: ptrs,
        scalars: [
          { label: 'm', value: String(m) },
          ...(winL !== undefined && winR !== undefined
            ? [{ label: '窗口', value: `[${winL}, ${winR}] 长度 ${winR - winL + 1}` }]
            : []),
          { label: 'res', value: `[${res.join(',')}]` },
          ...(opts.matchTag
            ? [{ label: 'have == need', value: opts.matchTag, cls: opts.matchTag === 'true' ? 'ok' : 'warn' }]
            : []),
        ],
        window: winL !== undefined && winR !== undefined ? { l: winL, r: winR } : undefined,
        kv: {
          title: 'have（窗口字符计数，仅显示非零）',
          emptyLabel: '空',
          entries: compactKv(have),
        },
        secondary: {
          label: 'need（p 的字符计数，仅显示非零）',
          values: compactKv(need).map((e) => `${e.key}:${e.value}`),
        },
        cellState: opts.cellState ?? {},
        result: res.length ? `[${res.join(',')}]` : undefined,
      },
    }
  }

  steps.push(snap(2, `n=${n}, m=${m}，先校验 n < m`))
  if (n < m) {
    steps.push(snap(3, `n < m，直接 return []`))
    return steps
  }
  steps.push(snap(5, '初始化 need/have 为长度 26 的零数组'))
  for (const ch of p) {
    need[idx(ch)] += 1
  }
  steps.push(snap(7, `统计 p="${p}" 的字符计数到 need`))
  steps.push(snap(9, '初始化 res = []'))

  for (let right = 0; right < n; right++) {
    have[idx(s[right])] += 1
    const winL = right - m + 1
    const winR = right
    steps.push(
      snap(11, `右进：have['${s[right]}'] += 1（right=${right}, ch='${s[right]}'）`, {
        right,
        winL: Math.max(0, winL),
        winR,
      }),
    )
    if (right >= m) {
      const out = right - m
      steps.push(
        snap(12, `right=${right} ≥ m=${m}，需要弹出 s[right-m] = s[${out}] = '${s[out]}'`, {
          right,
          leftOut: out,
          winL: out,
          winR,
        }),
      )
      have[idx(s[out])] -= 1
      steps.push(
        snap(13, `左出：have['${s[out]}'] -= 1`, {
          right,
          winL: winL,
          winR,
        }),
      )
    }
    if (right >= m - 1) {
      const eq = arrEq(have, need)
      steps.push(
        snap(14, `比较 have == need ⇒ ${eq}`, {
          right,
          winL,
          winR,
          matchTag: eq ? 'true' : 'false',
        }),
      )
      if (eq) {
        res.push(winL)
        steps.push(
          snap(15, `命中！起点 = right - m + 1 = ${winL}，res 记入 ${winL}`, {
            right,
            winL,
            winR,
            matchTag: 'true',
            cellState: Object.fromEntries(
              Array.from({ length: m }, (_, k) => [winL + k, 'found']),
            ),
          }),
        )
      }
    } else {
      steps.push(
        snap(14, `窗口未满（right=${right} < m-1=${m - 1}），跳过比较`, {
          right,
          winL: 0,
          winR,
        }),
      )
    }
  }
  steps.push(snap(16, `扫描结束，return res = [${res.join(',')}]`))
  return steps
}

const s = 'cbaebabacd'
const p = 'abc'

const trace: AlgoTrace = {
  title: `示例：s = "${s}", p = "${p}"`,
  inputLabel: `s="${s}", p="${p}"`,
  vizKind: 'array-board',
  vizConfig: { values: s.split(''), label: 's' },
  steps: generateSteps(s, p),
}

export default [trace]
