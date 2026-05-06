import type { AlgoTrace, TraceStep } from './types'

function generateSteps(s: string): TraceStep[] {
  const last: Record<string, number> = {}
  let left = 0, best = 0
  const steps: TraceStep[] = []

  const snap = (line: number, note: string, right: number): TraceStep => ({
    line,
    note,
    state: {
      pointers: [
        { name: 'L', idx: left, color: '#3b82f6' },
        { name: 'R', idx: right, color: '#22c55e' },
      ],
      scalars: [
        { label: 'best', value: String(best) },
        { label: '窗口', value: `[${left}, ${right}] 长度 ${Math.max(0, right - left + 1)}` },
      ],
      window: { l: left, r: right },
      kv: {
        title: 'last（字符 → 上次下标）',
        emptyLabel: '空',
        entries: Object.entries(last).map(([k, v]) => ({ key: `'${k}'`, value: v })),
      },
    },
  })

  steps.push(snap(4, '初始化 last={}, left=0, best=0', -1))
  for (let right = 0; right < s.length; right++) {
    const ch = s[right]
    steps.push({
      line: 5,
      note: `右端进入 right=${right}, ch='${ch}'`,
      state: {
        pointers: [
          { name: 'L', idx: left, color: '#3b82f6' },
          { name: 'R', idx: right, color: '#22c55e' },
        ],
        scalars: [
          { label: 'best', value: String(best) },
          { label: 'ch', value: `'${ch}'` },
        ],
        window: { l: left, r: right },
        kv: {
          title: 'last（字符 → 上次下标）',
          emptyLabel: '空',
          entries: Object.entries(last).map(([k, v]) => ({
            key: `'${k}'`, value: v, highlight: k === ch,
          })),
        },
      },
    })
    if (ch in last && last[ch] >= left) {
      const newLeft = last[ch] + 1
      steps.push(snap(7, `'${ch}' 上次出现 last[${ch}]=${last[ch]} 仍在窗口内，跳过：left = ${last[ch]} + 1 = ${newLeft}`, right))
      left = newLeft
    } else {
      steps.push(snap(6, `'${ch}' 不在窗口内（或没出现过），left 不变`, right))
    }
    last[ch] = right
    steps.push(snap(8, `更新 last['${ch}'] = ${right}`, right))
    const winLen = right - left + 1
    if (winLen > best) {
      best = winLen
      steps.push(snap(9, `窗口长度 ${winLen} > 旧 best，best = ${best}`, right))
    } else {
      steps.push(snap(9, `窗口长度 ${winLen} ≤ best=${best}，best 不变`, right))
    }
  }
  steps.push(snap(10, `return best = ${best}`, s.length - 1))
  return steps
}

const s = 'abcabcbb'

const trace: AlgoTrace = {
  title: `示例：s = "${s}"`,
  inputLabel: `s="${s}"`,
  vizKind: 'array-board',
  vizConfig: { values: s.split(''), label: 's' },
  steps: generateSteps(s),
}

export default [trace]
