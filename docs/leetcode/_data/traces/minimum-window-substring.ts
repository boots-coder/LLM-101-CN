import type { AlgoTrace, TraceStep } from './types'

function generateSteps(s: string, t: string): TraceStep[] {
  const steps: TraceStep[] = []
  const need: Record<string, number> = {}
  const have: Record<string, number> = {}
  let required = 0
  let formed = 0
  let left = 0
  let bestLen = Number.POSITIVE_INFINITY
  let bestL = 0

  const fmtNum = (v: number) => (v === Number.POSITIVE_INFINITY ? '∞' : String(v))

  const snap = (line: number, note: string, opts: any = {}): TraceStep => {
    const ptrs: any[] = []
    if (opts.right !== undefined) ptrs.push({ name: 'R', idx: opts.right, color: '#22c55e' })
    if (opts.right !== undefined) ptrs.push({ name: 'L', idx: left, color: '#3b82f6' })
    return {
      line,
      note,
      state: {
        pointers: ptrs,
        scalars: [
          { label: 'required', value: String(required) },
          { label: 'formed', value: String(formed), cls: formed === required && required > 0 ? 'ok' : undefined },
          { label: 'best_len', value: fmtNum(bestLen) },
          { label: 'best_l', value: String(bestL) },
          ...(opts.right !== undefined
            ? [{ label: '窗口', value: `[${left}, ${opts.right}] 长度 ${opts.right - left + 1}` }]
            : []),
        ],
        window: opts.right !== undefined ? { l: left, r: opts.right } : undefined,
        kv: {
          title: 'need / have（每对：need → have）',
          emptyLabel: '空',
          entries: Object.keys(need).map((k) => ({
            key: `'${k}'`,
            value: `${need[k]} / ${have[k] ?? 0}`,
            highlight:
              opts.hitChar === k ||
              (have[k] !== undefined && have[k] === need[k]),
          })),
        },
        cellState: opts.cellState ?? {},
        result:
          bestLen !== Number.POSITIVE_INFINITY
            ? `"${s.substring(bestL, bestL + bestLen)}" (len=${bestLen})`
            : undefined,
      },
    }
  }

  steps.push(snap(2, `校验：s.length=${s.length}, t.length=${t.length}`))
  steps.push(snap(4, '初始化 need = {}'))
  for (const ch of t) {
    need[ch] = (need[ch] ?? 0) + 1
  }
  required = Object.keys(need).length
  steps.push(snap(6, `统计 t="${t}" 的字符需求到 need`))
  steps.push(snap(7, `required = len(need) = ${required}（不同字符种数）`))
  steps.push(snap(8, '初始化 have = {}, formed = 0, left = 0, best_len = ∞'))

  for (let right = 0; right < s.length; right++) {
    const ch = s[right]
    have[ch] = (have[ch] ?? 0) + 1
    steps.push(
      snap(14, `右进 right=${right}, ch='${ch}'：have['${ch}'] = ${have[ch]}`, {
        right,
        hitChar: ch,
      }),
    )
    if (ch in need && have[ch] === need[ch]) {
      formed += 1
      steps.push(
        snap(16, `'${ch}' 在 need 中且 have == need，formed += 1 → ${formed}`, {
          right,
          hitChar: ch,
        }),
      )
    } else {
      steps.push(
        snap(15, `'${ch}' ${ch in need ? '已超量' : '不在 need'}，formed 不变（=${formed}）`, {
          right,
        }),
      )
    }

    while (formed === required) {
      steps.push(
        snap(17, `formed == required = ${required}，进入收缩内循环`, { right }),
      )
      const winLen = right - left + 1
      if (winLen < bestLen) {
        bestLen = winLen
        bestL = left
        steps.push(
          snap(19, `当前窗口长 ${winLen} < best_len，更新 best_len=${bestLen}, best_l=${bestL}`, {
            right,
            cellState: Object.fromEntries(
              Array.from({ length: bestLen }, (_, k) => [bestL + k, 'found']),
            ),
          }),
        )
      } else {
        steps.push(
          snap(18, `窗口长 ${winLen} ≥ best_len=${fmtNum(bestLen)}，不更新`, { right }),
        )
      }
      const lc = s[left]
      steps.push(snap(21, `准备弹左：lc = s[${left}] = '${lc}'`, { right, hitChar: lc }))
      have[lc] -= 1
      steps.push(snap(22, `have['${lc}'] -= 1 → ${have[lc]}`, { right, hitChar: lc }))
      if (lc in need && have[lc] < need[lc]) {
        formed -= 1
        steps.push(
          snap(24, `'${lc}' 在 need 且 have(${have[lc]}) < need(${need[lc]})，formed -= 1 → ${formed}`, {
            right,
            hitChar: lc,
          }),
        )
      } else {
        steps.push(
          snap(23, `'${lc}' ${lc in need ? '仍达标或不在 need' : '不在 need'}，formed 不变`, {
            right,
          }),
        )
      }
      left += 1
      steps.push(snap(25, `left += 1 → ${left}`, { right }))
    }
  }
  if (bestLen === Number.POSITIVE_INFINITY) {
    steps.push(snap(26, '一次都没覆盖，return ""'))
  } else {
    steps.push(
      snap(26, `return s[${bestL}:${bestL + bestLen}] = "${s.substring(bestL, bestL + bestLen)}"`),
    )
  }
  return steps
}

const s = 'ADOBECODEBANC'
const t = 'ABC'

const trace: AlgoTrace = {
  title: `示例：s = "${s}", t = "${t}"`,
  inputLabel: `s="${s}", t="${t}"`,
  vizKind: 'array-board',
  vizConfig: { values: s.split(''), label: 's' },
  steps: generateSteps(s, t),
}

export default [trace]
