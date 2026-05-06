import type { AlgoTrace, TraceStep } from './types'

function generateSteps(temps: number[]): TraceStep[] {
  const n = temps.length
  const answer: number[] = Array(n).fill(0)
  const stack: number[] = []
  const steps: TraceStep[] = []

  const snap = (line: number, note: string, opts: any = {}): TraceStep => {
    const cs: Record<number, string> = {}
    for (const idx of stack) cs[idx] = 'in-stack'
    if (opts.current !== undefined) cs[opts.current] = 'current'
    if (opts.popped !== undefined) cs[opts.popped] = 'popped'
    return {
      line,
      note,
      state: {
        pointers: opts.current !== undefined
          ? [{ name: 'i', idx: opts.current, color: '#22c55e' }]
          : [],
        scalars: [
          ...(opts.t !== undefined ? [{ label: 't', value: String(opts.t) }] : []),
          { label: 'stack', value: `[${stack.join(', ')}]` },
        ],
        cellState: cs,
        stack: [...stack],
        result: [...answer],
        poppingTop: opts.popping ?? false,
      },
    }
  }

  steps.push(snap(3, `初始化 answer = [${answer.join(', ')}], stack = []`))
  for (let i = 0; i < n; i++) {
    const t = temps[i]
    steps.push(snap(5, `进入 i=${i}, t=${t}`, { current: i, t }))
    while (stack.length > 0 && temps[stack[stack.length - 1]] < t) {
      const j = stack[stack.length - 1]
      steps.push(snap(6, `栈顶 j=${j}, temps[${j}]=${temps[j]} < t=${t} → 出栈结算`, {
        current: i, t,
      }))
      stack.pop()
      answer[j] = i - j
      steps.push(snap(8, `answer[${j}] = ${i} - ${j} = ${answer[j]}`, {
        current: i, t, popped: j, popping: true,
      }))
    }
    stack.push(i)
    steps.push(snap(9, `stack.push(${i})`, { current: i, t }))
  }
  steps.push(snap(10, `循环结束，return answer = [${answer.join(', ')}]`))
  return steps
}

const temps = [73, 74, 75, 71, 69, 72, 76, 73]

const trace: AlgoTrace = {
  title: `示例：temperatures = [${temps.join(', ')}]`,
  inputLabel: `temps=[${temps.join(',')}]`,
  vizKind: 'stack-board',
  vizConfig: { values: temps, label: 'temperatures' },
  steps: generateSteps(temps),
}

export default [trace]
