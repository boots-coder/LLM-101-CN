import type { AlgoTrace, TraceStep } from './types'

// 注：solutionCode 是 O(n log n) 的 patience sort 版；
// 这里按需求 trace **O(n²) DP** 解法：dp[i] = 以 nums[i] 结尾的 LIS 长度。
// 行号引用对应 problem.intuition / 类似 dp 思路的解读。

function generateSteps(nums: number[]): TraceStep[] {
  const n = nums.length
  const dp: (number | null)[] = Array(n).fill(null)
  const steps: TraceStep[] = []

  const snap = (line: number, note: string, opts: any = {}): TraceStep => {
    const cs: Record<string, string> = { ...(opts.cellState ?? {}) }
    return {
      line,
      note,
      state: {
        grid: [dp.map((v) => (v === null ? '·' : String(v)))],
        scalars: [
          { label: 'n', value: String(n) },
          ...(opts.i !== undefined ? [{ label: 'i', value: String(opts.i) }] : []),
          ...(opts.j !== undefined ? [{ label: 'j', value: String(opts.j) }] : []),
          ...(opts.best !== undefined ? [{ label: 'best', value: String(opts.best) }] : []),
          ...(opts.ans !== undefined ? [{ label: 'ans', value: String(opts.ans) }] : []),
        ],
        cellState: cs,
        current: opts.curIdx !== undefined ? [0, opts.curIdx] : undefined,
        sources: opts.srcIdx !== undefined ? [[0, opts.srcIdx]] : [],
        transition: opts.transition,
      },
    }
  }

  steps.push(snap(5, `初始化 dp 数组全为 1（每个元素自身就是长度 1 的 LIS）`))
  for (let i = 0; i < n; i++) dp[i] = 1
  steps.push(snap(5, `dp 全部填 1`, {
    cellState: Object.fromEntries(dp.map((_, i) => [`0,${i}`, 'base'])),
  }))

  for (let i = 1; i < n; i++) {
    steps.push(snap(6, `进入外层 i=${i}, nums[${i}]=${nums[i]}`, {
      i,
      curIdx: i,
      transition: `寻找所有 j<${i} 且 nums[j]<${nums[i]} 的 dp[j] 最大值`,
    }))

    let best = dp[i] as number
    for (let j = 0; j < i; j++) {
      if (nums[j] < nums[i]) {
        const cand = (dp[j] as number) + 1
        steps.push(snap(8, `检查 j=${j}: nums[${j}]=${nums[j]} < nums[${i}]=${nums[i]} ✓，候选 dp[${j}]+1 = ${cand}`, {
          i, j, curIdx: i, srcIdx: j, best,
          transition: `dp[${i}] 候选 = max(dp[${i}], dp[${j}]+1) = max(${best}, ${cand})`,
        }))
        if (cand > best) {
          best = cand
          steps.push(snap(9, `更新 best = ${best}（接在 nums[${j}] 之后更长）`, {
            i, j, curIdx: i, srcIdx: j, best,
          }))
        }
      } else {
        steps.push(snap(8, `检查 j=${j}: nums[${j}]=${nums[j]} ≥ nums[${i}]=${nums[i]} ✗，跳过`, {
          i, j, curIdx: i, best,
        }))
      }
    }

    dp[i] = best
    steps.push(snap(10, `dp[${i}] = ${best}`, {
      i, curIdx: i, best,
      cellState: { [`0,${i}`]: 'filled' },
      transition: `dp[${i}] ← ${best}`,
    }))
  }

  let ans = 0
  for (let i = 0; i < n; i++) ans = Math.max(ans, dp[i] as number)
  // 找一个达到 ans 的下标做 answer 高亮
  let ansIdx = 0
  for (let i = 0; i < n; i++) if (dp[i] === ans) { ansIdx = i; break }

  steps.push(snap(11, `return max(dp) = ${ans}`, {
    ans,
    cellState: { [`0,${ansIdx}`]: 'answer' },
  }))

  return steps
}

const nums = [10, 9, 2, 5, 3, 7, 101, 18]

const trace: AlgoTrace = {
  title: `示例：nums = [${nums.join(', ')}]（O(n²) DP 解法）`,
  inputLabel: `nums=[${nums.join(',')}]`,
  vizKind: 'dp-grid',
  vizConfig: {
    rows: 1,
    cols: nums.length,
    rowLabels: ['dp'],
    colLabels: nums.map((v) => `${v}`),
  },
  steps: generateSteps(nums),
}

export default [trace]
