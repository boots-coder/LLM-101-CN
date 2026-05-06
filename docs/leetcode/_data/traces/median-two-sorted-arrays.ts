import type { AlgoTrace, TraceStep } from './types'

// 行号对应 problems/median-two-sorted-arrays.ts 里的 solutionCode：
//   1: def findMedianSortedArrays(nums1, nums2) -> float:
//   2:     if len(nums1) > len(nums2):
//   3:         nums1, nums2 = nums2, nums1
//   4:     m, n = len(nums1), len(nums2)
//   5:     total_left = (m + n + 1) // 2
//   6:     lo, hi = 0, m
//   7:     while lo <= hi:
//   8:         i = (lo + hi) // 2
//   9:         j = total_left - i
//  10..13: l1, r1, l2, r2 哨兵取值
//  14:     if l1 <= r2 and l2 <= r1:
//  15..17:     return ...
//  18:     elif l1 > r2:
//  19:         hi = i - 1
//  20:     else:
//  21:         lo = i + 1

function fmt(x: number): string {
  if (x === Number.POSITIVE_INFINITY) return '+∞'
  if (x === Number.NEGATIVE_INFINITY) return '-∞'
  return String(x)
}

function generateSteps(a: number[], b: number[]): TraceStep[] {
  const steps: TraceStep[] = []

  // 保证 nums1 是较短的（演示 swap）
  let nums1 = a
  let nums2 = b
  let swapped = false
  steps.push({
    line: 1,
    note: `输入 nums1=[${a.join(',')}], nums2=[${b.join(',')}]`,
    state: {
      scalars: [
        { label: 'nums1', value: `[${a.join(',')}]` },
        { label: 'nums2', value: `[${b.join(',')}]` },
      ],
      secondary: { label: 'nums2', values: b, cellCls: {} },
    },
  })
  steps.push({
    line: 1,
    note: '思路：在较短数组上二分切点 i，使「左半总长 = total_left」，并满足「l1≤r2 且 l2≤r1」',
    state: {
      scalars: [
        { label: 'nums1', value: `[${a.join(',')}]` },
        { label: 'nums2', value: `[${b.join(',')}]` },
      ],
      secondary: { label: 'nums2', values: b, cellCls: {} },
    },
  })

  if (nums1.length > nums2.length) {
    ;[nums1, nums2] = [nums2, nums1]
    swapped = true
  }

  const m = nums1.length
  const n = nums2.length
  const totalLeft = ((m + n + 1) / 2) | 0

  const snap = (
    line: number,
    note: string,
    opts: {
      lo?: number
      hi?: number
      i?: number
      j?: number
      l1?: number
      r1?: number
      l2?: number
      r2?: number
      cellState?: Record<number, string>
      secCellCls?: Record<number, string>
      result?: any
      extra?: any[]
    } = {},
  ): TraceStep => {
    const ptrs: any[] = []
    // i 在 nums1（主数组）上：i 是切点（左侧前 i 个属于左半），用 i-1 / i 高亮
    if (opts.i !== undefined) {
      if (opts.i - 1 >= 0 && opts.i - 1 < m) {
        ptrs.push({ name: 'l1', idx: opts.i - 1, color: '#3b82f6' })
      }
      if (opts.i >= 0 && opts.i < m) {
        ptrs.push({ name: 'r1', idx: opts.i, color: '#a855f7' })
      }
    }
    const sec: any = { label: `nums2 (n=${n})`, values: nums2, cellCls: opts.secCellCls ?? {} }
    return {
      line,
      note,
      state: {
        pointers: ptrs,
        scalars: [
          { label: 'm', value: String(m) },
          { label: 'n', value: String(n) },
          { label: 'total_left', value: String(totalLeft) },
          ...(opts.lo !== undefined ? [{ label: 'lo', value: String(opts.lo) }] : []),
          ...(opts.hi !== undefined ? [{ label: 'hi', value: String(opts.hi) }] : []),
          ...(opts.i !== undefined ? [{ label: 'i', value: String(opts.i) }] : []),
          ...(opts.j !== undefined ? [{ label: 'j', value: String(opts.j) }] : []),
          ...(opts.l1 !== undefined ? [{ label: 'l1', value: fmt(opts.l1) }] : []),
          ...(opts.r1 !== undefined ? [{ label: 'r1', value: fmt(opts.r1) }] : []),
          ...(opts.l2 !== undefined ? [{ label: 'l2', value: fmt(opts.l2) }] : []),
          ...(opts.r2 !== undefined ? [{ label: 'r2', value: fmt(opts.r2) }] : []),
          ...(opts.extra ?? []),
        ],
        secondary: sec,
        cellState: opts.cellState ?? {},
        result: opts.result,
      },
    }
  }

  if (swapped) {
    steps.push({
      line: 2,
      note: `len(nums1)=${b.length} > len(nums2)=${a.length}，需要交换`,
      state: {
        scalars: [
          { label: 'nums1', value: `[${a.join(',')}]` },
          { label: 'nums2', value: `[${b.join(',')}]` },
        ],
        secondary: { label: 'nums2', values: b, cellCls: {} },
      },
    })
    steps.push({
      line: 3,
      note: `swap 后：nums1 = [${nums1.join(',')}] (短), nums2 = [${nums2.join(',')}] (长)`,
      state: {
        scalars: [
          { label: 'nums1', value: `[${nums1.join(',')}]` },
          { label: 'nums2', value: `[${nums2.join(',')}]` },
        ],
        secondary: { label: `nums2 (n=${n})`, values: nums2, cellCls: {} },
      },
    })
  } else {
    steps.push(snap(2, `len(nums1)=${m} ≤ len(nums2)=${n}，无需交换（保证在较短数组上二分）`))
  }

  steps.push(snap(4, `读出长度：m = len(nums1) = ${m}, n = len(nums2) = ${n}`))
  steps.push(
    snap(
      5,
      `total_left = (m+n+1)//2 = (${m}+${n}+1)//2 = ${totalLeft}（左半总长，奇数时多 1 在左）`,
    ),
  )
  steps.push(
    snap(
      5,
      `(m+n) 是${(m + n) % 2 === 1 ? '奇数 → 中位数 = max(l1,l2)' : '偶数 → 中位数 = (max(l1,l2)+min(r1,r2))/2'}`,
    ),
  )

  let lo = 0
  let hi = m
  steps.push(snap(6, `初始化二分范围 lo=0, hi=m=${m}（i 是 nums1 上的切点 ∈ [0, m]）`, { lo, hi }))

  let result: any
  let iter = 0
  while (lo <= hi) {
    iter++
    steps.push(snap(7, `=== 第 ${iter} 轮二分 === lo=${lo} ≤ hi=${hi}，进入循环`, { lo, hi }))
    const i = ((lo + hi) / 2) | 0
    steps.push(snap(8, `i = (lo + hi)//2 = (${lo}+${hi})//2 = ${i}（在 nums1 切点）`, { lo, hi, i }))
    const j = totalLeft - i
    steps.push(snap(9, `j = total_left - i = ${totalLeft} - ${i} = ${j}（在 nums2 切点）`, { lo, hi, i, j }))

    const l1 = i === 0 ? Number.NEGATIVE_INFINITY : nums1[i - 1]
    const r1 = i === m ? Number.POSITIVE_INFINITY : nums1[i]
    const l2 = j === 0 ? Number.NEGATIVE_INFINITY : nums2[j - 1]
    const r2 = j === n ? Number.POSITIVE_INFINITY : nums2[j]

    const secCls: Record<number, string> = {}
    for (let k = 0; k < n; k++) secCls[k] = k < j ? 'left-half' : 'right-half'
    const cellState: Record<number, string> = {}
    for (let k = 0; k < m; k++) cellState[k] = k < i ? 'left-half' : 'right-half'

    steps.push(
      snap(10, `按切点把两数组各分成左右两半（蓝=左半, 紫=右半）`, {
        lo, hi, i, j, cellState, secCellCls: secCls,
      }),
    )
    steps.push(
      snap(
        11,
        `nums1 的两侧值：l1=${fmt(l1)} (i==0?用-∞:nums1[${i - 1}]), r1=${fmt(r1)} (i==m?用+∞:nums1[${i}])`,
        { lo, hi, i, j, l1, r1, cellState, secCellCls: secCls },
      ),
    )
    steps.push(
      snap(
        12,
        `nums2 的两侧值：l2=${fmt(l2)} (j==0?用-∞:nums2[${j - 1}]), r2=${fmt(r2)} (j==n?用+∞:nums2[${j}])`,
        { lo, hi, i, j, l1, r1, l2, r2, cellState, secCellCls: secCls },
      ),
    )

    // 把合法性判断拆成两条，方便观察
    const condA = l1 <= r2
    const condB = l2 <= r1
    steps.push(
      snap(13, `合法性比较 A：l1=${fmt(l1)} ≤ r2=${fmt(r2)} ? ${condA ? '✓' : '✗'}`, {
        lo, hi, i, j, l1, r1, l2, r2, cellState, secCellCls: secCls,
      }),
    )
    steps.push(
      snap(13, `合法性比较 B：l2=${fmt(l2)} ≤ r1=${fmt(r1)} ? ${condB ? '✓' : '✗'}`, {
        lo, hi, i, j, l1, r1, l2, r2, cellState, secCellCls: secCls,
      }),
    )

    if (condA && condB) {
      steps.push(
        snap(
          14,
          `两条都满足，命中合法切点 → 计算中位数`,
          { lo, hi, i, j, l1, r1, l2, r2, cellState, secCellCls: secCls },
        ),
      )
      steps.push(
        snap(14, `当前左半 = nums1[:${i}] ∪ nums2[:${j}]，右半 = nums1[${i}:] ∪ nums2[${j}:]`, {
          lo, hi, i, j, l1, r1, l2, r2, cellState, secCellCls: secCls,
        }),
      )
      if ((m + n) % 2 === 1) {
        steps.push(
          snap(15, `(m+n)=${m + n} 为奇数，左半多 1 格 → 中位数 = max(l1, l2)`, {
            lo, hi, i, j, l1, r1, l2, r2, cellState, secCellCls: secCls,
          }),
        )
        result = Math.max(l1, l2)
        steps.push(
          snap(16, `return max(${fmt(l1)}, ${fmt(l2)}) = ${fmt(result)}`, {
            lo, hi, i, j, l1, r1, l2, r2, cellState, secCellCls: secCls, result,
          }),
        )
      } else {
        steps.push(
          snap(15, `(m+n)=${m + n} 为偶数，左右等长 → 中位数 = (max(l1,l2)+min(r1,r2))/2`, {
            lo, hi, i, j, l1, r1, l2, r2, cellState, secCellCls: secCls,
          }),
        )
        result = (Math.max(l1, l2) + Math.min(r1, r2)) / 2
        steps.push(
          snap(
            17,
            `return (${fmt(Math.max(l1, l2))}+${fmt(Math.min(r1, r2))})/2 = ${result}`,
            { lo, hi, i, j, l1, r1, l2, r2, cellState, secCellCls: secCls, result },
          ),
        )
      }
      return steps
    } else if (l1 > r2) {
      steps.push(
        snap(18, `l1=${fmt(l1)} > r2=${fmt(r2)}，说明 nums1 左半最大 > nums2 右半最小`, {
          lo, hi, i, j, l1, r1, l2, r2, cellState, secCellCls: secCls,
        }),
      )
      steps.push(
        snap(19, `nums1 切多了 → hi = i - 1 = ${i - 1}（i 需要左移）`, {
          lo, hi, i, j, l1, r1, l2, r2, cellState, secCellCls: secCls,
        }),
      )
      hi = i - 1
      steps.push(snap(7, `下一轮：lo=${lo}, hi=${hi}`, { lo, hi }))
    } else {
      steps.push(
        snap(20, `l2=${fmt(l2)} > r1=${fmt(r1)}，说明 nums2 左半最大 > nums1 右半最小`, {
          lo, hi, i, j, l1, r1, l2, r2, cellState, secCellCls: secCls,
        }),
      )
      steps.push(
        snap(21, `nums1 切少了 → lo = i + 1 = ${i + 1}（i 需要右移）`, {
          lo, hi, i, j, l1, r1, l2, r2, cellState, secCellCls: secCls,
        }),
      )
      lo = i + 1
      steps.push(snap(7, `下一轮：lo=${lo}, hi=${hi}`, { lo, hi }))
    }
  }

  steps.push(snap(7, `lo=${lo} > hi=${hi}，循环结束（题目保证有解，理论上不会到这）`, { lo, hi }))
  return steps
}

const nums1 = [1, 2]
const nums2 = [3, 4]

const trace: AlgoTrace = {
  title: `示例：nums1 = [${nums1.join(', ')}], nums2 = [${nums2.join(', ')}]`,
  inputLabel: `nums1=[${nums1.join(',')}], nums2=[${nums2.join(',')}]`,
  vizKind: 'array-board',
  // 主数组展示较短的那个（swap 后会是 nums1）；nums2 用 secondary 展示
  vizConfig: { values: nums1, label: 'nums1（较短，二分切点 i 在它上面）' },
  steps: generateSteps(nums1, nums2),
}

export default [trace]
