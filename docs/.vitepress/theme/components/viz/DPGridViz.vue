<template>
  <div class="dp-viz">
    <div v-if="state.scalars && state.scalars.length" class="dp-scalars">
      <span v-for="s in state.scalars" :key="s.label" class="dp-scalar">
        <span class="dp-scalar-key">{{ s.label }}</span>
        <span class="dp-scalar-val">{{ s.value }}</span>
      </span>
    </div>

    <div class="dp-board">
      <!-- 列标签 -->
      <div v-if="config.colLabels" class="dp-row col-headers">
        <div v-if="config.rowLabels" class="dp-cell axis"></div>
        <div v-for="(c, j) in config.colLabels" :key="j" class="dp-cell axis">{{ c }}</div>
      </div>
      <div v-for="(row, i) in displayGrid" :key="i" class="dp-row">
        <div v-if="config.rowLabels" class="dp-cell axis">{{ config.rowLabels[i] }}</div>
        <div
          v-for="(v, j) in row"
          :key="j"
          class="dp-cell"
          :class="cellClass(i, j)"
        >
          <div class="dp-val">{{ v ?? '·' }}</div>
        </div>
      </div>
    </div>

    <!-- 状态转移说明 -->
    <div v-if="state.transition" class="dp-transition">
      <span class="dp-tag">转移</span>
      <code>{{ state.transition }}</code>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  config: {
    rows: number
    cols: number
    rowLabels?: (string | number)[]
    colLabels?: (string | number)[]
  }
  state: any
}>()

const displayGrid = computed(() => {
  // state.grid 是 number[][] 或 number[]（1D 时 cols=1 或 rows=1）
  const g = props.state.grid
  if (!g) return []
  // 1D → 包成 1 行
  if (props.config.rows === 1 && Array.isArray(g) && !Array.isArray(g[0])) return [g]
  return g
})

function cellClass(i: number, j: number) {
  const cls: Record<string, boolean> = {}
  const ck = (props.state.cellState ?? {}) as Record<string, string>
  const k = `${i},${j}`
  if (ck[k]) cls[ck[k]] = true
  // 当前 cell
  if (props.state.current && props.state.current[0] === i && props.state.current[1] === j) {
    cls['current'] = true
  }
  // 转移源
  const sources = props.state.sources as [number, number][] | undefined
  if (sources?.some((s) => s[0] === i && s[1] === j)) cls['source'] = true
  return cls
}
</script>

<style scoped>
.dp-viz { display: flex; flex-direction: column; gap: 10px; }
.dp-scalars { display: flex; flex-wrap: wrap; gap: 8px; }
.dp-scalar {
  display: inline-flex; gap: 6px;
  background: var(--vp-c-bg-mute);
  padding: 3px 10px; border-radius: 999px;
  font-size: 12px; font-family: 'JetBrains Mono', Menlo, monospace;
}
.dp-scalar-key { color: var(--vp-c-text-2); }
.dp-scalar-val { font-weight: 700; }

.dp-board {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 8px;
  overflow-x: auto;
}
.dp-row { display: flex; gap: 3px; }
.dp-row + .dp-row { margin-top: 3px; }
.dp-cell {
  min-width: 36px; height: 36px;
  display: flex; align-items: center; justify-content: center;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  font-family: 'JetBrains Mono', Menlo, monospace;
  font-size: 13px;
  font-weight: 600;
  flex-shrink: 0;
  transition: all 0.15s;
}
.dp-cell.axis {
  background: transparent;
  border: none;
  color: var(--vp-c-text-3);
  font-weight: 700;
  font-size: 11px;
}
.dp-cell.base   { background: rgba(148, 163, 184, 0.15); color: var(--vp-c-text-2); }
.dp-cell.filled { background: rgba(59, 130, 246, 0.10); border-color: #3b82f6; color: #1d4ed8; }
.dp-cell.source { background: rgba(245, 158, 11, 0.18); border-color: #f59e0b; color: #b45309; }
.dp-cell.current {
  background: rgba(34, 197, 94, 0.22);
  border-color: #16a34a;
  color: #166534;
  box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.18);
  transform: scale(1.05);
}
.dp-cell.answer { background: rgba(34, 197, 94, 0.35); border-color: #16a34a; color: #166534; font-weight: 800; }
.dp-val { line-height: 1; }

.dp-transition {
  display: flex; gap: 8px; align-items: center;
  padding: 6px 10px;
  background: var(--vp-c-bg-soft);
  border-left: 3px solid #f59e0b;
  border-radius: 0 6px 6px 0;
  font-size: 12.5px;
}
.dp-tag {
  background: var(--vp-c-bg-mute);
  font-size: 10px; font-weight: 700;
  padding: 2px 6px; border-radius: 3px;
  color: var(--vp-c-text-2);
}
.dp-transition code {
  background: var(--vp-c-bg-mute);
  padding: 1px 6px; border-radius: 3px;
  font-size: 12px;
}
</style>
