<template>
  <div class="gr-viz">
    <div v-if="state.scalars && state.scalars.length" class="gr-scalars">
      <span v-for="s in state.scalars" :key="s.label" class="gr-scalar">
        <span class="gr-scalar-key">{{ s.label }}</span>
        <span class="gr-scalar-val">{{ s.value }}</span>
      </span>
    </div>

    <div class="gr-board">
      <div v-for="(row, i) in state.grid" :key="i" class="gr-row">
        <div
          v-for="(v, j) in row"
          :key="j"
          class="gr-cell"
          :class="cellClass(i, j)"
        >{{ v }}</div>
      </div>
    </div>

    <div v-if="state.queue" class="gr-aux">
      <span class="gr-tag">队列</span>
      <div class="gr-aux-cells">
        <div v-if="!state.queue.length" class="gr-aux-empty">空</div>
        <div v-for="(q, i) in state.queue" :key="i" class="gr-aux-cell">{{ q }}</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  config: any
  state: any
}>()

function cellClass(i: number, j: number) {
  const cls: Record<string, boolean> = {}
  const cs = (props.state.cellState ?? {}) as Record<string, string>
  const k = `${i},${j}`
  if (cs[k]) cls[cs[k]] = true
  if (props.state.current && props.state.current[0] === i && props.state.current[1] === j) {
    cls['current'] = true
  }
  return cls
}
</script>

<style scoped>
.gr-viz { display: flex; flex-direction: column; gap: 10px; }
.gr-scalars { display: flex; flex-wrap: wrap; gap: 8px; }
.gr-scalar {
  display: inline-flex; gap: 6px;
  background: var(--vp-c-bg-mute);
  padding: 3px 10px; border-radius: 999px;
  font-size: 12px; font-family: 'JetBrains Mono', Menlo, monospace;
}
.gr-scalar-key { color: var(--vp-c-text-2); }
.gr-scalar-val { font-weight: 700; }

.gr-board {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 8px;
  display: inline-block;
  align-self: flex-start;
}
.gr-row { display: flex; gap: 3px; }
.gr-row + .gr-row { margin-top: 3px; }
.gr-cell {
  width: 32px; height: 32px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  display: flex; align-items: center; justify-content: center;
  font-family: 'JetBrains Mono', Menlo, monospace;
  font-weight: 700; font-size: 13px;
  transition: all 0.15s;
}
.gr-cell.land    { background: rgba(34, 197, 94, 0.16); border-color: #22c55e; color: #16a34a; }
.gr-cell.water   { color: var(--vp-c-text-3); }
.gr-cell.sunk    { background: rgba(148, 163, 184, 0.18); color: var(--vp-c-text-3); }
.gr-cell.visited { background: rgba(59, 130, 246, 0.12); border-color: #3b82f6; color: #1d4ed8; }
.gr-cell.rotten  { background: rgba(245, 158, 11, 0.18); border-color: #f59e0b; color: #b45309; }
.gr-cell.fresh   { background: rgba(34, 197, 94, 0.10); }
.gr-cell.current {
  background: #22c55e; color: white; border-color: #16a34a;
  box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.22);
  transform: scale(1.1);
}

.gr-aux { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.gr-tag {
  background: var(--vp-c-bg-mute);
  font-size: 11px; font-weight: 700;
  padding: 2px 8px; border-radius: 4px;
  color: var(--vp-c-text-2);
}
.gr-aux-cells { display: flex; gap: 4px; flex-wrap: wrap; }
.gr-aux-empty { font-size: 12px; color: var(--vp-c-text-3); font-style: italic; }
.gr-aux-cell {
  font-family: 'JetBrains Mono', Menlo, monospace;
  font-size: 12px; font-weight: 700;
  padding: 3px 8px;
  border-radius: 4px;
  border: 1px solid #3b82f6;
  background: rgba(59, 130, 246, 0.12);
  color: #1d4ed8;
}
</style>
