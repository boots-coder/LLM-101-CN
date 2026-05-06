<template>
  <div class="sb-viz">
    <div v-if="state.scalars && state.scalars.length" class="sb-scalars">
      <span v-for="s in state.scalars" :key="s.label" class="sb-scalar">
        <span class="sb-scalar-key">{{ s.label }}</span>
        <span class="sb-scalar-val">{{ s.value }}</span>
      </span>
    </div>

    <div class="sb-row">
      <!-- 数组 -->
      <div class="sb-array">
        <div v-if="config.label" class="sb-array-label">{{ config.label }}</div>
        <div class="sb-cells">
          <div
            v-for="(v, idx) in config.values"
            :key="idx"
            class="sb-cell"
            :class="cellClass(idx)"
          >
            <div class="sb-cell-idx">{{ idx }}</div>
            <div class="sb-cell-val">{{ v }}</div>
            <div class="sb-cell-pointers">
              <span
                v-for="p in pointersAt(idx)"
                :key="p.name"
                class="sb-pointer"
                :style="{ background: p.color }"
              >{{ p.name }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 栈/堆（垂直） -->
      <div class="sb-stack">
        <div class="sb-array-label">{{ state.stackLabel || '栈' }}（顶 → 底）</div>
        <div v-if="!state.stack || !state.stack.length" class="sb-stack-empty">空</div>
        <div v-else class="sb-stack-cells">
          <div
            v-for="(s, i) in [...state.stack].reverse()"
            :key="i"
            class="sb-stack-cell"
            :class="{ top: i === 0, popping: state.poppingTop && i === 0 }"
          >{{ s }}</div>
        </div>
      </div>
    </div>

    <!-- 输出（如有） -->
    <div v-if="state.result" class="sb-result">
      <span class="sb-tag">result</span>
      <span class="sb-result-val">[{{ state.result.join(', ') }}]</span>
    </div>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  config: { values: any[]; label?: string }
  state: any
}>()

function pointersAt(idx: number) {
  const ps = (props.state.pointers ?? []) as { name: string; idx: number; color: string }[]
  return ps.filter((p) => p.idx === idx)
}

function cellClass(idx: number) {
  const cls: Record<string, boolean> = {}
  const cs = (props.state.cellState ?? {}) as Record<number, string>
  if (cs[idx]) cls[cs[idx]] = true
  if (props.state.resolvedIdx?.includes(idx)) cls['resolved'] = true
  return cls
}
</script>

<style scoped>
.sb-viz { display: flex; flex-direction: column; gap: 10px; }
.sb-scalars { display: flex; flex-wrap: wrap; gap: 8px; }
.sb-scalar {
  display: inline-flex; gap: 6px;
  background: var(--vp-c-bg-mute);
  padding: 3px 10px; border-radius: 999px;
  font-size: 12px; font-family: 'JetBrains Mono', Menlo, monospace;
}
.sb-scalar-key { color: var(--vp-c-text-2); }
.sb-scalar-val { font-weight: 700; }

.sb-row {
  display: grid;
  grid-template-columns: minmax(0, 2.2fr) minmax(120px, 1fr);
  gap: 12px;
  align-items: start;
}
@media (max-width: 700px) {
  .sb-row { grid-template-columns: 1fr; }
}

.sb-array, .sb-stack {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px 12px;
}
.sb-array-label { font-size: 11px; color: var(--vp-c-text-2); margin-bottom: 6px; font-weight: 600; }

.sb-cells {
  display: flex; gap: 4px;
  overflow-x: auto;
  padding-top: 16px;
}
.sb-cell {
  display: flex; flex-direction: column; align-items: center;
  flex-shrink: 0;
  min-width: 36px;
  padding: 4px 6px 6px;
  border-radius: 6px;
  background: var(--vp-c-bg);
  border: 1.5px solid var(--vp-c-divider);
  font-family: 'JetBrains Mono', Menlo, monospace;
  position: relative;
  transition: all 0.15s;
}
.sb-cell-idx { font-size: 9.5px; color: var(--vp-c-text-3); }
.sb-cell-val { font-size: 13px; font-weight: 700; }
.sb-cell.in-stack { background: rgba(168, 85, 247, 0.12); border-color: #a855f7; color: #7c3aed; }
.sb-cell.resolved { background: rgba(59, 130, 246, 0.12); border-color: #3b82f6; color: #1d4ed8; }
.sb-cell.popped   { background: rgba(34, 197, 94, 0.18); border-color: #22c55e; color: #16a34a; }
.sb-cell.current  {
  background: rgba(245, 158, 11, 0.20);
  border-color: #f59e0b;
  color: #b45309;
  box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.22);
}
.sb-cell-pointers {
  position: absolute; top: -16px; left: 0; right: 0;
  display: flex; justify-content: center; gap: 2px;
}
.sb-pointer {
  font-size: 10px; font-weight: 700; color: white;
  padding: 1px 5px; border-radius: 3px; white-space: nowrap;
}

.sb-stack-empty { font-size: 12px; color: var(--vp-c-text-3); font-style: italic; }
.sb-stack-cells {
  display: flex; flex-direction: column; gap: 2px;
}
.sb-stack-cell {
  font-family: 'JetBrains Mono', Menlo, monospace;
  font-size: 12px; font-weight: 700;
  padding: 4px 8px;
  border-radius: 4px;
  border: 1.5px solid #a855f7;
  background: rgba(168, 85, 247, 0.12);
  color: #7c3aed;
  text-align: center;
}
.sb-stack-cell.top { box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.3); }
.sb-stack-cell.popping { animation: sb-fly 0.3s ease-out; }
@keyframes sb-fly {
  0% { transform: translateY(0); opacity: 1; }
  100% { transform: translateY(-12px); opacity: 0; }
}

.sb-result {
  display: flex; gap: 10px; align-items: center;
  padding: 6px 10px;
  background: rgba(34, 197, 94, 0.10);
  border-left: 3px solid #16a34a;
  border-radius: 0 6px 6px 0;
}
.sb-tag {
  background: var(--vp-c-bg-mute);
  padding: 2px 8px; border-radius: 4px;
  font-size: 11px; color: var(--vp-c-text-2); font-weight: 700;
}
.sb-result-val {
  font-family: 'JetBrains Mono', Menlo, monospace;
  font-weight: 700; color: #166534; font-size: 13px;
}
</style>
