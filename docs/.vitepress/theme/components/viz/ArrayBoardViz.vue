<template>
  <div class="ab-viz">
    <!-- 标量行 -->
    <div v-if="state.scalars && state.scalars.length" class="ab-scalars">
      <span v-for="s in state.scalars" :key="s.label" class="ab-scalar" :class="s.cls">
        <span class="ab-scalar-key">{{ s.label }}</span>
        <span class="ab-scalar-val">{{ s.value }}</span>
      </span>
    </div>

    <!-- 主数组 -->
    <div class="ab-array">
      <div v-if="config.label" class="ab-array-label">{{ config.label }}</div>
      <div class="ab-cells">
        <div
          v-for="(v, idx) in config.values"
          :key="idx"
          class="ab-cell"
          :class="cellClass(idx)"
        >
          <div class="ab-cell-idx">{{ idx }}</div>
          <div class="ab-cell-val">{{ v }}</div>
          <div class="ab-cell-pointers">
            <span
              v-for="p in pointersAt(idx)"
              :key="p.name"
              class="ab-pointer"
              :style="{ background: p.color }"
            >{{ p.name }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 副数组 / 第二行 -->
    <div v-if="state.secondary" class="ab-array secondary">
      <div class="ab-array-label">{{ state.secondary.label }}</div>
      <div class="ab-cells">
        <div
          v-for="(v, idx) in state.secondary.values"
          :key="idx"
          class="ab-cell sec"
          :class="state.secondary.cellCls?.[idx]"
        >
          <div class="ab-cell-idx">{{ idx }}</div>
          <div class="ab-cell-val">{{ v }}</div>
        </div>
      </div>
    </div>

    <!-- KV 面板 -->
    <div v-if="state.kv" class="ab-kv">
      <div class="ab-kv-title">{{ state.kv.title }}</div>
      <div v-if="!state.kv.entries.length" class="ab-kv-empty">{{ state.kv.emptyLabel || '空' }}</div>
      <div v-else class="ab-kv-grid">
        <div
          v-for="e in state.kv.entries"
          :key="String(e.key)"
          class="ab-kv-entry"
          :class="{ hit: e.highlight }"
        >
          <span class="ab-kv-key">{{ e.key }}</span>
          <span class="ab-kv-arrow">→</span>
          <span class="ab-kv-val">{{ e.value }}</span>
        </div>
      </div>
    </div>

    <!-- 结果（如已有解） -->
    <div v-if="state.result !== undefined" class="ab-result">
      <span class="ab-tag">result</span>
      <span class="ab-result-val">{{ formatResult(state.result) }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

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
  const cs = props.state.cellState as Record<number, string> | undefined
  if (cs && cs[idx]) cls[cs[idx]] = true
  // window range
  const w = props.state.window as { l: number; r: number } | undefined
  if (w && idx >= w.l && idx <= w.r) cls['in-window'] = true
  return cls
}

function formatResult(r: any) {
  if (Array.isArray(r)) return JSON.stringify(r)
  return String(r)
}
</script>

<style scoped>
.ab-viz { display: flex; flex-direction: column; gap: 12px; }

.ab-scalars { display: flex; flex-wrap: wrap; gap: 8px; }
.ab-scalar {
  display: inline-flex;
  gap: 6px;
  align-items: center;
  background: var(--vp-c-bg-mute);
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-family: 'JetBrains Mono', Menlo, monospace;
}
.ab-scalar-key { color: var(--vp-c-text-2); font-weight: 600; }
.ab-scalar-val { font-weight: 700; color: var(--vp-c-text-1); }
.ab-scalar.ok  { background: rgba(34, 197, 94, 0.18); color: #16a34a; }
.ab-scalar.warn { background: rgba(245, 158, 11, 0.18); color: #b45309; }
.ab-scalar.bad { background: rgba(239, 68, 68, 0.18); color: #dc2626; }

.ab-array {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px 12px;
}
.ab-array.secondary { background: var(--vp-c-bg); }
.ab-array-label { font-size: 11px; color: var(--vp-c-text-2); margin-bottom: 6px; font-weight: 600; }
.ab-cells {
  display: flex;
  gap: 4px;
  overflow-x: auto;
  padding-bottom: 4px;
}
.ab-cell {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex-shrink: 0;
  min-width: 38px;
  padding: 4px 6px 6px;
  border-radius: 6px;
  background: var(--vp-c-bg);
  border: 1.5px solid var(--vp-c-divider);
  font-family: 'JetBrains Mono', Menlo, monospace;
  position: relative;
  transition: all 0.15s;
}
.ab-cell-idx { font-size: 9.5px; color: var(--vp-c-text-3); }
.ab-cell-val { font-size: 13px; font-weight: 700; color: var(--vp-c-text-1); }
.ab-cell.in-window { background: rgba(245, 158, 11, 0.12); border-color: #f59e0b; }
.ab-cell.visited   { background: rgba(148, 163, 184, 0.18); color: var(--vp-c-text-3); }
.ab-cell.found     { background: rgba(34, 197, 94, 0.22); border-color: #16a34a; box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.18); }
.ab-cell.skipped   { opacity: 0.4; text-decoration: line-through; }
.ab-cell.left-half { background: rgba(59, 130, 246, 0.10); }
.ab-cell.right-half { background: rgba(168, 85, 247, 0.10); }
.ab-cell-pointers {
  position: absolute;
  top: -16px;
  left: 0;
  right: 0;
  display: flex;
  justify-content: center;
  gap: 2px;
}
.ab-pointer {
  font-size: 10px;
  font-weight: 700;
  color: white;
  padding: 1px 5px;
  border-radius: 3px;
  white-space: nowrap;
}

.ab-kv {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px 12px;
}
.ab-kv-title { font-size: 11px; color: var(--vp-c-text-2); font-weight: 700; margin-bottom: 6px; }
.ab-kv-empty { font-size: 12px; color: var(--vp-c-text-3); font-style: italic; }
.ab-kv-grid { display: flex; flex-wrap: wrap; gap: 6px; }
.ab-kv-entry {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  padding: 3px 8px;
  font-size: 12px;
  font-family: 'JetBrains Mono', Menlo, monospace;
}
.ab-kv-entry.hit {
  background: rgba(34, 197, 94, 0.18);
  border-color: #16a34a;
  box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.18);
}
.ab-kv-key { font-weight: 700; }
.ab-kv-arrow { color: var(--vp-c-text-3); }
.ab-kv-val { color: var(--vp-c-brand-1); }

.ab-result {
  display: flex; gap: 10px; align-items: center;
  padding: 6px 10px;
  background: rgba(34, 197, 94, 0.12);
  border: 1px solid #16a34a;
  border-radius: 8px;
  font-family: 'JetBrains Mono', Menlo, monospace;
  font-size: 13px;
}
.ab-tag {
  background: var(--vp-c-bg);
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  color: var(--vp-c-text-2);
  font-weight: 700;
}
.ab-result-val { font-weight: 700; color: #16a34a; }
</style>
