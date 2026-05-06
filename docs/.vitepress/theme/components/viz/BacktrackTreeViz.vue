<template>
  <div class="bt-viz">
    <!-- 决策树 -->
    <div class="bt-tree-wrap">
      <div class="bt-tree">
        <BtNode :node="config.tree" :state="treeState" :is-root="true" />
      </div>
      <div class="bt-legend">
        <span class="lg current">●</span> 当前
        <span class="lg on-path">●</span> 路径祖先
        <span class="lg leaf">✓</span> 已收集叶子
        <span class="lg unvisited">○</span> 待探索
      </div>
    </div>

    <!-- 状态面板 -->
    <div class="bt-state">
      <div class="bt-state-row">
        <span class="bt-tag">path</span>
        <span class="bt-arr">[<template v-for="(v, i) in state.path" :key="i"><span class="bt-cell">{{ v }}</span></template>]</span>
      </div>
      <div class="bt-state-row">
        <span class="bt-tag">used</span>
        <span class="bt-arr">[<template v-for="(u, i) in state.used" :key="i"><span class="bt-cell" :class="{ on: u }">{{ u ? 'T' : 'F' }}</span></template>]</span>
        <span class="bt-idx">↑ 下标 {{ usedIdxLabel }}</span>
      </div>
      <div class="bt-state-row">
        <span class="bt-tag">res</span>
        <span class="bt-arr">
          <template v-if="state.res.length === 0">[]</template>
          <template v-else>
            [<span v-for="(r, i) in state.res" :key="i" class="bt-res">[{{ r.join(',') }}]<template v-if="i < state.res.length - 1">, </template></span>]
          </template>
        </span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, defineComponent, h } from 'vue'

const props = defineProps<{
  config: { nums: number[]; tree: any }
  state: any
}>()

const usedIdxLabel = computed(() => props.config.nums.map((_, i) => i).join(' '))

// 计算每个节点的状态
const treeState = computed(() => {
  const cur: string = (props.state.pathIdx as number[]).join('.')
  const ancestors = new Set<string>()
  for (let k = 0; k < props.state.pathIdx.length; k++) {
    ancestors.add((props.state.pathIdx as number[]).slice(0, k).join('.'))
  }
  ancestors.add('')
  ancestors.delete(cur)
  const leafSet = new Set<string>(props.state.collectedLeafKeys)
  return { current: cur, ancestors, leafSet }
})

// 递归节点组件，本地定义避免再多一个文件
const BtNode = defineComponent({
  name: 'BtNode',
  props: {
    node: { type: Object, required: true },
    state: { type: Object, required: true },
    isRoot: { type: Boolean, default: false },
  },
  setup(p) {
    const cls = computed(() => {
      const k = p.node.key
      if (k === p.state.current) return 'current'
      if (p.state.ancestors.has(k)) return 'on-path'
      if (p.state.leafSet.has(k)) return 'leaf-done'
      return 'unvisited'
    })
    const label = computed(() => p.node.label === '[]' ? '[ ]' : p.node.label)
    return () =>
      h('div', { class: ['bt-node', cls.value, p.isRoot ? 'root' : ''] }, [
        h('div', { class: 'bt-node-label' }, [
          label.value,
          p.state.leafSet.has(p.node.key) ? h('span', { class: 'bt-check' }, '✓') : null,
          p.node.key === p.state.current ? h('span', { class: 'bt-arrow' }, '◀') : null,
        ]),
        p.node.children.length > 0
          ? h(
              'div',
              { class: 'bt-children' },
              p.node.children.map((c: any) =>
                h(BtNode, { node: c, state: p.state, isRoot: false }),
              ),
            )
          : null,
      ])
  },
})
</script>

<style scoped>
.bt-viz { display: flex; flex-direction: column; gap: 14px; }

.bt-tree-wrap {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 14px 12px 10px;
  overflow-x: auto;
}
.bt-tree { font-size: 12px; line-height: 1.4; min-width: max-content; }

.bt-node {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin: 0 6px;
}
.bt-node.root { margin-top: 0; }
.bt-node-label {
  position: relative;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1.5px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  font-family: 'JetBrains Mono', Menlo, monospace;
  font-weight: 600;
  white-space: nowrap;
  transition: all 0.2s;
}
.bt-children {
  display: flex;
  margin-top: 14px;
  gap: 4px;
  position: relative;
}
.bt-children::before {
  content: '';
  position: absolute;
  top: -14px;
  left: 0;
  right: 0;
  height: 14px;
  border-top: 1.5px solid var(--vp-c-divider);
  border-left: 1.5px solid var(--vp-c-divider);
  border-right: 1.5px solid var(--vp-c-divider);
  border-radius: 6px 6px 0 0;
  border-bottom: none;
  pointer-events: none;
}

.bt-node.unvisited > .bt-node-label { color: var(--vp-c-text-3); opacity: 0.55; }
.bt-node.on-path > .bt-node-label { background: rgba(34, 197, 94, 0.12); border-color: #22c55e; color: #16a34a; }
.bt-node.current > .bt-node-label {
  background: #22c55e;
  color: white;
  border-color: #16a34a;
  box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.25);
  transform: scale(1.06);
}
.bt-node.leaf-done > .bt-node-label {
  background: rgba(59, 130, 246, 0.12);
  border-color: #3b82f6;
  color: #1d4ed8;
}

.bt-check { margin-left: 6px; color: #16a34a; font-weight: 800; }
.bt-arrow {
  position: absolute;
  right: -16px;
  top: 50%;
  transform: translateY(-50%);
  color: #f59e0b;
  font-size: 14px;
  animation: bt-pulse 0.9s ease-in-out infinite;
}
@keyframes bt-pulse {
  0%, 100% { transform: translateY(-50%) translateX(0); opacity: 1; }
  50%      { transform: translateY(-50%) translateX(-4px); opacity: 0.5; }
}
@media (prefers-reduced-motion: reduce) {
  .bt-arrow { animation: none; }
  .bt-node.current > .bt-node-label { transform: none; }
}

.bt-legend {
  font-size: 11px;
  color: var(--vp-c-text-2);
  margin-top: 12px;
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
}
.bt-legend .lg { font-weight: 700; margin-right: 2px; }
.bt-legend .lg.current { color: #22c55e; }
.bt-legend .lg.on-path { color: #86efac; }
.bt-legend .lg.leaf { color: #3b82f6; }
.bt-legend .lg.unvisited { color: var(--vp-c-text-3); }

.bt-state {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 13px;
  font-family: 'JetBrains Mono', Menlo, monospace;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.bt-state-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.bt-tag {
  background: var(--vp-c-bg-mute);
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11.5px;
  font-weight: 700;
  color: var(--vp-c-text-2);
  flex-shrink: 0;
  min-width: 38px;
  text-align: center;
}
.bt-arr { font-weight: 600; }
.bt-cell {
  display: inline-block;
  min-width: 18px;
  padding: 1px 6px;
  margin: 0 2px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  text-align: center;
}
.bt-cell.on { background: rgba(34, 197, 94, 0.18); color: #16a34a; border-color: #22c55e; }
.bt-idx { color: var(--vp-c-text-3); font-size: 11px; letter-spacing: 4px; margin-left: 4px; }
.bt-res { margin-right: 2px; color: #1d4ed8; }
</style>
