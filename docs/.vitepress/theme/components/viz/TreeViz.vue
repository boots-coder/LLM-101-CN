<template>
  <div class="tv-viz">
    <div v-if="state.scalars && state.scalars.length" class="tv-scalars">
      <span v-for="s in state.scalars" :key="s.label" class="tv-scalar">
        <span class="tv-scalar-key">{{ s.label }}</span>
        <span class="tv-scalar-val">{{ s.value }}</span>
      </span>
    </div>

    <!-- 树 -->
    <div class="tv-tree-wrap">
      <TvNode :node="config.tree" :state="state" :is-root="true" />
    </div>

    <!-- 栈/队列 -->
    <div v-if="state.stack" class="tv-aux">
      <span class="tv-tag">栈/递归</span>
      <div class="tv-aux-cells">
        <div v-if="!state.stack.length" class="tv-aux-empty">空</div>
        <div v-for="(s, i) in state.stack" :key="i" class="tv-aux-cell stack">{{ s }}</div>
      </div>
    </div>
    <div v-if="state.queue" class="tv-aux">
      <span class="tv-tag">队列</span>
      <div class="tv-aux-cells">
        <div v-if="!state.queue.length" class="tv-aux-empty">空</div>
        <div v-for="(q, i) in state.queue" :key="i" class="tv-aux-cell queue">{{ q }}</div>
      </div>
    </div>

    <!-- 输出 -->
    <div v-if="state.output !== undefined" class="tv-output">
      <span class="tv-tag">输出</span>
      <code>{{ formatOutput(state.output) }}</code>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, defineComponent, h } from 'vue'

const props = defineProps<{
  config: { tree: any }
  state: any
}>()

function formatOutput(o: any) {
  if (Array.isArray(o)) {
    if (o.length && Array.isArray(o[0])) return JSON.stringify(o)
    return '[' + o.join(', ') + ']'
  }
  return String(o)
}

const TvNode: any = defineComponent({
  name: 'TvNode',
  props: {
    node: { type: Object, required: true },
    state: { type: Object, required: true },
    isRoot: { type: Boolean, default: false },
  },
  setup(p) {
    const cls = computed(() => {
      const id = p.node.id
      if (p.state.currentId === id) return 'current'
      if (p.state.visitedIds?.includes(id)) return 'visited'
      if (p.state.stackIds?.includes(id)) return 'on-stack'
      return ''
    })
    return () =>
      h('div', { class: ['tv-node', cls.value, p.isRoot ? 'root' : ''] }, [
        h('div', { class: 'tv-node-label' }, [
          String(p.node.val),
          p.state.visitedIds?.includes(p.node.id) ? h('span', { class: 'tv-check' }, '✓') : null,
        ]),
        (p.node.left || p.node.right)
          ? h('div', { class: 'tv-children' }, [
              h('div', { class: 'tv-child slot-l' }, p.node.left
                ? [h(TvNode, { node: p.node.left, state: p.state })]
                : [h('div', { class: 'tv-empty' }, '·')]),
              h('div', { class: 'tv-child slot-r' }, p.node.right
                ? [h(TvNode, { node: p.node.right, state: p.state })]
                : [h('div', { class: 'tv-empty' }, '·')]),
            ])
          : null,
      ])
  },
})
</script>

<style scoped>
.tv-viz { display: flex; flex-direction: column; gap: 12px; }
.tv-scalars { display: flex; flex-wrap: wrap; gap: 8px; }
.tv-scalar {
  display: inline-flex; gap: 6px;
  background: var(--vp-c-bg-mute);
  padding: 3px 10px; border-radius: 999px;
  font-size: 12px; font-family: 'JetBrains Mono', Menlo, monospace;
}
.tv-scalar-key { color: var(--vp-c-text-2); }
.tv-scalar-val { font-weight: 700; }

.tv-tree-wrap {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 14px 12px;
  overflow-x: auto;
}
.tv-node { display: flex; flex-direction: column; align-items: center; }
.tv-node-label {
  width: 34px; height: 34px;
  border-radius: 50%;
  border: 1.5px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  display: flex; align-items: center; justify-content: center;
  font-family: 'JetBrains Mono', Menlo, monospace;
  font-weight: 700; font-size: 13px;
  position: relative;
  transition: all 0.15s;
}
.tv-children {
  display: flex;
  margin-top: 14px;
  gap: 8px;
  position: relative;
}
.tv-children::before {
  content: '';
  position: absolute;
  top: -14px; left: 0; right: 0; height: 14px;
  border-top: 1.5px solid var(--vp-c-divider);
  border-left: 1.5px solid var(--vp-c-divider);
  border-right: 1.5px solid var(--vp-c-divider);
  border-radius: 6px 6px 0 0;
}
.tv-child { display: flex; flex-direction: column; align-items: center; }
.tv-empty {
  width: 18px; height: 18px;
  border: 1px dashed var(--vp-c-divider);
  border-radius: 50%;
  color: var(--vp-c-text-3);
  display: flex; align-items: center; justify-content: center;
  font-size: 10px;
}

.tv-node.on-stack > .tv-node-label { background: rgba(168, 85, 247, 0.12); border-color: #a855f7; color: #7c3aed; }
.tv-node.visited > .tv-node-label { background: rgba(59, 130, 246, 0.12); border-color: #3b82f6; color: #1d4ed8; }
.tv-node.current > .tv-node-label {
  background: #22c55e; color: white; border-color: #16a34a;
  box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.25);
  transform: scale(1.08);
}
.tv-check {
  position: absolute; right: -6px; bottom: -4px;
  font-size: 11px; color: #16a34a; background: white;
  border-radius: 50%; padding: 1px 3px;
  font-weight: 800;
}

.tv-aux { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.tv-tag {
  background: var(--vp-c-bg-mute);
  font-size: 11px; font-weight: 700;
  padding: 2px 8px; border-radius: 4px;
  color: var(--vp-c-text-2);
}
.tv-aux-cells { display: flex; gap: 4px; flex-wrap: wrap; }
.tv-aux-empty { font-size: 12px; color: var(--vp-c-text-3); font-style: italic; }
.tv-aux-cell {
  font-family: 'JetBrains Mono', Menlo, monospace;
  font-size: 12px; font-weight: 700;
  padding: 3px 8px;
  border-radius: 4px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
}
.tv-aux-cell.stack { background: rgba(168, 85, 247, 0.12); border-color: #a855f7; color: #7c3aed; }
.tv-aux-cell.queue { background: rgba(59, 130, 246, 0.12); border-color: #3b82f6; color: #1d4ed8; }

.tv-output {
  display: flex; gap: 8px; align-items: center;
  padding: 6px 10px;
  background: rgba(34, 197, 94, 0.10);
  border-left: 3px solid #16a34a;
  border-radius: 0 6px 6px 0;
}
.tv-output code {
  background: transparent; padding: 0;
  font-family: 'JetBrains Mono', Menlo, monospace;
  font-size: 13px; font-weight: 700; color: #166534;
}
</style>
