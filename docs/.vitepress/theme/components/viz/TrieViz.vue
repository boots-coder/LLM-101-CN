<template>
  <div class="tr-viz">
    <div v-if="state.scalars && state.scalars.length" class="tr-scalars">
      <span v-for="s in state.scalars" :key="s.label" class="tr-scalar">
        <span class="tr-scalar-key">{{ s.label }}</span>
        <span class="tr-scalar-val">{{ s.value }}</span>
      </span>
    </div>

    <div class="tr-tree-wrap">
      <TrNode :node="state.root" :state="state" />
    </div>

    <div v-if="state.result !== undefined" class="tr-output">
      <span class="tr-tag">result</span>
      <code>{{ String(state.result) }}</code>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, defineComponent, h } from 'vue'

const props = defineProps<{
  config: any
  state: any
}>()

const TrNode: any = defineComponent({
  name: 'TrNode',
  props: {
    node: { type: Object, required: true },
    state: { type: Object, required: true },
  },
  setup(p) {
    const cls = computed(() => {
      const id = p.node.id
      if (p.state.currentId === id) return 'current'
      if (p.state.pathIds?.includes(id)) return 'on-path'
      return ''
    })
    return () =>
      h('div', { class: ['tr-node', cls.value, p.node.isEnd ? 'end' : ''] }, [
        h('div', { class: 'tr-label' }, [
          p.node.label === '' ? '·' : p.node.label,
          p.node.isEnd ? h('span', { class: 'tr-end-mark' }, '★') : null,
        ]),
        Object.keys(p.node.children).length > 0
          ? h(
              'div',
              { class: 'tr-children' },
              Object.values(p.node.children).map((c: any) =>
                h(TrNode, { node: c, state: p.state }),
              ),
            )
          : null,
      ])
  },
})
</script>

<style scoped>
.tr-viz { display: flex; flex-direction: column; gap: 12px; }
.tr-scalars { display: flex; flex-wrap: wrap; gap: 8px; }
.tr-scalar {
  display: inline-flex; gap: 6px;
  background: var(--vp-c-bg-mute);
  padding: 3px 10px; border-radius: 999px;
  font-size: 12px; font-family: 'JetBrains Mono', Menlo, monospace;
}
.tr-scalar-key { color: var(--vp-c-text-2); }
.tr-scalar-val { font-weight: 700; }

.tr-tree-wrap {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 14px 12px;
  overflow-x: auto;
}
.tr-node { display: flex; flex-direction: column; align-items: center; margin: 0 4px; }
.tr-label {
  width: 30px; height: 30px;
  border-radius: 50%;
  border: 1.5px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  display: flex; align-items: center; justify-content: center;
  font-family: 'JetBrains Mono', Menlo, monospace;
  font-weight: 700; font-size: 13px;
  position: relative;
  transition: all 0.15s;
}
.tr-children {
  display: flex; gap: 4px;
  margin-top: 14px;
  position: relative;
}
.tr-children::before {
  content: '';
  position: absolute;
  top: -14px; left: 0; right: 0; height: 14px;
  border-top: 1.5px solid var(--vp-c-divider);
  border-left: 1.5px solid var(--vp-c-divider);
  border-right: 1.5px solid var(--vp-c-divider);
  border-radius: 6px 6px 0 0;
}
.tr-node.on-path > .tr-label { background: rgba(34, 197, 94, 0.12); border-color: #22c55e; color: #16a34a; }
.tr-node.current > .tr-label {
  background: #22c55e; color: white; border-color: #16a34a;
  box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.22);
  transform: scale(1.1);
}
.tr-node.end > .tr-label { border-style: double; border-width: 3px; }
.tr-end-mark {
  position: absolute; right: -6px; bottom: -6px;
  color: #f59e0b; font-size: 12px; font-weight: 800;
  text-shadow: 0 0 2px white;
}

.tr-output {
  display: flex; gap: 8px; align-items: center;
  padding: 6px 10px;
  background: rgba(34, 197, 94, 0.10);
  border-left: 3px solid #16a34a;
  border-radius: 0 6px 6px 0;
}
.tr-tag {
  background: var(--vp-c-bg-mute);
  font-size: 11px; font-weight: 700;
  padding: 2px 8px; border-radius: 4px;
  color: var(--vp-c-text-2);
}
.tr-output code {
  background: transparent; padding: 0;
  font-size: 13px; font-weight: 700; color: #166534;
}
</style>
