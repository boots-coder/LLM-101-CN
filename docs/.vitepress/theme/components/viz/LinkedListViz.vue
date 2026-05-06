<template>
  <div class="ll-viz">
    <div v-if="state.scalars && state.scalars.length" class="ll-scalars">
      <span v-for="s in state.scalars" :key="s.label" class="ll-scalar">
        <span class="ll-scalar-key">{{ s.label }}</span>
        <span class="ll-scalar-val">{{ s.value }}</span>
      </span>
    </div>

    <div v-for="(chain, ci) in state.chains" :key="ci" class="ll-chain">
      <div v-if="chain.label" class="ll-chain-label">{{ chain.label }}</div>
      <div class="ll-nodes">
        <template v-for="(node, ni) in chain.nodes" :key="ni">
          <div class="ll-node-wrap">
            <div class="ll-pointers-top">
              <span
                v-for="p in pointersAt(ci, ni)"
                :key="p.name"
                class="ll-pointer"
                :style="{ background: p.color }"
              >{{ p.name }}</span>
            </div>
            <div class="ll-node" :class="node.cls">{{ node.val }}</div>
          </div>
          <div v-if="ni < chain.nodes.length - 1 || chain.tailArrow" class="ll-arrow">→</div>
        </template>
        <div v-if="chain.tailLabel" class="ll-tail">{{ chain.tailLabel }}</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  config: any
  state: any
}>()

function pointersAt(ci: number, ni: number) {
  const ps = (props.state.pointers ?? []) as { name: string; chain: number; idx: number; color: string }[]
  return ps.filter((p) => p.chain === ci && p.idx === ni)
}
</script>

<style scoped>
.ll-viz { display: flex; flex-direction: column; gap: 12px; }
.ll-scalars { display: flex; flex-wrap: wrap; gap: 8px; }
.ll-scalar {
  display: inline-flex; gap: 6px;
  background: var(--vp-c-bg-mute);
  padding: 3px 10px; border-radius: 999px;
  font-size: 12px; font-family: 'JetBrains Mono', Menlo, monospace;
}
.ll-scalar-key { color: var(--vp-c-text-2); }
.ll-scalar-val { font-weight: 700; }

.ll-chain {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 14px 12px 10px;
  overflow-x: auto;
}
.ll-chain-label { font-size: 11px; color: var(--vp-c-text-2); margin-bottom: 8px; font-weight: 600; }

.ll-nodes { display: flex; gap: 4px; align-items: center; min-width: max-content; }
.ll-node-wrap { display: flex; flex-direction: column; align-items: center; position: relative; }
.ll-pointers-top {
  display: flex; gap: 2px;
  position: absolute;
  top: -16px;
}
.ll-pointer {
  font-size: 10px;
  font-weight: 700;
  color: white;
  padding: 1px 5px;
  border-radius: 3px;
  white-space: nowrap;
}
.ll-node {
  width: 36px; height: 36px;
  border-radius: 6px;
  background: var(--vp-c-bg);
  border: 1.5px solid var(--vp-c-divider);
  display: flex; align-items: center; justify-content: center;
  font-family: 'JetBrains Mono', Menlo, monospace;
  font-weight: 700; font-size: 13px;
  transition: all 0.15s;
}
.ll-node.detached { opacity: 0.45; border-style: dashed; }
.ll-node.head     { background: rgba(34, 197, 94, 0.12); border-color: #22c55e; color: #16a34a; }
.ll-node.cycle    { background: rgba(239, 68, 68, 0.12); border-color: #ef4444; color: #dc2626; }
.ll-arrow { color: var(--vp-c-text-3); font-weight: 800; font-size: 16px; }
.ll-tail { font-size: 11px; color: var(--vp-c-text-3); font-family: 'JetBrains Mono', Menlo, monospace; padding: 0 4px; }
</style>
