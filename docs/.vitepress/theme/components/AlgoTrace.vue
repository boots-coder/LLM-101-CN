<template>
  <div class="algo-trace">
    <header class="at-header">
      <div class="at-title">
        <h4>🐞 单步可视化</h4>
        <span class="at-subtitle">{{ activeTrace.title }}</span>
      </div>
      <div v-if="traces.length > 1" class="at-tabs">
        <button
          v-for="(t, i) in traces"
          :key="i"
          class="at-tab"
          :class="{ active: i === activeIdx }"
          @click="selectTrace(i)"
        >{{ t.inputLabel }}</button>
      </div>
    </header>

    <div class="at-body">
      <!-- 左：代码 -->
      <div class="at-code-pane">
        <div class="at-pane-head">
          <span class="at-pane-label">代码</span>
          <span class="at-line-pill">L{{ currentStep.line }}</span>
        </div>
        <pre class="at-code"><code><span
          v-for="(line, idx) in codeLines"
          :key="idx"
          class="at-line"
          :class="{ active: idx + 1 === currentStep.line }"
        ><span class="at-line-no">{{ String(idx + 1).padStart(2, ' ') }}</span><span class="at-line-text" v-html="highlightPythonLine(line) || '&nbsp;'"></span></span></code></pre>
      </div>

      <!-- 右：可视化 -->
      <div class="at-viz-pane">
        <div class="at-pane-head">
          <span class="at-pane-label">数据流</span>
          <span class="at-action-pill" :class="actionClass">{{ actionLabel }}</span>
        </div>
        <BacktrackTreeViz
          v-if="activeTrace.vizKind === 'backtrack-tree'"
          :config="activeTrace.vizConfig"
          :state="currentStep.state"
        />
        <ArrayBoardViz
          v-else-if="activeTrace.vizKind === 'array-board'"
          :config="activeTrace.vizConfig"
          :state="currentStep.state"
        />
        <DPGridViz
          v-else-if="activeTrace.vizKind === 'dp-grid'"
          :config="activeTrace.vizConfig"
          :state="currentStep.state"
        />
        <TreeViz
          v-else-if="activeTrace.vizKind === 'tree'"
          :config="activeTrace.vizConfig"
          :state="currentStep.state"
        />
        <LinkedListViz
          v-else-if="activeTrace.vizKind === 'linked-list'"
          :config="activeTrace.vizConfig"
          :state="currentStep.state"
        />
        <GridViz
          v-else-if="activeTrace.vizKind === 'grid'"
          :config="activeTrace.vizConfig"
          :state="currentStep.state"
        />
        <TrieViz
          v-else-if="activeTrace.vizKind === 'trie'"
          :config="activeTrace.vizConfig"
          :state="currentStep.state"
        />
        <StackBoardViz
          v-else-if="activeTrace.vizKind === 'stack-board'"
          :config="activeTrace.vizConfig"
          :state="currentStep.state"
        />
      </div>
    </div>

    <!-- 旁白 -->
    <div class="at-note" :key="stepIdx">
      <span class="at-step-pill">{{ stepIdx + 1 }} / {{ totalSteps }}</span>
      <span class="at-note-text">{{ currentStep.note }}</span>
    </div>

    <!-- 控制 -->
    <div class="at-controls">
      <button class="at-btn" :disabled="stepIdx === 0" @click="restart" title="重置">⏮</button>
      <button class="at-btn" :disabled="stepIdx === 0" @click="prev" title="上一步">◀ 上一步</button>
      <button class="at-btn primary" @click="togglePlay" :title="playing ? '暂停' : '自动播放'">
        {{ playing ? '⏸ 暂停' : '▶ 播放' }}
      </button>
      <button class="at-btn" :disabled="stepIdx >= totalSteps - 1" @click="next" title="下一步">下一步 ▶</button>
      <button class="at-btn" :disabled="stepIdx >= totalSteps - 1" @click="finish" title="跳到结束">⏭</button>
      <label class="at-speed">
        速度
        <select v-model.number="intervalMs">
          <option :value="1200">慢</option>
          <option :value="600">中</option>
          <option :value="250">快</option>
        </select>
      </label>
    </div>

    <!-- 进度条 -->
    <div class="at-progress">
      <input
        type="range"
        :min="0"
        :max="totalSteps - 1"
        :value="stepIdx"
        @input="seek(($event.target as HTMLInputElement).valueAsNumber)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import BacktrackTreeViz from './viz/BacktrackTreeViz.vue'
import ArrayBoardViz from './viz/ArrayBoardViz.vue'
import DPGridViz from './viz/DPGridViz.vue'
import TreeViz from './viz/TreeViz.vue'
import LinkedListViz from './viz/LinkedListViz.vue'
import GridViz from './viz/GridViz.vue'
import TrieViz from './viz/TrieViz.vue'
import StackBoardViz from './viz/StackBoardViz.vue'
import { highlightPythonLine } from '../utils/highlight-python'
import type { AlgoTrace } from '../../../leetcode/_data/traces/types'

const props = defineProps<{
  traces: AlgoTrace[]
  solutionCode: string
}>()

const codeLines = computed(() => props.solutionCode.split('\n'))

const activeIdx = ref(0)
const stepIdx = ref(0)
const playing = ref(false)
const intervalMs = ref(600)
let timer: ReturnType<typeof setInterval> | null = null

const activeTrace = computed(() => props.traces[activeIdx.value])
const totalSteps = computed(() => activeTrace.value.steps.length)
const currentStep = computed(() => activeTrace.value.steps[stepIdx.value])

const actionLabel = computed(() => {
  const a = currentStep.value.state.action
  return ({
    init: '初始化', enter: '进入', leaf: '收叶子', unchoose: '撤销',
    skip: '跳过', return: '返回', done: '完成',
  } as Record<string, string>)[a] ?? a ?? ''
})
const actionClass = computed(() => `act-${currentStep.value.state.action ?? 'init'}`)

function selectTrace(i: number) {
  pause()
  activeIdx.value = i
  stepIdx.value = 0
}

function next() {
  if (stepIdx.value < totalSteps.value - 1) stepIdx.value++
  else pause()
}
function prev() { if (stepIdx.value > 0) stepIdx.value-- }
function restart() { pause(); stepIdx.value = 0 }
function finish() { pause(); stepIdx.value = totalSteps.value - 1 }
function seek(v: number) { pause(); stepIdx.value = v }

function play() {
  if (timer) return
  playing.value = true
  timer = setInterval(() => {
    if (stepIdx.value >= totalSteps.value - 1) { pause(); return }
    stepIdx.value++
  }, intervalMs.value)
}
function pause() {
  playing.value = false
  if (timer) { clearInterval(timer); timer = null }
}
function togglePlay() { playing.value ? pause() : play() }

watch(intervalMs, () => { if (playing.value) { pause(); play() } })
onUnmounted(pause)
</script>

<style scoped>
.algo-trace {
  margin: 18px 0 24px;
  padding: 14px 16px;
  background: var(--vp-c-bg);
  border: 2px solid var(--vp-c-divider);
  border-left: 4px solid #f59e0b;
  border-radius: 12px;
}

.at-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 10px;
}
.at-title h4 { margin: 0; font-size: 15px; }
.at-subtitle { font-size: 12px; color: var(--vp-c-text-2); margin-left: 6px; }

.at-tabs { display: flex; gap: 6px; flex-wrap: wrap; }
.at-tab {
  font-size: 11.5px;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  cursor: pointer;
  font-family: 'JetBrains Mono', Menlo, monospace;
  color: var(--vp-c-text-2);
}
.at-tab.active { background: #f59e0b; color: white; border-color: #d97706; }

.at-body {
  display: grid;
  grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr);
  gap: 12px;
}
@media (max-width: 860px) {
  .at-body { grid-template-columns: 1fr; }
}

.at-code-pane, .at-viz-pane {
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.at-pane-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 4px 6px;
  font-size: 11.5px;
  color: var(--vp-c-text-2);
}
.at-pane-label { font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; }

.at-line-pill {
  background: #f59e0b;
  color: white;
  padding: 2px 8px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 11px;
  font-family: 'JetBrains Mono', Menlo, monospace;
}

.at-action-pill {
  padding: 2px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 11px;
}
.act-init     { background: rgba(100, 116, 139, 0.15); color: #475569; }
.act-enter    { background: rgba(34, 197, 94, 0.15);   color: #16a34a; }
.act-leaf     { background: rgba(59, 130, 246, 0.18);  color: #1d4ed8; }
.act-unchoose { background: rgba(245, 158, 11, 0.18);  color: #b45309; }
.act-skip     { background: rgba(148, 163, 184, 0.18); color: #475569; }
.act-return   { background: rgba(168, 85, 247, 0.15);  color: #7c3aed; }
.act-done     { background: rgba(34, 197, 94, 0.22);   color: #166534; }

.at-code {
  background: #1e1e2e;
  color: #e4e4e8;
  border-radius: 8px;
  padding: 12px 12px;
  font-size: 12.5px;
  line-height: 1.6;
  margin: 0;
  overflow-x: auto;
  font-family: 'JetBrains Mono', 'Fira Code', Menlo, Consolas, monospace;
  max-height: 480px;
  overflow-y: auto;
}
.at-code code { background: transparent; color: inherit; padding: 0; font-size: inherit; }
.at-line {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  padding: 0 4px;
  white-space: pre;
  border-radius: 3px;
  transition: background 0.15s;
}
.at-line.active {
  background: rgba(245, 158, 11, 0.28);
  box-shadow: inset 3px 0 0 #f59e0b;
}
.at-line-no {
  flex-shrink: 0;
  width: 22px;
  text-align: right;
  color: #6a6a7c;
  user-select: none;
  font-size: 11px;
}
.at-line-text { flex: 1; }
.at-code :deep(.hl-kw)  { color: #c678dd; font-weight: 600; }
.at-code :deep(.hl-bi)  { color: #61afef; }
.at-code :deep(.hl-str) { color: #98c379; }
.at-code :deep(.hl-num) { color: #d19a66; }
.at-code :deep(.hl-com) { color: #7d8590; font-style: italic; }
.at-code :deep(.hl-op)  { color: #56b6c2; }

.at-note {
  margin-top: 10px;
  padding: 10px 14px;
  background: rgba(245, 158, 11, 0.1);
  border-left: 3px solid #f59e0b;
  border-radius: 0 8px 8px 0;
  font-size: 13.5px;
  line-height: 1.55;
  display: flex;
  gap: 10px;
  align-items: flex-start;
}
.at-step-pill {
  font-family: 'JetBrains Mono', Menlo, monospace;
  font-size: 11px;
  font-weight: 700;
  background: var(--vp-c-bg-soft);
  padding: 2px 8px;
  border-radius: 4px;
  color: var(--vp-c-text-2);
  flex-shrink: 0;
}

.at-controls {
  margin-top: 12px;
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  align-items: center;
}
.at-btn {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 6px 12px;
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
  color: var(--vp-c-text-1);
}
.at-btn:hover:not(:disabled) { border-color: #f59e0b; color: #b45309; }
.at-btn:disabled { opacity: 0.45; cursor: not-allowed; }
.at-btn.primary { background: #f59e0b; color: white; border-color: #d97706; }
.at-btn.primary:hover { background: #d97706; }

.at-speed {
  margin-left: auto;
  font-size: 12px;
  color: var(--vp-c-text-2);
  display: flex;
  gap: 6px;
  align-items: center;
}
.at-speed select {
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  border-radius: 6px;
  padding: 3px 6px;
  font-size: 12px;
}

.at-progress {
  margin-top: 10px;
}
.at-progress input[type=range] {
  width: 100%;
  accent-color: #f59e0b;
}
</style>
