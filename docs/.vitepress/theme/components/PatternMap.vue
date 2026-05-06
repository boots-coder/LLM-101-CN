<template>
  <div class="pattern-map">
    <header class="pm-header">
      <div class="pm-stats">
        <div class="pm-stat">
          <span class="pm-num">{{ state.xp }}</span>
          <span class="pm-label">XP</span>
        </div>
        <div class="pm-stat">
          <span class="pm-num">🔥 {{ state.streak.count }}</span>
          <span class="pm-label">连续天数</span>
        </div>
        <div class="pm-stat">
          <span class="pm-num">{{ completedCount }} / {{ totalProblems }}</span>
          <span class="pm-label">已通关</span>
        </div>
        <div class="pm-stat">
          <span class="pm-num">{{ state.dailyGoal.xpToday }} / {{ state.dailyGoal.xpTarget }}</span>
          <span class="pm-label">今日目标</span>
        </div>
      </div>
    </header>

    <div v-for="day in [1,2,3,4,5]" :key="day" class="pm-day">
      <h2 class="pm-day-title">D{{ day }} · {{ dayName(day) }}</h2>
      <div class="pm-path">
        <div
          v-for="(p, idx) in patternsByDay(day as 1|2|3|4|5)"
          :key="p.slug"
          class="pm-node-wrap"
          :class="['offset-' + (idx % 3)]"
        >
          <a :href="patternHref(p.slug)" class="pm-node" :class="{ done: isPatternDone(p) }">
            <span class="pm-node-icon">{{ patternIcon(p.slug) }}</span>
            <span class="pm-node-name">{{ p.name.zh }}</span>
            <span class="pm-node-progress">{{ doneCountIn(p) }} / {{ p.problemIds.length }}</span>
          </a>
          <div class="pm-node-hint">{{ p.oneLiner }}</div>
        </div>
      </div>
    </div>

    <footer class="pm-footer">
      <a class="pm-link" :href="srsHref">📚 今日复习</a>
      <a class="pm-link" :href="cheatHref">🐍 Python 速查</a>
      <button class="pm-link reset" @click="confirmReset">⟲ 重置进度</button>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { patterns, patternsByDay } from '../../../leetcode/_data/patterns'
import { problems } from '../../../leetcode/_data/problems/index'
import { loadState, refillHearts, saveState, resetAll } from '../../../leetcode/_data/progress'
import { useData, withBase } from 'vitepress'

const state = ref(loadState())

onMounted(() => {
  state.value = refillHearts(loadState())
  saveState(state.value)
})

const totalProblems = computed(() =>
  patterns.reduce((s, p) => s + p.problemIds.length, 0)
)

const completedCount = computed(() =>
  Object.keys(state.value.problemsCompleted).length
)

function isPatternDone(p: { problemIds: string[] }): boolean {
  return p.problemIds.every(id => !!state.value.problemsCompleted[id])
}

function doneCountIn(p: { problemIds: string[] }): number {
  return p.problemIds.filter(id => !!state.value.problemsCompleted[id]).length
}

function dayName(day: number): string {
  const ps = patternsByDay(day as 1|2|3|4|5).map(p => p.name.zh).join(' / ')
  return ps
}

function patternHref(slug: string): string {
  return withBase(`/leetcode/patterns/${slug}`)
}

const srsHref = withBase('/leetcode/srs')
const cheatHref = withBase('/leetcode/python-cheatsheet')

function patternIcon(slug: string): string {
  const map: Record<string, string> = {
    hashmap: '🗂️',
    'two-pointer': '👆',
    'sliding-window': '🪟',
    'binary-search': '🎯',
    'monotonic-stack': '📚',
    heap: '🏔️',
    'linked-list': '🔗',
    'tree-dfs': '🌳',
    'tree-bfs': '🌲',
    trie: '🔤',
    backtracking: '🎲',
    graph: '🕸️',
    'dp-1d': '📈',
    'dp-2d': '🧊',
    'bit-prefix': '⚡',
  }
  return map[slug] || '🧩'
}

function confirmReset() {
  if (typeof window === 'undefined') return
  if (window.confirm('确认重置所有 LC 学习进度？此操作不可恢复。')) {
    resetAll()
    state.value = loadState()
  }
}
</script>

<style scoped>
.pattern-map {
  max-width: 880px;
  margin: 0 auto;
}
.pm-header {
  background: var(--vp-c-bg-soft);
  border-radius: 16px;
  padding: 20px 24px;
  margin: 16px 0 28px;
  border: 2px solid var(--vp-c-divider);
}
.pm-stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
}
.pm-stat { display: flex; flex-direction: column; align-items: center; }
.pm-num { font-size: 22px; font-weight: 800; color: var(--vp-c-brand-1); }
.pm-label { font-size: 12px; color: var(--vp-c-text-2); margin-top: 2px; }

.pm-day { margin: 30px 0; }
.pm-day-title {
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 14px;
  padding-bottom: 6px;
  border-bottom: 2px dashed var(--vp-c-divider);
}

.pm-path {
  display: flex;
  flex-direction: column;
  gap: 18px;
}
.pm-node-wrap {
  display: flex;
  flex-direction: column;
  gap: 4px;
  width: 70%;
}
.pm-node-wrap.offset-0 { align-self: flex-start; }
.pm-node-wrap.offset-1 { align-self: center; }
.pm-node-wrap.offset-2 { align-self: flex-end; }

.pm-node {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 18px;
  background: var(--vp-c-bg-soft);
  border-radius: 14px;
  border: 2px solid var(--vp-c-divider);
  text-decoration: none !important;
  color: var(--vp-c-text-1) !important;
  font-weight: 600;
  transition: transform 0.15s, border-color 0.15s, box-shadow 0.15s;
}
.pm-node:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.pm-node.done {
  background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(234, 179, 8, 0.06));
  border-color: #f59e0b;
}

.pm-node-icon { font-size: 24px; }
.pm-node-name { flex: 1; font-size: 15px; }
.pm-node-progress {
  font-size: 12px;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-mute);
  padding: 2px 8px;
  border-radius: 6px;
}
.pm-node.done .pm-node-progress {
  background: #f59e0b;
  color: white;
}

.pm-node-hint {
  font-size: 12px;
  color: var(--vp-c-text-2);
  padding-left: 18px;
}

.pm-footer {
  display: flex;
  justify-content: center;
  gap: 16px;
  margin: 36px 0 16px;
  flex-wrap: wrap;
}
.pm-link {
  text-decoration: none !important;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  padding: 10px 18px;
  border-radius: 10px;
  font-size: 14px;
  font-weight: 600;
  color: var(--vp-c-text-1) !important;
  cursor: pointer;
}
.pm-link:hover { border-color: var(--vp-c-brand-1); }
.pm-link.reset { background: transparent; color: var(--vp-c-text-2) !important; }
</style>
