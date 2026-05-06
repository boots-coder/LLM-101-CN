<template>
  <div v-if="!problem" class="lc-lesson missing">
    <p>题目 <code>{{ problemId }}</code> 暂未上线，敬请期待。</p>
  </div>
  <div v-else class="lc-lesson">
    <header class="lc-header">
      <div class="lc-title-row">
        <span class="lc-no">#{{ problem.leetcodeNo }}</span>
        <span class="lc-title">{{ problem.title.zh }}</span>
        <span class="lc-diff" :class="problem.difficulty">{{ diffText }}</span>
      </div>
      <div class="lc-hud">
        <span class="hud-item" title="心形血量">
          <span v-for="i in hearts.max" :key="i" class="hud-heart" :class="{ off: i > hearts.current }">♥</span>
        </span>
        <span class="hud-item" title="本题 XP">+{{ xpThisLesson }} XP</span>
        <span class="hud-item" title="连续天数">🔥 {{ streak.count }}</span>
      </div>
    </header>

    <div class="lc-progress">
      <div class="lc-progress-bar" :style="{ width: progressPct + '%' }"></div>
      <span class="lc-progress-label">{{ answeredCount }} / {{ total }}</span>
    </div>

    <section class="lc-statement">
      <h4>📝 题目</h4>
      <div class="lc-statement-body" v-html="renderedStatement"></div>
      <div v-if="problem.examples?.length" class="lc-examples">
        <div v-for="(ex, i) in problem.examples" :key="i" class="lc-example">
          <div class="lc-ex-label">示例 {{ i + 1 }}</div>
          <div class="lc-ex-row"><span class="lc-ex-tag">输入</span><code>{{ ex.input }}</code></div>
          <div class="lc-ex-row"><span class="lc-ex-tag">输出</span><code>{{ ex.output }}</code></div>
          <div v-if="ex.note" class="lc-ex-note">💡 {{ ex.note }}</div>
        </div>
      </div>
      <div v-if="problem.constraints?.length" class="lc-constraints">
        <span class="lc-cons-label">约束</span>
        <ul>
          <li v-for="(c, i) in problem.constraints" :key="i"><code>{{ c }}</code></li>
        </ul>
      </div>
    </section>

    <section class="lc-intuition">
      <strong>💡 思路：</strong>{{ problem.intuition }}
    </section>

    <section class="lc-solution">
      <div class="lc-solution-head">
        <h4>✅ 完整解法</h4>
        <span class="lc-complexity">时间 {{ problem.complexity.time }} · 空间 {{ problem.complexity.space }}</span>
      </div>
      <p class="lc-solution-hint">先把整段解法看懂，下面的选择题是对这段代码逐行的理解检验。</p>
      <pre class="lc-code"><code><span
        v-for="(line, idx) in solutionLines"
        :key="idx"
        class="lc-line"
      ><span class="lc-line-no">{{ String(idx + 1).padStart(2, ' ') }}</span><span class="lc-line-text" v-html="highlightPythonLine(line) || '&nbsp;'"></span></span></code></pre>
    </section>

    <div class="lc-questions-head">
      <h4>🧠 理解检验</h4>
      <span class="lc-questions-sub">每题都对应解法的某一行或某个决策</span>
    </div>

    <div class="lc-questions">
      <MicroChoice
        v-for="(q, idx) in visibleQuestions"
        :key="q.id"
        :id="q.id"
        :prompt="q.prompt"
        :code-context="q.codeContext"
        :highlight-line="q.highlightLine"
        :options="q.options"
        :answer="q.answer"
        :explain="q.explain"
        :sound-enabled="settings.soundEnabled"
        :reduce-motion="settings.reduceMotion"
        @answered="onAnswered($event, idx)"
      />
    </div>

    <section v-if="finished" class="lc-finish">
      <h3>🎉 关卡完成！</h3>
      <p>正确率 {{ Math.round(accuracy * 100) }}% · 获得 {{ xpThisLesson }} XP</p>
      <button class="lc-restart" @click="restart">再练一次（不计 XP）</button>
    </section>

    <section v-if="hearts.current === 0 && !finished && settings.heartsEnabled" class="lc-no-hearts">
      <h3>💔 心形耗尽</h3>
      <p>休息一下，每 30 分钟自动回 1 颗。也可以在设置里切换"无限模式"。</p>
      <button class="lc-restart" @click="toggleInfinite">切换无限模式</button>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { problems } from '../../../leetcode/_data/problems/index'
import { highlightPythonLine } from '../utils/highlight-python'
import {
  loadState, saveState, refillHearts, loseHeart,
  addXp, bumpStreak, recordAnswer, markProblemCompleted
} from '../../../leetcode/_data/progress'

const props = defineProps<{
  problemId: string
  mode?: 'learn' | 'review'
}>()

const problem = computed(() => problems[props.problemId])
const total = computed(() => problem.value?.microQuestions.length ?? 0)
const visibleQuestions = computed(() => problem.value?.microQuestions ?? [])
const solutionLines = computed(() => (problem.value?.solutionCode ?? '').split('\n'))

const state = ref(loadState())
const hearts = computed(() => state.value.hearts)
const streak = computed(() => state.value.streak)
const settings = computed(() => state.value.settings)

const answeredIds = ref<Set<string>>(new Set())
const correctIds = ref<Set<string>>(new Set())
const xpThisLesson = ref(0)

const answeredCount = computed(() => answeredIds.value.size)
const progressPct = computed(() => total.value === 0 ? 0 : (answeredCount.value / total.value) * 100)
const finished = computed(() => total.value > 0 && answeredCount.value === total.value)
const accuracy = computed(() => total.value === 0 ? 0 : correctIds.value.size / total.value)

const diffText = computed(() => ({
  easy: '简单', medium: '中等', hard: '困难',
})[problem.value?.difficulty ?? 'easy'])

const renderedStatement = computed(() => {
  const s = problem.value?.statement ?? ''
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br/>')
})

onMounted(() => {
  state.value = refillHearts(loadState())
  saveState(state.value)
})

function onAnswered(payload: { id: string; correct: boolean; attempts: number; revealed: boolean }) {
  if (answeredIds.value.has(payload.id)) return
  answeredIds.value.add(payload.id)

  let grade: 1 | 2 | 3 | 4 = 1
  if (payload.correct && payload.attempts === 1) grade = 4
  else if (payload.correct) grade = 3
  else if (payload.revealed) grade = 2
  else grade = 1

  state.value = recordAnswer(state.value, payload.id, grade)

  if (payload.correct) {
    correctIds.value.add(payload.id)
    const earn = payload.attempts === 1 ? 5 : 3
    xpThisLesson.value += earn
    state.value = addXp(state.value, earn)
  } else if (!payload.revealed) {
    state.value = loseHeart(state.value)
  }

  if (finished.value) {
    state.value = bumpStreak(state.value)
    state.value = markProblemCompleted(state.value, props.problemId, accuracy.value)
  }

  saveState(state.value)
}

function restart() {
  answeredIds.value = new Set()
  correctIds.value = new Set()
  xpThisLesson.value = 0
}

function toggleInfinite() {
  state.value.settings.heartsEnabled = !state.value.settings.heartsEnabled
  if (!state.value.settings.heartsEnabled) {
    state.value.hearts.current = state.value.hearts.max
  }
  saveState(state.value)
}
</script>

<style scoped>
.lc-lesson {
  max-width: 760px;
  margin: 24px auto;
}
.lc-lesson.missing {
  padding: 40px;
  text-align: center;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
}

.lc-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
  padding: 14px 18px;
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  border: 2px solid var(--vp-c-divider);
}
.lc-title-row { display: flex; align-items: center; gap: 10px; }
.lc-no {
  font-size: 12px;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-mute);
  padding: 2px 8px;
  border-radius: 4px;
}
.lc-title { font-weight: 700; font-size: 18px; }
.lc-diff {
  font-size: 12px;
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: 600;
}
.lc-diff.easy { background: rgba(34, 197, 94, 0.15); color: #16a34a; }
.lc-diff.medium { background: rgba(245, 158, 11, 0.15); color: #d97706; }
.lc-diff.hard { background: rgba(239, 68, 68, 0.15); color: #dc2626; }

.lc-hud { display: flex; gap: 14px; font-size: 14px; font-weight: 600; }
.hud-item { display: inline-flex; gap: 2px; align-items: center; }
.hud-heart { color: #ef4444; font-size: 16px; }
.hud-heart.off { color: var(--vp-c-divider); }

.lc-progress {
  position: relative;
  height: 12px;
  background: var(--vp-c-bg-mute);
  border-radius: 999px;
  margin: 14px 0;
  overflow: hidden;
}
.lc-progress-bar {
  height: 100%;
  background: linear-gradient(90deg, #22c55e, #16a34a);
  border-radius: 999px;
  transition: width 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.lc-progress-label {
  position: absolute;
  right: 10px;
  top: -3px;
  font-size: 11px;
  font-weight: 700;
  color: var(--vp-c-text-2);
}

.lc-statement {
  margin: 18px 0;
  padding: 16px 20px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
}
.lc-statement h4 { margin: 0 0 10px; font-size: 15px; }
.lc-statement-body {
  font-size: 14.5px;
  line-height: 1.75;
  color: var(--vp-c-text-1);
}
.lc-statement-body :deep(code) {
  background: var(--vp-c-bg-mute);
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 13px;
}

.lc-examples { margin-top: 14px; }
.lc-example {
  background: var(--vp-c-bg-mute);
  padding: 10px 14px;
  border-radius: 8px;
  margin: 8px 0;
  font-size: 13.5px;
}
.lc-ex-label {
  font-weight: 600;
  color: var(--vp-c-text-2);
  font-size: 12px;
  margin-bottom: 4px;
}
.lc-ex-row { margin: 3px 0; }
.lc-ex-tag {
  display: inline-block;
  width: 36px;
  font-size: 11px;
  color: var(--vp-c-text-2);
  font-weight: 600;
}
.lc-ex-note {
  margin-top: 4px;
  font-size: 12.5px;
  color: var(--vp-c-text-2);
  font-style: italic;
}

.lc-constraints { margin-top: 12px; font-size: 13px; }
.lc-cons-label {
  font-weight: 600;
  color: var(--vp-c-text-2);
  font-size: 12px;
}
.lc-constraints ul { margin: 4px 0 0; padding-left: 20px; color: var(--vp-c-text-2); }
.lc-constraints li { margin: 2px 0; }

.lc-intuition {
  padding: 12px 16px;
  background: rgba(59, 130, 246, 0.08);
  border-left: 4px solid #3b82f6;
  border-radius: 6px;
  font-size: 14px;
  margin: 14px 0;
  line-height: 1.7;
}

.lc-finish, .lc-no-hearts {
  margin-top: 24px;
  padding: 22px;
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  text-align: center;
}
.lc-finish h3 { margin-top: 0; }

.lc-solution {
  margin: 18px 0 22px;
  padding: 16px 18px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-left: 4px solid #22c55e;
  border-radius: 10px;
}
.lc-solution-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 4px;
}
.lc-solution-head h4 { margin: 0; font-size: 15px; }
.lc-complexity {
  font-size: 12px;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-mute);
  padding: 3px 10px;
  border-radius: 999px;
  font-weight: 600;
}
.lc-solution-hint {
  font-size: 12.5px;
  color: var(--vp-c-text-2);
  margin: 4px 0 10px;
}
.lc-code {
  background: #1e1e2e;
  color: #e4e4e8;
  border-radius: 8px;
  padding: 14px 14px;
  font-size: 13.5px;
  line-height: 1.65;
  margin: 0;
  overflow-x: auto;
  font-family: 'JetBrains Mono', 'Fira Code', Menlo, Consolas, monospace;
}
.lc-code code { background: transparent; color: inherit; padding: 0; font-size: inherit; }
.lc-line { display: flex; gap: 12px; align-items: flex-start; padding: 0 4px; white-space: pre; }
.lc-line-no {
  flex-shrink: 0;
  width: 22px;
  text-align: right;
  color: #6a6a7c;
  user-select: none;
  font-size: 11.5px;
  padding-top: 1px;
}
.lc-line-text { flex: 1; }
.lc-code :deep(.hl-kw)  { color: #c678dd; font-weight: 600; }
.lc-code :deep(.hl-bi)  { color: #61afef; }
.lc-code :deep(.hl-str) { color: #98c379; }
.lc-code :deep(.hl-num) { color: #d19a66; }
.lc-code :deep(.hl-com) { color: #7d8590; font-style: italic; }
.lc-code :deep(.hl-op)  { color: #56b6c2; }

.lc-questions-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 10px;
  margin: 18px 0 6px;
}
.lc-questions-head h4 { margin: 0; font-size: 15px; }
.lc-questions-sub { font-size: 12px; color: var(--vp-c-text-2); }

.lc-restart {
  background: var(--vp-c-brand-1);
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  font-size: 14px;
}
</style>
