<template>
  <div class="micro-choice" :class="{ correct: state === 'correct', wrong: state === 'wrong', revealed: state === 'revealed' }">
    <div class="mc-prompt" v-html="renderedPrompt"></div>

    <pre v-if="codeContext" class="mc-code"><code><span
      v-for="(line, idx) in codeLines"
      :key="idx"
      class="mc-line"
      :class="{ highlight: idx + 1 === highlightLine }"
    >{{ line }}<br/></span></code></pre>

    <div class="mc-options">
      <button
        v-for="opt in options"
        :key="opt.id"
        class="mc-option"
        :class="{
          selected: selected === opt.id,
          'is-answer': revealedAnswer && opt.id === answer,
          'is-wrong': state === 'wrong' && selected === opt.id,
        }"
        :disabled="state === 'correct' || state === 'revealed'"
        @click="pick(opt.id)"
      >
        <span class="mc-bullet">{{ opt.id.toUpperCase() }}</span>
        <span class="mc-text">{{ opt.text }}</span>
      </button>
    </div>

    <div v-if="state === 'wrong' || state === 'revealed' || state === 'correct'" class="mc-feedback">
      <div v-if="state === 'wrong'" class="fb-bar fb-wrong">
        <span>差一点 — 再想想？</span>
        <button class="fb-btn" @click="reveal">看答案</button>
      </div>
      <div v-if="state === 'correct'" class="fb-bar fb-correct">
        <span>+5 XP · 答对了！</span>
      </div>
      <div v-if="state === 'revealed' || state === 'correct' || state === 'wrong'" class="mc-explain">
        <strong>{{ state === 'correct' ? '为什么对：' : '解析：' }}</strong>
        <span v-html="renderedExplain"></span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

const props = defineProps<{
  id: string
  prompt: string
  codeContext?: string
  highlightLine?: number
  options: { id: string; text: string }[]
  answer: string
  explain: string
  soundEnabled?: boolean
  reduceMotion?: boolean
}>()

const emit = defineEmits<{
  (e: 'answered', payload: { id: string; correct: boolean; attempts: number; revealed: boolean }): void
}>()

type State = 'idle' | 'wrong' | 'correct' | 'revealed'
const state = ref<State>('idle')
const selected = ref<string | null>(null)
const attempts = ref(0)
const revealedAnswer = ref(false)

const codeLines = computed(() => (props.codeContext || '').split('\n'))

const renderedPrompt = computed(() => inlineCode(props.prompt))
const renderedExplain = computed(() => inlineCode(props.explain))

function inlineCode(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
}

function playTone(freqs: number[], duration = 0.18) {
  if (props.soundEnabled === false) return
  if (typeof window === 'undefined' || !('AudioContext' in window || 'webkitAudioContext' in window)) return
  try {
    const Ctx: any = (window as any).AudioContext || (window as any).webkitAudioContext
    const ctx = new Ctx()
    const now = ctx.currentTime
    freqs.forEach((f, i) => {
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()
      osc.type = 'sine'
      osc.frequency.value = f
      gain.gain.setValueAtTime(0.0001, now + i * 0.06)
      gain.gain.exponentialRampToValueAtTime(0.18, now + i * 0.06 + 0.02)
      gain.gain.exponentialRampToValueAtTime(0.0001, now + i * 0.06 + duration)
      osc.connect(gain).connect(ctx.destination)
      osc.start(now + i * 0.06)
      osc.stop(now + i * 0.06 + duration + 0.02)
    })
    setTimeout(() => ctx.close(), 800)
  } catch {}
}

function pick(id: string) {
  if (state.value === 'correct' || state.value === 'revealed') return
  selected.value = id
  attempts.value += 1
  if (id === props.answer) {
    state.value = 'correct'
    playTone([523.25, 659.25, 783.99])
    emit('answered', { id: props.id, correct: true, attempts: attempts.value, revealed: false })
  } else {
    state.value = 'wrong'
    playTone([220, 196])
    if (typeof navigator !== 'undefined' && 'vibrate' in navigator && !props.reduceMotion) {
      try { navigator.vibrate(120) } catch {}
    }
  }
}

function reveal() {
  state.value = 'revealed'
  revealedAnswer.value = true
  emit('answered', { id: props.id, correct: false, attempts: attempts.value, revealed: true })
}
</script>

<style scoped>
.micro-choice {
  border: 2px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 16px 18px;
  margin: 18px 0;
  background: var(--vp-c-bg-soft);
  transition: border-color 0.2s, transform 0.2s;
}
.micro-choice.correct { border-color: #22c55e; }
.micro-choice.wrong { border-color: #ef4444; animation: shake 0.22s; }
.micro-choice.revealed { border-color: #f59e0b; }

@keyframes shake {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-6px); }
  75% { transform: translateX(6px); }
}
@media (prefers-reduced-motion: reduce) {
  .micro-choice.wrong { animation: none; }
}

.mc-prompt {
  font-size: 15px;
  font-weight: 500;
  margin-bottom: 12px;
  color: var(--vp-c-text-1);
  line-height: 1.6;
}
.mc-prompt :deep(code) {
  background: var(--vp-c-bg-mute);
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 13px;
}

.mc-code {
  background: var(--vp-c-bg-mute);
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 13px;
  margin: 8px 0 12px;
  overflow-x: auto;
}
.mc-line { display: inline-block; width: 100%; padding: 0 4px; border-radius: 3px; }
.mc-line.highlight { background: rgba(245, 158, 11, 0.18); }

.mc-options {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.mc-option {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  text-align: left;
  border: 2px solid var(--vp-c-divider);
  border-radius: 10px;
  padding: 10px 14px;
  background: var(--vp-c-bg);
  cursor: pointer;
  font-size: 14px;
  line-height: 1.5;
  transition: border-color 0.15s, background 0.15s, transform 0.1s;
  color: var(--vp-c-text-1);
}
.mc-option:hover:not(:disabled) {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-1px);
}
.mc-option:disabled { cursor: not-allowed; opacity: 0.85; }
.mc-option.selected.is-wrong { border-color: #ef4444; background: rgba(239, 68, 68, 0.08); }
.mc-option.is-answer { border-color: #22c55e; background: rgba(34, 197, 94, 0.1); }

.mc-bullet {
  flex-shrink: 0;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: var(--vp-c-bg-mute);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 700;
  color: var(--vp-c-text-2);
}
.mc-option.is-answer .mc-bullet { background: #22c55e; color: white; }
.mc-option.selected.is-wrong .mc-bullet { background: #ef4444; color: white; }

.mc-feedback { margin-top: 12px; }
.fb-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 8px;
}
.fb-correct { background: rgba(34, 197, 94, 0.12); color: #16a34a; }
.fb-wrong { background: rgba(239, 68, 68, 0.12); color: #dc2626; }
.fb-btn {
  background: white;
  border: 1px solid currentColor;
  border-radius: 6px;
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  color: inherit;
}
.mc-explain {
  font-size: 13.5px;
  line-height: 1.7;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-mute);
  padding: 10px 14px;
  border-radius: 8px;
}
.mc-explain :deep(code) {
  background: var(--vp-c-bg);
  padding: 1px 5px;
  border-radius: 3px;
}
</style>
