<template>
  <div class="code-masker">
    <div class="masker-toolbar">
      <span class="masker-title">{{ title || 'MLM 代码训练' }}</span>
      <div class="toolbar-right">
        <label class="ratio-label">
          挖空率
          <select v-model.number="currentRatio" class="ratio-select" @change="regenerate">
            <option :value="0.1">10%</option>
            <option :value="0.15">15%</option>
            <option :value="0.2">20%</option>
            <option :value="0.3">30%</option>
            <option :value="0.5">50%</option>
          </select>
        </label>
        <button class="masker-btn refresh-btn" @click="regenerate" title="刷新题目">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>
            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
          </svg>
          刷新
        </button>
        <button class="masker-btn check-btn" @click="checkAll" :disabled="!hasInput">
          验证
        </button>
        <button v-if="showAnswers && !revealed" class="masker-btn reveal-btn" @click="revealAll">
          显示答案
        </button>
        <button v-if="revealed" class="masker-btn retry-btn" @click="retryWithSameMask">
          重新作答
        </button>
      </div>
    </div>

    <div class="masker-stats">
      <span>共 {{ maskableTokens.length }} 个可挖空位 | 已挖 {{ maskedIndices.size }} 个</span>
      <span v-if="checked" :class="scoreClass">
        得分: {{ correctCount }}/{{ maskedIndices.size }}
        ({{ Math.round(correctCount / maskedIndices.size * 100) }}%)
      </span>
    </div>

    <div class="code-container">
      <pre class="code-pre"><code><template v-for="(token, i) in renderTokens" :key="i"><span
        v-if="token.type === 'text'"
        class="code-text">{{ token.value }}</span><span
        v-else-if="token.type === 'mask'"
        class="mask-slot"
        :class="{ correct: token.state === 'correct', wrong: token.state === 'wrong', revealed: token.state === 'revealed' }"
      ><input
          v-if="token.state !== 'revealed'"
          :ref="el => { if (el) inputRefs[token.idx] = el }"
          v-model="userAnswers[token.idx]"
          class="mask-input"
          :class="{ correct: token.state === 'correct', wrong: token.state === 'wrong' }"
          :style="{ width: Math.max(token.answer.length * 0.65, 2) + 'em' }"
          :placeholder="'_'.repeat(Math.min(token.answer.length, 6))"
          spellcheck="false"
          autocomplete="off"
          @keydown.enter="focusNext(token.idx)"
          @keydown.tab.prevent="focusNext(token.idx)"
        /><span v-else class="revealed-text">{{ token.answer }}</span></span></template></code></pre>
    </div>

    <div v-if="checked && correctCount === maskedIndices.size" class="perfect-banner">
      全部正确! 点击「刷新」挑战新的挖空组合
    </div>

    <!-- Hidden slot container for extracting code text -->
    <div ref="slotRef" style="display:none"><slot /></div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, nextTick } from 'vue'

const props = defineProps({
  code: { type: String, default: '' },
  codeBase64: { type: String, default: '' },
  lang: { type: String, default: 'python' },
  title: { type: String, default: '' },
  maskRatio: { type: Number, default: 0.15 },
  showAnswers: { type: Boolean, default: true },
})

const slots = defineSlots()
const slotRef = ref(null)

const currentRatio = ref(props.maskRatio)
const maskedIndices = ref(new Set())
const userAnswers = ref({})
const checked = ref(false)
const revealed = ref(false)
const inputRefs = ref({})
const seed = ref(0)

// Extract code: prefer base64-encoded prop (preserves whitespace perfectly),
// fall back to raw code prop, then slot innerHTML parsing
const sourceCode = computed(() => {
  if (props.codeBase64) {
    try { return atob(props.codeBase64).trim() } catch { /* fall through */ }
  }
  if (props.code) return props.code.trim()
  if (slotRef.value) {
    const html = slotRef.value.innerHTML
    let text = html
      .replace(/<br\s*\/?>/gi, '\n')
      .replace(/<\/p>\s*<p[^>]*>/gi, '\n')
      .replace(/<\/?p[^>]*>/gi, '')
      .replace(/<code[^>]*>([\s\S]*?)<\/code>/gi, '$1')
      .replace(/<[^>]+>/g, '')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&amp;/g, '&')
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'")
    return text.trim()
  }
  return ''
})

// Parse code into tokens: we split on meaningful boundaries
// keeping whitespace/newlines as non-maskable tokens
const allTokens = computed(() => {
  const code = sourceCode.value
  const tokens = []
  // Split into lines first, then tokenize each line
  const lines = code.split('\n')
  for (let li = 0; li < lines.length; li++) {
    const line = lines[li]
    if (li > 0) tokens.push({ value: '\n', maskable: false })

    // Tokenize line: split by spaces, operators, brackets, keeping them
    const parts = line.match(/(\s+|[a-zA-Z_]\w*|[0-9]+\.?[0-9]*|"[^"]*"|'[^']*'|[^\s])/g)
    if (!parts) continue

    for (const part of parts) {
      const isWhitespace = /^\s+$/.test(part)
      const isOperator = /^[=+\-*/<>!&|^~%@:,;.()\[\]{}]$/.test(part)
      const isKeyword = /^(def|class|if|else|elif|for|while|return|import|from|as|in|not|and|or|is|None|True|False|self|with|try|except|finally|raise|yield|lambda|pass|break|continue|global|nonlocal|assert|del)$/.test(part)
      const isComment = part.startsWith('#')

      // Maskable: identifiers, numbers, strings, keywords (not pure whitespace/single char operators)
      const maskable = !isWhitespace && !isComment && part.length >= 2
      tokens.push({ value: part, maskable })
    }
  }
  return tokens
})

// Indices of maskable tokens
const maskableTokens = computed(() => {
  return allTokens.value
    .map((t, i) => ({ ...t, index: i }))
    .filter(t => t.maskable)
})

// Generate masked indices based on ratio
function regenerate() {
  seed.value++
  checked.value = false
  revealed.value = false
  userAnswers.value = {}
  inputRefs.value = {}

  const candidates = maskableTokens.value.map(t => t.index)
  const count = Math.max(1, Math.round(candidates.length * currentRatio.value))

  // Shuffle and pick
  const shuffled = [...candidates]
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(pseudoRandom(seed.value, i) * (i + 1))
    ;[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
  }

  maskedIndices.value = new Set(shuffled.slice(0, count))

  nextTick(() => {
    // Focus first input
    const firstIdx = shuffled[0]
    if (inputRefs.value[firstIdx]) {
      inputRefs.value[firstIdx].focus()
    }
  })
}

// Simple deterministic random for reproducibility within a session
function pseudoRandom(s, i) {
  const x = Math.sin(s * 9301 + i * 49297 + 233280) * 49297
  return x - Math.floor(x)
}

// Build render tokens
const renderTokens = computed(() => {
  let maskIdx = 0
  return allTokens.value.map((token, i) => {
    if (maskedIndices.value.has(i)) {
      const idx = maskIdx++
      const answer = token.value
      let state = null
      if (checked.value) {
        const userAns = (userAnswers.value[idx] || '').trim()
        state = userAns === answer ? 'correct' : 'wrong'
      }
      if (userAnswers.value[idx] === '__REVEALED__') {
        state = 'revealed'
      }
      return { type: 'mask', idx, answer, state }
    }
    return { type: 'text', value: token.value }
  })
})

const hasInput = computed(() => {
  return Object.values(userAnswers.value).some(v => v && v.trim() && v !== '__REVEALED__')
})

const correctCount = computed(() => {
  if (!checked.value) return 0
  let count = 0
  let maskIdx = 0
  for (let i = 0; i < allTokens.value.length; i++) {
    if (maskedIndices.value.has(i)) {
      const userAns = (userAnswers.value[maskIdx] || '').trim()
      if (userAns === allTokens.value[i].value) count++
      maskIdx++
    }
  }
  return count
})

const scoreClass = computed(() => {
  if (!checked.value) return ''
  const ratio = correctCount.value / maskedIndices.value.size
  if (ratio === 1) return 'score-perfect'
  if (ratio >= 0.8) return 'score-good'
  if (ratio >= 0.5) return 'score-ok'
  return 'score-low'
})

function checkAll() {
  checked.value = true
}

function revealAll() {
  revealed.value = true
  let maskIdx = 0
  for (let i = 0; i < allTokens.value.length; i++) {
    if (maskedIndices.value.has(i)) {
      userAnswers.value[maskIdx] = '__REVEALED__'
      maskIdx++
    }
  }
  checked.value = false
}

function retryWithSameMask() {
  revealed.value = false
  checked.value = false
  userAnswers.value = {}
  inputRefs.value = {}
  nextTick(() => {
    if (inputRefs.value[0]) inputRefs.value[0].focus()
  })
}

function focusNext(currentIdx) {
  const nextIdx = currentIdx + 1
  nextTick(() => {
    if (inputRefs.value[nextIdx]) {
      inputRefs.value[nextIdx].focus()
    }
  })
}

onMounted(() => {
  // Wait a tick for slot content to render
  nextTick(() => regenerate())
})
</script>

<style scoped>
.code-masker {
  margin: 16px 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  overflow: hidden;
  background: var(--vp-c-bg-alt);
}

.masker-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: var(--vp-c-bg-soft);
  border-bottom: 1px solid var(--vp-c-divider);
  flex-wrap: wrap;
  gap: 8px;
}

.masker-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.toolbar-right {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.ratio-label {
  font-size: 12px;
  color: var(--vp-c-text-2);
  display: flex;
  align-items: center;
  gap: 4px;
}

.ratio-select {
  padding: 2px 4px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 12px;
}

.masker-btn {
  padding: 4px 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 12px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 4px;
  transition: all 0.15s;
}

.masker-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.masker-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.check-btn:hover { border-color: var(--vp-c-brand-1); }
.reveal-btn:hover { border-color: var(--vp-c-warning-1, #e2a308); color: var(--vp-c-warning-1, #e2a308); }
.retry-btn { border-color: var(--vp-c-green-1, #22c55e); color: var(--vp-c-green-1, #22c55e); }
.retry-btn:hover { background: rgba(34, 197, 94, 0.1); }

.masker-stats {
  padding: 6px 12px;
  font-size: 12px;
  color: var(--vp-c-text-3);
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid var(--vp-c-divider);
}

.score-perfect { color: var(--vp-c-green-1, #22c55e); font-weight: 600; }
.score-good { color: var(--vp-c-brand-1); font-weight: 600; }
.score-ok { color: var(--vp-c-warning-1, #e2a308); }
.score-low { color: var(--vp-c-danger-1, #ef4444); }

.code-container {
  overflow-x: auto;
  padding: 12px;
}

.code-pre {
  margin: 0;
  padding: 0;
  background: transparent;
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  line-height: 1.6;
}

.code-pre code {
  background: transparent;
  padding: 0;
}

.code-text {
  white-space: pre;
}

.mask-slot {
  display: inline;
}

.mask-input {
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  padding: 1px 4px;
  border: 1px dashed var(--vp-c-brand-1);
  border-radius: 3px;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-text-1);
  outline: none;
  min-width: 2em;
  line-height: 1.4;
}

.mask-input:focus {
  border-style: solid;
  box-shadow: 0 0 0 2px var(--vp-c-brand-soft);
}

.mask-input.correct {
  border-color: var(--vp-c-green-1, #22c55e);
  background: rgba(34, 197, 94, 0.1);
  color: var(--vp-c-green-1, #22c55e);
}

.mask-input.wrong {
  border-color: var(--vp-c-danger-1, #ef4444);
  background: rgba(239, 68, 68, 0.1);
  color: var(--vp-c-danger-1, #ef4444);
}

.revealed-text {
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  padding: 1px 4px;
  background: rgba(226, 163, 8, 0.15);
  border-radius: 3px;
  color: var(--vp-c-warning-1, #e2a308);
  border: 1px solid rgba(226, 163, 8, 0.3);
}

.perfect-banner {
  text-align: center;
  padding: 10px;
  background: rgba(34, 197, 94, 0.1);
  color: var(--vp-c-green-1, #22c55e);
  font-size: 14px;
  font-weight: 600;
  border-top: 1px solid var(--vp-c-divider);
}

@media (max-width: 640px) {
  .masker-toolbar { flex-direction: column; align-items: flex-start; }
  .toolbar-right { width: 100%; justify-content: flex-start; }
  .mask-input { font-size: 12px; }
}
</style>
