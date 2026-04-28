<template>
  <div class="ai-chat">
    <!-- Floating trigger button -->
    <button class="ai-chat-btn" :class="{ active: panelOpen }" @click="panelOpen = !panelOpen" title="AI 助教">
      <svg v-if="!panelOpen" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2a8 8 0 0 1 8 8c0 3.3-2 6.2-5 7.5V20a1 1 0 0 1-1 1h-4a1 1 0 0 1-1-1v-2.5C6 16.2 4 13.3 4 10a8 8 0 0 1 8-8z"/>
        <line x1="10" y1="22" x2="14" y2="22"/>
      </svg>
      <svg v-else width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
      </svg>
    </button>

    <!-- Chat panel -->
    <Transition name="panel-slide">
      <div v-if="panelOpen" class="ai-chat-panel">
        <!-- Header -->
        <div class="panel-header">
          <span class="panel-title">AI 助教</span>
          <button v-if="apiKey" class="settings-btn" @click="showSettings = !showSettings" title="设置">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.32 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
          </button>
        </div>

        <!-- Settings overlay -->
        <div v-if="showSettings && apiKey" class="settings-overlay">
          <p class="settings-label">当前 API Key</p>
          <code class="key-preview">{{ apiKey.slice(0, 8) }}...{{ apiKey.slice(-4) }}</code>
          <button class="btn btn-danger" @click="clearKey">删除 Key</button>
          <button class="btn btn-secondary" @click="showSettings = false">关闭</button>
        </div>

        <!-- Key setup view -->
        <div v-else-if="!apiKey" class="setup-view">
          <div class="setup-icon">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--vp-c-text-2)" stroke-width="1.5"><path d="M12 2a8 8 0 0 1 8 8c0 3.3-2 6.2-5 7.5V20a1 1 0 0 1-1 1h-4a1 1 0 0 1-1-1v-2.5C6 16.2 4 13.3 4 10a8 8 0 0 1 8-8z"/><line x1="10" y1="22" x2="14" y2="22"/></svg>
          </div>
          <p class="setup-title">AI 助教</p>
          <p class="setup-desc">输入 OpenAI API Key 即可对当前页面内容提问，AI 会智能检索相关章节来回答。</p>
          <input
            v-model="keyInput"
            type="password"
            class="key-input"
            placeholder="sk-..."
            @keydown.enter="saveKey"
          />
          <button class="btn btn-primary" :disabled="!keyInput.trim()" @click="saveKey">保存并开始</button>
          <p class="setup-hint">Key 仅存储在浏览器本地，不会上传到任何服务器。</p>
        </div>

        <!-- Chat view -->
        <template v-else>
          <div class="messages" ref="messagesEl">
            <div v-if="messages.length === 0" class="empty-hint">
              对当前页面内容有疑问？直接提问吧
            </div>
            <div v-for="(msg, i) in messages" :key="i" class="message" :class="msg.role">
              <div class="msg-content" v-html="msg.content"></div>
            </div>
            <div v-if="loading" class="message assistant">
              <div class="msg-content loading-dots">
                <span v-if="stage === 1">检索相关内容</span>
                <span v-else>思考中</span>
                <span class="dots">...</span>
              </div>
            </div>
          </div>
          <div class="input-area">
            <textarea
              ref="inputEl"
              v-model="userInput"
              :disabled="loading"
              placeholder="输入问题..."
              rows="1"
              @keydown.enter.exact.prevent="sendMessage"
              @input="autoResize"
            />
            <button class="send-btn" :disabled="!userInput.trim() || loading" @click="sendMessage">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M2 21l21-9L2 3v7l15 2-15 2z"/></svg>
            </button>
          </div>
        </template>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref, watch, nextTick, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vitepress'

const route = useRoute()
const panelOpen = ref(false)
const showSettings = ref(false)
const apiKey = ref('')
const keyInput = ref('')
const userInput = ref('')
const messages = ref([])
const loading = ref(false)
const stage = ref(0)
const messagesEl = ref(null)
const inputEl = ref(null)

let contentIndex = null
let abortController = null

const STORAGE_KEY = 'llm101-ai-key'
const INDEX_URL = import.meta.env.BASE_URL + 'content-index.json'
const API_URL = 'https://api.openai.com/v1/chat/completions'
const MODEL = 'gpt-4o-mini'
const MAX_SHORT_PAGE = 3000

// ── Key management ──

onMounted(() => {
  apiKey.value = localStorage.getItem(STORAGE_KEY) || ''
})

function saveKey() {
  const k = keyInput.value.trim()
  if (!k) return
  apiKey.value = k
  localStorage.setItem(STORAGE_KEY, k)
  keyInput.value = ''
}

function clearKey() {
  apiKey.value = ''
  localStorage.removeItem(STORAGE_KEY)
  showSettings.value = false
  messages.value = []
}

// ── Clear chat on page change ──

watch(() => route.path, () => {
  messages.value = []
  contentIndex = null // reload for fresh base URL resolution
})

// ── Load content index ──

async function loadIndex() {
  if (contentIndex) return contentIndex
  try {
    const res = await fetch(INDEX_URL)
    contentIndex = await res.json()
  } catch {
    contentIndex = {}
  }
  return contentIndex
}

// ── DOM section extraction ──

function getPageSections() {
  const doc = document.querySelector('.vp-doc')
  if (!doc) return { fullText: '', sections: [] }

  const fullText = doc.innerText
  const headings = doc.querySelectorAll('h2, h3')
  const sections = []

  headings.forEach((el, i) => {
    const level = el.tagName === 'H2' ? 2 : 3
    const heading = el.innerText.replace(/\u200B/g, '').trim()
    // Get text until next heading
    let text = ''
    let sibling = el.nextElementSibling
    const nextHeading = headings[i + 1]
    while (sibling && sibling !== nextHeading) {
      text += sibling.innerText + '\n'
      sibling = sibling.nextElementSibling
    }
    sections.push({ heading, level, text: text.trim() })
  })

  return { fullText, sections }
}

// ── Detect current module from route ──

function getCurrentPageKey() {
  // route.path is like /LLM-101-CN/architecture/transformer.html
  const p = route.path.replace(/^\/LLM-101-CN\//, '').replace(/\.html$/, '.md').replace(/\/$/, '/index.md')
  return p
}

function getCurrentModule() {
  const key = getCurrentPageKey()
  return key.split('/')[0] || ''
}

// ── Build retrieval prompt from index ──

function buildRetrievalContext() {
  const idx = contentIndex || {}
  const mod = getCurrentModule()
  const lines = []

  for (const [filePath, info] of Object.entries(idx)) {
    if (!filePath.startsWith(mod + '/')) continue
    lines.push(`\n## ${filePath} - ${info.title}`)
    if (info.topics.length) lines.push(`topics: ${info.topics.join(', ')}`)
    for (const s of info.sections) {
      const indent = s.level === 3 ? '  ' : ''
      lines.push(`${indent}- ${s.heading}`)
    }
  }
  return lines.join('\n')
}

// ── API call helper ──

async function callLLM(systemPrompt, userPrompt, stream = false) {
  const body = {
    model: MODEL,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ],
    temperature: 0.3,
    stream,
  }

  abortController = new AbortController()
  const res = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey.value}`,
    },
    body: JSON.stringify(body),
    signal: abortController.signal,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.error?.message || `API error ${res.status}`)
  }

  return res
}

// ── Stage 1: LLM-as-retriever ──

async function retrieveSections(question, moduleIndex) {
  const sysPrompt = `你是一个检索助手。根据用户问题，从以下教程目录中选出最相关的 2-3 个章节。
只返回章节标题，每行一个，不要其他内容。如果都不相关，返回"NONE"。

当前用户正在阅读: ${getCurrentPageKey()}

教程目录:
${moduleIndex}`

  const res = await callLLM(sysPrompt, question, false)
  const data = await res.json()
  const text = data.choices?.[0]?.message?.content || ''
  if (text.includes('NONE')) return []
  return text.split('\n').map(l => l.replace(/^[-*]\s*/, '').trim()).filter(Boolean)
}

// ── Stage 2: Answer with context (streaming) ──

async function answerWithContext(question, context, history) {
  const sysPrompt = `你是 LLM-101-CN 教程的 AI 助教。基于以下教程内容回答用户问题。
回答要简洁准确，使用中文。如果内容不足以回答，可以用你的知识补充，但请注明。
可以使用 Markdown 格式。

---
${context}
---`

  const msgs = [{ role: 'system', content: sysPrompt }]
  // Include recent history (last 4 turns max)
  const recent = history.slice(-4)
  for (const m of recent) {
    msgs.push({ role: m.role, content: m.raw || m.content })
  }
  msgs.push({ role: 'user', content: question })

  abortController = new AbortController()
  const res = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey.value}`,
    },
    body: JSON.stringify({
      model: MODEL,
      messages: msgs,
      temperature: 0.3,
      stream: true,
    }),
    signal: abortController.signal,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.error?.message || `API error ${res.status}`)
  }

  return res
}

// ── Stream reader ──

async function readStream(res, onChunk) {
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const data = line.slice(6)
      if (data === '[DONE]') return
      try {
        const json = JSON.parse(data)
        const delta = json.choices?.[0]?.delta?.content
        if (delta) onChunk(delta)
      } catch {}
    }
  }
}

// ── Simple markdown to HTML (minimal) ──

function mdToHtml(text) {
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>')
}

// ── Main send handler ──

async function sendMessage() {
  const question = userInput.value.trim()
  if (!question || loading.value) return

  userInput.value = ''
  if (inputEl.value) inputEl.value.style.height = 'auto'
  messages.value.push({ role: 'user', content: mdToHtml(question), raw: question })
  scrollToBottom()

  loading.value = true

  try {
    await loadIndex()
    const { fullText, sections } = getPageSections()
    let context = ''

    if (fullText.length < MAX_SHORT_PAGE) {
      // Short page: skip retrieval, use full text
      context = fullText
    } else {
      // Stage 1: retrieve relevant sections
      stage.value = 1
      const moduleIndex = buildRetrievalContext()
      const relevantHeadings = await retrieveSections(question, moduleIndex)

      // Extract matching sections from DOM
      const matched = []
      for (const heading of relevantHeadings) {
        const section = sections.find(s =>
          s.heading.includes(heading) || heading.includes(s.heading)
        )
        if (section) {
          matched.push(`## ${section.heading}\n${section.text}`)
        }
      }
      context = matched.length > 0
        ? matched.join('\n\n')
        : sections.slice(0, 3).map(s => `## ${s.heading}\n${s.text}`).join('\n\n')
    }

    // Stage 2: answer with streaming
    stage.value = 2
    const res = await answerWithContext(question, context, messages.value.slice(0, -1))

    // Add assistant message placeholder
    const assistantMsg = { role: 'assistant', content: '', raw: '' }
    messages.value.push(assistantMsg)

    await readStream(res, (chunk) => {
      assistantMsg.raw += chunk
      assistantMsg.content = mdToHtml(assistantMsg.raw)
      scrollToBottom()
    })

  } catch (err) {
    if (err.name === 'AbortError') return
    const errMsg = err.message.includes('Incorrect API key')
      ? 'API Key 无效，请点击右上角设置重新输入'
      : `出错了: ${err.message}`
    messages.value.push({ role: 'assistant', content: `<span class="error">${errMsg}</span>` })
  } finally {
    loading.value = false
    stage.value = 0
    scrollToBottom()
  }
}

// ── UI helpers ──

function scrollToBottom() {
  nextTick(() => {
    if (messagesEl.value) {
      messagesEl.value.scrollTop = messagesEl.value.scrollHeight
    }
  })
}

function autoResize(e) {
  const el = e.target
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 100) + 'px'
}

// ── Keyboard shortcut: Escape to close ──

function onKeydown(e) {
  if (e.key === 'Escape' && panelOpen.value) {
    panelOpen.value = false
  }
}

onMounted(() => document.addEventListener('keydown', onKeydown))
onUnmounted(() => {
  document.removeEventListener('keydown', onKeydown)
  abortController?.abort()
})
</script>

<style scoped>
.ai-chat {
  position: fixed;
  bottom: 24px;
  right: 24px;
  z-index: 100;
  font-family: var(--vp-font-family-base);
}

.ai-chat-btn {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-elv);
  color: var(--vp-c-text-1);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  transition: all 0.2s;
}
.ai-chat-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}
.ai-chat-btn.active {
  background: var(--vp-c-brand-1);
  color: #fff;
  border-color: var(--vp-c-brand-1);
}

/* Panel */
.ai-chat-panel {
  position: absolute;
  bottom: 56px;
  right: 0;
  width: 360px;
  max-height: 480px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.12);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--vp-c-divider);
  flex-shrink: 0;
}
.panel-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--vp-c-text-1);
}
.settings-btn {
  background: none;
  border: none;
  color: var(--vp-c-text-2);
  cursor: pointer;
  padding: 4px;
  border-radius: 4px;
}
.settings-btn:hover { color: var(--vp-c-text-1); background: var(--vp-c-bg-soft); }

/* Settings overlay */
.settings-overlay {
  padding: 20px 16px;
  text-align: center;
}
.settings-label {
  font-size: 12px;
  color: var(--vp-c-text-2);
  margin-bottom: 8px;
}
.key-preview {
  display: block;
  font-size: 12px;
  color: var(--vp-c-text-3);
  margin-bottom: 16px;
}

/* Setup view */
.setup-view {
  padding: 32px 20px 20px;
  text-align: center;
  flex: 1;
}
.setup-icon { margin-bottom: 12px; }
.setup-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: 8px;
}
.setup-desc {
  font-size: 13px;
  color: var(--vp-c-text-2);
  line-height: 1.5;
  margin-bottom: 16px;
}
.key-input {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  font-size: 13px;
  margin-bottom: 12px;
  outline: none;
}
.key-input:focus { border-color: var(--vp-c-brand-1); }
.setup-hint {
  font-size: 11px;
  color: var(--vp-c-text-3);
  margin-top: 12px;
}

/* Buttons */
.btn {
  display: inline-block;
  padding: 6px 16px;
  border: none;
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
  margin: 4px;
}
.btn-primary {
  background: var(--vp-c-brand-1);
  color: #fff;
  width: 100%;
}
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-danger { background: var(--vp-c-danger-1, #e53e3e); color: #fff; }
.btn-secondary { background: var(--vp-c-bg-soft); color: var(--vp-c-text-1); }

/* Messages */
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 12px 16px;
  min-height: 200px;
  max-height: 340px;
}
.empty-hint {
  text-align: center;
  color: var(--vp-c-text-3);
  font-size: 13px;
  margin-top: 60px;
}
.message {
  margin-bottom: 12px;
}
.message.user .msg-content {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-text-1);
  padding: 8px 12px;
  border-radius: 12px 12px 4px 12px;
  font-size: 13px;
  line-height: 1.5;
  max-width: 85%;
  margin-left: auto;
  word-break: break-word;
}
.message.assistant .msg-content {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  padding: 8px 12px;
  border-radius: 12px 12px 12px 4px;
  font-size: 13px;
  line-height: 1.6;
  max-width: 90%;
  word-break: break-word;
}
.message.assistant .msg-content :deep(pre) {
  background: var(--vp-c-bg-alt);
  padding: 8px;
  border-radius: 4px;
  overflow-x: auto;
  margin: 4px 0;
}
.message.assistant .msg-content :deep(code) {
  font-size: 12px;
  font-family: var(--vp-font-family-mono);
}
.message.assistant .msg-content :deep(.error) {
  color: var(--vp-c-danger-1, #e53e3e);
}

.loading-dots {
  color: var(--vp-c-text-2);
  font-size: 13px;
}
.loading-dots .dots {
  animation: blink 1.2s infinite;
}
@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

/* Input area */
.input-area {
  display: flex;
  align-items: flex-end;
  gap: 8px;
  padding: 12px;
  border-top: 1px solid var(--vp-c-divider);
  flex-shrink: 0;
}
.input-area textarea {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  font-size: 13px;
  font-family: var(--vp-font-family-base);
  resize: none;
  outline: none;
  line-height: 1.4;
  max-height: 100px;
}
.input-area textarea:focus { border-color: var(--vp-c-brand-1); }
.send-btn {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  border: none;
  background: var(--vp-c-brand-1);
  color: #fff;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

/* Transitions */
.panel-slide-enter-active, .panel-slide-leave-active {
  transition: all 0.2s ease;
}
.panel-slide-enter-from, .panel-slide-leave-to {
  opacity: 0;
  transform: translateY(8px) scale(0.95);
}

/* Mobile responsive */
@media (max-width: 640px) {
  .ai-chat { bottom: 16px; right: 16px; }
  .ai-chat-panel {
    width: calc(100vw - 32px);
    max-height: 60vh;
    right: 0;
  }
}
</style>
