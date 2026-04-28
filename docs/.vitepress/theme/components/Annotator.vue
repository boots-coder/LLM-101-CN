<template>
  <div class="annotator">
    <!-- Floating toolbar on text selection -->
    <Transition name="toolbar-fade">
      <div
        v-if="showToolbar"
        class="annotator-toolbar"
        :style="{ top: toolbarPos.top + 'px', left: toolbarPos.left + 'px' }"
        @mousedown.prevent
      >
        <button
          v-for="c in colors"
          :key="c.value"
          class="color-btn"
          :style="{ background: c.value }"
          :title="c.label"
          @click="selectColor(c.value)"
        />
      </div>
    </Transition>

    <!-- Note input popup -->
    <Transition name="toolbar-fade">
      <div
        v-if="showNoteInput"
        class="annotator-note-popup"
        :style="{ top: notePopupPos.top + 'px', left: notePopupPos.left + 'px' }"
        @mousedown.prevent
      >
        <div class="note-popup-header">添加笔记</div>
        <textarea
          ref="noteTextarea"
          v-model="noteText"
          placeholder="写下你的想法..."
          rows="3"
          @keydown.enter.ctrl="saveAnnotation"
          @keydown.escape="cancelNote"
        />
        <div class="note-popup-actions">
          <button class="note-btn cancel" @click="cancelNote">取消</button>
          <button class="note-btn save" @click="saveAnnotation">
            保存 <span class="shortcut">Ctrl+Enter</span>
          </button>
        </div>
      </div>
    </Transition>

    <!-- Annotation count badge -->
    <Transition name="toolbar-fade">
      <div
        v-if="currentPageAnnotations.length > 0"
        class="annotator-badge"
        :title="'本页有 ' + currentPageAnnotations.length + ' 条笔记'"
        @click="togglePanel"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
        </svg>
        <span class="badge-count">{{ currentPageAnnotations.length }}</span>
      </div>
    </Transition>

    <!-- Current page annotations panel -->
    <Transition name="panel-slide">
      <div v-if="showPanel" class="annotator-panel">
        <div class="panel-header">
          <span>本页笔记 ({{ currentPageAnnotations.length }})</span>
          <button class="panel-close" @click="showPanel = false">&times;</button>
        </div>
        <div class="panel-body">
          <div
            v-for="(ann, idx) in currentPageAnnotations"
            :key="ann.timestamp"
            class="panel-item"
          >
            <div class="panel-item-text" :style="{ borderLeftColor: ann.color }">
              <span class="highlighted-text">"{{ truncate(ann.text, 60) }}"</span>
              <span v-if="ann.note" class="note-content">{{ ann.note }}</span>
              <span class="note-time">{{ formatTime(ann.timestamp) }}</span>
            </div>
            <button class="panel-item-delete" title="删除" @click="deleteAnnotation(idx)">
              &times;
            </button>
          </div>
        </div>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { useRoute } from 'vitepress'

const STORAGE_KEY = 'llm101-annotations'
const route = useRoute()

const allAnnotations = ref([])
const showToolbar = ref(false)
const showNoteInput = ref(false)
const showPanel = ref(false)
const toolbarPos = ref({ top: 0, left: 0 })
const notePopupPos = ref({ top: 0, left: 0 })
const noteText = ref('')
const noteTextarea = ref(null)

const selectedText = ref('')
const selectedColor = ref('')
const selectionRange = ref(null)

const colors = [
  { value: '#fef08a', label: '黄色' },
  { value: '#bbf7d0', label: '绿色' },
  { value: '#bfdbfe', label: '蓝色' },
  { value: '#fbcfe8', label: '粉色' },
]

const currentPage = computed(() => route.path)
const currentPageAnnotations = computed(() =>
  allAnnotations.value.filter((a) => a.page === currentPage.value)
)

function loadAnnotations() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) allAnnotations.value = JSON.parse(raw)
  } catch {
    allAnnotations.value = []
  }
}

function saveToStorage() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(allAnnotations.value))
}

function getContentContainer() {
  return document.querySelector('.vp-doc') || document.querySelector('main') || document.body
}

function handleMouseUp(e) {
  // Ignore clicks inside our own UI
  if (e.target.closest('.annotator')) return

  const selection = window.getSelection()
  const text = selection?.toString().trim()

  if (!text || text.length < 2) {
    // Small delay so clicking a color button doesn't dismiss toolbar
    setTimeout(() => {
      if (!showNoteInput.value) showToolbar.value = false
    }, 200)
    return
  }

  // Only allow selection inside content area
  const container = getContentContainer()
  if (!container) return
  const range = selection.getRangeAt(0)
  if (!container.contains(range.commonAncestorContainer)) return

  selectedText.value = text
  selectionRange.value = range.cloneRange()

  const rect = range.getBoundingClientRect()
  toolbarPos.value = {
    top: rect.top + window.scrollY - 44,
    left: rect.left + window.scrollX + rect.width / 2 - 72,
  }
  showToolbar.value = true
}

function selectColor(color) {
  selectedColor.value = color
  showToolbar.value = false

  const rect = selectionRange.value?.getBoundingClientRect()
  if (rect) {
    notePopupPos.value = {
      top: rect.bottom + window.scrollY + 8,
      left: Math.max(10, rect.left + window.scrollX - 40),
    }
  }
  showNoteInput.value = true
  nextTick(() => noteTextarea.value?.focus())
}

function saveAnnotation() {
  const ann = {
    page: currentPage.value,
    text: selectedText.value,
    note: noteText.value.trim(),
    color: selectedColor.value,
    timestamp: Date.now(),
    textContext: getTextContext(selectedText.value),
  }
  allAnnotations.value.push(ann)
  saveToStorage()

  // Apply highlight to current selection
  applyHighlight(ann)

  // Clean up
  showNoteInput.value = false
  noteText.value = ''
  selectedText.value = ''
  selectionRange.value = null
  window.getSelection()?.removeAllRanges()
}

function cancelNote() {
  showNoteInput.value = false
  noteText.value = ''
  window.getSelection()?.removeAllRanges()
}

function deleteAnnotation(localIdx) {
  const ann = currentPageAnnotations.value[localIdx]
  const globalIdx = allAnnotations.value.indexOf(ann)
  if (globalIdx !== -1) {
    // Remove this specific highlight from DOM by timestamp
    removeHighlightByTimestamp(ann.timestamp)
    allAnnotations.value.splice(globalIdx, 1)
    saveToStorage()
  }
}

function removeHighlightByTimestamp(timestamp) {
  const container = getContentContainer()
  if (!container) return
  const mark = container.querySelector(`mark.user-annotation[data-timestamp="${timestamp}"]`)
  if (mark) {
    const parent = mark.parentNode
    while (mark.firstChild) {
      parent.insertBefore(mark.firstChild, mark)
    }
    parent.removeChild(mark)
    parent.normalize()
  }
}

function togglePanel() {
  showPanel.value = !showPanel.value
}

function getTextContext(text) {
  // Store surrounding text for better matching later
  const container = getContentContainer()
  if (!container) return ''
  const fullText = container.textContent || ''
  const idx = fullText.indexOf(text)
  if (idx === -1) return ''
  const start = Math.max(0, idx - 30)
  const end = Math.min(fullText.length, idx + text.length + 30)
  return fullText.slice(start, end)
}

function applyHighlight(ann) {
  // Try to highlight using the current selection range first
  if (selectionRange.value) {
    try {
      wrapRangeWithMark(selectionRange.value, ann)
      return
    } catch { /* fallback below */ }
  }
  // Fallback: find text in DOM
  highlightTextInDom(ann)
}

function wrapRangeWithMark(range, ann) {
  const mark = document.createElement('mark')
  mark.className = 'user-annotation'
  mark.style.backgroundColor = ann.color
  mark.style.borderRadius = '2px'
  mark.style.padding = '0 1px'
  mark.style.cursor = 'pointer'
  mark.dataset.timestamp = ann.timestamp
  if (ann.note) {
    mark.title = ann.note
  }
  try {
    range.surroundContents(mark)
  } catch {
    // If surroundContents fails (crosses element boundaries), use extractContents
    const fragment = range.extractContents()
    mark.appendChild(fragment)
    range.insertNode(mark)
  }
}

function highlightTextInDom(ann) {
  const container = getContentContainer()
  if (!container) return

  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null)
  const textToFind = ann.text
  let node

  while ((node = walker.nextNode())) {
    const idx = node.textContent.indexOf(textToFind)
    if (idx === -1) continue

    // Skip if already inside a user-annotation
    if (node.parentElement?.closest('.user-annotation')) continue

    const range = document.createRange()
    range.setStart(node, idx)
    range.setEnd(node, idx + textToFind.length)

    const mark = document.createElement('mark')
    mark.className = 'user-annotation'
    mark.style.backgroundColor = ann.color
    mark.style.borderRadius = '2px'
    mark.style.padding = '0 1px'
    mark.style.cursor = 'pointer'
    mark.dataset.timestamp = ann.timestamp
    if (ann.note) mark.title = ann.note

    try {
      range.surroundContents(mark)
    } catch {
      // skip if can't wrap
    }
    return // only first match
  }
}

function removeHighlightsFromDom() {
  const container = getContentContainer()
  if (!container) return
  const marks = container.querySelectorAll('mark.user-annotation')
  marks.forEach((mark) => {
    const parent = mark.parentNode
    while (mark.firstChild) {
      parent.insertBefore(mark.firstChild, mark)
    }
    parent.removeChild(mark)
    parent.normalize()
  })
}

function restoreHighlights() {
  currentPageAnnotations.value.forEach((ann) => {
    highlightTextInDom(ann)
  })
}

function formatTime(ts) {
  const d = new Date(ts)
  return d.toLocaleDateString('zh-CN') + ' ' + d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

function truncate(s, max) {
  return s.length > max ? s.slice(0, max) + '...' : s
}

// Sync when AnnotationList (or another tab) modifies annotations
function onAnnotationsChanged() {
  loadAnnotations()
  removeHighlightsFromDom()
  nextTick(() => restoreHighlights())
}

// Watch for route changes
watch(
  () => route.path,
  () => {
    showToolbar.value = false
    showNoteInput.value = false
    showPanel.value = false
    // Reload from storage in case AnnotationList changed it
    loadAnnotations()
    // Small delay to wait for content to render
    setTimeout(() => {
      removeHighlightsFromDom()
      restoreHighlights()
    }, 300)
  }
)

onMounted(() => {
  loadAnnotations()
  document.addEventListener('mouseup', handleMouseUp)
  window.addEventListener('annotations-changed', onAnnotationsChanged)
  // Restore highlights after page content is rendered
  setTimeout(restoreHighlights, 500)
})

onUnmounted(() => {
  document.removeEventListener('mouseup', handleMouseUp)
  window.removeEventListener('annotations-changed', onAnnotationsChanged)
})
</script>

<style scoped>
.annotator-toolbar {
  position: absolute;
  z-index: 100;
  display: flex;
  gap: 6px;
  padding: 6px 10px;
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
}

.color-btn {
  width: 24px;
  height: 24px;
  border: 2px solid transparent;
  border-radius: 50%;
  cursor: pointer;
  transition: transform 0.15s, border-color 0.15s;
}

.color-btn:hover {
  transform: scale(1.2);
  border-color: var(--vp-c-text-1);
}

.annotator-note-popup {
  position: absolute;
  z-index: 100;
  width: 280px;
  padding: 12px;
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.15);
}

.note-popup-header {
  font-size: 13px;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: 8px;
}

.annotator-note-popup textarea {
  width: 100%;
  padding: 8px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 13px;
  resize: vertical;
  font-family: inherit;
}

.annotator-note-popup textarea:focus {
  outline: none;
  border-color: var(--vp-c-brand-1);
}

.note-popup-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 8px;
}

.note-btn {
  padding: 4px 12px;
  border: none;
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.15s;
}

.note-btn.cancel {
  background: var(--vp-c-default-soft);
  color: var(--vp-c-text-2);
}

.note-btn.save {
  background: var(--vp-c-brand-1);
  color: white;
}

.note-btn .shortcut {
  opacity: 0.6;
  font-size: 10px;
}

.annotator-badge {
  position: fixed;
  bottom: 24px;
  right: 24px;
  z-index: 99;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 14px;
  background: var(--vp-c-brand-1);
  color: white;
  border-radius: 20px;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  transition: transform 0.2s, box-shadow 0.2s;
  font-size: 13px;
  font-weight: 500;
}

.annotator-badge:hover {
  transform: scale(1.05);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
}

.badge-count {
  font-weight: 700;
}

.annotator-panel {
  position: fixed;
  bottom: 70px;
  right: 24px;
  z-index: 99;
  width: 340px;
  max-height: 400px;
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid var(--vp-c-divider);
  font-size: 14px;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.panel-close {
  background: none;
  border: none;
  font-size: 20px;
  cursor: pointer;
  color: var(--vp-c-text-3);
  line-height: 1;
}

.panel-body {
  padding: 8px;
  overflow-y: auto;
  flex: 1;
}

.panel-item {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 8px;
  border-radius: 8px;
  transition: background 0.15s;
}

.panel-item:hover {
  background: var(--vp-c-default-soft);
}

.panel-item-text {
  flex: 1;
  border-left: 3px solid;
  padding-left: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.highlighted-text {
  font-size: 12px;
  color: var(--vp-c-text-2);
  line-height: 1.4;
}

.note-content {
  font-size: 13px;
  color: var(--vp-c-text-1);
  font-weight: 500;
}

.note-time {
  font-size: 11px;
  color: var(--vp-c-text-3);
}

.panel-item-delete {
  background: none;
  border: none;
  font-size: 16px;
  color: var(--vp-c-text-3);
  cursor: pointer;
  padding: 2px 6px;
  border-radius: 4px;
  transition: background 0.15s, color 0.15s;
  flex-shrink: 0;
}

.panel-item-delete:hover {
  background: #fee2e2;
  color: #dc2626;
}

/* Transitions */
.toolbar-fade-enter-active,
.toolbar-fade-leave-active {
  transition: opacity 0.2s, transform 0.2s;
}
.toolbar-fade-enter-from,
.toolbar-fade-leave-to {
  opacity: 0;
  transform: translateY(4px);
}

.panel-slide-enter-active,
.panel-slide-leave-active {
  transition: opacity 0.25s, transform 0.25s;
}
.panel-slide-enter-from,
.panel-slide-leave-to {
  opacity: 0;
  transform: translateY(16px);
}
</style>

<style>
/* Global styles for annotation marks (not scoped) */
mark.user-annotation {
  cursor: pointer;
  transition: filter 0.15s;
}
mark.user-annotation:hover {
  filter: brightness(0.92);
}
</style>
