<template>
  <div class="annotation-list">
    <div class="annotation-list-header">
      <div class="stats">
        <span class="stat-item">
          共 <strong>{{ annotations.length }}</strong> 条笔记
        </span>
        <span class="stat-item">
          来自 <strong>{{ pageCount }}</strong> 个页面
        </span>
      </div>
      <div class="actions">
        <button v-if="annotations.length > 0" class="action-btn export" @click="exportJSON">
          导出 JSON
        </button>
        <button v-if="annotations.length > 0" class="action-btn clear" @click="clearAll">
          清空全部
        </button>
      </div>
    </div>

    <div v-if="annotations.length === 0" class="empty-state">
      <div class="empty-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
        </svg>
      </div>
      <p class="empty-title">还没有笔记</p>
      <p class="empty-desc">
        在阅读文档时，选中任意文字即可添加高亮和笔记。<br />
        你的笔记会保存在浏览器本地存储中。
      </p>
    </div>

    <div v-for="group in groupedAnnotations" :key="group.page" class="page-group">
      <div class="page-title">
        <a :href="group.page" class="page-link">{{ group.page }}</a>
        <span class="page-count">{{ group.items.length }} 条</span>
      </div>
      <div class="annotation-cards">
        <div
          v-for="ann in group.items"
          :key="ann.timestamp"
          class="annotation-card"
          :style="{ borderLeftColor: ann.color }"
        >
          <div class="card-body">
            <div class="card-text" :style="{ backgroundColor: ann.color + '40' }">
              "{{ ann.text }}"
            </div>
            <div v-if="ann.note" class="card-note">{{ ann.note }}</div>
            <div class="card-meta">
              <span class="card-time">{{ formatTime(ann.timestamp) }}</span>
              <a :href="ann.page" class="card-link">跳转到页面</a>
            </div>
          </div>
          <button class="card-delete" title="删除此笔记" @click="deleteAnnotation(ann)">
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

const STORAGE_KEY = 'llm101-annotations'
const annotations = ref([])

function loadAnnotations() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) annotations.value = JSON.parse(raw)
  } catch {
    annotations.value = []
  }
}

function saveToStorage() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations.value))
}

const pageCount = computed(() => {
  const pages = new Set(annotations.value.map((a) => a.page))
  return pages.size
})

const groupedAnnotations = computed(() => {
  const groups = {}
  // Sort by timestamp descending
  const sorted = [...annotations.value].sort((a, b) => b.timestamp - a.timestamp)
  sorted.forEach((ann) => {
    if (!groups[ann.page]) {
      groups[ann.page] = { page: ann.page, items: [] }
    }
    groups[ann.page].items.push(ann)
  })
  return Object.values(groups)
})

function deleteAnnotation(ann) {
  const idx = annotations.value.findIndex((a) => a.timestamp === ann.timestamp)
  if (idx !== -1) {
    annotations.value.splice(idx, 1)
    saveToStorage()
    window.dispatchEvent(new Event('annotations-changed'))
  }
}

function clearAll() {
  if (confirm('确定要清空所有笔记吗？此操作不可撤销。')) {
    annotations.value = []
    saveToStorage()
    window.dispatchEvent(new Event('annotations-changed'))
  }
}

function exportJSON() {
  const data = JSON.stringify(annotations.value, null, 2)
  const blob = new Blob([data], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `llm101-annotations-${new Date().toISOString().slice(0, 10)}.json`
  a.click()
  URL.revokeObjectURL(url)
}

function formatTime(ts) {
  const d = new Date(ts)
  return d.toLocaleDateString('zh-CN') + ' ' + d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

onMounted(() => {
  loadAnnotations()
})
</script>

<style scoped>
.annotation-list {
  max-width: 720px;
  margin: 0 auto;
}

.annotation-list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--vp-c-divider);
}

.stats {
  display: flex;
  gap: 16px;
}

.stat-item {
  font-size: 14px;
  color: var(--vp-c-text-2);
}

.stat-item strong {
  color: var(--vp-c-text-1);
}

.actions {
  display: flex;
  gap: 8px;
}

.action-btn {
  padding: 6px 14px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
}

.action-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.action-btn.clear:hover {
  border-color: #dc2626;
  color: #dc2626;
}

.empty-state {
  text-align: center;
  padding: 60px 20px;
}

.empty-icon {
  color: var(--vp-c-text-3);
  margin-bottom: 16px;
}

.empty-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: 8px;
}

.empty-desc {
  font-size: 14px;
  color: var(--vp-c-text-3);
  line-height: 1.6;
}

.page-group {
  margin-bottom: 28px;
}

.page-title {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}

.page-link {
  font-size: 15px;
  font-weight: 600;
  color: var(--vp-c-brand-1);
  text-decoration: none;
}

.page-link:hover {
  text-decoration: underline;
}

.page-count {
  font-size: 12px;
  color: var(--vp-c-text-3);
  background: var(--vp-c-default-soft);
  padding: 2px 8px;
  border-radius: 10px;
}

.annotation-cards {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.annotation-card {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 14px;
  border: 1px solid var(--vp-c-divider);
  border-left: 4px solid;
  border-radius: 8px;
  background: var(--vp-c-bg);
  transition: box-shadow 0.2s;
}

.annotation-card:hover {
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
}

.card-body {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.card-text {
  font-size: 13px;
  color: var(--vp-c-text-2);
  line-height: 1.5;
  padding: 6px 10px;
  border-radius: 6px;
  word-break: break-word;
}

.card-note {
  font-size: 14px;
  color: var(--vp-c-text-1);
  font-weight: 500;
  line-height: 1.5;
}

.card-meta {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 12px;
}

.card-time {
  color: var(--vp-c-text-3);
}

.card-link {
  color: var(--vp-c-brand-1);
  text-decoration: none;
  font-weight: 500;
}

.card-link:hover {
  text-decoration: underline;
}

.card-delete {
  background: none;
  border: none;
  color: var(--vp-c-text-3);
  cursor: pointer;
  padding: 4px;
  border-radius: 4px;
  transition: all 0.15s;
  flex-shrink: 0;
}

.card-delete:hover {
  background: #fee2e2;
  color: #dc2626;
}
</style>
