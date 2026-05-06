<template>
  <div class="srs-review">
    <header class="srs-header">
      <h2>📚 今日复习池</h2>
      <p class="srs-sub">基于 SM-2 间隔重复算法，到期的微选择题会自动出现在这里。</p>
    </header>

    <div v-if="dueIds.length === 0" class="srs-empty">
      <p>✨ 今日没有到期的复习卡。</p>
      <p class="hint">先去 <a :href="mapHref">关卡地图</a> 学新内容吧！</p>
    </div>

    <div v-else class="srs-list">
      <p class="srs-count">今日 <strong>{{ dueIds.length }}</strong> 张卡片到期 · 按题目分组：</p>
      <div v-for="group in groups" :key="group.problemId" class="srs-group">
        <a class="srs-group-title" :href="problemHref(group.problemId)">
          <span>{{ group.problemTitle }}</span>
          <span class="srs-group-count">{{ group.cardIds.length }} 张</span>
        </a>
        <ul class="srs-card-list">
          <li v-for="cid in group.cardIds" :key="cid">
            <code>{{ cid }}</code>
            <span class="srs-due">due: {{ state.cards[cid].due }} · ease: {{ state.cards[cid].ease.toFixed(2) }}</span>
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { withBase } from 'vitepress'
import { loadState, dueCards } from '../../../leetcode/_data/progress'
import { problems } from '../../../leetcode/_data/problems/index'
import { patterns } from '../../../leetcode/_data/patterns'

const state = ref(loadState())
const dueIds = computed(() => dueCards(state.value))

const groups = computed(() => {
  const byProblem: Record<string, string[]> = {}
  for (const cid of dueIds.value) {
    const pid = cid.split('.')[0]
    ;(byProblem[pid] ||= []).push(cid)
  }
  return Object.entries(byProblem).map(([pid, cardIds]) => ({
    problemId: pid,
    problemTitle: problems[pid]?.title.zh ?? pid,
    cardIds,
  }))
})

const mapHref = withBase('/leetcode/')

function problemHref(pid: string): string {
  const p = problems[pid]
  if (!p) return mapHref
  const pattern = patterns.find(pt => pt.slug === p.pattern)
  return withBase(`/leetcode/patterns/${pattern?.slug ?? 'two-pointer'}`)
}

onMounted(() => {
  state.value = loadState()
})
</script>

<style scoped>
.srs-review { max-width: 720px; margin: 0 auto; }
.srs-header { text-align: center; margin: 20px 0; }
.srs-header h2 { margin: 0; }
.srs-sub { color: var(--vp-c-text-2); font-size: 14px; }

.srs-empty { padding: 40px; text-align: center; background: var(--vp-c-bg-soft); border-radius: 12px; }
.srs-empty .hint { color: var(--vp-c-text-2); font-size: 14px; }

.srs-count { font-size: 14px; color: var(--vp-c-text-2); margin: 12px 0; }
.srs-group {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  padding: 12px 16px;
  margin: 10px 0;
}
.srs-group-title {
  display: flex;
  justify-content: space-between;
  font-weight: 600;
  text-decoration: none !important;
  color: var(--vp-c-text-1) !important;
}
.srs-group-count {
  background: var(--vp-c-brand-1);
  color: white;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 12px;
}
.srs-card-list { margin: 8px 0 0; padding-left: 18px; font-size: 13px; }
.srs-card-list li { margin: 4px 0; }
.srs-due { color: var(--vp-c-text-2); font-size: 11px; margin-left: 8px; }
</style>
