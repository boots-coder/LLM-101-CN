<script setup lang="ts">
import { withBase } from 'vitepress'

const stats = [
  { value: '86', unit: '篇', label: '教程正文' },
  { value: '6.4', unit: '万行', label: '内容体量' },
  { value: '994', unit: '段', label: '可跑代码' },
  { value: '26', unit: '套', label: '渐进练习' },
]

const modules = [
  { no: '01', name: '基础知识', link: '/fundamentals/', desc: '数学、Python、神经网络、NLP', meta: '5 篇' },
  { no: '02', name: '模型架构', link: '/architecture/', desc: 'Transformer → GPT → Llama → DeepSeek → 高效注意力', meta: '11 篇', core: true },
  { no: '03', name: '训练', link: '/training/', desc: '预训练、SFT、RLHF / DPO / GRPO、推理模型', meta: '11 篇' },
  { no: '04', name: '工程化', link: '/engineering/', desc: '推理优化、量化、分布式、部署与评估', meta: '11 篇' },
  { no: '05', name: '应用', link: '/applications/', desc: 'Prompt、RAG、Agent、多模态、Harness', meta: '8 篇' },
  { no: '06', name: '深度剖析', link: '/deep-dives/', desc: 'vLLM / DeepSeek / Kimi 等源码与技术报告', meta: '21 篇' },
]

const features = [
  {
    title: '体系化，不是博客合集',
    desc: '六大模块按依赖顺序编排，每篇都标明「在大模型体系中的位置」。你随时知道自己在整张地图的哪里。',
    icon: 'grid',
  },
  {
    title: '每个概念都能写出来',
    desc: '994 段可运行代码，从 Scaled Dot-Product Attention 到 GRPO，核心机制一律从零实现，不调库糊弄。',
    icon: 'code',
  },
  {
    title: '苏格拉底式追问',
    desc: '不直接给结论。每章末尾用问题逼你想清楚边界条件，配四级练习：选择题 → 填空 → 模块实现 → 完整实现。',
    icon: 'spark',
  },
  {
    title: '读真正的源码',
    desc: '深入 vLLM、DeepSeek-V4、Kimi K2 的实现与论文，既看设计精妙处，也指出可疑之处。',
    icon: 'layers',
  },
]

// 因果注意力掩码——本站所讲内容本身，作为视觉母题
const size = 12
const cells = Array.from({ length: size * size }, (_, i) => {
  const r = Math.floor(i / size)
  const c = i % size
  if (c > r) return 0
  // 近因衰减 + 对首 token 的稳定关注（attention sink），确定性生成
  const recency = 1 / (1 + (r - c) * 0.55)
  const sink = c === 0 ? 0.45 : 0
  return Math.min(1, recency * 0.9 + sink)
})
</script>

<template>
  <div class="lh">
    <!-- ── Hero ───────────────────────────────── -->
    <section class="lh-hero">
      <div class="lh-hero-main">
        <p class="lh-eyebrow">LLM&nbsp;101&nbsp;·&nbsp;中文大模型教程</p>
        <h1 class="lh-title">
          把大模型<br />从原理学到<span class="lh-title-em">能手写</span>
        </h1>
        <p class="lh-lede">
          从 Transformer 的一次矩阵乘法，到 RLHF 的一条梯度路径，再到 2026 年各家在注意力机制上的分歧。
          一本按依赖顺序编排、每个结论都配可运行代码的在线教材。
        </p>
        <div class="lh-actions">
          <a class="lh-btn lh-btn-primary" :href="withBase('/architecture/transformer')">开始学习</a>
          <a class="lh-btn" :href="withBase('/roadmap')">学习路线</a>
          <a class="lh-btn" :href="withBase('/exercises/')">练习系统</a>
        </div>
      </div>

      <div class="lh-hero-visual" aria-hidden="true">
        <div class="lh-mask">
          <span
            v-for="(v, i) in cells"
            :key="i"
            class="lh-mask-cell"
            :style="{ opacity: v === 0 ? 0.035 : 0.10 + v * 0.62 }"
          />
        </div>
        <p class="lh-mask-cap">Causal attention mask · 本站第一章讲的就是它</p>
      </div>
    </section>

    <!-- ── Stats ──────────────────────────────── -->
    <section class="lh-stats">
      <div v-for="s in stats" :key="s.label" class="lh-stat">
        <p class="lh-stat-v">{{ s.value }}<span class="lh-stat-u">{{ s.unit }}</span></p>
        <p class="lh-stat-l">{{ s.label }}</p>
      </div>
    </section>

    <!-- ── Features ───────────────────────────── -->
    <section class="lh-section">
      <h2 class="lh-h2">这本教材和别的有什么不同</h2>
      <div class="lh-features">
        <article v-for="f in features" :key="f.title" class="lh-card">
          <svg class="lh-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">
            <template v-if="f.icon === 'grid'">
              <rect x="3" y="3" width="7" height="7" rx="1.5" />
              <rect x="14" y="3" width="7" height="7" rx="1.5" />
              <rect x="3" y="14" width="7" height="7" rx="1.5" />
              <rect x="14" y="14" width="7" height="7" rx="1.5" />
            </template>
            <template v-else-if="f.icon === 'code'">
              <path d="M8 18 2 12l6-6" /><path d="m16 6 6 6-6 6" /><path d="M13.5 4 10.5 20" />
            </template>
            <template v-else-if="f.icon === 'spark'">
              <path d="M12 3v3m0 12v3M3 12h3m12 0h3M5.6 5.6l2.1 2.1m8.6 8.6 2.1 2.1m0-12.8-2.1 2.1m-8.6 8.6-2.1 2.1" />
              <circle cx="12" cy="12" r="3.2" />
            </template>
            <template v-else>
              <path d="m12 2 9 5-9 5-9-5 9-5Z" /><path d="m3 12 9 5 9-5" /><path d="m3 17 9 5 9-5" />
            </template>
          </svg>
          <h3 class="lh-card-t">{{ f.title }}</h3>
          <p class="lh-card-d">{{ f.desc }}</p>
        </article>
      </div>
    </section>

    <!-- ── Modules ────────────────────────────── -->
    <section class="lh-section">
      <h2 class="lh-h2">六大模块</h2>
      <div class="lh-modules">
        <a v-for="m in modules" :key="m.no" class="lh-mod" :class="{ 'is-core': m.core }"
           :href="withBase(m.link)">
          <span class="lh-mod-no">{{ m.no }}</span>
          <span class="lh-mod-body">
            <span class="lh-mod-name">
              {{ m.name }}<em v-if="m.core" class="lh-mod-tag">核心</em>
            </span>
            <span class="lh-mod-desc">{{ m.desc }}</span>
          </span>
          <span class="lh-mod-meta">{{ m.meta }}</span>
        </a>
      </div>
    </section>
  </div>
</template>

<style>
/* 设计语言：暖白底、近黑主色、发丝边框、几乎无阴影、大量留白 */
.lh {
  --lh-bg: #fdfcfc;
  --lh-card: #ffffff;
  --lh-fg: #09090b;
  --lh-muted: #6b7280;
  --lh-line: rgba(0, 0, 0, 0.08);
  --lh-line-strong: rgba(0, 0, 0, 0.14);
  --lh-accent: #2563eb;
  --lh-r: 10px;

  max-width: 1152px;
  margin: 0 auto;
  padding: 0 24px 96px;
  color: var(--lh-fg);
  font-variant-numeric: tabular-nums;
}
html.dark .lh {
  --lh-bg: #0b0b0d;
  --lh-card: #141417;
  --lh-fg: #fafafa;
  --lh-muted: #a1a1aa;
  --lh-line: rgba(255, 255, 255, 0.10);
  --lh-line-strong: rgba(255, 255, 255, 0.18);
  --lh-accent: #60a5fa;
}

/* ── Hero ── */
.lh-hero {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
  gap: 64px;
  align-items: center;
  padding: 88px 0 72px;
}
.lh-eyebrow {
  margin: 0 0 22px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--lh-muted);
}
.lh-title {
  margin: 0;
  font-size: clamp(38px, 5.2vw, 62px);
  line-height: 1.14;
  letter-spacing: -0.022em;
  font-weight: 700;
  border: 0;
}
.lh-title-em {
  position: relative;
  white-space: nowrap;
}
.lh-title-em::after {
  content: '';
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0.08em;
  height: 0.30em;
  background: var(--lh-accent);
  opacity: 0.18;
  border-radius: 2px;
}
.lh-lede {
  margin: 26px 0 0;
  max-width: 34em;
  font-size: 16.5px;
  line-height: 1.78;
  color: var(--lh-muted);
}
.lh-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 38px;
}
.lh-btn {
  display: inline-flex;
  align-items: center;
  height: 42px;
  padding: 0 20px;
  border: 1px solid var(--lh-line-strong);
  border-radius: var(--lh-r);
  font-size: 14.5px;
  font-weight: 500;
  color: var(--lh-fg);
  text-decoration: none;
  transition: background-color 0.18s ease, border-color 0.18s ease, transform 0.18s ease;
}
.lh-btn:hover {
  background: rgba(0, 0, 0, 0.035);
  border-color: var(--lh-fg);
}
html.dark .lh-btn:hover { background: rgba(255, 255, 255, 0.06); }
.lh-btn-primary {
  background: var(--lh-fg);
  border-color: var(--lh-fg);
  color: var(--lh-bg);
}
.lh-btn-primary:hover {
  background: var(--lh-fg);
  opacity: 0.86;
  transform: translateY(-1px);
}

/* 因果掩码视觉 */
.lh-hero-visual { display: flex; flex-direction: column; align-items: center; gap: 18px; }
.lh-mask {
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: 4px;
  width: 100%;
  max-width: 340px;
  aspect-ratio: 1;
  padding: 18px;
  border: 1px solid var(--lh-line);
  border-radius: 14px;
  background: var(--lh-card);
}
.lh-mask-cell {
  border-radius: 2.5px;
  background: var(--lh-fg);
}
.lh-mask-cap {
  margin: 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11.5px;
  color: var(--lh-muted);
  text-align: center;
}

/* ── Stats ── */
.lh-stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  border-top: 1px solid var(--lh-line);
  border-bottom: 1px solid var(--lh-line);
}
.lh-stat { padding: 30px 8px; text-align: center; }
.lh-stat + .lh-stat { border-left: 1px solid var(--lh-line); }
.lh-stat-v {
  margin: 0;
  font-size: 32px;
  font-weight: 650;
  letter-spacing: -0.02em;
  line-height: 1.1;
}
.lh-stat-u { margin-left: 3px; font-size: 14px; font-weight: 500; color: var(--lh-muted); }
.lh-stat-l { margin: 8px 0 0; font-size: 13px; color: var(--lh-muted); }

/* ── Sections ── */
.lh-section { padding-top: 76px; }
.lh-h2 {
  margin: 0 0 28px;
  font-size: 21px;
  font-weight: 650;
  letter-spacing: -0.012em;
  border: 0;
  padding: 0;
}

.lh-features { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
.lh-card {
  padding: 28px;
  border: 1px solid var(--lh-line);
  border-radius: var(--lh-r);
  background: var(--lh-card);
  transition: border-color 0.18s ease;
}
.lh-card:hover { border-color: var(--lh-line-strong); }
.lh-icon { width: 22px; height: 22px; color: var(--lh-muted); }
.lh-card-t { margin: 18px 0 8px; font-size: 16px; font-weight: 600; letter-spacing: -0.008em; }
.lh-card-d { margin: 0; font-size: 14.5px; line-height: 1.75; color: var(--lh-muted); }

/* ── Modules ── */
.lh-modules { border-top: 1px solid var(--lh-line); }
.lh-mod {
  display: grid;
  grid-template-columns: 56px minmax(0, 1fr) auto;
  align-items: baseline;
  gap: 16px;
  padding: 22px 12px;
  border-bottom: 1px solid var(--lh-line);
  text-decoration: none;
  color: inherit;
  transition: background-color 0.16s ease, padding-left 0.16s ease;
}
.lh-mod:hover { background: rgba(0, 0, 0, 0.022); padding-left: 20px; }
html.dark .lh-mod:hover { background: rgba(255, 255, 255, 0.04); }
.lh-mod-no {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 13px;
  color: var(--lh-muted);
}
.lh-mod-body { display: flex; flex-direction: column; gap: 5px; min-width: 0; }
.lh-mod-name { font-size: 16px; font-weight: 600; letter-spacing: -0.008em; }
.lh-mod-tag {
  margin-left: 9px;
  padding: 2px 7px;
  border: 1px solid var(--lh-line-strong);
  border-radius: 5px;
  font-size: 11px;
  font-style: normal;
  font-weight: 500;
  color: var(--lh-muted);
  vertical-align: 2px;
}
.lh-mod-desc { font-size: 14px; line-height: 1.6; color: var(--lh-muted); }
.lh-mod-meta {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12.5px;
  color: var(--lh-muted);
  white-space: nowrap;
}

/* ── 首页下半部分（由 index.md 的 markdown 渲染） ── */
.lh-tail { padding-top: 0; }
.lh-tail h2 {
  margin: 76px 0 14px;
  padding: 0;
  border: 0;
  font-size: 21px;
  font-weight: 650;
  letter-spacing: -0.012em;
}
.lh-tail h3 {
  margin: 40px 0 8px;
  font-size: 16px;
  font-weight: 600;
  letter-spacing: -0.008em;
}
.lh-tail p {
  margin: 10px 0 0;
  max-width: 44em;
  font-size: 15px;
  line-height: 1.8;
  color: var(--lh-muted);
}
.lh-tail strong { color: var(--lh-fg); font-weight: 600; }
.lh-tail a { color: var(--lh-accent); text-decoration: none; }
.lh-tail a:hover { text-decoration: underline; }
.lh-tail .mermaid,
.lh-tail div[class*='mermaid'] {
  margin: 20px 0 0;
  padding: 24px 16px;
  border: 1px solid var(--lh-line);
  border-radius: var(--lh-r);
  background: var(--lh-card);
  overflow-x: auto;
}

/* ── 响应式 ── */
@media (max-width: 900px) {
  .lh-hero { grid-template-columns: 1fr; gap: 44px; padding: 56px 0 56px; }
  .lh-hero-visual { order: -1; }
  .lh-mask { max-width: 260px; }
  .lh-features { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
  .lh { padding: 0 18px 64px; }
  .lh-stats { grid-template-columns: repeat(2, 1fr); }
  .lh-stat:nth-child(3) { border-left: 0; }
  .lh-stat:nth-child(n + 3) { border-top: 1px solid var(--lh-line); }
  .lh-mod { grid-template-columns: 40px minmax(0, 1fr); }
  .lh-mod-meta { display: none; }
}
</style>
