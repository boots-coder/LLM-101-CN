/**
 * lint-content.js
 *
 * 校验项目内容的一致性和完整性：
 * 1. YAML frontmatter 必填字段检查（title, description, topics）
 * 2. config.ts sidebar 与实际文件的一致性检查
 * 3. 内部链接有效性检查（死链接检测）
 * 4. CONTENT_MAP.md 是否过期检查
 *
 * 用法：node scripts/lint-content.js
 * 或：  npm run lint
 */

import { readFileSync, readdirSync, existsSync } from 'fs'
import { join, dirname, basename, resolve } from 'path'
import { fileURLToPath } from 'url'
import { execSync } from 'child_process'

const __dirname = dirname(fileURLToPath(import.meta.url))
const ROOT = join(__dirname, '..')
const DOCS = join(ROOT, 'docs')

const MODULE_ORDER = [
  'fundamentals', 'architecture', 'training', 'engineering',
  'applications', 'deep-dives', 'exercises'
]

let errors = []
let warnings = []

function error(msg) { errors.push(`❌ ${msg}`) }
function warn(msg) { warnings.push(`⚠️  ${msg}`) }

// ── 1. Frontmatter 校验 ────────────────────────────────────

function parseFrontmatter(filePath) {
  const content = readFileSync(filePath, 'utf-8')
  const lines = content.split('\n')
  if (lines[0] !== '---') return null
  const endIdx = lines.indexOf('---', 1)
  if (endIdx < 0) return null
  const yaml = lines.slice(1, endIdx).join('\n')
  const result = {}
  for (const line of yaml.split('\n')) {
    const match = line.match(/^(\w+):\s*(.+)$/)
    if (match) {
      const [, key, value] = match
      if (value.startsWith('[') && value.endsWith(']')) {
        result[key] = value.slice(1, -1).split(',').map(s => s.trim().replace(/^["']|["']$/g, ''))
      } else {
        result[key] = value.replace(/^["']|["']$/g, '')
      }
    }
  }
  return result
}

function checkFrontmatter() {
  console.log('\n📋 检查 YAML Frontmatter...')
  let checked = 0

  for (const mod of MODULE_ORDER) {
    const modDir = join(DOCS, mod)
    if (!existsSync(modDir)) continue

    const files = readdirSync(modDir).filter(f => f.endsWith('.md') && f !== 'index.md')
    for (const f of files) {
      const filePath = join(modDir, f)
      const fm = parseFrontmatter(filePath)
      const rel = `${mod}/${f}`
      checked++

      if (!fm) {
        error(`${rel}: 缺少 YAML frontmatter`)
        continue
      }
      if (!fm.title) error(`${rel}: frontmatter 缺少 title`)
      if (!fm.description) error(`${rel}: frontmatter 缺少 description`)
      if (!fm.topics || fm.topics.length === 0) error(`${rel}: frontmatter 缺少 topics`)
    }
  }

  console.log(`   检查了 ${checked} 个文件`)
}

// ── 2. Sidebar 一致性校验 ──────────────────────────────────

function extractSidebarLinks(configContent) {
  // 提取 config.ts 中所有 link: '/xxx/yyy' 的路径
  const links = []
  const regex = /link:\s*['"]([^'"]+)['"]/g
  let match
  while ((match = regex.exec(configContent)) !== null) {
    links.push(match[1])
  }
  return links
}

function checkSidebarConsistency() {
  console.log('\n📋 检查 Sidebar 一致性...')
  const configPath = join(DOCS, '.vitepress', 'config.ts')
  const configContent = readFileSync(configPath, 'utf-8')
  const sidebarLinks = extractSidebarLinks(configContent)

  // 检查 sidebar 中的链接是否对应实际文件（跳过外部链接）
  for (const link of sidebarLinks) {
    if (link.startsWith('http')) continue
    // /module/ 形式指向 index.md
    if (link.endsWith('/')) {
      const indexPath = join(DOCS, link.slice(1), 'index.md')
      if (!existsSync(indexPath)) {
        error(`Sidebar 链接 "${link}" 对应的 index.md 不存在`)
      }
    } else {
      // /module/file 形式
      const filePath = join(DOCS, link.slice(1) + '.md')
      if (!existsSync(filePath)) {
        error(`Sidebar 链接 "${link}" 对应的文件 ${link.slice(1)}.md 不存在`)
      }
    }
  }

  // 检查实际文件是否在 sidebar 中
  for (const mod of MODULE_ORDER) {
    const modDir = join(DOCS, mod)
    if (!existsSync(modDir)) continue

    const files = readdirSync(modDir).filter(f => f.endsWith('.md') && f !== 'index.md')
    for (const f of files) {
      const expectedLink = `/${mod}/${f.replace('.md', '')}`
      if (!sidebarLinks.includes(expectedLink)) {
        error(`文件 ${mod}/${f} 存在但不在 sidebar 中（期望链接: ${expectedLink}）`)
      }
    }
  }

  console.log(`   检查了 ${sidebarLinks.length} 个 sidebar 条目`)
}

// ── 3. 内部链接校验 ────────────────────────────────────────

function checkInternalLinks() {
  console.log('\n📋 检查内部链接...')
  let checked = 0
  let linkCount = 0

  const allMdFiles = []
  for (const mod of MODULE_ORDER) {
    const modDir = join(DOCS, mod)
    if (!existsSync(modDir)) continue
    const files = readdirSync(modDir).filter(f => f.endsWith('.md'))
    for (const f of files) {
      allMdFiles.push(join(modDir, f))
    }
  }
  // 也检查顶层 md
  for (const f of readdirSync(DOCS).filter(f => f.endsWith('.md'))) {
    allMdFiles.push(join(DOCS, f))
  }

  for (const filePath of allMdFiles) {
    const rawContent = readFileSync(filePath, 'utf-8')
    // 移除代码块，避免代码中的 [i](xxx) 被误判为链接
    // 1) 三反引号 fenced code  2) 单行 inline code  3) <CodeMasker> 等 Vue 代码组件块（块内是裸 Python）
    const content = rawContent
      .replace(/```[\s\S]*?```/g, '')
      .replace(/<CodeMasker[\s\S]*?<\/CodeMasker>/g, '')
      .replace(/`[^`]+`/g, '')
    const rel = filePath.replace(DOCS + '/', '')
    checked++

    // 匹配 markdown 链接 [text](path) — 跳过外部链接和锚点
    const linkRegex = /\[([^\]]*)\]\(([^)]+)\)/g
    let match
    while ((match = linkRegex.exec(content)) !== null) {
      const linkTarget = match[2]

      // 跳过外部链接、锚点、图片 URL
      if (linkTarget.startsWith('http') || linkTarget.startsWith('#') || linkTarget.startsWith('mailto:')) continue

      linkCount++

      // 处理绝对路径 /module/file
      let targetPath
      if (linkTarget.startsWith('/')) {
        const cleanPath = linkTarget.split('#')[0] // 去掉锚点
        if (cleanPath.endsWith('/')) {
          targetPath = join(DOCS, cleanPath.slice(1), 'index.md')
        } else {
          targetPath = join(DOCS, cleanPath.slice(1) + '.md')
          if (!existsSync(targetPath)) {
            // 也尝试不加 .md（可能本身就带扩展名）
            targetPath = join(DOCS, cleanPath.slice(1))
          }
        }
      } else {
        // 相对路径
        const cleanPath = linkTarget.split('#')[0]
        const dir = dirname(filePath)
        targetPath = resolve(dir, cleanPath)
        if (!existsSync(targetPath) && !cleanPath.endsWith('.md')) {
          targetPath = resolve(dir, cleanPath + '.md')
        }
      }

      if (!existsSync(targetPath)) {
        error(`${rel}: 死链接 [${match[1]}](${linkTarget})`)
      }
    }
  }

  console.log(`   检查了 ${checked} 个文件中的 ${linkCount} 个内部链接`)
}

// ── 4. CONTENT_MAP 过期检查 ────────────────────────────────

function checkContentMapFreshness() {
  console.log('\n📋 检查 CONTENT_MAP.md 是否过期...')

  try {
    // 重新生成并对比
    const result = execSync('node scripts/generate-content-map.js --dry-run 2>&1 || true', {
      cwd: ROOT,
      encoding: 'utf-8',
    })

    // 简单方法：检查 CONTENT_MAP.md 中的日期
    const contentMap = readFileSync(join(ROOT, 'CONTENT_MAP.md'), 'utf-8')
    const dateMatch = contentMap.match(/Last generated: (\d{4}-\d{2}-\d{2})/)
    if (dateMatch) {
      const lastGen = new Date(dateMatch[1])
      const now = new Date()
      const daysDiff = Math.floor((now - lastGen) / (1000 * 60 * 60 * 24))
      if (daysDiff > 7) {
        warn(`CONTENT_MAP.md 最后生成于 ${dateMatch[1]}（${daysDiff} 天前），建议运行 npm run map 更新`)
      }
    }

    // 检查文件数量是否匹配
    let actualFiles = 0
    for (const mod of MODULE_ORDER) {
      const modDir = join(DOCS, mod)
      if (!existsSync(modDir)) continue
      actualFiles += readdirSync(modDir).filter(f => f.endsWith('.md') && f !== 'index.md').length
    }

    const totalMatch = contentMap.match(/Total: (\d+) content files/)
    if (totalMatch) {
      const indexedFiles = parseInt(totalMatch[1])
      if (indexedFiles !== actualFiles) {
        error(`CONTENT_MAP.md 索引了 ${indexedFiles} 个文件，但实际有 ${actualFiles} 个文件。运行 npm run map 更新`)
      }
    }
  } catch (e) {
    warn(`无法检查 CONTENT_MAP 新鲜度: ${e.message}`)
  }

  console.log('   完成')
}

// ── Main ────────────────────────────────────────────────────

console.log('🔍 LLM-101-CN 内容校验\n' + '='.repeat(40))

checkFrontmatter()
checkSidebarConsistency()
checkInternalLinks()
checkContentMapFreshness()

// 输出结果
console.log('\n' + '='.repeat(40))

if (warnings.length > 0) {
  console.log(`\n⚠️  警告 (${warnings.length}):`)
  for (const w of warnings) console.log(`  ${w}`)
}

if (errors.length > 0) {
  console.log(`\n❌ 错误 (${errors.length}):`)
  for (const e of errors) console.log(`  ${e}`)
  console.log(`\n校验失败，共 ${errors.length} 个错误`)
  process.exit(1)
} else {
  console.log(`\n✅ 全部通过！${warnings.length > 0 ? `（${warnings.length} 个警告）` : ''}`)
  process.exit(0)
}
