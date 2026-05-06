const KEYWORDS = new Set([
  'def', 'return', 'if', 'elif', 'else', 'for', 'while', 'in', 'not', 'and', 'or',
  'True', 'False', 'None', 'class', 'import', 'from', 'as', 'with', 'try', 'except',
  'finally', 'raise', 'pass', 'break', 'continue', 'yield', 'lambda', 'global',
  'nonlocal', 'is', 'del', 'assert', 'async', 'await',
])

const BUILTINS = new Set([
  'len', 'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed', 'sum',
  'min', 'max', 'abs', 'int', 'str', 'float', 'list', 'dict', 'set', 'tuple', 'bool',
  'print', 'input', 'type', 'isinstance', 'any', 'all', 'iter', 'next', 'open',
  'object', 'property', 'staticmethod', 'classmethod', 'self', 'cls',
  'deque', 'defaultdict', 'Counter', 'OrderedDict', 'heapq', 'bisect',
])

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

export function highlightPythonLine(line: string): string {
  // Token order matters: comment > string > number > word > operator
  const tokenRe = /(#[^\n]*)|([rRbBfFuU]?(?:"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))|(\b(?:0x[0-9a-fA-F]+|\d+(?:\.\d+)?)\b)|([A-Za-z_][A-Za-z0-9_]*)|([+\-*/%=<>!&|^~]+)/g

  let out = ''
  let last = 0
  let m: RegExpExecArray | null
  while ((m = tokenRe.exec(line)) !== null) {
    if (m.index > last) out += escapeHtml(line.slice(last, m.index))
    if (m[1] !== undefined) {
      out += `<span class="hl-com">${escapeHtml(m[1])}</span>`
    } else if (m[2] !== undefined) {
      out += `<span class="hl-str">${escapeHtml(m[2])}</span>`
    } else if (m[3] !== undefined) {
      out += `<span class="hl-num">${escapeHtml(m[3])}</span>`
    } else if (m[4] !== undefined) {
      const w = m[4]
      if (KEYWORDS.has(w)) out += `<span class="hl-kw">${w}</span>`
      else if (BUILTINS.has(w)) out += `<span class="hl-bi">${w}</span>`
      else out += escapeHtml(w)
    } else if (m[5] !== undefined) {
      out += `<span class="hl-op">${escapeHtml(m[5])}</span>`
    }
    last = tokenRe.lastIndex
  }
  if (last < line.length) out += escapeHtml(line.slice(last))
  return out
}

export function highlightPython(code: string): string {
  return code.split('\n').map(highlightPythonLine).join('\n')
}

export function highlightInlineSnippet(s: string): string {
  // For inline `code` segments inside prompts/explanations.
  return highlightPythonLine(s)
}
