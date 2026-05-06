---
title: Trie 前缀树 · LeetCode Pattern
description: 字符索引的多叉树，前缀匹配 O(L)
topics: [leetcode, trie, pattern]
prereqs: [leetcode/]
---

# Trie 前缀树 · Trie

> **一句话骨架**：字符做边、节点存「是否到此为止是个完整单词」的多叉树。`insert / search / startsWith` 都是 O(L)（L 是单词长度），与字典里词条数无关。

## 极简实现

```python
class Trie:
    def __init__(self):
        self.children = {}
        self.is_end = False

    def insert(self, word):
        node = self
        for ch in word:
            node = node.children.setdefault(ch, Trie())
        node.is_end = True

    def search(self, word):
        node = self._walk(word)
        return node is not None and node.is_end

    def startsWith(self, prefix):
        return self._walk(prefix) is not None

    def _walk(self, s):
        node = self
        for ch in s:
            if ch not in node.children: return None
            node = node.children[ch]
        return node
```

## 关卡 1 · 实现 Trie（前缀树）

::: info 学习目标
背下「children dict + is_end + setdefault」三件套；说清 search 与 startsWith 的差异。
:::

<LcLesson problem-id="implement-trie" />

## 关卡 2 · 单词搜索 II

::: info 学习目标
理解「灌 Trie 让多个单词共享前缀」为什么比逐题搜更快；掌握「占位 # + 出 dfs 还原」的网格回溯三段式。
:::

<LcLesson problem-id="word-search-ii" />

## 同 pattern 索引（hot100 同类题）

| 题号 | 题名 | 难度 | 关键变形 |
|------|------|------|----------|
| 211 | 添加与搜索单词 | 中等 | search 支持通配符 `.` → 把 `_walk` 改写成 DFS，遇到 `.` 时枚举所有 children |
| 648 | 单词替换 | 中等 | 把词根灌进 Trie，扫描时遇到第一个 `is_end` 就替换 |

> hot100 几乎没有更多典型 Trie 题；理解 208/212 即覆盖面试中绝大多数 Trie 考点。

## 面试常踩

- **`is_end` 字段不可省**——区分「prefix 走得通」和「整词正好结束」，缺它就 search 与 startsWith 都同语义。
- **dict vs 26 数组**：dict 通用、稀疏更省；数组（小写英文场景）少一次哈希、缓存友好。两种都要会写。
- **DFS 命中单词后不要 `return`**——可能还有更长的单词以它为前缀（如 "app" 与 "apple"）。
- **`del nxt['$']`** 防重复——多条路径都能拼出同一个单词时，必须命中后立刻抹标记。
