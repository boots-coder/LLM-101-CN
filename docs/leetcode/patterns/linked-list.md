---
title: 链表 · LeetCode Pattern
description: 哨兵节点 + 双指针，避开「头节点丢失」陷阱
topics: [leetcode, linked-list, pattern]
prereqs: [leetcode/]
---

# 链表 · Linked List

> **一句话骨架**：90% 链表题靠两件事——「哨兵 dummy」处理头节点变更，「快慢双指针」找中点 / 找环 / 找倒数第 K 个。

## 通用技巧

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

dummy = ListNode(0, head)   # 哨兵：让「删头」「插头」与一般节点同质
prev, curr = dummy, head
```

三把板斧：

| 招式 | 适用场景 | 关键不变量 |
|------|----------|------------|
| dummy 哨兵 | 头节点会变更（删除/插入头部） | 始终有「前驱」可以接 |
| 快慢指针 | 找中点、判环、找倒数第 K | fast 步幅是 slow 的 2 倍 |
| 三指针掉头 | 反转片段 | `prev → curr → nxt` 的暂存顺序不能错 |

## 关卡 1 · 反转链表

::: info 学习目标
能从空白纸默写出 6 行迭代反转，并讲清「为什么必须三个指针」。
:::

<LcLesson problem-id="reverse-linked-list" />

## 关卡 2 · 环形链表 II

::: info 学习目标
理解 Floyd 判圈两阶段为什么成立，能推导 a = c + (n-1)L。
:::

<LcLesson problem-id="linked-list-cycle-ii" />

## 关卡 3 · 合并 K 个升序链表

::: info 学习目标
说清「O(N log K) vs O(N log N)」的差别，知道 Python 堆为什么要塞 tiebreaker。
:::

<LcLesson problem-id="merge-k-sorted-lists" />

## 关卡 4 · LRU 缓存

::: info 学习目标
能徒手设计「哈希 + 双向链表」的 O(1) 缓存，并解释为什么单向链表不够。
:::

<LcLesson problem-id="lru-cache" />

## 同 pattern 索引（hot100 同类题）

| 题号 | 题名 | 难度 | 关键变形 |
|------|------|------|----------|
| 160 | 相交链表 | 简单 | 双指针「a→b、b→a」走完终在交点（或 None）相遇 |
| 21 | 合并两个有序链表 | 简单 | dummy + 双指针逐个比；merge-K 的子函数 |
| 2 | 两数相加 | 中等 | 双指针逐位相加 + 进位；注意进位最后一位 |
| 19 | 删除链表的倒数第 N 个结点 | 中等 | 快慢指针，fast 先走 N 步再同速；务必用 dummy 防删头 |
| 24 | 两两交换链表中的节点 | 中等 | dummy + 三指针；递归两行也很优雅 |
| 25 | K 个一组翻转链表 | 困难 | 反转链表的进阶版——分段反转 + 接回 |
| 138 | 随机链表的复制 | 中等 | 哈希「原节点 → 新节点」或「原地穿插再拆」 |
| 148 | 排序链表 | 中等 | 归并排序：快慢找中点 + merge two lists |
| 141 | 环形链表 I | 简单 | Floyd 判圈第一阶段 |
| 234 | 回文链表 | 简单 | 找中点 + 反转后半 + 双指针对比 |

## 面试常踩

- **指针操作顺序错位**：反转链表里漏掉 `nxt = curr.next` 暂存，第二步 `curr.next = prev` 立刻丢失对后续节点的引用。
- **链表对象比较用 `is` 不是 `==`**：节点 val 可能重复，比较「是不是同一对象」必须用 `is`。
- **dummy 不是万能但常常香**：删头 / 在头前插入 / 多链合并 / 链表分组，都用 dummy 让边界统一。
- **Floyd 判圈的速度差为什么是 1**：差为 1 才能保证「相遇 + 第二阶段同速回头」的数学结论成立；3:1 仍能判环但定位不到入口。
- **LRU 节点必须存 key**：淘汰节点时要回头从哈希表 `del key`，不存 key 就找不回去——典型设计陷阱。
