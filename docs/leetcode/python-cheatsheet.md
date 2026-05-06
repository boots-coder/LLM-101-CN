---
title: Python 速查 · 刷题高频小细节
description: 切片、enumerate、dict/set、collections、heapq 等容易忘的 Python 习惯用法速查
topics: [python, cheatsheet, leetcode, syntax]
prereqs: [fundamentals/python-ml]
---

# Python 速查 · 刷题高频小细节

> 写算法时，**让代码 Pythonic** 比"会写"更能体现熟练度。这里收录刷 LC 时反复用到、又最容易写错的小细节。每节末尾会有微选择题（构建中）。

## 1. 切片 / Slicing

```python
a = [0, 1, 2, 3, 4, 5]
a[1:4]     # [1, 2, 3]      —— 包左不包右
a[:3]      # [0, 1, 2]
a[3:]      # [3, 4, 5]
a[-2:]     # [4, 5]         —— 后两个
a[::-1]    # [5, 4, 3, 2, 1, 0]   —— 翻转
a[::2]     # [0, 2, 4]      —— 步长 2
a[1:5:2]   # [1, 3]
```

**陷阱**：
- `a[10:100]` **不会越界**，只会得到空列表/已有部分。但 `a[10]` 会越界。
- 切片**生成新对象**（浅拷贝），改 a[1:3] 不会改原 a。
- 字符串切片同理；但**不能切片赋值**到字符串（不可变）。

## 2. enumerate

```python
for i, x in enumerate(arr):           # 默认从 0 开始
    ...
for i, x in enumerate(arr, start=1):  # 显式起点
    ...
```

**陷阱**：用 `for i in range(len(arr))` 是最不 Pythonic 的写法之一，面试时少用。

## 3. dict / set

```python
d = {}              # 空字典
s = set()           # 空集合（{} 是 dict 不是 set！）

# in 查询都是 O(1)
'k' in d            # 查 key
1 in s

# 安全取值
d.get(k, default)   # k 不在时返回 default，不抛 KeyError

# 删除
d.pop(k, None)      # 安全删除
del d[k]            # k 不在会 KeyError

# 遍历
for k, v in d.items(): ...
for k in d: ...           # 等价于 d.keys()

# Python 3.7+ dict 保插入序
list(d.keys())      # 按插入顺序
```

## 4. collections（必背 4 个）

```python
from collections import Counter, defaultdict, deque, OrderedDict

# Counter —— 计数神器
Counter("aabbc")           # {'a': 2, 'b': 2, 'c': 1}
Counter(s) == Counter(t)   # 字母异位词判定一行解决
Counter(arr).most_common(k)  # Top-K 元素

# defaultdict —— 自动初始化
g = defaultdict(list)      # g[新key] 自动给 []
g['a'].append(1)           # 不需要先检查

# deque —— 双端队列，O(1) 头尾插删
q = deque()
q.append(x)        # 右进
q.popleft()        # 左出
q.appendleft(x)    # 左进
q.pop()            # 右出
# BFS 必用！

# OrderedDict —— LRU 缓存常用
od = OrderedDict()
od.move_to_end(k)        # 移动到末尾
od.popitem(last=False)   # 弹出最早的
```

## 5. heapq（最小堆）

```python
import heapq

h = []
heapq.heappush(h, x)     # 入堆
heapq.heappop(h)         # 弹最小
h[0]                     # 看堆顶不弹

heapq.heapify(arr)       # 原地建堆 O(n)
heapq.nlargest(k, arr)   # Top-K 大
heapq.nsmallest(k, arr)  # Top-K 小

# 求最大堆？取负！
heapq.heappush(h, -x)
-heapq.heappop(h)
```

## 6. bisect（二分查找）

```python
import bisect
bisect.bisect_left(arr, x)    # x 应插入的位置（左偏）
bisect.bisect_right(arr, x)   # 右偏
bisect.insort(arr, x)         # 二分插入
```

LIS 题用 `bisect_left` + tails 数组可压到 O(n log n)。

## 7. itertools / functools 常用

```python
from itertools import combinations, permutations, product, accumulate
list(combinations([1,2,3], 2))    # [(1,2),(1,3),(2,3)]
list(permutations([1,2,3], 2))    # 全排列
list(product([0,1], repeat=3))    # 笛卡尔积，回溯模板
list(accumulate([1,2,3,4]))       # 前缀和：[1,3,6,10]

from functools import lru_cache, reduce
@lru_cache(maxsize=None)          # 记忆化 DP 一行
def f(i, j): ...
```

## 8. 字符串常见操作

```python
s.lower() / s.upper() / s.strip()
s.split(',')             # 按逗号切
','.join(parts)          # 反向连接
s.startswith(p)          # 前缀判断
s.find(sub)              # 找不到返回 -1（不抛异常）
sorted(s)                # 字符列表，可作字母异位词 key
ord('a'), chr(97)        # 字符 ↔ 整数
```

## 9. 列表推导与生成器

```python
[x*2 for x in arr if x > 0]              # 列表推导
{x: i for i, x in enumerate(arr)}        # 字典推导
{x for x in arr}                         # 集合推导
sum(x*x for x in arr)                    # 生成器（不占内存）
```

## 10. 常踩的小坑

| 坑 | 说明 |
|---|---|
| `arr.sort()` 返回 `None` | 用 `sorted(arr)` 才有返回值 |
| `int / int` 默认是 `float` | 需要整除写 `//` |
| `nums[i] = nums[j]` 不会复制对象 | 改可变对象时仍指向同一个 |
| `dict` 在迭代中**不能改大小** | 用 `list(d.keys())` 拷贝再遍历 |
| `{}` 是 dict，`set()` 才是空集合 | 高频考点 |
| `is` vs `==` | `is` 比身份（id），`==` 比值；除 `is None` 外都用 `==` |

::: info 微选择题正在准备中
本页的每节之后都会挂一组 `<LcLesson>` 微选择题（如"`heapq.heappush(h, x)` 之后，`h[0]` 一定是最小元素吗？"），帮你把这些细节练成肌肉记忆。
:::
