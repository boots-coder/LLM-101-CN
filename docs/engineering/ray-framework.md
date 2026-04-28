---
title: "Ray 分布式框架"
description: "从 Actor 模型到分布式推理与训练，理解 vLLM/verl/OpenRLHF 的底层调度引擎"
topics: [ray, actor-model, distributed-inference, distributed-training, ray-train, vllm, verl, openrlhf, single-controller, weight-sync, hybrid-engine]
prereqs: [engineering/distributed]
---
# Ray 分布式框架

::: info 一句话总结
Ray 是一个通用的分布式计算框架，通过 **Task**（无状态远程函数）和 **Actor**（有状态远程对象）两个核心原语，让 Python 程序员用最少的代码改动就能将单机程序扩展到集群。vLLM 用它做多卡推理调度，verl 用它编排 PPO 的 rollout 与 training 流程。
:::

## 在大模型体系中的位置

```
┌─────────────────────────────────────────────────────────────┐
│                LLM Engineering Stack                         │
│                                                              │
│  Model Design → Distributed Training → Inference → Serving   │
│                      ↑                     ↑                 │
│                  Ray Train             Ray Serve              │
│                      ↑                     ↑                 │
│                ┌──────────────────────────────┐              │
│                │   Ray Core (Actor / Task)    │  ← 你在这里   │
│                └──────────────────────────────┘              │
│                                                              │
│  Ray 是底层调度引擎，vLLM、verl、DeepSpeed-Chat 等           │
│  上层框架都依赖它进行分布式资源管理与任务编排                   │
└─────────────────────────────────────────────────────────────┘
```

Ray 解决的核心问题：**如何把多台机器、多张 GPU 当做一个统一的计算资源池来使用？**

- 对训练来说：编排 generator（推理采样）+ trainer（梯度更新）的异步协作
- 对推理来说：调度多 GPU tensor parallel，管理请求队列和模型副本
- 对 RLHF/PPO 来说：协调 rollout、reward、training 三个阶段的数据流转

---

## 1. Ray 核心概念

### 1.1 四大组件

| 组件 | 作用 | 类比 |
|------|------|------|
| **Task** | 无状态远程函数，`@ray.remote` 装饰后 `.remote()` 调用 | 分布式的函数调用 |
| **Actor** | 有状态远程对象，`@ray.remote` 装饰类 | 分布式的 Python 对象 |
| **Object Store** | 共享内存，存储 Task/Actor 的输入输出 | 分布式的 Redis |
| **GCS (Global Control Store)** | 全局元数据服务，管理节点、Actor、资源信息 | 集群的"大脑" |

### 1.2 执行模型

```
Driver (你的主程序)
  │
  ├── ray.init()              # 连接/启动集群
  │
  ├── func.remote(args)       # 提交 Task → Worker 执行
  │     └── 返回 ObjectRef    # 未来值的引用（Future）
  │
  ├── Actor.remote()          # 创建 Actor → 常驻 Worker
  │     └── actor.method.remote()  # 调用 Actor 方法
  │
  └── ray.get(ref)            # 阻塞获取结果
```

::: tip 关键设计
Ray 的 `.remote()` 调用是**非阻塞**的，返回一个 `ObjectRef`（Future）。只有当你调用 `ray.get()` 时才会阻塞等待结果。这使得自然地表达异步并行成为可能。
:::

### 1.3 为什么 vLLM 和 verl 都选择 Ray？

1. **Python 原生**：不需要学习新的 DSL，`@ray.remote` 一行装饰器即可
2. **异构资源管理**：可以精确指定 `num_gpus=0.5`，多个 Actor 共享一张 GPU
3. **Actor 模型天然适配 LLM 服务**：每个模型副本就是一个 Actor，有状态、可通信
4. **动态调度**：不需要静态拓扑，可以运行时创建/销毁 Actor
5. **生态成熟**：Ray Serve（推理服务）、Ray Train（分布式训练）、Ray Data（数据处理）

---

## 2. Actor 编程模型

### 2.1 远程函数（Task）

最简单的 Ray 程序——将计算提交到远程 Worker：

```python
import ray
import torch

ray.init()

@ray.remote
def run_task(tensor, device):
    """远程函数：在 Worker 上执行矩阵运算"""
    tensor = tensor.to(device)
    norm_val = tensor.norm()        # L2 范数
    mean_val = tensor.mean()        # 均值
    return norm_val.cpu(), mean_val.cpu()

# 创建数据：5 个 4x4 矩阵
batch = torch.randn(5, 4, 4, device='cpu')

# 并行提交 5 个任务
futures = [run_task.remote(batch[i], 'cpu') for i in range(5)]

# 阻塞获取所有结果
results = ray.get(futures)
for i, (nv, mv) in enumerate(results):
    print(f"矩阵{i+1}: 范数={nv:.4f}, 均值={mv:.4f}")
```

**指定 GPU 资源**：

```python
@ray.remote(num_gpus=0.5)  # 每个 Task 占用半张 GPU
def compute_on_gpu(tensor):
    print('remote device:', tensor.device)
    return tensor.norm(), tensor.mean()
```

::: warning num_gpus 的含义
`num_gpus=0.5` 不是说只用半张 GPU 的算力——它是**资源声明**，告诉 Ray 调度器"我需要 0.5 个 GPU 资源单位"。这意味着一张 GPU 上最多调度 2 个这样的 Task/Actor。实际的显存隔离需要用户自己管理。
:::

### 2.2 Actor（有状态远程对象）

Actor 是 Ray 最强大的抽象——将一个 Python 类变成远程服务：

```python
@ray.remote
class Producer:
    def __init__(self):
        self.payload = [7, 21, 55, 3, 42, 16]

    def push(self, consumer, value):
        # 调用另一个 Actor 的方法——Actor 间通信
        return consumer.pull.remote(value)

    def get_payload(self):
        return self.payload

@ray.remote
class Consumer:
    def __init__(self):
        self.buffer = []

    def pull(self, value):
        self.buffer.append(value)
        return len(self.buffer)

    def get_buffer(self):
        return self.buffer

# 创建 Actor 实例（运行在远程 Worker 上）
producer = Producer.remote()
consumer = Consumer.remote()

# Actor 间通信：Producer → Consumer
payload = ray.get(producer.get_payload.remote())
for val in payload:
    future = ray.get(producer.push.remote(consumer, val))
    count = ray.get(future)
    print(f'Consumer 已接收数据量: {count}')
```

### 2.3 异步执行模式

Ray 天然支持异步：发射任务后不等待，继续执行其他代码：

```python
import ray
import torch
import time

ray.init()

@ray.remote
def matmul_task(x, y, iters=5):
    out = x
    for _ in range(iters):
        out = out @ y
        time.sleep(0.05)
    return out

@ray.remote
def elementwise_task(x, y, iters=5):
    out = x.clone()
    for _ in range(iters):
        out = out + y
        time.sleep(0.05)
    return out

X = torch.randn(512, 512)
Y = torch.randn(512, 512)

# 同步：阻塞直到结果返回
sync_result = ray.get(matmul_task.remote(X, Y))

# 异步：先发射两个任务，再统一等待
ref_a = matmul_task.remote(X, Y)
ref_b = elementwise_task.remote(X, Y)
result_a, result_b = ray.get([ref_a, ref_b])
```

::: tip 同步 vs 异步的关键区别
- `ray.get(func.remote(x))` → **同步**：提交后立即等待结果
- `ref = func.remote(x)` → **异步**：提交后继续执行，稍后 `ray.get(ref)` 获取结果
- 善用异步是 Ray 性能优化的关键
:::

---

## 3. 分布式通信模式

在大模型系统中，组件之间需要频繁通信。Ray 提供了多种通信模式。

### 3.1 Actor 间直接通信

最直观的方式——一个 Actor 直接调用另一个 Actor 的方法：

```python
# Producer 批量异步发送，Consumer 同步接收
@ray.remote
class Producer:
    def __init__(self, items):
        self.items = items

    def push_all(self, consumer):
        refs = []
        for val in self.items:
            ref = consumer.pull.remote(val)
            refs.append(ref)
        return refs

@ray.remote
class Consumer:
    def __init__(self):
        self.buffer = []

    def pull(self, value):
        self.buffer.append(value)
        return len(self.buffer)

    def get_buffer(self):
        return self.buffer
```

::: details 为什么异步发送但接收顺序不乱？
Ray 的 Actor 模型保证：**同一个 Actor 的方法调用是顺序执行的**（串行化）。即使发送方异步发出多个请求，接收方 Actor 内部依然按到达顺序逐个处理。这是 Actor Model 的核心保证——无需加锁，无需考虑竞态条件。
:::

### 3.2 多 Actor 并发通信

当多个 Sender 同时向一个 Receiver 发送数据时：

```python
producer_a = Producer.remote([100, 200, 300, 400])
producer_b = Producer.remote([51, 62, 73, 84, 95])
consumer = Consumer.remote()

# 两个 Producer 并行发送
refs_a = ray.get(producer_a.push_all.remote(consumer))
refs_b = ray.get(producer_b.push_all.remote(consumer))

# 合并等待
ray.get(refs_a + refs_b)
received = ray.get(consumer.get_buffer.remote())
print('接收到的数据:', received)
# 可能输出: [100, 51, 200, 62, 300, 73, 400, 84, 95]
# 两个 Producer 的数据交错到达，但每个 Producer 内部有序
```

### 3.3 Tensor 传输与共享对象

大模型场景中经常需要传输 GPU tensor。Ray 的 Object Store 支持零拷贝共享：

```python
import ray
import torch

ray.init()

# 将 tensor 放入共享内存
tensor_gpu = torch.zeros(4, 4, device='cuda:0')
tensor_ref = ray.put(tensor_gpu)  # 返回 ObjectRef

# 任何 Worker 都可以通过 ref 获取
result = ray.get(tensor_ref)
print(result.device)  # 获取到的 tensor
```

::: warning GPU tensor 传输注意事项
`ray.put()` 会将 GPU tensor 拷贝到 CPU 共享内存中。`ray.get()` 获取时得到的是 CPU tensor。如果需要在 GPU 上使用，需要手动 `.to(device)`。对于大型模型参数同步，建议使用 NCCL 集合通信而非 Ray Object Store。
:::

### 3.4 Actor Group + 数据切片下发模式

朴素的"共享队列"在 RLHF 训练里有个致命问题：**rollout 数据是大对象**（一批 prompt + response + logprob + values），如果用一个 Actor 当队列让所有训练 Worker 来 pop，这个 Actor 会成为带宽瓶颈。

OpenRLHF 给出的范式（参考 [openrlhf/trainer/ray/launcher.py](https://github.com/OpenRLHF/OpenRLHF) 的 `RayActorGroup.async_run_method_batch`）是 **"Driver 端预切片 → ray.put → 各 Worker 只拉自己那份"**：

```python
import ray

class RayActorGroup:
    """一组同构 ray actor 的代理对象"""

    def __init__(self, actor_handlers):
        self._actor_handlers = actor_handlers

    def async_run_method_batch(self, method_name, **kwargs):
        """把大 batch 切片后下发到各 actor，每个 actor 只看到自己的 chunk"""
        total_length = len(next(iter(kwargs.values())))
        num_actors   = len(self._actor_handlers)
        chunk_size   = (total_length + num_actors - 1) // num_actors

        refs = []
        for i, actor in enumerate(self._actor_handlers):
            start, end = i * chunk_size, min((i + 1) * chunk_size, total_length)

            # 关键：先切片，再 ray.put —— 每个 chunk 只在 object store 存一份
            # 而不是把整个 batch 推给所有 actor
            chunk = {k: v[start:end] for k, v in kwargs.items()}
            chunk_ref = ray.put(chunk)

            refs.append(actor.execute_batch.remote(method_name, chunk_ref, 0, end - start))
        return refs

# 使用方式
group = RayActorGroup(actor_handlers=[actor_0, actor_1, actor_2, actor_3])

# 一次性把 1024 条经验样本"切 4 份分发"，每个 actor 只收到 256 条
refs = group.async_run_method_batch(
    "compute_log_prob",
    sequences=experience_sequences,   # len = 1024
    attention_masks=attention_masks,  # len = 1024
)
results = ray.get(refs)  # [chunk0_result, chunk1_result, chunk2_result, chunk3_result]
```

**关键设计**（OpenRLHF 在大规模训练里学到的经验）：

| 朴素做法 | 上述做法 |
|---------|---------|
| `ray.put(big_batch)` 一次，所有 actor `ray.get()` 完整 batch | 切成 N 份，每个 chunk 独立 `ray.put` |
| 每个 actor 节点都要拉整个 batch（N×带宽） | 每个 actor 只拉自己那份（1×带宽） |
| 队列变成瓶颈 | 完全无锁、无队列 |

::: tip 对应 RLHF 数据流
RLHF 中的"生成 rollout → 算 reward → 算 advantage → PPO 更新"，每一步都涉及大 batch 在 worker 间流转。OpenRLHF 把这种"批量切片 + 异步下发"封装成 `RayActorGroup` 的标准方法，整个 PPOTrainer 几乎不需要直接接触 `ray.put` / `ray.get`。
:::

### 3.5 Ray + PyTorch Distributed 集合通信

Ray 负责"启动 N 个 Worker、分配 GPU、保证 Worker 命名解析"，PyTorch Distributed 负责"在已启动的 Worker 之间走 NCCL/Gloo 高速通信"——两者各司其职。OpenRLHF 把这套桥接模式抽象成 `BaseDistributedActor` 基类（参考 [openrlhf/trainer/ray/launcher.py](https://github.com/OpenRLHF/OpenRLHF) `BaseDistributedActor`）：

```python
import os, socket
import ray
import torch.distributed as dist


class BaseDistributedActor:
    """所有需要参与 torch.distributed 的 Ray Actor 的基类
       构造期把 MASTER_ADDR/PORT/WORLD_SIZE/RANK 写进环境变量"""

    def __init__(self, world_size, rank, master_addr, master_port):
        self._world_size = world_size
        self._rank       = rank
        # rank 0 自己挑端口；其他 rank 由 Driver 把 rank 0 的地址传进来
        self._master_addr = master_addr or self._get_node_ip()
        self._master_port = master_port or self._get_free_port()

        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"]  = str(self._world_size)
        os.environ["RANK"]        = str(self._rank)
        # Ray 已经为这个 Actor 设置了 CUDA_VISIBLE_DEVICES，所以本地 rank 永远是 0
        os.environ["LOCAL_RANK"]  = "0"

    @staticmethod
    def _get_node_ip():
        return ray._private.services.get_node_ip_address().strip("[]")

    @staticmethod
    def _get_free_port():
        with socket.socket() as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def get_master_addr_port(self):
        """暴露给 Driver / 其他 rank 用，让大家都连到 rank 0"""
        return self._master_addr, self._master_port


@ray.remote(num_gpus=1)
class TrainWorker(BaseDistributedActor):
    """业务 Worker——继承基类后只关心计算，不关心分布式初始化"""

    def setup(self):
        # 此时环境变量已就绪，直接用 env:// 拉起 process group
        dist.init_process_group(backend="nccl", init_method="env://")
        return dist.get_rank(), dist.get_world_size()

    def all_reduce_test(self, tensor):
        dist.all_reduce(tensor)
        return tensor
```

**Driver 端的两阶段启动**——OpenRLHF `RayActorGroup._initiate_actors()` 的标准做法：

```python
world_size = 4

# 阶段 1：先创建 rank 0，让它自己挑一个空闲端口
master = TrainWorker.remote(world_size, rank=0, master_addr=None, master_port=None)
master_addr, master_port = ray.get(master.get_master_addr_port.remote())

# 阶段 2：把 rank 0 的地址告诉所有 worker rank
workers = [master] + [
    TrainWorker.remote(world_size, rank=r, master_addr=master_addr, master_port=master_port)
    for r in range(1, world_size)
]

# 此时所有 worker 都已知 MASTER_ADDR/PORT，并行 init_process_group
ray.get([w.setup.remote() for w in workers])
```

::: tip 这个范式好在哪
- **职责分离**：基类管"分布式 bootstrap"（环境变量、地址注入），子类管"业务逻辑"（模型、训练）
- **rank 0 自荐端口**：避免端口硬编码（教程里写 `29500` 在多任务并发时会冲突）
- **两阶段启动**：rank 0 先建立、暴露 `master_addr/port`，rank ≥ 1 再启动并连入。这样无论是单机多卡还是跨机训练都用同一套代码
:::

::: tip Placement Group 的作用
当 Worker 之间需要走 NCCL/IB 高速通信时，把它们放在**同一物理节点**能极大降低延迟。OpenRLHF 在 `RayActorGroup` 中默认用 `placement_group(bundles, strategy="PACK")` 让 Ray 优先打包到同节点；如果一台机器装不下，再 fallback 到 `SPREAD`。
:::

---

## 4. Ray 分布式推理

### 4.1 基本思路

用 Ray 调度多 GPU 推理的核心设计：

```
                    ┌─────────────┐
  请求 ──────────→  │  Ray Driver  │
                    └──────┬──────┘
                           │ 调度
              ┌────────────┼────────────┐
              ↓            ↓            ↓
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ GPU 0    │ │ GPU 1    │ │ GPU 2    │
        │ Model    │ │ Model    │ │ Model    │
        │ Replica  │ │ Replica  │ │ Replica  │
        └──────────┘ └──────────┘ └──────────┘
```

### 4.2 用 Ray 实现 All-Reduce

当多个 GPU 上有分布的数据，需要聚合计算（如求均值）：

```python
@ray.remote(num_gpus=0.5)
def reduce_mean(object_refs, device="cuda:0"):
    chunks = [t.to(device) for t in ray.get(object_refs)]
    stacked = torch.stack(chunks, dim=0)
    return stacked.mean(dim=0)

shards = [torch.randn(8, 16) for _ in range(4)]
refs = [ray.put(s) for s in shards]

mean_tensor = ray.get(reduce_mean.remote(refs))
```

### 4.3 vLLM 中的 Ray 使用

vLLM 使用 Ray 来实现多 GPU tensor parallel 推理：

```
vLLM Engine
  │
  ├── RayGPUExecutor
  │     ├── 创建 N 个 Worker Actor（每个占 1 GPU）
  │     ├── 初始化 NCCL 通信组
  │     └── 调度 execute_model() 到所有 Worker
  │
  └── Worker Actor
        ├── 持有模型的一个 TP shard
        ├── 通过 NCCL 执行 tensor parallel 通信
        └── 执行前向推理
```

关键代码路径（vLLM 源码参考）：
- `vllm/executor/ray_gpu_executor.py` — Ray Worker 的创建和管理
- `vllm/worker/worker.py` — Worker Actor 的实现
- 每个 Worker 持有模型的 1/N 参数（tensor parallel），通过 NCCL all-reduce 同步

::: details 为什么 vLLM 不直接用 torchrun？
`torchrun` 适合静态的训练场景——启动 N 个进程，跑完就结束。但推理服务需要：
1. **动态伸缩**：根据负载增减模型副本
2. **细粒度资源管理**：0.5 GPU 粒度的分配
3. **异构调度**：不同模型可以用不同数量的 GPU
4. **故障恢复**：单个 Worker 挂了不影响整个服务

Ray 的 Actor 模型天然满足这些需求。
:::

---

## 5. verl 的 Single-Controller + Multi-Worker 架构

RLHF 训练同时涉及**推理采样（rollout）**、**奖励打分（reward）**、**参考策略（reference）**、**价值估计（critic）**、**策略更新（actor）**五种角色。如果每个角色各开一个独立的 Ray 程序，编排逻辑会爆炸。verl（Volcano Engine RL）的解法是 **Single-Controller + Multi-Worker**：所有的角色编排都在一个 Driver 进程里写成单机风格代码，分布式通信由框架自动处理。

> 本节代码样例参考自 [verl 仓库](https://github.com/volcengine/verl) `single_controller/ray/base.py` 与 `trainer/ppo/ray_trainer.py`，做了精简以聚焦核心思想。

### 5.1 设计哲学：把分布式藏在装饰器里

```
┌──────────────────────────────────────────────────────────────┐
│   Driver / Single Controller (你的主程序)                      │
│                                                                │
│     for batch in dataloader:                                  │
│         output = actor_rollout_wg.generate_sequences(batch)   │
│         logp   = actor_rollout_wg.compute_log_prob(output)    │
│         values = critic_wg.compute_values(output)             │
│         rewards= rm_wg.compute_rm_score(output)               │
│         actor_rollout_wg.update_actor(batch)                  │
│                              ↑                                 │
│   看起来是单机调用，背后 Dispatch 装饰器自动 split / gather    │
└──────────────────────────────────────────────────────────────┘
                              │
   ┌──────────────────────────┼──────────────────────────┐
   ↓                          ↓                          ↓
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│ WorkerGroup A│         │ WorkerGroup B│         │ WorkerGroup C│
│ (Actor +     │         │ (Critic)     │         │ (Reward)     │
│  Rollout +   │         │              │         │              │
│  Reference)  │         │              │         │              │
│  N×GPU       │         │  M×GPU       │         │  K×GPU       │
└──────────────┘         └──────────────┘         └──────────────┘
```

三个核心抽象（`verl/single_controller/ray/base.py`）：

| 抽象 | 角色 | 类比 |
|------|------|------|
| **`RayResourcePool`** | 把一组 GPU 打包成"资源池"，底层是 Placement Group | K8s 的 Node Pool |
| **`RayClassWithInitArgs`** | 延迟实例化的 Worker 模板 | `functools.partial` |
| **`RayWorkerGroup`** | 在资源池上启动一组同构 Worker，并暴露聚合方法 | torch DDP 的 `nn.parallel.DistributedDataParallel` |

### 5.2 启动一个 Worker Group

```python
from verl.single_controller.ray import (
    RayResourcePool,
    RayClassWithInitArgs,
    RayWorkerGroup,
)

# 1. 声明资源：单节点 8 卡
resource_pool = RayResourcePool(
    process_on_nodes=[8],
    use_gpu=True,
)

# 2. 包装 Worker 类（不立即创建，只记录"怎么创建"）
ray_actor_cls = RayClassWithInitArgs(
    cls=ray.remote(ActorRolloutRefWorker),  # 用户自定义 Worker 类
    config=actor_config,
)

# 3. 在资源池上启动整个 Worker Group
actor_rollout_wg = RayWorkerGroup(
    resource_pool=resource_pool,
    ray_cls_with_init=ray_actor_cls,
)

# 4. 直接调用——背后 Worker Group 会把方法分发到所有 Worker 上执行
result = actor_rollout_wg.generate_sequences(batch)
```

整个过程没有显式 `for worker in workers: worker.method.remote()` 这种循环，因为 **`RayWorkerGroup` 在初始化时已经为 Worker 类的所有 `@register` 装饰过的方法生成了代理**。

### 5.3 Dispatch 装饰器：把通信策略写进函数签名

`verl/single_controller/base/decorator.py` 提供的 `@register` 装饰器声明数据如何在 Driver 与 Worker 之间分发与汇聚：

```python
from verl.single_controller.base.decorator import register, Dispatch, Execute

class ActorRolloutRefWorker(Worker):

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # Driver 端调用 wg.generate_sequences(big_batch) 时：
        # 1. dispatch_fn 自动按 batch 维切成 N 份发给各 Worker
        # 2. 每个 Worker 在自己的 GPU 上执行此方法
        # 3. collect_fn 自动把 N 个返回值拼回成完整 batch
        return self.rollout.generate(prompts)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto) -> DataProto:
        return self.actor.compute_log_prob(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def save_checkpoint(self, path: str):
        # 仅 rank 0 执行，但保存路径需要广播给所有 Worker
        self.actor.save(path)
```

常用 Dispatch 模式：

| 模式 | 行为 | 典型用途 |
|------|------|---------|
| `DP_COMPUTE_PROTO` | 按 batch 切片 → 各 rank 并行 → 拼回 | 数据并行的前向/反向 |
| `ONE_TO_ALL` | 同一份输入广播到所有 rank | 配置下发、checkpoint 路径 |
| `ALL_TO_ALL` | 所有 rank 都拿到完整输入 | 通信原语包装 |
| `RANK_ZERO` | 仅 rank 0 执行 | I/O、日志、保存 |

::: tip 这是 verl 设计上最优雅的一笔
"分布式通信策略"原本是分散在调用代码里的散落细节（每个调用点都要决定怎么 split / gather），verl 把它**内聚到 Worker 方法的装饰器里**。一旦写好 Worker，Driver 端就只需要写单机风格代码。
:::

### 5.4 Hybrid Engine：让多个角色挤进同一个 Worker

朴素做法是给每个角色（Actor 训练、Rollout 推理、Reference 算 logp）开独立的 Worker Group——每张卡只承担一种角色。代价：

- 同一份模型权重在不同 Worker Group 里要存多份
- Rollout → Reference / Actor 之间要通过网络传 batch

verl 用 `create_colocated_worker_cls`（`verl/single_controller/ray/base.py`）把多个角色合并到同一个 Worker 内：

```python
# 同一个 Worker 进程内部维护一个角色字典
class WorkerDict(Worker):
    def __init__(self):
        # 三个角色共享同一张 GPU，权重在进程内可零拷贝复用
        self.worker_dict = {
            "actor":   ActorWorker(config_actor),
            "rollout": RolloutWorker(config_rollout),    # vLLM
            "ref":     ReferenceWorker(config_ref),
        }

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts):
        return self.worker_dict["rollout"].generate(prompts)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, batch):
        return self.worker_dict["actor"].update(batch)
```

效果：原本需要 3N 张 GPU 的 Actor + Rollout + Reference 现在只需要 N 张。代价是这张 GPU 必须在不同阶段切换显存形态——训练阶段卸载 KV cache，推理阶段卸载 optimizer state。

### 5.5 PPO 主循环：Driver 视角的五步

`verl/trainer/ppo/ray_trainer.py` 中的 `RayPPOTrainer.fit()` 把 PPO 迭代写得像单机代码：

```python
def fit(self):
    for batch in self.train_dataloader:
        # ─── Step 1: Rollout 生成 response（vLLM 推理）
        gen_batch = self.actor_rollout_wg.generate_sequences(batch)

        # ─── Step 2: 计算 old log_prob（PPO 的策略锚点）
        old_log_probs = self.actor_rollout_wg.compute_log_prob(gen_batch)
        ref_log_probs = self.actor_rollout_wg.compute_ref_log_prob(gen_batch)

        # ─── Step 3: 价值估计 + 奖励打分
        values  = self.critic_wg.compute_values(gen_batch)
        rewards = self.rm_wg.compute_rm_score(gen_batch)

        # ─── Step 4: 在 Driver CPU 上算 GAE（轻量，不需要分布式）
        batch = compute_advantage(
            batch, values, rewards, ref_log_probs,
            adv_estimator="gae",
        )

        # ─── Step 5: 各 Worker Group 各自 PPO 更新
        actor_metrics  = self.actor_rollout_wg.update_actor(batch)
        critic_metrics = self.critic_wg.update_critic(batch)

        self.log(actor_metrics, critic_metrics)
```

每一行调用形式都是 `worker_group.method(data)`——读起来像单机调用。背后 Dispatch 装饰器自动完成"广播 / 切分 / 收集"，这就是"single controller"的优雅之处。

::: tip 把握重点
- **Resource Pool**：声明式分配 GPU，不需要手算 rank
- **Worker Group**：一组同构 Worker 的代理对象，调用方法即广播
- **Dispatch 装饰器**：通信策略写在 Worker 类定义里，调用端无感
- **Hybrid Engine**：通过角色并置榨干每张 GPU
:::

---

## 6. 在 vLLM 和 verl 中的应用

### 6.1 vLLM：多卡 Tensor Parallel 调度

```python
# vLLM 简化版工作流程
import ray
from vllm import LLM

# vLLM 内部会自动使用 Ray
llm = LLM(
    model="meta-llama/Llama-3-8B",
    tensor_parallel_size=4,  # 4 卡 tensor parallel
    # vLLM 自动创建 4 个 Ray Worker Actor
)

# 每次推理请求：
# 1. Engine 将 prompt 放入调度队列
# 2. Scheduler 选择一批请求
# 3. 通过 Ray 调用所有 Worker 的 execute_model()
# 4. 各 Worker 通过 NCCL 做 tensor parallel 通信
# 5. 收集结果返回
```

**vLLM + Ray 的关键设计**：
- Worker 之间通过 **NCCL** 通信（不是 Ray Object Store），保证高带宽
- Ray 负责 **生命周期管理**（创建、销毁、故障检测）
- 支持 **多模型**共存：不同模型用不同的 Actor Group

### 6.2 OpenRLHF：vLLM 权重热更新的两条通路

PPO/GRPO 训练每走一步，actor 模型的权重就变了一次——而推理用的 vLLM 引擎里**还存着上一步的旧权重**。怎么把新权重快速同步过去？OpenRLHF（参考 [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)）给出了两条通路，对应不同的部署形态。

**架构概览**：

```
┌──────────────────┐       ┌──────────────────────────┐
│  Actor Trainer   │       │  vLLM Rollout Engines     │
│  (训练 Worker)    │       │  (RolloutRayActor × N)    │
│                  │       │                            │
│  每步训练后 ──→  │       │   接收新权重               │
│  调用 broadcast  │ ════→ │   load_weights() 应用      │
│  _to_vllm()      │       │                            │
└──────────────────┘       └──────────────────────────┘
        │                             │
        └────── 通路 A：NCCL ─────────┘  (异机/异 GPU)
        └────── 通路 B：CUDA IPC ─────┘  (同 GPU colocate)
```

#### 通路 A：NCCL Broadcast（disaggregate 部署）

当训练 Worker 与 vLLM 引擎在不同 GPU 上时：

```python
# openrlhf/trainer/ray/ppo_actor.py 简化逻辑

def _init_vllm_sync_group(self):
    """跨 actor train worker + vllm engines 建立 torch process group"""
    # world_size = vLLM 引擎数 × tp_size + 1（actor rank 0）
    world_size = self.vllm_num_engines * self.vllm_tp_size + 1
    refs = [
        engine.init_process_group.remote(
            master_address=self.master_addr,
            master_port=self.master_port,
            rank_offset=i * self.vllm_tp_size + 1,
            world_size=world_size,
            backend="nccl",
        )
        for i, engine in enumerate(self.vllm_engines)
    ]
    ray.get(refs)

def broadcast_to_vllm(self):
    """逐参数 NCCL 广播到所有 vLLM worker"""
    for name, param in self.actor_model.named_parameters():
        # actor rank 0 触发广播
        for engine in self.vllm_engines:
            engine.update_weight.remote(name, param.dtype, param.shape)
        # NCCL broadcast，vLLM 内部 WorkerWrap 接收
        torch.distributed.broadcast(param.data, src=0, group=self.sync_group)
```

vLLM 那侧（`openrlhf/trainer/ray/vllm_worker_wrap.py`）：

```python
class WorkerWrap:
    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        torch.distributed.broadcast(weight, src=0, group=self._model_update_group)
        # 应用到 vLLM 内部的模型实例
        self.model_runner.model.load_weights(weights=[(name, weight)])
```

#### 通路 B：CUDA IPC（colocate 部署）

当训练 Worker 和 vLLM 引擎共享同一张 GPU 时，可以用 CUDA IPC 实现**零拷贝**权重共享：

```python
# 训练 Worker 端：把 tensor 转成 IPC handle
import torch.multiprocessing.reductions as MPR

def share_tensor(param):
    func, args = MPR.reduce_tensor(param.data)
    return (func, args)  # handle，可序列化跨进程传

# vLLM Worker 端：从 handle 重建 tensor
def rebuild_from_handle(func, args):
    weight = func(*args)  # 重建出的 tensor 与原 tensor 共享显存
    self.model_runner.model.load_weights([(name, weight)])
```

CUDA IPC 比 NCCL broadcast 快得多——前者只是传一个 handle，后者要把整张权重张量真正打过去。

#### 两种通路的选择

| 条件 | 选择 | 原因 |
|------|------|------|
| Actor 和 vLLM 在不同物理 GPU | **NCCL** | 跨 GPU 必须走显式通信 |
| Actor 和 vLLM 共享 GPU（colocate_all） | **CUDA IPC** | 同 GPU 可以零拷贝 |
| 多节点跨机 | **NCCL（走 IB/RoCE）** | 跨机 IPC 不可用 |

OpenRLHF 在 `broadcast_to_vllm()` 中根据部署模式自动选择，用户不需要手动切换。

::: tip 为什么这是 RLHF 工程的重点
RLHF 训练里，**每步都需要权重同步**（这点和普通预训练完全不同）。如果同步耗时是 5 秒、训练步数是 1000，光同步就要 1.4 小时。CUDA IPC + colocate 把这部分压到几乎为零，是 OpenRLHF 在工程上最关键的优化之一。
:::

---

## 苏格拉底时刻

在继续之前，尝试回答这些问题：

1. **Task vs Actor**：什么场景用 Task？什么场景用 Actor？（提示：有无状态？是否需要多次调用？）

2. **顺序保证**：两个不同的 Sender Actor 同时向一个 Receiver Actor 发送数据，数据到达顺序是怎样的？（提示：同一 Actor 内串行，不同 Actor 间无序）

3. **权重同步延迟**：RLHF 中 Rollout 用的 vLLM 引擎里跑的可能是上一步的旧权重。如果同步耗时不可忽略，PPO 上看到的样本就是 off-policy 的。这会带来什么问题？为什么 PPO 仍然能 work？

4. **资源声明**：`num_gpus=0.5` 和 `num_gpus=1` 的实际区别是什么？如果两个 `num_gpus=0.5` 的 Actor 同时运行在一张 GPU 上，显存会冲突吗？

5. **设计权衡**：verl 的 Hybrid Engine（Actor + Rollout + Reference 并置在同一 Worker）和"每个角色独立 Worker Group"分离部署相比有什么优劣？

::: details 参考思路
1. Task 适合**无状态、一次性**的计算（如数据预处理、矩阵运算）；Actor 适合**有状态、长期存在**的服务（如模型推理服务、训练器）。
2. 同一 Sender 内部发送有序（因为 Actor 串行执行），不同 Sender 之间无序（并发执行）。总体结果是两个序列的交错。
3. Rollout 拉到的是"上一步"的权重，相对于当前 actor 是 off-policy。PPO 通过 importance sampling ratio + clipping 校正这个偏差，所以一定程度的 staleness 是被允许的——这也是 OpenRLHF 用 NCCL/CUDA IPC 把权重同步压到极快的工程动机。
4. `num_gpus` 是调度声明，不是物理隔离。两个 0.5 的 Actor 会共享显存空间，如果总显存超出 GPU 容量会 OOM。
5. Hybrid Engine 节省 GPU 数量（N 张而非 3N），且角色之间共享权重无需网络传输；代价是必须精细管理显存形态（训练阶段卸 KV cache，推理阶段卸 optimizer state）。分离部署实现简单但需要 3 倍 GPU 加上权重广播的网络开销。
:::

---

## 面试考点

### Q1: Ray 的 Actor 和 Erlang/Akka 的 Actor 有什么区别？

**参考答案**：Ray Actor 是 Python 原生的，支持直接传递 Python 对象和 NumPy/Torch tensor，有共享对象存储。与 Erlang 不同，Ray 不保证 Actor 间消息的有序性（只保证同一 Actor 内的方法调用有序），也不内置 supervision tree 的容错机制。

### Q2: 为什么 vLLM 用 NCCL 而不用 Ray Object Store 做 tensor parallel 通信？

**参考答案**：Ray Object Store 走的是 CPU 共享内存，GPU tensor 需要先 D2H 拷贝到 CPU，再 H2D 拷贝到目标 GPU。NCCL 支持 GPU 直接通信（通过 NVLink/NVSwitch/IB），带宽可达 300-900 GB/s，而经过 CPU 中转的延迟和带宽都不可接受。Ray 负责 Worker 的生命周期管理，NCCL 负责高性能数据传输，各司其职。

### Q3: verl Hybrid Engine 中 rollout 和 training 共享 GPU 的技术挑战？

**参考答案**：核心挑战是**显存管理**。训练时 model + optimizer + gradient + activation 占大量显存，推理时需要 KV cache。verl 的做法是在两个阶段之间**动态释放和重新分配显存**：训练结束后释放 optimizer state 和 gradient，为推理的 KV cache 腾空间；推理结束后释放 KV cache，重建训练所需的 tensor。

### Q4: verl 的 `@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)` 装饰器到底做了什么？

**参考答案**：装饰器在 Worker 类定义时为方法标注了"分布式调用契约"，`RayWorkerGroup` 初始化时扫描这些标注，自动为每个方法生成 Driver 端代理。代理负责三件事：(1) `dispatch_fn` 把 Driver 传入的完整 batch 按数据并行维度切成 N 份分发给各 Worker；(2) 在所有 Worker 上并行执行原方法；(3) `collect_fn` 把 N 个返回值拼回成完整结果。这样 Driver 端就可以写 `wg.compute_log_prob(big_batch)` 这种单机风格代码，所有的 split / gather 通信被装饰器隐藏。

### Q5: OpenRLHF 为什么要支持 NCCL 和 CUDA IPC 两种权重同步通路？

**参考答案**：取决于训练 Worker 与 vLLM 引擎的部署关系。**异 GPU**（disaggregate）必须走 NCCL，因为跨设备无法共享指针。**同 GPU**（colocate_all）下 CUDA IPC 只需传一个内存 handle，零拷贝、毫秒级，比逐参数广播快 1-2 个数量级。RLHF 每步训练后都要同步一次权重，单步多花几秒就会拖慢整个迭代——所以这条路径的优化在大规模 RLHF 里非常关键。

---

## 推荐资源

| 资源 | 链接 | 说明 |
|------|------|------|
| Ray 官方文档 | [docs.ray.io](https://docs.ray.io/) | 最权威的参考 |
| Ray Architecture Whitepaper | [arxiv.org/abs/1712.05889](https://arxiv.org/abs/1712.05889) | 理解 Ray 的设计动机 |
| vLLM 源码 | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) | 看 `executor/ray_gpu_executor.py` |
| verl 源码 | [github.com/volcengine/verl](https://github.com/volcengine/verl) | 看 `single_controller/ray/base.py` 与 `trainer/ppo/ray_trainer.py` |
| OpenRLHF 源码 | [github.com/OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 看 `trainer/ray/ppo_actor.py` 的 `broadcast_to_vllm()` |
| Ray Train 文档 | [docs.ray.io/en/latest/train](https://docs.ray.io/en/latest/train/train.html) | 分布式训练框架 |
| Actor Model 原论文 | [Hewitt & Baker, 1973](https://en.wikipedia.org/wiki/Actor_model) | 理解 Actor 模型的理论基础 |
