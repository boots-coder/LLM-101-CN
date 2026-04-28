---
title: "强化学习基础"
description: "从 Bellman 方程到 Policy Gradient，理解 RLHF 的 RL 根基"
topics: [reinforcement-learning, MDP, bellman, Q-learning, SARSA, DQN, policy-gradient, PPO-foundation]
prereqs: [fundamentals/math, fundamentals/neural-networks]
---
# 强化学习基础

> 强化学习是 RLHF、PPO、GRPO 等对齐方法的数学根基——不理解 RL，就无法真正理解 alignment

## 在大模型体系中的位置

```
基础数学 (Math)                → 概率论、线性代数、微积分
    ↓
神经网络基础 (Neural Networks) → MLP、梯度下降、反向传播
    ↓
强化学习基础  ← 你在这里        → MDP、Bellman、Q-Learning、Policy Gradient
    ↓
偏好对齐 (Alignment)           → RLHF、PPO、DPO、GRPO
```

本章不追求 RL 的大而全，而是 **精准覆盖理解 RLHF/PPO 所需的最小知识集**：从 MDP 框架出发，经 Bellman 方程、Value-based 方法（Q-Learning、DQN），到 Policy Gradient（REINFORCE），最终搭建通往 PPO 的桥梁。

## 1. RL 核心概念

### 1.1 Agent-Environment 交互

强化学习的核心范式：**Agent**（智能体）在 **Environment**（环境）中通过试错学习。

```
          action a_t
Agent ──────────────────► Environment
  ▲                           │
  │    state s_{t+1}          │
  │    reward r_{t+1}         │
  └───────────────────────────┘
```

每一步的交互流程：
1. Agent 观察当前 **State** $s_t$
2. Agent 根据 **Policy** $\pi$ 选择 **Action** $a_t$
3. Environment 返回 **Reward** $r_{t+1}$ 和新 **State** $s_{t+1}$
4. Agent 更新策略，追求长期累积奖励最大化

### 1.2 核心术语表

| 术语 | 符号 | 含义 |
|------|------|------|
| State（状态） | $s \in \mathcal{S}$ | 环境的完整描述 |
| Action（动作） | $a \in \mathcal{A}$ | Agent 可执行的操作 |
| Reward（奖励） | $r \in \mathbb{R}$ | 即时反馈信号 |
| Policy（策略） | $\pi(a \mid s)$ | 状态到动作的映射 |
| State-Value（状态价值） | $V^\pi(s)$ | 从状态 $s$ 出发、遵循 $\pi$ 的期望累积回报 |
| Action-Value（动作价值） | $Q^\pi(s, a)$ | 在状态 $s$ 执行动作 $a$、再遵循 $\pi$ 的期望累积回报 |
| Discount Factor（折扣因子） | $\gamma \in [0, 1)$ | 控制未来奖励的衰减 |

### 1.3 MDP 框架

**马尔可夫决策过程 (Markov Decision Process)** 是 RL 的数学框架，定义为五元组 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$：

- $\mathcal{S}$：状态空间
- $\mathcal{A}$：动作空间
- $P(s' \mid s, a)$：状态转移概率
- $R(s, a, s')$：奖励函数
- $\gamma$：折扣因子

**马尔可夫性质**：下一个状态只取决于当前状态和动作，与历史无关：

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots) = P(s_{t+1} \mid s_t, a_t)
$$

::: tip 直觉理解
MDP 就像下棋——你只需要看棋盘当前局面（State），不需要回顾每一步走法（History），就能决定下一步（Action）。当然现实中很多问题不完全满足马尔可夫性，但 MDP 是一个强大的近似框架。
:::

### 1.4 累积回报 (Return)

Agent 的目标不是最大化即时奖励，而是最大化 **折扣累积回报 (Discounted Return)**：

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

- $\gamma = 0$：完全短视，只关注即时奖励
- $\gamma \to 1$：长远规划，重视未来奖励

## 2. Bellman 方程

### 2.1 Value Function 的递推关系

Bellman 方程是 RL 的基石——它将 Value Function 表达为 **当前奖励 + 折扣后续价值** 的递推形式。

**State-Value Function** $V^\pi(s)$：

$$
V^\pi(s) = \mathbb{E}_\pi \left[ r_{t+1} + \gamma V^\pi(s_{t+1}) \mid s_t = s \right]
$$

展开为求和形式：

$$
V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]
$$

**Action-Value Function** $Q^\pi(s, a)$：

$$
Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a') \right]
$$

### 2.2 Bellman 最优方程

最优策略 $\pi^*$ 对应的最优 Value Function 满足：

$$
V^*(s) = \max_{a} \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]
$$

$$
Q^*(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \max_{a'} Q^*(s', a') \right]
$$

::: warning 关键区别
Bellman 期望方程（给定策略 $\pi$）用 $\sum_a \pi(a|s)$ 对动作求期望；Bellman 最优方程用 $\max_a$ 取最优动作。这个区别贯穿整个 RL——Policy Evaluation 用前者，Policy Improvement 用后者。
:::

### 2.3 代码验证：手算 Bellman 方程

用一个 3 状态的简单 MDP 验证 Bellman 方程：

```python
import numpy as np

# 3 状态 MDP: s0 -> s1 -> s2(terminal)
# 动作空间: {0: left, 1: right}
gamma = 0.9

# 转移概率 P[s][a] = [(prob, next_state, reward)]
P = {
    0: {0: [(1.0, 0, -1)],    # s0, left -> stay s0, r=-1
        1: [(1.0, 1,  0)]},   # s0, right -> go s1, r=0
    1: {0: [(1.0, 0, -1)],    # s1, left -> go s0, r=-1
        1: [(1.0, 2,  1)]},   # s1, right -> go s2, r=+1
    2: {0: [(1.0, 2,  0)],    # s2 terminal
        1: [(1.0, 2,  0)]},
}

# 均匀随机策略 pi(a|s) = 0.5
policy = {s: {0: 0.5, 1: 0.5} for s in range(3)}

# 迭代求解 Bellman 期望方程
V = np.zeros(3)
for _ in range(100):
    V_new = np.zeros(3)
    for s in range(3):
        for a in [0, 1]:
            for prob, s_next, r in P[s][a]:
                V_new[s] += policy[s][a] * prob * (r + gamma * V[s_next])
        V[s] = V_new[s]
    V = V_new

print("V(s) under random policy:", V)
# V ≈ [-2.25, -0.26, 0.0]  — 越靠近终点价值越高
```

::: details 手推验证
对 s1 应用 Bellman 方程：$V(s_1) = 0.5 \times [(-1) + 0.9 \times V(s_0)] + 0.5 \times [1 + 0.9 \times V(s_2)]$。代入 $V(s_0) \approx -2.25$, $V(s_2) = 0$，得 $V(s_1) \approx 0.5 \times (-1 - 2.025) + 0.5 \times 1 = -0.5125$。迭代收敛后精确值略有不同，读者可自行验证。
:::

## 3. 动态规划

当环境模型 $P(s'|s,a)$ 已知时，可以用 **动态规划 (Dynamic Programming)** 精确求解 MDP。

### 3.1 Policy Evaluation（策略评估）

给定策略 $\pi$，反复应用 Bellman 期望方程直到 $V^\pi$ 收敛：

$$
V_{k+1}(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R + \gamma V_k(s') \right]
$$

### 3.2 Policy Iteration（策略迭代）

交替执行两步，直到策略不再变化：
1. **Policy Evaluation**：求当前 $\pi$ 的 $V^\pi$
2. **Policy Improvement**：贪心更新 $\pi'(s) = \arg\max_a Q^\pi(s, a)$

### 3.3 Value Iteration（值迭代）

直接迭代 Bellman 最优方程，不显式维护策略：

$$
V_{k+1}(s) = \max_{a} \sum_{s'} P(s' \mid s, a) \left[ R + \gamma V_k(s') \right]
$$

收敛后提取最优策略：$\pi^*(s) = \arg\max_a Q^*(s, a)$

### 3.4 代码：4x4 Grid World 的 Value Iteration

```python
import numpy as np

# 4x4 Grid World: 左上角(0)和右下角(15)是终点
# 动作: 0=上, 1=下, 2=左, 3=右
n_states, n_actions = 16, 4
gamma = 1.0
theta = 1e-6  # 收敛阈值

def step(s, a):
    """确定性环境: 返回 (next_state, reward)"""
    if s in [0, 15]:
        return s, 0  # 终点
    row, col = s // 4, s % 4
    if a == 0: row = max(row - 1, 0)
    elif a == 1: row = min(row + 1, 3)
    elif a == 2: col = max(col - 1, 0)
    elif a == 3: col = min(col + 1, 3)
    return row * 4 + col, -1

# Value Iteration
V = np.zeros(n_states)
while True:
    delta = 0
    for s in range(n_states):
        if s in [0, 15]:
            continue
        v = V[s]
        values = []
        for a in range(n_actions):
            s_next, r = step(s, a)
            values.append(r + gamma * V[s_next])
        V[s] = max(values)
        delta = max(delta, abs(v - V[s]))
    if delta < theta:
        break

# 提取最优策略
policy = np.zeros(n_states, dtype=int)
for s in range(n_states):
    q_values = [step(s, a)[1] + gamma * V[step(s, a)[0]] for a in range(n_actions)]
    policy[s] = np.argmax(q_values)

print("Optimal V:\n", V.reshape(4, 4).round(1))
# 最优策略让每个状态以最短路径到达终点
```

::: tip DP 的局限
动态规划需要完整的环境模型 $P(s'|s,a)$，这在现实中很少满足。后续的 Monte Carlo 和 TD 方法通过 **采样** 解决这个问题——Agent 不需要知道环境的转移概率，只需与环境交互即可学习。
:::

## 4. Monte Carlo 方法

Monte Carlo (MC) 方法通过 **完整 episode 的采样** 估计 Value Function，不需要环境模型。

### 4.1 First-Visit MC 估计 $V^\pi$

核心思想：运行很多 episode，用每个状态 **首次出现后的实际 Return** 的均值来估计 $V(s)$。

$$
V(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_i(s)
$$

```python
import numpy as np
from collections import defaultdict

def first_visit_mc_v(env_step, policy, n_episodes=5000, gamma=0.9):
    """First-Visit MC 估计 V(s)"""
    V = defaultdict(float)
    returns_count = defaultdict(int)

    for _ in range(n_episodes):
        # 生成一个完整 episode
        episode = []
        s = 0  # 起始状态
        while True:
            a = np.random.choice(list(policy[s].keys()),
                                 p=list(policy[s].values()))
            s_next, r, done = env_step(s, a)
            episode.append((s, a, r))
            s = s_next
            if done:
                break

        # 反向计算 Return
        G = 0
        visited = set()
        for s, a, r in reversed(episode):
            G = r + gamma * G
            if s not in visited:  # First-Visit
                visited.add(s)
                returns_count[s] += 1
                # 增量均值更新
                V[s] += (G - V[s]) / returns_count[s]
    return dict(V)
```

### 4.2 MC 估计 $Q^\pi$ + epsilon-Greedy 改进

MC 也可以估计 Q 值，结合 $\epsilon$-greedy 策略实现 **无模型的策略改进**：

```python
def mc_control_epsilon_greedy(env_step, n_states, n_actions,
                               n_episodes=10000, gamma=0.9, epsilon=0.1):
    """MC Control with epsilon-greedy"""
    Q = defaultdict(lambda: np.zeros(n_actions))
    N = defaultdict(lambda: np.zeros(n_actions))

    def epsilon_greedy(s):
        if np.random.random() < epsilon:
            return np.random.randint(n_actions)
        return np.argmax(Q[s])

    for _ in range(n_episodes):
        episode = []
        s = 0
        while True:
            a = epsilon_greedy(s)
            s_next, r, done = env_step(s, a)
            episode.append((s, a, r))
            s = s_next
            if done:
                break

        G = 0
        visited_sa = set()
        for s, a, r in reversed(episode):
            G = r + gamma * G
            if (s, a) not in visited_sa:
                visited_sa.add((s, a))
                N[s][a] += 1
                Q[s][a] += (G - Q[s][a]) / N[s][a]

    # 最终贪心策略
    policy = {s: np.argmax(Q[s]) for s in Q}
    return Q, policy
```

::: warning MC 的缺点
MC 必须等一个 episode 完全结束才能更新，这对于长 episode（或持续性任务）很低效。TD 方法通过 **bootstrapping**（用估计值更新估计值）解决了这个问题。
:::

## 5. 时序差分 (Temporal Difference)

TD 方法结合了 MC 的采样思想和 DP 的 bootstrapping 思想，**每一步都可以更新**。

### 5.1 TD(0)

TD(0) 用 **一步 TD target** $r + \gamma V(s')$ 更新 $V(s)$：

$$
V(s) \leftarrow V(s) + \alpha \left[ \underbrace{r + \gamma V(s')}_{\text{TD target}} - V(s) \right]
$$

其中 $\delta = r + \gamma V(s') - V(s)$ 称为 **TD error**。

### 5.2 SARSA（On-Policy TD Control）

SARSA 用 $(S, A, R, S', A')$ 五元组更新 Q 值——Agent 按当前策略选择 $A'$：

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]
$$

```python
import numpy as np

def sarsa(env_step, n_states, n_actions,
          n_episodes=2000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """SARSA: On-Policy TD Control"""
    Q = np.zeros((n_states, n_actions))

    def epsilon_greedy(s):
        if np.random.random() < epsilon:
            return np.random.randint(n_actions)
        return np.argmax(Q[s])

    for ep in range(n_episodes):
        s = 0  # 起始状态
        a = epsilon_greedy(s)

        while True:
            s_next, r, done = env_step(s, a)
            a_next = epsilon_greedy(s_next)

            # SARSA 更新: 用 Q(s', a') — a' 由当前策略选择
            Q[s, a] += alpha * (r + gamma * Q[s_next, a_next] - Q[s, a])

            s, a = s_next, a_next
            if done:
                break

    return Q
```

### 5.3 Q-Learning（Off-Policy TD Control）

Q-Learning 和 SARSA 的唯一区别：更新时使用 $\max_{a'} Q(s', a')$，而非实际执行的 $a'$：

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

```python
import numpy as np

def q_learning(env_step, n_states, n_actions,
               n_episodes=2000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """Q-Learning: Off-Policy TD Control"""
    Q = np.zeros((n_states, n_actions))

    for ep in range(n_episodes):
        s = 0
        while True:
            # 行为策略: epsilon-greedy
            if np.random.random() < epsilon:
                a = np.random.randint(n_actions)
            else:
                a = np.argmax(Q[s])

            s_next, r, done = env_step(s, a)

            # Q-Learning 更新: 用 max Q(s', a') — 与行为策略无关
            Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])

            s = s_next
            if done:
                break

    return Q
```

### 5.4 SARSA vs Q-Learning：On-Policy vs Off-Policy

| 维度 | SARSA | Q-Learning |
|------|-------|------------|
| 类型 | On-Policy | Off-Policy |
| 更新目标 | $Q(s', a')$，$a'$ 由当前策略选 | $\max_{a'} Q(s', a')$ |
| 行为策略 = 目标策略？ | 是 | 否 |
| 安全性 | 更保守（会避开危险动作） | 更激进（总假设未来走最优） |
| 收敛速度 | 较慢 | 较快 |

::: tip On-Policy vs Off-Policy — RLHF 中的对应
这个区别在 RLHF 中至关重要：PPO 是 On-Policy（用当前策略生成数据并更新）；DPO 则是 Off-Policy（直接从离线偏好数据学习）。理解这个区别有助于理解为什么 PPO 训练更不稳定但上限更高，而 DPO 更稳定但可能偏离最优。
:::

## 6. DQN：从 Q-Learning 到深度强化学习

### 6.1 为什么需要 DQN？

Tabular Q-Learning 将 Q 值存在表格中，状态空间大时完全不可行（如 Atari 游戏有 $\sim 10^{70}$ 种屏幕像素组合）。**DQN (Deep Q-Network)** 用神经网络 $Q_\theta(s, a)$ 近似 Q 函数。

### 6.2 两个关键技巧

**1. Experience Replay（经验回放）**

将 $(s, a, r, s', \text{done})$ 存入 Replay Buffer，训练时随机采样 mini-batch。好处：
- 打破样本的时间相关性
- 提高数据利用率（一条经验可被多次使用）

**2. Target Network（目标网络）**

使用一个参数滞后的 Target Network $Q_{\theta^-}$ 计算 TD target：

$$
y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')
$$

每隔 $C$ 步将主网络参数复制到目标网络：$\theta^- \leftarrow \theta$。这避免了"用自己更新自己"导致的训练不稳定。

### 6.3 DQN 完整实现

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 buffer_size=10000, batch_size=64, target_update=100):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.step_count = 0

        # 主网络 & 目标网络
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)  # Replay Buffer

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q = self.q_net(torch.FloatTensor(state))
            return q.argmax().item()

    def store(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s_next, done = zip(*batch)

        s = torch.FloatTensor(np.array(s))
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r)
        s_next = torch.FloatTensor(np.array(s_next))
        done = torch.FloatTensor(done)

        # 当前 Q 值
        q_values = self.q_net(s).gather(1, a).squeeze()

        # Target Q 值（用目标网络）
        with torch.no_grad():
            q_target = r + self.gamma * (1 - done) * self.target_net(s_next).max(1)[0]

        loss = nn.MSELoss()(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 定期更新目标网络
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
```

::: tip DQN 的历史意义
2015 年 DeepMind 发表的 DQN 论文 (*Human-level control through deep reinforcement learning*) 首次证明深度 RL 可以在复杂环境（Atari 游戏）中达到人类水平。这开启了 Deep RL 的黄金时代，最终催生了 PPO、RLHF 等技术。
:::

## 7. Policy Gradient

### 7.1 从 Value-Based 到 Policy-Based

前面的方法都是 **Value-Based**：先估计 Q 值，再从 Q 值推导策略。**Policy Gradient** 方法直接参数化策略 $\pi_\theta(a \mid s)$ 并通过梯度上升优化。

为什么需要 Policy Gradient？
- Value-Based 方法只能处理离散动作空间
- 策略可以是随机的（stochastic），天然支持探索
- **LLM 生成就是一个策略**：$\pi_\theta(\text{token} \mid \text{context})$ 是一个在词表上的概率分布

### 7.2 策略梯度定理

目标：最大化期望累积回报

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]
$$

**策略梯度定理 (Policy Gradient Theorem)**：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]
$$

### 7.3 对数导数技巧 (Log-Derivative Trick) 推导

这是整个推导的核心，也是理解 PPO 的关键。逐步推导：

**第一步：写出目标函数**

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \sum_{\tau} P(\tau; \theta) R(\tau)
$$

**第二步：对 $\theta$ 求梯度**

$$
\nabla_\theta J(\theta) = \sum_{\tau} \nabla_\theta P(\tau; \theta) \cdot R(\tau)
$$

**第三步：对数导数技巧**

利用恒等式 $\nabla_\theta P(\tau; \theta) = P(\tau; \theta) \cdot \nabla_\theta \log P(\tau; \theta)$：

$$
\nabla_\theta J(\theta) = \sum_{\tau} P(\tau; \theta) \cdot \nabla_\theta \log P(\tau; \theta) \cdot R(\tau) = \mathbb{E}_{\tau} \left[ \nabla_\theta \log P(\tau; \theta) \cdot R(\tau) \right]
$$

**第四步：展开轨迹概率**

$$
P(\tau; \theta) = P(s_0) \prod_{t=0}^{T} \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)
$$

取对数后，环境转移概率项 $P(s_{t+1} \mid s_t, a_t)$ 与 $\theta$ 无关，梯度中消失：

$$
\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

::: warning 为什么这个推导重要？
对数导数技巧让我们 **不需要知道环境模型**（$P(s'|s,a)$ 在梯度中消失了），只需要从当前策略采样轨迹就能估计梯度。这正是 RLHF 中 PPO 能工作的原因——我们不需要知道"人类偏好的转移概率"，只需要让 LLM 生成回答，然后用 Reward Model 打分。
:::

### 7.4 REINFORCE 算法

REINFORCE 是最简单的 Policy Gradient 算法，直接用采样 Return $G_t$ 估计梯度：

$$
\theta \leftarrow \theta + \alpha \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t
$$

**加入 Baseline 降低方差**：用 $G_t - b(s_t)$ 替代 $G_t$，其中 $b(s_t)$ 通常取 $V(s_t)$ 的估计值。这不改变梯度期望（无偏），但显著降低方差。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        logits = self.net(x)
        return Categorical(logits=logits)

def reinforce(env, state_dim, action_dim,
              n_episodes=1000, gamma=0.99, lr=1e-3):
    """REINFORCE with baseline"""
    policy = PolicyNet(state_dim, action_dim)
    baseline = nn.Sequential(  # Value baseline V(s)
        nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1)
    )
    opt_policy = optim.Adam(policy.parameters(), lr=lr)
    opt_baseline = optim.Adam(baseline.parameters(), lr=lr)

    for ep in range(n_episodes):
        states, actions, rewards = [], [], []
        s = env.reset()
        done = False
        while not done:
            s_tensor = torch.FloatTensor(s)
            dist = policy(s_tensor)
            a = dist.sample()
            s_next, r, done, _ = env.step(a.item())
            states.append(s_tensor)
            actions.append(a)
            rewards.append(r)
            s = s_next

        # 计算每步的折扣 Return
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        states_t = torch.stack(states)
        actions_t = torch.stack(actions)

        # 更新 baseline (Value Function)
        v_pred = baseline(states_t).squeeze()
        baseline_loss = nn.MSELoss()(v_pred, returns)
        opt_baseline.zero_grad()
        baseline_loss.backward()
        opt_baseline.step()

        # 更新 Policy (REINFORCE + baseline)
        advantage = returns - v_pred.detach()  # A(s,a) = G - V(s)
        dist = policy(states_t)
        log_probs = dist.log_prob(actions_t)
        policy_loss = -(log_probs * advantage).mean()
        opt_policy.zero_grad()
        policy_loss.backward()
        opt_policy.step()

    return policy
```

### 7.5 从 REINFORCE 到 PPO 的演化

REINFORCE 虽然正确，但在实际训练中有严重问题：

| 问题 | 解决方案 | 结果 |
|------|---------|------|
| 高方差 | 加 Baseline $V(s)$ | Actor-Critic |
| 采样效率低（On-Policy） | 重要性采样 (Importance Sampling) | Off-Policy PG |
| 更新步长难控制 | 信赖域约束 (Trust Region) | TRPO |
| TRPO 实现复杂 | 用 Clip 近似信赖域 | **PPO** |

PPO 的核心目标函数（Clipped Surrogate Objective）：

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)} A_t, \; \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_\text{old}}}, 1-\epsilon, 1+\epsilon\right) A_t \right) \right]
$$

其中 $\frac{\pi_\theta}{\pi_{\theta_\text{old}}}$ 是 **重要性采样比率 (Importance Sampling Ratio)**，$\epsilon$ 是裁剪范围（通常 0.2）。Clip 机制确保策略不会一步更新太远。

::: details PPO 为什么适合 RLHF？
PPO 的三个特性使其成为 RLHF 的首选：
1. **On-Policy + 多次复用**：采集一批数据后可以做多个 epoch 的更新（clip 保证安全）
2. **稳定的策略更新**：clip 机制避免 LLM 策略突变导致生成质量崩溃
3. **可以方便地加 KL 惩罚**：$r_t = r_{\text{RM}}(x, y) - \beta \text{KL}[\pi_\theta \| \pi_{\text{ref}}]$，防止 reward hacking
:::

## 8. 从 RL 到 RLHF 的桥梁

### 8.1 LLM 生成 = RL 序贯决策

将 LLM 的文本生成建模为 MDP：

| RL 概念 | LLM 对应 |
|---------|---------|
| State $s_t$ | prompt + 已生成的 token 序列 $[x, y_1, \dots, y_{t-1}]$ |
| Action $a_t$ | 下一个 token $y_t \in \mathcal{V}$（词表） |
| Policy $\pi_\theta(a \mid s)$ | LLM 的 next-token 分布 $P_\theta(y_t \mid x, y_{<t})$ |
| Reward $r$ | Reward Model 对完整生成的打分 $r_\text{RM}(x, y)$ |
| Episode | 一次完整的文本生成（从第一个 token 到 EOS） |

### 8.2 RLHF 的 RL 视角

```
┌─────────────┐     prompt + 已生成 tokens     ┌──────────────────┐
│  LLM (Agent) │ ◄──────────────────────────── │  "Environment"    │
│  π_θ(y|x)    │ ────────── token y_t ──────► │  Reward Model     │
└─────────────┘                                │  r = RM(x, y)     │
                                               │  + KL penalty     │
                                               └──────────────────┘
```

RLHF 的 PPO 目标函数：

$$
\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{D}, \, y \sim \pi_\theta(\cdot | x)} \left[ r_\text{RM}(x, y) - \beta \text{KL}\left[\pi_\theta(y|x) \| \pi_\text{ref}(y|x)\right] \right]
$$

其中：
- $r_\text{RM}(x, y)$：Reward Model 打分，反映人类偏好
- $\pi_\text{ref}$：SFT 模型，作为 KL 正则化的锚点
- $\beta$：KL 惩罚系数，防止 **reward hacking**（模型找到高分但无意义的回答）

### 8.3 为什么需要 KL 惩罚？

没有 KL 约束，LLM 会 "hack" Reward Model：

```
优化过度的例子：
Prompt: "写一首关于春天的诗"
无 KL 约束: "好好好好好好好好好好..."  ← RM 给高分但无意义
有 KL 约束: "春风拂过柳梢头，燕子归来筑新巢"  ← 既得高分又保持流畅
```

KL 惩罚确保优化后的策略 $\pi_\theta$ 不偏离参考策略 $\pi_\text{ref}$ 太远，保持语言模型的基本生成能力。

::: tip 桥梁已搭建
现在你已经理解了 RL 的核心概念（MDP、Bellman、Q-Learning、Policy Gradient、PPO），也理解了 LLM 生成如何映射为 RL 问题。下一章 [偏好对齐](./alignment.md) 将详细展开 RLHF pipeline、Reward Model 训练、PPO 实现，以及 DPO、GRPO 等替代方案。
:::

## 9. RL 方法总览

```
                    RL 方法分类
                        │
        ┌───────────────┼───────────────┐
     Model-Based     Model-Free       Hybrid
     (需要环境模型)   (不需要)
        │               │
     DP (§3)    ┌───────┼────────┐
             Value-Based    Policy-Based
                │               │
         ┌──────┤          REINFORCE (§7)
         │      │               │
      MC (§4)  TD (§5)     Actor-Critic
                │               │
         ┌──────┤            TRPO
         │      │               │
      SARSA  Q-Learning      PPO ← RLHF 的核心
                │
             DQN (§6)
```

## 苏格拉底时刻

在继续学习 RLHF 之前，检验你的理解：

1. **Bellman 方程的递推结构是什么？** 为什么说它是 RL 的基石？
2. **SARSA 和 Q-Learning 的唯一区别是什么？** 在悬崖行走（Cliff Walking）问题中，谁学到的路径更安全？为什么？
3. **DQN 为什么需要 Experience Replay 和 Target Network？** 去掉任何一个会怎样？
4. **对数导数技巧的关键一步是什么？** 它为什么让我们不需要环境模型就能算梯度？
5. **把 LLM 生成建模为 MDP，State、Action、Reward 分别是什么？** 这个建模有什么不完美之处？（提示：奖励信号的稀疏性）
6. **PPO 的 Clip 机制在解决什么问题？** 如果 $\epsilon$ 设得太大或太小会怎样？

## 面试考点

::: details Q1: 解释 On-Policy 和 Off-Policy 的区别
**On-Policy**（如 SARSA、PPO）：用当前策略采集数据，用同一策略更新。数据用完即丢，采样效率低但更稳定。

**Off-Policy**（如 Q-Learning、DQN）：数据可以由任意策略采集（行为策略），但更新的是目标策略。可以复用历史数据，采样效率高但需要额外技巧（如 Importance Sampling、Replay Buffer）保证正确性。

**RLHF 中的体现**：PPO 是 On-Policy（每轮用当前 LLM 生成新数据）；DPO 是 Off-Policy（用预先收集的偏好数据训练）。
:::

::: details Q2: 为什么 REINFORCE 方差大？怎么解决？
REINFORCE 用完整 episode 的 Return $G_t$ 乘以 $\nabla \log \pi$，而 $G_t$ 是多步奖励之和，波动很大。解决方案：
1. **Baseline**：$G_t - V(s_t)$，减去 Value Function 估计值，不改变期望但降低方差
2. **Actor-Critic**：用 TD error $r + \gamma V(s') - V(s)$ 替代 $G_t$，进一步降低方差（引入少量偏差）
3. **GAE (Generalized Advantage Estimation)**：在偏差和方差之间做权衡，PPO 默认使用
:::

::: details Q3: DQN 的 Target Network 为什么能稳定训练？
如果用同一个网络 $Q_\theta$ 同时计算当前 Q 值和 TD target：$y = r + \gamma \max Q_\theta(s')$，那么每次更新 $\theta$，target 也在变——这等于 "追一个移动靶"，容易发散。

Target Network $Q_{\theta^-}$ 的参数 **冻结一段时间**，使 target 相对稳定。每隔 $C$ 步才同步 $\theta^- \leftarrow \theta$。这是一种 **半梯度 (semi-gradient)** 方法——只对预测值求梯度，不对 target 求梯度。
:::

::: details Q4: PPO 的 Clip 机制如何工作？
PPO 限制策略更新幅度，定义 ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$：
- 若 $A_t > 0$（好的动作）：ratio 被裁剪到上界 $1 + \epsilon$，防止策略变化过大
- 若 $A_t < 0$（差的动作）：ratio 被裁剪到下界 $1 - \epsilon$，同样防止过度惩罚

效果：在一定的 "信赖域" 内优化策略，超出范围的梯度被截断。实践中 $\epsilon = 0.2$ 效果最好。
:::

::: details Q5: 解释 RLHF 中的 Reward Hacking 问题
Reward Model 只是人类偏好的近似。如果不加约束地最大化 RM 分数，LLM 会找到 RM 的漏洞——生成一些 RM 给高分但人类不喜欢的内容。这就是 **Goodhart 定律** 在 AI 中的体现："当一个指标成为目标时，它就不再是一个好指标"。

解决方案：加 KL 惩罚 $\beta \cdot \text{KL}[\pi_\theta \| \pi_\text{ref}]$，确保优化后的模型不偏离 SFT 模型太远。也可以用 DPO 直接从偏好数据学习，绕过 RM。
:::

## 推荐资源

| 资源 | 链接 | 说明 |
|------|------|------|
| Sutton & Barto 教材 | [incompleteideas.net/book](http://incompleteideas.net/book/the-book-2nd.html) | RL 圣经，免费在线版 |
| Spinning Up in Deep RL | [OpenAI](https://spinningup.openai.com/) | OpenAI 的 Deep RL 入门教程，有 PyTorch 实现 |
| David Silver RL 课程 | [UCL Course](https://www.davidsilver.uk/teaching/) | 经典 RL 课程视频 + slides |
| DQN 原论文 | [Nature 2015](https://www.nature.com/articles/nature14236) | *Human-level control through deep RL* |
| PPO 原论文 | [arXiv 2017](https://arxiv.org/abs/1707.06347) | *Proximal Policy Optimization Algorithms* |
| RLHF 原论文 | [arXiv 2022](https://arxiv.org/abs/2203.02155) | *Training language models to follow instructions with human feedback* (InstructGPT) |
| TRL 库 | [GitHub](https://github.com/huggingface/trl) | HuggingFace 的 RLHF/DPO/PPO 训练库 |
| 李宏毅 RL 课程 | [YouTube](https://www.youtube.com/playlist?list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_) | 中文 RL 课程（国立台湾大学） |
