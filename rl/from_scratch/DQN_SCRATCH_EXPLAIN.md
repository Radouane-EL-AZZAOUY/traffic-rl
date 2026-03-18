### Overview

This folder contains a **from-scratch DQN implementation** using only NumPy:

- `Network`: 2-layer MLP with ReLU.
- `ReplayBuffer`: experience storage for DQN.
- `DQNConfig`: hyperparameter container.
- `DQNAgent`: DQN logic (epsilon-greedy, target network, SGD/backprop).

The goal is to closely mirror **Stable-Baselines3 DQN** while remaining small and fully transparent.

---

### High-level data flow

Training loop (in `train_dqn.py --impl scratch`):

```text
SUMO (TraCI)  →  SumoEnv  →  state (4 floats)
                                   ↓
                               DQNAgent
                               (policy_net)
                                   ↓
                           action (0 = NS, 1 = EW)
                                   ↓
                             SumoEnv.step()
                                   ↓
               (next_state, reward, done, info)
                                   ↓
                              ReplayBuffer
                                   ↓
                          DQNAgent.train_step()
                          (policy_net update +
                           target_net sync)
```

At eval time (in `evaluate_from_scratch.py`):

```text
Load .npz → build Network → SumoEnv loop → greedy argmax(Q(s))
```

---

### `Network` – 2×64 MLP for Q-values

```python
class Network:
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        rng = np.random.default_rng()
        self.w1 = rng.normal(0, 0.1, size=(hidden_size, state_size))
        self.b1 = np.zeros((hidden_size,))
        self.w2 = rng.normal(0, 0.1, size=(hidden_size, hidden_size))
        self.b2 = np.zeros((hidden_size,))
        self.w3 = rng.normal(0, 0.1, size=(action_size, hidden_size))
        self.b3 = np.zeros((action_size,))
```

- **Purpose**: Implements the function \( Q_\theta(s) \) that outputs one Q-value per action.
- **Weights/bias shapes**:
  - `w1`: \((64, state\_size)\), `b1`: \((64,)\)
  - `w2`: \((64, 64)\), `b2`: \((64,)\)
  - `w3`: \((action\_size, 64)\), `b3`: \((action\_size,)\)
- **Initialization**: Small Gaussian noise (`std=0.1`) for weights, zeros for biases.

```python
    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = x @ self.w1.T + self.b1      # (B, H)
        a1 = np.maximum(0.0, z1)          # ReLU
        z2 = a1 @ self.w2.T + self.b2
        a2 = np.maximum(0.0, z2)          # ReLU
        q = a2 @ self.w3.T + self.b3      # (B, A)
        return q
```

- **Input**: `x` of shape `(batch_size, state_size)`.
- **Hidden layers**:
  - `z1`, `z2`: linear combinations.
  - `a1`, `a2`: ReLU activations, \( \max(0, z) \).
- **Output**: `q` of shape `(batch_size, action_size)` – Q-values for each action.

Utility methods:

```python
    def copy_from(self, other: "Network") -> None:
        self.w1 = other.w1.copy()
        ...

    def parameters(self) -> Tuple[np.ndarray, ...]:
        return self.w1, self.b1, self.w2, self.b2, self.w3, self.b3
```

- **`copy_from`**: used to sync target network.
- **`parameters`**: convenience to iterate over all tensors if needed.

---

### `ReplayBuffer` – experience storage

```python
class ReplayBuffer:
    def __init__(self, capacity: int, state_size: int):
        self.capacity = capacity
        self.state_buf = np.zeros((capacity, state_size), dtype=np.float32)
        self.next_state_buf = np.zeros((capacity, state_size), dtype=np.float32)
        self.action_buf = np.zeros((capacity,), dtype=np.int64)
        self.reward_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.size = 0
```

- Pre-allocates arrays to avoid dynamic Python lists.
- `idx`: write pointer (cyclic).
- `size`: current number of stored transitions.

```python
    def push(self, state, action, reward, next_state, done) -> None:
        self.state_buf[self.idx] = state
        self.next_state_buf[self.idx] = next_state
        self.action_buf[self.idx] = action
        self.reward_buf[self.idx] = reward
        self.done_buf[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
```

- Stores one transition at `idx`, then advances pointer with wrap-around.
- `done` is stored as `0.0` (not done) or `1.0` (terminal).

```python
    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (...buffers indexed by idxs...)
```

- Uniform random sampling used by DQN update.

---

### `DQNConfig` – hyperparameters

```python
@dataclass
class DQNConfig:
    state_size: int
    action_size: int
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 32
    buffer_size: int = 10_000
    learning_starts: int = 1_000
    train_freq: int = 4
    target_update_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.2
```

- Mirrors SB3’s key DQN hyperparameters.
- `epsilon_decay_fraction` controls how fast exploration decays relative to total training steps.

---

### `DQNAgent` – algorithm logic

Constructor:

```python
class DQNAgent:
    def __init__(self, config: DQNConfig, total_timesteps: int):
        self.cfg = config
        self.policy_net = Network(config.state_size, config.action_size)
        self.target_net = Network(config.state_size, config.action_size)
        self.target_net.copy_from(self.policy_net)
        self.replay_buffer = ReplayBuffer(config.buffer_size, config.state_size)
        self.total_timesteps = max(1, total_timesteps)
        self.timestep = 0
```

- Two networks:
  - `policy_net`: the one being updated every train step.
  - `target_net`: lagged copy, stabilizes targets.
- `total_timesteps`: used for epsilon schedule.

#### Epsilon-greedy schedule

```python
    def epsilon(self) -> float:
        decay_steps = int(self.cfg.epsilon_decay_fraction * self.total_timesteps)
        decay_steps = max(decay_steps, 1)
        if self.timestep >= decay_steps:
            return self.cfg.epsilon_end
        frac = self.timestep / decay_steps
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)
```

- Linear interpolation from `epsilon_start` to `epsilon_end` over `decay_steps`.
- After that, epsilon stays at `epsilon_end`.

Action selection:

```python
    def act(self, state: np.ndarray) -> int:
        self.timestep += 1
        if np.random.rand() < self.epsilon():
            return np.random.randint(self.cfg.action_size)
        state_batch = state.reshape(1, -1).astype(np.float32)
        q_values = self.policy_net.forward(state_batch)[0]
        return int(np.argmax(q_values))
```

- Increments global `timestep`.
- With probability `epsilon`, picks random action.
- Otherwise runs `policy_net` and chooses `argmax_a Q(s,a)`.

#### Storing transitions

```python
    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)
```

- Thin wrapper around the replay buffer.

#### Target network sync

```python
    def maybe_update_target(self) -> None:
        if self.timestep % self.cfg.target_update_interval == 0:
            self.target_net.copy_from(self.policy_net)
```

- Every `target_update_interval` environment steps, the target network is updated.

---

### Backpropagation and SGD – `train_step`

The core of DQN:

```python
    def train_step(self) -> float | None:
        if (self.timestep < self.cfg.learning_starts
            or self.timestep % self.cfg.train_freq != 0
            or len(self.replay_buffer) < self.cfg.batch_size):
            return None
```

- **Warmup**: wait until `learning_starts` and enough transitions are stored.
- **Frequency**: only update every `train_freq` steps.

Sampling a batch:

```python
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.cfg.batch_size)
```

Compute Q-values and TD target:

```python
        q_values = self.policy_net.forward(states)      # (B, A)
        batch_indices = np.arange(self.cfg.batch_size)
        q_sa = q_values[batch_indices, actions]         # (B,)

        with np.errstate(over="ignore"):
            next_q_values = self.target_net.forward(next_states)
            max_next_q = np.max(next_q_values, axis=1)

        targets = rewards + self.cfg.gamma * max_next_q * (1.0 - dones)
```

- `q_sa`: Q-value for the action actually taken in each transition.
- `max_next_q`: \( \max_{a'} Q_{\text{target}}(s', a') \).
- `targets`: standard DQN target  
  \( r + \gamma \max_{a'} Q_{\text{target}}(s', a') \) (no Double-DQN here).

Loss and gradient wrt Q-values:

```python
        td_error = q_sa - targets
        loss = float(np.mean(td_error**2))
        grad_q_sa = (2.0 / self.cfg.batch_size) * td_error
```

- Mean-squared error loss.
- `grad_q_sa` is \( \frac{\partial L}{\partial Q(s,a)} \) for each sample.

#### Gradient through final layer

Recompute activations for this batch:

```python
        z1 = states @ self.policy_net.w1.T + self.policy_net.b1
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ self.policy_net.w2.T + self.policy_net.b2
        a2 = np.maximum(0.0, z2)
```

Gradients for last layer:

```python
        grad_w3 = np.zeros_like(self.policy_net.w3)
        grad_b3 = np.zeros_like(self.policy_net.b3)
        for i in range(self.cfg.batch_size):
            a = actions[i]
            grad_w3[a] += grad_q_sa[i] * a2[i]
            grad_b3[a] += grad_q_sa[i]
```

- For each sample `i`, only the row corresponding to `actions[i]` receives gradient (like indexing in PyTorch `gather`).

Backprop into `a2`:

```python
        grad_a2 = np.zeros_like(a2)
        for i in range(self.cfg.batch_size):
            grad_a2[i] = grad_q_sa[i] * self.policy_net.w3[actions[i]]
```

- Each `grad_q_sa[i]` is multiplied by weights connecting hidden units to the chosen action.

ReLU and second layer gradients:

```python
        grad_z2 = grad_a2 * (z2 > 0.0)
        grad_w2 = grad_z2.T @ a1
        grad_b2 = np.sum(grad_z2, axis=0)
```

- ReLU derivative is `1` where `z2 > 0`, else `0`.
- Standard fully-connected backprop using matrix multiplication.

Backprop into first layer:

```python
        grad_a1 = grad_z2 @ self.policy_net.w2
        grad_z1 = grad_a1 * (z1 > 0.0)
        grad_w1 = grad_z1.T @ states
        grad_b1 = np.sum(grad_z1, axis=0)
```

- Same pattern: propagate gradient backward through layer 2 weights, then through ReLU, then compute gradients for layer 1 weights.

#### Parameter update (SGD)

```python
        lr = self.cfg.lr
        self.policy_net.w3 -= lr * grad_w3
        self.policy_net.b3 -= lr * grad_b3
        self.policy_net.w2 -= lr * grad_w2
        self.policy_net.b2 -= lr * grad_b2
        self.policy_net.w1 -= lr * grad_w1
        self.policy_net.b1 -= lr * grad_b1
```

- Simple stochastic gradient descent, no momentum or Adam.

Return value:

```python
        return loss
```

- `train_step` returns the scalar loss (or `None` if no update was performed).

---

### Saving / loading

Saving (called from `train_dqn.py`):

```python
    def save(self, path: str) -> None:
        np.savez(
            path,
            w1=self.policy_net.w1,
            b1=self.policy_net.b1,
            w2=self.policy_net.w2,
            b2=self.policy_net.b2,
            w3=self.policy_net.w3,
            b3=self.policy_net.b3,
        )
```

- Stores all network parameters in a `.npz` archive.
- `evaluate_from_scratch.py` reads this file, reconstructs a `Network`, and uses greedy `argmax(Q(s))` actions.

---

### Visual summary of the math

For a single transition in a batch:

```text
state s
   ↓
policy_net(s) → Q(s, ·)
   ↓
pick Q(s, a) for taken action a
   ↓
target = r + γ max_a' Q_target(s', a')
   ↓
td_error = Q(s, a) − target
   ↓
loss = mean(td_error²)
   ↓
backpropagate loss through w3, w2, w1
   ↓
SGD update of policy_net parameters
   ↓
every N steps: target_net ← policy_net
```

This is exactly the original **DQN algorithm**, but implemented explicitly with NumPy so you can see every intermediate tensor and gradient.

