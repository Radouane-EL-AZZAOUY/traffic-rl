# NumPy + Math Visual Guide (From Scratch DQN)

This file summarizes the **math forms** and **NumPy functions** used in `rl/from_scratch`, with:

- intuitive explanations
- LaTeX-style formulas
- small visual demonstrations (ASCII)
- exact file references

---

## 1) End-to-End Visual Workflow

```text
State s
  |
  v
Policy Network forward:
  z1 = xW1^T + b1
  a1 = ReLU(z1)
  z2 = a1W2^T + b2
  a2 = ReLU(z2)
  q  = a2W3^T + b3
  |
  +--> choose action (epsilon-greedy)
          |
          v
Environment step -> (r, s', done)
          |
          v
ReplayBuffer store and sample batch
          |
          v
Target:
  y = r + gamma * max_a' Q_target(s', a') * (1-done)
          |
          v
Loss:
  L = mean((Q_policy(s,a)-y)^2)
          |
          v
Backprop gradients -> Clip -> Adam update
```

Main implementation: `rl/from_scratch/dqn_numpy.py`.

---

## 2) Core Math Forms (with intuition)

## 2.1 He Initialization

**Where**: `dqn_numpy.py` (`__init__`, around weight creation lines)

\[
\sigma = \sqrt{\frac{2}{\text{fan\_in}}}, \quad W \sim \mathcal{N}(0, \sigma^2)
\]

Intuition:
- If initial weights are too small -> signals fade (vanishing).
- If too large -> signals explode.
- He init picks a balanced spread, especially good with ReLU.

Visual:

```text
too small std      good std (He)         too large std
    /\                /\                      /\
   /  \              /  \                    /  \
--/----\--      ----/----\----          ___/----\___
  tiny spread      balanced spread         huge spread
```

---

## 2.2 Forward Pass (2 hidden layers + ReLU)

**Where**: `dqn_numpy.py:41-46`

\[
z_1 = xW_1^T + b_1,\quad a_1 = \mathrm{ReLU}(z_1)
\]
\[
z_2 = a_1W_2^T + b_2,\quad a_2 = \mathrm{ReLU}(z_2)
\]
\[
q = a_2W_3^T + b_3
\]

Intuition:
- Linear layers mix features.
- ReLU adds nonlinearity.
- Output `q` has one value per action.

Shape flow:

```text
x: (B,S)
  -> z1: (B,H)
  -> a1: (B,H)
  -> z2: (B,H)
  -> a2: (B,H)
  -> q : (B,A)
```

---

## 2.3 Epsilon Decay (exploration schedule)

**Where**: `dqn_numpy.py:146-153`

\[
\epsilon_t =
\epsilon_{\text{start}} +
\frac{t}{T_{\text{decay}}}
(\epsilon_{\text{end}} - \epsilon_{\text{start}})
\]

Intuition:
- Start very exploratory.
- Gradually become more greedy.

Visual:

```text
epsilon
1.0 |\
    | \
    |  \
0.05|---\________ time
```

---

## 2.4 TD Target in DQN

**Where**: `dqn_numpy.py:201-203`

\[
y = r + \gamma \max_{a'}Q_{\text{target}}(s',a')\cdot(1-\text{done})
\]

Intuition:
- Reward now + discounted best future value.
- If episode ends (`done=1`), future term is zero.

---

## 2.5 TD Error + MSE Loss

**Where**: `dqn_numpy.py:206-207`

\[
\delta = Q_{\text{policy}}(s,a) - y,\quad
\mathcal{L} = \frac{1}{B}\sum_{i=1}^{B}\delta_i^2
\]

Intuition:
- `delta` = prediction mistake.
- Squaring punishes large mistakes more.

---

## 2.6 Gradient of MSE wrt selected Q-values

**Where**: `dqn_numpy.py:209`

\[
\frac{\partial \mathcal{L}}{\partial Q(s,a)} = \frac{2}{B}\delta
\]

Intuition:
- Bigger error -> stronger correction.

---

## 2.7 ReLU Derivative Mask

**Where**: `dqn_numpy.py:232`, `dqn_numpy.py:239`

\[
\frac{d}{dz}\mathrm{ReLU}(z)=
\begin{cases}
1,& z>0\\
0,& z\le0
\end{cases}
\]

Implemented as element-wise mask:

\[
\nabla z = \nabla a \odot [z>0]
\]

Visual:

```text
z:      [-2, -0.1, 0.7, 3.2]
mask:   [ 0,   0,  1,   1 ]
grad z: grad a * mask
```

---

## 2.8 Dense Layer Gradients

**Where**: `dqn_numpy.py:234-235`, `dqn_numpy.py:241-242`

\[
\nabla W = (\nabla Z)^T X,\quad
\nabla b = \sum_{\text{batch}} \nabla Z
\]

Intuition:
- Weight gradient depends on upstream gradient and input activations.
- Bias gradient is batch sum.

---

## 2.9 Global Gradient Clipping (L2 norm)

**Where**: `dqn_numpy.py:247-251`

\[
\|\mathbf{g}\|_2 = \sqrt{\sum_k g_k^2}
\]
If \(\|\mathbf{g}\|_2 > c\), scale:
\[
\mathbf{g} \leftarrow \mathbf{g}\cdot \frac{c}{\|\mathbf{g}\|_2+\epsilon}
\]

Intuition:
- Prevents one bad step from exploding parameters.

Visual:

```text
raw grad norm: 25
clip limit   : 10
scale        : 10/25 = 0.4
all grads -> 40% of original magnitude
```

---

## 2.10 Adam Update

**Where**: `dqn_numpy.py:274-278`

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]
\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]
\[
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad
\hat v_t=\frac{v_t}{1-\beta_2^t}
\]
\[
\theta \leftarrow \theta - \alpha \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]

Intuition:
- `m`: momentum-like smoothed gradient direction.
- `v`: smoothed squared gradient (adaptive step size).
- Bias correction fixes early-step underestimation.

---

## 3) NumPy Functions Used (with examples)

Below are NumPy calls found in `rl/from_scratch/*.py`.

## 3.1 Randomness

### `np.random.seed(seed)`
- **Where**: `train_dqn_scratch.py:55`
- Purpose: reproducibility for global RNG.
- Example:
  ```python
  np.random.seed(42)
  ```

### `np.random.default_rng()`
- **Where**: `dqn_numpy.py:22`
- Purpose: create modern RNG object.
- Example:
  ```python
  rng = np.random.default_rng()
  ```

### `rng.normal(loc, scale, size=...)`
- **Where**: `dqn_numpy.py:25,27,29`
- Purpose: Gaussian sampling for weights.
- Example:
  ```python
  W = rng.normal(0, 0.1, size=(64, 4))
  ```

### `np.random.rand()`
- **Where**: `dqn_numpy.py:157`
- Purpose: random float in `[0,1)` for epsilon-greedy decision.

### `np.random.randint(low, high, size=...)`
- **Where**: `dqn_numpy.py:92,158`
- Purpose: random replay indices / random action ID.

---

## 3.2 Array creation and shape helpers

### `np.zeros(shape, dtype=...)`
- **Where**: `dqn_numpy.py:26,28,30,66-70`
- Purpose: initialize biases and replay buffers.

### `np.zeros_like(x)`
- **Where**: `dqn_numpy.py:140,142,219,220,227`
- Purpose: same shape/type as `x`, filled with zeros.

### `np.array(list, dtype=...)`
- **Where**: `evaluate_scratch.py:51`
- Purpose: convert Python lane-count list to ndarray.

### `np.arange(n)`
- **Where**: `dqn_numpy.py:196`
- Purpose: batch index vector `[0,1,...,n-1]`.

---

## 3.3 Elementwise ops and reductions

### `np.sqrt(x)`
- **Where**: `dqn_numpy.py:25,27,29,247,278`
- Purpose: He scale, norm, Adam denominator.

### `np.maximum(a, b)`
- **Where**: `dqn_numpy.py:42,44,214,216`
- Purpose: ReLU (`max(0,z)`).

### `np.max(x, axis=1)`
- **Where**: `dqn_numpy.py:201`
- Purpose: max over actions for target value.

### `np.mean(x)`
- **Where**: `dqn_numpy.py:207`, `train_dqn_scratch.py:121`
- Purpose: MSE loss / log average loss.

### `np.sum(x, axis=...)`
- **Where**: `dqn_numpy.py:235,242,247`
- Purpose: bias gradients, norm computation.

### `np.argmax(x)`
- **Where**: `dqn_numpy.py:161`, `evaluate_scratch.py:168`
- Purpose: greedy action selection.

---

## 3.4 Numeric context / IO

### `np.errstate(over="ignore")`
- **Where**: `dqn_numpy.py:199`
- Purpose: suppress overflow warnings in scoped block.

### `np.savez(path, **arrays)`
- **Where**: `dqn_numpy.py:284-292`
- Purpose: save model weights to `.npz`.

### `np.load(path)`
- **Where**: `evaluate_scratch.py:157`
- Purpose: load saved model weights.

---

## 4) NumPy ndarray methods/properties used (important)

These are not `np.function(...)` calls, but central in this code:

- `.T` transpose: `dqn_numpy.py:41,43,45,213,215,234,241`
- `.reshape(1, -1)`: `dqn_numpy.py:159`, `evaluate_scratch.py:167`
- `.astype(np.float32)`: `dqn_numpy.py:159`
- `.copy()`: `dqn_numpy.py:50-55`

---

## 5) Tiny concrete walkthrough (single sample)

Assume:
- `state_size=4`, `hidden_size=64`, `action_size=2`
- one state \(x\in\mathbb{R}^{1\times4}\)

Forward:

\[
x(1\times4)\to z_1(1\times64)\to a_1(1\times64)\to z_2(1\times64)\to a_2(1\times64)\to q(1\times2)
\]

If `q = [1.2, 0.7]`, then:
- greedy action = `argmax(q) = 0`

If replay target is \(y=0.3\) for chosen action:
- TD error \(=1.2-0.3=0.9\)
- MSE term \(=0.9^2=0.81\)
- update pushes \(Q(s,a)\) downward.

---

## 6) Quick mnemonic

```text
Init stable -> Predict Q -> Explore/Exploit -> Store -> Sample
-> Build target -> Compute TD error -> Backprop -> Clip -> Adam -> Repeat
```

