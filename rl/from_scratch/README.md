# From-Scratch DQN — Complete Beginner's Guide

This folder teaches one idea: **how to make a computer learn to control a traffic light, step by step, using nothing but math and NumPy**.  
No magic black boxes. Every number, every calculation, every line of code is explained below.

---

## Table of Contents

1. [The Real-World Problem](#1-the-real-world-problem)
2. [The Big Idea: Learning by Trial and Error](#2-the-big-idea-learning-by-trial-and-error)
3. [Vocabulary: Six Words You Need to Know](#3-vocabulary-six-words-you-need-to-know)
4. [What is a Q-Value? (The Core of DQN)](#4-what-is-a-q-value-the-core-of-dqn)
5. [The Neural Network — The Brain](#5-the-neural-network--the-brain)
6. [How the Network Learns — Backpropagation](#6-how-the-network-learns--backpropagation)
7. [The Replay Buffer — The Memory](#7-the-replay-buffer--the-memory)
8. [The Target Network — The Stable Critic](#8-the-target-network--the-stable-critic)
9. [Epsilon-Greedy — Try New Things or Use What You Know](#9-epsilon-greedy--try-new-things-or-use-what-you-know)
10. [The Complete Training Loop — Everything Together](#10-the-complete-training-loop--everything-together)
11. [File-by-File Walkthrough](#11-file-by-file-walkthrough)
12. [Hyperparameters Explained](#12-hyperparameters-explained)
13. [How to Run It](#13-how-to-run-it)
14. [Reading the Training Output](#14-reading-the-training-output)

---

## 1. The Real-World Problem

There is a **road intersection** shaped like a plus sign (`+`).  
Cars arrive from four directions: **North, South, East, West**.

```
          North
            |
            |  cars going ↓
            |
West -------+------- East
            |
            |  cars going ↑
            |
          South
```

A traffic light at the center (`+`) decides:  
- Either **North–South gets green** (cars flow vertically, East–West wait).  
- Or **East–West gets green** (cars flow horizontally, North–South wait).

The question is: **which direction should get green, and when?**

A bad decision (e.g. always keeping North–South green even when 20 cars are waiting from the East) makes cars wait a long time. A good decision minimizes total waiting time.

Our goal: **teach a computer to make good decisions automatically by letting it practice millions of times in a simulation (SUMO).**

---

## 2. The Big Idea: Learning by Trial and Error

Imagine training a dog.  
You don't explain the rules. You just give a **treat** when it does the right thing, and no treat (or a mild correction) when it does the wrong thing. Over many repetitions, the dog learns.

This is exactly **Reinforcement Learning (RL)**.

```
┌─────────────────────────────────────────────┐
│                                             │
│   AGENT (our DQN)                           │
│   "I see the intersection.                  │
│    I choose: NS green or EW green."         │
│                                             │
└──────────────┬──────────────────────────────┘
               │ takes action
               ▼
┌─────────────────────────────────────────────┐
│                                             │
│   ENVIRONMENT (SUMO simulation)             │
│   "Here is what happens:                   │
│    cars move, some wait, some pass."        │
│                                             │
└──────────────┬──────────────────────────────┘
               │ gives back: new situation + score
               ▼
┌─────────────────────────────────────────────┐
│   AGENT receives:                           │
│   - New situation (how many cars per lane)  │
│   - Score (reward): how good was that?      │
│   → Remembers this, adjusts future choices  │
└─────────────────────────────────────────────┘
```

This cycle repeats **30,000 times** (default). After enough repetitions the agent has learned which action is best in any situation.

---

## 3. Vocabulary: Six Words You Need to Know

These six words are used everywhere. Understand them here and the rest becomes easy.

---

### State (What the agent sees)

The **state** is a snapshot of the current situation. It answers: *"What does the intersection look like right now?"*

In this project, the state is **4 numbers** — one per incoming lane — counting how many vehicles are currently on each lane:

```
State = [lane_North, lane_South, lane_East, lane_West]
      = [  3,          1,          7,          0     ]
          ↑             ↑            ↑            ↑
       3 cars        1 car       7 cars       no cars
       waiting      waiting     waiting
```

These 4 numbers come directly from SUMO via TraCI (the Python API to SUMO) and are collected in `sumo_env.py`.

---

### Action (What the agent decides)

The **action** is the choice the agent makes. There are only **2 options**:

```
Action 0  →  North–South GREEN  (East–West must wait)
Action 1  →  East–West GREEN    (North–South must wait)
```

The agent looks at the current state and picks one of these two actions every 5 simulation seconds.

---

### Reward (The score for that decision)

After the agent acts, the simulation runs for 5 seconds, then a **reward** is calculated.

The reward answers: *"Was that a good or bad decision?"*

```
reward = −(total seconds all vehicles have been waiting)
```

Because of the minus sign:
- If no one is waiting → reward = 0 (perfect)
- If many cars are waiting → reward = −large number (bad)
- **Higher reward = less waiting = better decision**

---

### Episode

One full simulation run (0 to 360 seconds) is called an **episode**.  
During training, the simulation resets and starts a new episode many times.

```
Episode 1:  steps 1 → 72  (each step = 5 seconds → 360 seconds total)
Episode 2:  steps 1 → 72
Episode 3:  steps 1 → 72
...
(many episodes until 30,000 total steps)
```

---

### Policy (The agent's strategy)

The **policy** is the rule the agent uses to choose actions. After training, the policy is stored in the neural network weights.  
Think of it as the agent's "instinct" — given a situation, it instantly knows the best action to take.

---

### Timestep

A **timestep** is one decision-making moment.  
Each timestep = one action chosen, simulation runs for 5 seconds, one reward received.  
Training runs for 30,000 timesteps total.

---

## 4. What is a Q-Value? (The Core of DQN)

This is the most important concept. Everything else builds on it.

A **Q-value** answers:

> "If I am in situation S and I take action A, how much total reward can I expect to collect from now until the end of the episode?"

Written as `Q(state, action)`.

### A concrete example

```
State:  [3, 1, 7, 0]  ← 7 cars waiting from the East!

Q(this state, action=0: NS green) = −500   ← predicted future reward if I pick NS green
Q(this state, action=1: EW green) = −150   ← predicted future reward if I pick EW green
```

The agent picks the action with the **highest Q-value**, which is action 1 (EW green) because −150 > −500. That makes sense: there are 7 cars waiting from the East, so giving East–West the green light will reduce waiting more.

### The DQN update rule

After each experience, the Q-value estimate is corrected using this formula:

```
New estimate of Q(s, a)  =  actual reward received
                          +  γ × best Q-value in the next state
```

Where `γ` (gamma) = 0.99 is the **discount factor**: it says future rewards are slightly less valuable than immediate rewards (like preferring €100 today over €100 in a year).

In code (`dqn_numpy.py` line 188):
```python
targets = rewards + cfg.gamma * max_next_q * (1.0 - dones)
```

`(1.0 - dones)` sets the target to just `rewards` when the episode has ended (there is no "next state").

---

## 5. The Neural Network — The Brain

We cannot store a Q-value for every possible state, because the number of possible lane count combinations is astronomically large.

Instead, we use a **neural network** that takes the 4 lane counts as input and outputs 2 Q-values (one per action). The network *approximates* the Q-function.

### Architecture (from `dqn_numpy.py`)

```
Input layer          Hidden layer 1       Hidden layer 2       Output layer
(4 neurons)          (64 neurons)         (64 neurons)         (2 neurons)

  lane_N  ──────┐
  lane_S  ───── ├──→ [64 units] ──→ [64 units] ──→  Q(s, action=0: NS green)
  lane_E  ───── ┤                                    Q(s, action=1: EW green)
  lane_W  ──────┘

     4 inputs        64 hidden           64 hidden         2 outputs
```

### What happens inside each layer

Each connection has a **weight** (a number that is adjusted during training). Each neuron also has a **bias** (an extra number added). The calculation inside one layer is:

```
output = (input × weights) + bias
```

Then a **ReLU activation** is applied, which just means:
```
if output < 0:  → replace with 0
if output ≥ 0:  → keep as-is
```

This gives the network the ability to learn complex patterns (not just straight lines).

### Shapes of the weight matrices (from `dqn_numpy.py` lines 25–30)

```python
w1  shape: (64, 4)    # connects 4 inputs  → 64 hidden neurons  (layer 1)
w2  shape: (64, 64)   # connects 64 hidden → 64 hidden neurons  (layer 2)
w3  shape: (2, 64)    # connects 64 hidden → 2 output Q-values  (layer 3)
```

At the start of training, all weights are tiny random numbers close to zero. After 30,000 timesteps of corrections, they hold the learned policy.

### Forward pass (computing Q-values from a state)

From `dqn_numpy.py` lines 41–46:

```python
z1 = x @ w1.T + b1      # matrix multiply: (4 inputs) → (64 numbers)
a1 = max(0, z1)          # ReLU: remove negatives

z2 = a1 @ w2.T + b2     # (64 numbers) → (64 numbers)
a2 = max(0, z2)          # ReLU

q  = a2 @ w3.T + b3     # (64 numbers) → (2 Q-values)
```

The final output `q` is a vector of 2 numbers: `[Q(s, NS green), Q(s, EW green)]`.

---

## 6. How the Network Learns — Backpropagation

After collecting enough experience, the network needs to get better. This is done by **adjusting the weights** to make the Q-value predictions more accurate.

### Step 1 — Measure the error (loss)

The **TD error** (Temporal Difference error) is the gap between what the network predicted and what it should have predicted:

```
TD error = Q_predicted(s, a)  −  Q_target(s, a)

where Q_target = reward + γ × max Q_target_network(s')
```

The **loss** is the average of squared TD errors over a batch of 32 experiences:

```
Loss = mean( TD_error² )
```

### Step 2 — Backpropagation: tracing where the error came from

Backpropagation is just the **chain rule of calculus** applied backwards through the network. It answers: *"How much did each weight contribute to the error?"*

In plain terms: imagine the network is a pipeline of water pipes. Water (error) flows backward from the output, and at each junction we measure how much flow goes through each pipe. Those measurements tell us how to adjust each pipe.

```
Error at output (grad_q_sa)
         │
         ▼  (flows backward through w3)
Error at hidden layer 2 (grad_a2)
         │
         ▼  (passes through ReLU gate: closed if z2≤0, open if z2>0)
         ▼  (flows backward through w2)
Error at hidden layer 1 (grad_a1)
         │
         ▼  (passes through ReLU gate)
         ▼  (flows backward through w1)
Error signal for inputs (not used, inputs are fixed)
```

From `dqn_numpy.py` lines 217–227:

```python
grad_z2 = grad_a2 * (z2 > 0.0)   # ReLU gate: passes gradient only where z2 > 0
grad_w2 = grad_z2.T @ a1          # how much did w2 contribute?
grad_b2 = sum(grad_z2)            # how much did b2 contribute?

grad_a1 = grad_z2 @ w2            # error flowing back into layer 1
grad_z1 = grad_a1 * (z1 > 0.0)   # ReLU gate for layer 1
grad_w1 = grad_z1.T @ states      # how much did w1 contribute?
grad_b1 = sum(grad_z1)
```

### Step 3 — Gradient Descent: nudge the weights in the right direction

Once we know how much each weight contributed to the error, we **nudge it in the opposite direction**:

```
new_weight = old_weight  −  learning_rate × gradient
```

The **learning rate** (0.0005) controls how large each nudge is. Too large → unstable learning. Too small → very slow learning.

From `dqn_numpy.py` lines 229–236:

```python
w1 -= lr * grad_w1    # nudge every weight by a tiny amount
b1 -= lr * grad_b1
w2 -= lr * grad_w2
b2 -= lr * grad_b2
w3 -= lr * grad_w3
b3 -= lr * grad_b3
```

After 30,000 such corrections, the weights encode a good policy.

---

## 7. The Replay Buffer — The Memory

If the agent learned only from its very last experience, it would quickly forget past lessons and its learning would be unstable (learning from the last experience could contradict what it learned from experiences 100 steps ago).

The **Replay Buffer** is a **notebook** where every experience is written down. During training, a random page is selected from this notebook — not the most recent page, any page. This prevents the agent from over-fitting to recent events.

### Structure

```
Buffer (10,000 slots, arranged in a circle):

Slot 0:  [ state=[3,1,7,0], action=1, reward=-180, next_state=[1,0,3,0], done=False ]
Slot 1:  [ state=[2,2,2,2], action=0, reward= -90, next_state=[2,2,1,1], done=False ]
Slot 2:  [ state=[0,0,0,0], action=1, reward=   0, next_state=[1,0,0,0], done=True  ]
...
Slot 9999: (wraps around and overwrites slot 0 when full)
```

### Why a circle?

Because there are only 10,000 slots. When the 10,001st experience arrives, it overwrites slot 0 (the oldest). This keeps the buffer full of recent-ish experiences without using unlimited memory.

From `dqn_numpy.py` lines 88–89:

```python
self.idx = (self.idx + 1) % self.capacity   # move pointer forward; wrap at 10,000
self.size = min(self.size + 1, self.capacity)
```

### Sampling

During each training step, **32 random experiences** are pulled from the buffer (called a **batch**):

```python
idxs = np.random.randint(0, self.size, size=32)   # 32 random slot numbers
```

Using random samples breaks the correlation between consecutive steps, which makes the learning much more stable.

---

## 8. The Target Network — The Stable Critic

Here is a subtle problem: when we update the network's weights to reduce the TD error, we change both sides of the equation at the same time, because the target `Q_target(s')` is also computed by the same network we are updating.

It is like trying to measure a moving ruler with the same moving ruler. The target keeps shifting, making learning unstable.

**Solution**: use **two separate networks** with identical architecture:

```
┌────────────────────┐         ┌────────────────────┐
│   policy_net       │         │   target_net        │
│   (updated every   │         │   (frozen; only     │
│    4 steps)        │         │    copied from      │
│                    │         │    policy_net every │
│   → Q(s, a)        │         │    500 steps)       │
│     for training   │         │   → Q(s', a')       │
│     loss           │         │     for stable      │
│                    │         │     target values   │
└────────────────────┘         └────────────────────┘
```

The **target network** is a snapshot of the policy network. It is frozen for 500 steps at a time, giving the policy network a stable target to learn toward.

From `dqn_numpy.py` lines 158–160:

```python
def maybe_update_target(self) -> None:
    if self.timestep % 500 == 0:          # every 500 steps
        self.target_net.copy_from(self.policy_net)   # take a fresh snapshot
```

---

## 9. Epsilon-Greedy — Try New Things or Use What You Know

At the start of training the agent knows nothing. If it only used its (useless) Q-values, it would always pick the same action and never discover what happens with other actions.

**Epsilon-greedy** is the strategy that balances:
- **Exploration**: try a random action (learn new things).
- **Exploitation**: pick the action with the highest Q-value (use what you know).

```
ε (epsilon) = probability of picking a RANDOM action

Training timeline:
                       30,000 steps total
                       ←──────────────────────→
Exploration fraction = 20% of 30,000 = 6,000 steps

ε = 1.0                                ε = 0.05
│────────────────────────│──────────────────────│
step 0              step 6,000            step 30,000
 ↑                      ↑                      ↑
100% random         5% random              5% random
(explore)           (mostly exploit)       (exploit)
```

So the first 6,000 steps are almost entirely random exploration. After that, the agent mostly uses its learned Q-values but still makes random choices 5% of the time (to avoid getting stuck).

From `dqn_numpy.py` lines 131–138:

```python
def epsilon(self) -> float:
    decay_steps = int(0.2 * 30_000)   # = 6,000
    frac = self.timestep / decay_steps
    return 1.0 + frac * (0.05 - 1.0) # linearly goes from 1.0 down to 0.05
```

And how the action is chosen (lines 140–146):

```python
def act(self, state):
    if random_number < epsilon():
        return random action        # EXPLORE
    else:
        q = policy_net.forward(state)
        return argmax(q)            # EXPLOIT (pick best known action)
```

---

## 10. The Complete Training Loop — Everything Together

Now all pieces fit together. Here is one full timestep, step by step:

```
┌─────────────────────────────────────────────────────────────────┐
│  TIMESTEP t                                                     │
│                                                                 │
│  1. READ STATE                                                  │
│     obs = [lane_N, lane_S, lane_E, lane_W]                     │
│     e.g. obs = [3, 1, 7, 0]                                     │
│                                                                 │
│  2. CHOOSE ACTION  (epsilon-greedy)                             │
│     if random() < ε:  action = random (0 or 1)   ← explore     │
│     else:             action = argmax Q(obs)      ← exploit     │
│                                                                 │
│  3. APPLY ACTION TO SIMULATION                                  │
│     set traffic light phase → run 5 seconds                    │
│     → collect next_obs, reward, done                            │
│                                                                 │
│  4. STORE IN REPLAY BUFFER                                      │
│     buffer.push(obs, action, reward, next_obs, done)           │
│                                                                 │
│  5. TRAIN (every 4 steps, after 1,000 warmup steps)             │
│     a. sample 32 random experiences from buffer                 │
│     b. compute target = reward + 0.99 × max Q_target(next_obs)  │
│     c. compute TD error = Q_policy(obs, action) - target        │
│     d. backpropagate error through w3 → w2 → w1                 │
│     e. update weights with gradient descent                     │
│                                                                 │
│  6. UPDATE TARGET NETWORK (every 500 steps)                     │
│     target_net ← copy of policy_net                             │
│                                                                 │
│  7. obs ← next_obs                                              │
│     if done: reset simulation, start new episode                │
└─────────────────────────────────────────────────────────────────┘
repeat 30,000 times
```

Here it is as a flow diagram:

```
          ┌──────────────┐
          │  SumoEnv     │◄────────────────────────────────────┐
          │  (SUMO sim)  │                                     │
          └──────┬───────┘                                     │
                 │ obs (4 lane counts)                          │
                 ▼                                             │
          ┌──────────────┐                                     │
          │  DQNAgent    │                                     │
          │  .act(obs)   │──────────────────┐                  │
          └──────────────┘                 │ action (0 or 1)  │
                                           ▼                  │
                                   ┌──────────────┐           │
                                   │  SumoEnv     │           │
                                   │  .step(act)  │           │
                                   └──────┬───────┘           │
                                          │ next_obs,          │
                                          │ reward, done       │
                                          ▼                   │
                                   ┌──────────────┐           │
                                   │ ReplayBuffer │           │
                                   │  .push(...)  │           │
                                   └──────┬───────┘           │
                                          │ (every 4 steps)    │
                                          ▼                   │
                                   ┌──────────────┐           │
                                   │  .train_step │           │
                                   │  sample batch│           │
                                   │  compute TD  │           │
                                   │  backprop    │           │
                                   │  update w1..3│           │
                                   └──────────────┘           │
                                          │                   │
                                          │ obs ← next_obs    │
                                          └───────────────────┘
```

---

## 11. File-by-File Walkthrough

```
rl/from_scratch/
├── dqn_numpy.py          ← The algorithm (Network, ReplayBuffer, DQNAgent)
├── train_dqn_scratch.py  ← Run this to train; produces a .npz model file
├── evaluate_scratch.py   ← Run this after training to compare controllers
├── README.md             ← This file
└── DQN_SCRATCH_EXPLAIN.md ← Technical reference (shapes, math details)
```

---

### `dqn_numpy.py` — The Algorithm

This file has 4 classes. Think of them as 4 physical objects:

| Class | What it is | Real-world analogy |
|---|---|---|
| `Network` | The neural network | A calculator that takes lane counts and outputs Q-values |
| `ReplayBuffer` | Experience storage | A notebook of past events |
| `DQNConfig` | Hyperparameters | Knobs and dials that configure behavior |
| `DQNAgent` | Glues everything together | The agent's brain: chooses actions, learns, saves |

#### `Network` in detail

```
Network(state_size=4, action_size=2, hidden_size=64)

Weights created:
  w1: random numbers, shape (64, 4)   ← 64×4 = 256 numbers
  b1: zeros,          shape (64,)
  w2: random numbers, shape (64, 64)  ← 64×64 = 4,096 numbers
  b2: zeros,          shape (64,)
  w3: random numbers, shape (2, 64)   ← 2×64 = 128 numbers
  b3: zeros,          shape (2,)

Total parameters: 256 + 64 + 4096 + 64 + 128 + 2 = 4,610 numbers
```

All 4,610 numbers start small and random. After training, they collectively encode the traffic light control policy.

Key method: `forward(x)` — given lane counts, compute Q-values:
```
[3, 1, 7, 0] → Network.forward() → [Q(NS green), Q(EW green)]
                                    = [-500,        -150       ]
```

#### `ReplayBuffer` in detail

```
ReplayBuffer(capacity=10_000, state_size=4)

Pre-allocates 5 arrays:
  state_buf:      shape (10000, 4)  ← 10,000 stored states
  next_state_buf: shape (10000, 4)  ← 10,000 stored next states
  action_buf:     shape (10000,)    ← 10,000 stored actions (0 or 1)
  reward_buf:     shape (10000,)    ← 10,000 stored rewards
  done_buf:       shape (10000,)    ← 10,000 stored done flags (0.0 or 1.0)
```

Key methods:
- `push(...)` — write one experience at the current pointer position, then advance pointer.
- `sample(32)` — pick 32 random row indices; return those rows from all 5 arrays.

#### `DQNConfig` in detail

Just a data container. All values mirror `rl/train_dqn.py` exactly so both implementations are comparable:

```python
gamma               = 0.99     # discount factor (how much future rewards matter)
lr                  = 5e-4     # learning rate (how big each weight nudge is)
batch_size          = 32       # how many experiences per training step
buffer_size         = 10_000   # max experiences stored
learning_starts     = 1_000    # wait 1,000 steps before starting to learn
train_freq          = 4        # train every 4 steps
target_update_interval = 500   # copy policy_net → target_net every 500 steps
epsilon_start       = 1.0      # start 100% random
epsilon_end         = 0.05     # end at 5% random
epsilon_decay_fraction = 0.2   # decay over first 20% of training steps
```

#### `DQNAgent` in detail

The agent owns one `policy_net`, one `target_net`, one `ReplayBuffer`, and a `timestep` counter.

Key methods:

| Method | When called | What it does |
|---|---|---|
| `act(state)` | Every timestep | Increments timestep; returns random or best action |
| `store_transition(...)` | Every timestep | Writes experience to replay buffer |
| `train_step()` | Every timestep | If conditions met, samples buffer and updates policy_net weights |
| `maybe_update_target()` | Every timestep | If 500 steps elapsed, copies policy_net → target_net |
| `save(path)` | Once, after training | Saves 6 weight arrays to a `.npz` file |

---

### `train_dqn_scratch.py` — The Training Script

This is the script you run. It:

1. **Parses CLI arguments** — `--timesteps`, `--save-path`, `--seed`, `--gui`.
2. **Creates the environment** — `SumoEnv` wraps the SUMO simulation.
3. **Creates the agent** — `DQNAgent` with `DQNConfig` using the same hyperparameters as `rl/train_dqn.py`.
4. **Runs the training loop** — 30,000 timesteps of act → step → store → train → update target.
5. **Logs progress** every 1,000 steps.
6. **Saves the model** — calls `agent.save(path)` which writes `dqn_traffic_light_scratch.npz`.

The training loop (lines 99–127) is the most important part:

```python
for t in range(30_000):
    action = agent.act(obs)                              # step 2: choose
    next_obs, reward, terminated, truncated, _ = env.step(action)  # step 3: apply
    done = terminated or truncated

    agent.store_transition(obs, action, reward, next_obs, done)  # step 4: remember
    loss = agent.train_step()                           # step 5: learn
    agent.maybe_update_target()                         # step 6: sync target

    obs = next_obs
    if done:
        obs, _ = env.reset()    # new episode
```

---

### `evaluate_scratch.py` — The Evaluation Script

After training, this script compares **three controllers** side by side:

```
┌─────────────────┬──────────────────────────────────────────────────┐
│  Controller     │  How it decides the phase                        │
├─────────────────┼──────────────────────────────────────────────────┤
│  Fixed-time     │  SUMO's default: fixed green/red cycle, ignores  │
│                 │  actual traffic                                   │
├─────────────────┼──────────────────────────────────────────────────┤
│  Random         │  Picks NS or EW green at random every 5 seconds  │
├─────────────────┼──────────────────────────────────────────────────┤
│  RL-scratch     │  Loads the .npz weights, runs the neural network  │
│                 │  to pick the action with the highest Q-value     │
└─────────────────┴──────────────────────────────────────────────────┘
```

The output is a table like:

```
=================================================================
Comparison (lower waiting / higher reward is better)
=================================================================
  Fixed-time     | mean_waiting:   430.2 s | mean_queue: 8.1 | total_reward: -30000
  Random         | mean_waiting:   510.7 s | mean_queue: 9.4 | total_reward: -36800
  RL-scratch     | mean_waiting:   280.5 s | mean_queue: 4.9 | total_reward: -20100
=================================================================
```

A well-trained RL agent should have **lower mean_waiting** and **higher (less negative) total_reward** than both baselines.

---

## 12. Hyperparameters Explained

| Name | Value | Plain English |
|---|---|---|
| `gamma` = 0.99 | Discount factor | A reward 10 steps away is worth 0.99^10 ≈ 90% of an immediate reward. Keeps the agent focused on long-term improvement. |
| `lr` = 5e-4 = 0.0005 | Learning rate | How big is each weight nudge. Too big → unstable. Too small → too slow. |
| `batch_size` = 32 | Batch size | Train on 32 randomly sampled experiences at once. More stable than training on one at a time. |
| `buffer_size` = 10,000 | Replay buffer capacity | Remember the last 10,000 experiences. |
| `learning_starts` = 1,000 | Warmup period | Do not update weights until 1,000 random experiences have been collected. Ensures the buffer has enough variety before training starts. |
| `train_freq` = 4 | Training frequency | Update weights every 4 timesteps, not every 1. Reduces computational cost and helps stability. |
| `target_update_interval` = 500 | Target network lag | The target network is a frozen copy of the policy network, refreshed every 500 steps. Prevents the "moving target" instability. |
| `epsilon_start` = 1.0 | Initial exploration | Start by choosing actions 100% randomly. |
| `epsilon_end` = 0.05 | Final exploration | After decay, still choose randomly 5% of the time to avoid getting permanently stuck. |
| `epsilon_decay_fraction` = 0.2 | Decay speed | Epsilon decays from 1.0 to 0.05 over the first 20% of training (= first 6,000 of 30,000 steps). |

---

## 13. How to Run It

All commands are run from the **project root** (`traffic-rl/`).

### Step 1 — Train (produces the .npz model file)

```bash
python rl/from_scratch/train_dqn_scratch.py
```

Or with custom settings:

```bash
python rl/from_scratch/train_dqn_scratch.py \
    --timesteps 50000 \
    --save-path rl/models/dqn_scratch \
    --seed 123
```

This saves weights to `rl/models/dqn_traffic_light_scratch.npz`.

### Step 2 — Evaluate (compare with baselines)

```bash
python rl/from_scratch/evaluate_scratch.py
```

Or pointing to a custom model:

```bash
python rl/from_scratch/evaluate_scratch.py \
    --model rl/models/dqn_scratch.npz
```

---

## 14. Reading the Training Output

During training, a table is printed every 1,000 timesteps:

```
  Timestep   Episode    Ep reward        Loss   Epsilon
------------------------------------------------------------
      1000         2        -95.3      1234.5     0.833
      2000         4       -180.1       987.2     0.667
      4000         8       -220.5       543.1     0.333
      6000        12        -95.2       221.3     0.050
     10000        20        -60.0        98.7     0.050
     30000        60        -40.1        12.3     0.050
```

| Column | What it means |
|---|---|
| `Timestep` | How many decision steps have passed |
| `Episode` | How many full simulation runs (0–360 s) have completed |
| `Ep reward` | Cumulative reward for the current (unfinished) episode. Becomes less negative as the agent learns |
| `Loss` | Average TD error² over the last 1,000 training steps. Should trend downward |
| `Epsilon` | Current exploration probability. Decreases from 1.0 → 0.05 in the first 6,000 steps |

**Signs of healthy training:**
- `Ep reward` becomes less negative over time (less waiting → better).
- `Loss` generally decreases (predictions improve).
- `Epsilon` reaches 0.050 by step 6,000 and stays there.

**After `evaluate_scratch.py`:**  
The RL-scratch controller should show a lower `mean_waiting` than both Fixed-time and Random.
