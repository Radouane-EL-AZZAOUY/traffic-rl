# Traffic-RL: Deep Technical Analysis

## Table of Contents

1. [Global Architecture](#1)
2. [Project Phases (Pipeline)](#2)
3. [SUMO — Simulation Engine](#3)
4. [TraCI — The Bridge Between Python and SUMO](#4)
5. [Reinforcement Learning: MDP Formulation](#5)
6. [DQN: The AI Model in Detail](#6)
7. [Gymnasium Environment (`sumo_env.py`)](#7)
8. [Training, Evaluation & Baselines](#8)
9. [FastAPI Backend & KPI Service](#9)
10. [Real-World OSM Pipeline](#10)
11. [Python Libraries Explained](#11)

---

## 1 — Global Architecture

The project is a **complete end-to-end RL pipeline** for smart traffic light control. It has 3 major layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                       traffic-rl/                               │
│                                                                 │
│  ┌──────────────┐   TraCI    ┌──────────────┐   HTTP    ┌────┐ │
│  │  SUMO Layer  │◄──────────►│   RL Layer   │◄─────────►│API │ │
│  │  sumo/       │  TCP/IP   │   rl/        │  REST     │back│ │
│  │  sumo_osm/   │  protocol │  sumo_env.py │           │end/│ │
│  └──────────────┘           └──────────────┘           └────┘ │
│         │                         │                            │
│    Synthetic Grid           DQN (SB3 + Gym)              FastAPI│
│    + OSM Real World        Random + Fixed              + KPIs  │
└─────────────────────────────────────────────────────────────────┘
```

**Full File Tree with Roles:**

```
traffic-rl/
├── sumo/                          ← Phase 1–2: SUMO network + manual control
│   ├── intersection.nod.xml       ← 5 junction nodes (center B1 + 4 endpoints)
│   ├── intersection.edg.xml       ← 8 bidirectional edges (200m each)
│   ├── intersection.net.xml       ← Compiled SUMO network (generated)
│   ├── routes.rou.xml             ← 8 vehicle flows (~1400 veh/h total)
│   ├── simulation.sumocfg         ← SUMO config (links net + routes)
│   ├── generate_network.py        ← Phase 1: netgenerate/netconvert wrapper
│   └── traci_manual_control.py    ← Phase 2: Manual TL switching via TraCI
│
├── rl/                            ← Phase 3–5: RL pipeline
│   ├── sumo_utils.py              ← Constants (B1_PHASES, lane IDs) + path helpers
│   ├── sumo_env.py                ← Phase 4: Gymnasium SumoEnv (MDP wrapper)
│   ├── random_agent.py            ← Phase 3: Random baseline controller
│   ├── train_dqn.py               ← Phase 5: DQN training via Stable-Baselines3
│   ├── evaluate.py                ← Phase 5: Fixed vs Random vs RL comparison
│   └── models/
│       └── dqn_traffic_light.zip  ← Saved trained model
│
├── backend/                       ← Phase 6–7: API + KPI
│   ├── kpi_service.py             ← Phase 6: KPICollector + simulation runner
│   └── main.py                    ← Phase 7: FastAPI REST API (6 endpoints)
│
├── sumo_osm/                      ← Phase 9: Real-world OSM pipeline
│   ├── download_osm.py            ← Download from OpenStreetMap (Overpass API)
│   ├── build_net.py               ← OSM → SUMO net (netconvert)
│   ├── generate_routes.py         ← Random trips (randomTrips.py + duarouter)
│   ├── detect_intersection.py     ← Find best TL junction in net XML
│   └── run_rl_agent.py            ← Deploy trained DQN on real OSM net
│
└── requirements.txt               ← gymnasium, stable-baselines3, fastapi, etc.
```

---

## 2 — Project Phases (Pipeline)

```
Phase 1          Phase 2          Phase 3          Phase 4
──────────       ──────────       ──────────       ──────────
Network          TraCI Manual     Random           Gymnasium
Generation   →   Control      →   Baseline     →   Environment
generate_        traci_manual_    random_          sumo_env.py
network.py       control.py       agent.py

Phase 5          Phase 6          Phase 7          Phase 9
──────────       ──────────       ──────────       ──────────
DQN Training     KPI Service      FastAPI          OSM Real
& Evaluation →   Collection   →   Backend      →   World Deploy
train_dqn.py     kpi_service.py   main.py          sumo_osm/
evaluate.py
```

Each phase is **cumulative**: Phase 2 reuses Phase 1's network, Phase 5 reuses Phase 4's env, Phase 9 reuses Phase 5's model.

---

## 3 — SUMO: Simulation of Urban MObility

### What is SUMO?

SUMO is a **free, open-source, microscopic traffic simulator** developed by DLR (German Aerospace Center). "Microscopic" means it simulates **individual vehicles** — their positions, speeds, acceleration, lane changes, and waiting times — second by second.

### The Network: A 3×3 Grid

```
A0 ── A1 ── A2
│     │     │
B0 ── B1 ── B2      ← B1 is the controlled intersection
│     │     │
C0 ── C1 ── C2
```

The center junction `B1` is the only **traffic-light junction** the RL agent controls. From `sumo_utils.py`:

```python
CONTROLLED_TL_ID = "B1"
B1_INCOMING_LANES = ("A1B1_0", "B0B1_0", "B2B1_0", "C1B1_0")
#                      West      South      North      East
```

### Vehicle Flows (routes.rou.xml)

```
Direction    Flow ID     Route              veh/hour
──────────   ─────────   ────────────────   ────────
N→S          flow_ns     B0_B1_B2           300
S→N          flow_sn     B2_B1_B0           300
E→W          flow_ew     A1_B1_C1           280
W→E          flow_we     C1_B1_A1           280
Turns NE/SW  flow_ne/sw  mixed              120 each
Turns NW/SE  flow_nw/se  mixed              100 each
                                            ────────
                                    Total: ~1,480 veh/h
```

Vehicle type `car`: accel=2.6 m/s², decel=4.5 m/s², sigma=0.5 (driver randomness), length=5m, maxSpeed=50 km/h.

### Traffic Light Phase States

Each phase is a **16-character string** where each character represents one of 16 connections at B1 (4 incoming × 4 possible turns):

```
Phase 0 (NS Green):  "GGggrrrrGGggrrrr"
                      ││││    ││││
                      NS major/minor green, EW red

Phase 1 (NS Yellow): "yyyyrrrryyyyrrrr"   ← clearance

Phase 2 (EW Green):  "rrrrGGggrrrrGGgg"
                              ││││    ││││
                              EW major/minor green, NS red

Phase 3 (EW Yellow): "rrrryyyyrrrryyyy"   ← clearance
```

The RL agent **only selects phases 0 and 2** (green phases). It never manually sets yellow — SUMO handles yellow intervals automatically when the phase string changes.

### SUMO Configuration File

```xml
<!-- simulation.sumocfg -->
<input>
  <net-file value="intersection.net.xml"/>    ← Road network
  <route-files value="routes.rou.xml"/>       ← Vehicle flows
</input>
<time>
  <begin value="0"/>
  <end value="3600"/>        ← 1 hour simulation
  <step-length value="1"/>   ← 1 second per simulation step
</time>
```

---

## 4 — TraCI: How It Works (Full Deep Dive)

### The Protocol

TraCI (Traffic Control Interface) is a **client-server protocol** where:

- **Server** = SUMO process (runs as subprocess)
- **Client** = Python code (your scripts)
- **Transport** = TCP socket on localhost (usually port 8813)

```
Python Script                   SUMO Process
─────────────                   ─────────────
traci.start(cmd)   →→→ fork →→→ sumo -c sim.sumocfg --remote-port 8813
                    ◄── TCP ──►  [waiting for step command]

traci.simulationStep() →→→ "advance 1 second" →→→ SUMO computes physics
                       ←── response + data ←── vehicle states updated

traci.vehicle.getWaitingTime(vid) →→→ query →→→ lookup internal state
                                  ←── float ←── return value

traci.trafficlight.setRedYellowGreenState(tl, state) →→→ override TL
                                                    ←── ack ←── done

traci.close() →→→ "terminate" →→→ SUMO shuts down
```

### Key Point: Deterministic Control

SUMO **only advances time when Python tells it to** via `traci.simulationStep()`. This means Python is in full control — it can read any variable, change anything, then decide whether to advance time. This is what makes SUMO usable as an RL environment.

### TraCI Methods Used in This Project


| Method                                                 | Type      | Used In        | What it Returns                     |
| ------------------------------------------------------ | --------- | -------------- | ----------------------------------- |
| `traci.start(cmd)`                                     | Connect   | all scripts    | Opens TCP connection                |
| `traci.simulationStep()`                               | Advance   | all scripts    | Advances sim by 1s                  |
| `traci.simulation.getTime()`                           | Read      | all            | Current sim time (float)            |
| `traci.simulation.getMinExpectedNumber()`              | Read      | all            | Vehicles still expected (-1 = done) |
| `traci.simulation.getDepartedIDList()`                 | Read      | kpi_service.py | New vehicle IDs this step           |
| `traci.simulation.getArrivedIDList()`                  | Read      | kpi_service.py | Vehicles that finished trip         |
| `traci.vehicle.getIDList()`                            | Read      | all            | All current vehicle IDs             |
| `traci.vehicle.getWaitingTime(vid)`                    | Read      | all            | Seconds stopped (speed < 0.1 m/s)   |
| `traci.vehicle.getSpeed(vid)`                          | Read      | all            | Current speed m/s                   |
| `traci.vehicle.getPosition(vid)`                       | Read      | backend        | (x, y) tuple in SUMO coords         |
| `traci.lane.getLastStepVehicleNumber(lid)`             | Read      | all            | Vehicles on lane in last step       |
| `traci.lane.getLastStepHaltingNumber(lid)`             | Read      | all            | Stopped vehicles on lane            |
| `traci.trafficlight.setRedYellowGreenState(tl, state)` | **Write** | all            | Override TL phase                   |
| `traci.close()`                                        | Close     | all            | Terminates SUMO                     |


### How `_set_phase(action)` Works

```python
# From sumo_utils.py:
B1_PHASES = (
    "GGggrrrrGGggrrrr",  # index 0: NS green
    "yyyyrrrryyyyrrrr",  # index 1: NS yellow
    "rrrrGGggrrrrGGgg",  # index 2: EW green
    "rrrryyyyrrrryyyy",  # index 3: EW yellow
)
GREEN_PHASE_INDICES = (0, 2)  # only green phases

# action=0 → GREEN_PHASE_INDICES[0]=0 → B1_PHASES[0] = "GGggrrrrGGggrrrr"
# action=1 → GREEN_PHASE_INDICES[1]=2 → B1_PHASES[2] = "rrrrGGggrrrrGGgg"

def _set_phase(action: int) -> None:
    idx = GREEN_PHASE_INDICES[action]
    traci.trafficlight.setRedYellowGreenState(CONTROLLED_TL_ID, B1_PHASES[idx])
```

---

## 5 — Reinforcement Learning: MDP Formulation

### What is a Markov Decision Process (MDP)?

An MDP models sequential decision-making as a tuple **(S, A, R, P, γ)**:

```
S  = State space  (what the agent can observe)
A  = Action space (what the agent can do)
R  = Reward function (signal of how well it did)
P  = Transition function (how environment evolves)
γ  = Discount factor (weight of future vs present reward)
```

In this project:

```
────────────────────────────────────────────────────────────────
COMPONENT     VALUE                              CODE
────────────────────────────────────────────────────────────────
State (S)     4 lane vehicle counts              Box(0, 100, (4,), float32)
              [A1B1_0, B0B1_0, B2B1_0, C1B1_0]

Action (A)    Binary: 0=NS green, 1=EW green     Discrete(2)

Reward (R)    -Σ waitingTime(v) for all v        ≤ 0, minimize
              (negative total waiting time)

Transition(P) SUMO physics simulation            Implicit in env.step()

Discount (γ)  0.99                               DQN param: gamma=0.99

Episode       72 steps × 5s = 360 sim seconds   max_steps_per_episode=72
────────────────────────────────────────────────────────────────
```

### The RL Interaction Loop

```
┌─────────────────────────────────────────────────────────┐
│                  SumoEnv.step(action)                    │
│                                                          │
│  Agent          Environment (SumoEnv)       SUMO         │
│  ──────         ────────────────────        ────         │
│                                                          │
│  1. obs = [2, 5, 1, 3]   ← lane vehicle counts          │
│          │                                               │
│  2. Q(obs) → [−12.3, −9.1]   ← Q-network forward pass   │
│          │                                               │
│  3. action = argmax = 1  ← pick EW green                │
│          │                                               │
│  4.      │─────────────────────────────────► set phase 2 │
│          │                           (5×) simulationStep │
│          │                                               │
│  5.      │◄── obs=[1,4,2,1] reward=-47.3 ───────────────│
│          │                                               │
│  6. Store (s, a, r, s') in replay buffer                 │
│  7. Sample batch, compute Bellman loss, backprop          │
└─────────────────────────────────────────────────────────┘
```

### Why Negative Waiting Time as Reward?

The goal is to **minimize** vehicle waiting. RL maximizes reward, so by negating waiting time, the agent learns to prefer states where cars flow freely:

- **Bad state** (traffic jam): 50 cars waiting → reward = −50×30s = −1500
- **Good state** (smooth flow): 5 cars waiting → reward = −5×2s = −10

---

## 6 — DQN: The AI Model in Detail

### What is DQN (Deep Q-Network)?

DQN (Mnih et al., 2015, DeepMind) approximates the **optimal Q-function** `Q*(s, a)` which represents: *"The expected total future reward when taking action `a` in state `s` and then acting optimally thereafter."*

Instead of a lookup table (impractical for continuous states), DQN uses a **neural network** with parameters θ to approximate Q.

### Neural Network Architecture

```
Input Layer      Hidden Layer 1   Hidden Layer 2   Output Layer
────────────     ──────────────   ──────────────   ────────────
[lane_A1B1]  ─┐
[lane_B0B1]  ─┤─→ 64 neurons  →  64 neurons   →  Q(s, NS_green)
[lane_B2B1]  ─┤    + ReLU         + ReLU          Q(s, EW_green)
[lane_C1B1]  ─┘
  (4 inputs)     (4×64 = 256)    (64×64 = 4096)   (64×2 = 128)
```

Policy: `"MlpPolicy"` = Multi-Layer Perceptron
Architecture config: `net_arch=[64, 64]`

### Two Key Innovations of DQN

**1. Experience Replay Buffer** (`buffer_size=10_000`)

Instead of learning from each transition immediately (which would be correlated and unstable), DQN stores transitions `(s, a, r, s')` in a **circular buffer** and samples **random mini-batches**:

```
Buffer (10,000 slots):
┌──────┬──────┬──────┬──────┬──────┐
│(s,a,r│(s,a,r│(s,a,r│(s,a,r│ ...  │
│,s')₁ │,s')₂ │,s')₃ │,s')₄ │      │
└──────┴──────┴──────┴──────┴──────┘
      ↑ Random batch of 32 sampled for each update
```

**Why?** Consecutive traffic steps are highly correlated. Random sampling breaks this correlation, making gradient updates more stable.

**2. Target Network** (`target_update_interval=500`, `tau=1.0`)

Two networks: **online** (updated every 4 steps) and **target** (updated every 500 steps by hard copy):

```
Online Q-network (θ):   Updated every 4 steps via backprop
                                 ↓ copy every 500 steps
Target Q-network (θ̄):  Used to compute Q-targets → stable training
```

**Why?** Without the target network, you'd be chasing a moving target — the labels for training would change every update, causing oscillation or divergence.

### The Bellman Loss Function

```
Target:  y = r + γ × max_a' Q(s', a'; θ̄)   ← uses TARGET network
Loss:    L(θ) = E[(y - Q(s, a; θ))²]         ← MSE of online network
```

At `gamma=0.99` and `max_steps=72`, the agent effectively plans up to ~100 steps ahead (since `0.99^100 ≈ 0.37`).

### ε-Greedy Exploration

```
t=0           t=6,000        t=30,000
ε=1.0    →    ε=0.05    →    ε=0.05
(100%         (5%            (stays)
 random)       random)
           ↑
     exploration_fraction=0.2
     (20% of 30,000 = 6,000 steps for decay)
```

During training:

- With probability ε: random action (explore)
- With probability 1-ε: argmax Q(s, a) (exploit)

During evaluation (`deterministic=True`): always argmax.

### Hyperparameters Analysis


| Param                    | Value    | Why This Value                                          |
| ------------------------ | -------- | ------------------------------------------------------- |
| `learning_rate`          | 5e-4     | Conservative Adam LR — traffic is noisy, slow is stable |
| `buffer_size`            | 10,000   | ~138 full episodes stored — good diversity              |
| `learning_starts`        | 1,000    | Warm up buffer before any updates (~13 episodes)        |
| `batch_size`             | 32       | Standard for DQN, balances speed vs stability           |
| `gamma`                  | 0.99     | Near-myopic: values long-term waiting reduction         |
| `tau`                    | 1.0      | Hard copy (not Polyak averaging like in DDPG)           |
| `train_freq`             | 4        | Update every 4 env steps — not too frequent             |
| `target_update_interval` | 500      | ~6-7 full episodes between target syncs                 |
| `net_arch`               | [64, 64] | Small MLP — input is only 4 values, this is sufficient  |
| `total_timesteps`        | 30,000   | ~416 full episodes (30000/72=416)                       |


---

## 7 — Gymnasium Environment (`sumo_env.py`)

The `SumoEnv` class is the **bridge** between the RL algorithm and SUMO. It implements the Gymnasium API so any SB3 algorithm can use it.

### Class Structure

```python
class SumoEnv(gym.Env[np.ndarray, int]):
    # Observation: 4 lane counts, range [0, 100]
    observation_space = Box(low=0.0, high=100.0, shape=(4,), dtype=float32)
    
    # Action: 0 = NS green, 1 = EW green
    action_space = Discrete(2)
    
    def reset(seed=None) → (obs, info):
        # Starts SUMO, returns initial lane counts
        
    def step(action) → (obs, reward, terminated, truncated, info):
        # Sets TL phase, advances 5 sim steps, returns new state
        
    def close():
        # Closes TraCI connection
```

### Step Logic (5 Sim Seconds per Control Step)

```
env.step(action=1)  [EW Green requested]
    │
    ├─ _set_phase(1) → traci.trafficlight.setRedYellowGreenState("B1", "rrrrGGgg...")
    │
    ├─ for _ in range(5):
    │       traci.simulationStep()   ← advance SUMO by 1s
    │       if _is_done(): break     ← early stop check
    │
    ├─ obs = _get_lane_vehicle_counts()   → np.array([2, 1, 4, 3])
    ├─ reward = -_get_total_waiting_time()  → -47.3
    ├─ terminated = sim_time >= 360
    ├─ truncated = step_count >= 72
    └─ return obs, reward, terminated, truncated, info
```

### Why `control_interval=5`?

A minimum green phase of 5 seconds is realistic. Below this, frequent phase changes would create dangerous/inefficient traffic (constant yellow transitions). 5s × 72 steps = 360s (6 min simulated time).

---

## 8 — Training, Evaluation & Baselines

### Training Flow (`train_dqn.py`)

```
python rl/train_dqn.py --timesteps 30000

1. Create SumoEnv (control_interval=5, max_steps=72, sim_end=360)
2. Create DQN model with MlpPolicy
3. model.learn(30000):
   ├─ Episode 1:  env.reset() → SUMO starts
   │   Step 1:  ε=1.0 random → explore
   │   Step 2:  ε=0.99 mostly random
   │   ...
   │   Step 72: episode ends → total reward logged
   │   env.reset() → SUMO restarts for Episode 2
   ├─ After step 1000: start gradient updates
   ├─ Every 4 steps: sample 32 from buffer, compute loss, backprop
   ├─ Every 500 steps: copy online → target network
   └─ After 6000 steps: ε reaches 0.05 (final value)
4. model.save("rl/models/dqn_traffic_light")  → .zip file
```

### Three-Way Evaluation (`evaluate.py`)

```
Fixed-Time:   No phase override. SUMO's default program runs.
              Cycles: NS green (fixed) → yellow → EW green → yellow
              
Random:       Each control step: random.randint(0, 1)
              Equal chance NS or EW green, ignores traffic
              
RL (DQN):    model.predict(obs, deterministic=True)
              Uses trained Q-network, always picks argmax action
```

Output table:

```
============================================================
Comparison (lower waiting / higher reward is better)
============================================================
  Fixed-time   | mean_waiting:   XXX.X s | mean_queue: X.X | total_reward: -XXXXX
  Random       | mean_waiting:   XXX.X s | mean_queue: X.X | total_reward: -XXXXX
  RL (DQN)     | mean_waiting:   XXX.X s | mean_queue: X.X | total_reward: -XXXXX
============================================================
```

---

## 9 — FastAPI Backend & KPI Service

### Architecture

```
                   HTTP Client (Dashboard / curl)
                           │
                    ┌──────▼──────┐
                    │   FastAPI   │  uvicorn ASGI server
                    │   main.py   │  + CORS middleware
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    GET /state       POST /run        GET /kpis
    GET /phases      ──────────►      GET /network
    GET /vehicles    Background       GET /vehicles
           │         Thread           │
           │         │                │
           └────►  _lock  ◄────────────┘
                    │
              Shared State:
              _sim_time, _running
              _current_phase
              _lane_vehicle_counts
              _total_waiting
              _queue_length
              _avg_speed
              _vehicle_positions
              _current_collector
```

### Thread Safety Pattern

The background simulation thread runs SUMO via TraCI and calls `on_step()` callback after every simulation step. `on_step()` acquires the mutex lock to safely update all shared state variables. FastAPI handlers read those variables under the same lock:

```python
_lock = threading.Lock()

# Background thread writes:
def _on_step(step, phase, lane_counts, ...):
    with _lock:
        global _sim_time, _current_phase, ...
        _sim_time = step
        _current_phase = phase

# FastAPI reads:
@app.get("/state")
def get_state():
    with _lock:
        return {"sim_time": _sim_time, ...}
```

### KPI Collector (`kpi_service.py`)

The `KPICollector` class tracks detailed per-vehicle statistics:

```
Per step:
- Departed vehicles → record depart_time[vid] = step
- Arrived vehicles  → travel_time = step - depart_time[vid]
                      waiting_at_arrival = last recorded waiting time
- Phase switched?   → increment phase_switch_count
- total_waiting     → accumulate for average
- mean_speed        → accumulate for average

Final KPIs:
- average_waiting_time = mean(waiting_times_at_arrival)
- average_travel_time  = mean(travel_times)
- throughput           = n_arrived / sim_duration_hours  [veh/h]
- average_speed        = speed_sum / speed_count
- number_of_phase_switches
```

---

## 10 — Real-World OSM Pipeline (Phase 9)

```
Step 1: download_osm.py
   Overpass API query: nwr(south,west,north,east) in bbox
   Default: Paris area (48.84–48.86°N, 2.34–2.36°E)
   → area.osm.xml

Step 2: build_net.py
   netconvert --osm-files area.osm.xml --output area.net.xml
   Flags: --geometry.remove (simplify), --junctions.join,
          --tls.guess-signals, --tls.join
   → area.net.xml (SUMO network with real streets)

Step 3: generate_routes.py
   randomTrips.py -n area.net.xml -o area.trips.xml -e 3600 -p 2.0
   (one new vehicle every 2 seconds on average)
   duarouter -n area.net.xml -t area.trips.xml -o area.rou.xml
   → area.rou.xml (vehicle routes via Dijkstra)

Step 4: detect_intersection.py
   Parses area.net.xml, finds junctions with type="traffic_light"
   Selects one with most incoming lanes (best for control)
   Extracts: tl_id, incoming_lanes, phases (from tlLogic XML), green_indices
   → osm_intersection_config.json

Step 5: run_rl_agent.py
   Loads DQN model trained on synthetic grid
   Pads/truncates observation to 4 values (matches trained net input size)
   Runs same RL policy on real intersection
   obs = counts[:4] padded with 0s → model.predict(obs) → action → setRedYellowGreenState
```

This is **transfer learning** — the policy trained on a 3×3 synthetic grid generalizes to real-world intersections because the observation (lane counts) and the core problem (NS vs EW green) are the same.

---

## 11 — Python Libraries Explained

### `gymnasium` (≥0.29.0)

The **standard API contract** for RL environments. Any algorithm that understands Gymnasium can use any environment that implements it. In this project:

- `SumoEnv` **inherits** from `gym.Env[np.ndarray, int]`
- Declares `observation_space` and `action_space` so SB3 knows input/output shapes
- Implements `reset()` and `step()` — that's all SB3 needs

### `stable-baselines3` (≥2.0.0)

PyTorch-based library of battle-tested RL algorithms. DQN here uses:

- `**MlpPolicy`** — builds the neural network architecture automatically from `net_arch`
- `**model.learn(n)**` — full training loop: collect → buffer → sample → loss → update
- `**model.save(path)**` / `**DQN.load(path)**` — serializes model to `.zip` (weights + hyperparams + normalization stats)
- `**model.predict(obs, deterministic=True)**` — inference, returns `(action, state)` tuple

### `traci` (from SUMO tools)

Not on PyPI. Found via:

```python
sys.path.insert(0, os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci
```

Implements the TraCI protocol. All calls are **synchronous** — Python blocks until SUMO responds.

### `numpy` (≥1.24.0)

Used to build the `float32` observation arrays that Gymnasium and SB3 expect:

```python
obs = np.array([traci.lane.getLastStepVehicleNumber(l)
                for l in B1_INCOMING_LANES], dtype=np.float32)
# SB3 DQN expects Box(dtype=float32) — exact dtype must match
```

### `fastapi` (≥0.100.0)

Modern Python web framework with:

- **Type annotations** → automatic OpenAPI docs at `/docs`
- **Pydantic** for request validation (`RunRequest` model)
- **CORS middleware** for cross-origin dashboard access
- **StaticFiles** to serve the frontend at `/dashboard`

### `uvicorn[standard]` (≥0.22.0)

ASGI server (Asynchronous Server Gateway Interface). The `[standard]` extra adds `httptools` (faster HTTP parsing) and `uvloop` (faster event loop on Linux). Run with `--reload` for development auto-restart.

### `threading` (stdlib)

Critical for the backend: SUMO simulation must run in a background thread so FastAPI can keep serving requests. Key pattern:

```python
# Daemon thread: automatically killed when main process exits
thread = threading.Thread(target=_run_simulation, daemon=True)
thread.start()
```

The `threading.Lock()` prevents race conditions on shared state between the SUMO thread and FastAPI request handlers.

### `xml.etree.ElementTree` (stdlib)

Used to parse SUMO's XML files at runtime:

1. `backend/main.py` — parses `intersection.net.xml` to extract lane shapes (as `[[y,x],...]` coordinate pairs for Leaflet.js)
2. `sumo_osm/detect_intersection.py` — parses OSM net XML to find `junction[@type='traffic_light']` elements and extract `tlLogic/phase` states

---

## Visual Summary: Complete Data Flow

```
╔══════════════════════════════════════════════════════════════════╗
║                    FULL SYSTEM DATA FLOW                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  routes.rou.xml + intersection.net.xml                           ║
║          │                                                       ║
║          ▼                                                       ║
║   ┌─────────────┐   traci.start()   ┌────────────────────────┐  ║
║   │    SUMO      │◄─────────────────│   SumoEnv.reset()      │  ║
║   │  Simulator   │                  └────────────────────────┘  ║
║   │              │                                               ║
║   │  Physics:    │   traci.simulationStep() × 5                 ║
║   │  - car-      │◄─────────────────────────────────────────── ║
║   │    following │                                               ║
║   │  - TL states │                                               ║
║   │  - waiting   │──────────────────────────────────────────── ║
║   │    times     │   lane_counts, waiting_time, speed          ║
║   └─────────────┘                                               ║
║          │                          │                            ║
║          │                          ▼                            ║
║          │                ┌────────────────────────┐            ║
║          │                │     SumoEnv.step()     │            ║
║          │                │  obs = lane_counts     │            ║
║          │                │  reward = -waiting     │            ║
║          │                └────────────┬───────────┘            ║
║          │                             │                         ║
║          │                             ▼                         ║
║          │                ┌────────────────────────┐            ║
║          │                │   DQN (Stable-SB3)     │            ║
║          │                │   Q([2,5,1,3]) →       │            ║
║          │                │   [-12.3, -9.1]        │            ║
║          │                │   action = 1 (EW green)│            ║
║          │                └────────────┬───────────┘            ║
║          │                             │                         ║
║          │  setRedYellowGreenState()   │                         ║
║          │◄────────────────────────────┘                         ║
║          │                                                       ║
║          │           KPICollector                                ║
║          │──────────────────────────────► avg_wait, throughput  ║
║          │                                travel_time, speed    ║
║          │                                                       ║
║          │           FastAPI /state /kpis /phases               ║
║          └──────────────────────────────► HTTP JSON responses   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

This project is a **textbook implementation** of applied RL for traffic engineering. The key architectural insight is the separation of concerns: SUMO handles physics, TraCI handles communication, Gymnasium standardizes the interface, SB3 handles the learning algorithm, and FastAPI handles observability. Each layer is independently replaceable — for example, you could swap DQN for PPO or A2C just by changing one line in `train_dqn.py`.