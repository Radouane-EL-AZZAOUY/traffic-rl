# Smart Traffic Light Optimization

A Smart Traffic Light Optimization project using **SUMO**, **Reinforcement Learning** (Gymnasium + Stable-Baselines3), **FastAPI**, and **Leaflet/MapLibre** dashboard.

Pipeline: **OSM → SUMO → TraCI → RL Agent → Traffic Light Control → KPIs → FastAPI → Leaflet + Charts**

📖 **Detailed guide:** [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) — algorithms, DQN training, visual workflows, analogies, customization examples, and how to test and interpret results.

---

## Project structure

```
traffic-rl/
├── sumo/                    # SUMO network and simulation
│   ├── intersection.net.xml
│   ├── routes.rou.xml
│   ├── simulation.sumocfg
│   ├── generate_network.py
│   ├── traci_manual_control.py   # Phase 2: TraCI + manual TL control
│   ├── intersection.nod.xml
│   └── intersection.edg.xml
├── rl/                      # RL agents and env
│   ├── sumo_utils.py       # TraCI path, SUMO binary, B1 constants
│   ├── random_agent.py      # Phase 3: random TL controller + KPI log
│   ├── sumo_env.py         # (Phase 4) Gymnasium env
│   ├── train_dqn.py        # (Phase 5) DQN training
│   └── evaluate.py         # (Phase 5) Evaluation
├── backend/                 # KPI service + FastAPI
│   ├── kpi_service.py       # Phase 6: KPI collection → JSON
│   └── main.py              # (Phase 7) FastAPI
├── frontend/                # Phase 8: Dashboard
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── sumo_osm/                # Phase 9: OSM → SUMO → RL
│   ├── download_osm.py      # Download OSM (Overpass) → area.osm.xml
│   ├── build_net.py        # netconvert OSM → area.net.xml
│   ├── detect_intersection.py  # One TL junction → osm_intersection_config.json
│   ├── generate_routes.py   # randomTrips + duarouter → area.rou.xml
│   ├── run_rl_agent.py     # Run DQN on OSM net with detected intersection
│   ├── area.sumocfg        # SUMO config for OSM scenario
│   └── osm_intersection_config.json  # (generated) tl_id, lanes, phases
└── requirements.txt
```

---

## Prerequisites

- **Python 3.10+**
- **SUMO** (Eclipse SUMO) installed and either:
  - `SUMO_HOME` set (e.g. `C:\Program Files (x86)\Eclipse Sumo`), or
  - SUMO `bin` folder in your `PATH`

---

## Phase 1 — Simple intersection simulation

Phase 1 provides a **4-way intersection** (3×3 grid, one central traffic-light junction) and vehicle flows so you can run the simulation in **sumo-gui**.

### 1. Generate the network (if needed)

If `sumo/intersection.net.xml` is missing or you want to regenerate it:

```bash
# From project root
python sumo/generate_network.py
```

This uses SUMO’s **netgenerate** (grid 3×3, 200 m links, traffic lights). If `netgenerate` is not available, the script falls back to **netconvert** using `intersection.nod.xml` and `intersection.edg.xml` (plain XML).

### 2. Run the simulation

**With GUI (recommended for Phase 1):**

```bash
cd sumo
sumo-gui -c simulation.sumocfg
```

**Without GUI:**

```bash
cd sumo
sumo -c simulation.sumocfg
```

- Simulation time: 0–3600 s (1 hour).
- Traffic: flows from all four approaches through the center junction (B1) with default fixed-time traffic lights.

### 3. Files provided

| File | Description |
|------|-------------|
| `intersection.net.xml` | Road network (3×3 grid, one 4-way TL at center). |
| `routes.rou.xml` | Vehicle type, routes through center, and flows (veh/h). |
| `simulation.sumocfg` | SUMO config: net, routes, duration, step length. |
| `generate_network.py` | Script to (re)generate `intersection.net.xml` via netgenerate or netconvert. |
| `intersection.nod.xml` / `intersection.edg.xml` | Plain XML sources for netconvert fallback. |

---

## Phase 2 — Manual traffic light control with TraCI

Phase 2 provides a Python script that connects to SUMO via **TraCI**, prints **lane vehicle counts** for the center intersection (B1), and **manually switches** traffic light phases (NS green ↔ EW green every 30 s).

### Run Phase 2

From the project root:

```bash
# Headless (default: 300 steps)
python sumo/traci_manual_control.py

# Limit steps (e.g. 60)
python sumo/traci_manual_control.py --steps 60

# With sumo-gui
python sumo/traci_manual_control.py --gui --steps 120
```

- **Lane counts:** Printed each step for B1 incoming lanes (`A1B1_0`, `B0B1_0`, `B2B1_0`, `C1B1_0`).
- **Phase switch:** Every 30 simulation seconds the script sets the TL state to NS green or EW green (cycle).

Requires SUMO installed and `SUMO_HOME` set (or SUMO `bin` in `PATH`). The script adds SUMO’s `tools` directory to `sys.path` for the `traci` module.

---

## Phase 3 — Random agent

Phase 3 provides a **random traffic light controller**: at each control step it randomly chooses NS green or EW green, and **logs** per step:

- **Waiting time** — total waiting time (s) over all vehicles
- **Queue length** — number of halting vehicles on B1 incoming lanes
- **Average speed** — mean speed (m/s) over all vehicles

### Run Phase 3

From the project root:

```bash
# Default: 360 steps, phase change every 5 s, log every 10 steps
python rl/random_agent.py

# Shorter run, custom log interval
python rl/random_agent.py --steps 120 --log-every 20

# Reproducible (fixed seed), write CSV
python rl/random_agent.py --steps 200 --seed 42 --csv rl/phase3_log.csv

# With sumo-gui
python rl/random_agent.py --gui --steps 60
```

**Options:** `--steps`, `--interval` (seconds between random phase changes), `--seed`, `--log-every`, `--csv` (output file path).

At the end the script prints a **summary**: steps, average waiting time, total waiting, average queue length, average speed. Optional CSV has columns: `step`, `waiting_time`, `queue_length`, `average_speed`.

---

## Phase 4 — Gymnasium environment

Phase 4 provides a **Gymnasium** environment (`SumoEnv`) for traffic light control:

- **Observation space:** `Box(0, 100, (4,), float32)` — vehicle count on each of the 4 B1 incoming lanes.
- **Action space:** `Discrete(2)` — 0 = North–South green, 1 = East–West green.
- **Reward:** Negative total waiting time (sum over all vehicles) at the end of each step, so the agent is trained to **minimize total waiting time**.

One env step = set phase → advance simulation by `control_interval` seconds (default 5).

### Usage

```python
# With project root in PYTHONPATH: PYTHONPATH=traffic-rl python ...
from rl.sumo_env import SumoEnv
# Or from inside rl/: from sumo_env import SumoEnv

env = SumoEnv(
    control_interval=5,       # sim seconds per step
    max_steps_per_episode=72,
    sim_end=360,
    use_gui=False,
)
obs, info = env.reset(seed=42)
action = env.action_space.sample()  # 0 or 1
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

### Test the environment

From the project root:

```bash
python rl/sumo_env.py
```

Runs a short episode (10 steps, 60 s sim) with random actions and prints observations and rewards.

**Dependencies:** `gymnasium`, `numpy` (see `requirements.txt`). SUMO must be installed (Phase 1–3).

---

## Phase 5 — DQN agent

Phase 5 adds **training** with Stable-Baselines3 **DQN**, **saving** the trained model, and an **evaluation script** that compares:

- **Fixed-time** — default SUMO static program (no phase override)
- **Random** — random phase each control step
- **RL (DQN)** — trained agent

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the DQN agent

From the project root:

```bash
# Default: 30k timesteps, saves to rl/models/dqn_traffic_light.zip
python rl/train_dqn.py

# Custom timesteps and save path
python rl/train_dqn.py --timesteps 50000 --save-path rl/models/dqn_custom

# With sumo-gui (slower, for debugging)
python rl/train_dqn.py --gui --timesteps 5000
```

Training uses `SumoEnv` and saves the final model as `.zip`. Default path: `rl/models/dqn_traffic_light.zip`. (Evaluation during training is skipped because SUMO uses a single TraCI connection.)

### Evaluate and compare controllers

```bash
# Compare Fixed-time, Random, and RL (requires trained model)
python rl/evaluate.py

# Use a specific model and run length
python rl/evaluate.py --model rl/models/dqn_traffic_light.zip --steps 72 --sim-end 360

# Compare only Fixed-time and Random (no model needed)
python rl/evaluate.py --no-rl
```

The script prints a table with **mean waiting time**, **mean queue length**, and **total reward** for each controller. Lower waiting time and higher total reward indicate better performance.

---

## Phase 6 — KPI collection system

Phase 6 provides a **KPI module** in `backend/kpi_service.py` that computes and returns metrics in **structured JSON**:

| KPI | Description |
|-----|-------------|
| **average_waiting_time** | Mean waiting time (s) per vehicle at arrival |
| **average_travel_time** | Mean travel time (s) per vehicle at arrival |
| **throughput** | Vehicles per hour (and **throughput_total** count) |
| **average_speed** | Time-averaged mean speed (m/s) |
| **number_of_phase_switches** | Count of traffic light phase changes |

Additional fields: `sim_duration_s`, `n_steps`, `n_arrived`.

### Usage

**1. Programmatic (dict or JSON string):**

```python
from backend.kpi_service import run_simulation_and_collect_kpis, get_kpis_json, KPICollector

# Run simulation and get dict
results = run_simulation_and_collect_kpis(
    sim_end=360,
    control_interval=5,
    controller="fixed",  # or "random", "rl"
    model_path="rl/models/dqn_traffic_light.zip",  # when controller="rl"
    seed=42,
)

# Or get JSON string
json_str = get_kpis_json(sim_end=120, controller="random")
```

**2. Command line (run from project root):**

```bash
# Fixed-time, 60 s simulation
python backend/kpi_service.py --controller fixed --sim-end 60

# Random controller
python backend/kpi_service.py --controller random --sim-end 90

# RL controller (requires trained model)
python backend/kpi_service.py --controller rl --sim-end 120 --model rl/models/dqn_traffic_light.zip
```

**3. KPICollector (step-by-step):** Instantiate `KPICollector()`, call `update(...)` each simulation step with departed/arrived IDs, waiting times, total waiting, mean speed, and phase_switched. Then `get_results()` or `to_json()`.

Requires SUMO and the `rl` package (for TraCI and, if using `controller="rl"`, stable-baselines3). Run from **project root** so that `backend` and `rl` are importable.

---

## Phase 7 — FastAPI backend

Phase 7 provides a **FastAPI** backend that runs SUMO in the **background** and serves **real-time** state and KPIs.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| **GET** | `/kpis` | Latest KPIs (from current run so far, or last completed run). 404 if no run yet. |
| **GET** | `/state` | Current state: `running`, `sim_time`, `lane_vehicle_counts`, `total_waiting`, `queue_length`, `avg_speed`. |
| **GET** | `/phases` | Current traffic light phase: `tl_id` (B1), `current_phase`, `running`. |
| **POST** | `/run` | Start a simulation in the background. Body: `controller`, `sim_end`, `control_interval`, `model_path`, `seed`. |

### Run the backend

From the **project root**:

```bash
# Install deps if needed
pip install -r requirements.txt

# Start the API (default: http://127.0.0.1:8000)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
# or: python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Interactive docs: **http://127.0.0.1:8000/docs**

### Example usage

```bash
# Get current state (no run yet)
curl http://127.0.0.1:8000/state

# Start a 60 s fixed-time simulation
curl -X POST http://127.0.0.1:8000/run -H "Content-Type: application/json" -d "{\"controller\":\"fixed\",\"sim_end\":60}"

# Poll state while running
curl http://127.0.0.1:8000/state
curl http://127.0.0.1:8000/phases

# Get KPIs after (or during) the run
curl http://127.0.0.1:8000/kpis
```

**POST /run** body (all optional): `controller` (default `"fixed"`), `sim_end` (default 360), `control_interval` (default 5), `model_path` (for `controller="rl"`), `seed` (default 42). The simulation runs in a background thread; use **GET /state** and **GET /kpis** for live or final data.

Requires SUMO and the `rl` package (TraCI). Run from **project root** so `backend` and `rl` are importable.

---

## Phase 8 — Frontend dashboard

Phase 8 provides a **Leaflet + Chart.js** dashboard that shows the simulation state and KPIs in real time.

### Features

- **Leaflet map** — SUMO network area (0–400 m) with **live vehicle markers** (positions from **GET /vehicles**).
- **Charts (Chart.js)** — Over time: **waiting time**, **average speed**, **reward** (≈ −waiting).
- **Current traffic light phase** — Display from **GET /phases** (e.g. NS green / EW green).
- **Controls** — Start simulation (**POST /run**) with controller (fixed / random / RL) and sim end time.

### Run the dashboard

1. Start the backend (from project root):
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

2. Open the dashboard in a browser:
   - **From the API:** http://127.0.0.1:8000/dashboard  
   - **Or open file:** `frontend/index.html` (ensure backend is running on port 8000; the frontend uses `API_BASE = "http://127.0.0.1:8000"`).

3. Click **Start simulation** and watch the map (vehicle markers), charts, and phase badge update.

The dashboard polls **/state**, **/phases**, and **/vehicles** every 500 ms while a simulation is running. Charts accumulate points over time; the phase badge shows the current B1 phase (e.g. green for NS or EW).

---

## Phase 9 — OSM map and RL agent

Phase 9 uses a **real OSM area**: download map → build SUMO net → detect one traffic-light intersection → generate routes → run the **same trained DQN** on that intersection. The agent expects **4-dimensional observations** (lane vehicle counts); the OSM intersection’s incoming lanes are **padded or truncated to 4** so the existing model can be used without retraining.

### 1. Download OSM and build the net

From the **project root**:

```bash
# Download OSM (default bbox; override with --bbox min_lat min_lon max_lat max_lon)
python sumo_osm/download_osm.py

# Convert OSM to SUMO net (produces sumo_osm/area.net.xml)
python sumo_osm/build_net.py
```

### 2. Detect one intersection and generate config

```bash
# Detect one TL junction, write osm_intersection_config.json (tl_id, incoming_lanes, phases, green_phase_indices)
python sumo_osm/detect_intersection.py
```

### 3. Generate routes

Requires **SUMO** (`SUMO_HOME` set) so that `tools/randomTrips.py` and `bin/duarouter` are available:

```bash
# Random trips + duarouter → area.rou.xml (default end time 3600 s)
python sumo_osm/generate_routes.py

# Shorter run
python sumo_osm/generate_routes.py --end 600
```

### 4. Run the RL agent on the OSM net

Uses the **trained DQN** from Phase 5 (`rl/models/dqn_traffic_light.zip`). Observation = lane counts (padded to 4); action = which green phase to show.

```bash
# Default: area.net.xml, area.rou.xml, osm_intersection_config.json, sim end 360 s
python sumo_osm/run_rl_agent.py

# Custom paths and sim length
python sumo_osm/run_rl_agent.py --net sumo_osm/area.net.xml --routes sumo_osm/area.rou.xml --config sumo_osm/osm_intersection_config.json --model rl/models/dqn_traffic_light.zip --sim-end 600
```

Train the model first with `python rl/train_dqn.py` if `dqn_traffic_light.zip` is missing.

### Phase 9 files

| File | Description |
|------|-------------|
| `sumo_osm/download_osm.py` | Downloads OSM data via Overpass API → `area.osm.xml`. |
| `sumo_osm/build_net.py` | Runs netconvert on OSM → `area.net.xml`. |
| `sumo_osm/detect_intersection.py` | Parses net for one traffic_light junction; writes `osm_intersection_config.json` (tl_id, incoming_lanes, phases, green_phase_indices for 2-action RL). |
| `sumo_osm/generate_routes.py` | randomTrips.py + duarouter → `area.rou.xml` (and `area.trips.xml`). |
| `sumo_osm/run_rl_agent.py` | Loads config + DQN, runs SUMO with OSM net/routes, controls TL via the trained agent (lane counts padded to 4). |
| `sumo_osm/area.sumocfg` | SUMO config for manual runs (net + routes, end time). |
