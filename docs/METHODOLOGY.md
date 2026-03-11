# Methodology of Working in the Traffic-RL Project

Here is a practical methodology for working with this project: setup, development flow, experimentation, and maintenance.

---

## 1. Prerequisites and Setup

```
┌─────────────────────────────────────────────────────────────┐
│  BEFORE ANY WORK                                             │
├─────────────────────────────────────────────────────────────┤
│  1. Install Python 3.10+                                     │
│  2. Install Eclipse SUMO (set SUMO_HOME or add bin to PATH)  │
│  3. Create venv: python -m venv .venv && .venv\\Scripts\\activate│
│  4. pip install -r requirements.txt                          │
│  5. Verify: python sumo/generate_network.py  (creates net)   │
└─────────────────────────────────────────────────────────────┘
```

Always run commands from the project root (`traffic-rl/`). The code assumes this for imports and paths.

---

## 2. Phase-Based Development Flow

The project is built in phases. Work in this order:

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6–7 ──► Phase 9
  │           │           │           │           │
  │           │           │           │           └── Train & evaluate DQN
  │           │           │           └── Gymnasium env (SumoEnv)
  │           │           └── Random baseline (KPIs, logging)
  │           └── TraCI + manual TL control
  └── Network generation (netgenerate / netconvert)
```

Methodology:

1. Validate each phase before moving on.
2. Use the GUI for debugging (e.g. `-gui`).
3. Keep the synthetic grid working before changing OSM or real-world setups.

---

## 3. Typical Workflows

### A. First-Time Setup and Sanity Check

```
1. python sumo/generate_network.py          # Create intersection.net.xml
2. cd sumo && sumo-gui -c simulation.sumocfg # Visual check
3. python sumo/traci_manual_control.py --gui --steps 60  # TraCI check
4. python rl/random_agent.py --steps 60      # Random agent check
5. python rl/sumo_env.py                     # Env test (built-in)
6. python rl/train_dqn.py --timesteps 5000   # Short training test
7. python rl/evaluate.py --no-rl             # Fixed vs Random
8. python rl/evaluate.py                      # Full comparison (needs model)
```

### B. Training and Evaluation

```
1. Train:  python rl/train_dqn.py [--timesteps 30000] [--save-path ...]
2. Eval:   python rl/evaluate.py [--model path] [--steps 72] [--sim-end 360]
3. Compare: Fixed-time vs Random vs RL (lower waiting = better)
```

### C. API and Dashboard

```
1. uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
2. Open <http://127.0.0.1:8000/docs>
3. POST /run with {"controller":"rl","sim_end":120}
4. GET /state, GET /kpis, GET /phases while running
5. Open <http://127.0.0.1:8000/dashboard> for live view
```

### D. OSM Real-World Pipeline

```
1. python sumo_osm/download_osm.py [--south --west --north --east]
2. python sumo_osm/build_net.py
3. python sumo_osm/detect_intersection.py
4. python sumo_osm/generate_routes.py
5. python sumo_osm/run_rl_agent.py [--model rl/models/dqn_traffic_light.zip]
```

---

## 4. Development Practices

| Practice | How to Apply |
| --- | --- |
| Start small | Use `--steps 60`, `--sim-end 120`, `--timesteps 5000` for quick tests |
| Use seeds | `--seed 42` for reproducible runs |
| GUI for debugging | Add `--gui` when inspecting behavior |
| Check KPIs | Use `random_agent.py --csv` or `evaluate.py` to compare controllers |
| Keep baseline | Always compare new changes to Fixed-time and Random |
| One TraCI connection | Do not run multiple SUMO simulations in parallel (single TraCI) |

---

## 5. Where to Change Things

| Goal | Where to Edit |
| --- | --- |
| Network layout | `sumo/intersection.nod.xml`, `intersection.edg.xml` or `generate_network.py` |
| Traffic demand | `sumo/routes.rou.xml` (flows, vehsPerHour) |
| TL phases | `rl/sumo_utils.py` (B1_PHASES, GREEN_PHASE_INDICES) |
| Observation | `rl/sumo_env.py` (`_get_obs()`, `observation_space`) |
| Reward | `rl/sumo_env.py` (reward in `step()`) |
| DQN hyperparameters | `rl/train_dqn.py` (learning_rate, buffer_size, etc.) |
| Control interval | `SumoEnv(control_interval=5)` or `--interval` in scripts |
| API endpoints | `backend/main.py` |
| KPIs | `backend/kpi_service.py` (KPICollector) |

---

## 6. Testing Strategy

```
Level 1 — Unit-style checks
├── python rl/sumo_env.py           # Env reset/step
├── python backend/kpi_service.py --controller fixed --sim-end 30
└── python rl/random_agent.py --steps 20 --log-every 5

Level 2 — Integration
├── python rl/train_dqn.py --timesteps 1000
├── python rl/evaluate.py --no-rl
└── curl POST /run → GET /state → GET /kpis

Level 3 — Full pipeline
├── Train 30k steps → evaluate → compare table
└── OSM: download → build → detect → routes → run_rl_agent
```

---

## 7. Troubleshooting Order

1. **SUMO not found** → Set `SUMO_HOME` or add `sumo/bin` to `PATH`
2. **TraCI import error** → `add_sumo_to_path()` before `import traci`
3. **Config not found** → Run from project root; `sumo/simulation.sumocfg` must exist
4. **Model not found** → Run `python rl/train_dqn.py` first
5. **OSM net empty** → Check bbox and Overpass API; try another area
6. **API 404 on /kpis** → Start a simulation with `POST /run` first

---

## 8. Summary Workflow Diagram

```
                    ┌──────────────────┐
                    │  Prerequisites   │
                    │  Python, SUMO,    │
                    │  pip install     │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Phase 1–2       │
                    │ Generate net,    │
                    │ TraCI manual     │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Phase 3–4       │
                    │ Random agent,    │
                    │ SumoEnv          │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Phase 5         │
                    │ Train DQN,       │
                    │ Evaluate         │
                    └────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
┌────────▼────────┐ ┌────────▼────────┐ ┌───────▼───────┐
│ Phase 6–7       │ │ Phase 9          │ │ Iterate      │
│ KPI + FastAPI   │ │ OSM pipeline     │ │ Change obs,  │
│ + Dashboard     │ │ Real-world       │ │ reward, HPs   │
└─────────────────┘ └──────────────────┘ └───────────────┘
```

In short: set up prerequisites, validate each phase in order, run short tests first, use seeds and baselines, and change only one thing at a time when experimenting.