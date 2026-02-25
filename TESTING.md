# Testing all phases — command checklist

Run all commands from the **project root** (`traffic-rl/`), unless noted.

**Prerequisites:** Python 3.10+, SUMO installed and `SUMO_HOME` set (or SUMO `bin` in PATH). For Phases 5–9: `pip install -r requirements.txt`.

---

## Phase 1 — Simple intersection

| Step | Command | What to expect |
|------|---------|----------------|
| 1 | `python sumo/generate_network.py` | (Optional) Creates/updates `sumo/intersection.net.xml`. |
| 2 | `cd sumo` then `sumo-gui -c simulation.sumocfg` | GUI opens; vehicles run through the 3×3 grid for 1 hour. |
| 3 | (Alternative headless) `cd sumo` then `sumo -c simulation.sumocfg` | Same run without GUI. |

---

## Phase 2 — TraCI manual control

| Step | Command | What to expect |
|------|---------|----------------|
| 1 | `python sumo/traci_manual_control.py` | Headless: lane counts printed, phase switches every 30 s (default 300 steps). |
| 2 | `python sumo/traci_manual_control.py --steps 60` | Shorter run (60 steps). |
| 3 | `python sumo/traci_manual_control.py --gui --steps 120` | SUMO GUI; manual phase control. |

---

## Phase 3 — Random agent

| Step | Command | What to expect |
|------|---------|----------------|
| 1 | `python rl/random_agent.py` | Random phase every 5 s; logs waiting time, queue length, avg speed (default 360 steps). |
| 2 | `python rl/random_agent.py --steps 120 --seed 42 --csv rl/phase3_log.csv` | Reproducible run + CSV output. |
| 3 | `python rl/random_agent.py --gui --steps 60` | Same with SUMO GUI. |

---

## Phase 4 — Gymnasium environment

| Step | Command | What to expect |
|------|---------|----------------|
| 1 | `python rl/sumo_env.py` | Short episode (10 steps, 60 s sim) with random actions; prints obs and reward. |

---

## Phase 5 — DQN agent

| Step | Command | What to expect |
|------|---------|----------------|
| 1 | `pip install -r requirements.txt` | Install gymnasium, numpy, stable-baselines3. |
| 2 | `python rl/train_dqn.py` | Trains DQN (~30k steps), saves `rl/models/dqn_traffic_light.zip`. |
| 3 | `python rl/evaluate.py` | Compares fixed-time, random, and RL; prints waiting time, queue, reward. |
| 4 | (Optional) `python rl/evaluate.py --no-rl` | Compare only fixed-time vs random (no model needed). |

---

## Phase 6 — KPI collection

| Step | Command | What to expect |
|------|---------|----------------|
| 1 | `python backend/kpi_service.py --controller fixed --sim-end 60` | Runs 60 s fixed-time; prints KPI JSON. |
| 2 | `python backend/kpi_service.py --controller random --sim-end 90` | Random controller; KPI JSON. |
| 3 | `python backend/kpi_service.py --controller rl --sim-end 120 --model rl/models/dqn_traffic_light.zip` | RL controller; needs trained model. |

---

## Phase 7 — FastAPI backend

| Step | Command | What to expect |
|------|---------|----------------|
| 1 | `uvicorn backend.main:app --host 0.0.0.0 --port 8000` | API running at http://127.0.0.1:8000. |
| 2 | (New terminal) `curl http://127.0.0.1:8000/state` | JSON state (or 404 if no run yet). |
| 3 | `curl -X POST http://127.0.0.1:8000/run -H "Content-Type: application/json" -d "{\"controller\":\"fixed\",\"sim_end\":60}"` | Starts 60 s fixed-time simulation in background. |
| 4 | `curl http://127.0.0.1:8000/state` then `curl http://127.0.0.1:8000/kpis` | Poll state and KPIs. |
| 5 | Open http://127.0.0.1:8000/docs | Interactive API docs. |

---

## Phase 8 — Dashboard

| Step | Command | What to expect |
|------|---------|----------------|
| 1 | `uvicorn backend.main:app --host 0.0.0.0 --port 8000` | Start backend (if not already). |
| 2 | Open http://127.0.0.1:8000/dashboard in browser | Leaflet map, charts, phase badge, Start simulation. |
| 3 | Click **Start simulation** (e.g. fixed, 60 s) | Map and charts update; vehicles and phase shown. |

---

## Phase 9 — OSM and RL on real net

| Step | Command | What to expect |
|------|---------|----------------|
| 1 | `python sumo_osm/download_osm.py` | Downloads OSM → `sumo_osm/area.osm.xml`. |
| 2 | `python sumo_osm/build_net.py` | Converts OSM → `sumo_osm/area.net.xml`. |
| 3 | `python sumo_osm/detect_intersection.py` | Writes `sumo_osm/osm_intersection_config.json`. |
| 4 | `python sumo_osm/generate_routes.py` | Creates `sumo_osm/area.rou.xml` (needs SUMO tools). |
| 5 | `python sumo_osm/run_rl_agent.py` | Runs DQN on OSM net (needs `rl/models/dqn_traffic_light.zip` from Phase 5). |

**Note:** Phase 9 step 5 requires a trained model; run Phase 5 step 2 first if needed.

---

## Quick full pipeline (minimal test)

Copy-paste sequence from project root (one phase per block):

```bash
# Phase 1
python sumo/generate_network.py
cd sumo && sumo -c simulation.sumocfg && cd ..

# Phase 2
python sumo/traci_manual_control.py --steps 30

# Phase 3
python rl/random_agent.py --steps 30

# Phase 4
python rl/sumo_env.py

# Phase 5
pip install -r requirements.txt
python rl/train_dqn.py --timesteps 5000
python rl/evaluate.py --steps 24

# Phase 6
python backend/kpi_service.py --controller fixed --sim-end 30

# Phase 7 (start server; test in another terminal)
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Phase 8 — open http://127.0.0.1:8000/dashboard and click Start simulation

# Phase 9 (after OSM net + routes exist)
python sumo_osm/download_osm.py
python sumo_osm/build_net.py
python sumo_osm/detect_intersection.py
python sumo_osm/generate_routes.py
python sumo_osm/run_rl_agent.py --sim-end 120
```
