#!/usr/bin/env python3
"""
Phase 9: Run the trained DQN agent on the OSM net.

Loads osm_intersection_config.json (tl_id, incoming_lanes, green_phase_indices),
starts SUMO with the OSM net and routes, and runs the same RL policy:
observation = lane vehicle counts (padded to 4 for the trained DQN), action = phase choice.

Usage:
    python sumo_osm/run_rl_agent.py [--config osm_intersection_config.json] [--model ../rl/models/dqn_traffic_light.zip] [--sim-end 360]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# SUMO TraCI
from rl.sumo_utils import add_sumo_to_path
add_sumo_to_path()
import traci

import numpy as np
from stable_baselines3 import DQN


def find_sumo_bin() -> str | None:
    from rl.sumo_utils import find_sumo_bin as f
    return f(False)


def run_osm_rl(
    net_path: Path,
    route_path: Path,
    config_path: Path,
    model_path: Path,
    sim_end: int = 360,
    control_interval: int = 5,
) -> None:
    with open(config_path) as f:
        cfg = json.load(f)
    tl_id = cfg["tl_id"]
    incoming_lanes = cfg["incoming_lanes"]
    phases = cfg["phases"]
    green_indices = cfg["green_phase_indices"]
    if len(green_indices) < 2:
        green_indices = [0, 1] if len(phases) >= 2 else [0, 0]

    model = DQN.load(str(model_path))
    sumo_bin = find_sumo_bin()
    if not sumo_bin:
        raise RuntimeError("SUMO not found. Set SUMO_HOME.")
    sumo_cmd = [
        sumo_bin,
        "-n", str(net_path),
        "-r", str(route_path),
        "--no-step-log",
        "--no-warnings",
        "--end", str(sim_end),
    ]
    os.chdir(SCRIPT_DIR)
    traci.start(sumo_cmd)

    step = 0.0
    n_steps = 0
    total_reward = 0.0
    try:
        while step < sim_end:
            # Observation: lane vehicle counts (pad/truncate to 4 for trained DQN)
            counts = []
            for lid in incoming_lanes:
                try:
                    counts.append(traci.lane.getLastStepVehicleNumber(lid))
                except Exception:
                    counts.append(0)
            while len(counts) < 4:
                counts.append(0)
            obs = np.array(counts[:4], dtype=np.float32)

            action, _ = model.predict(obs, deterministic=True)
            action = int(action) % 2
            phase_idx = green_indices[action]
            state = phases[phase_idx]
            traci.trafficlight.setRedYellowGreenState(tl_id, state)

            for _ in range(control_interval):
                traci.simulationStep()
                step = traci.simulation.getTime()
                if step >= sim_end:
                    break
                if traci.simulation.getMinExpectedNumber() < 0:
                    break

            total_waiting = sum(traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList())
            reward = -total_waiting
            total_reward += reward
            n_steps += 1
            if n_steps % 10 == 0:
                print(f"Step {n_steps} sim_time={step:.0f} total_waiting={total_waiting:.0f} reward={reward:.0f}")
    finally:
        traci.close()
    print(f"Done. Total reward={total_reward:.0f} over {n_steps} control steps.")


def main() -> int:
    p = argparse.ArgumentParser(description="Run DQN agent on OSM net")
    p.add_argument("--net", type=str, default=str(SCRIPT_DIR / "area.net.xml"))
    p.add_argument("--routes", type=str, default=str(SCRIPT_DIR / "area.rou.xml"))
    p.add_argument("--config", type=str, default=str(SCRIPT_DIR / "osm_intersection_config.json"))
    p.add_argument("--model", type=str, default=str(PROJECT_ROOT / "rl" / "models" / "dqn_traffic_light.zip"))
    p.add_argument("--sim-end", type=int, default=360)
    p.add_argument("--interval", type=int, default=5)
    args = p.parse_args()
    net_path = Path(args.net)
    route_path = Path(args.routes)
    config_path = Path(args.config)
    model_path = Path(args.model)
    if not net_path.is_absolute():
        net_path = SCRIPT_DIR / net_path
    if not route_path.is_absolute():
        route_path = SCRIPT_DIR / route_path
    if not config_path.is_absolute():
        config_path = SCRIPT_DIR / config_path
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    for name, path in [("net", net_path), ("routes", route_path), ("config", config_path), ("model", model_path)]:
        if not path.exists():
            print(f"Not found: {path}", file=sys.stderr)
            return 1
    run_osm_rl(net_path, route_path, config_path, model_path, sim_end=args.sim_end, control_interval=args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
