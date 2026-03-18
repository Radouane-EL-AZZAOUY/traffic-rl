#!/usr/bin/env python3
"""
Evaluation script for the from-scratch DQN model.

Mirrors rl/evaluate.py: runs Fixed-time, Random, and RL (from-scratch DQN)
controllers and prints a comparison table of KPIs.

Run from project root:
    python rl/from_scratch/evaluate_scratch.py
    python rl/from_scratch/evaluate_scratch.py --model rl/models/dqn_traffic_light_scratch.npz --steps 72
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

# Allow importing sumo_env / sumo_utils from rl/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sumo_utils import (
    B1_INCOMING_LANES,
    B1_PHASES,
    CONTROLLED_TL_ID,
    GREEN_PHASE_INDICES,
    SUMO_CONFIG,
    SUMO_DIR,
    add_sumo_to_path,
    find_sumo_bin,
)

add_sumo_to_path()

import numpy as np
import traci

from from_scratch.dqn_numpy import Network


# ---------------------------------------------------------------------------
# Helpers shared by all controllers
# ---------------------------------------------------------------------------

def _get_lane_counts() -> np.ndarray:
    counts = [
        traci.lane.getLastStepVehicleNumber(lane_id)
        for lane_id in B1_INCOMING_LANES
    ]
    return np.array(counts, dtype=np.float32)


def _get_total_waiting() -> float:
    total = 0.0
    for veh_id in traci.vehicle.getIDList():
        total += traci.vehicle.getWaitingTime(veh_id)
    return total


def _get_queue_length() -> int:
    return sum(
        traci.lane.getLastStepHaltingNumber(lane_id)
        for lane_id in B1_INCOMING_LANES
    )


def _get_avg_speed() -> float:
    ids = traci.vehicle.getIDList()
    if not ids:
        return 0.0
    return sum(traci.vehicle.getSpeed(v) for v in ids) / len(ids)


def _set_phase(action: int) -> None:
    idx = GREEN_PHASE_INDICES[action]
    traci.trafficlight.setRedYellowGreenState(CONTROLLED_TL_ID, B1_PHASES[idx])


def _run_traci_episode(
    n_steps: int,
    control_interval: int,
    sim_end: int,
    get_action,          # callable(step, obs) -> int | None
    seed: int | None = None,
) -> list[dict]:
    if seed is not None:
        random.seed(seed)
    if not SUMO_CONFIG.exists():
        raise FileNotFoundError(SUMO_CONFIG)
    sumo_bin = find_sumo_bin(False)
    if not sumo_bin:
        raise RuntimeError("SUMO not found. Set SUMO_HOME or add sumo/bin to PATH.")
    sumo_cmd = [
        sumo_bin,
        "-c", str(SUMO_CONFIG),
        "--no-step-log",
        "--no-warnings",
        "--end", str(sim_end),
    ]
    os.chdir(SUMO_DIR)
    traci.start(sumo_cmd)
    records = []
    try:
        for step in range(n_steps):
            obs = _get_lane_counts()
            action = get_action(step, obs) if get_action is not None else None
            if action is not None:
                _set_phase(int(action) % 2)
            for _ in range(control_interval):
                traci.simulationStep()
                if int(traci.simulation.getTime()) >= sim_end - 1:
                    break
                if traci.simulation.getMinExpectedNumber() < 0:
                    break
            w = _get_total_waiting()
            q = _get_queue_length()
            s = _get_avg_speed()
            records.append({"total_waiting": w, "queue_length": q, "avg_speed": s, "reward": -w})
            if int(traci.simulation.getTime()) >= sim_end - 1:
                break
    finally:
        traci.close()
    return records


# ---------------------------------------------------------------------------
# Controller runners
# ---------------------------------------------------------------------------

def run_fixed_time(
    n_steps: int = 72, control_interval: int = 5, sim_end: int = 360,
    seed: int | None = 42,
) -> list[dict]:
    """SUMO fixed-time program (no phase override)."""
    return _run_traci_episode(n_steps, control_interval, sim_end, get_action=None, seed=seed)


def run_random(
    n_steps: int = 72, control_interval: int = 5, sim_end: int = 360,
    seed: int | None = 42,
) -> list[dict]:
    """Random phase selection each step."""
    def get_action(step: int, obs: np.ndarray) -> int:
        return random.randint(0, 1)
    return _run_traci_episode(n_steps, control_interval, sim_end, get_action=get_action, seed=seed)


def run_scratch_rl(
    model_path: str | Path,
    n_steps: int = 72,
    control_interval: int = 5,
    sim_end: int = 360,
    seed: int | None = 42,
) -> list[dict]:
    """Run with the from-scratch trained DQN (.npz weights)."""
    data = np.load(str(model_path))
    net = Network(state_size=4, action_size=2, hidden_size=64)
    net.w1 = data["w1"]
    net.b1 = data["b1"]
    net.w2 = data["w2"]
    net.b2 = data["b2"]
    net.w3 = data["w3"]
    net.b3 = data["b3"]

    def get_action(step: int, obs: np.ndarray) -> int:
        q = net.forward(obs.reshape(1, -1))[0]
        return int(np.argmax(q))

    return _run_traci_episode(
        n_steps, control_interval, sim_end, get_action=get_action, seed=seed
    )


# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------

def summarize(records: list[dict], name: str) -> dict:
    if not records:
        return {"name": name, "mean_waiting": 0, "mean_queue": 0, "mean_speed": 0, "total_reward": 0}
    n = len(records)
    return {
        "name": name,
        "mean_waiting": sum(r["total_waiting"] for r in records) / n,
        "mean_queue": sum(r["queue_length"] for r in records) / n,
        "mean_speed": sum(r["avg_speed"] for r in records) / n,
        "total_reward": sum(r["reward"] for r in records),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Fixed-time, Random, and from-scratch RL (DQN)"
    )
    parser.add_argument(
        "--model", type=str, default="rl/models/dqn_traffic_light_scratch.npz",
        help="Path to saved from-scratch DQN weights (.npz)",
    )
    parser.add_argument("--steps", type=int, default=72, help="Control steps per run")
    parser.add_argument("--sim-end", type=int, default=360, help="SUMO sim end (s)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-rl", action="store_true", help="Skip RL (e.g. if no model yet)")
    args = parser.parse_args()

    control_interval = 5
    seed = args.seed
    results = []

    print("Running Fixed-time...")
    rec_ft = run_fixed_time(n_steps=args.steps, control_interval=control_interval,
                             sim_end=args.sim_end, seed=seed)
    results.append(summarize(rec_ft, "Fixed-time"))

    print("Running Random...")
    rec_rand = run_random(n_steps=args.steps, control_interval=control_interval,
                          sim_end=args.sim_end, seed=seed)
    results.append(summarize(rec_rand, "Random"))

    if not args.no_rl:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent.parent
            model_path = project_root / model_path
        if model_path.exists():
            print("Running RL (from-scratch DQN)...")
            rec_rl = run_scratch_rl(model_path, n_steps=args.steps,
                                     control_interval=control_interval,
                                     sim_end=args.sim_end, seed=seed)
            results.append(summarize(rec_rl, "RL-scratch"))
        else:
            print(
                f"Model not found: {model_path}, skipping RL.\n"
                f"Train with: python rl/from_scratch/train_dqn_scratch.py"
            )
    else:
        print("Skipping RL (--no-rl).")

    print("\n" + "=" * 65)
    print("Comparison (lower waiting / higher reward is better)")
    print("=" * 65)
    for r in results:
        print(
            f"  {r['name']:14} | mean_waiting: {r['mean_waiting']:8.1f} s"
            f" | mean_queue: {r['mean_queue']:.1f}"
            f" | total_reward: {r['total_reward']:.0f}"
        )
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
