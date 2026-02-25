#!/usr/bin/env python3
"""
Phase 5: Evaluation script — compare Fixed-time, Random, and RL (DQN) controllers.

Runs the same simulation length with each controller and reports:
  mean total waiting time, mean queue length, mean speed, total reward.

Run from project root:
    python rl/evaluate.py
    python rl/evaluate.py --model rl/models/dqn_traffic_light.zip --steps 72
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def _get_lane_counts() -> np.ndarray:
    """Observation: vehicle count per B1 incoming lane."""
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
    get_action: None | callable,
    seed: int | None = None,
) -> list[dict]:
    """
    Run one episode with TraCI. get_action(step, obs) -> int | None.
    If None, do not set phase (fixed-time). Else set phase to action (0 or 1).
    Returns list of dicts with total_waiting, queue_length, avg_speed, reward (negative waiting).
    """
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
            records.append({
                "total_waiting": w,
                "queue_length": q,
                "avg_speed": s,
                "reward": -w,
            })
            if int(traci.simulation.getTime()) >= sim_end - 1:
                break
    finally:
        traci.close()
    return records


def run_fixed_time(
    n_steps: int = 72,
    control_interval: int = 5,
    sim_end: int = 360,
    seed: int | None = 42,
) -> list[dict]:
    """Run with default SUMO fixed-time program (no phase override)."""
    return _run_traci_episode(n_steps, control_interval, sim_end, get_action=None, seed=seed)


def run_random(
    n_steps: int = 72,
    control_interval: int = 5,
    sim_end: int = 360,
    seed: int | None = 42,
) -> list[dict]:
    """Run with random phase selection each step."""
    def get_action(step: int, obs: np.ndarray) -> int:
        return random.randint(0, 1)
    return _run_traci_episode(n_steps, control_interval, sim_end, get_action=get_action, seed=seed)


def run_rl(
    model_path: str | Path,
    n_steps: int = 72,
    control_interval: int = 5,
    sim_end: int = 360,
    seed: int | None = 42,
) -> list[dict]:
    """Run with trained DQN model (Stable-Baselines3)."""
    from stable_baselines3 import DQN
    from sumo_env import SumoEnv

    model = DQN.load(str(model_path))
    env = SumoEnv(
        control_interval=control_interval,
        max_steps_per_episode=n_steps,
        sim_end=sim_end,
        use_gui=False,
    )
    records = []
    obs, info = env.reset(seed=seed)
    records.append({
        "total_waiting": info.get("total_waiting", 0.0),
        "queue_length": info.get("queue_length", 0),
        "avg_speed": info.get("avg_speed", 0.0),
        "reward": -info.get("total_waiting", 0.0),
    })
    for _ in range(n_steps - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        records.append({
            "total_waiting": info.get("total_waiting", 0.0),
            "queue_length": info.get("queue_length", 0),
            "avg_speed": info.get("avg_speed", 0.0),
            "reward": reward,
        })
        if terminated or truncated:
            break
    env.close()
    return records


def summarize(records: list[dict], name: str) -> dict:
    """Compute mean total_waiting, mean queue_length, mean avg_speed, sum reward."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5: Compare Fixed-time, Random, RL")
    parser.add_argument("--model", type=str, default="rl/models/dqn_traffic_light.zip",
                        help="Path to saved DQN model (.zip)")
    parser.add_argument("--steps", type=int, default=72, help="Control steps per run")
    parser.add_argument("--sim-end", type=int, default=360, help="SUMO sim end (s)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-rl", action="store_true", help="Skip RL (e.g. if no model yet)")
    args = parser.parse_args()

    control_interval = 5
    seed = args.seed

    results = []

    print("Running Fixed-time...")
    rec_ft = run_fixed_time(n_steps=args.steps, control_interval=control_interval, sim_end=args.sim_end, seed=seed)
    results.append(summarize(rec_ft, "Fixed-time"))

    print("Running Random...")
    rec_rand = run_random(n_steps=args.steps, control_interval=control_interval, sim_end=args.sim_end, seed=seed)
    results.append(summarize(rec_rand, "Random"))

    if not args.no_rl:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            model_path = project_root / model_path
        if model_path.exists():
            print("Running RL (DQN)...")
            rec_rl = run_rl(model_path, n_steps=args.steps, control_interval=control_interval, sim_end=args.sim_end, seed=seed)
            results.append(summarize(rec_rl, "RL (DQN)"))
        else:
            print(f"Model not found: {model_path}, skipping RL. Train with: python rl/train_dqn.py")
    else:
        print("Skipping RL (--no-rl).")

    # Print comparison table
    print("\n" + "=" * 60)
    print("Comparison (lower waiting / higher reward is better)")
    print("=" * 60)
    for r in results:
        print(f"  {r['name']:12} | mean_waiting: {r['mean_waiting']:8.1f} s | mean_queue: {r['mean_queue']:.1f} | total_reward: {r['total_reward']:.0f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
