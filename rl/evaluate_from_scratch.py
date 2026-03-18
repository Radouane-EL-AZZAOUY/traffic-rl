#!/usr/bin/env python3
"""
Evaluate NumPy-based DQN (from_scratch) on SUMO traffic light control.

This script mirrors rl/evaluate.py but loads a .npz file produced by
rl/train_dqn.py --impl scratch and runs it inside SumoEnv.

Run from project root:
    python rl/evaluate_from_scratch.py
    python rl/evaluate_from_scratch.py --model rl/models/dqn_traffic_light_scratch.npz --steps 72
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import run_fixed_time, run_random, summarize  # type: ignore
from from_scratch.dqn_numpy import Network
from sumo_utils import add_sumo_to_path
from sumo_env import SumoEnv

add_sumo_to_path()


def run_rl_scratch(
    model_path: str | Path,
    n_steps: int = 72,
    control_interval: int = 5,
    sim_end: int = 360,
    seed: int | None = 42,
) -> list[dict]:
    """
    Run with trained NumPy DQN model saved as .npz.

    Expects parameters saved by DQNAgent.save():
        w1, b1, w2, b2, w3, b3
    """
    model_path = Path(model_path)
    env = SumoEnv(
        control_interval=control_interval,
        max_steps_per_episode=n_steps,
        sim_end=sim_end,
        use_gui=False,
    )

    # Infer sizes from environment
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)

    # Build network and load weights
    data = dict(np.load(model_path))  # type: ignore[name-defined]
    net = Network(state_size, action_size)
    net.w1 = data["w1"]
    net.b1 = data["b1"]
    net.w2 = data["w2"]
    net.b2 = data["b2"]
    net.w3 = data["w3"]
    net.b3 = data["b3"]

    records: list[dict] = []
    obs, info = env.reset(seed=seed)
    records.append(
        {
            "total_waiting": info.get("total_waiting", 0.0),
            "queue_length": info.get("queue_length", 0),
            "avg_speed": info.get("avg_speed", 0.0),
            "reward": -info.get("total_waiting", 0.0),
        }
    )

    for _ in range(n_steps - 1):
        state_batch = obs.reshape(1, -1).astype("float32")
        q_values = net.forward(state_batch)[0]
        action = int(q_values.argmax())

        obs, reward, terminated, truncated, info = env.step(action)
        records.append(
            {
                "total_waiting": info.get("total_waiting", 0.0),
                "queue_length": info.get("queue_length", 0),
                "avg_speed": info.get("avg_speed", 0.0),
                "reward": reward,
            }
        )
        if terminated or truncated:
            break

    env.close()
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate NumPy DQN (from_scratch) vs Fixed-time and Random"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rl/models/dqn_traffic_light_scratch.npz",
        help="Path to saved NumPy DQN model (.npz)",
    )
    parser.add_argument("--steps", type=int, default=72, help="Control steps per run")
    parser.add_argument("--sim-end", type=int, default=360, help="SUMO sim end (s)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-rl",
        action="store_true",
        help="Skip RL (e.g. if no model yet, only run baselines)",
    )
    args = parser.parse_args()

    control_interval = 5
    seed = args.seed

    results = []

    print("Running Fixed-time...")
    rec_ft = run_fixed_time(
        n_steps=args.steps,
        control_interval=control_interval,
        sim_end=args.sim_end,
        seed=seed,
    )
    results.append(summarize(rec_ft, "Fixed-time"))

    print("Running Random...")
    rec_rand = run_random(
        n_steps=args.steps,
        control_interval=control_interval,
        sim_end=args.sim_end,
        seed=seed,
    )
    results.append(summarize(rec_rand, "Random"))

    if not args.no_rl:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            model_path = project_root / model_path
        if model_path.exists():
            # Lazy import to avoid requiring NumPy before needed
            global np  # type: ignore[global-variable-not-assigned]
            import numpy as np  # type: ignore[import-not-found]

            print("Running RL (NumPy DQN)...")
            rec_rl = run_rl_scratch(
                model_path,
                n_steps=args.steps,
                control_interval=control_interval,
                sim_end=args.sim_end,
                seed=seed,
            )
            results.append(summarize(rec_rl, "RL (NumPy DQN)"))
        else:
            print(
                f"Model not found: {model_path}, skipping RL. "
                "Train with: python rl/train_dqn.py --impl scratch"
            )
    else:
        print("Skipping RL (--no-rl).")

    # Print comparison table (same style as rl/evaluate.py)
    print("\n" + "=" * 60)
    print("Comparison (lower waiting / higher reward is better)")
    print("=" * 60)
    for r in results:
        print(
            f"  {r['name']:12} | mean_waiting: {r['mean_waiting']:8.1f} s | "
            f"mean_queue: {r['mean_queue']:.1f} | total_reward: {r['total_reward']:.0f}"
        )
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

