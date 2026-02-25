#!/usr/bin/env python3
"""
Phase 5: Train DQN agent with Stable-Baselines3.

Trains on SumoEnv and saves the model to rl/models/dqn_traffic_light.zip.

Run from project root:
    python rl/train_dqn.py

Options: --timesteps, --save-path, --seed, --gui (for debugging).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure rl is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sumo_utils import add_sumo_to_path

add_sumo_to_path()

from stable_baselines3 import DQN

from sumo_env import SumoEnv

DEFAULT_TIMESTEPS = 30_000
DEFAULT_SAVE_PATH = Path(__file__).resolve().parent / "models" / "dqn_traffic_light"
DEFAULT_SEED = 42


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5: Train DQN traffic light agent")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS,
                        help="Total training timesteps")
    parser.add_argument("--save-path", type=str, default=str(DEFAULT_SAVE_PATH),
                        help="Path to save model (without .zip)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui (slower)")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        control_interval=5,
        max_steps_per_episode=72,
        sim_end=360,
        use_gui=args.gui,
    )
    env = SumoEnv(**env_kwargs)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=10_000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        seed=args.seed,
    )

    # No EvalCallback: SumoEnv uses a single TraCI connection, so we cannot run
    # a separate eval env in parallel. Use rl/evaluate.py after training to compare.
    print(f"Training for {args.timesteps} timesteps, save path: {save_path}.zip")
    model.learn(total_timesteps=args.timesteps)
    model.save(str(save_path))
    print(f"Model saved to {save_path}.zip")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
