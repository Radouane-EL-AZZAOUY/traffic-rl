#!/usr/bin/env python3
"""
From-scratch DQN training script using NumPy (no Stable-Baselines3).

Mirrors rl/train_dqn.py in interface and behaviour:
  - Same CLI flags: --timesteps, --save-path, --seed, --gui
  - Same SumoEnv environment and identical hyperparameters
  - Saves trained weights to <save-path>.npz  (load with evaluate_scratch.py)

Run from project root:
    python rl/from_scratch/train_dqn_scratch.py
    python rl/from_scratch/train_dqn_scratch.py --timesteps 50000 --save-path rl/models/dqn_scratch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow importing sumo_env / sumo_utils from rl/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sumo_utils import add_sumo_to_path

add_sumo_to_path()

from sumo_env import SumoEnv
from from_scratch.dqn_numpy import DQNAgent, DQNConfig

DEFAULT_TIMESTEPS = 30_000
DEFAULT_SAVE_PATH = Path(__file__).resolve().parent.parent / "models" / "dqn_traffic_light_scratch"
DEFAULT_SEED = 42

# Log every this many timesteps
LOG_INTERVAL = 1_000


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train DQN traffic light agent from scratch (NumPy only)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=DEFAULT_TIMESTEPS,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--save-path", type=str, default=str(DEFAULT_SAVE_PATH),
        help="Path to save model weights (without extension; .npz will be added)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui (slower)")
    args = parser.parse_args()

    import numpy as np
    np.random.seed(args.seed)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Environment (identical settings to train_dqn.py) ---
    env = SumoEnv(
        control_interval=5,
        max_steps_per_episode=72,
        sim_end=360,
        use_gui=args.gui,
    )

    state_size = env.observation_space.shape[0]   # 4
    action_size = env.action_space.n              # 2

    # --- Agent (hyperparameters mirror train_dqn.py) ---
    cfg = DQNConfig(
        state_size=state_size,
        action_size=action_size,
        gamma=0.99,
        lr=5e-4,
        batch_size=32,
        buffer_size=10_000,
        learning_starts=1_000,
        train_freq=4,
        target_update_interval=500,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_fraction=0.2,
    )
    agent = DQNAgent(cfg, total_timesteps=args.timesteps)

    print(f"Training for {args.timesteps} timesteps, save path: {save_path}.npz")
    print(f"State size: {state_size}  |  Action size: {action_size}")
    print(f"{'Timestep':>10}  {'Episode':>8}  {'Ep reward':>12}  {'Loss':>10}  {'Epsilon':>8}")
    print("-" * 60)

    obs, _ = env.reset(seed=args.seed)
    episode = 0
    ep_reward = 0.0
    last_losses: list[float] = []
    last_log_t = 0

    for t in range(args.timesteps):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store_transition(obs, action, reward, next_obs, done)
        loss = agent.train_step()
        agent.maybe_update_target()

        ep_reward += reward
        obs = next_obs

        if loss is not None:
            last_losses.append(loss)

        if done:
            episode += 1
            obs, _ = env.reset()
            ep_reward = 0.0

        # Log once per LOG_INTERVAL timesteps
        if (t + 1) % LOG_INTERVAL == 0:
            mean_loss = float(np.mean(last_losses)) if last_losses else float("nan")
            print(
                f"{t + 1:>10}  {episode:>8}  {ep_reward:>12.1f}"
                f"  {mean_loss:>10.4f}  {agent.epsilon():>8.3f}"
            )
            last_losses.clear()
            last_log_t = t + 1

    env.close()

    # Save weights
    agent.save(str(save_path))
    print(f"\nModel saved to {save_path}.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
