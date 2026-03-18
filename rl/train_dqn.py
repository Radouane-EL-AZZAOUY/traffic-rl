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

from sumo_env import SumoEnv
from from_scratch.dqn_numpy import DQNAgent, DQNConfig

DEFAULT_TIMESTEPS = 30_000
DEFAULT_SAVE_PATH = Path(__file__).resolve().parent / "models" / "dqn_traffic_light"
DEFAULT_SEED = 42
DEFAULT_IMPL = "sb3"  # "sb3" or "scratch"


def _train_with_sb3(args: argparse.Namespace, env: SumoEnv, save_path: Path) -> None:
    """Train DQN using Stable-Baselines3 (original implementation)."""
    try:
        from stable_baselines3 import DQN  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is not installed. Install it or use --impl scratch."
        ) from exc

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

    print(f"[sb3] Training for {args.timesteps} timesteps, save path: {save_path}.zip")
    model.learn(total_timesteps=args.timesteps)
    model.save(str(save_path))
    print(f"[sb3] Model saved to {save_path}.zip")


def _train_with_scratch(args: argparse.Namespace, env: SumoEnv, save_path: Path) -> None:
    """Train DQN using NumPy implementation (no Stable-Baselines3)."""
    # Observation space is Box(shape=(4,))
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)

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

    # If default save path is used, avoid clashing with SB3 model
    save_npz_path = save_path
    if str(save_npz_path.name) == "dqn_traffic_light":
        save_npz_path = save_npz_path.with_name(save_npz_path.name + "_scratch")

    print(
        f"[scratch] Training NumPy DQN for {args.timesteps} timesteps, "
        f"save path: {save_npz_path}.npz"
    )

    obs, _info = env.reset(seed=args.seed)
    obs = obs.astype("float32")
    episode_reward = 0.0
    for t in range(1, args.timesteps + 1):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _info = env.step(action)
        next_obs = next_obs.astype("float32")
        done = bool(terminated or truncated)

        agent.store_transition(obs, action, reward, next_obs, done)
        loss = agent.train_step()
        agent.maybe_update_target()

        episode_reward += reward
        obs = next_obs

        if done:
            print(
                f"[scratch] Step {t}/{args.timesteps} - "
                f"episode_reward={episode_reward:.2f}, epsilon={agent.epsilon():.3f}"
            )
            obs, _info = env.reset()
            obs = obs.astype("float32")
            episode_reward = 0.0

        if t % 10_000 == 0:
            msg = f"[scratch] Reached step {t}/{args.timesteps}"
            if loss is not None:
                msg += f", last_loss={loss:.4f}"
            print(msg)

    agent.save(str(save_npz_path))
    print(f"[scratch] NumPy DQN parameters saved to {save_npz_path}.npz")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5: Train DQN traffic light agent")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS,
                        help="Total training timesteps")
    parser.add_argument("--save-path", type=str, default=str(DEFAULT_SAVE_PATH),
                        help="Path to save model (without .zip)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui (slower)")
    parser.add_argument(
        "--impl",
        type=str,
        choices=["sb3", "scratch"],
        default=DEFAULT_IMPL,
        help="RL implementation to use: 'sb3' (Stable-Baselines3) or 'scratch' (NumPy DQN)",
    )
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

    if args.impl == "sb3":
        _train_with_sb3(args, env, save_path)
    else:
        _train_with_scratch(args, env, save_path)

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
