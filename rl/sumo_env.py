"""
Phase 4: Gymnasium environment for SUMO traffic light control.

SumoEnv wraps the SUMO simulation (TraCI) and exposes:
- observation_space: lane vehicle counts (4 lanes) at B1
- action_space: Discrete(2) — 0 = NS green, 1 = EW green
- reward: negative total waiting time (minimize waiting)

Usage:
    from sumo_env import SumoEnv
    env = SumoEnv()
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""
from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np

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
import traci


def _get_lane_vehicle_counts() -> np.ndarray:
    """Vehicle count per B1 incoming lane (last step). Shape (4,)."""
    counts = [
        traci.lane.getLastStepVehicleNumber(lane_id)
        for lane_id in B1_INCOMING_LANES
    ]
    return np.array(counts, dtype=np.float32)


def _get_total_waiting_time() -> float:
    """Total waiting time (s) over all vehicles."""
    total = 0.0
    for veh_id in traci.vehicle.getIDList():
        total += traci.vehicle.getWaitingTime(veh_id)
    return total


def _get_queue_length() -> int:
    """Halting vehicle count on B1 incoming lanes."""
    return sum(
        traci.lane.getLastStepHaltingNumber(lane_id)
        for lane_id in B1_INCOMING_LANES
    )


def _get_avg_speed() -> float:
    """Mean speed (m/s) over all vehicles. 0 if none."""
    ids = traci.vehicle.getIDList()
    if not ids:
        return 0.0
    return sum(traci.vehicle.getSpeed(v) for v in ids) / len(ids)


def _set_phase(action: int) -> None:
    """Set B1 to green phase: 0 = NS (index 0), 1 = EW (index 2)."""
    idx = GREEN_PHASE_INDICES[action]
    traci.trafficlight.setRedYellowGreenState(CONTROLLED_TL_ID, B1_PHASES[idx])


class SumoEnv(gym.Env[np.ndarray, int]):
    """
    Gymnasium environment for traffic light control at the center intersection (B1).

    One env step = one control step: set phase, then advance simulation by
    control_interval seconds. Observation = lane vehicle counts; reward =
    negative total waiting time (minimize waiting).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str | None = None,
        control_interval: int = 5,
        max_steps_per_episode: int = 72,
        sim_end: int = 360,
        use_gui: bool = False,
        **kwargs: Any,
    ):
        """
        Args:
            config_path: Path to simulation.sumocfg. Default: sumo/simulation.sumocfg.
            control_interval: Simulation seconds per env step (phase fixed for this duration).
            max_steps_per_episode: Max env steps per episode (truncation).
            sim_end: SUMO simulation end time (seconds).
            use_gui: If True, use sumo-gui.
        """
        super().__init__(**kwargs)
        self._config_path = str(config_path or SUMO_CONFIG)
        self._control_interval = control_interval
        self._max_steps_per_episode = max_steps_per_episode
        self._sim_end = sim_end
        self._use_gui = use_gui
        self._sumo_bin: str | None = None
        self._traci_connected = False
        self._step_count = 0

        # Observation: 4 lane vehicle counts (float32)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=100.0,  # reasonable upper bound per lane
            shape=(len(B1_INCOMING_LANES),),
            dtype=np.float32,
        )
        # Action: 0 = NS green, 1 = EW green
        self.action_space = gym.spaces.Discrete(2)

    def _start_simulation(self) -> None:
        """Start or restart SUMO via TraCI."""
        if self._traci_connected:
            try:
                traci.close()
            except Exception:
                pass
            self._traci_connected = False

        if self._sumo_bin is None:
            self._sumo_bin = find_sumo_bin(self._use_gui)
            if not self._sumo_bin:
                raise RuntimeError(
                    "SUMO not found. Set SUMO_HOME or add sumo/bin to PATH."
                )

        sumo_cmd = [
            self._sumo_bin,
            "-c", self._config_path,
            "--no-step-log",
            "--no-warnings",
            "--end", str(self._sim_end),
        ]
        os.chdir(SUMO_DIR)
        traci.start(sumo_cmd)
        self._traci_connected = True

    def _get_obs(self) -> np.ndarray:
        return _get_lane_vehicle_counts()

    def _is_done(self) -> bool:
        """True if simulation has ended (time or no vehicles left)."""
        if int(traci.simulation.getTime()) >= self._sim_end - 1:
            return True
        if traci.simulation.getMinExpectedNumber() < 0:
            return True
        return False

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment: start a new SUMO run and return initial observation."""
        super().reset(seed=seed)
        self._start_simulation()
        self._step_count = 0
        obs = self._get_obs()
        info = {
            "sim_time": 0,
            "total_waiting": _get_total_waiting_time(),
            "queue_length": _get_queue_length(),
            "avg_speed": _get_avg_speed(),
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply action (set phase), advance simulation by control_interval, return transition."""
        if not self._traci_connected:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Validate action
        action = int(action)
        if action not in (0, 1):
            action = 0

        _set_phase(action)

        # Advance simulation for control_interval seconds
        for _ in range(self._control_interval):
            traci.simulationStep()
            if self._is_done():
                break

        self._step_count += 1
        sim_time = int(traci.simulation.getTime())
        total_waiting = _get_total_waiting_time()
        obs = self._get_obs()

        # Reward: minimize total waiting time
        reward = -total_waiting

        terminated = self._is_done()
        truncated = self._step_count >= self._max_steps_per_episode

        info = {
            "sim_time": sim_time,
            "total_waiting": total_waiting,
            "queue_length": _get_queue_length(),
            "avg_speed": _get_avg_speed(),
            "step_count": self._step_count,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close TraCI connection."""
        if self._traci_connected:
            try:
                traci.close()
            except Exception:
                pass
            self._traci_connected = False


if __name__ == "__main__":
    """Quick test: run a few random steps and print obs/reward."""
    import sys
    from pathlib import Path
    # Allow running as python rl/sumo_env.py from project root
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    env = SumoEnv(control_interval=5, max_steps_per_episode=10, sim_end=60)
    obs, info = env.reset(seed=42)
    print("Initial obs (lane counts):", obs)
    total_reward = 0.0
    for t in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {t+1}: action={action}, reward={reward:.1f}, obs={obs}")
        if terminated or truncated:
            break
    print("Total reward:", total_reward)
    env.close()
    print("Done.")
