"""
Phase 6: KPI collection module.

Computes and returns in structured JSON:
- average_waiting_time (s, per vehicle at arrival)
- average_travel_time (s, per vehicle at arrival)
- throughput (vehicles per hour)
- average_speed (m/s, time-averaged)
- number_of_phase_switches

Use KPICollector to feed data each step, or run_simulation_and_collect_kpis()
to run a simulation and return JSON.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np

# Project root and rl package for TraCI
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# SUMO TraCI (for run_simulation_and_collect_kpis)
try:
    from rl.sumo_utils import (
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
    _TRACI_AVAILABLE = True
except Exception:
    _TRACI_AVAILABLE = False


class KPICollector:
    """
    Collects simulation step data and computes KPIs.

    Call update() each step with data from TraCI (or another source).
    Call get_results() for the final dict, or to_json() for JSON string.
    """

    def __init__(self) -> None:
        self.depart_times: dict[str, float] = {}
        self.travel_times: list[float] = []
        self.waiting_times_at_arrival: list[float] = []
        self.phase_switch_count = 0
        self.total_waiting_sum = 0.0
        self.speed_sum = 0.0
        self.speed_count = 0
        self.n_steps = 0
        self.sim_duration_s = 0.0
        self._last_phase: str | None = None

    def update(
        self,
        step: float,
        departed_ids: list[str],
        arrived_ids: list[str],
        arrived_waiting_times: dict[str, float],
        total_waiting: float,
        mean_speed: float,
        phase_switched: bool,
        current_phase: str | None = None,
    ) -> None:
        """
        Update collector with one simulation step.

        Args:
            step: Current simulation time (s).
            departed_ids: Vehicle IDs that departed this step.
            arrived_ids: Vehicle IDs that arrived (completed trip) this step.
            arrived_waiting_times: For each arrived id, its waiting time (s) before arrival.
            total_waiting: Sum of getWaitingTime over all current vehicles this step.
            mean_speed: Mean speed (m/s) over all current vehicles this step.
            phase_switched: True if the traffic light phase was changed this step.
            current_phase: Current TL state string (optional, for switch detection).
        """
        for vid in departed_ids:
            self.depart_times[vid] = step
        for vid in arrived_ids:
            self.travel_times.append(step - self.depart_times.get(vid, step))
            self.waiting_times_at_arrival.append(arrived_waiting_times.get(vid, 0.0))
            self.depart_times.pop(vid, None)
        if phase_switched:
            self.phase_switch_count += 1
        self.total_waiting_sum += total_waiting
        if mean_speed >= 0:
            self.speed_sum += mean_speed
            self.speed_count += 1
        self.n_steps += 1
        self.sim_duration_s = step
        self._last_phase = current_phase

    def get_results(self) -> dict:
        """Return KPIs as a structured dict."""
        n_arrived = len(self.travel_times)
        duration_h = self.sim_duration_s / 3600.0 if self.sim_duration_s > 0 else 1e-6

        return {
            "average_waiting_time": (
                sum(self.waiting_times_at_arrival) / n_arrived if n_arrived else 0.0
            ),
            "average_travel_time": (
                sum(self.travel_times) / n_arrived if n_arrived else 0.0
            ),
            "throughput": n_arrived / duration_h,  # vehicles per hour
            "throughput_total": n_arrived,
            "average_speed": (
                self.speed_sum / self.speed_count if self.speed_count else 0.0
            ),
            "number_of_phase_switches": self.phase_switch_count,
            "sim_duration_s": self.sim_duration_s,
            "n_steps": self.n_steps,
            "n_arrived": n_arrived,
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Return KPIs as a JSON string."""
        return json.dumps(self.get_results(), indent=indent)


def _get_total_waiting(traci) -> float:
    total = 0.0
    for veh_id in traci.vehicle.getIDList():
        total += traci.vehicle.getWaitingTime(veh_id)
    return total


def _get_mean_speed(traci) -> float:
    ids = traci.vehicle.getIDList()
    if not ids:
        return 0.0
    return sum(traci.vehicle.getSpeed(v) for v in ids) / len(ids)


def _set_phase(traci, action: int) -> None:
    idx = GREEN_PHASE_INDICES[action]
    traci.trafficlight.setRedYellowGreenState(CONTROLLED_TL_ID, B1_PHASES[idx])


def _get_queue_length(traci) -> int:
    return sum(
        traci.lane.getLastStepHaltingNumber(lane_id)
        for lane_id in B1_INCOMING_LANES
    )


def run_simulation_and_collect_kpis(
    sim_end: int = 360,
    control_interval: int = 5,
    controller: str = "fixed",
    model_path: str | Path | None = None,
    seed: int | None = 42,
    on_step: Callable[..., None] | None = None,
) -> dict:
    """
    Run a SUMO simulation with the given controller and return KPI dict.

    Args:
        sim_end: Simulation end time (s).
        control_interval: Seconds per control step (for random/RL).
        controller: "fixed" (no phase override), "random", or "rl".
        model_path: Path to DQN .zip when controller=="rl".
        seed: Random seed (for random controller).

    Returns:
        Dict from KPICollector.get_results() (same as structured JSON content).
    """
    if not _TRACI_AVAILABLE:
        raise RuntimeError("TraCI not available. Ensure SUMO is installed and rl package is on path.")
    if not SUMO_CONFIG.exists():
        raise FileNotFoundError(f"SUMO config not found: {SUMO_CONFIG}")
    sumo_bin = find_sumo_bin(False)
    if not sumo_bin:
        raise RuntimeError("SUMO not found. Set SUMO_HOME or add sumo/bin to PATH.")

    if seed is not None:
        random.seed(seed)

    model = None
    if controller == "rl" and model_path:
        from stable_baselines3 import DQN
        model = DQN.load(str(model_path))

    sumo_cmd = [
        sumo_bin,
        "-c", str(SUMO_CONFIG),
        "--no-step-log",
        "--no-warnings",
        "--end", str(sim_end),
    ]
    os.chdir(SUMO_DIR)
    traci.start(sumo_cmd)

    collector = KPICollector()
    last_waiting: dict[str, float] = {}
    next_control_at = 0.0
    current_phase: str | None = None
    step = 0.0

    try:
        while step < sim_end:
            # At start of each control block: optionally set phase (random/RL)
            phase_switched = False
            if controller != "fixed" and step >= next_control_at:
                if controller == "random":
                    action = random.randint(0, 1)
                else:  # rl
                    obs = np.array([
                        traci.lane.getLastStepVehicleNumber(lane_id)
                        for lane_id in B1_INCOMING_LANES
                    ], dtype=np.float32)
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action) % 2
                new_state = B1_PHASES[GREEN_PHASE_INDICES[action]]
                phase_switched = new_state != current_phase
                current_phase = new_state
                _set_phase(traci, action)
                next_control_at = step + control_interval

            # Advance simulation by control_interval steps; collect every sim step
            for _ in range(control_interval):
                for vid in traci.vehicle.getIDList():
                    last_waiting[vid] = traci.vehicle.getWaitingTime(vid)
                traci.simulationStep()
                step = traci.simulation.getTime()
                departed = list(traci.simulation.getDepartedIDList())
                arrived = list(traci.simulation.getArrivedIDList())
                arrived_waiting = {vid: last_waiting.get(vid, 0.0) for vid in arrived}
                for vid in arrived:
                    last_waiting.pop(vid, None)
                total_waiting = _get_total_waiting(traci)
                mean_speed = _get_mean_speed(traci)
                queue_length = _get_queue_length(traci)
                lane_counts = [traci.lane.getLastStepVehicleNumber(lid) for lid in B1_INCOMING_LANES]
                collector.update(
                    step=step,
                    departed_ids=departed,
                    arrived_ids=arrived,
                    arrived_waiting_times=arrived_waiting,
                    total_waiting=total_waiting,
                    mean_speed=mean_speed,
                    phase_switched=phase_switched,
                    current_phase=current_phase,
                )
                if on_step is not None:
                    vehicle_positions = [
                        {"id": vid, "x": traci.vehicle.getPosition(vid)[0], "y": traci.vehicle.getPosition(vid)[1]}
                        for vid in traci.vehicle.getIDList()
                    ]
                    on_step(step, current_phase, lane_counts, total_waiting, queue_length, mean_speed, collector, vehicle_positions)
                phase_switched = False
                if step >= sim_end or traci.simulation.getMinExpectedNumber() < 0:
                    break
            if step >= sim_end:
                break
    finally:
        traci.close()

    return collector.get_results()


def get_kpis_json(
    sim_end: int = 360,
    control_interval: int = 5,
    controller: str = "fixed",
    model_path: str | Path | None = None,
    seed: int | None = 42,
    indent: int | None = 2,
) -> str:
    """
    Run simulation and return KPIs as a JSON string.

    Same arguments as run_simulation_and_collect_kpis(); indent for json.dumps.
    """
    results = run_simulation_and_collect_kpis(
        sim_end=sim_end,
        control_interval=control_interval,
        controller=controller,
        model_path=model_path,
        seed=seed,
    )
    return json.dumps(results, indent=indent)


if __name__ == "__main__":
    """Run a short fixed-time simulation and print KPIs as JSON."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--controller", choices=("fixed", "random", "rl"), default="fixed")
    p.add_argument("--sim-end", type=int, default=120)
    p.add_argument("--model", type=str, default=None)
    args = p.parse_args()
    model_path = args.model or (str(PROJECT_ROOT / "rl" / "models" / "dqn_traffic_light.zip") if args.controller == "rl" else None)
    try:
        out = get_kpis_json(sim_end=args.sim_end, controller=args.controller, model_path=model_path)
        print(out)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
