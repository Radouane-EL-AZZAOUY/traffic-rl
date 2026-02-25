#!/usr/bin/env python3
"""
Phase 3: Random traffic light controller.

- At each control step, randomly chooses a phase (NS green or EW green).
- Logs per step: waiting time, queue length, average speed.
- Prints summary at the end.

Run from project root:
    python rl/random_agent.py

Requires: SUMO installed, SUMO_HOME set or sumo in PATH.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

# SUMO TraCI (must run before importing traci)
from sumo_utils import (
    B1_INCOMING_LANES,
    B1_PHASES,
    CONTROLLED_TL_ID,
    GREEN_PHASE_INDICES,
    SUMO_CONFIG,
    add_sumo_to_path,
    find_sumo_bin,
)

add_sumo_to_path()
import traci

# Control interval: change phase at most every N simulation seconds
DEFAULT_CONTROL_INTERVAL = 5
DEFAULT_MAX_STEP = 360
DEFAULT_SEED = 42


def get_waiting_time_total() -> float:
    """Total waiting time (s) over all vehicles in the network."""
    total = 0.0
    for veh_id in traci.vehicle.getIDList():
        total += traci.vehicle.getWaitingTime(veh_id)
    return total


def get_queue_length() -> int:
    """Number of halting vehicles on B1 incoming lanes (queue length at intersection)."""
    return sum(
        traci.lane.getLastStepHaltingNumber(lane_id)
        for lane_id in B1_INCOMING_LANES
    )


def get_average_speed() -> float:
    """Mean speed (m/s) over all vehicles in the network. Returns 0 if no vehicles."""
    ids = traci.vehicle.getIDList()
    if not ids:
        return 0.0
    return sum(traci.vehicle.getSpeed(v) for v in ids) / len(ids)


def run(
    max_step: int = DEFAULT_MAX_STEP,
    control_interval: int = DEFAULT_CONTROL_INTERVAL,
    seed: int = DEFAULT_SEED,
    use_gui: bool = False,
    log_interval: int = 10,
    output_csv: str | None = None,
) -> None:
    """Run simulation with random traffic light agent and collect KPIs."""
    random.seed(seed)
    if not SUMO_CONFIG.exists():
        print(f"Config not found: {SUMO_CONFIG}", file=sys.stderr)
        sys.exit(1)
    sumo_bin = find_sumo_bin(use_gui)
    if not sumo_bin:
        print("SUMO not found. Set SUMO_HOME or add sumo/bin to PATH.", file=sys.stderr)
        sys.exit(1)

    sumo_cmd = [
        sumo_bin,
        "-c", str(SUMO_CONFIG),
        "--no-step-log",
        "--no-warnings",
        "--end", str(max_step),
    ]
    os.chdir(SUMO_CONFIG.parent)
    traci.start(sumo_cmd)

    log_rows: list[dict] = []
    next_control_at = 0
    current_phase_index = 0

    try:
        while True:
            step = int(traci.simulation.getTime())

            # Random phase switch at control steps (only green phases)
            if step >= next_control_at:
                current_phase_index = random.randrange(len(GREEN_PHASE_INDICES))
                state_index = GREEN_PHASE_INDICES[current_phase_index]
                traci.trafficlight.setRedYellowGreenState(
                    CONTROLLED_TL_ID, B1_PHASES[state_index]
                )
                next_control_at = step + control_interval

            waiting_time = get_waiting_time_total()
            queue_length = get_queue_length()
            avg_speed = get_average_speed()

            row = {
                "step": step,
                "waiting_time": waiting_time,
                "queue_length": queue_length,
                "average_speed": avg_speed,
            }
            log_rows.append(row)

            if log_interval > 0 and step % log_interval == 0:
                print(
                    f"Step {step:4d} | waiting_time={waiting_time:7.1f} s | "
                    f"queue={queue_length:3d} | avg_speed={avg_speed:.2f} m/s"
                )

            traci.simulationStep()

            if step >= max_step - 1:
                break
            if traci.simulation.getMinExpectedNumber() < 0:
                break
    finally:
        traci.close()

    # Summary
    n = len(log_rows)
    if n == 0:
        print("No steps recorded.")
        return
    avg_waiting = sum(r["waiting_time"] for r in log_rows) / n
    avg_queue = sum(r["queue_length"] for r in log_rows) / n
    avg_speed_over_run = sum(r["average_speed"] for r in log_rows) / n
    total_waiting = sum(r["waiting_time"] for r in log_rows)

    print("\n--- Summary ---")
    print(f"Steps:           {n}")
    print(f"Avg waiting time: {avg_waiting:.1f} s")
    print(f"Total waiting:   {total_waiting:.1f} s")
    print(f"Avg queue length: {avg_queue:.1f}")
    print(f"Avg speed:       {avg_speed_over_run:.2f} m/s")

    if output_csv:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("step,waiting_time,queue_length,average_speed\n")
            for r in log_rows:
                f.write(f"{r['step']},{r['waiting_time']:.2f},{r['queue_length']},{r['average_speed']:.2f}\n")
        print(f"Log written to: {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 3: Random traffic light agent")
    p.add_argument("--steps", type=int, default=DEFAULT_MAX_STEP, help="Max simulation steps")
    p.add_argument("--interval", type=int, default=DEFAULT_CONTROL_INTERVAL,
                   help="Seconds between random phase changes")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    p.add_argument("--gui", action="store_true", help="Use sumo-gui")
    p.add_argument("--log-every", type=int, default=10, help="Print log every N steps (0=only summary)")
    p.add_argument("--csv", type=str, default=None, help="Write step log to CSV file")
    args = p.parse_args()
    run(
        max_step=args.steps,
        control_interval=args.interval,
        seed=args.seed,
        use_gui=args.gui,
        log_interval=args.log_every,
        output_csv=args.csv,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
