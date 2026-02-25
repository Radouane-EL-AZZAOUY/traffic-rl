#!/usr/bin/env python3
"""
Phase 2: Manual Traffic Light Control with TraCI.

- Connects to SUMO using TraCI.
- Prints lane vehicle counts for the center intersection (B1) each step.
- Manually switches traffic light phases every N seconds (cycle: NS green -> EW green).

Run from project root:
    python sumo/traci_manual_control.py

Or from sumo/:
    python traci_manual_control.py

Requires: SUMO installed, SUMO_HOME set or sumo/sumo-gui in PATH.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add SUMO tools to path for traci
SCRIPT_DIR = Path(__file__).resolve().parent
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    for prefix in (
        Path(os.environ.get("ProgramFiles(X86)", "C:\\Program Files (x86)")) / "Eclipse SUMO",
        Path(os.environ.get("ProgramFiles(X86)", "C:\\Program Files (x86)")) / "Eclipse" / "Sumo",
        Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "Eclipse SUMO",
    ):
        tools = prefix / "tools"
        if tools.exists():
            sys.path.insert(0, str(tools))
            break

import traci

# -----------------------------------------------------------------------------
# Center intersection (B1) configuration (from intersection.net.xml)
# -----------------------------------------------------------------------------
CONTROLLED_TL_ID = "B1"
# Incoming lanes to B1 (West, South, North, East approach)
B1_INCOMING_LANES = ("A1B1_0", "B0B1_0", "B2B1_0", "C1B1_0")
# B1 phase states: NS green, yellow, EW green, yellow (from net)
B1_PHASES = (
    "GGggrrrrGGggrrrr",  # 0: North-South green
    "yyyyrrrryyyyrrrr",  # 1: NS yellow
    "rrrrGGggrrrrGGgg",  # 2: East-West green
    "rrrryyyyrrrryyyy",  # 3: EW yellow
)
# How often to manually switch phase (simulation seconds)
PHASE_SWITCH_INTERVAL = 30
# Only cycle green phases (0=NS, 2=EW) for clearer demo; set to (0,1,2,3) for full cycle
GREEN_PHASE_INDICES = (0, 2)
# Sim duration (set to 0 to use config end)
MAX_STEP = 300


def find_sumo_bin(use_gui: bool = False) -> str | None:
    """Locate sumo or sumo-gui executable."""
    name = "sumo-gui" if use_gui else "sumo"
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        for suffix in ("", ".exe"):
            exe = Path(sumo_home) / "bin" / (name + suffix)
            if exe.exists():
                return str(exe)
    for prefix in (
        Path(os.environ.get("ProgramFiles(X86)", "C:\\Program Files (x86)")) / "Eclipse SUMO",
        Path(os.environ.get("ProgramFiles(X86)", "C:\\Program Files (x86)")) / "Eclipse" / "Sumo",
        Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "Eclipse SUMO",
    ):
        exe = prefix / "bin" / (name + ".exe")
        if exe.exists():
            return str(exe)
    return None


def get_lane_vehicle_counts() -> dict[str, int]:
    """Return last-step vehicle count per lane for B1 incoming lanes."""
    counts = {}
    for lane_id in B1_INCOMING_LANES:
        counts[lane_id] = traci.lane.getLastStepVehicleNumber(lane_id)
    return counts


def run(use_gui: bool = False, max_step: int = MAX_STEP) -> None:
    """Run simulation with TraCI: print lane counts and manually switch phases."""
    sumo_bin = find_sumo_bin(use_gui)
    if not sumo_bin:
        print("SUMO not found. Set SUMO_HOME or add sumo/bin to PATH.", file=sys.stderr)
        sys.exit(1)

    config = SCRIPT_DIR / "simulation.sumocfg"
    if not config.exists():
        print(f"Config not found: {config}", file=sys.stderr)
        sys.exit(1)

    sumo_cmd = [
        sumo_bin,
        "-c", str(config),
        "--no-step-log",
        "--no-warnings",
    ]
    if max_step > 0:
        sumo_cmd.extend(["--end", str(max_step)])

    print("Starting SUMO with TraCI:", " ".join(sumo_cmd))
    os.chdir(SCRIPT_DIR)
    traci.start(sumo_cmd)

    try:
        # Cycle only green phases (NS <-> EW) for demo
        phase_cycle = GREEN_PHASE_INDICES
        current_phase_index = 0
        next_switch_at = PHASE_SWITCH_INTERVAL

        while True:
            step = int(traci.simulation.getTime())

            # Manual phase switch: every PHASE_SWITCH_INTERVAL seconds
            if step > 0 and step >= next_switch_at:
                current_phase_index = (current_phase_index + 1) % len(phase_cycle)
                state_index = phase_cycle[current_phase_index]
                new_state = B1_PHASES[state_index]
                traci.trafficlight.setRedYellowGreenState(CONTROLLED_TL_ID, new_state)
                next_switch_at = step + PHASE_SWITCH_INTERVAL
                print(f"  [Phase switch] step {step} -> {('NS','EW')[current_phase_index]} green")

            # Lane vehicle counts
            counts = get_lane_vehicle_counts()
            total = sum(counts.values())
            print(f"Step {step:4d} | Lanes: {counts} | Total: {total}")

            traci.simulationStep()

            if max_step > 0 and step >= max_step - 1:
                break
            if traci.simulation.getMinExpectedNumber() < 0:
                break
    finally:
        traci.close()

    print("Simulation finished.")


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Phase 2: TraCI manual traffic light control")
    p.add_argument("--gui", action="store_true", help="Use sumo-gui instead of sumo")
    p.add_argument("--steps", type=int, default=MAX_STEP, help=f"Max simulation steps (0=use config, default {MAX_STEP})")
    args = p.parse_args()
    run(use_gui=args.gui, max_step=args.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
