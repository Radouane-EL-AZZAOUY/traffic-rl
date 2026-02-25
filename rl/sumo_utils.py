"""
Shared utilities for running SUMO with TraCI from the rl/ package.

- Adds SUMO tools to sys.path for traci import.
- Finds sumo/sumo-gui binary.
- Center intersection (B1) constants for traffic light control.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Project root (traffic-rl/)
RL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RL_DIR.parent
SUMO_DIR = PROJECT_ROOT / "sumo"
SUMO_CONFIG = SUMO_DIR / "simulation.sumocfg"

# Center intersection (B1) - must match intersection.net.xml
CONTROLLED_TL_ID = "B1"
B1_INCOMING_LANES = ("A1B1_0", "B0B1_0", "B2B1_0", "C1B1_0")
# B1 phase states: NS green, yellow, EW green, yellow (green indices 0 and 2)
B1_PHASES = (
    "GGggrrrrGGggrrrr",  # 0: North-South green
    "yyyyrrrryyyyrrrr",  # 1: NS yellow
    "rrrrGGggrrrrGGgg",  # 2: East-West green
    "rrrryyyyrrrryyyy",  # 3: EW yellow
)
GREEN_PHASE_INDICES = (0, 2)


def add_sumo_to_path() -> None:
    """Prepend SUMO tools directory to sys.path so 'import traci' works."""
    if any("traci" in p for p in sys.modules):
        return
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)
        return
    for prefix in (
        Path(os.environ.get("ProgramFiles(X86)", "C:\\Program Files (x86)")) / "Eclipse SUMO",
        Path(os.environ.get("ProgramFiles(X86)", "C:\\Program Files (x86)")) / "Eclipse" / "Sumo",
        Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "Eclipse SUMO",
    ):
        tools = prefix / "tools"
        if tools.exists():
            sys.path.insert(0, str(tools))
            return


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
    try:
        import subprocess
        subprocess.run([name, "--version"], capture_output=True, check=True, timeout=5)
        return name
    except Exception:
        return None
