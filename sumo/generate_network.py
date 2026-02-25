#!/usr/bin/env python3
"""
Generate intersection.net.xml for Phase 1.

Tries in order:
  1. netgenerate --grid (3x3 grid, one central 4-way traffic light)
  2. netconvert from plain XML (intersection.nod.xml, intersection.edg.xml)

Requires SUMO to be installed (SUMO_HOME set or in PATH).

Usage (from project root):
    python sumo/generate_network.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_NET = SCRIPT_DIR / "intersection.net.xml"
NOD_FILE = SCRIPT_DIR / "intersection.nod.xml"
EDG_FILE = SCRIPT_DIR / "intersection.edg.xml"


def find_sumo_bin(name: str) -> str | None:
    """Locate a SUMO binary (e.g. netgenerate or netconvert)."""
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        for suffix in ("", ".exe"):
            exe = Path(sumo_home) / "bin" / (name + suffix)
            if exe.exists():
                return str(exe)
    for prefix in [
        Path(os.environ.get("ProgramFiles(X86)", "C:\\Program Files (x86)")) / "Eclipse SUMO",
        Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "Eclipse SUMO",
    ]:
        exe = prefix / "bin" / (name + ".exe")
        if exe.exists():
            return str(exe)
    try:
        subprocess.run([name, "--version"], capture_output=True, check=True, timeout=5)
        return name
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def run_netgenerate() -> bool:
    """Generate net using grid (3x3, one 4-way center)."""
    exe = find_sumo_bin("netgenerate")
    if not exe:
        return False
    cmd = [
        exe,
        "--output-file", str(OUTPUT_NET),
        "--grid",
        "--grid.number", "3",
        "--grid.length", "200",
        "--default-junction-type", "traffic_light",
    ]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return r.returncode == 0


def run_netconvert() -> bool:
    """Generate net from plain XML (intersection.nod.xml, intersection.edg.xml)."""
    if not NOD_FILE.exists() or not EDG_FILE.exists():
        return False
    exe = find_sumo_bin("netconvert")
    if not exe:
        return False
    cmd = [
        exe,
        "--node-files", str(NOD_FILE),
        "--edge-files", str(EDG_FILE),
        "--output-file", str(OUTPUT_NET),
    ]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return r.returncode == 0


def main() -> int:
    os.chdir(SCRIPT_DIR)
    if run_netgenerate():
        print("Generated (netgenerate):", OUTPUT_NET)
        return 0
    if run_netconvert():
        print("Generated (netconvert):", OUTPUT_NET)
        return 0
    print(
        "SUMO not found. Set SUMO_HOME or install SUMO and add bin/ to PATH.",
        file=sys.stderr,
    )
    print("Example: set SUMO_HOME=C:\\Program Files (x86)\\Eclipse SUMO", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
