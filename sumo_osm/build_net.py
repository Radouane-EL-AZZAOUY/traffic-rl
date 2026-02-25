#!/usr/bin/env python3
"""
Phase 9: Convert OSM file to SUMO net using netconvert.

Usage:
    python sumo_osm/build_net.py [--osm area.osm.xml] [--output area.net.xml]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OSM = SCRIPT_DIR / "area.osm.xml"
DEFAULT_NET = SCRIPT_DIR / "area.net.xml"


def find_netconvert() -> str | None:
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        for suffix in ("", ".exe"):
            exe = Path(sumo_home) / "bin" / ("netconvert" + suffix)
            if exe.exists():
                return str(exe)
    for prefix in (
        Path(os.environ.get("ProgramFiles(X86)", "C:\\Program Files (x86)")) / "Eclipse SUMO",
        Path(os.environ.get("ProgramFiles(X86)", "C:\\Program Files (x86)")) / "Eclipse" / "Sumo",
    ):
        exe = prefix / "bin" / "netconvert.exe"
        if exe.exists():
            return str(exe)
    try:
        subprocess.run(["netconvert", "--version"], capture_output=True, check=True, timeout=5)
        return "netconvert"
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="Convert OSM to SUMO net (netconvert)")
    p.add_argument("--osm", type=str, default=str(DEFAULT_OSM), help="Input .osm.xml path")
    p.add_argument("--output", type=str, default=str(DEFAULT_NET), help="Output .net.xml path")
    args = p.parse_args()
    osm_path = Path(args.osm)
    net_path = Path(args.output)
    if not osm_path.is_absolute():
        osm_path = SCRIPT_DIR / osm_path
    if not net_path.is_absolute():
        net_path = SCRIPT_DIR / net_path
    if not osm_path.exists():
        print(f"OSM file not found: {osm_path}. Run download_osm.py first.", file=sys.stderr)
        return 1
    exe = find_netconvert()
    if not exe:
        print("SUMO netconvert not found. Set SUMO_HOME or add bin to PATH.", file=sys.stderr)
        return 1
    cmd = [
        exe,
        "--osm-files", str(osm_path),
        "--output", str(net_path),
        "--geometry.remove",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--tls.discard-simple",
        "--tls.join",
    ]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    if r.returncode != 0:
        return r.returncode
    print(f"Wrote: {net_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
