#!/usr/bin/env python3
"""
Phase 9: Generate routes for the OSM net using SUMO's randomTrips.py.

Produces area.rou.xml (and optionally area.trips.xml).
Requires SUMO tools (randomTrips.py, duarouter).

Usage:
    python sumo_osm/generate_routes.py [--net area.net.xml] [--end 3600]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NET = SCRIPT_DIR / "area.net.xml"
DEFAULT_TRIPS = SCRIPT_DIR / "area.trips.xml"
DEFAULT_ROUTES = SCRIPT_DIR / "area.rou.xml"


def find_sumo_tools() -> tuple[str | None, str | None]:
    """Return (path to randomTrips.py, path to duarouter or netconvert)."""
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        root = Path(sumo_home)
        trips = root / "tools" / "randomTrips.py"
        if trips.exists():
            duarouter = root / "bin" / "duarouter.exe" if os.name == "nt" else root / "bin" / "duarouter"
            if not duarouter.exists():
                duarouter = root / "bin" / "duarouter"
            return (str(trips), str(duarouter) if duarouter.exists() else None)
    for prefix in (
        Path(os.environ.get("ProgramFiles(X86)", "C:\\Program Files (x86)")) / "Eclipse SUMO",
        Path(os.environ.get("ProgramFiles(X86)", "C:\\Program Files (x86)")) / "Eclipse" / "Sumo",
    ):
        trips = prefix / "tools" / "randomTrips.py"
        if trips.exists():
            dur = prefix / "bin" / "duarouter.exe"
            return (str(trips), str(dur) if dur.exists() else None)
    return (None, None)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate routes for OSM net (randomTrips + duarouter)")
    p.add_argument("--net", type=str, default=str(DEFAULT_NET))
    p.add_argument("--end", type=int, default=3600, help="End time for trip generation (s)")
    p.add_argument("--period", type=float, default=2.0, help="Mean period between vehicle departures")
    args = p.parse_args()
    net_path = Path(args.net)
    if not net_path.is_absolute():
        net_path = SCRIPT_DIR / net_path
    if not net_path.exists():
        print(f"Net not found: {net_path}. Run build_net.py first.", file=sys.stderr)
        return 1
    trips_script, duarouter = find_sumo_tools()
    if not trips_script:
        print("SUMO tools not found (randomTrips.py). Set SUMO_HOME.", file=sys.stderr)
        return 1
    trips_path = SCRIPT_DIR / DEFAULT_TRIPS.name
    routes_path = SCRIPT_DIR / DEFAULT_ROUTES.name

    # randomTrips.py -n net.xml -o trips.xml -e end -p period
    cmd_trips = [
        sys.executable,
        trips_script,
        "-n", str(net_path),
        "-o", str(trips_path),
        "-e", str(args.end),
        "-p", str(args.period),
    ]
    print("Running randomTrips:", " ".join(cmd_trips))
    r = subprocess.run(cmd_trips, cwd=str(SCRIPT_DIR))
    if r.returncode != 0:
        return r.returncode
    if not trips_path.exists():
        print("randomTrips did not produce trips file.", file=sys.stderr)
        return 1

    if duarouter:
        cmd_dua = [
            duarouter,
            "-n", str(net_path),
            "-t", str(trips_path),
            "-o", str(routes_path),
            "--ignore-errors",
        ]
        print("Running duarouter:", " ".join(cmd_dua))
        r = subprocess.run(cmd_dua, cwd=str(SCRIPT_DIR))
        if r.returncode != 0:
            print("duarouter failed; you may use trips file directly if SUMO accepts it.", file=sys.stderr)
        elif routes_path.exists():
            print(f"Routes written to {routes_path}")
    else:
        print("duarouter not found; trips file at", trips_path, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
