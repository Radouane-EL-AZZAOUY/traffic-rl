#!/usr/bin/env python3
"""
Phase 9: Detect one traffic-light intersection from a SUMO net and write config.

Finds junctions with type="traffic_light", picks one (e.g. with most incoming lanes),
extracts tlLogic phases, and writes osm_intersection_config.json for the RL agent.

Usage:
    python sumo_osm/detect_intersection.py [--net area.net.xml] [--output osm_intersection_config.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NET = SCRIPT_DIR / "area.net.xml"
DEFAULT_CONFIG = SCRIPT_DIR / "osm_intersection_config.json"


def detect_intersection(net_path: Path) -> dict | None:
    tree = ET.parse(net_path)
    root = tree.getroot()
    ns = {}
    if root.tag.startswith("{"):
        ns["n"] = root.tag[1:root.tag.index("}")]
    else:
        ns["n"] = ""

    def find_all(tag: str):
        return root.findall(f".//{tag}", ns) if ns.get("n") else root.findall(f".//{tag}")
    def find_tl(tl_id: str):
        for tl in root.findall(".//tlLogic"):
            if tl.get("id") == tl_id:
                return tl
        return None

    junctions = find_all("junction")
    tl_junctions = [j for j in junctions if j.get("type") == "traffic_light"]
    if not tl_junctions:
        return None

    # Prefer junction with 4 incoming lanes (or most)
    def num_lanes(j):
        inc = j.get("incLanes") or ""
        return len([x for x in inc.split() if x.strip()])

    best = max(tl_junctions, key=num_lanes)
    tl_id = best.get("id")
    inc_lanes = [x.strip() for x in (best.get("incLanes") or "").split() if x.strip()]

    tl_elem = find_tl(tl_id)
    phases = []
    if tl_elem is not None:
        for phase in tl_elem.findall("phase"):
            state = phase.get("state", "")
            phases.append({"duration": int(phase.get("duration", 0)), "state": state})

    # Indices of "green" phases (state contains 'G' or 'g') for the RL agent's 2 actions
    green_indices = [i for i, p in enumerate(phases) if "G" in p["state"] or "g" in p["state"]]
    if len(green_indices) < 2:
        green_indices = [0, 1] if len(phases) >= 2 else [0, 0]
    green_indices = green_indices[:2]

    return {
        "tl_id": tl_id,
        "incoming_lanes": inc_lanes,
        "phases": [p["state"] for p in phases],
        "green_phase_indices": green_indices,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Detect one TL intersection from SUMO net")
    p.add_argument("--net", type=str, default=str(DEFAULT_NET))
    p.add_argument("--output", type=str, default=str(DEFAULT_CONFIG))
    args = p.parse_args()
    net_path = Path(args.net)
    if not net_path.is_absolute():
        net_path = SCRIPT_DIR / net_path
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = SCRIPT_DIR / out_path
    if not net_path.exists():
        print(f"Net not found: {net_path}", file=sys.stderr)
        return 1
    cfg = detect_intersection(net_path)
    if not cfg:
        print("No traffic-light junction found in net.", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config written to {out_path}: tl_id={cfg['tl_id']}, lanes={len(cfg['incoming_lanes'])}, green_indices={cfg['green_phase_indices']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
