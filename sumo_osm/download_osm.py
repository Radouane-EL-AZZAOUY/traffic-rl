#!/usr/bin/env python3
"""
Phase 9: Download a small OSM area via Overpass API.

Saves OSM XML to sumo_osm/area.osm.xml (or custom output path).
Default bbox: small area (e.g. a few streets).

Usage:
    python sumo_osm/download_osm.py
    python sumo_osm/download_osm.py --south 48.84 --west 2.34 --north 48.86 --east 2.36 --output my.osm.xml
"""
from __future__ import annotations

import argparse
import os
import urllib.request
import urllib.parse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = "area.osm.xml"


def download_bbox(south: float, west: float, north: float, east: float, output_path: str) -> None:
    # Overpass: nwr = nodes, ways, relations in bbox
    query = f"[out:xml];nwr({south},{west},{north},{east});out body;"
    url = "https://overpass-api.de/api/interpreter?" + urllib.parse.urlencode({"data": query})
    req = urllib.request.Request(url, headers={"User-Agent": "traffic-rl-phase9/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    with open(output_path, "wb") as f:
        f.write(data)
    print(f"Downloaded OSM data to {output_path} ({len(data)} bytes)")


def main() -> int:
    p = argparse.ArgumentParser(description="Download OSM area (Overpass API)")
    p.add_argument("--south", type=float, default=48.84, help="South latitude")
    p.add_argument("--west", type=float, default=2.34, help="West longitude")
    p.add_argument("--north", type=float, default=48.86, help="North latitude")
    p.add_argument("--east", type=float, default=2.36, help="East longitude")
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output .osm.xml path")
    args = p.parse_args()
    out = args.output
    if not os.path.isabs(out):
        out = os.path.join(SCRIPT_DIR, out)
    download_bbox(args.south, args.west, args.north, args.east, out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
