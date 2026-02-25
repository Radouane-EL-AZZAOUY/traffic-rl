
Here’s what `generate_network.py` does and how it works.

## Purpose

The script builds a SUMO road network file **`intersection.net.xml`** for Phase 1 of the traffic-rl project. It tries two methods in order:

1. **netgenerate** – grid-based generation (3×3 grid, one central 4-way intersection).
2. **netconvert** – build from plain XML node/edge files (`intersection.nod.xml`, `intersection.edg.xml`).

If SUMO isn’t found or both methods fail, it prints an error and exits with code 1.

---

## Structure

### Paths (lines 21–24)

```21:24:d:\FSTM\ILISI2\GL_Workshop\traffic-rl\sumo\generate_network.py
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_NET = SCRIPT_DIR / "intersection.net.xml"
NOD_FILE = SCRIPT_DIR / "intersection.nod.xml"
EDG_FILE = SCRIPT_DIR / "intersection.edg.xml"
```

- **SCRIPT_DIR**: directory of the script (the `sumo` folder).
- **OUTPUT_NET**: where the final network is written (`intersection.net.xml`).
- **NOD_FILE**, **EDG_FILE**: node and edge definition files used by the fallback `netconvert` method.

---

## `find_sumo_bin(name: str) -> str | None` (lines 27–46)

Finds a SUMO binary (e.g. `netgenerate` or `netconvert`) by:

1. **SUMO_HOME**: if set, looks for `SUMO_HOME/bin/<name>` (with and without `.exe`).
2. **Windows install paths**: checks  
   `Program Files (x86)/Eclipse SUMO/bin/<name>.exe` and  
   `Program Files/Eclipse SUMO/bin/<name>.exe`.
3. **PATH**: runs `name --version`; if it succeeds, returns `name` (assumes it’s on PATH).

Returns the path/name of the binary if found, otherwise `None`.

---

## `run_netgenerate() -> bool` (lines 49–64)

Uses SUMO’s **netgenerate** to create the network:

- **Grid**: `--grid` with `--grid.number 3` → 3×3 grid of junctions (one central 4-way).
- **Edge length**: `--grid.length 200` (200 m per grid cell).
- **Junction type**: `--default-junction-type traffic_light` so junctions get traffic lights.
- **Output**: writes to `OUTPUT_NET` (`intersection.net.xml`).

Runs the command in `SCRIPT_DIR`. Returns `True` if the process exits with 0, else `False`.

---

## `run_netconvert() -> bool` (lines 67–82)

Fallback: builds the network from **plain XML** node and edge files:

- Requires both `intersection.nod.xml` and `intersection.edg.xml` to exist; if either is missing, returns `False`.
- Runs **netconvert** with:
  - `--node-files` → `intersection.nod.xml`
  - `--edge-files` → `intersection.edg.xml`
  - `--output-file` → `intersection.net.xml`

Again runs in `SCRIPT_DIR`. Returns `True` only if the command exits with 0.

---

## `main() -> int` (lines 85–98)

1. Changes the current directory to `SCRIPT_DIR` so all paths are relative to the `sumo` folder.
2. Tries **netgenerate** first:
   - If it succeeds: prints that the net was generated with netgenerate and returns `0`.
3. If netgenerate failed (or wasn’t found), tries **netconvert**:
   - If it succeeds: prints that the net was generated with netconvert and returns `0`.
4. If both fail:
   - Prints to stderr that SUMO wasn’t found and suggests setting `SUMO_HOME` (with an example Windows path).
   - Returns `1`.

---

## Execution flow (summary)

1. `main()` runs in the script’s directory.
2. Call `run_netgenerate()`:
   - `find_sumo_bin("netgenerate")` locates the binary.
   - If found, run netgenerate with grid options and write `intersection.net.xml`.
   - Success → exit 0.
3. If not, call `run_netconvert()`:
   - Check that `intersection.nod.xml` and `intersection.edg.xml` exist.
   - `find_sumo_bin("netconvert")` locates netconvert.
   - If both exist and netconvert is found, run netconvert to produce `intersection.net.xml`.
   - Success → exit 0.
4. If both methods fail, print the SUMO-not-found message and exit with 1.

So: the script **automatically chooses** between a grid-generated network and a hand-defined node/edge network, and writes the result to `sumo/intersection.net.xml` for use in the traffic-rl SUMO simulation.