"""
Phase 7: FastAPI backend.

- GET /kpis   — latest KPIs (from current or last completed run)
- GET /state  — current simulation state (sim_time, lane counts, waiting, etc.)
- GET /phases — current traffic light phase
- POST /run  — start simulation in background

Run: uvicorn backend.main:app --reload
From project root: uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# Ensure project root on path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.kpi_service import KPICollector, run_simulation_and_collect_kpis

app = FastAPI(
    title="Smart Traffic Light API",
    description="KPIs, state, phases, and simulation control",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread-safe shared state (updated by background runner, read by API)
_lock = threading.Lock()
_running = False
_sim_time: float = 0.0
_current_phase: str | None = None
_lane_vehicle_counts: list[int] = []
_total_waiting: float = 0.0
_queue_length: int = 0
_avg_speed: float = 0.0
_current_collector: KPICollector | None = None
_last_kpis: dict[str, Any] | None = None
_vehicle_positions: list[dict[str, Any]] = []


def _parse_network_shapes() -> list[list[list[float]]]:
    """Parse SUMO net XML and return lane shapes for the map. Each shape is [[y,x], ...] for Leaflet CRS.Simple."""
    net_path = PROJECT_ROOT / "sumo" / "intersection.net.xml"
    if not net_path.exists():
        return []
    shapes: list[list[list[float]]] = []
    tree = ET.parse(net_path)
    root = tree.getroot()
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        for lane in edge.findall("lane"):
            shape_attr = lane.get("shape")
            if not shape_attr:
                continue
            # SUMO shape is "x,y x,y ..." -> Leaflet [lat,lng] = [y,x]
            points: list[list[float]] = []
            for part in shape_attr.strip().split():
                coords = part.split(",")
                if len(coords) >= 2:
                    x, y = float(coords[0]), float(coords[1])
                    points.append([y, x])
            if len(points) >= 2:
                shapes.append(points)
    return shapes


def _on_step(
    step: float,
    current_phase: str | None,
    lane_counts: list,
    total_waiting: float,
    queue_length: int,
    mean_speed: float,
    collector: KPICollector,
    vehicle_positions: list[dict[str, Any]] | None = None,
) -> None:
    with _lock:
        global _sim_time, _current_phase, _lane_vehicle_counts
        global _total_waiting, _queue_length, _avg_speed, _current_collector, _vehicle_positions
        _sim_time = step
        _current_phase = current_phase or "unknown"
        _lane_vehicle_counts = list(lane_counts) if lane_counts else []
        _total_waiting = total_waiting
        _queue_length = queue_length
        _avg_speed = mean_speed
        _current_collector = collector
        _vehicle_positions = list(vehicle_positions) if vehicle_positions else []


def _run_simulation(controller: str, sim_end: int, control_interval: int, model_path: str | None, seed: int | None) -> None:
    global _running, _last_kpis
    try:
        results = run_simulation_and_collect_kpis(
            sim_end=sim_end,
            control_interval=control_interval,
            controller=controller,
            model_path=model_path or (str(PROJECT_ROOT / "rl" / "models" / "dqn_traffic_light.zip") if controller == "rl" else None),
            seed=seed,
            on_step=_on_step,
        )
        with _lock:
            _last_kpis = results
    except Exception as e:
        with _lock:
            _last_kpis = {"error": str(e)}
    finally:
        with _lock:
            _running = False


class RunRequest(BaseModel):
    controller: str = Field(default="fixed", description="fixed | random | rl")
    sim_end: int = Field(default=360, ge=1, le=7200, description="Simulation end time (s)")
    control_interval: int = Field(default=5, ge=1, le=60)
    model_path: str | None = Field(default=None, description="Path to DQN model when controller=rl")
    seed: int | None = Field(default=42)


@app.get("/kpis", response_model=dict)
def get_kpis() -> dict:
    """Return latest KPIs (from current run so far, or last completed run)."""
    with _lock:
        if _current_collector is not None and _running:
            return _current_collector.get_results()
        if _last_kpis is not None:
            return _last_kpis
    raise HTTPException(status_code=404, detail="No KPI data yet. Start a simulation with POST /run.")


@app.get("/state", response_model=dict)
def get_state() -> dict:
    """Return current simulation state (live if running, else last snapshot)."""
    with _lock:
        return {
            "running": _running,
            "sim_time": _sim_time,
            "lane_vehicle_counts": _lane_vehicle_counts,
            "total_waiting": _total_waiting,
            "queue_length": _queue_length,
            "avg_speed": _avg_speed,
        }


@app.get("/vehicles", response_model=dict)
def get_vehicles() -> dict:
    """Return current vehicle positions (when simulation is running). For map markers."""
    with _lock:
        return {"vehicles": _vehicle_positions, "running": _running}


@app.get("/phases", response_model=dict)
def get_phases() -> dict:
    """Return current traffic light phase at the controlled intersection."""
    with _lock:
        return {
            "tl_id": "B1",
            "current_phase": _current_phase or "unknown",
            "running": _running,
        }


@app.post("/run", response_model=dict)
def post_run(body: RunRequest) -> dict:
    """Start a simulation in the background. Use GET /state and GET /kpis for progress."""
    global _running
    with _lock:
        if _running:
            raise HTTPException(status_code=409, detail="Simulation already running.")
        _running = True
        _last_kpis = None
        _current_collector = None
    thread = threading.Thread(
        target=_run_simulation,
        kwargs=dict(
            controller=body.controller,
            sim_end=body.sim_end,
            control_interval=body.control_interval,
            model_path=body.model_path,
            seed=body.seed,
        ),
        daemon=True,
    )
    thread.start()
    return {"status": "started", "controller": body.controller, "sim_end": body.sim_end}


@app.get("/network", response_model=dict)
def get_network() -> dict:
    """Return SUMO network lane shapes for the dashboard map. Each shape is [[y,x], ...] for Leaflet CRS.Simple."""
    shapes = _parse_network_shapes()
    return {"shapes": shapes}


@app.get("/")
def root() -> dict:
    return {"message": "Smart Traffic Light API", "docs": "/docs", "dashboard": "/dashboard"}


# Serve dashboard at /dashboard (open http://127.0.0.1:8000/dashboard)
_frontend_dir = PROJECT_ROOT / "frontend"
if _frontend_dir.exists():
    @app.get("/dashboard", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(_frontend_dir / "index.html")

    app.mount("/dashboard", StaticFiles(directory=_frontend_dir), name="dashboard")
