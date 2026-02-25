/**
 * Phase 8 — Dashboard: Leaflet map, vehicle markers, Chart.js, phase display.
 * Polls backend for /state, /phases, /vehicles when simulation is running.
 */
// Backend API (run: uvicorn backend.main:app --port 8000)
const API_BASE = "http://127.0.0.1:8000";

// SUMO net bounds (from intersection.net.xml convBoundary 0,0,400,400)
const MAP_BOUNDS = [[0, 0], [400, 400]];
const MAP_CENTER = [200, 200];

let map = null;
let vehicleMarkers = [];
let pollTimer = null;
let chartWaiting = null;
let chartSpeed = null;
let chartReward = null;
const chartData = { times: [], waiting: [], speed: [], reward: [] };

function initMap() {
  map = L.map("map", {
    crs: L.CRS.Simple,
    minZoom: -1,
    maxZoom: 2,
  });
  const bounds = L.latLngBounds(MAP_BOUNDS);
  map.setMaxBounds(bounds);
  map.setView(MAP_CENTER, 0);
  map.setMaxBounds(bounds.pad(0.1));

  // Optional: rectangle to show network area
  L.rectangle(bounds, {
    color: "#2d3a4d",
    fillOpacity: 0.05,
    weight: 1,
  }).addTo(map);
}

function updateVehicleMarkers(vehicles) {
  if (!map) return;
  vehicleMarkers.forEach((m) => map.removeLayer(m));
  vehicleMarkers = [];
  if (!Array.isArray(vehicles)) return;
  vehicles.forEach((v) => {
    const lat = v.y;
    const lng = v.x;
    const marker = L.circleMarker([lat, lng], {
      radius: 5,
      fillColor: "#58a6ff",
      color: "#1a2332",
      weight: 1,
      fillOpacity: 0.9,
    }).addTo(map);
    marker.bindTooltip(v.id || "", { permanent: false });
    vehicleMarkers.push(marker);
  });
}

function initCharts() {
  const opts = {
    type: "line",
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { title: { display: true, text: "Time (s)" }, min: 0 },
        y: { beginAtZero: true },
      },
    },
  };

  chartWaiting = new Chart(document.getElementById("chart-waiting"), {
    ...opts,
    data: {
      labels: chartData.times,
      datasets: [{ label: "Total waiting (s)", data: chartData.waiting, borderColor: "#d29922", fill: false, tension: 0.1 }],
    },
  });

  chartSpeed = new Chart(document.getElementById("chart-speed"), {
    ...opts,
    data: {
      labels: chartData.times,
      datasets: [{ label: "Avg speed (m/s)", data: chartData.speed, borderColor: "#3fb950", fill: false, tension: 0.1 }],
    },
  });

  chartReward = new Chart(document.getElementById("chart-reward"), {
    ...opts,
    data: {
      labels: chartData.times,
      datasets: [{ label: "Reward (−waiting)", data: chartData.reward, borderColor: "#58a6ff", fill: false, tension: 0.1 }],
    },
  });
}

function pushChartPoint(simTime, totalWaiting, avgSpeed) {
  chartData.times.push(Math.round(simTime));
  chartData.waiting.push(totalWaiting);
  chartData.speed.push(avgSpeed);
  chartData.reward.push(-totalWaiting);
  const maxPoints = 200;
  if (chartData.times.length > maxPoints) {
    chartData.times.shift();
    chartData.waiting.shift();
    chartData.speed.shift();
    chartData.reward.shift();
  }
  if (chartWaiting) {
    chartWaiting.data.labels = chartData.times;
    chartWaiting.data.datasets[0].data = chartData.waiting;
    chartWaiting.update("none");
  }
  if (chartSpeed) {
    chartSpeed.data.labels = chartData.times;
    chartSpeed.data.datasets[0].data = chartData.speed;
    chartSpeed.update("none");
  }
  if (chartReward) {
    chartReward.data.labels = chartData.times;
    chartReward.data.datasets[0].data = chartData.reward;
    chartReward.update("none");
  }
}

function setPhaseBadge(phaseStr, running) {
  const el = document.getElementById("phase-badge");
  if (!el) return;
  el.textContent = running ? `Phase: ${phaseStr}` : "Phase: —";
  el.classList.remove("ns", "ew");
  if (phaseStr && phaseStr.includes("GGgg")) el.classList.add("ns");
  else if (phaseStr && phaseStr.includes("rrrrGGgg")) el.classList.add("ew");
}

function setStatus(text) {
  const el = document.getElementById("status-text");
  if (el) el.textContent = text;
}

async function poll() {
  try {
    const [stateRes, phasesRes, vehiclesRes] = await Promise.all([
      fetch(`${API_BASE}/state`),
      fetch(`${API_BASE}/phases`),
      fetch(`${API_BASE}/vehicles`),
    ]);
    const state = await stateRes.json();
    const phases = await phasesRes.json();
    const vehicles = await vehiclesRes.json();

    setPhaseBadge(phases.current_phase || "—", state.running);
    updateVehicleMarkers(vehicles.vehicles || []);

    if (state.running && state.sim_time != null) {
      pushChartPoint(
        state.sim_time,
        state.total_waiting ?? 0,
        state.avg_speed ?? 0
      );
      setStatus(`Running — sim time: ${Math.round(state.sim_time)} s`);
    } else {
      setStatus(vehicles.running ? "Running…" : "Idle. Start a simulation.");
    }

    if (state.running) {
      pollTimer = setTimeout(poll, 500);
    } else {
      pollTimer = null;
      document.getElementById("btn-run").disabled = false;
    }
  } catch (e) {
    setStatus("Error: " + e.message);
    pollTimer = null;
    document.getElementById("btn-run").disabled = false;
  }
}

async function startSimulation() {
  const controller = document.getElementById("controller").value;
  const simEnd = parseInt(document.getElementById("sim-end").value, 10) || 120;
  document.getElementById("btn-run").disabled = true;
  chartData.times = [];
  chartData.waiting = [];
  chartData.speed = [];
  chartData.reward = [];
  setStatus("Starting…");

  try {
    const res = await fetch(`${API_BASE}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ controller, sim_end: simEnd }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    setStatus("Running…");
    pollTimer = setTimeout(poll, 300);
  } catch (e) {
    setStatus("Error: " + e.message);
    document.getElementById("btn-run").disabled = false;
  }
}

function init() {
  initMap();
  initCharts();
  document.getElementById("btn-run").addEventListener("click", startSimulation);
  poll();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
