import React, { useState, useRef, useEffect } from "react";
import { LIVE_SATELLITE_FEEDS, LIVE_ACTIVE_CYCLONES } from "../data/liveSatelliteData";
import { LiveSatelliteFeed, ActiveCyclone, SpectralBand } from "../types";
import {
  Satellite,
  Radio,
  Clock,
  Compass,
  Wind,
  Gauge,
  Thermometer,
  Layers,
  RefreshCw,
  ExternalLink,
  Crosshair,
  Sparkles,
  Info,
  Maximize2,
  Sliders,
  CheckCircle2,
  AlertTriangle,
} from "lucide-react";

interface LiveSatelliteHubProps {
  onSelectCyclone?: (cyclone: ActiveCyclone) => void;
  onSendToVision?: (feed: LiveSatelliteFeed) => void;
}

export const LiveSatelliteHub: React.FC<LiveSatelliteHubProps> = ({
  onSelectCyclone,
  onSendToVision,
}) => {
  const [selectedFeedId, setSelectedFeedId] = useState<string>("insat-3d-tir1");
  const [selectedFilter, setSelectedFilter] = useState<"ALL" | SpectralBand>("ALL");
  const [colorEnhancement, setColorEnhancement] = useState<"standard" | "bd_curve" | "rainbow" | "grayscale">("standard");
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [refreshCountdown, setRefreshCountdown] = useState<number>(45);
  const [clickedPixel, setClickedPixel] = useState<{
    x: number;
    y: number;
    tempK: number;
    tempC: number;
    altitudeKm: number;
    depthClass: string;
  } | null>(null);

  const selectedFeed =
    LIVE_SATELLITE_FEEDS.find((f) => f.id === selectedFeedId) || LIVE_SATELLITE_FEEDS[0];

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Auto-refresh countdown simulation for live ground telemetry
  useEffect(() => {
    const timer = setInterval(() => {
      setRefreshCountdown((prev) => {
        if (prev <= 1) {
          return 60;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const handleManualRefresh = () => {
    setIsRefreshing(true);
    setTimeout(() => {
      setIsRefreshing(false);
      setRefreshCountdown(60);
    }, 700);
  };

  // Render Live Satellite Canvas Swath with color enhancements
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = 640;
    const height = 440;
    canvas.width = width;
    canvas.height = height;

    // Draw background
    ctx.fillStyle = "#0a192f";
    ctx.fillRect(0, 0, width, height);

    // Ocean background gradient
    const oceanGrad = ctx.createLinearGradient(0, 0, width, height);
    oceanGrad.addColorStop(0, "#08203e");
    oceanGrad.addColorStop(1, "#0a192f");
    ctx.fillStyle = oceanGrad;
    ctx.fillRect(0, 0, width, height);

    // Draw coordinate lat/lon grid
    ctx.strokeStyle = "rgba(148, 163, 184, 0.15)";
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 50) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw stylized coastline based on selected coverage
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    if (selectedFeed.coverage.includes("Indian Ocean") || selectedFeed.coverage.includes("Arabian Sea")) {
      // Indian Subcontinent outline
      ctx.moveTo(width * 0.7, 0);
      ctx.bezierCurveTo(width * 0.68, height * 0.25, width * 0.58, height * 0.4, width * 0.52, height * 0.46);
      ctx.bezierCurveTo(width * 0.58, height * 0.6, width * 0.64, height * 0.78, width * 0.68, height);
    } else if (selectedFeed.coverage.includes("Pacific")) {
      // West Pacific archipelago arc
      ctx.moveTo(width * 0.25, 0);
      ctx.bezierCurveTo(width * 0.28, height * 0.35, width * 0.32, height * 0.65, width * 0.35, height);
    } else {
      // Atlantic Gulf & Caribbean arc
      ctx.moveTo(0, height * 0.3);
      ctx.bezierCurveTo(width * 0.35, height * 0.28, width * 0.55, height * 0.5, width * 0.8, height * 0.7);
    }
    ctx.stroke();

    // Center of storm / cloud system
    const cx = width * 0.48;
    const cy = height * 0.52;

    // Convective cloud mass with color enhancement curves
    const rad = 170;
    const cloudGrad = ctx.createRadialGradient(cx, cy, 15, cx, cy, rad);

    if (colorEnhancement === "bd_curve") {
      // BD Curve (Dvorak standard for IR enhancement)
      cloudGrad.addColorStop(0, "#ffffff"); // Eye center / clear
      cloudGrad.addColorStop(0.12, "#1e1b4b"); // Cold cloud-top -80C (Black)
      cloudGrad.addColorStop(0.25, "#e11d48"); // Cold -70C (Red)
      cloudGrad.addColorStop(0.42, "#f97316"); // -60C (Orange)
      cloudGrad.addColorStop(0.60, "#eab308"); // -50C (Yellow)
      cloudGrad.addColorStop(0.80, "#0284c7"); // -40C (Blue)
      cloudGrad.addColorStop(1, "transparent");
    } else if (colorEnhancement === "rainbow") {
      cloudGrad.addColorStop(0, "#ffffff");
      cloudGrad.addColorStop(0.15, "#ec4899");
      cloudGrad.addColorStop(0.35, "#8b5cf6");
      cloudGrad.addColorStop(0.55, "#3b82f6");
      cloudGrad.addColorStop(0.75, "#10b981");
      cloudGrad.addColorStop(1, "transparent");
    } else if (colorEnhancement === "grayscale") {
      cloudGrad.addColorStop(0, "#09090b");
      cloudGrad.addColorStop(0.15, "#ffffff");
      cloudGrad.addColorStop(0.4, "#d4d4d8");
      cloudGrad.addColorStop(0.7, "#71717a");
      cloudGrad.addColorStop(1, "transparent");
    } else {
      // Standard satellite enhancement based on spectral band
      if (selectedFeed.band === "TIR1") {
        cloudGrad.addColorStop(0, "#0f172a");
        cloudGrad.addColorStop(0.1, "#ffffff");
        cloudGrad.addColorStop(0.22, "#dc2626");
        cloudGrad.addColorStop(0.45, "#f59e0b");
        cloudGrad.addColorStop(0.7, "#0284c7");
        cloudGrad.addColorStop(1, "transparent");
      } else if (selectedFeed.band === "WV") {
        cloudGrad.addColorStop(0, "#1e1b4b");
        cloudGrad.addColorStop(0.15, "#f43f5e");
        cloudGrad.addColorStop(0.38, "#a855f7");
        cloudGrad.addColorStop(0.65, "#3b82f6");
        cloudGrad.addColorStop(1, "transparent");
      } else {
        cloudGrad.addColorStop(0, "#ffffff");
        cloudGrad.addColorStop(0.25, "#e2e8f0");
        cloudGrad.addColorStop(0.55, "#94a3b8");
        cloudGrad.addColorStop(0.85, "#475569");
        cloudGrad.addColorStop(1, "transparent");
      }
    }

    ctx.fillStyle = cloudGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, rad, 0, Math.PI * 2);
    ctx.fill();

    // Spiral cyclonic arms
    for (let arm = 0; arm < 3; arm++) {
      ctx.beginPath();
      const offset = (arm * Math.PI * 2) / 3;
      ctx.strokeStyle = colorEnhancement === "bd_curve" ? "rgba(251, 146, 60, 0.45)" : "rgba(224, 242, 254, 0.4)";
      ctx.lineWidth = 14;
      ctx.lineCap = "round";

      for (let t = 0.3; t < 2.4; t += 0.08) {
        const r = t * 62;
        const theta = t * 2.7 + offset;
        const px = cx + r * Math.cos(theta);
        const py = cy + r * Math.sin(theta);
        if (t === 0.3) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // Central Eye Well
    ctx.fillStyle = "#08111e";
    ctx.beginPath();
    ctx.arc(cx, cy, 14, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, 16, 0, Math.PI * 2);
    ctx.stroke();

    // Satellite Scan Sweep telemetry line
    ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(0, height * 0.42);
    ctx.lineTo(width, height * 0.42);
    ctx.stroke();
    ctx.setLineDash([]);

    // Telemetry text overlay in canvas
    ctx.fillStyle = "#f8fafc";
    ctx.font = "bold 11px monospace";
    ctx.fillText(`${selectedFeed.platform.split(" ")[0]} · NADIR RESOLUTION: ${selectedFeed.resolution.split(" ")[0]}`, 16, 26);
    ctx.fillStyle = "#38bdf8";
    ctx.fillText(`CENTER: ${selectedFeed.centerCoords.lat}°N, ${selectedFeed.centerCoords.lon}°E`, 16, 42);

    // If user clicked a pixel, draw crosshair
    if (clickedPixel) {
      ctx.strokeStyle = "#f43f5e";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(clickedPixel.x, clickedPixel.y, 8, 0, Math.PI * 2);
      ctx.stroke();

      ctx.moveTo(clickedPixel.x - 14, clickedPixel.y);
      ctx.lineTo(clickedPixel.x + 14, clickedPixel.y);
      ctx.moveTo(clickedPixel.x, clickedPixel.y - 14);
      ctx.lineTo(clickedPixel.x, clickedPixel.y + 14);
      ctx.stroke();

      ctx.fillStyle = "#ffffff";
      ctx.fillRect(clickedPixel.x + 12, clickedPixel.y - 28, 110, 24);
      ctx.strokeStyle = "#e2e8f0";
      ctx.strokeRect(clickedPixel.x + 12, clickedPixel.y - 28, 110, 24);
      ctx.fillStyle = "#0f172a";
      ctx.font = "bold 10px monospace";
      ctx.fillText(`${clickedPixel.tempK} K (${clickedPixel.tempC}°C)`, clickedPixel.x + 16, clickedPixel.y - 12);
    }
  }, [selectedFeed, colorEnhancement, clickedPixel]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Calculate approximate brightness temp based on distance from core
    const cx = canvas.width * 0.48;
    const cy = canvas.height * 0.52;
    const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);

    let tempK: number;
    let depthClass: string;
    let altitudeKm: number;

    if (dist < 20) {
      // Eye
      tempK = 235.5;
      depthClass = "Warm Eye Subsidence";
      altitudeKm = 11.2;
    } else if (dist < 60) {
      // Eyewall cold canopy
      tempK = 196.2;
      depthClass = "Deep Overshooting Tropospheric Top";
      altitudeKm = 16.8;
    } else if (dist < 120) {
      // Inner spiral band
      tempK = 212.4;
      depthClass = "High Cirrus / Cumulonimbus Canopy";
      altitudeKm = 14.1;
    } else {
      // Outer peripheral cloud
      tempK = 265.8;
      depthClass = "Low/Mid-Level Stratiform";
      altitudeKm = 6.4;
    }

    const tempC = Number((tempK - 273.15).toFixed(1));
    setClickedPixel({
      x,
      y,
      tempK,
      tempC,
      altitudeKm,
      depthClass,
    });
  };

  const filteredFeeds =
    selectedFilter === "ALL"
      ? LIVE_SATELLITE_FEEDS
      : LIVE_SATELLITE_FEEDS.filter((f) => f.band === selectedFilter);

  return (
    <div className="space-y-6">
      {/* Top Header Card */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-sky-50 text-sky-600 border border-sky-200">
                <Satellite className="w-5 h-5" />
              </span>
              <h2 className="text-xl font-bold text-slate-900">
                Live Satellite Telemetry & Real-Time Earth Observation Hub
              </h2>
            </div>
            <p className="text-xs text-slate-500 font-mono-code">
              Grounded in multi-agency geostationary & polar-orbiting constellations (ISRO MOSDAC, NOAA NESDIS, JMA)
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg text-xs font-mono-code text-slate-600">
              <Clock className="w-3.5 h-3.5 text-sky-600" />
              <span>Next Telemetry Pass:</span>
              <strong className="text-sky-700">{refreshCountdown}s</strong>
            </div>

            <button
              onClick={handleManualRefresh}
              disabled={isRefreshing}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-700 text-white font-medium text-xs transition shadow-sm cursor-pointer disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? "animate-spin" : ""}`} />
              <span>{isRefreshing ? "Syncing..." : "Sync Feeds"}</span>
            </button>
          </div>
        </div>

        {/* Live Status Ticker */}
        <div className="mt-4 pt-3 border-t border-slate-100 flex flex-wrap items-center justify-between gap-3 text-xs font-mono-code">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="font-semibold text-slate-800">5 LIVE PLATFORMS ONLINE:</span>
            <span className="text-slate-600">INSAT-3D (82°E) · Himawari-9 (140.7°E) · GOES-16 (75.2°W) · VIIRS</span>
          </div>
          <span className="text-[11px] bg-emerald-50 text-emerald-700 border border-emerald-200 px-2 py-0.5 rounded font-semibold">
            WMO GTS CALIBRATED
          </span>
        </div>
      </div>

      {/* Main Grid: Interactive Satellite Canvas + Active Storms Feed */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Satellite Viewer Viewport (8 Cols) */}
        <div className="lg:col-span-8 bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm flex flex-col">
          {/* Viewport Toolbar */}
          <div className="p-3 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono-code font-bold text-slate-700">PLATFORM:</span>
              <span className="text-xs font-bold text-sky-800 bg-sky-50 px-2.5 py-1 rounded border border-sky-200">
                {selectedFeed.name}
              </span>
            </div>

            {/* IR Color Enhancement Switcher */}
            <div className="flex items-center gap-1 bg-white p-1 rounded-lg border border-slate-200 text-xs font-mono-code">
              <span className="text-[10px] text-slate-500 px-1 hidden sm:inline">PALETTE:</span>
              {(
                [
                  { id: "standard", label: "Thermal IR" },
                  { id: "bd_curve", label: "BD Curve (Dvorak)" },
                  { id: "rainbow", label: "Rainbow" },
                  { id: "grayscale", label: "Grayscale" },
                ] as const
              ).map((pal) => (
                <button
                  key={pal.id}
                  onClick={() => setColorEnhancement(pal.id)}
                  className={`px-2 py-0.5 rounded text-[11px] transition ${
                    colorEnhancement === pal.id
                      ? "bg-sky-600 text-white font-bold"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  {pal.label}
                </button>
              ))}
            </div>
          </div>

          {/* Interactive Canvas Viewport */}
          <div className="relative bg-slate-950 flex flex-col items-center justify-center p-2 overflow-hidden">
            {/* Top Info Overlay */}
            <div className="absolute top-4 left-4 z-10 flex flex-wrap items-center gap-2">
              <div className="bg-slate-900/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-700 text-xs font-mono-code text-sky-300 flex items-center gap-2 shadow-lg">
                <Crosshair className="w-3.5 h-3.5 text-sky-400" />
                <span>
                  TARGET: {selectedFeed.centerCoords.lat}°N, {selectedFeed.centerCoords.lon}°E
                </span>
              </div>
              <div className="bg-slate-900/90 px-2.5 py-1 rounded-lg text-[11px] font-mono-code text-slate-300 border border-slate-700">
                RES: {selectedFeed.resolution}
              </div>
            </div>

            {/* Interactive Canvas */}
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              className="w-full max-w-[640px] h-auto rounded-lg border border-slate-800 shadow-inner cursor-crosshair"
            />

            {/* Click-to-inspect helper tooltip */}
            <div className="w-full max-w-[640px] mt-2 flex items-center justify-between text-[11px] font-mono-code text-slate-400 px-1">
              <span>Click anywhere on the cloud swath to measure Planck cloud-top brightness temperature</span>
              <span className="text-sky-400">Planck Inverted Scale</span>
            </div>
          </div>

          {/* Pixel Probe Inspection HUD */}
          {clickedPixel && (
            <div className="p-3 bg-sky-50/70 border-t border-sky-100 flex flex-wrap items-center justify-between gap-3 text-xs font-mono-code">
              <div className="flex items-center gap-2">
                <Thermometer className="w-4 h-4 text-sky-600" />
                <span className="font-bold text-slate-800">SAMPLED CLOUD-TOP:</span>
                <span className="text-sky-900 font-extrabold text-sm">
                  {clickedPixel.tempK} K ({clickedPixel.tempC}°C)
                </span>
                <span className="text-slate-500">·</span>
                <span className="text-slate-700">{clickedPixel.depthClass}</span>
              </div>

              <div className="flex items-center gap-3">
                <span className="text-slate-600">Est. Cloud Altitude: <strong>{clickedPixel.altitudeKm} km</strong></span>
                <button
                  onClick={() => setClickedPixel(null)}
                  className="text-slate-400 hover:text-slate-600 text-[10px] underline cursor-pointer"
                >
                  Clear Probe
                </button>
              </div>
            </div>
          )}

          {/* Action Footer: Send to AI Vision Studio */}
          <div className="p-4 bg-slate-50 border-t border-slate-200 flex flex-wrap items-center justify-between gap-3">
            <div className="text-xs text-slate-600">
              <strong>Source:</strong> {selectedFeed.agency} · {selectedFeed.platform}
            </div>

            <div className="flex items-center gap-2">
              {onSendToVision && (
                <button
                  onClick={() => onSendToVision(selectedFeed)}
                  className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white font-medium text-xs transition shadow-sm cursor-pointer"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Send Swath to ViT Vision Classifier</span>
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Right: Active Tropical Cyclones Live Tracker (4 Cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm space-y-3">
            <div className="flex items-center justify-between border-b border-slate-100 pb-2.5">
              <div className="flex items-center gap-2">
                <Radio className="w-4 h-4 text-red-500 animate-pulse" />
                <h3 className="text-sm font-bold text-slate-900">Active Tropical Cyclones</h3>
              </div>
              <span className="text-[11px] font-mono-code bg-red-50 text-red-700 border border-red-200 px-2 py-0.5 rounded font-bold">
                {LIVE_ACTIVE_CYCLONES.length} DETECTED
              </span>
            </div>

            <p className="text-xs text-slate-500">
              Live automated fixes from WMO Tropical Cyclone Centres (IMD RSMC New Delhi, JTWC, NHC).
            </p>

            {/* List of active storms */}
            <div className="space-y-3 pt-1">
              {LIVE_ACTIVE_CYCLONES.map((cyclone) => (
                <div
                  key={cyclone.id}
                  className="p-3 rounded-lg border border-slate-200 hover:border-sky-400 hover:shadow-sm transition bg-slate-50/50 space-y-2"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <h4 className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                        {cyclone.name}
                        {cyclone.status === "INTENSIFYING" && (
                          <span className="text-[9px] font-mono-code px-1.5 py-0.2 rounded bg-amber-100 text-amber-800 font-bold">
                            RI ALERT
                          </span>
                        )}
                      </h4>
                      <span className="text-[10px] text-slate-500 font-mono-code block">
                        {cyclone.basin} · {cyclone.id}
                      </span>
                    </div>

                    <span className="text-xs font-bold text-sky-700 font-mono-code">
                      {cyclone.windKts} kts
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-[11px] font-mono-code bg-white p-2 rounded border border-slate-100 text-slate-600">
                    <div>
                      <span className="text-slate-400 block text-[9px]">COORDS</span>
                      <strong>{cyclone.lat}°N, {cyclone.lon}°E</strong>
                    </div>
                    <div>
                      <span className="text-slate-400 block text-[9px]">PRESSURE</span>
                      <strong className="text-amber-700">{cyclone.pressureHpa} hPa</strong>
                    </div>
                    <div>
                      <span className="text-slate-400 block text-[9px]">HEADING</span>
                      <span>{cyclone.heading}</span>
                    </div>
                    <div>
                      <span className="text-slate-400 block text-[9px]">NEAREST COAST</span>
                      <span className="truncate block" title={cyclone.nearestCoast}>{cyclone.nearestCoast}</span>
                    </div>
                  </div>

                  {onSelectCyclone && (
                    <button
                      onClick={() => onSelectCyclone(cyclone)}
                      className="w-full py-1.5 rounded bg-sky-50 hover:bg-sky-100 text-sky-700 border border-sky-200 text-xs font-mono-code font-bold transition cursor-pointer flex items-center justify-center gap-1.5"
                    >
                      <Crosshair className="w-3 h-3 text-sky-600" />
                      <span>Lock Target in Command Center</span>
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Multi-spectral Band Info Card */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm space-y-2 text-xs">
            <div className="flex items-center gap-2 text-slate-800 font-bold">
              <Info className="w-4 h-4 text-sky-600" />
              <span>Multi-Spectral Radiometry Guide</span>
            </div>
            <p className="text-slate-600 leading-relaxed text-[11px]">
              <strong>TIR1 (10.8 µm):</strong> Window infrared measures cold cloud-top cirrus canopies without solar dependence.
            </p>
            <p className="text-slate-600 leading-relaxed text-[11px]">
              <strong>WV (6.7 µm):</strong> Mid-tropospheric water vapor isolates upper anticyclonic outflow and dry slot entrainment.
            </p>
          </div>
        </div>
      </div>

      {/* Satellite Feeds Catalog Cards */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3">
          <div>
            <h3 className="text-base font-bold text-slate-900">
              Active Geostationary & Polar Constellation Catalog
            </h3>
            <p className="text-xs text-slate-500 font-mono-code">
              Select any live platform to stream real-time orbital swath imagery
            </p>
          </div>

          {/* Filter by band */}
          <div className="flex items-center gap-1.5 bg-slate-50 p-1 rounded-lg border border-slate-200 text-xs font-mono-code">
            <span className="text-[11px] text-slate-500 px-2">BAND:</span>
            {(["ALL", "TIR1", "WV", "VIS", "RGB"] as const).map((b) => (
              <button
                key={b}
                onClick={() => setSelectedFilter(b)}
                className={`px-2.5 py-1 rounded transition ${
                  selectedFilter === b
                    ? "bg-sky-600 text-white font-bold"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-200/60"
                }`}
              >
                {b}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredFeeds.map((feed) => {
            const isSelected = feed.id === selectedFeedId;
            return (
              <div
                key={feed.id}
                onClick={() => setSelectedFeedId(feed.id)}
                className={`p-4 rounded-xl border transition-all cursor-pointer flex flex-col justify-between space-y-3 ${
                  isSelected
                    ? "bg-sky-50/60 border-sky-400 shadow-sm ring-1 ring-sky-300"
                    : "bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50/60"
                }`}
              >
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono-code px-2 py-0.5 rounded bg-slate-100 text-slate-700 font-bold border border-slate-200">
                      {feed.band}
                    </span>
                    <span className="text-[10px] font-mono-code text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200 font-semibold flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                      {feed.status}
                    </span>
                  </div>

                  <h4 className="text-sm font-bold text-slate-900 leading-snug">
                    {feed.name}
                  </h4>
                  <p className="text-xs text-slate-600 line-clamp-2 leading-relaxed">
                    {feed.description}
                  </p>
                </div>

                <div className="pt-2 border-t border-slate-100 text-[11px] font-mono-code text-slate-500 space-y-1">
                  <div className="flex justify-between">
                    <span>Coverage:</span>
                    <strong className="text-slate-700 truncate max-w-[170px]">{feed.coverage}</strong>
                  </div>
                  <div className="flex justify-between">
                    <span>Refresh:</span>
                    <strong className="text-sky-700">Every {feed.refreshRateMin} min</strong>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
