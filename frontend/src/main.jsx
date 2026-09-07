import React, { useState, useRef, useEffect } from "react";
import { createRoot } from "react-dom/client";
import {
  Upload,
  Satellite,
  Activity,
  TrendingUp,
  MapPin,
  Wind,
  Gauge,
  Eye,
  Layers,
  Database,
  AlertCircle,
  CheckCircle,
  XCircle,
  Loader,
  Radio,
  Target,
  Thermometer,
  CloudRain,
  Navigation,
} from "lucide-react";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Dvorak pattern metadata with severity colors
const PATTERN_META = {
  clear: { label: "Clear", severity: 0, color: "#64748b", icon: "○" },
  developing: { label: "Developing", severity: 1, color: "#06b6d4", icon: "◐" },
  curved_band: { label: "Curved Band", severity: 2, color: "#10b981", icon: "◓" },
  central_dense_overcast: { label: "CDO", severity: 3, color: "#f59e0b", icon: "●" },
  eye: { label: "Eye", severity: 4, color: "#f97316", icon: "◉" },
  sheared: { label: "Sheared", severity: 2, color: "#8b5cf6", icon: "◔" },
  dissipating: { label: "Dissipating", severity: 1, color: "#6366f1", icon: "◑" },
};

// IMD intensity class metadata
const INTENSITY_META = {
  depression: { label: "Depression", category: "D", color: "#06b6d4", wind: "< 34 kt" },
  tropical_storm: { label: "Cyclonic Storm", category: "CS", color: "#10b981", wind: "34-47 kt" },
  severe_cyclonic_storm: { label: "Severe CS", category: "SCS", color: "#f59e0b", wind: "48-63 kt" },
  very_severe_cyclonic_storm: { label: "Very Severe CS", category: "VSCS", color: "#ef4444", wind: "64-89 kt" },
};

// Real historical samples
const SAMPLE_STORMS = [
  { id: "amphan_peak", name: "AMPHAN Peak", file: "AMPHAN_MODIS_IR_2020-05-18.png", channel: "ir", date: "2020-05-18", wind: 140 },
  { id: "fani_eye", name: "FANI Eye", file: "FANI_MODIS_IR_2019-05-02.png", channel: "ir", date: "2019-05-02", wind: 130 },
  { id: "biparjoy", name: "BIPARJOY", file: "BIPARJOY_LATE_MODIS_VIS_2023-06-15.png", channel: "vis", date: "2023-06-15", wind: 105 },
];

const SAMPLE_TRACKS = {
  "AMPHAN (2020)": [
    { lat: 10.8, lon: 86.3, wind_kts: 45, pressure_hpa: 996 },
    { lat: 11.5, lon: 86.1, wind_kts: 75, pressure_hpa: 980 },
    { lat: 13.5, lon: 86.5, wind_kts: 140, pressure_hpa: 925 },
    { lat: 17.2, lon: 87.8, wind_kts: 155, pressure_hpa: 920 },
  ],
  "FANI (2019)": [
    { lat: 9.5, lon: 84.2, wind_kts: 35, pressure_hpa: 998 },
    { lat: 12.3, lon: 84.5, wind_kts: 65, pressure_hpa: 982 },
    { lat: 16.0, lon: 84.8, wind_kts: 130, pressure_hpa: 932 },
  ],
  "BIPARJOY (2023)": [
    { lat: 12.1, lon: 66.7, wind_kts: 55, pressure_hpa: 990 },
    { lat: 14.8, lon: 67.2, wind_kts: 95, pressure_hpa: 964 },
    { lat: 18.3, lon: 68.1, wind_kts: 105, pressure_hpa: 956 },
  ],
};

function App() {
  const [activeModule, setActiveModule] = useState("detection"); // detection | forecast
  const [file, setFile] = useState(null);
  const [channel, setChannel] = useState("ir");
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Forecast state
  const [observations, setObservations] = useState([]);
  const [forecast, setForecast] = useState(null);
  const [forecastLoading, setForecastLoading] = useState(false);

  const canvasRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    const validTypes = ["image/png", "image/jpeg", "image/jpg", "image/webp"];
    if (!validTypes.includes(selectedFile.type)) {
      setError("Invalid format. Upload PNG, JPG, or WebP.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("File exceeds 10 MB limit.");
      return;
    }

    setFile(selectedFile);
    setError(null);
    setResult(null);

    const reader = new FileReader();
    reader.onload = (evt) => setPreview(evt.target.result);
    reader.readAsDataURL(selectedFile);
  };

  const loadSample = async (sample) => {
    try {
      const path = `../data/real_satellite/${sample.file}`;
      const response = await fetch(path);
      const blob = await response.blob();
      const file = new File([blob], sample.file, { type: "image/png" });

      setFile(file);
      setChannel(sample.channel);
      setError(null);
      setResult(null);

      const reader = new FileReader();
      reader.onload = (evt) => setPreview(evt.target.result);
      reader.readAsDataURL(file);
    } catch (err) {
      setError(`Sample load failed: ${err.message}`);
    }
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API}/predict/detect_and_classify?channel=${channel}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Analysis failed");
      }

      const data = await response.json();
      setResult(data);

      // Draw detection overlay
      if (data.detection && preview && canvasRef.current) {
        const img = new Image();
        img.onload = () => {
          const canvas = canvasRef.current;
          const ctx = canvas.getContext("2d");
          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0);

          const [x, y, w, h] = data.detection.bbox;
          const [cx, cy] = data.detection.centroid;

          // Glow effect
          ctx.shadowBlur = 15;
          ctx.shadowColor = "#06b6d4";

          // Bounding box
          ctx.strokeStyle = "#06b6d4";
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, w, h);

          // Crosshair at centroid
          ctx.strokeStyle = "#10b981";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(cx - 10, cy);
          ctx.lineTo(cx + 10, cy);
          ctx.moveTo(cx, cy - 10);
          ctx.lineTo(cx, cy + 10);
          ctx.stroke();

          // Centroid dot
          ctx.fillStyle = "#10b981";
          ctx.shadowColor = "#10b981";
          ctx.beginPath();
          ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
          ctx.fill();
        };
        img.src = preview;
      }
    } catch (err) {
      setError(err.message || "Network error");
    } finally {
      setLoading(false);
    }
  };

  const runForecast = async () => {
    if (observations.length < 2) {
      setError("Minimum 2 observations required");
      return;
    }

    setForecastLoading(true);
    setError(null);
    setForecast(null);

    try {
      const response = await fetch(`${API}/predict/forecast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ observations }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Forecast failed");
      }

      const data = await response.json();
      setForecast(data);
    } catch (err) {
      setError(err.message || "Forecast failed");
    } finally {
      setForecastLoading(false);
    }
  };

  const loadTrack = (trackName) => {
    setObservations(SAMPLE_TRACKS[trackName]);
    setForecast(null);
    setError(null);
  };

  const addObservation = () => {
    setObservations([...observations, { lat: 0, lon: 0, wind_kts: 0, pressure_hpa: 1000 }]);
  };

  const updateObs = (i, field, value) => {
    const newObs = [...observations];
    newObs[i][field] = parseFloat(value) || 0;
    setObservations(newObs);
  };

  const removeObs = (i) => {
    setObservations(observations.filter((_, idx) => idx !== i));
  };

  const isCyclone = result?.is_cyclone === true;
  const detection = result?.detection;
  const classification = result?.classification;
  const patternMeta = classification ? PATTERN_META[classification.pattern] : null;

  return (
    <div className="hud">
      {/* Top Command Bar */}
      <header className="command-bar">
        <div className="command-left">
          <div className="logo-hud">
            <Satellite size={20} />
            <span>CYCLONESENSE</span>
            <span className="version">v1.0</span>
          </div>
          <div className="status-pill">
            <Radio size={14} className="pulse" />
            <span>OPERATIONAL</span>
          </div>
        </div>

        <nav className="command-nav">
          <button
            className={`nav-module ${activeModule === "detection" ? "active" : ""}`}
            onClick={() => setActiveModule("detection")}
          >
            <Target size={16} />
            <span>DETECTION</span>
          </button>
          <button
            className={`nav-module ${activeModule === "forecast" ? "active" : ""}`}
            onClick={() => setActiveModule("forecast")}
          >
            <Navigation size={16} />
            <span>FORECAST</span>
          </button>
        </nav>

        <div className="command-right">
          <div className="timestamp">
            {new Date().toISOString().slice(0, 19).replace("T", " ")} UTC
          </div>
        </div>
      </header>

      {/* Main Content Grid */}
      <main className="mission-grid">
        {activeModule === "detection" && (
          <>
            {/* Left: Satellite Viewer */}
            <section className="viewport">
              <div className="viewport-header">
                <div className="viewport-title">
                  <Layers size={16} />
                  <span>SATELLITE IMAGERY</span>
                </div>
                <div className="channel-badge">
                  {channel === "ir" ? "IR BAND 31" : "TRUE COLOR"}
                </div>
              </div>

              <div className="viewport-content">
                {preview ? (
                  <div className="image-display">
                    {detection ? (
                      <canvas ref={canvasRef} className="detection-canvas" />
                    ) : (
                      <img src={preview} alt="Satellite scene" />
                    )}
                  </div>
                ) : (
                  <div className="viewport-empty">
                    <Upload size={64} strokeWidth={1} />
                    <p>NO IMAGERY LOADED</p>
                  </div>
                )}
              </div>

              {/* Image Controls */}
              <div className="viewport-controls">
                <label className="file-input-btn">
                  <Upload size={16} />
                  <span>{file ? file.name : "UPLOAD IMAGE"}</span>
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/jpg,image/webp"
                    onChange={handleFileChange}
                  />
                </label>

                <select
                  className="sample-select"
                  onChange={(e) => {
                    const sample = SAMPLE_STORMS.find((s) => s.id === e.target.value);
                    if (sample) loadSample(sample);
                  }}
                  disabled={loading}
                >
                  <option value="">LOAD SAMPLE</option>
                  {SAMPLE_STORMS.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.name} ({s.date})
                    </option>
                  ))}
                </select>

                <div className="channel-switch">
                  <button
                    className={channel === "ir" ? "active" : ""}
                    onClick={() => setChannel("ir")}
                    disabled={loading}
                  >
                    IR
                  </button>
                  <button
                    className={channel === "vis" ? "active" : ""}
                    onClick={() => setChannel("vis")}
                    disabled={loading}
                  >
                    VIS
                  </button>
                </div>

                <button className="analyze-btn" onClick={analyze} disabled={!file || loading}>
                  {loading ? (
                    <>
                      <Loader size={16} className="spin" />
                      <span>ANALYZING</span>
                    </>
                  ) : (
                    <>
                      <Activity size={16} />
                      <span>RUN ANALYSIS</span>
                    </>
                  )}
                </button>
              </div>
            </section>

            {/* Right: Telemetry Panel */}
            <aside className="telemetry">
              <div className="telemetry-header">
                <Eye size={16} />
                <span>ANALYSIS TELEMETRY</span>
              </div>

              {error && (
                <div className="alert alert-error">
                  <XCircle size={18} />
                  <div>
                    <strong>ERROR</strong>
                    <p>{error}</p>
                  </div>
                </div>
              )}

              {result && !error && (
                <>
                  {/* Status Alert */}
                  <div className={`alert ${isCyclone ? "alert-success" : "alert-info"}`}>
                    {isCyclone ? <CheckCircle size={18} /> : <AlertCircle size={18} />}
                    <div>
                      <strong>{isCyclone ? "CYCLONE DETECTED" : "NO CYCLONE"}</strong>
                      <p>{result.message}</p>
                    </div>
                  </div>

                  {/* Classification Card */}
                  {classification && patternMeta && (
                    <div className="telemetry-card">
                      <div className="card-header">
                        <span>MORPHOLOGY</span>
                        <span className="confidence-badge" style={{ color: patternMeta.color }}>
                          {(classification.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="pattern-display">
                        <div className="pattern-icon" style={{ color: patternMeta.color }}>
                          {patternMeta.icon}
                        </div>
                        <div className="pattern-name" style={{ color: patternMeta.color }}>
                          {patternMeta.label.toUpperCase()}
                        </div>
                      </div>
                      <div className="confidence-bar">
                        <div
                          className="confidence-fill"
                          style={{
                            width: `${classification.confidence * 100}%`,
                            background: patternMeta.color,
                          }}
                        />
                      </div>
                      <div className="metric-tag">MODEL: {classification.model}</div>
                    </div>
                  )}

                  {/* Detection Metrics */}
                  {detection && (
                    <div className="telemetry-card">
                      <div className="card-header">
                        <MapPin size={14} />
                        <span>DETECTION</span>
                      </div>
                      <div className="metrics-grid">
                        <div className="metric">
                          <span className="metric-label">BBOX</span>
                          <span className="metric-value">[{detection.bbox.join(", ")}]</span>
                        </div>
                        <div className="metric">
                          <span className="metric-label">CENTROID</span>
                          <span className="metric-value">
                            ({detection.centroid[0].toFixed(1)}, {detection.centroid[1].toFixed(1)})
                          </span>
                        </div>
                        <div className="metric">
                          <span className="metric-label">AREA</span>
                          <span className="metric-value">{detection.area} px²</span>
                        </div>
                        <div className="metric">
                          <span className="metric-label">CONFIDENCE</span>
                          <span className="metric-value">{(detection.confidence * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                      <div className="metric-tag">METHOD: {detection.method}</div>
                    </div>
                  )}

                  {/* Explainability Note */}
                  <div className="info-note">
                    <AlertCircle size={14} />
                    <div>
                      <strong>INTERPRETABILITY</strong>
                      <p>Grad-CAM activation maps available in model but not exposed via API.</p>
                    </div>
                  </div>
                </>
              )}

              {!result && !error && !loading && (
                <div className="telemetry-empty">
                  <Activity size={48} strokeWidth={1} />
                  <p>AWAITING ANALYSIS</p>
                </div>
              )}
            </aside>
          </>
        )}

        {activeModule === "forecast" && (
          <>
            {/* Left: Track Input */}
            <section className="viewport">
              <div className="viewport-header">
                <div className="viewport-title">
                  <Navigation size={16} />
                  <span>OBSERVATION SEQUENCE</span>
                </div>
              </div>

              <div className="track-controls">
                <select
                  className="track-select"
                  onChange={(e) => {
                    if (e.target.value) loadTrack(e.target.value);
                  }}
                >
                  <option value="">LOAD HISTORICAL TRACK</option>
                  {Object.keys(SAMPLE_TRACKS).map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>

                <button className="add-obs-btn" onClick={addObservation}>
                  + ADD OBSERVATION
                </button>
              </div>

              <div className="obs-list">
                {observations.map((obs, i) => (
                  <div key={i} className="obs-row">
                    <span className="obs-index">{i + 1}</span>
                    <input
                      type="number"
                      placeholder="LAT"
                      value={obs.lat}
                      onChange={(e) => updateObs(i, "lat", e.target.value)}
                      step="0.1"
                    />
                    <input
                      type="number"
                      placeholder="LON"
                      value={obs.lon}
                      onChange={(e) => updateObs(i, "lon", e.target.value)}
                      step="0.1"
                    />
                    <input
                      type="number"
                      placeholder="WIND (kt)"
                      value={obs.wind_kts}
                      onChange={(e) => updateObs(i, "wind_kts", e.target.value)}
                      step="1"
                    />
                    <input
                      type="number"
                      placeholder="PRES (hPa)"
                      value={obs.pressure_hpa}
                      onChange={(e) => updateObs(i, "pressure_hpa", e.target.value)}
                      step="1"
                    />
                    <button className="remove-obs" onClick={() => removeObs(i)}>
                      <XCircle size={16} />
                    </button>
                  </div>
                ))}
              </div>

              <button
                className="forecast-btn"
                onClick={runForecast}
                disabled={observations.length < 2 || forecastLoading}
              >
                {forecastLoading ? (
                  <>
                    <Loader size={16} className="spin" />
                    <span>COMPUTING</span>
                  </>
                ) : (
                  <>
                    <TrendingUp size={16} />
                    <span>RUN LSTM FORECAST</span>
                  </>
                )}
              </button>
            </section>

            {/* Right: Forecast Result */}
            <aside className="telemetry">
              <div className="telemetry-header">
                <TrendingUp size={16} />
                <span>FORECAST OUTPUT</span>
              </div>

              {error && (
                <div className="alert alert-error">
                  <XCircle size={18} />
                  <div>
                    <strong>ERROR</strong>
                    <p>{error}</p>
                  </div>
                </div>
              )}

              {forecast && (
                <>
                  {/* Next Position */}
                  <div className="forecast-main">
                    <div className="forecast-label">NEXT POSITION (+6H)</div>
                    <div className="forecast-coords">
                      {forecast.next_lat.toFixed(2)}°N {forecast.next_lon.toFixed(2)}°E
                    </div>
                  </div>

                  {/* Intensity Metrics */}
                  <div className="telemetry-card">
                    <div className="card-header">
                      <Wind size={14} />
                      <span>INTENSITY</span>
                    </div>
                    <div className="metrics-grid">
                      <div className="metric">
                        <span className="metric-label">WIND SPEED</span>
                        <span className="metric-value">{forecast.predicted_wind_kts.toFixed(0)} kt</span>
                      </div>
                      <div className="metric">
                        <span className="metric-label">CATEGORY</span>
                        <span className="metric-value">
                          {INTENSITY_META[forecast.intensity_class]?.category || "N/A"}
                        </span>
                      </div>
                      <div className="metric">
                        <span className="metric-label">CONFIDENCE</span>
                        <span className="metric-value">{(forecast.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  </div>

                  {/* Info Note */}
                  <div className="info-note">
                    <AlertCircle size={14} />
                    <div>
                      <strong>SINGLE-STEP FORECAST</strong>
                      <p>6-hour prediction only. Multi-step trajectory rollout not implemented.</p>
                    </div>
                  </div>
                </>
              )}

              {!forecast && !error && !forecastLoading && (
                <div className="telemetry-empty">
                  <Navigation size={48} strokeWidth={1} />
                  <p>AWAITING FORECAST</p>
                </div>
              )}
            </aside>
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="hud-footer">
        <div className="footer-left">
          <Satellite size={14} />
          <span>CYCLONESENSE AI</span>
        </div>
        <div className="footer-center">
          Research prototype • Not an official meteorological warning system
        </div>
        <div className="footer-right">NASA MODIS • NOAA IBTrACS • PyTorch CNN+LSTM</div>
      </footer>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
