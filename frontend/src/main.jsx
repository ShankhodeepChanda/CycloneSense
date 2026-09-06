import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, CloudLightning, Satellite, Upload, Wind, AlertCircle, CheckCircle, XCircle } from "lucide-react";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [file, setFile] = useState(null);
  const [channel, setChannel] = useState("ir");
  const [result, setResult] = useState(null);
  const [fc, setFc] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    // Validate file type
    const validTypes = ["image/png", "image/jpeg", "image/jpg", "image/webp"];
    if (!validTypes.includes(selectedFile.type)) {
      setError("Invalid file type. Please upload PNG, JPG, or WebP.");
      setFile(null);
      setPreview(null);
      return;
    }

    // Validate file size (10 MB)
    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("File too large. Maximum size is 10 MB.");
      setFile(null);
      setPreview(null);
      return;
    }

    setFile(selectedFile);
    setError(null);
    setResult(null);

    // Generate preview
    const reader = new FileReader();
    reader.onload = (evt) => setPreview(evt.target.result);
    reader.readAsDataURL(selectedFile);
  };

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("channel", channel);

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
    } catch (err) {
      setError(err.message || "Network error. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  }

  async function forecastDemo() {
    try {
      const response = await fetch(`${API}/predict/forecast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          observations: [
            { lat: 14.2, lon: 72.1, wind_kts: 45, pressure_hpa: 995 },
            { lat: 14.7, lon: 72.8, wind_kts: 52, pressure_hpa: 989 },
            { lat: 15.1, lon: 73.5, wind_kts: 58, pressure_hpa: 984 },
          ],
        }),
      });
      setFc(await response.json());
    } catch (err) {
      console.error("Forecast error:", err);
    }
  }

  const isCyclone = result?.is_cyclone === true;
  const detection = result?.detection;
  const classification = result?.classification;

  return (
    <main>
      <nav>
        <div className="brand">
          <div className="logo-ring"></div>
          <span>CycloneSense</span>
        </div>
        <div className="status">
          <span className="status-dot"></span>
          <small>System Operational</small>
        </div>
      </nav>

      <section className="hero">
        <div className="hero-content">
          <small className="badge">MULTI-SPECTRAL EARTH OBSERVATION</small>
          <h1>
            AI-Powered Tropical Cyclone
            <br />
            <span className="gradient-text">Detection & Analysis</span>
          </h1>
          <p className="lead">
            Real-time identification, morphological classification, and track forecasting from NASA MODIS satellite imagery.
          </p>
        </div>
        <div className="hero-graphic">
          <div className="orbit orbit-1"></div>
          <div className="orbit orbit-2"></div>
          <div className="orbit orbit-3"></div>
          <div className="core-dot"></div>
        </div>
      </section>

      <section className="metrics">
        <div className="metric-card">
          <Satellite size={24} />
          <span className="metric-value">MODIS</span>
          <span className="metric-label">NASA Terra/Aqua</span>
        </div>
        <div className="metric-card">
          <CloudLightning size={24} />
          <span className="metric-value">7 Classes</span>
          <span className="metric-label">Dvorak Patterns</span>
        </div>
        <div className="metric-card">
          <Activity size={24} />
          <span className="metric-value">PyTorch</span>
          <span className="metric-label">CNN + LSTM</span>
        </div>
        <div className="metric-card">
          <Wind size={24} />
          <span className="metric-value">6 Hour</span>
          <span className="metric-label">Track Forecast</span>
        </div>
      </section>

      <section className="analysis-section">
        <div className="section-header">
          <div>
            <small className="section-number">01</small>
            <h2>Detection & Classification Pipeline</h2>
          </div>
        </div>

        <div className="pipeline-container">
          {/* Upload Panel */}
          <div className="upload-panel">
            <label className="file-drop">
              <Upload size={32} />
              <strong>{file ? file.name : "Upload Satellite Image"}</strong>
              <span>PNG, JPG, or WebP • Max 10 MB</span>
              <input type="file" accept="image/png,image/jpeg,image/jpg,image/webp" onChange={handleFileChange} />
            </label>

            {/* Channel Selector */}
            <div className="channel-selector">
              <label className="selector-label">Satellite Channel</label>
              <div className="toggle-group">
                <button
                  className={`toggle-btn ${channel === "ir" ? "active" : ""}`}
                  onClick={() => setChannel("ir")}
                  disabled={loading}
                >
                  <span className="toggle-icon">🌡️</span>
                  <div>
                    <strong>IR Band 31</strong>
                    <small>Thermal Infrared (11 µm)</small>
                  </div>
                </button>
                <button
                  className={`toggle-btn ${channel === "vis" ? "active" : ""}`}
                  onClick={() => setChannel("vis")}
                  disabled={loading}
                >
                  <span className="toggle-icon">🌍</span>
                  <div>
                    <strong>True Color</strong>
                    <small>Visible Composite (RGB)</small>
                  </div>
                </button>
              </div>
            </div>

            {preview && (
              <div className="image-preview">
                <img src={preview} alt="Uploaded satellite scene" />
              </div>
            )}

            <button className="analyze-btn" onClick={analyze} disabled={!file || loading}>
              {loading ? (
                <>
                  <div className="spinner"></div>
                  Processing...
                </>
              ) : (
                "Run Detection & Classification"
              )}
            </button>
          </div>

          {/* Results Panel */}
          <div className="results-panel">
            {error && (
              <div className="alert alert-error">
                <XCircle size={20} />
                <div>
                  <strong>Error</strong>
                  <p>{error}</p>
                </div>
              </div>
            )}

            {result && !error && (
              <>
                {isCyclone ? (
                  <div className="alert alert-success">
                    <CheckCircle size={20} />
                    <div>
                      <strong>Tropical Cyclone Detected</strong>
                      <p>{result.message}</p>
                    </div>
                  </div>
                ) : (
                  <div className="alert alert-info">
                    <AlertCircle size={20} />
                    <div>
                      <strong>No Cyclone Detected</strong>
                      <p>{result.message}</p>
                    </div>
                  </div>
                )}

                {/* Detection Results */}
                {detection && (
                  <div className="result-card">
                    <h3>Detection</h3>
                    <div className="result-grid">
                      <div className="result-item">
                        <span className="label">Bounding Box</span>
                        <span className="value">[{detection.bbox.join(", ")}]</span>
                      </div>
                      <div className="result-item">
                        <span className="label">Centroid</span>
                        <span className="value">
                          ({detection.centroid[0]}, {detection.centroid[1]})
                        </span>
                      </div>
                      <div className="result-item">
                        <span className="label">Area</span>
                        <span className="value">{detection.area} px²</span>
                      </div>
                      <div className="result-item">
                        <span className="label">Confidence</span>
                        <span className="value">{(detection.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Classification Results */}
                {classification && (
                  <div className="result-card">
                    <h3>Classification</h3>
                    <div className="pattern-display">
                      <span className="pattern-label">{classification.pattern.replace(/_/g, " ").toUpperCase()}</span>
                      <div className="confidence-bar">
                        <div className="confidence-fill" style={{ width: `${classification.confidence * 100}%` }}></div>
                      </div>
                      <span className="confidence-text">{(classification.confidence * 100).toFixed(1)}% confidence</span>
                    </div>
                    <small className="model-info">Model: {classification.model}</small>
                  </div>
                )}
              </>
            )}

            {!result && !error && !loading && (
              <div className="empty-state">
                <CloudLightning size={48} strokeWidth={1.5} />
                <p>Upload a satellite image and run the pipeline to see detection and classification results.</p>
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="forecast-section">
        <div className="section-header">
          <div>
            <small className="section-number">02</small>
            <h2>Track & Intensity Forecast</h2>
          </div>
        </div>

        <div className="forecast-container">
          <div className="forecast-map">
            <div className="map-grid"></div>
            <svg className="track-overlay" viewBox="0 0 400 300">
              <path d="M 80 200 Q 150 180, 220 150 T 340 100" stroke="#00d9ff" strokeWidth="3" fill="none" strokeDasharray="5,5" />
              <circle cx="80" cy="200" r="6" fill="#00d9ff" />
              <circle cx="220" cy="150" r="6" fill="#00d9ff" />
              <circle cx="340" cy="100" r="8" fill="#00ff9d" stroke="#002b1f" strokeWidth="2" />
            </svg>
          </div>

          <div className="forecast-results">
            {fc ? (
              <>
                <div className="forecast-card">
                  <h3>Next Position (+6 Hours)</h3>
                  <div className="forecast-value">
                    {fc.next_lat.toFixed(2)}°N, {fc.next_lon.toFixed(2)}°E
                  </div>
                </div>
                <div className="forecast-grid">
                  <div className="forecast-item">
                    <span className="label">Wind Speed</span>
                    <span className="value">{fc.predicted_wind_kts.toFixed(0)} kt</span>
                  </div>
                  <div className="forecast-item">
                    <span className="label">Intensity</span>
                    <span className="value">{fc.intensity_class.replace(/_/g, " ")}</span>
                  </div>
                  <div className="forecast-item">
                    <span className="label">Confidence</span>
                    <span className="value">{(fc.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </>
            ) : (
              <div className="empty-state">
                <Wind size={48} strokeWidth={1.5} />
                <p>Run the demo forecast to test the LSTM-based track prediction model.</p>
              </div>
            )}
            <button className="forecast-btn" onClick={forecastDemo}>
              Run Demo Forecast
            </button>
          </div>
        </div>
      </section>

      <footer>
        <div className="footer-content">
          <div className="brand">
            <div className="logo-ring"></div>
            <span>CycloneSense</span>
          </div>
          <p>Research prototype developed for Smart India Hackathon 2024</p>
          <small className="disclaimer">Not an official meteorological warning system. For educational and research purposes only.</small>
        </div>
      </footer>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
