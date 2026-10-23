import { useRef, useState, type FormEvent } from "react";

type MultiMetric = {
  observed_pressure_hpa: number;
  pressure_derived_wind_kt: number;
  image_derived_pressure_hpa: number;
  image_minus_pressure_kt: number;
  agreement: "close" | "moderate" | "poor" | string;
  relation: string;
};

type WindResult = {
  status: string;
  model: string;
  unit: string;
  normalized_model_output: number;
  raw_prediction_kt: number;
  application_prediction_kt: number;
  predicted_wind_kt: number;
  category: string;
  multi_metric: MultiMetric | null;
  disclaimer: string;
};

const AGREEMENT_TEXT: Record<string, string> = {
  close: "The two estimates agree within 10 kt.",
  moderate: "The two estimates differ by 10 to 25 kt.",
  poor: "The two estimates differ by more than 25 kt. Treat both with caution.",
};

const MAX_BYTES = 10 * 1024 * 1024;

function endpoint() {
  const configured = import.meta.env.VITE_API_URL?.trim().replace(/\/$/, "");
  return configured ? `${configured}/predict/wind` : "/api/predict/wind";
}

export function WindEstimator() {
  const [file, setFile] = useState<File | null>(null);
  const [pressure, setPressure] = useState("");
  const [result, setResult] = useState<WindResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function chooseFile(next: File | undefined) {
    setResult(null);
    setError("");
    if (!next) { setFile(null); return; }
    if (!next.type.startsWith("image/")) {
      setFile(null);
      setError("Choose an image file such as a PNG or JPEG.");
      return;
    }
    if (next.size > MAX_BYTES) {
      setFile(null);
      setError("This image is larger than 10 MB. Choose a smaller satellite image.");
      return;
    }
    setFile(next);
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || loading) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const body = new FormData();
      body.append("file", file);
      // Sent only when given. The image model ignores it; the API returns it as a
      // separate pressure-derived estimate to compare against.
      if (pressure.trim() !== "") body.append("pressure_hpa", pressure.trim());
      const response = await fetch(endpoint(), { method: "POST", body, signal: AbortSignal.timeout(120_000) });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = typeof payload.detail === "string" ? payload.detail : "The wind service could not process this image.";
        throw new Error(detail);
      }
      setResult(payload as WindResult);
    } catch (cause) {
      setError(cause instanceof DOMException && cause.name === "TimeoutError"
        ? "Wind estimation took too long. Check that the API is running, then try again."
        : cause instanceof TypeError
          ? "Could not reach the wind service. Check your connection and API configuration, then retry."
          : cause instanceof Error ? cause.message : "Wind estimation failed. Choose another image and retry.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="wind-estimator">
      <div className="wind-intro">
        <div>
          <h3>Estimate wind from an image</h3>
          <p>Upload one storm-centred infrared image. Model A returns a research estimate in knots.</p>
        </div>
        <span className="wind-model">Model A · ResNet-18</span>
      </div>
      <form className="wind-form" onSubmit={submit}>
        <label className="wind-file" htmlFor="wind-image">
          <span>{file ? file.name : "Choose a satellite image"}</span>
          <small>PNG or JPEG · up to 10 MB</small>
        </label>
        <input
          ref={inputRef}
          id="wind-image"
          type="file"
          accept="image/*"
          onChange={(event) => {
            chooseFile(event.currentTarget.files?.[0]);
            event.currentTarget.value = "";
          }}
          aria-describedby="wind-help wind-status"
        />
        <label className="wind-pressure" htmlFor="wind-pressure">
          <span>Central pressure <small>optional</small></span>
          <input
            id="wind-pressure"
            type="number"
            inputMode="decimal"
            min="800"
            max="1050"
            step="0.1"
            placeholder="e.g. 950"
            value={pressure}
            onChange={(event) => setPressure(event.currentTarget.value)}
            aria-describedby="wind-pressure-help"
          />
          <small id="wind-pressure-help">hPa, 800 to 1050</small>
        </label>
        <button className="wind-submit" type="submit" disabled={!file || loading}>
          {loading && <span className="wind-spinner" aria-hidden="true" />}
          {loading ? "Estimating…" : "Estimate wind"}
        </button>
      </form>
      <p id="wind-help" className="wind-help">Best suited to MODIS Band 31 style infrared chips. Preprocessing matches evaluation: RGB, resize to 224, then ImageNet normalization.</p>
      <div id="wind-status" aria-live="polite" aria-atomic="true">
        {error && <div className="wind-error" role="alert">{error}</div>}
        {result && (
          <div className="wind-result">
            <div className="wind-display">
              <span>Application estimate</span>
              <strong>{result.application_prediction_kt.toFixed(1)} <small>kt</small></strong>
              <b>{result.category}</b>
            </div>
            <dl>
              <div><dt>Raw model output</dt><dd>{result.raw_prediction_kt.toFixed(2)} kt</dd></div>
              <div><dt>Model scale</dt><dd>{result.normalized_model_output.toFixed(4)} × 100 kt</dd></div>
            </dl>
            <p>{result.disclaimer}</p>
            {result.multi_metric && (
              <div className="wind-multi">
                <h4>Compared with the pressure you entered</h4>
                <dl>
                  <div><dt>From {Math.round(result.multi_metric.observed_pressure_hpa)} hPa</dt><dd>{result.multi_metric.pressure_derived_wind_kt.toFixed(1)} kt</dd></div>
                  <div><dt>From the image</dt><dd>{result.raw_prediction_kt.toFixed(1)} kt</dd></div>
                  <div><dt>Image implies</dt><dd>{Math.round(result.multi_metric.image_derived_pressure_hpa)} hPa</dd></div>
                  <div><dt>Difference</dt><dd>{result.multi_metric.image_minus_pressure_kt > 0 ? "+" : ""}{result.multi_metric.image_minus_pressure_kt.toFixed(1)} kt</dd></div>
                </dl>
                <p className={`wind-agree ${result.multi_metric.agreement}`}>
                  {AGREEMENT_TEXT[result.multi_metric.agreement] ?? result.multi_metric.agreement}
                </p>
                <p className="wind-note">Pressure estimate via {result.multi_metric.relation}. The image model never sees this number.</p>
              </div>
            )}
            {result.raw_prediction_kt < 0 && <p className="wind-note">The raw estimate is negative. The application value is shown as 0 kt; research output remains raw.</p>}
          </div>
        )}
      </div>
    </div>
  );
}
