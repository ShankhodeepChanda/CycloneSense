import { useRef, useState, type FormEvent } from "react";

type WindResult = {
  status: string;
  model: string;
  unit: string;
  normalized_model_output: number;
  raw_prediction_kt: number;
  application_prediction_kt: number;
  predicted_wind_kt: number;
  category: string;
  disclaimer: string;
};

const MAX_BYTES = 10 * 1024 * 1024;

function endpoint() {
  const configured = import.meta.env.VITE_API_URL?.trim().replace(/\/$/, "");
  return configured ? `${configured}/predict/wind` : "/api/predict/wind";
}

export function WindEstimator() {
  const [file, setFile] = useState<File | null>(null);
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
            {result.raw_prediction_kt < 0 && <p className="wind-note">The raw estimate is negative. The application value is shown as 0 kt; research output remains raw.</p>}
          </div>
        )}
      </div>
    </div>
  );
}
