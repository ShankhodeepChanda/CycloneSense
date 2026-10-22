import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "models" / "temporal_test_predictions_weighted.csv"
OUT_JSON = ROOT / "models" / "weighted_error_analysis.json"
OUT_STORMS = ROOT / "models" / "weighted_storm_analysis.csv"

if not PREDICTIONS.exists():
    raise FileNotFoundError(PREDICTIONS)

df = pd.read_csv(PREDICTIONS)

required = {
    "sid", "name", "actual_wind_kt", "predicted_wind_kt",
    "actual_category", "predicted_category", "year"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df["error_kt"] = df["predicted_wind_kt"] - df["actual_wind_kt"]
df["absolute_error_kt"] = df["error_kt"].abs()
df["squared_error"] = df["error_kt"] ** 2

categories = ["TD", "TS", "C1", "C2", "C3", "C4", "C5"]
category_results = {}

for cat in categories:
    part = df[df["actual_category"] == cat]
    if part.empty:
        continue

    category_results[cat] = {
        "images": int(len(part)),
        "storms": int(part["sid"].nunique()),
        "actual_mean_kt": round(float(part["actual_wind_kt"].mean()), 3),
        "predicted_mean_kt": round(float(part["predicted_wind_kt"].mean()), 3),
        "mean_bias_kt": round(float(part["error_kt"].mean()), 3),
        "mae_kt": round(float(part["absolute_error_kt"].mean()), 3),
        "rmse_kt": round(float(np.sqrt(part["squared_error"].mean())), 3),
    }

storm_rows = []
for (sid, name, year), part in df.groupby(["sid", "name", "year"]):
    actual = part["actual_wind_kt"].to_numpy(dtype=float)
    pred = part["predicted_wind_kt"].to_numpy(dtype=float)
    storm_rows.append({
        "sid": sid,
        "name": name,
        "year": int(year),
        "images": int(len(part)),
        "actual_mean_kt": round(float(actual.mean()), 3),
        "predicted_mean_kt": round(float(pred.mean()), 3),
        "mean_bias_kt": round(float((pred - actual).mean()), 3),
        "mae_kt": round(float(np.abs(pred - actual).mean()), 3),
        "rmse_kt": round(float(np.sqrt(np.mean((pred - actual) ** 2))), 3),
    })

storms = pd.DataFrame(storm_rows).sort_values("mae_kt")
storms.to_csv(OUT_STORMS, index=False)

report = {
    "checkpoint": "models/multiyear45_resnet18_ir_weighted.pt",
    "source_predictions": str(PREDICTIONS.relative_to(ROOT)),
    "images": int(len(df)),
    "storms": int(df["sid"].nunique()),
    "overall": {
        "mae_kt": round(float(df["absolute_error_kt"].mean()), 3),
        "rmse_kt": round(float(np.sqrt(df["squared_error"].mean())), 3),
        "mean_bias_kt": round(float(df["error_kt"].mean()), 3),
    },
    "by_actual_category": category_results,
    "storm_summary": {
        "best_mae_storms": storms.head(5)[
            ["sid", "name", "year", "images", "mae_kt", "mean_bias_kt"]
        ].to_dict(orient="records"),
        "worst_mae_storms": storms.tail(5).sort_values(
            "mae_kt", ascending=False
        )[
            ["sid", "name", "year", "images", "mae_kt", "mean_bias_kt"]
        ].to_dict(orient="records"),
    },
    "caution": (
        "Images within a storm are correlated. Storm-level results are important, "
        "and this 15-storm sample is too small to establish broad operational reliability."
    ),
}

OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nWEIGHTED MODEL ERROR ANALYSIS")
print(json.dumps(report, indent=2))
print(f"\nSaved category report: {OUT_JSON}")
print(f"Saved per-storm report: {OUT_STORMS}")
