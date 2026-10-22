from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "models" / "temporal_test_predictions.csv"
WEIGHTED = ROOT / "models" / "temporal_test_predictions_weighted.csv"
OUT = ROOT / "models" / "model_storm_comparison.csv"

for path in (ORIGINAL, WEIGHTED):
    if not path.exists():
        raise FileNotFoundError(path)

def load_predictions(path, label):
    df = pd.read_csv(path)
    required = {"sid", "name", "year", "actual_wind_kt", "predicted_wind_kt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

    df["sid"] = df["sid"].astype(str)
    df["error"] = df["predicted_wind_kt"] - df["actual_wind_kt"]
    df["abs_error"] = df["error"].abs()

    return df.groupby(["sid", "name", "year"]).agg(
        **{
            f"{label}_images": ("actual_wind_kt", "size"),
            f"{label}_mae_kt": ("abs_error", "mean"),
            f"{label}_bias_kt": ("error", "mean"),
        }
    ).reset_index()

original = load_predictions(ORIGINAL, "original")
weighted = load_predictions(WEIGHTED, "weighted")

comparison = original.merge(
    weighted,
    on=["sid", "name", "year"],
    how="outer",
    validate="one_to_one",
    indicator=True,
)

if not (comparison["_merge"] == "both").all():
    raise RuntimeError("The two prediction files do not contain the same storms.")

if not (comparison["original_images"] == comparison["weighted_images"]).all():
    raise RuntimeError("Image counts differ between the two prediction files.")

comparison["mae_improvement_kt"] = (
    comparison["original_mae_kt"] - comparison["weighted_mae_kt"]
)
comparison["improved"] = comparison["mae_improvement_kt"] > 0
comparison = comparison.drop(columns="_merge")
comparison = comparison.sort_values("mae_improvement_kt", ascending=False)

comparison.to_csv(OUT, index=False)

print("\nSTORM-BY-STORM MODEL COMPARISON")
print(comparison.round(3).to_string(index=False))
print("\nSUMMARY")
print(f"Storms compared: {len(comparison)}")
print(f"Weighted model improved: {int(comparison['improved'].sum())}")
print(f"Weighted model worsened: {int((~comparison['improved']).sum())}")
print(
    "Mean storm-level MAE improvement (kt): "
    f"{comparison['mae_improvement_kt'].mean():.3f}"
)
print(f"\nSaved to: {OUT}")
