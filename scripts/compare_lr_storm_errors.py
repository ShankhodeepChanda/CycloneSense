"""Compare the saved weighted CycloneSense checkpoints on one storm-held-out split.

This is a diagnostic of the existing 2018-2020 test partition. It performs
inference only, aligns predictions by image_path, and never clips raw winds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.build_dataset import category
from ml.model import CycloneImageClassifier

DATA = ROOT / "kaggle_multiyear45"
OUT = ROOT / "models" / "storm_diagnostics_lr_comparison"
INDEX = DATA / "images_index.csv"
CATEGORIES = ["TD", "TS", "C1", "C2", "C3", "C4", "C5"]
MODELS = {
    "A": {
        "label": "Original weighted (lr=3e-4)",
        "checkpoint": "models/multiyear45_resnet18_ir_weighted.pt",
        "metrics": "models/multiyear45_metrics_ir_weighted.json",
    },
    "B": {
        "label": "Weighted, no augmentation",
        "checkpoint": "models/multiyear45_resnet18_ir_weighted_noaug.pt",
        "metrics": "models/multiyear45_metrics_ir_weighted_noaug.json",
    },
    "C": {
        "label": "Weighted, lr=1e-4",
        "checkpoint": "models/multiyear45_resnet18_ir_weighted_lr1e4.pt",
        "metrics": "models/multiyear45_metrics_ir_weighted_lr1e4.json",
    },
}
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class ChipDataset(Dataset):
    """Deterministic inference transform matching the training pipeline's eval path."""

    def __init__(self, root: Path, rows: pd.DataFrame, size: int = 224):
        self.root = root
        self.rows = rows.reset_index(drop=True)
        self.tf = T.Compose([T.Resize(size), T.ToTensor(), NORM])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        with Image.open(self.root / row["image_path"]) as image:
            return self.tf(image.convert("RGB"))


def summarize(actual, predicted, storm_ids) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    absolute = np.abs(error)
    return {
        "images": int(len(actual)),
        "storms": int(pd.Series(storm_ids).nunique()),
        "actual_mean_kt": float(actual.mean()),
        "predicted_mean_kt": float(predicted.mean()),
        "mae_kt": float(absolute.mean()),
        "rmse_kt": float(np.sqrt(np.mean(error**2))),
        "mean_bias_kt": float(error.mean()),
        "median_absolute_error_kt": float(np.median(absolute)),
        "maximum_absolute_error_kt": float(absolute.max()),
        "negative_prediction_count": int((predicted < 0).sum()),
    }


def storm_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per SID; positive A-minus-C improvement means C has lower MAE."""
    rows = []
    for sid, part in predictions.groupby("sid", sort=True):
        row = {"sid": sid, "name": part["name"].iloc[0], "images": len(part)}
        row["actual_mean_kt"] = float(part["actual_wind_kt"].mean())
        for model_id in MODELS:
            stats = summarize(
                part["actual_wind_kt"], part[f"predicted_{model_id}_kt"], part["sid"]
            )
            for key in ("predicted_mean_kt", "mae_kt", "rmse_kt", "mean_bias_kt", "maximum_absolute_error_kt"):
                row[f"{model_id.lower()}_{key}"] = stats[key]
        row["mae_improvement_c_vs_a_kt"] = row["a_mae_kt"] - row["c_mae_kt"]
        row["mae_change_c_minus_a_kt"] = -row["mae_improvement_c_vs_a_kt"]
        row["model_c_improved"] = row["mae_improvement_c_vs_a_kt"] > 0
        row["model_c_worsened"] = row["mae_improvement_c_vs_a_kt"] < 0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sid").reset_index(drop=True)


def category_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat in CATEGORIES:
        subset = predictions[predictions["actual_category"] == cat]
        if subset.empty:
            continue
        for model_id, spec in MODELS.items():
            stats = summarize(
                subset["actual_wind_kt"], subset[f"predicted_{model_id}_kt"], subset["sid"]
            )
            rows.append({
                "category": cat,
                "model": model_id,
                "model_name": spec["label"],
                **stats,
                "low_sample_flag": stats["images"] < 10 or stats["storms"] < 5,
                "sample_warning": "Fewer than 10 images or fewer than 5 storms" if stats["images"] < 10 or stats["storms"] < 5 else "",
            })
    return pd.DataFrame(rows)


def storm_category_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sid, cat), part in predictions.groupby(["sid", "actual_category"], sort=True):
        if cat not in {"C3", "C4", "C5"}:
            continue
        row = {"sid": sid, "name": part["name"].iloc[0], "category": cat, "images": len(part)}
        for model_id in ("A", "C"):
            stats = summarize(part["actual_wind_kt"], part[f"predicted_{model_id}_kt"], part["sid"])
            row[f"{model_id.lower()}_mae_kt"] = stats["mae_kt"]
        row["mae_improvement_c_vs_a_kt"] = row["a_mae_kt"] - row["c_mae_kt"]
        row["model_c_improved"] = row["mae_improvement_c_vs_a_kt"] > 0
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(storms: pd.DataFrame, repetitions: int = 10_000, seed: int = 42) -> dict:
    differences = storms["mae_change_c_minus_a_kt"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(repetitions, len(differences)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "estimand": "mean storm MAE(Model C) minus mean storm MAE(Model A), kt; negative favors C",
        "storms": int(len(differences)),
        "repetitions": repetitions,
        "seed": seed,
        "mean_difference_kt": float(differences.mean()),
        "percentile_95_ci_kt": [float(low), float(high)],
        "limitation": "Exploratory interval from only 15 storms; not a claim of statistical significance or generalization.",
    }


def _predict_one(model_id: str, rows: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    spec = MODELS[model_id]
    checkpoint_path = ROOT / spec["checkpoint"]
    metrics_path = ROOT / spec["metrics"]
    if not checkpoint_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(checkpoint_path if not checkpoint_path.is_file() else metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    args = metrics["args"]
    if args.get("channel") != "ir" or args.get("data_dir") != "kaggle_multiyear45":
        raise ValueError(f"{model_id} metrics do not describe the requested IR dataset: {args}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("channel") != "ir" or checkpoint.get("target") != "wind_kt/100":
        raise ValueError(f"Unexpected checkpoint metadata in {checkpoint_path}")
    model = CycloneImageClassifier(num_classes=1, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(ChipDataset(DATA, rows), batch_size=32, shuffle=False, num_workers=0)
    values = []
    with torch.inference_mode():
        for batch in loader:
            values.extend((model(batch.to(device)).squeeze(1) * 100.0).cpu().numpy().tolist())
    return pd.DataFrame({
        "image_path": rows["image_path"].to_numpy(),
        "actual_wind_kt": rows["wind_kt"].astype(float).to_numpy(),
        "sid": rows["sid"].astype(str).to_numpy(),
        f"predicted_{model_id}_kt": values,
    })


def generate_aligned_predictions(index: pd.DataFrame, device=None) -> pd.DataFrame:
    rows = index.sort_values("image_path").reset_index(drop=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    merged = rows[["image_path", "sid", "name", "time", "wind_kt", "category", "satellite"]].copy()
    merged = merged.rename(columns={"wind_kt": "actual_wind_kt", "category": "actual_category"})
    merged["sid"] = merged["sid"].astype(str)
    merged["actual_wind_kt"] = pd.to_numeric(merged["actual_wind_kt"], errors="raise").astype(float)
    for model_id in MODELS:
        predictions = _predict_one(model_id, rows, device)
        merged = merged.merge(predictions, on=["image_path", "sid", "actual_wind_kt"], how="outer", validate="one_to_one", indicator=f"_merge_{model_id}")
        if not (merged[f"_merge_{model_id}"] == "both").all():
            raise RuntimeError(f"{model_id} prediction paths or targets do not align with the test index")
        merged = merged.drop(columns=f"_merge_{model_id}")
    merged["actual_category"] = merged["actual_wind_kt"].map(category)
    return merged


def build_error_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = ["case_type", "model", "sid", "name", "time", "image_path", "actual_category", "actual_wind_kt", "predicted_A_kt", "predicted_B_kt", "predicted_C_kt", "error_A_kt", "error_B_kt", "error_C_kt", "abs_error_A_kt", "abs_error_B_kt", "abs_error_C_kt", "c_advantage_abs_error_kt"]
    frame = predictions.copy()
    for model_id in MODELS:
        frame[f"error_{model_id}_kt"] = frame[f"predicted_{model_id}_kt"] - frame["actual_wind_kt"]
        frame[f"abs_error_{model_id}_kt"] = frame[f"error_{model_id}_kt"].abs()
    frame["c_advantage_abs_error_kt"] = frame["abs_error_A_kt"] - frame["abs_error_C_kt"]
    cases = []
    for model_id in MODELS:
        top = frame.nlargest(10, f"abs_error_{model_id}_kt").copy()
        top["case_type"], top["model"] = "largest_absolute_errors", model_id
        cases.append(top)
        negatives = frame[frame[f"predicted_{model_id}_kt"] < 0].copy()
        if not negatives.empty:
            negatives["case_type"], negatives["model"] = "negative_prediction", model_id
            cases.append(negatives)
        extreme = frame[frame[f"abs_error_{model_id}_kt"] >= 50].copy()
        if not extreme.empty:
            extreme["case_type"], extreme["model"] = "absolute_error_at_least_50_kt", model_id
            cases.append(extreme)
    for case_type, part in (
        ("model_c_largest_paired_improvements", frame.nlargest(10, "c_advantage_abs_error_kt")),
        ("model_c_largest_paired_regressions", frame.nsmallest(10, "c_advantage_abs_error_kt")),
    ):
        part = part.copy()
        part["case_type"], part["model"] = case_type, "A_vs_C"
        cases.append(part)
    spike_rows = []
    timed = frame.assign(_time=pd.to_datetime(frame["time"], errors="coerce")).sort_values(["sid", "_time", "image_path"])
    for sid, part in timed.groupby("sid"):
        part = part.copy()
        gap_hours = part["_time"].diff().dt.total_seconds() / 3600
        for model_id in MODELS:
            step = part[f"predicted_{model_id}_kt"].diff()
            mask = (step.abs() >= 30) & (gap_hours <= 48)
            selected = part.loc[mask].drop(columns="_time").copy()
            if not selected.empty:
                selected["prediction_step_change_kt"] = step.loc[mask].to_numpy()
                selected["hours_since_previous_image"] = gap_hours.loc[mask].to_numpy()
                selected["case_type"], selected["model"] = "prediction_step_change_at_least_30_kt_within_48h", model_id
                spike_rows.append(selected)
    if spike_rows:
        cases.extend(spike_rows)
    result = pd.concat(cases, ignore_index=True).drop_duplicates(subset=["case_type", "model", "image_path"])
    for col in columns:
        if col not in result:
            result[col] = np.nan
    return result[columns + [c for c in ("prediction_step_change_kt", "hours_since_previous_image") if c in result]].sort_values(["case_type", "model", "sid", "time"], na_position="last")


def make_plots(predictions, storms, categories, storm_categories):
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "font.size": 9})
    ordered = storms.sort_values("mae_improvement_c_vs_a_kt")
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(ordered))
    ax.barh(y - 0.18, ordered["a_mae_kt"], height=0.34, label="Model A · original weighted")
    ax.barh(y + 0.18, ordered["c_mae_kt"], height=0.34, label="Model C · lr 1e-4")
    ax.set_yticks(y, [f"{n} ({sid})" for n, sid in zip(ordered["name"], ordered["sid"])])
    ax.set_xlabel("Storm-level MAE (kt)")
    ax.set_title("Per-storm MAE on the shared 2018–2020 test split")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "per_storm_mae_A_vs_C.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
    low = min(predictions["actual_wind_kt"].min(), predictions[["predicted_A_kt", "predicted_C_kt"]].min().min())
    high = max(predictions["actual_wind_kt"].max(), predictions[["predicted_A_kt", "predicted_C_kt"]].max().max())
    for ax, model_id, title in zip(axes, ("A", "C"), ("Original weighted", "Lower learning rate")):
        ax.scatter(predictions["actual_wind_kt"], predictions[f"predicted_{model_id}_kt"], s=14, alpha=0.55)
        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Actual wind speed (kt)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Predicted wind speed (kt)")
    fig.suptitle("Actual versus predicted wind speed · shared test images")
    fig.tight_layout()
    fig.savefig(OUT / "actual_vs_predicted_A_vs_C.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for model_id, label in (("A", "Model A"), ("B", "Model B"), ("C", "Model C")):
        part = categories[categories["model"] == model_id].set_index("category").reindex(CATEGORIES)
        ax.plot(CATEGORIES, part["mae_kt"], marker="o", label=label)
    ax.set_xlabel("Actual intensity category")
    ax.set_ylabel("MAE (kt)")
    ax.set_title("Image-level error by actual intensity category")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "category_mae_A_B_C.png")
    plt.close(fig)

    for rank, (slug, row) in enumerate((("most_improved", storms.nlargest(1, "mae_improvement_c_vs_a_kt").iloc[0]), ("most_regressed", storms.nsmallest(1, "mae_improvement_c_vs_a_kt").iloc[0]))):
        part = predictions[predictions["sid"] == row["sid"]].copy()
        part["date"] = pd.to_datetime(part["time"]).dt.date
        daily = part.groupby("date").agg(actual=("actual_wind_kt", "mean"), A=("predicted_A_kt", "mean"), B=("predicted_B_kt", "mean"), C=("predicted_C_kt", "mean")).reset_index()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(daily["date"], daily["actual"], marker="o", label="Actual")
        ax.plot(daily["date"], daily["A"], marker="o", label="Model A")
        ax.plot(daily["date"], daily["B"], marker="o", label="Model B")
        ax.plot(daily["date"], daily["C"], marker="o", label="Model C")
        ax.set_xlabel("Observation date")
        ax.set_ylabel("Wind speed (kt)")
        ax.set_title(f"{row['name']} ({row['sid']}) · daily mean across available satellite chips · {slug.replace('_', ' ')}")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(OUT / f"storm_series_{slug}.png")
        plt.close(fig)


def _markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def main() -> dict:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing diagnostics directory: {OUT}")
    if not INDEX.is_file():
        raise FileNotFoundError(INDEX)
    OUT.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(INDEX)
    test = index[(index["channel"] == "ir") & (index["split"] == "test")].copy()
    test = test.dropna(subset=["sid", "name", "time", "image_path", "wind_kt", "category"])
    if test["image_path"].duplicated().any():
        raise ValueError("Test index contains duplicate image_path values")
    absent = [p for p in test["image_path"] if not (DATA / p).is_file()]
    if absent:
        raise FileNotFoundError(f"Missing test images, examples: {absent[:5]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = generate_aligned_predictions(test, device)
    predictions.to_csv(OUT / "aligned_test_predictions.csv", index=False)

    storms = storm_comparison(predictions)
    cats = category_comparison(predictions)
    storm_cats = storm_category_comparison(predictions)
    storms.round(4).to_csv(OUT / "storm_level_comparison.csv", index=False)
    cats.round(4).to_csv(OUT / "category_comparison.csv", index=False)
    storm_cats.round(4).to_csv(OUT / "high_category_storm_comparison.csv", index=False)
    cases = build_error_cases(predictions)
    cases.to_csv(OUT / "error_cases.csv", index=False)

    model_summaries = {}
    metric_file_comparison = {}
    for model_id, spec in MODELS.items():
        actual = predictions["actual_wind_kt"]
        pred = predictions[f"predicted_{model_id}_kt"]
        model_summaries[model_id] = summarize(actual, pred, predictions["sid"])
        historical = json.loads((ROOT / spec["metrics"]).read_text(encoding="utf-8"))
        expected = historical["test_model"]
        metric_file_comparison[model_id] = {
            "metrics_file_test_model": expected,
            "recomputed_on_original_test_split": {
                "images": model_summaries[model_id]["images"],
                "mae_kt": round(model_summaries[model_id]["mae_kt"], 2),
                "rmse_kt": round(model_summaries[model_id]["rmse_kt"], 2),
            },
            "same_as_file_rounded_mae_rmse": round(model_summaries[model_id]["mae_kt"], 2) == expected["mae_kt"] and round(model_summaries[model_id]["rmse_kt"], 2) == expected["rmse_kt"],
        }

    improved = storms[storms["mae_improvement_c_vs_a_kt"] > 0]
    worsened = storms[storms["mae_improvement_c_vs_a_kt"] < 0]
    high_category_consistency = {}
    for cat in ("C3", "C4", "C5"):
        part = storm_cats[storm_cats["category"] == cat]
        high_category_consistency[cat] = {
            "storm_category_cases": int(len(part)),
            "storms_improved_by_c": int(part["model_c_improved"].sum()) if len(part) else 0,
            "storms_worsened_by_c": int((~part["model_c_improved"]).sum()) if len(part) else 0,
            "mean_mae_improvement_c_vs_a_kt": float(part["mae_improvement_c_vs_a_kt"].mean()) if len(part) else None,
            "storm_rows": part.sort_values("mae_improvement_c_vs_a_kt", ascending=False).to_dict(orient="records"),
        }

    summaries = {
        "evaluation": {
            "dataset": "kaggle_multiyear45",
            "channel": "ir",
            "split": "original storm-held-out test split (2018-2020)",
            "images": int(len(predictions)),
            "storms": int(predictions["sid"].nunique()),
            "device": str(device),
            "prediction_alignment_key": "image_path",
            "target_match_after_alignment": True,
            "raw_predictions_clipped": False,
            "separate_temporal_holdout_files_used_for_this_comparison": False,
        },
        "models": {k: {"name": MODELS[k]["label"], **v} for k, v in model_summaries.items()},
        "historical_metrics_crosscheck": metric_file_comparison,
        "model_a_vs_c_storm_level": {
            "error_difference_definition": "MAE(Model C) - MAE(Model A); negative favors Model C. Improvement column is the inverse, A minus C.",
            "storms_improved_by_c": int(len(improved)),
            "storms_worsened_by_c": int(len(worsened)),
            "ties": int(len(storms) - len(improved) - len(worsened)),
            "mean_mae_change_c_minus_a_kt": float(storms["mae_change_c_minus_a_kt"].mean()),
            "median_mae_change_c_minus_a_kt": float(storms["mae_change_c_minus_a_kt"].median()),
            "largest_improvements": storms.nlargest(5, "mae_improvement_c_vs_a_kt").to_dict(orient="records"),
            "largest_regressions": storms.nsmallest(5, "mae_improvement_c_vs_a_kt").to_dict(orient="records"),
            "paired_storm_bootstrap": paired_bootstrap(storms),
        },
        "category_metrics": cats.to_dict(orient="records"),
        "high_category_consistency": high_category_consistency,
        "failure_diagnostics": {
            "negative_predictions_by_model": {m: int((predictions[f"predicted_{m}_kt"] < 0).sum()) for m in MODELS},
            "negative_prediction_rows": predictions.loc[(predictions[[f"predicted_{m}_kt" for m in MODELS]] < 0).any(axis=1), ["sid", "name", "time", "image_path", "actual_wind_kt", "actual_category", *[f"predicted_{m}_kt" for m in MODELS]]].to_dict(orient="records"),
            "extreme_error_threshold_kt": 50,
            "extreme_error_case_count_by_model": {m: int(((predictions[f"predicted_{m}_kt"] - predictions["actual_wind_kt"]).abs() >= 50).sum()) for m in MODELS},
            "prediction_spike_definition": "absolute prediction change >=30 kt between observations no more than 48 hours apart; details are in error_cases.csv",
            "prediction_spike_case_count_by_model": {m: int(((cases["model"] == m) & (cases["case_type"] == "prediction_step_change_at_least_30_kt_within_48h")).sum()) for m in MODELS},
            "largest_errors_by_model": {m: cases[(cases["model"] == m) & (cases["case_type"] == "largest_absolute_errors")].head(10)[["sid", "name", "time", "image_path", "actual_category", "actual_wind_kt", f"predicted_{m}_kt", f"error_{m}_kt", f"abs_error_{m}_kt"]].to_dict(orient="records") for m in MODELS},
        },
        "caution": [
            "This is diagnostic analysis on a repeatedly examined test split, not an unbiased final estimate.",
            "The split has only 15 independent storms; images within a storm are correlated.",
            "Category-specific results, especially for categories with low_sample_flag, have few independent cases.",
            "These image-level metrics do not establish operational forecasting reliability.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    make_plots(predictions, storms, cats, storm_cats)

    headline = []
    for model_id, spec in MODELS.items():
        s = model_summaries[model_id]
        headline.append({"Model": model_id, "Experiment": spec["label"], "Images": s["images"], "Storms": s["storms"], "MAE (kt)": f"{s['mae_kt']:.2f}", "RMSE (kt)": f"{s['rmse_kt']:.2f}", "Category accuracy": f"{np.mean([category(max(p, 0)) == cat for p, cat in zip(predictions[f'predicted_{model_id}_kt'], predictions['actual_category'])]):.3f}"})
    top_cols = ["sid", "name", "images", "mae_improvement_c_vs_a_kt"]
    tops = [
        {"sid": r.sid, "name": r["name"], "images": int(r["images"]), "C improvement (kt)": f"{r['mae_improvement_c_vs_a_kt']:.2f}"}
        for _, r in storms.nlargest(5, "mae_improvement_c_vs_a_kt").iterrows()
    ]
    regressions = [
        {"sid": r.sid, "name": r["name"], "images": int(r["images"]), "C improvement (kt)": f"{r['mae_improvement_c_vs_a_kt']:.2f}"}
        for _, r in storms.nsmallest(5, "mae_improvement_c_vs_a_kt").iterrows()
    ]
    md = [
        "# Storm-level error analysis: weighted learning-rate comparison", "",
        f"**Evaluation:** original 2018–2020 storm-held-out IR test split; {len(predictions)} images from {predictions['sid'].nunique()} storms. Predictions are aligned by `image_path`, and actual targets match across all three checkpoints.", "",
        "This is diagnostic analysis of a test split already examined in prior work. It is not a final unbiased model-selection estimate.", "",
        "## Overall image-level results", "",
        _markdown_table(headline, ["Model", "Experiment", "Images", "Storms", "MAE (kt)", "RMSE (kt)", "Category accuracy"]), "",
        "## Model C versus Model A by storm", "",
        "Improvement is defined as MAE(A) − MAE(C), so positive values favor Model C. The stored change column C−A uses the opposite sign.", "",
        f"Model C improved {len(improved)} storms, worsened {len(worsened)}, and tied {len(storms) - len(improved) - len(worsened)}. Mean/median storm MAE change C−A: {storms['mae_change_c_minus_a_kt'].mean():.2f} / {storms['mae_change_c_minus_a_kt'].median():.2f} kt.", "",
        f"Paired storm bootstrap 95% percentile interval for mean C−A MAE: [{summaries['model_a_vs_c_storm_level']['paired_storm_bootstrap']['percentile_95_ci_kt'][0]:.2f}, {summaries['model_a_vs_c_storm_level']['paired_storm_bootstrap']['percentile_95_ci_kt'][1]:.2f}] kt, resampling {len(storms)} storms. This interval is exploratory and unstable with so few storms.", "",
        "### Five largest improvements for C", "",
        _markdown_table(tops, ["sid", "name", "images", "C improvement (kt)"]), "",
        "### Five largest regressions for C", "",
        _markdown_table(regressions, ["sid", "name", "images", "C improvement (kt)"]), "",
        "## High-intensity categories", "",
        _markdown_table(cats[cats.category.isin(["C3", "C4", "C5"])][["category", "model", "images", "storms", "actual_mean_kt", "predicted_mean_kt", "mae_kt", "rmse_kt", "mean_bias_kt", "median_absolute_error_kt", "maximum_absolute_error_kt", "low_sample_flag"]].round(2).to_dict(orient="records"), ["category", "model", "images", "storms", "actual_mean_kt", "predicted_mean_kt", "mae_kt", "rmse_kt", "mean_bias_kt", "median_absolute_error_kt", "maximum_absolute_error_kt", "low_sample_flag"]), "",
        "Per-storm C3/C4/C5 comparisons are in `high_category_storm_comparison.csv`; category-specific storm counts and C-versus-A consistency are also in `summary.json`. Categories flagged low-sample have fewer than 10 images or fewer than 5 storms.", "",
        "## Prediction checks", "",
        f"Negative predictions: {json.dumps(summaries['failure_diagnostics']['negative_predictions_by_model'])}. Errors ≥50 kt and ≥30 kt prediction steps within 48 hours are counted in `summary.json`; individual cases are in `error_cases.csv`.", "",
        "Plots: `per_storm_mae_A_vs_C.png`, `actual_vs_predicted_A_vs_C.png`, `category_mae_A_B_C.png`, `storm_series_most_improved.png`, and `storm_series_most_regressed.png`.", "",
        "## Limitations", "",
        "Only 15 independent storms are in this split, so images are correlated and storm-level estimates are noisy. C3–C5 may contain very few storms. The 2021–2023 temporal-holdout files and train/validation predictions were not used for this test-split comparison. These results do not establish operational forecasting reliability.", "",
        "## Artifacts", "",
        "- `aligned_test_predictions.csv` — raw, aligned predictions for all three checkpoints.",
        "- `storm_level_comparison.csv` — storm-by-storm A/B/C metrics and paired changes.",
        "- `category_comparison.csv` — category metrics for all three models.",
        "- `high_category_storm_comparison.csv` — C3/C4/C5 storm-level consistency data.",
        "- `error_cases.csv` — most severe errors, negative predictions, paired wins/losses, and prediction spikes.",
        "- `summary.json` — machine-readable analysis and metric-file cross-checks.",
    ]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"out_dir": str(OUT), "images": len(predictions), "storms": len(storms), "device": str(device), "models": model_summaries, "storms_improved_by_c": len(improved), "storms_worsened_by_c": len(worsened), "category_consistency": {k: {"improved": v["storms_improved_by_c"], "total": v["storm_category_cases"]} for k, v in high_category_consistency.items()}}, indent=2))
    return summaries


if __name__ == "__main__":
    main()
