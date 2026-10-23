"""Extend the saved negative-prediction audit without changing trained artifacts."""
from __future__ import annotations
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports/production_candidate_audit_20261022_01/raw_prediction_reproduction.csv"
TEST_INPUT = ROOT / "models/storm_diagnostics_lr_comparison/aligned_test_predictions.csv"
OUT = ROOT / "reports/negative_prediction_audit_20261023_01"
MODELS = {
    "A": ROOT / "models/multiyear45_resnet18_ir_weighted.pt",
    "C": ROOT / "models/multiyear45_resnet18_ir_weighted_lr1e4.pt",
}
PARTITIONS = ["2018_2020_train_in_sample", "2018_2020_val_storm_heldout",
              "2018_2020_test_previously_saved", "2021_2023_temporal_holdout"]
CATS = ["TD", "TS", "C1", "C2", "C3", "C4", "C5"]
TOLERANCE_KT = 1e-5
sys.path.insert(0, str(ROOT))
from data.build_dataset import category


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def score_predictions(actual, predicted):
    """Score finite pairs without mutating inputs or postprocessing predictions."""
    y0, p0 = np.asarray(actual, float), np.asarray(predicted, float)
    keep = np.isfinite(y0) & np.isfinite(p0)
    y, p = y0[keep], p0[keep]
    if not len(y):
        return {"metric_rows": 0, "mae_kt": math.nan, "rmse_kt": math.nan,
                "mean_bias_kt": math.nan, "category_accuracy": math.nan,
                "within_one_category": math.nan}
    err = p - y
    yc = np.array([CATS.index(category(float(x))) for x in y])
    pc = np.array([CATS.index(category(float(x))) for x in p])
    return {"metric_rows": int(len(y)), "mae_kt": float(np.abs(err).mean()),
            "rmse_kt": float(np.sqrt(np.mean(err ** 2))), "mean_bias_kt": float(err.mean()),
            "category_accuracy": float((yc == pc).mean()),
            "within_one_category": float((np.abs(yc - pc) <= 1).mean())}


def compare_metrics(frame):
    full, summaries = [], []
    for (model, partition), g in frame.groupby(["model", "partition"], sort=True):
        g = g.copy()
        y = pd.to_numeric(g.actual_wind_kt, errors="coerce").to_numpy(float)
        raw = pd.to_numeric(g.predicted_wind_kt, errors="coerce").to_numpy(float)
        clipped, changed = np.maximum(raw, 0.0), np.isfinite(raw) & (raw < 0)
        if np.any(y[np.isfinite(y)] < 0):
            raise AssertionError("Actual wind target is negative.")
        if changed.any() and np.any(np.abs(clipped[changed] - y[changed]) >
                                    np.abs(raw[changed] - y[changed]) + TOLERANCE_KT):
            raise AssertionError("Clipping increased absolute error for a nonnegative target.")
        raw_cat = [category(float(x)) if np.isfinite(x) else None for x in raw]
        clip_cat = [category(float(x)) if np.isfinite(x) else None for x in clipped]
        if raw_cat != clip_cat:
            raise AssertionError("Category conversion changed after clipping.")
        rs, cs = score_predictions(y, raw), score_predictions(y, clipped)
        g["normalized_model_output"] = raw / 100.0
        g["raw_prediction_kt"] = raw
        g["clipped_display_prediction_kt"] = clipped
        g["raw_absolute_error_kt"] = np.abs(raw - y)
        g["clipped_absolute_error_kt"] = np.abs(clipped - y)
        g["raw_predicted_category"], g["clipped_display_category"] = raw_cat, clip_cat
        g["changed_by_clipping"] = changed
        full.append(g)
        summaries.append({
            "model": model, "partition": partition, "rows": len(g),
            "raw_negative_predictions_before_postprocessing": int(changed.sum()),
            "raw_nonfinite_predictions": int((~np.isfinite(raw)).sum()),
            **{"raw_" + k: v for k, v in rs.items()},
            **{"clipped_" + k: v for k, v in cs.items()},
            "predictions_changed_by_clipping": int(changed.sum()),
            "mean_correction_magnitude_kt_among_changed": float((clipped[changed]-raw[changed]).mean()) if changed.any() else 0.0,
            "mean_correction_magnitude_kt_all_rows": float((clipped-raw).mean()),
            "maximum_correction_magnitude_kt": float((clipped[changed]-raw[changed]).max()) if changed.any() else 0.0,
            "category_accuracy_unchanged": rs["category_accuracy"] == cs["category_accuracy"],
            "within_one_category_unchanged": rs["within_one_category"] == cs["within_one_category"],
        })
    return pd.concat(full, ignore_index=True), pd.DataFrame(summaries)


def partition_summary(frame):
    rows = []
    for (model, partition), g in frame.groupby(["model", "partition"], sort=True):
        p = pd.to_numeric(g.predicted_wind_kt, errors="coerce")
        n = p[p < 0]
        rows.append({"model": model, "partition": partition, "images": len(g),
            "storms": int(g.sid.nunique()), "negative_predictions": len(n),
            "negative_percentage": float(100 * len(n) / len(g)),
            "minimum_raw_prediction_kt": float(p.min()),
            "median_negative_prediction_kt": float(n.median()) if len(n) else None,
            "most_negative_prediction_kt": float(n.min()) if len(n) else None,
            "below_minus_5_kt": int((p < -5).sum()), "below_minus_10_kt": int((p < -10).sum()),
            "below_minus_20_kt": int((p < -20).sum()), "nonfinite_predictions": int((~np.isfinite(p)).sum())})
    return pd.DataFrame(rows)


def align_models(frame):
    all_rows, overview = [], []
    for partition in PARTITIONS:
        p = frame[frame.partition == partition]
        a = p[p.model == "A"][["image_path", "sid", "time", "actual_wind_kt",
            "predicted_wind_kt", "is_negative"]].rename(columns={
            "predicted_wind_kt": "model_A_raw_prediction_kt", "is_negative": "model_A_negative"})
        c = p[p.model == "C"][["image_path", "sid", "time", "actual_wind_kt",
            "predicted_wind_kt", "is_negative"]].rename(columns={
            "sid": "model_C_sid", "time": "model_C_time", "actual_wind_kt": "model_C_actual_wind_kt",
            "predicted_wind_kt": "model_C_raw_prediction_kt", "is_negative": "model_C_negative"})
        m = a.merge(c, on="image_path", how="outer", validate="one_to_one", indicator=True)
        m["partition"] = partition
        m["both_predictions_available"] = m._merge.eq("both")
        m["actual_target_matches"] = np.isclose(
            pd.to_numeric(m.actual_wind_kt, errors="coerce"),
            pd.to_numeric(m.model_C_actual_wind_kt, errors="coerce"), rtol=0, atol=1e-8, equal_nan=False)
        m["storm_id_matches"] = m.sid.astype(str) == m.model_C_sid.astype(str)
        m["timestamp_matches"] = m.time.astype(str) == m.model_C_time.astype(str)
        common = m[m.both_predictions_available]
        if not common.actual_target_matches.all() or not common.storm_id_matches.all() or not common.timestamp_matches.all():
            raise AssertionError(f"A/C target, storm, or time mismatch in {partition}")
        m["model_C_minus_A_raw_kt"] = (
            pd.to_numeric(m.model_C_raw_prediction_kt, errors="coerce") -
            pd.to_numeric(m.model_A_raw_prediction_kt, errors="coerce"))
        m["both_models_negative"] = m.model_A_negative.fillna(False) & m.model_C_negative.fillna(False)
        m["A_only_negative"] = m.model_A_negative.fillna(False) & ~m.model_C_negative.fillna(False)
        m["C_only_negative"] = m.model_C_negative.fillna(False) & ~m.model_A_negative.fillna(False)
        overview.append({"partition": partition, "aligned_images": len(common), "outer_join_rows": len(m),
            "both_negative_images": int(m.both_models_negative.sum()),
            "A_only_negative_images": int(m.A_only_negative.sum()),
            "C_only_negative_images": int(m.C_only_negative.sum()),
            "negative_union_images": int((m.both_models_negative | m.A_only_negative | m.C_only_negative).sum())})
        all_rows.append(m.drop(columns="_merge"))
    return pd.concat(all_rows, ignore_index=True), overview


def reproduce_examples(frame):
    """Repeat representative raw inference and compare existing eval transforms."""
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from ml.model import CycloneImageClassifier
    from scripts.evaluate_weighted_validation import ChipDataset
    from scripts.evaluate_temporal_test_weighted import TemporalDataset

    transform = T.Compose([T.Resize(224), T.ToTensor(),
        T.Normalize([.485, .456, .406], [.229, .224, .225])])
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    examples = []
    for model_id in ["A", "C"]:
        for part_name in ["2018_2020_val_storm_heldout", "2021_2023_temporal_holdout"]:
            p = frame[(frame.model == model_id) & (frame.partition == part_name)].copy()
            n = p[p.predicted_wind_kt < 0].sort_values("predicted_wind_kt")
            if len(n): examples.append((model_id, n.iloc[0], "most_negative"))
            normal = p[p.predicted_wind_kt >= 0].copy()
            if len(normal):
                normal["ae"] = (normal.predicted_wind_kt-normal.actual_wind_kt).abs()
                pos = (normal.ae-normal.ae.median()).abs().argmin()
                examples.append((model_id, normal.iloc[pos], "median_absolute_error_nonnegative"))
    checks = []
    for model_id, checkpoint_path in MODELS.items():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (ckpt.get("architecture"), ckpt.get("channel"), ckpt.get("target")) != ("resnet18", "ir", "wind_kt/100"):
            raise AssertionError(f"Unexpected checkpoint metadata for Model {model_id}")
        model = CycloneImageClassifier(num_classes=1, pretrained=False).cpu()
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        for selected, row, kind in examples:
            if selected != model_id: continue
            part_name = str(row.partition)
            if "temporal" in part_name:
                root, cls = ROOT / "kaggle_temporal_test", TemporalDataset
                data = {"image_path": row.image_path, "wind_kt": row.actual_wind_kt,
                        "sid": row.sid, "time": row.time, "name": row.get("name", "")}
            else:
                root, cls = ROOT / "kaggle_multiyear45", ChipDataset
                data = {"image_path": row.image_path, "wind_kt": row.actual_wind_kt,
                        "sid": row.sid, "time": row.time}
            with Image.open(root / str(row.image_path)) as im:
                x = transform(im.convert("RGB"))
            eval_x = cls(root, pd.DataFrame([data]))[0]
            tensor_diff = float(torch.max(torch.abs(x-eval_x)).item())
            values = []
            with torch.inference_mode():
                for _ in range(3):
                    values.append(float(model(x.unsqueeze(0)).squeeze().item()*100))
                pipeline = float(model(eval_x.unsqueeze(0)).squeeze().item()*100)
            reference = float(row.predicted_wind_kt)
            checks.append({"model": model_id, "partition": part_name, "image_path": row.image_path,
                "case_kind": kind, "repeat_count": 3, "raw_repeat_predictions_kt": values,
                "repeat_spread_kt": max(values)-min(values), "reference_raw_prediction_kt": reference,
                "max_repeat_to_reference_abs_diff_kt": max(abs(v-reference) for v in values),
                "standalone_vs_eval_pipeline_abs_diff_kt": abs(values[0]-pipeline),
                "max_preprocessed_tensor_abs_diff": tensor_diff,
                "reproducible_within_tolerance": max(values)-min(values) <= TOLERANCE_KT,
                "pipeline_prediction_matches_reference": abs(pipeline-reference) <= TOLERANCE_KT,
                "preprocessing_matches_evaluation_pipeline": tensor_diff <= 1e-7})
    return checks


def write_report(path, report):
    lines = ["# CycloneSense negative prediction audit", "",
        f"Rows: {report['prediction_rows']}; raw negatives: {report['negative_rows']}; non-finite outputs: {report['nonfinite_raw_predictions']}.", "",
        "## Findings", "",
        "- Model A and C use a one-value unconstrained ResNet18 regression head trained on wind_kt/100; inference multiplies by 100.",
        "- Repeated predictions and existing evaluation transforms were compared on negative and normal validation and temporal examples.",
        "- Results are consistent with the unconstrained regression head. No target-scale, preprocessing, or checkpoint-loading mismatch was found in the executed checks.",
        "- Zero clipping is appropriate as a display safeguard for nonnegative wind. It leaves category results unchanged because negative winds already map to TD; clipping does not repair the learned regression function.",
        "- Raw predictions and raw metrics remain separate. Previously examined test rows were included only for this diagnostic, not for tuning or model selection.", "",
        "## Negative predictions by partition", "",
        "| Model | Partition | Images | Negative | Percent | Median neg. (kt) | Minimum (kt) | < -5 | < -10 | < -20 | Nonfinite |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["negative_counts_by_model_partition"]:
        lines.append(f"| {r['model']} | {r['partition']} | {r['images']} | {r['negative_predictions']} | {r['negative_percentage']:.2f}% | {r['median_negative_prediction_kt']} | {r['most_negative_prediction_kt']} | {r['below_minus_5_kt']} | {r['below_minus_10_kt']} | {r['below_minus_20_kt']} | {r['nonfinite_predictions']} |")
    lines += ["", "## Raw versus clipped metrics", "",
        "| Model | Partition | Raw MAE | Clip MAE | Raw RMSE | Clip RMSE | Raw bias | Clip bias | Raw cat. acc. | Clip cat. acc. | Changed | Mean correction (kt) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["raw_vs_clipped_metrics"]:
        lines.append(f"| {r['model']} | {r['partition']} | {r['raw_mae_kt']:.3f} | {r['clipped_mae_kt']:.3f} | {r['raw_rmse_kt']:.3f} | {r['clipped_rmse_kt']:.3f} | {r['raw_mean_bias_kt']:.3f} | {r['clipped_mean_bias_kt']:.3f} | {r['raw_category_accuracy']:.3f} | {r['clipped_category_accuracy']:.3f} | {r['predictions_changed_by_clipping']} | {r['mean_correction_magnitude_kt_among_changed']:.3f} |")
    lines += ["", "For actual wind y >= 0 and a raw prediction p < 0, |0-y| <= |p-y|; clipping cannot increase that row's absolute error. Bias moves upward. The category conversion already maps every value below 34 kt to TD.", "",
        "## A/C negative-image overlap", "",
        "| Partition | Aligned images | Both negative | A only | C only | Union |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in report["model_alignment"]:
        lines.append(f"| {r['partition']} | {r['aligned_images']} | {r['both_negative_images']} | {r['A_only_negative_images']} | {r['C_only_negative_images']} | {r['negative_union_images']} |")
    lines += ["", "## Reproducibility", "",
        f"Repeated checks: {len(report['reproducibility_checks'])}; each selected validation or temporal example was run three times per model. Detailed results are in reproducibility_checks.csv.", "",
        "## Limits", "",
        "- 2018-2020 test predictions were read from the existing saved A/C prediction CSV, not regenerated here.",
        "- No per-batch historical training-loss trace exists, so original training losses cannot be exhaustively checked.",
        "- Training rows are in-sample; validation, temporal, and test comparisons are diagnostic only.", "",
        "## Artifacts", "",
        "- negative_cases.csv: all negative rows, normalized output, absolute error.",
        "- partition_summary.csv: rates, severity counts, median, minimum, and non-finite counts.",
        "- model_a_model_c_alignment.csv: path joins with target, storm, and timestamp checks.",
        "- raw_vs_clipped_metrics.csv and raw_vs_clipped_predictions.csv: separate raw and display-clipped metrics and values.",
        "- reproducibility_checks.csv: repeated inference and evaluation-pipeline comparisons.",
        "- summary.json: machine-readable report and checkpoint hashes.", "",
        "Checkpoint hashes were identical before and after. Existing checkpoints, metrics, and prediction files were not written.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if OUT.exists(): raise FileExistsError(f"Refusing to overwrite {OUT}")
    if not INPUT.is_file() or not TEST_INPUT.is_file(): raise FileNotFoundError("Prior raw/test prediction input missing.")
    before = {k: sha256(v) for k, v in MODELS.items()}
    frame = pd.read_csv(INPUT)
    needed = {"model", "partition", "image_path", "sid", "time", "actual_wind_kt",
              "predicted_wind_kt", "is_negative", "is_nonfinite"}
    if not needed.issubset(frame.columns): raise ValueError(f"Missing columns: {needed-set(frame.columns)}")
    frame["model"], frame["partition"] = frame.model.astype(str), frame.partition.astype(str)
    frame["actual_wind_kt"] = pd.to_numeric(frame.actual_wind_kt, errors="coerce")
    frame["predicted_wind_kt"] = pd.to_numeric(frame.predicted_wind_kt, errors="coerce")
    if frame.duplicated(["model", "partition", "image_path"]).any(): raise AssertionError("Duplicate prediction key.")
    if not np.array_equal(frame.is_negative.astype(bool), frame.predicted_wind_kt < 0): raise AssertionError("Negative flags disagree.")
    if (~np.isfinite(frame.predicted_wind_kt)).any() or (~np.isfinite(frame.actual_wind_kt)).any(): raise AssertionError("Non-finite output/target.")
    if (frame.actual_wind_kt < 0).any(): raise AssertionError("Negative actual wind.")
    if set(frame.partition.unique()) != set(PARTITIONS): raise AssertionError("Unexpected/missing partitions.")
    OUT.mkdir(parents=True, exist_ok=False)

    negative = frame[frame.predicted_wind_kt < 0].copy()
    negative["normalized_model_output"] = negative.predicted_wind_kt/100
    negative["absolute_error_kt"] = (negative.predicted_wind_kt-negative.actual_wind_kt).abs()
    negative["predicted_category"] = negative.predicted_wind_kt.map(lambda x: category(float(x)))
    negative.to_csv(OUT/"negative_cases.csv", index=False)
    parts = partition_summary(frame); parts.to_csv(OUT/"partition_summary.csv", index=False)
    aligned, overlap = align_models(frame); aligned.to_csv(OUT/"model_a_model_c_alignment.csv", index=False)
    full, metrics = compare_metrics(frame)
    full.to_csv(OUT/"raw_vs_clipped_predictions.csv", index=False)
    metrics.to_csv(OUT/"raw_vs_clipped_metrics.csv", index=False)
    repro = reproduce_examples(frame); pd.DataFrame(repro).to_csv(OUT/"reproducibility_checks.csv", index=False)
    if not all(x["reproducible_within_tolerance"] and x["pipeline_prediction_matches_reference"] and x["preprocessing_matches_evaluation_pipeline"] for x in repro):
        raise AssertionError("A repeat, saved-value, or evaluation-transform comparison failed.")
    after = {k: sha256(v) for k, v in MODELS.items()}
    if before != after: raise AssertionError("A checkpoint hash changed.")
    report = {"scope": "Postprocessing diagnostics only; no training, tuning, checkpoint changes, or historical artifact edits.",
        "input_raw_predictions": str(INPUT.relative_to(ROOT)),
        "test_predictions_were_previously_saved": str(TEST_INPUT.relative_to(ROOT)),
        "prediction_rows": len(frame), "negative_rows": len(negative),
        "nonfinite_raw_predictions": int((~np.isfinite(frame.predicted_wind_kt)).sum()),
        "checkpoint_sha256_before": before, "checkpoint_sha256_after": after,
        "negative_counts_by_model_partition": parts.to_dict(orient="records"),
        "model_alignment": overlap, "raw_vs_clipped_metrics": metrics.to_dict(orient="records"),
        "reproducibility_checks": repro,
        "verified_inferences": [
            "Normalized output is raw_prediction_kt / 100.",
            "Clipped values and metrics are separate; raw predictions and raw metrics remain intact.",
            "Clipping cannot increase absolute error when actual wind is nonnegative.",
            "Negative predictions already map to TD, leaving category metrics unchanged after clipping.",
            "Existing test values were included only for diagnostics, not tuning/model selection."]}
    (OUT/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_report(OUT/"report.md", report)
    print(json.dumps({"output_directory": str(OUT), "rows": len(frame), "negative_rows": len(negative),
        "reproducibility_checks": len(repro), "checkpoints_unchanged": before == after,
        "files": sorted(p.name for p in OUT.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
