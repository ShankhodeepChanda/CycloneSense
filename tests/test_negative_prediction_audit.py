import numpy as np
import pandas as pd

from data.build_dataset import category
from scripts.negative_prediction_audit import compare_metrics, score_predictions


def test_raw_and_clipped_metrics_stay_separate_and_raw_values_are_preserved():
    original = pd.DataFrame({
        "model": ["A", "A"],
        "partition": ["validation", "validation"],
        "actual_wind_kt": [20.0, 40.0],
        "predicted_wind_kt": [-10.0, 30.0],
        "is_negative": [True, False],
    })
    before = original["predicted_wind_kt"].to_list()

    full, summary = compare_metrics(original)

    assert original["predicted_wind_kt"].to_list() == before
    assert full["raw_prediction_kt"].to_list() == before
    assert full["clipped_display_prediction_kt"].to_list() == [0.0, 30.0]
    assert summary.loc[0, "raw_mae_kt"] == 20.0
    assert summary.loc[0, "clipped_mae_kt"] == 15.0
    assert summary.loc[0, "predictions_changed_by_clipping"] == 1
    assert summary.loc[0, "category_accuracy_unchanged"]
    assert summary.loc[0, "within_one_category_unchanged"]


def test_nonfinite_predictions_are_excluded_from_scoring_and_negative_is_td():
    actual = np.array([20.0, 35.0, 40.0])
    raw = np.array([-5.0, np.nan, 50.0])
    untouched = raw.copy()

    scored = score_predictions(actual, raw)

    assert scored["metric_rows"] == 2
    assert np.isnan(raw[1])
    assert np.array_equal(raw, untouched, equal_nan=True)
    assert category(-5.0) == category(0.0) == "TD"
