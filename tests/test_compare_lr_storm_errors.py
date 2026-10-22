import pandas as pd

from scripts.compare_lr_storm_errors import category_comparison, paired_bootstrap, storm_comparison, summarize


def fixture_predictions():
    return pd.DataFrame([
        {"sid": "S1", "name": "ONE", "actual_wind_kt": 30.0, "actual_category": "TD", "predicted_A_kt": 20.0, "predicted_B_kt": 25.0, "predicted_C_kt": 30.0},
        {"sid": "S1", "name": "ONE", "actual_wind_kt": 50.0, "actual_category": "TS", "predicted_A_kt": 40.0, "predicted_B_kt": 45.0, "predicted_C_kt": 55.0},
        {"sid": "S2", "name": "TWO", "actual_wind_kt": 80.0, "actual_category": "C1", "predicted_A_kt": 82.0, "predicted_B_kt": 100.0, "predicted_C_kt": 100.0},
    ])


def test_summary_uses_predicted_minus_actual_bias_and_raw_negative_predictions():
    result = summarize([20, 30], [-5, 40], ["S1", "S2"])
    assert result["mean_bias_kt"] == -7.5
    assert result["negative_prediction_count"] == 1
    assert result["maximum_absolute_error_kt"] == 25


def test_storm_comparison_sign_is_positive_when_model_c_improves():
    result = storm_comparison(fixture_predictions()).set_index("sid")
    assert result.loc["S1", "mae_improvement_c_vs_a_kt"] == 7.5
    assert result.loc["S1", "mae_change_c_minus_a_kt"] == -7.5
    assert bool(result.loc["S1", "model_c_improved"])
    assert bool(result.loc["S2", "model_c_worsened"])


def test_category_counts_flag_small_independent_samples_and_bootstrap_is_repeatable():
    predictions = fixture_predictions()
    categories = category_comparison(predictions)
    c5 = categories[categories["category"] == "C5"]
    assert c5.empty
    assert categories["low_sample_flag"].all()
    storms = storm_comparison(predictions)
    first = paired_bootstrap(storms, repetitions=100, seed=8)
    second = paired_bootstrap(storms, repetitions=100, seed=8)
    assert first == second
    assert first["storms"] == 2
