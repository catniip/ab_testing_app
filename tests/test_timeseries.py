import pandas as pd
import numpy as np
import pytest

from src.experiment_platform.core import time_series_campaign
from src.experiment_platform.charts import time_series_line
from src.experiment_platform.timeseries import (
    _seasonal_period,
    assess_model_reliability,
    campaign_decision_date,
    classify_calibration,
    classify_historical_calibration,
    duration_power_status,
    build_business_interpretation,
    build_timeseries_preview,
    detect_frequency,
    fit_bsts_model,
    infer_timeseries_columns,
    prepare_timeseries_data,
    predictor_candidates,
    projected_operational_exposure,
    run_pre_post_analysis,
    run_duration_power_simulation,
    regularize_timeseries_data,
    wilson_interval,
    method_config_fingerprint,
    planning_durations,
    power_target_met,
    TimeSeriesConfig,
)
from src.experiment_platform import timeseries as timeseries_module


def test_timeseries_schema_and_frequency_detection():
    df = time_series_campaign()
    inferred = infer_timeseries_columns(df)
    assert inferred["date"] == "date"
    assert inferred["outcome"] == "applications"
    prepared = prepare_timeseries_data(df, "date", "applications")
    frequency, missing = detect_frequency(prepared)
    assert frequency == "Weekly"
    assert missing == 0


def test_timeseries_preview_summarizes_campaign_context():
    raw = time_series_campaign()
    preview = build_timeseries_preview(raw, "date", "applications", "campaign_flag")
    assert preview["frequency"] == "Weekly"
    assert preview["missing_periods"] == 0
    assert preview["campaign_launch"] == pd.Timestamp("2025-07-06")
    assert preview["pre_observations"] > preview["post_observations"] > 0
    assert "website_traffic" in preview["predictors"]
    assert "marketing_spend" not in preview["predictors"]


def test_monthly_data_uses_monthly_planning_cadence():
    raw = pd.DataFrame(
        {
            "month": pd.date_range("2022-01-01", periods=36, freq="MS"),
            "applications": 100 + np.arange(36),
        }
    )
    prepared = prepare_timeseries_data(raw, "month", "applications", frequency="Auto")
    frequency, missing = detect_frequency(prepared)

    assert frequency == "Monthly"
    assert missing == 0
    assert planning_durations(frequency, 12) == [2, 3, 4, 6, 9, 12]
    assert campaign_decision_date("2026-01-15", 3, frequency) == pd.Timestamp("2026-04-15")


def test_pre_post_analysis_reports_change():
    df = prepare_timeseries_data(time_series_campaign(), "date", "applications")
    result = run_pre_post_analysis(df, "2025-07-06")
    assert result["pre_observations"] > 0
    assert result["post_observations"] > 0
    assert result["absolute_change"] != 0


def test_bsts_counterfactual_uses_post_period_prediction():
    df = prepare_timeseries_data(time_series_campaign(), "date", "applications")
    result = fit_bsts_model(df, "2025-07-06", "Weekly", ["website_traffic"])
    post = result["result"]
    assert {"counterfactual", "lower", "upper", "impact"}.issubset(post.columns)
    assert len(post) == result["post_observations"]
    assert result["pre_observations"] > result["post_observations"]
    assert 0 <= result["posterior_probability_positive"] <= 1


def test_bsts_requires_enough_pre_history():
    dates = pd.date_range("2025-01-01", periods=25, freq="W")
    df = pd.DataFrame({"date": dates, "outcome": range(25)})
    prepared = prepare_timeseries_data(df, "date", "outcome")
    with pytest.raises(ValueError):
        fit_bsts_model(prepared, "2025-02-01", "Weekly")


def test_structural_model_respects_end_date_and_interval_level():
    prepared = prepare_timeseries_data(time_series_campaign(), "date", "applications")
    result = fit_bsts_model(prepared, "2025-07-06", "Weekly", campaign_end_date="2025-09-28", credible_interval=0.8, seed=99)
    assert result["result"]["_date"].max() == pd.Timestamp("2025-09-28")
    assert result["interval_level"] == 0.8
    assert result["method"] == "Structural Time Series Counterfactual"
    assert result["decision"] in {"positive", "negative", "inconclusive"}


def test_structural_simulation_is_reproducible():
    prepared = prepare_timeseries_data(time_series_campaign(), "date", "applications")
    first = fit_bsts_model(prepared, "2025-07-06", "Weekly", seed=7)
    second = fit_bsts_model(prepared, "2025-07-06", "Weekly", seed=7)
    assert first["prediction_interval"] == second["prediction_interval"]
    assert first["simulation_probability_positive"] == second["simulation_probability_positive"]


def test_duplicate_and_missing_period_policies():
    dates = pd.date_range("2025-01-05", periods=8, freq="W-SUN").delete(3)
    raw = pd.DataFrame({"date": list(dates) + [dates[0]], "outcome": list(range(8))[:7] + [10]})
    averaged = prepare_timeseries_data(raw, "date", "outcome", "Average", "Interpolate", "Weekly")
    summed = prepare_timeseries_data(raw, "date", "outcome", "Sum", "Fill missing counts with zero", "Weekly")
    assert averaged.loc[averaged["_date"] == dates[0], "_outcome"].iloc[0] == 5
    assert summed.loc[summed["_date"] == dates[0], "_outcome"].iloc[0] == 10
    assert len(averaged) == len(summed) == 8
    with pytest.raises(ValueError):
        prepare_timeseries_data(raw, "date", "outcome", "Average", "Fail validation", "Weekly")


def test_total_comparison_uses_equal_duration_windows():
    dates = pd.date_range("2025-01-05", periods=12, freq="W-SUN")
    prepared = prepare_timeseries_data(pd.DataFrame({"date": dates, "outcome": range(12)}), "date", "outcome", frequency="Weekly")
    result = run_pre_post_analysis(prepared, "2025-02-16", comparison_metric="Total")
    assert result["pre_observations"] == result["post_observations"]


def test_daily_seasonality_does_not_use_weekly_period_as_annual():
    assert _seasonal_period("Daily", "Weekly", 30) == 7
    assert _seasonal_period("Daily", "Annual", 365) is None
    assert _seasonal_period("Daily", "Annual", 730) == 365


def test_outcome_inference_and_predictor_filtering_are_business_safe():
    raw = pd.DataFrame(
        {
            "week": pd.date_range("2025-01-05", periods=4, freq="W-SUN"),
            "control_product_applications": [100, 101, 102, 103],
            "credit_card_applications": [200, 210, 220, 230],
            "campaign_spend": [1, 2, 3, 4],
            "website_traffic": [10, 11, 12, 13],
            "true_effect": [0, 0, 10, 10],
        }
    )
    from src.experiment_platform.timeseries import infer_timeseries_columns
    assert infer_timeseries_columns(raw)["outcome"] == "credit_card_applications"
    prepared = prepare_timeseries_data(raw, "week", "credit_card_applications", frequency="Weekly")
    assert predictor_candidates(prepared, "credit_card_applications", "week") == ["control_product_applications", "website_traffic"]


def test_method_fingerprints_ignore_other_method_settings():
    cfg = TimeSeriesConfig()
    pre_hash = method_config_fingerprint(cfg, "Pre–Post Analysis")
    structural_hash = method_config_fingerprint(cfg, "Structural Time Series Counterfactual")
    cfg.seasonality = "None"
    assert method_config_fingerprint(cfg, "Pre–Post Analysis") == pre_hash
    assert method_config_fingerprint(cfg, "Structural Time Series Counterfactual") != structural_hash
    cfg.comparison_window = "All available"
    baseline = TimeSeriesConfig(seasonality="None")
    assert method_config_fingerprint(cfg, "Structural Time Series Counterfactual") == method_config_fingerprint(baseline, "Structural Time Series Counterfactual")


def test_business_interpretation_weakens_positive_low_reliability_result():
    result = build_business_interpretation(100, 0.05, (20, 180), 0.999, "Poor", "applications")
    assert result["status"] == "Positive — Directional"
    assert "reliability is low" in result["headline"]
    assert result["probability_label"] == ">99%"


def test_business_interpretation_marks_zero_crossing_inconclusive():
    result = build_business_interpretation(100, 0.05, (-50, 180), 0.8, "Good", "revenue")
    assert result["status"] == "Inconclusive"


def test_duration_power_status_uses_threshold_not_monotonicity():
    assert duration_power_status(4, 0.75, 0.80, 6) == "Below target"
    assert duration_power_status(6, 0.85, 0.80, 6) == "Recommended"
    assert duration_power_status(8, 0.90, 0.80, 6) == "Above target"
    assert duration_power_status(20, 0.85, 0.80, 6) == "Above target"


def test_calibration_is_relative_to_alpha():
    assert classify_calibration(0.307, 0.05, simulations=20) == "poor"
    assert classify_calibration(0.055, 0.05, simulations=20) == "acceptable"
    assert classify_calibration(0.09, 0.05, simulations=20) == "caution"


def test_sparse_historical_calibration_uses_exact_binomial_evidence():
    assert classify_historical_calibration(1, 12, 0.05)[0] == "acceptable"
    assert classify_historical_calibration(2, 12, 0.05)[0] == "acceptable"
    assert classify_historical_calibration(3, 12, 0.05)[0] == "poor"
    assert classify_historical_calibration(1, 2, 0.05)[0] == "poor"


def test_power_recommendation_uses_lower_simulation_bound_when_available():
    assert power_target_met(0.81, (0.79, 0.83), 0.80) is False
    assert power_target_met(0.84, (0.82, 0.86), 0.80) is True
    assert power_target_met(0.84, (np.nan, np.nan), 0.80) is True


def test_wilson_interval_is_bounded_and_widens_for_small_samples():
    low, high = wilson_interval(1, 2)
    wider_low, wider_high = wilson_interval(1, 20)
    assert 0 <= low <= high <= 1
    assert (high - low) > (wider_high - wider_low)


def test_regularization_does_not_fill_covariates_from_future_periods():
    dates = pd.date_range("2025-01-05", periods=4, freq="W-SUN")
    raw = pd.DataFrame({"_date": dates.delete(1), "_outcome": [10, 30, 40], "predictor": [1.0, 3.0, 4.0]})
    regularized = regularize_timeseries_data(raw, "Weekly", "Interpolate")
    assert pd.isna(regularized.loc[regularized["_date"] == dates[1], "predictor"]).all()


def test_planner_alpha_changes_empirical_detection_threshold(monkeypatch):
    dates = pd.date_range("2020-01-05", periods=110, freq="W-SUN")
    data = pd.DataFrame({"_date": dates, "_outcome": 100.0})

    def fake_fit(frame, intervention_date, frequency, covariates, interval, **kwargs):
        post = frame.loc[frame["_date"] >= pd.Timestamp(intervention_date), ["_date", "_outcome"]].copy()
        post["counterfactual"] = 100.0
        values = np.linspace(-2, 2, kwargs.get("draws", 200))[:, None]
        forecast_draws = 100.0 + np.repeat(values, len(post), axis=1)
        return {"model_stable": True, "result": post, "forecast_draws": forecast_draws}

    timeseries_module._PLANNER_ORIGIN_CACHE.clear()
    monkeypatch.setattr("src.experiment_platform.timeseries.fit_bsts_model", fake_fit)
    common = dict(data=data, frequency="Weekly", covariates=[], seasonality="None", local_trend=False, interval=.95, effect=.05, effect_type="Relative Lift (%)", target_power=.8, maximum_duration=4, simulations=80, seed=7)
    strict = run_duration_power_simulation(alpha=.05, **common)
    relaxed = run_duration_power_simulation(alpha=.10, **common)
    assert strict["power"][0]["empirical_critical_value"] > relaxed["power"][0]["empirical_critical_value"]
    assert strict["confidence_level"] == .95
    assert relaxed["confidence_level"] == .90


def test_planner_validation_adapts_to_available_history():
    assert timeseries_module._planner_validation_design(157, "Weekly", "Auto", 26)["minimum_training"] == 104
    one_year = timeseries_module._planner_validation_design(52, "Weekly", "Auto", 12)
    assert one_year["mode"] == "Limited-history hybrid"
    assert one_year["seasonality"] == "None"
    assert timeseries_module._planner_validation_design(12, "Weekly", "Auto", 12)["mode"] == "Assumption-based"


def test_planner_evaluates_all_durations_after_first_recommendation(monkeypatch):
    dates = pd.date_range("2020-01-05", periods=120, freq="W-SUN")
    data = pd.DataFrame({"_date": dates, "_outcome": 100.0})
    calls = []

    def fake_fit(frame, intervention_date, *args, **kwargs):
        calls.append(kwargs.get("planner_mode"))
        post = frame.loc[frame["_date"] >= pd.Timestamp(intervention_date), ["_date", "_outcome"]].copy()
        post["counterfactual"] = 100.0
        rng = np.random.default_rng(kwargs.get("seed", 1))
        return {"model_stable": True, "result": post, "forecast_draws": rng.normal(100, .25, (kwargs.get("draws", 200), len(post)))}

    timeseries_module._PLANNER_ORIGIN_CACHE.clear()
    monkeypatch.setattr("src.experiment_platform.timeseries.fit_bsts_model", fake_fit)
    result = run_duration_power_simulation(data, "Weekly", [], "None", False, .95, .05, "Relative Lift (%)", maximum_duration=8, simulations=40, alpha=.10)
    assert result["recommended_duration"] == 4
    assert result["evaluated_durations"] == [4, 6, 8]
    assert result["skipped_durations"] == []
    assert result["stopped_early"] is False
    assert all(pd.notna(row["power"]) for row in result["power"])
    assert result["power"][0]["status"].startswith("Recommended")
    assert all(row["status"] == "Meets power target" for row in result["power"][1:])
    assert len(calls) == 24
    assert all(calls)


def test_planner_cache_reuses_origins_and_alpha_invalidates(monkeypatch):
    timeseries_module._PLANNER_ORIGIN_CACHE.clear()
    dates = pd.date_range("2020-01-05", periods=120, freq="W-SUN")
    data = pd.DataFrame({"_date": dates, "_outcome": 100.0})
    calls = []

    def fake_fit(frame, intervention_date, *args, **kwargs):
        calls.append(kwargs.get("planner_mode"))
        post = frame.loc[frame["_date"] >= pd.Timestamp(intervention_date), ["_date", "_outcome"]].copy()
        post["counterfactual"] = 100.0
        rng = np.random.default_rng(kwargs.get("seed", 1))
        return {"model_stable": True, "result": post, "forecast_draws": rng.normal(100, 1.0, (kwargs.get("draws", 200), len(post)))}

    monkeypatch.setattr("src.experiment_platform.timeseries.fit_bsts_model", fake_fit)
    args = (data, "Weekly", [], "None", False, .95, 0.0, "Relative Lift (%)")
    run_duration_power_simulation(*args, target_power=.8, maximum_duration=8, simulations=40, alpha=.10)
    first_call_count = len(calls)
    run_duration_power_simulation(*args, target_power=.8, maximum_duration=8, simulations=40, alpha=.10)
    assert first_call_count == 24
    assert len(calls) == first_call_count
    run_duration_power_simulation(*args, target_power=.8, maximum_duration=8, simulations=40, alpha=.05)
    assert len(calls) > first_call_count


def test_unknown_effect_returns_mde_without_claiming_a_forecast():
    dates = pd.date_range("2025-01-05", periods=12, freq="W-SUN")
    data = pd.DataFrame({"_date": dates, "_outcome": 100 + np.arange(12) + np.sin(np.arange(12))})
    result = run_duration_power_simulation(data, "Weekly", [], "Auto", True, .95, None, "Relative Lift (%)", maximum_duration=12)
    assert result["validation_mode"] == "Assumption-based"
    assert result["recommended_duration"] is None
    assert all(row["minimum_detectable_effect"] > 0 for row in result["power"])
    assert result["power"][-1]["minimum_detectable_effect"] < result["power"][0]["minimum_detectable_effect"]


def test_planner_mode_skips_rolling_backtest(monkeypatch):
    prepared = prepare_timeseries_data(time_series_campaign(), "date", "applications")
    monkeypatch.setattr("src.experiment_platform.timeseries._backtest", lambda *args, **kwargs: pytest.fail("planner mode should not run rolling backtests"))
    result = fit_bsts_model(prepared, "2025-07-06", "Weekly", planner_mode=True, draws=20)
    assert result["backtest"]["status"] == "Skipped in planner mode"


def test_operational_exposure_is_optional_and_not_time_series_sample_size():
    dates = pd.date_range("2025-01-05", periods=3, freq="W-SUN")
    data = pd.DataFrame({"_date": dates, "_outcome": [10, 20, 30], "customers": [100, 120, 140]})
    assert projected_operational_exposure(data, "None", 4) is None
    assert projected_operational_exposure(data, "customers", 4) == 480.0


def test_planning_chart_extends_through_recommendation_window():
    dates = pd.date_range("2025-01-05", periods=8, freq="W-SUN")
    chart = time_series_line(pd.DataFrame({"_date": dates, "_outcome": range(8)}), "", "Outcome", planned_launch_date="2025-03-09", recommended_end_date="2025-04-06")
    assert chart.layout.xaxis.range[1] > pd.Timestamp("2025-04-06")
    assert any(shape.x0 == pd.Timestamp("2025-03-09") for shape in chart.layout.shapes)


def test_reliability_treats_reasonable_error_and_moderate_coverage_as_fair():
    result = assess_model_reliability(
        {"status": "Pass", "mape": 0.041, "interval_coverage": 0.875, "bias": 40},
        0.95,
        0.03,
        104,
        10000,
    )
    assert result["components"]["forecast_error"] == "Good"
    assert result["components"]["forecast_range_coverage"] == "Fair"
    assert result["overall"] == "Fair"


def test_reliability_requires_multiple_major_failures_for_poor():
    result = assess_model_reliability(
        {"status": "Pass", "mape": 0.04, "interval_coverage": 0.87, "bias": 20},
        0.95,
        0.001,
        104,
        10000,
    )
    assert result["overall"] == "Fair"
