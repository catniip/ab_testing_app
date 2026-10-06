from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.structural import UnobservedComponents

from .core import infer_column, normalize_uploaded_dataset, safe_datetime_series, safe_numeric_series


@dataclass
class TimeSeriesConfig:
    data_source: str = "Upload File"
    dataset_name: str = ""
    date_column: str = "date"
    outcome_column: str = "applications"
    segment_column: str = "None"
    campaign_flag_column: str = "None"
    exposure_column: str = "None"
    frequency: str = "Auto"
    intervention_date: str = ""
    planned_launch_date: str = ""
    campaign_end_date: str = ""
    analysis_method: str = "Pre–Post Analysis"
    comparison_metric: str = "Average per period"
    pre_window: str = "All available pre-period"
    post_window: str = "All available post-period"
    comparison_window: str = "Symmetric"
    symmetric_periods: int = 12
    custom_pre_start: str = ""
    custom_pre_end: str = ""
    custom_post_start: str = ""
    custom_post_end: str = ""
    selected_covariates: list[str] = field(default_factory=list)
    seasonality: str = "Auto"
    prediction_interval: float = 0.95
    duplicate_policy: str = "Fail validation"
    missing_policy: str = "Keep"
    local_trend: bool = True
    simulation_seed: int = 123
    holdout_length: int = 8

    @property
    def credible_interval(self) -> float:
        """Backward-compatible alias for older Streamlit session state."""
        return self.prediction_interval


def infer_timeseries_columns(df: pd.DataFrame) -> dict[str, str]:
    columns = list(df.columns)
    date_col = infer_column(columns, ["date", "event_date", "observation_date", "week", "month", "timestamp"])
    numeric = [col for col in columns if safe_numeric_series(df, col).notna().sum() / max(len(df), 1) > 0.8]
    preferred = [col for col in numeric if _is_outcome_candidate(col)]
    outcome_col = infer_column(preferred, ["applications", "credit_card_applications", "outcome", "volume", "sales", "revenue", "conversions"], preferred[0] if preferred else "")
    segment_col = infer_column(columns, ["segment", "portfolio", "channel"], "None")
    flag_col = infer_column(columns, ["campaign_flag", "intervention_flag", "post_flag"], "None")
    return {"date": date_col, "outcome": outcome_col, "segment": segment_col, "campaign_flag": flag_col}


def _is_outcome_candidate(column: str) -> bool:
    name = str(column).lower()
    excluded = ("control", "true_effect", "true_impact", "post_treatment", "campaign_spend", "spend", "flag")
    return not any(token in name for token in excluded)


def humanize_column_name(column: str) -> str:
    """Convert technical column identifiers into business-readable labels."""
    value = str(column).strip().replace("_", " ")
    value = " ".join(value.split())
    return value[:1].upper() + value[1:] if value else "Outcome"


def incremental_outcome_label(column: str) -> str:
    label = humanize_column_name(column)
    if label.endswith("Balances"):
        label = label[:-1]
    return f"Incremental {label}"


def probability_positive_label(value: float) -> str:
    return ">99%" if value >= 0.995 else f"{value * 100:,.1f}%"


def build_business_interpretation(
    incremental_impact: float,
    relative_impact: float,
    interval: tuple[float, float],
    probability_positive: float,
    reliability: str,
    outcome_name: str,
    intervention_type: str = "campaign",
) -> dict[str, str]:
    """Combine effect evidence and model reliability into plain-language guidance."""
    low, high = interval
    if low <= 0 <= high:
        status = "Inconclusive"
        headline = "Evidence of incremental impact is inconclusive."
        interpretation = f"Although {humanize_column_name(outcome_name).lower()} may have changed after launch, the model cannot separate the {intervention_type} effect from expected underlying change."
        next_step = "Review the counterfactual specification and continue monitoring before making a high-stakes decision."
    elif incremental_impact > 0:
        if reliability == "Good":
            status = "Positive — Strong Evidence"
            headline = "Strong evidence of positive incremental impact."
            interpretation = f"The model estimates that the {intervention_type} added approximately {abs(relative_impact) * 100:.1f}% to {humanize_column_name(outcome_name).lower()} beyond the expected trajectory."
            next_step = "Use this result alongside business cost and benefit considerations."
        elif reliability == "Fair":
            status = "Positive — Directional"
            headline = "Estimated impact is positive, with moderate confidence in the counterfactual."
            interpretation = "The estimated effect is positive, but historical validation leaves some uncertainty about the expected trajectory."
            next_step = "Review diagnostics and validate the result against business context before scaling."
        else:
            status = "Positive — Directional"
            headline = "Estimated impact is positive, but model reliability is low."
            interpretation = "The model points to a positive effect, but unexplained time patterns make the estimate directional rather than definitive."
            next_step = "Review diagnostics, improve the counterfactual specification, or add stronger unaffected predictors."
    else:
        status = "Negative — Strong Evidence" if reliability == "Good" else "Negative — Directional"
        headline = "Strong evidence of negative incremental impact." if reliability == "Good" else "Estimated impact is negative, but model reliability is limited."
        interpretation = "The model estimates that the intervention reduced the outcome relative to the expected trajectory."
        next_step = "Review diagnostics and investigate the intervention before making a rollout decision."
    return {"status": status, "headline": headline, "interpretation": interpretation, "next_step": next_step, "probability_label": probability_positive_label(probability_positive)}


def predictor_candidates(df: pd.DataFrame, outcome_column: str, date_column: str = "", campaign_flag_column: str = "") -> list[str]:
    excluded = {outcome_column, "_outcome", date_column, "_date", campaign_flag_column, "exposure", "customer_count", "customers", "campaign_spend", "spend", "true_effect", "true_impact", "post_treatment"}
    excluded_tokens = ("campaign_spend", "spend", "true_effect", "true_impact", "post_treatment", "campaign_flag", "intervention_flag", "exposure", "customer_count", "customers")
    return [col for col in df.columns if col not in excluded and not any(token in str(col).lower() for token in excluded_tokens) and safe_numeric_series(df, col).notna().sum() / max(len(df), 1) > 0.8]


def assess_model_reliability(
    backtest: dict,
    interval_level: float,
    residual_pvalue: float | None,
    pre_observations: int,
    outcome_scale: float,
    stable: bool = True,
) -> dict:
    """Summarize model validation with transparent product heuristics.

    These Good/Fair/Poor thresholds are product diagnostics, not universal
    statistical standards. A single moderate issue should not make a model
    Poor; that rating requires multiple major problems or a clear failure.
    """
    def grade(value: float | None, good: float, fair: float, lower_is_better: bool = False) -> str:
        if value is None or not np.isfinite(value):
            return "Poor"
        if lower_is_better:
            return "Good" if value <= good else "Fair" if value <= fair else "Poor"
        return "Good" if value >= good else "Fair" if value >= fair else "Poor"

    mape = backtest.get("mape")
    coverage = backtest.get("interval_coverage")
    bias = abs(float(backtest.get("bias", np.nan))) / max(abs(float(outcome_scale)), 1e-9)
    coverage_good = max(0.0, interval_level - 0.05)
    coverage_fair = max(0.0, interval_level - 0.15)
    components = {
        "forecast_error": grade(mape, 0.05, 0.10, lower_is_better=True),
        "forecast_range_coverage": grade(coverage, coverage_good, coverage_fair),
        "time_pattern_fit": "Good" if residual_pvalue is None or residual_pvalue >= 0.05 else "Fair" if residual_pvalue >= 0.01 else "Poor",
        "forecast_bias": grade(bias, 0.02, 0.05, lower_is_better=True),
        "data_sufficiency": "Good" if pre_observations >= 52 else "Fair" if pre_observations >= 30 else "Poor",
        "holdout_validation": "Good" if backtest.get("status") == "Pass" else "Poor",
        "model_stability": "Good" if stable else "Poor",
    }
    poor_count = sum(value == "Poor" for value in components.values())
    fair_count = sum(value == "Fair" for value in components.values())
    overall = "Poor" if poor_count >= 2 or components["holdout_validation"] == "Poor" or not stable else "Fair" if poor_count == 1 or fair_count >= 1 else "Good"
    explanations = {
        "forecast_error": "Forecast error is low" if components["forecast_error"] == "Good" else "Forecast error is higher than preferred",
        "forecast_range_coverage": "Forecast range coverage is on target" if components["forecast_range_coverage"] == "Good" else "Forecast range coverage is below target",
        "time_pattern_fit": "Time patterns are mostly captured" if components["time_pattern_fit"] == "Good" else "Some time patterns remain unexplained",
        "forecast_bias": "Forecast bias is small" if components["forecast_bias"] == "Good" else "Forecast bias needs review",
        "data_sufficiency": "Historical data is sufficient" if components["data_sufficiency"] == "Good" else "Historical data is limited",
        "holdout_validation": "Historical holdout validation passed" if components["holdout_validation"] == "Good" else "Historical holdout validation did not pass",
        "model_stability": "The model fit was stable" if components["model_stability"] == "Good" else "The model fit was not stable",
    }
    return {"overall": overall, "components": components, "explanations": explanations, "residual_pvalue": residual_pvalue}


def planning_durations(frequency: str, maximum: int) -> list[int]:
    grids = {"Daily": [7, 14, 21, 28, 42, 56, 84], "Weekly": [4, 6, 8, 12, 16, 20, 26], "Monthly": [2, 3, 4, 6, 9, 12]}
    return [duration for duration in grids.get(frequency, grids["Weekly"]) if duration <= maximum]


def campaign_decision_date(launch_date: str | pd.Timestamp, duration: int, frequency: str) -> pd.Timestamp:
    """Return the first date after the requested post-launch observation window."""
    launch = pd.Timestamp(launch_date)
    if frequency == "Daily":
        return launch + pd.Timedelta(days=duration)
    if frequency == "Weekly":
        return launch + pd.Timedelta(weeks=duration)
    if frequency == "Monthly":
        return launch + pd.DateOffset(months=duration)
    raise ValueError("Campaign decision dates require Daily, Weekly, or Monthly frequency.")


def duration_power_status(duration: int, power: float, target_power: float, recommended_duration: int | None, result_status: str | None = None) -> str:
    """Classify each candidate independently; power need not be monotonic."""
    if result_status:
        return result_status
    if not np.isfinite(power) or power < target_power:
        return "Below target"
    return "Recommended" if duration == recommended_duration else "Above target"


def classify_calibration(false_positive_rate: float, alpha: float = 0.05, min_simulations: int = 12, simulations: int | None = None) -> str:
    """Classify empirical placebo calibration relative to the configured alpha."""
    if simulations is not None and simulations < min_simulations:
        return "poor"
    ratio = false_positive_rate / max(alpha, 1e-9)
    return "acceptable" if ratio <= 1.5 else "caution" if ratio <= 2.5 else "poor"


def classify_historical_calibration(false_positives: int, trials: int, alpha: float = 0.05, min_origins: int = 3) -> tuple[str, float]:
    """Evaluate sparse historical false alarms with an exact binomial test."""
    if trials < min_origins:
        return "poor", np.nan
    pvalue = float(stats.binomtest(false_positives, trials, alpha, alternative="greater").pvalue)
    status = "poor" if pvalue < 0.05 else "caution" if pvalue < 0.10 else "acceptable"
    return status, pvalue


def power_target_met(power: float, interval: tuple[float, float], target_power: float) -> bool:
    """Require the lower simulation bound to clear target when available."""
    if not np.isfinite(power):
        return False
    low = interval[0] if interval else np.nan
    return bool(low >= target_power) if np.isfinite(low) else bool(power >= target_power)


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return a binomial Wilson interval without relying on a normal approximation."""
    if trials <= 0:
        return (np.nan, np.nan)
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
    return (float(max(0.0, centre - margin)), float(min(1.0, centre + margin)))


def inject_effect(baseline: pd.Series, effect: float, effect_mode: str = "relative", effect_pattern: str = "constant", ramp_period: int = 1) -> pd.Series:
    """Inject a planned effect without changing the historical time structure."""
    steps = np.arange(len(baseline))
    scale = np.minimum((steps + 1) / max(1, ramp_period), 1.0) if effect_pattern == "gradual_ramp" else np.ones(len(baseline))
    return baseline.astype(float) * (1 + effect * scale) if effect_mode == "relative" else baseline.astype(float) + effect * scale


def _duration_test_statistic(fit: dict, data: pd.DataFrame, origin: int, duration: int, observed_override: np.ndarray | None = None) -> float:
    """Calculate a duration statistic from one reusable counterfactual fit."""
    post = data.iloc[origin : origin + duration]
    observed = observed_override if observed_override is not None else post["_outcome"].to_numpy(dtype=float)
    predicted = fit["result"]["counterfactual"].to_numpy(dtype=float)[:duration]
    draws = np.asarray(fit["forecast_draws"], dtype=float)[:, :duration]
    cumulative_draws = (observed[None, :] - draws).sum(axis=1)
    standard_error = max(float(np.std(cumulative_draws, ddof=1)), 1e-9)
    return float(np.sum(observed - predicted) / standard_error)


_PLANNER_ORIGIN_CACHE: dict[tuple, dict] = {}


def _planner_origin_cache_key(data: pd.DataFrame, origin: int, frequency: str, predictors: list[str], seasonality: str, local_trend: bool, alpha: float, seed: int, horizon: int) -> tuple:
    """Identify a planner fit by every setting that can change its forecast."""
    return (data_fingerprint(data), int(origin), frequency, tuple(predictors), seasonality, bool(local_trend), float(alpha), int(seed), int(horizon))


def _planner_validation_design(observations: int, frequency: str, seasonality: str, maximum_duration: int) -> dict:
    """Choose comparable pseudo-origins and an evidence tier for available history."""
    latest_origin = observations - maximum_duration
    if latest_origin < 20:
        return {"mode": "Assumption-based", "origins": [], "seasonality": "None", "minimum_training": 0}
    period = {"Daily": 7, "Weekly": 52, "Monthly": 12}.get(frequency) if seasonality == "Auto" else _seasonal_period(frequency, seasonality, 10_000)
    seasonal_minimum = 2 * period if period else 0
    if seasonal_minimum and latest_origin >= seasonal_minimum and latest_origin - seasonal_minimum + 1 >= 6:
        minimum_training = seasonal_minimum
        effective_seasonality = seasonality
    else:
        minimum_training = max(20, min(latest_origin, observations // 2))
        effective_seasonality = "None"
    eligible = list(range(minimum_training, latest_origin + 1))
    mode = "Robust empirical" if len(eligible) >= 32 else "Limited-history hybrid" if len(eligible) >= 12 else "Preliminary hybrid" if len(eligible) >= 6 else "Assumption-based"
    target_origins = {"Robust empirical": 24, "Limited-history hybrid": 12, "Preliminary hybrid": 6}.get(mode, 0)
    if target_origins and len(eligible) > target_origins:
        positions = np.linspace(0, len(eligible) - 1, target_origins).round().astype(int)
        eligible = [eligible[index] for index in sorted(set(positions.tolist()))]
    return {"mode": mode, "origins": eligible if mode != "Assumption-based" else [], "seasonality": effective_seasonality, "minimum_training": minimum_training}


def _planner_path_components(fit: dict, duration: int, effect_type: str, effect_pattern: str, ramp_period: int) -> tuple[np.ndarray, np.ndarray, float]:
    predicted = fit["result"]["counterfactual"].to_numpy(dtype=float)[:duration]
    observed = fit["result"]["_outcome"].to_numpy(dtype=float)[:duration]
    draws = np.asarray(fit["forecast_draws"], dtype=float)[:, :duration]
    forecast_totals = draws.sum(axis=1)
    standard_error = max(float(np.std(forecast_totals, ddof=1)), 1e-9)
    null_statistics = (forecast_totals - predicted.sum()) / standard_error
    steps = np.arange(duration)
    scale = np.minimum((steps + 1) / max(1, ramp_period), 1.0) if effect_pattern == "gradual_ramp" else np.ones(duration)
    if effect_type == "Relative Lift (%)":
        effect_sensitivity = (draws * scale[None, :]).sum(axis=1) / standard_error
    else:
        effect_sensitivity = np.full(len(draws), scale.sum() / standard_error)
    observed_statistic = float(np.sum(observed - predicted) / standard_error)
    return null_statistics, effect_sensitivity, observed_statistic


def _minimum_detectable_effect(components: list[tuple[np.ndarray, np.ndarray, float]], threshold: float, target_power: float, effect_type: str) -> float:
    if not components or not np.isfinite(threshold):
        return np.nan

    def estimated_power(candidate: float) -> float:
        rates = [float(np.mean(null + candidate * sensitivity > threshold)) for null, sensitivity, _ in components]
        return float(np.mean(rates))

    high = 0.01 if effect_type == "Relative Lift (%)" else 1.0
    limit = 5.0 if effect_type == "Relative Lift (%)" else 1e9
    while high < limit and estimated_power(high) < target_power:
        high *= 2
    if estimated_power(high) < target_power:
        return np.nan
    low = 0.0
    for _ in range(32):
        midpoint = (low + high) / 2
        if estimated_power(midpoint) >= target_power:
            high = midpoint
        else:
            low = midpoint
    return float(high)


def _analytical_duration_plan(data: pd.DataFrame, durations: list[int], effect: float | None, effect_type: str, target_power: float, alpha: float, ramp_period: int, effect_pattern: str) -> dict:
    """Provide a clearly labeled approximation when empirical validation is impossible."""
    outcome = data["_outcome"].astype(float).to_numpy()
    x = np.arange(len(outcome), dtype=float)
    fitted = np.polyval(np.polyfit(x, outcome, 1), x) if len(outcome) >= 4 else np.repeat(np.mean(outcome), len(outcome))
    residuals = outcome - fitted
    sigma = max(float(np.std(residuals, ddof=min(2, max(1, len(residuals) - 1)))), 1e-9)
    rho = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1]) if len(residuals) >= 6 and np.std(residuals[:-1]) > 0 and np.std(residuals[1:]) > 0 else 0.0
    rho = float(np.clip(rho, -0.5, 0.8))
    baseline = max(abs(float(np.mean(outcome))), 1e-9)
    critical = float(stats.norm.ppf(1 - alpha))
    target_z = float(stats.norm.ppf(target_power))
    rows = []
    recommended = None
    evaluated = []
    for duration in durations:
        scale = np.minimum((np.arange(duration) + 1) / max(1, ramp_period), 1.0) if effect_pattern == "gradual_ramp" else np.ones(duration)
        variance_multiplier = duration + 2 * sum((duration - lag) * rho**lag for lag in range(1, duration))
        cumulative_se = sigma * np.sqrt(max(variance_multiplier, 1e-9))
        signal_per_unit = baseline * scale.sum() if effect_type == "Relative Lift (%)" else scale.sum()
        mde = (critical + target_z) * cumulative_se / max(signal_per_unit, 1e-9)
        power = float(stats.norm.cdf((effect or 0.0) * signal_per_unit / cumulative_se - critical)) if effect is not None else np.nan
        row_status = "Below power target"
        if effect is not None and power >= target_power:
            if recommended is None:
                recommended = duration
                row_status = "Recommended (assumption-based)"
            else:
                row_status = "Meets power target"
        rows.append({"duration": duration, "power": power, "power_ci": (np.nan, np.nan), "false_positive_rate": alpha, "false_positive_ci": (np.nan, np.nan), "historical_false_positive_rate": np.nan, "historical_false_positive_ci": (np.nan, np.nan), "minimum_detectable_effect": mde, "calibration_status": "assumption_based", "status": row_status, "successful_fits": 0, "valid_placebos": 0, "failed_fits": 0, "detections": 0, "empirical_critical_value": critical})
        evaluated.append(duration)
    return {"status": "ok", "power": rows, "false_positive_rate": alpha, "calibration_status": "assumption_based", "recommendation_confidence": "Low", "recommended_duration": recommended, "validation_mode": "Assumption-based", "model_seasonality": "None", "historical_origins": 0, "minimum_training_observations": 0, "evaluated_durations": evaluated, "skipped_durations": [], "stopped_early": False, "assumption_note": "History is too short for independent pseudo-launch validation. Power uses detrended residual variance and an AR(1) approximation."}


def run_duration_power_simulation(data: pd.DataFrame, frequency: str, covariates: list[str], seasonality: str, local_trend: bool, interval: float, effect: float | None, effect_type: str, target_power: float = 0.8, maximum_duration: int = 26, ramp_period: int = 4, simulations: int = 20, seed: int = 123, alpha: float = 0.05, effect_pattern: str = "constant") -> dict:
    """Estimate duration or MDE with adaptive cross-fitted validation."""
    durations = planning_durations(frequency, maximum_duration)
    data = data.sort_values("_date").reset_index(drop=True)
    dates = pd.to_datetime(data["_date"]).sort_values().reset_index(drop=True)
    if not durations:
        return {"status": "insufficient_history", "power": [], "false_positive_rate": np.nan, "recommended_duration": None}
    maximum_duration = max(durations)
    design = _planner_validation_design(len(data), frequency, seasonality, maximum_duration)
    if design["mode"] == "Assumption-based":
        result = _analytical_duration_plan(data, durations, effect, effect_type, target_power, alpha, ramp_period, effect_pattern)
        result.update({"simulations": simulations, "confidence_level": 1 - alpha, "min_calibration_simulations": 0, "min_evaluation_simulations": 0})
        return result
    rng = np.random.default_rng(seed)
    confidence_level = 1 - alpha
    selected = design["origins"]
    calibration_origins = selected[::2]
    evaluation_origins = selected[1::2]
    min_calibration_simulations = len(calibration_origins)
    min_evaluation_simulations = len(evaluation_origins)
    origin_cache = {}
    for index in selected:
        pseudo_date = dates.iloc[index].isoformat()
        window = data.iloc[: index + maximum_duration].copy()
        origin_seed = int(rng.integers(1, 1_000_000))
        cache_key = _planner_origin_cache_key(window, index, frequency, covariates, design["seasonality"], local_trend, alpha, origin_seed, maximum_duration)
        if cache_key in _PLANNER_ORIGIN_CACHE:
            origin_cache[index] = _PLANNER_ORIGIN_CACHE[cache_key]
            continue
        try:
            origin_cache[index] = fit_bsts_model(window, pseudo_date, frequency, covariates, confidence_level, draws=max(200, int(simulations) * 5), seasonality=design["seasonality"], local_trend=local_trend, seed=origin_seed, holdout_length=8, planner_mode=True)
            _PLANNER_ORIGIN_CACHE[cache_key] = origin_cache[index]
        except Exception:
            origin_cache[index] = None
    power_rows = []
    total_false_positives = 0
    total_placebos = 0
    recommended = None
    evaluated_durations = []
    for duration in durations:
        calibration_components = []
        evaluation_components = []
        failed_fits = 0
        for index in calibration_origins:
            fit = origin_cache.get(index)
            if fit is None or not fit.get("model_stable", False):
                failed_fits += 1
                continue
            calibration_components.append(_planner_path_components(fit, duration, effect_type, effect_pattern, ramp_period))
        calibration_null = np.concatenate([component[0] for component in calibration_components]) if calibration_components else np.array([])
        threshold = float(np.quantile(calibration_null, confidence_level)) if len(calibration_null) else np.nan
        for index in evaluation_origins:
            fit = origin_cache.get(index)
            if fit is None or not fit.get("model_stable", False):
                failed_fits += 1
                continue
            evaluation_components.append(_planner_path_components(fit, duration, effect_type, effect_pattern, ramp_period))
        placebo_stats = np.concatenate([component[0] for component in evaluation_components]) if evaluation_components else np.array([])
        effect_stats = np.concatenate([component[0] + effect * component[1] for component in evaluation_components]) if evaluation_components and effect is not None else np.array([])
        historical_stats = [component[2] for component in evaluation_components]
        false_positives = int(np.sum(placebo_stats > threshold)) if len(placebo_stats) and np.isfinite(threshold) else 0
        detections = int(np.sum(effect_stats > threshold)) if len(effect_stats) and np.isfinite(threshold) else 0
        historical_false_positives = int(sum(value > threshold for value in historical_stats)) if np.isfinite(threshold) else 0
        placebo_rate = false_positives / len(placebo_stats) if len(placebo_stats) and np.isfinite(threshold) else np.nan
        historical_placebo_rate = historical_false_positives / len(historical_stats) if historical_stats and np.isfinite(threshold) else np.nan
        power = detections / len(effect_stats) if len(effect_stats) and np.isfinite(threshold) else np.nan
        power_ci = wilson_interval(detections, len(effect_stats)) if len(effect_stats) and np.isfinite(threshold) else (np.nan, np.nan)
        placebo_ci = wilson_interval(false_positives, len(placebo_stats)) if len(placebo_stats) and np.isfinite(threshold) else (np.nan, np.nan)
        historical_placebo_ci = wilson_interval(historical_false_positives, len(historical_stats)) if historical_stats and np.isfinite(threshold) else (np.nan, np.nan)
        total_false_positives += false_positives
        total_placebos += len(placebo_stats)
        model_calibration = classify_calibration(placebo_rate, alpha, simulations=len(placebo_stats)) if np.isfinite(placebo_rate) else "poor"
        historical_calibration, historical_calibration_pvalue = classify_historical_calibration(historical_false_positives, len(historical_stats), alpha) if np.isfinite(historical_placebo_rate) else ("poor", np.nan)
        calibration_status = "acceptable" if model_calibration == "acceptable" and historical_calibration == "acceptable" else "caution" if model_calibration != "poor" and historical_calibration != "poor" else "poor"
        mde = _minimum_detectable_effect(evaluation_components, threshold, target_power, effect_type)
        recommendation_ready = power_target_met(power, power_ci, target_power)
        row = {"duration": duration, "power": power, "power_ci": power_ci, "eligible_windows": len(selected), "attempted_simulations": len(selected), "calibration_simulations": len(calibration_components), "successful_fits": len(evaluation_components), "valid_placebos": len(historical_stats), "simulation_paths": len(placebo_stats), "failed_fits": failed_fits, "detections": detections, "false_positive_rate": placebo_rate, "historical_false_positives": historical_false_positives, "historical_calibration_pvalue": historical_calibration_pvalue, "false_positive_ci": placebo_ci, "historical_false_positive_rate": historical_placebo_rate, "historical_false_positive_ci": historical_placebo_ci, "empirical_critical_value": threshold, "minimum_detectable_effect": mde, "calibration_status": calibration_status, "status": "Below power target" if effect is not None else "MDE estimated"}
        if effect is not None and np.isfinite(power) and power >= target_power and not recommendation_ready:
            row["status"] = "Near power target; simulation uncertainty overlaps target"
        elif effect is not None and calibration_status != "acceptable" and recommendation_ready:
            row["status"] = "Power target met, calibration failed"
        elif effect is not None and calibration_status == "acceptable" and recommendation_ready and failed_fits / max(len(selected), 1) <= 0.25:
            if recommended is None:
                row["status"] = "Recommended" if design["mode"] == "Robust empirical" else "Recommended with limited-history validation"
                recommended = duration
            else:
                row["status"] = "Meets power target"
        elif not np.isfinite(threshold) or not evaluation_components:
            row["status"] = "Insufficient simulations"
        power_rows.append(row)
        evaluated_durations.append(duration)
    skipped_durations = []
    false_positive_rate = total_false_positives / total_placebos if total_placebos else np.nan
    evaluated_calibrations = [row["calibration_status"] for row in power_rows if row["calibration_status"] != "not_evaluated"]
    recommended_row = next((row for row in power_rows if row["duration"] == recommended), None)
    calibration_status = recommended_row["calibration_status"] if recommended_row else "poor" if "poor" in evaluated_calibrations else "caution" if "caution" in evaluated_calibrations else "acceptable"
    status = "ok" if evaluated_durations else "insufficient_calibration"
    base_confidence = {"Robust empirical": "High", "Limited-history hybrid": "Medium", "Preliminary hybrid": "Low"}[design["mode"]]
    recommendation_confidence = "Low" if calibration_status == "poor" else "Medium" if calibration_status == "caution" or base_confidence == "Medium" else base_confidence
    return {"status": status, "power": power_rows, "false_positive_rate": false_positive_rate, "calibration_by_duration": [{"duration": row["duration"], "false_positive_rate": row["false_positive_rate"], "false_positive_ci": row["false_positive_ci"], "historical_false_positive_rate": row["historical_false_positive_rate"], "historical_false_positive_ci": row["historical_false_positive_ci"], "historical_false_positives": row["historical_false_positives"], "historical_calibration_pvalue": row["historical_calibration_pvalue"], "valid_placebos": row["valid_placebos"], "empirical_critical_value": row["empirical_critical_value"]} for row in power_rows], "calibration_status": calibration_status, "recommendation_confidence": recommendation_confidence, "recommended_duration": recommended, "simulations": simulations, "min_calibration_simulations": min_calibration_simulations, "min_evaluation_simulations": min_evaluation_simulations, "confidence_level": confidence_level, "validation_mode": design["mode"], "model_seasonality": design["seasonality"], "historical_origins": len(selected), "minimum_training_observations": design["minimum_training"], "evaluated_durations": evaluated_durations, "skipped_durations": skipped_durations, "stopped_early": False}


def _frequency_spec(frequency: str, dates: pd.Series | None = None) -> tuple[str, int | None]:
    if frequency == "Daily":
        return "D", 7
    if frequency == "Weekly":
        return "W-SUN", 52
    if frequency == "Monthly":
        return "MS", 12
    if dates is not None:
        detected, _ = detect_frequency(pd.DataFrame({"_date": dates}))
        return _frequency_spec(detected)
    return "", None


def prepare_timeseries_data(
    df: pd.DataFrame,
    date_col: str,
    outcome_col: str,
    duplicate_policy: str = "Fail validation",
    missing_policy: str = "Keep",
    frequency: str = "Auto",
) -> pd.DataFrame:
    """Parse, aggregate, and optionally regularize a time series.

    Duplicate timestamps are aggregated before modeling. Missing periods are
    either retained, filled, or rejected according to the explicit policy.
    """
    df = normalize_uploaded_dataset(df)
    if date_col not in df.columns:
        raise ValueError("Date Column mapping is required.")
    if outcome_col not in df.columns:
        raise ValueError("Outcome Column mapping is required.")
    out = df.copy()
    out["_date"] = safe_datetime_series(out, date_col)
    out["_outcome"] = safe_numeric_series(out, outcome_col)
    out = out.drop(columns=[date_col, outcome_col], errors="ignore")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["_date", "_outcome"]).sort_values("_date")
    if out.empty:
        raise ValueError("No finite observations remain after parsing Date and Outcome columns.")
    if duplicate_policy == "Fail validation" and out.duplicated("_date").any():
        raise ValueError(f"{int(out.duplicated('_date').sum())} duplicate timestamps found. Choose Average or Sum explicitly.")
    if duplicate_policy not in {"Average", "Sum", "Fail validation"}:
        raise ValueError("Duplicate policy must be Fail validation, Average, or Sum.")
    agg = "mean" if duplicate_policy in {"Average", "Fail validation"} else "sum"
    aggregations = {
        col: (agg if col == "_outcome" else "mean" if pd.api.types.is_numeric_dtype(out[col]) else "first")
        for col in out.columns if col not in {"_date"}
    }
    out = out.groupby("_date", as_index=False).agg(aggregations).sort_values("_date")
    detected, _ = detect_frequency(out)
    selected_frequency = detected if frequency in {"Auto", ""} else frequency
    if selected_frequency not in {"Daily", "Weekly", "Monthly"}:
        raise ValueError("The time series must be Daily, Weekly, or Monthly.")
    freq_spec, _ = _frequency_spec(selected_frequency)
    expected = pd.date_range(out["_date"].min(), out["_date"].max(), freq=freq_spec)
    missing_periods = int(len(expected.difference(pd.DatetimeIndex(out["_date"]))))
    regularized = regularize_timeseries_data(out, selected_frequency, missing_policy)
    regularized.attrs["frequency"] = selected_frequency
    regularized.attrs["duplicate_policy"] = duplicate_policy
    regularized.attrs["missing_policy"] = missing_policy
    regularized.attrs["missing_periods"] = missing_periods
    return regularized


def regularize_timeseries_data(data: pd.DataFrame, frequency: str, missing_policy: str = "Keep") -> pd.DataFrame:
    """Reindex to a regular frequency; never silently invent observations."""
    if missing_policy not in {"Keep", "Fail validation", "Interpolate", "Fill missing counts with zero"}:
        raise ValueError("Unknown missing-period policy.")
    if frequency not in {"Daily", "Weekly", "Monthly"}:
        raise ValueError("Regularization requires Daily, Weekly, or Monthly frequency.")
    freq, _ = _frequency_spec(frequency)
    values = data.set_index("_date")["_outcome"].sort_index()
    expected = pd.date_range(values.index.min(), values.index.max(), freq=freq)
    out = data.set_index("_date").reindex(expected).rename_axis("_date").reset_index()
    if missing_policy == "Fail validation" and out["_outcome"].isna().any():
        raise ValueError(f"{int(out['_outcome'].isna().sum())} periods are missing from the regular time series.")
    if missing_policy == "Interpolate":
        out["_outcome"] = out["_outcome"].interpolate(limit_area="inside")
        if out["_outcome"].isna().any():
            raise ValueError("Interpolation cannot fill missing periods at the series boundary.")
    elif missing_policy == "Fill missing counts with zero":
        out["_outcome"] = out["_outcome"].fillna(0.0)
    # Keep covariates missing after regularization. Filling them here can use
    # values from after an intervention; model-specific preprocessing decides
    # whether a covariate is usable inside each temporal split.
    return out


def detect_frequency(data: pd.DataFrame) -> tuple[str, int]:
    dates = pd.to_datetime(data["_date"], errors="coerce").dropna().drop_duplicates().sort_values()
    if len(dates) < 3:
        return "Unknown", 0
    deltas = dates.diff().dropna().dt.days
    median = float(deltas.median())
    if median <= 2:
        freq, step = "Daily", "D"
    elif median <= 10:
        freq, step = "Weekly", "W-SUN"
    elif median <= 45:
        freq, step = "Monthly", "MS"
    else:
        return "Irregular", 0
    expected = pd.date_range(dates.min(), dates.max(), freq=step)
    missing = len(expected.difference(pd.DatetimeIndex(dates)))
    return freq, int(missing)


def duplicate_timestamp_count(data: pd.DataFrame) -> int:
    return int(data.duplicated("_date").sum())


def build_timeseries_preview(
    data: pd.DataFrame,
    date_col: str,
    outcome_col: str,
    campaign_flag_col: str = "None",
    duplicate_policy: str = "Fail validation",
    missing_policy: str = "Keep",
) -> dict:
    """Build read-only dataset context for the Time Series opening screen."""
    prepared = prepare_timeseries_data(data, date_col, outcome_col, duplicate_policy, missing_policy, "Auto")
    frequency, missing = detect_frequency(prepared)
    launch = None
    if campaign_flag_col != "None" and campaign_flag_col in data:
        source_dates = safe_datetime_series(data, date_col)
        flags = safe_numeric_series(data, campaign_flag_col).fillna(0)
        launched_dates = source_dates[flags > 0].dropna()
        launch = launched_dates.min() if not launched_dates.empty else None
    pre_count = int((prepared["_date"] < launch).sum()) if launch is not None else len(prepared)
    post_count = int((prepared["_date"] >= launch).sum()) if launch is not None else 0
    predictors = predictor_candidates(prepared, outcome_col, date_col, campaign_flag_col)
    return {
        "prepared": prepared,
        "frequency": frequency,
        "missing_periods": int(prepared.attrs.get("missing_periods", missing)),
        "start_date": prepared["_date"].min(),
        "end_date": prepared["_date"].max(),
        "observations": len(prepared),
        "campaign_launch": launch,
        "pre_observations": pre_count,
        "post_observations": post_count,
        "predictors": predictors,
    }


def intervention_periods(data: pd.DataFrame, intervention_date: str, campaign_end_date: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    launch = pd.Timestamp(intervention_date)
    end = pd.Timestamp(campaign_end_date) if campaign_end_date else data["_date"].max()
    if end < launch:
        raise ValueError("Campaign End Date must be on or after Campaign Launch.")
    pre = data[data["_date"] < launch].copy()
    post = data[(data["_date"] >= launch) & (data["_date"] <= end)].copy()
    return pre, post


def _select_windows(pre: pd.DataFrame, post: pd.DataFrame, comparison_metric: str, pre_window: str, post_window: str, comparison_window: str = "", symmetric_periods: int = 12, custom_ranges: tuple[str, str, str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if comparison_window == "Custom date range" and custom_ranges:
        pre_start, pre_end, post_start, post_end = custom_ranges
        if pre_start and pre_end:
            pre = pre[(pre["_date"] >= pd.Timestamp(pre_start)) & (pre["_date"] <= pd.Timestamp(pre_end))]
        if post_start and post_end:
            post = post[(post["_date"] >= pd.Timestamp(post_start)) & (post["_date"] <= pd.Timestamp(post_end))]
        return pre, post
    if comparison_window == "Symmetric":
        post = post.head(min(len(post), max(1, symmetric_periods))).copy()
        pre = pre.tail(min(len(pre), len(post))).copy()
        return pre, post
    if post_window == "First N post-periods":
        post = post.head(min(len(post), 12)).copy()
    if pre_window == "Last N pre-periods":
        pre = pre.tail(min(len(pre), len(post))).copy()
    elif comparison_metric == "Total" or pre_window == "Matching-length pre-period":
        pre = pre.tail(min(len(pre), len(post))).copy()
    return pre, post


def run_pre_post_analysis(
    data: pd.DataFrame,
    intervention_date: str,
    campaign_end_date: str = "",
    comparison_metric: str = "Average per period",
    pre_window: str = "All available pre-period",
    post_window: str = "All available post-period",
    comparison_window: str = "",
    symmetric_periods: int = 12,
    custom_ranges: tuple[str, str, str, str] | None = None,
) -> dict:
    pre, post = intervention_periods(data, intervention_date, campaign_end_date)
    pre, post = _select_windows(pre, post, comparison_metric, pre_window, post_window, comparison_window, symmetric_periods, custom_ranges)
    if pre.empty:
        raise ValueError("Pre period is empty. Select a later Campaign Launch date.")
    if len(post) < 1:
        raise ValueError("Post period is empty. Select an earlier Campaign Launch date.")
    if comparison_metric == "Total":
        pre_value, post_value = float(pre["_outcome"].sum()), float(post["_outcome"].sum())
    else:
        pre_value, post_value = float(pre["_outcome"].mean()), float(post["_outcome"].mean())
    change = post_value - pre_value
    return {
        "pre_observations": len(pre), "post_observations": len(post),
        "pre_start": pre["_date"].min(), "pre_end": pre["_date"].max(),
        "post_start": post["_date"].min(), "post_end": post["_date"].max(),
        "pre_mean": float(pre["_outcome"].mean()), "post_mean": float(post["_outcome"].mean()),
        "pre_total": float(pre["_outcome"].sum()), "post_total": float(post["_outcome"].sum()),
        "pre_value": pre_value, "post_value": post_value,
        "absolute_change": change, "percent_change": change / pre_value if pre_value else np.nan,
        "comparison_metric": comparison_metric, "causal": False,
    }


def _seasonal_period(frequency: str, seasonality: str, n_pre: int) -> int | None:
    if seasonality == "None":
        return None
    defaults = {"Daily": 7, "Weekly": 52, "Monthly": 12}
    annual = {"Daily": 365, "Weekly": 52, "Monthly": 12}
    if seasonality == "Weekly":
        period = 7 if frequency == "Daily" else 52 if frequency == "Weekly" else None
    elif seasonality in {"Monthly", "Annual"}:
        period = 12 if frequency == "Monthly" else annual.get(frequency)
    else:
        period = defaults.get(frequency)
    return period if period and n_pre >= period * 2 else None


def _build_exog(data: pd.DataFrame, covariates: list[str], index: pd.Index) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    warnings = []
    valid = []
    out = pd.DataFrame(index=index)
    for col in covariates:
        if col not in data.columns:
            warnings.append(f"Covariate '{col}' is not present and was excluded.")
            continue
        values = safe_numeric_series(data, col).replace([np.inf, -np.inf], np.nan)
        if values.loc[index].isna().any() or values.loc[index].nunique(dropna=True) < 2:
            warnings.append(f"Covariate '{col}' is incomplete or constant and was excluded.")
            continue
        selected = values.loc[index]
        if selected.isna().any():
            warnings.append(f"Covariate '{col}' is incomplete in the requested period and was excluded.")
            continue
        out[col] = selected
        valid.append(col)
    return (out[valid] if valid else None), valid, warnings


def _fit_structural(pre: pd.DataFrame, post: pd.DataFrame, frequency: str, seasonality: str, covariates: list[str], local_trend: bool, interval: float = 0.95):
    period = _seasonal_period(frequency, seasonality, len(pre))
    exog_all, valid_covariates, warnings = _build_exog(pd.concat([pre, post]), covariates, pd.concat([pre, post]).index)
    exog_pre = exog_all.loc[pre.index] if exog_all is not None else None
    exog_post = exog_all.loc[post.index] if exog_all is not None else None
    requested_level = "local linear trend" if local_trend else "local level"
    attempts = [(requested_level, "lbfgs", 500), (requested_level, "nm", 1000)]
    if requested_level != "local level":
        attempts.append(("local level", "lbfgs", 500))
    last_fitted, last_frame = None, None
    for level, method, maxiter in attempts:
        try:
            model = UnobservedComponents(pre["_outcome"].astype(float), level=level, seasonal=period, exog=exog_pre)
            fitted = model.fit(disp=False, method=method, maxiter=maxiter)
            frame = fitted.get_forecast(steps=len(post), exog=exog_post).summary_frame(alpha=1 - interval)
            last_fitted, last_frame = fitted, frame
            retvals = getattr(fitted, "mle_retvals", {}) or {}
            if retvals.get("converged", True) and np.isfinite(frame.to_numpy(dtype=float)).all():
                if (level, method) != attempts[0]:
                    warnings.append(f"Model fit used fallback specification: {level} ({method}).")
                return fitted, frame, period, valid_covariates, warnings
        except Exception as exc:
            warnings.append(f"Model fit attempt failed: {level} ({method}): {exc}")
    if last_fitted is None or last_frame is None:
        raise RuntimeError("No structural model specification could be fitted.")
    warnings.append("All structural model fit attempts failed convergence checks.")
    return last_fitted, last_frame, period, valid_covariates, warnings


def _backtest(pre: pd.DataFrame, frequency: str, seasonality: str, covariates: list[str], local_trend: bool, interval: float, holdout_length: int = 8) -> dict:
    if len(pre) < 30:
        return {"status": "Not enough pre-period history", "folds": 0, "failed_folds": 0, "mae": np.nan, "rmse": np.nan, "mape": np.nan, "wape": np.nan, "mase": np.nan, "interval_coverage": np.nan, "bias": np.nan, "failure_rate": np.nan}
    holdout = min(max(4, int(holdout_length)), max(4, len(pre) // 4))
    min_train = max(20, len(pre) // 2)
    origins = list(range(min_train, len(pre) - holdout + 1, max(1, holdout // 2)))
    rows = []
    failed = 0
    for origin in origins:
        train, test = pre.iloc[:origin].copy(), pre.iloc[origin : origin + holdout].copy()
        try:
            fitted, frame, _, _, _ = _fit_structural(train, test, frequency, seasonality, covariates, local_trend, interval)
            if not _model_is_stable(fitted, frame):
                failed += 1
                continue
            pred = frame["mean"].to_numpy(dtype=float)
            actual = test["_outcome"].to_numpy(dtype=float)
            low = frame["mean_ci_lower"].to_numpy(dtype=float)
            high = frame["mean_ci_upper"].to_numpy(dtype=float)
            errors = actual - pred
            rows.append({"errors": errors, "actual": actual, "low": low, "high": high})
        except Exception:
            failed += 1
    if not rows:
        return {"status": "Failed: no stable rolling-origin folds", "folds": 0, "failed_folds": failed, "mae": np.nan, "rmse": np.nan, "mape": np.nan, "wape": np.nan, "mase": np.nan, "interval_coverage": np.nan, "bias": np.nan, "failure_rate": 1.0}
    errors = np.concatenate([row["errors"] for row in rows])
    actual = np.concatenate([row["actual"] for row in rows])
    low = np.concatenate([row["low"] for row in rows])
    high = np.concatenate([row["high"] for row in rows])
    nonzero = np.abs(actual) > 1e-9
    scale = np.mean(np.abs(np.diff(pre["_outcome"].astype(float))))
    return {
        "status": "Pass", "folds": len(rows), "failed_folds": failed,
        "failure_rate": failed / max(len(origins), 1),
        "mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": float(np.mean(np.abs(errors[nonzero] / actual[nonzero]))) if nonzero.any() else np.nan,
        "wape": float(np.sum(np.abs(errors)) / max(np.sum(np.abs(actual)), 1e-9)),
        "mase": float(np.mean(np.abs(errors)) / max(scale, 1e-9)),
        "interval_coverage": float(np.mean((actual >= low) & (actual <= high))),
        "bias": float(np.mean(errors)),
    }


def _model_is_stable(fitted, frame: pd.DataFrame) -> bool:
    retvals = getattr(fitted, "mle_retvals", {}) or {}
    converged = retvals.get("converged", True)
    params = np.asarray(getattr(fitted, "params", []), dtype=float)
    return bool(converged and np.isfinite(params).all() and np.isfinite(frame.to_numpy(dtype=float)).all())


def _joint_forecast_draws(fitted, steps: int, exog: pd.DataFrame | None, frame: pd.DataFrame, draws: int, seed: int) -> np.ndarray:
    """Use statsmodels' state-space simulator to generate correlated paths."""
    simulated = fitted.simulate(steps, repetitions=draws, anchor="end", exog=exog, random_state=seed)
    values = simulated.to_numpy(dtype=float) if hasattr(simulated, "to_numpy") else np.asarray(simulated, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape != (steps, draws):
        values = values.reshape(steps, draws)
    return values.T


def fit_bsts_model(
    data: pd.DataFrame,
    intervention_date: str,
    frequency: str,
    covariates: list[str] | None = None,
    credible_interval: float = 0.95,
    draws: int = 1000,
    campaign_end_date: str = "",
    seasonality: str = "Auto",
    local_trend: bool = True,
    seed: int = 123,
    holdout_length: int = 8,
    planner_mode: bool = False,
) -> dict:
    """Fit a pre-period structural state-space model and forecast the counterfactual."""
    if not 0 < credible_interval < 1:
        raise ValueError("Prediction interval must be between 0 and 1.")
    pre, post = intervention_periods(data, intervention_date, campaign_end_date)
    if len(pre) < 20:
        raise ValueError("At least 20 pre-intervention observations are required.")
    if len(post) < 2:
        raise ValueError("At least two post-intervention observations are required.")
    covariates = covariates or []
    fitted, frame, period, valid_covariates, warnings = _fit_structural(pre, post, frequency, seasonality, covariates, local_trend, credible_interval)
    if not _model_is_stable(fitted, frame):
        raise RuntimeError("The structural model did not converge. Try a simpler trend or seasonality setting.")
    alpha = 1 - credible_interval
    frame = fitted.get_forecast(steps=len(post), exog=None if not valid_covariates else _build_exog(pd.concat([pre, post]), valid_covariates, pd.concat([pre, post]).index)[0].loc[post.index]).summary_frame(alpha=alpha)
    pred = frame["mean"].to_numpy(dtype=float)
    lower_col = f"mean_ci_lower" if "mean_ci_lower" in frame else frame.columns[2]
    upper_col = f"mean_ci_upper" if "mean_ci_upper" in frame else frame.columns[3]
    lower, upper = frame[lower_col].to_numpy(dtype=float), frame[upper_col].to_numpy(dtype=float)
    observed = post["_outcome"].to_numpy(dtype=float)
    impact = observed - pred
    exog_post = None if not valid_covariates else _build_exog(pd.concat([pre, post]), valid_covariates, pd.concat([pre, post]).index)[0].loc[post.index]
    simulations = _joint_forecast_draws(fitted, len(post), exog_post, frame, draws, seed)
    cumulative_sims = (observed - simulations).sum(axis=1)
    ci_lower, ci_upper = np.quantile(cumulative_sims, [alpha / 2, 1 - alpha / 2])
    impact_test_statistic = float(impact.sum() / max(float(np.std(cumulative_sims, ddof=1)), 1e-9))
    if planner_mode:
        residual_pvalue = None
        residual_autocorrelation = False
        backtest = {"status": "Skipped in planner mode", "folds": 0, "failed_folds": 0, "failure_rate": 0.0, "mae": np.nan, "rmse": np.nan, "mape": np.nan, "wape": np.nan, "mase": np.nan, "interval_coverage": np.nan, "bias": np.nan}
    else:
        residuals = np.asarray(fitted.resid).astype(float)
        ljung = acorr_ljungbox(residuals, lags=[min(10, max(1, len(residuals) // 5))], return_df=True)
        residual_pvalue = float(ljung["lb_pvalue"].iloc[-1]) if not ljung.empty else None
        residual_autocorrelation = bool(any(value < 0.05 for value in ljung["lb_pvalue"]))
        warnings.extend(["Residual autocorrelation remains material." for value in ljung["lb_pvalue"] if value < 0.05])
        backtest = _backtest(pre, frequency, seasonality, valid_covariates, local_trend, credible_interval, holdout_length=holdout_length)
        if backtest["status"] == "Pass" and pd.notna(backtest["interval_coverage"]) and backtest["interval_coverage"] < credible_interval * 0.8:
            warnings.append("Pre-period backtest interval coverage is below the configured level.")
    result = post[["_date", "_outcome"]].copy()
    result["counterfactual"] = pred
    result["lower"] = lower
    result["upper"] = upper
    result["impact"] = impact
    stable = _model_is_stable(fitted, frame)
    reliability = assess_model_reliability(backtest, credible_interval, residual_pvalue, len(pre), float(np.mean(np.abs(pre["_outcome"]))), stable=stable)
    return {
        "method": "Structural Time Series Counterfactual", "pre_observations": len(pre), "post_observations": len(post),
        "result": result, "observed_total": float(observed.sum()), "expected_total": float(pred.sum()),
        "cumulative_impact": float(impact.sum()), "relative_impact": float(impact.sum() / pred.sum()) if pred.sum() else np.nan,
        "prediction_interval": (float(ci_lower), float(ci_upper)), "interval_level": credible_interval,
        "simulation_probability_positive": float(np.mean(cumulative_sims > 0)), "seasonal_period": period,
        "covariates": valid_covariates, "model_specification": {"level": "local linear trend" if local_trend else "local level", "seasonality": seasonality, "seasonal_period": period},
        "backtest": backtest, "warnings": warnings, "warning": " ".join(warnings), "seed": seed, "reliability": reliability["overall"],
        "reliability_details": reliability,
        "model_stable": stable,
        "residual_autocorrelation": residual_autocorrelation,
        "decision": "positive" if ci_lower > 0 else "negative" if ci_upper < 0 else "inconclusive",
        "credible_interval": (float(ci_lower), float(ci_upper)),
        "posterior_probability_positive": float(np.mean(cumulative_sims > 0)),
        "impact_test_statistic": impact_test_statistic,
        "forecast_draws": simulations if planner_mode else None,
        "planner_mode": planner_mode,
        "uncertainty_note": "Conditional on fitted parameters; future state and observation shocks are simulated.",
    }


def config_fingerprint(config: TimeSeriesConfig) -> str:
    payload = {key: value for key, value in vars(config).items() if key not in {"dataset_name"}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def method_config_fingerprint(config: TimeSeriesConfig, method: str) -> str:
    """Hash only settings that affect the selected analysis method."""
    payload = {key: value for key, value in vars(config).items() if key not in {"dataset_name", "analysis_method"}}
    if method == "Pre–Post Analysis":
        payload = {key: value for key, value in payload.items() if key in {"date_column", "outcome_column", "frequency", "intervention_date", "campaign_end_date", "comparison_metric", "pre_window", "post_window", "comparison_window", "symmetric_periods", "custom_pre_start", "custom_pre_end", "custom_post_start", "custom_post_end", "duplicate_policy", "missing_policy"}}
    else:
        payload = {key: value for key, value in payload.items() if key not in {"comparison_metric", "pre_window", "post_window", "comparison_window", "symmetric_periods", "custom_pre_start", "custom_pre_end", "custom_post_start", "custom_post_end"}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def data_fingerprint(data: pd.DataFrame) -> str:
    """Return a stable fingerprint for prepared data used by a result."""
    values = pd.util.hash_pandas_object(data.sort_index(), index=True).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def projected_operational_exposure(data: pd.DataFrame, exposure_column: str, duration: int) -> float | None:
    """Project operational exposure; this is separate from statistical time points."""
    if not exposure_column or exposure_column == "None" or exposure_column not in data.columns:
        return None
    values = safe_numeric_series(data, exposure_column).dropna()
    if values.empty:
        return None
    return float(values.median() * max(int(duration), 0))


def planning_config_fingerprint(data: pd.DataFrame, config: TimeSeriesConfig, effect_mode: str, effect: float | None, effect_pattern: str, ramp_period: int, alpha: float, target_power: float, maximum_duration: int) -> str:
    launch = config.planned_launch_date or config.intervention_date
    payload = {"data": data_fingerprint(data), "date_column": config.date_column, "outcome_column": config.outcome_column, "exposure_column": config.exposure_column, "frequency": config.frequency, "launch": launch, "predictors": config.selected_covariates, "seasonality": config.seasonality, "local_trend": config.local_trend, "effect_mode": effect_mode, "effect": effect, "effect_pattern": effect_pattern, "ramp_period": ramp_period, "alpha": alpha, "target_power": target_power, "maximum_duration": maximum_duration, "seed": config.simulation_seed}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
