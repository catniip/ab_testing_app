from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .core import infer_column


DMA_CENTROIDS: dict[str, tuple[float, float]] = {
    "Atlanta": (33.7490, -84.3880),
    "Baltimore": (39.2904, -76.6122),
    "Boston": (42.3601, -71.0589),
    "Charlotte": (35.2271, -80.8431),
    "Chicago": (41.8781, -87.6298),
    "Dallas-Ft. Worth": (32.7767, -96.7970),
    "Denver": (39.7392, -104.9903),
    "Detroit": (42.3314, -83.0458),
    "Houston": (29.7604, -95.3698),
    "Los Angeles": (34.0522, -118.2437),
    "Miami-Ft. Lauderdale": (25.7617, -80.1918),
    "Minneapolis-St. Paul": (44.9778, -93.2650),
    "New York": (40.7128, -74.0060),
    "Philadelphia": (39.9526, -75.1652),
    "Phoenix": (33.4484, -112.0740),
    "San Francisco-Oakland-San Jose": (37.7749, -122.4194),
    "Seattle": (47.6062, -122.3321),
    "Tampa-St. Petersburg": (27.9506, -82.4572),
    "Washington DC": (38.9072, -77.0369),
}


@dataclass
class GeoConfig:
    data_source: str = "Upload File"
    dataset_name: str = ""
    date_column: str = ""
    dma_column: str = ""
    outcome_column: str = ""
    market_size_column: str = "None"
    spend_column: str = "None"
    group_column: str = "None"
    pair_column: str = "None"
    test_label: str = "Test"
    control_label: str = "Control"
    estimand: str = "Equal weight per DMA"
    planned_launch_date: str = ""
    effect_type: str = "Relative lift"
    expected_effect: float = 5.0
    alpha: float = 0.05
    target_power: float = 0.80
    max_duration: int = 26
    ramp_periods: int = 0
    outcome_delay_periods: int = 0
    campaign_start_date: str = ""
    campaign_end_date: str = ""
    outcome_view: str = "Indexed"


def infer_geo_schema(df: pd.DataFrame) -> dict[str, str]:
    columns = list(df.columns)
    numeric = [col for col in columns if pd.to_numeric(df[col], errors="coerce").notna().mean() > 0.8]
    return {
        "date": infer_column(columns, ["week", "date", "period", "month", "day"], columns[0] if columns else ""),
        "dma": infer_column(columns, ["dma", "market", "geo", "region", "designated_market_area"], columns[1] if len(columns) > 1 else ""),
        "outcome": infer_column(numeric, ["applications", "outcome", "revenue", "transactions", "sales", "conversions"], numeric[0] if numeric else ""),
        "market_size": infer_column(columns, ["population", "customers", "households", "market_size"], "None"),
        "spend": infer_column(columns, ["spend", "media_spend", "marketing_spend"], "None"),
        "group": infer_column(columns, ["treatment_group", "group", "assignment", "test_control"], "None"),
        "pair": infer_column(columns, ["pair_id", "matched_pair", "pair"], "None"),
    }


def prepare_geo_panel(df: pd.DataFrame, date_col: str, dma_col: str, outcome_col: str) -> pd.DataFrame:
    data = df.copy()
    data["_date"] = pd.to_datetime(data[date_col], errors="coerce")
    data["_dma"] = data[dma_col].astype(str).str.strip()
    data["_outcome"] = pd.to_numeric(data[outcome_col], errors="coerce")
    return data.dropna(subset=["_date", "_dma", "_outcome"]).sort_values(["_dma", "_date"]).reset_index(drop=True)


def validate_geo_panel(data: pd.DataFrame) -> dict[str, int | str]:
    duplicates = int(data.duplicated(["_dma", "_date"]).sum())
    dma_count = int(data["_dma"].nunique()) if "_dma" in data else 0
    period_count = int(data["_date"].nunique()) if "_date" in data else 0
    frequency = "Irregular"
    missing = 0
    dates = sorted(data["_date"].dropna().unique())
    if len(dates) >= 3:
        diffs = pd.Series(dates).diff().dropna().dt.days
        median = float(diffs.median())
        if 0.8 <= median <= 1.2:
            frequency = "Daily"
            expected = pd.date_range(min(dates), max(dates), freq="D")
            observed = pd.DatetimeIndex(dates).normalize()
            missing = int(len(expected.difference(observed)))
        elif 5.5 <= median <= 8.5:
            frequency = "Weekly"
            expected = pd.date_range(min(dates), max(dates), freq="7D")
            observed = pd.DatetimeIndex(dates).normalize()
            missing = int(len(expected.difference(observed)))
        elif 27 <= median <= 32:
            frequency = "Monthly"
            observed_periods = pd.DatetimeIndex(dates).to_period("M")
            expected_periods = pd.period_range(observed_periods.min(), observed_periods.max(), freq="M")
            missing = int(len(expected_periods.difference(observed_periods)))
    return {"duplicates": duplicates, "dma_count": dma_count, "period_count": period_count, "frequency": frequency, "missing_periods": missing}


def assignment_from_existing(
    data: pd.DataFrame,
    group_col: str,
    selected_dmas: list[str] | None = None,
    pair_col: str = "None",
    test_label: str = "Test",
    control_label: str = "Control",
) -> pd.DataFrame:
    """Return one authoritative assignment row per DMA without inventing pairs."""
    current = data[data["_dma"].isin(selected_dmas)].copy() if selected_dmas else data.copy()
    label_map = {test_label.strip().lower(): "Test", control_label.strip().lower(): "Control"}
    rows: list[dict] = []
    for dma, group in current.groupby("_dma", sort=True):
        labels = group[group_col].dropna().astype(str).str.strip()
        normalized = labels.str.lower().map(label_map).dropna().unique().tolist()
        if len(normalized) != 1:
            continue
        row = {"DMA": dma, "Group": normalized[0]}
        if pair_col != "None" and pair_col in group:
            pairs = group[pair_col].dropna().astype(str).str.strip()
            row["Pair"] = pairs.iloc[0] if len(pairs) else ""
        rows.append(row)
    columns = ["DMA", "Group"] + (["Pair"] if pair_col != "None" else [])
    return pd.DataFrame(rows, columns=columns)


def validate_fixed_assignment(
    data: pd.DataFrame,
    group_col: str,
    pair_col: str = "None",
    test_label: str = "Test",
    control_label: str = "Control",
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    if group_col == "None" or group_col not in data:
        return {"assignment": pd.DataFrame(columns=["DMA", "Group"]), "errors": ["Select the column containing the fixed Test and Control assignment."], "warnings": [], "counts": {}}
    if test_label.strip().lower() == control_label.strip().lower():
        errors.append("Test and Control source labels must be different.")
    expected = {test_label.strip().lower(), control_label.strip().lower()}
    unstable: list[str] = []
    unknown: set[str] = set()
    missing: list[str] = []
    for dma, group in data.groupby("_dma"):
        labels = group[group_col].dropna().astype(str).str.strip()
        normalized = set(labels.str.lower())
        if not normalized:
            missing.append(str(dma))
        elif len(normalized) > 1:
            unstable.append(str(dma))
        unknown.update(value for value in normalized if value not in expected)
    if unstable:
        errors.append(f"Assignment changes over time for {len(unstable)} DMA(s): {', '.join(unstable[:5])}.")
    if unknown:
        errors.append("Unrecognized assignment value(s): " + ", ".join(sorted(unknown)[:6]) + ". Map the exact Test and Control labels below.")
    if missing:
        errors.append(f"{len(missing)} DMA(s) have no assignment value.")
    assignment = assignment_from_existing(data, group_col, pair_col=pair_col, test_label=test_label, control_label=control_label)
    counts = assignment["Group"].value_counts().to_dict() if not assignment.empty else {}
    if counts.get("Test", 0) == 0 or counts.get("Control", 0) == 0:
        errors.append("The data must contain at least one Test DMA and one Control DMA.")
    if min(counts.get("Test", 0), counts.get("Control", 0)) < 4:
        warnings.append("Fewer than four DMAs in one arm makes clustered uncertainty and sensitivity checks fragile.")
    if pair_col != "None" and pair_col in data:
        unstable_pairs = data.groupby("_dma")[pair_col].nunique(dropna=True)
        if (unstable_pairs > 1).any():
            errors.append("Pair IDs change over time for one or more DMAs.")
        if not assignment.empty and "Pair" in assignment:
            pair_mix = assignment.groupby("Pair")["Group"].nunique()
            bad_pairs = pair_mix[pair_mix < 2].index.astype(str).tolist()
            if bad_pairs:
                warnings.append(f"{len(bad_pairs)} supplied pair ID(s) do not contain both Test and Control DMAs. Pairing will be treated as metadata only.")
    return {"assignment": assignment, "errors": errors, "warnings": warnings, "counts": counts}


def _assignment_maps(assignment: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    if {"DMA", "Group"}.issubset(assignment.columns):
        group_map = assignment.set_index("DMA")["Group"].to_dict()
        pair_map = assignment.set_index("DMA")["Pair"].to_dict() if "Pair" in assignment else {}
        return group_map, pair_map
    # Compatibility for historical saved sessions; new assignments are always long-form.
    group_map = {row["Test DMA"]: "Test" for _, row in assignment.iterrows()} | {row["Control DMA"]: "Control" for _, row in assignment.iterrows()}
    pair_map = {row["Test DMA"]: str(row["Pair"]) for _, row in assignment.iterrows()} | {row["Control DMA"]: str(row["Pair"]) for _, row in assignment.iterrows()}
    return group_map, pair_map


def evaluate_geo_balance(data: pd.DataFrame, assignment: pd.DataFrame, intervention_date: str) -> dict:
    if assignment.empty:
        return {}
    group_map, _ = _assignment_maps(assignment)
    pre = data[(data["_dma"].isin(group_map)) & (data["_date"] < pd.Timestamp(intervention_date))].copy()
    pre["group"] = pre["_dma"].map(group_map)
    agg = pre.groupby(["_date", "group"])["_outcome"].mean().unstack()
    corr = float(agg["Test"].corr(agg["Control"])) if {"Test", "Control"}.issubset(agg.columns) else np.nan
    means = pre.groupby("group")["_outcome"].mean()
    rel_diff = float((means.get("Test", np.nan) - means.get("Control", np.nan)) / max(abs(means.get("Control", 1.0)), 1.0))
    trends = {}
    for group, series in agg.items():
        y = series.dropna().to_numpy()
        trends[group] = float(np.polyfit(np.arange(len(y)), y, 1)[0]) if len(y) >= 2 else np.nan
    trend_diff = float((trends.get("Test", np.nan) - trends.get("Control", np.nan)) / max(abs(agg.mean().mean()), 1.0))
    if pd.notna(corr) and corr >= 0.95 and abs(rel_diff) <= 0.05 and abs(trend_diff) <= 0.02:
        status = "Excellent"
    elif pd.notna(corr) and corr >= 0.85 and abs(rel_diff) <= 0.12:
        status = "Good"
    elif pd.notna(corr) and corr >= 0.70 and abs(rel_diff) <= 0.20:
        status = "Fair"
    else:
        status = "Poor"
    return {"aggregate": agg.reset_index(), "correlation": corr, "relative_difference": rel_diff, "trend_difference": trend_diff, "status": status, "dma_count": len(assignment)}


def prepare_geo_analysis(
    data: pd.DataFrame,
    assignment: pd.DataFrame,
    intervention_date: str,
    end_date: str = "",
    weight_col: str = "None",
) -> pd.DataFrame:
    group_map, pair_map = _assignment_maps(assignment)
    panel = data[data["_dma"].isin(group_map)].copy()
    panel["group"] = panel["_dma"].map(group_map)
    panel["pair"] = panel["_dma"].map(pair_map).fillna("")
    panel["post"] = panel["_date"] >= pd.Timestamp(intervention_date)
    if weight_col != "None" and weight_col in panel:
        panel["_analysis_weight"] = pd.to_numeric(panel[weight_col], errors="coerce")
        panel["_analysis_weight"] = panel["_analysis_weight"].where(panel["_analysis_weight"] > 0)
        panel["_analysis_weight"] = panel["_analysis_weight"].fillna(panel["_analysis_weight"].median())
    else:
        panel["_analysis_weight"] = 1.0
    if end_date:
        panel = panel[panel["_date"] <= pd.Timestamp(end_date)]
    return panel


def _two_way_residualize(values: pd.Series, dma: pd.Series, dates: pd.Series, weights: pd.Series) -> pd.Series:
    frame = pd.DataFrame(
        {"value": values.astype(float), "dma": dma.to_numpy(), "date": dates.to_numpy(), "weight": weights.to_numpy()},
        index=values.index,
    )
    for _ in range(100):
        previous = frame["value"].to_numpy(copy=True)
        dma_mean = frame.groupby("dma", observed=True).apply(
            lambda group: np.average(group["value"], weights=group["weight"]), include_groups=False
        )
        frame["value"] = frame["value"] - frame["dma"].map(dma_mean)
        date_mean = frame.groupby("date", observed=True).apply(
            lambda group: np.average(group["value"], weights=group["weight"]), include_groups=False
        )
        frame["value"] = frame["value"] - frame["date"].map(date_mean)
        if np.max(np.abs(frame["value"].to_numpy() - previous)) < 1e-10:
            break
    return frame["value"]


def _fit_two_way_clustered(data: pd.DataFrame, regressor: str, weighted: bool) -> tuple[float, float, float, tuple[float, float]]:
    weights = data["_analysis_weight"].astype(float) if weighted else pd.Series(1.0, index=data.index)
    weights = weights / weights.mean()
    y = _two_way_residualize(data["_outcome"], data["_dma"], data["_date"], weights)
    x = _two_way_residualize(data[regressor], data["_dma"], data["_date"], weights)
    denominator = float(np.sum(weights * x * x))
    if denominator <= 1e-12:
        raise ValueError("The campaign indicator has no variation after accounting for DMA and date effects.")
    estimate = float(np.sum(weights * x * y) / denominator)
    residual = y - estimate * x
    cluster_score = (weights * x * residual).groupby(data["_dma"]).sum()
    clusters = len(cluster_score)
    observations = len(data)
    parameters = data["_dma"].nunique() + data["_date"].nunique()
    correction = clusters / max(clusters - 1, 1) * (observations - 1) / max(observations - parameters, 1)
    variance = correction * float(np.sum(cluster_score * cluster_score)) / denominator**2
    standard_error = float(np.sqrt(max(variance, 0.0)))
    degrees_freedom = max(clusters - 1, 1)
    t_value = estimate / standard_error if standard_error > 0 else np.inf
    pvalue = float(2 * stats.t.sf(abs(t_value), degrees_freedom))
    critical = float(stats.t.ppf(0.975, degrees_freedom))
    interval = (estimate - critical * standard_error, estimate + critical * standard_error)
    return estimate, standard_error, pvalue, interval


def run_panel_did(panel: pd.DataFrame, weighted: bool = False) -> dict:
    """Estimate a two-way fixed-effects DiD with standard errors clustered by DMA."""
    required = {"_dma", "_date", "_outcome", "group", "post"}
    if panel.empty or not required.issubset(panel.columns):
        raise ValueError("A mapped DMA panel with fixed Test and Control groups is required.")
    if panel["_dma"].nunique() < 4:
        raise ValueError("At least four DMAs are required to estimate clustered uncertainty.")
    if not panel["post"].any() or panel["post"].all():
        raise ValueError("Both pre-campaign and post-campaign observations are required.")
    if set(panel["group"].dropna().unique()) != {"Test", "Control"}:
        raise ValueError("Both Test and Control DMA groups are required.")

    model_data = panel.dropna(subset=["_outcome", "_dma", "_date"]).copy()
    model_data["treated_post"] = ((model_data["group"] == "Test") & model_data["post"]).astype(int)
    estimate, standard_error, pvalue, interval = _fit_two_way_clustered(model_data, "treated_post", weighted)
    low, high = interval
    baseline = float(model_data[(model_data["group"] == "Control") & (~model_data["post"])]["_outcome"].mean())
    post_periods = int(model_data.loc[model_data["post"], "_date"].nunique())
    test_dmas = int(model_data.loc[model_data["group"] == "Test", "_dma"].nunique())
    control_dmas = int(model_data.loc[model_data["group"] == "Control", "_dma"].nunique())
    lift = estimate / max(abs(baseline), 1e-12)
    incremental_volume = estimate * post_periods * test_dmas

    pre = model_data[~model_data["post"]].copy()
    pretrend_pvalue = np.nan
    if pre["_date"].nunique() >= 4:
        pre["_time"] = pre["_date"].rank(method="dense").astype(float)
        pre["_test_time"] = (pre["group"] == "Test").astype(int) * pre["_time"]
        try:
            _, _, pretrend_pvalue, _ = _fit_two_way_clustered(pre, "_test_time", weighted)
        except (ValueError, np.linalg.LinAlgError):
            pretrend_pvalue = np.nan

    cluster_count = test_dmas + control_dmas
    if cluster_count >= 20 and model_data[~model_data["post"]]["_date"].nunique() >= 12:
        reliability = "Good"
    elif cluster_count >= 8:
        reliability = "Review"
    else:
        reliability = "Limited"
    return {
        "effect_per_period": estimate,
        "lift": lift,
        "interval": (low, high),
        "pvalue": pvalue,
        "standard_error": standard_error,
        "incremental_volume": float(incremental_volume),
        "post_periods": post_periods,
        "test_dmas": test_dmas,
        "control_dmas": control_dmas,
        "cluster_count": cluster_count,
        "pretrend_pvalue": pretrend_pvalue,
        "reliability": reliability,
        "estimand": "Market-size-weighted average DMA" if weighted else "Equal-weight average DMA",
        "model": "DMA and date fixed-effects Difference-in-Differences",
    }


def _duration_candidates(frequency: str, maximum: int) -> list[int]:
    defaults = {
        "Daily": [7, 14, 21, 28, 42, 56, 84],
        "Weekly": [4, 6, 8, 12, 16, 20, 26],
        "Monthly": [2, 3, 4, 6, 9, 12],
    }.get(frequency, [2, 4, 6, 8, 12, 16, 20, 26])
    values = [value for value in defaults if value <= maximum]
    if maximum > 0 and maximum not in values:
        values.append(maximum)
    return sorted(set(values))


def _add_periods(date: pd.Timestamp, periods: int, frequency: str) -> pd.Timestamp:
    if frequency == "Daily":
        return date + pd.Timedelta(days=periods)
    if frequency == "Weekly":
        return date + pd.Timedelta(weeks=periods)
    if frequency == "Monthly":
        return date + pd.DateOffset(months=periods)
    return date + pd.Timedelta(days=periods)


def plan_geo_test(
    panel: pd.DataFrame,
    assignment: pd.DataFrame,
    planned_launch_date: str,
    frequency: str,
    effect_type: str,
    expected_effect: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
    max_duration: int = 26,
    ramp_periods: int = 0,
    outcome_delay_periods: int = 0,
    weighted: bool = False,
    weight_col: str = "None",
) -> dict:
    """Estimate duration and DMA requirements for a user-supplied effect scenario."""
    launch = pd.Timestamp(planned_launch_date)
    history = prepare_geo_analysis(panel, assignment, launch.isoformat(), weight_col=weight_col)
    history = history[history["_date"] < launch].copy()
    if history["_date"].nunique() < 6:
        raise ValueError("At least six historical time periods are required for geographic planning.")

    def group_average(group: pd.DataFrame) -> float:
        if weighted:
            return float(np.average(group["_outcome"], weights=group["_analysis_weight"]))
        return float(group["_outcome"].mean())

    trend = history.groupby(["_date", "group"], observed=True).apply(group_average, include_groups=False).rename("value").reset_index()
    pivot = trend.pivot(index="_date", columns="group", values="value").dropna(subset=["Test", "Control"])
    if len(pivot) < 6:
        raise ValueError("At least six complete historical periods are required in both groups.")
    differential = (pivot["Test"] - pivot["Control"]).astype(float)
    x = np.arange(len(differential), dtype=float)
    residual = differential - np.polyval(np.polyfit(x, differential.to_numpy(), 1), x)
    residual = pd.Series(residual, index=differential.index)
    residual_sd = float(residual.std(ddof=1))
    if not np.isfinite(residual_sd) or residual_sd <= 0:
        raise ValueError("Historical variation is too small to estimate planning uncertainty.")
    rho = float(residual.autocorr(lag=1)) if len(residual) >= 4 else 0.0
    rho = float(np.clip(rho if np.isfinite(rho) else 0.0, -0.5, 0.8))
    baseline = float(pivot["Control"].mean())
    assumed_effect = expected_effect / 100.0 * abs(baseline) if effect_type == "Relative lift" else expected_effect
    assumed_effect = abs(float(assumed_effect))
    if assumed_effect <= 0:
        raise ValueError("The expected effect scenario must be greater than zero.")

    counts = assignment["Group"].value_counts()
    test_dmas = int(counts.get("Test", 0))
    control_dmas = int(counts.get("Control", 0))
    current_total = test_dmas + control_dmas
    allocation = test_dmas / current_total
    z_alpha = float(stats.norm.ppf(1 - alpha / 2))
    z_power = float(stats.norm.ppf(target_power))
    rows: list[dict] = []
    prior_se = float("inf")
    recommended: int | None = None
    for duration in _duration_candidates(frequency, int(max_duration)):
        analytic_se = residual_sd / np.sqrt(duration) * np.sqrt((1 + rho) / max(1 - rho, 0.05))
        rolling = residual.rolling(duration).mean().dropna()
        empirical_se = float(rolling.std(ddof=1)) if len(rolling) >= 5 else np.nan
        raw_se = max(analytic_se, empirical_se) if np.isfinite(empirical_se) else analytic_se
        se = min(prior_se, raw_se)
        prior_se = se
        noncentrality = assumed_effect / se
        power = float(stats.norm.sf(z_alpha - noncentrality) + stats.norm.cdf(-z_alpha - noncentrality))
        mde = float((z_alpha + z_power) * se)
        required_total = max(4, int(np.ceil(current_total * (mde / assumed_effect) ** 2)))
        required_test = max(2, int(np.ceil(required_total * allocation)))
        required_control = max(2, required_total - required_test)
        required_total = required_test + required_control
        meets = power >= target_power
        if meets and recommended is None:
            recommended = duration
            status = "Recommended: shortest option meeting target"
        elif meets:
            status = "Meets target; additional time"
        else:
            status = "Below detection target"
        readout = _add_periods(launch, ramp_periods + duration + outcome_delay_periods, frequency)
        rows.append(
            {
                "Campaign Duration": duration,
                "Detection Chance": power,
                "Minimum Detectable Effect": mde,
                "MDE Relative": mde / max(abs(baseline), 1e-12),
                "Required Test DMAs": required_test,
                "Required Control DMAs": required_control,
                "Required Total DMAs": required_total,
                "Earliest Readout": readout.date(),
                "Status": status,
            }
        )
    table = pd.DataFrame(rows)
    return {
        "table": table,
        "recommended_duration": recommended,
        "current_test_dmas": test_dmas,
        "current_control_dmas": control_dmas,
        "current_total_dmas": current_total,
        "baseline": baseline,
        "assumed_effect_absolute": assumed_effect,
        "assumed_effect_relative": assumed_effect / max(abs(baseline), 1e-12),
        "historical_periods": len(pivot),
        "lag1_autocorrelation": rho,
        "method_note": "Planning evaluates a user-supplied effect scenario; it does not forecast the campaign's unknown effect.",
    }


def leave_one_dma_out(panel: pd.DataFrame, weighted: bool = False) -> pd.DataFrame:
    rows = []
    for dma in sorted(panel["_dma"].unique()):
        reduced = panel[panel["_dma"] != dma]
        if reduced.groupby("group")["_dma"].nunique().min() < 2:
            continue
        try:
            result = run_panel_did(reduced, weighted=weighted)
            rows.append({"Excluded DMA": dma, "Estimated Effect": result["effect_per_period"], "Relative Lift": result["lift"]})
        except (ValueError, np.linalg.LinAlgError):
            continue
    return pd.DataFrame(rows)


def aggregate_trend(panel: pd.DataFrame, view: str = "Indexed") -> pd.DataFrame:
    data = panel.groupby(["_date", "group"])["_outcome"].mean().reset_index()
    if view == "Indexed":
        bases = data[~data["_date"].isin(panel[panel["post"]]["_date"])].groupby("group")["_outcome"].mean().replace(0, np.nan)
        data["_display_outcome"] = data.apply(lambda row: row["_outcome"] / bases.get(row["group"], np.nan) * 100, axis=1)
    else:
        data["_display_outcome"] = data["_outcome"]
    return data


def build_geo_preview(data: pd.DataFrame, assignment: pd.DataFrame, market_size_col: str = "None") -> dict:
    """Build presentation-ready geography summaries without changing the analysis panel."""
    quality = validate_geo_panel(data)
    group_map, _ = _assignment_maps(assignment)
    preview = data.copy()
    preview["group"] = preview["_dma"].map(group_map)
    assigned = preview[preview["group"].isin(["Test", "Control"])].copy()

    if assigned.empty:
        trend = pd.DataFrame(columns=["_date", "group", "_outcome", "_display_outcome"])
    else:
        trend = assigned.groupby(["_date", "group"], as_index=False)["_outcome"].mean()
        trend["_display_outcome"] = trend["_outcome"]

    summary_spec: dict[str, tuple[str, str]] = {
        "Average Outcome": ("_outcome", "mean"),
        "Periods": ("_date", "nunique"),
        "Start Date": ("_date", "min"),
        "Latest Date": ("_date", "max"),
    }
    if market_size_col != "None" and market_size_col in preview:
        summary_spec["Market Size"] = (market_size_col, "median")
    markets = preview.groupby("_dma", as_index=False).agg(**summary_spec).rename(columns={"_dma": "DMA"})
    markets.insert(1, "Assignment", markets["DMA"].map(group_map).fillna("Not assigned"))
    markets = markets.sort_values(["Assignment", "DMA"], kind="stable").reset_index(drop=True)

    counts = assignment["Group"].value_counts().to_dict() if {"DMA", "Group"}.issubset(assignment.columns) else {}
    return {
        "quality": quality,
        "trend": trend,
        "markets": markets,
        "counts": counts,
        "start_date": data["_date"].min() if not data.empty else pd.NaT,
        "end_date": data["_date"].max() if not data.empty else pd.NaT,
    }


def run_placebo_tests(panel: pd.DataFrame, count: int = 6, weighted: bool = False) -> dict:
    pre_dates = sorted(panel.loc[~panel["post"], "_date"].unique())
    if len(pre_dates) < 12:
        return {"status": "Not enough pre-period history", "effects": []}
    cutoffs = pre_dates[-min(len(pre_dates) - 4, count + 4) : -4]
    effects = []
    for cutoff in cutoffs[-count:]:
        pseudo = panel[panel["_date"] < panel.loc[panel["post"], "_date"].min()].copy()
        pseudo["post"] = pseudo["_date"] >= cutoff
        try:
            effects.append(run_panel_did(pseudo, weighted=weighted)["lift"])
        except (ValueError, np.linalg.LinAlgError):
            continue
    median_abs = float(np.median(np.abs(effects))) if effects else np.nan
    status = "Good" if pd.notna(median_abs) and median_abs < 0.08 else "Review"
    return {"status": status, "effects": effects}
