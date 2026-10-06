from __future__ import annotations

import math
from scipy.stats import nct, t
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from dataclasses import dataclass
import pandas as pd
from .core import metric_baseline
from .core import DesignConfig, MetricConfig
from itertools import combinations
import numpy as np
from scipy.stats import t
import re
from .core import first_series, safe_datetime_series, safe_numeric_series
from .core import DataMappingConfig, MetricConfig, PopulationConfig
from copy import deepcopy
from typing import Any
from .core import StrategyConfig
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests


# --- Power ---

def adjusted_alpha(alpha: float, comparisons: int, method: str = "Holm") -> float:
    """Return the conservative per-comparison alpha used for design planning."""
    if comparisons <= 1 or method == "None":
        return alpha
    return alpha / comparisons


def sample_size_continuous(alpha: float, power: float, mde: float, std_dev: float) -> int:
    effect_size = abs(mde) / std_dev
    n = TTestIndPower().solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0)
    return int(math.ceil(n))


def power_continuous_unequal(alpha: float, effect: float, control_sd: float, treatment_sd: float, n_control: int, n_treatment: int | None = None) -> float:
    """Approximate two-sided Welch-test power for unequal variances and enrollment."""
    n_treatment = n_control if n_treatment is None else n_treatment
    if n_control < 2 or n_treatment < 2 or control_sd <= 0 or treatment_sd <= 0:
        return 0.0
    control_var = control_sd**2 / n_control
    treatment_var = treatment_sd**2 / n_treatment
    standard_error = math.sqrt(control_var + treatment_var)
    degrees_freedom = (control_var + treatment_var) ** 2 / (
        control_var**2 / (n_control - 1) + treatment_var**2 / (n_treatment - 1)
    )
    critical = t.ppf(1 - alpha / 2, degrees_freedom)
    noncentrality = abs(effect) / standard_error
    return float(nct.cdf(-critical, degrees_freedom, noncentrality) + 1 - nct.cdf(critical, degrees_freedom, noncentrality))


def power_binary_unequal(alpha: float, baseline_rate: float, effect: float, n_control: int, n_treatment: int, direction: str = "Higher is Better") -> float:
    """Two-proportion normal-test power with an unequal treatment/control ratio."""
    signed_effect = -abs(effect) if direction == "Lower is Better" else abs(effect)
    treatment_rate = min(max(baseline_rate + signed_effect, 1e-6), 1 - 1e-6)
    effect_size = abs(proportion_effectsize(baseline_rate, treatment_rate))
    return float(NormalIndPower().power(effect_size=effect_size, nobs1=n_control, alpha=alpha, ratio=n_treatment / n_control))


def sample_size_continuous_unequal(alpha: float, power: float, mde: float, control_sd: float, treatment_sd: float) -> int:
    """Solve equal per-arm enrollment for a two-sided unequal-variance comparison."""
    if mde <= 0 or control_sd <= 0 or treatment_sd <= 0:
        raise ValueError("MDE and both historical standard deviations must be greater than zero.")
    low, high = 2, 4
    while power_continuous_unequal(alpha, mde, control_sd, treatment_sd, high) < power:
        high *= 2
        if high > 10_000_000:
            raise ValueError("Required sample size exceeds the supported planning range.")
    while low < high:
        middle = (low + high) // 2
        if power_continuous_unequal(alpha, mde, control_sd, treatment_sd, middle) >= power:
            high = middle
        else:
            low = middle + 1
    return low


def detectable_effect_continuous_unequal(alpha: float, power: float, control_sd: float, treatment_sd: float, n_per_arm: int) -> float:
    """Solve the absolute detectable effect for unequal historical variances."""
    low = 0.0
    high = max(control_sd, treatment_sd)
    while power_continuous_unequal(alpha, high, control_sd, treatment_sd, n_per_arm) < power:
        high *= 2
    for _ in range(60):
        middle = (low + high) / 2
        if power_continuous_unequal(alpha, middle, control_sd, treatment_sd, n_per_arm) >= power:
            high = middle
        else:
            low = middle
    return high


def sample_size_binary(alpha: float, power: float, baseline_rate: float, mde: float, direction: str = "Higher is Better") -> int:
    signed_effect = -abs(mde) if direction == "Lower is Better" else abs(mde)
    treatment_rate = min(max(baseline_rate + signed_effect, 1e-6), 1 - 1e-6)
    effect_size = abs(proportion_effectsize(baseline_rate, treatment_rate))
    n = NormalIndPower().solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0)
    return int(math.ceil(n))


def duration(total_sample_size: int, eligible_count: int, frequency: str) -> float:
    if eligible_count <= 0:
        return float("inf")
    units = total_sample_size / eligible_count
    return units if frequency == "Week" else units


def detectable_effect_continuous(alpha: float, power: float, std_dev: float, n_per_arm: int) -> float:
    effect_size = TTestIndPower().solve_power(nobs1=n_per_arm, alpha=alpha, power=power, ratio=1.0)
    return abs(effect_size * std_dev)


def detectable_effect_binary(alpha: float, power: float, baseline_rate: float, n_per_arm: int, direction: str = "Higher is Better") -> float:
    grid = [idx / 10000 for idx in range(1, 5000)]
    for effect in grid:
        if power_binary(alpha, baseline_rate, effect, n_per_arm, direction) >= power:
            return effect
    return grid[-1]


def power_continuous(alpha: float, effect: float, std_dev: float, n_per_arm: int) -> float:
    return float(TTestIndPower().power(effect_size=abs(effect) / std_dev, nobs1=n_per_arm, alpha=alpha, ratio=1.0))


def power_binary(alpha: float, baseline_rate: float, effect: float, n_per_arm: int, direction: str = "Higher is Better") -> float:
    signed_effect = -abs(effect) if direction == "Lower is Better" else abs(effect)
    treatment_rate = min(max(baseline_rate + signed_effect, 1e-6), 1 - 1e-6)
    effect_size = abs(proportion_effectsize(baseline_rate, treatment_rate))
    return float(NormalIndPower().power(effect_size=effect_size, nobs1=n_per_arm, alpha=alpha, ratio=1.0))


# --- Assumptions ---

@dataclass
class NormalizedAssumption:
    baseline: float
    standard_deviation: float | None
    variance: float | None
    effect_absolute: float


def normalize_metric(metric: MetricConfig) -> NormalizedAssumption:
    if metric.metric_type == "Binary":
        baseline = metric.baseline_rate
        effect = metric.effect_value * baseline if metric.effect_type == "Relative %" else metric.effect_value
        return NormalizedAssumption(baseline, None, baseline * (1 - baseline), abs(effect))
    if metric.variability_input == "Variance":
        std = math.sqrt(metric.variance)
        variance = metric.variance
    else:
        std = metric.standard_deviation
        variance = std**2
    effect = metric.effect_value * metric.baseline_mean if metric.effect_type == "Relative %" else metric.effect_value
    return NormalizedAssumption(metric.baseline_mean, std, variance, abs(effect))


def update_assumption_from_history(metric: MetricConfig, df: pd.DataFrame, column: str) -> None:
    stats = metric_baseline(df, column, metric.metric_type)
    if metric.metric_type == "Binary":
        metric.baseline_rate = stats["baseline"]
    else:
        metric.baseline_mean = stats["baseline"]
        metric.standard_deviation = stats["std_dev"]
        metric.variance = stats["std_dev"] ** 2


def metric_summary_from_processed(metrics: dict[str, MetricConfig], df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in metrics.values():
        column = metric.processed_column or metric.column
        if column not in df.columns:
            continue
        update_assumption_from_history(metric, df, column)
        rows.append(
            {
                "Metric": metric.name,
                "Role": metric.role,
                "Aggregation": metric.aggregation_method,
                "MOB": metric.mob_horizon,
                "Baseline": metric.baseline_rate if metric.metric_type == "Binary" else metric.baseline_mean,
                "SD": (metric.baseline_rate * (1 - metric.baseline_rate)) ** 0.5 if metric.metric_type == "Binary" else metric.standard_deviation,
                "Variance": metric.baseline_rate * (1 - metric.baseline_rate) if metric.metric_type == "Binary" else metric.variance,
            }
        )
    return pd.DataFrame(rows)


def validate_metric_assumptions(metric: MetricConfig, alpha: float | None = None, target_power: float | None = None) -> list[str]:
    errors = []
    if metric.metric_type == "Binary":
        if not 0 < metric.baseline_rate < 1:
            errors.append(f"{metric.name}: Baseline Rate must be between 0 and 1.")
        else:
            absolute_effect = metric.effect_value * metric.baseline_rate if metric.effect_type == "Relative %" else metric.effect_value
            if metric.direction == "Lower is Better" and absolute_effect >= metric.baseline_rate:
                errors.append(f"{metric.name}: the detectable reduction must be smaller than the baseline rate.")
            if metric.direction == "Higher is Better" and absolute_effect >= 1 - metric.baseline_rate:
                errors.append(f"{metric.name}: the detectable increase must keep the treatment rate below 100%.")
    else:
        if metric.variability_input == "Variance" and (not math.isfinite(metric.variance) or metric.variance <= 0):
            errors.append(f"{metric.name}: Variance must be a finite value greater than zero.")
        if metric.variability_input == "Standard Deviation" and (not math.isfinite(metric.standard_deviation) or metric.standard_deviation <= 0):
            errors.append(f"{metric.name}: Standard Deviation must be a finite value greater than zero.")
    if metric.effect_value <= 0:
        errors.append(f"{metric.name}: effect value must be greater than zero.")
    if alpha is not None and not 0 < alpha < 1:
        errors.append("Alpha must be between 0 and 1.")
    if target_power is not None and not 0 < target_power < 1:
        errors.append("Target Power must be between 0 and 1.")
    return errors


def required_n(metric: MetricConfig, alpha: float, target_power: float) -> int:
    assumption = normalize_metric(metric)
    if metric.metric_type == "Binary":
        return sample_size_binary(alpha, target_power, assumption.baseline, assumption.effect_absolute, metric.direction)
    return sample_size_continuous(alpha, target_power, assumption.effect_absolute, max(assumption.standard_deviation or 0, 1e-9))


def design_sample_size(metrics: dict[str, MetricConfig], design: DesignConfig, comparisons: int = 1) -> tuple[int, str, list[dict]]:
    rows = []
    planning_alpha = adjusted_alpha(design.alpha, comparisons, design.multiplicity_method)
    for metric in metrics.values():
        if design.sample_size_basis == "Primary Metric Only" and metric.role != "Primary":
            continue
        n = required_n(metric, planning_alpha, design.target_power)
        rows.append({"Metric": metric.name, "Role": metric.role, "Required N / Arm": n})
    binding = max(rows, key=lambda row: row["Required N / Arm"])
    return int(binding["Required N / Arm"]), f"{binding['Metric']} — {binding['Role']}", rows


def design_sample_size_by_arm(
    metrics: dict[str, MetricConfig],
    design: DesignConfig,
    arm_stats_by_role: dict[str, pd.DataFrame],
    control: float,
    treatments: list[float],
) -> tuple[int, str, list[dict]]:
    """Plan numeric-arm enrollment from each arm's historical outcome variance."""
    planning_alpha = adjusted_alpha(design.alpha, max(len(treatments), 1), design.multiplicity_method)
    rows = []
    for role, metric in metrics.items():
        if design.sample_size_basis == "Primary Metric Only" and role != "Primary":
            continue
        stats = arm_stats_by_role.get(role, pd.DataFrame())
        control_row = _arm_stat_row(stats, control)
        if control_row is None:
            raise ValueError(f"{metric.name}: no historical support is available for the control point {control:,.0f}.")
        for treatment in treatments:
            treatment_row = _arm_stat_row(stats, treatment)
            if treatment_row is None:
                raise ValueError(f"{metric.name}: no historical support is available for treatment point {treatment:,.0f}.")
            baseline = float(control_row["Historical Mean"])
            effect = abs(float(metric.effect_value) * baseline) if metric.effect_type == "Relative %" else abs(float(metric.effect_value))
            if metric.metric_type == "Binary":
                required = sample_size_binary(planning_alpha, design.target_power, baseline, effect, metric.direction)
                control_sd = (baseline * (1 - baseline)) ** 0.5
                treatment_mean = max(0.0, min(1.0, baseline + (-effect if metric.direction == "Lower is Better" else effect)))
                treatment_sd = (treatment_mean * (1 - treatment_mean)) ** 0.5
            else:
                control_sd = float(control_row["Historical SD"])
                treatment_sd = float(treatment_row["Historical SD"])
                if not math.isfinite(control_sd) or control_sd <= 0 or not math.isfinite(treatment_sd) or treatment_sd <= 0:
                    raise ValueError(
                        f"{metric.name}: historical SD is unavailable for {control:,.0f} versus {treatment:,.0f}. "
                        "Use nearest-point grouping or provide more historical observations."
                    )
                required = sample_size_continuous_unequal(planning_alpha, design.target_power, effect, control_sd, treatment_sd)
            rows.append(
                {
                    "Metric": metric.name,
                    "Role": role,
                    "Comparison": f"{control:,.0f} vs {treatment:,.0f}",
                    "Control Historical N": int(control_row["Historical N"]),
                    "Treatment Historical N": int(treatment_row["Historical N"]),
                    "Control Mean": baseline,
                    "Treatment Mean": float(treatment_row["Historical Mean"]),
                    "Control SD": control_sd,
                    "Treatment SD": treatment_sd,
                    "Smallest Effect": effect,
                    "Required N / Arm": required,
                    "Historical Mapping": str(treatment_row["Mapping"]),
                    "Support": str(treatment_row["Support"]),
                }
            )
    if not rows:
        raise ValueError("No metric and treatment comparisons are available for sample-size planning.")
    binding = max(rows, key=lambda row: row["Required N / Arm"])
    label = f"{binding['Metric']} — {binding['Comparison']}"
    return int(binding["Required N / Arm"]), label, rows


def design_neyman_allocation(
    metrics: dict[str, MetricConfig],
    design: DesignConfig,
    arm_stats_by_role: dict[str, pd.DataFrame],
    control: float,
    treatments: list[float],
    family_comparisons: int | None = None,
) -> tuple[int, str, list[dict], list[dict]]:
    """Solve a generalized Neyman allocation for a shared-control multi-arm design."""
    arms = [float(control)] + [float(value) for value in treatments]
    primary_stats = arm_stats_by_role.get("Primary", pd.DataFrame())
    primary_rows = {arm: _arm_stat_row(primary_stats, arm) for arm in arms}
    missing = [arm for arm, row in primary_rows.items() if row is None]
    if missing:
        formatted = ", ".join(f"{value:,.0f}" for value in missing)
        raise ValueError(
            f"Primary metric historical support is unavailable for: {formatted}. "
            "Increase the historical matching distance, choose better-supported values, or provide more history."
        )

    shared_control_comparisons = max(len(treatments), 1)
    comparison_count = max(int(family_comparisons or shared_control_comparisons), 1)
    raw_weights = {}
    for arm, row in primary_rows.items():
        sd = float(row["Historical SD"])
        if not math.isfinite(sd) or sd <= 0:
            raise ValueError(f"Primary metric historical SD is unavailable for {arm:,.0f}.")
        raw_weights[arm] = sd * (math.sqrt(shared_control_comparisons) if arm == float(control) else 1.0)
    weight_total = sum(raw_weights.values())
    weights = {arm: weight / weight_total for arm, weight in raw_weights.items()}
    planning_alpha = adjusted_alpha(design.alpha, comparison_count, design.multiplicity_method)

    def counts_for_total(total: int) -> dict[float, int]:
        minimum = 2 * len(arms)
        total = max(total, minimum)
        remaining = total - minimum
        raw_extra = {arm: remaining * weights[arm] for arm in arms}
        counts = {arm: 2 + int(math.floor(raw_extra[arm])) for arm in arms}
        leftover = total - sum(counts.values())
        order = sorted(arms, key=lambda arm: raw_extra[arm] - math.floor(raw_extra[arm]), reverse=True)
        for arm in order[:leftover]:
            counts[arm] += 1
        return counts

    def comparison_rows(counts: dict[float, int]) -> list[dict]:
        rows = []
        for role, metric in metrics.items():
            if design.sample_size_basis == "Primary Metric Only" and role != "Primary":
                continue
            stats = arm_stats_by_role.get(role, pd.DataFrame())
            control_row = _arm_stat_row(stats, control)
            if control_row is None:
                raise ValueError(f"{metric.name}: no historical support is available for control point {control:,.0f}.")
            baseline = float(control_row["Historical Mean"])
            effect = abs(float(metric.effect_value) * baseline) if metric.effect_type == "Relative %" else abs(float(metric.effect_value))
            for treatment in treatments:
                treatment_row = _arm_stat_row(stats, treatment)
                if treatment_row is None:
                    raise ValueError(f"{metric.name}: no historical support is available for treatment point {treatment:,.0f}.")
                control_sd = float(control_row["Historical SD"])
                treatment_sd = float(treatment_row["Historical SD"])
                if metric.metric_type == "Binary":
                    achieved_power = power_binary_unequal(
                        planning_alpha,
                        baseline,
                        effect,
                        counts[float(control)],
                        counts[float(treatment)],
                        metric.direction,
                    )
                else:
                    if not math.isfinite(control_sd) or control_sd <= 0 or not math.isfinite(treatment_sd) or treatment_sd <= 0:
                        raise ValueError(f"{metric.name}: line-specific historical SD is unavailable for {control:,.0f} versus {treatment:,.0f}.")
                    achieved_power = power_continuous_unequal(
                        planning_alpha,
                        effect,
                        control_sd,
                        treatment_sd,
                        counts[float(control)],
                        counts[float(treatment)],
                    )
                rows.append(
                    {
                        "Metric": metric.name,
                        "Role": role,
                        "Comparison": f"{control:,.0f} vs {treatment:,.0f}",
                        "Control N": counts[float(control)],
                        "Treatment N": counts[float(treatment)],
                        "Control Mean": baseline,
                        "Control SD": control_sd,
                        "Treatment SD": treatment_sd,
                        "Smallest Effect": effect,
                        "Planned Power": achieved_power,
                        "Historical Mapping": str(treatment_row["Mapping"]),
                        "Support": str(treatment_row["Support"]),
                    }
                )
        return rows

    low = 2 * len(arms)
    high = max(32, low)
    while min(row["Planned Power"] for row in comparison_rows(counts_for_total(high))) < design.target_power:
        high *= 2
        if high > 20_000_000:
            raise ValueError("Required sample size exceeds the supported planning range.")
    while low < high:
        middle = (low + high) // 2
        if min(row["Planned Power"] for row in comparison_rows(counts_for_total(middle))) >= design.target_power:
            high = middle
        else:
            low = middle + 1

    final_counts = counts_for_total(low)
    comparisons = comparison_rows(final_counts)
    binding = min(comparisons, key=lambda row: row["Planned Power"])
    allocation_rows = []
    for arm in arms:
        row = primary_rows[arm]
        allocation_rows.append(
            {
                "Strategy Point": arm,
                "Role": "Control" if arm == float(control) else "Treatment",
                "Historical N": int(row["Historical N"]),
                "Historical Mean": float(row["Historical Mean"]),
                "Historical SD": float(row["Historical SD"]),
                "Observed Line Range": str(row.get("Observed Line Range", "")),
                "Allocation Share": final_counts[arm] / low,
                "Analyzable N": final_counts[arm],
                "Mapping": str(row["Mapping"]),
                "Support": str(row["Support"]),
            }
        )
    return low, f"{binding['Metric']} — {binding['Comparison']}", comparisons, allocation_rows


def _arm_stat_row(stats: pd.DataFrame, point: float):
    if stats.empty or "Strategy Point" not in stats.columns:
        return None
    match = stats[(stats["Strategy Point"].astype(float) - float(point)).abs() < 1e-9]
    if match.empty or int(match.iloc[0].get("Historical N", 0)) < 2:
        return None
    return match.iloc[0]


def detectable_for_metric(metric: MetricConfig, alpha: float, target_power: float, n_per_arm: int) -> dict:
    assumption = normalize_metric(metric)
    if metric.metric_type == "Binary":
        effect = detectable_effect_binary(alpha, target_power, assumption.baseline, n_per_arm, metric.direction)
    else:
        effect = detectable_effect_continuous(alpha, target_power, assumption.standard_deviation or 0, n_per_arm)
    relative = effect / assumption.baseline if assumption.baseline else None
    return {"Metric": metric.name, "Role": metric.role, "Minimum Detectable Absolute Effect": effect, "Minimum Detectable Relative Lift": relative}


def expected_power(metric: MetricConfig, alpha: float, n_per_arm: int) -> float | None:
    if metric.expected_effect is None:
        return None
    assumption = normalize_metric(metric)
    effect = metric.expected_effect * assumption.baseline if metric.effect_type == "Relative %" else metric.expected_effect
    if metric.metric_type == "Binary":
        return power_binary(alpha, assumption.baseline, effect, n_per_arm, metric.direction)
    return power_continuous(alpha, effect, assumption.standard_deviation or 1e-9, n_per_arm)


# --- Arm Selection ---

def numeric_candidates(min_value: float, max_value: float, increment: float) -> list[float]:
    count = int(np.floor((max_value - min_value) / increment)) + 1
    return [float(min_value + idx * increment) for idx in range(max(count, 0)) if min_value + idx * increment <= max_value + 1e-9]


def validate_numeric_strategy(control: float, treatments: list[float], min_value: float, max_value: float, increment: float) -> list[str]:
    errors = []
    if min_value >= max_value:
        errors.append("Minimum Allowed Value must be lower than Maximum Allowed Value.")
    if increment <= 0:
        errors.append("Allowed Increment must be greater than zero.")
    all_values = [control] + treatments
    for value in all_values:
        if value < min_value or value > max_value:
            errors.append(f"{value:,.2f} is outside the allowed strategy range.")
        if increment > 0 and abs(((value - min_value) / increment) - round((value - min_value) / increment)) > 1e-6:
            errors.append(f"{value:,.2f} does not follow the allowed increment.")
    if control in treatments:
        errors.append("Control / BAU value cannot also be configured as a Treatment.")
    if len(treatments) != len(set(treatments)):
        errors.append("Duplicate Treatment values are not allowed.")
    return errors


def validate_categorical_strategy(control: str, treatments: list[str]) -> list[str]:
    clean = [item.strip() for item in treatments if item.strip()]
    errors = []
    if not control.strip():
        errors.append("Control Arm is required.")
    if not clean:
        errors.append("At least one Treatment Arm is required.")
    if control.strip() in clean:
        errors.append("Control Arm cannot also be configured as a Treatment Arm.")
    if len(clean) != len(set(clean)):
        errors.append("Duplicate Treatment Arms are not allowed.")
    return errors


def suggest_numeric_designs(
    df: pd.DataFrame,
    strategy_col: str,
    metric_col: str,
    control: float,
    min_value: float,
    max_value: float,
    increment: float,
    treatment_count: int,
    sample_size_per_arm: int | None = None,
    metric_std: float | None = None,
) -> pd.DataFrame:
    candidates = [value for value in numeric_candidates(min_value, max_value, increment) if value != control]
    if treatment_count <= 0 or len(candidates) < treatment_count:
        return pd.DataFrame()

    values = pd.to_numeric(df[strategy_col], errors="coerce")
    metric = pd.to_numeric(df[metric_col], errors="coerce")
    data = pd.DataFrame({"strategy": values, "metric": metric}).dropna()
    radius = max(increment, (max_value - min_value) * 0.05)
    scored = []
    for design in combinations(candidates, treatment_count):
        design_values = sorted(design)
        coverage_scores = [_coverage_score(data["strategy"], value, radius) for value in design_values]
        coverage = float(np.mean(coverage_scores))
        separation = _separation_score([control] + design_values, min_value, max_value)
        information = _response_information_score(data, design_values, radius)
        detectability = _detectability_score(data, control, design_values, radius, sample_size_per_arm, metric_std)
        overall = 0.32 * coverage + 0.28 * separation + 0.22 * information + 0.18 * detectability
        scored.append(
            {
                "Suggested Treatments": design_values,
                "Coverage": coverage,
                "Separation": separation,
                "Detectability": detectability,
                "Historical Information": information,
                "Overall Score": overall,
            }
        )
    return pd.DataFrame(scored).sort_values("Overall Score", ascending=False).head(3).reset_index(drop=True)


def _coverage_score(series: pd.Series, value: float, radius: float) -> float:
    if series.empty:
        return 0.25
    count = ((series >= value - radius) & (series <= value + radius)).sum()
    return float(min(1.0, count / max(25, len(series) * 0.04)))


def _separation_score(values: list[float], min_value: float, max_value: float) -> float:
    span = max(max_value - min_value, 1e-9)
    ordered = sorted(values)
    gaps = np.diff(ordered) / span
    if len(gaps) == 0:
        return 0.0
    return float(min(1.0, np.mean(gaps) * len(values) * 1.8))


def _response_information_score(data: pd.DataFrame, values: list[float], radius: float) -> float:
    if data.empty or data["metric"].std(ddof=1) == 0:
        return 0.4
    local_means = []
    for value in values:
        local = data[(data["strategy"] >= value - radius) & (data["strategy"] <= value + radius)]
        if not local.empty:
            local_means.append(local["metric"].mean())
    if len(local_means) < 2:
        return 0.35
    return float(min(1.0, np.std(local_means) / max(data["metric"].std(ddof=1), 1e-9) * 2.5))


def _detectability_score(data: pd.DataFrame, control: float, values: list[float], radius: float, sample_size_per_arm: int | None, metric_std: float | None) -> float:
    if data.empty:
        return 0.35
    control_local = data[(data["strategy"] >= control - radius) & (data["strategy"] <= control + radius)]
    if control_local.empty:
        return 0.35
    std = metric_std or data["metric"].std(ddof=1)
    if not std or std <= 0:
        return 0.5
    n_factor = min(1.0, ((sample_size_per_arm or 100) / 400) ** 0.5)
    effects = []
    for value in values:
        local = data[(data["strategy"] >= value - radius) & (data["strategy"] <= value + radius)]
        if not local.empty:
            effects.append(abs(local["metric"].mean() - control_local["metric"].mean()) / std)
    if not effects:
        return 0.35
    return float(min(1.0, np.mean(effects) * n_factor * 2.0))


# --- Historical Strategy ---

def validate_fixed_unit_value(df: pd.DataFrame, unit_column: str, value_column: str) -> list[str]:
    """Validate that a historical treatment value is fixed within each unit."""
    if not value_column:
        return []
    if unit_column not in df.columns:
        return [f"{unit_column} is required to validate the historical treatment value."]
    if value_column not in df.columns:
        return [f"{value_column} is not present in the selected dataset."]
    values = df[[unit_column, value_column]].dropna(subset=[unit_column])
    changing = values.groupby(unit_column, dropna=False)[value_column].nunique(dropna=True)
    changing_count = int((changing > 1).sum())
    errors = []
    if changing_count:
        errors.append(
            f"{value_column} must be fixed for each experimental unit; "
            f"{changing_count:,} unit{'s have' if changing_count != 1 else ' has'} multiple values."
        )
    if values[value_column].notna().sum() == 0:
        errors.append(f"{value_column} has no usable historical treatment values.")
    return errors


def fixed_unit_values(df: pd.DataFrame, unit_column: str, value_columns: list[str]) -> pd.DataFrame:
    """Return one validated row of fixed attributes per experimental unit."""
    columns = list(dict.fromkeys([unit_column] + [column for column in value_columns if column and column in df.columns]))
    values = df[columns].drop_duplicates()
    return values.drop_duplicates(unit_column, keep="first").set_index(unit_column)


def map_to_strategy_points(
    values: pd.Series,
    strategy_points: list[float],
    grouping: str = "Exact acquisition lines",
    max_distance: float = 0.0,
) -> pd.DataFrame:
    """Map historical numeric levels to configured experiment points."""
    numeric = pd.to_numeric(values, errors="coerce")
    points = np.array(sorted({float(point) for point in strategy_points}), dtype=float)
    result = pd.DataFrame({"Historical Value": numeric}, index=values.index)
    result["Strategy Point"] = np.nan
    result["Distance"] = np.nan
    if points.size == 0:
        return result

    valid = numeric.notna()
    source = numeric.loc[valid].to_numpy(dtype=float)
    exact_grouping = grouping in {"Exact acquisition lines", "Exact lines"}
    if exact_grouping:
        distances = np.abs(source[:, None] - points[None, :])
        nearest_index = distances.argmin(axis=1)
        nearest_distance = distances[np.arange(len(source)), nearest_index]
        exact = np.isclose(nearest_distance, 0.0, atol=1e-9)
        mapped = np.where(exact, points[nearest_index], np.nan)
    else:
        distances = np.abs(source[:, None] - points[None, :])
        nearest_index = distances.argmin(axis=1)
        nearest_distance = distances[np.arange(len(source)), nearest_index]
        within_limit = np.ones(len(source), dtype=bool) if max_distance <= 0 else nearest_distance <= max_distance
        mapped = np.where(within_limit, points[nearest_index], np.nan)

    result.loc[valid, "Strategy Point"] = mapped
    result.loc[valid, "Distance"] = nearest_distance
    return result


def historical_arm_statistics(
    df: pd.DataFrame,
    strategy_column: str,
    metric_column: str,
    strategy_points: list[float],
    grouping: str = "Exact acquisition lines",
    max_distance: float = 0.0,
    metric_type: str = "Continuous",
) -> pd.DataFrame:
    """Calculate customer-level outcome statistics for each proposed strategy point."""
    columns = [
        "Strategy Point",
        "Historical N",
        "Historical Mean",
        "Historical SD",
        "95% CI Lower",
        "95% CI Upper",
        "Observed Line Range",
        "Mapping",
        "Support",
    ]
    if strategy_column not in df.columns or metric_column not in df.columns or not strategy_points:
        return pd.DataFrame(columns=columns)

    mapped = map_to_strategy_points(df[strategy_column], strategy_points, grouping, max_distance)
    data = mapped.assign(Metric=pd.to_numeric(df[metric_column], errors="coerce")).dropna()
    rows = []
    for point in sorted({float(value) for value in strategy_points}):
        local = data[np.isclose(data["Strategy Point"], point)]
        outcome = local["Metric"]
        n = int(outcome.size)
        mean = float(outcome.mean()) if n else np.nan
        if metric_type == "Binary" and n:
            sd = float(np.sqrt(max(mean * (1 - mean), 0.0)))
        else:
            sd = float(outcome.std(ddof=1)) if n >= 2 else np.nan
        if n >= 2 and np.isfinite(sd):
            margin = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n))
            lower, upper = mean - margin, mean + margin
        else:
            lower, upper = np.nan, np.nan
        if n >= 50:
            support = "Strong"
        elif n >= 20:
            support = "Limited"
        else:
            support = "Insufficient"
        observed = ""
        if n:
            low = float(local["Historical Value"].min())
            high = float(local["Historical Value"].max())
            observed = f"{low:,.0f}" if np.isclose(low, high) else f"{low:,.0f} to {high:,.0f}"
        rows.append(
            {
                "Strategy Point": point,
                "Historical N": n,
                "Historical Mean": mean,
                "Historical SD": sd,
                "95% CI Lower": lower,
                "95% CI Upper": upper,
                "Observed Line Range": observed,
                "Mapping": "Exact" if grouping in {"Exact acquisition lines", "Exact lines"} else ("Business width" if grouping == "Business-defined bin width" else "Closest line"),
                "Support": support,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def historical_curve_statistics(df: pd.DataFrame, strategy_column: str, metric_column: str, metric_type: str = "Continuous") -> pd.DataFrame:
    """Summarize the configured metric at every observed historical strategy level."""
    if strategy_column not in df.columns or metric_column not in df.columns:
        return pd.DataFrame()
    data = pd.DataFrame(
        {
            "Strategy Point": pd.to_numeric(df[strategy_column], errors="coerce"),
            "Metric": pd.to_numeric(df[metric_column], errors="coerce"),
        }
    ).dropna()
    rows = []
    for point, local in data.groupby("Strategy Point", sort=True):
        outcome = local["Metric"]
        n = int(outcome.size)
        mean = float(outcome.mean())
        sd = float(np.sqrt(max(mean * (1 - mean), 0.0))) if metric_type == "Binary" else (float(outcome.std(ddof=1)) if n >= 2 else np.nan)
        margin = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n >= 2 and np.isfinite(sd) else np.nan
        rows.append(
            {
                "Strategy Point": float(point),
                "Historical Mean": mean,
                "Historical SD": sd,
                "Historical N": n,
                "95% CI Lower": mean - margin if np.isfinite(margin) else np.nan,
                "95% CI Upper": mean + margin if np.isfinite(margin) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def binned_historical_statistics(
    df: pd.DataFrame,
    strategy_column: str,
    metric_column: str,
    metric_type: str = "Continuous",
    method: str = "Automatic fine bins",
    bin_width: float = 500.0,
    bin_count: int = 20,
) -> pd.DataFrame:
    """Summarize a metric using reusable acquisition-line bins for exploration."""
    if strategy_column not in df.columns or metric_column not in df.columns:
        return pd.DataFrame()
    data = pd.DataFrame(
        {
            "Historical Value": pd.to_numeric(df[strategy_column], errors="coerce"),
            "Metric": pd.to_numeric(df[metric_column], errors="coerce"),
        }
    ).dropna()
    if data.empty:
        return pd.DataFrame()

    unique_count = int(data["Historical Value"].nunique())
    requested_bins = max(2, min(int(bin_count), unique_count))
    if method == "Fixed bin width":
        width = max(float(bin_width), 1e-9)
        data["_bin"] = np.floor(data["Historical Value"] / width + 0.5) * width
        grouped_values = [(float(center), local, center - width / 2, center + width / 2) for center, local in data.groupby("_bin", sort=True)]
    else:
        if method == "Automatic fine bins":
            requested_bins = max(6, min(30, int(round(np.sqrt(len(data))))))
            requested_bins = min(requested_bins, unique_count)
            categories = pd.cut(data["Historical Value"], bins=requested_bins, include_lowest=True, duplicates="drop")
        else:
            categories = pd.qcut(data["Historical Value"], q=requested_bins, duplicates="drop")
        data["_bin"] = categories
        grouped_values = []
        for interval, local in data.groupby("_bin", observed=True, sort=True):
            center = float(local["Historical Value"].median())
            grouped_values.append((center, local, float(interval.left), float(interval.right)))

    rows = []
    for center, local, lower_line, upper_line in grouped_values:
        outcome = local["Metric"]
        n = int(outcome.size)
        mean = float(outcome.mean())
        sd = float(np.sqrt(max(mean * (1 - mean), 0.0))) if metric_type == "Binary" else (float(outcome.std(ddof=1)) if n >= 2 else np.nan)
        margin = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n >= 2 and np.isfinite(sd) else np.nan
        rows.append(
            {
                "Strategy Point": center,
                "Historical Mean": mean,
                "Historical SD": sd,
                "Historical N": n,
                "95% CI Lower": mean - margin if np.isfinite(margin) else np.nan,
                "95% CI Upper": mean + margin if np.isfinite(margin) else np.nan,
                "Bin Range": f"{lower_line:,.0f} to {upper_line:,.0f}",
            }
        )
    return pd.DataFrame(rows).sort_values("Strategy Point").reset_index(drop=True)


# --- Raw Processing ---

@dataclass
class ProcessingResult:
    analysis_df: pd.DataFrame
    diagnostics: dict
    metric_preview: pd.DataFrame
    cpc_breakdown: pd.DataFrame


def build_historical_metric_dataset(
    raw_df: pd.DataFrame,
    selected_cpcs: list[str],
    unit_id_column: str,
    cpc_column: str,
    mob_column: str,
    booking_date_column: str,
    metric_configs: dict[str, MetricConfig],
    cutoff_date: str = "",
    strategy_column: str | None = None,
) -> ProcessingResult:
    mapping = DataMappingConfig(
        unit_id_column=unit_id_column,
        cpc_column=cpc_column,
        mob_column=mob_column,
        booking_date_column=booking_date_column,
        cutoff_date=cutoff_date,
    )
    population = PopulationConfig(selected_cpcs=selected_cpcs)
    return build_analysis_dataset(raw_df, population, metric_configs, mapping, strategy_column)


def processed_metric_name(metric: MetricConfig) -> str:
    prefix = "avg" if metric.aggregation_method == "Average" else "cum"
    base = re.sub(r"[^a-z0-9]+", "_", metric.source_column.lower()).strip("_")
    start = max(int(getattr(metric, "mob_start", 1)), 1)
    suffix = f"mob{metric.mob_horizon}" if start == 1 else f"mob{start}_{metric.mob_horizon}"
    return f"{prefix}_{base}_{suffix}"


def validate_mapping(raw_df: pd.DataFrame, mapping: DataMappingConfig) -> list[str]:
    errors = []
    required = [mapping.unit_id_column]
    if mapping.data_structure == "Longitudinal (unit x period)":
        required.append(mapping.mob_column)
    for column in required:
        if column not in raw_df.columns:
            errors.append(f"{column} is required but is not present in the raw dataset.")
    if mapping.data_structure == "Longitudinal (unit x period)" and mapping.mob_column in raw_df.columns:
        mob = safe_numeric_series(raw_df, mapping.mob_column)
        if mob.isna().all():
            errors.append("MOB Column must contain numeric month-on-book values.")
    if mapping.data_structure == "Longitudinal (unit x period)" and mapping.booking_date_column and mapping.booking_date_column not in raw_df.columns:
        errors.append("Booking Date was selected but is not present in the raw dataset.")
    return errors


def common_horizon(metrics: dict[str, MetricConfig]) -> int:
    return max((int(metric.mob_horizon) for metric in metrics.values()), default=0)


def filter_cpcs(raw_df: pd.DataFrame, mapping: DataMappingConfig, selected_cpcs: list[str]) -> pd.DataFrame:
    if not mapping.cpc_column or mapping.cpc_column not in raw_df.columns or not selected_cpcs:
        return raw_df.copy()
    return raw_df[raw_df[mapping.cpc_column].astype(str).isin([str(cpc) for cpc in selected_cpcs])].copy()


def determine_mature_units(df: pd.DataFrame, mapping: DataMappingConfig, required_horizon: int) -> pd.Index:
    if mapping.booking_date_column and mapping.booking_date_column in df.columns and mapping.cutoff_date:
        dates = pd.DataFrame(
            {
                mapping.unit_id_column: first_series(df, mapping.unit_id_column),
                mapping.booking_date_column: safe_datetime_series(df, mapping.booking_date_column),
            }
        ).dropna().drop_duplicates(mapping.unit_id_column)
        booking = dates[mapping.booking_date_column]
        cutoff = pd.Timestamp(mapping.cutoff_date)
        maturity_months = (cutoff.year - booking.dt.year) * 12 + (cutoff.month - booking.dt.month)
        return pd.Index(dates.loc[maturity_months >= required_horizon, mapping.unit_id_column])
    mob = safe_numeric_series(df, mapping.mob_column)
    max_mob = df.assign(_mob=mob).groupby(mapping.unit_id_column)["_mob"].max()
    return max_mob[max_mob >= required_horizon].index


def completeness_by_unit(
    df: pd.DataFrame,
    mapping: DataMappingConfig,
    unit_ids: pd.Index,
    required_horizon: int,
    start_mob: int = 1,
) -> pd.DataFrame:
    scoped = df[df[mapping.unit_id_column].isin(unit_ids)].copy()
    scoped["_mob"] = safe_numeric_series(scoped, mapping.mob_column)
    start_mob = max(int(start_mob), 1)
    scoped = scoped[(scoped["_mob"] >= start_mob) & (scoped["_mob"] <= required_horizon)]
    observed = scoped.drop_duplicates([mapping.unit_id_column, "_mob"]).groupby(mapping.unit_id_column)["_mob"].nunique()
    out = pd.DataFrame({mapping.unit_id_column: unit_ids})
    out["Observed MOBs"] = out[mapping.unit_id_column].map(observed).fillna(0).astype(int)
    out["Expected MOBs"] = max(required_horizon - start_mob + 1, 0)
    out["Complete"] = out["Observed MOBs"] >= out["Expected MOBs"]
    return out


def aggregate_metric_by_mob(
    df: pd.DataFrame,
    unit_column: str,
    mob_column: str,
    metric_column: str,
    aggregation_method: str,
    mob_horizon: int,
    mob_start: int = 1,
) -> pd.Series:
    scoped = pd.DataFrame(
        {
            unit_column: first_series(df, unit_column),
            mob_column: first_series(df, mob_column),
            metric_column: first_series(df, metric_column),
        }
    )
    scoped["_mob"] = safe_numeric_series(scoped, mob_column)
    scoped["_metric"] = safe_numeric_series(scoped, metric_column)
    mob_start = max(int(mob_start), 1)
    scoped = scoped[(scoped["_mob"] >= mob_start) & (scoped["_mob"] <= mob_horizon)].dropna(subset=[unit_column, "_mob"])
    monthly = scoped.groupby([unit_column, "_mob"], as_index=False)["_metric"].mean()
    if aggregation_method == "Cumulative":
        return monthly.groupby(unit_column)["_metric"].sum()
    return monthly.groupby(unit_column)["_metric"].mean()


def duplicate_unit_mob_count(raw_df: pd.DataFrame, mapping: DataMappingConfig) -> int:
    if mapping.unit_id_column not in raw_df.columns or mapping.mob_column not in raw_df.columns:
        return 0
    return int(raw_df.duplicated([mapping.unit_id_column, mapping.mob_column]).sum())


def build_analysis_dataset(
    raw_df: pd.DataFrame,
    population: PopulationConfig,
    metrics: dict[str, MetricConfig],
    mapping: DataMappingConfig,
    strategy_column: str | None = None,
) -> ProcessingResult:
    errors = validate_mapping(raw_df, mapping)
    if errors:
        raise ValueError("; ".join(errors))
    selected = filter_cpcs(raw_df, mapping, population.selected_cpcs)
    for metric in metrics.values():
        if metric.source_column not in selected.columns:
            raise ValueError(f"{metric.name}: source column {metric.source_column} is not present in the selected dataset.")
    strategy_errors = validate_fixed_unit_value(selected, mapping.unit_id_column, strategy_column) if strategy_column in selected.columns and not selected.empty else []
    grouping_column = getattr(population, "grouping_column", "")
    grouping_errors = validate_fixed_unit_value(selected, mapping.unit_id_column, grouping_column) if grouping_column in selected.columns and not selected.empty else []
    if strategy_errors or grouping_errors:
        raise ValueError("; ".join(strategy_errors + grouping_errors))

    if mapping.data_structure == "Cross-sectional (one row per unit)":
        unit_info_cols = [mapping.unit_id_column]
        if mapping.cpc_column and mapping.cpc_column in selected.columns:
            unit_info_cols.append(mapping.cpc_column)
        if strategy_column and strategy_column in selected.columns:
            unit_info_cols.append(strategy_column)
        if grouping_column and grouping_column in selected.columns:
            unit_info_cols.append(grouping_column)
        unit_info = fixed_unit_values(selected, mapping.unit_id_column, unit_info_cols[1:])
        analysis = unit_info.copy()
        metric_preview_rows = []
        for metric in metrics.values():
            metric.processed_column = processed_metric_name(metric)
            metric.column = metric.processed_column
            values = selected.groupby(mapping.unit_id_column)[metric.source_column].mean()
            analysis[metric.processed_column] = values
            clean = pd.to_numeric(values, errors="coerce").dropna()
            baseline = float(clean.mean()) if not clean.empty else float("nan")
            std = float((baseline * (1 - baseline)) ** 0.5) if metric.metric_type == "Binary" else float(clean.std(ddof=1))
            if metric.metric_type == "Binary":
                metric.baseline_rate = baseline
            else:
                metric.baseline_mean = baseline
                metric.standard_deviation = std
                metric.variance = std**2
            metric_preview_rows.append({"Metric": metric.name, "Role": metric.role, "Aggregation": "One value per unit", "Observation Window": "As observed", "Baseline": baseline, "SD": std, "Eligible N": int(clean.size)})
        if mapping.cpc_column and mapping.cpc_column in analysis.columns:
            cpc_counts = analysis.reset_index().groupby(mapping.cpc_column)[mapping.unit_id_column].nunique().reset_index(name="Eligible Units")
            cpc_counts["Share"] = cpc_counts["Eligible Units"] / cpc_counts["Eligible Units"].sum()
        else:
            cpc_counts = pd.DataFrame()
        diagnostics = {
            "Raw Records": len(raw_df),
            "Unique Accounts": raw_df[mapping.unit_id_column].nunique(),
            "Selected CPC Accounts": selected[mapping.unit_id_column].nunique(),
            "Longest Required Horizon": 0,
            "Mature Accounts": selected[mapping.unit_id_column].nunique(),
            "Complete Eligible Accounts": len(analysis),
            "Final Analysis Population": len(analysis),
            "Excluded for Insufficient Maturity": 0,
            "Excluded for Data Completeness": 0,
            "Duplicate Unit-MOB Rows": int(selected.duplicated(mapping.unit_id_column).sum()),
            "Units Missing Historical Strategy": int(unit_info[strategy_column].isna().sum()) if strategy_column and strategy_column in unit_info.columns else 0,
        }
        return ProcessingResult(analysis.reset_index(), diagnostics, pd.DataFrame(metric_preview_rows), cpc_counts)

    required_horizon = common_horizon(metrics)
    population.common_mob_horizon = required_horizon
    mature_units = determine_mature_units(selected, mapping, required_horizon)
    completeness = completeness_by_unit(selected, mapping, mature_units, required_horizon)
    complete_units = pd.Index(completeness.loc[completeness["Complete"], mapping.unit_id_column])
    unit_info_cols = [mapping.unit_id_column]
    if mapping.cpc_column and mapping.cpc_column in selected.columns:
        unit_info_cols.append(mapping.cpc_column)
    if strategy_column and strategy_column in selected.columns:
        unit_info_cols.append(strategy_column)
    if grouping_column and grouping_column in selected.columns:
        unit_info_cols.append(grouping_column)
    unit_info = fixed_unit_values(selected, mapping.unit_id_column, unit_info_cols[1:])
    metric_preview_rows = []
    metric_values: dict[str, pd.Series] = {}
    eligible_units: set = set()
    primary_eligible_n = 0
    for metric in metrics.values():
        metric.processed_column = processed_metric_name(metric)
        metric.column = metric.processed_column
        metric_mature_units = determine_mature_units(selected, mapping, int(metric.mob_horizon))
        metric_start = max(int(getattr(metric, "mob_start", 1)), 1)
        metric_completeness = completeness_by_unit(selected, mapping, metric_mature_units, int(metric.mob_horizon), metric_start)
        metric_complete_units = pd.Index(metric_completeness.loc[metric_completeness["Complete"], mapping.unit_id_column])
        aggregated = aggregate_metric_by_mob(
            selected[selected[mapping.unit_id_column].isin(metric_complete_units)],
            mapping.unit_id_column,
            mapping.mob_column,
            metric.source_column,
            metric.aggregation_method,
            metric.mob_horizon,
            metric_start,
        )
        metric_values[metric.processed_column] = aggregated
        eligible_units.update(aggregated.index.tolist())
        clean = pd.to_numeric(aggregated, errors="coerce").dropna()
        baseline = float(clean.mean()) if not clean.empty else float("nan")
        std = float((baseline * (1 - baseline)) ** 0.5) if metric.metric_type == "Binary" else float(clean.std(ddof=1))
        if metric.metric_type == "Binary":
            metric.baseline_rate = baseline
        else:
            metric.baseline_mean = baseline
            metric.standard_deviation = std
            metric.variance = std**2
        if metric.role == "Primary":
            primary_eligible_n = int(clean.size)
        metric_preview_rows.append(
            {
                "Metric": metric.name,
                "Role": metric.role,
                "Aggregation": metric.aggregation_method,
                "Observation Window": (
                    f"Through MOB {metric.mob_horizon}"
                    if metric_start == 1
                    else f"MOB {metric_start} to {metric.mob_horizon}"
                ),
                "Baseline": baseline,
                "SD": std,
                "Eligible N": int(clean.size),
            }
        )
    analysis_index = unit_info.index.intersection(pd.Index(list(eligible_units)))
    analysis = unit_info.loc[analysis_index].copy()
    analysis.index.name = mapping.unit_id_column
    for column, values in metric_values.items():
        analysis[column] = values.reindex(analysis.index)
    if mapping.cpc_column and mapping.cpc_column in analysis.columns:
        cpc_counts = analysis.reset_index().groupby(mapping.cpc_column)[mapping.unit_id_column].nunique().reset_index(name="Eligible Units")
        if not cpc_counts.empty:
            cpc_counts["Share"] = cpc_counts["Eligible Units"] / cpc_counts["Eligible Units"].sum()
    else:
        cpc_counts = pd.DataFrame()
    diagnostics = {
        "Raw Records": len(raw_df),
        "Unique Accounts": raw_df[mapping.unit_id_column].nunique(),
        "Selected CPC Accounts": selected[mapping.unit_id_column].nunique(),
        "Longest Required Horizon": required_horizon,
        "Mature Accounts": len(mature_units),
        "Complete Eligible Accounts": len(complete_units),
        "Final Analysis Population": primary_eligible_n,
        "Any Metric Eligible Units": len(analysis),
        "Excluded for Insufficient Maturity": selected[mapping.unit_id_column].nunique() - len(mature_units),
        "Excluded for Data Completeness": len(mature_units) - len(complete_units),
        "Duplicate Unit-MOB Rows": duplicate_unit_mob_count(selected, mapping),
        "Units Missing Historical Strategy": int(unit_info[strategy_column].isna().sum()) if strategy_column and strategy_column in unit_info.columns else 0,
    }
    return ProcessingResult(analysis.reset_index(), diagnostics, pd.DataFrame(metric_preview_rows), cpc_counts)


# --- Customer Planning ---

def build_customer_preview(
    data: pd.DataFrame,
    unit_col: str,
    outcome_col: str,
    product_col: str = "",
    group_col: str = "",
    strategy_col: str = "",
    date_col: str = "",
    period_col: str = "",
) -> dict:
    """Aggregate longitudinal customer history into one presentation row per unit."""
    if unit_col not in data or outcome_col not in data:
        raise ValueError("Customer preview requires valid unit and outcome columns.")
    source = data.dropna(subset=[unit_col]).copy()
    source[outcome_col] = pd.to_numeric(source[outcome_col], errors="coerce")
    aggregations: dict[str, tuple[str, str]] = {"Average Outcome": (outcome_col, "mean")}
    if product_col and product_col in source:
        aggregations["Product / Brand"] = (product_col, "first")
    if group_col and group_col in source:
        aggregations["Customer Group"] = (group_col, "first")
    if strategy_col and strategy_col in source:
        aggregations["Historical Strategy"] = (strategy_col, "first")
    if date_col and date_col in source:
        source[date_col] = pd.to_datetime(source[date_col], errors="coerce")
        aggregations["Customer Start"] = (date_col, "min")
    if period_col and period_col in source:
        aggregations["Periods"] = (period_col, "nunique")
    accounts = source.groupby(unit_col, as_index=False).agg(**aggregations).rename(columns={unit_col: "Customer / Account"})
    product_counts = (
        accounts["Product / Brand"].astype(str).value_counts().rename_axis("Product / Brand").reset_index(name="Customers / Accounts")
        if "Product / Brand" in accounts else pd.DataFrame(columns=["Product / Brand", "Customers / Accounts"])
    )
    group_counts = (
        accounts["Customer Group"].astype(str).value_counts().rename_axis("Customer Group").reset_index(name="Customers / Accounts")
        if "Customer Group" in accounts else pd.DataFrame(columns=["Customer Group", "Customers / Accounts"])
    )
    starts = accounts["Customer Start"].dropna() if "Customer Start" in accounts else pd.Series(dtype="datetime64[ns]")
    return {
        "accounts": accounts,
        "source_rows": len(data),
        "unique_accounts": len(accounts),
        "product_counts": product_counts,
        "group_counts": group_counts,
        "product_count": int(accounts["Product / Brand"].nunique()) if "Product / Brand" in accounts else 0,
        "group_count": int(accounts["Customer Group"].nunique()) if "Customer Group" in accounts else 0,
        "start_date": starts.min() if not starts.empty else pd.NaT,
        "end_date": starts.max() if not starts.empty else pd.NaT,
    }


def default_numeric_groups(values: pd.Series, column_name: str) -> list[dict]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return []
    minimum, maximum = float(numeric.min()), float(numeric.max())
    if "fico" in column_name.lower() and minimum < 660 < 720 < maximum:
        edges = [minimum, 660.0, 720.0, maximum]
    else:
        quantiles = numeric.quantile([0, 1 / 3, 2 / 3, 1]).to_numpy(dtype=float)
        edges = list(dict.fromkeys(float(value) for value in quantiles))
        if len(edges) < 3:
            return [{"label": f"{minimum:,.0f} to {maximum:,.0f}", "lower": minimum, "upper": maximum}]
    definitions = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == 0 and "fico" in column_name.lower() and math.isclose(upper, 660.0):
            label = "Below 660"
        elif index == len(edges) - 2 and "fico" in column_name.lower() and math.isclose(lower, 720.0):
            label = "720+"
        else:
            label = f"{lower:,.0f} to {upper - 1 if float(upper).is_integer() else upper:,.0f}"
        definitions.append({"label": label, "lower": lower, "upper": upper})
    return definitions


def assign_group_labels(values: pd.Series, definitions: list[dict]) -> pd.Series:
    if not definitions:
        return pd.Series("All customers", index=values.index, dtype="object")
    if "value" in definitions[0]:
        lookup = {str(item["value"]): str(item["label"]) for item in definitions}
        return values.astype(str).map(lookup)
    numeric = pd.to_numeric(values, errors="coerce")
    labels = pd.Series(pd.NA, index=values.index, dtype="object")
    for index, item in enumerate(definitions):
        lower, upper = float(item["lower"]), float(item["upper"])
        mask = numeric.ge(lower) & (numeric.le(upper) if index == len(definitions) - 1 else numeric.lt(upper))
        labels.loc[mask] = str(item["label"])
    return labels


def candidate_lines(values: pd.Series, minimum: float, maximum: float, control: float, limit: int = 9) -> list[float]:
    observed = pd.to_numeric(values, errors="coerce").dropna()
    observed = observed[(observed >= minimum) & (observed <= maximum)]
    unique = np.sort(observed.unique().astype(float))
    if unique.size > limit:
        positions = np.linspace(0, unique.size - 1, limit).round().astype(int)
        unique = unique[positions]
    candidates = sorted({float(minimum), float(maximum), *unique.tolist()})
    return [value for value in candidates if not math.isclose(value, float(control))]


def automatic_matching_distance(strategy_points: list[float]) -> float:
    """Keep nearest-point evidence local while covering the full proposed range."""
    points = np.sort(np.unique(np.asarray(strategy_points, dtype=float)))
    if len(points) < 2:
        return 0.0
    return float(np.diff(points).max() / 2)


def planning_comparison_count(treatment_count: int, audience_count: int, decision_scope: str) -> int:
    """Return the number of confirmatory comparisons protected in planning."""
    audience_multiplier = max(int(audience_count), 1) if decision_scope == "One joint decision across audiences" else 1
    return max(int(treatment_count), 1) * audience_multiplier


def calculate_group_design(
    group_df: pd.DataFrame,
    strategy_column: str,
    metrics: dict[str, MetricConfig],
    design: DesignConfig,
    control: float,
    treatments: list[float],
    eligible_flow: float,
    group_label: str,
    family_comparisons: int | None = None,
    max_mapping_distance: float | None = None,
) -> tuple[list[dict], list[dict], str]:
    points = [float(control)] + [float(value) for value in treatments]
    matching_distance = (
        automatic_matching_distance(points)
        if max_mapping_distance is None or max_mapping_distance <= 0
        else float(max_mapping_distance)
    )
    stats_by_role = {}
    for role, metric in metrics.items():
        metric_column = metric.processed_column or metric.column
        stats_by_role[role] = historical_arm_statistics(
            group_df,
            strategy_column,
            metric_column,
            points,
            grouping="Closest testing line",
            max_distance=matching_distance,
            metric_type=metric.metric_type,
        )
    total_analyzable, binding, comparisons, allocation = design_neyman_allocation(
        metrics,
        design,
        stats_by_role,
        float(control),
        [float(value) for value in treatments],
        family_comparisons=family_comparisons,
    )
    enrollment = []
    total_enrollment = 0
    for row in allocation:
        enroll_n = int(math.ceil(int(row["Analyzable N"]) / max(1 - design.attrition_rate, 1e-9)))
        total_enrollment += enroll_n
        enrollment.append((row, enroll_n))
    output = []
    for row, enroll_n in enrollment:
        share = enroll_n / max(total_enrollment, 1)
        flow = float(eligible_flow) * share
        duration = enroll_n / flow if flow > 0 else math.inf
        output.append(
            {
                "Group": group_label,
                "Arm": float(row["Strategy Point"]),
                "Role": row["Role"],
                "Required Accounts": enroll_n,
                "Traffic Allocation": share,
                "Flow per Period": flow,
                "Test Duration": duration,
                "Historical N": int(row["Historical N"]),
                "Historical Mean": float(row["Historical Mean"]),
                "Historical SD": float(row["Historical SD"]),
                "Support": str(row["Support"]),
                "Historical Match Limit": matching_distance,
            }
        )
    return output, comparisons, binding


def calculate_variant_group_design(
    group_df: pd.DataFrame,
    metrics: dict[str, MetricConfig],
    design: DesignConfig,
    control: object,
    treatments: list[object],
    eligible_flow: float,
    group_label: str,
    family_comparisons: int | None = None,
) -> tuple[list[dict], list[dict], str]:
    """Plan named or unsupported-history arms from group-level outcome assumptions."""
    arms = [control, *treatments]
    if len(arms) < 2:
        raise ValueError("At least one control and one treatment arm are required.")
    group_metrics: dict[str, MetricConfig] = {}
    support_rows: list[dict] = []
    for role, metric in metrics.items():
        metric_column = metric.processed_column or metric.column
        if metric_column not in group_df.columns:
            raise ValueError(f"{metric.name}: historical outcome column is unavailable for this customer group.")
        values = pd.to_numeric(group_df[metric_column], errors="coerce").dropna()
        if len(values) < 2:
            raise ValueError(f"{metric.name}: at least two historical outcomes are required for this customer group.")
        copied = deepcopy(metric)
        stats = metric_baseline(group_df, metric_column, metric.metric_type)
        if metric.metric_type == "Binary":
            copied.baseline_rate = float(stats["baseline"])
            historical_sd = math.sqrt(max(copied.baseline_rate * (1 - copied.baseline_rate), 0))
        else:
            copied.baseline_mean = float(stats["baseline"])
            copied.standard_deviation = float(stats["std_dev"])
            copied.variance = copied.standard_deviation**2
            historical_sd = copied.standard_deviation
        group_metrics[role] = copied
        support_rows.append(
            {
                "Metric": copied.name,
                "Role": role,
                "Historical N": int(len(values)),
                "Historical Mean": float(stats["baseline"]),
                "Historical SD": float(historical_sd),
            }
        )

    n_per_arm, binding, comparisons = design_sample_size(
        group_metrics,
        design,
        comparisons=max(int(family_comparisons or len(treatments)), 1),
    )
    enrollment_n = int(math.ceil(n_per_arm / max(1 - design.attrition_rate, 1e-9)))
    total_enrollment = enrollment_n * len(arms)
    share = 1 / len(arms)
    flow = float(eligible_flow) * share
    duration = enrollment_n / flow if flow > 0 else math.inf
    primary_support = next(row for row in support_rows if row["Role"] == "Primary")
    output = []
    for index, arm in enumerate(arms):
        output.append(
            {
                "Group": group_label,
                "Arm": arm,
                "Role": "Control" if index == 0 else "Treatment",
                "Required Accounts": enrollment_n,
                "Traffic Allocation": enrollment_n / max(total_enrollment, 1),
                "Flow per Period": flow,
                "Test Duration": duration,
                "Historical N": primary_support["Historical N"],
                "Historical Mean": primary_support["Historical Mean"],
                "Historical SD": primary_support["Historical SD"],
                "Support": "Group-level pooled history",
            }
        )
    return output, comparisons, binding


def suggest_group_designs(
    group_df: pd.DataFrame,
    strategy_column: str,
    metrics: dict[str, MetricConfig],
    design: DesignConfig,
    control: float,
    minimum: float,
    maximum: float,
    treatment_count: int,
    eligible_flow: float,
    max_periods: int,
    limit: int = 3,
    family_comparisons: int | None = None,
    max_mapping_distance: float | None = None,
) -> pd.DataFrame:
    candidates = candidate_lines(group_df[strategy_column], minimum, maximum, control)
    if treatment_count <= 0 or len(candidates) < treatment_count:
        return pd.DataFrame()
    historical_values = pd.to_numeric(group_df[strategy_column], errors="coerce").dropna().to_numpy(dtype=float)
    preliminary = []
    for proposed in combinations(candidates, treatment_count):
        points = np.array(sorted([float(control), *[float(value) for value in proposed]]), dtype=float)
        if historical_values.size:
            distances = np.abs(historical_values[:, None] - points[None, :])
            nearest = distances.argmin(axis=1)
            nearest_distance = distances[np.arange(len(historical_values)), nearest]
            matching_distance = (
                automatic_matching_distance(points.tolist())
                if max_mapping_distance is None or max_mapping_distance <= 0
                else float(max_mapping_distance)
            )
            supported = nearest_distance <= matching_distance
            counts = np.bincount(nearest[supported], minlength=len(points))
            minimum_count = int(counts.min())
        else:
            minimum_count = 0
        preliminary.append((proposed, minimum_count, float(points.max() - points.min())))
    preliminary.sort(key=lambda item: (item[1], item[2]), reverse=True)
    rows = []
    for proposed, _, _ in preliminary[:6]:
        try:
            plan, _, binding = calculate_group_design(
                group_df,
                strategy_column,
                metrics,
                design,
                control,
                list(proposed),
                eligible_flow,
                "",
                family_comparisons=family_comparisons,
                max_mapping_distance=max_mapping_distance,
            )
        except ValueError:
            continue
        duration = max(row["Test Duration"] for row in plan)
        total = sum(row["Required Accounts"] for row in plan)
        minimum_support = min(row["Historical N"] for row in plan)
        span = max([control, *proposed]) - min([control, *proposed])
        rows.append(
            {
                "Suggested Treatments": list(proposed),
                "Required Accounts": total,
                "Test Duration": duration,
                "Minimum Historical N": minimum_support,
                "Binding Metric": binding,
                "Within Window": duration <= max_periods,
                "_span": span,
            }
        )
    if not rows:
        return pd.DataFrame()
    ranked = pd.DataFrame(rows).sort_values(
        ["Within Window", "Test Duration", "Minimum Historical N", "_span"],
        ascending=[False, True, False, False],
    )
    return ranked.head(limit).drop(columns=["_span"]).reset_index(drop=True)


# --- Customer Templates ---

CUSTOMER_EXPERIMENT_TEMPLATES: dict[str, dict[str, Any]] = {
    "Acquisition Credit Line": {
        "summary": "Compare credit lines assigned when a new account is opened.",
        "examples": "3,000 vs 5,000 vs 8,000 acquisition lines",
        "arm_format": "Ordered numeric levels",
        "strategy_name": "Acquisition Credit Line",
        "strategy_goal": "Optimize the acquisition credit line offered to eligible customers.",
        "value_label": "Credit Line",
        "value_format": "Currency",
        "assignment_column": "assigned_credit_line",
        "historical_column": "current_credit_line",
        "historical_evidence": "Use historical strategy values",
        "control_name": "BAU",
        "control_value": 5000.0,
        "treatment_names": ["Lower Line", "Higher Line"],
        "treatment_values": [3000.0, 8000.0],
        "minimum": 2000.0,
        "maximum": 10000.0,
        "increment": 500.0,
    },
    "Proactive Credit Line Increase": {
        "summary": "Test proactive line-increase policies for existing customers.",
        "examples": "No increase vs +10% vs +25%, or fixed-dollar increase policies",
        "arm_format": "Ordered numeric levels",
        "strategy_name": "Proactive Line Increase",
        "strategy_goal": "Measure the incremental value and risk of proactive credit line increases.",
        "value_label": "Line Increase",
        "value_format": "Percent",
        "assignment_column": "assigned_line_increase_pct",
        "historical_column": "",
        "historical_evidence": "Use group-level outcome history",
        "control_name": "No Increase",
        "control_value": 0.0,
        "treatment_names": ["Moderate Increase", "High Increase"],
        "treatment_values": [10.0, 25.0],
        "minimum": 0.0,
        "maximum": 50.0,
        "increment": 5.0,
    },
    "Multiple Offer Strategy": {
        "summary": "Compare complete offer packages that may differ on several attributes.",
        "examples": "BAU vs cash bonus vs APR offer vs rewards bundle",
        "arm_format": "Named variants",
        "strategy_name": "Offer Strategy",
        "strategy_goal": "Select the offer package with the best incremental customer and business outcome.",
        "assignment_column": "assigned_offer",
        "historical_column": "",
        "historical_evidence": "Use group-level outcome history",
        "control": "BAU Offer",
        "treatments": ["Cash Bonus", "APR Offer", "Rewards Bundle"],
        "descriptions": {
            "BAU Offer": "Current standard offer",
            "Cash Bonus": "One-time acquisition or activation bonus",
            "APR Offer": "Promotional interest-rate offer",
            "Rewards Bundle": "Enhanced rewards and benefits package",
        },
    },
    "Pricing / Fee": {
        "summary": "Test ordered prices, annual fees, or promotional rates.",
        "examples": "0 vs 49 vs 95 annual fee",
        "arm_format": "Ordered numeric levels",
        "strategy_name": "Price or Fee",
        "strategy_goal": "Find the price or fee that maximizes risk-adjusted value.",
        "value_label": "Price / Fee",
        "value_format": "Currency",
        "assignment_column": "assigned_price",
        "historical_column": "",
        "historical_evidence": "Use group-level outcome history",
        "control_name": "Current Price",
        "control_value": 10.0,
        "treatment_names": ["Lower", "Higher"],
        "treatment_values": [5.0, 15.0],
        "minimum": 0.0,
        "maximum": 100.0,
        "increment": 1.0,
    },
    "Retention / Save Offer": {
        "summary": "Compare interventions intended to retain an at-risk customer.",
        "examples": "Standard outreach vs fee waiver vs points vs specialist call",
        "arm_format": "Named variants",
        "strategy_name": "Retention Strategy",
        "strategy_goal": "Increase retained value while controlling incentive cost and adverse selection.",
        "assignment_column": "assigned_retention_strategy",
        "historical_column": "",
        "historical_evidence": "Use group-level outcome history",
        "control": "Standard Outreach",
        "treatments": ["Fee Waiver", "Bonus Points", "Specialist Call"],
    },
    "Rewards / Incentive": {
        "summary": "Compare rewards, bonuses, or spend incentives.",
        "examples": "No bonus vs statement credit vs points multiplier",
        "arm_format": "Named variants",
        "strategy_name": "Rewards Strategy",
        "strategy_goal": "Identify the incentive that creates the greatest incremental value net of cost.",
        "assignment_column": "assigned_reward",
        "historical_column": "",
        "historical_evidence": "Use group-level outcome history",
        "control": "Current Rewards",
        "treatments": ["Statement Credit", "Points Multiplier"],
    },
    "Channel / Message": {
        "summary": "Compare contact channels, creative, cadence, or message framing.",
        "examples": "Email vs push vs SMS, or benefit-led vs urgency-led copy",
        "arm_format": "Named variants",
        "strategy_name": "Contact Strategy",
        "strategy_goal": "Choose the contact experience that improves response without increasing opt-outs or complaints.",
        "assignment_column": "assigned_contact_strategy",
        "historical_column": "",
        "historical_evidence": "Use group-level outcome history",
        "control": "Current Contact",
        "treatments": ["Email Variant", "Push Variant", "SMS Variant"],
    },
    "Card / Product Design": {
        "summary": "Compare product configurations, card designs, or benefit packages.",
        "examples": "Current card vs premium design vs eco design",
        "arm_format": "Named variants",
        "strategy_name": "Product Design",
        "strategy_goal": "Measure whether a product design changes activation, usage, satisfaction, or retention.",
        "assignment_column": "assigned_product_design",
        "historical_column": "",
        "historical_evidence": "Use group-level outcome history",
        "control": "Current Design",
        "treatments": ["Design A", "Design B"],
    },
    "Collections Treatment": {
        "summary": "Compare treatment paths for delinquent or financially stressed customers.",
        "examples": "BAU collections vs digital self-service vs payment plan",
        "arm_format": "Named variants",
        "strategy_name": "Collections Treatment",
        "strategy_goal": "Improve cure and repayment outcomes while protecting customer experience.",
        "assignment_column": "assigned_collections_treatment",
        "historical_column": "",
        "historical_evidence": "Use group-level outcome history",
        "control": "BAU Collections",
        "treatments": ["Digital Self-Service", "Payment Plan"],
    },
    "Custom": {
        "summary": "Build a custom randomized customer-level strategy test.",
        "examples": "Any mutually exclusive set of assignable strategies",
        "arm_format": "Named variants",
        "strategy_name": "Treatment",
        "strategy_goal": "",
        "assignment_column": "assigned_treatment",
        "historical_column": "",
        "historical_evidence": "Use group-level outcome history",
        "control": "Control",
        "treatments": ["Treatment A"],
        "descriptions": {
            "Control": "Current experience or business as usual",
            "Treatment A": "Proposed alternative",
        },
    },
}


def apply_customer_template(strategy: StrategyConfig, template: str) -> None:
    if template not in CUSTOMER_EXPERIMENT_TEMPLATES:
        raise ValueError(f"Unknown customer experiment template: {template}")
    strategy.experiment_template = template
    config = deepcopy(CUSTOMER_EXPERIMENT_TEMPLATES[template])
    strategy.arm_format = config["arm_format"]
    strategy.strategy_type = "Numeric Strategy" if strategy.arm_format == "Ordered numeric levels" else "Categorical Strategy"
    strategy.strategy_name = config["strategy_name"]
    strategy.strategy_goal = config["strategy_goal"]
    strategy.assignment_column = config["assignment_column"]
    strategy.historical_column = config.get("historical_column", "")
    strategy.historical_evidence = config.get("historical_evidence", "Use group-level outcome history")
    strategy.value_label = config.get("value_label", "Strategy Value")
    strategy.value_format = config.get("value_format", "Number")
    strategy.group_arms = {}
    if strategy.strategy_type == "Numeric Strategy":
        strategy.control_name = config["control_name"]
        strategy.control_value = config["control_value"]
        strategy.treatment_names = list(config["treatment_names"])
        strategy.treatment_values = list(config["treatment_values"])
        strategy.min_value = config["minimum"]
        strategy.max_value = config["maximum"]
        strategy.increment = config["increment"]
        strategy.number_of_arms = len(strategy.treatment_values) + 1
        strategy.arm_descriptions = {}
    else:
        strategy.categorical_control = config["control"]
        strategy.categorical_treatments = list(config["treatments"])
        strategy.number_of_arms = len(strategy.categorical_treatments) + 1
        strategy.arm_descriptions = dict(config.get("descriptions", {}))


# --- Analysis ---

def _apply_multiplicity(rows: list[dict], alpha: float, method: str) -> None:
    treatment_rows = [row for row in rows if not row["Is Control"]]
    if not treatment_rows:
        return
    raw = np.array([row["Raw p-value"] for row in treatment_rows], dtype=float)
    if method == "None" or len(raw) == 1:
        adjusted = raw
    else:
        adjusted = multipletests(raw, alpha=alpha, method="holm")[1]
    for row, adjusted_p in zip(treatment_rows, adjusted):
        row["Adjusted p-value"] = float(adjusted_p)
        row["p-value"] = float(adjusted_p)
        row["Significant"] = bool(adjusted_p < alpha)
        row["Result"] = "Significant" if adjusted_p < alpha else "Not significant"


def validate_analysis_data(
    df: pd.DataFrame,
    assignment_col: str,
    outcomes: dict[str, tuple[str, str]],
    control_arm,
    treatment_arms: list,
    unit_id_col: str = "",
    expected_shares: dict | None = None,
) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    if assignment_col not in df.columns:
        return ["The assignment column is not present in the result dataset."], warnings
    arms = [control_arm] + list(treatment_arms)
    if not treatment_arms:
        errors.append("Select at least one treatment arm.")
    counts = df[df[assignment_col].isin(arms)].groupby(assignment_col).size()
    for arm in arms:
        if int(counts.get(arm, 0)) < 2:
            errors.append(f"Arm {arm} needs at least two observations.")
    selected_columns = [column for column, _ in outcomes.values()]
    if len(selected_columns) != len(set(selected_columns)):
        errors.append("Each configured metric must map to a different result column.")
    for role, (column, metric_type) in outcomes.items():
        if column not in df.columns:
            errors.append(f"{role} result column is missing.")
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().sum() == 0:
            errors.append(f"{role} result column must contain numeric values.")
        for arm in arms:
            arm_values = numeric[df[assignment_col] == arm].dropna()
            if len(arm_values) < 2:
                errors.append(f"{role} needs at least two non-missing observations in arm {arm}.")
        if metric_type == "Binary" and not set(numeric.dropna().unique()).issubset({0, 1}):
            errors.append(f"{role} binary result must be coded as 0/1.")
    if not unit_id_col or unit_id_col not in df.columns:
        errors.append("Select the experimental unit ID. Customer-level analysis requires one row per randomized unit.")
    else:
        assignments_per_unit = df.groupby(unit_id_col)[assignment_col].nunique(dropna=True)
        if (assignments_per_unit > 1).any():
            errors.append("Some experimental units appear in more than one assignment arm.")
        if df[unit_id_col].duplicated().any():
            errors.append("Result data contains repeated unit IDs. Aggregate outcomes to one row per randomized unit before analysis.")
    if len(counts) == len(arms) and counts.sum() > 0:
        observed = np.array([counts.get(arm, 0) for arm in arms], dtype=float)
        if expected_shares:
            shares = np.array([float(expected_shares.get(arm, 0)) for arm in arms], dtype=float)
            shares = shares / shares.sum() if shares.sum() > 0 else np.repeat(1 / len(arms), len(arms))
        else:
            shares = np.repeat(1 / len(arms), len(arms))
        expected = observed.sum() * shares
        statistic = float(np.sum((observed - expected) ** 2 / np.maximum(expected, 1e-12)))
        srm_p = float(stats.chi2.sf(statistic, len(arms) - 1))
        if srm_p < 0.01:
            warnings.append(f"Possible sample-ratio mismatch across arms (p = {srm_p:.4f}).")
    return errors, warnings


def analysis_integrity_summary(
    df: pd.DataFrame,
    assignment_col: str,
    outcomes: dict[str, tuple[str, str]],
    arms: list,
    unit_id_col: str,
    expected_shares: dict | None = None,
) -> dict:
    scoped = df[df[assignment_col].isin(arms)].copy() if assignment_col in df else pd.DataFrame()
    arm_counts = scoped.groupby(assignment_col).size().to_dict() if not scoped.empty else {}
    missing = {
        role: int(pd.to_numeric(scoped[column], errors="coerce").isna().sum())
        for role, (column, _) in outcomes.items()
        if column in scoped
    }
    duplicate_units = int(scoped[unit_id_col].duplicated().sum()) if unit_id_col in scoped else 0
    srm_pvalue = np.nan
    expected_counts: dict = {}
    observed_shares: dict = {}
    expected_share_map: dict = {}
    if len(arm_counts) == len(arms) and sum(arm_counts.values()) > 0:
        observed = np.array([arm_counts.get(arm, 0) for arm in arms], dtype=float)
        if expected_shares:
            shares = np.array([float(expected_shares.get(arm, 0)) for arm in arms], dtype=float)
            shares = shares / shares.sum() if shares.sum() > 0 else np.repeat(1 / len(arms), len(arms))
        else:
            shares = np.repeat(1 / len(arms), len(arms))
        expected = observed.sum() * shares
        statistic = float(np.sum((observed - expected) ** 2 / np.maximum(expected, 1e-12)))
        srm_pvalue = float(stats.chi2.sf(statistic, len(arms) - 1))
        expected_counts = {arm: float(value) for arm, value in zip(arms, expected)}
        observed_shares = {arm: float(value / observed.sum()) for arm, value in zip(arms, observed)}
        expected_share_map = {arm: float(value) for arm, value in zip(arms, shares)}
    return {
        "rows": len(scoped),
        "unique_units": int(scoped[unit_id_col].nunique()) if unit_id_col in scoped else 0,
        "duplicate_units": duplicate_units,
        "arm_counts": arm_counts,
        "expected_counts": expected_counts,
        "observed_shares": observed_shares,
        "expected_shares": expected_share_map,
        "missing_by_metric": missing,
        "srm_pvalue": srm_pvalue,
        "srm_status": "Review" if pd.notna(srm_pvalue) and srm_pvalue < 0.01 else "Pass",
    }


def continuous_results(
    df: pd.DataFrame,
    arm_col: str,
    metric_col: str,
    control_arm,
    treatment_arms: list,
    alpha: float,
    multiplicity_method: str = "None",
) -> pd.DataFrame:
    control = pd.to_numeric(df.loc[df[arm_col] == control_arm, metric_col], errors="coerce").dropna()
    rows = [_summary_row(control_arm, control, True)]
    interval_alpha = alpha / max(len(treatment_arms), 1) if multiplicity_method != "None" else alpha
    for arm in treatment_arms:
        treatment = pd.to_numeric(df.loc[df[arm_col] == arm, metric_col], errors="coerce").dropna()
        test = stats.ttest_ind(treatment, control, equal_var=False, nan_policy="omit")
        effect = treatment.mean() - control.mean()
        se = np.sqrt(treatment.var(ddof=1) / len(treatment) + control.var(ddof=1) / len(control))
        dfree = _welch_df(treatment, control)
        critical = stats.t.ppf(1 - interval_alpha / 2, dfree)
        rows.append(_effect_row(arm, treatment, effect, effect / control.mean(), se, critical, test.statistic, test.pvalue, alpha))
    _apply_multiplicity(rows, alpha, multiplicity_method)
    return pd.DataFrame(rows)


def binary_results(
    df: pd.DataFrame,
    arm_col: str,
    metric_col: str,
    control_arm,
    treatment_arms: list,
    alpha: float,
    multiplicity_method: str = "None",
) -> pd.DataFrame:
    control = pd.to_numeric(df.loc[df[arm_col] == control_arm, metric_col], errors="coerce").dropna()
    rows = [_summary_row(control_arm, control, True)]
    interval_alpha = alpha / max(len(treatment_arms), 1) if multiplicity_method != "None" else alpha
    for arm in treatment_arms:
        treatment = pd.to_numeric(df.loc[df[arm_col] == arm, metric_col], errors="coerce").dropna()
        counts = np.array([treatment.sum(), control.sum()])
        nobs = np.array([len(treatment), len(control)])
        stat, pvalue = proportions_ztest(counts, nobs)
        effect = treatment.mean() - control.mean()
        se = np.sqrt(treatment.mean() * (1 - treatment.mean()) / len(treatment) + control.mean() * (1 - control.mean()) / len(control))
        critical = stats.norm.ppf(1 - interval_alpha / 2)
        rows.append(_effect_row(arm, treatment, effect, effect / control.mean(), se, critical, stat, pvalue, alpha))
    _apply_multiplicity(rows, alpha, multiplicity_method)
    return pd.DataFrame(rows)


def response_summary(df: pd.DataFrame, arm_col: str, metric_col: str, alpha: float, metric_type: str) -> pd.DataFrame:
    rows = []
    critical = stats.norm.ppf(1 - alpha / 2)
    for arm, group in df.groupby(arm_col):
        values = pd.to_numeric(group[metric_col], errors="coerce").dropna()
        estimate = values.mean()
        if metric_type == "Binary":
            se = np.sqrt(estimate * (1 - estimate) / len(values))
        else:
            se = values.std(ddof=1) / np.sqrt(len(values))
        rows.append({"Credit Line": arm, "N": len(values), "Estimate": estimate, "CI Lower": estimate - critical * se, "CI Upper": estimate + critical * se})
    return pd.DataFrame(rows).sort_values("Credit Line")


def decision_text(results: pd.DataFrame, primary_metric: str, control_arm) -> tuple[str, list[str]]:
    treatment_rows = results[~results["Is Control"]].copy()
    significant_positive = treatment_rows[(treatment_rows["Effect vs Control"] > 0) & (treatment_rows["Significant"]) & (treatment_rows["CI Lower"] > 0)]
    lines = []
    for _, row in treatment_rows.iterrows():
        arm = _format_arm(row["Credit Line"])
        control = _format_arm(control_arm)
        if row["Significant"]:
            lines.append(
                f"{arm} produced a statistically significant {row['Effect vs Control']:+,.2f} change in {primary_metric} versus the {control} Control "
                f"(95% CI: {row['CI Lower']:,.2f} to {row['CI Upper']:,.2f}, p = {row['p-value']:.3f})."
            )
        else:
            lines.append(f"{arm} did not show a statistically significant difference from the {control} Control.")
    if significant_positive.empty:
        recommendation = "No tested treatment produced a positive statistically significant effect on the selected primary metric."
    else:
        best = significant_positive.sort_values("Effect vs Control", ascending=False).iloc[0]
        recommendation = f"Recommended Test Outcome: {_format_arm(best['Credit Line'])} had the largest positive statistically significant effect among the tested credit lines."
    return recommendation, lines


def _summary_row(arm, values: pd.Series, is_control: bool) -> dict:
    return {
        "Credit Line": arm,
        "Is Control": is_control,
        "N": len(values),
        "Mean / Rate": values.mean(),
        "Effect vs Control": 0.0 if is_control else np.nan,
        "Relative Lift": 0.0 if is_control else np.nan,
        "CI Lower": np.nan,
        "CI Upper": np.nan,
        "Statistic": np.nan,
        "p-value": np.nan,
        "Raw p-value": np.nan,
        "Adjusted p-value": np.nan,
        "Significant": False,
        "Result": "Control",
    }


def _effect_row(arm, values, effect, lift, se, critical, statistic, pvalue, alpha) -> dict:
    significant = bool(pvalue < alpha)
    return {
        "Credit Line": arm,
        "Is Control": False,
        "N": len(values),
        "Mean / Rate": values.mean(),
        "Effect vs Control": effect,
        "Relative Lift": lift,
        "CI Lower": effect - critical * se,
        "CI Upper": effect + critical * se,
        "Statistic": statistic,
        "p-value": pvalue,
        "Raw p-value": pvalue,
        "Adjusted p-value": pvalue,
        "Significant": significant,
        "Result": "Significant" if significant else "Not significant",
    }


def _welch_df(a: pd.Series, b: pd.Series) -> float:
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    numerator = (va / na + vb / nb) ** 2
    denominator = (va**2 / (na**2 * (na - 1))) + (vb**2 / (nb**2 * (nb - 1)))
    return numerator / denominator


def _format_arm(value) -> str:
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


# --- Decision ---

def guardrail_status(row: pd.Series, threshold: float | None, direction: str) -> str:
    if threshold is None:
        return "REVIEW"
    effect = row["Effect vs Control"]
    if direction == "Lower is Better":
        if effect > threshold:
            return "FAIL"
        upper = row.get("CI Upper")
        return "REVIEW" if pd.notna(upper) and upper > threshold else "PASS"
    if effect < -abs(threshold):
        return "FAIL"
    lower = row.get("CI Lower")
    return "REVIEW" if pd.notna(lower) and lower < -abs(threshold) else "PASS"


def experiment_recommendation(
    primary: pd.DataFrame,
    guardrail: pd.DataFrame | None,
    guardrail_threshold: float | None,
    guardrail_direction: str,
    primary_direction: str = "Higher is Better",
) -> str:
    treatments = primary[~primary["Is Control"]].copy()
    if primary_direction == "Lower is Better":
        beneficial = (treatments["Effect vs Control"] < 0) & (treatments["CI Upper"] < 0)
        treatments["Benefit Score"] = -treatments["Effect vs Control"]
    else:
        beneficial = (treatments["Effect vs Control"] > 0) & (treatments["CI Lower"] > 0)
        treatments["Benefit Score"] = treatments["Effect vs Control"]
    candidates = treatments[beneficial & treatments["Significant"]]
    if candidates.empty:
        return "No treatment produced a beneficial statistically significant primary-metric result. Do not scale based on this experiment."
    best = candidates.sort_values("Benefit Score", ascending=False).iloc[0]
    if guardrail is None or guardrail.empty:
        return f"Promising result: {best['Credit Line']} improved the primary outcome, but no safety outcome was configured. Complete a risk review before rollout."
    if guardrail is not None and not guardrail.empty:
        g = guardrail[guardrail["Credit Line"] == best["Credit Line"]]
        if not g.empty:
            status = guardrail_status(g.iloc[0], guardrail_threshold, guardrail_direction)
            if status == "FAIL":
                return "Primary metric improved, but the configured guardrail threshold was breached. Do not recommend scaling."
            if status == "REVIEW":
                return "Primary metric improved, but guardrail uncertainty includes the configured harm threshold. Review before scaling."
    return f"Recommended Strategy: {best['Credit Line']}. Proceed for the evaluated population if operational review confirms readiness."


def arm_decision_scorecard(
    primary: pd.DataFrame,
    primary_direction: str,
    meaningful_effect: float,
    guardrail: pd.DataFrame | None = None,
    guardrail_threshold: float | None = None,
    guardrail_direction: str = "Lower is Better",
) -> pd.DataFrame:
    """Summarize evidence, practical magnitude, and safety for every tested arm."""
    rows = []
    for _, result in primary.loc[~primary["Is Control"]].iterrows():
        effect = float(result["Effect vs Control"])
        beneficial = effect < 0 if primary_direction == "Lower is Better" else effect > 0
        clears_zero = result["CI Upper"] < 0 if primary_direction == "Lower is Better" else result["CI Lower"] > 0
        evidence = "Clear benefit" if beneficial and clears_zero and bool(result["Significant"]) else "Possible benefit" if beneficial else "No benefit"
        practical = abs(effect) >= abs(meaningful_effect)
        safety = "Not configured"
        if guardrail is not None and not guardrail.empty:
            matched = guardrail[guardrail["Credit Line"] == result["Credit Line"]]
            if not matched.empty:
                safety = guardrail_status(matched.iloc[0], guardrail_threshold, guardrail_direction).title()
        if safety == "Fail":
            recommendation = "Do not scale"
        elif evidence == "Clear benefit" and practical and safety == "Pass":
            recommendation = "Scale candidate"
        elif evidence == "Clear benefit" and practical and safety == "Not configured":
            recommendation = "Promising; add safety check"
        elif evidence == "Clear benefit":
            recommendation = "Review magnitude or safety"
        else:
            recommendation = "Keep control"
        rows.append(
            {
                "Treatment Arm": result["Credit Line"],
                "Decision": recommendation,
                "Primary Evidence": evidence,
                "Guardrail": safety,
                "Primary Effect": effect,
                "Relative Lift": result.get("Relative Lift", float("nan")),
                "Adjusted p-value": result.get("Adjusted p-value", result.get("p-value")),
                "Meets Planned Effect": practical,
            }
        )
    return pd.DataFrame(rows)


def recommendation_status(message: str) -> str:
    lower = message.lower()
    if "guardrail threshold was breached" in lower:
        return "error"
    if "review before scaling" in lower:
        return "warning"
    if "no safety outcome" in lower or "promising result" in lower:
        return "warning"
    if "no treatment produced" in lower or "do not scale" in lower:
        return "warning"
    return "success"


def project_rollout_impact(
    result: pd.Series,
    rollout_population: int,
    primary_direction: str = "Higher is Better",
    value_per_outcome_unit: float = 1.0,
    implementation_cost: float = 0.0,
) -> dict[str, float]:
    """Project a measured per-unit effect to a user-supplied rollout population."""
    population = max(int(rollout_population), 0)
    direction = -1.0 if primary_direction == "Lower is Better" else 1.0
    effect = direction * float(result["Effect vs Control"]) * population
    interval = sorted(
        [
            direction * float(result["CI Lower"]) * population,
            direction * float(result["CI Upper"]) * population,
        ]
    )
    value = max(float(value_per_outcome_unit), 0.0)
    cost = max(float(implementation_cost), 0.0)
    return {
        "population": population,
        "outcome_impact": effect,
        "outcome_lower": interval[0],
        "outcome_upper": interval[1],
        "gross_value": effect * value,
        "net_value": effect * value - cost,
        "net_lower": interval[0] * value - cost,
        "net_upper": interval[1] * value - cost,
    }


# --- Design ---

def design_candidate_lines(min_line: int, max_line: int, increment: int) -> np.ndarray:
    return np.arange(min_line, max_line + increment, increment, dtype=int)


def portfolio_summary(df: pd.DataFrame, line_col: str, metric_col: str) -> dict[str, float]:
    line = pd.to_numeric(df[line_col], errors="coerce").dropna()
    metric = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    return {
        "rows": len(df),
        "line_min": float(line.min()),
        "line_p25": float(line.quantile(0.25)),
        "line_median": float(line.median()),
        "line_p75": float(line.quantile(0.75)),
        "line_max": float(line.max()),
        "metric_mean": float(metric.mean()),
        "metric_std": float(metric.std(ddof=1)),
        "metric_median": float(metric.median()),
    }


def suggest_test_lines(
    df: pd.DataFrame,
    line_col: str,
    metric_col: str,
    min_line: int,
    max_line: int,
    increment: int,
    arms: int,
) -> pd.DataFrame:
    candidates = design_candidate_lines(min_line, max_line, increment)
    line_values = pd.to_numeric(df[line_col], errors="coerce").dropna()
    metric_values = pd.to_numeric(df[metric_col], errors="coerce")
    valid = pd.DataFrame({"line": line_values, "metric": metric_values}).dropna()
    if valid.empty:
        selected = np.linspace(min_line, max_line, arms)
        selected = np.array([candidates[np.abs(candidates - value).argmin()] for value in selected])
    else:
        q_targets = np.linspace(0.2, 0.8, arms)
        historical_targets = np.quantile(valid["line"], q_targets)
        range_targets = np.linspace(min_line, max_line, arms + 2)[1:-1]
        blended = 0.6 * historical_targets + 0.4 * range_targets
        selected = []
        min_gap = max(increment, (max_line - min_line) / max(arms + 1, 2) * 0.45)
        for target in blended:
            ranked = sorted(candidates, key=lambda c: (abs(c - target), c))
            for candidate in ranked:
                if all(abs(candidate - existing) >= min_gap for existing in selected):
                    selected.append(int(candidate))
                    break
        if len(selected) < arms:
            for candidate in candidates:
                if candidate not in selected:
                    selected.append(int(candidate))
                if len(selected) == arms:
                    break
        selected = np.array(sorted(selected[:arms]))

    rows = []
    labels = ["Low", "Mid", "High"] if arms == 3 else [f"Arm {idx}" for idx in range(1, arms + 1)]
    for idx, line in enumerate(selected):
        radius = max(increment, (max_line - min_line) * 0.08)
        coverage = int(((line_values >= line - radius) & (line_values <= line + radius)).sum())
        pct = coverage / max(len(line_values), 1)
        local = valid[(valid["line"] >= line - radius) & (valid["line"] <= line + radius)]
        local_metric = float(local["metric"].mean()) if not local.empty else np.nan
        if idx == 0:
            rationale = "Lower feasible range with sufficient portfolio coverage"
        elif idx == len(selected) - 1:
            rationale = "Upper feasible range with sufficient separation and historical support"
        else:
            rationale = "Near the center of the historical credit-line distribution"
        rows.append(
            {
                "Arm": labels[idx],
                "Credit Line": int(line),
                "Nearby Historical Customers": coverage,
                "Coverage Share": pct,
                "Nearby Metric Mean": local_metric,
                "Rationale": rationale,
            }
        )
    return pd.DataFrame(rows)
