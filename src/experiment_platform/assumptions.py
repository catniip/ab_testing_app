from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .metrics import metric_baseline
from .models import DesignConfig, MetricConfig
from .power import (
    adjusted_alpha,
    detectable_effect_binary,
    detectable_effect_continuous,
    power_binary,
    power_binary_unequal,
    power_continuous,
    power_continuous_unequal,
    sample_size_binary,
    sample_size_continuous,
    sample_size_continuous_unequal,
)


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
) -> tuple[int, str, list[dict], list[dict]]:
    """Solve a generalized Neyman allocation for a shared-control multi-arm design."""
    arms = [float(control)] + [float(value) for value in treatments]
    primary_stats = arm_stats_by_role.get("Primary", pd.DataFrame())
    primary_rows = {arm: _arm_stat_row(primary_stats, arm) for arm in arms}
    missing = [arm for arm, row in primary_rows.items() if row is None]
    if missing:
        formatted = ", ".join(f"{value:,.0f}" for value in missing)
        raise ValueError(f"Primary metric historical support is unavailable for: {formatted}.")

    comparison_count = max(len(treatments), 1)
    raw_weights = {}
    for arm, row in primary_rows.items():
        sd = float(row["Historical SD"])
        if not math.isfinite(sd) or sd <= 0:
            raise ValueError(f"Primary metric historical SD is unavailable for {arm:,.0f}.")
        raw_weights[arm] = sd * (math.sqrt(comparison_count) if arm == float(control) else 1.0)
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
