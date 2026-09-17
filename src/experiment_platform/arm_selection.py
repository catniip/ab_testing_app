from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


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
