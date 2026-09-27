from __future__ import annotations

from itertools import combinations
import math

import numpy as np
import pandas as pd

from .assumptions import design_neyman_allocation
from .historical_strategy import historical_arm_statistics
from .models import DesignConfig, MetricConfig


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


def calculate_group_design(
    group_df: pd.DataFrame,
    strategy_column: str,
    metrics: dict[str, MetricConfig],
    design: DesignConfig,
    control: float,
    treatments: list[float],
    eligible_flow: float,
    group_label: str,
) -> tuple[list[dict], list[dict], str]:
    points = [float(control)] + [float(value) for value in treatments]
    stats_by_role = {}
    for role, metric in metrics.items():
        metric_column = metric.processed_column or metric.column
        stats_by_role[role] = historical_arm_statistics(
            group_df,
            strategy_column,
            metric_column,
            points,
            grouping="Closest testing line",
            metric_type=metric.metric_type,
        )
    total_analyzable, binding, comparisons, allocation = design_neyman_allocation(
        metrics,
        design,
        stats_by_role,
        float(control),
        [float(value) for value in treatments],
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
                "Line": float(row["Strategy Point"]),
                "Role": row["Role"],
                "Required Accounts": enroll_n,
                "Traffic Allocation": share,
                "Flow per Period": flow,
                "Test Duration": duration,
                "Historical N": int(row["Historical N"]),
                "Historical Mean": float(row["Historical Mean"]),
                "Historical SD": float(row["Historical SD"]),
                "Support": str(row["Support"]),
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
) -> pd.DataFrame:
    candidates = candidate_lines(group_df[strategy_column], minimum, maximum, control)
    if treatment_count <= 0 or len(candidates) < treatment_count:
        return pd.DataFrame()
    historical_values = pd.to_numeric(group_df[strategy_column], errors="coerce").dropna().to_numpy(dtype=float)
    preliminary = []
    for proposed in combinations(candidates, treatment_count):
        points = np.array(sorted([float(control), *[float(value) for value in proposed]]), dtype=float)
        if historical_values.size:
            nearest = np.abs(historical_values[:, None] - points[None, :]).argmin(axis=1)
            counts = np.bincount(nearest, minlength=len(points))
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
