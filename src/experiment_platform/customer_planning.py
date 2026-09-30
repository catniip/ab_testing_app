from __future__ import annotations

from itertools import combinations
import math
from copy import deepcopy

import numpy as np
import pandas as pd

from .assumptions import design_neyman_allocation
from .assumptions import design_sample_size
from .historical_strategy import historical_arm_statistics
from .metrics import metric_baseline
from .models import DesignConfig, MetricConfig


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
