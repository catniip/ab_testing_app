from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t


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
