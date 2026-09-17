from __future__ import annotations

import pandas as pd

from .data_validation import first_series, infer_schema, safe_numeric_series


def column_schema(df: pd.DataFrame) -> pd.DataFrame:
    schema = infer_schema(df)
    schema["Suggested Metric Type"] = [suggest_metric_type(df, col) for col in schema["Column"]]
    return schema


def suggest_metric_type(df: pd.DataFrame, column: str) -> str:
    series = first_series(df, column)
    if series is None:
        return "Not Numeric"
    values = series.dropna()
    unique = set(values.unique().tolist())
    if len(unique) == 2 and unique.issubset({0, 1, 0.0, 1.0, False, True}):
        return "Binary"
    if pd.api.types.is_numeric_dtype(values) or safe_numeric_series(df, column).notna().sum() / max(len(values), 1) > 0.9:
        return "Continuous"
    return "Not Numeric"


def validate_metric(df: pd.DataFrame, column: str, metric_type: str) -> list[str]:
    if column not in df.columns:
        return [f"{column} is not present in the selected dataset."]
    values = first_series(df, column)
    if values is None:
        return [f"{column} is not present in the selected dataset."]
    values = values.dropna()
    if metric_type == "Continuous" and safe_numeric_series(df, column).notna().sum() == 0:
        return [f"{column} must be numeric for a continuous metric."]
    if metric_type == "Binary":
        unique = set(values.unique().tolist())
        if len(unique) != 2:
            return [f"{column} should contain exactly two valid states for a binary metric."]
    return []


def metric_baseline(df: pd.DataFrame, column: str, metric_type: str) -> dict[str, float]:
    values = safe_numeric_series(df, column).dropna()
    if metric_type == "Binary":
        return {"baseline": float(values.mean()), "std_dev": float((values.mean() * (1 - values.mean())) ** 0.5)}
    return {"baseline": float(values.mean()), "std_dev": float(values.std(ddof=1))}
