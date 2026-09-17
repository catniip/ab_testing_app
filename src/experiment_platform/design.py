from __future__ import annotations

import numpy as np
import pandas as pd


def candidate_lines(min_line: int, max_line: int, increment: int) -> np.ndarray:
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
    candidates = candidate_lines(min_line, max_line, increment)
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
