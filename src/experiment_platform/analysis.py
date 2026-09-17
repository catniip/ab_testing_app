from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests


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
        expected = counts.sum() / len(arms)
        statistic = float((((counts - expected) ** 2) / expected).sum())
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
    if len(arm_counts) == len(arms) and sum(arm_counts.values()) > 0:
        observed = np.array([arm_counts.get(arm, 0) for arm in arms], dtype=float)
        expected = observed.sum() / len(arms)
        statistic = float(np.sum((observed - expected) ** 2 / expected))
        srm_pvalue = float(stats.chi2.sf(statistic, len(arms) - 1))
    return {
        "rows": len(scoped),
        "unique_units": int(scoped[unit_id_col].nunique()) if unit_id_col in scoped else 0,
        "duplicate_units": duplicate_units,
        "arm_counts": arm_counts,
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
