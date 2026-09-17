from __future__ import annotations

import pandas as pd


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
        elif evidence == "Clear benefit" and practical and safety in {"Pass", "Not configured"}:
            recommendation = "Scale candidate"
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
    if "no treatment produced" in lower or "do not scale" in lower:
        return "warning"
    return "success"
