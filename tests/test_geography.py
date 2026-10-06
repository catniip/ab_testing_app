import pandas as pd

from src.experiment_platform.core import load_geo_demo
from src.experiment_platform.geography import (
    aggregate_trend,
    build_geo_preview,
    evaluate_geo_balance,
    infer_geo_schema,
    leave_one_dma_out,
    plan_geo_test,
    prepare_geo_analysis,
    prepare_geo_panel,
    run_panel_did,
    run_placebo_tests,
    validate_fixed_assignment,
    validate_geo_panel,
)


def _demo_inputs():
    raw = load_geo_demo()
    panel = prepare_geo_panel(raw, "week", "dma", "applications")
    validation = validate_fixed_assignment(panel, "treatment_group")
    assert validation["errors"] == []
    return panel, validation["assignment"]


def test_geo_demo_schema_and_validation():
    df = load_geo_demo()
    inferred = infer_geo_schema(df)
    assert inferred["date"] == "week"
    assert inferred["dma"] == "dma"
    assert inferred["outcome"] == "applications"
    assert inferred["group"] == "treatment_group"
    panel = prepare_geo_panel(df, inferred["date"], inferred["dma"], inferred["outcome"])
    quality = validate_geo_panel(panel)
    assert quality["frequency"] == "Weekly"
    assert quality["dma_count"] >= 10
    assert quality["duplicates"] == 0


def test_geo_preview_keeps_market_assignment_and_source_coverage():
    panel, assignment = _demo_inputs()
    preview = build_geo_preview(panel, assignment, "population")
    assert preview["quality"]["dma_count"] == panel["_dma"].nunique()
    assert len(preview["markets"]) == panel["_dma"].nunique()
    assert set(preview["markets"]["Assignment"]) == {"Test", "Control"}
    assert {"Test", "Control"}.issubset(set(preview["trend"]["group"]))
    assert "Market Size" in preview["markets"]
    assert preview["start_date"] == panel["_date"].min()
    assert preview["end_date"] == panel["_date"].max()


def test_fixed_assignment_is_long_form_and_never_invents_pairs():
    panel, assignment = _demo_inputs()
    assert list(assignment.columns) == ["DMA", "Group"]
    assert len(assignment) == panel["_dma"].nunique()
    assert set(assignment["Group"]) == {"Test", "Control"}
    assert "Pair" not in assignment


def test_assignment_validation_rejects_group_changes_over_time():
    panel, _ = _demo_inputs()
    dma = panel["_dma"].iloc[0]
    indices = panel.index[panel["_dma"] == dma]
    panel.loc[indices[-1], "treatment_group"] = "Control" if panel.loc[indices[0], "treatment_group"] == "Test" else "Test"
    validation = validate_fixed_assignment(panel, "treatment_group")
    assert any("changes over time" in issue for issue in validation["errors"])


def test_geo_planner_is_monotonic_and_reports_dma_requirements():
    panel, assignment = _demo_inputs()
    plan = plan_geo_test(panel, assignment, "2026-09-06", "Weekly", "Relative lift", 7.5, max_duration=26)
    table = plan["table"]
    assert table["Detection Chance"].is_monotonic_increasing
    assert table["Minimum Detectable Effect"].is_monotonic_decreasing
    assert (table["Required Total DMAs"] >= 4).all()
    assert (table["Required Test DMAs"] >= 2).all()
    assert (table["Required Control DMAs"] >= 2).all()
    assert plan["current_total_dmas"] == len(assignment)


def test_panel_did_recovers_positive_demo_effect_and_diagnostics():
    panel, assignment = _demo_inputs()
    balance = evaluate_geo_balance(panel, assignment, "2026-09-06")
    analysis_panel = prepare_geo_analysis(panel, assignment, "2026-09-06", "2026-12-27")
    result = run_panel_did(analysis_panel)
    trend = aggregate_trend(analysis_panel, "Indexed")
    placebo = run_placebo_tests(analysis_panel, count=3)
    sensitivity = leave_one_dma_out(analysis_panel)
    assert balance["dma_count"] == len(assignment)
    assert result["effect_per_period"] > 0
    assert result["lift"] > 0
    assert result["test_dmas"] == 5
    assert result["control_dmas"] == 7
    assert result["post_periods"] > 0
    assert result["interval"][0] < result["interval"][1]
    assert {"Test", "Control"}.issubset(set(trend["group"]))
    assert "status" in placebo
    assert len(sensitivity) == len(assignment)


def test_weighted_panel_did_runs_with_market_size_weights():
    panel, assignment = _demo_inputs()
    analysis_panel = prepare_geo_analysis(panel, assignment, "2026-09-06", "2026-12-27", "population")
    result = run_panel_did(analysis_panel, weighted=True)
    assert result["estimand"] == "Market-size-weighted average DMA"
    assert pd.notna(result["pvalue"])
