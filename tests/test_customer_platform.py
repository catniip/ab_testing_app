import pandas as pd

from src.experiment_platform.analysis import analysis_integrity_summary, binary_results, continuous_results, validate_analysis_data
from src.experiment_platform.arm_selection import (
    numeric_candidates,
    suggest_numeric_designs,
    validate_categorical_strategy,
    validate_numeric_strategy,
)
from src.experiment_platform.decision import arm_decision_scorecard, experiment_recommendation, guardrail_status, recommendation_status
from src.experiment_platform.demo_data import experiment_results, historical_portfolio
from src.experiment_platform.charts import historical_association
from src.experiment_platform.metrics import column_schema, suggest_metric_type, validate_metric
from src.experiment_platform.power import detectable_effect_binary, detectable_effect_continuous, power_binary, power_continuous, sample_size_binary, sample_size_continuous


def test_schema_and_metric_type_inference():
    df = historical_portfolio(rows=200)
    schema = column_schema(df)
    assert "current_credit_line" in schema["Column"].tolist()
    assert suggest_metric_type(df, "default_flag") == "Binary"
    assert suggest_metric_type(df, "revolving_balance") == "Continuous"


def test_metric_validation():
    df = historical_portfolio(rows=200)
    assert validate_metric(df, "revolving_balance", "Continuous") == []
    assert validate_metric(df, "default_flag", "Binary") == []
    assert validate_metric(df, "risk_segment", "Continuous")


def test_numeric_candidate_generation_and_validation():
    assert numeric_candidates(2000, 3000, 500) == [2000.0, 2500.0, 3000.0]
    assert validate_numeric_strategy(5000, [3000, 8000], 2000, 10000, 500) == []
    errors = validate_numeric_strategy(5000, [5000, 5250, 5250], 2000, 10000, 500)
    assert any("Control" in error for error in errors)
    assert any("Duplicate" in error for error in errors)
    assert any("increment" in error for error in errors)


def test_categorical_strategy_validation():
    assert validate_categorical_strategy("BAU", ["Offer A", "Offer B"]) == []
    assert validate_categorical_strategy("BAU", ["BAU"])


def test_auto_recommendation_returns_ranked_designs():
    df = historical_portfolio(rows=1000)
    suggestions = suggest_numeric_designs(df, "current_credit_line", "revolving_balance", 5000, 2000, 10000, 500, 2)
    assert len(suggestions) >= 1
    assert {"Coverage", "Separation", "Detectability", "Overall Score"}.issubset(suggestions.columns)
    assert 5000 not in suggestions.iloc[0]["Suggested Treatments"]


def test_historical_association_allows_same_selected_column():
    df = historical_portfolio(rows=200)
    fig = historical_association(df, "current_credit_line", "current_credit_line")
    assert len(fig.data) == 1


def test_historical_association_uses_proposed_line_groups_when_provided():
    df = historical_portfolio(rows=300)
    fig = historical_association(
        df,
        "current_credit_line",
        "revolving_balance",
        strategy_points=[3000, 8000],
    )
    assert list(fig.data[0].x) == [3000, 8000]
    assert sum(row[0] for row in fig.data[0].customdata) == len(df)


def test_historical_association_reuses_binning_and_legends_selected_lines():
    df = historical_portfolio(rows=600)
    fig = historical_association(
        df,
        "current_credit_line",
        "revolving_balance",
        strategy_points=[3000, 8000],
        strategy_point_labels=["Assigned line · BAU: 3,000", "Assigned line · HIGH: 8,000"],
        binning_method="Fixed bin width",
        bin_width=500,
    )
    assert [trace.name for trace in fig.data] == ["Historical mean", "Assigned line · BAU: 3,000", "Assigned line · HIGH: 8,000"]
    assert len(fig.layout.shapes) == 0


def test_power_calculations_return_positive_sample_sizes():
    assert sample_size_continuous(0.05, 0.8, 100, 1000) > 0
    assert sample_size_binary(0.05, 0.8, 0.05, 0.01) > 0
    assert detectable_effect_continuous(0.05, 0.8, 1000, 200) > detectable_effect_continuous(0.05, 0.8, 1000, 400)
    assert detectable_effect_binary(0.05, 0.8, 0.05, 200) > detectable_effect_binary(0.05, 0.8, 0.05, 400)
    assert 0 < power_continuous(0.05, 100, 1000, 200) < 1
    assert 0 < power_binary(0.05, 0.05, 0.01, 200) < 1


def test_continuous_and_binary_multi_arm_analysis():
    df = experiment_results(rows_per_arm=120)
    continuous = continuous_results(df, "assigned_credit_line", "revolving_balance", 5000, [3000, 8000], 0.05)
    binary = binary_results(df, "assigned_credit_line", "default_flag", 5000, [3000, 8000], 0.05)
    assert len(continuous) == 3
    assert len(binary) == 3
    assert "Effect vs Control" in continuous.columns


def test_guardrail_status_and_decision_logic():
    primary = pd.DataFrame(
        [
            {"Credit Line": 5000, "Is Control": True, "Effect vs Control": 0.0, "Significant": False, "CI Lower": None},
            {"Credit Line": 8000, "Is Control": False, "Effect vs Control": 10.0, "Significant": True, "CI Lower": 3.0},
        ]
    )
    guardrail = pd.DataFrame(
        [
            {"Credit Line": 8000, "Is Control": False, "Effect vs Control": 0.001, "Significant": False, "CI Lower": -0.001},
        ]
    )
    assert guardrail_status(guardrail.iloc[0], 0.005, "Lower is Better") == "PASS"
    assert "Recommended Strategy" in experiment_recommendation(primary, guardrail, 0.005, "Lower is Better")
    guardrail.loc[0, "Effect vs Control"] = 0.02
    breached = experiment_recommendation(primary, guardrail, 0.005, "Lower is Better")
    assert "guardrail threshold was breached" in breached
    assert recommendation_status(breached) == "error"
    no_effect = experiment_recommendation(primary.assign(Significant=False), None, None, "Lower is Better")
    assert recommendation_status(no_effect) == "warning"


def test_lower_is_better_primary_can_be_recommended():
    primary = pd.DataFrame(
        [
            {"Credit Line": "Control", "Is Control": True, "Effect vs Control": 0.0, "Significant": False, "CI Lower": None, "CI Upper": None},
            {"Credit Line": "Treatment", "Is Control": False, "Effect vs Control": -2.0, "Significant": True, "CI Lower": -3.0, "CI Upper": -1.0},
        ]
    )
    recommendation = experiment_recommendation(primary, None, None, "Lower is Better", "Lower is Better")
    assert "Recommended Strategy" in recommendation


def test_holm_adjustment_and_result_validation():
    df = experiment_results(rows_per_arm=80)
    results = continuous_results(df, "assigned_credit_line", "revolving_balance", 5000, [3000, 8000], 0.05, "Holm")
    treatments = results[~results["Is Control"]]
    assert (treatments["Adjusted p-value"] >= treatments["Raw p-value"]).all()
    errors, _ = validate_analysis_data(
        df,
        "assigned_credit_line",
        {"Primary": ("revolving_balance", "Continuous"), "Guardrail": ("revolving_balance", "Continuous")},
        5000,
        [3000, 8000],
        "customer_id",
    )
    assert any("different result column" in error for error in errors)


def test_customer_analysis_requires_one_row_per_experimental_unit():
    df = experiment_results(rows_per_arm=20)
    no_id_errors, _ = validate_analysis_data(
        df,
        "assigned_credit_line",
        {"Primary": ("revolving_balance", "Continuous")},
        5000,
        [3000, 8000],
        "",
    )
    assert any("experimental unit ID" in error for error in no_id_errors)
    repeated = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    duplicate_errors, _ = validate_analysis_data(
        repeated,
        "assigned_credit_line",
        {"Primary": ("revolving_balance", "Continuous")},
        5000,
        [3000, 8000],
        "customer_id",
    )
    assert any("repeated unit IDs" in error for error in duplicate_errors)


def test_analysis_integrity_summary_reports_arm_balance_and_missingness():
    df = experiment_results(rows_per_arm=20)
    df.loc[df.index[0], "revolving_balance"] = None
    summary = analysis_integrity_summary(
        df,
        "assigned_credit_line",
        {"Primary": ("revolving_balance", "Continuous")},
        [5000, 3000, 8000],
        "customer_id",
    )
    assert summary["unique_units"] == 60
    assert summary["duplicate_units"] == 0
    assert summary["missing_by_metric"]["Primary"] == 1
    assert summary["srm_status"] == "Pass"


def test_arm_decision_scorecard_combines_evidence_magnitude_and_guardrail():
    primary = pd.DataFrame(
        [
            {"Credit Line": "Control", "Is Control": True, "Effect vs Control": 0.0, "Relative Lift": 0.0, "Significant": False, "CI Lower": None, "CI Upper": None},
            {"Credit Line": "Offer A", "Is Control": False, "Effect vs Control": 12.0, "Relative Lift": 0.08, "Significant": True, "CI Lower": 3.0, "CI Upper": 21.0, "Adjusted p-value": 0.01},
        ]
    )
    guardrail = pd.DataFrame(
        [{"Credit Line": "Offer A", "Is Control": False, "Effect vs Control": 0.01, "CI Lower": 0.005, "CI Upper": 0.02}]
    )
    scorecard = arm_decision_scorecard(primary, "Higher is Better", 10.0, guardrail, 0.005, "Lower is Better")
    assert scorecard.loc[0, "Primary Evidence"] == "Clear benefit"
    assert bool(scorecard.loc[0, "Meets Planned Effect"])
    assert scorecard.loc[0, "Guardrail"] == "Fail"
    assert scorecard.loc[0, "Decision"] == "Do not scale"
