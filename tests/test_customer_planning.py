import numpy as np
import pandas as pd

from src.experiment_platform.customer import automatic_matching_distance, assign_group_labels, build_customer_preview, calculate_group_design, calculate_variant_group_design, default_numeric_groups, planning_comparison_count, suggest_group_designs
from src.experiment_platform.core import load_customer_demo
from src.experiment_platform.core import DesignConfig, MetricConfig


def planning_history(seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for line, mean, sd in [(3000, 900, 250), (5000, 1400, 400), (8000, 2050, 620)]:
        for _ in range(160):
            rows.append({"current_credit_line": line, "outcome": rng.normal(mean, sd)})
    return pd.DataFrame(rows)


def test_default_fico_groups_and_assignment():
    values = pd.Series([580, 650, 680, 719, 740, 800])
    definitions = default_numeric_groups(values, "fico")
    labels = assign_group_labels(values, definitions)
    assert definitions[0]["label"] == "Below 660"
    assert definitions[-1]["label"] == "720+"
    assert labels.tolist() == ["Below 660", "Below 660", "660 to 719", "660 to 719", "720+", "720+"]


def test_customer_preview_is_one_row_per_account_with_business_context():
    raw = load_customer_demo("General Customer Test")
    preview = build_customer_preview(
        raw,
        "account_id",
        "primary_outcome",
        "cpc",
        "risk_segment",
        "historical_experience",
        "booking_date",
        "mob",
    )
    assert preview["unique_accounts"] == raw["account_id"].nunique()
    assert len(preview["accounts"]) == raw["account_id"].nunique()
    assert preview["product_count"] == raw["cpc"].nunique()
    assert preview["group_count"] == raw["risk_segment"].nunique()
    assert preview["accounts"]["Customer / Account"].is_unique
    assert {"Historical Strategy", "Average Outcome", "Periods"}.issubset(preview["accounts"].columns)


def test_group_plan_allocates_flow_to_finish_lines_together():
    metric = MetricConfig(
        "Outcome",
        "outcome",
        "Primary",
        effect_type="Relative %",
        effect_value=0.15,
        source_column="outcome",
        processed_column="outcome",
    )
    design = DesignConfig(sample_size_basis="Primary Metric Only")
    rows, comparisons, _ = calculate_group_design(
        planning_history(),
        "current_credit_line",
        {"Primary": metric},
        design,
        5000,
        [3000, 8000],
        eligible_flow=300,
        group_label="Prime",
    )
    assert len(rows) == 3
    assert comparisons
    assert abs(sum(row["Traffic Allocation"] for row in rows) - 1) < 1e-9
    assert max(row["Test Duration"] for row in rows) - min(row["Test Duration"] for row in rows) < 1e-9


def test_suggestions_stay_inside_business_range_and_report_duration():
    metric = MetricConfig(
        "Outcome",
        "outcome",
        "Primary",
        effect_type="Relative %",
        effect_value=0.15,
        source_column="outcome",
        processed_column="outcome",
    )
    suggestions = suggest_group_designs(
        planning_history(),
        "current_credit_line",
        {"Primary": metric},
        DesignConfig(sample_size_basis="Primary Metric Only"),
        control=5000,
        minimum=3000,
        maximum=8000,
        treatment_count=2,
        eligible_flow=300,
        max_periods=12,
    )
    assert not suggestions.empty
    assert (suggestions["Test Duration"] > 0).all()
    assert all(3000 <= value <= 8000 for values in suggestions["Suggested Treatments"] for value in values)


def test_named_offer_plan_uses_group_level_history_without_strategy_column():
    history = pd.DataFrame({"outcome": np.linspace(80, 120, 300)})
    metric = MetricConfig(
        "Revenue",
        "outcome",
        "Primary",
        effect_type="Relative %",
        effect_value=0.1,
        source_column="outcome",
        processed_column="outcome",
    )
    rows, comparisons, binding = calculate_variant_group_design(
        history,
        {"Primary": metric},
        DesignConfig(sample_size_basis="Primary Metric Only"),
        "BAU Offer",
        ["Cash Bonus", "Rewards Bundle"],
        eligible_flow=600,
        group_label="Prime",
    )
    assert [row["Arm"] for row in rows] == ["BAU Offer", "Cash Bonus", "Rewards Bundle"]
    assert all(row["Support"] == "Group-level pooled history" for row in rows)
    assert abs(sum(row["Traffic Allocation"] for row in rows) - 1) < 1e-9
    assert comparisons and "Revenue" in binding


def test_automatic_matching_distance_excludes_remote_history():
    assert automatic_matching_distance([3000, 8000]) == 2500
    history = pd.DataFrame(
        {
            "current_credit_line": [-5000, 2500, 3000, 3500, 7500, 8000, 8500, 20000],
            "outcome": [10, 80, 100, 120, 180, 200, 220, 500],
        }
    )
    metric = MetricConfig(
        "Outcome",
        "outcome",
        "Primary",
        effect_type="Absolute",
        effect_value=20,
        source_column="outcome",
        processed_column="outcome",
    )
    rows, _, _ = calculate_group_design(
        history,
        "current_credit_line",
        {"Primary": metric},
        DesignConfig(sample_size_basis="Primary Metric Only"),
        3000,
        [8000],
        eligible_flow=100,
        group_label="All customers",
    )
    assert sum(row["Historical N"] for row in rows) == 6
    assert all(row["Historical Match Limit"] == 2500 for row in rows)


def test_joint_audience_scope_protects_all_treatment_comparisons():
    assert planning_comparison_count(2, 3, "Separate decision per audience") == 2
    assert planning_comparison_count(2, 3, "One joint decision across audiences") == 6

    metric = MetricConfig(
        "Outcome",
        "outcome",
        "Primary",
        effect_type="Relative %",
        effect_value=0.15,
        source_column="outcome",
        processed_column="outcome",
    )
    design = DesignConfig(sample_size_basis="Primary Metric Only")
    separate, _, _ = calculate_group_design(
        planning_history(),
        "current_credit_line",
        {"Primary": metric},
        design,
        5000,
        [3000, 8000],
        eligible_flow=300,
        group_label="Prime",
        family_comparisons=2,
    )
    joint, _, _ = calculate_group_design(
        planning_history(),
        "current_credit_line",
        {"Primary": metric},
        design,
        5000,
        [3000, 8000],
        eligible_flow=300,
        group_label="Prime",
        family_comparisons=6,
    )
    assert sum(row["Required Accounts"] for row in joint) > sum(row["Required Accounts"] for row in separate)
