import pandas as pd
import pytest

from src.experiment_platform.metrics import metric_baseline
from src.experiment_platform.models import DataMappingConfig, MetricConfig, PopulationConfig
from src.experiment_platform.raw_processing import (
    aggregate_metric_by_mob,
    build_historical_metric_dataset,
    build_analysis_dataset,
    common_horizon,
    determine_mature_units,
    processed_metric_name,
)


def raw_fixture():
    rows = []
    for account, cpc, booking, max_mob in [
        ("A1", "CPC_A", "2023-01-01", 36),
        ("A2", "CPC_B", "2023-01-01", 36),
        ("A3", "CPC_A", "2025-01-01", 12),
        ("A4", "CPC_C", "2023-01-01", 36),
    ]:
        for mob in range(1, max_mob + 1):
            if account == "A2" and mob == 20:
                continue
            rows.append(
                {
                    "account_id": account,
                    "cpc": cpc,
                    "booking_date": booking,
                    "mob": mob,
                    "current_credit_line": 5000 if account != "A4" else 8000,
                    "balance": mob * 10,
                    "revenue": mob,
                    "loss": 1,
                }
            )
    rows.append({"account_id": "A1", "cpc": "CPC_A", "booking_date": "2023-01-01", "mob": 1, "current_credit_line": 5000, "balance": 20, "revenue": 2, "loss": 2})
    return pd.DataFrame(rows)


def metrics():
    return {
        "Primary": MetricConfig("Average Balance", "avg_balance_mob12", "Primary", source_column="balance", aggregation_method="Average", mob_horizon=12),
        "Secondary": MetricConfig("Cumulative Revenue", "cum_revenue_mob36", "Secondary", source_column="revenue", aggregation_method="Cumulative", mob_horizon=36),
        "Guardrail": MetricConfig("Cumulative Loss", "cum_loss_mob24", "Guardrail", source_column="loss", aggregation_method="Cumulative", mob_horizon=24),
    }


def test_single_and_multiple_cpc_grouping():
    raw = raw_fixture()
    mapping = DataMappingConfig(cutoff_date="2026-01-01")
    one = build_analysis_dataset(raw, PopulationConfig(["CPC_A"]), metrics(), mapping, "current_credit_line")
    many = build_analysis_dataset(raw, PopulationConfig(["CPC_A", "CPC_B"]), metrics(), mapping, "current_credit_line")
    assert set(one.analysis_df["cpc"]) == {"CPC_A"}
    assert set(many.analysis_df["cpc"]) == {"CPC_A", "CPC_B"}
    assert many.diagnostics["Selected CPC Accounts"] == 3


def test_invalid_cpc_returns_empty_population():
    result = build_analysis_dataset(raw_fixture(), PopulationConfig(["NOPE"]), metrics(), DataMappingConfig(cutoff_date="2026-01-01"), "current_credit_line")
    assert result.analysis_df.empty


def test_common_horizon_and_booking_date_maturity():
    raw = raw_fixture()
    mapping = DataMappingConfig(cutoff_date="2026-01-01")
    assert common_horizon(metrics()) == 36
    mature = determine_mature_units(raw, mapping, 36)
    assert "A3" not in set(mature)
    assert "A1" in set(mature)


def test_average_and_cumulative_aggregation_with_duplicate_unit_mob():
    raw = raw_fixture()
    avg = aggregate_metric_by_mob(raw[raw["account_id"] == "A1"], "account_id", "mob", "balance", "Average", 12)
    cum = aggregate_metric_by_mob(raw[raw["account_id"] == "A1"], "account_id", "mob", "revenue", "Cumulative", 36)
    assert round(avg.loc["A1"], 2) == round((((10 + 20) / 2) + sum(m * 10 for m in range(2, 13))) / 12, 2)
    assert cum.loc["A1"] == ((1 + 2) / 2) + sum(range(2, 37))


def test_metric_range_uses_only_selected_mobs():
    raw = raw_fixture()
    ranged = aggregate_metric_by_mob(raw[raw["account_id"] == "A1"], "account_id", "mob", "balance", "Average", 12, 6)
    assert ranged.loc["A1"] == sum(mob * 10 for mob in range(6, 13)) / 7
    metric = MetricConfig("Average Balance", "unused", "Primary", source_column="balance", aggregation_method="Average", mob_start=6, mob_horizon=12)
    assert processed_metric_name(metric) == "avg_balance_mob6_12"


def test_common_eligible_units_and_incomplete_history_diagnostics():
    result = build_analysis_dataset(raw_fixture(), PopulationConfig(["CPC_A", "CPC_B"]), metrics(), DataMappingConfig(cutoff_date="2026-01-01"), "current_credit_line")
    assert result.diagnostics["Longest Required Horizon"] == 36
    assert result.diagnostics["Excluded for Insufficient Maturity"] == 1
    assert result.diagnostics["Excluded for Data Completeness"] == 1
    assert result.analysis_df["account_id"].tolist() == ["A1", "A2", "A3"]
    assert result.diagnostics["Final Analysis Population"] == 3
    assert result.diagnostics["Complete Eligible Accounts"] == 1
    assert result.metric_preview.set_index("Role")["Eligible N"].to_dict() == {"Primary": 3, "Secondary": 1, "Guardrail": 1}


def test_power_baseline_uses_processed_metric_column():
    result = build_analysis_dataset(raw_fixture(), PopulationConfig(["CPC_A", "CPC_B"]), metrics(), DataMappingConfig(cutoff_date="2026-01-01"), "current_credit_line")
    baseline = metric_baseline(result.analysis_df, "avg_balance_mob12", "Continuous")
    assert baseline["baseline"] == result.analysis_df["avg_balance_mob12"].mean()


def test_central_historical_metric_dataset_function_and_unit_variance():
    result = build_historical_metric_dataset(
        raw_fixture(),
        ["CPC_A", "CPC_B"],
        "account_id",
        "cpc",
        "mob",
        "booking_date",
        metrics(),
        cutoff_date="2026-01-01",
        strategy_column="current_credit_line",
    )
    assert len(result.analysis_df) == 3
    assert "avg_balance_mob12" in result.analysis_df.columns
    assert result.metric_preview.set_index("Role").loc["Primary", "Eligible N"] == 3


def test_cross_sectional_dataset_does_not_require_segment_or_mob():
    raw = pd.DataFrame({"customer_id": ["A", "B", "C"], "outcome": [10.0, 12.0, 11.0]})
    mapping = DataMappingConfig(data_structure="Cross-sectional (one row per unit)", unit_id_column="customer_id", cpc_column="", mob_column="")
    configured = {"Primary": MetricConfig("Outcome", "outcome", "Primary", source_column="outcome")}
    result = build_analysis_dataset(raw, PopulationConfig(), configured, mapping)
    assert result.diagnostics["Final Analysis Population"] == 3
    assert result.metric_preview.loc[0, "Observation Window"] == "As observed"


def test_processing_rejects_a_changing_acquisition_line():
    raw = raw_fixture()
    raw.loc[(raw["account_id"] == "A1") & (raw["mob"] == 2), "current_credit_line"] = 8000
    with pytest.raises(ValueError, match="must be fixed for each experimental unit"):
        build_analysis_dataset(
            raw,
            PopulationConfig(["CPC_A", "CPC_B"]),
            metrics(),
            DataMappingConfig(cutoff_date="2026-01-01"),
            "current_credit_line",
        )
