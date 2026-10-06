import pandas as pd

from src.experiment_platform.core import (
    date_like_columns,
    deduplicate_columns,
    infer_mob_column,
    infer_schema,
    safe_numeric_series,
)
from src.experiment_platform.core import column_schema, suggest_metric_type
from src.experiment_platform.core import DataMappingConfig, MetricConfig, PopulationConfig
from src.experiment_platform.customer import build_analysis_dataset


def upload_like_df():
    rows = []
    for customer in ["C1", "C2"]:
        for mob in range(1, 13):
            rows.append(
                {
                    "customer_id": customer,
                    "account_id": customer.replace("C", "A"),
                    "cpc": "CPC_A",
                    "booking_date": "2024-01-01",
                    "observation_month": f"2024-{mob:02d}-01",
                    "mob": str(mob),
                    "data_cutoff_date": "2025-01-01",
                    "current_credit_line": "5000",
                    "risk_segment": "Prime",
                    "revolving_balance": str(100 + mob),
                    "risk_adjusted_revenue": str(10 + mob),
                    "loss": "1",
                    "default_flag": "0",
                }
            )
    return pd.DataFrame(rows)


def test_schema_inference_handles_duplicate_column_names():
    df = upload_like_df()
    df.columns = ["mob" if col == "revolving_balance" else col for col in df.columns]
    deduped = deduplicate_columns(df)
    schema = infer_schema(deduped)
    assert deduped.columns.is_unique
    assert "mob" in schema["Column"].tolist()
    assert safe_numeric_series(deduped, "mob").notna().sum() > 0


def test_column_schema_and_metric_type_do_not_raise_on_uploaded_strings():
    df = upload_like_df()
    schema = column_schema(df)
    assert "mob" in schema["Column"].tolist()
    assert suggest_metric_type(df, "revolving_balance") == "Continuous"


def test_date_mapping_candidates_exclude_numeric_period_columns():
    candidates = date_like_columns(upload_like_df())
    assert "booking_date" in candidates
    assert "observation_month" in candidates
    assert "mob" not in candidates


def test_infer_mob_and_build_uploaded_analysis_dataset():
    df = upload_like_df()
    mapping = DataMappingConfig(
        unit_id_column="customer_id",
        cpc_column="cpc",
        mob_column=infer_mob_column(df),
        booking_date_column="booking_date",
        cutoff_date="2025-01-01",
    )
    metrics = {
        "Primary": MetricConfig(
            "Revolving Balance",
            "avg_revolving_balance_mob12",
            "Primary",
            source_column="revolving_balance",
            aggregation_method="Average",
            mob_horizon=12,
        )
    }
    result = build_analysis_dataset(df, PopulationConfig(["CPC_A"]), metrics, mapping, "current_credit_line")
    assert len(result.analysis_df) == 2
    assert "avg_revolving_balance_mob12" in result.analysis_df.columns
