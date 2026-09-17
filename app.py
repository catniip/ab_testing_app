from __future__ import annotations

import math
import pandas as pd
import streamlit as st
from datetime import datetime

from src.experiment_platform.analysis import analysis_integrity_summary, binary_results, continuous_results, response_summary, validate_analysis_data
from src.experiment_platform.assumptions import (
    design_sample_size,
    design_sample_size_by_arm,
    design_neyman_allocation,
    detectable_for_metric,
    expected_power,
    normalize_metric,
    update_assumption_from_history,
    validate_metric_assumptions,
)
from src.experiment_platform.arm_selection import suggest_numeric_designs, validate_categorical_strategy, validate_numeric_strategy
from src.experiment_platform.charts import bsts_counterfactual_chart, cumulative_impact_chart, detectable_effect_curve, forest_plot, geo_balance_chart, geo_dma_map, geo_trend_chart, historical_association, impact_chart, pre_post_bar, response_plot, time_series_line, timeseries_chart_context
from src.experiment_platform.data_access import load_geo_demo, load_raw_demo, load_results_demo, load_timeseries_demo, load_timeseries_planning_demo, load_uploaded_csv
from src.experiment_platform.data_validation import data_quality_warnings, date_like_columns, first_series, infer_column, infer_mob_column, normalize_uploaded_dataset, safe_numeric_series
from src.experiment_platform.decision import arm_decision_scorecard, experiment_recommendation, guardrail_status, recommendation_status
from src.experiment_platform.formatting import money, number, p_value, percent
from src.experiment_platform.geography import DMA_CENTROIDS, GeoConfig, aggregate_trend, evaluate_geo_balance, infer_geo_schema, leave_one_dma_out, plan_geo_test, prepare_geo_analysis, prepare_geo_panel, run_panel_did, run_placebo_tests, validate_fixed_assignment, validate_geo_panel
from src.experiment_platform.historical_strategy import historical_arm_statistics, validate_fixed_unit_value
from src.experiment_platform.metrics import column_schema, metric_baseline, suggest_metric_type, validate_metric
from src.experiment_platform.models import AnalysisConfig, DataConfig, DataMappingConfig, DesignConfig, MetricConfig, PopulationConfig, StrategyConfig
from src.experiment_platform.power import adjusted_alpha, detectable_effect_binary, detectable_effect_continuous, detectable_effect_continuous_unequal, duration, power_binary, power_continuous, power_continuous_unequal, sample_size_binary, sample_size_continuous
from src.experiment_platform.raw_processing import build_analysis_dataset
from src.experiment_platform.timeseries import TimeSeriesConfig, build_business_interpretation, campaign_decision_date, classify_calibration, config_fingerprint, data_fingerprint, detect_frequency, duplicate_timestamp_count, duration_power_status, fit_bsts_model, humanize_column_name, incremental_outcome_label, infer_timeseries_columns, method_config_fingerprint, planning_config_fingerprint, planning_durations, prepare_timeseries_data, predictor_candidates, probability_positive_label, projected_operational_exposure, run_duration_power_simulation, run_pre_post_analysis


st.set_page_config(page_title="Experiment Platform", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1380px;}
    h1 {font-size: 1.55rem; letter-spacing: 0;}
    h2 {font-size: 1.22rem; letter-spacing: 0;}
    h3 {font-size: 1.02rem; letter-spacing: 0;}
    [data-testid="stMetric"] {border: 1px solid #e3e6ea; padding: .75rem; border-radius: 6px; background: #fff;}
    div[data-testid="stDataFrame"] {border: 1px solid #e3e6ea; border-radius: 6px;}
    .subtle-note {border-left: 3px solid #7891b3; padding: .55rem .75rem; background: #f7f9fc; color: #2f3b4a;}
    .step-line {font-size: .92rem; line-height: 1.9;}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    st.session_state.setdefault("experiment_name", "Credit Line Experiment")
    st.session_state.setdefault("data_config", DataConfig())
    st.session_state.setdefault("strategy_config", StrategyConfig())
    st.session_state.setdefault("design_config", DesignConfig())
    st.session_state.setdefault("analysis_config", AnalysisConfig())
    st.session_state.setdefault(
        "metrics_config",
        {
            "Primary": MetricConfig("Average Revolving Balance", "avg_revolving_balance_mob12", "Primary", "Continuous", "Higher is Better", "Absolute", 150.0, source_column="revolving_balance", processed_column="avg_revolving_balance_mob12", aggregation_method="Average", mob_horizon=12),
            "Secondary": MetricConfig("Cumulative Revenue", "cum_revenue_mob36", "Secondary", "Continuous", "Higher is Better", "Relative %", 0.03, source_column="revenue", processed_column="cum_revenue_mob36", aggregation_method="Cumulative", mob_horizon=36),
            "Guardrail": MetricConfig("Cumulative Loss", "cum_loss_mob36", "Guardrail", "Continuous", "Lower is Better", "Absolute", 50.0, guardrail_threshold=50.0, source_column="loss", processed_column="cum_loss_mob36", aggregation_method="Cumulative", mob_horizon=36),
        },
    )
    st.session_state.setdefault("data_mapping", DataMappingConfig())
    st.session_state.setdefault("population_config", PopulationConfig())
    if not hasattr(st.session_state.data_mapping, "data_structure"):
        st.session_state.data_mapping.data_structure = "Longitudinal (unit x period)"
    if not hasattr(st.session_state.strategy_config, "experiment_template"):
        st.session_state.strategy_config.experiment_template = "Credit Line"
    if not hasattr(st.session_state.strategy_config, "arm_format"):
        st.session_state.strategy_config.arm_format = "Ordered numeric levels" if st.session_state.strategy_config.strategy_type == "Numeric Strategy" else "Named variants"
    if not hasattr(st.session_state.strategy_config, "control_name"):
        st.session_state.strategy_config.control_name = "BAU"
    if not hasattr(st.session_state.strategy_config, "historical_grouping"):
        st.session_state.strategy_config.historical_grouping = "Nearest proposed point"
    if not hasattr(st.session_state.strategy_config, "max_mapping_distance"):
        st.session_state.strategy_config.max_mapping_distance = 0.0
    strategy_defaults = StrategyConfig()
    for field_name in ["selection_mode", "curve_binning_method", "curve_bin_width", "curve_bin_count", "curve_smoothing", "planning_grouping", "planning_bin_width"]:
        if not hasattr(st.session_state.strategy_config, field_name):
            setattr(st.session_state.strategy_config, field_name, getattr(strategy_defaults, field_name))
    if not st.session_state.get("customer_ui_v2_migrated"):
        apply_experiment_template(st.session_state.strategy_config, st.session_state.strategy_config.experiment_template)
        st.session_state.customer_ui_v2_migrated = True
    for field_name, default in {
        "multiplicity_method": "Holm",
        "attrition_rate": 0.0,
        "planned_launch_date": "",
        "outcome_delay_value": 0,
        "outcome_delay_unit": "Days",
        "arm_sample_sizes": {},
    }.items():
        if not hasattr(st.session_state.design_config, field_name):
            setattr(st.session_state.design_config, field_name, default)
    for field_name, default in {"unit_id_column": "", "multiplicity_method": "Holm"}.items():
        if not hasattr(st.session_state.analysis_config, field_name):
            setattr(st.session_state.analysis_config, field_name, default)
    st.session_state.setdefault("raw_df", load_raw_demo())
    st.session_state.setdefault("historical_df", pd.DataFrame())
    st.session_state.setdefault("processing_result", None)
    st.session_state.setdefault("results_df", None)
    st.session_state.setdefault("analysis_results_by_role", {})
    st.session_state.setdefault("response_by_role", {})
    st.session_state.setdefault("historical_arm_stats_by_role", {})
    for metric in st.session_state.metrics_config.values():
        if not hasattr(metric, "observation_unit"):
            metric.observation_unit = "Months"
    st.session_state.setdefault("customer_analysis_fingerprint", "")
    st.session_state.setdefault("customer_analysis_is_current", False)
    st.session_state.setdefault("ts_config", TimeSeriesConfig())
    if not hasattr(st.session_state.ts_config, "planned_launch_date"):
        st.session_state.ts_config.planned_launch_date = ""
    if not hasattr(st.session_state.ts_config, "exposure_column"):
        st.session_state.ts_config.exposure_column = "None"
    st.session_state.setdefault("ts_intent", None)
    st.session_state.setdefault("ts_raw_df", pd.DataFrame())
    st.session_state.setdefault("ts_prepared_df", pd.DataFrame())
    st.session_state.setdefault("ts_prepost_result", None)
    st.session_state.setdefault("ts_bsts_result", None)
    st.session_state.setdefault("ts_prepost_fingerprint", "")
    st.session_state.setdefault("ts_structural_fingerprint", "")
    st.session_state.setdefault("ts_prepost_timestamp", "")
    st.session_state.setdefault("ts_structural_timestamp", "")
    st.session_state.setdefault("ts_structural_status", "idle")
    st.session_state.setdefault("ts_structural_started_at", "")
    st.session_state.setdefault("ts_structural_error", "")
    st.session_state.setdefault("ts_duration_plan", None)
    st.session_state.setdefault("ts_duration_plan_fingerprint", "")
    st.session_state.setdefault("geo_config", GeoConfig())
    st.session_state.setdefault("geo_raw_df", pd.DataFrame())
    st.session_state.setdefault("geo_panel_df", pd.DataFrame())
    st.session_state.setdefault("geo_assignment", pd.DataFrame())
    st.session_state.setdefault("geo_balance", {})
    st.session_state.setdefault("geo_analysis", None)
    st.session_state.setdefault("geo_plan", None)
    geo_defaults = GeoConfig()
    for name in geo_defaults.__dataclass_fields__:
        if not hasattr(st.session_state.geo_config, name):
            setattr(st.session_state.geo_config, name, getattr(geo_defaults, name))


def numeric_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for col in df.columns:
        values = safe_numeric_series(df, col)
        if len(values) and values.notna().sum() / max(len(values), 1) > 0.8:
            columns.append(col)
    return columns


def duration_unit(frequency: str) -> str:
    return {"Daily": "day", "Weekly": "week", "Monthly": "month"}.get(frequency, "period")


def duration_label(value: int, frequency: str) -> str:
    unit = duration_unit(frequency)
    return f"{value} {unit}{'' if value == 1 else 's'}"


def default_column(columns: list[str], candidates: list[str], fallback: int = 0) -> str:
    return infer_column(columns, candidates, columns[fallback] if columns else "")


def bounded_date(value: str, minimum, maximum, fallback):
    """Keep persisted Streamlit date defaults inside the current data bounds."""
    try:
        candidate = pd.Timestamp(value).date() if value else fallback
    except (TypeError, ValueError):
        candidate = fallback
    return min(max(candidate, minimum), maximum)


def selected_raw_preview(raw_df: pd.DataFrame, mapping: DataMappingConfig, population: PopulationConfig, metrics: dict[str, MetricConfig]) -> pd.DataFrame:
    if mapping.data_structure == "Cross-sectional (one row per unit)":
        if mapping.cpc_column in raw_df.columns and population.selected_cpcs:
            mask = first_series(raw_df, mapping.cpc_column).astype(str).isin([str(value) for value in population.selected_cpcs])
            return raw_df.loc[mask].copy()
        return raw_df.copy()
    if mapping.mob_column not in raw_df.columns:
        return pd.DataFrame()
    mob = safe_numeric_series(raw_df, mapping.mob_column)
    max_horizon = max((metric.mob_horizon for metric in metrics.values()), default=0)
    mask = mob.between(1, max_horizon)
    if mapping.cpc_column in raw_df.columns and population.selected_cpcs:
        mask &= first_series(raw_df, mapping.cpc_column).astype(str).isin([str(value) for value in population.selected_cpcs])
    useful = [mapping.unit_id_column, mapping.cpc_column, mapping.mob_column, st.session_state.strategy_config.historical_column]
    for extra in ["current_credit_line", "fico", "risk_segment"]:
        if extra not in useful:
            useful.append(extra)
    useful.extend(metric.source_column for metric in metrics.values())
    columns = [column for column in dict.fromkeys(useful) if column in raw_df.columns]
    return raw_df.loc[mask, columns].copy()


def default_metric_source(raw_df: pd.DataFrame, role: str, current: str) -> str:
    numeric = numeric_columns(raw_df)
    if current in numeric:
        return current
    candidates_by_role = {
        "Primary": ["revolving_balance", "balance"],
        "Secondary": ["risk_adjusted_revenue", "revenue", "rar"],
        "Guardrail": ["loss", "net_credit_loss", "default_flag"],
    }
    return default_column(numeric, candidates_by_role.get(role, []))


def maybe_money(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric) >= 100:
        return money(numeric)
    return number(numeric, 2)


def configured_arms(strategy: StrategyConfig) -> tuple[object, list[object]]:
    if strategy.strategy_type == "Numeric Strategy":
        return strategy.control_value, strategy.treatment_values
    return strategy.categorical_control, strategy.categorical_treatments


def strategy_point_stat(stats: pd.DataFrame, point: float, column: str) -> float:
    if stats.empty or column not in stats.columns:
        return float("nan")
    match = stats[(stats["Strategy Point"].astype(float) - float(point)).abs() < 1e-9]
    if match.empty or pd.isna(match.iloc[0][column]):
        return float("nan")
    return float(match.iloc[0][column])


def apply_experiment_template(strategy: StrategyConfig, template: str) -> None:
    strategy.experiment_template = template
    if template == "Credit Line":
        strategy.arm_format = "Ordered numeric levels"
        strategy.strategy_type = "Numeric Strategy"
        strategy.strategy_name = "Credit Line"
        strategy.assignment_column = "assigned_credit_line"
        strategy.historical_column = "current_credit_line"
        strategy.control_name = "BAU"
        strategy.control_value = 5000
        strategy.min_value = 2000
        strategy.max_value = 10000
        strategy.increment = 500
        strategy.treatment_values = [3000, 8000]
        strategy.treatment_names = ["LOW", "HIGH"]
    elif template == "Pricing / Fee":
        strategy.arm_format = "Ordered numeric levels"
        strategy.strategy_type = "Numeric Strategy"
        strategy.strategy_name = "Price or Fee"
        strategy.assignment_column = "assigned_price"
        strategy.control_name = "Current Price"
        strategy.control_value = 10
        strategy.min_value = 0
        strategy.max_value = 100
        strategy.increment = 1
        strategy.treatment_values = [9, 11]
        strategy.treatment_names = ["LOWER", "HIGHER"]
    elif template != "Custom":
        strategy.arm_format = "Named variants"
        presets = {
            "Marketing Offer": ("Marketing Offer", "BAU", ["Offer A", "Offer B"]),
            "Digital Experience": ("Experience", "Current Experience", ["Variant A"]),
            "Retention": ("Retention Treatment", "Standard Outreach", ["Enhanced Outreach", "Incentive"]),
        }
        name, control, treatments = presets[template]
        strategy.strategy_type = "Categorical Strategy"
        strategy.strategy_name = name
        strategy.assignment_column = "assigned_treatment"
        strategy.categorical_control = control
        strategy.categorical_treatments = treatments


def format_effect_table(df: pd.DataFrame, metric_type: str, strategy_label: str = "Strategy Arm") -> pd.DataFrame:
    out = df.copy().rename(columns={"Credit Line": strategy_label})
    out[strategy_label] = out[strategy_label].apply(maybe_money)
    out["Mean / Rate"] = out["Mean / Rate"].apply(lambda value: percent(value, 2) if metric_type == "Binary" else number(value, 2))
    out["Effect vs Control"] = out["Effect vs Control"].apply(lambda value: "" if pd.isna(value) else (percent(value, 2) if metric_type == "Binary" else number(value, 2)))
    out["Relative Lift"] = out["Relative Lift"].apply(lambda value: "" if pd.isna(value) else percent(value, 2))
    out["95% CI"] = df.apply(lambda row: "" if pd.isna(row["CI Lower"]) else f"{number(row['CI Lower'], 2)} to {number(row['CI Upper'], 2)}", axis=1)
    if "Adjusted p-value" not in out.columns:
        out["Adjusted p-value"] = out["p-value"]
    out["Adjusted p-value"] = out["Adjusted p-value"].apply(lambda value: "" if pd.isna(value) else p_value(value))
    return out[[strategy_label, "N", "Mean / Rate", "Effect vs Control", "Relative Lift", "95% CI", "Adjusted p-value", "Result"]]


def metric_sample_size(metric: MetricConfig, df: pd.DataFrame, alpha: float, target_power: float) -> int:
    stats = metric_baseline(df, metric.column, metric.metric_type)
    mde = abs(stats["baseline"] * metric.effect_value) if metric.effect_type == "Relative %" else abs(metric.effect_value)
    if metric.metric_type == "Binary":
        return sample_size_binary(alpha, target_power, stats["baseline"], max(mde, 1e-9), metric.direction)
    return sample_size_continuous(alpha, target_power, max(mde, 1e-9), max(stats["std_dev"], 1e-9))


def baseline_label(metric: MetricConfig) -> str:
    a = normalize_metric(metric)
    return percent(a.baseline, 2) if metric.metric_type == "Binary" else number(a.baseline, 2)


def variability_label(metric: MetricConfig) -> str:
    a = normalize_metric(metric)
    if metric.metric_type == "Binary":
        return "Binomial"
    return f"SD = {number(a.standard_deviation or 0, 2)}"


def guardrail_absolute_threshold(metric: MetricConfig | None) -> float | None:
    if metric is None or metric.guardrail_threshold is None:
        return None
    if metric.effect_type == "Relative %":
        return abs(float(metric.guardrail_threshold) * normalize_metric(metric).baseline)
    return abs(float(metric.guardrail_threshold))


def effect_value_label(metric: MetricConfig) -> str:
    return percent(metric.effect_value, 1) if metric.effect_type in {"Relative %", "Percentage Point"} else number(metric.effect_value, 2)


def show_overview() -> None:
    st.title("Experiment Platform")
    st.write("Internal Experiment Design & Measurement Platform for customer, geography, and time-series experimentation.")
    st.info("Customer-Level Experimentation, Time Series Analysis, and Geographic Test workflows are implemented in this MVP.")


def configure_metric_compact(role: str, raw_df: pd.DataFrame) -> None:
    numeric = numeric_columns(raw_df)
    cfg = st.session_state.metrics_config[role]
    cfg.source_column = default_metric_source(raw_df, role, cfg.source_column)
    if role == "Secondary" and cfg.source_column == "risk_adjusted_revenue" and cfg.name == "Cumulative Revenue":
        cfg.name = "Risk Adjusted Revenue"
    if role == "Primary" and cfg.source_column == "revolving_balance":
        cfg.name = "Revolving Balance" if cfg.name == "Average Revolving Balance" else cfg.name
    if role == "Guardrail" and cfg.source_column == "loss":
        cfg.name = "Loss" if cfg.name == "Cumulative Loss" else cfg.name
    cross_sectional = st.session_state.data_mapping.data_structure == "Cross-sectional (one row per unit)"
    cols = st.columns([1.4, 1.5, 1, 1] if cross_sectional else [1.4, 1.5, 1, 0.8, 1, 1])
    cfg.name = cols[0].text_input("Metric Name", cfg.name, key=f"compact_{role}_name")
    cfg.source_column = cols[1].selectbox("Source Column", numeric, index=numeric.index(cfg.source_column) if cfg.source_column in numeric else 0, key=f"compact_{role}_source")
    if cross_sectional:
        cfg.aggregation_method = "Average"
        cfg.metric_type = cols[2].selectbox("Metric Type", ["Continuous", "Binary"], index=["Continuous", "Binary"].index(cfg.metric_type), key=f"compact_{role}_type")
        cfg.direction = cols[3].selectbox("Direction", ["Higher is Better", "Lower is Better"], index=["Higher is Better", "Lower is Better"].index(cfg.direction), key=f"compact_{role}_dir")
    else:
        cfg.aggregation_method = cols[2].selectbox("Aggregation", ["Average", "Cumulative"], index=["Average", "Cumulative"].index(cfg.aggregation_method), key=f"compact_{role}_agg")
        mob_col = st.session_state.data_mapping.mob_column
        mobs = sorted(pd.to_numeric(raw_df[mob_col], errors="coerce").dropna().astype(int).unique().tolist()) if mob_col in raw_df.columns else [12, 36]
        cfg.mob_horizon = int(cols[3].selectbox("Through MOB", mobs, index=mobs.index(cfg.mob_horizon) if cfg.mob_horizon in mobs else 0, key=f"compact_{role}_mob"))
        cfg.metric_type = cols[4].selectbox("Metric Type", ["Continuous", "Binary"], index=["Continuous", "Binary"].index(cfg.metric_type), key=f"compact_{role}_type")
        cfg.direction = cols[5].selectbox("Direction", ["Higher is Better", "Lower is Better"], index=["Higher is Better", "Lower is Better"].index(cfg.direction), key=f"compact_{role}_dir")
    cfg.assumption_source = "Calculate from Historical Data"
    st.session_state.metrics_config[role] = cfg


def refresh_customer_processing():
    raw_df = st.session_state.raw_df
    mapping = st.session_state.data_mapping
    population = st.session_state.population_config
    result = build_analysis_dataset(raw_df, population, st.session_state.metrics_config, mapping, st.session_state.strategy_config.historical_column)
    st.session_state.processing_result = result
    st.session_state.historical_df = result.analysis_df
    return result


def customer_data_step() -> None:
    st.subheader("Data")
    st.session_state.experiment_name = st.text_input("Experiment Name", st.session_state.experiment_name, key="customer_experiment_name_v2")
    cfg: DataConfig = st.session_state.data_config
    source_options = ["Internal Data", "Upload File", "Synthetic Demo"]
    cfg.source_type = st.radio("Data Source", source_options, index=source_options.index(cfg.source_type) if cfg.source_type in source_options else 2, horizontal=True, key="customer_source_v2")
    if cfg.source_type == "Internal Data":
        c1, c2, c3 = st.columns(3)
        cfg.catalog = c1.text_input("Catalog", cfg.catalog, key="customer_catalog_v2")
        cfg.schema = c2.text_input("Schema", cfg.schema, key="customer_schema_v2")
        cfg.table = c3.text_input("Table", cfg.table, key="customer_table_v2")
        st.info("Internal Databricks access is not connected yet. Synthetic data is shown until the connector is configured.")
        st.session_state.raw_df = load_raw_demo()
    elif cfg.source_type == "Upload File":
        uploaded = st.file_uploader("Upload customer data", type=["csv", "parquet"], key="customer_upload_v2")
        if uploaded is not None:
            try:
                signature = hash(uploaded.getvalue())
                if st.session_state.get("customer_upload_signature") != signature:
                    uploaded.seek(0)
                    st.session_state.raw_df = normalize_uploaded_dataset(pd.read_parquet(uploaded) if uploaded.name.endswith(".parquet") else load_uploaded_csv(uploaded))
                    st.session_state.customer_upload_signature = signature
                    st.session_state.processing_result = None
                    st.session_state.historical_df = pd.DataFrame()
                    st.session_state.analysis_results_by_role = {}
                cfg.dataset_name = uploaded.name
            except Exception as exc:
                st.error("We could not read this file. Confirm that it is a valid CSV or Parquet dataset.")
                print(f"Customer upload parsing failed: {exc}")
    else:
        st.session_state.raw_df = load_raw_demo()
        cfg.dataset_name = "Synthetic Raw Portfolio Data"

    raw_df = st.session_state.raw_df
    mapping: DataMappingConfig = st.session_state.data_mapping
    population: PopulationConfig = st.session_state.population_config
    columns = list(raw_df.columns)
    mapping.unit_id_column = mapping.unit_id_column if mapping.unit_id_column in columns else default_column(columns, ["customer_id", "account_id", "user_id", "member_id"])
    segment_candidates = {"cpc", "portfolio", "product", "segment", "market", "cohort"}
    mapping.cpc_column = mapping.cpc_column if mapping.cpc_column in columns else next((column for column in columns if column.lower() in segment_candidates), "")
    mapping.mob_column = mapping.mob_column if mapping.mob_column in columns else infer_mob_column(raw_df)
    dates = date_like_columns(raw_df)
    mapping.booking_date_column = mapping.booking_date_column if mapping.booking_date_column in dates else next((column for column in dates if column.lower() in {"booking_date", "vintage_date", "open_date", "start_date"}), "")

    mapping.data_structure = st.radio("Dataset Structure", ["Longitudinal (unit x period)", "Cross-sectional (one row per unit)"], index=["Longitudinal (unit x period)", "Cross-sectional (one row per unit)"].index(mapping.data_structure), horizontal=True, key="customer_structure_v2")
    available_segments = sorted(raw_df[mapping.cpc_column].dropna().astype(str).unique().tolist()) if mapping.cpc_column else []
    if available_segments:
        if not population.selected_cpcs:
            population.selected_cpcs = [value for value in ["CPC_A", "CPC_B"] if value in available_segments] or available_segments
        population.selected_cpcs = st.multiselect("Population Segments", available_segments, default=[value for value in population.selected_cpcs if value in available_segments], key="customer_segments_v2")
    else:
        population.selected_cpcs = []
        st.caption("Population: all eligible units")

    with st.expander("Advanced Data Mapping", expanded=False):
        c1, c2 = st.columns(2)
        mapping.unit_id_column = c1.selectbox("Experimental Unit ID", columns, index=columns.index(mapping.unit_id_column), key="customer_unit_map_v2")
        segment_options = ["None"] + columns
        selected_segment = c2.selectbox("Population Segment Column", segment_options, index=segment_options.index(mapping.cpc_column) if mapping.cpc_column in columns else 0, key="customer_segment_map_v2")
        mapping.cpc_column = "" if selected_segment == "None" else selected_segment
        if mapping.data_structure == "Longitudinal (unit x period)":
            c1, c2, c3 = st.columns(3)
            mapping.mob_column = c1.selectbox("Observation Period", columns, index=columns.index(mapping.mob_column) if mapping.mob_column in columns else 0, key="customer_period_map_v2")
            date_options = ["None"] + dates
            selected_date = c2.selectbox("Cohort Start Date", date_options, index=date_options.index(mapping.booking_date_column) if mapping.booking_date_column in dates else 0, key="customer_date_map_v2")
            mapping.booking_date_column = "" if selected_date == "None" else selected_date
            if mapping.booking_date_column and mapping.mob_column in columns:
                parsed = pd.to_datetime(raw_df[mapping.booking_date_column], errors="coerce")
                max_period = safe_numeric_series(raw_df, mapping.mob_column).max()
                inferred_cutoff = parsed.max() + pd.DateOffset(months=int(max_period)) if parsed.notna().any() and pd.notna(max_period) else pd.Timestamp.today()
                cutoff = pd.Timestamp(mapping.cutoff_date).date() if mapping.cutoff_date else inferred_cutoff.date()
                mapping.cutoff_date = c3.date_input("Data Cutoff Date", value=cutoff, key="customer_cutoff_v2").isoformat()

    try:
        result = refresh_customer_processing()
    except Exception as exc:
        st.error(str(exc))
        return
    d = result.diagnostics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{d['Raw Records']:,.0f}")
    c2.metric("Unique Units", f"{d['Unique Accounts']:,.0f}")
    c3.metric("Primary Eligible", f"{d['Final Analysis Population']:,.0f}")
    c4.metric("Segments", f"{len(available_segments):,.0f}" if available_segments else "All")
    if d["Final Analysis Population"] == 0:
        st.error("No units are eligible for the primary metric. Review the population and observation window.")
    with st.expander("Eligibility Details", expanded=False):
        st.write(f"Longest configured observation window: {d['Longest Required Horizon']} periods")
        st.write(f"Complete for every configured metric: {d['Complete Eligible Accounts']:,.0f}")
        st.write(f"Excluded for insufficient maturity: {d['Excluded for Insufficient Maturity']:,.0f}")
        st.write(f"Excluded from the all-metric cohort for incomplete history: {d['Excluded for Data Completeness']:,.0f}")

    st.markdown("**Data Preview**")
    preview_mode = st.radio("Preview", ["Raw Data", "Processed Metrics", "Selected Raw Rows"], horizontal=True, label_visibility="collapsed", key="customer_preview_v2")
    if preview_mode == "Raw Data":
        preview = raw_df.head(20)
        caption = f"Showing 20 of {len(raw_df):,.0f} source rows · {len(raw_df.columns)} columns"
    elif preview_mode == "Selected Raw Rows":
        selected = selected_raw_preview(raw_df, mapping, population, st.session_state.metrics_config)
        preview = selected.head(20)
        caption = f"Showing 20 of {len(selected):,.0f} selected source rows · {len(selected.columns)} columns"
    else:
        display_names = {mapping.unit_id_column: "Experimental Unit"}
        if mapping.cpc_column:
            display_names[mapping.cpc_column] = "Population Segment"
        for metric in st.session_state.metrics_config.values():
            display_names[metric.processed_column] = f"{metric.name} ({'Through MOB ' + str(metric.mob_horizon) if mapping.data_structure.startswith('Longitudinal') else 'As observed'})"
        preview = result.analysis_df.head(20).rename(columns=display_names)
        caption = f"Showing 20 of {len(result.analysis_df):,.0f} analysis-ready units"
    st.caption(caption)
    st.dataframe(preview, use_container_width=True, height=460)
    if not result.cpc_breakdown.empty:
        with st.expander("Population Segment Breakdown", expanded=False):
            breakdown = result.cpc_breakdown.copy()
            breakdown["Share"] = breakdown["Share"].apply(lambda value: percent(value, 1))
            st.dataframe(breakdown, use_container_width=True, hide_index=True)


def configure_metric_panel(role: str, raw_df: pd.DataFrame) -> None:
    numeric = numeric_columns(raw_df)
    cfg = st.session_state.metrics_config[role]
    cfg.source_column = default_metric_source(raw_df, role, cfg.source_column)
    c1, c2, c3, c4 = st.columns([1.3, 1.5, 1, 1])
    cfg.name = c1.text_input("Metric Name", cfg.name, key=f"metric_panel_name_{role}")
    cfg.source_column = c2.selectbox("Source Column", numeric, index=numeric.index(cfg.source_column) if cfg.source_column in numeric else 0, key=f"metric_panel_source_{role}")
    cfg.metric_type = c3.selectbox("Metric Type", ["Continuous", "Binary"], index=["Continuous", "Binary"].index(cfg.metric_type), key=f"metric_panel_type_{role}")
    cfg.direction = c4.selectbox("Success Direction", ["Higher is Better", "Lower is Better"], index=["Higher is Better", "Lower is Better"].index(cfg.direction), key=f"metric_panel_direction_{role}")
    if st.session_state.data_mapping.data_structure == "Longitudinal (unit x period)":
        mob_col = st.session_state.data_mapping.mob_column
        periods = sorted(pd.to_numeric(raw_df[mob_col], errors="coerce").dropna().astype(int).unique().tolist()) if mob_col in raw_df.columns else [12]
        c1, c2 = st.columns(2)
        cfg.aggregation_method = c1.selectbox("Unit-Level Aggregation", ["Average", "Cumulative"], index=["Average", "Cumulative"].index(cfg.aggregation_method), key=f"metric_panel_agg_{role}")
        cfg.mob_horizon = int(c2.selectbox("Observation Window", periods, index=periods.index(cfg.mob_horizon) if cfg.mob_horizon in periods else 0, format_func=lambda value: f"Through MOB {value}", key=f"metric_panel_window_{role}"))
    else:
        cfg.aggregation_method = "Average"
        st.caption("One outcome value per experimental unit")
    if role == "Guardrail":
        threshold_scales = ["Absolute", "Relative %"] if cfg.metric_type == "Continuous" else ["Percentage Point", "Relative %"]
        if cfg.effect_type not in threshold_scales:
            cfg.effect_type = threshold_scales[0]
            if cfg.metric_type == "Binary":
                cfg.guardrail_threshold = 0.005
        threshold_left, threshold_right = st.columns(2)
        previous_scale = cfg.effect_type
        cfg.effect_type = threshold_left.selectbox("Harm Threshold Scale", threshold_scales, index=threshold_scales.index(cfg.effect_type), key="guardrail_threshold_scale_v3")
        if cfg.effect_type != previous_scale:
            current_threshold = 0.005 if cfg.effect_type == "Percentage Point" else 0.05 if cfg.effect_type == "Relative %" else abs(normalize_metric(cfg).baseline * float(cfg.guardrail_threshold or 0.05))
        else:
            current_threshold = float(cfg.guardrail_threshold or cfg.effect_value)
        if cfg.effect_type in {"Relative %", "Percentage Point"}:
            shown_threshold = threshold_right.number_input("Maximum Acceptable Harm (%)", min_value=0.001, max_value=99.999, value=min(current_threshold * 100, 99.999), step=0.1, key=f"guardrail_threshold_percent_{cfg.effect_type}_v3")
            cfg.guardrail_threshold = shown_threshold / 100
        else:
            cfg.guardrail_threshold = threshold_right.number_input("Maximum Acceptable Harm", min_value=0.0001, value=current_threshold, step=1.0, key="guardrail_threshold_absolute_v3")
        cfg.effect_value = cfg.guardrail_threshold
    cfg.assumption_source = "Calculate from Historical Data"
    st.session_state.metrics_config[role] = cfg


def customer_metrics_step() -> None:
    st.subheader("Metrics")
    raw_df = st.session_state.raw_df
    st.caption("Historical assumptions update from each metric's eligible experimental units.")
    c1, c2 = st.columns(2)
    secondary_enabled = c1.checkbox("Include Secondary Metric", value="Secondary" in st.session_state.metrics_config, key="enable_secondary_v2")
    guardrail_enabled = c2.checkbox("Include Guardrail Metric", value="Guardrail" in st.session_state.metrics_config, key="enable_guardrail_v2")
    if secondary_enabled:
        st.session_state.metrics_config.setdefault("Secondary", MetricConfig("Revenue", "revenue", "Secondary", source_column="revenue", aggregation_method="Cumulative", mob_horizon=36))
    else:
        st.session_state.metrics_config.pop("Secondary", None)
    if guardrail_enabled:
        st.session_state.metrics_config.setdefault("Guardrail", MetricConfig("Loss", "loss", "Guardrail", direction="Lower is Better", source_column="loss", aggregation_method="Cumulative", mob_horizon=36))
    else:
        st.session_state.metrics_config.pop("Guardrail", None)
    roles = [role for role in ["Primary", "Secondary", "Guardrail"] if role in st.session_state.metrics_config]
    metric_tabs = st.tabs([f"{role}: {st.session_state.metrics_config[role].name}" for role in roles])
    for tab, role in zip(metric_tabs, roles):
        with tab:
            configure_metric_panel(role, raw_df)
    try:
        result = refresh_customer_processing()
    except Exception as exc:
        st.error(str(exc))
        return
    summary = result.metric_preview.copy()
    for column in ["Baseline", "SD"]:
        if column in summary.columns:
            summary[column] = summary[column].map(lambda value: round(value, 3) if pd.notna(value) else value)
    if st.session_state.strategy_config.strategy_type == "Numeric Strategy" and "SD" in summary.columns:
        summary = summary.drop(columns=["SD"])
        summary["Variability"] = "Calculated by selected historical grouping in Design"
    st.markdown("**Historical Assumptions**")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    if summary["Eligible N"].nunique() > 1:
        st.caption("Eligible sample sizes differ because each metric uses its own observation window. Primary planning uses the primary metric cohort unless configured otherwise in Design.")

    strategy: StrategyConfig = st.session_state.strategy_config
    historical_df = result.analysis_df
    historical_column = strategy.historical_column
    if strategy.strategy_type == "Numeric Strategy" and historical_column in historical_df.columns:
        st.markdown("**Metric History by Acquisition Line**")
        curve_left, curve_middle = st.columns([1.15, 1])
        binning_options = ["Automatic fine bins", "Fixed bin width", "Equal-count bins"]
        strategy.curve_binning_method = curve_left.selectbox(
            "Curve Binning",
            binning_options,
            index=binning_options.index(strategy.curve_binning_method) if strategy.curve_binning_method in binning_options else 0,
            key="metric_curve_binning_v4",
        )
        if strategy.curve_binning_method == "Fixed bin width":
            strategy.curve_bin_width = curve_middle.number_input(
                "Bin Width",
                min_value=0.01,
                value=float(strategy.curve_bin_width),
                step=max(float(strategy.increment), 0.01),
                key="metric_curve_width_v4",
            )
        elif strategy.curve_binning_method == "Equal-count bins":
            strategy.curve_bin_count = int(curve_middle.number_input("Number of Bins", min_value=4, max_value=50, value=int(strategy.curve_bin_count), step=1, key="metric_curve_count_v4"))
        else:
            curve_middle.caption("Uses 6–30 fine bins based on the eligible customer count.")
        visual_tabs = st.tabs([f"{role}: {st.session_state.metrics_config[role].name}" for role in roles])
        for visual_tab, role in zip(visual_tabs, roles):
            with visual_tab:
                metric = st.session_state.metrics_config[role]
                metric_column = metric.processed_column or metric.column
                st.plotly_chart(
                    historical_association(
                        historical_df,
                        historical_column,
                        metric_column,
                        metric.metric_type,
                        strategy_label=strategy.strategy_name,
                        metric_label=metric.name,
                        binning_method=strategy.curve_binning_method,
                        bin_width=float(strategy.curve_bin_width),
                        bin_count=int(strategy.curve_bin_count),
                    ),
                    use_container_width=True,
                    key=f"metric_history_chart_{role}_v4",
                )
        st.caption("The line connects fine-bin historical means; vertical intervals are 95% confidence intervals. These are descriptive historical associations, not causal estimates.")


def data_metrics_step() -> None:
    st.subheader("Data & Metrics")
    st.session_state.experiment_name = st.text_input("Experiment Name", st.session_state.experiment_name)
    left, right = st.columns([0.9, 1.25])
    with left:
        cfg: DataConfig = st.session_state.data_config
        st.markdown("**Data Source**")
        options = ["Internal Data", "Upload File", "Synthetic Demo"]
        cfg.source_type = st.radio("Data Source", options, index=options.index(cfg.source_type) if cfg.source_type in options else 2, horizontal=True, label_visibility="collapsed")
        if cfg.source_type == "Internal Data":
            c1, c2, c3 = st.columns(3)
            cfg.catalog = c1.text_input("Catalog", cfg.catalog)
            cfg.schema = c2.text_input("Schema", cfg.schema)
            cfg.table = c3.text_input("Table", cfg.table)
            st.info("Internal Databricks access is a placeholder. Synthetic data is used locally until the connector is added.")
            st.session_state.raw_df = load_raw_demo()
            cfg.dataset_name = "Synthetic Raw Portfolio Data"
        elif cfg.source_type == "Upload File":
            uploaded = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])
            if uploaded is not None:
                try:
                    st.session_state.raw_df = normalize_uploaded_dataset(pd.read_parquet(uploaded) if uploaded.name.endswith(".parquet") else load_uploaded_csv(uploaded))
                    cfg.dataset_name = uploaded.name
                    st.session_state.processing_result = None
                    st.session_state.historical_df = pd.DataFrame()
                except Exception as exc:
                    st.error("Data validation issue: We could not read the uploaded file. Confirm it is a valid CSV or Parquet file.")
                    print(f"Upload parsing failed: {exc}")
        else:
            st.session_state.raw_df = load_raw_demo()
            cfg.dataset_name = "Synthetic Raw Portfolio Data"

        raw_df = st.session_state.raw_df
        mapping: DataMappingConfig = st.session_state.data_mapping
        columns = list(raw_df.columns)
        mapping.data_structure = st.radio("Dataset Structure", ["Longitudinal (unit x period)", "Cross-sectional (one row per unit)"], index=["Longitudinal (unit x period)", "Cross-sectional (one row per unit)"].index(mapping.data_structure), horizontal=True)
        mapping.unit_id_column = mapping.unit_id_column if mapping.unit_id_column in columns else default_column(columns, ["customer_id", "account_id", "user_id", "member_id"])
        segment_candidates = ["cpc", "portfolio", "product", "segment", "market", "cohort"]
        inferred_segment = default_column(columns, segment_candidates) if any(candidate in [column.lower() for column in columns] for candidate in segment_candidates) else ""
        mapping.cpc_column = mapping.cpc_column if mapping.cpc_column in columns else inferred_segment
        numeric = numeric_columns(raw_df)
        mapping.mob_column = mapping.mob_column if mapping.mob_column in columns else infer_mob_column(raw_df)
        mapping.booking_date_column = mapping.booking_date_column if mapping.booking_date_column in columns else default_column(columns, ["booking_date", "vintage_date", "open_date"], 0)
        if mapping.data_structure == "Longitudinal (unit x period)" and mapping.booking_date_column in columns and mapping.mob_column in columns:
            parsed_dates = pd.to_datetime(raw_df[mapping.booking_date_column], errors="coerce")
            max_mob = safe_numeric_series(raw_df, mapping.mob_column).max()
            if parsed_dates.notna().any() and pd.notna(max_mob):
                inferred_cutoff = parsed_dates.max() + pd.DateOffset(months=int(max_mob))
                mapping.cutoff_date = mapping.cutoff_date or inferred_cutoff.date().isoformat()

        st.markdown("**Population**")
        available_cpcs = sorted(raw_df[mapping.cpc_column].dropna().astype(str).unique().tolist()) if mapping.cpc_column and mapping.cpc_column in raw_df.columns else []
        population: PopulationConfig = st.session_state.population_config
        if not population.selected_cpcs:
            population.selected_cpcs = [cpc for cpc in ["CPC_A", "CPC_B"] if cpc in available_cpcs] or available_cpcs[:2]
        if available_cpcs:
            population.selected_cpcs = st.multiselect("Population Segments", available_cpcs, default=[c for c in population.selected_cpcs if c in available_cpcs])
            st.caption("Selected Population: " + (" + ".join(population.selected_cpcs) if population.selected_cpcs else "All segments"))
        else:
            population.selected_cpcs = []
            st.caption("Selected Population: all eligible units")
        with st.expander("Advanced Data Mapping", expanded=False):
            mapping.unit_id_column = st.selectbox("Experimental Unit ID", columns, index=columns.index(mapping.unit_id_column))
            segment_options = ["None"] + columns
            current_segment = mapping.cpc_column if mapping.cpc_column in columns else "None"
            segment = st.selectbox("Population Segment Column (optional)", segment_options, index=segment_options.index(current_segment))
            mapping.cpc_column = "" if segment == "None" else segment
            if mapping.data_structure == "Longitudinal (unit x period)":
                mapping.mob_column = st.selectbox("Observation Period Column", columns, index=columns.index(mapping.mob_column) if mapping.mob_column in columns else 0)
                date_options = ["None"] + columns
                current_booking = mapping.booking_date_column if mapping.booking_date_column in columns else "None"
                booking = st.selectbox("Cohort Start Date", date_options, index=date_options.index(current_booking))
                mapping.booking_date_column = "" if booking == "None" else booking
                if mapping.booking_date_column and mapping.cutoff_date:
                    mapping.cutoff_date = st.date_input("Data Cutoff Date", value=pd.Timestamp(mapping.cutoff_date).date()).isoformat()

        st.markdown("**Metrics**")
        configure_metric_compact("Primary", raw_df)
        if st.checkbox("Add Secondary Metric", value="Secondary" in st.session_state.metrics_config):
            st.session_state.metrics_config.setdefault("Secondary", MetricConfig("Revenue", "cum_revenue_mob36", "Secondary", source_column="revenue", aggregation_method="Cumulative", mob_horizon=36))
            configure_metric_compact("Secondary", raw_df)
        else:
            st.session_state.metrics_config.pop("Secondary", None)
        if st.checkbox("Add Guardrail", value="Guardrail" in st.session_state.metrics_config):
            st.session_state.metrics_config.setdefault("Guardrail", MetricConfig("Loss", "cum_loss_mob36", "Guardrail", direction="Lower is Better", source_column="loss", aggregation_method="Cumulative", mob_horizon=36))
            configure_metric_compact("Guardrail", raw_df)
        else:
            st.session_state.metrics_config.pop("Guardrail", None)

    with right:
        raw_df = st.session_state.raw_df
        mapping = st.session_state.data_mapping
        population = st.session_state.population_config
        try:
            warnings = data_quality_warnings(raw_df, mapping.unit_id_column, mapping.cpc_column, mapping.mob_column) if mapping.data_structure == "Longitudinal (unit x period)" else ([] if mapping.unit_id_column in raw_df.columns else ["Experimental Unit ID mapping is required."])
            if warnings:
                with st.expander(f"{len(warnings)} data quality warning{'s' if len(warnings) != 1 else ''}", expanded=False):
                    for warning in warnings:
                        st.warning(warning)
            result = build_analysis_dataset(raw_df, population, st.session_state.metrics_config, mapping, st.session_state.strategy_config.historical_column)
            st.session_state.processing_result = result
            st.session_state.historical_df = result.analysis_df
            summary = result.metric_preview
            st.markdown("**Population Summary**")
            d = result.diagnostics
            mob_values = safe_numeric_series(raw_df, mapping.mob_column).dropna() if mapping.data_structure == "Longitudinal (unit x period)" else pd.Series(dtype=float)
            mob_text = f"Periods {int(mob_values.min())}–{int(mob_values.max())}" if not mob_values.empty else "One row per unit"
            segments_count = raw_df[mapping.cpc_column].nunique() if mapping.cpc_column and mapping.cpc_column in raw_df.columns else 0
            segment_text = f"{segments_count:,.0f} segments" if segments_count else "all units"
            st.success(f"Data loaded successfully | {len(raw_df):,.0f} rows | {d['Unique Accounts']:,.0f} units | {segment_text} | {mob_text} | 0 critical validation issues")
            if d["Selected CPC Accounts"] == 0:
                st.error("The selected population segments have no records.")
            if d["Final Analysis Population"] == 0:
                st.error("No eligible experimental units remain after maturity and completeness checks.")
            horizon_text = f"Required Observation Horizon: {d['Longest Required Horizon']} periods | " if mapping.data_structure == "Longitudinal (unit x period)" else ""
            st.write(f"Raw Records: {d['Raw Records']:,.0f} | Unique Units: {d['Unique Accounts']:,.0f} | {horizon_text}Eligible Units: {d['Final Analysis Population']:,.0f} | Excluded for Insufficient Maturity: {d['Excluded for Insufficient Maturity']:,.0f}")
            if d["Duplicate Unit-MOB Rows"]:
                st.warning(f"{d['Duplicate Unit-MOB Rows']:,.0f} duplicate unit-period rows were found. Duplicates are averaged before metric aggregation.")
            if d["Excluded for Data Completeness"]:
                st.warning(f"{d['Excluded for Data Completeness']:,.0f} mature units were excluded from the common joint-analysis cohort. Metric assumptions below use each metric's own eligible history.")
            st.markdown("**Metric Summary**")
            st.dataframe(summary, use_container_width=True, hide_index=True)
            view = st.radio("Preview", ["Processed Data", "Selected Rows", "Raw Data"], horizontal=True)
            if view == "Processed Data":
                st.caption(f"Showing 20 of {len(result.analysis_df):,.0f} eligible experimental units.")
                st.dataframe(result.analysis_df.head(20), use_container_width=True)
            elif view == "Selected Rows":
                selected_preview = selected_raw_preview(raw_df, mapping, population, st.session_state.metrics_config)
                st.caption(f"Showing 20 of {len(selected_preview):,.0f} selected raw rows.")
                st.dataframe(selected_preview.head(20), use_container_width=True)
            else:
                st.caption(f"Showing 20 of {len(raw_df):,.0f} raw records.")
                st.dataframe(raw_df.head(20), use_container_width=True)
            st.markdown("**Population Segment Breakdown**")
            breakdown = result.cpc_breakdown.copy()
            if not breakdown.empty:
                breakdown["Share"] = breakdown["Share"].apply(lambda value: percent(value, 1))
            st.dataframe(breakdown, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error("Data validation issue: We could not build the eligible experiment population. Review the unit, segment, observation-period, and metric mappings.")
            print(f"Population processing failed: {exc}")


def data_step() -> None:
    st.subheader("Data Source")
    cfg: DataConfig = st.session_state.data_config
    options = ["Upload File", "Internal Data", "Synthetic Demo"]
    cfg.source_type = st.radio("Data Source", options, index=options.index(cfg.source_type) if cfg.source_type in options else 2, horizontal=True)
    if cfg.source_type == "Internal Data":
        c1, c2, c3 = st.columns(3)
        cfg.catalog = c1.text_input("Catalog", cfg.catalog)
        cfg.schema = c2.text_input("Schema", cfg.schema)
        cfg.table = c3.text_input("Table", cfg.table)
        st.info("Internal Databricks table access is represented as a modular data source. Credentials and workspace-specific settings are intentionally not hard-coded.")
        if st.button("Use Synthetic Data Until Internal Connector Is Added"):
            st.session_state.raw_df = load_raw_demo()
            cfg.dataset_name = "Synthetic Raw Portfolio Data"
    elif cfg.source_type == "Upload File":
        uploaded = st.file_uploader("Upload Customer-Level Dataset", type=["csv", "parquet"])
        if uploaded is not None:
            st.session_state.raw_df = pd.read_parquet(uploaded) if uploaded.name.endswith(".parquet") else load_uploaded_csv(uploaded)
            cfg.dataset_name = uploaded.name
    else:
        st.session_state.raw_df = load_raw_demo()
        cfg.dataset_name = "Synthetic Raw Portfolio Data"
    df = st.session_state.raw_df
    st.markdown('<div class="subtle-note">Raw data may contain multiple monthly observations per account. The platform will construct one common eligible row per experimental unit before design.</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Name", cfg.dataset_name)
    c2.metric("Source Type", cfg.source_type)
    c3.metric("Rows", f"{len(df):,.0f}")
    c4.metric("Columns", f"{len(df.columns):,.0f}")
    st.markdown("**Preview**")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown("**Detected Column Types**")
    st.dataframe(column_schema(df), use_container_width=True, hide_index=True)


def population_step() -> None:
    st.subheader("Experiment Population")
    raw_df = st.session_state.raw_df
    mapping: DataMappingConfig = st.session_state.data_mapping
    population: PopulationConfig = st.session_state.population_config
    cols = list(raw_df.columns)
    numeric = numeric_columns(raw_df)
    c1, c2, c3 = st.columns(3)
    mapping.unit_id_column = c1.selectbox("Experimental Unit ID", cols, index=cols.index(mapping.unit_id_column) if mapping.unit_id_column in cols else 0)
    mapping.cpc_column = c2.selectbox("CPC Column", cols, index=cols.index(mapping.cpc_column) if mapping.cpc_column in cols else 0)
    mapping.mob_column = c3.selectbox("MOB Column", numeric, index=numeric.index(mapping.mob_column) if mapping.mob_column in numeric else 0)
    date_options = ["None"] + cols
    current_booking = mapping.booking_date_column if mapping.booking_date_column in cols else "None"
    booking = st.selectbox("Booking Date", date_options, index=date_options.index(current_booking))
    mapping.booking_date_column = "" if booking == "None" else booking
    if mapping.booking_date_column:
        inferred_cutoff = pd.to_datetime(raw_df[mapping.booking_date_column], errors="coerce").max() + pd.DateOffset(months=int(pd.to_numeric(raw_df[mapping.mob_column], errors="coerce").max()))
        default_cutoff = pd.Timestamp(mapping.cutoff_date) if mapping.cutoff_date else inferred_cutoff
        mapping.cutoff_date = st.date_input("Data Cutoff Date", value=default_cutoff.date()).isoformat()
    available_cpcs = sorted(raw_df[mapping.cpc_column].dropna().astype(str).unique().tolist()) if mapping.cpc_column in raw_df.columns else []
    if not population.selected_cpcs:
        population.selected_cpcs = available_cpcs[:2]
    population.selected_cpcs = st.multiselect("CPCs", available_cpcs, default=[cpc for cpc in population.selected_cpcs if cpc in available_cpcs])
    st.write("Selected CPC Group: " + (" + ".join(population.selected_cpcs) if population.selected_cpcs else "None selected"))
    population.completeness_policy = st.selectbox("Completeness Policy", ["Exclude incomplete metric histories"], index=0)
    if st.button("Build Analysis Population", type="primary"):
        try:
            result = build_analysis_dataset(raw_df, population, st.session_state.metrics_config, mapping, st.session_state.strategy_config.historical_column)
            st.session_state.processing_result = result
            st.session_state.historical_df = result.analysis_df
            st.success("Analysis population created.")
        except ValueError as exc:
            st.error(str(exc))
    result = st.session_state.processing_result
    if result is not None:
        st.markdown("**Analysis Population Preview**")
        d = result.diagnostics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Records", f"{d['Raw Records']:,.0f}")
        c2.metric("Selected CPC Accounts", f"{d['Selected CPC Accounts']:,.0f}")
        c3.metric("Common Observation Horizon", f"MOB {d['Longest Required Horizon']:,.0f}")
        c4.metric("Final Analysis Population", f"{d['Final Analysis Population']:,.0f}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mature Accounts", f"{d['Mature Accounts']:,.0f}")
        c2.metric("Complete Eligible Accounts", f"{d['Complete Eligible Accounts']:,.0f}")
        c3.metric("Excluded for Insufficient Maturity", f"{d['Excluded for Insufficient Maturity']:,.0f}")
        c4.metric("Excluded for Data Completeness", f"{d['Excluded for Data Completeness']:,.0f}")
        if d["Duplicate Unit-MOB Rows"]:
            st.warning(f"{d['Duplicate Unit-MOB Rows']:,.0f} duplicate unit × MOB rows were found. Duplicates are averaged within month before metric aggregation.")
        if d["Mature Accounts"] and d["Excluded for Data Completeness"] / d["Mature Accounts"] > 0.05:
            st.warning(f"{percent(d['Excluded for Data Completeness'] / d['Mature Accounts'], 1)} of otherwise mature accounts have incomplete monthly observations within the required MOB window. Review metric-specific handling before final experiment design.")
        st.markdown("**Metric Preview**")
        st.dataframe(result.metric_preview, use_container_width=True, hide_index=True)
        st.markdown("**CPC Breakdown**")
        breakdown = result.cpc_breakdown.copy()
        if not breakdown.empty:
            breakdown["Share"] = breakdown["Share"].apply(lambda v: percent(v, 1))
        st.dataframe(breakdown, use_container_width=True, hide_index=True)


def strategy_step() -> None:
    st.subheader("Treatment Setup")
    strategy: StrategyConfig = st.session_state.strategy_config
    raw_df = st.session_state.raw_df
    columns = list(raw_df.columns)
    setup_left, setup_middle, setup_right = st.columns([1.1, 1, 1.35])
    strategy.strategy_name = setup_left.text_input("Treatment Dimension", strategy.strategy_name, key="treatment_dimension_v3")
    format_options = ["Named variants", "Ordered numeric levels"]
    strategy.arm_format = setup_middle.selectbox("Arm Format", format_options, index=format_options.index(strategy.arm_format), help="Use ordered levels for quantities such as credit lines or prices. Use named variants for offers, messages, or designs.", key="arm_format_v3")
    strategy.strategy_type = "Numeric Strategy" if strategy.arm_format == "Ordered numeric levels" else "Categorical Strategy"
    assignment_options = list(dict.fromkeys(([strategy.assignment_column] if strategy.assignment_column else []) + columns))
    if not assignment_options:
        assignment_options = ["assigned_treatment"]
    strategy.assignment_column = setup_right.selectbox(
        "Experiment Assignment Column",
        assignment_options,
        index=assignment_options.index(strategy.assignment_column) if strategy.assignment_column in assignment_options else 0,
        format_func=lambda column: column if column in columns else f"{column} (expected after launch)",
        help="This column identifies the randomized arm in experiment-result data. It may not exist in historical planning data yet.",
        key="assignment_column_v3",
    )
    if strategy.assignment_column not in columns:
        st.caption(f"{strategy.assignment_column} is expected in experiment-result data and is not required in this historical planning dataset.")

    if strategy.arm_format == "Ordered numeric levels":
        numeric = numeric_columns(raw_df)
        if not numeric:
            st.error("A numeric historical acquisition-line column is required for ordered treatment levels.")
            st.session_state.historical_arm_stats_by_role = {}
            return
        strategy.historical_column = st.selectbox(
            "Historical Acquisition Line Column",
            numeric,
            index=numeric.index(strategy.historical_column) if strategy.historical_column in numeric else 0,
            key="historical_level_v4",
            help="This value must be fixed for each customer across every historical record.",
        )
        fixed_errors = validate_fixed_unit_value(raw_df, st.session_state.data_mapping.unit_id_column, strategy.historical_column)
        if fixed_errors:
            for error in fixed_errors:
                st.error(error)
            st.session_state.historical_arm_stats_by_role = {}
            return
        try:
            result = refresh_customer_processing()
        except ValueError as exc:
            st.error(str(exc))
            st.session_state.historical_arm_stats_by_role = {}
            return
        historical_df = result.analysis_df
        primary = st.session_state.metrics_config["Primary"]
        primary_column = primary.processed_column or primary.column

        with st.expander("Allowed Numeric Range", expanded=False):
            c1, c2, c3 = st.columns(3)
            strategy.min_value = c1.number_input("Minimum Allowed Level", value=float(strategy.min_value), key="arm_min_v3")
            strategy.max_value = c2.number_input("Maximum Allowed Level", value=float(strategy.max_value), key="arm_max_v3")
            strategy.increment = c3.number_input("Allowed Increment", min_value=0.0001, value=float(strategy.increment), key="arm_increment_v3")

        strategy.selection_mode = st.radio(
            "Line Selection",
            ["Suggested lines", "Manual lines"],
            index=0 if strategy.selection_mode == "Suggested lines" else 1,
            horizontal=True,
            key="line_selection_mode_v4",
        )
        st.markdown("**Proposed Experiment Points**")
        if strategy.selection_mode == "Suggested lines":
            suggest_left, suggest_right = st.columns(2)
            strategy.control_value = suggest_left.number_input("Control / BAU Line", value=float(strategy.control_value), step=float(strategy.increment), key="suggested_control_v4")
            treatment_count = int(suggest_right.number_input("Number of Treatment Lines", min_value=1, max_value=4, value=max(1, len(strategy.treatment_values)), step=1, key="suggested_count_v4"))
            suggestions = suggest_numeric_designs(
                historical_df,
                strategy.historical_column,
                primary_column,
                float(strategy.control_value),
                float(strategy.min_value),
                float(strategy.max_value),
                float(strategy.increment),
                treatment_count,
                metric_std=float(primary.standard_deviation),
            )
            if suggestions.empty:
                st.warning("No valid suggested design is available for this range and number of treatments. Adjust the range or switch to Manual lines.")
            else:
                labels = [f"Option {idx + 1}: " + " and ".join(f"{float(value):,.0f}" for value in row) for idx, row in enumerate(suggestions["Suggested Treatments"])]
                selected_label = st.radio("Suggested Design", labels, key="suggested_design_v4")
                selected_index = labels.index(selected_label)
                strategy.treatment_values = [float(value) for value in suggestions.iloc[selected_index]["Suggested Treatments"]]
                strategy.treatment_names = [f"Treatment {idx + 1}" for idx in range(len(strategy.treatment_values))]
                strategy.number_of_arms = len(strategy.treatment_values) + 1
                shown_suggestions = suggestions.copy()
                shown_suggestions["Suggested Treatments"] = shown_suggestions["Suggested Treatments"].map(lambda values: ", ".join(maybe_money(value) for value in values))
                for column in ["Coverage", "Separation", "Detectability", "Historical Information", "Overall Score"]:
                    shown_suggestions[column] = shown_suggestions[column].map(lambda value: percent(value, 0))
                st.dataframe(shown_suggestions, use_container_width=True, hide_index=True)
                top_score = float(suggestions.iloc[0]["Overall Score"])
                tied_count = int((suggestions["Overall Score"].sub(top_score).abs() < 1e-9).sum())
                if tied_count > 1:
                    st.info(f"{tied_count} candidate designs are tied under the current historical ranking. Treat them as equally ranked and choose using business constraints.")
                st.caption("Historical ranking weights: customer coverage 32%, separation between lines 28%, variation in historical metric means 22%, and standardized difference from BAU 18%. Final sample planning is calculated separately in Design.")
        else:
            rows = [{"Role": "Control", "Arm Name": strategy.control_name, "Numeric Level": float(strategy.control_value)}]
            rows.extend({"Role": "Treatment", "Arm Name": strategy.treatment_names[idx] if idx < len(strategy.treatment_names) else f"Treatment {idx + 1}", "Numeric Level": float(value)} for idx, value in enumerate(strategy.treatment_values))
            edited = st.data_editor(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                disabled=["Role"],
                column_config={
                    "Role": st.column_config.TextColumn("Role"),
                    "Arm Name": st.column_config.TextColumn("Arm Name", required=True),
                    "Numeric Level": st.column_config.NumberColumn("Numeric Level", required=True, step=float(strategy.increment)),
                },
                key="ordered_arm_editor_v5",
            )
            clean = edited.dropna(subset=["Arm Name", "Numeric Level"]).copy()
            clean = clean[clean["Arm Name"].astype(str).str.strip() != ""]
            if len(clean) < 2:
                st.error("Keep one control arm and at least one treatment arm.")
            else:
                strategy.control_value = float(clean.iloc[0]["Numeric Level"])
                strategy.control_name = str(clean.iloc[0]["Arm Name"])
                strategy.treatment_names = clean.iloc[1:]["Arm Name"].astype(str).tolist()
                strategy.treatment_values = clean.iloc[1:]["Numeric Level"].astype(float).tolist()
                strategy.number_of_arms = len(clean)

        for error in validate_numeric_strategy(float(strategy.control_value), [float(value) for value in strategy.treatment_values], float(strategy.min_value), float(strategy.max_value), float(strategy.increment)):
            st.error(error)
        points = [float(strategy.control_value)] + [float(value) for value in strategy.treatment_values]
        point_labels = [f"Assigned line · {strategy.control_name}: {float(strategy.control_value):,.0f}"]
        point_labels.extend(
            f"Assigned line · {strategy.treatment_names[idx] if idx < len(strategy.treatment_names) else f'Treatment {idx + 1}'}: {float(value):,.0f}"
            for idx, value in enumerate(strategy.treatment_values)
        )
        roles = list(st.session_state.metrics_config)
        selected_role = st.selectbox("Metric Shown", roles, format_func=lambda role: f"{role}: {st.session_state.metrics_config[role].name}", key="historical_metric_role_v4")
        metric = st.session_state.metrics_config[selected_role]
        metric_column = metric.processed_column or metric.column
        st.plotly_chart(
            historical_association(
                historical_df,
                strategy.historical_column,
                metric_column,
                metric.metric_type,
                strategy_points=points,
                strategy_label=strategy.strategy_name,
                metric_label=metric.name,
                binning_method=strategy.curve_binning_method,
                bin_width=float(strategy.curve_bin_width),
                bin_count=int(strategy.curve_bin_count),
                strategy_point_labels=point_labels,
            ),
            use_container_width=True,
            key="treatment_history_chart_v4",
        )
        st.caption(f"Uses the {strategy.curve_binning_method.lower()} configured in Metrics. Dotted lines show the selected experiment assignments; the historical relationship is descriptive, not causal.")
        st.session_state.historical_arm_stats_by_role = {}
    else:
        st.session_state.historical_arm_stats_by_role = {}
        st.markdown("**Proposed Experiment Variants**")
        rows = [{"Role": "Control", "Arm Name": strategy.categorical_control}]
        rows.extend({"Role": "Treatment", "Arm Name": value} for value in strategy.categorical_treatments)
        edited = st.data_editor(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            disabled=["Role"],
            column_config={"Role": st.column_config.TextColumn("Role"), "Arm Name": st.column_config.TextColumn("Arm Name", required=True)},
            key="named_arm_editor_v4",
        )
        clean = edited.dropna(subset=["Arm Name"])
        clean = clean[clean["Arm Name"].astype(str).str.strip() != ""]
        if not clean.empty:
            strategy.categorical_control = str(clean.iloc[0]["Arm Name"])
            strategy.categorical_treatments = clean.iloc[1:]["Arm Name"].astype(str).tolist()
            strategy.number_of_arms = len(clean)
        for error in validate_categorical_strategy(strategy.categorical_control, strategy.categorical_treatments):
            st.error(error)
    control, treatments = configured_arms(strategy)
    st.caption(f"{len(treatments) + 1} arms · 1 control · {len(treatments)} treatments")


def metrics_step() -> None:
    st.subheader("Experiment Metrics")
    raw_df = st.session_state.raw_df
    processed_df = st.session_state.historical_df
    numeric = numeric_columns(raw_df)
    for role in ["Primary", "Secondary", "Guardrail"]:
        with st.expander(role, expanded=role == "Primary"):
            enabled = True if role == "Primary" else st.checkbox(f"Configure {role} Metric", value=role in st.session_state.metrics_config, key=f"enable_{role}")
            if not enabled:
                st.session_state.metrics_config.pop(role, None)
                continue
            cfg = st.session_state.metrics_config.get(role) or MetricConfig(role, numeric[0], role)
            c1, c2, c3 = st.columns([1.2, 1, 1])
            cfg.name = c1.text_input("Display Name", cfg.name, key=f"{role}_name")
            cfg.metric_type = c2.selectbox("Metric Type", ["Continuous", "Binary"], index=["Continuous", "Binary"].index(cfg.metric_type), key=f"{role}_type")
            cfg.direction = c3.selectbox("Direction", ["Higher is Better", "Lower is Better"], index=["Higher is Better", "Lower is Better"].index(cfg.direction), key=f"{role}_direction")
            cfg.assumption_source = st.radio("Assumption Source", ["Enter Manually", "Calculate from Historical Data"], index=["Enter Manually", "Calculate from Historical Data"].index(cfg.assumption_source), horizontal=True, key=f"{role}_assumption_source")
            if cfg.assumption_source == "Calculate from Historical Data":
                cfg.source_column = st.selectbox("Historical Metric Column", numeric, index=numeric.index(cfg.source_column) if cfg.source_column in numeric else 0, key=f"{role}_source_column")
                inferred = suggest_metric_type(raw_df, cfg.source_column)
                cfg.metric_type = st.selectbox("Historical Metric Type", ["Continuous", "Binary"], index=["Continuous", "Binary"].index("Binary" if inferred == "Binary" else cfg.metric_type), key=f"{role}_hist_type")
                update_assumption_from_history(cfg, raw_df if processed_df.empty else processed_df, cfg.source_column if cfg.source_column in (raw_df if processed_df.empty else processed_df).columns else cfg.column)
                st.caption("Calculated from historical data and stored as the same normalized metric assumptions used by manual mode.")
            if cfg.metric_type == "Continuous":
                c4, c5, c6 = st.columns(3)
                cfg.baseline_mean = c4.number_input("Baseline Mean", value=float(cfg.baseline_mean), step=10.0, format="%.4f", key=f"{role}_baseline_mean")
                cfg.variability_input = c5.selectbox("Variability Input", ["Standard Deviation", "Variance"], index=["Standard Deviation", "Variance"].index(cfg.variability_input), key=f"{role}_variability")
                if cfg.variability_input == "Variance":
                    cfg.variance = c6.number_input("Variance", min_value=0.0, value=float(cfg.variance), step=1000.0, format="%.4f", key=f"{role}_variance")
                    cfg.standard_deviation = normalize_metric(cfg).standard_deviation or 0
                    st.caption(f"Implied Standard Deviation: {number(cfg.standard_deviation, 2)}")
                else:
                    cfg.standard_deviation = c6.number_input("Standard Deviation", min_value=0.0, value=float(cfg.standard_deviation), step=10.0, format="%.4f", key=f"{role}_sd")
                    cfg.variance = cfg.standard_deviation**2
            else:
                cfg.baseline_rate = st.number_input("Baseline Rate", min_value=0.0001, max_value=0.9999, value=float(cfg.baseline_rate), step=0.001, format="%.4f", key=f"{role}_baseline_rate")
            if role == "Primary":
                c7, c8, c9, c10 = st.columns(4)
                effect_options = ["Relative %", "Absolute"] if cfg.metric_type == "Continuous" else ["Percentage Point", "Relative %"]
                if cfg.effect_type == "Absolute" and cfg.metric_type == "Binary":
                    cfg.effect_type = "Percentage Point"
                if cfg.effect_type == "Percentage Point" and cfg.metric_type == "Continuous":
                    cfg.effect_type = "Absolute"
                cfg.effect_type = c7.selectbox("Minimum Detectable Effect Type", effect_options, index=effect_options.index(cfg.effect_type), key=f"{role}_effect_type")
                cfg.effect_value = c8.number_input("Minimum Detectable Effect", min_value=0.0001, value=float(cfg.effect_value), step=0.01, format="%.4f", key=f"{role}_effect")
                st.session_state.design_config.alpha = c9.number_input("Alpha", min_value=0.0001, max_value=0.9999, value=float(st.session_state.design_config.alpha), step=0.005, format="%.4f", key=f"{role}_alpha")
                st.session_state.design_config.target_power = c10.number_input("Target Power", min_value=0.01, max_value=0.9999, value=float(st.session_state.design_config.target_power), step=0.01, format="%.4f", key=f"{role}_power")
            else:
                c7, c8 = st.columns(2)
                cfg.effect_type = c7.selectbox("Effect Type", ["Relative %", "Absolute"] if cfg.metric_type == "Continuous" else ["Percentage Point", "Relative %"], key=f"{role}_effect_type")
                cfg.expected_effect = c8.number_input("Expected Effect", value=float(cfg.expected_effect or 0.0), step=0.01, format="%.4f", key=f"{role}_expected") or None
            if role == "Guardrail":
                cfg.guardrail_threshold = st.number_input("Guardrail Threshold", min_value=0.0001, value=float(cfg.guardrail_threshold or cfg.effect_value), step=0.01 if cfg.metric_type == "Continuous" else 0.001, format="%.4f", key=f"{role}_threshold")
                cfg.effect_value = cfg.guardrail_threshold
            for error in validate_metric_assumptions(cfg, st.session_state.design_config.alpha if role == "Primary" else None, st.session_state.design_config.target_power if role == "Primary" else None):
                st.error(error)
            st.caption(f"Normalized assumptions: baseline {baseline_label(cfg)}, {variability_label(cfg)}")
            st.session_state.metrics_config[role] = cfg


def design_step() -> None:
    st.subheader("Experiment Design")
    strategy: StrategyConfig = st.session_state.strategy_config
    design: DesignConfig = st.session_state.design_config
    control, treatments = configured_arms(strategy)
    if strategy.strategy_type == "Numeric Strategy":
        arms = [{"Arm": strategy.control_name, "Treatment": control, "Role": "Control"}] + [{"Arm": strategy.treatment_names[idx] if idx < len(strategy.treatment_names) else f"T{idx + 1}", "Treatment": value, "Role": "Treatment"} for idx, value in enumerate(treatments)]
    else:
        arms = [{"Arm": str(control), "Treatment": str(control), "Role": "Control"}] + [{"Arm": str(value), "Treatment": str(value), "Role": "Treatment"} for value in treatments]
    left, main = st.columns([0.8, 1.35])
    primary = st.session_state.metrics_config["Primary"]
    allocation_rows: list[dict] = []
    with left:
        st.markdown("**Statistical Settings**")
        design.alpha = st.number_input("False-positive Rate", min_value=0.0001, max_value=0.9999, value=float(design.alpha), step=0.005, format="%.4f")
        design.target_power = st.number_input("Target Detection Chance", min_value=0.01, max_value=0.9999, value=float(design.target_power), step=0.01, format="%.4f")
        effect_options = ["Relative %", "Absolute"] if primary.metric_type == "Continuous" else ["Percentage Point", "Relative %"]
        if primary.effect_type not in effect_options:
            primary.effect_type = effect_options[0]
        previous_effect_type = primary.effect_type
        previous_effect_absolute = normalize_metric(primary).effect_absolute
        primary.effect_type = st.selectbox("Effect Scale", effect_options, index=effect_options.index(primary.effect_type))
        if primary.effect_type != previous_effect_type:
            primary.effect_value = previous_effect_absolute / max(abs(normalize_metric(primary).baseline), 1e-12) if primary.effect_type == "Relative %" else previous_effect_absolute
        if primary.effect_type in {"Relative %", "Percentage Point"}:
            shown_effect = st.number_input("Smallest Effect Worth Detecting (%)", min_value=0.001, max_value=99.999, value=min(float(primary.effect_value) * 100, 99.999), step=0.1, key=f"primary_effect_percent_{primary.effect_type}_v3")
            primary.effect_value = shown_effect / 100
        else:
            primary.effect_value = st.number_input("Smallest Effect Worth Detecting", min_value=0.0001, value=float(primary.effect_value), step=1.0, key="primary_effect_absolute_v3")
        primary_assumption = normalize_metric(primary)
        if strategy.strategy_type == "Numeric Strategy":
            grouping_options = ["Closest testing line", "Business-defined bin width", "Exact lines"]
            strategy.planning_grouping = st.selectbox(
                "Historical Mean and SD Grouping",
                grouping_options,
                index=grouping_options.index(strategy.planning_grouping) if strategy.planning_grouping in grouping_options else 0,
                help="Controls which historical customers provide the mean and variability used for sample-size planning.",
                key="planning_grouping_v4",
            )
            mapping_distance = 0.0
            if strategy.planning_grouping == "Business-defined bin width":
                strategy.planning_bin_width = st.number_input(
                    "Business Bin Width",
                    min_value=0.01,
                    value=float(strategy.planning_bin_width),
                    step=max(float(strategy.increment), 0.01),
                    help="Each test line uses historical customers within half this width. A width of 2,000 gives a +/-1,000 neighborhood.",
                    key="planning_bin_width_v4",
                )
                mapping_distance = float(strategy.planning_bin_width) / 2

            points = [float(control)] + [float(value) for value in treatments]
            historical_stats: dict[str, pd.DataFrame] = {}
            for role, metric in st.session_state.metrics_config.items():
                metric_column = metric.processed_column or metric.column
                historical_stats[role] = historical_arm_statistics(
                    st.session_state.historical_df,
                    strategy.historical_column,
                    metric_column,
                    points,
                    grouping=strategy.planning_grouping,
                    max_distance=mapping_distance,
                    metric_type=metric.metric_type,
                )
            st.session_state.historical_arm_stats_by_role = historical_stats
        with st.expander("Historical Assumptions", expanded=False):
            if strategy.strategy_type == "Numeric Strategy":
                primary_stats = st.session_state.historical_arm_stats_by_role.get("Primary", pd.DataFrame())
                if primary_stats.empty:
                    st.warning("No historical acquisition-line assumptions could be calculated for these test points.")
                else:
                    shown = primary_stats[["Strategy Point", "Observed Line Range", "Historical N", "Historical Mean", "Historical SD", "Mapping", "Support"]].copy()
                    shown["Strategy Point"] = shown["Strategy Point"].map(maybe_money)
                    shown["Historical Mean"] = shown["Historical Mean"].round(2)
                    shown["Historical SD"] = shown["Historical SD"].round(2)
                    st.dataframe(shown, use_container_width=True, hide_index=True)
                    if strategy.planning_grouping == "Closest testing line":
                        st.caption("Every eligible customer is assigned to the closest proposed test line; ties go to the lower line.")
                    elif strategy.planning_grouping == "Business-defined bin width":
                        st.caption(f"Each point uses customers within +/-{strategy.planning_bin_width / 2:,.0f}; customers outside every neighborhood are excluded.")
                    else:
                        st.caption("Only customers whose historical acquisition line exactly equals a proposed test line are used.")
            else:
                st.write(f"Baseline: {baseline_label(primary)}")
                st.write(f"Historical SD: {number(primary_assumption.standard_deviation or 0, 2) if primary.metric_type == 'Continuous' else 'Binomial'}")
                st.caption("Calculated from the primary metric's eligible historical population.")
        with st.expander("Advanced Settings", expanded=False):
            design.sample_size_basis = st.selectbox("Sample Size Basis", ["Primary Metric Only", "Power All Configured Metrics"], index=["Primary Metric Only", "Power All Configured Metrics"].index(design.sample_size_basis) if design.sample_size_basis in ["Primary Metric Only", "Power All Configured Metrics"] else 0)
            design.multiplicity_method = st.selectbox("Multiple-Comparison Control", ["Holm", "None"], index=["Holm", "None"].index(design.multiplicity_method))
            design.attrition_rate = st.number_input("Expected Attrition / Missing Outcomes", min_value=0.0, max_value=0.8, value=float(design.attrition_rate), step=0.01, format="%.2f")
    with main:
        st.markdown("**Proposed Strategy**")
        st.dataframe(pd.DataFrame(arms), use_container_width=True, hide_index=True)
    try:
        if strategy.strategy_type == "Numeric Strategy":
            total_analyzable, design.binding_metric, required_rows, allocation_rows = design_neyman_allocation(
                st.session_state.metrics_config,
                design,
                st.session_state.historical_arm_stats_by_role,
                float(control),
                [float(value) for value in treatments],
            )
        else:
            analyzable_n_per_arm, design.binding_metric, required_rows = design_sample_size(st.session_state.metrics_config, design, comparisons=max(len(treatments), 1))
    except ValueError as exc:
        design.required_n_per_arm = 0
        design.total_sample_size = 0
        st.error(str(exc))
        st.info("Return to Treatments and choose historically supported points before calculating experiment size.")
        return
    if strategy.strategy_type == "Numeric Strategy":
        arm_names = {float(control): strategy.control_name}
        arm_names.update({float(value): strategy.treatment_names[idx] if idx < len(strategy.treatment_names) else f"Treatment {idx + 1}" for idx, value in enumerate(treatments)})
        for row in allocation_rows:
            row["Arm"] = arm_names.get(float(row["Strategy Point"]), str(row["Strategy Point"]))
            row["Enroll N"] = int(math.ceil(int(row["Analyzable N"]) / max(1 - design.attrition_rate, 1e-9)))
        design.arm_sample_sizes = {str(row["Strategy Point"]): int(row["Enroll N"]) for row in allocation_rows}
        design.required_n_per_arm = max((int(row["Enroll N"]) for row in allocation_rows), default=0)
        design.total_sample_size = sum(int(row["Enroll N"]) for row in allocation_rows)
    else:
        total_analyzable = analyzable_n_per_arm * (len(treatments) + 1)
        design.required_n_per_arm = int(math.ceil(analyzable_n_per_arm / max(1 - design.attrition_rate, 1e-9)))
        design.total_sample_size = design.required_n_per_arm * (len(treatments) + 1)
        design.arm_sample_sizes = {str(arm["Treatment"]): design.required_n_per_arm for arm in arms}
    assumption_rows = []
    for metric in st.session_state.metrics_config.values():
        effect_text = f"Smallest effect = {effect_value_label(metric)}" if metric.role == "Primary" else f"Effect / Threshold = {effect_value_label(metric)}"
        if metric.role == "Guardrail":
            effect_text = f"Max harm = {percent(metric.guardrail_threshold or metric.effect_value, 1)}" if metric.effect_type in {"Relative %", "Percentage Point"} else f"Max harm = {number(metric.guardrail_threshold or metric.effect_value, 4)}"
        numeric_strategy = strategy.strategy_type == "Numeric Strategy"
        variability = "Line-specific historical SD" if numeric_strategy else variability_label(metric)
        baseline = "Control acquisition-line mean" if numeric_strategy else baseline_label(metric)
        assumption_rows.append({"Metric": metric.name, "Role": metric.role, "Baseline": baseline, "Variability": variability, "Effect / Threshold": effect_text})
    st.markdown("**Recommended Experiment Size**")
    c1, c2 = st.columns(2)
    c1.metric("Total Analyzable", f"{total_analyzable:,.0f}")
    c2.metric("Total Enrollment", f"{design.total_sample_size:,.0f}")
    c3, c4 = st.columns(2)
    c3.metric("Number of Arms", f"{len(treatments) + 1}")
    c4.metric("Allocation", "Neyman" if strategy.strategy_type == "Numeric Strategy" else "Equal")
    if strategy.strategy_type == "Numeric Strategy":
        shown_allocation = pd.DataFrame(allocation_rows)[
            ["Arm", "Role", "Strategy Point", "Observed Line Range", "Historical N", "Historical Mean", "Historical SD", "Allocation Share", "Analyzable N", "Enroll N", "Support"]
            if "Observed Line Range" in allocation_rows[0]
            else ["Arm", "Role", "Strategy Point", "Historical N", "Historical Mean", "Historical SD", "Allocation Share", "Analyzable N", "Enroll N", "Mapping", "Support"]
        ].copy()
        shown_allocation["Strategy Point"] = shown_allocation["Strategy Point"].map(maybe_money)
        shown_allocation["Historical Mean"] = shown_allocation["Historical Mean"].round(2)
        shown_allocation["Historical SD"] = shown_allocation["Historical SD"].round(2)
        shown_allocation["Allocation Share"] = shown_allocation["Allocation Share"].map(lambda value: percent(value, 1))
        st.markdown("**Estimated Sample by Arm**")
        st.dataframe(shown_allocation, use_container_width=True, hide_index=True)
        st.caption("Generalized Neyman allocation gives more observations to higher-variance arms and accounts for the control being reused in every treatment comparison.")
    st.write(f"Primary Metric: {primary.name} through MOB {primary.mob_horizon}")
    st.write(f"Smallest effect: {effect_value_label(primary)} | Detection chance: {percent(design.target_power, 0)} | False-positive rate: {percent(design.alpha, 1)}")
    planning_alpha = adjusted_alpha(design.alpha, max(len(treatments), 1), design.multiplicity_method)
    with st.expander("Metric Assumptions and Sample Drivers", expanded=False):
        if strategy.strategy_type == "Numeric Strategy" or design.sample_size_basis == "Power All Configured Metrics":
            st.markdown("**Planned Power by Comparison**" if strategy.strategy_type == "Numeric Strategy" else "**Required N by Metric**")
            shown_requirements = pd.DataFrame(required_rows).copy()
            if "Planned Power" in shown_requirements:
                shown_requirements["Planned Power"] = shown_requirements["Planned Power"].map(lambda value: percent(value, 1))
            st.dataframe(shown_requirements, use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame(assumption_rows), use_container_width=True, hide_index=True)
        if len(treatments) > 1 and design.multiplicity_method != "None":
            st.caption(f"Planning uses a conservative per-comparison alpha of {planning_alpha:.4f}; final p-values use Holm adjustment.")
    traffic_left, traffic_right = st.columns(2)
    design.eligible_customers = traffic_left.number_input("Eligible Units Entering per Period", min_value=1, value=int(design.eligible_customers), step=25)
    frequency_options = ["Daily", "Weekly", "Monthly"]
    frequency_index = frequency_options.index(design.traffic_frequency) if design.traffic_frequency in frequency_options else (0 if design.traffic_frequency in ["Day", "Per Day"] else 1)
    design.traffic_frequency = traffic_right.radio("Traffic Frequency", frequency_options, index=frequency_index, horizontal=True)
    unit = {"Daily": "day", "Weekly": "week", "Monthly": "month"}[design.traffic_frequency]
    estimated = duration(design.total_sample_size, design.eligible_customers, design.traffic_frequency)
    rounded = int(pd.Series([estimated]).apply(lambda x: int(x) if x == int(x) else int(x) + 1).iloc[0])
    launch_default = pd.Timestamp(design.planned_launch_date).date() if design.planned_launch_date else datetime.now().date()
    design.planned_launch_date = st.date_input("Planned Launch Date", value=launch_default).isoformat()
    if design.traffic_frequency == "Daily":
        enrollment_end = pd.Timestamp(design.planned_launch_date) + pd.Timedelta(days=rounded)
    elif design.traffic_frequency == "Weekly":
        enrollment_end = pd.Timestamp(design.planned_launch_date) + pd.Timedelta(weeks=rounded)
    else:
        enrollment_end = pd.Timestamp(design.planned_launch_date) + pd.DateOffset(months=rounded)
    if st.session_state.data_mapping.data_structure == "Longitudinal (unit x period)":
        followup_value = max((metric.mob_horizon for metric in st.session_state.metrics_config.values()), default=0)
        followup_label = f"{followup_value} months"
        primary_followup_label = f"{primary.mob_horizon} months"
        primary_readout_date = enrollment_end + pd.DateOffset(months=primary.mob_horizon)
        readout_date = enrollment_end + pd.DateOffset(months=followup_value)
    else:
        c1, c2 = st.columns(2)
        design.outcome_delay_value = int(c1.number_input("Outcome Follow-up", min_value=0, value=int(design.outcome_delay_value), step=1))
        design.outcome_delay_unit = c2.selectbox("Follow-up Unit", ["Days", "Weeks", "Months"], index=["Days", "Weeks", "Months"].index(design.outcome_delay_unit))
        followup_label = f"{design.outcome_delay_value} {design.outcome_delay_unit.lower()}"
        primary_followup_label = followup_label
        if design.outcome_delay_unit == "Months":
            readout_date = enrollment_end + pd.DateOffset(months=design.outcome_delay_value)
        elif design.outcome_delay_unit == "Weeks":
            readout_date = enrollment_end + pd.Timedelta(weeks=design.outcome_delay_value)
        else:
            readout_date = enrollment_end + pd.Timedelta(days=design.outcome_delay_value)
        primary_readout_date = readout_date
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Enrollment Duration", f"{rounded:,.0f} {unit}s")
    c2.metric("Primary Follow-up", primary_followup_label)
    c3.metric("Primary Readout", primary_readout_date.date().isoformat())
    c4.metric("Full Decision Readout", readout_date.date().isoformat())
    st.caption(f"Enrollment is expected to finish around {enrollment_end.date().isoformat()}. Full decision follow-up is {followup_label} because it includes every configured metric.")
    st.caption("Traffic is split using the planned per-arm allocation shown above." if strategy.strategy_type == "Numeric Strategy" else "Traffic is split equally across all arms.")
    if design.multiplicity_method == "None" and len(treatments) > 1:
        st.warning("Multiple-comparison control is disabled. The chance of at least one false positive increases with each treatment arm.")
    if not st.session_state.historical_df.empty and design.total_sample_size > len(st.session_state.historical_df):
        st.warning("Required sample exceeds the currently eligible historical population. Confirm future traffic availability before launching.")
    expected_rows = []
    if strategy.strategy_type == "Numeric Strategy":
        with st.expander("Detection Sensitivity", expanded=False):
            shown_power = pd.DataFrame(required_rows)[["Metric", "Role", "Comparison", "Control N", "Treatment N", "Control SD", "Treatment SD", "Smallest Effect", "Planned Power", "Support"]].copy()
            shown_power["Control SD"] = shown_power["Control SD"].round(2)
            shown_power["Treatment SD"] = shown_power["Treatment SD"].round(2)
            shown_power["Smallest Effect"] = shown_power["Smallest Effect"].round(4)
            shown_power["Planned Power"] = shown_power["Planned Power"].map(lambda value: percent(value, 1))
            st.dataframe(shown_power, use_container_width=True, hide_index=True)
            st.caption("Planned power is recalculated with each comparison's actual control and treatment allocation and its line-specific historical variability.")
        return
    else:
        rows = [detectable_for_metric(metric, planning_alpha, design.target_power, analyzable_n_per_arm) for metric in st.session_state.metrics_config.values()]
        for metric in st.session_state.metrics_config.values():
            power_value = expected_power(metric, planning_alpha, analyzable_n_per_arm)
            if power_value is not None:
                expected_rows.append({"Metric": metric.name, "Role": metric.role, "Expected Effect": metric.expected_effect, "Power": power_value})
            if metric.role == "Guardrail" and metric.guardrail_threshold is not None:
                original_expected = metric.expected_effect
                metric.expected_effect = metric.guardrail_threshold
                guardrail_power = expected_power(metric, planning_alpha, analyzable_n_per_arm)
                metric.expected_effect = original_expected
                if guardrail_power is not None and guardrail_power < design.target_power:
                    st.warning(f"At the proposed experiment size, the guardrail has only {percent(guardrail_power, 0)} power to detect the configured threshold.")
    primary = st.session_state.metrics_config["Primary"]
    primary_assumption = normalize_metric(primary)
    curve_rows = []
    max_n = max(analyzable_n_per_arm * 2, 100)
    binding_primary = None
    primary_baseline = primary_assumption.baseline
    for n in pd.Series([max(20, int(max_n * pct / 12)) for pct in range(1, 13)]).drop_duplicates():
        if primary.metric_type == "Continuous" and binding_primary is not None:
            effect = detectable_effect_continuous_unequal(planning_alpha, design.target_power, float(binding_primary["Control SD"]), float(binding_primary["Treatment SD"]), int(n))
        elif primary.metric_type == "Continuous":
            effect = detectable_effect_continuous(planning_alpha, design.target_power, primary_assumption.standard_deviation or 0, int(n))
        else:
            effect = detectable_effect_binary(planning_alpha, design.target_power, primary_baseline, int(n), primary.direction)
        curve_rows.append({"Sample Size per Arm": int(n), "Minimum Detectable Effect": effect})
    with st.expander("Detection Sensitivity", expanded=False):
        st.markdown("**Detectable Effect by Metric**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if expected_rows:
            st.markdown("**Expected Detection Chance**")
            st.dataframe(pd.DataFrame(expected_rows), use_container_width=True, hide_index=True)
        if binding_primary is not None:
            st.caption(f"Sensitivity curve uses the most demanding primary comparison: {binding_primary['Comparison']}.")
        st.plotly_chart(detectable_effect_curve(pd.DataFrame(curve_rows), analyzable_n_per_arm), use_container_width=True)


def analysis_step() -> None:
    st.subheader("Analysis")
    strategy: StrategyConfig = st.session_state.strategy_config
    design: DesignConfig = st.session_state.design_config
    source = st.radio("Experiment Result Data Source", ["Upload File", "Internal Data", "Synthetic Demo"], index=2, horizontal=True)
    if source == "Upload File":
        uploaded = st.file_uploader("Upload Experiment Results", type=["csv", "parquet"], key="analysis_upload")
        if uploaded is not None:
            signature = hash(uploaded.getvalue())
            if st.session_state.get("customer_results_upload_signature") != signature:
                uploaded.seek(0)
                st.session_state.results_df = pd.read_parquet(uploaded) if uploaded.name.endswith(".parquet") else load_uploaded_csv(uploaded)
                st.session_state.customer_results_upload_signature = signature
                st.session_state.analysis_results_by_role = {}
    elif source == "Synthetic Demo":
        st.session_state.results_df = load_results_demo()
        if strategy.strategy_type == "Categorical Strategy":
            demo = st.session_state.results_df.copy()
            source_assignment = "assigned_credit_line"
            source_arms = sorted(demo[source_assignment].dropna().unique().tolist())
            source_control = 5000 if 5000 in source_arms else source_arms[0]
            source_treatments = [arm for arm in source_arms if arm != source_control]
            mapping = {source_control: strategy.categorical_control}
            mapping.update({arm: strategy.categorical_treatments[idx] for idx, arm in enumerate(source_treatments) if idx < len(strategy.categorical_treatments)})
            demo[strategy.assignment_column] = demo[source_assignment].map(mapping)
            st.session_state.results_df = demo.dropna(subset=[strategy.assignment_column])
    else:
        st.info("Internal result-data access is modular and will use the future Databricks connector.")
        st.session_state.results_df = load_results_demo()
    df = st.session_state.results_df
    if df is None:
        st.info("Load experiment-result data to run Analysis.")
        return
    cols = list(df.columns)
    analysis_config: AnalysisConfig = st.session_state.analysis_config
    inferred_unit = analysis_config.unit_id_column if analysis_config.unit_id_column in cols else next((column for column in cols if column.lower() in {"customer_id", "account_id", "user_id", "member_id"}), "")
    default_assignment = strategy.assignment_column if strategy.assignment_column in cols else ("assigned_credit_line" if "assigned_credit_line" in cols else cols[0])
    map_left, map_right = st.columns(2)
    analysis_config.unit_id_column = map_left.selectbox("Experimental Unit ID", cols, index=cols.index(inferred_unit) if inferred_unit in cols else 0)
    assignment_col = map_right.selectbox("Assignment Column", cols, index=cols.index(default_assignment))
    control, treatments = configured_arms(strategy)
    available_arms = sorted(df[assignment_col].dropna().unique().tolist())
    arm_left, arm_right = st.columns(2)
    control = arm_left.selectbox("Control Arm", available_arms, index=available_arms.index(control) if control in available_arms else 0)
    treatments = arm_right.multiselect("Treatment Arms", [arm for arm in available_arms if arm != control], default=[arm for arm in available_arms if arm != control])
    result_columns = {}
    invalid_mapping = False
    used_result_columns = set()
    for role, metric in st.session_state.metrics_config.items():
        numeric = [col for col in numeric_columns(df) if col != assignment_col]
        preferred = None
        role_candidates = {
            "Primary": ["outcome", "conversion", "revolving_balance"],
            "Secondary": ["revenue", "risk_adjusted_revenue", "engagement"],
            "Guardrail": ["loss", "default_flag", "complaint_rate", "cost"],
        }
        for candidate in [metric.column, metric.processed_column, metric.source_column, "risk_adjusted_revenue" if metric.source_column == "revenue" else ""] + role_candidates.get(role, []):
            if candidate in numeric and candidate not in used_result_columns:
                preferred = candidate
                break
        if not numeric:
            st.error("No numeric result columns are available after excluding the assignment column.")
            invalid_mapping = True
            continue
        if preferred is None:
            preferred = next((column for column in numeric if column not in used_result_columns), numeric[0])
        result_columns[role] = st.selectbox(f"{role} Result Column", numeric, index=numeric.index(preferred), key=f"result_{role}_v2")
        used_result_columns.add(result_columns[role])
        if result_columns[role] == assignment_col:
            st.error(f"{role} Result Column cannot be the Experimental Assignment Column.")
            invalid_mapping = True
    outcomes = {role: (result_columns[role], metric.metric_type) for role, metric in st.session_state.metrics_config.items() if role in result_columns}
    validation_errors, validation_warnings = validate_analysis_data(df, assignment_col, outcomes, control, treatments, analysis_config.unit_id_column)
    for warning in validation_warnings:
        st.warning(warning)
    for error in validation_errors:
        st.error(error)
    integrity = analysis_integrity_summary(df, assignment_col, outcomes, [control] + treatments, analysis_config.unit_id_column)
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Analysis Units", f"{integrity['unique_units']:,}")
    i2.metric("Duplicate Units", f"{integrity['duplicate_units']:,}")
    i3.metric("Assignment Balance", integrity["srm_status"])
    i4.metric("Missing Primary", f"{integrity['missing_by_metric'].get('Primary', 0):,}")
    with st.expander("Analysis Data Audit", expanded=False):
        audit_rows = [{"Arm": arm, "Units": integrity["arm_counts"].get(arm, 0)} for arm in [control] + treatments]
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)
        st.write(f"Sample-ratio-mismatch p-value: {p_value(integrity['srm_pvalue']) if pd.notna(integrity['srm_pvalue']) else 'Not available'}")
        st.write("Missing outcomes by metric: " + ", ".join(f"{role}: {count:,}" for role, count in integrity["missing_by_metric"].items()))
    signature_columns = [assignment_col] + [column for column, _ in outcomes.values()]
    if analysis_config.unit_id_column:
        signature_columns.append(analysis_config.unit_id_column)
    data_hash = int(pd.util.hash_pandas_object(df[list(dict.fromkeys(signature_columns))], index=True).sum()) if signature_columns else 0
    metric_decisions = tuple((role, metric.direction, metric.effect_type, metric.guardrail_threshold) for role, metric in st.session_state.metrics_config.items())
    current_fingerprint = repr((data_hash, assignment_col, control, tuple(treatments), tuple(sorted(outcomes.items())), metric_decisions, design.alpha, design.multiplicity_method))
    st.session_state.customer_analysis_is_current = st.session_state.customer_analysis_fingerprint == current_fingerprint
    if st.button("Run Configured Analysis", type="primary", disabled=bool(validation_errors) or invalid_mapping):
        if invalid_mapping or validation_errors:
            st.error("Fix result-column mappings before running Analysis.")
            return
        by_role, responses = {}, {}
        for role, metric in st.session_state.metrics_config.items():
            col = result_columns[role]
            if metric.metric_type == "Binary":
                result = binary_results(df, assignment_col, col, control, treatments, design.alpha, design.multiplicity_method)
            else:
                result = continuous_results(df, assignment_col, col, control, treatments, design.alpha, design.multiplicity_method)
            by_role[role] = result
            responses[role] = response_summary(df[df[assignment_col].isin([control] + treatments)], assignment_col, col, design.alpha, metric.metric_type)
        st.session_state.analysis_results_by_role = by_role
        st.session_state.response_by_role = responses
        st.session_state.customer_analysis_fingerprint = current_fingerprint
        st.session_state.customer_analysis_is_current = True
    if st.session_state.analysis_results_by_role and not st.session_state.customer_analysis_is_current:
        st.warning("Results are out of date because the dataset or analysis configuration changed. Run the configured analysis again.")
        return
    if st.session_state.analysis_results_by_role:
        if design.multiplicity_method == "Holm" and len(treatments) > 1:
            st.info("Treatment-versus-control p-values are adjusted with the Holm method.")
        view_role = st.selectbox("Result Metric", list(st.session_state.analysis_results_by_role.keys()), format_func=lambda role: f"{role}: {st.session_state.metrics_config[role].name}")
        result = st.session_state.analysis_results_by_role[view_role]
        metric = st.session_state.metrics_config[view_role]
        table = format_effect_table(result, metric.metric_type)
        if design.multiplicity_method == "Holm" and len(treatments) > 1:
            table = table.rename(columns={"95% CI": "95% Familywise CI"})
        if view_role == "Guardrail":
            threshold = guardrail_absolute_threshold(metric)
            table["Guardrail Status"] = result.apply(lambda row: "Control" if row["Is Control"] else guardrail_status(row, threshold, metric.direction), axis=1)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.plotly_chart(forest_plot(result, strategy.strategy_name), use_container_width=True)
        if strategy.strategy_type == "Numeric Strategy":
            st.plotly_chart(response_plot(st.session_state.response_by_role[view_role], strategy.strategy_name, metric.name), use_container_width=True)


def decision_step() -> None:
    st.subheader("Decision")
    results = st.session_state.analysis_results_by_role
    if "Primary" not in results or not st.session_state.customer_analysis_is_current:
        st.info("Run Analysis before generating the Decision summary.")
        return
    guardrail_metric = st.session_state.metrics_config.get("Guardrail")
    primary_metric = st.session_state.metrics_config["Primary"]
    meaningful_effect = normalize_metric(primary_metric).effect_absolute
    guardrail_threshold = guardrail_absolute_threshold(guardrail_metric)
    scorecard = arm_decision_scorecard(
        results["Primary"],
        primary_metric.direction,
        meaningful_effect,
        results.get("Guardrail"),
        guardrail_threshold,
        guardrail_metric.direction if guardrail_metric else "Lower is Better",
    )
    display = scorecard.copy()
    display["Primary Effect"] = display["Primary Effect"].apply(lambda value: number(value, 2))
    display["Relative Lift"] = display["Relative Lift"].apply(lambda value: percent(value, 1) if pd.notna(value) else "")
    display["Adjusted p-value"] = display["Adjusted p-value"].apply(lambda value: p_value(value) if pd.notna(value) else "")
    display["Meets Planned Effect"] = display["Meets Planned Effect"].map({True: "Yes", False: "No"})
    st.markdown("**Treatment Decision Scorecard**")
    st.dataframe(display, use_container_width=True, hide_index=True)
    recommendation = experiment_recommendation(results["Primary"], results.get("Guardrail"), guardrail_threshold, guardrail_metric.direction if guardrail_metric else "Lower is Better", primary_metric.direction)
    st.markdown("**Recommended Action**")
    status = recommendation_status(recommendation)
    if status == "error":
        st.error(recommendation)
    elif status == "warning":
        st.warning(recommendation)
    else:
        st.success(recommendation)
    st.caption("The recommendation applies only to the tested arms, eligible population, observation windows, and configured metrics.")


def ts_data_step() -> None:
    st.subheader("Data")
    cfg: TimeSeriesConfig = st.session_state.ts_config
    source = st.radio("Data Source", ["Upload File", "Internal Data", "Synthetic Demo"], index=["Upload File", "Internal Data", "Synthetic Demo"].index(cfg.data_source), horizontal=True)
    cfg.data_source = source
    if source == "Upload File":
        uploaded = st.file_uploader("Upload Time-Series CSV or Parquet", type=["csv", "parquet"], key="ts_upload")
        if uploaded is not None:
            try:
                st.session_state.ts_raw_df = normalize_uploaded_dataset(pd.read_parquet(uploaded) if uploaded.name.endswith(".parquet") else load_uploaded_csv(uploaded))
                cfg.dataset_name = uploaded.name
            except Exception as exc:
                st.error("Data validation issue: We could not read the uploaded file. Confirm it is a valid CSV or Parquet file.")
                print(f"Time-series upload failed: {exc}")
    elif source == "Synthetic Demo":
        st.session_state.ts_raw_df = load_timeseries_planning_demo() if st.session_state.get("ts_intent") == "plan" else load_timeseries_demo()
        cfg.dataset_name = "Synthetic Historical Time Series" if st.session_state.get("ts_intent") == "plan" else "Synthetic Campaign Time Series"
    else:
        st.info("Internal time-series data access is a placeholder until the Databricks connector is added.")
        st.session_state.ts_raw_df = load_timeseries_demo()
        cfg.dataset_name = "Synthetic Campaign Time Series"

    df = st.session_state.ts_raw_df
    if df.empty:
        st.info("Upload a historical time-series dataset to evaluate a campaign without a randomized control group.")
        return
    inferred = infer_timeseries_columns(df)
    cfg.date_column = cfg.date_column if cfg.date_column in df.columns else inferred["date"]
    numeric = numeric_columns(df)
    cfg.outcome_column = cfg.outcome_column if cfg.outcome_column in numeric else inferred["outcome"]
    cfg.segment_column = cfg.segment_column if cfg.segment_column in ["None"] + list(df.columns) else inferred["segment"]
    cfg.campaign_flag_column = cfg.campaign_flag_column if cfg.campaign_flag_column in ["None"] + list(df.columns) else inferred["campaign_flag"]
    cfg.exposure_column = cfg.exposure_column if cfg.exposure_column in ["None"] + numeric else "None"

    st.markdown("**Data Mapping**")
    c1, c2, c3, c4, c5 = st.columns(5)
    cfg.date_column = c1.selectbox("Date Column", list(df.columns), index=list(df.columns).index(cfg.date_column) if cfg.date_column in df.columns else 0)
    cfg.outcome_column = c2.selectbox("Outcome Column", numeric, index=numeric.index(cfg.outcome_column) if cfg.outcome_column in numeric else 0)
    duplicate_options = ["Fail validation", "Average", "Sum"]
    cfg.duplicate_policy = c3.selectbox("Duplicate Timestamps", duplicate_options, index=duplicate_options.index(cfg.duplicate_policy) if cfg.duplicate_policy in duplicate_options else 0)
    cfg.missing_policy = c4.selectbox("Missing Periods", ["Fail validation", "Interpolate", "Fill missing counts with zero", "Keep"], index=["Fail validation", "Interpolate", "Fill missing counts with zero", "Keep"].index(cfg.missing_policy))
    cfg.exposure_column = c5.selectbox("Exposure / Customer Count (Optional)", ["None"] + [column for column in numeric if column != cfg.outcome_column], index=(["None"] + [column for column in numeric if column != cfg.outcome_column]).index(cfg.exposure_column) if cfg.exposure_column in (["None"] + [column for column in numeric if column != cfg.outcome_column]) else 0, help="Optional operational volume used only to project expected exposure. It is not the statistical time-series sample size.")
    try:
        # Detect cadence from this raw dataset so a previous upload's frequency
        # cannot leak into the current analysis.
        prepared = prepare_timeseries_data(df, cfg.date_column, cfg.outcome_column, cfg.duplicate_policy, cfg.missing_policy, "Auto")
        st.session_state.ts_prepared_df = prepared
        detected, missing = detect_frequency(prepared)
        missing = int(prepared.attrs.get("missing_periods", missing))
        cfg.frequency = detected
        duplicates = duplicate_timestamp_count(prepared)
        st.success(f"Data loaded successfully | {len(prepared):,.0f} observations | {prepared['_date'].min().date()} – {prepared['_date'].max().date()} | Detected frequency: {detected} | {missing} missing periods")
        if duplicates:
            st.warning(f"{duplicates:,.0f} duplicate timestamps were found.")
        if missing:
            st.warning(f"{missing:,.0f} {detected.lower()} observations are missing from the time series.")
        st.markdown("**Data Preview**")
        preview_mode = st.radio("Preview", ["All columns", "Mapped columns only"], horizontal=True, key="ts_preview_mode", label_visibility="collapsed")
        internal_columns = {"_date", "_outcome", "_normalized_date", "_internal_metric"}
        original_columns = [column for column in df.columns if column not in internal_columns and not str(column).startswith("_")]
        mapped_columns = [column for column in [cfg.date_column, cfg.outcome_column] + list(cfg.selected_covariates) if column in original_columns]
        preview_columns = original_columns if preview_mode == "All columns" else list(dict.fromkeys(mapped_columns))
        roles = {cfg.date_column: "Date", cfg.outcome_column: "Outcome", **{column: "Predictor" for column in cfg.selected_covariates}}
        display_headers = {column: f"{humanize_column_name(column)} [{roles[column]}]" if column in roles else humanize_column_name(column) for column in preview_columns}
        preview = df.loc[:, preview_columns].head(20).rename(columns=display_headers)
        st.caption(f"Showing {min(20, len(df)):,} of {len(df):,} rows · {len(preview_columns):,} columns")
        st.dataframe(preview, use_container_width=True, height=460, hide_index=True)
        if preview_mode == "All columns":
            st.caption("Mapped fields are marked in the headers. Original column names are unchanged in the analysis data.")
    except Exception as exc:
        st.error("Data validation issue: We could not interpret the selected Date and Outcome columns.")
        print(f"Time-series preparation failed: {exc}")


def ts_setup_step() -> None:
    st.subheader("Setup")
    cfg: TimeSeriesConfig = st.session_state.ts_config
    data = st.session_state.ts_prepared_df
    if data.empty:
        st.info("Upload and map time-series data before defining the campaign.")
        return
    min_date = data["_date"].min().date()
    max_date = data["_date"].max().date()
    default_launch = bounded_date(cfg.intervention_date, min_date, max_date, data["_date"].quantile(0.75).date())
    c1, c2, c3 = st.columns(3)
    launch = c1.date_input("Campaign Launch", value=default_launch, min_value=min_date, max_value=max_date)
    cfg.intervention_date = launch.isoformat()
    end_enabled = c2.checkbox("Campaign End Date", value=bool(cfg.campaign_end_date))
    if end_enabled:
        end_value = pd.Timestamp(cfg.campaign_end_date).date() if cfg.campaign_end_date else max_date
        cfg.campaign_end_date = c2.date_input("End Date", value=end_value, min_value=launch, max_value=max_date).isoformat()
    else:
        cfg.campaign_end_date = ""
    detected, missing = detect_frequency(data)
    cfg.frequency = detected
    interval_text = {"Daily": "1-day", "Weekly": "7-day", "Monthly": "monthly"}.get(detected, "date")
    st.markdown(f"**Data Frequency**  \n{detected}  \n<small>Detected from {interval_text} observation intervals.</small>", unsafe_allow_html=True)
    if st.button("Change Frequency", key="ts_change_frequency"):
        st.session_state.ts_frequency_override = True
    if st.session_state.get("ts_frequency_override", False):
        freq_options = ["Daily", "Weekly", "Monthly"]
        cfg.frequency = st.selectbox("Manual Frequency", freq_options, index=freq_options.index(cfg.frequency) if cfg.frequency in freq_options else 0)
    pre, post = data[data["_date"] < pd.Timestamp(cfg.intervention_date)], data[data["_date"] >= pd.Timestamp(cfg.intervention_date)]
    st.write(f"Pre period: {pre['_date'].min().date() if not pre.empty else 'None'} – {pre['_date'].max().date() if not pre.empty else 'None'}")
    first_treated = post["_date"].min() if not post.empty else None
    st.write(f"Intervention Date: {pd.Timestamp(cfg.intervention_date).strftime('%b %-d, %Y')}")
    st.write(f"First Treated Observation: {first_treated.strftime('%b %-d, %Y') if first_treated is not None else 'None'}")
    st.write(f"Post period: {post['_date'].min().date() if not post.empty else 'None'} – {post['_date'].max().date() if not post.empty else 'None'}")
    st.plotly_chart(time_series_line(data, cfg.intervention_date, humanize_column_name(cfg.outcome_column)), use_container_width=True)


def ts_plan_setup_step() -> None:
    st.subheader("Campaign Setup")
    cfg: TimeSeriesConfig = st.session_state.ts_config
    data = st.session_state.ts_prepared_df
    if data.empty:
        st.info("Load historical time-series data before defining a planned campaign.")
        return
    min_date, historical_end = data["_date"].min().date(), data["_date"].max().date()
    offsets = {"Daily": pd.Timedelta(days=1), "Weekly": pd.Timedelta(days=7), "Monthly": pd.DateOffset(months=1)}
    default_launch = bounded_date(cfg.planned_launch_date, min_date, historical_end + pd.Timedelta(days=730), historical_end + offsets.get(cfg.frequency, pd.Timedelta(days=7)))
    launch = st.date_input("Planned Campaign Launch", value=default_launch, min_value=min_date, max_value=historical_end + pd.Timedelta(days=730))
    cfg.planned_launch_date = launch.isoformat()
    detected, _ = detect_frequency(data)
    cfg.frequency = detected
    unit = {"Daily": "day", "Weekly": "week", "Monthly": "month"}.get(cfg.frequency, "period")
    st.markdown(f"**Historical Data Window**  \n{min_date.strftime('%b %-d, %Y')} – {historical_end.strftime('%b %-d, %Y')}")
    st.markdown(f"**Data Frequency**  \n{cfg.frequency}")
    st.caption(f"Planned campaign launch: {launch.strftime('%b %-d, %Y')} · Historical data available for planning: {len(data):,} observations")
    st.info(f"The planner uses only observations before the planned launch. Ramp-up and stable measurement settings are configured in Detectability & Duration.")
    st.plotly_chart(time_series_line(data, "", humanize_column_name(cfg.outcome_column), planned_launch_date=cfg.planned_launch_date), use_container_width=True)


def ts_analysis_step() -> None:
    st.subheader("Analysis")
    cfg: TimeSeriesConfig = st.session_state.ts_config
    data = st.session_state.ts_prepared_df
    if data.empty:
        st.info("Upload a historical time-series dataset before running analysis.")
        return
    if not cfg.intervention_date:
        st.info("Select the campaign launch date to define the pre- and post-periods.")
        return
    cfg.analysis_method = st.radio("Analysis Method", ["Pre–Post Analysis", "Structural Time Series Counterfactual"], horizontal=True)
    if cfg.analysis_method == "Pre–Post Analysis":
        st.write("Descriptive comparison only; it does not adjust for trend, seasonality, or concurrent changes.")
        cfg.comparison_window = st.radio("Comparison Window", ["Symmetric", "All available", "Custom date range"], index=["Symmetric", "All available", "Custom date range"].index(cfg.comparison_window) if cfg.comparison_window in ["Symmetric", "All available", "Custom date range"] else 0, horizontal=True)
        if cfg.comparison_window == "Symmetric":
            cfg.symmetric_periods = st.number_input("Periods before and after", min_value=2, max_value=104, value=int(cfg.symmetric_periods), step=1)
        elif cfg.comparison_window == "All available":
            cfg.pre_window, cfg.post_window = "All available pre-period", "All available post-period"
        else:
            pre_data = data[data["_date"] < pd.Timestamp(cfg.intervention_date)]
            post_data = data[data["_date"] >= pd.Timestamp(cfg.intervention_date)]
            a, b = st.columns(2)
            cfg.custom_pre_start = a.date_input("Pre-period start", value=pre_data["_date"].min().date(), min_value=pre_data["_date"].min().date(), max_value=pre_data["_date"].max().date()).isoformat()
            cfg.custom_pre_end = a.date_input("Pre-period end", value=pre_data["_date"].max().date(), min_value=pre_data["_date"].min().date(), max_value=pre_data["_date"].max().date()).isoformat()
            cfg.custom_post_start = b.date_input("Post-period start", value=post_data["_date"].min().date(), min_value=post_data["_date"].min().date(), max_value=post_data["_date"].max().date()).isoformat()
            cfg.custom_post_end = b.date_input("Post-period end", value=post_data["_date"].max().date(), min_value=post_data["_date"].min().date(), max_value=post_data["_date"].max().date()).isoformat()
        cfg.comparison_metric = st.radio("Comparison Metric", ["Average per period", "Total"], horizontal=True)
        prepost_fingerprint = f"{method_config_fingerprint(cfg, 'Pre–Post Analysis')}:{data_fingerprint(data)}"
        try:
            st.session_state.ts_prepost_result = run_pre_post_analysis(data, cfg.intervention_date, cfg.campaign_end_date, cfg.comparison_metric, cfg.pre_window, cfg.post_window, cfg.comparison_window, int(cfg.symmetric_periods), (cfg.custom_pre_start, cfg.custom_pre_end, cfg.custom_post_start, cfg.custom_post_end))
            st.session_state.ts_prepost_fingerprint = prepost_fingerprint
            st.session_state.ts_prepost_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except Exception as exc:
            st.session_state.ts_prepost_result = None
            st.session_state.ts_prepost_fingerprint = ""
            st.error(f"Pre–Post analysis is not ready: {exc}")
        if st.session_state.ts_prepost_result:
            result = st.session_state.ts_prepost_result
            st.success("Pre–Post Analysis Ready")
            st.caption(f"{result['pre_observations']} periods before vs {result['post_observations']} periods after | Estimated change: {percent(result['percent_change'], 1)}")
    else:
        st.write("Estimate the expected no-intervention trajectory using pre-intervention history.")
        numeric = predictor_candidates(data, cfg.outcome_column, cfg.date_column, cfg.campaign_flag_column)
        st.markdown("**Optional Predictors**")
        st.caption("Select time series associated with the outcome that are not affected by the intervention.")
        cfg.selected_covariates = st.multiselect("Optional Predictors", numeric, default=[col for col in cfg.selected_covariates if col in numeric], format_func=humanize_column_name)
        cfg.seasonality = st.selectbox("Seasonality", ["Auto", "None", "Weekly", "Monthly", "Annual"], index=["Auto", "None", "Weekly", "Monthly", "Annual"].index(cfg.seasonality))
        cfg.prediction_interval = st.selectbox("Confidence Level", [0.8, 0.9, 0.95], index=[0.8, 0.9, 0.95].index(cfg.prediction_interval), format_func=lambda value: f"{int(value * 100)}%")
        with st.expander("Advanced Settings", expanded=False):
            cfg.local_trend = st.checkbox("Use Local Trend", value=cfg.local_trend)
            cfg.simulation_seed = st.number_input("Simulation Seed", min_value=1, max_value=999999, value=int(cfg.simulation_seed), step=1)
            cfg.holdout_length = st.number_input("Holdout Length", min_value=4, max_value=24, value=int(cfg.holdout_length), step=1)
        current_fingerprint = f"{method_config_fingerprint(cfg, 'Structural Time Series Counterfactual')}:{data_fingerprint(data)}"
        status = st.session_state.ts_structural_status
        has_result = bool(st.session_state.ts_bsts_result)
        action_label = "Recalculate Analysis" if has_result else "Run Analysis"
        if st.session_state.ts_structural_fingerprint and st.session_state.ts_structural_fingerprint != current_fingerprint:
            status = "stale"
            st.session_state.ts_structural_status = status
            action_label = "Update Analysis"
            st.warning("⚠ Results out of date\n\nSettings changed since the last analysis.")
        if status == "running":
            st.info("⟳ Running analysis...\n\nFitting the expected no-intervention trajectory")
        elif status == "completed":
            st.success("✓ Analysis completed")
            st.caption(f"Last calculated: {st.session_state.ts_structural_timestamp}")
        elif status == "failed":
            st.warning("Analysis failed\n\nThe model could not complete with the current settings. Review the configuration and try again.")
        button_disabled = status == "running"
        if st.button("Running Analysis..." if button_disabled else action_label, type="primary", key="ts_run_structural", disabled=button_disabled):
            st.session_state.ts_structural_status = "running"
            st.session_state.ts_structural_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.ts_structural_error = ""
            with st.spinner("Fitting counterfactual model..."):
                try:
                    result = fit_bsts_model(data, cfg.intervention_date, cfg.frequency, cfg.selected_covariates, cfg.prediction_interval, campaign_end_date=cfg.campaign_end_date, seasonality=cfg.seasonality, local_trend=cfg.local_trend, seed=int(cfg.simulation_seed), holdout_length=int(cfg.holdout_length))
                    result["config_fingerprint"] = current_fingerprint
                    st.session_state.ts_bsts_result = result
                    st.session_state.ts_structural_fingerprint = result["config_fingerprint"]
                    st.session_state.ts_structural_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.ts_structural_status = "completed"
                except Exception as exc:
                    st.session_state.ts_structural_status = "failed"
                    st.session_state.ts_structural_error = str(exc)
                    print(f"BSTS failed: {exc}")
            st.rerun()


def ts_results_step() -> None:
    st.subheader("Results")
    cfg: TimeSeriesConfig = st.session_state.ts_config
    data = st.session_state.ts_prepared_df
    if data.empty or not cfg.intervention_date:
        st.info("Configure the intervention period before viewing results.")
        return
    evaluation = data[data["_date"] >= pd.Timestamp(cfg.intervention_date)]
    first_treated = evaluation["_date"].min() if not evaluation.empty else pd.Timestamp(cfg.intervention_date)
    evaluation_end = evaluation["_date"].max() if not evaluation.empty else data["_date"].max()
    st.markdown("**Analysis Definition**")
    d1, d2, d3 = st.columns(3)
    d1.write(f"**Outcome**\n\n{humanize_column_name(cfg.outcome_column)}")
    d2.write(f"**Intervention**\n\n{pd.Timestamp(cfg.intervention_date).strftime('%b %-d, %Y')}\n\n**First Treated Observation**\n\n{first_treated.strftime('%b %-d, %Y')}")
    d3.write(f"**Evaluation Period**\n\n{first_treated.strftime('%b %-d, %Y')} – {evaluation_end.strftime('%b %-d, %Y')} · {len(evaluation)} weeks\n\n**Method**\n\n{cfg.analysis_method}")
    current_fingerprint = f"{method_config_fingerprint(cfg, cfg.analysis_method)}:{data_fingerprint(data)}"
    if cfg.analysis_method == "Pre–Post Analysis":
        prepost = st.session_state.ts_prepost_result if st.session_state.ts_prepost_fingerprint == current_fingerprint else None
        if prepost:
            st.plotly_chart(time_series_line(data, cfg.intervention_date, humanize_column_name(cfg.outcome_column)), use_container_width=True)
            st.markdown("**Pre–Post Analysis**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Before Campaign", number(prepost["pre_value"], 2))
            c2.metric("After Campaign", number(prepost["post_value"], 2))
            c3.metric("Absolute Change", number(prepost["absolute_change"], 2))
            c4.metric("Percent Change", percent(prepost["percent_change"], 1))
            st.plotly_chart(pre_post_bar(prepost["pre_value"], prepost["post_value"], humanize_column_name(cfg.outcome_column)), use_container_width=True)
            st.info("Pre–Post comparison does not adjust for trend, seasonality, or concurrent changes.")
        else:
            st.warning("Pre–Post analysis is not available for the current settings.")
        return

    bsts = st.session_state.ts_bsts_result if st.session_state.ts_structural_fingerprint == current_fingerprint else None
    if st.session_state.ts_structural_fingerprint and not bsts:
        st.warning("⚠ Results out of date\n\nAnalysis settings changed since the last calculation.")
    if bsts:
        label = humanize_column_name(cfg.outcome_column)
        interp = build_business_interpretation(bsts["cumulative_impact"], bsts["relative_impact"], bsts["prediction_interval"], bsts["simulation_probability_positive"], bsts.get("reliability", "Poor"), cfg.outcome_column)
        st.markdown("**Campaign Impact**")
        st.caption("Method: Structural Time Series Counterfactual")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(incremental_outcome_label(cfg.outcome_column), number(bsts["cumulative_impact"], 0))
        c2.metric("Estimated Lift", percent(bsts["relative_impact"], 1))
        low, high = bsts["prediction_interval"]
        c3.metric(f"Likely Impact Range ({percent(bsts['interval_level'], 0)})", f"{low:+,.0f} to {high:+,.0f}", help="The range of campaign impact values that are reasonably consistent with the model and observed data.")
        c4.metric("Probability Impact Is Positive", probability_positive_label(bsts["simulation_probability_positive"]), help="The share of model simulations in which the estimated campaign effect is above zero. This depends on the counterfactual model being appropriate.")
        st.markdown("**Actual Outcome**")
        a1, a2 = st.columns(2)
        a1.metric(f"Actual {label}", number(bsts["observed_total"], 0))
        a2.metric("Expected Without Campaign", number(bsts["expected_total"], 0))
        st.markdown("**Why this result?**")
        st.write(interp["interpretation"])
        if interp["status"] == "Positive — Strong Evidence":
            st.success(interp["headline"])
        else:
            st.warning(interp["headline"])
        st.caption(f"Recommended next step: {interp['next_step']}")
        st.markdown("**What does this mean?**")
        st.write(f"During the {len(bsts['result'])}-week evaluation period, actual {label} were approximately {abs(bsts['relative_impact']) * 100:.1f}% {'higher' if bsts['cumulative_impact'] >= 0 else 'lower'} than the estimated trajectory without the intervention. This corresponds to approximately {abs(bsts['cumulative_impact']):,.0f} incremental {label.lower()}.")
        st.markdown("**Actual vs Expected**")
        chart_context = timeseries_chart_context(data, cfg.intervention_date, cfg.campaign_end_date)
        st.plotly_chart(bsts_counterfactual_chart(data, bsts["result"], cfg.intervention_date, label, chart_context), use_container_width=True)
        st.markdown("**Cumulative Estimated Impact**")
        st.caption(f"Running total of estimated incremental {label.lower()} since campaign launch. Positive values mean cumulative outcomes were above the estimated no-campaign trajectory; negative values mean they were below it.")
        st.plotly_chart(cumulative_impact_chart(bsts["result"], label, chart_context, bsts["cumulative_impact"]), use_container_width=True)
        st.markdown("**Model Reliability**")
        backtest = bsts.get("backtest", {})
        details = bsts.get("reliability_details", {})
        components = details.get("components", {})
        explanations = details.get("explanations", {})
        v1, v2, v3, v4 = st.columns(4)
        mape = backtest.get("mape")
        coverage = backtest.get("interval_coverage")
        error_grade = components.get("forecast_error", "Review")
        coverage_grade = components.get("forecast_range_coverage", "Review")
        pattern_grade = components.get("time_pattern_fit", "Review")
        overall_grade = bsts.get("reliability", "Review")
        v1.metric("Average Forecast Error", f"{mape * 100:.1f}%" if pd.notna(mape) else "Review", help="On historical holdout data, the model's forecasts differed from actual outcomes by about this amount on average. Lower is better.")
        v1.caption(error_grade)
        v2.metric("Forecast Range Coverage", f"{coverage * 100:.1f}%" if pd.notna(coverage) else "Review", help=f"The percentage of historical observations inside the model's expected range. This should generally be reasonably close to the selected {bsts.get('interval_level', 0.95) * 100:.0f}% confidence level.")
        v2.caption(f"{coverage_grade} | {bsts.get('interval_level', 0.95) * 100:.0f}% target")
        v3.metric("Time Pattern Fit", "Good" if pattern_grade == "Good" else "Needs attention", help="Checks whether forecast errors still contain recurring time-related patterns. Remaining patterns can mean trend or seasonality is not fully captured.")
        v3.caption("Some recurring structure remains" if pattern_grade != "Good" else "Recurring structure mostly captured")
        v4.metric("Overall Reliability", overall_grade, help="Reliability ratings summarize model validation diagnostics. They are product-level heuristics, not formal statistical significance categories.")
        v4.caption("Use with caution" if overall_grade == "Fair" else "Review before relying on the estimate" if overall_grade == "Poor" else "Validation is broadly consistent")
        if overall_grade == "Fair":
            st.info("**Model reliability is fair.** Historical forecasts are reasonably accurate, but some diagnostics need attention. Use the impact estimate with moderate caution.")
        elif overall_grade == "Poor":
            st.warning("**Model reliability is low.** Historical validation indicates that the counterfactual may not be reliable enough for a high-confidence decision.")
        reasons = [f"{'✓' if grade == 'Good' else '△'} {explanations.get(key, '')}" for key, grade in components.items() if explanations.get(key)]
        with st.expander("Why this rating?", expanded=False):
            st.caption("Reliability combines forecast error, forecast-range coverage, remaining time patterns, forecast bias, data sufficiency, holdout validation, and model stability.")
            for reason in reasons:
                st.write(reason)
        with st.expander("Diagnostics", expanded=False):
            st.plotly_chart(impact_chart(bsts["result"]), use_container_width=True)
            st.write("Residual autocorrelation detected" if bsts.get("residual_autocorrelation") else "No material residual autocorrelation detected")
            st.caption("Forecast errors still follow a time pattern, which suggests that some trend, seasonality, or recurring behavior has not been fully captured by the model.")
            st.write(f"MAE: {backtest.get('mae', float('nan')):,.2f} | RMSE: {backtest.get('rmse', float('nan')):,.2f} | Bias: {backtest.get('bias', float('nan')):,.2f}")
            specification = bsts.get("model_specification", {})
            st.write("Model specification: " + ", ".join(f"{key.replace('_', ' ')} = {value}" for key, value in specification.items()))
            st.write(f"Simulation seed: {bsts.get('seed')}")
        st.info("This state-space forecast assumes pre-period dynamics remain valid and selected predictors are not affected by the campaign. Simulation intervals include future state and observation uncertainty and are conditional on the fitted parameters.")
    prepost_current_fingerprint = f"{method_config_fingerprint(cfg, 'Pre–Post Analysis')}:{data_fingerprint(data)}"
    prepost = st.session_state.ts_prepost_result if st.session_state.ts_prepost_fingerprint == prepost_current_fingerprint else None
    if prepost:
        with st.expander("Naive Pre–Post Benchmark", expanded=False):
            st.metric("Descriptive Benchmark", percent(prepost["percent_change"], 1))
            st.caption("Pre–Post comparison does not adjust for trend, seasonality, or concurrent changes.")
            st.plotly_chart(pre_post_bar(prepost["pre_value"], prepost["post_value"], humanize_column_name(cfg.outcome_column)), use_container_width=True)
    if st.session_state.ts_structural_timestamp:
        st.caption(f"Last calculated: {st.session_state.ts_structural_timestamp}")


def ts_planning_step() -> None:
    """Pre-launch planning workflow; intentionally does not expose an analysis-method choice."""
    st.subheader("Campaign Duration Planner")
    st.caption("Estimate how long a campaign needs to run to detect a meaningful change using historical outcome behavior.")
    cfg: TimeSeriesConfig = st.session_state.ts_config
    data = st.session_state.ts_prepared_df
    if data.empty:
        st.info("Load and map historical time-series data in the Data step before planning.")
        return
    launch = pd.Timestamp(cfg.planned_launch_date) if cfg.planned_launch_date else data["_date"].max() + pd.Timedelta(days=7)
    planning_data = data[data["_date"] < launch].copy()
    if launch <= data["_date"].max():
        st.warning("Planned campaign launch occurs before the end of the available historical data. Only data before launch will be used for planning.")
    planning_objective = st.radio("Planning Objective", ["Find duration for a meaningful effect", "Find detectable effect by duration"], horizontal=True, key="ts_plan_objective")
    effect_type = st.selectbox("Effect Scale", ["Relative Lift (%)", "Absolute Effect"], key="ts_plan_effect_type")
    if planning_objective == "Find duration for a meaningful effect":
        effect = st.number_input("Assumed Minimum Meaningful Effect", min_value=0.0, value=5.0 if effect_type == "Relative Lift (%)" else 200.0, step=0.5, key="ts_plan_effect", help="This is a planning assumption or business threshold, not a forecast of the campaign's actual effect.")
        planning_effect = effect / 100 if effect_type == "Relative Lift (%)" else effect
    else:
        effect = None
        planning_effect = None
        st.info("No campaign effect is assumed. The planner will estimate the minimum detectable effect for each candidate duration.")
    st.markdown("**Detectability Settings**")
    c1, c2 = st.columns(2)
    alpha = c1.selectbox("Significance Level (Alpha)", [0.05, 0.10], format_func=lambda value: f"{value:.2f}", key="ts_plan_alpha")
    target_power = c2.selectbox("Target Statistical Power", [0.8, 0.9], format_func=lambda value: f"{int(value * 100)}%", key="ts_plan_power")
    st.markdown("**Planning Constraints**")
    candidate_durations = planning_durations(cfg.frequency, 10_000)
    maximum = st.selectbox(
        "Maximum Campaign Duration",
        candidate_durations,
        index=len(candidate_durations) - 1,
        format_func=lambda value: duration_label(value, cfg.frequency),
        key=f"ts_plan_max_{cfg.frequency.lower()}",
    )
    st.caption(f"Longest campaign duration the planner will consider. Candidate durations use the detected {cfg.frequency.lower()} data cadence.")
    gradual = st.checkbox("Gradual ramp-up", key="ts_plan_ramp_enabled")
    default_ramp = {"Daily": 7, "Weekly": 4, "Monthly": 1}.get(cfg.frequency, 1)
    ramp = st.number_input(f"Ramp-up duration ({duration_unit(cfg.frequency)}s)", min_value=1, max_value=maximum, value=min(default_ramp, maximum), step=1, key=f"ts_plan_ramp_{cfg.frequency.lower()}") if gradual else 1
    effect_pattern = "gradual_ramp" if gradual else "constant"
    with st.expander("Advanced / Model Inputs", expanded=False):
        st.write(f"Historical observations: {len(planning_data):,} usable before launch")
        st.write(f"Detected frequency: {cfg.frequency}")
        available_predictors = predictor_candidates(planning_data, cfg.outcome_column, cfg.date_column, cfg.campaign_flag_column)
        cfg.selected_covariates = st.multiselect(
            "Optional Unaffected Predictors",
            available_predictors,
            default=[column for column in cfg.selected_covariates if column in available_predictors],
            format_func=humanize_column_name,
            help="Use only variables that the campaign cannot change and that will be available throughout the measurement period.",
            key="ts_plan_predictors",
        )
        cfg.seasonality = st.selectbox("Seasonality", ["Auto", "None", "Weekly", "Monthly", "Annual"], index=["Auto", "None", "Weekly", "Monthly", "Annual"].index(cfg.seasonality), key="ts_plan_seasonality")
        cfg.local_trend = st.checkbox("Use Local Linear Trend", value=cfg.local_trend, key="ts_plan_local_trend")
        st.caption("Predictors can reduce long-horizon counterfactual uncertainty when they are stable, unaffected by the campaign, and available after launch.")
    current_plan_fingerprint = planning_config_fingerprint(planning_data, cfg, effect_type, planning_effect, effect_pattern, int(ramp), alpha, target_power, int(maximum))
    if st.session_state.ts_duration_plan and st.session_state.ts_duration_plan_fingerprint != current_plan_fingerprint:
        st.warning("Duration estimate out of date. Update the inputs and calculate a new estimate.")
    if planning_objective == "Find detectable effect by duration":
        calculate_label = "Update Detectable Effects" if st.session_state.ts_duration_plan and st.session_state.ts_duration_plan_fingerprint != current_plan_fingerprint else "Calculate Detectable Effects"
        spinner_text = "Calculating minimum detectable effects..."
    else:
        calculate_label = "Update Duration Estimate" if st.session_state.ts_duration_plan and st.session_state.ts_duration_plan_fingerprint != current_plan_fingerprint else "Calculate Required Duration"
        spinner_text = "Calculating required campaign duration..."
    if st.button(calculate_label, type="primary", key="ts_plan_calculate"):
        with st.spinner(spinner_text):
            st.session_state.ts_duration_plan = run_duration_power_simulation(planning_data, cfg.frequency, cfg.selected_covariates, cfg.seasonality, cfg.local_trend, 1 - alpha, planning_effect, effect_type, target_power, int(maximum), int(ramp), simulations=40, seed=int(cfg.simulation_seed), alpha=alpha, effect_pattern="gradual_ramp" if gradual else "constant")
            st.session_state.ts_duration_plan["effect"] = effect
            st.session_state.ts_duration_plan["effect_type"] = effect_type
            st.session_state.ts_duration_plan["planning_objective"] = planning_objective
            st.session_state.ts_duration_plan["alpha"] = alpha
            st.session_state.ts_duration_plan["target_power"] = target_power
            st.session_state.ts_duration_plan["ramp"] = ramp
            st.session_state.ts_duration_plan["predictors"] = list(cfg.selected_covariates)
            st.session_state.ts_duration_plan_fingerprint = current_plan_fingerprint
        plan_status = st.session_state.ts_duration_plan
        if planning_objective == "Find detectable effect by duration":
            st.success("Calculation completed — minimum detectable effects estimated by duration.")
        elif plan_status.get("calibration_status") == "poor":
            st.warning("Calculation completed — target power achieved, but model calibration is poor." if plan_status.get("recommended_duration") else "Calculation completed with model calibration warnings.")
        elif plan_status.get("recommended_duration") and plan_status.get("calibration_status") == "caution":
            st.info("Calculation completed — target power achieved with calibration caution.")
        elif plan_status.get("recommended_duration"):
            st.success("Calculation completed — target power achieved.")
        else:
            st.info("Calculation completed — target power not achieved within the selected maximum duration.")
    plan = st.session_state.ts_duration_plan if st.session_state.ts_duration_plan_fingerprint == current_plan_fingerprint else None
    if not plan:
        return
    if plan.get("status") in {"insufficient_history", "insufficient_calibration"}:
        st.warning("Insufficient independent calibration or evaluation windows to estimate campaign duration reliably.")
        return
    if plan.get("status") != "ok":
        return
    recommendation = plan.get("recommended_duration")
    validation_mode = plan.get("validation_mode", "Review")
    confidence = plan.get("recommendation_confidence", "Review")
    st.markdown("**Validation Evidence**")
    st.write(f"{validation_mode} · Recommendation confidence: {confidence}")
    st.caption(f"{plan.get('historical_origins', 0):,} historical pseudo-launch origins · Minimum training history: {plan.get('minimum_training_observations', 0):,} observations · Predictors: {len(plan.get('predictors', [])):,} · Planner seasonality: {plan.get('model_seasonality', 'None')}")
    if validation_mode == "Assumption-based":
        st.warning(plan.get("assumption_note", "History is too short for empirical pseudo-launch validation. Treat this estimate as preliminary."))
    elif validation_mode != "Robust empirical":
        st.info("Historical validation is limited. Forecast paths supplement the available pseudo-launch origins, so treat the result as directional rather than fully empirical.")
    st.markdown("**Campaign Duration Recommendation**" if effect is not None else "**Minimum Detectable Effect by Duration**")
    if recommendation:
        selected = next(row for row in plan["power"] if row["duration"] == recommendation)
        stable = max(1, recommendation - int(ramp))
        launch_anchor = pd.Timestamp(cfg.planned_launch_date or cfg.intervention_date)
        end = campaign_decision_date(launch_anchor, recommendation, cfg.frequency)
        st.success(duration_label(recommendation, cfg.frequency))
        st.write(f"Minimum duration to detect an assumed {plan['effect']}{'%' if plan['effect_type'] == 'Relative Lift (%)' else ' outcome-unit'} effect with {selected['power']:.0%} estimated power. The assumed effect is a business planning threshold, not a forecast.")
        st.caption(f"Ramp-up: {duration_label(ramp, cfg.frequency)} · Stable measurement window: {duration_label(stable, cfg.frequency)} · Earliest reliable decision date: {end.strftime('%b %-d, %Y')}")
        m1, m2 = st.columns(2)
        m1.metric("Required Campaign Duration", duration_label(recommendation, cfg.frequency))
        m2.metric("Required Time-Series Sample", f"{recommendation:,} time points", help="This is the required number of post-launch time observations, not the number of customers or transactions.")
        m2.caption(f"Post-launch {cfg.frequency.lower()} aggregates")
        m3, m4 = st.columns(2)
        m3.metric("Historical Observations Used", f"{len(planning_data):,}")
        m4.metric("Earliest Reliable Decision", end.strftime("%b %-d, %Y"))
        operational_exposure = projected_operational_exposure(planning_data, cfg.exposure_column, recommendation)
        if operational_exposure is not None:
            st.metric("Expected Operational Exposure", f"{operational_exposure:,.0f}", help="Operational projection: median historical exposure per period multiplied by the recommended duration. This is not a statistical customer sample size.")
        st.plotly_chart(time_series_line(data, "", humanize_column_name(cfg.outcome_column), planned_launch_date=cfg.planned_launch_date, recommended_end_date=end.isoformat()), use_container_width=True)
    elif effect is not None:
        st.warning("Target power is not reached within the selected maximum duration.")
    else:
        reference_row = next((row for row in plan["power"] if row["duration"] == 12 and pd.notna(row.get("minimum_detectable_effect"))), None)
        if reference_row is None:
            reference_row = next((row for row in reversed(plan["power"]) if pd.notna(row.get("minimum_detectable_effect"))), None)
        if reference_row:
            mde = reference_row["minimum_detectable_effect"]
            display_mde = f"{mde * 100:.1f}%" if effect_type == "Relative Lift (%)" else f"{mde:,.1f}"
            st.metric(f"Minimum Detectable Effect at {duration_label(reference_row['duration'], cfg.frequency)}", display_mde)
            st.caption(f"Smallest modeled effect expected to reach {target_power:.0%} power at that duration. This is detectability, not a forecast of actual campaign performance.")
    evaluated_rows = sorted(plan["power"], key=lambda row: int(row["duration"]))
    if effect is None:
        st.markdown("**Detectable Effect by Duration**")
        st.caption(f"Smallest modeled effect detectable at {target_power:.0%} power. The gain column shows whether the additional campaign time meaningfully improves sensitivity.")
        sensitivity_rows = []
        previous_mde = None
        previous_duration = None
        for row in evaluated_rows:
            duration = int(row["duration"])
            mde = row.get("minimum_detectable_effect")
            if pd.isna(mde):
                display_mde = "Not evaluated"
                gain = "Not evaluated"
            elif previous_mde is None:
                display_mde = f"{mde * 100:.1f}%" if effect_type == "Relative Lift (%)" else f"{mde:,.1f}"
                gain = "Baseline duration"
            else:
                display_mde = f"{mde * 100:.1f}%" if effect_type == "Relative Lift (%)" else f"{mde:,.1f}"
                improvement = previous_mde - mde
                previous_label = duration_label(previous_duration, cfg.frequency)
                if effect_type == "Relative Lift (%)":
                    change = abs(improvement) * 100
                    gain = f"No material change from {previous_label}" if change < 0.05 else f"{change:.1f} pp smaller than {previous_label}" if improvement > 0 else f"{change:.1f} pp larger than {previous_label}"
                else:
                    change = abs(improvement)
                    gain = f"No material change from {previous_label}" if change < 0.05 else f"{change:,.1f} units smaller than {previous_label}" if improvement > 0 else f"{change:,.1f} units larger than {previous_label}"
            sensitivity_rows.append(
                {
                    "Campaign Duration": duration_label(duration, cfg.frequency),
                    "Minimum Detectable Lift" if effect_type == "Relative Lift (%)" else "Minimum Detectable Effect": display_mde,
                    "Sensitivity Gain vs Shorter Option": gain,
                }
            )
            if pd.notna(mde):
                previous_mde = mde
                previous_duration = duration
        sensitivity = pd.DataFrame(sensitivity_rows)
        st.dataframe(sensitivity, hide_index=True, use_container_width=True)
    else:
        st.markdown("**Power by Campaign Duration**")
        st.caption(f"Chance of detecting the assumed {effect:.1f}{'%' if effect_type == 'Relative Lift (%)' else '-unit'} effect when it is truly present. The shortest valid option reaching {target_power:.0%} is recommended.")
        finite_power_rows = [row for row in evaluated_rows if pd.notna(row.get("power"))]
        if finite_power_rows:
            peak_row = max(finite_power_rows, key=lambda row: row["power"])
            later_rows = [row for row in finite_power_rows if int(row["duration"]) > int(peak_row["duration"])]
            if later_rows and min(row["power"] for row in later_rows) < peak_row["power"] - 0.03:
                st.warning(f"Sensitivity peaks near {duration_label(int(peak_row['duration']), cfg.frequency)} and declines at longer horizons because counterfactual uncertainty grows faster than the assumed effect. Review unaffected predictors and trend settings before extending the campaign solely to collect more periods.")
        sensitivity_rows = []
        for row in evaluated_rows:
            duration = int(row["duration"])
            power = row.get("power", float("nan"))
            mde = row.get("minimum_detectable_effect", float("nan"))
            calibration = row.get("calibration_status", "not_evaluated")
            row_status = str(row.get("status", ""))
            if duration == recommendation:
                guidance = "Recommended: shortest valid option"
            elif not pd.notna(power):
                guidance = "Could not evaluate"
            elif row_status.startswith("Near power target"):
                guidance = "Near target; uncertainty overlaps"
            elif power < target_power:
                guidance = "Below detection target"
            elif calibration in {"caution", "poor"}:
                guidance = "Target met; review calibration"
            else:
                guidance = "Meets target; additional time"
            sensitivity_rows.append(
                {
                    "Campaign Duration": duration_label(duration, cfg.frequency),
                    "Detection Chance": f"{power:.0%}" if pd.notna(power) else "Not evaluated",
                    "Minimum Detectable Lift" if effect_type == "Relative Lift (%)" else "Minimum Detectable Effect": f"{mde * 100:.1f}%" if effect_type == "Relative Lift (%)" and pd.notna(mde) else f"{mde:,.1f}" if pd.notna(mde) else "Not evaluated",
                    "Guidance": guidance,
                }
            )
        st.dataframe(pd.DataFrame(sensitivity_rows), hide_index=True, use_container_width=True)
        interval_rows = []
        for row in evaluated_rows:
            low, high = row.get("power_ci", (float("nan"), float("nan")))
            if pd.notna(low) and pd.notna(high):
                interval_rows.append(
                    {
                        "Campaign Duration": duration_label(int(row["duration"]), cfg.frequency),
                        "Estimated Detection Chance": f"{row['power']:.1%}",
                        "Simulation 95% Interval": f"{low:.0%} to {high:.0%}",
                        "Simulated Paths": f"{int(row.get('simulation_paths', 0)):,}",
                    }
                )
        if interval_rows:
            with st.expander("Simulation precision details", expanded=False):
                st.write("This interval describes uncertainty from estimating detection probability with a finite number of simulated paths. It is not the uncertainty interval for the campaign's effect.")
                st.dataframe(pd.DataFrame(interval_rows), hide_index=True, use_container_width=True)
    st.markdown("**Model Calibration**")
    if plan["calibration_status"] == "assumption_based":
        st.info(f"Calibration is assumption-based. The calculation uses a nominal {alpha:.0%} false-positive rate because history is too short for held-out placebo validation.")
    elif plan["calibration_status"] == "acceptable":
        st.success(f"✓ Calibration acceptable · False-positive behavior is consistent with the selected {alpha:.0%} target.")
    else:
        st.warning(f"⚠ Calibration needs review. False-positive behavior is not consistently aligned with the selected {alpha:.0%} target, so treat duration and MDE estimates cautiously.")
    if plan["calibration_status"] != "assumption_based":
        with st.expander("Calibration details", expanded=False):
            st.write("Model FPR is the false-alarm rate across simulated no-campaign forecast paths. Historical false alarms come from held-out pseudo-launches. Because historical counts are often small, calibration uses a one-sided exact binomial test against the selected false-positive target instead of comparing raw percentages directly.")
            calibration_rows = []
            for row in evaluated_rows:
                model_fpr = row.get("false_positive_rate", float("nan"))
                historical_fpr = row.get("historical_false_positive_rate", float("nan"))
                historical_origins = int(row.get("valid_placebos", 0))
                historical_false_positives = int(row.get("historical_false_positives", round(historical_fpr * historical_origins))) if pd.notna(historical_fpr) else 0
                historical_pvalue = row.get("historical_calibration_pvalue", float("nan"))
                historical_check = "Excess false alarms" if pd.notna(historical_pvalue) and historical_pvalue < 0.05 else "Borderline" if pd.notna(historical_pvalue) and historical_pvalue < 0.10 else "Consistent with target" if pd.notna(historical_pvalue) else "Not evaluated"
                calibration_rows.append(
                    {
                        "Campaign Duration": duration_label(int(row["duration"]), cfg.frequency),
                        "Model FPR": f"{model_fpr:.1%}" if pd.notna(model_fpr) else "Not evaluated",
                        "Historical False Alarms": f"{historical_false_positives} of {historical_origins} ({historical_fpr:.1%})" if pd.notna(historical_fpr) else "Not evaluated",
                        "Historical Check": f"{historical_check} (p={historical_pvalue:.3f})" if pd.notna(historical_pvalue) else historical_check,
                        "Calibration": humanize_column_name(str(row.get("calibration_status", "review"))),
                    }
                )
            st.dataframe(pd.DataFrame(calibration_rows), hide_index=True, use_container_width=True)
            st.caption(f"Target false-positive rate: {alpha:.0%}. A historical p-value below 0.05 indicates evidence of excess false alarms; a large raw percentage alone is not treated as failure when only a few origins are available.")
    with st.expander("Methodology", expanded=False):
        st.write(f"One observation is one {cfg.frequency.lower()} aggregate. The planner uses {validation_mode.lower()} validation. Pseudo-launches use a consistent model specification and at least {plan.get('minimum_training_observations', 0)} pre-launch observations. Calibration origins determine a one-sided critical threshold for a planned positive effect; separate held-out origins estimate false-positive rate and power. Sparse historical false alarms are evaluated with an exact binomial test. Correlated no-campaign forecast paths are reused to evaluate assumed effects and to solve for the minimum detectable effect at each duration. When history is too short for structural validation, the planner switches to a labeled analytical approximation using detrended residual variance and autocorrelation. A supplied effect is a scenario, not a forecast. When simulation intervals are available, the lower bound must clear the target before a duration is recommended. Every candidate through the selected maximum duration is evaluated. Required Time-Series Sample means post-launch time points, not customers or transactions.")


def ts_decision_step() -> None:
    st.subheader("Decision")
    cfg: TimeSeriesConfig = st.session_state.ts_config
    current_fingerprint = f"{method_config_fingerprint(cfg, cfg.analysis_method)}:{data_fingerprint(st.session_state.ts_prepared_df)}"
    if cfg.analysis_method == "Structural Time Series Counterfactual" and st.session_state.ts_structural_fingerprint and st.session_state.ts_structural_fingerprint != current_fingerprint:
        st.warning("The previous result is stale because the time-series configuration changed. Re-run the analysis.")
        return
    if cfg.analysis_method == "Structural Time Series Counterfactual":
        result = st.session_state.ts_bsts_result if st.session_state.ts_structural_fingerprint == current_fingerprint else None
    else:
        result = st.session_state.ts_prepost_result if st.session_state.ts_prepost_fingerprint == current_fingerprint else None
    if not result:
        st.info("Run an analysis before generating the decision summary.")
        return
    st.markdown("**Analysis Conclusion**")
    st.write(f"Outcome: {humanize_column_name(cfg.outcome_column)}")
    st.write(f"Evaluation Period: {cfg.intervention_date} – {cfg.campaign_end_date or st.session_state.ts_prepared_df['_date'].max().date()}")
    if cfg.analysis_method == "Structural Time Series Counterfactual":
        low, high = result["prediction_interval"]
        interp = build_business_interpretation(result["cumulative_impact"], result["relative_impact"], result["prediction_interval"], result["simulation_probability_positive"], result.get("reliability", "Poor"), cfg.outcome_column)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Estimated Lift", percent(result["relative_impact"], 1))
        c2.metric(incremental_outcome_label(cfg.outcome_column), number(result["cumulative_impact"], 0))
        c3.metric("Likely Impact Range", f"{low:+,.0f} to {high:+,.0f}")
        c4.metric("Probability Impact Is Positive", probability_positive_label(result["simulation_probability_positive"]))
        st.write(f"Model Reliability: {result.get('reliability', 'Review')}")
        st.markdown("**Interpretation**")
        st.write(interp["interpretation"])
        st.markdown("**Analytical Conclusion**")
        conclusion = "Positive evidence — high confidence" if result.get("decision") == "positive" and result.get("reliability") == "Good" else "Positive evidence — moderate confidence" if result.get("decision") == "positive" else "Negative evidence — high confidence" if result.get("decision") == "negative" and result.get("reliability") == "Good" else "Negative evidence — moderate confidence" if result.get("decision") == "negative" else "Inconclusive"
        st.write(conclusion)
        st.markdown("**Analytical Next Step**")
        st.write(interp["next_step"])
    else:
        st.write("Method: Pre–Post Analysis")
        st.write(f"Observed Change: {number(result['absolute_change'], 2)} ({percent(result['percent_change'], 1)})")
        st.warning("Pre–Post results are descriptive and should not be interpreted as causal impact.")


def geo_map_frame(data: pd.DataFrame, assignment: pd.DataFrame | None = None) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=["dma", "lat", "lon", "status", "historical_outcome"])
    latest = data.groupby("_dma")["_outcome"].mean().reset_index(name="historical_outcome")
    latest["status"] = "Not assigned"
    if assignment is not None and not assignment.empty:
        group_map = assignment.set_index("DMA")["Group"].to_dict()
        latest["status"] = latest["_dma"].map(group_map).fillna("Not assigned")
    latest["lat"] = latest["_dma"].map(lambda dma: DMA_CENTROIDS.get(dma, (None, None))[0])
    latest["lon"] = latest["_dma"].map(lambda dma: DMA_CENTROIDS.get(dma, (None, None))[1])
    return latest.rename(columns={"_dma": "dma"})


def geo_data_step() -> None:
    st.subheader("Data")
    cfg: GeoConfig = st.session_state.geo_config
    cfg.data_source = st.radio("Data Source", ["Upload File", "Internal Data", "Synthetic Demo"], index=["Upload File", "Internal Data", "Synthetic Demo"].index(cfg.data_source), horizontal=True, key="geo_source")
    if cfg.data_source == "Upload File":
        uploaded = st.file_uploader("Upload DMA x Time CSV or Parquet", type=["csv", "parquet"], key="geo_upload")
        if uploaded is not None:
            try:
                signature = hash(uploaded.getvalue())
                if st.session_state.get("geo_upload_signature") != signature:
                    uploaded.seek(0)
                    st.session_state.geo_raw_df = normalize_uploaded_dataset(pd.read_parquet(uploaded) if uploaded.name.endswith(".parquet") else load_uploaded_csv(uploaded))
                    st.session_state.geo_upload_signature = signature
                    st.session_state.geo_assignment = pd.DataFrame()
                    st.session_state.geo_analysis = None
                    st.session_state.geo_plan = None
                cfg.dataset_name = uploaded.name
            except Exception as exc:
                st.error("Data validation issue: We could not read the uploaded file. Confirm it is a valid CSV or Parquet file.")
                print(f"Geographic upload failed: {exc}")
    elif cfg.data_source == "Synthetic Demo":
        st.session_state.geo_raw_df = load_geo_demo()
        cfg.dataset_name = "Synthetic DMA Campaign Panel"
    else:
        st.info("Internal geographic data access is a placeholder until the Databricks connector is added.")
        st.session_state.geo_raw_df = load_geo_demo()
        cfg.dataset_name = "Synthetic DMA Campaign Panel"

    raw = st.session_state.geo_raw_df
    if raw.empty:
        st.info("Upload a DMA x time panel with date, DMA, and outcome columns.")
        return
    inferred = infer_geo_schema(raw)
    numeric = numeric_columns(raw)
    columns = list(raw.columns)
    schema_signature = (cfg.dataset_name, tuple(columns))
    if st.session_state.get("geo_schema_signature") != schema_signature:
        cfg.date_column = inferred["date"]
        cfg.dma_column = inferred["dma"]
        cfg.outcome_column = inferred["outcome"]
        cfg.market_size_column = inferred["market_size"]
        cfg.spend_column = inferred["spend"]
        cfg.group_column = inferred["group"]
        cfg.pair_column = inferred["pair"]
        cfg.campaign_start_date = "2026-09-06" if cfg.data_source == "Synthetic Demo" else ""
        cfg.planned_launch_date = ""
        st.session_state.geo_schema_signature = schema_signature
    cfg.date_column = cfg.date_column if cfg.date_column in columns else inferred["date"]
    cfg.dma_column = cfg.dma_column if cfg.dma_column in columns else inferred["dma"]
    cfg.outcome_column = cfg.outcome_column if cfg.outcome_column in numeric else inferred["outcome"]
    cfg.market_size_column = cfg.market_size_column if cfg.market_size_column in ["None"] + columns else inferred["market_size"]
    cfg.spend_column = cfg.spend_column if cfg.spend_column in ["None"] + columns else inferred["spend"]
    cfg.group_column = cfg.group_column if cfg.group_column in columns else inferred["group"]
    cfg.pair_column = cfg.pair_column if cfg.pair_column in ["None"] + columns else inferred["pair"]

    st.markdown("**Data Mapping**")
    c1, c2, c3 = st.columns(3)
    cfg.date_column = c1.selectbox("Date Column", columns, index=columns.index(cfg.date_column))
    cfg.dma_column = c2.selectbox("DMA Column", columns, index=columns.index(cfg.dma_column))
    cfg.outcome_column = c3.selectbox("Primary Outcome", numeric, index=numeric.index(cfg.outcome_column) if cfg.outcome_column in numeric else 0)
    c4, c5, c6 = st.columns(3)
    cfg.group_column = c4.selectbox("Test / Control Group", ["None"] + columns, index=(["None"] + columns).index(cfg.group_column))
    cfg.market_size_column = c5.selectbox("Market-size Weight (optional)", ["None"] + columns, index=(["None"] + columns).index(cfg.market_size_column))
    cfg.pair_column = c6.selectbox("Pair ID (optional)", ["None"] + columns, index=(["None"] + columns).index(cfg.pair_column), help="Use only when pair IDs were defined before launch. The platform will not create pairs.")
    cfg.spend_column = st.selectbox("Campaign Spend / Exposure (optional)", ["None"] + columns, index=(["None"] + columns).index(cfg.spend_column))
    try:
        panel = prepare_geo_panel(raw, cfg.date_column, cfg.dma_column, cfg.outcome_column)
        st.session_state.geo_panel_df = panel
        quality = validate_geo_panel(panel)
        min_date, max_date = panel["_date"].min().date(), panel["_date"].max().date()
        st.success(f"{quality['period_count']:,.0f} periods | {quality['dma_count']:,.0f} DMAs | {min_date} – {max_date} | Detected frequency: {quality['frequency']}")
        if quality["duplicates"]:
            st.warning(f"{quality['duplicates']:,.0f} duplicate DMA/date records were found. Resolve duplicates before finalizing a design.")
        if quality["missing_periods"]:
            st.warning(f"{quality['missing_periods']:,.0f} time periods appear missing from the panel.")
        if not cfg.campaign_start_date:
            cfg.campaign_start_date = panel["_date"].quantile(0.75).date().isoformat()
        if not cfg.planned_launch_date:
            step = pd.DateOffset(months=1) if quality["frequency"] == "Monthly" else pd.Timedelta(weeks=1) if quality["frequency"] == "Weekly" else pd.Timedelta(days=1)
            cfg.planned_launch_date = (panel["_date"].max() + step).date().isoformat()
        validation = validate_fixed_assignment(panel, cfg.group_column, cfg.pair_column, cfg.test_label, cfg.control_label)
        st.session_state.geo_assignment = validation["assignment"] if not validation["errors"] else pd.DataFrame()
        for issue in validation["errors"]:
            st.error(issue)
        st.dataframe(raw.head(20), use_container_width=True)
    except Exception as exc:
        st.error("Data validation issue: We could not interpret the selected Date, DMA, and Outcome columns.")
        print(f"Geographic preparation failed: {exc}")


def geo_assignment_step() -> None:
    st.subheader("A/B Assignment")
    cfg: GeoConfig = st.session_state.geo_config
    panel = st.session_state.geo_panel_df
    if panel.empty:
        st.info("Upload and map a DMA panel first.")
        return
    if cfg.group_column == "None":
        st.info("Map the Test / Control Group column on the Data tab.")
        return
    c1, c2 = st.columns(2)
    cfg.test_label = c1.text_input("Source value meaning Test", value=cfg.test_label)
    cfg.control_label = c2.text_input("Source value meaning Control", value=cfg.control_label)
    validation = validate_fixed_assignment(panel, cfg.group_column, cfg.pair_column, cfg.test_label, cfg.control_label)
    for issue in validation["errors"]:
        st.error(issue)
    for warning in validation["warnings"]:
        st.warning(warning)
    if validation["errors"]:
        st.session_state.geo_assignment = pd.DataFrame()
        return
    assignment = validation["assignment"]
    st.session_state.geo_assignment = assignment
    counts = validation["counts"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Test DMAs", counts.get("Test", 0))
    m2.metric("Control DMAs", counts.get("Control", 0))
    m3.metric("Total DMAs", len(assignment))
    st.success("Fixed assignment is valid. The platform will use these groups as provided and will not rematch or reassign DMAs.")
    left, right = st.columns([1.25, 0.75])
    with left:
        st.plotly_chart(geo_dma_map(geo_map_frame(panel, assignment), "Provided Test and Control Assignment"), use_container_width=True)
    with right:
        st.dataframe(assignment, use_container_width=True, hide_index=True, height=390)
    missing_geo = [dma for dma in assignment["DMA"] if dma not in DMA_CENTROIDS]
    if missing_geo:
        st.caption(f"{len(missing_geo)} DMA(s) have no stored map centroid and are omitted from the map; they remain in all calculations.")
    balance_date = cfg.planned_launch_date or panel["_date"].max().date().isoformat()
    balance = evaluate_geo_balance(panel, assignment, balance_date)
    st.session_state.geo_balance = balance
    if balance:
        st.markdown("**Pre-Launch Comparability**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Historical Correlation", number(balance["correlation"], 2))
        m2.metric("Mean Relative Difference", percent(balance["relative_difference"], 1))
        m3.metric("Trend Difference", percent(balance["trend_difference"], 1))
        m4.metric("Overall Balance", balance["status"])
        st.plotly_chart(geo_balance_chart(balance["aggregate"], balance_date, cfg.outcome_column), use_container_width=True)
        st.caption("These diagnostics describe comparability; they do not change the provided assignment.")


def geo_planning_step() -> None:
    st.subheader("Plan Rollout")
    cfg: GeoConfig = st.session_state.geo_config
    panel = st.session_state.geo_panel_df
    assignment = st.session_state.geo_assignment
    if panel.empty or assignment.empty:
        st.info("Map and validate the fixed A/B assignment before planning rollout.")
        return
    quality = validate_geo_panel(panel)
    frequency = str(quality["frequency"])
    st.caption("Supply a decision-relevant effect scenario. The planner evaluates detectability; it does not predict an unknown campaign effect.")
    c1, c2, c3 = st.columns(3)
    cfg.planned_launch_date = c1.date_input("Planned Launch Date", value=pd.Timestamp(cfg.planned_launch_date).date(), min_value=panel["_date"].min().date()).isoformat()
    cfg.effect_type = c2.radio("Effect Scenario", ["Relative lift", "Absolute change"], index=["Relative lift", "Absolute change"].index(cfg.effect_type), horizontal=True)
    effect_label = "Effect to Detect (%)" if cfg.effect_type == "Relative lift" else f"Effect to Detect ({cfg.outcome_column})"
    cfg.expected_effect = c3.number_input(effect_label, min_value=0.01, value=float(cfg.expected_effect), step=0.5 if cfg.effect_type == "Relative lift" else 1.0)
    c4, c5, c6 = st.columns(3)
    cfg.target_power = c4.slider("Target Detection Chance", min_value=0.60, max_value=0.95, value=float(cfg.target_power), step=0.05)
    cfg.alpha = c5.select_slider("False-positive Rate", options=[0.01, 0.025, 0.05, 0.10], value=float(cfg.alpha))
    cfg.max_duration = c6.number_input(f"Longest Campaign Duration ({duration_unit(frequency)}s)", min_value=2, max_value=104, value=int(cfg.max_duration), step=1)
    c7, c8, c9 = st.columns(3)
    cfg.ramp_periods = c7.number_input("Ramp-up Periods", min_value=0, max_value=24, value=int(cfg.ramp_periods), step=1)
    cfg.outcome_delay_periods = c8.number_input("Outcome Maturation Delay", min_value=0, max_value=52, value=int(cfg.outcome_delay_periods), step=1)
    cfg.estimand = c9.radio("DMA Weighting", ["Equal weight per DMA", "Market-size weighted"], index=["Equal weight per DMA", "Market-size weighted"].index(cfg.estimand), horizontal=True)
    weighted = cfg.estimand == "Market-size weighted"
    if weighted and cfg.market_size_column == "None":
        st.error("Select a Market-size Weight column on the Data tab or use equal DMA weighting.")
        return
    try:
        plan = plan_geo_test(
            panel,
            assignment,
            cfg.planned_launch_date,
            frequency,
            cfg.effect_type,
            cfg.expected_effect,
            cfg.alpha,
            cfg.target_power,
            int(cfg.max_duration),
            int(cfg.ramp_periods),
            int(cfg.outcome_delay_periods),
            weighted,
            cfg.market_size_column,
        )
        st.session_state.geo_plan = plan
        planning_panel = prepare_geo_analysis(panel, assignment, cfg.planned_launch_date, weight_col=cfg.market_size_column)
        st.plotly_chart(geo_trend_chart(aggregate_trend(planning_panel, "Indexed"), cfg.planned_launch_date, "Indexed Outcome"), use_container_width=True)
        recommended = plan["recommended_duration"]
        row = plan["table"].loc[plan["table"]["Campaign Duration"] == recommended].iloc[0] if recommended is not None else plan["table"].iloc[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current DMA Sample", f"{plan['current_total_dmas']} total", f"{plan['current_test_dmas']} Test / {plan['current_control_dmas']} Control")
        m2.metric("Recommended Duration", duration_label(int(recommended), frequency) if recommended is not None else "Not reached")
        m3.metric("DMAs Required", int(row["Required Total DMAs"]), f"{int(row['Required Test DMAs'])} Test / {int(row['Required Control DMAs'])} Control")
        m4.metric("Earliest Readout", str(row["Earliest Readout"]))
        display = plan["table"].copy()
        display["Campaign Duration"] = display["Campaign Duration"].apply(lambda value: duration_label(int(value), frequency))
        display["Detection Chance"] = display["Detection Chance"].apply(lambda value: percent(value, 0))
        display["Minimum Detectable Effect"] = display.apply(lambda row: percent(row["MDE Relative"], 1) if cfg.effect_type == "Relative lift" else number(row["Minimum Detectable Effect"], 2), axis=1)
        display = display.drop(columns=["MDE Relative"])
        st.markdown("**Duration and DMA Requirements**")
        st.dataframe(display, use_container_width=True, hide_index=True)
        if recommended is None:
            st.warning("The selected effect scenario does not reach the target detection chance within the maximum duration. Add DMAs, accept a larger detectable effect, or extend the search window.")
        else:
            st.success(f"The shortest evaluated option meeting the target is {duration_label(int(recommended), frequency)}.")
        with st.expander("How this planning calculation works"):
            st.write(plan["method_note"])
            st.write("Historical Test-minus-Control variation is detrended, adjusted for lag-1 autocorrelation, and converted into the uncertainty of an average campaign effect. Detection chance uses a two-sided normal test. Required DMAs scale with squared uncertainty relative to the supplied effect scenario; duration and DMA count are separate design levers.")
            st.write(f"Historical periods used: {plan['historical_periods']} | Estimated lag-1 autocorrelation: {number(plan['lag1_autocorrelation'], 2)}")
    except Exception as exc:
        st.session_state.geo_plan = None
        st.error(str(exc))


def geo_analysis_step() -> None:
    st.subheader("Analyze Results")
    cfg: GeoConfig = st.session_state.geo_config
    panel = st.session_state.geo_panel_df
    assignment = st.session_state.geo_assignment
    if panel.empty or assignment.empty:
        st.info("Create or map a fixed assignment before running geographic analysis.")
        return
    min_date, max_date = panel["_date"].min().date(), panel["_date"].max().date()
    c1, c2, c3 = st.columns(3)
    cfg.campaign_start_date = c1.date_input("Actual Campaign Start", value=pd.Timestamp(cfg.campaign_start_date).date(), min_value=min_date, max_value=max_date).isoformat()
    end_enabled = c2.checkbox("Set Analysis End Date", value=bool(cfg.campaign_end_date), key="geo_actual_end_enabled")
    cfg.campaign_end_date = c2.date_input("Analysis End", value=pd.Timestamp(cfg.campaign_end_date or max_date).date(), min_value=pd.Timestamp(cfg.campaign_start_date).date(), max_value=max_date).isoformat() if end_enabled else ""
    cfg.outcome_view = c3.radio("Outcome View", ["Indexed", "Absolute"], index=["Indexed", "Absolute"].index(cfg.outcome_view), horizontal=True)
    weighted = cfg.estimand == "Market-size weighted"
    if weighted and cfg.market_size_column == "None":
        st.error("Select a Market-size Weight column on Data or change DMA Weighting in Plan Rollout.")
        return
    signature = (cfg.campaign_start_date, cfg.campaign_end_date, cfg.outcome_column, cfg.estimand, tuple(zip(assignment["DMA"], assignment["Group"])))
    try:
        analysis_panel = prepare_geo_analysis(panel, assignment, cfg.campaign_start_date, cfg.campaign_end_date, cfg.market_size_column)
        st.session_state.geo_balance = evaluate_geo_balance(panel, assignment, cfg.campaign_start_date)
        trend = aggregate_trend(analysis_panel, cfg.outcome_view)
        st.plotly_chart(geo_trend_chart(trend, cfg.campaign_start_date, "Indexed Outcome" if cfg.outcome_view == "Indexed" else cfg.outcome_column), use_container_width=True)
        if st.button("Run Geographic Analysis", type="primary"):
            result = run_panel_did(analysis_panel, weighted=weighted)
            result["_signature"] = signature
            st.session_state.geo_analysis = result
            st.success("Panel Difference-in-Differences analysis completed.")
        elif st.session_state.geo_analysis and st.session_state.geo_analysis.get("_signature") != signature:
            st.warning("Results are out of date for the current assignment, dates, outcome, or weighting. Run the analysis again.")
    except Exception as exc:
        st.error("Geographic analysis could not run with the current assignment and campaign window.")
        print(f"Geographic analysis failed: {exc}")

    result = st.session_state.geo_analysis
    if result and result.get("_signature") == signature:
        low, high = result["interval"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Estimated Relative Lift", percent(result["lift"], 1))
        m2.metric("Effect per DMA / Period", number(result["effect_per_period"], 2))
        m3.metric("95% Interval", f"{number(low, 2)} to {number(high, 2)}")
        m4.metric("P-value", p_value(result["pvalue"]))
        m5.metric("Model Reliability", result["reliability"])
        st.write(f"Model: {result['model']} | Estimand: {result['estimand']} | {result['test_dmas']} Test and {result['control_dmas']} Control DMA clusters")
        st.write(f"Projected incremental outcome across Test DMAs and {result['post_periods']} post periods: {number(result['incremental_volume'], 0)}")
        if pd.notna(result["pretrend_pvalue"]) and result["pretrend_pvalue"] < 0.05:
            st.warning("Pre-campaign trends differ statistically between Test and Control DMAs. The parallel-trends assumption needs review.")
        else:
            st.success("No statistically clear differential pre-trend was detected.")
        with st.expander("Robustness checks"):
            if st.button("Run Placebo and Leave-One-DMA-Out Checks"):
                result["placebo"] = run_placebo_tests(analysis_panel, weighted=weighted)
                result["leave_one_out"] = leave_one_dma_out(analysis_panel, weighted=weighted)
            if "placebo" in result:
                st.write(f"Historical placebo stability: {result['placebo']['status']}")
                sensitivity = result.get("leave_one_out", pd.DataFrame()).copy()
                if not sensitivity.empty:
                    sensitivity["Relative Lift"] = sensitivity["Relative Lift"].apply(lambda value: percent(value, 1))
                    st.dataframe(sensitivity, use_container_width=True, hide_index=True)
        st.info("The geographic market is the experimental unit. Results should not be interpreted as if underlying customers were independently randomized.")


def geo_decision_step() -> None:
    st.subheader("Decision")
    cfg: GeoConfig = st.session_state.geo_config
    result = st.session_state.geo_analysis
    balance = st.session_state.geo_balance
    if not result:
        st.info("Run geographic analysis before generating decision support.")
        return
    assignment = st.session_state.geo_assignment
    signature = (cfg.campaign_start_date, cfg.campaign_end_date, cfg.outcome_column, cfg.estimand, tuple(zip(assignment["DMA"], assignment["Group"])))
    if result.get("_signature") != signature:
        st.warning("The saved result is out of date. Run Geographic Analysis with the current configuration before making a decision.")
        return
    low, high = result["interval"]
    st.markdown("**Geographic Test Decision Support**")
    st.write(f"During the configured campaign period, Test DMAs generated an estimated {percent(result['lift'], 1)} incremental change in {cfg.outcome_column} relative to the provided Control group.")
    st.write(f"Estimated incremental volume is approximately {number(result['incremental_volume'], 0)}.")
    st.write(f"Pre-period balance: {balance.get('status', 'Review') if balance else 'Review'}")
    if pd.notna(low) and low <= 0 <= high:
        st.warning("The observed lift is directionally positive or negative, but the interval includes zero. Interpret the campaign impact cautiously.")
    elif result["lift"] > 0:
        st.success("The fixed geographic design shows positive incremental impact under the configured Difference-in-Differences assumptions.")
    else:
        st.warning("The fixed geographic design does not show positive incremental impact under the configured assumptions.")
    st.info("Keep the provided Test and Control assignment fixed after launch. Review parallel trends, contamination, spillovers, and major DMA-specific events before making a rollout decision.")


def geographic_page() -> None:
    st.title("Geographic Test")
    st.caption("Plan and analyze a market-level experiment using fixed Test and Control DMA groups.")
    tabs = st.tabs(["Data", "A/B Assignment", "Plan Rollout", "Analyze Results", "Decision"])
    with tabs[0]:
        geo_data_step()
    with tabs[1]:
        geo_assignment_step()
    with tabs[2]:
        geo_planning_step()
    with tabs[3]:
        geo_analysis_step()
    with tabs[4]:
        geo_decision_step()


def time_series_page() -> None:
    st.title("Time Series Analysis")
    st.caption("Evaluate an intervention without a randomized control group.")
    intent = st.session_state.get("ts_intent")
    if not intent:
        st.subheader("What would you like to do?")
        left, right = st.columns(2)
        with left:
            st.markdown("**Plan a Campaign**")
            st.write("Determine how long a campaign needs to run and whether the expected impact will be statistically detectable.")
            if st.button("Start Planning", type="primary", key="ts_start_planning"):
                st.session_state.ts_intent = "plan"
                st.rerun()
        with right:
            st.markdown("**Analyze Campaign Results**")
            st.write("Estimate campaign impact using observed pre- and post-campaign data.")
            if st.button("Start Analysis", key="ts_start_analysis"):
                st.session_state.ts_intent = "analyze"
                st.rerun()
        return
    nav_left, nav_right = st.columns([3, 1])
    nav_left.caption(f"Time Series / {'Plan Campaign' if intent == 'plan' else 'Analyze Results'}")
    if nav_right.button("Switch workflow", key="ts_switch_workflow"):
        st.session_state.ts_intent = None
        st.rerun()
    if intent == "plan":
        tabs = st.tabs(["Data", "Campaign Setup", "Detectability & Duration"])
        with tabs[0]:
            ts_data_step()
        with tabs[1]:
            ts_plan_setup_step()
        with tabs[2]:
            ts_planning_step()
        return
    tabs = st.tabs(["Data", "Setup", "Analysis", "Results", "Decision"])
    with tabs[0]:
        ts_data_step()
    with tabs[1]:
        ts_setup_step()
    with tabs[2]:
        ts_analysis_step()
    with tabs[3]:
        ts_results_step()
    with tabs[4]:
        ts_decision_step()


def customer_page() -> None:
    st.title("Customer-Level Experiment")
    steps = ["Data", "Metrics", "Treatments", "Design", "Analysis", "Decision"]
    completed = {
        "Data": not st.session_state.raw_df.empty,
        "Metrics": not st.session_state.historical_df.empty,
        "Treatments": bool(configured_arms(st.session_state.strategy_config)[1]),
        "Design": st.session_state.design_config.required_n_per_arm > 0,
        "Analysis": bool(st.session_state.analysis_results_by_role),
        "Decision": bool(st.session_state.analysis_results_by_role),
    }
    with st.sidebar:
        completed_count = sum(completed.values())
        st.caption(f"CUSTOMER WORKFLOW · {completed_count}/{len(steps)} READY")
        st.progress(completed_count / len(steps))
    tab_data, tab_metrics, tab_treatments, tab_design, tab_analysis, tab_decision = st.tabs(["Data", "Metrics", "Treatments", "Design", "Analyze", "Decide"])
    with tab_data:
        customer_data_step()
    with tab_metrics:
        customer_metrics_step()
    with tab_treatments:
        strategy_step()
    with tab_design:
        design_step()
    with tab_analysis:
        analysis_step()
    with tab_decision:
        decision_step()


init_state()
with st.sidebar:
    st.title("Experiment Platform")
    page = st.radio("Overview", ["Overview", "Customer", "Geography", "Time Series"], label_visibility="collapsed")

if page == "Overview":
    show_overview()
elif page == "Customer":
    customer_page()
elif page == "Geography":
    geographic_page()
else:
    time_series_page()
