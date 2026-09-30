from __future__ import annotations

import html
import math
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import asdict, fields
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
from src.experiment_platform.charts import bsts_counterfactual_chart, cumulative_impact_chart, customer_allocation_health_chart, customer_category_bar, customer_decision_map, customer_design_tradeoff_chart, customer_group_summary_chart, customer_history_coverage, customer_metric_distribution, customer_option_chart, customer_required_accounts_chart, customer_rollout_impact_chart, customer_strategy_outcome_chart, customer_traffic_allocation_chart, detectable_effect_curve, forest_plot, geo_balance_chart, geo_dma_map, geo_trend_chart, historical_association, impact_chart, portfolio_value_chart, pre_post_bar, response_plot, time_series_line, timeseries_chart_context
from src.experiment_platform.customer_planning import assign_group_labels, build_customer_preview, calculate_group_design, calculate_variant_group_design, default_numeric_groups, planning_comparison_count, suggest_group_designs
from src.experiment_platform.customer_templates import apply_customer_template
from src.experiment_platform.data_access import CUSTOMER_DEMO_SCENARIOS, load_customer_demo, load_geo_demo, load_raw_demo, load_results_demo, load_timeseries_demo, load_timeseries_planning_demo, load_uploaded_csv
from src.experiment_platform.data_validation import data_quality_warnings, date_like_columns, first_series, infer_column, infer_mob_column, normalize_uploaded_dataset, safe_numeric_series
from src.experiment_platform.decision import arm_decision_scorecard, experiment_recommendation, guardrail_status, project_rollout_impact, recommendation_status
from src.experiment_platform.formatting import money, number, p_value, percent
from src.experiment_platform.geography import DMA_CENTROIDS, GeoConfig, aggregate_trend, build_geo_preview, evaluate_geo_balance, infer_geo_schema, leave_one_dma_out, plan_geo_test, prepare_geo_analysis, prepare_geo_panel, run_panel_did, run_placebo_tests, validate_fixed_assignment, validate_geo_panel
from src.experiment_platform.historical_strategy import fixed_unit_values, historical_arm_statistics, validate_fixed_unit_value
from src.experiment_platform.metrics import column_schema, metric_baseline, suggest_metric_type, validate_metric
from src.experiment_platform.models import AnalysisConfig, DataConfig, DataMappingConfig, DesignConfig, MetricConfig, PopulationConfig, StrategyConfig
from src.experiment_platform.power import adjusted_alpha, detectable_effect_binary, detectable_effect_continuous, detectable_effect_continuous_unequal, duration, power_binary, power_continuous, power_continuous_unequal, sample_size_binary, sample_size_continuous
from src.experiment_platform.portfolio import PORTFOLIO_STATUSES, build_rollout_plan, business_report_html, duplicate_experiment, freeze_decision_snapshot, load_experiments, portfolio_summary, pulse_summary, save_configuration, upsert_experiment, value_timeline
from src.experiment_platform.raw_processing import build_analysis_dataset
from src.experiment_platform.timeseries import TimeSeriesConfig, build_business_interpretation, build_timeseries_preview, campaign_decision_date, classify_calibration, config_fingerprint, data_fingerprint, detect_frequency, duplicate_timestamp_count, duration_power_status, fit_bsts_model, humanize_column_name, incremental_outcome_label, infer_timeseries_columns, method_config_fingerprint, planning_config_fingerprint, planning_durations, prepare_timeseries_data, predictor_candidates, probability_positive_label, projected_operational_exposure, run_duration_power_simulation, run_pre_post_analysis


st.set_page_config(page_title="Experiment Platform", layout="wide")

st.markdown(
    """
    <style>
    :root {--coral:#ef4d4d; --ink:#232936; --muted:#697386; --line:#dfe4ec; --soft:#f6f8fb; --teal:#16856b; --gold:#d69a1f;}
    .stApp {background:#ffffff; color:var(--ink);}
    .block-container {padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1320px;}
    [data-testid="stSidebar"] {background:#f6f8fb; border-right:1px solid #e5e9f0;}
    [data-testid="stSidebar"] .block-container {padding-top:1.4rem;}
    h1 {font-size:1.7rem; line-height:1.22; letter-spacing:0; color:var(--ink);}
    h2 {font-size:1.25rem; letter-spacing:0; color:var(--ink);}
    h3 {font-size:1.08rem; letter-spacing:0; color:var(--ink); margin-top:.3rem;}
    p, label, [data-testid="stCaptionContainer"] {color:#4f596b;}
    [data-testid="stMetric"] {border:1px solid var(--line); border-top:3px solid #91a2bb; padding:.75rem .85rem; border-radius:6px; background:#fff; box-shadow:0 1px 2px rgba(24,35,52,.04);}
    [data-testid="stMetricValue"] {font-size:1.35rem; color:var(--ink);}
    div[data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:6px; overflow:hidden;}
    .stButton > button {border-radius:5px; border-color:#cfd6e1; transition:border-color .15s ease, box-shadow .15s ease, transform .15s ease;}
    .stButton > button:hover {border-color:var(--coral); color:#c93636; box-shadow:0 2px 7px rgba(239,77,77,.12); transform:translateY(-1px);}
    .stButton > button[kind="primary"] {background:var(--coral); border-color:var(--coral); color:white;}
    .stButton > button[kind="primary"] p {color:white;}
    div[data-baseweb="tab-list"] {gap:.35rem; padding:0 .55rem; border:1px solid var(--line); border-radius:7px; background:#f8f9fb; margin:.15rem 0 1rem;}
    button[data-baseweb="tab"] {padding:.72rem .65rem; color:#4c5565;}
    button[data-baseweb="tab"][aria-selected="true"] {color:var(--coral); font-weight:650;}
    [data-baseweb="input"], [data-baseweb="select"] > div {background:#f8f9fc; border-color:#dce2eb; border-radius:5px;}
    [data-testid="stExpander"] {border-color:var(--line); border-radius:6px; background:#fff;}
    [data-testid="stAlert"] {border-radius:6px;}
    [data-testid="stProgressBar"] > div > div {background:var(--coral);}
    .subtle-note {border-left:3px solid #7891b3; padding:.55rem .75rem; background:#f7f9fc; color:#2f3b4a;}
    .step-line {font-size: .92rem; line-height: 1.9;}
    .experiment-summary {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border:1px solid var(--line); border-radius:6px; margin:0 0 1.2rem; background:#fff;}
    .summary-item {padding:.72rem .9rem; min-width:0;}
    .summary-item + .summary-item {border-left:1px solid var(--line);}
    .summary-label {font-size:.7rem; color:#7a8495; text-transform:uppercase; font-weight:700; margin-bottom:.18rem;}
    .summary-value {font-size:.88rem; color:var(--ink); font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
    .strategy-banner {border-left:4px solid var(--coral); background:#fff8f7; padding:.75rem .9rem; margin:.35rem 0 1rem;}
    .strategy-banner strong {display:block; color:var(--ink); font-size:.92rem; margin-bottom:.15rem;}
    .strategy-banner span {color:#626c7d; font-size:.82rem;}
    .arm-overview {display:flex; gap:.5rem; flex-wrap:wrap; margin:.35rem 0 .9rem;}
    .arm-chip {border:1px solid var(--line); border-left:4px solid var(--gold); padding:.38rem .65rem; border-radius:4px; background:#fff; font-size:.8rem; color:var(--ink);}
    .arm-chip.control {border-left-color:#2457a6; background:#f4f7fc;}
    .arm-chip b {font-weight:700; margin-right:.35rem;}
    .next-step {border-top:1px solid var(--line); margin-top:1rem; padding-top:.75rem; color:#4e5869; font-size:.84rem;}
    .business-guide {display:flex; justify-content:space-between; gap:1rem; align-items:center; border-left:4px solid #91a2bb; padding:.62rem .8rem; margin:.05rem 0 1rem; background:#f7f9fc; color:#596476; font-size:.8rem; line-height:1.45;}
    .business-guide b {flex:0 0 auto; color:var(--ink); font-size:.74rem; font-weight:750;}
    .overview-hero {position:relative; overflow:hidden; border:1px solid #dce2eb; border-left:5px solid var(--coral); padding:1.25rem 1.4rem 1.15rem; margin:.2rem 0 1.35rem; background:#fff; box-shadow:0 8px 24px rgba(35,41,54,.06);}
    .overview-hero:after {content:""; position:absolute; right:0; top:0; width:32%; height:4px; background:var(--gold);}
    .overview-kicker {font-size:.7rem; text-transform:uppercase; font-weight:800; color:var(--coral); margin-bottom:.35rem;}
    .overview-hero h1 {font-size:2rem; margin:.05rem 0 .35rem; max-width:760px;}
    .overview-hero p {font-size:.94rem; max-width:760px; margin:0; color:#596476;}
    .overview-signal {display:flex; gap:1.2rem; flex-wrap:wrap; margin-top:.9rem; padding-top:.8rem; border-top:1px solid #edf0f4;}
    .overview-signal span {font-size:.75rem; color:#697386;}
    .overview-signal b {color:var(--ink); margin-right:.25rem;}
    .overview-section-title {margin:.35rem 0 .15rem; font-size:1.05rem; font-weight:750; color:var(--ink);}
    .overview-section-copy {margin:0 0 .85rem; font-size:.84rem; color:#6c7687;}
    .workflow-card {height:310px; border:1px solid var(--line); border-top:4px solid #91a2bb; border-radius:7px; padding:1rem 1rem .85rem; background:#fff; transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;}
    .workflow-card:hover {transform:translateY(-3px); box-shadow:0 10px 24px rgba(35,41,54,.09); border-color:#cdd5e1;}
    .workflow-card.customer {border-top-color:var(--coral);}
    .workflow-card.geo {border-top-color:var(--teal);}
    .workflow-card.time {border-top-color:var(--gold);}
    .workflow-index {font-size:.68rem; font-weight:800; color:#8a94a5; text-transform:uppercase;}
    .workflow-card h3 {font-size:1.05rem; margin:.38rem 0 .42rem;}
    .workflow-card p {font-size:.82rem; line-height:1.48; min-height:3.7rem; margin:0; color:#626d7e;}
    .workflow-fit {margin-top:.8rem; padding-top:.65rem; border-top:1px solid #edf0f4; font-size:.76rem; color:#697386;}
    .workflow-fit b {display:block; color:var(--ink); margin-bottom:.12rem;}
    .overview-flow {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); margin:1.45rem 0 .8rem; border:1px solid var(--line); border-radius:7px; background:#f8f9fb;}
    .flow-step {position:relative; padding:.85rem 1rem .8rem 2.8rem; min-height:74px;}
    .flow-step + .flow-step {border-left:1px solid var(--line);}
    .flow-number {position:absolute; left:.9rem; top:.9rem; display:grid; place-items:center; width:1.35rem; height:1.35rem; border-radius:50%; background:#fff; border:1px solid #cbd3df; color:var(--coral); font-size:.68rem; font-weight:800;}
    .flow-step b {display:block; font-size:.8rem; color:var(--ink); margin-bottom:.12rem;}
    .flow-step span {display:block; font-size:.72rem; line-height:1.35; color:#737d8e;}
    .customer-choice {min-height:145px; border:1px solid var(--line); border-top:4px solid var(--coral); border-radius:7px; padding:1rem; margin:.2rem 0 .55rem; background:#fff;}
    .customer-choice.analysis {border-top-color:var(--teal);}
    .customer-choice h3 {margin:0 0 .45rem; font-size:1.05rem;}
    .customer-choice p {margin:0; font-size:.84rem; line-height:1.5; color:#626d7e;}
    .customer-product-bar {display:flex; align-items:center; gap:.72rem; min-height:42px;}
    .customer-product-mark {display:grid; place-items:center; width:34px; height:34px; border:1px solid #f1b8b8; border-radius:6px; background:#fff1f1; color:#cf3737; font-size:.72rem; font-weight:850;}
    .customer-product-name {font-size:.9rem; font-weight:760; color:var(--ink);}
    .customer-product-context {font-size:.7rem; color:#7b8595; margin-top:.05rem;}
    .customer-page-heading {position:relative; padding:.75rem 0 1rem; border-bottom:1px solid var(--line); margin-bottom:.7rem;}
    .customer-page-heading:after {content:""; position:absolute; left:0; bottom:-1px; width:82px; height:3px; background:var(--coral);}
    .customer-page-heading h1 {font-size:2rem; margin:0 0 .28rem;}
    .customer-page-heading p {font-size:.9rem; margin:0; color:#647084;}
    .customer-demo-actions {display:flex; align-items:center; justify-content:space-between; gap:1rem; margin:.2rem 0 .45rem;}
    .customer-demo-actions strong {font-size:.86rem; color:var(--ink);}
    .customer-demo-actions span {display:block; margin-top:.08rem; font-size:.74rem; color:#737d8e;}
    .customer-preview-summary {min-height:402px; border:1px solid var(--line); border-top:4px solid var(--coral); border-radius:7px; background:#fff; overflow:hidden; box-shadow:0 7px 22px rgba(35,41,54,.06);}
    .customer-preview-summary h3 {font-size:.96rem; margin:0; padding:.9rem 1rem .72rem; border-bottom:1px solid #edf0f4;}
    .customer-preview-grid {display:grid; grid-template-columns:repeat(2,minmax(0,1fr));}
    .customer-preview-stat {min-height:94px; padding:.82rem 1rem; border-bottom:1px solid #edf0f4;}
    .customer-preview-stat:nth-child(even) {border-left:1px solid #edf0f4;}
    .customer-preview-label {font-size:.67rem; color:#7a8495; font-weight:750; text-transform:uppercase; margin-bottom:.28rem;}
    .customer-preview-value {font-size:1.02rem; line-height:1.25; color:var(--ink); font-weight:760; overflow-wrap:anywhere;}
    .customer-preview-detail {font-size:.7rem; color:#727d8f; margin-top:.2rem;}
    .customer-mix {padding:.72rem 1rem .55rem;}
    .customer-mix-title {display:flex; justify-content:space-between; gap:.5rem; font-size:.7rem; color:#5e697a; margin-bottom:.36rem;}
    .customer-mix-row {display:grid; grid-template-columns:74px 1fr 28px; gap:.5rem; align-items:center; margin:.32rem 0; font-size:.68rem; color:#687386;}
    .customer-mix-track {height:7px; background:#edf0f4; border-radius:3px; overflow:hidden;}
    .customer-mix-fill {height:100%; background:#2457a6;}
    .customer-mix-row:nth-child(3) .customer-mix-fill {background:#ef4d4d;}
    .customer-mix-row:nth-child(4) .customer-mix-fill {background:#d69a1f;}
    .customer-preview-note {margin:.25rem 1rem .85rem; padding:.6rem .72rem; border-left:3px solid var(--coral); background:#fff5f5; color:#5d6572; font-size:.7rem; line-height:1.42;}
    .customer-section-heading {display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; margin:1.15rem 0 .48rem;}
    .customer-section-heading strong {font-size:.96rem; color:var(--ink);}
    .customer-section-heading span {font-size:.72rem; color:#778194;}
    .st-key-customer_demo_plan button {background:var(--coral) !important; border-color:var(--coral) !important; color:#fff !important; font-weight:700;}
    .st-key-customer_demo_plan button p, .st-key-customer_demo_analyze button p {color:#fff !important;}
    .st-key-customer_demo_analyze button {background:var(--teal) !important; border-color:var(--teal) !important; color:#fff !important; font-weight:700;}
    .customer-choice-v2 {display:grid; grid-template-columns:56px 1fr; column-gap:1rem; min-height:156px; border:1px solid #efb5b5; border-radius:7px; padding:1.2rem 1.25rem; background:#fff8f8; box-shadow:0 7px 22px rgba(35,41,54,.06);}
    .customer-choice-v2.analysis {border-color:#a8d2ca; background:#f6fbfa;}
    .customer-choice-icon {grid-row:1 / span 3; display:grid; place-items:center; align-self:start; width:52px; height:52px; border-radius:6px; background:#ffdada; color:#bb2f2f; font-size:1rem; font-weight:850;}
    .customer-choice-v2.analysis .customer-choice-icon {background:#cfe8e2; color:#0d6a58;}
    .customer-choice-v2 h3 {font-size:1.18rem; margin:.05rem 0 .28rem;}
    .customer-choice-v2 p {font-size:.84rem; line-height:1.5; color:#5e697b; margin:0; max-width:520px;}
    .customer-choice-result {align-self:end; margin-top:.65rem; border-top:1px solid #eadede; padding-top:.55rem; color:#4f596b; font-size:.76rem; font-weight:650;}
    .customer-choice-v2.analysis .customer-choice-result {border-top-color:#dceae7;}
    .customer-entry-note {font-size:.75rem; color:#697386; text-align:center; margin:.65rem 0 0;}
    .customer-path {display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); border:1px solid var(--line); border-radius:7px; margin:.2rem 0 1rem; background:#f8f9fb;}
    .customer-path.analyze {grid-template-columns:repeat(2,minmax(0,1fr));}
    .customer-path-step {position:relative; min-height:62px; padding:.72rem .8rem .65rem 2.7rem;}
    .customer-path-step + .customer-path-step {border-left:1px solid var(--line);}
    .customer-path-number {position:absolute; left:.85rem; top:.8rem; display:grid; place-items:center; width:1.3rem; height:1.3rem; border-radius:50%; background:#fff; border:1px solid #cbd3df; color:var(--coral); font-size:.66rem; font-weight:800;}
    .customer-path.analyze .customer-path-number {color:var(--teal);}
    .customer-path-step b {display:block; font-size:.79rem; color:var(--ink); margin-bottom:.08rem;}
    .customer-path-step span {display:block; font-size:.7rem; color:#737d8e;}
    .customer-kpi-panel {border:1px solid var(--line); border-radius:7px; background:#fff; box-shadow:0 5px 18px rgba(35,41,54,.05); margin:.5rem 0 1rem; overflow:hidden;}
    .customer-kpi-head {display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; padding:.82rem 1rem .7rem; border-top:5px solid var(--coral); border-bottom:1px solid #edf0f4;}
    .customer-kpi-panel.analysis .customer-kpi-head {border-top-color:var(--teal);}
    .customer-kpi-head strong {display:block; font-size:.98rem; color:var(--ink); margin-bottom:.1rem;}
    .customer-kpi-head span {display:block; font-size:.76rem; color:#687386;}
    .customer-status {flex:0 0 auto; border:1px solid #efc0c0; background:#fff2f2; color:#b92f2f; padding:.28rem .5rem; border-radius:4px; font-size:.68rem; font-weight:750;}
    .customer-kpi-panel.analysis .customer-status {border-color:#b9ddd5; background:#edf8f5; color:#0b6b58;}
    .customer-kpis {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); padding:.25rem 0;}
    .customer-kpi {padding:.75rem 1rem .85rem; min-width:0;}
    .customer-kpi + .customer-kpi {border-left:1px solid #edf0f4;}
    .customer-kpi-label {font-size:.69rem; color:#758092; margin-bottom:.25rem;}
    .customer-kpi-value {font-size:1.34rem; line-height:1.15; font-weight:760; color:var(--ink); overflow-wrap:anywhere;}
    .customer-kpi-value.treatment {color:#dc4343;}
    .customer-kpi-value.analysis {color:#117865;}
    .st-key-customer_start_plan button {background:var(--coral) !important; border-color:var(--coral) !important; color:#fff !important;}
    .st-key-customer_start_plan button p {color:#fff !important;}
    .st-key-customer_start_analysis button {background:var(--teal); border-color:var(--teal); color:#fff;}
    .st-key-customer_start_analysis button p {color:#fff;}
    .ts-product-bar {display:flex; align-items:center; gap:.72rem; min-height:42px;}
    .ts-product-mark {display:grid; place-items:center; width:34px; height:34px; border:1px solid #ead28c; border-radius:6px; background:#fff7dc; color:#8d6505; font-size:.72rem; font-weight:850;}
    .ts-product-name {font-size:.9rem; font-weight:760; color:var(--ink);}
    .ts-product-context {font-size:.7rem; color:#7b8595; margin-top:.05rem;}
    .ts-page-heading {position:relative; padding:.75rem 0 1rem; border-bottom:1px solid var(--line); margin-bottom:.7rem;}
    .ts-page-heading:after {content:""; position:absolute; left:0; bottom:-1px; width:82px; height:3px; background:var(--gold);}
    .ts-page-heading h1 {font-size:2rem; margin:0 0 .28rem;}
    .ts-page-heading p {font-size:.9rem; margin:0; color:#647084;}
    .ts-demo-actions {display:flex; align-items:center; justify-content:space-between; gap:1rem; margin:.2rem 0 .45rem;}
    .ts-demo-actions strong {font-size:.86rem; color:var(--ink);}
    .ts-demo-actions span {display:block; margin-top:.08rem; font-size:.74rem; color:#737d8e;}
    .ts-preview-summary {min-height:422px; border:1px solid var(--line); border-top:4px solid var(--gold); border-radius:7px; background:#fff; overflow:hidden; box-shadow:0 7px 22px rgba(35,41,54,.06);}
    .ts-preview-summary h3 {font-size:.96rem; margin:0; padding:.9rem 1rem .72rem; border-bottom:1px solid #edf0f4;}
    .ts-preview-grid {display:grid; grid-template-columns:repeat(2,minmax(0,1fr));}
    .ts-preview-stat {min-height:94px; padding:.82rem 1rem; border-bottom:1px solid #edf0f4;}
    .ts-preview-stat:nth-child(even) {border-left:1px solid #edf0f4;}
    .ts-preview-label {font-size:.67rem; color:#7a8495; font-weight:750; text-transform:uppercase; margin-bottom:.28rem;}
    .ts-preview-value {font-size:1.02rem; line-height:1.25; color:var(--ink); font-weight:760; overflow-wrap:anywhere;}
    .ts-preview-detail {font-size:.7rem; color:#727d8f; margin-top:.2rem;}
    .ts-period-split {padding:.82rem 1rem .72rem;}
    .ts-period-row {display:flex; align-items:center; justify-content:space-between; gap:.6rem; font-size:.72rem; color:#5e697a; margin-bottom:.45rem;}
    .ts-period-track {display:flex; width:100%; height:9px; background:#edf0f4; overflow:hidden; border-radius:3px;}
    .ts-period-pre {height:100%; background:#2457a6;}
    .ts-period-post {height:100%; background:#ef4d4d;}
    .ts-period-legend {display:flex; justify-content:space-between; margin-top:.38rem; font-size:.68rem; color:#778194;}
    .ts-preview-note {margin:.25rem 1rem .9rem; padding:.65rem .72rem; border-left:3px solid var(--gold); background:#fffaf0; color:#5d6572; font-size:.72rem; line-height:1.42;}
    .ts-section-heading {display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; margin:1.15rem 0 .48rem;}
    .ts-section-heading strong {font-size:.96rem; color:var(--ink);}
    .ts-section-heading span {font-size:.72rem; color:#778194;}
    .st-key-ts_demo_plan button {background:var(--gold) !important; border-color:var(--gold) !important; color:#fff !important; font-weight:700;}
    .st-key-ts_demo_plan button p, .st-key-ts_demo_analyze button p {color:#fff !important;}
    .st-key-ts_demo_analyze button {background:var(--teal) !important; border-color:var(--teal) !important; color:#fff !important; font-weight:700;}
    .ts-data-preview-note {font-size:.74rem; color:#697386; margin:-.2rem 0 .5rem;}
    .ts-choice {display:grid; grid-template-columns:56px 1fr; column-gap:1rem; min-height:156px; border:1px solid #e5ca79; border-radius:7px; padding:1.2rem 1.25rem; background:#fffdf7; box-shadow:0 7px 22px rgba(35,41,54,.06);}
    .ts-choice.analysis {border-color:#a8d2ca; background:#f6fbfa;}
    .ts-choice-icon {grid-row:1 / span 3; display:grid; place-items:center; align-self:start; width:52px; height:52px; border-radius:6px; background:#f5df9d; color:#815b00; font-size:1rem; font-weight:850;}
    .ts-choice.analysis .ts-choice-icon {background:#cfe8e2; color:#0d6a58;}
    .ts-choice h3 {font-size:1.18rem; margin:.05rem 0 .28rem;}
    .ts-choice p {font-size:.84rem; line-height:1.5; color:#5e697b; margin:0; max-width:520px;}
    .ts-choice-result {align-self:end; margin-top:.65rem; border-top:1px solid #eadfbf; padding-top:.55rem; color:#4f596b; font-size:.76rem; font-weight:650;}
    .ts-choice.analysis .ts-choice-result {border-top-color:#dceae7;}
    .ts-entry-note {font-size:.75rem; color:#697386; text-align:center; margin:.65rem 0 0;}
    .ts-path {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); border:1px solid var(--line); border-radius:7px; margin:.2rem 0 1rem; background:#f8f9fb;}
    .ts-path.analyze {grid-template-columns:repeat(5,minmax(0,1fr));}
    .ts-path-step {position:relative; min-height:62px; padding:.72rem .8rem .65rem 2.7rem;}
    .ts-path-step + .ts-path-step {border-left:1px solid var(--line);}
    .ts-path-number {position:absolute; left:.85rem; top:.8rem; display:grid; place-items:center; width:1.3rem; height:1.3rem; border-radius:50%; background:#fff; border:1px solid #cbd3df; color:#a66f00; font-size:.66rem; font-weight:800;}
    .ts-path.analyze .ts-path-number {color:var(--teal);}
    .ts-path-step b {display:block; font-size:.79rem; color:var(--ink); margin-bottom:.08rem;}
    .ts-path-step span {display:block; font-size:.7rem; color:#737d8e;}
    .ts-kpi-panel {border:1px solid var(--line); border-radius:7px; background:#fff; box-shadow:0 5px 18px rgba(35,41,54,.05); margin:.5rem 0 1rem; overflow:hidden;}
    .ts-kpi-head {display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; padding:.82rem 1rem .7rem; border-top:5px solid var(--gold); border-bottom:1px solid #edf0f4;}
    .ts-kpi-panel.analysis .ts-kpi-head {border-top-color:var(--teal);}
    .ts-kpi-head strong {display:block; font-size:.98rem; color:var(--ink); margin-bottom:.1rem;}
    .ts-kpi-head span {display:block; font-size:.76rem; color:#687386;}
    .ts-status {flex:0 0 auto; border:1px solid #ead596; background:#fff8df; color:#8a6100; padding:.28rem .5rem; border-radius:4px; font-size:.68rem; font-weight:750;}
    .ts-kpi-panel.analysis .ts-status {border-color:#b9ddd5; background:#edf8f5; color:#0b6b58;}
    .ts-kpis {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); padding:.25rem 0;}
    .ts-kpi {padding:.75rem 1rem .85rem; min-width:0;}
    .ts-kpi + .ts-kpi {border-left:1px solid #edf0f4;}
    .ts-kpi-label {font-size:.69rem; color:#758092; margin-bottom:.25rem;}
    .ts-kpi-value {font-size:1.34rem; line-height:1.15; font-weight:760; color:var(--ink); overflow-wrap:anywhere;}
    .ts-kpi-value.positive {color:#117865;}
    .ts-kpi-value.negative {color:#dc4343;}
    .ts-panel-title {font-size:.86rem; font-weight:750; color:var(--ink); margin:.15rem 0 .5rem;}
    .st-key-ts_start_planning button {background:var(--gold) !important; border-color:var(--gold) !important; color:#fff !important;}
    .st-key-ts_start_planning button p {color:#fff !important;}
    .st-key-ts_start_analysis button {background:var(--teal); border-color:var(--teal); color:#fff;}
    .st-key-ts_start_analysis button p {color:#fff;}
    .geo-intro {position:relative; overflow:hidden; border:1px solid var(--line); border-left:5px solid var(--teal); padding:1rem 1.15rem; margin:.15rem 0 1.15rem; background:#fff; box-shadow:0 8px 24px rgba(35,41,54,.06);}
    .geo-intro:after {content:""; position:absolute; right:0; top:0; width:28%; height:4px; background:var(--gold);}
    .geo-intro strong {display:block; font-size:1rem; color:var(--ink); margin-bottom:.2rem;}
    .geo-intro span {display:block; max-width:780px; color:#657083; font-size:.84rem; line-height:1.5;}
    .geo-choice {min-height:168px; border:1px solid var(--line); border-top:5px solid var(--gold); border-radius:7px; padding:1.05rem 1.1rem .95rem; margin:.15rem 0 .55rem; background:#fff; box-shadow:0 5px 18px rgba(35,41,54,.05);}
    .geo-choice.analysis {border-top-color:var(--teal);}
    .geo-choice-number {font-size:.7rem; color:#8a94a5; font-weight:800; text-transform:uppercase; margin-bottom:.5rem;}
    .geo-choice h3 {font-size:1.12rem; margin:0 0 .42rem; color:var(--ink);}
    .geo-choice p {font-size:.84rem; line-height:1.5; color:#626d7e; margin:0 0 .65rem; max-width:510px;}
    .geo-choice-result {border-top:1px solid #edf0f4; padding-top:.55rem; color:#4f596b; font-size:.76rem; font-weight:650;}
    .geo-path {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); border:1px solid var(--line); border-radius:7px; margin:.2rem 0 1rem; background:#f8f9fb;}
    .geo-path-step {position:relative; min-height:62px; padding:.72rem .8rem .65rem 2.7rem;}
    .geo-path-step + .geo-path-step {border-left:1px solid var(--line);}
    .geo-path-number {position:absolute; left:.85rem; top:.8rem; display:grid; place-items:center; width:1.3rem; height:1.3rem; border-radius:50%; background:#fff; border:1px solid #cbd3df; color:var(--teal); font-size:.66rem; font-weight:800;}
    .geo-path.plan .geo-path-number {color:#a66f00;}
    .geo-path-step b {display:block; font-size:.79rem; color:var(--ink); margin-bottom:.08rem;}
    .geo-path-step span {display:block; font-size:.7rem; color:#737d8e;}
    .geo-result-banner {border:1px solid var(--line); border-left:5px solid var(--teal); padding:.82rem 1rem; margin:.4rem 0 1rem; background:#f4fbf8;}
    .geo-result-banner.plan {border-left-color:var(--gold); background:#fffaf0;}
    .geo-result-banner strong {display:block; color:var(--ink); font-size:.94rem; margin-bottom:.16rem;}
    .geo-result-banner span {font-size:.8rem; color:#5f697a;}
    .geo-section-copy {margin:-.15rem 0 .9rem; max-width:760px; color:#697386; font-size:.84rem; line-height:1.5;}
    .geo-product-bar {display:flex; align-items:center; gap:.72rem; min-height:42px;}
    .geo-product-mark {display:grid; place-items:center; width:34px; height:34px; border:1px solid #bfd8d1; border-radius:6px; background:#eef8f5; color:var(--teal); font-size:.78rem; font-weight:850;}
    .geo-product-name {font-size:.9rem; font-weight:760; color:var(--ink);}
    .geo-product-context {font-size:.7rem; color:#7b8595; margin-top:.05rem;}
    .geo-page-heading {position:relative; padding:.75rem 0 1rem; border-bottom:1px solid var(--line); margin-bottom:.7rem;}
    .geo-page-heading:after {content:""; position:absolute; left:0; bottom:-1px; width:96px; height:3px; background:var(--teal);}
    .geo-page-heading h1 {font-size:2rem; margin:0 0 .28rem;}
    .geo-page-heading p {font-size:.9rem; margin:0; color:#647084;}
    .geo-demo-actions {display:flex; align-items:center; justify-content:space-between; gap:1rem; margin:.2rem 0 .45rem;}
    .geo-demo-actions strong {font-size:.86rem; color:var(--ink);}
    .geo-demo-actions span {display:block; margin-top:.08rem; font-size:.74rem; color:#737d8e;}
    .geo-preview-summary {min-height:422px; border:1px solid var(--line); border-top:4px solid var(--teal); border-radius:7px; background:#fff; overflow:hidden; box-shadow:0 7px 22px rgba(35,41,54,.06);}
    .geo-preview-summary h3 {font-size:.96rem; margin:0; padding:.9rem 1rem .72rem; border-bottom:1px solid #edf0f4;}
    .geo-preview-grid {display:grid; grid-template-columns:repeat(2,minmax(0,1fr));}
    .geo-preview-stat {min-height:94px; padding:.82rem 1rem; border-bottom:1px solid #edf0f4;}
    .geo-preview-stat:nth-child(even) {border-left:1px solid #edf0f4;}
    .geo-preview-label {font-size:.67rem; color:#7a8495; font-weight:750; text-transform:uppercase; margin-bottom:.28rem;}
    .geo-preview-value {font-size:1.02rem; line-height:1.25; color:var(--ink); font-weight:760; overflow-wrap:anywhere;}
    .geo-preview-detail {font-size:.7rem; color:#727d8f; margin-top:.2rem;}
    .geo-assignment-bars {padding:.82rem 1rem .72rem;}
    .geo-assignment-row {display:grid; grid-template-columns:58px 1fr 24px; gap:.55rem; align-items:center; margin:.42rem 0; font-size:.72rem; color:#5e697a;}
    .geo-assignment-track {height:7px; background:#edf0f4; overflow:hidden; border-radius:3px;}
    .geo-assignment-fill {height:100%; background:#ef4d4d;}
    .geo-assignment-fill.control {background:#2457a6;}
    .geo-preview-note {margin:.25rem 1rem .9rem; padding:.65rem .72rem; border-left:3px solid var(--gold); background:#fffaf0; color:#5d6572; font-size:.72rem; line-height:1.42;}
    .geo-section-heading {display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; margin:1.15rem 0 .48rem;}
    .geo-section-heading strong {font-size:.96rem; color:var(--ink);}
    .geo-section-heading span {font-size:.72rem; color:#778194;}
    .st-key-geo_demo_plan button {background:#d69a1f !important; border-color:#d69a1f !important; color:#fff !important; font-weight:700;}
    .st-key-geo_demo_plan button p, .st-key-geo_demo_analyze button p {color:#fff !important;}
    .st-key-geo_demo_analyze button {background:#16856b !important; border-color:#16856b !important; color:#fff !important; font-weight:700;}
    .geo-data-preview-note {font-size:.74rem; color:#697386; margin:-.2rem 0 .5rem;}
    .geo-entry-note {font-size:.75rem; color:#697386; text-align:center; margin:.65rem 0 0;}
    .geo-choice-v2 {display:grid; grid-template-columns:56px 1fr; column-gap:1rem; min-height:156px; border:1px solid #e1c778; border-radius:7px; padding:1.2rem 1.25rem; background:#fffdf7; box-shadow:0 7px 22px rgba(35,41,54,.06);}
    .geo-choice-v2.analysis {border-color:#a8d2ca; background:#f6fbfa;}
    .geo-choice-icon {grid-row:1 / span 3; display:grid; place-items:center; align-self:start; width:52px; height:52px; border-radius:6px; background:#f5df9d; color:#815b00; font-size:1rem; font-weight:850;}
    .geo-choice-v2.analysis .geo-choice-icon {background:#cfe8e2; color:#0d6a58;}
    .geo-choice-v2 h3 {font-size:1.18rem; margin:.05rem 0 .28rem;}
    .geo-choice-v2 p {font-size:.84rem; line-height:1.5; color:#5e697b; margin:0; max-width:520px;}
    .geo-choice-v2 .geo-choice-result {align-self:end; margin-top:.65rem;}
    .geo-kpi-panel {border:1px solid var(--line); border-radius:7px; background:#fff; box-shadow:0 5px 18px rgba(35,41,54,.05); margin:.5rem 0 1rem; overflow:hidden;}
    .geo-kpi-head {display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; padding:.82rem 1rem .7rem; border-top:5px solid var(--teal); border-bottom:1px solid #edf0f4;}
    .geo-kpi-panel.plan .geo-kpi-head {border-top-color:var(--gold);}
    .geo-kpi-head strong {display:block; font-size:.98rem; color:var(--ink); margin-bottom:.1rem;}
    .geo-kpi-head span {display:block; font-size:.76rem; color:#687386;}
    .geo-status {flex:0 0 auto; border:1px solid #b9ddd5; background:#edf8f5; color:#0b6b58; padding:.28rem .5rem; border-radius:4px; font-size:.68rem; font-weight:750;}
    .geo-kpi-panel.plan .geo-status {border-color:#ead596; background:#fff8df; color:#8a6100;}
    .geo-kpis {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); padding:.25rem 0;}
    .geo-kpi {padding:.75rem 1rem .85rem; min-width:0;}
    .geo-kpi + .geo-kpi {border-left:1px solid #edf0f4;}
    .geo-kpi-label {font-size:.69rem; color:#758092; margin-bottom:.25rem;}
    .geo-kpi-value {font-size:1.34rem; line-height:1.15; font-weight:760; color:var(--ink); overflow-wrap:anywhere;}
    .geo-kpi-value.test {color:#dc4343;}
    .geo-kpi-value.control {color:#117865;}
    .geo-panel-title {font-size:.86rem; font-weight:750; color:var(--ink); margin:.15rem 0 .5rem;}
    .st-key-geo_start_plan button {background:#d69a1f !important; border-color:#d69a1f !important; color:#fff !important;}
    .st-key-geo_start_plan button p {color:#fff !important;}
    .st-key-geo_start_analysis button {background:#16856b; border-color:#16856b; color:#fff;}
    .st-key-geo_start_analysis button p {color:#fff;}
    @media (max-width: 800px) {
      .experiment-summary {grid-template-columns:repeat(2,1fr);}
      .summary-item:nth-child(3) {border-left:0; border-top:1px solid var(--line);}
      .summary-item:nth-child(4) {border-top:1px solid var(--line);}
      .overview-hero h1 {font-size:1.55rem;}
      .workflow-card {height:auto; min-height:190px;}
      .overview-flow {grid-template-columns:repeat(2,1fr);}
      .flow-step:nth-child(3) {border-left:0; border-top:1px solid var(--line);}
      .flow-step:nth-child(4) {border-top:1px solid var(--line);}
      .geo-choice {min-height:145px;}
      .geo-path {grid-template-columns:1fr;}
      .geo-path-step + .geo-path-step {border-left:0; border-top:1px solid var(--line);}
      .geo-choice-v2 {grid-template-columns:42px 1fr; padding:1rem; column-gap:.75rem;}
      .geo-choice-icon {width:40px; height:40px;}
      .geo-kpis {grid-template-columns:repeat(2,minmax(0,1fr));}
      .geo-kpi:nth-child(3) {border-left:0; border-top:1px solid #edf0f4;}
      .geo-kpi:nth-child(4) {border-top:1px solid #edf0f4;}
      .customer-choice-v2 {grid-template-columns:42px 1fr; padding:1rem; column-gap:.75rem;}
      .customer-choice-icon {width:40px; height:40px;}
      .customer-path, .customer-path.analyze {grid-template-columns:1fr;}
      .customer-path-step + .customer-path-step {border-left:0; border-top:1px solid var(--line);}
      .customer-kpis {grid-template-columns:repeat(2,minmax(0,1fr));}
      .customer-kpi:nth-child(3) {border-left:0; border-top:1px solid #edf0f4;}
      .customer-kpi:nth-child(4) {border-top:1px solid #edf0f4;}
      .ts-choice {grid-template-columns:42px 1fr; padding:1rem; column-gap:.75rem;}
      .ts-choice-icon {width:40px; height:40px;}
      .ts-path, .ts-path.analyze {grid-template-columns:1fr;}
      .ts-path-step + .ts-path-step {border-left:0; border-top:1px solid var(--line);}
      .ts-kpis {grid-template-columns:repeat(2,minmax(0,1fr));}
      .ts-kpi:nth-child(3) {border-left:0; border-top:1px solid #edf0f4;}
      .ts-kpi:nth-child(4) {border-top:1px solid #edf0f4;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .command-header {display:flex; align-items:flex-end; justify-content:space-between; gap:2rem; margin:.2rem 0 1rem;}
    .command-header h1 {font-size:2rem; margin:0 0 .22rem;}
    .command-header p {margin:0; font-size:.9rem; color:#657084;}
    .command-kpis {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); border:1px solid var(--line); border-radius:7px; background:#fff; margin-bottom:1rem; overflow:hidden;}
    .command-kpi {position:relative; padding:.8rem 1rem .78rem 3.25rem; min-height:72px;}
    .command-kpi + .command-kpi {border-left:1px solid var(--line);}
    .command-kpi-icon {position:absolute; left:1rem; top:1rem; display:grid; place-items:center; width:30px; height:30px; border-radius:50%; background:#e8f6f2; color:var(--teal); font-weight:850; font-size:.76rem;}
    .command-kpi-icon.ready {background:#fff4dd; color:#ad7200;}
    .command-kpi-icon.value {background:#fff0ef; color:#d73d3d;}
    .command-kpi-value {font-size:1.3rem; line-height:1.1; font-weight:780; color:var(--ink);}
    .command-kpi-label {font-size:.74rem; color:#697386; margin-top:.18rem;}
    .command-section-head {display:flex; align-items:baseline; gap:.65rem; margin:.85rem 0 .55rem;}
    .command-section-head strong {font-size:.98rem; color:var(--ink);}
    .command-section-head span {font-size:.72rem; color:#7a8495;}
    .lifecycle-rail {display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); border:1px solid var(--line); border-radius:7px; background:#fff; padding:.8rem .65rem .75rem; margin-bottom:1rem;}
    .lifecycle-stage {position:relative; text-align:center; min-width:0;}
    .lifecycle-stage:before {content:""; position:absolute; left:0; right:0; top:27px; height:2px; background:#dfe4ec;}
    .lifecycle-stage:first-child:before {left:50%;}
    .lifecycle-stage:last-child:before {right:50%;}
    .lifecycle-name {height:20px; font-size:.68rem; font-weight:720; color:#465164; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
    .lifecycle-count {position:relative; z-index:1; display:grid; place-items:center; width:25px; height:25px; margin:.25rem auto .4rem; border-radius:50%; background:#fff; border:2px solid #aeb8c7; color:#596476; font-size:.68rem; font-weight:800;}
    .lifecycle-stage.active .lifecycle-count {background:var(--teal); border-color:var(--teal); color:#fff; box-shadow:0 0 0 4px #e8f6f2;}
    .lifecycle-stage.ready .lifecycle-count {background:var(--gold); border-color:var(--gold); color:#fff; box-shadow:0 0 0 4px #fff4dd;}
    .lifecycle-example {height:28px; padding:0 .25rem; font-size:.66rem; line-height:1.25; color:#778194; overflow:hidden;}
    .attention-list {border:1px solid var(--line); border-radius:7px; background:#fff; overflow:hidden;}
    .attention-item {padding:.68rem .78rem; border-left:3px solid var(--gold);}
    .attention-item + .attention-item {border-top:1px solid #edf0f4;}
    .attention-item b {display:block; font-size:.77rem; color:var(--ink); margin-bottom:.12rem;}
    .attention-item span {display:block; font-size:.68rem; line-height:1.35; color:#727d8f;}
    .command-empty {padding:.9rem; font-size:.76rem; color:#697386; border:1px solid var(--line); border-radius:7px; background:#fff;}
    .experiment-detail-head {display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.25rem 0 .75rem; border-bottom:1px solid #edf0f4; margin-bottom:.8rem;}
    .experiment-detail-head strong {display:block; font-size:1rem; color:var(--ink);}
    .experiment-detail-head span {font-size:.72rem; color:#737d8e;}
    .pulse-row {display:grid; grid-template-columns:160px 1fr 62px; align-items:center; gap:.65rem; margin:.55rem 0;}
    .pulse-row span {font-size:.74rem; color:#596476;}
    .pulse-track {height:7px; border-radius:4px; background:#e8ecf2; overflow:hidden;}
    .pulse-fill {height:100%; background:var(--teal); border-radius:4px;}
    .pulse-fill.warn {background:var(--gold);}
    .pulse-value {text-align:right; font-size:.72rem; font-weight:720; color:var(--ink);}
    .pulse-summary {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border:1px solid var(--line); border-radius:6px; margin:.2rem 0 .85rem; background:#fff;}
    .pulse-summary-item {padding:.65rem .72rem; min-width:0;}
    .pulse-summary-item + .pulse-summary-item {border-left:1px solid var(--line);}
    .pulse-summary-item span {display:block; font-size:.63rem; color:#7a8495; text-transform:uppercase; font-weight:720;}
    .pulse-summary-item b {display:block; margin-top:.18rem; color:var(--ink); font-size:.8rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
    .allocation-row {display:grid; grid-template-columns:150px 1fr 86px; align-items:center; gap:.65rem; margin:.5rem 0;}
    .allocation-row span {font-size:.72rem; color:#596476; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
    .allocation-track {position:relative; height:10px; background:#e8ecf2; border-radius:5px; overflow:visible;}
    .allocation-fill {height:100%; background:#2457a6; border-radius:5px;}
    .allocation-plan {position:absolute; top:-3px; bottom:-3px; width:2px; background:#ef4d4d;}
    .allocation-row b {font-size:.7rem; color:var(--ink); text-align:right;}
    .interim-note {border-left:4px solid #7891b3; background:#f6f8fb; padding:.65rem .75rem; margin:.7rem 0; font-size:.72rem; line-height:1.4; color:#566174;}
    .record-version {display:inline-block; border:1px solid #dce2eb; border-radius:4px; padding:.2rem .42rem; font-size:.67rem; font-weight:720; color:#596476; background:#f8f9fb;}
    .decision-feed {border-top:1px solid #edf0f4;}
    .decision-feed-row {display:grid; grid-template-columns:12px 1fr auto; gap:.55rem; align-items:start; padding:.62rem 0; border-bottom:1px solid #edf0f4;}
    .decision-dot {width:8px; height:8px; margin-top:.25rem; border-radius:50%; background:var(--teal);}
    .decision-feed-row b {display:block; font-size:.74rem; color:var(--ink);}
    .decision-feed-row span {display:block; font-size:.67rem; color:#7a8495; margin-top:.08rem;}
    .decision-feed-row time {font-size:.65rem; color:#8a94a5; white-space:nowrap;}
    div[data-testid="stButton"] button {white-space:nowrap;}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    st.session_state.setdefault("portfolio_selected_id", "")
    st.session_state.setdefault("active_portfolio_id", "")
    st.session_state.setdefault("portfolio_show_create", False)
    st.session_state.setdefault("experiment_name", "New Customer Experiment")
    st.session_state.setdefault("data_config", DataConfig())
    st.session_state.setdefault("strategy_config", StrategyConfig())
    st.session_state.setdefault("design_config", DesignConfig())
    st.session_state.setdefault("analysis_config", AnalysisConfig())
    st.session_state.setdefault(
        "metrics_config",
        {
            "Primary": MetricConfig("Primary Outcome", "avg_revolving_balance_mob12", "Primary", "Continuous", "Higher is Better", "Relative %", 0.15, source_column="revolving_balance", processed_column="avg_revolving_balance_mob12", aggregation_method="Average", mob_horizon=12),
        },
    )
    st.session_state.setdefault("data_mapping", DataMappingConfig())
    st.session_state.setdefault("population_config", PopulationConfig())
    if not hasattr(st.session_state.data_mapping, "data_structure"):
        st.session_state.data_mapping.data_structure = "Longitudinal (unit x period)"
    if not hasattr(st.session_state.strategy_config, "experiment_template"):
        st.session_state.strategy_config.experiment_template = "Custom"
    if not hasattr(st.session_state.strategy_config, "arm_format"):
        st.session_state.strategy_config.arm_format = "Ordered numeric levels" if st.session_state.strategy_config.strategy_type == "Numeric Strategy" else "Named variants"
    if not hasattr(st.session_state.strategy_config, "control_name"):
        st.session_state.strategy_config.control_name = "BAU"
    if not hasattr(st.session_state.strategy_config, "historical_grouping"):
        st.session_state.strategy_config.historical_grouping = "Nearest proposed point"
    if not hasattr(st.session_state.strategy_config, "max_mapping_distance"):
        st.session_state.strategy_config.max_mapping_distance = 0.0
    strategy_defaults = StrategyConfig()
    for field_name in ["strategy_goal", "value_label", "value_format", "historical_evidence", "arm_descriptions", "selection_mode", "curve_binning_method", "curve_bin_width", "curve_bin_count", "curve_smoothing", "planning_grouping", "planning_bin_width"]:
        if not hasattr(st.session_state.strategy_config, field_name):
            setattr(st.session_state.strategy_config, field_name, getattr(strategy_defaults, field_name))
    if st.session_state.strategy_config.experiment_template == "Credit Line":
        st.session_state.strategy_config.experiment_template = "Acquisition Credit Line"
    if not st.session_state.get("customer_ui_v2_migrated"):
        apply_experiment_template(st.session_state.strategy_config, st.session_state.strategy_config.experiment_template)
        st.session_state.customer_ui_v2_migrated = True
    if not st.session_state.get("customer_general_workspace_v1"):
        strategy = st.session_state.strategy_config
        default_credit_setup = (
            strategy.experiment_template == "Acquisition Credit Line"
            and strategy.assignment_column == "assigned_credit_line"
            and strategy.strategy_name in {"Credit Line", "Acquisition Credit Line"}
        )
        if default_credit_setup:
            apply_experiment_template(strategy, "Custom")
            st.session_state["customer_experiment_template_v3"] = "Custom"
        if st.session_state.experiment_name == "Customer Strategy Experiment":
            st.session_state.experiment_name = "New Customer Experiment"
            st.session_state.pop("customer_experiment_name_v2", None)
        neutral_names = {
            "Primary": ({"Average Revolving Balance", "Revolving Balance"}, "Primary Outcome"),
            "Secondary": ({"Cumulative Revenue", "Risk Adjusted Revenue"}, "Secondary Outcome"),
            "Guardrail": ({"Cumulative Loss", "Loss"}, "Guardrail Outcome"),
        }
        for role, (legacy_names, neutral_name) in neutral_names.items():
            metric = st.session_state.metrics_config.get(role)
            if metric and metric.name in legacy_names:
                metric.name = neutral_name
                st.session_state.pop(f"metric_panel_name_{role}", None)
        st.session_state.design_config.group_plan_rows = []
        st.session_state.group_analysis_results = {}
        st.session_state.group_response_results = {}
        st.session_state.customer_general_workspace_v1 = True
    for field_name, default in {
        "multiplicity_method": "Holm",
        "audience_decision_scope": "Separate decision per audience",
        "attrition_rate": 0.0,
        "planned_launch_date": "",
        "outcome_delay_value": 0,
        "outcome_delay_unit": "Days",
        "arm_sample_sizes": {},
        "max_enrollment_periods": 0,
        "group_plan_rows": [],
    }.items():
        if not hasattr(st.session_state.design_config, field_name):
            setattr(st.session_state.design_config, field_name, default)
    for field_name, default in {"unit_id_column": "", "multiplicity_method": "Holm"}.items():
        if not hasattr(st.session_state.analysis_config, field_name):
            setattr(st.session_state.analysis_config, field_name, default)
    st.session_state.setdefault("raw_df", load_raw_demo())
    st.session_state.setdefault("customer_intent", None)
    st.session_state.setdefault("customer_demo_scenario", "General Customer Test")
    st.session_state.setdefault("historical_df", pd.DataFrame())
    st.session_state.setdefault("processing_result", None)
    st.session_state.setdefault("results_df", None)
    st.session_state.setdefault("analysis_results_by_role", {})
    st.session_state.setdefault("response_by_role", {})
    st.session_state.setdefault("group_analysis_results", {})
    st.session_state.setdefault("group_response_results", {})
    st.session_state.setdefault("historical_arm_stats_by_role", {})
    for metric in st.session_state.metrics_config.values():
        if not hasattr(metric, "observation_unit"):
            metric.observation_unit = "Months"
        if not hasattr(metric, "mob_start"):
            metric.mob_start = 1
    population_defaults = PopulationConfig()
    for field_name in ["grouping_column", "grouping_definitions", "group_settings"]:
        if not hasattr(st.session_state.population_config, field_name):
            setattr(st.session_state.population_config, field_name, getattr(population_defaults, field_name))
    if not hasattr(st.session_state.strategy_config, "group_arms"):
        st.session_state.strategy_config.group_arms = {}
    if not st.session_state.get("customer_group_plan_migrated"):
        for metric in st.session_state.metrics_config.values():
            metric.effect_type = "Relative %"
            metric.effect_value = 0.15
            if metric.role == "Guardrail":
                metric.guardrail_threshold = 0.15
        st.session_state.design_config.sample_size_basis = "Power All Configured Metrics"
        st.session_state.design_config.traffic_frequency = "Monthly"
        st.session_state.customer_group_plan_migrated = True
    if not st.session_state.get("customer_simple_flow_v1"):
        st.session_state.metrics_config = {"Primary": st.session_state.metrics_config["Primary"]}
        st.session_state.pop("enable_secondary_v2", None)
        st.session_state.pop("enable_guardrail_v2", None)
        st.session_state.customer_simple_flow_v1 = True
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
    st.session_state.setdefault("geo_intent", None)
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


def metric_window_label(metric: MetricConfig) -> str:
    start = max(int(getattr(metric, "mob_start", 1)), 1)
    return f"Through MOB {metric.mob_horizon}" if start == 1 else f"MOB {start} to MOB {metric.mob_horizon}"


def grouping_candidates(raw_df: pd.DataFrame, unit_column: str, excluded: set[str]) -> list[str]:
    if unit_column not in raw_df.columns:
        return []
    candidates = []
    for column in raw_df.columns:
        if column in excluded or column.startswith("_"):
            continue
        values = raw_df[[unit_column, column]].dropna(subset=[unit_column])
        if values.empty:
            continue
        fixed = values.groupby(unit_column)[column].nunique(dropna=True).max() <= 1
        unique = values[column].nunique(dropna=True)
        if fixed and 1 < unique < max(values[unit_column].nunique(), 200):
            candidates.append(column)
    return candidates


def strategy_candidates(raw_df: pd.DataFrame, unit_column: str, excluded: set[str]) -> list[str]:
    """Find stable per-customer fields that could describe historical treatments."""
    if unit_column not in raw_df.columns:
        return []
    candidates = []
    for column in raw_df.columns:
        if column in excluded or column.startswith("_"):
            continue
        values = raw_df[[unit_column, column]].dropna(subset=[unit_column])
        if values.empty or values.groupby(unit_column)[column].nunique(dropna=True).max() > 1:
            continue
        unique = values[column].nunique(dropna=True)
        if 1 < unique <= 50:
            candidates.append(column)
    strategy_hints = ("strategy", "treatment", "offer", "line", "price", "fee", "rate", "channel", "message", "design", "reward", "incentive", "action", "experience", "policy", "variant", "arm")
    outcome_hints = ("outcome", "revenue", "loss", "score", "flag", "cost", "balance", "utilization", "value")
    preferred = [
        column
        for column in candidates
        if any(token in column.lower() for token in strategy_hints)
        and not any(token in column.lower() for token in outcome_hints)
    ]
    return preferred or candidates


def reset_customer_data_choices() -> None:
    """Clear choices that must be inferred again when the underlying data changes."""
    population: PopulationConfig = st.session_state.population_config
    population.selected_cpcs = []
    population.grouping_column = ""
    population.grouping_definitions = []
    population.group_settings = {}
    apply_experiment_template(st.session_state.strategy_config, "Custom")
    st.session_state.metrics_config = {
        "Primary": MetricConfig(
            "Primary Outcome",
            "primary_outcome",
            "Primary",
            "Continuous",
            "Higher is Better",
            "Relative %",
            0.15,
            source_column="",
            processed_column="primary_outcome",
            aggregation_method="Average",
            mob_horizon=12,
        )
    }
    st.session_state.processing_result = None
    st.session_state.historical_df = pd.DataFrame()
    st.session_state.design_config.group_plan_rows = []
    st.session_state.group_analysis_results = {}
    st.session_state.group_response_results = {}
    st.session_state.pop("outcome_strategy_view_v1", None)
    reset_prefixes = (
        "customer_grouping_",
        "customer_group_bands_",
        "customer_group_settings_",
        "metric_panel_",
        "metric_curve_",
        "group_metric_",
        "strategy_data_field_",
        "strategy_new_format_",
        "generic_numeric_arms_",
        "generic_named_arms_",
    )
    for key in list(st.session_state):
        if key.startswith(reset_prefixes):
            st.session_state.pop(key, None)


def apply_demo_metric_default(scenario: str) -> None:
    defaults = {
        "General Customer Test": ("Primary Outcome", "primary_outcome", "Continuous", "Average"),
        "Offer & Engagement": ("Engagement Score", "engagement_score", "Continuous", "Average"),
        "Retention Strategy": ("Retention Rate", "retained_flag", "Binary", "Average"),
        "Credit Strategy": ("Revolving Balance", "revolving_balance", "Continuous", "Average"),
    }
    name, source, metric_type, aggregation = defaults[scenario]
    st.session_state.metrics_config["Primary"] = MetricConfig(
        name,
        source,
        "Primary",
        metric_type,
        "Higher is Better",
        "Relative %",
        0.15,
        source_column=source,
        processed_column=source,
        aggregation_method=aggregation,
        mob_horizon=12,
    )


def apply_strategy_column_defaults(strategy: StrategyConfig, raw_df: pd.DataFrame, unit_column: str, column: str) -> None:
    unit_values = raw_df[[unit_column, column]].dropna().drop_duplicates(unit_column)[column]
    numeric_values = pd.to_numeric(unit_values, errors="coerce")
    is_numeric = numeric_values.notna().mean() > 0.95
    strategy.experiment_template = "Custom"
    strategy.strategy_name = humanize_column_name(column)
    strategy.assignment_column = f"assigned_{column}"
    strategy.historical_column = column
    strategy.historical_evidence = "Use historical strategy values" if is_numeric else "Use group-level outcome history"
    strategy.strategy_goal = ""
    strategy.group_arms = {}
    if is_numeric:
        unique = np.sort(numeric_values.dropna().unique().astype(float))
        strategy.arm_format = "Ordered numeric levels"
        strategy.strategy_type = "Numeric Strategy"
        strategy.value_label = humanize_column_name(column)
        lower_name = column.lower()
        strategy.value_format = "Currency" if any(token in lower_name for token in ["line", "price", "fee", "cost", "amount"]) else "Percent" if any(token in lower_name for token in ["percent", "pct", "rate"]) else "Number"
        strategy.min_value = float(unique.min())
        strategy.max_value = float(unique.max())
        differences = np.diff(unique)
        strategy.increment = float(np.median(differences[differences > 0])) if (differences > 0).any() else 1.0
        mode = float(numeric_values.mode().iloc[0])
        strategy.control_name = "Control"
        strategy.control_value = mode
        quantile_targets = numeric_values.quantile([0.25, 0.75]).tolist()
        proposed = []
        for target in quantile_targets:
            point = float(unique[np.abs(unique - float(target)).argmin()])
            if point != mode and point not in proposed:
                proposed.append(point)
        if not proposed:
            proposed = [float(value) for value in unique if float(value) != mode][:2]
        strategy.treatment_values = proposed[:2]
        strategy.treatment_names = [f"Treatment {index}" for index in range(1, len(strategy.treatment_values) + 1)]
        strategy.number_of_arms = len(strategy.treatment_values) + 1
        strategy.arm_descriptions = {}
    else:
        ordered = unit_values.astype(str).value_counts().index.tolist()
        strategy.arm_format = "Named variants"
        strategy.strategy_type = "Categorical Strategy"
        strategy.categorical_control = ordered[0]
        strategy.categorical_treatments = ordered[1:4]
        strategy.number_of_arms = len(strategy.categorical_treatments) + 1
        strategy.arm_descriptions = {value: "Observed in historical data" for value in ordered[:4]}


def group_labeled_history() -> pd.DataFrame:
    data = st.session_state.historical_df.copy()
    population: PopulationConfig = st.session_state.population_config
    column = getattr(population, "grouping_column", "")
    if not column or column not in data.columns:
        data["_planning_group"] = "All customers"
    else:
        data["_planning_group"] = assign_group_labels(data[column], population.grouping_definitions)
    return data.dropna(subset=["_planning_group"])


def sync_group_settings(data: pd.DataFrame) -> None:
    population: PopulationConfig = st.session_state.population_config
    strategy: StrategyConfig = st.session_state.strategy_config
    groups = data["_planning_group"].astype(str).drop_duplicates().tolist() if "_planning_group" in data else ["All customers"]
    total = max(len(data), 1)
    historical = pd.to_numeric(data.get(strategy.historical_column), errors="coerce") if strategy.strategy_type == "Numeric Strategy" and strategy.historical_column in data else pd.Series(dtype=float)
    overall_min = float(historical.min()) if not historical.empty and historical.notna().any() else float(strategy.min_value)
    overall_max = float(historical.max()) if not historical.empty and historical.notna().any() else float(strategy.max_value)
    updated = {}
    for group in groups:
        scoped = data[data["_planning_group"].astype(str) == group]
        lines = pd.to_numeric(scoped.get(strategy.historical_column), errors="coerce").dropna() if strategy.strategy_type == "Numeric Strategy" and strategy.historical_column in scoped else pd.Series(dtype=float)
        minimum = float(lines.min()) if not lines.empty else overall_min
        maximum = float(lines.max()) if not lines.empty else overall_max
        control = float(lines.median()) if not lines.empty else (minimum + maximum) / 2
        previous = population.group_settings.get(group, {})
        strategy_signature = (strategy.strategy_type, strategy.historical_column)
        if tuple(previous.get("_strategy_signature", ())) != strategy_signature:
            previous = {}
        updated[group] = {
            "min_line": float(previous.get("min_line", minimum)),
            "max_line": float(previous.get("max_line", maximum)),
            "eligible_flow": int(previous.get("eligible_flow", max(1, round(st.session_state.design_config.eligible_customers * len(scoped) / total)))),
            "control_line": float(previous.get("control_line", control)),
            "treatment_lines": [float(value) for value in previous.get("treatment_lines", [])],
            "selection_mode": str(previous.get("selection_mode", "Suggested lines")),
            "control_arm": previous.get("control_arm", strategy.categorical_control),
            "treatment_arms": list(previous.get("treatment_arms", strategy.categorical_treatments)),
            "_strategy_signature": strategy_signature,
        }
    population.group_settings = updated


def synthetic_group_results() -> pd.DataFrame:
    design: DesignConfig = st.session_state.design_config
    strategy: StrategyConfig = st.session_state.strategy_config
    if not design.group_plan_rows:
        return load_results_demo()
    rng = np.random.default_rng(2026)
    records = []
    group_totals = {}
    group_primary_baselines = {}
    for row in design.group_plan_rows:
        group_totals[str(row["Group"])] = group_totals.get(str(row["Group"]), 0) + int(row["Required Accounts"])
        if str(row["Role"]) == "Control":
            group_primary_baselines[str(row["Group"])] = float(row["Historical Mean"])
    for plan_row in design.group_plan_rows:
        group = str(plan_row["Group"])
        arm = plan_row["Arm"]
        role = str(plan_row["Role"])
        scale = min(1.0, 900 / max(group_totals[group], 1))
        count = max(2, int(round(int(plan_row["Required Accounts"]) * scale)))
        for index in range(count):
            arm_slug = "".join(character if character.isalnum() else "_" for character in str(arm))[:32]
            record = {
                "customer_id": f"{group}_{arm_slug}_{index:05d}",
                "_planning_group": group,
                strategy.assignment_column: arm,
            }
            for metric in st.session_state.metrics_config.values():
                baseline = metric.baseline_rate if metric.metric_type == "Binary" else metric.baseline_mean
                if metric.role == "Primary":
                    baseline = group_primary_baselines.get(group, float(plan_row["Historical Mean"]))
                direction = -1 if metric.direction == "Lower is Better" else 1
                effect = 0 if role == "Control" else direction * abs(float(metric.effect_value) * baseline) * 1.15
                if metric.metric_type == "Binary":
                    probability = min(max(baseline + effect, 0.001), 0.999)
                    value = int(rng.binomial(1, probability))
                else:
                    standard_deviation = float(plan_row["Historical SD"]) if metric.role == "Primary" else max(float(metric.standard_deviation), abs(baseline) * 0.1, 1.0)
                    value = float(rng.normal(baseline + effect, standard_deviation))
                record[metric.source_column or metric.column] = value
            records.append(record)
    return pd.DataFrame(records)


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


def format_strategy_value(value: object, strategy: StrategyConfig) -> str:
    if strategy.strategy_type != "Numeric Strategy":
        return str(value)
    numeric = float(value)
    if strategy.value_format == "Currency":
        return maybe_money(numeric)
    if strategy.value_format == "Percent":
        return f"{numeric:,.1f}%"
    return number(numeric, 2)


def render_business_guide(message: str, output: str) -> None:
    st.markdown(
        f'<div class="business-guide"><span>{html.escape(message)}</span><b>{html.escape(output)}</b></div>',
        unsafe_allow_html=True,
    )


def render_customer_summary() -> None:
    population: PopulationConfig = st.session_state.population_config
    strategy: StrategyConfig = st.session_state.strategy_config
    products = population.selected_cpcs
    if not products:
        product_text = "All products / brands"
    elif len(products) <= 2:
        product_text = " + ".join(str(value) for value in products)
    else:
        product_text = f"{len(products)} products / brands"
    grouping_text = humanize_column_name(population.grouping_column) if population.grouping_column else "All customers"
    primary = st.session_state.metrics_config.get("Primary")
    values = [
        ("Product scope", product_text),
        ("Customer grouping", grouping_text),
        ("Primary metric", primary.name if primary else "Not configured"),
        ("Strategy", strategy.experiment_template),
    ]
    content = "".join(
        f'<div class="summary-item"><div class="summary-label">{html.escape(label)}</div>'
        f'<div class="summary-value" title="{html.escape(str(value))}">{html.escape(str(value))}</div></div>'
        for label, value in values
    )
    st.markdown(f'<div class="experiment-summary">{content}</div>', unsafe_allow_html=True)


def render_arm_overview(strategy: StrategyConfig) -> None:
    control, treatments = configured_arms(strategy)
    items = [
        f'<div class="arm-chip control"><b>Control</b>{html.escape(format_strategy_value(control, strategy))}</div>'
    ]
    items.extend(
        f'<div class="arm-chip"><b>Treatment {index}</b>{html.escape(format_strategy_value(arm, strategy))}</div>'
        for index, arm in enumerate(treatments, start=1)
    )
    st.markdown(f'<div class="arm-overview">{"".join(items)}</div>', unsafe_allow_html=True)


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
    legacy = {
        "Credit Line": "Acquisition Credit Line",
        "Marketing Offer": "Multiple Offer Strategy",
        "Digital Experience": "Card / Product Design",
        "Retention": "Retention / Save Offer",
    }
    apply_customer_template(strategy, legacy.get(template, template))


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


def open_workspace(page: str) -> None:
    st.session_state.main_page = page


def compact_money(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"


def _restore_dataclass(target: object, payload: dict) -> object:
    allowed = {item.name for item in fields(target)}
    for key, value in payload.items():
        if key in allowed:
            setattr(target, key, value)
    return target


def workspace_configuration(experiment_type: str) -> dict:
    if experiment_type == "Customer":
        return {
            "data_config": asdict(st.session_state.data_config),
            "data_mapping": asdict(st.session_state.data_mapping),
            "population_config": asdict(st.session_state.population_config),
            "strategy_config": asdict(st.session_state.strategy_config),
            "design_config": asdict(st.session_state.design_config),
            "analysis_config": asdict(st.session_state.analysis_config),
            "metrics_config": {role: asdict(metric) for role, metric in st.session_state.metrics_config.items()},
        }
    if experiment_type == "Geography":
        return {"geo_config": asdict(st.session_state.geo_config)}
    return {"ts_config": asdict(st.session_state.ts_config)}


def workspace_result_summary(experiment_type: str) -> dict:
    if experiment_type == "Customer":
        primary = st.session_state.analysis_results_by_role.get("Primary")
        if primary is None or primary.empty or not st.session_state.get("customer_analysis_is_current"):
            return {}
        treatments = primary[~primary["Is Control"]].copy()
        if treatments.empty:
            return {"headline": "Customer analysis completed", "confidence": "No treatment comparison available"}
        direction = st.session_state.metrics_config["Primary"].direction
        ordered = treatments.sort_values("Effect vs Control", ascending=direction == "Lower is Better")
        leader = ordered.iloc[0]
        lift = leader.get("Relative Lift", float("nan"))
        headline = f'{leader["Credit Line"]} is the leading tested option'
        if pd.notna(lift):
            headline += f" at {lift:+.1%} versus control"
        return {
            "headline": headline,
            "confidence": str(leader.get("Result", "Analysis complete")),
            "leading_option": str(leader["Credit Line"]),
            "relative_lift": None if pd.isna(lift) else float(lift),
        }
    if experiment_type == "Geography":
        result = st.session_state.get("geo_analysis")
        if not result:
            return {}
        return {
            "headline": f'Estimated geographic lift: {float(result.get("lift", 0)):+.1%}',
            "confidence": "Geographic analysis complete",
            "incremental_volume": float(result.get("incremental_volume", 0)),
        }
    cfg: TimeSeriesConfig = st.session_state.ts_config
    result = st.session_state.get("ts_bsts_result") if cfg.analysis_method == "Structural Time Series Counterfactual" else st.session_state.get("ts_prepost_result")
    if not result:
        return {}
    if cfg.analysis_method == "Structural Time Series Counterfactual":
        return {
            "headline": f'Estimated incremental impact: {float(result.get("cumulative_impact", 0)):+,.0f}',
            "confidence": f'Model reliability: {result.get("reliability", "Review")}',
            "relative_impact": float(result.get("relative_impact", 0)),
        }
    return {
        "headline": f'Observed pre–post change: {float(result.get("percent_change", 0)):+.1%}',
        "confidence": "Descriptive result; not a causal estimate",
    }


def save_current_workspace(experiment_type: str) -> dict:
    experiments = load_experiments()
    active_id = st.session_state.get("active_portfolio_id", "")
    existing = next((item for item in experiments if item["id"] == active_id and item["type"] == experiment_type), None)
    if existing is None:
        existing = {
            "name": st.session_state.experiment_name if experiment_type == "Customer" else f"New {experiment_type} Experiment",
            "type": experiment_type,
            "owner": "Unassigned",
            "primary_outcome": "Primary outcome",
        }
    record = save_configuration(existing, workspace_configuration(experiment_type))
    result = workspace_result_summary(experiment_type)
    now = datetime.now().isoformat(timespec="seconds")
    if experiment_type == "Customer":
        design: DesignConfig = st.session_state.design_config
        primary = st.session_state.metrics_config.get("Primary")
        result_frame = st.session_state.analysis_results_by_role.get("Primary")
        sample_collected = int(result_frame["N"].sum()) if result_frame is not None and not result_frame.empty else 0
        sample_target = int(design.total_sample_size or sum(design.arm_sample_sizes.values()))
        planned = design.arm_sample_sizes
        observed = result_frame.set_index("Credit Line")["N"].to_dict() if result_frame is not None and not result_frame.empty else {}
        all_arms = list(dict.fromkeys([*planned, *observed]))
        planned_total = max(sum(planned.values()), 1)
        observed_total = max(sum(observed.values()), 1)
        record.update(
            {
                "name": st.session_state.experiment_name,
                "primary_outcome": primary.name if primary else "Primary outcome",
                "planned_launch_date": design.planned_launch_date or record.get("planned_launch_date", ""),
                "sample_collected": sample_collected,
                "sample_target": sample_target,
                "traffic_per_period": int(design.eligible_customers),
                "traffic_frequency": duration_unit(design.traffic_frequency),
                "allocation": [
                    {
                        "arm": str(arm),
                        "observed_pct": 100 * float(observed.get(arm, 0)) / observed_total,
                        "planned_pct": 100 * float(planned.get(arm, 0)) / planned_total,
                    }
                    for arm in all_arms
                ],
            }
        )
    elif experiment_type == "Geography":
        cfg: GeoConfig = st.session_state.geo_config
        assignment = st.session_state.geo_assignment
        target = int(st.session_state.geo_plan.get("required_total_dmas", 0)) if st.session_state.geo_plan else 0
        observed = assignment["Group"].value_counts().to_dict() if assignment is not None and not assignment.empty and "Group" in assignment else {}
        total = max(sum(observed.values()), 1)
        record.update(
            {
                "primary_outcome": humanize_column_name(cfg.outcome_column) if cfg.outcome_column else record.get("primary_outcome", "Primary outcome"),
                "planned_launch_date": cfg.planned_launch_date or record.get("planned_launch_date", ""),
                "sample_collected": int(sum(observed.values())),
                "sample_target": target,
                "traffic_per_period": 0,
                "traffic_frequency": "markets",
                "allocation": [{"arm": str(arm), "observed_pct": 100 * count / total, "planned_pct": 50.0} for arm, count in observed.items()],
            }
        )
    else:
        cfg: TimeSeriesConfig = st.session_state.ts_config
        plan = st.session_state.get("ts_duration_plan") or {}
        target = int(plan.get("recommended_duration") or 0)
        prepared = st.session_state.get("ts_prepared_df")
        collected = 0
        if prepared is not None and not prepared.empty and cfg.intervention_date:
            collected = int((pd.to_datetime(prepared["_date"]) >= pd.Timestamp(cfg.intervention_date)).sum())
        record.update(
            {
                "primary_outcome": humanize_column_name(cfg.outcome_column) if cfg.outcome_column else record.get("primary_outcome", "Primary outcome"),
                "planned_launch_date": cfg.planned_launch_date or cfg.intervention_date or record.get("planned_launch_date", ""),
                "sample_collected": collected,
                "sample_target": target,
                "traffic_per_period": 1,
                "traffic_frequency": f'{cfg.frequency.lower()} observation' if cfg.frequency != "Auto" else "observation period",
                "allocation": [],
            }
        )
    record["data_updated_at"] = now
    if result:
        record["latest_result"] = result
        record["analysis_timestamp"] = now
        record["status"] = "Ready to decide"
        record["progress"] = 100
        record["action"] = "Review and decide"
        record["primary_direction"] = result.get("headline", "Analysis complete")
    else:
        record.setdefault("status", "Planning")
        target = max(int(record.get("sample_target", 0)), 0)
        collected = max(int(record.get("sample_collected", 0)), 0)
        record["progress"] = min(int(100 * collected / target), 99) if target else int(record.get("progress", 15))
    saved = upsert_experiment(record)
    st.session_state.active_portfolio_id = saved["id"]
    st.session_state.portfolio_selected_id = saved["id"]
    return saved


def restore_portfolio_experiment(experiment: dict) -> None:
    payload = experiment.get("config_payload") or {}
    if experiment["type"] == "Customer" and payload:
        _restore_dataclass(st.session_state.data_config, payload.get("data_config", {}))
        _restore_dataclass(st.session_state.data_mapping, payload.get("data_mapping", {}))
        _restore_dataclass(st.session_state.population_config, payload.get("population_config", {}))
        _restore_dataclass(st.session_state.strategy_config, payload.get("strategy_config", {}))
        _restore_dataclass(st.session_state.design_config, payload.get("design_config", {}))
        _restore_dataclass(st.session_state.analysis_config, payload.get("analysis_config", {}))
        st.session_state.metrics_config = {
            role: _restore_dataclass(MetricConfig("", "", role), metric_payload)
            for role, metric_payload in payload.get("metrics_config", {}).items()
        } or st.session_state.metrics_config
        st.session_state.experiment_name = experiment["name"]
        st.session_state.pop("customer_experiment_name_v2", None)
        st.session_state.analysis_results_by_role = {}
        st.session_state.response_by_role = {}
        st.session_state.customer_analysis_is_current = False
        if st.session_state.data_config.source_type == "Upload File":
            st.session_state.raw_df = pd.DataFrame()
            st.session_state.processing_result = None
    elif experiment["type"] == "Geography" and payload:
        _restore_dataclass(st.session_state.geo_config, payload.get("geo_config", {}))
        st.session_state.geo_analysis = None
        st.session_state.geo_plan = None
        if st.session_state.geo_config.data_source == "Upload File":
            st.session_state.geo_raw_df = pd.DataFrame()
            st.session_state.geo_panel_df = pd.DataFrame()
            st.session_state.geo_assignment = pd.DataFrame()
    elif experiment["type"] == "Time Series" and payload:
        _restore_dataclass(st.session_state.ts_config, payload.get("ts_config", {}))
        st.session_state.ts_prepost_result = None
        st.session_state.ts_bsts_result = None
        st.session_state.ts_duration_plan = None
        if st.session_state.ts_config.data_source == "Upload File":
            st.session_state.ts_raw_df = pd.DataFrame()
            st.session_state.ts_prepared_df = pd.DataFrame()
    st.session_state.active_portfolio_id = experiment["id"]
    st.session_state.portfolio_selected_id = experiment["id"]
    intent = "analyze" if experiment["status"] in {"Running", "Collecting outcomes", "Ready to decide", "Rolling out", "Completed"} else "plan"
    portfolio_workspace(experiment["type"], intent)


def portfolio_workspace(experiment_type: str, intent: str = "plan") -> None:
    st.session_state.main_page = experiment_type
    if experiment_type == "Customer":
        st.session_state.customer_intent = intent
    elif experiment_type == "Geography":
        st.session_state.geo_intent = intent
    else:
        st.session_state.ts_intent = intent


@st.dialog("Start a new experiment", width="large")
def create_experiment_dialog() -> None:
    st.caption("Answer four business questions. The platform will open the right planning workflow.")
    with st.form("portfolio_create_experiment"):
        name = st.text_input("What are you testing?", placeholder="Example: New retention offer")
        delivery = st.selectbox(
            "Who can receive different experiences?",
            [
                "Customers or accounts",
                "Markets or regions",
                "Everyone at the same time",
            ],
        )
        left, right = st.columns(2)
        owner = left.text_input("Owner", placeholder="Team or person")
        outcome = right.text_input("Main outcome", placeholder="Example: 90-day retention")
        launch_col, readout_col = st.columns(2)
        launch = launch_col.date_input("Planned launch", value=pd.Timestamp.today().date() + pd.Timedelta(days=14))
        readout = readout_col.date_input("Expected decision", value=pd.Timestamp.today().date() + pd.Timedelta(days=60))
        submitted = st.form_submit_button("Create and start planning", type="primary", use_container_width=True)
    if not submitted:
        return
    if not name.strip() or not outcome.strip():
        st.error("Add the experiment name and main outcome before continuing.")
        return
    experiment_type = {
        "Customers or accounts": "Customer",
        "Markets or regions": "Geography",
        "Everyone at the same time": "Time Series",
    }[delivery]
    record = upsert_experiment(
        {
            "name": name.strip(),
            "description": "New experiment created from the Command Center.",
            "type": experiment_type,
            "owner": owner.strip() or "Unassigned",
            "primary_outcome": outcome.strip(),
            "status": "Planning",
            "progress": 8,
            "planned_launch_date": launch.isoformat(),
            "readout_date": readout.isoformat(),
            "action": "Complete test plan",
            "projected_value": 0,
            "allocation_health": 100,
            "alert": "",
            "guardrails": [],
            "decision_snapshot": None,
            "rollout_plan": [],
        }
    )
    st.session_state.portfolio_selected_id = record["id"]
    st.session_state.active_portfolio_id = record["id"]
    if experiment_type == "Customer":
        st.session_state.experiment_name = record["name"]
        st.session_state.pop("customer_experiment_name_v2", None)
    portfolio_workspace(experiment_type, "plan")
    st.rerun()


def lifecycle_html(experiments: list[dict]) -> str:
    stages = []
    for status in PORTFOLIO_STATUSES:
        matches = [item for item in experiments if item.get("status") == status]
        example = matches[0]["name"] if matches else "No experiment"
        tone = "ready" if status == "Ready to decide" else "active" if matches and status not in {"Planning", "Completed"} else ""
        stages.append(
            f'<div class="lifecycle-stage {tone}"><div class="lifecycle-name">{html.escape(status)}</div>'
            f'<div class="lifecycle-count">{len(matches)}</div><div class="lifecycle-example">{html.escape(example)}</div></div>'
        )
    return f'<div class="lifecycle-rail">{"".join(stages)}</div>'


def render_attention_list(experiments: list[dict]) -> None:
    alerts = [item for item in experiments if item.get("alert")]
    if not alerts:
        st.markdown('<div class="command-empty">Nothing needs attention right now.</div>', unsafe_allow_html=True)
        return
    items = "".join(
        f'<div class="attention-item"><b>{html.escape(item["name"])}</b><span>{html.escape(item["alert"])}</span></div>'
        for item in alerts[:3]
    )
    st.markdown(f'<div class="attention-list">{items}</div>', unsafe_allow_html=True)


def render_experiment_detail(experiment: dict) -> None:
    with st.expander(f'Open experiment details · {experiment["name"]}', expanded=False):
        st.markdown(
            f'<div class="experiment-detail-head"><div><strong>{html.escape(experiment["name"])}</strong>'
            f'<span>{html.escape(experiment.get("description", ""))}</span></div><span class="record-version">Config v{int(experiment.get("config_version", 1))}</span></div>',
            unsafe_allow_html=True,
        )
        pulse_tab, record_tab, decision_tab, rollout_tab = st.tabs(["Pulse", "Record", "Decision", "Rollout"])
        with pulse_tab:
            pulse = pulse_summary(experiment)
            progress = int(round(100 * float(pulse["sample_progress"])))
            st.markdown(
                f'<div class="pulse-summary">'
                f'<div class="pulse-summary-item"><span>Sample collected</span><b>{pulse["sample_collected"]:,} / {pulse["sample_target"]:,}</b></div>'
                f'<div class="pulse-summary-item"><span>Current traffic</span><b>{pulse["traffic_per_period"]:,.0f} / {html.escape(str(pulse["traffic_frequency"]))}</b></div>'
                f'<div class="pulse-summary-item"><span>Estimated completion</span><b>{html.escape(str(pulse["estimated_completion"]))}</b></div>'
                f'<div class="pulse-summary-item"><span>Data updated</span><b>{html.escape(str(pulse["data_updated_at"]))[:16]}</b></div>'
                f'</div>'
                f'<div class="pulse-row"><span>Collection progress</span><div class="pulse-track"><div class="pulse-fill" style="width:{progress}%"></div></div><div class="pulse-value">{progress}%</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(f'**Primary metric direction:** {experiment["primary_outcome"]} · {pulse["primary_direction"]}')
            st.markdown('<div class="interim-note">This is a monitoring signal only. The platform will not declare a winner before the planned sample and analysis are complete.</div>', unsafe_allow_html=True)
            if pulse["allocation"]:
                st.markdown("**Traffic allocation**")
                allocation_rows = "".join(
                    f'<div class="allocation-row"><span>{html.escape(str(row["arm"]))}</span><div class="allocation-track">'
                    f'<div class="allocation-fill" style="width:{max(0, min(float(row["observed_pct"]), 100))}%"></div>'
                    f'<div class="allocation-plan" style="left:{max(0, min(float(row["planned_pct"]), 100))}%"></div></div>'
                    f'<b>{float(row["observed_pct"]):.1f}% / {float(row["planned_pct"]):.1f}%</b></div>'
                    for row in pulse["allocation"]
                )
                st.markdown(allocation_rows, unsafe_allow_html=True)
            guardrails = experiment.get("guardrails", [])
            if guardrails:
                st.markdown("**Guardrails**")
                st.dataframe(pd.DataFrame(guardrails).rename(columns={"name": "Safety check", "value": "Current", "threshold": "Limit", "status": "Status"}), hide_index=True, use_container_width=True)
            for alert in pulse["alerts"]:
                st.warning(alert)
            st.button(
                f'Reopen {experiment["type"]} experiment',
                type="primary",
                key=f'portfolio_open_{experiment["id"]}',
                on_click=restore_portfolio_experiment,
                args=(experiment,),
            )
        with record_tab:
            st.caption("Save the business record here. Uploaded source files are not copied into the portfolio.")
            try:
                launch_default = pd.Timestamp(experiment.get("planned_launch_date")).date()
            except (TypeError, ValueError):
                launch_default = pd.Timestamp.today().date()
            try:
                readout_default = pd.Timestamp(experiment.get("readout_date")).date()
            except (TypeError, ValueError):
                readout_default = pd.Timestamp.today().date() + pd.Timedelta(days=60)
            with st.form(f'portfolio_record_{experiment["id"]}'):
                owner_col, status_col = st.columns(2)
                owner = owner_col.text_input("Owner", value=experiment.get("owner", ""))
                status = status_col.selectbox("Status", PORTFOLIO_STATUSES, index=PORTFOLIO_STATUSES.index(experiment["status"]) if experiment["status"] in PORTFOLIO_STATUSES else 0)
                launch_col, readout_col = st.columns(2)
                launch = launch_col.date_input("Planned launch", value=launch_default)
                readout = readout_col.date_input("Expected result", value=readout_default)
                save_record = st.form_submit_button("Save record", type="primary")
            if save_record:
                updated = {**experiment, "owner": owner.strip() or "Unassigned", "status": status, "planned_launch_date": launch.isoformat(), "readout_date": readout.isoformat()}
                upsert_experiment(updated)
                st.rerun()
            meta_left, meta_right = st.columns(2)
            meta_left.caption(f'Configuration version: v{int(experiment.get("config_version", 1))}')
            meta_right.caption(f'Last analysis: {experiment.get("analysis_timestamp") or "Not analyzed"}')
            action_left, action_mid, action_right = st.columns(3)
            if action_left.button("Copy experiment", use_container_width=True, key=f'copy_experiment_{experiment["id"]}'):
                copied = upsert_experiment(duplicate_experiment(experiment))
                st.session_state.portfolio_selected_id = copied["id"]
                st.rerun()
            action_mid.download_button(
                "Download one-page report",
                data=business_report_html(experiment).encode("utf-8"),
                file_name=f'{experiment["id"]}-business-report.html',
                mime="text/html",
                use_container_width=True,
                key=f'report_{experiment["id"]}',
            )
            action_right.button(
                "Reopen experiment",
                use_container_width=True,
                key=f'reopen_record_{experiment["id"]}',
                on_click=restore_portfolio_experiment,
                args=(experiment,),
            )
        with decision_tab:
            snapshot = experiment.get("decision_snapshot")
            if snapshot:
                st.success(f'{snapshot["decision"]} · saved by {snapshot["reviewer"]} on {snapshot["created_at"]}')
                st.write(snapshot.get("rationale", ""))
                st.caption(f'{len(experiment.get("frozen_snapshots", []))} frozen decision snapshot(s). Saved snapshots are never overwritten by later analysis.')
            with st.form(f'portfolio_decision_{experiment["id"]}'):
                decision = st.selectbox("Final decision", ["Launch", "Iterate", "Stop"])
                reviewer = st.text_input("Reviewer", value=experiment.get("owner", ""))
                rationale = st.text_area("Decision reason", placeholder="One sentence is enough.")
                save_decision = st.form_submit_button("Save and freeze snapshot", type="primary")
            if save_decision:
                upsert_experiment(freeze_decision_snapshot(experiment, decision, reviewer.strip(), rationale.strip()))
                st.rerun()
        with rollout_tab:
            st.caption("Turn the decision into four easy-to-review rollout checkpoints.")
            existing_plan = experiment.get("rollout_plan", [])
            if existing_plan:
                rollout_table = pd.DataFrame(existing_plan).rename(columns={"stage": "Step", "percentage": "Audience %", "date": "Planned date", "status": "Status"})
                st.dataframe(rollout_table, hide_index=True, use_container_width=True)
            rollout_left, rollout_right = st.columns(2)
            start = rollout_left.date_input("Rollout starts", value=pd.Timestamp.today().date(), key=f'rollout_start_{experiment["id"]}')
            cadence = rollout_right.selectbox("Time between checkpoints", [7, 14, 21, 28], format_func=lambda value: f"{value} days", key=f'rollout_cadence_{experiment["id"]}')
            if st.button("Build rollout plan", key=f'build_rollout_{experiment["id"]}'):
                updated = {**experiment, "rollout_plan": build_rollout_plan(start, [10, 25, 50, 100], cadence)}
                upsert_experiment(updated)
                st.rerun()


def show_overview() -> None:
    experiments = load_experiments()
    summary = portfolio_summary(experiments)
    title_col, action_col = st.columns([4.4, 1.25])
    title_col.markdown(
        '<div class="command-header"><div><h1>Experiment Command Center</h1>'
        '<p>See what is running, what needs attention, and what is ready for a decision.</p></div></div>',
        unsafe_allow_html=True,
    )
    if action_col.button("New Experiment", type="primary", use_container_width=True, key="portfolio_new_experiment"):
        create_experiment_dialog()
    st.markdown(
        f'<div class="command-kpis">'
        f'<div class="command-kpi"><div class="command-kpi-icon">RUN</div><div class="command-kpi-value">{summary["running"]}</div><div class="command-kpi-label">Experiments in market</div></div>'
        f'<div class="command-kpi"><div class="command-kpi-icon ready">NOW</div><div class="command-kpi-value">{summary["ready"]}</div><div class="command-kpi-label">Ready for a decision</div></div>'
        f'<div class="command-kpi"><div class="command-kpi-icon value">$</div><div class="command-kpi-value">{compact_money(float(summary["projected_value"]))}</div><div class="command-kpi-label">Projected portfolio value</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="command-section-head"><strong>Experiment journey</strong><span>Every experiment, from idea to rollout</span></div>', unsafe_allow_html=True)
    st.markdown(lifecycle_html(experiments), unsafe_allow_html=True)

    portfolio_col, attention_col = st.columns([3.4, 1.15])
    with portfolio_col:
        st.markdown('<div class="command-section-head"><strong>What is happening now</strong><span>Select a row for details</span></div>', unsafe_allow_html=True)
        filter_left, filter_right = st.columns(2)
        selected_status = filter_left.selectbox("Status", ["All statuses", *PORTFOLIO_STATUSES], label_visibility="collapsed", key="portfolio_status_filter")
        selected_type = filter_right.selectbox("Type", ["All types", "Customer", "Geography", "Time Series"], label_visibility="collapsed", key="portfolio_type_filter")
        visible = [item for item in experiments if (selected_status == "All statuses" or item["status"] == selected_status) and (selected_type == "All types" or item["type"] == selected_type)]
        table = pd.DataFrame(
            [
                {
                    "ID": item["id"],
                    "Experiment": item["name"],
                    "Status": item["status"],
                    "Progress": int(item.get("progress", 0)),
                }
                for item in visible
            ]
        )
        if table.empty:
            st.info("No experiments match these filters.")
        else:
            event = st.dataframe(
                table,
                hide_index=True,
                use_container_width=True,
                height=min(38 * (len(table) + 1), 338),
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "ID": None,
                    "Progress": st.column_config.ProgressColumn("Progress", min_value=0, max_value=100, format="%d%%"),
                },
                key="portfolio_experiment_table",
            )
            selected_rows = event.selection.rows if event and hasattr(event, "selection") else []
            if selected_rows:
                st.session_state.portfolio_selected_id = table.iloc[selected_rows[0]]["ID"]
    with attention_col:
        st.markdown('<div class="command-section-head"><strong>Needs your attention</strong><span>Act next</span></div>', unsafe_allow_html=True)
        render_attention_list(experiments)

    selected_id = st.session_state.portfolio_selected_id
    selected = next((item for item in experiments if item["id"] == selected_id), None)
    if selected is None:
        selected = next((item for item in experiments if item.get("alert")), experiments[0] if experiments else None)
    if selected:
        render_experiment_detail(selected)

    with st.expander("Portfolio value and recent decisions", expanded=False):
        value_col, decisions_col = st.columns([1.6, 1])
        with value_col:
            st.markdown("**Expected value over time**")
            st.plotly_chart(portfolio_value_chart(pd.DataFrame(value_timeline(experiments))), use_container_width=True, key="portfolio_value_chart")
        with decisions_col:
            st.markdown("**Latest decisions**")
            snapshots = [item for item in experiments if item.get("decision_snapshot")]
            feed = "".join(
                f'<div class="decision-feed-row"><div class="decision-dot"></div><div><b>{html.escape(item["name"])}</b>'
                f'<span>{html.escape(item["decision_snapshot"]["decision"])} · {html.escape(item.get("action", ""))}</span></div>'
                f'<time>{html.escape(item["decision_snapshot"]["created_at"])}</time></div>'
                for item in snapshots[:5]
            )
            st.markdown(f'<div class="decision-feed">{feed}</div>', unsafe_allow_html=True)


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
    render_business_guide(
        "Start with past customer data. We will identify the available products, audiences, outcomes, and strategy fields.",
        "Output: usable customer history",
    )
    st.session_state.experiment_name = st.text_input("Experiment Name", st.session_state.experiment_name, key="customer_experiment_name_v2")
    cfg: DataConfig = st.session_state.data_config
    source_options = ["Internal Data", "Upload File", "Synthetic Demo"]
    cfg.source_type = st.radio("Data Source", source_options, index=source_options.index(cfg.source_type) if cfg.source_type in source_options else 2, horizontal=True, key="customer_source_v2")
    if cfg.source_type == "Internal Data":
        c1, c2, c3 = st.columns(3)
        cfg.catalog = c1.text_input("Catalog", cfg.catalog, key="customer_catalog_v2")
        cfg.schema = c2.text_input("Schema", cfg.schema, key="customer_schema_v2")
        cfg.table = c3.text_input("Table", cfg.table, key="customer_table_v2")
        st.info("Internal Databricks access is not connected yet. General demo data is shown until the connector is configured.")
        source_signature = ("Internal Data", cfg.catalog, cfg.schema, cfg.table)
        if st.session_state.get("customer_active_data_signature") != source_signature:
            reset_customer_data_choices()
            st.session_state.raw_df = load_customer_demo("General Customer Test")
            apply_demo_metric_default("General Customer Test")
            st.session_state.customer_active_data_signature = source_signature
    elif cfg.source_type == "Upload File":
        uploaded = st.file_uploader("Upload customer data", type=["csv", "parquet"], key="customer_upload_v2")
        if uploaded is not None:
            try:
                signature = hash(uploaded.getvalue())
                if st.session_state.get("customer_upload_signature") != signature:
                    uploaded.seek(0)
                    uploaded_df = normalize_uploaded_dataset(pd.read_parquet(uploaded) if uploaded.name.endswith(".parquet") else load_uploaded_csv(uploaded))
                    reset_customer_data_choices()
                    st.session_state.raw_df = uploaded_df
                    st.session_state.customer_upload_signature = signature
                    st.session_state.customer_active_data_signature = ("Upload File", signature)
                    st.session_state.analysis_results_by_role = {}
                cfg.dataset_name = uploaded.name
            except Exception as exc:
                st.error("We could not read this file. Confirm that it is a valid CSV or Parquet dataset.")
                print(f"Customer upload parsing failed: {exc}")
        else:
            st.info("Upload a CSV or Parquet file to populate Groups, Metrics, and Strategy options.")
            return
    else:
        scenario = st.selectbox(
            "Demo Scenario",
            CUSTOMER_DEMO_SCENARIOS,
            index=CUSTOMER_DEMO_SCENARIOS.index(st.session_state.customer_demo_scenario) if st.session_state.customer_demo_scenario in CUSTOMER_DEMO_SCENARIOS else 0,
            key="customer_demo_scenario_v1",
        )
        source_signature = ("Synthetic Demo", scenario)
        if st.session_state.get("customer_active_data_signature") != source_signature:
            reset_customer_data_choices()
            st.session_state.raw_df = load_customer_demo(scenario)
            apply_demo_metric_default(scenario)
            st.session_state.customer_demo_scenario = scenario
            st.session_state.customer_active_data_signature = source_signature
        cfg.dataset_name = f"{scenario} Demo"

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
    primary = st.session_state.metrics_config["Primary"]
    primary.source_column = default_metric_source(raw_df, "Primary", primary.source_column)
    if not primary.source_column:
        st.error("No numeric outcome column was detected. Upload data with at least one numeric result field.")
        return

    mapping.data_structure = st.radio(
        "How is the customer data organized?",
        ["Longitudinal (unit x period)", "Cross-sectional (one row per unit)"],
        index=["Longitudinal (unit x period)", "Cross-sectional (one row per unit)"].index(mapping.data_structure),
        horizontal=True,
        format_func=lambda value: "Multiple rows per customer over time" if value.startswith("Longitudinal") else "One row per customer",
        key="customer_structure_v2",
    )
    available_segments = sorted(raw_df[mapping.cpc_column].dropna().astype(str).unique().tolist()) if mapping.cpc_column else []
    if available_segments:
        current_products = [value for value in population.selected_cpcs if value in available_segments]
        if not current_products:
            current_products = available_segments
        population.selected_cpcs = st.multiselect(
            "Products / Brands (CPC)",
            available_segments,
            default=current_products,
            help="Choose any combination of products or brands to include in historical exploration and experiment planning.",
            key="customer_product_scope_v3",
        )
        if not population.selected_cpcs:
            st.error("Select at least one product or brand to continue.")
            return
        st.caption(f"Exploring {len(population.selected_cpcs)} of {len(available_segments)} products / brands.")
    else:
        population.selected_cpcs = []
        st.caption("Product / brand scope: all available records")

    historical_period = "Not available"
    if mapping.booking_date_column and mapping.booking_date_column in columns:
        parsed = pd.to_datetime(raw_df[mapping.booking_date_column], errors="coerce")
        if parsed.notna().any():
            if mapping.data_structure == "Longitudinal (unit x period)" and mapping.mob_column in columns:
                periods = safe_numeric_series(raw_df, mapping.mob_column).fillna(0).astype(int)
                observation_periods = parsed.dt.to_period("M") + periods
                historical_start = observation_periods.min().to_timestamp()
                historical_end = observation_periods.max().to_timestamp()
            else:
                historical_start = parsed.min()
                historical_end = parsed.max()
            mapping.cutoff_date = historical_end.date().isoformat()
            historical_period = f"{historical_start:%b %Y} to {historical_end:%b %Y}"
    st.info(f"Historical data available: {historical_period}")

    with st.expander("Advanced data mapping", expanded=False):
        c1, c2 = st.columns(2)
        mapping.unit_id_column = c1.selectbox("Customer or Account ID", columns, index=columns.index(mapping.unit_id_column), key="customer_unit_map_v2")
        segment_options = ["None"] + columns
        selected_segment = c2.selectbox("Product / Brand Column (CPC)", segment_options, index=segment_options.index(mapping.cpc_column) if mapping.cpc_column in columns else 0, key="customer_segment_map_v2")
        mapping.cpc_column = "" if selected_segment == "None" else selected_segment
        if mapping.data_structure == "Longitudinal (unit x period)":
            c1, c2 = st.columns(2)
            mapping.mob_column = c1.selectbox("Time Since Customer Start", columns, index=columns.index(mapping.mob_column) if mapping.mob_column in columns else 0, key="customer_period_map_v2")
            date_options = ["None"] + dates
            selected_date = c2.selectbox("Acquisition Date Column", date_options, index=date_options.index(mapping.booking_date_column) if mapping.booking_date_column in dates else 0, key="customer_date_map_v2")
            mapping.booking_date_column = "" if selected_date == "None" else selected_date

    try:
        result = refresh_customer_processing()
    except Exception as exc:
        st.error(str(exc))
        return
    d = result.diagnostics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{d['Raw Records']:,.0f}")
    c2.metric("Customers / Accounts", f"{d['Unique Accounts']:,.0f}")
    c3.metric("Ready for Primary Outcome", f"{d['Final Analysis Population']:,.0f}")
    c4.metric("Products / Brands", f"{len(population.selected_cpcs):,.0f}" if available_segments else "All")
    visual_count = int(bool(mapping.cpc_column and mapping.cpc_column in raw_df.columns)) + int(bool(mapping.booking_date_column and mapping.booking_date_column in raw_df.columns))
    if visual_count:
        st.markdown(
            '<div class="customer-section-heading"><strong>Customer coverage</strong><span>See which products are represented and when customers entered the history.</span></div>',
            unsafe_allow_html=True,
        )
        visual_columns = st.columns(visual_count)
        visual_index = 0
        scoped_raw = raw_df
        if mapping.cpc_column and population.selected_cpcs:
            scoped_raw = raw_df[raw_df[mapping.cpc_column].astype(str).isin(population.selected_cpcs)]
        if mapping.cpc_column and mapping.cpc_column in raw_df.columns:
            with visual_columns[visual_index]:
                st.caption("Customer coverage by product or brand")
                st.plotly_chart(
                    customer_category_bar(scoped_raw, mapping.cpc_column, mapping.unit_id_column, "Product / Brand"),
                    use_container_width=True,
                    key="customer_product_coverage_chart_v1",
                )
            visual_index += 1
        if mapping.booking_date_column and mapping.booking_date_column in raw_df.columns:
            with visual_columns[visual_index]:
                st.caption("When customers or accounts entered the available history")
                st.plotly_chart(
                    customer_history_coverage(scoped_raw, mapping.booking_date_column, mapping.unit_id_column),
                    use_container_width=True,
                    key="customer_history_coverage_chart_v1",
                )
    if d["Final Analysis Population"] == 0:
        st.error("No units are eligible for the primary metric. Review the population and observation window.")
    with st.expander("Eligibility Details", expanded=False):
        st.write(f"Longest configured observation window: {d['Longest Required Horizon']} periods")
        st.write(f"Complete for every configured metric: {d['Complete Eligible Accounts']:,.0f}")
        st.write(f"Excluded for insufficient maturity: {d['Excluded for Insufficient Maturity']:,.0f}")
        st.write(f"Excluded from the all-metric cohort for incomplete history: {d['Excluded for Data Completeness']:,.0f}")

    st.markdown(
        '<div class="customer-section-heading"><strong>Customer data preview</strong><span>Inspect the source records or the analysis-ready customer view.</span></div>',
        unsafe_allow_html=True,
    )
    preview_mode = st.radio("Preview", ["Raw Data", "Processed Metrics", "Selected Raw Rows"], horizontal=True, label_visibility="collapsed", key="customer_preview_v2")
    if preview_mode == "Raw Data":
        preview = raw_df.head(20)
        caption = f"Showing 20 of {len(raw_df):,.0f} source rows · {len(raw_df.columns)} columns"
    elif preview_mode == "Selected Raw Rows":
        selected = selected_raw_preview(raw_df, mapping, population, st.session_state.metrics_config)
        preview = selected.head(20)
        caption = f"Showing 20 of {len(selected):,.0f} selected source rows · {len(selected.columns)} columns"
    else:
        display_names = {mapping.unit_id_column: "Customer / Account"}
        if mapping.cpc_column:
            display_names[mapping.cpc_column] = "Product / Brand"
        for metric in st.session_state.metrics_config.values():
            display_names[metric.processed_column] = f"{metric.name} ({metric_window_label(metric) if mapping.data_structure.startswith('Longitudinal') else 'As observed'})"
        preview = result.analysis_df.head(20).rename(columns=display_names)
        caption = f"Showing 20 of {len(result.analysis_df):,.0f} analysis-ready units"
    st.caption(caption)
    st.dataframe(preview, use_container_width=True, height=460)
    if not result.cpc_breakdown.empty:
        with st.expander("Product / Brand Breakdown", expanded=False):
            breakdown = result.cpc_breakdown.copy()
            breakdown["Share"] = breakdown["Share"].apply(lambda value: percent(value, 1))
            st.dataframe(breakdown, use_container_width=True, hide_index=True)


def customer_groups_step() -> None:
    render_business_guide(
        "Choose one customer characteristic only when the business needs a separate plan for each audience.",
        "Output: audiences planned separately",
    )
    raw_df = st.session_state.raw_df
    mapping: DataMappingConfig = st.session_state.data_mapping
    population: PopulationConfig = st.session_state.population_config
    strategy: StrategyConfig = st.session_state.strategy_config
    excluded = {
        mapping.unit_id_column,
        mapping.mob_column,
        mapping.booking_date_column,
        mapping.cpc_column,
        strategy.historical_column,
        *[metric.source_column for metric in st.session_state.metrics_config.values()],
    }
    excluded.update(date_like_columns(raw_df))
    candidates = grouping_candidates(raw_df, mapping.unit_id_column, excluded)
    treatment_fields = set(strategy_candidates(raw_df, mapping.unit_id_column, excluded))
    non_group_hints = ("offer", "line", "treatment", "strategy", "action", "experience", "incentive", "cost", "price", "fee", "channel", "message", "design", "reward")
    candidates = [
        column
        for column in candidates
        if column not in treatment_fields and not any(token in column.lower() for token in non_group_hints)
    ]
    if not candidates:
        st.info("No stable customer grouping fields were detected in the current data. The experiment will use all customers as one group.")
    else:
        st.caption(f"Detected {len(candidates)} eligible customer fields from the current dataset. This list updates when the data changes.")
    options = ["No grouping"] + candidates
    current = population.grouping_column if population.grouping_column in candidates else "No grouping"
    selected = st.selectbox(
        "Group Customers By",
        options,
        index=options.index(current),
        format_func=lambda value: value.replace("_", " ").title(),
        help="Use one baseline customer characteristic, such as FICO band, risk level, or revenue band.",
        key="customer_grouping_column_v1",
    )
    selected_column = "" if selected == "No grouping" else selected
    signature = (selected_column, tuple(raw_df.columns), len(raw_df))
    if st.session_state.get("customer_grouping_signature") != signature:
        population.grouping_column = selected_column
        population.group_settings = {}
        strategy.group_arms = {}
        if not selected_column:
            population.grouping_definitions = []
        else:
            unit_values = raw_df[[mapping.unit_id_column, selected_column]].drop_duplicates(mapping.unit_id_column)[selected_column]
            numeric_values = pd.to_numeric(unit_values, errors="coerce")
            is_numeric_group = numeric_values.notna().mean() > 0.95 and numeric_values.nunique() > 12
            if is_numeric_group:
                population.grouping_definitions = default_numeric_groups(numeric_values, selected_column)
            else:
                population.grouping_definitions = [
                    {"label": str(value), "value": str(value)}
                    for value in sorted(unit_values.dropna().astype(str).unique().tolist())
                ]
        st.session_state.customer_grouping_signature = signature

    if population.grouping_definitions and "lower" in population.grouping_definitions[0]:
        st.markdown("**Group Bands**")
        bands = pd.DataFrame(
            [
                {"Group": item["label"], "Minimum": float(item["lower"]), "Maximum": float(item["upper"])}
                for item in population.grouping_definitions
            ]
        )
        edited_bands = st.data_editor(
            bands,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "Group": st.column_config.TextColumn("Group", required=True),
                "Minimum": st.column_config.NumberColumn("Minimum", required=True),
                "Maximum": st.column_config.NumberColumn("Maximum", required=True),
            },
            key="customer_group_bands_v1",
        )
        valid_bands = edited_bands.dropna(subset=["Group", "Minimum", "Maximum"])
        population.grouping_definitions = [
            {"label": str(row["Group"]), "lower": float(row["Minimum"]), "upper": float(row["Maximum"])}
            for _, row in valid_bands.iterrows()
        ]

    try:
        refresh_customer_processing()
    except ValueError as exc:
        st.error(str(exc))
        return
    grouped = group_labeled_history()
    sync_group_settings(grouped)
    counts = grouped.groupby("_planning_group", sort=False).size().to_dict()
    rows = []
    for group, settings in population.group_settings.items():
        row = {
            "Group": group,
            "Historical Accounts": int(counts.get(group, 0)),
            f"Eligible Flow per {duration_unit(st.session_state.design_config.traffic_frequency).title()}": int(settings["eligible_flow"]),
        }
        if strategy.strategy_type == "Numeric Strategy":
            row["Minimum Strategy Value"] = float(settings["min_line"])
            row["Maximum Strategy Value"] = float(settings["max_line"])
        rows.append(row)
    st.markdown("**Business Constraints and Traffic**")
    flow_column = f"Eligible Flow per {duration_unit(st.session_state.design_config.traffic_frequency).title()}"
    column_config = {
        "Group": st.column_config.TextColumn("Group"),
        "Historical Accounts": st.column_config.NumberColumn("Historical Accounts", format="%d"),
        flow_column: st.column_config.NumberColumn(flow_column, min_value=1, required=True, format="%d"),
    }
    if strategy.strategy_type == "Numeric Strategy":
        column_config.update(
            {
                "Minimum Strategy Value": st.column_config.NumberColumn("Minimum Strategy Value", required=True),
                "Maximum Strategy Value": st.column_config.NumberColumn("Maximum Strategy Value", required=True),
            }
        )
    edited = st.data_editor(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=["Group", "Historical Accounts"],
        column_config=column_config,
        key="customer_group_settings_v1",
    )
    for _, row in edited.iterrows():
        group = str(row["Group"])
        settings = population.group_settings[group]
        settings["eligible_flow"] = int(row[flow_column])
        if strategy.strategy_type == "Numeric Strategy":
            settings["min_line"] = float(row["Minimum Strategy Value"])
            settings["max_line"] = float(row["Maximum Strategy Value"])
            if settings["min_line"] >= settings["max_line"]:
                st.error(f"{group}: minimum strategy value must be below maximum strategy value.")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.caption("Historical customers or accounts by audience")
        st.plotly_chart(
            customer_group_summary_chart(edited, "Historical Accounts", "Historical Customers / Accounts"),
            use_container_width=True,
            key="customer_group_history_chart_v1",
        )
    with chart_right:
        st.caption(f"Eligible customers arriving per {duration_unit(st.session_state.design_config.traffic_frequency)}")
        st.plotly_chart(
            customer_group_summary_chart(edited, flow_column, flow_column, color="#16856b"),
            use_container_width=True,
            key="customer_group_flow_chart_v1",
        )
    if strategy.strategy_type == "Numeric Strategy":
        st.caption("The business range limits numeric strategy recommendations. Sparse historical evidence produces a warning, but it does not silently change these limits.")
    else:
        st.caption("Eligible flow is allocated across the configured strategy arms so each customer group finishes at approximately the same time.")


def configure_metric_panel(role: str, raw_df: pd.DataFrame) -> None:
    numeric = numeric_columns(raw_df)
    cfg = st.session_state.metrics_config[role]
    cfg.source_column = default_metric_source(raw_df, role, cfg.source_column)
    c1, c2, c3, c4 = st.columns([1.3, 1.5, 1, 1])
    cfg.name = c1.text_input("Outcome Name", cfg.name, key=f"metric_panel_name_{role}")
    cfg.source_column = c2.selectbox("Data Column", numeric, index=numeric.index(cfg.source_column) if cfg.source_column in numeric else 0, key=f"metric_panel_source_{role}")
    cfg.metric_type = c3.selectbox(
        "Outcome Format",
        ["Continuous", "Binary"],
        index=["Continuous", "Binary"].index(cfg.metric_type),
        format_func=lambda value: "Number" if value == "Continuous" else "Yes / No",
        key=f"metric_panel_type_{role}",
    )
    cfg.direction = c4.selectbox("A Better Result Is", ["Higher is Better", "Lower is Better"], index=["Higher is Better", "Lower is Better"].index(cfg.direction), key=f"metric_panel_direction_{role}")
    if st.session_state.data_mapping.data_structure == "Longitudinal (unit x period)":
        mob_col = st.session_state.data_mapping.mob_column
        periods = sorted(pd.to_numeric(raw_df[mob_col], errors="coerce").dropna().astype(int).unique().tolist()) if mob_col in raw_df.columns else [12]
        cfg.mob_start = max(int(getattr(cfg, "mob_start", 1)), 1)
        c1, c2, c3 = st.columns(3)
        cfg.aggregation_method = c1.selectbox(
            "Combine Each Customer's Records By",
            ["Average", "Cumulative"],
            index=["Average", "Cumulative"].index(cfg.aggregation_method),
            format_func=lambda value: "Average" if value == "Average" else "Total",
            key=f"metric_panel_agg_{role}",
        )
        window_mode = c2.selectbox(
            "When Should It Be Measured?",
            ["Through MOB", "From MOB to MOB"],
            index=0 if cfg.mob_start == 1 else 1,
            format_func=lambda value: "From start through a month" if value == "Through MOB" else "Between two months",
            key=f"metric_panel_window_mode_{role}",
        )
        cfg.mob_horizon = int(c3.selectbox("End Month (MOB)", periods, index=periods.index(cfg.mob_horizon) if cfg.mob_horizon in periods else 0, key=f"metric_panel_window_end_{role}"))
        if window_mode == "From MOB to MOB":
            valid_starts = [period for period in periods if period <= cfg.mob_horizon]
            cfg.mob_start = int(c2.selectbox("Start Month (MOB)", valid_starts, index=valid_starts.index(cfg.mob_start) if cfg.mob_start in valid_starts else 0, key=f"metric_panel_window_start_{role}"))
        else:
            cfg.mob_start = 1
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
    render_business_guide(
        "Choose the business outcome that determines whether the experiment succeeds. Additional outcomes are optional.",
        "Output: success measure",
    )
    raw_df = st.session_state.raw_df
    if raw_df.empty:
        st.info("Load data first. Outcome options will update automatically from the dataset.")
        return
    numeric = numeric_columns(raw_df)
    if not numeric:
        st.error("No numeric outcome columns were detected in the current dataset.")
        return
    st.caption(f"Choose the main business outcome. {len(numeric)} numeric fields were detected in the current data.")
    st.markdown("**Main Outcome**")
    configure_metric_panel("Primary", raw_df)
    with st.expander("Add another outcome or safety limit", expanded=False):
        st.caption("Keep this closed unless the experiment needs another business outcome or a risk limit.")
        c1, c2 = st.columns(2)
        secondary_enabled = c1.checkbox("Add Another Outcome", value="Secondary" in st.session_state.metrics_config, key="enable_secondary_v3")
        guardrail_enabled = c2.checkbox("Add Safety Outcome", value="Guardrail" in st.session_state.metrics_config, key="enable_guardrail_v3")
        if secondary_enabled:
            secondary_source = next((column for column in numeric if column != st.session_state.metrics_config["Primary"].source_column), numeric[0])
            st.session_state.metrics_config.setdefault(
                "Secondary",
                MetricConfig("Secondary Outcome", secondary_source, "Secondary", effect_type="Relative %", effect_value=0.15, source_column=secondary_source, aggregation_method="Average", mob_horizon=12),
            )
            st.markdown("**Additional Outcome**")
            configure_metric_panel("Secondary", raw_df)
        else:
            st.session_state.metrics_config.pop("Secondary", None)
        if guardrail_enabled:
            used = {metric.source_column for metric in st.session_state.metrics_config.values()}
            guardrail_source = next((column for column in numeric if column not in used), numeric[0])
            st.session_state.metrics_config.setdefault(
                "Guardrail",
                MetricConfig("Guardrail Outcome", guardrail_source, "Guardrail", direction="Lower is Better", effect_type="Relative %", effect_value=0.15, guardrail_threshold=0.15, source_column=guardrail_source, aggregation_method="Average", mob_horizon=12),
            )
            st.markdown("**Safety Outcome**")
            configure_metric_panel("Guardrail", raw_df)
        else:
            st.session_state.metrics_config.pop("Guardrail", None)
    roles = [role for role in ["Primary", "Secondary", "Guardrail"] if role in st.session_state.metrics_config]
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
        summary["Historical Variation"] = "Calculated from similar past strategy values when building the plan"
    st.markdown("**Historical Snapshot**")
    display_summary = summary.rename(
        columns={
            "Metric": "Outcome",
            "Role": "Use",
            "Aggregation": "How It Is Calculated",
            "Observation Window": "When It Is Measured",
            "Baseline": "Historical Average",
            "SD": "Typical Variation",
            "Eligible N": "Usable Customers / Accounts",
        }
    )
    st.dataframe(display_summary, use_container_width=True, hide_index=True)
    if summary["Eligible N"].nunique() > 1:
        st.caption("Eligible sample sizes differ because each metric uses its own observation window. The Plan uses all configured metrics by default.")

    st.markdown("**What Have These Outcomes Looked Like Historically?**")
    for index in range(0, len(roles), 2):
        visible_roles = roles[index : index + 2]
        distribution_columns = st.columns(len(visible_roles))
        for column, role in zip(distribution_columns, visible_roles):
            with column:
                metric = st.session_state.metrics_config[role]
                metric_column = metric.processed_column or metric.column
                st.caption(f"{metric.name} · {metric_window_label(metric)}")
                st.plotly_chart(
                    customer_metric_distribution(result.analysis_df, metric_column, metric.name, metric.metric_type),
                    use_container_width=True,
                    key=f"customer_metric_distribution_{role}_v1",
                )

    mapping: DataMappingConfig = st.session_state.data_mapping
    population: PopulationConfig = st.session_state.population_config
    strategy: StrategyConfig = st.session_state.strategy_config
    excluded = {
        mapping.unit_id_column,
        mapping.cpc_column,
        mapping.mob_column,
        mapping.booking_date_column,
        population.grouping_column,
        *[metric.source_column for metric in st.session_state.metrics_config.values()],
    }
    excluded.update(date_like_columns(raw_df))
    comparison_candidates = strategy_candidates(raw_df, mapping.unit_id_column, excluded)
    if comparison_candidates:
        st.markdown("**Outcome by Historical Strategy**")
        preferred_strategy = strategy.historical_column if strategy.historical_column in comparison_candidates else comparison_candidates[0]
        stored_strategy = st.session_state.get("outcome_strategy_view_v1")
        if stored_strategy not in comparison_candidates:
            st.session_state.pop("outcome_strategy_view_v1", None)
        grouped_history = group_labeled_history()
        group_options = ["All groups"] + grouped_history["_planning_group"].astype(str).drop_duplicates().tolist()
        selector_left, selector_right = st.columns([1.25, 1])
        selected_strategy = selector_left.selectbox(
            "Test Strategy",
            comparison_candidates,
            index=comparison_candidates.index(preferred_strategy),
            format_func=humanize_column_name,
            help="Choose the historical strategy to place on the horizontal axis. Confirm the actual experiment strategy later in Define Options.",
            key="outcome_strategy_view_v1",
        )
        selected_group = selector_right.selectbox("Customer Group", group_options, key="metric_history_group_v2")

        if selected_strategy not in grouped_history.columns:
            strategy_values = fixed_unit_values(raw_df, mapping.unit_id_column, [selected_strategy]).reset_index()
            grouped_history = grouped_history.merge(strategy_values, on=mapping.unit_id_column, how="left")
        chart_history = grouped_history if selected_group == "All groups" else grouped_history[grouped_history["_planning_group"].astype(str) == selected_group]
        numeric_strategy = pd.to_numeric(chart_history[selected_strategy], errors="coerce")
        is_numeric_strategy = numeric_strategy.notna().mean() >= 0.95
        binning_method = "Automatic fine bins"
        bin_width = float(strategy.curve_bin_width)
        bin_count = int(strategy.curve_bin_count)
        if is_numeric_strategy:
            with st.expander("Chart settings", expanded=False):
                setting_left, setting_right = st.columns(2)
                binning_options = ["Automatic fine bins", "Fixed bin width", "Equal-count bins"]
                binning_method = setting_left.selectbox("How to group strategy values", binning_options, key="metric_curve_binning_v5")
                if binning_method == "Fixed bin width":
                    bin_width = setting_right.number_input(
                        "Bin Width",
                        min_value=0.01,
                        value=max(float(strategy.curve_bin_width), 0.01),
                        step=max(float(strategy.increment), 0.01),
                        key="metric_curve_width_v5",
                    )
                elif binning_method == "Equal-count bins":
                    bin_count = int(setting_right.number_input("Number of Bins", min_value=4, max_value=50, value=max(4, int(strategy.curve_bin_count)), step=1, key="metric_curve_count_v5"))
                else:
                    setting_right.caption("Automatically uses enough bins to reveal the pattern without making the chart noisy.")

        strategy_label = humanize_column_name(selected_strategy)
        for index in range(0, len(roles), 2):
            visible_roles = roles[index : index + 2]
            chart_columns = st.columns(len(visible_roles))
            for column, role in zip(chart_columns, visible_roles):
                with column:
                    metric = st.session_state.metrics_config[role]
                    metric_column = metric.processed_column or metric.column
                    st.caption(f"{metric.name} · {metric_window_label(metric)}")
                    if is_numeric_strategy:
                        figure = historical_association(
                            chart_history,
                            selected_strategy,
                            metric_column,
                            metric.metric_type,
                            strategy_label=strategy_label,
                            metric_label=metric.name,
                            binning_method=binning_method,
                            bin_width=bin_width,
                            bin_count=bin_count,
                        )
                    else:
                        figure = customer_strategy_outcome_chart(
                            chart_history,
                            selected_strategy,
                            metric_column,
                            strategy_label,
                            metric.name,
                            metric.metric_type,
                        )
                    st.plotly_chart(figure, use_container_width=True, key=f"outcome_by_strategy_{role}_v1")
        st.caption("Historical association only. Customers were not randomized across these strategy values, so the chart supports option selection but does not prove causation. Vertical intervals show 95% confidence intervals.")
    else:
        st.info("No stable historical strategy field was detected. You can still define a new strategy in Define Options.")

    grouped_history = group_labeled_history()
    with st.expander("Group-Level Historical Summary", expanded=False):
        group_rows = []
        for group, scoped in grouped_history.groupby("_planning_group", sort=False):
            for role in roles:
                metric = st.session_state.metrics_config[role]
                values = pd.to_numeric(scoped[metric.processed_column or metric.column], errors="coerce").dropna()
                group_rows.append(
                    {
                        "Group": group,
                        "Role": role,
                        "Metric": metric.name,
                        "Historical Accounts": len(values),
                        "Average": values.mean(),
                        "SD": values.std(ddof=1),
                    }
                )
        shown_groups = pd.DataFrame(group_rows)
        shown_groups["Average"] = shown_groups["Average"].round(2)
        shown_groups["SD"] = shown_groups["SD"].round(2)
        st.dataframe(shown_groups, use_container_width=True, hide_index=True)


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
    render_business_guide(
        "Define what stays as business as usual and which alternatives customers may receive.",
        "Output: current and test options",
    )
    strategy: StrategyConfig = st.session_state.strategy_config
    raw_df = st.session_state.raw_df
    mapping: DataMappingConfig = st.session_state.data_mapping
    population: PopulationConfig = st.session_state.population_config
    if raw_df.empty or mapping.unit_id_column not in raw_df.columns:
        st.info("Load data first. Strategy options will be detected from stable customer fields.")
        return
    excluded = {
        mapping.unit_id_column,
        mapping.cpc_column,
        mapping.mob_column,
        mapping.booking_date_column,
        population.grouping_column,
        *[metric.source_column for metric in st.session_state.metrics_config.values()],
    }
    excluded.update(date_like_columns(raw_df))
    candidates = strategy_candidates(raw_df, mapping.unit_id_column, excluded)
    new_strategy = "New strategy (not in historical data)"
    options = [new_strategy] + candidates
    outcome_strategy = st.session_state.get("outcome_strategy_view_v1")
    current = strategy.historical_column if strategy.historical_column in candidates else outcome_strategy if outcome_strategy in candidates else new_strategy
    selected = st.selectbox(
        "Which field describes the current strategy?",
        options,
        index=options.index(current),
        format_func=lambda value: value if value == new_strategy else humanize_column_name(value),
        help="The list contains fields that are stable for each customer in the current dataset. Choose New strategy when the proposed treatment has no historical equivalent.",
        key="strategy_data_field_v2",
    )
    signature = (st.session_state.get("customer_active_data_signature"), selected)
    if st.session_state.get("customer_strategy_field_signature") != signature:
        if selected == new_strategy:
            apply_experiment_template(strategy, "Custom")
        else:
            apply_strategy_column_defaults(strategy, raw_df, mapping.unit_id_column, selected)
        for key in ["strategy_goal_v1", "strategy_dimension_v6", "strategy_assignment_column_v6", "strategy_value_label_v1", "strategy_value_format_v1", "strategy_min_v6", "strategy_max_v6", "strategy_increment_v6"]:
            st.session_state.pop(key, None)
        st.session_state.customer_strategy_field_signature = signature
        st.session_state.design_config.group_plan_rows = []
        st.session_state.group_analysis_results = {}
        st.session_state.group_response_results = {}
    if selected == new_strategy:
        st.caption("No historical treatment field will be used. Planning relies on the selected customer groups and outcome history.")
        format_options = ["Named variants", "Ordered numeric levels"]
        strategy.arm_format = st.radio(
            "Treatment Format",
            format_options,
            index=format_options.index(strategy.arm_format) if strategy.arm_format in format_options else 0,
            horizontal=True,
            format_func=lambda value: "Named options" if value == "Named variants" else "Numeric values",
            key="strategy_new_format_v1",
        )
        strategy.strategy_type = "Numeric Strategy" if strategy.arm_format == "Ordered numeric levels" else "Categorical Strategy"
        strategy.historical_column = ""
        strategy.historical_evidence = "Use group-level outcome history"
    else:
        detected_type = "Numeric values" if strategy.strategy_type == "Numeric Strategy" else "Named options"
        st.success(f"Detected {detected_type.lower()} from {humanize_column_name(selected)}. Suggested arms below use values found in the data.")
    strategy.strategy_goal = st.text_input(
        "What business decision should this experiment support?",
        strategy.strategy_goal,
        key="strategy_goal_v1",
    )
    with st.expander("Advanced strategy settings", expanded=False):
        setup_left, setup_right = st.columns(2)
        strategy.strategy_name = setup_left.text_input("Strategy Dimension", strategy.strategy_name, key="strategy_dimension_v6")
        strategy.assignment_column = setup_right.text_input(
            "Result Assignment Column",
            strategy.assignment_column,
            help="The post-launch result file should contain this randomized-arm field.",
            key="strategy_assignment_column_v6",
        )
        st.write("Historical evidence: " + (humanize_column_name(strategy.historical_column) if strategy.historical_column else "Group-level outcome history"))

    st.markdown("**Test Options**")
    render_arm_overview(strategy)
    if strategy.strategy_type == "Numeric Strategy":
        with st.expander("Numeric value settings", expanded=False):
            value_left, value_right = st.columns(2)
            strategy.value_label = value_left.text_input("Value Label", strategy.value_label or "Strategy Value", key="strategy_value_label_v1")
            formats = ["Currency", "Percent", "Number"]
            strategy.value_format = value_right.selectbox("Value Display", formats, index=formats.index(strategy.value_format) if strategy.value_format in formats else 2, key="strategy_value_format_v1")
            c1, c2, c3 = st.columns(3)
            strategy.min_value = c1.number_input("Minimum Allowed Value", value=float(strategy.min_value), key="strategy_min_v6")
            strategy.max_value = c2.number_input("Maximum Allowed Value", value=float(strategy.max_value), key="strategy_max_v6")
            strategy.increment = c3.number_input("Allowed Increment", min_value=0.0001, value=float(strategy.increment), key="strategy_increment_v6")
        rows = [{"Role": "Control", "Arm Name": strategy.control_name, strategy.value_label: float(strategy.control_value)}]
        rows.extend(
            {
                "Role": "Treatment",
                "Arm Name": strategy.treatment_names[index] if index < len(strategy.treatment_names) else f"Treatment {index + 1}",
                strategy.value_label: float(value),
            }
            for index, value in enumerate(strategy.treatment_values)
        )
        edited = st.data_editor(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            disabled=["Role"],
            column_config={
                "Role": st.column_config.TextColumn("Type"),
                "Arm Name": st.column_config.TextColumn("Option Name", required=True),
                strategy.value_label: st.column_config.NumberColumn(strategy.value_label, required=True, step=float(strategy.increment)),
            },
            key=f"generic_numeric_arms_{selected}_v2",
        )
        clean = edited.dropna(subset=["Arm Name", strategy.value_label])
        clean = clean[clean["Arm Name"].astype(str).str.strip() != ""]
        if len(clean) >= 2:
            strategy.control_name = str(clean.iloc[0]["Arm Name"])
            strategy.control_value = float(clean.iloc[0][strategy.value_label])
            strategy.treatment_names = clean.iloc[1:]["Arm Name"].astype(str).tolist()
            strategy.treatment_values = clean.iloc[1:][strategy.value_label].astype(float).tolist()
            strategy.number_of_arms = len(clean)
        for error in validate_numeric_strategy(float(strategy.control_value), [float(value) for value in strategy.treatment_values], float(strategy.min_value), float(strategy.max_value), float(strategy.increment)):
            st.error(error)
    else:
        rows = [{"Role": "Control", "Arm Name": strategy.categorical_control, "Description": strategy.arm_descriptions.get(strategy.categorical_control, "")}]
        rows.extend(
            {"Role": "Treatment", "Arm Name": arm, "Description": strategy.arm_descriptions.get(arm, "")}
            for arm in strategy.categorical_treatments
        )
        edited = st.data_editor(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            disabled=["Role"],
            column_config={
                "Role": st.column_config.TextColumn("Type"),
                "Arm Name": st.column_config.TextColumn("Option Name", required=True),
                "Description": st.column_config.TextColumn("What the customer receives"),
            },
            key=f"generic_named_arms_{selected}_v2",
        )
        clean = edited.dropna(subset=["Arm Name"])
        clean = clean[clean["Arm Name"].astype(str).str.strip() != ""]
        if len(clean) >= 2:
            strategy.categorical_control = str(clean.iloc[0]["Arm Name"])
            strategy.categorical_treatments = clean.iloc[1:]["Arm Name"].astype(str).tolist()
            strategy.arm_descriptions = {str(row["Arm Name"]): str(row.get("Description", "")) for _, row in clean.iterrows()}
            strategy.number_of_arms = len(clean)
        for error in validate_categorical_strategy(strategy.categorical_control, strategy.categorical_treatments):
            st.error(error)
    control, treatments = configured_arms(strategy)
    treatment_labels = (
        [strategy.treatment_names[index] if index < len(strategy.treatment_names) else f"Test {index + 1}" for index in range(len(treatments))]
        if strategy.strategy_type == "Numeric Strategy"
        else [str(value) for value in treatments]
    )
    st.markdown("**Experiment at a Glance**")
    st.caption("Blue is the current / BAU experience. Other colors are the options being tested.")
    st.plotly_chart(
        customer_option_chart(
            control,
            list(treatments),
            strategy.control_name if strategy.strategy_type == "Numeric Strategy" else str(control),
            treatment_labels,
            strategy.value_label if strategy.strategy_type == "Numeric Strategy" else strategy.strategy_name,
            numeric=strategy.strategy_type == "Numeric Strategy",
        ),
        use_container_width=True,
        key="customer_option_chart_v1",
    )
    st.markdown(
        f'<div class="next-step"><strong>{len(treatments) + 1} options ready.</strong> '
        'Continue to Get Plan to calculate required accounts, traffic allocation, and test duration.</div>',
        unsafe_allow_html=True,
    )


def render_customer_plan(
    plan_rows: list[dict],
    comparisons: list[dict],
    binding_labels: list[str],
    group_names: list[str],
    methodology: str,
) -> None:
    design: DesignConfig = st.session_state.design_config
    strategy: StrategyConfig = st.session_state.strategy_config
    if not plan_rows:
        design.group_plan_rows = []
        design.required_n_per_arm = 0
        design.total_sample_size = 0
        return
    signature = tuple(
        (str(row["Group"]), str(row["Arm"]), int(row["Required Accounts"]), round(float(row["Traffic Allocation"]), 8))
        for row in plan_rows
    )
    if st.session_state.get("customer_group_plan_signature") not in {None, signature}:
        st.session_state.group_analysis_results = {}
        st.session_state.group_response_results = {}
    st.session_state.customer_group_plan_signature = signature
    design.group_plan_rows = plan_rows
    design.total_sample_size = sum(int(row["Required Accounts"]) for row in plan_rows)
    design.required_n_per_arm = max(int(row["Required Accounts"]) for row in plan_rows)
    design.binding_metric = "; ".join(binding_labels)
    total_duration = max(float(row["Test Duration"]) for row in plan_rows)
    arm_count = len({str(row["Arm"]) for row in plan_rows})
    within_window = total_duration <= design.max_enrollment_periods
    render_customer_kpi_panel(
        "Recommended Test Plan",
        "The smallest complete design across all selected customer groups and configured metrics.",
        "Ready to review" if within_window else "Needs adjustment",
        [
            ("Required Accounts", f"{design.total_sample_size:,.0f}", "treatment"),
            ("Longest Test Duration", duration_label(int(math.ceil(total_duration)), design.traffic_frequency), ""),
            ("Customer Groups", f"{len(group_names):,.0f}", ""),
            ("Test Options", f"{arm_count:,.0f}", ""),
        ],
    )
    primary = st.session_state.metrics_config["Primary"]
    if primary.effect_type in {"Relative %", "Percentage Point"}:
        tradeoff_effect = abs(float(primary.effect_value)) * 100
        tradeoff_suffix = "%"
    else:
        tradeoff_effect = abs(float(primary.effect_value))
        tradeoff_suffix = ""
    st.markdown("**Test Size Trade-Off**")
    st.caption("Explore the core planning trade-off: detecting a smaller improvement requires more accounts and a longer test.")
    st.plotly_chart(
        customer_design_tradeoff_chart(
            tradeoff_effect,
            design.total_sample_size,
            total_duration,
            duration_unit(design.traffic_frequency),
            tradeoff_suffix,
        ),
        use_container_width=True,
        key="customer_design_tradeoff_v1",
    )
    st.caption("The curve holds the current audiences, test options, variability, and traffic pattern fixed. It uses the standard inverse-square planning relationship around the current design.")
    raw_output = pd.DataFrame(plan_rows)[["Group", "Arm", "Role", "Required Accounts", "Traffic Allocation", "Flow per Period", "Test Duration"]].copy()
    raw_output["Test Option"] = raw_output["Arm"].map(lambda value: format_strategy_value(value, strategy))
    st.markdown("**How the Test Will Be Sized**")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.caption("Accounts required for every audience and test option")
        st.plotly_chart(
            customer_required_accounts_chart(raw_output),
            use_container_width=True,
            key="customer_required_accounts_chart_v1",
        )
    with chart_right:
        st.caption("How incoming eligible traffic should be split")
        st.plotly_chart(
            customer_traffic_allocation_chart(raw_output),
            use_container_width=True,
            key="customer_traffic_allocation_chart_v1",
        )
    output = raw_output.drop(columns=["Arm"]).copy()
    output["Traffic Allocation"] = output["Traffic Allocation"].map(lambda value: percent(value, 1))
    flow_label = f"Flow per {duration_unit(design.traffic_frequency).title()}"
    output["Flow per Period"] = output["Flow per Period"].map(lambda value: f"{value:,.1f}")
    output["Test Duration"] = output["Test Duration"].map(lambda value: duration_label(int(math.ceil(value)), design.traffic_frequency))
    output = output.rename(
        columns={
            "Role": "Type",
            "Traffic Allocation": "Share of Eligible Traffic",
            "Flow per Period": flow_label,
        }
    )
    output = output[["Group", "Test Option", "Type", "Required Accounts", "Share of Eligible Traffic", flow_label, "Test Duration"]]
    st.dataframe(output, use_container_width=True, hide_index=True)
    weak_support = [
        row for row in plan_rows
        if str(row.get("Support", "")) in {"Limited", "Insufficient"}
    ]
    if weak_support:
        affected = sorted({f"{row['Group']} · {format_strategy_value(row['Arm'], strategy)}" for row in weak_support})
        st.warning(
            "Historical variability is based on limited evidence for "
            + ", ".join(affected[:5])
            + (" and additional options." if len(affected) > 5 else ".")
            + " Review the advanced calculation details before launch."
        )
    if total_duration > design.max_enrollment_periods:
        st.warning(
            f"This design needs about {duration_label(int(math.ceil(total_duration)), design.traffic_frequency)}, "
            f"which is longer than the {duration_label(design.max_enrollment_periods, design.traffic_frequency)} enrollment window. "
            "Reduce the number of test options, expand eligible traffic, increase the decision threshold, or allow a longer test."
        )
    else:
        st.success(f"All groups can collect their required samples within {duration_label(design.max_enrollment_periods, design.traffic_frequency)}.")
    with st.expander("How this plan was calculated (advanced)", expanded=False):
        st.write(methodology)
        if design.audience_decision_scope == "One joint decision across audiences":
            st.write("The false-positive budget is protected across all customer audiences and treatment-versus-BAU comparisons in this plan.")
        else:
            st.write("Each customer audience is treated as a separate decision family; protection applies across treatment-versus-BAU comparisons within that audience.")
        st.write("Traffic is allocated from the required account targets so options within each customer group finish at approximately the same time.")
        if comparisons:
            detail = pd.DataFrame(comparisons)
            if "Planned Power" in detail.columns:
                detail["Planned Power"] = detail["Planned Power"].map(lambda value: percent(value, 1))
            st.dataframe(detail, use_container_width=True, hide_index=True)


def customer_plan_step() -> None:
    strategy: StrategyConfig = st.session_state.strategy_config
    design: DesignConfig = st.session_state.design_config
    primary = st.session_state.metrics_config["Primary"]
    historical_df = st.session_state.historical_df
    if historical_df.empty:
        st.info("Complete Bring Data, Choose Audience, and Pick Outcome before building the plan.")
        return

    render_business_guide(
        "Set the smallest improvement worth acting on and the longest test the business can run.",
        "Output: accounts, traffic, and duration",
    )
    st.markdown("**What Makes This Test Worth Running?**")
    st.caption("Use a decision threshold, not a forecast. A smaller improvement requires more customers.")
    c1, c2, c3 = st.columns(3)
    percentage_effect = primary.effect_type in {"Relative %", "Percentage Point"}
    shown_effect = c1.number_input(
        "Smallest Change Worth Acting On (%)" if percentage_effect else "Smallest Change Worth Acting On",
        min_value=0.1 if percentage_effect else 0.0001,
        max_value=100.0 if percentage_effect else None,
        value=float(primary.effect_value) * 100 if percentage_effect else float(primary.effect_value),
        step=0.5 if percentage_effect else max(abs(float(primary.effect_value)) * 0.05, 0.01),
        help=(
            "The smallest relative difference from BAU that would justify a business decision. Smaller effects require more accounts. This is a planning threshold, not a forecast."
            if primary.effect_type == "Relative %"
            else "The smallest absolute difference from BAU that would justify a business decision. This is a planning threshold, not a forecast."
        ),
        key=f"group_plan_mde_v2_{primary.effect_type}",
    )
    primary.effect_value = shown_effect / 100 if percentage_effect else shown_effect
    frequency_options = ["Monthly", "Weekly", "Daily"]
    design.traffic_frequency = c2.selectbox(
        "How Often Eligible Customers Arrive",
        frequency_options,
        index=frequency_options.index(design.traffic_frequency) if design.traffic_frequency in frequency_options else 0,
        key="group_plan_frequency_v1",
    )
    default_window = int(primary.mob_horizon) if st.session_state.data_mapping.data_structure.startswith("Longitudinal") else 12
    if design.max_enrollment_periods <= 0:
        design.max_enrollment_periods = default_window
    design.max_enrollment_periods = int(
        c3.number_input(
            f"Longest Test You Can Run ({duration_unit(design.traffic_frequency)}s)",
            min_value=1,
            value=int(design.max_enrollment_periods),
            step=1,
            help="The longest period available to enroll the required accounts. It does not include outcome maturity time.",
            key="group_plan_max_duration_v1",
        )
    )
    st.markdown(
        f'<div class="next-step"><strong>Planning target:</strong> detect a {shown_effect:.1f}{"%" if percentage_effect else ""} change '
        f'within {design.max_enrollment_periods} {duration_unit(design.traffic_frequency)}s. '
        'The app will recommend accounts and traffic for every customer group.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Advanced statistical settings", expanded=False):
        st.caption("These defaults are suitable for most experiments. Change them only when the experiment has a reviewed statistical protocol.")
        a1, a2 = st.columns(2)
        design.alpha = a1.number_input(
            "False-positive Rate",
            min_value=0.001,
            max_value=0.25,
            value=float(design.alpha),
            step=0.005,
            format="%.3f",
            help="Probability of declaring an effect when no real effect exists. The default 5% is standard for most business experiments.",
            key="group_plan_alpha_v2",
        )
        design.target_power = a2.number_input(
            "Detection Chance",
            min_value=0.5,
            max_value=0.99,
            value=float(design.target_power),
            step=0.01,
            format="%.2f",
            help="Chance of detecting the selected minimum effect when it is real. Higher values require more accounts.",
            key="group_plan_power_v2",
        )
        design.sample_size_basis = st.selectbox(
            "Metrics Included in Sample Planning",
            ["Power All Configured Metrics", "Primary Metric Only"],
            index=0 if design.sample_size_basis == "Power All Configured Metrics" else 1,
            help="Using all metrics makes the sample large enough for every configured decision metric.",
            key="group_plan_metric_basis_v2",
        )
        design.multiplicity_method = st.selectbox(
            "Protection Across Test Options",
            ["Holm", "None"],
            index=0 if design.multiplicity_method == "Holm" else 1,
            format_func=lambda value: "Conservative family-wise protection" if value == "Holm" else "No adjustment",
            help="Planning reserves a conservative share of the false-positive budget for every treatment-versus-BAU comparison. Completed-result p-values use Holm step-down adjustment.",
            key="group_plan_multiplicity_v2",
        )
        design.audience_decision_scope = st.selectbox(
            "Audience Decision Scope",
            ["Separate decision per audience", "One joint decision across audiences"],
            index=0 if design.audience_decision_scope == "Separate decision per audience" else 1,
            help="Choose separate decisions when each audience can launch independently. Choose one joint decision when any audience finding will support the same overall claim.",
            key="group_plan_audience_scope_v1",
        )
        design.attrition_rate = st.number_input(
            "Expected Missing Outcomes",
            min_value=0.0,
            max_value=0.8,
            value=float(design.attrition_rate),
            step=0.01,
            format="%.2f",
            help="Expected share of enrolled accounts that will not have a usable outcome at analysis time.",
            key="group_plan_attrition_v2",
        )
        if strategy.strategy_type == "Numeric Strategy" and strategy.historical_evidence == "Use historical strategy values":
            strategy.max_mapping_distance = st.number_input(
                "Maximum Historical Distance from a Test Value",
                min_value=0.0,
                value=float(strategy.max_mapping_distance),
                step=max(float(strategy.increment), 0.01),
                help="Use 0 for automatic protection: half of the widest gap between proposed values. More distant historical customers are excluded from the arm-specific mean and SD.",
                key="group_plan_max_mapping_distance_v1",
            )
        st.markdown("**Metric-specific thresholds**")
        st.caption("The primary metric uses the business target above. Existing units for additional outcomes and safety limits are preserved.")
        threshold_rows = []
        for metric in st.session_state.metrics_config.values():
            if metric.role != "Primary":
                threshold_rows.append(
                    {
                        "Role": metric.role,
                        "Metric": metric.name,
                        "Scale": metric.effect_type,
                        "Decision Threshold": float(metric.effect_value) * 100 if metric.effect_type in {"Relative %", "Percentage Point"} else float(metric.effect_value),
                    }
                )
        if threshold_rows:
            edited_thresholds = st.data_editor(
                pd.DataFrame(threshold_rows),
                use_container_width=True,
                hide_index=True,
                disabled=["Role", "Metric", "Scale"],
                column_config={"Decision Threshold": st.column_config.NumberColumn("Decision Threshold", min_value=0.0001, required=True)},
                key="group_metric_thresholds_v2",
            )
            for _, row in edited_thresholds.iterrows():
                metric = st.session_state.metrics_config[str(row["Role"])]
                metric.effect_value = float(row["Decision Threshold"]) / 100 if metric.effect_type in {"Relative %", "Percentage Point"} else float(row["Decision Threshold"])
                if metric.role == "Guardrail":
                    metric.guardrail_threshold = metric.effect_value
        else:
            st.caption("No additional outcomes or safety limits are configured.")

    use_point_history = (
        strategy.strategy_type == "Numeric Strategy"
        and strategy.historical_evidence == "Use historical strategy values"
        and strategy.historical_column in st.session_state.raw_df.columns
    )
    if not use_point_history:
        try:
            refresh_customer_processing()
        except ValueError as exc:
            st.error(str(exc))
            return
        grouped = group_labeled_history()
        sync_group_settings(grouped)
        population: PopulationConfig = st.session_state.population_config
        group_names = list(population.group_settings)
        control, treatments = configured_arms(strategy)
        if not treatments:
            st.error("Configure at least one treatment arm in Strategy before planning.")
            return
        family_comparisons = planning_comparison_count(
            len(treatments),
            len(group_names),
            design.audience_decision_scope,
        )
        st.markdown("**Test Options**")
        arm_rows = [{"Role": "Control", "Strategy Arm": format_strategy_value(control, strategy)}]
        arm_rows.extend({"Role": "Treatment", "Strategy Arm": format_strategy_value(arm, strategy)} for arm in treatments)
        shown_arms = pd.DataFrame(arm_rows)
        if strategy.strategy_type == "Categorical Strategy":
            shown_arms["What the customer receives"] = [strategy.arm_descriptions.get(str(arm), "") for arm in [control, *treatments]]
        st.dataframe(shown_arms, use_container_width=True, hide_index=True)
        st.caption("These options are applied consistently across customer groups. Required accounts and traffic are calculated separately for each group.")

        all_plan_rows: list[dict] = []
        all_comparisons: list[dict] = []
        binding_labels: list[str] = []
        group_tabs = st.tabs(group_names)
        for tab, group in zip(group_tabs, group_names):
            with tab:
                settings = population.group_settings[group]
                scoped = grouped[grouped["_planning_group"].astype(str) == group].copy()
                s1, s2 = st.columns(2)
                s1.metric("Historical Accounts", f"{len(scoped):,.0f}")
                s2.metric(f"Eligible Flow / {duration_unit(design.traffic_frequency).title()}", f"{settings['eligible_flow']:,.0f}")
                strategy.group_arms[group] = {"control": control, "treatments": list(treatments)}
                try:
                    plan_rows, comparisons, binding = calculate_variant_group_design(
                        scoped,
                        st.session_state.metrics_config,
                        design,
                        control,
                        list(treatments),
                        float(settings["eligible_flow"]),
                        group,
                        family_comparisons=family_comparisons,
                    )
                    all_plan_rows.extend(plan_rows)
                    all_comparisons.extend([{**row, "Group": group} for row in comparisons])
                    binding_labels.append(f"{group}: {binding}")
                except ValueError as exc:
                    st.error(str(exc))
        render_customer_plan(
            all_plan_rows,
            all_comparisons,
            binding_labels,
            group_names,
            "Because comparable historical strategy arms are unavailable or intentionally not used, baseline and variability are estimated from all eligible historical customers within each group. Randomization makes the planned arm comparison causal after launch.",
        )
        return

    st.markdown("**Numeric Strategy Values by Group**")
    numeric = numeric_columns(st.session_state.raw_df)
    if not numeric:
        st.error("A numeric historical strategy column is required for point-specific planning.")
        return
    st.caption(f"{strategy.strategy_name} · historical field: {strategy.historical_column} · point-specific variability")
    try:
        refresh_customer_processing()
    except ValueError as exc:
        st.error(str(exc))
        return
    grouped = group_labeled_history()
    sync_group_settings(grouped)
    population: PopulationConfig = st.session_state.population_config
    group_names = list(population.group_settings)
    if not group_names:
        st.error("No customer groups are available for planning.")
        return

    all_plan_rows: list[dict] = []
    all_comparisons: list[dict] = []
    binding_labels: list[str] = []
    group_tabs = st.tabs(group_names)
    for group_index, (tab, group) in enumerate(zip(group_tabs, group_names)):
        with tab:
            settings = population.group_settings[group]
            scoped = grouped[grouped["_planning_group"].astype(str) == group].copy()
            summary_left, summary_middle, summary_right = st.columns(3)
            summary_left.metric("Historical Accounts", f"{len(scoped):,.0f}")
            summary_middle.metric("Business Test Range", f"{format_strategy_value(settings['min_line'], strategy)} to {format_strategy_value(settings['max_line'], strategy)}")
            summary_right.metric(f"Eligible Flow / {duration_unit(design.traffic_frequency).title()}", f"{settings['eligible_flow']:,.0f}")
            top_left, top_right = st.columns(2)
            settings["control_line"] = top_left.number_input(
                f"Control {strategy.value_label}",
                min_value=float(settings["min_line"]),
                max_value=float(settings["max_line"]),
                value=min(max(float(settings["control_line"]), float(settings["min_line"])), float(settings["max_line"])),
                key=f"group_bau_{group_index}_v1",
            )
            settings["selection_mode"] = top_right.radio(
                "Strategy Value Selection",
                ["Suggested lines", "Manual lines"],
                index=0 if settings["selection_mode"] == "Suggested lines" else 1,
                horizontal=True,
                format_func=lambda value: "Suggested values" if value == "Suggested lines" else "Manual values",
                key=f"group_line_mode_{group_index}_v1",
            )
            if settings["selection_mode"] == "Suggested lines":
                treatment_count = int(
                    st.number_input(
                        "Number of Treatment Values",
                        min_value=1,
                        max_value=4,
                        value=max(1, len(settings["treatment_lines"]) or 2),
                        step=1,
                        key=f"group_treatment_count_{group_index}_v1",
                    )
                )
                suggestions = suggest_group_designs(
                    scoped,
                    strategy.historical_column,
                    st.session_state.metrics_config,
                    design,
                    float(settings["control_line"]),
                    float(settings["min_line"]),
                    float(settings["max_line"]),
                    treatment_count,
                    float(settings["eligible_flow"]),
                    int(design.max_enrollment_periods),
                    family_comparisons=planning_comparison_count(
                        treatment_count,
                        len(group_names),
                        design.audience_decision_scope,
                    ),
                    max_mapping_distance=float(strategy.max_mapping_distance),
                )
                if suggestions.empty:
                    st.warning("No statistically usable suggestion is available inside this business range. Expand the range or choose values manually.")
                    settings["treatment_lines"] = []
                else:
                    labels = [" and ".join(f"{float(value):,.0f}" for value in values) for values in suggestions["Suggested Treatments"]]
                    current_label = next((label for label, values in zip(labels, suggestions["Suggested Treatments"]) if list(values) == list(settings["treatment_lines"])), labels[0])
                    selected_label = st.radio("Suggested Values", labels, index=labels.index(current_label), key=f"group_suggestion_{group_index}_v1")
                    selected_row = suggestions.iloc[labels.index(selected_label)]
                    settings["treatment_lines"] = [float(value) for value in selected_row["Suggested Treatments"]]
                    shown = suggestions[["Suggested Treatments", "Required Accounts", "Test Duration", "Within Window"]].copy()
                    shown["Suggested Treatments"] = shown["Suggested Treatments"].map(lambda values: ", ".join(format_strategy_value(value, strategy) for value in values))
                    shown["Test Duration"] = shown["Test Duration"].map(lambda value: duration_label(int(math.ceil(value)), design.traffic_frequency))
                    shown["Within Window"] = shown["Within Window"].map({True: "Fits", False: "Too long"})
                    st.dataframe(shown.rename(columns={"Within Window": "Enrollment Window"}), use_container_width=True, hide_index=True)
            else:
                manual_values = settings["treatment_lines"] or [float(settings["min_line"]), float(settings["max_line"])]
                edited_lines = st.data_editor(
                    pd.DataFrame({"Treatment Value": manual_values}),
                    use_container_width=True,
                    hide_index=True,
                    num_rows="dynamic",
                    column_config={"Treatment Value": st.column_config.NumberColumn("Treatment Value", min_value=float(settings["min_line"]), max_value=float(settings["max_line"]), required=True)},
                    key=f"group_manual_lines_{group_index}_v1",
                )
                settings["treatment_lines"] = [float(value) for value in edited_lines["Treatment Value"].dropna().tolist()]

            treatments = list(dict.fromkeys(float(value) for value in settings["treatment_lines"] if not math.isclose(float(value), float(settings["control_line"]))))
            settings["treatment_lines"] = treatments
            strategy.group_arms[group] = {"control": float(settings["control_line"]), "treatments": treatments}
            if not treatments:
                st.info("Choose at least one treatment value to calculate this group’s sample plan.")
                continue
            points = [float(settings["control_line"]), *treatments]
            labels = [f"{strategy.control_name}: {format_strategy_value(settings['control_line'], strategy)}"] + [f"{strategy.treatment_names[index] if index < len(strategy.treatment_names) else f'Treatment {index + 1}'}: {format_strategy_value(value, strategy)}" for index, value in enumerate(treatments)]
            st.plotly_chart(
                historical_association(
                    scoped,
                    strategy.historical_column,
                    primary.processed_column or primary.column,
                    primary.metric_type,
                    strategy_points=points,
                    strategy_label=strategy.strategy_name,
                    metric_label=primary.name,
                    binning_method=strategy.curve_binning_method,
                    bin_width=float(strategy.curve_bin_width),
                    bin_count=int(strategy.curve_bin_count),
                    strategy_point_labels=labels,
                ),
                use_container_width=True,
                key=f"group_plan_chart_{group_index}_v1",
            )
            try:
                plan_rows, comparisons, binding = calculate_group_design(
                    scoped,
                    strategy.historical_column,
                    st.session_state.metrics_config,
                    design,
                    float(settings["control_line"]),
                    treatments,
                    float(settings["eligible_flow"]),
                    group,
                    family_comparisons=planning_comparison_count(
                        len(treatments),
                        len(group_names),
                        design.audience_decision_scope,
                    ),
                    max_mapping_distance=float(strategy.max_mapping_distance),
                )
                all_plan_rows.extend(plan_rows)
                all_comparisons.extend([{**row, "Group": group} for row in comparisons])
                binding_labels.append(f"{group}: {binding}")
            except ValueError as exc:
                st.error(str(exc))

    render_customer_plan(
        all_plan_rows,
        all_comparisons,
        binding_labels,
        group_names,
        "Historical customers are assigned to the closest proposed numeric strategy value within each customer group. Automatic distance protection excludes observations farther than half of the widest gap between test values unless a custom limit is supplied. Means and standard deviations are estimated separately for every group and value, and the design uses generalized Neyman allocation.",
    )


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


def customer_group_analysis_step() -> None:
    st.subheader("Analyze")
    strategy: StrategyConfig = st.session_state.strategy_config
    design: DesignConfig = st.session_state.design_config
    population: PopulationConfig = st.session_state.population_config
    if not strategy.group_arms or not design.group_plan_rows:
        st.info("Complete the Plan before analyzing experiment results.")
        return
    source = st.radio("Experiment Result Data Source", ["Upload File", "Internal Data", "Synthetic Demo"], index=2, horizontal=True, key="group_analysis_source_v1")
    if source == "Upload File":
        uploaded = st.file_uploader("Upload Experiment Results", type=["csv", "parquet"], key="group_analysis_upload_v1")
        if uploaded is not None:
            signature = hash(uploaded.getvalue())
            if st.session_state.get("group_results_upload_signature") != signature:
                uploaded.seek(0)
                st.session_state.results_df = pd.read_parquet(uploaded) if uploaded.name.endswith(".parquet") else load_uploaded_csv(uploaded)
                st.session_state.group_results_upload_signature = signature
                st.session_state.group_analysis_results = {}
    elif source == "Synthetic Demo":
        st.session_state.results_df = synthetic_group_results()
    else:
        st.info("Internal result-data access will use the future Databricks connector.")
        return
    df = st.session_state.results_df
    if df is None or df.empty:
        st.info("Load experiment-result data to run the analysis.")
        return
    columns = list(df.columns)
    analysis_config: AnalysisConfig = st.session_state.analysis_config
    default_unit = analysis_config.unit_id_column if analysis_config.unit_id_column in columns else next((column for column in columns if column.lower() in {"customer_id", "account_id", "user_id", "member_id"}), columns[0])
    default_assignment = strategy.assignment_column if strategy.assignment_column in columns else next((column for column in columns if "assign" in column.lower()), columns[0])
    default_group = "_planning_group" if "_planning_group" in columns else population.grouping_column if population.grouping_column in columns else ""
    mapping_columns = st.columns(3)
    analysis_config.unit_id_column = mapping_columns[0].selectbox("Experimental Unit ID", columns, index=columns.index(default_unit), key="group_analysis_unit_v1")
    assignment_column = mapping_columns[1].selectbox("Assignment Column", columns, index=columns.index(default_assignment), key="group_analysis_assignment_v1")
    group_options = ["No grouping column"] + columns
    selected_group_column = mapping_columns[2].selectbox("Customer Group Column", group_options, index=group_options.index(default_group) if default_group in columns else 0, key="group_analysis_group_v1")
    if len(strategy.group_arms) > 1 and selected_group_column == "No grouping column":
        st.error("A customer group column is required because this experiment has group-specific strategy arms.")
        return
    analysis_df = df.copy()
    if selected_group_column == "_planning_group":
        analysis_df["_analysis_group"] = analysis_df[selected_group_column].astype(str)
    elif selected_group_column != "No grouping column":
        if population.grouping_definitions and "lower" in population.grouping_definitions[0]:
            analysis_df["_analysis_group"] = assign_group_labels(analysis_df[selected_group_column], population.grouping_definitions)
        else:
            analysis_df["_analysis_group"] = analysis_df[selected_group_column].astype(str)
    else:
        analysis_df["_analysis_group"] = "All customers"

    numeric_result_columns = [column for column in numeric_columns(analysis_df) if column != assignment_column]
    result_columns = {}
    used = set()
    for role, metric in st.session_state.metrics_config.items():
        preferred = next((column for column in [metric.source_column, metric.processed_column, metric.column] if column in numeric_result_columns and column not in used), None)
        preferred = preferred or next((column for column in numeric_result_columns if column not in used), numeric_result_columns[0] if numeric_result_columns else "")
        if not preferred:
            st.error("No numeric result columns are available for analysis.")
            return
        result_columns[role] = st.selectbox(f"{role} Result Column", numeric_result_columns, index=numeric_result_columns.index(preferred), key=f"group_result_column_{role}_v1")
        used.add(result_columns[role])

    validation_messages = []
    group_frames = {}
    for group, arms in strategy.group_arms.items():
        scoped = analysis_df[analysis_df["_analysis_group"].astype(str) == str(group)].copy()
        control = arms["control"]
        treatments = list(arms["treatments"])
        outcomes = {role: (result_columns[role], metric.metric_type) for role, metric in st.session_state.metrics_config.items()}
        expected_shares = {
            row["Arm"]: row["Traffic Allocation"]
            for row in design.group_plan_rows
            if str(row["Group"]) == str(group)
        }
        errors, warnings = validate_analysis_data(scoped, assignment_column, outcomes, control, treatments, analysis_config.unit_id_column, expected_shares)
        validation_messages.extend(("error", f"{group}: {message}") for message in errors)
        validation_messages.extend(("warning", f"{group}: {message}") for message in warnings)
        group_frames[group] = scoped
    for level, message in validation_messages:
        if level == "error":
            st.error(message)
        else:
            st.warning(message)
    blocking = any(level == "error" for level, _ in validation_messages)
    if st.button("Run Group Analysis", type="primary", disabled=blocking, key="run_group_analysis_v1"):
        by_group, responses_by_group = {}, {}
        for group, scoped in group_frames.items():
            arms = strategy.group_arms[group]
            control, treatments = arms["control"], list(arms["treatments"])
            by_role, responses = {}, {}
            for role, metric in st.session_state.metrics_config.items():
                column = result_columns[role]
                result = (
                    binary_results(scoped, assignment_column, column, control, treatments, design.alpha, design.multiplicity_method)
                    if metric.metric_type == "Binary"
                    else continuous_results(scoped, assignment_column, column, control, treatments, design.alpha, design.multiplicity_method)
                )
                by_role[role] = result
                responses[role] = response_summary(scoped, assignment_column, column, design.alpha, metric.metric_type)
            by_group[group] = by_role
            responses_by_group[group] = responses
        st.session_state.group_analysis_results = by_group
        st.session_state.group_response_results = responses_by_group
        st.session_state.analysis_results_by_role = next(iter(by_group.values()), {})
        st.session_state.customer_analysis_is_current = True
        st.rerun()
    results_by_group = st.session_state.group_analysis_results
    if not results_by_group:
        return
    result_tabs = st.tabs(list(results_by_group))
    for group_index, (tab, group) in enumerate(zip(result_tabs, results_by_group)):
        with tab:
            scoped = group_frames.get(group, pd.DataFrame())
            arms = strategy.group_arms[group]
            outcomes = {role: (result_columns[role], metric.metric_type) for role, metric in st.session_state.metrics_config.items()}
            expected_shares = {
                row["Arm"]: row["Traffic Allocation"]
                for row in design.group_plan_rows
                if str(row["Group"]) == str(group)
            }
            integrity = analysis_integrity_summary(scoped, assignment_column, outcomes, [arms["control"], *arms["treatments"]], analysis_config.unit_id_column, expected_shares)
            i1, i2, i3 = st.columns(3)
            i1.metric("Analysis Accounts", f"{integrity['unique_units']:,}")
            i2.metric("Assignment Balance", integrity["srm_status"])
            i3.metric("Missing Primary", f"{integrity['missing_by_metric'].get('Primary', 0):,}")
            roles = list(results_by_group[group])
            view_role = st.selectbox("Result Metric", roles, format_func=lambda role: f"{role}: {st.session_state.metrics_config[role].name}", key=f"group_result_role_{group_index}_v1")
            result = results_by_group[group][view_role]
            metric = st.session_state.metrics_config[view_role]
            table = format_effect_table(result, metric.metric_type, strategy.strategy_name)
            if view_role == "Guardrail":
                threshold = guardrail_absolute_threshold(metric)
                table["Guardrail Status"] = result.apply(lambda row: "Control" if row["Is Control"] else guardrail_status(row, threshold, metric.direction), axis=1)
            st.dataframe(table, use_container_width=True, hide_index=True)
            result_left, result_right = st.columns(2)
            with result_left:
                st.caption("Estimated lift versus the current / BAU option")
                st.plotly_chart(forest_plot(result, strategy.strategy_name), use_container_width=True, key=f"group_forest_{group_index}_{view_role}_v1")
            with result_right:
                st.caption("Observed outcome for every test option")
                st.plotly_chart(response_plot(st.session_state.group_response_results[group][view_role], strategy.strategy_name, metric.name), use_container_width=True, key=f"group_response_{group_index}_{view_role}_v1")


def customer_group_decision_step() -> None:
    st.subheader("Decide")
    results_by_group = st.session_state.group_analysis_results
    if not results_by_group:
        st.info("Run the group analysis before generating recommendations.")
        return
    primary_metric = st.session_state.metrics_config["Primary"]
    guardrail_metric = st.session_state.metrics_config.get("Guardrail")
    guardrail_threshold = guardrail_absolute_threshold(guardrail_metric)
    tabs = st.tabs(list(results_by_group))
    for tab, (group, results) in zip(tabs, results_by_group.items()):
        with tab:
            scorecard = arm_decision_scorecard(
                results["Primary"],
                primary_metric.direction,
                normalize_metric(primary_metric).effect_absolute,
                results.get("Guardrail"),
                guardrail_threshold,
                guardrail_metric.direction if guardrail_metric else "Lower is Better",
            )
            display = scorecard.copy()
            display["Primary Effect"] = display["Primary Effect"].map(lambda value: number(value, 2))
            display["Relative Lift"] = display["Relative Lift"].map(lambda value: percent(value, 1) if pd.notna(value) else "")
            display["Adjusted p-value"] = display["Adjusted p-value"].map(lambda value: p_value(value) if pd.notna(value) else "")
            display["Meets Planned Effect"] = display["Meets Planned Effect"].map({True: "Yes", False: "No"})
            st.dataframe(display, use_container_width=True, hide_index=True)
            recommendation = experiment_recommendation(results["Primary"], results.get("Guardrail"), guardrail_threshold, guardrail_metric.direction if guardrail_metric else "Lower is Better", primary_metric.direction)
            status = recommendation_status(recommendation)
            if status == "error":
                st.error(recommendation)
            elif status == "warning":
                st.warning(recommendation)
            else:
                st.success(recommendation)
    st.caption("Recommendations are calculated independently for each configured customer group and only for the randomized strategy arms.")


def analysis_step() -> None:
    render_business_guide(
        "Upload the completed randomized test. We will compare each test option with the current experience.",
        "Output: lift and uncertainty",
    )
    strategy: StrategyConfig = st.session_state.strategy_config
    design: DesignConfig = st.session_state.design_config
    source = st.radio("Experiment Result Data Source", ["Upload File", "Internal Data", "Synthetic Demo"], index=0, horizontal=True, key="customer_analysis_source_v2")
    if source == "Upload File":
        uploaded = st.file_uploader("Upload Experiment Results", type=["csv", "parquet"], key="analysis_upload")
        if uploaded is not None:
            signature = hash(uploaded.getvalue())
            if st.session_state.get("customer_results_upload_signature") != signature:
                uploaded.seek(0)
                st.session_state.results_df = pd.read_parquet(uploaded) if uploaded.name.endswith(".parquet") else load_uploaded_csv(uploaded)
                st.session_state.customer_results_upload_signature = signature
                st.session_state.analysis_results_by_role = {}
        else:
            st.info("Upload completed experiment data to populate the analysis options.")
            return
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
    analysis_config.unit_id_column = map_left.selectbox("Customer or Account ID", cols, index=cols.index(inferred_unit) if inferred_unit in cols else 0)
    assignment_col = map_right.selectbox("Test Group Column", cols, index=cols.index(default_assignment))
    control, treatments = configured_arms(strategy)
    available_arms = sorted(df[assignment_col].dropna().unique().tolist())
    arm_left, arm_right = st.columns(2)
    control = arm_left.selectbox("Current / BAU Group", available_arms, index=available_arms.index(control) if control in available_arms else 0)
    treatments = arm_right.multiselect("Test Groups", [arm for arm in available_arms if arm != control], default=[arm for arm in available_arms if arm != control])
    with st.expander("Additional outcomes and advanced settings", expanded=False):
        primary = st.session_state.metrics_config["Primary"]
        p1, p2 = st.columns(2)
        primary.metric_type = p1.selectbox("Primary Metric Type", ["Continuous", "Binary"], index=["Continuous", "Binary"].index(primary.metric_type), key="analysis_primary_type_v1")
        primary.direction = p2.selectbox("Primary Success Direction", ["Higher is Better", "Lower is Better"], index=["Higher is Better", "Lower is Better"].index(primary.direction), key="analysis_primary_direction_v1")
        a1, a2 = st.columns(2)
        add_secondary = a1.checkbox("Add Secondary Metric", value="Secondary" in st.session_state.metrics_config, key="analysis_add_secondary_v1")
        add_guardrail = a2.checkbox("Add Guardrail Metric", value="Guardrail" in st.session_state.metrics_config, key="analysis_add_guardrail_v1")
        if add_secondary:
            st.session_state.metrics_config.setdefault("Secondary", MetricConfig("Secondary Outcome", "", "Secondary", source_column=""))
        else:
            st.session_state.metrics_config.pop("Secondary", None)
        if add_guardrail:
            st.session_state.metrics_config.setdefault("Guardrail", MetricConfig("Guardrail Outcome", "", "Guardrail", direction="Lower is Better", source_column="", guardrail_threshold=0.15))
        else:
            st.session_state.metrics_config.pop("Guardrail", None)
        design.multiplicity_method = st.selectbox("Multiple-Comparison Control", ["Holm", "None"], index=["Holm", "None"].index(design.multiplicity_method), key="analysis_multiplicity_v1")
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
        result_columns[role] = st.selectbox(f"{role} Outcome Column", numeric, index=numeric.index(preferred), key=f"result_{role}_v2")
        used_result_columns.add(result_columns[role])
        if result_columns[role] == assignment_col:
            st.error(f"{role} Result Column cannot be the Experimental Assignment Column.")
            invalid_mapping = True
    outcomes = {role: (result_columns[role], metric.metric_type) for role, metric in st.session_state.metrics_config.items() if role in result_columns}
    selected_arms = [control] + treatments
    planned_sizes = {arm: float(design.arm_sample_sizes.get(str(arm), 0)) for arm in selected_arms}
    expected_shares = planned_sizes if planned_sizes and all(value > 0 for value in planned_sizes.values()) else None
    validation_errors, validation_warnings = validate_analysis_data(
        df,
        assignment_col,
        outcomes,
        control,
        treatments,
        analysis_config.unit_id_column,
        expected_shares,
    )
    for warning in validation_warnings:
        st.warning(warning)
    for error in validation_errors:
        st.error(error)
    integrity = analysis_integrity_summary(df, assignment_col, outcomes, selected_arms, analysis_config.unit_id_column, expected_shares)
    analysis_ready = not validation_errors and not invalid_mapping
    health_status = "Ready to run" if analysis_ready and integrity["srm_status"] == "Pass" else "Review allocation" if analysis_ready else "Review data"
    render_customer_kpi_panel(
        "Experiment Health",
        "Confirm that assignment and outcome data are trustworthy before using the result.",
        health_status,
        [
            ("Analysis Units", f"{integrity['unique_units']:,}", "analysis"),
            ("Duplicate Units", f"{integrity['duplicate_units']:,}", ""),
            ("Assignment Balance", str(integrity["srm_status"]), ""),
            ("Missing Primary", f"{integrity['missing_by_metric'].get('Primary', 0):,}", ""),
        ],
        analysis=True,
    )
    health_chart, health_explanation = st.columns([1.55, 1])
    with health_chart:
        st.caption("Observed customers / accounts compared with the planned traffic split")
        st.plotly_chart(
            customer_allocation_health_chart(integrity["arm_counts"], integrity["expected_counts"]),
            use_container_width=True,
            key="customer_allocation_health_v1",
        )
    with health_explanation:
        st.markdown("**What this means**")
        if integrity["srm_status"] == "Review":
            st.warning("The group sizes differ more than random variation would usually explain. Check assignment rules and tracking before trusting the effect estimate.")
        else:
            st.success("The observed group sizes are consistent with the planned split. No allocation anomaly was detected.")
        if integrity["duplicate_units"]:
            st.error("Some customers or accounts appear more than once. Aggregate to one row per randomized unit.")
        elif integrity["missing_by_metric"].get("Primary", 0):
            st.info("Some primary outcomes are missing. Confirm that missingness is not concentrated in one test group.")
        else:
            st.caption("Customer IDs are unique and the primary outcome is complete for the analyzed rows.")
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
    if st.button("Calculate Results", type="primary", disabled=bool(validation_errors) or invalid_mapping):
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
        result_left, result_right = st.columns(2)
        with result_left:
            st.caption("Estimated lift versus the current / BAU option")
            st.plotly_chart(forest_plot(result, strategy.strategy_name), use_container_width=True)
        with result_right:
            st.caption("Observed outcome for every test option")
            st.plotly_chart(response_plot(st.session_state.response_by_role[view_role], strategy.strategy_name, metric.name), use_container_width=True)


def decision_step() -> None:
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
    recommendation = experiment_recommendation(results["Primary"], results.get("Guardrail"), guardrail_threshold, guardrail_metric.direction if guardrail_metric else "Lower is Better", primary_metric.direction)
    if scorecard.empty:
        st.info("No treatment arm is available for a decision comparison.")
        return
    scale_candidates = scorecard[scorecard["Decision"] == "Scale candidate"]
    promising_candidates = scorecard[scorecard["Decision"] == "Promising; add safety check"]
    leading_row = scale_candidates.iloc[0] if not scale_candidates.empty else promising_candidates.iloc[0] if not promising_candidates.empty else scorecard.iloc[0]
    leading_lift = leading_row.get("Relative Lift", float("nan"))
    decision_status = "Scale candidate" if not scale_candidates.empty else "Promising result" if not promising_candidates.empty else "Review result"
    render_customer_kpi_panel(
        "Experiment Decision",
        "The leading tested treatment based on primary-metric evidence and configured safety checks.",
        decision_status,
        [
            ("Leading Treatment", str(leading_row["Treatment Arm"]), "treatment"),
            ("Relative Lift", percent(leading_lift, 1) if pd.notna(leading_lift) else "Not available", "analysis" if pd.notna(leading_lift) and leading_lift > 0 else ""),
            ("Primary Evidence", str(leading_row["Primary Evidence"]), ""),
            ("Guardrail", str(leading_row["Guardrail"]), ""),
        ],
        analysis=True,
    )
    if guardrail_metric is None:
        if decision_status == "Promising result":
            st.warning("The primary outcome improved, but no safety outcome was configured. Complete operational and risk checks before rollout.")
        else:
            st.info("No safety outcome was configured. The result describes primary-outcome performance, but it cannot confirm rollout safety.")

    st.markdown("**Treatment Decision Map**")
    st.caption("Options farther right deliver more business benefit. Options higher on the chart have stronger evidence.")
    st.plotly_chart(
        customer_decision_map(
            scorecard,
            results["Primary"],
            meaningful_effect,
            primary_metric.direction,
            primary_metric.name,
        ),
        use_container_width=True,
        key="customer_decision_map_v1",
    )

    leading_result = results["Primary"][results["Primary"]["Credit Line"] == leading_row["Treatment Arm"]]
    if not leading_result.empty:
        st.markdown("**Full-Rollout Impact Simulator**")
        st.caption("Translate the measured per-customer effect into the total outcome change expected for the population you may roll out to.")
        analyzed_population = int(results["Primary"]["N"].sum())
        rollout_population = int(
            st.number_input(
                "Customers / Accounts Eligible for Full Rollout",
                min_value=1,
                value=max(analyzed_population, 1),
                step=max(100, int(max(analyzed_population, 1) / 10)),
                help="Enter the future population that would receive the winning option, not the experiment sample size.",
                key="customer_rollout_population_v1",
            )
        )
        projection = project_rollout_impact(leading_result.iloc[0], rollout_population, primary_metric.direction)
        if projection["outcome_impact"] < 0:
            impact_label = "Expected Outcome Change" if primary_metric.metric_type == "Binary" else "Expected Total Outcome Change"
        elif primary_metric.metric_type == "Binary":
            impact_label = "Expected Additional Outcomes" if primary_metric.direction == "Higher is Better" else "Expected Avoided Outcomes"
        else:
            impact_label = "Expected Total Outcome Gain" if primary_metric.direction == "Higher is Better" else "Expected Total Outcome Reduction"
        impact_columns = st.columns(3)
        impact_columns[0].metric("Full Rollout Population", f"{rollout_population:,.0f}")
        impact_columns[1].metric(impact_label, f"{projection['outcome_impact']:+,.0f}")
        impact_columns[2].metric("95% Plausible Range", f"{projection['outcome_lower']:+,.0f} to {projection['outcome_upper']:+,.0f}")
        pace_left, pace_right = st.columns([1, 1.35])
        rollout_periods = int(
            pace_left.select_slider(
                "Periods to Reach Full Rollout",
                options=[1, 2, 3, 4, 6, 9, 12, 18, 24],
                value=6,
                help="Choose how many operating periods it will take to reach the full eligible population.",
                key="customer_rollout_periods_v1",
            )
        )
        rollout_pattern = pace_right.selectbox(
            "Rollout Pattern",
            ["Even pace", "Pilot, then scale", "Fast start"],
            help="Pilot, then scale starts cautiously. Fast start reaches more customers earlier.",
            key="customer_rollout_pattern_v1",
        )
        st.plotly_chart(
            customer_rollout_impact_chart(
                leading_result.iloc[0],
                rollout_population,
                primary_metric.direction,
                rollout_periods,
                rollout_pattern,
            ),
            use_container_width=True,
            key="customer_rollout_impact_v1",
        )
        with st.expander("Convert outcome impact to business value", expanded=False):
            value_left, value_right = st.columns(2)
            value_per_unit = value_left.number_input(
                "Business Value per Outcome Unit ($)",
                min_value=0.0,
                value=1.0,
                step=1.0,
                help="For example, enter contribution margin per retained customer or economic value per unit of the primary outcome.",
                key="customer_value_per_outcome_v1",
            )
            implementation_cost = value_right.number_input(
                "Estimated Rollout Cost ($)",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                key="customer_rollout_cost_v1",
            )
            valued_projection = project_rollout_impact(
                leading_result.iloc[0],
                rollout_population,
                primary_metric.direction,
                value_per_unit,
                implementation_cost,
            )
            value_metrics = st.columns(2)
            value_metrics[0].metric("Expected Net Business Value", money(valued_projection["net_value"]))
            value_metrics[1].metric("95% Value Range", f"{money(valued_projection['net_lower'])} to {money(valued_projection['net_upper'])}")
        st.caption("Projection assumes the measured per-customer effect continues at full rollout. It does not include market saturation, operational constraints, or unmeasured safety effects.")
    display = scorecard.copy()
    display["Primary Effect"] = display["Primary Effect"].apply(lambda value: number(value, 2))
    display["Relative Lift"] = display["Relative Lift"].apply(lambda value: percent(value, 1) if pd.notna(value) else "")
    display["Adjusted p-value"] = display["Adjusted p-value"].apply(lambda value: p_value(value) if pd.notna(value) else "")
    display["Meets Planned Effect"] = display["Meets Planned Effect"].map({True: "Yes", False: "No"})
    st.markdown("**Treatment Decision Scorecard**")
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.markdown("**Recommended Action**")
    status = recommendation_status(recommendation)
    if status == "error":
        st.error(recommendation)
    elif status == "warning":
        st.warning(recommendation)
    else:
        st.success(recommendation)
    st.caption("The recommendation applies only to the tested arms, eligible population, observation windows, and configured metrics.")


def start_time_series_workflow(intent: str) -> None:
    """Enter a Time Series workflow with the matching demo when no custom data is active."""
    cfg: TimeSeriesConfig = st.session_state.ts_config
    if st.session_state.ts_raw_df.empty or cfg.data_source == "Synthetic Demo":
        cfg.data_source = "Synthetic Demo"
        if intent == "plan":
            st.session_state.ts_raw_df = load_timeseries_planning_demo()
            cfg.dataset_name = "Synthetic Historical Time Series"
        else:
            st.session_state.ts_raw_df = load_timeseries_demo()
            cfg.dataset_name = "Synthetic Campaign Time Series"
        st.session_state.ts_prepared_df = pd.DataFrame()
    st.session_state.ts_intent = intent


def ts_landing_preview() -> None:
    """Show the time-series story before asking the user to configure the workflow."""
    cfg: TimeSeriesConfig = st.session_state.ts_config
    using_workspace = not st.session_state.ts_raw_df.empty
    raw = st.session_state.ts_raw_df if using_workspace else load_timeseries_demo()
    inferred = infer_timeseries_columns(raw)
    columns = list(raw.columns)
    date_col = cfg.date_column if using_workspace and cfg.date_column in columns else inferred["date"]
    outcome_col = cfg.outcome_column if using_workspace and cfg.outcome_column in columns else inferred["outcome"]
    flag_col = cfg.campaign_flag_column if using_workspace and cfg.campaign_flag_column in columns else inferred["campaign_flag"]
    try:
        preview = build_timeseries_preview(raw, date_col, outcome_col, flag_col, cfg.duplicate_policy, cfg.missing_policy)
    except Exception:
        raw = load_timeseries_demo()
        date_col, outcome_col, flag_col = "date", "applications", "campaign_flag"
        preview = build_timeseries_preview(raw, date_col, outcome_col, flag_col)
        using_workspace = False

    prepared = preview["prepared"]
    launch = preview["campaign_launch"]
    if launch is None and cfg.intervention_date:
        candidate = pd.Timestamp(cfg.intervention_date)
        if preview["start_date"] <= candidate <= preview["end_date"]:
            launch = candidate
    pre_count = int((prepared["_date"] < launch).sum()) if launch is not None else int(preview["observations"])
    post_count = int((prepared["_date"] >= launch).sum()) if launch is not None else 0
    total_count = max(pre_count + post_count, 1)
    pre_share = pre_count / total_count * 100
    post_share = post_count / total_count * 100
    period_label = f"{pd.Timestamp(preview['start_date']):%b %Y} – {pd.Timestamp(preview['end_date']):%b %Y}"
    source_label = "your current dataset" if using_workspace else "the built-in campaign demo"
    launch_label = pd.Timestamp(launch).strftime("%b %-d, %Y") if launch is not None else "Not launched"
    context_note = (
        f"Campaign launch is visible on {launch_label}. The shaded region is the period available for impact analysis."
        if launch is not None
        else "This is historical-only planning data. The workflow will place the planned launch after the available history."
    )

    action_copy, plan_action, analyze_action = st.columns([2.4, 1.15, 1.15])
    action_copy.markdown(
        f'<div class="ts-demo-actions"><div><strong>See the campaign story first</strong><span>Previewing {html.escape(source_label)}. Choose what you need next.</span></div></div>',
        unsafe_allow_html=True,
    )
    if plan_action.button("Plan a Campaign", type="primary", use_container_width=True, key="ts_demo_plan"):
        start_time_series_workflow("plan")
        st.rerun()
    if analyze_action.button("Analyze Campaign Results", use_container_width=True, key="ts_demo_analyze"):
        start_time_series_workflow("analyze")
        st.rerun()

    chart_col, summary_col = st.columns([1.75, 0.75])
    with chart_col:
        with st.container(border=True):
            st.plotly_chart(
                time_series_line(prepared, pd.Timestamp(launch).date().isoformat() if launch is not None else "", humanize_column_name(outcome_col)),
                use_container_width=True,
                config={"displayModeBar": False},
            )
    with summary_col:
        st.markdown(
            f"""
            <section class="ts-preview-summary">
              <h3>Data for this campaign</h3>
              <div class="ts-preview-grid">
                <div class="ts-preview-stat"><div class="ts-preview-label">Observations</div><div class="ts-preview-value">{int(preview['observations']):,}</div><div class="ts-preview-detail">{int(preview['missing_periods'])} missing periods</div></div>
                <div class="ts-preview-stat"><div class="ts-preview-label">Time period</div><div class="ts-preview-value">{html.escape(period_label)}</div><div class="ts-preview-detail">{html.escape(str(preview['frequency']))} data</div></div>
                <div class="ts-preview-stat"><div class="ts-preview-label">Outcome</div><div class="ts-preview-value">{html.escape(humanize_column_name(outcome_col))}</div><div class="ts-preview-detail">Business result to measure</div></div>
                <div class="ts-preview-stat"><div class="ts-preview-label">Campaign launch</div><div class="ts-preview-value">{html.escape(launch_label)}</div><div class="ts-preview-detail">{len(preview['predictors'])} usable predictor{'s' if len(preview['predictors']) != 1 else ''}</div></div>
              </div>
              <div class="ts-period-split">
                <div class="ts-period-row"><span>Timeline coverage</span><b>{pre_count} before · {post_count} after</b></div>
                <div class="ts-period-track"><div class="ts-period-pre" style="width:{pre_share:.1f}%"></div><div class="ts-period-post" style="width:{post_share:.1f}%"></div></div>
                <div class="ts-period-legend"><span>Historical baseline</span><span>Campaign period</span></div>
              </div>
              <div class="ts-preview-note">{html.escape(context_note)}</div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    preview_columns = [column for column in raw.columns if not str(column).startswith("_")]
    roles = {date_col: "Date", outcome_col: "Outcome"}
    if flag_col != "None":
        roles[flag_col] = "Campaign"
    display_headers = {column: f"{humanize_column_name(column)} [{roles[column]}]" if column in roles else humanize_column_name(column) for column in preview_columns}
    display = raw.loc[:, preview_columns].head(12).copy()
    display[date_col] = pd.to_datetime(display[date_col], errors="coerce").dt.strftime("%b %-d, %Y")
    display = display.rename(columns=display_headers)
    st.markdown(
        f'<div class="ts-section-heading"><strong>Campaign data preview</strong><span>Showing {min(12, len(raw))} of {len(raw):,} rows · all {len(preview_columns)} source columns.</span></div>',
        unsafe_allow_html=True,
    )
    st.dataframe(display, use_container_width=True, hide_index=True, height=330)


def ts_data_step() -> None:
    render_business_guide(
        "Bring one historical series with a date and the business outcome you want to measure.",
        "Output: analysis-ready history",
    )
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
    cfg.campaign_flag_column = cfg.campaign_flag_column if cfg.campaign_flag_column in df.columns else inferred["campaign_flag"]
    cfg.exposure_column = cfg.exposure_column if cfg.exposure_column in ["None"] + numeric else "None"

    st.markdown("**Tell Us What Each Column Means**")
    c1, c2, c3 = st.columns(3)
    cfg.date_column = c1.selectbox("Date", list(df.columns), index=list(df.columns).index(cfg.date_column) if cfg.date_column in df.columns else 0)
    cfg.outcome_column = c2.selectbox("Business Outcome", numeric, index=numeric.index(cfg.outcome_column) if cfg.outcome_column in numeric else 0)
    exposure_options = ["None"] + [column for column in numeric if column != cfg.outcome_column]
    cfg.exposure_column = c3.selectbox(
        "Volume / Customer Count (optional)",
        exposure_options,
        index=exposure_options.index(cfg.exposure_column) if cfg.exposure_column in exposure_options else 0,
        help="Used only to project operational volume. It is not the statistical time-series sample size.",
    )
    duplicate_options = ["Fail validation", "Average", "Sum"]
    with st.expander("Advanced data handling", expanded=False):
        a1, a2 = st.columns(2)
        cfg.duplicate_policy = a1.selectbox("When a date appears more than once", duplicate_options, index=duplicate_options.index(cfg.duplicate_policy) if cfg.duplicate_policy in duplicate_options else 0)
        cfg.missing_policy = a2.selectbox("When a date is missing", ["Fail validation", "Interpolate", "Fill missing counts with zero", "Keep"], index=["Fail validation", "Interpolate", "Fill missing counts with zero", "Keep"].index(cfg.missing_policy))
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
        preview_launch = ""
        if cfg.campaign_flag_column != "None" and cfg.campaign_flag_column in df:
            source_dates = pd.to_datetime(df[cfg.date_column], errors="coerce")
            source_flags = pd.to_numeric(df[cfg.campaign_flag_column], errors="coerce").fillna(0)
            launch_dates = source_dates[source_flags > 0].dropna()
            preview_launch = launch_dates.min().date().isoformat() if not launch_dates.empty else ""
        st.markdown(
            '<div class="ts-section-heading"><strong>Series preview</strong><span>Check the cadence, trend, and campaign boundary before continuing.</span></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            time_series_line(prepared, preview_launch, humanize_column_name(cfg.outcome_column)),
            use_container_width=True,
            config={"displayModeBar": False},
        )
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
    render_business_guide(
        "Mark when the campaign began. The chart below confirms which observations are before and after launch.",
        "Output: campaign measurement window",
    )
    cfg: TimeSeriesConfig = st.session_state.ts_config
    data = st.session_state.ts_prepared_df
    if data.empty:
        st.info("Upload and map time-series data before defining the campaign.")
        return
    min_date = data["_date"].min().date()
    max_date = data["_date"].max().date()
    default_launch = bounded_date(cfg.intervention_date, min_date, max_date, data["_date"].quantile(0.75).date())
    c1, c2, c3 = st.columns(3)
    launch = c1.date_input("When did the campaign start?", value=default_launch, min_value=min_date, max_value=max_date)
    cfg.intervention_date = launch.isoformat()
    end_enabled = c2.checkbox("The campaign had a fixed end date", value=bool(cfg.campaign_end_date))
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
    render_business_guide(
        "Choose the planned launch date so the recommendation can show the expected readout date.",
        "Output: planned launch point",
    )
    cfg: TimeSeriesConfig = st.session_state.ts_config
    data = st.session_state.ts_prepared_df
    if data.empty:
        st.info("Load historical time-series data before defining a planned campaign.")
        return
    min_date, historical_end = data["_date"].min().date(), data["_date"].max().date()
    offsets = {"Daily": pd.Timedelta(days=1), "Weekly": pd.Timedelta(days=7), "Monthly": pd.DateOffset(months=1)}
    default_launch = bounded_date(cfg.planned_launch_date, min_date, historical_end + pd.Timedelta(days=730), historical_end + offsets.get(cfg.frequency, pd.Timedelta(days=7)))
    launch = st.date_input("When do you plan to launch?", value=default_launch, min_value=min_date, max_value=historical_end + pd.Timedelta(days=730))
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
    render_business_guide(
        "Choose a quick comparison or an adjusted impact estimate. The selected result will appear automatically in See Impact.",
        "Output: measured campaign impact",
    )
    cfg: TimeSeriesConfig = st.session_state.ts_config
    data = st.session_state.ts_prepared_df
    if data.empty:
        st.info("Upload a historical time-series dataset before running analysis.")
        return
    if not cfg.intervention_date:
        st.info("Select the campaign launch date to define the pre- and post-periods.")
        return
    cfg.analysis_method = st.radio(
        "How should impact be measured?",
        ["Pre–Post Analysis", "Structural Time Series Counterfactual"],
        horizontal=True,
        format_func=lambda value: "Quick before-and-after comparison" if value == "Pre–Post Analysis" else "Adjusted impact estimate",
    )
    if cfg.analysis_method == "Pre–Post Analysis":
        st.caption("Fast descriptive comparison. Use it for context, not as a causal claim: it does not adjust for trend, seasonality, or other changes.")
        cfg.comparison_window = st.radio(
            "Which periods should be compared?",
            ["Symmetric", "All available", "Custom date range"],
            index=["Symmetric", "All available", "Custom date range"].index(cfg.comparison_window) if cfg.comparison_window in ["Symmetric", "All available", "Custom date range"] else 0,
            horizontal=True,
            format_func=lambda value: {
                "Symmetric": "Same number before and after",
                "All available": "All available history",
                "Custom date range": "Choose dates",
            }[value],
        )
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
        cfg.comparison_metric = st.radio(
            "Compare",
            ["Average per period", "Total"],
            horizontal=True,
            format_func=lambda value: "Average outcome per period" if value == "Average per period" else "Total outcome",
        )
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
            render_ts_kpi_panel(
                "Before-and-After Result Ready",
                "The result updates automatically whenever the selected window or comparison settings change.",
                "Ready to view",
                [
                    ("Periods Before", f"{result['pre_observations']:,}", ""),
                    ("Periods After", f"{result['post_observations']:,}", ""),
                    ("Estimated Change", percent(result["percent_change"], 1), "positive" if result["percent_change"] >= 0 else "negative"),
                    ("Method", "Pre–Post", ""),
                ],
                analysis=True,
            )
    else:
        st.caption("Estimate what likely would have happened without the campaign using pre-campaign patterns and unaffected comparison signals.")
        numeric = predictor_candidates(data, cfg.outcome_column, cfg.date_column, cfg.campaign_flag_column)
        st.markdown("**Unaffected Comparison Signals (Optional)**")
        st.caption("Choose related series that the campaign could not have changed, such as a control product or external demand index.")
        cfg.selected_covariates = st.multiselect("Comparison Signals", numeric, default=[col for col in cfg.selected_covariates if col in numeric], format_func=humanize_column_name)
        with st.expander("Advanced model settings", expanded=False):
            cfg.seasonality = st.selectbox("Seasonality", ["Auto", "None", "Weekly", "Monthly", "Annual"], index=["Auto", "None", "Weekly", "Monthly", "Annual"].index(cfg.seasonality))
            cfg.prediction_interval = st.selectbox("Confidence Level", [0.8, 0.9, 0.95], index=[0.8, 0.9, 0.95].index(cfg.prediction_interval), format_func=lambda value: f"{int(value * 100)}%")
            cfg.local_trend = st.checkbox("Use Local Trend", value=cfg.local_trend)
            cfg.simulation_seed = st.number_input("Simulation Seed", min_value=1, max_value=999999, value=int(cfg.simulation_seed), step=1)
            cfg.holdout_length = st.number_input("Holdout Length", min_value=4, max_value=24, value=int(cfg.holdout_length), step=1)
        current_fingerprint = f"{method_config_fingerprint(cfg, 'Structural Time Series Counterfactual')}:{data_fingerprint(data)}"
        status = st.session_state.ts_structural_status
        has_result = bool(st.session_state.ts_bsts_result)
        action_label = "Recalculate Impact" if has_result else "Calculate Impact"
        if st.session_state.ts_structural_fingerprint and st.session_state.ts_structural_fingerprint != current_fingerprint:
            status = "stale"
            st.session_state.ts_structural_status = status
            action_label = "Update Impact"
            st.warning("Results are out of date because the settings changed.")
        if status == "running":
            st.info("Calculating the expected no-campaign outcome...")
        elif status == "completed":
            st.success("Impact calculation completed")
            st.caption(f"Last calculated: {st.session_state.ts_structural_timestamp}")
        elif status == "failed":
            st.warning("Analysis failed\n\nThe model could not complete with the current settings. Review the configuration and try again.")
        pre_observations = int((data["_date"] < pd.Timestamp(cfg.intervention_date)).sum())
        render_ts_kpi_panel(
            "Adjusted Impact Estimate Ready",
            "Build an expected no-campaign outcome from the available history and comparison signals.",
            "Result available" if has_result and status == "completed" else "Ready to run",
            [
                ("History Before Campaign", f"{pre_observations:,}", ""),
                ("Comparison Signals", f"{len(cfg.selected_covariates):,}", ""),
                ("Data Frequency", cfg.frequency, ""),
                ("Model Setup", "Automatic", ""),
            ],
            analysis=True,
        )
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
    cfg: TimeSeriesConfig = st.session_state.ts_config
    data = st.session_state.ts_prepared_df
    if data.empty or not cfg.intervention_date:
        st.info("Configure the intervention period before viewing results.")
        return
    evaluation = data[data["_date"] >= pd.Timestamp(cfg.intervention_date)]
    first_treated = evaluation["_date"].min() if not evaluation.empty else pd.Timestamp(cfg.intervention_date)
    evaluation_end = evaluation["_date"].max() if not evaluation.empty else data["_date"].max()
    with st.expander("What was analyzed", expanded=False):
        d1, d2, d3 = st.columns(3)
        d1.write(f"**Outcome**\n\n{humanize_column_name(cfg.outcome_column)}")
        d2.write(f"**Intervention**\n\n{pd.Timestamp(cfg.intervention_date).strftime('%b %-d, %Y')}\n\n**First Treated Observation**\n\n{first_treated.strftime('%b %-d, %Y')}")
        d3.write(f"**Evaluation Period**\n\n{first_treated.strftime('%b %-d, %Y')} – {evaluation_end.strftime('%b %-d, %Y')} · {len(evaluation)} {duration_unit(cfg.frequency)}s\n\n**Method**\n\n{cfg.analysis_method}")
    current_fingerprint = f"{method_config_fingerprint(cfg, cfg.analysis_method)}:{data_fingerprint(data)}"
    if cfg.analysis_method == "Pre–Post Analysis":
        prepost = st.session_state.ts_prepost_result if st.session_state.ts_prepost_fingerprint == current_fingerprint else None
        if prepost:
            render_ts_kpi_panel(
                "Before-and-After Result",
                "A descriptive before-and-after comparison for the selected campaign window.",
                "Descriptive result",
                [
                    ("Before Campaign", number(prepost["pre_value"], 2), ""),
                    ("After Campaign", number(prepost["post_value"], 2), ""),
                    ("Absolute Change", number(prepost["absolute_change"], 2), "positive" if prepost["absolute_change"] >= 0 else "negative"),
                    ("Percent Change", percent(prepost["percent_change"], 1), "positive" if prepost["percent_change"] >= 0 else "negative"),
                ],
                analysis=True,
            )
            st.markdown('<div class="ts-panel-title">Outcome Through Time</div>', unsafe_allow_html=True)
            st.plotly_chart(time_series_line(data, cfg.intervention_date, humanize_column_name(cfg.outcome_column)), use_container_width=True)
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
        low, high = bsts["prediction_interval"]
        impact_positive = bsts["cumulative_impact"] >= 0
        render_ts_kpi_panel(
            "Adjusted Campaign Impact",
            f"{percent(bsts['interval_level'], 0)} impact interval: {low:+,.0f} to {high:+,.0f} · Probability positive: {probability_positive_label(bsts['simulation_probability_positive'])}",
            f"{bsts.get('reliability', 'Review')} reliability",
            [
                ("Observed", number(bsts["observed_total"], 0), ""),
                ("Expected Without Campaign", number(bsts["expected_total"], 0), ""),
                (incremental_outcome_label(cfg.outcome_column), number(bsts["cumulative_impact"], 0), "positive" if impact_positive else "negative"),
                ("Relative Impact", percent(bsts["relative_impact"], 1), "positive" if impact_positive else "negative"),
            ],
            analysis=True,
        )
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
        st.markdown("**Can This Estimate Be Trusted?**")
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
        with st.expander("Simple Before-and-After Comparison", expanded=False):
            st.metric("Before-and-After Change", percent(prepost["percent_change"], 1))
            st.caption("Pre–Post comparison does not adjust for trend, seasonality, or concurrent changes.")
            st.plotly_chart(pre_post_bar(prepost["pre_value"], prepost["post_value"], humanize_column_name(cfg.outcome_column)), use_container_width=True)
    if st.session_state.ts_structural_timestamp:
        st.caption(f"Last calculated: {st.session_state.ts_structural_timestamp}")


def ts_planning_step() -> None:
    """Pre-launch planning workflow; intentionally does not expose an analysis-method choice."""
    render_business_guide(
        "Tell us the smallest impact that would change a business decision and how long the campaign can run.",
        "Output: recommended readout date",
    )
    cfg: TimeSeriesConfig = st.session_state.ts_config
    data = st.session_state.ts_prepared_df
    if data.empty:
        st.info("Load and map historical time-series data in the Data step before planning.")
        return
    launch = pd.Timestamp(cfg.planned_launch_date) if cfg.planned_launch_date else data["_date"].max() + pd.Timedelta(days=7)
    planning_data = data[data["_date"] < launch].copy()
    if launch <= data["_date"].max():
        st.warning("Planned campaign launch occurs before the end of the available historical data. Only data before launch will be used for planning.")
    planning_objective = st.radio(
        "What do you want to learn?",
        ["Find duration for a meaningful effect", "Find detectable effect by duration"],
        horizontal=True,
        format_func=lambda value: "How long should the campaign run?" if value.startswith("Find duration") else "What impact can each duration detect?",
        key="ts_plan_objective",
    )
    effect_type = st.selectbox(
        "How should impact be shown?",
        ["Relative Lift (%)", "Absolute Effect"],
        format_func=lambda value: "Percent change" if value == "Relative Lift (%)" else "Outcome units",
        key="ts_plan_effect_type",
    )
    if planning_objective == "Find duration for a meaningful effect":
        effect_label = "Smallest improvement worth acting on (%)" if effect_type == "Relative Lift (%)" else "Smallest improvement worth acting on (outcome units)"
        effect = st.number_input(effect_label, min_value=0.0, value=5.0 if effect_type == "Relative Lift (%)" else 200.0, step=0.5, key="ts_plan_effect", help="This is a business decision threshold, not a forecast of the campaign's actual effect.")
        planning_effect = effect / 100 if effect_type == "Relative Lift (%)" else effect
    else:
        effect = None
        planning_effect = None
        st.info("No campaign effect is assumed. The planner will estimate the minimum detectable effect for each candidate duration.")
    candidate_durations = planning_durations(cfg.frequency, 10_000)
    maximum = st.selectbox(
        "Longest campaign you can run",
        candidate_durations,
        index=len(candidate_durations) - 1,
        format_func=lambda value: duration_label(value, cfg.frequency),
        key=f"ts_plan_max_{cfg.frequency.lower()}",
    )
    st.caption(f"Longest campaign duration the planner will consider. Candidate durations use the detected {cfg.frequency.lower()} data cadence.")
    gradual = st.checkbox("Campaign reaches full strength gradually", key="ts_plan_ramp_enabled")
    default_ramp = {"Daily": 7, "Weekly": 4, "Monthly": 1}.get(cfg.frequency, 1)
    ramp = st.number_input(f"Ramp-up duration ({duration_unit(cfg.frequency)}s)", min_value=1, max_value=maximum, value=min(default_ramp, maximum), step=1, key=f"ts_plan_ramp_{cfg.frequency.lower()}") if gradual else 1
    effect_pattern = "gradual_ramp" if gradual else "constant"
    with st.expander("Advanced model settings", expanded=False):
        c1, c2 = st.columns(2)
        alpha = c1.selectbox("False-positive tolerance", [0.05, 0.10], format_func=lambda value: f"{value:.0%}", key="ts_plan_alpha")
        target_power = c2.selectbox("Detection confidence target", [0.8, 0.9], format_func=lambda value: f"{int(value * 100)}%", key="ts_plan_power")
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
        calculate_label = "Update Recommendation" if st.session_state.ts_duration_plan and st.session_state.ts_duration_plan_fingerprint != current_plan_fingerprint else "Build Recommendation"
        spinner_text = "Calculating minimum detectable effects..."
    else:
        calculate_label = "Update Recommendation" if st.session_state.ts_duration_plan and st.session_state.ts_duration_plan_fingerprint != current_plan_fingerprint else "Build Recommendation"
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
            st.success("Recommendation ready. Detectable impacts are shown for each duration.")
        elif plan_status.get("calibration_status") == "poor":
            st.warning("Recommendation ready, but historical validation needs review.")
        elif plan_status.get("recommended_duration") and plan_status.get("calibration_status") == "caution":
            st.info("Recommendation ready with limited historical validation.")
        elif plan_status.get("recommended_duration"):
            st.success("Recommendation ready.")
        else:
            st.info("The selected improvement is not detectable within the longest campaign duration.")
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
    if recommendation:
        selected = next(row for row in plan["power"] if row["duration"] == recommendation)
        stable = max(1, recommendation - int(ramp))
        launch_anchor = pd.Timestamp(cfg.planned_launch_date or cfg.intervention_date)
        end = campaign_decision_date(launch_anchor, recommendation, cfg.frequency)
        render_ts_kpi_panel(
            "Recommended Campaign Plan",
            f"Shortest evaluated option expected to detect the {plan['effect']}{'%' if plan['effect_type'] == 'Relative Lift (%)' else '-unit'} planning threshold.",
            f"{confidence} confidence",
            [
                ("Required Duration", duration_label(recommendation, cfg.frequency), ""),
                ("Post-Launch Sample", f"{recommendation:,} time points", ""),
                ("Estimated Detection Chance", f"{selected['power']:.0%}", "positive"),
                ("Earliest Reliable Decision", end.strftime("%b %-d, %Y"), ""),
            ],
        )
        st.caption(f"Ramp-up: {duration_label(ramp, cfg.frequency)} · Stable measurement: {duration_label(stable, cfg.frequency)} · Historical observations used: {len(planning_data):,}")
        operational_exposure = projected_operational_exposure(planning_data, cfg.exposure_column, recommendation)
        if operational_exposure is not None:
            st.metric("Expected Operational Exposure", f"{operational_exposure:,.0f}", help="Operational projection: median historical exposure per period multiplied by the recommended duration. This is not a statistical customer sample size.")
        st.markdown('<div class="ts-panel-title">Historical Series and Planned Campaign Window</div>', unsafe_allow_html=True)
        st.plotly_chart(time_series_line(data, "", humanize_column_name(cfg.outcome_column), planned_launch_date=cfg.planned_launch_date, recommended_end_date=end.isoformat()), use_container_width=True)
    elif effect is not None:
        render_ts_kpi_panel(
            "More Time or a Larger Effect Is Needed",
            "The selected effect does not reach the detection target within the longest campaign duration.",
            "Review design",
            [
                ("Longest Duration Checked", duration_label(int(maximum), cfg.frequency), ""),
                ("Effect Scenario", f"{effect:.1f}%" if effect_type == "Relative Lift (%)" else f"{effect:,.1f}", ""),
                ("Target Detection Chance", f"{target_power:.0%}", ""),
                ("Historical Observations", f"{len(planning_data):,}", ""),
            ],
        )
    else:
        reference_row = next((row for row in plan["power"] if row["duration"] == 12 and pd.notna(row.get("minimum_detectable_effect"))), None)
        if reference_row is None:
            reference_row = next((row for row in reversed(plan["power"]) if pd.notna(row.get("minimum_detectable_effect"))), None)
        if reference_row:
            mde = reference_row["minimum_detectable_effect"]
            display_mde = f"{mde * 100:.1f}%" if effect_type == "Relative Lift (%)" else f"{mde:,.1f}"
            render_ts_kpi_panel(
                "Detectable Effect Reference",
                "The smallest modeled effect expected to meet the target at the reference duration. This is not a forecast.",
                "Sensitivity estimate",
                [
                    ("Reference Duration", duration_label(reference_row["duration"], cfg.frequency), ""),
                    ("Minimum Detectable Effect", display_mde, ""),
                    ("Target Detection Chance", f"{target_power:.0%}", "positive"),
                    ("Historical Observations", f"{len(planning_data):,}", ""),
                ],
            )
    with st.expander("Validation evidence", expanded=False):
        st.write(f"{validation_mode} · Recommendation confidence: {confidence}")
        st.caption(f"{plan.get('historical_origins', 0):,} historical pseudo-launch origins · Minimum training history: {plan.get('minimum_training_observations', 0):,} observations · Predictors: {len(plan.get('predictors', [])):,} · Planner seasonality: {plan.get('model_seasonality', 'None')}")
        if validation_mode == "Assumption-based":
            st.warning(plan.get("assumption_note", "History is too short for empirical pseudo-launch validation. Treat this estimate as preliminary."))
        elif validation_mode != "Robust empirical":
            st.info("Historical validation is limited. Forecast paths supplement the available pseudo-launch origins, so treat this result as directional rather than fully empirical.")
    evaluated_rows = sorted(plan["power"], key=lambda row: int(row["duration"]))
    if effect is None:
        st.markdown("**Smallest Detectable Impact by Duration**")
        st.caption("Use this table to see whether a longer campaign produces a meaningfully more sensitive readout.")
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
                    "Smallest Detectable Lift" if effect_type == "Relative Lift (%)" else "Smallest Detectable Impact": display_mde,
                    "Benefit vs Shorter Option": gain,
                }
            )
            if pd.notna(mde):
                previous_mde = mde
                previous_duration = duration
        sensitivity = pd.DataFrame(sensitivity_rows)
        st.dataframe(sensitivity, hide_index=True, use_container_width=True)
    else:
        st.markdown("**Detection Chance by Campaign Duration**")
        st.caption(f"Chance of detecting the assumed {effect:.1f}{'%' if effect_type == 'Relative Lift (%)' else '-unit'} improvement when it is real. The shortest reliable option is recommended.")
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
                    "Smallest Detectable Lift" if effect_type == "Relative Lift (%)" else "Smallest Detectable Impact": f"{mde * 100:.1f}%" if effect_type == "Relative Lift (%)" and pd.notna(mde) else f"{mde:,.1f}" if pd.notna(mde) else "Not evaluated",
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
    with st.expander("Model validation (advanced)", expanded=False):
        if plan["calibration_status"] == "assumption_based":
            st.info(f"Validation is assumption-based because the available history is too short for held-out placebo checks. The calculation uses a nominal {alpha:.0%} false-positive rate.")
        elif plan["calibration_status"] == "acceptable":
            st.success(f"Historical validation is consistent with the selected {alpha:.0%} false-positive tolerance.")
        else:
            st.warning("Historical false-alarm behavior needs review, so treat the duration and detectable-impact estimates cautiously.")
        if plan["calibration_status"] != "assumption_based":
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
    with st.expander("How this recommendation was calculated (advanced)", expanded=False):
        st.write(f"One observation is one {cfg.frequency.lower()} aggregate. The planner uses {validation_mode.lower()} validation. Pseudo-launches use a consistent model specification and at least {plan.get('minimum_training_observations', 0)} pre-launch observations. Calibration origins determine a one-sided critical threshold for a planned positive effect; separate held-out origins estimate false-positive rate and power. Sparse historical false alarms are evaluated with an exact binomial test. Correlated no-campaign forecast paths are reused to evaluate assumed effects and to solve for the minimum detectable effect at each duration. When history is too short for structural validation, the planner switches to a labeled analytical approximation using detrended residual variance and autocorrelation. A supplied effect is a scenario, not a forecast. When simulation intervals are available, the lower bound must clear the target before a duration is recommended. Every candidate through the selected maximum duration is evaluated. Required Time-Series Sample means post-launch time points, not customers or transactions.")


def ts_decision_step() -> None:
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
    if cfg.analysis_method == "Structural Time Series Counterfactual":
        low, high = result["prediction_interval"]
        interp = build_business_interpretation(result["cumulative_impact"], result["relative_impact"], result["prediction_interval"], result["simulation_probability_positive"], result.get("reliability", "Poor"), cfg.outcome_column)
        conclusion = "Positive evidence — high confidence" if result.get("decision") == "positive" and result.get("reliability") == "Good" else "Positive evidence — moderate confidence" if result.get("decision") == "positive" else "Negative evidence — high confidence" if result.get("decision") == "negative" and result.get("reliability") == "Good" else "Negative evidence — moderate confidence" if result.get("decision") == "negative" else "Inconclusive"
        positive = result["cumulative_impact"] >= 0
        render_ts_kpi_panel(
            "Campaign Decision",
            interp["interpretation"],
            conclusion,
            [
                ("Estimated Lift", percent(result["relative_impact"], 1), "positive" if positive else "negative"),
                (incremental_outcome_label(cfg.outcome_column), number(result["cumulative_impact"], 0), "positive" if positive else "negative"),
                ("Likely Impact Range", f"{low:+,.0f} to {high:+,.0f}", ""),
                ("Model Reliability", result.get("reliability", "Review"), ""),
            ],
            analysis=True,
        )
        st.markdown("**Recommended Next Step**")
        st.write(interp["next_step"])
    else:
        positive = result["absolute_change"] >= 0
        render_ts_kpi_panel(
            "Before-and-After Summary",
            "A descriptive comparison only; use a counterfactual method when a causal estimate is required.",
            "Descriptive evidence",
            [
                ("Before Campaign", number(result["pre_value"], 2), ""),
                ("After Campaign", number(result["post_value"], 2), ""),
                ("Absolute Change", number(result["absolute_change"], 2), "positive" if positive else "negative"),
                ("Percent Change", percent(result["percent_change"], 1), "positive" if positive else "negative"),
            ],
            analysis=True,
        )
        st.warning("Pre–Post results are descriptive and should not be interpreted as causal impact.")
    with st.expander("What this recommendation covers", expanded=False):
        st.write(f"Outcome: {humanize_column_name(cfg.outcome_column)}")
        st.write(f"Evaluation Period: {cfg.intervention_date} – {cfg.campaign_end_date or st.session_state.ts_prepared_df['_date'].max().date()}")


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
    latest["label"] = latest["_dma"].replace(
        {
            "Dallas-Ft. Worth": "Dallas",
            "Miami-Ft. Lauderdale": "Miami",
            "Minneapolis-St. Paul": "Minneapolis",
            "San Francisco-Oakland-San Jose": "San Francisco",
            "Tampa-St. Petersburg": "Tampa",
            "Washington DC": "DC",
        }
    )
    return latest.rename(columns={"_dma": "dma"})


def start_geography_workflow(intent: str) -> None:
    """Enter a workflow with the visible demo data when no workspace data exists yet."""
    if st.session_state.geo_raw_df.empty:
        cfg: GeoConfig = st.session_state.geo_config
        cfg.data_source = "Synthetic Demo"
        cfg.dataset_name = "Synthetic DMA Campaign Panel"
        st.session_state.geo_raw_df = load_geo_demo()
    st.session_state.geo_intent = intent


def geo_landing_preview() -> None:
    """Render a read-only overview that explains the geography workflow at a glance."""
    cfg: GeoConfig = st.session_state.geo_config
    using_workspace = not st.session_state.geo_raw_df.empty
    raw = st.session_state.geo_raw_df if using_workspace else load_geo_demo()
    inferred = infer_geo_schema(raw)
    columns = list(raw.columns)
    date_col = cfg.date_column if using_workspace and cfg.date_column in columns else inferred["date"]
    dma_col = cfg.dma_column if using_workspace and cfg.dma_column in columns else inferred["dma"]
    outcome_col = cfg.outcome_column if using_workspace and cfg.outcome_column in columns else inferred["outcome"]
    group_col = cfg.group_column if using_workspace and cfg.group_column in columns else inferred["group"]
    market_size_col = cfg.market_size_column if using_workspace and cfg.market_size_column in columns else inferred["market_size"]

    try:
        panel = prepare_geo_panel(raw, date_col, dma_col, outcome_col)
        validation = validate_fixed_assignment(panel, group_col, test_label=cfg.test_label, control_label=cfg.control_label)
        assignment = validation["assignment"] if not validation["errors"] else pd.DataFrame(columns=["DMA", "Group"])
        preview = build_geo_preview(panel, assignment, market_size_col)
    except Exception:
        raw = load_geo_demo()
        panel = prepare_geo_panel(raw, "week", "dma", "applications")
        assignment = validate_fixed_assignment(panel, "treatment_group")["assignment"]
        preview = build_geo_preview(panel, assignment, "population")
        date_col, dma_col, outcome_col = "week", "dma", "applications"
        using_workspace = False

    quality = preview["quality"]
    counts = preview["counts"]
    total_markets = int(quality["dma_count"])
    test_markets = int(counts.get("Test", 0))
    control_markets = int(counts.get("Control", 0))
    assigned_total = max(test_markets + control_markets, 1)
    start_date = pd.Timestamp(preview["start_date"])
    end_date = pd.Timestamp(preview["end_date"])
    period_label = f"{start_date:%b %Y} – {end_date:%b %Y}"
    source_label = "your current dataset" if using_workspace else "the built-in demo"

    action_copy, plan_action, analyze_action = st.columns([2.4, 1.15, 1.15])
    action_copy.markdown(
        f'<div class="geo-demo-actions"><div><strong>See the market story first</strong><span>Previewing {html.escape(source_label)}. Choose a workflow when the structure looks right.</span></div></div>',
        unsafe_allow_html=True,
    )
    if plan_action.button("Plan a Geographic Test", type="primary", use_container_width=True, key="geo_demo_plan"):
        start_geography_workflow("plan")
        st.rerun()
    if analyze_action.button("Analyze Test Results", use_container_width=True, key="geo_demo_analyze"):
        start_geography_workflow("analyze")
        st.rerun()

    map_col, summary_col = st.columns([1.75, 0.75])
    with map_col:
        with st.container(border=True):
            st.plotly_chart(
                geo_dma_map(geo_map_frame(panel, assignment), "", height=420, show_labels=True),
                use_container_width=True,
                config={"displayModeBar": False},
            )
    with summary_col:
        test_share = test_markets / assigned_total * 100
        control_share = control_markets / assigned_total * 100
        assignment_note = (
            "Assignments are fixed. The same Test and Control markets are used for planning and analysis."
            if not assignment.empty
            else "The data is visible, but Test and Control labels still need to be confirmed."
        )
        st.markdown(
            f"""
            <section class="geo-preview-summary">
              <h3>Data for this experiment</h3>
              <div class="geo-preview-grid">
                <div class="geo-preview-stat"><div class="geo-preview-label">Markets</div><div class="geo-preview-value">{total_markets:,}</div><div class="geo-preview-detail">{test_markets} Test · {control_markets} Control</div></div>
                <div class="geo-preview-stat"><div class="geo-preview-label">Time period</div><div class="geo-preview-value">{html.escape(period_label)}</div><div class="geo-preview-detail">{html.escape(str(quality['frequency']))} data</div></div>
                <div class="geo-preview-stat"><div class="geo-preview-label">Outcome</div><div class="geo-preview-value">{html.escape(humanize_column_name(outcome_col))}</div><div class="geo-preview-detail">Compared through time</div></div>
                <div class="geo-preview-stat"><div class="geo-preview-label">Coverage</div><div class="geo-preview-value">{int(quality['period_count']):,} periods</div><div class="geo-preview-detail">{len(raw):,} source rows</div></div>
              </div>
              <div class="geo-assignment-bars">
                <div class="geo-assignment-row"><span>Test</span><div class="geo-assignment-track"><div class="geo-assignment-fill" style="width:{test_share:.1f}%"></div></div><b>{test_markets}</b></div>
                <div class="geo-assignment-row"><span>Control</span><div class="geo-assignment-track"><div class="geo-assignment-fill control" style="width:{control_share:.1f}%"></div></div><b>{control_markets}</b></div>
              </div>
              <div class="geo-preview-note">{html.escape(assignment_note)}</div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="geo-section-heading"><strong>How Test and Control move together</strong><span>The launch marker shows where impact measurement begins.</span></div>',
        unsafe_allow_html=True,
    )
    launch_date = "2026-09-06" if not using_workspace else cfg.campaign_start_date
    st.plotly_chart(
        geo_trend_chart(preview["trend"], launch_date, humanize_column_name(outcome_col)),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    market_preview = preview["markets"].copy()
    market_preview["Average Outcome"] = market_preview["Average Outcome"].round(0)
    market_preview["Start Date"] = pd.to_datetime(market_preview["Start Date"]).dt.strftime("%b %Y")
    market_preview["Latest Date"] = pd.to_datetime(market_preview["Latest Date"]).dt.strftime("%b %Y")
    if "Market Size" in market_preview:
        market_preview["Market Size"] = market_preview["Market Size"].round(0)
    st.markdown(
        f'<div class="geo-section-heading"><strong>Market data preview</strong><span>Showing {min(10, len(market_preview))} of {len(market_preview)} markets from {html.escape(source_label)}.</span></div>',
        unsafe_allow_html=True,
    )
    st.dataframe(market_preview.head(10), use_container_width=True, hide_index=True, height=300)


def geo_data_step() -> None:
    render_business_guide(
        "Use one row per market and time period. Your existing Test and Control markets will never be rematched.",
        "Output: usable market history",
    )
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

    st.markdown("**Tell us what each column means**")
    c1, c2, c3, c4 = st.columns(4)
    cfg.date_column = c1.selectbox("Date", columns, index=columns.index(cfg.date_column))
    cfg.dma_column = c2.selectbox("Market / DMA", columns, index=columns.index(cfg.dma_column))
    cfg.outcome_column = c3.selectbox("Business Outcome", numeric, index=numeric.index(cfg.outcome_column) if cfg.outcome_column in numeric else 0)
    cfg.group_column = c4.selectbox("Test Group", ["None"] + columns, index=(["None"] + columns).index(cfg.group_column))
    with st.expander("Advanced data fields", expanded=False):
        c5, c6, c7 = st.columns(3)
        cfg.market_size_column = c5.selectbox("Market-size Weight", ["None"] + columns, index=(["None"] + columns).index(cfg.market_size_column), help="Optional. Use a positive market-size field only when large DMAs should contribute more to the result.")
        cfg.pair_column = c6.selectbox("Pair ID", ["None"] + columns, index=(["None"] + columns).index(cfg.pair_column), help="Optional. Use only when pair IDs were defined before launch. The platform will not create pairs.")
        cfg.spend_column = c7.selectbox("Campaign Spend / Exposure", ["None"] + columns, index=(["None"] + columns).index(cfg.spend_column), help="Optional metadata for reporting.")
    try:
        panel = prepare_geo_panel(raw, cfg.date_column, cfg.dma_column, cfg.outcome_column)
        st.session_state.geo_panel_df = panel
        quality = validate_geo_panel(panel)
        min_date, max_date = panel["_date"].min().date(), panel["_date"].max().date()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Markets", f"{quality['dma_count']:,.0f}")
        m2.metric("Time Periods", f"{quality['period_count']:,.0f}")
        m3.metric("Frequency", str(quality["frequency"]))
        m4.metric("Date Range", f"{min_date} to {max_date}")
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
        st.markdown(
            '<div class="geo-section-heading"><strong>Preview the market panel</strong><span>Confirm the geography, time coverage, and source columns before continuing.</span></div>',
            unsafe_allow_html=True,
        )
        map_preview, table_preview = st.columns([0.9, 1.1])
        with map_preview:
            st.plotly_chart(
                geo_dma_map(geo_map_frame(panel, st.session_state.geo_assignment), "", height=350),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            if st.session_state.geo_assignment.empty:
                st.caption("Markets are shown in gray until Test and Control labels are confirmed.")
            else:
                st.caption("Red markets are Test; blue markets are Control. The supplied assignment is not changed.")
        with table_preview:
            st.markdown(
                f'<div class="geo-data-preview-note">Showing 12 of {len(raw):,} source rows · all {len(raw.columns):,} columns</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(raw.head(12), use_container_width=True, height=350)
    except Exception as exc:
        st.error("Data validation issue: We could not interpret the selected Date, DMA, and Outcome columns.")
        print(f"Geographic preparation failed: {exc}")


def geo_assignment_step(show_heading: bool = True, balance_date: str | None = None) -> bool:
    if show_heading:
        render_business_guide(
            "Confirm which labels mean Test and Control, then review the markets included in each group.",
            "Output: confirmed market groups",
        )
    cfg: GeoConfig = st.session_state.geo_config
    panel = st.session_state.geo_panel_df
    if panel.empty:
        st.info("Upload and map a DMA panel first.")
        return False
    if cfg.group_column == "None":
        st.info("Map the Test / Control Group column on the Data tab.")
        return False
    source_labels = sorted(panel[cfg.group_column].dropna().astype(str).str.strip().unique().tolist())
    if not source_labels:
        st.error("The selected group column does not contain any labels.")
        return False
    if cfg.test_label not in source_labels:
        cfg.test_label = next((value for value in source_labels if value.lower() in {"test", "treatment", "treated"}), source_labels[0])
    if cfg.control_label not in source_labels:
        cfg.control_label = next((value for value in source_labels if value.lower() in {"control", "holdout", "bau"}), source_labels[min(1, len(source_labels) - 1)])
    c1, c2 = st.columns(2)
    cfg.test_label = c1.selectbox("Which label means Test?", source_labels, index=source_labels.index(cfg.test_label), key="geo_test_label")
    cfg.control_label = c2.selectbox("Which label means Control?", source_labels, index=source_labels.index(cfg.control_label), key="geo_control_label")
    validation = validate_fixed_assignment(panel, cfg.group_column, cfg.pair_column, cfg.test_label, cfg.control_label)
    for issue in validation["errors"]:
        st.error(issue)
    for warning in validation["warnings"]:
        st.warning(warning)
    if validation["errors"]:
        st.session_state.geo_assignment = pd.DataFrame()
        return False
    assignment = validation["assignment"]
    st.session_state.geo_assignment = assignment
    counts = validation["counts"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Test DMAs", counts.get("Test", 0))
    m2.metric("Control DMAs", counts.get("Control", 0))
    m3.metric("Total DMAs", len(assignment))
    st.success("Groups confirmed. These markets stay fixed throughout planning and analysis.")
    left, right = st.columns([1.25, 0.75])
    with left:
        st.plotly_chart(geo_dma_map(geo_map_frame(panel, assignment), "Provided Test and Control Assignment"), use_container_width=True)
    with right:
        st.dataframe(assignment, use_container_width=True, hide_index=True, height=390)
    missing_geo = [dma for dma in assignment["DMA"] if dma not in DMA_CENTROIDS]
    if missing_geo:
        st.caption(f"{len(missing_geo)} DMA(s) have no stored map centroid and are omitted from the map; they remain in all calculations.")
    comparison_date = balance_date or cfg.planned_launch_date or panel["_date"].max().date().isoformat()
    balance = evaluate_geo_balance(panel, assignment, comparison_date)
    st.session_state.geo_balance = balance
    if balance:
        st.markdown(
            f'<div class="geo-result-banner"><strong>Historical comparability: {html.escape(balance["status"])}</strong><span>This is a diagnostic only. Your supplied Test and Control groups are never changed.</span></div>',
            unsafe_allow_html=True,
        )
        with st.expander("Review historical comparability", expanded=False):
            m1, m2, m3 = st.columns(3)
            m1.metric("Historical Correlation", number(balance["correlation"], 2))
            m2.metric("Mean Relative Difference", percent(balance["relative_difference"], 1))
            m3.metric("Trend Difference", percent(balance["trend_difference"], 1))
            st.plotly_chart(geo_balance_chart(balance["aggregate"], comparison_date, cfg.outcome_column), use_container_width=True)
            st.caption("These diagnostics describe comparability; they do not change the provided assignment.")
    return True


def geo_planning_step() -> None:
    render_business_guide(
        "Choose the smallest change worth acting on. Historical market variation determines the duration and number of markets needed.",
        "Output: duration and market requirement",
    )
    cfg: GeoConfig = st.session_state.geo_config
    panel = st.session_state.geo_panel_df
    assignment = st.session_state.geo_assignment
    if panel.empty or assignment.empty:
        st.info("Map and validate the fixed A/B assignment before planning rollout.")
        return
    quality = validate_geo_panel(panel)
    frequency = str(quality["frequency"])
    c1, c2, c3, c4 = st.columns(4)
    cfg.planned_launch_date = c1.date_input("Planned Launch Date", value=pd.Timestamp(cfg.planned_launch_date).date(), min_value=panel["_date"].min().date()).isoformat()
    cfg.effect_type = c2.selectbox("Effect Scale", ["Relative lift", "Absolute change"], index=["Relative lift", "Absolute change"].index(cfg.effect_type), help="Use relative lift for a percentage change, or absolute change when the metric has a natural unit.")
    effect_label = "Smallest Lift Worth Acting On (%)" if cfg.effect_type == "Relative lift" else f"Smallest Change Worth Acting On ({cfg.outcome_column})"
    cfg.expected_effect = c3.number_input(effect_label, min_value=0.01, value=float(cfg.expected_effect), step=0.5 if cfg.effect_type == "Relative lift" else 1.0)
    cfg.max_duration = c4.number_input(f"Longest Campaign You Can Run ({duration_unit(frequency)}s)", min_value=2, max_value=104, value=int(cfg.max_duration), step=1)
    with st.expander("Advanced settings", expanded=False):
        c5, c6, c7 = st.columns(3)
        cfg.target_power = c5.slider("Target Detection Chance", min_value=0.60, max_value=0.95, value=float(cfg.target_power), step=0.05)
        cfg.alpha = c6.select_slider("False-positive Rate", options=[0.01, 0.025, 0.05, 0.10], value=float(cfg.alpha))
        cfg.estimand = c7.radio("DMA Weighting", ["Equal weight per DMA", "Market-size weighted"], index=["Equal weight per DMA", "Market-size weighted"].index(cfg.estimand), horizontal=True)
        c8, c9 = st.columns(2)
        cfg.ramp_periods = c8.number_input("Ramp-up Periods", min_value=0, max_value=24, value=int(cfg.ramp_periods), step=1)
        cfg.outcome_delay_periods = c9.number_input("Outcome Maturation Delay", min_value=0, max_value=52, value=int(cfg.outcome_delay_periods), step=1)
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
        recommended = plan["recommended_duration"]
        row = plan["table"].loc[plan["table"]["Campaign Duration"] == recommended].iloc[0] if recommended is not None else plan["table"].iloc[-1]
        if recommended is None:
            panel_title = "More scale or time is needed"
            panel_subtitle = "The selected effect is not detectable within the longest duration. Review the alternatives below."
            panel_status = "Review design"
        else:
            panel_title = "Recommended Plan"
            panel_subtitle = "The shortest evaluated option meeting the selected detection target."
            panel_status = "Ready to review"
        render_geo_kpi_panel(
            panel_title,
            panel_subtitle,
            panel_status,
            [
                ("Recommended Duration", duration_label(int(recommended), frequency) if recommended is not None else "Not reached", ""),
                ("Required Test DMAs", str(int(row["Required Test DMAs"])), "test"),
                ("Required Control DMAs", str(int(row["Required Control DMAs"])), "control"),
                ("Earliest Readout", str(row["Earliest Readout"]), ""),
            ],
            plan=True,
        )
        st.caption(f"Current assignment: {plan['current_test_dmas']} Test DMAs and {plan['current_control_dmas']} Control DMAs.")
        planning_panel = prepare_geo_analysis(panel, assignment, cfg.planned_launch_date, weight_col=cfg.market_size_column)
        visual, group_list = st.columns([1.35, 0.65])
        with visual:
            st.markdown('<div class="geo-panel-title">Your DMA Groups</div>', unsafe_allow_html=True)
            st.plotly_chart(geo_dma_map(geo_map_frame(panel, assignment), ""), use_container_width=True)
        with group_list:
            st.markdown('<div class="geo-panel-title">Markets from Your Data</div>', unsafe_allow_html=True)
            st.dataframe(assignment, use_container_width=True, hide_index=True, height=390)
        st.markdown('<div class="geo-panel-title">Historical Test and Control Trend</div>', unsafe_allow_html=True)
        st.plotly_chart(geo_trend_chart(aggregate_trend(planning_panel, "Indexed"), cfg.planned_launch_date, "Indexed Outcome"), use_container_width=True)
        display = plan["table"].copy()
        display["Campaign Duration"] = display["Campaign Duration"].apply(lambda value: duration_label(int(value), frequency))
        display["Detection Chance"] = display["Detection Chance"].apply(lambda value: percent(value, 0))
        display = display[["Campaign Duration", "Detection Chance", "Required Test DMAs", "Required Control DMAs", "Earliest Readout", "Status"]]
        st.markdown("**Other Duration Options**")
        st.dataframe(display, use_container_width=True, hide_index=True)
        if recommended is None:
            st.warning("Add markets, accept a larger detectable effect, or extend the longest campaign duration.")
        with st.expander("How this plan was calculated (advanced)", expanded=False):
            st.write(plan["method_note"])
            st.write("Historical Test-minus-Control variation is detrended, adjusted for lag-1 autocorrelation, and converted into the uncertainty of an average campaign effect. Detection chance uses a two-sided normal test. Required DMAs scale with squared uncertainty relative to the supplied effect scenario; duration and DMA count are separate design levers.")
            st.write(f"Historical periods used: {plan['historical_periods']} | Estimated lag-1 autocorrelation: {number(plan['lag1_autocorrelation'], 2)}")
    except Exception as exc:
        st.session_state.geo_plan = None
        st.error(str(exc))


def geo_analysis_signature() -> tuple:
    cfg: GeoConfig = st.session_state.geo_config
    assignment = st.session_state.geo_assignment
    return (cfg.campaign_start_date, cfg.campaign_end_date, cfg.outcome_column, cfg.estimand, tuple(zip(assignment.get("DMA", []), assignment.get("Group", []))))


def geo_analysis_setup_step() -> None:
    render_business_guide(
        "Confirm when the campaign started and which markets were Test versus Control.",
        "Output: campaign measurement window",
    )
    cfg: GeoConfig = st.session_state.geo_config
    panel = st.session_state.geo_panel_df
    if panel.empty:
        st.info("Upload and map a DMA panel first.")
        return
    min_date, max_date = panel["_date"].min().date(), panel["_date"].max().date()
    c1, c2 = st.columns(2)
    cfg.campaign_start_date = c1.date_input("When did the campaign start?", value=pd.Timestamp(cfg.campaign_start_date).date(), min_value=min_date, max_value=max_date).isoformat()
    end_enabled = c2.checkbox("The campaign had a fixed end date", value=bool(cfg.campaign_end_date), key="geo_actual_end_enabled")
    cfg.campaign_end_date = c2.date_input("Analysis End", value=pd.Timestamp(cfg.campaign_end_date or max_date).date(), min_value=pd.Timestamp(cfg.campaign_start_date).date(), max_value=max_date).isoformat() if end_enabled else ""
    st.markdown("**Test and Control Markets**")
    valid = geo_assignment_step(show_heading=False, balance_date=cfg.campaign_start_date)
    if not valid:
        return
    with st.expander("Advanced analysis settings", expanded=False):
        c3, c4 = st.columns(2)
        cfg.outcome_view = c3.radio("Chart View", ["Indexed", "Absolute"], index=["Indexed", "Absolute"].index(cfg.outcome_view), horizontal=True)
        cfg.estimand = c4.radio("DMA Weighting", ["Equal weight per DMA", "Market-size weighted"], index=["Equal weight per DMA", "Market-size weighted"].index(cfg.estimand), horizontal=True)
    if cfg.estimand == "Market-size weighted" and cfg.market_size_column == "None":
        st.error("Select a Market-size Weight column on Data or use equal weight per DMA.")
        return
    st.markdown('<div class="geo-result-banner"><strong>Ready to analyze</strong><span>Open Results and run the Difference-in-Differences analysis.</span></div>', unsafe_allow_html=True)


def geo_analysis_results_step() -> None:
    render_business_guide(
        "Review the estimated incremental impact first. Detailed statistical checks remain available below.",
        "Output: campaign impact",
    )
    cfg: GeoConfig = st.session_state.geo_config
    panel = st.session_state.geo_panel_df
    assignment = st.session_state.geo_assignment
    if panel.empty or assignment.empty:
        st.info("Complete Data and Setup before running the analysis.")
        return
    weighted = cfg.estimand == "Market-size weighted"
    if weighted and cfg.market_size_column == "None":
        st.error("Select a Market-size Weight column on Data or use equal weight per DMA in Setup.")
        return
    signature = geo_analysis_signature()
    try:
        analysis_panel = prepare_geo_analysis(panel, assignment, cfg.campaign_start_date, cfg.campaign_end_date, cfg.market_size_column)
        st.session_state.geo_balance = evaluate_geo_balance(panel, assignment, cfg.campaign_start_date)
        trend = aggregate_trend(analysis_panel, cfg.outcome_view)
        current = bool(st.session_state.geo_analysis and st.session_state.geo_analysis.get("_signature") == signature)
        action_label = "Recalculate Impact" if current else "Calculate Impact"
        if st.button(action_label, type="primary", key="geo_run_analysis"):
            result = run_panel_did(analysis_panel, weighted=weighted)
            result["_signature"] = signature
            st.session_state.geo_analysis = result
            st.rerun()
        elif st.session_state.geo_analysis and st.session_state.geo_analysis.get("_signature") != signature:
            st.warning("Results are out of date for the current assignment, dates, outcome, or weighting. Run the analysis again.")
    except Exception as exc:
        st.error("Geographic analysis could not run with the current assignment and campaign window.")
        print(f"Geographic analysis failed: {exc}")
        return

    result = st.session_state.geo_analysis
    if result and result.get("_signature") == signature:
        low, high = result["interval"]
        if pd.notna(low) and low > 0:
            headline = "Evidence of positive incremental impact"
            explanation = "The uncertainty interval is above zero for the configured campaign period."
        elif pd.notna(high) and high < 0:
            headline = "Evidence of negative incremental impact"
            explanation = "The uncertainty interval is below zero for the configured campaign period."
        else:
            headline = "Result is not yet conclusive"
            explanation = "The estimated effect may be meaningful, but the uncertainty interval still includes zero."
        render_geo_kpi_panel(
            headline,
            explanation,
            "Reliable result" if result["reliability"] == "Good" else "Review diagnostics",
            [
                ("Estimated Relative Lift", percent(result["lift"], 1), "control" if result["lift"] >= 0 else "test"),
                ("95% Effect Interval", f"{number(low, 2)} to {number(high, 2)}", ""),
                ("Incremental Outcome", number(result["incremental_volume"], 0), ""),
                ("Result Reliability", result["reliability"], "control" if result["reliability"] == "Good" else ""),
            ],
        )
        trend_col, map_col = st.columns([1.25, 0.75])
        with trend_col:
            st.markdown('<div class="geo-panel-title">Test vs Control Trend</div>', unsafe_allow_html=True)
            st.plotly_chart(geo_trend_chart(trend, cfg.campaign_start_date, "Indexed Outcome" if cfg.outcome_view == "Indexed" else cfg.outcome_column), use_container_width=True)
        with map_col:
            st.markdown('<div class="geo-panel-title">Markets Analyzed</div>', unsafe_allow_html=True)
            st.plotly_chart(geo_dma_map(geo_map_frame(panel, assignment), ""), use_container_width=True)
        with st.expander("Model details and diagnostics", expanded=False):
            d1, d2, d3 = st.columns(3)
            d1.metric("Effect per DMA / Period", number(result["effect_per_period"], 2))
            d2.metric("P-value", p_value(result["pvalue"]))
            d3.metric("Markets Analyzed", result["cluster_count"])
            st.write(f"Model: {result['model']} | Estimand: {result['estimand']} | {result['test_dmas']} Test and {result['control_dmas']} Control DMA clusters")
            if pd.notna(result["pretrend_pvalue"]) and result["pretrend_pvalue"] < 0.05:
                st.warning("Pre-campaign trends differ statistically between Test and Control DMAs. Review the parallel-trends assumption before acting.")
            else:
                st.success("No statistically clear differential pre-trend was detected.")
        with st.expander("Robustness checks", expanded=False):
            if st.button("Run Placebo and Leave-One-DMA-Out Checks"):
                result["placebo"] = run_placebo_tests(analysis_panel, weighted=weighted)
                result["leave_one_out"] = leave_one_dma_out(analysis_panel, weighted=weighted)
            if "placebo" in result:
                st.write(f"Historical placebo stability: {result['placebo']['status']}")
                sensitivity = result.get("leave_one_out", pd.DataFrame()).copy()
                if not sensitivity.empty:
                    sensitivity["Relative Lift"] = sensitivity["Relative Lift"].apply(lambda value: percent(value, 1))
                    st.dataframe(sensitivity, use_container_width=True, hide_index=True)
        st.caption("The geographic market is the experimental unit. Do not interpret underlying customers as independently randomized observations.")


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


def leave_geographic_workspace() -> None:
    st.session_state.geo_intent = None
    st.session_state.main_page = "Overview"


def render_geo_shell(intent: str | None) -> None:
    accent = "#16856b" if intent == "analyze" else "#d69a1f"
    landing_secondary = "" if intent else '.stButton > button[kind="secondary"] {background:#16856b; border-color:#16856b; color:#fff;} .stButton > button[kind="secondary"] p {color:#fff;} .stButton > button[kind="secondary"]:hover {background:#11735f; border-color:#11735f; color:#fff;}'
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"], header[data-testid="stHeader"] {{display:none !important;}}
        .stApp {{background:#fbfcfe;}}
        .block-container {{max-width:1480px; padding:1rem 2rem 3rem;}}
        .stButton > button[kind="primary"] {{background:{accent}; border-color:{accent}; color:#fff;}}
        .stButton > button[kind="primary"]:hover {{background:{accent}; border-color:{accent}; filter:brightness(.96);}}
        {landing_secondary}
        @media (max-width:800px) {{.block-container {{padding:.75rem 1rem 2rem;}}}}
        </style>
        """,
        unsafe_allow_html=True,
    )
    if intent:
        brand, save_col, exit_col = st.columns([4.2, .8, 1])
    else:
        brand, exit_col = st.columns([5, 1])
    brand.markdown(
        '<div class="geo-product-bar"><div class="geo-product-mark">GEO</div><div><div class="geo-product-name">Experiment Platform</div><div class="geo-product-context">Market experimentation workspace</div></div></div>',
        unsafe_allow_html=True,
    )
    if intent and save_col.button("Save", use_container_width=True, key="geo_save_experiment", help="Save this configuration and the latest available result to the Command Center."):
        saved = save_current_workspace("Geography")
        st.toast(f'Saved {saved["name"]} · config v{saved["config_version"]}')
    exit_col.button("All experiments", use_container_width=True, key="geo_all_experiments", on_click=leave_geographic_workspace)


def render_geo_kpi_panel(title: str, subtitle: str, status: str, values: list[tuple[str, str, str]], plan: bool = False) -> None:
    cells = "".join(
        f'<div class="geo-kpi"><div class="geo-kpi-label">{html.escape(label)}</div><div class="geo-kpi-value {html.escape(tone)}">{html.escape(value)}</div></div>'
        for label, value, tone in values
    )
    st.markdown(
        f'<section class="geo-kpi-panel{" plan" if plan else ""}"><div class="geo-kpi-head"><div><strong>{html.escape(title)}</strong><span>{html.escape(subtitle)}</span></div><div class="geo-status">{html.escape(status)}</div></div><div class="geo-kpis">{cells}</div></section>',
        unsafe_allow_html=True,
    )


def geographic_page() -> None:
    intent = st.session_state.get("geo_intent")
    render_geo_shell(intent)
    st.markdown(
        '<div class="geo-page-heading"><h1>Geographic Experiment</h1><p>Use your existing Test and Control markets to plan a geographic test or analyze completed results.</p></div>',
        unsafe_allow_html=True,
    )
    if not intent:
        geo_landing_preview()
        return

    _, nav_right = st.columns([5, 1])
    if nav_right.button("Switch workflow", key="geo_switch_workflow"):
        st.session_state.geo_intent = None
        st.rerun()

    if intent == "plan":
        tab_data, tab_groups, tab_plan = st.tabs(["1 · Bring Data", "2 · Confirm Markets", "3 · Get Plan"])
        with tab_data:
            geo_data_step()
        with tab_groups:
            geo_assignment_step()
        with tab_plan:
            geo_planning_step()
        return

    tab_data, tab_setup, tab_results = st.tabs(["1 · Bring Data", "2 · Confirm Test", "3 · See Impact"])
    with tab_data:
        geo_data_step()
    with tab_setup:
        geo_analysis_setup_step()
    with tab_results:
        geo_analysis_results_step()


def leave_time_series_workspace() -> None:
    st.session_state.ts_intent = None
    st.session_state.main_page = "Overview"


def render_ts_shell(intent: str | None) -> None:
    accent = "#16856b" if intent == "analyze" else "#d69a1f"
    landing_secondary = "" if intent else '.stButton > button[kind="secondary"] {background:#16856b; border-color:#16856b; color:#fff;} .stButton > button[kind="secondary"] p {color:#fff;} .stButton > button[kind="secondary"]:hover {background:#11735f; border-color:#11735f; color:#fff;}'
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"], header[data-testid="stHeader"] {{display:none !important;}}
        .stApp {{background:#fbfcfe;}}
        .block-container {{max-width:1480px; padding:1rem 2rem 3rem;}}
        .stButton > button[kind="primary"] {{background:{accent}; border-color:{accent}; color:#fff;}}
        .stButton > button[kind="primary"]:hover {{background:{accent}; border-color:{accent}; filter:brightness(.96);}}
        button[data-baseweb="tab"][aria-selected="true"] {{color:{accent};}}
        {landing_secondary}
        @media (max-width:800px) {{.block-container {{padding:.75rem 1rem 2rem;}}}}
        </style>
        """,
        unsafe_allow_html=True,
    )
    if intent:
        brand, save_col, exit_col = st.columns([4.2, .8, 1])
    else:
        brand, exit_col = st.columns([5, 1])
    brand.markdown(
        '<div class="ts-product-bar"><div class="ts-product-mark">TIME</div><div><div class="ts-product-name">Experiment Platform</div><div class="ts-product-context">Time-series experimentation workspace</div></div></div>',
        unsafe_allow_html=True,
    )
    if intent and save_col.button("Save", use_container_width=True, key="ts_save_experiment", help="Save this configuration and the latest available result to the Command Center."):
        saved = save_current_workspace("Time Series")
        st.toast(f'Saved {saved["name"]} · config v{saved["config_version"]}')
    exit_col.button("All experiments", use_container_width=True, key="ts_all_experiments", on_click=leave_time_series_workspace)


def render_ts_kpi_panel(title: str, subtitle: str, status: str, values: list[tuple[str, str, str]], analysis: bool = False) -> None:
    cells = "".join(
        f'<div class="ts-kpi"><div class="ts-kpi-label">{html.escape(label)}</div><div class="ts-kpi-value {html.escape(tone)}">{html.escape(value)}</div></div>'
        for label, value, tone in values
    )
    st.markdown(
        f'<section class="ts-kpi-panel{" analysis" if analysis else ""}"><div class="ts-kpi-head"><div><strong>{html.escape(title)}</strong><span>{html.escape(subtitle)}</span></div><div class="ts-status">{html.escape(status)}</div></div><div class="ts-kpis">{cells}</div></section>',
        unsafe_allow_html=True,
    )


def time_series_page() -> None:
    intent = st.session_state.get("ts_intent")
    render_ts_shell(intent)
    st.markdown(
        '<div class="ts-page-heading"><h1>Time Series Experiment</h1><p>Plan or measure a campaign when no randomized control group is available.</p></div>',
        unsafe_allow_html=True,
    )
    if not intent:
        ts_landing_preview()
        return
    _, nav_right = st.columns([5, 1])
    if nav_right.button("Switch workflow", key="ts_switch_workflow"):
        st.session_state.ts_intent = None
        st.rerun()
    if intent == "plan":
        tabs = st.tabs(["1 · Bring Data", "2 · Set Launch", "3 · Get Recommendation"])
        with tabs[0]:
            ts_data_step()
        with tabs[1]:
            ts_plan_setup_step()
        with tabs[2]:
            ts_planning_step()
        return
    tabs = st.tabs(["1 · Bring Data", "2 · Mark Campaign", "3 · Choose Method", "4 · See Impact", "5 · Recommendation"])
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


def start_customer_workflow(intent: str) -> None:
    """Enter a customer workflow with demo data when no user dataset is active."""
    cfg: DataConfig = st.session_state.data_config
    if intent == "plan" and not st.session_state.get("customer_active_data_signature"):
        scenario = st.session_state.get("customer_demo_scenario", "General Customer Test")
        reset_customer_data_choices()
        st.session_state.raw_df = load_customer_demo(scenario)
        apply_demo_metric_default(scenario)
        cfg.source_type = "Synthetic Demo"
        cfg.dataset_name = f"{scenario} Demo"
        st.session_state.customer_active_data_signature = ("Synthetic Demo", scenario)
    if intent == "analyze" and st.session_state.results_df is None:
        st.session_state.results_df = load_results_demo()
        st.session_state["customer_analysis_source_v2"] = "Synthetic Demo"
    st.session_state.customer_intent = intent


def customer_landing_preview() -> None:
    """Show a business-readable customer data story before workflow configuration."""
    cfg: DataConfig = st.session_state.data_config
    active_signature = st.session_state.get("customer_active_data_signature")
    using_workspace = bool(active_signature) and not st.session_state.raw_df.empty
    scenario = st.session_state.get("customer_demo_scenario", "General Customer Test")
    raw = st.session_state.raw_df if using_workspace else load_customer_demo(scenario)
    mapping: DataMappingConfig = st.session_state.data_mapping
    population: PopulationConfig = st.session_state.population_config
    columns = list(raw.columns)
    unit_col = mapping.unit_id_column if mapping.unit_id_column in columns else default_column(columns, ["account_id", "customer_id", "user_id", "member_id"])
    product_col = mapping.cpc_column if mapping.cpc_column in columns else default_column(columns, ["cpc", "product", "brand", "portfolio"])
    date_col = mapping.booking_date_column if mapping.booking_date_column in columns else default_column(columns, ["booking_date", "start_date", "open_date", "acquisition_date"])
    period_col = mapping.mob_column if mapping.mob_column in columns else infer_mob_column(raw)

    scenario_fields = {
        "General Customer Test": ("primary_outcome", "historical_experience", "Continuous"),
        "Offer & Engagement": ("engagement_score", "historical_offer", "Continuous"),
        "Retention Strategy": ("retained_flag", "historical_retention_action", "Binary"),
        "Credit Strategy": ("revolving_balance", "current_credit_line", "Continuous"),
    }
    demo_outcome, demo_strategy, demo_metric_type = scenario_fields.get(scenario, ("", "", "Continuous"))
    primary = st.session_state.metrics_config.get("Primary")
    configured_outcome = primary.source_column if primary else ""
    outcome_col = demo_outcome if cfg.source_type == "Synthetic Demo" and demo_outcome in columns else configured_outcome
    if outcome_col not in columns:
        outcome_col = default_metric_source(raw, "Primary", configured_outcome)
    metric_type = demo_metric_type if outcome_col == demo_outcome else (primary.metric_type if primary else "Continuous")

    excluded = {unit_col, product_col, date_col, period_col, outcome_col}
    strategy_options = strategy_candidates(raw, unit_col, excluded)
    configured_strategy = st.session_state.strategy_config.historical_column
    strategy_col = demo_strategy if cfg.source_type == "Synthetic Demo" and demo_strategy in columns else configured_strategy
    if strategy_col not in columns:
        strategy_col = strategy_options[0] if strategy_options else ""
    group_col = population.grouping_column if population.grouping_column in columns else next(
        (column for column in ["risk_segment", "fico_band", "revenue_band", "segment"] if column in columns),
        "",
    )

    try:
        preview = build_customer_preview(raw, unit_col, outcome_col, product_col, group_col, strategy_col, date_col, period_col)
    except Exception:
        raw = load_customer_demo("General Customer Test")
        scenario = "General Customer Test"
        unit_col, product_col, group_col = "account_id", "cpc", "risk_segment"
        outcome_col, strategy_col, date_col, period_col = "primary_outcome", "historical_experience", "booking_date", "mob"
        metric_type = "Continuous"
        preview = build_customer_preview(raw, unit_col, outcome_col, product_col, group_col, strategy_col, date_col, period_col)
        using_workspace = False

    accounts = preview["accounts"]
    source_label = "your current dataset" if using_workspace else f"the {scenario} demo"
    outcome_label = humanize_column_name(outcome_col)
    strategy_label = humanize_column_name(strategy_col) if strategy_col else "Not detected"
    if pd.notna(preview["start_date"]) and pd.notna(preview["end_date"]):
        history_label = f"{pd.Timestamp(preview['start_date']):%b %Y} – {pd.Timestamp(preview['end_date']):%b %Y}"
    else:
        history_label = "Date not mapped"

    action_copy, plan_action, analyze_action = st.columns([2.4, 1.15, 1.15])
    action_copy.markdown(
        f'<div class="customer-demo-actions"><div><strong>See the customer story first</strong><span>Previewing {html.escape(source_label)}. Choose what you need next.</span></div></div>',
        unsafe_allow_html=True,
    )
    if plan_action.button("Plan an Experiment", type="primary", use_container_width=True, key="customer_demo_plan"):
        start_customer_workflow("plan")
        st.rerun()
    if analyze_action.button("Analyze Results", use_container_width=True, key="customer_demo_analyze"):
        start_customer_workflow("analyze")
        st.rerun()

    chart_title = "Historical outcome by strategy" if strategy_col else "Historical outcome distribution"
    chart_note = "Historical association only — the randomized experiment provides causal evidence." if strategy_col else "See the outcome distribution before choosing a test design."
    st.markdown(
        f'<div class="customer-section-heading"><strong>{html.escape(chart_title)}</strong><span>{html.escape(chart_note)}</span></div>',
        unsafe_allow_html=True,
    )
    chart_col, summary_col = st.columns([1.75, 0.75])
    with chart_col:
        with st.container(border=True):
            if strategy_col and "Historical Strategy" in accounts:
                st.plotly_chart(
                    customer_strategy_outcome_chart(accounts, "Historical Strategy", "Average Outcome", strategy_label, outcome_label, metric_type, height=380),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.plotly_chart(
                    customer_metric_distribution(accounts, "Average Outcome", outcome_label, metric_type),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
    with summary_col:
        group_counts = preview["group_counts"].head(3)
        mix_rows = "".join(
            f'<div class="customer-mix-row"><span>{html.escape(str(row["Customer Group"]))}</span><div class="customer-mix-track"><div class="customer-mix-fill" style="width:{float(row["Customers / Accounts"]) / max(preview["unique_accounts"], 1) * 100:.1f}%"></div></div><b>{int(row["Customers / Accounts"]):,}</b></div>'
            for _, row in group_counts.iterrows()
        )
        st.markdown(
            f"""
            <section class="customer-preview-summary">
              <h3>Customers in this experiment</h3>
              <div class="customer-preview-grid">
                <div class="customer-preview-stat"><div class="customer-preview-label">Customers / Accounts</div><div class="customer-preview-value">{int(preview['unique_accounts']):,}</div><div class="customer-preview-detail">{int(preview['source_rows']):,} source rows</div></div>
                <div class="customer-preview-stat"><div class="customer-preview-label">Products / Brands</div><div class="customer-preview-value">{int(preview['product_count']):,}</div><div class="customer-preview-detail">Customer starts {html.escape(history_label)}</div></div>
                <div class="customer-preview-stat"><div class="customer-preview-label">Primary outcome</div><div class="customer-preview-value">{html.escape(outcome_label)}</div><div class="customer-preview-detail">{html.escape(metric_type)} metric</div></div>
                <div class="customer-preview-stat"><div class="customer-preview-label">Historical strategy</div><div class="customer-preview-value">{html.escape(strategy_label)}</div><div class="customer-preview-detail">Detected from the data</div></div>
              </div>
              <div class="customer-mix"><div class="customer-mix-title"><span>Customer mix</span><b>{int(preview['group_count'])} groups</b></div>{mix_rows}</div>
              <div class="customer-preview-note">Fields are detected from the dataset, not hardcoded. Uploading new data refreshes the available groups, outcomes, and strategies.</div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    visual_count = int(bool(product_col)) + int(bool(date_col))
    if visual_count:
        st.markdown(
            '<div class="customer-section-heading"><strong>Who is represented in the history?</strong><span>Compare product coverage and when customers entered the available data.</span></div>',
            unsafe_allow_html=True,
        )
        visual_columns = st.columns(visual_count)
        visual_index = 0
        if product_col:
            with visual_columns[visual_index]:
                st.plotly_chart(customer_category_bar(raw, product_col, unit_col, "Product / Brand"), use_container_width=True, config={"displayModeBar": False})
            visual_index += 1
        if date_col:
            with visual_columns[visual_index]:
                st.plotly_chart(customer_history_coverage(raw, date_col, unit_col), use_container_width=True, config={"displayModeBar": False})

    account_preview = accounts.head(10).copy()
    account_preview["Average Outcome"] = account_preview["Average Outcome"].round(2)
    if "Customer Start" in account_preview:
        account_preview["Customer Start"] = pd.to_datetime(account_preview["Customer Start"], errors="coerce").dt.strftime("%b %-d, %Y")
    st.markdown(
        f'<div class="customer-section-heading"><strong>Customer data preview</strong><span>Showing {min(10, len(accounts))} of {len(accounts):,} customer or account summaries.</span></div>',
        unsafe_allow_html=True,
    )
    st.dataframe(account_preview, use_container_width=True, hide_index=True, height=310)


def leave_customer_workspace() -> None:
    st.session_state.customer_intent = None
    st.session_state.main_page = "Overview"


def render_customer_shell(intent: str | None) -> None:
    accent = "#16856b" if intent == "analyze" else "#ef4d4d"
    landing_secondary = "" if intent else '.stButton > button[kind="secondary"] {background:#16856b; border-color:#16856b; color:#fff;} .stButton > button[kind="secondary"] p {color:#fff;} .stButton > button[kind="secondary"]:hover {background:#11735f; border-color:#11735f; color:#fff;}'
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"], header[data-testid="stHeader"] {{display:none !important;}}
        .stApp {{background:#fbfcfe;}}
        .block-container {{max-width:1480px; padding:1rem 2rem 3rem;}}
        .stButton > button[kind="primary"] {{background:{accent}; border-color:{accent}; color:#fff;}}
        .stButton > button[kind="primary"]:hover {{background:{accent}; border-color:{accent}; filter:brightness(.96);}}
        {landing_secondary}
        @media (max-width:800px) {{.block-container {{padding:.75rem 1rem 2rem;}}}}
        </style>
        """,
        unsafe_allow_html=True,
    )
    if intent:
        brand, save_col, exit_col = st.columns([4.2, .8, 1])
    else:
        brand, exit_col = st.columns([5, 1])
    brand.markdown(
        '<div class="customer-product-bar"><div class="customer-product-mark">CUST</div><div><div class="customer-product-name">Experiment Platform</div><div class="customer-product-context">Customer experimentation workspace</div></div></div>',
        unsafe_allow_html=True,
    )
    if intent and save_col.button("Save", use_container_width=True, key="customer_save_experiment", help="Save this configuration and the latest available result to the Command Center."):
        saved = save_current_workspace("Customer")
        st.toast(f'Saved {saved["name"]} · config v{saved["config_version"]}')
    exit_col.button("All experiments", use_container_width=True, key="customer_all_experiments", on_click=leave_customer_workspace)


def render_customer_kpi_panel(title: str, subtitle: str, status: str, values: list[tuple[str, str, str]], analysis: bool = False) -> None:
    cells = "".join(
        f'<div class="customer-kpi"><div class="customer-kpi-label">{html.escape(label)}</div><div class="customer-kpi-value {html.escape(tone)}">{html.escape(value)}</div></div>'
        for label, value, tone in values
    )
    st.markdown(
        f'<section class="customer-kpi-panel{" analysis" if analysis else ""}"><div class="customer-kpi-head"><div><strong>{html.escape(title)}</strong><span>{html.escape(subtitle)}</span></div><div class="customer-status">{html.escape(status)}</div></div><div class="customer-kpis">{cells}</div></section>',
        unsafe_allow_html=True,
    )


def customer_page() -> None:
    intent = st.session_state.get("customer_intent")
    render_customer_shell(intent)
    st.markdown(
        '<div class="customer-page-heading"><h1>Customer Experiment</h1><p>Plan and analyze randomized experiments across customers or accounts.</p></div>',
        unsafe_allow_html=True,
    )
    if not intent:
        customer_landing_preview()
        return
    _, nav_right = st.columns([5, 1])
    if nav_right.button("Switch workflow", key="customer_switch_workflow"):
        st.session_state.customer_intent = None
        st.rerun()
    if intent == "plan":
        tab_data, tab_groups, tab_metrics, tab_strategy, tab_plan = st.tabs(["1 · Bring Data", "2 · Choose Audience", "3 · Pick Outcome", "4 · Define Options", "5 · Get Plan"])
        with tab_data:
            customer_data_step()
        with tab_groups:
            customer_groups_step()
        with tab_metrics:
            customer_metrics_step()
        with tab_strategy:
            strategy_step()
        with tab_plan:
            customer_plan_step()
        return
    tab_analysis, tab_decision = st.tabs(["1 · Upload Results", "2 · Recommendation"])
    with tab_analysis:
        analysis_step()
    with tab_decision:
        decision_step()


init_state()
with st.sidebar:
    st.title("Experiment Platform")
    page = st.radio("Overview", ["Overview", "Customer", "Geography", "Time Series"], label_visibility="collapsed", key="main_page")

if page == "Overview":
    show_overview()
elif page == "Customer":
    customer_page()
elif page == "Geography":
    geographic_page()
else:
    time_series_page()
