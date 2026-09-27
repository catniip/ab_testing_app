from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .data_validation import first_series, safe_datetime_series, safe_numeric_series
from .historical_strategy import fixed_unit_values, validate_fixed_unit_value
from .models import DataMappingConfig, MetricConfig, PopulationConfig


@dataclass
class ProcessingResult:
    analysis_df: pd.DataFrame
    diagnostics: dict
    metric_preview: pd.DataFrame
    cpc_breakdown: pd.DataFrame


def build_historical_metric_dataset(
    raw_df: pd.DataFrame,
    selected_cpcs: list[str],
    unit_id_column: str,
    cpc_column: str,
    mob_column: str,
    booking_date_column: str,
    metric_configs: dict[str, MetricConfig],
    cutoff_date: str = "",
    strategy_column: str | None = None,
) -> ProcessingResult:
    mapping = DataMappingConfig(
        unit_id_column=unit_id_column,
        cpc_column=cpc_column,
        mob_column=mob_column,
        booking_date_column=booking_date_column,
        cutoff_date=cutoff_date,
    )
    population = PopulationConfig(selected_cpcs=selected_cpcs)
    return build_analysis_dataset(raw_df, population, metric_configs, mapping, strategy_column)


def processed_metric_name(metric: MetricConfig) -> str:
    prefix = "avg" if metric.aggregation_method == "Average" else "cum"
    base = re.sub(r"[^a-z0-9]+", "_", metric.source_column.lower()).strip("_")
    start = max(int(getattr(metric, "mob_start", 1)), 1)
    suffix = f"mob{metric.mob_horizon}" if start == 1 else f"mob{start}_{metric.mob_horizon}"
    return f"{prefix}_{base}_{suffix}"


def validate_mapping(raw_df: pd.DataFrame, mapping: DataMappingConfig) -> list[str]:
    errors = []
    required = [mapping.unit_id_column]
    if mapping.data_structure == "Longitudinal (unit x period)":
        required.append(mapping.mob_column)
    for column in required:
        if column not in raw_df.columns:
            errors.append(f"{column} is required but is not present in the raw dataset.")
    if mapping.data_structure == "Longitudinal (unit x period)" and mapping.mob_column in raw_df.columns:
        mob = safe_numeric_series(raw_df, mapping.mob_column)
        if mob.isna().all():
            errors.append("MOB Column must contain numeric month-on-book values.")
    if mapping.data_structure == "Longitudinal (unit x period)" and mapping.booking_date_column and mapping.booking_date_column not in raw_df.columns:
        errors.append("Booking Date was selected but is not present in the raw dataset.")
    return errors


def common_horizon(metrics: dict[str, MetricConfig]) -> int:
    return max((int(metric.mob_horizon) for metric in metrics.values()), default=0)


def filter_cpcs(raw_df: pd.DataFrame, mapping: DataMappingConfig, selected_cpcs: list[str]) -> pd.DataFrame:
    if not mapping.cpc_column or mapping.cpc_column not in raw_df.columns or not selected_cpcs:
        return raw_df.copy()
    return raw_df[raw_df[mapping.cpc_column].astype(str).isin([str(cpc) for cpc in selected_cpcs])].copy()


def determine_mature_units(df: pd.DataFrame, mapping: DataMappingConfig, required_horizon: int) -> pd.Index:
    if mapping.booking_date_column and mapping.booking_date_column in df.columns and mapping.cutoff_date:
        dates = pd.DataFrame(
            {
                mapping.unit_id_column: first_series(df, mapping.unit_id_column),
                mapping.booking_date_column: safe_datetime_series(df, mapping.booking_date_column),
            }
        ).dropna().drop_duplicates(mapping.unit_id_column)
        booking = dates[mapping.booking_date_column]
        cutoff = pd.Timestamp(mapping.cutoff_date)
        maturity_months = (cutoff.year - booking.dt.year) * 12 + (cutoff.month - booking.dt.month)
        return pd.Index(dates.loc[maturity_months >= required_horizon, mapping.unit_id_column])
    mob = safe_numeric_series(df, mapping.mob_column)
    max_mob = df.assign(_mob=mob).groupby(mapping.unit_id_column)["_mob"].max()
    return max_mob[max_mob >= required_horizon].index


def completeness_by_unit(
    df: pd.DataFrame,
    mapping: DataMappingConfig,
    unit_ids: pd.Index,
    required_horizon: int,
    start_mob: int = 1,
) -> pd.DataFrame:
    scoped = df[df[mapping.unit_id_column].isin(unit_ids)].copy()
    scoped["_mob"] = safe_numeric_series(scoped, mapping.mob_column)
    start_mob = max(int(start_mob), 1)
    scoped = scoped[(scoped["_mob"] >= start_mob) & (scoped["_mob"] <= required_horizon)]
    observed = scoped.drop_duplicates([mapping.unit_id_column, "_mob"]).groupby(mapping.unit_id_column)["_mob"].nunique()
    out = pd.DataFrame({mapping.unit_id_column: unit_ids})
    out["Observed MOBs"] = out[mapping.unit_id_column].map(observed).fillna(0).astype(int)
    out["Expected MOBs"] = max(required_horizon - start_mob + 1, 0)
    out["Complete"] = out["Observed MOBs"] >= out["Expected MOBs"]
    return out


def aggregate_metric_by_mob(
    df: pd.DataFrame,
    unit_column: str,
    mob_column: str,
    metric_column: str,
    aggregation_method: str,
    mob_horizon: int,
    mob_start: int = 1,
) -> pd.Series:
    scoped = pd.DataFrame(
        {
            unit_column: first_series(df, unit_column),
            mob_column: first_series(df, mob_column),
            metric_column: first_series(df, metric_column),
        }
    )
    scoped["_mob"] = safe_numeric_series(scoped, mob_column)
    scoped["_metric"] = safe_numeric_series(scoped, metric_column)
    mob_start = max(int(mob_start), 1)
    scoped = scoped[(scoped["_mob"] >= mob_start) & (scoped["_mob"] <= mob_horizon)].dropna(subset=[unit_column, "_mob"])
    monthly = scoped.groupby([unit_column, "_mob"], as_index=False)["_metric"].mean()
    if aggregation_method == "Cumulative":
        return monthly.groupby(unit_column)["_metric"].sum()
    return monthly.groupby(unit_column)["_metric"].mean()


def duplicate_unit_mob_count(raw_df: pd.DataFrame, mapping: DataMappingConfig) -> int:
    if mapping.unit_id_column not in raw_df.columns or mapping.mob_column not in raw_df.columns:
        return 0
    return int(raw_df.duplicated([mapping.unit_id_column, mapping.mob_column]).sum())


def build_analysis_dataset(
    raw_df: pd.DataFrame,
    population: PopulationConfig,
    metrics: dict[str, MetricConfig],
    mapping: DataMappingConfig,
    strategy_column: str | None = None,
) -> ProcessingResult:
    errors = validate_mapping(raw_df, mapping)
    if errors:
        raise ValueError("; ".join(errors))
    selected = filter_cpcs(raw_df, mapping, population.selected_cpcs)
    for metric in metrics.values():
        if metric.source_column not in selected.columns:
            raise ValueError(f"{metric.name}: source column {metric.source_column} is not present in the selected dataset.")
    strategy_errors = validate_fixed_unit_value(selected, mapping.unit_id_column, strategy_column) if strategy_column in selected.columns and not selected.empty else []
    grouping_column = getattr(population, "grouping_column", "")
    grouping_errors = validate_fixed_unit_value(selected, mapping.unit_id_column, grouping_column) if grouping_column in selected.columns and not selected.empty else []
    if strategy_errors or grouping_errors:
        raise ValueError("; ".join(strategy_errors + grouping_errors))

    if mapping.data_structure == "Cross-sectional (one row per unit)":
        unit_info_cols = [mapping.unit_id_column]
        if mapping.cpc_column and mapping.cpc_column in selected.columns:
            unit_info_cols.append(mapping.cpc_column)
        if strategy_column and strategy_column in selected.columns:
            unit_info_cols.append(strategy_column)
        if grouping_column and grouping_column in selected.columns:
            unit_info_cols.append(grouping_column)
        unit_info = fixed_unit_values(selected, mapping.unit_id_column, unit_info_cols[1:])
        analysis = unit_info.copy()
        metric_preview_rows = []
        for metric in metrics.values():
            metric.processed_column = processed_metric_name(metric)
            metric.column = metric.processed_column
            values = selected.groupby(mapping.unit_id_column)[metric.source_column].mean()
            analysis[metric.processed_column] = values
            clean = pd.to_numeric(values, errors="coerce").dropna()
            baseline = float(clean.mean()) if not clean.empty else float("nan")
            std = float((baseline * (1 - baseline)) ** 0.5) if metric.metric_type == "Binary" else float(clean.std(ddof=1))
            if metric.metric_type == "Binary":
                metric.baseline_rate = baseline
            else:
                metric.baseline_mean = baseline
                metric.standard_deviation = std
                metric.variance = std**2
            metric_preview_rows.append({"Metric": metric.name, "Role": metric.role, "Aggregation": "One value per unit", "Observation Window": "As observed", "Baseline": baseline, "SD": std, "Eligible N": int(clean.size)})
        if mapping.cpc_column and mapping.cpc_column in analysis.columns:
            cpc_counts = analysis.reset_index().groupby(mapping.cpc_column)[mapping.unit_id_column].nunique().reset_index(name="Eligible Units")
            cpc_counts["Share"] = cpc_counts["Eligible Units"] / cpc_counts["Eligible Units"].sum()
        else:
            cpc_counts = pd.DataFrame()
        diagnostics = {
            "Raw Records": len(raw_df),
            "Unique Accounts": raw_df[mapping.unit_id_column].nunique(),
            "Selected CPC Accounts": selected[mapping.unit_id_column].nunique(),
            "Longest Required Horizon": 0,
            "Mature Accounts": selected[mapping.unit_id_column].nunique(),
            "Complete Eligible Accounts": len(analysis),
            "Final Analysis Population": len(analysis),
            "Excluded for Insufficient Maturity": 0,
            "Excluded for Data Completeness": 0,
            "Duplicate Unit-MOB Rows": int(selected.duplicated(mapping.unit_id_column).sum()),
            "Units Missing Historical Strategy": int(unit_info[strategy_column].isna().sum()) if strategy_column and strategy_column in unit_info.columns else 0,
        }
        return ProcessingResult(analysis.reset_index(), diagnostics, pd.DataFrame(metric_preview_rows), cpc_counts)

    required_horizon = common_horizon(metrics)
    population.common_mob_horizon = required_horizon
    mature_units = determine_mature_units(selected, mapping, required_horizon)
    completeness = completeness_by_unit(selected, mapping, mature_units, required_horizon)
    complete_units = pd.Index(completeness.loc[completeness["Complete"], mapping.unit_id_column])
    unit_info_cols = [mapping.unit_id_column]
    if mapping.cpc_column and mapping.cpc_column in selected.columns:
        unit_info_cols.append(mapping.cpc_column)
    if strategy_column and strategy_column in selected.columns:
        unit_info_cols.append(strategy_column)
    if grouping_column and grouping_column in selected.columns:
        unit_info_cols.append(grouping_column)
    unit_info = fixed_unit_values(selected, mapping.unit_id_column, unit_info_cols[1:])
    metric_preview_rows = []
    metric_values: dict[str, pd.Series] = {}
    eligible_units: set = set()
    primary_eligible_n = 0
    for metric in metrics.values():
        metric.processed_column = processed_metric_name(metric)
        metric.column = metric.processed_column
        metric_mature_units = determine_mature_units(selected, mapping, int(metric.mob_horizon))
        metric_start = max(int(getattr(metric, "mob_start", 1)), 1)
        metric_completeness = completeness_by_unit(selected, mapping, metric_mature_units, int(metric.mob_horizon), metric_start)
        metric_complete_units = pd.Index(metric_completeness.loc[metric_completeness["Complete"], mapping.unit_id_column])
        aggregated = aggregate_metric_by_mob(
            selected[selected[mapping.unit_id_column].isin(metric_complete_units)],
            mapping.unit_id_column,
            mapping.mob_column,
            metric.source_column,
            metric.aggregation_method,
            metric.mob_horizon,
            metric_start,
        )
        metric_values[metric.processed_column] = aggregated
        eligible_units.update(aggregated.index.tolist())
        clean = pd.to_numeric(aggregated, errors="coerce").dropna()
        baseline = float(clean.mean()) if not clean.empty else float("nan")
        std = float((baseline * (1 - baseline)) ** 0.5) if metric.metric_type == "Binary" else float(clean.std(ddof=1))
        if metric.metric_type == "Binary":
            metric.baseline_rate = baseline
        else:
            metric.baseline_mean = baseline
            metric.standard_deviation = std
            metric.variance = std**2
        if metric.role == "Primary":
            primary_eligible_n = int(clean.size)
        metric_preview_rows.append(
            {
                "Metric": metric.name,
                "Role": metric.role,
                "Aggregation": metric.aggregation_method,
                "Observation Window": (
                    f"Through MOB {metric.mob_horizon}"
                    if metric_start == 1
                    else f"MOB {metric_start} to {metric.mob_horizon}"
                ),
                "Baseline": baseline,
                "SD": std,
                "Eligible N": int(clean.size),
            }
        )
    analysis_index = unit_info.index.intersection(pd.Index(list(eligible_units)))
    analysis = unit_info.loc[analysis_index].copy()
    analysis.index.name = mapping.unit_id_column
    for column, values in metric_values.items():
        analysis[column] = values.reindex(analysis.index)
    if mapping.cpc_column and mapping.cpc_column in analysis.columns:
        cpc_counts = analysis.reset_index().groupby(mapping.cpc_column)[mapping.unit_id_column].nunique().reset_index(name="Eligible Units")
        if not cpc_counts.empty:
            cpc_counts["Share"] = cpc_counts["Eligible Units"] / cpc_counts["Eligible Units"].sum()
    else:
        cpc_counts = pd.DataFrame()
    diagnostics = {
        "Raw Records": len(raw_df),
        "Unique Accounts": raw_df[mapping.unit_id_column].nunique(),
        "Selected CPC Accounts": selected[mapping.unit_id_column].nunique(),
        "Longest Required Horizon": required_horizon,
        "Mature Accounts": len(mature_units),
        "Complete Eligible Accounts": len(complete_units),
        "Final Analysis Population": primary_eligible_n,
        "Any Metric Eligible Units": len(analysis),
        "Excluded for Insufficient Maturity": selected[mapping.unit_id_column].nunique() - len(mature_units),
        "Excluded for Data Completeness": len(mature_units) - len(complete_units),
        "Duplicate Unit-MOB Rows": duplicate_unit_mob_count(selected, mapping),
        "Units Missing Historical Strategy": int(unit_info[strategy_column].isna().sum()) if strategy_column and strategy_column in unit_info.columns else 0,
    }
    return ProcessingResult(analysis.reset_index(), diagnostics, pd.DataFrame(metric_preview_rows), cpc_counts)
