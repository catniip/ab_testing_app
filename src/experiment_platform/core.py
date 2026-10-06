from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import pandas as pd
import warnings
import numpy as np


# --- Models ---

@dataclass
class DataConfig:
    source_type: str = "Synthetic Demo"
    dataset_name: str = "Synthetic Raw Portfolio Data"
    catalog: str = ""
    schema: str = ""
    table: str = ""


@dataclass
class DataMappingConfig:
    data_structure: str = "Longitudinal (unit x period)"
    unit_id_column: str = "account_id"
    cpc_column: str = "cpc"
    mob_column: str = "mob"
    booking_date_column: str = "booking_date"
    cutoff_date: str = ""


@dataclass
class PopulationConfig:
    selected_cpcs: list[str] = field(default_factory=list)
    common_mob_horizon: int = 36
    completeness_policy: str = "Exclude incomplete metric histories"
    grouping_column: str = ""
    grouping_definitions: list[dict[str, Any]] = field(default_factory=list)
    group_settings: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    experiment_template: str = "Custom"
    arm_format: str = "Named variants"
    strategy_type: str = "Categorical Strategy"
    strategy_name: str = "Treatment"
    strategy_goal: str = ""
    value_label: str = "Strategy Value"
    value_format: str = "Number"
    assignment_column: str = "assigned_treatment"
    historical_column: str = ""
    historical_evidence: str = "Use group-level outcome history"
    historical_grouping: str = "Nearest proposed point"
    max_mapping_distance: float = 0.0
    selection_mode: str = "Suggested lines"
    curve_binning_method: str = "Automatic fine bins"
    curve_bin_width: float = 500.0
    curve_bin_count: int = 20
    curve_smoothing: bool = True
    planning_grouping: str = "Closest testing line"
    planning_bin_width: float = 2000.0
    control_name: str = "Control"
    control_value: Any = 0
    min_value: float = 0
    max_value: float = 100
    increment: float = 1
    number_of_arms: int = 2
    treatment_values: list[Any] = field(default_factory=lambda: [1])
    treatment_names: list[str] = field(default_factory=lambda: ["Treatment A"])
    categorical_control: str = "Control"
    categorical_treatments: list[str] = field(default_factory=lambda: ["Treatment A"])
    arm_descriptions: dict[str, str] = field(default_factory=dict)
    group_arms: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class MetricConfig:
    name: str
    column: str
    role: str
    metric_type: str = "Continuous"
    direction: str = "Higher is Better"
    effect_type: str = "Relative %"
    effect_value: float = 0.05
    expected_effect: float | None = None
    guardrail_threshold: float | None = None
    source_column: str = ""
    processed_column: str = ""
    aggregation_method: str = "Average"
    mob_start: int = 1
    mob_horizon: int = 12
    observation_unit: str = "Months"
    assumption_source: str = "Enter Manually"
    variability_input: str = "Standard Deviation"
    baseline_mean: float = 2300.0
    standard_deviation: float = 900.0
    variance: float = 810000.0
    baseline_rate: float = 0.032


@dataclass
class DesignConfig:
    alpha: float = 0.05
    target_power: float = 0.8
    allocation: str = "Traffic Allocation"
    sample_size_basis: str = "Power All Configured Metrics"
    multiplicity_method: str = "Holm"
    audience_decision_scope: str = "Separate decision per audience"
    attrition_rate: float = 0.0
    planned_launch_date: str = ""
    outcome_delay_value: int = 0
    outcome_delay_unit: str = "Days"
    eligible_customers: int = 300
    traffic_frequency: str = "Monthly"
    max_enrollment_periods: int = 0
    required_n_per_arm: int = 0
    total_sample_size: int = 0
    binding_metric: str = ""
    arm_sample_sizes: dict[str, int] = field(default_factory=dict)
    group_plan_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AnalysisConfig:
    unit_id_column: str = ""
    assignment_column: str = ""
    multiplicity_method: str = "Holm"
    control_arm: Any = None
    treatment_arms: list[Any] = field(default_factory=list)


# --- Formatting ---

def money(value: float) -> str:
    return f"${value:,.0f}"


def number(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:,.{digits}f}%"


def p_value(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


# --- Data Validation ---

def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    seen: dict[str, int] = {}
    names = []
    for column in df.columns:
        name = str(column).strip()
        count = seen.get(name, 0)
        names.append(name if count == 0 else f"{name}_{count + 1}")
        seen[name] = count + 1
    df.columns = names
    return df


def first_series(df: pd.DataFrame, column: str | None) -> pd.Series | None:
    if column is None or column not in df.columns:
        return None
    selected = df[column]
    if isinstance(selected, pd.DataFrame):
        return selected.iloc[:, 0]
    return selected


def safe_numeric_series(df: pd.DataFrame, column: str | None) -> pd.Series:
    series = first_series(df, column)
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def safe_datetime_series(df: pd.DataFrame, column: str | None) -> pd.Series:
    series = first_series(df, column)
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(series, errors="coerce")


def date_like_columns(df: pd.DataFrame, minimum_valid_share: float = 0.8) -> list[str]:
    columns = []
    for column in df.columns:
        series = first_series(df, column)
        if series is None or pd.api.types.is_numeric_dtype(series):
            continue
        non_null = series.dropna()
        if not non_null.empty and safe_datetime_series(df, column).dropna().size / len(non_null) >= minimum_valid_share:
            columns.append(column)
    return columns


def normalize_uploaded_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return deduplicate_columns(df)


def infer_column(columns: list[str], candidates: list[str], fallback: str = "") -> str:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return fallback or (columns[0] if columns else "")


def infer_mob_column(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    named = infer_column(columns, ["mob", "months_on_book", "month_on_book"])
    if named:
        return named
    best = ""
    best_valid = -1
    for column in columns:
        values = safe_numeric_series(df, column)
        valid = values.dropna()
        if valid.empty:
            continue
        integer_like = ((valid % 1).abs() < 1e-9).mean()
        in_range = valid.between(0, 120).mean()
        score = int(len(valid) * integer_like * in_range)
        if score > best_valid:
            best = column
            best_valid = score
    return best


def infer_schema(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = first_series(df, column)
        numeric = safe_numeric_series(df, column)
        parsed_numeric = int(numeric.notna().sum())
        parsed_date = int(safe_datetime_series(df, column).notna().sum())
        non_null = int(series.notna().sum()) if series is not None else 0
        if non_null and parsed_numeric / non_null > 0.9:
            suggested = "Numeric"
        elif non_null and parsed_date / non_null > 0.8:
            suggested = "Date"
        else:
            suggested = "Categorical"
        unique = int(series.nunique(dropna=True)) if series is not None else 0
        rows.append(
            {
                "Column": column,
                "Detected Type": str(series.dtype) if series is not None else "unknown",
                "Non-Null Rows": non_null,
                "Unique Values": unique,
                "Suggested Type": suggested,
            }
        )
    return pd.DataFrame(rows)


def data_quality_warnings(df: pd.DataFrame, unit_col: str, cpc_col: str, mob_col: str) -> list[str]:
    warnings = []
    if unit_col not in df.columns:
        warnings.append("Experimental Unit ID mapping is required.")
    if cpc_col and cpc_col not in df.columns:
        warnings.append("The selected population segment column is not present.")
    mob = safe_numeric_series(df, mob_col)
    if mob.empty or mob.notna().sum() == 0:
        warnings.append("We could not interpret the selected MOB column as numeric. Review the column mapping.")
    elif mob.isna().sum() > 0:
        warnings.append(f"{mob.isna().sum():,.0f} rows have missing or invalid MOB values.")
    return warnings


# --- Metrics ---

def column_schema(df: pd.DataFrame) -> pd.DataFrame:
    schema = infer_schema(df)
    schema["Suggested Metric Type"] = [suggest_metric_type(df, col) for col in schema["Column"]]
    return schema


def suggest_metric_type(df: pd.DataFrame, column: str) -> str:
    series = first_series(df, column)
    if series is None:
        return "Not Numeric"
    values = series.dropna()
    unique = set(values.unique().tolist())
    if len(unique) == 2 and unique.issubset({0, 1, 0.0, 1.0, False, True}):
        return "Binary"
    if pd.api.types.is_numeric_dtype(values) or safe_numeric_series(df, column).notna().sum() / max(len(values), 1) > 0.9:
        return "Continuous"
    return "Not Numeric"


def validate_metric(df: pd.DataFrame, column: str, metric_type: str) -> list[str]:
    if column not in df.columns:
        return [f"{column} is not present in the selected dataset."]
    values = first_series(df, column)
    if values is None:
        return [f"{column} is not present in the selected dataset."]
    values = values.dropna()
    if metric_type == "Continuous" and safe_numeric_series(df, column).notna().sum() == 0:
        return [f"{column} must be numeric for a continuous metric."]
    if metric_type == "Binary":
        unique = set(values.unique().tolist())
        if len(unique) != 2:
            return [f"{column} should contain exactly two valid states for a binary metric."]
    return []


def metric_baseline(df: pd.DataFrame, column: str, metric_type: str) -> dict[str, float]:
    values = safe_numeric_series(df, column).dropna()
    if metric_type == "Binary":
        return {"baseline": float(values.mean()), "std_dev": float((values.mean() * (1 - values.mean())) ** 0.5)}
    return {"baseline": float(values.mean()), "std_dev": float(values.std(ddof=1))}


# --- Validation ---

def validate_constraints(min_line: int, max_line: int, increment: int) -> list[str]:
    errors = []
    if min_line >= max_line:
        errors.append("Minimum Credit Line must be lower than Maximum Credit Line.")
    if increment <= 0:
        errors.append("Allowed Credit Line Increment must be greater than zero.")
    return errors


def validate_test_lines(lines: list[int], min_line: int, max_line: int, increment: int) -> list[str]:
    errors = []
    if not lines:
        return ["At least one test line is required."]
    if len(lines) != len(set(lines)):
        errors.append("Duplicate credit lines are not allowed.")
    for line in lines:
        if line < min_line or line > max_line:
            errors.append(f"{line:,.0f} is outside the allowed credit-line range.")
        if (line - min_line) % increment != 0:
            errors.append(f"{line:,.0f} does not follow the allowed increment.")
    return errors


# --- Demo Data ---

def historical_portfolio(seed: int = 42, rows: int = 6000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    risk_segment = rng.choice(["Prime", "Near Prime", "Subprime"], size=rows, p=[0.48, 0.36, 0.16])
    fico_base = np.select(
        [risk_segment == "Prime", risk_segment == "Near Prime", risk_segment == "Subprime"],
        [735, 675, 610],
    )
    fico = np.clip(rng.normal(fico_base, 28), 520, 820).round().astype(int)
    current_line_raw = 1800 + (fico - 560) * 28 + rng.normal(0, 900, rows)
    current_credit_line = np.clip(np.round(current_line_raw / 500) * 500, 1000, 12000).astype(int)
    utilization = np.clip(rng.beta(2.4, 5.0, rows) + (680 - fico) / 1200, 0.02, 0.97)
    revolving_balance = np.maximum(0, current_credit_line * utilization + rng.normal(0, 450, rows))
    risk_adjusted_revenue = revolving_balance * rng.normal(0.085, 0.018, rows) - np.maximum(0, 690 - fico) * 1.8
    default_logit = -5.5 + utilization * 2.8 + (660 - fico) / 85
    default_prob = 1 / (1 + np.exp(-default_logit))
    default_flag = rng.binomial(1, np.clip(default_prob, 0.005, 0.25))
    return pd.DataFrame(
        {
            "customer_id": [f"C{idx:06d}" for idx in range(1, rows + 1)],
            "current_credit_line": current_credit_line,
            "revolving_balance": revolving_balance.round(2),
            "risk_adjusted_revenue": risk_adjusted_revenue.round(2),
            "utilization": utilization.round(4),
            "fico": fico,
            "risk_segment": risk_segment,
            "default_flag": default_flag,
        }
    )


def raw_longitudinal_portfolio(seed: int = 84, accounts: int = 1800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cutoff = pd.Timestamp("2026-06-30")
    records = []
    cpcs = rng.choice(["CPC_A", "CPC_B", "CPC_C"], size=accounts, p=[0.46, 0.36, 0.18])
    risk_segments = rng.choice(["Prime", "Near Prime", "Subprime"], size=accounts, p=[0.47, 0.37, 0.16])
    revenue_bands = rng.choice(["Low", "Medium", "High"], size=accounts, p=[0.30, 0.48, 0.22])
    booking_offsets = rng.integers(3, 54, size=accounts)
    for idx in range(accounts):
        account_id = f"A{idx + 1:06d}"
        booking_date = cutoff - pd.DateOffset(months=int(booking_offsets[idx]))
        max_mob = int(((cutoff.year - booking_date.year) * 12) + cutoff.month - booking_date.month)
        credit_line = int(np.clip(np.round(rng.normal(5600, 1900) / 500) * 500, 1000, 12000))
        base_balance = max(150, credit_line * rng.uniform(0.18, 0.55))
        cpc = cpcs[idx]
        risk_segment = risk_segments[idx]
        fico_center = {"Prime": 750, "Near Prime": 685, "Subprime": 620}[risk_segment]
        fico = int(np.clip(rng.normal(fico_center, 24), 520, 820))
        fico_band = "Below 660" if fico < 660 else "660 to 719" if fico < 720 else "720+"
        incomplete = rng.random() < 0.08 and max_mob >= 18
        missing_mobs = set(rng.choice(np.arange(1, min(max_mob, 36) + 1), size=min(2, max_mob), replace=False).tolist()) if incomplete else set()
        for mob in range(1, min(max_mob, 40) + 1):
            if mob in missing_mobs:
                continue
            balance = max(0, base_balance * (1 + 0.008 * mob) + rng.normal(0, 250))
            revenue = balance * rng.normal(0.018, 0.004)
            loss = max(0, rng.normal(8 + mob * 0.4 + max(0, 6500 - credit_line) / 1600, 8))
            default_prob = np.clip(0.006 + mob * 0.0009 + (7000 - credit_line) / 500000, 0.001, 0.08)
            records.append(
                {
                    "customer_id": f"C{idx + 1:06d}",
                    "account_id": account_id,
                    "cpc": cpc,
                    "risk_segment": risk_segment,
                    "fico": fico,
                    "fico_band": fico_band,
                    "revenue_band": revenue_bands[idx],
                    "booking_date": booking_date.date().isoformat(),
                    "mob": mob,
                    "current_credit_line": credit_line,
                    "revolving_balance": round(balance, 2),
                    "revenue": round(revenue, 2),
                    "loss": round(loss, 2),
                    "default_flag": int(rng.binomial(1, default_prob)),
                    "utilization": round(min(balance / credit_line, 1.4), 4),
                }
            )
    return pd.DataFrame(records)


def experiment_results(seed: int = 202, rows_per_arm: int = 850) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lines = np.array([3000, 5000, 8000])
    records = []
    for line in lines:
        lift = {3000: -80, 5000: 0, 8000: 220}[int(line)]
        for idx in range(rows_per_arm):
            fico = int(np.clip(rng.normal(690 + (line - 5000) / 350, 45), 540, 820))
            risk_segment = "Subprime" if fico < 660 else "Near Prime" if fico < 720 else "Prime"
            fico_band = "Below 660" if fico < 660 else "660 to 719" if fico < 720 else "720+"
            utilization = np.clip(rng.beta(2.2, 5.2) + (line - 5000) / 26000, 0.01, 0.98)
            balance = max(0, 1320 + lift + (line - 5000) * 0.055 + rng.normal(0, 1125))
            default_prob = np.clip(0.035 + utilization * 0.035 + (660 - fico) / 6000, 0.003, 0.18)
            default_flag = int(rng.binomial(1, default_prob))
            loss = max(0, default_flag * 700 + rng.normal(20, 10))
            records.append(
                {
                    "customer_id": f"E{line}_{idx:05d}",
                    "assigned_credit_line": line,
                    "revolving_balance": round(balance, 2),
                    "risk_adjusted_revenue": round(balance * 0.082 - default_prob * 700, 2),
                    "loss": round(loss, 2),
                    "utilization": round(utilization, 4),
                    "fico": fico,
                    "fico_band": fico_band,
                    "risk_segment": risk_segment,
                    "revenue_band": "Low" if balance < 900 else "Medium" if balance < 1900 else "High",
                    "default_flag": default_flag,
                }
            )
    return pd.DataFrame(records)


def time_series_campaign(seed: int = 77) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", "2025-12-28", freq="W-SUN")
    intervention = pd.Timestamp("2025-07-06")
    t = np.arange(len(dates))
    seasonal = 420 * np.sin(2 * np.pi * t / 52)
    trend = 18 * t
    website_traffic = 62000 + 210 * t + 2600 * np.sin(2 * np.pi * t / 52 + 0.4) + rng.normal(0, 1200, len(t))
    marketing_spend = np.where(dates >= intervention, 9000, 4200) + rng.normal(0, 600, len(t))
    campaign_lift = np.where(dates >= intervention, 1450 + 6 * np.maximum(t - np.where(dates >= intervention)[0][0], 0), 0)
    applications = 9200 + trend + seasonal + 0.035 * website_traffic + campaign_lift + rng.normal(0, 520, len(t))
    return pd.DataFrame(
        {
            "date": dates,
            "applications": applications.round().astype(int),
            "website_traffic": website_traffic.round().astype(int),
            "marketing_spend": marketing_spend.round(2),
            "campaign_flag": (dates >= intervention).astype(int),
            "campaign_name": np.where(dates >= intervention, "National Summer Campaign", ""),
            "segment": "All Portfolio",
        }
    )


def time_series_planning(seed: int = 77) -> pd.DataFrame:
    """Historical-only series for pre-launch duration planning."""
    data = time_series_campaign(seed).copy()
    data = data.drop(columns=["campaign_flag", "campaign_name"])
    data["applications"] = data["applications"] - np.where(data["date"] >= pd.Timestamp("2025-07-06"), 1450 + 6 * np.maximum(np.arange(len(data)) - int((data["date"] >= pd.Timestamp("2025-07-06")).argmax()), 0), 0)
    return data


def geographic_panel(seed: int = 91) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", "2026-12-27", freq="W-SUN")
    campaign_start = pd.Timestamp("2026-09-06")
    markets = [
        ("New York", 19500000, 1.24),
        ("Los Angeles", 13200000, 1.17),
        ("Chicago", 9500000, 0.96),
        ("Dallas-Ft. Worth", 8100000, 0.94),
        ("Houston", 7200000, 0.87),
        ("Atlanta", 6100000, 0.83),
        ("Washington DC", 6300000, 0.82),
        ("Philadelphia", 6200000, 0.80),
        ("Phoenix", 5000000, 0.70),
        ("Seattle", 4100000, 0.68),
        ("Boston", 4900000, 0.72),
        ("Miami-Ft. Lauderdale", 6700000, 0.78),
    ]
    treated = {"New York", "Chicago", "Houston", "Washington DC", "Phoenix"}
    records = []
    for market_idx, (dma, population, scale) in enumerate(markets):
        base = 7200 * scale + rng.normal(0, 180)
        trend = rng.normal(9.5, 1.3) * np.arange(len(dates))
        seasonal = 380 * scale * np.sin(2 * np.pi * np.arange(len(dates)) / 52 + market_idx / 5)
        customers = population / 1000 * rng.normal(1.0, 0.015, len(dates))
        spend = np.where(dates >= campaign_start, rng.normal(14500, 1200, len(dates)), rng.normal(1200, 350, len(dates)))
        lift = np.where((dates >= campaign_start) & (dma in treated), 0.075 * (base + trend), 0)
        outcome = base + trend + seasonal + 0.0009 * customers + lift + rng.normal(0, 260 * scale, len(dates))
        revenue = outcome * rng.normal(58, 4, len(dates))
        for idx, date in enumerate(dates):
            records.append(
                {
                    "week": date,
                    "dma": dma,
                    "applications": round(float(outcome[idx]), 2),
                    "spend": round(float(spend[idx]), 2),
                    "population": population,
                    "existing_customers": round(float(customers[idx]), 0),
                    "historical_revenue": round(float(revenue[idx]), 2),
                    "treatment_group": "Test" if dma in treated else "Control",
                }
            )
    return pd.DataFrame(records)


# --- Data Access ---

def load_uploaded_csv(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    return normalize_uploaded_dataset(pd.read_csv(uploaded_file))


def load_historical_demo() -> pd.DataFrame:
    return historical_portfolio()


def load_raw_demo() -> pd.DataFrame:
    return raw_longitudinal_portfolio()


CUSTOMER_DEMO_SCENARIOS = [
    "General Customer Test",
    "Offer & Engagement",
    "Retention Strategy",
    "Credit Strategy",
]


def load_customer_demo(scenario: str = "General Customer Test") -> pd.DataFrame:
    """Return reproducible customer-history demos with different strategy fields."""
    data = raw_longitudinal_portfolio().copy()
    account_codes = pd.factorize(data["account_id"], sort=True)[0]
    row_codes = np.arange(len(data))
    rng = np.random.default_rng(20260927)

    if scenario == "Offer & Engagement":
        offers = np.array(["Standard", "Cash Back", "Low APR", "Rewards"])
        data["historical_offer"] = offers[account_codes % len(offers)]
        offer_lift = pd.Series(data["historical_offer"]).map({"Standard": 0, "Cash Back": 7, "Low APR": 4, "Rewards": 9}).to_numpy()
        data["engagement_score"] = np.clip(data["utilization"].to_numpy() * 70 + offer_lift + rng.normal(0, 6, len(data)), 0, 100).round(2)
        conversion_probability = np.clip(0.04 + data["engagement_score"].to_numpy() / 500, 0.02, 0.35)
        data["conversion_flag"] = (rng.random(len(data)) < conversion_probability).astype(int)
        data["incentive_cost"] = pd.Series(data["historical_offer"]).map({"Standard": 0, "Cash Back": 40, "Low APR": 25, "Rewards": 30}).to_numpy()
    elif scenario == "Retention Strategy":
        actions = np.array(["Standard Service", "Fee Waiver", "Bonus Points", "Specialist Call"])
        data["historical_retention_action"] = actions[account_codes % len(actions)]
        risk = pd.Series(data["risk_segment"]).map({"Prime": 0.08, "Near Prime": 0.16, "Subprime": 0.28}).fillna(0.14).to_numpy()
        action_reduction = pd.Series(data["historical_retention_action"]).map({"Standard Service": 0, "Fee Waiver": 0.025, "Bonus Points": 0.02, "Specialist Call": 0.04}).to_numpy()
        data["retained_flag"] = (rng.random(len(data)) > np.clip(risk - action_reduction, 0.01, 0.8)).astype(int)
        data["customer_value"] = (data["revenue"].to_numpy() * 2.5 + data["revolving_balance"].to_numpy() * 0.015).round(2)
        data["service_cost"] = pd.Series(data["historical_retention_action"]).map({"Standard Service": 2, "Fee Waiver": 35, "Bonus Points": 20, "Specialist Call": 14}).to_numpy()
    elif scenario == "General Customer Test":
        experiences = np.array(["Current Experience", "Experience A", "Experience B"])
        data["historical_experience"] = experiences[account_codes % len(experiences)]
        data["primary_outcome"] = (data["revenue"].to_numpy() + data["utilization"].to_numpy() * 100 + rng.normal(0, 8, len(data))).round(2)
        data["satisfaction_score"] = np.clip(55 + data["utilization"].to_numpy() * 35 + rng.normal(0, 7, len(data)), 0, 100).round(1)
    elif scenario != "Credit Strategy":
        raise ValueError(f"Unknown customer demo scenario: {scenario}")

    data["demo_row_id"] = row_codes
    return data


def load_results_demo() -> pd.DataFrame:
    return experiment_results()


def load_timeseries_demo() -> pd.DataFrame:
    return time_series_campaign()


def load_timeseries_planning_demo() -> pd.DataFrame:
    return time_series_planning()


def load_geo_demo() -> pd.DataFrame:
    return geographic_panel()


def load_unity_catalog_table(_table_name: str) -> pd.DataFrame:
    raise NotImplementedError("Unity Catalog table loading is reserved for a later Databricks integration.")
