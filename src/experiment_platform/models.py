from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
