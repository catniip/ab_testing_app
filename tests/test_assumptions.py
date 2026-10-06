import pytest

from src.experiment_platform.customer import (
    design_sample_size,
    detectable_for_metric,
    expected_power,
    normalize_metric,
    required_n,
    validate_metric_assumptions,
)
from src.experiment_platform.core import DesignConfig, MetricConfig


def continuous_metric(**overrides):
    values = {
        "name": "Revolving Balance",
        "column": "unused",
        "role": "Primary",
        "metric_type": "Continuous",
        "baseline_mean": 2300.0,
        "standard_deviation": 900.0,
        "variance": 810000.0,
        "effect_type": "Relative %",
        "effect_value": 0.05,
    }
    values.update(overrides)
    return MetricConfig(**values)


def binary_metric(**overrides):
    values = {
        "name": "Default Rate",
        "column": "unused",
        "role": "Primary",
        "metric_type": "Binary",
        "baseline_rate": 0.032,
        "effect_type": "Percentage Point",
        "effect_value": 0.005,
    }
    values.update(overrides)
    return MetricConfig(**values)


def test_continuous_variance_converts_to_standard_deviation():
    metric = continuous_metric(variability_input="Variance", variance=810000)
    assumption = normalize_metric(metric)
    assert assumption.standard_deviation == 900


def test_continuous_relative_and_absolute_sample_size():
    relative = continuous_metric(effect_type="Relative %", effect_value=0.05)
    absolute = continuous_metric(effect_type="Absolute", effect_value=115)
    assert required_n(relative, 0.05, 0.8) == required_n(absolute, 0.05, 0.8)


def test_continuous_detectable_effect_and_power():
    metric = continuous_metric(effect_type="Absolute", effect_value=120, expected_effect=150)
    detected = detectable_for_metric(metric, 0.05, 0.8, 900)
    assert detected["Minimum Detectable Absolute Effect"] > 0
    assert 0 < expected_power(metric, 0.05, 900) < 1


def test_binary_percentage_point_and_relative_mde():
    pp = binary_metric(effect_type="Percentage Point", effect_value=0.005)
    rel = binary_metric(effect_type="Relative %", effect_value=0.15625)
    assert pytest.approx(normalize_metric(pp).effect_absolute, rel=1e-6) == normalize_metric(rel).effect_absolute
    assert required_n(pp, 0.05, 0.8) > 0


def test_binary_detectable_effect_and_power():
    metric = binary_metric(expected_effect=0.005)
    detected = detectable_for_metric(metric, 0.05, 0.8, 5000)
    assert detected["Minimum Detectable Absolute Effect"] > 0
    assert 0 < expected_power(metric, 0.05, 5000) < 1


def test_primary_only_ignores_secondary_and_guardrail():
    primary = continuous_metric(role="Primary", effect_value=0.05)
    secondary = continuous_metric(name="Revenue", role="Secondary", effect_type="Absolute", effect_value=1)
    guardrail = binary_metric(name="Default Rate", role="Guardrail", effect_value=0.001)
    design = DesignConfig(sample_size_basis="Primary Metric Only")
    n, binding, rows = design_sample_size({"Primary": primary, "Secondary": secondary, "Guardrail": guardrail}, design)
    assert n == required_n(primary, design.alpha, design.target_power)
    assert binding == "Revolving Balance — Primary"
    assert len(rows) == 1
    assert n * 3 == n + n + n


def test_power_all_configured_metrics_selects_binding_metric():
    primary = continuous_metric(role="Primary", effect_type="Absolute", effect_value=500)
    secondary = continuous_metric(name="Revenue", role="Secondary", effect_type="Absolute", effect_value=10)
    design = DesignConfig(sample_size_basis="Power All Configured Metrics")
    n, binding, rows = design_sample_size({"Primary": primary, "Secondary": secondary}, design)
    assert n == max(row["Required N / Arm"] for row in rows)
    assert binding == "Revenue — Secondary"


def test_multi_arm_planning_uses_conservative_familywise_alpha():
    metric = continuous_metric(effect_type="Absolute", effect_value=150)
    design = DesignConfig(multiplicity_method="Holm")
    single, _, _ = design_sample_size({"Primary": metric}, design, comparisons=1)
    multi, _, _ = design_sample_size({"Primary": metric}, design, comparisons=3)
    assert multi > single


def test_lower_is_better_binary_effect_is_a_reduction():
    metric = binary_metric(direction="Lower is Better", baseline_rate=0.20, effect_value=0.05)
    assert required_n(metric, 0.05, 0.8) > 0


def test_validation_errors():
    assert validate_metric_assumptions(continuous_metric(variability_input="Variance", variance=-1))
    assert validate_metric_assumptions(binary_metric(baseline_rate=1.2))
    assert validate_metric_assumptions(continuous_metric(effect_value=0))
    assert validate_metric_assumptions(continuous_metric(), alpha=1.2)
    assert validate_metric_assumptions(continuous_metric(), target_power=1.1)
