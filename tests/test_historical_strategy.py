import numpy as np
import pandas as pd
import pytest

from src.experiment_platform.customer import design_neyman_allocation, design_sample_size_by_arm
from src.experiment_platform.customer import binned_historical_statistics, historical_arm_statistics, map_to_strategy_points, validate_fixed_unit_value
from src.experiment_platform.core import DesignConfig, MetricConfig
from src.experiment_platform.customer import sample_size_continuous_unequal


def historical_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    rows = []
    for line, mean, sd in [(3000, 1000, 80), (5000, 1400, 160), (8000, 1900, 320)]:
        for index, outcome in enumerate(rng.normal(mean, sd, 80)):
            rows.append({"customer_id": f"{line}-{index}", "acquisition_line": line, "outcome": outcome})
    return pd.DataFrame(rows)


def test_fixed_acquisition_line_validation_rejects_changing_values():
    data = pd.DataFrame(
        {
            "customer_id": ["A", "A", "B", "B"],
            "acquisition_line": [3000, 5000, 8000, 8000],
        }
    )
    errors = validate_fixed_unit_value(data, "customer_id", "acquisition_line")
    assert len(errors) == 1
    assert "1 unit has multiple values" in errors[0]


def test_exact_and_nearest_point_mapping_are_distinct():
    values = pd.Series([2800, 3000, 4700, 8000, 9000])
    exact = map_to_strategy_points(values, [3000, 5000, 8000], "Exact acquisition lines")
    nearest = map_to_strategy_points(values, [3000, 5000, 8000], "Nearest proposed point")
    assert exact["Strategy Point"].notna().sum() == 2
    assert nearest["Strategy Point"].tolist() == [3000, 3000, 5000, 8000, 8000]


def test_maximum_mapping_distance_excludes_distant_history():
    mapped = map_to_strategy_points(pd.Series([3000, 3600, 5000]), [3000, 5000], "Nearest proposed point", max_distance=500)
    assert mapped["Strategy Point"].tolist()[0] == 3000
    assert pd.isna(mapped["Strategy Point"].tolist()[1])
    assert mapped["Strategy Point"].tolist()[2] == 5000


def test_historical_sd_is_calculated_separately_for_each_line():
    stats = historical_arm_statistics(historical_fixture(), "acquisition_line", "outcome", [3000, 5000, 8000])
    by_line = stats.set_index("Strategy Point")
    assert by_line.loc[3000, "Historical N"] == 80
    assert by_line.loc[3000, "Historical SD"] < by_line.loc[5000, "Historical SD"]
    assert by_line.loc[5000, "Historical SD"] < by_line.loc[8000, "Historical SD"]


def test_nearest_point_groups_use_every_historical_customer_once():
    stats = historical_arm_statistics(
        historical_fixture(),
        "acquisition_line",
        "outcome",
        [3000, 8000],
        grouping="Nearest proposed point",
    )
    assert stats["Historical N"].sum() == len(historical_fixture())
    assert stats["Strategy Point"].tolist() == [3000, 8000]


def test_business_width_uses_bounded_neighborhoods_around_test_lines():
    data = pd.DataFrame(
        {
            "acquisition_line": [2500, 3000, 3500, 5000, 7500, 8000, 8500],
            "outcome": [1, 2, 3, 99, 7, 8, 9],
        }
    )
    stats = historical_arm_statistics(
        data,
        "acquisition_line",
        "outcome",
        [3000, 8000],
        grouping="Business-defined bin width",
        max_distance=500,
    )
    assert stats["Historical N"].tolist() == [3, 3]
    assert stats["Observed Line Range"].tolist() == ["2,500 to 3,500", "7,500 to 8,500"]


def test_fixed_width_curve_binning_is_stable_and_ordered():
    data = pd.DataFrame(
        {
            "acquisition_line": [1900, 2000, 2100, 3900, 4000, 4100],
            "outcome": [10, 12, 14, 20, 22, 24],
        }
    )
    stats = binned_historical_statistics(data, "acquisition_line", "outcome", method="Fixed bin width", bin_width=1000)
    assert stats["Strategy Point"].tolist() == [2000.0, 4000.0]
    assert stats["Historical N"].tolist() == [3, 3]
    assert stats["Historical Mean"].tolist() == [12.0, 22.0]


def test_higher_treatment_variance_requires_more_customers():
    low_variance = sample_size_continuous_unequal(0.05, 0.8, 100, 100, 100)
    high_variance = sample_size_continuous_unequal(0.05, 0.8, 100, 100, 300)
    assert high_variance > low_variance


def test_design_uses_the_most_demanding_line_specific_comparison():
    metric = MetricConfig(
        "Outcome",
        "outcome",
        "Primary",
        metric_type="Continuous",
        effect_type="Absolute",
        effect_value=100,
    )
    stats = historical_arm_statistics(historical_fixture(), "acquisition_line", "outcome", [3000, 5000, 8000])
    required, binding, rows = design_sample_size_by_arm(
        {"Primary": metric},
        DesignConfig(),
        {"Primary": stats},
        5000,
        [3000, 8000],
    )
    by_comparison = {row["Comparison"]: row["Required N / Arm"] for row in rows}
    assert by_comparison["5,000 vs 8,000"] > by_comparison["5,000 vs 3,000"]
    assert required == by_comparison["5,000 vs 8,000"]
    assert "5,000 vs 8,000" in binding


def test_design_fails_closed_when_a_point_has_no_historical_sd():
    metric = MetricConfig("Outcome", "outcome", "Primary", metric_type="Continuous", effect_type="Absolute", effect_value=100)
    stats = historical_arm_statistics(historical_fixture(), "acquisition_line", "outcome", [3000, 5000, 7000])
    with pytest.raises(ValueError, match="no historical support"):
        design_sample_size_by_arm({"Primary": metric}, DesignConfig(), {"Primary": stats}, 5000, [7000])


def test_neyman_allocation_gives_more_sample_to_higher_variance_arm():
    metric = MetricConfig("Outcome", "outcome", "Primary", metric_type="Continuous", effect_type="Absolute", effect_value=100)
    stats = historical_arm_statistics(historical_fixture(), "acquisition_line", "outcome", [3000, 5000, 8000])
    total, binding, comparisons, allocation = design_neyman_allocation(
        {"Primary": metric},
        DesignConfig(target_power=0.8),
        {"Primary": stats},
        5000,
        [3000, 8000],
    )
    by_line = {row["Strategy Point"]: row["Analyzable N"] for row in allocation}
    assert sum(by_line.values()) == total
    assert by_line[8000] > by_line[3000]
    assert all(row["Planned Power"] >= 0.8 for row in comparisons)
    assert binding
