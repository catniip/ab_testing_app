from __future__ import annotations

import math

from scipy.stats import nct, t
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def adjusted_alpha(alpha: float, comparisons: int, method: str = "Holm") -> float:
    """Return the conservative per-comparison alpha used for design planning."""
    if comparisons <= 1 or method == "None":
        return alpha
    return alpha / comparisons


def sample_size_continuous(alpha: float, power: float, mde: float, std_dev: float) -> int:
    effect_size = abs(mde) / std_dev
    n = TTestIndPower().solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0)
    return int(math.ceil(n))


def power_continuous_unequal(alpha: float, effect: float, control_sd: float, treatment_sd: float, n_control: int, n_treatment: int | None = None) -> float:
    """Approximate two-sided Welch-test power for unequal variances and enrollment."""
    n_treatment = n_control if n_treatment is None else n_treatment
    if n_control < 2 or n_treatment < 2 or control_sd <= 0 or treatment_sd <= 0:
        return 0.0
    control_var = control_sd**2 / n_control
    treatment_var = treatment_sd**2 / n_treatment
    standard_error = math.sqrt(control_var + treatment_var)
    degrees_freedom = (control_var + treatment_var) ** 2 / (
        control_var**2 / (n_control - 1) + treatment_var**2 / (n_treatment - 1)
    )
    critical = t.ppf(1 - alpha / 2, degrees_freedom)
    noncentrality = abs(effect) / standard_error
    return float(nct.cdf(-critical, degrees_freedom, noncentrality) + 1 - nct.cdf(critical, degrees_freedom, noncentrality))


def power_binary_unequal(alpha: float, baseline_rate: float, effect: float, n_control: int, n_treatment: int, direction: str = "Higher is Better") -> float:
    """Two-proportion normal-test power with an unequal treatment/control ratio."""
    signed_effect = -abs(effect) if direction == "Lower is Better" else abs(effect)
    treatment_rate = min(max(baseline_rate + signed_effect, 1e-6), 1 - 1e-6)
    effect_size = abs(proportion_effectsize(baseline_rate, treatment_rate))
    return float(NormalIndPower().power(effect_size=effect_size, nobs1=n_control, alpha=alpha, ratio=n_treatment / n_control))


def sample_size_continuous_unequal(alpha: float, power: float, mde: float, control_sd: float, treatment_sd: float) -> int:
    """Solve equal per-arm enrollment for a two-sided unequal-variance comparison."""
    if mde <= 0 or control_sd <= 0 or treatment_sd <= 0:
        raise ValueError("MDE and both historical standard deviations must be greater than zero.")
    low, high = 2, 4
    while power_continuous_unequal(alpha, mde, control_sd, treatment_sd, high) < power:
        high *= 2
        if high > 10_000_000:
            raise ValueError("Required sample size exceeds the supported planning range.")
    while low < high:
        middle = (low + high) // 2
        if power_continuous_unequal(alpha, mde, control_sd, treatment_sd, middle) >= power:
            high = middle
        else:
            low = middle + 1
    return low


def detectable_effect_continuous_unequal(alpha: float, power: float, control_sd: float, treatment_sd: float, n_per_arm: int) -> float:
    """Solve the absolute detectable effect for unequal historical variances."""
    low = 0.0
    high = max(control_sd, treatment_sd)
    while power_continuous_unequal(alpha, high, control_sd, treatment_sd, n_per_arm) < power:
        high *= 2
    for _ in range(60):
        middle = (low + high) / 2
        if power_continuous_unequal(alpha, middle, control_sd, treatment_sd, n_per_arm) >= power:
            high = middle
        else:
            low = middle
    return high


def sample_size_binary(alpha: float, power: float, baseline_rate: float, mde: float, direction: str = "Higher is Better") -> int:
    signed_effect = -abs(mde) if direction == "Lower is Better" else abs(mde)
    treatment_rate = min(max(baseline_rate + signed_effect, 1e-6), 1 - 1e-6)
    effect_size = abs(proportion_effectsize(baseline_rate, treatment_rate))
    n = NormalIndPower().solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0)
    return int(math.ceil(n))


def duration(total_sample_size: int, eligible_count: int, frequency: str) -> float:
    if eligible_count <= 0:
        return float("inf")
    units = total_sample_size / eligible_count
    return units if frequency == "Week" else units


def detectable_effect_continuous(alpha: float, power: float, std_dev: float, n_per_arm: int) -> float:
    effect_size = TTestIndPower().solve_power(nobs1=n_per_arm, alpha=alpha, power=power, ratio=1.0)
    return abs(effect_size * std_dev)


def detectable_effect_binary(alpha: float, power: float, baseline_rate: float, n_per_arm: int, direction: str = "Higher is Better") -> float:
    grid = [idx / 10000 for idx in range(1, 5000)]
    for effect in grid:
        if power_binary(alpha, baseline_rate, effect, n_per_arm, direction) >= power:
            return effect
    return grid[-1]


def power_continuous(alpha: float, effect: float, std_dev: float, n_per_arm: int) -> float:
    return float(TTestIndPower().power(effect_size=abs(effect) / std_dev, nobs1=n_per_arm, alpha=alpha, ratio=1.0))


def power_binary(alpha: float, baseline_rate: float, effect: float, n_per_arm: int, direction: str = "Higher is Better") -> float:
    signed_effect = -abs(effect) if direction == "Lower is Better" else abs(effect)
    treatment_rate = min(max(baseline_rate + signed_effect, 1e-6), 1 - 1e-6)
    effect_size = abs(proportion_effectsize(baseline_rate, treatment_rate))
    return float(NormalIndPower().power(effect_size=effect_size, nobs1=n_per_arm, alpha=alpha, ratio=1.0))
