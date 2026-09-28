from src.experiment_platform.customer_templates import CUSTOMER_EXPERIMENT_TEMPLATES, apply_customer_template
from src.experiment_platform.models import StrategyConfig


def test_general_customer_templates_cover_core_strategy_families():
    assert {
        "Acquisition Credit Line",
        "Proactive Credit Line Increase",
        "Multiple Offer Strategy",
        "Pricing / Fee",
        "Retention / Save Offer",
        "Rewards / Incentive",
        "Channel / Message",
        "Card / Product Design",
        "Collections Treatment",
        "Custom",
    }.issubset(CUSTOMER_EXPERIMENT_TEMPLATES)


def test_offer_template_creates_named_randomized_arms():
    strategy = StrategyConfig()
    apply_customer_template(strategy, "Multiple Offer Strategy")
    assert strategy.strategy_type == "Categorical Strategy"
    assert strategy.categorical_control == "BAU Offer"
    assert len(strategy.categorical_treatments) == 3
    assert strategy.assignment_column == "assigned_offer"
    assert strategy.historical_column == ""


def test_proactive_line_increase_does_not_require_historical_treatment_values():
    strategy = StrategyConfig()
    apply_customer_template(strategy, "Proactive Credit Line Increase")
    assert strategy.strategy_type == "Numeric Strategy"
    assert strategy.value_format == "Percent"
    assert strategy.control_value == 0
    assert strategy.historical_evidence == "Use group-level outcome history"
