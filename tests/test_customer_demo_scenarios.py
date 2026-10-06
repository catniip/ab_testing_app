from src.experiment_platform.core import CUSTOMER_DEMO_SCENARIOS, load_customer_demo
from src.experiment_platform.core import StrategyConfig


def test_customer_demo_scenarios_have_customer_history_and_numeric_outcomes():
    for scenario in CUSTOMER_DEMO_SCENARIOS:
        data = load_customer_demo(scenario)
        assert not data.empty
        assert {"account_id", "mob", "booking_date"}.issubset(data.columns)
        assert data.select_dtypes(include="number").shape[1] >= 3


def test_demo_historical_strategy_fields_are_fixed_per_account():
    for scenario, field in {
        "General Customer Test": "historical_experience",
        "Offer & Engagement": "historical_offer",
        "Retention Strategy": "historical_retention_action",
        "Credit Strategy": "current_credit_line",
    }.items():
        data = load_customer_demo(scenario)
        assert data.groupby("account_id")[field].nunique(dropna=True).max() == 1


def test_customer_strategy_defaults_are_business_neutral():
    strategy = StrategyConfig()
    assert strategy.experiment_template == "Custom"
    assert strategy.strategy_type == "Categorical Strategy"
    assert strategy.categorical_control == "Control"
    assert strategy.categorical_treatments == ["Treatment A"]
