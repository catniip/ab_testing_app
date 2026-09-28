from __future__ import annotations

from copy import deepcopy
from typing import Any

from .models import StrategyConfig


CUSTOMER_EXPERIMENT_TEMPLATES: dict[str, dict[str, Any]] = {
    "Acquisition Credit Line": {
        "summary": "Compare credit lines assigned when a new account is opened.",
        "examples": "3,000 vs 5,000 vs 8,000 acquisition lines",
        "arm_format": "Ordered numeric levels",
        "strategy_name": "Acquisition Credit Line",
        "strategy_goal": "Optimize the acquisition credit line offered to eligible customers.",
        "value_label": "Credit Line",
        "value_format": "Currency",
        "assignment_column": "assigned_credit_line",
        "historical_column": "current_credit_line",
        "historical_evidence": "Use historical strategy values",
        "control_name": "BAU",
        "control_value": 5000.0,
        "treatment_names": ["Lower Line", "Higher Line"],
        "treatment_values": [3000.0, 8000.0],
        "minimum": 2000.0,
        "maximum": 10000.0,
        "increment": 500.0,
    },
    "Proactive Credit Line Increase": {
        "summary": "Test proactive line-increase policies for existing customers.",
        "examples": "No increase vs +10% vs +25%, or fixed-dollar increase policies",
        "arm_format": "Ordered numeric levels",
        "strategy_name": "Proactive Line Increase",
        "strategy_goal": "Measure the incremental value and risk of proactive credit line increases.",
        "value_label": "Line Increase",
        "value_format": "Percent",
        "assignment_column": "assigned_line_increase_pct",
        "historical_column": "",
        "historical_evidence": "Use group-level outcome history",
        "control_name": "No Increase",
        "control_value": 0.0,
        "treatment_names": ["Moderate Increase", "High Increase"],
        "treatment_values": [10.0, 25.0],
        "minimum": 0.0,
        "maximum": 50.0,
        "increment": 5.0,
    },
    "Multiple Offer Strategy": {
        "summary": "Compare complete offer packages that may differ on several attributes.",
        "examples": "BAU vs cash bonus vs APR offer vs rewards bundle",
        "arm_format": "Named variants",
        "strategy_name": "Offer Strategy",
        "strategy_goal": "Select the offer package with the best incremental customer and business outcome.",
        "assignment_column": "assigned_offer",
        "historical_column": "",
        "historical_evidence": "Use group-level outcome history",
        "control": "BAU Offer",
        "treatments": ["Cash Bonus", "APR Offer", "Rewards Bundle"],
        "descriptions": {
            "BAU Offer": "Current standard offer",
            "Cash Bonus": "One-time acquisition or activation bonus",
            "APR Offer": "Promotional interest-rate offer",
            "Rewards Bundle": "Enhanced rewards and benefits package",
        },
    },
    "Pricing / Fee": {
        "summary": "Test ordered prices, annual fees, or promotional rates.",
        "examples": "0 vs 49 vs 95 annual fee",
        "arm_format": "Ordered numeric levels",
        "strategy_name": "Price or Fee",
        "strategy_goal": "Find the price or fee that maximizes risk-adjusted value.",
        "value_label": "Price / Fee",
        "value_format": "Currency",
        "assignment_column": "assigned_price",
        "historical_column": "",
        "historical_evidence": "Use group-level outcome history",
        "control_name": "Current Price",
        "control_value": 10.0,
        "treatment_names": ["Lower", "Higher"],
        "treatment_values": [5.0, 15.0],
        "minimum": 0.0,
        "maximum": 100.0,
        "increment": 1.0,
    },
    "Retention / Save Offer": {
        "summary": "Compare interventions intended to retain an at-risk customer.",
        "examples": "Standard outreach vs fee waiver vs points vs specialist call",
        "arm_format": "Named variants",
        "strategy_name": "Retention Strategy",
        "strategy_goal": "Increase retained value while controlling incentive cost and adverse selection.",
        "assignment_column": "assigned_retention_strategy",
        "historical_column": "",
        "historical_evidence": "Use group-level outcome history",
        "control": "Standard Outreach",
        "treatments": ["Fee Waiver", "Bonus Points", "Specialist Call"],
    },
    "Rewards / Incentive": {
        "summary": "Compare rewards, bonuses, or spend incentives.",
        "examples": "No bonus vs statement credit vs points multiplier",
        "arm_format": "Named variants",
        "strategy_name": "Rewards Strategy",
        "strategy_goal": "Identify the incentive that creates the greatest incremental value net of cost.",
        "assignment_column": "assigned_reward",
        "historical_column": "",
        "historical_evidence": "Use group-level outcome history",
        "control": "Current Rewards",
        "treatments": ["Statement Credit", "Points Multiplier"],
    },
    "Channel / Message": {
        "summary": "Compare contact channels, creative, cadence, or message framing.",
        "examples": "Email vs push vs SMS, or benefit-led vs urgency-led copy",
        "arm_format": "Named variants",
        "strategy_name": "Contact Strategy",
        "strategy_goal": "Choose the contact experience that improves response without increasing opt-outs or complaints.",
        "assignment_column": "assigned_contact_strategy",
        "historical_column": "",
        "historical_evidence": "Use group-level outcome history",
        "control": "Current Contact",
        "treatments": ["Email Variant", "Push Variant", "SMS Variant"],
    },
    "Card / Product Design": {
        "summary": "Compare product configurations, card designs, or benefit packages.",
        "examples": "Current card vs premium design vs eco design",
        "arm_format": "Named variants",
        "strategy_name": "Product Design",
        "strategy_goal": "Measure whether a product design changes activation, usage, satisfaction, or retention.",
        "assignment_column": "assigned_product_design",
        "historical_column": "",
        "historical_evidence": "Use group-level outcome history",
        "control": "Current Design",
        "treatments": ["Design A", "Design B"],
    },
    "Collections Treatment": {
        "summary": "Compare treatment paths for delinquent or financially stressed customers.",
        "examples": "BAU collections vs digital self-service vs payment plan",
        "arm_format": "Named variants",
        "strategy_name": "Collections Treatment",
        "strategy_goal": "Improve cure and repayment outcomes while protecting customer experience.",
        "assignment_column": "assigned_collections_treatment",
        "historical_column": "",
        "historical_evidence": "Use group-level outcome history",
        "control": "BAU Collections",
        "treatments": ["Digital Self-Service", "Payment Plan"],
    },
    "Custom": {
        "summary": "Build a custom randomized customer-level strategy test.",
        "examples": "Any mutually exclusive set of assignable strategies",
        "arm_format": "Named variants",
        "strategy_name": "Treatment",
        "strategy_goal": "",
        "assignment_column": "assigned_treatment",
        "historical_column": "",
        "historical_evidence": "Use group-level outcome history",
        "control": "Control",
        "treatments": ["Treatment A"],
        "descriptions": {
            "Control": "Current experience or business as usual",
            "Treatment A": "Proposed alternative",
        },
    },
}


def apply_customer_template(strategy: StrategyConfig, template: str) -> None:
    if template not in CUSTOMER_EXPERIMENT_TEMPLATES:
        raise ValueError(f"Unknown customer experiment template: {template}")
    strategy.experiment_template = template
    config = deepcopy(CUSTOMER_EXPERIMENT_TEMPLATES[template])
    strategy.arm_format = config["arm_format"]
    strategy.strategy_type = "Numeric Strategy" if strategy.arm_format == "Ordered numeric levels" else "Categorical Strategy"
    strategy.strategy_name = config["strategy_name"]
    strategy.strategy_goal = config["strategy_goal"]
    strategy.assignment_column = config["assignment_column"]
    strategy.historical_column = config.get("historical_column", "")
    strategy.historical_evidence = config.get("historical_evidence", "Use group-level outcome history")
    strategy.value_label = config.get("value_label", "Strategy Value")
    strategy.value_format = config.get("value_format", "Number")
    strategy.group_arms = {}
    if strategy.strategy_type == "Numeric Strategy":
        strategy.control_name = config["control_name"]
        strategy.control_value = config["control_value"]
        strategy.treatment_names = list(config["treatment_names"])
        strategy.treatment_values = list(config["treatment_values"])
        strategy.min_value = config["minimum"]
        strategy.max_value = config["maximum"]
        strategy.increment = config["increment"]
        strategy.number_of_arms = len(strategy.treatment_values) + 1
        strategy.arm_descriptions = {}
    else:
        strategy.categorical_control = config["control"]
        strategy.categorical_treatments = list(config["treatments"])
        strategy.number_of_arms = len(strategy.categorical_treatments) + 1
        strategy.arm_descriptions = dict(config.get("descriptions", {}))
