from __future__ import annotations

import numpy as np
import pandas as pd

from .demo_data import experiment_results, geographic_panel, historical_portfolio, raw_longitudinal_portfolio, time_series_campaign, time_series_planning
from .data_validation import normalize_uploaded_dataset


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
