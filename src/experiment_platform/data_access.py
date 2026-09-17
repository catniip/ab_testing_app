from __future__ import annotations

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
