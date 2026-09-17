from __future__ import annotations

import numpy as np
import pandas as pd


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
    booking_offsets = rng.integers(3, 54, size=accounts)
    for idx in range(accounts):
        account_id = f"A{idx + 1:06d}"
        booking_date = cutoff - pd.DateOffset(months=int(booking_offsets[idx]))
        max_mob = int(((cutoff.year - booking_date.year) * 12) + cutoff.month - booking_date.month)
        credit_line = int(np.clip(np.round(rng.normal(5600, 1900) / 500) * 500, 1000, 12000))
        base_balance = max(150, credit_line * rng.uniform(0.18, 0.55))
        cpc = cpcs[idx]
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
