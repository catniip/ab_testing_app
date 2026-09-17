# Experiment Design & Measurement Platform

Internal Streamlit MVP for customer-level experiment design, geographic market testing, campaign time-series analysis, measurement, and decisioning.

Implemented workflow:

- Customer-Level Experimentation
- Time Series Analysis
- Geographic Test
- Data Source, Population, Metrics, Strategy, Design, Analysis, and Decision steps with shared session configuration
- Manual metric assumptions as the default fast path for experiment design
- Raw customer data processing into analysis-ready unit-level metric datasets
- Longitudinal unit-period and cross-sectional one-row-per-unit customer datasets
- Optional population segmentation, maturity eligibility, completeness diagnostics, and metric-specific historical cohorts
- Unified discrete treatment-arm setup with named variants and optional ordered numeric levels
- Reusable Credit Line, Marketing Offer, Pricing / Fee, Digital Experience, and Retention templates
- Separate Data and Metrics workflow steps with compact diagnostics and progressive disclosure
- Role-based Primary, Secondary, and Guardrail metric tabs with configurable harm thresholds
- Customer result-data audits for one-row-per-unit integrity, missing outcomes, arm counts, and sample-ratio mismatch
- Generic treatment-arm charts and arm-level decision scorecards combining evidence, practical magnitude, and guardrail safety
- Synthetic raw longitudinal portfolio data and synthetic experiment-result data
- Synthetic campaign time-series data
- Separate time-series workflow for CSV/Parquet uploads, internal-data placeholder, and synthetic demo data
- Time-series date/outcome mapping, frequency checks, duplicate timestamp diagnostics, and campaign launch setup
- Descriptive Pre-Post analysis and structural state-space counterfactual analysis with prediction intervals, pointwise impact, cumulative impact, pre-period backtesting, and decision-support summary
- Separate geographic-test workflow for DMA x time uploads, internal-data placeholder, and synthetic demo data
- Geographic date/DMA/outcome/group mapping, frequency checks, duplicate DMA/date diagnostics, fixed-assignment validation, and centroid-based U.S. map
- Rollout planning from a user-supplied effect scenario, with campaign duration, earliest readout, detection chance, MDE, and required Test/Control DMA counts
- Optional pre-existing pair metadata and market-size weighting; the platform never manufactures or changes Test/Control assignments
- Historical comparability diagnostics and panel Difference-in-Differences with DMA/date fixed effects, DMA-clustered uncertainty, pre-trend checks, placebos, and leave-one-DMA-out sensitivity
- CSV upload support, with Parquet upload support when available through pandas/pyarrow
- Modular data access, design, power, analysis, formatting, and chart logic
- Direction-aware binary and continuous power calculations
- Holm-adjusted multi-arm inference with simultaneous confidence intervals
- Separate enrollment duration, primary outcome readout, and full decision readout dates
- Result-data validation for assignment integrity, sample-ratio mismatch, binary coding, and accidental duplicate metric mappings
- Databricks Apps entrypoint in `app.yaml`

Planned but not implemented in this MVP:

- Unity Catalog table loading
- Full Bayesian state-space sampler for time-series impact modeling (the current model is a statsmodels structural state-space forecast with prediction intervals)
- DMA polygon GeoJSON layer, when an approved geometry asset is supplied
- Randomization-based geographic power simulation for sufficiently large market pools
- Alternative allocation strategies
- Covariate-adjusted, count/rate, ratio, and time-to-event estimators

## Local Run

Use Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Monthly Campaign-Planning Sample

Use `sample_data/time_series_monthly_campaign_planning.csv` to exercise the
Time Series -> Plan Campaign workflow with monthly data. It contains 84
historical month-start observations from January 2019 through December 2025
and no post-launch campaign effect.

- Date: `date`
- Outcome: `credit_card_applications`
- Candidate predictors: `control_product_applications`,
  `organic_search_index`, `consumer_demand_index`, and `fed_funds_rate`
- Optional operational exposure: `eligible_customers`

The app should detect `Monthly` frequency with no missing periods. The planner
will offer 2, 3, 4, 6, 9, and 12-month candidate durations. Under
`Advanced / Model Inputs`, optionally select predictors only when they will be
available after launch and cannot be changed by the campaign.

## Databricks App

The `app.yaml` starts Streamlit on port `8000`:

```yaml
command:
  - streamlit
  - run
  - app.py
  - --server.address=0.0.0.0
  - --server.port=8000
```
