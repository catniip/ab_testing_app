# Experiment Design & Measurement Platform

A business-first Streamlit application for planning, monitoring, analyzing, and documenting experiments across three common operating models:

- **Customer experiments**: randomized tests across customers or accounts
- **Geographic tests**: experiments with pre-assigned Test and Control markets
- **Time-series experiments**: campaign measurement when no randomized control group is available

The interface is designed for desktop demos and business users. Each workflow starts with a visual data story, keeps the main path simple, and places technical controls inside advanced settings.

## What The Platform Does

### Experiment Command Center

The home page gives teams a shared view of the experiment portfolio:

- Lifecycle status from Planning through Completed
- Owner, planned launch, expected readout, and next action
- Experiment Pulse for sample progress, traffic pace, allocation health, data freshness, and missing-data alerts
- Saved configurations with version history, copy, and reopen actions
- Frozen Launch / Iterate / Stop decision snapshots with reviewer and rationale
- Printable one-page HTML business reports and staged rollout plans

Saved experiments are stored locally in `.experiment_portfolio.json`. This file is intentionally ignored by Git.

### Customer Experiments

Use this workflow for acquisition policy, proactive credit-line increase, offers, pricing, retention, messaging, card design, product experiences, and other customer-level strategies.

The opening page immediately shows:

- Historical outcome by strategy
- Customer and account coverage
- Product / brand coverage
- Customer-group mix
- Available customer history and a one-row-per-account preview

The planning flow is:

1. **Bring Data**: upload customer history or use one of four demo scenarios.
2. **Choose Audience**: optionally plan separately by one business grouping such as FICO band, risk tier, or revenue band.
3. **Pick Outcome**: configure the primary outcome; secondary metrics and guardrails remain optional.
4. **Define Options**: create named variants or ordered numeric strategy values.
5. **Get Plan**: receive required accounts, traffic allocation, flow per option, and expected test duration.

Customer planning supports longitudinal unit-period data and cross-sectional one-row-per-unit data. Numeric strategies can estimate arm-specific historical means and standard deviations using closest-test-value assignment, business-defined bins, or exact values. The recommended design uses generalized Neyman allocation so higher-variance options receive the sample they need while traffic is allocated toward a common completion time.

The analysis flow accepts completed randomized-test results, checks one-row-per-unit integrity, missing outcomes, assignment balance, and sample-ratio mismatch, then produces arm-level lift, uncertainty, guardrail status, and a business recommendation.

### Geographic Tests

Use this workflow when the business already supplies fixed Test and Control DMAs or other markets. The platform does not manufacture or optimize market assignments.

The first page previews the market panel visually before configuration, including geographic coverage, Test / Control composition, outcome history, and source data. Planning then estimates the market count and rollout duration supported by the supplied groups and effect scenario.

Analysis uses panel Difference-in-Differences with:

- DMA and date fixed effects
- DMA-clustered uncertainty
- Pre-trend diagnostics
- Placebo intervention checks
- Leave-one-DMA-out sensitivity
- Optional market-size weighting and existing pair metadata

### Time-Series Experiments

Use this workflow for campaigns or policy changes where no randomized control group exists. Weekly and monthly data are supported.

The opening page shows the full historical series, campaign timing, available pre/post observations, detected frequency, predictor availability, and raw source columns.

The planning flow estimates detectable effect and campaign duration from historical pseudo-interventions. The expected campaign effect is a user-supplied business scenario, not a forecast of an unknown future effect. Results are evaluated through the longest selected duration and include calibration and limited-history safeguards.

The analysis flow offers two methods:

- **Pre-Post Analysis**: an automatically refreshed descriptive before/after comparison. It does not adjust for trend, seasonality, or concurrent changes.
- **Structural Time Series Counterfactual**: an explicitly run state-space forecast using pre-campaign history and optional unaffected predictors. It reports expected outcome without intervention, incremental impact, uncertainty, backtesting, and model-reliability diagnostics.

The current structural model uses `statsmodels`; it is not a full Bayesian sampler.

## Statistical Design Notes

- Continuous and binary outcomes use direction-aware power calculations.
- Multi-arm customer inference uses Holm-adjusted comparisons and simultaneous confidence intervals.
- Numeric customer strategies use group- and strategy-specific historical variability where support is available.
- Geography analysis assumes Test / Control assignment is fixed before upload.
- Time-series planning reports detection capability under a stated effect scenario; it cannot know the true campaign effect before launch.
- Interim Experiment Pulse directions are operational signals and are not presented as final causal conclusions.

## Quick Start

Use Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open [http://127.0.0.1:8501](http://127.0.0.1:8501).

## Suggested Demo Story

For a short presentation:

1. Open the **Command Center** to show the experiment portfolio and items requiring attention.
2. Open **Customer** to show historical strategy/outcome evidence, then enter Plan to see the business-guided workflow.
3. Open **Geography** to show the market map and fixed Test / Control story.
4. Open **Time Series** to show the campaign timeline and the distinction between planning and impact analysis.
5. Return to the Command Center and open a saved experiment, Experiment Pulse, or a frozen decision report.

## Data Inputs

CSV and Parquet uploads are supported.

### Customer

Required fields depend on the workflow, but typically include:

- Customer or account ID
- Outcome column
- Assigned treatment arm for completed-result analysis
- Optional product / brand, audience group, acquisition date, observation period, and historical strategy fields

### Geography

- Date
- DMA or market identifier
- Outcome
- Fixed Test / Control group
- Optional covariates, market weights, pair IDs, campaign spend, and exposure

### Time Series

- Date
- Outcome
- Optional unaffected comparison signals, campaign flag, spend, and operational exposure

Internal Databricks controls are UI placeholders in this MVP; Unity Catalog loading is not connected yet.

## Monthly Planning Sample

`sample_data/time_series_monthly_campaign_planning.csv` contains 84 month-start observations from January 2019 through December 2025 for the **Time Series -> Plan Campaign** workflow.

- Date: `date`
- Outcome: `credit_card_applications`
- Candidate predictors: `control_product_applications`, `organic_search_index`, `consumer_demand_index`, and `fed_funds_rate`
- Optional operational exposure: `eligible_customers`

The app should detect Monthly frequency with no missing periods and offer 2, 3, 4, 6, 9, and 12-month candidate durations. Select predictors only when they will remain available after launch and cannot be affected by the campaign.

## Project Structure

```text
app.py                              Streamlit application and workflow UI
src/experiment_platform/core.py     Shared configuration, data access, and validation
src/experiment_platform/customer.py Customer design, power, analysis, and decisions
src/experiment_platform/geography.py Geographic planning and impact analysis
src/experiment_platform/timeseries.py Time-series planning and counterfactual analysis
src/experiment_platform/charts.py   Shared Plotly visualizations
src/experiment_platform/portfolio.py Saved experiments, Pulse, reports, and decisions
sample_data/                        Example time-series input
tests/                              Unit and workflow regression tests
app.yaml                            Databricks Apps entrypoint
```

Only the files above are needed for development. Local virtual environments,
Python caches, Git metadata, and saved local portfolio state are intentionally
excluded from transfer and deployment packages.

Run the test suite with:

```bash
pytest -q
```

## Databricks App

`app.yaml` starts Streamlit on port `8000`:

```yaml
command:
  - streamlit
  - run
  - app.py
  - --server.address=0.0.0.0
  - --server.port=8000
```

## MVP Boundaries

The following are not yet implemented:

- Unity Catalog table loading and production authentication
- Shared database-backed portfolio storage and multi-user permissions
- Full Bayesian structural time-series sampling
- Approved DMA polygon geometry; the current map uses market centroids
- Randomization-based geographic power simulation for large candidate-market pools
- Covariate-adjusted, count/rate, ratio, and time-to-event customer estimators
- Production audit logging, deployment hardening, and automated data pipelines
