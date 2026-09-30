from datetime import date

from src.experiment_platform.portfolio import (
    PORTFOLIO_STATUSES,
    business_report_html,
    build_rollout_plan,
    duplicate_experiment,
    freeze_decision_snapshot,
    load_experiments,
    portfolio_summary,
    pulse_summary,
    save_configuration,
    save_experiments,
    seed_experiments,
    upsert_experiment,
    value_timeline,
)


def test_seed_portfolio_covers_the_full_lifecycle():
    experiments = seed_experiments()
    statuses = {item["status"] for item in experiments}
    assert set(PORTFOLIO_STATUSES).issubset(statuses)
    assert {item["type"] for item in experiments} == {"Customer", "Geography", "Time Series"}


def test_portfolio_persists_and_updates_records(tmp_path):
    target = tmp_path / "portfolio.json"
    save_experiments([], target)
    created = upsert_experiment(
        {
            "name": "Test experiment",
            "type": "Customer",
            "status": "Planning",
            "projected_value": 100,
        },
        target,
    )
    assert created["id"]
    created["status"] = "Running"
    upsert_experiment(created, target)
    saved = load_experiments(target)
    assert len(saved) == 1
    assert saved[0]["status"] == "Running"


def test_portfolio_summary_uses_business_statuses():
    summary = portfolio_summary(seed_experiments())
    assert summary["running"] == 3
    assert summary["ready"] == 1
    assert summary["attention"] == 2
    assert summary["projected_value"] > 0


def test_rollout_plan_uses_a_stable_cadence():
    plan = build_rollout_plan(date(2026, 1, 1), [10, 25, 50, 100], 7)
    assert [row["percentage"] for row in plan] == [10, 25, 50, 100]
    assert plan[-1]["date"] == "2026-01-22"


def test_value_timeline_is_ordered_and_projected_is_not_below_realized():
    timeline = value_timeline(seed_experiments())
    assert len(timeline) == 7
    assert [row["date"] for row in timeline] == sorted(row["date"] for row in timeline)
    assert all(row["projected"] >= row["realized"] for row in timeline)


def test_configuration_versions_only_change_when_configuration_changes():
    experiment = seed_experiments()[0]
    first = save_configuration(experiment, {"alpha": 0.05, "power": 0.8})
    unchanged = save_configuration(first, {"power": 0.8, "alpha": 0.05})
    changed = save_configuration(unchanged, {"alpha": 0.05, "power": 0.9})
    assert first["config_version"] == 1
    assert unchanged["config_version"] == 1
    assert changed["config_version"] == 2


def test_copy_starts_a_new_planning_record_without_decision_history():
    source = seed_experiments()[5]
    copied = duplicate_experiment(source)
    assert copied["id"] != source["id"]
    assert copied["status"] == "Planning"
    assert copied["decision_snapshot"] is None
    assert copied["frozen_snapshots"] == []


def test_final_decision_freezes_result_and_configuration_version():
    experiment = save_configuration(seed_experiments()[2], {"metric": "retention"})
    experiment["latest_result"] = {"headline": "Offer B increased retention"}
    frozen = freeze_decision_snapshot(experiment, "Launch", "Alex", "Primary and guardrail checks passed.")
    experiment["latest_result"]["headline"] = "Later recalculation"
    assert frozen["decision_snapshot"]["decision"] == "Launch"
    assert frozen["frozen_snapshots"][-1]["config_version"] == 1
    assert frozen["frozen_snapshots"][-1]["result"]["headline"] == "Offer B increased retention"


def test_pulse_reports_progress_completion_and_data_alerts():
    experiment = {
        "id": "pulse-test",
        "sample_collected": 600,
        "sample_target": 1000,
        "traffic_per_period": 100,
        "traffic_frequency": "week",
        "allocation": [
            {"arm": "Control", "observed_pct": 42, "planned_pct": 50},
            {"arm": "Treatment", "observed_pct": 58, "planned_pct": 50},
        ],
        "missing_data_rate": 3.0,
    }
    pulse = pulse_summary(experiment, today=date(2026, 1, 1))
    assert pulse["sample_progress"] == 0.6
    assert pulse["estimated_completion"] == "2026-01-29"
    assert any("allocation" in alert.lower() for alert in pulse["alerts"])
    assert any("missing data" in alert.lower() for alert in pulse["alerts"])


def test_business_report_contains_record_result_and_decision():
    experiment = freeze_decision_snapshot(seed_experiments()[0], "Stop", "Reviewer", "No practical lift.")
    report = business_report_html(experiment)
    assert "Experiment pulse" in report
    assert "No practical lift." in report
    assert "Configuration v1" in report
