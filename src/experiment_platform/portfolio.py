from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from scipy import stats
from typing import Any
from uuid import uuid4


PORTFOLIO_STATUSES = [
    "Planning",
    "Running",
    "Collecting outcomes",
    "Ready to decide",
    "Rolling out",
    "Completed",
]

DEFAULT_PORTFOLIO_PATH = Path(
    os.environ.get(
        "EXPERIMENT_PORTFOLIO_PATH",
        Path(__file__).resolve().parents[2] / ".experiment_portfolio.json",
    )
)


def _day(offset: int) -> str:
    return (date.today() + timedelta(days=offset)).isoformat()


PULSE_DEMO_DEFAULTS: dict[str, dict[str, Any]] = {
    "customer-acquisition-line": {
        "planned_launch_date": _day(-28),
        "sample_collected": 6800,
        "sample_target": 10000,
        "traffic_per_period": 850,
        "traffic_frequency": "week",
        "allocation": [
            {"arm": "BAU", "observed_pct": 49.2, "planned_pct": 50.0},
            {"arm": "New line strategy", "observed_pct": 50.8, "planned_pct": 50.0},
        ],
        "primary_direction": "Up 4.8% so far",
        "missing_data_rate": 0.4,
        "latest_result": {"headline": "Primary outcome is trending up 4.8%", "confidence": "Interim — not a final decision"},
    },
    "geo-holiday-media": {
        "planned_launch_date": _day(-21),
        "sample_collected": 11,
        "sample_target": 24,
        "traffic_per_period": 2,
        "traffic_frequency": "week",
        "allocation": [
            {"arm": "Control markets", "observed_pct": 47.0, "planned_pct": 50.0},
            {"arm": "Test markets", "observed_pct": 53.0, "planned_pct": 50.0},
        ],
        "primary_direction": "Direction not yet stable",
        "missing_data_rate": 3.2,
        "latest_result": {"headline": "Data collection is still in progress", "confidence": "Interim — not a final decision"},
    },
    "customer-retention-offer": {
        "planned_launch_date": _day(-72),
        "sample_collected": 12000,
        "sample_target": 12000,
        "traffic_per_period": 1400,
        "traffic_frequency": "week",
        "allocation": [
            {"arm": "BAU", "observed_pct": 33.4, "planned_pct": 33.3},
            {"arm": "Offer A", "observed_pct": 33.1, "planned_pct": 33.3},
            {"arm": "Offer B", "observed_pct": 33.5, "planned_pct": 33.4},
        ],
        "primary_direction": "Up 3.4 percentage points",
        "missing_data_rate": 0.2,
        "latest_result": {"headline": "Offer B improved 90-day retention by 3.4 points", "confidence": "Analysis complete"},
    },
    "customer-activation": {
        "planned_launch_date": _day(18),
        "sample_collected": 0,
        "sample_target": 9000,
        "traffic_per_period": 1100,
        "traffic_frequency": "week",
        "allocation": [],
        "primary_direction": "Not launched",
        "missing_data_rate": 0.0,
        "latest_result": {},
    },
    "ts-search-demand": {
        "planned_launch_date": _day(-70),
        "sample_collected": 10,
        "sample_target": 16,
        "traffic_per_period": 1,
        "traffic_frequency": "weekly observation",
        "allocation": [],
        "primary_direction": "Above expected trajectory",
        "missing_data_rate": 0.0,
        "latest_result": {"headline": "Observed demand is above the expected trajectory", "confidence": "Interim — not a final decision"},
    },
    "customer-dining-offer": {
        "planned_launch_date": _day(-95),
        "sample_collected": 8400,
        "sample_target": 8400,
        "traffic_per_period": 1200,
        "traffic_frequency": "week",
        "allocation": [
            {"arm": "BAU", "observed_pct": 50.1, "planned_pct": 50.0},
            {"arm": "Dining offer", "observed_pct": 49.9, "planned_pct": 50.0},
        ],
        "primary_direction": "Up $18 per eligible account",
        "missing_data_rate": 0.1,
        "latest_result": {"headline": "Dining offer increased eligible spend", "confidence": "Decision frozen"},
    },
    "ts-brand-reengagement": {
        "planned_launch_date": _day(-130),
        "sample_collected": 12,
        "sample_target": 12,
        "traffic_per_period": 1,
        "traffic_frequency": "weekly observation",
        "allocation": [],
        "primary_direction": "Up 14% versus expected",
        "missing_data_rate": 0.0,
        "latest_result": {"headline": "Weekly active accounts increased 14%", "confidence": "Decision frozen"},
    },
    "geo-southeast-expansion": {
        "planned_launch_date": _day(32),
        "sample_collected": 0,
        "sample_target": 18,
        "traffic_per_period": 0,
        "traffic_frequency": "week",
        "allocation": [],
        "primary_direction": "Not launched",
        "missing_data_rate": 0.0,
        "latest_result": {},
    },
}


def normalize_experiment(experiment: dict[str, Any]) -> dict[str, Any]:
    record = copy.deepcopy(experiment)
    pulse = PULSE_DEMO_DEFAULTS.get(str(record.get("id", "")), {})
    now = datetime.now().isoformat(timespec="seconds")
    defaults: dict[str, Any] = {
        "description": "",
        "owner": "Unassigned",
        "primary_outcome": "Primary outcome",
        "status": "Planning",
        "progress": 0,
        "planned_launch_date": _day(14),
        "readout_date": _day(60),
        "action": "Complete test plan",
        "projected_value": 0,
        "allocation_health": 100,
        "alert": "",
        "guardrails": [],
        "decision_snapshot": None,
        "rollout_plan": [],
        "config_version": 1,
        "config_fingerprint": "",
        "config_payload": {},
        "analysis_timestamp": "",
        "latest_result": {},
        "frozen_snapshots": [],
        "sample_collected": 0,
        "sample_target": 0,
        "traffic_per_period": 0,
        "traffic_frequency": "week",
        "allocation": [],
        "primary_direction": "No result yet",
        "missing_data_rate": 0.0,
        "data_updated_at": record.get("updated_at", now),
        "updated_at": now,
    }
    for key, value in {**defaults, **pulse}.items():
        record.setdefault(key, copy.deepcopy(value))
    if record.get("traffic_frequency") == "markets per week":
        record["traffic_frequency"] = "week"
    if record.get("decision_snapshot") and not record.get("frozen_snapshots"):
        record["frozen_snapshots"] = [
            {
                "snapshot_id": f'snapshot-{record["id"]}-1',
                "created_at": record["decision_snapshot"].get("created_at", record["updated_at"]),
                "config_version": record.get("config_version", 1),
                "analysis_timestamp": record.get("analysis_timestamp", ""),
                "decision": copy.deepcopy(record["decision_snapshot"]),
                "result": copy.deepcopy(record.get("latest_result", {})),
            }
        ]
    return record


def seed_experiments() -> list[dict[str, Any]]:
    """Return realistic records that make the command center useful on first launch."""
    return [
        {
            "id": "customer-acquisition-line",
            "name": "Acquisition Line Optimization",
            "description": "Compare three acquisition-line strategies for eligible card applicants.",
            "type": "Customer",
            "owner": "Credit Strategy",
            "primary_outcome": "12-month revolving balance",
            "status": "Running",
            "progress": 68,
            "readout_date": _day(24),
            "action": "Keep running",
            "projected_value": 620000,
            "allocation_health": 97,
            "alert": "",
            "guardrails": [
                {"name": "Loss rate", "value": "3.1%", "threshold": "< 3.5%", "status": "Healthy"},
                {"name": "Approval rate", "value": "41.8%", "threshold": "> 40%", "status": "Healthy"},
            ],
            "decision_snapshot": None,
            "rollout_plan": [],
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
        {
            "id": "geo-holiday-media",
            "name": "Holiday Market Messaging",
            "description": "Measure incremental applications across fixed Test and Control DMAs.",
            "type": "Geography",
            "owner": "Growth Marketing",
            "primary_outcome": "Applications per market",
            "status": "Collecting outcomes",
            "progress": 46,
            "readout_date": _day(38),
            "action": "Monitor closely",
            "projected_value": 410000,
            "allocation_health": 88,
            "alert": "Two Test markets have delayed data",
            "guardrails": [
                {"name": "Market coverage", "value": "94%", "threshold": "> 95%", "status": "Review"},
                {"name": "Pre-trend stability", "value": "Pass", "threshold": "Pass", "status": "Healthy"},
            ],
            "decision_snapshot": None,
            "rollout_plan": [],
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
        {
            "id": "customer-retention-offer",
            "name": "Premium Retention Offer",
            "description": "Compare retention offers for high-value customers at renewal.",
            "type": "Customer",
            "owner": "Lifecycle Strategy",
            "primary_outcome": "90-day retention",
            "status": "Ready to decide",
            "progress": 100,
            "readout_date": _day(-2),
            "action": "Review and decide",
            "projected_value": 730000,
            "allocation_health": 99,
            "alert": "Decision review is due",
            "guardrails": [
                {"name": "Offer cost", "value": "$42", "threshold": "< $50", "status": "Healthy"},
                {"name": "Complaint rate", "value": "0.7%", "threshold": "< 1%", "status": "Healthy"},
            ],
            "decision_snapshot": None,
            "rollout_plan": [],
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
        {
            "id": "customer-activation",
            "name": "Card Activation Journey",
            "description": "Test a simplified activation journey for newly booked accounts.",
            "type": "Customer",
            "owner": "Digital Product",
            "primary_outcome": "30-day activation rate",
            "status": "Planning",
            "progress": 18,
            "readout_date": _day(62),
            "action": "Complete test plan",
            "projected_value": 280000,
            "allocation_health": 100,
            "alert": "",
            "guardrails": [],
            "decision_snapshot": None,
            "rollout_plan": [],
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
        {
            "id": "ts-search-demand",
            "name": "Search Demand Campaign",
            "description": "Estimate demand generated by a nationwide paid-search campaign.",
            "type": "Time Series",
            "owner": "Acquisition Marketing",
            "primary_outcome": "Weekly applications",
            "status": "Running",
            "progress": 63,
            "readout_date": _day(31),
            "action": "Continue as planned",
            "projected_value": 360000,
            "allocation_health": 100,
            "alert": "",
            "guardrails": [
                {"name": "Data freshness", "value": "1 day", "threshold": "< 3 days", "status": "Healthy"},
            ],
            "decision_snapshot": None,
            "rollout_plan": [],
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
        {
            "id": "customer-dining-offer",
            "name": "Weekend Dining Offer",
            "description": "Scale the winning dining reward to the eligible portfolio.",
            "type": "Customer",
            "owner": "Rewards",
            "primary_outcome": "Incremental spend",
            "status": "Rolling out",
            "progress": 100,
            "readout_date": _day(-18),
            "action": "Scale to 50%",
            "projected_value": 520000,
            "allocation_health": 98,
            "alert": "",
            "guardrails": [
                {"name": "Reward cost", "value": "2.4%", "threshold": "< 3%", "status": "Healthy"},
            ],
            "decision_snapshot": {
                "decision": "Launch",
                "reviewer": "Portfolio Review",
                "rationale": "Primary outcome cleared the planned effect and all safety checks passed.",
                "created_at": _day(-17),
            },
            "rollout_plan": build_rollout_plan(date.today() - timedelta(days=10), [10, 25, 50, 100], 7),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
        {
            "id": "ts-brand-reengagement",
            "name": "Brand Re-engagement",
            "description": "Measure nationwide re-engagement messaging against a counterfactual.",
            "type": "Time Series",
            "owner": "Brand Marketing",
            "primary_outcome": "Weekly active accounts",
            "status": "Completed",
            "progress": 100,
            "readout_date": _day(-45),
            "action": "Completed",
            "projected_value": 470000,
            "allocation_health": 100,
            "alert": "",
            "guardrails": [],
            "decision_snapshot": {
                "decision": "Launch",
                "reviewer": "Marketing Analytics",
                "rationale": "Sustained positive lift with reliable model diagnostics.",
                "created_at": _day(-43),
            },
            "rollout_plan": [],
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
        {
            "id": "geo-southeast-expansion",
            "name": "Southeast Market Expansion",
            "description": "Plan a fixed-market pilot for a new acquisition offer.",
            "type": "Geography",
            "owner": "Regional Growth",
            "primary_outcome": "New accounts",
            "status": "Planning",
            "progress": 12,
            "readout_date": _day(84),
            "action": "Confirm market groups",
            "projected_value": 310000,
            "allocation_health": 100,
            "alert": "",
            "guardrails": [],
            "decision_snapshot": None,
            "rollout_plan": [],
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    ]


def load_experiments(path: Path | str = DEFAULT_PORTFOLIO_PATH) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        experiments = seed_experiments()
        save_experiments(experiments, target)
        return [normalize_experiment(item) for item in experiments]
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return [normalize_experiment(item) for item in seed_experiments()]
    records = payload if isinstance(payload, list) else seed_experiments()
    return [normalize_experiment(item) for item in records]


def save_experiments(experiments: list[dict[str, Any]], path: Path | str = DEFAULT_PORTFOLIO_PATH) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(json.dumps(experiments, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(target)


def upsert_experiment(experiment: dict[str, Any], path: Path | str = DEFAULT_PORTFOLIO_PATH) -> dict[str, Any]:
    experiments = load_experiments(path)
    record = normalize_experiment(experiment)
    record.setdefault("id", f"experiment-{uuid4().hex[:10]}")
    record["updated_at"] = datetime.now().isoformat(timespec="seconds")
    for index, existing in enumerate(experiments):
        if existing.get("id") == record["id"]:
            experiments[index] = {**existing, **record}
            break
    else:
        experiments.append(record)
    save_experiments(experiments, path)
    return record


def configuration_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_configuration(experiment: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    record = normalize_experiment(experiment)
    fingerprint = configuration_fingerprint(payload)
    previous = record.get("config_fingerprint", "")
    if previous and previous != fingerprint:
        record["config_version"] = int(record.get("config_version", 1)) + 1
    record["config_fingerprint"] = fingerprint
    record["config_payload"] = copy.deepcopy(payload)
    return record


def duplicate_experiment(experiment: dict[str, Any], name: str | None = None) -> dict[str, Any]:
    source = normalize_experiment(experiment)
    duplicate = copy.deepcopy(source)
    duplicate["id"] = f"experiment-{uuid4().hex[:10]}"
    duplicate["name"] = name or f'Copy of {source["name"]}'
    duplicate["status"] = "Planning"
    duplicate["progress"] = 0
    duplicate["action"] = "Review copied plan"
    duplicate["decision_snapshot"] = None
    duplicate["frozen_snapshots"] = []
    duplicate["rollout_plan"] = []
    duplicate["analysis_timestamp"] = ""
    duplicate["latest_result"] = {}
    duplicate["sample_collected"] = 0
    duplicate["alert"] = ""
    duplicate["config_version"] = 1
    duplicate["planned_launch_date"] = _day(14)
    duplicate["readout_date"] = _day(60)
    duplicate["updated_at"] = datetime.now().isoformat(timespec="seconds")
    return duplicate


def freeze_decision_snapshot(experiment: dict[str, Any], decision: str, reviewer: str, rationale: str) -> dict[str, Any]:
    if decision not in {"Launch", "Iterate", "Stop"}:
        raise ValueError("Decision must be Launch, Iterate, or Stop")
    record = normalize_experiment(experiment)
    created_at = datetime.now().isoformat(timespec="seconds")
    decision_record = {
        "decision": decision,
        "reviewer": reviewer or "Unassigned",
        "rationale": rationale or "No rationale recorded.",
        "created_at": created_at,
    }
    snapshot = {
        "snapshot_id": f"snapshot-{uuid4().hex[:10]}",
        "created_at": created_at,
        "config_version": int(record.get("config_version", 1)),
        "analysis_timestamp": record.get("analysis_timestamp", ""),
        "decision": copy.deepcopy(decision_record),
        "result": copy.deepcopy(record.get("latest_result", {})),
    }
    record["decision_snapshot"] = decision_record
    record["frozen_snapshots"] = [*record.get("frozen_snapshots", []), snapshot]
    record["action"] = {"Launch": "Prepare rollout", "Iterate": "Revise and retest", "Stop": "Close experiment"}[decision]
    record["status"] = "Rolling out" if decision == "Launch" else "Planning" if decision == "Iterate" else "Completed"
    return record


def pulse_summary(experiment: dict[str, Any], today: date | None = None) -> dict[str, Any]:
    record = normalize_experiment(experiment)
    collected = max(int(record.get("sample_collected", 0) or 0), 0)
    target = max(int(record.get("sample_target", 0) or 0), 0)
    traffic = max(float(record.get("traffic_per_period", 0) or 0), 0)
    progress = min(collected / target, 1.0) if target else float(record.get("progress", 0)) / 100
    remaining_periods = math.ceil(max(target - collected, 0) / traffic) if traffic > 0 else None
    anchor = today or date.today()
    frequency = str(record.get("traffic_frequency", "week")).lower()
    days_per_period = 30 if "month" in frequency else 1 if "day" in frequency else 7
    estimated_completion = (anchor + timedelta(days=days_per_period * remaining_periods)).isoformat() if remaining_periods is not None else record.get("readout_date", "")
    alerts = []
    allocation = record.get("allocation", [])
    max_drift = max((abs(float(row.get("observed_pct", 0)) - float(row.get("planned_pct", 0))) for row in allocation), default=0.0)
    allocation_p_value = None
    if allocation and collected > 0:
        observed_total = sum(max(float(row.get("observed_pct", 0)), 0) for row in allocation)
        planned_total = sum(max(float(row.get("planned_pct", 0)), 0) for row in allocation)
        if observed_total > 0 and planned_total > 0:
            observed_counts = [collected * max(float(row.get("observed_pct", 0)), 0) / observed_total for row in allocation]
            expected_counts = [collected * max(float(row.get("planned_pct", 0)), 0) / planned_total for row in allocation]
            if all(value > 0 for value in expected_counts):
                allocation_p_value = float(stats.chisquare(observed_counts, f_exp=expected_counts).pvalue)
    if max_drift > 5 or (allocation_p_value is not None and allocation_p_value < 0.01):
        alerts.append(f"Traffic allocation may not match the plan; largest difference is {max_drift:.1f} points.")
    missing = float(record.get("missing_data_rate", 0) or 0)
    if missing > 2:
        alerts.append(f"Missing data is {missing:.1f}%; check data delivery before interpreting the trend.")
    if record.get("alert"):
        alerts.append(str(record["alert"]))
    return {
        "sample_progress": progress,
        "sample_collected": collected,
        "sample_target": target,
        "traffic_per_period": traffic,
        "traffic_frequency": record.get("traffic_frequency", "week"),
        "estimated_completion": estimated_completion,
        "allocation": allocation,
        "allocation_p_value": allocation_p_value,
        "primary_direction": record.get("primary_direction", "No result yet"),
        "data_updated_at": record.get("data_updated_at", record.get("updated_at", "")),
        "alerts": list(dict.fromkeys(alerts)),
    }


def business_report_html(experiment: dict[str, Any]) -> str:
    record = normalize_experiment(experiment)
    pulse = pulse_summary(record)
    snapshot = record.get("decision_snapshot") or {}
    result = record.get("latest_result") or {}
    frozen_count = len(record.get("frozen_snapshots", []))
    guardrail_rows = "".join(
        f"<tr><td>{html.escape(str(row.get('name', '')))}</td><td>{html.escape(str(row.get('value', '')))}</td><td>{html.escape(str(row.get('status', '')))}</td></tr>"
        for row in record.get("guardrails", [])
    ) or '<tr><td colspan="3">No guardrails recorded</td></tr>'
    decision_copy = (
        f"<strong>{html.escape(str(snapshot.get('decision', 'No decision yet')))}</strong><br>"
        f"{html.escape(str(snapshot.get('rationale', '')))}"
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(record['name'])}</title>
<style>@page{{size:letter;margin:.55in}}body{{font-family:Arial,sans-serif;color:#232936;max-width:900px;margin:28px auto;line-height:1.35}}h1{{margin:0;font-size:28px}}h2{{font-size:16px;margin:22px 0 8px;border-bottom:1px solid #dfe4ec;padding-bottom:5px}}.sub{{color:#697386;margin:4px 0 20px}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid #dfe4ec}}.cell{{padding:10px;border-right:1px solid #dfe4ec}}.cell:last-child{{border:0}}.label{{font-size:10px;text-transform:uppercase;color:#7a8495}}.value{{font-size:14px;font-weight:700;margin-top:3px}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #dfe4ec;padding:7px;text-align:left;font-size:12px}}.note{{background:#f6f8fb;border-left:4px solid #ef4d4d;padding:10px;font-size:12px}}footer{{margin-top:24px;color:#7a8495;font-size:10px}}</style></head>
<body><h1>{html.escape(record['name'])}</h1><div class="sub">{html.escape(record.get('description', ''))}</div>
<div class="grid"><div class="cell"><div class="label">Owner</div><div class="value">{html.escape(record['owner'])}</div></div><div class="cell"><div class="label">Status</div><div class="value">{html.escape(record['status'])}</div></div><div class="cell"><div class="label">Launch</div><div class="value">{html.escape(record['planned_launch_date'])}</div></div><div class="cell"><div class="label">Readout</div><div class="value">{html.escape(record['readout_date'])}</div></div></div>
<h2>Experiment pulse</h2><p><strong>{pulse['sample_collected']:,} of {pulse['sample_target']:,}</strong> required observations collected. Current traffic: {pulse['traffic_per_period']:,.0f} / {html.escape(str(pulse['traffic_frequency']))}. Estimated completion: {html.escape(str(pulse['estimated_completion']))}.</p>
<p>Primary metric: <strong>{html.escape(record['primary_outcome'])}</strong> — {html.escape(str(pulse['primary_direction']))}.</p>
<div class="note">Running results are directional monitoring signals, not a final experiment decision.</div>
<h2>Latest result</h2><p>{html.escape(str(result.get('headline', 'No analysis result saved.')))}<br><span class="sub">{html.escape(str(result.get('confidence', '')))}</span></p>
<h2>Guardrails</h2><table><thead><tr><th>Guardrail</th><th>Current</th><th>Status</th></tr></thead><tbody>{guardrail_rows}</tbody></table>
<h2>Decision</h2><p>{decision_copy}</p><p>Reviewer: {html.escape(str(snapshot.get('reviewer', 'Not assigned')))} · Frozen snapshots: {frozen_count}</p>
<footer>Configuration v{record['config_version']} · Analysis time: {html.escape(str(record.get('analysis_timestamp') or 'Not analyzed'))} · Generated {datetime.now().isoformat(timespec='minutes')}</footer></body></html>"""


def portfolio_summary(experiments: list[dict[str, Any]]) -> dict[str, int | float]:
    return {
        "running": sum(item.get("status") in {"Running", "Collecting outcomes"} for item in experiments),
        "ready": sum(item.get("status") == "Ready to decide" for item in experiments),
        "attention": sum(bool(item.get("alert")) for item in experiments),
        "projected_value": float(sum(float(item.get("projected_value", 0) or 0) for item in experiments)),
    }


def build_rollout_plan(start: date | str, percentages: list[int], cadence_days: int) -> list[dict[str, Any]]:
    anchor = date.fromisoformat(start) if isinstance(start, str) else start
    return [
        {
            "stage": index,
            "percentage": int(percentage),
            "date": (anchor + timedelta(days=(index - 1) * int(cadence_days))).isoformat(),
            "status": "Planned",
        }
        for index, percentage in enumerate(percentages, start=1)
    ]


def value_timeline(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = sum(float(item.get("projected_value", 0) or 0) for item in experiments if item.get("status") == "Completed")
    rollout = sum(float(item.get("projected_value", 0) or 0) for item in experiments if item.get("status") == "Rolling out")
    active = sum(float(item.get("projected_value", 0) or 0) for item in experiments if item.get("status") not in {"Completed", "Rolling out"})
    today = date.today().replace(day=1)
    rows = []
    for offset in range(-4, 3):
        month_index = today.month - 1 + offset
        year = today.year + month_index // 12
        month = month_index % 12 + 1
        point_date = date(year, month, 1)
        realized_share = min(max((offset + 5) / 7, 0), 1)
        projected_share = min(max((offset + 5) / 7, 0.15), 1)
        rows.append(
            {
                "date": point_date.isoformat(),
                "realized": completed * realized_share + rollout * min(realized_share * 0.7, 1),
                "projected": completed + rollout + active * projected_share,
            }
        )
    return rows
