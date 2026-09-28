from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .historical_strategy import binned_historical_statistics, historical_arm_statistics, historical_curve_statistics


CONTROL_COLOR = "#2457a6"
TREATMENT_COLORS = ["#ef4d4d", "#d69a1f", "#16856b", "#8a4f9e"]


def customer_category_bar(
    df: pd.DataFrame,
    category_col: str,
    unit_col: str,
    category_label: str,
    title: str = "",
):
    """Show unique customer/account coverage across a business category."""
    if df.empty or category_col not in df.columns or unit_col not in df.columns:
        return go.Figure()
    data = df[[category_col, unit_col]].dropna().drop_duplicates().copy()
    counts = (
        data.groupby(category_col, dropna=False)[unit_col]
        .nunique()
        .sort_values(ascending=True)
        .reset_index(name="Customers / Accounts")
    )
    counts[category_col] = counts[category_col].astype(str)
    fig = go.Figure(
        go.Bar(
            x=counts["Customers / Accounts"],
            y=counts[category_col],
            orientation="h",
            marker_color=CONTROL_COLOR,
            text=counts["Customers / Accounts"].map(lambda value: f"{value:,.0f}"),
            textposition="auto",
            hovertemplate=f"{category_label}: %{{y}}<br>Customers / Accounts: %{{x:,.0f}}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=max(280, 46 * len(counts) + 90),
        margin=dict(l=20, r=45, t=35 if title else 20, b=35),
        title=title,
        xaxis_title="Customers / Accounts",
        yaxis_title=category_label,
        showlegend=False,
    )
    return fig


def customer_history_coverage(df: pd.DataFrame, date_col: str, unit_col: str):
    """Show when customers/accounts in the available history entered the portfolio."""
    if df.empty or date_col not in df.columns or unit_col not in df.columns:
        return go.Figure()
    data = df[[unit_col, date_col]].drop_duplicates(unit_col).copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])
    if data.empty:
        return go.Figure()
    data["Entry Month"] = data[date_col].dt.to_period("M").dt.to_timestamp()
    monthly = data.groupby("Entry Month")[unit_col].nunique().reset_index(name="New Customers / Accounts")
    fig = go.Figure(
        go.Scatter(
            x=monthly["Entry Month"],
            y=monthly["New Customers / Accounts"],
            mode="lines+markers",
            line=dict(color="#16856b", width=2),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(22,133,107,0.10)",
            hovertemplate="%{x|%b %Y}<br>New customers / accounts: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=20, r=20, t=20, b=35),
        xaxis_title="Customer Start Month",
        yaxis_title="New Customers / Accounts",
        showlegend=False,
    )
    return fig


def customer_group_summary_chart(summary: pd.DataFrame, value_col: str, value_label: str, color: str = CONTROL_COLOR):
    if summary.empty or "Group" not in summary.columns or value_col not in summary.columns:
        return go.Figure()
    data = summary[["Group", value_col]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce").fillna(0)
    data = data.sort_values(value_col, ascending=True)
    fig = go.Figure(
        go.Bar(
            x=data[value_col],
            y=data["Group"].astype(str),
            orientation="h",
            marker_color=color,
            text=data[value_col].map(lambda value: f"{value:,.0f}"),
            textposition="auto",
            hovertemplate=f"Audience: %{{y}}<br>{value_label}: %{{x:,.0f}}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=max(270, 48 * len(data) + 80),
        margin=dict(l=20, r=50, t=20, b=35),
        xaxis_title=value_label,
        yaxis_title="Audience",
        showlegend=False,
    )
    return fig


def customer_metric_distribution(df: pd.DataFrame, metric_col: str, metric_label: str, metric_type: str = "Continuous"):
    """Summarize the historical distribution of one analysis-ready outcome."""
    if df.empty or metric_col not in df.columns:
        return go.Figure()
    values = pd.to_numeric(_first_column(df, metric_col), errors="coerce").dropna()
    if values.empty:
        return go.Figure()
    if metric_type == "Binary":
        rate = float(values.mean())
        categories = pd.DataFrame(
            {
                "Result": ["No", "Yes"],
                "Share": [max(0.0, 1.0 - rate), min(1.0, rate)],
            }
        )
        fig = go.Figure(
            go.Bar(
                x=categories["Result"],
                y=categories["Share"],
                marker_color=["#cbd3df", CONTROL_COLOR],
                text=categories["Share"].map(lambda value: f"{value:.1%}"),
                textposition="outside",
                hovertemplate="%{x}: %{y:.1%}<extra></extra>",
            )
        )
        fig.update_yaxes(tickformat=".0%", range=[0, min(1.0, max(categories["Share"]) * 1.2)])
        y_title = "Share of Customers / Accounts"
    else:
        fig = go.Figure(
            go.Histogram(
                x=values,
                nbinsx=min(30, max(10, int(len(values) ** 0.5))),
                marker_color=CONTROL_COLOR,
                opacity=0.88,
                hovertemplate=f"{metric_label}: %{{x:,.2f}}<br>Customers / Accounts: %{{y:,.0f}}<extra></extra>",
            )
        )
        fig.add_vline(
            x=float(values.median()),
            line_dash="dash",
            line_color="#ef4d4d",
            annotation_text=f"Median {values.median():,.1f}",
            annotation_position="top right",
        )
        y_title = "Customers / Accounts"
    fig.update_layout(
        template="plotly_white",
        height=300,
        margin=dict(l=20, r=20, t=35, b=35),
        xaxis_title=metric_label,
        yaxis_title=y_title,
        showlegend=False,
    )
    return fig


def customer_strategy_outcome_chart(
    df: pd.DataFrame,
    strategy_col: str,
    metric_col: str,
    strategy_label: str,
    metric_label: str,
    metric_type: str = "Continuous",
):
    """Compare historical outcomes across named strategy categories."""
    if df.empty or strategy_col not in df.columns or metric_col not in df.columns:
        return go.Figure()
    data = pd.DataFrame(
        {
            "Strategy": _first_column(df, strategy_col).astype(str),
            "Outcome": pd.to_numeric(_first_column(df, metric_col), errors="coerce"),
        }
    ).dropna()
    if data.empty:
        return go.Figure()
    rows = []
    for strategy, scoped in data.groupby("Strategy", sort=False):
        values = scoped["Outcome"]
        count = int(values.size)
        mean = float(values.mean())
        standard_deviation = float((max(mean * (1 - mean), 0.0)) ** 0.5) if metric_type == "Binary" else float(values.std(ddof=1)) if count >= 2 else float("nan")
        margin = 1.96 * standard_deviation / (count**0.5) if count >= 2 and pd.notna(standard_deviation) else 0.0
        rows.append({"Strategy": strategy, "Outcome": mean, "Customers / Accounts": count, "Margin": margin})
    summary = pd.DataFrame(rows)
    colors = [CONTROL_COLOR, *[TREATMENT_COLORS[index % len(TREATMENT_COLORS)] for index in range(max(0, len(summary) - 1))]]
    fig = go.Figure(
        go.Bar(
            x=summary["Strategy"],
            y=summary["Outcome"],
            error_y=dict(type="data", array=summary["Margin"], color="#7891b3"),
            marker_color=colors[: len(summary)],
            customdata=summary[["Customers / Accounts"]],
            text=summary["Outcome"].map(lambda value: f"{value:.1%}" if metric_type == "Binary" else f"{value:,.2f}"),
            textposition="auto",
            hovertemplate=(
                f"{strategy_label}: %{{x}}<br>{metric_label}: "
                + ("%{y:.1%}" if metric_type == "Binary" else "%{y:,.2f}")
                + "<br>Customers / Accounts: %{customdata[0]:,.0f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=20, r=20, t=35, b=55),
        xaxis_title=strategy_label,
        yaxis_title=metric_label,
        showlegend=False,
    )
    if metric_type == "Binary":
        fig.update_yaxes(tickformat=".0%")
    return fig


def customer_option_chart(
    control: object,
    treatments: list[object],
    control_label: str,
    treatment_labels: list[str],
    value_label: str = "Test Option",
    numeric: bool = False,
):
    """Show business-as-usual and proposed test options with stable color semantics."""
    values = [control, *treatments]
    labels = [control_label, *treatment_labels]
    roles = ["Current / BAU", *["Test Option"] * len(treatments)]
    x_values = [float(value) for value in values] if numeric else list(range(len(values)))
    colors = [CONTROL_COLOR, *[TREATMENT_COLORS[index % len(TREATMENT_COLORS)] for index in range(len(treatments))]]
    fig = go.Figure()
    for index, (x_value, label, role, color) in enumerate(zip(x_values, labels, roles, colors)):
        display_value = f"{float(values[index]):,.0f}" if numeric else str(values[index])
        fig.add_trace(
            go.Scatter(
                x=[x_value],
                y=[0],
                mode="markers+text",
                name=role if index < 2 else None,
                showlegend=index < 2,
                marker=dict(size=22, color=color, line=dict(color="white", width=2)),
                text=[label],
                textposition="top center",
                hovertemplate=f"{role}<br>{value_label}: {display_value}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_white",
        height=250,
        margin=dict(l=30, r=30, t=45, b=45),
        xaxis_title=value_label,
        yaxis=dict(visible=False, range=[-0.5, 0.65]),
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="closest",
    )
    if not numeric:
        fig.update_xaxes(visible=False)
    return fig


def customer_required_accounts_chart(plan: pd.DataFrame):
    if plan.empty:
        return go.Figure()
    data = plan.copy()
    data["Audience and Option"] = data["Group"].astype(str) + " · " + data["Test Option"].astype(str)
    colors = data["Role"].map({"Control": CONTROL_COLOR, "Treatment": "#ef4d4d"}).fillna("#d69a1f")
    fig = go.Figure(
        go.Bar(
            x=data["Required Accounts"],
            y=data["Audience and Option"],
            orientation="h",
            marker_color=colors,
            text=data["Required Accounts"].map(lambda value: f"{value:,.0f}"),
            textposition="auto",
            customdata=data[["Role"]],
            hovertemplate="%{y}<br>Required accounts: %{x:,.0f}<br>Role: %{customdata[0]}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=max(300, 42 * len(data) + 100),
        margin=dict(l=20, r=55, t=30, b=35),
        xaxis_title="Required Accounts",
        yaxis_title="Audience · Test Option",
        showlegend=False,
    )
    return fig


def customer_traffic_allocation_chart(plan: pd.DataFrame):
    if plan.empty:
        return go.Figure()
    data = plan.copy()
    groups = data["Group"].astype(str).drop_duplicates().tolist()
    option_order = data[["Test Option", "Role"]].drop_duplicates().copy()
    option_order["_sort"] = option_order["Role"].ne("Control").astype(int)
    options = option_order.sort_values(["_sort", "Test Option"])["Test Option"].astype(str).tolist()
    fig = go.Figure()
    for index, option in enumerate(options):
        scoped = data[data["Test Option"].astype(str) == option].set_index(data[data["Test Option"].astype(str) == option]["Group"].astype(str))
        shares = [float(scoped.loc[group, "Traffic Allocation"]) if group in scoped.index else 0.0 for group in groups]
        color = CONTROL_COLOR if (scoped["Role"] == "Control").any() else TREATMENT_COLORS[(index - 1) % len(TREATMENT_COLORS)]
        fig.add_trace(
            go.Bar(
                x=shares,
                y=groups,
                name=option,
                orientation="h",
                marker_color=color,
                text=[f"{share:.0%}" if share >= 0.06 else "" for share in shares],
                textposition="inside",
                hovertemplate=f"{option}<br>Traffic share: %{{x:.1%}}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        height=max(300, 54 * len(groups) + 150),
        margin=dict(l=20, r=20, t=35, b=35),
        xaxis_title="Share of Eligible Traffic",
        yaxis_title="Audience",
        xaxis=dict(tickformat=".0%", range=[0, 1]),
        legend=dict(orientation="h", y=1.16, x=0),
    )
    return fig


def credit_line_histogram(df: pd.DataFrame, line_col: str):
    fig = px.histogram(df, x=line_col, nbins=28, labels={line_col: "Current Credit Line"}, template="plotly_white")
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), bargap=0.05)
    return fig


def historical_association(
    df: pd.DataFrame,
    line_col: str,
    metric_col: str,
    metric_type: str = "Continuous",
    strategy_points: list[float] | None = None,
    strategy_label: str = "Strategy Point",
    metric_label: str | None = None,
    binning_method: str | None = None,
    bin_width: float = 500.0,
    bin_count: int = 20,
    smooth: bool = False,
    strategy_point_labels: list[str] | None = None,
):
    if binning_method:
        data = binned_historical_statistics(df, line_col, metric_col, metric_type, binning_method, bin_width, bin_count)
    elif strategy_points:
        data = historical_arm_statistics(
            df,
            line_col,
            metric_col,
            strategy_points,
            grouping="Nearest proposed point",
            max_distance=0.0,
            metric_type=metric_type,
        )
    else:
        data = historical_curve_statistics(df, line_col, metric_col, metric_type)
    metric_label = metric_label or metric_col
    if data.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", height=340, margin=dict(l=20, r=20, t=30, b=20), xaxis_title=strategy_label, yaxis_title=metric_label)
        return fig
    fig = go.Figure(
        go.Scatter(
            x=data["Strategy Point"],
            y=data["Historical Mean"],
            mode="lines+markers",
            name="Historical mean",
            line=dict(color="#2457a6"),
            marker=dict(size=9),
            error_y=dict(
                type="data",
                symmetric=False,
                array=data["95% CI Upper"] - data["Historical Mean"],
                arrayminus=data["Historical Mean"] - data["95% CI Lower"],
                color="#7891b3",
            ),
            customdata=data[["Historical N", "Historical SD"]],
            hovertemplate=(
                f"{strategy_label}: %{{x:,.0f}}<br>{metric_label}: %{{y:,.2f}}"
                "<br>Historical customers: %{customdata[0]:,.0f}<br>Historical SD: %{customdata[1]:,.2f}<extra></extra>"
            ),
        )
    )
    bounds = pd.concat([data["95% CI Lower"], data["95% CI Upper"], data["Historical Mean"]]).dropna()
    y_min = float(bounds.min()) if not bounds.empty else 0.0
    y_max = float(bounds.max()) if not bounds.empty else 1.0
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    colors = ["#2457a6", "#c43b2f", "#d59a18", "#8a4f9e", "#087e8b"]
    for index, point in enumerate(strategy_points or []):
        label = strategy_point_labels[index] if strategy_point_labels and index < len(strategy_point_labels) else f"Assigned line: {float(point):,.0f}"
        fig.add_trace(
            go.Scatter(
                x=[float(point), float(point)],
                y=[y_min, y_max],
                mode="lines",
                name=label,
                line=dict(color=colors[index % len(colors)], dash="dot", width=2),
                hovertemplate=f"{label}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=35, b=20),
        xaxis_title=strategy_label,
        yaxis_title=metric_label,
        hovermode="x unified",
    )
    return fig


def _first_column(df: pd.DataFrame, column: str) -> pd.Series:
    selected = df[column]
    if isinstance(selected, pd.DataFrame):
        return selected.iloc[:, 0]
    return selected


def forest_plot(results: pd.DataFrame, arm_label: str = "Treatment Arm"):
    data = results[~results["Is Control"]].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["Effect vs Control"],
            y=data["Credit Line"].astype(str),
            mode="markers",
            error_x=dict(type="data", symmetric=False, array=data["CI Upper"] - data["Effect vs Control"], arrayminus=data["Effect vs Control"] - data["CI Lower"]),
            marker=dict(size=10, color=data["Significant"].map({True: "#1b6b4a", False: "#68707a"})),
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#5f6368")
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Effect vs Control", yaxis_title=arm_label)
    return fig


def response_plot(response: pd.DataFrame, arm_label: str = "Treatment Arm", outcome_label: str = "Observed Outcome"):
    fig = go.Figure()
    x_values = response["Credit Line"]
    numeric_x = pd.to_numeric(x_values, errors="coerce").notna().all()
    error = dict(type="data", symmetric=False, array=response["CI Upper"] - response["Estimate"], arrayminus=response["Estimate"] - response["CI Lower"])
    if numeric_x:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=response["Estimate"],
                mode="lines+markers",
                error_y=error,
                marker=dict(size=9, color=CONTROL_COLOR),
                line=dict(color=CONTROL_COLOR),
                hovertemplate=f"{arm_label}: %{{x}}<br>{outcome_label}: %{{y:,.2f}}<extra></extra>",
            )
        )
    else:
        colors = [CONTROL_COLOR, *[TREATMENT_COLORS[index % len(TREATMENT_COLORS)] for index in range(max(0, len(response) - 1))]]
        fig.add_trace(
            go.Bar(
                x=x_values.astype(str),
                y=response["Estimate"],
                error_y=error,
                marker_color=colors[: len(response)],
                text=response["Estimate"].map(lambda value: f"{value:,.2f}"),
                textposition="outside",
                hovertemplate=f"{arm_label}: %{{x}}<br>{outcome_label}: %{{y:,.2f}}<extra></extra>",
            )
        )
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=20, r=20, t=30, b=20), xaxis_title=arm_label, yaxis_title=outcome_label)
    return fig


def detectable_effect_curve(curve: pd.DataFrame, current_n: int):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["Sample Size per Arm"], y=curve["Minimum Detectable Effect"], mode="lines", line=dict(color="#2457a6")))
    fig.add_vline(x=current_n, line_dash="dash", line_color="#5f6368")
    fig.update_layout(template="plotly_white", height=280, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Sample Size per Arm", yaxis_title="Minimum Detectable Lift / Effect")
    return fig


def time_series_line(data: pd.DataFrame, intervention_date: str, outcome_label: str, planned_launch_date: str = "", recommended_end_date: str = ""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["_date"], y=data["_outcome"], mode="lines", name="Observed Outcome", line=dict(color="#2457a6")))
    if intervention_date:
        launch = pd.Timestamp(intervention_date)
        fig.add_shape(type="line", x0=launch, x1=launch, y0=0, y1=1, xref="x", yref="paper", line=dict(color="#b42318", dash="dash"))
        fig.add_annotation(x=launch, y=1, xref="x", yref="paper", text="Campaign Launch", showarrow=False, yanchor="bottom")
    planned = pd.Timestamp(planned_launch_date) if planned_launch_date else None
    recommended_end = pd.Timestamp(recommended_end_date) if recommended_end_date else None
    if planned is not None:
        fig.add_shape(type="line", x0=planned, x1=planned, y0=0, y1=1, xref="x", yref="paper", line=dict(color="#b42318", dash="dash"))
        fig.add_annotation(x=planned, y=1.02, xref="x", yref="paper", text="Planned<br>Launch", showarrow=False, yanchor="bottom", xanchor="right", align="right")
    if planned is not None and recommended_end is not None:
        fig.add_vrect(x0=planned, x1=recommended_end, fillcolor="#2457a6", opacity=0.08, line_width=0)
        fig.add_shape(type="line", x0=recommended_end, x1=recommended_end, y0=0, y1=1, xref="x", yref="paper", line=dict(color="#2457a6", dash="dot"))
        fig.add_annotation(x=recommended_end, y=1.02, xref="x", yref="paper", text="Earliest Reliable<br>Decision", showarrow=False, yanchor="bottom", xanchor="right", align="right")
        fig.add_annotation(x=planned + (recommended_end - planned) / 2, y=0.04, xref="x", yref="paper", text="Recommended<br>Campaign Window", showarrow=False, yanchor="bottom", align="center", font=dict(size=10))
    dates = pd.to_datetime(data["_date"])
    axis_end = max([dates.max(), planned, recommended_end], key=lambda value: value if value is not None else pd.Timestamp.min)
    if planned is not None and recommended_end is not None:
        window = max(recommended_end - planned, pd.Timedelta(days=1))
        axis_end = max(axis_end, recommended_end + window * 0.55)
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=20, r=20, t=55, b=20), xaxis_title="Date", yaxis_title=outcome_label)
    fig.update_xaxes(range=[dates.min(), axis_end])
    return fig


def timeseries_chart_context(data: pd.DataFrame, intervention_date: str, campaign_end_date: str = "") -> dict:
    dates = pd.to_datetime(data["_date"])
    return {"start": dates.min(), "end": dates.max(), "intervention": pd.Timestamp(intervention_date) if intervention_date else None, "campaign_end": pd.Timestamp(campaign_end_date) if campaign_end_date else None}


def _apply_shared_time_axis(fig, context: dict) -> None:
    start, end = context["start"], context["end"]
    fig.update_xaxes(range=[start, end], dtick="M6", tickformat="%b %Y", matches=None)


def _add_shared_intervention_context(fig, context: dict) -> None:
    launch = context.get("intervention")
    if launch is None:
        return
    line = dict(color="#b42318", dash="dash", width=2)
    fig.add_shape(type="line", x0=launch, x1=launch, y0=0, y1=1, xref="x", yref="paper", line=line)
    fig.add_annotation(x=launch, y=1, xref="x", yref="paper", text="Campaign Launch", showarrow=False, yanchor="bottom")
    end = context.get("campaign_end")
    if end is not None:
        fig.add_vrect(x0=launch, x1=end, fillcolor="#b42318", opacity=0.05, line_width=0, annotation_text="Campaign period", annotation_position="top left")


def pre_post_bar(pre_value: float, post_value: float, label: str):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Before Campaign", "After Campaign"], y=[pre_value, post_value], marker_color=["#68707a", "#2457a6"]))
    fig.update_layout(template="plotly_white", height=280, margin=dict(l=20, r=20, t=30, b=20), yaxis_title=label)
    return fig


def bsts_counterfactual_chart(data: pd.DataFrame, post_result: pd.DataFrame, intervention_date: str, outcome_label: str, context: dict | None = None):
    context = context or timeseries_chart_context(data, intervention_date)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["_date"], y=data["_outcome"], mode="lines", name="Actual", line=dict(color="#2457a6")))
    fig.add_trace(go.Scatter(x=post_result["_date"], y=post_result["upper"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=post_result["_date"], y=post_result["lower"], mode="lines", fill="tonexty", fillcolor="rgba(36,87,166,0.18)", line=dict(width=0), name="Likely Range"))
    fig.add_trace(go.Scatter(x=post_result["_date"], y=post_result["counterfactual"], mode="lines", name="Expected Without Campaign", line=dict(color="#8a3ffc", dash="dash")))
    _add_shared_intervention_context(fig, context)
    _apply_shared_time_axis(fig, context)
    fig.update_layout(template="plotly_white", height=390, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Date", yaxis_title=outcome_label, legend=dict(orientation="h", y=1.08, x=0))
    return fig


def impact_chart(post_result: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=post_result["_date"], y=post_result["impact"], mode="lines", name="Observed - Counterfactual", line=dict(color="#1b6b4a")))
    fig.add_hline(y=0, line_dash="dash", line_color="#5f6368")
    fig.update_layout(template="plotly_white", height=280, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Date", yaxis_title="Estimated Incremental Impact")
    return fig


def cumulative_impact_chart(post_result: pd.DataFrame, outcome_label: str = "Outcome", context: dict | None = None, final_value: float | None = None):
    data = post_result.sort_values("_date").copy()
    data["cumulative_impact"] = data["impact"].cumsum()
    final_value = final_value if final_value is not None else (float(data["cumulative_impact"].iloc[-1]) if not data.empty else 0.0)
    fig = go.Figure(go.Scatter(x=data["_date"], y=data["cumulative_impact"], mode="lines+markers", name="Cumulative Estimated Impact", line=dict(color="#1b6b4a"), hovertemplate="%{x|%b %-d, %Y}<br>Cumulative impact: %{y:,.0f}<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="#5f6368")
    if context:
        _add_shared_intervention_context(fig, context)
        _apply_shared_time_axis(fig, context)
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Date", yaxis_title=f"Cumulative Incremental {outcome_label}")
    if not data.empty:
        fig.add_annotation(x=data["_date"].iloc[-1], y=final_value, text=f"{final_value:+,.0f} total", showarrow=True, arrowhead=2)
    return fig


def geo_dma_map(markets: pd.DataFrame, title: str = "Geographic Markets"):
    colors = {
        "Available DMA": "#68707a",
        "Selected Candidate DMA": "#2457a6",
        "Excluded DMA": "#c9ced6",
        "Test": "#b42318",
        "Control": "#2457a6",
        "Not assigned": "#c9ced6",
    }
    data = markets.dropna(subset=["lat", "lon"]).copy()
    fig = px.scatter_geo(
        data,
        lat="lat",
        lon="lon",
        color="status",
        color_discrete_map=colors,
        hover_name="dma",
        hover_data={"status": True, "historical_outcome": ":,.0f", "lat": False, "lon": False},
        scope="usa",
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="#ffffff")))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=35, b=10), title=title, legend_title_text="")
    return fig


def geo_balance_chart(balance: pd.DataFrame, intervention_date: str, view_label: str = "Outcome"):
    fig = go.Figure()
    if not balance.empty:
        for group, color in [("Test", "#b42318"), ("Control", "#2457a6")]:
            if group in balance.columns:
                fig.add_trace(go.Scatter(x=balance["_date"], y=balance[group], mode="lines", name=group, line=dict(color=color)))
    if intervention_date:
        launch = pd.Timestamp(intervention_date)
        fig.add_shape(type="line", x0=launch, x1=launch, y0=0, y1=1, xref="x", yref="paper", line=dict(color="#b42318", dash="dash"))
        fig.add_annotation(x=launch, y=1, xref="x", yref="paper", text="Campaign Launch", showarrow=False, yanchor="bottom")
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Date", yaxis_title=view_label)
    return fig


def geo_trend_chart(trend: pd.DataFrame, intervention_date: str, view_label: str = "Outcome"):
    fig = go.Figure()
    for group, color in [("Test", "#b42318"), ("Control", "#2457a6")]:
        group_data = trend[trend["group"] == group]
        if not group_data.empty:
            fig.add_trace(go.Scatter(x=group_data["_date"], y=group_data["_display_outcome"], mode="lines", name=group, line=dict(color=color)))
    if intervention_date:
        launch = pd.Timestamp(intervention_date)
        fig.add_shape(type="line", x0=launch, x1=launch, y0=0, y1=1, xref="x", yref="paper", line=dict(color="#b42318", dash="dash"))
        fig.add_annotation(x=launch, y=1, xref="x", yref="paper", text="Campaign Launch", showarrow=False, yanchor="bottom")
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Date", yaxis_title=view_label)
    return fig
