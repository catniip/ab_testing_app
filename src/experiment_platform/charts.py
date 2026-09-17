from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .historical_strategy import binned_historical_statistics, historical_arm_statistics, historical_curve_statistics


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
    colors = ["#b42318", "#c56a1a", "#8a4f9e", "#087e8b", "#68707a"]
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
    fig.add_trace(
        go.Scatter(
            x=response["Credit Line"],
            y=response["Estimate"],
            mode="lines+markers",
            error_y=dict(type="data", symmetric=False, array=response["CI Upper"] - response["Estimate"], arrayminus=response["Estimate"] - response["CI Lower"]),
            marker=dict(size=9, color="#2457a6"),
            line=dict(color="#2457a6"),
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
