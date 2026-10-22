"""Plotly figure builders for the dashboard (Streamlit-free, so they can be smoke-tested)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from metrics import PRICES, dimension_order

# Categorical palette, assigned in this fixed order (validated for colour-vision
# deficiency on adjacent marks such as grouped and stacked bars).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER_COLOR = "#a3a29c"
FRONTIER_COLOR = PALETTE[0]
DOMINATED_COLOR = "#8a8984"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
SURFACE = "#ffffff"


def model_colors(models: list[str]) -> dict[str, str]:
    """Stable model -> colour map.

    Colour follows the model, never its rank: known models take slots in the
    fixed PRICES order, unknown ones follow alphabetically, so filtering the
    data never repaints the remaining series. Past 8 models, the rest share a
    neutral grey.
    """
    known = [m for m in PRICES if m in models]
    ordered = known + sorted(m for m in set(models) if m not in PRICES)
    return {m: PALETTE[i] if i < len(PALETTE) else OTHER_COLOR for i, m in enumerate(ordered)}


def model_order(models: list[str]) -> list[str]:
    """Legend/category order consistent with `model_colors`."""
    return list(model_colors(models))


def _base_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT_PRIMARY, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=TEXT_SECONDARY)),
        hoverlabel=dict(bgcolor="white", font_color=TEXT_PRIMARY),
        barcornerradius=4,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=TEXT_SECONDARY))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=TEXT_SECONDARY))
    return fig


def pareto_chart(summary: pd.DataFrame, log_x: bool = True) -> go.Figure:
    """Cost per problem (x) vs accuracy (y), one labelled point per model.

    Pareto-optimal models are drawn in blue and joined by the frontier
    staircase; dominated models are grey. Error bars are 95% Wilson intervals.
    """
    fig = go.Figure()
    frontier = summary[summary["pareto"]].sort_values("cost_per_problem")
    if len(frontier) > 1:
        fig.add_trace(
            go.Scatter(
                x=frontier["cost_per_problem"],
                y=frontier["accuracy"],
                mode="lines",
                line=dict(color=FRONTIER_COLOR, width=2, shape="hv", dash="dot"),
                name="Pareto frontier",
                hoverinfo="skip",
            )
        )
    for is_frontier, name, color in ((True, "Pareto-optimal", FRONTIER_COLOR), (False, "Dominated", DOMINATED_COLOR)):
        part = summary[summary["pareto"] == is_frontier]
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["cost_per_problem"],
                y=part["accuracy"],
                mode="markers+text",
                name=name,
                text=part["modelo"],
                # Dominated labels go below so models at similar cost do not collide.
                textposition="top center" if is_frontier else "bottom center",
                textfont=dict(color=TEXT_PRIMARY, size=12),
                marker=dict(
                    size=14,
                    color=color if is_frontier else SURFACE,
                    line=dict(color=color, width=2),
                    symbol="circle",
                ),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=part["acc_hi"] - part["accuracy"],
                    arrayminus=part["accuracy"] - part["acc_lo"],
                    color=GRID if not is_frontier else "#b7d3f6",
                    thickness=2,
                    width=0,
                ),
                customdata=part[["n", "correct", "acc_lo", "acc_hi", "total_cost", "mean_latency"]].to_numpy(),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Accuracy %{y:.1%} (%{customdata[1]}/%{customdata[0]})<br>"
                    "95% CI %{customdata[2]:.1%} – %{customdata[3]:.1%}<br>"
                    "Cost/problem $%{x:.5f}<br>"
                    "Total cost $%{customdata[4]:.4f}<br>"
                    "Mean latency %{customdata[5]:.1f}s<extra></extra>"
                ),
            )
        )
    _base_layout(fig, height=440)
    fig.update_xaxes(
        title_text="Cost per problem (USD" + (", log scale)" if log_x else ")"),
        type="log" if log_x else "linear",
        dtick=1 if log_x else None,  # label powers of ten only
        tickprefix="$",
    )
    lo = max(0.0, float(summary["acc_lo"].min()) - 0.05)
    fig.update_yaxes(title_text="Accuracy", tickformat=".0%", range=[lo, 1.04])
    return fig


def accuracy_bars(acc: pd.DataFrame, dimension: str) -> go.Figure:
    """Grouped bars: accuracy per value of `dimension`, one bar per model."""
    order = dimension_order(acc[dimension], dimension)
    colors = model_colors(acc["modelo"].unique().tolist())
    fig = go.Figure()
    for model in model_order(acc["modelo"].unique().tolist()):
        part = acc[acc["modelo"] == model].set_index(dimension).reindex(order).reset_index()
        fig.add_trace(
            go.Bar(
                x=part[dimension],
                y=part["accuracy"],
                name=model,
                marker=dict(color=colors[model], line=dict(color=SURFACE, width=2)),
                customdata=part[["correct", "n"]].to_numpy(),
                hovertemplate=f"<b>{model}</b><br>%{{x}}: %{{y:.0%}} (%{{customdata[0]}}/%{{customdata[1]}})<extra></extra>",
            )
        )
    _base_layout(fig)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_yaxes(title_text="Accuracy", tickformat=".0%", range=[0, 1.05])
    fig.update_xaxes(title_text=dimension.replace("_", " ").capitalize(), categoryorder="array", categoryarray=order)
    return fig


def cost_breakdown_chart(breakdown: pd.DataFrame) -> go.Figure:
    """Horizontal stacked bars: input-token vs output-token cost per model."""
    data = breakdown.assign(total=breakdown["cost_in"] + breakdown["cost_out"]).sort_values("total")
    fig = go.Figure()
    for col, name, color in (("cost_in", "Input tokens", PALETTE[0]), ("cost_out", "Output tokens (incl. thinking)", PALETTE[1])):
        share = (data[col] / data["total"].where(data["total"] > 0)).fillna(0)
        fig.add_trace(
            go.Bar(
                y=data["modelo"],
                x=data[col],
                name=name,
                orientation="h",
                marker=dict(color=color, line=dict(color=SURFACE, width=2)),
                customdata=share,
                hovertemplate=f"<b>%{{y}}</b><br>{name}: $%{{x:.4f}} (%{{customdata:.0%}})<extra></extra>",
            )
        )
    _base_layout(fig, height=max(260, 70 * len(data) + 80))
    fig.update_layout(barmode="stack", bargap=0.35, legend_traceorder="normal")
    fig.update_xaxes(title_text="Total cost (USD)", tickprefix="$")
    return fig
