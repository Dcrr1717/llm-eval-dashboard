"""Streamlit dashboard for LLM evaluation results.

Turns raw eval results (cost, accuracy, latency, tokens per model call) into a
clear cost-vs-accuracy story a hiring team can read in ~10 seconds. The
headline visual is a cost-vs-accuracy Pareto scatter.

Runs fully offline on the bundled `sample_results.jsonl` with NO API key. An
optional, clearly guarded "Run eval" tab can call the Anthropic API to append
fresh numbers (it spends money).

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics
import live_eval

SAMPLE_PATH = "sample_results.jsonl"

# A readable, colorblind-friendly qualitative palette (used consistently for
# models across every chart).
PALETTE = px.colors.qualitative.Safe

st.set_page_config(
    page_title="LLM Eval Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load_sample() -> pd.DataFrame:
    """Load the bundled sample results (cached across reruns)."""
    return analytics.load_results(SAMPLE_PATH)


def _model_color_map(models: list[str]) -> dict[str, str]:
    """Stable model -> color mapping so colors agree across all charts."""
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(sorted(models))}


def _format_usd(value: float) -> str:
    """Format a dollar amount with sensible precision for small numbers."""
    if value < 0.01:
        return f"${value:.5f}"
    return f"${value:,.4f}"


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def pareto_chart(df: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    """Headline: cost-per-problem (x) vs accuracy (y), one point per model.

    Pareto-optimal models are drawn large and connected by a dashed frontier
    line; dominated models are drawn faded. Lower cost + higher accuracy is
    better, so the frontier hugs the top-left.
    """
    summary = analytics.per_model_summary(df)
    pareto = analytics.pareto_frontier(summary)

    fig = go.Figure()

    # Dashed frontier line through the Pareto-optimal points (sorted by cost).
    frontier = pareto[pareto["pareto"]].sort_values("cost_per_problem")
    fig.add_trace(
        go.Scatter(
            x=frontier["cost_per_problem"],
            y=frontier["accuracy"],
            mode="lines",
            line=dict(color="#888", width=2, dash="dash"),
            name="Pareto frontier",
            hoverinfo="skip",
        )
    )

    # One marker per model.
    for _, row in pareto.iterrows():
        is_opt = bool(row["pareto"])
        fig.add_trace(
            go.Scatter(
                x=[row["cost_per_problem"]],
                y=[row["accuracy"]],
                mode="markers+text",
                marker=dict(
                    size=22 if is_opt else 14,
                    color=colors.get(row["modelo"], "#444"),
                    opacity=1.0 if is_opt else 0.45,
                    line=dict(
                        width=2 if is_opt else 1,
                        color="#222" if is_opt else "#aaa",
                    ),
                    symbol="star" if is_opt else "circle",
                ),
                text=[row["modelo"]],
                textposition="top center",
                textfont=dict(size=12),
                name=row["modelo"],
                customdata=[[row["total_cost"], row["mean_latency"], is_opt]],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Cost/problem: $%{x:.5f}<br>"
                    "Accuracy: %{y:.1%}<br>"
                    "Total cost: $%{customdata[0]:.4f}<br>"
                    "Mean latency: %{customdata[1]:.2f}s<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Cost vs. accuracy - the Pareto frontier (stars are optimal)",
        xaxis_title="Cost per problem (USD, log scale)",
        yaxis_title="Accuracy",
        xaxis=dict(type="log"),
        yaxis=dict(tickformat=".0%"),
        template="plotly_white",
        height=520,
        margin=dict(t=60, r=30, b=50, l=60),
    )
    return fig


def accuracy_by_chart(df: pd.DataFrame, dimension: str, colors: dict) -> go.Figure:
    """Grouped bar chart: accuracy per model, broken down by `dimension`."""
    data = analytics.accuracy_by(df, dimension)
    fig = px.bar(
        data,
        x=dimension,
        y="accuracy",
        color="modelo",
        barmode="group",
        color_discrete_map=colors,
        labels={"accuracy": "Accuracy", dimension: dimension.capitalize()},
    )
    fig.update_layout(
        template="plotly_white",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        legend_title_text="Model",
        height=420,
        margin=dict(t=30, r=20, b=40, l=50),
    )
    return fig


def cost_breakdown_chart(df: pd.DataFrame) -> go.Figure:
    """Stacked bar: input-token cost vs output-token cost per model."""
    cb = analytics.cost_breakdown(df).sort_values("total_cost")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cb["modelo"],
            y=cb["input_cost"],
            name="Input tokens",
            marker_color="#4C78A8",
            hovertemplate="%{x}<br>Input: $%{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=cb["modelo"],
            y=cb["output_cost"],
            name="Output tokens",
            marker_color="#F58518",
            hovertemplate="%{x}<br>Output: $%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        yaxis_title="Total cost (USD)",
        xaxis_title="Model",
        legend_title_text="Token type",
        height=420,
        margin=dict(t=30, r=20, b=40, l=50),
    )
    return fig


# --------------------------------------------------------------------------- #
# Dashboard sections
# --------------------------------------------------------------------------- #
def render_kpis(df: pd.DataFrame) -> None:
    """Per-model KPI row: accuracy, total cost, cost/problem, mean latency."""
    summary = analytics.per_model_summary(df)
    st.subheader("Model KPIs")
    cols = st.columns(len(summary))
    for col, (_, row) in zip(cols, summary.iterrows()):
        with col:
            st.markdown(f"**{row['modelo']}**")
            st.metric("Accuracy", f"{row['accuracy']:.1%}")
            st.metric("Total cost", _format_usd(row["total_cost"]))
            st.metric("Cost / problem", _format_usd(row["cost_per_problem"]))
            st.metric("Mean latency", f"{row['mean_latency']:.2f}s")


def render_drilldown(df: pd.DataFrame) -> None:
    """Filterable per-problem table: expected vs extracted, correctness, cost."""
    st.subheader("Drill-down: per-problem results")
    c1, c2, c3 = st.columns(3)
    with c1:
        modelo = st.selectbox("Model", ["All"] + sorted(df["modelo"].unique()))
    with c2:
        topic = st.selectbox("Topic", ["All"] + sorted(df["topic"].unique()))
    with c3:
        difficulty = st.selectbox(
            "Difficulty", ["All"] + sorted(df["difficulty"].unique())
        )

    filtered = analytics.filter_results(
        df, modelo=modelo, topic=topic, difficulty=difficulty
    )

    display_cols = [
        "id",
        "topic",
        "difficulty",
        "modelo",
        "esperado",
        "extraido",
        "correcto",
        "costo",
        "seg",
    ]
    view = filtered[display_cols].rename(
        columns={
            "id": "ID",
            "topic": "Topic",
            "difficulty": "Difficulty",
            "modelo": "Model",
            "esperado": "Expected",
            "extraido": "Extracted",
            "correcto": "Correct",
            "costo": "Cost (USD)",
            "seg": "Latency (s)",
        }
    )
    if len(view):
        st.caption(
            f"{len(view)} rows - {int(filtered['correcto'].sum())} correct "
            f"({filtered['correcto'].mean():.1%} accuracy)"
        )
    else:
        st.caption("No rows match the current filters.")
    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Correct": st.column_config.CheckboxColumn(disabled=True),
            "Cost (USD)": st.column_config.NumberColumn(format="$%.5f"),
            "Latency (s)": st.column_config.NumberColumn(format="%.2f"),
        },
    )


def render_live_tab(active_df_key: str) -> None:
    """Optional, guarded tab that calls the Anthropic API to append results."""
    st.subheader("Run a live eval (optional)")
    st.warning(
        "This tab calls the Anthropic API and **spends money** - one request "
        "per selected model per problem. It is disabled unless an API key is "
        "available. The rest of the dashboard works fully offline.",
        icon="💸",
    )

    key_ok = live_eval.api_key_present()
    if key_ok:
        st.success("ANTHROPIC_API_KEY detected in the environment.", icon="✅")
    else:
        st.info(
            "No ANTHROPIC_API_KEY found. Set it in your environment or a .env "
            "file to enable live runs. See `.env.example`.",
            icon="🔑",
        )

    problems = live_eval.load_problems()
    st.caption(f"{len(problems)} bundled problems available for the live eval.")

    models = st.multiselect(
        "Models to run",
        live_eval.SELECTABLE_MODELS,
        default=["claude-haiku-5-5"],
    )
    model_override = st.text_input(
        "Model id override (optional)",
        value="",
        help=(
            "If set, every selected model is replaced by this exact id. "
            f"Default model: {live_eval.DEFAULT_MODEL}."
        ),
    ).strip()

    run = st.button(
        "Run live eval",
        type="primary",
        disabled=not key_ok or (not models and not model_override),
    )

    if run:
        progress = st.progress(0.0, text="Starting...")

        def _cb(fraction: float, label: str) -> None:
            progress.progress(min(fraction, 1.0), text=label)

        try:
            with st.spinner("Calling the Anthropic API..."):
                rows = live_eval.run_live_eval(
                    models=models,
                    problems=problems,
                    model_id_override=model_override or None,
                    progress_cb=_cb,
                )
        except Exception as exc:  # surface API/key errors in the UI
            st.error(f"Live eval failed: {exc}")
            return

        new_df = analytics.load_results(analytics.results_to_jsonl(pd.DataFrame(rows)))
        # Append to whatever results the dashboard is currently showing.
        current = st.session_state.get(active_df_key)
        combined = (
            pd.concat([current, new_df], ignore_index=True)
            if current is not None
            else new_df
        )
        st.session_state[active_df_key] = combined
        st.success(
            f"Appended {len(rows)} live results. Switch to the Dashboard tab to "
            "see the charts refresh."
        )
        st.dataframe(new_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download these results (JSONL)",
            data=analytics.results_to_jsonl(new_df),
            file_name="resultados_live.jsonl",
            mime="application/x-ndjson",
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("📊 LLM Eval Dashboard")
    st.caption(
        "Cost vs. accuracy across models - pick the cheapest model that still "
        "meets your quality bar."
    )

    # --- Sidebar: choose the data source --------------------------------- #
    st.sidebar.header("Data source")
    source = st.sidebar.radio(
        "Results to visualize",
        ["Bundled sample", "Upload JSONL"],
        help="The dashboard ships with realistic synthetic data so it renders "
        "with no API key.",
    )

    df_key = "active_df"

    if source == "Upload JSONL":
        upload = st.sidebar.file_uploader(
            "Upload a resultados.jsonl", type=["jsonl", "json", "txt"]
        )
        if upload is not None:
            try:
                st.session_state[df_key] = analytics.load_results(upload)
                st.sidebar.success("File loaded.")
            except analytics.ResultsFormatError as exc:
                st.sidebar.error(f"Could not load file: {exc}")
                st.session_state.pop(df_key, None)
        elif df_key not in st.session_state:
            st.sidebar.info("Upload a file, or the sample is shown below.")
    else:
        # Reset to the bundled sample when explicitly selected.
        st.session_state[df_key] = _load_sample()

    df = st.session_state.get(df_key)
    if df is None:
        df = _load_sample()

    colors = _model_color_map(list(df["modelo"].unique()))

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Rows:** {len(df)}  \n"
        f"**Models:** {df['modelo'].nunique()}  \n"
        f"**Problems:** {df['id'].nunique()}"
    )
    st.sidebar.caption(
        "Data model: one JSON object per line "
        "(id, topic, difficulty, modelo, correcto, costo, tok_in, tok_out, seg)."
    )

    tab_dash, tab_live = st.tabs(["Dashboard", "Run eval (live)"])

    with tab_dash:
        render_kpis(df)
        st.markdown("---")

        st.subheader("Headline: cost vs. accuracy")
        st.plotly_chart(
            pareto_chart(df, colors), use_container_width=True
        )

        st.markdown("---")
        left, right = st.columns(2)
        with left:
            st.subheader("Accuracy by difficulty")
            st.plotly_chart(
                accuracy_by_chart(df, "difficulty", colors),
                use_container_width=True,
            )
        with right:
            st.subheader("Accuracy by topic")
            st.plotly_chart(
                accuracy_by_chart(df, "topic", colors),
                use_container_width=True,
            )

        st.markdown("---")
        st.subheader("Cost breakdown: input vs output tokens")
        st.plotly_chart(cost_breakdown_chart(df), use_container_width=True)

        st.markdown("---")
        render_drilldown(df)

    with tab_live:
        render_live_tab(df_key)


if __name__ == "__main__":
    main()
