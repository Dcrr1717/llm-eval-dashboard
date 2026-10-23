"""LLM eval dashboard: the cost-vs-accuracy story behind a set of eval results.

Run with:  streamlit run app.py
Works offline with the bundled sample; the "Run eval" tab is the only part that
calls the Anthropic API, and only when explicitly confirmed.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

import charts
import metrics
from live_eval import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    EFFORT_LEVELS,
    LIVE_MODELS,
    estimate_cost,
    load_problems,
    run_eval,
)

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "sample_results.jsonl"
PROBLEMS_PATH = ROOT / "problems.jsonl"
LIVE_LOG_PATH = ROOT / "results" / "live_results.jsonl"
# The live-eval tab (the only part that touches the API) is hidden unless explicitly enabled,
# so a public deployment shows no API controls at all.
LIVE_EVAL_ENABLED = os.environ.get("ENABLE_LIVE_EVAL", "").strip().lower() in {"1", "true", "yes", "on"}

st.set_page_config(page_title="LLM Eval Dashboard", page_icon="📊", layout="wide")


# ---------------------------------------------------------------- data loading
@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return metrics.load_results(SAMPLE_PATH)


@st.cache_data(show_spinner=False)
def load_upload(data: bytes) -> pd.DataFrame:
    return metrics.load_results(io.BytesIO(data))


@st.cache_data(show_spinner=False)
def problem_bank() -> list[dict]:
    return load_problems(PROBLEMS_PATH)


def to_jsonl(df: pd.DataFrame) -> bytes:
    """Serialize rows back to the harness JSONL format (drops display-only columns)."""
    cols = [c for c in df.columns if c not in {"question", "source"}]
    records = df[cols].to_dict(orient="records")
    clean = [{k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in r.items()} for r in records]
    return "".join(json.dumps(r, default=str) + "\n" for r in clean).encode()


st.session_state.setdefault("live_rows", [])

with st.sidebar:
    st.header("Data")
    source = st.radio("Results source", ["Bundled sample", "Upload JSONL"], label_visibility="collapsed")
    base: pd.DataFrame | None = None
    if source == "Bundled sample":
        base = load_sample().assign(source="sample")
        st.caption(f"`sample_results.jsonl` — synthetic, {len(base)} calls.")
    else:
        upload = st.file_uploader("Upload resultados.jsonl", type=["jsonl", "json", "txt"])
        if upload is not None:
            try:
                base = load_upload(upload.getvalue()).assign(source="upload")
                st.caption(f"Loaded {len(base)} calls from `{upload.name}`.")
            except metrics.ResultsFormatError as exc:
                st.error(f"Could not read that file: {exc}")
        with st.expander("Expected format"):
            st.code(
                '{"id":"ALG1","topic":"algebra","difficulty":"easy","modelo":"claude-haiku-5-5",'
                '"correcto":true,"costo":0.0004,"tok_in":120,"tok_out":300,"seg":1.1}',
                language="json",
            )
            st.caption("Required: id, modelo, correcto, costo. Optional: topic, difficulty, tok_in, "
                       "tok_out, seg, esperado (expected), extraido (extracted).")

    live_rows = st.session_state["live_rows"]
    include_live = False
    if live_rows:
        include_live = st.checkbox(f"Include {len(live_rows)} live result(s) from this session", value=True)

frames = [base] if base is not None else []
if include_live:
    frames.append(metrics.records_to_frame(list(live_rows)).assign(source="live"))

st.title("LLM eval dashboard")
st.caption("Cost vs. accuracy across models — which model is worth paying for?")

if not frames:
    st.info("Upload a results file in the sidebar, or switch back to the bundled sample.")
    st.stop()

data = pd.concat(frames, ignore_index=True)

with st.sidebar:
    st.header("Filter")
    all_models = charts.model_order(data["modelo"].unique().tolist())
    chosen = st.multiselect("Models", all_models, default=all_models)
    log_x = st.toggle("Log-scale cost axis", value=True, help="Model prices span ~100x; log scale keeps all points readable.")
    st.download_button(
        "Download current data (JSONL)", to_jsonl(data), file_name="resultados.jsonl", mime="application/json"
    )

df = data[data["modelo"].isin(chosen)]

tab_names = ["📈 Dashboard", "🔎 Drill-down"]
if LIVE_EVAL_ENABLED:
    tab_names.append("⚡ Run eval (uses API)")
tabs = st.tabs(tab_names)
tab_dash, tab_drill = tabs[0], tabs[1]

# ------------------------------------------------------------------ dashboard
with tab_dash:
    if df.empty:
        st.warning("Select at least one model in the sidebar.")
    else:
        summary = metrics.model_summary(df)
        st.markdown(metrics.headline(summary))

        # KPI cards, one per model, cheapest first; wrap every 5 models.
        per_row = 5
        for start in range(0, len(summary), per_row):
            cols = st.columns(per_row)
            for col, (_, row) in zip(cols, summary.iloc[start : start + per_row].iterrows()):
                with col.container(border=True):
                    badge = " ⭐" if row["pareto"] else ""
                    st.markdown(f"**{row['modelo']}**{badge}")
                    st.metric("Accuracy", f"{row['accuracy']:.0%}", help=(
                        f"{row['correct']}/{row['n']} correct · 95% CI {row['acc_lo']:.0%}–{row['acc_hi']:.0%}"))
                    latency = "—" if pd.isna(row["mean_latency"]) else f"{row['mean_latency']:.1f}s"
                    st.markdown(
                        f"Total cost **\\${row['total_cost']:.4f}**  \n"
                        f"Cost / problem **\\${row['cost_per_problem']:.5f}**  \n"
                        f"Mean latency **{latency}**"
                    )
        st.caption("⭐ = Pareto-optimal: no other model is both cheaper and at least as accurate.")

        st.subheader("Cost vs. accuracy")
        st.plotly_chart(charts.pareto_chart(summary, log_x=log_x), width="stretch")
        st.caption("Error bars: 95% Wilson interval on accuracy. With small eval sets, overlapping "
                   "intervals mean the accuracy difference is not yet statistically clear.")

        left, right = st.columns(2)
        with left:
            st.subheader("Accuracy by difficulty")
            st.plotly_chart(charts.accuracy_bars(metrics.accuracy_by(df, "difficulty"), "difficulty"),
                            width="stretch")
        with right:
            st.subheader("Accuracy by topic")
            st.plotly_chart(charts.accuracy_bars(metrics.accuracy_by(df, "topic"), "topic"),
                            width="stretch")

        st.subheader("Cost breakdown")
        st.plotly_chart(charts.cost_breakdown_chart(metrics.cost_breakdown(df)), width="stretch")
        st.caption("Each call's recorded cost is split by its input/output token counts at list prices. "
                   "Output tokens include thinking, which is why they dominate.")

        with st.expander("Summary table"):
            st.dataframe(
                summary.drop(columns=["acc_lo", "acc_hi"]),
                hide_index=True,
                width="stretch",
                column_config={
                    "modelo": "Model",
                    "accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
                    "total_cost": st.column_config.NumberColumn("Total cost", format="$%.4f"),
                    "cost_per_problem": st.column_config.NumberColumn("Cost/problem", format="$%.5f"),
                    "mean_latency": st.column_config.NumberColumn("Mean latency (s)", format="%.1f"),
                    "pareto": "Pareto-optimal",
                },
            )

# ------------------------------------------------------------------ drill-down
with tab_drill:
    questions = {p["id"]: p["question"] for p in problem_bank()}
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1.4])
    models_f = f1.multiselect("Model", charts.model_order(df["modelo"].unique().tolist()), key="dd_model")
    topics_f = f2.multiselect("Topic", sorted(df["topic"].unique()), key="dd_topic")
    diffs_f = f3.multiselect("Difficulty", metrics.dimension_order(df["difficulty"], "difficulty"), key="dd_diff")
    outcome = f4.selectbox("Outcome", ["All", "Correct", "Incorrect"], key="dd_outcome")

    view = df.copy()
    if models_f:
        view = view[view["modelo"].isin(models_f)]
    if topics_f:
        view = view[view["topic"].isin(topics_f)]
    if diffs_f:
        view = view[view["difficulty"].isin(diffs_f)]
    if outcome != "All":
        view = view[view["correcto"] == (outcome == "Correct")]

    view = view.assign(
        result=view["correcto"].map({True: "✅ correct", False: "❌ incorrect"}),
        question=view["id"].map(questions),
    )
    st.caption(f"{len(view)} of {len(df)} calls · accuracy {view['correcto'].mean():.0%}" if len(view) else "No calls match.")
    st.dataframe(
        view[["id", "modelo", "topic", "difficulty", "question", "esperado", "extraido", "result",
              "costo", "tok_in", "tok_out", "seg"]],
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "id": "Problem",
            "modelo": "Model",
            "topic": "Topic",
            "difficulty": "Difficulty",
            "question": st.column_config.TextColumn("Question", width="medium"),
            "esperado": "Expected",
            "extraido": "Extracted",
            "result": "Result",
            "costo": st.column_config.NumberColumn("Cost", format="$%.5f"),
            "tok_in": "Tokens in",
            "tok_out": "Tokens out",
            "seg": st.column_config.NumberColumn("Latency (s)", format="%.1f"),
        },
    )
    st.download_button("Download filtered rows (JSONL)", to_jsonl(view.drop(columns=["result"])),
                       file_name="filtered_results.jsonl", mime="application/json")

    st.subheader("Hardest problems")
    hard = metrics.hardest_problems(df).assign(question=lambda d: d["id"].map(questions))
    st.dataframe(
        hard,
        hide_index=True,
        width="stretch",
        column_config={
            "id": "Problem",
            "topic": "Topic",
            "difficulty": "Difficulty",
            "accuracy": st.column_config.ProgressColumn("Accuracy across models", format="percent", min_value=0, max_value=1),
            "models": "Models",
            "question": st.column_config.TextColumn("Question", width="large"),
        },
    )

# ------------------------------------------------------------------ run eval
if LIVE_EVAL_ENABLED:
    with tabs[2]:
        st.warning(
            "**This tab calls the Anthropic API and spends real money** on the key you provide. "
            "Everything else in the dashboard works offline.",
            icon="💸",
        )
        env_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        session_key = st.text_input(
            "Anthropic API key (optional)",
            type="password",
            help="Used only for this browser session and never stored. Leave blank to use ANTHROPIC_API_KEY "
                 "from the environment.",
            placeholder="Using ANTHROPIC_API_KEY from the environment" if env_key else "sk-ant-...",
        )
        has_key = env_key or bool(session_key)

        problems = problem_bank()
        by_id = {p["id"]: p for p in problems}
        default_model = os.environ.get("EVAL_MODEL", DEFAULT_MODEL)
        c1, c2, c3 = st.columns([3, 1, 1])
        models_run = c1.multiselect(
            "Models", LIVE_MODELS, default=[default_model] if default_model in LIVE_MODELS else [LIVE_MODELS[0]]
        )
        effort = c2.selectbox("Effort", EFFORT_LEVELS, index=EFFORT_LEVELS.index("medium"),
                              help="Same effort for every model, so the comparison is like-for-like.")
        max_tokens = c3.number_input("Max tokens", min_value=1024, max_value=32000, value=DEFAULT_MAX_TOKENS, step=1024)
        picked = st.multiselect(
            "Problems",
            list(by_id),
            default=["ALG1", "CAL3", "PRB4"],
            format_func=lambda pid: f"{pid} · {by_id[pid]['difficulty']} · {by_id[pid]['question'][:60]}",
        )

        n_calls = len(models_run) * len(picked)
        typical, worst = estimate_cost(models_run, len(picked), int(max_tokens))
        st.markdown(f"**{n_calls} API call(s)** · typical ≈ **\\${typical:.3f}** · worst case ≤ \\${worst:.2f}")
        confirm = st.checkbox("I understand this will spend money on my Anthropic account.")

        if not has_key:
            st.info("Set `ANTHROPIC_API_KEY` (or paste a key above) to enable live runs.")
        run_clicked = st.button(
            "Run eval", type="primary", disabled=not (has_key and confirm and n_calls > 0)
        )

        if run_clicked:
            import anthropic  # imported lazily: only needed for live runs

            client = anthropic.Anthropic(api_key=session_key) if session_key else anthropic.Anthropic()
            bar = st.progress(0.0, text="Starting…")

            def on_progress(done: int, total: int, label: str) -> None:
                bar.progress(done / total, text=f"{done}/{total} · {label}")

            rows, errors = run_eval(client, [by_id[p] for p in picked], models_run, effort, int(max_tokens), on_progress)
            st.session_state["live_rows"].extend(rows)
            st.session_state["last_run"] = {"rows": rows, "errors": errors}
            try:  # best-effort local log; the filesystem is ephemeral on Streamlit Cloud
                LIVE_LOG_PATH.parent.mkdir(exist_ok=True)
                with open(LIVE_LOG_PATH, "a", encoding="utf-8") as fh:
                    fh.writelines(json.dumps(r) + "\n" for r in rows)
            except OSError:
                pass
            st.rerun()

        last = st.session_state.get("last_run")
        if last:
            rows, errors = last["rows"], last["errors"]
            if rows:
                spent = sum(r["costo"] for r in rows)
                acc = sum(r["correcto"] for r in rows) / len(rows)
                st.success(f"Last run: {len(rows)} call(s), accuracy {acc:.0%}, spent \\${spent:.4f}. "
                           "Results are added to the dashboard (see the sidebar toggle).")
                st.dataframe(pd.DataFrame(rows)[["id", "modelo", "esperado", "extraido", "correcto", "costo", "seg",
                                                  "stop_reason"]], hide_index=True, width="stretch")
            for err in errors:
                st.error(err)
