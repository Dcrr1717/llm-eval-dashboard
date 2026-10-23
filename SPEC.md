# Build spec — LLM eval dashboard (portfolio demo #4)

**This repo is a seed.** A Claude Code cloud session should read this SPEC and build the project
in this repo. Portfolio piece for César Daniel Reinoso (mathematician) to land AI-engineering
gigs (e.g., Mercor). It is the most visual piece: it turns raw eval results into a clear
cost-vs-accuracy story a hiring team can read in 10 seconds. Quality and clarity over size.

## First
Load the `claude-api` skill and follow it for correct Anthropic SDK usage and model IDs
(default `claude-opus-5-5`, configurable). Read `ANTHROPIC_API_KEY` from the environment; never
hard-code a key. All code comments and the README in English.

## What to build
A **Streamlit dashboard** that visualizes LLM evaluation results — primarily the cost vs.
accuracy trade-off across models — and can also trigger a fresh eval run.

### Data model
Ingest results in this JSONL shape (one call per line), the format produced by a sibling eval
harness; include a `sample_results.jsonl` with realistic synthetic data so the dashboard renders
with no API key:
```
{"id":"ALG1","topic":"algebra","difficulty":"easy","model":"claude-haiku-5-5","correct":true,"cost":0.0004,"tok_in":120,"tok_out":300,"latency_s":1.1}
```

### Dashboard (app.py, Streamlit)
- **Header KPIs**: per model — accuracy %, total cost, cost/problem, mean latency.
- **Cost vs. accuracy chart**: scatter/Pareto (x = cost, y = accuracy), one point per model, with
  labels; highlight the Pareto-optimal frontier. This is the headline visual.
- **Accuracy by difficulty** and **by topic**: grouped bar charts per model.
- **Cost breakdown**: stacked bar of input vs output token cost per model.
- **Drill-down table**: filter by model/topic/difficulty; show each problem, expected vs extracted
  answer, correct/incorrect, cost. 
- **Upload / pick results**: let the user upload a `results.jsonl` or use the bundled sample.
- Clean layout, sensible color use, works in light theme. Use plotly or altair for charts.

### Optional "Run eval" tab (uses the API)
A tab that, given an `ANTHROPIC_API_KEY`, runs a tiny eval (a few bundled problems x selected
models) live and appends to the results — so the dashboard can refresh with real numbers. Keep it
clearly separate and guarded (warn it spends money). Reuse a minimal symbolic grader.

### Deliverables
- `app.py` (Streamlit), `sample_results.jsonl`, a small `grader.py` + `problems.jsonl` for the
  live tab, `requirements.txt` (streamlit, plotly or altair, pandas, anthropic, sympy),
  `.gitignore` (ignore `.env`), `.env.example`.
- `README.md` (English): what it shows, why cost/accuracy matters for model selection, how to run
  (`streamlit run app.py`), how to deploy free on Streamlit Community Cloud, and a screenshot
  placeholder section.
- `tests/` with offline unit tests for the data-loading / Pareto / aggregation functions.

### Quality bar
- Runs out of the box with the bundled sample data and **no API key**.
- Clean, typed, commented Python; charts are readable and correctly computed (verify the Pareto
  logic with a unit test). No secrets committed.
- End the README with the exact run/deploy commands and confirm it is portfolio-ready.
