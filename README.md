# LLM Eval Dashboard

A Streamlit dashboard that turns raw LLM evaluation results into a clear
**cost-vs-accuracy** story across models, so you can pick the cheapest model
that still meets your quality bar.

It runs **fully offline on bundled synthetic data with no API key**, and has an
optional, clearly guarded tab that runs a tiny live eval against the Anthropic
API.

![Dashboard screenshot placeholder](docs/screenshot.png)

> **Screenshot:** replace `docs/screenshot.png` with a capture of the running
> dashboard (the headline Pareto chart makes the best thumbnail).

---

## What it shows

- **Model KPIs** - accuracy %, total cost, cost per problem, and mean latency
  for each model under test.
- **Cost vs. accuracy (the headline)** - a scatter/Pareto chart with one point
  per model. Cost per problem is on the x-axis (log scale), accuracy on the
  y-axis. The **Pareto-optimal** models (no other model is both cheaper *and*
  more accurate) are drawn as stars and connected by the frontier line;
  dominated models are faded.
- **Accuracy by difficulty** and **by topic** - grouped bar charts that show
  where each model is strong or weak.
- **Cost breakdown** - a stacked bar of input-token cost vs output-token cost
  per model (output tokens usually dominate the bill).
- **Drill-down table** - filter by model, topic, and difficulty; inspect each
  problem's expected vs extracted answer, correctness, cost, and latency.
- **Upload / pick results** - visualize the bundled sample or upload your own
  `resultados.jsonl`.

## Why cost vs. accuracy matters

More capable models cost more per token and are slower. The right model for a
task is rarely the most accurate one - it's the cheapest one that clears your
accuracy threshold. A Pareto chart makes that trade-off obvious: any model
*not* on the frontier is strictly worse than some alternative (someone else is
both cheaper and at least as accurate), so it should never be chosen. This
dashboard is built to make that call in about ten seconds.

## Data format

Results are JSONL - one model call per line - the format produced by a sibling
eval harness. Field names are in Spanish because that is the harness's
contract:

```json
{"id":"ALG1","topic":"algebra","difficulty":"easy","modelo":"claude-haiku-5-5","correcto":true,"costo":0.0004,"tok_in":120,"tok_out":300,"seg":1.1,"esperado":"4","extraido":"4"}
```

| Field | Meaning |
|-------|---------|
| `id` | Problem id |
| `topic` | Subject area (e.g. `algebra`, `calculus`) |
| `difficulty` | `easy` \| `medium` \| `hard` |
| `modelo` | Model id under test |
| `correcto` | Whether the extracted answer was correct |
| `costo` | Total USD cost of the call |
| `tok_in` / `tok_out` | Input / output tokens |
| `seg` | Latency in seconds |
| `esperado` / `extraido` | *(optional)* expected and extracted answers, shown in the drill-down |

`sample_results.jsonl` ships with 168 realistic synthetic rows (4 models x 42
problems) so everything renders immediately.

## Project layout

```
app.py                 # Streamlit dashboard (UI + charts)
analytics.py           # Pure data logic: load, aggregate, Pareto frontier
grader.py              # Symbolic answer grader (SymPy) for the live tab
live_eval.py           # Optional live eval runner (the only file that calls the API)
problems.jsonl         # Small problem set for the live tab
sample_results.jsonl   # Bundled synthetic results (no API key needed)
requirements.txt
.env.example           # Copy to .env to enable the live tab
tests/                 # Offline unit tests (Pareto logic, aggregation, grader)
```

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

The data logic - including the Pareto frontier - is covered by offline unit
tests that need no network and no API key:

```bash
pytest -q
```

## The optional live "Run eval" tab

The dashboard is fully usable without ever touching the API. The **Run eval**
tab is opt-in and **spends money**: it makes one Anthropic API call per selected
model per problem, grades each answer with the symbolic grader, and appends the
fresh results to the charts.

To enable it, provide a key via the environment (it is read from
`ANTHROPIC_API_KEY` and never hard-coded):

```bash
cp .env.example .env     # then edit .env and paste your key
# or export it directly:
export ANTHROPIC_API_KEY=sk-ant-...   # Windows: setx ANTHROPIC_API_KEY sk-ant-...
```

The default model is `claude-opus-5-5` and is configurable from the tab (model
multiselect plus an optional exact-id override).

## Deploy free on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to <https://share.streamlit.io>, sign in with GitHub, and click
   **New app**.
3. Select this repository, branch `main`, and main file `app.py`.
4. Click **Deploy**. It runs immediately on the bundled sample data - no secret
   needed.
5. *(Optional)* To enable the live tab in the cloud, add your key under
   **Settings -> Secrets** as `ANTHROPIC_API_KEY = "sk-ant-..."` (never commit
   it to the repo).

## Run

```bash
streamlit run app.py
```

This project is portfolio-ready: it runs out of the box on the bundled sample
data with no API key, the Pareto logic is unit-tested, and no secrets are
committed.
