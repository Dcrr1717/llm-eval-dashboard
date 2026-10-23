"""Generate `sample_results.jsonl`: realistic synthetic results for the dashboard.

Every problem in `problems.jsonl` is "answered" once by each model. Correctness
is drawn from per-model, per-difficulty skill levels; token counts follow each
model's typical verbosity (output includes thinking tokens); cost uses list
prices from `metrics.PRICES`; latency follows typical throughput. Seeded, so
re-running reproduces the bundled file exactly.

    python scripts/generate_sample.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics import PRICES  # noqa: E402

SEED = 7

# P(correct) by difficulty (easy, medium, hard).
SKILL = {
    "claude-opus-5-5": (0.99, 0.96, 0.86),
    "claude-opus-5": (0.98, 0.93, 0.80),
    "claude-sonnet-5-5": (0.98, 0.91, 0.72),
    "claude-sonnet-5": (0.96, 0.84, 0.58),
    "claude-haiku-5-5": (0.93, 0.74, 0.42),
}
# Mean output tokens (incl. thinking) by difficulty.
VERBOSITY = {
    "claude-opus-5-5": (380, 950, 2300),
    "claude-opus-5": (460, 1200, 2900),
    "claude-sonnet-5-5": (320, 820, 1900),
    "claude-sonnet-5": (360, 900, 2050),
    "claude-haiku-5-5": (260, 620, 1250),
}
# (fixed overhead seconds, output tokens per second)
SPEED = {
    "claude-opus-5-5": (1.1, 85),
    "claude-opus-5": (1.3, 70),
    "claude-sonnet-5-5": (0.8, 125),
    "claude-sonnet-5": (0.9, 110),
    "claude-haiku-5-5": (0.4, 240),
}
# A plausible wrong final answer for each problem (common slips).
WRONG = {
    "ALG1": "29/3", "ALG2": "x - 3", "ALG3": "-2, -3", "ALG4": "49", "ALG5": "0",
    "CAL1": "6", "CAL2": "e - 1", "CAL3": "1", "CAL4": "sqrt(pi)", "CAL5": "pi^2/8",
    "PRB1": "5/36", "PRB2": "1/2", "PRB3": "3", "PRB4": "1/169", "PRB5": "1/2",
    "NUM1": "21", "NUM2": "4", "NUM3": "12", "NUM4": "7", "NUM5": "53",
    "GEO1": "6*pi", "GEO2": "17", "GEO3": "108", "GEO4": "32*pi/3", "GEO5": "7",
}
DIFF_INDEX = {"easy": 0, "medium": 1, "hard": 2}


def main() -> None:
    rng = random.Random(SEED)
    problems = [json.loads(line) for line in (ROOT / "problems.jsonl").read_text().splitlines() if line.strip()]
    rows = []
    for model in SKILL:
        pin, pout = PRICES[model]
        overhead, tps = SPEED[model]
        for p in problems:
            d = DIFF_INDEX[p["difficulty"]]
            correct = rng.random() < SKILL[model][d]
            tok_in = 70 + len(p["question"]) // 4 + rng.randint(0, 12)
            tok_out = max(40, int(VERBOSITY[model][d] * rng.lognormvariate(0, 0.35)))
            if not correct:
                tok_out = int(tok_out * 1.25)  # wrong answers tend to come after longer struggles
            extracted: str | None = p["answer"] if correct else WRONG[p["id"]]
            if not correct and rng.random() < 0.12:
                extracted = None  # ran out of budget / no parseable final answer
            cost = tok_in * pin / 1e6 + tok_out * pout / 1e6
            latency = overhead + tok_out / tps * rng.uniform(0.85, 1.2)
            rows.append(
                {
                    "id": p["id"],
                    "topic": p["topic"],
                    "difficulty": p["difficulty"],
                    "model": model,
                    "correct": correct,
                    "cost": round(cost, 6),
                    "tok_in": tok_in,
                    "tok_out": tok_out,
                    "latency_s": round(latency, 2),
                    "expected": p["answer"],
                    "extracted": extracted,
                }
            )
    out = ROOT / "sample_results.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
