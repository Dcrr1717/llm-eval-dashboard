"""Tiny live eval: send bundled problems to Claude models and grade the answers.

Produces rows in the same JSONL schema the dashboard reads, so live results can
be appended to the loaded data. Only used by the guarded "Run eval" tab; the
rest of the app never touches the network.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic

from grader import grade
from metrics import PRICES

DEFAULT_MODEL = "claude-opus-5-5"
# Current-generation models offered in the UI (all accept `output_config.effort`).
LIVE_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
EFFORT_LEVELS = ["low", "medium", "high"]
DEFAULT_MAX_TOKENS = 16000
# Rough per-call token use for a short math problem, for the pre-run estimate.
TYPICAL_TOKENS_IN, TYPICAL_TOKENS_OUT = 150, 1500

SYSTEM_PROMPT = (
    "You are solving a short math problem. Work it out, then end your reply with one line of "
    "the form 'ANSWER: <final answer>'. Write the final answer in plain math notation "
    "(for example 3/4, sqrt(2)/2, 9*pi, or '2, 3' when there are several values) with no units."
)


def load_problems(path: str | Path) -> list[dict]:
    """Read the bundled problem bank (id, topic, difficulty, question, answer)."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def call_cost(model: str, tok_in: int, tok_out: int) -> float:
    """USD cost of one call from token counts and list prices."""
    pin, pout = PRICES[model]
    return tok_in * pin / 1e6 + tok_out * pout / 1e6


def estimate_cost(models: list[str], n_problems: int, max_tokens: int = DEFAULT_MAX_TOKENS) -> tuple[float, float]:
    """(typical, worst-case) USD for running n_problems on each model.

    Worst case assumes every call uses the full `max_tokens` output budget.
    """
    typical = sum(call_cost(m, TYPICAL_TOKENS_IN, TYPICAL_TOKENS_OUT) for m in models) * n_problems
    worst = sum(call_cost(m, TYPICAL_TOKENS_IN, max_tokens) for m in models) * n_problems
    return typical, worst


def _response_text(response: Any) -> str:
    # content is a list of typed blocks (thinking, text, ...); keep only text.
    return "\n".join(block.text for block in response.content if block.type == "text")


def solve_one(
    client: anthropic.Anthropic,
    model: str,
    problem: dict,
    effort: str = "medium",
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Ask one model one problem and return a graded result row.

    Thinking is left at the models' adaptive default; `effort` is set
    explicitly so every model is compared at the same setting. Server-side
    refusal fallbacks are deliberately NOT enabled: a fallback would answer
    with a different model and mislabel the row. A refusal is recorded as an
    incorrect answer with its stop_reason instead.
    """
    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": problem["question"]}],
        output_config={"effort": effort},
    )
    elapsed = time.perf_counter() - start

    if response.stop_reason == "refusal":
        extracted, correct = None, False
    else:
        extracted, correct = grade(_response_text(response), problem["answer"])

    tok_in = response.usage.input_tokens
    tok_out = response.usage.output_tokens  # includes thinking tokens, which are billed as output
    return {
        "id": problem["id"],
        "topic": problem["topic"],
        "difficulty": problem["difficulty"],
        "model": model,
        "correct": correct,
        "cost": round(call_cost(model, tok_in, tok_out), 6),
        "tok_in": tok_in,
        "tok_out": tok_out,
        "latency_s": round(elapsed, 2),
        "expected": problem["answer"],
        "extracted": extracted,
        "stop_reason": response.stop_reason,
        "effort": effort,
        "source": "live",
    }


def run_eval(
    client: anthropic.Anthropic,
    problems: list[dict],
    models: list[str],
    effort: str = "medium",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[dict], list[str]]:
    """Run every (model, problem) pair sequentially.

    Returns (rows, errors). Transient per-call failures are collected and the
    run continues; an authentication/permission error aborts the run, since
    every later call would fail the same way.
    """
    rows: list[dict] = []
    errors: list[str] = []
    total = len(models) * len(problems)
    done = 0
    for model in models:
        for problem in problems:
            label = f"{model} · {problem['id']}"
            try:
                rows.append(solve_one(client, model, problem, effort, max_tokens))
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
                errors.append(f"{label}: authentication failed ({exc.status_code}); run aborted")
                return rows, errors
            except anthropic.NotFoundError:
                errors.append(f"{label}: model not available to this API key")
            except anthropic.RateLimitError:
                errors.append(f"{label}: rate limited after retries")
            except anthropic.APIStatusError as exc:
                errors.append(f"{label}: API error {exc.status_code}")
            except anthropic.APIConnectionError:
                errors.append(f"{label}: connection error")
            done += 1
            if on_progress:
                on_progress(done, total, label)
    return rows, errors
