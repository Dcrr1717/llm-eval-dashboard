"""Optional live eval runner for the "Run eval" tab.

This module is the ONLY place that calls the Anthropic API. It is imported by
the dashboard but never exercised unless the user explicitly starts a run from
the guarded tab. The Anthropic SDK is imported lazily inside the function so
the dashboard renders even if `anthropic` is missing or no key is set.

The API key is read from the environment (`ANTHROPIC_API_KEY`); it is never
hard-coded. Each problem is sent as a single, cheap request and graded with the
offline symbolic grader.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Iterable

import grader
from analytics import PRICING

# Default model, overridable from the UI. Per the claude-api skill the current
# default Opus model id is `claude-opus-5-5`.
DEFAULT_MODEL = "claude-opus-5-5"

# Models offered in the live tab (ordered cheap -> capable).
SELECTABLE_MODELS = [
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "claude-opus-5-5",
]

# System prompt steering the model to a terse, extractable final answer.
_SYSTEM_PROMPT = (
    "You are a careful math solver. Solve the problem, then end your reply "
    "with a line in exactly the form 'ANSWER: <result>' giving only the final "
    "result (a number or a simple expression), with no units or extra words."
)


def api_key_present() -> bool:
    """True if an Anthropic API key is available in the environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def load_problems(path: str = "problems.jsonl") -> list[dict]:
    """Load the bundled live-eval problems."""
    problems: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def _cost_usd(model: str, tok_in: int, tok_out: int) -> float:
    """Compute USD cost from token counts and published pricing."""
    rates = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return tok_in * rates["input"] / 1_000_000 + tok_out * rates["output"] / 1_000_000


def run_live_eval(
    models: Iterable[str],
    problems: list[dict],
    model_id_override: str | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> list[dict]:
    """Run a tiny live eval and return result rows in the dashboard's schema.

    WARNING: this spends money - it makes one API call per (model, problem).

    Args:
        models: Model ids to evaluate.
        problems: Problem dicts with keys id/topic/difficulty/pregunta/esperado.
        model_id_override: If set, replaces every selected model id (lets the
            user point the whole run at a single configurable model).
        progress_cb: Optional callback(fraction, label) for UI progress.

    Returns:
        A list of result rows matching the JSONL schema used by `analytics`.

    Raises:
        RuntimeError: If no API key is present.
    """
    if not api_key_present():
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Set it in your environment or a .env "
            "file before running a live eval."
        )

    # Lazy import so the dashboard works without the SDK installed.
    import anthropic

    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY from the env

    model_list = [model_id_override] if model_id_override else list(models)
    total = len(model_list) * len(problems)
    done = 0
    rows: list[dict] = []

    for model in model_list:
        for problem in problems:
            if progress_cb is not None:
                progress_cb(done / total, f"{model} - {problem['id']}")

            start = time.perf_counter()
            # Adaptive thinking per the claude-api skill default; modest
            # max_tokens because these are short answers.
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                thinking={"type": "adaptive"},
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": problem["pregunta"]}],
            )
            seconds = time.perf_counter() - start

            # Concatenate text blocks (skip thinking blocks) for grading.
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            correcto, extraido = grader.grade(problem["esperado"], text)

            tok_in = response.usage.input_tokens
            tok_out = response.usage.output_tokens
            rows.append(
                {
                    "id": problem["id"],
                    "topic": problem["topic"],
                    "difficulty": problem["difficulty"],
                    "modelo": model,
                    "correcto": bool(correcto),
                    "costo": round(_cost_usd(model, tok_in, tok_out), 6),
                    "tok_in": int(tok_in),
                    "tok_out": int(tok_out),
                    "seg": round(seconds, 2),
                    "esperado": problem["esperado"],
                    "extraido": extraido if extraido is not None else "",
                }
            )
            done += 1

    if progress_cb is not None:
        progress_cb(1.0, "done")
    return rows
