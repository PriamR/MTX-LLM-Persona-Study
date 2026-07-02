"""Survey execution against Together (serverless, pay-per-token).

We do not free-generate and classify the answer. We read one token with
``logprobs``/``top_logprobs`` and softmax over just the option tokens, recovering
the model's latent P(recommend) instead of a single sampled vote — the fix for
the "temperature != sampling" trap.

Together returns logprobs in its own shape (``logprobs.top_logprobs`` is a list of
``{token: logprob}`` dicts, one per position), which the OpenAI SDK doesn't map —
so we call the endpoint directly and parse that shape.
"""
from __future__ import annotations

import math
import time

import httpx

from llmsonas.config import TOGETHER_API_KEY

URL = "https://api.together.xyz/v1/chat/completions"
_TRANSIENT = (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError)
# Serverless endpoints return these while overloaded / rate-limited; they are
# transient, so retry rather than aborting a whole multi-hundred-call run.
_RETRY_STATUS = {429, 500, 502, 503, 504}
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        if not TOGETHER_API_KEY:
            raise RuntimeError("TOGETHER_API_KEY is empty — add it to .env")
        _client = httpx.Client(
            headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"},
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
    return _client


# Which token means "recommend" is a run configuration, not a constant: the
# label-swap control (A=Not recommend / B=Recommend) and the Yes/No variant
# must be a different ``options``/``recommend`` pair, never a forked parser.
OPTIONS = ("A", "B")


def _other(options: tuple[str, str], recommend: str) -> str:
    if recommend not in options:
        raise ValueError(f"recommend={recommend!r} not in options {options}")
    return options[1] if options[0] == recommend else options[0]


def _option_logprobs(
    dist: dict[str, float], options: tuple[str, str] = OPTIONS
) -> dict[str, float]:
    """Raw logprobs of the option tokens found in one position's dist, keyed by
    the canonical option strings (token matching is whitespace/case-insensitive)."""
    wanted = {o.strip().upper(): o for o in options}
    lp: dict[str, float] = {}
    for token, logprob in dist.items():
        if logprob is None:
            continue
        label = token.strip().upper()
        if label in wanted and wanted[label] not in lp:
            lp[wanted[label]] = logprob
    return lp


def _p_from_dist(
    dist: dict[str, float],
    options: tuple[str, str] = OPTIONS,
    recommend: str = "A",
) -> float | None:
    """Softmax P(recommend) over the two option tokens in one position's dist."""
    lp = _option_logprobs(dist, options)
    if not lp:
        return None
    floor = min(lp.values()) - 10.0
    a = lp.get(recommend, floor)
    b = lp.get(_other(options, recommend), floor)
    ea, eb = math.exp(a), math.exp(b)
    return ea / (ea + eb)


def logit_gap(
    lp: dict[str, float],
    options: tuple[str, str] = OPTIONS,
    recommend: str = "A",
) -> float | None:
    """logprob(recommend) − logprob(the other option), same floor as ``_p_from_dist``.

    The per-persona signal a saturated P hides: at P≈0 every persona rounds to
    the same number, but the raw gap still shows whether the features moved the
    model at all (a −11 vs −19 nat gap is invisible after the softmax).
    """
    if not lp:
        return None
    floor = min(lp.values()) - 10.0
    return lp.get(recommend, floor) - lp.get(_other(options, recommend), floor)


def answer_probability(
    messages: list[dict],
    model: str,
    *,
    options: tuple[str, str] = OPTIONS,
    recommend: str = "A",
    top_logprobs: int = 5,
    retries: int = 5,
    detail: list[dict[str, float]] | None = None,
) -> float | None:
    """P(recommend) for one persona, or None if the option tokens never appear.

    ``detail``, when given, collects the raw option-token logprobs per call so a
    run can report the un-softmaxed logit gaps alongside the P values.
    """
    client = _get_client()
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    for attempt in range(retries):
        try:
            resp = client.post(URL, json=body)
            resp.raise_for_status()
            logprobs = resp.json()["choices"][0].get("logprobs") or {}
            positions = logprobs.get("top_logprobs") or []
            dist = positions[0] if positions else {}
            if detail is not None:
                detail.append(_option_logprobs(dist, options))
            return _p_from_dist(dist, options, recommend) if dist else None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _RETRY_STATUS or attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
        except _TRANSIENT:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return None


def completion_text(
    messages: list[dict],
    model: str,
    *,
    max_tokens: int = 8,
    retries: int = 5,
) -> str:
    """A short temperature-0 generation, for elicitations whose answer is not a
    single option token (e.g. a 0–100 frequency, which is multi-token — option
    logprobs don't apply). Same retry policy as the logprob path."""
    client = _get_client()
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    for attempt in range(retries):
        try:
            resp = client.post(URL, json=body)
            resp.raise_for_status()
            return (resp.json()["choices"][0]["message"]["content"] or "").strip()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _RETRY_STATUS or attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
        except _TRANSIENT:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return ""
