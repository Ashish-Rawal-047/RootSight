"""Runtime telemetry: latency spans, model-call accounting, token usage, cost.

Design rule: telemetry is *measured*, never asserted.  Where a number cannot be
measured (token counts in deterministic-template mode) it is labelled
ESTIMATED and the estimator is named.  A judge must be able to tell the
difference between a measurement and a guess.
"""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any

from . import config


@dataclass
class Span:
    name: str
    layer: str                 # "NON_LLM" | "LLM" | "IO"
    started_ms: float
    duration_ms: float = 0.0
    meta: dict = field(default_factory=dict)


@dataclass
class ModelCall:
    call_id: str
    purpose: str               # narrative_render | clarification | extraction
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    token_source: str          # MEASURED | ESTIMATED
    cost_usd: float
    cost_inr: float
    stop_reason: str = ""
    retry_index: int = 0


class Telemetry:
    """One instance per analysis request."""

    def __init__(self, request_id: str | None = None):
        self.request_id = request_id or f"req-{uuid.uuid4().hex[:12]}"
        self.t0 = time.perf_counter()
        self.spans: list[Span] = []
        self.model_calls: list[ModelCall] = []
        self.counters: dict[str, int] = {}
        self.notes: list[str] = []
        self._stack: list[Span] = []

    # ------------------------------------------------------------------ spans
    @contextmanager
    def span(self, name: str, layer: str = "NON_LLM", **meta):
        sp = Span(name=name, layer=layer,
                  started_ms=(time.perf_counter() - self.t0) * 1000.0, meta=meta)
        self._stack.append(sp)
        t = time.perf_counter()
        try:
            yield sp
        finally:
            sp.duration_ms = (time.perf_counter() - t) * 1000.0
            self._stack.pop()
            self.spans.append(sp)

    def incr(self, key: str, by: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + by

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    # ------------------------------------------------------------ model calls
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Character/4 heuristic.  Named so its use is always visible."""
        return max(1, int(len(text) / 4))

    def record_model_call(self, *, purpose: str, provider: str, model: str,
                          prompt_tokens: int, completion_tokens: int,
                          latency_ms: float, token_source: str,
                          stop_reason: str = "", retry_index: int = 0) -> ModelCall:
        px = config.COSTS.models.get(model, {"in": 0.0, "out": 0.0})
        cost_usd = (prompt_tokens / 1e6) * px["in"] + (completion_tokens / 1e6) * px["out"]
        call = ModelCall(
            call_id=f"mc-{uuid.uuid4().hex[:8]}", purpose=purpose, provider=provider,
            model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            latency_ms=latency_ms, token_source=token_source,
            cost_usd=round(cost_usd, 8), cost_inr=round(cost_usd * config.COSTS.usd_to_inr, 6),
            stop_reason=stop_reason, retry_index=retry_index)
        self.model_calls.append(call)
        self.incr("model_calls")
        return call

    # ---------------------------------------------------------------- reports
    def total_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0

    def layer_split(self) -> dict:
        # only top-level, non-overlapping stage spans are summed
        stage = [s for s in self.spans if s.meta.get("stage")]
        by = {}
        for s in stage:
            by[s.layer] = by.get(s.layer, 0.0) + s.duration_ms
        tot = sum(by.values()) or 1.0
        return {"ms_by_layer": {k: round(v, 2) for k, v in by.items()},
                "pct_by_layer": {k: round(100 * v / tot, 1) for k, v in by.items()}}

    def report(self) -> dict:
        total = self.total_ms()
        tok_in = sum(c.prompt_tokens for c in self.model_calls)
        tok_out = sum(c.completion_tokens for c in self.model_calls)
        cost_usd = sum(c.cost_usd for c in self.model_calls)
        measured = all(c.token_source == "MEASURED" for c in self.model_calls) if self.model_calls else True
        stages = [{"stage": s.name, "layer": s.layer, "ms": round(s.duration_ms, 2),
                   **{k: v for k, v in s.meta.items() if k != "stage"}}
                  for s in self.spans if s.meta.get("stage")]
        stages.sort(key=lambda x: -x["ms"])
        return {
            "request_id": self.request_id,
            "latency": {
                "total_ms": round(total, 2),
                "stages_ms": stages,
                **self.layer_split(),
            },
            "model_calls": {
                "count": len(self.model_calls),
                "max_allowed_per_analysis": config.MAX_LLM_CALLS_PER_ANALYSIS,
                "calls": [asdict(c) for c in self.model_calls],
            },
            "tokens": {
                "prompt": tok_in, "completion": tok_out, "total": tok_in + tok_out,
                "source": "MEASURED" if measured else "ESTIMATED",
                "estimator": None if measured else "chars/4 heuristic (deterministic-render mode)",
            },
            "cost": {
                "usd_per_analysis": round(cost_usd, 6),
                "inr_per_analysis": round(cost_usd * config.COSTS.usd_to_inr, 4),
                "price_recorded_on": config.COSTS.price_recorded_on,
                "fx_recorded_on": config.COSTS.fx_recorded_on,
                "projection_50_analyses_per_day_usd_month": round(cost_usd * 50 * 30, 4),
            },
            "counters": dict(self.counters),
            "notes": list(self.notes),
            "model_versions": dict(config.MODEL_VERSIONS),
        }
