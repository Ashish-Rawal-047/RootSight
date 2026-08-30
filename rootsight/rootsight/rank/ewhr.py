"""Evidence-Weighted Hypothesis Rank.

EWHR answers exactly one question: **which hypotheses deserve a human's
attention first?**

It is not a probability, not a confidence, not a posterior, and not a measure of
causal truth.  The V4 design used EWHR thresholds to assign the epistemic tier
(`ewhr > 0.60 -> LIKELY`).  V5 removes that path entirely:

    * the CAUSAL STATUS comes from identification alone (identify.py)
    * the DECISION PRIORITY comes from the materiality matrix (materiality.py)
    * EWHR only sorts the queue

`assert_no_status_derivation` is called by the trust-contract builder to make
that separation enforceable rather than aspirational: if a caller ever tries to
map an EWHR value onto a causal status, it raises.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

WEIGHTS = {
    "evidence_quality": 0.20,
    "temporal_compatibility": 0.20,
    "identification_quality": 0.20,
    "independent_corroboration": 0.15,
    "structural_support": 0.15,
    "data_completeness": 0.10,
}

IDENTIFICATION_SCORE = {
    "SUPPORTED_BY_INTERVENTION": 1.00,
    "SUPPORTED_BY_DESIGN": 0.90,
    "NOT_POINT_IDENTIFIED": 0.40,
    "ASSOCIATION_ONLY": 0.20,
    "INSUFFICIENT_EVIDENCE": 0.00,
}

STRUCTURE_SCORE = {"SUPPORTED": 1.0, "INCONCLUSIVE": 0.5,
                   "NOT_TESTABLE": 0.25, "CONTRADICTED": 0.0}


class EwhrMisuse(Exception):
    """Raised if EWHR is used to derive an epistemic or causal status."""


@dataclass
class EwhrResult:
    hypothesis_id: str
    dimensions: dict
    weights: dict
    score: float
    rank: int | None = None
    is_probability: bool = False
    meaning: str = ("attention-ranking score. Higher means 'look at this sooner', "
                    "not 'more likely to be true'.")
    caveats: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(self.score, 4)
        return d


def assert_no_status_derivation(source: str) -> None:
    raise EwhrMisuse(
        f"{source} attempted to derive a causal or epistemic status from EWHR. "
        "EWHR ranks attention only. Causal status comes from the identification "
        "pipeline; decision priority comes from the materiality matrix.")


def compute_ewhr(hypothesis: dict, retrieval: dict, *,
                 target_independent_sources: int = 3) -> EwhrResult:
    temporal = hypothesis.get("temporal") or {}
    corr = temporal.get("best_corr")
    temporal_pass = bool(temporal.get("passed"))
    t_score = 0.0
    if temporal_pass and corr is not None:
        t_score = float(min(1.0, abs(corr) / 0.7))
    elif corr is not None:
        t_score = float(min(0.35, abs(corr) / 2.0))

    supporting = retrieval.get("supporting", [])
    eq = float(np.mean([e["confidence"] for e in supporting])) if supporting else 0.0

    n_src = retrieval.get("n_independent_source_systems", 0)
    corrob = min(1.0, n_src / max(1, target_independent_sources))

    struct = STRUCTURE_SCORE.get(hypothesis.get("structural_support", "NOT_TESTABLE"), 0.25)
    ident = IDENTIFICATION_SCORE.get(hypothesis.get("causal_status", ""), 0.0)

    cov_vals = []
    for v in (hypothesis.get("missing", {}).get("coverage") or {}).values():
        if isinstance(v, dict) and "observed_pct" in v:
            cov_vals.append(v["observed_pct"] / 100.0)
    completeness = float(np.mean(cov_vals)) if cov_vals else 1.0
    gaps = retrieval.get("gaps", [])
    completeness *= max(0.4, 1.0 - 0.12 * len(gaps))

    dims = {
        "evidence_quality": {"value": round(eq, 4),
                             "basis": "mean confidence weight of supporting evidence"},
        "temporal_compatibility": {"value": round(t_score, 4),
                                   "basis": f"gate passed={temporal_pass}, |corr|={corr}"},
        "identification_quality": {"value": round(ident, 4),
                                   "basis": f"causal status {hypothesis.get('causal_status')}"},
        "independent_corroboration": {"value": round(corrob, 4),
                                      "basis": f"{n_src} independent source systems"},
        "structural_support": {"value": round(struct, 4),
                               "basis": f"structure screen {hypothesis.get('structural_support')}"},
        "data_completeness": {"value": round(min(1.0, completeness), 4),
                              "basis": f"mean column coverage, penalised by {len(gaps)} data gap(s)"},
    }
    score = sum(WEIGHTS[k] * dims[k]["value"] for k in WEIGHTS)
    caveats = [
        "EWHR is an ordering device. Two hypotheses with similar scores are not "
        "equally likely; they are equally worth reading.",
        "The weights are design weights set by the analytics owner. They are not "
        "empirically calibrated and are versioned in config.MODEL_VERSIONS.",
        "No threshold on this score grants permission to make a causal claim.",
    ]
    return EwhrResult(hypothesis_id=hypothesis["hypothesis_id"], dimensions=dims,
                      weights=dict(WEIGHTS), score=float(score), caveats=caveats)


def rank_all(results: list[EwhrResult]) -> list[EwhrResult]:
    ordered = sorted(results, key=lambda r: -r.score)
    for i, r in enumerate(ordered, start=1):
        r.rank = i
    return ordered
