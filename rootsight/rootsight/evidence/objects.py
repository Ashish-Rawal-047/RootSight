"""The evidence object.

One shape for every fact the system will ever show a user or send to a model.
If a number is not expressible as an Evidence object it cannot appear in a
narrative.

`confidence` is deliberately decomposed.  It is an ENGINEERING WEIGHT used for
retrieval priority and for the data-completeness dimension of the ranking.  It
is not a probability, and it is never the confidence in a causal claim - that
comes from the identification status, which lives on the hypothesis, not here.
Every component is exposed so a reader can see exactly why a number is 0.62 and
not 0.87.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Any

SUPPORT = "SUPPORTS"
CONTRADICT = "CONTRADICTS"
CONTEXT = "CONTEXT"
GAP = "GAP"

FRESHNESS_PENALTY = {"CURRENT": 1.0, "LAGGING": 0.85, "STALE": 0.6, "MISSING": 0.0}


@dataclass
class Evidence:
    evidence_id: str
    source_id: str
    source_type: str                 # OPERATIONAL_METRIC | KPI_MOVEMENT | DECOMPOSITION |
                                     # CAUSAL_ESTIMATE | TICKET_CLUSTER | EXTERNAL_SIGNAL |
                                     # DATA_GAP | DIAGNOSTIC | FINANCIAL_EXPOSURE
    entity: str                      # region / warehouse / product line / campaign
    metric: str
    value: Any
    unit: str
    timestamp: str                   # when the underlying data was last refreshed
    period: str                      # the business period the value describes
    grain: str
    freshness: dict                  # {status, age_hours, expected_cadence_hours}
    method: str                      # how the value was produced
    contribution: dict | None        # {kind, value, unit} - ARITHMETIC or CAUSAL or NONE
    confidence: float
    confidence_components: dict
    lineage: dict
    support_or_contradiction: str
    access_classification: str
    model_version: str
    hypothesis_ids: list[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    def redacted_for(self, denied_fields: set[str]) -> dict:
        d = self.as_dict()
        d.pop("lineage", None) if "lineage" in denied_fields else None
        return d


def evidence_id_for(*parts: Any) -> str:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8]
    return f"EV-{h}"


def compute_confidence(*, source_reliability: float, coverage_pct: float,
                       freshness_status: str, dq_defect_rate: float,
                       n_observations: int, n_reference: int = 60) -> tuple[float, dict]:
    """Composite retrieval weight with every factor visible.

    Each factor is in [0, 1] and multiplicative.  A reader can therefore see
    which single factor is responsible for a low weight, which is the whole
    point of not collapsing this into an opaque score.
    """
    cov = max(0.0, min(1.0, coverage_pct / 100.0))
    fresh = FRESHNESS_PENALTY.get(freshness_status, 0.5)
    dq = max(0.0, 1.0 - min(1.0, dq_defect_rate * 3.0))
    size = min(1.0, (n_observations / n_reference) ** 0.5) if n_reference else 1.0
    value = source_reliability * cov * fresh * dq * size
    components = {
        "source_reliability": round(source_reliability, 3),
        "coverage_factor": round(cov, 3),
        "freshness_factor": round(fresh, 3),
        "data_quality_factor": round(dq, 3),
        "sample_size_factor": round(size, 3),
        "formula": ("source_reliability x coverage x freshness x data_quality x "
                    "sqrt(min(1, n/n_reference))"),
        "is_probability": False,
        "meaning": ("retrieval and completeness weight for this single fact; it says "
                    "nothing about whether a causal claim is warranted"),
    }
    return round(min(1.0, value), 4), components
