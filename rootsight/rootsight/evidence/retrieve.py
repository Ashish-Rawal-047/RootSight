"""Evidence assembly and contradiction-aware retrieval.

Three rules:

  1. Contradicting evidence is retrieved by construction, not by the model's
     goodwill.  For every hypothesis the retriever actively searches for facts
     that cut against it, and the Trust Contract makes disclosure mandatory when
     any are found.
  2. Absence is evidence.  A missing source, a stale source, an unavailable
     variable and insufficient history each become a DATA_GAP evidence object
     with the same shape as everything else, so they cannot be dropped on the
     way to the narrative.
  3. Access classification travels with the fact.  Filtering happens here,
     before analysis and long before any prompt is built.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from .. import config
from ..contracts.kpi_contract import registry
from ..detect.changepoint import Movement
from ..decompose.accounting import Decomposition
from ..ingest.dq import DataQualityReport
from ..security.policy import AccessDecision
from .objects import (CONTEXT, CONTRADICT, GAP, SUPPORT, Evidence,
                      compute_confidence, evidence_id_for)


class EvidenceBuilder:
    def __init__(self, dq: DataQualityReport, decision: AccessDecision):
        self.dq = dq
        self.decision = decision
        self.reg = registry()
        self.items: list[Evidence] = []

    # ------------------------------------------------------------------ utils
    def _freshness(self, source_id: str) -> dict:
        f = self.dq.freshness.get(source_id)
        if not f:
            return {"status": "UNKNOWN", "age_hours": None, "expected_cadence_hours": None}
        return {"status": f.status, "age_hours": round(f.age_hours, 2),
                "expected_cadence_hours": f.expected_cadence_hours,
                "last_refresh_at": f.last_refresh_at,
                "data_period_end": f.data_period_end,
                "coverage_pct": f.coverage_pct, "coverage_note": f.coverage_note}

    def _defect_rate(self, source_id: str) -> float:
        rates = [d.rate for d in self.dq.defects if d.source_id.startswith(source_id)]
        return float(max(rates)) if rates else 0.0

    def _conf(self, source_id: str, *, coverage_pct: float, n: int) -> tuple[float, dict]:
        src = self.reg.sources.get(source_id)
        rel = src.reliability_weight if src else 0.5
        fr = self.dq.freshness.get(source_id)
        return compute_confidence(
            source_reliability=rel, coverage_pct=coverage_pct,
            freshness_status=fr.status if fr else "UNKNOWN",
            dq_defect_rate=self._defect_rate(source_id), n_observations=n)

    def add(self, ev: Evidence) -> Evidence:
        cls = ev.access_classification
        if cls in ("PII", "PII_FREETEXT", "HR") and cls not in ("PUBLIC",):
            # never materialise restricted classes for a role not entitled to them
            if any(f in self.decision.denied_fields for f in ("customer_email", "employee_id")):
                ev.value = "[redacted by policy]"
                ev.note = (ev.note + " Field-level policy removed the underlying "
                                     "identifier before this object was created.").strip()
        self.items.append(ev)
        return ev

    # ------------------------------------------------------------- generators
    def from_movement(self, m: Movement, kpi_id: str, entity: str,
                      hypothesis_ids: list[str] | None = None) -> Evidence:
        k = self.reg.get(kpi_id)
        src = k.data_sources[0]
        conf, comp = self._conf(src, coverage_pct=100.0, n=m.periods_available)
        return self.add(Evidence(
            evidence_id=evidence_id_for("mov", kpi_id, entity, m.focus_window),
            source_id=src, source_type="KPI_MOVEMENT", entity=entity, metric=kpi_id,
            value=(None if m.pct_change is None or (isinstance(m.pct_change, float)
                                                    and np.isnan(m.pct_change))
                   else round(m.pct_change, 3)),
            unit="percent change vs baseline",
            timestamp=self._freshness(src).get("last_refresh_at", ""),
            period=f"{m.focus_window[0]}..{m.focus_window[1]} vs "
                   f"{m.baseline_window[0]}..{m.baseline_window[1]}",
            grain=self.decision.effective_grain, freshness=self._freshness(src),
            method=m.method, contribution=None, confidence=conf,
            confidence_components=comp,
            lineage={"kpi_contract": self.reg.lineage_card(kpi_id),
                     "changepoint_date": (m.changepoint_date.isoformat()
                                          if m.changepoint_date else None),
                     "significance": {"statistic": m.shift_statistic,
                                      "p_value": m.p_value,
                                      "test": "block permutation, block=7d"},
                     "calendar_adjustment": m.seasonal_adjustment,
                     "baseline_exclusions": m.calendar_exclusions},
            support_or_contradiction=CONTEXT,
            access_classification=self.reg.sources[src].access_classification,
            model_version=config.MODEL_VERSIONS["changepoint_detector"],
            hypothesis_ids=hypothesis_ids or [],
            note="; ".join(m.notes)))

    def from_decomposition(self, dec: Decomposition, entity: str) -> list[Evidence]:
        out = []
        src = self.reg.get(dec.kpi_id).data_sources[0]
        for t in dec.terms:
            conf, comp = self._conf(src, coverage_pct=100.0, n=60)
            out.append(self.add(Evidence(
                evidence_id=evidence_id_for("dec", dec.kpi_id, t.term_id, entity),
                source_id=src, source_type="DECOMPOSITION", entity=entity,
                metric=f"{dec.kpi_id}:{t.term_id}", value=round(t.value_pp, 3),
                unit="percentage points of baseline",
                timestamp=self._freshness(src).get("last_refresh_at", ""),
                period=f"{dec.focus_window[0]}..{dec.focus_window[1]}",
                grain=self.decision.effective_grain, freshness=self._freshness(src),
                method=dec.method,
                contribution={"kind": "ARITHMETIC_CONTRIBUTION",
                              "value": round(t.value_pp, 3),
                              "unit": "pp of baseline",
                              "is_causal": False,
                              "share_of_total_change_pct": round(t.share_of_change_pct, 2)},
                confidence=conf, confidence_components=comp,
                lineage={"identity": dec.method, "residual": dec.residual_abs,
                         "identity_closes": dec.identity_closes,
                         "cells_common": dec.cells_common,
                         "cells_entered": dec.cells_entered,
                         "per_cell_detail": t.detail},
                support_or_contradiction=CONTEXT,
                access_classification=self.reg.sources[src].access_classification,
                model_version=config.MODEL_VERSIONS["decomposition"],
                note=t.interpretation)))
        return out

    def from_operational(self, *, entity: str, metric: str, value: float, unit: str,
                         period: str, source_id: str, method: str,
                         hypothesis_ids: list[str], stance: str,
                         n: int, coverage_pct: float = 100.0,
                         extra_lineage: dict | None = None,
                         note: str = "") -> Evidence:
        conf, comp = self._conf(source_id, coverage_pct=coverage_pct, n=n)
        return self.add(Evidence(
            evidence_id=evidence_id_for("ops", metric, entity, period, stance),
            source_id=source_id, source_type="OPERATIONAL_METRIC", entity=entity,
            metric=metric, value=round(float(value), 4), unit=unit,
            timestamp=self._freshness(source_id).get("last_refresh_at", ""),
            period=period, grain=self.decision.effective_grain,
            freshness=self._freshness(source_id), method=method, contribution=None,
            confidence=conf, confidence_components=comp,
            lineage={"source": source_id, **(extra_lineage or {})},
            support_or_contradiction=stance,
            access_classification=self.reg.sources[source_id].access_classification,
            model_version=config.MODEL_VERSIONS["rootsight"],
            hypothesis_ids=hypothesis_ids, note=note))

    def from_causal_estimate(self, hypothesis: dict) -> Evidence | None:
        eff = hypothesis.get("effect", {})
        if eff.get("kind") != "POINT_WITH_CI":
            return None
        src = "SRC_OPS"
        conf, comp = self._conf(src, coverage_pct=100.0, n=hypothesis.get(
            "sufficiency", {}).get("checks", [{}])[0].get("value", 60) or 60)
        return self.add(Evidence(
            evidence_id=evidence_id_for("est", hypothesis["hypothesis_id"]),
            source_id=src, source_type="CAUSAL_ESTIMATE",
            entity=", ".join(hypothesis["scope"].get("regions", [])),
            metric=f"ATT({hypothesis['driver_id']} -> {hypothesis['outcome_kpi']})",
            value=eff["point_estimate"], unit=eff.get("unit", ""),
            timestamp=self._freshness(src).get("last_refresh_at", ""),
            period=str(hypothesis["scope"].get("start", "")) + ".." +
                   str(hypothesis["scope"].get("end", "")),
            grain=self.decision.effective_grain, freshness=self._freshness(src),
            method=eff.get("estimator", ""),
            contribution={"kind": "CAUSAL_EFFECT", "value": eff["point_estimate"],
                          "unit": eff.get("unit", ""), "is_causal": True,
                          "confidence_interval": eff.get("confidence_interval")},
            confidence=conf, confidence_components=comp,
            lineage={"estimand": eff.get("estimand"),
                     "fixed_effects": eff.get("fixed_effects"),
                     "inference": eff.get("p_value_primary_basis"),
                     "conditional_on": eff.get("conditional_on", []),
                     "robustness": hypothesis.get("robustness", {})},
            support_or_contradiction=SUPPORT,
            access_classification="OPERATIONAL",
            model_version=config.MODEL_VERSIONS["estimator_did"],
            hypothesis_ids=[hypothesis["hypothesis_id"]],
            note="Causal effect, conditional on the assumptions listed in `conditional_on`."))

    def data_gap(self, *, kind: str, detail: str, source_id: str,
                 hypothesis_ids: list[str], entity: str = "n/a",
                 metric: str = "coverage") -> Evidence:
        fr = self._freshness(source_id)
        return self.add(Evidence(
            evidence_id=evidence_id_for("gap", kind, source_id, entity, detail[:40]),
            source_id=source_id, source_type="DATA_GAP", entity=entity, metric=metric,
            value=kind, unit="gap", timestamp=fr.get("last_refresh_at", ""),
            period="analysis window", grain=self.decision.effective_grain,
            freshness=fr, method="ingestion coverage and freshness assessment",
            contribution=None, confidence=1.0,
            confidence_components={"note": "a gap is observed with certainty; the "
                                           "uncertainty it creates lives downstream",
                                   "is_probability": False},
            lineage={"source": source_id,
                     "known_coverage_gaps": (self.reg.sources[source_id].known_coverage_gaps
                                             if source_id in self.reg.sources else [])},
            support_or_contradiction=GAP,
            access_classification=(self.reg.sources[source_id].access_classification
                                   if source_id in self.reg.sources else "PUBLIC"),
            model_version=config.MODEL_VERSIONS["rootsight"],
            hypothesis_ids=hypothesis_ids, note=detail))

    # -------------------------------------------------------------- retrieval
    def retrieve_for(self, hypothesis_id: str, *, top_supporting: int = 4) -> dict:
        rel = [e for e in self.items if hypothesis_id in e.hypothesis_ids or not e.hypothesis_ids]
        support = sorted([e for e in rel if e.support_or_contradiction == SUPPORT],
                         key=lambda e: -e.confidence)
        contra = sorted([e for e in rel if e.support_or_contradiction == CONTRADICT],
                        key=lambda e: -e.confidence)
        gaps = [e for e in rel if e.support_or_contradiction == GAP]
        context = sorted([e for e in rel if e.support_or_contradiction == CONTEXT],
                         key=lambda e: -e.confidence)
        # independent corroboration: distinct source systems among supporting evidence
        source_types = {e.source_id for e in support}
        return {
            "supporting": [e.as_dict() for e in support[:top_supporting]],
            "contradicting": [e.as_dict() for e in contra],       # never truncated
            "gaps": [e.as_dict() for e in gaps],                  # never truncated
            "context": [e.as_dict() for e in context[:6]],
            "independent_source_systems": sorted(source_types),
            "n_independent_source_systems": len(source_types),
            "retrieval_policy": (
                "top-N by confidence for supporting evidence; ALL contradicting "
                "evidence and ALL data gaps, regardless of confidence or count"),
        }

    def all_as_dict(self) -> list[dict]:
        return [e.as_dict() for e in self.items]
