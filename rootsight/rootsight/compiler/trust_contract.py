"""The Trust Contract.

A machine-readable specification of what may be said about one analysis, for one
persona, built deterministically BEFORE any model is called, hashed, and stored
with the narrative.

Two properties make it more than a prompt:

  * the numbers a narrative may contain are enumerated in `allowed_numbers`.
    Anything else in the output is a hallucination by definition, and the
    validator can prove it mechanically.
  * the claim grammar is selected by CAUSAL STATUS, not by a score.  There is no
    path from an EWHR value to a permitted verb.

Persona affects breadth and framing (which evidence, which fields, how much
depth, which decision frame) and never the epistemic ceiling.  Two personas
looking at the same hypothesis get different narratives and the same permitted
strength of claim.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone

from .. import config
from ..security.policy import AccessDecision, Persona

# ---------------------------------------------------------------- claim grammar
UNIVERSALLY_FORBIDDEN = [
    "root cause", "proves", "proven", "proof that", "confirms that", "definitively",
    "conclusively", "guarantees", "undoubtedly", "certainly caused", "the reason is",
    "beyond doubt",
]

CLAIM_GRAMMAR = {
    "SUPPORTED_BY_DESIGN": {
        "allowed_claim_types": ["MEASURED_FACT", "ARITHMETIC_CONTRIBUTION",
                                "TEMPORAL_COMPATIBILITY", "CAUSAL_EFFECT_CONDITIONAL"],
        "forbidden_claim_types": ["CAUSAL_EFFECT_UNCONDITIONAL", "ROOT_CAUSE",
                                   "POINT_CLAIM_WITHOUT_INTERVAL"],
        "allowed_verbs": [
            "is estimated to have reduced", "is estimated to have increased",
            "is estimated to account for", "is attributable to",
            "under the stated assumptions, reduced",
            "under the stated assumptions, increased",
            "was associated with", "preceded", "contributed",
        ],
        "forbidden_verbs": ["caused", "drove", "resulted in", "is responsible for",
                             "explains", "was due to"],
        "quantification": "point estimate WITH its confidence interval and the "
                          "conditioning assumptions named in the same sentence",
        "mandatory_sections": ["measured", "estimate", "assumptions", "contradicting",
                                "gaps", "action"],
    },
    "SUPPORTED_BY_INTERVENTION": {
        "allowed_claim_types": ["MEASURED_FACT", "CAUSAL_EFFECT_CONDITIONAL",
                                "POST_INTERVENTION_CONSISTENCY"],
        "forbidden_claim_types": ["CAUSAL_EFFECT_UNCONDITIONAL", "ROOT_CAUSE"],
        "allowed_verbs": [
            "post-intervention evidence is consistent with",
            "the observed recovery is consistent with",
            "is estimated to have reduced", "is attributable to",
        ],
        "forbidden_verbs": ["proves", "caused", "confirms the cause"],
        "quantification": "point estimate with interval, plus the pre-intervention "
                          "prediction it is being compared against",
        "mandatory_sections": ["measured", "estimate", "prediction_vs_outcome",
                                "assumptions", "gaps", "action"],
    },
    "NOT_POINT_IDENTIFIED": {
        "allowed_claim_types": ["MEASURED_FACT", "ARITHMETIC_CONTRIBUTION",
                                "TEMPORAL_COMPATIBILITY", "HYPOTHESIS_RANKING",
                                "IDENTIFICATION_FAILURE"],
        "forbidden_claim_types": ["CAUSAL_EFFECT_CONDITIONAL",
                                   "CAUSAL_EFFECT_UNCONDITIONAL", "ROOT_CAUSE",
                                   "EFFECT_MAGNITUDE", "EFFECT_INTERVAL"],
        "allowed_verbs": [
            "is the leading hypothesis for", "cannot be ruled out as a contributor to",
            "was associated with", "co-moved with", "preceded",
            "remains unresolved as an explanation for",
        ],
        "forbidden_verbs": ["caused", "drove", "resulted in", "is responsible for",
                             "accounts for", "explains", "contributed",
                             "is estimated to have reduced",
                             "is estimated to have increased"],
        "quantification": "NO effect magnitude and NO interval of any kind. The "
                          "observed association may be reported and must be labelled "
                          "as an association.",
        "mandatory_sections": ["measured", "why_not_identified", "association",
                                "contradicting", "gaps", "action"],
    },
    "ASSOCIATION_ONLY": {
        "allowed_claim_types": ["MEASURED_FACT", "ARITHMETIC_CONTRIBUTION",
                                "ASSOCIATION", "IDENTIFICATION_FAILURE"],
        "forbidden_claim_types": ["CAUSAL_EFFECT_CONDITIONAL",
                                   "CAUSAL_EFFECT_UNCONDITIONAL", "ROOT_CAUSE",
                                   "TEMPORAL_COMPATIBILITY", "EFFECT_MAGNITUDE"],
        "allowed_verbs": ["was associated with", "co-moved with",
                           "moved at the same time as"],
        "forbidden_verbs": ["caused", "drove", "led to", "preceded", "resulted in",
                             "contributed", "accounts for", "explains",
                             "is the leading hypothesis for"],
        "quantification": "correlation coefficient only, explicitly labelled as an "
                          "association, with no lag claim",
        "mandatory_sections": ["measured", "association", "why_not_identified", "gaps"],
    },
    "INSUFFICIENT_EVIDENCE": {
        "allowed_claim_types": ["MEASURED_FACT", "ARITHMETIC_CONTRIBUTION",
                                "ABSTENTION", "DATA_REQUIREMENT",
                                "CLARIFICATION_REQUEST", "BENCHMARK_COMPARISON"],
        "forbidden_claim_types": ["ASSOCIATION", "TEMPORAL_COMPATIBILITY",
                                   "CAUSAL_EFFECT_CONDITIONAL",
                                   "CAUSAL_EFFECT_UNCONDITIONAL", "ROOT_CAUSE",
                                   "HYPOTHESIS_RANKING", "EFFECT_MAGNITUDE"],
        "allowed_verbs": ["cannot be assessed", "is not estimable from",
                           "would require", "changed by", "was recorded as"],
        "forbidden_verbs": ["caused", "drove", "suggests", "indicates", "implies",
                             "points to", "is likely", "appears to be driven by",
                             "was associated with"],
        "quantification": "descriptive values only; no association measure, no effect",
        "mandatory_sections": ["measured", "abstention", "what_is_missing",
                                "what_would_resolve_it"],
    },
}

DISPLAY_LABEL = {
    "SUPPORTED_BY_DESIGN": "EXPLAINED (design-supported)",
    "SUPPORTED_BY_INTERVENTION": "VALIDATED BY INTERVENTION",
    "NOT_POINT_IDENTIFIED": "LEADING HYPOTHESIS (not point-identified)",
    "ASSOCIATION_ONLY": "ASSOCIATION ONLY",
    "INSUFFICIENT_EVIDENCE": "INSUFFICIENT EVIDENCE",
}


@dataclass
class TrustContract:
    claim_id: str
    hypothesis_id: str
    kpi_id: str                       # the KPI whose movement is under investigation
    hypothesis_outcome_kpi: str       # the KPI the driver acts on (may differ)
    scenario_id: str
    persona: str
    persona_profile: str
    decision_frame: str
    access_scope: dict
    evidence_ids: list[str]
    contradictory_evidence_ids: list[str]
    gap_evidence_ids: list[str]
    entity_ids: list[str]
    causal_status: str
    epistemic_status: str
    display_label: str
    estimand: str | None
    estimator: str | None
    effect_estimate: dict
    uncertainty: dict
    assumptions: list[dict]
    missing_data: list[str]
    missing_evidence: list[str]
    allowed_claim_types: list[str]
    forbidden_claim_types: list[str]
    allowed_numbers: list[dict]
    allowed_entities: list[str]
    allowed_fields: list[str]
    allowed_verbs: list[str]
    forbidden_verbs: list[str]
    quantification_rule: str
    mandatory_sections: list[str]
    materiality: dict
    ewhr: dict
    lineage: dict
    model_versions: dict
    graph_snapshot: dict
    max_words: int
    built_at: str = field(default_factory=lambda:
                          datetime.now(timezone.utc).isoformat(timespec="seconds"))
    contract_hash: str = ""
    llm_may_decide: list[str] = field(default_factory=lambda: [
        "sentence order within a section", "connective phrasing", "word choice from "
        "the allowed verb list", "how to summarise a list the contract already fixed"])
    llm_may_not_decide: list[str] = field(default_factory=lambda: [
        "the causal status", "any numeric value", "which evidence is included",
        "whether a section may be omitted", "the recommended action",
        "the confidence or its interpretation", "any entity not in allowed_entities"])

    def as_dict(self) -> dict:
        return asdict(self)

    def finalise(self) -> "TrustContract":
        payload = json.dumps({k: v for k, v in self.as_dict().items()
                              if k not in ("contract_hash", "built_at")},
                             sort_keys=True, default=str)
        self.contract_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return self

    # -------------------------------------------------------------- accessors
    def number_whitelist(self) -> list[float]:
        return [n["value"] for n in self.allowed_numbers]

    def is_number_allowed(self, x: float) -> tuple[bool, str | None]:
        for n in self.allowed_numbers:
            if abs(x - n["value"]) <= n.get("tolerance", 0.05):
                return True, n["label"]
        return False, None


def _num(label: str, value, unit: str = "", tolerance: float = 0.05,
         source: str = "") -> dict | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:                                          # NaN
        return None
    return {"label": label, "value": round(v, 6), "unit": unit,
            "tolerance": tolerance, "source": source}


_SKIP_KEYS = {"tolerance", "weights", "alpha", "max_words", "plan_version"}
_STRING_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _collect_numbers(obj, prefix: str, out: dict, depth: int = 0) -> None:
    """Every numeric value the deterministic engine produced, by JSON path.

    Whitelist policy, stated plainly because it is a trade-off: the contract
    admits every number the engine computed for this analysis, plus the lakh and
    crore renderings of monetary values, because those are the forms the prose
    actually prints.  The list is therefore large.  What it still excludes is the
    thing that matters: a value the engine never produced.  A model that invents
    "the effect was -4.8 percentage points" fails, because -4.8 appears nowhere
    in the analysis.  Combined with V5 (locked sentences must survive verbatim),
    every number that carries a claim is pinned twice.
    """
    if depth > 8:
        return
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        v = float(obj)
        if v != v or v in (float("inf"), float("-inf")):
            return
        for scale, suffix in ((1.0, ""), (1e-5, "_lakh"), (1e-7, "_crore")):
            sv = v * scale
            if abs(sv) < 1e-9 and scale != 1.0:
                continue
            # both signs: prose writes "-Rs 5.45 lakh", where the minus binds to
            # the currency symbol and the numeral scans as positive
            for signed in (round(sv, 4), round(abs(sv), 4)):
                if signed not in out:
                    out[signed] = prefix + suffix
        return
    if isinstance(obj, str):
        # engine-produced strings (evidence notes, method descriptions, defect
        # details) legitimately contain numbers that the prose may quote
        for tok in _STRING_NUMBER_RE.findall(obj):
            try:
                v = float(tok.replace(",", ""))
            except ValueError:
                continue
            for signed in (round(v, 4), round(abs(v), 4)):
                if signed not in out:
                    out[signed] = prefix + "_in_text"
        return
    if isinstance(obj, dict):
        for k, val in obj.items():
            if k in _SKIP_KEYS:
                continue
            _collect_numbers(val, f"{prefix}.{k}", out, depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            _collect_numbers(val, f"{prefix}[{i}]", out, depth + 1)


def _sources_for(hypothesis: dict, movement: dict) -> set[str]:
    """Source systems the analysis actually read, which the narrative may cite."""
    from ..contracts.kpi_contract import registry
    reg = registry()
    out: set[str] = set()
    kpi = hypothesis.get("outcome_kpi")
    if kpi in reg.kpis:
        out.update(reg.get(kpi).data_sources)
    for d in reg.get(kpi).drivers if kpi in reg.kpis else []:
        if d.source in reg.sources:
            out.add(d.source)
    return out


class TrustContractBuilder:
    """Deterministic.  No model is consulted anywhere in this class."""

    def build(self, *, scenario_id: str, hypothesis: dict, retrieval: dict,
              persona: Persona, decision: AccessDecision, movement: dict,
              decomposition: dict | None, materiality: dict, ewhr: dict,
              lineage: dict, graph_snapshot: dict,
              movement_kpi_id: str | None = None,
              clarification: dict | None = None,
              recommendations: list[dict] | None = None,
              benchmark: dict | None = None) -> TrustContract:
        status = hypothesis["causal_status"]
        grammar = CLAIM_GRAMMAR[status]
        eff = hypothesis.get("effect", {})

        # ------------------------------------------------ number whitelist
        numbers: list[dict] = []
        add = lambda d: numbers.append(d) if d else None            # noqa: E731
        add(_num("kpi_pct_change", movement.get("pct_change"), "percent",
                 tolerance=0.06, source="movement detector"))
        add(_num("kpi_focus_value", movement.get("focus_value"), "INR/day",
                 tolerance=1.0, source="movement detector"))
        add(_num("kpi_baseline_value", movement.get("baseline_value"), "INR/day",
                 tolerance=1.0, source="movement detector"))
        if decomposition:
            for t in decomposition.get("terms", []):
                add(_num(f"decomposition_{t['term_id']}_pp", t["value_pp"],
                         "pp of baseline", tolerance=0.06,
                         source="accounting decomposition"))
                add(_num(f"decomposition_{t['term_id']}_inr", t["value_abs"],
                         "INR/day", tolerance=1.0, source="accounting decomposition"))
        if eff.get("kind") == "POINT_WITH_CI":
            add(_num("effect_point", eff.get("point_estimate"), eff.get("unit", ""),
                     tolerance=0.05, source=eff.get("estimator", "")))
            ci = eff.get("confidence_interval") or []
            if len(ci) == 2:
                add(_num("effect_ci_low", ci[0], eff.get("unit", ""), 0.05, "CI"))
                add(_num("effect_ci_high", ci[1], eff.get("unit", ""), 0.05, "CI"))
            add(_num("effect_point_pct", eff.get("point_estimate_pct"), "percent",
                     0.06, "derived"))
            add(_num("p_value", eff.get("p_value_primary"), "", 0.0005,
                     eff.get("p_value_primary_basis", "")))
        temporal = hypothesis.get("temporal") or {}
        if status != "INSUFFICIENT_EVIDENCE":
            add(_num("association_corr", temporal.get("best_corr"), "correlation",
                     0.01, "cross-correlation"))
        if status not in ("ASSOCIATION_ONLY", "INSUFFICIENT_EVIDENCE"):
            add(_num("lag_days", temporal.get("best_lag"), "days", 0.0,
                     "cross-correlation argmax"))
        add(_num("exposure_inr_per_day", materiality.get("exposure", {}).get("inr_per_day"),
                 "INR/day", 1.0, "materiality engine"))
        add(_num("exposure_inr_per_month",
                 materiality.get("impact_basis", {}).get("exposure_inr_per_month"),
                 "INR/month", 1.0, "materiality engine"))
        for e in retrieval.get("supporting", []) + retrieval.get("context", []) + \
                retrieval.get("contradicting", []):
            add(_num(f"evidence_{e['evidence_id']}", e.get("value"), e.get("unit", ""),
                     tolerance=0.06, source=e["source_id"]))

        # every number the engine produced, so the validator can prove that
        # anything else in the prose was invented
        derived: dict = {}
        _collect_numbers(hypothesis, "hypothesis", derived)
        _collect_numbers(movement, "movement", derived)
        _collect_numbers(decomposition or {}, "decomposition", derived)
        _collect_numbers(materiality, "materiality", derived)
        _collect_numbers(ewhr, "ewhr", derived)
        _collect_numbers(retrieval, "retrieval", derived)
        _collect_numbers(recommendations or [], "recommendations", derived)
        _collect_numbers(benchmark or {}, "benchmark", derived)
        _collect_numbers(clarification or {}, "clarification", derived)
        curated = {round(n["value"], 4) for n in numbers}
        for v, path in sorted(derived.items()):
            if v in curated:
                continue
            numbers.append({"label": path, "value": v, "unit": "",
                            "tolerance": max(0.006, abs(v) * 0.005),
                            "source": "engine-derived value"})

        # Authorised entity vocabulary.  Everything here reached the contract
        # through an access-filtered path, so naming it is authorised by
        # construction; the validator treats anything else in the entity universe
        # as leakage.
        # Persona-level entitlement governs which entities may be NAMED; the
        # effective grain governs the resolution of the DATA. An operations
        # manager entitled to WH-N1 may be told about WH-N1 in a recommendation
        # even when the revenue figures are only available to them by region.
        ent: set[str] = (set(decision.allowed_regions) | set(decision.allowed_warehouses)
                         | set(persona.regions) | set(persona.warehouses))
        for e in (retrieval.get("supporting", []) + retrieval.get("contradicting", []) +
                  retrieval.get("context", []) + retrieval.get("gaps", [])):
            if e.get("entity"):
                ent.add(e["entity"])
            if e.get("source_id"):
                ent.add(e["source_id"])
        ent.update(config.MODEL_VERSIONS.keys() & set())
        for s_ in (decomposition or {}).get("cells_entered", []) +                   (decomposition or {}).get("cells_exited", []):
            ent.add(s_)
        for t in (decomposition or {}).get("terms", []):
            for d_ in t.get("detail", []):
                if d_.get("cell"):
                    ent.add(d_["cell"])
        for src in _sources_for(hypothesis, movement):
            ent.add(src)
        entities = sorted(ent)

        allowed_fields = sorted(
            {"date", "region", "product_line", "warehouse", "metric", "value", "unit",
             "period", "freshness", "method", "contribution", "confidence"}
            - set(decision.denied_fields))

        mandatory = list(grammar["mandatory_sections"])
        if clarification:
            mandatory.append("clarification_request")

        tc = TrustContract(
            claim_id=f"CLM-{scenario_id}-{hypothesis['hypothesis_id']}-{persona.persona_id}",
            hypothesis_id=hypothesis["hypothesis_id"],
            kpi_id=movement_kpi_id or hypothesis["outcome_kpi"],
            hypothesis_outcome_kpi=hypothesis["outcome_kpi"],
            scenario_id=scenario_id, persona=persona.persona_id,
            persona_profile=persona.narrative_profile,
            decision_frame=persona.decision_frame,
            access_scope=decision.as_dict(),
            evidence_ids=[e["evidence_id"] for e in retrieval.get("supporting", [])] +
                         [e["evidence_id"] for e in retrieval.get("context", [])],
            contradictory_evidence_ids=[e["evidence_id"]
                                        for e in retrieval.get("contradicting", [])],
            gap_evidence_ids=[e["evidence_id"] for e in retrieval.get("gaps", [])],
            entity_ids=entities, causal_status=status, epistemic_status=status,
            display_label=DISPLAY_LABEL[status],
            estimand=eff.get("estimand"), estimator=eff.get("estimator"),
            effect_estimate={
                "kind": eff.get("kind"),
                "point_estimate": eff.get("point_estimate"),
                "unit": eff.get("unit"),
                "confidence_interval": eff.get("confidence_interval"),
                "interval_withheld_reason": eff.get("interval_withheld_reason"),
                "observed_association": eff.get("observed_association"),
                "conditional_on": eff.get("conditional_on", []),
            },
            uncertainty={
                "p_value_primary": eff.get("p_value_primary"),
                "p_value_basis": eff.get("p_value_primary_basis"),
                "ci_basis": eff.get("ci_basis"),
                "robustness": hypothesis.get("robustness", {}),
                "power_context": hypothesis.get("sufficiency", {}).get("power_context", {}),
                "traceable_to": ["identification status", "estimator inference",
                                  "robustness checks", "data coverage and freshness"],
                "not_derived_from": ["EWHR", "any single confidence number"],
            },
            assumptions=hypothesis.get("assumptions", []),
            missing_data=sorted(set(
                hypothesis.get("missing", {}).get("unavailable_variables", []))),
            missing_evidence=[e["note"] for e in retrieval.get("gaps", [])],
            allowed_claim_types=grammar["allowed_claim_types"],
            forbidden_claim_types=grammar["forbidden_claim_types"],
            allowed_numbers=numbers, allowed_entities=entities,
            allowed_fields=allowed_fields,
            allowed_verbs=grammar["allowed_verbs"],
            forbidden_verbs=sorted(set(grammar["forbidden_verbs"]) |
                                   set(UNIVERSALLY_FORBIDDEN)),
            quantification_rule=grammar["quantification"] + (
                "  |  Number whitelist policy: allowed_numbers contains every value "
                "the deterministic engine produced for this analysis, plus the lakh "
                "and crore renderings of monetary values. Any numeral in the prose "
                "that is not in this list was not produced by the analysis."),
            mandatory_sections=mandatory, materiality=materiality, ewhr=ewhr,
            lineage=lineage, model_versions=dict(config.MODEL_VERSIONS),
            graph_snapshot=graph_snapshot,
            max_words=persona.max_narrative_words)
        return tc.finalise()
