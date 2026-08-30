"""Narrative plan construction - deterministic, no model involved.

In V4 the LLM produced the narrative plan and a validator checked it afterwards.
V5 moves plan construction fully into code.  The model's remaining job is prose
rendering of a plan it cannot alter, which shrinks the surface where a
hallucination can enter from "the whole argument" to "the connective tissue".

Every fact carries `locked`.  A locked sentence is inserted verbatim and the
validator checks it survived intact; every number in the narrative lives in a
locked sentence.

Persona changes: which sections exist, their order, how much detail each holds,
and the decision frame.  Persona never changes the causal status, the numbers,
or which contradicting evidence is disclosed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from ..security.policy import PERSONAS
from .trust_contract import TrustContract


def _fmt_inr(x: float) -> str:
    a = abs(x)
    if a >= 1e7:
        return f"{'-' if x < 0 else ''}Rs {a / 1e7:.2f} crore"
    if a >= 1e5:
        return f"{'-' if x < 0 else ''}Rs {a / 1e5:.2f} lakh"
    return f"{'-' if x < 0 else ''}Rs {a:,.0f}"


@dataclass
class Fact:
    fact_id: str
    text: str
    locked: bool
    evidence_ids: list[str] = field(default_factory=list)
    numbers: list[float] = field(default_factory=list)
    claim_type: str = "MEASURED_FACT"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Section:
    section_id: str
    heading: str
    facts: list[Fact]
    required: bool = True

    def as_dict(self) -> dict:
        return {"section_id": self.section_id, "heading": self.heading,
                "required": self.required, "facts": [f.as_dict() for f in self.facts]}


@dataclass
class NarrativePlan:
    plan_version: str
    claim_id: str
    persona: str
    display_label: str
    causal_status: str
    sections: list[Section]
    permitted_verbs: list[str]
    forbidden_verbs: list[str]
    max_words: int
    built_by: str = "deterministic plan builder (no LLM)"

    def as_dict(self) -> dict:
        return {"plan_version": self.plan_version, "claim_id": self.claim_id,
                "persona": self.persona, "display_label": self.display_label,
                "causal_status": self.causal_status, "built_by": self.built_by,
                "permitted_verbs": self.permitted_verbs,
                "forbidden_verbs": self.forbidden_verbs, "max_words": self.max_words,
                "sections": [s.as_dict() for s in self.sections]}

    def all_numbers(self) -> list[float]:
        return [n for s in self.sections for f in s.facts for n in f.numbers]

    def locked_sentences(self) -> list[str]:
        return [f.text for s in self.sections for f in s.facts if f.locked]


PROFILE_SECTIONS = {
    "EXECUTIVE_FINANCIAL": ["headline", "exposure", "driver", "limits",
                            "contradicting", "gaps", "decision"],
    "OPERATIONAL_TACTICAL": ["headline", "operational", "driver", "affected",
                             "limits", "contradicting", "gaps", "decision"],
    "ANALYTICAL_DETAILED": ["headline", "decomposition", "driver", "method",
                            "diagnostics", "limits", "contradicting", "gaps",
                            "decision"],
}


class PlanBuilder:
    PLAN_VERSION = "narrative-plan-v5.0"

    def build(self, tc: TrustContract, *, movement: dict, decomposition: dict | None,
              retrieval: dict, hypothesis: dict,
              recommendations: list[dict] | None = None,
              clarification: dict | None = None,
              benchmark: dict | None = None) -> NarrativePlan:
        p = PERSONAS[tc.persona]
        order = PROFILE_SECTIONS[p.narrative_profile]
        builders = {
            "headline": self._headline, "exposure": self._exposure,
            "decomposition": self._decomposition, "driver": self._driver,
            "operational": self._operational, "affected": self._affected,
            "method": self._method, "diagnostics": self._diagnostics,
            "limits": self._limits, "contradicting": self._contradicting,
            "gaps": self._gaps, "decision": self._decision,
        }
        ctx = {"tc": tc, "movement": movement, "decomposition": decomposition,
               "retrieval": retrieval, "hypothesis": hypothesis,
               "recommendations": recommendations or [], "benchmark": benchmark}
        sections: list[Section] = []
        for sid in order:
            s = builders[sid](ctx)
            if s and s.facts:
                sections.append(s)
        if clarification:
            sections.append(Section(
                "clarification_request", "What would resolve this",
                [Fact("clarify_q", clarification["question"], True,
                      claim_type="CLARIFICATION_REQUEST"),
                 Fact("clarify_why", clarification["why_it_matters"], True,
                      claim_type="DATA_REQUIREMENT")]))
        if tc.causal_status == "INSUFFICIENT_EVIDENCE":
            sections.insert(1, self._abstention(ctx))
            if benchmark:
                sections.append(self._benchmark(ctx))
        return NarrativePlan(
            plan_version=self.PLAN_VERSION, claim_id=tc.claim_id, persona=tc.persona,
            display_label=tc.display_label, causal_status=tc.causal_status,
            sections=sections, permitted_verbs=tc.allowed_verbs,
            forbidden_verbs=tc.forbidden_verbs, max_words=tc.max_words)

    # -------------------------------------------------------------- sections
    def _headline(self, c) -> Section:
        tc, m = c["tc"], c["movement"]
        facts = []
        pct = m.get("pct_change")
        if pct is not None:
            facts.append(Fact(
                "h_move",
                f"{tc.kpi_id.replace('_', ' ').title()} changed by {pct:+.2f}% in "
                f"{m['focus_window'][0]} to {m['focus_window'][1]} against the "
                f"{m['baseline_window'][0]} to {m['baseline_window'][1]} baseline, "
                f"which exceeds the {m['threshold_pct']}% alert threshold declared in "
                f"the KPI contract.",
                locked=True, numbers=[pct, m["threshold_pct"]],
                claim_type="MEASURED_FACT"))
        else:
            facts.append(Fact(
                "h_nobase",
                f"{tc.kpi_id.replace('_', ' ').title()} has no comparable prior period, "
                f"so no percentage change against a baseline can be reported.",
                locked=True, claim_type="MEASURED_FACT"))
        facts.append(Fact(
            "h_status",
            f"Status for this hypothesis: {tc.display_label}.",
            locked=True, claim_type="MEASURED_FACT"))
        return Section("headline", "What moved", facts)

    def _exposure(self, c) -> Section:
        tc = c["tc"]
        mt = tc.materiality
        if not mt:
            return Section("exposure", "Financial exposure", [])
        ib = mt.get("impact_basis", {})
        day = mt.get("exposure", {}).get("inr_per_day")
        mon = ib.get("exposure_inr_per_month")
        facts = []
        if day is not None and mon is not None:
            kind = ("a causal effect estimate" if mt.get("exposure", {}).get("is_causal")
                    else "an arithmetic contribution, not a causal effect")
            facts.append(Fact(
                "x_exposure",
                f"Exposure attached to this hypothesis is {_fmt_inr(day)} per day "
                f"({_fmt_inr(mon)} per month), derived from "
                f"{mt['exposure']['derivation']}. This figure is {kind}.",
                locked=True, numbers=[day, mon], claim_type="ARITHMETIC_CONTRIBUTION"))
            facts.append(Fact(
                "x_material",
                f"Business impact tier {mt['impact_tier']} at "
                f"{ib.get('ratio_to_threshold')} times the materiality threshold in the "
                f"KPI contract; statistical evidence tier {mt['statistical_tier']}; "
                f"decision priority {mt['decision_priority']}.",
                locked=True, numbers=[ib.get("ratio_to_threshold")],
                claim_type="MEASURED_FACT"))
        for w in mt.get("warnings", []):
            facts.append(Fact(f"x_warn_{len(facts)}", w, locked=True,
                              claim_type="MEASURED_FACT"))
        return Section("exposure", "Financial exposure and materiality", facts)

    def _decomposition(self, c) -> Section:
        d = c["decomposition"]
        if not d:
            return Section("decomposition", "Arithmetic decomposition", [])
        facts = [Fact(
            "d_intro",
            "The movement decomposes exactly into arithmetic contributions. These are "
            "identities, not causes: they say what moved, not why.",
            locked=True, claim_type="ARITHMETIC_CONTRIBUTION")]
        for t in d["terms"]:
            # deliberately verbless: "contributed" reads as causal in English and is
            # forbidden at several epistemic levels, yet the arithmetic itself is
            # always reportable
            facts.append(Fact(
                f"d_{t['term_id']}",
                f"{t['label']}: {t['value_pp']:+.2f} percentage points of the movement "
                f"({_fmt_inr(t['value_abs'])} per day), as an identity term.",
                locked=True, numbers=[t["value_pp"], t["value_abs"]],
                claim_type="ARITHMETIC_CONTRIBUTION"))
        facts.append(Fact(
            "d_close",
            f"The identity closes with a residual of {d['residual_abs']}, so no part of "
            f"the movement is unaccounted for arithmetically.",
            locked=True, numbers=[d["residual_abs"]],
            claim_type="ARITHMETIC_CONTRIBUTION"))
        if d.get("cells_entered"):
            facts.append(Fact(
                "d_entry",
                f"Newly launched cells in the window ({', '.join(d['cells_entered'])}) "
                f"are reported as a separate entry term because they have no baseline "
                f"price and cannot be expressed as volume, mix or price.",
                locked=True, claim_type="ARITHMETIC_CONTRIBUTION"))
        return Section("decomposition", "Arithmetic decomposition (not causal)", facts)

    def _driver(self, c) -> Section:
        tc, h = c["tc"], c["hypothesis"]
        eff = tc.effect_estimate
        facts = []
        if tc.causal_status == "SUPPORTED_BY_DESIGN":
            ci = eff.get("confidence_interval") or [None, None]
            facts.append(Fact(
                "dr_effect",
                f"{h['label']} is estimated to account for {eff['point_estimate']:+.2f} "
                f"{eff.get('unit', '')} in {tc.hypothesis_outcome_kpi.replace('_', ' ')} "
                f"(95% interval {ci[0]:+.2f} to {ci[1]:+.2f}), under the stated "
                f"assumptions, using {tc.estimator}.",
                locked=True,
                numbers=[eff["point_estimate"], ci[0], ci[1]],
                evidence_ids=tc.evidence_ids[:3],
                claim_type="CAUSAL_EFFECT_CONDITIONAL"))
            facts.append(Fact(
                "dr_estimand",
                f"The quantity estimated is: {tc.estimand}",
                locked=True, claim_type="CAUSAL_EFFECT_CONDITIONAL"))
            p = tc.uncertainty.get("p_value_primary")
            if p is not None:
                facts.append(Fact(
                    "dr_p",
                    f"Primary inference gives p = {p:.4f} ({tc.uncertainty['p_value_basis']}).",
                    locked=True, numbers=[p], claim_type="MEASURED_FACT"))
        elif tc.causal_status == "NOT_POINT_IDENTIFIED":
            assoc = eff.get("observed_association") or {}
            facts.append(Fact(
                "dr_leading",
                f"{h['label']} is the leading hypothesis for this movement, but its "
                f"causal effect is not point-identified.",
                locked=True, evidence_ids=tc.evidence_ids[:3],
                claim_type="HYPOTHESIS_RANKING"))
            if assoc.get("value") is not None:
                facts.append(Fact(
                    "dr_assoc",
                    f"The observed association is {assoc['value']:+.2f} at a lag of "
                    f"{assoc['lag_days']} day(s). This is an association, not an effect.",
                    locked=True, numbers=[assoc["value"], assoc["lag_days"]],
                    claim_type="ASSOCIATION"))
            facts.append(Fact(
                "dr_nointerval",
                f"No effect magnitude or interval is reported: "
                f"{eff.get('interval_withheld_reason', '')}",
                locked=True, claim_type="IDENTIFICATION_FAILURE"))
        elif tc.causal_status == "ASSOCIATION_ONLY":
            assoc = eff.get("observed_association") or {}
            if assoc.get("value") is not None:
                facts.append(Fact(
                    "dr_assoc_only",
                    f"{h['driver_id'].replace('_', ' ')} was associated with "
                    f"{tc.kpi_id.replace('_', ' ')} at {assoc['value']:+.2f}. No timing "
                    f"claim is made and no effect is estimated.",
                    locked=True, numbers=[assoc["value"]], claim_type="ASSOCIATION"))
        return Section("driver", "Driver assessment", facts)

    def _operational(self, c) -> Section:
        r = c["retrieval"]
        facts = []
        for e in r.get("supporting", [])[:4]:
            if e["source_type"] not in ("OPERATIONAL_METRIC", "TICKET_CLUSTER"):
                continue
            facts.append(Fact(
                f"op_{e['evidence_id']}",
                f"{e['entity']}: {e['metric']} = {e['value']} {e['unit']} for "
                f"{e['period']} (source {e['source_id']}, freshness "
                f"{e['freshness']['status']}, method {e['method']}).",
                locked=True, numbers=[e["value"]] if isinstance(e["value"], (int, float)) else [],
                evidence_ids=[e["evidence_id"]], claim_type="MEASURED_FACT"))
        return Section("operational", "Operational readings", facts)

    def _affected(self, c) -> Section:
        tc = c["tc"]
        facts = [Fact(
            "af_scope",
            f"Entities in scope for your role: {', '.join(tc.access_scope['allowed_regions'])} "
            f"at grain {tc.access_scope['effective_grain']}.",
            locked=True, claim_type="MEASURED_FACT")]
        if tc.access_scope.get("downgrades"):
            facts.append(Fact(
                "af_downgrade",
                "Scope note: " + "; ".join(tc.access_scope["downgrades"]) + ".",
                locked=True, claim_type="MEASURED_FACT"))
        return Section("affected", "Affected scope", facts)

    def _method(self, c) -> Section:
        tc, h = c["tc"], c["hypothesis"]
        facts = [Fact(
            "m_chain",
            f"Identification chain: graphical strategy "
            f"{h.get('graphical', {}).get('strategy', 'n/a')} with adjustment set "
            f"{h.get('graphical', {}).get('adjustment_set', [])}; design "
            f"{h.get('chosen_design') or 'none qualified'}; structure screen "
            f"{h.get('structural_support')}.",
            locked=True, claim_type="MEASURED_FACT")]
        for d in h.get("designs", []):
            facts.append(Fact(
                f"m_design_{d['design']}",
                f"{d['design']} eligibility: {'YES' if d['eligible'] else 'NO'} - {d['reason']}",
                locked=True, claim_type="MEASURED_FACT"))
        return Section("method", "Method and identification", facts)

    def _diagnostics(self, c) -> Section:
        h = c["hypothesis"]
        facts = []
        for chk in h.get("robustness", {}).get("checks", []):
            bits = [f"{k}={v}" for k, v in chk.items()
                    if k in ("estimate", "range", "p_value", "p_randomisation",
                              "relative_change_pct", "verdict")]
            facts.append(Fact(
                f"g_{chk.get('check')}",
                f"Robustness - {chk.get('check')}: " + ", ".join(bits) +
                (f". {chk.get('description', '')}" if chk.get("description") else ""),
                locked=True, claim_type="MEASURED_FACT"))
        pw = h.get("sufficiency", {}).get("power_context", {})
        if pw:
            facts.append(Fact(
                "g_power",
                f"Minimum detectable effect at this sample size and autocorrelation: "
                f"{pw.get('minimum_detectable_effect_pct_of_mean')}% of the mean. A "
                f"smaller true effect could be real and still invisible here.",
                locked=True, numbers=[pw.get("minimum_detectable_effect_pct_of_mean")],
                claim_type="MEASURED_FACT"))
        return Section("diagnostics", "Robustness and power", facts)

    def _limits(self, c) -> Section:
        tc = c["tc"]
        facts = []
        bad = [a for a in tc.assumptions if a["status"] in ("VIOLATED", "UNKNOWN")]
        for a in bad[:4]:
            facts.append(Fact(
                f"l_{a['id']}",
                f"{a['id']} {a['name']}: {a['status']} - {a['evidence']}",
                locked=True, claim_type="MEASURED_FACT"))
        assumed = [a for a in tc.assumptions if a["status"] == "ASSUMED"]
        if assumed:
            facts.append(Fact(
                "l_assumed",
                "Assumed but not testable from this data: " +
                ", ".join(f"{a['id']} {a['name']}" for a in assumed) + ".",
                locked=True, claim_type="MEASURED_FACT"))
        return Section("limits", "What this rests on", facts)

    def _contradicting(self, c) -> Section:
        r = c["retrieval"]
        items = r.get("contradicting", [])
        if not items:
            return Section("contradicting", "Evidence against", [Fact(
                "c_none",
                "No contradicting evidence was found in the retrieved window. Absence of "
                "contradicting evidence is not corroboration.",
                locked=True, claim_type="MEASURED_FACT")])
        facts = []
        for e in items:
            facts.append(Fact(
                f"c_{e['evidence_id']}",
                f"Against: {e['entity']} {e['metric']} = {e['value']} {e['unit']} "
                f"({e['period']}, source {e['source_id']}). {e['note']}",
                locked=True,
                numbers=[e["value"]] if isinstance(e["value"], (int, float)) else [],
                evidence_ids=[e["evidence_id"]], claim_type="MEASURED_FACT"))
        return Section("contradicting", "Evidence against this hypothesis", facts)

    def _gaps(self, c) -> Section:
        r = c["retrieval"]
        items = r.get("gaps", [])
        if not items:
            return Section("gaps", "Data gaps", [])
        facts = []
        for e in items:
            facts.append(Fact(
                f"gp_{e['evidence_id']}",
                f"Gap ({e['value']}): {e['note']} Source {e['source_id']}, freshness "
                f"{e['freshness']['status']}.",
                locked=True, evidence_ids=[e["evidence_id"]],
                claim_type="DATA_REQUIREMENT"))
        return Section("gaps", "What is missing or stale", facts)

    def _decision(self, c) -> Section:
        recs = c["recommendations"]
        tc = c["tc"]
        if not recs:
            return Section("decision", "Recommended action", [Fact(
                "r_none",
                f"No action is recommended: licensed action type for this evidence is "
                f"{tc.materiality.get('action_type', 'NO_ACTION')}.",
                locked=True, claim_type="MEASURED_FACT")])
        facts = []
        for r in recs:
            facts.append(Fact(
                f"r_{r['recommendation_id']}",
                f"{r['title']} - action type {r['action_type']}, score {r['score']:.2f}, "
                f"cost {r['cost_band']}, reversibility {r['reversibility']}. "
                f"{r['persona_framing']}",
                locked=True, numbers=[r["score"]], claim_type="MEASURED_FACT"))
        return Section("decision", "Recommended action", facts)

    def _abstention(self, c) -> Section:
        tc, h = c["tc"], c["hypothesis"]
        facts = [Fact(
            "ab_1",
            "RootSight is abstaining from a driver explanation for this movement. The "
            "evidence does not support one.",
            locked=True, claim_type="ABSTENTION"),
            Fact("ab_2",
                 "Reason: " + (h.get("notes") or ["insufficient evidence"])[0],
                 locked=True, claim_type="ABSTENTION")]
        if tc.missing_data:
            facts.append(Fact(
                "ab_3",
                "Unavailable variables: " + ", ".join(tc.missing_data) + ". These are "
                "absent, not zero.",
                locked=True, claim_type="DATA_REQUIREMENT"))
        return Section("abstention", "Why no explanation is offered", facts)

    def _benchmark(self, c) -> Section:
        bm = c["benchmark"]
        facts = [Fact(
            "bm_intro",
            "In place of a causal explanation, here is what can be said responsibly.",
            locked=True, claim_type="BENCHMARK_COMPARISON")]
        for line in bm.get("lines", []):
            facts.append(Fact(
                f"bm_{len(facts)}", line["text"], locked=True,
                numbers=line.get("numbers", []), claim_type="BENCHMARK_COMPARISON"))
        return Section("benchmark", "Descriptive view and comparable benchmark", facts)
