"""Clarification requests and abstention.

When the engine cannot answer, it must do one of two things and say which:

  ABSTAIN     no explanation is offered, and the reason is named
  CLARIFY     a specific, answerable question is asked of a named owner

The question is SELECTED deterministically from a template bank keyed by the
dominant gap code, and the slots are filled from the analysis.  A model may
rephrase it; a model may never invent it.  That matters because a clarification
request is an instruction to a human being: it has to be answerable, addressed
to someone who can answer it, and true about what is missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date

from ..contracts.kpi_contract import registry

TEMPLATES = {
    "SOURCE_COVERAGE_GAP": {
        "question": ("{source_name} emits no rows for {entity}. Was {activity} active in "
                     "{entity} between {window_start} and {window_end}, and if so at what "
                     "spend or intensity?"),
        "why": ("Without it, {driver} cannot be included as a candidate driver at all. "
                "It is an unavailable variable, not a zero, so the analysis cannot "
                "distinguish 'no campaign' from 'campaign not recorded'."),
        "owner_hint": "marketing operations owner for {source_id}",
        "resolves": "adds {driver} to the candidate set and re-runs identification",
    },
    "SOURCE_STALE": {
        "question": ("{source_name} last refreshed {age_days} days ago and its data ends "
                     "{period_end}, while the analysis window runs to {window_end}. Can "
                     "the feed be refreshed, or should the window be shortened to "
                     "{period_end}?"),
        "why": ("{covered_days} of the {window_days} days in the window have no "
                "{driver} data. Any statement about {driver} would be extrapolation."),
        "owner_hint": "data platform owner for {source_id}",
        "resolves": "either restores coverage or narrows the claim to the covered window",
    },
    "GRAIN_TOO_COARSE": {
        "question": ("{driver} is recorded at {resolution}-day resolution but the "
                     "hypothesis needs a {required_lag}-day lag. Can {source_name} emit "
                     "daily values, or is there a daily proxy for {driver}?"),
        "why": ("Spreading a weekly total across seven days does not create daily "
                "information. No amount of extra history fixes this; the "
                "instrumentation has to change."),
        "owner_hint": "owner of {source_id}",
        "resolves": "makes the lag measurable and allows a design to be attempted",
    },
    "INSUFFICIENT_HISTORY": {
        "question": ("{kpi_name} has {history_days} days of history since launch on "
                     "{launched_on}. The earliest date a {design} design becomes "
                     "eligible is {eligible_date}. Do you want a descriptive read now, "
                     "or a causal read scheduled for {eligible_date}?"),
        "why": ("A {design} design needs at least {required_days} pre-period days. "
                "Estimating on {history_days} days would produce a number with no "
                "defensible interval."),
        "owner_hint": "KPI owner {owner}",
        "resolves": "sets an honest expectation and schedules the analysis when it can run",
    },
    "UNRESOLVED_CONFOUNDER": {
        "question": ("{confounder} is not instrumented and confounds {driver} with "
                     "{kpi_name}. Is there any internal or purchasable measure of "
                     "{confounder} for {window_start} to {window_end}?"),
        "why": ("Every valid adjustment set for this effect requires {confounder}. "
                "Without it the effect is not point-identified and no interval can be "
                "reported honestly."),
        "owner_hint": "commercial insights owner",
        "resolves": "opens a backdoor adjustment set and makes the effect estimable",
    },
    "CONCURRENT_SHOCK": {
        "question": ("{other_driver} changed within {gap_days} day(s) of {driver} at "
                     "{t0}. Can you confirm the exact start date of {other_driver} so "
                     "the two can be separated?"),
        "why": ("With no control group, two shocks at the same time are one shock as "
                "far as the data is concerned."),
        "owner_hint": "operations owner",
        "resolves": "may separate the two onsets and restore an ITS design",
    },
}


@dataclass
class Clarification:
    clarification_id: str
    gap_code: str
    question: str
    why_it_matters: str
    addressed_to: str
    would_resolve: str
    slots: dict
    selected_by: str = "deterministic template selection from the dominant gap code"
    may_be_rephrased_by_model: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class AbstentionResult:
    abstained: bool
    reason_codes: list[str]
    explanation: str
    clarification: Clarification | None = None
    alternatives_offered: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["clarification"] = self.clarification.as_dict() if self.clarification else None
        return d


def _fill(t: str, slots: dict) -> str:
    out = t
    for k, v in slots.items():
        out = out.replace("{" + k + "}", str(v))
    return out


class ClarificationEngine:
    def __init__(self):
        self.reg = registry()

    def select(self, *, hypotheses: list[dict], dq: dict, kpi_id: str,
               window: tuple[date, date]) -> AbstentionResult:
        codes: list[str] = []
        slots: dict = {"window_start": window[0].isoformat(),
                       "window_end": window[1].isoformat(),
                       "kpi_name": self.reg.get(kpi_id).name,
                       "owner": self.reg.get(kpi_id).owner}
        chosen: str | None = None

        # priority order: a gap that blocks candidate enumeration outranks one that
        # only weakens an estimate
        for h in hypotheses:
            miss = h.get("missing", {})
            if miss.get("unavailable_variables"):
                drv = miss["unavailable_variables"][0]
                src = self._source_for_driver(kpi_id, drv)
                codes.append("SOURCE_COVERAGE_GAP")
                chosen = chosen or "SOURCE_COVERAGE_GAP"
                slots.update({
                    "driver": drv, "source_id": src,
                    "source_name": self.reg.sources[src].name if src in self.reg.sources else src,
                    "entity": ", ".join(h.get("scope", {}).get("regions", []) or ["this scope"]),
                    "activity": "a marketing campaign" if "marketing" in drv else drv,
                })
        for h in hypotheses:
            t = h.get("temporal") or {}
            if "GRAIN_TOO_COARSE" in (t.get("failure_codes") or []):
                drv = h["driver_id"]
                src = self._source_for_driver(kpi_id, drv)
                codes.append("GRAIN_TOO_COARSE")
                chosen = chosen or "GRAIN_TOO_COARSE"
                slots.update({
                    "driver": drv, "source_id": src,
                    "source_name": self.reg.sources[src].name if src in self.reg.sources else src,
                    "resolution": int(t.get("driver_resolution_days", 7)),
                    "required_lag": t.get("best_lag"),
                })
        for h in hypotheses:
            unobs = (h.get("graphical") or {}).get("unobserved_required") or []
            if unobs:
                codes.append("UNRESOLVED_CONFOUNDER")
                chosen = chosen or "UNRESOLVED_CONFOUNDER"
                slots.update({"confounder": unobs[0], "driver": h["driver_id"]})
        for sid, fr in (dq.get("freshness") or {}).items():
            if fr.get("status") == "STALE":
                codes.append("SOURCE_STALE")
                chosen = chosen or "SOURCE_STALE"
                slots.update({
                    "source_id": sid,
                    "source_name": self.reg.sources[sid].name if sid in self.reg.sources else sid,
                    "age_days": int(fr["age_hours"] / 24),
                    "period_end": fr.get("data_period_end"),
                    "driver": "marketing_spend" if sid == "SRC_MKT" else sid,
                    "window_days": (window[1] - window[0]).days + 1,
                    "covered_days": max(0, (date.fromisoformat(fr["data_period_end"])
                                            - window[0]).days + 1)
                    if fr.get("data_period_end") else 0,
                })

        codes = sorted(set(codes))
        if not chosen:
            return AbstentionResult(
                abstained=True, reason_codes=codes or ["UNSPECIFIED"],
                explanation="No candidate driver reached an identifiable status and no "
                            "single dominant data gap was isolated.",
                alternatives_offered=["descriptive movement", "manual investigation"])

        t = TEMPLATES[chosen]
        cl = Clarification(
            clarification_id=f"CLR-{kpi_id}-{chosen}",
            gap_code=chosen,
            question=_fill(t["question"], slots),
            why_it_matters=_fill(t["why"], slots),
            addressed_to=_fill(t["owner_hint"], slots),
            would_resolve=_fill(t["resolves"], slots), slots=slots)
        return AbstentionResult(
            abstained=True, reason_codes=codes,
            explanation=("The engine is abstaining from a driver explanation because the "
                         "evidence cannot support one. A specific, answerable question "
                         "has been generated instead."),
            clarification=cl,
            alternatives_offered=["descriptive movement", "comparable benchmark",
                                  "qualitative ticket evidence",
                                  "scheduled re-analysis once data requirements are met"])

    def sparse_history(self, *, kpi_id: str, history_days: int, launched_on: date,
                       design: str, required_days: int) -> Clarification:
        k = self.reg.get(kpi_id)
        eligible = launched_on.fromordinal(launched_on.toordinal() + required_days)
        slots = {"kpi_name": k.name, "history_days": history_days,
                 "launched_on": launched_on.isoformat(), "design": design,
                 "eligible_date": eligible.isoformat(),
                 "required_days": required_days, "owner": k.owner}
        t = TEMPLATES["INSUFFICIENT_HISTORY"]
        return Clarification(
            clarification_id=f"CLR-{kpi_id}-INSUFFICIENT_HISTORY",
            gap_code="INSUFFICIENT_HISTORY",
            question=_fill(t["question"], slots),
            why_it_matters=_fill(t["why"], slots),
            addressed_to=_fill(t["owner_hint"], slots),
            would_resolve=_fill(t["resolves"], slots), slots=slots)

    def _source_for_driver(self, kpi_id: str, driver_id: str) -> str:
        for d in self.reg.get(kpi_id).drivers:
            if d.driver_id == driver_id:
                return d.source
        return "SRC_ERP"
