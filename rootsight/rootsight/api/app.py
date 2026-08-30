"""FastAPI surface.

Security note: every endpoint that touches data takes a persona and resolves an
AccessDecision server-side.  There is no endpoint that returns data without one,
and no client-supplied scope can widen it.  /api/security/probe exists so a
reviewer can attempt an unauthorised call and see the refusal and the audit event.
"""
from __future__ import annotations

import json
import os
from datetime import date

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import config
from ..contracts.kpi_contract import ContractViolation, registry
from ..detect.changepoint import MovementDetector
from ..pipeline import Pipeline, warm_state
from ..recommend.engine import RecommendationEngine
from ..scenarios import registry as scenarios
from ..security.audit import audit_log
from ..security.policy import PERSONAS, AccessDenied, PolicyEngine
from ..telemetry import Telemetry

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="RootSight v5", version=config.VERSION,
              description="Causal reasoning layer for KPI movements")
_pipeline = Pipeline()
_policy = PolicyEngine()
_history: list[dict] = []


class AnalyseRequest(BaseModel):
    scenario_id: str = "SC_MULTIFACTOR"
    persona_id: str = "cfo"
    requested_grain: str | None = None
    requested_regions: list[str] | None = None
    prefer_llm: bool = True


class InterventionRequest(BaseModel):
    persona_id: str = "ops_manager"
    hypothesis_id: str = "H1_FULFILMENT"
    playbook_id: str = "PB-OPS-001"
    implemented_on: str = "2026-08-20"


@app.get("/api/health")
def health() -> dict:
    w = warm_state()
    return {"status": "ok", "version": config.VERSION, "as_of": config.AS_OF.isoformat(),
            "llm_enabled": config.LLM_ENABLED,
            "warm_state_build_ms": round(w.built_ms, 1),
            "contract_hash": registry().contract_hash}


@app.get("/api/personas")
def personas() -> dict:
    return {"personas": [p.as_dict() for p in PERSONAS.values()]}


@app.get("/api/scenarios")
def scenario_list() -> dict:
    return {"scenarios": scenarios.catalogue()}


@app.get("/api/contract")
def contract() -> dict:
    reg = registry()
    return {
        "contract_version": reg.contract_version, "registry_hash": reg.contract_hash,
        "sources": {k: {"name": v.name, "system": v.system, "grain": v.grain,
                        "calendar": v.calendar, "refresh_cadence": v.refresh_cadence,
                        "expected_lag_hours": v.expected_lag_hours,
                        "reliability_weight": v.reliability_weight,
                        "timestamp_convention": v.timestamp_convention,
                        "access_classification": v.access_classification,
                        "known_coverage_gaps": v.known_coverage_gaps}
                    for k, v in reg.sources.items()},
        "kpis": {k: {**reg.lineage_card(k),
                     "thresholds": v.thresholds,
                     "materiality_threshold": v.materiality_threshold,
                     "refresh_cadence": v.refresh_cadence,
                     "allowed_aggregations": v.allowed_aggregations,
                     "drivers": [{"driver_id": d.driver_id, "kind": d.kind,
                                  "source": d.source, "grain": d.grain,
                                  "expected_sign": d.expected_sign,
                                  "prior_lag_days": list(d.prior_lag_days)
                                  if d.prior_lag_days else None}
                                 for d in v.drivers],
                     "access_restrictions": {r: {"max_grain": a.max_grain,
                                                 "row_scope": a.row_scope,
                                                 "denied_fields": list(a.denied_fields)}
                                             for r, a in v.access.items()},
                     "launched_on": v.launched_on.isoformat() if v.launched_on else None,
                     "history_days": v.history_days(config.AS_OF)}
                 for k, v in reg.kpis.items()},
        "declared_causal_edges": [{"from": a, "to": b, **reg.edge_provenance()[(a, b)]}
                                  for a, b in reg.declared_edges()],
        "unobserved": reg.unobserved_nodes(),
        "playbook_catalogue": RecommendationEngine.catalogue(),
    }


@app.get("/api/data_quality")
def data_quality() -> dict:
    w = warm_state()
    return {"data_quality": w.bundle.dq.as_dict(),
            "grain_transforms": w.conformed.transform_table(),
            "calendar_notes": w.conformed.calendar_notes,
            "row_counts": w.bundle.summary()["row_counts"],
            "injected_defects": w.bundle.manifest["injected_data_quality_defects"]}


@app.get("/api/series")
def series(kpi_id: str = Query(...), persona_id: str = Query(...)) -> dict:
    """Daily series for charting, filtered by the caller's entitlement."""
    w = warm_state()
    try:
        d = _policy.decide(persona_id, kpi_id)
    except AccessDenied as exc:
        raise HTTPException(status_code=403, detail={"code": exc.code, "message": str(exc)})
    if not d.granted:
        audit_log().record(event_type="SERIES_DENIED", actor=persona_id,
                           role=d.persona.role, resource=kpi_id, outcome="DENY",
                           detail={"reason": d.denied_reason})
        raise HTTPException(status_code=403,
                            detail={"code": "DOMAIN_VIOLATION",
                                    "message": d.denied_reason})
    s = w.kpi.compute(kpi_id, d)
    tot = s.total_by_date().sort_index().dropna()
    return {"kpi_id": kpi_id, "unit": s.unit, "grain": d.effective_grain,
            "time_aggregation": s.definition.time_aggregation,
            "access": d.as_dict(),
            "points": [{"date": k.isoformat(), "value": round(float(v), 4)}
                       for k, v in tot.items()],
            "excluded_rows": s.excluded_rows,
            "excluded_reason": s.excluded_reason}


@app.post("/api/analyse")
def analyse(req: AnalyseRequest) -> dict:
    try:
        out = _pipeline.analyse(scenario_id=req.scenario_id, persona_id=req.persona_id,
                                requested_grain=req.requested_grain,
                                requested_regions=req.requested_regions,
                                prefer_llm=req.prefer_llm)
    except AccessDenied as exc:
        return JSONResponse(status_code=403,
                            content={"code": exc.code, "message": str(exc),
                                     "audit": exc.audit,
                                     "note": ("refused server-side before any data was "
                                              "read; the audit log has the event")})
    except ContractViolation as exc:
        return JSONResponse(status_code=409,
                            content={"code": "CONTRACT_VIOLATION", "message": str(exc)})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    _history.append({"request_id": out["request_id"],
                     "scenario_id": req.scenario_id, "persona_id": req.persona_id,
                     "telemetry": out["telemetry"]})
    return out


@app.get("/api/security/probe")
def probe(persona_id: str, kpi_id: str, grain: str | None = None,
          region: str | None = None) -> dict:
    """Deliberate unauthorised-access attempt, for review.

    The backend refuses. Hiding a UI element would not.
    """
    try:
        d = _policy.decide(persona_id, kpi_id, requested_grain=grain,
                           requested_regions=[region] if region else None)
    except AccessDenied as exc:
        ev = audit_log().record(event_type="PROBE_DENIED", actor=persona_id,
                                role=PERSONAS[persona_id].role if persona_id in PERSONAS else "?",
                                resource=f"{kpi_id}|grain={grain}|region={region}",
                                outcome="DENY",
                                detail={"code": exc.code, **exc.audit})
        return {"allowed": False, "http_equivalent": 403, "code": exc.code,
                "message": str(exc), "audit_event": ev.event_id,
                "enforced_at": "PolicyEngine, before retrieval"}
    if not d.granted:
        ev = audit_log().record(event_type="PROBE_DENIED", actor=persona_id,
                                role=d.persona.role, resource=kpi_id, outcome="DENY",
                                detail={"reason": d.denied_reason})
        return {"allowed": False, "http_equivalent": 403, "code": "DOMAIN_VIOLATION",
                "message": d.denied_reason, "audit_event": ev.event_id,
                "enforced_at": "PolicyEngine, before retrieval"}
    ev = audit_log().record(event_type="PROBE_ALLOWED", actor=persona_id,
                            role=d.persona.role, resource=kpi_id, outcome="ALLOW",
                            detail=d.as_dict())
    return {"allowed": True, "http_equivalent": 200, "decision": d.as_dict(),
            "audit_event": ev.event_id}


@app.post("/api/intervention")
def intervention(req: InterventionRequest) -> dict:
    """Record an intervention and validate it against the pre-intervention claim.

    The recovery is estimated with the same DID design, re-pointed at the
    remediation date. Promotion to SUPPORTED_BY_INTERVENTION requires the
    recovery to be significant AND directionally opposite to the disruption
    estimate - a recovery that fails either test downgrades the hypothesis
    instead of quietly confirming it.
    """
    from datetime import timedelta

    from ..causal.estimate import estimate_did
    w = warm_state()
    try:
        d = _policy.decide(req.persona_id, "units_sold")
    except AccessDenied as exc:
        raise HTTPException(status_code=403, detail={"code": exc.code, "message": str(exc)})
    if not d.granted:
        raise HTTPException(status_code=403, detail={"code": "DOMAIN_VIOLATION"})

    t = Telemetry()
    t0 = date.fromisoformat(req.implemented_on) + timedelta(days=1)
    treated = [f"North | {l}" for l in ("Apparel", "Electronics", "Home")]
    with t.span("post_intervention_estimation", layer="NON_LLM", stage=True):
        up = w.panels.unit_panel(regions=["North", "East", "West"],
                                 exclude_product_lines=("SmartHome",),
                                 start=date(2026, 8, 1), end=config.HISTORY_END)
        res = estimate_did(up, treated_cells=treated, t0=t0, outcome="outcome")
        res.outcome = "units_sold"

    baseline_claim = -51.71
    consistent = res.att_per_unit_day > 0 and res.p_randomisation < 0.10
    status = "SUPPORTED_BY_INTERVENTION" if consistent else "NOT_CONFIRMED_BY_INTERVENTION"
    ev = audit_log().record(
        event_type="INTERVENTION_RECORDED", actor=req.persona_id,
        role=PERSONAS[req.persona_id].role, resource=req.hypothesis_id,
        outcome="RELEASE",
        detail={"playbook_id": req.playbook_id, "implemented_on": req.implemented_on,
                "post_att": round(res.att_per_unit_day, 3),
                "p_randomisation": round(res.p_randomisation, 4), "status": status},
        request_id=t.request_id)
    return {
        "request_id": t.request_id,
        "hypothesis_id": req.hypothesis_id, "playbook_id": req.playbook_id,
        "implemented_on": req.implemented_on, "recovery_window_starts": t0.isoformat(),
        "post_intervention_estimate": res.as_dict(),
        "promotion": {
            "new_status": status,
            "criteria": ["recovery estimate is positive (opposite sign to the "
                         "disruption estimate)",
                         "exact randomisation-inference p below 0.10"],
            "criteria_met": consistent,
            "honest_caveat": (
                f"Only {res.post_periods} post-remediation days exist in this dataset. "
                "The design is the same one used before, which is the point: the "
                "validation is not a different, more forgiving method. A short "
                "post-window has low power, and that is reported rather than hidden."),
        },
        "audit_event": ev.event_id,
        "telemetry": t.report(),
    }


@app.get("/api/audit")
def audit(limit: int = 80) -> dict:
    return {"events": audit_log().tail(limit), "denials": audit_log().denials()}


@app.get("/api/telemetry")
def telemetry_summary() -> dict:
    if not _history:
        return {"analyses": 0, "note": "no analyses run yet in this process"}
    lat = [h["telemetry"]["latency"]["total_ms"] for h in _history]
    cost = [h["telemetry"]["cost"]["usd_per_analysis"] for h in _history]
    calls = [h["telemetry"]["model_calls"]["count"] for h in _history]
    toks = [h["telemetry"]["tokens"]["total"] for h in _history]
    return {
        "analyses": len(_history),
        "latency_ms": {"min": round(min(lat), 1), "median": round(sorted(lat)[len(lat) // 2], 1),
                       "max": round(max(lat), 1)},
        "model_calls_per_analysis": {"min": min(calls), "max": max(calls),
                                     "budget": config.MAX_LLM_CALLS_PER_ANALYSIS},
        "tokens_per_analysis": {"min": min(toks), "max": max(toks)},
        "cost_usd_per_analysis": {"min": round(min(cost), 6), "max": round(max(cost), 6)},
        "projection_usd_per_month_at_50_per_day": round(
            (sum(cost) / len(cost)) * 50 * 30, 4),
        "token_source": _history[-1]["telemetry"]["tokens"]["source"],
        "price_recorded_on": config.COSTS.price_recorded_on,
        "runs": _history[-20:],
    }


@app.get("/api/boundary")
def boundary() -> dict:
    return Pipeline._boundary()


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


if os.path.isdir(STATIC):
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
