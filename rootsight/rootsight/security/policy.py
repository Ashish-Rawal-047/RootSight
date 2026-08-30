"""Access control: personas, roles, row / column / domain security.

Enforcement order is the whole point:

    authenticate -> AccessDecision -> FILTER AT RETRIEVAL -> analyse -> render

Unauthorised data is never loaded, therefore never analysed, therefore never
placed in a prompt.  The LLM is not asked to keep a secret it was told.
`assert_prompt_safe` is a belt-and-braces tripwire that raises if any evidence
object carrying a restricted classification or an out-of-scope row reaches the
prompt builder.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from ..contracts.kpi_contract import KpiContractRegistry, registry

GRAIN_RANK = {
    "DENIED": -1,
    "fiscal_period x region": 0,
    "day x region": 1,
    "day x region x product_line": 2,
    "day x region x ticket_category": 2,
    "day x warehouse": 3,
}


class AccessDenied(Exception):
    def __init__(self, message: str, *, code: str, audit: dict | None = None):
        super().__init__(message)
        self.code = code
        self.audit = audit or {}


@dataclass(frozen=True)
class Persona:
    persona_id: str
    display_name: str
    title: str
    role: str
    regions: tuple[str, ...]            # row scope
    warehouses: tuple[str, ...]
    kpi_domain: tuple[str, ...]         # domain scope: KPIs this persona may touch
    denied_fields: tuple[str, ...]      # column scope, persona-level
    narrative_profile: str              # drives narrative depth & framing
    decision_frame: str
    max_narrative_words: int
    wants_evidence_types: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


PERSONAS: dict[str, Persona] = {
    "cfo": Persona(
        persona_id="cfo",
        display_name="Priya Nair",
        title="Chief Financial Officer",
        role="cfo",
        regions=("North", "South", "East", "West"),
        warehouses=(),                                   # no warehouse-level access
        kpi_domain=("net_revenue", "gross_revenue", "recognized_revenue_finance",
                    "units_sold", "avg_selling_price", "on_time_dispatch_rate",
                    "complaint_rate", "subscription_arr"),
        denied_fields=("customer_id", "customer_email", "employee_id", "shift_id",
                       "staff_productivity_score", "ticket_body"),
        narrative_profile="EXECUTIVE_FINANCIAL",
        decision_frame="capital allocation, earnings exposure, disclosure risk",
        max_narrative_words=190,
        wants_evidence_types=("KPI_MOVEMENT", "DECOMPOSITION", "CAUSAL_ESTIMATE",
                              "FINANCIAL_EXPOSURE", "DATA_GAP"),
    ),
    "ops_manager": Persona(
        persona_id="ops_manager",
        display_name="Rahul Menon",
        title="Regional Operations Manager - North",
        role="ops_manager",
        regions=("North",),                              # ROW-LEVEL: North only
        warehouses=("WH-N1", "WH-N2"),
        kpi_domain=("units_sold", "avg_selling_price", "on_time_dispatch_rate",
                    "complaint_rate", "net_revenue"),
        denied_fields=("gross_margin_pct", "refund_liability_inr",
                       "discount_policy_note", "customer_id", "customer_email",
                       "deferred_delta"),
        narrative_profile="OPERATIONAL_TACTICAL",
        decision_frame="capacity, SLA recovery, queue prioritisation, staffing",
        max_narrative_words=260,
        wants_evidence_types=("OPERATIONAL_METRIC", "TICKET_CLUSTER", "KPI_MOVEMENT",
                              "CAUSAL_ESTIMATE", "DATA_GAP"),
    ),
    "finance_analyst": Persona(
        persona_id="finance_analyst",
        display_name="Ananya Rao",
        title="Senior Finance Analyst",
        role="finance_analyst",
        regions=("North", "South", "East", "West"),
        warehouses=(),
        kpi_domain=("net_revenue", "gross_revenue", "recognized_revenue_finance",
                    "units_sold", "avg_selling_price", "on_time_dispatch_rate",
                    "complaint_rate", "subscription_arr"),
        denied_fields=("customer_id", "customer_email", "employee_id", "shift_id",
                       "staff_productivity_score", "ticket_body"),
        narrative_profile="ANALYTICAL_DETAILED",
        decision_frame="variance explanation, forecast revision, audit trail",
        max_narrative_words=380,
        wants_evidence_types=("KPI_MOVEMENT", "DECOMPOSITION", "CAUSAL_ESTIMATE",
                              "DIAGNOSTIC", "DATA_GAP", "OPERATIONAL_METRIC",
                              "TICKET_CLUSTER", "FINANCIAL_EXPOSURE"),
    ),
}

# Field-level sensitivity classes.  Evidence objects are tagged with one.
FIELD_CLASSIFICATION = {
    "customer_id": "PII",
    "customer_email": "PII",
    "ticket_body": "PII_FREETEXT",
    "employee_id": "HR",
    "shift_id": "HR",
    "staff_productivity_score": "HR",
    "gross_margin_pct": "RESTRICTED_FINANCIAL",
    "refund_liability_inr": "RESTRICTED_FINANCIAL",
    "deferred_delta": "RESTRICTED_FINANCIAL",
    "discount_policy_note": "RESTRICTED_COMMERCIAL",
}


@dataclass
class AccessDecision:
    persona: Persona
    kpi_id: str
    granted: bool
    effective_grain: str
    allowed_regions: tuple[str, ...]
    allowed_warehouses: tuple[str, ...]
    denied_fields: tuple[str, ...]
    denied_reason: str | None = None
    downgrades: list[str] = field(default_factory=list)
    policy_trace: list[str] = field(default_factory=list)

    def allows_region(self, region: str) -> bool:
        return region in self.allowed_regions

    def allows_field(self, fld: str) -> bool:
        return fld not in self.denied_fields

    def as_dict(self) -> dict:
        return {
            "persona_id": self.persona.persona_id, "role": self.persona.role,
            "kpi_id": self.kpi_id, "granted": self.granted,
            "effective_grain": self.effective_grain,
            "allowed_regions": list(self.allowed_regions),
            "allowed_warehouses": list(self.allowed_warehouses),
            "denied_fields": sorted(self.denied_fields),
            "denied_reason": self.denied_reason,
            "downgrades": list(self.downgrades),
            "policy_trace": list(self.policy_trace),
        }


class PolicyEngine:
    """Combines persona entitlements with the KPI contract's own restrictions.

    Both must permit an access: the intersection is the effective entitlement.
    Neither can widen the other.
    """

    def __init__(self, reg: KpiContractRegistry | None = None):
        self.reg = reg or registry()

    def persona(self, persona_id: str) -> Persona:
        if persona_id not in PERSONAS:
            raise AccessDenied(f"unknown persona {persona_id!r}", code="UNKNOWN_PERSONA")
        return PERSONAS[persona_id]

    def decide(self, persona_id: str, kpi_id: str, *,
               requested_grain: str | None = None,
               requested_regions: list[str] | None = None) -> AccessDecision:
        p = self.persona(persona_id)
        trace: list[str] = []

        # ---- domain scope -------------------------------------------------
        if kpi_id not in p.kpi_domain:
            return AccessDecision(
                persona=p, kpi_id=kpi_id, granted=False, effective_grain="DENIED",
                allowed_regions=(), allowed_warehouses=(), denied_fields=("ALL",),
                denied_reason=f"DOMAIN: role {p.role!r} has no entitlement to KPI {kpi_id!r}",
                policy_trace=["domain_scope: DENY"])
        trace.append(f"domain_scope: {kpi_id} in persona domain -> ALLOW")

        rule = self.reg.get(kpi_id).access_for(p.role)
        if rule.denied:
            return AccessDecision(
                persona=p, kpi_id=kpi_id, granted=False, effective_grain="DENIED",
                allowed_regions=(), allowed_warehouses=(), denied_fields=("ALL",),
                denied_reason=(f"CONTRACT: KPI {kpi_id!r} declares role {p.role!r} "
                               f"as DENIED in access_restrictions"),
                policy_trace=trace + ["contract_rule: DENY"])
        trace.append(f"contract_rule: max_grain={rule.max_grain} row_scope={rule.row_scope}")

        # ---- row scope ----------------------------------------------------
        if rule.row_scope == "OWN_REGION":
            allowed_regions = p.regions
        else:
            allowed_regions = p.regions
        downgrades: list[str] = []
        if requested_regions:
            bad = [r for r in requested_regions if r not in allowed_regions]
            if bad:
                raise AccessDenied(
                    f"row-level security: role {p.role!r} is scoped to regions "
                    f"{list(allowed_regions)} and requested {bad}",
                    code="ROW_SCOPE_VIOLATION",
                    audit={"persona": persona_id, "kpi_id": kpi_id,
                           "requested_regions": requested_regions,
                           "allowed_regions": list(allowed_regions)})
        trace.append(f"row_scope: regions={list(allowed_regions)}")

        # ---- grain (a form of domain/aggregation security) ----------------
        effective = rule.max_grain
        if requested_grain:
            if GRAIN_RANK.get(requested_grain, 99) > GRAIN_RANK.get(rule.max_grain, -1):
                raise AccessDenied(
                    f"grain security: role {p.role!r} may not query {kpi_id!r} at "
                    f"grain {requested_grain!r}; maximum permitted is {rule.max_grain!r}",
                    code="GRAIN_VIOLATION",
                    audit={"persona": persona_id, "kpi_id": kpi_id,
                           "requested_grain": requested_grain,
                           "max_grain": rule.max_grain})
            effective = requested_grain
        trace.append(f"grain: effective={effective}")

        # ---- column scope -------------------------------------------------
        denied_fields = tuple(sorted(set(rule.denied_fields) | set(p.denied_fields)))
        warehouses = p.warehouses if GRAIN_RANK.get(effective, 0) >= 3 else ()
        if not p.warehouses and GRAIN_RANK.get(rule.max_grain, 0) < 3:
            downgrades.append("warehouse-level detail not available at this role's grain")
        trace.append(f"column_scope: denied={len(denied_fields)} fields")

        return AccessDecision(
            persona=p, kpi_id=kpi_id, granted=True, effective_grain=effective,
            allowed_regions=tuple(allowed_regions), allowed_warehouses=tuple(warehouses),
            denied_fields=denied_fields, downgrades=downgrades, policy_trace=trace)

    # ------------------------------------------------------------- tripwire
    @staticmethod
    def assert_prompt_safe(payload: Any, decision: AccessDecision) -> None:
        """Raises if anything about to be sent to an LLM breaches the decision.

        Walks the serialised payload looking for (a) denied field names used as
        keys, (b) region values outside the row scope, (c) any PII / HR
        classification marker.
        """
        offences: list[str] = []
        denied = set(decision.denied_fields)
        allowed_regions = set(decision.allowed_regions)

        def walk(node: Any, path: str) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in denied:
                        offences.append(f"denied field {k!r} present at {path}")
                    if k == "region" and isinstance(v, str) and allowed_regions and v not in allowed_regions:
                        offences.append(f"out-of-scope region {v!r} at {path}")
                    if k == "access_classification" and v in ("PII", "PII_FREETEXT", "HR"):
                        offences.append(f"classification {v} present at {path}")
                    walk(v, f"{path}.{k}")
            elif isinstance(node, (list, tuple)):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")

        walk(payload, "$")
        if offences:
            raise AccessDenied(
                "prompt-safety tripwire: refusing to send restricted data to the "
                "model:\n  - " + "\n  - ".join(sorted(set(offences))),
                code="PROMPT_SAFETY_VIOLATION",
                audit={"persona": decision.persona.persona_id,
                       "kpi_id": decision.kpi_id, "offences": sorted(set(offences))})


def redact(record: dict, decision: AccessDecision) -> tuple[dict, list[str]]:
    """Column-level security applied to a single record."""
    out, removed = {}, []
    for k, v in record.items():
        if k in decision.denied_fields:
            removed.append(k)
            continue
        out[k] = v
    return out, removed
