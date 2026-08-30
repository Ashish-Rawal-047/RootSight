"""Scenario registry.

Scenarios are declarative.  The engine is generic; a scenario says which KPI,
which windows, which candidate drivers, and which quasi-experimental designs are
even on the table.  `demonstrates` lists the Round-2 acceptance criteria each
scenario is there to exercise, so the mapping from requirement to running code is
checkable rather than asserted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from .. import config
from ..causal.identify import DidSpec, HypothesisSpec, ItsSpec

DISRUPTION_T0 = config.GT.disruption_start          # operational onset (2026-07-28)
DEMAND_T0 = config.GT.disruption_start + timedelta(days=config.GT.disruption_lag_days)
CORE_LINES = ["Apparel", "Electronics", "Home"]
ALL_REGIONS = ["North", "South", "East", "West"]


@dataclass
class Scenario:
    scenario_id: str
    title: str
    subtitle: str
    kpi_id: str
    focus_window: tuple[date, date]
    baseline_window: tuple[date, date]
    scope_regions: list[str]
    scope_product_lines: list[str] | None
    hypotheses: list[HypothesisSpec]
    demonstrates: list[str]
    expected_outcome: str
    exposure_rules: dict = field(default_factory=dict)
    benchmark_kind: str | None = None

    def as_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id, "title": self.title,
            "subtitle": self.subtitle, "kpi_id": self.kpi_id,
            "focus_window": [d.isoformat() for d in self.focus_window],
            "baseline_window": [d.isoformat() for d in self.baseline_window],
            "scope_regions": self.scope_regions,
            "scope_product_lines": self.scope_product_lines,
            "hypotheses": [{"hypothesis_id": h.hypothesis_id, "driver_id": h.driver_id,
                            "label": h.label, "mechanism": h.mechanism,
                            "designs_on_the_table": [d for d, s in
                                                     (("DID", h.did), ("ITS", h.its)) if s]}
                           for h in self.hypotheses],
            "demonstrates": self.demonstrates,
            "expected_outcome": self.expected_outcome,
        }


# ============================================================ SC_MULTIFACTOR
def _multifactor() -> Scenario:
    hyps = [
        HypothesisSpec(
            hypothesis_id="H1_FULFILMENT",
            driver_id="on_time_dispatch_rate", outcome_kpi="units_sold",
            label="North fulfilment disruption",
            mechanism=("SLA breach at WH-N1/WH-N2 raises delivery times, which raises "
                       "complaints and suppresses reorder and cart completion in North"),
            scope_regions=["North"], scope_product_lines=CORE_LINES,
            did=DidSpec(
                treated_cells=[f"North | {l}" for l in CORE_LINES],
                t0=DEMAND_T0, regions=["North", "East", "West"],
                exclude_product_lines=("SmartHome",),
                window_start=date(2026, 5, 1), window_end=date(2026, 8, 23)),
            its=ItsSpec(regions=["North"], t0=DEMAND_T0, product_lines=CORE_LINES,
                        window_start=date(2026, 4, 1), window_end=date(2026, 8, 23)),
            unit_conversion_note=(
                "Units are converted to revenue by multiplying by the treated cells' "
                "baseline realised price. That multiplication is deterministic "
                "arithmetic, not a second causal estimate.")),
        HypothesisSpec(
            hypothesis_id="H2_COMPETITOR",
            driver_id="competitor_promo", outcome_kpi="units_sold",
            label="Competitor promotion on Electronics",
            mechanism="price-led substitution away from our Electronics range",
            scope_regions=ALL_REGIONS, scope_product_lines=["Electronics"]),
        HypothesisSpec(
            hypothesis_id="H3_MARKETING",
            driver_id="marketing_spend", outcome_kpi="units_sold",
            label="National marketing spend reduction",
            mechanism="lower paid demand generation reduces top-of-funnel volume",
            scope_regions=["North", "East", "West"], scope_product_lines=CORE_LINES),
        HypothesisSpec(
            hypothesis_id="H4_PRICE",
            driver_id="avg_selling_price", outcome_kpi="units_sold",
            label="Realised price movement",
            mechanism="own-price elasticity: a higher realised price suppresses volume",
            scope_regions=ALL_REGIONS, scope_product_lines=CORE_LINES),
        HypothesisSpec(
            hypothesis_id="H5_COMPLAINTS",
            driver_id="complaint_rate", outcome_kpi="units_sold",
            label="Complaint visibility",
            mechanism="visible complaints and poor reviews reduce conversion",
            scope_regions=["North"], scope_product_lines=CORE_LINES),
    ]
    return Scenario(
        scenario_id="SC_MULTIFACTOR",
        title="Net revenue fell across the company",
        subtitle=("Five candidate drivers, three of which cannot be point-identified for "
                  "three different reasons"),
        kpi_id="net_revenue", focus_window=config.FOCUS_WINDOW,
        baseline_window=config.BASELINE_WINDOW, scope_regions=ALL_REGIONS,
        scope_product_lines=None, hypotheses=hyps,
        exposure_rules={
            "H1_FULFILMENT": "DID_UNITS_X_BASELINE_PRICE",
            "H2_COMPETITOR": "SEGMENT_CO_MOVEMENT",
            "H3_MARKETING": "SEGMENT_CO_MOVEMENT",
            "H4_PRICE": "DECOMPOSITION_TERM:D_PRICE",
            "H5_COMPLAINTS": "SEGMENT_CO_MOVEMENT",
        },
        demonstrates=[
            "R1 three to five connected KPIs", "R14 multi-factor KPI movement",
            "R15 known simulated drivers", "R22 contribution",
            "R21 analytical method", "R23 confidence", "R24 lineage",
            "R12 persona narratives", "R13 persona actions",
        ],
        expected_outcome=(
            "H1 SUPPORTED_BY_DESIGN with a DID point estimate and interval; H2 and H4 "
            "NOT_POINT_IDENTIFIED because an unobserved promotion intensity confounds "
            "them; H3 ASSOCIATION_ONLY because weekly grain cannot resolve the required "
            "lag; H5 NOT_POINT_IDENTIFIED with no design available."))


# =============================================================== SC_LOWCONF
def _lowconf() -> Scenario:
    hyps = [
        HypothesisSpec(
            hypothesis_id="H1_SOUTH_MARKETING",
            driver_id="marketing_spend", outcome_kpi="units_sold",
            label="South campaign activity",
            mechanism="regional campaigns drive regional volume",
            scope_regions=["South"], scope_product_lines=CORE_LINES),
        HypothesisSpec(
            hypothesis_id="H2_SOUTH_COMPETITOR",
            driver_id="competitor_promo", outcome_kpi="units_sold",
            label="Competitor promotion in South",
            mechanism="price-led substitution",
            scope_regions=["South"], scope_product_lines=["Electronics"]),
        HypothesisSpec(
            hypothesis_id="H3_SOUTH_FULFILMENT",
            driver_id="on_time_dispatch_rate", outcome_kpi="units_sold",
            label="South fulfilment performance",
            mechanism="SLA performance affects reorder behaviour",
            scope_regions=["South"], scope_product_lines=CORE_LINES,
            did=DidSpec(
                treated_cells=[f"South | {l}" for l in CORE_LINES],
                t0=date(2026, 8, 7), regions=["South", "East", "West"],
                exclude_product_lines=("SmartHome",),
                window_start=date(2026, 6, 15), window_end=date(2026, 8, 25))),
    ]
    return Scenario(
        scenario_id="SC_LOWCONF",
        title="Region South moved and the engine cannot say why",
        subtitle=("A whole source is missing for this region, a second is stale, a new "
                  "product line launched mid-window, and pre-trends diverge"),
        kpi_id="net_revenue",
        focus_window=(date(2026, 8, 7), date(2026, 8, 25)),
        baseline_window=(date(2026, 6, 15), date(2026, 8, 6)),
        scope_regions=["South"], scope_product_lines=None, hypotheses=hyps,
        exposure_rules={h.hypothesis_id: "SEGMENT_CO_MOVEMENT" for h in hyps},
        demonstrates=["R16 low-confidence scenario", "R17 clarification or abstention",
                       "R20 evidence freshness", "R23 confidence calibration"],
        expected_outcome=(
            "Every hypothesis returns INSUFFICIENT_EVIDENCE or ASSOCIATION_ONLY. The "
            "engine abstains and issues one specific, answerable clarification request "
            "addressed to a named owner."))


# ================================================================ SC_SPARSE
def _sparse() -> Scenario:
    return Scenario(
        scenario_id="SC_SPARSE",
        title="A newly launched KPI with 21 days of history",
        subtitle="Subscription ARR launched 2026-08-05; no design is eligible yet",
        kpi_id="subscription_arr",
        focus_window=(date(2026, 8, 12), date(2026, 8, 25)),
        baseline_window=(date(2026, 8, 5), date(2026, 8, 11)),
        scope_regions=ALL_REGIONS, scope_product_lines=None,
        hypotheses=[HypothesisSpec(
            hypothesis_id="H1_ARR_MARKETING",
            driver_id="marketing_spend", outcome_kpi="subscription_arr",
            label="Marketing support for the subscription launch",
            mechanism="paid acquisition drives subscription activations",
            scope_regions=ALL_REGIONS, scope_product_lines=None)],
        exposure_rules={"H1_ARR_MARKETING": "SEGMENT_CO_MOVEMENT"},
        benchmark_kind="NEW_LINE_COHORT",
        demonstrates=["R18 sparse history scenario", "R17 abstention",
                       "R20 evidence freshness"],
        expected_outcome=(
            "INSUFFICIENT_EVIDENCE. No causal estimate. The output is a descriptive "
            "movement, a comparable-cohort benchmark explicitly labelled as a reference "
            "rather than a counterfactual, the data requirement, and the earliest date "
            "a causal read becomes possible."))


# ============================================================== SC_SECURITY
def _security() -> Scenario:
    s = _multifactor()
    s2 = Scenario(
        scenario_id="SC_SECURITY",
        title="The same movement, seen through role entitlements",
        subtitle=("Row, column, grain and domain restrictions applied before retrieval, "
                  "with denied attempts audited"),
        kpi_id="net_revenue", focus_window=config.FOCUS_WINDOW,
        baseline_window=config.BASELINE_WINDOW,
        scope_regions=ALL_REGIONS, scope_product_lines=None,
        hypotheses=s.hypotheses, exposure_rules=s.exposure_rules,
        demonstrates=["R10 KPI access restrictions", "R19 role-based security",
                       "R11 two personas", "R25 LLM boundary"],
        expected_outcome=(
            "The operations manager sees North only at warehouse grain with financial "
            "fields stripped; the CFO sees all regions but no warehouse grain and no "
            "employee fields; unauthorised requests are refused at the API with an "
            "audit event, not hidden in the UI."))
    return s2


SCENARIOS: dict[str, Scenario] = {}
for _s in (_multifactor(), _lowconf(), _sparse(), _security()):
    SCENARIOS[_s.scenario_id] = _s


def get(scenario_id: str) -> Scenario:
    if scenario_id not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario_id!r}; known: {sorted(SCENARIOS)}")
    return SCENARIOS[scenario_id]


def catalogue() -> list[dict]:
    return [s.as_dict() for s in SCENARIOS.values()]
