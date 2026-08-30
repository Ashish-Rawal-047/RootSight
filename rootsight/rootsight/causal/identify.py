"""Identification: deciding what may be claimed, before anything is estimated.

The V5 correction to V4 is that identification is a THREE-LAYER conjunction, and
all three must hold before a causal effect is reported:

  L1  GRAPHICAL     does a valid adjustment set / front door / instrument exist
                    in the declared graph, using only observed variables?
  L2  DESIGN        is there a quasi-experimental design whose own assumptions
                    are compatible with this data?  DID and ITS are assessed
                    INDEPENDENTLY.  A failed DID does not hand the question to
                    ITS; ITS must qualify on its own terms or not run.
  L3  DATA          is the driver measured at a resolution and coverage that can
                    support the claim?

Statuses produced:

  SUPPORTED_BY_DESIGN     all three layers hold; a point estimate with a
                          confidence interval is reported, conditional on the
                          named assumptions
  NOT_POINT_IDENTIFIED    the association is real but no design identifies a
                          point effect.  NO INTERVAL IS INVENTED.  V4 fabricated
                          "-2.1pp to -4.8pp" from an E-value; V5 reports the
                          association, names what is missing, and stops.
  ASSOCIATION_ONLY        temporal compatibility failed, or the driver's grain
                          cannot support the required lag
  INSUFFICIENT_EVIDENCE   the sufficiency gate failed
  SUPPORTED_BY_INTERVENTION  a post-intervention design confirmed the hypothesis

There is no E-value anywhere in V5.  The E-value is defined on a risk-ratio
scale for binary exposures and outcomes; applying it to a percentage-point
change in a continuous KPI, as V4 did, is not a valid use of the statistic.
Robustness is instead evidenced by a placebo test, an alternative specification,
a leave-one-control-out range and a negative-control outcome - all of which are
computed on the actual estimator being used.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .. import config
from ..contracts.kpi_contract import Driver, KpiDefinition, registry
from .dag import CausalDAG, find_identification_strategy
from .estimate import DidResult, ItsResult, estimate_did, estimate_its
from .gates import (SufficiencyResult, TemporalCompatibility, data_sufficiency,
                    temporal_compatibility)
from .panel import Panel, PanelBuilder
from .structure import StructuralEvidence

STATUS_SUPPORTED = "SUPPORTED_BY_DESIGN"
STATUS_NOT_POINT = "NOT_POINT_IDENTIFIED"
STATUS_ASSOC = "ASSOCIATION_ONLY"
STATUS_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
STATUS_INTERVENTION = "SUPPORTED_BY_INTERVENTION"


# --------------------------------------------------------------------- specs
@dataclass
class DidSpec:
    treated_cells: list[str]
    t0: date
    regions: list[str]
    exclude_product_lines: tuple[str, ...] = ()
    outcome_col: str = "units_sold"
    window_start: date | None = None
    window_end: date | None = None
    negative_control_outcome: str = "list_price"


@dataclass
class ItsSpec:
    regions: list[str]
    t0: date
    product_lines: list[str] | None = None
    series_col: str = "units_sold"
    window_start: date | None = None
    window_end: date | None = None


@dataclass
class HypothesisSpec:
    hypothesis_id: str
    driver_id: str
    outcome_kpi: str
    label: str
    mechanism: str
    scope_regions: list[str]
    scope_product_lines: list[str] | None = None
    did: DidSpec | None = None
    its: ItsSpec | None = None
    unit_conversion_note: str = ""


@dataclass
class DesignAssessment:
    design: str
    eligible: bool
    checks: list[dict]
    reason: str
    assessed_independently: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Hypothesis:
    hypothesis_id: str
    driver_id: str
    outcome_kpi: str
    label: str
    mechanism: str
    causal_status: str
    scope: dict
    sufficiency: dict
    temporal: dict
    structural_support: str
    structural_detail: dict
    graphical: dict
    designs: list[dict]
    chosen_design: str | None
    effect: dict
    robustness: dict
    assumptions: list[dict]
    missing: dict
    evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    ewhr: dict = field(default_factory=dict)
    materiality: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def has_point_estimate(self) -> bool:
        return self.effect.get("kind") == "POINT_WITH_CI"


# ---------------------------------------------------------------- evaluator
class HypothesisEvaluator:
    def __init__(self, panels: PanelBuilder, dag: CausalDAG,
                 structural: StructuralEvidence,
                 concurrent_events: list[dict] | None = None):
        self.pb = panels
        self.dag = dag
        self.structural = structural
        self.concurrent = concurrent_events or []
        self.reg = registry()

    # ------------------------------------------------------------------ main
    def evaluate(self, spec: HypothesisSpec, *, focus: tuple[date, date],
                 baseline: tuple[date, date]) -> Hypothesis:
        panel = self.pb.timeseries(regions=spec.scope_regions,
                                   product_lines=spec.scope_product_lines)
        f = panel.frame
        # net_revenue movements are analysed through the volume channel, because a
        # driver such as fulfilment acts on units, and units are converted to
        # revenue afterwards by deterministic arithmetic
        outcome_col = ("units_sold" if spec.outcome_kpi in ("net_revenue", "units_sold")
                       else spec.outcome_kpi)
        driver_col = spec.driver_id
        notes: list[str] = []
        missing: dict = {"unavailable_variables": [], "stale_sources": [],
                         "coverage": {}, "insufficient_history": []}

        if outcome_col not in f.columns:
            missing["unavailable_variables"].append(outcome_col)
            return self._insufficient(
                spec, panel,
                reason=(f"the outcome series {outcome_col} is not materialised in the "
                        f"analysis panel for scope {spec.scope_regions}, so no driver "
                        f"analysis can be attempted against it"), missing=missing)
        if driver_col not in f.columns or driver_col in panel.unavailable:
            missing["unavailable_variables"].append(driver_col)
            return self._insufficient(
                spec, panel, reason=(f"{driver_col} is not available in scope "
                                     f"{spec.scope_regions}"), missing=missing)

        cov = panel.coverage.get(driver_col, {})
        missing["coverage"][driver_col] = cov
        res = panel.temporal_resolution.get(driver_col, 1.0)

        # The temporal analysis runs on the ANALYSIS SPAN (baseline start to focus
        # end), not on all available history.  Diluting a three-week disruption
        # across eight months of data hides it; extending the span beyond the
        # question being asked imports regimes that are not under investigation.
        span = f[(f["date"] >= baseline[0]) & (f["date"] <= focus[1])].reset_index(drop=True)
        pre = span[(span["date"] >= baseline[0]) & (span["date"] <= baseline[1])]
        post = span[(span["date"] >= focus[0]) & (span["date"] <= focus[1])]
        paired = lambda g: int((g[driver_col].notna() & g[outcome_col].notna()).sum())
        x_all = span[driver_col].to_numpy(dtype=float)
        y_all = span[outcome_col].to_numpy(dtype=float)
        span_dates = list(span["date"])
        miss_frac = float(np.mean(~np.isfinite(x_all)))

        suff = data_sufficiency(y_all, x_all, pre_n=paired(pre), post_n=paired(post),
                                missing_frac=miss_frac)
        drv: Driver | None = next(
            (d for d in self.reg.get(spec.outcome_kpi).drivers
             if d.driver_id == spec.driver_id), None)
        grain_note = None
        if drv and drv.prior_lag_days and res > drv.prior_lag_days[0]:
            grain_note = (
                f"Separately from data volume: {driver_col} is delivered at "
                f"{res:.0f}-day resolution while the contract's expected lag range for "
                f"it starts at {drv.prior_lag_days[0]} day(s). Even with complete "
                f"coverage, a lag shorter than {res:.0f} days is not measurable for "
                f"this driver, so the instrumentation - not the history length - is the "
                f"binding constraint.")
        if not suff.passed:
            return self._insufficient(
                spec, panel,
                reason="data sufficiency gate failed: " + ", ".join(suff.failure_codes),
                missing=missing, sufficiency=suff.as_dict(), extra_note=grain_note)
        temporal = temporal_compatibility(x_all, y_all, driver_meta=drv,
                                          resolution_days=res, dates=span_dates)

        struct_level = self.structural.support_level(spec.driver_id, outcome_col)
        struct_detail = (self.structural.verdict_for(spec.driver_id, outcome_col).as_dict()
                         if self.structural.verdict_for(spec.driver_id, outcome_col)
                         else {"verdict": "NOT_TESTABLE"})

        graphical = find_identification_strategy(self.dag, spec.driver_id, outcome_col)

        # --------------------------- L2: designs, assessed independently -----
        designs: list[DesignAssessment] = []
        did_res: DidResult | None = None
        its_res: ItsResult | None = None
        if spec.did:
            a, did_res = self._assess_did(spec)
            designs.append(a)
        if spec.its:
            a, its_res = self._assess_its(spec)
            designs.append(a)
        if not designs:
            designs.append(DesignAssessment(
                design="NONE", eligible=False, checks=[],
                reason=("no quasi-experimental design is defined for this driver: the "
                        "driver has no unit-localised onset (no treated/control split) "
                        "and no isolated intervention date")))

        # --------------------------- status decision -------------------------
        status, chosen, effect, why = self._decide(
            spec, temporal, graphical, designs, did_res, its_res, panel, res)
        notes.extend(why)
        if grain_note:
            notes.append(grain_note)

        robustness = self._robustness(spec, chosen, did_res, its_res)
        assumptions = self._assumptions(spec, temporal, graphical, designs, did_res,
                                        its_res, struct_level, panel, chosen)
        # a violated assumption can only downgrade, never upgrade
        status, downgrade_note = self._apply_assumption_downgrades(status, assumptions)
        if downgrade_note:
            notes.append(downgrade_note)
            if status != STATUS_SUPPORTED and effect.get("kind") == "POINT_WITH_CI":
                effect = self._demote_effect(effect, downgrade_note)

        for sid, fr in panel.coverage.items():
            if fr.get("status") in ("SPARSE", "ABSENT"):
                missing["coverage"][sid] = fr
        if driver_col in panel.unavailable:
            missing["unavailable_variables"].append(driver_col)

        return Hypothesis(
            hypothesis_id=spec.hypothesis_id, driver_id=spec.driver_id,
            outcome_kpi=spec.outcome_kpi, label=spec.label, mechanism=spec.mechanism,
            causal_status=status, scope=panel.scope, sufficiency=suff.as_dict(),
            temporal=temporal.as_dict(), structural_support=struct_level,
            structural_detail=struct_detail, graphical=graphical.as_dict(),
            designs=[d.as_dict() for d in designs], chosen_design=chosen,
            effect=effect, robustness=robustness, assumptions=assumptions,
            missing=missing, notes=notes)

    # ------------------------------------------------------------------ DID
    def _assess_did(self, spec: HypothesisSpec) -> tuple[DesignAssessment, DidResult | None]:
        s = spec.did
        up = self.pb.unit_panel(regions=s.regions,
                                exclude_product_lines=s.exclude_product_lines,
                                start=s.window_start, end=s.window_end,
                                outcome=s.outcome_col)
        cells = sorted(up["cell"].unique())
        treated = [c for c in s.treated_cells if c in cells]
        control = [c for c in cells if c not in treated]
        pre_days = int(up[up["date"] < s.t0]["date"].nunique())
        post_days = int(up[up["date"] >= s.t0]["date"].nunique())
        g = config.GATES

        checks = [
            {"check": "treated_units", "value": len(treated),
             "minimum": g.min_did_treated_units, "passed": len(treated) >= g.min_did_treated_units},
            {"check": "control_units", "value": len(control),
             "minimum": g.min_did_control_units, "passed": len(control) >= g.min_did_control_units},
            {"check": "pre_periods", "value": pre_days, "minimum": g.min_did_pre_periods,
             "passed": pre_days >= g.min_did_pre_periods},
            {"check": "post_periods", "value": post_days, "minimum": 3,
             "passed": post_days >= 3},
            {"check": "treatment_timing_known", "value": s.t0.isoformat(), "passed": True,
             "note": "onset taken from the operational changepoint, not from the outcome series"},
        ]
        if not all(c["passed"] for c in checks):
            failed = [c["check"] for c in checks if not c["passed"]]
            return DesignAssessment(
                design="DID", eligible=False, checks=checks,
                reason=f"DID structural requirements not met: {failed}"), None

        res = estimate_did(up, treated_cells=treated, t0=s.t0, outcome="outcome")
        res.outcome = s.outcome_col          # report the KPI name, not the panel column
        pt = res.parallel_trends
        checks.append({
            "check": "parallel_trends_pre_test",
            "value": {"p_value": pt.get("p_value"), "verdict": pt.get("verdict")},
            "passed": pt.get("verdict") == "COMPATIBLE",
            "note": "a screen, not a proof: failing to reject is not evidence of parallelism"})
        checks.append({
            "check": "no_anticipation",
            "value": "checked by the pre-trend interaction above",
            "passed": pt.get("verdict") == "COMPATIBLE",
            "note": "anticipation would show as a treated-specific pre-trend"})
        conc = self._concurrent_near(s.t0, exclude=spec.driver_id,
                                     outcome=spec.outcome_kpi)
        checks.append({
            "check": "concurrent_shocks_absorbed",
            "value": {"concurrent_events": [c["driver_id"] for c in conc],
                      "absorbed_by": res.fixed_effects},
            "passed": True,
            "note": ("national and product-line shocks in the window are absorbed by "
                     "the product-line x day fixed effects, which is why they do not "
                     "invalidate this comparison")})
        checks.append({
            "check": "spillover_between_treated_and_control",
            "value": "not verifiable from data; declared as an assumption",
            "passed": True,
            "note": "SUTVA across regions is asserted by the operations owner, not tested"})

        eligible = all(c["passed"] for c in checks)
        return DesignAssessment(
            design="DID", eligible=eligible, checks=checks,
            reason=("all DID requirements and its own pre-tests are satisfied"
                    if eligible else
                    "DID pre-tests failed: " +
                    ", ".join(c["check"] for c in checks if not c["passed"]))), res

    # ------------------------------------------------------------------ ITS
    def _assess_its(self, spec: HypothesisSpec) -> tuple[DesignAssessment, ItsResult | None]:
        s = spec.its
        p = self.pb.timeseries(regions=s.regions, product_lines=s.product_lines,
                               start=s.window_start, end=s.window_end)
        f = p.frame
        y = f[s.series_col].to_numpy(dtype=float)
        dates = list(f["date"])
        pre_n = sum(1 for d in dates if d < s.t0)
        post_n = sum(1 for d in dates if d >= s.t0)
        g = config.GATES

        checks = [
            {"check": "intervention_date_known", "value": s.t0.isoformat(), "passed": True},
            {"check": "pre_period_points", "value": pre_n, "minimum": g.min_its_pre_points,
             "passed": pre_n >= g.min_its_pre_points},
            {"check": "post_period_points", "value": post_n, "minimum": g.min_its_post_points,
             "passed": post_n >= g.min_its_post_points},
        ]
        conc = self._concurrent_near(s.t0, exclude=spec.driver_id,
                                     outcome=spec.outcome_kpi)
        checks.append({
            "check": "no_concurrent_intervention",
            "value": [c["driver_id"] for c in conc],
            "passed": len(conc) == 0,
            "note": ("ITS has no control group, so any other shock near the "
                     "intervention date is indistinguishable from the intervention. "
                     "This is assessed on its own terms - a failed DID does not make "
                     "ITS acceptable here.")})
        if not all(c["passed"] for c in checks):
            failed = [c["check"] for c in checks if not c["passed"]]
            return DesignAssessment(
                design="ITS", eligible=False, checks=checks,
                reason=("ITS is not independently valid on this series: " + str(failed))), None

        res = estimate_its(y, dates, s.t0, holiday=f["is_holiday"].to_numpy(),
                           promo=f["promo_calendar"].to_numpy())
        checks.append({"check": "placebo_pre_period",
                       "value": {"p_value": res.placebo.get("p_value"),
                                 "verdict": res.placebo.get("verdict")},
                       "passed": res.placebo.get("verdict") == "PASS"})
        checks.append({"check": "residual_autocorrelation_handled",
                       "value": {"durbin_watson": res.diagnostics["durbin_watson"],
                                 "ljung_box_p": res.diagnostics["ljung_box_p"],
                                 "hac_lags": res.hac_lags},
                       "passed": res.diagnostics["ljung_box_p"] > 0.01,
                       "note": "HAC standard errors applied; severe autocorrelation "
                               "still invalidates the trend extrapolation"})
        eligible = all(c["passed"] for c in checks)
        return DesignAssessment(
            design="ITS", eligible=eligible, checks=checks,
            reason=("ITS is independently valid: isolated intervention date, adequate "
                    "pre-period, placebo passes"
                    if eligible else
                    "ITS failed its own checks: " +
                    ", ".join(c["check"] for c in checks if not c["passed"]))), res

    def _concurrent_near(self, t0: date, *, exclude: str, outcome: str = "",
                         window: int = 6) -> list[dict]:
        """Other shocks landing near the intervention date.

        The outcome itself is excluded (of course it moved), and so is anything
        downstream of the driver in the declared graph: a mediator moving is the
        mechanism operating, not a competing explanation.  Counting mediators as
        concurrent shocks would make every real mechanism look confounded.
        """
        downstream = self.dag.descendants(exclude) if exclude in self.dag.nodes else set()
        blocked = {exclude, outcome} | downstream
        return [c for c in self.concurrent
                if c.get("driver_id") not in blocked and c.get("changepoint")
                and abs((c["changepoint"] - t0).days) <= window]

    # -------------------------------------------------------------- decision
    def _decide(self, spec, temporal, graphical, designs, did_res, its_res,
                panel, resolution) -> tuple[str, str | None, dict, list[str]]:
        why: list[str] = []
        if not temporal.passed:
            codes = temporal.failure_codes
            if "GRAIN_TOO_COARSE" in codes:
                why.append(
                    f"{spec.driver_id} is measured at {resolution:.0f}-day resolution, "
                    f"but the hypothesis requires a lag of {temporal.best_lag} day(s). "
                    "Allocating the coarse series across days does not create the "
                    "resolution the claim needs, so no causal effect is estimated.")
            else:
                why.append("temporal compatibility failed: " + ", ".join(codes))
            return (STATUS_ASSOC, None,
                    self._association_effect(temporal, "temporal compatibility failed"),
                    why)

        if graphical.strategy == "NONE":
            why.append(
                "no observed adjustment set, front door or instrument identifies this "
                f"effect in the declared graph: {graphical.reason}")
            return (STATUS_NOT_POINT, None,
                    self._association_effect(temporal, graphical.reason), why)

        eligible = [d for d in designs if d.eligible]
        if not eligible:
            reasons = "; ".join(f"{d.design}: {d.reason}" for d in designs)
            why.append("no quasi-experimental design qualifies (" + reasons + ")")
            return (STATUS_NOT_POINT, None,
                    self._association_effect(temporal, reasons), why)

        # preference order: DID (has a control group) then ITS
        order = {"DID": 0, "ITS": 1}
        chosen = sorted(eligible, key=lambda d: order.get(d.design, 9))[0].design
        why.append(
            f"{chosen} qualifies on its own assumptions; identification is "
            f"{graphical.strategy} in the declared graph and the driver's measurement "
            "resolution supports the estimated lag")
        if chosen == "DID" and did_res is not None:
            return STATUS_SUPPORTED, "DID", self._did_effect(spec, did_res), why
        if chosen == "ITS" and its_res is not None:
            return STATUS_SUPPORTED, "ITS", self._its_effect(spec, its_res), why
        return (STATUS_NOT_POINT, None,
                self._association_effect(temporal, "estimator failed to produce a result"),
                why)

    @staticmethod
    def _association_effect(temporal: TemporalCompatibility, reason: str) -> dict:
        return {
            "kind": "NO_POINT_ESTIMATE",
            "point_estimate": None,
            "confidence_interval": None,
            "interval_withheld_reason": (
                "No formally valid bounding model applies to this hypothesis, so no "
                "interval is reported. Publishing an interval here would be a "
                "fabrication. " + reason),
            "observed_association": {
                "measure": "cross-correlation of detrended series at the best lag",
                "value": (None if np.isnan(temporal.best_corr)
                          else round(temporal.best_corr, 4)),
                "lag_days": temporal.best_lag,
                "lag_bootstrap_ci": list(temporal.lag_ci) if temporal.lag_ci else None,
                "is_causal": False,
            },
            "what_would_identify_it": [],
        }

    @staticmethod
    def _did_effect(spec: HypothesisSpec, r: DidResult) -> dict:
        return {
            "kind": "POINT_WITH_CI",
            "estimator": r.estimator,
            "estimand": ("ATT on the treated cells: average change in "
                         f"{r.outcome} per cell per day attributable to the treatment, "
                         "relative to the contemporaneous control cells"),
            "point_estimate": round(r.att_per_unit_day, 4),
            "unit": f"{r.outcome} per cell per day",
            "point_estimate_pct": round(r.att_pct_of_treated_base, 3),
            "confidence_interval": [round(x, 4) for x in r.ci95],
            "ci_basis": "cluster-robust by cell, t(G-1)",
            "p_value_primary": round(r.p_randomisation, 5),
            "p_value_primary_basis": (
                f"exact randomisation inference over {r.n_permutations} assignments"),
            "p_value_cluster": round(r.p_cluster, 6),
            "n_clusters": r.n_clusters,
            "fixed_effects": r.fixed_effects,
            "conditional_on": [
                "parallel trends between treated and control cells",
                "no anticipation before the onset date",
                "no spillover from treated to control regions (SUTVA)",
                "stable cell composition across the window",
            ],
            "unit_conversion_note": spec.unit_conversion_note,
            "is_causal": True,
        }

    @staticmethod
    def _its_effect(spec: HypothesisSpec, r: ItsResult) -> dict:
        return {
            "kind": "POINT_WITH_CI",
            "estimator": r.estimator,
            "estimand": "level change in the series at the intervention date",
            "point_estimate": round(r.level_change, 4),
            "unit": "series units per day",
            "confidence_interval": [round(x, 4) for x in r.level_ci95],
            "ci_basis": f"Newey-West HAC, {r.hac_lags} lags, Bartlett kernel",
            "p_value_primary": round(r.level_p, 6),
            "p_value_primary_basis": "HAC t-test",
            "cumulative_effect": round(r.cumulative_effect, 2),
            "cumulative_pct": round(r.cumulative_pct, 3),
            "conditional_on": [
                "the pre-period trend would have continued absent the intervention",
                "no other shock occurred near the intervention date",
                "the intervention is a level (and possibly slope) change",
            ],
            "unit_conversion_note": spec.unit_conversion_note,
            "is_causal": True,
        }

    @staticmethod
    def _demote_effect(effect: dict, reason: str) -> dict:
        return {
            "kind": "NO_POINT_ESTIMATE",
            "point_estimate": None,
            "confidence_interval": None,
            "interval_withheld_reason": reason,
            "withdrawn_estimate": {k: effect.get(k) for k in
                                   ("estimator", "point_estimate", "confidence_interval")},
            "observed_association": effect.get("observed_association"),
            "is_causal": False,
        }

    # ------------------------------------------------------------ robustness
    def _robustness(self, spec: HypothesisSpec, chosen: str | None,
                    did_res: DidResult | None, its_res: ItsResult | None) -> dict:
        out: dict = {
            "note": ("V5 removes the E-value. It is defined on a risk-ratio scale for "
                     "binary exposure and outcome; V4 applied it to a continuous "
                     "percentage-point KPI change, which is not a valid use. Robustness "
                     "below is computed on the estimator actually used."),
            "checks": [],
        }
        if chosen == "DID" and did_res is not None and spec.did:
            s = spec.did
            up = self.pb.unit_panel(regions=s.regions,
                                    exclude_product_lines=s.exclude_product_lines,
                                    start=s.window_start, end=s.window_end,
                                    outcome=s.outcome_col)
            base = did_res.att_per_unit_day

            alt = estimate_did(up, treated_cells=did_res.treated_cells, t0=s.t0,
                               outcome="outcome", absorb_line_by_day=False)
            out["checks"].append({
                "check": "alternative_specification",
                "description": "day fixed effects instead of product-line x day",
                "estimate": round(alt.att_per_unit_day, 4),
                "baseline_estimate": round(base, 4),
                "relative_change_pct": round(100 * (alt.att_per_unit_day / base - 1), 2)
                if base else None,
                "verdict": ("STABLE" if base and abs(alt.att_per_unit_day / base - 1) < 0.30
                            else "SENSITIVE"),
            })

            loo = []
            for drop in did_res.control_cells:
                keep = up[up["cell"] != drop]
                r = estimate_did(keep, treated_cells=did_res.treated_cells, t0=s.t0,
                                 outcome="outcome")
                loo.append({"dropped_control_cell": drop,
                            "estimate": round(r.att_per_unit_day, 4)})
            vals = [d["estimate"] for d in loo]
            out["checks"].append({
                "check": "leave_one_control_out",
                "description": "re-estimate dropping each control cell in turn",
                "range": [round(min(vals), 4), round(max(vals), 4)] if vals else None,
                "detail": loo,
                "verdict": ("STABLE" if vals and (max(vals) - min(vals)) < abs(base) * 0.5
                            else "SENSITIVE"),
                "note": ("This is a stability range for the estimate under control-group "
                         "choice. It is NOT a partial-identification bound and must not "
                         "be reported as one."),
            })

            neg_col = s.negative_control_outcome
            if neg_col in up.columns:
                neg_panel = up.copy()
                neg_panel["outcome"] = neg_panel[neg_col]
                nr = estimate_did(neg_panel, treated_cells=did_res.treated_cells, t0=s.t0,
                                  outcome="outcome")
                out["checks"].append({
                    "check": "negative_control_outcome",
                    "description": (f"same design applied to {neg_col}, which the "
                                    "mechanism cannot affect (list price is set "
                                    "centrally, not by a warehouse)"),
                    "estimate": round(nr.att_per_unit_day, 6),
                    "p_randomisation": round(nr.p_randomisation, 4),
                    "verdict": "PASS" if nr.p_randomisation > 0.10 else "FAIL",
                    "note": ("A significant effect on an outcome the mechanism cannot "
                             "touch would mean the design is picking up something else."),
                })
        if chosen == "ITS" and its_res is not None:
            out["checks"].append({
                "check": "placebo_pre_period", **its_res.placebo,
            })
            out["checks"].append({
                "check": "residual_diagnostics", **its_res.diagnostics,
            })
        if not out["checks"]:
            out["checks"].append({
                "check": "none_applicable",
                "description": "no point estimate was produced, so there is nothing to "
                               "stress-test",
                "verdict": "N/A"})
        return out

    # ------------------------------------------------------------ assumptions
    def _assumptions(self, spec, temporal, graphical, designs, did_res, its_res,
                     struct_level, panel, chosen) -> list[dict]:
        def row(aid, name, requires, status, evidence, testable):
            return {"id": aid, "name": name, "requires": requires, "status": status,
                    "evidence": evidence, "testable_from_this_data": testable}

        rows = [row(
            "A1", "Temporal compatibility",
            "the driver moves before the KPI, at a lag the data can resolve",
            "SATISFIED" if temporal.passed else "VIOLATED",
            (f"best lag {temporal.best_lag}d, |corr| {abs(temporal.best_corr):.2f}, "
             f"subsample lag range {temporal.lag_ci}, driver resolution "
             f"{temporal.driver_resolution_days:.0f}d"),
            True)]

        unobs = graphical.unobserved_required
        rows.append(row(
            "A2", "No unblocked confounding",
            "a valid adjustment set exists using observed variables only",
            "SATISFIED" if graphical.strategy != "NONE" else "VIOLATED",
            (f"strategy={graphical.strategy}, adjustment set={graphical.adjustment_set}"
             if graphical.strategy != "NONE"
             else f"requires unobserved {unobs}: {graphical.reason}"),
            True))

        flags = (temporal.detail or {}).get("review_flags") or []
        if flags:
            rows.append(row(
                "A1b", "Observed lag matches the declared prior",
                "the measured lag falls inside the contract's expected range",
                "UNKNOWN",
                (f"raised for human review: {', '.join(flags)}; measured lag "
                 f"{temporal.best_lag}d against contract prior "
                 f"{temporal.required_lag_range}"),
                True))
        rows.append(row(
            "A3", "Structural consistency",
            "the declared edge is not contradicted by the data",
            {"SUPPORTED": "SATISFIED", "CONTRADICTED": "VIOLATED",
             "INCONCLUSIVE": "UNKNOWN", "NOT_TESTABLE": "UNKNOWN"}[struct_level],
            f"structure screen verdict: {struct_level}", True))

        def scoped(design: str, status: str) -> str:
            """A design's assumptions bind only when that design is the one used."""
            return status if chosen == design else "NOT_APPLICABLE"

        did_a = next((d for d in designs if d.design == "DID"), None)
        if did_a:
            pt = (did_res.parallel_trends if did_res else {})
            rows.append(row(
                "A4", "Parallel trends (DID)",
                "treated and control cells would have moved together absent treatment",
                scoped("DID", "SATISFIED" if pt.get("verdict") == "COMPATIBLE" else
                       "VIOLATED" if pt.get("verdict") == "INCOMPATIBLE" else "UNKNOWN"),
                (f"pre-trend interaction p={pt.get('p_value')} "
                 f"({pt.get('coefficient_pct_of_base_per_day')}%/day)"
                 if pt.get("tested") else "not tested") +
                ("" if chosen == "DID" else "  [DID was not the design used]"),
                True))
            rows.append(row(
                "A5", "No spillover between treated and control (SUTVA)",
                "treating the affected region does not change the control regions",
                scoped("DID", "ASSUMED"),
                "not testable from observational data; asserted by the operations owner",
                False))
        its_a = next((d for d in designs if d.design == "ITS"), None)
        if its_a:
            conc = [c["check"] for c in its_a.checks
                    if c["check"] == "no_concurrent_intervention" and not c["passed"]]
            rows.append(row(
                "A6", "No concurrent intervention (ITS)",
                "no other shock lands near the intervention date",
                scoped("ITS", "VIOLATED" if conc else "SATISFIED"),
                next((str(c["value"]) for c in its_a.checks
                      if c["check"] == "no_concurrent_intervention"), "") +
                ("" if chosen == "ITS" else
                 "  [ITS was not the design used; a control group removes this "
                 "requirement because a common shock differences out]"),
                True))
            if its_res:
                rows.append(row(
                    "A7", "Pre-trend extrapolation valid (ITS)",
                    "the pre-period trend is a credible counterfactual",
                    scoped("ITS",
                           "SATISFIED" if its_res.placebo.get("verdict") == "PASS"
                           else "VIOLATED" if its_res.placebo.get("run") else "UNKNOWN"),
                    f"placebo at {its_res.placebo.get('pseudo_t0')}: "
                    f"p={its_res.placebo.get('p_value')}", True))

        worst_cov = min((v.get("observed_pct", 100.0) for v in panel.coverage.values()
                         if isinstance(v, dict)), default=100.0)
        rows.append(row(
            "A8", "Measurement adequacy",
            "the driver and KPI are measured without systematic gaps in the window",
            "SATISFIED" if worst_cov > 90 else "UNKNOWN" if worst_cov > 50 else "VIOLATED",
            f"lowest column coverage in scope: {worst_cov:.1f}%", True))
        return rows

    @staticmethod
    def _apply_assumption_downgrades(status: str,
                                     assumptions: list[dict]) -> tuple[str, str | None]:
        by = {a["id"]: a["status"] for a in assumptions
              if a["status"] != "NOT_APPLICABLE"}
        if status != STATUS_SUPPORTED:
            return status, None
        if by.get("A1") == "VIOLATED":
            return STATUS_ASSOC, ("A1 violated: temporal compatibility does not hold, so "
                                  "no causal claim is permitted")
        for aid, label in (("A2", "unblocked confounding"),
                           ("A4", "parallel trends"),
                           ("A6", "concurrent intervention"),
                           ("A7", "pre-trend extrapolation")):
            if by.get(aid) == "VIOLATED":
                return STATUS_NOT_POINT, (
                    f"{aid} violated ({label}): the point estimate is withdrawn. No "
                    "replacement interval is published because no valid bounding model "
                    "applies.")
        if by.get("A3") == "VIOLATED":
            return STATUS_NOT_POINT, (
                "A3 violated: the data contradict the declared structural edge, so the "
                "estimate is withdrawn")
        return status, None

    # ----------------------------------------------------------- insufficient
    def _insufficient(self, spec: HypothesisSpec, panel: Panel, *, reason: str,
                      missing: dict, sufficiency: dict | None = None,
                      extra_note: str | None = None) -> Hypothesis:
        return Hypothesis(
            hypothesis_id=spec.hypothesis_id, driver_id=spec.driver_id,
            outcome_kpi=spec.outcome_kpi, label=spec.label, mechanism=spec.mechanism,
            causal_status=STATUS_INSUFFICIENT, scope=panel.scope,
            sufficiency=sufficiency or {}, temporal={}, structural_support="NOT_TESTABLE",
            structural_detail={}, graphical={}, designs=[], chosen_design=None,
            effect={"kind": "NO_ESTIMATE", "point_estimate": None,
                    "confidence_interval": None,
                    "interval_withheld_reason": reason},
            robustness={"checks": [{"check": "none_applicable", "verdict": "N/A"}]},
            assumptions=[], missing=missing, notes=[reason])
