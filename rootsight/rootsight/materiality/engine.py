"""The Materiality Engine.

Statistical evidence and business materiality are different questions and V5
keeps them on separate axes, then combines them through an explicit matrix
rather than a multiplication.

  * A tightly estimated effect worth 4 lakh a month is statistically strong and
    commercially irrelevant.  Priority LOW.
  * A weakly identified effect worth 6 crore a quarter is commercially urgent
    and statistically unproven.  Priority HIGH - but the action is INVESTIGATE,
    never REMEDIATE, because acting on an unidentified effect is a gamble.

That second case is precisely what a p-value-driven pipeline gets wrong, and it
is the case that costs money.

Business impact is measured against thresholds declared in the KPI semantic
contract (`materiality_threshold`), not against a number chosen in code.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from ..contracts.kpi_contract import registry

STAT_TIERS = ["NONE", "WEAK", "MODERATE", "STRONG"]
IMPACT_TIERS = ["NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "VERY_HIGH"]

# rows = statistical evidence, columns = business impact
PRIORITY_MATRIX = {
    "STRONG":   {"NEGLIGIBLE": "LOW",     "LOW": "LOW",      "MODERATE": "MEDIUM",
                 "HIGH": "HIGH",          "VERY_HIGH": "CRITICAL"},
    "MODERATE": {"NEGLIGIBLE": "LOW",     "LOW": "LOW",      "MODERATE": "MEDIUM",
                 "HIGH": "HIGH",          "VERY_HIGH": "HIGH"},
    "WEAK":     {"NEGLIGIBLE": "NONE",    "LOW": "LOW",      "MODERATE": "LOW",
                 "HIGH": "MEDIUM",        "VERY_HIGH": "HIGH"},
    "NONE":     {"NEGLIGIBLE": "NONE",    "LOW": "NONE",     "MODERATE": "LOW",
                 "HIGH": "MEDIUM",        "VERY_HIGH": "MEDIUM"},
}

# what kind of action the pair licenses
ACTION_MATRIX = {
    ("STRONG", "CRITICAL"): "REMEDIATE",
    ("STRONG", "HIGH"): "REMEDIATE",
    ("STRONG", "MEDIUM"): "REMEDIATE",
    ("MODERATE", "HIGH"): "REMEDIATE_WITH_MONITORING",
    ("MODERATE", "MEDIUM"): "MONITOR",
    ("WEAK", "HIGH"): "INVESTIGATE",
    ("WEAK", "MEDIUM"): "INVESTIGATE",
    ("NONE", "MEDIUM"): "INVESTIGATE",
    ("NONE", "LOW"): "MONITOR",
}


@dataclass
class MaterialityAssessment:
    hypothesis_id: str
    statistical_tier: str
    statistical_basis: dict
    impact_tier: str
    impact_basis: dict
    decision_priority: str
    action_type: str
    exposure: dict
    threshold: dict
    explanation: str
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _statistical_tier(hyp: dict) -> tuple[str, dict]:
    status = hyp.get("causal_status")
    eff = hyp.get("effect", {})
    rob = hyp.get("robustness", {}).get("checks", [])
    verdicts = [c.get("verdict") for c in rob if c.get("verdict")]
    n_fail = sum(1 for v in verdicts if v in ("FAIL", "SENSITIVE"))
    p = eff.get("p_value_primary")
    basis = {"causal_status": status, "p_value_primary": p,
             "p_value_basis": eff.get("p_value_primary_basis"),
             "robustness_verdicts": verdicts,
             "robustness_failures": n_fail,
             "rule": ("STRONG requires an identified design, p<0.05 on the primary "
                      "inference, and no failed robustness check. Statistical strength "
                      "never substitutes for identification: an unidentified effect "
                      "cannot be STRONG at any p-value.")}
    if status == "SUPPORTED_BY_INTERVENTION":
        return "STRONG", basis
    if status == "SUPPORTED_BY_DESIGN":
        if p is not None and p < 0.05 and n_fail == 0:
            return "STRONG", basis
        if p is not None and p < 0.10 and n_fail <= 1:
            return "MODERATE", basis
        return "WEAK", basis
    if status == "NOT_POINT_IDENTIFIED":
        return "WEAK", basis
    if status == "ASSOCIATION_ONLY":
        return "WEAK", basis
    return "NONE", basis


def _impact_tier(exposure_inr_per_day: float, threshold: dict, segment_base_per_day: float,
                 days_active: int) -> tuple[str, dict]:
    ann = threshold.get("annualised_multiplier", 12)
    monthly = abs(exposure_inr_per_day) * 30.0
    annualised = abs(exposure_inr_per_day) * 30.0 * ann
    pct_of_segment = (100.0 * abs(exposure_inr_per_day) / segment_base_per_day
                      if segment_base_per_day else float("nan"))
    abs_thr = float(threshold.get("absolute_inr", 0))
    pct_thr = float(threshold.get("pct_of_segment_revenue", 0))
    ratio_abs = monthly / abs_thr if abs_thr else 0.0
    ratio_pct = pct_of_segment / pct_thr if pct_thr else 0.0
    ratio = max(ratio_abs, ratio_pct)
    if ratio >= 4.0:
        tier = "VERY_HIGH"
    elif ratio >= 2.0:
        tier = "HIGH"
    elif ratio >= 1.0:
        tier = "MODERATE"
    elif ratio >= 0.4:
        tier = "LOW"
    else:
        tier = "NEGLIGIBLE"
    basis = {
        "exposure_inr_per_day": round(exposure_inr_per_day, 2),
        "exposure_inr_per_month": round(monthly, 2),
        "exposure_annualised_inr": round(annualised, 2),
        "pct_of_segment_daily_revenue": (None if np.isnan(pct_of_segment)
                                         else round(pct_of_segment, 3)),
        "days_observed_active": days_active,
        "threshold_absolute_inr_per_month": abs_thr,
        "threshold_pct_of_segment": pct_thr,
        "ratio_to_threshold": round(ratio, 3),
        "rule": ("tier is set by the larger of (monthly exposure / absolute threshold) "
                 "and (share of segment revenue / percentage threshold), both declared "
                 "in the KPI semantic contract"),
    }
    return tier, basis


class MaterialityEngine:
    def __init__(self):
        self.reg = registry()

    def assess(self, hyp: dict, *, exposure_inr_per_day: float,
               segment_base_per_day: float, days_active: int,
               exposure_derivation: str) -> MaterialityAssessment:
        k = self.reg.get(hyp["outcome_kpi"])
        stat_tier, stat_basis = _statistical_tier(hyp)
        imp_tier, imp_basis = _impact_tier(exposure_inr_per_day, k.materiality_threshold,
                                           segment_base_per_day, days_active)
        priority = PRIORITY_MATRIX[stat_tier][imp_tier]
        action = ACTION_MATRIX.get((stat_tier, priority))
        if action is None:
            action = ("REMEDIATE" if stat_tier == "STRONG" and priority in ("HIGH", "CRITICAL")
                      else "INVESTIGATE" if priority in ("HIGH", "CRITICAL", "MEDIUM")
                      else "MONITOR" if priority == "LOW" else "NO_ACTION")

        warns: list[str] = []
        if stat_tier in ("WEAK", "NONE") and priority in ("HIGH", "CRITICAL", "MEDIUM"):
            warns.append(
                "High business exposure with weak statistical identification. The "
                "licensed action is to INVESTIGATE (buy information), not to REMEDIATE "
                "(spend on a mechanism that has not been established).")
        if stat_tier == "STRONG" and priority in ("LOW", "NONE"):
            warns.append(
                "The effect is well identified but too small to matter commercially. "
                "Statistical significance is not a reason to act.")

        explanation = (
            f"Statistical evidence: {stat_tier}. Business impact: {imp_tier} "
            f"({imp_basis['ratio_to_threshold']}x the contract materiality threshold). "
            f"Decision priority {priority} from the explicit matrix - the two axes are "
            f"never multiplied into a single number. Licensed action: {action}.")

        return MaterialityAssessment(
            hypothesis_id=hyp["hypothesis_id"], statistical_tier=stat_tier,
            statistical_basis=stat_basis, impact_tier=imp_tier, impact_basis=imp_basis,
            decision_priority=priority, action_type=action,
            exposure={"inr_per_day": round(exposure_inr_per_day, 2),
                      "derivation": exposure_derivation,
                      "is_causal": hyp.get("effect", {}).get("is_causal", False)},
            threshold=dict(k.materiality_threshold), explanation=explanation,
            warnings=warns)
