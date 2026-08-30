"""Statistical structure screen.

V4 claimed FCI -> PAG as the structural step and then leaned on the PAG for
identification.  V5 makes two changes:

  1. The structure step is a SCREEN, not a discovery step.  The graph comes from
     the KPI semantic contract (analyst-approved, versioned).  The screen can
     mark a declared edge CONTRADICTED (conditional independence not rejected,
     stably) or INCONCLUSIVE.  It can never add an edge.  A business KPI panel
     of 200-odd autocorrelated daily observations does not support reliable
     structure discovery, and pretending otherwise is the error V4 made.

  2. Instability is a first-class verdict.  Each edge test is re-run on
     overlapping subsamples; if the verdict does not hold in at least
     `stability_threshold` of them, the result is INCONCLUSIVE and downstream
     structural support is set to UNKNOWN rather than to a number.

Method: partial correlation of X and Y given Z = parents(Y) \\ {X}, computed by
residualising both on Z (plus a linear trend, because both series are
non-stationary and a spurious correlation between two trending series is the
classic false positive).  Significance via the Fisher z transform.

MVP substitution, stated plainly: production RootSight runs FCI (causal-learn)
here and compares the resulting PAG with the declared graph.  The MVP runs this
partial-correlation screen instead.  Both emit the identical
`StructuralEvidence` contract, so nothing downstream changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy import stats

from .. import config
from .dag import CausalDAG


@dataclass
class EdgeVerdict:
    source: str
    target: str
    conditioned_on: list[str]
    partial_corr: float
    p_value: float
    n: int
    verdict: str                # SUPPORTED | CONTRADICTED | INCONCLUSIVE | NOT_TESTABLE
    stability: float
    reason: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["partial_corr"] = None if np.isnan(self.partial_corr) else round(self.partial_corr, 4)
        d["p_value"] = None if np.isnan(self.p_value) else round(self.p_value, 6)
        d["stability"] = round(self.stability, 3)
        return d


@dataclass
class StructuralEvidence:
    method: str
    is_discovery: bool
    edges: list[EdgeVerdict]
    stability_threshold: float
    n_subsamples: int
    notes: list[str] = field(default_factory=list)

    def verdict_for(self, source: str, target: str) -> EdgeVerdict | None:
        for e in self.edges:
            if e.source == source and e.target == target:
                return e
        return None

    def support_level(self, source: str, target: str) -> str:
        v = self.verdict_for(source, target)
        return v.verdict if v else "NOT_TESTABLE"

    def as_dict(self) -> dict:
        return {"method": self.method, "is_discovery": self.is_discovery,
                "stability_threshold": self.stability_threshold,
                "n_subsamples": self.n_subsamples,
                "edges": [e.as_dict() for e in self.edges],
                "notes": self.notes}


def _partial_corr(x: np.ndarray, y: np.ndarray,
                  Z: np.ndarray) -> tuple[float, float, int, tuple[float, float] | None]:
    """Partial correlation with a Fisher-z confidence interval.

    The interval is what lets the screen distinguish EVIDENCE OF NO EDGE from NO
    EVIDENCE OF AN EDGE.  Returning only a p-value forces the caller to treat a
    failure to reject as a refutation, which is exactly the inference error that
    makes automated structure screens dangerous on short business series.
    """
    ok = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    x, y, Z = x[ok], y[ok], Z[ok]
    n = len(x)
    if n < 20:
        return float("nan"), float("nan"), n, None
    bx = np.linalg.lstsq(Z, x, rcond=None)[0]
    by = np.linalg.lstsq(Z, y, rcond=None)[0]
    rx, ry = x - Z @ bx, y - Z @ by
    if rx.std(ddof=1) <= 0 or ry.std(ddof=1) <= 0:
        return float("nan"), float("nan"), n, None
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = n - Z.shape[1] - 2
    if dof <= 3 or abs(r) >= 0.999:
        return r, float("nan"), n, None
    zr = 0.5 * np.log((1 + r) / (1 - r))
    se = 1.0 / np.sqrt(dof - 1)
    p = float(2 * stats.norm.sf(abs(zr / se)))
    crit = 1.959964
    lo, hi = np.tanh(zr - crit * se), np.tanh(zr + crit * se)
    return r, p, n, (float(lo), float(hi))


class StructureScreen:
    def __init__(self, alpha: float = 0.05,
                 stability_threshold: float = config.GATES.structure_stability_threshold,
                 n_subsamples: int = 12, subsample_frac: float = 0.7,
                 negligible_partial_corr: float = 0.15):
        self.alpha = alpha
        self.negligible = negligible_partial_corr
        self.stability_threshold = stability_threshold
        self.n_subsamples = n_subsamples
        self.frac = subsample_frac

    def run(self, dag: CausalDAG, frame: pd.DataFrame,
            *, available: list[str]) -> StructuralEvidence:
        notes = [
            "This is a screen against the analyst-approved graph, not structure "
            "discovery. An edge can be contradicted or found inconclusive; no edge "
            "can be created here.",
            "Both series are residualised on a linear trend before testing, because "
            "two independent trending series are correlated by construction.",
        "The conditioning set excludes descendants of the source variable. "
            "Conditioning on a mediator would block the very path being screened.",
        "A failure to reject conditional independence is reported as INCONCLUSIVE "
            "unless the confidence interval also excludes any association above the "
            "negligible band. Absence of evidence is not evidence of absence, and an "
            "underpowered test must not be allowed to veto a declared edge.",
        ]
        verdicts: list[EdgeVerdict] = []
        n_rows = len(frame)
        for (a, b) in dag.edges:
            if a not in available or b not in available:
                verdicts.append(EdgeVerdict(
                    a, b, [], float("nan"), float("nan"), 0, "NOT_TESTABLE", 0.0,
                    reason=("one or both variables are not observed in this scope "
                            f"(a_observed={a in available}, b_observed={b in available})")))
                continue
            # Condition on the other parents of the target, but NEVER on a
            # descendant of the source.  A mediator is part of the mechanism being
            # screened: conditioning on it blocks the indirect path and makes a
            # real total effect look absent.  This is the most common way a
            # conditional-independence screen produces a false negative.
            cond = sorted((dag.parents(b) - {a} - dag.descendants(a)) & set(available))
            v_full = self._test_edge(frame, a, b, cond)
            # stability over contiguous subsamples
            agree = 0
            tested = 0
            span = max(30, int(n_rows * self.frac))
            starts = np.linspace(0, max(0, n_rows - span), self.n_subsamples).astype(int)
            for s0 in starts:
                sub = frame.iloc[s0:s0 + span]
                vs = self._test_edge(sub, a, b, cond)
                if vs[3] == "NOT_TESTABLE":
                    continue
                tested += 1
                if vs[3] == v_full[3]:
                    agree += 1
            stability = agree / tested if tested else 0.0
            verdict = v_full[3]
            reason = v_full[4]
            if verdict != "NOT_TESTABLE" and stability < self.stability_threshold:
                reason = (f"verdict {verdict} held in only {stability:.0%} of "
                          f"{tested} subsamples (threshold {self.stability_threshold:.0%}); "
                          "structural evidence is treated as inconclusive")
                verdict = "INCONCLUSIVE"
            verdicts.append(EdgeVerdict(a, b, cond, v_full[0], v_full[1], v_full[2],
                                        verdict, stability, reason))
        return StructuralEvidence(
            method="partial-correlation conditional-independence screen with "
                   "subsample stability (MVP substitute for FCI; identical output "
                   "contract)",
            is_discovery=False, edges=verdicts,
            stability_threshold=self.stability_threshold,
            n_subsamples=self.n_subsamples, notes=notes)

    def _test_edge(self, frame: pd.DataFrame, a: str, b: str,
                   cond: list[str]) -> tuple[float, float, int, str, str]:
        if a not in frame.columns or b not in frame.columns:
            return float("nan"), float("nan"), 0, "NOT_TESTABLE", "column absent"
        x = frame[a].to_numpy(dtype=float)
        y = frame[b].to_numpy(dtype=float)
        t = np.arange(len(frame), dtype=float)
        cols = [np.ones(len(frame)), t]
        for c in cond:
            if c in frame.columns:
                cols.append(frame[c].to_numpy(dtype=float))
        Z = np.column_stack(cols)
        r, p, n, ci = _partial_corr(x, y, Z)
        if not np.isfinite(r) or not np.isfinite(p):
            return r, p, n, "NOT_TESTABLE", f"insufficient usable rows (n={n})"
        if p < self.alpha:
            return (r, p, n, "SUPPORTED",
                    f"conditional independence rejected (partial r={r:+.3f}, p={p:.4g}, "
                    f"n={n}) given {cond or 'trend only'}: consistent with the declared "
                    f"edge")
        # Failure to reject is not refutation.  Only if the interval also EXCLUDES
        # any association large enough to matter can the edge be called
        # contradicted; otherwise the test was simply underpowered.
        if ci is not None and abs(ci[0]) < self.negligible and abs(ci[1]) < self.negligible:
            return (r, p, n, "CONTRADICTED",
                    f"conditional independence not rejected (partial r={r:+.3f}, "
                    f"p={p:.4g}, n={n}) AND the 95% interval [{ci[0]:+.3f}, {ci[1]:+.3f}] "
                    f"excludes any association above the negligible band of "
                    f"{self.negligible:.2f}: the data actively contradict this edge")
        band = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "unavailable"
        return (r, p, n, "INCONCLUSIVE",
                f"conditional independence not rejected (partial r={r:+.3f}, p={p:.4g}, "
                f"n={n}) but the 95% interval {band} still admits a material "
                f"association, so this is absence of evidence, not evidence of absence")
