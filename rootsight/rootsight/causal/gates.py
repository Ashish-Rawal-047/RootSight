"""Gates: data sufficiency and TEMPORAL COMPATIBILITY.

V4 called the second gate "temporal ordering" and treated passing it as a step
towards causation.  V5 renames it TEMPORAL COMPATIBILITY and states its logic
explicitly:

    Passing this gate establishes only that the observed timing is COMPATIBLE
    with the hypothesised direction, at the resolution the data actually has.
    Failing it rules a hypothesis out.  Passing it rules nothing in.

Three failure modes are distinguished, because they call for different
responses:

  GRAIN_TOO_COARSE    the driver's true temporal resolution is coarser than the
                      lag the hypothesis requires.  Marketing arrives weekly;
                      spreading it over 7 days does not make a 3-day lag
                      measurable.  No amount of extra history fixes this - the
                      instrumentation has to change.
  NO_LEAD             the maximum cross-correlation occurs at a non-positive
                      lag, i.e. the "driver" moves with or after the KPI.
  UNSTABLE_LAG        the bootstrap interval for the best lag is so wide that no
                      particular lag is supported.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from .. import config
from ..contracts.kpi_contract import Driver


@dataclass
class GateResult:
    gate: str
    passed: bool
    checks: list[dict]
    failure_codes: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SufficiencyResult(GateResult):
    power_context: dict = field(default_factory=dict)


def data_sufficiency(y: np.ndarray, x: np.ndarray, *, pre_n: int, post_n: int,
                     missing_frac: float, gates=config.GATES) -> SufficiencyResult:
    outcome_missing = float(np.mean(~np.isfinite(y)))
    driver_observed = 1.0 - missing_frac
    checks = [
        {"check": "usable_paired_pre_observations", "value": pre_n,
         "minimum": gates.min_pre_obs, "passed": pre_n >= gates.min_pre_obs,
         "note": ("counted on rows where BOTH driver and outcome are observed. "
                  "Engineering minimum for any windowed comparison, NOT a power "
                  "guarantee")},
        {"check": "usable_paired_post_observations", "value": post_n,
         "minimum": gates.min_post_obs, "passed": post_n >= gates.min_post_obs},
        {"check": "outcome_missing_fraction", "value": round(outcome_missing, 4),
         "maximum": gates.max_missing_frac,
         "passed": outcome_missing < gates.max_missing_frac,
         "note": "the KPI itself must be measured; a gap here blocks the analysis"},
        {"check": "driver_observed_fraction", "value": round(driver_observed, 4),
         "minimum": 0.50, "passed": driver_observed >= 0.50,
         "note": ("a partially observed driver can still support an association if "
                  "enough paired days exist, but the coverage is carried forward into "
                  "assumption A8 and into the data-completeness dimension of the "
                  "ranking rather than being silently ignored")},
        {"check": "driver_variance_positive",
         "value": round(float(np.nanvar(x)), 8), "passed": float(np.nanvar(x)) > 0},
        {"check": "outcome_variance_positive",
         "value": round(float(np.nanvar(y)), 8), "passed": float(np.nanvar(y)) > 0},
    ]
    codes = [c["check"].upper() for c in checks if not c["passed"]]

    # ---- power context, reported separately and never used as a gate --------
    ok = np.isfinite(y)
    ys = y[ok]
    power: dict = {}
    if len(ys) > 20:
        d = np.diff(ys)
        rho1 = (float(np.corrcoef(ys[:-1], ys[1:])[0, 1]) if len(ys) > 3 else 0.0)
        sd = float(np.nanstd(ys, ddof=1))
        n_eff = len(ys) * (1 - rho1) / (1 + rho1) if rho1 > -0.99 else len(ys)
        mde = 2.8 * sd / np.sqrt(max(n_eff, 1.0))            # ~80% power, alpha=0.05
        power = {
            "outcome_sd": round(sd, 4),
            "lag1_autocorrelation": round(rho1, 4),
            "effective_n_after_autocorrelation": round(float(n_eff), 1),
            "minimum_detectable_effect_abs": round(float(mde), 4),
            "minimum_detectable_effect_pct_of_mean": round(
                float(100 * mde / abs(np.nanmean(ys))), 3) if np.nanmean(ys) else None,
            "interpretation": ("An effect smaller than the minimum detectable effect "
                               "may be real yet invisible at this sample size. This is "
                               "reported alongside the gate, never folded into it."),
        }
    return SufficiencyResult(gate="GATE_1_DATA_SUFFICIENCY", passed=not codes,
                             checks=checks, failure_codes=codes, power_context=power)


@dataclass
class TemporalCompatibility(GateResult):
    best_lag: int | None = None
    best_corr: float = float("nan")
    lag_ci: tuple[int, int] | None = None
    driver_resolution_days: float = 1.0
    required_lag_range: tuple[int, int] | None = None
    lag_profile: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["best_corr"] = None if np.isnan(self.best_corr) else round(self.best_corr, 4)
        d["lag_ci"] = list(self.lag_ci) if self.lag_ci else None
        d["required_lag_range"] = (list(self.required_lag_range)
                                   if self.required_lag_range else None)
        return d


def _prewhiten(v: np.ndarray, dates=None) -> np.ndarray:
    """Remove linear trend and, when dates are supplied, calendar structure.

    Two business series that share a weekly rhythm are correlated whatever their
    causal relationship, and that shared rhythm swamps the lag signal.  Removing
    day-of-week, holiday and promotional structure from BOTH series before
    cross-correlating is what makes the remaining association informative about
    timing.  Skipping this step is the single most common way a KPI monitor
    invents a driver.
    """
    n = len(v)
    t = np.arange(n, dtype=float)
    cols = [np.ones(n), t]
    if dates is not None and len(dates) == n:
        from ..bizcalendar import attributes as _attrs
        a = [_attrs(d) for d in dates]
        for k in range(1, 7):
            cols.append(np.array([1.0 if x["dow"] == k else 0.0 for x in a]))
        cols.append(np.array([1.0 if x["is_holiday"] else 0.0 for x in a]))
        cols.append(np.array([1.0 if x["promo_window"] else 0.0 for x in a]))
    X = np.column_stack(cols)
    ok = np.isfinite(v)
    if ok.sum() < max(15, X.shape[1] + 2):
        return v - np.nanmean(v)
    beta = np.linalg.lstsq(X[ok], v[ok], rcond=None)[0]
    return v - X @ beta


def temporal_compatibility(driver: np.ndarray, outcome: np.ndarray, *,
                           driver_meta: Driver | None,
                           resolution_days: float,
                           dates=None,
                           max_lag: int = 14,
                           n_bootstrap: int = config.GATES.bootstrap_n,
                           gates=config.GATES,
                           seed: int = 11) -> TemporalCompatibility:
    """Cross-correlation of calendar-adjusted series across lags.

    Stability is assessed by CONTIGUOUS SUBSAMPLING rather than block
    resampling.  Reshuffling blocks would destroy the very time alignment the
    lag is measured on, so the argmax lag is instead recomputed on overlapping
    sub-windows and its spread reported.
    """
    x = _prewhiten(np.asarray(driver, dtype=float), dates)
    y = _prewhiten(np.asarray(outcome, dtype=float), dates)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    required = tuple(driver_meta.prior_lag_days) if (driver_meta and driver_meta.prior_lag_days) else None

    def corr_at(lag: int, xs: np.ndarray, ys: np.ndarray) -> float:
        if lag >= 0:
            a, b = xs[:len(xs) - lag] if lag else xs, ys[lag:]
        else:
            a, b = xs[-lag:], ys[:len(ys) + lag]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 15 or np.nanstd(a[m]) == 0 or np.nanstd(b[m]) == 0:
            return float("nan")
        return float(np.corrcoef(a[m], b[m])[0, 1])

    lags = list(range(-max_lag, max_lag + 1))
    profile = [{"lag_days": L, "corr": (None if np.isnan(corr_at(L, x, y))
                                        else round(corr_at(L, x, y), 4))}
               for L in lags]
    vals = np.array([corr_at(L, x, y) for L in lags])
    if np.all(np.isnan(vals)):
        return TemporalCompatibility(
            gate="GATE_2_TEMPORAL_COMPATIBILITY", passed=False,
            checks=[{"check": "cross_correlation_computable", "passed": False}],
            failure_codes=["NOT_COMPUTABLE"], best_lag=None,
            driver_resolution_days=resolution_days, required_lag_range=required,
            lag_profile=profile,
            detail={"reason": "cross-correlation could not be computed at any lag"})

    best_i = int(np.nanargmax(np.abs(vals)))
    best_lag, best_corr = lags[best_i], float(vals[best_i])

    # ---- stability of the argmax lag under contiguous subsampling -----------
    span = max(40, int(n * 0.7))
    starts = np.unique(np.linspace(0, max(0, n - span), 24).astype(int))
    boot: list[int] = []
    for s0 in starts:
        xb, yb = x[s0:s0 + span], y[s0:s0 + span]
        vb = np.array([corr_at(L, xb, yb) for L in lags])
        if np.all(np.isnan(vb)):
            continue
        boot.append(lags[int(np.nanargmax(np.abs(vb)))])
    lag_iqr = ((int(np.percentile(boot, 25)), int(np.percentile(boot, 75)))
               if len(boot) >= 5 else None)
    lag_ci = ((int(np.percentile(boot, 5)), int(np.percentile(boot, 95)))
              if len(boot) >= 5 else None)

    checks = [
        {"check": "association_strength", "value": round(abs(best_corr), 4),
         "minimum": gates.min_abs_corr, "passed": abs(best_corr) >= gates.min_abs_corr,
         "note": "an association too weak to measure cannot support a timing claim"},
        {"check": "driver_leads_outcome", "value": best_lag,
         "passed": best_lag > 0,
         "note": "lag > 0 means the driver moves first; equal or later timing is "
                 "incompatible with the hypothesised direction"},
        {"check": "lag_resolvable_at_data_resolution",
         "value": {"best_lag_days": best_lag,
                   "driver_resolution_days": resolution_days,
                   "applicable": best_lag > 0},
         "passed": bool(best_lag <= 0 or best_lag >= resolution_days),
         "note": ("A lag finer than the driver's true temporal resolution is not "
                  "measurable, however the values were allocated across days. Not "
                  "applicable when there is no positive lead: the no-lead failure "
                  "already covers that case.")},
        {"check": "lag_stability",
         "value": {"interquartile_lag_range": list(lag_iqr) if lag_iqr else None,
                   "full_subsample_range": list(lag_ci) if lag_ci else None,
                   "n_subsamples": len(boot)},
         "passed": bool(lag_iqr and (lag_iqr[1] - lag_iqr[0]) <= 5 and lag_iqr[0] >= 1),
         "note": ("gated on the INTERQUARTILE range of the argmax lag across "
                  "contiguous sub-windows. The full range is reported too and is "
                  "expected to be wide, because sub-windows that predate the movement "
                  "contain no lag to find; requiring the full range to be tight would "
                  "reject every real event.")},
    ]
    prior_ok = True
    if required:
        prior_ok = bool(required[0] - 2 <= best_lag <= required[1] + 2)
        checks.append({
            "check": "lag_consistent_with_declared_prior",
            "value": {"best_lag": best_lag, "contract_prior_range": list(required)},
            "passed": prior_ok, "blocking": False,
            "note": ("The contract's expected lag range is expert opinion, not data. A "
                     "disagreement is raised for human review and carried into the "
                     "assumption table; it does not by itself downgrade the analysis, "
                     "because the prior is as likely to be wrong as the estimate.")})

    codes = []
    if not checks[0]["passed"]:
        codes.append("ASSOCIATION_TOO_WEAK")
    if not checks[1]["passed"]:
        codes.append("NO_LEAD")
    if not checks[2]["passed"]:
        codes.append("GRAIN_TOO_COARSE")
    if not checks[3]["passed"]:
        codes.append("UNSTABLE_LAG")
    review_flags = []
    if required and not prior_ok:
        review_flags.append("LAG_INCONSISTENT_WITH_CONTRACT_PRIOR")

    return TemporalCompatibility(
        gate="GATE_2_TEMPORAL_COMPATIBILITY", passed=not codes, checks=checks,
        failure_codes=codes, best_lag=best_lag, best_corr=best_corr, lag_ci=lag_ci,
        driver_resolution_days=resolution_days, required_lag_range=required,
        lag_profile=profile,
        detail={"review_flags": review_flags,
                "interpretation": (
            "Temporal compatibility is necessary, never sufficient. Passing means the "
            "observed timing does not rule the hypothesis out at this resolution.")})


def benjamini_hochberg(pvals: dict[str, float], q: float = config.GATES.bh_q) -> dict:
    """FDR control across the candidate drivers tested for one KPI movement."""
    items = [(k, v) for k, v in pvals.items() if v is not None and np.isfinite(v)]
    if not items:
        return {"q": q, "tested": 0, "rejected": [], "adjusted": {}, "threshold": None}
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 1.0
    for i in range(m - 1, -1, -1):
        k, p = items[i]
        running = min(running, p * m / (i + 1))
        adjusted[k] = round(float(running), 6)
    thresh = None
    rejected = []
    for i, (k, p) in enumerate(items, start=1):
        if p <= q * i / m:
            thresh = p
    for k, p in items:
        if thresh is not None and p <= thresh:
            rejected.append(k)
    return {"q": q, "tested": m, "rejected": sorted(rejected), "adjusted": adjusted,
            "threshold": thresh,
            "note": ("Benjamini-Hochberg FDR control across all candidate drivers "
                     "evaluated for this movement. A driver that does not survive "
                     "correction is reported as not distinguishable from noise once "
                     "multiplicity is accounted for.")}
