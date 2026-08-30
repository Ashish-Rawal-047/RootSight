"""KPI movement detection.

Three stages, deliberately separated:

  1. CALENDAR / SEASONAL ADJUSTMENT
     Day-of-week, holiday and promotional-window effects are removed by OLS on a
     training span.  Promo-blackout dates (declared in the business calendar) are
     excluded from baseline windows entirely rather than "adjusted".

  2. CHANGEPOINT LOCATION
     Binary segmentation on the adjusted series using a mean-shift statistic.

  3. SIGNIFICANCE
     A *block* permutation test (block length = 7 days) so that autocorrelation
     is preserved under the null.  An iid permutation test would report absurdly
     small p-values on autocorrelated business series; this is the single most
     common false-alarm source in KPI monitoring.

Detection produces a MOVEMENT, not an explanation.  Nothing here is causal.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, timedelta

import numpy as np

from .. import config
from ..bizcalendar import attributes
from ..contracts.kpi_contract import KpiDefinition
from ..kpi.compute import KpiSeries


# ------------------------------------------------------------------ regression
def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def calendar_design(dates: list[date], with_trend: bool = True) -> tuple[np.ndarray, list[str]]:
    rows, names = [], []
    for i, d in enumerate(dates):
        a = attributes(d)
        row = [1.0]
        row += [1.0 if a["dow"] == k else 0.0 for k in range(1, 7)]   # Monday is base
        row += [1.0 if a["is_holiday"] else 0.0]
        row += [1.0 if a["promo_window"] else 0.0]
        if with_trend:
            row += [float(i)]
        rows.append(row)
    names = (["const"] + [f"dow_{k}" for k in range(1, 7)]
             + ["holiday", "promo"] + (["trend"] if with_trend else []))
    return np.asarray(rows, dtype=float), names


@dataclass
class Movement:
    kpi_id: str
    detected: bool
    changepoint_date: date | None
    focus_window: tuple[date, date]
    baseline_window: tuple[date, date]
    focus_value: float
    baseline_value: float
    abs_change: float
    pct_change: float
    direction: str
    method: str
    shift_statistic: float
    p_value: float
    threshold_pct: float
    threshold_breached: bool
    min_periods_required: int
    periods_available: int
    seasonal_adjustment: dict
    calendar_exclusions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["changepoint_date"] = self.changepoint_date.isoformat() if self.changepoint_date else None
        d["focus_window"] = [x.isoformat() for x in self.focus_window]
        d["baseline_window"] = [x.isoformat() for x in self.baseline_window]
        for k in ("focus_value", "baseline_value", "abs_change", "pct_change",
                  "shift_statistic", "p_value"):
            d[k] = None if d[k] is None or np.isnan(d[k]) else round(float(d[k]), 6)
        return d


class MovementDetector:
    def __init__(self, rng_seed: int = 7, n_permutations: int = 800, block: int = 7):
        self.rng = np.random.default_rng(rng_seed)
        self.n_perm = n_permutations
        self.block = block

    # --------------------------------------------------------------- internals
    def _adjust(self, y: np.ndarray, dates: list[date]) -> tuple[np.ndarray, dict]:
        """Remove day-of-week / holiday / promo structure, retain level and trend."""
        X, names = calendar_design(dates, with_trend=False)
        beta = _ols(X, y)
        # keep the intercept, subtract only the calendar terms
        cal = X[:, 1:] @ beta[1:]
        adj = y - cal
        fitted = X @ beta
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum((y - fitted) ** 2)) / ss_tot if ss_tot > 0 else 0.0
        return adj, {"method": "OLS day-of-week + holiday + promo dummies",
                     "terms": names, "r2_of_calendar_model": round(r2, 4),
                     "coefficients": {n: round(float(b), 4) for n, b in zip(names, beta)}}

    @staticmethod
    def _binseg(y: np.ndarray, min_seg: int = 10) -> tuple[int, float]:
        """Single best mean-shift location and its normalised (Welch t) statistic.

        Vectorised over candidate split points via cumulative sums: the
        permutation test calls this thousands of times, so an O(n) sweep instead
        of an O(n^2) loop is what keeps end-to-end latency inside the budget.
        """
        n = len(y)
        if n < 2 * min_seg + 1:
            return -1, 0.0
        c1 = np.concatenate(([0.0], np.cumsum(y)))
        c2 = np.concatenate(([0.0], np.cumsum(y * y)))
        i = np.arange(min_seg, n - min_seg)
        na, nb = i.astype(float), float(n) - i
        sa, sb = c1[i], c1[n] - c1[i]
        qa, qb = c2[i], c2[n] - c2[i]
        ma, mb = sa / na, sb / nb
        va = np.maximum(qa - na * ma * ma, 0.0) / np.maximum(na - 1.0, 1.0)
        vb = np.maximum(qb - nb * mb * mb, 0.0) / np.maximum(nb - 1.0, 1.0)
        sd = np.sqrt(va / na + vb / nb)
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = np.where(sd > 0, np.abs(mb - ma) / sd, 0.0)
        j = int(np.nanargmax(stat))
        return int(i[j]), float(stat[j])

    def _block_permutation_p(self, y: np.ndarray, stat: float) -> float:
        n = len(y)
        nb = max(2, n // self.block)
        blocks = np.array_split(y, nb)
        count = 0
        for _ in range(self.n_perm):
            order = self.rng.permutation(len(blocks))
            perm = np.concatenate([blocks[j] for j in order])
            _, s = self._binseg(perm)
            if s >= stat:
                count += 1
        return (count + 1) / (self.n_perm + 1)

    # ------------------------------------------------------------------ public
    def detect(self, series: KpiSeries, *,
               focus: tuple[date, date] = config.FOCUS_WINDOW,
               baseline: tuple[date, date] = config.BASELINE_WINDOW) -> Movement:
        k: KpiDefinition = series.definition
        full = series.total_by_date().sort_index().dropna()
        dates = list(full.index)
        y = full.to_numpy(dtype=float)
        notes: list[str] = []

        # calendar exclusions: promo-blackout days never enter a baseline
        excl = [d.isoformat() for d in dates
                if baseline[0] <= d <= baseline[1] and not attributes(d)["baseline_eligible"]]
        focus_excl = [d.isoformat() for d in dates
                      if focus[0] <= d <= focus[1] and not attributes(d)["baseline_eligible"]]
        if focus_excl:
            notes.append(
                f"{len(focus_excl)} promotional day(s) fall inside the focus window; "
                "they are retained in the focus value (they are part of what happened) "
                "but excluded from the baseline, and the promo dummy absorbs the "
                "systematic part of their effect")

        periods = int(((full.index >= baseline[0]) & (full.index <= focus[1])).sum())
        if len(y) < 20:
            return Movement(
                kpi_id=k.kpi_id, detected=False, changepoint_date=None,
                focus_window=focus, baseline_window=baseline,
                focus_value=float("nan"), baseline_value=float("nan"),
                abs_change=float("nan"), pct_change=float("nan"), direction="none",
                method="none (insufficient observations)", shift_statistic=float("nan"),
                p_value=float("nan"), threshold_pct=float(k.thresholds["alert_pct_change"]),
                threshold_breached=False,
                min_periods_required=int(k.thresholds["min_periods_for_alert"]),
                periods_available=len(y), seasonal_adjustment={},
                notes=[f"only {len(y)} daily observations exist for {k.kpi_id}; "
                       "calendar adjustment and changepoint testing are not run"])

        adj, adj_meta = self._adjust(y, dates)
        i, stat = self._binseg(adj)
        cp = dates[i] if i > 0 else None
        p = self._block_permutation_p(adj, stat) if i > 0 else float("nan")

        # window values honour the contract's time_aggregation
        base_dates = [d for d in dates
                      if baseline[0] <= d <= baseline[1] and attributes(d)["baseline_eligible"]]
        if len(base_dates) < 10:
            return Movement(
                kpi_id=k.kpi_id, detected=False, changepoint_date=None,
                focus_window=focus, baseline_window=baseline,
                focus_value=series.window_value(*focus), baseline_value=float("nan"),
                abs_change=float("nan"), pct_change=float("nan"), direction="none",
                method="none (no comparable baseline window)",
                shift_statistic=stat, p_value=p,
                threshold_pct=float(k.thresholds["alert_pct_change"]),
                threshold_breached=False,
                min_periods_required=int(k.thresholds["min_periods_for_alert"]),
                periods_available=len(y), seasonal_adjustment=adj_meta,
                calendar_exclusions=excl,
                notes=notes + [
                    f"only {len(base_dates)} baseline-eligible day(s) exist for "
                    f"{k.kpi_id} in {baseline[0]}..{baseline[1]}; the KPI has "
                    f"{len(y)} days of history in total, so no percentage change "
                    "against a comparable prior period can be computed"])
        if k.time_aggregation == "FLOW":
            fv = float(full[(full.index >= focus[0]) & (full.index <= focus[1])].mean())
            bv = float(full[full.index.isin(base_dates)].mean())
            unit_note = "compared as a daily mean because the windows differ in length"
        else:
            fv = series.window_value(*focus)
            bv = float(full[full.index.isin(base_dates)].mean())
            unit_note = "level comparison"
        notes.append(f"FLOW/RATIO handling: {unit_note}")

        pct = 100.0 * (fv / bv - 1.0) if bv else float("nan")
        return Movement(
            kpi_id=k.kpi_id, detected=bool(i > 0 and p < 0.05),
            changepoint_date=cp, focus_window=focus, baseline_window=baseline,
            focus_value=fv, baseline_value=bv, abs_change=fv - bv, pct_change=pct,
            direction="down" if pct < 0 else "up",
            method="calendar-adjusted binary segmentation + block permutation test (block=7d)",
            shift_statistic=stat, p_value=p,
            threshold_pct=float(k.thresholds["alert_pct_change"]),
            threshold_breached=bool(k.threshold_breached(pct)),
            min_periods_required=int(k.thresholds["min_periods_for_alert"]),
            periods_available=periods, seasonal_adjustment=adj_meta,
            calendar_exclusions=excl, notes=notes)
