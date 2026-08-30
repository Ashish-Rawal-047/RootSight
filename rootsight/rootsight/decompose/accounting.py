"""Accounting decomposition of a revenue movement.

THIS MODULE PRODUCES NO CAUSAL CLAIMS.  It answers a purely arithmetic question:
given that net revenue is identically sum(units_i * asp_i) over cells, how much
of the observed change is attributable to the *arithmetic* of volume, of mix, of
price, and of cells entering or leaving the portfolio?

The identity implemented is exact and closes to zero residual:

    dR = volume + mix + price + entry - exit

  volume = (Q1 - Q0) * pbar0                     Q = total units, pbar0 = base avg price
  mix    = Q1 * (sum_i s1_i * p0_i - pbar0)      s1 = new share, valued at BASE prices
  price  = sum_i q1_i * (p1_i - p0_i)            price change valued at NEW quantities
  entry  = sum_{new cells}   q1_i * p1_i
  exit   = sum_{gone cells}  q0_i * p0_i

Entry and exit exist because a newly launched product line has no base price;
folding it into "mix" would silently invent a counterfactual price for a product
that did not exist.  The identity is verified numerically on every call and a
ContractViolation is raised if it does not close.

Language rule enforced downstream: a decomposition term is a CONTRIBUTION, never
an EFFECT.  "Volume contributed -4.3pp" is a statement about arithmetic.
"Volume caused -4.3pp" is a causal claim this module cannot support.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date

import numpy as np
import pandas as pd

from ..contracts.kpi_contract import ContractViolation, registry
from ..kpi.compute import KpiSeries


@dataclass
class Term:
    term_id: str
    label: str
    kind: str                 # ARITHMETIC_CONTRIBUTION
    value_abs: float          # INR per day
    value_pp: float           # percentage points of the base
    share_of_change_pct: float
    interpretation: str
    detail: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("value_abs", "value_pp", "share_of_change_pct"):
            d[k] = round(float(d[k]), 4)
        return d


@dataclass
class Decomposition:
    kpi_id: str
    method: str
    is_causal: bool
    focus_window: tuple[date, date]
    baseline_window: tuple[date, date]
    base_value: float
    focus_value: float
    total_change_abs: float
    total_change_pp: float
    terms: list[Term]
    residual_abs: float
    identity_closes: bool
    cells_common: int
    cells_entered: list[str]
    cells_exited: list[str]
    caveats: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "kpi_id": self.kpi_id, "method": self.method, "is_causal": self.is_causal,
            "focus_window": [d.isoformat() for d in self.focus_window],
            "baseline_window": [d.isoformat() for d in self.baseline_window],
            "base_value_per_day": round(self.base_value, 2),
            "focus_value_per_day": round(self.focus_value, 2),
            "total_change_abs_per_day": round(self.total_change_abs, 2),
            "total_change_pp": round(self.total_change_pp, 4),
            "terms": [t.as_dict() for t in self.terms],
            "residual_abs": round(self.residual_abs, 8),
            "identity_closes": self.identity_closes,
            "cells_common": self.cells_common,
            "cells_entered": self.cells_entered,
            "cells_exited": self.cells_exited,
            "caveats": self.caveats,
        }


class PriceVolumeMixDecomposer:
    """Exact price / volume / mix / entry / exit decomposition."""

    TOL = 1e-6

    def __init__(self, cell_dims: tuple[str, ...] = ("region", "product_line")):
        self.cell_dims = cell_dims
        self.reg = registry()

    def run(self, revenue: KpiSeries, units: KpiSeries, *,
            focus: tuple[date, date], baseline: tuple[date, date]) -> Decomposition:
        # R2: the two series must be contract-compatible
        self.reg.assert_comparable_ok = True
        self.reg.assert_series_consistent(revenue.kpi_id, [units.kpi_id],
                                          context="price/volume/mix decomposition")
        k = revenue.definition
        if k.time_aggregation != "FLOW":
            raise ContractViolation(
                f"{k.kpi_id} is {k.time_aggregation}; the price/volume/mix identity is "
                "defined for FLOW measures only")

        dims = [d for d in self.cell_dims if d in revenue.frame.columns
                and d in units.frame.columns]
        if not dims:
            raise ContractViolation(
                "price/volume/mix decomposition needs at least one cell dimension; "
                f"the caller's grain exposes none of {list(self.cell_dims)}. "
                "This usually means the requesting role is not entitled to the "
                "product-line grain.")

        r = self._cells(revenue, dims, "net_revenue", focus, baseline)
        u = self._cells(units, dims, "units_sold", focus, baseline)
        cells = r.join(u, how="outer", lsuffix="_rev", rsuffix="_units").fillna(0.0)

        q0, q1 = cells["base_units_sold"], cells["focus_units_sold"]
        v0, v1 = cells["base_net_revenue"], cells["focus_net_revenue"]
        common = (q0 > 0) & (q1 > 0)
        entered = (q0 <= 0) & (q1 > 0)
        exited = (q0 > 0) & (q1 <= 0)

        p0 = np.where(q0 > 0, v0 / np.where(q0 > 0, q0, 1.0), 0.0)
        p1 = np.where(q1 > 0, v1 / np.where(q1 > 0, q1, 1.0), 0.0)

        Q0 = float(q0[common].sum())
        Q1 = float(q1[common].sum())
        R0c = float(v0[common].sum())
        R1c = float(v1[common].sum())
        pbar0 = R0c / Q0 if Q0 else 0.0

        volume = (Q1 - Q0) * pbar0
        mix = float(np.sum(q1[common] * p0[common.to_numpy()])) - Q1 * pbar0
        price = float(np.sum(q1[common] * (p1[common.to_numpy()] - p0[common.to_numpy()])))
        entry = float(v1[entered].sum())
        exit_ = float(v0[exited].sum())

        base_total = float(v0.sum())
        focus_total = float(v1.sum())
        total_change = focus_total - base_total
        residual = total_change - (volume + mix + price + entry - exit_)
        closes = abs(residual) <= max(self.TOL, abs(total_change) * 1e-9)
        if not closes:
            raise ContractViolation(
                f"decomposition identity failed to close: residual={residual:.6f} on a "
                f"change of {total_change:.2f}. Refusing to publish an inexact "
                "decomposition.")

        def pp(x: float) -> float:
            return 100.0 * x / base_total if base_total else float("nan")

        def share(x: float) -> float:
            return 100.0 * x / total_change if total_change else float("nan")

        line_detail = self._per_cell_detail(cells, dims, p0, p1, common, pbar0, Q1)

        terms = [
            Term("D_VOLUME", "Volume", "ARITHMETIC_CONTRIBUTION", volume, pp(volume),
                 share(volume),
                 "Units sold changed, holding the base average price and base mix fixed. "
                 "This is arithmetic, not a cause: something made volume move.",
                 detail=line_detail["volume"]),
            Term("D_MIX", "Product mix", "ARITHMETIC_CONTRIBUTION", mix, pp(mix), share(mix),
                 "The composition of units shifted between cells with different base "
                 "prices, valued at BASE prices so no price change leaks into this term.",
                 detail=line_detail["mix"]),
            Term("D_PRICE", "Realised price", "ARITHMETIC_CONTRIBUTION", price, pp(price),
                 share(price),
                 "Realised price per unit changed within cells (list price and discount "
                 "depth combined), valued at FOCUS quantities.",
                 detail=line_detail["price"]),
        ]
        if entry:
            terms.append(Term(
                "D_ENTRY", "New cells entering", "ARITHMETIC_CONTRIBUTION", entry,
                pp(entry), share(entry),
                "Cells with no baseline sales. They have no base price, so they cannot "
                "be represented as volume, mix or price; they are reported separately "
                "rather than given an invented counterfactual price.",
                detail=[{"cell": c} for c in cells.index[entered].tolist()]))
        if exit_:
            terms.append(Term(
                "D_EXIT", "Cells exiting", "ARITHMETIC_CONTRIBUTION", -exit_, pp(-exit_),
                share(-exit_), "Cells with baseline sales and no focus-window sales.",
                detail=[{"cell": c} for c in cells.index[exited].tolist()]))

        caveats = [
            "Every term above is an ACCOUNTING CONTRIBUTION, not a causal effect. The "
            "terms are exhaustive by construction, so they cannot tell you why any of "
            "them moved.",
            "Mix and price are not separately identified without a cell dimension; they "
            "are computed at the finest grain the requesting role may see, which is "
            f"{' x '.join(dims)}.",
        ]
        if entry:
            caveats.append(
                "A newly launched cell is present. Its contribution is real revenue but "
                "it is not a like-for-like change, and it is excluded from mix and price.")

        return Decomposition(
            kpi_id=revenue.kpi_id,
            method="exact price/volume/mix identity with entry and exit terms "
                   "(base-priced mix, focus-quantity price)",
            is_causal=False, focus_window=focus, baseline_window=baseline,
            base_value=base_total, focus_value=focus_total,
            total_change_abs=total_change, total_change_pp=pp(total_change),
            terms=terms, residual_abs=residual, identity_closes=closes,
            cells_common=int(common.sum()),
            cells_entered=cells.index[entered].tolist(),
            cells_exited=cells.index[exited].tolist(), caveats=caveats)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _cells(series: KpiSeries, dims: list[str], value_col: str,
               focus: tuple[date, date], baseline: tuple[date, date]) -> pd.DataFrame:
        f = series.frame.copy()
        f["_cell"] = f[dims].astype(str).agg(" | ".join, axis=1)
        out = {}
        for tag, (s, e) in (("base", baseline), ("focus", focus)):
            w = f[(f["date"] >= s) & (f["date"] <= e)]
            n_days = max(1, w["date"].nunique())
            out[f"{tag}_{value_col}"] = w.groupby("_cell")["value"].sum() / n_days
        return pd.DataFrame(out)

    @staticmethod
    def _per_cell_detail(cells: pd.DataFrame, dims: list[str], p0, p1,
                         common, pbar0: float, Q1: float) -> dict:
        idx = cells.index.tolist()
        cm = common.to_numpy()
        q0 = cells["base_units_sold"].to_numpy()
        q1 = cells["focus_units_sold"].to_numpy()
        vol, mixd, prc = [], [], []
        for i, c in enumerate(idx):
            if not cm[i]:
                continue
            vol.append({"cell": c, "units_base_per_day": round(float(q0[i]), 1),
                        "units_focus_per_day": round(float(q1[i]), 1),
                        "units_change_pct": round(100 * (q1[i] / q0[i] - 1), 2)})
            s0 = q0[i] / q0[cm].sum()
            s1 = q1[i] / q1[cm].sum()
            # share CHANGE valued at base price: sums exactly to the mix term
            mixd.append({"cell": c, "base_price": round(float(p0[i]), 2),
                         "share_base_pct": round(100 * s0, 2),
                         "share_focus_pct": round(100 * s1, 2),
                         "contribution_abs": round(float(Q1 * p0[i] * (s1 - s0)), 2)})
            prc.append({"cell": c, "price_base": round(float(p0[i]), 2),
                        "price_focus": round(float(p1[i]), 2),
                        "price_change_pct": round(100 * (p1[i] / p0[i] - 1), 2),
                        "contribution_abs": round(float(q1[i] * (p1[i] - p0[i])), 2)})
        key = lambda r: -abs(r["contribution_abs"])       # noqa: E731
        return {"volume": sorted(vol, key=lambda r: r["units_change_pct"])[:8],
                "mix": sorted(mixd, key=key)[:8],
                "price": sorted(prc, key=key)[:8]}
