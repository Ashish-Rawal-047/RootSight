"""Analysis panels.

Two shapes are needed downstream:

  * a **time-series panel** (one row per day) of every node in the causal graph,
    restricted to the caller's entitled scope, used by the structure screen, the
    temporal-compatibility gate and ITS;
  * a **unit panel** (one row per cell per day) used by DID.

Both carry a per-column coverage report.  A column that is unavailable in the
requested scope is absent-and-declared, never zero-filled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ..bizcalendar import attributes
from ..ingest.reconcile import ConformedData

GRAPH_COLUMNS = ["net_revenue", "units_sold", "avg_selling_price",
                 "on_time_dispatch_rate", "complaint_rate", "marketing_spend",
                 "competitor_promo", "seasonality", "promo_calendar",
                 "subscription_arr"]


@dataclass
class Panel:
    frame: pd.DataFrame
    scope: dict
    coverage: dict[str, dict] = field(default_factory=dict)
    unavailable: list[str] = field(default_factory=list)
    temporal_resolution: dict[str, float] = field(default_factory=dict)

    def available(self, cols: list[str]) -> list[str]:
        return [c for c in cols if c in self.frame.columns and c not in self.unavailable]

    def window(self, start: date, end: date) -> pd.DataFrame:
        f = self.frame
        return f[(f["date"] >= start) & (f["date"] <= end)]

    def coverage_report(self) -> dict:
        return {"scope": self.scope, "columns": self.coverage,
                "unavailable": self.unavailable,
                "temporal_resolution_days": self.temporal_resolution}


class PanelBuilder:
    def __init__(self, conformed: ConformedData):
        self.c = conformed

    # ------------------------------------------------------------- time series
    def timeseries(self, *, regions: list[str], product_lines: list[str] | None = None,
                   start: date | None = None, end: date | None = None) -> Panel:
        f = self.c.fact_daily
        f = f[f["region"].isin(regions) & ~f["_quarantined"]]
        if product_lines:
            f = f[f["product_line"].isin(product_lines)]
        agg = (f.groupby("date")[["gross_sales", "refunds", "discounts",
                                  "units_shipped", "units_returned"]].sum().reset_index())
        agg["net_revenue"] = agg["gross_sales"] - agg["refunds"] - agg["discounts"]
        agg["units_sold"] = agg["units_shipped"] - agg["units_returned"]
        agg["avg_selling_price"] = agg["net_revenue"] / agg["units_sold"].replace(0, np.nan)
        sub = (f.groupby("date")["monthly_subscription_revenue"].sum().reset_index())
        sub["subscription_arr"] = 12.0 * sub["monthly_subscription_revenue"]
        # a KPI that did not exist before its launch date is NaN, never zero
        sub.loc[sub["monthly_subscription_revenue"] <= 0, "subscription_arr"] = np.nan
        agg = agg.merge(sub[["date", "subscription_arr"]], on="date", how="left")

        ops = self.c.ops_region_daily
        ops = ops[ops["region"].isin(regions)]
        o = (ops.groupby("date")[["dispatched_within_sla", "dispatch_attempts",
                                  "complaint_tickets", "shipped_orders"]].sum().reset_index())
        o["on_time_dispatch_rate"] = o["dispatched_within_sla"] / o["dispatch_attempts"]
        o["complaint_rate"] = 1000.0 * o["complaint_tickets"] / o["shipped_orders"]
        df = agg.merge(o[["date", "on_time_dispatch_rate", "complaint_rate"]],
                       on="date", how="left")

        mkt = self.c.mkt_region_daily
        mkt_scope = mkt[mkt["region"].isin(regions)]
        unavailable: list[str] = []
        if mkt_scope.empty:
            df["marketing_spend"] = np.nan
            unavailable.append("marketing_spend")
        else:
            covered = sorted(set(mkt_scope["region"].unique()))
            missing = [r for r in regions if r not in covered]
            m = mkt_scope.groupby("date")["marketing_spend"].sum().reset_index()
            df = df.merge(m, on="date", how="left")
            if missing:
                # partial coverage: keep the column but record which regions are absent
                self._partial_marketing = missing
        ext = self.c.ext_line_daily
        if product_lines:
            ext = ext[ext["product_line"].isin(product_lines)]
        e = (ext.groupby("date")
                .apply(lambda g: (np.nan if g["promo_active"].notna().sum() == 0
                                  else float(g["promo_active"].mean(skipna=True))),
                       include_groups=False)
                .rename("competitor_promo").reset_index())
        df = df.merge(e, on="date", how="left")

        attrs = pd.DataFrame([attributes(d) for d in df["date"]])
        df["promo_calendar"] = attrs["promo_window"].notna().astype(float).to_numpy()
        df["is_holiday"] = attrs["is_holiday"].astype(float).to_numpy()
        df["dow"] = attrs["dow"].to_numpy()
        df["seasonality"] = self._seasonal_index(df)

        if start:
            df = df[df["date"] >= start]
        if end:
            df = df[df["date"] <= end]
        df = df.sort_values("date").reset_index(drop=True)

        coverage = {}
        for col in GRAPH_COLUMNS:
            if col not in df.columns:
                coverage[col] = {"observed_pct": 0.0, "status": "ABSENT"}
                if col not in unavailable:
                    unavailable.append(col)
                continue
            obs = float(df[col].notna().mean()) * 100.0
            coverage[col] = {
                "observed_pct": round(obs, 1),
                "status": ("COMPLETE" if obs > 99.0 else
                           "PARTIAL" if obs >= 50.0 else
                           "SPARSE" if obs > 0 else "ABSENT"),
                "n_observed": int(df[col].notna().sum()),
                "n_rows": int(len(df)),
            }
            if obs == 0.0 and col not in unavailable:
                unavailable.append(col)
        res = {c: self.c.temporal_resolution(c) for c in GRAPH_COLUMNS}
        return Panel(frame=df,
                     scope={"regions": regions, "product_lines": product_lines or "ALL",
                            "start": str(df["date"].min()), "end": str(df["date"].max())},
                     coverage=coverage, unavailable=unavailable, temporal_resolution=res)

    @staticmethod
    def _seasonal_index(df: pd.DataFrame) -> np.ndarray:
        """A deterministic, calendar-only nuisance covariate.

        Estimated from day-of-week and holiday structure on the FIRST 60% of the
        span so it cannot absorb the movement being investigated.
        """
        y = df["units_sold"].to_numpy(dtype=float)
        n = len(y)
        train = slice(0, max(30, int(n * 0.6)))
        X = np.column_stack([
            np.ones(n),
            *[(df["dow"].to_numpy() == k).astype(float) for k in range(1, 7)],
            df["is_holiday"].to_numpy(dtype=float),
        ])
        yt = y[train]
        Xt = X[train]
        ok = ~np.isnan(yt)
        if ok.sum() < 20:
            return np.zeros(n)
        beta = np.linalg.lstsq(Xt[ok], yt[ok], rcond=None)[0]
        fitted = X @ beta
        base = float(beta[0]) if beta[0] != 0 else 1.0
        return fitted / base

    # ---------------------------------------------------------------- DID unit
    def unit_panel(self, *, outcome: str = "units_sold",
                   regions: list[str] | None = None,
                   exclude_product_lines: tuple[str, ...] = (),
                   start: date | None = None, end: date | None = None) -> pd.DataFrame:
        f = self.c.fact_daily
        f = f[~f["_quarantined"]]
        if regions:
            f = f[f["region"].isin(regions)]
        if exclude_product_lines:
            f = f[~f["product_line"].isin(exclude_product_lines)]
        if start:
            f = f[f["date"] >= start]
        if end:
            f = f[f["date"] <= end]
        g = (f.groupby(["date", "region", "product_line"])[
                 ["gross_sales", "refunds", "discounts", "units_shipped", "units_returned"]]
              .sum().reset_index())
        g["net_revenue"] = g["gross_sales"] - g["refunds"] - g["discounts"]
        g["units_sold"] = g["units_shipped"] - g["units_returned"]
        g["cell"] = g["region"] + " | " + g["product_line"]
        g["outcome"] = g[outcome]
        attrs = pd.DataFrame([attributes(d) for d in g["date"]])
        g["is_holiday"] = attrs["is_holiday"].astype(float).to_numpy()
        g["promo_calendar"] = attrs["promo_window"].notna().astype(float).to_numpy()
        g["dow"] = attrs["dow"].to_numpy()
        return g.sort_values(["cell", "date"]).reset_index(drop=True)
