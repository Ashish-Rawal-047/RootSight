"""KPI calculation driven by the semantic contract, with end-to-end lineage.

Every value returned here can answer "where did this number come from?" down to
source table, partition, row count, sample row ids, transformation chain and the
formula string from the contract.  Access control is applied *before*
aggregation, so an unauthorised row never contributes to a number the caller can
see.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .. import config
from ..bizcalendar import attributes
from ..contracts.kpi_contract import (ContractViolation, KpiContractRegistry,
                                      KpiDefinition, registry)
from ..ingest.reconcile import ConformedData
from ..security.policy import AccessDecision


@dataclass
class KpiSeries:
    kpi_id: str
    definition: KpiDefinition
    frame: pd.DataFrame              # columns: date, <dims...>, value, (+ components)
    grain: str
    unit: str
    lineage: dict
    excluded_rows: int = 0
    excluded_reason: list[str] = field(default_factory=list)
    access: AccessDecision | None = None

    def total_by_date(self) -> pd.Series:
        """Aggregate across dimensions for each date, honouring the contract.

        RATIO KPIs are recomputed from their components (never averaged); FLOW
        and STOCK KPIs are summed across dimensions.  Averaging a ratio of
        ratios is the single most common KPI arithmetic error and the contract
        makes it unrepresentable here.
        """
        if self.definition.time_aggregation == "RATIO":
            num, den = self._ratio_columns()
            g = self.frame.groupby("date")[[num, den]].sum()
            return (g[num] / g[den].replace(0, np.nan)) * self._ratio_scale()
        return self.frame.groupby("date")["value"].sum()

    def _ratio_columns(self) -> tuple[str, str]:
        return {"avg_selling_price": ("net_revenue", "units_sold"),
                "on_time_dispatch_rate": ("dispatched_within_sla", "dispatch_attempts"),
                "complaint_rate": ("complaint_tickets", "shipped_orders")}[self.kpi_id]

    def _ratio_scale(self) -> float:
        return 1000.0 if self.kpi_id == "complaint_rate" else 1.0

    def window_value(self, start: date, end: date) -> float:
        """Window aggregation dictated by `time_aggregation` in the contract."""
        s = self.total_by_date().sort_index()
        s = s[(s.index >= start) & (s.index <= end)]
        if s.empty:
            return float("nan")
        ta = self.definition.time_aggregation
        if ta == "FLOW":
            return float(s.sum())
        if ta == "STOCK":
            return float(s.iloc[-1])          # a stock is a level, never a sum over days
        num, den = self._ratio_columns()
        f = self.frame
        f = f[(f["date"] >= start) & (f["date"] <= end)]
        d = float(f[den].sum())
        return float(f[num].sum() / d * self._ratio_scale()) if d else float("nan")

    def window_daily_mean(self, start: date, end: date) -> float:
        s = self.total_by_date()
        s = s[(s.index >= start) & (s.index <= end)]
        return float(s.mean())

    def series_array(self, start: date, end: date) -> tuple[np.ndarray, list[date]]:
        s = self.total_by_date().sort_index()
        s = s[(s.index >= start) & (s.index <= end)]
        return s.to_numpy(dtype=float), list(s.index)


class KpiEngine:
    def __init__(self, conformed: ConformedData, reg: KpiContractRegistry | None = None):
        self.c = conformed
        self.reg = reg or registry()

    # ------------------------------------------------------------------ public
    def compute(self, kpi_id: str, decision: AccessDecision,
                *, start: date | None = None, end: date | None = None) -> KpiSeries:
        k = self.reg.get(kpi_id)
        if not decision.granted or decision.kpi_id != kpi_id:
            raise ContractViolation(
                f"compute({kpi_id}) called without a granted access decision")
        builder = {
            "net_revenue": self._erp_kpi, "gross_revenue": self._erp_kpi,
            "units_sold": self._erp_kpi, "avg_selling_price": self._erp_kpi,
            "subscription_arr": self._erp_kpi,
            "recognized_revenue_finance": self._period_kpi,
            "on_time_dispatch_rate": self._dispatch_kpi,
            "complaint_rate": self._complaint_kpi,
        }[kpi_id]
        series = builder(k, decision)
        if start or end:
            f = series.frame
            if start:
                f = f[f["date"] >= start]
            if end:
                f = f[f["date"] <= end]
            series.frame = f
        return series

    # ------------------------------------------------------------------- ERP
    def _erp_kpi(self, k: KpiDefinition, dec: AccessDecision) -> KpiSeries:
        df = self.c.fact_daily.copy()
        excluded = []
        n0 = len(df)
        df = df[df["region"].isin(dec.allowed_regions)]
        if len(df) < n0:
            excluded.append(f"row-level security: {n0 - len(df)} rows outside "
                            f"{list(dec.allowed_regions)}")
        n1 = len(df)
        q = int(df["_quarantined"].sum())
        df = df[~df["_quarantined"]]
        if q:
            excluded.append(f"data quality: {q} quarantined cells excluded (not zeroed)")

        dims = self._dims_for(dec.effective_grain, k)
        if k.kpi_id == "subscription_arr":
            df = df[df["monthly_subscription_revenue"] > 0]
            g = df.groupby(["date"] + dims)["monthly_subscription_revenue"].sum().reset_index()
            g["value"] = 12.0 * g["monthly_subscription_revenue"]
        elif k.kpi_id == "avg_selling_price":
            g = (df.groupby(["date"] + dims)[["net_revenue", "units_sold"]]
                   .sum().reset_index())
            g["value"] = g["net_revenue"] / g["units_sold"].replace(0, np.nan)
        elif k.kpi_id == "units_sold":
            g = (df.groupby(["date"] + dims)[["units_shipped", "units_returned"]]
                   .sum().reset_index())
            g["value"] = g.apply(
                lambda r: k.compute({"units_shipped": r["units_shipped"],
                                     "units_returned": r["units_returned"]}), axis=1)
            g["units_sold"] = g["value"]
        elif k.kpi_id == "gross_revenue":
            g = df.groupby(["date"] + dims)[["gross_sales"]].sum().reset_index()
            g["value"] = g["gross_sales"]
        else:                                        # net_revenue
            g = (df.groupby(["date"] + dims)[["gross_sales", "refunds", "discounts"]]
                   .sum().reset_index())
            g["value"] = g.apply(
                lambda r: k.compute({"gross_sales": r["gross_sales"],
                                     "refunds": r["refunds"],
                                     "discounts": r["discounts"]}), axis=1)
            g["net_revenue"] = g["value"]
        return KpiSeries(kpi_id=k.kpi_id, definition=k, frame=g,
                         grain=dec.effective_grain, unit=k.unit,
                         lineage=self._lineage(k, dec, df, dims, n1),
                         excluded_rows=(n0 - n1) + q, excluded_reason=excluded, access=dec)

    def _period_kpi(self, k: KpiDefinition, dec: AccessDecision) -> KpiSeries:
        df = self.c.fact_daily.copy()
        df = df[df["region"].isin(dec.allowed_regions) & ~df["_quarantined"]]
        g = (df.groupby(["fiscal_period", "region"])[
                ["gross_sales", "refunds", "discounts", "deferred_delta"]]
               .sum().reset_index())
        g["value"] = g.apply(lambda r: k.compute({
            "gross_sales": r["gross_sales"], "refunds": r["refunds"],
            "discounts": r["discounts"], "deferred_delta": r["deferred_delta"]}), axis=1)
        # a fiscal-period KPI has no daily date; use the last business date observed
        # in each period so the value remains joinable to the calendar
        last = df.groupby("fiscal_period")["date"].max().to_dict()
        g["date"] = [last[p] for p in g["fiscal_period"]]
        return KpiSeries(kpi_id=k.kpi_id, definition=k, frame=g,
                         grain=dec.effective_grain, unit=k.unit,
                         lineage=self._lineage(k, dec, df, ["region"], len(df)), access=dec)

    # ------------------------------------------------------------------- OPS
    def _dispatch_kpi(self, k: KpiDefinition, dec: AccessDecision) -> KpiSeries:
        wh = self.c.ops_wh_daily.copy()
        wh = wh[wh["region"].isin(dec.allowed_regions)]
        excluded = []
        if dec.allowed_warehouses:
            wh = wh[wh["warehouse"].isin(dec.allowed_warehouses)]
        dims = ["warehouse", "region"] if "warehouse" in dec.effective_grain else ["region"]
        if "warehouse" not in dec.effective_grain:
            excluded.append("grain security: warehouse detail aggregated away for this role")
        g = (wh.groupby(["date"] + dims)[["dispatched_within_sla", "dispatch_attempts",
                                          "mean_actual_hours"]].sum().reset_index())
        g["value"] = g.apply(lambda r: k.compute({
            "dispatched_within_sla": r["dispatched_within_sla"],
            "dispatch_attempts": r["dispatch_attempts"]}), axis=1)
        return KpiSeries(kpi_id=k.kpi_id, definition=k, frame=g,
                         grain=dec.effective_grain, unit=k.unit,
                         lineage=self._lineage(k, dec, wh, dims, len(wh)),
                         excluded_reason=excluded, access=dec)

    def _complaint_kpi(self, k: KpiDefinition, dec: AccessDecision) -> KpiSeries:
        reg = self.c.ops_region_daily.copy()
        reg = reg[reg["region"].isin(dec.allowed_regions)]
        g = (reg.groupby(["date", "region"])[["complaint_tickets", "shipped_orders"]]
               .sum().reset_index())
        g["value"] = g.apply(lambda r: k.compute({
            "complaint_tickets": r["complaint_tickets"],
            "shipped_orders": r["shipped_orders"]}), axis=1)
        return KpiSeries(kpi_id=k.kpi_id, definition=k, frame=g,
                         grain=dec.effective_grain, unit=k.unit,
                         lineage=self._lineage(k, dec, reg, ["region"], len(reg)), access=dec)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _dims_for(effective_grain: str, k: KpiDefinition) -> list[str]:
        parts = [p.strip() for p in effective_grain.split("x")]
        dims = [p for p in parts if p not in ("day", "fiscal_period")]
        available = {"region", "product_line", "warehouse", "ticket_category"}
        return [d for d in dims if d in available]

    def _lineage(self, k: KpiDefinition, dec: AccessDecision, df: pd.DataFrame,
                 dims: list[str], rows_used: int) -> dict:
        transforms = [t.as_dict() for t in self.c.transforms
                      if any(s in t.source_id for s in k.data_sources)]
        sample_ids = []
        for col in ("source_row_id", "event_id"):
            if col in df.columns:
                sample_ids = df[col].head(5).tolist()
                break
        partitions = []
        if "date" in df.columns and len(df):
            try:
                partitions = [f"date={min(df['date'])}..{max(df['date'])}"]
            except (TypeError, ValueError):
                partitions = []
        card = self.reg.lineage_card(k.kpi_id)
        card.update({
            "requested_by": dec.persona.persona_id,
            "effective_grain": dec.effective_grain,
            "row_scope_applied": list(dec.allowed_regions),
            "column_scope_applied": sorted(dec.denied_fields),
            "aggregation_dimensions": dims,
            "source_rows_used": int(rows_used),
            "source_partitions": partitions,
            "sample_source_row_ids": sample_ids,
            "grain_transforms": transforms,
            "calendar_notes": list(self.c.calendar_notes),
            "model_versions": {"kpi_engine": config.MODEL_VERSIONS["rootsight"]},
        })
        return card
