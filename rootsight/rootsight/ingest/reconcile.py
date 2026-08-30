"""Grain, cadence and calendar reconciliation.

Three source grains become one analytical grain (day x region x product_line)
plus two conformed side tables (day x warehouse, day x region).  Every
transformation emits a GrainTransform record which travels with the data into
lineage, and — critically — carries `temporal_resolution_days`.

The reason that field exists: marketing arrives weekly.  Allocating it to days
does not create daily information.  A driver whose true temporal resolution is
7 days cannot support a claim about a 3-day lag, no matter how the numbers are
spread.  The temporal-compatibility gate reads this field and refuses.  Grain
reconciliation is therefore not a convenience step; it is where a whole class of
false causal claims is prevented.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .. import config
from ..bizcalendar import attributes, iso_week_start
from .loaders import SourceBundle


@dataclass
class GrainTransform:
    transform_id: str
    source_id: str
    from_grain: str
    to_grain: str
    method: str
    temporal_resolution_days: float
    is_imputed: bool
    note: str
    rows_in: int = 0
    rows_out: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConformedData:
    fact_daily: pd.DataFrame                  # day x region x product_line
    ops_wh_daily: pd.DataFrame                # day x warehouse
    ops_region_daily: pd.DataFrame            # day x region
    mkt_region_daily: pd.DataFrame            # day x region (allocated from weekly)
    ext_line_daily: pd.DataFrame              # day x product_line (partial)
    transforms: list[GrainTransform] = field(default_factory=list)
    calendar_notes: list[str] = field(default_factory=list)

    def transform_table(self) -> list[dict]:
        return [t.as_dict() for t in self.transforms]

    def temporal_resolution(self, driver_id: str) -> float:
        """Finest lag, in days, that a driver series can legitimately support."""
        mapping = {
            "marketing_spend": 7.0,
            "competitor_promo": 1.0,
            "on_time_dispatch_rate": 1.0,
            "complaint_rate": 1.0,
            "avg_selling_price": 1.0,
            "seasonality": 1.0,
            "promo_calendar": 1.0,
        }
        return mapping.get(driver_id, 1.0)


class Reconciler:
    def __init__(self, bundle: SourceBundle):
        self.b = bundle
        self.transforms: list[GrainTransform] = []
        self.notes: list[str] = []

    # ------------------------------------------------------------------ facts
    def _fact_daily(self) -> pd.DataFrame:
        o = self.b.orders
        r = self.b.refunds[["date", "region", "product_line", "refunds"]]
        df = o.merge(r, on=["date", "region", "product_line"], how="left")
        df["refunds"] = df["refunds"].fillna(0.0)

        # contract formulas: net_revenue = gross_sales - refunds - discounts
        df["net_revenue"] = df["gross_sales"] - df["refunds"] - df["discounts"]
        df["units_sold"] = df["units_shipped"] - df["units_returned"]
        df.loc[df["_quarantined"], ["net_revenue", "units_sold"]] = np.nan
        df["avg_selling_price"] = df["net_revenue"] / df["units_sold"].replace(0, np.nan)

        attrs = pd.DataFrame([attributes(d) for d in df["date"]])
        df = pd.concat([df.reset_index(drop=True),
                        attrs[["fiscal_period", "fiscal_week_start", "iso_week_start",
                               "is_weekend", "is_holiday", "promo_window",
                               "baseline_eligible"]].reset_index(drop=True)], axis=1)

        self.transforms.append(GrainTransform(
            transform_id="TR-ERP-01", source_id="SRC_ERP",
            from_grain="day x region x product_line (raw, duplicated)",
            to_grain="day x region x product_line (conformed)",
            method="dedupe_by_latest_ingest -> contract_formula_eval -> calendar_stamp",
            temporal_resolution_days=1.0, is_imputed=False,
            note=("net_revenue and units_sold computed from the contract formulas; "
                  "quarantined cells are NaN, never zero"),
            rows_in=len(o), rows_out=len(df)))
        self.notes.append(
            "ERP is stamped with fiscal weeks (Sunday-start); marketing uses ISO weeks "
            "(Monday-start). Both week keys are retained so neither is silently coerced.")
        return df

    # -------------------------------------------------------------------- ops
    def _ops(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        d = self.b.dispatch
        ok = d[~d["_quarantined"] & d["date"].notna()]
        wh = (ok.groupby(["date", "warehouse", "region"])
                .agg(dispatch_attempts=("sla_met", "size"),
                     dispatched_within_sla=("sla_met", "sum"),
                     mean_actual_hours=("actual_hours", "mean"),
                     p90_actual_hours=("actual_hours", lambda s: float(np.percentile(s, 90))))
                .reset_index())
        wh["on_time_dispatch_rate"] = wh["dispatched_within_sla"] / wh["dispatch_attempts"]
        self.transforms.append(GrainTransform(
            transform_id="TR-OPS-01", source_id="SRC_OPS",
            from_grain="event-level dispatch scans (mixed timestamp conventions)",
            to_grain="day x warehouse",
            method="timestamp_normalise_to_IST -> quarantine_missing_duration -> daily_aggregate",
            temporal_resolution_days=1.0, is_imputed=False,
            note=("event feed is near-real-time; aggregation to daily is a downgrade of "
                  "resolution to match the ERP grain, not an upgrade of information"),
            rows_in=len(d), rows_out=len(wh)))

        reg = (wh.groupby(["date", "region"])
                 .agg(dispatch_attempts=("dispatch_attempts", "sum"),
                      dispatched_within_sla=("dispatched_within_sla", "sum"),
                      mean_actual_hours=("mean_actual_hours", "mean"))
                 .reset_index())
        reg["on_time_dispatch_rate"] = reg["dispatched_within_sla"] / reg["dispatch_attempts"]

        # complaint_rate needs the ERP order count: a cross-source, cross-cadence join
        t = self.b.tickets
        t = t[~t["_quarantined"]]
        tick = (t.groupby(["date", "region"])
                 .agg(complaint_tickets=("complaint_tickets", "sum"),
                      total_tickets=("total_tickets", "sum"))
                 .reset_index())
        orders_daily = (self.b.orders[~self.b.orders["_quarantined"]]
                        .groupby(["date", "region"])["shipped_orders"].sum().reset_index())
        reg = reg.merge(tick, on=["date", "region"], how="left").merge(
            orders_daily, on=["date", "region"], how="left")
        reg["complaint_rate"] = 1000.0 * reg["complaint_tickets"] / reg["shipped_orders"]
        self.transforms.append(GrainTransform(
            transform_id="TR-OPS-02", source_id="SRC_OPS+SRC_ERP",
            from_grain="15-minute ops micro-batch  x  daily 02:00 ERP batch",
            to_grain="day x region",
            method="cadence_align(ops -> erp business date) -> ratio_with_erp_denominator",
            temporal_resolution_days=1.0, is_imputed=False,
            note=("complaint_rate mixes a near-real-time numerator with a daily "
                  "denominator; the most recent day is therefore provisional until the "
                  "02:00 ERP batch lands"),
            rows_in=len(t), rows_out=len(reg)))
        return wh, reg

    # -------------------------------------------------------------- marketing
    def _marketing(self) -> pd.DataFrame:
        m = self.b.marketing
        wk = (m.groupby(["week_start", "region"])["spend_inr"].sum().reset_index())
        rows = []
        for _, row in wk.iterrows():
            for i in range(7):
                d = row["week_start"] + timedelta(days=i)
                if d > config.HISTORY_END:
                    continue
                rows.append({"date": d, "region": row["region"],
                             "marketing_spend": row["spend_inr"] / 7.0,
                             "allocation_method": "WEEKLY_UNIFORM",
                             "is_imputed": True,
                             "source_week_start": row["week_start"]})
        df = pd.DataFrame(rows)
        self.transforms.append(GrainTransform(
            transform_id="TR-MKT-01", source_id="SRC_MKT",
            from_grain="iso_week (Monday-start) x campaign x region",
            to_grain="day x region",
            method="sum_campaigns -> uniform_within_week_allocation",
            temporal_resolution_days=7.0, is_imputed=True,
            note=("allocation spreads a weekly total across 7 days. It does NOT create "
                  "daily information: temporal_resolution_days stays 7, and the "
                  "temporal-compatibility gate will refuse any lag claim finer than that. "
                  "Region South emits no rows at all and is therefore UNAVAILABLE, not zero."),
            rows_in=len(m), rows_out=len(df)))
        self.notes.append(
            "Marketing weeks start Monday; ERP fiscal weeks start Sunday. A weekly "
            "marketing figure therefore straddles two ERP fiscal weeks by one day.")
        return df

    # --------------------------------------------------------------- external
    def _external(self) -> pd.DataFrame:
        e = self.b.external
        full = pd.MultiIndex.from_product(
            [[config.HISTORY_START + timedelta(days=i)
              for i in range((config.HISTORY_END - config.HISTORY_START).days + 1)],
             config.PRODUCT_LINES], names=["date", "product_line"]).to_frame(index=False)
        df = full.merge(e[["date", "product_line", "promo_active"]],
                        on=["date", "product_line"], how="left")
        df["observed"] = df["promo_active"].notna()
        # unobserved stays NaN: UNKNOWN is not the same as 0
        self.transforms.append(GrainTransform(
            transform_id="TR-EXT-01", source_id="SRC_EXT",
            from_grain="day x product_line (approx 63% observed)",
            to_grain="day x product_line (explicit NaN for unobserved)",
            method="left_join_on_full_calendar -> preserve_NaN",
            temporal_resolution_days=1.0, is_imputed=False,
            note=("unobserved cells are left NaN. Filling them with 0 would assert "
                  "'no competitor promotion' from an absence of evidence"),
            rows_in=len(e), rows_out=len(df)))
        return df

    # ------------------------------------------------------------------- run
    def run(self) -> ConformedData:
        fact = self._fact_daily()
        wh, reg = self._ops()
        mkt = self._marketing()
        ext = self._external()
        return ConformedData(fact_daily=fact, ops_wh_daily=wh, ops_region_daily=reg,
                             mkt_region_daily=mkt, ext_line_daily=ext,
                             transforms=self.transforms, calendar_notes=self.notes)


def conform(bundle: SourceBundle | None = None) -> tuple[SourceBundle, ConformedData]:
    b = bundle or SourceBundle()
    return b, Reconciler(b).run()
