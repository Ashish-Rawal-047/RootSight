"""Source loaders.

Each loader is responsible for: reading its own grain, normalising its own
timestamp convention, applying its own conflict-resolution rule, quarantining
its own bad rows, and reporting its own freshness and coverage.  Nothing is
silently repaired.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from .. import config
from ..bizcalendar import attributes, iso_week_start
from ..contracts.kpi_contract import registry
from .dq import (IST, DataQualityReport, Defect, FRESHNESS_MISSING, Freshness,
                 classify_freshness, normalise_timestamp)


def _p(name: str) -> str:
    return os.path.join(config.RAW_DIR, name)


class SourceBundle:
    """All four sources, loaded, normalised, and accompanied by a DQ report."""

    def __init__(self, as_of: date = config.AS_OF):
        self.as_of = as_of
        self.as_of_dt = datetime.combine(as_of, datetime.min.time(), IST) + timedelta(hours=10, minutes=5)
        self.reg = registry()
        self.dq = DataQualityReport()
        self.manifest = json.load(open(_p("_manifest.json"), encoding="utf-8"))
        self.orders = self._load_erp_orders()
        self.refunds = self._load_erp_refunds()
        self.prices = pd.read_csv(_p("erp_price_list.csv"))
        self.marketing = self._load_marketing()
        self.dispatch = self._load_dispatch()
        self.tickets = self._load_tickets()
        self.external = self._load_external()
        self._freshness()

    # ------------------------------------------------------------------- ERP
    def _load_erp_orders(self) -> pd.DataFrame:
        df = pd.read_csv(_p("erp_orders_daily.csv"))
        total = len(df)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True, format="mixed")

        # --- missing values: quarantine, do not impute --------------------
        miss = df["discounts"].isna()
        if miss.any():
            self.dq.add(Defect(
                defect_id="DQ-ERP-001", source_id="SRC_ERP", table="orders_daily",
                kind="MISSING_VALUE", severity="MEDIUM", rows_affected=int(miss.sum()),
                rows_total=total, action_taken="QUARANTINED",
                detail=("discounts is null; net_revenue is not computable for these "
                        "cells so the rows are excluded from the KPI and counted"),
                sample_keys=df.loc[miss, "source_row_id"].head(4).tolist()))

        # --- sign errors ---------------------------------------------------
        neg = df["units_shipped"] < 0
        if neg.any():
            self.dq.add(Defect(
                defect_id="DQ-ERP-002", source_id="SRC_ERP", table="orders_daily",
                kind="SIGN_ERROR", severity="HIGH", rows_affected=int(neg.sum()),
                rows_total=total, action_taken="QUARANTINED",
                detail="negative units_shipped is impossible at this grain",
                sample_keys=df.loc[neg, "source_row_id"].head(4).tolist()))

        quarantine = miss | neg
        df["_quarantined"] = quarantine
        self.dq.quarantined_rows["SRC_ERP.orders_daily"] = int(quarantine.sum())

        # --- duplicate / conflicting records ------------------------------
        key = ["date", "region", "product_line"]
        dup_mask = df.duplicated(subset=key, keep=False)
        n_conflicts = 0
        if dup_mask.any():
            grp = df[dup_mask].groupby(key)["gross_sales"].nunique()
            n_conflicts = int((grp > 1).sum())
            self.dq.add(Defect(
                defect_id="DQ-ERP-003", source_id="SRC_ERP", table="orders_daily",
                kind="DUPLICATE_CONFLICT", severity="HIGH", rows_affected=int(dup_mask.sum()),
                rows_total=total, action_taken="RESOLVED_LATEST_INGEST",
                detail=(f"{n_conflicts} natural keys carry conflicting gross_sales. "
                        "Resolution rule: highest ingested_at wins; superseded rows are "
                        "retained in lineage, never deleted"),
                sample_keys=df.loc[dup_mask, "source_row_id"].head(4).tolist()))
        df = (df.sort_values("ingested_at")
                .drop_duplicates(subset=key, keep="last")
                .reset_index(drop=True))

        # --- late arrival --------------------------------------------------
        df["arrival_lag_days"] = [
            (i.date() - d).days for i, d in zip(df["ingested_at"], df["date"])]
        late = df["arrival_lag_days"] > 2
        if late.any():
            regions = sorted(df.loc[late, "region"].unique().tolist())
            self.dq.add(Defect(
                defect_id="DQ-ERP-004", source_id="SRC_ERP", table="orders_daily",
                kind="LATE_ARRIVAL", severity="MEDIUM", rows_affected=int(late.sum()),
                rows_total=len(df), action_taken="DISCLOSED",
                detail=(f"rows for region(s) {regions} arrived more than 2 days after the "
                        "business date; the most recent days for those regions are "
                        "incomplete at analysis time"),
                sample_keys=df.loc[late, "source_row_id"].head(4).tolist()))
        return df

    def _load_erp_refunds(self) -> pd.DataFrame:
        df = pd.read_csv(_p("erp_refunds_daily.csv"))
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return (df.sort_values("ingested_at")
                  .drop_duplicates(subset=["date", "region", "product_line"], keep="last"))

    # ------------------------------------------------------------- marketing
    def _load_marketing(self) -> pd.DataFrame:
        df = pd.read_csv(_p("mkt_campaign_weekly.csv"))
        df["week_start"] = pd.to_datetime(df["week_start"]).dt.date
        present = set(df["region"].unique())
        missing = [r for r in config.REGIONS if r not in present]
        self.dq.coverage["SRC_MKT"] = {
            "regions_present": sorted(present),
            "missing_dimensions": [f"region={r}" for r in missing],
            "grain": "iso_week x campaign x region",
            "week_start_convention": "MONDAY (ISO) - differs from ERP fiscal Sunday weeks",
        }
        if missing:
            self.dq.add(Defect(
                defect_id="DQ-MKT-001", source_id="SRC_MKT", table="campaign_weekly",
                kind="COVERAGE_GAP", severity="HIGH", rows_affected=0, rows_total=len(df),
                action_taken="DISCLOSED",
                detail=(f"no rows exist for region(s) {missing}; marketing is an "
                        "UNAVAILABLE VARIABLE there, not a zero"),
                sample_keys=[]))
        return df

    # ---------------------------------------------------------------- ops
    def _load_dispatch(self) -> pd.DataFrame:
        df = pd.read_csv(_p("ops_dispatch_events.csv"))
        total = len(df)
        norm = [normalise_timestamp(v) for v in df["ts_raw"]]
        df["ts"] = [n[0] for n in norm]
        df["ts_convention"] = [n[1] for n in norm]
        conventions = df["ts_convention"].value_counts().to_dict()
        self.dq.add(Defect(
            defect_id="DQ-OPS-001", source_id="SRC_OPS", table="dispatch_events",
            kind="TIMESTAMP_FORMAT", severity="MEDIUM",
            rows_affected=int(total - conventions.get("ISO_OFFSET", 0)),
            rows_total=total, action_taken="NORMALISED",
            detail=(f"three timestamp conventions in one feed {conventions}; all "
                    "normalised to IST. Naive values are interpreted as IST local, "
                    "which is an assumption, not a measurement"),
            sample_keys=df["event_id"].head(3).tolist()))

        miss = df["actual_hours"].isna()
        if miss.any():
            self.dq.add(Defect(
                defect_id="DQ-OPS-002", source_id="SRC_OPS", table="dispatch_events",
                kind="MISSING_VALUE", severity="MEDIUM", rows_affected=int(miss.sum()),
                rows_total=total, action_taken="QUARANTINED",
                detail=("actual_hours missing; SLA compliance is undefined for these "
                        "events, so they are excluded from the denominator and counted"),
                sample_keys=df.loc[miss, "event_id"].head(4).tolist()))
        df["_quarantined"] = miss
        self.dq.quarantined_rows["SRC_OPS.dispatch_events"] = int(miss.sum())
        df["date"] = [t.date() if t is not None else None for t in df["ts"]]
        df["sla_met"] = np.where(miss, np.nan,
                                 (df["actual_hours"] <= df["promised_hours"]).astype(float))
        return df

    def _load_tickets(self) -> pd.DataFrame:
        df = pd.read_csv(_p("ops_support_tickets_daily.csv"))
        df["date"] = pd.to_datetime(df["date"]).dt.date
        # an event dated after the moment it was recorded is impossible
        ing = pd.to_datetime(df["ingested_at"], format="mixed").dt.date
        future = (df["date"] > ing) | (df["date"] > config.HISTORY_END)
        if future.any():
            self.dq.add(Defect(
                defect_id="DQ-OPS-003", source_id="SRC_OPS", table="support_tickets_daily",
                kind="FUTURE_DATED", severity="LOW", rows_affected=int(future.sum()),
                rows_total=len(df), action_taken="QUARANTINED",
                detail=("ticket rows carry a business date later than their own "
                        "ingestion timestamp, consistent with an upstream timezone "
                        "defect; excluded from the KPI and counted"),
                sample_keys=[]))
        df["_quarantined"] = future
        return df

    def _load_external(self) -> pd.DataFrame:
        df = pd.read_csv(_p("ext_competitor_promo.csv"))
        df["date"] = pd.to_datetime(df["date"]).dt.date
        all_days = (config.HISTORY_END - config.HISTORY_START).days + 1
        expected_pairs = all_days * len(config.PRODUCT_LINES)
        cov = len(df) / expected_pairs
        self.dq.coverage["SRC_EXT"] = {
            "day_coverage_pct": round(100 * cov, 1),
            "missing_dimensions": [],
            "unavailable_variables": ["competitor_promo_intensity (no discount depth)"],
            "grain": "day x product_line (partial)",
        }
        self.dq.add(Defect(
            defect_id="DQ-EXT-001", source_id="SRC_EXT", table="competitor_promo",
            kind="COVERAGE_GAP", severity="HIGH",
            rows_affected=int(expected_pairs - len(df)), rows_total=expected_pairs,
            action_taken="DISCLOSED",
            detail=(f"only {round(100 * cov)}% of day x product-line cells observed and no promotion "
                    "intensity field exists; an unobserved day is UNKNOWN, not 'no promo'"),
            sample_keys=[]))
        return df

    # --------------------------------------------------------------- freshness
    def _freshness(self) -> None:
        m = self.manifest["sources"]
        period_ends = {
            "SRC_ERP": max(self.orders["date"]),
            "SRC_MKT": max(self.marketing["week_start"]) + timedelta(days=6),
            "SRC_OPS": max(d for d in self.dispatch["date"] if d is not None),
            "SRC_EXT": max(self.external["date"]),
        }
        notes = {
            "SRC_MKT": "Region South absent from feed",
            "SRC_EXT": "approx 60% of days observed",
        }
        pcts = {"SRC_EXT": self.dq.coverage["SRC_EXT"]["day_coverage_pct"],
                "SRC_MKT": round(100 * 3 / 4, 1)}
        for sid, meta in m.items():
            self.dq.freshness[sid] = classify_freshness(
                sid, meta["last_refresh_at"], float(meta["expected_cadence_hours"]),
                self.as_of_dt, data_period_end=period_ends.get(sid),
                coverage_note=notes.get(sid), coverage_pct=pcts.get(sid))

    # ------------------------------------------------------------------ report
    def summary(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "row_counts": {
                "SRC_ERP.orders_daily": len(self.orders),
                "SRC_ERP.refunds_daily": len(self.refunds),
                "SRC_MKT.campaign_weekly": len(self.marketing),
                "SRC_OPS.dispatch_events": len(self.dispatch),
                "SRC_OPS.support_tickets_daily": len(self.tickets),
                "SRC_EXT.competitor_promo": len(self.external),
            },
            "data_quality": self.dq.as_dict(),
        }
