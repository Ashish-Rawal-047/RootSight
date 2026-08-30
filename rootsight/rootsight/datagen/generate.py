"""Synthetic multi-source generator.

This is an explicit structural causal model, not a random dataset.  Because the
SCM is known, the evaluation harness has real ground truth: for the North
fulfilment disruption we compute the exact counterfactual (same noise draws,
disruption multiplier removed) and store the true ATT.

Deliberately built in:
  * three grains          day x region x line | iso_week x campaign x region | event-level
  * three cadences        daily 02:00 batch | weekly Monday | 15-minute micro-batch
  * three calendars       fiscal (Sun weeks) | ISO (Mon weeks) | calendar day
  * data-quality defects  missing values, delayed arrival, mixed timestamp formats,
                          duplicate/conflicting records, whole-region coverage gap,
                          partially observed external feed, sign errors
  * a multi-factor movement whose true drivers differ in size AND in whether a
    causal design exists for them at all
  * a newly launched product line and a newly launched KPI (sparse history)
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone

import numpy as np

from .. import config
from ..bizcalendar import attributes, iso_week_start

IST = timezone(timedelta(hours=5, minutes=30))

# base daily units by region x line (pre-disruption level)
BASE_UNITS = {
    "North": {"Apparel": 520, "Electronics": 300, "Home": 260, "SmartHome": 0},
    "South": {"Apparel": 430, "Electronics": 250, "Home": 235, "SmartHome": 0},
    "East":  {"Apparel": 300, "Electronics": 170, "Home": 160, "SmartHome": 0},
    "West":  {"Apparel": 390, "Electronics": 235, "Home": 205, "SmartHome": 0},
}
LIST_PRICE = {"Apparel": 1450.0, "Electronics": 8900.0, "Home": 2600.0, "SmartHome": 5400.0}
BASE_DISCOUNT = {"Apparel": 0.11, "Electronics": 0.07, "Home": 0.09, "SmartHome": 0.15}
BASE_REFUND = {"Apparel": 0.055, "Electronics": 0.030, "Home": 0.040, "SmartHome": 0.070}
REGION_LEVEL_TREND = 0.00018          # identical slope across regions -> parallel trends
TICKET_CATEGORIES = ["late_delivery", "damaged_item", "billing", "product_query"]


# ------------------------------------------------------------------ components
def seasonality(d: date) -> float:
    a = attributes(d)
    s = 1.0
    s *= 1.16 if a["is_weekend"] else 0.98
    # monsoon softness Jun-Aug, annual shape
    doy = d.timetuple().tm_yday
    s *= 1.0 + 0.045 * np.sin(2 * np.pi * (doy - 40) / 365.0)
    if a["is_holiday"]:
        s *= 1.22
    if a["promo_window"]:
        s *= 1.18
    return float(s)


def marketing_weekly_spend(wk: date, region: str, gt: config.GroundTruth) -> float:
    base = {"North": 2_650_000.0, "South": 2_050_000.0,
            "East": 1_320_000.0, "West": 1_780_000.0}[region]
    a = attributes(wk)
    m = 1.0 + (0.10 if a["promo_window"] else 0.0)
    if wk >= gt.marketing_cut_week_start:
        m *= (1.0 - gt.marketing_cut_fraction)
    return base * m


def _in(d: date, s: date, e: date) -> bool:
    return s <= d <= e


# ------------------------------------------------------------------- generator
class ScenarioGenerator:
    def __init__(self, gt: config.GroundTruth = config.GT):
        self.gt = gt
        self.rng = np.random.default_rng(gt.seed)
        self.dates = [config.HISTORY_START + timedelta(days=i)
                      for i in range((config.HISTORY_END - config.HISTORY_START).days + 1)]
        self.truth: dict = {"scm": gt.as_dict(), "counterfactual_units": {},
                            "counterfactual_revenue": {}}

    # ------------------------------------------------------- effect multipliers
    def dispatch_multiplier(self, d: date, region: str) -> float:
        """Fulfilment disruption effect on demand, lagged."""
        gt = self.gt
        if region != gt.disruption_region:
            return 1.0
        s = gt.disruption_start + timedelta(days=gt.disruption_lag_days)
        e = gt.disruption_end + timedelta(days=gt.disruption_lag_days)
        if not _in(d, s, e):
            return 1.0
        # ramp in over 3 days, ramp out over 3 days (a real disruption is not a step)
        ramp_in = min(1.0, (d - s).days / 3.0 + 0.34)
        ramp_out = min(1.0, (e - d).days / 3.0 + 0.34)
        intensity = min(ramp_in, ramp_out)
        return 1.0 - (1.0 - gt.disruption_units_multiplier) * intensity

    def competitor_multiplier(self, d: date, line: str) -> float:
        gt = self.gt
        if line != gt.competitor_line or not _in(d, gt.competitor_start, gt.competitor_end):
            return 1.0
        return gt.competitor_units_multiplier

    def marketing_multiplier(self, d: date) -> float:
        gt = self.gt
        onset = gt.marketing_cut_week_start + timedelta(days=5)
        return gt.marketing_units_multiplier if d >= onset else 1.0

    def mix_multiplier(self, d: date, line: str) -> float:
        gt = self.gt
        if d < gt.mix_drift_start:
            return 1.0
        if line == "Electronics":
            return 1.0 - gt.mix_drift
        if line == "Home":
            return 1.0 + gt.mix_drift * 0.85
        return 1.0

    def list_price(self, d: date, line: str) -> float:
        p = LIST_PRICE[line]
        if line == self.gt.price_line and d >= self.gt.price_start:
            p *= (1.0 + self.gt.price_increase)
        return p

    def price_elasticity_multiplier(self, d: date, line: str) -> float:
        """Own-price elasticity: the price rise itself suppresses volume a little."""
        if line == self.gt.price_line and d >= self.gt.price_start:
            return 1.0 - 0.9 * self.gt.price_increase      # elasticity ~ -0.9
        return 1.0

    def south_launch_trend(self, d: date, region: str) -> float:
        """South diverges after the SmartHome launch: breaks parallel trends there."""
        if region != "South" or d < config.NEW_LINE_LAUNCH:
            return 1.0
        return 1.0 + 0.0015 * (d - config.NEW_LINE_LAUNCH).days

    def dispatch_on_time_rate(self, d: date, warehouse: str) -> float:
        region = config.WAREHOUSES[warehouse]
        base = {"WH-N1": 0.945, "WH-N2": 0.932, "WH-S1": 0.938,
                "WH-E1": 0.951, "WH-W1": 0.944}[warehouse]
        a = attributes(d)
        r = base - (0.018 if a["is_weekend"] else 0.0) - (0.030 if a["promo_window"] else 0.0)
        if region == self.gt.disruption_region and _in(d, self.gt.disruption_start,
                                                       self.gt.disruption_end):
            ramp = min(1.0, (d - self.gt.disruption_start).days / 2.0 + 0.4)
            ramp = min(ramp, (self.gt.disruption_end - d).days / 3.0 + 0.4)
            r -= 0.335 * ramp
        return float(np.clip(r + self.rng.normal(0, 0.011), 0.05, 0.995))

    # --------------------------------------------------------------- ERP path
    def build_erp(self) -> tuple[list[dict], list[dict], list[dict]]:
        orders, refunds = [], []
        rid = 0
        for d in self.dates:
            seas = seasonality(d)
            for region in config.REGIONS:
                for line in config.PRODUCT_LINES:
                    if line == config.NEW_LINE:
                        if region != config.NEW_LINE_REGION or d < config.NEW_LINE_LAUNCH:
                            continue
                        base = 18 + 1.6 * (d - config.NEW_LINE_LAUNCH).days
                    else:
                        base = BASE_UNITS[region][line]
                    trend = 1.0 + REGION_LEVEL_TREND * (d - config.HISTORY_START).days
                    noise = float(self.rng.normal(1.0, 0.038))

                    counterfactual_mult = (
                        seas * trend * noise
                        * self.competitor_multiplier(d, line)
                        * self.marketing_multiplier(d)
                        * self.mix_multiplier(d, line)
                        * self.price_elasticity_multiplier(d, line)
                        * self.south_launch_trend(d, region))
                    disruption = self.dispatch_multiplier(d, region)
                    units_actual = base * counterfactual_mult * disruption
                    units_cf = base * counterfactual_mult          # exact counterfactual

                    units = max(0.0, units_actual)
                    lp = self.list_price(d, line)
                    disc_rate = BASE_DISCOUNT[line] + (
                        0.05 if attributes(d)["promo_window"] else 0.0)
                    # defensive discounting while the competitor promo is live
                    if line == self.gt.competitor_line and _in(
                            d, self.gt.competitor_start, self.gt.competitor_end):
                        disc_rate += 0.015
                    ref_rate = BASE_REFUND[line] + (
                        0.012 if (region == self.gt.disruption_region and
                                  _in(d, self.gt.disruption_start,
                                      self.gt.disruption_end)) else 0.0)

                    gross = units * lp
                    discounts = gross * disc_rate
                    refund_amt = gross * ref_rate
                    units_returned = units * ref_rate * 0.72

                    if d >= date(2026, 8, 5) and line != config.NEW_LINE:
                        sub_rev = {"North": 41000.0, "South": 33000.0,
                                   "East": 19000.0, "West": 27000.0}[region] / 3.0
                        sub_rev *= (1.0 + 0.052 * (d - date(2026, 8, 5)).days)
                        sub_rev *= float(self.rng.normal(1.0, 0.10))
                    else:
                        sub_rev = 0.0

                    rid += 1
                    ingested = datetime.combine(d + timedelta(days=1),
                                                datetime.min.time(), IST) + timedelta(hours=2)
                    # DQ: East arrives 3 days late for the final 3 days
                    if region == "East" and d >= config.HISTORY_END - timedelta(days=2):
                        ingested += timedelta(days=3)

                    orders.append({
                        "source_row_id": f"ERP-O-{rid:06d}",
                        "date": d.isoformat(), "region": region, "product_line": line,
                        "units_shipped": round(units, 2),
                        "units_returned": round(units_returned, 2),
                        "gross_sales": round(gross, 2),
                        "discounts": round(discounts, 2),
                        "list_price": round(lp, 2),
                        "gross_margin_pct": round(
                            float(np.clip(0.34 + self.rng.normal(0, 0.02), 0.1, 0.6)), 4),
                        "monthly_subscription_revenue": round(sub_rev, 2),
                        "deferred_delta": round(gross * 0.011, 2),
                        "shipped_orders": int(max(1, round(units / 1.85))),
                        "ingested_at": ingested.isoformat(),
                    })
                    refunds.append({
                        "source_row_id": f"ERP-R-{rid:06d}",
                        "date": d.isoformat(), "region": region, "product_line": line,
                        "refunds": round(refund_amt, 2),
                        "refund_liability_inr": round(refund_amt * 1.14, 2),
                        "ingested_at": ingested.isoformat(),
                    })
                    key = f"{d.isoformat()}|{region}|{line}"
                    self.truth["counterfactual_units"][key] = round(units_cf, 4)
                    self.truth["counterfactual_revenue"][key] = round(
                        units_cf * lp * (1 - disc_rate - ref_rate), 2)

        prices = []
        for line in config.PRODUCT_LINES:
            prices.append({"effective_from": config.HISTORY_START.isoformat(),
                           "product_line": line, "list_price": LIST_PRICE[line],
                           "discount_policy_note": f"standard {line} ladder, tier-2 approval"})
            if line == self.gt.price_line:
                prices.append({"effective_from": self.gt.price_start.isoformat(),
                               "product_line": line,
                               "list_price": round(LIST_PRICE[line] * (1 + self.gt.price_increase), 2),
                               "discount_policy_note": "FY27 list correction, CFO approved"})
        return orders, refunds, prices

    # ---------------------------------------------------------- marketing path
    def build_marketing(self) -> list[dict]:
        rows = []
        wk = iso_week_start(config.HISTORY_START)
        campaigns = {"North": ["CMP-N-BRAND", "CMP-N-PERF"],
                     "South": ["CMP-S-BRAND", "CMP-S-PERF", "CMP-S-LAUNCH"],
                     "East": ["CMP-E-PERF"], "West": ["CMP-W-PERF", "CMP-W-BRAND"]}
        # feed is three weeks behind: deliberately STALE for the demo
        last_week = iso_week_start(config.HISTORY_END) - timedelta(days=21)
        while wk <= last_week:
            for region in config.REGIONS:
                # DQ: Region South is not onboarded to CampaignHub -> no rows at all
                if region == "South":
                    continue
                total = marketing_weekly_spend(wk, region, self.gt)
                cs = campaigns[region]
                shares = self.rng.dirichlet(np.ones(len(cs)) * 6)
                for c, sh in zip(cs, shares):
                    rows.append({
                        "week_start": wk.isoformat(),         # Monday: ISO convention
                        "campaign_id": c, "region": region,
                        "channel": "search" if "PERF" in c else "display",
                        "spend_inr": round(total * float(sh), 2),
                        "impressions": int(total * float(sh) / 0.42),
                        "ingested_at": (datetime.combine(wk + timedelta(days=7),
                                                         datetime.min.time(), IST)
                                        + timedelta(hours=9)).isoformat(),
                    })
            wk = wk + timedelta(days=7)
        return rows

    # ---------------------------------------------------------------- ops path
    def build_ops(self) -> tuple[list[dict], list[dict]]:
        events, tickets = [], []
        eid = 0
        for d in self.dates:
            for wh, region in config.WAREHOUSES.items():
                rate = self.dispatch_on_time_rate(d, wh)
                n_ev = int(24 * (1.15 if attributes(d)["is_weekend"] else 1.0))
                for h in range(n_ev):
                    eid += 1
                    met = self.rng.random() < rate
                    promised = 24.0
                    actual = promised * (float(self.rng.uniform(0.35, 0.98)) if met
                                         else float(self.rng.uniform(1.05, 3.4)))
                    ts = datetime.combine(d, datetime.min.time(), IST) + timedelta(
                        hours=int(h % 24), minutes=int(self.rng.integers(0, 60)))
                    # DQ: three different timestamp conventions in one feed
                    r = self.rng.random()
                    if r < 0.60:
                        ts_raw = ts.isoformat()
                    elif r < 0.85:
                        ts_raw = ts.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        ts_raw = str(int(ts.timestamp() * 1000))
                    # DQ: 4.5% missing durations
                    missing = self.rng.random() < 0.045
                    events.append({
                        "event_id": f"OPS-D-{eid:07d}", "ts_raw": ts_raw,
                        "warehouse": wh, "region": region,
                        "promised_hours": promised,
                        "actual_hours": "" if missing else round(actual, 2),
                        "employee_id": f"EMP-{int(self.rng.integers(1000, 1400))}",
                        "shift_id": f"SH-{1 + int(self.rng.integers(0, 3))}",
                        "staff_productivity_score": round(
                            float(np.clip(self.rng.normal(0.78, 0.09), 0.2, 1.0)), 3),
                    })

            for region in config.REGIONS:
                whs = [w for w, r in config.WAREHOUSES.items() if r == region]
                otd = float(np.mean([self.dispatch_on_time_rate(d, w) for w in whs]))
                shipped = sum(BASE_UNITS[region].values()) / 1.85 * seasonality(d)
                for cat in TICKET_CATEGORIES:
                    if cat == "late_delivery":
                        base_rate = 6.2 + 46.0 * max(0.0, 0.94 - otd)
                    else:
                        base_rate = {"damaged_item": 3.1, "billing": 2.4,
                                     "product_query": 4.8}[cat]
                    n = max(0, int(self.rng.poisson(base_rate * shipped / 1000.0 * 1.0)))
                    # DQ: 1.5% of ticket rows are future-dated by a day (tz bug upstream)
                    tdate = d + timedelta(days=1) if self.rng.random() < 0.015 else d
                    tickets.append({
                        "date": tdate.isoformat(), "region": region,
                        "ticket_category": cat,
                        "complaint_tickets": n if cat != "product_query" else 0,
                        "total_tickets": n,
                        "shipped_orders_hint": int(shipped),
                        "customer_email": f"cust{int(self.rng.integers(1, 9999))}@example.com",
                        "ticket_body": ("Order dispatched late from the North hub; "
                                        "no update for three days."
                                        if cat == "late_delivery"
                                        else f"{cat.replace('_', ' ')} issue reported"),
                        "ingested_at": (datetime.combine(d, datetime.min.time(), IST)
                                        + timedelta(hours=23, minutes=45)).isoformat(),
                    })
        return events, tickets

    # ----------------------------------------------------------- external path
    def build_external(self) -> list[dict]:
        rows = []
        for d in self.dates:
            for line in config.PRODUCT_LINES:
                # DQ: only ~60% of days observed, no intensity field
                if self.rng.random() > 0.60:
                    continue
                active = (line == self.gt.competitor_line and
                          _in(d, self.gt.competitor_start, self.gt.competitor_end))
                rows.append({
                    "date": d.isoformat(), "product_line": line,
                    "promo_active": int(active), "observed": 1,
                    "source_note": "MarketWatch scrape, unverified, no discount depth",
                })
        return rows

    # -------------------------------------------------------- DQ corruption
    def corrupt(self, orders: list[dict]) -> tuple[list[dict], dict]:
        rng = self.rng
        stats = {"missing_discounts": 0, "duplicate_conflicts": 0, "negative_units": 0}
        for row in orders:
            if rng.random() < 0.018:
                row["discounts"] = ""                      # missing value
                stats["missing_discounts"] += 1
        # duplicate + conflicting records for the same natural key
        idx = rng.choice(len(orders), size=12, replace=False)
        dupes = []
        for i in idx:
            src = dict(orders[int(i)])
            src["source_row_id"] = src["source_row_id"] + "-DUP"
            src["gross_sales"] = round(float(src["gross_sales"]) * 1.09, 2)
            src["ingested_at"] = (datetime.fromisoformat(src["ingested_at"])
                                  + timedelta(hours=6)).isoformat()
            dupes.append(src)
            stats["duplicate_conflicts"] += 1
        orders.extend(dupes)
        # sign errors
        idx2 = rng.choice(len(orders), size=3, replace=False)
        for i in idx2:
            orders[int(i)]["units_shipped"] = -abs(float(orders[int(i)]["units_shipped"]))
            stats["negative_units"] += 1
        return orders, stats

    # ------------------------------------------------------------------- write
    def run(self, out_dir: str | None = None) -> dict:
        out_dir = out_dir or config.RAW_DIR
        os.makedirs(out_dir, exist_ok=True)
        orders, refunds, prices = self.build_erp()
        orders, dq_stats = self.corrupt(orders)
        mkt = self.build_marketing()
        events, tickets = self.build_ops()
        ext = self.build_external()

        def write(name: str, rows: list[dict]) -> None:
            path = os.path.join(out_dir, name)
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

        write("erp_orders_daily.csv", orders)
        write("erp_refunds_daily.csv", refunds)
        write("erp_price_list.csv", prices)
        write("mkt_campaign_weekly.csv", mkt)
        write("ops_dispatch_events.csv", events)
        write("ops_support_tickets_daily.csv", tickets)
        write("ext_competitor_promo.csv", ext)

        now = datetime.combine(config.AS_OF, datetime.min.time(), IST) + timedelta(hours=10, minutes=5)
        manifest = {
            "generated_at": now.isoformat(),
            "as_of": config.AS_OF.isoformat(),
            "sources": {
                "SRC_ERP": {"last_refresh_at": (now.replace(hour=2, minute=0)).isoformat(),
                            "expected_cadence_hours": 24, "row_counts":
                                {"orders_daily": len(orders), "refunds_daily": len(refunds),
                                 "price_list": len(prices)}},
                "SRC_MKT": {"last_refresh_at": max(r["ingested_at"] for r in mkt),
                            "expected_cadence_hours": 168,
                            "row_counts": {"campaign_weekly": len(mkt)}},
                "SRC_OPS": {"last_refresh_at": (now - timedelta(minutes=12)).isoformat(),
                            "expected_cadence_hours": 0.25,
                            "row_counts": {"dispatch_events": len(events),
                                           "support_tickets_daily": len(tickets)}},
                "SRC_EXT": {"last_refresh_at": (now - timedelta(days=2, hours=6)).isoformat(),
                            "expected_cadence_hours": 48,
                            "row_counts": {"competitor_promo": len(ext)}},
            },
            "injected_data_quality_defects": {
                **dq_stats,
                "mixed_timestamp_formats_in_ops": "iso+05:30 / naive local / epoch millis",
                "missing_dispatch_durations_pct": 4.5,
                "marketing_region_coverage_gap": "South (never onboarded)",
                "marketing_feed_staleness_days": (
                    config.AS_OF - (iso_week_start(config.HISTORY_END)
                                    - timedelta(days=21) + timedelta(days=7))).days,
                "external_feed_day_coverage_pct": 60,
                "east_region_arrival_delay_days": 3,
                "future_dated_ticket_rows_pct": 1.5,
            },
        }
        with open(os.path.join(out_dir, "_manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        with open(os.path.join(out_dir, "_ground_truth.json"), "w", encoding="utf-8") as fh:
            json.dump(self.truth, fh)
        return manifest


def main() -> None:
    m = ScenarioGenerator().run()
    print(json.dumps({"as_of": m["as_of"],
                      "rows": {k: v["row_counts"] for k, v in m["sources"].items()},
                      "defects": m["injected_data_quality_defects"]}, indent=2))


if __name__ == "__main__":
    main()
