"""Business calendar reconciliation.

Three sources, three calendars:
  * ERP        -> Indian fiscal calendar (FY starts 1 April), fiscal weeks Sun-start
  * Marketing  -> ISO weeks (Mon-start), campaign weeks labelled by week-start date
  * Operations -> plain calendar days, local time (Asia/Kolkata)

Nothing downstream is allowed to assume these agree.  Every fact carries the
canonical `date` plus the calendar attributes needed to explain it.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

FISCAL_YEAR_START_MONTH = 4

HOLIDAYS = {
    date(2026, 1, 1): "New Year",
    date(2026, 1, 26): "Republic Day",
    date(2026, 3, 4): "Holi",
    date(2026, 4, 14): "Baisakhi",
    date(2026, 5, 1): "May Day",
    date(2026, 8, 15): "Independence Day",
    date(2026, 8, 26): "Onam",
}

# Promotional periods are business-defined, overlap holidays, and do NOT align
# with fiscal boundaries.  This is the point.
PROMO_WINDOWS = [
    ("PROMO_REPUBLIC", date(2026, 1, 20), date(2026, 1, 27)),
    ("PROMO_SUMMER", date(2026, 5, 8), date(2026, 5, 20)),
    ("PROMO_FREEDOM", date(2026, 8, 11), date(2026, 8, 17)),
]

# Windows that must never be compared against non-promo windows without
# adjustment.  The KPI contract references these ids.
BLACKOUT_FOR_BASELINE = {"PROMO_FREEDOM"}


def fiscal_year(d: date) -> str:
    y = d.year if d.month >= FISCAL_YEAR_START_MONTH else d.year - 1
    return f"FY{str(y + 1)[-2:]}"


def fiscal_quarter(d: date) -> str:
    m = (d.month - FISCAL_YEAR_START_MONTH) % 12
    return f"{fiscal_year(d)}-Q{m // 3 + 1}"


def fiscal_period(d: date) -> str:
    m = (d.month - FISCAL_YEAR_START_MONTH) % 12 + 1
    return f"{fiscal_year(d)}-P{m:02d}"


def iso_week_start(d: date) -> date:
    """Monday-start ISO week (marketing convention)."""
    return d - timedelta(days=d.weekday())


def fiscal_week_start(d: date) -> date:
    """Sunday-start fiscal week (ERP convention)."""
    return d - timedelta(days=(d.weekday() + 1) % 7)


def is_holiday(d: date) -> bool:
    return d in HOLIDAYS


def promo_window(d: date) -> str | None:
    for pid, s, e in PROMO_WINDOWS:
        if s <= d <= e:
            return pid
    return None


@lru_cache(maxsize=4096)
def attributes(d: date) -> dict:
    pw = promo_window(d)
    return {
        "date": d.isoformat(),
        "dow": d.weekday(),
        "is_weekend": d.weekday() >= 5,
        "fiscal_year": fiscal_year(d),
        "fiscal_quarter": fiscal_quarter(d),
        "fiscal_period": fiscal_period(d),
        "fiscal_week_start": fiscal_week_start(d).isoformat(),
        "iso_week_start": iso_week_start(d).isoformat(),
        "iso_week": d.isocalendar()[1],
        "is_holiday": is_holiday(d),
        "holiday_name": HOLIDAYS.get(d),
        "promo_window": pw,
        "baseline_eligible": pw not in BLACKOUT_FOR_BASELINE,
    }


def calendar_conflicts(d: date) -> list[str]:
    """Explicit reporting of where the three calendars disagree for a date."""
    out = []
    if fiscal_week_start(d) != iso_week_start(d):
        out.append(
            f"week_start_mismatch: erp_fiscal={fiscal_week_start(d).isoformat()} "
            f"marketing_iso={iso_week_start(d).isoformat()}")
    if is_holiday(d) and not promo_window(d):
        out.append("holiday_without_promo: demand shape differs from promo days")
    if promo_window(d) in BLACKOUT_FOR_BASELINE:
        out.append(f"promo_blackout: {promo_window(d)} is excluded from baselines")
    return out


def window_dates(start: date, end: date) -> list[date]:
    n = (end - start).days
    return [start + timedelta(days=i) for i in range(n + 1)]


def baseline_eligible_dates(start: date, end: date) -> list[date]:
    return [d for d in window_dates(start, end) if attributes(d)["baseline_eligible"]]
