"""Data-quality assessment and freshness.

Policy: expose, never hide.  Every defect becomes a first-class object that can
be attached to evidence, surfaced in the UI, and cited by the narrative.  Rows
are quarantined (kept, excluded from computation, counted) rather than dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

IST = timezone(timedelta(hours=5, minutes=30))

FRESHNESS_CURRENT = "CURRENT"
FRESHNESS_LAGGING = "LAGGING"
FRESHNESS_STALE = "STALE"
FRESHNESS_MISSING = "MISSING"


@dataclass
class Defect:
    defect_id: str
    source_id: str
    table: str
    kind: str            # MISSING_VALUE | DUPLICATE_CONFLICT | SIGN_ERROR |
                         # TIMESTAMP_FORMAT | FUTURE_DATED | COVERAGE_GAP | LATE_ARRIVAL
    severity: str        # LOW | MEDIUM | HIGH
    rows_affected: int
    rows_total: int
    action_taken: str    # QUARANTINED | RESOLVED_LATEST_INGEST | NORMALISED | DISCLOSED
    detail: str
    sample_keys: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.rows_affected / self.rows_total if self.rows_total else 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["rate"] = round(self.rate, 5)
        return d


@dataclass
class Freshness:
    source_id: str
    last_refresh_at: str
    expected_cadence_hours: float
    age_hours: float
    status: str
    data_period_end: str | None
    coverage_note: str | None = None
    coverage_pct: float | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["age_hours"] = round(self.age_hours, 2)
        return d


def classify_freshness(source_id: str, last_refresh_at: str,
                       expected_cadence_hours: float, as_of: datetime,
                       data_period_end: date | None = None,
                       coverage_note: str | None = None,
                       coverage_pct: float | None = None) -> Freshness:
    lr = datetime.fromisoformat(last_refresh_at)
    if lr.tzinfo is None:
        lr = lr.replace(tzinfo=IST)
    age = (as_of - lr).total_seconds() / 3600.0
    if age <= expected_cadence_hours * 1.25:
        status = FRESHNESS_CURRENT
    elif age <= expected_cadence_hours * 2.0:
        status = FRESHNESS_LAGGING
    else:
        status = FRESHNESS_STALE
    return Freshness(source_id=source_id, last_refresh_at=lr.isoformat(),
                     expected_cadence_hours=expected_cadence_hours, age_hours=age,
                     status=status,
                     data_period_end=data_period_end.isoformat() if data_period_end else None,
                     coverage_note=coverage_note, coverage_pct=coverage_pct)


def normalise_timestamp(raw: Any) -> tuple[datetime | None, str]:
    """Handle the three conventions in the ops feed.

    Returns (timestamp_in_IST, convention_detected).  A naive timestamp is
    interpreted as IST local because that is the documented upstream behaviour;
    the interpretation is returned so it can be disclosed, not assumed silently.
    """
    if raw is None or raw == "":
        return None, "EMPTY"
    s = str(raw).strip()
    if s.isdigit() and len(s) >= 12:
        return datetime.fromtimestamp(int(s) / 1000.0, tz=IST), "EPOCH_MILLIS"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None, "UNPARSEABLE"
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST), "NAIVE_ASSUMED_IST"
    return dt.astimezone(IST), "ISO_OFFSET"


@dataclass
class DataQualityReport:
    defects: list[Defect] = field(default_factory=list)
    freshness: dict[str, Freshness] = field(default_factory=dict)
    quarantined_rows: dict[str, int] = field(default_factory=dict)
    coverage: dict[str, dict] = field(default_factory=dict)

    def add(self, d: Defect) -> None:
        self.defects.append(d)

    def worst_severity(self) -> str:
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        return max((d.severity for d in self.defects), key=lambda s: order[s], default="LOW")

    def stale_sources(self) -> list[str]:
        return [s for s, f in self.freshness.items()
                if f.status in (FRESHNESS_STALE, FRESHNESS_MISSING)]

    def blocking_gaps(self) -> list[str]:
        out = []
        for s, cov in self.coverage.items():
            for missing in cov.get("missing_dimensions", []):
                out.append(f"{s}: no data for {missing}")
        return out

    def as_dict(self) -> dict:
        return {
            "defects": [d.as_dict() for d in self.defects],
            "freshness": {k: v.as_dict() for k, v in self.freshness.items()},
            "quarantined_rows": dict(self.quarantined_rows),
            "coverage": dict(self.coverage),
            "worst_severity": self.worst_severity(),
            "stale_sources": self.stale_sources(),
            "blocking_gaps": self.blocking_gaps(),
        }
