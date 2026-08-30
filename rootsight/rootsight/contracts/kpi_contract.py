"""KPI Semantic Contract loader, validator and query surface.

The contract is the single source of truth for what a KPI *is*.  Three rules are
enforced in code, not in prose:

  R1  A KPI that is not in the registry cannot be analysed.
  R2  Two KPIs listed as `not_interchangeable_with` each other can never appear
      on the two sides of the same comparison, decomposition or driver series.
      Attempting it raises ContractViolation.  This is what stops "Revenue"
      silently meaning gross in one place and net in another.
  R3  Driver enumeration, thresholds, materiality and access restrictions are
      read from the contract by the engine.  There is no second list anywhere.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import yaml

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kpi_registry.yaml")


class ContractViolation(Exception):
    """Raised when an analysis would breach the semantic contract."""


class ContractLookupError(Exception):
    """Raised when an unregistered KPI or source is requested."""


# --------------------------------------------------------------------- helpers
def _to_date(v: Any) -> date | None:
    if v is None:
        return None
    if isinstance(v, date):
        return v
    return datetime.fromisoformat(str(v)).date()


_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


def safe_formula_eval(formula: str, values: dict[str, float]) -> float:
    """Evaluate a contract formula over named column aggregates.

    Only arithmetic over declared names is permitted; no calls, no attributes,
    no subscripts.  A formula is a contract clause, not a scripting hook.
    """
    tree = ast.parse(formula, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ContractViolation(
                f"formula contains disallowed syntax {type(node).__name__!r}: {formula}")
        if isinstance(node, ast.Name) and node.id not in values:
            raise ContractViolation(
                f"formula references undeclared column {node.id!r}: {formula}")
    return float(eval(compile(tree, "<formula>", "eval"), {"__builtins__": {}}, dict(values)))


# ---------------------------------------------------------------- data classes
@dataclass(frozen=True)
class Driver:
    driver_id: str
    kind: str                       # DECOMPOSITION | CAUSAL_CANDIDATE | NUISANCE
    source: str
    grain: str
    expected_sign: str
    prior_lag_days: tuple[int, int] | None = None

    @property
    def is_causal_candidate(self) -> bool:
        return self.kind == "CAUSAL_CANDIDATE"


@dataclass(frozen=True)
class AccessRule:
    role: str
    max_grain: str
    row_scope: str                  # ALL | OWN_REGION | NONE
    denied_fields: tuple[str, ...]

    @property
    def denied(self) -> bool:
        return self.max_grain == "DENIED" or self.row_scope == "NONE"


@dataclass
class KpiDefinition:
    kpi_id: str
    name: str
    definition: str
    formula: str
    formula_columns: list[str]
    unit: str
    grain: str
    dimensions: list[str]
    time_semantics: str
    calendar: str
    data_sources: list[str]
    refresh_cadence: str
    allowed_aggregations: list[str]
    is_additive: bool
    time_aggregation: str
    owner: str
    version: str
    effective_from: date
    not_interchangeable_with: list[str]
    lineage: dict
    thresholds: dict
    materiality_threshold: dict
    drivers: list[Driver]
    access: dict[str, AccessRule]
    launched_on: date | None = None
    raw: dict = field(default_factory=dict)

    # -------------------------------------------------------------- behaviour
    def causal_candidates(self) -> list[Driver]:
        return [d for d in self.drivers if d.kind == "CAUSAL_CANDIDATE"]

    def decomposition_drivers(self) -> list[Driver]:
        return [d for d in self.drivers if d.kind == "DECOMPOSITION"]

    def nuisance_drivers(self) -> list[Driver]:
        return [d for d in self.drivers if d.kind == "NUISANCE"]

    def grain_dimensions(self) -> list[str]:
        return [p.strip() for p in self.grain.split("x")]

    def is_newly_launched(self, as_of: date, min_days: int = 60) -> bool:
        start = self.launched_on or self.effective_from
        return (as_of - start).days < min_days

    def history_days(self, as_of: date) -> int:
        start = self.launched_on or self.effective_from
        return max(0, (as_of - start).days)

    def access_for(self, role: str) -> AccessRule:
        if role not in self.access:
            # closed by default: an undeclared role gets nothing
            return AccessRule(role=role, max_grain="DENIED", row_scope="NONE",
                              denied_fields=("ALL",))
        return self.access[role]

    def compute(self, column_aggregates: dict[str, float]) -> float:
        missing = [c for c in self.formula_columns if c not in column_aggregates]
        if missing:
            raise ContractViolation(
                f"{self.kpi_id}: cannot compute, missing declared columns {missing}")
        return safe_formula_eval(self.formula, column_aggregates)

    def threshold_breached(self, pct_change: float) -> bool:
        t = self.thresholds
        d = t.get("alert_direction", "both")
        lim = float(t.get("alert_pct_change", 0.0))
        if d == "up":
            return pct_change >= lim
        if d == "down":
            return pct_change <= -lim
        return abs(pct_change) >= lim


@dataclass
class SourceDefinition:
    source_id: str
    name: str
    system: str
    tables: list[str]
    grain: str
    calendar: str
    refresh_cadence: str
    expected_lag_hours: float
    reliability_weight: float
    timestamp_convention: str
    access_classification: str
    known_coverage_gaps: list[str] = field(default_factory=list)


# -------------------------------------------------------------------- registry
class KpiContractRegistry:
    def __init__(self, path: str = REGISTRY_PATH):
        with open(path, "r", encoding="utf-8") as fh:
            raw_text = fh.read()
        self.raw = yaml.safe_load(raw_text)
        self.path = path
        self.contract_hash = hashlib.sha256(raw_text.encode()).hexdigest()[:16]
        self.contract_version = self.raw["contract_version"]
        self.sources: dict[str, SourceDefinition] = {}
        self.kpis: dict[str, KpiDefinition] = {}
        self._load()
        self._validate()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        for sid, s in self.raw["sources"].items():
            self.sources[sid] = SourceDefinition(
                source_id=sid, name=s["name"], system=s["system"], tables=s["tables"],
                grain=s["grain"], calendar=s["calendar"],
                refresh_cadence=s["refresh_cadence"],
                expected_lag_hours=float(s["expected_lag_hours"]),
                reliability_weight=float(s["reliability_weight"]),
                timestamp_convention=s["timestamp_convention"],
                access_classification=s["access_classification"],
                known_coverage_gaps=list(s.get("known_coverage_gaps", [])))

        for kid, k in self.raw["kpis"].items():
            drivers = [
                Driver(driver_id=d["driver_id"], kind=d["kind"], source=d["source"],
                       grain=d["grain"], expected_sign=d["expected_sign"],
                       prior_lag_days=tuple(d["prior_lag_days"]) if d.get("prior_lag_days") else None)
                for d in (k.get("drivers") or [])
            ]
            access = {
                role: AccessRule(role=role, max_grain=a["max_grain"],
                                 row_scope=a["row_scope"],
                                 denied_fields=tuple(a.get("denied_fields") or []))
                for role, a in (k.get("access_restrictions") or {}).items()
            }
            self.kpis[kid] = KpiDefinition(
                kpi_id=kid, name=k["name"], definition=" ".join(k["definition"].split()),
                formula=k["formula"], formula_columns=list(k["formula_columns"]),
                unit=k["unit"], grain=k["grain"], dimensions=list(k["dimensions"]),
                time_semantics=k["time_semantics"], calendar=k["calendar"],
                data_sources=list(k["data_sources"]), refresh_cadence=str(k["refresh_cadence"]),
                allowed_aggregations=list(k["allowed_aggregations"]),
                is_additive=bool(k["is_additive"]),
                time_aggregation=k["time_aggregation"], owner=k["owner"], version=k["version"],
                effective_from=_to_date(k["effective_from"]),
                not_interchangeable_with=list(k.get("not_interchangeable_with") or []),
                lineage=dict(k["lineage"]), thresholds=dict(k["thresholds"]),
                materiality_threshold=dict(k["materiality_threshold"]),
                drivers=drivers, access=access,
                launched_on=_to_date(k.get("launched_on")), raw=k)

    # -------------------------------------------------------------- validate
    def _validate(self) -> None:
        errors: list[str] = []
        for kid, k in self.kpis.items():
            for s in k.data_sources:
                if s not in self.sources:
                    errors.append(f"{kid}: unknown source {s}")
            for other in k.not_interchangeable_with:
                if other not in self.kpis:
                    errors.append(f"{kid}: not_interchangeable_with unknown kpi {other}")
                elif kid not in self.kpis[other].not_interchangeable_with:
                    errors.append(
                        f"non-interchangeability not symmetric: {kid} <-> {other}")
            if k.calendar not in self.raw["calendars"]:
                errors.append(f"{kid}: unknown calendar {k.calendar}")
            if k.time_aggregation not in ("FLOW", "STOCK", "RATIO"):
                errors.append(f"{kid}: time_aggregation must be FLOW|STOCK|RATIO")
            if k.time_aggregation == "RATIO" and k.is_additive:
                errors.append(f"{kid}: a RATIO KPI cannot be additive")
            if k.time_aggregation == "STOCK" and "sum" in k.allowed_aggregations:
                errors.append(f"{kid}: a STOCK KPI must not allow plain sum over time")
            try:
                safe_formula_eval(k.formula, {c: 1.0 for c in k.formula_columns})
            except Exception as exc:                      # noqa: BLE001
                errors.append(f"{kid}: formula not evaluable ({exc})")
        for e in self.raw["kpi_graph"]["edges"]:
            for node in (e["from"], e["to"]):
                known = node in self.kpis or any(
                    node == d.driver_id for k in self.kpis.values() for d in k.drivers)
                if not known:
                    errors.append(f"kpi_graph: edge node {node} is neither a KPI nor a driver")
        if errors:
            raise ContractViolation("registry invalid:\n  - " + "\n  - ".join(errors))

    # ----------------------------------------------------------------- query
    def get(self, kpi_id: str) -> KpiDefinition:
        if kpi_id not in self.kpis:
            raise ContractLookupError(
                f"{kpi_id!r} is not in the KPI semantic contract "
                f"(registered: {sorted(self.kpis)})")
        return self.kpis[kpi_id]

    def source(self, source_id: str) -> SourceDefinition:
        if source_id not in self.sources:
            raise ContractLookupError(f"{source_id!r} is not a registered source")
        return self.sources[source_id]

    # ------------------------------------------------------------------- R2
    def assert_comparable(self, a: str, b: str, context: str = "") -> None:
        """R2 enforcement.  Called by every comparison / decomposition path."""
        ka, kb = self.get(a), self.get(b)
        if b in ka.not_interchangeable_with or a in kb.not_interchangeable_with:
            raise ContractViolation(
                f"KPI definition conflict{' in ' + context if context else ''}: "
                f"{ka.name} ({a}, formula: {ka.formula}) and {kb.name} ({b}, formula: "
                f"{kb.formula}) are declared not interchangeable in the semantic "
                f"contract.  They measure different things and must not be compared, "
                f"summed, or substituted for one another.")

    def assert_series_consistent(self, kpi_id: str, series_kpi_ids: list[str],
                                 context: str = "") -> None:
        for other in series_kpi_ids:
            if other in self.kpis and other != kpi_id:
                self.assert_comparable(kpi_id, other, context=context)

    # --------------------------------------------------------------- graph
    def declared_edges(self) -> list[tuple[str, str]]:
        return [(e["from"], e["to"]) for e in self.raw["kpi_graph"]["edges"]]

    def edge_provenance(self) -> dict[tuple[str, str], dict]:
        return {(e["from"], e["to"]): {k: v for k, v in e.items() if k not in ("from", "to")}
                for e in self.raw["kpi_graph"]["edges"]}

    def unobserved_nodes(self) -> list[dict]:
        return list(self.raw["kpi_graph"].get("unobserved") or [])

    # --------------------------------------------------------------- export
    def lineage_card(self, kpi_id: str) -> dict:
        k = self.get(kpi_id)
        return {
            "kpi_id": kpi_id, "name": k.name, "definition": k.definition,
            "formula": k.formula, "unit": k.unit, "grain": k.grain,
            "time_aggregation": k.time_aggregation, "is_additive": k.is_additive,
            "time_semantics": k.time_semantics, "calendar": k.calendar,
            "owner": k.owner, "contract_version": k.version,
            "effective_from": k.effective_from.isoformat(),
            "sources": [{"source_id": s, "system": self.sources[s].system,
                         "tables": self.sources[s].tables,
                         "cadence": self.sources[s].refresh_cadence}
                        for s in k.data_sources],
            "calculation": k.lineage.get("calculation"),
            "source_tables": k.lineage.get("source_tables", []),
            "transformations": k.lineage.get("transformations", []),
            "not_interchangeable_with": k.not_interchangeable_with,
            "registry_hash": self.contract_hash,
        }

    def to_json(self) -> str:
        return json.dumps({"contract_version": self.contract_version,
                           "registry_hash": self.contract_hash,
                           "kpis": sorted(self.kpis),
                           "sources": sorted(self.sources)}, indent=2)


_REGISTRY: KpiContractRegistry | None = None


def registry() -> KpiContractRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = KpiContractRegistry()
    return _REGISTRY
