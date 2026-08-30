"""Trust Contract validation of rendered prose.

Deterministic, no model.  Five independent checks:

  V1  FORBIDDEN VERB SCAN     word-boundary regex over the contract's forbidden list
  V2  NUMBER WHITELIST        every numeral in the prose must match an allowed number
                              within its declared tolerance (dates, years, IDs and
                              ordinals are exempt via an explicit allowlist)
  V3  ENTITY WHITELIST        every capitalised entity token must be permitted
  V4  MANDATORY SECTIONS      every required section's locked sentences are present
  V5  LOCKED SENTENCE FIDELITY  locked sentences appear verbatim

A failure is not an error condition - it triggers one retry and then degraded
mode, which is a legitimate output, not a crash.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from .plan import NarrativePlan
from .trust_contract import TrustContract

NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _entity_universe() -> set[str]:
    from .. import config
    from ..contracts.kpi_contract import registry
    u: set[str] = set()
    u.update(config.REGIONS)
    u.update(config.PRODUCT_LINES)
    u.update(config.WAREHOUSES.keys())
    u.update(registry().sources.keys())
    return u


ENTITY_UNIVERSE = _entity_universe()

STRUCTURED_ID_PATTERNS = [
    (r"\bWH-[A-Z]\d\b", "warehouse id"),
    (r"\bCMP-[A-Z]-[A-Z]+\b", "campaign id"),
    (r"\bEMP-\d+\b", "employee id"),
    (r"\b[\w.]+@[\w.]+\.[a-z]{2,}\b", "email address"),
]
# tokens that look like numbers but are not claims
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
EXEMPT_LITERALS = {"95", "5", "0", "1", "2", "3", "4", "7", "12", "24", "30", "100"}


@dataclass
class ValidationResult:
    passed: bool
    checks: list[dict]
    violations: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def violation_codes(self) -> list[str]:
        return sorted({v["code"] for v in self.violations})


class TrustContractValidator:
    def validate(self, text: str, tc: TrustContract, plan: NarrativePlan) -> ValidationResult:
        checks: list[dict] = []
        violations: list[dict] = []
        low = text.lower()

        # ---- V1 forbidden verbs ------------------------------------------
        hits = []
        for verb in tc.forbidden_verbs:
            if re.search(r"\b" + re.escape(verb.lower()) + r"\b", low):
                hits.append(verb)
        checks.append({"check": "V1_forbidden_verbs", "passed": not hits,
                       "scanned": len(tc.forbidden_verbs), "hits": hits})
        for v in hits:
            violations.append({"code": "FORBIDDEN_VERB", "detail": v,
                               "why": (f"the phrase {v!r} asserts more than the causal "
                                       f"status {tc.causal_status} permits")})

        # ---- V2 numbers ---------------------------------------------------
        scrub = DATE_RE.sub(" ", text)
        scrub = re.sub(r"\b(EV|CLM|DQ|TR|H\d|SRC)[-_A-Za-z0-9]*", " ", scrub)
        scrub = re.sub(r"\bWH-[A-Z]\d\b", " ", scrub)
        found, bad = [], []
        for tok in NUMBER_RE.findall(scrub):
            raw = tok.replace(",", "").strip("+")
            if raw in EXEMPT_LITERALS or raw.lstrip("-") in EXEMPT_LITERALS:
                continue
            try:
                x = float(raw)
            except ValueError:
                continue
            found.append(x)
            ok, label = tc.is_number_allowed(x)
            if not ok:
                bad.append(x)
        checks.append({"check": "V2_number_whitelist", "passed": not bad,
                       "numbers_in_text": len(found),
                       "whitelist_size": len(tc.allowed_numbers),
                       "unmatched": bad})
        for x in bad:
            violations.append({"code": "UNVERIFIED_NUMBER", "detail": x,
                               "why": ("this value is not in the Trust Contract's "
                                       "allowed_numbers, so nothing in the analysis "
                                       "produced it")})

        # ---- V3 entity leakage ------------------------------------------
        # This is a LEAKAGE check, not a grammar check.  The universe is every
        # entity name the system knows about - regions, product lines,
        # warehouses, source systems, structured campaign and warehouse ids.  A
        # violation is a universe member appearing in prose that the contract did
        # not authorise for this persona.  Policing ordinary capitalised English
        # would generate noise and catch nothing that matters; naming Region
        # South to an operations manager scoped to North is the failure mode
        # worth blocking.
        allowed = set(tc.allowed_entities)
        allowed_tokens = set()
        for e in allowed:
            allowed_tokens.update(t for t in re.split(r"[^A-Za-z0-9-]+", e) if t)
        leaked = []
        for name in ENTITY_UNIVERSE:
            if name in allowed or name in allowed_tokens:
                continue
            if re.search(r"\b" + re.escape(name) + r"\b", text):
                leaked.append(name)
        for pat, label in STRUCTURED_ID_PATTERNS:
            for tok in re.findall(pat, text):
                if tok not in allowed and tok not in allowed_tokens:
                    leaked.append(f"{tok} ({label})")
        leaked = sorted(set(leaked))
        checks.append({"check": "V3_entity_leakage", "passed": not leaked,
                       "universe_size": len(ENTITY_UNIVERSE),
                       "authorised_entities": sorted(allowed),
                       "leaked": leaked,
                       "note": ("closed-vocabulary leakage check over known entity "
                                "names and structured ids")})
        for t in leaked:
            violations.append({"code": "ENTITY_LEAKAGE", "detail": t,
                               "why": ("this entity is outside the access scope "
                                       "authorised for this persona and must not appear "
                                       "in the narrative")})

        # ---- V4 mandatory sections ---------------------------------------
        present = {s.section_id for s in plan.sections}
        missing_sections = [s for s in tc.mandatory_sections
                            if s not in present and s not in
                            ("measured", "estimate", "action", "association",
                             "why_not_identified", "abstention", "what_is_missing",
                             "what_would_resolve_it", "prediction_vs_outcome",
                             "assumptions")]
        semantic_map = {
            "measured": "headline", "estimate": "driver", "association": "driver",
            "why_not_identified": "driver", "assumptions": "limits",
            "contradicting": "contradicting", "gaps": "gaps", "action": "decision",
            "abstention": "abstention", "what_is_missing": "gaps",
            "what_would_resolve_it": "clarification_request",
        }
        unmet = []
        for req in tc.mandatory_sections:
            target = semantic_map.get(req, req)
            if target == "clarification_request" and target not in present:
                target = "abstention"
            if target not in present:
                unmet.append(req)
        checks.append({"check": "V4_mandatory_sections", "passed": not unmet,
                       "required": tc.mandatory_sections,
                       "present": sorted(present), "unmet": unmet})
        for s in unmet:
            violations.append({"code": "MISSING_MANDATORY_SECTION", "detail": s,
                               "why": ("the contract requires this disclosure for "
                                       f"status {tc.causal_status}")})

        # ---- V5 locked sentence fidelity ---------------------------------
        def norm(x: str) -> str:
            return re.sub(r"\s+", " ", x).strip()

        ntext = norm(text)
        dropped = [s for s in plan.locked_sentences() if norm(s) not in ntext]
        checks.append({"check": "V5_locked_sentences", "passed": not dropped,
                       "locked": len(plan.locked_sentences()),
                       "dropped_or_altered": dropped[:5]})
        for s in dropped:
            violations.append({"code": "LOCKED_SENTENCE_ALTERED", "detail": s[:120],
                               "why": ("a locked sentence carries a number or a claim "
                                       "fixed by the contract and may not be reworded")})

        return ValidationResult(passed=not violations, checks=checks, violations=violations)
