"""Prose rendering, validation and degraded mode.

Three render modes, all producing a validated narrative:

  DETERMINISTIC_TEMPLATE  no model.  Locked sentences plus section headings.
                          This is the default when no API key is configured and
                          it is a complete, shippable output.
  LLM_RENDERED            a single model call renders the plan.  The output is
                          validated against the Trust Contract; one retry with
                          the violations fed back; then degraded mode.
  DEGRADED_EVIDENCE_TABLE validation failed twice.  The user gets the structured
                          evidence and the locked sentences with no prose. This
                          is a legitimate answer, not an error page.

Token accounting: in deterministic mode the "prompt" that WOULD have been sent is
measured for size and recorded as an ESTIMATED token count so the cost panel
still has an honest number, explicitly labelled as an estimate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

from .. import config
from ..security.policy import AccessDecision
from ..telemetry import Telemetry
from .llm import LlmBudgetExceeded, LlmClient, LlmUnavailable
from .plan import NarrativePlan
from .trust_contract import TrustContract
from .validator import TrustContractValidator, ValidationResult

MODE_TEMPLATE = "DETERMINISTIC_TEMPLATE"
MODE_LLM = "LLM_RENDERED"
MODE_DEGRADED = "DEGRADED_EVIDENCE_TABLE"


@dataclass
class NarrativeResult:
    text: str
    render_mode: str
    validation: dict
    attempts: int
    degraded: bool
    llm_calls: int
    contract_hash: str
    plan_version: str
    word_count: int
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _template_prose(plan: NarrativePlan) -> str:
    out = []
    for s in plan.sections:
        out.append(s.heading.upper())
        for f in s.facts:
            out.append(f"  {f.text}")
        out.append("")
    return "\n".join(out).strip()


def _degraded_output(plan: NarrativePlan, tc: TrustContract,
                     violations: list[dict]) -> str:
    lines = [
        "NARRATIVE GENERATION DID NOT PASS TRUST CONTRACT VALIDATION.",
        "The analysis itself is unaffected: it is deterministic and complete. Only the "
        "prose layer failed. The validated facts are below.",
        "",
        f"Status: {tc.display_label}   (causal status {tc.causal_status})",
        f"Validation failures: " + ", ".join(sorted({v['code'] for v in violations})),
        "",
    ]
    for s in plan.sections:
        lines.append(s.heading.upper())
        for f in s.facts:
            lines.append(f"  - {f.text}")
        lines.append("")
    return "\n".join(lines).strip()


class NarrativeCompiler:
    def __init__(self, telemetry: Telemetry):
        self.telemetry = telemetry
        self.validator = TrustContractValidator()

    def compile(self, tc: TrustContract, plan: NarrativePlan,
                decision: AccessDecision, *, prefer_llm: bool = True) -> NarrativeResult:
        payload = {
            "instruction": "Render the narrative_plan below as prose for the persona.",
            "trust_contract": {
                "causal_status": tc.causal_status,
                "display_label": tc.display_label,
                "persona": tc.persona,
                "persona_profile": tc.persona_profile,
                "decision_frame": tc.decision_frame,
                "permitted_verbs": tc.allowed_verbs,
                "forbidden_verbs": tc.forbidden_verbs,
                "allowed_entities": tc.allowed_entities,
                "quantification_rule": tc.quantification_rule,
                "mandatory_sections": tc.mandatory_sections,
                "max_words": tc.max_words,
                "llm_may_decide": tc.llm_may_decide,
                "llm_may_not_decide": tc.llm_may_not_decide,
            },
            "narrative_plan": plan.as_dict(),
        }
        notes: list[str] = []

        client = LlmClient(self.telemetry)
        if prefer_llm and client.available:
            result = self._llm_path(client, payload, tc, plan, decision, notes)
            if result is not None:
                return result

        # ---------------------------- deterministic path --------------------
        with self.telemetry.span("narrative_render_template", layer="NON_LLM",
                                 stage=True, mode=MODE_TEMPLATE):
            text = _template_prose(plan)
        v = self.validator.validate(text, tc, plan)
        prompt_str = json.dumps(payload, default=str)
        self.telemetry.record_model_call(
            purpose="narrative_render", provider="local",
            model="deterministic-template-v5",
            prompt_tokens=Telemetry.estimate_tokens(prompt_str),
            completion_tokens=Telemetry.estimate_tokens(text),
            latency_ms=0.0, token_source="ESTIMATED", stop_reason="template_complete")
        if not client.available:
            notes.append("ANTHROPIC_API_KEY not set: rendered deterministically. Token "
                         "counts for this mode are ESTIMATED (chars/4) and cost is zero.")
        if not v.passed:
            notes.append("the deterministic renderer itself failed validation, which "
                         "indicates a contract or plan defect rather than a model error")
            return NarrativeResult(
                text=_degraded_output(plan, tc, v.violations), render_mode=MODE_DEGRADED,
                validation=v.as_dict(), attempts=1, degraded=True, llm_calls=0,
                contract_hash=tc.contract_hash, plan_version=plan.plan_version,
                word_count=len(text.split()), notes=notes)
        return NarrativeResult(
            text=text, render_mode=MODE_TEMPLATE, validation=v.as_dict(), attempts=1,
            degraded=False, llm_calls=0, contract_hash=tc.contract_hash,
            plan_version=plan.plan_version, word_count=len(text.split()), notes=notes)

    # ------------------------------------------------------------------ LLM
    def _llm_path(self, client: LlmClient, payload: dict, tc: TrustContract,
                  plan: NarrativePlan, decision: AccessDecision,
                  notes: list[str]) -> NarrativeResult | None:
        last_v: ValidationResult | None = None
        for attempt in range(config.MAX_LLM_CALLS_PER_ANALYSIS):
            body = dict(payload)
            if last_v is not None:
                body["previous_attempt_violations"] = last_v.violations
                body["fix_instruction"] = (
                    "Your previous output violated the Trust Contract as listed. "
                    "Reproduce every locked sentence verbatim and add no new numbers.")
            try:
                with self.telemetry.span("narrative_render_llm", layer="LLM",
                                         stage=True, attempt=attempt):
                    reply = client.render(body, decision, retry_index=attempt)
            except (LlmUnavailable, LlmBudgetExceeded) as exc:
                notes.append(f"model path abandoned: {exc}")
                return None
            except Exception as exc:                              # noqa: BLE001
                notes.append(f"model call failed ({type(exc).__name__}: {exc}); "
                             "falling back to the deterministic renderer")
                return None
            v = self.validator.validate(reply.text, tc, plan)
            last_v = v
            if v.passed:
                return NarrativeResult(
                    text=reply.text, render_mode=MODE_LLM, validation=v.as_dict(),
                    attempts=attempt + 1, degraded=False, llm_calls=attempt + 1,
                    contract_hash=tc.contract_hash, plan_version=plan.plan_version,
                    word_count=len(reply.text.split()), notes=notes)
            notes.append(f"attempt {attempt + 1} failed validation: {v.violation_codes}")
        return NarrativeResult(
            text=_degraded_output(plan, tc, last_v.violations if last_v else []),
            render_mode=MODE_DEGRADED,
            validation=last_v.as_dict() if last_v else {}, attempts=2, degraded=True,
            llm_calls=2, contract_hash=tc.contract_hash, plan_version=plan.plan_version,
            word_count=0, notes=notes)
