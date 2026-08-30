# RootSight v5 Codebase Study Guide

This document is a comprehensive, deep-dive reverse-engineering of the RootSight v5 Round 2 Prototype codebase. It is designed to serve as a complete onboarding and study guide for a developer to deeply understand the architecture, data flow, causal reasoning layer, and exact implementation of the system.

---

## PART 1 — REPOSITORY MAP

The RootSight codebase is structured as a deterministic pipeline. Most of the logic resides in the `rootsight/` Python module.

### Directory Structure & Responsibilities

- **`api/`**: The FastAPI web server. Exposes HTTP endpoints.
- **`causal/`**: The statistical causal reasoning engine. Computes estimations, significance, and validates causal assumptions.
- **`compiler/`**: The narrative generation logic. Includes plan generation, clarification selection, Trust Contract assembly, and LLM rendering constraints.
- **`contracts/`**: Semantic business definitions. Includes the YAML registry mapping KPIs to drivers and their data access restrictions.
- **`datagen/`**: Simulates the initial operational, commercial, and financial data used for analysis.
- **`decompose/`**: Purely arithmetic (accounting) breakdowns, such as price/volume/mix analysis.
- **`detect/`**: Timeseries changepoint and anomaly detection.
- **`evidence/`**: Construction and retrieval of operational evidence for hypotheses (supporting, contradicting, context, gaps).
- **`ingest/`**: Data loading and data quality (DQ) checks, handling freshness, reconciliation, and defect flagging.
- **`kpi/`**: Centralised calculation logic for KPIs.
- **`materiality/`**: Scoring engine that assesses the financial impact/exposure of a hypothesis.
- **`rank/`**: Prioritisation engine (using EWHR) to sort hypotheses.
- **`recommend/`**: Suggestion engine using the playbook catalogue to match hypotheses with remediation or investigation actions.
- **`scenarios/`**: Hardcoded declarative scenarios for testing the system under different known states (e.g., `SC_MULTIFACTOR`).
- **`security/`**: Access control enforcement, determining what data, metrics, and evidence a persona is permitted to view.

### File-by-File Details

| File | Responsibility | Main Functions/Classes | Called By | Calls | Deterministic/LLM |
|------|----------------|------------------------|-----------|-------|-------------------|
| `api/app.py` | FastAPI entry point, handles routing & HTTP requests. | `analyse()`, `intervention()`, `probe()` | HTTP clients (Web UI) | `Pipeline.analyse()`, `PolicyEngine.decide()` | Deterministic |
| `pipeline.py` | The main orchestrator. Traces a request from access control to narrative. | `Pipeline.analyse()`, `warm_state()` | `api/app.py` | Almost all core modules (`PolicyEngine`, `MovementDetector`, `HypothesisEvaluator`, `EvidenceBuilder`, `TrustContractBuilder`, `NarrativeCompiler`) | Deterministic orchestrator |
| `contracts/kpi_registry.yaml` | Machine-readable declarative configuration of KPIs, their drivers, thresholds, and role access restrictions. | N/A | Parsed by `KpiContractRegistry` | N/A | N/A |
| `contracts/kpi_contract.py` | Python wrapper parsing `kpi_registry.yaml` | `registry()`, `KpiDefinition`, `Driver` | Many modules (`policy.py`, `pipeline.py`) | YAML parser | Deterministic |
| `causal/identify.py` | L1/L2/L3 identification logic determining causal status. | `HypothesisEvaluator.evaluate()` | `pipeline.py` | `estimate_did()`, `estimate_its()`, `temporal_compatibility()` | Deterministic |
| `causal/estimate.py` | Statistical estimators (OLS, DID with exact randomisation, ITS with HAC). | `estimate_did()`, `estimate_its()`, `fit_ols()` | `identify.py` | `scipy.stats`, `numpy` | Deterministic |
| `causal/gates.py` | Data sufficiency and temporal compatibility gates (e.g., lag matching). | `data_sufficiency()`, `temporal_compatibility()`, `benjamini_hochberg()` | `identify.py`, `pipeline.py` | N/A | Deterministic |
| `security/policy.py` | Enforces row, column, and domain-level security before analysis. | `PolicyEngine.decide()`, `AccessDecision` | `pipeline.py`, `api/app.py` | `contracts.kpi_contract` | Deterministic |
| `decompose/accounting.py` | Exact accounting decomposition (price/volume/mix + entry/exit). | `PriceVolumeMixDecomposer.run()` | `pipeline.py` | `KpiSeries` | Deterministic |
| `detect/changepoint.py` | KPI anomaly detection (seasonal adjustment + binary segmentation). | `MovementDetector.detect()` | `pipeline.py` | Numpy, Scipy | Deterministic |
| `compiler/trust_contract.py` | Constructs a verifiable contract of allowed facts and grammar. | `TrustContractBuilder.build()` | `pipeline.py` | Config | Deterministic |
| `compiler/render.py` | Validates and renders the narrative, falling back to deterministic template if LLM fails. | `NarrativeCompiler.compile()`, `_degraded_output()` | `pipeline.py` | `LlmClient`, `TrustContractValidator` | **LLM (rendering only)** / Deterministic |
| `recommend/engine.py` | Decision-theoretic playbook scoring. | `RecommendationEngine.recommend()` | `pipeline.py` | None | Deterministic |
| `scenarios/registry.py` | Defines expected states for evaluation (e.g. `SC_MULTIFACTOR`). | `get()`, `catalogue()` | `pipeline.py`, `api/app.py` | `identify.DidSpec` | Deterministic |

---

## PART 2 — ARCHITECTURE

### Plain English
RootSight v5 is a highly deterministic pipeline designed to explain *why* a business metric moved without letting an AI hallucinate the reasons. When a user requests an analysis, the system first checks their access rights, ensuring they can't even see data they shouldn't. It computes the metric, finds when the shift happened, and breaks it down arithmetically. Then, it evaluates a predefined list of "hypotheses" (e.g., a fulfilment disruption) using strict causal inference (Difference-in-Differences, Interrupted Time Series). It gathers evidence, ranks the hypotheses, and assigns a rigid "causal status". Finally, it locks all facts and numbers into a "Trust Contract". The LLM is only invoked at the very end to format these locked facts into natural language prose, and its output is strictly validated against the contract.

### Technical Pipeline

| Pipeline Stage | Exact File(s) | Function/Class | What Happens |
|----------------|---------------|----------------|--------------|
| Access Control | `security/policy.py` | `PolicyEngine.decide()` | Calculates `AccessDecision` (row/column/grain scope) based on Persona and KPI registry constraints. |
| KPI Calculation | `kpi/compute.py` | `KpiEngine.compute()` | Computes the target metric respecting the access decision constraints. |
| Movement Detection | `detect/changepoint.py` | `MovementDetector.detect()` | Removes calendar seasonality (OLS) and finds the changepoint using binary segmentation and block permutation test. |
| Decomposition | `decompose/accounting.py` | `PriceVolumeMixDecomposer.run()` | Computes price, volume, mix, entry, and exit arithmetic contributions. |
| Hypothesis Evaluation (Identify/Estimate) | `causal/identify.py` | `HypothesisEvaluator.evaluate()` | Validates temporal/sufficiency gates, checks graphical identification, runs DID/ITS estimators, applies robustness checks, sets causal status. |
| Multiplicity Control | `causal/gates.py` | `benjamini_hochberg()` | Corrects p-values for multiple comparisons across the tested hypotheses. |
| Evidence Assembly | `evidence/retrieve.py` | `EvidenceBuilder` | Collects operational, gap, and causal evidence attached to the hypotheses. |
| Materiality & Ranking | `rank/ewhr.py`, `materiality/engine.py` | `compute_ewhr()`, `MaterialityEngine.assess()` | Scores exposure INR and ranks hypotheses. |
| Recommendations | `recommend/engine.py` | `RecommendationEngine.recommend()` | Scores `PLAYBOOKS` items (benefit minus cost/risk) ensuring action types match causal status. |
| Trust Contract | `compiler/trust_contract.py` | `TrustContractBuilder.build()` | Generates a hashable whitelist of allowed numbers, fields, and grammar tied to causal status. |
| Narrative Render | `compiler/render.py` | `NarrativeCompiler.compile()` | Invokes LLM, runs validator on result. Falls back to degraded mode if validation fails. |

---

## PART 3 — TRACE ONE COMPLETE REQUEST

This traces the `SC_MULTIFACTOR` scenario for the `cfo` persona requesting analysis on `net_revenue`.

1. **USER/API REQUEST**
   ↓
   `api/app.py: analyse(req: AnalyseRequest)`
   ↓
2. **Access Control**
   `security/policy.py: PolicyEngine.decide(persona_id="cfo", kpi_id="net_revenue")`
   Calculates the CFO's access: allows all regions, denies warehouse grain and specific fields (e.g., HR/PII data).
   ↓
3. **KPI & Movement Detection**
   `kpi/compute.py: KpiEngine.compute()` (computes net revenue)
   `detect/changepoint.py: MovementDetector.detect()` (finds the drop at ~2026-07-28)
   ↓
4. **Accounting Decomposition**
   `decompose/accounting.py: PriceVolumeMixDecomposer.run()`
   Separates net revenue into Volume (-3.78pp), Mix (-0.05pp), and Realised Price (-3.38pp).
   ↓
5. **Hypothesis Evaluation Loop**
   `causal/identify.py: HypothesisEvaluator.evaluate()` is called for each hypothesis defined in the scenario.
   - For **H1 (Fulfilment)**:
     - `causal/gates.py: temporal_compatibility()` (verifies delay)
     - `causal/identify.py: _assess_did()`
     - `causal/estimate.py: estimate_did()` (Runs FWL two-way FE DID, yielding ATT and exact p-value)
     - Yields `SUPPORTED_BY_DESIGN`
   - For **H2 (Competitor)** & **H4 (Price)**:
     - `causal/dag.py: find_identification_strategy()` finds an unobserved confounder (promo intensity).
     - Yields `NOT_POINT_IDENTIFIED`.
   - For **H3 (Marketing)**:
     - `causal/gates.py: temporal_compatibility()` fails due to coarse grain.
     - Yields `ASSOCIATION_ONLY`.
   ↓
6. **Multiplicity Control**
   `pipeline.py: Pipeline.analyse()` calls `causal/gates.py: benjamini_hochberg()` on all valid p-values to prevent p-hacking.
   ↓
7. **Evidence Assembly**
   `pipeline.py: _operational_evidence()` and `_gap_evidence()` inject supporting/contradicting context (e.g., dispatch rates falling in North, missing data).
   ↓
8. **Ranking & Materiality**
   `pipeline.py` calls `MaterialityEngine.assess()` and `rank_all(ewhrs)`.
   ↓
9. **Recommendation Selection**
   `recommend/engine.py: RecommendationEngine.recommend()` evaluates `PLAYBOOKS` against the hypotheses and their materiality.
   ↓
10. **Trust Contract & Plan**
    `compiler/trust_contract.py: TrustContractBuilder.build()` creates the strict whitelist of numbers and vocabulary.
    `compiler/plan.py: PlanBuilder.build()` formats sections (measured, estimate, assumptions).
    ↓
11. **Narrative Render & Validation**
    `compiler/render.py: NarrativeCompiler.compile()` calls the LLM, then `TrustContractValidator.validate(reply.text, tc)` checks the LLM's output.
    ↓
12. **FINAL RESPONSE**
    Returns JSON to the client.

---

## PART 4 — USE THE SC_MULTIFACTOR EXAMPLE

`SC_MULTIFACTOR` scenario (from `scenarios/registry.py`) analyzes a **-7.22%** drop in `net_revenue`.

**Decomposition Breakdown:**
In `pipeline.py`, `PriceVolumeMixDecomposer.run()` is called. It uses the exact identity: `dR = volume + mix + price + entry - exit`.
- **Volume:** `(Q1 - Q0) * pbar0` (Calculates to ~-3.78pp)
- **Mix:** Base-priced share shift (Calculates to ~-0.05pp)
- **Realised Price:** Focus-quantity price shift (Calculates to ~-3.38pp)

**Hypothesis Path: H1 (North fulfilment disruption)**
- Defined in `scenarios/registry.py` under `_multifactor()`.
- **Temporal compatibility:** Tested in `causal/gates.py` checking if the driver (`on_time_dispatch_rate`) moves before/with the KPI.
- **Structure/graphical identification:** Checked in `causal/dag.py`. The DAG identifies an adjustment set.
- **DID eligibility:** `causal/identify.py::_assess_did()` checks for treated/control units and sufficient pre/post periods.
- **DID Estimation:** `causal/estimate.py::estimate_did()` is called. It uses Frisch-Waugh-Lovell to absorb cell FE and product-line-by-day FE.
- **ATT & p-value:** The ATT is the reduced coefficient. The p-value (`p_randomisation`) comes from an exact randomisation inference looping over all possible treated assignments (since 9 clusters make cluster-robust SEs untrustworthy).
- **Status:** Passes all three layers (Graphical, Design, Data) resulting in `SUPPORTED_BY_DESIGN` defined in `causal/identify.py`.

**Hypothesis Path: H4 (Price)**
- Result: `NOT_POINT_IDENTIFIED`
- Why: `causal/dag.py:find_identification_strategy()` is called. It detects that to identify the effect of price on volume, it must control for `competitor_promo_intensity`.
- Since this variable is structurally marked as unobserved, `identify.py` returns `NOT_POINT_IDENTIFIED`. No estimator runs.

**Hypotheses H3 (Marketing):**
- Result: `ASSOCIATION_ONLY`.
- Why: `temporal_compatibility()` detects `GRAIN_TOO_COARSE`. The marketing feed is weekly, but the mechanism requires a lag shorter than 7 days.

**Hypothesis H5 (Complaints visibility):**
- Result: `NOT_POINT_IDENTIFIED` (No quasi-experimental design is defined for it).

---

## PART 5 — DATA FLOW

Trace of `SC_MULTIFACTOR`:

1. **INPUT:** Raw ERP, Marketing, and Ops rows loaded by `ingest/loaders.py`.
   ↓
2. **TRANSFORMATION (Reconciliation):** `ingest/reconcile.py` conforms dates and standardizes columns, producing a `ConformedData` object.
   ↓
3. **TRANSFORMATION (Calculation):** `kpi/compute.py` uses `ConformedData` + `AccessDecision` to aggregate daily arrays (`pd.Series`).
   ↓
4. **MOVEMENT:** `detect/changepoint.py` takes the series and outputs a `Movement` dataclass containing `pct_change`, `p_value`, etc.
   ↓
5. **EVIDENCE:** `EvidenceBuilder` creates `list[dict]` of evidence items (e.g. `stance: SUPPORT`, `source_id: SRC_OPS`).
   ↓
6. **HYPOTHESIS RESULTS:** `HypothesisEvaluator` outputs `Hypothesis` dataclasses detailing `causal_status`, `effect` dict, and `assumptions`.
   ↓
7. **TRUST CONTRACT:** `TrustContractBuilder` generates a `TrustContract` dataclass, mapping allowed verbs based on causal status and extracting an exhaustive `allowed_numbers` list from all previous objects.
   ↓
8. **NARRATIVE:** `NarrativeCompiler` outputs a `NarrativeResult` containing validated Markdown prose.

---

## PART 6 — CAUSAL ENGINE DEEP DIVE

RootSight's engine decides causal status in `causal/identify.py`. It uses a 3-layer conjunction:
1. **L1 (Graphical):** `dag.py` looks for an adjustment set via backdoor/frontdoor logic.
2. **L2 (Design):** Pre-tests for DID/ITS are run.
3. **L3 (Data):** Sufficiency and temporal compatibility.

- **Temporal Compatibility (`causal/gates.py`):** Ensures the driver moves before the KPI via detrended cross-correlation. Fails if grain is too coarse (e.g. Weekly data for a 2-day lag mechanism).
- **Structure Screen (`causal/structure.py`):** Tests if observational data contradicts the DAG's declared edges.
- **DID Estimation (`causal/estimate.py`):**
  - *Stat Concept:* Compares the change in treated units to the change in control units to difference out common shocks.
  - *Implementation:* Uses `estimate_did()`. Employs Frisch-Waugh-Lovell to absorb Fixed Effects (Cell and Line x Day).
  - *Randomisation Inference:* Since cluster-robust SEs fail with few clusters, the code exhaustively permutes the treatment assignment across all cells to calculate an exact p-value (`p_randomisation`).
- **Parallel Trends:** Checked as a pre-test in DID using an interaction term (`t * treated_unit`). If it fails, DID design is rejected.
- **Robustness Checks:** RootSight explicitly rejects E-Values (which are for risk ratios). Instead, `identify.py::_robustness()` runs: Leave-One-Control-Out, Alternative FE Specifications, and Negative-Control Outcomes (e.g. checking if the mechanism improperly affects `list_price`).
- **Abstention:** If NO hypotheses can be point-identified, `compiler/clarify.py` generates a request to the user to fix missing data instead of forcing a fake result.

---

## PART 7 — EVIDENCE SYSTEM

The evidence system (`evidence/retrieve.py` & `pipeline.py`) attaches multi-source context to hypotheses.
- **Creation:** Operational evidence is manually assembled in `pipeline.py::_operational_evidence()`. E.g., it checks if regional dispatch rates fell.
- **Stances:** Evidence is tagged as `SUPPORT` or `CONTRADICT`. For example, if untreated control regions *also* fell significantly, it generates `CONTRADICT` evidence against the North Fulfilment hypothesis.
- **Gap Evidence:** `pipeline.py::_gap_evidence()` injects data-quality defects (stale feeds, missing variables) directly into the evidence payload.
- **Retrieval:** Evidence is filtered via `AccessDecision`. An ops manager might see warehouse-level ticket clusters, while a CFO only sees regional aggregates.

---

## PART 8 — TRUST CONTRACT

Defined in `compiler/trust_contract.py`. This is the security boundary against LLM hallucinations.

- **What it is:** A deterministically built, hashed configuration generated *before* any LLM call.
- **Allowed Numbers (`number_whitelist()`):** Recursively traverses the entire JSON payload of the deterministic engine to find *every* number produced. Only these numbers (plus lakh/crore formats) are permitted in the output.
- **Grammar Restrictions (`CLAIM_GRAMMAR`):** Based strictly on `causal_status`.
  - If `SUPPORTED_BY_DESIGN`: Allows "is estimated to have reduced", mandates CI inclusion.
  - If `NOT_POINT_IDENTIFIED`: Forbids "caused", "drove". Only allows "co-moved with", "is the leading hypothesis for". **Forbids reporting an effect magnitude entirely.**
- **Enforcement:** `compiler/validator.py` parses the LLM output. It runs regex to extract every number and verifies it exists in the Trust Contract whitelist (with tolerance). If a number is invented, or a forbidden word is used, the validation fails.
- **Why LLMs can't invent "marketing caused it":** Because if Marketing is `ASSOCIATION_ONLY`, the grammar explicitly forbids the word "caused", and forbids printing an effect size. The validator will catch it and downgrade to `MODE_DEGRADED` (a bulleted list).

---

## PART 9 — LLM / NON-LLM BOUNDARY

⚠️ **DOCUMENTATION vs IMPLEMENTATION MISMATCH:** None. The architecture accurately reflects the strict boundary.

| Component | LLM? | Responsibility | Can change numbers? | Can decide causality? |
|-----------|------|----------------|---------------------|-----------------------|
| `kpi.compute` | NON_LLM | Computes metrics | Yes | No |
| `causal.identify` | NON_LLM | Determines if effect is valid | Yes | Yes (Deterministically) |
| `trust_contract` | NON_LLM | Locks facts and grammar | No | No |
| `compiler.render` | **LLM** | Translates plan to prose | NO (Strictly enforced) | NO |

**Enforcement:** `compiler/render.py` calls the model. The output is tested by `TrustContractValidator`. If the model hallucinates, it is retried once. If it fails again, the output reverts to `DEGRADED_EVIDENCE_TABLE`—a UI rendering without prose.
The metric "Measured share of pipeline time: NON_LLM 100%" is literal: `pipeline.py` tags `Telemetry.span` with `layer="NON_LLM"` or `"LLM"`. The LLM is technically 0% of the *analysis*, merely acting as a presentation layer.

---

## PART 10 — RECOMMENDATION ENGINE

Located in `recommend/engine.py`. Maps causal findings to the `PLAYBOOKS` catalog.

- **The Rule:** The `action_type` (REMEDIATE, INVESTIGATE, MONITOR) is hard-capped by the causal status. You cannot "REMEDIATE" a `NOT_POINT_IDENTIFIED` hypothesis.
- **Scoring:** `score = benefit - cost - risk`.
  - `benefit = min(impact_index, identification_cap)`
  - The `identification_cap` (e.g., 0.85 for DESIGN, 0.45 for NOT_POINT) acts as a strict ceiling.
- **SC_MULTIFACTOR Example:**
  1. H1 (Fulfilment) is `SUPPORTED_BY_DESIGN`. Allows `REMEDIATE`. Selects PB-OPS-001 (Activate overflow 3PL) because its benefit (0.72) is below the cap (0.85), scoring high.
  2. H4 (Price) is `NOT_POINT_IDENTIFIED`. Forces `INVESTIGATE`. Selects PB-PRC-001 (Review realised-price erosion).
- **Persona Framing:** Playbooks have different descriptions based on persona. The CFO sees "Incremental logistics spend", the Ops Manager sees "Raise 3PL overflow lane".

---

## PART 11 — SECURITY / PERSONA

Handled in `security/policy.py`.

- **Entitlements:** Handled via `PolicyEngine.decide()`. Intersects the persona definition (`PERSONAS`) with the KPI restrictions (`kpi_registry.yaml`).
- **SC_SECURITY Example:**
  - `cfo` has a row scope of all regions, but a column scope that denies `ticket_body` and a grain cap that prevents warehouse-level data.
  - `ops_manager` has a row scope restricted to `"North"` and warehouse-level grain, but denied commercial fields like `gross_margin_pct`.
- **Enforcement:** Applied *before* data processing. The `ConformedData` is sliced using the `AccessDecision`. Additionally, `assert_prompt_safe` acts as a final tripwire before calling the LLM, ensuring no PII/HR fields leak into the prompt payload.

---

## PART 12 — DATA QUALITY

Handled in `ingest/loaders.py` and `pipeline.py::_gap_evidence`.

- **Detection:** Injected defects (e.g., negative units shipped, stale marketing feeds) are flagged by `Reconciler` and loaders.
- **Impact:**
  - *Negative Units:* Detected as a severe defect; rows are quarantined (excluded from `net_revenue` calculation).
  - *Stale Feed (South Marketing):* Emitted as a `STALE` gap. Fails the temporal/sufficiency gate.
  - *Missing Promo Intensity:* DAG marks it unobserved -> Identification fails -> Result is `NOT_POINT_IDENTIFIED`.
- **Narrative:** `_gap_evidence` converts these into evidence objects. The Trust Contract mandates their inclusion in the `gaps` section of the prose.

---

## PART 13 — LINEAGE

Trace for Net Revenue -7.22%:
1. `SRC_ERP.orders_daily` (Raw data)
2. `Reconciler` (Transforms grain, removes quarantined DQ rows)
3. `KpiEngine.compute('net_revenue')` (Applies aggregation rules from YAML)
4. `MovementDetector.detect()` (Calculates Baseline vs Focus means: Yields -7.22%)
5. Passed as `movement.as_dict()` to `TrustContractBuilder`.
6. Stored in `kpi_contract` (lineage payload) in the final JSON, making it perfectly reproducible by an auditor since the exact partitions and rules are appended.

---

## PART 14 — TELEMETRY

Handled in `telemetry.py` and orchestrated in `pipeline.py`.

- **Timings:** Wraps every major block in `with t.span("name", layer="NON_LLM")`.
- **Tokens/Cost:** `Telemetry.record_model_call` tracks prompts/completions. If deterministic mode is on, it estimates tokens (`chars / 4`).
- **Reporting:** `/api/telemetry` reads history and outputs latency medians and cost projections.

---

## PART 15 — FILE-BY-FILE STUDY GUIDE

**LEVEL 1 — Start here**
1. `pipeline.py`: Read this because it is the entire orchestration layer. Before reading, understand the basic pipeline flow.
2. `api/app.py`: Read this because it's the entry point. Look at `/api/analyse`.

**LEVEL 2 — Core pipeline**
3. `contracts/kpi_registry.yaml` & `contracts/kpi_contract.py`: Understand how business logic is declaratively defined.
4. `security/policy.py`: Understand how access is intercepted *first*.

**LEVEL 3 — Causal engine**
5. `causal/identify.py`: The heart of the reasoning layer (L1/L2/L3 logic).
6. `causal/estimate.py`: The actual statistical implementations. Look closely at `estimate_did()`.
7. `causal/gates.py`: Understand temporal delays and multiplicity control.

**LEVEL 4 — Evidence/trust/security**
8. `compiler/trust_contract.py`: Vital. Understand the `_collect_numbers` whitelist and the `CLAIM_GRAMMAR`.
9. `compiler/render.py`: See how the LLM is constrained and forced into degraded mode if it fails.

**LEVEL 5 & 6 — Recommendations & Scenarios**
10. `recommend/engine.py`: Understand how business actions are strictly capped by causal certainty.

---

## PART 16 — FUNCTION CALL GRAPH (SC_MULTIFACTOR)

```text
Pipeline.analyse()
├── PolicyEngine.decide()
├── KpiEngine.compute(net_revenue)
├── MovementDetector.detect()
├── Pipeline._connected()
├── PriceVolumeMixDecomposer.run()
├── HypothesisEvaluator.evaluate()
│   ├── data_sufficiency()
│   ├── temporal_compatibility()
│   ├── find_identification_strategy()
│   ├── HypothesisEvaluator._assess_did()
│   │   └── estimate_did()
│   └── HypothesisEvaluator._decide()
├── benjamini_hochberg()
├── EvidenceBuilder.from_movement()
├── Pipeline._operational_evidence()
├── MaterialityEngine.assess()
├── rank_all()
├── RecommendationEngine.recommend()
├── TrustContractBuilder.build()
│   └── _collect_numbers()
├── PlanBuilder.build()
└── NarrativeCompiler.compile()
    ├── LlmClient.render()
    └── TrustContractValidator.validate()
```

---

## PART 17 — "IF I CHANGE THIS FILE, WHAT BREAKS?"

- **`contracts/kpi_registry.yaml`**: Changes here propagate everywhere. Changing `max_grain` alters access control; changing a driver's `prior_lag_days` can break temporal compatibility in `causal/gates.py`.
- **`causal/identify.py`**: Changing status mappings here directly affects `recommend/engine.py` (which playbooks are allowed) and `trust_contract.py` (which verbs the LLM can use).
- **`compiler/trust_contract.py`**: If you add a new metric to the pipeline but forget to whitelist it in `_collect_numbers`, the Validator will block the LLM from outputting it, resulting in Degraded Mode.

---

## PART 18 — TESTS AS SPECIFICATION

*(Inferred from architecture)*
Tests (e.g., `acceptance_report.md` mentioned in README) act as architectural guarantees:
- **Causal tests:** Ensure DID fails if pre-trends diverge.
- **Trust Contract tests:** Inject fake numbers into mock LLM outputs to prove the `TrustContractValidator` correctly catches them and triggers Degraded Mode.
- **Security tests:** Attempt to query `net_revenue` with warehouse grain as a CFO, proving it raises `AccessDenied` before any data is processed.

---

## PART 19 — DO NOT HIDE COMPLEXITY

- **`estimate_did()` Randomisation Inference:** This is a computationally intense workaround because cluster-robust SEs fail for small N (9 warehouses). It runs a full permutation test inside the web request loop. It's affordable only because of Frisch-Waugh-Lovell matrix absorption, but it's a known bottleneck.
- **`pipeline.py` Evidence Assembly:** The `_operational_evidence()` method has hardcoded regional/product-line strings (e.g., "North", "Electronics"). This is a clear prototype shortcut. In a real system, this would be abstracted to a config or a separate expert system rulebase.
- **LLM Usage:** Calling an LLM to generate prose *after* generating a perfectly structured `TrustContract` and `NarrativePlan` is arguably unnecessary. The system could just concatenate strings. It exists to demonstrate "safe" LLM usage, but architecturally, it's a very expensive string formatter.

---

## PART 20 — FINAL MENTAL MODEL

### "If I receive a new KPI analysis request tomorrow, what happens?"
1. **Gatekeeper:** `PolicyEngine` checks your badge. Can you see this KPI? Which regions? (If no, reject immediately).
2. **Calculator:** `KpiEngine` and `MovementDetector` fetch your scoped data, adjust for holidays, and find the percentage drop.
3. **Scientist:** `HypothesisEvaluator` takes the known theories. It checks if the data exists (`gates.py`), if the math is possible (`dag.py`), and runs the experiment (`estimate_did()`). It assigns a strict label (e.g., `SUPPORTED_BY_DESIGN`).
4. **Accountant:** `MaterialityEngine` calculates how much money is at stake.
5. **Manager:** `RecommendationEngine` looks at the Scientist's label and Accountant's numbers, then picks an action from a fixed playbook.
6. **Lawyer:** `TrustContractBuilder` writes down every single number produced above, and dictates exactly what words can be used to describe the Scientist's findings.
7. **Speaker:** `NarrativeCompiler` (The LLM) is handed the Lawyer's contract and asked to read it aloud nicely. If it ad-libs, the Lawyer fires it and prints a bulleted list instead.

### 20 Questions I Should Be Able To Answer After Studying This:
1. Why doesn't RootSight use E-Values?
2. How does `TrustContractValidator` prevent LLM hallucinations?
3. What is the difference between `NOT_POINT_IDENTIFIED` and `ASSOCIATION_ONLY`?
4. How does `PolicyEngine` enforce grain-level security?
5. Why are day-of-week and holiday effects removed *before* binary segmentation?
6. Why does `estimate_did()` use exact randomisation inference instead of cluster-robust standard errors?
7. How does Frisch-Waugh-Lovell (FWL) make randomisation inference fast enough for an API?
8. Why is `competitor_promo` marked as `NOT_POINT_IDENTIFIED` in SC_MULTIFACTOR?
9. How does the recommendation engine score playbooks?
10. Why is the identification cap in `RecommendationEngine` a ceiling and not a multiplier?
11. How are newly launched products handled in the price/volume/mix decomposition?
12. Where are the allowed verbs for causal claims defined?
13. How does `assert_prompt_safe` act as a tripwire?
14. What happens if the LLM output fails Trust Contract validation twice?
15. How are concurrent shocks handled in ITS versus DID?
16. How does RootSight handle multiple testing (p-hacking)?
17. Why is `units_sold` calculated when the requested KPI is `net_revenue`?
18. How does `temporal_compatibility()` handle coarse-grained data?
19. What does `_collect_numbers()` do in `trust_contract.py`?
20. Why does `SC_SPARSE` return a descriptive benchmark instead of a causal counterfactual?
