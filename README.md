# RootSight

### Evidence-Driven Intelligence for Explaining Business KPI Movements

RootSight is a deterministic, evidence-driven intelligence pipeline designed to explain business KPI movements.

It goes beyond identifying what changed. RootSight combines KPI contracts, data-quality checks, movement detection, deterministic decomposition, causal reasoning, evidence retrieval, hypothesis ranking, materiality assessment, confidence handling, security controls, and persona-specific recommendations to help explain why a KPI moved, what evidence supports the explanation, how confident the system is, and what action may be appropriate.

> **Core principle:** The language model phrases the answer; it never decides the analytical conclusion.

---

## 1. Prototype Overview

```text
Data Sources
     ↓
Ingestion & Data Quality
     ↓
KPI / Semantic Contract
     ↓
KPI Calculation
     ↓
Movement Detection
     ↓
Driver Decomposition
     ↓
Causal Reasoning & Evidence
     ↓
Hypothesis Ranking
     ↓
Materiality & Confidence
     ↓
Recommendations
     ↓
Trust-Validated Narrative
     ↓
Persona-Specific Insight
```

The prototype is designed so deterministic analytical components establish the facts and evidence before the language model is used for narrative rendering.

---

## 2. What the Prototype Demonstrates

| Capability | Demonstrated Through |
|---|---|
| Multi-factor KPI movement | `SC_MULTIFACTOR` |
| Low-confidence handling | `SC_LOWCONF` |
| Clarification / abstention | `SC_LOWCONF` |
| Sparse-history KPI | `SC_SPARSE` |
| Role-based access control | `SC_SECURITY` |
| Persona-specific intelligence | CFO / Operations Manager / Finance Analyst |
| KPI semantic contract | KPI Contract view |
| Data-quality validation | Data Quality view |
| Source freshness | Data Quality / Evidence |
| Evidence and lineage | Analyse view |
| Analytical method | Analyse / Evidence |
| LLM vs non-LLM boundary | Boundary view |
| Runtime telemetry | Telemetry view |
| Intervention and re-analysis | Actions card |

---

## 3. Running RootSight

### Requirements

Python 3.x

Install the core dependencies:

```bash
pip install numpy pandas scipy pyyaml fastapi uvicorn pydantic pytest
```

### Optional LLM dependency

Anthropic is optional and is only required to exercise the LLM rendering path:

```bash
pip install anthropic
```

---

## 4. Generate Prototype Data

From the project directory:

```bash
cd rootsight
python -m rootsight.datagen.generate
```

This writes the data required by the prototype to:

```text
data/raw/
```

The generated dataset includes the prototype source data and ground-truth information used for evaluation.

---

## 5. Run Tests

```bash
python -m pytest -q
```

The test suite produces:

```text
artifacts/acceptance_report.md
```

The repository also contains:

```text
artifacts/acceptance_report.json
```

---

## 6. Start the Application

```bash
python -m uvicorn rootsight.api.app:app --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

No frontend build step is required. The prototype UI is delivered as a single HTML file. There are no `node_modules` or external services required for the deterministic prototype path.

---

## 7. Optional LLM Rendering Path

RootSight can optionally use Anthropic for the language-model rendering path.

### Windows PowerShell

```powershell
$env:ANTHROPIC_API_KEY="your-api-key"
```

### Linux / macOS

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Without an API key, RootSight uses its deterministic renderer.

The deterministic renderer passes through the same validation path. When model usage is not available, token counts are labelled as **ESTIMATED** rather than being presented as actual model usage.

---

## 8. What to Explore in the UI

### Analyse

The Analyse view provides:

- KPI movement
- movement magnitude
- decomposition waterfall
- five hypothesis cards
- supporting analytical panels
- narrative
- recommended actions
- evidence table
- lineage
- LLM boundary
- runtime telemetry

This is the primary view for understanding how RootSight moves from a KPI change to evidence, reasoning, and action.

---

## 9. Prototype Scenarios

### SC_MULTIFACTOR

Demonstrates a KPI movement with multiple underlying factors.

The movement can be decomposed into:

- Volume
- Price
- Mix

This demonstrates multi-factor KPI reasoning using deterministic analytical calculations.

### SC_LOWCONF

Demonstrates low-confidence reasoning.

When evidence or analytical conditions are insufficient to support a strong conclusion, RootSight can:

- downgrade confidence
- request clarification
- abstain from unsupported causal claims

> When evidence is insufficient, the system should know when not to claim causality.

### SC_SPARSE

Demonstrates a sparse-history / newly launched KPI scenario.

The scenario uses a KPI with limited history, including the documented 21-day KPI case.

This tests whether the analytical engine appropriately handles limited historical evidence rather than blindly applying assumptions that require a long historical baseline.

Relevant situations include:

- newly launched products
- newly introduced KPIs
- new operational processes
- new regions
- recently introduced business initiatives

### SC_SECURITY

Demonstrates role-based security and entitlement enforcement.

The security probe exercises eight access attempts, with restricted requests refused and associated with audit IDs.

Controls include:

- persona
- KPI access
- permitted grain
- regional scope
- restricted data
- auditability

The API returns `403` when a requested series is not permitted for the selected persona.

---

## 10. Persona-Specific Intelligence

RootSight supports:

### CFO

Executive-oriented narrative focused on:

- business impact
- financial significance
- material drivers
- high-level actions

### Operations Manager

Operational narrative with appropriate operational depth and permitted regional scope.

The prototype includes a **North-only** entitlement scenario.

### Finance Analyst

More analytical narrative with greater analytical depth.

### Same Evidence, Different Insight

Personas can receive different:

- narrative depth
- emphasis
- recommendations
- permitted data
- operational context

while the underlying analytical evidence remains controlled and consistent.

```text
Underlying Evidence
        ↓
Persona / Entitlement
        ↓
Appropriate Narrative
        ↓
Appropriate Action
```

---

## 11. KPI / Semantic Contract

RootSight uses a machine-readable KPI contract defining the meaning and analytical rules associated with KPIs.

The contract includes:

- KPI definitions
- calculations
- drivers
- thresholds
- lineage
- source information
- grain
- cadence
- access restrictions

The prototype specifically demonstrates that different revenue definitions are not automatically interchangeable.

The semantic layer helps prevent ambiguity between business definitions and ensures downstream analytical and narrative components operate against controlled KPI definitions.

---

## 12. Data Quality & Freshness

RootSight includes data-quality and source-freshness checks.

The prototype demonstrates nine injected defect classes, along with:

- freshness validation
- grain reconciliation
- cadence reconciliation
- calendar reconciliation
- source consistency checks

These controls help establish whether data is suitable for analytical reasoning before conclusions are generated.

---

## 13. Evidence & Lineage

RootSight does not treat an analytical narrative as sufficient evidence by itself.

The Analyse view exposes:

- supporting evidence
- analytical contribution
- source information
- lineage
- analytical method
- confidence-related information

This enables a reviewer to trace an insight back toward underlying analytical evidence and source data.

---

## 14. Causal Reasoning

RootSight contains a dedicated causal reasoning layer.

Components include:

- DAG construction
- causal gates
- identification checks
- analytical structure
- estimators
- temporal reasoning
- causal effect estimation where supported

The system distinguishes between:

```text
Observed KPI movement
        ↓
Potential driver
        ↓
Supporting evidence
        ↓
Analytical validation
        ↓
Causal claim, if justified
```

A potential correlation is therefore not automatically presented as a proven causal relationship.

---

## 15. Hypothesis Ranking

RootSight generates and ranks potential explanations for a KPI movement.

The ranking layer considers analytical evidence and materiality so that the most relevant hypotheses can be surfaced.

The UI exposes five hypothesis cards for investigation.

---

## 16. Materiality

RootSight includes a materiality engine for assessing whether an identified movement or driver is significant enough to matter from a decision-making perspective.

Materiality is used alongside evidence and confidence rather than treating every detected change as equally important.

---

## 17. Recommendations & Playbooks

RootSight converts validated analytical findings into recommended actions.

The recommendation layer includes playbooks that connect identified hypotheses with appropriate actions.

Recommendations are downstream of the analytical pipeline rather than being generated independently by the language model.

The Analyse interface exposes the resulting actions.

---

## 18. Intervention & Re-analysis

The Actions card supports recording an intervention.

Endpoint:

```text
POST /api/intervention
```

An intervention can be recorded with:

```text
persona_id
hypothesis_id
playbook_id
implemented_on
```

After an intervention is recorded, RootSight re-runs the same difference-in-differences design on the recovery window where applicable.

```text
Insight
 ↓
Decision
 ↓
Intervention
 ↓
Observed Outcome
 ↓
Re-analysis
```

---

## 19. LLM vs Non-LLM Processing

A central design principle is a clear separation between analytical computation and language generation.

### Non-LLM / Deterministic Components

The deterministic pipeline performs:

- KPI calculation
- data-quality checks
- freshness checks
- reconciliation
- movement detection
- changepoint detection
- price / volume / mix decomposition
- causal gates
- identification checks
- causal estimation
- evidence construction
- hypothesis ranking
- materiality assessment
- security / entitlement enforcement
- validation
- trust-contract enforcement

### LLM Components

The language model is used primarily for:

- natural-language rendering
- narrative generation
- persona-specific communication

The LLM is not responsible for independently deciding the analytical conclusion.

> **The language model phrases the answer; it never decides it.**

---

## 20. Trust Contract

RootSight uses a Trust Contract layer to constrain narrative generation.

```text
Analytical Results
        ↓
Evidence
        ↓
Confidence
        ↓
Allowed Claims
        ↓
Trust Contract
        ↓
Narrative
```

The analytical system first produces structured information describing what can be stated. The narrative layer then renders that information into human-readable language.

---

## 21. Security & Auditability

RootSight includes:

- role-based access
- entitlement checks
- permitted grain
- regional restrictions
- restricted fields
- access denial
- audit IDs

The security probe exposes eight access attempts, including refused requests with audit identifiers.

```text
GET /api/security/probe
```

Series access is entitlement-aware:

```text
GET /api/series?kpi_id=&persona_id=
```

Unauthorized requests return:

```text
403
```

---

## 22. Runtime Telemetry

RootSight exposes runtime telemetry including:

- latency
- stage-level timing
- model calls
- token usage
- estimated cost

Endpoint:

```text
GET /api/telemetry
```

When the deterministic renderer is used without an LLM API call, token counts are labelled as **ESTIMATED**.

---

## 23. API

### Health and discovery

```text
GET /api/health
GET /api/personas
GET /api/scenarios
```

### Contract and system information

```text
GET /api/contract
GET /api/data_quality
GET /api/boundary
```

### KPI series

```text
GET /api/series?kpi_id=&persona_id=
```

Unauthorized requests return `403` when the persona is not entitled to the requested series.

### Analysis

```text
POST /api/analyse
```

Request parameters:

```text
scenario_id
persona_id
requested_grain?
requested_regions?
```

### Security

```text
GET /api/security/probe?persona_id=&kpi_id=&grain=&region=
```

### Intervention

```text
POST /api/intervention
```

Parameters:

```text
persona_id
hypothesis_id
playbook_id
implemented_on
```

### Audit

```text
GET /api/audit
```

### Telemetry

```text
GET /api/telemetry
```

---

## 24. Codebase Structure

```text
rootsight/
│
├── contracts/
│   └── semantic contract
│
├── security/
│   └── policy + audit
│
├── datagen/
│   └── prototype data generation
│
├── ingest/
│   └── data quality, freshness, reconciliation
│
├── kpi/
│   └── KPI calculation + lineage
│
├── detect/
│   └── changepoint / movement detection
│
├── decompose/
│   └── exact price / volume / mix decomposition
│
├── causal/
│   └── DAG, gates, structure, estimators, identification
│
├── evidence/
│   └── typed evidence objects and retrieval
│
├── rank/
│   └── hypothesis ranking
│
├── materiality/
│   └── materiality engine
│
├── recommend/
│   └── recommendation playbooks
│
├── compiler/
│   └── Trust Contract, plan, renderer,
│       validator and clarification
│
├── scenarios/
│   └── declarative prototype scenarios
│
├── pipeline.py
│   └── end-to-end orchestration
│
└── api/
    └── FastAPI application and UI
```

---

## 25. Repository Artifacts

```text
artifacts/
├── acceptance_report.json
├── acceptance_report.md
└── audit_log.jsonl
```

These provide additional evidence of:

- prototype validation
- acceptance testing
- audit information
- system behaviour

---

## 26. Prototype Evaluation Checklist

### Connected KPIs and multiple data sources

RootSight combines KPI relationships and multiple data sources with different grains/cadences.

### Semantic / KPI contract

Implemented through the machine-readable KPI contract.

### Multiple personas

Implemented through:

- CFO
- Operations Manager
- Finance Analyst

### Multi-factor KPI movement

Implemented through:

```text
SC_MULTIFACTOR
```

### Low-confidence scenario

Implemented through:

```text
SC_LOWCONF
```

including clarification and abstention behaviour.

### Sparse-history scenario

Implemented through:

```text
SC_SPARSE
```

including the 21-day KPI case.

### Role-based security

Implemented through:

```text
SC_SECURITY
```

and the security probe.

### Evidence and lineage

Available through the Analyse interface.

### Source freshness

Available through Data Quality and evidence information.

### Analytical method

Exposed through the analytical/evidence pipeline.

### Contribution

Shown through decomposition and evidence.

### Confidence

Used in determining how strongly RootSight communicates analytical conclusions.

### LLM vs non-LLM boundary

Explicitly exposed through the Boundary view and architecture.

### Runtime telemetry

Includes:

- latency
- model calls
- token usage
- estimated cost

---

## 27. Key Design Principles

### Evidence before narrative

RootSight generates analytical evidence before producing the final narrative.

### Deterministic computation before LLM rendering

Core numerical and analytical decisions are handled by deterministic components.

### Confidence-aware intelligence

The system can downgrade, clarify, or abstain when evidence is insufficient.

### Persona-aware communication

The same underlying evidence can be communicated differently depending on the user's role and entitlement.

### Security before exposure

Access restrictions are enforced before restricted information is exposed to a persona.

### Traceability

Insights are connected to evidence, lineage, analytical methods and audit information.

### Action-oriented intelligence

RootSight connects analytical findings to recommendations and allows interventions to be recorded for subsequent analysis.

---

## 28. Summary

RootSight demonstrates an end-to-end approach to evidence-driven business intelligence.

Instead of stopping at:

> **"The KPI changed."**

RootSight aims to provide:

> **What changed → what may have driven it → what evidence supports those drivers → what can be confidently claimed → what the relevant persona should do.**

The prototype combines deterministic analytics, causal reasoning, evidence, confidence, security, persona-aware communication, controlled LLM usage and runtime telemetry into a single workflow.

The result is an analytical system designed to make KPI explanations more **evidence-driven, transparent, actionable and trustworthy**.

---

## 29. Quick Start

```bash
pip install numpy pandas scipy pyyaml fastapi uvicorn pydantic pytest

cd rootsight

python -m rootsight.datagen.generate

python -m pytest -q

python -m uvicorn rootsight.api.app:app --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

Start with the **Analyse** view and explore:

1. `SC_MULTIFACTOR`
2. `SC_LOWCONF`
3. `SC_SPARSE`
4. `SC_SECURITY`

Then switch between:

- CFO
- Operations Manager
- Finance Analyst

to observe how the same evidence produces different persona-specific insights.

---

**RootSight — Evidence-driven intelligence for explaining business KPI movements.**
