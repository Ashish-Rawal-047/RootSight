# RootSight  — Round 2 prototype

A deterministic causal reasoning pipeline for KPI movements. The language model
phrases the answer; it never decides it.

## Run it

```bash
pip install numpy pandas scipy pyyaml fastapi uvicorn pydantic pytest
# optional: only needed to exercise the LLM rendering path
pip install anthropic

cd rootsight
python -m rootsight.datagen.generate      # writes data/raw/ (4 sources + ground truth)
python -m pytest -q                       # 38 tests -> artifacts/acceptance_report.md
python -m uvicorn rootsight.api.app:app --port 8000
# open http://127.0.0.1:8000
```

No build step, no node_modules, no external services. The UI is one HTML file.

To exercise the model path: `export ANTHROPIC_API_KEY=...` (or `$env:ANTHROPIC_API_KEY`
on Windows). Without a key the deterministic renderer runs, passes the same
validation, and token counts are labelled `ESTIMATED`.

## What to look at

| In the UI | Shows |
|---|---|
| **Analyse** (default) | movement, decomposition waterfall, five hypothesis cards with three panels each, narrative, actions, evidence table, lineage, LLM boundary, telemetry |
| Scenario selector | `SC_MULTIFACTOR` · `SC_LOWCONF` (abstention + clarification) · `SC_SPARSE` (21-day KPI) · `SC_SECURITY` |
| Persona selector | CFO · Operations Manager (North only) · Finance Analyst — same evidence, different narrative, depth and actions |
| **KPI contract** | the machine-readable contract the engine enforces, including the three non-interchangeable revenue definitions |
| **Data quality** | 9 injected defect classes, freshness, grain/cadence/calendar reconciliation |
| **Security probe** | 8 access attempts, 4 refused with audit ids |
| **Telemetry** | latency by stage and layer, model calls, tokens, cost |
| Button on the actions card | records an intervention and re-runs the *same* DID design on the recovery window |

## API

```
GET  /api/health          /api/personas       /api/scenarios
GET  /api/contract        /api/data_quality   /api/boundary
GET  /api/series?kpi_id=&persona_id=          (403s when not entitled)
POST /api/analyse         {scenario_id, persona_id, requested_grain?, requested_regions?}
GET  /api/security/probe?persona_id=&kpi_id=&grain=&region=
POST /api/intervention    {persona_id, hypothesis_id, playbook_id, implemented_on}
GET  /api/audit           /api/telemetry
```

## Layout

`contracts/` semantic contract · `security/` policy + audit · `datagen/` the SCM ·
`ingest/` DQ, freshness, reconciliation · `kpi/` calculation + lineage ·
`detect/` changepoint · `decompose/` exact price/volume/mix · `causal/` DAG,
gates, structure screen, estimators, identification · `evidence/` typed objects ·
`rank/` EWHR · `materiality/` two-axis engine · `recommend/` playbooks ·
`compiler/` Trust Contract, plan, renderer, validator, clarification ·
`scenarios/` declarative scenarios · `pipeline.py` · `api/`


