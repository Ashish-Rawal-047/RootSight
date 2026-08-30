"""Global configuration. Every tunable that affects a result lives here so it can
be version-stamped into the reproduction bundle."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from datetime import date

VERSION = "5.0.0"

# ---------------------------------------------------------------- time windows
HISTORY_START = date(2026, 1, 1)
HISTORY_END = date(2026, 8, 25)          # "today" in the demo world is 2026-08-26
AS_OF = date(2026, 8, 26)

# ---------------------------------------------------------------- entities
REGIONS = ["North", "South", "East", "West"]
PRODUCT_LINES = ["Apparel", "Electronics", "Home", "SmartHome"]
WAREHOUSES = {
    "WH-N1": "North", "WH-N2": "North",
    "WH-S1": "South", "WH-E1": "East", "WH-W1": "West",
}

# SmartHome is the newly launched line (sparse-history scenario)
NEW_LINE = "SmartHome"
NEW_LINE_REGION = "South"
NEW_LINE_LAUNCH = date(2026, 8, 7)


# ---------------------------------------------------------------- ground truth
# The synthetic generator is an explicit structural causal model.  These knobs
# ARE the ground truth and are used by the evaluation harness (never by the
# analysis pipeline).
@dataclass(frozen=True)
class GroundTruth:
    seed: int = 20260826

    # dispatch disruption in North
    disruption_region: str = "North"
    disruption_start: date = date(2026, 7, 28)
    disruption_end: date = date(2026, 8, 20)
    disruption_units_multiplier: float = 0.86      # -14% units in North while active
    disruption_lag_days: int = 3                   # dispatch -> order effect lag

    # competitor promotion (external, partially observed)
    competitor_start: date = date(2026, 7, 25)
    competitor_end: date = date(2026, 8, 12)
    competitor_line: str = "Electronics"
    competitor_units_multiplier: float = 0.90      # -10% Electronics units, all regions

    # marketing spend cut (weekly grain only)
    marketing_cut_week_start: date = date(2026, 7, 27)
    marketing_cut_fraction: float = 0.22
    marketing_units_multiplier: float = 0.975      # -2.5% units, national

    # deliberate price action
    price_line: str = "Apparel"
    price_start: date = date(2026, 8, 1)
    price_increase: float = 0.03

    # mix drift: Electronics share falls, Home share rises
    mix_drift_start: date = date(2026, 8, 1)
    mix_drift: float = 0.060

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: (v.isoformat() if isinstance(v, date) else v) for k, v in d.items()}


GT = GroundTruth()

# ---------------------------------------------------------------- analysis window
FOCUS_WINDOW = (date(2026, 8, 1), date(2026, 8, 25))
BASELINE_WINDOW = (date(2026, 6, 1), date(2026, 7, 24))

# ---------------------------------------------------------------- engineering gates
@dataclass(frozen=True)
class Gates:
    min_pre_obs: int = 30          # engineering minimum, NOT a power guarantee
    min_post_obs: int = 10
    max_missing_frac: float = 0.30
    min_did_treated_units: int = 2
    min_did_control_units: int = 2
    min_did_pre_periods: int = 8
    min_its_pre_points: int = 20
    min_its_post_points: int = 5
    min_abs_corr: float = 0.30
    structure_stability_threshold: float = 0.75   # subsample edge agreement
    parallel_trends_alpha: float = 0.10           # pre-test is a screen, so lenient
    placebo_alpha: float = 0.10
    bh_q: float = 0.05
    bootstrap_n: int = 400
    subsample_n: int = 60


GATES = Gates()

# ---------------------------------------------------------------- cost model
@dataclass(frozen=True)
class CostModel:
    """Prices are USD per 1M tokens.  Source: Anthropic public pricing, stamped
    with the date it was recorded so the number is auditable, not folklore."""
    price_recorded_on: str = "2026-08-26"
    models: dict = field(default_factory=lambda: {
        "claude-haiku-4-5-20251001": {"in": 1.00, "out": 5.00},
        "claude-sonnet-5": {"in": 3.00, "out": 15.00},
        "claude-opus-5": {"in": 15.00, "out": 75.00},
        "deterministic-template-v5": {"in": 0.0, "out": 0.0},
    })
    usd_to_inr: float = 87.4
    fx_recorded_on: str = "2026-08-26"


COSTS = CostModel()

# ---------------------------------------------------------------- LLM
NARRATIVE_MODEL = os.environ.get("ROOTSIGHT_NARRATIVE_MODEL", "claude-haiku-4-5-20251001")
EXTRACTION_MODEL = os.environ.get("ROOTSIGHT_EXTRACTION_MODEL", "claude-haiku-4-5-20251001")
LLM_ENABLED = bool(os.environ.get("ANTHROPIC_API_KEY"))
MAX_LLM_CALLS_PER_ANALYSIS = 2
PROMPT_TEMPLATE_VERSION = "ccc-prompt-v5.0.1"

MODEL_VERSIONS = {
    "rootsight": VERSION,
    "changepoint_detector": "stl-cusum-v5.0",
    "structure_screen": "pcorr-subsample-v5.0",
    "identification": "bayesball-backdoor-v5.0",
    "estimator_did": "ols-cluster-ri-v5.0",
    "estimator_its": "ols-hac-fourier-v5.0",
    "decomposition": "lmdi-pvm-v5.0",
    "ewhr": "ewhr-v5.0",
    "materiality": "materiality-matrix-v5.0",
    "trust_contract": "tc-v5.0",
    "prompt_template": PROMPT_TEMPLATE_VERSION,
    "narrative_model": NARRATIVE_MODEL if LLM_ENABLED else "deterministic-template-v5",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
