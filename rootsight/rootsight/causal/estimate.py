"""Estimators, implemented directly so every standard error is inspectable.

Contains:
  * OLS with classic / HC1 / cluster-robust / Newey-West HAC covariance
  * DID  : two-way fixed effects (cell FE + product-line x day FE), cluster-robust
           SEs by cell, PLUS an exact randomisation-inference p-value over every
           possible assignment of treated cells.  With 9 clusters, cluster-robust
           asymptotics are not trustworthy; randomisation inference is exact
           under the sharp null and is reported as the primary p-value.
  * ITS  : level + slope change with Fourier weekly terms, holiday and promo
           controls, Newey-West HAC SEs, plus a pre-period placebo test.

Nothing in this module decides whether an estimate is *allowed*.  It computes,
diagnoses, and reports.  The identification layer decides.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------------ OLS
@dataclass
class OlsFit:
    beta: np.ndarray
    names: list[str]
    resid: np.ndarray
    xtx_inv: np.ndarray
    X: np.ndarray
    y: np.ndarray
    n: int
    k: int          # rank of X, used for degrees of freedom
    p: int          # number of columns of X, used for matrix dimensions

    def se(self, kind: str = "classic", *, clusters: np.ndarray | None = None,
           hac_lags: int | None = None) -> np.ndarray:
        return np.sqrt(np.diag(self.vcov(kind, clusters=clusters, hac_lags=hac_lags)))

    def vcov(self, kind: str = "classic", *, clusters: np.ndarray | None = None,
             hac_lags: int | None = None) -> np.ndarray:
        n, k, p, X, u = self.n, self.k, self.p, self.X, self.resid
        A = self.xtx_inv
        if kind == "classic":
            s2 = float(u @ u) / max(n - k, 1)
            return s2 * A
        if kind == "hc1":
            meat = (X * (u ** 2)[:, None]).T @ X
            return (n / max(n - k, 1)) * A @ meat @ A
        if kind == "cluster":
            if clusters is None:
                raise ValueError("cluster covariance requires cluster ids")
            groups = pd.unique(clusters)
            g = len(groups)
            meat = np.zeros((p, p))
            for gid in groups:
                m = clusters == gid
                Xg, ug = X[m], u[m]
                s = Xg.T @ ug
                meat += np.outer(s, s)
            adj = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
            return adj * A @ meat @ A
        if kind == "hac":
            L = hac_lags if hac_lags is not None else int(np.floor(4 * (n / 100.0) ** (2 / 9)))
            Xu = X * u[:, None]
            meat = Xu.T @ Xu
            for lag in range(1, L + 1):
                w = 1.0 - lag / (L + 1.0)                      # Bartlett kernel
                G = Xu[lag:].T @ Xu[:-lag]
                meat += w * (G + G.T)
            return (n / max(n - k, 1)) * A @ meat @ A
        raise ValueError(f"unknown covariance kind {kind!r}")

    def t_and_p(self, idx: int, kind: str = "classic", *, clusters=None,
                hac_lags=None, df: int | None = None) -> tuple[float, float, float]:
        se = self.se(kind, clusters=clusters, hac_lags=hac_lags)[idx]
        b = float(self.beta[idx])
        t = b / se if se > 0 else np.nan
        dof = df if df is not None else max(self.n - self.k, 1)
        p = float(2 * stats.t.sf(abs(t), dof)) if np.isfinite(t) else np.nan
        return b, se, p

    def ci(self, idx: int, kind: str = "classic", *, clusters=None, hac_lags=None,
           df: int | None = None, level: float = 0.95) -> tuple[float, float]:
        b, se, _ = self.t_and_p(idx, kind, clusters=clusters, hac_lags=hac_lags, df=df)
        dof = df if df is not None else max(self.n - self.k, 1)
        crit = float(stats.t.ppf(0.5 + level / 2, dof))
        return b - crit * se, b + crit * se

    @property
    def r2(self) -> float:
        ss_tot = float(np.sum((self.y - self.y.mean()) ** 2))
        return 1.0 - float(self.resid @ self.resid) / ss_tot if ss_tot > 0 else np.nan


def fit_ols(X: np.ndarray, y: np.ndarray, names: list[str]) -> OlsFit:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)
    return OlsFit(beta=beta, names=names, resid=resid, xtx_inv=xtx_inv, X=X, y=y,
                  n=len(y), k=int(np.linalg.matrix_rank(X)), p=X.shape[1])


# ------------------------------------------------------------------ diagnostics
def durbin_watson(u: np.ndarray) -> float:
    d = np.diff(u)
    return float(d @ d / (u @ u)) if u @ u > 0 else np.nan


def ljung_box(u: np.ndarray, lags: int = 10) -> tuple[float, float]:
    n = len(u)
    u = u - u.mean()
    denom = float(u @ u)
    q = 0.0
    for k in range(1, lags + 1):
        r = float(u[k:] @ u[:-k]) / denom
        q += r * r / (n - k)
    q *= n * (n + 2)
    return q, float(stats.chi2.sf(q, lags))


def fourier_terms(t: np.ndarray, period: float, harmonics: int) -> np.ndarray:
    cols = []
    for h in range(1, harmonics + 1):
        cols.append(np.sin(2 * np.pi * h * t / period))
        cols.append(np.cos(2 * np.pi * h * t / period))
    return np.column_stack(cols)


# ------------------------------------------------------------------------ DID
@dataclass
class DidResult:
    estimator: str
    outcome: str
    treated_cells: list[str]
    control_cells: list[str]
    t0: date
    att_per_unit_day: float
    att_pct_of_treated_base: float
    se_cluster: float
    ci95: tuple[float, float]
    p_cluster: float
    p_randomisation: float
    n_permutations: int
    n_clusters: int
    n_obs: int
    pre_periods: int
    post_periods: int
    parallel_trends: dict
    fixed_effects: str
    diagnostics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["t0"] = self.t0.isoformat()
        d["ci95"] = [round(float(x), 4) for x in self.ci95]
        for k in ("att_per_unit_day", "att_pct_of_treated_base", "se_cluster",
                  "p_cluster", "p_randomisation"):
            d[k] = round(float(d[k]), 6)
        return d


def _dummies(labels: np.ndarray, drop_first: bool = True) -> tuple[np.ndarray, list[str]]:
    cats = list(pd.unique(labels))
    cats = cats[1:] if drop_first else cats
    if not cats:
        return np.zeros((len(labels), 0)), []
    M = np.column_stack([(labels == c).astype(float) for c in cats])
    return M, [str(c) for c in cats]


def _orthonormal_basis(W, tol: float = 1e-9):
    """Rank-safe orthonormal basis of the column space of W (via SVD)."""
    U, sv, _ = np.linalg.svd(W, full_matrices=False)
    keep = sv > (tol * (sv[0] if len(sv) else 1.0))
    return U[:, keep]


def _absorb(basis, z):
    """Residualise z on the fixed-effect design (Frisch-Waugh-Lovell)."""
    return z - basis @ (basis.T @ z)


def estimate_did(panel: pd.DataFrame, *, treated_cells: list[str], t0: date,
                 outcome: str = "outcome",
                 absorb_line_by_day: bool = True) -> DidResult:
    """Two-way FE DID with exact randomisation inference.

    The second fixed effect is product-line x day, not day.  A competitor
    promotion that hits one product line in every region is a line-by-day shock;
    plain day effects would leave it in the error term and it would contaminate
    the treated coefficient.  Absorbing line x day is what makes this estimate
    robust to the concurrent national shocks in this scenario.

    Implementation note: the fixed effects are absorbed once via an orthonormal
    basis of the FE design, then Frisch-Waugh-Lovell reduces the treatment
    coefficient and its cluster-robust variance to one-dimensional algebra.
    That is what makes exhaustive randomisation inference affordable - every
    possible assignment is evaluated, not a sample of them.
    """
    df = panel.copy()
    df = df[df[outcome].notna()].reset_index(drop=True)
    df["treated_unit"] = df["cell"].isin(treated_cells).astype(float)
    df["post"] = (df["date"] >= t0).astype(float)

    cell_D, cell_names = _dummies(df["cell"].to_numpy())
    if absorb_line_by_day and "product_line" in df.columns:
        fe_key = (df["product_line"].astype(str) + "@" + df["date"].astype(str)).to_numpy()
        fe_label = "cell FE + (product_line x day) FE"
    else:
        fe_key = df["date"].astype(str).to_numpy()
        fe_label = "cell FE + day FE"
    time_D, _ = _dummies(fe_key)

    W = np.column_stack([np.ones(len(df)), cell_D, time_D])
    basis = _orthonormal_basis(W)
    k_fe = basis.shape[1]

    y = df[outcome].to_numpy(dtype=float)
    post = df["post"].to_numpy(dtype=float)
    cells_col = df["cell"].to_numpy()
    y_t = _absorb(basis, y)

    def fwl(assignment):
        d = np.isin(cells_col, list(assignment)).astype(float) * post
        d_t = _absorb(basis, d)
        dd_local = float(d_t @ d_t)
        if dd_local <= 1e-12:
            return float("nan"), d_t, y_t
        beta_local = float(d_t @ y_t / dd_local)
        return beta_local, d_t, y_t - beta_local * d_t

    b, d_t, u = fwl(set(treated_cells))
    dd = float(d_t @ d_t)

    groups = list(pd.unique(cells_col))
    g = len(groups)
    n = len(df)
    k_total = k_fe + 1
    meat = 0.0
    for gid in groups:
        m = cells_col == gid
        meat += float(d_t[m] @ u[m]) ** 2
    adj = (g / max(g - 1, 1)) * ((n - 1) / max(n - k_total, 1))
    var = adj * meat / (dd ** 2) if dd > 0 else float("nan")
    se = float(np.sqrt(var))
    dof = max(g - 1, 1)
    t_stat = b / se if se > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(t_stat), dof))
    crit = float(stats.t.ppf(0.975, dof))
    lo, hi = b - crit * se, b + crit * se

    treated_base = float(df[(df["treated_unit"] == 1) & (df["post"] == 0)][outcome].mean())
    att_pct = 100.0 * b / treated_base if treated_base else float("nan")

    # ---- exact randomisation inference over every assignment of equal size ---
    cells = sorted(groups)
    k_treated = len(treated_cells)
    assignments = list(combinations(cells, k_treated))
    null_stats = np.array([abs(fwl(set(a))[0]) for a in assignments])
    null_stats = null_stats[np.isfinite(null_stats)]
    p_ri = float(np.mean(null_stats >= abs(b))) if len(null_stats) else float("nan")

    # ------------------------- parallel-trends pre-test ---------------------
    pre = df[df["post"] == 0].copy().reset_index(drop=True)
    pt = {"tested": False, "note": "too few pre-period observations to test"}
    if len(pre) > 30:
        tvals = (pd.to_datetime(pre["date"]) -
                 pd.to_datetime(pre["date"]).min()).dt.days.to_numpy(dtype=float)
        inter = tvals * pre["treated_unit"].to_numpy(dtype=float)
        cD2, _ = _dummies(pre["cell"].to_numpy())
        key2 = ((pre["product_line"].astype(str) + "@" + pre["date"].astype(str)).to_numpy()
                if absorb_line_by_day and "product_line" in pre.columns
                else pre["date"].astype(str).to_numpy())
        tD2, _ = _dummies(key2)
        W2 = np.column_stack([np.ones(len(pre)), cD2, tD2])
        basis2 = _orthonormal_basis(W2)
        yp = _absorb(basis2, pre[outcome].to_numpy(dtype=float))
        xp = _absorb(basis2, inter)
        dd2 = float(xp @ xp)
        if dd2 > 1e-12:
            bb = float(xp @ yp / dd2)
            up = yp - bb * xp
            cl = pre["cell"].to_numpy()
            gs = list(pd.unique(cl))
            meat2 = sum(float(xp[cl == q] @ up[cl == q]) ** 2 for q in gs)
            adj2 = ((len(gs) / max(len(gs) - 1, 1))
                    * ((len(pre) - 1) / max(len(pre) - basis2.shape[1] - 1, 1)))
            se2 = float(np.sqrt(adj2 * meat2 / dd2 ** 2))
            pp = (float(2 * stats.t.sf(abs(bb / se2), max(len(gs) - 1, 1)))
                  if se2 > 0 else float("nan"))
            daily_pct = 100.0 * bb / treated_base if treated_base else float("nan")
            pt = {"tested": True, "test": "treated x linear pre-trend interaction",
                  "coefficient_units_per_day": round(bb, 5),
                  "coefficient_pct_of_base_per_day": round(daily_pct, 5),
                  "se": round(se2, 5), "p_value": round(pp, 5), "alpha": 0.10,
                  "verdict": "COMPATIBLE" if pp > 0.10 else "INCOMPATIBLE",
                  "note": ("A pre-test that fails to reject is not proof of parallel "
                           "trends; it is the absence of detectable divergence at "
                           "this sample size.")}

    warns = []
    if g < 20:
        warns.append(
            "only %d clusters: cluster-robust asymptotics are unreliable, so the "
            "exact randomisation-inference p-value is reported as primary" % g)
    if pt.get("verdict") == "INCOMPATIBLE":
        warns.append("pre-period trends diverge: the parallel-trends assumption is "
                     "contradicted by the data and DID is not a valid design here")

    ss_tot = float(np.sum((y_t - y_t.mean()) ** 2))
    r2_within = 1.0 - float(u @ u) / ss_tot if ss_tot > 0 else float("nan")

    return DidResult(
        estimator="DID (two-way fixed effects, FWL-absorbed)", outcome=outcome,
        treated_cells=sorted(treated_cells),
        control_cells=[c for c in cells if c not in treated_cells],
        t0=t0, att_per_unit_day=b, att_pct_of_treated_base=att_pct,
        se_cluster=se, ci95=(lo, hi), p_cluster=p, p_randomisation=p_ri,
        n_permutations=len(assignments), n_clusters=g, n_obs=n,
        pre_periods=int(df[df["post"] == 0]["date"].nunique()),
        post_periods=int(df[df["post"] == 1]["date"].nunique()),
        parallel_trends=pt, fixed_effects=fe_label,
        diagnostics={"within_r2": round(r2_within, 4),
                     "treated_pre_mean": round(treated_base, 3),
                     "fe_rank_absorbed": int(k_fe),
                     "inference": ("exact randomisation inference over %d possible "
                                   "assignments of %d treated cells among %d"
                                   % (len(assignments), k_treated, len(cells)))},
        warnings=warns)


# ------------------------------------------------------------------------ ITS
@dataclass
class ItsResult:
    estimator: str
    outcome: str
    t0: date
    level_change: float
    level_se: float
    level_p: float
    level_ci95: tuple[float, float]
    slope_change: float
    slope_se: float
    slope_p: float
    cumulative_effect: float
    cumulative_pct: float
    pre_points: int
    post_points: int
    hac_lags: int
    diagnostics: dict
    placebo: dict
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["t0"] = self.t0.isoformat()
        d["level_ci95"] = [round(float(x), 4) for x in self.level_ci95]
        for k in ("level_change", "level_se", "level_p", "slope_change", "slope_se",
                  "slope_p", "cumulative_effect", "cumulative_pct"):
            d[k] = round(float(d[k]), 6)
        return d


def _its_design(dates: list[date], t0: date, holiday: np.ndarray,
                promo: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n = len(dates)
    t = np.arange(n, dtype=float)
    D = np.array([1.0 if d >= t0 else 0.0 for d in dates])
    i0 = int(np.argmax(D)) if D.any() else n
    since = np.maximum(t - i0, 0.0) * D
    F = fourier_terms(t, period=7.0, harmonics=2)
    X = np.column_stack([np.ones(n), t, D, since, F, holiday, promo])
    names = (["const", "trend", "level_change", "slope_change",
              "sin7_1", "cos7_1", "sin7_2", "cos7_2", "holiday", "promo"])
    return X, names


def estimate_its(y: np.ndarray, dates: list[date], t0: date, *,
                 holiday: np.ndarray, promo: np.ndarray,
                 placebo_offset_days: int = 21) -> ItsResult:
    y = np.asarray(y, dtype=float)
    X, names = _its_design(dates, t0, holiday, promo)
    fit = fit_ols(X, y, names)
    L = int(np.floor(4 * (len(y) / 100.0) ** (2 / 9)))
    b2, se2, p2 = fit.t_and_p(2, "hac", hac_lags=L)
    lo, hi = fit.ci(2, "hac", hac_lags=L)
    b3, se3, p3 = fit.t_and_p(3, "hac", hac_lags=L)

    post_idx = [i for i, d in enumerate(dates) if d >= t0]
    n_post = len(post_idx)
    cum = sum(b2 + b3 * j for j in range(n_post))
    pre_mean = float(np.nanmean([v for v, d in zip(y, dates) if d < t0]))
    cum_pct = 100.0 * cum / (pre_mean * n_post) if pre_mean and n_post else np.nan

    dw = durbin_watson(fit.resid)
    lb_q, lb_p = ljung_box(fit.resid, lags=10)

    # ---------------------------- placebo -----------------------------------
    placebo: dict = {"run": False}
    pre_dates = [d for d in dates if d < t0]
    if len(pre_dates) >= 40:
        fake_t0 = t0 - timedelta(days=placebo_offset_days)
        keep = [i for i, d in enumerate(dates) if d < t0]
        Xp, _ = _its_design([dates[i] for i in keep], fake_t0,
                            holiday[keep], promo[keep])
        fp = fit_ols(Xp, y[keep], names)
        Lp = int(np.floor(4 * (len(keep) / 100.0) ** (2 / 9)))
        pb, pse, ppv = fp.t_and_p(2, "hac", hac_lags=Lp)
        placebo = {
            "run": True, "pseudo_t0": fake_t0.isoformat(),
            "level_change": round(float(pb), 4), "se": round(float(pse), 4),
            "p_value": round(float(ppv), 5), "alpha": 0.10,
            "verdict": "PASS" if ppv > 0.10 else "FAIL",
            "note": ("A placebo intervention placed inside the pre-period must NOT "
                     "produce a significant level change. A FAIL means the model "
                     "detects breaks that are not there and the real estimate cannot "
                     "be trusted."),
        }

    warns = []
    if dw < 1.3 or dw > 2.7:
        warns.append(f"Durbin-Watson {dw:.2f} indicates residual autocorrelation; "
                     "HAC standard errors are used but the point estimate may still "
                     "be sensitive to the trend specification")
    if lb_p < 0.05:
        warns.append(f"Ljung-Box p={lb_p:.4f}: residuals are not white noise")
    if placebo.get("verdict") == "FAIL":
        warns.append("placebo test FAILED: this ITS design is not credible on this series")

    return ItsResult(
        estimator="ITS (level + slope change, Fourier weekly, HAC SEs)",
        outcome="series", t0=t0, level_change=float(b2), level_se=float(se2),
        level_p=float(p2), level_ci95=(float(lo), float(hi)),
        slope_change=float(b3), slope_se=float(se3), slope_p=float(p3),
        cumulative_effect=float(cum), cumulative_pct=float(cum_pct),
        pre_points=len(pre_dates), post_points=n_post, hac_lags=L,
        diagnostics={"r2": round(fit.r2, 4), "durbin_watson": round(dw, 3),
                     "ljung_box_q": round(lb_q, 3), "ljung_box_p": round(lb_p, 5),
                     "pre_period_mean": round(pre_mean, 3),
                     "hac_kernel": "Bartlett", "hac_lags": L},
        placebo=placebo, warnings=warns)
