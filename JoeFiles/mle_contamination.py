# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC
from __future__ import annotations
from typing import Sequence, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import math

# ---------- Likelihood pieces for the pass/fail model ----------

def _loglik_gamma(gamma: float, n: np.ndarray, r: np.ndarray, e: float) -> float:
    """
    Log-likelihood L(gamma) for pass/fail inspection data, given detection rate e.
    n: array of sample sizes per lot (n_i >= 1)
    r: array of indicators (0 = accept, 1 = reject) per lot
    e: detection rate in (0, 1]; scalar modifier on γ
    Domain: gamma in [0, 1/e)
    """
    if gamma < 0 or gamma >= 1.0 / e:
        return -np.inf
    one_minus_ge = 1.0 - gamma * e
    one_minus_ge = np.clip(one_minus_ge, 1e-300, 1.0)
    # accept terms
    term_accept = (1 - r) * n * np.log(one_minus_ge)
    # reject terms
    pow_term = np.power(one_minus_ge, n)
    inner = 1.0 - pow_term
    inner = np.clip(inner, 1e-300, 1.0)  # avoid log(0)
    term_reject = r * np.log(inner)
    return float(np.sum(term_accept + term_reject))

def _dloglik_gamma(gamma: float, n: np.ndarray, r: np.ndarray, e: float) -> float:
    """
    First derivative L'(gamma) for root finding.
    """
    # keep strictly inside domain
    if gamma <= 0:
        gamma = 0.0 + 1e-12
    if gamma >= 1.0 / e:
        gamma = (1.0 / e) - 1e-12
    one_minus_ge = 1.0 - gamma * e
    part_accept = (1 - r) * n * (-e) / one_minus_ge
    pow_nm1 = np.power(one_minus_ge, np.maximum(n - 1, 0))
    pow_n = np.power(one_minus_ge, n)
    denom = 1.0 - pow_n
    denom = np.clip(denom, 1e-300, np.inf)
    part_reject = r * e * n * pow_nm1 / denom
    return float(np.sum(part_accept + part_reject))

def _bracket_interval(n: np.ndarray, r: np.ndarray, e: float) -> Tuple[float, float]:
    """Safe search bracket inside [0, 1/e)."""
    lo = 0.0 + 1e-10
    hi = (1.0 / e) - 1e-10
    return lo, hi

def _grid_max(n: np.ndarray, r: np.ndarray, e: float) -> float:
    lo, hi = _bracket_interval(n, r, e)
    grid1 = np.linspace(lo, hi, 2000)
    ll = np.array([_loglik_gamma(g, n, r, e) for g in grid1])
    idx = int(np.nanargmax(ll))
    g_star = grid1[idx]
    # local refine
    w = min(1e-2, max(1e-6, (hi - lo) / 200))
    lo2 = max(lo, g_star - w)
    hi2 = min(hi, g_star + w)
    grid2 = np.linspace(lo2, hi2, 2000)
    ll2 = np.array([_loglik_gamma(g, n, r, e) for g in grid2])
    return float(grid2[int(np.nanargmax(ll2))])

def _bisect_root(n: np.ndarray, r: np.ndarray, e: float,
                 tol: float = 1e-10, maxiter: int = 200) -> float:
    """
    Solve L'(γ) = 0 on (0, 1/e) via bisection if sign change exists.
    Falls back to grid search if derivative does not bracket a root.
    """
    lo, hi = _bracket_interval(n, r, e)
    f_lo = _dloglik_gamma(lo, n, r, e)
    f_hi = _dloglik_gamma(hi, n, r, e)
    if np.isnan(f_lo) or np.isnan(f_hi):
        return _grid_max(n, r, e)
    if f_lo > 0 and f_hi < 0:
        for _ in range(maxiter):
            mid = 0.5 * (lo + hi)
            f_mid = _dloglik_gamma(mid, n, r, e)
            if abs(f_mid) < tol or (hi - lo) < tol:
                return mid
            if f_mid > 0:
                lo, f_lo = mid, f_mid
            else:
                hi, f_hi = mid, f_mid
        return 0.5 * (lo + hi)
    return _grid_max(n, r, e)

def mle_gamma_from_lots(n: Sequence[int], r: Sequence[int], e: float) -> Dict[str, float]:
    """
    MLE of γ for given detection rate e from lot-level data.
    n: list/array of lot sample sizes n_i
    r: list/array of lot indicators (0 = accepted, 1 = rejected)
    e: detection rate in (0,1]
    Returns dict with gamma_hat, loglik_hat, std_err (obs. information), and Wald 95% CI.
    """
    n = np.asarray(n, dtype=float)
    r = np.asarray(r, dtype=float)
    if n.shape != r.shape:
        raise ValueError("n and r must have the same length")
    if not (0 < e <= 1):
        raise ValueError("Detection rate e must be in (0, 1].")
    gamma_hat = _bisect_root(n, r, e)
    ll_hat = _loglik_gamma(gamma_hat, n, r, e)
    # numeric second deriv
    h = max(1e-6, gamma_hat * 1e-4)
    lo, hi = _bracket_interval(n, r, e)
    gmh = max(lo + 1e-12, gamma_hat - h)
    gph = min(hi - 1e-12, gamma_hat + h)
    l1 = _loglik_gamma(gmh, n, r, e)
    l2 = _loglik_gamma(gamma_hat, n, r, e)
    l3 = _loglik_gamma(gph, n, r, e)
    second = (l3 - 2 * l2 + l1) / (h ** 2)
    if second >= 0:
        std_err = float("nan")
        var = float("nan")
    else:
        var = 1.0 / (-second)
        std_err = float(math.sqrt(max(var, 0.0)))
    z = 1.959963984540054
    ci_lo = max(lo, gamma_hat - z * std_err) if not math.isnan(std_err) else float("nan")
    ci_hi = min(hi, gamma_hat + z * std_err) if not math.isnan(std_err) else float("nan")
    return {
        "gamma_hat": float(gamma_hat),
        "loglik_hat": float(ll_hat),
        "std_err": float(std_err),
        "var_hat": float(var),
        "ci95_lo": float(ci_lo),
        "ci95_hi": float(ci_hi),
    }

# ---------- Data wrangling from row-level inspection-unit data ----------

def aggregate_to_lots(df: pd.DataFrame, id_col: str = "Inspection_ID", action_col: str = "action") -> pd.DataFrame:
    """
    Collapses row-level inspection-unit data into lot-level (n_i, r_i) by unique Inspection_ID.
    n_i = number of rows for that ID (sample size)
    r_i = 1 if any action=1 for that ID, else 0
    """
    if id_col not in df.columns or action_col not in df.columns:
        raise KeyError(f"DataFrame must include columns '{id_col}' and '{action_col}'")
    grouped = df.groupby(id_col)[action_col]
    n = grouped.size().rename("n")
    r = (grouped.sum() > 0).astype(int).rename("r")
    out = pd.concat([n, r], axis=1).reset_index()
    return out  # columns: Inspection_ID, n, r

# ---------- Fit Beta(alpha, beta) to the MLE via moment matching ----------

def beta_from_mle(gamma_hat: float, var_hat: float, min_k: float = 1e-6) -> Dict[str, float]:
    """
    Fit Beta(alpha, beta) whose mean and variance match the MLE mean (gamma_hat)
    and its asymptotic variance (var_hat). If var_hat is tiny/invalid, falls back
    to a minimum concentration to avoid degenerate parameters.
    """
    m = float(gamma_hat)
    if not (0 < m < 1):
        # keep within (0,1) open interval
        m = min(max(m, 1e-12), 1.0 - 1e-12)
    if not (var_hat is not None) or not np.isfinite(var_hat) or var_hat <= 0:
        # fallback: pick a modest concentration so CI isn't absurdly tight
        k = 1.0 / min_k
        alpha = m * k
        beta = (1 - m) * k
        return {"alpha": float(alpha), "beta": float(beta), "note": "fallback_k"}
    # For Beta(a,b): mean = a/(a+b)=m, var = m(1-m)/(a+b+1)
    s = (m * (1 - m) / var_hat) - 1.0
    # guard
    s = max(s, 1e-6)
    alpha = m * s
    beta = (1 - m) * s
    return {"alpha": float(alpha), "beta": float(beta), "note": "moment_match"}

# ---------- One-stop function ----------

def fit_beta_from_passfail(df: pd.DataFrame, detection_rate: float,
                           id_col: str = "INSPECTION_ID", action_col: str = "action") -> Dict[str, float]:
    """
    Given row-level pass/fail inspection-unit data with columns:
      - id_col: shipment/consignment ID (multiple rows per lot)
      - action_col: 0/1 indicator if that unit had a contaminant found
    and a fixed detection_rate e, compute:
      - MLE gamma for the population contamination rate
      - Asymptotic SE and 95% CI
      - Fitted Beta(alpha, beta) parameters that match (mean, variance)
    Returns a dict with all results plus the aggregated lot-level table.
    """
    lots = aggregate_to_lots(df, id_col=id_col, action_col=action_col)
    res = mle_gamma_from_lots(lots["n"].to_numpy(), lots["r"].to_numpy(), detection_rate)
    beta_params = beta_from_mle(res["gamma_hat"], res["var_hat"])
    out = {**res, **beta_params}
    out["lots"] = lots
    out["detection_rate"] = detection_rate
    return out
