# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC
from __future__ import annotations
import numpy as np
import pandas as pd
import math
from typing import Dict, Sequence, Optional

# ----- Helpers shared with the previous module's aggregation idea -----

def aggregate_to_lots(df: pd.DataFrame, id_col: str = "Inspection_ID", action_col: str = "action") -> pd.DataFrame:
    grouped = df.groupby(id_col)[action_col]
    n = grouped.size().rename("n")
    r = (grouped.sum() > 0).astype(int).rename("r")
    return pd.concat([n, r], axis=1).reset_index()

# ----- Cloglog (Poisson-rate) model -----
# Model: P(reject_i | n_i, theta) = 1 - exp(- n_i * theta)
# where theta = epsilon * gamma  (epsilon = detection rate, gamma = per-unit contamination rate).
# If linear predictor is eta_i = log(n_i) + lambda, then exp(lambda) = theta.
# The "mean contamination rate" per unit under this model is r = 1 - exp(-theta) = 1 - exp(-exp(lambda)).

def _loglik_theta(theta: float, n: np.ndarray, r: np.ndarray) -> float:
    if theta <= 0:
        return -np.inf
    # p_i = 1 - exp(- n_i * theta)
    nt = n * theta
    # stability
    exp_neg = np.exp(-np.clip(nt, 0, 700))
    p = 1.0 - exp_neg
    p = np.clip(p, 1e-300, 1-1e-15)
    ll = r * np.log(p) + (1 - r) * (-nt)  # since log(1-p) = -n_i * theta
    return float(np.sum(ll))

def _dloglik_theta(theta: float, n: np.ndarray, r: np.ndarray) -> float:
    # derivative of sum[ r_i * log(1 - e^{-n_i theta}) - (1 - r_i) n_i theta ]
    if theta <= 0:
        theta = 1e-12
    nt = n * theta
    exp_neg = np.exp(-np.clip(nt, 0, 700))
    denom = 1.0 - exp_neg
    denom = np.clip(denom, 1e-300, np.inf)
    term_reject = r * (n * exp_neg / denom)
    term_accept = -(1 - r) * n
    return float(np.sum(term_reject + term_accept))

def _mle_theta(n: np.ndarray, r: np.ndarray) -> float:
    # Find root of derivative; bracket search on a reasonable interval
    # Start with bounds based on crude rates
    # Avoid theta too huge; use grid + refine
    lo, hi = 1e-12, 10.0 / max(n.max(), 1.0)
    # widen upper bound if many rejections
    if r.mean() > 0.1:
        hi = max(hi, 1.0 / max(n.mean(), 1.0))
    # Try bisection on derivative sign change
    f_lo, f_hi = _dloglik_theta(lo, n, r), _dloglik_theta(hi, n, r)
    if not np.isnan(f_lo) and not np.isnan(f_hi) and f_lo > 0 and f_hi < 0:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f_mid = _dloglik_theta(mid, n, r)
            if abs(f_mid) < 1e-10 or (hi - lo) < 1e-12:
                return mid
            if f_mid > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    # Fall back: grid search then local refine
    grid = np.geomspace(1e-8, 10.0 / max(n.max(), 1.0), 4000)
    ll = np.array([_loglik_theta(t, n, r) for t in grid])
    t0 = grid[int(np.nanargmax(ll))]
    # local refine via golden-section
    a = max(1e-12, t0/10)
    b = t0
    c = min(10.0 / max(n.max(), 1.0), t0*10)
    phi = (1 + 5 ** 0.5) / 2
    invphi = 1/phi
    invphi2 = 1/phi**2
    x1 = b - (b - a) * invphi
    x2 = a + (b - a) * invphi
    f1 = _loglik_theta(x1, n, r)
    f2 = _loglik_theta(x2, n, r)
    for _ in range(120):
        if f1 < f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (b - a) * invphi
            f2 = _loglik_theta(x2, n, r)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - (b - a) * invphi
            f1 = _loglik_theta(x1, n, r)
        if abs(b - a) < 1e-12:
            break
    return 0.5 * (a + b)

def fit_cloglog_intercept_mle(df: pd.DataFrame, detection_rate: float,
                              id_col: str = "INSPECTION_ID", action_col: str = "action") -> Dict[str, float]:
    """
    Fits the cloglog model with offset log(n_i):
      P(reject_i) = 1 - exp(- n_i * theta),  theta = epsilon * gamma
    Returns:
      - theta_hat (per-unit detection-adjusted rate)
      - lambda_hat (intercept; exp(lambda_hat)=theta_hat)
      - mean_r = 1 - exp(-theta_hat)  (inverse cloglog of intercept)
      - se_theta (via observed info), and Wald CI for mean_r
    """
    lots = aggregate_to_lots(df, id_col=id_col, action_col=action_col)
    n = lots["n"].to_numpy(dtype=float)
    r = lots["r"].to_numpy(dtype=float)
    theta_hat = _mle_theta(n, r)
    lam_hat = math.log(theta_hat)
    # mean contamination rate per unit under this model
    mean_r = 1.0 - math.exp(-theta_hat)  # = 1 - exp(-exp(lambda))
    # observed information for theta (numeric second derivative)
    h = max(1e-6, theta_hat * 1e-4)
    l1 = _loglik_theta(theta_hat - h, n, r)
    l2 = _loglik_theta(theta_hat, n, r)
    l3 = _loglik_theta(theta_hat + h, n, r)
    second = (l3 - 2*l2 + l1) / (h*h)
    if second >= 0:
        se_theta = float("nan")
    else:
        var = 1.0 / (-second)
        se_theta = float(max(0.0, var)) ** 0.5
    # Delta method to get SE for mean_r = 1 - exp(-theta)
    # d( mean_r )/d theta = exp(-theta)
    se_mean_r = math.exp(-theta_hat) * se_theta if np.isfinite(se_theta) else float("nan")
    z = 1.959963984540054
    ci_mean_r = (max(0.0, mean_r - z*se_mean_r), min(1.0, mean_r + z*se_mean_r))
    return {
        "theta_hat": float(theta_hat),
        "lambda_hat": float(lam_hat),
        "mean_r": float(mean_r),
        "se_theta": float(se_theta),
        "se_mean_r": float(se_mean_r),
        "ci95_mean_r_lo": float(ci_mean_r[0]),
        "ci95_mean_r_hi": float(ci_mean_r[1]),
        "lots": lots,
        "detection_rate": float(detection_rate)
    }

# ----- Calibrate Beta(alpha, beta) around mean_r to match target failure rate -----

def _simulate_fail_rate(lot_sizes: np.ndarray, alpha: float, beta: float, detection_rate: float, rng: np.random.Generator) -> float:
    # Draw per-consignment per-unit contamination rate gamma ~ Beta(alpha, beta)
    # Probability consignment fails given gamma: 1 - (1 - epsilon*gamma)^{n}
    # (exact binomial model; could use exp(-epsilon*gamma*n) as an approx)
    gammas = rng.beta(alpha, beta, size=lot_sizes.shape[0])
    eps = detection_rate
    p_fail = 1.0 - np.power(1.0 - np.clip(eps * gammas, 0.0, 1.0 - 1e-15), lot_sizes)
    return float(np.mean(p_fail))

def calibrate_beta_to_failrate(lot_sizes: Sequence[int], mean_r: float, detection_rate: float,
                               target_fail_rate: float, sims: int = 50000, tol: float = 1e-4,
                               maxiter: int = 40, seed: Optional[int] = 42, use_exact: bool = True) -> Dict[str, float]:
    """
    Given mean_r (from the cloglog intercept), choose Beta(alpha,beta) parameters so that:
      - E[gamma] = mean_r_effective where mean_r_effective solves mean_r = 1 - exp(-epsilon*gamma_mean)
      - Simulated overall failure rate matches target_fail_rate for the provided lot size distribution.

    We parameterize Beta via (mean, concentration s): alpha = m*s, beta = (1-m)*s.
    We search over s using bisection on the difference between simulated and target failure rate.
    """
    rng = np.random.default_rng(seed)
    lot_sizes = np.asarray(lot_sizes, dtype=int)
    eps = detection_rate

    # Convert mean_r (per-unit mean failure prob under Poisson model) back to mean per-unit gamma
    # mean_r = 1 - exp(-eps * gamma_mean)  => gamma_mean = -log(1 - mean_r) / eps
    gamma_mean = -math.log(max(1e-15, 1.0 - mean_r)) / eps
    # ensure m in (0,1)
    m = min(max(gamma_mean, 1e-12), 1.0 - 1e-12)

    # Bracket concentration s to hit the target
    s_lo, s_hi = 1e-2, 1e6
    def fail_at(s):
        a, b = m * s, (1 - m) * s
        return _simulate_fail_rate(lot_sizes, a, b, eps, rng)

    f_lo = fail_at(s_lo)
    f_hi = fail_at(s_hi)
    # Ensure the target is bracketed; if not, clamp to nearest
    if not (min(f_lo, f_hi) <= target_fail_rate <= max(f_lo, f_hi)):
        # pick the closer endpoint
        s_best = s_lo if abs(f_lo - target_fail_rate) < abs(f_hi - target_fail_rate) else s_hi
        alpha, beta = m * s_best, (1 - m) * s_best
        return {"alpha": float(alpha), "beta": float(beta), "concentration": float(s_best),
                "achieved_fail_rate": float(fail_at(s_best)), "bracketed": False}

    # Bisection on s
    s_left, s_right = s_lo, s_hi
    fl, fr = f_lo, f_hi
    for _ in range(maxiter):
        s_mid = math.sqrt(s_left * s_right)  # multiplicative bisection
        fm = fail_at(s_mid)
        if abs(fm - target_fail_rate) < tol:
            s_star = s_mid
            break
        # Decide which side to keep; monotonicity may be data-dependent, so choose side by closeness
        if (fl - target_fail_rate) * (fm - target_fail_rate) <= 0:
            s_right, fr = s_mid, fm
        else:
            s_left, fl = s_mid, fm
    else:
        s_star = s_mid

    alpha = m * s_star
    beta = (1 - m) * s_star
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "concentration": float(s_star),
        "achieved_fail_rate": float(fail_at(s_star)),
        "bracketed": True
    }

def fit_beta_glm_match_failrate(df: pd.DataFrame, detection_rate: float, target_fail_rate: float,
                                id_col: str = "INSPECTION_ID", action_col: str = "action",
                                sims: int = 50000, seed: Optional[int] = 42) -> Dict[str, float]:
    """
    Pipeline:
      1) Aggregate row-level data -> lots (n_i, r_i)
      2) Fit theta_hat via cloglog MLE (offset log n_i)
      3) Convert to mean_r = 1 - exp(-theta_hat)
      4) Calibrate Beta(alpha,beta) around per-unit gamma_mean so simulated consignment
         failure rate matches target_fail_rate for the observed lot sizes.
    """
    lots = aggregate_to_lots(df, id_col=id_col, action_col=action_col)
    n = lots["n"].to_numpy(dtype=int)

    # Step 2 & 3
    theta_hat = _mle_theta(n.astype(float), lots["r"].to_numpy(dtype=float))
    mean_r = 1.0 - math.exp(-theta_hat)

    # Step 4
    calib = calibrate_beta_to_failrate(n, mean_r, detection_rate, target_fail_rate, sims=sims, seed=seed)

    out = {
        "mean_r": float(mean_r),
        "theta_hat": float(theta_hat),
        **calib,
        "target_fail_rate": float(target_fail_rate),
        "detection_rate": float(detection_rate),
        "lots": lots,
    }
    return out
