# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

# pip install statsmodels pandas numpy
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass
from typing import Optional, Tuple
import math


def beta_from_mean_sd(mean, sd, eps=1e-12):
    # clip mean to (0,1) to avoid boundary issues
    m = min(max(mean, eps), 1.0 - eps)
    vmax = m * (1.0 - m)  # max possible variance for a Beta (as K -> 0+)
    v = sd * sd
    if v >= vmax:
        # soften to a feasible value
        v = 0.999 * vmax
    K = (m * (1.0 - m)) / v - 1.0
    alpha = m * K
    beta = (1.0 - m) * K
    return float(alpha), float(beta), float(K)

def beta_from_mean_ci(mean, lower, upper, z=1.959963984540054):
    # infer sd from a symmetric normal approx to the CI, then call above
    sd = (upper - lower) / (2.0 * z)
    return beta_from_mean_sd(mean, sd)

def beta_from_mean_K(mean, K, eps=1e-12):
    m = min(max(mean, eps), 1.0 - eps)
    if K <= 0:
        raise ValueError("K must be > 0 for a Beta distribution.")
    alpha = m * K
    beta = (1.0 - m) * K
    # implied sd if you're curious:
    sd = math.sqrt(m * (1.0 - m) / (K + 1.0))
    return float(alpha), float(beta), float(sd)



@dataclass
class CloglogGLMResult:
    alpha_hat: float                 # intercept on cloglog scale
    alpha_se: float                  # std error of alpha
    alpha_ci: Tuple[float, float]    # 95% CI for alpha
    mean_r: float                    # mean contamination rate = 1 - exp(-exp(alpha))
    mean_r_se: float                 # delta-method SE for mean_r
    mean_r_ci: Tuple[float, float]   # 95% CI for mean_r
    theta_hat: float                 # = exp(alpha_hat)
    lots: pd.DataFrame               # aggregated (Inspection_ID, n, r)
    detection_rate: Optional[float]  # epsilon, if provided
    gamma_mean: Optional[float]      # per-unit mean contamination (if epsilon provided)

def aggregate_to_lots(df: pd.DataFrame,
                      id_col: str = "Inspection_ID",
                      action_col: str = "action") -> pd.DataFrame:
    """
    Collapse row-level inspection-unit data into lot-level counts.
    n_i = number of rows for an Inspection_ID
    r_i = 1 if ANY action==1 for that ID, else 0
    """
    g = df.groupby(id_col)[action_col]
    n = g.size().rename("n")
    r = (g.sum() > 0).astype(int).rename("r")
    return pd.concat([n, r], axis=1).reset_index()

def fit_cloglog_statsmodels(df: pd.DataFrame,
                            id_col: str = "INSPECTION_ID",
                            action_col: str = "action",
                            detection_rate: Optional[float] = None) -> CloglogGLMResult:
    """
    Direct GLM fit via statsmodels:
        r_i ~ Bernoulli(p_i),  cloglog(p_i) = alpha + log(n_i)  (offset)
      => p_i = 1 - exp(-n_i * exp(alpha)) with alpha = log(theta)
    """
    lots = aggregate_to_lots(df, id_col=id_col, action_col=action_col)
    n = lots["n"].to_numpy(dtype=float)
    r = lots["r"].to_numpy(dtype=float)

    # Design: intercept-only model; offset = log(n_i)
    X = np.ones((len(lots), 1))                    # intercept column
    offset = np.log(np.clip(n, 1.0, None))        # guard log(0)

    family = sm.families.Binomial(link=sm.families.links.CLogLog())
    model = sm.GLM(r, X, family=family, offset=offset)

    # Fit with a couple of robust fallbacks
    try:
        res = model.fit()
    except Exception:
        res = model.fit(method="bfgs", maxiter=1000, disp=False)

    alpha_hat = float(res.params[0])
    cov = res.cov_params()
    alpha_var = float(cov[0, 0])
    alpha_se = alpha_var ** 0.5 if np.isfinite(alpha_var) and alpha_var >= 0 else np.nan

    # Mean contamination rate per-unit implied by intercept: mean_r = 1 - exp(-exp(alpha))
    mean_r = 1.0 - np.exp(-np.exp(alpha_hat))

    # Delta method for SE of mean_r:
    # d/dalpha [1 - exp(-exp(alpha))] = exp(alpha) * exp(-exp(alpha)) = exp(alpha - exp(alpha))
    d_meanr_d_alpha = np.exp(alpha_hat - np.exp(alpha_hat))
    mean_r_se = float(abs(d_meanr_d_alpha) * alpha_se) if np.isfinite(alpha_se) else np.nan

    z = 1.959963984540054
    alpha_ci = (alpha_hat - z * alpha_se, alpha_hat + z * alpha_se) if np.isfinite(alpha_se) else (np.nan, np.nan)
    mean_r_ci = (max(0.0, mean_r - z * mean_r_se), min(1.0, mean_r + z * mean_r_se)) if np.isfinite(mean_r_se) else (np.nan, np.nan)

    theta_hat = float(np.exp(alpha_hat))

    gamma_mean = None
    if detection_rate is not None:
        eps = float(detection_rate)
        # mean_r = 1 - exp(-eps * gamma_mean)  =>  gamma_mean = -log(1 - mean_r) / eps
        gamma_mean = float(-np.log(max(1e-15, 1.0 - mean_r)) / eps)

    return CloglogGLMResult(
        alpha_hat=alpha_hat,
        alpha_se=alpha_se,
        alpha_ci=alpha_ci,
        mean_r=float(mean_r),
        mean_r_se=float(mean_r_se),
        mean_r_ci=(float(mean_r_ci[0]), float(mean_r_ci[1])),
        theta_hat=theta_hat,
        lots=lots,
        detection_rate=detection_rate,
        gamma_mean=gamma_mean
    )

# -----------------------
# Example usage:
if __name__ == "__main__":
    # tiny toy dataset: two consignments, one fail
    df = pd.DataFrame({
        "Inspection_ID": ["A"]*10 + ["B"]*12,
        "action":        [0]*10  + [0]*11 + [1],
    })
    res = fit_cloglog_statsmodels(df, detection_rate=0.8)
    print("alpha_hat:", res.alpha_hat)
    print("alpha_se:", res.alpha_se, "  95% CI:", res.alpha_ci)
    print("mean_r  :", res.mean_r, "  95% CI:", res.mean_r_ci)
    print("theta_hat (exp(alpha)):", res.theta_hat)
    print("gamma_mean (if eps given):", res.gamma_mean)
    print("\nLots used:\n", res.lots.head())
    print('')
