# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC


import os
import re
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.optimize import minimize_scalar
from scipy import stats
import pandas as pd


# === CONFIG ===
# Your R script (specify path to R script)
R_SCRIPT_PATH_test = r"C:\Users\agorjk1\PycharmProjects\plant-inspection-station-simulation\JoeFiles\clarke_2023_code\add_function.R"

# Beta-Binomial Model

R_SCRIPT_PATH_bb_model_clarke = r"C:\Users\agorjk1\PycharmProjects\plant-inspection-station-simulation\JoeFiles\clarke_2023_code\simstudy_paperspace_27_04_2023.R"
R_SCRIPT_PATH_bb_cli = r"C:\Users\agorjk1\PycharmProjects\plant-inspection-station-simulation\JoeFiles\clarke_2023_code\clarke_bb_model.R"

# Absolute path to conda.exe (adjust for your install if needed)
CONDA_EXE_PATH = r"C:\Users\agorjk1\AppData\Local\anaconda3\Scripts\conda.exe"

# If you installed R via Conda, set your env name here (preferred when using Conda):
CONDA_ENV_NAME: Optional[str] = "rbb"

# Or, if you want to use a specific Rscript.exe:
#RSCRIPT_ABS_PATH: Optional[str] = r"C:\Users\agorjk1\AppData\Local\anaconda3\envs\rstudio\Scripts\Rscript.exe"
#RSCRIPT_ABS_PATH: Optional[str] = r"C:\Users\agorjk1\AppData\Local\anaconda3\envs\rbb\Scripts\Rscript.exe"
RSCRIPT_ABS_PATH: Optional[str] = None

# If both CONDA_ENV_NAME and RSCRIPT_ABS_PATH are None, this will try plain "Rscript" on PATH.


def _pick_rscript_command() -> List[str]:
    """
    Decide how to call Rscript:
      1) conda run -n <env> Rscript (using absolute conda.exe if available)
      2) absolute path to Rscript.exe
      3) Rscript from PATH
    """
    # Option 1: Conda env
    if CONDA_ENV_NAME:
        conda_exe = Path(CONDA_EXE_PATH)
        if conda_exe.exists():
            return [str(conda_exe), "run", "-n", CONDA_ENV_NAME, "Rscript"]
        else:
            # fall back to PATH search
            conda = shutil.which("conda")
            if conda:
                return [conda, "run", "-n", CONDA_ENV_NAME, "Rscript"]
            else:
                raise RuntimeError(
                    f"Conda not found. Looked for {CONDA_EXE_PATH} and on PATH. "
                    "Either install conda, adjust CONDA_EXE_PATH, or unset CONDA_ENV_NAME."
                )

    # Option 2: absolute Rscript.exe
    if RSCRIPT_ABS_PATH:
        if not Path(RSCRIPT_ABS_PATH).exists():
            raise FileNotFoundError(f"Rscript.exe not found at: {RSCRIPT_ABS_PATH}")
        return [RSCRIPT_ABS_PATH]

    # Option 3: Rscript on PATH
    r_on_path = shutil.which("Rscript")
    if r_on_path:
        return [r_on_path]

    raise RuntimeError(
        "Could not find Rscript. Set CONDA_ENV_NAME, or RSCRIPT_ABS_PATH, "
        "or add Rscript to your system PATH."
    )


def _parse_numeric_stdout(stdout: str) -> float:
    """
    Grab the last numeric token from stdout.
    This tolerates benign warnings/messages printed before/after the result.
    """
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", stdout)
    if not nums:
        raise ValueError(f"Could not find a numeric result in R output:\n{stdout}")
    return float(nums[-1])


def _parse_json_from_r_stdout(stdout: str):
    # 1) Try last non-empty line that looks like a JSON object
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            try:
                return json.loads(ln)
            except json.JSONDecodeError:
                pass  # keep looking

    # 2) Fallback: extract the last {...} block from the whole stream
    start = stdout.rfind("{")
    end = stdout.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidate = stdout[start:end+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 3) Nothing worked — raise with context for debugging
    raise ValueError("Expected JSON from R; got:\n" + stdout)


def run_r_add_numbers(a, b) -> float:
    # Validate the R script path early for friendlier errors
    if not Path(R_SCRIPT_PATH_test).exists():
        raise FileNotFoundError(f"R script not found: {R_SCRIPT_PATH_test}")

    cmd = _pick_rscript_command()

    # Call Rscript with the two numbers (your add_function.R should read args and print result)
    proc = subprocess.run(
        cmd + [R_SCRIPT_PATH_test, str(a), str(b)],
        capture_output=True,
        text=True
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "R script failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDERR:\n{proc.stderr}\n"
            f"STDOUT:\n{proc.stdout}"
        )

    return _parse_numeric_stdout(proc.stdout)

def run_clarke_bb_model(ty, b, B, Nbar, freq, theta, R, startval, se) -> float:
    # Validate the R script path early for friendlier errors
    if not Path(R_SCRIPT_PATH_test).exists():
        raise FileNotFoundError(f"R script not found: {R_SCRIPT_PATH_test}")

    cmd = _pick_rscript_command()

    # Call Rscript from Clarke Paper
    proc = subprocess.run(
        cmd + [R_SCRIPT_PATH_bb_model_clarke, ty, b, B, Nbar, freq,theta, R, startval, se],
        capture_output=True,
        text=True
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "R script failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDERR:\n{proc.stderr}\n"
            f"STDOUT:\n{proc.stdout}"
        )

    return _parse_numeric_stdout(proc.stdout)


def run_bb_group_model(
    ty: List[int],
    b: int,
    B: int,
    Nbar: int,
    freq: List[int],
    theta: float,
    R: int,
    startval: List[float],
    se: bool,
) -> Dict[str, Any]:
    if not Path(R_SCRIPT_PATH_bb_cli).exists():
        raise FileNotFoundError(f"R script not found: {R_SCRIPT_PATH_bb_cli}")

    cmd = _pick_rscript_command()

    theta_json = "Inf" if (isinstance(theta, float) and np.isinf(theta)) else float(theta)

    payload = {
        "ty": list(map(int, ty)),
        "b": int(b),
        "B": int(B),
        "Nbar": int(Nbar),
        "freq": list(map(int, freq)),
        "theta": theta_json,
        "R": int(R),
        "startval": list(map(float, startval)),
        "se": bool(se),
    }

    proc = subprocess.run(
        cmd + [R_SCRIPT_PATH_bb_cli, json.dumps(payload)],
        capture_output=True,
        text=True
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "R script failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDERR:\n{proc.stderr}\n"
            f"STDOUT:\n{proc.stdout}"
        )

    # Parse the final line as JSON (ignore startup messages)
    out_line = proc.stdout.strip().splitlines()[-1]
    try:
        #return json.loads(out_line)
        return _parse_json_from_r_stdout(proc.stdout)
    except json.JSONDecodeError as e:
        raise ValueError("Expected JSON from R; got:\n" + proc.stdout) from e



def lambda_of_theta(theta, Nbar=100):
    """
    Compute lambda(theta) = Nbar / (1 + sum_{k=1}^{Nbar-1} theta/(theta+k)).
    """
    k = np.arange(1, Nbar)   # 1..Nbar-1
    denom = 1.0 + np.sum(theta / (theta + k))
    return Nbar / denom


def theta_from_lambda(lambda_target, Nbar=100):
    """
    Solve for theta given target lambda in [1, Nbar].
    Works with a single lambda (float) or an array-like of lambdas.
    """
    lambda_target = np.atleast_1d(lambda_target)  # handle scalars & arrays

    def solve_single(lam_target):
        def loss(logtheta):
            theta = np.exp(logtheta)     # enforce positivity
            lam = lambda_of_theta(theta, Nbar)
            return (lam - lam_target)**2

        result = minimize_scalar(loss, bounds=(-10, 10), method="bounded")
        return np.exp(result.x)

    thetas = np.array([solve_single(lam) for lam in lambda_target])

    # Return scalar if input was scalar
    return thetas if lambda_target.size > 1 else thetas.item()




def sample_Xij_matrix(alpha, beta, theta, Nbar, I, J, rng=None):
    """
    Vectorized sampler for:
        p_i ~ Beta(alpha, beta)                          (size I)
        p_ij | p_i ~ Beta(theta * p_i, theta*(1-p_i))    (size I x J)
        X_ij | p_ij ~ Binomial(Nbar, p_ij)               (size I x J)

    Special handling:
        If theta == np.inf at any position, we set p_ij = p_i there.

    Parameters
    ----------
    alpha, beta : float
    theta : float or array-like
        - scalar (shared for all i,j), may be np.inf
        - shape (I,), (I,1), (1,J), or (I,J) (broadcastable). Entries may be np.inf.
    Nbar : int or array-like
        - scalar or broadcastable to (I,J)
    I, J : int
    rng : np.random.Generator or seed or None

    Returns
    -------
    X : (I, J) int array
    p_i : (I,) float array
    p_ij : (I, J) float array
    """
    rng = np.random.default_rng(rng)

    # 1) p_i ~ Beta(alpha, beta), shape (I,)
    p_i = rng.beta(alpha, beta, size=I)

    # --- Broadcast theta to (I, J)
    theta = np.asarray(theta)
    if theta.ndim == 0:
        theta_ij = np.full((I, J), theta, dtype=float)
    elif theta.shape == (I,):
        theta_ij = np.repeat(theta[:, None], J, axis=1)
    elif theta.shape == (I, 1) or theta.shape == (1, J) or theta.shape == (I, J):
        theta_ij = np.broadcast_to(theta, (I, J)).astype(float)
    else:
        theta_ij = np.broadcast_to(theta, (I, J)).astype(float)

    # 2) p_ij | p_i
    # Start with the degenerate case p_ij = p_i for all cells,
    # then overwrite where theta is finite.
    p_ij = np.broadcast_to(p_i[:, None], (I, J)).copy()

    finite_mask = np.isfinite(theta_ij)  # True where theta is finite
    if np.any(finite_mask):
        # Parameters only where theta is finite
        a_ij = theta_ij * p_i[:, None]
        b_ij = theta_ij * (1.0 - p_i[:, None])

        # Draw only for finite cells (masked 1D arrays)
        a = a_ij[finite_mask]
        b = b_ij[finite_mask]

        # NOTE: rng.beta accepts array-shaped a,b and returns matching shape
        p_ij[finite_mask] = rng.beta(a, b)

    # 3) X_ij | p_ij ~ Binomial(Nbar, p_ij)
    Nbar = np.asarray(Nbar)
    if Nbar.ndim == 0:
        N_ij = np.full((I, J), int(Nbar))
    else:
        N_ij = np.broadcast_to(Nbar, (I, J)).astype(int)

    X = rng.binomial(N_ij, p_ij)
    return X, p_i, p_ij









#########################################################################
#########################################################################
## Functions to check the fit against beta distribution
#########################################################################
#########################################################################

import numpy as np
from scipy.stats import beta, binom
from math import log, sqrt

# ---------- helpers: parameter transforms ----------
def ab_to_mu_rho(alpha, beta_):
    mu = alpha / (alpha + beta_)
    rho = 1.0 / (alpha + beta_ + 1.0)  # same definition used in your R code
    return mu, rho

def mu_rho_to_ab(mu, rho):
    # inverse mapping used by your R code
    t = (1.0 / rho) - 1.0
    alpha = mu * t
    beta_ = (1.0 - mu) * t
    return alpha, beta_

# ---------- distances between Beta mixing distributions over p_i ----------
def hellinger_beta(alpha1, beta1, alpha2, beta2, m=2000):
    # numeric quadrature on an equally spaced grid in (0,1)
    p = (np.arange(1, m+1) - 0.5) / m
    g1 = beta.pdf(p, alpha1, beta1)
    g2 = beta.pdf(p, alpha2, beta2)
    # trapezoid not crucial here because grid is dense/even; use simple mean
    return np.sqrt(1.0 - np.mean(np.sqrt(g1 * g2)))

def kl_beta_numeric(alpha1, beta1, alpha2, beta2, m=2000, eps=1e-12):
    # KL( Beta(alpha1,beta1) || Beta(alpha2,beta2) ) via numeric integration
    p = (np.arange(1, m+1) - 0.5) / m
    g1 = beta.pdf(p, alpha1, beta1) + eps
    g2 = beta.pdf(p, alpha2, beta2) + eps
    return np.mean(np.log(g1) - np.log(g2))  # expectation under g1 (Riemann approx)

# ---------- hierarchical simulation respecting theta ----------
def simulate_hierarchical_counts(B, b, N_tilde, alpha, beta_, theta, rng=None):
    """
    B consignments, each with b groups (subsamples),
    each group measuring N_tilde items.
    """
    rng = np.random.default_rng() if rng is None else rng
    # draw p_i ~ Beta(alpha,beta)
    p_i = rng.beta(alpha, beta_, size=B)
    # for each consignment i and group j, draw p_ij | p_i ~ Beta(theta*p_i, theta*(1-p_i))
    # then counts X_ij | p_ij ~ Bin(N_tilde, p_ij)
    X = np.empty((B, b), dtype=int)
    for i in range(B):
        a = theta * p_i[i] + 1e-12
        bpar = theta * (1.0 - p_i[i]) + 1e-12
        p_ij = rng.beta(a, bpar, size=b)
        X[i, :] = rng.binomial(N_tilde, p_ij)
    return X

# ---------- fitted-vs-true comparisons ----------
def summarize_fit_vs_truth(alpha_hat, beta_hat, alpha_true, beta_true):
    mu_hat, rho_hat = ab_to_mu_rho(alpha_hat, beta_hat)
    mu_true, rho_true = ab_to_mu_rho(alpha_true, beta_true)

    H = hellinger_beta(alpha_hat, beta_hat, alpha_true, beta_true)
    KL_ht = kl_beta_numeric(alpha_hat, beta_hat, alpha_true, beta_true)
    KL_th = kl_beta_numeric(alpha_true, beta_true, alpha_hat, beta_hat)

    summary = {
        "alpha_hat": alpha_hat, "beta_hat": beta_hat,
        "alpha_true": alpha_true, "beta_true": beta_true,
        "mu_hat": mu_hat, "rho_hat": rho_hat,
        "mu_true": mu_true, "rho_true": rho_true,
        "hellinger": H,
        "kl_hat_to_true": KL_ht,
        "kl_true_to_hat": KL_th
    }
    return summary

# ---------- optional: simple predictive check statistic ----------
def predictive_tail_prob_total_positives(y_obs, B, b, N_tilde, alpha, beta_, theta, R=2000, rng=None):
    """
    Posterior-predictive style (but using prior/fitted parameters as plug-in):
    simulate replicated datasets from the hierarchical model and compare
    the total number of positives across all groups to the observed total.
    """
    rng = np.random.default_rng() if rng is None else rng
    tot_obs = np.sum(y_obs)
    reps = np.empty(R, dtype=int)
    for r in range(R):
        X = simulate_hierarchical_counts(B, b, N_tilde, alpha, beta_, theta, rng=rng)
        reps[r] = X.sum()
    # two-sided tail-ish p-value
    return np.mean(np.abs(reps - reps.mean()) >= np.abs(tot_obs - reps.mean()))






# 1) Put in your "true" Beta on p_i (from your data-generation or paper's example)
alpha0, beta0 = 2.5, 60.0  # <- example placeholders

# 2) Put in your fitted parameters from the R routine (alpha_hat, beta_hat)
alpha_hat, beta_hat = 2.8, 58.0  # <- replace with your fit

# 3) Compare parameters and densities
out = summarize_fit_vs_truth(alpha_hat, beta_hat, alpha0, beta0)
for k,v in out.items():
    print(f"{k:>18}: {v}")

# 4) If you want a predictive check that respects θ:
theta = 10.0          # your chosen/fixed θ used in fitting/sensitivity
B = 100               # number of consignments (i)
b = 5                 # groups per consignment (j)
N_tilde = 30          # items tested per group
# Suppose you have observed counts y_obs shaped (B,b). If not, simulate a fake one:
y_obs = simulate_hierarchical_counts(B, b, N_tilde, alpha0, beta0, theta)

p_tail = predictive_tail_prob_total_positives(
    y_obs, B, b, N_tilde,
    alpha_hat, beta_hat, theta, R=2000
)
print(f"\nPredictive tail prob (total positives): {p_tail:.3f}")


#########################################################################
#########################################################################










#########################################################################
#########################################################################
## Functions to check the fit against more general underlying distributions
## The idea is to compare the mean and variances of the fitted beta
## binomal and the true underlying distribution
#########################################################################
#########################################################################

import numpy as np
from scipy.stats import beta
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

def summarize_moments(alpha, beta_):
    mean = alpha / (alpha + beta_)
    var  = (alpha * beta_) / ((alpha + beta_)**2 * (alpha + beta_ + 1))
    return mean, var

# Example: compare fitted Beta vs. "true" distribution (any distribution with samples)
def compare_to_true_distribution(alpha_hat, beta_hat, true_samples):
    mean_hat, var_hat = summarize_moments(alpha_hat, beta_hat)
    mean_true, var_true = np.mean(true_samples), np.var(true_samples)

    return {
        "mean_hat": mean_hat, "mean_true": mean_true,
        "var_hat": var_hat,   "var_true": var_true
    }



def distribution_distances(fitted_sampler, true_sampler, m=5000, bins=200):
    # sample both distributions
    x_fit = fitted_sampler(m)
    x_true = true_sampler(m)

    # histogram approximation
    hist_range = (0,1)  # for probabilities
    p_fit, edges = np.histogram(x_fit, bins=bins, range=hist_range, density=True)
    p_true, _    = np.histogram(x_true, bins=bins, range=hist_range, density=True)

    # normalize to PMF
    p_fit = p_fit / p_fit.sum()
    p_true = p_true / p_true.sum()

    # distances
    js = jensenshannon(p_fit, p_true)
    wd = wasserstein_distance(x_fit, x_true)
    return {"JS": js, "Wasserstein": wd}













import numpy as np
from scipy.stats import beta, norm
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

# ---------------------------
# Generic samplers over p in (0,1)
# ---------------------------
def sampler_beta(alpha, beta_):
    """Return a function that samples from Beta(alpha, beta_)."""
    def _s(n, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        return rng.beta(alpha, beta_, size=n)
    return _s

def sampler_logitnormal(mu, sigma):
    """Return a sampler for the logit-normal on (0,1): p = logistic(N(mu, sigma^2))."""
    def _s(n, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        z = rng.normal(mu, sigma, size=n)
        return 1.0 / (1.0 + np.exp(-z))
    return _s

def sampler_empirical(p_samples):
    """Return a sampler that resamples from given empirical probabilities (bootstrap)."""
    p_samples = np.asarray(p_samples)
    def _s(n, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        idx = rng.integers(0, len(p_samples), size=n)
        return p_samples[idx]
    return _s

# ---------------------------
# Moments from samples (works for any distribution on (0,1))
# ---------------------------
def moment_summary_from_samples(samples):
    s = np.asarray(samples)
    return {
        "mean": float(np.mean(s)),
        "var":  float(np.var(s, ddof=0)),
        "skew": float(((s - s.mean())**3).mean() / (s.std(ddof=0)**3 + 1e-12))
    }

# ---------------------------
# Distribution distances (agnostic to parametric form)
# ---------------------------
def distances_from_samplers(fitted_sampler, true_sampler, m=5000, bins=200, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x_fit  = fitted_sampler(m, rng=rng)
    x_true = true_sampler(m,  rng=rng)

    # histogram on (0,1); convert to discrete pmf for JS/Hellinger
    hist_range = (0, 1)
    p_fit, edges = np.histogram(x_fit, bins=bins, range=hist_range, density=True)
    p_true, _    = np.histogram(x_true, bins=bins, range=hist_range, density=True)

    # Convert to pmf (sum to 1) for distances
    p_fit = p_fit / (p_fit.sum() + 1e-15)
    p_true = p_true / (p_true.sum() + 1e-15)

    # Jensen–Shannon (symmetric, bounded [0,1])
    JS = float(jensenshannon(p_fit, p_true))

    # Hellinger (bounded [0,1])
    H = float(np.sqrt(0.5 * np.sum((np.sqrt(p_fit) - np.sqrt(p_true))**2)))

    # Wasserstein-1 (Earth Mover’s) on raw samples
    W = float(wasserstein_distance(x_fit, x_true))

    # Moment summaries too
    mom_fit  = moment_summary_from_samples(x_fit)
    mom_true = moment_summary_from_samples(x_true)

    return {
        "JS": JS,
        "Hellinger": H,
        "Wasserstein": W,
        "moments_fitted": mom_fit,
        "moments_true": mom_true,
        "bins": edges.tolist()
    }

# ---------------------------
# Hierarchical predictive simulation respecting theta
# p_i ~ <sampler>, p_ij | p_i ~ Beta(theta p_i, theta (1 - p_i)),
# X_ij | p_ij ~ Bin(N_tilde, p_ij)
# ---------------------------
def simulate_hierarchical_counts(B, b, N_tilde, p_i_sampler, theta, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    p_i = p_i_sampler(B, rng=rng)                  # B consignments
    X = np.empty((B, b), dtype=int)
    for i in range(B):
        a = max(theta * p_i[i], 1e-12)
        c = max(theta * (1.0 - p_i[i]), 1e-12)
        p_ij = rng.beta(a, c, size=b)              # within-consignment clustering
        X[i, :] = rng.binomial(N_tilde, p_ij)
    return X

def predictive_tail_prob_total_positives(y_obs, B, b, N_tilde, fitted_sampler, theta, R=2000, rng=None):
    """
    Plug-in predictive check: simulate replicated datasets under the FITTED mixing sampler,
    then compare total positives to the observed.
    """
    rng = np.random.default_rng() if rng is None else rng
    tot_obs = int(np.sum(y_obs))
    reps = np.empty(R, dtype=int)
    for r in range(R):
        X = simulate_hierarchical_counts(B, b, N_tilde, fitted_sampler, theta, rng=rng)
        reps[r] = X.sum()
    # two-sided tail probability around the simulated mean
    return float(np.mean(np.abs(reps - reps.mean()) >= np.abs(tot_obs - reps.mean())))

# ---------------------------
# Convenience: fitted Beta moments (if you want to print them)
# ---------------------------
def beta_moments(alpha, beta_):
    mean = alpha / (alpha + beta_)
    var  = (alpha * beta_) / ((alpha + beta_)**2 * (alpha + beta_ + 1.0))
    return {"mean": float(mean), "var": float(var)}








#### “True” = Logit-Normal, “Fitted” = Beta ####


# --- define "true" distribution over p_i ---
true_mu, true_sigma = -3.2, 0.8
true_sampler = sampler_logitnormal(true_mu, true_sigma)

# --- suppose you fitted a Beta to the data ---
alpha_hat, beta_hat = 2.8, 58.0
fitted_sampler = sampler_beta(alpha_hat, beta_hat)

# 1) Distribution distances + moment comparison (no data needed)
dist_out = distances_from_samplers(fitted_sampler, true_sampler, m=8000, bins=250)
print("Distances:\n", {k: dist_out[k] for k in ("JS","Hellinger","Wasserstein")})
print("\nFitted Beta moments:", dist_out["moments_fitted"])
print("True (sampled) moments:", dist_out["moments_true"])

# 2) Hierarchical predictive check using synthetic observed data from the TRUE model
B, b, N_tilde, theta = 100, 5, 30, 10.0
y_obs = simulate_hierarchical_counts(B, b, N_tilde, true_sampler, theta)

p_tail = predictive_tail_prob_total_positives(y_obs, B, b, N_tilde, fitted_sampler, theta, R=2000)
print(f"\nPredictive tail prob (total positives) under fitted model: {p_tail:.3f}")

# (Optional) print the fitted Beta’s closed-form moments for reference
print("\nClosed-form moments of fitted Beta:", beta_moments(alpha_hat, beta_hat))






















###### “True” = Empirical (e.g., from external study), “Fitted” = Beta ######

# Pretend these empirical p_i came from a prior survey or a simulation:
p_empirical = np.clip(np.r_[np.full(60, 0.02), np.full(40, 0.08)], 1e-9, 1-1e-9)
true_sampler_emp = sampler_empirical(p_empirical)

alpha_hat, beta_hat = 3.0, 120.0
fitted_sampler = sampler_beta(alpha_hat, beta_hat)

dist_out = distances_from_samplers(fitted_sampler, true_sampler_emp, m=8000, bins=200)
print("Distances vs empirical truth:\n", {k: dist_out[k] for k in ("JS","Hellinger","Wasserstein")})
print("\nMoments (fitted):", dist_out["moments_fitted"])
print("Moments (empirical):", dist_out["moments_true"])




#### “True” = Beta, “Fitted” = Beta (baseline sanity check) ####

true_alpha, true_beta = 2.5, 60.0
true_sampler = sampler_beta(true_alpha, true_beta)
fitted_sampler = sampler_beta(2.6, 59.0)

dist_out = distances_from_samplers(fitted_sampler, true_sampler, m=8000, bins=200)
print("Distances (Beta vs Beta):\n", {k: dist_out[k] for k in ("JS","Hellinger","Wasserstein")})



#########################################################################
#########################################################################

















if __name__ == "__main__":
    # Testing from Clarke et al. (2023) paper
    # 72 consignments, 800 "groups" in each one, 94 "groups" sampled
    # 68 had 0 groups with identified contamination
    # Of the 4 consignments that had contamination, the number of groups out of the 94 sampled
    # that were contaminated for each of those consignments were 4, 4, 7, 21
    '''
    ty = [0, 4, 7, 21] # Number of groups testing positive
    freq = [68, 2, 1, 1] # Frequency of observations/consignments where that many groups tested positive
    b = 94 # Total groups sampled
    B = 800 # Total groups per consignment
    Nbar = 100 #Group size
    theta = np.inf
    R = 1000
    startval = (0.0, 0.0)
    se = True
    '''


    ### Our Synthetic Data Implementation

    # Pull in the synthetic data
    # Pull in synthetic information and compare
    dir = r'C:\Users\agorjk1\Box\NHH15 - USDA APHIS EDISON\05 PPQ Engagement\PPQ RBS Data (Folder shared with APHIS)\APL Created Data Related Items\Synthetic_Data'
    filename = os.path.join(dir, f'synthetic_pis_data.csv')
    # Load in task log from a single replication of a run
    df_pis_synthetic = pd.read_csv(filename)

    # Create a dataframe to store results
    columns = ['INSPECTION_ID',
               'COUNTRY_OF_ORIGIN_NAME',
               'Number of Boxes with Action']
    action_summary = pd.DataFrame(columns=columns)

    for inspection_id, group_df in df_pis_synthetic.groupby("INSPECTION_ID"):
        group_df = group_df.reset_index(drop=True)

        action_summary.loc[len(action_summary)] = {
            'INSPECTION_ID': inspection_id,
            'COUNTRY_OF_ORIGIN_NAME': group_df['COUNTRY_OF_ORIGIN_NAME'][0],
            'Number of Boxes with Action': sum(group_df['action']),
        }

    value_counts = action_summary["Number of Boxes with Action"].value_counts().sort_index()
    ty = value_counts.index.to_list()  # Number of boxes that have been identified actions
    freq = value_counts.values.tolist()  # Frequency of observations/consignments where that many groups tested positive


    #ty = [0, 1, 3, 13, 18, 23]  # Number of groups testing positive
    #freq = [94, 1, 1, 1, 1, 2]  # Frequency of observations/consignments where that many groups tested positive
    b = 25
    B = 100
    Nbar = 200
    theta = np.inf
    lambda_test = 1
    R = 1000
    startval = (0.0, 0.0)
    se = True

    #lambda_test = 5
    #theta = theta_from_lambda(lambda_test, Nbar=Nbar)
    # Calculate a theta from a fixed lambda
    # Case 1: single lambda
    #theta_single = theta_from_lambda(50, Nbar=100)
    #print("Theta for lambda=50:", theta_single)

    # Case 2: array of lambdas
    lambdas = [1, 10, 25, 50, 75, 90]
    thetas = theta_from_lambda(lambdas, Nbar=Nbar)
    '''
    print('')
    print(f'FITTING WITH CLARKE 2023 METHOD (BETA-BINOMIAL APPROACH)')
    print(f'   Parameters when NO CLUSTERING (theta = {theta}/lambda = {lambda_test})')
    res = run_bb_group_model(ty, b, B, Nbar, freq, theta, R, startval, se)
    print(f'      Alpha = {res["alpha"]}')
    print(f'      Beta = {res["beta"]}')
    print('')

    '''
    # Initialize an empty DataFrame once before the loop
    df_results = pd.DataFrame(columns=[
        "Method",
        "Lambda",
        "Number of contaminated consignments",
        "Five number summary",
        "Mean contamination rate",
        "95% CI Lower",
        "95% CI Upper",
        "# Contaminated Plants/Units (Avg for methods and actual for ground truth)"
    ])

    for i in range(len(thetas)):
        if lambdas[i] == 1:
            theta = np.inf
            print(f'CLARKE APPROACH:  Parameters when NO CLUSTERING (theta = {theta}/lambda = 1)')
        else:
            theta = thetas[i]
            print(f'CLARKE APPROACH: Parameters when NO CLUSTERING (theta = {theta}/lambda = {lambdas[i]})')
        res = run_bb_group_model(ty, b, B, Nbar, freq, theta, R, startval, se)
        print(f'   Alpha = {res["alpha"]}')
        print(f'   Beta = {res["beta"]}')
        print('')


        # Example:  I number of consignments, J number of groups in each consignment
        I, J = 100, 100
        alpha, beta = res["alpha"], res["beta"]
        theta = np.Inf  # scalar; you can also pass an array of shape (I,) to vary by group
        Nbar = 200  # scalar; or pass an (I,J) array if trials vary

        X, p_i, p_ij = sample_Xij_matrix(alpha, beta, theta, Nbar, I, J, rng=1)

        # Calculate the summary stats across the 100 groups of the 100 consignments
        row_sums = np.sum(X, axis=1)

        # Step 1: mask rows with at least one value > 0
        mask = (X > 0).any(axis=1)
        X_filtered = X[mask]

        # Step 2: row sums
        row_sums = np.sum(X_filtered, axis=1)

        # Step 3a: total number of qualifying rows
        n_rows = len(row_sums)


        # Contamination rate summary
        contamination_rate = row_sums / (J*Nbar)
        five_num_summary_rate = {
            "min": np.min(contamination_rate),
            "Q1": np.percentile(contamination_rate, 25),
            "median": np.median(contamination_rate),
            "Q3": np.percentile(contamination_rate, 75),
            "max": np.max(contamination_rate),
        }

        n = len(contamination_rate)
        mean_rate = np.mean(contamination_rate)
        sem_rate = stats.sem(contamination_rate)  # standard error of the mean
        ci_low_rate, ci_high_rate = stats.t.interval(0.95, df=n - 1, loc=mean_rate, scale=sem_rate)

        print('')
        print(f'   FITTED (CLARKE APPROACH):  CONTAMINATION RATE')
        print("      Number of contaminated consignments:", n_rows)
        print("      Five number summary:", five_num_summary_rate)
        print("      Mean contamination rate:", mean_rate)
        print("      95% CI for mean contamination rate:", (ci_low_rate, ci_high_rate))



        # Number of contaminated plants
        # Step 3b: five number summary
        five_num_summary = {
            "min": np.min(row_sums),
            "Q1": np.percentile(row_sums, 25),
            "median": np.median(row_sums),
            "Q3": np.percentile(row_sums, 75),
            "max": np.max(row_sums),
        }

        # Step 3c: mean and 95% CI
        mean_total_plants_contaminated = np.mean(row_sums)
        sem = stats.sem(row_sums)
        ci_low, ci_high = stats.t.interval(0.95, df=n_rows - 1, loc=mean, scale=sem)

        print('')
        print(f'   FITTED (CLARKE APPROACH):  TOTAL PLANTS CONTAMINATED')
        print("      Number of contaminated consignments:", n_rows)
        print("      Five number summary:", five_num_summary)
        print("      Average number (across 100 consignments) plants infected (out of those that are infected):", mean_total_plants_contaminated)
        print(f'     Total number of plants contaminated (across 100 consignments): {sum(row_sums)}')
        print("      95% CI for mean infected:", (ci_low, ci_high))
        print('')

        # Build a row as a dict
        row = {
            "Method": "CLARKE",
            "Lambda": lambdas[i],
            "Number of contaminated consignments": n_rows,
            "Five number summary": five_num_summary_rate,
            "Mean contamination rate": mean_rate,
            "95% CI Lower": ci_low_rate,
            "95% CI Upper": ci_high_rate,
            "# Contaminated Plants/Units (Avg for methods and actual for ground truth)": sum(row_sums)
        }

        # Append row to DataFrame
        df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)






    # Chen/PoPS Border Approach
    import pandas as pd
    from JoeFiles.mle_contamination_glm import fit_beta_glm_match_failrate
    from JoeFiles.mle_stats_models_approach import *

    print(f'PoPS BORDER APPROACH:  Chen 2018 + Trouve 2020')

    res = fit_cloglog_statsmodels(df_pis_synthetic, detection_rate=0.8)
    print('')
    print(f'Initial CLogLog Model Fit')
    print("   alpha_hat:", res.alpha_hat)
    print("   alpha_se:", res.alpha_se, "  95% CI:", res.alpha_ci)
    print("   mean_r  :", res.mean_r, "  95% CI:", res.mean_r_ci)
    print("   theta_hat (exp(alpha)):", res.theta_hat)
    print("   gamma_mean (if eps given):", res.gamma_mean)
    print('')



    # Use outputs to generate beta distribution parameters
      # Key is to choose a standard deviation for the beta distribution
      # Option 1:  Set standard deviation yourself
      # Option 2:  Match CI width from original input data (data-driven spread)

    # Option 1:  Set standard deviation yourself
    #mu = res.mean_r  # target mean
    #sd = 0.05  # <- arbitrary choice (similar to PoPS border)
    #alpha_pops, beta_pops, K_pops = beta_from_mean_sd(mu, sd)

    # Option 2:  Match CI width from original input data (data-driven spread)
    mu = res.mean_r
    lwr, upr = res.mean_r_ci
    alpha_pops, beta_pops, K_pops = beta_from_mean_ci(mu, lwr, upr)

    print(f'   Alpha = {alpha_pops}')
    print(f'   Beta = {beta_pops}')

    # Example:  I number of consignments, J number of groups in each consignment
    I, J = 100, 100
    alpha, beta = alpha_pops, beta_pops
    theta = np.Inf  # scalar; you can also pass an array of shape (I,) to vary by group
    Nbar = 200  # scalar; or pass an (I,J) array if trials vary

    X, p_i, p_ij = sample_Xij_matrix(alpha, beta, theta, Nbar, I, J, rng=1)

    # Calculate the summary stats across the 100 groups of the 100 consignments
    row_sums = np.sum(X, axis=1)

    # Step 1: mask rows with at least one value > 0
    mask = (X > 0).any(axis=1)
    X_filtered = X[mask]

    # Step 2: row sums
    row_sums = np.sum(X_filtered, axis=1)

    # Step 3a: total number of qualifying rows
    n_rows = len(row_sums)

    # Contamination rate summary
    contamination_rate = row_sums / (J * Nbar)
    five_num_summary_rate = {
        "min": np.min(contamination_rate),
        "Q1": np.percentile(contamination_rate, 25),
        "median": np.median(contamination_rate),
        "Q3": np.percentile(contamination_rate, 75),
        "max": np.max(contamination_rate),
    }

    n = len(contamination_rate)
    mean_rate = np.mean(contamination_rate)
    sem_rate = stats.sem(contamination_rate)  # standard error of the mean
    ci_low_rate, ci_high_rate = stats.t.interval(0.95, df=n - 1, loc=mean_rate, scale=sem_rate)

    from scipy.stats import beta

    alpha, beta_param = alpha_pops, beta_pops
    mean_rate = alpha / (alpha + beta_param)

    print('')
    print(f'   FITTED (POPS BORDER APPROACH):  CONTAMINATION RATE')
    print("      Mean contamination rate:", mean_rate)



    # Build a row as a dict to be added to results
    row = {
        "Method": "PoPS_Border (Chen 2018 and Trouve 2020)",
        "Lambda": "N/A",
        "Number of contaminated consignments": "N/A",
        "Five number summary": "N/A",
        "Mean contamination rate": mean_rate,
        "95% CI Lower": "N/A",
        "95% CI Upper": 'N/A'
    }

    # Append row to DataFrame
    df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)

    from scipy.stats import beta

    alpha, beta_param = 0.0102, 5.3839
    beta_mean = alpha / (alpha + beta_param)


    print('')
    print(f'GROUND TRUTH:  CONTAMINATION RATE')
    print(f'Theoretical mean contamination rate: {beta_mean}')
    print(f'   Alpha parameter = {alpha}, beta parameter = {beta_param}')
    if ci_low_rate <= beta_mean <= ci_high_rate:
        print("   The mean contamination rate is consistent with the observed mean (within 95% CI).")
    else:
        print("   The mean contamination rate is outside the observed 95% CI.")

    # Build a row as a dict to be added to results
    row = {
        "Method": "Ground Truth (from synthetic data)",
        "Lambda": "N/A",
        "Number of contaminated consignments": "N/A",
        "Five number summary": "N/A",
        "Mean contamination rate": beta_mean,
        "95% CI Lower": "N/A",
        "95% CI Upper": "N/A"
    }

    # Append row to DataFrame
    df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)

    # Write out results
    result_dir = r'C:\Users\agorjk1\Box\NHH15 - USDA APHIS EDISON\05 PPQ Engagement\PPQ RBS Data (Folder shared with APHIS)\APL Created Data Related Items\Contamination_Experiments'
    filepath = os.path.join(result_dir, f'contamination_experiment_results.csv')
    df_results.to_csv(filepath, index=False)
    print('')













    # Pull in synthetic information and compare
    dir = r'C:\Users\agorjk1\Box\NHH15 - USDA APHIS EDISON\05 PPQ Engagement\PPQ RBS Data (Folder shared with APHIS)\APL Created Data Related Items\Synthetic_Data'
    filename = os.path.join(dir, f'synthetic_consignment_data.csv')
    # Load in task log from a single replication of a run
    df_consignment_synthetic = pd.read_csv(filename)

    # Mask rows with at least one value > 0
    df_temp = df_consignment_synthetic[df_consignment_synthetic['Total Items Contaminated'] > 0]

    average_synthetic_data_plants_contaminated = df_temp['Total Items Contaminated'].mean()
    total_number_contaminated = sum(df_temp['Total Items Contaminated'])

    print('')
    print(f'GROUND TRUTH:  TOTAL PLANTS CONTAMINATED')
    print(f'Total number of consignments contaminated in synthetic data: {df_temp.shape[0]}')
    print(
        f'Average number plants infected (out of those that are infected) in synthetic data (across infected consignments): {average_synthetic_data_plants_contaminated}')
    print(
        f'Total number (across infected consignments) plants infected (out of those that are infected) in synthetic data: {total_number_contaminated}')
    if ci_low <= average_synthetic_data_plants_contaminated <= ci_high:
        print("The average contaminated items (plants) mean is consistent with the observed mean (within 95% CI).")
    else:
        print("The average contaminated items (plants) is outside the observed 95% CI.")


    print('')