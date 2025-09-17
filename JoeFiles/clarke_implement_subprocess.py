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




# Function to sample the distribution of X_ij given parameters alpha, beta, theta, and Nbar
'''
def sample_Xij_matrix(alpha, beta, theta, Nbar, I, J, rng=None):
    """
    Vectorized sampler for the hierarchy:
        p_i ~ Beta(alpha, beta)                          (size I)
        p_ij | p_i ~ Beta(theta * p_i, theta*(1-p_i))    (size I x J)
        X_ij | p_ij ~ Binomial(Nbar, p_ij)               (size I x J)

    Parameters
    ----------
    alpha, beta : float
        Beta prior hyperparameters for p_i.
    theta : float or array-like
        Precision/concentration parameter for p_ij|p_i. Can be:
          - scalar (shared)
          - shape (I,) to vary by i
          - shape (I,1) or (1,J) will broadcast as usual
    Nbar : int or array-like
        Binomial trials. Can be:
          - scalar (shared)
          - shape (I,J) to vary per (i,j)
          - shape (I,1) or (1,J) will broadcast as usual
    I, J : int
        Number of consignments i and groups j.
    rng : None or np.random.Generator
        Random generator (pass a seeded Generator for reproducibility).

    Returns
    -------
    X : ndarray, shape (I, J)
        Samples of X_ij.
    p_i : ndarray, shape (I,)
        The sampled p_i (returned for convenience).
    p_ij : ndarray, shape (I, J)
        The sampled p_ij (returned for convenience).
    """
    rng = np.random.default_rng(rng)

    # 1) p_i ~ Beta(alpha, beta), shape (I,)
    p_i = rng.beta(alpha, beta, size=I)  # (I,)

    # 2) p_ij | p_i ~ Beta(theta * p_i, theta*(1 - p_i)), shape (I, J)
    if theta == np.Inf:

    else:
        theta = np.asarray(theta)
        # Broadcast theta to (I, J) via (I,1) * (1,J)
        theta_ij = np.broadcast_to(theta.reshape(-1, 1) if theta.ndim == 1 and theta.size == I
                                   else np.array(theta), (I, J)) if theta.ndim != 0 else np.full((I, J), theta)

        a_ij = theta_ij * p_i[:, None]  # (I, J)
        b_ij = theta_ij * (1.0 - p_i[:, None])  # (I, J)
        p_ij = rng.beta(a_ij, b_ij)  # (I, J)

        # 3) X_ij | p_ij ~ Binomial(Nbar, p_ij), shape (I, J)
        Nbar = np.asarray(Nbar)
        if Nbar.ndim == 0:
            N_ij = Nbar * np.ones((I, J), dtype=int)
        else:
            N_ij = np.broadcast_to(Nbar, (I, J)).astype(int)

        X = rng.binomial(N_ij, p_ij)  # (I, J)
        return X, p_i, p_ij

'''

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
    lambdas = [10, 25, 50, 75, 90]
    thetas = theta_from_lambda(lambdas, Nbar=Nbar)

    print('')
    print(f'FITTING WITH CLARKE 2023 METHOD (BETA-BINOMIAL APPROACH)')
    print(f'   Parameters when NO CLUSTERING (theta = {theta}/lambda = {lambda_test})')
    res = run_bb_group_model(ty, b, B, Nbar, freq, theta, R, startval, se)
    print(f'      Alpha = {res["alpha"]}')
    print(f'      Beta = {res["beta"]}')
    print('')

    '''
    print(f'Parameters with CLUSTERING (0 < theta < Infinity)')
    for i in range(len(thetas)):
        theta = thetas[i]
        res = run_bb_group_model(ty, b, B, Nbar, freq, theta, R, startval, se)
        print(f'   lambda = {lambdas[i]}')
        print(f'   theta = {thetas[i]}')
        print(f'      Alpha = {res["alpha"]}')
        print(f'      Beta = {res["beta"]}')
        print('')
    print('')
    '''

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
    print(f'FITTED:  CONTAMINATION RATE')
    print("Number of contaminated consignments:", n_rows)
    print("Five number summary:", five_num_summary_rate)
    print("Mean contamination rate:", mean_rate)
    print("95% CI for mean contamination rate:", (ci_low_rate, ci_high_rate))

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

    # Step 3b: five number summary
    five_num_summary = {
        "min": np.min(row_sums),
        "Q1": np.percentile(row_sums, 25),
        "median": np.median(row_sums),
        "Q3": np.percentile(row_sums, 75),
        "max": np.max(row_sums),
    }

    # Step 3c: mean and 95% CI
    mean = np.mean(row_sums)
    sem = stats.sem(row_sums)
    ci_low, ci_high = stats.t.interval(0.95, df=n_rows - 1, loc=mean, scale=sem)

    print('')
    print(f'FITTED:  TOTAL PLANTS CONTAMINATED')
    print("Number of contaminated consignments:", n_rows)
    print("Five number summary:", five_num_summary)
    print("Average number (across 100 consignments) plants infected (out of those that are infected):", mean)
    print(f'Total number of plants contaminated (across 100 consignments): {sum(row_sums)}')
    print("95% CI for mean infected:", (ci_low, ci_high))
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