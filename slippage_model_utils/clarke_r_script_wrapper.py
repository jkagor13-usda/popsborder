import os
import re
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping, Sequence, TypedDict, Literal, Tuple, SupportsFloat
import numpy as np
from scipy.optimize import minimize_scalar
from scipy import stats
import pandas as pd


# === CONFIG ===
# Beta-Binomial Model
#R_SCRIPT_PATH_bb_cli = r"/plant-inspection-station-simulation/JoeFiles/clarke_2023_code/clarke_bb_model.R"
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

# ---- Nested schema for `optim` ----
class OptimResult(TypedDict):
    value: float
    par: Tuple[float, float]                       # two parameters
    counts: Tuple[int, Literal["NA"]]             # R's optim: (fn_evals, "NA")
    convergence: int                               # 0 means success
    message: Mapping[str, Any]                     # R often returns an empty list/dict here
    hessian: Tuple[Tuple[float, float], Tuple[float, float]]  # 2x2 matrix


# ---- Top-level result ----
class BBResult(TypedDict):
    optim: OptimResult

    # point estimates
    alpha: float
    beta: float
    mu: float
    rho: float
    D: float

    # derived quantities / diagnostics
    E_leak: float
    prob_leak: float
    log_prob_leak: float
    pty0: float

    # standard errors (scalars)
    se_alpha: float
    se_beta: float
    se_mu: float
    se_rho: float
    se_D: float

    # standard errors (vector)
    se_par: Sequence[float]





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


def run_clarke_bb_group_model(
    ty: List[int],
    b: int,
    B: int,
    Nbar: int,
    freq: List[int],
    theta: float,
    R: int,
    startval: Sequence[SupportsFloat],
    se: bool,
) -> BBResult:
    """
    Runs the Clarke BB group model via an R script and returns parsed JSON.
    Raises FileNotFoundError if the script is missing, or CalledProcessError if R fails.
    Raises ValueError if stdout is not valid JSON.
    """
    if not Path(R_SCRIPT_PATH_bb_cli).exists():
        raise FileNotFoundError(f"R script not found: {R_SCRIPT_PATH_bb_cli}")

    cmd = _pick_rscript_command()

    theta_json = "Inf" if (isinstance(theta, float) and np.isinf(theta)) else float(theta)

    # Build the payload with explicit conversions
    payload: Dict[str, Any] = {
        "ty": [int(x) for x in ty],
        "b": int(b),
        "B": int(B),
        "Nbar": int(Nbar),
        "freq": [int(x) for x in freq],
        "theta": theta_json,
        "R": int(R),
        "startval": [float(x) for x in startval],
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

