from __future__ import annotations
import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping, Sequence, TypedDict, Literal, Tuple, SupportsFloat
import numpy as np
from scipy.optimize import minimize_scalar

# === CONFIG ===
R_SCRIPT_PATH_bb_cli = r"C:\Users\agorjk1\PycharmProjects\plant-inspection-station-simulation\slippage_model_utils\clarke_bb_model.R"
CONDA_EXE_PATH = r"C:\Users\agorjk1\AppData\Local\anaconda3\Scripts\conda.exe"
CONDA_ENV_NAME: Optional[str] = "rbb"
RSCRIPT_ABS_PATH: Optional[str] = None


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

def _parse_json_from_r_stdout(stdout: str) -> dict[str, Any]:
    """
    Extract the final JSON object from mixed R stdout (startup messages, warnings, etc.).
    Returns a raw dict; use parse_bb_result(...) to validate/narrow to BBResult.
    """
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            return json.loads(ln)

    start = stdout.rfind("{")
    end = stdout.rfind("}")
    if start != -1 and end != -1 and start < end:
        return json.loads(stdout[start:end + 1])

    raise ValueError("Expected JSON from R; nothing that looks like a JSON object was found.")


def run_clarke_bb_group_model(
    ty: Sequence[int],
    b: int,
    B: int,
    Nbar: int,
    freq: Sequence[int],
    theta: float,
    R: int,
    startval: Sequence[SupportsFloat],
    se: bool,
    *,
    timeout_sec: float = 120.0,
) -> BBResult:
    """
    Runs the Clarke BB group model via an R script and returns parsed JSON.
    Raises:
      - FileNotFoundError if the script is missing
      - CalledProcessError if the R subprocess fails
      - TimeoutError if R does not complete in time
      - ValueError if stdout is not valid JSON in the expected schema
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

    try:
        proc: subprocess.CompletedProcess[str] = subprocess.run(
            cmd + [str(Path(R_SCRIPT_PATH_bb_cli)), json.dumps(payload)],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(
            f"R script timed out after {timeout_sec}s. "
            f"Partial stdout: {e.output!r}, stderr: {e.stderr!r}"
        ) from e

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=proc.returncode,
            cmd=proc.args,
            output=proc.stdout,
            stderr=proc.stderr,
        )

    # Parse the final line as JSON (ignore startup messages)
    try:
        return _parse_json_from_r_stdout(proc.stdout)  # must return BBResult
    except ValueError as e:
        # fall back to a shorter preview for debugging
        preview = proc.stdout[:500].replace("\n", "\\n")
        raise ValueError(f"Expected JSON from R; got (preview): {preview}") from e

