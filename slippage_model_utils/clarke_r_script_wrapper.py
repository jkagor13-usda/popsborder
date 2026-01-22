from __future__ import annotations
import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping, Sequence, TypedDict, Literal, Tuple, SupportsFloat
import numpy as np
from scipy.optimize import minimize_scalar
import platform
import sys
from math import isinf

# === CONFIG ===
# Name of the conda environment that contains R + required R packages.
# Can be overridden by setting the environment variable POPS_R_CONDA_ENV.
CONDA_ENV_NAME: Optional[str] = os.getenv("POPS_R_CONDA_ENV", "rbb")
REPO_NAME = "plant-inspection-station-simulation"
R_SCRIPT_REL = Path("slippage_model_utils") / "clarke_bb_model.R"

def _find_repo_root() -> Optional[Path]:
    """
    Find the repo root by looking for REPO_NAME or '.git'
    """
    # 2) Walk up from anchor (or this file) to find REPO_NAME or .git
    start = Path(__file__).resolve()
    for parent in [start, *start.parents]:
        if parent.name == REPO_NAME or (parent / ".git").exists():
            return parent

    return None

def get_r_script_path() -> Path:
    """
    Resolve the path to clarke_bb_model.R via repo root discovery + R_SCRIPT_REL
    Raises FileNotFoundError if not found.
    """
    # Find repo root
    repo_root = _find_repo_root()
    # From repo root, return path to R script
    if repo_root:
        candidate = (repo_root / R_SCRIPT_REL).resolve()
        if candidate.exists():
            return candidate

    # Raise error if the R script was not found
    raise FileNotFoundError(
        "Could not locate 'clarke_bb_model.R'.\n"
    )


def _find_conda_exe() -> Optional[Path]:
    """
    Auto-locate conda executable via:
      1) POPS_CONDA_EXE / CONDA_EXE
      2) PATH
      3) Common Anaconda/Miniconda install locations
    """

    # Respect explicit overrides first
    for env_var in ("POPS_CONDA_EXE", "CONDA_EXE"):
        v = os.getenv(env_var)
        if v and Path(v).exists():
            return Path(v).resolve()

    # Then try PATH
    which = shutil.which("conda.exe" if platform.system() == "Windows" else "conda")
    if which:
        return Path(which).resolve()

    # If above two options don't work, go through all "Common" install paths by OS
    candidates: list[Path] = []

    if platform.system() == "Windows":
        local_appdata = os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        candidates += [
            Path(local_appdata) / "anaconda3" / "Scripts" / "conda.exe",
            Path(local_appdata) / "miniconda3" / "Scripts" / "conda.exe",
            Path(local_appdata) / "Programs" / "Anaconda3" / "Scripts" / "conda.exe",
        ]
        # add standard home locations too
        candidates += [
            Path.home() / "anaconda3" / "Scripts" / "conda.exe",
            Path.home() / "miniconda3" / "Scripts" / "conda.exe",
        ]
    else:
        # macOS / Linux
        candidates += [
            Path.home() / "anaconda3" / "bin" / "conda",
            Path.home() / "miniconda3" / "bin" / "conda",
            Path("/opt/anaconda3/bin/conda"),
            Path("/usr/local/anaconda3/bin/conda"),
        ]

    for c in candidates:
        if c.exists():
            return c.resolve()

    return None


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

def _pick_rscript_command() -> list[str]:
    """
    Determine command to run Rscript via:
      conda run -n <env> Rscript

    Conda executable is auto-discovered, or can be explicitly set via:
      POPS_CONDA_EXE or CONDA_EXE environment variables.
    """

    if not CONDA_ENV_NAME:
        raise RuntimeError(
            "CONDA_ENV_NAME is not set. "
            "Set POPS_R_CONDA_ENV to the name of a conda environment "
            "that contains R and required packages."
        )

    conda_exe = _find_conda_exe()
    if conda_exe:
        return [str(conda_exe), "run", "-n", CONDA_ENV_NAME, "Rscript"]

    raise RuntimeError(
        "Could not locate the conda executable.\n\n"
        "Tried:\n"
        "  - POPS_CONDA_EXE environment variable\n"
        "  - CONDA_EXE environment variable\n"
        "  - conda on PATH\n"
        "  - common Anaconda / Miniconda install locations\n\n"
        "Fix one of the following:\n"
        "  1) Run this command from an Anaconda Prompt\n"
        "  2) Add conda to your PATH\n"
        "  3) Set POPS_CONDA_EXE to your conda executable, e.g.:\n"
        "       setx POPS_CONDA_EXE \"C:\\Users\\<you>\\miniconda3\\Scripts\\conda.exe\"\n\n"
        f"Expected conda environment name: '{CONDA_ENV_NAME}'"
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
    r_script_path_bb_cli = str(get_r_script_path())
    if not Path(r_script_path_bb_cli).exists():
        raise FileNotFoundError(f"R script not found: {r_script_path_bb_cli}")

    cmd = _pick_rscript_command()

    #theta_json = "Inf" if (isinstance(theta, float) and np.isinf(theta)) else float(theta)
    theta_json = None if isinf(float(theta)) else float(theta)

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
            cmd + [str(Path(r_script_path_bb_cli)), json.dumps(payload, allow_nan=False)],
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

