import os
import re
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


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

# (optional) keep your simple add_numbers test — but fix the R bug (a+b)




















# === Example usage ===
if __name__ == "__main__":
    # Testing from Clarke et al. (2023) paper
    ty = [0, 4, 4, 7, 21]
    freq = [68, 1, 1, 1, 1]
    b = 94
    B = 1000
    Nbar = 100
    theta = np.inf
    R = 1000
    startval = (0.0, 0.0)
    se = True

    #bb_output = run_bb_group_model(ty, b, B, Nbar, freq,theta, R, startval, se)
    # BB.group.model <- function(ty,b,B,Nbar,freq,theta=Inf,R=1000,startval,SE=FALSE)

    res = run_bb_group_model(ty, b, B, Nbar, freq, theta, R, startval, se)
    print(res)  # dict with fields: optim, alpha, beta, mu, rho, D, E_leak, prob_leak, log_prob_leak, pty0, and SE fields if requested
    print(res["optim"]["par"])  # example: access MLEs (log-alpha, log-beta)
    print('')


    #print(f'Output of R Run = : {run_r_add_numbers(5, 7)}')