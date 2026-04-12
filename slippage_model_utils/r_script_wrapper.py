from __future__ import annotations

import ast
import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
from math import isinf
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
)

import numpy as np
import pandas as pd
from numpy.random import Generator

# === LOGGING ===
logger = logging.getLogger(__name__)

# === CONFIG ===
# Name of the conda environment that contains R + required R packages.
# Can be overridden by setting the environment variable POPS_R_CONDA_ENV.
CONDA_ENV_NAME: Optional[str] = os.getenv("POPS_R_CONDA_ENV", "rbb")
REPO_NAME = "plant-inspection-station-simulation"
R_SCRIPT_REL = Path("slippage_model_utils") / "clarke_bb_model.R"


class OptimResult(TypedDict):
    value: float
    par: Tuple[float, float]  # two parameters
    counts: Tuple[int, str]  # R's optim: (fn_evals, "NA")
    convergence: int  # 0 means success
    message: Mapping[str, Any]  # R often returns an empty list/dict here
    hessian: Tuple[Tuple[float, float], Tuple[float, float]]  # 2x2 matrix


#### Repo & script path utilities ####
def find_repo_root() -> Optional[Path]:
    """
    Find the repo root by looking for REPO_NAME or '.git'.
    """
    start = Path(__file__).resolve()
    for parent in [start, *start.parents]:
        if parent.name == REPO_NAME or (parent / ".git").exists():
            return parent
    return None


def get_script_path(relative: Path, *, repo_root: Optional[Path] = None) -> Path:
    """
    Resolve path to an R script via repo root discovery.

    Args:
        relative: Path relative to repo root.
        repo_root: Explicit repo root. If None, auto-detect via find_repo_root().

    Returns:
        Absolute path to the script.

    Raises:
        FileNotFoundError: If script cannot be located.
    """
    # Try explicit root first
    if repo_root:
        candidate = (repo_root / relative).resolve()
        if candidate.exists():
            return candidate

    # Auto-detect root
    auto_root = find_repo_root()
    if auto_root:
        candidate = (auto_root / relative).resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not locate '{relative.name}'. "
        f"Searched from repo_root={repo_root or 'auto-detected root'}; "
        f"expected relative path: {relative}"
    )


def get_r_script_path() -> Path:
    """
    Resolve the path to clarke_bb_model.R via repo root discovery + R_SCRIPT_REL.

    Raises:
        FileNotFoundError if not found.
    """
    return get_script_path(R_SCRIPT_REL)



#### Conda / R discovery utilities ####

def _find_conda_exe() -> Optional[Path]:
    """
    Auto-locate conda executable via:
      1) POPS_CONDA_EXE / CONDA_EXE
      2) PATH
      3) Common Anaconda/Miniconda install locations
    """
    # Explicit overrides
    for env_var in ("POPS_CONDA_EXE", "CONDA_EXE"):
        v = os.getenv(env_var)
        if v and Path(v).exists():
            return Path(v).resolve()

    # PATH
    which = shutil.which("conda.exe" if platform.system() == "Windows" else "conda")
    if which:
        return Path(which).resolve()

    # Common install locations
    candidates: list[Path] = []

    if platform.system() == "Windows":
        local_appdata = os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        candidates += [
            Path(local_appdata) / "anaconda" / "Scripts" / "conda.exe",
            Path(local_appdata) / "anaconda3" / "Scripts" / "conda.exe",
            Path(local_appdata) / "miniconda3" / "Scripts" / "conda.exe",
            Path(local_appdata) / "Programs" / "anaconda" / "Scripts" / "conda.exe",
        ]
        candidates += [
            Path.home() / "anaconda" / "Scripts" / "conda.exe",
            Path.home() / "anaconda3" / "Scripts" / "conda.exe",
            Path.home() / "miniconda3" / "Scripts" / "conda.exe",
        ]
    else:
        candidates += [
            Path.home() / "anaconda" / "bin" / "conda",
            Path.home() / "anaconda3" / "bin" / "conda",
            Path.home() / "miniconda3" / "bin" / "conda",
            Path("/opt/anaconda/bin/conda"),
            Path("/opt/anaconda3/bin/conda"),
            Path("/usr/local/anaconda/bin/conda"),
            Path("/usr/local/anaconda3/bin/conda"),
        ]

    for c in candidates:
        if c.exists():
            return c.resolve()

    return None


def _find_conda_env_dir(env_name: str) -> Optional[Path]:
    """
    Locate the directory of a conda environment by name.

    Args:
        env_name: Name of the conda environment

    Returns:
        Path to the environment directory, or None if not found.
    """
    # Explicit override
    override = os.getenv("POPS_CONDA_ENV_DIR")
    if override:
        override_path = Path(override)
        if override_path.exists() and override_path.is_dir():
            return override_path.resolve()

    conda_exe = _find_conda_exe()
    if not conda_exe:
        return None

    # Method 1: 'conda env list --json'
    try:
        result = subprocess.run(
            [str(conda_exe), "env", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=True,
        )
        stdout = result.stdout.strip()
        json_start = stdout.find("{")
        json_end = stdout.rfind("}")
        if json_start != -1 and json_end != -1:
            json_str = stdout[json_start : json_end + 1]
            env_data = json.loads(json_str)

            # Check 'envs' list
            envs = env_data.get("envs", [])
            for env_path in envs:
                env_path_obj = Path(env_path)
                if env_path_obj.name == env_name:
                    return env_path_obj

            # Check 'envs_details'
            envs_details = env_data.get("envs_details", {})
            for env_path_str, details in envs_details.items():
                if details.get("name") == env_name:
                    env_path_obj = Path(env_path_str)
                    if env_path_obj.exists():
                        return env_path_obj

    except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired):
        pass

    # Method 2: 'conda info --envs'
    try:
        result = subprocess.run(
            [str(conda_exe), "info", "--envs"],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=True,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            curr_env_name = parts[0]
            env_path_str = parts[-1]
            if curr_env_name == env_name:
                env_path_obj = Path(env_path_str)
                if env_path_obj.exists():
                    return env_path_obj

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    # Method 3: common env locations
    conda_root = conda_exe.parent.parent  # Scripts/bin -> conda root
    candidates = [
        conda_root / "envs" / env_name,
        Path.home() / ".conda" / "envs" / env_name,
        Path.home() / "anaconda3" / "envs" / env_name,
        Path.home() / "miniconda3" / "envs" / env_name,
    ]

    if platform.system() == "Windows":
        local_appdata = os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        candidates += [
            Path(local_appdata) / "anaconda" / "envs" / env_name,
            Path(local_appdata) / "anaconda3" / "envs" / env_name,
            Path(local_appdata) / "miniconda3" / "envs" / env_name,
        ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            if platform.system() == "Windows":
                if (candidate / "Scripts" / "activate.bat").exists() or (candidate / "python.exe").exists():
                    return candidate
            else:
                if (candidate / "bin" / "activate").exists() or (candidate / "bin" / "python").exists():
                    return candidate

    return None


def _build_windows_r_env(env_dir: Path) -> dict[str, str]:
    """
    Build environment variables for running Rscript on Windows via a conda env.
    """
    env = os.environ.copy()
    prepend = [
        str(env_dir / "Library" / "bin"),
        str(env_dir / "Scripts"),
        str(env_dir),
    ]
    env["PATH"] = os.pathsep.join(prepend + [env.get("PATH", "")])
    env["R_HOME"] = str(env_dir / "Lib" / "R")
    return env


def _pick_rscript_command() -> tuple[list[str], dict[str, str]]:
    """
    Determine command to run Rscript and environment variables.

    Returns:
        Tuple of (command_list, env_dict)
        - command_list: The command to execute
        - env_dict: Environment variables to use (or empty dict to use conda run)
    """
    if not CONDA_ENV_NAME:
        raise RuntimeError(
            "CONDA_ENV_NAME is not set. "
            "Set POPS_R_CONDA_ENV to the name of a conda environment "
            "that contains R and required packages."
        )

    conda_exe = _find_conda_exe()

    if platform.system() == "Windows":
        env_dir = _find_conda_env_dir(CONDA_ENV_NAME)
        if not env_dir:
            available_envs = "(could not retrieve environment list)"
            if conda_exe:
                try:
                    result = subprocess.run(
                        [str(conda_exe), "env", "list"],
                        capture_output=True,
                        text=True,
                        timeout=30.0,
                    )
                    available_envs = result.stdout
                except Exception:
                    pass

            raise RuntimeError(
                f"Could not find conda environment '{CONDA_ENV_NAME}'.\n\n"
                f"Conda executable found at: {conda_exe}\n\n"
                f"Available environments:\n{available_envs}\n\n"
                "Please ensure the environment exists and try:\n"
                f"  conda activate {CONDA_ENV_NAME}\n"
                f"  conda list | findstr R\n\n"
                "Or set the environment path explicitly:\n"
                f"  set POPS_CONDA_ENV_DIR=C:\\path\\to\\envs\\{CONDA_ENV_NAME}"
            )

        rscript = env_dir / "Scripts" / "Rscript.exe"
        if not rscript.exists():
            raise RuntimeError(
                f"Rscript.exe not found in environment '{CONDA_ENV_NAME}'.\n"
                f"Expected at: {rscript}\n"
                f"Environment directory: {env_dir}\n\n"
                "Please ensure R is installed in the environment:\n"
                f"  conda activate {CONDA_ENV_NAME}\n"
                "  conda install r-base"
            )

        env = _build_windows_r_env(env_dir)
        return [str(rscript)], env

    # Unix-like: use conda run
    if not conda_exe:
        raise RuntimeError(
            "Could not locate conda executable. "
            "Ensure conda is installed and CONDA_EXE or POPS_CONDA_EXE is set "
            "if using nonstandard locations."
        )

    return [str(conda_exe), "run", "-n", CONDA_ENV_NAME, "Rscript"], {}


#### R subprocess helpers ####
def _run_rscript(
    cmd: list[str],
    script_path: str,
    arg: str,
    *,
    env: Optional[dict[str, str]] = None,
    timeout_sec: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    """
    Run Rscript with a single JSON or file-path argument and common error handling.
    """
    subprocess_env = env if env else None

    try:
        proc: subprocess.CompletedProcess[str] = subprocess.run(
            cmd + [script_path, arg],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
            env=subprocess_env,
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(
            f"R script '{script_path}' timed out after {timeout_sec}s. "
            f"Partial stdout: {e.output!r}, stderr: {e.stderr!r}"
        ) from e

    if proc.returncode != 0:
        # Let callers wrap this if they want more context
        raise subprocess.CalledProcessError(
            returncode=proc.returncode,
            cmd=proc.args,
            output=proc.stdout,
            stderr=proc.stderr,
        )

    return proc


def _parse_json_from_r_stdout(stdout: str) -> dict[str, Any]:
    """
    Extract the final JSON object from mixed R stdout (startup messages, warnings, etc.).
    Returns a raw dict.
    """
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            return json.loads(ln)

    start = stdout.rfind("{")
    end = stdout.rfind("}")
    if start != -1 and end != -1 and start < end:
        return json.loads(stdout[start : end + 1])

    raise ValueError("Expected JSON from R; nothing that looks like a JSON object was found.")


def _safe_parse_json_from_r_stdout(stdout: str, context: str) -> dict[str, Any]:
    """
    Parse JSON from R stdout, raising a ValueError with context on failure.
    """
    try:
        return _parse_json_from_r_stdout(stdout)
    except ValueError as e:
        preview = stdout[:500].replace("\n", "\\n")
        raise ValueError(
            f"Expected JSON from R ({context}); got (preview): {preview}"
        ) from e


#### Helper for tuple range dictionary keys (if you still use it here) ####
def get_range_key(
    d: Dict[str, Any],
    num_plants: float,
) -> Optional[str]:
    """
    Get the key in a dictionary corresponding to a numeric range that
    contains the given number of plants.

    The dictionary `d` is expected to have some keys that are string
    representations of 2‑tuples, e.g. "(0, 10)", "(10, 20)", etc.
    Each such key defines a half-open interval `(lower, upper]`.
    """
    last_key: Optional[str] = None
    max_upper: float = float("-inf")

    for key in d:
        if key.startswith("(") and key.endswith(")"):
            lower, upper = ast.literal_eval(key)

            if upper > max_upper:
                max_upper = upper
                last_key = key

            if lower < num_plants <= upper:
                return key

    return last_key



#### Clark BB wrapper ####
def run_clarke_bb_group_model(
    ty: Sequence[int],
    b: int,
    B: int,
    Nbar: int,
    freq: Sequence[int],
    theta: float,
    R: int,
    startval: Sequence[float],
    se: bool,
    *,
    timeout_sec: float = 120.0,
) -> dict[str, Any]:
    """
    Runs the Clark BB group model via an R script and returns parsed JSON.

    Raises:
      - FileNotFoundError if the script is missing
      - CalledProcessError if the R subprocess fails
      - TimeoutError if R does not complete in time
      - ValueError if stdout is not valid JSON in the expected schema
    """
    r_script_path_bb_cli = str(get_r_script_path())
    if not Path(r_script_path_bb_cli).exists():
        raise FileNotFoundError(f"R script not found: {r_script_path_bb_cli}")

    cmd, env = _pick_rscript_command()

    theta_json = None if isinf(float(theta)) else float(theta)

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

    proc = _run_rscript(
        cmd,
        r_script_path_bb_cli,
        json.dumps(payload, allow_nan=False),
        env=env,
        timeout_sec=timeout_sec,
    )

    return _safe_parse_json_from_r_stdout(proc.stdout, "run_clarke_bb_group_model")


#### DataFrame serialization helpers ####
def _write_df_temp(
    df: pd.DataFrame,
    *,
    use_parquet: bool,
    temp_dir: str,
) -> str:
    """
    Write a DataFrame to a temp file and return its path.
    """
    if use_parquet:
        tmp_file = tempfile.NamedTemporaryFile(
            suffix=".parquet",
            delete=False,
            dir=temp_dir,
        )
        tmp_file.close()
        df.to_parquet(tmp_file.name, engine="pyarrow", index=False)
    else:
        tmp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
            newline="",
            encoding="utf-8",
            dir=temp_dir,
        )
        df.to_csv(tmp_file.name, index=False)
        tmp_file.close()
    return tmp_file.name


def _read_df_from_path(path: str) -> pd.DataFrame:
    """
    Read a DataFrame from a parquet or CSV file.
    """
    if path.endswith(".parquet"):
        return pd.read_parquet(path, engine="pyarrow")
    return pd.read_csv(path)


#### R variable creator wrapper class for engineered features ####
class RVariableCreator:
    """
    Executes R functions from variable_creator.R using the conda-based R wrapper infrastructure.
    """

    # Name of the R script (relative to repo root)
    R_SCRIPT_REL = Path("slippage_model_utils") / "variable_creator.R"

    def __init__(self, repo_root: Optional[str] = None) -> None:
        """
        Initialize variable creator.

        Args:
            repo_root: Root directory of the repository. If None, attempts auto-detection.
        """
        self.repo_root = Path(repo_root) if repo_root is not None else None

    def _get_r_script_path(self) -> Path:
        """
        Resolve the path to variable_creator.R.

        Returns:
            Path to the R script.

        Raises:
            FileNotFoundError: If script cannot be found.
        """
        return get_script_path(self.R_SCRIPT_REL, repo_root=self.repo_root)

    def _call_r_function(
        self,
        function_name: str,
        args: Optional[Dict[str, Any]] = None,
        timeout_sec: float = 120.0,
    ) -> Dict[str, Any]:
        """
        Call a specific R function from variable_creator.R using JSON payload.
        """
        r_script_path = str(self._get_r_script_path())
        cmd, env = _pick_rscript_command()

        payload: Dict[str, Any] = {
            "func_name": function_name,
            "args": args if args is not None else {},
        }

        try:
            proc = _run_rscript(
                cmd,
                r_script_path,
                json.dumps(payload, allow_nan=False),
                env=env,
                timeout_sec=timeout_sec,
            )
        except subprocess.CalledProcessError as e:
            error_msg = (
                f"R function '{function_name}' failed with return code {e.returncode}\n"
                f"{'=' * 60}\n"
                f"STDOUT:\n{e.output}\n"
                f"{'=' * 60}\n"
                f"STDERR:\n{e.stderr}\n"
                f"{'=' * 60}\n"
                f"Command: {' '.join(map(str, e.cmd))}\n"
                f"{'=' * 60}"
            )
            logger.error(error_msg)
            raise

        return _safe_parse_json_from_r_stdout(
            proc.stdout,
            f"function '{function_name}'",
        )

    def _call_r_function_df(
        self,
        function_name: str,
        args: Optional[Dict[str, Any]] = None,
        timeout_sec: float = 120.0,
        use_parquet: bool = True,
    ) -> Dict[str, Any]:
        """
        Call a specific R function from variable_creator.R using temp files for data transfer.
        """
        r_script_path = str(self._get_r_script_path())
        cmd, env = _pick_rscript_command()

        temp_files: list[str] = []
        temp_dir = tempfile.gettempdir()

        try:
            df_paths: Dict[str, str] = {}
            simple_args: Dict[str, Any] = {}

            if args:
                for key, value in args.items():
                    if isinstance(value, pd.DataFrame):
                        file_path = _write_df_temp(
                            value,
                            use_parquet=use_parquet,
                            temp_dir=temp_dir,
                        )
                        df_paths[key] = file_path
                        temp_files.append(file_path)
                    else:
                        simple_args[key] = value

            payload = {
                "func_name": function_name,
                "args": simple_args,
                "df_args": df_paths,
                "use_parquet": use_parquet,
                "temp_dir": temp_dir,
            }

            tmp_json = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
                encoding="utf-8",
                dir=temp_dir,
            )
            json.dump(payload, tmp_json, allow_nan=False)
            tmp_json.close()
            temp_files.append(tmp_json.name)

            proc = _run_rscript(
                cmd,
                r_script_path,
                tmp_json.name,
                env=env,
                timeout_sec=timeout_sec,
            )

        except subprocess.TimeoutExpired as e:
            raise TimeoutError(
                f"R function '{function_name}' timed out after {timeout_sec}s."
            ) from e
        finally:
            # Clean up INPUT temp files only (not R-generated output files)
            for tmp_file in temp_files:
                try:
                    if os.path.exists(tmp_file):
                        os.unlink(tmp_file)
                except Exception as cleanup_err:
                    logger.warning("Could not delete temp file %s: %s", tmp_file, cleanup_err)

        # Parse JSON output and then handle df result files
        result = _safe_parse_json_from_r_stdout(
            proc.stdout,
            f"function '{function_name}' (df)",
        )

        if isinstance(result, dict) and "df_results" in result:
            file_paths = result["df_results"]
            df_results: Dict[str, pd.DataFrame] = {}

            for key, file_path in file_paths.items():
                try:
                    logger.debug("Looking for result file: %s", file_path)
                    if os.path.exists(file_path):
                        df_results[key] = _read_df_from_path(file_path)
                        logger.debug(
                            "Successfully read %d rows for key %s",
                            len(df_results[key]),
                            key,
                        )
                        try:
                            os.unlink(file_path)
                            logger.debug("Cleaned up: %s", file_path)
                        except Exception as e:
                            logger.warning("Could not delete %s: %s", file_path, e)
                    else:
                        logger.warning("File not found: %s", file_path)
                except Exception as read_err:
                    logger.exception("Error reading file %s: %s", file_path, read_err)

            # Merge DataFrame results with other results
            result = {**result, **df_results}
            if "df_results" in result:
                del result["df_results"]

        return result

    def basic_text_preproc(
        self,
        text_field: str,
        suffix_string: Optional[str] = None,
        prefix_string: Optional[str] = None,
        timeout_sec: float = 120.0,
    ) -> str:
        """
        Executes R basic_text_preproc function to clean and normalize text.
        """
        args: Dict[str, Any] = {"text_field": str(text_field)}
        if suffix_string is not None:
            args["suffix_string"] = str(suffix_string)
        if prefix_string is not None:
            args["prefix_string"] = str(prefix_string)

        result = self._call_r_function(
            "basic_text_preproc",
            args=args,
            timeout_sec=timeout_sec,
        )

        if isinstance(result, dict):
            if result.get("status") == "error":
                error_msg = result.get("error", "Unknown error from R")
                raise ValueError(f"R function error: {error_msg}")

            if "processed_text" in result:
                processed_text = result["processed_text"]
                logger.debug("basic_text_preproc result: %r", processed_text)
                return processed_text

        raise ValueError(
            f"Expected dict with 'processed_text' key from R, got: {result}"
        )

    def batch_basic_text_preproc(
            self,
            text_fields: Sequence[str],
            suffix_string: Optional[str] = None,
            prefix_string: Optional[str] = None,
            timeout_sec: float = 120.0,
    ) -> List[str]:
        """
        Batch version of basic_text_preproc that sends a list of text fields to R.

        This version matches the original behavior where the payload is
        written to a temporary JSON file, and the path is passed to R.
        The R script is expected to treat the argument as a file path.
        """
        # Prepare argument payload
        args: Dict[str, Any] = {"text_field": list(map(str, text_fields))}
        if suffix_string is not None:
            args["suffix_string"] = str(suffix_string)
        if prefix_string is not None:
            args["prefix_string"] = str(prefix_string)

        payload: Dict[str, Any] = {
            "func_name": "basic_text_preproc",
            "args": args,
        }

        r_script_path = str(self._get_r_script_path())
        cmd, env = _pick_rscript_command()
        subprocess_env = env if env else None

        # Write payload to temp JSON file
        with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
                encoding="utf-8",
        ) as f:
            json.dump(payload, f, allow_nan=False)
            temp_json_path = f.name

        try:
            # Use raw subprocess.run here instead of _run_rscript, because we
            # want the exact same semantics as the original implementation.
            proc = subprocess.run(
                cmd + [r_script_path, temp_json_path],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_sec,
                env=subprocess_env,
            )
        except subprocess.TimeoutExpired as e:
            # Ensure cleanup of the temp file on timeout
            if os.path.exists(temp_json_path):
                try:
                    os.unlink(temp_json_path)
                except Exception as cleanup_err:
                    logger.warning(
                        "Could not delete temp JSON file %s: %s",
                        temp_json_path,
                        cleanup_err,
                    )
            raise TimeoutError(
                f"R function 'basic_text_preproc' timed out after {timeout_sec}s. "
                f"Partial stdout: {e.output!r}, stderr: {e.stderr!r}"
            ) from e
        finally:
            # Best-effort cleanup of the temp JSON file
            if os.path.exists(temp_json_path):
                try:
                    os.unlink(temp_json_path)
                except Exception as cleanup_err:
                    logger.warning(
                        "Could not delete temp JSON file %s: %s",
                        temp_json_path,
                        cleanup_err,
                    )

        if proc.returncode != 0:
            error_msg = (
                f"R function 'basic_text_preproc' failed with return code {proc.returncode}\n"
                f"{'=' * 60}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"{'=' * 60}\n"
                f"STDERR:\n{proc.stderr}\n"
                f"{'=' * 60}\n"
                f"Command: {' '.join(map(str, proc.args))}\n"
                f"{'=' * 60}"
            )
            logger.error(error_msg)
            raise subprocess.CalledProcessError(
                returncode=proc.returncode,
                cmd=proc.args,
                output=proc.stdout,
                stderr=proc.stderr,
            )

        result = _safe_parse_json_from_r_stdout(
            proc.stdout,
            "basic_text_preproc batch",
        )

        # Your R returns {processed_text: [cleaned names...]}
        if "processed_text" in result:
            return list(result["processed_text"])

        raise ValueError("R did not return processed_text")


    def entity_resolution(
        self,
        df: pd.DataFrame,
        entity_resolution_lookup_table: pd.DataFrame,
        use_parquet: bool = True,
        timeout_sec: float = 120.0,
    ) -> pd.DataFrame:
        """
        Call the R entity_resolution function to create PRODUCER_GROUP_NAME and PRODUCER_GROUP_NAME1.

        Args:
            df: main DataFrame; must contain:
                 - PRODUCER_NAME
                 - PRODUCER_NAME1
                 - QUANTITY
                 - COUNTRY_OF_ORIGIN_NAME
                 - PROPAGATIVE_MATERIAL_TYPE
                 - INSPECTION_NUMBER
            entity_resolution_lookup_table: DataFrame with columns 'name' and 'group'.

        Returns:
            DataFrame with PRODUCER_GROUP_NAME and PRODUCER_GROUP_NAME1 appended.
        """
        required_dt = {
            "PRODUCER_NAME",
            "PRODUCER_NAME1",
            "QUANTITY",
            "COUNTRY_OF_ORIGIN_NAME",
            "PROPAGATIVE_MATERIAL_TYPE",
            "INSPECTION_NUMBER",
        }
        missing_dt = required_dt - set(df.columns)
        if missing_dt:
            raise ValueError(
                f"df missing required columns for entity_resolution: {missing_dt}"
            )

        required_lookup = {"name", "group"}
        missing_lookup = required_lookup - set(entity_resolution_lookup_table.columns)
        if missing_lookup:
            raise ValueError(
                "entity_resolution_lookup_table missing required columns: "
                f"{missing_lookup}"
            )

        args: Dict[str, Any] = {
            "dt": df,
            "entity_resolution_lookup_table": entity_resolution_lookup_table,
        }

        result = self._call_r_function_df(
            "entity_resolution",
            args=args,
            timeout_sec=timeout_sec,
            use_parquet=use_parquet,
        )

        if isinstance(result, dict) and result.get("status") == "error":
            raise ValueError(f"R function error: {result.get('error')}")

        if "result_df" in result:
            result_df = result["result_df"]
            if isinstance(result_df, pd.DataFrame):
                return result_df
            raise ValueError(f"Expected DataFrame, got {type(result_df)}")

        raise ValueError(f"No 'result_df' in result: {result.keys()}")



    def generate_quantity_binaries(
        self,
        df: pd.DataFrame,
        quantity_threshold: float = 200.0,
        group_cols: Optional[List[str]] = None,
        use_parquet: bool = True,
        timeout_sec: float = 120.0,
    ) -> pd.DataFrame:
        """
        Aggregate data by inspection (or custom grouping) and calculate quantity binary features.

        Args:
            df: Input DataFrame with QUANTITY column.
            quantity_threshold: Threshold for binary classification.
            group_cols: Columns to group by (default: ['RISK_UNIT'] on R side).
            use_parquet: If True, use Parquet format for file transfer (faster for large data).
        """
        if "QUANTITY" not in df.columns:
            raise ValueError("DataFrame must contain 'QUANTITY' column")

        args: Dict[str, Any] = {
            "df": df,
            "quantity_threshold": float(quantity_threshold),
        }
        if group_cols is not None:
            args["group_cols"] = list(group_cols)

        result = self._call_r_function_df(
            "generate_quantity_binaries",
            args=args,
            timeout_sec=timeout_sec,
            use_parquet=use_parquet,
        )

        if isinstance(result, dict) and result.get("status") == "error":
            raise ValueError(f"R function error: {result.get('error')}")

        if "result_df" in result:
            result_df = result["result_df"]
            if isinstance(result_df, pd.DataFrame):
                logger.debug("Aggregated to %d groups", result_df.shape[0])
                return result_df
            raise ValueError(f"Expected DataFrame, got {type(result_df)}")

        raise ValueError(f"No 'result_df' in result: {result.keys()}")

    def generate_producer_top_strata_features(
        self,
        df: pd.DataFrame,
        max_strat_count: int = -1,
        min_action_rate: float = -1.0,
        min_records: int = -1,
        use_parquet: bool = True,
        timeout_sec: float = 120.0,
    ) -> pd.DataFrame:
        """
        Add PRODUCER_GROUP_TOP based on PRODUCER_GROUP_NAME1 and action.

        Args:
            df: Input DataFrame with 'action' and 'PRODUCER_GROUP_NAME1'.
        """
        required = {"action", "PRODUCER_GROUP_NAME1"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        args: Dict[str, Any] = {
            "df": df,
            "maxStratCount": int(max_strat_count),
            "minActionRate": float(min_action_rate),
            "minRecords": int(min_records),
        }

        result = self._call_r_function_df(
            "generate_producer_top_strata_features",
            args=args,
            timeout_sec=timeout_sec,
            use_parquet=use_parquet,
        )

        if isinstance(result, dict) and result.get("status") == "error":
            raise ValueError(f"R function error: {result.get('error')}")

        if "result_df" in result:
            result_df = result["result_df"]
            if isinstance(result_df, pd.DataFrame):
                return result_df
            raise ValueError(f"Expected DataFrame, got {type(result_df)}")

        raise ValueError(f"No 'result_df' in result: {result.keys()}")


    def generate_importer_top_strata_features(
        self,
        df: pd.DataFrame,
        max_strat_count: int = -1,
        min_action_rate: float = -1.0,
        min_records: int = -1,
        use_parquet: bool = True,
        timeout_sec: float = 120.0,
    ) -> pd.DataFrame:
        """
        Add IMPORTER_NAME_TOP based on IMPORTER_NAME1 and action.

        Args:
            df: Input DataFrame with 'action' and 'IMPORTER_NAME1'.
        """
        required = {"action", "IMPORTER_NAME1"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        args: Dict[str, Any] = {
            "df": df,
            "maxStratCount": int(max_strat_count),
            "minActionRate": float(min_action_rate),
            "minRecords": int(min_records),
        }

        result = self._call_r_function_df(
            "generate_importer_top_strata_features",
            args=args,
            timeout_sec=timeout_sec,
            use_parquet=use_parquet,
        )

        if isinstance(result, dict) and result.get("status") == "error":
            raise ValueError(f"R function error: {result.get('error')}")

        if "result_df" in result:
            result_df = result["result_df"]
            if isinstance(result_df, pd.DataFrame):
                return result_df
            raise ValueError(f"Expected DataFrame, got {type(result_df)}")

        raise ValueError(f"No 'result_df' in result: {result.keys()}")


