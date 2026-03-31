# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC
from __future__ import annotations
import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping, Sequence, TypedDict, Literal, Tuple, SupportsFloat, Callable
import numpy as np
from scipy.optimize import minimize_scalar
import platform
import sys
from math import isinf
import time
import pandas as pd

import tempfile

# === CONFIG ===
# Name of the conda environment that contains R + required R packages.
# Can be overridden by setting the environment variable POPS_R_CONDA_ENV.
CONDA_ENV_NAME: Optional[str] = os.getenv("POPS_R_CONDA_ENV", "rbb")
REPO_NAME = "plant-inspection-station-simulation"
R_SCRIPT_REL = Path("slippage_model_utils") / "clarke_bb_model.R"

def find_repo_root() -> Optional[Path]:
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
    repo_root = find_repo_root()
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
            Path(local_appdata) / "anaconda" / "Scripts" / "conda.exe",
            Path(local_appdata) / "anaconda3" / "Scripts" / "conda.exe",
            Path(local_appdata) / "miniconda3" / "Scripts" / "conda.exe",
            Path(local_appdata) / "Programs" / "anaconda" / "Scripts" / "conda.exe",
        ]
        # add standard home locations too
        candidates += [
            Path.home() / "anaconda" / "Scripts" / "conda.exe",
            Path.home() / "anaconda3" / "Scripts" / "conda.exe",
            Path.home() / "miniconda3" / "Scripts" / "conda.exe",
        ]
    else:
        # macOS / Linux
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
        Path to the environment directory, or None if not found
    """
    # Check for explicit override first
    override = os.getenv("POPS_CONDA_ENV_DIR")
    if override:
        override_path = Path(override)
        if override_path.exists() and override_path.is_dir():
            return override_path.resolve()

    conda_exe = _find_conda_exe()
    if not conda_exe:
        return None

    # Method 1: Try 'conda env list --json'
    try:
        result = subprocess.run(
            [str(conda_exe), "env", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=True,
        )

        # Parse JSON more robustly - find the JSON object in the output
        stdout = result.stdout.strip()

        # Try to find JSON object boundaries
        json_start = stdout.find('{')
        json_end = stdout.rfind('}')

        if json_start != -1 and json_end != -1:
            json_str = stdout[json_start:json_end + 1]
            env_data = json.loads(json_str)

            # Method 1a: Check 'envs' list (paths only)
            envs = env_data.get("envs", [])
            for env_path in envs:
                env_path_obj = Path(env_path)
                if env_path_obj.name == env_name:
                    return env_path_obj

            # Method 1b: Check 'envs_details' dict (more detailed info)
            envs_details = env_data.get("envs_details", {})
            for env_path_str, details in envs_details.items():
                if details.get("name") == env_name:
                    env_path_obj = Path(env_path_str)
                    if env_path_obj.exists():
                        return env_path_obj

    except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
        # Silent fallback to other methods
        pass

    # Method 2: Try 'conda info --envs' (plain text parsing)
    try:
        result = subprocess.run(
            [str(conda_exe), "info", "--envs"],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=True,
        )

        # Parse output like:
        # # conda environments:
        # #
        # base                  *  C:\Users\user\anaconda3
        # rbb                      C:\Users\user\anaconda3\envs\rbb

        for line in result.stdout.splitlines():
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Split by whitespace, handle asterisk for active env
            parts = line.split()
            if not parts:
                continue

            # First part is env name, last part is path
            curr_env_name = parts[0]
            env_path_str = parts[-1]

            if curr_env_name == env_name:
                env_path_obj = Path(env_path_str)
                if env_path_obj.exists():
                    return env_path_obj

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # Silent fallback to other methods
        pass

    # Method 3: Try common conda environment locations manually
    if conda_exe:
        conda_root = conda_exe.parent.parent  # Go up from Scripts/bin to conda root

        # Common env locations relative to conda installation
        candidates = [
            conda_root / "envs" / env_name,  # Standard location
            Path.home() / ".conda" / "envs" / env_name,  # User envs
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
                # Verify it's actually a conda env by checking for key files
                if platform.system() == "Windows":
                    if (candidate / "Scripts" / "activate.bat").exists() or \
                            (candidate / "python.exe").exists():
                        return candidate
                else:
                    if (candidate / "bin" / "activate").exists() or \
                            (candidate / "bin" / "python").exists():
                        return candidate

    return None


def _pick_rscript_command() -> Tuple[List[str], Dict[str, str]]:
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

    # Test the conda environment detection
    conda_exe = _find_conda_exe()

    # On Windows, we need to directly invoke Rscript with proper env vars
    # On Unix, conda run works better
    if platform.system() == "Windows":
        env_dir = _find_conda_env_dir(CONDA_ENV_NAME)
        if not env_dir:
            # Provide detailed debugging information
            try:
                result = subprocess.run(
                    [str(conda_exe), "env", "list"],
                    capture_output=True,
                    text=True,
                    timeout=30.0,
                )
                available_envs = result.stdout
            except:
                available_envs = "(could not retrieve environment list)"

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

        # Build the command to directly invoke Rscript
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

        # Build environment variables
        env = os.environ.copy()

        # Critical conda DLL locations for R + packages
        prepend = [
            str(env_dir / "Library" / "bin"),
            str(env_dir / "Scripts"),
            str(env_dir),
        ]
        env["PATH"] = os.pathsep.join(prepend + [env.get("PATH", "")])

        # Help R find its home
        env["R_HOME"] = str(env_dir / "Lib" / "R")

        return [str(rscript)], env

    else:
        # On Unix, conda run works well
        return [str(conda_exe), "run", "-n", CONDA_ENV_NAME, "Rscript"], {}

# ---- Nested schema for `optim` ----
class OptimResult(TypedDict):
    value: float
    par: Tuple[float, float]                       # two parameters
    counts: Tuple[int, Literal["NA"]]             # R's optim: (fn_evals, "NA")
    convergence: int                               # 0 means success
    message: Mapping[str, Any]                     # R often returns an empty list/dict here
    hessian: Tuple[Tuple[float, float], Tuple[float, float]]  # 2x2 matrix



def _parse_json_from_r_stdout(stdout: str) -> dict[str, Any]:
    """
    Extract the final JSON object from mixed R stdout (startup messages, warnings, etc.).
    Returns a raw dict
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
) -> dict[str, Any]:
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

    cmd, env = _pick_rscript_command()

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
        # Use custom env if provided (Windows), otherwise use current env
        subprocess_env = env if env else None

        proc: subprocess.CompletedProcess[str] = subprocess.run(
            cmd + [str(Path(r_script_path_bb_cli)), json.dumps(payload, allow_nan=False)],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
            env=subprocess_env,
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
        return _parse_json_from_r_stdout(proc.stdout)
    except ValueError as e:
        preview = proc.stdout[:500].replace("\n", "\\n")
        raise ValueError(f"Expected JSON from R; got (preview): {preview}") from e





class RVariableCreator:
    """
    Executes R functions from variable_creator.R using the conda-based R wrapper infrastructure.
    """

    # Name of the R script (relative to repo root)
    R_SCRIPT_REL = Path("slippage_model_utils") / "variable_creator.R"

    def __init__(self, repo_root: Optional[str] = None) -> None:
        """
        Initialize variable creator

        Args:
            repo_root: Root directory of the repository. If None, attempts auto-detection
        """
        self.repo_root = Path(repo_root) if repo_root is not None else None
        self.function_execution_times: List[Tuple[str, float]] = []

        # List of functions to execute: (name, method)
        self.functions_to_execute: List[Tuple[str, Callable[..., Any]]] = [
            ('basic_text_preproc', self.basic_text_preproc),
            ('generate_quantity_binaries', self.generate_quantity_binaries),
        ]

    def _get_r_script_path(self) -> Path:
        """
        Resolve the path to variable_creator.R

        Returns:
            Path to the R script

        Raises:
            FileNotFoundError: If script cannot be found
        """
        # Try from provided repo root first
        if self.repo_root:
            candidate = self.repo_root / self.R_SCRIPT_REL
            if candidate.exists():
                return candidate.resolve()

        # Try using the existing repo root finder
        repo_root = find_repo_root()
        if repo_root:
            candidate = repo_root / self.R_SCRIPT_REL
            if candidate.exists():
                return candidate.resolve()

        raise FileNotFoundError(
            f"Could not locate 'variable_creator.R'.\n"
            f"Searched in: {self.repo_root if self.repo_root else 'auto-detected locations'}\n"
            f"Expected relative path: {self.R_SCRIPT_REL}"
        )


    def _call_r_function(
            self,
            function_name: str,
            args: Optional[Dict[str, Any]] = None,
            timeout_sec: float = 120.0
    ) -> Dict[str, Any]:
        """
        Call a specific R function from variable_creator.R
        """
        r_script_path = str(self._get_r_script_path())

        # Get the Rscript command and environment (uses existing infrastructure)
        cmd, env = _pick_rscript_command()

        # Build payload - use "func_name" instead of "function" (reserved in R)
        payload: Dict[str, Any] = {
            "func_name": function_name,  # Changed from "function"
            "args": args if args is not None else {}
        }

        try:
            # Use custom env if provided (Windows), otherwise use current env
            subprocess_env = env if env else None

            proc = subprocess.run(
                cmd + [r_script_path, json.dumps(payload, allow_nan=False)],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_sec,
                env=subprocess_env,
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(
                f"R function '{function_name}' timed out after {timeout_sec}s. "
                f"Partial stdout: {e.output!r}, stderr: {e.stderr!r}"
            ) from e

        if proc.returncode != 0:
            # Enhanced error message showing R's actual error
            error_msg = (
                f"R function '{function_name}' failed with return code {proc.returncode}\n"
                f"{'=' * 60}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"{'=' * 60}\n"
                f"STDERR:\n{proc.stderr}\n"
                f"{'=' * 60}\n"
                f"Command: {' '.join(proc.args)}\n"
                f"{'=' * 60}"
            )
            print(error_msg)  # Print for immediate visibility
            raise subprocess.CalledProcessError(
                returncode=proc.returncode,
                cmd=proc.args,
                output=proc.stdout,
                stderr=proc.stderr,
            )

        # Parse JSON output (reuses existing parser)
        try:
            return _parse_json_from_r_stdout(proc.stdout)
        except ValueError as e:
            preview = proc.stdout[:500].replace("\n", "\\n")
            raise ValueError(
                f"Expected JSON from R function '{function_name}'; got (preview): {preview}"
            ) from e

    def _call_r_function_df(
            self,
            function_name: str,
            args: Optional[Dict[str, Any]] = None,
            timeout_sec: float = 120.0,
            use_parquet: bool = True
    ) -> Dict[str, Any]:
        """
        Call a specific R function from variable_creator.R using temp files for data transfer.
        """
        r_script_path = str(self._get_r_script_path())
        cmd, env = _pick_rscript_command()

        temp_files = []
        temp_dir = tempfile.gettempdir()  # 🟢 Get Python's temp directory

        try:
            # Separate DataFrames from other arguments
            df_paths = {}
            simple_args = {}

            if args:
                for key, value in args.items():
                    if isinstance(value, pd.DataFrame):
                        if use_parquet:
                            # Use temp_dir explicitly
                            tmp_file = tempfile.NamedTemporaryFile(
                                suffix='.parquet',
                                delete=False,
                                dir=temp_dir  # 🟢 Specify directory
                            )
                            tmp_file.close()
                            value.to_parquet(tmp_file.name, engine='pyarrow', index=False)
                        else:
                            tmp_file = tempfile.NamedTemporaryFile(
                                mode='w',
                                suffix='.csv',
                                delete=False,
                                newline='',
                                encoding='utf-8',
                                dir=temp_dir  # 🟢 Specify directory
                            )
                            value.to_csv(tmp_file.name, index=False)
                            tmp_file.close()

                        df_paths[key] = tmp_file.name
                        temp_files.append(tmp_file.name)
                    else:
                        simple_args[key] = value

            payload = {
                "func_name": function_name,
                "args": simple_args,
                "df_args": df_paths,
                "use_parquet": use_parquet,
                "temp_dir": temp_dir  # 🟢 Pass temp directory to R
            }

            tmp_json = tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.json',
                delete=False,
                encoding='utf-8',
                dir=temp_dir  # 🟢 Specify directory
            )
            json.dump(payload, tmp_json, allow_nan=False)
            tmp_json.close()
            temp_files.append(tmp_json.name)

            subprocess_env = env if env else None

            proc = subprocess.run(
                cmd + [r_script_path, tmp_json.name],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_sec,
                env=subprocess_env,
            )

        except subprocess.TimeoutExpired as e:
            raise TimeoutError(
                f"R function '{function_name}' timed out after {timeout_sec}s."
            ) from e
        finally:
            # Clean up INPUT temp files only (not output files yet!)
            for tmp_file in temp_files:
                try:
                    if os.path.exists(tmp_file):
                        os.unlink(tmp_file)
                except Exception as cleanup_err:
                    print(f"Warning: Could not delete temp file {tmp_file}: {cleanup_err}")

        if proc.returncode != 0:
            error_msg = (
                f"R function '{function_name}' failed with return code {proc.returncode}\n"
                f"STDOUT: {proc.stdout}\n"
                f"STDERR: {proc.stderr}"
            )
            print(error_msg)
            raise subprocess.CalledProcessError(
                returncode=proc.returncode,
                cmd=proc.args,
                output=proc.stdout,
                stderr=proc.stderr,
            )

        # Parse JSON output from R
        try:
            result = _parse_json_from_r_stdout(proc.stdout)

            # Check if R returned file paths for DataFrames
            if isinstance(result, dict) and "df_results" in result:
                file_paths = result["df_results"]
                df_results = {}

                for key, file_path in file_paths.items():
                    try:
                        # 🟢 Add debug output
                        print(f"Looking for result file: {file_path}")
                        print(f"File exists: {os.path.exists(file_path)}")

                        if os.path.exists(file_path):
                            if file_path.endswith('.parquet'):
                                df_results[key] = pd.read_parquet(file_path, engine='pyarrow')
                            else:
                                df_results[key] = pd.read_csv(file_path)

                            print(f"Successfully read {len(df_results[key])} rows")

                            # 🟢 Clean up AFTER reading
                            try:
                                os.unlink(file_path)
                                print(f"Cleaned up: {file_path}")
                            except Exception as e:
                                print(f"Warning: Could not delete {file_path}: {e}")
                        else:
                            print(f"Warning: File not found: {file_path}")
                            # 🟢 List files in temp directory for debugging
                            if os.path.exists(temp_dir):
                                print(f"Files in {temp_dir}:")
                                for f in os.listdir(temp_dir)[:10]:  # Show first 10
                                    print(f"  - {f}")

                    except Exception as read_err:
                        print(f"Error reading file {file_path}: {read_err}")
                        import traceback
                        traceback.print_exc()

                # Merge DataFrame results with other results
                result = {**result, **df_results}
                if "df_results" in result:
                    del result["df_results"]

            return result

        except ValueError as e:
            preview = proc.stdout[:500].replace("\n", "\\n")
            raise ValueError(
                f"Expected JSON from R function '{function_name}'; got (preview): {preview}"
            ) from e

    def basic_text_preproc(
            self,
            text_field: str,
            suffix_string: Optional[str] = None,
            prefix_string: Optional[str] = None
    ) -> str:
        """
        Executes R basic_text_preproc function to clean and normalize text

        Args:
            text_field: Text string to process
            suffix_string: Optional custom suffix patterns to remove (regex)
            prefix_string: Optional custom prefix patterns to remove (regex)

        Returns:
            Processed text string

        Raises:
            ValueError: If R function fails or returns unexpected format
        """
        try:
            # Prepare arguments for R function with explicit type conversion
            args: Dict[str, Any] = {
                "text_field": str(text_field)
            }

            if suffix_string is not None:
                args["suffix_string"] = str(suffix_string)
            if prefix_string is not None:
                args["prefix_string"] = str(prefix_string)

            # Call R function
            result = self._call_r_function("basic_text_preproc", args)

            # Check if R returned an error
            if isinstance(result, dict):
                if result.get("status") == "error":
                    error_msg = result.get("error", "Unknown error from R")
                    raise ValueError(f"R function error: {error_msg}")

                if "processed_text" in result:
                    processed_text = result["processed_text"]
                    print(f"basic_text_preproc result: '{processed_text}'")
                    return processed_text

            raise ValueError(
                f"Expected dict with 'processed_text' key from R, got: {result}"
            )

        except Exception as e:
            print(f"Error in execution of basic_text_preproc function: {e}")
            raise

    def batch_basic_text_preproc(self, text_fields, suffix_string=None, prefix_string=None, timeout_sec=120.0):
        # Prepare argument payload
        args = {"text_field": list(map(str, text_fields))}
        if suffix_string is not None:
            args["suffix_string"] = str(suffix_string)
        if prefix_string is not None:
            args["prefix_string"] = str(prefix_string)
        payload = {
            "func_name": "basic_text_preproc",
            "args": args
        }

        # Write to temp file
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(payload, f, allow_nan=False)
            temp_json_path = f.name

        r_script_path = str(self._get_r_script_path())
        cmd, env = _pick_rscript_command()
        subprocess_env = env if env else None

        try:
            proc = subprocess.run(
                cmd + [r_script_path, temp_json_path],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_sec,
                env=subprocess_env,
            )
        finally:
            # Clean up temp file
            if os.path.exists(temp_json_path):
                os.unlink(temp_json_path)

        if proc.returncode != 0:
            error_msg = (
                f"R function 'basic_text_preproc' failed with return code {proc.returncode}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}\n"
                f"Command: {' '.join(proc.args)}\n"
            )
            print(error_msg)
            raise subprocess.CalledProcessError(
                returncode=proc.returncode,
                cmd=proc.args,
                output=proc.stdout,
                stderr=proc.stderr,
            )

        result = _parse_json_from_r_stdout(proc.stdout)
        # Your R returns {processed_text: [cleaned names...]}
        if "processed_text" in result:
            return result["processed_text"]
        raise ValueError("R did not return processed_text")

    def generate_quantity_binaries(
            self,
            df: pd.DataFrame,
            quantity_threshold: float = 200.0,
            group_cols: Optional[List[str]] = None,
            use_parquet: bool = True
    ) -> pd.DataFrame:
        """
        Aggregate data by inspection (or custom grouping) and calculate quantity binary features

        Args:
            df: Input DataFrame with QUANTITY column
            quantity_threshold: Threshold for binary classification
            group_cols: Columns to group by (default: ['RISK_UNIT'])
            use_parquet: If True, use Parquet format for file transfer (faster for large data)

        Returns:
            Aggregated DataFrame with quantity features
        """
        try:
            if 'QUANTITY' not in df.columns:
                raise ValueError("DataFrame must contain 'QUANTITY' column")

            args: Dict[str, Any] = {
                "df": df,
                "quantity_threshold": float(quantity_threshold)
            }

            if group_cols is not None:
                args["group_cols"] = list(group_cols)

            # Call R function with parquet option
            result = self._call_r_function_df(
                "generate_quantity_binaries",
                args,
                use_parquet=use_parquet
            )

            if isinstance(result, dict) and result.get("status") == "error":
                raise ValueError(f"R function error: {result.get('error')}")

            if "result_df" in result:
                result_df = result["result_df"]
                if isinstance(result_df, pd.DataFrame):
                    print(f"Aggregated to {result_df.shape[0]} groups")
                    return result_df
                else:
                    raise ValueError(f"Expected DataFrame, got {type(result_df)}")
            else:
                raise ValueError(f"No 'result_df' in result: {result.keys()}")

        except Exception as e:
            print(f"Error in generate_quantity_binaries: {e}")
            raise

    def run_all(self) -> Dict[str, Any]:
        """
        Execute all registered functions and track their execution times

        Returns:
            Dictionary containing execution summary with timing information
        """
        results: Dict[str, Any] = {}
        self.function_execution_times = []

        print(f"Executing {len(self.functions_to_execute)} functions...")

        for func_name, func in self.functions_to_execute:
            print(f"\n--- Executing {func_name} ---")
            start_time = time.time()

            try:
                success = func()
                elapsed = time.time() - start_time

                self.function_execution_times.append((func_name, elapsed))
                results[func_name] = {
                    "success": success,
                    "execution_time": elapsed
                }

                print(f"✓ {func_name} completed in {elapsed:.2f}s")

            except Exception as e:
                elapsed = time.time() - start_time
                self.function_execution_times.append((func_name, elapsed))
                results[func_name] = {
                    "success": False,
                    "execution_time": elapsed,
                    "error": str(e)
                }
                print(f"✗ {func_name} failed after {elapsed:.2f}s: {e}")

        # Print summary
        total_time = sum(t for _, t in self.function_execution_times)
        print(f"\n{'=' * 50}")
        print(f"Execution Summary:")
        print(f"  Total functions: {len(self.functions_to_execute)}")
        print(f"  Successful: {sum(1 for r in results.values() if r['success'])}")
        print(f"  Failed: {sum(1 for r in results.values() if not r['success'])}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"{'=' * 50}\n")

        return results

