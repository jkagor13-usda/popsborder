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





class VariableCreator:
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
        self.functions_to_execute: List[Tuple[str, Callable[..., bool]]] = [
            ('Function1', self.function1),
            ('Function2', self.function2),
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
        repo_root = _find_repo_root()
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

        Args:
            function_name: Name of the R function to call
            args: Dictionary of arguments to pass to the R function
            timeout_sec: Maximum execution time in seconds

        Returns:
            Dictionary containing the R function's return value

        Raises:
            FileNotFoundError: If R script not found
            subprocess.CalledProcessError: If R subprocess fails
            TimeoutError: If execution exceeds timeout
            ValueError: If R output is not valid JSON
        """
        r_script_path = str(self._get_r_script_path())

        # Get the Rscript command and environment (uses existing infrastructure)
        cmd, env = _pick_rscript_command()

        # Build payload with explicit type conversions (matching run_clarke_bb_group_model pattern)
        payload: Dict[str, Any] = {
            "function": function_name,
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

    def function1(self, param1: Optional[int] = None, param2: Optional[int] = None) -> bool:
        """
        Executes R function1 to create variable 1

        Args:
            param1: Optional integer parameter
            param2: Optional integer parameter

        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare arguments for R function with explicit type conversion
            args: Dict[str, Any] = {}
            if param1 is not None:
                args["param1"] = int(param1)
            if param2 is not None:
                args["param2"] = int(param2)

            # Call R function
            result = self._call_r_function("function1", args)

            # Process result as needed
            print(f"Function1 result: {result}")
            return True

        except Exception as e:
            print(f"Error in function1: {e}")
            return False

    def function2(self, data: Optional[List[float]] = None) -> bool:
        """
        Executes R function2 to create variable 2

        Args:
            data: Optional list of numeric data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare arguments for R function with explicit type conversion
            args: Dict[str, Any] = {}
            if data is not None:
                args["data"] = [float(x) for x in data]

            # Call R function
            result = self._call_r_function("function2", args)

            # Process result as needed
            print(f"Function2 result: {result}")
            return True

        except Exception as e:
            print(f"Error in function2: {e}")
            return False

    def function3(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes R function3 to add a column to a dataframe

        Args:
            df: Pandas DataFrame to process

        Returns:
            Pandas DataFrame with the new column added
        """
        try:
            # Convert pandas DataFrame to dict (orient='list' matches R's column format)
            df_dict = df.to_dict(orient='list')

            # Prepare arguments for R function
            args: Dict[str, Any] = {
                "df": df_dict
            }

            # Call R function
            result = self._call_r_function("function3", args)

            # Convert result back to pandas DataFrame
            # Result should be a dict with column names as keys
            if isinstance(result, dict):
                result_df = pd.DataFrame(result)
                print(f"Function3 result: Added column(s), shape: {result_df.shape}")
                return result_df
            else:
                raise ValueError(f"Expected dict from R, got {type(result)}")

        except Exception as e:
            print(f"Error in function3: {e}")
            raise

    def function4(self, df: pd.DataFrame, operation: str = "sum", multiplier: float = 1.0) -> pd.DataFrame:
        """
        Executes R function3 with custom parameters

        Args:
            df: Pandas DataFrame to process
            operation: Type of operation ('sum', 'product', 'mean')
            multiplier: Multiplier to apply

        Returns:
            Pandas DataFrame with the new column added
        """
        try:
            df_dict = df.to_dict(orient='list')

            args: Dict[str, Any] = {
                "df": df_dict,
                "operation": operation,
                "multiplier": float(multiplier)
            }

            result = self._call_r_function("function4", args)

            if isinstance(result, dict):
                return pd.DataFrame(result)
            else:
                raise ValueError(f"Expected dict from R, got {type(result)}")

        except Exception as e:
            print(f"Error in function3: {e}")
            raise

    def run(self) -> Dict[str, Any]:
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



