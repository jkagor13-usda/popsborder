from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from popsborder.generator import SyntheticConsignmentDataGenerator, save_to_csv
from popsborder.inputs import (
    load_compliance_lookup_csv,
    load_configuration,
    load_scenario_table,
    text_to_value,
)
from popsborder.outputs import save_scenario_result_to_pandas
from popsborder.scenarios import run_scenarios as run_scenarios_fn
from slippage_model_utils.clarke_model_support_functions import gen_clarke_model_inputs
from slippage_model_utils.clarke_r_script_wrapper import run_clarke_bb_group_model


@dataclass
class SyntheticOptions:
    """Placeholder for interface compatibility (not used in simplified pipeline)."""

    n_samples: int = 10
    sampling_method: str = "sequential"


DEFAULT_DATA_DIR = Path("data_input")
RESULT_COLUMNS = [
    "num_inspections",
    "intercepted",
    "false_neg",
    "missing",
    "true_contamination_rate",
    "avg_missed_contamination_rate",
    "max_missed_contamination_rate",
    "total_missed_contaminants",
    "total_intercepted_contaminants",
]
CONFIG_COLUMNS = [
    "contamination/contamination_unit",
    "contamination/contamination_rate/distribution",
    "contamination/contamination_rate/value",
    "contamination/arrangement",
    "inspection/sample_strategy",
    "inspection/proportion/value",
    "inspection/tolerance_level",
    "name",
]

CONS_FILENAME = "consignment_uploaded_rbs_data.csv"
COMPLIANCE_FILENAME = "compliance_table.csv"
CONFIG_FILENAME = "config.yml"
SCENARIO_FILENAME = "scenario_table.csv"



@dataclass
class SlippagePaths:
    """Container for all filesystem paths used by the slippage pipeline."""

    data_dir: Path = DEFAULT_DATA_DIR
    config: Path = DEFAULT_DATA_DIR / "config.yml"
    scenario_table: Path = DEFAULT_DATA_DIR / "pis_contaminate_scenarios.csv"
    compliance_lookup: Path = DEFAULT_DATA_DIR / "compliance_table.csv"
    pis_data: Optional[Path] = None
    rbs_data: Optional[Path] = None
    synthetic_seed: Optional[Path] = None
    synthetic_output: Path = Path("tmp") / "synthetic_consignment_data.csv"
    output_dir: Path = Path("output")


@dataclass
class ExperimentPaths:
    """Container for inputs in a specific experiment folder."""

    experiment_dir: Path
    scenario_table: Path
    consignment: Optional[Path] = None
    compliance: Optional[Path] = None
    config: Optional[Path] = None


@dataclass
class ClarkeFit:
    """Fitted contamination parameters."""

    alpha: float
    beta: float
    theta: float
    raw_result: Dict[str, Any]


@dataclass
class PipelineResult:
    """Outputs from a pipeline run."""

    contamination_fit: ClarkeFit
    scenario_results: pd.DataFrame
    config: Dict[str, Any]
    compliance_table: Any
    pis_data: pd.DataFrame
    rbs_data: pd.DataFrame
    num_consignments: int


def create_default_paths(base_dir: Path = DEFAULT_DATA_DIR) -> SlippagePaths:
    """Return default data locations using ``base_dir``."""
    pis_path = base_dir / "synthetic_pis_data.csv"
    rbs_enriched = base_dir / "synthetic_rbs_calc_data_enriched.csv"
    rbs_plain = base_dir / "synthetic_rbs_calc_data.csv"
    rbs_path = rbs_enriched if rbs_enriched.exists() else (rbs_plain if rbs_plain.exists() else None)
    synthetic_seed = rbs_path if rbs_path and rbs_path.exists() else None
    return SlippagePaths(
        data_dir=base_dir,
        pis_data=pis_path if pis_path.exists() else None,
        rbs_data=rbs_path,
        synthetic_seed=synthetic_seed,
    )


def load_scenario_dataframe(path: Path, *, dtype: str = "object"):
    """Load the slippage scenario table using the shared popsborder loader and return a DataFrame."""
    scenarios = load_scenario_table(path)
    return scenarios

def generate_synthetic_data(
    seed_path: Path,
    output_path: Path,
    options: SyntheticOptions,
) -> pd.DataFrame:
    """Generate synthetic consignment data and persist it."""
    if seed_path is None or not Path(seed_path).exists():
        raise FileNotFoundError(f"Seed data not found at {seed_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = SyntheticConsignmentDataGenerator(seed_path)
    synth_data = generator.generate_from_input_data(
        n_samples=options.n_samples,
        sampling_method=options.sampling_method,
    )
    save_to_csv(synth_data, filename=output_path)
    return synth_data


def _infer_num_consignments(consignment_path: Optional[Path]) -> int:
    """Estimate consignments by counting unique inspection IDs in the provided consignment file."""
    if consignment_path and Path(consignment_path).exists():
        try:
            df = pd.read_csv(consignment_path)
            id_cols = [
                "INSPECTION_NUMBER",
                "INSPECTION_ID",
                "inspection_number",
                "inspection_id",
            ]
            for col in id_cols:
                if col in df.columns:
                    n_unique = df[col].nunique(dropna=True)
                    if n_unique > 0:
                        return max(1, int(n_unique))
            return max(1, len(df))
        except Exception:
            pass
    return 1



def fit_contamination_distribution(
    pis_data_path: Path,
    rbs_data_path: Path,
) -> Tuple[ClarkeFit, pd.DataFrame, pd.DataFrame]:
    """Fit contamination parameters using the Clarke beta-binomial model."""
    if pis_data_path is None or not Path(pis_data_path).exists():
        raise FileNotFoundError("PIS action data not provided. Upload on Page 2 - Contamination Fit.")
    if rbs_data_path is None or not Path(rbs_data_path).exists():
        raise FileNotFoundError("RBS calculator data not provided. Upload on Page 1 or Page 2.")
    pis_df = pd.read_csv(pis_data_path)
    if "action" not in pis_df.columns:
        pis_df["action"] = 0
    rbs_df = pd.read_csv(rbs_data_path)
    # Ensure we have overlapping inspection IDs; if INSPECTION_ID is missing/empty, fall back to INSPECTION_NUMBER.
    for df in (pis_df, rbs_df):
        if "INSPECTION_ID" not in df.columns and "INSPECTION_NUMBER" in df.columns:
            df["INSPECTION_ID"] = df["INSPECTION_NUMBER"]
        elif "INSPECTION_ID" in df.columns and "INSPECTION_NUMBER" in df.columns:
            df["INSPECTION_ID"] = df["INSPECTION_ID"].fillna(df["INSPECTION_NUMBER"])

    if "INSPECTION_ID" not in pis_df.columns or "INSPECTION_ID" not in rbs_df.columns:
        raise ValueError("PIS/RBS files must include INSPECTION_ID or INSPECTION_NUMBER to align for fitting.")
    shared_ids = set(pis_df["INSPECTION_ID"]).intersection(set(rbs_df["INSPECTION_ID"]))
    if not shared_ids:
        raise ValueError(
            "No shared inspection IDs between PIS and RBS files. "
            "Ensure both files reference the same consignments."
        )
    try:
        inputs = gen_clarke_model_inputs(pis_df, rbs_df)
        print(inputs)
    except StopIteration as exc:
        raise ValueError(
            "Fitting failed: no compatible scenarios found between PIS and RBS data. "
            "Verify that shared INSPECTION_NUMBER rows contain sampling/plant quantities."
        ) from exc

    if not inputs.freq or sum(inputs.freq) <= 0:
        raise ValueError("Fitting failed: no frequency counts available after aligning PIS and RBS data.")

    # Use Clarke inputs, but if any are invalid fall back to averages from RBS.
    b_val, B_val, Nbar_val = inputs.b, inputs.B, inputs.Nbar

    def _fallback_mean(df: pd.DataFrame, col: str, default: float = 1.0) -> float:
        if col in df.columns and df[col].notna().any():
            return float(pd.to_numeric(df[col], errors="coerce").dropna().mean())
        return default

    fb_b = _fallback_mean(rbs_df, "TOTAL_SAMPLING_UNITS", 1.0)
    fb_nbar = _fallback_mean(rbs_df, "TOTAL_PLANT_QUANTITY", 1.0)
    fb_B = max(1, len(shared_ids))

    if not (np.isfinite(b_val) and b_val > 0):
        b_val = fb_b
    if not (np.isfinite(Nbar_val) and Nbar_val > 0):
        Nbar_val = fb_nbar
    if not (np.isfinite(B_val) and B_val > 0):
        B_val = fb_B

    if not (np.isfinite(b_val) and np.isfinite(B_val) and np.isfinite(Nbar_val) and b_val > 0 and B_val > 0 and Nbar_val > 0):
        raise ValueError(
            "Fitting failed: computed b/B/Nbar are invalid (NaN or <=0) even after fallback. "
            "Check RBS calculator columns (TOTAL_SAMPLING_UNITS, TOTAL_PLANT_QUANTITY) and PIS/RBS alignment."
        )
    # Always force theta to infinity when invoking the Clarke model
    theta_val = float("inf")

    # Stabilize start values (R can fail when both are zero)
    start_vals = [float(v) for v in (inputs.start_val or [])]
    if not start_vals or all(abs(v) < 1e-9 for v in start_vals):
        start_vals = [0.1, 0.1]

    try:
        result = run_clarke_bb_group_model(
            inputs.ty,
            int(round(b_val)),
            int(round(B_val)),
            int(round(Nbar_val)),
            inputs.freq,
            theta_val,
            inputs.R,
            start_vals,
            inputs.se,
        )
    except subprocess.CalledProcessError as exc:
        stderr_preview = (exc.stderr or "")[:500].replace("\n", " | ")
        stdout_preview = (exc.output or "")[:500].replace("\n", " | ")
        raise ValueError(
            f"Fitting failed in R (returncode {exc.returncode}). "
            f"stdout: {stdout_preview} stderr: {stderr_preview}"
        ) from exc
    alpha_val = float(result.get("alpha", 0) or 0)
    beta_val = float(result.get("beta", 0) or 0)
    alpha_val = max(alpha_val, 1e-3)
    beta_val = max(beta_val, 1e-3)
    fit = ClarkeFit(alpha=alpha_val, beta=beta_val, theta=theta_val, raw_result=result)
    return fit, pis_df, rbs_df



def run_slippage_pipeline(
    exp_paths: ExperimentPaths,
    *,
    seed: int = 42,
    num_simulations: int = 1,
) -> PipelineResult:
    """Execute the pipeline using files inside a specific experiment folder."""

    experiment_dir = exp_paths.experiment_dir
    scenario_table = exp_paths.scenario_table
    consignment_path = exp_paths.consignment
    compliance_path = exp_paths.compliance
    config_path = exp_paths.config or DEFAULT_DATA_DIR / "config.yml"

    scenarios = load_scenario_dataframe(scenario_table)
    if isinstance(scenarios, pd.DataFrame):
        scenarios = scenarios.to_dict(orient="records")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Scenario table is empty or unreadable.")

    def _resolve_first_path(records: List[Dict[str, Any]], key: str) -> Optional[Path]:
        for rec in records:
            val = rec.get(key)
            if val not in (None, "", "None", "nan"):
                return Path(str(val))
        return None

    def _with_experiment_dir(p: Path) -> Path:
        return p if p.is_absolute() else (experiment_dir / p.name if not p.exists() else p)

    # Resolve consignment/compliance paths (prefer explicit exp_paths, else first scenario reference)
    cons_candidates: List[Path] = []
    if consignment_path and consignment_path.exists():
        cons_candidates.append(consignment_path)
    for rec in scenarios:
        cand_val = rec.get("consignment/input_file/file_name")
        if cand_val not in (None, "", "None", "nan"):
            cand = _with_experiment_dir(Path(str(cand_val)))
            if cand.exists():
                cons_candidates.append(cand)
    cons_path = cons_candidates[0] if cons_candidates else None
    if not cons_path:
        raise FileNotFoundError(f"Consignment file missing in experiment folder {experiment_dir}.")

    comp_path = compliance_path if compliance_path and compliance_path.exists() else None
    if not comp_path:
        cand = _resolve_first_path(scenarios, "inspection/compliance_table/file_name")
        if cand:
            cand = _with_experiment_dir(cand)
            comp_path = cand if cand.exists() else None

    if not config_path or not Path(config_path).exists():
        raise FileNotFoundError(f"Config file missing in experiment folder {experiment_dir} (expected {CONFIG_FILENAME}).")
    config = load_configuration(config_path)

    cons_path_str = str(cons_path).replace("\\", "/")
    comp_lookup_path = comp_path if comp_path and comp_path.exists() else DEFAULT_DATA_DIR / "compliance_table.csv"
    if not comp_lookup_path or not Path(comp_lookup_path).exists():
        raise FileNotFoundError(f"Compliance table not found for experiment {experiment_dir}. Expected {COMPLIANCE_FILENAME}.")
    comp_lookup_str = str(comp_lookup_path).replace("\\", "/")

    # Normalize scenario inspection and contamination parameters to avoid over-sampling / missing rates
    norm_scenarios = []
    for rec in scenarios:
        rec = rec.copy()
        try:
            prop_val = float(rec.get("inspection/proportion/value", 0) or 0)
        except Exception:
            prop_val = 0.0
        if prop_val <= 0:
            prop_val = 0.02
        if prop_val > 1:
            prop_val = 1.0
        rec["inspection/proportion/value"] = prop_val
        if not rec.get("inspection/sample_strategy"):
            rec["inspection/sample_strategy"] = "rbs"
        # Contamination rate handling
        cont_rate_key = "contamination/contamination_rate/value"
        if cont_rate_key in rec and rec.get(cont_rate_key) not in ("", None, "None", "nan"):
            try:
                rec[cont_rate_key] = float(rec[cont_rate_key])
            except Exception:
                rec[cont_rate_key] = rec[cont_rate_key]
        else:
            # If no explicit rate, leave it to config or beta-binomial params
            pass
        alpha_key = "contamination/contamination_rate/beta_binomial_parameters/alpha"
        beta_key = "contamination/contamination_rate/beta_binomial_parameters/beta"
        for k, default in ((alpha_key, 0.01), (beta_key, 5.0)):
            if k in rec:
                try:
                    rec[k] = float(rec.get(k) or default)
                except Exception:
                    rec[k] = default
        if "contamination/contamination_unit" not in rec or not rec.get("contamination/contamination_unit"):
            rec["contamination/contamination_unit"] = "plant"
        norm_scenarios.append(rec)
    scenarios = norm_scenarios

    # Ensure config points to the experiment consignment and compliance
    config.setdefault("consignment", {}).setdefault("input_file", {})
    config["consignment"]["input_file"]["rbs_file_name"] = cons_path_str
    config["consignment"]["input_file"]["file_name"] = cons_path_str
    config.setdefault("inspection", {}).setdefault("compliance_table", {})
    config["inspection"]["compliance_table"]["file_name"] = comp_lookup_str

    # Use the total consignment count across all referenced consignment files (deduped by path)
    unique_cons_files = []
    seen = set()
    for p in cons_candidates:
        key = str(Path(p))
        if key not in seen:
            seen.add(key)
            unique_cons_files.append(p)
    num_consignments = sum(_infer_num_consignments(p) for p in unique_cons_files) if unique_cons_files else 1
    compliance_table = load_compliance_lookup_csv(comp_lookup_path)

    def _run_with_config(cfg):
        return run_scenarios_fn(
            config=cfg,
            scenario_table=scenarios,
            seed=seed,
            num_simulations=num_simulations,
            num_consignments=num_consignments,
            compliance_table=compliance_table,
            detailed=True,
        )

    try:
        scenario_results_raw = _run_with_config(config)
    except ValueError as exc:
        msg = str(exc)
        if "Sample larger than population" in msg:
            # Retry once with a conservative sampling proportion to avoid crash
            cfg_retry = copy.deepcopy(config)
            cfg_retry.setdefault("inspection", {}).setdefault("proportion", {})
            current_prop = cfg_retry["inspection"]["proportion"].get("value", 0.02) or 0.02
            cfg_retry["inspection"]["proportion"]["value"] = min(float(current_prop), 0.01)
            cfg_retry["inspection"]["min_inspection_units"] = 0
            try:
                scenario_results_raw = _run_with_config(cfg_retry)
            except Exception as inner_exc:  # pylint: disable=broad-except
                raise ValueError(
                    "Sampling request exceeded available population even after clamping. "
                    "Reduce inspection proportion or check TOTAL_SAMPLING_UNITS in the consignment file."
                ) from inner_exc
        else:
            raise
    except ZeroDivisionError as exc:
        raise ValueError(
            "Division by zero during scenario run. Check inspection proportion, sampling units, and config values."
        ) from exc
    except StopIteration as exc:
        raise ValueError(
            "Scenario execution failed: no valid consignments were generated. "
            "Ensure the consignment file has valid inspection identifiers."
        ) from exc

    scenario_results = [(result, cfg) for _details, result, cfg in scenario_results_raw]
    output_dir = experiment_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df = save_scenario_result_to_pandas(
        scenario_results,
        config_columns=CONFIG_COLUMNS,
        result_columns=RESULT_COLUMNS,
    )
    results_path = output_dir / "pis_contamination_scenario_results.csv"
    results_df.to_csv(results_path, index=False)

    fit = ClarkeFit(alpha=0.0, beta=0.0, theta=float("inf"), raw_result={"source": "scenario_table"})

    return PipelineResult(
        contamination_fit=fit,
        scenario_results=results_df,
        config=config,
        compliance_table=compliance_table,
        pis_data=pd.DataFrame(),
        rbs_data=pd.DataFrame(),
        num_consignments=num_consignments,
    )


__all__ = [
    "SlippagePaths",
    "ExperimentPaths",
    "ClarkeFit",
    "PipelineResult",
    "create_default_paths",
    "load_scenario_dataframe",
    "generate_synthetic_data",
    "fit_contamination_distribution",
    "run_slippage_pipeline",
    "SyntheticOptions",
]
