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
class ClarkeFit:
    """Fitted contamination parameters."""

    alpha: float
    beta: float
    theta: float
    raw_result: Dict[str, Any]


@dataclass
class PipelineResult:
    """Outputs from a pipeline run."""

    synthetic_data: pd.DataFrame
    contamination_fit: ClarkeFit
    scenario_results: pd.DataFrame
    config: Dict[str, Any]
    scenarios: List[Dict[str, Any]]
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
    records = load_scenario_table(path)
    return records


def dataframe_to_scenarios(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a dataframe representation of the scenario table into dictionaries."""
    if isinstance(df, list):
        # Already a list of records from load_scenario_table
        return [dict(rec) for rec in df]
    scenarios: List[Dict[str, Any]] = []
    selectable_columns = [col for col in df.columns if col and not str(col).startswith("Unnamed")]
    cleaned_df = df.copy()
    for column in selectable_columns:
        if pd.api.types.is_numeric_dtype(cleaned_df[column]):
            cleaned_df[column] = cleaned_df[column]
        else:
            cleaned_df[column] = cleaned_df[column].astype("object")

    for _, row in cleaned_df.iterrows():
        scenario: Dict[str, Any] = {}
        for column in selectable_columns:
            value = row[column]
            if pd.isna(value):
                scenario[column] = None
                continue
            if isinstance(value, str):
                value = value.strip()
            if value in ("", "None", "nan"):
                scenario[column] = None
                continue
            scenario[column] = text_to_value(value)
        scenarios.append(scenario)
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
            "Fitting failed: no compatible records found between PIS and RBS data. "
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
    paths: SlippagePaths,
    *,
    seed: int = 42,
    num_simulations: int = 1,
) -> PipelineResult:
    """Execute the pipeline using the scenario CSV only."""
    # Load scenario records (list of dicts)
    records = load_scenario_dataframe(paths.scenario_table)
    if isinstance(records, pd.DataFrame):
        records = records.to_dict(orient="records")
    if not isinstance(records, list):
        raise ValueError("Scenario table could not be loaded as records.")
    if not records:
        raise ValueError("Scenario table is empty. Provide a scenario_table.csv with at least one row.")

    # Remove unnamed columns
    scenario_records: List[Dict[str, Any]] = [
        {k: v for k, v in rec.items() if not str(k).startswith("Unnamed")} for rec in records
    ]

    def _first_value(key: str) -> Any:
        for rec in scenario_records:
            val = rec.get(key, None)
            if val not in (None, "", "None", "nan"):
                return val
        return None

    def _unique_values(key: str) -> List[Any]:
        vals: List[Any] = []
        seen: set[Any] = set()
        for rec in scenario_records:
            val = rec.get(key, None)
            if val in (None, "", "None", "nan"):
                continue
            if val in seen:
                continue
            seen.add(val)
            vals.append(val)
        return vals

    # Pull contamination parameters from scenario records
    scenario_alpha = float(_first_value("contamination/contamination_rate/beta_binomial_parameters/alpha") or 0)
    scenario_beta = float(_first_value("contamination/contamination_rate/beta_binomial_parameters/beta") or 0)
    theta_val = _first_value("contamination/contamination_rate/beta_binomial_parameters/theta")
    scenario_theta = float(theta_val) if theta_val not in (None, "", "None", "nan") else float("inf")

    experiment_dir = Path(paths.scenario_table).parent

    def _resolve_from_experiment(name: str, fallback_dir: Path) -> Optional[Path]:
        if not name:
            return None
        cand = experiment_dir / name
        if cand.exists():
            return cand
        fallback = fallback_dir / name
        if fallback.exists():
            dest = experiment_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(fallback.read_bytes())
            return dest
        return None

    # Resolve consignment and compliance paths based on scenario records
    consignment_col = (
        "consignment/input_file/file_name"
        if any("consignment/input_file/file_name" in r for r in scenario_records)
        else "consignment name"
    )
    compliance_col = (
        "inspection/compliance_table/file_name"
        if any("inspection/compliance_table/file_name" in r for r in scenario_records)
        else "inspection name"
    )

    rbs_path: Optional[Path] = None
    compliance_path: Optional[Path] = experiment_dir / "compliance_table.csv"

    consignment_names = _unique_values(consignment_col)
    if consignment_names:
        rbs_path = experiment_dir / consignment_names[0]
        if not rbs_path.exists():
            rbs_path = _resolve_from_experiment(consignment_names[0], Path("tmp") / "consignments")

    compliance_names = _unique_values(compliance_col)
    if compliance_names:
        candidate = experiment_dir / compliance_names[0]
        if candidate.exists():
            compliance_path = candidate
        else:
            fallback = _resolve_from_experiment(compliance_names[0], Path("tmp") / "compliance")
            if fallback:
                compliance_path = fallback

    if rbs_path is None or not rbs_path.exists():
        raise FileNotFoundError("No consignment RBS data found for the selected experiment.")
    synthetic_df = pd.read_csv(rbs_path)
    if synthetic_df.empty:
        raise ValueError("RBS data is empty. Upload a non-empty consignment file in tmp/consignments.")

    # Load config and compliance
    config_path = experiment_dir / "config.yml"
    config = load_configuration(config_path if config_path.exists() else paths.config)
    config["consignment"]["input_file"]["rbs_file_name"] = str(rbs_path)
    num_consignments = max(1, _infer_num_consignments(paths, config, synthetic_df))
    compliance_lookup_path = compliance_path if compliance_path and compliance_path.exists() else paths.compliance_lookup
    compliance_table = load_compliance_lookup_csv(compliance_lookup_path)

    # Convert records to scenarios for popsborder
    scenarios = dataframe_to_scenarios(scenario_records)
    fit = ClarkeFit(alpha=scenario_alpha, beta=scenario_beta, theta=scenario_theta, raw_result={"source": "scenario_table"})
    pis_df = pd.DataFrame()
    rbs_df = pd.DataFrame()

    try:
        scenario_results_raw = run_scenarios_fn(
            config=config,
            scenario_table=scenarios,
            seed=seed,
            num_simulations=num_simulations,
            num_consignments=num_consignments,
            compliance_table=compliance_table,
            detailed=True,
        )
    except StopIteration as exc:
        raise ValueError(
            "Scenario execution failed: no valid synthetic consignments were generated. "
            "Ensure the RBS seed file has non-empty TOTAL_SAMPLING_UNITS and TOTAL_PLANT_QUANTITY columns. "
            f"(Synthetic input: {paths.synthetic_output})"
        ) from exc

    scenario_results = [(result, cfg) for _details, result, cfg in scenario_results_raw]
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    results_df = save_scenario_result_to_pandas(
        scenario_results,
        config_columns=CONFIG_COLUMNS,
        result_columns=RESULT_COLUMNS,
    )
    out_dir = experiment_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "pis_contamination_scenario_results.csv"
    results_df.to_csv(results_path, index=False)

    return PipelineResult(
        synthetic_data=synthetic_df,
        contamination_fit=fit,
        scenario_results=results_df,
        config=config,
        scenarios=scenarios,
        compliance_table=compliance_table,
        pis_data=pis_df,
        rbs_data=rbs_df,
        num_consignments=num_consignments,
    )


__all__ = [
    "SlippagePaths",
    "ClarkeFit",
    "PipelineResult",
    "create_default_paths",
    "load_scenario_dataframe",
    "dataframe_to_scenarios",
    "generate_synthetic_data",
    "fit_contamination_distribution",
    "run_slippage_pipeline",
    "SyntheticOptions",
]
