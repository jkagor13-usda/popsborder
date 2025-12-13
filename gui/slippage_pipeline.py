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
class SyntheticOptions:
    """Options controlling synthetic consignment data generation."""

    n_samples: int = 10
    sampling_method: str = "sequential"


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


def load_scenario_dataframe(path: Path, *, dtype: str = "object") -> pd.DataFrame:
    """Load the slippage scenario CSV into a dataframe suitable for UI editing."""
    df = pd.read_csv(path, dtype=dtype)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def dataframe_to_scenarios(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a dataframe representation of the scenario table into dictionaries."""
    scenarios: List[Dict[str, Any]] = []
    selectable_columns = [col for col in df.columns if col and not str(col).startswith("Unnamed")]
    # Ensure we don't mutate caller dataframe
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


def _infer_num_consignments(
    paths: SlippagePaths,
    config: Dict[str, Any],
    synthetic_df: Optional[pd.DataFrame],
) -> int:
    """Estimate how many consignments are available for simulation."""
    consignment_cfg = config.get("consignment", {})
    input_cfg = consignment_cfg.get("input_file", {}) or {}
    candidate_files = [
        input_cfg.get("rbs_file_name"),
        input_cfg.get("file_name"),
        input_cfg.get("pis_file_name"),
    ]
    for candidate in candidate_files:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        data: Optional[pd.DataFrame] = None
        if synthetic_df is not None and candidate_path == paths.synthetic_output:
            data = synthetic_df
        elif candidate_path.exists():
            try:
                data = pd.read_csv(candidate_path)
            except Exception:  # pragma: no cover - best-effort fallback
                data = None
        if data is None:
            continue
        if "INSPECTION_NUMBER" in data.columns:
            count = data["INSPECTION_NUMBER"].nunique()
        else:
            count = len(data)
        return max(1, int(count))
    return max(1, int(consignment_cfg.get("num_consignments", 1)))


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


def apply_contamination_parameters(
    scenarios: List[Dict[str, Any]],
    fit: ClarkeFit,
) -> List[Dict[str, Any]]:
    """Inject beta-binomial parameters into scenario records."""
    new_scenarios = copy.deepcopy(scenarios)

    for scenario in new_scenarios:
        scenario["contamination/contamination_rate/beta_binomial_parameters/alpha"] = fit.alpha
        scenario["contamination/contamination_rate/beta_binomial_parameters/beta"] = fit.beta
        scenario["contamination/contamination_rate/beta_binomial_parameters/theta"] = fit.theta
        scenario["contamination/contamination_rate/value"] = None
    return new_scenarios


def run_slippage_pipeline(
    paths: SlippagePaths,
    scenario_df: Optional[pd.DataFrame] = None,
    *,
    synthetic_options: SyntheticOptions = SyntheticOptions(),
    seed: int = 42,
    num_simulations: int = 1,
    fit_override: Optional[ClarkeFit] = None,
    pis_df_override: Optional[pd.DataFrame] = None,
    rbs_df_override: Optional[pd.DataFrame] = None,
) -> PipelineResult:
    """Execute the full slippage pipeline and return artifacts for UI consumption."""
    scenario_df = scenario_df if scenario_df is not None else load_scenario_dataframe(paths.scenario_table)
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

    # Resolve consignment and compliance paths based on the scenario table
    consignment_col = "consignment name"
    compliance_col = "inspection name"
    rbs_path = None
    compliance_path = None
    if consignment_col in scenario_df.columns:
        consignment_names = [c for c in scenario_df[consignment_col].dropna().unique().tolist() if c]
        if consignment_names:
            rbs_path = _resolve_from_experiment(consignment_names[0], Path("tmp") / "consignments")
    if compliance_col in scenario_df.columns:
        compliance_names = [c for c in scenario_df[compliance_col].dropna().unique().tolist() if c]
        if compliance_names:
            compliance_path = _resolve_from_experiment(compliance_names[0], Path("tmp") / "compliance")

    if rbs_path is None or not rbs_path.exists():
        raise FileNotFoundError("No consignment RBS data found for the selected experiment.")
    rbs_df = pd.read_csv(rbs_path)
    synthetic_df = rbs_df.copy()
    if synthetic_df.empty:
        raise ValueError("RBS data is empty. Upload a non-empty consignment file in tmp/consignments.")

    # Use experiment-local config if present; else fallback to default
    config_path = experiment_dir / "config.yml"
    config = load_configuration(config_path if config_path.exists() else paths.config)
    config["consignment"]["input_file"]["rbs_file_name"] = str(rbs_path)
    num_consignments = max(1, _infer_num_consignments(paths, config, synthetic_df))
    scenarios = dataframe_to_scenarios(scenario_df)
    compliance_lookup_path = compliance_path if compliance_path else paths.compliance_lookup
    compliance_table = load_compliance_lookup_csv(compliance_lookup_path)

    if fit_override is not None:
        fit = fit_override
        if pis_df_override is not None:
            pis_df = pis_df_override
        elif paths.pis_data and Path(paths.pis_data).exists():
            pis_df = pd.read_csv(paths.pis_data)
        else:
            pis_df = pd.DataFrame()
        if rbs_df_override is not None:
            rbs_df = rbs_df_override
        elif paths.rbs_data and Path(paths.rbs_data).exists():
            rbs_df = pd.read_csv(paths.rbs_data)
        else:
            rbs_df = pd.DataFrame()
    else:
        # Pull contamination parameters from the saved JSON produced on Page 2
        param_path = Path("tmp") / "contamination" / "contamination_parameter_sets.json"
        alpha_val = beta_val = None
        theta_val = float("inf")
    if param_path.exists():
        try:
            with open(param_path, "r") as f:
                param_sets = json.load(f)
            if isinstance(param_sets, dict) and param_sets:
                last_key = list(param_sets.keys())[-1]
                params = param_sets.get(last_key, {})
                alpha_val = float(params.get("alpha", alpha_val))
                beta_val = float(params.get("beta", beta_val))
                theta_raw = params.get("theta", theta_val)
                theta_val = float("inf") if theta_raw is None or np.isinf(theta_raw) else float(theta_raw)
        except Exception:
            pass

        alpha_default = config.get("contamination", {}).get("contamination_rate", {}).get("parameters", [0.2, 5])[0]
        beta_default = config.get("contamination", {}).get("contamination_rate", {}).get("parameters", [0.2, 5])[1]
        alpha_val = alpha_val if alpha_val is not None else alpha_default
        beta_val = beta_val if beta_val is not None else beta_default

        fit = ClarkeFit(alpha=alpha_val, beta=beta_val, theta=theta_val, raw_result={"source": "saved_json"})
        pis_df = pd.DataFrame()
        rbs_df = pd.DataFrame()

    scenarios = apply_contamination_parameters(scenarios, fit)

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
    results_path = experiment_dir / "pis_contamination_scenario_results.csv"
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
    "SyntheticOptions",
    "SlippagePaths",
    "ClarkeFit",
    "PipelineResult",
    "create_default_paths",
    "load_scenario_dataframe",
    "dataframe_to_scenarios",
    "generate_synthetic_data",
    "fit_contamination_distribution",
    "apply_contamination_parameters",
    "run_slippage_pipeline",
]
