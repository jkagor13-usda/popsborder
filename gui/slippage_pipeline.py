from __future__ import annotations

import copy
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from popsborder.generator import SyntheticConsignmentDataGenerator, save_to_csv
from popsborder.inputs import (
    load_compliance_lookup_csv,
    load_configuration,
    load_scenario_table,
    text_to_value,
)
from popsborder.outputs import save_scenario_result_to_pandas
from popsborder.scenarios import run_scenarios
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
    pis_data: Path = DEFAULT_DATA_DIR / "synthetic_pis_data.csv"
    rbs_data: Path = DEFAULT_DATA_DIR / "synthetic_rbs_calc_data_enriched.csv"
    synthetic_seed: Path = DEFAULT_DATA_DIR / "synthetic_pis_data.csv"
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
    return SlippagePaths(data_dir=base_dir)


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
    if not seed_path.exists():
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
    pis_df = pd.read_csv(pis_data_path)
    if "action" not in pis_df.columns:
        pis_df["action"] = 0
    rbs_df = pd.read_csv(rbs_data_path)
    inputs = gen_clarke_model_inputs(pis_df, rbs_df)
    result = run_clarke_bb_group_model(
        inputs.ty,
        inputs.b,
        inputs.B,
        inputs.Nbar,
        inputs.freq,
        inputs.theta,
        inputs.R,
        inputs.start_val,
        inputs.se,
    )
    fit = ClarkeFit(alpha=result["alpha"], beta=result["beta"], theta=inputs.theta, raw_result=result)
    return fit, pis_df, rbs_df


def apply_contamination_parameters(
    config: Dict[str, Any],
    scenarios: List[Dict[str, Any]],
    fit: ClarkeFit,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Inject beta-binomial parameters into config and scenario records."""
    new_config = copy.deepcopy(config)
    new_scenarios = copy.deepcopy(scenarios)

    try:
        new_config["contamination"]["contamination_rate"]["parameters"][0] = fit.alpha
        new_config["contamination"]["contamination_rate"]["parameters"][1] = fit.beta
    except KeyError as exc:
        raise KeyError("Configuration is missing expected contamination rate parameter slots.") from exc

    for scenario in new_scenarios:
        scenario["contamination/contamination_rate/beta_binomial_parameters/alpha"] = fit.alpha
        scenario["contamination/contamination_rate/beta_binomial_parameters/beta"] = fit.beta
        scenario["contamination/contamination_rate/beta_binomial_parameters/theta"] = fit.theta
        scenario["contamination/contamination_rate/value"] = None
    return new_config, new_scenarios


def run_slippage_pipeline(
    paths: SlippagePaths,
    scenario_df: Optional[pd.DataFrame] = None,
    *,
    synthetic_options: SyntheticOptions = SyntheticOptions(),
    seed: int = 42,
    num_simulations: int = 1,
    run_scenarios: bool = True,
) -> PipelineResult:
    """Execute the full slippage pipeline and return artifacts for UI consumption."""
    scenario_df = scenario_df if scenario_df is not None else load_scenario_dataframe(paths.scenario_table)

    synthetic_df = generate_synthetic_data(
        paths.synthetic_seed,
        paths.synthetic_output,
        synthetic_options,
    )
    config = load_configuration(paths.config)
    config["consignment"]["input_file"]["rbs_file_name"] = str(paths.synthetic_output)
    num_consignments = _infer_num_consignments(paths, config, synthetic_df)
    scenarios = dataframe_to_scenarios(scenario_df)
    compliance_table = load_compliance_lookup_csv(paths.compliance_lookup)

    fit, pis_df, rbs_df = fit_contamination_distribution(paths.pis_data, paths.rbs_data)
    config, scenarios = apply_contamination_parameters(config, scenarios, fit)

    if run_scenarios:
        scenario_results_raw = run_scenarios(
            config=config,
            scenario_table=scenarios,
            seed=seed,
            num_simulations=num_simulations,
            num_consignments=num_consignments,
            compliance_table=compliance_table,
            detailed=True,
        )
        scenario_results = [(result, cfg) for _details, result, cfg in scenario_results_raw]
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        results_df = save_scenario_result_to_pandas(
            scenario_results,
            config_columns=CONFIG_COLUMNS,
            result_columns=RESULT_COLUMNS,
        )
        results_path = paths.output_dir / "pis_contamination_scenario_results.csv"
        results_df.to_csv(results_path, index=False)
    else:
        results_df = pd.DataFrame()

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
