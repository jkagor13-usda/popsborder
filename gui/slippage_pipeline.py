# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

from __future__ import annotations

import copy
import pickle
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .runtime_warnings import suppress_optional_dependency_warnings

suppress_optional_dependency_warnings()

import pandas as pd

from popsborder.generator import SyntheticConsignmentDataGenerator, save_to_csv
from popsborder.inputs import (
    load_compliance_lookup_csv,
    load_configuration,
    load_scenario_table,
)
from popsborder.outputs import save_scenario_result_to_pandas
from popsborder.scenarios import run_scenarios
from slippage_model_utils.clarke_model_support_functions import gen_clarke_model_inputs
from slippage_model_utils.r_script_wrapper import run_clarke_bb_group_model


# Default config columns to persist into results
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

# Default result columns to persist into outputs
RESULT_COLUMNS = [
    "num_inspections",
    "intercepted",
    "false_neg",
    "missing",
    "num_inspection_units",
    "num_sample_units",
    "num_plants",
    "avg_inspection_units_opened_completion",
    "avg_inspection_units_opened_detection",
    "pct_inspection_units_opened_completion",
    "pct_inspection_units_opened_detection",
    "avg_sample_units_inspected_completion",
    "avg_sample_units_inspected_detection",
    "pct_sample_units_inspected_completion",
    "pct_sample_units_inspected_detection",
    "avg_plant_units_inspected_completion",
    "avg_plant_units_inspected_detection",
    "pct_plant_units_inspected_completion",
    "pct_plant_units_inspected_detection",
    "total_missed_contaminants",
    "total_intercepted_contaminants",
    "total_slipped_units",
    "avg_slipped_units_per_consignment",
    "avg_slipped_sample_units_per_consignment",
    "total_contaminated_units",
    "total_contaminated_sample_units",
    "total_contaminated_inspection_units",
    "total_intercepted_inspection_units",
    "total_slipped_inspection_units",
]

@dataclass
class SyntheticOptions:
    """Placeholder for interface compatibility (not used in simplified pipeline)."""

    n_samples: int = 10
    sampling_method: str = "sequential"


DEFAULT_DATA_DIR = Path("data_input")
COMPLIANCE_FILENAME = "compliance_table.csv"
CONFIG_FILENAME = "config.yml"



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
    output_dir: Path
    output_files: List[Path]


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


def load_compliance_policy(path: Path) -> Dict[str, Any]:
    """Load either a CSV compliance table or a pickled compliance policy."""
    policy_path = Path(path)
    if policy_path.suffix.lower() == ".pkl":
        with open(policy_path, "rb") as handle:
            return pickle.load(handle)
    return load_compliance_lookup_csv(policy_path)

def generate_synthetic_data(
    seed_path: Path,
    output_path: Path,
    options: SyntheticOptions,
    producer_grouping_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate synthetic consignment data and persist it."""
    if seed_path is None:
        raise FileNotFoundError("Seed data path was not provided.")
    seed_path = Path(seed_path).resolve()
    output_path = Path(output_path).resolve()
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed data not found at {seed_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = None
    config_path = Path(DEFAULT_DATA_DIR / CONFIG_FILENAME).resolve()
    if config_path.exists():
        config = load_configuration(config_path)
    producer_grouping = None
    if producer_grouping_path is not None:
        producer_grouping_path = Path(producer_grouping_path).resolve()
        if not producer_grouping_path.exists():
            raise FileNotFoundError(f"Producer grouping file not found at {producer_grouping_path}")
        producer_grouping = pd.read_csv(producer_grouping_path)
    generator = SyntheticConsignmentDataGenerator(
        config=config,
        producer_group_mapping=producer_grouping,
        input_data_file=seed_path,
    )
    synth_data = generator.generate_from_input_data(
        n_consignments=options.n_samples,
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
) -> Tuple[Dict[Tuple,Any], pd.DataFrame, Dict[Any, Any], list[tuple[tuple, str]]]:
    """Fit contamination parameters using the Clarke beta-binomial model."""
    if pis_data_path is None or not Path(pis_data_path).exists():
        raise FileNotFoundError("PIS action data not provided. Upload on Page 2 - Contamination Fit.")
    pis_df = pd.read_csv(pis_data_path)
    if "action" not in pis_df.columns:
        raise ValueError("Data files must include binary 'action' column to align for fitting.")
    try:
        inputs_by_quantity = gen_clarke_model_inputs(pis_df)
    except StopIteration as exc:
        raise ValueError(
            "Fitting failed: check the uploaded data. "
            "Verify required columns are there."
        ) from exc

    if not inputs_by_quantity:
        raise ValueError("Fitting failed: generating of Clark inputs has failed.  Check input data.")

    res = {}
    warnings: list[tuple[tuple, str]] = []
    # Defaults for if/when parameters for alpha/beta are both zero
    k = 10_000
    default_alpha = 0.02 * k  # 200
    default_beta = 0.98 * k  # 9800
    print(f'\nNow Executing Clarke Model Based on Quantities')
    for (lower, upper), inputs in inputs_by_quantity.items():
        print(f'   Calculating for Quantity Range:  {(lower, upper)}')
        params = run_clarke_bb_group_model(
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

        alpha = params.get("alpha", 0)
        beta = params.get("beta", 0)

        if alpha == 0 and beta == 0:
            # Create a warning message
            warning_msg = (
                f"Fitted alpha and beta were zero for quantity range {(lower, upper)}; "
                f"values displayed above are defaults "
                f"for a mean rate ≈ 0.02 and 95% CI = [0.0175,0.0229]."
            )
            warnings.append(((lower, upper), warning_msg))

            # override
            params["alpha"] = default_alpha
            params["beta"] = default_beta

        res[(lower, upper)] = params

    return res, pis_df, inputs_by_quantity, warnings



def run_slippage_pipeline(
    exp_paths: ExperimentPaths,
    *,
    seed: int = 42,
    num_simulations: int = 1,
) -> PipelineResult:
    """Execute the pipeline using files inside a specific experiment folder."""

    experiment_dir = exp_paths.experiment_dir
    scenario_table = exp_paths.scenario_table
    config_path = exp_paths.config or (DEFAULT_DATA_DIR / CONFIG_FILENAME)

    # --- Load scenarios ---
    scenarios = load_scenario_dataframe(scenario_table)
    if isinstance(scenarios, pd.DataFrame):
        scenarios = scenarios.to_dict(orient="records")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Scenario table is empty or unreadable.")

    # --- Small helpers ---
    def _is_missing(v: Any) -> bool:
        return v in (None, "", "None", "nan")

    def _maybe_path(v: Any) -> Optional[Path]:
        return None if _is_missing(v) else Path(str(v))

    def _in_exp_dir(p: Path) -> Path:
        # If given a relative path or a missing path, prefer experiment_dir/<filename>
        if p.is_absolute():
            return p
        if p.exists():
            return p
        return experiment_dir / p.name

    def _first_existing_path(values: List[Optional[Path]]) -> Optional[Path]:
        for p in values:
            if p and Path(p).exists():
                return Path(p)
        return None

    def _scenario_path(rec: Dict[str, Any], key: str) -> Optional[Path]:
        p = _maybe_path(rec.get(key))
        return _in_exp_dir(p) if p else None

    def _apply_contam_defaults(rec: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
        rec = rec.copy()
        base = cfg.get("contamination", {}).get("contamination_rate", {})
        if _is_missing(rec.get("contamination/contamination_rate/distribution")):
            rec["contamination/contamination_rate/distribution"] = base.get("distribution", "beta_binomial")

        bb_defaults = base.get("beta_binomial_parameters", {})
        for key in ("alpha", "beta", "theta", "N_bar", "I", "J"):
            col = f"contamination/contamination_rate/beta_binomial_parameters/{key}"
            if _is_missing(rec.get(col)):
                rec[col] = bb_defaults.get(key)
        return rec

    def _normalize_inspection(rec: Dict[str, Any]) -> Dict[str, Any]:
        rec = rec.copy()
        try:
            prop = float(rec.get("inspection/proportion/value", 0) or 0)
        except Exception:
            prop = 0.0
        if prop <= 0:
            prop = 0.02
        rec["inspection/proportion/value"] = min(prop, 1.0)
        rec.setdefault("inspection/sample_strategy", "rbs")
        return rec

    def _build_cfg_for_scenario(
        base_cfg: Dict[str, Any],
        cons_path: Optional[Path],
        comp_path: Optional[Path],
    ) -> Dict[str, Any]:
        cfg = copy.deepcopy(base_cfg)

        if cons_path and cons_path.exists():
            cons_str = str(cons_path).replace("\\", "/")
            cfg.setdefault("consignment", {}).setdefault("input_file", {})
            cfg["consignment"]["input_file"]["rbs_file_name"] = cons_str
            cfg["consignment"]["input_file"]["file_name"] = cons_str

        if comp_path and comp_path.exists():
            comp_str = str(comp_path).replace("\\", "/")
            cfg.setdefault("inspection", {}).setdefault("compliance_table", {})
            cfg["inspection"]["compliance_table"]["file_name"] = comp_str

        return cfg

    # --- Resolve base files (consignment / compliance / config) ---
    # Candidate consignment paths: explicit, plus any scenario references that exist
    cons_candidates: List[Path] = []
    if exp_paths.consignment and exp_paths.consignment.exists():
        cons_candidates.append(exp_paths.consignment)

    for rec in scenarios:
        p = _scenario_path(rec, "consignment/input_file/file_name")
        if p and p.exists():
            cons_candidates.append(p)

    cons_path = _first_existing_path(cons_candidates)
    if not cons_path:
        raise FileNotFoundError(f"Consignment file missing in experiment folder {experiment_dir}.")

    # Compliance: explicit, else first scenario reference, else default dir fallback (must exist)
    comp_path = None
    if exp_paths.compliance and exp_paths.compliance.exists():
        comp_path = exp_paths.compliance
    else:
        for rec in scenarios:
            p = _scenario_path(rec, "inspection/compliance_table/file_name")
            if p and p.exists():
                comp_path = p
                break

    if not config_path or not Path(config_path).exists():
        raise FileNotFoundError(
            f"Config file missing in experiment folder {experiment_dir} (expected {CONFIG_FILENAME})."
        )
    config = load_configuration(config_path)

    comp_lookup_path = comp_path if (comp_path and comp_path.exists()) else (DEFAULT_DATA_DIR / COMPLIANCE_FILENAME)
    if not comp_lookup_path.exists():
        raise FileNotFoundError(
            f"Compliance table not found for experiment {experiment_dir}. Expected {COMPLIANCE_FILENAME}."
        )

    compliance_table = load_compliance_policy(comp_lookup_path)

    # Ensure base config points to base consignment/compliance (scenario overrides still apply later)
    config = copy.deepcopy(config)
    config.setdefault("consignment", {}).setdefault("input_file", {})
    config["consignment"]["input_file"]["rbs_file_name"] = str(cons_path).replace("\\", "/")
    config["consignment"]["input_file"]["file_name"] = str(cons_path).replace("\\", "/")
    config.setdefault("inspection", {}).setdefault("compliance_table", {})
    config["inspection"]["compliance_table"]["file_name"] = str(comp_lookup_path).replace("\\", "/")

    # Estimate consignments conservatively across unique consignment files
    unique_cons_files = list({str(Path(p)): p for p in cons_candidates}.values())
    counts = [_infer_num_consignments(p) for p in unique_cons_files] if unique_cons_files else []
    num_consignments_default = min(counts) if counts else 1

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = experiment_dir / f"output_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    def _run(
        cfg: Dict[str, Any],
        rec: Dict[str, Any],
        cons_count: int,
        *,
        num_sims: int,
        comp_table: Dict[str, Any],
        seed_override: Optional[int] = None,
    ):
        return run_scenarios(
            config=cfg,
            scenario_table=[rec],
            seed=seed if seed_override is None else seed_override,
            num_simulations=num_sims,
            num_consignments=cons_count,
            compliance_table=comp_table,
            detailed=True,
            output_root=output_dir,
        )

    # --- Core executor (supports a single retry config) ---
    skipped_scenarios: List[str] = []

    def _execute_all(base_cfg: Dict[str, Any]) -> Tuple[List[Tuple[Any, Dict, Dict]], List[int]]:
        scenario_results_raw: List[Tuple[Any, Dict, Dict]] = []
        consignment_counts: List[int] = []

        for rec0 in scenarios:
            rec = _normalize_inspection(_apply_contam_defaults(rec0, base_cfg))

            rec_cons_path = _scenario_path(rec, "consignment/input_file/file_name")
            rec_comp_path = _scenario_path(rec, "inspection/compliance_table/file_name")
            rec_num_consignments = _infer_num_consignments(rec_cons_path) if rec_cons_path else 1
            consignment_counts.append(rec_num_consignments)

            cfg_local = _build_cfg_for_scenario(base_cfg, rec_cons_path, rec_comp_path)
            comp_table = load_compliance_policy(rec_comp_path) if rec_comp_path else compliance_table

            # Aggregated run
            last_exc: Optional[Exception] = None
            for retry_idx in range(10):
                try:
                    scenario_results_raw.extend(
                        _run(
                            cfg_local,
                            rec,
                            rec_num_consignments,
                            num_sims=num_simulations,
                            comp_table=comp_table,
                            seed_override=seed + retry_idx if seed is not None else None,
                        )
                    )
                    last_exc = None
                    break
                except ValueError as exc:
                    if "a <= 0" not in str(exc):
                        raise
                    last_exc = exc
                    continue
            if last_exc is not None:
                skipped_scenarios.append(str(rec.get("name", f"scenario_{len(skipped_scenarios) + 1}")))
                continue
        return scenario_results_raw, consignment_counts

    # --- Run with one retry path for "Sample larger than population" ---
    try:
        scenario_results_raw, consignment_counts = _execute_all(config)
    except ValueError as exc:
        if "Sample larger than population" not in str(exc):
            raise

        cfg_retry = copy.deepcopy(config)
        cfg_retry.setdefault("inspection", {}).setdefault("proportion", {})
        current_prop = cfg_retry["inspection"]["proportion"].get("value", 0.02) or 0.02
        cfg_retry["inspection"]["proportion"]["value"] = min(float(current_prop), 0.001)
        cfg_retry["inspection"]["min_inspection_units"] = 0

        scenario_results_raw, consignment_counts = _execute_all(cfg_retry)
    except ZeroDivisionError as exc:
        raise ValueError(
            "Division by zero during scenario run. Check inspection proportion, sampling units, and config values."
        ) from exc
    except StopIteration as exc:
        raise ValueError(
            "Scenario execution failed: no valid consignments were generated. "
            "Ensure the consignment file has valid inspection identifiers."
        ) from exc

    # --- Convert results and write outputs ---
    if not scenario_results_raw:
        if skipped_scenarios:
            raise ValueError(
                "All scenarios were skipped after repeated 'a <= 0' contamination errors: "
                + ", ".join(skipped_scenarios)
            )
        raise ValueError("No scenario results were generated.")

    scenario_results = [(result, cfg) for _details, result, cfg in scenario_results_raw]

    results_df = save_scenario_result_to_pandas(
        scenario_results,
        config_columns=CONFIG_COLUMNS,
        result_columns=RESULT_COLUMNS,
    )

    results_output_path = output_dir / "pis_contamination_scenario_results.csv"
    results_df.to_csv(results_output_path, index=False)
    if skipped_scenarios:
        skipped_path = output_dir / "skipped_scenarios.txt"
        skipped_path.write_text(
            "Skipped after repeated 'a <= 0' errors:\n" + "\n".join(skipped_scenarios),
            encoding="utf-8",
        )

    # Per-replication output
    output_files: List[Path] = [results_output_path]
    if skipped_scenarios:
        output_files.append(skipped_path)
    runs_records: List[Dict[str, Any]] = []
    for _details, result, cfg in scenario_results_raw:
        rep_df = getattr(result, "replication_outputs", None)
        if rep_df is None or getattr(rep_df, "empty", True):
            continue
        rep_records = rep_df.copy()
        if isinstance(cfg, dict) and "name" in cfg:
            rep_records["name"] = cfg.get("name")
        runs_records.extend(rep_records.to_dict(orient="records"))
    if runs_records:
        all_runs_output_path = output_dir / "all_runs.csv"
        pd.DataFrame(runs_records).to_csv(all_runs_output_path, index=False)
        output_files.append(all_runs_output_path)

    fit = ClarkeFit(alpha=0.0, beta=0.0, theta=float("inf"), raw_result={"source": "scenario_table"})
    total_cons = sum(consignment_counts) if consignment_counts else num_consignments_default

    return PipelineResult(
        contamination_fit=fit,
        scenario_results=results_df,
        config=config,
        compliance_table=compliance_table,
        pis_data=pd.DataFrame(),
        rbs_data=pd.DataFrame(),
        num_consignments=total_cons,
        output_dir=output_dir,
        output_files=output_files,
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
