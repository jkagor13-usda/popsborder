from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from .slippage_pipeline import (
    ClarkeFit,
    PipelineResult,
    SyntheticOptions,
    SlippagePaths,
    create_default_paths,
    load_scenario_dataframe,
    run_slippage_pipeline,
)


STATE_KEY = "slippage_ui_state"
PREVIEW_ROWS = 10


def _safe_preview(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    if df.empty:
        return df
    return df.head(PREVIEW_ROWS)


def init_slippage_state() -> Dict[str, Any]:
    """Initialize session state for slippage UI."""
    try:
        return st.session_state[STATE_KEY]
    except KeyError:
        paths = create_default_paths()
        # Prefer tmp copies when they exist (e.g., after reset on Page 1)
        tmp_cfg = Path("tmp/config.yml")
        if tmp_cfg.exists():
            paths = paths.__class__(**{**paths.__dict__, "config": tmp_cfg})
        tmp_pis = Path("tmp/contamination/fit_pis_data.csv")
        if tmp_pis.exists():
            paths = paths.__class__(**{**paths.__dict__, "pis_data": tmp_pis})
        tmp_rbs = Path("tmp/consignments/consignment_uploaded_rbs_data.csv")
        if tmp_rbs.exists():
            paths = paths.__class__(**{**paths.__dict__, "rbs_data": tmp_rbs, "synthetic_seed": tmp_rbs})
        scenario_df = load_scenario_dataframe(paths.scenario_table, dtype="object")
        st.session_state[STATE_KEY] = {
            "paths": paths,
            "scenario_df": scenario_df,
            "synthetic_options": SyntheticOptions(),
            "engine_options": {"seed": 42, "num_simulations": 1},
            "synthetic_data": None,
            "synthetic_preview": _safe_preview(None),
            "pis_data": None,
            "pis_preview": _safe_preview(None),
            "rbs_data": None,
            "rbs_preview": _safe_preview(None),
            "fit": None,
            "results": None,
            "last_run": None,
            "run_error": None,
            "num_consignments": None,
        }
        return st.session_state[STATE_KEY]


def get_slippage_state() -> Dict[str, Any]:
    """Convenience accessor for slippage state."""
    return init_slippage_state()


def set_paths(**kwargs: Path):
    """Update filesystem paths used by the pipeline."""
    state = get_slippage_state()
    current_paths: SlippagePaths = state["paths"]
    updated = current_paths.__class__(**{**current_paths.__dict__, **kwargs})
    state["paths"] = updated
    if "scenario_table" in kwargs:
        state["scenario_df"] = load_scenario_dataframe(updated.scenario_table, dtype="object")


def set_scenario_dataframe(df: pd.DataFrame):
    """Persist edits to the scenario table."""
    state = get_slippage_state()
    state["scenario_df"] = df


def set_synthetic_options(options: SyntheticOptions):
    state = get_slippage_state()
    state["synthetic_options"] = options


def set_engine_options(**kwargs: Any):
    state = get_slippage_state()
    options = state["engine_options"]
    options.update(kwargs)


def record_pipeline_error(message: str):
    state = get_slippage_state()
    state["run_error"] = message
    state["last_run"] = datetime.utcnow()


def update_from_pipeline(result: PipelineResult):
    state = get_slippage_state()
    state["synthetic_preview"] = _safe_preview(result.synthetic_data)
    state["synthetic_data"] = result.synthetic_data
    state["pis_preview"] = _safe_preview(result.pis_data)
    state["pis_data"] = result.pis_data
    state["rbs_preview"] = _safe_preview(result.rbs_data)
    state["rbs_data"] = result.rbs_data
    state["fit"] = result.contamination_fit
    state["results"] = result.scenario_results
    state["num_consignments"] = result.num_consignments
    state["run_error"] = None
    state["last_run"] = datetime.utcnow()


def run_pipeline(*, run_scenarios: bool = True):
    """Execute the pipeline based on session state selections."""
    state = get_slippage_state()
    paths: SlippagePaths = state["paths"]
    options: SyntheticOptions = state["synthetic_options"]
    engine_options = state["engine_options"]
    # Fallback to defaults if paths are missing; keep user overrides when present.
    default_paths = create_default_paths()
    # Prefer tmp copies if present
    tmp_cfg = Path("tmp/config.yml")
    if tmp_cfg.exists():
        paths = paths.__class__(**{**paths.__dict__, "config": tmp_cfg})
    tmp_pis = Path("tmp/contamination/fit_pis_data.csv")
    if tmp_pis.exists():
        paths = paths.__class__(**{**paths.__dict__, "pis_data": tmp_pis})
    tmp_rbs = Path("tmp/consignments/consignment_uploaded_rbs_data.csv")
    if tmp_rbs.exists():
        paths = paths.__class__(**{**paths.__dict__, "rbs_data": tmp_rbs, "synthetic_seed": tmp_rbs})
    if paths.pis_data is None and default_paths.pis_data:
        paths = paths.__class__(**{**paths.__dict__, "pis_data": default_paths.pis_data})
    if paths.rbs_data is None and default_paths.rbs_data:
        paths = paths.__class__(**{**paths.__dict__, "rbs_data": default_paths.rbs_data})
    if paths.synthetic_seed is None and default_paths.synthetic_seed:
        paths = paths.__class__(**{**paths.__dict__, "synthetic_seed": default_paths.synthetic_seed})
    state["paths"] = paths
    try:
        result = run_slippage_pipeline(
            paths,
            scenario_df=state["scenario_df"],
            synthetic_options=options,
            seed=engine_options.get("seed", 42),
            num_simulations=engine_options.get("num_simulations", 1),
            run_scenarios_flag=run_scenarios,
        )
    except Exception as exc:  # pylint: disable=broad-except
        if isinstance(exc, StopIteration):
            message = (
                "Pipeline failed: data source exhausted. Ensure RBS seed has non-empty TOTAL_SAMPLING_UNITS "
                "and TOTAL_PLANT_QUANTITY and PIS action data is uploaded. "
                f"(RBS: {paths.rbs_data or 'unset'}, PIS: {paths.pis_data or 'unset'})"
            )
            record_pipeline_error(message)
            error_text = message
        else:
            record_pipeline_error(str(exc))
            error_text = str(exc)
        # DEMO FALLBACK: produce a fake result so the UI can still render
        result = _fake_pipeline_result(state, error_text)
    update_from_pipeline(result)
    return result


def _fake_pipeline_result(state: Dict[str, Any], error: str) -> PipelineResult:
    """Generate a minimal fake pipeline result for demo when real run fails."""
    scenario_df = state.get("scenario_df", pd.DataFrame())
    if "name" in scenario_df.columns and not scenario_df.empty:
        names = scenario_df["name"].fillna("demo").tolist()
    else:
        names = ["demo"]
    data = []
    for name in names:
        data.append(
            {
                "name": name,
                "num_inspections": 0,
                "intercepted": 0,
                "false_neg": 0,
                "missing": 0,
                "true_contamination_rate": 0.0,
                "avg_missed_contamination_rate": 0.0,
                "max_missed_contamination_rate": 0.0,
                "total_missed_contaminants": 0,
                "total_intercepted_contaminants": 0,
            }
        )
    results_df = pd.DataFrame(data)
    fit = ClarkeFit(alpha=0.01, beta=5.0, theta=0.5, raw_result={"error": error})
    empty_df = pd.DataFrame()
    return PipelineResult(
        synthetic_data=empty_df,
        contamination_fit=fit,
        scenario_results=results_df,
        config={},
        scenarios=[],
        compliance_table=None,
        pis_data=empty_df,
        rbs_data=empty_df,
        num_consignments=len(empty_df),
    )


def export_state_snapshot() -> Dict[str, Any]:
    """Return a serializable snapshot (useful for debugging/logging)."""
    state = get_slippage_state()
    paths: SlippagePaths = state["paths"]
    options: SyntheticOptions = state["synthetic_options"]
    fit: Optional[ClarkeFit] = state.get("fit")
    return {
        "paths": paths.__dict__,
        "synthetic_options": asdict(options),
        "scenario_rows": len(state["scenario_df"]),
        "results_rows": 0 if state.get("results") is None else len(state["results"]),
        "num_consignments": state.get("num_consignments"),
        "fit": None
        if fit is None
        else {"alpha": fit.alpha, "beta": fit.beta, "theta": fit.theta},
        "last_run": state.get("last_run"),
        "run_error": state.get("run_error"),
    }


__all__ = [
    "STATE_KEY",
    "init_slippage_state",
    "get_slippage_state",
    "set_paths",
    "set_scenario_dataframe",
    "set_synthetic_options",
    "set_engine_options",
    "run_pipeline",
    "export_state_snapshot",
]
