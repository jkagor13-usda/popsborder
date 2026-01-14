from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from .slippage_pipeline import (
    ExperimentPaths,
    SyntheticOptions,
    SlippagePaths,
    create_default_paths,
    load_scenario_dataframe,
    run_slippage_pipeline,
)

CONS_FILENAME = "consignment_uploaded_rbs_data.csv"
COMPLIANCE_FILENAME = "compliance_table.csv"
SCENARIO_FILENAME = "scenario_table.csv"
CONFIG_FILENAME = "config.yml"


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




def run_pipeline(experiment_dir):
    """Execute the pipeline. If experiment_dir is provided, load all inputs from that folder."""
    state = get_slippage_state()
    paths: SlippagePaths = state["paths"]
    engine_options = state["engine_options"]

    cfg_path = experiment_dir / CONFIG_FILENAME
    scenario_path = experiment_dir / SCENARIO_FILENAME
    # Resolve consignment/compliance filenames from the scenario table (supports custom names)
    consignment_path = None
    compliance_path = None
    try:
        scenario_df = pd.read_csv(scenario_path)
        def _first_value(col: str) -> Optional[Path]:
            if col not in scenario_df.columns:
                return None
            for val in scenario_df[col].dropna().tolist():
                if val:
                    return Path(str(val))
            return None
        cons_val = _first_value("consignment/input_file/file_name")
        comp_val = _first_value("inspection/compliance_table/file_name")
        if cons_val:
            consignment_path = experiment_dir / cons_val.name if not cons_val.is_absolute() else cons_val
        if comp_val:
            compliance_path = experiment_dir / comp_val.name if not comp_val.is_absolute() else comp_val
    except Exception:
        scenario_df = None

    exp_paths = ExperimentPaths(
        experiment_dir=experiment_dir,
        scenario_table=scenario_path,
        consignment=consignment_path if consignment_path and consignment_path.exists() else None,
        compliance=compliance_path if compliance_path and compliance_path.exists() else None,
        config=cfg_path if cfg_path.exists() else paths.config,
    )
    # Keep state paths in sync with the experiment we are about to run
    state["paths"] = paths.__class__(
        **{
            **paths.__dict__,
            "scenario_table": scenario_path,
            "config": exp_paths.config or paths.config,
            "rbs_data": consignment_path if consignment_path and consignment_path.exists() else paths.__dict__.get("rbs_data"),
            "compliance_lookup": compliance_path if compliance_path and compliance_path.exists() else paths.__dict__.get("compliance_lookup"),
        }
    )

    # Track num_consignments even on failure (best-effort)
    state["num_consignments"] = None

    try:
        run_seed = engine_options.get("seed")
        if run_seed in (None, "", 0):
            from time import time  # pylint: disable=import-outside-toplevel

            run_seed = int(time())
        result = run_slippage_pipeline(
            exp_paths,
            seed=run_seed,
            num_simulations=engine_options.get("num_simulations", 1),
        )
    except Exception as exc:  # pylint: disable=broad-except
        # Best-effort inference of num_consignments for debugging
        try:
            import pandas as pd  # pylint: disable=import-outside-toplevel
            cons_guess = exp_paths.consignment
            if cons_guess and cons_guess.exists():
                df = pd.read_csv(cons_guess)
                if "INSPECTION_NUMBER" in df.columns:
                    state["num_consignments"] = int(df["INSPECTION_NUMBER"].nunique())
                elif "INSPECTION_ID" in df.columns:
                    state["num_consignments"] = int(df["INSPECTION_ID"].nunique())
            else:
                scen_df = pd.read_csv(exp_paths.scenario_table)
                val = scen_df.get("consignment/input_file/file_name", pd.Series()).dropna()
                if not val.empty:
                    cons_path = Path(str(val.iloc[0]))
                    if not cons_path.is_absolute():
                        cons_path = exp_paths.experiment_dir / cons_path.name
                    if cons_path.exists():
                        df = pd.read_csv(cons_path)
                        if "INSPECTION_NUMBER" in df.columns:
                            state["num_consignments"] = int(df["INSPECTION_NUMBER"].nunique())
                        elif "INSPECTION_ID" in df.columns:
                            state["num_consignments"] = int(df["INSPECTION_ID"].nunique())
        except Exception:
            pass
        # Surface the underlying error directly
        print(f"Pipeline failed: {exc}")
        raise
    # Persist results so Page 5 can render visuals
    state["results"] = result.scenario_results
    state["num_consignments"] = result.num_consignments
    state["run_error"] = None
    return result


__all__ = [
    "STATE_KEY",
    "init_slippage_state",
    "get_slippage_state",
    "set_paths",
    "set_scenario_dataframe",
    "set_synthetic_options",
    "set_engine_options",
    "run_pipeline",
]
