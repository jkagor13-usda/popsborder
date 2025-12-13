from pathlib import Path
from typing import Optional, Union
import json
import re
import shutil

import pandas as pd
import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.slippage_ui import get_slippage_state

# --- Constants / setup --------------------------------------------------------
TMP_DIR = Path("tmp")
SCENARIO_ROOT = TMP_DIR / "experiments"
TEMPLATE_SCENARIO = Path("data_input") / "pis_contaminate_scenarios.csv"
CONTAM_PARAM_PATH = TMP_DIR / "contamination" / "contamination_parameter_sets.json"
SCENARIO_FILENAME = "scenario_table.csv"
CONS_FILENAME = "consignment_uploaded_rbs_data.csv"
COMPLIANCE_FILENAME = "compliance_table.csv"
CONFIG_FILENAME = "config.yml"

TMP_DIR.mkdir(exist_ok=True)
SCENARIO_ROOT.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Scenario & Experiment Builder",
    page_icon=":test_tube:",
    layout="wide",
)

init_state()
state = get_slippage_state()
render_sidebar_navigation()

# --- Helpers ------------------------------------------------------------------
def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    return slug or "scenario"


def _list_files(folder: Path, pattern: str) -> list[Path]:
    return sorted(folder.glob(pattern)) if folder.exists() else []


def _load_param_sets() -> dict:
    if not CONTAM_PARAM_PATH.exists():
        return {}
    try:
        return json.loads(CONTAM_PARAM_PATH.read_text())
    except Exception:  # pylint: disable=broad-except
        return {}

def _copy_inputs(rows_df: pd.DataFrame, scenario_dir: Path) -> None:
    """Copy consignment/compliance files and contamination params/config into scenario_dir with standard names."""
    missing: list[str] = []
    # Copy consignment file as CONS_FILENAME
    cons_col = "consignment/input_file/file_name"
    if cons_col in rows_df.columns:
        vals = [v for v in rows_df[cons_col].dropna().unique().tolist() if v]
        if vals:
            name = Path(str(vals[0])).name
            src_candidates = [
                Path(str(vals[0])),
                Path("tmp") / "consignments" / name,
                Path(name),
            ]
            src = next((p for p in src_candidates if p.exists()), None)
            if src:
                dest = scenario_dir / CONS_FILENAME
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(src.read_bytes())
            else:
                missing.append(f"consignment file '{name}'")
        else:
            missing.append("consignment file (none listed)")
    else:
        missing.append("consignment file column missing")

    # Copy compliance file as COMPLIANCE_FILENAME
    comp_col = "inspection/compliance_table/file_name"
    if comp_col in rows_df.columns:
        vals = [v for v in rows_df[comp_col].dropna().unique().tolist() if v]
        if vals:
            name = Path(str(vals[0])).name
            src_candidates = [
                Path(str(vals[0])),
                Path("tmp") / "compliance" / name,
                Path(name),
            ]
            src = next((p for p in src_candidates if p.exists()), None)
            if src:
                dest = scenario_dir / COMPLIANCE_FILENAME
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(src.read_bytes())
            else:
                missing.append(f"compliance file '{name}'")
        else:
            missing.append("compliance file (none listed)")
    else:
        missing.append("compliance file column missing")

    # Copy contamination parameter sets snapshot
    if CONTAM_PARAM_PATH.exists():
        (scenario_dir / "contamination_parameter_sets.json").write_bytes(
            CONTAM_PARAM_PATH.read_bytes()
        )

    # Copy config
    config_src = (
        state["paths"].config
        if state.get("paths") and getattr(state["paths"], "config", None)
        else Path("data_input") / "config.yml"
    )
    config_src = Path(config_src)
    if config_src.exists():
        (scenario_dir / CONFIG_FILENAME).write_bytes(config_src.read_bytes())
    if missing:
        raise FileNotFoundError("; ".join(missing))


def _normalize_rows(rows_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize scenario rows so downstream RBS run does not fail."""
    df = rows_df.copy()
    if "inspection/unit" in df.columns:
        df["inspection/unit"] = "sample_units"
    if "inspection/sample_strategy" in df.columns:
        df["inspection/sample_strategy"] = df["inspection/sample_strategy"].replace("", "rbs").fillna("rbs")
    if "inspection/proportion/value" in df.columns:
        def _norm_prop(val):
            try:
                v = float(val)
            except Exception:
                v = 0.0
            return 0.02 if v <= 0 else v
        df["inspection/proportion/value"] = df["inspection/proportion/value"].apply(_norm_prop)
    alpha_col = "contamination/contamination_rate/beta_binomial_parameters/alpha"
    beta_col = "contamination/contamination_rate/beta_binomial_parameters/beta"
    if alpha_col in df.columns:
        df[alpha_col] = df[alpha_col].apply(lambda v: 0.01 if pd.isna(v) or float(v) <= 0 else float(v))
    if beta_col in df.columns:
        df[beta_col] = df[beta_col].apply(lambda v: 5.0 if pd.isna(v) or float(v) <= 0 else float(v))
    return df


# --- Page header --------------------------------------------------------------
st.title("Page 4 - Experiment Builder")
st.caption(
    "Assemble scenarios using outputs from Pages 1-3: pick consignments (RBS), contamination parameter set, "
    "and compliance table. Saved experiment packages are written to tmp/experiments."
)

tabs = st.tabs(["Upload custom scenario", "Build experiments", "Saved experiments"])

# --- Tab 1: Upload custom scenario -------------------------------------------
with tabs[0]:
    st.subheader("Upload custom scenario table")
    uploaded = st.file_uploader("Upload scenario CSV", type=["csv"], key="custom_scenario_upload")
    custom_name = st.text_input("Save as experiment set name", value="custom_experiment")

    if uploaded:
        try:
            uploaded.seek(0)
            df_preview = pd.read_csv(uploaded)
            st.dataframe(df_preview.head(50), use_container_width=True)
        except Exception:  # pylint: disable=broad-except
            st.info("Unable to preview upload.")

    if uploaded and st.button("Save custom scenario table", type="primary"):
        try:
            uploaded.seek(0)
            df = pd.read_csv(uploaded)
            set_slug = _slugify(custom_name)
            scenario_dir = SCENARIO_ROOT / set_slug
            scenario_dir.mkdir(parents=True, exist_ok=True)
            dest = scenario_dir / "scenario_table.csv"
            df.to_csv(dest, index=False)
            state["paths"] = state["paths"].__class__(
                **{**state["paths"].__dict__, "scenario_table": dest}
            )
            st.success(f"Saved custom scenario table to {dest}")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed to save custom scenario table: {exc}")

# --- Tab 2: Build experiments -------------------------------------------------
with tabs[1]:
    state.setdefault("experiment_rows", [])
    col_left, col_right = st.columns([2, 1])

    consignment_files = _list_files(TMP_DIR / "consignments", "*.csv")
    compliance_files = _list_files(TMP_DIR / "compliance", "*.csv")
    param_sets = _load_param_sets()
    param_keys = list(param_sets.keys())

    with col_left:
        scenario_label = st.text_input(
            "Scenario label",
            value=f"scenario_{len(state['experiment_rows']) + 1}",
        )

        consignment_choice = (
            st.selectbox(
                "Consignment (RBS) file",
                consignment_files,
                format_func=lambda p: p.name,
            )
            if consignment_files
            else None
        )

        compliance_choice = (
            st.selectbox(
                "Compliance table",
                compliance_files,
                format_func=lambda p: p.name,
            )
            if compliance_files
            else None
        )

        param_choice = (
            st.selectbox("Contamination parameter set", param_keys)
            if param_keys
            else None
        )

    with col_right:
        with st.expander("Files available", expanded=True):
            st.write(
                f"**Consignments ({len(consignment_files)}):** "
                f"{', '.join(p.name for p in consignment_files) if consignment_files else 'none'}"
            )
            st.write(
                f"**Compliance tables ({len(compliance_files)}):** "
                f"{', '.join(p.name for p in compliance_files) if compliance_files else 'none'}"
            )
            st.write(
                f"**Param sets ({len(param_keys)}):** "
                f"{', '.join(param_keys) if param_keys else 'none'}"
            )

    add_ready = all([scenario_label, consignment_choice, compliance_choice, param_choice])
    if st.button("Add scenario row", type="primary", disabled=not add_ready):
        template_cols = (
            pd.read_csv(TEMPLATE_SCENARIO, nrows=0).columns.tolist()
            if TEMPLATE_SCENARIO.exists()
            else []
        )
        param_snapshot = param_sets.get(param_choice, {})
        scenario_row = {col: "" for col in template_cols} if template_cols else {}
        scenario_row.update(
            {
                "name": scenario_label,
                "consignment/input_file/file_name": consignment_choice.name,
                "inspection/compliance_table/file_name": compliance_choice.name,
                "consignment/generation_method": "RBS",
                "consignment/input_file/file_type": "RBS",
                "contamination/contamination_unit": "plant",
                "contamination/contamination_rate/distribution": "beta-binomial",
                "contamination/contamination_rate/value": "",
                "contamination/contamination_rate/beta_binomial_parameters/alpha": param_snapshot.get(
                    "alpha"
                ),
                "contamination/contamination_rate/beta_binomial_parameters/beta": param_snapshot.get(
                    "beta"
                ),
                "contamination/contamination_rate/beta_binomial_parameters/theta": param_snapshot.get(
                    "theta"
                ),
                "contamination/arrangement": "random",
                "inspection/sample_strategy": "rbs",
                "inspection/proportion/value": 0.02,
                "inspection/unit": "sample_units",
                "inspection/min_boxes": 0,
                "inspection/selection_strategy": "random",
                "inspection/within_box_proportion": 1,
            }
        )
        state["experiment_rows"].append(scenario_row)
        st.success(f"Added scenario row '{scenario_label}'")

    rows_df = pd.DataFrame(state["experiment_rows"])
    if not rows_df.empty:
        st.dataframe(rows_df, use_container_width=True)
    else:
        st.info("Add at least one scenario row.")

    col_actions = st.columns(2)
    with col_actions[0]:
        if st.button("Clear current rows", type="secondary", disabled=rows_df.empty):
            state["experiment_rows"] = []
            st.rerun()
    with col_actions[1]:
        if st.button("Remove last row", type="secondary", disabled=rows_df.empty):
            if state["experiment_rows"]:
                state["experiment_rows"].pop()
            st.rerun()

    scenario_set = st.text_input("Experiment set name (CSV)", value="experiment_set_1")
    can_save = bool(scenario_set) and not rows_df.empty

    if st.button("Save experiment package", type="primary", disabled=not can_save):
        set_slug = _slugify(scenario_set)
        scenario_dir = SCENARIO_ROOT / set_slug
        try:
            scenario_dir.mkdir(parents=True, exist_ok=True)
            scenario_path = scenario_dir / SCENARIO_FILENAME
            # Always use forward-slash tmp-relative paths in the saved scenario table
            base_prefix = f"tmp/experiments/{set_slug}"

            # Template columns
            raw_rows_df = rows_df.copy()
            template_cols = (
                pd.read_csv(TEMPLATE_SCENARIO, nrows=0).columns.tolist()
                if TEMPLATE_SCENARIO.exists()
                else rows_df.columns.tolist()
            )
            # Normalize legacy names
            if "consignment name" in template_cols and "consignment/input_file/file_name" not in template_cols:
                template_cols = [c for c in template_cols if c != "consignment name"]
                template_cols.append("consignment/input_file/file_name")
            if "inspection name" in template_cols and "inspection/compliance_table/file_name" not in template_cols:
                template_cols = [c for c in template_cols if c != "inspection name"]
                template_cols.append("inspection/compliance_table/file_name")

            # Keep file path columns together before unnamed columns
            ordered_targets = [
                "consignment/input_file/file_name",
                "inspection/compliance_table/file_name",
            ]
            filtered_cols = [c for c in template_cols if c not in ordered_targets]
            anchor = next(
                (i for i, c in enumerate(filtered_cols) if str(c).startswith("Unnamed")),
                len(filtered_cols),
            )
            template_cols = (
                filtered_cols[:anchor]
                + [c for c in ordered_targets if c in template_cols]
                + filtered_cols[anchor:]
            )

            # Make file paths portable (within scenario_dir, forward slashes)
            if "consignment/input_file/file_name" in rows_df.columns:
                rows_df["consignment/input_file/file_name"] = f"{base_prefix}/{CONS_FILENAME}"
            if "inspection/compliance_table/file_name" in rows_df.columns:
                rows_df["inspection/compliance_table/file_name"] = f"{base_prefix}/{COMPLIANCE_FILENAME}"

            rows_df = _normalize_rows(rows_df)
            scenario_table = rows_df.reindex(columns=template_cols, fill_value="")
            scenario_table.to_csv(scenario_path, index=False)

            _copy_inputs(raw_rows_df, scenario_dir)

            state["paths"] = state["paths"].__class__(
                **{**state["paths"].__dict__, "scenario_table": scenario_path}
            )
            st.success(f"Experiment saved to {scenario_path}")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed to save experiment: {exc}")

# --- Tab 3: Saved experiments -------------------------------------------------
with tabs[2]:
    st.subheader("Saved experiments (tmp/experiments)")
    saved_files = list(SCENARIO_ROOT.glob("*/scenario_table.csv"))

    if not saved_files:
        st.info("No experiments saved yet.")
    else:
        sel = st.selectbox(
            "Select an experiment set", saved_files, format_func=lambda p: p.parent.name
        )
        try:
            preview = pd.read_csv(sel)
            st.dataframe(preview, use_container_width=True)
            st.caption(f"Path: {sel}")
            if st.button("Delete this experiment set", type="secondary"):
                try:
                    shutil.rmtree(sel.parent)
                    st.success(f"Deleted {sel.parent}")
                    st.rerun()
                except Exception as exc:  # pylint: disable=broad-except
                    st.error(f"Unable to delete experiment: {exc}")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unable to preview experiment: {exc}")

# --- Bottom navigation --------------------------------------------------------
st.divider()
nav_cols = st.columns(2)

with nav_cols[0]:
    if st.button("Reset and Return Home", type="secondary"):
        try:
            shutil.rmtree(TMP_DIR)
        except Exception:
            pass
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        st.session_state.clear()
        init_state()
        st.switch_page("frontend.py")

with nav_cols[1]:
    prev_next = st.columns(2)
    with prev_next[0]:
        if st.button("Previous Page", type="primary", key="nav_back_page4"):
            st.switch_page("pages/3_Inspection_Process.py")
    with prev_next[1]:
        if st.button("Next Page", type="primary", key="nav_forward_page5"):
            st.switch_page("pages/5_Run_Simulation.py")
