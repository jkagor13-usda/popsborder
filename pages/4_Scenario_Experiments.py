from pathlib import Path
import re
import shutil

import pandas as pd
import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.slippage_ui import get_slippage_state, set_engine_options, create_default_paths

st.set_page_config(
    page_title="Scenario & Experiment Builder",
    page_icon=":test_tube:",
    layout="wide",
)
init_state()

state = get_slippage_state()
render_sidebar_navigation()
scenario_df = state["scenario_df"]
engine_options = state["engine_options"]
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)
SCENARIO_ROOT = TMP_DIR / "experiments"
SCENARIO_ROOT.mkdir(parents=True, exist_ok=True)
MASTER_SCENARIO_TABLE = SCENARIO_ROOT / "scenario_table.csv"
TEMPLATE_SCENARIO = Path("data_input") / "pis_contaminate_scenarios.csv"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    return slug or "scenario"


def _list_files(folder: Path, pattern: str) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob(pattern))


def _load_param_sets() -> dict:
    param_path = TMP_DIR / "contamination" / "contamination_parameter_sets.json"
    if not param_path.exists():
        return {}
    try:
        import json  # pylint: disable=import-outside-toplevel

        return json.loads(param_path.read_text())
    except Exception:  # pylint: disable=broad-except
        return {}


st.title("Page 4 - Experiment Builder")
st.caption(
    "Assemble scenarios using outputs from Pages 1-3: pick consignments (RBS), contamination parameter set, "
    "and compliance table. Saved experiment packages are written to tmp/experiments."
)

tabs = st.tabs(["Upload custom scenario", "Build experiments", "Saved experiments"])

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
            # update master table
            master_df = pd.DataFrame()
            if MASTER_SCENARIO_TABLE.exists():
                try:
                    master_df = pd.read_csv(MASTER_SCENARIO_TABLE)
                except Exception:  # pylint: disable=broad-except
                    master_df = pd.DataFrame()
            master_df = pd.concat([master_df, df], ignore_index=True).drop_duplicates()
            MASTER_SCENARIO_TABLE.parent.mkdir(parents=True, exist_ok=True)
            MASTER_SCENARIO_TABLE.write_text(master_df.to_csv(index=False))
            state["paths"] = state["paths"].__class__(**{**state["paths"].__dict__, "scenario_table": dest})
            st.success(f"Saved custom scenario table to {dest}")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed to save custom scenario table: {exc}")


with tabs[1]:
    state.setdefault("experiment_rows", [])
    col_left, col_right = st.columns([2, 1])

    with col_left:
        scenario_label = st.text_input("Scenario label", value=f"scenario_{len(state['experiment_rows'])+1}")
        scenario_note = st.text_input("Optional note/description", value="")

        consignment_files = _list_files(TMP_DIR / "consignments", "*.csv")
        consignment_choice = st.selectbox(
            "Consignment (RBS) file",
            consignment_files,
            format_func=lambda p: p.name,
        ) if consignment_files else None

        compliance_files = _list_files(TMP_DIR / "compliance", "*.csv")
        compliance_choice = st.selectbox(
            "Compliance table",
            compliance_files,
            format_func=lambda p: p.name,
        ) if compliance_files else None

        param_sets = _load_param_sets()
        param_keys = list(param_sets.keys())
        param_choice = st.selectbox(
            "Contamination parameter set",
            param_keys,
        ) if param_keys else None

    with col_right:
        num_simulations = st.number_input(
            "Simulation repetitions",
            min_value=1,
            max_value=500,
            value=int(engine_options.get("num_simulations", 1)),
            step=1,
        )
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=10_000_000,
            value=int(engine_options.get("seed", 42)),
            step=1,
        )

        st.markdown("**Files available**")
        st.caption(f"Consignments: {len(consignment_files)} | Compliance tables: {len(compliance_files)} | Param sets: {len(param_keys)}")

    add_ready = all([scenario_label, consignment_choice, compliance_choice, param_choice])
    if st.button("Add scenario row", type="primary", disabled=not add_ready):
        template_cols = (
            pd.read_csv(TEMPLATE_SCENARIO, nrows=0).columns.tolist()
            if TEMPLATE_SCENARIO.exists()
            else []
        )
        param_snapshot = param_sets.get(param_choice, {})
        alpha = param_snapshot.get("alpha", None)
        beta = param_snapshot.get("beta", None)
        theta = param_snapshot.get("theta", None)
        scenario_row = {col: "" for col in template_cols} if template_cols else {}
        scenario_row.update(
            {
                "name": scenario_label,
                "consignment name": consignment_choice.name,
                "inspection name": compliance_choice.name,
                "consignment/generation_method": "RBS",
                "consignment/input_file/file_type": "RBS",
                "contamination/contamination_unit": "plant",
                "contamination/contamination_rate/distribution": "beta-binomial",
                "contamination/contamination_rate/value": "",
                "contamination/contamination_rate/beta_binomial_parameters/alpha": alpha,
                "contamination/contamination_rate/beta_binomial_parameters/beta": beta,
                "contamination/contamination_rate/beta_binomial_parameters/theta": theta,
                "contamination/arrangement": "random",
                "inspection/sample_strategy": "rbs",
                "inspection/proportion/value": 0.02,
            }
        )
        state["experiment_rows"].append(scenario_row)
        st.success(f"Added scenario row '{scenario_label}'")

    rows_df = pd.DataFrame(state["experiment_rows"])
    if not rows_df.empty:
        st.dataframe(rows_df, use_container_width=True)
    else:
        st.info("Add at least one scenario row.")

    scenario_set = st.text_input("Experiment set name (CSV)", value="experiment_set_1")

    can_save = all([scenario_set]) and not rows_df.empty
    if st.button("Save experiment package", type="primary", disabled=not can_save):
        set_slug = _slugify(scenario_set)
        scenario_dir = SCENARIO_ROOT / set_slug
        try:
            scenario_dir.mkdir(parents=True, exist_ok=True)
            scenario_path = scenario_dir / "scenario_table.csv"
            template_cols = (
                pd.read_csv(TEMPLATE_SCENARIO, nrows=0).columns.tolist()
                if TEMPLATE_SCENARIO.exists()
                else rows_df.columns.tolist()
            )
            scenario_table = rows_df.reindex(columns=template_cols, fill_value="")
            # Append to set-specific table
            existing = pd.DataFrame()
            if scenario_path.exists():
                try:
                    existing = pd.read_csv(scenario_path)
                except Exception:  # pylint: disable=broad-except
                    existing = pd.DataFrame(columns=template_cols)
            existing = existing.reindex(columns=scenario_table.columns, fill_value="")
            scenario_table = pd.concat([existing, scenario_table], ignore_index=True)
            scenario_table.to_csv(scenario_path, index=False)
            # Save contamination params snapshot for reference
            import json  # pylint: disable=import-outside-toplevel
            (scenario_dir / "contamination_params.json").write_text(
                json.dumps(_load_param_sets(), indent=2)
            )
            if scenario_note:
                (scenario_dir / "README.txt").write_text(scenario_note)
            # Update engine options and state paths
            set_engine_options(num_simulations=int(num_simulations), seed=int(seed))
            # Update master scenario table
            master_df = pd.DataFrame()
            if MASTER_SCENARIO_TABLE.exists():
                try:
                    master_df = pd.read_csv(MASTER_SCENARIO_TABLE)
                except Exception:  # pylint: disable=broad-except
                    master_df = pd.DataFrame(columns=scenario_table.columns)
            master_df = master_df.reindex(columns=scenario_table.columns, fill_value="")
            master_df = pd.concat([master_df, scenario_table], ignore_index=True)
            MASTER_SCENARIO_TABLE.parent.mkdir(parents=True, exist_ok=True)
            MASTER_SCENARIO_TABLE.write_text(master_df.to_csv(index=False))

            state["paths"] = state["paths"].__class__(**{**state["paths"].__dict__, "scenario_table": MASTER_SCENARIO_TABLE})
            st.success(f"Experiment saved to {scenario_dir} and appended to {MASTER_SCENARIO_TABLE}")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed to save experiment: {exc}")

with tabs[2]:
    st.subheader("Saved experiments (tmp/experiments)")
    saved_files = list(SCENARIO_ROOT.glob("*/scenario_table.csv"))
    if not saved_files:
        st.info("No experiments saved yet.")
    else:
        sel = st.selectbox("Select an experiment set", saved_files, format_func=lambda p: p.parent.name)
        try:
            preview = pd.read_csv(sel)
            st.dataframe(preview, use_container_width=True)
            st.caption(f"Path: {sel}")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unable to preview experiment: {exc}")

st.divider()
nav_cols = st.columns(2)
with nav_cols[0]:
    if st.button("Back to Page 3", type="primary", key="nav_back_page3"):
        st.switch_page("pages/3_Inspection_Process.py")
with nav_cols[1]:
    if st.button("Continue to Page 5", type="primary", key="nav_forward_page5"):
        st.switch_page("pages/5_Run_Simulation.py")
