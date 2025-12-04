from pathlib import Path
from typing import Optional
import shutil

import pandas as pd
import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.slippage_ui import get_slippage_state, set_paths, create_default_paths


st.set_page_config(
    page_title="Inspection Process",
    page_icon=":mag_right:",
    layout="wide",
)
init_state()

state = get_slippage_state()
render_sidebar_navigation()
paths = state["paths"]
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)


def _load_compliance_table(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


def _compliance_level_counts(series: pd.Series):
    if series is None:
        return {"Low": 0, "Medium": 0, "High": 0}
    normalized = series.astype(str).str.lower()
    return {
        "Low": normalized.str.contains("low").sum(),
        "Medium": normalized.str.contains("med").sum(),
        "High": normalized.str.contains("high").sum(),
    }


st.title("Page 3 - Inspection Process")
st.caption(
    "Ingest the compliance lookup table, configure low/medium/high types, and review RBS parameters."
)

tabs = st.tabs(["Upload table", "Create policy manually"])
with tabs[0]:
    st.subheader("Compliance table (upload)")
    col_uploads = st.columns(2)
    compliance_upload = col_uploads[0].file_uploader("Upload compliance table CSV", type=["csv"])
    if compliance_upload is not None:
        try:
            new_compliance_df = pd.read_csv(compliance_upload)
            target_path = TMP_DIR / "compliance_table.csv"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            new_compliance_df.to_csv(target_path, index=False)
            set_paths(compliance_lookup=target_path)
            st.success(f"Compliance table saved to {target_path}")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed to save compliance table: {exc}")

with tabs[1]:
    st.subheader("Compliance table (manual)")
    st.caption("Build a simple compliance lookup with entity and compliance level.")
    default_rows = [{"entity": "Example Producer", "compliance_level": lvl} for lvl in ["Low", "Medium", "High"]]
    manual_df = state.get("manual_compliance_df")
    if manual_df is None:
        manual_df = pd.DataFrame(default_rows)
    edited_df = st.data_editor(
        manual_df,
        num_rows="dynamic",
        use_container_width=True,
        key="compliance_editor",
    )
    if st.button("Save manual compliance table", type="secondary"):
        target_path = Path("tmp") / "manual_compliance_table.csv"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        edited_df.to_csv(target_path, index=False)
        state["manual_compliance_df"] = edited_df
        set_paths(compliance_lookup=target_path)
        st.success(f"Manual compliance table saved to {target_path}")

compliance_df = _load_compliance_table(state["paths"].compliance_lookup)
if compliance_df is None:
    st.warning("Compliance table not found. Upload a CSV to proceed.")
else:
    st.dataframe(compliance_df, use_container_width=True, height=280)
    string_columns = [col for col in compliance_df.columns if compliance_df[col].dtype == "object"]
    selected_column = string_columns[0] if string_columns else None
    counts = _compliance_level_counts(compliance_df[selected_column]) if selected_column else {"Low": 0, "Medium": 0, "High": 0}
    level_cols = st.columns(3)
    level_cols[0].metric("Low type rows", counts["Low"])
    level_cols[1].metric("Medium type rows", counts["Medium"])
    level_cols[2].metric("High type rows", counts["High"])

st.divider()
nav_cols = st.columns(3)
with nav_cols[0]:
    if st.button(
        "Reset and Return Home",
        type="secondary",
        key="nav_reset_page4",
        help="Delete temporary files and restart from the home page",
    ):
        try:
            if TMP_DIR.exists():
                shutil.rmtree(TMP_DIR)
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            state["paths"] = create_default_paths()
            st.switch_page("frontend.py")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unable to reset temporary files: {exc}")
with nav_cols[1]:
    if st.button("Previous Page", type="primary", key="nav_back_page3_proc"):
        st.switch_page("pages/2_Contamination_Fit.py")
with nav_cols[2]:
    if st.button("Next Page", type="primary", key="nav_forward_page5_proc"):
        st.switch_page("pages/4_Scenario_Experiments.py")

st.info("Inspection scenarios are managed on the experiment setup page.")

