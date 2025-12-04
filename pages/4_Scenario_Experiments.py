from pathlib import Path
import shutil
import re

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
tmp_dir = Path("tmp")
tmp_dir.mkdir(exist_ok=True)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    return slug or "scenario"


def _compliance_options() -> list[tuple[str, Path]]:
    seen = {}
    candidates = []
    current = Path(state["paths"].compliance_lookup)
    if current.exists():
        candidates.append(current)
    manual = tmp_dir / "manual_compliance_table.csv"
    if manual.exists():
        candidates.append(manual)
    upload = tmp_dir / "compliance_table.csv"
    if upload.exists():
        candidates.append(upload)
    for path in (tmp_dir / "experiments").glob("*/compliance_table.csv"):
        candidates.append(path)
    options: list[tuple[str, Path]] = []
    for path in candidates:
        key = str(path.resolve())
        if key in seen:
            continue
        seen[key] = True
        label = path.name
        if path.parent.name != "tmp":
            label = f"{path.parent.name}/{path.name}"
        options.append((label, path))
    return options

st.title("Page 4 - Scenario & Experiment Builder")
st.caption(
    "Summarize configured scenarios and decide on the experiment setup "
    "(number of simulations and expected time). Consignments per run "
    "now mirror however many synthetic records were generated on Page 1. Choose the compliance table to bundle "
    "with your scenario file created here."
)

st.subheader("Create scenario package & experimental setup")
with st.form("scenario_package_form"):
    col_name, col_comp = st.columns(2)
    scenario_name = col_name.text_input("Scenario/experiment name", value="scenario_1")
    available_compliance = _compliance_options()
    if available_compliance:
        labels = [lbl for lbl, _ in available_compliance]
        selected_label = col_comp.selectbox("Compliance table", labels)
        selected_path = dict(available_compliance)[selected_label]
    else:
        selected_label = None
        selected_path = None
        col_comp.warning("No compliance tables found. Create one on Page 3 first.")

    scenario_note = st.text_input("Optional note/description", value="")
    num_simulations = st.number_input(
        "Simulation repetitions",
        min_value=1,
        max_value=500,
        value=int(engine_options.get("num_simulations", 1)),
        step=1,
    )
    submit_package = st.form_submit_button(
        "Save scenario package and experimental setup",
        use_container_width=True,
        disabled=scenario_df.empty or not selected_path,
    )

if submit_package:
    slug = _slugify(scenario_name)
    scenario_dir = tmp_dir / "experiments" / slug
    try:
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_path = scenario_dir / "scenario_table.csv"
        scenario_df.to_csv(scenario_path, index=False)
        if selected_path:
            shutil.copy(selected_path, scenario_dir / "compliance_table.csv")
        if scenario_note:
            (scenario_dir / "README.txt").write_text(scenario_note)
        set_engine_options(num_simulations=int(num_simulations))
        st.success(f"Scenario package saved to {scenario_dir} and experimental setup recorded.")
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Failed to save scenario package: {exc}")

scenario_count = max(1, len(scenario_df))
consignments_per_run = state.get("num_consignments")
if consignments_per_run is None:
    synthetic_preview = state.get("synthetic_data")
    consignments_per_run = None if synthetic_preview is None else len(synthetic_preview)
estimated_minutes = (
    num_simulations * (consignments_per_run or 100) * scenario_count / 120.0
)  # simple heuristic

st.markdown("#### Estimated simulation time")
st.info(
    f"Based on the selected setup, {scenario_count} scenarios, and "
    f"{consignments_per_run or 'unknown'} consignments per run (derived from synthetic data), "
    f"expect approximately {estimated_minutes:.1f} minutes of compute time per full run (heuristic)."
)

st.divider()
if scenario_df.empty:
    st.warning("The scenario table is empty. Build consignments on Page 1 first.")
else:
    st.subheader("Scenario overview")
    overview_cols = st.columns(4)
    overview_cols[0].metric("Scenario rows", len(scenario_df))
    if "inspection/compliance_level" in scenario_df.columns:
        overview_cols[1].metric(
            "Compliance levels",
            ", ".join(sorted(scenario_df["inspection/compliance_level"].dropna().unique().tolist())),
        )
    else:
        overview_cols[1].metric("Compliance levels", "n/a")
    if "inspection/sample_strategy" in scenario_df.columns:
        overview_cols[2].metric(
            "Sample strategies",
            scenario_df["inspection/sample_strategy"].nunique(),
        )
    else:
        overview_cols[2].metric("Sample strategies", "n/a")
    if "inspection/proportion/value" in scenario_df.columns:
        avg_prop = pd.to_numeric(scenario_df["inspection/proportion/value"], errors="coerce").mean()
        overview_cols[3].metric("Avg inspection proportion", f"{avg_prop:.3f}" if pd.notna(avg_prop) else "n/a")
    else:
        overview_cols[3].metric("Avg inspection proportion", "n/a")

    st.dataframe(scenario_df, use_container_width=True, height=320)
    st.download_button(
        "Download scenario table (CSV)",
        data=scenario_df.to_csv(index=False).encode("utf-8"),
        file_name="scenario_table.csv",
        use_container_width=True,
    )

st.divider()
nav_cols = st.columns(2)
with nav_cols[0]:
    if st.button("Back to Page 4", type="primary", key="nav_back_page4"):
        st.switch_page("pages/4_Inspection_Process.py")
with nav_cols[1]:
    if st.button("Continue to Page 6", type="primary", key="nav_forward_page6"):
        st.switch_page("pages/6_Run_Simulation.py")

