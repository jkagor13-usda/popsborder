import pandas as pd
import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.slippage_ui import get_slippage_state, set_engine_options


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

st.title("Page 5 - Scenario & Experiment Builder")
st.caption(
    "Summarize configured scenarios and decide on the experiment setup "
    "(number of simulations, random seed, and expected time). Consignments per run "
    "now mirror however many synthetic records were generated on Page 1."
)

if scenario_df.empty:
    st.warning("The scenario table is empty. Build consignments on Page 2 first.")
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
st.subheader("Experimental setup")
with st.form("experiment_form"):
    c1, c2 = st.columns(2)
    num_simulations = c1.number_input(
        "Simulation repetitions",
        min_value=1,
        max_value=500,
        value=int(engine_options.get("num_simulations", 1)),
        step=1,
    )
    random_seed = c2.number_input(
        "Random seed",
        min_value=0,
        max_value=999_999,
        value=int(engine_options.get("seed", 42)),
        step=1,
    )
    submit_experiment = st.form_submit_button("Save experimental setup", use_container_width=True)

if submit_experiment:
    set_engine_options(
        num_simulations=int(num_simulations),
        seed=int(random_seed),
    )
    st.success("Experimental setup saved. Page 5 will use these values.")

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
nav_cols = st.columns(2)
with nav_cols[0]:
    if st.button("Back to Page 4", type="primary", key="nav_back_page4"):
        st.switch_page("pages/4_Inspection_Process.py")
with nav_cols[1]:
    if st.button("Continue to Page 6", type="primary", key="nav_forward_page6"):
        st.switch_page("pages/6_Run_Simulation.py")

