from pathlib import Path

import pandas as pd
import altair as alt
import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.slippage_ui import get_slippage_state, run_pipeline, set_engine_options
from gui.slippage_pipeline import create_default_paths


st.set_page_config(
    page_title="Run Simulation",
    page_icon=":arrow_forward:",
    layout="wide",
)
init_state()

state = get_slippage_state()
render_sidebar_navigation()
engine_options = state["engine_options"]
run_error = state.get("run_error")
paths = state["paths"]


def _rbs_ready() -> tuple[bool, str]:
    """Check that the RBS input exists and has key columns before running."""
    # Use defaults if state is missing paths
    default_paths = create_default_paths()
    if paths.rbs_data is None and default_paths.rbs_data:
        state["paths"] = paths.__class__(**{**paths.__dict__, "rbs_data": default_paths.rbs_data})
    current_paths = state["paths"]
    try:
        if current_paths.rbs_data is None:
            return False, "No RBS calculator file selected. Upload on Page 1."
        rbs_path = Path(current_paths.rbs_data)
        if not rbs_path.exists():
            return False, f"RBS calculator file not found at {rbs_path}."
        df = pd.read_csv(rbs_path, nrows=200)
        required = {"TOTAL_SAMPLING_UNITS", "TOTAL_PLANT_QUANTITY"}
        missing = required - set(df.columns)
        if missing:
            return False, f"RBS calculator file is missing required columns: {', '.join(sorted(missing))}"
        if df[["TOTAL_SAMPLING_UNITS", "TOTAL_PLANT_QUANTITY"]].dropna().empty:
            return False, "RBS calculator file has no non-empty sampling/plant quantity rows."
        return True, ""
    except Exception as exc:  # pylint: disable=broad-except
        return False, f"Unable to validate RBS calculator file: {exc}"

st.title("Page 5 - Run Simulation")
st.caption("Execute the slippage pipeline and compare policies based on slippage metrics.")

with st.sidebar:
    st.subheader("Execution options")
    seed = st.number_input("Random seed", value=int(engine_options.get("seed", 42)), step=1)
    simulations = st.number_input(
        "Simulation repetitions",
        min_value=1,
        max_value=500,
        value=int(engine_options.get("num_simulations", 1)),
        step=1,
    )
    if (
        seed != engine_options.get("seed")
        or simulations != engine_options.get("num_simulations")
    ):
        set_engine_options(seed=int(seed), num_simulations=int(simulations))
    consignment_count = state.get("num_consignments")
    if consignment_count is None:
        synth_total = state.get("synthetic_data")
        if synth_total is not None:
            consignment_count = len(synth_total)
    st.caption(
        "Consignments per scenario are derived from the available data: "
        f"{consignment_count if consignment_count is not None else 'generate data on Page 1'}."
    )

    st.divider()
    rbs_ok, rbs_msg = _rbs_ready()
    if not rbs_ok:
        st.warning(rbs_msg)
    run_now = st.button("Run pipeline", use_container_width=True, disabled=not rbs_ok)
    if run_now:
        with st.spinner("Running slippage pipeline..."):
            try:
                run_pipeline()
                st.success("Pipeline finished.")
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Pipeline failed: {exc!r}")

if run_error:
    st.error(f"Last pipeline error: {run_error}")
    st.caption(
        "Verify that Page 1 has RBS data (or synthetic seed) and Page 2 has PIS action data uploaded before running."
    )

results_df = state.get("results")

if results_df is None or results_df.empty:
    st.info("Run the pipeline to generate scenario results.")
    st.stop()

st.markdown("### Overall summary")
summary = (
    results_df.groupby("name")[
        [
            "num_inspections",
            "intercepted",
            "false_neg",
            "missing",
            "total_missed_contaminants",
            "total_intercepted_contaminants",
        ]
    ]
    .sum()
    .reset_index()
)
summary["detection_rate"] = summary["total_intercepted_contaminants"] / (
    summary["total_intercepted_contaminants"] + summary["total_missed_contaminants"]
)
summary["slippage_rate"] = 1 - summary["detection_rate"]
summary = summary.fillna(0)

kpi_cols = st.columns(5)
kpi_cols[0].metric("Scenarios", len(summary))
kpi_cols[1].metric("Total inspections", f"{int(summary['num_inspections'].sum()):,}")
kpi_cols[2].metric(
    "Interceptions",
    f"{int(summary['total_intercepted_contaminants'].sum()):,}",
)
kpi_cols[3].metric("False negatives", f"{int(summary['false_neg'].sum()):,}")
overall_slippage = (
    summary["total_missed_contaminants"].sum()
    / max(1, summary["total_intercepted_contaminants"].sum() + summary["total_missed_contaminants"].sum())
)
kpi_cols[4].metric("Overall slippage rate", f"{100 * overall_slippage:.1f}%")
st.markdown("### Slippage and detection by scenario")

st.bar_chart(
    summary[["name", "total_intercepted_contaminants", "total_missed_contaminants"]].set_index("name"),
    use_container_width=True,
)

rate_chart = (
    alt.Chart(summary)
    .transform_calculate(slippage_pct="1 - datum.detection_rate")
    .mark_bar()
    .encode(
        x=alt.X("name:N", title="Scenario"),
        y=alt.Y("slippage_pct:Q", title="Slippage rate"),
        tooltip=[
            alt.Tooltip("detection_rate:Q", format=".2%", title="Detection rate"),
            alt.Tooltip("slippage_pct:Q", format=".2%", title="Slippage rate"),
            alt.Tooltip("num_inspections:Q", title="Inspections"),
        ],
        color=alt.Color("slippage_pct:Q", scale=alt.Scale(scheme="reds")),
    )
)
st.altair_chart(rate_chart, use_container_width=True)

st.markdown("### Slippage vs inspected units")
scatter_source = summary[["name", "num_inspections", "slippage_rate"]].rename(
    columns={"num_inspections": "Inspected Units", "slippage_rate": "Slippage Rate"}
)
st.scatter_chart(scatter_source, x="Inspected Units", y="Slippage Rate", size=None, color="name")

st.markdown("### Scenario Results Output")
st.dataframe(results_df, use_container_width=True)
st.download_button(
    "Download scenario results (CSV)",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name="pis_contamination_scenario_results.csv",
    use_container_width=True,
)

st.markdown("### Raw configuration output")
# config_cols = [
#     "contamination/contamination_unit",
#     "contamination/contamination_rate/distribution",
#     "contamination/arrangement",
#     "inspection/sample_strategy",
#     "inspection/proportion/value",
# ]
# existing_cols = [col for col in config_cols if col in scenario_df.columns]
# if existing_cols:
#     st.dataframe(scenario_df[existing_cols], use_container_width=True)

st.divider()
nav_cols = st.columns(2)
with nav_cols[0]:
    if st.button("Back to Page 4", type="primary", key="nav_back_page5"):
        st.switch_page("pages/4_Scenario_Experiments.py")
with nav_cols[1]:
    if st.button("Finish and Return Home", type="primary", key="nav_finish"):
        st.switch_page("frontend.py")

