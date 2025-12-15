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
TMP_DIR = Path("tmp")


st.title("Page 5 - Run Simulation")
st.caption("Execute the slippage pipeline and compare policies based on slippage metrics.")

with st.sidebar:
    st.subheader("Execution options")
    simulations = st.number_input(
        "Simulation repetitions",
        min_value=1,
        max_value=500,
        value=int(engine_options.get("num_simulations", 1)),
        step=1,
    )
    if simulations != engine_options.get("num_simulations"):
        set_engine_options(num_simulations=int(simulations))

    experiment_sets = sorted(Path("tmp/experiments").glob("*/scenario_table.csv"))
    if experiment_sets:
        labels = [p.parent.name for p in experiment_sets]
        selected_label = st.selectbox("Experiment to run", labels, index=0)
        selected_experiment = dict(zip(labels, experiment_sets)).get(selected_label)
        if selected_experiment:
            try:
                scenario_df = pd.read_csv(selected_experiment)
                state["scenario_df"] = scenario_df
                state["paths"] = state["paths"].__class__(**{**state["paths"].__dict__, "scenario_table": selected_experiment})
                st.caption(f"Loaded experiment: {selected_experiment}")
       
            except Exception as exc:  # pylint: disable=broad-except
                st.warning(f"Unable to load selected experiment: {exc}")
    else:
        st.info("No experiments saved yet on Page 4.")
    st.divider()
    run_disabled = not experiment_sets  # only disable when no experiments
    if st.button("Run pipeline", use_container_width=True, disabled=run_disabled):
        # Defer execution to main pane to show spinner there
        state["run_request_experiment"] = selected_experiment.parent if selected_experiment else None
        st.session_state["_trigger_run_pipeline"] = True

# Main-pane run handler with spinner
run_placeholder = st.empty()
if st.session_state.get("_trigger_run_pipeline"):
    st.session_state["_trigger_run_pipeline"] = False
    exp_dir = state.pop("run_request_experiment", None)
    with st.spinner("Running slippage pipeline..."):
        try:
            st.info(f"Running experiment at: {exp_dir}")
            results = run_pipeline(exp_dir)
            state["run_error"] = None
            state.pop("run_error_message", None)
            st.success("Pipeline finished.")
        except Exception as exc:  # pylint: disable=broad-except
            state["run_error"] = exc
            detail = getattr(exc, "stderr", None) or getattr(exc, "output", None)
            state["run_error_message"] = f"{exc}\n{detail}" if detail else str(exc)

if run_error:
    msg = state.get("run_error_message") or str(run_error)
    st.error("Pipeline failed")
    st.code(msg, language="text")
    st.caption(
        "Common fixes: confirm the selected experiment folder has a scenario_table.csv with non-empty inspection fields, "
        "RBS data in tmp/consignments, and PIS action data on Page 2. Check the full message above for the failing file."
    )
    # Surface current paths to help debugging
    paths_obj = state.get("paths")
    st.write(
        {
            "scenario_table": str(paths_obj.scenario_table) if paths_obj else None,
            "rbs_data": str(getattr(paths_obj, "rbs_data", None)) if paths_obj else None,
            "compliance_lookup": str(getattr(paths_obj, "compliance_lookup", None)) if paths_obj else None,
            "config": str(getattr(paths_obj, "config", None)) if paths_obj else None,
            "num_consignments": state.get("num_consignments"),
        }
    )

results_df = state.get("results")

if results_df is None or results_df.empty:
    st.info("Run the pipeline to generate scenario results.")
    st.stop()

# Persist results into the selected experiment folder for convenience
try:
    scenario_table = state["paths"].scenario_table if hasattr(state["paths"], "scenario_table") else None
    if scenario_table:
        exp_dir = Path(scenario_table).parent
        out_dir = exp_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "pis_contamination_scenario_results.csv"
        results_df.to_csv(out_path, index=False)
        st.caption(f"Results saved to {out_path}")
except Exception as exc:  # pylint: disable=broad-except
    st.warning(f"Could not save results to experiment folder: {exc}")

st.markdown("### Overall summary")
summary_cols = [
    "num_inspections",
    "intercepted",
    "false_neg",
    "missing",
    "total_missed_contaminants",
    "total_intercepted_contaminants",
]
summary = results_df.groupby("name")[summary_cols].sum().reset_index()
denom = summary["total_intercepted_contaminants"] + summary["total_missed_contaminants"]
summary["detection_rate"] = (
    summary["total_intercepted_contaminants"] / denom.replace(0, pd.NA)
).fillna(0)
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
kpi_row = st.columns(1)
kpi_row[0].metric("Missing contaminants", f"{int(summary['missing'].sum()):,}")
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

# Extended inspection/contamination statistics
stat_cols = [
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
    "pct_contaminant_unreported_if_detection",
    "true_contamination_rate",
]
existing_cols = [c for c in stat_cols if c in results_df.columns]
if existing_cols:
    st.markdown("### Inspection and contamination stats")
    st.dataframe(results_df[["name"] + existing_cols], use_container_width=True)
else:
    st.info("No detailed inspection/contamination stats available in results.")

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
nav_cols = st.columns(3)
with nav_cols[0]:
    if st.button("Reset and Return Home", type="secondary", key="nav_reset_page5"):
        try:
            if TMP_DIR.exists():
                import shutil  # pylint: disable=import-outside-toplevel
                shutil.rmtree(TMP_DIR)
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            st.session_state.clear()
            state["paths"] = create_default_paths()
            st.switch_page("frontend.py")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unable to reset temporary files: {exc}")
with nav_cols[1]:
    if st.button("Previous Page", type="primary", key="nav_back_page5"):
        st.switch_page("pages/4_Scenario_Experiments.py")
with nav_cols[2]:
    if st.button("Finish and Return Home", type="primary", key="nav_finish"):
        st.switch_page("frontend.py")
