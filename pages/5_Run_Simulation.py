# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

from pathlib import Path

from gui.runtime_warnings import suppress_optional_dependency_warnings

suppress_optional_dependency_warnings()

import pandas as pd
import altair as alt
import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.page_styles import apply_shared_page_styles, render_page_intro
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
apply_shared_page_styles()
engine_options = state["engine_options"]
run_error = state.get("run_error")
paths = state["paths"]
TMP_DIR = Path("tmp")


st.warning(
    "**Test Deployment Notice: This is a test deployment with limited functionality and is under active development. "
    "Features may be incomplete and subject to change. Results have not been validated.**"
)
st.title("Page 5 - Run Simulation")
render_page_intro("Execute the slippage pipeline and compare policies based on slippage metrics.")

with st.sidebar:
    st.subheader("Execution options")
    simulations = st.number_input(
        "Simulation replications",
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
summary = results_df.copy()

kpi_cols = st.columns(3)
kpi_cols[0].metric("Scenarios", len(summary))
kpi_cols[1].metric("Total inspections", f"{int(summary['num_inspections'].sum()):,}")
slipped_total = int(summary["total_slipped_units"].sum()) if "total_slipped_units" in summary.columns else 0
kpi_cols[2].metric("Slipped plant units", f"{slipped_total:,}")

# Slippage across scenarios (contaminated plant units that slipped)
if "total_slipped_units" in results_df.columns and "name" in results_df.columns:
    st.markdown("### Slippage by scenario (contaminated plant units missed)")
    slip_df = results_df[["name", "total_slipped_units"]]
    slip_chart = (
        alt.Chart(slip_df)
        .mark_bar()
        .encode(
            y=alt.Y("name:N", title="Scenario"),
            x=alt.X("total_slipped_units:Q", title="Slipped plant units"),
            tooltip=["name", "total_slipped_units"],
        )
    )
    st.altair_chart(slip_chart, use_container_width=True)

# Inspected quantities by level
if all(
    col in results_df.columns
    for col in [
        "name",
        "avg_plant_units_inspected_completion",
        "avg_sample_units_inspected_completion",
        "avg_inspection_units_opened_completion",
    ]
):
    st.markdown("### Inspected quantities (completion) by scenario")
    inspected_df = results_df[
        [
            "name",
            "avg_plant_units_inspected_completion",
            "avg_sample_units_inspected_completion",
            "avg_inspection_units_opened_completion",
        ]
    ]
    inspected_long = inspected_df.rename(
        columns={
            "avg_plant_units_inspected_completion": "Plants inspected (avg)",
            "avg_sample_units_inspected_completion": "Sample units inspected (avg)",
            "avg_inspection_units_opened_completion": "Inspection units opened (avg)",
        }
    ).melt(id_vars="name", var_name="level", value_name="count")

    st.dataframe(
        inspected_long.pivot(index="name", columns="level", values="count"),
        use_container_width=True,
    )

# Contamination totals by level (plant, sample, inspection)
required_cols = [
    "name",
    "total_contaminated_units",
    "total_contaminated_sample_units",
    "total_contaminated_inspection_units",
    "num_plants",
    "num_sample_units",
    "num_inspection_units",
]
if all(col in results_df.columns for col in required_cols):
    st.markdown("### Contamination totals by level")

    def level_chart(level_label, contam_col, total_col):
        data = results_df[["name", contam_col, total_col]].copy()
        data = data.rename(columns={contam_col: "contaminated", total_col: "total"})
        data["not_contaminated"] = data["total"] - data["contaminated"]
        data = data.melt(id_vars="name", value_vars=["contaminated", "not_contaminated"], var_name="metric", value_name="value")
        return (
            alt.Chart(data)
            .mark_bar()
            .encode(
                y=alt.Y("name:N", title="Scenario"),
                x=alt.X("value:Q", title="Count"),
                color=alt.Color("metric:N", title="Metric"),
                tooltip=["name", "metric", "value"],
            )
            .properties(title=level_label)
        )

    st.altair_chart(
        level_chart("Plant units", "total_contaminated_units", "num_plants"),
        use_container_width=True,
    )
    st.altair_chart(
        level_chart(
            "Sample units", "total_contaminated_sample_units", "num_sample_units"
        ),
        use_container_width=True,
    )
    st.altair_chart(
        level_chart(
            "Inspection units",
            "total_contaminated_inspection_units",
            "num_inspection_units",
        ),
        use_container_width=True,
    )
else:
    st.info("Contamination totals by level are unavailable in the current results.")

st.markdown("### Raw Output")
st.dataframe(state["results"], use_container_width=True)

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
