# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

from pathlib import Path
from typing import Optional

from gui.runtime_warnings import suppress_optional_dependency_warnings

suppress_optional_dependency_warnings()

import pandas as pd
import altair as alt
import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.page_styles import (
    apply_shared_page_styles,
    render_labeled_help,
    render_metric_card,
    render_page_intro,
    render_section_header,
)
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


def _latest_output_run_dir(experiment_dir: Optional[Path]) -> Optional[Path]:
    if experiment_dir is None or not experiment_dir.exists():
        return None
    candidates = [p for p in experiment_dir.glob("output_*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_all_runs_df(output_dir: Optional[Path]) -> Optional[pd.DataFrame]:
    if output_dir is None:
        return None
    all_runs_path = output_dir / "all_runs.csv"
    if not all_runs_path.exists():
        return None
    try:
        return pd.read_csv(all_runs_path)
    except Exception:  # pylint: disable=broad-except
        return None


st.warning(
    "**Test Deployment Notice: This is a test deployment with limited functionality and is under active development. "
    "Features may be incomplete and subject to change. Results have not been validated.**"
)
st.title("Page 5 - Run Simulation")
render_page_intro("Execute the slippage pipeline and compare policies based on slippage metrics.")

selected_experiment = None
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

saved_output_files = state.get("run_output_files") or []
selected_experiment_dir = selected_experiment.parent if selected_experiment else None
latest_output_dir = _latest_output_run_dir(selected_experiment_dir)
all_runs_df = _load_all_runs_df(latest_output_dir)
saved_output_dir = state.get("run_output_dir")
if latest_output_dir is not None:
    saved_output_dir = str(latest_output_dir)
    latest_output_files = sorted(p for p in latest_output_dir.glob("*.csv") if p.is_file())
    if latest_output_files:
        saved_output_files = [str(path) for path in latest_output_files]
if saved_output_dir:
    output_dir_path = Path(saved_output_dir)
    output_file_names = ", ".join(Path(path).name for path in saved_output_files) if saved_output_files else "none"
    st.caption(
        f"Latest output: `{output_dir_path.name}` in `tmp/experiments/{output_dir_path.parent.parent.name}`. "
        f"Files: {output_file_names}."
    )

render_labeled_help(
    "Overall summary",
    "Summarizes how many scenarios were run, the total inspections performed, and the mean slipped plant units aggregated across the scenario results.",
)
summary = results_df.copy()

kpi_cols = st.columns(3)
slipped_total = int(summary["total_slipped_units"].sum()) if "total_slipped_units" in summary.columns else 0
with kpi_cols[0]:
    render_metric_card("Scenarios", f"{len(summary):,}", "Number of scenarios included in the current results.")
with kpi_cols[1]:
    render_metric_card(
        "Total inspections",
        f"{int(summary['num_inspections'].sum()):,}",
        "Total number of inspections across the displayed scenario results.",
    )
with kpi_cols[2]:
    render_metric_card(
        "Slipped plant units",
        f"{slipped_total:,}",
        "Mean contaminated plant units missed, summed across the displayed scenario results.",
    )

# Slippage across scenarios (contaminated plant units that slipped)
if "total_slipped_units" in results_df.columns and "name" in results_df.columns:
    render_labeled_help(
        "Slippage by scenario",
        "Bars show the mean number of contaminated plant units missed for each scenario across simulation replications. When available, error bars show the 95% interval across replications within that same scenario.",
    )
    slip_df = results_df[["name", "total_slipped_units"]]
    slip_chart = (
        alt.Chart(slip_df)
        .mark_bar(color="#1f77b4")
        .encode(
            y=alt.Y("name:N", title="Scenario"),
            x=alt.X("total_slipped_units:Q", title="Slipped plant units"),
            tooltip=[
                "name",
                alt.Tooltip("total_slipped_units:Q", title="Slipped plant units", format=".2f"),
            ],
        )
    )
    if all_runs_df is not None and {"name", "total_slipped_units"}.issubset(all_runs_df.columns):
        slip_interval_df = (
            all_runs_df[["name", "total_slipped_units"]]
            .dropna()
            .groupby("name")["total_slipped_units"]
            .agg(
                lower=lambda s: s.quantile(0.025),
                upper=lambda s: s.quantile(0.975),
                n="size",
            )
            .reset_index()
        )
        slip_interval_df = slip_interval_df[slip_interval_df["n"] > 1]
        if not slip_interval_df.empty:
            slip_error_bars = (
                alt.Chart(slip_interval_df)
                .mark_errorbar(color="#08306b", ticks=True)
                .encode(
                    y=alt.Y("name:N", title="Scenario"),
                    x=alt.X("lower:Q"),
                    x2=alt.X2("upper:Q"),
                    tooltip=[
                        "name",
                        alt.Tooltip("lower:Q", title="95% interval lower", format=".2f"),
                        alt.Tooltip("upper:Q", title="95% interval upper", format=".2f"),
                    ],
                )
            )
            slip_chart = slip_chart + slip_error_bars
    st.altair_chart(slip_chart, use_container_width=True)
    if all_runs_df is not None and not all_runs_df.empty and "replication" in all_runs_df.columns:
        st.caption("Bars show scenario means. Error bars show 95% intervals across replications within the same scenario when multiple simulation replications are available.")

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
    render_labeled_help(
        "Number of units inspected",
        (
            "Includes the number of inspection units opened, sample units inspected, "
            "and plant units inspected during the simulation."
        ),
    )
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
    render_labeled_help(
        "Contamination totals by level",
        "For each scenario, bars show the mean contaminated and not-contaminated counts across simulation replications at the plant, sample unit, and inspection unit levels. Error bars show 95% intervals across replications within the same scenario when available.",
    )

    def level_chart(level_label, contam_col, total_col):
        data = results_df[["name", contam_col, total_col]].copy()
        data = data.rename(columns={contam_col: "contaminated", total_col: "total"})
        data["not_contaminated"] = data["total"] - data["contaminated"]
        data = data.melt(
            id_vars="name",
            value_vars=["contaminated", "not_contaminated"],
            var_name="metric",
            value_name="value",
        )
        interval_df = None
        if all_runs_df is not None and contam_col in all_runs_df.columns and total_col in all_runs_df.columns:
            interval_source = all_runs_df[["name", contam_col, total_col]].dropna()
            if not interval_source.empty:
                interval_source = interval_source.rename(columns={contam_col: "contaminated", total_col: "total"})
                interval_source["not_contaminated"] = interval_source["total"] - interval_source["contaminated"]
                interval_source = interval_source.melt(
                    id_vars="name",
                    value_vars=["contaminated", "not_contaminated"],
                    var_name="metric",
                    value_name="value",
                )
                interval_df = (
                    interval_source.groupby(["name", "metric"])["value"]
                    .agg(
                        lower=lambda s: s.quantile(0.025),
                        upper=lambda s: s.quantile(0.975),
                        n="size",
                    )
                    .reset_index()
                )
                interval_df = interval_df[interval_df["n"] > 1]

        charts = []
        for metric_name, metric_title, metric_color in [
            ("contaminated", "Contaminated", "#1f77b4"),
            ("not_contaminated", "Not Contaminated", "#9ecae1"),
        ]:
            metric_data = data[data["metric"] == metric_name]
            base = alt.Chart(metric_data).encode(
                y=alt.Y("name:N", title="Scenario"),
                x=alt.X("value:Q", title="Count"),
                tooltip=[
                    "name",
                    alt.Tooltip("value:Q", title="Count", format=".2f"),
                ],
            )
            metric_chart = base.mark_bar(color=metric_color).properties(title=metric_title)
            if interval_df is not None and not interval_df.empty:
                metric_intervals = interval_df[interval_df["metric"] == metric_name]
                if not metric_intervals.empty:
                    error_bars = (
                        alt.Chart(metric_intervals)
                        .mark_errorbar(color="#08306b", ticks=True)
                        .encode(
                            y=alt.Y("name:N", title="Scenario"),
                            x=alt.X("lower:Q"),
                            x2=alt.X2("upper:Q"),
                            tooltip=[
                                "name",
                                alt.Tooltip("lower:Q", title="95% interval lower", format=".2f"),
                                alt.Tooltip("upper:Q", title="95% interval upper", format=".2f"),
                            ],
                        )
                    )
                    metric_chart = metric_chart + error_bars
            charts.append(metric_chart)
        return alt.hconcat(*charts).properties(title=level_label)

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
    if all_runs_df is not None and not all_runs_df.empty and "replication" in all_runs_df.columns:
        st.caption("Bars show scenario means. Error bars show 95% intervals across replications within the same scenario when multiple simulation replications are available.")
else:
    st.info("Contamination totals by level are unavailable in the current results.")

render_section_header("Raw output")
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
