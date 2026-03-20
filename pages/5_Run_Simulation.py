# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

from pathlib import Path
from typing import Optional

from gui.runtime_warnings import suppress_optional_dependency_warnings

suppress_optional_dependency_warnings()

import pandas as pd
import altair as alt
import numpy as np
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
        "Total number of consignments inspected across the displayed scenario results.",
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
            "Shows the mean number of inspection units opened, sample units inspected, "
            "and plant units inspected for each scenario. When available, 95% intervals "
            "are computed across replications within the same scenario."
        ),
    )
    inspected_df = results_df[
        [
            "name",
            "avg_plant_units_inspected_completion",
            "avg_sample_units_inspected_completion",
            "avg_inspection_units_opened_completion",
        ]
    ].rename(
        columns={
            "avg_plant_units_inspected_completion": "Plants inspected",
            "avg_sample_units_inspected_completion": "Sample units inspected",
            "avg_inspection_units_opened_completion": "Inspection units opened",
        }
    )

    if all_runs_df is not None and {
        "name",
        "avg_plant_units_inspected_completion",
        "avg_sample_units_inspected_completion",
        "avg_inspection_units_opened_completion",
    }.issubset(all_runs_df.columns):
        inspected_intervals = (
            all_runs_df[
                [
                    "name",
                    "avg_plant_units_inspected_completion",
                    "avg_sample_units_inspected_completion",
                    "avg_inspection_units_opened_completion",
                ]
            ]
            .dropna()
            .groupby("name")
            .agg(
                plant_lower=("avg_plant_units_inspected_completion", lambda s: s.quantile(0.025)),
                plant_upper=("avg_plant_units_inspected_completion", lambda s: s.quantile(0.975)),
                sample_lower=("avg_sample_units_inspected_completion", lambda s: s.quantile(0.025)),
                sample_upper=("avg_sample_units_inspected_completion", lambda s: s.quantile(0.975)),
                inspection_lower=("avg_inspection_units_opened_completion", lambda s: s.quantile(0.025)),
                inspection_upper=("avg_inspection_units_opened_completion", lambda s: s.quantile(0.975)),
                replications=("avg_plant_units_inspected_completion", "size"),
            )
            .reset_index()
        )
        inspected_df = inspected_df.merge(inspected_intervals, on="name", how="left")
        inspected_df["Plants 95% interval"] = inspected_df.apply(
            lambda row: f"{row['plant_lower']:.1f} - {row['plant_upper']:.1f}"
            if pd.notna(row.get("plant_lower")) and row.get("replications", 0) > 1
            else "n/a",
            axis=1,
        )
        inspected_df["Sample units 95% interval"] = inspected_df.apply(
            lambda row: f"{row['sample_lower']:.1f} - {row['sample_upper']:.1f}"
            if pd.notna(row.get("sample_lower")) and row.get("replications", 0) > 1
            else "n/a",
            axis=1,
        )
        inspected_df["Inspection units 95% interval"] = inspected_df.apply(
            lambda row: f"{row['inspection_lower']:.1f} - {row['inspection_upper']:.1f}"
            if pd.notna(row.get("inspection_lower")) and row.get("replications", 0) > 1
            else "n/a",
            axis=1,
        )

    display_cols = [
        "name",
        "Plants inspected",
        "Plants 95% interval",
        "Sample units inspected",
        "Sample units 95% interval",
        "Inspection units opened",
        "Inspection units 95% interval",
    ]
    available_cols = [col for col in display_cols if col in inspected_df.columns]
    display_df = inspected_df[available_cols].rename(columns={"name": "Scenario"}).copy()
    numeric_cols = [
        col for col in ["Plants inspected", "Sample units inspected", "Inspection units opened"]
        if col in display_df.columns
    ]
    for col in numeric_cols:
        display_df[col] = display_df[col].map(lambda value: f"{value:,.1f}")
    styled_inspected = (
        display_df.style
        .set_properties(subset=["Scenario"], **{"font-weight": "600", "color": "#1f3b63"})
        .set_properties(
            subset=[col for col in ["Plants inspected", "Sample units inspected", "Inspection units opened"] if col in display_df.columns],
            **{"background-color": "#eaf3fb", "font-weight": "600", "color": "#16324f"},
        )
        .set_properties(
            subset=[col for col in ["Plants 95% interval", "Sample units 95% interval", "Inspection units 95% interval"] if col in display_df.columns],
            **{"background-color": "#f5f9fd", "color": "#355070"},
        )
    )
    st.dataframe(styled_inspected, use_container_width=True, height=min(420, 70 + 38 * max(len(display_df), 1)))

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
            ("not_contaminated", "Not contaminated", "#9ecae1"),
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
        level_chart("PLANT UNITS", "total_contaminated_units", "num_plants"),
        use_container_width=True,
    )
    st.altair_chart(
        level_chart(
            "SAMPLE UNITS", "total_contaminated_sample_units", "num_sample_units"
        ),
        use_container_width=True,
    )
    st.altair_chart(
        level_chart(
            "INSPECTION UNITS", "total_contaminated_inspection_units", "num_inspection_units",
        ),
        use_container_width=True,
    )
    if all_runs_df is not None and not all_runs_df.empty and "replication" in all_runs_df.columns:
        st.caption("Bars show scenario means. Error bars show 95% intervals across replications within the same scenario when multiple simulation replications are available.")

    pct_df = results_df[
        [
            "name",
            "total_contaminated_units",
            "total_contaminated_sample_units",
            "total_contaminated_inspection_units",
            "num_plants",
            "num_sample_units",
            "num_inspection_units",
        ]
    ].copy()
    pct_df["Plant contamination %"] = np.where(
        pct_df["num_plants"] > 0,
        (pct_df["total_contaminated_units"] / pct_df["num_plants"]) * 100.0,
        np.nan,
    )
    pct_df["Sample unit contamination %"] = np.where(
        pct_df["num_sample_units"] > 0,
        (pct_df["total_contaminated_sample_units"] / pct_df["num_sample_units"]) * 100.0,
        np.nan,
    )
    pct_df["Inspection unit contamination %"] = np.where(
        pct_df["num_inspection_units"] > 0,
        (pct_df["total_contaminated_inspection_units"] / pct_df["num_inspection_units"]) * 100.0,
        np.nan,
    )

    if all_runs_df is not None and {
        "name",
        "total_contaminated_units",
        "total_contaminated_sample_units",
        "total_contaminated_inspection_units",
        "num_plants",
        "num_sample_units",
        "num_inspection_units",
    }.issubset(all_runs_df.columns):
        pct_runs = all_runs_df[
            [
                "name",
                "total_contaminated_units",
                "total_contaminated_sample_units",
                "total_contaminated_inspection_units",
                "num_plants",
                "num_sample_units",
                "num_inspection_units",
            ]
        ].copy()
        pct_runs["plant_pct"] = np.where(
            pct_runs["num_plants"] > 0,
            (pct_runs["total_contaminated_units"] / pct_runs["num_plants"]) * 100.0,
            np.nan,
        )
        pct_runs["sample_pct"] = np.where(
            pct_runs["num_sample_units"] > 0,
            (pct_runs["total_contaminated_sample_units"] / pct_runs["num_sample_units"]) * 100.0,
            np.nan,
        )
        pct_runs["inspection_pct"] = np.where(
            pct_runs["num_inspection_units"] > 0,
            (pct_runs["total_contaminated_inspection_units"] / pct_runs["num_inspection_units"]) * 100.0,
            np.nan,
        )
        pct_intervals = (
            pct_runs.groupby("name")
            .agg(
                plant_lower=("plant_pct", lambda s: s.quantile(0.025)),
                plant_upper=("plant_pct", lambda s: s.quantile(0.975)),
                sample_lower=("sample_pct", lambda s: s.quantile(0.025)),
                sample_upper=("sample_pct", lambda s: s.quantile(0.975)),
                inspection_lower=("inspection_pct", lambda s: s.quantile(0.025)),
                inspection_upper=("inspection_pct", lambda s: s.quantile(0.975)),
                replications=("plant_pct", "size"),
            )
            .reset_index()
        )
        pct_df = pct_df.merge(pct_intervals, on="name", how="left")
        pct_df["Plant 95% interval"] = pct_df.apply(
            lambda row: f"{row['plant_lower']:.2f}% - {row['plant_upper']:.2f}%"
            if pd.notna(row.get("plant_lower")) and row.get("replications", 0) > 1
            else "n/a",
            axis=1,
        )
        pct_df["Sample unit 95% interval"] = pct_df.apply(
            lambda row: f"{row['sample_lower']:.2f}% - {row['sample_upper']:.2f}%"
            if pd.notna(row.get("sample_lower")) and row.get("replications", 0) > 1
            else "n/a",
            axis=1,
        )
        pct_df["Inspection unit 95% interval"] = pct_df.apply(
            lambda row: f"{row['inspection_lower']:.2f}% - {row['inspection_upper']:.2f}%"
            if pd.notna(row.get("inspection_lower")) and row.get("replications", 0) > 1
            else "n/a",
            axis=1,
        )

    render_labeled_help(
        "Contamination percentages by level",
        "Shows the mean percentage contaminated at the plant, sample unit, and inspection unit levels for each scenario. When available, 95% intervals are computed across replications within the same scenario.",
    )
    pct_display_cols = [
        "name",
        "Plant contamination %",
        "Plant 95% interval",
        "Sample unit contamination %",
        "Sample unit 95% interval",
        "Inspection unit contamination %",
        "Inspection unit 95% interval",
    ]
    pct_available_cols = [col for col in pct_display_cols if col in pct_df.columns]
    pct_display_df = pct_df[pct_available_cols].rename(columns={"name": "Scenario"}).copy()
    pct_numeric_cols = [
        col
        for col in [
            "Plant contamination %",
            "Sample unit contamination %",
            "Inspection unit contamination %",
        ]
        if col in pct_display_df.columns
    ]
    for col in pct_numeric_cols:
        pct_display_df[col] = pct_display_df[col].map(lambda value: f"{value:.2f}%")
    styled_pct = (
        pct_display_df.style
        .set_properties(subset=["Scenario"], **{"font-weight": "600", "color": "#1f3b63"})
        .set_properties(
            subset=[col for col in pct_numeric_cols if col in pct_display_df.columns],
            **{"background-color": "#eaf3fb", "font-weight": "600", "color": "#16324f"},
        )
        .set_properties(
            subset=[
                col
                for col in [
                    "Plant 95% interval",
                    "Sample unit 95% interval",
                    "Inspection unit 95% interval",
                ]
                if col in pct_display_df.columns
            ],
            **{"background-color": "#f5f9fd", "color": "#355070"},
        )
    )
    st.dataframe(styled_pct, use_container_width=True, height=min(420, 70 + 38 * max(len(pct_display_df), 1)))
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
