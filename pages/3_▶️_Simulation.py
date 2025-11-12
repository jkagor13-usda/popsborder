import pandas as pd
import streamlit as st

from lib.models import init_state
from lib.slippage_ui import (
    get_slippage_state,
    run_pipeline,
    set_engine_options,
)

st.set_page_config(page_title="Simulation", page_icon="▶️", layout="wide")
init_state()

state = get_slippage_state()
engine_options = state["engine_options"]

st.title("▶️ Run Slippage Simulation")
st.caption("Execute the PoPS Border slippage pipeline and explore scenario outcomes.")

with st.sidebar:
    st.subheader("Execution options")
    seed = st.number_input("Random seed", value=int(engine_options.get("seed", 42)), step=1)
    consignments = st.number_input(
        "Consignments per scenario",
        min_value=1,
        max_value=1000,
        value=int(engine_options.get("num_consignments", 5)),
        step=1,
    )
    simulations = st.number_input(
        "Simulation repetitions",
        min_value=1,
        max_value=100,
        value=int(engine_options.get("num_simulations", 1)),
        step=1,
    )
    if (
        seed != engine_options.get("seed")
        or consignments != engine_options.get("num_consignments")
        or simulations != engine_options.get("num_simulations")
    ):
        set_engine_options(seed=int(seed), num_consignments=int(consignments), num_simulations=int(simulations))

    st.divider()
    run_now = st.button("Run pipeline", use_container_width=True)
    if run_now:
        with st.spinner("Running slippage pipeline..."):
            try:
                run_pipeline()
                st.success("Pipeline finished.")
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Pipeline failed: {exc}")

results_df = state.get("results")

if results_df is None or results_df.empty:
    st.info("Run the pipeline to generate scenario results.")
    st.stop()

st.markdown("### Scenario summary")
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
summary = summary.fillna(0)

kpi_cols = st.columns(4)
kpi_cols[0].metric("Scenarios", len(summary))
kpi_cols[1].metric("Total inspections", f"{int(summary['num_inspections'].sum()):,}")
kpi_cols[2].metric(
    "Interceptions",
    f"{int(summary['total_intercepted_contaminants'].sum()):,}",
)
overall_detection = summary["total_intercepted_contaminants"].sum() / max(
    1, summary["total_intercepted_contaminants"].sum() + summary["total_missed_contaminants"].sum()
)
kpi_cols[3].metric("Overall detection rate", f"{100 * overall_detection:.1f}%")

chart_data = summary[["name", "total_intercepted_contaminants", "total_missed_contaminants"]].set_index("name")
st.bar_chart(chart_data, use_container_width=True)

st.markdown("### Detailed results")
st.dataframe(results_df, use_container_width=True)
st.download_button(
    "Download scenario results (CSV)",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name="pis_contamination_scenario_results.csv",
    use_container_width=True,
)

st.markdown("### Detection rate by scenario")
rate_chart_data = summary[["name", "detection_rate"]].set_index("name")
st.bar_chart(rate_chart_data, use_container_width=True)

st.markdown("### Raw configuration columns")
config_cols = [
    "contamination/contamination_unit",
    "contamination/contamination_rate/distribution",
    "contamination/arrangement",
    "inspection/sample_strategy",
    "inspection/proportion/value",
]
existing_cols = [col for col in config_cols if col in state["scenario_df"].columns]
if existing_cols:
    st.dataframe(state["scenario_df"][existing_cols], use_container_width=True)
