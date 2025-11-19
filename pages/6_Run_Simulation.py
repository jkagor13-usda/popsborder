import pandas as pd
import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.slippage_ui import get_slippage_state, run_pipeline, set_engine_options


st.set_page_config(
    page_title="Run Simulation",
    page_icon=":arrow_forward:",
    layout="wide",
)
init_state()

state = get_slippage_state()
render_sidebar_navigation()
engine_options = state["engine_options"]

st.title("Page 6 - Run Simulation")
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

st.bar_chart(
    summary[["name", "total_intercepted_contaminants", "total_missed_contaminants"]].set_index("name"),
    use_container_width=True,
)

st.markdown("### Policy comparison")
scenario_df = state["scenario_df"]
scenario_meta_cols = ["inspection/sample_strategy", "inspection/compliance_level"]
if not scenario_df.empty and "name" in scenario_df.columns:
    available_meta_cols = ["name"] + [col for col in scenario_meta_cols if col in scenario_df.columns]
    scenario_meta = scenario_df[available_meta_cols].drop_duplicates("name")
else:
    scenario_meta = pd.DataFrame(columns=["name"])
enriched = results_df.merge(scenario_meta, on="name", how="left")

if "inspection/sample_strategy" in enriched.columns and enriched["inspection/sample_strategy"].notna().any():
    strategy_chart = (
        enriched.groupby("inspection/sample_strategy")
        .agg(
            inspections=("num_inspections", "sum"),
            missed=("total_missed_contaminants", "sum"),
            intercepted=("total_intercepted_contaminants", "sum"),
        )
        .reset_index()
    )
    strategy_chart["detection_rate"] = strategy_chart["intercepted"] / (
        strategy_chart["intercepted"] + strategy_chart["missed"]
    )
    strategy_chart = strategy_chart.fillna(0)
    st.write("Detection rate by inspection policy")
    st.bar_chart(strategy_chart.set_index("inspection/sample_strategy")[["detection_rate"]], use_container_width=True)
else:
    st.info("Add inspection sample strategies on Page 3 to compare policies.")

if "inspection/compliance_level" in enriched.columns and enriched["inspection/compliance_level"].notna().any():
    compliance_chart = (
        enriched.groupby("inspection/compliance_level")
        .agg(
            slippage=("total_missed_contaminants", "sum"),
            inspections=("num_inspections", "sum"),
        )
        .reset_index()
    )
    compliance_chart = compliance_chart.fillna(0)
    st.write("Total slippage by compliance type")
    st.bar_chart(compliance_chart.set_index("inspection/compliance_level")[["slippage"]], use_container_width=True)
else:
    st.info("Assign compliance levels on Page 3 to view this comparison.")

st.markdown("### Slippage vs inspected units")
scatter_source = summary[["name", "num_inspections", "slippage_rate"]].rename(
    columns={"num_inspections": "Inspected Units", "slippage_rate": "Slippage Rate"}
)
st.scatter_chart(scatter_source, x="Inspected Units", y="Slippage Rate", size=None, color="name")

st.markdown("### Detailed results")
st.dataframe(results_df, use_container_width=True)
st.download_button(
    "Download scenario results (CSV)",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name="pis_contamination_scenario_results.csv",
    use_container_width=True,
)

st.markdown("### Raw configuration columns")
config_cols = [
    "contamination/contamination_unit",
    "contamination/contamination_rate/distribution",
    "contamination/arrangement",
    "inspection/sample_strategy",
    "inspection/proportion/value",
]
existing_cols = [col for col in config_cols if col in scenario_df.columns]
if existing_cols:
    st.dataframe(scenario_df[existing_cols], use_container_width=True)

st.divider()
nav_cols = st.columns(2)
with nav_cols[0]:
    st.page_link("pages/5_Scenario_Experiments.py", label="⬅️ Back to Page 5")
with nav_cols[1]:
    st.page_link("frontend.py", label="Finish & Return Home", icon="🏁")
