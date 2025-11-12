from pathlib import Path

import streamlit as st

from lib.models import init_state
from lib.slippage_pipeline import SyntheticOptions, load_scenario_dataframe
from lib.slippage_ui import (
    get_slippage_state,
    init_slippage_state,
    run_pipeline,
    set_paths,
    set_scenario_dataframe,
    set_synthetic_options,
)


SAMPLING_METHODS = ["naive", "sequential", "gmm", "gaussian_copula"]


st.set_page_config(page_title="PoPS Border - Pipeline", page_icon="🧪", layout="wide")
init_state()
state = init_slippage_state()
paths = state["paths"]
synthetic_options = state["synthetic_options"]


with st.sidebar:
    st.subheader("Slippage pipeline")
    samples = st.number_input(
        "Synthetic samples",
        min_value=10,
        max_value=50000,
        step=10,
        value=int(synthetic_options.n_samples),
    )
    method = st.selectbox(
        "Sampling method",
        SAMPLING_METHODS,
        index=SAMPLING_METHODS.index(synthetic_options.sampling_method),
    )
    if samples != synthetic_options.n_samples or method != synthetic_options.sampling_method:
        set_synthetic_options(SyntheticOptions(n_samples=int(samples), sampling_method=method))

    st.divider()
    run_clicked = st.button("Run contamination pipeline", use_container_width=True)
    if run_clicked:
        with st.spinner("Executing slippage pipeline..."):
            try:
                run_pipeline()
                st.success("Pipeline completed.")
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Pipeline failed: {exc}")

st.title("PoPS Border Inspection Simulation")
st.caption("Interact with configuration, synthetic data generation, and scenario analysis.")


# Quick summary from session data classes
s = st.session_state.scenario
i = st.session_state.inspection
col = st.columns(6)
col[0].metric("Units", f"{s.units_in_shipment:,}")
col[1].metric("Items / unit", f"{s.items_per_unit:,}")
col[2].metric("Prevalence (%)", f"{s.contamination_prevalence_pct:.2f}")
col[3].metric("Arrangement", s.contamination_arrangement)
col[4].metric("Method", i.method)
col[5].metric("Sample units", f"{i.sample_units:,}")

status_cols = st.columns(3)
status_cols[0].metric("Scenario rows", len(state["scenario_df"]))
results_df = state.get("results")
status_cols[1].metric("Last pipeline run", value=state.get("last_run"))
status_cols[2].metric(
    "Result scenarios",
    value=0 if results_df is None else len(results_df),
)
if state.get("run_error"):
    st.error(f"Latest pipeline error: {state['run_error']}")


tabs = st.tabs(
    [
        "Overview",
        "Data inputs",
        "Synthetic data",
        "Contamination fit",
        "Results",
    ]
)


with tabs[0]:
    st.markdown("### Pipeline snapshot")
    st.json(
        {
            "config": str(paths.config),
            "scenario_table": str(paths.scenario_table),
            "compliance_lookup": str(paths.compliance_lookup),
            "synthetic_seed": str(paths.synthetic_seed),
            "synthetic_output": str(paths.synthetic_output),
            "pis_data": str(paths.pis_data),
            "rbs_data": str(paths.rbs_data),
            "last_run": state.get("last_run"),
            "error": state.get("run_error"),
        }
    )
    st.markdown("### Scenario table preview")
    st.dataframe(state["scenario_df"].head(20), use_container_width=True)


with tabs[1]:
    st.subheader("File locations")
    col1, col2 = st.columns(2)
    config_path = col1.text_input("Config YAML", str(paths.config))
    scenario_path = col1.text_input("Scenario CSV", str(paths.scenario_table))
    compliance_path = col1.text_input("Compliance lookup CSV", str(paths.compliance_lookup))
    pis_path = col2.text_input("PIS data CSV", str(paths.pis_data))
    rbs_path = col2.text_input("RBS calculator CSV", str(paths.rbs_data))
    synthetic_seed_path = col2.text_input("Synthetic seed CSV", str(paths.synthetic_seed))
    synthetic_output_path = col2.text_input("Synthetic output CSV", str(paths.synthetic_output))

    if st.button("Update paths", key="update_paths"):
        try:
            set_paths(
                config=Path(config_path),
                scenario_table=Path(scenario_path),
                compliance_lookup=Path(compliance_path),
                pis_data=Path(pis_path),
                rbs_data=Path(rbs_path),
                synthetic_seed=Path(synthetic_seed_path),
                synthetic_output=Path(synthetic_output_path),
            )
            st.success("Paths updated. Scenario table reloaded.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unable to update paths: {exc}")

    if st.button("Reload scenario from disk"):
        try:
            refreshed_df = load_scenario_dataframe(Path(scenario_path), dtype="object")
            set_scenario_dataframe(refreshed_df)
            st.success("Scenario table reloaded.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed to reload scenario table: {exc}")

    st.markdown("#### Scenario data editor")
    edited_scenarios = st.data_editor(
        state["scenario_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="scenario_editor",
    )
    if st.button("Save scenario edits"):
        set_scenario_dataframe(edited_scenarios)
        st.success("Scenario table updated in session.")

    st.download_button(
        "Download scenario table (CSV)",
        data=edited_scenarios.to_csv(index=False).encode("utf-8"),
        file_name="scenario_table.csv",
        use_container_width=True,
    )


with tabs[2]:
    st.subheader("Synthetic consignment data")
    preview = state.get("synthetic_preview")
    if preview is None:
        st.info("Run the pipeline to generate synthetic consignment data.")
    else:
        st.dataframe(preview, use_container_width=True)
        st.download_button(
            "Download synthetic dataset (CSV)",
            data=state["synthetic_data"].to_csv(index=False).encode("utf-8"),
            file_name="synthetic_data.csv",
            use_container_width=True,
        )


with tabs[3]:
    st.subheader("Clarke beta-binomial fit")
    fit = state.get("fit")
    if fit is None:
        st.info("Run the pipeline to fit contamination parameters.")
    else:
        st.metric("Alpha", f"{fit.alpha:.6f}")
        st.metric("Beta", f"{fit.beta:.6f}")
        st.metric("Theta", f"{fit.theta:.6f}")
        st.markdown("#### PIS data sample")
        st.dataframe(state["pis_preview"], use_container_width=True)
        st.markdown("#### RBS data sample")
        st.dataframe(state["rbs_preview"], use_container_width=True)
        st.download_button(
            "Download fitted PIS data (CSV)",
            data=state["pis_data"].to_csv(index=False).encode("utf-8"),
            file_name="pis_data.csv",
            use_container_width=True,
        )
        st.download_button(
            "Download RBS calculator data (CSV)",
            data=state["rbs_data"].to_csv(index=False).encode("utf-8"),
            file_name="rbs_data.csv",
            use_container_width=True,
        )


with tabs[4]:
    st.subheader("Scenario results")
    if results_df is None:
        st.info("Run the pipeline to view scenario outcomes.")
    else:
        st.dataframe(results_df, use_container_width=True)
        st.download_button(
            "Download scenario results (CSV)",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="pis_contamination_scenario_results.csv",
            use_container_width=True,
        )

