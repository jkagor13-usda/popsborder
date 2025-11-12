import pandas as pd
import streamlit as st

from lib.models import init_state, set_scenario
from lib.slippage_pipeline import SyntheticOptions
from lib.slippage_ui import (
    get_slippage_state,
    set_scenario_dataframe,
    set_synthetic_options,
)

st.set_page_config(page_title="Scenario Builder", page_icon="📦", layout="wide")
init_state()

st.title("📦 Scenario Builder")
st.caption("Define shipment structure, synthetic data options, and slippage scenarios.")

state = get_slippage_state()
scenario_df = state["scenario_df"]
synthetic_opts: SyntheticOptions = state["synthetic_options"]

with st.form("scenario_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        title = st.text_input("Scenario name", value=st.session_state.scenario.title)
        seed = st.number_input("Random seed", value=st.session_state.scenario.seed, min_value=0, step=1)
        runs = st.slider("Simulation runs", 50, 5000, st.session_state.scenario.runs, step=50)
    with col2:
        unit_type = st.selectbox(
            "Consignment unit",
            ["Boxes", "Crates", "Pallets"],
            index=["Boxes", "Crates", "Pallets"].index(st.session_state.scenario.unit_type),
        )
        units = st.number_input(
            f"Number of {unit_type.lower()}",
            1,
            20000,
            st.session_state.scenario.units_in_shipment,
            step=50,
        )
        items_per_unit = st.number_input(
            "Items per unit",
            1,
            10000,
            st.session_state.scenario.items_per_unit,
            step=5,
        )
    with col3:
        packaging = st.selectbox(
            "Packaging type",
            ["Cardboard", "Wood", "Plastic", "Mixed"],
            index=["Cardboard", "Wood", "Plastic", "Mixed"].index(st.session_state.scenario.packaging_type),
        )
        prevalence = st.slider(
            "Contamination prevalence (%)",
            0.01,
            20.0,
            st.session_state.scenario.contamination_prevalence_pct,
            step=0.01,
        )
        arrangement = st.selectbox(
            "Arrangement pattern",
            ["Random", "Clustered", "Edge-loaded"],
            index=["Random", "Clustered", "Edge-loaded"].index(st.session_state.scenario.contamination_arrangement),
        )

    cv = st.slider(
        "Dispersion (CV)",
        0.1,
        3.0,
        st.session_state.scenario.contamination_cv,
        step=0.1,
        help="Higher CV → more variable contamination across units.",
    )
    notes = st.text_area("Notes", value=st.session_state.scenario.notes, height=80)

    submitted = st.form_submit_button("Save wireframe scenario", use_container_width=True)
    if submitted:
        set_scenario(
            title=title,
            seed=int(seed),
            runs=int(runs),
            unit_type=unit_type,
            units_in_shipment=int(units),
            items_per_unit=int(items_per_unit),
            packaging_type=packaging,
            contamination_prevalence_pct=float(prevalence),
            contamination_arrangement=arrangement,
            contamination_cv=float(cv),
            notes=notes,
        )
        st.success("Scenario saved.")

s = st.session_state.scenario
col = st.columns(4)
col[0].metric("Units", f"{s.units_in_shipment:,}")
col[1].metric("Items / unit", f"{s.items_per_unit:,}")
col[2].metric("Prevalence (%)", f"{s.contamination_prevalence_pct:.2f}")
col[3].metric("Arrangement", s.contamination_arrangement)

st.divider()
st.subheader("Synthetic data controls")
syn_col1, syn_col2 = st.columns(2)
samples = syn_col1.number_input(
    "Synthetic consignments",
    min_value=10,
    max_value=50000,
    step=10,
    value=int(synthetic_opts.n_samples),
)
method = syn_col2.selectbox(
    "Sampling method",
    options=["naive", "sequential", "gmm", "gaussian_copula"],
    index=["naive", "sequential", "gmm", "gaussian_copula"].index(synthetic_opts.sampling_method),
)
if samples != synthetic_opts.n_samples or method != synthetic_opts.sampling_method:
    set_synthetic_options(SyntheticOptions(n_samples=int(samples), sampling_method=method))
    st.info("Synthetic data options updated. Re-run the pipeline to regenerate data.")

st.divider()
st.subheader("Slippage scenario table editor")
scenario_options = [
    f"{idx}: {row.get('name', '(unnamed)')}"
    for idx, row in scenario_df.iterrows()
]

if scenario_options:
    selection = st.selectbox("Select scenario row", scenario_options)
    selected_index = int(selection.split(":")[0])
    current_row = scenario_df.loc[selected_index].to_dict()

    with st.form("scenario_row_editor"):
        name = st.text_input("Display name", value=str(current_row.get("name", "")))
        available_distributions = ["beta-binomial", "beta", "fixed"]
        current_distribution = current_row.get("contamination/contamination_rate/distribution", "beta-binomial")
        if current_distribution not in available_distributions:
            available_distributions.append(current_distribution)
        distribution = st.selectbox(
            "Contamination rate distribution",
            available_distributions,
            index=available_distributions.index(current_distribution),
        )
        value = st.text_input(
            "Contamination value / parameters",
            value=str(current_row.get("contamination/contamination_rate/value", "")),
            help="Provide numeric value or JSON array for parameterized distributions.",
        )
        arrangement = st.text_input(
            "Contamination arrangement",
            value=str(current_row.get("contamination/arrangement", "random")),
        )
        sample_strategy = st.text_input(
            "Inspection sample strategy",
            value=str(current_row.get("inspection/sample_strategy", "")),
        )
        proportion = st.number_input(
            "Inspection proportion value",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=float(current_row.get("inspection/proportion/value", 0) or 0),
        )
        detection_level = st.number_input(
            "Hypergeometric detection level",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=float(current_row.get("inspection/hypergeometric/detection_level", 0) or 0),
        )
        submit_update = st.form_submit_button("Update scenario row")

    if submit_update:
        scenario_df.at[selected_index, "name"] = name
        scenario_df.at[selected_index, "contamination/contamination_rate/distribution"] = distribution
        scenario_df.at[selected_index, "contamination/contamination_rate/value"] = value
        scenario_df.at[selected_index, "contamination/arrangement"] = arrangement
        scenario_df.at[selected_index, "inspection/sample_strategy"] = sample_strategy
        scenario_df.at[selected_index, "inspection/proportion/value"] = proportion
        scenario_df.at[selected_index, "inspection/hypergeometric/detection_level"] = detection_level
        set_scenario_dataframe(scenario_df)
        st.success("Scenario row updated.")
else:
    st.info("No scenarios available. Add one below.")

with st.expander("Add new scenario entry", expanded=False):
    with st.form("add_scenario_row"):
        new_name = st.text_input("Scenario identifier")
        new_arrangement = st.text_input("Arrangement", value="random")
        new_distribution = st.selectbox(
            "Contamination distribution",
            ["beta-binomial", "beta", "fixed"],
            key="new_distribution",
        )
        new_value = st.text_input("Contamination value / parameters", value="0.05", key="new_value")
        new_strategy = st.text_input("Sample strategy", value="rbs", key="new_strategy")
        new_proportion = st.number_input(
            "Inspection proportion",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=0.05,
            key="new_proportion",
        )
        create_row = st.form_submit_button("Add scenario")

    if create_row:
        new_entry = {
            "name": new_name or "custom-scenario",
            "contamination/contamination_rate/distribution": new_distribution,
            "contamination/contamination_rate/value": new_value,
            "contamination/arrangement": new_arrangement,
            "inspection/sample_strategy": new_strategy,
            "inspection/proportion/value": new_proportion,
        }
        updated_df = pd.concat([scenario_df, pd.DataFrame([new_entry])], ignore_index=True)
        set_scenario_dataframe(updated_df)
        st.success("New scenario added. Re-run the pipeline to include it.")
