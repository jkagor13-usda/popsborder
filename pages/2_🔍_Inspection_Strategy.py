import streamlit as st

from lib.models import init_state, set_inspection
from lib.slippage_ui import get_slippage_state, set_scenario_dataframe

st.set_page_config(page_title="Inspection Strategy", page_icon="🔍", layout="wide")
init_state()

st.title("🔍 Inspection Strategy")
st.caption("Choose a sampling method and sample sizes; set acceptance number.")

with st.form("inspection_form"):
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    with c1:
        method = st.selectbox(
            "Sampling method",
            ["Simple random", "Systematic", "Stratified", "Targeted"],
            index=["Simple random","Systematic","Stratified","Targeted"].index(st.session_state.inspection.method)
        )
    with c2:
        sample_units = st.number_input("Sampled units", 1, 2000, st.session_state.inspection.sample_units, step=1)
    with c3:
        items_per = st.number_input("Items per sampled unit", 1, 2000, st.session_state.inspection.sample_items_per_unit, step=1)
    with c4:
        aql = st.number_input("Acceptance number", 0, 1000, st.session_state.inspection.acceptance_number, step=1,
                              help="0 → any detection triggers reject.")

    saved = st.form_submit_button("Save strategy", use_container_width=True)
    if saved:
        set_inspection(method=method, sample_units=int(sample_units), sample_items_per_unit=int(items_per), acceptance_number=int(aql))
        st.success("Inspection strategy saved.")

# Derived preview using current scenario
s = st.session_state.scenario
i = st.session_state.inspection
N_items = s.units_in_shipment * s.items_per_unit
sampled = i.sample_units * i.sample_items_per_unit
st.markdown("#### At a glance")
cc = st.columns(4)
cc[0].metric("Total items", f"{N_items:,}")
cc[1].metric("Items sampled", f"{sampled:,}")
cc[2].metric("Sampling fraction", f"{100*sampled/max(1,N_items):.3f}%")
cc[3].metric("Acceptance #", f"{i.acceptance_number}")

st.divider()
st.subheader("Slippage inspection parameters")
state = get_slippage_state()
scenario_df = state["scenario_df"]

if scenario_df.empty:
    st.info("The slippage scenario table is empty. Add scenarios on the Scenario Builder page.")
else:
    options = [f"{idx}: {row.get('name', '(unnamed)')}" for idx, row in scenario_df.iterrows()]
    selected = st.selectbox("Select scenario", options, key="slippage_inspection_selector")
    selected_idx = int(selected.split(":")[0])
    current = scenario_df.loc[selected_idx]

    with st.form("slippage_inspection_form"):
        sample_strategy = st.text_input(
            "Sample strategy",
            value=str(current.get("inspection/sample_strategy", "")),
            help="Matches inspection/sample_strategy in the scenario table.",
        )
        selection_strategy = st.text_input(
            "Selection strategy",
            value=str(current.get("inspection/selection_strategy", "")),
        )
        within_box = st.number_input(
            "Within-box proportion",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=float(current.get("inspection/within_box_proportion", 0) or 0),
        )
        min_boxes = st.number_input(
            "Minimum boxes",
            min_value=0,
            max_value=1000,
            step=1,
            value=int(current.get("inspection/min_boxes", 0) or 0),
        )
        proportion_value = st.number_input(
            "Inspection proportion value",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=float(current.get("inspection/proportion/value", 0) or 0),
        )
        update_row = st.form_submit_button("Update inspection parameters")

    if update_row:
        scenario_df.at[selected_idx, "inspection/sample_strategy"] = sample_strategy
        scenario_df.at[selected_idx, "inspection/selection_strategy"] = selection_strategy
        scenario_df.at[selected_idx, "inspection/within_box_proportion"] = within_box
        scenario_df.at[selected_idx, "inspection/min_boxes"] = min_boxes
        scenario_df.at[selected_idx, "inspection/proportion/value"] = proportion_value
        set_scenario_dataframe(scenario_df)
        st.success("Scenario inspection parameters updated. Re-run the pipeline to apply changes.")
