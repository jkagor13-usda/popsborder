import streamlit as st
from lib.models import init_state, set_inspection

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
