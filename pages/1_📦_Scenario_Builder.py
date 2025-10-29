import streamlit as st
from lib.models import init_state, set_scenario

st.set_page_config(page_title="Scenario Builder", page_icon="📦", layout="wide")
init_state()

st.title("📦 Scenario Builder")
st.caption("Define shipment structure and contamination characteristics.")

with st.form("scenario_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        title = st.text_input("Scenario name", value=st.session_state.scenario.title)
        seed = st.number_input("Random seed", value=st.session_state.scenario.seed, min_value=0, step=1)
        runs = st.slider("Simulation runs", 50, 5000, st.session_state.scenario.runs, step=50)
    with col2:
        unit_type = st.selectbox("Consignment unit", ["Boxes", "Crates", "Pallets"], index=["Boxes","Crates","Pallets"].index(st.session_state.scenario.unit_type))
        units = st.number_input(f"Number of {unit_type.lower()}", 1, 20000, st.session_state.scenario.units_in_shipment, step=50)
        items_per_unit = st.number_input("Items per unit", 1, 10000, st.session_state.scenario.items_per_unit, step=5)
    with col3:
        packaging = st.selectbox("Packaging type", ["Cardboard", "Wood", "Plastic", "Mixed"], index=["Cardboard","Wood","Plastic","Mixed"].index(st.session_state.scenario.packaging_type))
        prevalence = st.slider("Contamination prevalence (%)", 0.01, 20.0, st.session_state.scenario.contamination_prevalence_pct, step=0.01)
        arrangement = st.selectbox("Arrangement pattern", ["Random", "Clustered", "Edge-loaded"], index=["Random","Clustered","Edge-loaded"].index(st.session_state.scenario.contamination_arrangement))

    cv = st.slider("Dispersion (CV)", 0.1, 3.0, st.session_state.scenario.contamination_cv, step=0.1,
                   help="Higher CV → more variable contamination across units.")
    notes = st.text_area("Notes", value=st.session_state.scenario.notes, height=80)

    submitted = st.form_submit_button("Save scenario", use_container_width=True)
    if submitted:
        set_scenario(
            title=title, seed=int(seed), runs=int(runs),
            unit_type=unit_type, units_in_shipment=int(units), items_per_unit=int(items_per_unit),
            packaging_type=packaging, contamination_prevalence_pct=float(prevalence),
            contamination_arrangement=arrangement, contamination_cv=float(cv), notes=notes
        )
        st.success("Scenario saved.")

# Quick summary
s = st.session_state.scenario
col = st.columns(4)
col[0].metric("Units", f"{s.units_in_shipment:,}")
col[1].metric("Items / unit", f"{s.items_per_unit:,}")
col[2].metric("Prevalence (%)", f"{s.contamination_prevalence_pct:.2f}")
col[3].metric("Arrangement", s.contamination_arrangement)
