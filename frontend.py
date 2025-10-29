import streamlit as st
from lib.models import init_state, Scenario, Inspection

st.set_page_config(page_title="PoPS Border – Wireframe", page_icon="🧰", layout="wide")
init_state()

st.title("PoPS Border – Inspection Simulation (Wireframe)")
st.caption("Three pages: Scenario Builder → Inspection Strategy → Simulation")

s = st.session_state.scenario
i = st.session_state.inspection

col = st.columns(6)
col[0].metric("Units", f"{s.units_in_shipment:,}")
col[1].metric("Items / unit", f"{s.items_per_unit:,}")
col[2].metric("Prevalence (%)", f"{s.contamination_prevalence_pct:.2f}")
col[3].metric("Arrangement", s.contamination_arrangement)
col[4].metric("Method", i.method)
col[5].metric("Sample units", f"{i.sample_units:,}")

st.markdown("#### Quick links")
st.page_link("pages/1_📦_Scenario_Builder.py", label="Scenario Builder", icon="📦")
st.page_link("pages/2_🔍_Inspection_Strategy.py", label="Inspection Strategy", icon="🔍")
st.page_link("pages/3_▶️_Simulation.py", label="Run Simulation", icon="▶️")

st.divider()
st.markdown(
    """
    **How it works**  
    - Define shipment & contamination on **Scenario Builder**.  
    - Configure sampling method & sample sizes on **Inspection Strategy**.  
    - Execute the (mock) simulation and visualize outcomes on **Simulation**.  
    """
)
