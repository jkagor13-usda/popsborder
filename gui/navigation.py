import streamlit as st


def render_sidebar_navigation():
    with st.sidebar:
        st.image("gui/PoPS Border Logo.png", use_container_width=True)
        st.page_link("frontend.py", label="Home")
        st.page_link("pages/1_Consignment_Generation.py", label="Page 1 - Consignment Generation")
        st.page_link("pages/2_Contamination_Fit.py", label="Page 2 - Contamination Fit")
        st.page_link("pages/3_Inspection_Process.py", label="Page 3 - Inspection Process")
        st.page_link("pages/4_Scenario_Experiments.py", label="Page 4 - Scenario Experiments")
        st.page_link("pages/5_Run_Simulation.py", label="Page 5 - Run Simulation")
