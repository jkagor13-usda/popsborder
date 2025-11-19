import streamlit as st


def render_sidebar_navigation():
    with st.sidebar:
        st.image("gui/PoPS Border Logo.png", use_container_width=True)
        st.page_link("frontend.py", label="Home")
        st.page_link("pages/1_Data_Ingest.py", label="Page 1 - Data Ingest")
        st.page_link("pages/2_Consignment_Generation.py", label="Page 2 - Consignment Generation")
        st.page_link("pages/3_Contamination_Fit.py", label="Page 3 - Contamination Fit")
        st.page_link("pages/4_Inspection_Process.py", label="Page 4 - Inspection Process")
        st.page_link("pages/5_Scenario_Experiments.py", label="Page 5 - Scenario Experiments")
        st.page_link("pages/6_Run_Simulation.py", label="Page 6 - Run Simulation")
