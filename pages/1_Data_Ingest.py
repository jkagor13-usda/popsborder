import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation


st.set_page_config(
    page_title="Data Ingest (Merged with Consignment Generation)",
    page_icon=":inbox_tray:",
    layout="wide",
)
init_state()
render_sidebar_navigation()

st.title("Data Ingest & Consignment Generation")
st.caption(
    "This workflow has been consolidated into the **Consignment Generation** page. "
    "Use the button below to continue."
)
st.info(
    "You can upload curated PIS/RBS files, generate synthetic consignments, or define consignments from scratch "
    "from the combined page."
)

if st.button("Open Page 1 - Consignment Generation", type="primary", use_container_width=True):
    st.switch_page("pages/2_Consignment_Generation.py")

st.button("Back to Home", key="nav_home_from_legacy", on_click=lambda: st.switch_page("frontend.py"))
