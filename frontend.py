import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.slippage_ui import export_state_snapshot, init_slippage_state


st.set_page_config(page_title="Home", layout="wide")
init_state()
state = init_slippage_state()
render_sidebar_navigation()

st.title("POP")
st.caption("Start here to decide whether you have data ready and review the workflow.")

choice = st.radio(
    "Do you have PIS + RBS data ready to ingest?",
    [
        "Yes, I already have curated data.",
        "No, I need to rely on synthetic consignments.",
        "I am not sure yet.",
    ],
)

if choice == "Yes, I already have curated data.":
    st.success(
        "Head to **Page 1 - Data Ingest & Synthetic Consignments** to upload your files, "
        "review the basic statistics, and indicate whether you want to rely on historical consignments."
    )
elif choice == "No, I need to rely on synthetic consignments.":
    st.info(
        "Since you need synthetic consignments, jump directly to **Page 2 - User-defined Consignments** "
        "to build the scenario CSV. After that, you can proceed to inspection configuration and simulations."
    )
    if st.button("Go to Page 2 - User-defined Consignments"):
        st.query_params["page"] = "pages/2_Scenario_Builder.py"
else:
    st.warning(
        "You can explore the workflow in order and come back once you know which "
        "datasets you want to use."
    )

st.markdown(
    """
**Workflow overview**

- **Page 1 – Data Ingest, Synthetic Consignment Generation, Contamination Fitting**
  Upload PIS/RBS data, view RBS origin/material summaries, choose historical vs. generated consignments,
  and fit the beta-binomial parameters used later.
- **Page 2 – User-Defined Consignment Builder**
  Select this page if you do not have ingested data. Define simple consignment structures and create a scenario CSV.
- **Page 3 – Inspection Process**
  Ingest the compliance table, classify Low/Medium/High profiles, configure RBS parameters, and choose inspection policies.
- **Page 4 – Scenario & Experiment Builder**
  Summarize configured scenarios and set the experiment controls (number of simulations, consignments, expected runtime).
- **Page 5 – Run Simulation**
  Execute the pipeline, review slippage metrics, compare policies, and visualize slippage vs. inspected units.
"""
)

st.divider()
st.subheader("Disclaimer")
st.warning(
    "This tool is for pre-decisional exploration. Any values, results, or policy "
    "recommendations are illustrative only until validated and approved through the "
    "appropriate USDA APHIS channels."
)
st.caption(
    "Confirm that the data you upload is sanitized and cleared for analysis in this environment."
)

snapshot = export_state_snapshot()
st.divider()
st.subheader("Current session snapshot")
snapshot_cols = st.columns(4)
snapshot_cols[0].metric("Scenario rows", snapshot["scenario_rows"])
snapshot_cols[1].metric("Results rows", snapshot["results_rows"])
last_run = snapshot.get("last_run")
snapshot_cols[2].metric("Last run", value=str(last_run) if last_run else "Never")
snapshot_cols[3].metric("Pipeline status", snapshot.get("run_error") or "Ready")

st.json(snapshot)
