import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.slippage_ui import init_slippage_state


st.set_page_config(page_title="Home", layout="wide")
init_state()
state = init_slippage_state()
render_sidebar_navigation()

hero_cols = st.columns([2, 1])
with hero_cols[0]:
    st.title("PoPS Border Risk Based Sampling (RBS) Inspection Simulation")
    st.caption("Use this workspace to upload or generate consignments, fit contamination, and run inspection simulations.")
with hero_cols[1]:
    st.empty()


st.success(
    "Start with **Consignment Generation** to upload curated PIS/RBS files and generate synthetic consignments "
    "or define consignments from scratch."
)
if st.button(
    "Open Consignment Generation",
    type="primary",
    help="Upload PIS/RBS data, create synthetic consignments, or define consignments manually",
):
    st.switch_page("pages/1_Consignment_Generation.py")
st.markdown(
    """
**Workflow overview**

- **Page 1 - Consignment Generation**
  Generate consignments, optionally attach an RBS calculator, or define consignments when no data exists.
- **Page 2 - Contamination Fit**
  Upload PIS action data (and RBS if needed) and run the Clarke beta-binomial fitting to update contamination parameters.
- **Page 3 - Inspection Process**
  Ingest the compliance table, classify Low/Medium/High profiles, and configure inspection strategies.
- **Page 4 - Scenario & Experiment Builder**
  Summarize scenarios and set simulation controls (number of simulations, seed).
- **Page 5 - Run Simulation**
  Execute the pipeline, review slippage metrics, and compare inspection policies.
"""
)

st.divider()
st.subheader("Disclaimer")
st.warning(
    "This Information is for Authorized use only. Your ability to access this information is granted with the "
    "expectation and understanding that you will comply with and not violate privacy information policies. "
    "This is a private system and is only to be used by authorized users. By continuing, the user is stating "
    "that they are the indicated user.\n\n"
    "NO WARRANTY\n\n"
    'THE JOHNS HOPKINS UNIVERSITY APPLIED PHYSICS LABORATORY (JHU/APL) PROVIDES THIS SOFTWARE "AS IS" '
    "WITHOUT WARRANTY OF ANY KIND. JHU/APL DOES NOT WARRANT THAT (i) THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR "
    "FREE, OR (ii) THE DATA PRODUCED BY THE SOFTWARE WILL BE ERROR FREE. JHU/APL DISCLAIMS ALL WARRANTIES, WHETHER "
    "EXPRESS OR IMPLIED, INCLUDING (BUT NOT LIMITED TO) ANY AND ALL IMPLIED WARRANTIES OF PERFORMANCE, "
    "MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, NON-INTERFERENCE, AND ACCURACY OF "
    "INFORMATIONAL CONTENT. YOU THE USER ASSUME THE ENTIRE RISK AND LIABILITY OF USING THIS SOFTWARE OR THE DATA "
    "PRODUCED THEREBY, INCLUDING USE IN COMPLIANCE WITH ANY THIRD PARTY RIGHTS. JHU/APL SHALL NOT BE LIABLE FOR ANY "
    "ACTUAL, INDIRECT, CONSEQUENTIAL, SPECIAL OR OTHER DAMAGES ARISING FROM THE USE OF, OR INABILITY TO USE, THIS "
    "SOFTWARE OR THE DATA PRODUCED THEREBY, INCLUDING, BUT NOT LIMITED TO, ANY DAMAGES FOR LOST PROFITS, BUSINESS "
    "INTERRUPTION OR LOSS OF DATA EVEN IF JHU/APL HAS BEEN ADVISED OF THE PROBABILITY OF SUCH DAMAGES.\n\n" 

    "© 2025 Johns Hopkins University Applied Physics Laboratory"
)


st.divider()
logo_cols = st.columns([1, 1, 1])
with logo_cols[0]:
    st.image("gui/APHIS.svg", width=150)
with logo_cols[2]:
    st.image("gui/JHU_APL_logo.png", width=300)

st.markdown(
    "<div style='text-align:center; color:#b00000; font-weight:bold; margin-top:0.5rem;'>NOT FOR PUBLIC DISCLOSURE</div>",
    unsafe_allow_html=True,
)