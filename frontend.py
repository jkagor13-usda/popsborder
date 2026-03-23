# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

import streamlit as st

from pathlib import Path

from gui.env_check import build_update_commands, get_package_mismatches
from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.page_styles import apply_shared_page_styles, render_page_intro
from gui.runtime_warnings import suppress_optional_dependency_warnings
from gui.slippage_ui import init_slippage_state

suppress_optional_dependency_warnings()


st.set_page_config(page_title="Home", layout="wide")
init_state()
state = init_slippage_state()
render_sidebar_navigation()
apply_shared_page_styles()

st.warning(
    "**Test Deployment Notice: This is a test deployment with limited functionality and is under active development. "
    "Features may be incomplete and subject to change. Results have not been validated.**"
)
st.title("PoPS Border Risk Based Sampling (RBS) Inspection Simulation")
render_page_intro(
    "Use this workspace to upload or generate consignments, fit contamination, and run inspection simulations."
)



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
requirements_path = Path("requirements.txt")
if requirements_path.exists():
    mismatches = get_package_mismatches(requirements_path)
    if mismatches:
        st.markdown(
            f"<div style='color:#b00020; font-weight:600;'>{len(mismatches)} mismatches detected.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='color:#2e7d32; font-weight:600;'>All package versions matched.</div>",
            unsafe_allow_html=True,
        )
    expander_label = "Package version check"
    with st.expander(expander_label, expanded=False):
        if mismatches:
            st.markdown(
                "<div style='color:#b00020; font-weight:600;'>Package version mismatches detected.</div>",
                unsafe_allow_html=True,
            )
            for item in mismatches:
                st.markdown(
                    f"<div style='color:#b00020;'>- <code>{item.package}</code>: "
                    f"<span style='color:#111;'>required <code>{item.required}</code>, installed "
                    f"<code>{item.installed or 'not installed'}</code></span></div>",
                    unsafe_allow_html=True,
                )
            update_commands = build_update_commands(requirements_path, mismatches)
            st.caption("Run these commands in your virtual environment to update the mismatched packages:")
            st.code(update_commands, language="powershell")
        else:
            st.markdown(
                "<div style='color:#2e7d32; font-weight:600;'>Installed package versions match requirements.txt.</div>",
                unsafe_allow_html=True,
            )

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
