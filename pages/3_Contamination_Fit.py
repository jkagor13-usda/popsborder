from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.slippage_ui import get_slippage_state, run_pipeline


def _load_preview(path: Path, rows: int = 50) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    try:
        return pd.read_csv(path).head(rows)
    except Exception:  # pylint: disable=broad-except
        return None


st.set_page_config(page_title="Contamination Fitting", layout="wide")
init_state()

state = get_slippage_state()
render_sidebar_navigation()
paths = state["paths"]

st.title("Page 3 - Contamination Fitting")
st.caption(
    "Review the current PIS and RBS inputs and fit the beta-binomial contamination parameters before "
    "configuring inspection policies."
)

col1, col2 = st.columns(2)
col1.metric("PIS data path", str(paths.pis_data))
col2.metric("RBS calculator path", str(paths.rbs_data))

pis_preview = _load_preview(paths.pis_data)
rbs_preview = _load_preview(paths.rbs_data)

st.subheader("PIS dataset preview")
if pis_preview is not None:
    st.dataframe(pis_preview, use_container_width=True)
else:
    st.info("Load or generate PIS data on Page 1 or Page 2.")

st.subheader("RBS calculator preview")
if rbs_preview is not None:
    st.dataframe(rbs_preview, use_container_width=True)
else:
    st.info("Load or generate an RBS calculator file on Page 1 or Page 2.")

st.divider()
if st.button("Fit contamination parameters", use_container_width=True):
    with st.spinner("Running Clarke beta-binomial fit..."):
        try:
            run_pipeline(run_scenarios=False)
            st.success("Contamination parameters updated. Continue to Page 4 for inspection policies.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Fitting failed: {exc}")

fit = state.get("fit")
if fit is None:
    st.info("No contamination fit available yet. Run the fitting process above.")
else:
    st.subheader("Latest contamination parameters")
    metrics = st.columns(3)
    metrics[0].metric("Alpha", f"{fit.alpha:.6f}")
    metrics[1].metric("Beta", f"{fit.beta:.6f}")
    metrics[2].metric("Theta", f"{fit.theta:.6f}")

st.caption(
    "These parameters are injected into the PoPS Border configuration and scenarios so that downstream "
    "inspection pages use the updated contamination distribution."
)

st.divider()
nav_cols = st.columns(2)
with nav_cols[0]:
    if st.button("Back to Page 2", type="primary", key="nav_back_page2"):
        st.switch_page("pages/2_Consignment_Generation.py")
with nav_cols[1]:
    if st.button("Continue to Page 4", type="primary", key="nav_forward_page4"):
        st.switch_page("pages/4_Inspection_Process.py")

