from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.slippage_pipeline import ClarkeFit
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

st.title("Page 2 - Contamination Fitting")
st.caption(
    "Review the current PIS and RBS inputs and fit the beta-binomial contamination parameters before "
    "configuring inspection policies."
)

fit_tab, assign_tab = st.tabs(["Fit contamination", "Assign contamination"])

with fit_tab:
    pis_preview = _load_preview(paths.pis_data)
    rbs_preview = _load_preview(paths.rbs_data)

    with st.expander("PIS/RBS previews", expanded=True):
        prev_cols = st.columns(2)
        with prev_cols[0]:
            st.subheader("PIS dataset preview")
            if pis_preview is not None:
                st.dataframe(pis_preview, use_container_width=True)
            else:
                st.info("Load or generate PIS data on Page 1.")
        with prev_cols[1]:
            st.subheader("RBS calculator preview")
            if rbs_preview is not None:
                st.dataframe(rbs_preview, use_container_width=True)
            else:
                st.info("Load or generate an RBS calculator file on Page 1.")

    st.divider()
    if st.button("Fit contamination parameters", type="secondary", use_container_width=True):
        with st.spinner("Running Clarke beta-binomial fit..."):
            try:
                run_pipeline(run_scenarios=False)
                st.success("Contamination parameters updated. Continue to Page 4 for inspection policies.")
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Fitting failed: {exc}")

with assign_tab:
    st.subheader("Assign contamination parameters manually")
    fit_current = state.get("fit")
    alpha_default = float(fit_current.alpha) if fit_current else 0.01
    beta_default = float(fit_current.beta) if fit_current else 5.0
    theta_default = float(fit_current.theta) if fit_current else 0.5
    col_a, col_b, col_t = st.columns(3)
    alpha_val = col_a.number_input("Alpha", min_value=0.0, value=alpha_default, step=0.001, format="%.6f")
    beta_val = col_b.number_input("Beta", min_value=0.0, value=beta_default, step=0.001, format="%.6f")
    theta_val = col_t.number_input("Theta", min_value=0.0, max_value=1.0, value=theta_default, step=0.01)
    if st.button("Save assigned contamination parameters", type="secondary", use_container_width=True):
        state["fit"] = ClarkeFit(alpha=float(alpha_val), beta=float(beta_val), theta=float(theta_val), raw_result={})
        st.success("Contamination parameters assigned. Downstream steps will use these values until refit.")

fit = state.get("fit")
if fit is None:
    st.info("No contamination fit available yet. Fit or assign parameters above.")
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

