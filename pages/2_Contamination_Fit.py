from pathlib import Path
from typing import Optional
import math
import shutil

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.slippage_pipeline import ClarkeFit, create_default_paths
from gui.slippage_ui import get_slippage_state, run_pipeline


def _load_preview(path: Path, rows: int = 50) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    try:
        return pd.read_csv(path).head(rows)
    except Exception:  # pylint: disable=broad-except
        return None


def _beta_pdf(alpha: float, beta: float, num_points: int = 200) -> pd.DataFrame:
    """Compute a beta PDF from alpha/beta without relying on SciPy.

    The curve is area-normalized so the integral from 0 to 1 equals 1.
    """
    eps = 1e-6
    xs = np.linspace(eps, 1 - eps, num_points)
    log_norm = math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
    ys = np.exp(log_norm + (alpha - 1) * np.log(xs) + (beta - 1) * np.log(1 - xs))
    area = np.trapz(ys, xs)
    if area > 0:
        ys = ys / area
    return pd.DataFrame({"prevalence": xs, "density": ys})


def _beta_chart(alpha: float, beta: float, title: str) -> alt.Chart:
    pdf = _beta_pdf(alpha, beta)
    y_max = float(pdf["density"].max() * 1.05) if not pdf.empty else 1.0
    base = alt.Chart(pdf).encode(
        x=alt.X("prevalence:Q", title="Contamination prevalence"),
        y=alt.Y("density:Q", title="Density", scale=alt.Scale(domain=[0, y_max])),
    )
    area = base.mark_area(opacity=0.3, color="#1f77b4")
    line = base.mark_line(color="#1f77b4", strokeWidth=2)
    return (area + line).properties(height=200, title=title)


st.set_page_config(page_title="Contamination Fitting", layout="wide")
init_state()

state = get_slippage_state()
render_sidebar_navigation()
paths = state["paths"]
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)

st.title("Page 2 - Contamination Fitting")
st.caption(
    "Upload or confirm PIS action and RBS inputs, then fit the beta-binomial contamination parameters before "
    "configuring inspection policies."
)

fit_tab, assign_tab = st.tabs(["Fit contamination", "Assign contamination"])

with fit_tab:
    st.subheader("Upload/confirm inputs")
    upload_cols = st.columns(2)
    pis_upload = upload_cols[0].file_uploader("PIS action data CSV", type=["csv"], key="fit_pis_upload")
    rbs_upload = upload_cols[1].file_uploader("RBS calculator CSV", type=["csv"], key="fit_rbs_upload")
    if pis_upload is not None:
        try:
            pis_path = TMP_DIR / "fit_pis_data.csv"
            pis_path.parent.mkdir(parents=True, exist_ok=True)
            pis_path.write_bytes(pis_upload.getbuffer())
            set_paths(pis_data=pis_path, synthetic_seed=pis_path)
            st.success(f"PIS action data saved to {pis_path}")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unable to save PIS action data: {exc}")
    if rbs_upload is not None:
        try:
            rbs_path = TMP_DIR / "fit_rbs_data.csv"
            rbs_path.parent.mkdir(parents=True, exist_ok=True)
            rbs_path.write_bytes(rbs_upload.getbuffer())
            set_paths(rbs_data=rbs_path)
            st.success(f"RBS data saved to {rbs_path}")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unable to save RBS data: {exc}")

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
    st.altair_chart(
        _beta_chart(float(fit.alpha), float(fit.beta), "Latest contamination beta PDF"),
        use_container_width=True,
    )

st.caption(
    "These parameters are injected into the PoPS Border configuration and scenarios so that downstream "
    "inspection pages use the updated contamination distribution."
)

st.divider()
nav_cols = st.columns(3)
with nav_cols[0]:
    if st.button(
        "Reset and Return Home",
        type="secondary",
        key="nav_reset_page3",
        help="Delete temporary files and restart from the home page",
    ):
        try:
            if TMP_DIR.exists():
                shutil.rmtree(TMP_DIR)
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            state["paths"] = create_default_paths()
            state["fit"] = None
            st.switch_page("frontend.py")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unable to reset temporary files: {exc}")
with nav_cols[1]:
    if st.button("Previous Page", type="primary", key="nav_back_page2"):
        st.switch_page("pages/1_Consignment_Generation.py")
with nav_cols[2]:
    if st.button("Next Page", type="primary", key="nav_forward_page4"):
        st.switch_page("pages/4_Inspection_Process.py")

