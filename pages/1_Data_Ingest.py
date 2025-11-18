from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.slippage_pipeline import SyntheticOptions
from gui.slippage_ui import (
    get_slippage_state,
    run_pipeline,
    set_paths,
    set_synthetic_options,
)


st.set_page_config(
    page_title="Data Ingest & Synthetic Consignments",
    page_icon=":inbox_tray:",
    layout="wide",
)
init_state()
state = get_slippage_state()
render_sidebar_navigation()
paths = state["paths"]
state.setdefault("consignment_source", "synthetic")
synthetic_options: SyntheticOptions = state["synthetic_options"]


def _persist_upload(df: pd.DataFrame, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def _select_columns(df: pd.DataFrame, keywords: List[str]) -> List[str]:
    selected: List[str] = []
    for word in keywords:
        match = next((col for col in df.columns if word in col.lower()), None)
        if match and match not in selected:
            selected.append(match)
    return selected


def _summarize_rbs(df: pd.DataFrame) -> pd.DataFrame:
    grouping_cols = _select_columns(df, ["consignment", "origin", "material"])
    if not grouping_cols:
        return df.head(25)
    summary = df.groupby(grouping_cols).size().reset_index(name="records")
    return summary.sort_values("records", ascending=False).head(50)


st.title("Page 1 - Data Ingest, Synthetic Consignment Generation, Contamination Fit")
st.caption(
    "Upload PoPS Inspection Station (PIS) data and RBS Calculator extracts, "
    "decide whether you want to use them directly or generate synthetic consignments, "
    "and fit beta-binomial contamination parameters."
)

source_choice = st.radio(
    "Consignment source for downstream analysis",
    [
        "Generate synthetic consignments",
        "Use uploaded historical consignments",
    ],
    index=0 if state["consignment_source"] == "synthetic" else 1,
    horizontal=True,
)
state["consignment_source"] = "synthetic" if source_choice.startswith("Generate") else "historical"

with st.expander("Upload inspection program data", expanded=True):
    col_pis, col_rbs = st.columns(2)
    pis_upload = col_pis.file_uploader("PIS data CSV", type=["csv"])
    rbs_upload = col_rbs.file_uploader("RBS calculator CSV", type=["csv"])

    if pis_upload is not None:
        pis_df = pd.read_csv(pis_upload)
        state["pis_data"] = pis_df
        state["pis_preview"] = pis_df.head(10)
        col_pis.success(f"Loaded {len(pis_df):,} PIS records.")
        if col_pis.button("Use uploaded PIS data", key="use_pis"):
            target = paths.data_dir / "uploaded_pis_data.csv"
            _persist_upload(pis_df, target)
            set_paths(pis_data=target)
            col_pis.success(f"PIS data path set to {target}")

    if rbs_upload is not None:
        rbs_df = pd.read_csv(rbs_upload)
        state["rbs_data"] = rbs_df
        state["rbs_preview"] = rbs_df.head(10)
        col_rbs.success(f"Loaded {len(rbs_df):,} RBS records.")
        if col_rbs.button("Use uploaded RBS calculator data", key="use_rbs"):
            target = paths.data_dir / "uploaded_rbs_data.csv"
            _persist_upload(rbs_df, target)
            set_paths(rbs_data=target)
            col_rbs.success(f"RBS calculator path set to {target}")

pis_df: Optional[pd.DataFrame] = state.get("pis_data")
rbs_df: Optional[pd.DataFrame] = state.get("rbs_data")

preview_cols = st.columns(2)
with preview_cols[0]:
    if pis_df is not None and not pis_df.empty:
        st.subheader("PIS - inspection overview")
        st.metric("Total observations", f"{len(pis_df):,}")
        numeric_cols = pis_df.select_dtypes("number")
        if not numeric_cols.empty:
            st.line_chart(numeric_cols.iloc[:, : min(3, numeric_cols.shape[1])])
        st.dataframe(pis_df.head(25), use_container_width=True, height=300)
    else:
        st.info("Upload PIS data to view summary statistics.")

with preview_cols[1]:
    if rbs_df is not None and not rbs_df.empty:
        st.subheader("RBS - consignment summary by origin/material")
        summary = _summarize_rbs(rbs_df)
        st.dataframe(summary, use_container_width=True, height=300)
    else:
        st.info("Upload RBS calculator data to view consignment summaries.")

if state["consignment_source"] == "historical" and (
    (pis_df is None or pis_df.empty) or (rbs_df is None or rbs_df.empty)
):
    st.warning("Upload both PIS and RBS data to rely on historical consignments.")

st.divider()
st.subheader("Synthetic consignment controls")
col_syn = st.columns(3)
syn_samples = col_syn[0].number_input(
    "Synthetic consignments",
    min_value=10,
    max_value=50000,
    value=int(synthetic_options.n_samples),
    step=50,
)
syn_method = col_syn[1].selectbox(
    "Sampling method",
    ["naive", "sequential", "gmm", "gaussian_copula"],
    index=["naive", "sequential", "gmm", "gaussian_copula"].index(synthetic_options.sampling_method),
)
syn_seed = col_syn[2].number_input(
    "Random seed",
    min_value=0,
    max_value=999999,
    value=int(state["engine_options"].get("seed", 42)),
    step=1,
)

if syn_samples != synthetic_options.n_samples or syn_method != synthetic_options.sampling_method:
    set_synthetic_options(SyntheticOptions(n_samples=int(syn_samples), sampling_method=syn_method))
    st.caption("Synthetic generation options updated.")

if syn_seed != state["engine_options"].get("seed"):
    state["engine_options"]["seed"] = int(syn_seed)

st.markdown(
    "Select whether you want to **generate synthetic consignments** (recommended for quick experiments) "
    "or rely on uploaded historical consignments. When you generate synthetic data, the contamination "
    "distribution (beta-binomial) will also be re-fit."
)

if st.button("Generate synthetic consignments & fit contamination model", use_container_width=True):
    with st.spinner("Running generator and fitting contamination model..."):
        try:
            run_pipeline()
            st.success("Synthetic data regenerated and contamination parameters updated.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Pipeline failed: {exc}")

st.divider()
st.subheader("Synthetic consignments & contamination fit")
synthetic_preview = state.get("synthetic_preview")
synthetic_df = state.get("synthetic_data")

metrics_cols = st.columns(3)
if synthetic_df is not None and not synthetic_df.empty:
    metrics_cols[0].metric("Consignments generated", f"{len(synthetic_df):,}")
    unique_origins = synthetic_df["COUNTRY_OF_ORIGIN_NAME"].nunique() if "COUNTRY_OF_ORIGIN_NAME" in synthetic_df else 0
    metrics_cols[1].metric("Unique origins", unique_origins or "n/a")
    units_col = next((col for col in synthetic_df.columns if "TOTAL_SAMPLING_UNITS" in col), None)
    if units_col:
        metrics_cols[2].metric("Avg sampling units", f"{synthetic_df[units_col].mean():,.1f}")
    else:
        metrics_cols[2].metric("Avg sampling units", "n/a")
else:
    metrics_cols[0].metric("Consignments generated", "0")
    metrics_cols[1].metric("Unique origins", "n/a")
    metrics_cols[2].metric("Avg sampling units", "n/a")

if synthetic_preview is None or synthetic_preview.empty:
    st.info("Run the synthetic generator to view a preview.")
else:
    st.dataframe(synthetic_preview, use_container_width=True, height=320)

fit = state.get("fit")
if fit is None:
    st.info("Once the beta-binomial model is fit, alpha/beta/theta will be shown here.")
else:
    st.markdown("#### Beta-binomial contamination parameters")
    fit_cols = st.columns(3)
    fit_cols[0].metric("Alpha", f"{fit.alpha:.4f}")
    fit_cols[1].metric("Beta", f"{fit.beta:.4f}")
    fit_cols[2].metric("Theta", f"{fit.theta:.4f}")

    st.caption(
        "Beta-binomial parameters are injected into downstream scenarios so the inspection "
        "process page works with simulated contamination probabilities."
    )
