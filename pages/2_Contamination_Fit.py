from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.slippage_pipeline import ClarkeFit, create_default_paths, fit_contamination_distribution
from gui.slippage_ui import get_slippage_state, set_paths

TMP_DIR = Path("tmp")
CONTAM_DIR = TMP_DIR / "contamination"
PARAM_STORE = CONTAM_DIR / "contamination_parameter_sets.json"
FALLBACK_ALPHA = 0.194628
FALLBACK_BETA = 4.7609372
FALLBACK_THETA = float("inf")

TMP_DIR.mkdir(parents=True, exist_ok=True)
CONTAM_DIR.mkdir(parents=True, exist_ok=True)


def _load_preview(path: Optional[Path], rows: int = 50) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    try:
        return pd.read_csv(path).head(rows)
    except Exception:  # pylint: disable=broad-except
        return None


def _beta_pdf(alpha: float, beta: float, num_points: int = 200) -> pd.DataFrame:
    """Compute a normalized beta PDF."""

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
    return (area + line).properties(height=220, title=title)


def _read_param_store() -> dict:
    if not PARAM_STORE.exists():
        return {}
    try:
        return json.loads(PARAM_STORE.read_text())
    except Exception:  # pylint: disable=broad-except
        return {}


def _write_param_store(store: dict):
    PARAM_STORE.parent.mkdir(parents=True, exist_ok=True)
    PARAM_STORE.write_text(json.dumps(store, indent=2))


def _next_param_name(store: dict, base: str = "contamination_param_set") -> str:
    idx = 1
    while f"{base}_{idx}" in store:
        idx += 1
    return f"{base}_{idx}"


def _save_param_set(name: str, alpha: float, beta: float, theta: float) -> str:
    store = _read_param_store()
    if not name:
        name = _next_param_name(store)
    store[name] = {"alpha": alpha, "beta": beta, "theta": theta}
    _write_param_store(store)
    st.session_state["last_saved_param_set"] = name
    return name


# ---- Page setup ----
init_state()
slippage_state = get_slippage_state()
paths = slippage_state["paths"]
render_sidebar_navigation()

st.title("Page 2 - Contamination Fit")
st.caption(
    "Fit, assign, and manage contamination parameters. PIS upload and RBS selection live in the Fit tab. "
    "All outputs are written to tmp/contamination."
)

# ---- Tabs ----
fit_tab, assign_tab, saved_tab = st.tabs(
    ["Fit contamination", "Assign contamination manually", "Saved parameter sets"]
)

# Fit contamination tab
with fit_tab:
    st.subheader("Fit beta-binomial contamination parameters")
    st.caption("Fits are based on the PIS action upload and RBS calculator selection in this tab.")

    ingest_cols = st.columns(2)
    with ingest_cols[0]:
        st.subheader("PIS action data upload")
        pis_upload = st.file_uploader(
            "Upload PIS action CSV",
            type=["csv"],
            key="pis_upload",
            help="PIS action data is required for contamination fitting.",
        )
        if pis_upload is not None:
            dest = CONTAM_DIR / "fit_pis_data.csv"
            df = pd.read_csv(pis_upload)
            dest.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(dest, index=False)
            set_paths(pis_data=dest)
            paths = slippage_state["paths"]
            st.success(f"Saved PIS data to {dest}")
        pis_preview = _load_preview(slippage_state["paths"].pis_data)
        if pis_preview is not None:
            st.dataframe(pis_preview, use_container_width=True)
        else:
            st.info("No PIS action data loaded yet.")

        st.subheader("Select RBS calculator file")
        rbs_candidates = sorted(
            [p for p in (Path("tmp") / "consignments").glob("*.csv") if "rbs" in p.name.lower()]
        )
        if not rbs_candidates:
            st.warning("No RBS files found in tmp/consignments. Add one on Page 1.")
        else:
            current_rbs = slippage_state["paths"].rbs_data
            default_idx = 0
            if current_rbs in rbs_candidates:
                default_idx = rbs_candidates.index(current_rbs)
            chosen_rbs = st.selectbox("RBS file", rbs_candidates, index=default_idx, format_func=lambda p: p.name)
            set_paths(rbs_data=chosen_rbs, synthetic_seed=chosen_rbs)
            paths = slippage_state["paths"]

    with ingest_cols[1]:
        st.subheader("PIS action summary statistics")
        pis_df = None
        if slippage_state["paths"].pis_data and Path(slippage_state["paths"].pis_data).exists():
            try:
                pis_df = pd.read_csv(slippage_state["paths"].pis_data)
            except Exception:  # pylint: disable=broad-except
                pis_df = None

        if pis_df is None or pis_df.empty:
            st.info("No PIS data loaded yet.")
        else:
            def _find_col(df, substrings):
                lower_map = {c.lower(): c for c in df.columns}
                for s in substrings:
                    for lc, orig in lower_map.items():
                        if s in lc:
                            return orig
                return None

            ins_col = _find_col(pis_df, ["inspection"])
            samp_col = _find_col(pis_df, ["total_sampling"])
            plant_col = _find_col(pis_df, ["total_plant"])

            n_rows = len(pis_df)
            unique_inspections = pis_df[ins_col].nunique() if ins_col else 0
            total_sampling = pis_df[samp_col].sum() if samp_col else 0
            total_plants = pis_df[plant_col].sum() if plant_col else 0
            action_col = _find_col(pis_df, ["action"])
            action_ones = int((pis_df[action_col] == 1).sum()) if action_col else 0

            stats = st.columns(2)
            stats[0].metric("Rows", f"{n_rows}")
            stats[1].metric("Unique inspections", f"{unique_inspections}")
            stats2 = st.columns(2)
            stats2[0].metric("Total sampling units", f"{total_sampling}")
            stats2[1].metric("Total plant quantity", f"{total_plants}")
            st.metric("Rows with action = 1", f"{action_ones}")

    fit_cols = st.columns(2)
    with fit_cols[0]:
        if st.button("Fit contamination parameters", type="primary"):
            if paths.pis_data is None or paths.rbs_data is None:
                st.error("Upload PIS data and select an RBS file before fitting.")
            else:
                try:
                    fit, pis_df, rbs_df = fit_contamination_distribution(paths.pis_data, paths.rbs_data)
                    slippage_state["fit"] = fit
                    st.session_state["last_fit_params"] = {
                        "alpha": fit.alpha,
                        "beta": fit.beta,
                        "theta": fit.theta,
                    }
                    st.success(
                        f"Fit succeeded: alpha={fit.alpha:.6f}, beta={fit.beta:.6f}, theta={fit.theta}"
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    st.error(f"Fitting failed: {exc}")
                    fit = ClarkeFit(
                        alpha=FALLBACK_ALPHA,
                        beta=FALLBACK_BETA,
                        theta=FALLBACK_THETA,
                        raw_result={"fallback": True, "error": str(exc)},
                    )
                    slippage_state["fit"] = fit
                    st.info("Applied fallback parameters.")

    fit_to_show: Optional[ClarkeFit] = slippage_state.get("fit")
    if fit_to_show:
        metrics = st.columns(3)
        metrics[0].metric("Alpha", f"{fit_to_show.alpha:.6f}")
        metrics[1].metric("Beta", f"{fit_to_show.beta:.6f}")
        metrics[2].metric("Theta", f"{fit_to_show.theta}")
        st.altair_chart(
            _beta_chart(fit_to_show.alpha, fit_to_show.beta, "Beta PDF (fitted)"),
            use_container_width=True,
        )

    st.markdown("**Save fitted parameters**")
    name_input = st.text_input(
        "Parameter set name",
        value=st.session_state.get("last_saved_param_set", ""),
        key="fit_save_name",
    )
    if st.button("Save fitted parameters", key="save_fit_params"):
        if fit_to_show is None:
            st.error("No parameters to save. Fit parameters first.")
        else:
            saved_name = _save_param_set(
                name_input or _next_param_name(_read_param_store()),
                fit_to_show.alpha,
                fit_to_show.beta,
                fit_to_show.theta,
            )
            st.success(
                f"Saved '{saved_name}' with alpha={fit_to_show.alpha:.6f}, "
                f"beta={fit_to_show.beta:.6f}, theta={fit_to_show.theta}"
            )

# Manual assignment tab
with assign_tab:
    st.subheader("Assign contamination parameters manually")
    st.caption(
        "Adjust alpha/beta directly. Theta is fixed to infinity by default. The beta PDF preview updates live."
    )

    assigned_state = st.session_state.get(
        "page2_assigned_fit",
        {"alpha": FALLBACK_ALPHA, "beta": FALLBACK_BETA, "theta": FALLBACK_THETA},
    )

    col_a, col_b, col_t = st.columns(3)
    alpha_val = col_a.number_input("Alpha", min_value=0.000001, value=float(assigned_state["alpha"]), step=0.01)
    beta_val = col_b.number_input("Beta", min_value=0.000001, value=float(assigned_state["beta"]), step=0.01)
    theta_str = col_t.text_input("Theta", value="inf")
    try:
        theta_val = float("inf") if theta_str.lower() == "inf" else float(theta_str)
    except ValueError:
        theta_val = float("inf")

    st.session_state["page2_assigned_fit"] = {"alpha": alpha_val, "beta": beta_val, "theta": theta_val}

    st.altair_chart(
        _beta_chart(alpha_val, beta_val, "Beta PDF (manual)"),
        use_container_width=True,
    )

    manual_name = st.text_input(
        "Parameter set name for manual values",
        value=st.session_state.get("last_saved_param_set", ""),
        key="manual_save_name",
    )
    if st.button("Save current parameters", key="save_manual_params"):
        saved_name = _save_param_set(
            manual_name or _next_param_name(_read_param_store()),
            alpha_val,
            beta_val,
            theta_val,
        )
        st.success(
            f"Saved '{saved_name}' with alpha={alpha_val:.6f}, beta={beta_val:.6f}, theta={theta_val}"
        )

# Saved sets tab
with saved_tab:
    st.subheader("Saved contamination parameter sets")
    saved = _read_param_store()
    if not saved:
        st.info("No saved parameter sets yet.")
    else:
        saved_keys = list(saved.keys())
        default_idx = 0
        last_saved = st.session_state.get("last_saved_param_set")
        if last_saved and last_saved in saved_keys:
            default_idx = saved_keys.index(last_saved)
        sel = st.selectbox("Select a saved set", saved_keys, index=default_idx)
        params = saved.get(sel, {})
        if params:
            alpha = float(params.get("alpha", FALLBACK_ALPHA))
            beta = float(params.get("beta", FALLBACK_BETA))
            theta = params.get("theta", FALLBACK_THETA)
            st.metric("Alpha", f"{alpha:.6f}")
            st.metric("Beta", f"{beta:.6f}")
            st.metric("Theta", f"{theta}")
            st.altair_chart(
                _beta_chart(alpha, beta, f"Beta PDF for {sel}"),
                use_container_width=True,
            )
            st.info("Sets are stored in tmp/contamination/contamination_parameter_sets.json.")

# ---- Footer ----
st.caption(
    "These parameters are injected into PoPS Border configuration so downstream pages use the updated contamination distribution."
)

st.divider()
nav_cols = st.columns(3)
with nav_cols[0]:
    if st.button(
        "Reset and Return Home",
        type="secondary",
        key="nav_reset_page2",
        help="Delete temporary files and restart from the home page",
    ):
        try:
            # Remove saved parameter sets and all tmp artifacts
            if PARAM_STORE.exists():
                PARAM_STORE.unlink()
            if TMP_DIR.exists():
                shutil.rmtree(TMP_DIR)
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            # Clear in-memory state
            st.session_state.pop("last_saved_param_set", None)
            st.session_state.pop("page2_assigned_fit", None)
            slippage_state["paths"] = create_default_paths()
            slippage_state["fit"] = None
            st.switch_page("frontend.py")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unable to reset temporary files: {exc}")
with nav_cols[1]:
    if st.button("Previous Page", type="primary", key="nav_back_page2"):
        st.switch_page("pages/1_Consignment_Generation.py")
with nav_cols[2]:
    if st.button("Next Page", type="primary", key="nav_forward_page4"):
        st.switch_page("pages/3_Inspection_Process.py")
