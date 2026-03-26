# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Optional

import altair as alt
import numpy as np
from gui.runtime_warnings import suppress_optional_dependency_warnings

suppress_optional_dependency_warnings()

import pandas as pd
import streamlit as st
from scipy.stats import betabinom

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.page_styles import (
    apply_shared_page_styles,
    render_labeled_help,
    render_metric_card,
    render_page_intro,
)
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
        x=alt.X("prevalence:Q", title="Contamination prevalence", axis=alt.Axis(format="%")),
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


def _save_param_set(name: str, alpha: float, beta: float, theta: float, sample_unit_rate: Optional[float] = None) -> str:
    store = _read_param_store()
    if not name:
        name = _next_param_name(store)
    entry = {"alpha": alpha, "beta": beta, "theta": theta}
    if sample_unit_rate is not None:
        entry["sample_unit_contamination_rate"] = sample_unit_rate
    store[name] = entry
    _write_param_store(store)
    st.session_state["last_saved_param_set"] = name
    return name


def calculate_beta_binomial_params_old(
        sample_unit_rate: float,
        concentration_input: float,
        n_trials: int = 100
) -> dict:
    # Transform input to concentration
    concentration = transform_input_to_concentration(concentration_input)

    # Clamp mean to avoid exact 0 or 1
    mean_clamped = np.clip(sample_unit_rate, 0.001, 0.999)

    # Calculate alpha and beta
    alpha = mean_clamped * concentration
    beta = (1 - mean_clamped) * concentration

    # Calculate rho and variance
    rho = 1.0 / (concentration + 1.0)
    variance = n_trials * mean_clamped * (1 - mean_clamped) * (1 + (n_trials - 1) * rho)

    return {
        'alpha': alpha,
        'beta': beta,
        'concentration': concentration,      # Already included
        'variance': variance,
        'rho': rho,
        'mean_clamped': mean_clamped        # Add this for convenience
    }


def calculate_beta_binomial_params(
        sample_unit_rate: float,
        concentration_input: float,
        n_trials: int = 100,
        confidence_level: float = 0.95
) -> dict:
    """
    Calculate beta-binomial parameters and summary statistics.
    """
    # Transform input to concentration
    concentration = transform_input_to_concentration(concentration_input)

    # Clamp mean to avoid exact 0 or 1
    mean_clamped = np.clip(sample_unit_rate, 0.001, 0.999)

    # Calculate alpha and beta
    alpha = mean_clamped * concentration
    beta = (1 - mean_clamped) * concentration

    # Calculate rho and variance
    rho = 1.0 / (concentration + 1.0)
    variance = n_trials * mean_clamped * (1 - mean_clamped) * (1 + (n_trials - 1) * rho)

    # Create Beta-Binomial distribution for CI calculation
    bb_dist = betabinom(n=n_trials, a=alpha, b=beta)

    # Calculate mean and confidence interval
    mean_value = n_trials * mean_clamped
    alpha_level = (1 - confidence_level) / 2
    lower_bound = bb_dist.ppf(alpha_level)
    upper_bound = bb_dist.ppf(1 - alpha_level)

    return {
        'alpha': alpha,
        'beta': beta,
        'concentration': concentration,
        'variance': variance,
        'rho': rho,
        'mean_clamped': mean_clamped,
        'mean': mean_value,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'confidence_level': confidence_level
    }


def transform_input_to_concentration(
        concentration_input: float,
        conc_min: float = 0.5,
        conc_max: float = 100.0
) -> float:
    """
    Transform user input (0-100%) to concentration parameter using log scale.

    Args:
        concentration_input: User input in range [0, 100]
        conc_min: Minimum concentration (default: 0.5)
        conc_max: Maximum concentration (default: 100.0)

    Returns:
        Concentration parameter in range [conc_min, conc_max]

    Mathematical justification:
        - Log scaling provides proportional changes: equal input intervals
          produce equal multiplicative changes in concentration
        - This aligns with how concentration affects variance (1/κ² relationship)
        - Provides better control in low-concentration regime where
          overdispersion effects are strongest
    """
    # Clamp input to valid range
    concentration_input = np.clip(concentration_input, 0.0, 100.0)

    # Logarithmic interpolation
    log_min = np.log(conc_min)
    log_max = np.log(conc_max)
    log_concentration = log_min + (concentration_input / 100.0) * (log_max - log_min)

    concentration = np.exp(log_concentration)

    return concentration


def transform_concentration_to_input(
        concentration: float,
        conc_min: float = 0.5,
        conc_max: float = 100.0
) -> float:
    """Invert the UI concentration transform back to the 0-100 control scale."""
    concentration = float(np.clip(concentration, conc_min, conc_max))
    log_min = np.log(conc_min)
    log_max = np.log(conc_max)
    return float(((np.log(concentration) - log_min) / (log_max - log_min)) * 100.0)


# ---- Page setup ----
init_state()
slippage_state = get_slippage_state()
paths = slippage_state["paths"]
render_sidebar_navigation()
apply_shared_page_styles()


st.warning(
    "**Test Deployment Notice: This is a test deployment with limited functionality and is under active development. "
    "Features may be incomplete and subject to change. Results have not been validated.**"
)
st.title("Page 2 - Contamination Fit")
render_page_intro(
    "Fit, assign, and manage contamination parameters. "
    "PIS upload and RBS selection live in the <i>Fit Contamination</i> tab. "
    "All outputs are written to <i>tmp/contamination</i>."
)


########### Notes on using Markdowns ############
#################################################

## Style tagging
# Tag	            Effect	        Example
# <em> or <i>	    Italic	        <em>text</em>
# <strong> or <b>	Bold	        <strong>text</strong>
# <u>	            Underline	    <u>text</u>
# <s> or <del>	    Strikethrough	<s>text</s>
# <mark>	        Highlight	    <mark>text</mark>
# <small>	        Smaller	        <small>text</small>
# <sup>	            Superscript	    x<sup>2</sup>
# <sub>	            Subscript	    H<sub>2</sub>O

#### CSS Properties Reference
# Property	        Effect	                Example Values
# font-size	        Tab text size	        14px, 20px, 28px
# font-weight	    Boldness	            400 (normal), 600 (semi-bold), 700 (bold)
# color	            Text color	            #000000, #1f77b4
# padding	        Space inside tab	    10px 20px (top/bottom left/right)
# letter-spacing	Space between letters	0.5px, 1px
# text-transform	Case	                uppercase, lowercase


# Multiple Lines via mark down
# st.markdown("""
#     <div style='font-size: 20px; color: #2c3e50; line-height: 0.5;'>
#         <p style='margin-bottom: 12px;'>
#             Fit, assign, and manage contamination parameters.
#             PIS upload and RBS selection live in the Fit tab.
#             All outputs are written to tmp/contamination.
#         </p>
#         <p style='margin-bottom: 0;'>
#             All outputs are written to tmp/contamination.
#         </p>
#     </div>
# """, unsafe_allow_html=True)

# st.caption(
#     "Fit, assign, and manage contamination parameters. PIS upload and RBS selection live in the Fit tab. "
#     "All outputs are written to tmp/contamination."
# )
















####################################################################
####################################################################
## Mark down and CSS customizable options

# st.markdown("""
#     <style>
#     /* ========== LABEL TEXT STYLING ========== */
#     .selectbox-label-text {
#         font-size: 28px;                    /* Text size */
#         font-weight: 700;                   /* Bold text */
#         color: #0d47a1;                     /* Dark blue color */
#         text-transform: capitalize;         /* Capitalize first letter */
#         letter-spacing: 0.5px;              /* Space between letters */
#         line-height: 1.4;                   /* Line height */
#         font-family: 'Arial', sans-serif;   /* Font family */
#         text-shadow: 1px 1px 2px rgba(0,0,0,0.1); /* Subtle shadow */
#     }
#
#     /* ========== INFO ICON STYLING ========== */
#     .info-icon {
#         display: inline-block;
#         width: 24px;                        /* Icon width */
#         height: 24px;                       /* Icon height */
#         border-radius: 50%;                 /* Circular shape */
#         background-color: #1976d2;          /* Blue background */
#         color: white;                       /* White text */
#         text-align: center;
#         line-height: 24px;                  /* Vertical centering */
#         font-size: 15px;                    /* Icon character size */
#         margin-left: 10px;                  /* Space from label */
#         cursor: help;                       /* Help cursor on hover */
#         vertical-align: middle;             /* Align with text */
#         border: 2px solid #0d47a1;          /* Border around icon */
#         box-shadow: 0 2px 4px rgba(0,0,0,0.15); /* Subtle shadow */
#         opacity: 0.9;                       /* Slight transparency */
#         position: relative;
#     }
#
#     /* Icon hover effect */
#     .info-icon:hover {
#         background-color: #0d47a1;          /* Darker on hover */
#         transform: scale(1.1);              /* Slightly larger */
#         opacity: 1;                         /* Full opacity */
#     }
#
#     /* ========== TOOLTIP BOX STYLING ========== */
#     .info-icon .tooltip {
#         visibility: hidden;
#         width: 350px;                       /* Tooltip width */
#         max-width: 90vw;                    /* Max width for mobile */
#         background-color: #263238;          /* Dark gray background */
#         color: #ffffff;                     /* White text */
#         text-align: left;                   /* Left-aligned text */
#         border-radius: 10px;                /* Rounded corners */
#         padding: 18px;                      /* Internal spacing */
#         position: absolute;
#         z-index: 999;                       /* Layer on top */
#         bottom: 140%;                       /* Position above icon */
#         left: 50%;
#         margin-left: -175px;                /* Center horizontally */
#         opacity: 0;                         /* Initially invisible */
#         transition: opacity 0.4s ease-in-out; /* Smooth fade */
#         font-size: 15px;                    /* Text size */
#         font-weight: 400;                   /* Normal weight */
#         line-height: 1.7;                   /* Comfortable line spacing */
#         box-shadow: 0 6px 12px rgba(0,0,0,0.4); /* Drop shadow */
#         border: 1px solid rgba(255,255,255,0.1); /* Subtle border */
#         font-family: 'Segoe UI', sans-serif; /* Font family */
#     }
#
#     /* ========== TOOLTIP ARROW ========== */
#     .info-icon .tooltip::after {
#         content: "";
#         position: absolute;
#         top: 100%;                          /* Position at bottom of tooltip */
#         left: 50%;
#         margin-left: -8px;                  /* Center the arrow */
#         border-width: 8px;                  /* Arrow size */
#         border-style: solid;
#         border-color: #263238 transparent transparent transparent; /* Arrow color */
#     }
#
#     /* Show tooltip on hover */
#     .info-icon:hover .tooltip {
#         visibility: visible;
#         opacity: 1;                         /* Fully visible */
#         transform: translateY(-5px);        /* Slide up slightly */
#     }
#
#     /* ========== SELECTBOX DROPDOWN STYLING ========== */
#     .stSelectbox div[data-baseweb="select"] > div {
#         font-size: 20px !important;         /* Dropdown text size */
#         font-weight: 500 !important;        /* Medium weight */
#         color: #1a237e !important;          /* Indigo text */
#         background-color: #f5f7fa !important; /* Light background */
#         border: 2px solid #1976d2 !important; /* Blue border */
#         border-radius: 8px !important;      /* Rounded corners */
#         padding: 12px 16px !important;      /* Internal padding */
#         box-shadow: 0 2px 4px rgba(0,0,0,0.08) !important; /* Subtle shadow */
#     }
#
#     /* Dropdown hover effect */
#     .stSelectbox div[data-baseweb="select"] > div:hover {
#         border-color: #0d47a1 !important;   /* Darker border on hover */
#         box-shadow: 0 4px 8px rgba(0,0,0,0.12) !important; /* Stronger shadow */
#     }
#
#     /* ========== DROPDOWN OPTIONS STYLING ========== */
#     .stSelectbox div[data-baseweb="select"] ul li {
#         font-size: 18px !important;
#         padding: 14px 18px !important;
#         font-weight: 500 !important;
#     }
#
#     /* Option hover effect */
#     .stSelectbox div[data-baseweb="select"] ul li:hover {
#         background-color: #e3f2fd !important; /* Light blue on hover */
#     }
#     </style>
# """, unsafe_allow_html=True)

####################################################################
####################################################################





saved_tab, fit_tab, assign_tab = st.tabs(
    ["Saved Parameter Sets", "Fit Contamination Using Data", "Assign Contamination Manually"]
)

# Fit contamination tab
with fit_tab:
    st.subheader("Fit beta-binomial contamination parameters")
    st.caption("Fits are based on the PIS action upload and RBS calculator selection in this tab.")

    st.subheader("PIS action data upload")
    rbs_candidates = sorted((Path("tmp") / "consignments").glob("*.csv"))
    rbs_source = st.radio(
        "Consignment data source",
        ["Use consignment data from Page 1", "Upload a new consignment data file"],
        index=0,
        key="page2_rbs_source",
    )
    if rbs_source == "Use consignment data from Page 1":
        if not rbs_candidates:
            st.warning("No consignment files found in tmp/consignments. Add one on Page 1 or upload a file below.")
        else:
            current_rbs = slippage_state["paths"].rbs_data
            default_idx = 0
            if current_rbs in rbs_candidates:
                default_idx = rbs_candidates.index(current_rbs)
            chosen_rbs = st.selectbox(
                "Consignment file",
                rbs_candidates,
                index=default_idx,
                format_func=lambda p: p.name,
            )
            set_paths(pis_data=chosen_rbs, rbs_data=chosen_rbs, synthetic_seed=chosen_rbs)
            paths = slippage_state["paths"]
    else:
        rbs_upload = st.file_uploader(
            "Upload consignment data CSV",
            type=["csv"],
            key="page2_rbs_upload",
            help="Upload a new consignment data file to use for contamination fitting on this page.",
        )
        if rbs_upload is not None:
            dest = CONTAM_DIR / "fit_rbs_data.csv"
            rbs_df = pd.read_csv(rbs_upload)
            dest.parent.mkdir(parents=True, exist_ok=True)
            rbs_df.to_csv(dest, index=False)
            set_paths(pis_data=dest, rbs_data=dest, synthetic_seed=dest)
            paths = slippage_state["paths"]
            st.success(f"Saved consignment data to {dest}")

    pis_df = None
    pis_preview = _load_preview(slippage_state["paths"].rbs_data)
    if pis_preview is not None:
        st.dataframe(pis_preview, use_container_width=True)
        try:
            pis_df = pd.read_csv(slippage_state["paths"].rbs_data)
        except Exception:  # pylint: disable=broad-except
            pis_df = None
    else:
        st.info("No consignment data loaded yet.")

    # Summary stats beneath the table (one-column flow)
    if pis_df is not None and not pis_df.empty:
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
        action_col = _find_col(pis_df, ["action"])

        n_rows = len(pis_df)
        unique_inspections = pis_df[ins_col].nunique() if ins_col else 0
        total_sampling = pis_df[samp_col].sum() if samp_col else 0
        total_plants = pis_df[plant_col].sum() if plant_col else 0
        action_ones = int((pis_df[action_col] == 1).sum()) if action_col else 0

        stats = st.columns(3)
        stats[0].metric("Rows", f"{n_rows}")
        stats[1].metric("Unique inspections", f"{unique_inspections}")
        stats[2].metric("Rows with action = 1", f"{action_ones}")
        stats2 = st.columns(1)
        stats2[0].metric("Total sampling units", f"{total_sampling:,}")

    fit_cols = st.columns(2)
    with fit_cols[0]:
        if st.button("Fit contamination parameters*", type="primary"):
            if paths.pis_data is None or paths.rbs_data is None:
                st.error("Upload PIS data and select an RBS file before fitting.")
            else:
                try:
                    fit, pis_df, rbs_df = fit_contamination_distribution(paths.pis_data, paths.rbs_data)
                    # Force theta to infinity per requirement
                    fit = ClarkeFit(alpha=fit.alpha, beta=fit.beta, theta=float("inf"), raw_result=fit.raw_result)
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
            _beta_chart(fit_to_show.alpha, fit_to_show.beta, "Beta-Binomial Probability Density Function"),
            use_container_width=True,
        )

    st.markdown("**Save fitted parameters**")
    name_input = st.text_input(
        "Parameter set name",
        value=st.session_state.get("last_saved_param_set", ""),
        key="fit_save_name",
    )
    can_save_fitted_parameters = fit_to_show is not None and bool(name_input.strip())
    if st.button("Save fitted parameters", key="save_fit_params", disabled=not can_save_fitted_parameters):
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

    st.caption("*Clark, R.G., Barnes, B. & Parsa, M. Clustered and Unclustered Group Testing for Biosecurity. JABES 29, 193–211 (2024). https://doi.org/10.1007/s13253-023-00566-x")

# Manual assignment tab
with assign_tab:

    render_labeled_help(
        "Choose How to Assign Contamination",
        "Options include specifying beta-binomial parameters directly or specifying a contamination rate at the lowest unit level.",
    )

    # Selectbox without label (since we added custom one above)
    mode = st.selectbox(
        "mode_select",  # Hidden label
        ["Specify Beta-Binomial Parameters (alpha and beta)", "Specify Contamination Rate (at lowest unit level)"],
        index=1,
        label_visibility="collapsed"  # Hide the default label
    )

    assigned_state = st.session_state.get(
        "page2_assigned_fit",
        {"alpha": FALLBACK_ALPHA, "beta": FALLBACK_BETA, "theta": FALLBACK_THETA, "sample_unit_rate": 0.01},
    )

    if mode == "Specify Contamination Rate (at lowest unit level)":
        #st.caption("Set a target mean contamination rate at the plant/sample-unit level and tune width via concentration.")
        default_conc = max(
            assigned_state.get("alpha", FALLBACK_ALPHA) + assigned_state.get("beta", FALLBACK_BETA),
            1e-6,
        )
        stored_mean = st.session_state.get("manual_mean_rate", float(assigned_state.get("sample_unit_rate", 0.01)))
        stored_mean_input = st.session_state.get("manual_mean_input_pct", float(stored_mean) * 100.0)
        pct_default = float(stored_mean_input)

        st.write("")
        render_labeled_help(
            "Average Contamination Rate (0%-100%)",
            "Enter the percentage of plants that are typically contaminated in your samples.",
        )

        pct_input = st.number_input(
            "pct",
            min_value=0.1,
            max_value=99.9,
            value=pct_default,
            step=0.05,
            format="%.1f",
            key="manual_mean_input",
            label_visibility="collapsed"
        )


        sample_unit_rate = pct_input / 100.0
        stored_conc = st.session_state.get(
            "manual_concentration",
            float(max(2.0, assigned_state.get("alpha", FALLBACK_ALPHA) + assigned_state.get("beta", FALLBACK_BETA))),
        )
        stored_concentration_input = st.session_state.get(
            "manual_concentration_input",
            transform_concentration_to_input(stored_conc),
        )

        st.write("")
        render_labeled_help(
            "% Confidence in Specified Contamination Rate",
            "(0% = No Confidence, 100% = Full Confidence)",
        )

        concentration_input = st.slider(
            "concentration",
            min_value=0.0,
            max_value=100.0,
            value=float(stored_concentration_input),
            step=0.5,
            key="manual_concentration_slider",
            label_visibility="collapsed"
        )

        n_trials = 100
        params = calculate_beta_binomial_params(sample_unit_rate, concentration_input, n_trials=n_trials)

        adj_alpha = params['alpha']
        adj_beta = params['beta']
        variance = params['variance']
        mean_clamped = params['mean_clamped']
        concentration = params['concentration']
        lb = params['lower_bound']
        ub = params['upper_bound']
        mean = params['mean']

        theta_val = float("inf")

        st.session_state["page2_assigned_fit"] = {
            "alpha": adj_alpha,
            "beta": adj_beta,
            "theta": theta_val,
            "sample_unit_rate": sample_unit_rate,
        }
        st.session_state["manual_concentration"] = concentration
        st.session_state["manual_concentration_input"] = concentration_input
        st.session_state["manual_mean_rate"] = sample_unit_rate
        st.session_state["manual_mean_input_pct"] = pct_input


        st.write("Contamination Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            render_metric_card(
                "Expected Average",
                f"{mean:.1f} out of {n_trials} plants",
                "Average number of contaminated plants expected within a sample of 100 plants.",
            )
        with col2:
            render_metric_card(
                "95% Range",
                f"{lb:.1f} - {ub:.1f}",
                "The typical range where 95% of observed contaminated plant counts will fall. This accounts for natural variability in the sampling process.",
            )
        with col3:
            render_metric_card(
                "Alpha",
                f"{adj_alpha:.6f}",
                "Alpha parameter of the beta-binomial distribution.",
            )
        with col4:
            render_metric_card(
                "Beta",
                f"{adj_beta:.6f}",
                "Beta parameter of the beta-binomial distribution.",
            )
        st.caption("*Theta is fixed to inf (no clustering) for manual contamination assignment.")

        st.write("")
        st.write("")
        st.altair_chart(
            _beta_chart(adj_alpha, adj_beta, "Beta-Binomial Probability Density Function"),
            use_container_width=True,
        )

        st.write("")
        st.write("")
        render_labeled_help(
            "Parameter Set Name",
            'File name to appear in the "Save Parameter Sets" associated with these set parameters',
        )

        manual_name = st.text_input(
            "Parameter set name for manual values",
            value=st.session_state.get("last_saved_param_set", ""),
            key="manual_save_name_sample_rate",
            label_visibility="collapsed"
        )
        can_save_manual_sample_rate = bool(manual_name.strip())
        if st.button(
            "Save current parameters",
            key="save_manual_params_sample_rate",
            disabled=not can_save_manual_sample_rate,
        ):
            saved_name = _save_param_set(
                manual_name or _next_param_name(_read_param_store()),
                adj_alpha,
                adj_beta,
                theta_val,
                sample_unit_rate,
            )
            st.success(
                f"Saved '{saved_name}' with alpha={adj_alpha:.6f}, beta={adj_beta:.6f}, "
                f" and sample unit rate={sample_unit_rate}"
            )
    else:
        st.caption("Adjust alpha/beta directly. Theta is fixed to infinity by default.")
        col_a, col_b, col_t = st.columns(3)
        alpha_val = col_a.number_input(
            "Alpha",
            min_value=0.000001,
            value=float(assigned_state["alpha"]),
            step=0.0005,
            format="%.4f",
        )
        beta_val = col_b.number_input(
            "Beta",
            min_value=0.000001,
            value=float(assigned_state["beta"]),
            step=0.0005,
            format="%.4f",
        )
        theta_str = col_t.text_input("Theta", value="inf")
        try:
            theta_val = float("inf") if theta_str.lower() == "inf" else float(theta_str)
        except ValueError:
            theta_val = float("inf")

        st.session_state["page2_assigned_fit"] = {
            "alpha": alpha_val,
            "beta": beta_val,
            "theta": theta_val,
            "sample_unit_rate": assigned_state.get("sample_unit_rate", 0.01),
        }

        st.altair_chart(
            _beta_chart(alpha_val, beta_val, "Beta-Binomial Probability Density Function"),
            use_container_width=True,
        )
        manual_name = st.text_input(
            "Parameter set name for manual values",
            value=st.session_state.get("last_saved_param_set", ""),
            key="manual_save_name_alpha_beta",
        )
        can_save_manual_alpha_beta = bool(manual_name.strip())
        if st.button(
            "Save current parameters",
            key="save_manual_params_alpha_beta",
            disabled=not can_save_manual_alpha_beta,
        ):
            saved_name = _save_param_set(
                manual_name or _next_param_name(_read_param_store()),
                alpha_val,
                beta_val,
                theta_val,
                assigned_state.get("sample_unit_rate", None),
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
            sample_unit_rate = params.get("sample_unit_contamination_rate")
            summary_cols = st.columns(4)
            with summary_cols[0]:
                render_metric_card("Alpha", f"{alpha:.6f}", "Alpha parameter of the beta-binomial distribution.")
            with summary_cols[1]:
                render_metric_card("Beta", f"{beta:.6f}", "Beta parameter of the beta-binomial distribution.")
            with summary_cols[2]:
                render_metric_card("Theta", f"{theta}", "Third parameter of the contamination model.")
            with summary_cols[3]:
                render_metric_card(
                    "Plant unit contamination rate",
                    f"{sample_unit_rate}" if sample_unit_rate is not None else "n/a",
                    "Saved plant unit contamination rate when the parameter set was created from the rate-based workflow.",
                )
            st.altair_chart(
                _beta_chart(alpha, beta, f"Beta-binomial PDF for {sel}"),
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
            st.session_state.clear()
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
