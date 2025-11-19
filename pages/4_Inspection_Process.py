from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from gui.models import init_state, set_inspection
from gui.navigation import render_sidebar_navigation
from gui.slippage_ui import get_slippage_state, set_paths, set_scenario_dataframe


METHOD_OPTIONS = [
    ("Random search", "random"),
    ("Convenience search", "convenience"),
    ("Cluster search", "cluster"),
]

SAMPLE_STRATEGIES = ["proportion", "hypergeometric", "fixed_n", "all", "rbs"]
SELECTION_STRATEGIES = ["random", "convenience", "cluster"]


st.set_page_config(
    page_title="Inspection Process",
    page_icon=":mag_right:",
    layout="wide",
)
init_state()

state = get_slippage_state()
render_sidebar_navigation()
paths = state["paths"]
scenario_df = state["scenario_df"]


def _load_compliance_table(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


def _compliance_level_counts(series: pd.Series):
    if series is None:
        return {"Low": 0, "Medium": 0, "High": 0}
    normalized = series.astype(str).str.lower()
    return {
        "Low": normalized.str.contains("low").sum(),
        "Medium": normalized.str.contains("med").sum(),
        "High": normalized.str.contains("high").sum(),
    }


st.title("Page 4 - Inspection Process")
st.caption(
    "Ingest the compliance lookup table, configure low/medium/high types, "
    "review RBS parameters, and assign inspection strategies to each scenario."
)

with st.form("inspection_form"):
    cols = st.columns([1.2, 1, 1, 1])
    current_method_label = next(
        (label for label, value in METHOD_OPTIONS if value == st.session_state.inspection.method),
        METHOD_OPTIONS[0][0],
    )
    method_label = cols[0].selectbox(
        "Sampling method (descriptive)",
        [label for label, _ in METHOD_OPTIONS],
        index=[label for label, _ in METHOD_OPTIONS].index(current_method_label),
    )
    method = dict(METHOD_OPTIONS)[method_label]
    sample_units = cols[1].number_input(
        "Sampled units",
        1,
        2000,
        st.session_state.inspection.sample_units,
        step=1,
    )
    items_per_unit = cols[2].number_input(
        "Items per sampled unit",
        1,
        2000,
        st.session_state.inspection.sample_items_per_unit,
        step=1,
    )
    acceptance = cols[3].number_input(
        "Acceptance number",
        0,
        1000,
        st.session_state.inspection.acceptance_number,
        step=1,
        help="0 -> any detection triggers rejection.",
    )
    saved = st.form_submit_button("Save base inspection strategy", use_container_width=True)
    if saved:
        set_inspection(
            method=method,
            sample_units=int(sample_units),
            sample_items_per_unit=int(items_per_unit),
            acceptance_number=int(acceptance),
        )
        st.success("Inspection strategy saved.")

scenario_snapshot = st.session_state.scenario
overview_cols = st.columns(4)
overview_cols[0].metric("Total items", f"{scenario_snapshot.units_in_shipment * scenario_snapshot.items_per_unit:,}")
overview_cols[1].metric("Items sampled", f"{st.session_state.inspection.sample_units * st.session_state.inspection.sample_items_per_unit:,}")
sampling_fraction = 100 * (
    st.session_state.inspection.sample_units * st.session_state.inspection.sample_items_per_unit
) / max(1, scenario_snapshot.units_in_shipment * scenario_snapshot.items_per_unit)
overview_cols[2].metric("Sampling fraction", f"{sampling_fraction:.3f}%")
overview_cols[3].metric("Acceptance #", f"{st.session_state.inspection.acceptance_number}")

st.divider()
st.subheader("Compliance table")
st.caption("Upload or point to a compliance lookup table used to categorize consignments (Low / Medium / High).")

current_compliance_path = st.text_input("Compliance table path", str(paths.compliance_lookup))
if st.button("Update compliance path"):
    try:
        set_paths(compliance_lookup=Path(current_compliance_path))
        st.success("Compliance table path updated.")
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Unable to update compliance path: {exc}")

compliance_upload = st.file_uploader("Upload and replace compliance table", type=["csv"])
if compliance_upload is not None:
    try:
        new_compliance_df = pd.read_csv(compliance_upload)
        target_path = paths.data_dir / "uploaded_compliance_table.csv"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        new_compliance_df.to_csv(target_path, index=False)
        set_paths(compliance_lookup=target_path)
        st.success(f"Compliance table saved to {target_path}")
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Failed to save compliance table: {exc}")

compliance_df = _load_compliance_table(state["paths"].compliance_lookup)
if compliance_df is None:
    st.warning("Compliance table not found. Upload a CSV to proceed.")
else:
    st.dataframe(compliance_df, use_container_width=True, height=280)
    string_columns = [col for col in compliance_df.columns if compliance_df[col].dtype == "object"]
    selected_column = st.selectbox(
        "Column containing compliance type (Low / Medium / High)",
        options=string_columns or compliance_df.columns.tolist(),
    )
    counts = _compliance_level_counts(compliance_df[selected_column]) if selected_column else {"Low": 0, "Medium": 0, "High": 0}
    level_cols = st.columns(3)
    level_cols[0].metric("Low type rows", counts["Low"])
    level_cols[1].metric("Medium type rows", counts["Medium"])
    level_cols[2].metric("High type rows", counts["High"])

st.divider()
st.subheader("Scenario-specific inspection parameters")

if scenario_df.empty:
    st.info("The slippage scenario table is empty. Add scenarios on Page 2 first.")
else:
    options = [f"{idx}: {row.get('name', '(unnamed)')}" for idx, row in scenario_df.iterrows()]
    selected = st.selectbox("Select scenario", options, key="inspection_scenario_selector")
    selected_idx = int(selected.split(":")[0])
    current = scenario_df.loc[selected_idx]

    with st.form("slippage_inspection_form"):
        compliance_level = st.selectbox(
            "Compliance type",
            ["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index(current.get("inspection/compliance_level", "Medium"))
            if current.get("inspection/compliance_level") in ["Low", "Medium", "High"]
            else 1,
        )
        current_sample_strategy = str(current.get("inspection/sample_strategy", "") or "").strip().lower()
        sample_options = SAMPLE_STRATEGIES.copy()
        if current_sample_strategy and current_sample_strategy not in sample_options:
            sample_options.insert(0, current_sample_strategy)
        sample_strategy = st.selectbox(
            "Sample strategy",
            sample_options,
            index=sample_options.index(current_sample_strategy) if current_sample_strategy in sample_options else 0,
        )
        current_selection_strategy = str(current.get("inspection/selection_strategy", "") or "").strip().lower()
        selection_options = SELECTION_STRATEGIES.copy()
        if current_selection_strategy and current_selection_strategy not in selection_options:
            selection_options.insert(0, current_selection_strategy)
        selection_strategy = st.selectbox(
            "Selection strategy",
            selection_options,
            index=selection_options.index(current_selection_strategy) if current_selection_strategy in selection_options else 0,
        )
        within_box = st.number_input(
            "Within-box proportion",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=float(current.get("inspection/within_box_proportion", 0) or 0),
        )
        min_boxes = st.number_input(
            "Minimum boxes",
            min_value=0,
            max_value=1000,
            step=1,
            value=int(current.get("inspection/min_boxes", 0) or 0),
        )
        proportion_value = st.number_input(
            "Inspection proportion value",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=float(current.get("inspection/proportion/value", 0) or 0),
        )
        detection_level = st.number_input(
            "RBS hypergeometric detection level",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=float(current.get("inspection/hypergeometric/detection_level", 0) or 0),
        )
        hyper_sample = st.number_input(
            "RBS hypergeometric sample size",
            min_value=0,
            max_value=10000,
            step=1,
            value=int(current.get("inspection/hypergeometric/sample_size", 0) or 0),
        )
        hyper_population = st.number_input(
            "RBS hypergeometric population",
            min_value=0,
            max_value=1000000,
            step=10,
            value=int(current.get("inspection/hypergeometric/population", 0) or 0),
        )
        submit = st.form_submit_button("Update inspection parameters")

    if submit:
        scenario_df.at[selected_idx, "inspection/compliance_level"] = compliance_level
        scenario_df.at[selected_idx, "inspection/sample_strategy"] = sample_strategy
        scenario_df.at[selected_idx, "inspection/selection_strategy"] = selection_strategy
        scenario_df.at[selected_idx, "inspection/within_box_proportion"] = within_box
        scenario_df.at[selected_idx, "inspection/min_boxes"] = min_boxes
        scenario_df.at[selected_idx, "inspection/proportion/value"] = proportion_value
        scenario_df.at[selected_idx, "inspection/hypergeometric/detection_level"] = detection_level
        scenario_df.at[selected_idx, "inspection/hypergeometric/sample_size"] = hyper_sample
        scenario_df.at[selected_idx, "inspection/hypergeometric/population"] = hyper_population
        set_scenario_dataframe(scenario_df)
        st.success("Inspection parameters updated for this scenario.")

st.markdown("#### Current inspection summary across scenarios")
inspection_cols = [
    "name",
    "inspection/compliance_level",
    "inspection/sample_strategy",
    "inspection/proportion/value",
    "inspection/hypergeometric/detection_level",
]
available = [col for col in inspection_cols if col in scenario_df.columns]
st.dataframe(scenario_df[available], use_container_width=True)

st.divider()
nav_cols = st.columns(2)
with nav_cols[0]:
    if st.button("Back to Page 3", type="primary", key="nav_back_page3"):
        st.switch_page("pages/3_Contamination_Fit.py")
with nav_cols[1]:
    if st.button("Continue to Page 5", type="primary", key="nav_forward_page5"):
        st.switch_page("pages/5_Scenario_Experiments.py")

