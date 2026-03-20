# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

from pathlib import Path
import itertools
import re
import shutil
import yaml
from typing import List, Optional

import altair as alt
import numpy as np
from gui.runtime_warnings import suppress_optional_dependency_warnings

suppress_optional_dependency_warnings()

import pandas as pd
import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.page_styles import apply_shared_page_styles, render_page_intro
from gui.slippage_pipeline import SyntheticOptions, generate_synthetic_data
from gui.slippage_ui import (
    get_slippage_state,
    set_paths,
    set_synthetic_options,
)


TMP_DIR = Path("tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)
CONSIGNMENT_ROOT = TMP_DIR / "consignments"
CONSIGNMENT_ROOT.mkdir(parents=True, exist_ok=True)


def _consignment_dir() -> Path:
    base = CONSIGNMENT_ROOT
    base.mkdir(parents=True, exist_ok=True)
    return base


def _consignment_paths(base_name: Optional[str] = None) -> dict[str, Path]:
    base = _consignment_dir()
    base_name = base_name or st.session_state.get("consignment_base_name", "consignment") or "consignment"
    return {
        "uploaded_rbs": base / f"{base_name}.csv",
        "manual_rbs": base / f"{base_name}.csv",
    }


def _unique_path(path: Path) -> Path:
    """Return a non-conflicting path by appending an incrementing suffix if needed."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for i in itertools.count(1):
        candidate = path.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate


def _is_tmp_consignment_path(path: Optional[Path]) -> bool:
    if path is None:
        return False
    try:
        resolved = Path(path).resolve()
        consignment_root = CONSIGNMENT_ROOT.resolve()
        return str(resolved).startswith(str(consignment_root))
    except Exception:
        return False


def _save_rbs_to_tmp(
    current_rbs: Optional[Path],
    pending_manual_rbs: Optional[pd.DataFrame],
    pending_upload_rbs: Optional[pd.DataFrame] = None,
    *,
    base_name: Optional[str] = None,
    producer_grouping_path: Optional[Path] = None,
) -> tuple[bool, str]:
    paths_map = _consignment_paths(base_name)
    base_name = base_name or st.session_state.get("consignment_base_name", "consignment") or "consignment"
    if current_rbs is None and pending_manual_rbs is None and pending_upload_rbs is None:
        return False, "No RBS data available. Upload a calculator file or create consignments manually first."
    try:
        state = get_slippage_state()
        cons_source = state.get("consignment_source", "synthetic")

        if cons_source == "synthetic":
            # Always generate a new synthetic file into a unique output
            dest_rbs = _unique_path(paths_map["uploaded_rbs"])
            dest_rbs.parent.mkdir(parents=True, exist_ok=True)
            if pending_upload_rbs is not None:
                seed_path = _consignment_dir() / f"{base_name}_seed_rbs.csv"
                pending_upload_rbs.to_csv(seed_path, index=False)
                state["pending_rbs_upload"] = None
            elif current_rbs and Path(current_rbs).exists():
                seed_path = Path(current_rbs)
            elif paths_map["uploaded_rbs"].exists():
                seed_path = paths_map["uploaded_rbs"]
            else:
                return False, "No RBS seed available for synthetic generation. Upload/select an RBS file first."

            options: SyntheticOptions = state.get("synthetic_options", SyntheticOptions())
            synth_df = generate_synthetic_data(
                seed_path,
                dest_rbs,
                options,
                producer_grouping_path=producer_grouping_path,
            )
            state["rbs_preview"] = synth_df.head(10)
            set_paths(rbs_data=dest_rbs, synthetic_seed=seed_path)
            return True, f"Generated and saved RBS file with base '{base_name}' to tmp/consignments."

        # Historical/manual: write exactly what is provided
        dest_rbs = _unique_path(paths_map["uploaded_rbs"])
        dest_rbs.parent.mkdir(parents=True, exist_ok=True)
        if pending_upload_rbs is not None:
            pending_upload_rbs.to_csv(dest_rbs, index=False)
            state["rbs_preview"] = pending_upload_rbs.head(10)
            state["pending_rbs_upload"] = None
            set_paths(rbs_data=dest_rbs)
            return True, f"Saved RBS file with base '{base_name}' to tmp/consignments."
        if pending_manual_rbs is not None:
            pending_manual_rbs.to_csv(dest_rbs, index=False)
            state["rbs_preview"] = pending_manual_rbs.head(10)
            set_paths(rbs_data=dest_rbs)
            return True, f"Saved RBS file with base '{base_name}' to tmp/consignments."
        if current_rbs is not None and Path(current_rbs).exists():
            if Path(current_rbs).resolve() != dest_rbs.resolve():
                shutil.copy(current_rbs, dest_rbs)
            state["rbs_preview"] = pd.read_csv(dest_rbs).head(10)
            set_paths(rbs_data=dest_rbs)
            return True, f"Saved RBS file with base '{base_name}' to tmp/consignments."

        return False, "No RBS source file found."

    except Exception as exc:  # pylint: disable=broad-except
        return False, f"Unable to save consignment files: {exc}"


def _save_historical_rbs(
    current_rbs: Optional[Path],
    pending_upload_rbs: Optional[pd.DataFrame],
    *,
    base_name: Optional[str] = None,
) -> tuple[bool, str]:
    """Persist uploaded historical RBS data to tmp/consignments without generation."""
    base_name = base_name or st.session_state.get("consignment_base_name", "Historical") or "Historical"
    paths_map = _consignment_paths(base_name)
    dest_rbs = _unique_path(paths_map["uploaded_rbs"])
    dest_rbs.parent.mkdir(parents=True, exist_ok=True)

    try:
        if pending_upload_rbs is not None and not pending_upload_rbs.empty:
            pending_upload_rbs.to_csv(dest_rbs, index=False)
            state = get_slippage_state()
            state["rbs_preview"] = pending_upload_rbs.head(10)
            state["pending_rbs_upload"] = None
            set_paths(rbs_data=dest_rbs)
            return True, f"Saved uploaded RBS file with base '{base_name}' to tmp/consignments."

        if current_rbs is not None and Path(current_rbs).exists():
            if Path(current_rbs).resolve() != dest_rbs.resolve():
                shutil.copy(current_rbs, dest_rbs)
            state = get_slippage_state()
            state["rbs_preview"] = pd.read_csv(dest_rbs).head(10)
            set_paths(rbs_data=dest_rbs)
            return True, f"Saved RBS file with base '{base_name}' to tmp/consignments."

        return False, "No RBS source file found."
    except Exception as exc:  # pylint: disable=broad-except
        return False, f"Unable to save consignment files: {exc}"


def _save_manual_rbs(
    pending_manual_rbs: Optional[pd.DataFrame],
    *,
    base_name: Optional[str] = None,
) -> tuple[bool, str]:
    """Persist a manual RBS dataset to a unique path under tmp/consignments."""
    if pending_manual_rbs is None or pending_manual_rbs.empty:
        return False, "No manual RBS data to save."
    try:
        paths_map = _consignment_paths(base_name)
        dest_rbs = _unique_path(paths_map["manual_rbs"])
        dest_rbs.parent.mkdir(parents=True, exist_ok=True)
        pending_manual_rbs.to_csv(dest_rbs, index=False)
        state = get_slippage_state()
        state["rbs_preview"] = pending_manual_rbs.head(10)
        set_paths(rbs_data=dest_rbs)
        return True, f"Saved manual RBS file with base '{base_name or st.session_state.get('consignment_base_name', 'consignment')}' to tmp/consignments."
    except Exception as exc:  # pylint: disable=broad-except
        return False, f"Unable to save manual RBS file: {exc}"




def _build_rbs_dataset(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in seed_df.iterrows():
        rows.append(
            {
                "INSPECTION_ID": row["INSPECTION_ID"],
                "INSPECTION_NUMBER": row["INSPECTION_NUMBER"],
                "INSPECTION_LOCATION_NAME": row["INSPECTION_LOCATION_NAME"],
                "COUNTRY_OF_ORIGIN_NAME": row["COUNTRY_OF_ORIGIN_NAME"],
                "PATHWAY": row["PATHWAY"],
                "PROPAGATIVE_MATERIAL_TYPE": row["PROPAGATIVE_MATERIAL_TYPE"],
                "PRODUCER_NAME": row["PRODUCER"],
                "TOTAL_SAMPLING_UNITS": row["TOTAL_SAMPLING_UNITS"],
                "TOTAL_PLANT_QUANTITY": row["TOTAL_PLANT_QUANTITY"],
                "CONFIDENCE_LEVEL": row["CONFIDENCE_LEVEL"],
                "DETECTION_LEVEL": row["DETECTION_LEVEL"],
                "REQUIRED_NUMBER_OF_BOXES": row["REQUIRED_NUMBER_OF_BOXES"],
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(page_title="Consignment Generation", layout="wide")
init_state()

state = get_slippage_state()
render_sidebar_navigation()
apply_shared_page_styles()
paths = state["paths"]
state.setdefault("consignment_source", "synthetic")
state.setdefault("consignment_base_name", "consignment")
state.setdefault("pending_rbs_upload", None)
state.setdefault("use_custom_producer_grouping", False)
default_producer_grouping = Path("data_input/producer_grouping.csv")
if "producer_grouping_path" not in state:
    state["producer_grouping_path"] = default_producer_grouping if default_producer_grouping.exists() else None
# Normalize current RBS references for later save buttons
_state_rbs = state.get("rbs_data")
current_rbs = None
if isinstance(_state_rbs, (str, Path)):
    current_rbs = Path(_state_rbs)
elif isinstance(state["paths"].rbs_data, (str, Path)):
    current_rbs = Path(state["paths"].rbs_data)

pending_manual_rbs = state.get("pending_manual_rbs") if isinstance(state.get("pending_manual_rbs"), pd.DataFrame) else None
synthetic_options: SyntheticOptions = state["synthetic_options"]
config = yaml.safe_load(Path("config.yml").read_text())
config_origins = config.get("consignment", {}).get("parameter_based", {}).get("origins", [])
config_ports = config.get("consignment", {}).get("parameter_based", {}).get("ports", [])
config_materials = config.get("consignment", {}).get("parameter_based", {}).get("flowers", [])

st.warning(
    "**Test Deployment Notice: This is a test deployment with limited functionality and is under active development. "
    "Features may be incomplete and subject to change. Results have not been validated.**"
)
st.title("Page 1 - Consignment Generation")
render_page_intro(
    "Ingest curated consignment data or define consignments from scratch. "
    "Use the <i>Generate consignments based on data</i> tab to upload data and create synthetic consignments. "
    "Generated outputs are written to <i>tmp/consignments</i>."
)

ingest_tab, manual_tab, saved_tab, producer_grouping_tab = st.tabs(
    [
        "Generate consignments based on data",
        "Define consignments manually",
        "Saved consignments",
        "Producer grouping",
    ]
)

with ingest_tab:

    pis_df: Optional[pd.DataFrame] = state.get("pis_data")
    rbs_df: Optional[pd.DataFrame] = state.get("pending_rbs_upload")

    st.subheader("Upload Actual Consignment Data")
    rbs_upload = st.file_uploader("RBS calculator CSV", type=["csv"], key="rbs_upload_ingest")
    if rbs_upload is not None:
        rbs_df = pd.read_csv(rbs_upload)
        state["pending_rbs_upload"] = rbs_df
        st.success(f"Loaded {len(rbs_df):,} RBS records. Save below to persist to tmp/consignments.")
    if rbs_df is not None and not rbs_df.empty:
        st.dataframe(rbs_df.head(25), use_container_width=True, height=300)
        # Inline summary (mirrors Page 3 layout: table + stats together)
        metrics_top = st.columns(4)
        if "INSPECTION_NUMBER" in rbs_df.columns:
            metrics_top[0].metric("Consignments", f"{rbs_df['INSPECTION_NUMBER'].nunique():,}")
        if "PATHWAY" in rbs_df.columns:
            metrics_top[1].metric("Pathways", f"{rbs_df['PATHWAY'].nunique():,}")
        if "INSPECTION_LOCATION_NAME" in rbs_df.columns:
            metrics_top[2].metric("Locations", f"{rbs_df['INSPECTION_LOCATION_NAME'].nunique():,}")
        if "COUNTRY_OF_ORIGIN_NAME" in rbs_df.columns:
            metrics_top[3].metric("Countries", f"{rbs_df['COUNTRY_OF_ORIGIN_NAME'].nunique():,}")

        totals_cards = st.columns(2)
        if "TOTAL_PLANT_QUANTITY" in rbs_df.columns:
            totals_cards[0].metric("Total plant units", f"{int(rbs_df['TOTAL_PLANT_QUANTITY'].sum()):,}")
        if "TOTAL_SAMPLING_UNITS" in rbs_df.columns:
            totals_cards[1].metric("Total sampling units", f"{int(rbs_df['TOTAL_SAMPLING_UNITS'].sum()):,}")
        st.info("Save or generate to view rich plots in the Saved consignments tab.")
    else:
        st.info("Upload RBS calculator data here.")

    source_choice = st.radio(
        "Consignment source for downstream analysis",
        [
            "Generate synthetic consignments from data",
            "Use historical consignments",
        ],
        index=0 if state["consignment_source"] == "synthetic" else 1,
    )
    state["consignment_source"] = "synthetic" if source_choice.startswith("Generate") else "historical"

    # Base name input before actions, with defaults per mode
    default_base = "Generated" if state["consignment_source"] == "synthetic" else "Historical"
    # If the stored value is one of the old defaults, realign it to the current mode before rendering the widget
    if st.session_state.get("consignment_base_name") in ("Generated", "Historical", "Generated_Historical"):
        st.session_state["consignment_base_name"] = default_base
    current_base = st.session_state.get("consignment_base_name", default_base) or default_base
    st.text_input(
        "Consignment input file base name",
        value=current_base,
        key="consignment_base_name",
        help="Used to name RBS files in tmp/consignments (e.g., <name>.csv).",
    )
    state["consignment_base_name"] = st.session_state.get("consignment_base_name", current_base) or default_base

    if state["consignment_source"] == "synthetic":
        st.markdown("### Synthetic consignment generation")
        gen_cols = st.columns(2)
        n_samples = gen_cols[0].number_input("Number of consignments to generate", min_value=1, max_value=10000, value=20, step=10)
        method_dict = {
            "multinomial sequential": "sequential",
            "gaussian mixture": "gmm",
        }
        method_selection = gen_cols[1].selectbox(
            "Sampling method",
            options=list(method_dict),
            index=0,
        )
        method = method_dict[method_selection]
        state["use_custom_producer_grouping"] = st.checkbox(
            "Use producer grouping CSV during synthetic generation",
            value=bool(state.get("use_custom_producer_grouping", False)),
            key="use_custom_producer_grouping",
            help="When enabled, synthetic generation will load producer grouping from the selected CSV on the Producer grouping tab.",
        )
        has_tmp_saved_consignments = any(CONSIGNMENT_ROOT.glob("*.csv"))
        has_synthetic_seed = (
            (state.get("pending_rbs_upload") is not None and not state.get("pending_rbs_upload").empty)
            or has_tmp_saved_consignments
        )
        if st.button(
            "Generate synthetic consignments",
            type="primary",
            use_container_width=True,
            disabled=not has_synthetic_seed,
        ):
            set_synthetic_options(SyntheticOptions(n_samples=int(n_samples), sampling_method=method))
            set_paths(synthetic_seed=_consignment_paths()["uploaded_rbs"])
            producer_grouping_path = None
            if state.get("use_custom_producer_grouping"):
                producer_grouping_path = state.get("producer_grouping_path")
            ok, msg = _save_rbs_to_tmp(
                current_rbs,
                pending_manual_rbs,
                state.get("pending_rbs_upload"),
                producer_grouping_path=producer_grouping_path,
            )
            if ok:
                if state["paths"].rbs_data:
                    current_rbs = Path(state["paths"].rbs_data)
                st.success(f"Generated and saved synthetic consignments: {msg}")
            else:
                st.warning(msg)
    else:
        if state["consignment_source"] == "historical" and (
            (pis_df is None or pis_df.empty) or (rbs_df is None or rbs_df.empty)
        ):
            st.warning("Upload both PIS action and RBS data on **Page 2 - Contamination Fit** to rely on historical consignments.")
        can_save_uploaded_consignments = (
            (state.get("pending_rbs_upload") is not None and not state.get("pending_rbs_upload").empty)
            or any(CONSIGNMENT_ROOT.glob("*.csv"))
        )
        if st.button(
            "Save uploaded consignments",
            type="primary",
            key="save_consignment_ingest",
            disabled=not can_save_uploaded_consignments,
        ):
            ok, msg = _save_historical_rbs(current_rbs, state.get("pending_rbs_upload"), base_name=current_base)
            if ok:
                if state["paths"].rbs_data:
                    current_rbs = Path(state["paths"].rbs_data)
                st.success(msg)
            else:
                st.warning(msg)


    current_pis_path = paths.pis_data
    current_rbs_path = paths.rbs_data
    pis_ready = current_pis_path is not None and Path(current_pis_path).exists()
    rbs_ready = current_rbs_path is not None and Path(current_rbs_path).exists()

with manual_tab:
    st.subheader("Define consignments one by one")
    st.info(
        "Add inspection units first (one port, one origin, one material per unit), then bundle them into a consignment."
    )
    state.setdefault("manual_consignments", [])
    state.setdefault("manual_units", [])
    consignment_name = st.text_input(
        "Consignment name/ID (unique per consignment)",
        value=f"CONS-{len(state['manual_consignments'])+1:03d}",
    )

    with st.form("inspection_unit_form"):
        c1, c2 = st.columns(2)
        port = c1.selectbox(
            "Port of entry",
            options=config_ports or ["Miami PIS"],
            index=0,
        )
        origin = c2.selectbox(
            "Country of origin",
            options=config_origins or ["Japan"],
            index=0,
        )
        material = st.selectbox(
            "Propagative material type",
            options=config_materials or ["Bulb, Corm, Rhizome, Tuberous Stem; Rooted Plant (including grafted)"],
            index=0,
        )
        pathway = st.selectbox("Pathway", ["Air", "Sea", "Land"], index=1)
        sample_units = c1.number_input(
            "Sample units (per inspection unit)", min_value=1, max_value=5000, value=200, step=10
        )
        plants_per_sample = c2.number_input("Plants per sample unit", min_value=1, max_value=1000, value=5, step=1)
        producer = st.text_input("Producer name", value="User Defined Producer")
        add_unit = st.form_submit_button("Add inspection unit", type="secondary", use_container_width=True)

    if add_unit:
        state["manual_units"].append(
            {
                "port": port.strip(),
                "origin": origin.strip(),
                "material": material.strip(),
                "pathway": pathway,
                "sample_units": int(sample_units),
                "plants_per_sample": int(plants_per_sample),
                "producer": producer.strip() or "User Defined Producer",
            }
        )
        st.success(f"Added inspection unit: {port} / {origin} / {material}")

    if state["manual_units"]:
        st.subheader(f"Inspection units in {consignment_name or 'current consignment'}")
        st.dataframe(pd.DataFrame(state["manual_units"]), use_container_width=True)

    st.markdown("---")
    detection_level = 0.01
    confidence_level = 0.8
    st.caption(
        "Required inspection units are auto-computed."
    )
    if st.button(
        "Build consignment from inspection units above",
        type="secondary",
        use_container_width=True,
        disabled=not state["manual_units"],
    ):
        consignment_uid = consignment_name or f"CONS-{len(state['manual_consignments'])+1:03d}"
        rows = []
        for idx, unit in enumerate(state["manual_units"]):
            total_sampling_units = int(unit["sample_units"])
            total_plants = total_sampling_units * int(unit["plants_per_sample"])
            required_boxes = max(1, int(total_sampling_units * detection_level * confidence_level))
            rows.append(
                {
                    "INSPECTION_NUMBER": consignment_uid,
                    "INSPECTION_ID": consignment_uid,
                    "INSPECTION_LOCATION_NAME": unit["port"],
                    "PATHWAY": unit["pathway"],
                    "COUNTRY_OF_ORIGIN_NAME": unit["origin"],
                    "PROPAGATIVE_MATERIAL_TYPE": unit["material"],
                    "TOTAL_SAMPLING_UNITS": total_sampling_units,
                    "TOTAL_PLANT_QUANTITY": total_plants,
                    "TOTAL_PLANTS_CONTAMINATED": int(round(total_plants * 0.02)),
                    "PRODUCER": unit["producer"],
                    "CREATED_DATETIME": "00:00.0",
                    "IS_RBS": 1,
                    "RBS_STATUS": "RBS",
                    "SIMULATED_CONTAMINATION_RATE": 0.02,
                    "DETECTION_LEVEL": detection_level,
                    "CONFIDENCE_LEVEL": confidence_level,
                    "REQUIRED_NUMBER_OF_BOXES": int(required_boxes),
                    "action": 0,
                }
            )

        seed_df = pd.DataFrame(rows)
        rbs_df = _build_rbs_dataset(seed_df)

        state["manual_consignments"].append(
            {
                "name": consignment_uid,
                "seed": seed_df,
                "rbs": rbs_df,
            }
        )
        state["manual_units"] = []

        combined_rbs = pd.concat([c["rbs"] for c in state["manual_consignments"]], ignore_index=True)

        state["manual_rbs_preview"] = combined_rbs.head(200)
        state["pending_manual_rbs"] = combined_rbs
        st.success(
            f"Saved consignment '{consignment_uid}' with {len(rows)} inspection units. "
            "Use the generation button below to write the RBS file to tmp."
        )

    if state.get("manual_consignments"):
        st.subheader("Current consignments")
        summary_rows = [
            {"Consignment": c["name"], "Inspection units": len(c["seed"])}
            for c in state["manual_consignments"]
        ]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    rbs_preview = state.get("manual_rbs_preview")
    with st.expander("Synthetic RBS preview", expanded=False):
        if rbs_preview is not None:
            st.dataframe(rbs_preview, use_container_width=True)
            st.download_button(
                "Download RBS dataset",
                data=rbs_preview.to_csv(index=False).encode("utf-8"),
                file_name="user_defined_rbs_data.csv",
                use_container_width=True,
            )
        else:
            st.info("Create a seed dataset to preview and download the generated RBS file.")

    manual_base = st.text_input(
        "Consignment input file base name (manual)",
        value=st.session_state.get("consignment_base_name_manual", "Manual"),
        key="consignment_base_name_manual",
        help="Used to name RBS files in tmp/consignments (e.g., <name>.csv).",
    ) or "Manual"
    if st.button("Save manual consignments", type="primary", key="save_consignment_manual"):
        ok, msg = _save_manual_rbs(pending_manual_rbs, base_name=manual_base)
        if ok:
            st.success(msg)
        else:
            st.warning(msg)

# Saved consignments tab
with saved_tab:
    st.subheader("Saved consignments")
    saved_files = sorted(CONSIGNMENT_ROOT.glob("*.csv"))
    refresh = st.button("Refresh saved consignments", type="secondary")
    if refresh:
        st.rerun()
    if not saved_files:
        st.info("No consignment files saved yet in tmp/consignments.")
    else:
        sel = st.selectbox("Select a saved consignment file", saved_files, format_func=lambda p: p.name)
        st.caption(f"Location: {sel}")
        try:
            full_df = pd.read_csv(sel)
            st.dataframe(full_df, use_container_width=True, height=500)
            # Summary under the table (single column)
            st.markdown("### Summary statistics")
            stats_cols = st.columns(4)
            if "INSPECTION_NUMBER" in full_df.columns:
                stats_cols[0].metric("Consignments", f"{full_df['INSPECTION_NUMBER'].nunique():,}")
            if "PATHWAY" in full_df.columns:
                stats_cols[1].metric("Pathways", f"{full_df['PATHWAY'].nunique():,}")
            if "INSPECTION_LOCATION_NAME" in full_df.columns:
                stats_cols[2].metric("Locations", f"{full_df['INSPECTION_LOCATION_NAME'].nunique():,}")
            if "COUNTRY_OF_ORIGIN_NAME" in full_df.columns:
                stats_cols[3].metric("Countries", f"{full_df['COUNTRY_OF_ORIGIN_NAME'].nunique():,}")
            totals_cols = st.columns(2)
            if "TOTAL_PLANT_QUANTITY" in full_df.columns:
                totals_cols[0].metric("Total plant units", f"{int(full_df['TOTAL_PLANT_QUANTITY'].sum()):,}")
            if "TOTAL_SAMPLING_UNITS" in full_df.columns:
                totals_cols[1].metric("Total sampling units", f"{int(full_df['TOTAL_SAMPLING_UNITS'].sum()):,}")

            st.markdown("### Distributions")
            plots_two_col = st.columns(2)
            with plots_two_col[0]:
                plot_subcols = st.columns(3)
                if "COUNTRY_OF_ORIGIN_NAME" in full_df.columns:
                    plot_subcols[0].markdown("**Top origins**")
                    plot_subcols[0].bar_chart(full_df["COUNTRY_OF_ORIGIN_NAME"].value_counts().head(10).rename("Count"))
                if "INSPECTION_LOCATION_NAME" in full_df.columns:
                    plot_subcols[1].markdown("**Top inspection locations**")
                    plot_subcols[1].bar_chart(full_df["INSPECTION_LOCATION_NAME"].value_counts().head(10).rename("Count"))
                if "PROPAGATIVE_MATERIAL_TYPE" in full_df.columns:
                    plot_subcols[2].markdown("**Top material types**")
                    plot_subcols[2].bar_chart(full_df["PROPAGATIVE_MATERIAL_TYPE"].value_counts().head(10).rename("Count"))
            with plots_two_col[1]:
                if "TOTAL_PLANT_QUANTITY" in full_df.columns and "TOTAL_SAMPLING_UNITS" in full_df.columns:
                    st.markdown("**Plant units vs sampling units (frequency)**")
                    quantities = full_df["TOTAL_PLANT_QUANTITY"].dropna().to_numpy()
                    sampling_units = full_df["TOTAL_SAMPLING_UNITS"].dropna().to_numpy()
                    if quantities.size > 0 and sampling_units.size == quantities.size and sampling_units.size > 0:
                        q_min, q_max = float(quantities.min()), float(quantities.max())
                        s_min, s_max = float(sampling_units.min()), float(sampling_units.max())
                        q_bins = np.linspace(q_min, q_max, num=21) if q_min != q_max else np.array([q_min, q_max + 1])
                        s_bins = np.linspace(s_min, s_max, num=11) if s_min != s_max else np.array([s_min, s_max + 1])
                        heat, q_edges, s_edges = np.histogram2d(quantities, sampling_units, bins=[q_bins, s_bins])
                        heat_df = pd.DataFrame(
                            {
                                "plant_bin_start": np.repeat(q_edges[:-1], len(s_edges) - 1),
                                "plant_bin_end": np.repeat(q_edges[1:], len(s_edges) - 1),
                                "sample_bin_start": np.tile(s_edges[:-1], len(q_edges) - 1),
                                "sample_bin_end": np.tile(s_edges[1:], len(q_edges) - 1),
                                "frequency": heat.flatten(),
                            }
                        )
                        chart = (
                            alt.Chart(heat_df)
                            .mark_rect()
                            .encode(
                                x=alt.X(
                                    "plant_bin_start:Q",
                                    bin=alt.Bin(binned=True, step=float(q_bins[1] - q_bins[0])),
                                    title="Plant units (bin start)",
                                ),
                                x2="plant_bin_end:Q",
                                y=alt.Y(
                                    "sample_bin_start:Q",
                                    bin=alt.Bin(binned=True, step=float(s_bins[1] - s_bins[0])),
                                    title="Sampling units (bin start)",
                                ),
                                y2="sample_bin_end:Q",
                                color=alt.Color("frequency:Q", title="Frequency", scale=alt.Scale(scheme="blues")),
                            )
                        )
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("Need both TOTAL_PLANT_QUANTITY and TOTAL_SAMPLING_UNITS to render the heat map.")
                else:
                    st.info("Need both TOTAL_PLANT_QUANTITY and TOTAL_SAMPLING_UNITS to render the heat map.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unable to preview file: {exc}")

with producer_grouping_tab:
    st.subheader("Producer grouping")
    st.caption("Optional producer grouping used during synthetic consignment generation on Page 1.")

    if default_producer_grouping.exists():
        st.caption(f"Default file available: {default_producer_grouping}")
    else:
        st.info("No default producer grouping file found in data_input.")

    producer_grouping_upload = st.file_uploader(
        "Upload custom producer grouping CSV",
        type=["csv"],
        key="producer_grouping_upload",
        help="Expected columns include PRODUCER_NAME and grouping.",
    )
    if producer_grouping_upload is not None:
        producer_grouping_path = TMP_DIR / "producer_grouping.csv"
        producer_grouping_df = pd.read_csv(producer_grouping_upload)
        producer_grouping_df.to_csv(producer_grouping_path, index=False)
        state["producer_grouping_path"] = producer_grouping_path
        st.success(f"Saved custom producer grouping to {producer_grouping_path}")

    if state.get("producer_grouping_path"):
        current_grouping_path = Path(state["producer_grouping_path"])
        st.caption(f"Current producer grouping: {current_grouping_path}")
        try:
            producer_grouping_df = pd.read_csv(current_grouping_path)
            st.dataframe(producer_grouping_df.head(50), use_container_width=True, height=320)
            st.metric("Rows", f"{len(producer_grouping_df):,}")
        except Exception as exc:  # pylint: disable=broad-except
            st.warning(f"Unable to preview producer grouping file: {exc}")

pending_manual_rbs = state.get("pending_manual_rbs")

st.divider()
nav_cols = st.columns(3)
with nav_cols[0]:
    if st.button(
        "Reset and Return Home",
        type="secondary",
        key="nav_reset_tmp",
        help="Delete temporary files and restart from the home page",
    ):
        try:
            if TMP_DIR.exists():
                shutil.rmtree(TMP_DIR)
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            # Initialize a fresh copy of config.yml into tmp for downstream use
            src_cfg = Path("data_input/config.yml")
            if src_cfg.exists():
                dst_cfg = TMP_DIR / "config.yml"
                shutil.copy(src_cfg, dst_cfg)
            st.session_state.clear()
            st.success("Temporary files cleared. Returning to Home...")
            st.switch_page("frontend.py")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unable to reset temporary files: {exc}")
with nav_cols[1]:
    if st.button("Previous Page", type="primary", key="nav_home_page2"):
        st.switch_page("frontend.py")
with nav_cols[2]:
    if st.button(
        "Next Page",
        type="primary",
        key="nav_forward_page2",
        disabled=not current_rbs,
    ):
        set_paths(
            rbs_data=current_rbs,
        )
        st.switch_page("pages/2_Contamination_Fit.py")
