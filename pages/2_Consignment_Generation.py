from pathlib import Path
import itertools
import re
import shutil
import yaml
from typing import List, Optional

import pandas as pd
import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.slippage_pipeline import SyntheticOptions, generate_synthetic_data, create_default_paths
from gui.slippage_ui import (
    get_slippage_state,
    set_paths,
    set_synthetic_options,
)


TMP_DIR = Path("tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)
CONSIGNMENT_ROOT = TMP_DIR / "consignments"
CONSIGNMENT_ROOT.mkdir(parents=True, exist_ok=True)


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


def _parse_list(text: str, fallback: List[str]) -> List[str]:
    cleaned = [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]
    return cleaned or fallback


def _load_preview(path: Optional[Path], rows: int = 50) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    try:
        return pd.read_csv(path).head(rows)
    except Exception:  # pylint: disable=broad-except
        return None


def _consignment_dir() -> Path:
    base = CONSIGNMENT_ROOT
    base.mkdir(parents=True, exist_ok=True)
    return base


def _consignment_paths() -> dict[str, Path]:
    base = _consignment_dir()
    base_name = state.get("consignment_base_name") or "consignment"
    return {
        "uploaded_pis": base / f"{base_name}_uploaded_pis_data.csv",
        "uploaded_rbs": base / f"{base_name}_uploaded_rbs_data.csv",
        "manual_pis": base / f"{base_name}_user_defined_pis_data.csv",
        "manual_rbs": base / f"{base_name}_user_defined_rbs_data.csv",
        "synthetic_output": base / f"{base_name}_synthetic_consignment_data.csv",
    }


def _build_seed_dataset(
    count: int,
    prefix: str,
    origins: List[str],
    ports: List[str],
    pathways: List[str],
    materials: List[str],
    sample_units: int,
    plants_per_sample: int,
    producer: str,
    contamination_rate: float,
    detection_level: float,
    confidence_level: float,
    required_boxes: int,
) -> pd.DataFrame:
    rows = []
    for idx in range(count):
        inspection_number = f"{prefix}-{idx+1:04d}"
        origin = origins[idx % len(origins)]
        port = ports[idx % len(ports)]
        pathway = pathways[idx % len(pathways)]
        material = materials[idx % len(materials)]
        total_sampling_units = sample_units
        total_plants = sample_units * plants_per_sample
        created_time = f"{(idx * 7) % 59:02d}:{(idx * 11) % 59:02d}.{(idx * 3) % 10}"
        total_plants_contaminated = int(round(total_plants * contamination_rate))
        rows.append(
            {
                "INSPECTION_NUMBER": inspection_number,
                "INSPECTION_ID": inspection_number,
                "INSPECTION_LOCATION_NAME": port,
                "PATHWAY": pathway,
                "COUNTRY_OF_ORIGIN_NAME": origin,
                "PROPAGATIVE_MATERIAL_TYPE": material,
                "TOTAL_SAMPLING_UNITS": total_sampling_units,
                "TOTAL_PLANT_QUANTITY": total_plants,
                "TOTAL_PLANTS_CONTAMINATED": total_plants_contaminated,
                "PRODUCER": producer,
                "CREATED_DATETIME": created_time,
                "IS_RBS": 1,
                "RBS_STATUS": "RBS",
                "SIMULATED_CONTAMINATION_RATE": contamination_rate,
                "DETECTION_LEVEL": detection_level,
                "CONFIDENCE_LEVEL": confidence_level,
                "REQUIRED_NUMBER_OF_BOXES": required_boxes,
                "action": 0,
            }
        )
    return pd.DataFrame(rows)


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
paths = state["paths"]
state.setdefault("consignment_source", "synthetic")
state.setdefault("consignment_base_name", "consignment")
synthetic_options: SyntheticOptions = state["synthetic_options"]
config = yaml.safe_load(Path("config.yml").read_text())
config_origins = config.get("consignment", {}).get("parameter_based", {}).get("origins", [])
config_ports = config.get("consignment", {}).get("parameter_based", {}).get("ports", [])
config_materials = config.get("consignment", {}).get("parameter_based", {}).get("flowers", [])
set_paths(synthetic_output=_consignment_paths()["synthetic_output"])

st.title("Page 1 - Consignment Generation")
st.caption(
    "Ingest curated PIS action data or define consignments from scratch. Both paths create the inputs needed for "
    "contamination fitting and simulation."
)
st.info(
    "Choose your workflow below: upload existing datasets and optionally generate synthetic consignments, "
    "or manually build consignments when no historical data exists. RBS uploads happen on Page 2."
)

ingest_tab, manual_tab = st.tabs(
    ["Upload or ingest PIS action data", "Define consignments manually"]
)

with ingest_tab:
    st.subheader("Consignment Parameters")

    st.info("Upload PIS action data and RBS on **Page 2 - Contamination Fit**. Use this page to generate or define consignments.")

    pis_df: Optional[pd.DataFrame] = state.get("pis_data")
    rbs_df: Optional[pd.DataFrame] = state.get("rbs_data")

    preview_cols = st.columns(2)
    with preview_cols[0]:
        if pis_df is not None and not pis_df.empty:
            st.subheader("PIS action data - inspection overview")
            st.metric("Total observations", f"{len(pis_df):,}")
            cat_cols = ["COUNTRY_OF_ORIGIN_NAME", "INSPECTION_LOCATION_NAME", "PROPAGATIVE_MATERIAL_TYPE"]
            action_col = "action" if "action" in pis_df.columns else None
            vis_cols = st.columns(3)
            for idx, col in enumerate(cat_cols):
                if col in pis_df.columns and action_col:
                    agg = pis_df.groupby(col)[action_col].sum().sort_values(ascending=False)
                    vis_cols[idx].bar_chart(agg.rename("Actions"))
            with st.expander("Preview PIS data"):
                st.dataframe(pis_df.head(25), use_container_width=True, height=250)
        else:
            st.info("Upload PIS data to view summary statistics.")

    with preview_cols[1]:
        st.subheader("RBS calculator data")
        st.info("RBS uploads are now handled on Page 2 - Contamination Fit.")

    source_choice = st.radio(
        "Consignment source for downstream analysis",
        [
            "Generate synthetic consignments from data",
            "Use historical consignments",
        ],
        index=0 if state["consignment_source"] == "synthetic" else 1,
        horizontal=True,
    )
    state["consignment_source"] = "synthetic" if source_choice.startswith("Generate") else "historical"

    if state["consignment_source"] == "historical" and (
        (pis_df is None or pis_df.empty) or (rbs_df is None or rbs_df.empty)
    ):
        st.warning("Upload both PIS and RBS data to rely on historical consignments.")

    current_pis_path = paths.pis_data
    current_rbs_path = paths.rbs_data
    pis_ready = current_pis_path is not None and Path(current_pis_path).exists()
    rbs_ready = current_rbs_path is not None and Path(current_rbs_path).exists()

    if state["consignment_source"] == "synthetic":
        st.divider()
        st.subheader("Synthetic consignment controls")
        col_syn = st.columns(2)
        syn_samples = col_syn[0].number_input(
            "Synthetic consignments",
            min_value=1,
            max_value=50000,
            value=max(1, int(synthetic_options.n_samples)),
            step=50,
        )
        syn_method = col_syn[1].selectbox(
            "Sampling method",
            ["naive", "sequential", "gmm", "gaussian_copula"],
            index=["naive", "sequential", "gmm", "gaussian_copula"].index(synthetic_options.sampling_method),
        )

        if syn_samples != synthetic_options.n_samples or syn_method != synthetic_options.sampling_method:
            set_synthetic_options(SyntheticOptions(n_samples=int(syn_samples), sampling_method=syn_method))
            st.caption("Synthetic generation options updated.")

        st.markdown(
            "Generate synthetic consignments for experimentation or rely on uploaded historical consignments. "
            "Synthetic generation will also re-fit the contamination distribution (beta-binomial)."
        )

        st.divider()
        if st.button("Generate synthetic PIS consignments", type="secondary", use_container_width=True):
            with st.spinner("Generating synthetic PIS consignments..."):
                try:
                    synthetic_df = generate_synthetic_data(
                        paths.synthetic_seed,
                        _consignment_paths()["synthetic_output"],
                        synthetic_options,
                    )
                    state["synthetic_data"] = synthetic_df
                    state["synthetic_preview"] = synthetic_df.head(10)
                    synthetic_path = _consignment_paths()["synthetic_output"]
                    st.success(f"Saved {len(synthetic_df):,} synthetic PIS records to {synthetic_path}.")
                    set_paths(pis_data=synthetic_path, synthetic_output=synthetic_path)
                except Exception as exc:  # pylint: disable=broad-except
                    st.error(f"Generation failed: {exc}")

        current_pis_path = _consignment_paths()["synthetic_output"]
        pis_ready = current_pis_path is not None and Path(current_pis_path).exists()

        syn_paths = state["paths"]
        syn_pis_preview = _load_preview(syn_paths.pis_data)
        syn_rbs_preview = _load_preview(syn_paths.rbs_data)
        with st.expander("Synthetic PIS/RBS previews", expanded=False):
            syn_cols = st.columns(2)
            if syn_pis_preview is not None:
                with syn_cols[0]:
                    st.subheader("Synthetic PIS preview")
                    st.dataframe(syn_pis_preview, use_container_width=True)
            if syn_rbs_preview is not None:
                with syn_cols[1]:
                    st.subheader("Synthetic RBS preview")
                    st.dataframe(syn_rbs_preview, use_container_width=True)
        st.caption("Synthetic generation produces PIS data. RBS must come from uploads or manual creation.")

    current_pis = current_pis_path
    current_rbs = current_rbs_path

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
        "Detection level is fixed at 1% and confidence level at 80% for generated RBS rows. "
        "Required inspection units are auto-computed."
    )
    if st.button(
        "Save consignment from inspection units",
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

        combined_seed = pd.concat([c["seed"] for c in state["manual_consignments"]], ignore_index=True)
        combined_rbs = pd.concat([c["rbs"] for c in state["manual_consignments"]], ignore_index=True)

        paths_map = _consignment_paths()
        combined_seed.to_csv(paths_map["manual_pis"], index=False)
        combined_rbs.to_csv(paths_map["manual_rbs"], index=False)

        set_paths(
            synthetic_seed=paths_map["manual_pis"],
            pis_data=paths_map["manual_pis"],
            rbs_data=paths_map["manual_rbs"],
            synthetic_output=paths_map["synthetic_output"],
        )
        set_synthetic_options(SyntheticOptions(n_samples=int(len(combined_seed)), sampling_method="sequential"))
        state["manual_seed_preview"] = combined_seed.head(200)
        state["manual_rbs_preview"] = combined_rbs.head(200)
        st.success(
            f"Saved consignment '{consignment_uid}' with {len(rows)} inspection units. "
            f"Files stored in tmp/consignments with base name '{state['consignment_base_name']}' have been updated."
        )

    if state.get("manual_consignments"):
        st.subheader("Current consignments")
        summary_rows = [
            {"Consignment": c["name"], "Inspection units": len(c["seed"])}
            for c in state["manual_consignments"]
        ]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    preview = state.get("manual_seed_preview")
    rbs_preview = state.get("manual_rbs_preview")
    with st.expander("PIS/RBS previews", expanded=False):
        cols_prev = st.columns(2)
        if preview is not None:
            with cols_prev[0]:
                st.subheader("PIS action seed dataset preview")
                st.dataframe(preview, use_container_width=True)
                st.download_button(
                    "Download PIS action dataset",
                    data=preview.to_csv(index=False).encode("utf-8"),
                    file_name="user_defined_pis_action_data.csv",
                    use_container_width=True,
                )
        if rbs_preview is not None:
            with cols_prev[1]:
                st.subheader("RBS calculator preview")
                st.dataframe(rbs_preview, use_container_width=True)
                st.download_button(
                    "Download RBS dataset",
                    data=rbs_preview.to_csv(index=False).encode("utf-8"),
                    file_name="user_defined_rbs_data.csv",
                    use_container_width=True,
                )
        if preview is None and rbs_preview is None:
            st.info("Create a seed dataset to preview and download the generated files.")

current_pis = state["paths"].pis_data
current_rbs = state["paths"].rbs_data

st.text_input(
    "Consignment file base name",
    value=state["consignment_base_name"],
    key="consignment_base_name",
    help="Used to name PIS/RBS/synthetic files in tmp/consignments (e.g., <name>_uploaded_pis_data.csv).",
)
# refresh paths to use current base name
set_paths(synthetic_output=_consignment_paths()["synthetic_output"])

# refresh current paths after any base-name change
current_pis = state["paths"].pis_data
current_rbs = state["paths"].rbs_data

if st.button("Generate consignment files with this name", type="secondary", use_container_width=True):
    paths_map = _consignment_paths()
    if current_pis is None or current_rbs is None:
        st.warning("No PIS/RBS data available. Upload or create consignments first.")
    else:
        try:
            dest_pis = paths_map["uploaded_pis"]
            dest_rbs = paths_map["uploaded_rbs"]
            dest_pis.parent.mkdir(parents=True, exist_ok=True)
            if Path(current_pis).resolve() != dest_pis.resolve():
                shutil.copy(current_pis, dest_pis)
            if Path(current_rbs).resolve() != dest_rbs.resolve():
                shutil.copy(current_rbs, dest_rbs)
            set_paths(pis_data=dest_pis, rbs_data=dest_rbs, synthetic_seed=dest_pis)
            st.success(
                f"Saved consignment files with base '{state['consignment_base_name']}' to tmp/consignments."
            )
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unable to save consignment files: {exc}")
    st.caption("Current files point to: PIS -> {state['paths'].pis_data}, RBS -> {state['paths'].rbs_data}")

st.info("After preparing consignments, go to **Page 2 - Contamination Fit** to update beta-binomial parameters.")

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
            state["paths"] = create_default_paths()
            state["pis_data"] = None
            state["pis_preview"] = None
            state["rbs_data"] = None
            state["rbs_preview"] = None
            state["synthetic_data"] = None
            state["synthetic_preview"] = None
            state["manual_units"] = []
            state["manual_consignments"] = []
            st.success("Temporary files cleared. Returning to Home...")
            st.switch_page("frontend.py")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unable to reset temporary files: {exc}")
with nav_cols[1]:
    if st.button("Back to Home", type="primary", key="nav_home_page2"):
        st.switch_page("frontend.py")
with nav_cols[2]:
    if st.button(
        "Confirm upload and continue to next page",
        type="primary",
        key="nav_forward_page2",
        disabled=not (current_pis and current_rbs),
    ):
        set_paths(
            pis_data=current_pis,
            rbs_data=current_rbs,
            synthetic_seed=paths.synthetic_seed or current_pis,
        )
        st.switch_page("pages/3_Contamination_Fit.py")
