from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.slippage_pipeline import SyntheticOptions
from gui.slippage_ui import get_slippage_state, set_paths, set_synthetic_options


TMP_DIR = Path("tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)
MANUAL_PIS_PATH = TMP_DIR / "user_defined_pis_data.csv"
MANUAL_RBS_PATH = TMP_DIR / "user_defined_rbs_data.csv"


def _parse_list(text: str, fallback: List[str]) -> List[str]:
    cleaned = [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]
    return cleaned or fallback


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

st.title("Page 2 - Consignment Generation")
st.caption(
    "When you do not have historical PIS data, use this page to define consignments and create the "
    "two files PoPS Border needs (PIS records + RBS calculator rows)."
)
st.info("This workflow is for analysts with **no curated PIS/RBS data**. If you already have data, remain on Page 1.")

with st.form("manual_consignment_form"):
    c1, c2 = st.columns(2)
    num_consignments = c1.number_input("Number of consignments", min_value=1, max_value=2000, value=25, step=1)
    id_prefix = c2.text_input("Inspection number prefix", value="INS")

    origins_text = c1.text_input("Countries of origin (comma separated)", value="Japan, Netherlands, Mexico")
    ports_text = c2.text_input("Ports of entry (comma separated)", value="Miami PIS, Los Angeles PIS")
    pathways = st.multiselect("Pathways", ["Air", "Sea", "Land"], default=["Sea"])
    materials_text = st.text_input(
        "Propagative material types (comma separated)",
        value="Bulb, Corm, Rhizome, Tuberous Stem; Rooted Plant (including grafted)",
    )

    sample_units = c1.number_input("Sample units per consignment", min_value=1, max_value=5000, value=200, step=10)
    plants_per_sample = c2.number_input("Plants per sample unit", min_value=1, max_value=1000, value=5, step=1)
    contamination_rate = st.slider(
        "Assumed contamination prevalence (fraction of plants)",
        min_value=0.0,
        max_value=0.5,
        value=0.02,
        step=0.005,
    )
    st.caption("Detection level is fixed at 1% and confidence level at 80% for the generated RBS rows.")
    detection_level = 0.01
    confidence_level = 0.8
    default_boxes = max(1, int(sample_units * detection_level * confidence_level))
    required_boxes = st.number_input(
        "Required inspection units (boxes) per consignment",
        min_value=1,
        max_value=int(sample_units),
        value=default_boxes,
        step=1,
    )
    producer = st.text_input("Producer name", value="User Defined Producer")

    submitted = st.form_submit_button("Create PIS + RBS files", use_container_width=True)

if submitted:
    origins = _parse_list(origins_text, ["Japan"])
    ports = _parse_list(ports_text, ["Miami PIS"])
    materials = _parse_list(materials_text, ["Bulb, Corm, Rhizome, Tuberous Stem"])
    if not pathways:
        pathways = ["Sea"]

    seed_df = _build_seed_dataset(
        count=int(num_consignments),
        prefix=id_prefix.strip() or "INS",
        origins=origins,
        ports=ports,
        pathways=pathways,
        materials=materials,
        sample_units=int(sample_units),
        plants_per_sample=int(plants_per_sample),
        producer=producer.strip() or "User Defined Producer",
        contamination_rate=float(contamination_rate),
        detection_level=float(detection_level),
        confidence_level=float(confidence_level),
        required_boxes=int(required_boxes),
    )
    seed_df.to_csv(MANUAL_PIS_PATH, index=False)

    rbs_df = _build_rbs_dataset(seed_df)
    rbs_df.to_csv(MANUAL_RBS_PATH, index=False)

    set_paths(
        synthetic_seed=MANUAL_PIS_PATH,
        pis_data=MANUAL_PIS_PATH,
        rbs_data=MANUAL_RBS_PATH,
    )
    set_synthetic_options(SyntheticOptions(n_samples=int(num_consignments), sampling_method="sequential"))
    state["manual_seed_preview"] = seed_df.head(200)
    state["manual_rbs_preview"] = rbs_df.head(200)
    st.success(
        f"Generated {len(seed_df)} consignment rows and matching RBS calculator entries. "
        "Files stored in tmp/ have been wired into the pipeline."
    )

preview = state.get("manual_seed_preview")
if preview is not None:
    st.subheader("PIS seed dataset preview")
    st.dataframe(preview, use_container_width=True)
    st.download_button(
        "Download PIS dataset",
        data=preview.to_csv(index=False).encode("utf-8"),
        file_name="user_defined_pis_data.csv",
        use_container_width=True,
    )
    rbs_preview = state.get("manual_rbs_preview")
    if rbs_preview is not None:
        st.subheader("RBS calculator preview")
        st.dataframe(rbs_preview, use_container_width=True)
        st.download_button(
            "Download RBS dataset",
            data=rbs_preview.to_csv(index=False).encode("utf-8"),
            file_name="user_defined_rbs_data.csv",
            use_container_width=True,
        )
else:
    st.info("Create a seed dataset to preview and download the generated files.")

st.info("After generating consignments, proceed to **Page 3 - Contamination Fit** to update beta-binomial parameters.")

st.divider()
nav_cols = st.columns(2)
with nav_cols[0]:
    if st.button("Back to Page 1", type="primary", key="nav_back_page1"):
        st.switch_page("pages/1_Data_Ingest.py")
with nav_cols[1]:
    if st.button("Continue to Page 3", type="primary", key="nav_forward_pages3"):
        st.switch_page("pages/3_Contamination_Fit.py")
