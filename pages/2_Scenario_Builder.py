import pandas as pd
import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.slippage_ui import get_slippage_state, set_paths, set_scenario_dataframe


st.set_page_config(page_title="Simple Consignment Builder", layout="wide")
init_state()

state = get_slippage_state()
render_sidebar_navigation()
scenario_df = state["scenario_df"]
paths = state["paths"]

st.title("Page 2 - User-defined Consignments")
st.caption("Select this page if you do not have ingested data and want to build a simple scenario table manually.")

with st.form("consignment_form"):
    col1, col2 = st.columns(2)
    prefix = col1.text_input("Scenario name prefix", value="custom-scenario")
    num_rows = col2.number_input("Number of consignments to generate", min_value=1, max_value=50, value=3)
    consignment_label = col1.text_input("Consignment label", value="Generated Consignment")
    inspection_label = col2.text_input("Inspection label", value="Generated Inspection")

    generation_method = col1.selectbox("Generation method", ["RBS", "PIS", "Synthetic"])
    file_type = col2.selectbox("Input file type", ["RBS", "PIS", "Synthetic"])

    units_in_shipment = col1.number_input("Units per consignment", min_value=1, max_value=50000, value=1200, step=10)
    items_per_unit = col2.number_input("Items per unit", min_value=1, max_value=10000, value=100, step=5)

    contamination_unit = st.selectbox("Contamination unit", ["plant", "unit", "item"])
    distribution = st.selectbox("Contamination distribution", ["beta-binomial", "beta", "fixed"])
    contamination_value = st.number_input("Contamination value / mean prevalence", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    arrangement = st.selectbox("Arrangement pattern", ["random", "clustered", "edge-loaded"])

    sample_strategy = st.selectbox("Inspection sample strategy", ["rbs", "simple_random", "systematic"])
    inspection_unit = st.selectbox("Inspection unit", ["boxes", "crates", "pallets"])
    proportion_value = st.number_input("Inspection proportion", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    detection_level = st.number_input("RBS hypergeometric detection level", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
    within_box_proportion = st.number_input("Within-box proportion", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
    min_boxes = st.number_input("Minimum boxes sampled", min_value=0, max_value=1000, value=0, step=1)
    selection_strategy = st.text_input("Selection strategy", value="random")
    compliance_level = st.selectbox("Compliance level", ["Low", "Medium", "High"])

    submitted = st.form_submit_button("Generate scenario table", use_container_width=True)

if submitted:
    rows = []
    for idx in range(int(num_rows)):
        name = f"{prefix}-{idx+1}"
        row = {
            "name": name,
            "consignment name": f"{consignment_label} {idx+1}",
            "inspection name": inspection_label,
            "consignment/generation_method": generation_method,
            "consignment/input_file/file_type": file_type,
            "contamination/contamination_unit": contamination_unit,
            "contamination/contamination_rate/distribution": distribution,
            "contamination/contamination_rate/value": contamination_value,
            "contamination/arrangement": arrangement,
            "consignment/units_in_shipment": units_in_shipment,
            "consignment/items_per_unit": items_per_unit,
            "inspection/unit": inspection_unit,
            "inspection/min_boxes": min_boxes,
            "inspection/sample_strategy": sample_strategy,
            "inspection/proportion/value": proportion_value,
            "inspection/hypergeometric/detection_level": detection_level,
            "inspection/selection_strategy": selection_strategy,
            "inspection/within_box_proportion": within_box_proportion,
            "inspection/compliance_level": compliance_level,
        }
        rows.append(row)
    new_df = pd.DataFrame(rows)
    set_scenario_dataframe(new_df)
    target = paths.data_dir / "generated_scenarios.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(target, index=False)
    set_paths(scenario_table=target)
    scenario_df = new_df
    st.success(f"Generated {len(new_df)} scenario rows and saved to {target}.")

if scenario_df.empty:
    st.info("No scenarios defined yet. Use the form above to create synthetic consignments.")
else:
    st.subheader("Generated scenario table")
    metrics = st.columns(3)
    metrics[0].metric("Scenario rows", len(scenario_df))
    if "inspection/proportion/value" in scenario_df.columns:
        avg_prop = pd.to_numeric(scenario_df["inspection/proportion/value"], errors="coerce").mean()
        metrics[1].metric("Average inspection proportion", f"{avg_prop:.3f}")
    else:
        metrics[1].metric("Average inspection proportion", "n/a")
    if "inspection/compliance_level" in scenario_df.columns:
        compliance_values = (
            scenario_df["inspection/compliance_level"].dropna().astype(str).unique().tolist()
        )
        metrics[2].metric(
            "Compliance levels represented",
            ", ".join(sorted(compliance_values)) if compliance_values else "n/a",
        )
    else:
        metrics[2].metric("Compliance levels represented", "n/a")
    st.dataframe(scenario_df, use_container_width=True, height=360)
    st.download_button(
        "Download scenario table (CSV)",
        data=scenario_df.to_csv(index=False).encode("utf-8"),
        file_name="generated_scenarios.csv",
        use_container_width=True,
    )
