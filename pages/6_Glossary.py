# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.page_styles import apply_shared_page_styles, render_page_intro


GLOSSARY_TERMS = [
    {
        "group": "Consignments",
        "term": "Consignment",
        "definition": "A named bundle of inspection units saved as one input file for downstream pages.",
    },
    {
        "group": "Consignments",
        "term": "Inspection unit",
        "definition": "One unit with a single port, origin, material type, pathway, and sampling quantity.",
    },
    {
        "group": "Consignments",
        "term": "Consignment name/ID",
        "definition": "The unique label used to group inspection units into one consignment.",
    },
    {
        "group": "Consignments",
        "term": "Synthetic consignments",
        "definition": "Consignments generated from an uploaded seed file instead of being entered manually.",
    },
    {
        "group": "Consignments",
        "term": "Historical consignments",
        "definition": "Uploaded consignment records preserved as-is for reuse in later pages.",
    },
    {
        "group": "Consignments",
        "term": "Sampling method",
        "definition": "The algorithm used to generate synthetic consignments from source data.",
    },
    {
        "group": "Consignments",
        "term": "Producer grouping",
        "definition": "An optional CSV that groups producers so synthetic generation can preserve producer structure.",
    },
    {
        "group": "Consignments",
        "term": "Port of entry",
        "definition": "The inspection location assigned to an inspection unit.",
    },
    {
        "group": "Consignments",
        "term": "Country of origin",
        "definition": "The source country assigned to an inspection unit.",
    },
    {
        "group": "Consignments",
        "term": "Propagative material type",
        "definition": "The plant material category assigned to an inspection unit.",
    },
    {
        "group": "Consignments",
        "term": "Pathway",
        "definition": "The transport route used for an inspection unit, such as air, sea, or land.",
    },
    {
        "group": "Consignments",
        "term": "Sample units (per inspection unit)",
        "definition": "The number of sample units contained in one inspection unit.",
    },
    {
        "group": "Consignments",
        "term": "Plants per sample unit",
        "definition": "How many plants each sample unit represents.",
    },
    {
        "group": "Consignments",
        "term": "RBS dataset",
        "definition": "The saved CSV output used by downstream pages for consignment, policy, and scenario workflows.",
    },
    {
        "group": "Contamination",
        "term": "Contamination fit",
        "definition": "The workflow that estimates contamination parameters from selected consignment data.",
    },
    {
        "group": "Contamination",
        "term": "Consignment data source",
        "definition": "The file source used to fit contamination parameters on Page 2.",
    },
    {
        "group": "Contamination",
        "term": "Parameter set name",
        "definition": "The name assigned to a saved contamination parameter set.",
    },
    {
        "group": "Contamination",
        "term": "Average contamination rate",
        "definition": "The contamination rate expressed as a percentage.",
    },
    {
        "group": "Contamination",
        "term": "Alpha",
        "definition": "A beta-binomial shape parameter that helps control contamination probability.",
    },
    {
        "group": "Contamination",
        "term": "Beta",
        "definition": "The beta-binomial shape parameter paired with alpha.",
    },
    {
        "group": "Contamination",
        "term": "Theta",
        "definition": "A clustering parameter for the contamination model; in manual assignment it is fixed to infinity.",
    },
    {
        "group": "Contamination",
        "term": "Beta-binomial",
        "definition": "The probability distribution used in the app to model contamination counts.",
    },
    {
        "group": "Contamination",
        "term": "Beta-binomial PDF",
        "definition": "The chart that visualizes the fitted or saved beta-binomial contamination distribution.",
    },
    {
        "group": "Contamination",
        "term": "Saved contamination parameter sets",
        "definition": "Stored contamination definitions written to the temporary JSON parameter store.",
    },
    {
        "group": "Compliance",
        "term": "RBS compliance policy",
        "definition": "A saved policy file that maps feature combinations to compliance behavior used downstream.",
    },
    {
        "group": "Compliance",
        "term": "Compliance level upload",
        "definition": "The CSV that defines which compliance category applies to each policy combination.",
    },
    {
        "group": "Compliance",
        "term": "Detection/confidence mapping",
        "definition": "The CSV that maps compliance categories to detection and confidence values.",
    },
    {
        "group": "Compliance",
        "term": "Feature columns",
        "definition": "Columns from the source data used to build compliance-policy rows.",
    },
    {
        "group": "Compliance",
        "term": "Compliance level",
        "definition": "The policy category assigned to generated rows, typically Low, Medium, or High.",
    },
    {
        "group": "Compliance",
        "term": "Detection level",
        "definition": "A numeric value used by the inspection logic to represent detection strength.",
    },
    {
        "group": "Compliance",
        "term": "Confidence level",
        "definition": "A numeric value used by the inspection logic to represent confidence.",
    },
    {
        "group": "Compliance",
        "term": "Values for <column>",
        "definition": "The allowed values selected for a chosen feature column when manually building a policy.",
    },
    {
        "group": "Scenarios And Experiments",
        "term": "Experiment package",
        "definition": "A saved bundle of scenario rows and copied input files stored under tmp/experiments.",
    },
    {
        "group": "Scenarios And Experiments",
        "term": "Experiment set",
        "definition": "The named folder that contains one saved experiment package.",
    },
    {
        "group": "Scenarios And Experiments",
        "term": "Scenario table",
        "definition": "The CSV file containing the scenario rows for a saved experiment package.",
    },
    {
        "group": "Scenarios And Experiments",
        "term": "Scenario row",
        "definition": "One runnable configuration for a simulation.",
    },
    {
        "group": "Scenarios And Experiments",
        "term": "Scenario label",
        "definition": "The unique name used to identify a scenario row and its results.",
    },
    {
        "group": "Scenarios And Experiments",
        "term": "Contamination parameter set",
        "definition": "The saved contamination settings selected for a scenario or run.",
    },
    {
        "group": "Scenarios And Experiments",
        "term": "Consignment (RBS) file",
        "definition": "The consignment CSV used as the input file for a scenario.",
    },
    {
        "group": "Scenarios And Experiments",
        "term": "Files available",
        "definition": "The inventory of selectable files currently available for building scenarios.",
    },
    {
        "group": "Simulation",
        "term": "Execution options",
        "definition": "The sidebar section where you choose replications and the experiment package to run.",
    },
    {
        "group": "Simulation",
        "term": "Simulation replications",
        "definition": "The number of times each scenario is rerun to estimate variation.",
    },
    {
        "group": "Simulation",
        "term": "Run experiment",
        "definition": "The action that starts the simulation pipeline for the selected experiment package.",
    },
    {
        "group": "Simulation",
        "term": "Run details",
        "definition": "The collapsed table showing the scenario setup that was executed.",
    },
    {
        "group": "Simulation",
        "term": "Run summary",
        "definition": "The KPI section summarizing scenarios, replications, shipments, inspections, and totals.",
    },
    {
        "group": "Simulation",
        "term": "Shipments per replication",
        "definition": "The number of consignments processed in each replication.",
    },
    {
        "group": "Simulation",
        "term": "Simulation summary view",
        "definition": "The control that switches the results display between mean, median, or one replication.",
    },
    {
        "group": "Simulation Results",
        "term": "Slippage",
        "definition": "In this app, contaminated units that evade detection rather than a generic loss metric.",
    },
    {
        "group": "Simulation Results",
        "term": "Slippage Level",
        "definition": "Results showing missed contaminated units at the consignment, inspection, sample, and plant levels.",
    },
    {
        "group": "Simulation Results",
        "term": "Action Level",
        "definition": "Results showing intercepted versus slipped outcomes by level.",
    },
    {
        "group": "Simulation Results",
        "term": "Contamination Level",
        "definition": "Results showing contaminated versus clean counts and percentages by level.",
    },
    {
        "group": "Simulation Results",
        "term": "Inspection Workload Level",
        "definition": "Results showing inspection workload and completion metrics.",
    },
    {
        "group": "Simulation Results",
        "term": "Intercepted",
        "definition": "Contaminated units detected by the inspection process at the reported level.",
    },
    {
        "group": "Simulation Results",
        "term": "Slipped",
        "definition": "Contaminated units not detected by the inspection process at the reported level.",
    },
    {
        "group": "App Notes",
        "term": "RBS",
        "definition": "An app-specific acronym used throughout the consignment, compliance, and scenario workflow. The code uses the acronym but does not expand it.",
    },
    {
        "group": "App Notes",
        "term": "PIS",
        "definition": "An app-specific acronym used for input and workflow labels. The code uses the acronym but does not expand it.",
    },
    {
        "group": "App Notes",
        "term": "Slippage UI state",
        "definition": "The shared session-state container that tracks paths, options, results, and loaded data across pages.",
    },
]

_GLOSSARY_STYLES = """
<style>
.glossary-summary {
    border: 1px solid rgba(31, 119, 180, 0.18);
    border-radius: 14px;
    background: linear-gradient(180deg, rgba(31,119,180,0.06), rgba(31,119,180,0.02));
    padding: 16px 18px;
    margin-bottom: 18px;
}

.glossary-summary-title {
    font-size: 18px;
    font-weight: 700;
    color: #1f77b4;
    margin-bottom: 6px;
}

.glossary-summary-text {
    font-size: 16px;
    color: #2c3e50;
    line-height: 1.55;
}

.glossary-card {
    border: 1px solid rgba(31, 119, 180, 0.18);
    border-radius: 14px;
    background: #ffffff;
    padding: 16px 18px;
    margin-bottom: 16px;
    box-shadow: 0 6px 16px rgba(31, 119, 180, 0.06);
    min-height: 176px;
}

.glossary-badge {
    display: inline-block;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #1f77b4;
    background: rgba(31, 119, 180, 0.10);
    border: 1px solid rgba(31, 119, 180, 0.20);
    border-radius: 999px;
    padding: 4px 10px;
    margin-bottom: 12px;
}

.glossary-term {
    font-size: 20px;
    font-weight: 700;
    color: #1f77b4;
    margin: 0 0 10px 0;
    line-height: 1.25;
}

.glossary-definition {
    font-size: 16px;
    color: #2c3e50;
    line-height: 1.6;
    margin: 0;
}
</style>
"""


def _render_summary(total_terms: int) -> None:
    st.markdown(
        f"""
        <div class="glossary-summary">
            <div class="glossary-summary-title">Glossary Overview</div>
            <div class="glossary-summary-text">
                {total_terms} terms shown. Entries are listed alphabetically and tagged with the app section where they are mainly used.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_glossary_card(item: dict) -> None:
    st.markdown(
        f"""
        <div class="glossary-card">
            <div class="glossary-badge">{item["group"]}</div>
            <p class="glossary-term">{item["term"]}</p>
            <p class="glossary-definition">{item["definition"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Glossary",
    page_icon=":book:",
    layout="wide",
)

init_state()
render_sidebar_navigation()
apply_shared_page_styles()
st.markdown(_GLOSSARY_STYLES, unsafe_allow_html=True)

st.warning(
    "**Test Deployment Notice: This is a test deployment with limited functionality and is under active development. "
    "Features may be incomplete and subject to change. Results have not been validated.**"
)
st.title("Glossary")
render_page_intro(
    "Reference definitions for terminology used across the app. "
    "These entries were compiled from the current page labels, controls, and results sections."
)

filter_cols = st.columns([2, 1])
with filter_cols[0]:
    search_text = st.text_input("Search glossary", value="", help="Filter glossary terms by name or definition.")
with filter_cols[1]:
    section_options = ["All sections"] + sorted({item["group"] for item in GLOSSARY_TERMS})
    selected_section = st.selectbox("Section", section_options, index=0)

query = search_text.strip().lower()

filtered = [
    item
    for item in GLOSSARY_TERMS
    if (selected_section == "All sections" or item["group"] == selected_section)
    and (not query or query in item["term"].lower() or query in item["definition"].lower() or query in item["group"].lower())
]

_render_summary(len(filtered))

if not filtered:
    st.info("No glossary terms match the current search.")
else:
    sorted_terms = sorted(filtered, key=lambda entry: entry["term"].lower())
    left_col, right_col = st.columns(2, gap="large")
    for idx, item in enumerate(sorted_terms):
        with (left_col if idx % 2 == 0 else right_col):
            _render_glossary_card(item)

st.divider()
nav_cols = st.columns(3)
with nav_cols[0]:
    if st.button("Reset and Return Home", type="secondary", key="nav_reset_page6"):
        st.session_state.clear()
        st.switch_page("frontend.py")
with nav_cols[1]:
    if st.button("Previous Page", type="primary", key="nav_back_page6"):
        st.switch_page("pages/5_Run_Simulation.py")
with nav_cols[2]:
    if st.button("Finish and Return Home", type="primary", key="nav_finish_page6"):
        st.switch_page("frontend.py")
