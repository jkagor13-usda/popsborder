# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

import streamlit as st

from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.page_styles import apply_shared_page_styles, render_page_intro


GLOSSARY_TERMS = [
    {
        "group": "Consignments",
        "term": "Consignment",
        "definition": "A named bundle of inspection units; also called a shipment.",
    },
    {
        "group": "Consignments",
        "term": "Inspection unit",
        "definition": "The single lowest, readily-distinguishable taxon, cultivar, or variety that is clearly defined as being from one source and in similar condition on the invoice, packing list, or phytosanitary certificate.",
    },
    {
        "group": "Consignments",
        "term": "Consignment name/ID",
        "definition": "The unique label for a consignment.",
    },
    {
        "group": "Consignments",
        "term": "Synthetic consignments",
        "definition": "Fake consignments generated to mimic realistic consignments.",
    },
    {
        "group": "Consignments",
        "term": "Historical consignments",
        "definition": "Actual consignments from past inspections at plant inspection stations.",
    },
    {
        "group": "Consignments",
        "term": "Sampling method",
        "definition": "The algorithm used to generate synthetic consignments from source data.",
    },
    {
        "group": "Consignments",
        "term": "Producer grouping file",
        "definition": "An optional CSV file that groups producers; generally this file is used for entity resolition (maps raw names to groups that represent resolved entity names).", # JOE: "before it said so synthetic generation can preserve producer structure"; are groups used in synthetic generation?
    },
    {
        "group": "Consignments",
        "term": "Port of entry",
        "definition": "An official location where goods can legally enter the country.", # JOE: before it said "The inspection location assigned to an inspection unit." PIS != POE. Should we change the word on the consignment generation page
    },
    {
        "group": "Consignments",
        "term": "Country of origin",
        "definition": "The source country of an inspection unit.",
    },
    {
        "group": "Consignments",
        "term": "Propagative material type", # Joe - the manual entry page lists PMs that are not the usual PM types. They look like taxa.
        "definition": "A class of propogative materials (e.g., unrooted cutting, tissue culture) as defined by PPQ.",
    },
    {
        "group": "Consignments",
        "term": "Pathway", # Joe - these don't look like the normal pathways. Is this expected?
        "definition": "The transport route used for a consignment (e.g., Airport - Aircraft - Cargo - PIS)",
    },
    {
        "group": "Consignments",
        "term": "Sample units",
        "definition": "A sample unit is an individual member (e.g., box), selected from a population (e.g., boxes on a consignment), that is inspected.",
    },
    {
        "group": "Consignments",
        "term": "Sample units per inspection unit",
        "definition": "The number of sample units associated with a single inspection unit.",
    },
    {
        "group": "Consignments",
        "term": "Plants per sample unit",
        "definition": "How many plants are within each sample unit.",
    },
    {
        "group": "Consignments",
        "term": "RBS dataset",
        "definition": "The saved CSV output used by downstream pages for consignment, policy, and scenario workflows.", # Joe - I'll need your help on better describing what this actually is
    },
    {
        "group": "Consignments",
        "term": "Multinomial Sequential",
        "definition": "An approach to generate synthetic consignments that samples from multinomial distributions", # Joe - Please review
    },
    {
        "group": "Consignments",
        "term": "Gaussian Mixure",
        "definition": "An approach to generate synthetic consignments that samples from gaussian distributions", # Joe - Please review
    },
    {
        "group": "Contamination",
        "term": "Contaminated",
        "definition": "Whether an item (e.g., plant, sampling unit) had at least one pest, pathogen, or other contaminate.",
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
        "term": "Contamination parameter set name",
        "definition": "The name assigned to a saved contamination parameter set.",
    },
    {
        "group": "Contamination",
        "term": "Average contamination rate",
        "definition": "Given an uncertain contamination rate, the mean of the distribution describing the contamination rate.",
    },
    {
        "group": "Contamination",
        "term": "Alpha",
        "definition": "A beta-binomial shape parameter that helps control contamination rate.",
    },
    {
        "group": "Contamination",
        "term": "Beta",
        "definition": "The beta-binomial shape parameter paired with alpha, representing the uncertainty about the contamination rate.",
    },
    {
        "group": "Contamination",
        "term": "Theta", # Is this needed? I don't see it mentioned anywhere to the user
        "definition": "A clustering parameter for the contamination model; in manual assignment it is fixed to infinity.",
    },
    {
        "group": "Contamination",
        "term": "Beta-binomial",
        "definition": "The probability distribution used in the app to model uncertain contamination rates.",
    },
    {
        "group": "Contamination",
        "term": "Beta-binomial Probability Density Function (PDF)",
        "definition": "The chart that visualizes the fitted or saved beta-binomial contamination distribution.",
    },
    {
        "group": "Contamination",
        "term": "Saved contamination parameter sets", # Is this necessary? I don't see it on the pages & we've already defined a parameter set
        "definition": "Stored contamination definitions written to the temporary JSON parameter store.", # Is it really temporary? If I relaunch the app, will they be there?
    },
    {
        "group": "Compliance",
        "term": "RBS compliance policy",
        "definition": "A policy that maps feature combinations of risk units to compliance levels.",
    },
    {
        "group": "Compliance",
        "term": "Compliance level upload",
        "definition": "A CSV that defines an RBS compliance policy.",
    },
    {
        "group": "Compliance",
        "term": "Detection/confidence mapping",
        "definition": "The CSV that maps compliance levels to detection and confidence values used in hypergeometric sampling.",
    },
    {
        "group": "Compliance",
        "term": "Feature columns",
        "definition": "Attributes of risk units used to determine their compliance level.",
    },
    {
        "group": "Compliance",
        "term": "Compliance level",
        "definition": "The policy category assigned to risk units, typically Poor, Low, Medium, or High.",
    },
    {
        "group": "Compliance",
        "term": "Detection level",
        "definition": "The smallest proportion of contaminated items in a finite lot that would be expected to be detected by the sampling plan.",
    },
    {
        "group": "Compliance",
        "term": "Confidence level",
        "definition": "The probability that the sampling plan will detect contamination when contamination is present at the detection level.",
    },
    {
        "group": "Compliance",
        "term": "Values for <feature>",
        "definition": "The allowed values selected for a chosen feature when manually building a policy.",
    },
    {
        "group": "Scenarios And Experiments",
        "term": "Experiment package",
        "definition": "A bundle of scenario rows and associated input files/parameters.",
    },
    {
        "group": "Scenarios And Experiments",
        "term": "Experiment set", # It is not clear to me if/how this is different than the experiment package
        "definition": "The named folder that contains one saved experiment package.",
    },
    {
        "group": "Scenarios And Experiments",
        "term": "Scenario table",
        "definition": "A table describing the scenario rows for a saved experiment package.",
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
        "term": "Files available",
        "definition": "The inventory of selectable datasets/parameter sets currently available for building scenarios.",
    },
    {
        "group": "Simulation",
        "term": "Action",
        "definition": "Action is taken when an actionable contaminate and/or pest was identified within the sampling units.",
    },{
        "group": "Simulation",
        "term": "Execution options",
        "definition": "The sidebar section where you choose replications and the experiment package to run.",
    },
    {
        "group": "Simulation",
        "term": "Simulation replications",
        "definition": "the number of times the same scenario is run to produce stable, reliable results despite randomness.",
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
        "definition": "The section summarizing performance across scenarios, replications, shipments, inspections, and totals.",
    },
    {
        "group": "Simulation",
        "term": "Shipments per replication", # Why use the term shipments here?
        "definition": "The number of consignments processed in each replication.",
    },
    {
        "group": "Simulation",
        "term": "Simulation summary view",
        "definition": "The control that switches the results to display either mean across replications, median across replications, or a single selected replication.",
    },
    {
        "group": "Simulation Results",
        "term": "Slippage",
        "definition": "Contaminated units (e.g., plants, sampling units) that evade detection.",
    },
    {
        "group": "Simulation Results",
        "term": "Slippage Level Results",
        "definition": "Results showing missed contaminated units at the consignment, inspection, sample, and plant levels.", # Joe - I am going to need some help here. It is not clear to me what I am looking at. Doesn't seem to match description.
    },
    {
        "group": "Simulation Results",
        "term": "Action Level Results", # Does it make sense to name this section something else? Like "Interceptions"?
        "definition": "Results showing intercepted versus slipped outcomes by level.", 
    },
    {
        "group": "Simulation Results",
        "term": "Contamination Level Results",
        "definition": "Results showing contaminated versus clean counts and percentages by level.", # Joe - I am going to need some help here. It is not clear to me what I am looking at
    },
    {
        "group": "Simulation Results",
        "term": "Inspection Workload Level Results",
        "definition": "Results showing inspection workload and completion metrics.", # Joe - I am going to need some help here. It is not clear to me what I am looking at
    },
    {
        "group": "Simulation Results",
        "term": "Intercepted",
        "definition": "Contaminated units detected by the inspection process at the reported level.",
    },
    {
        "group": "Simulation Results",
        "term": "Slipped",
        "definition": "Contaminated units not detected by the inspection process at the reported level, representing slippage.",
    },
    {
        "group": "Simulation Results",
        "term": "95% Interval",
        "definition": "A 95% confidence interval for the mean of the metric of interest; computed assuming a normal distribution.",
    },
    {
        "group": "Simulation Results",
        "term": "lower",
        "definition": "The minimum observed value across all replications", # Joe - is this correct?
    },
    {
        "group": "Simulation Results",
        "term": "upper",
        "definition": "The maximum observed value across all replications", # Joe - is this correct?
    },
    {
        "group": "Simulation Results",
        "term": "Workload",
        "definition": "The number of items inspected.", 
    },
    {
        "group": "App Notes",
        "term": "RBS",
        "definition": "Risk-based Sampling",
    },
    {
        "group": "App Notes",
        "term": "PIS",
        "definition": "Plant Inspection Station",
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
