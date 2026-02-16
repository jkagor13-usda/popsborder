# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

from dataclasses import dataclass
import streamlit as st


@dataclass
class Scenario:
    title: str = "Demo fruit shipment - port of entry"
    seed: int = 42
    runs: int = 500
    unit_type: str = "Boxes"  # Boxes / Crates / Pallets
    units_in_shipment: int = 1200
    items_per_unit: int = 100
    packaging_type: str = "Cardboard"
    contamination_prevalence_pct: float = 2.0
    contamination_arrangement: str = "Random"  # Random / Clustered / Edge-loaded
    contamination_cv: float = 1.0
    notes: str = ""


@dataclass
class Inspection:
    method: str = "random"  # random / convenience / cluster
    sample_units: int = 60
    sample_items_per_unit: int = 10
    acceptance_number: int = 0


def init_state():
    if "scenario" not in st.session_state:
        st.session_state.scenario = Scenario()
    if "inspection" not in st.session_state:
        st.session_state.inspection = Inspection()


def set_scenario(**kwargs):
    scenario = st.session_state.scenario
    for key, value in kwargs.items():
        setattr(scenario, key, value)


def set_inspection(**kwargs):
    inspection = st.session_state.inspection
    for key, value in kwargs.items():
        setattr(inspection, key, value)
