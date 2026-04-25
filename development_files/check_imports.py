# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

"""
Sanity-check script to verify imports for the slippage model project.

Place this file in:
    plant-inspection-station-simulation/development_files/check_imports.py

Usage (from project root):
    cd plant-inspection-station-simulation
    python development_files/check_imports.py
"""

import importlib
import sys
from pathlib import Path
from textwrap import indent


# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------

# This file: .../plant-inspection-station-simulation/development_files/check_imports.py
THIS_FILE = Path(__file__).resolve()
DEV_DIR = THIS_FILE.parent
PROJECT_ROOT = DEV_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Modules to check
# ---------------------------------------------------------------------------

# Third-party packages (from requirements + code review)
THIRD_PARTY_MODULES = [
    # Core scientific and data packages
    "numpy",
    "pandas",
    "scipy",
    "sklearn",      # scikit-learn imports as "sklearn"
    "yaml",         # pyyaml provides the "yaml" module

    # Optional typing backport
    "typing_extensions",

    # Development / testing utilities (optional to check at runtime)
    "pytest",
    "flake8",
    "pylint",
    "black",
    "pytest_datadir",

    # GUI and visualization packages
    "streamlit",
    "chardet",
    "plotly",
    "docx",         # python-docx imports as "docx"
    "matplotlib",
    "altair",
]

# Local / in-repo packages or namespaces
LOCAL_MODULES = [
    # popsborder package
    "popsborder",
    "popsborder.generator",
    "popsborder.inputs",
    "popsborder.outputs",
    "popsborder.inspections",
    "popsborder.scenarios",
    "popsborder.contamination",
    "popsborder.consignments",
    "popsborder.simulation",

    # slippage_model_utils package
    "slippage_model_utils",
    "slippage_model_utils.clarke_model_support_functions",
    "slippage_model_utils.r_script_wrapper",
    "slippage_model_utils.references",
    "slippage_model_utils.paths",

    # gui package
    "gui",
    "gui.env_check",
    "gui.models",
    "gui.navigation",
    "gui.page_styles",
    "gui.report_export",
    "gui.runtime_warnings",
    "gui.slippage_pipeline",
    "gui.slippage_ui",
    # "pages.1_Consignment_Generation",
    # "pages.2_Contamination_Fit",
    # "pages.3_Inspection_Process",
    # "pages.4_Scenario_Experiments",
    # "pages.5_Run_Simulation",
    # "pages.6_Glossary",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def check_module(name: str) -> tuple[str, bool, str | None]:
    """Try to import a module by name; return (name, success, error_message_or_None)."""
    try:
        importlib.import_module(name)
        return name, True, None
    except Exception as exc:  # noqa: BLE001 - we want to catch any import failure
        return name, False, repr(exc)


def run_checks() -> int:
    print(f"Project root assumed to be: {PROJECT_ROOT}")
    print("=== Checking third-party modules (requirements.txt) ===")
    missing_third_party: list[str] = []
    for mod in THIRD_PARTY_MODULES:
        name, ok, err = check_module(mod)
        if ok:
            print(f"  [OK]   {name}")
        else:
            print(f"  [FAIL] {name}")
            print(indent(f"Error: {err}", "         "))
            missing_third_party.append(name)

    print("\n=== Checking local / in-repo modules ===")
    missing_local: list[str] = []
    for mod in LOCAL_MODULES:
        name, ok, err = check_module(mod)
        if ok:
            print(f"  [OK]   {name}")
        else:
            print(f"  [FAIL] {name}")
            print(indent(f"Error: {err}", "         "))
            missing_local.append(name)

    print("\n=== Summary ===")
    if not missing_third_party and not missing_local:
        print("All modules imported successfully.")
        return 0

    if missing_third_party:
        print("\nMissing third-party modules (check requirements.txt / pip install):")
        for mod in missing_third_party:
            print(f"  - {mod}")

    if missing_local:
        print("\nMissing local modules (check your package layout / PYTHONPATH / editable installs):")
        for mod in missing_local:
            print(f"  - {mod}")

    return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(run_checks())
