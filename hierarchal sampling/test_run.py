from popsborder.scenarios import run_scenarios
from popsborder.inputs import load_configuration, load_scenario_table, load_compliance_lookup_csv
from popsborder.outputs import save_scenario_result_to_pandas

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


datadir = Path("popsborder-main/hierarchal sampling/data")
# datadir = Path("data")  # run in dedicated terminal
basic_config = load_configuration(datadir / "config.yml")

# Modify a directory for the plots here
plotsdir = Path("plots")
# Make sure the directory exists
plotsdir.mkdir(exist_ok=True)

# Load csv with configurations for each example scenario
contaminate_scenarios = load_scenario_table(datadir / "contaminate_examples.csv")
sample_scenarios = load_scenario_table(datadir / "sampling_examples.csv")
compliance_table = load_compliance_lookup_csv(datadir / "compliance_table.csv")


# Run contamination examples
num_consignments_1 = 3
contaminate_examples = run_scenarios(
    config=basic_config,
    scenario_table=contaminate_scenarios,
    seed=42,
    num_simulations=1,
    num_consignments=num_consignments_1,
    compliance_table = compliance_table,
    detailed=True,
)

# print(contaminate_examples)

# Run sampling examples
num_consignments = 3
sample_examples = run_scenarios(
    config=basic_config,
    scenario_table=sample_scenarios,
    seed=42,
    num_simulations=1,
    num_consignments=num_consignments,
    compliance_table = compliance_table,
    detailed=True,
)