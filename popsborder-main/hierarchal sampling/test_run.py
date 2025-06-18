from popsborder.scenarios import run_scenarios
from popsborder.inputs import load_configuration, load_scenario_table
from popsborder.outputs import save_scenario_result_to_pandas

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


datadir = Path("popsborder-main/hierarchal sampling/data")
basic_config = load_configuration(datadir / "config.yml")

# Modify a directory for the plots here
plotsdir = Path("plots")
# Make sure the directory exists
plotsdir.mkdir(exist_ok=True)

# Load csv with configurations for each example scenario
contaminate_scenarios = load_scenario_table(datadir / "contaminate_examples.csv")
sample_scenarios = load_scenario_table(datadir / "sampling_examples.csv")

# Run contamination examples
num_consignments_1 = 3
contaminate_examples = run_scenarios(
    config=basic_config,
    scenario_table=contaminate_scenarios,
    seed=42,
    num_simulations=1,
    num_consignments=num_consignments_1,
    detailed=True,
)

print(contaminate_examples)