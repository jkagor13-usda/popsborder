from popsborder.scenarios import run_scenarios
from popsborder.inputs import load_configuration, load_scenario_table, load_compliance_lookup_csv
from popsborder.outputs import save_scenario_result_to_pandas
from popsborder.generator import SyntheticConsignmentDataGenerator
import old.rbs_consignment as rcon

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wasserstein_distance, energy_distance

# Define 
datadir = Path("hierarchal sampling/data")
basic_config = load_configuration(datadir / "config.yml")
plotsdir = Path("plots")
