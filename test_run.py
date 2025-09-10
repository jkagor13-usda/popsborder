from popsborder.scenarios import run_scenarios
from popsborder.inputs import load_configuration, load_scenario_table, load_compliance_lookup_csv
from popsborder.outputs import save_scenario_result_to_pandas
import popsborder.rbs_consignment as rcon

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wasserstein_distance, energy_distance


datadir = Path("hierarchal sampling/data")
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
try:
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
except:
    print("Contamination examples failed to run")

# Run sampling examples
try:
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
except:
    print("Sampling examples failed to run")

# Create synthetic data
try:
    raw_df, synth_data = rcon.generate_synthetic_consignment_data(num_rows=500, 
                                             output_file=f'{datadir}/synthetic_data.csv', 
                                             input_file=f'{datadir}/fake_pis_data.csv', 
                                             mask_cats=False)
    
except Exception as e:
    print(f"An error occurred: {e}")
else:
    cat_cols =['INSPECTION_LOCATION_NAME','PATHWAY','COUNTRY_OF_ORIGIN_NAME','PROPAGATIVE_MATERIAL_TYPE','TOTAL_SAMPLING_UNITS','TOTAL_PLANT_QUANTITY','PRODUCER']
    rcon.plot_pairplots(raw_df, synth_data, cat_cols, output_dir="plots")


    # Select only numeric columns
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
    X = raw_df[numeric_cols].to_numpy()
    Y = synth_data[numeric_cols].to_numpy()

    import ot  # POT: Python Optimal Transport

    # Uniform weights for empirical distributions
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]

    # Compute cost matrix (Euclidean distances)
    M = ot.dist(X, Y, metric='euclidean')

    # Compute the 2-Wasserstein distance (squared)
    wass2 = ot.emd2(a, b, M)
    wass = np.sqrt(wass2)
    print(f"Multivariate Wasserstein distance: {wass}")


# Feed in synthetic data
# try:

# except: