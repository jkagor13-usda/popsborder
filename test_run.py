from popsborder.scenarios import run_scenarios
from popsborder.inputs import load_configuration, load_scenario_table, load_compliance_lookup_csv
from popsborder.outputs import save_scenario_result_to_pandas
from popsborder.generator import SyntheticConsignmentDataGenerator
import old.rbs_consignment as rcon

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

# Create synthetic data
# Example usage with input data file
input_file = "fake_pis_data.csv"  # Update path as needed

generator = SyntheticConsignmentDataGenerator(input_data_file=input_file)

# Generate synthetic dataset using advanced sampling
num_records = 5000
print(f"Generating {num_records} synthetic records using Gaussian Copula sampling...")
    
try:
    synthetic_dataset = generator.generate_dataset(num_records, use_input_data=True)
    
    # Save to files
    generator.save_to_csv(synthetic_dataset, f"{datadir}/synthetic_consignments_advanced.csv")
    generator.save_to_json(synthetic_dataset, f"{datadir}/synthetic_consignments_advanced.json")
    
    # Generate and display statistics
    stats = generator.generate_statistics(synthetic_dataset)
    print("\nSynthetic Dataset Statistics:")
    print(f"Total records: {stats['total_records']}")
    print(f"Columns: {len(stats['columns'])}")
    print(f"Numeric columns: {stats['numeric_columns']}")
    print(f"Categorical columns: {stats['categorical_columns']}")
    
    # Calculate quality metrics if original data is available
    if generator.input_data is not None:
        quality_metrics = generator.calculate_quality_metrics(
            generator.input_data, synthetic_dataset
        )
        print("\nQuality Metrics (Wasserstein Distance):")
        for metric, value in quality_metrics.items():
            print(f"{metric}: {value:.4f}")
    
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

# Ingest synthetic data or RBS calculator

# calculate contamination probability from data (Clarke)


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
    print("Contamination examples completed successfully!")
except Exception as e:
    print("Contamination examples failed to run")
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

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
    print("Sampling examples completed successfully!")
except Exception as e:
    print("Sampling examples failed to run")
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()


# plot output
