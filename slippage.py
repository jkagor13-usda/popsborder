import sys
from pathlib import Path
import pandas as pd

# Import functions from popsborder
from popsborder.scenarios import run_scenarios
from popsborder.inputs import load_configuration, load_scenario_table, load_compliance_lookup_csv
from popsborder.outputs import save_scenario_result_to_pandas
from popsborder.generator import SyntheticConsignmentDataGenerator, save_to_csv

def main():
    # Set up data folder and file names
    data_dir = Path("slippage_data")
    config_file = data_dir / "config.yml"
    compliance_file = data_dir / "compliance_table.csv"
    scenario_file = data_dir / "pis_contaminate_scenarios.csv"

    # Load configuration and compliance table
    config = load_configuration(config_file)

    # Synthetic data generation
    synthetic_data_generator = SyntheticConsignmentDataGenerator(data_dir / "fake_pis_data.csv")
    synth_data = synthetic_data_generator.generate_from_input_data(n_samples=10, sampling_method="sequential")
    save_to_csv(synth_data, filename= data_dir / "synth_data.csv")

    config["consignment"]["input_file"]["rbs_file_name"] = str(data_dir / "synth_data.csv")

    # TODO: Contamination data
    compliance_table = load_compliance_lookup_csv(compliance_file)

    # Load scenario table
    scenarios = load_scenario_table(scenario_file)
    print(f"Loaded {len(scenarios)} scenarios from {scenario_file}")

    # Run one scenario analysis simulation
    scenario_results_raw = run_scenarios(
        config=config,
        scenario_table=scenarios,
        seed=42,
        num_simulations=1,            # Only one simulation
        num_consignments=5,           # You can change this number if needed
        compliance_table=compliance_table,
        detailed=True
    )

    # Prepare results for saving
    scenario_results = [(result, config) for details, result, config in scenario_results_raw]
    config_columns = ['contamination/contamination_unit', 'contamination/contamination_rate/distribution',
                      'contamination/contamination_rate/value', 'contamination/arrangement',
                      'inspection/sample_strategy', 'inspection/proportion/value', 'inspection/tolerance_level', 'name']
    result_columns = ['num_inspections', 'intercepted', 'false_neg', 'missing', 'true_contamination_rate',
                      'avg_missed_contamination_rate', 'max_missed_contamination_rate',
                      'total_missed_contaminants', 'total_intercepted_contaminants']

    # Save results to CSV
    results_df = save_scenario_result_to_pandas(scenario_results,
                                                config_columns=config_columns,
                                                result_columns=result_columns)
    results_df.to_csv("output/pis_contamination_scenario_results.csv", index=False)
    print("Results saved to output/pis_contamination_scenario_results.csv")

if __name__ == "__main__":
    main()