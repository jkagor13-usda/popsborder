import sys
from pathlib import Path
import pandas as pd

# Import functions from popsborder
from popsborder.scenarios import run_scenarios
from popsborder.inputs import load_configuration, load_scenario_table, load_compliance_lookup_csv
from popsborder.outputs import save_scenario_result_to_pandas
from popsborder.generator import SyntheticConsignmentDataGenerator, save_to_csv

# Import utility functions for contamination module
from slippage_model_utils.clarke_r_script_wrapper import *
from slippage_model_utils.clarke_model_support_functions import *


def main():
    # Set up data folder and file names
    data_dir = Path("slippage_data")
    config_file = data_dir / "config.yml"
    compliance_file = data_dir / "compliance_table.csv"
    #scenario_file = data_dir / "test_scenario.csv"
    scenario_file = data_dir / "pis_contaminate_scenarios.csv"
    pis_data = data_dir / 'synthetic_pis_data.csv'
    rbs_calc_data = data_dir / 'synthetic_rbs_calc_data.csv'

    # Load configuration and compliance table
    config = load_configuration(config_file)

    # Synthetic data generation
    synthetic_data_generator = SyntheticConsignmentDataGenerator(data_dir / "fake_pis_data.csv")
    synth_data = synthetic_data_generator.generate_from_input_data(n_samples=10, sampling_method="sequential")
    save_to_csv(synth_data, filename= data_dir / "synth_data.csv")

    config["consignment"]["input_file"]["rbs_file_name"] = str(data_dir / "synth_data.csv")

    ####################################################################
    ####################################################################
    #################    CONTAMINATION MODULE  #########################
    ####################################################################
    ####################################################################

    ####################
    ### Read in Data ###
    ####################

    #############################################################
    ##### TODO: Replace this block with the appropriate data ####
    #############################################################
    # Load in PIS Data
    df_pis_data = pd.read_csv(pis_data)

    # Load in RBS Calculator Data
    df_rbs_calculator = pd.read_csv(rbs_calc_data)
    #############################################################
    ##### TODO: Replace this block with the appropriate data ####
    #############################################################

    ### Generate clarke inputs via input data
    inputs = gen_clarke_model_inputs(df_pis_data, df_rbs_calculator)

    # Run clarke model
    res = run_clarke_bb_group_model(inputs.ty,
                                    inputs.b,
                                    inputs.B,
                                    inputs.Nbar,
                                    inputs.freq,
                                    inputs.theta,
                                    inputs.R,
                                    inputs.start_val,
                                    inputs.se)

    print('\nFINAL CLARKE MODEL BETA-BINOMIAL PARAMETERS:')
    print(f'   Alpha = {res["alpha"]}')
    print(f'   Beta = {res["beta"]}')
    print('')

    # Update original parameters of config
    config['contamination']['contamination_rate']['parameters'][0] = res["alpha"]
    config['contamination']['contamination_rate']['parameters'][1] = res["beta"]

    ####################################################################
    ####################################################################
    ################    END CONTAMINATION MODULE  ######################
    ####################################################################
    ####################################################################

    compliance_table = load_compliance_lookup_csv(compliance_file)

    # Load scenario table
    scenarios = load_scenario_table(scenario_file)
    print(f"Loaded {len(scenarios)} scenarios from {scenario_file}")

    ####################################################################
    ####################################################################
    ######## CONTAMINATION MODULE (Scenario Update)  ###################
    ####################################################################
    ####################################################################

    # Loop through each of the scenarios stored in "scenarios" (which is
    # a list of dictionaries) and replace with the updated fitted
    # contamination parameters
    for scenario in scenarios:
        scenario["contamination/contamination_rate/beta_binomial_parameters/alpha"] = res["alpha"]
        scenario["contamination/contamination_rate/beta_binomial_parameters/beta"] =  res["beta"]
        scenario["contamination/contamination_rate/beta_binomial_parameters/theta"] = inputs.theta
        scenario["contamination/contamination_rate/value"] = None

    ####################################################################
    ####################################################################
    ######## END CONTAMINATION MODULE (Scenario Update)  ###############
    ####################################################################
    ####################################################################




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

    # Create output folder if not there already
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save results to CSV
    results_df = save_scenario_result_to_pandas(scenario_results,f
                                                config_columns=config_columns,
                                                result_columns=result_columns)
    results_df.to_csv(output_dir / "pis_contamination_scenario_results.csv", index=False)
    print("Results saved to output/pis_contamination_scenario_results.csv")

if __name__ == "__main__":
    main()