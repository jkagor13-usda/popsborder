# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import random

# Import functions from popsborder
from popsborder.scenarios import run_scenarios
from popsborder.inputs import load_configuration, load_scenario_table, load_compliance_lookup_csv, build_compliance_lookup_table
from popsborder.outputs import save_scenario_result_to_pandas
from popsborder.outputs import save_inspection_unit_detection_records_to_csv
from popsborder.generator import SyntheticConsignmentDataGenerator, save_to_csv
from popsborder.consignments import get_consignment_generator

from popsborder.inspections import normalize_rbs_variables_against_consignment, construct_risk_units, normalize_rbs_variables_using_risk_unit_config

# Import utility functions for contamination module
from slippage_model_utils.r_script_wrapper import *
from slippage_model_utils.clarke_model_support_functions import *
from slippage_model_utils.validation_utils import *
from slippage_model_utils.paths import BoxPaths, DefaultPaths
from pathlib import Path
import pickle
import time


def main():
    start = time.time()
    # Set up data folder and file names
    box_paths = BoxPaths()
    shared_ppq_data_path = box_paths.shared_ppq_data()
    val_data_path = box_paths.validation_data()
    default_paths = DefaultPaths()
    data_dir = default_paths.slippage_data_dir()
    config_file = data_dir / "val_config.yml"
    compliance_file = data_dir / "compliance_table.csv"
    base_compliance_table = data_dir / "base_compliance_table.csv"
    base_compliance_table_with_producer = data_dir / "base_compliance_table_with_producer.csv"
    compliance_mapping_to_detection_confidence = data_dir / "compliance_mapping_detection_confidence_levels.csv"
    scenario_file = data_dir / "validation_scenario.csv"
    pis_data_train = val_data_path / 'train.csv'
    pis_data_test_path = val_data_path / 'test.csv'

    # Load configuration and compliance table
    config = load_configuration(config_file)

    # Load in the test data to understand how many consignments to generate
    df_pis_test_data = pd.read_csv(pis_data_test_path)
    # Group by inspection number to create consignments
    consignment_groups = list(df_pis_test_data.groupby('INSPECTION_NUMBER'))
    num_consignments_to_simulate = len(consignment_groups) # Define number of consignments in the test dataset

    config["consignment"]["input_file"]["file_name"] = str(val_data_path / "test.csv")

    ### Other data loading
    producer_group_mapping_path = box_paths.disambiguated_producer_table_mapping()

    # Load configuration and compliance table
    config = load_configuration(config_file)

    # Load producer group mapping
    producer_group_mapping = pd.read_csv(producer_group_mapping_path)

    ### Synthetic data generation
    historical = True
    #num_consignments_to_simulate = 5  # Added input parameter to be the number of consignments you want simulated
    synthetic_data_generator = SyntheticConsignmentDataGenerator(config=config,
                                                                 producer_group_mapping=producer_group_mapping,
                                                                 input_data_file=pis_data_test_path)

    if historical:
        synth_data = synthetic_data_generator.input_data
        # included_inspection_nums = synthetic_data_generator.input_data["INSPECTION_NUMBER"].sample(
        #     n=num_consignments_to_simulate)
        # synth_data = synthetic_data_generator.input_data[
        #     synthetic_data_generator.input_data["INSPECTION_NUMBER"].isin(included_inspection_nums)]
        synth_data.loc[:, 'Row_ID'] = 'CR-' + (synth_data.index + 1).astype(str)
        synth_out_path = data_dir / "Historical_PIS_SampleQuantity.csv"
        config["consignment"]["input_file"]["file_name"] = str(synth_out_path)
    else:
        synth_data = synthetic_data_generator.generate_from_input_data(
            n_consignments=num_consignments_to_simulate,
            sampling_method="sequential"
        )
        synth_out_path = data_dir / "Synthetic_PIS_SampleQuantity.csv"
        config["consignment"]["input_file"]["file_name"] = str(synth_out_path)

    # # Pull in the VariableCreator object to use R code to create engineered columns
    # creator = VariableCreator()
    #
    # quantity_binary_variables = creator.generate_quantity_binaries(synth_data, quantity_threshold=200,
    #                                                                group_cols=['RISK_UNIT'])
    #
    # synth_data = synth_data.merge(
    #     quantity_binary_variables,
    #     on='RISK_UNIT',
    #     how='left'
    # )

    synth_data.to_csv(synth_out_path)

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
    df_pis_train_data = pd.read_csv(pis_data_train)

    #############################################################
    ##### TODO: Replace this block with the appropriate data ####
    #############################################################

    ## Generate clarke inputs via input data
    inputs_by_quantity = gen_clarke_model_inputs(df_pis_train_data)

    #inputs_by_quantity = gen_clarke_model_inputs(df_pis_test_data)

    # Run clarke model
    res = {}
    print(f'\nNow Executing Clarke Model Based on Quantities')
    for (lower, upper), inputs in inputs_by_quantity.items():
        print(f'   Calculating for Quantity Range:  {(lower, upper)}')
        res[(lower, upper)] = run_clarke_bb_group_model(inputs.ty,
                                        inputs.b,
                                        inputs.B,
                                        inputs.Nbar,
                                        inputs.freq,
                                        inputs.theta,
                                        inputs.R,
                                        inputs.start_val,
                                        inputs.se)

    # # Setting values for testing
    # res = {}
    # inputs_by_quantity = {}
    # for key in [(-0.001, 10.0),
    #             (10.0, 50.0),
    #             (50.0, 150.0),
    #             (150.0, 300.0),
    #             (300.0, 579.0),
    #             (579.0, 1000.0)]:
    #     inputs_by_quantity[key] = {'theta': np.inf, 'B': 200}
    #     res[key] = {
    #         'alpha': random.uniform(0.01, 0.25),
    #         "beta": random.uniform(2, 8),
    #         'mu': 0.0,
    #         'rho': 0.0,
    #         'D': 0.0
    #     }

    print('\nFINAL CLARKE MODEL BETA-BINOMIAL PARAMETERS:')
    n = 1000
    for (lower, upper), results in res.items():
        alpha = results["alpha"]
        beta = results["beta"]

        mean = n * alpha / (alpha + beta)
        variance = (n * alpha * beta * (alpha + beta + n)) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        std_dev = variance ** 0.5

        print(f'   For quantities ranging in {(lower, upper)}:')
        print(f'      α={alpha:.4f}, β={beta:.4f} | '
              f'Mean={mean:.2f}, SD={std_dev:.2f} (N={n})')
        # print(f'      Theta (from inputs) = {results.theta}')
        # print('\n      Full Clarke model result payload:')
        # for k, v in results.items():
        #     print(f'   {k}: {v}')
        print('')

    # Update original parameters of config
    # config['contamination']['contamination_rate']['parameters'][0] = 0.194628
    # config['contamination']['contamination_rate']['parameters'][1] = 4.7609372


    ####################################################################
    ####################################################################
    ################    END CONTAMINATION MODULE  ######################
    ####################################################################
    ####################################################################

    #################################################################
    #################################################################
    #################    INSPECTION MODULE  #########################
    #################################################################
    #################################################################

    # Load compliance table
    compliance_table = build_compliance_lookup_table(
        compliance_table_filepath=base_compliance_table,
        mapping_filepath=compliance_mapping_to_detection_confidence
    )

    # Normalize RBS variables against RiskUnit attributes
    updated, mapping2, unmapped2 = normalize_rbs_variables_using_risk_unit_config(
        compliance_table['rbs_variables']
    )

    # Update the compliance table variable names to be used later in sim to match attributes of consignment object
    compliance_table['rbs_variables'] = updated

    # Output files
    compliance_lookup_pkl = data_dir / 'compliance_lookup_final.pkl'

    # Now use these throughout your code
    with open(compliance_lookup_pkl, 'wb') as f:
        pickle.dump(compliance_table, f, protocol=pickle.HIGHEST_PROTOCOL)

    config["inspection"]["compliance_table"]['file_name'] = data_dir / 'compliance_lookup_final.pkl'

    ##################################################################
    ##################################################################
    ################# END INSPECTION MODULE  #########################
    ##################################################################
    ##################################################################

    # Load scenario table
    scenarios = load_scenario_table(scenario_file)
    print(f"\nLoaded {len(scenarios)} scenarios from {scenario_file}")

    ####################################################################
    ####################################################################
    ######## CONTAMINATION MODULE (Scenario Update)  ###################
    ####################################################################
    ####################################################################

    # Loop through each of the scenarios stored in "scenarios" (which is
    # a list of dictionaries) and replace with the updated fitted
    # contamination parameters
    for scenario in scenarios:
        #scenario["contamination/contamination_rate/beta_binomial_parameters/alpha"] = res["alpha"]
        #scenario["contamination/contamination_rate/beta_binomial_parameters/beta"] =  res["beta"]
        #scenario["contamination/contamination_rate/beta_binomial_parameters/alpha"] = 0.194628
        #scenario["contamination/contamination_rate/beta_binomial_parameters/beta"] = 4.1 #20.12345
        #scenario["contamination/contamination_rate/beta_binomial_parameters/theta"] = inputs.theta
        #scenario["contamination/contamination_rate/value"] = None
        #scenario["contamination/contamination_rate/value"] = 1.23456
        #scenario[f"contamination/arrangement"] = "clustered"

        # Setting actual paramters vaues
        for key in res.keys():
            print(key)
            scenario[f"contamination/contamination_rate/beta_binomial_parameters/{key}/alpha"] = res[key]['alpha']
            scenario[f"contamination/contamination_rate/beta_binomial_parameters/{key}/beta"] = res[key]['beta']
            scenario[f"contamination/contamination_rate/beta_binomial_parameters/{key}/mu"] = res[key]['mu']
            scenario[f"contamination/contamination_rate/beta_binomial_parameters/{key}/rho"] = res[key]['rho']
            scenario[f"contamination/contamination_rate/beta_binomial_parameters/{key}/D"] = res[key]['D']
            scenario[f"contamination/contamination_rate/beta_binomial_parameters/{key}/theta"] = inputs_by_quantity[key].theta
            scenario[f"contamination/contamination_rate/beta_binomial_parameters/{key}/J"] = inputs_by_quantity[
                key].B
            # scenario[f"contamination/contamination_rate/beta_binomial_parameters/{key}/theta"] = inputs_by_quantity[
            #     key]['theta']
            # scenario[f"contamination/contamination_rate/beta_binomial_parameters/{key}/J"] = inputs_by_quantity[
            #     key]['B']

    ####################################################################
    ####################################################################
    ######## END CONTAMINATION MODULE (Scenario Update)  ###############
    ####################################################################
    ####################################################################




    # Run one scenario analysis simulation
    detailed_bool = True
    num_replications = 240
    scenario_results_raw = run_scenarios(
        config=config,
        scenario_table=scenarios,
        seed=42,
        num_simulations=num_replications,            # Only one simulation
        num_consignments=num_consignments_to_simulate,
        compliance_table=compliance_table,
        detailed=detailed_bool
    )

    start = time.time()
    # Post process outputs across replications/num_simulations to validate against previously seen action rates
    # Configuration
    scenarios = ["Validation"]
    # Specify fields you want to produce action rate validation on
    sets_of_val_fields = [
        ["COUNTRY_OF_ORIGIN_NAME", "PROPAGATIVE_MATERIAL_TYPE"],
        ["COUNTRY_OF_ORIGIN_NAME"],
        ["PROPAGATIVE_MATERIAL_TYPE"],
        [],
    ]
    run_ts = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    for val_group_fields in sets_of_val_fields:
        temp_start = time.time()
        if len(val_group_fields) == 0:
            output_dir = DefaultPaths().validation_output_dir() / f"validation_updated_overall_{run_ts}"
            output_dir.mkdir(exist_ok=True)
        else:
            folder_name = '_'.join(val_group_fields)
            output_dir = DefaultPaths().validation_output_dir() / f"validation_updated_{folder_name}_{run_ts}"
            output_dir.mkdir(exist_ok=True)

        # List available simulation runs
        print("Available simulation runs:")
        available_runs = list_available_simulation_runs(default_paths.output_dir())
        for i, run in enumerate(available_runs, 1):
            print(f"{i}. {run['name']} (Timestamp: {run['timestamp']})")




        # Option 1: Use latest simulation run
        results = calculate_action_rates_by_scenario(
            ground_truth_path=pis_data_test_path,
            simulation_output_path=default_paths.output_dir(),
            scenarios=scenarios,
            num_replications=num_replications,
            filter_fields=val_group_fields,
            simulation_base_path="latest"  # Auto-select most recent
        )

        # Option 2: Use specific simulation run
        # results = calculate_action_rates_by_scenario(
        #     ground_truth_path=ground_truth_path,
        #     simulation_output_path=simulation_output_path,
        #     scenarios=scenarios,
        #     num_replications=num_replications,
        #     filter_fields=filter_fields,
        #     simulation_base_path="pops_border_scenario_data_03_02_2026_17_21_12"
        # )

        # Save results
        save_results(results=results, output_dir=output_dir)

        # Create and save statistical comparison summaries (now includes ground_truth_path)
        summaries = summarize_statistical_comparison(
            results=results,
            filter_fields=val_group_fields,
            ground_truth_path=pis_data_test_path,
            output_dir=output_dir
        )

        # Display summary
        if results:
            first_scenario = scenarios[0]
            print(f"\n{first_scenario} Overall Statistical Summary:")
            print(summaries[first_scenario]['overall_summary'])

        total_time = time.time() - temp_start
        total_time_mins = total_time / 60
        total_time_hours = total_time_mins / 60
        total_time_days = total_time_hours / 24
        print(f'\nTIMING SUMMARY\n')
        print(f'   Total Time to Execute Validation with {num_replications} Replications Over Fields {val_group_fields}')
        print(f'      Minutes:  {total_time_mins}')
        print(f'      Hours:  {total_time_hours}')
        print(f'      Days:  {total_time_days}')
        print('')

    total_time = time.time() - start
    total_time_mins = total_time / 60
    total_time_hours = total_time_mins / 60
    total_time_days = total_time_hours / 24
    print(f'\nTIMING SUMMARY\n')
    print(f'   Total Time to Execute Validation with {num_replications} Replications Overall Fields')
    print(f'      Minutes:  {total_time_mins}')
    print(f'      Hours:  {total_time_hours}')
    print(f'      Days:  {total_time_days}')
    print('')



if __name__ == "__main__":
    main()
