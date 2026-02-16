# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

# Import functions from popsborder
from popsborder.scenarios import run_scenarios
from popsborder.inputs import load_configuration, load_scenario_table, load_compliance_lookup_csv
from popsborder.outputs import save_scenario_result_to_pandas
from popsborder.generator import SyntheticConsignmentDataGenerator, save_to_csv
from popsborder.consignments import get_consignment_generator

from popsborder.inspections import normalize_rbs_variables_against_consignment

# Import utility functions for contamination module
from slippage_model_utils.clarke_r_script_wrapper import *
from slippage_model_utils.clarke_model_support_functions import *
from slippage_model_utils.paths import BoxPaths, DefaultPaths


def main():
    # Set up data folder and file names
    box_paths = BoxPaths()
    shared_ppq_data_path = box_paths.shared_ppq_data()
    val_data_path = box_paths.validation_data()
    default_paths = DefaultPaths()
    data_dir = default_paths.slippage_data_dir()
    config_file = data_dir / "val_config.yml"
    compliance_file = data_dir / "compliance_table.csv"
    scenario_file = data_dir / "validation_scenario.csv"
    pis_data_train = val_data_path / 'train.csv'
    pis_data_test = val_data_path / 'test.csv'

    # Load configuration and compliance table
    config = load_configuration(config_file)

    # Load in the test data to understand how many consignments to generate
    df_pis_test_data = pd.read_csv(pis_data_test)
    # Group by inspection number to create consignments
    consignment_groups = list(df_pis_test_data.groupby('INSPECTION_NUMBER'))
    num_consignments_to_simulate = len(consignment_groups) # Define number of consignments in the test dataset

    config["consignment"]["input_file"]["rbs_file_name"] = str(val_data_path / "test.csv")

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

    ### Generate clarke inputs via input data
    inputs_by_quantity = gen_clarke_model_inputs(df_pis_train_data)

    # # Run clarke model
    # res = {}
    # print(f'\nNow Executing Clarke Model Based on Quantities')
    # for (lower, upper), inputs in inputs_by_quantity.items():
    #     print(f'   Calculating for Quantity Range:  {(lower, upper)}')
    #     res[(lower, upper)] = run_clarke_bb_group_model(inputs.ty,
    #                                     inputs.b,
    #                                     inputs.B,
    #                                     inputs.Nbar,
    #                                     inputs.freq,
    #                                     inputs.theta,
    #                                     inputs.R,
    #                                     inputs.start_val,
    #                                     inputs.se)
    #
    # print('\nFINAL CLARKE MODEL BETA-BINOMIAL PARAMETERS:')
    # for (lower, upper), results in res.items():
    #     print(f'   For quantities ranging in {(lower, upper)}:')
    #     print(f'      Alpha = {results["alpha"]}')
    #     print(f'      Beta = {results["beta"]}')
    #     #print(f'      Theta (from inputs) = {results.theta}')
    #     #print('\n      Full Clarke model result payload:')
    #     # for k, v in results.items():
    #     #     print(f'   {k}: {v}')
    #     print('')

    # Update original parameters of config
    config['contamination']['contamination_rate']['parameters'][0] = 0.194628
    config['contamination']['contamination_rate']['parameters'][1] = 4.7609372


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
    compliance_table = load_compliance_lookup_csv(compliance_file)

    # Generate a temporary consignment that will be generated during simulation
    consignment_generator = get_consignment_generator(config)
    temp_consignment = consignment_generator.generate_consignment()

    # Use temporarily generated consignment to find mappings of compliance table variables to attributes
    updated, mapping, unmapped= normalize_rbs_variables_against_consignment(
        compliance_table['rbs_variables'],
        temp_consignment
    )

    print(f'\n\nPre-Processed Submitted Compliance Table')
    print(f'   You have submitted the following variables in your compliance table and '
          f'they will be mapped to attributes that '
          f'the slippage model is generating for each consignment.')
    print("   === Original/Submitted Compliance Table Variables ===", compliance_table['rbs_variables'])

    print("\n   === Mappings Executed ===")
    for k, v in mapping.items():
        print(f"   {k!r} -> {v!r}")

    print("\n   === Unmapped Variables ===")
    for var in unmapped:
        print(f'      {var}')

    print("\n   === Updated Compliance Table Variables ===", updated)

    # Update the compliance table variable names to be used later in sim to match attributes of consignment object
    compliance_table['rbs_variables'] = updated

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
        scenario["contamination/contamination_rate/beta_binomial_parameters/alpha"] = 0.194628
        scenario["contamination/contamination_rate/beta_binomial_parameters/beta"] = 4.1
        scenario["contamination/contamination_rate/value"] = None

    ####################################################################
    ####################################################################
    ######## END CONTAMINATION MODULE (Scenario Update)  ###############
    ####################################################################
    ####################################################################




    # Run one scenario analysis simulation
    detailed_bool = True
    scenario_results_raw = run_scenarios(
        config=config,
        scenario_table=scenarios,
        seed=42,
        num_simulations=2,            # Only one simulation
        num_consignments=num_consignments_to_simulate,
        compliance_table=compliance_table,
        detailed=detailed_bool
    )


    # Post process outputs across replications/num_simulations to validate against previously seen action rates



if __name__ == "__main__":
    main()
