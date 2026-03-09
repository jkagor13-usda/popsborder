# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import random

# Import functions from popsborder
from popsborder.scenarios import run_scenarios
from popsborder.inputs import load_configuration, load_scenario_table, load_compliance_lookup_csv
from popsborder.outputs import save_scenario_result_to_pandas
from popsborder.outputs import save_inspection_unit_detection_records_to_csv
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
    default_paths = DefaultPaths()
    data_dir = default_paths.slippage_data_dir()
    config_file = data_dir / "config.yml"
    compliance_file = data_dir / "compliance_table.csv"
    scenario_file = data_dir / "test_scenario.csv"

    pis_data_updated = shared_ppq_data_path / 'updated_pis_data.csv'  # PIS data
    #pis_data_updated = data_dir / "TEST_PIS_SampleQuantity.csv"       # Test data

    # Load configuration and compliance table
    config = load_configuration(config_file)

    # Synthetic data generation
    historical = False
    num_consignments_to_simulate = 5 # Added input parameter to be the number of consignments you want simulated
    synthetic_data_generator = SyntheticConsignmentDataGenerator(input_data_file=pis_data_updated)

    if historical:
        included_inspection_nums = synthetic_data_generator.input_data["INSPECTION_NUMBER"].sample(n=num_consignments_to_simulate)
        hist_data = synthetic_data_generator.input_data[synthetic_data_generator.input_data["INSPECTION_NUMBER"].isin(included_inspection_nums)]
        hist_data['Row_ID'] = 'CR-' + (hist_data.index + 1).astype(str)
        hist_out_path = data_dir / "Historical_PIS_SampleQuantity.csv"
        hist_data.to_csv(hist_out_path)
        config["consignment"]["input_file"]["file_name"] = "slippage_data/Historical_PIS_SampleQuantity.csv"
    else:
        synth_data = synthetic_data_generator.generate_from_input_data(
            n_consignments=num_consignments_to_simulate,
            sampling_method="sequential"
        )
        synth_out_path = data_dir / "Syntehtic_PIS_SampleQuantity.csv"
        synth_data.to_csv(synth_out_path)




    ####################################################################
    ####################################################################
    #################    CONTAMINATION MODULE  #########################
    ####################################################################
    ####################################################################

    ####################
    ### Read in Data ###
    ####################

    # # Load in PIS Data
    # df_pis_data = pd.read_csv(pis_data_updated)
    #
    # #############################################################
    # ##### TODO: Replace this block with the appropriate data ####
    # #############################################################
    #
    # ### Generate clarke inputs via input data
    # inputs_by_quantity = gen_clarke_model_inputs(df_pis_data)
    #
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

    # Setting values for testing
    res = {}
    inputs_by_quantity = {}
    for key in [(-0.001, 10.0),
                (10.0, 50.0),
                (50.0, 150.0),
                (150.0, 300.0),
                (300.0, 579.0),
                (579.0, 1000.0)]:
        inputs_by_quantity[key] = {'theta': np.inf, 'B': 200}
        res[key] = {
            'alpha': random.uniform(0.01, 0.25),
            "beta": random.uniform(2, 8),
            'mu': 0.0,
            'rho': 0.0,
            'D': 0.0
        }


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
        #scenario["contamination/contamination_rate/beta_binomial_parameters/alpha"] = res["alpha"]
        #scenario["contamination/contamination_rate/beta_binomial_parameters/beta"] =  res["beta"]
        #scenario["contamination/contamination_rate/beta_binomial_parameters/alpha"] = 0.194628
        #scenario["contamination/contamination_rate/beta_binomial_parameters/beta"] = 4.1 #20.12345
        #scenario["contamination/contamination_rate/beta_binomial_parameters/theta"] = inputs.theta
        #scenario["contamination/contamination_rate/value"] = None
        #scenario["contamination/contamination_rate/value"] = 1.23456
        scenario[f"contamination/arrangement"] = "clustered"

        # Setting actual paramters vaues
        for key in res.keys():
            scenario[f"contamination/contamination_rate/beta_binomial_parameters/{key}/alpha"] = res[key]['alpha']
            scenario[f"contamination/contamination_rate/beta_binomial_parameters/{key}/beta"] = res[key]['beta']
            scenario[f"contamination/contamination_rate/beta_binomial_parameters/{key}/mu"] = res[key]['mu']
            scenario[f"contamination/contamination_rate/beta_binomial_parameters/{key}/rho"] = res[key]['rho']
            scenario[f"contamination/contamination_rate/beta_binomial_parameters/{key}/D"] = res[key]['D']
            # scenario[f"contamination/contamination_rate/beta_binomial_parameters/{key}/theta"] = inputs_by_quantity[key].theta
            # scenario[f"contamination/contamination_rate/beta_binomial_parameters/{key}/J"] = inputs_by_quantity[
            #     key].B
            scenario[f"contamination/contamination_rate/beta_binomial_parameters/{key}/theta"] = inputs_by_quantity[
                key]['theta']
            scenario[f"contamination/contamination_rate/beta_binomial_parameters/{key}/J"] = inputs_by_quantity[
                key]['B']

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

    # Prepare results for saving
    if detailed_bool:
        scenario_results = [(result, config) for details, result, config in scenario_results_raw]
    else:
        scenario_results = [(result, config) for  result, config in scenario_results_raw]
    config_columns = ['contamination/contamination_unit', 'contamination/contamination_rate/distribution',
                      'contamination/contamination_rate/value', 'contamination/arrangement',
                      'inspection/sample_strategy', 'inspection/proportion/value', 'inspection/tolerance_level', 'name']
    result_columns = list(vars(scenario_results[0][0]).keys())

    # Create output folder if not there already
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save results to CSV
    results_df = save_scenario_result_to_pandas(scenario_results,
                                                config_columns=config_columns,
                                                result_columns=result_columns)
    results_df.to_csv(output_dir / "pis_contamination_scenario_results2.csv", index=False)
    print("Results saved to output/pis_contamination_scenario_results2.csv")

    if detailed_bool:
        inspection_unit_records = []
        for details, _, scenario_config in scenario_results_raw:
            if len(details) >= 3:
                for row in details[2]:
                    row_with_scenario = dict(row)
                    row_with_scenario["scenario_name"] = scenario_config.get("name")
                    inspection_unit_records.append(row_with_scenario)
        if inspection_unit_records:
            save_inspection_unit_detection_records_to_csv(
                inspection_unit_records,
                output_dir / "inspection_unit_detection_records.csv",
            )
            print("Results saved to output/inspection_unit_detection_records.csv")
    print('')

if __name__ == "__main__":
    main()
