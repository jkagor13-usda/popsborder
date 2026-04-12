# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import random
from datetime import datetime
import hashlib

# Import functions from popsborder
import sys
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

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
from slippage_model_utils.engineered_feature_creator import  create_engineered_features, map_group_to_shortest_name
from slippage_model_utils.paths import BoxPaths, DefaultPaths
from pathlib import Path
import pickle
import time


def main():

    ### Initialize default paths
    default_paths = DefaultPaths()
    box_paths = BoxPaths()

    ### Set up data folder and file names
    val_data_path = box_paths.validation_data()
    shared_ppq_data_path = box_paths.shared_ppq_data()
    model_testing_data_path = box_paths.model_testing_data_folder()
    data_dir = default_paths.slippage_data_dir()

    ### Configuration file  specification
    config_file = "config_test.yml"

    ### Compliance table
    compliance_file = data_dir / "compliance_table.csv"
    scenario_file = data_dir / "test_scenario.csv"
    compliance_table_path = data_dir / "base_compliance_table.csv"
    compliance_table_path_with_producer = data_dir / "base_compliance_table_with_producer.csv"
    compliance_mapping_to_detection_confidence = data_dir / "compliance_mapping_detection_confidence_levels.csv"

    ### PIS Inspection/RBS Calculator Data
    pis_data_updated = val_data_path / 'train.csv'
    #pis_data_updated = shared_ppq_data_path / 'updated_pis_data.csv'  # PIS data
    # pis_data_updated = data_dir / "TEST_PIS_SampleQuantity.csv"       # Test data
    # pis_data_updated = data_dir / "Synthetic_PIS_SampleQuantity_test.csv"       # Test data

    ### Other data loading
    producer_group_mapping_path = box_paths.disambiguated_producer_table_mapping()

    # Load configuration and compliance table
    config = load_configuration(config_file)

    # Load producer group mapping
    producer_group_mapping = pd.read_csv(producer_group_mapping_path)

    ### Synthetic data generation
    historical = False
    num_consignments_to_simulate = 5 # Added input parameter to be the number of consignments you want simulated
    synthetic_data_generator = SyntheticConsignmentDataGenerator(config=config,
                                                                 producer_group_mapping=producer_group_mapping,
                                                                 input_data_file=pis_data_updated)

    if historical:
        included_inspection_nums = synthetic_data_generator.input_data["INSPECTION_NUMBER"].sample(n=num_consignments_to_simulate)
        synth_data = synthetic_data_generator.input_data[synthetic_data_generator.input_data["INSPECTION_NUMBER"].isin(included_inspection_nums)]
        synth_data.loc[:, 'Row_ID'] = 'CR-' + (synth_data.index + 1).astype(str)
        synth_out_path = data_dir / "Historical_PIS_SampleQuantity.csv"
        config["consignment"]["input_file"]["file_name"] = "development_files/slippage_data/Historical_PIS_SampleQuantity.csv"
    else:
        synth_data = synthetic_data_generator.generate_from_input_data(
            n_consignments=num_consignments_to_simulate,
            sampling_method="sequential"
        )
        synth_out_path = data_dir / "Synthetic_PIS_SampleQuantity.csv"
        config["consignment"]["input_file"]["file_name"] = "development_files/slippage_data/Synthetic_PIS_SampleQuantity.csv"

    ### Creating of Engineered Features ###
    # Create features from the R script using the R wrapper
    creator = RVariableCreator()

    print(f'  Cleaning (and grouping where applicable) Categorical Names')
    print(f'      Cleaning Producer Name')
    synth_data['PRODUCER_NAME_RAW'] = synth_data['PRODUCER_NAME']
    synth_data['PRODUCER_NAME1'] = creator.batch_basic_text_preproc(text_fields=synth_data['PRODUCER_NAME_RAW'])

    producer_group_mapping = producer_group_mapping.rename(
        columns={"PRODUCER_NAME": "name", "grouping": "group"}
    )

    print(f'      Creating producer group mappings')
    synth_data = creator.entity_resolution(
        df=synth_data,
        entity_resolution_lookup_table=producer_group_mapping,
        use_parquet=False,  # or True, as you prefer
    )

    print(f'      Cleaning Importer Name')
    # Create a raw IMPORTER_NAME column with the original importer name
    synth_data['IMPORTER_NAME_RAW'] = synth_data['IMPORTER_NAME']

    # Update the IMPORTER_NAME column with the cleaned version.
    synth_data['IMPORTER_NAME1'] = creator.batch_basic_text_preproc(synth_data['IMPORTER_NAME_RAW'])

    # Reconstruct risk units based on configuration specification
    synth_data['PRODUCER_NAME'] = synth_data['PRODUCER_GROUP_NAME1']
    synth_data['IMPORTER_NAME'] = synth_data['IMPORTER_NAME1']
    synth_data = construct_risk_units(config=config, data=synth_data)

    # Pull in the VariableCreator object to use R code to create engineered columns based on created risk units
    synth_data = create_engineered_features(
        synth_data=synth_data,
        producer_group_mapping=producer_group_mapping
    )

    # Create the actual producer name that will be used to reference in the compliance table.
    synth_data = map_group_to_shortest_name(
        synth_data=synth_data,
        group_col="PRODUCER_GROUP_TOP",
        producer_group_mapping=producer_group_mapping,
        output_col="PRODUCER_GROUP_TOP",  # or None to overwrite
    )

    # Create a producer_group column
    synth_data['producer_group'] = synth_data['PRODUCER_GROUP_TOP']

    synth_data.to_parquet(data_dir / "Synthetic_Base.parquet", compression='snappy', index=False)
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
    df_pis_data = pd.read_csv(pis_data_updated)

    ### Generate clarke inputs via input data
    inputs_by_quantity = gen_clarke_model_inputs(df_pis_data)
    #
    # # Run clarke model
    res = {}

    # Defaults for if/when parameters for alpha/beta are both zero
    k = 10_000  # adjust upward/downward to change concentration
    default_alpha = 0.02 * k  # 200
    default_beta = 0.98 * k  # 9800
    print(f'\nNow Executing Clarke Model Based on Quantities')
    for (lower, upper), inputs in inputs_by_quantity.items():
        print(f'   Calculating for Quantity Range:  {(lower, upper)}')
        params = run_clarke_bb_group_model(
            inputs.ty,
            inputs.b,
            inputs.B,
            inputs.Nbar,
            inputs.freq,
            inputs.theta,
            inputs.R,
            inputs.start_val,
            inputs.se,
        )

        alpha = params.get("alpha", 0)
        beta = params.get("beta", 0)

        if alpha == 0 and beta == 0:
            # warn user
            print(
                f"   WARNING: Fitted alpha and beta were zero for range {(lower, upper)}; "
                f"defaulting to Beta(alpha={default_alpha}, beta={default_beta}) "
                f"(mean ~ 0.02 and 95% CI of [0.0173, 0.0227]).\n"
            )
            # override
            params["alpha"] = default_alpha
            params["beta"] = default_beta

        res[(lower, upper)] = params



    # for key, params in res.items():
    #     alpha = params.get("alpha", 0)
    #     beta = params.get("beta", 0)
    #
    #     if alpha == 0 and beta == 0:
    #         params["alpha"] = default_alpha
    #         params["beta"] = default_beta

    # Setting values for testing
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
        print('')


    ####################################################################
    ####################################################################
    ################    END CONTAMINATION MODULE  ######################
    ####################################################################
    ####################################################################

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

        # Setting actual parameters values
        for key in res.keys():
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

    #################################################################
    #################################################################
    #################    INSPECTION MODULE  #########################
    #################################################################
    #################################################################

    # Load compliance table
    compliance_table = build_compliance_lookup_table(
        compliance_table_filepath=compliance_table_path,
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



    temp_compliance_table = pd.read_csv(compliance_table_path)

    if "prod_group_name" in temp_compliance_table.columns:
        # Find values in synth_data that are NOT in temp_compliance_table
        mask = ~synth_data["producer_group"].isin(temp_compliance_table["prod_group_name"])

        # Collect the values that will be replaced (unique)
        replaced_values = synth_data.loc[mask, "producer_group"].unique()

        # Create a DataFrame for these values
        replaced_df = pd.DataFrame(replaced_values, columns=["Replaced_Producer_Group"])

        # Write to CSV
        replaced_df.to_csv(data_dir / "replaced_producer_group_from_prod_group_name.csv", index=False)

        # Replace those values with "Reference"
        synth_data.loc[mask, "producer_group"] = "Reference"

        # Check if ALL values in synth_data are in temp_compliance_table
        all_present = synth_data["producer_group"].isin(temp_compliance_table["prod_group_name"]).all()

        if all_present:
            print("✓ All producer groups are valid!")
        else:
            print("✗ Some producer groups are missing from temp_compliance_table")
    elif "PRODUCER_GROUP_TOP" in temp_compliance_table.columns:
        # Find values in synth_data that are NOT in temp_compliance_table
        mask = ~synth_data["producer_group"].isin(temp_compliance_table["PRODUCER_GROUP_TOP"])

        # Collect the values that will be replaced (unique)
        replaced_values = synth_data.loc[mask, "producer_group"].unique()

        # Create a DataFrame for these values
        replaced_df = pd.DataFrame(replaced_values, columns=["Replaced_Producer_Group"])

        # Write to CSV
        replaced_df.to_csv(data_dir / "replaced_producer_group_from_PRODUCER_GROUP_TOP.csv", index=False)

        # Replace those values with "Reference"
        synth_data.loc[mask, "producer_group"] = "Reference"

        # Check if ALL values in synth_data are in temp_compliance_table
        all_present = synth_data["producer_group"].isin(temp_compliance_table["PRODUCER_GROUP_TOP"]).all()

        if all_present:
            print("✓ All producer groups are valid!")
        else:
            print("✗ Some producer groups are missing from temp_compliance_table")
    if "IMPORTER_NAME_TOP" in temp_compliance_table.columns:
        # Find values in synth_data that are NOT in temp_compliance_table
        mask = ~synth_data["IMPORTER_NAME"].isin(temp_compliance_table["IMPORTER_NAME_TOP"])

        # Collect the replaced rows (both columns)
        replaced_df = synth_data.loc[mask, ["IMPORTER_NAME", "IMPORTER_NAME_RAW"]].drop_duplicates()

        # Rename columns for clarity
        replaced_df.rename(columns={"IMPORTER_NAME": "Replaced_Importer_Name",
                                    "IMPORTER_NAME_RAW": "Raw_Importer_Name"}, inplace=True)

        # Write to CSV
        replaced_df.to_csv(data_dir / "replaced_importer_names.csv", index=False)

        # Replace those values with "Reference"
        synth_data.loc[mask, "IMPORTER_NAME"] = "Reference"

        # Check if ALL values in synth_data are in temp_compliance_table
        all_present = synth_data["IMPORTER_NAME"].isin(temp_compliance_table["IMPORTER_NAME_TOP"]).all()

        if all_present:
            print("✓ All importer names are valid!")
        else:
            print("✗ Some importer names are missing from temp_compliance_table")

    synth_data.to_parquet(data_dir / "Synthetic_Base_Use.parquet", compression='snappy', index=False)
    synth_data.to_csv(data_dir / "Synthetic_Base_Use.csv", index=False)

    config["consignment"]["input_file"]["file_name"] = "development_files/slippage_data/Synthetic_Base_Use.csv"

    ##################################################################
    ##################################################################
    ################# END INSPECTION MODULE  #########################
    ##################################################################
    ##################################################################

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
