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


def main(df1=None):
    start = time.time()
    #### Main Input Parameter ######
    # Specify how many consignments you want to generate
    num_consignments_to_simulate = 20
    # Specify how many replications you want the simulation to execute
    num_replications = 50

    # Have already generated synthetic consignments you want to use?  Set to True, otherwise set to False (and
    # num_consignments_to_simulate will be generated)
    synthetic_data_generated = False

    # Are Clark model parameters cached and saved? True if yes, and False if not
    clark_parameters_cached = True









    ### Initialize default paths
    default_paths = DefaultPaths()
    box_paths = BoxPaths()

    val_data_path = box_paths.validation_data()
    pis_data_train = val_data_path / 'train.csv'
    pis_data_test_path = val_data_path / 'test.csv'

    ### Set up data folder and file names
    shared_ppq_data_path = box_paths.shared_ppq_data()
    model_testing_data_path = box_paths.model_testing_data_folder()
    data_dir = default_paths.slippage_data_dir()


    ### Configuration file  specification
    config_file = "config_test.yml"

    ### Compliance table
    compliance_file = data_dir / "compliance_table.csv"
    scenario_file = data_dir / "test_scenario.csv"
    base_compliance_table = data_dir / "base_compliance_table.csv"
    base_compliance_table_with_producer = data_dir / "base_compliance_table_with_producer.csv"
    compliance_mapping_to_detection_confidence = data_dir / "compliance_mapping_detection_confidence_levels.csv"

    ### PIS Inspection/RBS Calculator Data
    pis_data_updated = val_data_path / 'train.csv'
    #pis_data_updated = shared_ppq_data_path / 'updated_pis_data.csv'  # PIS data
    #pis_data_updated_p = shared_ppq_data_path / 'updated_pis_data.parquet'  # PIS data
    # pis_data_updated = data_dir / "TEST_PIS_SampleQuantity.csv"       # Test data
    # pis_data_updated = data_dir / "Synthetic_PIS_SampleQuantity_test.csv"       # Test data
    # df_pis_data = pd.read_csv(pis_data_updated)

    ### Other data loading
    producer_group_mapping_path = box_paths.disambiguated_producer_table_mapping()
    # Load producer group mapping
    producer_group_mapping = pd.read_csv(producer_group_mapping_path)
    # Load configuration and compliance table
    config = load_configuration(config_file)

    # df_pis_data = create_engineered_features(synth_data=df_pis_data, producer_group_mapping=producer_group_mapping)


    if synthetic_data_generated:
        synth_data = pd.read_csv(data_dir / "Synthetic_Base.csv")
        total_time_seconds = time.time() - start
        time_minutes = total_time_seconds / 60
        time_hours = time_minutes / 60
        time_days = time_hours / 24
        print(f'\nTIMING SUMMARY FOR LOADING DATA PRE-GENERATED SYNTHETIC DATA')
        print(f'   Total Time (seconds): {total_time_seconds}')
        print(f'   Total Time (minutes): {time_minutes}')

        config["consignment"]["input_file"]["file_name"] = str(data_dir  / "Synthetic_Base.csv")
    else:
        ### Synthetic data generation
        historical = False

        synthetic_data_generator = SyntheticConsignmentDataGenerator(config=config,
                                                                     producer_group_mapping=producer_group_mapping,
                                                                     input_data_file=pis_data_updated)

        total_time_seconds = time.time() - start
        time_minutes = total_time_seconds / 60
        time_hours = time_minutes / 60
        time_days = time_hours / 24
        print(f'\nTIMING SUMMARY FOR LOADING DATA (EXECUTING INIT METHOD OF SYNTHETIC CONSIGNMENT GENERATOR)')
        print(f'   Total Time (seconds): {total_time_seconds}')
        print(f'   Total Time (minutes): {time_minutes}')
        print(f'   Total Time (hours): {time_hours}')
        print(f'   Total Time (days): {time_days}')

        if historical:
            df_pis_data = pd.read_csv(pis_data_updated)
            num_consignments_to_simulate = len(df_pis_data["INSPECTION_NUMBER"].unique())
            included_inspection_nums = synthetic_data_generator.input_data["INSPECTION_NUMBER"].sample(n=num_consignments_to_simulate)
            synth_data = synthetic_data_generator.input_data[synthetic_data_generator.input_data["INSPECTION_NUMBER"].isin(included_inspection_nums)]
            synth_data.loc[:, 'Row_ID'] = 'CR-' + (synth_data.index + 1).astype(str)
            synth_out_path = data_dir / "Historical_PIS_SampleQuantity.csv"
            config["consignment"]["input_file"]["file_name"] = str(synth_out_path)
        else:
            print(f'\nStarting Consignment Generation Process of {num_consignments_to_simulate} Requested Consignments...')
            synth_data = synthetic_data_generator.generate_from_input_data(
                n_consignments=num_consignments_to_simulate,
                sampling_method="sequential"
            )
            synth_out_path = data_dir / "Synthetic_Base.csv"
            #synth_out_path = data_dir / "Synthetic_TEST.csv"
            config["consignment"]["input_file"]["file_name"] = str(synth_out_path)

        synth_data.to_parquet(data_dir / "Synthetic_Base_after_generate.parquet", compression='snappy', index=False)
        synth_data.to_csv(data_dir / "Synthetic_Base_after_generate.csv")
        # synth_data.to_parquet(data_dir / "Synthetic_Base_after_generate_TEST.parquet", compression='snappy', index=False)
        # synth_data.to_csv(data_dir / "Synthetic_Base_after_generate_TEST.csv")
        total_time_seconds = time.time() - start
        time_minutes = total_time_seconds / 60
        time_hours = time_minutes / 60
        time_days = time_hours / 24
        print(f'\nTIMING SUMMARY FOR GENERATING SYNTHETIC DATA')
        print(f'   Total Time (seconds): {total_time_seconds}')
        print(f'   Total Time (minutes): {time_minutes}')
        print(f'   Total Time (hours): {time_hours}')
        print(f'   Total Time (days): {time_days}')

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
            output_col="PRODUCER_GROUP_NAME_SHORT",  # or None to overwrite
        )

        # Create a producer_group column
        synth_data['producer_group'] = synth_data['PRODUCER_GROUP_NAME_SHORT']

        #synth_data.to_parquet(data_dir / "Synthetic_Base_TEST.parquet", compression='snappy', index=False)
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

    #############################################################
    ##### TODO: Replace this block with the appropriate data ####
    #############################################################

    ### Generate clarke inputs via input data
    inputs_by_quantity = gen_clarke_model_inputs(df_pis_data)
    #
    # # Run clarke model
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

    compliance_table_dir = default_paths.prediction_model_compliance_dir()

    # Check if directory exists
    if not compliance_table_dir.exists():
        print(f"Directory does not exist: {compliance_table_dir}")
    elif not compliance_table_dir.is_dir():
        print(f"Path is not a directory: {compliance_table_dir}")
    else:
        # Get list of files
        files = [f.name for f in compliance_table_dir.iterdir() if f.is_file()]

        if not files:
            print(f"No compliance tables found found in directory {compliance_table_dir}")
        else:
            print(f"Found {len(files)} file(s)")

            # Loop through file names
            sim_count = 1
            for filename in files:
                clean_filename = filename.replace('.csv', '')
                run_ts = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
                run_dir = compliance_table_dir / f"{clean_filename}_results_{run_ts}"
                run_dir = Path(run_dir)
                run_dir.mkdir(exist_ok=True)
                print(f"\n===============================================================")
                print(f"===============================================================")
                print(f"PROCESSING COMPLIANCE TABLE: {clean_filename}")
                print(f"===============================================================")
                print(f"===============================================================\n")

                # Update scenario table so results are not overwritten
                scenario[f"name"] = f'{clean_filename}'
                scenario[f"consignment name"] = f'{clean_filename}'
                scenario[f"inspection name"] = f'{clean_filename}'

                # Construct full path if needed
                compliance_table_path = compliance_table_dir / filename

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
                compliance_lookup_pkl = default_paths.compliance_dir() / 'compliance_lookup_final.pkl'

                # Now use these throughout your code
                with open(compliance_lookup_pkl, 'wb') as f:
                    pickle.dump(compliance_table, f, protocol=pickle.HIGHEST_PROTOCOL)

                config["inspection"]["compliance_table"]['file_name'] = 'compliance_lookup_final.pkl'

                ##################################################################
                ##################################################################
                ################# END INSPECTION MODULE  #########################
                ##################################################################
                ##################################################################

                temp_compliance_table = pd.read_csv(compliance_table_path)

                # Helper to count "Reference" in a column
                def count_reference(df: pd.DataFrame, col: str) -> int:
                    return (df[col] == "Reference").sum()

                # --- PRODUCER GROUP ---

                if "prod_group_name" in temp_compliance_table.columns:
                    col = "producer_group"
                    comp_col = "prod_group_name"

                    before_ref = count_reference(synth_data, col)

                    # Find values in synth_data that are NOT in temp_compliance_table
                    mask = ~synth_data[col].isin(temp_compliance_table[comp_col])

                    # Collect the values that will be replaced (unique)
                    replaced_values = synth_data.loc[mask, col].unique()

                    # Create a DataFrame for these values
                    replaced_df = pd.DataFrame(replaced_values, columns=["Replaced_Producer_Group"])
                    replaced_df.to_csv(data_dir / "replaced_producer_group_from_prod_group_name.csv", index=False)

                    # Replace those values with "Reference"
                    synth_data.loc[mask, col] = "Reference"

                    after_ref = count_reference(synth_data, col)

                    print(f"[Producer groups vs {comp_col}] 'Reference' count before: {before_ref}, after: {after_ref}")

                    # Check if ALL values in synth_data are in temp_compliance_table
                    all_present = synth_data[col].isin(temp_compliance_table[comp_col]).all()
                    if all_present:
                        print("✓ All producer groups are valid!")
                    else:
                        print("✗ Some producer groups are missing from temp_compliance_table")

                elif "PRODUCER_GROUP_TOP" in temp_compliance_table.columns:
                    col = "producer_group"
                    comp_col = "PRODUCER_GROUP_TOP"

                    before_ref = count_reference(synth_data, col)

                    # Find values in synth_data that are NOT in temp_compliance_table
                    mask = ~synth_data[col].isin(temp_compliance_table[comp_col])

                    # Collect the values that will be replaced (unique)
                    replaced_values = synth_data.loc[mask, col].unique()

                    # Create a DataFrame for these values
                    replaced_df = pd.DataFrame(replaced_values, columns=["Replaced_Producer_Group"])
                    replaced_df.to_csv(data_dir / "replaced_producer_group_from_PRODUCER_GROUP_TOP.csv", index=False)

                    # Replace those values with "Reference"
                    synth_data.loc[mask, col] = "Reference"

                    after_ref = count_reference(synth_data, col)

                    print(f"[Producer groups vs {comp_col}] 'Reference' count before: {before_ref}, after: {after_ref}")

                    # Check if ALL values in synth_data are in temp_compliance_table
                    all_present = synth_data[col].isin(temp_compliance_table[comp_col]).all()
                    if all_present:
                        print("✓ All producer groups are valid!")
                    else:
                        print("✗ Some producer groups are missing from temp_compliance_table")

                # --- IMPORTER NAME ---

                if "IMPORTER_NAME_TOP" in temp_compliance_table.columns:
                    col = "IMPORTER_NAME"
                    comp_col = "IMPORTER_NAME_TOP"

                    before_ref = count_reference(synth_data, col)

                    # Find values in synth_data that are NOT in temp_compliance_table
                    mask = ~synth_data[col].isin(temp_compliance_table[comp_col])

                    # Collect the replaced rows (both columns)
                    replaced_df = synth_data.loc[mask, ["IMPORTER_NAME", "IMPORTER_NAME_RAW"]].drop_duplicates()
                    replaced_df.rename(
                        columns={
                            "IMPORTER_NAME": "Replaced_Importer_Name",
                            "IMPORTER_NAME_RAW": "Raw_Importer_Name",
                        },
                        inplace=True,
                    )
                    replaced_df.to_csv(data_dir / "replaced_importer_names.csv", index=False)

                    # Replace those values with "Reference"
                    synth_data.loc[mask, col] = "Reference"

                    after_ref = count_reference(synth_data, col)

                    print(f"[Importer names vs {comp_col}] 'Reference' count before: {before_ref}, after: {after_ref}")

                    # Check if ALL values in synth_data are in temp_compliance_table
                    all_present = synth_data[col].isin(temp_compliance_table[comp_col]).all()
                    if all_present:
                        print("✓ All importer names are valid!")
                    else:
                        print("✗ Some importer names are missing from temp_compliance_table")

                # if "prod_group_name" in temp_compliance_table.columns:
                #     # Find values in synth_data that are NOT in temp_compliance_table
                #     mask = ~synth_data["producer_group"].isin(temp_compliance_table["prod_group_name"])
                #
                #     # Collect the values that will be replaced (unique)
                #     replaced_values = synth_data.loc[mask, "producer_group"].unique()
                #
                #     # Create a DataFrame for these values
                #     replaced_df = pd.DataFrame(replaced_values, columns=["Replaced_Producer_Group"])
                #
                #     # Write to CSV
                #     replaced_df.to_csv(data_dir / "replaced_producer_group_from_prod_group_name.csv", index=False)
                #
                #     # Replace those values with "Reference"
                #     synth_data.loc[mask, "producer_group"] = "Reference"
                #
                #     # Check if ALL values in synth_data are in temp_compliance_table
                #     all_present = synth_data["producer_group"].isin(temp_compliance_table["prod_group_name"]).all()
                #
                #     if all_present:
                #         print("✓ All producer groups are valid!")
                #     else:
                #         print("✗ Some producer groups are missing from temp_compliance_table")
                # elif "PRODUCER_GROUP_TOP" in temp_compliance_table.columns:
                #     # Find values in synth_data that are NOT in temp_compliance_table
                #     mask = ~synth_data["producer_group"].isin(temp_compliance_table["PRODUCER_GROUP_TOP"])
                #
                #     # Collect the values that will be replaced (unique)
                #     replaced_values = synth_data.loc[mask, "producer_group"].unique()
                #
                #     # Create a DataFrame for these values
                #     replaced_df = pd.DataFrame(replaced_values, columns=["Replaced_Producer_Group"])
                #
                #     # Write to CSV
                #     replaced_df.to_csv(data_dir / "replaced_producer_group_from_PRODUCER_GROUP_TOP.csv", index=False)
                #
                #     # Replace those values with "Reference"
                #     synth_data.loc[mask, "producer_group"] = "Reference"
                #
                #     # Check if ALL values in synth_data are in temp_compliance_table
                #     all_present = synth_data["producer_group"].isin(temp_compliance_table["PRODUCER_GROUP_TOP"]).all()
                #
                #     if all_present:
                #         print("✓ All producer groups are valid!")
                #     else:
                #         print("✗ Some producer groups are missing from temp_compliance_table")
                # if "IMPORTER_NAME_TOP" in temp_compliance_table.columns:
                #     # Find values in synth_data that are NOT in temp_compliance_table
                #     mask = ~synth_data["IMPORTER_NAME"].isin(temp_compliance_table["IMPORTER_NAME_TOP"])
                #
                #     # Collect the replaced rows (both columns)
                #     replaced_df = synth_data.loc[mask, ["IMPORTER_NAME", "IMPORTER_NAME_RAW"]].drop_duplicates()
                #
                #     # Rename columns for clarity
                #     replaced_df.rename(columns={"IMPORTER_NAME": "Replaced_Importer_Name",
                #                                 "IMPORTER_NAME_RAW": "Raw_Importer_Name"}, inplace=True)
                #
                #     # Write to CSV
                #     replaced_df.to_csv(data_dir / "replaced_importer_names.csv", index=False)
                #
                #     # Replace those values with "Reference"
                #     synth_data.loc[mask, "IMPORTER_NAME"] = "Reference"
                #
                #     # Check if ALL values in synth_data are in temp_compliance_table
                #     all_present = synth_data["IMPORTER_NAME"].isin(temp_compliance_table["IMPORTER_NAME_TOP"]).all()
                #
                #     if all_present:
                #         print("✓ All importer names are valid!")
                #     else:
                #         print("✗ Some importer names are missing from temp_compliance_table")


                synth_data.to_parquet(data_dir / "Synthetic_Base_Use.parquet", compression='snappy', index=False)
                synth_data.to_csv(data_dir / "Synthetic_Base_Use.csv", index=False)

                config["consignment"]["input_file"]["file_name"] = "development_files/slippage_data/Synthetic_Base_Use.csv"

                # Run one scenario analysis simulation
                detailed_bool = True
                scenario_results_raw = run_scenarios(
                    config=config,
                    scenario_table=scenarios,
                    seed=42,
                    num_simulations=num_replications,            # Only one simulation
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
                # output_dir = Path("output")
                # output_dir.mkdir(exist_ok=True)

                # Save results to CSV
                results_df = save_scenario_result_to_pandas(scenario_results,
                                                            config_columns=config_columns,
                                                            result_columns=result_columns)

                MAX_PATH_LENGTH = 200

                clean_filename = filename.replace('.csv', '')
                full_path = run_dir / f"slippage_{clean_filename}.csv"

                if len(str(full_path)) > MAX_PATH_LENGTH:
                    # Calculate available space
                    base_path_length = len(str(run_dir)) + len("slippage_") + len(".csv") + 1  # +1 for separator
                    max_name_length = MAX_PATH_LENGTH - base_path_length - 10  # Buffer

                    # Create a hash of the original filename for uniqueness
                    hash_suffix = hashlib.md5(clean_filename.encode()).hexdigest()[:8]

                    # Truncate and add hash
                    truncated_name = clean_filename[:max(0, max_name_length - 9)]  # -9 for underscore + hash
                    clean_filename = f"{truncated_name}_{hash_suffix}"

                    full_path = run_dir / f"slippage_{clean_filename}.csv"

                results_df.to_csv(full_path, index=False)
                print(f"Results saved to {full_path}")


                if detailed_bool:
                    inspection_unit_records = []
                    for details, _, scenario_config in scenario_results_raw:
                        if len(details) >= 3:
                            for row in details[2]:
                                row_with_scenario = dict(row)
                                row_with_scenario["scenario_name"] = scenario_config.get("name")
                                inspection_unit_records.append(row_with_scenario)
                    if inspection_unit_records:

                        clean_filename = filename.replace('.csv', '')
                        full_path = run_dir / f"iu_detection_records_{clean_filename}.csv"

                        if len(str(full_path)) > MAX_PATH_LENGTH:
                            # Calculate available space
                            base_path_length = len(str(run_dir)) + len("iu_detection_records_") + len(
                                ".csv") + 1  # +1 for separator
                            max_name_length = MAX_PATH_LENGTH - base_path_length - 10  # Buffer

                            # Create a hash of the original filename for uniqueness
                            hash_suffix = hashlib.md5(clean_filename.encode()).hexdigest()[:8]

                            # Truncate and add hash
                            truncated_name = clean_filename[:max(0, max_name_length - 9)]  # -9 for underscore + hash
                            clean_filename = f"{truncated_name}_{hash_suffix}"

                            full_path = run_dir / f"iu_detection_records_{clean_filename}.csv"
                        save_inspection_unit_detection_records_to_csv(
                            inspection_unit_records,
                            full_path,
                        )
                        print(f"Results saved to output/inspection_unit_detection_records_{filename}.csv")
                print('')
    total_time_seconds = time.time() - start
    time_minutes = total_time_seconds / 60
    time_hours = time_minutes / 60
    time_days = time_hours / 24
    print(f'\nTIMING SUMMARY OVERALL')
    print(f'   Total Time (seconds): {total_time_seconds}')
    print(f'   Total Time (minutes): {time_minutes}')
    print(f'   Total Time (hours): {time_hours}')
    print(f'   Total Time (days): {time_days}')

if __name__ == "__main__":
    main()
