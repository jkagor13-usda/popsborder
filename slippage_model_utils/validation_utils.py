import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Union
from datetime import datetime
import warnings
import re
from slippage_model_utils.paths import DefaultPaths
from scipy import stats


def find_latest_simulation_base_path(output_dir: Path, base_name: str = "pops_border_scenario_data") -> Path:
    """
    Find the most recent simulation base path based on timestamp in directory name.

    Parameters:
    -----------
    output_dir : Path
        The main output directory
    base_name : str
        Base name of the simulation directories (default: 'pops_border_scenario_data')

    Returns:
    --------
    Path
        Path to the most recent simulation base directory
    """
    # Pattern to match simulation directories with timestamp
    pattern = re.compile(rf"{base_name}_(\d{{2}})_(\d{{2}})_(\d{{4}})_(\d{{2}})_(\d{{2}})_(\d{{2}})")

    simulation_dirs = []

    # Find all matching directories
    for item in output_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                # Extract timestamp components: month, day, year, hour, minute, second
                month, day, year, hour, minute, second = match.groups()

                # Create datetime object for comparison
                try:
                    timestamp = datetime(
                        int(year), int(month), int(day),
                        int(hour), int(minute), int(second)
                    )
                    simulation_dirs.append((timestamp, item))
                except ValueError as e:
                    warnings.warn(f"Invalid timestamp in directory name {item.name}: {e}")

    if not simulation_dirs:
        raise FileNotFoundError(f"No simulation directories found matching pattern '{base_name}_*' in {output_dir}")

    # Sort by timestamp and return the most recent
    simulation_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_dir = simulation_dirs[0][1]

    print(f"Found {len(simulation_dirs)} simulation run(s)")
    print(f"Selected most recent: {latest_dir.name}")

    return latest_dir


def get_simulation_base_path(
        simulation_output_path: str | Path,
        simulation_base_path: str = "latest"
) -> Path:
    """
    Get the simulation base path, either by name or finding the latest.

    Parameters:
    -----------
    simulation_output_path : str | Path
        The main output directory
    simulation_base_path : str | Path
        Either 'latest' to auto-select most recent, or specific directory name

    Returns:
    --------
    Path
        Full path to the simulation base directory
    """
    output_dir = Path(simulation_output_path)

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    if isinstance(simulation_base_path, str) and simulation_base_path.lower() == "latest":
        return find_latest_simulation_base_path(output_dir)
    else:
        # Use specified directory
        base_path = output_dir / simulation_base_path
        if not base_path.exists():
            raise FileNotFoundError(f"Simulation base path does not exist: {base_path}")
        return base_path


def calculate_action_rates_by_scenario(
        ground_truth_path: str | Path,
        simulation_output_path: str | Path,
        scenarios: List[str],
        num_replications: int,
        filter_fields: List[str],
        simulation_base_path: str = "latest",
        output_file: str = "synthetic_commodity_line_results_data.csv",
        practical_equivalence_threshold: float = 0.01
) -> Dict[str, pd.DataFrame]:
    """
    Calculate action rates from simulation data filtered by ground truth criteria.

    Parameters:
    -----------
    ground_truth_path : str
        Path to the ground truth CSV file
    simulation_output_path : str | Path
        Main output directory (the 'output' folder)
    scenarios : List[str]
        List of scenario names (e.g., ['scenario1', 'scenario2', ...])
    num_replications : int
        Number of replications per scenario
    filter_fields : List[str]
        Fields from ground truth to use for filtering
    simulation_base_path : str, optional
        Either 'latest' to auto-select most recent run, or specific directory name
        (default: 'latest')
    output_file : str, optional
        Name of the simulation output file (default: 'synthetic_commodity_line_results_data.csv')

    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with scenario names as keys and DataFrames containing analysis results
    """

    # Get the actual simulation base path
    base_path = get_simulation_base_path(simulation_output_path, simulation_base_path)
    print(f"\nUsing simulation base path: {base_path}")

    # Load ground truth data
    print(f"Loading ground truth data from {ground_truth_path}")
    ground_truth = pd.read_csv(ground_truth_path)

    # Convert all column names to lowercase
    ground_truth.columns = ground_truth.columns.str.lower()

    # Convert filter_fields to lowercase for consistency
    filter_fields_lower = [field.lower() for field in filter_fields]

    # Validate required columns
    required_cols = ['inspection_number', 'action'] + filter_fields_lower
    missing_cols = [col for col in required_cols if col not in ground_truth.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in ground truth: {missing_cols}")

    # Get unique combinations of filter fields and their inspection numbers
    print(f"Creating filter combinations based on: {filter_fields_lower}")
    filter_combinations = ground_truth[filter_fields_lower + ['inspection_number', 'action']].copy()

    # Group by filter fields to get unique combinations
    grouped = filter_combinations.groupby(filter_fields_lower)

    results = {}

    for scenario in scenarios:
        print(f"\nProcessing {scenario}...")
        scenario_results = []

        for combination_values, group_data in grouped:
            # Create a dictionary for the current combination
            combination_dict = dict(
                zip(filter_fields_lower, combination_values if len(filter_fields_lower) > 1 else [combination_values]))

            # Get unique inspection numbers for this combination
            unique_inspections = group_data['inspection_number'].unique()

            # Calculate ground truth action rate for this combination
            ground_truth_subset = ground_truth[
                ground_truth['inspection_number'].isin(unique_inspections)
            ]
            gt_action_rate = ground_truth_subset['action'].mean()

            # Initialize list to store action rates from each replication
            replication_action_rates = []

            # Process each replication (starting from rep_0)
            for rep in range(num_replications):
                # Construct path to simulation output file
                sim_file_path = base_path / scenario / f"rep_{rep}" / output_file

                try:
                    # Load simulation data
                    sim_data = pd.read_csv(sim_file_path)

                    # Validate required columns
                    if 'inspection_number' not in sim_data.columns or 'action' not in sim_data.columns:
                        warnings.warn(f"Missing columns in {sim_file_path}")
                        replication_action_rates.append(np.nan)
                        continue

                    # Filter simulation data to matching inspection numbers
                    sim_subset = sim_data[sim_data['inspection_number'].isin(unique_inspections)]

                    if len(sim_subset) > 0:
                        rep_action_rate = sim_subset['action'].mean()
                        replication_action_rates.append(rep_action_rate)
                    else:
                        warnings.warn(
                            f"No matching inspection numbers in {sim_file_path} for combination {combination_dict}")
                        replication_action_rates.append(np.nan)

                except FileNotFoundError:
                    warnings.warn(f"File not found: {sim_file_path}")
                    replication_action_rates.append(np.nan)
                except Exception as e:
                    warnings.warn(f"Error processing {sim_file_path}: {str(e)}")
                    replication_action_rates.append(np.nan)

            # Calculate statistics across replications
            valid_rates = [r for r in replication_action_rates if not np.isnan(r)]

            # Perform two-sided t-test if we have valid rates
            if len(valid_rates) > 1:
                # Calculate mean and std
                mean_rate = np.mean(valid_rates)
                std_rate = np.std(valid_rates, ddof=1)  # Use sample std deviation

                # Check if rates are essentially identical (no variance)
                if std_rate < 1e-10:  # Very small threshold for numerical stability
                    # If all replications are the same, check if they match ground truth
                    if abs(mean_rate - gt_action_rate) < 1e-10:
                        statistically_same = 1
                        p_value = 1.0  # Perfect match, maximum p-value
                    else:
                        statistically_same = 0
                        p_value = 0.0  # Clear difference, minimum p-value
                else:
                    # Normal t-test when there is variance
                    t_statistic, p_value = stats.ttest_1samp(valid_rates, gt_action_rate)
                    # If p-value > 0.05, we fail to reject null (they are statistically the same)
                    statistically_same = 1 if p_value > 0.05 else 0

                # Also check practical equivalence (e.g., within 5% or 0.01 absolute difference)
                absolute_diff = abs(mean_rate - gt_action_rate)

                # Consider practically equivalent if within threshold
                practically_equivalent = 1 if (absolute_diff < practical_equivalence_threshold) else 0

            elif len(valid_rates) == 1:
                # Can't perform t-test with only one observation
                # Check if the single value matches ground truth
                if abs(valid_rates[0] - gt_action_rate) < 1e-10:
                    statistically_same = 1
                    p_value = 1.0

                    # Consider practically equivalent if within threshold
                    absolute_diff = abs(valid_rates[0] - gt_action_rate)
                    # Consider practically equivalent if within threshold
                    practically_equivalent = 1 if (absolute_diff < 0.01) else 0

                else:
                    statistically_same = np.nan  # Insufficient data for reliable test
                    p_value = np.nan
                    practically_equivalent = np.nan
                    absolute_diff = np.nan
            else:
                # No valid rates
                statistically_same = np.nan
                p_value = np.nan
                practically_equivalent = np.nan
                absolute_diff = np.nan



            result_row = {
                **combination_dict,
                'ground_truth_action_rate': gt_action_rate,
                'mean_simulation_action_rate': np.mean(valid_rates) if valid_rates else np.nan,
                'std_simulation_action_rate': np.std(valid_rates) if valid_rates else np.nan,
                'num_inspection_numbers': len(unique_inspections),
                'num_valid_replications': len(valid_rates),
                'simulation_action_rate_statistically_same_as_ground_truth': statistically_same,
                'p_value': p_value,
                'practically_equivalent': practically_equivalent,
                'practical_threshold': practical_equivalence_threshold,
                'action_rate_abs_diff': absolute_diff
            }

            # Add individual replication rates (starting from rep_0)
            for rep_idx, rate in enumerate(replication_action_rates):
                result_row[f'rep_{rep_idx}_action_rate'] = rate

            scenario_results.append(result_row)

        # Convert to DataFrame
        results[scenario] = pd.DataFrame(scenario_results)
        print(f"Completed {scenario}: {len(scenario_results)} filter combinations processed")

    return results


def list_available_simulation_runs(
        simulation_output_path: str | Path,
        base_name: str = "pops_border_scenario_data"
) -> List[Dict[str, Union[str, datetime]]]:
    """
    List all available simulation runs in the output directory.

    Parameters:
    -----------
    simulation_output_path : str | Path
        The main output directory
    base_name : str
        Base name of the simulation directories

    Returns:
    --------
    List[Dict]
        List of dictionaries containing 'name' and 'timestamp' for each run
    """
    output_dir = Path(simulation_output_path)
    pattern = re.compile(rf"{base_name}_(\d{{2}})_(\d{{2}})_(\d{{4}})_(\d{{2}})_(\d{{2}})_(\d{{2}})")

    runs = []

    for item in output_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                month, day, year, hour, minute, second = match.groups()
                try:
                    timestamp = datetime(
                        int(year), int(month), int(day),
                        int(hour), int(minute), int(second)
                    )
                    runs.append({
                        'name': item.name,
                        'timestamp': timestamp,
                        'path': item
                    })
                except ValueError:
                    pass

    # Sort by timestamp (most recent first)
    runs.sort(key=lambda x: x['timestamp'], reverse=True)

    return runs


def save_results(results: Dict[str, pd.DataFrame], output_dir: Path = DefaultPaths().validation_output_dir()) -> None:
    """
    Save analysis results to CSV files.

    Parameters:
    -----------
    results : Dict[str, pd.DataFrame]
        Dictionary of results from calculate_action_rates_by_scenario
    output_dir : str
        Directory to save output files
    """
    output_dir.mkdir(exist_ok=True)

    for scenario, df in results.items():
        output_file = output_dir / f"{scenario}_action_rates_validation.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved results for {scenario} to {output_file}")




def summarize_statistical_comparison(
        results: Dict[str, pd.DataFrame],
        filter_fields: List[str],
        ground_truth_path: str | Path,
        output_dir: Path = DefaultPaths().validation_output_dir()
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Create summary statistics for statistical and practical comparison between simulation and ground truth.

    Parameters:
    -----------
    results : Dict[str, pd.DataFrame]
        Dictionary of results from calculate_action_rates_by_scenario
    filter_fields : List[str]
        The filter fields used in the analysis
    ground_truth_path : str
        Path to the ground truth CSV file (needed to count total rows)
    output_dir : str
        Directory to save output files

    Returns:
    --------
    Dict[str, Dict[str, pd.DataFrame]]
        Nested dictionary with scenario names as keys, each containing:
        - 'overall_summary': DataFrame with overall statistics (both statistical and practical)
        - 'statistically_different_combinations': DataFrame with combinations where rates differ statistically
        - 'practically_different_combinations': DataFrame with combinations where rates differ practically
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load ground truth data for row counting
    print(f"Loading ground truth data from {ground_truth_path}")
    ground_truth = pd.read_csv(ground_truth_path)
    ground_truth.columns = ground_truth.columns.str.lower()

    # Convert filter_fields to lowercase for consistency
    filter_fields_lower = [field.lower() for field in filter_fields]

    all_summaries = {}

    for scenario, df in results.items():
        print(f"\nProcessing summary for {scenario}...")

        # Filter out rows where tests couldn't be performed (NaN values)
        df_valid_stat = df[df['simulation_action_rate_statistically_same_as_ground_truth'].notna()].copy()
        df_valid_pract = df[df['practically_equivalent'].notna()].copy()

        # --- Calculate statistics for both tests ---
        # Statistical test counts
        stat_same_count = (df_valid_stat['simulation_action_rate_statistically_same_as_ground_truth'] == 1).sum()
        stat_different_count = (df_valid_stat['simulation_action_rate_statistically_same_as_ground_truth'] == 0).sum()
        stat_total_valid = stat_same_count + stat_different_count

        # Practical test counts
        pract_same_count = (df_valid_pract['practically_equivalent'] == 1).sum()
        pract_different_count = (df_valid_pract['practically_equivalent'] == 0).sum()
        pract_total_valid = pract_same_count + pract_different_count

        # Helper function to calculate metrics for a given mask and dataframe
        def calculate_metrics(df_subset, mask):
            inspection_nums = df_subset.loc[mask, 'num_inspection_numbers'].sum()

            # Calculate rows in ground truth
            rows = 0
            for _, row in df_subset[mask].iterrows():
                filter_condition = pd.Series([True] * len(ground_truth))
                for field in filter_fields_lower:
                    if field in ground_truth.columns:
                        filter_condition &= (ground_truth[field] == row[field])
                rows += filter_condition.sum()

            return inspection_nums, rows

        # Calculate metrics for statistical test
        stat_same_mask = df_valid_stat['simulation_action_rate_statistically_same_as_ground_truth'] == 1
        stat_different_mask = df_valid_stat['simulation_action_rate_statistically_same_as_ground_truth'] == 0

        stat_same_inspection_nums, stat_same_rows = calculate_metrics(df_valid_stat, stat_same_mask)
        stat_different_inspection_nums, stat_different_rows = calculate_metrics(df_valid_stat, stat_different_mask)
        stat_total_inspection_nums = stat_same_inspection_nums + stat_different_inspection_nums
        stat_total_rows = stat_same_rows + stat_different_rows

        # Calculate metrics for practical test
        pract_same_mask = df_valid_pract['practically_equivalent'] == 1
        pract_different_mask = df_valid_pract['practically_equivalent'] == 0

        pract_same_inspection_nums, pract_same_rows = calculate_metrics(df_valid_pract, pract_same_mask)
        pract_different_inspection_nums, pract_different_rows = calculate_metrics(df_valid_pract, pract_different_mask)
        pract_total_inspection_nums = pract_same_inspection_nums + pract_different_inspection_nums
        pract_total_rows = pract_same_rows + pract_different_rows

        # Create overall summary with both tests
        overall_summary = pd.DataFrame({
            'comparison_result': [
                'Statistically Same',
                'Statistically Different',
                'Total Valid (Statistical)',
                'Practically Same',
                'Practically Different',
                'Total Valid (Practical)'
            ],
            'num_combinations': [
                stat_same_count,
                stat_different_count,
                stat_total_valid,
                pract_same_count,
                pract_different_count,
                pract_total_valid
            ],
            'percentage_of_combinations': [
                (stat_same_count / stat_total_valid * 100) if stat_total_valid > 0 else 0,
                (stat_different_count / stat_total_valid * 100) if stat_total_valid > 0 else 0,
                100.0,
                (pract_same_count / pract_total_valid * 100) if pract_total_valid > 0 else 0,
                (pract_different_count / pract_total_valid * 100) if pract_total_valid > 0 else 0,
                100.0
            ],
            'num_inspection_numbers': [
                stat_same_inspection_nums,
                stat_different_inspection_nums,
                stat_total_inspection_nums,
                pract_same_inspection_nums,
                pract_different_inspection_nums,
                pract_total_inspection_nums
            ],
            'percentage_of_inspection_numbers': [
                (stat_same_inspection_nums / stat_total_inspection_nums * 100) if stat_total_inspection_nums > 0 else 0,
                (
                            stat_different_inspection_nums / stat_total_inspection_nums * 100) if stat_total_inspection_nums > 0 else 0,
                100.0,
                (
                            pract_same_inspection_nums / pract_total_inspection_nums * 100) if pract_total_inspection_nums > 0 else 0,
                (
                            pract_different_inspection_nums / pract_total_inspection_nums * 100) if pract_total_inspection_nums > 0 else 0,
                100.0
            ],
            'num_ground_truth_rows': [
                stat_same_rows,
                stat_different_rows,
                stat_total_rows,
                pract_same_rows,
                pract_different_rows,
                pract_total_rows
            ],
            'percentage_of_ground_truth_rows': [
                (stat_same_rows / stat_total_rows * 100) if stat_total_rows > 0 else 0,
                (stat_different_rows / stat_total_rows * 100) if stat_total_rows > 0 else 0,
                100.0,
                (pract_same_rows / pract_total_rows * 100) if pract_total_rows > 0 else 0,
                (pract_different_rows / pract_total_rows * 100) if pract_total_rows > 0 else 0,
                100.0
            ]
        })

        # Add scenario information
        overall_summary.insert(0, 'scenario', scenario)

        # --- Statistically Different Combinations ---
        stat_different_combinations = df_valid_stat[
            df_valid_stat['simulation_action_rate_statistically_same_as_ground_truth'] == 0
            ].copy()

        # Add row count for each statistically different combination
        stat_row_counts = []
        for _, row in stat_different_combinations.iterrows():
            filter_condition = pd.Series([True] * len(ground_truth))
            for field in filter_fields_lower:
                if field in ground_truth.columns:
                    filter_condition &= (ground_truth[field] == row[field])
            stat_row_counts.append(filter_condition.sum())

        # Select relevant columns for statistically different combinations
        stat_columns_to_include = (
                filter_fields_lower +
                [
                    'ground_truth_action_rate',
                    'mean_simulation_action_rate',
                    'std_simulation_action_rate',
                    'num_inspection_numbers',
                    'num_valid_replications',
                    'p_value',
                    'action_rate_abs_diff',
                    'practically_equivalent',
                    'practical_threshold'
                ]
        )

        stat_columns_to_include = [col for col in stat_columns_to_include if col in stat_different_combinations.columns]
        stat_different_combinations = stat_different_combinations[stat_columns_to_include].copy()
        stat_different_combinations['num_ground_truth_rows'] = stat_row_counts

        # Calculate relative difference
        stat_different_combinations['relative_difference_pct'] = (
                (stat_different_combinations['action_rate_abs_diff'] /
                 stat_different_combinations['ground_truth_action_rate']) * 100
        )

        # Sort by absolute difference
        stat_different_combinations = stat_different_combinations.sort_values(
            'action_rate_abs_diff',
            ascending=False
        ).reset_index(drop=True)

        stat_different_combinations.insert(0, 'scenario', scenario)

        # --- Practically Different Combinations ---
        pract_different_combinations = df_valid_pract[
            df_valid_pract['practically_equivalent'] == 0
            ].copy()

        # Add row count for each practically different combination
        pract_row_counts = []
        for _, row in pract_different_combinations.iterrows():
            filter_condition = pd.Series([True] * len(ground_truth))
            for field in filter_fields_lower:
                if field in ground_truth.columns:
                    filter_condition &= (ground_truth[field] == row[field])
            pract_row_counts.append(filter_condition.sum())

        # Select relevant columns for practically different combinations
        pract_columns_to_include = (
                filter_fields_lower +
                [
                    'ground_truth_action_rate',
                    'mean_simulation_action_rate',
                    'std_simulation_action_rate',
                    'num_inspection_numbers',
                    'num_valid_replications',
                    'action_rate_abs_diff',
                    'practical_threshold',
                    'p_value',
                    'simulation_action_rate_statistically_same_as_ground_truth'
                ]
        )

        pract_columns_to_include = [col for col in pract_columns_to_include if
                                    col in pract_different_combinations.columns]
        pract_different_combinations = pract_different_combinations[pract_columns_to_include].copy()
        pract_different_combinations['num_ground_truth_rows'] = pract_row_counts

        # Calculate relative difference
        pract_different_combinations['relative_difference_pct'] = (
                (pract_different_combinations['action_rate_abs_diff'] /
                 pract_different_combinations['ground_truth_action_rate']) * 100
        )

        # Sort by absolute difference
        pract_different_combinations = pract_different_combinations.sort_values(
            'action_rate_abs_diff',
            ascending=False
        ).reset_index(drop=True)

        pract_different_combinations.insert(0, 'scenario', scenario)

        # Store results
        all_summaries[scenario] = {
            'overall_summary': overall_summary,
            'statistically_different_combinations': stat_different_combinations,
            'practically_different_combinations': pract_different_combinations
        }

        # Save to CSV
        overall_file = output_path / f"{scenario}_overall_summary.csv"
        stat_different_file = output_path / f"{scenario}_statistically_different_combinations.csv"
        pract_different_file = output_path / f"{scenario}_practically_different_combinations.csv"

        overall_summary.to_csv(overall_file, index=False)
        stat_different_combinations.to_csv(stat_different_file, index=False)
        pract_different_combinations.to_csv(pract_different_file, index=False)

        print(f"Saved overall summary to: {overall_file}")
        print(f"Saved statistically different combinations to: {stat_different_file}")
        print(f"Saved practically different combinations to: {pract_different_file}")
        print(
            f"  Statistical Test: {stat_same_count} same, {stat_different_count} different ({stat_different_rows:,} ground truth rows)")
        print(
            f"  Practical Test: {pract_same_count} same, {pract_different_count} different ({pract_different_rows:,} ground truth rows)")

    # Create combined summaries across all scenarios
    if len(results) > 1:
        # Combine overall summaries
        combined_overall = pd.concat(
            [summary['overall_summary'] for summary in all_summaries.values()],
            ignore_index=True
        )
        combined_overall_file = output_path / "all_scenarios_overall_summary.csv"
        combined_overall.to_csv(combined_overall_file, index=False)
        print(f"\nSaved combined overall summary to: {combined_overall_file}")

        # Combine statistically different combinations
        combined_stat_different = pd.concat(
            [summary['statistically_different_combinations'] for summary in all_summaries.values()],
            ignore_index=True
        )
        combined_stat_different_file = output_path / "all_scenarios_statistically_different_combinations.csv"
        combined_stat_different.to_csv(combined_stat_different_file, index=False)
        print(f"Saved combined statistically different combinations to: {combined_stat_different_file}")

        # Combine practically different combinations
        combined_pract_different = pd.concat(
            [summary['practically_different_combinations'] for summary in all_summaries.values()],
            ignore_index=True
        )
        combined_pract_different_file = output_path / "all_scenarios_practically_different_combinations.csv"
        combined_pract_different.to_csv(combined_pract_different_file, index=False)
        print(f"Saved combined practically different combinations to: {combined_pract_different_file}")

    return all_summaries





