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
        output_file: str = "synthetic_commodity_line_results_data.csv"
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
                from scipy import stats
                # Two-sided one-sample t-test: H0: mean of valid_rates == ground_truth_action_rate
                t_statistic, p_value = stats.ttest_1samp(valid_rates, gt_action_rate)
                # If p-value > 0.05, we fail to reject null (they are statistically the same)
                statistically_same = 1 if p_value > 0.05 else 0
            elif len(valid_rates) == 1:
                # Can't perform t-test with only one observation
                statistically_same = np.nan
                p_value = np.nan
            else:
                # No valid rates
                statistically_same = np.nan
                p_value = np.nan

            result_row = {
                **combination_dict,
                'ground_truth_action_rate': gt_action_rate,
                'mean_simulation_action_rate': np.mean(valid_rates) if valid_rates else np.nan,
                'std_simulation_action_rate': np.std(valid_rates) if valid_rates else np.nan,
                'num_inspection_numbers': len(unique_inspections),
                'num_valid_replications': len(valid_rates),
                'simulation_action_rate_statistically_same_as_ground_truth': statistically_same,
                'p_value': p_value
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
    Create summary statistics for statistical comparison between simulation and ground truth.

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
        - 'overall_summary': DataFrame with overall statistics
        - 'different_combinations': DataFrame with combinations where rates differ
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

        # Filter out rows where statistical test couldn't be performed (NaN values)
        df_valid = df[df['simulation_action_rate_statistically_same_as_ground_truth'].notna()].copy()

        # --- Summary 1: Overall Statistics ---
        same_count = (df_valid['simulation_action_rate_statistically_same_as_ground_truth'] == 1).sum()
        different_count = (df_valid['simulation_action_rate_statistically_same_as_ground_truth'] == 0).sum()
        total_valid = same_count + different_count

        # Calculate inspection numbers for each category
        same_mask = df_valid['simulation_action_rate_statistically_same_as_ground_truth'] == 1
        different_mask = df_valid['simulation_action_rate_statistically_same_as_ground_truth'] == 0

        same_inspection_numbers = df_valid.loc[same_mask, 'num_inspection_numbers'].sum()
        different_inspection_numbers = df_valid.loc[different_mask, 'num_inspection_numbers'].sum()
        total_inspection_numbers = same_inspection_numbers + different_inspection_numbers

        # Calculate number of rows in ground truth for each category
        same_rows = 0
        different_rows = 0

        for _, row in df_valid.iterrows():
            # Build filter condition for this combination
            filter_condition = pd.Series([True] * len(ground_truth))
            for field in filter_fields_lower:
                if field in ground_truth.columns:
                    filter_condition &= (ground_truth[field] == row[field])

            # Count rows for this combination
            num_rows = filter_condition.sum()

            # Add to appropriate category
            if row['simulation_action_rate_statistically_same_as_ground_truth'] == 1:
                same_rows += num_rows
            else:
                different_rows += num_rows

        total_rows = same_rows + different_rows

        overall_summary = pd.DataFrame({
            'comparison_result': ['Statistically Same', 'Statistically Different', 'Total Valid'],
            'num_combinations': [same_count, different_count, total_valid],
            'percentage_of_combinations': [
                (same_count / total_valid * 100) if total_valid > 0 else 0,
                (different_count / total_valid * 100) if total_valid > 0 else 0,
                100.0
            ],
            'num_inspection_numbers': [
                same_inspection_numbers,
                different_inspection_numbers,
                total_inspection_numbers
            ],
            'percentage_of_inspection_numbers': [
                (same_inspection_numbers / total_inspection_numbers * 100) if total_inspection_numbers > 0 else 0,
                (different_inspection_numbers / total_inspection_numbers * 100) if total_inspection_numbers > 0 else 0,
                100.0
            ],
            'num_ground_truth_rows': [
                same_rows,
                different_rows,
                total_rows
            ],
            'percentage_of_ground_truth_rows': [
                (same_rows / total_rows * 100) if total_rows > 0 else 0,
                (different_rows / total_rows * 100) if total_rows > 0 else 0,
                100.0
            ]
        })

        # Add scenario information
        overall_summary.insert(0, 'scenario', scenario)

        # --- Summary 2: Combinations Where Rates are Statistically Different ---
        different_combinations = df_valid[
            df_valid['simulation_action_rate_statistically_same_as_ground_truth'] == 0
            ].copy()

        # Add row count for each different combination
        row_counts = []
        for _, row in different_combinations.iterrows():
            filter_condition = pd.Series([True] * len(ground_truth))
            for field in filter_fields_lower:
                if field in ground_truth.columns:
                    filter_condition &= (ground_truth[field] == row[field])
            row_counts.append(filter_condition.sum())

        # Select relevant columns for the detailed view
        columns_to_include = (
                filter_fields_lower +
                [
                    'ground_truth_action_rate',
                    'mean_simulation_action_rate',
                    'std_simulation_action_rate',
                    'num_inspection_numbers',
                    'num_valid_replications',
                    'p_value'
                ]
        )

        # Only include columns that exist in the dataframe
        columns_to_include = [col for col in columns_to_include if col in different_combinations.columns]
        different_combinations = different_combinations[columns_to_include].copy()

        # Add row count column
        different_combinations['num_ground_truth_rows'] = row_counts

        # Calculate absolute and relative differences
        different_combinations['absolute_difference'] = (
                different_combinations['mean_simulation_action_rate'] -
                different_combinations['ground_truth_action_rate']
        )
        different_combinations['relative_difference_pct'] = (
                (different_combinations['absolute_difference'] /
                 different_combinations['ground_truth_action_rate']) * 100
        )

        # Sort by absolute difference (descending)
        different_combinations = different_combinations.sort_values(
            'absolute_difference',
            ascending=False,
            key=abs
        ).reset_index(drop=True)

        # Add scenario information
        different_combinations.insert(0, 'scenario', scenario)

        # Store results
        all_summaries[scenario] = {
            'overall_summary': overall_summary,
            'different_combinations': different_combinations
        }

        # Save to CSV
        overall_file = output_path / f"{scenario}_overall_summary.csv"
        different_file = output_path / f"{scenario}_statistically_different_combinations.csv"

        overall_summary.to_csv(overall_file, index=False)
        different_combinations.to_csv(different_file, index=False)

        print(f"Saved overall summary to: {overall_file}")
        print(f"Saved different combinations to: {different_file}")
        print(f"  - {same_count} combinations statistically same ({same_rows:,} ground truth rows)")
        print(f"  - {different_count} combinations statistically different ({different_rows:,} ground truth rows)")

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

        # Combine different combinations
        combined_different = pd.concat(
            [summary['different_combinations'] for summary in all_summaries.values()],
            ignore_index=True
        )
        combined_different_file = output_path / "all_scenarios_statistically_different_combinations.csv"
        combined_different.to_csv(combined_different_file, index=False)
        print(f"Saved combined different combinations to: {combined_different_file}")

    return all_summaries










