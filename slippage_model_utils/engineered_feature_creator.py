# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC
from slippage_model_utils.r_script_wrapper import RVariableCreator
import pandas as pd
from typing import Dict, Optional


def build_shortest_name_lookup(producer_group_mapping: pd.DataFrame) -> Dict[str, str]:
    """Build a lookup from group ID to shortest name in a mapping table.

    For each unique value in ``producer_group_mapping["group"]``, this function
    finds the shortest corresponding ``"name"`` string (by character length,
    breaking ties using alphabetical order) and returns a mapping.

    Args:
        producer_group_mapping: DataFrame with at least ``"name"`` and
            ``"group"`` columns.

    Returns:
        Dictionary of the form ``{group_value_as_str: shortest_name}``.

    Raises:
        ValueError: If required columns are missing from
            ``producer_group_mapping``.
    """
    # Ensure required columns exist
    required = {"name", "group"}
    missing = required - set(producer_group_mapping.columns)
    if missing:
        raise ValueError(f"producer_group_mapping missing columns: {missing}")

    # Drop NA in required columns
    pgm = producer_group_mapping.dropna(subset=["name", "group"]).copy()

    # Coerce 'group' to string to standardize key type
    pgm["group_str"] = pgm["group"].astype(str)

    # Compute length of name and pick shortest per group_str
    shortest = (
        pgm.assign(name_len=pgm["name"].astype(str).str.len())
           .sort_values(["group_str", "name_len", "name"])
           .drop_duplicates(subset=["group_str"], keep="first")
    )

    # Build mapping: group_str -> name
    return dict(zip(shortest["group_str"], shortest["name"]))


def map_group_to_shortest_name(
    synth_data: pd.DataFrame,
    group_col: str,
    producer_group_mapping: pd.DataFrame,
    output_col: str = None,
    default_value: str = "Reference",
) -> pd.DataFrame:
    """Map group IDs to the shortest name for that group.

    Group IDs in ``synth_data[group_col]`` are mapped to the shortest name
    per group based on ``producer_group_mapping``. All group values are
    coerced to strings for alignment, and missing mappings are filled with a
    default label.

    Args:
        synth_data: DataFrame containing a column of group IDs (e.g.,
            ``"PRODUCER_GROUP_TOP"``).
        group_col: Name of the column in ``synth_data`` that contains group
            values (numeric or string).
        producer_group_mapping: DataFrame with ``"name"`` and ``"group"``
            columns used to derive shortest names per group.
        output_col: Name of the output column to store mapped names. If None,
            ``group_col`` is overwritten in-place.
        default_value: Value used for group IDs not found in the mapping
            (e.g., ``"Reference"``).

    Returns:
        The input ``synth_data`` with the mapped column added or overwritten.

    Raises:
        ValueError: If ``group_col`` is not found in ``synth_data``.
    """
    if group_col not in synth_data.columns:
        raise ValueError(f"{group_col} not found in synth_data")

    # Build lookup {group_str -> shortest name}
    group_to_shortest_name = build_shortest_name_lookup(producer_group_mapping)

    # Determine target column name
    if output_col is None:
        output_col = group_col

    # Coerce group_col to string to align with mapping keys
    group_values_str = synth_data[group_col].astype(str)

    # Map; values not found will become NaN
    mapped = group_values_str.map(group_to_shortest_name)

    # Fill missing mappings with default_value (e.g. "Reference")
    mapped = mapped.fillna(default_value)

    synth_data[output_col] = mapped

    return synth_data


def create_engineered_features(
    synth_data: pd.DataFrame = None,
    dt_train: pd.DataFrame = None,
    producer_group_mapping: pd.DataFrame = None,
    status: Optional["st.delta_generator.DeltaGenerator"] = None,
) -> pd.DataFrame:
    """Create engineered features for synthetic PIS/RBS data.

    This function:

    1. Validates that ``synth_data`` is provided.
    2. Uses the R-based ``RVariableCreator`` to generate:
       * ``PRODUCER_GROUP_TOP`` strata features,
       * ``IMPORTER_NAME_TOP`` strata features,
       * binary quantity features per risk unit.
    3. Merges the generated quantity-binary features into ``synth_data``,
       after dropping any existing conflicting columns.

    Args:
        synth_data: Synthetic data DataFrame with risk-unit level records.
        dt_train: Training DataFrame used by the R script for strata feature
            generation.
        producer_group_mapping: Optional producer grouping DataFrame (not used
            directly here but may be required upstream).

    Returns:
        Updated ``synth_data`` DataFrame with engineered features added.

    Raises:
        ValueError: If ``synth_data`` is None.
    """

    # Logger for GUI
    def log(msg: str):
        if status is not None:
            status.write(msg)
        else:
            print(msg)

    if synth_data is None:
        raise ValueError('Synthetic Data Passed is None')

    print(f'\nCreating Engineered Features...')

    print(f'   Creating features generated by the R script')
    if status is not None: log(f'---Creating features generated by the R script')
    # Create features from the R script using the R wrapper
    creator = RVariableCreator()

    print(f'      Creating PRODUCER_GROUP_TOP feature')
    if status is not None: log(f'------Creating PRODUCER_GROUP_TOP feature')
    synth_data = creator.generate_producer_top_strata_features(
        df=synth_data,
        dt_train=dt_train,
        max_strat_count=50,
        min_action_rate=0.02,
        min_records=5,
        use_parquet=False,
    )

    print(f'      Creating IMPORTER_NAME_TOP feature')
    if status is not None: log(f'------Creating IMPORTER_NAME_TOP feature')
    synth_data = creator.generate_importer_top_strata_features(
        df=synth_data,
        dt_train=dt_train,
        max_strat_count=50,
        min_action_rate=0.02,
        min_records=5,
        use_parquet=False,
    )

    print(f'      Creating Quantity Binary Features')
    if status is not None: log(f'------Creating Quantity Binary Features')
    # Fall back to CSV if needed (smaller data, compatibility)
    key_col = "RISK_UNIT"
    quantity_binary_variables = creator.generate_quantity_binaries(
        df=synth_data,
        quantity_threshold=200,
        group_cols=[key_col],
        use_parquet=False
    )

    # Identify non-key columns coming from quantity_binary_variables
    new_cols = [c for c in quantity_binary_variables.columns if c != key_col]

    # Drop any of those columns from synth_data if they already exist
    cols_to_drop = [c for c in new_cols if c in synth_data.columns]
    if cols_to_drop:
        synth_data = synth_data.drop(columns=cols_to_drop)

    # 4. Merge – no name conflict → no _x/_y suffixes
    # TODO: Avoid using merge for larger dataframes
    synth_data = synth_data.merge(
        quantity_binary_variables,
        on=key_col,
        how='left',
    )

    print(f'      Done')
    if status is not None: log(f'------Done')
    return synth_data
