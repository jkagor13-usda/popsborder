# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC
from slippage_model_utils.r_script_wrapper import RVariableCreator
import pandas as pd
from pathlib import Path
import re



def preprocess_producer_name(name, suffix_string=None, prefix_string=None):
    """
    Preprocess a single producer name according to specified rules.
    Mimics the R function basic_text_preproc.

    Args:
        name: Raw producer name string
        suffix_string: Custom regex pattern for suffixes to remove
        prefix_string: Custom regex pattern for prefixes to remove

    Returns:
        Preprocessed and truncated name (max 15 chars)
    """
    # Convert to string and lowercase FIRST
    text = str(name).lower() if pd.notna(name) else name

    # Handle NA/Not Selected - convert to "missing"
    if pd.isna(text) or text.strip() == "not selected":
        return "missing"  # Changed from 'NA/NOT SELECTED' to match R

    # Remove punctuation (this happens BEFORE box removal in R)
    text = re.sub(r'[^\w\s]', '', text)

    # Remove end of string starting with "box"
    # R code finds position of "box" and truncates there
    box_match = re.search(r'box', text)
    if box_match:
        box_position = box_match.start()
        if box_position > 0:
            text = text[:box_position]
        # If box_position == 0, keep the whole string (mimics R's if_else logic)

    # Remove stand-alone numeric sequences
    text = re.sub(r'\b\d+\b', '', text)

    # Replace multiple blanks with single blank AND strip leading/trailing
    # (str_squish in R does both)
    text = re.sub(r'\s+', ' ', text).strip()

    # Define default suffix pattern if not provided
    if suffix_string is None:
        # Note: R uses " sa| s a|sociedad" (space before "sa", no space before "|sociedad")
        suffix_string = r'( sa| s a|sociedad anonima| inc| llc| ltd| ltda| cv| rl| co| co ltd| corp| bv| b v| corporation| company| limited)$'

    # Define default prefix pattern if not provided
    if prefix_string is None:
        prefix_string = r'^(mr |m r )'

    # Remove suffixes and prefixes
    text = re.sub(suffix_string, '', text)
    text = re.sub(prefix_string, '', text)

    # Final cleanup after suffix/prefix removal
    text = re.sub(r'\s+', ' ', text).strip()

    # Truncate to first 15 characters
    text = text[:15]

    return text


def create_producer_mapping(producer_group_mapping_df, use_shortest_name=True):
    """
    Create a mapping dictionary from producer names to groups.

    Args:
        producer_group_mapping_df: DataFrame with 'PRODUCER_NAME' and 'grouping' columns
        use_shortest_name: If True, use shortest raw name as group label instead of numeric grouping

    Returns:
        Dictionary mapping preprocessed producer names to group labels
    """
    # Create a copy to avoid modifying original
    mapping_df = producer_group_mapping_df.copy()

    # Preprocess all producer names in the mapping file
    mapping_df['producer_name_preprocessed'] = mapping_df['PRODUCER_NAME'].apply(preprocess_producer_name)

    # If using shortest name as group label
    if use_shortest_name:
        # For each group, find the shortest original name
        group_labels = (
            mapping_df.groupby('grouping')['PRODUCER_NAME']
            .apply(lambda x: min(x, key=len))
            .to_dict()
        )
        # Map each preprocessed name to its group's shortest name
        mapping_df['group_label'] = mapping_df['grouping'].map(group_labels)
    else:
        # Use the numeric grouping as-is
        mapping_df['group_label'] = mapping_df['grouping']

    # Create the mapping dictionary
    producer_to_group = mapping_df.set_index('producer_name_preprocessed')['group_label'].to_dict()

    return producer_to_group


def apply_producer_grouping(input_data, producer_group_mapping_df, use_shortest_name=True):
    """
    Apply producer name preprocessing and grouping to input data.

    Args:
        input_data: DataFrame with 'PRODUCER_NAME' column
        producer_group_mapping_df: DataFrame with 'PRODUCER_NAME' and 'grouping' columns
        use_shortest_name: If True, use shortest raw name as group label.  If False, use the numeric grouping number

    Returns:
        DataFrame with added 'producer_name_preprocessed' and 'producer_group' columns
    """
    # Create a copy to avoid modifying original
    data = input_data.copy()

    merged_data = data.merge(
        producer_group_mapping_df,
        on='PRODUCER_NAME',
        how='left'  # 'left' keeps all rows from 'data' and adds 'grouping' where possible
    )

    # Identify shortest name in the group mapping
    shortest_name_per_group = (
        producer_group_mapping_df
        .groupby('grouping')['PRODUCER_NAME']
        .apply(lambda names: min([preprocess_producer_name(n) for n in names], key=len))
        .reset_index()
        .rename(columns={'PRODUCER_NAME': 'producer_group'})
    )

    # Merge the shortest name input into the merged_data
    final_merged = merged_data.merge(shortest_name_per_group, on='grouping', how='left')

    # Fill any producers without matching with a no group match indicator string
    final_merged['producer_group'] = final_merged['producer_group'].fillna('NO_GROUP_MATCH')

    return final_merged



def create_engineered_features(
    synth_data: pd.DataFrame = None,
    producer_group_mapping: pd.DataFrame=None,
) -> pd.DataFrame:

    if synth_data is None:
        raise ValueError('Synthetic Data Passed is None')


    print(f'\nCreating Engineered Features...')
    print(f'   Creating producer mappings')
    # Create producer mappings
    synth_data = apply_producer_grouping(
        synth_data,
        producer_group_mapping,
        use_shortest_name=True  # Set to False if wanting to use numeric grouping labels
    )

    print(f'   Creating features generated by the R script')
    # Create features from the R script using the R wrapper
    creator = RVariableCreator()

    print(f'      Cleaning Importer Name')
    # Create a raw IMPORTER_NAME column with the original importer name
    synth_data['IMPORTER_NAME_RAW'] = synth_data['IMPORTER_NAME']

    # Update the IMPORTER_NAME column with the cleaned version.
    synth_data['IMPORTER_NAME'] = creator.batch_basic_text_preproc(synth_data['IMPORTER_NAME_RAW'])

    print(f'      Creating Quantity Binary Features')
    # Fall back to CSV if needed (smaller data, compatibility)
    quantity_binary_variables = creator.generate_quantity_binaries(
        df=synth_data,
        quantity_threshold=200,
        group_cols=['RISK_UNIT'],
        use_parquet=False
    )

    synth_data = synth_data.merge(
        quantity_binary_variables,
        on='RISK_UNIT',
        how='left'
    )

    print(f'      Done')

    return synth_data
