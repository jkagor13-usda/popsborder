# Simulation of contaminated consignments and their inspections
# Copyright (C) 2018-2022 Vaclav Petras and others (see below)
# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

"""
Modifications:
- 10/3/2025: Modifications described below (Gary Lin)
    New Functions Added
    -------------------
    - load_compliance_lookup_csv():
        * Loads compliance level CSV files for RBS inspection workflows
        * Parses country/material type combinations with associated detection and confidence levels
        * Returns dictionary structure for efficient compliance level lookup during simulation

    - load_input_consignment_data():
        * Loads RBS calculator data for realistic consignment generation
        * Supports various input formats (CSV, Excel) for consignment parameter specifications
        * Integrates with synthetic data generation workflows for enhanced simulation realism
    -------------------

    Configuration Enhancements
    --------------------------
    - Enhanced configuration validation for new terminology (inspection_units vs boxes, sample_units vs items)
    - Added backward compatibility parameter mapping throughout configuration loading
    - Improved error handling and validation for RBS-specific configuration parameters
    --------------------------
"""

# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.

# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.

# You should have received a copy of the GNU General Public License along with
# this program; if not, see https://www.gnu.org/licenses/gpl-2.0.html


"""Inputs, especially loading of configuration

.. codeauthor:: Vaclav Petras <wenzeslaus gmail com>
.. codeauthor:: Gary Lin <Gary.Lin jhuapl edu>
.. codeauthor:: Joseph Agor <Joseph.Agor jhuapl edu>
"""

import copy
import json
import math
import sys
import types
from collections.abc import Iterable, Mapping
from pathlib import Path
import csv
from typing import Dict, Tuple, List, Set, Union, Optional
import warnings
import chardet
import pandas as pd


def text_to_value(arg):
    """Convert text to int, float, or interpret it as JSON if possible

    If the argument is not string, it is returned as is.
    If conversion is not possible, the original text is returned.
    None is returned for both None and an empty string.
    """
    # This is a short function with return statements only.
    # pylint: disable=too-many-return-statements
    if not isinstance(arg, str):
        return arg
    if not arg:
        return None
    try:
        return int(arg)
    except ValueError:
        try:
            return float(arg)
        except ValueError:
            # Supports JSON booleans, Python-like booleans, and all YAML booleans.
            # We do not support old YAML booleans.
            if arg in ["true", "True", "TRUE"]:
                return True
            if arg in ["false", "False", "FALSE"]:
                return False
            try:
                return json.loads(arg)
            except json.JSONDecodeError:
                # add yaml.safe_load here?
                # Return the original value.
                return arg


def update_nested_dict_by_dict(dictionary, update):
    """Recursively update nested dictionary by anther nested dictionary"""
    for key, value in update.items():
        if isinstance(value, Mapping):
            dictionary[key] = update_nested_dict_by_dict(dictionary.get(key, {}), value)
        else:
            dictionary[key] = value
    return dictionary


def update_nested_dict_by_item(dictionary, keys, value):
    """Update nested dictionary by a nested keys-value pair

    An item is a list of keys to navigate the nested dictionary and a value to place
    in the given position.

    When a key can be represented as an int, it is used as a list index.
    List must already exists in the given size or the only index used must be 0.

    Floating point keys are not supported.
    """
    if len(keys) == 1:
        key = keys[0]
        try:
            key = int(key)
        except ValueError:
            pass
        dictionary[key] = value
    else:
        if keys[0] not in dictionary:
            try:
                # Test if the next key is an integer and thus index in a list.
                int(keys[1])
                # In case it is a list, we require keys are items are filled in order.
                dictionary[keys[0]] = [None]
            except ValueError:
                dictionary[keys[0]] = {}
        update_nested_dict_by_item(dictionary[keys[0]], keys[1:], value)


def record_to_nested_dictionary(record):
    """Convert dictionary with key/subkey/subsubkey keys into a nested dictionary"""
    out = {}
    for path, value in record.items():
        try:
            keys = path.split("/")
        except AttributeError as error:
            raise ValueError(
                f"Record items need to be strings, not {path} ({type(path).__name__})"
            ) from error
        update_nested_dict_by_item(out, keys, value)
    return out


def update_config(config, record):
    """Update config dictionary by a dictionary with key/subkey/subsubkey keys"""
    config = copy.deepcopy(config)
    update = record_to_nested_dictionary(record)
    update_nested_dict_by_dict(config, update)
    return config


def load_configuration_yaml_from_text(text):
    """Return configuration dictionary from YAML in a string"""
    import yaml  # pylint: disable=import-outside-toplevel

    if hasattr(yaml, "full_load"):
        return yaml.full_load(text)
    return yaml.load(text)  # pylint: disable=no-value-for-parameter


def load_one_configuration(
    filename: Union[str, Path, Iterable],
    sheet: Optional[str] = None,
    key_column: Optional[str] = None,
    value_column: Optional[str] = None,
):
    """Get the configuration from a JSON or YAML file ..."""

    # If filename is already an iterable of records, treat it as such.
    if isinstance(filename, Iterable) and not isinstance(filename, (str, Path)):
        return record_to_nested_dictionary(filename)

    filename_str = str(filename)
    if "::" in filename_str:
        filename_str, info_text = filename_str.rsplit("::", maxsplit=1)
        info = table_info_from_text(info_text)
    else:
        info = table_info_from_text("")

    path = Path(filename_str)

    if sheet:
        info.sheet = sheet
    if key_column:
        info.key_column = key_column
    if value_column:
        info.value_column = value_column

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open() as f:
            return json.load(f)
    elif suffix in [".yaml", ".yml"]:
        import yaml  # pylint: disable=import-outside-toplevel

        with path.open() as f:
            if hasattr(yaml, "full_load"):
                return yaml.full_load(f)
            return yaml.load(f)  # pylint: disable=no-value-for-parameter
    elif suffix in [".csv", ".xlsx", ".ods"]:
        return load_config_table(
            path,
            sheet=info.sheet,
            key_column=info.key_column,
            value_column=info.value_column,
        )
    else:
        sys.exit(f"Unknown file extension (file: {path})")


def resolve_included_files(dictionary: dict, base_file_name: Optional[Union[str, Path]] = None) -> None:
    """Replace links to files by the file content (materialize included files)"""
    for key, value in dictionary.items():
        if isinstance(value, dict):
            if len(value) == 1 and "include_file" in value:
                value = value["include_file"]
                nested_file = value["file_name"]
                if base_file_name:
                    parent = Path(base_file_name).parent
                    nested_file = parent / nested_file
                nested_sheet = value.get("sheet")
                nested_key_column = value.get("key_column")
                nested_value_column = value.get("value_column")
                file_format = value.get("file_format")
                if file_format == "list":
                    values = load_scenario_table(nested_file)
                    new_value = []
                    for item in values:
                        new_value.append(record_to_nested_dictionary(item))
                else:
                    new_value = load_one_configuration(
                        nested_file,
                        sheet=nested_sheet,
                        key_column=nested_key_column,
                        value_column=nested_value_column,
                    )
                dictionary[key] = new_value
            else:
                resolve_included_files(value, base_file_name)


def load_configuration(
    filename: Union[str, Path],
    sheet: Optional[str] = None,
    key_column: Optional[str] = None,
    value_column: Optional[str] = None,
):
    """Get the configuration from a JSON or YAML file

    The format is decided based on the file extension.
    It uses full_load() (FullLoader) to read YAML.

    The parameter can be a string or a path object (path-like object).

    If the *filename* contains `::`, anything after the last `::` is considered
    parameters determining where in the spreadsheet or table are the relevant
    columns. The format is multiple key-value pairs with key and value separated by
    `=`, `:`, or `: ` and individual pairs separated by `,`.
    The same information can be passed directly as function parameters.
    If both are provided, function parameters take precedence.

    Any file specified under the `include_file` key is included.
    """
    path = Path(filename)
    config = load_one_configuration(
        path, sheet=sheet, key_column=key_column, value_column=value_column
    )
    resolve_included_files(config, base_file_name=path)
    return config


def table_info_from_text(text, sheet=None, key_column=None, value_column=None):
    """Convert comma-separated list of key-value pairs to info about table

    Items are separated by comma. Key and value can be separated by
    `=` or `:` and also by `: ` and ` = ` because spaces surrounding
    key and value are removed.

    If there is no key-value pair, the whole value is used as a
    value for key_column.

    Function parameters are used as default values.

    Accepted keys are 'sheet', 'key_column', and 'value_column'.
    """
    info = types.SimpleNamespace(
        sheet=sheet, key_column=key_column, value_column=value_column
    )
    if not text:
        return info
    items = text.split(",")
    for item in items:
        key = None
        for separator in ["=", ":"]:
            if separator in item:
                key, value = item.split(separator, maxsplit=1)
                key = key.strip()
                value = value.strip()
                break
        if not key:
            info.key_column = item
        elif key in ["sheet", "key_column", "value_column"]:
            setattr(info, key, value)
        else:
            raise ValueError(f"Unknown key '{key}' in table info specification")
    return info


def load_config_table(
    filename: Union[str, Path],
    sheet: Optional[str] = None,
    key_column: Optional[str] = None,
    value_column: Optional[str] = None,
):
    """Load a CSV file into a list of dictionaries

    Values which can be converted into int or float are converted. Cells which can be
    parsed as JSON, will be loaded into Python data structures (dicts, lists, etc.).

    A whole file is read and loaded into memory unlike with the ``csv.reader()``
    function.
    """
    path = Path(filename)
    suffix = path.suffix.lower()

    if suffix not in [".csv", ".ods"]:
        return load_config_xlsx(
            path, sheet=sheet, key_column=key_column, value_column=value_column
        )

    if suffix == ".ods":
        return load_config_ods(
            path, sheet=sheet, key_column=key_column, value_column=value_column
        )

    return load_config_csv(path, key_column=key_column, value_column=value_column)


def column_from_string(arg, default, fallback):
    """Return zero-based column index created from a column specification.

    Values convertable to integer are converted and considered to be one-based
    column index and thus further converted to zero-based index.

    For values not convertable to integer, fallback function is called and its
    result is returned. This allows for a backend-specific column conversion function
    to be used to resolve the complex cases.

    For None and empty string, the default value is returned.
    """
    if not arg:
        return default
    else:
        try:
            return int(arg) - 1
        except ValueError:
            return fallback(arg)


def validate_key(arg):
    """Return a validated key or None if it cannot be validated.

    Accepts any value which is supposed to be a configuration key
    and returns that value ready to be used as a key, i.e., a string
    without any extra whitespace around it (in case of user-input errors
    in a spreadsheet).

    For an empty string, NaN (not a number), and None returns None,
    i.e., the key is invalid and the associated value in a key-value pair (if any)
    should be ignored. Additionally, it returns None also for keys containing
    a space (after being stripped from leading and trainling spaces). These are
    assumed to be headings inside a spreadsheet or other values to be ignored
    because valid keys don't have spaces in them.
    """
    if isinstance(arg, str):
        # Allow for whitespace around the key.
        arg = arg.strip()
        if not arg or " " in arg:
            return None
        return arg
    if arg is None:
        return None
    if isinstance(arg, float) and math.isnan(arg):
        # NaN is None in our context since we don't use NaN in configuration
        # and Pandas reader defaults to NaN for no-data values.
        return None
    # All keys should be strings, so convert numbers and anything else to strings.
    return str(arg)


def load_config_csv(
    filename: Union[str, Path],
    key_column: Optional[str] = None,
    value_column: Optional[str] = None,
):
    """Read configuration from a CSV table.

    The key_column and value_column parameters can be one-based column indices or
    spreadsheet-like column letters in range A-Z.

    Rows without keys or with invalid keys are ignored. See :func:`validate_key`
    for details.

    Values which can be converted from string to other types are converted
    automatically. See :func:`text_to_value` for details.
    """
    path = Path(filename)
    table = {}
    with path.open() as file:
        # pylint: disable=import-outside-toplevel
        import csv

        key_column_idx = column_from_string(
            key_column, default=0, fallback=lambda x: ord(x) - ord("A")
        )
        value_column_idx = column_from_string(
            value_column, default=key_column_idx + 1, fallback=lambda x: ord(x) - ord("A")
        )

        for row in csv.reader(file):
            key = validate_key(row[key_column_idx])
            if key:
                value = text_to_value(row[value_column_idx])
                table[key] = value

    return record_to_nested_dictionary(table)


def load_config_xlsx(
    filename: Union[str, Path],
    sheet: Optional[str] = None,
    key_column: Optional[str] = None,
    value_column: Optional[str] = None,
):
    """Read configuration from a XLSX table.

    The sheet parameter is name of the sheet within the given spreadsheet file.

    The key_column and value_column parameters can be one-based column indices or
    spreadsheet-like column letters.

    See :func:`load_config_csv` for handling of keys and values.
    """
    path = Path(filename)
    table = {}

    import openpyxl
    from openpyxl.utils import column_index_from_string

    key_column_idx = column_from_string(
        key_column, default=0, fallback=lambda x: column_index_from_string(x) - 1
    )
    value_column_idx = column_from_string(
        value_column,
        default=key_column_idx + 1,
        fallback=lambda x: column_index_from_string(x) - 1,
    )

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Data Validation extension is not supported"
        )
        workbook = None
        try:
            workbook = openpyxl.load_workbook(path, read_only=True)
            ws = workbook[sheet] if sheet else workbook.active
            for row in ws.iter_rows(values_only=True):
                key = validate_key(row[key_column_idx])
                if key:
                    value = text_to_value(row[value_column_idx])
                    table[key] = value
        finally:
            if workbook:
                workbook.close()
    return record_to_nested_dictionary(table)


def load_config_ods(
    filename: Union[str, Path],
    sheet: Optional[Union[int, str]] = None,
    key_column: Optional[str] = None,
    value_column: Optional[str] = None,
):
    """Read configuration from a ODS table.

    The sheet parameter is name of the sheet within the given spreadsheet file.

    The key_column and value_column parameters can be one-based column indices or
    spreadsheet-like column letters. If the value column comes before the key
    (parameter name) column in the file, only columns in the range A-Z are supported
    when specified using letters (specifying using one-based indices works
    in any case).

    See :func:`load_config_csv` for handling of keys and values.
    """
    table = {}

    # pylint: disable=import-outside-toplevel

    sheet = sheet if sheet else 0

    key_column = column_from_string(key_column, default=0, fallback=lambda x: x)
    value_column = column_from_string(value_column, default=1, fallback=lambda x: x)

    # The columns are always returned in the order as in the table, not in
    # the order specified in the parameter, so we need to know the relative order
    # of the two columns.
    flip_order = False
    # Assuming both are of the same type.
    if isinstance(key_column, int) and isinstance(value_column, int):
        cols = [key_column, value_column]
        if key_column > value_column:
            flip_order = True
    else:
        cols = f"{key_column},{value_column}"
        # Multi-letter columns which are in opposite order are not handled.
        if (
            len(key_column) == 1
            and len(value_column) == 1
            and ord(key_column) > ord(value_column)
        ):
            flip_order = True

    if flip_order:
        key_column_index = 2
        value_column_index = 1
    else:
        key_column_index = 1
        value_column_index = 2

    data = pandas.read_excel(
        Path(filename), sheet_name=sheet, header=None, usecols=cols
    )
    table = {}
    for row in data.itertuples():
        key = validate_key(row[key_column_index])
        if key:
            value = text_to_value(row[value_column_index])
            if isinstance(value, float) and math.isnan(value):
                value = None
            table[key] = value
    return record_to_nested_dictionary(table)


def add_dict_config_to_table(table, value, keys=None):
    """Add a nested dictionary to a table represented by a mapping

    This is meant for internal use, if possible, use :func:`dict_config_to_table`.
    """
    if keys is None:
        keys = []
    if isinstance(value, Mapping):
        for nested_key, nested_value in value.items():
            new_keys = keys.copy()
            new_keys.append(nested_key)
            add_dict_config_to_table(table, value=nested_value, keys=new_keys)
    else:
        table["/".join(keys)] = value


def dict_config_to_table(value):
    """Convert a nested dictionary to a table represented by a mapping"""
    table = {}
    add_dict_config_to_table(table, value)
    return table


def print_table_config(config, file=None):
    """Print a table represented by a mapping

    Generates key-value pairs separated by a pipe.
    It does not quote or espace the separator in values.
    """
    for key, value in config.items():
        if isinstance(value, Iterable) and not isinstance(value, str):
            value = json.dumps(value)
        print(f"{key}|{value}", file=file)


def load_scenario_table(filename: Union[str, Path]):
    """Load a CSV file into a list of dictionaries

    Values which can be converted into int or float are converted. Cells which can be
    parsed as JSON, will be loaded into Python data structures (dicts, lists, etc.).

    A whole file is read and loaded into memory unlike with the ``csv.reader()``
    function.
    """
    path = Path(filename)
    table = []

    if path.suffix.lower() != ".csv":
        import openpyxl

        workbook = None
        try:
            workbook = openpyxl.load_workbook(path, read_only=True)
            sheet = workbook.active
            header = [cell.value for cell in sheet[1]]
            for old_row in sheet.iter_rows(min_row=2):
                new_row = {}
                for key, cell in zip(header, old_row):
                    new_row[key] = text_to_value(cell.value)
                table.append(new_row)
        finally:
            if workbook:
                workbook.close()
        return table

    # CSV path
    with path.open() as file:
        import csv

        for row in csv.DictReader(file):
            for key, value in row.items():
                row[key] = text_to_value(value)
            table.append(row)
    return table


def load_cfrp_schedule(filename, date_format=None):
    """Load cut flower release program (CFRP) schedule

    Supports one CSV file with all combinations of date, origin, and commodity (flower)
    row by row with columns date, origin_nm, and commodity (caseinsensitive).
    Duplicate records don't appear in the resulting data structure.

    The values in the date column are parsed according to *date_format*.

    Returns a dictionary with keys being tuples of commodity and origin and values
    being a set of dates.
    """
    if not date_format:
        date_format = "%Y-%m-%d"
    schedule = {}
    # Read as CSV
    with open(filename) as file:
        # Import file- or format-specific items only when need.
        # pylint: disable=import-outside-toplevel
        import csv
        from datetime import datetime

        for row in csv.DictReader(file):
            for key, value in row.items():
                if key.lower() == "date":
                    date = datetime.strptime(value, date_format).date()
                elif key.lower() == "origin_nm":
                    origin = value
                elif key.lower() == "commodity":
                    commodity = value
            combo = (commodity, origin)
            if combo not in schedule:
                # Using set to ensure we have no duplicate dates.
                schedule[combo] = set()
            schedule[combo].add(date)
    return schedule


def load_skip_lot_consignment_records(filename, tracked_properties):
    """Load records associating consignment with skip lot compliance levels.

    Only the *tracked_properties* are considered and are assumed to uniquely
    distinguish consignment records with distinct compliance levels.
    If the level can be converted to number, it is converted.
    """
    records = {}
    # Read as CSV
    with open(filename) as file:
        # Import file- or format-specific items only when need.
        # pylint: disable=import-outside-toplevel
        import csv

        for row in csv.DictReader(file):
            combo = []
            for tracked_property in tracked_properties:
                combo.append(row[tracked_property])
            level = row["compliance_level"]
            records[tuple(combo)] = text_to_value(level)
    return records


def get_validated_effectiveness(config):
    """Set the effectiveness of the inspector.

    :param config: Configuration file
    """
    effectiveness = config["inspection"].get("effectiveness", 1)
    if 0 <= effectiveness <= 1:
        return effectiveness
    raise ValueError("Effectiveness must be between 0 and 1")


def load_compliance_lookup_csv(filepath: Path):

    """
    Build comp_table where:
      key   = N-tuple of values from all columns BEFORE 'Compliance'
      value = (Detection Level, Confidence Levels)
    Raises ValueError if required columns are missing.
    """
    comp_table = {}

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        headers = [h.strip() for h in next(r)]
        if not headers:
            raise ValueError("CSV appears to be empty or missing a header row.")

        required = {"Compliance", "Detection Level", "Confidence Levels"}
        missing = [c for c in required if c not in headers]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")

        comp_idx = headers.index("Compliance")
        key_cols = headers[:comp_idx]  # all columns BEFORE 'Compliance' will be excluded
        comp_table['rbs_variables'] = list(key_cols)

        for row in r:
            # Map row to header names (short rows are padded automatically by zip)
            row_map = {h: (row[i].strip() if i < len(row) else "") for i, h in enumerate(headers)}

            key = tuple(row_map[col] for col in key_cols)
            value = (row_map["Detection Level"], row_map["Confidence Levels"])
            comp_table[key] = value  # last occurrence wins

    return comp_table

def detect_encoding(filepath: Path) -> str:
    """
    Detect the encoding of a file.

    Args:
        filepath: Path to file

    Returns:
        Detected encoding string
    """
    with open(filepath, 'rb') as f:
        raw_data = f.read(10000)  # Read first 10KB
        result = chardet.detect(raw_data)
        return result['encoding']


def load_compliance_mapping_csv(filepath: Path) -> Dict[str, Tuple[str, str]]:
    """
    Load the compliance mapping table.

    Expected columns: Compliance | Detection Level | Confidence Levels

    Args:
        filepath: Path to compliance mapping CSV file

    Returns:
        Dictionary mapping Compliance value to (Detection Level, Confidence Levels)

    Raises:
        ValueError: If required columns are missing or file is empty
        FileNotFoundError: If filepath doesn't exist
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Compliance mapping file not found: {filepath}")

    # Auto-detect encoding if not provided
    try:
        encoding = detect_encoding(filepath)
        print(f"Auto-detected encoding for {filepath.name}: {encoding}")
    except Exception as e:
        warnings.warn(f"Could not detect encoding, trying common encodings: {e}")
        encoding = 'utf-8'

    # Try multiple encodings
    encodings_to_try = [
        encoding,
        'utf-8-sig',
        'utf-8',
        'latin-1',
        'iso-8859-1',
        'cp1252',
        'windows-1252'
    ]

    last_error = None
    for enc in encodings_to_try:
        try:
            return _load_compliance_mapping_with_encoding(filepath, enc)
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            # If it's not an encoding error, raise it
            raise

    # If all encodings failed
    raise UnicodeDecodeError(
        'utf-8', b'', 0, 1,
        f"Failed to decode file with any encoding. Last error: {last_error}"
    )

def _load_compliance_mapping_with_encoding(filepath: Path, encoding: str) -> Dict[str, Tuple[str, str]]:
    """Internal helper to load compliance mapping with specific encoding."""
    mapping = {}

    with open(filepath, newline="", encoding=encoding, errors='replace') as f:
        reader = csv.reader(f)

        # Read and validate headers
        try:
            headers = [h.strip() for h in next(reader)]
        except StopIteration:
            raise ValueError("Compliance mapping CSV file is empty.")

        if not headers or all(h == "" for h in headers):
            raise ValueError("Compliance mapping CSV is missing a header row.")

        # Validate required columns
        required = {"Compliance", "Detection Level", "Confidence Levels"}
        missing = required - set(headers)
        if missing:
            raise ValueError(
                f"Compliance mapping CSV missing required column(s): {', '.join(sorted(missing))}"
            )

        # Get column indices
        comp_idx = headers.index("Compliance")
        det_level_idx = headers.index("Detection Level")
        conf_levels_idx = headers.index("Confidence Levels")

        # Process data rows - track all occurrences
        compliance_occurrences = {}  # compliance -> list of (row_num, full_row_data)

        for row_num, row in enumerate(reader, start=2):
            if not row or all(cell.strip() == "" for cell in row):
                continue  # Skip empty rows

            if len(row) <= max(comp_idx, det_level_idx, conf_levels_idx):
                warnings.warn(f"Row {row_num} has insufficient columns, skipping.")
                continue

            # Build row map with all columns
            row_map = {
                h: (row[i].strip() if i < len(row) else "")
                for i, h in enumerate(headers)
            }

            compliance = row_map["Compliance"]
            detection_level = row_map["Detection Level"]
            confidence_levels = row_map["Confidence Levels"]

            if not compliance:
                warnings.warn(f"Row {row_num} has empty Compliance value, skipping.")
                continue

            # Track all occurrences of this compliance value with full row data
            if compliance not in compliance_occurrences:
                compliance_occurrences[compliance] = []
            compliance_occurrences[compliance].append((row_num, row_map))

            mapping[compliance] = (detection_level, confidence_levels)

        # Find duplicates (compliance values that appear more than once)
        duplicates = {k: v for k, v in compliance_occurrences.items() if len(v) > 1}

        # Enhanced duplicate warning with table format showing all occurrences
        if duplicates:
            total_duplicate_rows = sum(len(occurrences) for occurrences in duplicates.values())
            first_dup_compliance = next(iter(duplicates.keys()))
            first_dup_row = duplicates[first_dup_compliance][0][0]

            # Build the warning message
            dup_lines = [
                f"Found {len(duplicates)} duplicate Compliance value(s) in mapping "
                f"({total_duplicate_rows} total rows affected). "
                f"Last occurrence will be used. First duplicate at row {first_dup_row}: '{first_dup_compliance}'",
                ""  # Blank line for readability
            ]

            # Determine max duplicates to show in detail
            max_keys_to_show = 3
            keys_shown = 0

            for compliance, occurrences in list(duplicates.items())[:max_keys_to_show]:
                keys_shown += 1

                # Show header for this duplicate group
                dup_lines.append(f"\nDuplicate #{keys_shown} - Compliance: '{compliance}'")
                dup_lines.append(f"  Found {len(occurrences)} occurrence(s):")

                # Table header with ALL columns
                header = "  Row | " + " | ".join(headers)
                dup_lines.append(header)
                dup_lines.append("  " + "-" * (len(header) - 2))

                # Show all occurrences of this duplicate
                for row_num, row_data in occurrences:
                    # Build row display with all columns in order
                    row_values = [row_data.get(h, "") for h in headers]
                    row_display = f"  {row_num:4d} | " + " | ".join(str(v) for v in row_values)
                    dup_lines.append(row_display)

            # Summary if there are more duplicates
            if len(duplicates) > max_keys_to_show:
                remaining = len(duplicates) - max_keys_to_show
                dup_lines.append(f"\n... and {remaining} more duplicate Compliance value(s) not shown")

            warnings.warn("\n".join(dup_lines))

    return mapping



def load_compliance_table_csv(
        filepath: Path,
        use_parquet: bool = True
) -> Dict:
    """
    Load compliance table with automatic format detection.

    Args:
        filepath: Path to compliance file (.csv or .parquet)
        use_parquet: If True and .parquet exists, use it instead
    """
    # Check for parquet version
    parquet_path = filepath.with_suffix('.parquet')
    if use_parquet and parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df):,} rows from parquet")
    else:
        # Fall back to CSV
        if not filepath.exists():
            raise FileNotFoundError(f"Compliance table file not found: {filepath}")

        # Auto-detect encoding if not provided
        try:
            encoding = detect_encoding(filepath)
            print(f"Auto-detected encoding for {filepath.name}: {encoding}")
        except Exception as e:
            warnings.warn(f"Could not detect encoding, trying common encodings: {e}")
            encoding = 'utf-8'

        # Try multiple encodings
        encodings_to_try = [
            encoding,
            'utf-8-sig',
            'utf-8',
            'latin-1',
            'iso-8859-1',
            'cp1252',
            'windows-1252'
        ]

        last_error = None
        for enc in encodings_to_try:
            try:
                return _load_compliance_table_with_encoding(filepath, enc)
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except Exception as e:
                raise

        raise UnicodeDecodeError(
            'utf-8', b'', 0, 1,
            f"Failed to decode file with any encoding. Last error: {last_error}"
        )

    rbs_variables = [col for col in df.columns if col != 'Compliance']

    # Convert to numpy for faster iteration
    keys_array = df[rbs_variables].values
    compliance_array = df['Compliance'].values

    compliance_dict = {
        'rbs_variables': rbs_variables,
        '_compliance_values': set(compliance_array)
    }

    # Build dictionary using numpy arrays (faster)
    for i in range(len(keys_array)):
        key = tuple(keys_array[i])
        compliance_dict[key] = compliance_array[i]

    return compliance_dict


def _load_compliance_table_with_encoding(filepath: Path, encoding: str) -> Dict:
    """Internal helper to load compliance table with specific encoding."""
    compliance_table = {}

    with open(filepath, newline="", encoding=encoding, errors='replace') as f:
        reader = csv.reader(f)

        # Read and validate headers
        try:
            headers = [h.strip() for h in next(reader)]
        except StopIteration:
            raise ValueError("Compliance table CSV file is empty.")

        if not headers or all(h == "" for h in headers):
            raise ValueError("Compliance table CSV is missing a header row.")

        # Validate required column
        if "Compliance" not in headers:
            raise ValueError("Compliance table CSV missing required column: 'Compliance'")

        # Identify key columns (everything before 'Compliance')
        comp_idx = headers.index("Compliance")
        key_cols = headers[:comp_idx]

        if not key_cols:
            raise ValueError("No key columns found before 'Compliance' column.")

        compliance_table['rbs_variables'] = list(key_cols)

        # Process data rows - track all occurrences
        key_occurrences = {}  # key -> list of (row_num, full_row_data)
        compliance_values = set()

        for row_num, row in enumerate(reader, start=2):
            if not row or all(cell.strip() == "" for cell in row):
                continue  # Skip empty rows

            # Build row map with proper padding
            row_map = {
                h: (row[i].strip() if i < len(row) else "")
                for i, h in enumerate(headers)
            }

            # Create key tuple from key columns
            key = tuple(row_map[col] for col in key_cols)
            compliance = row_map["Compliance"]

            # Track compliance values
            if compliance:
                compliance_values.add(compliance)

            # Track all occurrences of this key with full row data
            if key not in key_occurrences:
                key_occurrences[key] = []
            key_occurrences[key].append((row_num, row_map))

            compliance_table[key] = compliance

        # Find duplicates (keys that appear more than once)
        duplicates = {k: v for k, v in key_occurrences.items() if len(v) > 1}

        # Enhanced duplicate warning with table format showing all occurrences
        if duplicates:
            total_duplicate_rows = sum(len(occurrences) for occurrences in duplicates.values())
            first_dup_key = next(iter(duplicates.keys()))
            first_dup_row = duplicates[first_dup_key][0][0]

            # Build the warning message
            dup_lines = [
                f"Found {len(duplicates)} duplicate key(s) in compliance table "
                f"({total_duplicate_rows} total rows affected). "
                f"Last occurrence will be used. First duplicate at row {first_dup_row}",
                ""  # Blank line for readability
            ]

            # Determine max duplicates to show in detail
            max_keys_to_show = 3
            keys_shown = 0

            for dup_key, occurrences in list(duplicates.items())[:max_keys_to_show]:
                keys_shown += 1

                # Show header for this duplicate group
                dup_lines.append(f"\nDuplicate #{keys_shown} - Key: {dup_key}")
                dup_lines.append(f"  Found {len(occurrences)} occurrence(s):")

                # Table header with ALL columns
                header = "  Row | " + " | ".join(headers)
                dup_lines.append(header)
                dup_lines.append("  " + "-" * (len(header) - 2))

                # Show all occurrences of this duplicate
                for row_num, row_data in occurrences:
                    # Build row display with all columns in order
                    row_values = [row_data.get(h, "") for h in headers]
                    row_display = f"  {row_num:4d} | " + " | ".join(str(v) for v in row_values)
                    dup_lines.append(row_display)

            # Summary if there are more duplicates
            if len(duplicates) > max_keys_to_show:
                remaining = len(duplicates) - max_keys_to_show
                dup_lines.append(f"\n... and {remaining} more duplicate key(s) not shown")

            warnings.warn("\n".join(dup_lines))

        # Store compliance values for validation
        compliance_table['_compliance_values'] = compliance_values

    return compliance_table


def build_compliance_lookup_table(
        compliance_table_filepath: Path,
        mapping_filepath: Path,
        validate_completeness: bool = True
) -> Dict:
    """
    Build complete compliance lookup table by joining compliance table and mapping.

    Args:
        compliance_table_filepath: Path to compliance table CSV (key columns + Compliance)
        mapping_filepath: Path to compliance mapping CSV (Compliance -> Detection/Confidence)
        validate_completeness: If True, warns about missing mappings

    Returns:
        Dictionary with:
        - 'rbs_variables': List of key column names
        - tuple keys: N-tuples mapping to (Detection Level, Confidence Levels)

    Raises:
        ValueError: If validation fails
    """
    # Load both tables
    compliance_table = load_compliance_table_csv(filepath=compliance_table_filepath)
    mapping = load_compliance_mapping_csv(filepath=mapping_filepath)

    # Build final lookup table
    comp_table = {
        'rbs_variables': compliance_table['rbs_variables']
    }

    # Track unmapped compliance values
    unmapped = set()

    # Join the tables
    for key, compliance in compliance_table.items():
        if key in ('rbs_variables', '_compliance_values'):
            continue

        if compliance in mapping:
            comp_table[key] = mapping[compliance]
        else:
            unmapped.add(compliance)
            comp_table[key] = ("UNKNOWN", "UNKNOWN")

    # Validation
    if validate_completeness:
        compliance_in_rbs = compliance_table.get('_compliance_values', set())
        compliance_in_mapping = set(mapping.keys())

        # Check for unmapped compliance values
        if unmapped:
            warnings.warn(
                f"Found {len(unmapped)} Compliance value(s) in compliance table "
                f"without mapping: {sorted(unmapped)}"
            )

        # Check for unused mappings
        unused = compliance_in_mapping - compliance_in_rbs
        if unused:
            warnings.warn(
                f"Found {len(unused)} Compliance value(s) in mapping table "
                f"not used in compliance table: {sorted(unused)}"
            )

    return comp_table