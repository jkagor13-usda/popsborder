# Simulation of contaminated consignments and their inspections
# Copyright (C) 2018-2022 Vaclav Petras and others (see below)
# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

"""


=====================================
JHU/APL Extensions and Modifications:
=====================================

Contributors: Gary Lin, Joseph Agor (Johns Hopkins University Applied Physics Laboratory)

Modified Functions:
------------------
- pretty_header():
    * Updated terminology in output headers from "Inspection Units" and "Sample Units"
    * Maintains backward compatibility with both old and new terminology display

- pretty_consignment_inspection_units():
    * Updated to handle refactored consignment structure with inspection_units terminology
    * Enhanced display formatting for hierarchical inspection unit organization

- config_to_simplified_simulation_params():
    * Added backward compatibility for configuration parameter mapping
    * Handles cluster_sample_unit_width (formerly cluster_item_width) parameter conversion
    * Updated within_inspection_unit_proportion (formerly within_box_proportion) handling
    * Enhanced support for both old and new contamination_unit terminology

- print_totals_as_text():
    * Updated output text to use new terminology (inspection_units, sample_units)
    * Added backward compatibility for displaying both old and new unit terminology
    * Enhanced reporting format for contamination and inspection statistics

Notes:
------
- Updated all output formatting and reporting functions to use consistent terminology
- Enhanced configuration parameter handling for backward compatibility
- Maintains support for legacy configuration while displaying updated terminology
- All output functions now support both inspection_unit/sample_unit and box/item terminology

Modifications:
- 10/3/2025: Modeifications described below (Gary Lin and Joseph Agor)
    Following New Classes Added
    ------------------
    - SimData:
        * Class to store and write out all simulated data
    ------------------

    Following Functions Modified
    ----------------
    - pretty_header():
        * Updated terminology in output headers from "Inspection Units" and "Sample Units"
        * Maintains backward compatibility with both old and new terminology display

    - pretty_consignment_inspection_units():
        * Updated to handle refactored consignment structure with inspection_units terminology
        * Enhanced display formatting for hierarchical inspection unit organization

    - config_to_simplified_simulation_params():
        * Added backward compatibility for configuration parameter mapping
        * Handles cluster_sample_unit_width (formerly cluster_item_width) parameter conversion
        * Updated within_inspection_unit_proportion (formerly within_box_proportion) handling
        * Enhanced support for both old and new contamination_unit terminology

    - print_totals_as_text():
        * Updated output text to use new terminology (inspection_units, sample_units)
        * Added backward compatibility for displaying both old and new unit terminology
        * Enhanced reporting format for contamination and inspection statistics

    - pretty_consignment_items():
        * Renamed to pretty_consignment_sample_units()

    - pretty_consignment_boxes():
        * Renamed to pretty_consignment_inspection_units()

    - pretty_consignment_boxes_only():
        * Renamed to pretty_consignment_inspection_units_only()
    ----------------

    Terminology Refactoring
    ----------------
    - Systematically refactored: boxes -> inspection_units, items -> sample_units
    - Updated all class names, method names, and variable names for consistency
    - Added comprehensive backward compatibility for existing configurations
    - Maintained dual access patterns for smooth migration from legacy terminology
    ----------------

    Backward Compatibility
    ----------------
    - Configuration parameter mapping: boxes -> inspection_units, items -> sample_units
    - Legacy attribute access in Consignment class via __getattr__ and __hasattr__
    - Support for both items_per_box and sample_units_per_inspection_unit configuration keys
    - Maintained existing F280 and AQIM consignment generator functionality
    ----------------
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


"""Generating of various simulation outputs

.. codeauthor:: Vaclav Petras <wenzeslaus gmail com>
.. codeauthor:: Kellyn P. Montgomery <kellynmontgomery gmail com>
.. codeauthor:: Gary Lin <Gary.Lin jhuapl edu>
.. codeauthor:: Joseph Agor <Joseph.Agor jhuapl edu>
"""

import csv
import operator
import shutil
import types
import weakref
from collections import Counter
from collections.abc import MutableMapping
from functools import reduce
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .consignments import Consignment



def pretty_content(array, config=None):
    """Return string with array content nicely visualized as unicode text

    Values evaluating to False are replaced with a flower, others with a bug.
    """
    config = config if config else {}
    flower_sign = config.get("flower", "\N{BLACK FLORETTE}")
    bug_sign = config.get("bug", "\N{BUG}")
    spaces = config.get("spaces", True)
    if spaces:
        separator = " "
    else:
        separator = ""

    def replace(number):
        if number:
            return bug_sign
        else:
            return flower_sign

    pretty = [replace(i) for i in array]
    return separator.join(pretty)


def pretty_header(consignment, line=None, config=None):
    """Return header for a consignment

    Basic info about the consignment is included and the remaining space
    in a terminal window is filled with horizontal box characters.
    (The assumption is that this will be printed in the terminal.)
    """
    config = config if config else {}
    size = 80
    if hasattr(shutil, "get_terminal_size"):
        size = shutil.get_terminal_size().columns
    if line is None:
        # We test None but not for "" to allow use of an empty string.
        line = config.get("horizontal_line", "heavy")
    if line.lower() == "heavy":
        horizontal = "\N{BOX DRAWINGS HEAVY HORIZONTAL}"
    elif line.lower() == "light":
        horizontal = "\N{BOX DRAWINGS LIGHT HORIZONTAL}"
    elif line == "space":
        horizontal = " "
    else:
        horizontal = line
    header = (
        f"{horizontal}{horizontal} Consignment"
        f" {horizontal}{horizontal}"
        f" Inspection Units: {consignment.num_inspection_units} {horizontal}{horizontal}"
        f" Sample Units: {consignment.num_sample_units} "
    )
    if size > len(header):
        size = size - len(header)
    else:
        size = 0
    rule = horizontal * size  # pylint: disable=possibly-unused-variable
    return f"{header}{rule}"


def pretty_consignment_sample_units(consignment, config=None):
    """Pretty-print consignment focusing on individual sample_units"""
    config = config if config else {}
    header = pretty_header(consignment, config=config)
    body = pretty_content(consignment["sample_units"], config=config)
    return f"{header}\n{body}"


def pretty_consignment_inspection_units(consignment, config=None):
    """Pretty-print consignment showing individual sample_units in inspection_units"""
    config = config if config else {}
    line = config.get("inspection_unit_line", "|")
    spaces = config.get("spaces", True)
    if line == "pipe":
        line = "|"
    if spaces:
        separator = f" {line} "
    else:
        separator = line
    header = pretty_header(consignment, config=config)
    body = separator.join(
        [pretty_content(inspection_unit.included_units, config=config) for inspection_unit in consignment["inspection_units"]]
    )
    return f"{header}\n{body}"


def pretty_consignment_inspection_units_only(consignment, config=None):
    """Pretty-print consignment showing individual inspection_units"""
    config = config if config else {}
    line = config.get("horizontal_line", "light")
    header = pretty_header(consignment, line=line, config=config)
    body = pretty_content(consignment["inspection_units"], config=config)
    return f"{header}\n{body}"


def pretty_consignment(consignment, style, config=None):
    """Pretty-print consignment in a given style

    :param style: Style of pretty-printing (inspection_units, inspection_units_only, sample_units)
    """
    config = config if config else {}
    if style == "inspection_units":
        return pretty_consignment_inspection_units(consignment, config=config)
    elif style == "inspection_units_only":
        return pretty_consignment_inspection_units_only(consignment, config=config)
    elif style == "sample_units":
        return pretty_consignment_sample_units(consignment, config=config)
    else:
        raise ValueError(
            f"Unknown style value for pretty printing of consignments: {style}"
        )


class PrintReporter(object):
    """Reporter class which prints a message for each consignment"""

    # Reporter objects carry functions, but many not use any attributes.
    # pylint: disable=no-self-use,missing-function-docstring
    def true_negative(self):
        print("Inspection worked, didn't miss anything (no contaminants) [TN]")

    def true_positive(self):
        print("Inspection worked, found contaminant [TP]")

    def false_negative(self, consignment):
        print(
            f"Inspection failed, missed {count_contaminated_inspection_units(consignment)} "
            "inspection_units with contaminants [FN]"
        )


class MuteReporter(object):
    """Reporter class which is completely silent"""

    # pylint: disable=no-self-use,missing-function-docstring
    def true_negative(self):
        pass

    def true_positive(self):
        pass

    def false_negative(self, consignment):
        pass


class Form280(object):
    """Creates F280 records from the simulated data"""

    def __init__(self, file, disposition_codes, separator=","):
        """Prepares file for writing

        :param file: Name of the file to write to or ``-`` (dash) for printing
        :param disposition_codes: Conversion table for output disposition codes
        :param separator: Value (field) separator for the output CSV file
        """
        self.print_to_stdout = False
        self.file = None
        if file:
            if file in ("-", "stdout", "print"):
                self.print_to_stdout = True
            else:
                self.file = open(file, "w")
                self._finalizer = weakref.finalize(self, self.file.close)
        self.codes = disposition_codes
        # selection and order of columns to output
        columns = ["REPORT_DT", "LOCATION", "ORIGIN_NM", "COMMODITY", "disposition"]

        if self.file:
            self.writer = csv.writer(
                self.file,
                delimiter=separator,
                quotechar='"',
                lineterminator="\n",
                quoting=csv.QUOTE_NONNUMERIC,
            )
            self.writer.writerow(columns)

    def disposition(self, ok, must_inspect, applied_program):
        """Get disposition code for the given parameters

        Provides defaults if the disposition code table does not contain
        a specific value.

        See :meth:`fill` for details about the parameters.
        """
        codes = self.codes
        if applied_program in ["naive_cfrp"]:
            if must_inspect:
                if ok:
                    disposition = codes.get("cfrp_inspected_ok", "OK CFRP Inspected")
                else:
                    disposition = codes.get(
                        "cfrp_inspected_pest", "Pest Found CFRP Inspected"
                    )
            else:
                disposition = codes.get("cfrp_not_inspected", "CFRP Not Inspected")
        else:
            if ok:
                disposition = codes.get("inspected_ok", "OK Inspected")
            else:
                disposition = codes.get("inspected_pest", "Pest Found")
        return disposition

    def fill(self, date, consignment, ok, must_inspect, applied_program):
        """Fill one entry in the F280 form

        :param date: Consignment or inspection date
        :param consignment: Consignment which was tested
        :param ok: True if the consignment was tested negative (no pest present)
        :param must_inspect: True if the consignment was selected for inspection
        :param applied_program: Identifier of the program applied or None
        """
        disposition_code = self.disposition(ok, must_inspect, applied_program)
        if self.file:
            self.writer.writerow(
                [
                    date.strftime("%Y-%m-%d"),
                    consignment["port"],
                    consignment["origin"],
                    consignment["flower"],
                    disposition_code,
                ]
            )
        elif self.print_to_stdout:
            print(
                f"F280: {date:%Y-%m-%d} | {consignment.port} | {consignment.origin}"
                f" | {consignment.flower} | {disposition_code}"
            )


class SuccessRates(object):
    """Record and accumulate success rates"""

    def __init__(self, reporter):
        """Initialize values to zero and set the reporter object"""
        self.ok = 0
        self.true_positive = 0
        self.true_negative = 0
        self.false_negative = 0
        self.reporter = reporter

    def record_success_rate(self, checked_ok, actually_ok, consignment):
        """Record testing result for one consignment

        :param checked_ok: True if no contaminant was found in consignment
        :param actually_ok: True if the consignment actually does not have contamination
        :param consignment: The shipment itself (for reporting purposes)
        """
        if checked_ok and actually_ok:
            self.true_negative += 1
            self.ok += 1
            self.reporter.true_negative()
        elif not checked_ok and not actually_ok:
            self.true_positive += 1
            self.reporter.true_positive()
        elif checked_ok and not actually_ok:
            self.false_negative += 1
            self.reporter.false_negative(consignment)
        elif not checked_ok and actually_ok:
            raise RuntimeError(
                "Inspection result is contaminated,"
                " but actually the consignment is not contaminated (programmer error)"
            )


def config_to_simplified_simulation_params(config):
    """Convert configuration into a simplified set of selected parameters"""
    sim_params = types.SimpleNamespace(
        tolerance_level="",
        contamination_unit="",
        contamination_type="",
        contamination_param="",
        contaminant_arrangement="",
        contaminated_units_per_cluster="",
        contaminant_distribution="",
        cluster_item_width="",
        inspection_unit="",
        within_box_proportion="",
        sample_strategy="",
        sample_params="",
        selection_strategy="",
        selection_param_1="",
        selection_param_2="",
    )

    sim_params.tolerance_level = config["inspection"]["tolerance_level"]
    sim_params.contamination_unit = config["contamination"]["contamination_unit"]
    sim_params.contamination_type = config["contamination"]["contamination_rate"][
        "distribution"
    ]
    if sim_params.contamination_type == "fixed_value":
        sim_params.contamination_param = config["contamination"]["contamination_rate"][
            "value"
        ]
    elif sim_params.contamination_type == "beta":
        sim_params.contamination_param = config["contamination"]["contamination_rate"][
            "parameters"
        ]
    else:
        sim_params.contamination_param = None
    sim_params.contaminant_arrangement = config["contamination"]["arrangement"]
    if sim_params.contaminant_arrangement == "clustered":
        sim_params.contaminated_units_per_cluster = config["contamination"][
            "clustered"
        ]["contaminated_units_per_cluster"]
        sim_params.contaminant_distribution = config["contamination"]["clustered"][
            "distribution"
        ]
        sim_params.cluster_sample_unit_width = config["contamination"]["clustered"]["random"][
            "cluster_sample_unit_width"
        ]
    else:
        sim_params.contaminated_units_per_cluster = None
        sim_params.cluster_sample_unit_width = None
        sim_params.contaminant_distribution = None
    sim_params.inspection_unit = config["inspection"]["unit"]
    sim_params.within_box_proportion = config["inspection"]["within_box_proportion"]
    sim_params.sample_strategy = config["inspection"]["sample_strategy"]
    if sim_params.sample_strategy == "proportion":
        sim_params.sample_params = config["inspection"]["proportion"]["value"]
    elif sim_params.sample_strategy == "hypergeometric":
        sim_params.sample_params = config["inspection"]["hypergeometric"][
            "detection_level"
        ]
    elif sim_params.sample_strategy == "fixed_n":
        sim_params.sample_params = config["inspection"]["fixed_n"]
    else:
        sim_params.sample_params = None
    sim_params.selection_strategy = config["inspection"]["selection_strategy"]
    if sim_params.selection_strategy == "cluster":
        sim_params.selection_param_1 = config["inspection"]["cluster"][
            "cluster_selection"
        ]
        if sim_params.selection_param_1 == "interval":
            sim_params.selection_param_2 = config["inspection"]["cluster"]["interval"]
    else:
        sim_params.selection_param_1 = None
        sim_params.selection_param_2 = None
    return sim_params


def print_totals_as_text(num_consignments, config, totals):
    """Prints simulation result as text"""
    # This is straightforward printing with simpler branches. Only few variables.
    # pylint: disable=too-many-branches,too-many-statements

    sim_params = config_to_simplified_simulation_params(config)

    # "On average, inspecting {0:.0f}% of consignments.".format(100 *
    #    totals.num_inspections / float(args.num_consignments))
    print("\n")
    print("Simulation parameters:")
    print("----------------------------------------------------------")
    print(f"consignments:\n\t Number consignments simulated: {num_consignments:,.0f}")
    print(
        "\t Avg. number of inspection_units per consignment: "
        f"{round(totals.num_inspection_units / num_consignments):,d}"
    )
    print(
        "\t Avg. number of sample_units per consignment: "
        f"{round(totals.num_sample_units / num_consignments):,d}"
    )

    print(
        f"contamination:\n\t unit: {sim_params.contamination_unit}\n\t type: "
        f"{sim_params.contamination_type}"
    )
    if sim_params.contamination_type == "fixed_value":
        print(f"\t\t contamination rate: {sim_params.contamination_param}")
    elif sim_params.contamination_type == "beta":
        print(
            "\t\t contamination distribution parameters: "
            f"{sim_params.contamination_param}"
        )
    print(f"\t contaminant arrangement: {sim_params.contaminant_arrangement}")
    if sim_params.contaminant_arrangement == "clustered":
        if sim_params.contamination_unit in ["inspection_unit", "inspection_units", "box", "boxes"]:
            print(
                "\t\t maximum contaminated inspection_units per cluster: "
                f"{sim_params.contaminated_units_per_cluster:,} inspection_units"
            )
        if sim_params.contamination_unit in ["sample_unit", "sample_units", "item", "items"]:
            print(
                "\t\t maximum contaminated sample_units per cluster: "
                f"{sim_params.contaminated_units_per_cluster:,} sample_units"
            )
            print(f"\t\t cluster distribution: {sim_params.contaminant_distribution}")
            if sim_params.contaminant_distribution == "random":
                print(f"\t\t cluster width: {sim_params.cluster_sample_unit_width:,} sample_units")

    print(
        f"inspection:\n\t unit: {sim_params.inspection_unit}\n\t sample strategy: "
        f"{sim_params.sample_strategy}"
    )
    if sim_params.sample_strategy == "proportion":
        print(f"\t\t value: {sim_params.sample_params}")
    elif sim_params.sample_strategy == "hypergeometric":
        print(f"\t\t detection level: {sim_params.sample_params}")
    elif sim_params.sample_strategy == "fixed_n":
        print(f"\t\t sample size: {sim_params.sample_params}")
    print(f"\t selection strategy: {sim_params.selection_strategy}")
    if sim_params.selection_strategy == "cluster":
        print(f"\t\t box selection strategy: {sim_params.selection_param_1}")
        if sim_params.selection_param_1 == "interval":
            print(f"\t\t box selection interval: {sim_params.selection_param_2}")
    if (
        sim_params.inspection_unit in ["inspection_unit", "inspection_units", "box", "boxes"]
        or sim_params.selection_strategy == "cluster"
    ):
        print(
            "\t minimum proportion of sample_units inspected within inspection_unit: "
            f"{sim_params.within_inspection_unit_proportion}"
        )
    print(f"\t tolerance level: {sim_params.tolerance_level}")
    print("\n")

    print("Simulation results: (averaged across all simulation runs)")
    print("----------------------------------------------------------")
    print(f"Avg. % contaminated consignments slipped: {totals.missing:.2f}%")
    if totals.false_neg + totals.intercepted:
        adj_avg_slipped = (
            (totals.false_neg - totals.missed_within_tolerance)
            / (totals.false_neg + totals.intercepted)
        ) * 100
    else:
        # For consignments with zero contamination
        adj_avg_slipped = 0

    print(
        "Adjusted avg. % contaminated consignments slipped (excluding slipped "
        "consignments with contamination rates below tolerance level): "
        f"{adj_avg_slipped:.2f}%"
    )
    print(f"Avg. num. consignments slipped: {totals.false_neg:,.0f}")
    print(
        "Avg. num. slipped consignments within tolerance "
        f"level: {totals.missed_within_tolerance:,.0f}"
    )
    print(f"Avg. num. consignments intercepted: {totals.intercepted:,.0f}")
    print(
        "Total number of slipped contaminants: "
        f"{totals.total_missed_contaminants:,.0f}"
    )
    print(
        "Total number of intercepted contaminants: "
        f"{totals.total_intercepted_contaminants:,.0f}"
    )
    print("Contamination rate:")
    print(f"\tOverall avg: {totals.true_contamination_rate:.3f}")
    if totals.max_missed_contamination_rate is not None:
        print(
            "\tSlipped consignments avg.: "
            f"{totals.avg_missed_contamination_rate:.3f}\n"
            "\tSlipped consignments max.: "
            f"{totals.max_missed_contamination_rate:.3f}"
        )
    if totals.max_intercepted_contamination_rate is not None:
        print(
            "\tIntercepted consignments avg.: "
            f"{totals.avg_intercepted_contamination_rate:.3f}\n"
            "\tIntercepted consignments max.: "
            f"{totals.max_intercepted_contamination_rate:.3f}"
        )
    print(
        "Avg. number of inspection_units opened per consignment:\n\t to completion: "
        f"{totals.avg_inspection_units_opened_completion:,.0f}\n"
        f"\t to detection: {totals.avg_inspection_units_opened_detection:,.0f}"
    )
    print(
        "Avg. number of sample_units inspected per consignment:\n\t to completion: "
        f"{totals.avg_sample_units_inspected_completion:,.0f}\n"
        f"\t to detection: {totals.avg_sample_units_inspected_detection:,.0f}"
    )
    print(
        "Avg. % contaminated items unreported if sample ends at detection: "
        f"{totals.pct_contaminant_unreported_if_detection:.2f}%"
    )


def get_item_from_nested_dict(dictionary, keys):
    """Get value from a nested dictionary by a nested keys-value pair"""
    return reduce(operator.getitem, keys, dictionary)


def _flatten_nested_dict_generator(dictionary, parent_key):
    for key, value in dictionary.items():
        new_key = f"{parent_key}/{key}" if parent_key else key
        if isinstance(value, MutableMapping):
            yield from flatten_nested_dict(value, new_key).items()
        else:
            yield new_key, value


def flatten_nested_dict(dictionary, parent_key=None):
    """Make a nested dictionary flat with key/subkey/subsubkey keys"""
    return dict(_flatten_nested_dict_generator(dictionary, parent_key))


def save_scenario_result_to_table(filename, results, config_columns, result_columns):
    """Save selected values for a scenario results to CSV including configuration

    The results parameter is list of tuples which is output from the run_scenarios()
    function.

    Values from configuration or results are selected by columns parameters which are
    in format key/subkey/subsubkey.
    """
    with open(filename, "w") as file:
        writer = csv.DictWriter(
            file,
            config_columns + result_columns,
            delimiter=",",
            quotechar='"',
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for result, config in results:
            row = {}
            for column in config_columns:
                keys = column.split("/")
                row[column] = get_item_from_nested_dict(config, keys)
            for column in result_columns:
                keys = column.split("/")
                row[column] = get_item_from_nested_dict(result.__dict__, keys)
            writer.writerow(row)


def save_simulation_result_to_pandas(
    result, config=None, config_columns=None, result_columns=None
):
    """Save result of one simulation to pandas DataFrame"""
    return save_scenario_result_to_pandas(
        [(result, config)], config_columns=config_columns, result_columns=result_columns
    )


def save_scenario_result_to_pandas(results, config_columns=None, result_columns=None):
    """Save selected values for a scenario to a pandas DataFrame.

    The results parameter is list of tuples which is output from the run_scenarios()
    function.

    Values from configuration or results are selected by columns parameters which are
    in format key/subkey/subsubkey.
    """
    # We don't want a special dependency to fail import of this file
    # in case this function is not used.
    import pandas as pd  # pylint: disable=import-outside-toplevel

    rows = []
    for result, config in results:
        row = {}
        if config:
            if config_columns:
                for column in config_columns:
                    keys = column.split("/")
                    row[column] = get_item_from_nested_dict(config, keys)
            elif config_columns is None:
                row = flatten_nested_dict(config)
            # When falsy, but not None, we assume it is an empty list and thus an
            # explicit request for no config columns to be included.
        if result_columns:
            for column in result_columns:
                keys = column.split("/")
                row[column] = get_item_from_nested_dict(result.__dict__, keys)
        else:
            row.update(vars(result))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def inspection_unit_detection_records_to_pandas(records):
    """Convert per-inspection-unit detection records (list of dicts) to DataFrame."""
    return pd.DataFrame.from_records(records)


def save_inspection_unit_detection_records_to_csv(records, filename):
    """Save per-inspection-unit detection records to CSV and return DataFrame."""
    df = pd.DataFrame.from_records(records)
    df.to_csv(filename, index=False)
    return df

class PISSimData:
    """
    Data collection class for simulation runs with multiple replications.
    Efficiently collects data during simulation and provides analysis tools.
    """

    # Class-level constants (shared across all instances)
    # Column schemas as class constants
    PIS_COLUMNS = [
        "comm_ID",
        "INSPECTION_ID",
        "REF_COMMODITY_ID",
        "CERTIFIED_FACILITY_NAME",
        "CERTIFIED_FACILITY_NUMBER",
        "COMMODITY_CLASSIFICATION",
        "COMMODITY_COMMON_NAME",
        "COMMODITY_DISPLAY_NAME",
        "COMMODITY_TAXONOMIC_DISPLAY_NAME",
        "COMMODITY_HOST_TYPE",
        "COMMODITY_TYPE",
        "COUNTRY_OF_ORIGIN_NAME",
        "CONSIGNEE_NAME",
        "DESTINATION_STATE_NAME",
        "DISPOSITION_CODE",
        "GENUS_NAME",
        "ENTRY_NUMBER",
        "ENTRY_LINE_NUMBER",
        "PGA_LINE_NUMBER",
        "PRODUCER_ID",
        "PRODUCER_NAME",
        "PROPAGATIVE_MATERIAL_TYPE",
        "QUANTITY",
        "QUANTITY_UNITS_NAME",
        "WADS_CODE",
        "SAMPLING_UNITS",
        "IS_RBS",
        "RBS_STATUS",
        "GROWING_MEDIA_PRESENCE",
        "CREATED_DATETIME",
        "INSPECTION_DATETIME",
        "BROKER_NAME",
        "CATEGORY",
        "SUBCATEGORY",
        "IMPORTER_NAME",
        "INSPECTION_LOCATION_NAME",
        "INSPECTION_LOCATION_ID",
        "INSPECTION_NUMBER",
        "PATHWAY_ID",
        "PATHWAY",
        "SHIPPER_NAME",
        "INSPECTION_LOCATION_STATE_CODE",
        "DOCUMENT_REVIEW_OVERTIME_ID",
        "DOCUMENT_REVIEW_OVERTIME_NAME",
        "INSPECTION_RESULTS_OVERTIME_ID",
        "INSPECTION_RESULTS_OVERTIME_NAME",
        "TAXONOMY_ORDER",
        "TAXONOMY_FAMILY",
        "TAXONOMY_GENUS",
        "TAXONOMY_SPECIES",
        "TAXONOMY_SUBSPECIES",
        "MODE_OF_TRANSPORT",
        "inspection",
        "shipment",
        "action",
        "HOST_PROXIMITY_ID",
        "HOST_PROXIMITY",
        "year",
        "month",
        "RISK_UNIT",
        "TOTAL_SAMPLING_UNITS_FOR_RISK_UNIT",
        "QUANTITY",
        "REQUIRED_NUMBER_OF_BOXES",
    ]

    RBS_COLUMNS = [
        "INSPECTION_ID",
        "INSPECTION_NUMBER",
        "INSPECTION_LOCATION_ID",
        "INSPECTION_LOCATION_NAME",
        "INSPECTION_LOCATION_STATE_CODE",
        "CATEGORY",
        "SUBCATEGORY",
        "PATHWAY_ID",
        "PATHWAY",
        "ID",
        "COUNTRY_OF_ORIGIN_ID",
        "COUNTRY_OF_ORIGIN_NAME",
        "PROPAGATIVE_MATERIAL_TYPE_ID",
        "PROPAGATIVE_MATERIAL_TYPE",
        "PRODUCER_ID",
        "PRODUCER_NAME",
        "TOTAL_SAMPLING_UNITS",
        "TOTAL_PLANT_QUANTITY",
        "SUBMITTED_DATETIME",
        "REMARKS",
        "CONFIDENCE_LEVEL",
        "RISK_RATING",
        "DETECTION_LEVEL",
        "REQUIRED_NUMBER_OF_BOXES",
        "PULL_NUMBERS",
        "IS_CERTIFIED_OFFSHORE_GREENHOUSE",
        "IS_ACTIVE",
        "CREATED_BY_USER_ID",
        "CREATED_BY_USER_FIRST_NAME",
        "CREATED_BY_USER_MIDDLE_NAME",
        "CREATED_BY_USER_LAST_NAME",
        "CREATED_DATETIME",
        "MODIFIED_BY_USER_ID",
        "MODIFIED_BY_USER_FIRST_NAME",
        "MODIFIED_BY_USER_MIDDLE_NAME",
        "MODIFIED_BY_USER_LAST_NAME",
        "MODIFIED_DATETIME",
        "remarks2",
        "pack_plant",
        'RISK_UNIT',
    ]

    CONSIGNMENT_COLUMNS = [
        'ID',
        'Origin',
        'Pathway',
        'Port',
        'Total Number of Risk Units',
        'Total Number of Sample Units',
        'Total Units (Plants)',
        'Total Units Infested',
        'Total Number of Risk Units Infested',
        'Total Number of Inspection Units Infested',
        'Total Units Infested in Each Inspection Unit',
    ]

    OUTPUT_FILENAMES = {
        "consignments": "synthetic_consignment_data.csv",
        "pis": "synthetic_pis_data.csv",
        "rbs": "synthetic_rbs_calc_data.csv",
        "commodity_line_results": "synthetic_commodity_line_results_data.csv",
    }

    def __init__(self, output_dir_rep: Optional[Path] = None, config: dict = None):
        """
        Initialize SimData for a simulation replication.

        Args:
            output_dir_rep: Optional directory path for saving replication data.
                           Directory will be created if it doesn't exist.
           config: Config object.
        """
        self.output_dir_rep = output_dir_rep
        if output_dir_rep:
            Path(output_dir_rep).mkdir(parents=True, exist_ok=True)

        # ID counter for consignments
        self.current_id: int = 0

        # Read in the input file that has the commodity line/inspection unit data that's being used for the simulation
        config = config["consignment"]
        generation_method = config["generation_method"]
        if (generation_method == "input_file") and (
                config["input_file"]["file_type"] == "PIS"
        ):
            filename = config["input_file"]["file_name"]
            self.pis_synthetic_data: Optional[pd.DataFrame] = pd.read_csv(filename, sep=",")
        else:
            self.pis_synthetic_data: Optional[pd.DataFrame] = None


        # Efficient collection using lists (converted to DataFrames later)
        self.rbs_records: List[Dict] = []
        self.consignment_records: List[Dict] = []
        self.inspection_unit_detection_records: List[Dict] = []

        # Final DataFrames (populated by finalize_dataframes())
        self.rbs_calc_synthetic_data: Optional[pd.DataFrame] = None
        self.consignments: Optional[pd.DataFrame] = None
        self.commodity_line_results: Optional[pd.DataFrame] = None

    def __repr__(self) -> str:
        """Provide useful string representation."""
        return (
            f"SimData(records_collected={self._commodity_line_record_count() + len(self.rbs_records) + len(self.consignment_records)}, "
            f"current_id={self.current_id}, "
            f"output_dir={self.output_dir_rep})"
        )

    def _commodity_line_record_count(self) -> int:
        if self.commodity_line_results is not None:
            return len(self.commodity_line_results)
        return len(self.inspection_unit_detection_records)

    @staticmethod
    def _has_rows(df: Optional[pd.DataFrame]) -> bool:
        return df is not None and not df.empty

    @classmethod
    def _output_paths(cls, output_path: Path) -> Dict[str, Path]:
        return {key: output_path / filename for key, filename in cls.OUTPUT_FILENAMES.items()}

    @staticmethod
    def _write_dataframe_to_csv(df: Optional[pd.DataFrame], destination: Path, empty_message: str) -> None:
        if df is not None and not df.empty:
            df.to_csv(destination, index=False)
        else:
            print(empty_message)

    def clear_all(self) -> None:
        """Clear all DataFrames and reset ID counter."""
        self.pis_synthetic_data = pd.DataFrame(columns=self.PIS_COLUMNS)
        self.rbs_calc_synthetic_data = pd.DataFrame(columns=self.RBS_COLUMNS)
        self.consignments = pd.DataFrame(columns=self.CONSIGNMENT_COLUMNS)
        self.commodity_line_results = pd.DataFrame()
        self.rbs_records = []
        self.consignment_records = []
        self.inspection_unit_detection_records = []
        self.current_id = 0


    def write_synthetic_data_to_csv(self) -> None:
        """
        Write all synthetic datasets to CSV files in the output directory.

        Creates three CSV files:
        - synthetic_consignment_data.csv: Consignment-level data
        - synthetic_pis_data.csv: Plant Inspection System records
        - synthetic_rbs_calc_data.csv: Risk-Based Sampling calculation data

        Raises:
            ValueError: If output_dir_rep is not set
            OSError: If unable to write to output directory

        Notes:
            - If using list collection pattern, call finalize_dataframes() first
            - Existing files will be overwritten
            - Creates output directory if it doesn't exist
        """
        if not self.output_dir_rep:
            raise ValueError(
                "output_dir_rep must be set before writing data. "
                "Initialize PISSimData with an output directory."
            )

        # Ensure output directory exists
        output_path = Path(self.output_dir_rep)
        output_path.mkdir(parents=True, exist_ok=True)
        output_files = self._output_paths(output_path)

        # Write DataFrames to CSV
        try:
            self._write_dataframe_to_csv(
                self.consignments,
                output_files["consignments"],
                "Warning: No consignment data to write",
            )
            self._write_dataframe_to_csv(
                self.pis_synthetic_data,
                output_files["pis"],
                "Warning: No PIS data to write",
            )
            self._write_dataframe_to_csv(
                self.rbs_calc_synthetic_data,
                output_files["rbs"],
                "Warning: No RBS data to write",
            )
            self._write_dataframe_to_csv(
                self.commodity_line_results,
                output_files["commodity_line_results"],
                "Warning: No commodity-line results to write",
            )

            print(f"Successfully wrote synthetic data to {output_path}")

        except OSError as e:
            raise OSError(f"Failed to write CSV files to {output_path}: {e}") from e

    def finalize_dataframes(self) -> None:
        """
        Convert collected record lists to pandas DataFrames.

        Call this method after data collection is complete and before writing to CSV
        or performing analysis. This is only needed if using the efficient list
        collection pattern.

        Notes:
            - Converts rbs_records and consignment_records to DataFrames
            - Safe to call multiple times (won't duplicate data)
            - No-op if records are already converted or empty
        """
        if self.rbs_records and not self._has_rows(self.rbs_calc_synthetic_data):
            self.rbs_calc_synthetic_data = pd.DataFrame(self.rbs_records, columns=self.RBS_COLUMNS)
            print(f"Finalized {len(self.rbs_records)} RBS calculator records")

        if self.consignment_records and not self._has_rows(self.consignments):
            self.consignments = pd.DataFrame(self.consignment_records, columns=self.CONSIGNMENT_COLUMNS)
            print(f"Finalized {len(self.consignment_records)} consignment records")

        self.commodity_line_results = inspection_unit_detection_records_to_pandas(self.inspection_unit_detection_records)

    def get_next_consignment_id(self) -> int:
        """
        Generate and return the next unique consignment ID.

        Increments the internal ID counter and returns the new value.
        IDs are sequential integers starting from 0.

        Returns:
            Integer ID for the next consignment
        """
        self.current_id += 1
        return self.current_id

    def reset_id_counter(self) -> None:
        """
        Reset the consignment ID counter to 0.

        Useful when starting a new replication or simulation run.
        """
        self.current_id = 0

    def get_summary_stats(self) -> dict:
        """
        Get summary statistics for all collected data.

        Returns:
            Dictionary with record counts and basic statistics
        """
        commodity_line_count = self._commodity_line_record_count()
        rbs_count = len(self.rbs_records) if self.rbs_records else (
            len(self.rbs_calc_synthetic_data) if self.rbs_calc_synthetic_data is not None else 0
        )
        consignment_count = len(self.consignment_records) if self.consignment_records else (
            len(self.consignments) if self.consignments is not None else 0
        )

        return {
            'commodity_line_records': commodity_line_count,
            'rbs_records': rbs_count,
            'consignments': consignment_count,
            'total_records': commodity_line_count + rbs_count + consignment_count,
            'current_id': self.current_id,
            'output_dir': str(self.output_dir_rep) if self.output_dir_rep else None,
        }


    def add_consignment(self, consignment: Consignment) -> None:
        """
            Add a consignment record to the dataset with contamination analysis.

            This method processes a Consignment object (which inherits from UserDict),
            analyzes contamination across inspection units and sample units, and adds
            the summarized data to the consignments collection.

            Args:
                consignment: A Consignment object containing:
                    - inspection_number: Unique identifier for the inspection
                    - origin: Country or location of origin
                    - pathway: Import pathway
                    - port: Port of entry
                    - num_sample_units: Total number of sample units
                    - plants_per_sample_unit: Number of plants per sample unit
                    - num_plants: Total number of plants in consignment
                    - plants: List/array of contamination counts
                    - inspection_units: List of InspectionUnit objects, each containing
                      included_unit_objects with plants attributes

            Notes:
                - Contamination is determined by checking if sum(consignment.plants) > 0
                - Each inspection unit is counted as contaminated only once, even if
                  multiple sample units within it are contaminated
                - Records are appended to internal list for efficient batch processing
            """
        # Optional validation
        self._validate_consignment(consignment)

        # Analyze contamination if present
        contamination_data = self._analyze_infestation(consignment)

        # Create consignment record
        consignment_record = {
            'ID': consignment.inspection_number,
            'Origin': consignment.origin,
            'Pathway': consignment.pathway,
            'Port': consignment.port,
            'Total Number of Risk Units': len(consignment.risk_units),
            'Total Number of Sample Units': consignment.num_sample_units,
            'Total Units (Plants)': consignment.num_plants,
            'Total Units Infested': contamination_data['total_infested'],
            'Total Number of Risk Units Infested': contamination_data['num_infested_risk_units'],
            'Total Number of Inspection Units Infested': contamination_data['num_infested_inspection_units'],
            'Total Units Infested in Each Inspection Unit': contamination_data['pests_per_inspection_unit'],
        }

        # Add to collection (efficient for batch processing)
        self.consignment_records.append(consignment_record)

    @staticmethod
    def _analyze_infestation(consignment: Consignment) -> Dict[str, Any]:
        """
        Analyze infestation distribution across risk, inspection, and sample units.

        Processes the consignment's inspection units to determine:
        1. Total infested plants
        2. Number of inspection units with at least one infested sample unit
        3. Detailed infestation count per sample unit within each inspection unit
        3. Number of risk units with at least one infested inspection unit

        Args:
            consignment: Consignment object with risk unit objects, inspection unit objects, sample
                         unit objects, and plant array

        Returns:
            Dictionary with keys:
                - total_infested (int): Sum of all infested plants
                - num_infested_inspection_units (int): Count of inspection units with contamination
                - pests_per_inspection_unit (List[List[int]]): Nested list where each inner
                  list contains contamination counts for sample units within an inspection unit
                - num_infested_risk_units (int):  Count of risk units on consignment with contamination


        Example:
            If consignment has 1 risk unit with 2 inspection units:
            - Inspection Unit 1: [0, 5, 0] (3 sample units, 1 infested with 5 plants)
            - Inspection Unit 2: [2, 0, 3] (3 sample units, 2 infested with 2 and 3 plants)

            Returns:
            {
                'total_infested': 10,
                'num_infested_units': 2,
                'contaminants_per_unit': [[0, 5, 0], [2, 0, 3]]
            }
            {
                'total_infested': 10,
                'num_infested_inspection_units': 2,
                'pests_per_inspection_unit': [[0, 5, 0], [2, 0, 3]],
                'num_infested_risk_units': 1,
            }
        """
        # Calculate total infested plants across entire consignment
        total_infested = sum(consignment.plants) if consignment.plants is not None else 0

        # Initialize infestation tracking structures
        pests_per_inspection_unit: List[List[int]] = []
        num_infested_inspection_units = 0
        num_infested_risk_units = 0

        # Only analyze if contamination is present
        if total_infested > 0:
            for risk_unit in consignment.risk_units:
                infection_indicator = 0
                for inspection_unit_id in risk_unit.inspection_unit_ids:
                    inspection_unit = consignment.inspection_units[inspection_unit_id]
                    contaminants_per_sample_unit: List[int] = []
                    unit_has_contamination = False

                    # Check each sample unit within the inspection unit
                    for sample_unit in inspection_unit.included_unit_objects:
                        contamination_count = sum(sample_unit.plants) if sample_unit.plants is not None else 0
                        contaminants_per_sample_unit.append(contamination_count)

                        # Mark that this inspection unit has contamination
                        if contamination_count > 0:
                            unit_has_contamination = True

                    # Store contamination data for this inspection unit
                    pests_per_inspection_unit.append(contaminants_per_sample_unit)

                    # Count this inspection unit if it has any contamination
                    if unit_has_contamination:
                        num_infested_inspection_units += 1
                        infection_indicator = 1
                if infection_indicator == 1: num_infested_risk_units+=1


        return {
            'total_infested': total_infested,
            'num_infested_inspection_units': num_infested_inspection_units,
            'pests_per_inspection_unit': pests_per_inspection_unit,
            'num_infested_risk_units': num_infested_risk_units,
        }

    @staticmethod
    def _validate_consignment(consignment: Consignment) -> None:
        """
        Validate that consignment has required attributes.

        Args:
            consignment: Consignment object to validate

        Raises:
            AttributeError: If required attributes are missing
            ValueError: If data is invalid
        """
        required_attrs = [
            'inspection_number', 'origin', 'pathway', 'port',
            'num_sample_units', 'num_plants', 'inspection_units'
        ]

        for attr in required_attrs:
            if not hasattr(consignment, attr) or getattr(consignment, attr) is None:
                raise AttributeError(
                    f"Consignment missing required attribute: {attr}"
                )

        if consignment.num_inspection_units != len(consignment.inspection_units):
            raise ValueError(
                f"Mismatch: num_inspection_units={consignment.num_inspection_units} "
                f"but got {len(consignment.inspection_units)} inspection_units"
            )

    def add_to_pis_synthetic_data(
            self,
            ret: SimpleNamespace,
            consignment: Consignment,
            n_units_to_inspect: int
    ) -> None:
        """
        Add inspection results to PIS and RBS synthetic datasets.

        For each inspected box/inspection unit in the inspection results, creates
        corresponding records in both the PIS (Plant Inspection Data) and RBS
        (Risk Calculator Data) calculation datasets.

        Args:
            ret: SimpleNamespace object returned from inspect() containing:
                - inspected_box_indexes: List of inspection unit indices that were inspected
                - inspected_box_result: List of results per box (1=contamination found, 0=clean)
                - inspected_sample_unit_indexes: List of all sample unit indices inspected
                - inspection_units_opened_completion: Total inspection units opened
                - sample_units_inspected_completion: Total sample units inspected
                - contaminated_sample_units_completion: Total contaminated units found
                - consignment_checked_ok: Boolean indicating if consignment passed
            consignment: Consignment object with inspection details
            n_units_to_inspect: Required number of boxes/units to inspect (from sampling plan)

        Notes:
            - Creates one PIS record and one RBS record per inspected box
            - Only processes records if inspected_box_result is populated
            - Uses efficient list collection for batch DataFrame creation
            - Box result codes: 1 = contamination found, 0 = no contamination

        Raises:
            ValueError: If inspected_box_result length doesn't match inspected_box_indexes
        """

        sample_unit_to_inspection = getattr(consignment, "sample_unit_to_inspection_unit", {}) or {}
        inspected_sample_unit_indexes = ret.inspected_sample_unit_indexes
        inspected_counts_by_inspection_unit = Counter()
        for sample_unit_index in ret.inspected_sample_unit_indexes:
            inspection_unit_index = sample_unit_to_inspection.get(
                sample_unit_index,
                consignment.get_inspection_unit_and_sample_unit_index(sample_unit_index)[0],
            )
            inspected_counts_by_inspection_unit[inspection_unit_index] += 1

        for inspection_unit_index, inspection_unit in enumerate(consignment.inspection_units):
            included_unit_objects = getattr(inspection_unit, "included_unit_objects", [])
            num_plants_in_inspection_unit = sum(len(su.plants) for su in included_unit_objects)
            infected_plants_in_inspection_unit = sum(
                int(np.count_nonzero(su.plants)) for su in included_unit_objects
            )
            risk_unit_id = None
            risk_unit_ids = getattr(inspection_unit, "risk_unit_ids", None)
            if risk_unit_ids:
                risk_unit_id = risk_unit_ids[0]
            elif getattr(inspection_unit, "risk_unit", None) is not None:
                risk_unit_id = inspection_unit.risk_unit.id

            is_infected = bool(getattr(inspection_unit, "is_infected", bool(inspection_unit)))
            is_detected = bool(getattr(inspection_unit, "is_detected", False))
            self.inspection_unit_detection_records.append(
                {
                    "inspection_number": getattr(consignment, "inspection_number", None),
                    "inspection_unit_index": inspection_unit_index,
                    "inspection_unit_id": getattr(inspection_unit, "id", inspection_unit_index),
                    "risk_unit_id": risk_unit_id,
                    "num_sample_units": getattr(inspection_unit, "num_sample_units", len(included_unit_objects)),
                    "num_plants": num_plants_in_inspection_unit,
                    "infected_plants": infected_plants_in_inspection_unit,
                    "is_infected": is_infected,
                    "is_detected": is_detected,
                    "action": 1 if is_detected else 0,
                    "missed": bool(is_infected and not is_detected),
                    "was_inspected": bool(
                        inspected_counts_by_inspection_unit[inspection_unit_index] > 0
                    ),
                    "inspected_sample_units": int(inspected_counts_by_inspection_unit[inspection_unit_index]),
                }
            )


        # Create the calculator records
        self._create_rbs_records(
            consignment=consignment,
            n_units_to_inspect=n_units_to_inspect,
            inspection_result=ret
        )


    def _create_rbs_records(
            self,
            consignment: Consignment,
            n_units_to_inspect: int,
            inspection_result: SimpleNamespace
    ) -> None:
        """
        Create a single RBS (Risk-Based Sampling) calculation record.

        Args:
            consignment: Consignment object with inspection details
            n_units_to_inspect: Required number of boxes to inspect
            inspection_result: SimpleNamespace from inspect() with inspection metrics

        Returns:
            Dictionary with RBS record data including sampling plan and results
        """
        for risk_unit in consignment.risk_units:
            rbs_record = {
                'RISK_UNIT': risk_unit.id,
                'INSPECTION_NUMBER': consignment.inspection_number,
                'COUNTRY_OF_ORIGIN_NAME': risk_unit.origin,
                'PATHWAY': risk_unit.pathway,
                'TOTAL_PLANT_QUANTITY': len(risk_unit.plant_ids),
                'TOTAL_SAMPLING_UNITS': risk_unit.num_sample_units,
                'REQUIRED_NUMBER_OF_BOXES': len(risk_unit.sample_unit_ids),
                # Optional: Add inspection results if needed in RBS data
                # 'INSPECTION_UNITS_OPENED': inspection_result.inspection_units_opened_completion,
                # 'SAMPLE_UNITS_INSPECTED': inspection_result.sample_units_inspected_completion,
                # 'CONTAMINATED_UNITS_FOUND': inspection_result.contaminated_sample_units_completion,
            }

            # Add to collections (efficient batch approach)
            self.rbs_records.append(rbs_record)



