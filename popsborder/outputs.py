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
.. codeauthor:: Gary Lin <Gary.Lin@jhuapl.edu>
.. codeauthor:: Joseph Agor <Joseph.Agor@jhuapl.edu>
"""

import csv
import operator
import shutil
import types
import weakref
from collections.abc import MutableMapping
from functools import reduce
import pandas as pd
import os
from pathlib import Path

from .inspections import count_contaminated_boxes
from slippage_model_utils.clarke_r_script_wrapper import _find_repo_root

# TODO: Update how we save out any output data (currently will save to an "outputs" folder in the repository similar to what is done in slippage.py)
# Define output directory for the simulated data
output_dir = _find_repo_root() / "output" / "pops_border_sim_data"


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
        [pretty_content(inspection_unit.sample_units, config=config) for inspection_unit in consignment["inspection_units"]]
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


class SimData(object):

    def __init__(self):
        """Initialize output files"""
        self.pis_synthetic_data = pd.DataFrame(
            columns=[
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
                "TOTAL_SAMPLING_UNITS (rbs calc data)",
                "REQUIRED_NUMBER_OF_BOXES (rbs calc data)",
            ]
        )

        self.rbs_calc_synthetic_data = pd.DataFrame(
            columns=[
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
                "pack_plant"
            ]
        )

        self.consignments = pd.DataFrame(
            columns = [
                'ID',
                'Origin',
                'Pathway',
                'Port',
                'Total Number of Sample Units',
                'Plants Per Sample Unit',
                'Total Items',
                'Plants Per Item',
                'Total Plants',
                'Total Plants Contaminated',
                'Total Number of Inspection Units Contaminated',
                'Total Number Contaminated in Each Inspection Unit',
            ]
        )
        self.current_id = 0


    def write_synthetic_data_to_csv(self,output_dir=output_dir):
        # Create the output directory if it does not exist
        output_dir.mkdir(parents=True,exist_ok=True)
        filepath = os.path.join(output_dir, 'synthetic_consignment_data.csv')
        self.consignments.to_csv(filepath, index = False)
        filepath = os.path.join(output_dir, 'synthetic_pis_data.csv')
        self.pis_synthetic_data.to_csv(filepath, index = False)
        filepath = os.path.join(output_dir, 'synthetic_rbs_calc_data.csv')
        self.rbs_calc_synthetic_data.to_csv(filepath, index = False)

    def gen_consignment_id(self):
        self.current_id += 1
        return str(self.current_id)

    def add_consignment(self,consignment):

        # If contaminated, determine which sample units are contaminated
        num_contaminats_per_inspection_unit = []
        num_contaminated_inspection_units = 0
        if sum(consignment.plants)>0:
            for inspection_unit in consignment.inspection_units:
                num_contaminats_per_sample_unit = []
                for sample_unit in inspection_unit.sample_unit_objects:
                    num_contaminats_per_sample_unit.append(sum(sample_unit.plants))
                    if sum(sample_unit.plants)>0:
                        num_contaminated_inspection_units += 1
                num_contaminats_per_inspection_unit.append(num_contaminats_per_sample_unit)


        # Fill with placeholder values based on column type/meaning
        default_row_consignment = {col: pd.NA for col in self.consignments.columns}

        # Set column values for consignment data
        default_row_consignment.update({
            'ID': consignment.inspection_number,
            'Origin': consignment.origin,
            'Pathway': consignment.pathway,
            'Port': consignment.port,
            'Total Number of Sample Units': consignment.num_sample_units,
            'Plants Per Sample Unit': consignment.plants_per_sample_unit,
            'Total Plants': consignment.num_plants,
            'Total Plants Contaminated': sum(consignment.plants),
            'Total Number of Inspection Units Contaminated': num_contaminated_inspection_units ,
            'Total Number Contaminated in Each Inspection Unit': num_contaminats_per_inspection_unit,
        })

        # Add rows to synthetic consignment data set
        self.consignments.loc[len(self.consignments)] = default_row_consignment




    def add_to_synthetic_data(self,ret,consignment, n_units_to_inspect):
        # Loop through each box that was inspected and add a row to PIS synthetic data the
        for box in range(len(ret.inspected_box_indexes)):
            # Fill with placeholder values based on column type/meaning
            default_row_pis = {col: pd.NA for col in self.pis_synthetic_data.columns}
            default_row_rbs_calc = {col: pd.NA for col in self.rbs_calc_synthetic_data}

            # Set column values for rbs_calc data
            default_row_rbs_calc.update({
                'TOTAL_PLANT_QUANTITY': consignment.num_plants,
                "TOTAL_SAMPLING_UNITS": consignment.num_sample_units,
                "REQUIRED_NUMBER_OF_BOXES": n_units_to_inspect,
                "INSPECTION_ID": self.current_id,
                "COUNTRY_OF_ORIGIN_NAME": consignment.origin,
                "PATHWAY": consignment.pathway,
            })

            # Set column values for pis data
            default_row_pis.update({
                'QUANTITY': consignment.num_plants,
                "INSPECTION_ID": consignment.inspection_number,
                "COUNTRY_OF_ORIGIN_NAME": consignment.origin,
                "year": consignment.date.year,
                "month": consignment.date.month,
                "PATHWAY": consignment.pathway,
                "action": ret.inspected_box_result[box],
                "TOTAL_SAMPLING_UNITS (rbs calc data)": consignment.num_sample_units,
                "REQUIRED_NUMBER_OF_BOXES (rbs calc data)": n_units_to_inspect,
            })

            # Add rows to synthetic data sets
            self.pis_synthetic_data.loc[len(self.pis_synthetic_data)] = default_row_pis
            self.rbs_calc_synthetic_data.loc[len(self.rbs_calc_synthetic_data)] = default_row_rbs_calc