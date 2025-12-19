# Simulation of contaminated consignments and their inspections
# Copyright (C) 2018-2021 Vaclav Petras and others (see below)

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


"""Inspections of consignments

.. codeauthor:: Vaclav Petras <wenzeslaus gmail com>
.. codeauthor:: Kellyn P. Montgomery <kellynmontgomery gmail com>

=====================================
JHU/APL Extensions and Modifications:
=====================================

Contributors: Gary Lin, Joseph Agor (Johns Hopkins University Applied Physics Laboratory)

New Functions Added:
-------------------
- sample_rbs():
    * Implements risk-based sampling methodology using compliance-based detection levels
    * Retrieves country/propagative material specific compliance parameters from lookup table
    * Calculates sample size using hypergeometric distribution based on risk assessment

- select_units_to_inspect():
    * Unified function for selecting units to inspect based on selection strategy
    * Supports random, cluster, and convenience selection strategies
    * Handles both inspection_unit and sample_unit selection

- count_contaminated_inspection_units():
    * Counts contaminated inspection units in consignment
    * Supports refactored terminology (inspection_units vs boxes)

- count_contaminated_sample_units():
    * Counts contaminated sample units in consignment
    * Supports refactored terminology (sample_units vs items)

Modified Functions:
----------------
- get_sample_function():
    * Added RBS structured consignment inspection
    * Enhanced to support compliance table parameter passing

- sample_proportion():
    * Added backward compatibility for min_inspection_units (formerly min_boxes)
    * Updated to handle both old and new terminology in configuration

- sample_n():
    * Added backward compatibility for within_inspection_unit_proportion (formerly within_box_proportion)
    * Added backward compatibility for min_inspection_units (formerly min_boxes)

- convert_sample_units_to_inspection_units_fixed_proportion():
    * Added backward compatibility for within_inspection_unit_proportion

- compute_n_clusters_to_inspect():
    * Added backward compatibility for within_inspection_unit_proportion and min_inspection_units

- inspect():
    * Added backward compatibility for within_inspection_unit_proportion
    * Enhanced detailed tracking with sample_unit_in_inspection_unit_to_sample_unit_index()

- get_detection_and_confidence():
    * Added ability to process beyond just origin and PM Type

Backward Compatibility:
----------------------
- Added aliases: count_contaminated_boxes() -> count_contaminated_inspection_units()
- Added aliases: count_contaminated_items() -> count_contaminated_sample_units()
- Configuration parameter mapping: boxes -> inspection_units, items -> sample_units
- Maintained support for legacy configuration keys while enabling new terminology

"""

import math
import os
import random
import types

import numpy as np

from .inputs import get_validated_effectiveness, load_compliance_lookup_csv

from slippage_model_utils.references import country_of_origin_names, pm_type_names
import re
from collections import defaultdict


def inspect_first(consignment):
    """Inspect only the first inspection_unit in the consignment"""
    if consignment.inspection_units[0]:
        return False, 1
    return True, 1


def inspect_one_random(consignment):
    """Inspect only one randomly picked inspection_unit in the consignment"""
    if random.choice(consignment.inspection_units):
        return False, 1
    return True, 1


def inspect_all(consignment):
    """Inspect all inspection_units in the consignment"""
    return not is_consignment_contaminated(consignment), consignment.num_inspection_units


def inspect_first_n(num_inspection_units, consignment):
    """Inspect only the first n inspection_units in the consignment

    :param num_inspection_units: Number of inspection_units to inspect
    :param consignment: Consignment to inspect
    """
    num_inspection_units = min(len(consignment.inspection_units), num_inspection_units)
    for i in range(num_inspection_units):
        if consignment.inspection_units[i]:
            return False, i + 1
    return True, num_inspection_units


def sample_proportion(config, consignment):
    """Set sample size to sample units from consignment using proportion strategy.
    Return number of units to inspect.

    :param config: Configuration to be used
    :param consignment: Consignment to be inspected
    """
    unit = config["inspection"]["unit"]
    ratio = config["inspection"]["proportion"]["value"]
    num_sample_units = consignment.num_sample_units
    num_inspection_units = consignment.num_inspection_units
    # Handle backward compatibility for min inspection units
    min_inspection_units = config["inspection"].get("min_inspection_units",
                                                      config["inspection"].get("min_boxes", 0))

    if unit in ["sample_unit", "sample_units", "item", "items"]:
        n_units_to_inspect = round(ratio * num_sample_units)
    elif unit in ["inspection_unit", "inspection_units", "box", "boxes"]:
        n_units_to_inspect = round(ratio * num_inspection_units)
        n_units_to_inspect = max(min_inspection_units, n_units_to_inspect)
        n_units_to_inspect = min(num_inspection_units, n_units_to_inspect)
    else:
        raise RuntimeError(f"Unknown sampling unit: {unit}")
    return n_units_to_inspect


def compute_hypergeometric(detection_level, confidence_level, population_size):
    """Get sample size using hypergeometric distribution

    Compute sample size using hypergeometric distribution based on population
    size (total number of items or boxes in consignment), detection level,
    and confidence level.
    """
    detection_level = float(detection_level)
    confidence_level = float(confidence_level)

    # Equation comes from RBS spreadsheet for calculating hypergeometric
    # sample sizes created by IICA, USDA APHIS PPQ, and NAPPO.
    sample_size = math.ceil(
        (1 - ((1 - confidence_level) ** (1 / (detection_level * population_size))))
        * (population_size - (((detection_level * population_size) - 1) / 2))
    )

    # The computation gives sample size > num boxes when using 1% detection
    # Make max sample size = population size
    sample_size = min(sample_size, population_size)
    return sample_size


def sample_hypergeometric(config, consignment):
    """Set sample size to sample units from consignment using hypergeometric/detection
    level strategy. Return number of units to inspect.

    :param config: Configuration to be used
    :param consignment: Consignment to be inspected
    """
    unit = config["inspection"]["unit"]
    detection_level = config["inspection"]["hypergeometric"]["detection_level"]
    confidence_level = config["inspection"]["hypergeometric"]["confidence_level"]
    num_sample_units = consignment.num_sample_units
    num_inspection_units = consignment.num_inspection_units

    if unit in ["sample_unit", "sample_units", "item", "items"]:
        n_units_to_inspect = compute_hypergeometric(
            detection_level, confidence_level, num_sample_units
        )
    elif unit in ["inspection_unit", "inspection_units", "box", "boxes"]:
        n_units_to_inspect = compute_hypergeometric(
            detection_level, confidence_level, num_inspection_units
        )
    else:
        raise RuntimeError(f"Unknown sampling unit: {unit}")
    return n_units_to_inspect


def sample_all(config, consignment):
    """Set sample size to sample all units from consignment.
    Return number of units to inspect.

    :param config: Configuration to be used
    :param consignment: Consignment to be inspected
    """
    unit = config["inspection"]["unit"]
    if unit in ["sample_unit", "sample_units", "item", "items"]:
        n_units_to_inspect = consignment.num_sample_units
    elif unit in ["inspection_unit", "inspection_units", "box", "boxes"]:
        n_units_to_inspect = consignment.num_inspection_units
    return n_units_to_inspect


def sample_n(config, consignment):
    """Set sample size to sample fixed number of units from consignment.
    Check if fixed number is <= max units for inspection.
    Return number of units to inspect.

    :param config: Configuration to be used
    :param consignment: Consignment to be inspected
    """
    fixed_n = config["inspection"]["fixed_n"]
    unit = config["inspection"]["unit"]
    # Handle backward compatibility for within inspection unit proportion
    within_inspection_unit_proportion = config["inspection"].get("within_inspection_unit_proportion",
                                                                   config["inspection"].get("within_box_proportion", 1.0))
    sample_units_per_inspection_unit = consignment.sample_units_per_inspection_unit
    num_sample_units = consignment.num_sample_units
    num_inspection_units = consignment.num_inspection_units
    # Handle backward compatibility for min inspection units
    min_inspection_units = config["inspection"].get("min_inspection_units",
                                                      config["inspection"].get("min_boxes", 0))

    if unit in ["sample_unit", "sample_units", "item", "items"]:
        max_sample_units = compute_max_inspectable_sample_units(
            num_sample_units, sample_units_per_inspection_unit, within_inspection_unit_proportion
        )
        # Check if max number of sample_units that can be inspected is less than fixed number.
        n_units_to_inspect = min(max_sample_units, fixed_n)
    elif unit in ["inspection_unit", "inspection_units", "box", "boxes"]:
        n_units_to_inspect = fixed_n
        n_units_to_inspect = max(min_inspection_units, n_units_to_inspect)
        n_units_to_inspect = min(num_inspection_units, n_units_to_inspect)
    return n_units_to_inspect


def sample_rbs(config, consignment, compliance_table_dict):
    """Set sample size to sample units from consignment using hypergeometric/detection 
    level strategy based on compliance levels. Return number of units to inspect.

    :param config: Configuration to be used
    :param consignment: Consignment to be inspected
    """

    unit = config["inspection"]["unit"]
    num_sample_units = consignment.num_sample_units
    num_inspection_units = consignment.num_inspection_units
    detection_confidence_levels = get_detection_and_confidence(consignment, compliance_table_dict)
    n_units_to_inspect = {}
    if unit in ["sample_unit", "sample_units", "item", "items"]:
        for inspect_number in detection_confidence_levels.keys():
            detection_level, confidence_level = detection_confidence_levels[inspect_number][0], \
                detection_confidence_levels[inspect_number][1]
            num_sample_units = consignment.inspection_units[inspect_number].num_sample_units
            n_units_to_inspect[inspect_number] = compute_hypergeometric(
                detection_level, confidence_level, num_sample_units
            )
    elif unit in ["inspection_unit", "inspection_units", "box", "boxes"]:
        for inspect_number in detection_confidence_levels.keys():
            detection_level, confidence_level = detection_confidence_levels[inspect_number][0], \
            detection_confidence_levels[inspect_number][1]
            n_units_to_inspect[inspect_number] = compute_hypergeometric(
                detection_level, confidence_level, num_sample_units
            )
    else:
        raise RuntimeError(f"Unknown sampling unit: {unit}")
    return n_units_to_inspect


def convert_sample_units_to_inspection_units_fixed_proportion(config, consignment, n_sample_units_to_inspect):
    """Convert number of sample_units to inspect to number of inspection_units to inspect based on
    the number of sample_units per inspection_unit and the proportion of sample_units to inspect per inspection_unit
    specified in the config. Adjust number of inspection_units to inspect to be at least
    the minimum number of inspection_units to inspect specified in the config and at most the
    total number of inspection_units in the consignment.
    Return number of inspection_units to inspect.

    :param config: Configuration to be used
    :param consignment: Consignment to be inspected
    :param n_sample_units_to_inspect: Number of sample_units to inspect defined in sample functions.
    """
    sample_units_per_inspection_unit = consignment.sample_units_per_inspection_unit
    # Handle backward compatibility for within inspection unit proportion
    within_inspection_unit_proportion = config["inspection"].get("within_inspection_unit_proportion",
                                                                   config["inspection"].get("within_box_proportion", 1.0))
    min_inspection_units = config["inspection"]["min_inspection_units"]
    num_inspection_units = consignment.num_inspection_units
    inspect_per_inspection_unit = int(math.ceil(within_inspection_unit_proportion * sample_units_per_inspection_unit))

    n_inspection_units_to_inspect = math.ceil(n_sample_units_to_inspect / inspect_per_inspection_unit)
    n_inspection_units_to_inspect = max(min_inspection_units, n_inspection_units_to_inspect)
    n_inspection_units_to_inspect = min(num_inspection_units, n_inspection_units_to_inspect)
    return n_inspection_units_to_inspect


def compute_n_clusters_to_inspect(config, consignment, n_sample_units_to_inspect):
    """Compute number of cluster units (inspection_units) that need to be opened to achieve sample_unit
    sample size when using the cluster selection strategy. Use config within inspection_unit
    proportion if possible or compute minimum number of sample_units to inspect per inspection_unit
    required to achieve sample_unit sample size.
    Return number of inspection_units to inspect and number of sample_units to inspect per inspection_unit.

    :param config: Configuration to be used
    :param consignment: Consignment to be inspected
    :param n_sample_units_to_inspect: Number of sample_units to inspect defined by sample functions.
    """
    cluster_selection = config["inspection"]["cluster"]["cluster_selection"]
    sample_units_per_inspection_unit = consignment.sample_units_per_inspection_unit
    # Handle backward compatibility for within inspection unit proportion
    within_inspection_unit_proportion = config["inspection"].get("within_inspection_unit_proportion",
                                                                   config["inspection"].get("within_box_proportion", 1.0))
    # Handle backward compatibility for min inspection units
    min_inspection_units = config["inspection"].get("min_inspection_units",
                                                      config["inspection"].get("min_boxes", 0))
    num_inspection_units = consignment.num_inspection_units
    num_sample_units = consignment.num_sample_units

    if cluster_selection == "random":
        # Check if within box proportion is high enough to achieve sample size.
        max_sample_units = compute_max_inspectable_sample_units(
            num_sample_units, sample_units_per_inspection_unit, within_inspection_unit_proportion
        )
        if max_sample_units >= n_sample_units_to_inspect:
            inspect_per_inspection_unit = math.ceil(within_inspection_unit_proportion * sample_units_per_inspection_unit)
            n_inspection_units_to_inspect = math.ceil(n_sample_units_to_inspect / inspect_per_inspection_unit)
        else:
            # If not, divide sample size across number of inspection_units to get number
            # of sample_units to inspect per inspection_unit.
            print(
                "Warning: Within inspection_unit proportion is too low to achieve sample size. "
                "Automatically increasing within inspection_unit proportion to achieve sample size."
            )
            inspect_per_inspection_unit = math.ceil(n_sample_units_to_inspect / num_inspection_units)
            n_inspection_units_to_inspect = math.ceil(n_sample_units_to_inspect / inspect_per_inspection_unit)

    elif cluster_selection == "interval":  # Every nth inspection_unit, where n = interval
        interval = config["inspection"]["cluster"]["interval"]
        # Maximum num inspection_units that can be inspected based on interval.
        # Should be at least 1.
        max_inspection_units = max(1, round(num_inspection_units / interval))
        # Assumes full inspection_units, no remainder partial inspection_unit.
        max_sample_units = max_inspection_units * (math.ceil(within_inspection_unit_proportion * sample_units_per_inspection_unit))
        # Check if within inspection_unit proportion is high enough and/or interval is
        # low enough to achieve sample size
        if max_sample_units >= n_sample_units_to_inspect:
            inspect_per_inspection_unit = math.ceil(within_inspection_unit_proportion * sample_units_per_inspection_unit)
            n_inspection_units_to_inspect = math.ceil(n_sample_units_to_inspect / inspect_per_inspection_unit)
        # If not, divide sample size across max inspection_units to get number of
        # sample_units to inspect per inspection_unit.
        else:
            print(
                "Warning: Within inspection_unit proportion is too low and/or interval is too "
                "high to achieve sample size. Automatically increasing within inspection_unit "
                "proportion to achieve sample size."
            )
            inspect_per_inspection_unit = math.ceil(n_sample_units_to_inspect / max_inspection_units)
            # If not enough inspection_units to achieve sample size, inspect all sample_units
            # and increase n_inspection_units_to_inspect as needed.
            inspect_per_inspection_unit = min(inspect_per_inspection_unit, sample_units_per_inspection_unit)
            n_inspection_units_to_inspect = math.ceil(n_sample_units_to_inspect / inspect_per_inspection_unit)
    else:
        raise RuntimeError(f"Unknown cluster selection method: {cluster_selection}")

    # Allow user specified inspection_unit minimum override calculations
    n_inspection_units_to_inspect = max(min_inspection_units, n_inspection_units_to_inspect)
    assert num_inspection_units >= n_inspection_units_to_inspect

    return n_inspection_units_to_inspect, inspect_per_inspection_unit


def compute_max_inspectable_sample_units(num_sample_units, sample_units_per_inspection_unit, within_inspection_unit_proportion):
    """Compute maximum number of sample_units that can be inspected in a consignment based
    on within inspection_unit proportion. If within inspection_unit proportion is less than 1 (partial inspection_unit
    inspections), then maximum number of sample_units that can be inspected will be
    less than the total number of sample_units in the consignment.

    :param num_sample_units: total number of sample_units in consignment
    :param sample_units_per_inspection_unit: number of sample_units in each inspection_unit
    :param within_inspection_unit_proportion: proportion of sample_units to be inspected per inspection_unit
    """
    inspect_per_inspection_unit = math.ceil(within_inspection_unit_proportion * sample_units_per_inspection_unit)
    num_full_inspection_units = math.floor(num_sample_units / sample_units_per_inspection_unit)
    full_inspection_unit_inspectable_sample_units = num_full_inspection_units * inspect_per_inspection_unit
    remainder_inspection_unit = num_sample_units % sample_units_per_inspection_unit
    # Assume that num of sample_units to inspect is based on num of sample_units
    # in full inspection_unit. Same inspect_per_inspection_unit will be applied to
    # full and partial inspection_units.
    remainder_inspection_unit_inspectable_sample_units = min(remainder_inspection_unit, inspect_per_inspection_unit)
    max_sample_units = full_inspection_unit_inspectable_sample_units + remainder_inspection_unit_inspectable_sample_units
    return max_sample_units


def select_random_indexes(unit, consignment, n_units_to_inspect):
    """Select units (indexes) from consignment based on sample size and
    random selection strategy.

    :param unit: Unit to be used for inspection (inspection_unit or sample_unit)
    :param consignment: Consignment to be inspected
    :param n_units_to_inspect: Number of units to inspect defined in sample functions.
    """
    if unit in ["sample_unit", "sample_units", "item", "items"]:
        indexes_to_inspect = random.sample(
            list(range(consignment.num_sample_units)), n_units_to_inspect
        )
    elif unit in ["inspection_unit", "inspection_units", "box", "boxes"]:
        indexes_to_inspect = random.sample(
            list(range(consignment.num_inspection_units)), n_units_to_inspect
        )
    else:
        raise RuntimeError(f"Unknown unit: {unit}")
    indexes_to_inspect.sort()
    return indexes_to_inspect


def select_random_indexes_rbs(unit, consignment, n_units_to_inspect):
    """Select units (indexes) from consignment based on sample size and
    random selection strategy.

    :param unit: Unit to be used for inspection (inspection_unit or sample_unit)
    :param consignment: Consignment to be inspected
    :param n_units_to_inspect: Number of units to inspect defined in sample functions.
    """
    
    indexes_to_inspect = []
    if unit in ["sample_unit", "sample_units", "item", "items"]:
        current_idx = 0
        inspection_unit_counter = 0
        inspection_units_to_inspect = {}
        for inspection_unit in consignment.inspection_units:
            indexes_to_inspect_temp = random.sample(
                list(range(len(inspection_unit.sample_unit_objects))), n_units_to_inspect[inspection_unit_counter]
            )
            inspection_units_to_inspect[inspection_unit_counter] = indexes_to_inspect_temp
            indexes_to_inspect_temp = [x+ current_idx for x in indexes_to_inspect_temp]
            current_idx += len(inspection_unit.sample_unit_objects)
            indexes_to_inspect = indexes_to_inspect + indexes_to_inspect_temp
            inspection_unit_counter += 1
    else:
        raise RuntimeError(f"Inspection process unit specified in config is: {unit}.  "
                           f"For Sampling Strategy = RBS, only supports that parameter being = sampling_units")
    indexes_to_inspect.sort()
    return indexes_to_inspect, inspection_units_to_inspect



def select_cluster_indexes(config, consignment, n_units_to_inspect):
    """Select units (indexes) from consignment based on sample size and
    cluster selection strategy.

    :param config: Configuration to be used
    :param consignment: Consignment to be inspected
    :param n_units_to_inspect: Number of units to inspect defined in sample functions.
    """
    unit = config["inspection"]["unit"]
    cluster_selection = config["inspection"]["cluster"]["cluster_selection"]

    if unit in ["sample_unit", "sample_units", "item", "items"]:
        if cluster_selection == "random":
            n_inspection_units_to_inspect = (
                compute_n_clusters_to_inspect(config, consignment, n_units_to_inspect)
            )[0]
            # Choose inspection_unit indexes randomly
            indexes_to_inspect = random.sample(
                list(range(consignment.num_inspection_units)), n_inspection_units_to_inspect
            )
        elif cluster_selection == "interval":
            interval = config["inspection"]["cluster"]["interval"]
            n_inspection_units_to_inspect = (
                compute_n_clusters_to_inspect(config, consignment, n_units_to_inspect)
            )[0]
            max_inspection_units = max(1, round(consignment.num_inspection_units / interval))
            # Check to see if interval is small enough to achieve n_inspection_units_to_inspect
            # If not, decrease interval.
            if n_inspection_units_to_inspect > max_inspection_units:
                interval = round(consignment.num_inspection_units / n_inspection_units_to_inspect)
            # Create list of indexes incremented by interval size
            indexes_to_inspect = []
            index = 0
            for unused_i in range(n_inspection_units_to_inspect):
                indexes_to_inspect.append(index)
                index += interval
        else:
            raise RuntimeError(f"Unknown cluster selection method: {cluster_selection}")
    elif unit in ["inspection_unit", "inspection_units", "box", "boxes"]:
        raise RuntimeError(
            "Cannot use cluster selection strategy with inspection_unit sampling unit"
        )
    else:
        raise RuntimeError(f"Unknown unit: {unit}")
    indexes_to_inspect.sort()
    return indexes_to_inspect


def select_units_to_inspect(config, consignment, n_units_to_inspect):
    """Select units to inspect based on selection strategy.

    :param config: Configuration to be used
    :param consignment: Consignment to be inspected
    :param n_units_to_inspect: Number of units to inspect
    """
    unit = config["inspection"]["unit"]
    selection_strategy = config["inspection"]["selection_strategy"]
    sample_strategy = config["inspection"]["sample_strategy"]

    if sample_strategy == "rbs":
        if selection_strategy == "random":
            return select_random_indexes_rbs(unit, consignment, n_units_to_inspect)
        elif selection_strategy == "cluster":
            return select_cluster_indexes(config, consignment, n_units_to_inspect)
        elif selection_strategy == "convenience":
            # Convenience sampling - just select the first n units
            return list(range(min(n_units_to_inspect, consignment.num_sample_units if unit in ["sample_unit", "sample_units", "item", "items"] else consignment.num_inspection_units)))
        else:
            raise RuntimeError(f"Unknown selection strategy: {selection_strategy}")
    else:
        if selection_strategy == "random":
            return select_random_indexes(unit, consignment, n_units_to_inspect)
        elif selection_strategy == "cluster":
            return select_cluster_indexes(config, consignment, n_units_to_inspect)
        elif selection_strategy == "convenience":
            # Convenience sampling - just select the first n units
            return list(range(min(n_units_to_inspect, consignment.num_sample_units if unit in ["sample_unit", "sample_units", "item", "items"] else consignment.num_inspection_units)))
        else:
            raise RuntimeError(f"Unknown selection strategy: {selection_strategy}")


def inspect_sample_unit(sample_unit, effectiveness):
    """Tests whether sample_unit is contaminated considering effectiveness"""
    if sample_unit == 0:
        return False
    return random.random() < effectiveness


def inspect(config, consignment, n_units_to_inspect, detailed):
    """Inspect selected units using both end strategies (to detection, to completion)
    Return number of inspection_units opened, sample_units inspected, and contaminated sample_units found for
    each end strategy.

    :param config: Configuration to be used
    :param consignment: Consignment to be inspected
    :param n_units_to_inspect: Number of units to inspect defined by sample functions.
    """
    # Disabling warnings, possible future TODO is splitting this function.
    # pylint: disable=too-many-locals,too-many-statements
    # pylint: disable=too-many-branches,too-many-nested-blocks

    
    unit = config["inspection"]["unit"]
    selection_strategy = config["inspection"]["selection_strategy"]
    sample_strategy = config["inspection"]["sample_strategy"]
    sample_units_per_inspection_unit = consignment.sample_units_per_inspection_unit

    if sample_strategy == "rbs":
        indexes_to_inspect, inspection_units_to_inspect = select_units_to_inspect(
            config, consignment, n_units_to_inspect
        )
    else:
        indexes_to_inspect = select_units_to_inspect(
            config, consignment, n_units_to_inspect
        )

    effectiveness = get_validated_effectiveness(config)

    # Inspect selected inspection_units, count opened inspection_units, inspected sample_units, and contaminated
    # sample_units to detection and completion
    ret = types.SimpleNamespace(
        inspected_sample_unit_indexes=[],
        insepcted_box_indexes=[],
        inspected_box_result=[],
        inspection_units_opened_completion=0,
        inspection_units_opened_detection=0,
        sample_units_inspected_completion=0,
        sample_units_inspected_detection=0,
        plant_units_inspected_completion=0,
        plant_units_inspected_detection=0,
        contaminated_sample_units_completion=0,
        contaminated_sample_units_detection=0,
        number_sample_units_missed=0,
        number_units_missed=0
    )

    if sample_strategy == "rbs":

        """
        TODO: Investigate min guard and impact on oversampling
        """
        if unit in ["sample_unit", "sample_units", "item", "items"]:
            detected = False
            if selection_strategy == "cluster":
                raise RuntimeError(f"Selection strategy = '{selection_strategy}' is not supported for"
                                   f" sampling_strategy = {sample_strategy}")
            else:  # All other sample_unit selection strategies inspected the same way
                # Empty lists to hold opened inspection_units indexes, will be duplicates bc inspection_unit index
                # computed per inspected sample_unit
                inspection_units_opened_completion = []
                inspection_units_opened_detection = []
                # Loop through sample_units in sorted index list (sorted in index functions)
                # Inspection progresses through indexes in ascending order
                for sample_unit_index in indexes_to_inspect:
                    if detailed:
                        ret.inspected_sample_unit_indexes.append(sample_unit_index)
                    ret.sample_units_inspected_completion += 1
                    # Count plant units inspected (all plants in this sample unit)
                    iu_idx = math.floor(sample_unit_index / sample_units_per_inspection_unit)
                    su_local_idx = sample_unit_index % sample_units_per_inspection_unit
                    try:
                        su_obj = consignment.inspection_units[iu_idx].sample_unit_objects[su_local_idx]
                        ret.plant_units_inspected_completion += len(su_obj.plants)
                    except Exception:
                        pass
                    # Compute inspection_unit index number
                    inspection_units_opened_completion.append(
                        math.floor(sample_unit_index / sample_units_per_inspection_unit))
                    if not detected:
                        ret.sample_units_inspected_detection += 1
                        try:
                            su_obj = consignment.inspection_units[iu_idx].sample_unit_objects[su_local_idx]
                            ret.plant_units_inspected_detection += len(su_obj.plants)
                        except Exception:
                            pass
                        # Compute inspection_unit index number
                        inspection_units_opened_detection.append(
                            math.floor(sample_unit_index / sample_units_per_inspection_unit)
                        )
                    # Debug hook to confirm we are inspecting individual sample units
                    if os.environ.get("SLIPPAGE_DEBUG_INSPECTION"):
                        print(f"Inspecting sample unit {sample_unit_index}")
                    if inspect_sample_unit(consignment.sample_units[sample_unit_index], effectiveness):
                        # Count every contaminated sample_unit in sample
                        ret.contaminated_sample_units_completion += 1
                        if not detected:
                            ret.contaminated_sample_units_detection += 1
                            detected = True
                    # Should be only 1 contaminated sample_unit if to detection
                    if detected:
                        assert ret.contaminated_sample_units_detection == 1
                # Number of inspection_units opened is number of unique inspection_units indexes in inspection_units
                # opened lists
                ret.inspection_units_opened_completion = len(set(inspection_units_opened_completion))
                ret.inspection_units_opened_detection = len(set(inspection_units_opened_detection))

                inspection_unit_counter = 0
                for inspect_unit in consignment.inspection_units:
                    contaminant_found =False
                    inspection_unit_contaminated = False
                    sample_unit_counter = 0
                    total_contaminated_sample_units = 0
                    total_contaminated_units = 0
                    for samp_unit in inspect_unit.sample_unit_objects:
                        if sum(samp_unit.plants) > 0:
                            inspection_unit_contaminated = True
                            total_contaminated_sample_units+=1
                            total_contaminated_units+=sum(samp_unit.plants)
                            if sample_unit_counter in inspection_units_to_inspect[inspection_unit_counter]:
                                contaminant_found = True
                        sample_unit_counter += 1
                    if inspection_unit_contaminated and not contaminant_found:
                        ret.number_sample_units_missed += total_contaminated_sample_units
                        ret.number_units_missed += total_contaminated_units
                    inspection_unit_counter += 1

        elif unit in ["inspection_unit", "inspection_units", "box", "boxes"]:
            raise RuntimeError(f"Selection unit = '{unit}' within the inspect method of the"
                               f" inspections.py module is not supported for"
                               f" sampling_strategy = {sample_strategy}")
    else:
        if unit in ["sample_unit", "sample_units", "item", "items"]:
            detected = False
            if selection_strategy == "cluster":
                # Compute num sample_units to inspect per inspection_unit to achieve sample size
                # based on within inspection_unit proportion.
                inspect_per_inspection_unit = (
                    compute_n_clusters_to_inspect(config, consignment, n_units_to_inspect)
                )[1]
                ret.inspection_units_opened_completion = len(indexes_to_inspect)
                sample_units_inspected = 0
                # Loop through selected inspection_unit indexes (random or interval selection)
                for inspection_unit_index in indexes_to_inspect:
                    if not detected:
                        ret.inspection_units_opened_detection += 1
                    # Number of sample_units to inspect is based on either config within inspection_unit
                    # proportion or required number of sample_units to inspect per inspection_unit
                    # to achieve sample size.
                    sample_remainder = n_units_to_inspect - sample_units_inspected
                    # If sample_remainder is less than inspect_per_inspection_unit, set inspect_per_inspection_unit
                    # to sample_remainder to avoid inspecting more sample_units than computed
                    # sample size.
                    if sample_remainder < inspect_per_inspection_unit:
                        inspect_per_inspection_unit = sample_remainder
                    # In each inspection_unit, loop through first n sample_units (n = inspect_per_inspection_unit)
                    for sample_unit_in_inspection_unit_index, sample_unit in enumerate(
                            (consignment.inspection_units[inspection_unit_index]).sample_units[
                                0:inspect_per_inspection_unit]
                    ):
                        if detailed:
                            sample_unit_index = consignment.sample_unit_in_inspection_unit_to_sample_unit_index(
                                inspection_unit_index, sample_unit_in_inspection_unit_index
                            )
                            ret.inspected_sample_unit_indexes.append(sample_unit_index)
                        ret.sample_units_inspected_completion += 1
                        if not detected:
                            ret.sample_units_inspected_detection += 1
                        if inspect_sample_unit(sample_unit, effectiveness):
                            # Count all contaminated sample_units in sample, regardless of
                            # detected variable
                            ret.contaminated_sample_units_completion += 1
                            if not detected:
                                # Count contaminated sample_units in inspection_unit if not yet detected
                                ret.contaminated_sample_units_detection += 1
                    if ret.contaminated_sample_units_detection > 0:
                        # Update detected variable if contaminated sample_units found in inspection_unit
                        detected = True
                    sample_units_inspected += inspect_per_inspection_unit
                # assert (
                #     ret.sample_units_inspected_completion == n_units_to_inspect
                # ), """Check if number of sample_units is evenly divisible by sample_units per inspection_unit.
                # Partial inspection_units not supported when using cluster selection."""
            else:  # All other sample_unit selection strategies inspected the same way
                # Empty lists to hold opened inspection_units indexes, will be duplicates bc inspection_unit index
                # computed per inspected sample_unit
                inspection_units_opened_completion = []
                inspection_units_opened_detection = []
                # Loop through sample_units in sorted index list (sorted in index functions)
                # Inspection progresses through indexes in ascending order
                for sample_unit_index in indexes_to_inspect:
                    if detailed:
                        ret.inspected_sample_unit_indexes.append(sample_unit_index)
                    ret.sample_units_inspected_completion += 1
                    # Compute inspection_unit index number
                    inspection_units_opened_completion.append(
                        math.floor(sample_unit_index / sample_units_per_inspection_unit))
                    if not detected:
                        ret.sample_units_inspected_detection += 1
                        # Compute inspection_unit index number
                        inspection_units_opened_detection.append(
                            math.floor(sample_unit_index / sample_units_per_inspection_unit)
                        )
                    if inspect_sample_unit(consignment.sample_units[sample_unit_index], effectiveness):
                        # Count every contaminated sample_unit in sample
                        ret.contaminated_sample_units_completion += 1
                        if not detected:
                            ret.contaminated_sample_units_detection += 1
                            detected = True
                    # Should be only 1 contaminated sample_unit if to detection
                    if detected:
                        assert ret.contaminated_sample_units_detection == 1
                # Number of inspection_units opened is number of unique inspection_units indexes in inspection_units
                # opened lists
                ret.inspection_units_opened_completion = len(set(inspection_units_opened_completion))
                ret.inspection_units_opened_detection = len(set(inspection_units_opened_detection))
        elif unit in ["inspection_unit", "inspection_units", "box", "boxes"]:
            ret.inspected_box_indexes = indexes_to_inspect
            # Partial inspection_unit inspections allowed to reduce number of sample_units inspected if desired
            # Handle backward compatibility for within inspection unit proportion
            within_inspection_unit_proportion = config["inspection"].get("within_inspection_unit_proportion",
                                                                         config["inspection"].get(
                                                                             "within_box_proportion", 1.0))
            inspect_per_inspection_unit = int(
                math.ceil(within_inspection_unit_proportion * sample_units_per_inspection_unit))
            detected = False
            ret.inspection_units_opened_completion = n_units_to_inspect
            ret.sample_units_inspected_completion = n_units_to_inspect * inspect_per_inspection_unit
            for inspection_unit_index in indexes_to_inspect:
                if not detected:
                    ret.inspection_units_opened_detection += 1
                # In each inspection_unit, loop through first n sample_units (n = inspect_per_inspection_unit)
                for sample_unit_in_inspection_unit_index, sample_unit in enumerate(
                        (consignment.inspection_units[inspection_unit_index]).sample_units[
                            0:inspect_per_inspection_unit]
                ):
                    if detailed:
                        sample_unit_index = consignment.sample_unit_in_inspection_unit_to_sample_unit_index(
                            inspection_unit_index, sample_unit_in_inspection_unit_index
                        )
                        ret.inspected_sample_unit_indexes.append(sample_unit_index)
                    if not detected:
                        ret.sample_units_inspected_detection += 1
                    if inspect_sample_unit(sample_unit, effectiveness):
                        # Count every contaminated sample_unit in sample
                        ret.contaminated_sample_units_completion += 1
                        # If first contaminated inspection_unit inspected,
                        # count contaminated sample_units in inspection_unit
                        if not detected:
                            ret.contaminated_sample_units_detection += 1
                # If inspection_unit contained contaminated sample_units, changed detected variable
                if ret.contaminated_sample_units_detection > 0:
                    detected = True
                    ret.inspected_box_result.append(1)
                else:
                    ret.inspected_box_result.append(0)

    ret.consignment_checked_ok = ret.contaminated_sample_units_completion == 0
    return ret


def get_sample_function(config, compliance_table=None):
    """Based on config, return function to sample a consignment."""
    sample_strategy = config["inspection"]["sample_strategy"]
    if sample_strategy == "proportion":

        def sample(consignment):
            return sample_proportion(config=config, consignment=consignment)

    elif sample_strategy == "hypergeometric":

        def sample(consignment):
            return sample_hypergeometric(config=config, consignment=consignment)

    elif sample_strategy == "fixed_n":

        def sample(consignment):
            return sample_n(config=config, consignment=consignment)

    elif sample_strategy == "all":

        def sample(consignment):
            return sample_all(config=config, consignment=consignment)
        
    elif sample_strategy == "rbs":

        def sample(consignment):
            return sample_rbs(config=config, consignment=consignment, 
                              compliance_table_dict=compliance_table)

    else:
        raise RuntimeError(f"Unknown sample strategy: {sample_strategy}")
    return sample


def is_consignment_contaminated(consignment):
    """Return True if at least one inspection_unit contains contaminants"""
    for inspection_unit in consignment.inspection_units:
        if inspection_unit:
            return True
    return False


def consignment_contamination_rate(consignment):
    """Get (true) contamination rate of a consignment

    Contamination rate is here defined as number of
    contaminated sample_units divided by the number sample_units.
    """
    count = np.count_nonzero(consignment.sample_units)
    return count / consignment.num_sample_units


def get_detection_and_confidence(consignment,
                                 compliance_table_dict,
                                 default_detection=0.01,
                                 default_confidence=0.8
    ):
    """
    Fetch detection and confidence levels for specified rbs variables.
    If not found, defaults to low compliance values.
    Returns a tuple: (detection_level, confidence_level)
    """
    rbs_variables = compliance_table_dict['rbs_variables']
    n_units_to_inspect = {}
    #key = (origin_country, pm_type)
    if len(rbs_variables) == 0:
        # If no variables detected in the compliance table, default to low compliance
        # print(f"\nWARNING: No compliance variables found in submitted compliance table. "
        #       f"Using low compliance defaults for ALL inspection units:")
        # print(f"      Default Detection Level: {default_detection}")
        # print(f"      Default Confidence Level: {default_confidence}")
        for inspection_unit in range(len(consignment.inspection_units)):
            n_units_to_inspect[inspection_unit] = (default_detection, default_confidence)
    else:
        inspection_unit_idx = 0
        for inspection_unit in consignment.inspection_units:
            values = {attr: getattr(inspection_unit, attr, None) for attr in rbs_variables}
            if any(v is None for v in values.values()):
                # If not all variable specified in compliance table not detected in consignment, then default to low compliance
                none_attrs = [k for k, v in values.items() if v is None]
                # print(
                #     f"\nWARNING: Some compliance tables variables not found as attributes of the consignment."
                #     f" Namely, {none_attrs}."
                #     f" Using low compliance defaults:")
                # print(f"      Default Detection Level: {default_detection}")
                # print(f"      Default Confidence Level: {default_confidence}")
                result = (default_detection, default_confidence)
                #return n_units_to_inspect
            else:
                # If variables found in consignment, attempt to look up in table
                key = tuple(values[attr] for attr in rbs_variables)
                result = compliance_table_dict.get(key)
            if result is not None:
                # If a reference found, then return the associated detection and confidence levels
                n_units_to_inspect[inspection_unit_idx] = result
                inspection_unit_idx+=1
            else:
                # If no reference found, print warning and use low compliance defaults.
                # print(
                #     f"\nWARNING: The variables {key} for inspection unit {inspection_unit_idx} are not found in compliance table. Using low compliance defaults:")
                # print(f"      Default Detection Level: {default_detection}")
                # print(f"      Default Confidence Level: {default_confidence}")
                n_units_to_inspect[inspection_unit_idx] = (default_detection, default_confidence)
                inspection_unit_idx+=1
    return n_units_to_inspect


def count_contaminated_inspection_units(consignment):
    """Return number of inspection_units containing contaminants"""
    count = 0
    for inspection_unit in consignment.inspection_units:
        if inspection_unit:
            count += 1
    return count


def count_contaminated_sample_units(consignment):
    """Return number of contaminated sample_units"""
    count = np.count_nonzero(consignment.sample_units)
    return count


# Backward compatibility aliases
def count_contaminated_boxes(consignment):
    """Return number of boxes containing contaminants (backward compatibility)"""
    return count_contaminated_inspection_units(consignment)


def count_contaminated_items(consignment):
    """Return number of contaminated items (backward compatibility)"""
    return count_contaminated_sample_units(consignment)

def _norm(s: str) -> str:
    """Normalize for matching: lowercase, strip non-alphanum."""
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def normalize_rbs_variables_against_consignment(
    rbs_variables,
    consignment,
):
    """
    Map free-form field names in rbs_variables to actual attributes on a Consignment
    instance, using case-insensitive aliasing. Keeps unmapped items unchanged.

    Returns:
        updated_vars: list[str]  # rbs_variables with matched items replaced by canonical attrs
        mapping: dict[str, str]  # original string -> canonical attribute
        unmapped: list[str]      # originals that didn't match anything
    """

    # 1) Canonical attribute keys from the instance (thanks to UserDict)
    canonical_attrs = set(consignment.keys())

    # 2) Auto-generate basic aliases from the canonical names
    #    (e.g., "material_type" -> "material type", "Material Type", etc.)
    auto_aliases = defaultdict(set)
    for attr in canonical_attrs:
        spaced = attr.replace('_', ' ')
        auto_aliases[attr].update({
            attr,
            spaced,
            spaced.title(),          # "Material Type"
            attr.title(),            # "Material_Type" (rare, but harmless)
        })

    # Add any pre-specified aliases (see reference.py file in slippage_model_utils)
    if 'origin' in canonical_attrs:
        auto_aliases['origin'].update(country_of_origin_names)
    if 'material_type' in canonical_attrs:
        auto_aliases['material_type'].update(pm_type_names)

    # Build a lookup: normalized alias -> canonical attribute
    alias_index = {}
    for attr, names in auto_aliases.items():
        for name in names:
            alias_index[_norm(name)] = attr

    # Walk the input list, map to canonical attributes when possible
    updated_vars = []
    mapping = {}
    unmapped = []

    for original in rbs_variables:
        key = _norm(original)
        if key in alias_index:
            canonical = alias_index[key]
            mapping[original] = canonical
            updated_vars.append(canonical)
        else:
            # Heuristic fallback: try to match substrings like "origin" inside long names
            # (Keeps false positives low by requiring the canonical word to appear)
            matched = None
            for attr in canonical_attrs:
                if re.search(rf'\b{re.escape(attr.replace("_", " "))}\b', original, flags=re.I):
                    matched = attr
                    break
            if matched:
                mapping[original] = matched
                updated_vars.append(matched)
            else:
                unmapped.append(original)
                updated_vars.append(original)

    return updated_vars, mapping, unmapped
