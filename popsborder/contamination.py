# Simulation of contaminated consignments and their inspections
# Copyright (C) 2018-2022 Vaclav Petras and others (see below)

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


"""Contaminant addition to consignments

.. codeauthor:: Vaclav Petras <wenzeslaus gmail com>
.. codeauthor:: Kellyn P. Montgomery <kellynmontgomery gmail com>
"""

import math
import random
import copy
import math
import random
from collections.abc import Mapping
from datetime import datetime

import numpy as np
from scipy import stats

from .inputs import update_nested_dict_by_dict


# This function is not used or working, consider updating or removing.
def add_contaminant_to_random_inspection_unit(config, consignment, contamination_rate=None):
    """Add contaminant to consignment

    Assuming a list of inspection_units with the non-contaminated inspection_units set to False.

    Each sample_unit (inspection_unit) in inspection_units (list) is set to True if a contaminant is
    there, False otherwise.

    :param config: ``random_inspection_unit`` config dictionary
    :param consignment: Consignment to contaminate
    :param contamination_rate: ``contamination_rate`` config dictionary
    """
    contaminant_probability = config["probability"]
    contaminant_ratio = config["ratio"]
    if random.random() >= contaminant_probability:
        return
    for inspection_unit in consignment.inspection_units:
        if random.random() < contaminant_ratio:
            in_inspection_unit = config.get("in_inspection_unit_arrangement", "all")
            if in_inspection_unit == "first":
                # simply put one contaminant to first sample_unit in the inspection_unit
                inspection_unit.sample_units[0] = 1
                # Plant contamination
                if hasattr(inspection_unit, 'sampleunit'):
                    inspection_unit.sampleunit[0].plants.fill(1)
                    inspection_unit.sample_units[0] = inspection_unit.sampleunit[0].plants.sum()
            elif in_inspection_unit == "all":
                inspection_unit.sample_units.fill(1)
                # Plant contamination
                if hasattr(inspection_unit, 'sampleunit'):
                    for samp_index in range(inspection_unit.num_sample_units):
                        if hasattr(inspection_unit.sampleunit[samp_index], 'plants'):
                            inspection_unit.sampleunit[samp_index].plants.fill(1)
                            inspection_unit.sample_units[samp_index] = inspection_unit.sampleunit[samp_index].plants.sum()
            elif in_inspection_unit == "one_random":
                index = np.random.choice(inspection_unit.num_sample_units - 1)
                inspection_unit.sample_units[index] = 1
                # Plant contamination
                if hasattr(inspection_unit, 'sampleunit'):
                    inspection_unit.sampleunit[index].plants.fill(1)
                    inspection_unit.sample_units[index] = inspection_unit.sampleunit[index].plants.sum()
            elif in_inspection_unit == "random":
                if not contamination_rate:
                    raise ValueError(
                        "contamination_rate must be set if arrangement is random"
                    )
                num_contaminated_sample_units = num_sample_units_to_contaminate(
                    contamination_rate, inspection_unit.num_sample_units
                )
                if num_contaminated_sample_units == 0:
                    continue
                indexes = np.random.choice(
                    inspection_unit.num_sample_units, num_contaminated_sample_units, replace=False
                )
                np.put(inspection_unit.sample_units, indexes, 1)
                # Plant contamination
                if hasattr(inspection_unit, 'sampleunit'):
                    for idx in indexes:
                        if hasattr(inspection_unit.sampleunit[idx], 'plants'):
                            inspection_unit.sampleunit[idx].plants.fill(1)
                            inspection_unit.sample_units[idx] = inspection_unit.sampleunit[idx].plants.sum()

# modify to include clarke parameters -
def get_contamination_rate(config):
    """Get contamination rate.

    Config is the ``contamination_rate`` dictionary.
    """
    distribution = config["distribution"]
    if distribution == "fixed_value":
        return config["value"]
    if distribution == "beta":
        parameters = config["parameters"]
        if isinstance(parameters, Mapping):
            param1 = parameters["a"]
            param2 = parameters["b"]
        else:
            param1, param2 = parameters
        return float(stats.beta.rvs(param1, param2, size=1))
    raise RuntimeError(f"Unknown contamination rate distribution: {distribution}")


def num_sample_units_to_contaminate(config, num_sample_units):
    """Return number of sample_units to be contaminated
    Rounds up or down to nearest integer.

    Config is the ``contamination_rate`` dictionary.
    """
    contamination_rate = get_contamination_rate(config)
    contaminated_sample_units = round(num_sample_units * contamination_rate)
    return contaminated_sample_units


def num_inspection_units_to_contaminate(config, num_inspection_units):
    """Return number of inspection_units to be contaminated as float.

    Config is the ``contamination_rate`` dictionary.
    """
    contamination_rate = get_contamination_rate(config)
    contaminated_inspection_units = num_inspection_units * contamination_rate
    return contaminated_inspection_units


def add_contaminant_uniform_random(config, consignment):
    """Add contaminants to consignment using uniform random distribution

    Contamination rate is determined using the ``contamination_rate`` config key.
    """
    contamination_unit = config["contamination_unit"]
    
    # Handle backward compatibility for old terminology
    if contamination_unit in ["box", "boxes"]:
        contamination_unit = "inspection_unit"
    elif contamination_unit in ["item", "items"]:
        contamination_unit = "sample_unit"

    if contamination_unit in ["inspection_unit", "inspection_units"]:
        contaminated_inspection_units = num_inspection_units_to_contaminate(
            config["contamination_rate"], consignment.num_inspection_units
        )
        if contaminated_inspection_units == 0.0:
            return
        inspection_unit_indexes = np.random.choice(
            consignment.num_inspection_units, math.ceil(contaminated_inspection_units), replace=False
        )
        # Mark contaminated sample_units in all contaminated inspection_units (full and partial)
        for inspection_unit_index in inspection_unit_indexes[:-1]:
            consignment.inspection_units[inspection_unit_index].sample_units.fill(1)
        partial_inspection_unit_proportion = math.modf(contaminated_inspection_units)[0]
        if partial_inspection_unit_proportion == 0.0:
            partial_inspection_unit_proportion = 1
        partial_inspection_unit_contaminated_stems = round(
            consignment.inspection_units[inspection_unit_indexes[-1]].num_sample_units * partial_inspection_unit_proportion
        )
        consignment.inspection_units[inspection_unit_indexes[-1]].sample_units[0:partial_inspection_unit_contaminated_stems].fill(1)

        # Pooled plant-level contamination for all contaminated sample_units in all contaminated inspection_units
        if consignment.num_plants is not None:
            # Gather all (inspection_unit_idx, samp_index) tuples for contaminated sample_units
            contaminated_sample_units = []
            for inspection_unit_index in inspection_unit_indexes[:-1]:
                for samp_index in range(consignment.inspection_units[inspection_unit_index].num_sample_units):
                    contaminated_sample_units.append((inspection_unit_index, samp_index))
            for samp_index in range(partial_inspection_unit_contaminated_stems):
                contaminated_sample_units.append((inspection_unit_indexes[-1], samp_index))
            perc_plants_contaminated = config["clustered"]["percentage_plants_contaminated"]
            _apply_pooled_plant_level_contamination(consignment, contaminated_sample_units, perc_plants_contaminated)
        else:
            # No plant unit exists, so set sample_unit array directly for all contaminated sample_units
            for inspection_unit_index in inspection_unit_indexes[:-1]:
                consignment.inspection_units[inspection_unit_index].sample_units.fill(1)
            for samp_index in range(partial_inspection_unit_contaminated_stems):
                consignment.inspection_units[inspection_unit_indexes[-1]].sample_units[samp_index] = 1

        assert np.count_nonzero(consignment.inspection_units) in (
            math.ceil(contaminated_inspection_units),
            math.floor(contaminated_inspection_units),
        )
    elif contamination_unit in ["sample_unit", "sample_units"]:
        contaminated_sample_units = num_sample_units_to_contaminate(
            config["contamination_rate"], consignment.num_sample_units
        )
        if contaminated_sample_units == 0:
            return
        sample_unit_indexes = np.random.choice(
            consignment.num_sample_units, contaminated_sample_units, replace=False
        )
        if consignment.num_plants is not None:
            perc_plants_contaminated = config["clustered"]["percentage_plants_contaminated"]
            _apply_pooled_plant_level_contamination(consignment, list(sample_unit_indexes), perc_plants_contaminated)
            # For plant-level contamination, the assertion is more flexible since some sample_units
            # may end up with zero contaminated plants due to the probabilistic nature
            actual_contaminated = np.count_nonzero(consignment.sample_units)
            assert actual_contaminated <= contaminated_sample_units, f"Expected at most {contaminated_sample_units} contaminated sample_units, got {actual_contaminated}"
        else:
            # No plant unit exists, so set sample_unit array directly
            np.put(consignment.sample_units, sample_unit_indexes, 1)
            assert np.count_nonzero(consignment.sample_units) == contaminated_sample_units
    elif contamination_unit in ["plant", "plants"]:
        # Contaminate plants directly
        # Assume consignment has inspection_units, each inspection_unit has sampleunit, each sampleunit has plants
        # Flatten all plants into a 1D array for indexing
        all_plants = []
        plant_indices = []  # (inspection_unit_idx, sampleunit_idx, plant_idx)
        for inspection_unit_idx, inspection_unit in enumerate(consignment.inspection_units):
            for sampleunit_idx, sampleunit in enumerate(inspection_unit.sampleunit):
                for plant_idx in range(len(sampleunit.plants)):
                    all_plants.append(sampleunit.plants)
                    plant_indices.append((inspection_unit_idx, sampleunit_idx, plant_idx))
        num_plants = len(plant_indices)
        contaminated_plants = num_sample_units_to_contaminate(config["contamination_rate"], num_plants)
        if contaminated_plants == 0:
            return
        plant_indexes = np.random.choice(num_plants, contaminated_plants, replace=False)
        for idx in plant_indexes:
            inspection_unit_idx, sampleunit_idx, plant_idx = plant_indices[idx]
            consignment.inspection_units[inspection_unit_idx].sampleunit[sampleunit_idx].plants[plant_idx] = 1
        # Update consignment.sample_units to sum of contaminated plants for each sample_unit
        sample_unit_counter = 0
        for inspection_unit in consignment.inspection_units:
            for sampleunit in inspection_unit.sampleunit:
                consignment.sample_units[sample_unit_counter] = sampleunit.plants.sum()
                sample_unit_counter += 1
        # Test correct number contaminated
        total_contaminated = sum(
            (sampleunit.plants == 1).sum() for inspection_unit in consignment.inspection_units for sampleunit in inspection_unit.sampleunit
        )
        assert total_contaminated == contaminated_plants
    else:
        raise RuntimeError(f"Unknown contamination unit: {contamination_unit}")


def _contaminated_sample_units_to_cluster_sizes(
    contaminated_sample_units, contaminated_units_per_cluster
):
    """Get list of cluster sizes for a given number of contaminated sample_units

    The size of each cluster is limited by contaminated_units_per_cluster.
    """
    if contaminated_sample_units > contaminated_units_per_cluster:
        # Split into n clusters so that n-1 clusters have the max size and
        # the last one has the remaining sample_units.
        sum_sample_units = 0
        cluster_sizes = []
        while sum_sample_units < contaminated_sample_units - contaminated_units_per_cluster:
            sum_sample_units += contaminated_units_per_cluster
            cluster_sizes.append(contaminated_units_per_cluster)
        # add remaining sample_units
        cluster_sizes.append(contaminated_sample_units - sum_sample_units)
        sum_sample_units += contaminated_sample_units - sum_sample_units
        assert sum_sample_units == contaminated_sample_units
    else:
        cluster_sizes = [contaminated_sample_units]
    return cluster_sizes


def _contaminated_inspection_units_to_cluster_sizes(contaminated_inspection_units, contaminated_units_per_cluster):
    """Get list of cluster sizes for a given number of contaminated sample_units

    The size of each cluster is limited by contaminated_units_per_cluster.
    """
    contaminated_inspection_units = math.ceil(contaminated_inspection_units)
    if contaminated_inspection_units > contaminated_units_per_cluster:
        # Split into n clusters so that n-1 clusters have the max size and
        # the last one has the remaining sample_units.
        sum_inspection_units = 0
        cluster_sizes = []
        while sum_inspection_units < contaminated_inspection_units - contaminated_units_per_cluster:
            sum_inspection_units += contaminated_units_per_cluster
            cluster_sizes.append(contaminated_units_per_cluster)
        # add last cluster with remaining contaminated inspection_units
        cluster_sizes.append(contaminated_inspection_units - sum_inspection_units)
        sum_inspection_units += contaminated_inspection_units - sum_inspection_units
        assert sum_inspection_units == contaminated_inspection_units
    else:
        cluster_sizes = [math.ceil(contaminated_inspection_units)]
    return cluster_sizes


def choose_strata_for_clusters(num_units, cluster_width, num_clusters):
    """Divide array of sample_units or inspection_units into strata wide enough for clusters
    so that they do not overlap. If array is not equally divisible by cluster_width,
    create one smaller stratum that can be used for a smaller cluster if needed.
    This is important for very high contamination rates that require nearly all units
    to be contaminated.
    Randomly select strata to place contaminant clusters. If contamination rate is
    low enough that not all strata are needed, omit smaller strata created from
    remainder and only select from strata wide enough to contain full sized cluster.
    Return strata selected to contaminate with clusters.

    num_units: number of inspection_units or sample_units in consignment
    cluster_width: size of cluster in terms of inspection_units or units
    num_clusters: number of clusters to contaminate
    """
    # Round up so that one smaller remainder stratum is included
    num_strata = max(1, math.ceil(num_units / cluster_width))
    # Make sure there are enough strata for the number of clusters needed.
    if num_strata < num_clusters:
        raise ValueError(
            """Cannot avoid overlapping clusters. Increase
            contaminated_units_per_cluster
            or decrease cluster_sample_unit_width (if using sample_unit contamination_unit)"""
        )
    # If all strata are needed, all strata are selected for clusters
    if num_clusters == num_strata:
        cluster_strata = np.arange(num_strata)
    # If not all strata needed (num of clusters is less than num of strata), do not use
    # last strata if its smaller than cluster_width (remainder from rounding up)
    else:
        # if no remainder (all strata are equal length), select any strata for clusters
        if num_units % cluster_width == 0:
            cluster_strata = np.random.choice(num_strata, num_clusters, replace=False)
        # if last strata is smaller and not all strata are needed,
        # do not include last strata as option for placing clusters
        else:
            cluster_strata = np.random.choice(
                num_strata - 1, num_clusters, replace=False
            )
    return cluster_strata


def add_contaminant_clusters_to_inspection_units(config, consignment):
    """Add contaminant clusters to inspection_units in a consignment"""
    contaminated_units_per_cluster = config["clustered"]["contaminated_units_per_cluster"]
    num_inspection_units = consignment.num_inspection_units
    contaminated_inspection_units = num_inspection_units_to_contaminate(
        config["contamination_rate"], num_inspection_units
    )
    if contaminated_inspection_units == 0:
        return
    cluster_sizes = _contaminated_inspection_units_to_cluster_sizes(
        contaminated_inspection_units, contaminated_units_per_cluster
    )
    cluster_strata = choose_strata_for_clusters(
        num_inspection_units, contaminated_units_per_cluster, len(cluster_sizes)
    )
    # Mark contaminated sample_units in all contaminated inspection_units (full and partial)
    for index, cluster_size in enumerate(cluster_sizes[:-1]):
        cluster_start = contaminated_units_per_cluster * cluster_strata[index]
        cluster_indexes = np.arange(
            start=cluster_start, stop=cluster_start + cluster_size
        )
        for cluster_index in cluster_indexes:
            consignment.inspection_units[cluster_index].sample_units.fill(1)
    cluster_start = (
        contaminated_units_per_cluster * cluster_strata[len(cluster_sizes) - 1]
    )
    cluster_indexes = np.arange(
        start=cluster_start, stop=cluster_start + cluster_sizes[-1]
    )
    for cluster_index in cluster_indexes[:-1]:
        consignment.inspection_units[cluster_index].sample_units.fill(1)
    partial_inspection_unit_proportion = math.modf(contaminated_inspection_units)[0]
    if partial_inspection_unit_proportion == 0.0:
        partial_inspection_unit_proportion = 1
    partial_inspection_unit_contaminated_stems = round(
        consignment.inspection_units[cluster_indexes[-1]].num_sample_units * partial_inspection_unit_proportion
    )
    consignment.inspection_units[cluster_indexes[-1]].sample_units[0:partial_inspection_unit_contaminated_stems].fill(1)

    # Pooled plant-level contamination for all contaminated sample_units in all contaminated inspection_units
    if consignment.num_plants is not None:
        perc_plants_contaminated = config["clustered"]["percentage_plants_contaminated"]
        plant_tuples = []  # (inspection_unit_idx, samp_index, plant_idx)
        # Collect all contaminated sample_units in all contaminated inspection_units
        # Full contaminated inspection_units (all except last cluster inspection_unit if partial)
        for index, cluster_size in enumerate(cluster_sizes[:-1]):
            cluster_start = contaminated_units_per_cluster * cluster_strata[index]
            cluster_indexes = np.arange(start=cluster_start, stop=cluster_start + cluster_size)
            for cluster_index in cluster_indexes:
                for samp_index in range(consignment.inspection_units[cluster_index].num_sample_units):
                    sampleunit = consignment.inspection_units[cluster_index].sampleunit[samp_index]
                    num_plants = len(sampleunit.plants)
                    for plant_idx in range(num_plants):
                        plant_tuples.append((cluster_index, samp_index, plant_idx))
        # Partial inspection_unit (last cluster inspection_unit, possibly partial)
        cluster_start = contaminated_units_per_cluster * cluster_strata[len(cluster_sizes) - 1]
        cluster_indexes = np.arange(start=cluster_start, stop=cluster_start + cluster_sizes[-1])
        for cluster_index in cluster_indexes[:-1]:
            for samp_index in range(consignment.inspection_units[cluster_index].num_sample_units):
                sampleunit = consignment.inspection_units[cluster_index].sampleunit[samp_index]
                num_plants = len(sampleunit.plants)
                for plant_idx in range(num_plants):
                    plant_tuples.append((cluster_index, samp_index, plant_idx))
        # Partial inspection_unit: only the contaminated sample_units
        partial_inspection_unit_proportion = math.modf(contaminated_inspection_units)[0]
        if partial_inspection_unit_proportion == 0.0:
            partial_inspection_unit_proportion = 1
        partial_inspection_unit_contaminated_stems = round(
            consignment.inspection_units[cluster_indexes[-1]].num_sample_units * partial_inspection_unit_proportion
        )
        for samp_index in range(partial_inspection_unit_contaminated_stems):
            sampleunit = consignment.inspection_units[cluster_indexes[-1]].sampleunit[samp_index]
            num_plants = len(sampleunit.plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((cluster_indexes[-1], samp_index, plant_idx))
        total_plants = len(plant_tuples)
        num_contaminated_plants = max(1, round(total_plants * perc_plants_contaminated))
        # Set all plants in all contaminated sample_units to 0 first
        for inspection_unit_idx, samp_index, plant_idx in plant_tuples:
            consignment.inspection_units[inspection_unit_idx].sampleunit[samp_index].plants[plant_idx] = 0
        # Randomly contaminate the required number of plants across all pooled plants
        contaminated_plant_indices = np.random.choice(total_plants, num_contaminated_plants, replace=False)
        for idx in contaminated_plant_indices:
            inspection_unit_idx, samp_index, plant_idx = plant_tuples[idx]
            consignment.inspection_units[inspection_unit_idx].sampleunit[samp_index].plants[plant_idx] = 1
        # Update sample_units array to sum of contaminated plants per sample_unit
        for inspection_unit_idx, inspection_unit in enumerate(consignment.inspection_units):
            for samp_index, sampleunit in enumerate(inspection_unit.sampleunit):
                consignment.inspection_units[inspection_unit_idx].sample_units[samp_index] = sampleunit.plants.sum()

    # Check if correct number of inspection_units contaminated, should be rounded up
    # contaminated_inspection_units, or may be rounded down contaminated_inspection_units
    # if no stems were contaminated in last partial inspection_unit
    
    assert np.count_nonzero(consignment.inspection_units) in (
        math.ceil(contaminated_inspection_units),
        math.ceil(contaminated_inspection_units) - 1,
    )


def add_contaminant_clusters_to_sample_units_with_subset_clustering(config, consignment):
    """Add contaminant cluster to sample_units in a consignment using a single parameter

    Clustering equal to 0 means all sample_units in the consignment can be contaminated with
    equal probability, i.e., the cluster spreads over the whole consignment. Clustering
    equal to 1 means that all sample_units in the cluster are contaminated. The size of the
    cluster is then directly determined by the contamination rate.
    If the cluster would spread over the end of the consignment, we put the extra part
    of the cluster at the beginning of the consignment.
    """
    clustering = config["clustered"]["value"]
    num_of_contaminated_sample_units = num_sample_units_to_contaminate(
        config["contamination_rate"], consignment.num_sample_units
    )
    if num_of_contaminated_sample_units == 0:
        return
    subset_size = round(consignment.num_sample_units * (1 - clustering))
    subset_size = max(subset_size, num_of_contaminated_sample_units)
    start_index2 = None
    end_index2 = None
    if subset_size == consignment.num_sample_units:
        start_index = 0
        end_index = consignment.num_sample_units
    else:
        start_index = np.random.randint(0, consignment.num_sample_units)
        if start_index + subset_size > consignment.num_sample_units:
            start_index2 = 0
            end_index2 = subset_size - (consignment.num_sample_units - start_index)
            end_index = consignment.num_sample_units
            assert (end_index - start_index) + (
                end_index2 - start_index2
            ) == subset_size
        else:
            end_index = start_index + subset_size
            assert end_index - start_index == subset_size

    potential_indexes = np.arange(start_index, end_index)
    if start_index2 is not None and end_index2 is not None:
        potential_indexes = np.concatenate(
            (potential_indexes, np.arange(start_index2, end_index2))
        )
    assert len(potential_indexes) == subset_size
    indexes = np.random.choice(
        potential_indexes,
        num_of_contaminated_sample_units,
        replace=False,
    )
    consignment.sample_units[indexes] = 1
    # Contaminate all plants in the sample unit (sample_unit) for every contaminated sample_unit
    if consignment.num_plants is not None:
        # Gather all plant indices (as tuples: (sample_unit_index, inspection_unit_idx, sampleunit_idx, plant_idx)) for selected sample_units
        plant_tuples = []
        for sample_unit_index in indexes:
            inspection_unit_idx, sampleunit_idx = consignment.get_inspection_unit_and_sampleunit_index(sample_unit_index)
            num_plants = len(consignment.inspection_units[inspection_unit_idx].sampleunit[sampleunit_idx].plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((sample_unit_index, inspection_unit_idx, sampleunit_idx, plant_idx))
        total_plants = len(plant_tuples)
        perc_plants_contaminated = config["clustered"]["percentage_plants_contaminated"]
        num_contaminated_plants = max(1, round(total_plants * perc_plants_contaminated))
        contaminated_plant_indices = np.random.choice(total_plants, num_contaminated_plants, replace=False)
        # Set all plants in selected sample_units to 0 first
        for sample_unit_index in indexes:
            inspection_unit_idx, sampleunit_idx = consignment.get_inspection_unit_and_sampleunit_index(sample_unit_index)
            consignment.inspection_units[inspection_unit_idx].sampleunit[sampleunit_idx].plants.fill(0)
        # Set contaminated plants
        for idx in contaminated_plant_indices:
            sample_unit_index, inspection_unit_idx, sampleunit_idx, plant_idx = plant_tuples[idx]
            consignment.inspection_units[inspection_unit_idx].sampleunit[sampleunit_idx].plants[plant_idx] = 1
        # Update sample_units array for each sample_unit
        for sample_unit_index in indexes:
            inspection_unit_idx, sampleunit_idx = consignment.get_inspection_unit_and_sampleunit_index(sample_unit_index)
            consignment.inspection_units[inspection_unit_idx].sample_units[sampleunit_idx] = consignment.inspection_units[inspection_unit_idx].sampleunit[sampleunit_idx].plants.sum()

    assert np.count_nonzero(consignment.sample_units) == num_of_contaminated_sample_units


def add_contaminant_clusters_to_sample_units(config, consignment):
    """Add contaminant clusters to sample_units in a consignment"""
    contaminated_units_per_cluster = config["clustered"]["contaminated_units_per_cluster"]
    num_sample_units = consignment.num_sample_units
    contaminated_sample_units = num_sample_units_to_contaminate(
        config["contamination_rate"], num_sample_units
    )
    if contaminated_sample_units == 0:
        return
    cluster_sizes = _contaminated_sample_units_to_cluster_sizes(
        contaminated_sample_units, contaminated_units_per_cluster
    )
    cluster_indexes = []
    distribution = config["clustered"]["distribution"]
    if distribution == "random":
        # Handle backward compatibility for cluster width terminology
        cluster_sample_unit_width = config["clustered"]["random"].get("cluster_sample_unit_width", 
                                                                       config["clustered"]["random"].get("cluster_item_width", 1))
        if cluster_sample_unit_width < contaminated_units_per_cluster:
            raise ValueError(
                f"Maximum cluster width, currently {cluster_sample_unit_width}, needs"
                " to be at least as large as contaminated_units_per_cluster"
                " (currently {contaminated_units_per_cluster})"
            )
        # cluster can't be wider/longer than the current list of sample_units
        cluster_sample_unit_width = min(cluster_sample_unit_width, num_sample_units)
        cluster_strata = choose_strata_for_clusters(
            num_sample_units, cluster_sample_unit_width, len(cluster_sizes)
        )
        for index, cluster_size in enumerate(cluster_sizes):
            cluster_start = cluster_sample_unit_width * cluster_strata[index]
            # Use smaller cluster width if placing sample_units in smaller remainder stratum
            cluster_width = min(
                cluster_sample_unit_width, (consignment.num_sample_units - cluster_start)
            )
            assert (
                cluster_width >= cluster_size
            ), "Not enough sample_units available to contaminate in selected cluster stratum."
            cluster = np.random.choice(cluster_width, cluster_size, replace=False)
            cluster += cluster_start
            cluster_indexes.extend(list(cluster))
    elif distribution == "continuous":
        cluster_strata = choose_strata_for_clusters(
            num_sample_units, contaminated_units_per_cluster, len(cluster_sizes)
        )
        for index, cluster_size in enumerate(cluster_sizes):
            cluster = np.arange(cluster_size)
            cluster_start = contaminated_units_per_cluster * cluster_strata[index]
            cluster += cluster_start
            cluster_indexes.extend(list(cluster))
    else:
        raise RuntimeError(f"Unknown cluster distribution: {distribution}")
    cluster_indexes = np.array(cluster_indexes, dtype=np.int64)
    assert np.min(cluster_indexes) >= 0, "Cluster values need to be valid indices"
    assert np.max(cluster_indexes) < num_sample_units
    np.put(consignment.sample_units, cluster_indexes, 1)
    if consignment.num_plants is not None:
        # Gather all plant indices (as tuples: (sample_unit_index, inspection_unit_idx, sampleunit_idx, plant_idx)) for selected sample_units
        plant_tuples = []
        for sample_unit_index in cluster_indexes:
            inspection_unit_idx, sampleunit_idx = consignment.get_inspection_unit_and_sampleunit_index(sample_unit_index)
            num_plants = len(consignment.inspection_units[inspection_unit_idx].sampleunit[sampleunit_idx].plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((sample_unit_index, inspection_unit_idx, sampleunit_idx, plant_idx))
        total_plants = len(plant_tuples)
        perc_plants_contaminated = config["clustered"]["percentage_plants_contaminated"]
        num_contaminated_plants = max(1, round(total_plants * perc_plants_contaminated))
        contaminated_plant_indices = np.random.choice(total_plants, num_contaminated_plants, replace=False)
        # Set all plants in selected sample_units to 0 first
        for sample_unit_index in cluster_indexes:
            inspection_unit_idx, sampleunit_idx = consignment.get_inspection_unit_and_sampleunit_index(sample_unit_index)
            consignment.inspection_units[inspection_unit_idx].sampleunit[sampleunit_idx].plants.fill(0)
        # Set contaminated plants
        for idx in contaminated_plant_indices:
            sample_unit_index, inspection_unit_idx, sampleunit_idx, plant_idx = plant_tuples[idx]
            consignment.inspection_units[inspection_unit_idx].sampleunit[sampleunit_idx].plants[plant_idx] = 1
        # Update sample_units array for each sample_unit
        for sample_unit_index in cluster_indexes:
            inspection_unit_idx, sampleunit_idx = consignment.get_inspection_unit_and_sampleunit_index(sample_unit_index)
            consignment.inspection_units[inspection_unit_idx].sample_units[sampleunit_idx] = consignment.inspection_units[inspection_unit_idx].sampleunit[sampleunit_idx].plants.sum()
        # For plant-level contamination, the assertion is more flexible since some sample_units
        # may end up with zero contaminated plants due to the probabilistic nature
        actual_contaminated = np.count_nonzero(consignment.sample_units)
        assert actual_contaminated <= contaminated_sample_units, f"Expected at most {contaminated_sample_units} contaminated sample_units, got {actual_contaminated}"
    else:
        assert np.count_nonzero(consignment.sample_units) == contaminated_sample_units


def add_contaminant_clusters(config, consignment):
    """Add contaminant clusters to consignment

    Sample_unit (separately or in inspection_units) with contaminant in *consignment* evaluate
    to True after running this function.
    This function does not touch the not sample_units not selected for contamination.
    However, they are expected to be zero.
    """
    contamination_unit = config["contamination_unit"]
    if contamination_unit in ["inspection_unit", "inspection_units", "box", "boxes"]:
        if config["clustered"]["distribution"] == "single":
            raise RuntimeError(
                "clustering distribution 'single' is not supported for inspection_units"
            )
        add_contaminant_clusters_to_inspection_units(config, consignment)
    elif contamination_unit in ["sample_unit", "sample_units", "item", "items"]:
        if config["clustered"]["distribution"] == "single":
            add_contaminant_clusters_to_sample_units_with_subset_clustering(
                config, consignment
            )
        else:
            add_contaminant_clusters_to_sample_units(config, consignment)
    elif contamination_unit in ["plant", "plants"]:
        if config["clustered"]["distribution"] == "single":
            raise RuntimeError(
                "clustering distribution 'single' is not supported for plants"
            )
        else:
            raise RuntimeError(
                "clustering distribution for plants is not yet implemented"
            )
    else:
        raise RuntimeError(f"Unknown contamination unit: {contamination_unit}")


def _apply_pooled_plant_level_contamination(consignment, sample_unit_indexes, percentage):
    """
    Helper to apply pooled plant-level contamination across a set of sample_units.
    sample_unit_indexes: list of sample_unit indices (for sample_unit-level) or (inspection_unit_idx, samp_index) tuples (for inspection_unit-level)
    percentage: float, fraction of plants to contaminate
    """
    if not sample_unit_indexes:
        return
    # Ensure sample_unit_indexes is a list
    sample_unit_indexes = list(sample_unit_indexes)
    # Robustly check if sample_unit_indexes are tuples (inspection_unit-level) or ints (sample_unit-level)
    first_elem = sample_unit_indexes[0]
    is_tuple = isinstance(first_elem, tuple)
    plant_tuples = []
    if is_tuple:
        # inspection_unit-level: sample_unit_indexes are (inspection_unit_idx, samp_index) tuples
        for inspection_unit_idx, samp_index in sample_unit_indexes:
            sampleunit = consignment.inspection_units[inspection_unit_idx].sampleunit[samp_index]
            num_plants = len(sampleunit.plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((None, inspection_unit_idx, samp_index, plant_idx))
    else:
        # sample_unit-level: sample_unit_indexes are sample_unit indices (int or numpy int)
        for sample_unit_index in sample_unit_indexes:
            inspection_unit_idx, sampleunit_idx = consignment.get_inspection_unit_and_sampleunit_index(int(sample_unit_index))
            sampleunit = consignment.inspection_units[inspection_unit_idx].sampleunit[sampleunit_idx]
            num_plants = len(sampleunit.plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((sample_unit_index, inspection_unit_idx, sampleunit_idx, plant_idx))
    total_plants = len(plant_tuples)
    if total_plants == 0:
        return
    num_contaminated_plants = max(1, round(total_plants * percentage))
    # Set all plants in all contaminated sample_units to 0 first
    for _, inspection_unit_idx, sampleunit_idx, plant_idx in plant_tuples:
        consignment.inspection_units[inspection_unit_idx].sampleunit[sampleunit_idx].plants[plant_idx] = 0
    contaminated_plant_indices = np.random.choice(total_plants, num_contaminated_plants, replace=False)
    for idx in contaminated_plant_indices:
        _, inspection_unit_idx, sampleunit_idx, plant_idx = plant_tuples[idx]
        consignment.inspection_units[inspection_unit_idx].sampleunit[sampleunit_idx].plants[plant_idx] = 1
    # Update sample_units array to sum of contaminated plants per sample_unit
    if not is_tuple:
        # sample_unit-level
        for sample_unit_index in sample_unit_indexes:
            inspection_unit_idx, sampleunit_idx = consignment.get_inspection_unit_and_sampleunit_index(int(sample_unit_index))
            consignment.inspection_units[inspection_unit_idx].sample_units[sampleunit_idx] = consignment.inspection_units[inspection_unit_idx].sampleunit[sampleunit_idx].plants.sum()
    else:
        # inspection_unit-level
        for inspection_unit_idx, samp_index in sample_unit_indexes:
            consignment.inspection_units[inspection_unit_idx].sample_units[samp_index] = consignment.inspection_units[inspection_unit_idx].sampleunit[samp_index].plants.sum()


def consignment_matches_selection_rule(rule, consignment):
    """Return True if the *consignment* matches the selection *rule*."""
    # Commodity properties used for selection default to None.
    commodity = rule.get("commodity")
    origin = rule.get("origin")
    port = rule.get("port")
    # All the properties needs to match, but if the property value is not
    # provided in configuration, we count it as match so that consignment
    # can be selected using only one property.
    selected = (
        (not commodity or commodity == consignment.commodity)
        and (not origin or origin == consignment.origin)
        and (not port or port == consignment.port)
    )
    if not selected:
        return False
    start_date = rule.get("start_date")
    end_date = rule.get("end_date")
    # YAML converts to date, but JSON and other load config methods do not.
    if start_date and isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if end_date and isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    if not start_date and not end_date:
        return True
    elif start_date and consignment.date < start_date:
        return False
    elif end_date and consignment.date > end_date:
        return False
    return True


def get_contamination_config_for_consignment(config, consignment):
    """Return configuration for contamination for a given consignment"""
    contamination_config = config["contamination"]
    consignments_config = contamination_config.get("consignments")
    if not consignments_config:
        return contamination_config
    for name, selection_and_config in consignments_config.sample_units():
        selection_rule = selection_and_config["selection"]
        if consignment_matches_selection_rule(selection_rule, consignment):
            # We found matching configuration.
            # Now, we update contamination configuration by this specific one.
            specific_config = selection_and_config["config"]
            # Create a copy we can modify.
            specific_contamination_config = copy.deepcopy(contamination_config)
            update_nested_dict_by_dict(specific_contamination_config, specific_config)
            return specific_contamination_config
    # No matching configuration found, use the original one.
    return contamination_config


def get_contaminant_function(config):
    """Based on config, return function to contaminate a consignment."""
    contamination_config = get_contamination_config_for_consignment(config, consignment=None)
    arrangement = contamination_config["arrangement"]
    if arrangement == "random_inspection_unit":

        def add_contaminant(consignment):
            specific_contamination_config = get_contamination_config_for_consignment(
                config, consignment
            )
            return add_contaminant_to_random_inspection_unit(
                specific_contamination_config["random_inspection_unit"],
                consignment,
                contamination_rate=specific_contamination_config.get("contamination_rate"),
            )

    elif arrangement == "random":

        def add_contaminant(consignment):
            specific_contamination_config = get_contamination_config_for_consignment(
                config, consignment
            )
            return add_contaminant_uniform_random(specific_contamination_config, consignment)

    elif arrangement == "clustered":

        def add_contaminant(consignment):
            specific_contamination_config = get_contamination_config_for_consignment(
                config, consignment
            )
            return add_contaminant_clusters(specific_contamination_config, consignment)

    else:
        raise RuntimeError(f"Unknown contaminant arrangement: {arrangement}")
    return add_contaminant

