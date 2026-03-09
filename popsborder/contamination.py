# Simulation of contaminated consignments and their inspections
# Copyright (C) 2018-2022 Vaclav Petras and others (see below)
# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

"""
Modifications:
- 10/3/2025: Modified following functions (Gary Lin)
    - add_contaminant_uniform_random():
        * Added backward compatibility for contamination_unit parameter
        * Maps old terminology: "box"/"boxes" -> "inspection_unit", "item"/"items" -> "sample_unit"
        * Enhanced to support both inspection_unit and sample_unit level contamination
        * Added plant-level contamination support with pooled contamination methodology

    - add_contaminant_clusters():
        * Updated contamination_unit parameter handling for backward compatibility
        * Supports legacy "box"/"item" terminology while using new "inspection_unit"/"sample_unit" internally
        * Enhanced clustering algorithms for hierarchical contamination patterns

    - add_contaminant_clusters_to_sample_units():
        * Added backward compatibility for cluster_sample_unit_width (formerly cluster_item_width)
        * Updated to handle both old and new terminology in clustering configuration
        * Enhanced plant-level contamination with percentage-based pooled contamination
    - num_items_to_contaminate(): renamed to num_sample_units_to_contaminate()
    - num_boxes_to_contaminate(): renamed to num_inspection_units_to_contaminate()

- 10/28/2025: Added the following support function for new data-driven contamination procedure (Joseph Agor)
    - add_contaminant_beta_binomial_for_groups(): Beta-binomial contamination sampler where each 'group' is an inspection unit
    - add_contaminant_beta_binomial(): Vectorized sampling functon for Beta-Binomial distribution
    - contaminate_units_by_group(): Function to add contaminants to different "groups"
    - get_range_key(): Function that finds the parameters based on what range the quantities fall into
    - _set_beta_binomial_params():  Function that sets the beta-binomial parameters needed based on the main config file.

- 10/28/2025:  Modified the following functions (Joseph Agor)
    - add_contaminant_uniform_random():
        * Added functionality to use the beta-binomial model from Clarke et. al. 2023 paper for plant units
    - get_contaminant_function():
        * Updated to include ability to contaminate using the beta-binomial approach
        * Embedded logic from previously existing create_contaminant_function() into this function
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


"""Contaminant addition to consignments

.. codeauthor:: Vaclav Petras <wenzeslaus gmail com>
.. codeauthor:: Kellyn P. Montgomery <kellynmontgomery gmail com>
.. codeauthor:: Gary Lin <Gary.Lin jhuapl edu>
.. codeauthor:: Joseph Agor <Joseph.Agor jhuapl edu>
"""


import math
import random
import copy
import math
import random
from collections.abc import Mapping
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy import stats

from .inputs import update_nested_dict_by_dict
import warnings
import ast


##########################################################################
## START: Updated New Functions for Fitting Distributions with RBS data ##
##########################################################################

def get_range_key(d, num_plants):
    last_key = None
    max_upper = float("-inf")

    for key in d:
        if key.startswith("(") and key.endswith(")"):
            lower, upper = ast.literal_eval(key)

            # Track the tuple with the highest upper bound
            if upper > max_upper:
                max_upper = upper
                last_key = key

            # Normal range match
            if lower < num_plants <= upper:
                return key

    # If no match found, return the tuple with highest upper bound
    return last_key




def _set_beta_binomial_params(contamination_config, consignment):
    """
    Function to set all appropriate beta-binomial parameters based on plant quantity on the consignment.

    INPUTS
    contamination_config: contamination config
    consignment:  Consignment object

    OUTPUTS
    config_beta_binomial:  dictionary with beta-binomial parameters
    """
    beta_binomial_params = {}
    # Get the number of plants on the consignment
    if consignment.get('num_plants') is None:
        if consignment.get('plants') is None:
            warnings.warn(
                "Attempting to set the beta binomial parameters in the '_set_beta_binomial_params' function"
                             " of the contamination.py module and no plant information was found (i.e., no 'num_plants'"
                             " or 'plants' attribute in the consignment object).  Default values are being set"
                " for parameters that will not reflect any data used for training.",
                UserWarning
            )
            return contamination_config["contamination_rate"]["beta_binomial_parameters"]['default']
        else:
            num_plants = len(consignment.get('plants'))
    else:
        num_plants = consignment.get('num_plants')

    # Get the appropriate alpha and beta parameters based on the plant quantity of the consignment, otherwise default to the default parameters
    param_dict = contamination_config["contamination_rate"]["beta_binomial_parameters"]
    key = get_range_key(param_dict, num_plants)
    beta_binomial_params = param_dict[key] if key is not None else param_dict["default"]

    # Get the J parameter (number of groups) needed for contaminating via beta binomial
    if consignment.get('num_sample_units') is None:
        if consignment.get('sample_units') is None:
            warnings.warn(
                "Attempting to set the 'J' (number of groups/sample units on the consignment)"
                " beta binomial parameters in the '_set_beta_binomial_params' function"
                " of the contamination.py module and no sample unit information was found (i.e., no 'num_sample_units'"
                " or 'sample_units' attribute in the consignment object).  Default value found from the generation of"
                " Clarke input values function being used.",
                UserWarning
            )
        else:
            beta_binomial_params['J'] = len(consignment.get('sample_units'))
    else:
        beta_binomial_params['J'] = consignment.get('num_sample_units')

    # Get the N_bar parameter (number of units per group) needed for contaminating via beta binomial
    num_sample_units = beta_binomial_params['J']
    beta_binomial_params['N_bar'] = int(num_plants / num_sample_units)

    # Get theta parameter
    if beta_binomial_params['theta'] is None:
        beta_binomial_params['theta'] = np.inf

    # Set Clustering parameter
    if param_dict.get('default').get('p') is None:
        beta_binomial_params['p'] = 0
    else:
        beta_binomial_params['p'] = 1-param_dict.get('default').get('p')

    return beta_binomial_params


def add_contaminant_beta_binomial_for_groups(config, group_sizes, rng=None):
    """
    Beta-binomial contamination sampler where each 'group' is an inspection unit.

    group_sizes: array-like, shape (J,)
        Number of plants available in each inspection unit j.
    Returns:
        contaminated_plants: shape (J,)
        Number of contaminated plants in group j, guaranteed <= group_sizes[j].
    """
    group_sizes = np.asarray(group_sizes, dtype=int)
    J = group_sizes.shape[0]
    I = 1  # one p_i for the entire consignment (or whatever your model is)

    beta_binomial_config = config["beta_binomial_parameters"]
    alpha = beta_binomial_config["alpha"]
    beta = beta_binomial_config["beta"]
    theta = beta_binomial_config["theta"]

    # Seed RNG
    seed = beta_binomial_config.get("seed", None)
    rng = np.random.default_rng(seed)

    # 1) p_i ~ Beta(alpha, beta), shape (I,)
    p_i = rng.beta(alpha, beta, size=I)  # shape (1,)

    # 2) Broadcast theta to (I, J)
    theta = np.asarray(theta)
    if theta.ndim == 0:
        theta_ij = np.full((I, J), theta, dtype=float)
    elif theta.shape == (I,):
        theta_ij = np.repeat(theta[:, None], J, axis=1)
    elif theta.shape in [(I, 1), (1, J), (I, J)]:
        theta_ij = np.broadcast_to(theta, (I, J)).astype(float)
    else:
        theta_ij = np.broadcast_to(theta, (I, J)).astype(float)

    # 3) p_ij | p_i
    p_ij = np.broadcast_to(p_i[:, None], (I, J)).copy()
    finite_mask = np.isfinite(theta_ij)

    if np.any(finite_mask):
        a_ij = theta_ij * p_i[:, None]
        b_ij = theta_ij * (1.0 - p_i[:, None])

        a = a_ij[finite_mask]
        b = b_ij[finite_mask]

        p_ij[finite_mask] = rng.beta(a, b)

    # 4) X_ij | p_ij ~ Binomial(N_ij, p_ij),
    #    but now N_ij is the actual number of plants in each inspection unit.
    N_ij = group_sizes.reshape(1, J)  # shape (1, J)
    X = rng.binomial(N_ij, p_ij)
    contaminated_plants = X[0]  # shape (J,)

    return contaminated_plants

def add_contaminant_beta_binomial(beta_binomial_config):
    """
    Args:
        beta_binomial_config:  Dictionary with the folowing beta-binomial parameters as keys:
                               alpha: float
                               beta : float
                               theta : float or array-like
                                       - scalar (shared for all i,j), may be np.inf
                                       - shape (I,), (I,1), (1,J), or (I,J) (broadcastable). Entries may be np.inf.
                               Nbar : int or array-like
                                      - scalar or broadcastable to (I,J)
                               J : int
                               p: clustering parameter (0 implying not clustered at all and 1 meaning "fully clustered")

    Returns:
        X: Vector with number of infected units (plants) per group (sample unit)

    Vectorized sampler for:
        p_i ~ Beta(alpha, beta)                          (size I)
        p_ij | p_i ~ Beta(theta * p_i, theta*(1-p_i))    (size I x J)
        X_ij | p_ij ~ Binomial(Nbar, p_ij)               (size I x J)

    Special handling:
        If theta == np.inf at any position, we set p_ij = p_i there.
    """
    alpha = beta_binomial_config["alpha"]
    beta = beta_binomial_config["beta"]
    theta = beta_binomial_config["theta"]
    N_bar = beta_binomial_config["N_bar"]
    I = 1
    J = beta_binomial_config['J']

    # Seed random number generator
    rng = 1
    rng = np.random.default_rng() if rng is None else np.random.default_rng(rng)

    # 1) p_i ~ Beta(alpha, beta), shape (I,)
    p_i = rng.beta(alpha, beta, size=I)
    # --- Broadcast theta to (I, J)
    theta = np.asarray(theta)
    if theta.ndim == 0:
        theta_ij = np.full((I, J), theta, dtype=float)
    elif theta.shape == (I,):
        theta_ij = np.repeat(theta[:, None], J, axis=1)
    elif theta.shape == (I, 1) or theta.shape == (1, J) or theta.shape == (I, J):
        theta_ij = np.broadcast_to(theta, (I, J)).astype(float)
    else:
        theta_ij = np.broadcast_to(theta, (I, J)).astype(float)
    # Calculate p_ij | p_i
    # Start with the degenerate case p_ij = p_i for all cells,
    # then overwrite where theta is finite.
    p_ij = np.broadcast_to(p_i[:, None], (I, J)).copy()
    finite_mask = np.isfinite(theta_ij)  # True where theta is finite
    if np.any(finite_mask):
        # Parameters only where theta is finite
        a_ij = theta_ij * p_i[:, None]
        b_ij = theta_ij * (1.0 - p_i[:, None])

        # Draw only for finite cells (masked 1D arrays)
        a = a_ij[finite_mask]
        b = b_ij[finite_mask]

        # NOTE: rng.beta accepts array-shaped a,b and returns matching shape
        p_ij[finite_mask] = rng.beta(a, b)
    # Calculate X_ij | p_ij ~ Binomial(N_bar, p_ij)
    N_bar = np.asarray(N_bar)
    if N_bar.ndim == 0:
        N_ij = np.full((I, J), int(N_bar))
    else:
        N_ij = np.broadcast_to(N_bar, (I, J)).astype(int)
    X = rng.binomial(N_ij, p_ij)
    if len(X)>0: X = X[0]

    # Apply clustering
    if beta_binomial_config['p'] > 0 and sum(X)>0:
        n = min(int(round(J * (1-beta_binomial_config['p']), 0)),len(X))
        m = len(X)

        # Choose indices to become zero
        if n == m:
            zero_idx = np.random.choice(m, size=n-1, replace=False)
        else:
            zero_idx = np.random.choice(m, size=n, replace=False)

        # Identify indices that remain nonzero-eligible
        nonzero_idx = np.setdiff1d(np.arange(m), zero_idx)

        # Store values that will be removed
        values_to_redistribute = X[zero_idx].copy()

        # Zero out chosen indices
        X[zero_idx] = 0

        # Redistribute any nonzero removed values
        for val in values_to_redistribute:
            if val != 0:
                target = np.random.choice(nonzero_idx)
                X[target] += val


    return X


def contaminate_units_by_group(contaminated_plants, plant_indices):
    contaminated_plants = np.asarray(contaminated_plants).ravel()

    # Group by inspection unit, but store *indices*, not tuples
    by_inspection_unit = defaultdict(list)
    for idx, t in enumerate(plant_indices):
        a, b, c = t
        by_inspection_unit[a].append(idx)

    sampled_indices_global = []
    rng = np.random.default_rng(123)

    for inspect_unit, count in enumerate(contaminated_plants):
        count = int(count)
        if count > 0:
            group = by_inspection_unit[inspect_unit]  # list of indices into plant_indices
            if len(group) == 0:
                continue
            n = min(count, len(group))
            chosen_local = rng.choice(len(group), size=n, replace=False)
            sampled_indices_global.extend(group[i] for i in chosen_local)

    return sampled_indices_global

#######################################################################
## END Updated New Functions for Fitting Distributions with RBS data ##
#######################################################################

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
                inspection_unit.included_units[0] = 1
                # Plant contamination
                if hasattr(inspection_unit, 'included_unit_objects'):
                    inspection_unit.included_unit_objects[0].plants.fill(1)
                    inspection_unit.included_units[0] = inspection_unit.included_unit_objects[0].plants.sum()
            elif in_inspection_unit == "all":
                inspection_unit.included_units.fill(1)
                # Plant contamination
                if hasattr(inspection_unit, 'included_unit_objects'):
                    for samp_index in range(inspection_unit.num_sample_units):
                        if hasattr(inspection_unit.included_unit_objects[samp_index], 'plants'):
                            inspection_unit.included_unit_objects[samp_index].plants.fill(1)
                            inspection_unit.included_units[samp_index] = inspection_unit.included_unit_objects[samp_index].plants.sum()
            elif in_inspection_unit == "one_random":
                index = np.random.choice(inspection_unit.num_sample_units - 1)
                inspection_unit.included_units[index] = 1
                # Plant contamination
                if hasattr(inspection_unit, 'included_unit_objects'):
                    inspection_unit.included_unit_objects[index].plants.fill(1)
                    inspection_unit.included_units[index] = inspection_unit.included_unit_objects[index].plants.sum()
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
                np.put(inspection_unit.included_units, indexes, 1)
                # Plant contamination
                if hasattr(inspection_unit, 'included_unit_objects'):
                    for idx in indexes:
                        if hasattr(inspection_unit.included_unit_objects[idx], 'plants'):
                            inspection_unit.included_unit_objects[idx].plants.fill(1)
                            inspection_unit.included_units[idx] = inspection_unit.included_unit_objects[idx].plants.sum()

# modify to include clarke parameters -
def get_contamination_rate(config):
    """Get contamination rate.

    Config is the ``contamination_rate`` dictionary.
    """
    distribution = config["distribution"]
    if distribution == "fixed_value":
        return config["value"]
    if distribution in ["beta_binomial", "beta-binomial"]:
        params = config.get("beta_binomial_parameters", {})
        alpha = float(params['default'].get("alpha", 0))
        beta = float(params['default'].get("beta", 0))
        denom = alpha + beta
        return 0.0 if denom <= 0 else alpha / denom
    if distribution == "beta":
        parameters = config["parameters"]
        if isinstance(parameters, Mapping):
            param1 = parameters["a"]
            param2 = parameters["b"]
        else:
            param1, param2 = parameters
        return float(stats.beta.rvs(param1, param2, size=1))
    raise RuntimeError(f"Unknown contamination rate distribution: {distribution}")



def num_units_to_contaminate(config, num_units):
    """Return number of plant units to be contaminated
    Rounds up or down to nearest integer.

    Config is the ``contamination_rate`` dictionary.
    """
    contamination_rate = get_contamination_rate(config)
    contaminated_units = round(num_units * contamination_rate)
    return contaminated_units

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


def synchronize_contamination_arrays_from_plants(consignment):
    """Synchronize sample-unit and plant arrays to match plant-level contamination truth."""
    if not hasattr(consignment, "inspection_units"):
        return

    global_sample_unit_idx = 0
    global_plant_idx = 0
    has_global_sample_units = hasattr(consignment, "sample_units") and consignment.sample_units is not None
    has_global_plants = hasattr(consignment, "plants") and consignment.plants is not None

    for inspection_unit in consignment.inspection_units:
        sample_unit_objects = getattr(inspection_unit, "included_unit_objects", [])
        for local_su_idx, sample_unit_object in enumerate(sample_unit_objects):
            contaminated_plants_in_sample_unit = int(np.count_nonzero(sample_unit_object.plants))

            if hasattr(inspection_unit, "included_units") and local_su_idx < len(inspection_unit.included_units):
                inspection_unit.included_units[local_su_idx] = contaminated_plants_in_sample_unit

            if has_global_sample_units and global_sample_unit_idx < len(consignment.sample_units):
                consignment.sample_units[global_sample_unit_idx] = contaminated_plants_in_sample_unit
            global_sample_unit_idx += 1

            if has_global_plants:
                plant_values = np.asarray(sample_unit_object.plants, dtype=np.int64)
                n_plants = len(plant_values)
                end_idx = min(global_plant_idx + n_plants, len(consignment.plants))
                write_n = end_idx - global_plant_idx
                if write_n > 0:
                    consignment.plants[global_plant_idx:end_idx] = plant_values[:write_n]
                global_plant_idx += n_plants


def add_contaminant_uniform_random(config, consignment):
    """Add contaminants to consignment using uniform random distribution

    Contamination rate is determined using the ``contamination_rate`` config key.
    """
    contamination_unit = config["contamination_unit"]

    if contamination_unit in ["box", "boxes"]:
        contaminated_boxes = num_units_to_contaminate(
            config["contamination_rate"], consignment.num_inspection_units
        )
        if contaminated_boxes == 0.0:
            return
        box_indexes = np.random.choice(
            consignment.num_inspection_units, math.ceil(contaminated_boxes), replace=False
        )
        # Contaminate full boxes except for last one
        for box_index in box_indexes[:-1]:
            consignment.inspection_units[box_index].items.fill(1)
        # Use remainder of contaminated_boxes to partially contaminate
        # last box if needed
        partial_box_proportion = math.modf(contaminated_boxes)[0]
        # If contaminated_boxes is whole number, contaminate full box
        if partial_box_proportion == 0.0:
            partial_box_proportion = 1
        partial_box_contaminated_stems = round(
            consignment.inspection_units[box_indexes[-1]].num_items * partial_box_proportion
        )
        consignment.inspection_units[box_indexes[-1]].items[0:partial_box_contaminated_stems].fill(
            1
        )
        # Check if correct number of boxes contaminated, should be rounded up
        # contaminated_boxes, or may be rounded down contaminated_boxes
        # if no stems were contaminated in last partial box
        assert np.count_nonzero(consignment.inspection_units) in (
            math.ceil(contaminated_boxes),
            math.floor(contaminated_boxes),
        )
    elif contamination_unit in ["item", "items"]:
        contaminated_items = num_units_to_contaminate(
            config["contamination_rate"], consignment.num_plants
        )
        if contaminated_items == 0:
            return
        item_indexes = np.random.choice(
            consignment.num_plants, contaminated_items, replace=False
        )
        np.put(consignment.plants, item_indexes, 1)
        assert np.count_nonzero(consignment.plants) == contaminated_items
    elif contamination_unit in ["plant", "plants"]:
        # Contaminate plants directly per inspection unit
        if config["contamination_rate"]['distribution'] == 'beta-binomial':
            beta_binomial_params = _set_beta_binomial_params(config, consignment)
            contaminated_plants = np.asarray(add_contaminant_beta_binomial(beta_binomial_params), dtype=int).ravel()
            print(f"      contam: random | plants beta-binomial -> total contaminated plants {int(np.sum(contaminated_plants))} of {consignment.num_plants}")
            if np.all(contaminated_plants == 0):
                return
        else:
            # Fallback fixed-rate contamination across all plants
            total_plants = sum(len(su.plants) for iu in consignment.inspection_units for su in iu.included_unit_objects)
            contaminated_total = num_units_to_contaminate(config["contamination_rate"], total_plants)
            contaminated_plants = np.zeros(total_plants, dtype=int)
            if contaminated_total > 0:
                # Distribute proportionally by plant counts per inspection unit
                unit_sizes = [sum(len(su.plants) for su in iu.included_unit_objects) for iu in consignment.inspection_units]
                total_size = sum(unit_sizes)
                for idx, size in enumerate(unit_sizes):
                    share = int(round(contaminated_total * size / total_size)) if total_size else 0
                    contaminated_plants[idx] = min(share, size)
            if contaminated_plants.sum() == 0:
                return

        # Apply contamination per inspection unit
        su_counter = 0 # Define a sampling unit counter to
        for iu_idx, inspection_unit in enumerate(consignment.inspection_units):
            k = int(contaminated_plants[iu_idx]) if iu_idx < len(contaminated_plants) else 0
            if k <= 0:
                continue
            su_objs = inspection_unit.included_unit_objects
            lengths = [len(su.plants) for su in su_objs]
            unit_total = sum(lengths)
            if unit_total <= 0:
                continue
            k = min(k, unit_total)
            chosen = np.random.choice(unit_total, size=k, replace=False)
            # Map flat indices to sample units
            cum = 0
            for su_obj, n in zip(su_objs, lengths):
                if n == 0:
                    continue
                mask = (chosen >= cum) & (chosen < cum + n)
                rel_idx = chosen[mask] - cum
                if rel_idx.size > 0:
                    su_obj.plants[rel_idx] = 1
                cum += n
        # Update consignment.sample_units to sum of contaminated plants for each sample_unit
        sample_unit_counter = 0
        for inspection_unit in consignment.inspection_units:
            for sample_unit_object in inspection_unit.included_unit_objects:
                consignment.sample_units[sample_unit_counter] = sample_unit_object.plants.sum()
                sample_unit_counter += 1

        # Test that the correct number contaminated
        total_contaminated = sum(
            (sample_unit_object.plants == 1).sum()
            for inspection_unit in consignment.inspection_units
            for sample_unit_object in inspection_unit.included_unit_objects
        )
        if config["contamination_rate"]['distribution'] == 'beta-binomial':
            expected = int(np.sum(contaminated_plants))
        else:
            expected = int(contaminated_plants)
        if total_contaminated != expected:
            print(
                f"WARNING: Contaminated plant count mismatch. Expected {expected}, got {total_contaminated}."
            )
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
            consignment.inspection_units[cluster_index].included_units.fill(1)
    cluster_start = (
        contaminated_units_per_cluster * cluster_strata[len(cluster_sizes) - 1]
    )
    cluster_indexes = np.arange(
        start=cluster_start, stop=cluster_start + cluster_sizes[-1]
    )
    for cluster_index in cluster_indexes[:-1]:
        consignment.inspection_units[cluster_index].included_units.fill(1)
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
                    sample_unit_object = consignment.inspection_units[cluster_index].included_unit_objects[samp_index]
                    num_plants = len(sample_unit_object.plants)
                    for plant_idx in range(num_plants):
                        plant_tuples.append((cluster_index, samp_index, plant_idx))
        # Partial inspection_unit (last cluster inspection_unit, possibly partial)
        cluster_start = contaminated_units_per_cluster * cluster_strata[len(cluster_sizes) - 1]
        cluster_indexes = np.arange(start=cluster_start, stop=cluster_start + cluster_sizes[-1])
        for cluster_index in cluster_indexes[:-1]:
            for samp_index in range(consignment.inspection_units[cluster_index].num_sample_units):
                sample__unit_object_object = consignment.inspection_units[cluster_index].sample__unit_objects_objects[samp_index]
                num_plants = len(sample__unit_object_object.plants)
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
            sample_unit_object = consignment.inspection_units[cluster_indexes[-1]].included_unit_objects[samp_index]
            num_plants = len(sample_unit_object.plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((cluster_indexes[-1], samp_index, plant_idx))
        total_plants = len(plant_tuples)
        num_contaminated_plants = max(1, round(total_plants * perc_plants_contaminated))
        # Set all plants in all contaminated sample_units to 0 first
        for inspection_unit_idx, samp_index, plant_idx in plant_tuples:
            consignment.inspection_units[inspection_unit_idx].included_unit_objects[samp_index].plants[plant_idx] = 0
        # Randomly contaminate the required number of plants across all pooled plants
        contaminated_plant_indices = np.random.choice(total_plants, num_contaminated_plants, replace=False)
        for idx in contaminated_plant_indices:
            inspection_unit_idx, samp_index, plant_idx = plant_tuples[idx]
            consignment.inspection_units[inspection_unit_idx].included_unit_objects[samp_index].plants[plant_idx] = 1
        # Update sample_units array to sum of contaminated plants per sample_unit
        for inspection_unit_idx, inspection_unit in enumerate(consignment.inspection_units):
            for samp_index, sample_unit_object in enumerate(inspection_unit.included_unit_objects):
                consignment.inspection_units[inspection_unit_idx].included_units[samp_index] = sample_unit_object.plants.sum()

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
        # Gather all plant indices (as tuples: (sample_unit_index, inspection_unit_idx, sample_unit_idx, plant_idx)) for selected sample_units
        plant_tuples = []
        for sample_unit_index in indexes:
            inspection_unit_idx, sample_unit_idx = consignment.get_inspection_unit_and_sample_unit_index(sample_unit_index)
            num_plants = len(consignment.inspection_units[inspection_unit_idx].included_unit_objects[sample_unit_idx].plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((sample_unit_index, inspection_unit_idx, sample_unit_idx, plant_idx))
        total_plants = len(plant_tuples)
        perc_plants_contaminated = config["clustered"]["percentage_plants_contaminated"]
        num_contaminated_plants = max(1, round(total_plants * perc_plants_contaminated))
        contaminated_plant_indices = np.random.choice(total_plants, num_contaminated_plants, replace=False)
        # Set all plants in selected sample_units to 0 first
        for sample_unit_index in indexes:
            inspection_unit_idx, sample_unit_idx = consignment.get_inspection_unit_and_sample_unit_index(sample_unit_index)
            consignment.inspection_units[inspection_unit_idx].included_unit_objects[sample_unit_idx].plants.fill(0)
        # Set contaminated plants
        for idx in contaminated_plant_indices:
            sample_unit_index, inspection_unit_idx, sample_unit_idx, plant_idx = plant_tuples[idx]
            consignment.inspection_units[inspection_unit_idx].included_unit_objects[sample_unit_idx].plants[plant_idx] = 1
        # Update sample_units array for each sample_unit
        for sample_unit_index in indexes:
            inspection_unit_idx, sample_unit_idx = consignment.get_inspection_unit_and_sample_unit_index(sample_unit_index)
            consignment.inspection_units[inspection_unit_idx].included_units[sample_unit_idx] = consignment.inspection_units[inspection_unit_idx].included_unit_objects[sample_unit_idx].plants.sum()

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
        # Gather all plant indices (as tuples: (sample_unit_index, inspection_unit_idx, sample_unit_idx, plant_idx)) for selected sample_units
        plant_tuples = []
        for sample_unit_index in cluster_indexes:
            inspection_unit_idx, sample_unit_idx = consignment.get_inspection_unit_and_sample_unit_index(sample_unit_index)
            num_plants = len(consignment.inspection_units[inspection_unit_idx].included_unit_objects[sample_unit_idx].plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((sample_unit_index, inspection_unit_idx, sample_unit_idx, plant_idx))
        total_plants = len(plant_tuples)
        perc_plants_contaminated = config["clustered"]["percentage_plants_contaminated"]
        num_contaminated_plants = max(1, round(total_plants * perc_plants_contaminated))
        contaminated_plant_indices = np.random.choice(total_plants, num_contaminated_plants, replace=False)
        # Set all plants in selected sample_units to 0 first
        for sample_unit_index in cluster_indexes:
            inspection_unit_idx, sample_unit_idx = consignment.get_inspection_unit_and_sample_unit_index(sample_unit_index)
            consignment.inspection_units[inspection_unit_idx].included_unit_objects[sample_unit_idx].plants.fill(0)
        # Set contaminated plants
        for idx in contaminated_plant_indices:
            sample_unit_index, inspection_unit_idx, sample_unit_idx, plant_idx = plant_tuples[idx]
            consignment.inspection_units[inspection_unit_idx].included_unit_objects[sample_unit_idx].plants[plant_idx] = 1
        # Update sample_units array for each sample_unit
        for sample_unit_index in cluster_indexes:
            inspection_unit_idx, sample_unit_idx = consignment.get_inspection_unit_and_sample_unit_index(sample_unit_index)
            consignment.inspection_units[inspection_unit_idx].included_units[sample_unit_idx] = consignment.inspection_units[inspection_unit_idx].included_unit_objects[sample_unit_idx].plants.sum()
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
            sample_unit_object = consignment.inspection_units[inspection_unit_idx].included_unit_objects[samp_index]
            num_plants = len(sample_unit_object.plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((None, inspection_unit_idx, samp_index, plant_idx))
    else:
        # sample_unit-level: sample_unit_indexes are sample_unit indices (int or numpy int)
        for sample_unit_index in sample_unit_indexes:
            inspection_unit_idx, sample_unit_idx = consignment.get_inspection_unit_and_sample_unit_index(int(sample_unit_index))
            sample_unit_object = consignment.inspection_units[inspection_unit_idx].included_unit_objects[sample_unit_idx]
            num_plants = len(sample_unit_object.plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((sample_unit_index, inspection_unit_idx, sample_unit_idx, plant_idx))
    total_plants = len(plant_tuples)
    if total_plants == 0:
        return
    num_contaminated_plants = max(1, round(total_plants * percentage))
    # Set all plants in all contaminated sample_units to 0 first
    for _, inspection_unit_idx, sample_unit_idx, plant_idx in plant_tuples:
        consignment.inspection_units[inspection_unit_idx].included_unit_objects[sample_unit_idx].plants[plant_idx] = 0
    contaminated_plant_indices = np.random.choice(total_plants, num_contaminated_plants, replace=False)
    for idx in contaminated_plant_indices:
        _, inspection_unit_idx, sample_unit_idx, plant_idx = plant_tuples[idx]
        consignment.inspection_units[inspection_unit_idx].included_unit_objects[sample_unit_idx].plants[plant_idx] = 1
    # Update sample_units array to sum of contaminated plants per sample_unit
    if not is_tuple:
        # sample_unit-level
        for sample_unit_index in sample_unit_indexes:
            inspection_unit_idx, sample_unit_idx = consignment.get_inspection_unit_and_sample_unit_index(int(sample_unit_index))
            consignment.inspection_units[inspection_unit_idx].included_units[sample_unit_idx] = consignment.inspection_units[inspection_unit_idx].included_unit_objects[sample_unit_idx].plants.sum()
    else:
        # inspection_unit-level
        for inspection_unit_idx, samp_index in sample_unit_indexes:
            consignment.inspection_units[inspection_unit_idx].included_units[samp_index] = consignment.inspection_units[inspection_unit_idx].included_unit_objects[samp_index].plants.sum()


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
            print("      contam: arrangement=random_inspection_unit")
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
            print("      contam: arrangement=random")
            specific_contamination_config = get_contamination_config_for_consignment(
                config, consignment
            )
            return add_contaminant_uniform_random(specific_contamination_config, consignment)

    elif arrangement == "clustered":

        def add_contaminant(consignment):
            print("      contam: arrangement=clustered")
            specific_contamination_config = get_contamination_config_for_consignment(
                config, consignment
            )
            return add_contaminant_clusters(specific_contamination_config, consignment)
    else:
        raise RuntimeError(f"Unknown contaminant arrangement: {arrangement}")
    return add_contaminant



