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
from collections.abc import Mapping
from datetime import datetime

import numpy as np
from scipy import stats

from popsborder.inputs import update_nested_dict_by_dict


###################################################################
## Updated New Functions for Fitting Distributions with RBS data ##
###################################################################

def rbs_data_fitter():
    print('WORKS')

## Translated R functionality from Clark et al. (2023) paper and their corresponding GitHub
import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, log1p  # log1p(x)=log(1+x)

# ------------------------------------------------------------------
# 1.  Helpers -------------------------------------------------------
# ------------------------------------------------------------------
def beta_ratio(alpha, beta, bplus):
    """beta(alpha, beta+bplus) / beta(alpha, beta)  (stable log–space)"""
    return np.exp(betaln(alpha, beta + bplus) - betaln(alpha, beta))

def beta_diff_ratio(alpha, beta, add1, add2):
    """
    [Beta(α, β+add1) - Beta(α, β+add2)] / Beta(α, β),  with 0 <= add1 < add2.
    """
    if add1 >= add2:
        raise ValueError("add1 must be strictly less than add2")
    # log-diff-exp: log(exp(x) - exp(y)) = x + log(1 - exp(y-x)), with x>y
    x = betaln(alpha, beta + add1) - betaln(alpha, beta)
    y = betaln(alpha, beta + add2) - betaln(alpha, beta)
    # ensure x >= y; if not, swap
    if y > x:
        x, y = y, x
    return np.exp(x + log1mexp(y - x))

def log1mexp(logx):
    """
    Stable log(1-exp(logx)) for logx<=0   (R's Rmpfr::log1mexp)
    """
    # Two regions for stability
    if logx > -0.6931471805599453:               # ln(0.5)
        return np.log(-np.expm1(logx))           # log(1-exp) when exp(logx) close to 1
    else:
        return np.log1p(-np.exp(logx))           # generic case






# ------------------------------------------------------------------
# 2.  Negative log-likelihood (CHATGPT conversion of CLARKE ET AL. (2023) WORK) --------------------------------------
# ------------------------------------------------------------------

## 2a)  Closed-form predictive pmf  P(T_y = k)
def _dty_scalar(k, b, Nbar, alpha, beta):
    """
    P(Ty = k), where                        ┌ p ~ Beta(α,β)
      Ty | p  ~  Binom(b, 1 − (1−p)^Nbar)   └ Ty = # positive groups

    Closed-form integral:
        P =  C(b,k)  Σ_{j=0}^k (−1)^j C(k,j) · B(α, β + Nbar·(b−k+j)) / B(α,β)
    See derivation in the comments.
    """
    if k < 0 or k > b:
        return 0.0

    n = Nbar * b
    if k == 0:
        # Beta(α, β+n)/Beta(α,β)
        return beta_ratio(alpha, beta, n)
    if k == 1:
        # b * [Beta(α, β+n-Nbar) - Beta(α, β+n)] / Beta(α,β)
        return b * beta_diff_ratio(alpha, beta, n - Nbar, n)

    """
    term_sum = 0.0
    for j in range(k + 1):
        coeff = ((-1) ** j) * math.comb(k, j)
        term_sum += coeff * beta_ratio(alpha, beta, Nbar * (b - k + j))

    return math.comb(b, k) * term_sum
    """

def dty(ty, b, Nbar, *, alpha, beta, theta=np.inf):
    if theta != np.inf:
        raise NotImplementedError("θ ≠ ∞ not implemented yet.")
    ty = np.atleast_1d(np.asarray(ty, dtype=int))
    b = np.full(ty.shape, int(b)) if np.ndim(b) == 0 else np.asarray(b, dtype=int)
    Nbar = np.full(ty.shape, int(Nbar)) if np.ndim(Nbar) == 0 else np.asarray(Nbar, dtype=int)
    out = np.empty_like(ty, dtype=float)
    for i, (k, bi, Ni) in enumerate(zip(ty, b, Nbar)):
        out[i] = _dty_scalar(int(k), int(bi), int(Ni), float(alpha), float(beta))
    return out


## 2b)  Paper’s negll() conversion

def negll(par, ty, freq, b, Nbar, R=1000, theta=np.inf):
    par = np.asarray(par, dtype=float)
    if par.size < 2:
        return np.inf
    alpha, beta = np.exp(par[0]), np.exp(par[1])
    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0 or beta <= 0:
        return np.inf

    probs = dty(ty, b, Nbar, alpha=alpha, beta=beta, theta=theta)
    if np.any(~np.isfinite(probs)) or np.any(probs <= 0):
        return np.inf

    freq = np.ones_like(ty, dtype=float) if freq is None else np.asarray(freq, dtype=float)
    return -np.sum(freq * np.log(probs))









# ------------------------------------------------------------------
# 2.  Negative log-likelihood (ANOTHER EXAMPLE - NOT CLARKE ET AL. (2023) WORK) --------------------------------------
# ------------------------------------------------------------------

def negll2(par, ty, freq, b, Nbar, R, theta):     # <<<<<< DEFINE!
    """
    You know exactly what was in the R function; replicate it here.
    par    : log(alpha), log(beta)
    ty     : array of counts (length m)
    freq   : frequencies for those counts
    b      : number of *tested* units per group
    Nbar   : mean group size
    R,theta: whatever roles they play in your model
    Return : *negative* log-likelihood (scalar)
    """
    # Example: vanilla Beta-Binomial component
    # ---------------------------------------------------------------
    alpha, beta = np.exp(par)        # positivity constraints
    n = Nbar * b
    ll = 0.0
    for k, f in zip(ty, freq):
        # Beta-Binomial pmf in log-space
        comb = (np.math.lgamma(n + 1)
                - np.math.lgamma(k + 1)
                - np.math.lgamma(n - k + 1))
        ll += f * (comb + betaln(k + alpha, n - k + beta) - betaln(alpha, beta))
    # ---------------------------------------------------------------
    # add any extra terms involving R, theta etc. here
    # ---------------------------------------------------------------
    return -ll                      # SciPy minimises
    # ----------------------------------------------------------------







# ------------------------------------------------------------------
# 3.  Main wrapper --------------------------------------------------
# ------------------------------------------------------------------
def bb_group_model(ty, b, B, Nbar, freq, theta=np.inf, R=1000,
                   startval=(0.0, 0.0), se=False):
    ty = np.asarray(ty, dtype=int)
    freq = np.asarray(freq, dtype=int)

    # Corner case: all zero observations
    if np.all(ty == 0):
        out = dict(alpha=0, beta=0, mu=0, rho=0, D=1,
                   E_leak=0, prob_leak=0, log_prob_leak=-np.inf,
                   pty0=1, optim_results=None)
        if se:
            out.update(se_alpha=0, se_beta=0, se_mu=0,
                       se_rho=0, se_D=0, se_par=np.zeros(2))
        return out

    opt = minimize(negll, x0=np.asarray(startval, float),
                   args=(ty, freq, b, Nbar, R, theta),
                   method="BFGS", options=dict(disp=False))

    if not opt.success:
        print('Optimization failed.')
        # You can choose to raise, or keep going with best-so-far params.
        # Here we proceed but surface the message.
        pass

    # ----------------------------------------------------------------
    # 4.  Derived estimates ------------------------------------------
    # ----------------------------------------------------------------
    alpha, beta = np.exp(opt.x)
    mu = alpha / (alpha + beta)
    rho = 1.0 / (alpha + beta + 1.0)
    D = 1.0 + (Nbar - 1.0) * rho

    N = Nbar * B
    n = Nbar * b

    # p(T_y = 0)
    pty0 = np.exp(betaln(alpha, beta + n) - betaln(alpha, beta))

    # E[leak]  &  p(leak)   (ported exactly from your R)
    E_leak = (N - n) * alpha / (alpha + beta + n) * beta_ratio(alpha, beta, n)

    s1 = np.sum(np.log1p(-alpha / (alpha + beta + np.arange(0, n))))
    s2 = np.sum(np.log1p(-alpha / (alpha + beta + np.arange(n, N))))
    log_prob_leak = s1 + log1mexp(s2)
    prob_leak = np.exp(log_prob_leak)

    # ----------------------------------------------------------------
    # 5.  Standard errors via Hessian -------------------------------
    # ----------------------------------------------------------------
    out = dict(optim_results=opt, alpha=alpha, beta=beta, mu=mu, rho=rho,
               D=D, E_leak=E_leak, prob_leak=prob_leak,
               log_prob_leak=log_prob_leak, pty0=pty0)

    if se:
        # opt.hess_inv is an *inverse* Hessian approximation (BFGS)
        par_vcov = opt.hess_inv
        se_par = np.sqrt(np.diag(par_vcov))
        se_alpha = alpha * se_par[0]
        se_beta = beta * se_par[1]

        dmu_dpar = np.array([1, -1]) * alpha * beta / (alpha + beta) ** 2
        se_mu = np.sqrt(dmu_dpar @ par_vcov @ dmu_dpar)

        drho_dpar = -np.array([alpha, beta]) / (alpha + beta + 1) ** 2
        se_rho = np.sqrt(drho_dpar @ par_vcov @ drho_dpar)
        se_D = (Nbar - 1) * se_rho

        out.update(se_alpha=se_alpha, se_beta=se_beta, se_mu=se_mu,
                   se_rho=se_rho, se_D=se_D, se_par=se_par)

    return out



# This function is not used or working, consider updating or removing.
def add_contaminant_to_random_box(config, consignment, contamination_rate=None):
    """Add contaminant to consignment

    Assuming a list of boxes with the non-contaminated boxes set to False.

    Each item (box) in boxes (list) is set to True if a contaminant is
    there, False otherwise.

    :param config: ``random_box`` config dictionary
    :param consignment: Consignment to contaminate
    :param contamination_rate: ``contamination_rate`` config dictionary
    """
    contaminant_probability = config["probability"]
    contaminant_ratio = config["ratio"]
    if random.random() >= contaminant_probability:
        return
    for box in consignment.boxes:
        if random.random() < contaminant_ratio:
            in_box = config.get("in_box_arrangement", "all")
            if in_box == "first":
                # simply put one contaminant to first item in the box
                box.items[0] = 1
                # Plant contamination
                if hasattr(box, 'sampleunit'):
                    box.sampleunit[0].plants.fill(1)
                    box.items[0] = box.sampleunit[0].plants.sum()
            elif in_box == "all":
                box.items.fill(1)
                # Plant contamination
                if hasattr(box, 'sampleunit'):
                    for samp_index in range(box.num_items):
                        if hasattr(box.sampleunit[samp_index], 'plants'):
                            box.sampleunit[samp_index].plants.fill(1)
                            box.items[samp_index] = box.sampleunit[samp_index].plants.sum()
            elif in_box == "one_random":
                index = np.random.choice(box.num_items - 1)
                box.items[index] = 1
                # Plant contamination
                if hasattr(box, 'sampleunit'):
                    box.sampleunit[index].plants.fill(1)
                    box.items[index] = box.sampleunit[index].plants.sum()
            elif in_box == "random":
                if not contamination_rate:
                    raise ValueError(
                        "contamination_rate must be set if arrangement is random"
                    )
                num_contaminated_items = num_items_to_contaminate(
                    contamination_rate, box.num_items
                )
                if num_contaminated_items == 0:
                    continue
                indexes = np.random.choice(
                    box.num_items, num_contaminated_items, replace=False
                )
                np.put(box.items, indexes, 1)
                # Plant contamination
                if hasattr(box, 'sampleunit'):
                    for idx in indexes:
                        if hasattr(box.sampleunit[idx], 'plants'):
                            box.sampleunit[idx].plants.fill(1)
                            box.items[idx] = box.sampleunit[idx].plants.sum()


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
        return float(stats.beta.rvs(param1, param2, size=1)[0])
    raise RuntimeError(f"Unknown contamination rate distribution: {distribution}")


def num_items_to_contaminate(config, num_items):
    """Return number of items to be contaminated
    Rounds up or down to nearest integer.

    Config is the ``contamination_rate`` dictionary.
    """
    contamination_rate = get_contamination_rate(config)
    contaminated_items = round(num_items * contamination_rate)
    return contaminated_items


def num_boxes_to_contaminate(config, num_boxes):
    """Return number of boxes to be contaminated as float.

    Config is the ``contamination_rate`` dictionary.
    """
    contamination_rate = get_contamination_rate(config)
    contaminated_boxes = num_boxes * contamination_rate
    return contaminated_boxes


def add_contaminant_uniform_random(config, consignment):
    """Add contaminants to consignment using uniform random distribution

    Contamination rate is determined using the ``contamination_rate`` config key.
    """
    contamination_unit = config["contamination_unit"]

    if contamination_unit in ["box", "boxes"]:
        contaminated_boxes = num_boxes_to_contaminate(
            config["contamination_rate"], consignment.num_boxes
        )
        if contaminated_boxes == 0.0:
            return
        box_indexes = np.random.choice(
            consignment.num_boxes, math.ceil(contaminated_boxes), replace=False
        )
        # Mark contaminated items in all contaminated boxes (full and partial)
        for box_index in box_indexes[:-1]:
            consignment.boxes[box_index].items.fill(1)
        partial_box_proportion = math.modf(contaminated_boxes)[0]
        if partial_box_proportion == 0.0:
            partial_box_proportion = 1
        partial_box_contaminated_stems = round(
            consignment.boxes[box_indexes[-1]].num_items * partial_box_proportion
        )
        consignment.boxes[box_indexes[-1]].items[0:partial_box_contaminated_stems].fill(1)

        # Pooled plant-level contamination for all contaminated items in all contaminated boxes
        if consignment.num_plants is not None:
            # Gather all (box_idx, samp_index) tuples for contaminated items
            contaminated_items = []
            for box_index in box_indexes[:-1]:
                for samp_index in range(consignment.boxes[box_index].num_items):
                    contaminated_items.append((box_index, samp_index))
            for samp_index in range(partial_box_contaminated_stems):
                contaminated_items.append((box_indexes[-1], samp_index))
            perc_plants_contaminated = config["clustered"]["percentage_plants_contaminated"]
            _apply_pooled_plant_level_contamination(consignment, contaminated_items, perc_plants_contaminated)
        else:
            # No plant unit exists, so set item array directly for all contaminated items
            for box_index in box_indexes[:-1]:
                consignment.boxes[box_index].items.fill(1)
            for samp_index in range(partial_box_contaminated_stems):
                consignment.boxes[box_indexes[-1]].items[samp_index] = 1

        assert np.count_nonzero(consignment.boxes) in (
            math.ceil(contaminated_boxes),
            math.floor(contaminated_boxes),
        )
    elif contamination_unit in ["item", "items"]:
        contaminated_items = num_items_to_contaminate(
            config["contamination_rate"], consignment.num_items
        )
        if contaminated_items == 0:
            return
        item_indexes = np.random.choice(
            consignment.num_items, contaminated_items, replace=False
        )
        if consignment.num_plants is not None:
            perc_plants_contaminated = config["clustered"]["percentage_plants_contaminated"]
            _apply_pooled_plant_level_contamination(consignment, list(item_indexes), perc_plants_contaminated)
        else:
            # No plant unit exists, so set item array directly
            np.put(consignment.items, item_indexes, 1)
        assert np.count_nonzero(consignment.items) == contaminated_items
    elif contamination_unit in ["plant", "plants"]:
        # Contaminate plants directly
        # Assume consignment has boxes, each box has sampleunit, each sampleunit has plants
        # Flatten all plants into a 1D array for indexing
        all_plants = []
        plant_indices = []  # (box_idx, sampleunit_idx, plant_idx)
        for box_idx, box in enumerate(consignment.boxes):
            for sampleunit_idx, sampleunit in enumerate(box.sampleunit):
                for plant_idx in range(len(sampleunit.plants)):
                    all_plants.append(sampleunit.plants)
                    plant_indices.append((box_idx, sampleunit_idx, plant_idx))
        num_plants = len(plant_indices)
        contaminated_plants = num_items_to_contaminate(config["contamination_rate"], num_plants)
        if contaminated_plants == 0:
            return
        plant_indexes = np.random.choice(num_plants, contaminated_plants, replace=False)
        for idx in plant_indexes:
            box_idx, sampleunit_idx, plant_idx = plant_indices[idx]
            consignment.boxes[box_idx].sampleunit[sampleunit_idx].plants[plant_idx] = 1
        # Update consignment.items to sum of contaminated plants for each item
        item_counter = 0
        for box in consignment.boxes:
            for sampleunit in box.sampleunit:
                consignment.items[item_counter] = sampleunit.plants.sum()
                item_counter += 1
        # Test correct number contaminated
        total_contaminated = sum(
            (sampleunit.plants == 1).sum() for box in consignment.boxes for sampleunit in box.sampleunit
        )
        assert total_contaminated == contaminated_plants
    else:
        raise RuntimeError(f"Unknown contamination unit: {contamination_unit}")


def _contaminated_items_to_cluster_sizes(
    contaminated_items, contaminated_units_per_cluster
):
    """Get list of cluster sizes for a given number of contaminated items

    The size of each cluster is limited by contaminated_units_per_cluster.
    """
    if contaminated_items > contaminated_units_per_cluster:
        # Split into n clusters so that n-1 clusters have the max size and
        # the last one has the remaining items.
        # Alternative would be sth like:
        # round(contaminated_items/contaminated_units_per_cluster)
        sum_items = 0
        cluster_sizes = []
        while sum_items < contaminated_items - contaminated_units_per_cluster:
            sum_items += contaminated_units_per_cluster
            cluster_sizes.append(contaminated_units_per_cluster)
        # add remaining items
        cluster_sizes.append(contaminated_items - sum_items)
        sum_items += contaminated_items - sum_items
        assert sum_items == contaminated_items
    else:
        cluster_sizes = [contaminated_items]
    return cluster_sizes


def _contaminated_boxes_to_cluster_sizes(contaminated_boxes, contaminated_units_per_cluster):
    """Get list of cluster sizes for a given number of contaminated items

    The size of each cluster is limited by contaminated_units_per_cluster.
    """
    contaminated_boxes = math.ceil(contaminated_boxes)
    if contaminated_boxes > contaminated_units_per_cluster:
        # Split into n clusters so that n-1 clusters have the max size and
        # the last one has the remaining items.
        sum_boxes = 0
        cluster_sizes = []
        while sum_boxes < contaminated_boxes - contaminated_units_per_cluster:
            sum_boxes += contaminated_units_per_cluster
            cluster_sizes.append(contaminated_units_per_cluster)
        # add last cluster with remaining contaminated boxes
        cluster_sizes.append(contaminated_boxes - sum_boxes)
        sum_boxes += contaminated_boxes - sum_boxes
        assert sum_boxes == contaminated_boxes
    else:
        cluster_sizes = [math.ceil(contaminated_boxes)]
    return cluster_sizes


def choose_strata_for_clusters(num_units, cluster_width, num_clusters):
    """Divide array of items or boxes into strata wide enough for clusters
    so that they do not overlap. If array is not equally divisible by cluster_width,
    create one smaller stratum that can be used for a smaller cluster if needed.
    This is important for very high contamination rates that require nearly all units
    to be contaminated.
    Randomly select strata to place contaminant clusters. If contamination rate is
    low enough that not all strata are needed, omit smaller strata created from
    remainder and only select from strata wide enough to contain full sized cluster.
    Return strata selected to contaminate with clusters.

    num_units: number of boxes or items in consignment
    cluster_width: size of cluster in terms of boxes or units
    num_clusters: number of clusters to contaminate
    """
    # Round up so that one smaller remainder stratum is included
    num_strata = max(1, math.ceil(num_units / cluster_width))
    # Make sure there are enough strata for the number of clusters needed.
    if num_strata < num_clusters:
        raise ValueError(
            """Cannot avoid overlapping clusters. Increase
            contaminated_units_per_cluster
            or decrease cluster_item_width (if using item contamination_unit)"""
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


def add_contaminant_clusters_to_boxes(config, consignment):
    """Add contaminant clusters to boxes in a consignment"""
    contaminated_units_per_cluster = config["clustered"]["contaminated_units_per_cluster"]
    num_boxes = consignment.num_boxes
    contaminated_boxes = num_boxes_to_contaminate(
        config["contamination_rate"], num_boxes
    )
    if contaminated_boxes == 0:
        return
    cluster_sizes = _contaminated_boxes_to_cluster_sizes(
        contaminated_boxes, contaminated_units_per_cluster
    )
    cluster_strata = choose_strata_for_clusters(
        num_boxes, contaminated_units_per_cluster, len(cluster_sizes)
    )
    # Mark contaminated items in all contaminated boxes (full and partial)
    for index, cluster_size in enumerate(cluster_sizes[:-1]):
        cluster_start = contaminated_units_per_cluster * cluster_strata[index]
        cluster_indexes = np.arange(
            start=cluster_start, stop=cluster_start + cluster_size
        )
        for cluster_index in cluster_indexes:
            consignment.boxes[cluster_index].items.fill(1)
    cluster_start = (
        contaminated_units_per_cluster * cluster_strata[len(cluster_sizes) - 1]
    )
    cluster_indexes = np.arange(
        start=cluster_start, stop=cluster_start + cluster_sizes[-1]
    )
    for cluster_index in cluster_indexes[:-1]:
        consignment.boxes[cluster_index].items.fill(1)
    partial_box_proportion = math.modf(contaminated_boxes)[0]
    if partial_box_proportion == 0.0:
        partial_box_proportion = 1
    partial_box_contaminated_stems = round(
        consignment.boxes[cluster_indexes[-1]].num_items * partial_box_proportion
    )
    consignment.boxes[cluster_indexes[-1]].items[0:partial_box_contaminated_stems].fill(1)

    # Pooled plant-level contamination for all contaminated items in all contaminated boxes
    if consignment.num_plants is not None:
        perc_plants_contaminated = config["clustered"]["percentage_plants_contaminated"]
        plant_tuples = []  # (box_idx, samp_index, plant_idx)
        # Collect all contaminated items in all contaminated boxes
        # Full contaminated boxes (all except last cluster box if partial)
        for index, cluster_size in enumerate(cluster_sizes[:-1]):
            cluster_start = contaminated_units_per_cluster * cluster_strata[index]
            cluster_indexes = np.arange(start=cluster_start, stop=cluster_start + cluster_size)
            for cluster_index in cluster_indexes:
                for samp_index in range(consignment.boxes[cluster_index].num_items):
                    sampleunit = consignment.boxes[cluster_index].sampleunit[samp_index]
                    num_plants = len(sampleunit.plants)
                    for plant_idx in range(num_plants):
                        plant_tuples.append((cluster_index, samp_index, plant_idx))
        # Partial box (last cluster box, possibly partial)
        cluster_start = contaminated_units_per_cluster * cluster_strata[len(cluster_sizes) - 1]
        cluster_indexes = np.arange(start=cluster_start, stop=cluster_start + cluster_sizes[-1])
        for cluster_index in cluster_indexes[:-1]:
            for samp_index in range(consignment.boxes[cluster_index].num_items):
                sampleunit = consignment.boxes[cluster_index].sampleunit[samp_index]
                num_plants = len(sampleunit.plants)
                for plant_idx in range(num_plants):
                    plant_tuples.append((cluster_index, samp_index, plant_idx))
        # Partial box: only the contaminated items
        partial_box_proportion = math.modf(contaminated_boxes)[0]
        if partial_box_proportion == 0.0:
            partial_box_proportion = 1
        partial_box_contaminated_stems = round(
            consignment.boxes[cluster_indexes[-1]].num_items * partial_box_proportion
        )
        for samp_index in range(partial_box_contaminated_stems):
            sampleunit = consignment.boxes[cluster_indexes[-1]].sampleunit[samp_index]
            num_plants = len(sampleunit.plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((cluster_indexes[-1], samp_index, plant_idx))
        total_plants = len(plant_tuples)
        num_contaminated_plants = max(1, round(total_plants * perc_plants_contaminated))
        # Set all plants in all contaminated items to 0 first
        for box_idx, samp_index, plant_idx in plant_tuples:
            consignment.boxes[box_idx].sampleunit[samp_index].plants[plant_idx] = 0
        # Randomly contaminate the required number of plants across all pooled plants
        contaminated_plant_indices = np.random.choice(total_plants, num_contaminated_plants, replace=False)
        for idx in contaminated_plant_indices:
            box_idx, samp_index, plant_idx = plant_tuples[idx]
            consignment.boxes[box_idx].sampleunit[samp_index].plants[plant_idx] = 1
        # Update items array to sum of contaminated plants per item
        for box_idx, box in enumerate(consignment.boxes):
            for samp_index, sampleunit in enumerate(box.sampleunit):
                consignment.boxes[box_idx].items[samp_index] = sampleunit.plants.sum()

    # Check if correct number of boxes contaminated, should be rounded up
    # contaminated_boxes, or may be rounded down contaminated_boxes
    # if no stems were contaminated in last partial box
    
    # DEBUG
    # print([bool(box) for box in consignment.boxes])
    # print(np.count_nonzero(consignment.boxes))
    
    assert np.count_nonzero(consignment.boxes) in (
        math.ceil(contaminated_boxes),
        math.ceil(contaminated_boxes) - 1,
    )


def add_contaminant_clusters_to_items_with_subset_clustering(config, consignment):
    """Add contaminant cluster to items in a consignment using a single parameter

    Clustering equal to 0 means all items in the consignment can be contaminated with
    equal probability, i.e., the cluster spreads over the whole consignment. Clustering
    equal to 1 means that all items in the cluster are contaminated. The size of the
    cluster is then directly determined by the contamination rate.
    If the cluster would spread over the end of the consignment, we put the extra part
    of the cluster at the beginning of the consignment.
    """
    clustering = config["clustered"]["value"]
    num_of_contaminated_items = num_items_to_contaminate(
        config["contamination_rate"], consignment.num_items
    )
    if num_of_contaminated_items == 0:
        return
    subset_size = round(consignment.num_items * (1 - clustering))
    subset_size = max(subset_size, num_of_contaminated_items)
    start_index2 = None
    end_index2 = None
    if subset_size == consignment.num_items:
        start_index = 0
        end_index = consignment.num_items
    else:
        start_index = np.random.randint(0, consignment.num_items)
        if start_index + subset_size > consignment.num_items:
            start_index2 = 0
            end_index2 = subset_size - (consignment.num_items - start_index)
            end_index = consignment.num_items
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
        num_of_contaminated_items,
        replace=False,
    )
    consignment.items[indexes] = 1
    # Contaminate all plants in the sample unit (item) for every contaminated item
    if consignment.num_plants is not None:
        # Gather all plant indices (as tuples: (item_index, box_idx, sampleunit_idx, plant_idx)) for selected items
        plant_tuples = []
        for item_index in indexes:
            box_idx, sampleunit_idx = consignment.get_box_and_sampleunit_index(item_index)
            num_plants = len(consignment.boxes[box_idx].sampleunit[sampleunit_idx].plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((item_index, box_idx, sampleunit_idx, plant_idx))
        total_plants = len(plant_tuples)
        perc_plants_contaminated = config["clustered"]["percentage_plants_contaminated"]
        num_contaminated_plants = max(1, round(total_plants * perc_plants_contaminated))
        contaminated_plant_indices = np.random.choice(total_plants, num_contaminated_plants, replace=False)
        # Set all plants in selected items to 0 first
        for item_index in indexes:
            box_idx, sampleunit_idx = consignment.get_box_and_sampleunit_index(item_index)
            consignment.boxes[box_idx].sampleunit[sampleunit_idx].plants.fill(0)
        # Set contaminated plants
        for idx in contaminated_plant_indices:
            item_index, box_idx, sampleunit_idx, plant_idx = plant_tuples[idx]
            consignment.boxes[box_idx].sampleunit[sampleunit_idx].plants[plant_idx] = 1
        # Update items array for each item
        for item_index in indexes:
            box_idx, sampleunit_idx = consignment.get_box_and_sampleunit_index(item_index)
            consignment.boxes[box_idx].items[sampleunit_idx] = consignment.boxes[box_idx].sampleunit[sampleunit_idx].plants.sum()

    assert np.count_nonzero(consignment.items) == num_of_contaminated_items


def add_contaminant_clusters_to_items(config, consignment):
    """Add contaminant clusters to items in a consignment"""
    contaminated_units_per_cluster = config["clustered"]["contaminated_units_per_cluster"]
    num_items = consignment.num_items
    contaminated_items = num_items_to_contaminate(
        config["contamination_rate"], num_items
    )
    if contaminated_items == 0:
        return
    cluster_sizes = _contaminated_items_to_cluster_sizes(
        contaminated_items, contaminated_units_per_cluster
    )
    cluster_indexes = []
    distribution = config["clustered"]["distribution"]
    if distribution == "random":
        cluster_item_width = config["clustered"]["random"]["cluster_item_width"]
        if cluster_item_width < contaminated_units_per_cluster:
            raise ValueError(
                f"Maximum cluster width, currently {cluster_item_width}, needs"
                " to be at least as large as contaminated_units_per_cluster"
                " (currently {contaminated_units_per_cluster})"
            )
        # cluster can't be wider/longer than the current list of items
        cluster_item_width = min(cluster_item_width, num_items)
        cluster_strata = choose_strata_for_clusters(
            num_items, cluster_item_width, len(cluster_sizes)
        )
        for index, cluster_size in enumerate(cluster_sizes):
            cluster_start = cluster_item_width * cluster_strata[index]
            # Use smaller cluster width if placing items in smaller remainder stratum
            cluster_width = min(
                cluster_item_width, (consignment.num_items - cluster_start)
            )
            assert (
                cluster_width >= cluster_size
            ), "Not enough items available to contaminate in selected cluster stratum."
            cluster = np.random.choice(cluster_width, cluster_size, replace=False)
            cluster += cluster_start
            cluster_indexes.extend(list(cluster))
    elif distribution == "continuous":
        cluster_strata = choose_strata_for_clusters(
            num_items, contaminated_units_per_cluster, len(cluster_sizes)
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
    assert np.max(cluster_indexes) < num_items
    np.put(consignment.items, cluster_indexes, 1)
    if consignment.num_plants is not None:
        # Gather all plant indices (as tuples: (item_index, box_idx, sampleunit_idx, plant_idx)) for selected items
        plant_tuples = []
        for item_index in cluster_indexes:
            box_idx, sampleunit_idx = consignment.get_box_and_sampleunit_index(item_index)
            num_plants = len(consignment.boxes[box_idx].sampleunit[sampleunit_idx].plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((item_index, box_idx, sampleunit_idx, plant_idx))
        total_plants = len(plant_tuples)
        perc_plants_contaminated = config["clustered"]["percentage_plants_contaminated"]
        num_contaminated_plants = max(1, round(total_plants * perc_plants_contaminated))
        contaminated_plant_indices = np.random.choice(total_plants, num_contaminated_plants, replace=False)
        # Set all plants in selected items to 0 first
        for item_index in cluster_indexes:
            box_idx, sampleunit_idx = consignment.get_box_and_sampleunit_index(item_index)
            consignment.boxes[box_idx].sampleunit[sampleunit_idx].plants.fill(0)
        # Set contaminated plants
        for idx in contaminated_plant_indices:
            item_index, box_idx, sampleunit_idx, plant_idx = plant_tuples[idx]
            consignment.boxes[box_idx].sampleunit[sampleunit_idx].plants[plant_idx] = 1
        # Update items array for each item
        for item_index in cluster_indexes:
            box_idx, sampleunit_idx = consignment.get_box_and_sampleunit_index(item_index)
            consignment.boxes[box_idx].items[sampleunit_idx] = consignment.boxes[box_idx].sampleunit[sampleunit_idx].plants.sum()
    assert np.count_nonzero(consignment.items) == contaminated_items


def add_contaminant_clusters(config, consignment):
    """Add contaminant clusters to consignment

    Item (separately or in boxes) with contaminant in *consignment* evaluate
    to True after running this function.
    This function does not touch the not items not selected for contamination.
    However, they are expected to be zero.
    """
    contamination_unit = config["contamination_unit"]
    if contamination_unit in ["box", "boxes"]:
        if config["clustered"]["distribution"] == "single":
            raise RuntimeError(
                "clustering distribution 'single' is not supported for boxes"
            )
        add_contaminant_clusters_to_boxes(config, consignment)
    elif contamination_unit in ["item", "items"]:
        if config["clustered"]["distribution"] == "single":
            add_contaminant_clusters_to_items_with_subset_clustering(
                config, consignment
            )
        else:
            add_contaminant_clusters_to_items(config, consignment)
    elif contamination_unit in ["plant", "plants"]:
        if config["clustered"]["distribution"] == "single":
            raise RuntimeError(
                "clustering distribution 'single' is not supported for plants"
            )
        else:
            raise RuntimeError(
                f"clustering distribution '{contamination_unit}' is not supported for plants"
            )
    else:
        raise RuntimeError(f"Unknown contamination unit: {contamination_unit}")


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
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if end_date and isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    if not start_date and not end_date:
        return True
    elif start_date and consignment.date < start_date:
        return False
    elif end_date and consignment.date > end_date:
        return False
    return True


def get_contamination_config_for_consignment(config, consignment):
    """Get contamination configuration for specific consignment

    If the *config* contains consignment-specific settings under
    the consignments key, contamination configuration is selected
    based on the specified rules. If the consignment properties match
    the selection rules or if it passes the probability challenge,
    consignment-specific configuration is returned.

    If there is contamination key associated with the consignment settings,
    the associated value is returned as the consignment-specific configuration.
    If there is also a use_contamination_defaults key with value set to true,
    the top-level contamination configuration is used as the basis for the
    consignment-specific configuration and the values under the contamination key
    are used to modify or enhance the top-level config.

    If there is no contamination key, the consignment-specific configuration is the
    top-level contamination configuration.

    If the consignment properties do not match
    the selection rules or if it does not pass the probability challenge,
    None is returned.

    If the *config* does not contain consignment-specific settings under
    the consignments key, the function always returns a the provided
    *config*.

    In all cases, a copy the config dictionary is returned.
    """
    contaminated_consignments = config.get("consignments")
    if not contaminated_consignments:
        # No consignment-specific info, all consignments use the same config.
        return config.copy()
    # Consignment-specific input provided, create the right config for the consignment
    # if the consignment is configured to be contaminated.
    for item in contaminated_consignments:
        if consignment_matches_selection_rule(rule=item, consignment=consignment):
            # The consignment matches the selection rule. Now test if we should
            # contaminate this specific consignment.
            probability = item.get("probability")
            if probability is None or random.random() < probability:
                # This specific consignment should contaminated.
                consignment_specific_config = item.get("contamination")
                if not consignment_specific_config:
                    # If missing or empty, use the global/default one.
                    consignment_specific_config = config.copy()
                    del consignment_specific_config["consignments"]
                elif item.get("use_contamination_defaults"):
                    # There is specifc config, but the global/main contamination
                    # config should be used as the bases for the consignment-specific
                    # config.
                    default_values = config.copy()
                    del default_values["consignments"]
                    update_nested_dict_by_dict(
                        default_values, consignment_specific_config
                    )
                    consignment_specific_config = default_values
                else:
                    # In all other cases, we return a copy, so let's do for the
                    # straightforward case too.
                    consignment_specific_config = consignment_specific_config.copy()
                return consignment_specific_config
            else:
                # Only the first consignment rule is matched.
                break
    # Consignment not selected for contamination based on selection rules.
    return None


def create_contaminant_function(config):
    """Create a function based on the contamination config

    An arrangement key must be provided to specify which function should be used.
    """
    arrangement = config.get("arrangement")
    if arrangement == "random_box":

        def add_contaminant_function(consignment):
            return add_contaminant_to_random_box(
                config=config["random_box"],
                consignment=consignment,
                contamination_rate=config["contamination_rate"],
            )

    elif arrangement == "random":

        def add_contaminant_function(consignment):
            return add_contaminant_uniform_random(
                config=config, consignment=consignment
            )

    elif arrangement == "clustered":

        def add_contaminant_function(consignment):
            return add_contaminant_clusters(config=config, consignment=consignment)

    elif arrangement is None:
        raise RuntimeError("Contaminant arrangement must be set")
    else:
        raise RuntimeError(f"Unknown contaminant arrangement: {arrangement}")
    return add_contaminant_function


def get_contaminant_function(config):
    """Get function for adding contaminant to a consignment based on configuration"""
    if "consignments" in config["contamination"]:
        # If there is config for individual consignments, we define a new function
        # which first picks the right config based on its consignment parameter, then
        # creates an add contaminant function based on this config, and then it calls
        # the function with the consignment.

        def add_contaminant_function(consignment):
            """Picks config for the consignment and then call the specific function"""
            consignment_specific_config = get_contamination_config_for_consignment(
                config["contamination"], consignment
            )
            if not consignment_specific_config:
                # Do not contaminate this consignment.
                # No modification to the existing consignment provided as a parameter
                # and returning None (as all the add contaiminant functions do).
                return None
            contaminant_function = create_contaminant_function(
                consignment_specific_config
            )
            return contaminant_function(consignment)

        return add_contaminant_function

    # If there is config for individual consignments, we just create the function with
    # the default settings.
    return create_contaminant_function(config["contamination"])


def _apply_pooled_plant_level_contamination(consignment, item_indexes, percentage):
    """
    Helper to apply pooled plant-level contamination across a set of items.
    item_indexes: list of item indices (for item-level) or (box_idx, samp_index) tuples (for box-level)
    percentage: float, fraction of plants to contaminate
    """
    if not item_indexes:
        return
    # Ensure item_indexes is a list
    item_indexes = list(item_indexes)
    # Robustly check if item_indexes are tuples (box-level) or ints (item-level)
    first_elem = item_indexes[0]
    is_tuple = isinstance(first_elem, tuple)
    plant_tuples = []
    if is_tuple:
        # box-level: item_indexes are (box_idx, samp_index) tuples
        for box_idx, samp_index in item_indexes:
            sampleunit = consignment.boxes[box_idx].sampleunit[samp_index]
            num_plants = len(sampleunit.plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((None, box_idx, samp_index, plant_idx))
    else:
        # item-level: item_indexes are item indices (int or numpy int)
        for item_index in item_indexes:
            box_idx, sampleunit_idx = consignment.get_box_and_sampleunit_index(int(item_index))
            sampleunit = consignment.boxes[box_idx].sampleunit[sampleunit_idx]
            num_plants = len(sampleunit.plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((item_index, box_idx, sampleunit_idx, plant_idx))
    total_plants = len(plant_tuples)
    if total_plants == 0:
        return
    num_contaminated_plants = max(1, round(total_plants * percentage))
    # Set all plants in all contaminated items to 0 first
    for _, box_idx, sampleunit_idx, plant_idx in plant_tuples:
        consignment.boxes[box_idx].sampleunit[sampleunit_idx].plants[plant_idx] = 0
    # Randomly contaminate the required number of plants across all pooled plants
    contaminated_plant_indices = np.random.choice(total_plants, num_contaminated_plants, replace=False)
    for idx in contaminated_plant_indices:
        _, box_idx, sampleunit_idx, plant_idx = plant_tuples[idx]
        consignment.boxes[box_idx].sampleunit[sampleunit_idx].plants[plant_idx] = 1
    # Update items array to sum of contaminated plants per item
    if not is_tuple:
        # item-level
        for item_index in item_indexes:
            box_idx, sampleunit_idx = consignment.get_box_and_sampleunit_index(int(item_index))
            consignment.boxes[box_idx].items[sampleunit_idx] = consignment.boxes[box_idx].sampleunit[sampleunit_idx].plants.sum()
    else:
        # box-level
        for box_idx, samp_index in item_indexes:
            consignment.boxes[box_idx].items[samp_index] = consignment.boxes[box_idx].sampleunit[samp_index].plants.sum()

