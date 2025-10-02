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

=====================================
JHU/APL Extensions and Modifications:
=====================================

Contributors: Gary Lin, Joseph Agor (Johns Hopkins University Applied Physics Laboratory)

Modified Functions:
------------------
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

Notes:
------
- Updated contamination_unit parameter mapping throughout contamination functions
- All contamination functions now support both legacy and new terminology
- Enhanced support for hierarchical contamination at inspection_unit, sample_unit, and plant levels
- Maintains full backward compatibility with existing contamination configuration files
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
                if hasattr(inspection_unit, 'sample_unit_objects'):
                    inspection_unit.sample_unit_objects[0].plants.fill(1)
                    inspection_unit.sample_units[0] = inspection_unit.sample_unit_objects[0].plants.sum()
            elif in_inspection_unit == "all":
                inspection_unit.sample_units.fill(1)
                # Plant contamination
                if hasattr(inspection_unit, 'sample_unit_objects'):
                    for samp_index in range(inspection_unit.num_sample_units):
                        if hasattr(inspection_unit.sample_unit_objects[samp_index], 'plants'):
                            inspection_unit.sample_unit_objects[samp_index].plants.fill(1)
                            inspection_unit.sample_units[samp_index] = inspection_unit.sample_unit_objects[samp_index].plants.sum()
            elif in_inspection_unit == "one_random":
                index = np.random.choice(inspection_unit.num_sample_units - 1)
                inspection_unit.sample_units[index] = 1
                # Plant contamination
                if hasattr(inspection_unit, 'sample_unit_objects'):
                    inspection_unit.sample_unit_objects[index].plants.fill(1)
                    inspection_unit.sample_units[index] = inspection_unit.sample_unit_objects[index].plants.sum()
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
                if hasattr(inspection_unit, 'sample_unit_objects'):
                    for idx in indexes:
                        if hasattr(inspection_unit.sample_unit_objects[idx], 'plants'):
                            inspection_unit.sample_unit_objects[idx].plants.fill(1)
                            inspection_unit.sample_units[idx] = inspection_unit.sample_unit_objects[idx].plants.sum()

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
        # Mark ALL sample_units in ALL contaminated inspection_units as contaminated
        # When contaminating at inspection unit level, the entire inspection unit should be contaminated
        for inspection_unit_index in inspection_unit_indexes:
            consignment.inspection_units[inspection_unit_index].sample_units.fill(1)

        # Pooled plant-level contamination for all contaminated sample_units in all contaminated inspection_units
        if consignment.num_plants is not None:
            # Gather all (inspection_unit_idx, samp_index) tuples for contaminated sample_units in contaminated inspection_units
            contaminated_sample_units = []
            for inspection_unit_index in inspection_unit_indexes:
                for samp_index in range(consignment.inspection_units[inspection_unit_index].num_sample_units):
                    contaminated_sample_units.append((inspection_unit_index, samp_index))
            perc_plants_contaminated = config["clustered"]["percentage_plants_contaminated"]
            _apply_pooled_plant_level_contamination(consignment, contaminated_sample_units, perc_plants_contaminated)
        # Note: sample_unit arrays were already set to 1 above, no additional action needed for non-plant cases
        
        # Synchronize global sample_units array with inspection unit arrays after contamination
        if hasattr(consignment, 'sample_units'):
            sample_unit_idx = 0
            for inspection_unit in consignment.inspection_units:
                for su_idx, su_value in enumerate(inspection_unit.sample_units):
                    if sample_unit_idx < len(consignment.sample_units):
                        consignment.sample_units[sample_unit_idx] = su_value
                        sample_unit_idx += 1

        assert len(inspection_unit_indexes) in (
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
            
            # Synchronize global sample_units array with inspection unit arrays after contamination
            sample_unit_idx = 0
            for inspection_unit in consignment.inspection_units:
                for su_idx, su_value in enumerate(inspection_unit.sample_units):
                    consignment.sample_units[sample_unit_idx] = su_value
                    sample_unit_idx += 1
            
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
        # Assume consignment has inspection_units, each inspection_unit has sample_unit_objects, each sample_unit_object has plants
        # Flatten all plants into a 1D array for indexing
        all_plants = []
        plant_indices = []  # (inspection_unit_idx, sample_unit_idx, plant_idx)
        for inspection_unit_idx, inspection_unit in enumerate(consignment.inspection_units):
            for sample_unit_idx, sample_unit_object in enumerate(inspection_unit.sample_unit_objects):
                for plant_idx in range(len(sample_unit_object.plants)):
                    all_plants.append(sample_unit_object.plants)
                    plant_indices.append((inspection_unit_idx, sample_unit_idx, plant_idx))
        num_plants = len(plant_indices)
        contaminated_plants = num_sample_units_to_contaminate(config["contamination_rate"], num_plants)
        if contaminated_plants == 0:
            return
        plant_indexes = np.random.choice(num_plants, contaminated_plants, replace=False)
        for idx in plant_indexes:
            inspection_unit_idx, sample_unit_idx, plant_idx = plant_indices[idx]
            consignment.inspection_units[inspection_unit_idx].sample_unit_objects[sample_unit_idx].plants[plant_idx] = 1
        # Update consignment.sample_units to sum of contaminated plants for each sample_unit
        sample_unit_counter = 0
        for inspection_unit in consignment.inspection_units:
            for sample_unit_object in inspection_unit.sample_unit_objects:
                consignment.sample_units[sample_unit_counter] = sample_unit_object.plants.sum()
                sample_unit_counter += 1
        # Test correct number contaminated
        total_contaminated = sum(
            (sample_unit_object.plants == 1).sum() for inspection_unit in consignment.inspection_units for sample_unit_object in inspection_unit.sample_unit_objects
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
                    sample_unit_object = consignment.inspection_units[cluster_index].sample_unit_objects[samp_index]
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
            sample_unit_object = consignment.inspection_units[cluster_indexes[-1]].sample_unit_objects[samp_index]
            num_plants = len(sample_unit_object.plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((cluster_indexes[-1], samp_index, plant_idx))
        total_plants = len(plant_tuples)
        num_contaminated_plants = max(1, round(total_plants * perc_plants_contaminated))
        # Set all plants in all contaminated sample_units to 0 first
        for inspection_unit_idx, samp_index, plant_idx in plant_tuples:
            consignment.inspection_units[inspection_unit_idx].sample_unit_objects[samp_index].plants[plant_idx] = 0
        # Randomly contaminate the required number of plants across all pooled plants
        contaminated_plant_indices = np.random.choice(total_plants, num_contaminated_plants, replace=False)
        for idx in contaminated_plant_indices:
            inspection_unit_idx, samp_index, plant_idx = plant_tuples[idx]
            consignment.inspection_units[inspection_unit_idx].sample_unit_objects[samp_index].plants[plant_idx] = 1
        # Update sample_units array to sum of contaminated plants per sample_unit
        for inspection_unit_idx, inspection_unit in enumerate(consignment.inspection_units):
            for samp_index, sample_unit_object in enumerate(inspection_unit.sample_unit_objects):
                consignment.inspection_units[inspection_unit_idx].sample_units[samp_index] = sample_unit_object.plants.sum()

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
            num_plants = len(consignment.inspection_units[inspection_unit_idx].sample_unit_objects[sample_unit_idx].plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((sample_unit_index, inspection_unit_idx, sample_unit_idx, plant_idx))
        total_plants = len(plant_tuples)
        perc_plants_contaminated = config["clustered"]["percentage_plants_contaminated"]
        num_contaminated_plants = max(1, round(total_plants * perc_plants_contaminated))
        contaminated_plant_indices = np.random.choice(total_plants, num_contaminated_plants, replace=False)
        # Set all plants in selected sample_units to 0 first
        for sample_unit_index in indexes:
            inspection_unit_idx, sample_unit_idx = consignment.get_inspection_unit_and_sample_unit_index(sample_unit_index)
            consignment.inspection_units[inspection_unit_idx].sample_unit_objects[sample_unit_idx].plants.fill(0)
        # Set contaminated plants
        for idx in contaminated_plant_indices:
            sample_unit_index, inspection_unit_idx, sample_unit_idx, plant_idx = plant_tuples[idx]
            consignment.inspection_units[inspection_unit_idx].sample_unit_objects[sample_unit_idx].plants[plant_idx] = 1
        # Update sample_units array for each sample_unit
        for sample_unit_index in indexes:
            inspection_unit_idx, sample_unit_idx = consignment.get_inspection_unit_and_sample_unit_index(sample_unit_index)
            consignment.inspection_units[inspection_unit_idx].sample_units[sample_unit_idx] = consignment.inspection_units[inspection_unit_idx].sample_unit_objects[sample_unit_idx].plants.sum()

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
            num_plants = len(consignment.inspection_units[inspection_unit_idx].sample_unit_objects[sample_unit_idx].plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((sample_unit_index, inspection_unit_idx, sample_unit_idx, plant_idx))
        total_plants = len(plant_tuples)
        perc_plants_contaminated = config["clustered"]["percentage_plants_contaminated"]
        num_contaminated_plants = max(1, round(total_plants * perc_plants_contaminated))
        contaminated_plant_indices = np.random.choice(total_plants, num_contaminated_plants, replace=False)
        # Set all plants in selected sample_units to 0 first
        for sample_unit_index in cluster_indexes:
            inspection_unit_idx, sample_unit_idx = consignment.get_inspection_unit_and_sample_unit_index(sample_unit_index)
            consignment.inspection_units[inspection_unit_idx].sample_unit_objects[sample_unit_idx].plants.fill(0)
        # Set contaminated plants
        for idx in contaminated_plant_indices:
            sample_unit_index, inspection_unit_idx, sample_unit_idx, plant_idx = plant_tuples[idx]
            consignment.inspection_units[inspection_unit_idx].sample_unit_objects[sample_unit_idx].plants[plant_idx] = 1
        # Update sample_units array for each sample_unit
        for sample_unit_index in cluster_indexes:
            inspection_unit_idx, sample_unit_idx = consignment.get_inspection_unit_and_sample_unit_index(sample_unit_index)
            consignment.inspection_units[inspection_unit_idx].sample_units[sample_unit_idx] = consignment.inspection_units[inspection_unit_idx].sample_unit_objects[sample_unit_idx].plants.sum()
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
            sample_unit_object = consignment.inspection_units[inspection_unit_idx].sample_unit_objects[samp_index]
            num_plants = len(sample_unit_object.plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((None, inspection_unit_idx, samp_index, plant_idx))
    else:
        # sample_unit-level: sample_unit_indexes are sample_unit indices (int or numpy int)
        for sample_unit_index in sample_unit_indexes:
            inspection_unit_idx, sample_unit_idx = consignment.get_inspection_unit_and_sample_unit_index(int(sample_unit_index))
            sample_unit_object = consignment.inspection_units[inspection_unit_idx].sample_unit_objects[sample_unit_idx]
            num_plants = len(sample_unit_object.plants)
            for plant_idx in range(num_plants):
                plant_tuples.append((sample_unit_index, inspection_unit_idx, sample_unit_idx, plant_idx))
    total_plants = len(plant_tuples)
    if total_plants == 0:
        return
    num_contaminated_plants = max(1, round(total_plants * percentage))
    # Set all plants in all contaminated sample_units to 0 first
    for _, inspection_unit_idx, sample_unit_idx, plant_idx in plant_tuples:
        consignment.inspection_units[inspection_unit_idx].sample_unit_objects[sample_unit_idx].plants[plant_idx] = 0
    contaminated_plant_indices = np.random.choice(total_plants, num_contaminated_plants, replace=False)
    for idx in contaminated_plant_indices:
        _, inspection_unit_idx, sample_unit_idx, plant_idx = plant_tuples[idx]
        consignment.inspection_units[inspection_unit_idx].sample_unit_objects[sample_unit_idx].plants[plant_idx] = 1
    # Update sample_units array to sum of contaminated plants per sample_unit
    if not is_tuple:
        # sample_unit-level
        for sample_unit_index in sample_unit_indexes:
            inspection_unit_idx, sample_unit_idx = consignment.get_inspection_unit_and_sample_unit_index(int(sample_unit_index))
            consignment.inspection_units[inspection_unit_idx].sample_units[sample_unit_idx] = consignment.inspection_units[inspection_unit_idx].sample_unit_objects[sample_unit_idx].plants.sum()
    else:
        # inspection_unit-level
        for inspection_unit_idx, samp_index in sample_unit_indexes:
            consignment.inspection_units[inspection_unit_idx].sample_units[samp_index] = consignment.inspection_units[inspection_unit_idx].sample_unit_objects[samp_index].plants.sum()


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

