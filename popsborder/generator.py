# Simulation of contaminated consignments and their inspections
# Copyright (C) 2018-2022 Vaclav Petras and others (see below)
# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

# Modifications:
# 
# - 2/17/2026 – 
# New Classes Added:
# SyntheticConsignmentDataGenerator:
#     * Generates synthetic consignment data for testing and simulation purposes
#     * Creates realistic consignment records with randomized attributes
#     * Uses advanced sampling techniques including Gaussian copulas
#     * Supports multiple sampling methods: naive, sequential, GMM, gaussian_copula
#     * Integrates with real PIS data for training synthetic data generation

# Sampling Methods Implemented:
# ----------------------------
# - multinomial_sample(): Naive approach sampling each column independently
# - sequential_multinomial_sample(): Sequential sampling preserving conditional dependencies
# - gmm_sample(): Gaussian Mixture Model sampling for numeric columns
# - gaussian_copula_sample(): Category-conditional Gaussian copula preserving correlations

# Data Generation Features:
# ------------------------
# - Configurable consignment attributes (origins, ports, pathways, commodities)
# - Propagative material and flower commodity support
# - Contamination modeling with configurable probability and quantities
# - Quality metrics calculation comparing original and synthetic data
# - Multiple output formats (CSV, JSON) with comprehensive statistics

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

"""
.. codeauthor:: Vaclav Petras <wenzeslaus gmail com>
.. codeauthor:: Kellyn P. Montgomery <kellynmontgomery gmail com>
.. codeauthor:: Gary Lin (Johns Hopkins University Applied Physics Laboratory) 
.. codeauthor:: Joseph Agor (Johns Hopkins University Applied Physics Laboratory) 
"""

import csv
import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, wasserstein_distance
from sklearn.mixture import GaussianMixture
import chardet
from scipy import stats
import warnings
import re
from typing import Optional, Union
from scipy import stats
from popsborder.inspections import construct_risk_units


class SyntheticConsignmentDataGenerator:
    """Generate synthetic consignment data using advanced sampling techniques
    
    This generator creates realistic consignment records with randomized attributes
    based on configurable parameters or input data files. It supports multiple
    sampling methods for preserving statistical relationships in the data.
    """
    
    def __init__(self,
                 config: dict = None,
                 producer_group_mapping: pd.DataFrame=None,
                 input_data_file: Path = None) -> None:
        """Initialize the synthetic data generator

        :param config: Optional path to input data file for training
        :param producer_group_mapping: Optional path to input data file for training
        :param input_data_file: Optional path to input data file for training
        """

        self.input_data = self._load_input_data(input_data_file)

        # Create a mapping dictionary
        producer_to_group = producer_group_mapping.set_index('PRODUCER_NAME')['grouping'].to_dict()

        # Map the values, using 'NO_GROUP_MATCH' as default
        self.input_data['producer_group'] = self.input_data['PRODUCER_NAME'].map(producer_to_group).fillna(
            'NO_GROUP_MATCH')

        self.input_data  = construct_risk_units(config=config, data=self.input_data)
        
        # Initialize random seed for reproducible results
        random.seed(42)
        np.random.seed(42)

    def _load_input_data(self, input_file):
        """Load input data file for training sampling models.

        :param input_file: Path to input data file (.csv, .xlsx, .xls)
        :return: DataFrame with loaded data or None if failed
        """
        try:
            # Normalize path and detect extension
            input_path = Path(input_file)
            ext = input_path.suffix.lower()

            # Load based on extension
            if ext == ".csv":
                # Detect encoding for CSV
                with open(input_path, "rb") as f:
                    result = chardet.detect(f.read(100000))
                detected = result.get("encoding") or "utf-8"

                # Try detected encoding first, then common fallbacks.
                encodings_to_try = []
                for enc in [detected, "utf-8-sig", "utf-8", "cp1252", "latin-1"]:
                    if enc and enc.lower() not in [e.lower() for e in encodings_to_try]:
                        encodings_to_try.append(enc)

                last_error = None
                for enc in encodings_to_try:
                    try:
                        df = pd.read_csv(input_path, encoding=enc)
                        break
                    except UnicodeDecodeError as e:
                        last_error = e
                        df = None
                if df is None and last_error is not None:
                    raise last_error

            elif ext in {".xlsx", ".xls"}:
                # For Excel files, pandas handles encoding internally.
                # You may specify engine="openpyxl" if you want to be explicit.
                df = pd.read_excel(input_path)  # engine="openpyxl" for .xlsx if needed

            else:
                raise ValueError(
                    f"Unsupported file type '{ext}'. Supported types are .csv, .xlsx, .xls."
                )

            # ----- Cleaning logic -----
            # Keep any row that has at least one value so we don't throw everything away.
            df = df.dropna(how="all")
            if df.empty:
                raise ValueError("Input data has no rows after removing empty records.")

            # Fill missing values to avoid numpy.choice errors downstream.
            for col in df.columns:
                if np.issubdtype(df[col].dtype, np.number):
                    # If column is entirely NaN, fill with 0; otherwise use median.
                    if df[col].dropna().empty:
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna("Unknown")

            print(f"Loaded {len(df)} records from {input_path} (after cleaning)")
            return df

        except Exception as e:
            print(f"Error loading input data file '{input_file}': {e}")
            return None

    def fit_best_continuous_distribution(self, data, distributions=None, criterion="aic"):
        """
        Fit several continuous distributions to 1D numeric data and select the best.
        If no distribution can be selected, fall back to fitting a beta distribution.
        RuntimeWarnings from SciPy are suppressed during fitting.

        Returns
        -------
        dist_name : str
        params : tuple
            Parameters as returned by dist.fit(data).
        """
        # --- Basic cleaning ---
        data = np.asarray(data, dtype=float)
        # Remove NaN / inf
        data = data[np.isfinite(data)]
        if data.size == 0:
            raise ValueError("No valid data to fit distribution.")

        # (Optional) clip extreme values to reduce numerical issues
        # comment these two lines out if you don't want clipping
        lo, hi = np.percentile(data, [0.1, 99.9])
        data = np.clip(data, lo, hi)

        if distributions is None:
            distributions = [
                stats.norm,
                stats.lognorm,
                stats.expon,
                stats.gamma,
                stats.beta,
            ]

        best_score = np.inf
        best_dist = None
        best_params = None

        for dist in distributions:
            try:
                # Suppress SciPy's runtime warnings inside this block
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    params = dist.fit(data)

                    if criterion == "aic":
                        # log-likelihood
                        ll = np.sum(dist.logpdf(data, *params))
                        k = len(params)
                        score = 2 * k - 2 * ll  # AIC
                    elif criterion == "ks":
                        ks_stat, _ = stats.kstest(data, dist.name, args=params)
                        score = ks_stat
                    else:
                        raise ValueError("criterion must be 'aic' or 'ks'")

                # If score is nan/inf, treat as bad fit
                if not np.isfinite(score):
                    continue

                if score < best_score:
                    best_score = score
                    best_dist = dist
                    best_params = params

            except Exception:
                # Any fitting failure: skip this distribution
                continue

        # Fallback: try beta if nothing else worked
        if best_dist is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                try:
                    beta_params = stats.beta.fit(data)
                    return "beta", beta_params
                except Exception:
                    raise RuntimeError(
                        "No distribution could be fitted successfully, "
                        "including beta fallback."
                    )

        return best_dist.name, best_params

    def sample_mixed_with_inspection(self,
                                     df,
                                     columns,
                                     n_consignments,
                                     num_inspection_units,
                                     inspection_col="INSPECTION_NUMBER",
                                     random_state=None):
        """
        - INSPECTION_NUMBER: not sampled from PMF.
          Instead, create n_consignments unique inspection numbers and repeat each
          according to sampled_uniform[i].
        - Other columns:
          * numeric -> sample from fitted continuous distribution
          * non-numeric -> sample from empirical PMF
        """
        rng = np.random.default_rng(random_state)

        num_inspection_units = np.asarray(num_inspection_units, dtype=int)
        assert len(num_inspection_units) == n_consignments, \
            "num_inspection_units must have length n_consignments"

        total_rows = int(num_inspection_units.sum())

        sampled = {}

        # 1. Build INSPECTION_NUMBER values deterministically from num_inspection_units
        #    Example: "INS_0", "INS_1", ..., but you can change this pattern.
        inspection_ids = [f"INS_{i}" for i in range(n_consignments)]
        inspection_col_values = np.repeat(inspection_ids, num_inspection_units)

        if len(inspection_col_values) != total_rows:
            raise ValueError("Sum of num_inspection_units must equal total number of rows.")

        sampled[inspection_col] = inspection_col_values

        # 2. Sample other columns according to type
        for col in columns:
            if col == inspection_col:
                continue  # already handled

            s = df[col]

            if pd.api.types.is_numeric_dtype(s):
                # Numeric column: fit and sample continuous distribution
                numeric_data = s.astype(float).to_numpy()
                numeric_data = numeric_data[~np.isnan(numeric_data)]

                if numeric_data.size == 0:
                    sampled[col] = np.full(total_rows, np.nan)
                    continue

                dist_name, params = self.fit_best_continuous_distribution(numeric_data)
                dist = getattr(stats, dist_name)
                sampled[col] = dist.rvs(*params, size=total_rows, random_state=rng)

            else:
                # Non-numeric: empirical PMF
                values, counts = np.unique(s.to_numpy(), return_counts=True)
                probs = counts / counts.sum()
                sampled[col] = rng.choice(values, size=total_rows, p=probs)

        # 3. Return as DataFrame
        sampled_df = pd.DataFrame(sampled)

        return sampled_df

    def resolve_producer_names(
            self,
            df: pd.DataFrame,
            input_col: str = "PRODUCER_NAME",
            output_col: str = "PRODUCER_NAME_RESOLVED",
            alias_map: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Simple producer entity resolution.

        - Normalizes text (strip, remove parentheses, collapse spaces).
        - Optionally applies an alias map whose KEYS are normalized
          lowercase strings and VALUES are canonical producer names.

        Returns a *copy* of df with a new column `output_col`.
        """

        df = df.copy()

        def _normalize(name: Union[str, float]):
            if pd.isna(name):
                return name
            name = str(name)

            # Strip leading/trailing spaces
            name = name.strip()

            # Remove anything in parentheses, e.g. "Producer 1 (boxes 1-2)" -> "Producer 1"
            name = re.sub(r"\s*\(.*?\)\s*", " ", name)

            # Collapse multiple spaces
            name = re.sub(r"\s+", " ", name)

            return name

        # Step 1: basic normalization
        normalized = df[input_col].map(_normalize)

        # Step 2: optional alias mapping (for handling typos / variants)
        # alias_map keys should be *normalized & lowercased* forms.
        if alias_map is not None:
            def _apply_alias(n):
                if pd.isna(n):
                    return n
                key = str(n).lower()
                return alias_map.get(key, n)

            resolved = normalized.map(_apply_alias)
        else:
            resolved = normalized

        df[output_col] = resolved
        return df

    def identify_num_inspection_units(self, df, n_consignments=1):
        # First sample the number of inspection units per consignment uniformly based on data
        counts = df["INSPECTION_NUMBER"].value_counts()
        min_count = counts.min()
        max_count = counts.max()

        num_inspection_units = np.random.uniform(low=min_count,
                                                 high=max_count,
                                                 size=n_consignments)
        return np.round(num_inspection_units).astype(int)

    def identify_num_inspection_units_conditional(self, df, n_consignments=1,
                                                  cols=None,
                                                  producer_alias_map: Optional[dict] = None):
        """
        Build a nested dictionary keyed by:
          - Case1 ("Miami PIS"):
                ("COUNTRY_OF_ORIGIN_NAME", "PROPAGATIVE_MATERIAL_TYPE", "PRODUCER_NAME")
          - Case2 (not "Miami PIS"):
                ("INSPECTION_LOCATION_NAME", "COUNTRY_OF_ORIGIN_NAME", "PROPAGATIVE_MATERIAL_TYPE")

        And (optionally) sample inspection units from a given subset later.
        """
        if cols is None:
            cols = [
                "INSPECTION_LOCATION_NAME",
                "COUNTRY_OF_ORIGIN_NAME",
                "PROPAGATIVE_MATERIAL_TYPE",
                "PRODUCER_NAME",
            ]

        df = self.resolve_producer_names(
            df,
            input_col="PRODUCER_NAME",
            output_col="PRODUCER_NAME_RESOLVED",
            alias_map=producer_alias_map,
        )

        # Top-level dict for the two cases
        result = {
            "Miami PIS": {},  # Case1
            "Non-Miami PIS": {}       # Case2
        }

        # Masks for the two cases
        miami_mask = df["INSPECTION_LOCATION_NAME"] == "Miami PIS"
        miami_df = df.loc[miami_mask]
        other_df = df.loc[~miami_mask]

        # ----- Case1: INSPECTION_LOCATION_NAME == "Miami PIS" -----
        # Keys: 3-tuples (COUNTRY_OF_ORIGIN_NAME, PROPAGATIVE_MATERIAL_TYPE, PRODUCER_NAME)
        case1_group_cols = [
            "COUNTRY_OF_ORIGIN_NAME",
            "PROPAGATIVE_MATERIAL_TYPE",
            "PRODUCER_NAME_RESOLVED",
        ]

        if not miami_df.empty:
            for key_tuple, sub_df in miami_df.groupby(case1_group_cols, dropna=False):
                # key_tuple is a 3-tuple because we grouped on 3 columns
                result["Miami PIS"][key_tuple] = sub_df

        # ----- Case2: INSPECTION_LOCATION_NAME != "Miami PIS" -----
        # Keys: 3-tuples (INSPECTION_LOCATION_NAME, COUNTRY_OF_ORIGIN_NAME, PROPAGATIVE_MATERIAL_TYPE)
        case2_group_cols = [
            "INSPECTION_LOCATION_NAME",
            "COUNTRY_OF_ORIGIN_NAME",
            "PROPAGATIVE_MATERIAL_TYPE",
        ]

        if not other_df.empty:
            for key_tuple, sub_df in other_df.groupby(case2_group_cols, dropna=False):
                result["Non-Miami PIS"][key_tuple] = sub_df

        # 3) Compute and store sampling probabilities based on row counts
        self._case_sampling_info = {}
        for case_name, case_dict in result.items():
            if not case_dict:
                continue

            keys = list(case_dict.keys())
            weights = np.array([len(case_dict[k]) for k in keys], dtype=float)
            probs = weights / weights.sum()

            self._case_sampling_info[case_name] = {
                "keys": keys,
                "probs": probs,
            }

        return result

    def sample_random_key(self, case: str = "Miami PIS"):
        """
        Sample a random key from a given case using the probabilities
        computed in identify_num_inspection_units_conditional.

        Returns:
            key (tuple) or None if no sampling info is available.
        """
        info = getattr(self, "_case_sampling_info", None)
        if not info or case not in info:
            return None

        keys = info[case]["keys"]
        probs = info[case]["probs"]

        # Make sure we have something to sample from
        if not keys:
            return None

        # Sample an index, not the tuples directly
        idx = np.random.choice(len(keys), p=probs)
        return keys[idx]

    def compute_rowcount_pmf_for_location(
            self,
            df: pd.DataFrame,
            location_name: str,
            total_units_col: str = 'SAMPLING_UNITS_FOR_INSPECTION_UNIT',
            loc_col: str = "INSPECTION_LOCATION_NAME",
            inspection_col: str = "INSPECTION_NUMBER",
            risk_unit_col: str = "RISK_UNIT",
            count_mode: str = "rows_per_inspection",
    ) -> pd.Series:
        """
        For a given inspection location:

        1. Filter df to that location.
        2. Drop rows where SAMPLING_UNITS is 0, blank, None, or NA.
           (We treat the column as numeric and keep rows with value > 0.)
        3. Compute counts based on `count_mode`:
           - "rows_per_inspection": number of rows per INSPECTION_NUMBER
           - "risk_units_per_inspection": number of unique risk units per INSPECTION_NUMBER
           - "rows_per_risk_unit": number of rows per risk unit within each INSPECTION_NUMBER
        4. Build an empirical PMF over these counts.

        Returns
        -------
        pmf : pd.Series
            Index: possible row counts (int)
            Values: probabilities (float, summing to 1).
        """
        # 1) subset by location
        df_loc = df[df[loc_col] == location_name].copy()

        if df_loc.empty:
            raise ValueError(f"No rows found for location '{location_name}'.")

        # 2) filter out rows with SAMPLING_UNITS <= 0 or non-numeric/blank
        #    This handles 0, blank, None, NA by coercing to numeric.
        total_units_numeric = pd.to_numeric(df_loc[total_units_col], errors="coerce")
        valid_mask = total_units_numeric > 0
        df_valid = df_loc[valid_mask]

        if df_valid.empty:
            raise ValueError(
                f"No valid rows (SAMPLING_UNITS > 0) for location '{location_name}'."
            )

        # 3) compute counts based on mode
        if count_mode == "rows_per_inspection":
            counts = df_valid.groupby(inspection_col).size()
        elif count_mode == "risk_units_per_inspection":
            if risk_unit_col not in df_valid.columns:
                raise ValueError(
                    f"Column '{risk_unit_col}' not found for risk unit counting."
                )
            df_ru = df_valid[df_valid[risk_unit_col].notna() & (df_valid[risk_unit_col] != "")]
            counts = df_ru.groupby(inspection_col)[risk_unit_col].nunique()
        elif count_mode == "rows_per_risk_unit":
            if risk_unit_col not in df_valid.columns:
                raise ValueError(
                    f"Column '{risk_unit_col}' not found for risk unit counting."
                )
            df_ru = df_valid[df_valid[risk_unit_col].notna() & (df_valid[risk_unit_col] != "")]
            counts = df_ru.groupby([inspection_col, risk_unit_col]).size()
        else:
            raise ValueError(
                "count_mode must be one of: "
                "'rows_per_inspection', 'risk_units_per_inspection', 'rows_per_risk_unit'."
            )

        if counts.empty:
            raise ValueError(
                f"No counts available for location '{location_name}' with count_mode='{count_mode}'."
            )

        # 4) empirical PMF over these counts
        #    value_counts gives how often each count occurs
        freq_by_count = counts.value_counts().sort_index()  # index = count
        pmf = freq_by_count / freq_by_count.sum()

        return pmf

    def sample_num_rows_from_location_pmf(
            self,
            df: pd.DataFrame,
            location_name: str,
            total_units_col: str = 'SAMPLING_UNITS_FOR_INSPECTION_UNIT',
            loc_col: str = "INSPECTION_LOCATION_NAME",
            inspection_col: str = "INSPECTION_NUMBER",
            risk_unit_col: str = "RISK_UNIT",
            count_mode: str = "rows_per_inspection",
    ) -> tuple[int, pd.Series]:
        """
        Convenience wrapper:

        - Computes the pmf via compute_rowcount_pmf_for_location
        - Samples a row-count according to that pmf.

        Returns
        -------
        sampled_count : int
            A sampled number of rows (e.g., 1, 2, 3, ...)
        pmf : pd.Series
            The pmf used for sampling (index = counts, values = probabilities).
        """
        pmf = self.compute_rowcount_pmf_for_location(
            df=df,
            location_name=location_name,
            total_units_col=total_units_col,
            loc_col=loc_col,
            inspection_col=inspection_col,
            risk_unit_col=risk_unit_col,
            count_mode=count_mode,
        )

        counts = pmf.index.to_numpy()
        probs = pmf.to_numpy()

        sampled_count = int(np.random.choice(counts, p=probs))
        return sampled_count

    def best_fit_discrete_distribution(self, data, candidate_dists=None):
        """
        Fit multiple SciPy *discrete* distributions and return:
            (best_dist_name, best_dist_obj, best_params, best_aic)

        If no candidate distribution fits, returns:
            (None, None, None, np.inf)
        """

        if candidate_dists is None:
            candidate_dists = {
                "poisson": stats.poisson,
                "nbinom": stats.nbinom,
                "geom": stats.geom,
            }

        data = np.asarray(data, dtype=int)

        best_dist = None
        best_name = None
        best_params = None
        best_aic = np.inf

        for name, dist in candidate_dists.items():
            try:
                # Fit distribution parameters by MLE
                params = dist.fit(data)

                # log-likelihood
                loglik = np.sum(dist.logpmf(data, *params))

                # number of parameters
                k = len(params)

                # AIC
                aic = 2 * k - 2 * loglik

                if np.isfinite(aic) and aic < best_aic:
                    best_aic = aic
                    best_dist = dist
                    best_name = name
                    best_params = params

            except Exception:
                # Some distributions may fail to fit; skip them
                continue

        return best_name, best_dist, best_params, best_aic

    def sample_positive_from_discrete_best_fit(
            self,
            df: pd.DataFrame,
            inspection_location_name: str,
            country_of_origin_name: str,
            propagative_material_type: str,
            total_units_col: str = 'SAMPLING_UNITS_FOR_INSPECTION_UNIT',
            loc_col: str = "INSPECTION_LOCATION_NAME",
            country_col: str = "COUNTRY_OF_ORIGIN_NAME",
            material_col: str = "PROPAGATIVE_MATERIAL_TYPE",
    ):
        """
        1. Subset df to given (location, country, material type).
        2. Extract strictly positive integer SAMPLING_UNITS values.
        3. Try to fit discrete distributions (Poisson, NB, Geom) by AIC.
        4. If all fails, fall back to empirical PMF over the observed support.
        5. Return a single sampled integer > 0 and info about what was used.

        Returns
        -------
        sample : int
        info : dict
        """

        # ---- 1) Filter df ----
        mask = (
                (df[loc_col] == inspection_location_name) &
                (df[country_col] == country_of_origin_name) &
                (df[material_col] == propagative_material_type)
        )
        df_sub = df.loc[mask]

        if df_sub.empty:
            raise ValueError(
                "No rows found for "
                f"{loc_col}={inspection_location_name!r}, "
                f"{country_col}={country_of_origin_name!r}, "
                f"{material_col}={propagative_material_type!r}."
            )

        # ---- 2) Coerce to integer, keep strictly positive ----
        vals = pd.to_numeric(df_sub[total_units_col], errors="coerce")
        vals = vals[vals > 0].dropna().astype(int)

        if vals.empty:
            return 1, {"dist_name": "one_due_to_no_data", "params": None}

        # ---- 3) Try discrete best fit ----
        best_name, best_dist, best_params, best_aic = self.best_fit_discrete_distribution(vals)

        # ---- 4) Empirical PMF fallback if no fit worked ----
        if best_dist is None:
            # empirical PMF over observed support
            counts = vals.value_counts(normalize=True).sort_index()
            support = counts.index.to_numpy()
            probs = counts.to_numpy()

            sample = int(np.random.choice(support, p=probs))
            info = {
                "dist_name": "empirical_pmf_fallback",
                "params": None,
                "aic": None,
                "n_obs": len(vals),
            }
            return sample, info

        # ---- 5) Sample from best discrete distribution ----
        sample = int(best_dist.rvs(*best_params))

        # Safety: ensure > 0; if not, fall back to empirical PMF
        if sample <= 0:
            counts = vals.value_counts(normalize=True).sort_index()
            support = counts.index.to_numpy()
            probs = counts.to_numpy()
            sample = int(np.random.choice(support, p=probs))
            info = {
                "dist_name": f"{best_name}_with_empirical_fallback",
                "params": best_params,
                "aic": best_aic,
                "n_obs": len(vals),
            }
        else:
            info = {
                "dist_name": best_name,
                "params": best_params,
                "aic": best_aic,
                "n_obs": len(vals),
            }

        return sample, info

    def multinomial_sample(self, df, columns, n_consignments=1, random_state=None):
        """Naive approach - sample each column independently"""
        np.random.seed(random_state)


        num_inspection_units = self.identify_num_inspection_units(self, df=df, n_consignments=n_consignments)

        sampled_df = self.sample_mixed_with_inspection(
            df=df,
            columns=columns,
            n_consignments=n_consignments,
            num_inspection_units=num_inspection_units,
            inspection_col="INSPECTION_NUMBER",
            random_state=42,
        )

        return sampled_df

    def sequential_multinomial_sample(self, df, columns, n_consignments=1, random_state=None):
        """Generate synthetic rows using sequential conditional multinomial sampling.

        The sampler builds each synthetic inspection in layers:
        1. Sample an inspection location.
        2. Sample how many risk units should appear in that inspection.
        3. For each risk unit, choose a conditional base subset
           (Miami vs Non-Miami logic), optionally pin a specific RISK_UNIT value,
           then sample how many rows that risk unit contributes.
        4. For each row, sample remaining columns sequentially, conditioning each
           next column on prior sampled values by filtering the working subset.

        Parameters
        ----------
        df : pd.DataFrame
            Source data used to estimate empirical distributions.
        columns : list[str]
            Requested output columns. `INSPECTION_NUMBER` is used internally and
            only retained in output if explicitly requested.
        n_consignments : int, default=1
            Number of synthetic inspections to generate.
        random_state : int | None, default=None
            Seed passed to NumPy random sampling for reproducibility.

        Returns
        -------
        pd.DataFrame
            Synthetic dataset with requested columns, preserving conditional
            structure from observed data where possible.
        """
        np.random.seed(random_state)
        risk_unit_col = "RISK_UNIT"

        # Build cases + probabilities
        num_inspection_units_conditional = self.identify_num_inspection_units_conditional(df=df)

        # Make sure INSPECTION_NUMBER is represented
        inspection_col = "INSPECTION_NUMBER"
        include_inspection_col_in_output = inspection_col in columns
        if inspection_col not in columns:
            columns = [inspection_col] + list(columns)

        risk_unit_number_col = "RISK_UNIT_NUMBER"
        include_risk_unit_number_in_output = risk_unit_number_col in columns
        if include_risk_unit_number_in_output and risk_unit_number_col not in columns:
            columns = [risk_unit_number_col] + list(columns)

        # Empirical location sampler P(location) from observed frequencies.
        def sample_location():
            values, counts = np.unique(df['INSPECTION_LOCATION_NAME'], return_counts=True)
            probs = counts / counts.sum()
            return np.random.choice(values, p=probs)

        # Sample a conditional key for a case, optionally constrained to one location.
        def pick_key(case, fixed_location=None):
            key = self.sample_random_key(case=case)
            if fixed_location is None:
                return key
            while key[0] != fixed_location:
                key = self.sample_random_key(case=case)
            return key

        # Build the base sample + subset that subsequent column sampling conditions on.
        def build_base(chosen_location, fixed_location):
            base_sample = {'INSPECTION_LOCATION_NAME': chosen_location}
            is_miami = chosen_location == 'Miami PIS'
            case_label = "Miami PIS" if is_miami else "Non-Miami PIS"

            cols_to_remove = [
                'INSPECTION_LOCATION_NAME',
                'COUNTRY_OF_ORIGIN_NAME',
                'PROPAGATIVE_MATERIAL_TYPE',
                'SAMPLING_UNITS_FOR_INSPECTION_UNIT',
            ]
            if is_miami:
                cols_to_remove.append('PRODUCER_NAME')

            key = pick_key(case=case_label, fixed_location=None if is_miami else fixed_location)
            if not is_miami and fixed_location is None:
                fixed_location = key[0]

            base_sample['INSPECTION_LOCATION_NAME'] = fixed_location if not is_miami else chosen_location
            base_sample['COUNTRY_OF_ORIGIN_NAME'] = key[0 if is_miami else 1]
            base_sample['PROPAGATIVE_MATERIAL_TYPE'] = key[1 if is_miami else 2]
            if is_miami:
                base_sample['PRODUCER_NAME'] = key[2]

            base_subset = num_inspection_units_conditional[case_label][key]
            return base_sample, base_subset, cols_to_remove, fixed_location

        # Create synthetic inspection IDs: INS_0, INS_1, ...
        inspection_ids = [f"INS_{i}" for i in range(n_consignments)]

        samples = []
        for ins_id in inspection_ids:
            # 1) Sample top-level inspection location.
            chosen_location = sample_location()

            # 2) Sample number of risk units in this inspection from location-specific PMF.
            num_risk_units = self.sample_num_rows_from_location_pmf(
                df,
                location_name=chosen_location,
                total_units_col="SAMPLING_UNITS_FOR_INSPECTION_UNIT",
                count_mode="risk_units_per_inspection",
                risk_unit_col="RISK_UNIT",
            )

            fixed_pis_location = None

            # 3) Generate each risk unit under this inspection.
            for risk_unit_number in range(num_risk_units):
                base_sample, base_subset, cols_to_remove, fixed_pis_location = build_base(
                    chosen_location=chosen_location,
                    fixed_location=fixed_pis_location,
                )

                # Assign a synthetic risk unit number per inspection
                if include_risk_unit_number_in_output:
                    base_sample[risk_unit_number_col] = f"RU_{ins_id}_{risk_unit_number}"
                    cols_to_remove = cols_to_remove + [risk_unit_number_col]

                # Set a consistent risk unit value for this risk unit
                if risk_unit_col in df.columns:
                    cols_to_remove = cols_to_remove + [risk_unit_col]
                    ru_values = base_subset[risk_unit_col].dropna().unique()
                    if len(ru_values) > 0:
                        risk_unit_value = np.random.choice(ru_values)
                        base_sample[risk_unit_col] = risk_unit_value
                        base_subset = base_subset[base_subset[risk_unit_col] == risk_unit_value]

                # Sample number of rows for this risk unit
                num_rows_for_risk_unit = self.sample_num_rows_from_location_pmf(
                    df,
                    location_name=chosen_location,
                    total_units_col="SAMPLING_UNITS_FOR_INSPECTION_UNIT",
                    count_mode="rows_per_risk_unit",
                    risk_unit_col="RISK_UNIT",
                )

                # 4) Generate each row belonging to this risk unit.
                for row_id in range(num_rows_for_risk_unit):
                    subset = base_subset
                    sample = dict(base_sample)

                    # Get a random number of sampling units based on what has already been populated by the sample
                    # (i.e., the PIS station, Origin, and PM Type)
                    df_for_sampling = base_subset if not base_subset.empty else df
                    num_sample_units, info = self.sample_positive_from_discrete_best_fit(
                        df=df_for_sampling,
                        inspection_location_name=sample['INSPECTION_LOCATION_NAME'],
                        country_of_origin_name=sample['COUNTRY_OF_ORIGIN_NAME'],
                        propagative_material_type=sample['PROPAGATIVE_MATERIAL_TYPE'],
                        total_units_col="SAMPLING_UNITS_FOR_INSPECTION_UNIT",
                    )

                    sample['SAMPLING_UNITS_FOR_INSPECTION_UNIT'] = num_sample_units



                    for col in columns:
                        # Do not multinomial-sample INSPECTION_NUMBER; we set it explicitly
                        if col == inspection_col or col in cols_to_remove:
                            continue

                        values, counts = np.unique(subset[col], return_counts=True)
                        probs = counts / counts.sum()
                        sampled_value = np.random.choice(values, p=probs)
                        sample[col] = sampled_value

                        # Condition on this choice for subsequent columns
                        subset = subset[subset[col] == sampled_value]

                        if subset.empty:
                            # If no rows left, sample remaining columns from original df marginals
                            for rem_col in columns:
                                if rem_col in sample or rem_col == inspection_col:
                                    continue
                                values, counts = np.unique(df[rem_col], return_counts=True)
                                probs = counts / counts.sum()
                                sample[rem_col] = np.random.choice(values, p=probs)
                            break

                    # Now set inspection and row ID for this row
                    sample[inspection_col] = ins_id
                    sample['Row_ID'] = f"{ins_id}_{row_id}"

                    samples.append(sample)

        sampled_df = pd.DataFrame(samples)

        return sampled_df

    def gmm_sample(self, df, columns, n_consignments=1, random_state=None, n_components=3):
        """
        GMM sampling for numeric columns + Gaussian-copula-based sampling
        for categorical columns, with grouped INSPECTION_NUMBER values.

        - Computes num_inspection_units via self.identify_num_inspection_units(df, n_consignments)
        - Output:
            sum(num_inspection_units) rows
            n_consignments unique INSPECTION_NUMBER values
        """

        rng = np.random.default_rng(random_state)
        inspection_col = "INSPECTION_NUMBER"

        # Ensure INSPECTION_NUMBER is part of columns
        if inspection_col not in columns:
            columns = [inspection_col] + list(columns)

        # ----- 1. Figure out how many rows to generate per inspection -----
        num_inspection_units = self.identify_num_inspection_units(
            df=df,
            n_consignments=n_consignments,
        )
        num_inspection_units = np.asarray(num_inspection_units, dtype=int)

        if len(num_inspection_units) != n_consignments:
            raise ValueError(
                "identify_num_inspection_units must return an array of length n_consignments"
            )

        total_rows = int(num_inspection_units.sum())

        # Create synthetic inspection IDs; change format if you want real IDs
        inspection_ids = np.array([f"INS_{i}" for i in range(n_consignments)])
        inspection_values = np.repeat(inspection_ids, num_inspection_units)
        if len(inspection_values) != total_rows:
            raise ValueError("Sum of num_inspection_units must equal total_rows.")

        # ----- 2. Split columns into numeric vs categorical -----
        numeric_cols = [
            c for c in columns
            if c != inspection_col and np.issubdtype(df[c].dtype, np.number)
        ]
        cat_cols = [c for c in columns if c not in numeric_cols]

        # We will *not* sample INSPECTION_NUMBER via copula; we assign it manually
        cat_cols_no_ins = [c for c in cat_cols if c != inspection_col]

        # ----- 3. Sample categorical columns via Gaussian copula sampler -----
        if cat_cols_no_ins:
            # gaussian_copula_sample returns a DataFrame with these columns
            sampled_cat = self.gaussian_copula_sample(
                df=df,
                columns=cat_cols_no_ins,
                n_samples=total_rows,
                random_state=random_state,
            )
        else:
            sampled_cat = pd.DataFrame(index=range(total_rows))

        # Add INSPECTION_NUMBER column explicitly
        sampled_cat[inspection_col] = inspection_values

        # ----- 4. Fit and sample GMM for numeric columns -----
        if numeric_cols:
            gmm = GaussianMixture(
                n_components=n_components,
                random_state=random_state
            )
            gmm.fit(df[numeric_cols])

            sampled_numeric, _ = gmm.sample(total_rows)
            sampled_numeric = pd.DataFrame(sampled_numeric, columns=numeric_cols)

            # Optional: round and clip like original
            for col in numeric_cols:
                # If your numeric cols are not all integer-like, you can relax this
                sampled_numeric[col] = np.round(sampled_numeric[col]).astype(int)
                sampled_numeric[col] = sampled_numeric[col].clip(
                    df[col].min(),
                    df[col].max()
                )
        else:
            sampled_numeric = pd.DataFrame(index=range(total_rows))

        # ----- 5. Combine categorical + numeric samples -----
        if not sampled_cat.empty and not sampled_numeric.empty:
            sampled = pd.concat(
                [sampled_cat.reset_index(drop=True),
                 sampled_numeric.reset_index(drop=True)],
                axis=1,
            )
        elif not sampled_cat.empty:
            sampled = sampled_cat
        else:
            sampled = sampled_numeric

        # Enforce requested column order
        sampled = sampled[[c for c in columns if c in sampled.columns]]

        return sampled
    
    def _make_psd(self, S):
        """Clip tiny negative eigenvalues to ensure PSD."""
        w, V = np.linalg.eigh(S)
        w = np.maximum(w, 1e-8)
        return (V * w) @ V.T
    
    def _rank_to_z(self, mat):
        """Empirical CDF per column -> latent normal; expects a 2D numpy array."""
        eps = 1e-6
        n, m = mat.shape
        Z = np.empty_like(mat, dtype=float)
        for j in range(m):
            x = mat[:, j]
            ranks = pd.Series(x).rank(method='average').to_numpy()
            u = (ranks - 0.5) / n
            u = np.clip(u, eps, 1 - eps)
            Z[:, j] = norm.ppf(u)
        return Z
    
    def gaussian_copula_sample(self, df, columns, n_samples=1, random_state=None, min_group_corr_rows=20):
        """
        Category-conditional Gaussian copula sampler that preserves P(C) and P(X|C).
        """
        rng = np.random.default_rng(random_state)
        X = df[columns].copy()
        
        # Split columns
        numeric_cols = [c for c in columns if np.issubdtype(X[c].dtype, np.number)]
        cat_cols = [c for c in columns if c not in numeric_cols]
        m = len(numeric_cols)
        
        # If no categoricals, revert to a plain Gaussian copula on numeric
        if len(cat_cols) == 0:
            if m == 0:
                return pd.DataFrame(index=range(n_samples), columns=columns)
            
            num_complete = X[numeric_cols].dropna()
            if len(num_complete) < 2:
                raise ValueError("Not enough non-missing numeric rows to fit copula.")
            
            Z = self._rank_to_z(num_complete.to_numpy())
            Sigma_global = self._make_psd(np.corrcoef(Z, rowvar=False))
            Z_samp = rng.multivariate_normal(np.zeros(m), Sigma_global, size=n_samples)
            U = norm.cdf(Z_samp)
            
            out = pd.DataFrame(index=range(n_samples), columns=columns)
            for j, c in enumerate(numeric_cols):
                vals = np.sort(X[c].dropna().to_numpy())
                if len(vals) == 0:
                    out[c] = np.nan
                    continue
                idx = np.floor(U[:, j] * len(vals)).astype(int).clip(0, len(vals)-1)
                x = vals[idx]
                if np.issubdtype(X[c].dtype, np.integer):
                    x = np.round(x).astype(int)
                out[c] = x
            return out[columns]
        
        # Handle missing categories
        C = X[cat_cols].astype("object").fillna("__NA__")
        
        # Empirical joint over C
        counts = C.value_counts(dropna=False, sort=False)
        keys = counts.index.tolist()
        probs = (counts / counts.sum()).to_numpy()
        
        # Precompute global numeric marginals + global correlation as fallback
        global_sorted = {}
        for c in numeric_cols:
            vals = np.sort(X[c].dropna().to_numpy())
            global_sorted[c] = vals
        
        if m >= 2:
            num_complete = X[numeric_cols].dropna()
            if len(num_complete) >= 2:
                Zg = self._rank_to_z(num_complete.to_numpy())
                Sigma_global = self._make_psd(np.corrcoef(Zg, rowvar=False))
            else:
                Sigma_global = np.eye(m)
        elif m == 1:
            Sigma_global = np.array([[1.0]])
        else:
            Sigma_global = None
        
        # Build per-group info
        group_info = {}
        X_aug = X.copy()
        for c in cat_cols:
            X_aug[c] = C[c]
        
        grouped = X_aug.groupby(cat_cols, sort=False, dropna=False)
        for key, gdf in grouped:
            key = key if isinstance(key, tuple) else (key,)
            
            info = {}
            
            # Group-specific numeric marginals
            marginals = {}
            intlike = {}
            gmins, gmaxs = {}, {}
            
            for c in numeric_cols:
                vals = np.sort(gdf[c].dropna().to_numpy())
                marginals[c] = vals
                intlike[c] = np.issubdtype(X[c].dtype, np.integer)
                if len(vals) > 0:
                    gmins[c], gmaxs[c] = vals[0], vals[-1]
                else:
                    g = global_sorted[c]
                    gmins[c] = g[0] if len(g) else np.nan
                    gmaxs[c] = g[-1] if len(g) else np.nan
            
            info["marginals"] = marginals
            info["intlike"] = intlike
            info["mins"] = gmins
            info["maxs"] = gmaxs
            
            # Group-specific numeric copula correlation
            if m >= 2:
                num_complete_g = gdf[numeric_cols].dropna()
                if len(num_complete_g) >= max(2, min_group_corr_rows):
                    Zg = self._rank_to_z(num_complete_g.to_numpy())
                    Sg = np.corrcoef(Zg, rowvar=False)
                    if np.all(np.isfinite(Sg)):
                        info["Sigma"] = self._make_psd(Sg)
                    else:
                        info["Sigma"] = Sigma_global
                else:
                    info["Sigma"] = Sigma_global
            elif m == 1:
                info["Sigma"] = np.array([[1.0]])
            else:
                info["Sigma"] = None
            
            group_info[key] = info
        
        # Sampling
        out = pd.DataFrame(index=range(n_samples), columns=columns)
        
        # Sample category choices
        key_indices = np.arange(len(keys))
        chosen_idx = rng.choice(key_indices, size=n_samples, p=probs)
        chosen_keys = [keys[i] for i in chosen_idx]
        
        # Fill categorical columns
        for i, key in enumerate(chosen_keys):
            key = key if isinstance(key, tuple) else (key,)
            for col, val in zip(cat_cols, key):
                out.at[i, col] = (np.nan if val == "__NA__" else val)
        
        # Fill numeric columns conditional on chosen categories
        if m > 0:
            for i, key in enumerate(chosen_keys):
                key = key if isinstance(key, tuple) else (key,)
                info = group_info[key]
                Sigma = info["Sigma"]
                
                # Draw latent Z and convert to U
                z = rng.multivariate_normal(np.zeros(m), Sigma)
                u = norm.cdf(z)
                
                # Map through group-specific inverse CDFs
                for j, c in enumerate(numeric_cols):
                    vals = info["marginals"][c]
                    if len(vals) == 0:
                        vals = global_sorted[c]
                    if len(vals) == 0:
                        out.at[i, c] = np.nan
                        continue
                    
                    idx = int(np.floor(u[j] * len(vals)))
                    if idx == len(vals):
                        idx -= 1
                    x = vals[idx]
                    
                    if info["intlike"][c]:
                        x = int(np.round(x))
                        x = int(np.clip(x, info["mins"][c], info["maxs"][c]))
                    else:
                        x = float(np.clip(x, info["mins"][c], info["maxs"][c]))
                    out.at[i, c] = x
        
        return out[columns]
    
    def generate_from_input_data(self, n_consignments=1000, sampling_method=None):
        """Generate synthetic data based on input data file using specified sampling method
        
        :param n_consignments: Number of unique consignments/shipments to generate
        :param sampling_method: Sampling method to use (naive, sequential, gmm, gaussian_copula)
        :return: DataFrame with synthetic data
        """
        if self.input_data is None or len(self.input_data) == 0:
            raise ValueError("No usable input data loaded. Please provide a non-empty input_data_file.")
        
        method = sampling_method
        # Define columns to use for sampling
        available_cols = self.input_data.columns.tolist()
        target_cols = [col for col in available_cols if col in [
            'INSPECTION_NUMBER',
            'COMMODITY_COMMON_NAME',
            'COUNTRY_OF_ORIGIN_NAME',
            'PRODUCER_NAME',
            'PROPAGATIVE_MATERIAL_TYPE',
            'QUANTITY',
            'BROKER_NAME',
            'INSPECTION_LOCATION_NAME',
            'PATHWAY',
            'SHIPPER_NAME',
            'TAXONOMY_ORDER',
            'TAXONOMY_FAMILY',
            'TAXONOMY_GENUS',
            'TAXONOMY_SPECIES',
            'action',
            'RISK_UNIT',
            'TOTAL_SAMPLING_UNITS_FOR_RISK_UNIT',
            'SAMPLING_UNITS_FOR_INSPECTION_UNIT',
            'REQUIRED_NUMBER_OF_BOXES',
        ]]
        
        if not target_cols:
            # Fallback to all available columns
            target_cols = available_cols
        
        # print(f"Using sampling method: {method}")
        # print(f"Sampling columns: {target_cols}")
        
        if method == "naive":
            synthetic_data = self.multinomial_sample(
                self.input_data, target_cols, n_consignments, random_state=42
            )
        elif method == "sequential":
            synthetic_data = self.sequential_multinomial_sample(
                self.input_data, target_cols, n_consignments, random_state=42
            )
        elif method == "gmm": # Experimental
            synthetic_data = self.gmm_sample(
                self.input_data, target_cols, n_consignments, random_state=42
            )
        elif method == "gaussian_copula":
            synthetic_data = self.gaussian_copula_sample(
                self.input_data, target_cols, n_consignments, random_state=42
            )
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        return synthetic_data
    
   
    def calculate_quality_metrics(self, original_df, synthetic_df):
        """Calculate quality metrics comparing original and synthetic data
        
        :param original_df: Original DataFrame for comparison
        :param synthetic_df: Synthetic DataFrame to evaluate
        :return: Dictionary of quality metrics
        """
        if original_df is None:
            return {}
        
        metrics = {}
        
        # Get common columns
        common_cols = set(original_df.columns) & set(synthetic_df.columns)
        
        for col in common_cols:
            if col in original_df.columns and col in synthetic_df.columns:
                orig_vals = original_df[col].dropna()
                synth_vals = synthetic_df[col].dropna()
                
                if len(orig_vals) > 0 and len(synth_vals) > 0:
                    if np.issubdtype(orig_vals.dtype, np.number):
                        # For numeric columns, use Wasserstein distance
                        distance = wasserstein_distance(
                            orig_vals.astype(float).to_numpy(),
                            synth_vals.astype(float).to_numpy()
                        )
                    else:
                        # For categorical columns, use alphabetical ordering
                        # Get unique categories from both datasets
                        all_vals = pd.concat([
                            orig_vals.astype(str),
                            synth_vals.astype(str)
                        ])
                        unique_vals = all_vals.drop_duplicates().values
                        cats = pd.Index(unique_vals).sort_values()
                        
                        enc = lambda s: pd.Categorical(s.astype(str), categories=cats, ordered=True).codes
                        oa, sa = map(enc, (orig_vals, synth_vals))
                        
                        va, vs = map(pd.Series, (oa, sa))
                        orig_support, orig_weights = va.value_counts().sort_index().index.to_numpy(), va.value_counts().sort_index().values.astype(float)
                        synth_support, synth_weights = vs.value_counts().sort_index().index.to_numpy(), vs.value_counts().sort_index().values.astype(float)
                        
                        distance = wasserstein_distance(
                            orig_support, synth_support, 
                            u_weights=orig_weights, v_weights=synth_weights
                        )
                    
                    metrics[f"wasserstein_{col}"] = distance
        
        return metrics
    
    def generate_statistics(self, dataset):
        """Generate statistics about the synthetic dataset
        
        :param dataset: DataFrame to analyze
        :return: Dictionary of dataset statistics
        """
        if dataset.empty:
            return {}
        
        stats = {
            "total_records": len(dataset),
            "columns": list(dataset.columns),
            "numeric_columns": [col for col in dataset.columns if np.issubdtype(dataset[col].dtype, np.number)],
            "categorical_columns": [col for col in dataset.columns if not np.issubdtype(dataset[col].dtype, np.number)],
        }
        
        # Add unique value counts for categorical columns
        for col in stats["categorical_columns"]:
            stats[f"unique_{col}"] = dataset[col].nunique()
        
        # Add basic statistics for numeric columns
        for col in stats["numeric_columns"]:
            stats[f"{col}_mean"] = dataset[col].mean()
            stats[f"{col}_std"] = dataset[col].std()
            stats[f"{col}_min"] = dataset[col].min()
            stats[f"{col}_max"] = dataset[col].max()
        
        return stats


def save_to_csv(dataset, filename):
    """Save dataset to CSV file
    
    :param dataset: DataFrame to save
    :param filename: Output CSV filename
    """
    dataset.to_csv(filename, index=False)
    print(f"Saved {len(dataset)} records to {filename}")

def save_to_json(dataset, filename):
    """Save dataset to JSON file
    
    :param dataset: DataFrame to save
    :param filename: Output JSON filename
    """
    dataset.to_json(filename, orient='records', indent=2)
    print(f"Saved {len(dataset)} records to {filename}")
