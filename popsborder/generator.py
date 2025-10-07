# Simulation of contaminated consignments and their inspections
# Copyright (C) 2018-2025 Vaclav Petras and others (see below)

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


"""Synthetic consignment data generation

Contributors: Gary Lin, Joseph Agor (Johns Hopkins University Applied Physics Laboratory)

New Classes Added:
------------------
- SyntheticConsignmentDataGenerator:
    * Generates synthetic consignment data for testing and simulation purposes
    * Creates realistic consignment records with randomized attributes
    * Uses advanced sampling techniques including Gaussian copulas
    * Supports multiple sampling methods: naive, sequential, GMM, gaussian_copula
    * Integrates with real PIS data for training synthetic data generation

Sampling Methods Implemented:
----------------------------
- multinomial_sample(): Naive approach sampling each column independently
- sequential_multinomial_sample(): Sequential sampling preserving conditional dependencies
- gmm_sample(): Gaussian Mixture Model sampling for numeric columns
- gaussian_copula_sample(): Category-conditional Gaussian copula preserving correlations

Data Generation Features:
------------------------
- Configurable consignment attributes (origins, ports, pathways, commodities)
- Propagative material and flower commodity support
- Contamination modeling with configurable probability and quantities
- Quality metrics calculation comparing original and synthetic data
- Multiple output formats (CSV, JSON) with comprehensive statistics
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


class SyntheticConsignmentDataGenerator:
    """Generate synthetic consignment data using advanced sampling techniques
    
    This generator creates realistic consignment records with randomized attributes
    based on configurable parameters or input data files. It supports multiple
    sampling methods for preserving statistical relationships in the data.
    """
    
    def __init__(self, input_data_file=None):
        """Initialize the synthetic data generator
        
        :param input_data_file: Optional path to input data file for training
        """

        self.input_data = None
        if input_data_file:
            self.input_data = self._load_input_data(input_data_file)
        
        # Initialize random seed for reproducible results
        random.seed(42)
        np.random.seed(42)
    
    def _load_input_data(self, input_file):
        """Load input data file for training sampling models
        
        :param input_file: Path to input data file
        :return: DataFrame with loaded data or None if failed
        """
        try:
            df = pd.read_csv(input_file)
            df = df.dropna()
            print(f"Loaded {len(df)} records from {input_file}")
            return df
        except Exception as e:
            print(f"Error loading input data file: {e}")
            return None
    
    def multinomial_sample(self, df, columns, n_samples=1, random_state=None):
        """Naive approach - sample each column independently"""
        np.random.seed(random_state)
        sampled = {}
        for col in columns:
            values, counts = np.unique(df[col], return_counts=True)
            probs = counts / counts.sum()
            sampled[col] = np.random.choice(values, size=n_samples, p=probs)
        return pd.DataFrame(sampled)
    
    def sequential_multinomial_sample(self, df, columns, n_samples=1, random_state=None):
        """Sequential sampling to preserve conditional dependencies"""
        np.random.seed(random_state)
        samples = []
        for _ in range(n_samples):
            subset = df
            sample = {}
            for col in columns:
                values, counts = np.unique(subset[col], return_counts=True)
                probs = counts / counts.sum()
                chosen = np.random.choice(values, p=probs)
                sample[col] = chosen
                subset = subset[subset[col] == chosen]
                if subset.empty:
                    # If no rows left, sample from original df for remaining columns
                    for rem_col in columns[len(sample):]:
                        values, counts = np.unique(df[rem_col], return_counts=True)
                        probs = counts / counts.sum()
                        sample[rem_col] = np.random.choice(values, p=probs)
                    break
            samples.append(sample)
        return pd.DataFrame(samples)
    
    def gmm_sample(self, df, columns, n_samples=1, random_state=None, n_components=3):
        """Gaussian Mixture Model sampling for numeric columns with sequential for categorical"""
        numeric_cols = [col for col in columns if np.issubdtype(df[col].dtype, np.number)]
        cat_cols = [col for col in columns if col not in numeric_cols]
        
        # Fit GMM on numeric columns
        if numeric_cols:
            gmm = GaussianMixture(n_components=n_components, random_state=random_state)
            gmm.fit(df[numeric_cols])
            sampled_numeric, _ = gmm.sample(n_samples)
            sampled_numeric = pd.DataFrame(sampled_numeric, columns=numeric_cols)
            
            # Round and clip numeric columns
            for col in numeric_cols:
                sampled_numeric[col] = np.round(sampled_numeric[col]).astype(int)
                sampled_numeric[col] = sampled_numeric[col].clip(df[col].min(), df[col].max())
        else:
            sampled_numeric = pd.DataFrame()
        
        # Sequential sampling for categorical columns
        if cat_cols:
            sampled_cat = self.sequential_multinomial_sample(
                df, cat_cols, n_samples, random_state
            )
        else:
            sampled_cat = pd.DataFrame(index=range(n_samples))
        
        # Combine results
        if not sampled_numeric.empty and not sampled_cat.empty:
            sampled = pd.concat([sampled_cat, sampled_numeric], axis=1)
        elif not sampled_numeric.empty:
            sampled = sampled_numeric
        else:
            sampled = sampled_cat
        
        return sampled[columns] if columns else sampled
    
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
    
    def generate_from_input_data(self, n_samples=1000, sampling_method=None):
        """Generate synthetic data based on input data file using specified sampling method
        
        :param n_samples: Number of synthetic samples to generate
        :param sampling_method: Sampling method to use (naive, sequential, gmm, gaussian_copula)
        :return: DataFrame with synthetic data
        """
        if self.input_data is None:
            raise ValueError("No input data loaded. Please provide input_data_file parameter.")
        
        method = sampling_method
        # Define columns to use for sampling
        available_cols = self.input_data.columns.tolist()
        target_cols = [col for col in available_cols if col in [
            'INSPECTION_NUMBER', 'INSPECTION_LOCATION_NAME', 'PATHWAY',
            'COUNTRY_OF_ORIGIN_NAME', 'PROPAGATIVE_MATERIAL_TYPE',
            'TOTAL_SAMPLING_UNITS', 'TOTAL_PLANT_QUANTITY', 'PRODUCER'
        ]]
        
        if not target_cols:
            # Fallback to all available columns
            target_cols = available_cols
        
        print(f"Using sampling method: {method}")
        print(f"Sampling columns: {target_cols}")
        
        if method == "naive":
            synthetic_data = self.multinomial_sample(
                self.input_data, target_cols, n_samples, random_state=42
            )
        elif method == "sequential":
            synthetic_data = self.sequential_multinomial_sample(
                self.input_data, target_cols, n_samples, random_state=42
            )
        elif method == "gmm":
            synthetic_data = self.gmm_sample(
                self.input_data, target_cols, n_samples, random_state=42
            )
        elif method == "gaussian_copula":
            synthetic_data = self.gaussian_copula_sample(
                self.input_data, target_cols, n_samples, random_state=42
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