# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

import os
import re
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.optimize import minimize_scalar
from scipy import stats
import pandas as pd
from collections import defaultdict




from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional


@dataclass
class ClarkeModelInputs:
    # required by the BB model
    ty: List[int]                  # unique counts of groups testing positive
    freq: List[int]                # frequency per ty
    b: int                         # groups per consignment (boxes per inspection)
    B: int                         # number of consignments (inspections)
    Nbar: int                      # items per group
    theta: float                   # clustering hyperparameter
    lambda_test: float             # extra param you track
    R: int                         # MC / sensitivity runs
    start_val: Tuple[float, float]  # optimizer start (log-alpha, log-beta)
    se: bool                       # request standard errors

    def to_kwargs(self) -> dict:
        """Convenience: kwargs for your BB model call (omit DataFrame)."""
        d = asdict(self)
        d.pop("action_summary", None)
        return d

    @staticmethod
    def default() -> 'ClarkeModelInputs':
        return ClarkeModelInputs(
            ty=[],
            freq=[],
            b=25,
            B=100,
            Nbar=200,
            theta=np.inf,
            lambda_test=1.0,
            R=1000,
            start_val=(0.0, 0.0),
            se=False,
        )


def remove_outliers_iqr(df, cols, factor=1.5):
    clean_df = df.copy()
    for col in cols:
        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
    return clean_df

def group_by_inspection_id(
        df_pis_data: pd.DataFrame,
        df_rbs_calculator: pd.DataFrame,
):
    """
    Function that will group the data by inspection ids and, if not already done,
    will filter to those shared by the two data sets.

    It will return two dictionaries where each key represents an inspection id and
    the value is the dataframe associated with that inspection id.
    """

    inspection_id_grouped_pis_data_orig = {k: v for k, v in df_pis_data.groupby("INSPECTION_ID")}

    inspection_id_grouped_rbs_data_orig = {k: v for k, v in df_rbs_calculator.groupby("INSPECTION_ID")}

    # Find intersection of inspection ids
    shared_inspection_ids = set(inspection_id_grouped_pis_data_orig.keys()) & set(
        inspection_id_grouped_rbs_data_orig.keys())

    # Filter both dicts to only shared keys
    inspection_id_grouped_pis_data = {k: inspection_id_grouped_pis_data_orig[k] for k in shared_inspection_ids}
    inspection_id_grouped_rbs_data = {k: inspection_id_grouped_rbs_data_orig[k] for k in shared_inspection_ids}

    return inspection_id_grouped_pis_data, inspection_id_grouped_rbs_data

def identify_relevant_consignments(
        df_pis_data: pd.DataFrame,
        df_rbs_calculator: pd.DataFrame,
):
    """
    Function that will filter the PIS and Calculator data to be only those relevant to RBS.
    Namely, if the variable IS_RBS = 1 AND RBS_STATUS != 'Not RBS' as indicated
    in the PIS data.

    Function also find the consignments/inspection IDs that are shared across both the PIS and calculator data
    """
    print(f"\nIdentifying Relevant Consignments for Use in Contamination Parameterization Process")
    print(f'Conditions Include:')
    print(f"   Within PIS Data, the 'IS_RBS' variable needs to be 1 (indicating that inspection ID comes fro an RBS process)")
    print(
        f"   Within PIS Data, removes any commodity line where the 'RBS_STATUS' variable is 'Not RBS' "
        f"(indicating that inspection ID does not comes from an RBS process)\n")
    # Count how many inspection IDs in PIS data are IS_RBS = 1 and RBS_STATS != 'Not RBS'
    pis_data_rbs_only= df_pis_data[df_pis_data.IS_RBS == 1]
    print(f'   After remove of IS_RBS == 1 ONLY:')
    print(f'       # of Rows: {pis_data_rbs_only.shape[0]}')
    print(f'       # of Rows Removed: {df_pis_data.shape[0]-pis_data_rbs_only.shape[0]}')
    print(f'       % of Rows Removed: {round(((df_pis_data.shape[0]-pis_data_rbs_only.shape[0])/df_pis_data.shape[0])*100,2)}%')
    pis_data_rbs_only2 = pis_data_rbs_only[pis_data_rbs_only.RBS_STATUS != 'Not RBS']
    print(f'   Size after remove of RBS_STATUS != Not RBS:')
    print(f'       # of Rows: {pis_data_rbs_only2.shape[0]}')
    print(f'       # of Rows Removed: {pis_data_rbs_only2.shape[0]-pis_data_rbs_only2.shape[0]}')
    print(f'       % of Rows Removed: {round(((pis_data_rbs_only2.shape[0]-pis_data_rbs_only2.shape[0])/pis_data_rbs_only2.shape[0])*100,2)}%\n')

    # Find intersection of inspection ids
    shared_inspection_ids = set(pis_data_rbs_only2.INSPECTION_ID.unique()) & set(df_rbs_calculator.INSPECTION_ID.unique())

    # Filter the data to only intersected and relevant values
    df_rbs_calculator_filtered = df_rbs_calculator[df_rbs_calculator.INSPECTION_ID.isin(shared_inspection_ids)]
    df_pis_data_filtered = df_pis_data[df_pis_data.INSPECTION_ID.isin(shared_inspection_ids)]

    print(f"   Number of shared inspection IDs (between RBS Calculator and PIS Data after removal): {len(shared_inspection_ids)}\n")

    return df_pis_data_filtered, df_rbs_calculator_filtered


def _get_b_B_nbar_inputs_old(
        df_pis_data_filtered: pd.DataFrame
):
    print(f"   Determining Inputs 'b', 'B' and 'Nbar'")
    # Function to generate b, B, and Nbar input parameters
    # Get applicable and interesected inspection ids
    df_pis_data_grouped = {k: v for k, v in df_pis_data_filtered.groupby("INSPECTION_ID")}

    # Check to ensure if column name for "required number of boxes" exists
    req_boxes_col = None
    for candidate in ["REQUIRED_NUMBER_OF_BOXES", "REQUIRED_NUMBER_OF_BOXES"]:
        if candidate in next(iter(df_pis_data_grouped.values())).columns:
            req_boxes_col = candidate
            break

    if req_boxes_col is None:
        raise KeyError("Could not find REQUIRED_NUMBER_OF_BOXES in supplied data set.")

    # Go through each of the different sets of the inspection ids and calculate the parameters
    # averaging if there are multiple records per inspection id in the idealized rbs data set
    per_inspection = {}
    for insp_id, df in df_pis_data_grouped.items():
        # Option 1: Aggregate at consignment level
        # Consignment aggregation
        #

        # Option 2
        # Ensure numeric, ignore NaNs in means
        req_boxes = pd.to_numeric(df[req_boxes_col], errors="coerce")
        total_sample_units = pd.to_numeric(df["TOTAL_SAMPLING_UNITS"], errors="coerce")
        total_plants = pd.to_numeric(df["QUANTITY"], errors="coerce")

        # b_i = avg required boxes for this inspection_id
        b_i = req_boxes.mean()

        # B_i = avg total sampling units for this inspection_id
        B_i = total_sample_units.mean()

        # Nbar_i = avg over rows of (QUANTITY / TOTAL_SAMPLING_UNITS) for this inspection_id
        with np.errstate(divide="ignore", invalid="ignore"):
            nbar_rows = total_plants / total_sample_units
        nbar_rows = nbar_rows.replace([np.inf, -np.inf], np.nan)
        Nbar_i = nbar_rows.mean()

        per_inspection[insp_id] = {"b": b_i, "B": B_i, "Nbar": Nbar_i}

    # Convert to a tidy dataframe (index = INSPECTION_ID)
    per_inspection_df = pd.DataFrame.from_dict(per_inspection, orient="index")

    # Remove outliers from per_inspection_df
    per_inspection_df_no_outliers = remove_outliers_iqr(per_inspection_df, ["b", "B", "Nbar"])
    per_inspection_df = per_inspection_df_no_outliers.copy()

    # Overall (unweighted) averages across inspection IDs
    b = per_inspection_df["b"].mean()
    B = per_inspection_df["B"].mean()
    Nbar = per_inspection_df["Nbar"].mean()

    print("      Final inputs (averaged across inspection IDs):")
    print(f"         b = {round(b,0)}")
    print(f"         B = {round(B,0)}")
    print(f"         Nbar = {round(Nbar,0)}\n")

    return round(b,0), round(B,0), round(Nbar,0)

def _get_b_B_nbar_inputs(df_pis_data_filtered: pd.DataFrame):
    print("   Determining Inputs 'b', 'B' and 'Nbar'")

    # ---- Required columns ----
    required_cols = [
        "INSPECTION_ID",
        "RISK_UNIT",
        "action",
        "TOTAL_SAMPLING_UNITS_FOR_RISK_UNIT",
        "REQUIRED_NUMBER_OF_BOXES",
        "QUANTITY",
    ]
    missing = [c for c in required_cols if c not in df_pis_data_filtered.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Ensure numeric where needed
    df = df_pis_data_filtered.copy()
    df["TOTAL_SAMPLING_UNITS_FOR_RISK_UNIT"] = pd.to_numeric(
        df["TOTAL_SAMPLING_UNITS_FOR_RISK_UNIT"], errors="coerce"
    )
    df["REQUIRED_NUMBER_OF_BOXES"] = pd.to_numeric(
        df["REQUIRED_NUMBER_OF_BOXES"], errors="coerce"
    )
    df["QUANTITY"] = pd.to_numeric(df["QUANTITY"], errors="coerce")

    per_inspection = {}

    # Group by INSPECTION_ID-risk_unit
    num_groups = df.groupby(["INSPECTION_ID", "RISK_UNIT"]).ngroups
    count = 1
    for (insp_id, ru), g in df.groupby(["INSPECTION_ID", "RISK_UNIT"], sort=False):
        if count == int(round(0.1*num_groups,0)):
            print(f'      10% Complete ({count} out of {num_groups} consignment-risk unit pairs)')
        elif count == int(round(0.2 * num_groups, 0)):
            print(f'      20% Complete ({count} out of {num_groups} consignment-risk unit pairs)')
        elif count == int(round(0.3 * num_groups, 0)):
            print(f'      30% Complete ({count} out of {num_groups} consignment-risk unit pairs)')
        elif count == int(round(0.4 * num_groups, 0)):
            print(f'      40% Complete ({count} out of {num_groups} consignment-risk unit pairs)')
        elif count == int(round(0.5 * num_groups, 0)):
            print(f'      50% Complete ({count} out of {num_groups} consignment-risk unit pairs)')
        elif count == int(round(0.6 * num_groups, 0)):
            print(f'      60% Complete ({count} out of {num_groups} consignment-risk unit pairs)')
        elif count == int(round(0.7 * num_groups, 0)):
            print(f'      70% Complete ({count} out of {num_groups} consignment-risk unit pairs)')
        elif count == int(round(0.8 * num_groups, 0)):
            print(f'      80% Complete ({count} out of {num_groups} consignment-risk unit pairs)')
        elif count == int(round(0.9 * num_groups, 0)):
            print(f'      90% Complete ({count} out of {num_groups} consignment-risk unit pairs)')
        elif count == int(round(1 * num_groups, 0)):
            print(f'      100% Complete ({count} out of {num_groups} consignment-risk unit pairs)')
        count+=1

        # 1) Determine action: if mixed -> 1
        action_vals = g["action"].dropna().unique()
        if len(action_vals) == 1:
            action_out = int(action_vals[0])
        else:
            # includes len==0 (all NaN) or mixed values
            action_out = 1

        # 2) B_i: should be constant within group
        B_vals = g["TOTAL_SAMPLING_UNITS_FOR_RISK_UNIT"].dropna().unique()
        if len(B_vals) == 0:
            B_i = np.nan
        elif len(B_vals) == 1:
            B_i = float(B_vals[0])
        else:
            raise ValueError(
                "TOTAL_SAMPLING_UNITS_FOR_RISK_UNIT not constant for "
                f"(INSPECTION_ID={insp_id}, RISK_UNIT={ru}). Values={B_vals}"
            )

        # 3) b_i: should be constant within group
        b_vals = g["REQUIRED_NUMBER_OF_BOXES"].dropna().unique()
        if len(b_vals) == 0:
            b_i = np.nan
        elif len(b_vals) == 1:
            b_i = float(b_vals[0])
        else:
            raise ValueError(
                "REQUIRED_NUMBER_OF_BOXES not constant for "
                f"(INSPECTION_ID={insp_id}, RISK_UNIT={ru}). Values={b_vals}"
            )

        # 4) Nbar_i = sum(QUANTITY) / B_i
        qty_sum = g["QUANTITY"].sum(skipna=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            Nbar_i = qty_sum / B_i if pd.notna(B_i) else np.nan
        if np.isinf(Nbar_i):
            Nbar_i = np.nan

        per_inspection[(insp_id, ru)] = {
            "b": b_i,
            "B": B_i,
            "Nbar": Nbar_i,
            "action": action_out,
        }

    # Convert to tidy dataframe (index = (INSPECTION_ID, risk_unit))
    per_inspection_df = pd.DataFrame.from_dict(per_inspection, orient="index")
    per_inspection_df.index = pd.MultiIndex.from_tuples(
        per_inspection_df.index, names=["INSPECTION_ID", "risk_unit"]
    )

    # Optional: remove outliers
    per_inspection_df_no_outliers = remove_outliers_iqr(
        per_inspection_df, ["b", "B", "Nbar"]
    )
    per_inspection_df = per_inspection_df_no_outliers.copy()

    # Overall averages across INSPECTION_ID-risk_unit pairs (unweighted)
    b = per_inspection_df["b"].mean(skipna=True)
    B = per_inspection_df["B"].mean(skipna=True)
    Nbar = per_inspection_df["Nbar"].mean(skipna=True)

    print("      Final inputs (averaged across INSPECTION_ID-risk_unit pairs):")
    print(f"         b = {int(round(b)) if pd.notna(b) else b}")
    print(f"         B = {int(round(B)) if pd.notna(B) else B}")
    print(f"         Nbar = {int(round(Nbar)) if pd.notna(Nbar) else Nbar}\n")

    # Return the calculated parameters
    return int(round(b)), int(round(B)), int(round(Nbar)) , per_inspection



def calc_ty_freq(per_inspection: dict):
    """
    Calculate Clarke/BB-group model inputs from per_inspection dict.

    INPUT
    per_inspection: dict
        Keys: (INSPECTION_ID, risk_unit)   (or similar pair key)
        Values: dict with at least {"action": 0/1, ...}

    OUTPUT
    ty:   List[int]  unique counts of groups testing positive (per inspection)
    freq: List[int]  frequency per ty
    """
    print("   Determining Inputs 'ty' and 'freq' (from per_inspection)")

    # Sum actions across risk_units for each inspection
    actions_per_insp = defaultdict(float)
    risk_unit_actions_per_insp = defaultdict(int)
    for key, metrics in per_inspection.items():
        # Expect tuple key (insp_id, risk_unit)
        try:
            insp_id, _risk_unit = key
        except Exception as e:
            raise ValueError(
                "Expected per_inspection keys like (INSPECTION_ID, risk_unit). "
                f"Got key={key!r}"
            ) from e

        action_val = metrics.get("action", 0)

        # Treat missing/NaN as 0; force binary-ish int
        if pd.isna(action_val):
            action_val = 0
        action_val = int(action_val)

        # If action_val is not 0/1, still counts as positive if >0
        risk_unit_actions_per_insp[insp_id] += 1 if action_val > 0 else 0
        actions_per_insp[insp_id] += metrics['b'] if action_val > 0 else 0

    # Convert to Series and compute value counts
    action_counts = pd.Series(actions_per_insp, name="n_groups_with_action")
    value_counts = action_counts.value_counts().sort_index()

    ty = value_counts.index.astype(int).to_list()
    freq = value_counts.values.astype(int).tolist()

    print("      Final inputs:")
    print(f"         ty = {ty}")
    print(f"         freq = {freq}")
    print("      Represents...")
    for t, f in zip(ty, freq):
        print(f"         There is/are {f} consignments with {t} sample units finding a pest/contaminant.")

    return ty, freq


from typing import List, Tuple

def cap_ty_at_B(ty: List[int], freq: List[int], B: int) -> Tuple[List[int], List[int]]:
    """
    Ensure no ty exceeds B.
    - Remove all entries with ty > B
    - Insert/merge a single entry ty == B whose freq is the sum of removed freqs
    - If ty already contains B, add the removed freq to that existing freq
    Preserves relative order of the kept entries.
    """
    if len(ty) != len(freq):
        raise ValueError(f"ty and freq must be same length, got {len(ty)} and {len(freq)}")

    kept_ty: List[int] = []
    kept_freq: List[int] = []
    overflow_freq_sum = 0

    for t, f in zip(ty, freq):
        if t > B:
            overflow_freq_sum += f
        else:
            kept_ty.append(t)
            kept_freq.append(f)

    if overflow_freq_sum == 0:
        return kept_ty, kept_freq

    # Merge into existing B if present; otherwise append a new (B, overflow_sum).
    try:
        idx_B = kept_ty.index(B)
    except ValueError:
        kept_ty.append(B)
        kept_freq.append(overflow_freq_sum)
    else:
        kept_freq[idx_B] += overflow_freq_sum

    return kept_ty, kept_freq


def gen_clarke_model_inputs(
    df_pis_data: pd.DataFrame
) -> ClarkeModelInputs:
    """
    Function to construct inputs for the Clarke/BB-group model:
    ty: List[int]                  # unique counts of groups testing positive
    freq: List[int]                # frequency per ty
    b: int                         # groups per consignment (boxes per inspection)
    B: int                         # number of consignments (inspections)
    Nbar: int                      # items per group
    theta: float                   # clustering hyperparameter
    lambda_default: float          # clustering hyperparameter
    R: int                         # MC / sensitivity runs
    start_val: Tuple[float, float]  # optimizer start (log-alpha, log-beta)
    se: bool                       # request standard errors


    INPUTS
    df_pis_data:        Pandas Dataframe with columns:
                        - INSPECTION_ID
                        - COUNTRY_OF_ORIGIN_NAME
                        - action (0/1 per group/box)


    OUTPUTS
    ClarkeModelInputs data class with the appropriate values listed above.
    """

    # Define Defaults
    clarke_inputs = ClarkeModelInputs.default()

    # Get applicable and intersected inspection ids (inspection ids common across both data sets)
    #df_pis_data_filtered_by_inspection_id, df_rbs_calculator_filtered_by_inspection_id = identify_relevant_consignments(df_pis_data)

    # Function to generate b, B, and Nbar input parameters
    print(f"Determining All Clarke Model Required Inputs")
    clarke_inputs.b, clarke_inputs.B, clarke_inputs.Nbar, per_inspection_data = _get_b_B_nbar_inputs(df_pis_data)

    # Calculate the ty and freq input parameters
    clarke_inputs.ty, clarke_inputs.freq = calc_ty_freq(per_inspection_data)
    clarke_inputs.ty, clarke_inputs.freq = cap_ty_at_B(clarke_inputs.ty, clarke_inputs.freq, clarke_inputs.b)

    print("      Final inputs after removal:")
    print(f"         ty = {clarke_inputs.ty}")
    print(f"         freq = {clarke_inputs.freq}")
    print("      Represents...")
    for t, f in zip(clarke_inputs.ty, clarke_inputs.freq):
        print(f"         There is/are {f} consignments with {t} sample units finding a pest/contaminant.")

    return clarke_inputs


def lambda_of_theta(theta, Nbar=100):
    """
    Compute lambda(theta) = Nbar / (1 + sum_{k=1}^{Nbar-1} theta/(theta+k)).
    """
    k = np.arange(1, Nbar)   # 1..Nbar-1
    denom = 1.0 + np.sum(theta / (theta + k))
    return Nbar / denom


def theta_from_lambda(lambda_target, Nbar=100):
    """
    Solve for theta given target lambda in [1, Nbar].
    Works with a single lambda (float) or an array-like of lambdas.
    """
    lambda_target = np.atleast_1d(lambda_target)  # handle scalars & arrays

    def solve_single(lam_target):
        def loss(logtheta):
            theta = np.exp(logtheta)     # enforce positivity
            lam = lambda_of_theta(theta, Nbar)
            return (lam - lam_target)**2

        result = minimize_scalar(loss, bounds=(-10, 10), method="bounded")
        return np.exp(result.x)

    thetas = np.array([solve_single(lam) for lam in lambda_target])

    # Return scalar if input was scalar
    return thetas if lambda_target.size > 1 else thetas.item()

