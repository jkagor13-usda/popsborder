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




from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


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


def _get_b_B_nbar_inputs(
        df_pis_data_filtered: pd.DataFrame,
        df_rbs_calculator_filtered: pd.DataFrame
):
    print(f"   Determining Inputs 'b', 'B' and 'Nbar'")
    # Function to generate b, B, and Nbar input parameters
    # Get applicable and interesected inspection ids
    df_pis_data_grouped, df_rbs_calculator_grouped = group_by_inspection_id(df_pis_data_filtered, df_rbs_calculator_filtered)

    # Check to ensure if column name for "required number of boxes" exists
    req_boxes_col = None
    for candidate in ["REQUIRED_NUMBER_OF_BOXES", "REQUIRED_NUMBER_OF_BOXES"]:
        if candidate in next(iter(df_rbs_calculator_grouped.values())).columns:
            req_boxes_col = candidate
            break

    if req_boxes_col is None:
        raise KeyError("Could not find REQUIRED_NUMBER_OF_BOXES in RBS Calculator Data columns.")

    # Go through each of the different sets of the inspection ids and calculate the parameters
    # averaging if there are multiple records per inspection id in the RBS calculator data
    per_inspection = {}
    for insp_id, df in df_rbs_calculator_grouped.items():
        # Ensure numeric, ignore NaNs in means
        req_boxes = pd.to_numeric(df[req_boxes_col], errors="coerce")
        total_units = pd.to_numeric(df["TOTAL_SAMPLING_UNITS"], errors="coerce")
        total_plants = pd.to_numeric(df["TOTAL_PLANT_QUANTITY"], errors="coerce")

        # b_i = avg required boxes for this inspection_id
        b_i = req_boxes.mean()

        # B_i = avg total sampling units for this inspection_id
        B_i = total_units.mean()

        # Nbar_i = avg over rows of (TOTAL_PLANT_QUANTITY / TOTAL_SAMPLING_UNITS) for this inspection_id
        with np.errstate(divide="ignore", invalid="ignore"):
            nbar_rows = total_plants / total_units
        nbar_rows = nbar_rows.replace([np.inf, -np.inf], np.nan)
        Nbar_i = nbar_rows.mean()

        per_inspection[insp_id] = {"b": b_i, "B": B_i, "Nbar": Nbar_i}

    # Convert to a tidy dataframe (index = INSPECTION_ID)
    per_inspection_df = pd.DataFrame.from_dict(per_inspection, orient="index")

    # Remove outliers from per_inspection_df
    per_inspection_df_no_outliers = remove_outliers_iqr(per_inspection_df, ["b", "B", "Nbar"])
    per_inspection_df = per_inspection_df_no_outliers.copy()

    # Overall (unweighted) averages across inspection IDs with graceful fallbacks
    b = per_inspection_df["b"].mean(skipna=True)
    B = per_inspection_df["B"].mean(skipna=True)
    Nbar = per_inspection_df["Nbar"].mean(skipna=True)

    # Fallbacks if anything is NaN/inf/non-positive
    if (not np.isfinite(b)) or b <= 0:
        b = pd.to_numeric(df_rbs_calculator_filtered["TOTAL_SAMPLING_UNITS"], errors="coerce").dropna().mean()
    if (not np.isfinite(Nbar)) or Nbar <= 0:
        Nbar = pd.to_numeric(df_rbs_calculator_filtered["TOTAL_PLANT_QUANTITY"], errors="coerce").dropna().mean()
    if (not np.isfinite(B)) or B <= 0:
        B = float(len(df_rbs_calculator_filtered["INSPECTION_ID"].unique()))

    # Final safety defaults
    b = b if np.isfinite(b) and b > 0 else 1.0
    B = B if np.isfinite(B) and B > 0 else 1.0
    Nbar = Nbar if np.isfinite(Nbar) and Nbar > 0 else 1.0

    print("      Final inputs (averaged across inspection IDs):")
    print(f"         b = {round(b,0)}")
    print(f"         B = {round(B,0)}")
    print(f"         Nbar = {round(Nbar,0)}\n")

    return round(b,0), round(B,0), round(Nbar,0)


def calc_ty_freq(df_pis_data_filtered_by_inspection_id: pd.DataFrame):
    """
    Function to calculate the two inputs below for the Clarke/BB-group model:
    ty: List[int]                  # unique counts of groups testing positive
    freq: List[int]                # frequency per ty

    INPUTS
    df_pis_data_filtered_by_inspection_id:  Pandas Dataframe with columns:
                                               - INSPECTION_ID
                                               - COUNTRY_OF_ORIGIN_NAME
                                               - action (0/1 per group/box)
    OUTPUTS
    Two lists ty and freq as defined above.
    """
    print(f"   Determining Inputs 'ty' and 'freq'")
    # Create a dataframe to store results
    columns = ['INSPECTION_ID',
               'COUNTRY_OF_ORIGIN_NAME',
               'Number of Commodity Lines with Action']
    action_summary = pd.DataFrame(columns=columns)

    for inspection_id, group_df in df_pis_data_filtered_by_inspection_id.groupby("INSPECTION_ID"):
        group_df = group_df.reset_index(drop=True)

        action_summary.loc[len(action_summary)] = {
            'INSPECTION_ID': inspection_id,
            'COUNTRY_OF_ORIGIN_NAME': group_df['COUNTRY_OF_ORIGIN_NAME'][0],
            'Number of Commodity Lines with Action': sum(group_df['action']),
        }

    value_counts = action_summary["Number of Commodity Lines with Action"].value_counts().sort_index()
    ty = value_counts.index.to_list()  # Number of boxes that have been identified actions
    freq = value_counts.values.tolist()  # Frequency of observations/consignments where that many groups tested positive

    print("      Final inputs:")
    print(f"         ty = {ty}")
    print(f"         freq = {freq}")
    print("      Represents...")
    for i in range(len(ty)):
        print(f"         There is/are {freq[i]} consignments with {ty[i]} inspections finding a pest/contaminant.")
    return ty, freq


def gen_clarke_model_inputs(
    df_pis_data: pd.DataFrame,
    df_rbs_calculator: pd.DataFrame
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
    df_rbs_calculator:  Pandas Dataframe with RBS Calculator columns


    OUTPUTS
    ClarkeModelInputs data class with the appropriate values listed above.
    """

    # Define Defaults
    clarke_inputs = ClarkeModelInputs.default()

    # Get applicable and intersected inspection ids (inspection ids common across both data sets)
    df_pis_data_filtered_by_inspection_id, df_rbs_calculator_filtered_by_inspection_id = identify_relevant_consignments(df_pis_data, df_rbs_calculator)

    # Function to generate b, B, and Nbar input parameters
    print(f"Determining All Clarke Model Required Inputs")
    clarke_inputs.b, clarke_inputs.B, clarke_inputs.Nbar = _get_b_B_nbar_inputs(df_pis_data_filtered_by_inspection_id, df_rbs_calculator_filtered_by_inspection_id)

    # Calculate the ty and freq input parameters
    clarke_inputs.ty, clarke_inputs.freq = calc_ty_freq(df_pis_data_filtered_by_inspection_id)

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

