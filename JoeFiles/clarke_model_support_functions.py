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
    startval: Tuple[float, float]  # optimizer start (log-alpha, log-beta)
    se: bool                       # request standard errors

    # optional: keep a summary table for debugging/reporting
    action_summary: Optional[pd.DataFrame] = None

    def to_kwargs(self) -> dict:
        """Convenience: kwargs for your BB model call (omit DataFrame)."""
        d = asdict(self)
        d.pop("action_summary", None)
        return d


def gen_clarke_model_inputs(
    df_pis_synthetic: pd.DataFrame,
    *,
    # model hyperparameters (give sensible defaults)
    b: Optional[int] = None,       # groups per consignment; will try to infer if None
    Nbar: int = 200,
    theta: float = np.inf,
    lambda_test: float = 1.0,
    R: int = 1000,
    startval: Tuple[float, float] = (0.0, 0.0),
    se: bool = True,
    ensure_full_range: bool = True # include counts 0..b in the histogram
) -> ClarkeModelInputs:
    """
    Build inputs for the Clarke/BB-group model from a long dataframe with columns:
      - INSPECTION_ID
      - COUNTRY_OF_ORIGIN_NAME
      - action (0/1 per group/box)
    """

    required_cols = {"INSPECTION_ID", "COUNTRY_OF_ORIGIN_NAME", "action"}
    missing = required_cols - set(df_pis_synthetic.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    # Vectorized consignment-level aggregation
    # Number of boxes with action per INSPECTION_ID (retain country for reporting)
    agg = (
        df_pis_synthetic
        .groupby(["INSPECTION_ID", "COUNTRY_OF_ORIGIN_NAME"], as_index=False)
        .agg(**{"Number of Boxes with Action": ("action", "sum")})
    )
    action_summary = agg.copy()

    # B: number of consignments
    B = int(len(action_summary))

    # Infer b if not provided (and optionally validate constancy)
    if b is None:
        counts_per_inspection = (
            df_pis_synthetic.groupby("INSPECTION_ID", as_index=False).size()["size"]
        )
        if counts_per_inspection.nunique() != 1:
            raise ValueError(
                "Cannot infer a single b: number of groups per consignment varies. "
                "Pass b explicitly or pre-harmonize your data."
            )
        b = int(counts_per_inspection.iloc[0])

    # Build histogram of "Number of Boxes with Action"
    vc = action_summary["Number of Boxes with Action"].value_counts().sort_index()

    # Ensure we have the full support (0..b), so zeros aren't dropped
    if ensure_full_range:
        full_index = pd.RangeIndex(0, b + 1, 1)
        vc = vc.reindex(full_index, fill_value=0)

    # Extract ty (only values with positive freq) and aligned freq
    nonzero = vc[vc > 0]
    ty = nonzero.index.astype(int).tolist()
    freq = nonzero.values.astype(int).tolist()

    return ClarkeModelInputs(
        ty=ty,
        freq=freq,
        b=b,
        B=B,
        Nbar=Nbar,
        theta=theta,
        lambda_test=lambda_test,
        R=R,
        startval=startval,
        se=se,
        action_summary=action_summary
    )





















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

