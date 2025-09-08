# rpy2 wrapper for BB.group.model (R)
# -----------------------------------
import os
import numpy as np

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.vectors import IntVector, FloatVector, StrVector
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter
from rpy2.rinterface_lib.embedded import RRuntimeError

# Enable NumPy <-> R auto-conversion
numpy2ri.activate()

def _rlist_to_py(res):
    """
    Convert an R 'list' (ListVector) with names into a Python dict,
    converting elements to native Python types when possible.
    """
    from rpy2.robjects import vectors, conversion

    py = {}
    names = list(res.names) if getattr(res, "names", None) is not None else [None] * len(res)
    for key, val in zip(names, res):
        # Try rpy2's generic conversion first
        try:
            with localconverter(default_converter + numpy2ri.converter):
                p = ro.conversion.rpy2py(val)
        except Exception:
            p = val

        # Unbox common 1-length vectors to scalars
        if hasattr(p, "__len__") and not isinstance(p, (str, bytes)) and len(p) == 1:
            # Some come back as numpy arrays length 1
            try:
                p = p.item()
            except Exception:
                pass

        # If still an R vector, coerce to Python list
        if "rpy2" in p.__class__.__module__:
            try:
                p = list(p)
            except Exception:
                pass

        py[key] = p
    return py

def load_r_source(path_to_r):
    """Source the R file once; safe to call multiple times."""
    if not os.path.exists(path_to_r):
        raise FileNotFoundError(f"R source file not found: {path_to_r}")
    ro.r['source'](path_to_r)

def bb_group_model_r(ty, b, B, Nbar, freq=None, theta=np.inf, SE=True,
                     r_source_path="simstudy_paperspace_27_04_2023.R"):
    """
    Call the R function BB.group.model from Python via rpy2.

    Parameters
    ----------
    ty : array-like of ints
    b, B, Nbar : ints
    freq : array-like of ints (optional; defaults to 1s)
    theta : float (np.inf for unclustered); 'Inf' handled automatically
    SE : bool
    r_source_path : path to the R file that defines BB.group.model

    Returns
    -------
    dict : keys mirror the R list (alpha, beta, mu, rho, D, E.leak, prob.leak, etc.)
    """
    # Ensure R code is loaded
    load_r_source(r_source_path)

    # Grab the function from R global env
    BB_group_model = ro.r['BB.group.model']

    # Build R vectors
    ty = np.asarray(ty, dtype=int)
    r_ty = IntVector(list(ty))

    if freq is None:
        r_freq = IntVector([1] * len(ty))
    else:
        r_freq = IntVector(list(np.asarray(freq, dtype=int)))

    # Theta handling: pass R's literal Inf for unclustered
    r_theta = ro.r('Inf') if np.isinf(theta) else float(theta)

    # Call R
    try:
        res = BB_group_model(ty=r_ty,
                             freq=r_freq,
                             b=int(b),
                             B=int(B),
                             Nbar=int(Nbar),
                             theta=r_theta,
                             SE=bool(SE))
    except RRuntimeError as e:
        # Surface R errors cleanly
        raise RuntimeError(f"R error from BB.group.model: {e}") from e

    # Convert R list to Python dict
    return _rlist_to_py(res)
