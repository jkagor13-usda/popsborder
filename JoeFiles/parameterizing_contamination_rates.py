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


from JoeFiles.clarke_r_script_wrapper import *
from JoeFiles.clarke_model_support_functions import *

if __name__ == "__main__":

    ### Generate clarke inputs via input data

    # Pull in the  data
    # Pull in synthetic information and compare
    dir = r'C:\Users\agorjk1\Box\NHH15 - USDA APHIS EDISON\05 PPQ Engagement\PPQ RBS Data (Folder shared with APHIS)\APL Created Data Related Items\Synthetic_Data'
    filename = os.path.join(dir, f'synthetic_pis_data.csv')
    # Load in task log from a single replication of a run
    df_pis_synthetic = pd.read_csv(filename)

    inputs = gen_clarke_model_inputs(df_pis_synthetic,
                                     b=25,  # or let it infer if constant
                                     Nbar=200,
                                     theta=np.inf,
                                     lambda_test=1,
                                     R=1000,
                                     startval=(0.0, 0.0),
                                     se=True)

    # Retrieve exactly what you want:
    print(inputs.ty)
    print(inputs.freq)
    print(inputs.b, inputs.B, inputs.Nbar, inputs.theta, inputs.lambda_test,
          inputs.R, inputs.startval, inputs.se)

    # Or pass straight into your model call
    kwargs = inputs.to_kwargs()
    # run_bb_group_model(ty=inputs.ty, b=inputs.b, B=inputs.B, Nbar=inputs.Nbar,
    #                    freq=inputs.freq, theta=inputs.theta, R=inputs.R,
    #                    startval=inputs.startval, se=inputs.se)

    # Create a dataframe to store results
    columns = ['INSPECTION_ID',
               'COUNTRY_OF_ORIGIN_NAME',
               'Number of Boxes with Action']
    action_summary = pd.DataFrame(columns=columns)

    for inspection_id, group_df in df_pis_synthetic.groupby("INSPECTION_ID"):
        group_df = group_df.reset_index(drop=True)

        action_summary.loc[len(action_summary)] = {
            'INSPECTION_ID': inspection_id,
            'COUNTRY_OF_ORIGIN_NAME': group_df['COUNTRY_OF_ORIGIN_NAME'][0],
            'Number of Boxes with Action': sum(group_df['action']),
        }

    value_counts = action_summary["Number of Boxes with Action"].value_counts().sort_index()
    ty = value_counts.index.to_list()  # Number of boxes that have been identified actions
    freq = value_counts.values.tolist()  # Frequency of observations/consignments where that many groups tested positive


    #ty = [0, 1, 3, 13, 18, 23]  # Number of groups testing positive
    #freq = [94, 1, 1, 1, 1, 2]  # Frequency of observations/consignments where that many groups tested positive
    b = 25
    B = 100
    Nbar = 200
    theta = np.inf
    lambda_test = 1
    R = 1000
    startval = (0.0, 0.0)
    se = True

    #lambda_test = 5
    #theta = theta_from_lambda(lambda_test, Nbar=Nbar)
    # Calculate a theta from a fixed lambda
    # Case 1: single lambda
    #theta_single = theta_from_lambda(50, Nbar=100)
    #print("Theta for lambda=50:", theta_single)

    # Case 2: array of lambdas
    lambdas = [1, 10, 25, 50, 75, 90]
    thetas = theta_from_lambda(lambdas, Nbar=Nbar)
    '''
    print('')
    print(f'FITTING WITH CLARKE 2023 METHOD (BETA-BINOMIAL APPROACH)')
    print(f'   Parameters when NO CLUSTERING (theta = {theta}/lambda = {lambda_test})')
    res = run_bb_group_model(ty, b, B, Nbar, freq, theta, R, startval, se)
    print(f'      Alpha = {res["alpha"]}')
    print(f'      Beta = {res["beta"]}')
    print('')

    '''
    # Initialize an empty DataFrame once before the loop
    df_results = pd.DataFrame(columns=[
        "Method",
        "Lambda",
        "Number of contaminated consignments",
        "Five number summary",
        "Mean contamination rate",
        "95% CI Lower",
        "95% CI Upper",
        "# Contaminated Plants/Units (Avg for methods and actual for ground truth)"
    ])

    for i in range(len(thetas)):
        if lambdas[i] == 1:
            theta = np.inf
            print(f'CLARKE APPROACH:  Parameters when NO CLUSTERING (theta = {theta}/lambda = 1)')
        else:
            theta = thetas[i]
            print(f'CLARKE APPROACH: Parameters when NO CLUSTERING (theta = {theta}/lambda = {lambdas[i]})')
        res = run_bb_group_model(ty, b, B, Nbar, freq, theta, R, startval, se)
        print(f'   Alpha = {res["alpha"]}')
        print(f'   Beta = {res["beta"]}')
        print('')