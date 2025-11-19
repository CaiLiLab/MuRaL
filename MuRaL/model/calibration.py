import math
import numpy as np
import pandas as pd

def lambda_calib(prob0, prob):
    lambda_calib = -math.log(prob0)
    prob_calib = lambda_calib * prob / (1 - prob0)
    return prob_calib

def poisson_calibrate(df_pred):

    prob0 = np.clip(df_pred['prob0'], 1e-10, 1.0)
    lambda_calib = -np.log(prob0)
    denominator = 1 - prob0
    prob_cols = [col for col in df_pred.columns if col.startswith('prob') and col != 'prob0']

    df_calib = df_pred.copy()
    # Apply calibration to each probability column
    for col in prob_cols:
        df_calib[col] = lambda_calib * df_pred[col] / denominator

    df_calib['prob0'] = 1 - lambda_calib
    return df_calib
