import sys
import re
import warnings

import pandas as pd
import numpy as np

from pybedtools import BedTool


def apply_scaling(pred_file, scale_factor, n_class, out_file):
    """
    Apply scaling to the predictions in the given file.
    
    Args:
        pred_file (str): Path to the prediction file.
        scale_factor (float): Scaling factor to apply.
    
    Returns:
        pd.DataFrame: DataFrame with scaled predictions.
    """
    # Read the prediction file
    df = pd.read_csv(pred_file, sep="\t", header=0)
    column_names = [f'prob{i}' for i in range(1, n_class)]
    
    # Scale the predictions
    df[column_names] = df[column_names].astype(float) * scale_factor
    df['prob0'] = 1 - df[column_names].sum(axis=1)
    df.to_csv(out_file, sep="\t", index=False, float_format='%.4g')

def scaling_files(pred_files, scale_factors, n_class, out_file):
    for idx, pred_file in enumerate(pred_files):
        scale_factor = scale_factors[idx]
        out_file = out_files[idx]
        apply_scaling(pred_file, scale_factor, n_class, out_file[idx])


def calc_mu_scaling_factor(args):

   # Read the prediction file
    benchmark_regions = args.benchmark_regions
    
    genomewide_mu = args.genomewide_mu
    
    g_proportions = args.g_proportions
    
    m_proportions = args.m_proportions
    
    pred_files = args.pred_files
    
    do_scaling = args.do_scaling

    n_class = args.n_class
    
    if len(m_proportions) != len(pred_files):
        print('ERROR: length of proportions does not equal to length of pred_files!', file=sys.stderr)
        
        sys.exit()
    
    for i in range(len(pred_files)):

        df = pd.read_table(pred_files[i], sep='\t', header=0)

        if benchmark_regions:
            benchmark_bed = BedTool(benchmark_regions)


        df_name = pd.DataFrame('.', index=range(df.shape[0]), columns=['name'])

        prob_cols = [f'prob{i}' for i in range(1, n_class)]
        df_score = df[prob_cols].sum(axis=1).to_frame(name='score')

        pred_df = pd.concat((df[['chrom', 'start', 'end']], df_name, df_score, df['strand']), axis=1)

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        pred_df_bed = BedTool.from_dataframe(pred_df)

        if benchmark_regions:
            probs = pred_df_bed.intersect(benchmark_bed).to_dataframe()[['score']]
        else:
            probs = pred_df_bed.to_dataframe()[['score']]

        prob_sum = np.sum(probs.values)

        n_sites = probs.shape[0]

        scale_factor = (genomewide_mu * n_sites * m_proportions[i] / g_proportions[i]) / prob_sum
        print('\nType '+str(i+1)+':\n'+ 'pred_file:', pred_files[i])
        print('genomewide_mu:', genomewide_mu)
        print('n_sites:', n_sites)
        print('g_proportion:', g_proportions[i])
        print('m_proportion:', m_proportions[i])
        print('prob_sum: %.3e' % prob_sum)
        print('scaling factor: %.3e' % scale_factor)
        
        if do_scaling:
            pred_file = pred_files[i]
            out_file = pred_file +'.scaled.tsv.gz'
            apply_scaling(pred_file, scale_factor, n_class, out_file)
    
    return scale_factor