import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from pybedtools import BedTool

import sys
import argparse
import textwrap
#from sklearn.preprocessing import LabelEncoder
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import pickle

import os
import time
import datetime

#from MuRaL.nn_models import *
#from MuRaL.nn_utils import *
from MuRaL.preprocessing import *
#from MuRaL.evaluation import *
from MuRaL._version import __version__

import subprocess
import h5py


#from pynvml import *

#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.allow_tf32 = True

def parse_arguments(parser):
    """
    Parse parameters from the command line
    """ 
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('Required arguments')
    optional.title = 'Other arguments' 
    
    required.add_argument('--ref_genome', type=str, metavar='FILE', default='',  
                          required=True, help=textwrap.dedent("""
                          File path of the reference genome in FASTA format.""").strip())
    
    required.add_argument('--bed_file', type=str, metavar='FILE', required=True,
                          help= textwrap.dedent("""
                          File path of test data to do prediction, in BED format.""").strip())
    
    optional.add_argument('--bw_paths', type=str, metavar='FILE', default=None,
                          help=textwrap.dedent("""
                          File path for a list of BigWig files for non-sequence 
                          features such as the coverage track. Default: None.""").strip())

    optional.add_argument('--distal_radius', type=int, metavar='INT', default=200, 
                          help=textwrap.dedent("""
                          Radius of the expanded sequence to be considered in the model. 
                          Length of the expanded sequence = distal_radius*2+1 bp.
                          Values should be >=100. Default: 200. 
                          """ ).strip())
    
    optional.add_argument('--distal_order', type=int, metavar='INT', default=1, 
                          help=textwrap.dedent("""
                          Order of distal sequences to be considered. Kept for 
                          future development. Default: 1. """ ).strip())
    
    optional.add_argument('--i_file', type=int, metavar='INT', default=0, 
                          help=textwrap.dedent("""
                          Number of HDF5 files. Default: 0. """ ).strip())
    
    optional.add_argument('--n_files', type=int, metavar='INT', default=1, 
                          help=textwrap.dedent("""
                          Number of HDF5 files. Default: 1. """ ).strip())
    

    optional.add_argument('--chunk_size', type=int, metavar='INT', default=5000, 
                          help=textwrap.dedent("""
                          Bioseq read chunk size. Default: 5000. """ ).strip())
    
    optional.add_argument('--out_format', type=str, metavar='STR', default='h5',  
                          help=textwrap.dedent("""
                          Generate HDF5 ('h5') or Zarr ('zarr').
                          """).strip())
    
    optional.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))
    
    parser._action_groups.append(optional)
    
    if len(sys.argv) == 1:
        parser.parse_args(['--help'])
    else:
        args = parser.parse_args()

    return args

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="""
    Overview
    -------- 
    This tool uses a trained MuRaL model to do prediction for the sites in the 
    input BED file.
    
    * Input data 
    The required input files for prediction include the reference FASTA file, 
    a BED-formated data file and a trained model. The BED file is organized 
    in the same way as that for training. The 5th column can be set to '0' 
    if no observed mutations for the sites in the prediction BED. The 
    model-related files for input are 'model' and 'model.config.pkl', which 
    are generated at the training step. The file 'model.fdiri_cal.pkl', which 
    is for calibrating predicted mutation rates, is optional.
   
    * Output data 
    The output of `mural_predict` is a tab-separated file containing the 
    sequence coordinates and the predicted probabilities for all possible 
    mutation types. Usually, the 'prob0' column contains probalities for the 
    non-mutated class and other 'probX' columns for mutated classes. 
   
    Some example lines of a prediction output file are shown below:
    chrom   start   end    strand mut_type  prob0   prob1   prob2   prob3
    chr1    10006   10007   -       0       0.9797  0.003134 0.01444 0.002724
    chr1    10007   10008   +       0       0.9849  0.005517 0.00707 0.002520
    chr1    10008   10009   +       0       0.9817  0.004801 0.01006 0.003399
    chr1    10012   10013   -       0       0.9711  0.004898 0.02029 0.003746

    Command line examples
    ---------------------
    1. The following command will predict mutation rates for all sites in 
    'testing.bed.gz' using model files under the 'checkpoint_6/' folder 
    and save prediction results into 'testing.ckpt6.fdiri.tsv.gz'.
    
        mural_predict --ref_genome seq.fa --bed_file testing.bed.gz \\
        --model_path checkpoint_6/model \\
        --model_config_path checkpoint_6/model.config.pkl \\
        --calibrator_path checkpoint_6/model.fdiri_cal.pkl \\
        --pred_file testing.ckpt6.fdiri.tsv.gz \\
        > test.out 2> test.err
    """) 
    
    args = parse_arguments(parser)
    
    # Print command line
    print(' '.join(sys.argv))
    for k,v in vars(args).items():
        print("{0}: {1}".format(k,v))

    # Set input file
    bed_file = args.bed_file   
    ref_genome= args.ref_genome
    
    # Whether to generate H5 file for distal data
    out_format = args.out_format


    distal_radius = args.distal_radius
    distal_order = args.distal_order # reserved for future improvement
    
    i_file = args.i_file
    n_files = args.n_files

    chunk_size = args.chunk_size
    
    start_time = time.time()
    print('Start time:', datetime.datetime.now())
    sys.stdout.flush()
    
    # Read BED files
    test_bed = BedTool(bed_file)

    # Read bigWig file names
    bw_paths = args.bw_paths
    bw_files = []
    bw_names = []
    
    if bw_paths:
        try:
            bw_list = pd.read_table(bw_paths, sep='\s+', header=None, comment='#')
            bw_files = list(bw_list[0])
            bw_names = list(bw_list[1])
        except pd.errors.EmptyDataError:
            print('Warnings: no bigWig files provided in', bw_paths)
    else:
        print('NOTE: no bigWig files provided.')

    if i_file == 0:
        if n_files == 1:
            h5f_path = get_h5f_path(bed_file, bw_names, distal_radius, distal_order)
            generate_h5f(test_bed, h5f_path, ref_genome, distal_radius, distal_order, bw_files, 1, chunk_size)
            #generate_h5f2(test_bed, h5f_path, ref_genome, distal_radius, distal_order, bw_files, 1, chunk_size)
            #generate_h5fv2(test_bed, h5f_path, ref_genome, distal_radius, distal_order, bw_files, 1, chunk_size)
            #generate_h5f_mp(test_bed, test_h5f_path, ref_genome, distal_radius, distal_order, bw_files, 1, 5)
        elif n_files > 1:
            cmd = sys.argv[0]
            
            ps = []
            for i in range(n_files):
                args = [cmd, 
                         '--ref_genome', ref_genome, 
                         '--bed_file', bed_file, 
                         '--distal_radius', str(distal_radius), 
                         '--distal_order', str(distal_order), 
                         '--i_file', str(i+1), 
                         '--n_files', str(n_files), 
                         '--chunk_size', str(chunk_size)]
                if bw_paths != None:
                    args.append('--bw_paths')
                    args.append(bw_paths)
                #'--bw_paths', bw_paths, 
                p = subprocess.Popen(args)
                ps.append(p)
            for p in ps:
                p.wait()
            
            h5f_path = get_h5f_path(bed_file, bw_names, distal_radius, distal_order)
            
            with h5py.File(h5f_path, 'w') as hf:
                for i in  range(n_files):
                    h5f_path_i = re.sub('h5$', str(i+1)+'.h5', h5f_path)
                    hf['distal_X'+str(i+1)] = h5py.ExternalLink(h5f_path_i, 'distal_X')
            with h5py.File(h5f_path, 'r') as hf:
                data_size = 0
                for key in hf.keys():
                    data_size += hf[key].shape[0]
                    print(key, hf[key].shape[0])
                print('hf.keys():', hf.keys(), data_size)

            
    else:
        h5f_path = get_h5f_path(bed_file, bw_names, distal_radius, distal_order)
        single_size = int(np.ceil(len(test_bed)/float(n_files)))
        h5f_path_i = re.sub('h5$', str(i_file)+'.h5', h5f_path)
        bed_regions = BedTool(test_bed.at(range((i_file-1)*single_size,np.min([i_file*single_size, len(test_bed)]))))
        
        #generate_h5f_single2(bed_regions, h5f_path_i, ref_genome, distal_radius, distal_order, bw_files, 1, i_file, single_size)
        generate_h5f_singlev2(bed_regions, h5f_path_i, ref_genome, distal_radius, distal_order, bw_files, i_file, chunk_size)
    
    #test_bed.at(range(single_size, bed_end))
    
    ##########################
        
    sys.stdout.flush()
    
    # hf.keys() to get number of datasets
  
  
    print('Total time used: %s seconds' % (time.time() - start_time))
   
    
if __name__ == "__main__":
    main()


