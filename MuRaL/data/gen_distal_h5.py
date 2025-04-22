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
#import resource

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
    
    optional.add_argument('--without_bw_distal', default=False, action='store_true', 
                          help=textwrap.dedent("""
                          Do not use BigWig tracks for distal-related layers. Default: False.""").strip())

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
    
    optional.add_argument('--distal_binsize', type=int, metavar='INT', default=1, 
                          help=textwrap.dedent("""
                          Bin size of distal sequences. Kept for 
                          future development. Default: 1. """ ).strip())
    
    optional.add_argument('--i_file', type=int, metavar='INT', default=0, 
                          help=textwrap.dedent("""
                          Number of HDF5 files. Default: 0. """ ).strip())
    
    optional.add_argument('--n_files', type=int, metavar='INT', default=1, 
                          help=textwrap.dedent("""
                          Number of HDF5 files. Default: 1. """ ).strip())
    

    optional.add_argument('--chunk_size', type=int, metavar='INT', default=10000, 
                          help=textwrap.dedent("""
                          Bioseq read chunk size. Default: 10000. """ ).strip())
    
    optional.add_argument('--out_format', type=str, metavar='STR', default='h5',  
                          help=textwrap.dedent("""
                          Generate HDF5 ('h5').
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
    This tool generates HDF file(s) of expanded regions for a BED file.
    
    More doc to be added.
""") 
    
    args = parse_arguments(parser)
    
    # Print command line
    print(' '.join(sys.argv))
    for k,v in vars(args).items():
        print("{0}: {1}".format(k,v))

    # Set input file
    bed_file = os.path.abspath(args.bed_file)
    ref_genome= os.path.abspath(args.ref_genome)
    
    # Whether to generate H5 file for distal data
    out_format = args.out_format


    distal_radius = args.distal_radius
    distal_order = args.distal_order # reserved for future improvement
    distal_binsize = args.distal_binsize
    
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
    without_bw_distal = args.without_bw_distal
    
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
            h5f_path = get_h5f_path(bed_file, bw_names, distal_radius, distal_order, without_bw_distal)
            
            #test_bed = BedTool(bed_file)
            generate_h5f(test_bed, h5f_path, ref_genome, distal_radius, distal_order, bw_files, 1, chunk_size, without_bw_distal)
            #generate_h5fv2(test_bed, h5f_path, ref_genome, distal_radius, distal_order, bw_files, 1, chunk_size)

        elif n_files > 1:
            cmd = sys.argv[0]
            
            #resource.setrlimit(resource.RLIMIT_CPU, (1, 4))
            
            ps = []
            for i in range(n_files):
                args = [cmd, 
                         '--ref_genome', ref_genome, 
                         '--bed_file', bed_file,
                         '--distal_radius', str(distal_radius), 
                         '--distal_order', str(distal_order), 
                         '--i_file', str(i+1), 
                         '--n_files', str(n_files), 
                         '--chunk_size', str(chunk_size),
                         '--distal_binsize', str(distal_binsize)]
                if bw_paths != None:
                    args.append('--bw_paths')
                    args.append(bw_paths)
                if without_bw_distal:
                    args.append('--without_bw_distal')
                
                p = subprocess.Popen(args)
                ps.append(p)
            for p in ps:
                p.wait()
            
            h5f_path = get_h5f_path(bed_file, bw_names, distal_radius, distal_order, without_bw_distal)
            
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
        #resource.setrlimit(resource.RLIMIT_CPU, (1, n_files))
        
        h5f_path = get_h5f_path(bed_file, bw_names, distal_radius, distal_order, without_bw_distal)
        
        #test_bed = BedTool(bed_file)
        
        single_size = int(np.ceil(len(test_bed)/float(n_files)))
        h5f_path_i = re.sub('h5$', str(i_file)+'.h5', h5f_path)
        bed_regions = BedTool(test_bed.at(range((i_file-1)*single_size,np.min([i_file*single_size, len(test_bed)]))))
        
        if distal_binsize == 1:
            generate_h5f_singlev1(bed_regions, h5f_path_i, ref_genome, distal_radius, distal_order, bw_files, chunk_size, without_bw_distal)
        else:
            generate_h5f_singlev2(bed_regions, h5f_path_i, ref_genome, distal_radius, distal_order, distal_binsize, bw_files, chunk_size, without_bw_distal)
    
    #test_bed.at(range(single_size, bed_end))
    
    ##########################
        
    sys.stdout.flush()
    
    # hf.keys() to get number of datasets
  
  
    print('Total time used: %s seconds' % (time.time() - start_time))
   
    
if __name__ == "__main__":
    main()


