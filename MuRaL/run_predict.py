import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from pybedtools import BedTool

import sys
import argparse
import textwrap
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import pickle

import os
import time
import datetime

from MuRaL.nn_models import *
from MuRaL.nn_utils import *
from MuRaL.preprocessing import *
from MuRaL.evaluation import *
from MuRaL._version import __version__


from pynvml import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True

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
    
    required.add_argument('--test_data', type=str, metavar='FILE', required=True,
                          help= textwrap.dedent("""
                          File path of test data to do prediction, in BED format.""").strip())
    
    required.add_argument('--model_path', type=str, metavar='FILE', required=True,
                          help=textwrap.dedent("""
                          File path of the trained model.
                          """ ).strip())
        
    required.add_argument('--model_config_path', type=str, metavar='FILE', required=True,
                          help=textwrap.dedent("""
                          File path for the paired configurations of the trained model.
                          """ ).strip()) 

    optional.add_argument('--pred_file', type=str, metavar='FILE', default='pred.tsv', help=textwrap.dedent("""
                          Name of the output file for prediction results.
                          Default: 'pred.tsv'.
                          """ ).strip())
        
    optional.add_argument('--calibrator_path', type=str, metavar='FILE', default='',help=textwrap.dedent("""
                          File path for the paired calibrator of the trained model.
                          """ ).strip())
    
    optional.add_argument('--bw_paths', type=str, metavar='FILE', default=None,
                          help=textwrap.dedent("""
                          File path for a list of BigWig files for non-sequence 
                          features such as the coverage track. Default: None.""").strip())
    
    optional.add_argument('--without_h5', default=False, action='store_true',  
                          help=textwrap.dedent("""
                          Do not generate HDF5 file the BED file. If the file is large, 
                          this could take much GPU/CPU memory. Default: False.
                          """).strip())
    
    optional.add_argument('--cpu_only', default=False, action='store_true',  
                          help=textwrap.dedent("""
                          Only use CPU computing. Default: False.
                          """).strip())
        
    optional.add_argument('--pred_batch_size', metavar='INT', default=16, 
                          help=textwrap.dedent("""
                          Size of mini batches for prediction. Default: 16.
                          """ ).strip())
    
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
    
        mural_predict --ref_genome seq.fa --test_data testing.bed.gz \\
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
    test_file = args.test_data   
    ref_genome= args.ref_genome

    pred_batch_size = args.pred_batch_size
    
    # Output file path
    pred_file = args.pred_file
    
    # Whether to generate H5 file for distal data
    without_h5 = args.without_h5
    cpu_only = args.cpu_only

    # Get saved model-related files
    model_path = args.model_path
    model_config_path = args.model_config_path
    calibrator_path = args.calibrator_path

    # Load model config (hyperparameters)
    if model_config_path != '':
        with open(model_config_path, 'rb') as fconfig:
            config = pickle.load(fconfig)
    else:
        print('Error: no model config file provided!')
        sys.exit()
        
    # Set hyperparameters
    local_radius = config['local_radius']
    local_order = config['local_order']
    local_hidden1_size = config['local_hidden1_size']
    local_hidden2_size = config['local_hidden2_size']
    distal_radius = config['distal_radius']
    distal_order = 1 # reserved for future improvement
    CNN_kernel_size = config['CNN_kernel_size']  
    CNN_out_channels = config['CNN_out_channels']
    emb_dropout = config['emb_dropout']
    local_dropout = config['local_dropout']
    distal_fc_dropout = config['distal_fc_dropout']
    emb_dims = config['emb_dims']
    
    n_class = config['n_class']
    model_no = config['model_no']
    seq_only = config['seq_only']
   
    
    start_time = time.time()
    print('Start time:', datetime.datetime.now())
    sys.stdout.flush()
    
    # Read BED files
    test_bed = BedTool(test_file)

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

    # Get the H5 file path for testing data
    test_h5f_path = get_h5f_path(test_file, bw_names, distal_radius, distal_order)

    # Prepare testing data 
    if without_h5:
        #dataset_test = prepare_dataset(test_bed, ref_genome, bw_files, bw_names, local_radius, local_order, distal_radius, distal_order, seq_only)
        dataset_test = prepare_dataset_np(test_bed, ref_genome, bw_files, bw_names, local_radius, local_order, distal_radius, distal_order, seq_only)
        print('using prepare_dataset_np ...')
    else:
        dataset_test = prepare_dataset_h5(test_bed, ref_genome, bw_files, bw_names, local_radius, local_order, distal_radius, distal_order, test_h5f_path, 1, seq_only)
    data_local_test = dataset_test.data_local
    n_cont = len(dataset_test.cont_cols)
    
    test_size = len(dataset_test)

    sys.stdout.flush()
    
    if cpu_only:
        device = torch.device('cpu')
    else:
        # Find a GPU with enough memory
        nvmlInit()
        cuda_id = '0'
        for i in range(nvmlDeviceGetCount()):
            h = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(h)
            if info.free > 2.0*(2**30): # Reserve 2GB GPU memory
                cuda_id = str(i)
                break

        print('CUDA: ', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('using'  , 'cuda:'+cuda_id)
        device = torch.device('cuda:'+cuda_id if torch.cuda.is_available() else 'cpu')
    
    # Choose the network model
    if model_no == 0:
        model = Network0(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[local_hidden1_size, local_hidden2_size], emb_dropout=emb_dropout, lin_layer_dropouts=[local_dropout, local_dropout], n_class=n_class, emb_padding_idx=4**local_order).to(device)
    elif model_no == 1:
        model = Network1(in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=CNN_kernel_size, distal_radius=distal_radius, distal_order=distal_order, distal_fc_dropout=distal_fc_dropout, n_class=n_class).to(device)
    elif model_no == 2:
        model = Network2(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[local_hidden1_size, local_hidden2_size], emb_dropout=emb_dropout, lin_layer_dropouts=[local_dropout, local_dropout], in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=CNN_kernel_size, distal_radius=distal_radius, distal_order=distal_order, distal_fc_dropout=distal_fc_dropout, n_class=n_class, emb_padding_idx=4**local_order).to(device)
    elif model_no == 10:
        model = Network10(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[local_hidden1_size, local_hidden2_size], emb_dropout=emb_dropout, lin_layer_dropouts=[local_dropout, local_dropout], in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=CNN_kernel_size, distal_radius=distal_radius, distal_order=distal_order, distal_fc_dropout=distal_fc_dropout, n_class=n_class, emb_padding_idx=4**local_order).to(device)
    else:
        print('Error: no model selected!')
        sys.exit() 

    print('model:')
    print(model)

    # Load the saved model object
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)
    
    del model_state
    torch.cuda.empty_cache() 

    # Loss function
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    # Set prob names for mutation types
    prob_names = ['prob'+str(i) for i in range(n_class)]

    # Dataloader for testing data    
    if cpu_only:
        dataloader = DataLoader(dataset_test, batch_size=pred_batch_size, shuffle=False, num_workers=0)
    else:
        dataloader = DataLoader(dataset_test, batch_size=pred_batch_size, shuffle=False, num_workers=1)   

    # Do the prediction
    pred_y, test_total_loss = model_predict_m(model, dataloader, criterion, device, n_class, distal=True)
    
    # Print some data for debugging
    print('pred_y:', F.softmax(pred_y[1:10], dim=1))
    for i in range(1, n_class):
        print('min and max of pred_y: type', i, np.min(to_np(F.softmax(pred_y, dim=1))[:,i]), np.max(to_np(F.softmax(pred_y, dim=1))[:,i]))
        
    # Get the predicted probabilities, as the returns of model are logits    
    y_prob = pd.DataFrame(data=to_np(F.softmax(pred_y, dim=1)), columns=prob_names)
    
    # Do probability calibration using saved calibrator
    if calibrator_path != '':
        with open(calibrator_path, 'rb') as fcal:   
            print('using calibrator for scaling ...')
            calibr = pickle.load(fcal)         
            prob_cal = calibr.predict_proba(y_prob.to_numpy())  
            y_prob = pd.DataFrame(data=np.copy(prob_cal), columns=prob_names)
    
    print('Mean Loss, Total Loss, Test Size:', test_total_loss/test_size, test_total_loss, test_size)
    
    # Combine data and do k-mer evaluation
    data_and_prob = pd.concat([data_local_test, y_prob], axis=1)         
    print('3mer correlation: ', freq_kmer_comp_multi(data_and_prob, 3, n_class))
    print('5mer correlation: ', freq_kmer_comp_multi(data_and_prob, 5, n_class))
    print('7mer correlation: ', freq_kmer_comp_multi(data_and_prob, 7, n_class))

    # Write the prediction
    test_pred_df = data_and_prob[['mut_type'] + prob_names]
    pred_df = pd.concat((test_bed.to_dataframe()[['chrom', 'start', 'end', 'strand']], test_pred_df), axis=1)
    pred_df.columns = ['chrom', 'start', 'end', 'strand', 'mut_type'] +  prob_names
    pred_df.to_csv(pred_file, sep='\t', float_format='%.4g', index=False)
    
    # Calculate regional correlations for a few window sizes
    for win_size in [10000, 50000, 200000]:
        corr = corr_calc_sub(pred_df, win_size, prob_names)
        print('regional corr:', str(win_size)+'bp', corr)
  
    print('Total time used: %s seconds' % (time.time() - start_time))
   
    
if __name__ == "__main__":
    main()


