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

from MuRaL.model.nn_utils import *
from MuRaL.model.calibration import poisson_calibrate
from MuRaL.data.preprocessing import *
from MuRaL.evaluation.evaluation import *
from MuRaL.utils.gpu_utils import get_available_gpu, check_cuda_id 

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True

def run_predict_pipline(args, model_type='snv'):
    
    # Set input file
    test_file = args.test_data   
    ref_genome= args.ref_genome

    pred_batch_size = args.pred_batch_size
    sampled_segments = 1
    # Output file path
    pred_file = args.pred_file
    
    # Whether to generate H5 file for distal data
    with_h5 = args.with_h5
    n_h5_files = args.n_h5_files
    cpu_only = args.cpu_only
    cuda_id = args.cuda_id

    # Get saved model-related files
    model_path = args.model_path
    model_config_path = args.model_config_path
    calibrator_path = args.calibrator_path
    
    kmer_corr = args.kmer_corr
    region_corr = args.region_corr

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
    if 'without_bw_distal' in config: 
        without_bw_distal = config['without_bw_distal']
    else:
        without_bw_distal = False
    
    # set segment_center   
    if not args.segment_center:
        args.segment_center = segment_center = config['segment_center']
    else:
        segment_center = args.segment_center

    seq_only = config['seq_only']
    # custom_dataloader = args.custom_dataloader
    # Print command line
    print(' '.join(sys.argv))
    for k,v in vars(args).items():
        print("{0}: {1}".format(k,v))
   
    
    start_time = time.time()
    print('Start time:', datetime.datetime.now())
    sys.stdout.flush()
    
    # Read BED files
    test_bed = BedTool(test_file)

    # Read bigWig file names
    bw_paths = args.bw_paths
    bw_files = []
    bw_names = []
    bw_radii = []
    
    if bw_paths:
        try:
            bw_list = pd.read_table(bw_paths, sep='\s+', header=None, comment='#')
            bw_files = list(bw_list[0])
            bw_names = list(bw_list[1])
            if bw_list.shape[1]>2:
                bw_radii = list(bw_list[2].astype(int))
            else:
                bw_radii = [local_radius]*len(bw_files)
        except pd.errors.EmptyDataError:
            print('Warnings: no bigWig files provided in', bw_paths)
    else:
        print('NOTE: no bigWig files provided.')

    if with_h5:
        print("Warming: recommend don`t used --with_h5", file=sys.stderr)
        dataset = prepare_dataset_h5(test_bed, ref_genome, bw_paths, bw_files, bw_names, bw_radii, \
                                    segment_center, local_radius, local_order, distal_radius, distal_order, \
                                    h5f_path=h5f_path, chunk_size=5000, seq_only=seq_only, n_h5_files=n_h5_files, \
                                    without_bw_distal=without_bw_distal, model_type=model_type)
    else:
        print('using numpy/pandas for distal_seq ...')
        dataset_test = prepare_dataset_np(test_bed, ref_genome, bw_files, bw_names, bw_radii, \
                                     segment_center, local_radius, local_order, distal_radius, distal_order, seq_only=seq_only, model_type=model_type)
        if not dataset_test.distal_info:
            dataset_test.get_distal_encoding_infomation()
    print("test set preprocess time:", (time.time() - start_time))
    data_local_test = dataset_test.data_local.reset_index(drop=True)

    n_cont = len(dataset_test.cont_cols)
    
    test_size = len(data_local_test)

    sys.stdout.flush()
    
    if cpu_only:
        if cuda_id != None:
            print('Warning: --cpu_only is set, but cuda_id is provided. Ignoring cuda_id')
        device = torch.device('cpu')
    else:
        # Find a GPU with enough memory
        if cuda_id == None:
            cuda_id = get_available_gpu(1)
        else:
            check_cuda_id(cuda_id)
        print('CUDA: ', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('using'  , 'cuda:'+cuda_id)
        device = torch.device('cuda:'+cuda_id if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(f'cuda:{cuda_id}')
  
    #####
    if without_bw_distal:
        in_channels = 4**distal_order
    else:
        in_channels = 4**distal_order+n_cont
    #####

    # Choose the network model
    common_model_config = {
        'emb_dims': emb_dims,
        'n_cont': n_cont,
        'n_class': n_class,
        'distal_order': distal_order,
        'in_channels': in_channels,
        }
    model = model_choice(model_no, config, common_model_config, model_type)

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

    segmentLoader_test = DataLoader(dataset_test, 1, shuffle=False, pin_memory=False)
    dataloader= generate_data_batches(segmentLoader_test, sampled_segments, pred_batch_size, shuffle=False)
        

    # Do the prediction
    if not args.pred_time_view:
        pred_y, test_total_loss = model_predict_m(model, dataloader, criterion, device, n_class, distal=True, model_type=model_type)
    else:
        pred_y, test_total_loss = run_time_view_model_predict_m(model, dataloader, criterion, device, n_class, distal=True, model_type=model_type)
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
    
    if args.poisson_calib or model_type == 'indel':
        y_prob = poisson_calibrate(y_prob)
    
    print('Mean Loss, Total Loss, Test Size:', test_total_loss/test_size, test_total_loss, test_size)
    
    # Combine data 
    data_and_prob = pd.concat([data_local_test, y_prob], axis=1)         

    # Write the prediction
    test_pred_df = data_and_prob[['mut_type'] + prob_names]
    chr_pos = get_position_info(test_bed, segment_center)
    pred_df = pd.concat((chr_pos, test_pred_df), axis=1)
    pred_df.columns = ['chrom', 'start', 'end', 'strand', 'mut_type'] +  prob_names
    pred_df.sort_values(['chrom', 'start'], inplace=True)
    pred_df.reset_index(drop=True, inplace=True)
    pred_df.to_csv(pred_file, sep='\t', float_format='%.4g', index=False)
    
    #do k-mer evaluation
    if len(kmer_corr) > 0:
        modes = [i%2 for i in kmer_corr]
        
        if sum(modes) != len(kmer_corr) or min(kmer_corr) < 0:
            print('Warning: please provide odd positive mumbers for k-mer lengths', kmer_corr, '. No k-mer correlation was calculated.')
        else:
            for kmer in kmer_corr:
                print(str(kmer)+'mer correlation: ', freq_kmer_comp_multi(data_and_prob, kmer, n_class))
   
    # Calculate regional correlations for a few window sizes
    #for win_size in [10000, 50000, 200000]:
    if len(region_corr) > 0:
        if min(region_corr) <=0:
            print('Warning: please provide  positive mumbers for window sizes. No regional correlation was calculated.')
        else:      
            pred_df.sort_values(['chrom', 'start'], inplace=True)
            
            for win_size in region_corr:
                corr = corr_calc_sub(pred_df, win_size, prob_names)
                print('regional corr:', str(win_size)+'bp', corr)

    print('Total time used: %s seconds' % (time.time() - start_time))