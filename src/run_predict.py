import warnings
warnings.filterwarnings('ignore',category=FutureWarning)


from pybedtools import BedTool

import sys
import argparse
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


from sklearn import metrics, calibration

from nn_models import *
from nn_utils import *

from preprocessing import *
from evaluation import *
from pynvml import *

#=============
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
#=============


def parse_arguments(parser):
## data
    parser.add_argument('--train_data', type=str, default='merge.95win.A.pos.101bp.19cols.train.30k.bed.gz',
                        help='path for training data')
    
    parser.add_argument('--test_data', type=str, default='merge.95win.A.pos.101bp.19cols.test.30k.bed.gz',
                        help='path for testing data')
    
    parser.add_argument('--train_data_h5f', type=str, default='', help='path for training data in HDF5 format')
    
    parser.add_argument('--test_data_h5f', type=str, default='', help='path for testing data in HDF5 format')
    
    parser.add_argument('--ref_genome', type=str, default='/public/home/licai/DNMML/data/hg19/hg19_ucsc_ordered.fa',
                        help='reference genome')
    
    parser.add_argument('--bw_paths', type=str, default='', help='path for the list of BigWig files for non-sequence features')
    
    parser.add_argument('--seq_only', default=False, action='store_true')

    
    parser.add_argument('--n_class', type=int, default=4, help='number of mutation classes')
    
    parser.add_argument('--local_radius', type=int, default=5, help='radius of local sequences to be considered')
    parser.add_argument('--local_order', type=int, default=1, help='order of local sequences to be considered')
    
    parser.add_argument('--local_hidden1_size', type=int, default=150, nargs='+', help='size of 1st hidden layer for local data')
    
    parser.add_argument('--local_hidden2_size', type=int, default=80, nargs='+', help='size of 2nd hidden layer for local data')
    
    parser.add_argument('--distal_radius', type=int, default=50, help='radius of distal sequences to be considered')
    
    parser.add_argument('--distal_order', type=int, default=1, help='order of distal sequences to be considered')
        
    parser.add_argument('--pred_batch_size', type=int, default=10, help='size of mini batches for test data')
    
    parser.add_argument('--CNN_kernel_size', type=int, default=3, help='kernel size for CNN layers')
    
    parser.add_argument('--CNN_out_channels', type=int, default=32, help='number of output channels for CNN layers')
    
    parser.add_argument('--distal_fc_dropout', type=float, default=0.25,  help='dropout rate for distal fc layer')
    
    parser.add_argument('--model_no', type=int, default=2, help=' which NN model to be used')
    
    parser.add_argument('--pred_file', type=str, default='pred.tsv', help='Output file for saving predictions')
            
    parser.add_argument('--model_path', type=str, default='', help='model path')
    
    parser.add_argument('--optim', default=False, action='store_true')
    
    parser.add_argument('--calibrator_path', type=str, default='', help='calibrator path')
    
    parser.add_argument('--model_config_path', type=str, default='', help='model config path')
    
    args = parser.parse_args()

    return args
def main():
    parser = argparse.ArgumentParser(description='Mutation rate modeling using machine learning')
    args = parse_arguments(parser)

    print(' '.join(sys.argv))
    
    #torch.cuda.set_device(1) 
    
    # Set train file
    #train_file = args.train_data
    test_file = args.test_data   
    #train_h5f_path = args.train_data_h5f
    test_h5f_path = args.test_data_h5f   
    ref_genome= args.ref_genome

    pred_batch_size = args.pred_batch_size
  
    #model_no = args.model_no   
    pred_file = args.pred_file   

    model_path = args.model_path
    model_config_path = args.model_config_path
    optim = args.optim
    calibrator_path = args.calibrator_path
    #seq_only = args.seq_only 
    
    #n_class = args.n_class
    
    if model_config_path != '':
        with open(model_config_path, 'rb') as fconfig:
            config = pickle.load(fconfig)
    
    local_radius = config['local_radius']
    local_order = config['local_order']
    local_hidden1_size = config['local_hidden1_size']
    local_hidden2_size = config['local_hidden2_size']
    distal_radius = config['distal_radius']
    distal_order = 1 # reserved 
    CNN_kernel_size = config['CNN_kernel_size']  
    CNN_out_channels = config['CNN_out_channels']
    emb_dropout = config['emb_dropout']
    local_dropout = config['local_dropout']
    distal_fc_dropout = config['distal_fc_dropout']
    emb_dims = config['emb_dims']
    
        
    #n_cont = config['n_cont']
    n_class = config['n_class']
    model_no = config['model_no']
    #bw_paths = config['bw_paths']
    seq_only = config['seq_only']
   
    
    start_time = time.time()
    print('Start time:', datetime.datetime.now())
    
    # Read BED files
    #train_bed = BedTool(train_file)
    test_bed = BedTool(test_file)

    # Read bigWig file names
    bw_paths = args.bw_paths
    bw_files = []
    bw_names = []
    n_cont = 0
    try:
        bw_list = pd.read_table(bw_paths, sep='\s+', header=None, comment='#')
        bw_files = list(bw_list[0])
        bw_names = list(bw_list[1])
        n_cont = len(bw_names)
    except pd.errors.EmptyDataError:
        print('Warnings: no bigWig files provided')

    if len(test_h5f_path) == 0:
        test_h5f_path = test_file + '.distal_' + str(distal_radius)
        if(distal_order >1):
            test_h5f_path = test_h5f_path + '_' + str(distal_order)
        if len(bw_names) > 0:
            test_h5f_path = test_h5f_path + '.' + '.'.join(list(bw_names))
        test_h5f_path = test_h5f_path + '.h5'   

    # Prepare testing data 
    dataset_test = prepare_dataset(test_bed, ref_genome, bw_files, bw_names, local_radius, local_order, distal_radius, distal_order, test_h5f_path, 1)
    data_local_test = dataset_test.data_local
    
    test_size = len(dataset_test)

    # Dataloader for testing data
    dataloader1 = DataLoader(dataset_test, batch_size=pred_batch_size, shuffle=False, num_workers=1)

    # Find a GPU with enough memory
    nvmlInit()
    cuda_id = '0'
    for i in range(nvmlDeviceGetCount()):
        h = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(h)
        if info.free > 1.5*(2**30): #reserve 1.5GB
            cuda_id = str(i)
            break
        
    print("CUDA: ", torch.cuda.is_available())
    print("using  ", "cuda:"+cuda_id)
    device = torch.device("cuda:"+cuda_id if torch.cuda.is_available() else "cpu")
    
    # Choose the network model
    if model_no == 0:
        model = Network0(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[local_hidden1_size, local_hidden2_size], emb_dropout=emb_dropout, lin_layer_dropouts=[local_dropout, local_dropout], n_class=n_class, emb_padding_idx=4**local_order).to(device)

    elif model_no == 1:
        model = Network1(in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=CNN_kernel_size, distal_radius=distal_radius, distal_order=distal_order, distal_fc_dropout=distal_fc_dropout, n_class=n_class, emb_padding_idx=4**local_order).to(device)

    elif model_no == 2:
        model = Network2(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[local_hidden1_size, local_hidden2_size], emb_dropout=emb_dropout, lin_layer_dropouts=[local_dropout, local_dropout], in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=CNN_kernel_size, distal_radius=distal_radius, distal_order=distal_order, distal_fc_dropout=distal_fc_dropout, n_class=n_class, emb_padding_idx=4**local_order).to(device)


    else:
        print('Error: no model selected!')
        sys.exit() 

    print('model:')
    print(model)


    if optim:
        model_state, optimizer_state = torch.load(model_path, map_location=device)
    else:
        model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)
    #model2.load_state_dict(torch.load(model2_path, map_location=device))

    #criterion = torch.nn.NLLLoss()
    criterion = torch.nn.CrossEntropyLoss()
    
    best_loss = 0
    pred_df = None
    test_pred_df = None


    prob_names = ['prob'+str(i) for i in range(n_class)]
    
    #print("1. current CUDA:", torch.cuda.current_device())

    pred_y, test_total_loss = model_predict_m(model, dataloader1, criterion, device, n_class, distal=True)
    #print('pred_y:', torch.exp(pred_y[1:10]))
    print('pred_y:', F.softmax(pred_y[1:10], dim=1))
    #print('pred_y2:', torch.exp(pred_y2[1:10]))
    
    #y_prob = pd.Series(data=to_np(torch.exp(pred_y)).T[1], name="prob")    
    y_prob = pd.DataFrame(data=to_np(F.softmax(pred_y, dim=1)), columns=prob_names)
    
    #######
    if calibrator_path != '':
        with open(calibrator_path, 'rb') as fcal:
            
            print('using calibrator for scaling ...')
            calibr = pickle.load(fcal)
            
            prob_cal = calibr.predict_proba(y_prob.to_numpy())
            
            y_prob = pd.DataFrame(data=np.copy(prob_cal), columns=prob_names)
    #######
    
    data_and_prob = pd.concat([data_local_test, y_prob], axis=1)        
    
    #print("2. current CUDA:", torch.cuda.current_device())
    # For FeedForward-only model  
    #y_prob2 = pd.DataFrame(data=to_np(torch.exp(pred_y2)), columns=prob_names)
    #data_and_prob2 = pd.concat([data_local_test, y_prob2], axis=1)
    
    print('3mer correlation: ', freq_kmer_comp_multi(data_and_prob, 3, n_class))
    print('5mer correlation: ', freq_kmer_comp_multi(data_and_prob, 5, n_class))
    print('7mer correlation: ', freq_kmer_comp_multi(data_and_prob, 7, n_class))

    #print('3mer correlation(FF only): ', freq_kmer_comp_multi(data_and_prob2, 3, n_class))
    #print('5mer correlation(FF only): ', freq_kmer_comp_multi(data_and_prob2, 5, n_class))
    #print('7mer correlation(FF only): ', freq_kmer_comp_multi(data_and_prob2, 7, n_class))

    test_pred_df = data_and_prob[['mut_type'] + prob_names]
    #test_pred_df2 = data_and_prob2[['mut_type'] + prob_names]

    #torch.save(model.state_dict(), pred_file+'.model1')
    #torch.save(model2.state_dict(), pred_file+'.model2')

    # Get the scores
    #auc_score = metrics.roc_auc_score(to_np(test_y), to_np(pred_y))
    test_y = data_local_test['mut_type']

    # Print some data for debugging
    for i in range(1, n_class):
        print('min and max of pred_y: type', i, np.min(to_np(F.softmax(pred_y, dim=1))[:,i]), np.max(to_np(F.softmax(pred_y, dim=1))[:,i]))
        #print('min and max of pred_y2: type', i, np.min(to_np(torch.exp(pred_y2))[:,i]), np.max(to_np(torch.exp(pred_y2))[:,i]))

    # Write the prediction
    pred_df = pd.concat((test_bed.to_dataframe()[['chrom', 'start', 'end', 'strand']], test_pred_df), axis=1)
    pred_df.columns = ['chrom', 'start', 'end', 'strand', 'mut_type'] +  prob_names

    pred_df.to_csv(pred_file, sep='\t', index=False)
    
    for win_size in [10000, 50000, 200000]:
        corr1 = corr_calc_sub(pred_df, win_size, prob_names)
        #corr2 = corr_calc_sub(pred_df, win_size, [name+'_M2' for name in prob_names])
        print('regional corr:', str(win_size)+'bp', corr1)
        #print('regional corr2:', str(win_size)+'bp', corr2)
    #os.remove(train_h5f_path)
    #os.remove(test_h5f_path)

    print('Total time used: %s seconds' % (time.time() - start_time))
   
    
if __name__ == "__main__":
    main()


