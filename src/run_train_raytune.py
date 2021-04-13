"""
Code for training models with RayTune
"""

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from pybedtools import BedTool

import sys
import argparse
import pandas as pd
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

#=============
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True
#=============

from functools import partial
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import os
import time
import datetime

from nn_models import *
from nn_utils import *
from preprocessing import *
from evaluation import *

from torch.utils.tensorboard import SummaryWriter

def parse_arguments(parser):
    """
    Parse parameters from the command line
    """

    parser.add_argument('--train_data', type=str, default='merge.95win.A.pos.101bp.19cols.train.30k.bed.gz',
                        help='path for training data')
    
    parser.add_argument('--test_data', type=str, default='merge.95win.A.pos.101bp.19cols.test.30k.bed.gz',
                        help='path for testing data')
    
    parser.add_argument('--ref_genome', type=str, default='/public/home/licai/DNMML/data/hg19/hg19_ucsc_ordered.fa',
                        help='reference genome')
    
    parser.add_argument('--bw_paths', type=str, default='', help='path for the list of BigWig files for non-sequence features')
    
    parser.add_argument('--seq_only', default=False, action='store_true', help='use only genomic sequence and ignore bigWig tracks')
    
    parser.add_argument('--n_class', type=int, default='4', help='number of mutation classes')
    
    parser.add_argument('--local_radius', type=int, default=[5], nargs='+', help='radius of local sequences to be considered')
    
    parser.add_argument('--local_order', type=int, default=[1], nargs='+', help='order of local sequences to be considered')
    
    parser.add_argument('--local_hidden1_size', type=int, default=[150], nargs='+', help='size of 1st hidden layer for local data')
    
    parser.add_argument('--local_hidden2_size', type=int, default=[80], nargs='+', help='size of 2nd hidden layer for local data')
    
    parser.add_argument('--distal_radius', type=int, default=[50], nargs='+', help='radius of distal sequences to be considered')
    
    parser.add_argument('--distal_order', type=int, default=1, help='order of distal sequences to be considered')
    
    #parser.add_argument('--emb_4th_root', default=False, action='store_true')
    
    parser.add_argument('--batch_size', type=int, default=[128], nargs='+', help='size of mini batches')
    
    parser.add_argument('--emb_dropout', type=float, default=[0.2], nargs='+', help='dropout rate for k-mer embedding')
    
    parser.add_argument('--local_dropout', type=float, default=[0.15], nargs='+', help='dropout rate for local network')
    
    parser.add_argument('--CNN_kernel_size', type=int, default=[3], nargs='+', help='kernel size for CNN layers')
    
    parser.add_argument('--CNN_out_channels', type=int, default=[32], nargs='+', help='number of output channels for CNN layers')
    
    parser.add_argument('--distal_fc_dropout', type=float, default=[0.25], nargs='+', help='dropout rate for distal fc layer')
    
    
    parser.add_argument('--model_no', type=int, default=2, help='which NN model to be used')
    
    #parser.add_argument('--pred_file', type=str, default='pred.tsv', help='Output file for saving predictions')
    
    parser.add_argument('--optim', type=str, default=['Adam'], nargs='+', help='Optimization method')
    
    parser.add_argument('--cuda_id', type=str, default='0', help='the GPU to be used')
    
    parser.add_argument('--valid_ratio', type=float, default='0.2', help='the ratio of validation data relative to the whole training data')
    
    parser.add_argument('--learning_rate', type=float, default=[0.005], nargs='+', help='learning rate for training')
    
    parser.add_argument('--weight_decay', type=float, default=[1e-5], nargs='+', help='weight decay (regularization) for training')
    
    parser.add_argument('--LR_gamma', type=float, default=[0.5], nargs='+', help='gamma for learning rate change during training')
    
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
    
    parser.add_argument('--grace_period', type=int, default=5, help='grace_period for early stopping')
    
    
    parser.add_argument('--n_trials', type=int, default=3, help='number of trials for training')
    
    parser.add_argument('--experiment_name', type=str, default='my_experiment', help='Ray.Tune experiment name')
    
    parser.add_argument('--ray_ncpus', type=int, default=6, help='number of CPUs used by Ray')
    
    parser.add_argument('--ray_ngpus', type=int, default=1, help='number of GPUs used by Ray')
        
    parser.add_argument('--resume_ray', default=False, action='store_true', help='resume incomplete Ray experiment')
    
    args = parser.parse_args()

    return args
def main():
    
    #parse the command line
    parser = argparse.ArgumentParser(description='Mutation rate modeling using machine learning')
    args = parse_arguments(parser)
    
    start_time = time.time()
    print('Start time:', datetime.datetime.now())
    
    print(' '.join(sys.argv)) # print the command line
    train_file = args.train_data
    ref_genome= args.ref_genome
    local_radius = args.local_radius
    local_order = args.local_order
    local_hidden1_size = args.local_hidden1_size
    local_hidden2_size = args.local_hidden2_size
    distal_radius = args.distal_radius  
    distal_order = args.distal_order
    batch_size = args.batch_size 
    emb_dropout = args.emb_dropout
    local_dropout = args.local_dropout
    CNN_kernel_size = args.CNN_kernel_size   
    CNN_out_channels = args.CNN_out_channels
    distal_fc_dropout = args.distal_fc_dropout
    model_no = args.model_no   
    #pred_file = args.pred_file   
    optim = args.optim
    learning_rate = args.learning_rate   
    weight_decay = args.weight_decay  
    LR_gamma = args.LR_gamma  
    epochs = args.epochs
    grace_period = args.grace_period
    n_trials = args.n_trials
    experiment_name = args.experiment_name
    n_class = args.n_class  
    cuda_id = args.cuda_id
    valid_ratio = args.valid_ratio
    resume_ray = args.resume_ray
    ray_ncpus = args.ray_ncpus
    ray_ngpus = args.ray_ngpus
    
    
    # Read bigWig file names
    bw_paths = args.bw_paths
    bw_files = []
    bw_names = []
    
    try:
        bw_list = pd.read_table(bw_paths, sep='\s+', header=None, comment='#')
        bw_files = list(bw_list[0])
        bw_names = list(bw_list[1])
    except pd.errors.EmptyDataError:
        print('Warnings: no bigWig files provided')
    
    # Prepare min/max for the loguniform samplers if one value is provided
    if len(learning_rate) == 1:
        learning_rate = learning_rate*2
    if len(weight_decay) == 1:
        weight_decay = weight_decay*2
    
    # Read the train datapoints
    train_bed = BedTool(train_file)
    
    # Generate H5 files for storing distal regions before training, one file for each possible distal radius
    for d_radius in distal_radius:
        h5f_path = get_h5f_path(train_file, bw_names, d_radius, distal_order)
        generate_h5f(train_bed, h5f_path, ref_genome, d_radius, distal_order, bw_files, 1)
    
    # Set visible GPU(s)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id 
    
    # Allocate CPU/GPU resources for this Ray job
    ray.init(num_cpus=ray_ncpus, num_gpus=ray_ngpus, dashboard_host="0.0.0.0")
    
    # Configure the search space for relavant hyperparameters
    config = {
        'local_radius': tune.grid_search(local_radius),
        'local_order': tune.choice(local_order),
        'local_hidden1_size': tune.choice(local_hidden1_size),
        'local_hidden2_size': tune.choice(local_hidden2_size),
        'distal_radius': tune.choice(distal_radius),
        'emb_dropout': tune.choice(emb_dropout),
        'local_dropout': tune.choice(local_dropout),
        'CNN_kernel_size': tune.choice(CNN_kernel_size),
        'CNN_out_channels': tune.choice(CNN_out_channels),
        'distal_fc_dropout': tune.choice(distal_fc_dropout),
        'batch_size': tune.choice(batch_size),
        'learning_rate': tune.loguniform(learning_rate[0], learning_rate[1]),
        #'learning_rate': tune.choice(learning_rate),
        'optim': tune.choice(optim),
        'LR_gamma': tune.choice(LR_gamma),
        'weight_decay': tune.loguniform(weight_decay[0], weight_decay[1]),
        #'weight_decay': tune.choice(weight_decay),
    }
    
    # Set the scheduler for parallel training 
    scheduler = ASHAScheduler(
    #metric='loss',
    metric='score', # Use the custom score metric for model selection
    mode='min',
    max_t=epochs,
    grace_period=grace_period,
    reduction_factor=2)
    
    # Information to be shown in the progress table
    reporter = CLIReporter(parameter_columns=['local_radius', 'local_order', 'local_hidden1_size', 'local_hidden2_size', 'distal_radius', 'emb_dropout', 'local_dropout', 'CNN_kernel_size', 'CNN_out_channels', 'distal_fc_dropout', 'optim', 'learning_rate', 'weight_decay', 'LR_gamma', ], metric_columns=['loss', 'fdiri_loss', 'score', 'training_iteration'])
    
    trainable_id = 'Train'
    tune.register_trainable(trainable_id, partial(train, args=args))
    
    # Execute the training
    result = tune.run(
    trainable_id,
    name=experiment_name,
    resources_per_trial={'cpu': 3, 'gpu': 0.2},
    config=config,
    num_samples=n_trials,
    local_dir='./ray_results',
    scheduler=scheduler,
    progress_reporter=reporter,
    resume=resume_ray)
    
    # Print the best trial at the ende
    #best_trial = result.get_best_trial('loss', 'min', 'last')
    best_trial = result.get_best_trial('loss', 'min', 'last-5-avg')
    print('Best trial config: {}'.format(best_trial.config))
    print('Best trial final validation loss: {}'.format(best_trial.last_result['loss'])) 
    
    best_checkpoint = result.get_best_checkpoint(best_trial, metric='loss', mode='min')
    print('best_checkpoint:', best_checkpoint)
    
    # Shutdown Ray
    if ray.is_initialized():
        ray.shutdown() 

def train(config, args, checkpoint_dir=None):
    """
    Training funtion.
    
    Args:
        config: configuration of hyperparameters
        args: input args from the command line
        checkpoint_dir: checkpoint dir
    """

    # Get parameters from the command line
    train_file = args.train_data
    ref_genome= args.ref_genome
    local_radius = args.local_radius
    local_order = args.local_order
    distal_radius = args.distal_radius  
    distal_order = args.distal_order
    batch_size = args.batch_size 
    local_dropout = args.local_dropout
    CNN_kernel_size = args.CNN_kernel_size   
    CNN_out_channels = args.CNN_out_channels
    distal_fc_dropout = args.distal_fc_dropout
    model_no = args.model_no   
    #pred_file = args.pred_file   
    optim = args.optim
    learning_rate = args.learning_rate   
    weight_decay = args.weight_decay  
    LR_gamma = args.LR_gamma  
    epochs = args.epochs
    n_class = args.n_class  
    cuda_id = args.cuda_id
    valid_ratio = args.valid_ratio
    seq_only = args.seq_only 
    
    bw_paths = args.bw_paths
    bw_files = []
    bw_names = []
    try:
        bw_list = pd.read_table(bw_paths, sep='\s+', header=None, comment='#')
        bw_files = list(bw_list[0])
        bw_names = list(bw_list[1])
    except pd.errors.EmptyDataError:
        print('Warnings: no bigWig files provided')

    # Read BED files
    train_bed = BedTool(train_file)
    
    # Get the H5 file path
    train_h5f_path = get_h5f_path(train_file, bw_names, config['distal_radius'], distal_order)
    
    # Prepare the datasets for trainging
    dataset = prepare_dataset(train_bed, ref_genome, bw_files, bw_names, config['local_radius'], config['local_order'], config['distal_radius'], distal_order, train_h5f_path, seq_only=seq_only)
    
    data_local = dataset.data_local
    categorical_features = dataset.cat_cols
    n_cont = len(dataset.cont_cols)
    
    #config['n_cont'] = n_cont
    config['n_class'] = n_class
    config['model_no'] = model_no
    #config['bw_paths'] = bw_paths
    config['seq_only'] = seq_only
    #print('n_cont: ', n_cont)
    
    # Set the device
    print('CUDA is available: ', torch.cuda.is_available())
    #device = torch.device('cuda:'+cuda_id if torch.cuda.is_available() else 'cpu')    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split the data into two parts - training data and validation data
    valid_size = int(len(dataset)*valid_ratio)
    train_size = len(dataset) - valid_size
    print('train_size, valid_size:', train_size, valid_size)
    
    dataset_train, dataset_valid = random_split(dataset, [train_size, valid_size])
    dataset_valid.indices.sort()
    #data_local_valid = dataset_valid.data_local
    
    # Dataloader for training
    dataloader_train = DataLoader(dataset_train, config['batch_size'], shuffle=True, num_workers=2, pin_memory=True) #shuffle=False for HybridLoss

    # Dataloader for predicting
    dataloader_valid = DataLoader(dataset_valid, config['batch_size'], shuffle=False, num_workers=1, pin_memory=True)

    
    # Number of categorical features
    #cat_dims = [int(data_local[col].nunique()) for col in categorical_features]
    cat_dims = dataset.cat_dims
    
    # Set embedding dimensions for categorical features
    # According to https://stackoverflow.com/questions/48479915/what-is-the-preferred-ratio-between-the-vocabulary-size-and-embedding-dimension
    emb_dims = [(x, min(16, int(x**0.25))) for x in cat_dims] 
    config['emb_dims'] = emb_dims


    # Choose the network model for training
    if model_no == 0:
        # Local-only model
        model = Network0(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], n_class=n_class, emb_padding_idx=4**config['local_order']).to(device)

    elif model_no == 1:
        # ResNet model
        model = Network1(in_channels=4**distal_order+n_cont, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'],  distal_radius=config['distal_radius'], distal_order=distal_order, distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class).to(device)

    elif model_no == 2:
        # Combined model
        model = Network2(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], in_channels=4**distal_order+n_cont, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'], distal_radius=config['distal_radius'], distal_order=distal_order, distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order']).to(device)

    else:
        print('Error: no model selected!')
        sys.exit() 
    
    # Count the parameters in the model
    count_parameters(model)
    print('model:')
    print(model)
    
    '''
    writer = SummaryWriter('runs/test')
    dataiter = iter(dataloader_train)
    y, cont_x, cat_x, distal_x = dataiter.next()
    cat_x = cat_x.to(device)
    cont_x = cont_x.to(device)
    distal_x = distal_x.to(device)
    writer.add_graph(model, ((cont_x, cat_x), distal_x))
    writer.close()
    '''

    # Initiating weights of the models;
    model.apply(weights_init)
    
    # Set loss function
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    # Set Optimizer
    if config['optim'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    elif config['optim'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        
    elif config['optim'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'], momentum=0.98, nesterov=True)
     
    else:
        print('Error: unsupported optimization method', config['optim'])
        sys.exit()


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config['LR_gamma'])

    print('optimizer:', optimizer)
    print('scheduler:', scheduler)

    prob_names = ['prob'+str(i) for i in range(n_class)]
    
    # Training loop
    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for y, cont_x, cat_x, distal_x in dataloader_train:
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            distal_x = distal_x.to(device)
            y  = y.to(device)


            # Forward Pass
            preds = model.forward((cont_x, cat_x), distal_x)
            loss = criterion(preds, y.long().squeeze())
                   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Flush StdOut buffer
        sys.stdout.flush()
        
        print('optimizer learning rate:', optimizer.param_groups[0]['lr'])
        # Update learning rate
        scheduler.step()
        
        model.eval()
        with torch.no_grad():

            valid_pred_y, valid_total_loss = model_predict_m(model, dataloader_valid, criterion, device, n_class, distal=True)

            valid_y_prob = pd.DataFrame(data=to_np(F.softmax(valid_pred_y, dim=1)), columns=prob_names)
            valid_data_and_prob = pd.concat([data_local.iloc[dataset_valid.indices, ].reset_index(drop=True), valid_y_prob], axis=1)    
            
            valid_y = valid_data_and_prob['mut_type'].to_numpy().squeeze()
            
            # Train the calibrator using the validataion data
            fdiri_cal, fdiri_nll = calibrate_prob(valid_y_prob.to_numpy(), valid_y, device, calibr_name='FullDiri')
            #fdirio_cal, _ = calibrate_prob(valid_y_prob.to_numpy(), valid_y, device, calibr_name='FullDiriODIR')
            #vec_cal, _ = calibrate_prob(valid_y_prob.to_numpy(), valid_y, device, calibr_name='VectS')
            #tmp_cal, _ = calibrate_prob(valid_y_prob.to_numpy(), valid_y, device, calibr_name='TempS')
            
            # Compare observed/predicted 3/5/7mer mutation frequencies
            print('3mer correlation - all: ', freq_kmer_comp_multi(valid_data_and_prob, 3, n_class))
            print('5mer correlation - all: ', freq_kmer_comp_multi(valid_data_and_prob, 5, n_class))
            print('7mer correlation - all: ', freq_kmer_comp_multi(valid_data_and_prob, 7, n_class))
            
            print ('Validation Loss: ', valid_total_loss/valid_size)
            print ('Validation Loss (after fdiri_cal): ', fdiri_nll) 
            
            # Calculate a custom score by looking obs/pred 3/5-mer correlations in binned windows
            region_size = 10000
            n_regions = valid_size//region_size
            print('n_regions:', n_regions)
            
            score = 0
            corr_3mer = []
            corr_5mer = []
            
            for i in range(n_regions):
                corr_3mer = freq_kmer_comp_multi(valid_data_and_prob.iloc[region_size*i:region_size*(i+1), ], 3, n_class)    
                corr_5mer = freq_kmer_comp_multi(valid_data_and_prob.iloc[region_size*i:region_size*(i+1), ], 5, n_class)
                
                score += np.sum([(1-corr)**2 for corr in corr_3mer]) + np.sum([(1-corr)**2 for corr in corr_5mer])
           
            print('corr_3mer:', corr_3mer)
            print('corr_5mer:', corr_5mer)
            print('regional score:', score, n_regions)
            
            # Output genomic positions and predicted probabilities
            chr_pos = train_bed.to_dataframe().loc[dataset_valid.indices,['chrom', 'start', 'end']].reset_index(drop=True)
            valid_pred_df = pd.concat((chr_pos, valid_data_and_prob[['mut_type'] + prob_names]), axis=1)
            valid_pred_df.columns = ['chrom', 'start', 'end','mut_type'] + prob_names
            
            print('valid_pred_df: ', valid_pred_df.head())
            
            # Print regional correlations
            for win_size in [20000, 100000, 500000]:
                corr_win = corr_calc_sub(valid_pred_df, win_size, prob_names)
                print('regional corr (validation):', str(win_size)+'bp', corr_win)
            
            # Save model data for each checkpoint
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, 'model')
                torch.save(model.state_dict(), path)
            
                with open(path + '.fdiri_cal.pkl', 'wb') as pkl_file:
                    pickle.dump(fdiri_cal, pkl_file)
                
                with open(path + '.config.pkl', 'wb') as fp:
                    pickle.dump(config, fp)


            tune.report(loss=valid_total_loss/valid_size, fdiri_loss=fdiri_nll, score=score)
                
    
if __name__ == '__main__':
    main()


