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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True

from functools import partial
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import os
import time
import datetime
import random

from nn_models import *
from nn_utils import *
from preprocessing import *
from evaluation import *
from training import *

from torch.utils.tensorboard import SummaryWriter

def parse_arguments(parser):
    """
    Parse parameters from the command line
    """   
    parser.add_argument('--ref_genome', type=str, default='', help='reference genome')
    
    parser.add_argument('--train_data', type=str, default='', help='path for training data')
    
    parser.add_argument('--validation_data', type=str, default='', help='path for validation data')    
    parser.add_argument('--bw_paths', type=str, default='', help='path for the list of BigWig files for non-sequence features')
    
    parser.add_argument('--seq_only', default=False, action='store_true', help='use only genomic sequence and ignore bigWig tracks')
    
    parser.add_argument('--n_class', type=int, default='4', help='number of mutation classes')
    
    parser.add_argument('--local_radius', type=int, default=[5], nargs='+', help='radius of local sequences to be considered')
    
    parser.add_argument('--local_order', type=int, default=[1], nargs='+', help='order of local sequences to be considered')
    
    parser.add_argument('--local_hidden1_size', type=int, default=[150], nargs='+', help='size of 1st hidden layer for local data')
    
    parser.add_argument('--local_hidden2_size', type=int, default=[0], nargs='+', help='size of 2nd hidden layer for local data')
    
    parser.add_argument('--distal_radius', type=int, default=[50], nargs='+', help='radius of distal sequences to be considered')
    
    parser.add_argument('--distal_order', type=int, default=1, help='order of distal sequences to be considered')
    
    #parser.add_argument('--emb_4th_root', default=False, action='store_true')
    
    parser.add_argument('--batch_size', type=int, default=[128], nargs='+', help='size of mini batches')
    
    parser.add_argument('--emb_dropout', type=float, default=[0.1], nargs='+', help='dropout rate for k-mer embedding')
    
    parser.add_argument('--local_dropout', type=float, default=[0.1], nargs='+', help='dropout rate for local network')
    
    parser.add_argument('--CNN_kernel_size', type=int, default=[3], nargs='+', help='kernel size for CNN layers')
    
    parser.add_argument('--CNN_out_channels', type=int, default=[32], nargs='+', help='number of output channels for CNN layers')
    
    parser.add_argument('--distal_fc_dropout', type=float, default=[0.25], nargs='+', help='dropout rate for distal fc layer')
    
    
    parser.add_argument('--model_no', type=int, default=2, help='which NN model to be used')
    
    #parser.add_argument('--pred_file', type=str, default='pred.tsv', help='Output file for saving predictions')
    
    parser.add_argument('--optim', type=str, default=['Adam'], nargs='+', help='Optimization method')
    
    parser.add_argument('--cuda_id', type=str, default='0', help='the GPU to be used')
    
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='the ratio of validation data relative to the whole training data')
    
    parser.add_argument('--split_seed', type=int, default=-1, help='seed for randomly splitting data into training and validation sets')
    
    parser.add_argument('--learning_rate', type=float, default=[0.005], nargs='+', help='learning rate for training')
    
    parser.add_argument('--weight_decay', type=float, default=[1e-5], nargs='+', help='weight decay (regularization) for training')
    
    parser.add_argument('--LR_gamma', type=float, default=[0.5], nargs='+', help='gamma for learning rate change during training')
    
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
    
    parser.add_argument('--grace_period', type=int, default=5, help='grace_period for early stopping')
    
    
    parser.add_argument('--n_trials', type=int, default=3, help='number of trials for training')
    
    parser.add_argument('--experiment_name', type=str, default='my_experiment', help='Ray.Tune experiment name')
    
    parser.add_argument('--ASHA_metric', type=str, default='loss', help='metric for ASHA schedualing; the value can be "loss" or "score"')
    
    parser.add_argument('--ray_ncpus', type=int, default=6, help='number of CPUs requested by Ray')
    
    parser.add_argument('--ray_ngpus', type=int, default=1, help='number of GPUs requested by Ray')
    
    parser.add_argument('--cpu_per_trial', type=int, default=3, help='number of CPUs per trial')
    
    parser.add_argument('--gpu_per_trial', type=float, default=0.19, help='number of GPUs per trial')
        
    parser.add_argument('--save_valid_preds', default=False, action='store_true', help='Save prediction results for validation data')
    
    parser.add_argument('--rerun_failed', default=False, action='store_true', help='Rerun failed trials')
    
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
    valid_file = args.validation_data
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
    ASHA_metric = args.ASHA_metric
    n_class = args.n_class  
    cuda_id = args.cuda_id
    valid_ratio = args.valid_ratio
    save_valid_preds = args.save_valid_preds
    rerun_failed = args.rerun_failed
    ray_ncpus = args.ray_ncpus
    ray_ngpus = args.ray_ngpus
    cpu_per_trial = args.cpu_per_trial
    gpu_per_trial = args.gpu_per_trial
    
    if args.split_seed < 0:
        args.split_seed = random.randint(0, 1000000)
    print('args.split_seed:', args.split_seed)
    
    
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
    
    if valid_file != '':
        valid_bed = BedTool(valid_file)
        for d_radius in distal_radius:
            valid_h5f_path = get_h5f_path(valid_file, bw_names, d_radius, distal_order)
            generate_h5f(valid_bed, valid_h5f_path, ref_genome, d_radius, distal_order, bw_files, 1)
    
    if ray_ngpus > 0 or gpu_per_trial > 0:
        if not torch.cuda.is_available():
            print('Error: You requested GPU computing, but CUDA is not available! If you want to run without GPU, please set "--ray_ngpus 0 --gpu_per_trial 0"', file=sys.stderr)
            sys.exit()
        # Set visible GPU(s)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
        print('Ray is using GPU device', 'cuda:'+cuda_id)
    else:
        print('Ray is using only CPUs ...')
    
    if rerun_failed:
        resume_flag = 'ERRORED_ONLY'
    else:
        resume_flag = False
    
    # Allocate CPU/GPU resources for this Ray job
    ray.init(num_cpus=ray_ncpus, num_gpus=ray_ngpus, dashboard_host="0.0.0.0")
    
    sys.stdout.flush()
    
    # Configure the search space for relavant hyperparameters
    config = {
        'local_radius': tune.choice(local_radius),
        'local_order': tune.choice(local_order),
        'local_hidden1_size': tune.choice(local_hidden1_size),
        #'local_hidden2_size': tune.choice(local_hidden2_size),
        'local_hidden2_size': tune.choice(local_hidden2_size) if local_hidden2_size[0]>0 else tune.sample_from(lambda spec: spec.config.local_hidden1_size//2), # default local_hidden2_size = local_hidden1_size//2
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
        'transfer_learning': False,
    }
    

    # Set the scheduler for parallel training 
    scheduler = ASHAScheduler(
    #metric='loss',
    metric=ASHA_metric, # Use a metric for model selection
    mode='min',
    max_t=epochs,
    grace_period=grace_period,
    reduction_factor=2)
    
    # Information to be shown in the progress table
    reporter = CLIReporter(parameter_columns=['local_radius', 'local_order', 'local_hidden1_size', 'local_hidden2_size', 'distal_radius', 'emb_dropout', 'local_dropout', 'CNN_kernel_size', 'CNN_out_channels', 'distal_fc_dropout', 'optim', 'learning_rate', 'weight_decay', 'LR_gamma', ], metric_columns=['loss', 'fdiri_loss', 'score', 'total_params', 'training_iteration'])
    
    trainable_id = 'Train'
    tune.register_trainable(trainable_id, partial(train, args=args))
    
    # Execute the training
    result = tune.run(
    trainable_id,
    name=experiment_name,
    resources_per_trial={'cpu': cpu_per_trial, 'gpu': gpu_per_trial},
    config=config,
    num_samples=n_trials,
    local_dir='./ray_results',
    scheduler=scheduler,
    progress_reporter=reporter,
    resume=resume_flag)
    
    # Print the best trial at the ende
    #best_trial = result.get_best_trial('loss', 'min', 'last')
    #best_trial = result.get_best_trial('loss', 'min', 'last-5-avg')
    #print('Best trial config: {}'.format(best_trial.config))
    #print('Best trial final validation loss: {}'.format(best_trial.last_result['loss'])) 
    
    #best_checkpoint = result.get_best_checkpoint(best_trial, metric='loss', mode='min')
    #print('best_checkpoint:', best_checkpoint)
    
    # Shutdown Ray
    if ray.is_initialized():
        ray.shutdown() 

    print('Total time used: %s seconds' % (time.time() - start_time))
            
    
if __name__ == '__main__':
    main()


