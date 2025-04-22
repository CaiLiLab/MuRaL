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

#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.allow_tf32 = True

from functools import partial
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import os
import time
import datetime
import random

from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from utils.printer_utils import get_printer
from model.nn_models import *
from model.nn_utils import *
from data.preprocessing import *
from evaluation.evaluation import *
from utils.train_utils import run_standalong_training
from utils.gpu_utils import get_available_gpu, check_cuda_id 
from _version import __version__

import textwrap
#from torch.utils.tensorboard import SummaryWriter


def run_train_pipline(args, model_type):
    """
    according args.modle = SNV or Indel choice train func
    """

    from training import train
    
    #args = parse_arguments(parser)
    train = partial(train, model_type=model_type)

    start_time = time.time()
    current_time = datetime.datetime.now()
    
    # Creat tmp log file
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    if not args.use_ray:
        experiment_dir = f'results/{args.experiment_name}'
        os.makedirs(experiment_dir, exist_ok=True)
        tmp_log_file = f"{experiment_dir}/{args.experiment_name}_{timestamp}.log"
    else:
        tmp_log_file = None
    args.tmp_log_file = tmp_log_file
    # Ensure output can be viewed in real-time in a distributed environment
    print = get_printer(args.use_ray, tmp_log_file)
    
    print('Start time:', current_time)
    sys.stdout.flush()

    print(' '.join(sys.argv)) # print the command line
    for k,v in vars(args).items():
        print("{0}: {1}".format(k,v))
    
    # Ray requires absolute paths
    train_file  = args.train_data = os.path.abspath(args.train_data) 
    valid_file = args.validation_data
    if valid_file: 
        args.validation_data = os.path.abspath(args.validation_data) 

    ref_genome = args.ref_genome =  os.path.abspath(args.ref_genome)
    n_h5_files = args.n_h5_files
    
    sampled_segments = args.sampled_segments
    segment_center = args.segment_center
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
    sample_weights = args.sample_weights
    if sample_weights:
        args.sample_weights = os.path.abspath(args.sample_weights)
    #ImbSampler = args.ImbSampler
    optim = args.optim
    lr_scheduler = args.lr_scheduler
    learning_rate = args.learning_rate   
    weight_decay = args.weight_decay
    weight_decay_auto = args.weight_decay_auto
        
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
    use_ray = args.use_ray
    
    if args.split_seed < 0:
        args.split_seed = random.randint(0, 1000000)
    print('args.split_seed:', args.split_seed)
    
    
    # Read bigWig file names
    bw_paths = args.bw_paths
    without_bw_distal = args.without_bw_distal
    if bw_paths:
       args.bw_paths =  os.path.abspath(args.bw_paths)
    
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
    
    # Prepare min/max for the loguniform samplers if one value is provided
    if len(learning_rate) == 1:
        learning_rate = learning_rate*2
    if len(weight_decay) == 1:
        weight_decay = weight_decay*2
    
    
    # Use GPU
    if ray_ngpus > 0 or gpu_per_trial > 0:
        if not torch.cuda.is_available():
            print('Error: You requested GPU computing, but CUDA is not available! If you want to run without GPU, please set "--ray_ngpus 0 --gpu_per_trial 0"', file=sys.stderr)
            sys.exit()

        from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
        nvmlInit()
        gpu_device_number = nvmlDeviceGetCount()
        # Find a GPU with enough memory
        if cuda_id == None: 
            cuda_id = get_available_gpu(ray_ncpus/cpu_per_trial)
            args.cuda_id = cuda_id
        # Check cuda_id exists
        else:
            check_cuda_id(cuda_id)
        print('CUDA: ', torch.cuda.is_available())
        print('using'  , 'cuda:'+cuda_id)
        print('Train using GPU device', 'cuda:'+cuda_id)
    # Use CPU
    else:
        print('Train using only CPUs ...')
    
    if not use_ray:
        print("Ray not used in model training !")
        config = {
        'local_radius': local_radius[0],
        'segment_center': segment_center,
        'local_order': local_order[0],
        'local_hidden1_size': local_hidden1_size[0],
        #'local_hidden2_size': tune.choice(local_hidden2_size),
        'local_hidden2_size': local_hidden2_size[0] if local_hidden2_size[0]>0 else local_hidden1_size[0]//2, # default local_hidden2_size = local_hidden1_size//2
        'distal_radius': distal_radius[0],
        'emb_dropout': emb_dropout[0],
        'local_dropout': local_dropout[0],
        'CNN_kernel_size': CNN_kernel_size[0],
        'CNN_out_channels': CNN_out_channels[0],
        'distal_fc_dropout': distal_fc_dropout[0],
        'batch_size': batch_size[0],
        'sampled_segments': sampled_segments[0],
        'learning_rate': learning_rate[0],
        #'learning_rate': tune.choice(learning_rate),
        'optim': optim[0],
        'lr_scheduler':lr_scheduler[0],
        'LR_gamma': LR_gamma[0],
        'weight_decay': weight_decay[0],
        #'weight_decay': tune.choice(weight_decay),
        'transfer_learning': False,
        'use_ray' : False,
        'custom_dataloader' : args.custom_dataloader 
    }
        para=False
        run_standalong_training(train, n_trials, config, args,para)
        #train(config, args)
        print('Total time used: %s seconds' % (time.time() - start_time))
        return 0

    # Set visible GPU(s)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    
    if rerun_failed:
        resume_flag = 'ERRORED_ONLY'
    else:
        resume_flag = False
    
    # Allocate CPU/GPU resources for this Ray job
    ray.init(num_cpus=ray_ncpus, num_gpus=ray_ngpus, dashboard_host="0.0.0.0")
    print(ray.cluster_resources())
    #ray.init(num_cpus=ray_ncpus, num_gpus=ray_ngpus)
    
    sys.stdout.flush()

    # Configure the search space for relavant hyperparameters
    config = {
        'local_radius': tune.choice(local_radius),
        'segment_center': segment_center,
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
        'sampled_segments': tune.choice(sampled_segments),
        'learning_rate': tune.loguniform(learning_rate[0], learning_rate[1]),
        #'learning_rate': tune.choice(learning_rate),
        'optim': tune.choice(optim),
        'lr_scheduler':tune.choice(lr_scheduler),
        'LR_gamma': tune.choice(LR_gamma),
        'weight_decay': tune.loguniform(weight_decay[0], weight_decay[1]),

        #'weight_decay': tune.choice(weight_decay),
        'transfer_learning': False,
        'use_ray' : True,
        'custom_dataloader' : args.custom_dataloader
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
    reporter = CLIReporter(parameter_columns=['local_radius', 'local_order', 'local_hidden1_size', 'local_hidden2_size', 'distal_radius', 'emb_dropout', 'local_dropout', 'CNN_kernel_size', 'CNN_out_channels', 'distal_fc_dropout', 'optim', 'learning_rate', 'weight_decay', 'LR_gamma', 'batch_size'], metric_columns=['loss', 'fdiri_loss', 'after_min_loss',  'score', 'total_params', 'training_iteration'])
    
    trainable_id = 'Train'
    tune.register_trainable(trainable_id, partial(train, args=args))
    
    def trial_dirname_string(trial):
        return "{}_{}".format(trial.trainable_name, trial.trial_id)

    # Execute the training
    result = tune.run(
    trainable_id,
    name=experiment_name,
    resources_per_trial={'cpu': 1, 'gpu': gpu_per_trial, 'extra_cpu':cpu_per_trial-1},
    config=config,
    num_samples=n_trials,
    local_dir='./ray_results',
    trial_dirname_creator=trial_dirname_string,
    scheduler=scheduler,
    stop={'after_min_loss':3},
    progress_reporter=reporter,
    resume=resume_flag,
    log_to_file=True)

    # Shutdown Ray
    if ray.is_initialized():
        ray.shutdown() 

    print('Total time used: %s seconds' % (time.time() - start_time))
if __name__ == '__main__':
    main()


