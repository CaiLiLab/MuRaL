import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from pybedtools import BedTool

import sys
import argparse
import textwrap

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

from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from utils.printer_utils import get_printer
from model.nn_models import *
from model.nn_utils import *
from data.preprocessing import *
from utils.train_utils import run_standalong_training
from utils.gpu_utils import get_available_gpu, check_cuda_id 
from evaluation import *
from training import *
from _version import __version__


from pynvml import *

from functools import partial
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True

def para_read_from_config(para, config):
    try:
        value = config[para]
    except:
        print(f"Error: {para} not in spcify model, please set non-zero value for this para")
        sys.exit()
    return value

def run_transfer_pipline(args, model_type):

    from training import train

    start_time = time.time()
    current_time = datetime.datetime.now()
    # Creat tmp log file
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")

    train = partial(train, model_type=model_type)
    if not args.use_ray:
        experiment_dir = f'results/{args.experiment_name}'
        os.makedirs(experiment_dir, exist_ok=True)
        tmp_log_file = f"{experiment_dir}/{args.experiment_name}_{timestamp}.log"
        args.tmp_log_file = tmp_log_file
    else:
        tmp_log_file = args.tmp_log_file = None
    
    # Ensure output can be viewed in real-time in a distributed environment
    print = get_printer(args.use_ray, tmp_log_file)
    
    print('Start time:', current_time)
    print(' '.join(sys.argv))
    
    for k,v in vars(args).items():
        print("{0}: {1}".format(k,v))
    sys.stdout.flush()

    train_file  = args.train_data = os.path.abspath(args.train_data) 
    valid_file = args.validation_data
    if valid_file: 
        args.validation_data = os.path.abspath(args.validation_data)     
    ref_genome = args.ref_genome =  os.path.abspath(args.ref_genome)
    n_h5_files = args.n_h5_files
    
    sample_weights = args.sample_weights
    if sample_weights:
        args.sample_weights = os.path.abspath(args.sample_weights)
    #ImbSampler = args.ImbSampler
    segment_center = args.segment_center
    sampled_segments = args.sampled_segments
    batch_size = args.batch_size
    custom_dataloader = args.custom_dataloader
    #n_class = args.n_class
    #model_no = args.model_no
    #seq_only = args.seq_only
    #pred_file = args.pred_file
    valid_ratio = args.valid_ratio
    save_valid_preds = args.save_valid_preds
    rerun_failed = args.rerun_failed

    
    ray_ncpus = args.ray_ncpus
    ray_ngpus = args.ray_ngpus
    cpu_per_trial = args.cpu_per_trial
    gpu_per_trial = args.gpu_per_trial
    cuda_id = args.cuda_id
    
    optim = args.optim
    learning_rate = args.learning_rate
    lr_scheduler = args.lr_scheduler
    weight_decay = args.weight_decay  
    LR_gamma = args.LR_gamma  
    epochs = args.epochs
    
    grace_period = args.grace_period
    n_trials = args.n_trials
    experiment_name = args.experiment_name
    ASHA_metric = args.ASHA_metric
    
    train_all = args.train_all
    init_fc_with_pretrained = args.init_fc_with_pretrained
    
    #model_path = args.model_path
    if args.model_path:
        args.model_path =  os.path.abspath(args.model_path)
    
    #calibrator_path = args.calibrator_path
    model_config_path = args.model_config_path

    if args.split_seed < 0:
        args.split_seed = random.randint(0, 10000)
    print('args.split_seed:', args.split_seed)

    # Load model config (hyperparameters)

    with open(model_config_path, 'rb') as fconfig:
        config = pickle.load(fconfig)

        local_radius = args.local_radius = config['local_radius']
        local_order = args.local_order = config['local_order']
        local_hidden1_size = args.local_hidden1_size = config['local_hidden1_size']
        local_hidden2_size = args.local_hidden2_size = config['local_hidden2_size']
        distal_radius = args.distal_radius = config['distal_radius']
        distal_order = args.distal_order = 1 # reserved for future improvement
        CNN_kernel_size = args.CNN_kernel_size = config['CNN_kernel_size']  
        CNN_out_channels = args.CNN_out_channels = config['CNN_out_channels']
        emb_dropout = args.emb_dropout = config['emb_dropout']
        local_dropout = args.local_dropout = config['local_dropout']
        distal_fc_dropout = args.distal_fc_dropout = config['distal_fc_dropout']
        emb_dims = config['emb_dims']

        args.n_class = config['n_class']
        args.model_no = config['model_no']
        args.down_list = config.get('down_list', None)
        
        if 'without_bw_distal' in config: 
            args.without_bw_distal = without_bw_distal = config['without_bw_distal']
        else:
            args.without_bw_distal = without_bw_distal = False
        
        if not segment_center:
            segment_center = args.segment_center = para_read_from_config('segment_center',config)

        if not sampled_segments:
            sampled_segments = args.sampled_segments = para_read_from_config('sampled_segments', config)
        else:
            sampled_segments = args.sampled_segments[0]

        args.seq_only = config['seq_only']
    
    
    start_time = time.time()
    print('Start time:', datetime.datetime.now())
    
    # Prepare min/max for the loguniform samplers if one value is provided
    if len(learning_rate) == 1:
        learning_rate = learning_rate*2
    if len(weight_decay) == 1:
        weight_decay = weight_decay*2
    
    # Read bigWig file names
    bw_paths = args.bw_paths
    if bw_paths:
       args.bw_paths =  os.path.abspath(args.bw_paths)
    
    bw_files = []
    bw_names = []
    n_cont = 0
    
    if bw_paths:
        try:
            bw_list = pd.read_table(bw_paths, sep='\s+', header=None, comment='#')
            bw_files = list(bw_list[0])
            bw_names = list(bw_list[1])
            n_cont = len(bw_names)
        except pd.errors.EmptyDataError:
            print('Warnings: no bigWig files provided in', bw_paths)
    else:
        print('NOTE: no bigWig files provided.')

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
    else:
        print('Train is using only CPUs ...')
    
    if not args.use_ray:
        print("Ray not used in model training !")
        config = {
        'local_radius': local_radius,
        'segment_center': segment_center,
        'local_order': local_order,
        'local_hidden1_size': local_hidden1_size,
        #'local_hidden2_size': tune.choice(local_hidden2_size),
        'local_hidden2_size': local_hidden2_size,  
        'distal_radius': distal_radius,
        'emb_dropout': emb_dropout,
        'local_dropout': local_dropout,
        'CNN_kernel_size': CNN_kernel_size,
        'CNN_out_channels': CNN_out_channels,
        'distal_fc_dropout': distal_fc_dropout,
        'batch_size': batch_size[0],
        'sampled_segments': sampled_segments,
        'learning_rate': learning_rate[0],
        #'learning_rate': tune.choice(learning_rate),
        'optim': optim[0],
        'lr_scheduler':lr_scheduler[0],
        'LR_gamma': LR_gamma[0],
        'weight_decay': weight_decay[0],
        #'weight_decay': tune.choice(weight_decay),
        'transfer_learning': False,
        'use_ray' : False,
        'custom_dataloader' : custom_dataloader
        }

        para = False
        run_standalong_training(train, n_trials, config, args, para)
        #train(config, args)

        return 0

    # Set visible GPU(s)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    print('Ray is using GPU device', 'cuda:'+cuda_id)
    if rerun_failed:
        resume_flag = 'ERRORED_ONLY'
    else:
        resume_flag = False
    
    # Allocate CPU/GPU resources for this Ray job
    ray.init(num_cpus=ray_ncpus, num_gpus=ray_ngpus, dashboard_host="0.0.0.0")
    
    sys.stdout.flush()
    
    # Configure the search space for relavant hyperparameters
    config_ray = {
        'local_radius': local_radius,
        'local_order': local_order,
        'segment_center': segment_center,
        'local_hidden1_size': local_hidden1_size,
        'local_hidden2_size': local_hidden2_size,
        'distal_radius': distal_radius,
        'emb_dropout': emb_dropout,
        'local_dropout': local_dropout,
        'CNN_kernel_size': CNN_kernel_size,
        'CNN_out_channels': CNN_out_channels,
        'distal_fc_dropout': distal_fc_dropout,
        'batch_size': tune.choice(batch_size),
        'sampled_segments': sampled_segments,
        'learning_rate': tune.loguniform(learning_rate[0], learning_rate[1]),
        'lr_scheduler':tune.choice(lr_scheduler),
        #'learning_rate': tune.choice(learning_rate),
        'optim': tune.choice(optim),
        'LR_gamma': tune.choice(LR_gamma),
        'weight_decay': tune.loguniform(weight_decay[0], weight_decay[1]),
        'transfer_learning': True,
        'train_all': train_all,
        'init_fc_with_pretrained': init_fc_with_pretrained,
        'emb_dims':emb_dims,
        'use_ray' : True,
        'custom_dataloader' : custom_dataloader
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
    reporter = CLIReporter(parameter_columns=['local_radius', 'local_order', 'local_hidden1_size', 'local_hidden2_size', 'distal_radius', 'emb_dropout', 'local_dropout', 'CNN_kernel_size', 'CNN_out_channels', 'distal_fc_dropout', 'transfer_learning', 'train_all', 'init_fc_with_pretrained', 'optim', 'learning_rate', 'weight_decay', 'LR_gamma'], metric_columns=['loss', 'fdiri_loss', 'after_min_loss','score', 'total_params', 'training_iteration'])
    
    trainable_id = 'Train'
    tune.register_trainable(trainable_id, partial(train, args=args))
    
    def trial_dirname_string(trial):
        return "{}_{}".format(trial.trainable_name, trial.trial_id)
    
    # Execute the training
    result = tune.run(
    trainable_id,
    name=experiment_name,
    resources_per_trial={'cpu': cpu_per_trial, 'gpu': gpu_per_trial},
    config=config_ray,
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
    
    
if __name__ == "__main__":
    main()
