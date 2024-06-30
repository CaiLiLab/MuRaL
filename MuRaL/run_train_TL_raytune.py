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

from MuRaL.printer_utils import get_printer
from MuRaL.nn_models import *
from MuRaL.nn_utils import *
from MuRaL.preprocessing import *
from MuRaL.train_utils import run_train
from MuRaL.evaluation import *
from MuRaL.training import *
from MuRaL._version import __version__


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

def parse_arguments(parser):
    """
    Parse parameters from the command line
    """
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('Required arguments')
    data_args = parser.add_argument_group('Data-related arguments')
    model_args = parser.add_argument_group('Transfer learning arguments')
    learn_args = parser.add_argument_group('Learning-related arguments')
    raytune_args = parser.add_argument_group('RayTune-related arguments')
    optional.title = 'Other arguments' 
    
    required.add_argument('--ref_genome', type=str, metavar='FILE', default='',  
                          required=True, help=textwrap.dedent("""
                          File path of the reference genome in FASTA format.""").strip())
    
    required.add_argument('--train_data', type=str, metavar='FILE', default='',  
                          required=True, help= textwrap.dedent("""
                          File path of training data in a sorted BED format. If the options
                          --validation_data and --valid_ratio not specified, 10%% of the
                          sites sampled from the training BED will be used as the
                          validation data.""").strip())
    
    required.add_argument('--model_path', type=str, metavar='FILE', required=True,
                          help=textwrap.dedent("""
                          File path of the trained model.
                          """ ).strip())  
    
    required.add_argument('--model_config_path', type=str, metavar='FILE', required=True,
                          help=textwrap.dedent("""
                          File path for the configurations of the trained model.
                          """ ).strip())    
    
    model_args.add_argument('--train_all', default=False, action='store_true', 
                          help= textwrap.dedent("""
                          Train all parameters of the model. If False, only the parameters
                          in the last FC layers will be trained. Default: False.""").strip())
    
    
    model_args.add_argument('--init_fc_with_pretrained', default=False, action='store_true', 
                          help= textwrap.dedent("""
                          Use the weights of the pre-trained model to initialize the last 
                          FC layers. If False, parameters of last FC layers are randomly 
                          initialized. Default: False.""").strip())  
   
    data_args.add_argument('--validation_data', type=str, metavar='FILE', default=None,
                          help=textwrap.dedent("""
                          File path for validation data. If this option is set,
                          the value of --valid_ratio will be ignored. Default: None.
                          """).strip()) 
    
    data_args.add_argument('--sample_weights', type=str, metavar='FILE', default=None,
                          help=textwrap.dedent("""
                          File path for sample weights. Default: None.
                          """).strip())
    
    data_args.add_argument('--valid_ratio', type=float, metavar='FLOAT', default=0.1, 
                          help=textwrap.dedent("""
                          Ratio of validation data relative to the whole training data.
                          Default: 0.1. 
                          """ ).strip())
    
    data_args.add_argument('--split_seed', type=int, metavar='INT', default=-1, 
                          help=textwrap.dedent("""
                          Seed for randomly splitting data into training and validation
                          sets. Default: a random number generated by the job.
                          """ ).strip())
    
    data_args.add_argument('--save_valid_preds', default=False, action='store_true', 
                          help=textwrap.dedent("""
                          Save prediction results for validation data in the checkpoint
                          folders. Default: False.
                          """ ).strip())
    
    data_args.add_argument('--bw_paths', type=str, metavar='FILE', default=None,
                          help=textwrap.dedent("""
                          File path for a list of BigWig files for non-sequence 
                          features such as the coverage track. If the pre-trained model
                          used some bigWig tracks, tracks with same names are needed 
                          to be provided with this option. Default: None.""").strip())
    
    data_args.add_argument('--with_h5', default=False, action='store_true', 
                          help=textwrap.dedent("""
                          Generate HDF5 files for input BED files. Default: False.""").strip())
    
    data_args.add_argument('--h5f_path', type=str, default=None,
                         help=textwrap.dedent("""
                         Specify the folder to generate HDF5. Default: Folder containing the BED file.""").strip())
    
    data_args.add_argument('--n_h5_files', type=int, metavar='INT', default=1, 
                          help=textwrap.dedent("""
                          Number of HDF5 files for each BED file. When the BED file has many
                          positions and the distal radius is large, increasing the value for 
                          --n_h5_files files can reduce the time for generating HDF5 files.
                          Default: 1.
                          """ ).strip())            

    learn_args.add_argument('--segment_center', type=int, metavar='INT', default=300000, 
                          help=textwrap.dedent("""
                          The maximum encoding unit of the sequence. It affects trade-off 
                          between RAM memory and preprocessing speed. It is recommended to use 300k.
                          Default: 300000.""" ).strip())

    learn_args.add_argument('--sampled_segments', type=int, metavar='INT', default=[10], nargs='+',
                          help=textwrap.dedent("""
                          Number of segments chosen for generating samples for batches in DataLoader.
                          Default: 10.
                          """ ).strip())
    
    learn_args.add_argument('--batch_size', type=int, metavar='INT', default=[128], nargs='+', 
                          help=textwrap.dedent("""
                          Size of mini batches for model training. Default: 128.
                          """ ).strip())    
                          
    learn_args.add_argument('--custom_dataloader', default=False, action='store_true',  
                          help=textwrap.dedent("""
                          Specify the way to load data. Default: False.
                          """ ).strip())

    learn_args.add_argument('--optim', type=str, metavar='STR', default=['Adam'], nargs='+', 
                          help=textwrap.dedent("""
                          Optimization method for parameter learning: 'Adam' or 'AdamW'.
                          Default: 'Adam'.
                          """ ).strip())
 
    learn_args.add_argument('--learning_rate', type=float, metavar='FLOAT', default=[0.0001], nargs='+', 
                          help=textwrap.dedent("""
                          Learning rate for parameter learning, an argument for the 
                          optimization method.  Default: 0.0001.
                          """ ).strip())
    
    learn_args.add_argument('--lr_scheduler', type=str, metavar='STR', default=['StepLR'], nargs='+', 
                          help=textwrap.dedent("""
                          Learning rate scheduler.
                          Default: 'StepLR'.
                          """ ).strip())
    
    learn_args.add_argument('--weight_decay_auto', type=float, metavar='FLOAT', default=0.1, 
                          help=textwrap.dedent("""
                          Calcaute 'weight_decay' (regularization parameter) based on total 
                          training steps. It automatically adjusts 'weight_decay' for different 
                          batch sizes, training sizes and epochs. Its value MUST be smaller than 1.
                          For values in the range 0~1, smaller values mean stronger regularization.
                          Set a value of <=0 to turn this off.
                          Default: 0.1.
                          """ ).strip())
    
    learn_args.add_argument('--weight_decay', type=float, metavar='FLOAT', default=[1e-5], nargs='+', 
                          help=textwrap.dedent("""
                          'weight_decay' argument (regularization) for the optimization method. 
                          If you want to use this option, please also set '--weight_decay_auto' to 0.
                          Default: 1e-5.
                          """ ).strip())
    
    learn_args.add_argument('--restart_lr', type=float, metavar='FLOAT', default=1e-4, 
                          help=textwrap.dedent("""
                          When the learning rate reaches the mininum rate, reset it to 
                          a larger one. Default: 1e-4.
                          """ ).strip())
    
    learn_args.add_argument('--min_lr', type=float, metavar='FLOAT', default=1e-6, 
                          help=textwrap.dedent("""
                          The minimum learning rate. Default: 1e-6.
                          """ ).strip())
    
    learn_args.add_argument('--LR_gamma', type=float, metavar='FLOAT', default=[0.9], nargs='+', 
                          help=textwrap.dedent("""
                          'gamma' argument for the learning rate scheduler.
                           Default: 0.9.
                           """ ).strip())
    
    learn_args.add_argument('--cudnn_benchmark_false', default=False, action='store_true', 
                          help=textwrap.dedent("""
                          If set, use only genomic sequences for the model and ignore
                          bigWig tracks. Default: False.""").strip())

    raytune_args.add_argument('--use_ray', default=False, action='store_true',
                          help=textwrap.dedent("""
                          Use ray to run multiple trials in parallel.  Default: False.
                          """ ).strip())

    raytune_args.add_argument('--experiment_name', type=str, metavar='STR', default='my_experiment',
                          help=textwrap.dedent("""
                          Ray-Tune experiment name.  Default: 'my_experiment'.
                          """ ).strip()) 
    
    raytune_args.add_argument('--n_trials', type=int, metavar='INT', default=2, 
                          help=textwrap.dedent("""
                          Number of trials for this training job.  Default: 2.
                          """ ).strip())
    
    raytune_args.add_argument('--epochs', type=int, metavar='INT', default=10, 
                          help=textwrap.dedent("""
                          Maximum number of epochs for each trial.  Default: 10.
                          """ ).strip())
    
    raytune_args.add_argument('--grace_period', type=int, metavar='INT', default=5, 
                          help=textwrap.dedent("""
                          'grace_period' parameter for early stopping. 
                           Default: 5.
                           """ ).strip())
    
    raytune_args.add_argument('--ASHA_metric', type=str, metavar='STR', default='loss', 
                          help=textwrap.dedent("""
                          Metric for ASHA schedualing; the value can be 'loss' or 'fdiri_loss'.
                          Default: 'loss'.
                          """ ).strip())
    
    raytune_args.add_argument('--ray_ncpus', type=int, metavar='INT', default=2, 
                          help=textwrap.dedent("""
                          Number of CPUs requested by Ray-Tune. Default: 2.
                          """ ).strip())
    
    raytune_args.add_argument('--ray_ngpus', type=int, metavar='INT', default=1, 
                          help=textwrap.dedent("""
                          Number of GPUs requested by Ray-Tune. Default: 1.
                          """ ).strip())
    
    raytune_args.add_argument('--cpu_per_trial', type=int, metavar='INT', default=2, 
                          help=textwrap.dedent("""
                          Number of CPUs used per trial. Default: 2.
                          """ ).strip())
    
    raytune_args.add_argument('--gpu_per_trial', type=float, metavar='FLOAT', default=0.15, 
                          help=textwrap.dedent("""
                          Number of GPUs used per trial. Default: 0.15.
                          """ ).strip())
    
    raytune_args.add_argument('--cuda_id', type=str, metavar='STR', default='0', 
                          help=textwrap.dedent("""
                          Which GPU device to be used. Default: '0'. 
                          """ ).strip())
    
    raytune_args.add_argument('--rerun_failed', default=False, action='store_true', 
                          help=textwrap.dedent("""
                          Rerun errored or incomplete trials. Default: False.
                          """ ).strip())
    
    optional.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    parser._action_groups.append(optional)

    if len(sys.argv) == 1:
        parser.parse_args(['--help'])
    else:
        args = parser.parse_args()

    return args

def para_read_from_config(para, config):
    try:
        value = config[para]
    except:
        print(f"Error: {para} not in spcify model, please set non-zero value for this para")
        sys.exit()
    return value

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="""
    Overview
    --------    
    This tool uses learned weights from a pre-trained MuRaL model to train new models. 
    The inputs include training and validation mutation data and training results will
    be saved under the "./ray_results/" folder.
    
    * Input data 
    The input files include the reference FASTA file (required), a training data file 
    (required), a validation data file (optional), and model-related files of a trained 
    model (required). The required model-related files are 'model' and 'model.config.pkl'
    under a specific checkpoint folder, which are normally produced by `mural_train` 
    or `mural_train_TL`. 
   
    * Output data 
    Output data has the same structure as that of `mural_train`.

    Command line examples
    ---------------------
    1. The following command will train a transfer learning model using training data 
    in 'training.sorted.bed', the validation data in 'validation.sorted.bed', and the model
    files under 'checkpoint_6/'.
   
        mural_train_TL --ref_genome seq.fa --train_data training.sorted.bed \\
        --validation_data validation.sorted.bed --model_path checkpoint_6/model \\
        --model_config_path checkpoint_6/model.config.pkl --train_all \\
        --init_fc_with_pretrained --experiment_name example4 > test4.out 2> test4.err
    """)
    args = parse_arguments(parser)
    
    start_time = time.time()
    current_time = datetime.datetime.now()
    # Creat tmp log file
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
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
        print('Train is using GPU device', 'cuda:'+cuda_id)
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
        'use_ray' : True,
        'custom_dataloader' : custom_dataloader
        }

        para = False
        run_train(train, n_trials, config, args, para)
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
        'use_ray' : False,
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
