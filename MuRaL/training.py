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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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

from MuRaL.nn_models import *
from MuRaL.nn_utils import *
from MuRaL.preprocessing import *
from MuRaL.evaluation import *

#from torchsampler import ImbalancedDatasetSampler


def train(config, args, checkpoint_dir=None):
    """
    Training funtion.
    
    Args:
        config: configuration of hyperparameters
        args: input args from the command line
        checkpoint_dir: checkpoint dir
    """

    # Get parameters from the command line
    train_file = args.train_data # Ray requires absolute paths
    valid_file = args.validation_data
    ref_genome= args.ref_genome
    n_h5_files = args.n_h5_files

    local_radius = args.local_radius
    local_order = args.local_order
    distal_radius = args.distal_radius  
    distal_order = args.distal_order
    batch_size = args.batch_size
    ####
    sample_weights = args.sample_weights
    #ImbSampler = args.ImbSampler
    local_dropout = args.local_dropout
    CNN_kernel_size = args.CNN_kernel_size   
    CNN_out_channels = args.CNN_out_channels
    distal_fc_dropout = args.distal_fc_dropout
    model_no = args.model_no   
    #pred_file = args.pred_file   
    optim = args.optim
    learning_rate = args.learning_rate   
    weight_decay = args.weight_decay
    weight_decay_auto = args.weight_decay_auto
    LR_gamma = args.LR_gamma 
    restart_lr = args.restart_lr
    min_lr = args.min_lr
    epochs = args.epochs
    n_class = args.n_class  
    cuda_id = args.cuda_id
    valid_ratio = args.valid_ratio
    seq_only = args.seq_only
    cudnn_benchmark_false = args.cudnn_benchmark_false
    without_h5 = args.without_h5
    split_seed = args.split_seed
    gpu_per_trial = args.gpu_per_trial
    cpu_per_trial = args.cpu_per_trial
    save_valid_preds = args.save_valid_preds
    
    bw_paths = args.bw_paths
    without_bw_distal = args.without_bw_distal
    bw_files = []
    bw_names = []
    bw_radii = []
    
    if cudnn_benchmark_false:
        torch.backends.cudnn.benchmark = False
        print('NOTE: setting torch.backends.cudnn.benchmark = False')
    
    if bw_paths:
        try:
            bw_list = pd.read_table(bw_paths, sep='\s+', header=None, comment='#')
            bw_files = list(bw_list[0])
            bw_names = list(bw_list[1])
            if bw_list.shape[1]>2:
                bw_radii = list(bw_list[2].astype(int))
            else:
                bw_radii = [config['local_radius']]*len(bw_files)
            
            print("bw_radii:", bw_radii)
        except pd.errors.EmptyDataError:
            print('Warnings: no bigWig files provided in', bw_paths)
    else:
        print('NOTE: no bigWig files provided.')

    # Read BED files
    train_bed = BedTool(train_file)
    
    if without_h5:
        dataset = prepare_dataset_np(train_bed, ref_genome, bw_files, bw_names, bw_radii, config['local_radius'], config['local_order'], config['distal_radius'], distal_order, seq_only=seq_only, without_bw_distal=without_bw_distal)
        print('using numpy/pandas for distal_seq ...')
    else:
        # Get the H5 file path
        train_h5f_path = get_h5f_path(train_file, bw_names, config['distal_radius'], distal_order, without_bw_distal)

        # Prepare the datasets for trainging
        dataset = prepare_dataset_h5(train_bed, ref_genome, bw_paths, bw_files, bw_names, bw_radii, config['local_radius'], config['local_order'], config['distal_radius'], distal_order, train_h5f_path, chunk_size=5000, seq_only=seq_only, n_h5_files=n_h5_files, without_bw_distal=without_bw_distal)
        
        #prepare_dataset_h5(bed_regions, ref_genome, bw_paths, bw_files, bw_names, local_radius=5, local_order=1, distal_radius=50, distal_order=1, h5f_path='distal_data.h5', chunk_size=5000, seq_only=False, n_h5_files=1)
    
    data_local = dataset.data_local
    categorical_features = dataset.cat_cols
    n_cont = len(dataset.cont_cols)
    
    #config['n_cont'] = n_cont
    config['n_class'] = n_class
    config['model_no'] = model_no
    #config['bw_paths'] = bw_paths
    config['without_bw_distal'] = without_bw_distal
    config['seq_only'] = seq_only
    config['restart_lr'] = restart_lr
    config['min_lr'] = min_lr
    #print('n_cont: ', n_cont)
    
    ################
    if valid_file:
        print('using given validation file:', valid_file)
        valid_bed = BedTool(valid_file)
        valid_h5f_path = get_h5f_path(valid_file, bw_names, config['distal_radius'], distal_order, without_bw_distal)
        if without_h5:
            dataset_valid = prepare_dataset_np(valid_bed, ref_genome, bw_files, bw_names, bw_radii, config['local_radius'], config['local_order'], config['distal_radius'], distal_order, seq_only=seq_only, without_bw_distal=without_bw_distal)
        else:
            dataset_valid = prepare_dataset_h5(valid_bed, ref_genome, bw_paths, bw_files, bw_names, bw_radii, config['local_radius'], config['local_order'], config['distal_radius'], distal_order, valid_h5f_path, chunk_size=5000, seq_only=seq_only, n_h5_files=n_h5_files, without_bw_distal=without_bw_distal)
        
        data_local_valid = dataset_valid.data_local
    ################
    
    device = torch.device('cpu')
    if gpu_per_trial > 0:
        # Set the device
        if not torch.cuda.is_available():
            print('Warning: You requested GPU computing, but CUDA is not available! If you want to run without GPU, please set "--ray_ngpus 0 --gpu_per_trial 0"', file=sys.stderr)
        
        #device = torch.device('cuda:'+cuda_id if torch.cuda.is_available() else 'cpu')    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    #valid_size = int(len(dataset)*valid_ratio)
    #train_size = len(dataset) - valid_size
    #print('train_size, valid_size:', train_size, valid_size)
    
    if not valid_file:
        valid_size = int(len(dataset)*valid_ratio)
        train_size = len(dataset) - valid_size
    
        # Split the data into two parts - training data and validation data
        dataset_train, dataset_valid = random_split(dataset, [train_size, valid_size], torch.Generator().manual_seed(split_seed))
        dataset_valid.indices.sort()
        #data_local_valid = dataset.data_local.iloc[dataset_valid.indices, :]
        data_local_valid = data_local.iloc[dataset_valid.indices, ].reset_index(drop=True)
    else:
        dataset_train = dataset
        train_size = len(dataset_train)
        valid_size = len(dataset_valid)
    
    print('train_size, valid_size:', train_size, valid_size)
    # Dataloader for training
    #if not ImbSampler: 
    if not sample_weights:
        dataloader_train = DataLoader(dataset_train, config['batch_size'], shuffle=True, num_workers=cpu_per_trial-1, pin_memory=True)
    else:
        weights = pd.read_csv(sample_weights, sep='\t', header=None)
        weights = weights[3]
        weighted_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        
        dataloader_train = DataLoader(dataset_train, config['batch_size'], shuffle=False, sampler=weighted_sampler, num_workers=cpu_per_trial-1, pin_memory=True)
        #dataloader_train = DataLoader(dataset_train, config['batch_size'], shuffle=False, sampler=ImbalancedDatasetSampler(dataset_train), num_workers=cpu_per_trial-1, pin_memory=True)
    
    # Dataloader for predicting
    dataloader_valid = DataLoader(dataset_valid, config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    if config['transfer_learning']:
        emb_dims = config['emb_dims']
    else:
        # Number of categorical features
        cat_dims = dataset.cat_dims

        # Set embedding dimensions for categorical features
        # According to https://stackoverflow.com/questions/48479915/what-is-the-preferred-ratio-between-the-vocabulary-size-and-embedding-dimension
        emb_dims = [(x, min(16, int(x**0.25))) for x in cat_dims] 
        config['emb_dims'] = emb_dims    
    #####
    if without_bw_distal:
        in_channels = 4**distal_order
    else:
        in_channels = 4**distal_order+n_cont
    #####
    
    # Choose the network model for training
    if model_no == 0:
        # Local-only model
        model = Network0(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], n_class=n_class, emb_padding_idx=4**config['local_order'])

    elif model_no == 1:
        # ResNet model
        model = Network1(in_channels=in_channels, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'],  distal_radius=config['distal_radius'], distal_order=distal_order, distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class)

    elif model_no == 2:
        # Combined model
        model = Network2(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], in_channels=in_channels, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'], distal_radius=config['distal_radius'], distal_order=distal_order, distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'])
        
    elif model_no == 3:
        # Combined model
        model = Network3(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], in_channels=in_channels, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'], distal_radius=config['distal_radius'], distal_order=distal_order, distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'])
        
    elif model_no == 4:
        # Combined model
        model = Network4(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], in_channels=4**distal_order+n_cont, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'], distal_radius=config['distal_radius'], distal_order=distal_order, distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'])

    elif model_no == 5:
        # Combined model
        model = Network3(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], in_channels=4**distal_order-1, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'], distal_radius=config['distal_radius'], distal_order=distal_order, distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'])
        
    elif model_no == 6:
        # Combined model
        model = Network6(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], in_channels=4**distal_order+n_cont, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'], distal_radius=config['distal_radius'], distal_order=distal_order, distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'])
        
    elif model_no == 7:
        # Combined model
        model = Network7(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], in_channels=4**distal_order, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'], distal_radius=config['distal_radius'], distal_order=distal_order, distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'])

    elif model_no == 8:
        # Combined model
        model = Network8(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], in_channels=4**distal_order, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'], distal_radius=config['distal_radius'], distal_order=distal_order, distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'])

    elif model_no == 9:
        # Combined model
        model = Network9(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], in_channels=4**distal_order, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'], distal_radius=config['distal_radius'], distal_order=distal_order, distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'])
        
    elif model_no == 10:
        # Combined model
        model = Network10(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], in_channels=4**distal_order, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'], distal_radius=config['distal_radius'], distal_order=distal_order, distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'])
        
    elif model_no == 11:
        # Combined model
        model = Network11(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], in_channels=4**distal_order, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'], distal_radius=config['distal_radius'], distal_order=distal_order, distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'])

    elif model_no == 12:
        # Combined model
        model = Network12(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], in_channels=4**distal_order, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'], distal_radius=config['distal_radius'], distal_order=distal_order, distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'])

    elif model_no == 13:
        # Combined model
        model = Network13(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], in_channels=4**distal_order, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'], distal_radius=config['distal_radius'], distal_order=distal_order, distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'])
    elif model_no == 14:
        # Combined model
        model = Network14(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], in_channels=4**distal_order, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'], distal_radius=config['distal_radius'], distal_order=distal_order, distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'])
    elif model_no == 15:
        # Combined model
        model = Network15(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], in_channels=4**distal_order, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'], distal_radius=config['distal_radius'], distal_order=distal_order, distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'])
    elif model_no == 16:
        # Combined model
        model = Network15(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], in_channels=4**distal_order-1, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'], distal_radius=config['distal_radius'], distal_order=distal_order, distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'])
    else:
        print('Error: no model selected!')
        sys.exit() 
    
    model.to(device)
    # Count the parameters in the model
    total_params = count_parameters(model)
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
    
    if config['transfer_learning']:
        #model_state = torch.load(args.model_path, map_location=device)
        model_state = torch.load(args.model_path, map_location='cpu')
        model.to(torch.device('cpu'))# if loaded into GPU, it will double the GPU memory!
        model.load_state_dict(model_state)
        model.to(device)
        
        del model_state
        torch.cuda.empty_cache() 

        criterion = torch.nn.CrossEntropyLoss(reduction='sum')

        if config['train_all']:
            # Train all parameters
            for param in model.parameters():
                param.requires_grad = True
        else:
            # Train only the final fc layers
            for param in model.parameters():
                param.requires_grad = False
            model.local_fc[-1].weight.requires_grad = True
            model.local_fc[-1].bias.requires_grad = True
            model.distal_fc[-1].weight.requires_grad = True
            model.distal_fc[-1].bias.requires_grad = True

        if not config['init_fc_with_pretrained']:
            # Re-initialize fc layers
            model.local_fc[-1].apply(weights_init)
            model.distal_fc[-1].apply(weights_init)    
    else:
        # Initiating weights of the models;
        model.apply(weights_init)
    
    # Set loss function
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    #weights = torch.tensor([0.00515898, 0.44976093, 0.23657462, 0.30850547]).to(device)
    #weights = torch.tensor([0.00378079, 0.42361806, 0.11523816, 0.45736299]).to(device)
    #criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction='sum')
    
    #criterion = PCSoftmaxCrossEntropyV1(lb_proportion=[0.952381, 0.0140095, 0.0198, 0.0138095], reduction='sum')
    #print("using Focal Loss ...")
    #criterion = FocalLoss(gamma=2, size_average=False)
    #criterion = CBLoss(samples_per_cls=[400000, 5884, 8316, 5800], no_of_classes=4, loss_type="sigmoid", beta=0.999999, gamma=1)
    #criterion = CBLoss(samples_per_cls=[400000, 5884, 8316, 5800], no_of_classes=4, loss_type="focal", beta=0.999999, gamma=1)
    
    if weight_decay_auto != None and weight_decay_auto > 0:
        #print("rewriting config['weight_decay'] ...")
        if weight_decay_auto >=1:
            print('Please set a value smaller than 1 for --weight_decay_auto.', file=sys.stderr)
            sys.exit()
        config['weight_decay'] = 1- weight_decay_auto **(config['batch_size']/(epochs*train_size))
        print("NOTE: rewriting config['weight_decay'], new weight_decay: ", config['weight_decay'])
    
    # Set Optimizer
    if config['optim'] == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    elif config['optim'] == 'AdamW':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config['weight_decay'], amsgrad=True)

    elif config['optim'] == 'AdamW2':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config['weight_decay'], amsgrad=True)
        
    elif config['optim'] == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config['weight_decay'], momentum=0.98, nesterov=True)
     
    else:
        print('Error: unsupported optimization method', config['optim'])
        sys.exit()


    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config['LR_gamma'])
    if config['lr_scheduler'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(5000*128)//config['batch_size'], gamma=config['LR_gamma'])
    elif config['lr_scheduler'] == 'StepLR2':
        gamma = (config['min_lr']/config['restart_lr'])**(1/(train_size//config['batch_size']))
        print('learning rate gamma:', gamma)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    elif config['lr_scheduler'] == 'ROP':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.2, patience=1, threshold=0.0001, min_lr=1e-7)
        print("using lr_scheduler.ReduceLROnPlateau ...")

    print('optimizer:', optimizer)
    print('scheduler:', scheduler)

    prob_names = ['prob'+str(i) for i in range(n_class)]
    
    min_loss = 0
    min_loss_epoch = 0
    after_min_loss = 0
    # Training loop
    for epoch in range(epochs):

        model.train()
        total_loss = 0
        
        if epoch > 0 and config['lr_scheduler'] == 'StepLR2':
            for g in optimizer.param_groups:
                g['lr'] = config['restart_lr']            

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
            
            #Clips gradient norm to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, error_if_nonfinite=False)
            
            optimizer.step()
            total_loss += loss.item()
            
            if config['lr_scheduler'] != 'ROP':
                scheduler.step()
                
                # avoid very small learning rates
                if optimizer.param_groups[0]['lr'] < config['min_lr']:
                    print("optimizer.param_groups[0]:", optimizer.param_groups[0]['lr'])
                    for g in optimizer.param_groups:
                        g['lr'] = config['restart_lr']
                        #scheduler.step(1)
        
        # Flush StdOut buffer
        sys.stdout.flush()
        
        print('optimizer learning rate:', optimizer.param_groups[0]['lr'])
        # Update learning rate
        #scheduler.step()
        
        model.eval()
        with torch.no_grad():
            #if model_no != 0:
                #print('F.softmax(model.models_w):', F.softmax(model.models_w, dim=0))
                #print('model.conv1[0].weight.grad:', model.conv1[0].weight.grad)
            #print('model.conv1.0.weight.grad:', model.conv1.0.weight)

            valid_pred_y, valid_total_loss = model_predict_m(model, dataloader_valid, criterion, device, n_class, distal=True)

            valid_y_prob = pd.DataFrame(data=to_np(F.softmax(valid_pred_y, dim=1)), columns=prob_names)
            
            if not valid_file:
                valid_data_and_prob = pd.concat([data_local.iloc[dataset_valid.indices, ].reset_index(drop=True), valid_y_prob], axis=1)
                
            else:
                valid_data_and_prob = pd.concat([data_local_valid, valid_y_prob], axis=1)
            
            valid_y = valid_data_and_prob['mut_type'].to_numpy().squeeze()
            
            # Train the calibrator using the validataion data
            #valid_y_prob = valid_y_prob.reset_index() #### for "ValueError: Input contains NaN"

            fdiri_cal, fdiri_nll = calibrate_prob(valid_y_prob.to_numpy(), valid_y, device, calibr_name='FullDiri')
            #fdirio_cal, _ = calibrate_prob(valid_y_prob.to_numpy(), valid_y, device, calibr_name='FullDiriODIR')
            #vec_cal, _ = calibrate_prob(valid_y_prob.to_numpy(), valid_y, device, calibr_name='VectS')
            #tmp_cal, _ = calibrate_prob(valid_y_prob.to_numpy(), valid_y, device, calibr_name='TempS')
            
            print("valid_data_and_prob.iloc[0:10]", valid_data_and_prob.iloc[0:10])
            
            # Compare observed/predicted 3/5/7mer mutation frequencies
            print('3mer correlation - all: ', freq_kmer_comp_multi(valid_data_and_prob, 3, n_class))
            print('5mer correlation - all: ', freq_kmer_comp_multi(valid_data_and_prob, 5, n_class))
            print('7mer correlation - all: ', freq_kmer_comp_multi(valid_data_and_prob, 7, n_class))
            
            print ('Training Loss: ', total_loss/train_size)
            
            print ('Validation Loss: ', valid_total_loss/valid_size)
            print ('Validation Loss (after fdiri_cal): ', fdiri_nll)
            
            ###########
            prob_cal = fdiri_cal.predict_proba(valid_y_prob.to_numpy())  
            y_prob = pd.DataFrame(data=np.copy(prob_cal), columns=prob_names)
    
            # Combine data and do k-mer evaluation
            valid_cal_data_and_prob = pd.concat([data_local_valid, y_prob], axis=1)         
            print('3mer correlation(after fdiri_cal): ', freq_kmer_comp_multi(valid_cal_data_and_prob, 3, n_class))
            print('5mer correlation(after fdiri_cal): ', freq_kmer_comp_multi(valid_cal_data_and_prob, 5, n_class))
            print('7mer correlation(after fdiri_cal): ', freq_kmer_comp_multi(valid_cal_data_and_prob, 7, n_class))
            ############      
            
            # Calculate a custom score by looking obs/pred 3/5-mer correlations in binned windows
            if valid_size > 10000 *10:
                region_size = 10000
            else:
                region_size = valid_size // 10
            
            n_regions = valid_size//region_size
            print('n_regions:', n_regions)
            
            score = 0
            corr_3mer = []
            corr_5mer = []
            
            region_avg = []
            for i in range(n_regions):
                corr_3mer = freq_kmer_comp_multi(valid_data_and_prob.iloc[region_size*i:region_size*(i+1), ], 3, n_class)    
                corr_5mer = freq_kmer_comp_multi(valid_data_and_prob.iloc[region_size*i:region_size*(i+1), ], 5, n_class)
                
                score += np.sum([(1-corr)**2 for corr in corr_3mer]) + np.sum([(1-corr)**2 for corr in corr_5mer])
                
                avg_prob = calc_avg_prob(valid_data_and_prob.iloc[region_size*i:region_size*(i+1)], n_class)
                region_avg.append(avg_prob)
                #print("avg_prob:", avg_prob, i)
            
            region_avg = pd.DataFrame(region_avg)
            #print('region_avg.head():', region_avg.head())
            corr_list = []
            for i in range(n_class):
                corr_list.append(region_avg[i].corr(region_avg[i + n_class]))
            
            print('corr_list:', corr_list)
            #print('corr_3mer:', corr_3mer)
            #print('corr_5mer:', corr_5mer)
            print('regional score:', score, n_regions)
            
            # Output genomic positions and predicted probabilities
            if not valid_file:
                chr_pos = train_bed.to_dataframe().loc[dataset_valid.indices,['chrom', 'start', 'end', 'strand']].reset_index(drop=True)
            else:
                chr_pos = valid_bed.to_dataframe()[['chrom', 'start', 'end', 'strand']]
                
            valid_pred_df = pd.concat((chr_pos, valid_data_and_prob[['mut_type'] + prob_names]), axis=1)
            valid_pred_df.columns = ['chrom', 'start', 'end', 'strand', 'mut_type'] + prob_names
            ####
            valid_cal_pred_df = pd.concat((chr_pos, valid_cal_data_and_prob[['mut_type'] + prob_names]), axis=1)
            valid_cal_pred_df.columns = ['chrom', 'start', 'end', 'strand', 'mut_type'] + prob_names
            
            ####
            
            print('valid_pred_df: ', valid_pred_df.head())
            
            # Save model data for each checkpoint
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, 'model')
                torch.save(model.state_dict(), path)
            
                with open(path + '.fdiri_cal.pkl', 'wb') as pkl_file:
                    pickle.dump(fdiri_cal, pkl_file)
                
                with open(path + '.config.pkl', 'wb') as fp:
                    pickle.dump(config, fp)
                if save_valid_preds:
                    valid_pred_df.to_csv(path + '.valid_preds.tsv.gz', sep='\t', float_format='%.4g', index=False)
                    
            # Print regional correlations
            valid_pred_df.sort_values(['chrom', 'start'], inplace=True)
            valid_cal_pred_df.sort_values(['chrom', 'start'], inplace=True)
            #for win_size in [20000, 100000, 500000]:
            for win_size in [100000, 500000]:
                
                corr_win = corr_calc_sub(valid_pred_df, win_size, prob_names)
                print('regional corr (validation):', str(win_size)+'bp', corr_win)
                
                corr_win_cal = corr_calc_sub(valid_cal_pred_df, win_size, prob_names)
                print('regional corr (validation, after calibration):', str(win_size)+'bp', corr_win_cal)
                    
            current_loss = valid_total_loss/valid_size
            if epoch == 0 or current_loss < min_loss:
                min_loss = current_loss
                min_loss_epoch = epoch
                after_min_loss = 0
            else:
                after_min_loss = epoch - min_loss_epoch
                
            tune.report(loss=current_loss, fdiri_loss=fdiri_nll, after_min_loss=after_min_loss, score=score, total_params=total_params)
            
            #####
            if config['lr_scheduler'] == 'ROP':
                scheduler.step(current_loss)
            
            torch.cuda.empty_cache() 
            sys.stdout.flush()
