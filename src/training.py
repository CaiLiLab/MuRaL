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
    split_seed = args.split_seed
    gpu_per_trial = args.gpu_per_trial
    save_valid_preds = args.save_valid_preds
    
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
    dataset = prepare_dataset_h5(train_bed, ref_genome, bw_files, bw_names, config['local_radius'], config['local_order'], config['distal_radius'], distal_order, train_h5f_path, h5_chunk_size=1, seq_only=seq_only)
    
    data_local = dataset.data_local
    categorical_features = dataset.cat_cols
    n_cont = len(dataset.cont_cols)
    
    #config['n_cont'] = n_cont
    config['n_class'] = n_class
    config['model_no'] = model_no
    #config['bw_paths'] = bw_paths
    config['seq_only'] = seq_only
    #print('n_cont: ', n_cont)
    
    device = torch.device('cpu')
    if gpu_per_trial > 0:
        # Set the device
        if not torch.cuda.is_available():
            print('Warning: You requested GPU computing, but CUDA is not available! If you want to run without GPU, please set "--ray_ngpus 0 --gpu_per_trial 0"', file=sys.stderr)
        
        #device = torch.device('cuda:'+cuda_id if torch.cuda.is_available() else 'cpu')    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split the data into two parts - training data and validation data
    valid_size = int(len(dataset)*valid_ratio)
    train_size = len(dataset) - valid_size
    print('train_size, valid_size:', train_size, valid_size)
    
    dataset_train, dataset_valid = random_split(dataset, [train_size, valid_size], torch.Generator().manual_seed(split_seed))
    dataset_valid.indices.sort()
    #data_local_valid = dataset_valid.data_local
    
    # Dataloader for training
    dataloader_train = DataLoader(dataset_train, config['batch_size'], shuffle=True, num_workers=2, pin_memory=True) #shuffle=False for HybridLoss

    # Dataloader for predicting
    dataloader_valid = DataLoader(dataset_valid, config['batch_size'], shuffle=False, num_workers=1, pin_memory=True)

    if config['transfer_learning']:
        emb_dims = config['emb_dims']
    else:
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
        model_state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(model_state)

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
    
    # Set Optimizer
    if config['optim'] == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    elif config['optim'] == 'AdamW':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        
    elif config['optim'] == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config['weight_decay'], momentum=0.98, nesterov=True)
     
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
            
            print("valid_data_and_prob.iloc[0:10]", valid_data_and_prob.iloc[0:10])
            
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
            
            region_avg = []
            for i in range(n_regions):
                corr_3mer = freq_kmer_comp_multi(valid_data_and_prob.iloc[region_size*i:region_size*(i+1), ], 3, n_class)    
                corr_5mer = freq_kmer_comp_multi(valid_data_and_prob.iloc[region_size*i:region_size*(i+1), ], 5, n_class)
                
                score += np.sum([(1-corr)**2 for corr in corr_3mer]) + np.sum([(1-corr)**2 for corr in corr_5mer])
                
                avg_prob = calc_avg_prob(valid_data_and_prob.iloc[region_size*i:region_size*(i+1)], n_class)
                region_avg.append(avg_prob)
                #print("avg_prob:", avg_prob, i)
            
            region_avg = pd.DataFrame(region_avg)
            corr_list = []
            for i in range(n_class):
                corr_list.append(region_avg[i].corr(region_avg[i + n_class]))
            
            print('corr_list:', corr_list)
            #print('corr_3mer:', corr_3mer)
            #print('corr_5mer:', corr_5mer)
            print('regional score:', score, n_regions)
            
            # Output genomic positions and predicted probabilities
            chr_pos = train_bed.to_dataframe().loc[dataset_valid.indices,['chrom', 'start', 'end', 'strand']].reset_index(drop=True)
            valid_pred_df = pd.concat((chr_pos, valid_data_and_prob[['mut_type'] + prob_names]), axis=1)
            valid_pred_df.columns = ['chrom', 'start', 'end', 'strand', 'mut_type'] + prob_names
            
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
                if save_valid_preds:
                    valid_pred_df.to_csv(path + '.valid_preds.tsv.gz', sep='\t', float_format='%.4g', index=False)

            tune.report(loss=valid_total_loss/valid_size, fdiri_loss=fdiri_nll, score=score, total_params=total_params)