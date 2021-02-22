import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from pybedtools import BedTool

import sys
import argparse
#from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle

from torch.utils.data import random_split


from functools import partial
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import os
import time
import datetime


#from sklearn import metrics, calibration

from NN_utils_test import *
from preprocessing import *
from evaluation import *



#from temperature_scaling import ModelWithTemperature




def parse_arguments(parser):

    parser.add_argument('--train_data', type=str, default='merge.95win.A.pos.101bp.19cols.train.30k.bed.gz',
                        help='path for training data')
    
    parser.add_argument('--test_data', type=str, default='merge.95win.A.pos.101bp.19cols.test.30k.bed.gz',
                        help='path for testing data')
    
    parser.add_argument('--train_data_h5f', type=str, default='', help='path for training data in HDF5 format')
    
    parser.add_argument('--test_data_h5f', type=str, default='', help='path for testing data in HDF5 format')
    
    parser.add_argument('--ref_genome', type=str, default='/public/home/licai/DNMML/data/hg19/hg19_ucsc_ordered.fa',
                        help='reference genome')
    #parser.add_argument('--', type=str, default='', help='')
    parser.add_argument('--bw_paths', type=str, default='/public/home/licai/DNMML/analysis/test/bw_files.txt', help='path for the list of BigWig files for non-sequence features')
    
    parser.add_argument('--n_class', type=int, default='4', help='number of mutation classes')
    
    parser.add_argument('--local_radius', type=int, default=[5], nargs='+', help='radius of local sequences to be considered')
    
    parser.add_argument('--local_order', type=int, default=[1], nargs='+', help='order of local sequences to be considered')
    
    parser.add_argument('--distal_radius', type=int, default=[50], nargs='+', help='radius of distal sequences to be considered')
    
    parser.add_argument('--distal_order', type=int, default=1, help='order of distal sequences to be considered')
    
    parser.add_argument('--emb_4th_root', default=False, action='store_true')
    
    parser.add_argument('--batch_size', type=int, default=[128], nargs='+', help='size of mini batches')
    
    parser.add_argument('--emb_dropout', type=float, default=[0.2], nargs='+', help='dropout rate for k-mer embedding')
    
    parser.add_argument('--local_dropout', type=float, default=[0.15], nargs='+', help='dropout rate for local network')
    
    parser.add_argument('--CNN_kernel_size', type=int, default=[3], nargs='+', help='kernel size for CNN layers')
    
    parser.add_argument('--CNN_out_channels', type=int, default=[32], nargs='+', help='number of output channels for CNN layers')
    
    parser.add_argument('--RNN_hidden_size', type=int, default=0, help='number of hidden neurons for RNN layers')
    
    parser.add_argument('--model_no', type=int, default=2, help=' which NN model to be used')
    
    parser.add_argument('--pred_file', type=str, default='pred.tsv', help='Output file for saving predictions')
    
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
    
    parser.add_argument('--temperature_scaling', default=False, action='store_true')
    
    parser.add_argument('--label_smoothing', default=False, action='store_true')
    
    parser.add_argument('--MultiStepLR', default=False, action='store_true')
    
    parser.add_argument('--mixup', default=False, action='store_true')
    
    parser.add_argument('--resume_ray', default=False, action='store_true', help='resume incomplete Ray experiment')
    
    args = parser.parse_args()

    return args
def main():
    #torch.backends.cudnn.benchmark=True
    
    parser = argparse.ArgumentParser(description='Mutation rate modeling using machine learning')
    args = parse_arguments(parser)
    
    start_time = time.time()
    print('Start time:', datetime.datetime.now())
    
    #request resources
    ray.init(num_cpus=9, num_gpus=1, dashboard_host="0.0.0.0")

    print(' '.join(sys.argv))
    train_file = args.train_data
    test_file = args.test_data   
    train_h5f_path = args.train_data_h5f
    test_h5f_path = args.test_data_h5f   
    ref_genome= args.ref_genome
    local_radius = args.local_radius
    local_order = args.local_order
    distal_radius = args.distal_radius  
    distal_order = args.distal_order
    emb_4th_root = args.emb_4th_root
    batch_size = args.batch_size 
    emb_dropout = args.emb_dropout
    local_dropout = args.local_dropout
    CNN_kernel_size = args.CNN_kernel_size   
    CNN_out_channels = args.CNN_out_channels    
    RNN_hidden_size = args.RNN_hidden_size   
    model_no = args.model_no   
    pred_file = args.pred_file   
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
    
    
    # Read bigWig file names
    bw_paths = args.bw_paths
    bw_files = []
    bw_names = []
    
    
    #print('emb_dropout: ', emb_dropout)
    #print('local_dropout: ', local_dropout)
    #print('CNN_kernel_size:', CNN_kernel_size)
    
    try:
        bw_list = pd.read_table(bw_paths, sep='\s+', header=None, comment='#')
        bw_files = list(bw_list[0])
        bw_names = list(bw_list[1])
    except pd.errors.EmptyDataError:
        print('Warnings: no bigWig files provided')
    
    
    if len(learning_rate) == 1:
        learning_rate = learning_rate*2
    if len(weight_decay) == 1:
        weight_decay = weight_decay*2
    
    train_bed = BedTool(train_file)
    for d_radius in distal_radius:
        h5f_path = get_h5f_path(train_file, bw_names, d_radius, distal_order)
        generate_h5f(train_bed, h5f_path, ref_genome, d_radius, distal_order, bw_files, 1)
    
    config = {
        'local_radius': tune.choice(local_radius),
        'local_order': tune.choice(local_order),
        'distal_radius': tune.choice(distal_radius),
        'emb_dropout': tune.choice(emb_dropout),
        'local_dropout': tune.choice(local_dropout),
        'CNN_kernel_size': tune.choice(CNN_kernel_size),
        'CNN_out_channels': tune.choice(CNN_out_channels),
        'batch_size': tune.choice(batch_size),
        'learning_rate': tune.loguniform(learning_rate[0], learning_rate[1]),
        'optim': tune.choice(optim),
        'LR_gamma': tune.choice(LR_gamma),
        'weight_decay': tune.loguniform(weight_decay[0], weight_decay[1]),
        #'bw_files': bw_files,
        #'bw_names': bw_names,
    }
    
    scheduler = ASHAScheduler(
    metric='loss',
    mode='min',
    max_t=epochs,
    grace_period=grace_period,
    reduction_factor=2)
    
    reporter = CLIReporter(parameter_columns=['local_radius', 'local_order', 'distal_radius', 'emb_dropout', 'local_dropout', 'CNN_out_channels', 'optim', 'learning_rate', 'weight_decay', 'LR_gamma', ], metric_columns=['loss', 'score', 'training_iteration'])
    
    trainable_id = 'Train'
    tune.register_trainable(trainable_id, partial(train, args=args))
                       
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

    #best_trial = result.get_best_trial('loss', 'min', 'last')
    best_trial = result.get_best_trial('loss', 'min', 'last-5-avg')
    print('Best trial config: {}'.format(best_trial.config))
    print('Best trial final validation loss: {}'.format(best_trial.last_result['loss'])) 
    
    best_checkpoint = result.get_best_checkpoint(best_trial, metric='loss', mode='min')
    print('best_checkpoint:', best_checkpoint)
    
    if ray.is_initialized():
        ray.shutdown() 

def get_h5f_path(bed_file, bw_names, distal_radius, distal_order):
    
    h5f_path = bed_file + '.distal_' + str(distal_radius)
    if(distal_order >1):
        h5f_path = h5f_path + '_' + str(distal_order)
    if len(bw_names) > 0:
        h5f_path = h5f_path + '.' + '.'.join(list(bw_names))
    h5f_path = h5f_path + '.h5'
    
    return h5f_path


def train(config, args, checkpoint_dir=None):

    # Set train file
    train_file = args.train_data
    test_file = args.test_data   
    train_h5f_path = args.train_data_h5f
    test_h5f_path = args.test_data_h5f   
    ref_genome= args.ref_genome
    local_radius = args.local_radius
    local_order = args.local_order
    distal_radius = args.distal_radius  
    distal_order = args.distal_order
    emb_4th_root = args.emb_4th_root
    batch_size = args.batch_size 
    local_dropout = args.local_dropout
    CNN_kernel_size = args.CNN_kernel_size   
    CNN_out_channels = args.CNN_out_channels    
    RNN_hidden_size = args.RNN_hidden_size   
    model_no = args.model_no   
    pred_file = args.pred_file   
    optim = args.optim
    learning_rate = args.learning_rate   
    weight_decay = args.weight_decay  
    LR_gamma = args.LR_gamma  
    epochs = args.epochs
    n_class = args.n_class  
    cuda_id = args.cuda_id
    temperature_scaling = args.temperature_scaling
    valid_ratio = args.valid_ratio
    MultiStepLR = args.MultiStepLR
    label_smoothing =args.label_smoothing
    mixup = args.mixup
    
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
    #test_bed = BedTool(test_file)


    train_h5f_path = get_h5f_path(train_file, bw_names, config['distal_radius'], distal_order)
    #test_h5f_path = get_h5f_path(bw_names, config['distal_radius'], distal_order)

    
    # Prepare the datasets for trainging
    dataset = prepare_dataset1(train_bed, ref_genome, bw_files, bw_names, config['local_radius'], config['local_order'], config['distal_radius'], distal_order, train_h5f_path)
    data_local = dataset.data_local
    categorical_features = dataset.cat_cols
    n_cont = len(dataset.cont_cols)
    print('n_cont: ', n_cont)
    
    #train_size = len(dataset)

    # Dataloader for training
    dataloader = DataLoader(dataset, config['batch_size'], shuffle=True, num_workers=2, pin_memory=True) #shuffle=False for HybridLoss

    '''
    # Dataloader for predicting
    dataloader2 = DataLoader(dataset,  config['batch_size'], shuffle=False, num_workers=2)

    # Prepare testing data 
    dataset_test = prepare_dataset1(test_bed, ref_genome, bw_files, bw_names, config['local_radius'], config['local_order'], config['distal_radius'], distal_order, test_h5f_path, 1)
    data_local_test = dataset_test.data_local
    
    test_size = len(dataset_test)

    # Dataloader for testing data
    dataloader1 = DataLoader(dataset_test, batch_size=10, shuffle=False, num_workers=2) 
    '''
    
    print('CUDA is available: ', torch.cuda.is_available())
    device = torch.device('cuda:'+cuda_id if torch.cuda.is_available() else 'cpu')  
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    valid_size = int(len(dataset)*valid_ratio)
    train_size = len(dataset) - valid_size
    print('train_size, valid_size:', train_size, valid_size)
    
    dataset_train, dataset_valid = random_split(dataset, [train_size, valid_size])
    dataset_valid.indices.sort()
    #data_local_valid = dataset_valid.data_local
    
    # Dataloader for training
    dataloader_train = DataLoader(dataset_train,  config['batch_size'], shuffle=True, num_workers=2, pin_memory=True) #shuffle=False for HybridLoss

    # Dataloader for predicting
    dataloader_valid = DataLoader(dataset_valid,  config['batch_size'], shuffle=False, num_workers=1, pin_memory=True)

    
    # Number of categorical features
    #cat_dims = [int(data_local[col].nunique()) for col in categorical_features]
    cat_dims = dataset.cat_dims
    
    #Embedding dimensions for categorical features
    if emb_4th_root:
        emb_dims = [(x, min(16, int(x**0.25))) for x in cat_dims]  
    else:
        emb_dims = [(x, min(16, (x + 1) // 2)) for x in cat_dims]
    #emb_dims
    

    # Choose the network model
    if model_no == 0:
        model = Network(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=cnn_kernel_size, RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=distal_radius, distal_order=distal_order).to(device)

    elif model_no == 1:
        model = Network2(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=CNN_kernel_size, RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=distal_radius, distal_order=distal_order).to(device)

    elif model_no == 2:
        model = Network3m(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']], in_channels=4**distal_order+n_cont, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'], RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=config['distal_radius'], distal_order=distal_order, n_class=n_class, emb_padding_idx=4**config['local_order']).to(device)

    else:
        print('Error: no model selected!')
        sys.exit() 
    
    count_parameters(model)
    print('model:')
    print(model)

    '''
    # FeedForward-only model for comparison
    #model2 = FeedForwardNN(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15]).to(device)
    model2 = FeedForwardNNm(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], n_class=n_class, emb_padding_idx=4**config['local_order']).to(device)
    
    count_parameters(model2)
    print('model2:')
    print(model2)
    '''
    # Initiating weights of the models;
    
    model.apply(weights_init)
    #model2.apply(weights_init)

    # Loss function
    #criterion = torch.nn.BCELoss()
    #criterion = torch.nn.NLLLoss(reduction='mean')
    if label_smoothing:
        criterion = LabelSmoothingCrossEntropy(epsilon=0.1)
        print('using LabelSmoothingCrossEntropy ...')
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    # Set Optimizer
    if config['optim'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        #optimizer2 = torch.optim.Adam(model2.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        
    elif config['optim'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'], momentum=0.98, nesterov=True)
        #optimizer2 = torch.optim.SGD(model2.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'], momentum=0.98, nesterov=True)      
    else:
        print('Error: unsupported optimization method', config['optim'])
        sys.exit()

    if MultiStepLR:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6], gamma=config['LR_gamma'])
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config['LR_gamma'])

    #scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=config['LR_gamma'])
    print('optimizer:', optimizer)
    #print('scheduler, scheduler2:', scheduler, scheduler2)

    best_loss = 0
    pred_df = None
    last_pred_df = None

    '''
    best_loss2 = 0
    pred_df2 = None
    last_pred_df2 = None
    '''
    
    prob_names = ['prob'+str(i) for i in range(n_class)]
    
    # Training
    for epoch in range(epochs):

        model.train()
        #model2.train()

        total_loss = 0
        #total_loss2 = 0
        
        #re-shuffling
        #dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)

        for y, cont_x, cat_x, distal_x in dataloader_train:
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            distal_x = distal_x.to(device)
            y  = y.to(device)
            
            #print('before mixup - cat_x, cont_x, distal_x:', cat_x.shape, cont_x.shape, distal_x.shape)
            
            if not mixup:   
                # Forward Pass
                #preds = model(cont_x, cat_x) #original
                preds = model.forward((cont_x, cat_x), distal_x)
                loss = criterion(preds, y.long().squeeze())
            else:
                cat_x, cont_x, distal_x, y_a, y_b, lam = mixup_data(cat_x, cont_x, distal_x, y.long().squeeze(), alpha=0.2)
                
                #print('cat_x, cont_x, distal_x:', cat_x.shape, cont_x.shape, distal_x.shape)
                preds = model.forward((cont_x, cat_x), distal_x)
                
                loss_func = mixup_criterion(y_a, y_b, lam)
                loss = loss_func(criterion, preds)
                
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            '''
            preds2 = model2.forward(cont_x, cat_x)
            loss2 = criterion(preds2, y.long().squeeze())
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            total_loss2 += loss2.item()
            '''
            #print('in the training loop...')

        model.eval()
        #model2.eval()
        with torch.no_grad():
            print('optimizer learning rate:', optimizer.param_groups[0]['lr'])
            scheduler.step()
            #scheduler2.step()
            
            valid_pred_y, valid_total_loss = model_predict_m(model, dataloader_valid, criterion, device, n_class, distal=True)

            #all_y_prob = pd.Series(data=to_np(F.softmax(all_pred_y)).T[1], name='prob')
            valid_y_prob = pd.DataFrame(data=to_np(F.softmax(valid_pred_y, dim=1)), columns=prob_names)
            valid_data_and_prob = pd.concat([data_local.iloc[dataset_valid.indices, ].reset_index(drop=True), valid_y_prob], axis=1)    
            
            ###############
            #y_prob = pd.DataFrame(data=to_np(F.softmax(pred_y, dim=1)), columns=prob_names)
            #data_and_prob = pd.concat([data_local_test, y_prob], axis=1)
            
            valid_y = valid_data_and_prob['mut_type'].to_numpy().squeeze()
            
            vec_cal, _ = calibrate_prob(valid_y_prob.to_numpy(), valid_y, device, calibr_name='VectS')
            tmp_cal, _ = calibrate_prob(valid_y_prob.to_numpy(), valid_y, device, calibr_name='TempS')
            
            fdiri_cal, _ = calibrate_prob(valid_y_prob.to_numpy(), valid_y, device, calibr_name='FullDiri')
            fdirio_cal, _ = calibrate_prob(valid_y_prob.to_numpy(), valid_y, device, calibr_name='FullDiriODIR')
            ##############
            

            # Compare observed/predicted 3/5/7mer mutation frequencies
            print('3mer correlation - all: ', freq_kmer_comp_multi(valid_data_and_prob, 3, n_class))
            print('5mer correlation - all: ', freq_kmer_comp_multi(valid_data_and_prob, 5, n_class))
            print('7mer correlation - all: ', freq_kmer_comp_multi(valid_data_and_prob, 7, n_class))
            
            print ('Validation Loss: ', valid_total_loss/valid_size)  
            
            ###############
            region_size = 10000
            n_regions = valid_size//region_size
            
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
            ###############
            
            chr_pos = train_bed.to_dataframe().loc[dataset_valid.indices,['chrom', 'start', 'end']].reset_index(drop=True)
            valid_pred_df = pd.concat((chr_pos, valid_data_and_prob[['mut_type'] + prob_names]), axis=1)
            valid_pred_df.columns = ['chrom', 'start', 'end','mut_type'] + prob_names
            
            print('valid_pred_df: ', valid_pred_df.head())
    
            for win_size in [20000, 100000, 500000]:
                #corr = corr_calc(valid_pred_df, win_size, 'valid_prob')
                corr_win = corr_calc_sub(valid_pred_df, win_size, prob_names)
                print('regional corr (validation):', str(win_size)+'bp', corr_win)
            
            ##################
            #valid_pred_df = pd.concat((chr_pos, valid_data_and_prob), axis=1)
            #valid_pred_df.columns = ['chrom', 'start', 'end'] + valid_data_and_prob.columns
            
            ###################
            
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, 'checkpoint')
                torch.save((model.state_dict(), optimizer.state_dict()), path)
            
                with open(path + '.fdiri_cal.pkl', 'wb') as pkl_file:
                    pickle.dump(fdiri_cal, pkl_file)
                with open(path + '.vec_cal.pkl', 'wb') as pkl_file:
                    pickle.dump(vec_cal, pkl_file)

            tune.report(loss=valid_total_loss/valid_size, score=score)
    #print('Total time used: %s seconds' % (time.time() - start_time))
                
        ################
            if temperature_scaling and epoch > 5:
                modelS = ModelWithTemperature(model)
                modelS.set_temperature(dataloader_valid, device)
                
                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, 'checkpointS')
                    torch.save((modelS.state_dict(), optimizer.state_dict()), path)
                
                valid_pred_y, valid_total_loss = model_predict_m(modelS, dataloader_valid, criterion, device, n_class, distal=True)

                valid_y_prob = pd.DataFrame(data=to_np(F.softmax(valid_pred_y, dim=1)), columns=prob_names)
                valid_data_and_prob = pd.concat([data_local.iloc[dataset_valid.indices, ].reset_index(drop=True), valid_y_prob], axis=1)        

                print('3mer correlation (scaling): ', freq_kmer_comp_multi(valid_data_and_prob, 3, n_class))
                print('5mer correlation (scaling):  ', freq_kmer_comp_multi(valid_data_and_prob, 5, n_class))
                print('7mer correlation (scaling): ', freq_kmer_comp_multi(valid_data_and_prob, 7, n_class))

                print ('Validation Loss  (scaling): ', valid_total_loss/valid_size)
            
                ################Vector scaling
                #y= valid_data_and_prob[['mut_type']
                
                ################
                
                #####
                chr_pos = train_bed.to_dataframe().loc[dataset_valid.indices,['chrom', 'start', 'end']].reset_index(drop=True)
                valid_pred_df = pd.concat((chr_pos, valid_data_and_prob[['mut_type'] + prob_names]), axis=1)
                valid_pred_df.columns = ['chrom', 'start', 'end','mut_type'] + prob_names
                print('valid_pred_df: ', valid_pred_df.head())

                for win_size in [20000, 100000, 500000]:
                    #corr = corr_calc(valid_pred_df, win_size, 'valid_prob')
                    corr_win = corr_calc_sub(valid_pred_df, win_size, prob_names)
                    print('regional corr (scaling):', str(win_size)+'bp', corr_win)

    
    
if __name__ == '__main__':
    main()


