import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from pybedtools import BedTool

import sys
import argparse
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import numpy as np

import os
import time
import datetime


from sklearn import metrics, calibration

from NN_utils import *
from preprocessing import *
from evaluation import *

from torch.utils.data import random_split, Subset
import random


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
    #parser.add_argument('--', type=str, default='', help='')
    parser.add_argument('--bw_paths', type=str, default='/public/home/licai/DNMML/analysis/test/bw_files.txt', help='path for the list of BigWig files for non-sequence features')
    
    parser.add_argument('--local_radius', type=int, default='5', help='radius of local sequences to be considered')
    
    parser.add_argument('--distal_radius', type=int, default='50', help='radius of distal sequences to be considered')
    
    parser.add_argument('--distal_order', type=int, default='1', help='order of distal sequences to be considered')
    
    parser.add_argument('--batch_size', type=int, default='200', help='size of mini batches')
    
    parser.add_argument('--CNN_kernel_size', type=int, default='3', help='kernel size for CNN layers')
    
    parser.add_argument('--CNN_out_channels', type=int, default='60', help='number of output channels for CNN layers')
    
    parser.add_argument('--RNN_hidden_size', type=int, default='0', help='number of hidden neurons for RNN layers')
    
    parser.add_argument('--model_no', type=int, default='2', help=' which NN model to be used')
    
    parser.add_argument('--pred_file', type=str, default='pred.tsv', help='Output file for saving predictions')
    
    parser.add_argument('--learning_rate', type=float, default='0.005', help='learning rate for training')
    
    parser.add_argument('--weight_decay', type=float, default='1e-5', help='weight decay (regularization) for training')
    
    parser.add_argument('--LR_gamma', type=float, default='0.5', help='gamma for learning rate change during training')
    
    parser.add_argument('--epochs', type=int, default='15', help='numbe of epochs for training')
    
    args = parser.parse_args()

    return args
def main2():
    parser = argparse.ArgumentParser(description='Mutation rate modeling using machine learning')
    args = parse_arguments(parser)
    
    start_time = time.time()
    print('Start time:', datetime.datetime.now())

    print("CUDA: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(' '.join(sys.argv))

    # Set train file
    train_file = args.train_data
    test_file = args.test_data   
    train_h5f_path = args.train_data_h5f
    test_h5f_path = args.test_data_h5f   
    ref_genome= args.ref_genome
    local_radius = args.local_radius    
    distal_radius = args.distal_radius  
    distal_order = args.distal_order
    batch_size = args.batch_size  
    CNN_kernel_size = args.CNN_kernel_size   
    CNN_out_channels = args.CNN_out_channels    
    RNN_hidden_size = args.RNN_hidden_size   
    model_no = args.model_no   
    pred_file = args.pred_file   
    learning_rate = args.learning_rate   
    weight_decay = args.weight_decay  
    LR_gamma = args.LR_gamma  
    epochs = args.epochs
    
    # Read BED files
    train_bed = BedTool(train_file)
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
    
    if len(train_h5f_path) == 0:
        if len(bw_names) > 0:
            train_h5f_path = train_file + '.distal_' + str(distal_radius) + '.'.join(list(bw_names)) + '.h5'
        else:
            train_h5f_path = train_file + '.distal_' + str(distal_radius) + '.'.join(list(bw_names)) + '.h5'
    
    if len(test_h5f_path) == 0:
        if len(bw_names) > 0:
            test_h5f_path = test_file + '.distal_' + str(distal_radius) + '.'.join(list(bw_names)) + '.h5'
        else:
            test_h5f_path = test_file + '.distal_' + str(distal_radius) + '.'.join(list(bw_names)) + '.h5'
    
    # Prepare the datasets for trainging
    dataset, data_local, categorical_features = prepare_dataset2(train_bed, ref_genome, bw_files, bw_names, local_radius, distal_radius, distal_order, train_h5f_path)
    #data_local.to_csv('data_local.tsv', sep='\t', index=False)
    
    train_size = int(len(dataset)*0.7)
    valid_size = len(dataset) - train_size
    print('train_size, valid_size:', train_size, valid_size)
    
    dataset_train, dataset_valid = random_split(dataset, [train_size, valid_size])
    dataset_valid.indices.sort()
    
    #dataset_valid = Subset(dataset, dataset_valid.indices)

    # Dataloader for training
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2) #shuffle=False for HybridLoss

    # Dataloader for predicting
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=1)

    # Number of categorical features
    cat_dims = [int(data_local[col].nunique()) for col in categorical_features]

    #Embedding dimensions for categorical features
    emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
    #emb_dims

    # Prepare testing data 
    dataset_test, data_local_test, _ = prepare_dataset2(test_bed, ref_genome, bw_files, bw_names, local_radius, distal_radius, distal_order, test_h5f_path, 1)
    
    test_size = len(dataset_test)

    # Dataloader for testing data
    dataloader_test = DataLoader(dataset_test, batch_size=10, shuffle=False, num_workers=2)

    # Choose the network model
    if model_no == 0:
        model = Network(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=cnn_kernel_size, RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=distal_radius, distal_order=distal_order).to(device)

    elif model_no == 1:
        model = Network2(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=CNN_kernel_size, RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=distal_radius, distal_order=distal_order).to(device)

    elif model_no == 2:
        model = Network3(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=CNN_kernel_size, RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=distal_radius, distal_order=distal_order).to(device)

    elif model_no == 3:
        model = Network4(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=CNN_kernel_size, RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=distal_radius, distal_order=distal_order).to(device)

    else:
        print('Error: no model selected!')
        sys.exit() 

    print('model:')
    print(model)

    # FeedForward-only model for comparison
    model2 = FeedForwardNN(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15]).to(device)
    print('model2:')
    print(model2)

    # Initiating weights of the models;
    model.apply(weights_init)
    model2.apply(weights_init)

    # Loss function
    criterion = torch.nn.BCELoss()

    # Set Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LR_gamma)

    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=LR_gamma)
    print('optimizer, optimizer2:', optimizer, optimizer2)
    #print('scheduler, scheduler2:', scheduler, scheduler2)

    best_loss = 0
    pred_df = None
    last_pred_df = None

    best_loss2 = 0
    pred_df2 = None
    last_pred_df2 = None
    
    # Training
    for epoch in range(epochs):

        model.train()
        model2.train()

        total_loss = 0
        total_loss2 = 0

        torch.cuda.empty_cache()
        
        #re-shuffling
        #dataloader = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=2)

        for y, cont_x, cat_x, distal_x in dataloader_train:
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            distal_x = distal_x.to(device)
            y  = y.to(device)

            # Forward Pass
            #preds = model(cont_x, cat_x) #original
            preds = model.forward((cont_x, cat_x), distal_x)  
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds2 = model2.forward(cont_x, cat_x)
            loss2 = criterion(preds2, y) 
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            total_loss += loss.item()
            total_loss2 += loss2.item()
            #print('in the training loop...')

        model.eval()
        model2.eval()
        with torch.no_grad():

            print('optimizer learning rate:', optimizer.param_groups[0]['lr'])
            scheduler.step()
            scheduler2.step()

            #if epoch <5:
            #    continue

            # Do predictions for training data
            #valid_pred_y, valid_total_loss = model.batch_predict(dataloader_valid, criterion, device)      
            valid_pred_y, valid_total_loss, valid_pred_y2, valid_total_loss2 = two_model_predict(model, model2, dataloader_valid, criterion, device)

            valid_y_prob = pd.Series(data=to_np(valid_pred_y).T[0], name="prob")
            
            valid_data_and_prob = pd.concat([data_local.iloc[dataset_valid.indices, ].reset_index(drop=True), valid_y_prob], axis=1)
            #print('valid_data_and_prob', valid_data_and_prob.iloc[1:10,], valid_data_and_prob.shape, a.shape, data_local.shape, len(dataset_valid.indices), len(valid_y_prob))

            #valid_pred_y2, valid_total_loss2 = model2.batch_predict(dataloader_valid, criterion, device)     
            valid_y_prob2 = pd.Series(data=to_np(valid_pred_y2).T[0], name="prob")
            valid_data_and_prob2 = pd.concat([data_local.iloc[dataset_valid.indices, ].reset_index(drop=True), valid_y_prob2], axis=1)
            #print('valid_data_and_prob2', valid_data_and_prob2.iloc[1:10,])

            # Compare observed/predicted 3/5/7mer mutation frequencies
            print ('3mer correlation - valid: ' + str(f3mer_comp(valid_data_and_prob)))
            print ('5mer correlation - valid: ' + str(f5mer_comp(valid_data_and_prob)))
            print ('7mer correlation - valid: ' + str(f7mer_comp(valid_data_and_prob)))


            print ('3mer correlation - valid (FF only): ' + str(f3mer_comp(valid_data_and_prob2)))
            print ('5mer correlation - valid (FF only): ' + str(f5mer_comp(valid_data_and_prob2)))
            print ('7mer correlation - valid (FF only): ' + str(f7mer_comp(valid_data_and_prob2)))

            print ('Total Loss: ', valid_total_loss/valid_size, valid_total_loss2/valid_size)        
            
            valid_pred_df = pd.concat((train_bed.to_dataframe().loc[dataset_valid.indices, ['chrom', 'start', 'end']].reset_index(drop=True), valid_data_and_prob[['mut_type','prob']], valid_data_and_prob2['prob']), axis=1)
            valid_pred_df.columns = ['chrom', 'start', 'end','mut_type', 'valid_prob', 'valid_prob2']
            
            for win_size in [10000, 100000]:
                corr1 = corr_calc(valid_pred_df, win_size, 'valid_prob')
                corr2 = corr_calc(valid_pred_df, win_size, 'valid_prob2')
                print('regional corr (validation):', str(win_size)+'bp', corr1, corr2)
    
    # Do predictions for testing data
    #pred_y, test_total_loss = model.batch_predict(dataloader_test, criterion, device)
    pred_y, test_total_loss, pred_y2, test_total_loss2 = two_model_predict(model, model2, dataloader_test, criterion, device)
    y_prob = pd.Series(data=to_np(pred_y).T[0], name="prob")    
    data_and_prob = pd.concat([data_local_test, y_prob], axis=1) 
    #print('data_and_prob', data_and_prob.iloc[1:10,], data_and_prob.shape)

    # For FeedForward-only model
    #pred_y2, test_total_loss2 = model2.batch_predict(dataloader_test, criterion, device)
    y_prob2 = pd.Series(data=to_np(pred_y2).T[0], name="prob")    
    data_and_prob2 = pd.concat([data_local_test, y_prob2], axis=1)

    print ('3mer correlation - test: ' + str(f3mer_comp(data_and_prob)))
    print ('5mer correlation - test: ' + str(f5mer_comp(data_and_prob)))
    print ('7mer correlation - test: ' + str(f7mer_comp(data_and_prob)))
    print ('3mer correlation - test (FF only): ' + str(f3mer_comp(data_and_prob2)))
    print ('5mer correlation - test (FF only): ' + str(f5mer_comp(data_and_prob2)))
    print ('7mer correlation - test (FF only): ' + str(f7mer_comp(data_and_prob2)))


    last_pred_df = data_and_prob[['mut_type','prob']]
    last_pred_df2 = data_and_prob2[['mut_type','prob']]

    torch.save(model.state_dict(), pred_file+'.model1')
    torch.save(model2.state_dict(), pred_file+'.model2')

    # Get the scores
    #auc_score = metrics.roc_auc_score(to_np(test_y), to_np(pred_y))
    test_y = data_local_test['mut_type']

    # Print some data for debugging
    print('min and max of pred_y:', np.min(to_np(pred_y)), np.max(to_np(pred_y)))
    print('min and max of pred_y2:', np.min(to_np(pred_y2)), np.max(to_np(pred_y2)))

    # Write the prediction
    pred_df = pd.concat((test_bed.to_dataframe()[['chrom', 'start', 'end']], last_pred_df, last_pred_df2['prob']), axis=1)
    pred_df.columns = ['chrom', 'start', 'end','mut_type', 'last_prob', 'last_prob2']

    pred_df.to_csv(pred_file, sep='\t', index=False)
    
    for win_size in [1000, 5000,20000, 50000, 100000]:
        corr1 = corr_calc(pred_df, win_size, 'last_prob')
        corr2 = corr_calc(pred_df, win_size, 'last_prob2')
        print('regional corr:', str(win_size)+'bp', corr1, corr2)
    #os.remove(train_h5f_path)
    #os.remove(test_h5f_path)

    print('Total time used: %s seconds' % (time.time() - start_time))

def main():
    parser = argparse.ArgumentParser(description='Mutation rate modeling using machine learning')
    args = parse_arguments(parser)
    
    start_time = time.time()
    print('Start time:', datetime.datetime.now())

    print("CUDA: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(' '.join(sys.argv))

    # Set train file
    train_file = args.train_data
    test_file = args.test_data   
    train_h5f_path = args.train_data_h5f
    test_h5f_path = args.test_data_h5f   
    ref_genome= args.ref_genome
    local_radius = args.local_radius    
    distal_radius = args.distal_radius  
    distal_order = args.distal_order
    batch_size = args.batch_size  
    CNN_kernel_size = args.CNN_kernel_size   
    CNN_out_channels = args.CNN_out_channels    
    RNN_hidden_size = args.RNN_hidden_size   
    model_no = args.model_no   
    pred_file = args.pred_file   
    learning_rate = args.learning_rate   
    weight_decay = args.weight_decay  
    LR_gamma = args.LR_gamma  
    epochs = args.epochs
    
    # Read BED files
    train_bed = BedTool(train_file)
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
    
    if len(train_h5f_path) == 0:
        if len(bw_names) > 0:
            train_h5f_path = train_file + '.distal_' + str(distal_radius) + '.'.join(list(bw_names)) + '.h5'
        else:
            train_h5f_path = train_file + '.distal_' + str(distal_radius) + '.'.join(list(bw_names)) + '.h5'
    
    if len(test_h5f_path) == 0:
        if len(bw_names) > 0:
            test_h5f_path = test_file + '.distal_' + str(distal_radius) + '.'.join(list(bw_names)) + '.h5'
        else:
            test_h5f_path = test_file + '.distal_' + str(distal_radius) + '.'.join(list(bw_names)) + '.h5'
    
    # Prepare the datasets for trainging
    #dataset, data_local, categorical_features = prepare_dataset2(train_bed, ref_genome, bw_files, bw_names, local_radius, distal_radius, distal_order, train_h5f_path)
    dataset = prepare_dataset2(train_bed, ref_genome, bw_files, bw_names, local_radius, distal_radius, distal_order, train_h5f_path)
    data_local = dataset.data_local
    categorical_features = dataset.cat_cols
    #data_local.to_csv('data_local.tsv', sep='\t', index=False)
    
    # Prepare testing data 
    #dataset_test, data_local_test, _ = prepare_dataset2(test_bed, ref_genome, bw_files, bw_names, local_radius, distal_radius, distal_order, test_h5f_path, 1)
    dataset_test = prepare_dataset2(test_bed, ref_genome, bw_files, bw_names, local_radius, distal_radius, distal_order, test_h5f_path, 1)
    data_local_test = dataset_test.data_local
    
    # Number of categorical features
    cat_dims = [int(data_local[col].nunique()) for col in categorical_features]

    #Embedding dimensions for categorical features
    emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
    #emb_dims

    # Choose the network model
    if model_no == 0:
        model = Network(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=cnn_kernel_size, RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=distal_radius, distal_order=distal_order).to(device)

    elif model_no == 1:
        model = Network2(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=CNN_kernel_size, RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=distal_radius, distal_order=distal_order).to(device)

    elif model_no == 2:
        model = Network3(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=CNN_kernel_size, RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=distal_radius, distal_order=distal_order).to(device)

    elif model_no == 3:
        model = Network4(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=CNN_kernel_size, RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=distal_radius, distal_order=distal_order).to(device)

    else:
        print('Error: no model selected!')
        sys.exit() 

    print('model:')
    print(model)

    # FeedForward-only model for comparison
    model2 = FeedForwardNN(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15]).to(device)
    print('model2:')
    print(model2)


    # Loss function
    criterion = torch.nn.BCELoss()
    
    #
    model, model2 = train_model(model, model2, train_bed, dataset, data_local, batch_size, learning_rate, weight_decay, LR_gamma, epochs, criterion, device)
    
    test_model(model, model2, test_bed, dataset_test, data_local_test, pred_file, criterion, device)


def train_one_epoch(model, model2, dataloader_train, optimizer, optimizer2, criterion, device):
    model.train()
    model2.train()

    total_loss = 0
    total_loss2 = 0

    torch.cuda.empty_cache()


    for y, cont_x, cat_x, distal_x in dataloader_train:
        cat_x = cat_x.to(device)
        cont_x = cont_x.to(device)
        distal_x = distal_x.to(device)
        y  = y.to(device)

        # Forward Pass
        #preds = model(cont_x, cat_x) #original
        preds = model.forward((cont_x, cat_x), distal_x)  
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds2 = model2.forward(cont_x, cat_x)
        loss2 = criterion(preds2, y) 
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        total_loss += loss.item()
        total_loss2 += loss2.item()
        #print('in the training loop...')  
    
    return model, model2, total_loss, total_loss2

def calc_eval_score(valid_pred_y, data_local, dataset_valid, train_bed):
    
    valid_y_prob = pd.Series(data=to_np(valid_pred_y).T[0], name="prob")

    valid_data_and_prob = pd.concat([data_local.iloc[dataset_valid.indices, ].reset_index(drop=True), valid_y_prob], axis=1)

    # Compare observed/predicted 3/5/7mer mutation frequencies
    print ('3mer correlation - valid: ' + str(f3mer_comp(valid_data_and_prob)))
    print ('5mer correlation - valid: ' + str(f5mer_comp(valid_data_and_prob)))
    print ('7mer correlation - valid: ' + str(f7mer_comp(valid_data_and_prob)))

    valid_pred_df = pd.concat((train_bed.to_dataframe().loc[dataset_valid.indices, ['chrom', 'start', 'end']].reset_index(drop=True), valid_data_and_prob[['mut_type','prob']]), axis=1)
    valid_pred_df.columns = ['chrom', 'start', 'end','mut_type', 'valid_prob']
    
    score = 0
    for win_size in [10000, 100000]:
        corr = corr_calc(valid_pred_df, win_size, 'valid_prob')
        print('regional corr (validation):', str(win_size)+'bp', corr)
        score += 1 - corr**2
        
    return score


def two_model_eval(valid_pred_y, valid_pred_y2, data_local, dataset_valid, train_bed):
    
    valid_y_prob = pd.Series(data=to_np(valid_pred_y).T[0], name="prob")

    valid_data_and_prob = pd.concat([data_local.iloc[dataset_valid.indices, ].reset_index(drop=True), valid_y_prob], axis=1)

    valid_y_prob2 = pd.Series(data=to_np(valid_pred_y2).T[0], name="prob")
    valid_data_and_prob2 = pd.concat([data_local.iloc[dataset_valid.indices, ].reset_index(drop=True), valid_y_prob2], axis=1)
    #print('valid_data_and_prob2', valid_data_and_prob2.iloc[1:10,])

    # Compare observed/predicted 3/5/7mer mutation frequencies
    print ('3mer correlation - valid: ' + str(f3mer_comp(valid_data_and_prob)))
    print ('5mer correlation - valid: ' + str(f5mer_comp(valid_data_and_prob)))
    print ('7mer correlation - valid: ' + str(f7mer_comp(valid_data_and_prob)))


    print ('3mer correlation - valid (FF only): ' + str(f3mer_comp(valid_data_and_prob2)))
    print ('5mer correlation - valid (FF only): ' + str(f5mer_comp(valid_data_and_prob2)))
    print ('7mer correlation - valid (FF only): ' + str(f7mer_comp(valid_data_and_prob2)))

    valid_pred_df = pd.concat((train_bed.to_dataframe().loc[dataset_valid.indices, ['chrom', 'start', 'end']].reset_index(drop=True), valid_data_and_prob[['mut_type','prob']], valid_data_and_prob2['prob']), axis=1)
    valid_pred_df.columns = ['chrom', 'start', 'end','mut_type', 'valid_prob', 'valid_prob2']
    
    #corr1 = corr2 = 0
    for win_size in [10000, 100000]:
        corr1 = corr_calc(valid_pred_df, win_size, 'valid_prob')
        corr2 = corr_calc(valid_pred_df, win_size, 'valid_prob2')
        print('regional corr (validation):', str(win_size)+'bp', corr1, corr2)
        

class HyperParam():

    def __init__(self, learning_rate, weight_decay, LR_gamma, epochs):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.LR_gamma = LR_gamma
        self.epochs = epochs

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate    
        return self
    
    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay 
        return self
    
    def set_LR_gamma(self, LR_gamma):
        self.LR_gamma = LR_gamma
        return self
    
    def set_epochs(self, epochs):
        self.epochs = epochs
        return self
    
def get_learning_hyperparam(model, train_bed, dataloader_train, dataloader_valid, dataset_valid, data_local, criterion, device):
    
    best_score = 100
    best_trial = 0
    best_loss = 100

    model.apply(weights_init)
    untrained_model = model
    
    best_hyper_param = HyperParam(learning_rate=0, weight_decay=0, LR_gamma=0, epochs=0)
    
    for number in range(10):
        
        learning_rate_list = [1e-2, 5e-3, 1e-3]
        learning_rate = random.choice(learning_rate_list)
        weight_decay_list = [1e-4, 5e-5, 1e-5]
        weight_decay = random.choice(weight_decay_list)
        LR_gamma_list = [0.5, 0.6, 0.7]
        LR_gamma = random.choice(LR_gamma_list)
        epochs_list = [8, 10, 15]
        epochs = random.choice(epochs_list)

        print('learning_rate:', learning_rate)
        print('weight_decay:', weight_decay)
        print('LR_gamma:', LR_gamma)
        print('epochs:', epochs)

        model = untrained_model
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LR_gamma)
        
        
        # Training
        for epoch in range(epochs):
            model.train()

            for y, cont_x, cat_x, distal_x in dataloader_train:
                cat_x = cat_x.to(device)
                cont_x = cont_x.to(device)
                distal_x = distal_x.to(device)
                y  = y.to(device)

                # Forward Pass
                preds = model.forward((cont_x, cat_x), distal_x)  
                loss = criterion(preds, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #print('optimizer learning rate:', optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        with torch.no_grad():

            valid_pred_y, valid_total_loss = model.batch_predict(dataloader_valid, criterion, device)

            score = calc_eval_score(valid_pred_y, data_local, dataset_valid, train_bed)
            valid_total_loss = valid_total_loss/len(valid_pred_y)
            print('Trial',number+1, 'valid_total_loss, score:', valid_total_loss, score)
            
            #if score < best_score:
            if valid_total_loss < best_loss:
                    best_hyper_param.set_learning_rate(learning_rate)
                    best_hyper_param.set_weight_decay(weight_decay)
                    best_hyper_param.set_LR_gamma(LR_gamma)
                    best_hyper_param.set_epochs(epochs)
                    #best_score = score
                    best_loss = valid_total_loss
                    best_trial = number + 1
    
    print('best_trial:', best_trial)
    print('best_learning_rate:', best_hyper_param.learning_rate)
    print('best_weight_decay:', best_hyper_param.weight_decay)
    print('best_LR_gamma:', best_hyper_param.LR_gamma)
    print('best_epochs:', best_hyper_param.epochs)
    
    #return best_learning_rate, best_weight_decay, best_LR_gamma, best_epochs
    return best_hyper_param


def get_learning_hyperparamCV(model, train_bed, dataset, data_local, batch_size, criterion, device):
    
    split_size = len(dataset)//3
    split_sizes = [split_size, split_size, len(dataset) - 2*split_size]
    
    #train_size = int(len(dataset)*0.7)
    #valid_size = len(dataset) - train_size
    #print('train_size, valid_size:', train_size, valid_size)
    
    dataset_split1, dataset_split2, dataset_split3 = random_split(dataset, split_sizes)
    
    dataset_train1 = ConcatDataset([dataset_split1, dataset_split2])
    dataset_valid1 = dataset_split3
    dataset_valid1.indices.sort()
    
    dataset_train2 = ConcatDataset([dataset_split1, dataset_split3])
    dataset_valid2 = dataset_split2
    dataset_valid2.indices.sort()
    
    dataset_train3 = ConcatDataset([dataset_split2, dataset_split3])
    dataset_valid3 = dataset_split1
    dataset_valid3.indices.sort()

    # Dataloader for training
    #dataloader_train1 = DataLoader(dataset_train1, batch_size=batch_size, shuffle=True, num_workers=2) 
    #dataloader_valid1 = DataLoader(dataset_valid1, batch_size=batch_size, shuffle=False, num_workers=1)
    
    #dataloader_train2 = DataLoader(dataset_train2, batch_size=batch_size, shuffle=True, num_workers=2) 
    #dataloader_valid2 = DataLoader(dataset_valid2, batch_size=batch_size, shuffle=False, num_workers=1)

    #dataloader_train3 = DataLoader(dataset_train3, batch_size=batch_size, shuffle=True, num_workers=2) 
    #dataloader_valid3 = DataLoader(dataset_valid3, batch_size=batch_size, shuffle=False, num_workers=1)
    
    best_trial = 0
    best_loss = 100
    best_score = 100

    model.apply(weights_init)
    untrained_model = model
    
    best_hyper_param = HyperParam(learning_rate=0, weight_decay=0, LR_gamma=0, epochs=0)
    
    for number in range(10):
        
        learning_rate_list = [1e-2, 5e-3, 1e-3]
        learning_rate = random.choice(learning_rate_list)
        weight_decay_list = [1e-4, 5e-5, 1e-5]
        weight_decay = random.choice(weight_decay_list)
        LR_gamma_list = [0.5, 0.6, 0.7]
        LR_gamma = random.choice(LR_gamma_list)
        epochs_list = [8, 10, 15]
        epochs = random.choice(epochs_list)

        print('learning_rate:', learning_rate)
        print('weight_decay:', weight_decay)
        print('LR_gamma:', LR_gamma)
        print('epochs:', epochs)
        
        score_list = []
        valid_total_loss_list = []

        for dataset_train, dataset_valid  in ((dataset_train1, dataset_valid1), (dataset_train2, dataset_valid2,), (dataset_train3, dataset_valid3)):
            
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2) 
            dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=1)
    
            
            model = untrained_model      
            
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LR_gamma)


            # Training
            for epoch in range(epochs):
                model.train()

                for y, cont_x, cat_x, distal_x in dataloader_train:
                    cat_x = cat_x.to(device)
                    cont_x = cont_x.to(device)
                    distal_x = distal_x.to(device)
                    y  = y.to(device)

                    # Forward Pass
                    preds = model.forward((cont_x, cat_x), distal_x)  
                    loss = criterion(preds, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                #print('optimizer learning rate:', optimizer.param_groups[0]['lr'])
                scheduler.step()

            with torch.no_grad():

                valid_pred_y, valid_total_loss = model.batch_predict(dataloader_valid, criterion, device)

                score = calc_eval_score(valid_pred_y, data_local, dataset_valid, train_bed)
                valid_total_loss = valid_total_loss/len(valid_pred_y)
                
                score_list.append(score)
                valid_total_loss_list.append(valid_total_loss)
            
        print('Trial',number+1, 'valid_total_loss, score:', valid_total_loss_list, score_list)
        #if score < best_score:
        if np.mean(valid_total_loss_list) < best_loss:
                best_hyper_param.set_learning_rate(learning_rate)
                best_hyper_param.set_weight_decay(weight_decay)
                best_hyper_param.set_LR_gamma(LR_gamma)
                best_hyper_param.set_epochs(epochs)
                #best_score = score
                best_loss = np.mean(valid_total_loss_list)
                best_trial = number + 1
    
    print('best_trial:', best_trial)
    print('best_learning_rate:', best_hyper_param.learning_rate)
    print('best_weight_decay:', best_hyper_param.weight_decay)
    print('best_LR_gamma:', best_hyper_param.LR_gamma)
    print('best_epochs:', best_hyper_param.epochs)
    
    #return best_learning_rate, best_weight_decay, best_LR_gamma, best_epochs
    return best_hyper_param
    
        
def train_model(model, model2, train_bed, dataset, data_local, batch_size, learning_rate, weight_decay, LR_gamma, epochs, criterion, device):

    train_size = int(len(dataset)*0.7)
    valid_size = len(dataset) - train_size
    print('train_size, valid_size:', train_size, valid_size)
    
    dataset_train, dataset_valid = random_split(dataset, [train_size, valid_size])
    dataset_valid.indices.sort()
    
    # Dataloader for training
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2) #shuffle=False for HybridLoss

    # Dataloader for predicting
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=1)
    
    hyper_param = HyperParam(learning_rate=learning_rate, weight_decay=weight_decay, LR_gamma=LR_gamma, epochs=epochs)
    #################
    #learning_rate, weight_decay, LR_gamma, epochs = get_learning_hyperparam(model, train_bed, dataloader_train, dataloader_valid, dataset_valid, data_local, criterion, device)
    #hyper_param = get_learning_hyperparam(model, train_bed, dataloader_train, dataloader_valid, dataset_valid, data_local, criterion, device)
    hyper_param = get_learning_hyperparamCV(model, train_bed, dataset, data_local, batch_size, criterion, device)
    ###################
    
    model.apply(weights_init)
    model2.apply(weights_init)

    # Set Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_param.learning_rate, weight_decay=hyper_param.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=hyper_param.LR_gamma)

    optimizer2 = torch.optim.Adam(model2.parameters(), lr=hyper_param.learning_rate, weight_decay=hyper_param.weight_decay)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=hyper_param.LR_gamma)
    print('optimizer, optimizer2:', optimizer, optimizer2)
    #print('scheduler, scheduler2:', scheduler, scheduler2)

    best_loss = 0
    pred_df = None
    last_pred_df = None

    best_loss2 = 0
    pred_df2 = None
    last_pred_df2 = None
    
    # Training
    for epoch in range(hyper_param.epochs):
        model.train()
        model2.train()
        
        model, model2, total_loss, total_loss2 = train_one_epoch(model, model2, dataloader_train,  optimizer, optimizer2, criterion, device)

        model.eval()
        model2.eval()
        
        print('optimizer learning rate:', optimizer.param_groups[0]['lr'])
        scheduler.step()
        scheduler2.step()
        
        with torch.no_grad():

            valid_pred_y, valid_total_loss, valid_pred_y2, valid_total_loss2 = two_model_predict(model, model2, dataloader_valid, criterion, device)

            two_model_eval(valid_pred_y, valid_pred_y2, data_local, dataset_valid, train_bed)
            
            print ("Total Loss: ", valid_total_loss/valid_size, valid_total_loss2/valid_size)
    
    return model, model2

def test_model(model, model2, test_bed, dataset_test, data_local_test, pred_file, criterion, device):
   
    test_size = len(dataset_test)

    # Dataloader for testing data
    dataloader_test = DataLoader(dataset_test, batch_size=10, shuffle=False, num_workers=2)
    
    # Do predictions for testing data
    pred_y, test_total_loss, pred_y2, test_total_loss2 = two_model_predict(model, model2, dataloader_test, criterion, device)
    y_prob = pd.Series(data=to_np(pred_y).T[0], name="prob")    
    data_and_prob = pd.concat([data_local_test, y_prob], axis=1) 
    #print('data_and_prob', data_and_prob.iloc[1:10,], data_and_prob.shape)

    # For FeedForward-only model
    y_prob2 = pd.Series(data=to_np(pred_y2).T[0], name="prob")    
    data_and_prob2 = pd.concat([data_local_test, y_prob2], axis=1)

    print ('3mer correlation - test: ' + str(f3mer_comp(data_and_prob)))
    print ('5mer correlation - test: ' + str(f5mer_comp(data_and_prob)))
    print ('7mer correlation - test: ' + str(f7mer_comp(data_and_prob)))
    print ('3mer correlation - test (FF only): ' + str(f3mer_comp(data_and_prob2)))
    print ('5mer correlation - test (FF only): ' + str(f5mer_comp(data_and_prob2)))
    print ('7mer correlation - test (FF only): ' + str(f7mer_comp(data_and_prob2)))


    last_pred_df = data_and_prob[['mut_type','prob']]
    last_pred_df2 = data_and_prob2[['mut_type','prob']]

    torch.save(model.state_dict(), pred_file+'.model1')
    torch.save(model2.state_dict(), pred_file+'.model2')

    # Get the scores
    #auc_score = metrics.roc_auc_score(to_np(test_y), to_np(pred_y))
    test_y = data_local_test['mut_type']

    # Print some data for debugging
    print('min and max of pred_y:', np.min(to_np(pred_y)), np.max(to_np(pred_y)))
    print('min and max of pred_y2:', np.min(to_np(pred_y2)), np.max(to_np(pred_y2)))

    # Write the prediction
    pred_df = pd.concat((test_bed.to_dataframe()[['chrom', 'start', 'end']], last_pred_df, last_pred_df2['prob']), axis=1)
    pred_df.columns = ['chrom', 'start', 'end','mut_type', 'last_prob', 'last_prob2']

    pred_df.to_csv(pred_file, sep='\t', index=False)
    
    for win_size in [1000, 5000,20000, 50000, 100000]:
        corr1 = corr_calc(pred_df, win_size, 'last_prob')
        corr2 = corr_calc(pred_df, win_size, 'last_prob2')
        print('regional corr:', str(win_size)+'bp', corr1, corr2)
    #os.remove(train_h5f_path)
    #os.remove(test_h5f_path)

    
if __name__ == "__main__":
    main()