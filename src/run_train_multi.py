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

import os
import time
import datetime


from sklearn import metrics, calibration

from NN_utils import *
from preprocessing import *
from evaluation import *


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
    
    n_class = 2
    
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
    dataset = prepare_dataset2(train_bed, ref_genome, bw_files, bw_names, local_radius, distal_radius, distal_order, train_h5f_path)
    data_local = dataset.data_local
    categorical_features = dataset.cat_cols
    
    train_size = len(dataset)

    # Dataloader for training
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2) #shuffle=False for HybridLoss

    # Dataloader for predicting
    dataloader2 = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Number of categorical features
    cat_dims = [int(data_local[col].nunique()) for col in categorical_features]

    #Embedding dimensions for categorical features
    emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
    #emb_dims

    # Prepare testing data 
    dataset_test = prepare_dataset2(test_bed, ref_genome, bw_files, bw_names, local_radius, distal_radius, distal_order, test_h5f_path, 1)
    data_local_test = dataset_test.data_local
    
    test_size = len(dataset_test)

    # Dataloader for testing data
    dataloader1 = DataLoader(dataset_test, batch_size=10, shuffle=False, num_workers=2)

    # Choose the network model
    if model_no == 0:
        model = Network(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=cnn_kernel_size, RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=distal_radius, distal_order=distal_order).to(device)

    elif model_no == 1:
        model = Network2(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=CNN_kernel_size, RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=distal_radius, distal_order=distal_order).to(device)

    elif model_no == 2:
        model = Network3m(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=CNN_kernel_size, RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=distal_radius, distal_order=distal_order, n_class=n_class).to(device)

    elif model_no == 3:
        model = Network4(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=CNN_kernel_size, RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=distal_radius, distal_order=distal_order).to(device)

    else:
        print('Error: no model selected!')
        sys.exit() 

    print('model:')
    print(model)

    # FeedForward-only model for comparison
    #model2 = FeedForwardNN(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15]).to(device)
    model2 = FeedForwardNNm(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], n_class=n_class).to(device)
    
    print('model2:')
    print(model2)

    # Initiating weights of the models;
    
    model.apply(weights_init)
    model2.apply(weights_init)

    # Loss function
    #criterion = torch.nn.BCELoss()
    criterion = torch.nn.NLLLoss()

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
        dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)

        for y, cont_x, cat_x, distal_x in dataloader:
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            distal_x = distal_x.to(device)
            y  = y.to(device)

            # Forward Pass
            #preds = model(cont_x, cat_x) #original
            preds = model.forward((cont_x, cat_x), distal_x)
            
            loss = criterion(preds, y.long().squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds2 = model2.forward(cont_x, cat_x)
            loss2 = criterion(preds2, y.long().squeeze())
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

            if epoch <5:
                continue

            # Do predictions for training data
            #all_pred_y, train_total_loss = model.batch_predict(dataloader2, criterion, device)      
            all_pred_y, train_total_loss, all_pred_y2, train_total_loss2 = two_model_predict_m(model, model2, dataloader2, criterion, device, n_class)

            all_y_prob = pd.Series(data=to_np(torch.exp(all_pred_y)).T[1], name="prob")
            all_data_and_prob = pd.concat([data_local, all_y_prob], axis=1)

            #all_pred_y2, train_total_loss2 = model2.batch_predict(dataloader2, criterion, device)     
            all_y_prob2 = pd.Series(data=to_np(torch.exp(all_pred_y2)).T[1], name="prob")
            all_data_and_prob2 = pd.concat([data_local, all_y_prob2], axis=1)        

            # Compare observed/predicted 3/5/7mer mutation frequencies
            print ('3mer correlation - all: ' + str(f3mer_comp(all_data_and_prob)))
            print ('5mer correlation - all: ' + str(f5mer_comp(all_data_and_prob)))
            print ('7mer correlation - all: ' + str(f7mer_comp(all_data_and_prob)))


            print ('3mer correlation - all (FF only): ' + str(f3mer_comp(all_data_and_prob2)))
            print ('5mer correlation - all (FF only): ' + str(f5mer_comp(all_data_and_prob2)))
            print ('7mer correlation - all (FF only): ' + str(f7mer_comp(all_data_and_prob2)))

            print ("Total Loss: ", train_total_loss/train_size, train_total_loss2/train_size)        
            # Do predictions for testing data
            if epoch == epochs-1:
                #pred_y, test_total_loss = model.batch_predict(dataloader1, criterion, device)
                pred_y, test_total_loss, pred_y2, test_total_loss2 = two_model_predict_m(model, model2, dataloader1, criterion, device, n_class)
                y_prob = pd.Series(data=to_np(torch.exp(pred_y)).T[1], name="prob")    
                data_and_prob = pd.concat([data_local_test, y_prob], axis=1)        

                # For FeedForward-only model
                #pred_y2, test_total_loss2 = model2.batch_predict(dataloader1, criterion, device)
                y_prob2 = pd.Series(data=to_np(torch.exp(pred_y2)).T[1], name="prob")    
                data_and_prob2 = pd.concat([data_local_test, y_prob2], axis=1)

                print ('3mer correlation - test: ' + str(f3mer_comp(data_and_prob)))
                print ('5mer correlation - test: ' + str(f5mer_comp(data_and_prob)))
                print ('7mer correlation - test: ' + str(f7mer_comp(data_and_prob)))
                print ('3mer correlation - test (FF only): ' + str(f3mer_comp(data_and_prob2)))
                print ('5mer correlation - test (FF only): ' + str(f5mer_comp(data_and_prob2)))
                print ('7mer correlation - test (FF only): ' + str(f7mer_comp(data_and_prob2)))


            if epoch == epochs-1:
                last_pred_df = data_and_prob[['mut_type','prob']]
                last_pred_df2 = data_and_prob2[['mut_type','prob']]

                torch.save(model.state_dict(), pred_file+'.model1')
                torch.save(model2.state_dict(), pred_file+'.model2')

                # Get the scores
                #auc_score = metrics.roc_auc_score(to_np(test_y), to_np(pred_y))
                test_y = data_local_test['mut_type']

                # Print some data for debugging
                print('min and max of pred_y:', np.min(to_np(torch.exp(pred_y))[:,1]), np.max(to_np(torch.exp(pred_y))[:,1]))
                print('min and max of pred_y2:', np.min(to_np(torch.exp(pred_y2))[:,1]), np.max(to_np(torch.exp(pred_y2))[:,1]))

    # Write the prediction
    pred_df = pd.concat((test_bed.to_dataframe()[['chrom', 'start', 'end']], last_pred_df, last_pred_df2['prob']), axis=1)
    pred_df.columns = ['chrom', 'start', 'end','mut_type', 'last_prob', 'last_prob2']

    pred_df.to_csv(pred_file, sep='\t', index=False)
    
    for win_size in [1000, 5000, 50000]:
        corr1 = corr_calc(pred_df, win_size, 'last_prob')
        corr2 = corr_calc(pred_df, win_size, 'last_prob2')
        print('regional corr:', str(win_size)+'bp', corr1, corr2)
    #os.remove(train_h5f_path)
    #os.remove(test_h5f_path)

    print('Total time used: %s seconds' % (time.time() - start_time))
   
    
if __name__ == "__main__":
    main()


