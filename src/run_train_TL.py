import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from pybedtools import BedTool

import sys
import argparse

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

from nn_models import *
from nn_utils import *
from preprocessing import *
from evaluation import *
from pynvml import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True

def parse_arguments(parser):
    """
    Parse parameters from the command line
    """
    parser.add_argument('--train_data', type=str, default='',
                        help='path for training data')
    
    parser.add_argument('--test_data', type=str, default='',
                        help='path for testing data')
    
    parser.add_argument('--train_data_h5f', type=str, default='', help='path for training data in HDF5 format')
    
    parser.add_argument('--test_data_h5f', type=str, default='', help='path for testing data in HDF5 format')
    
    parser.add_argument('--ref_genome', type=str, default='/public/home/licai/DNMML/data/hg19/hg19_ucsc_ordered.fa',
                        help='reference genome')

    parser.add_argument('--bw_paths', type=str, default='', help='path for the list of BigWig files for non-sequence features')
    
    parser.add_argument('--n_class', type=int, default=2, help='number of mutation classes')
    
    parser.add_argument('--local_radius', type=int, default=5, help='radius of local sequences to be considered')
    parser.add_argument('--local_order', type=int, default=1, help='order of local sequences to be considered')
        
    parser.add_argument('--distal_radius', type=int, default=50, help='radius of distal sequences to be considered')
    
    parser.add_argument('--distal_order', type=int, default=1, help='order of distal sequences to be considered')
        
    parser.add_argument('--emb_dropout', type=float, default=0.2, help='dropout rate for k-mer embedding')
    
    parser.add_argument('--local_dropout', type=float, default=0.15, help='dropout rate for local network')
    
    parser.add_argument('--batch_size', type=int, default=128, help='size of mini batches')
    
    parser.add_argument('--pred_batch_size', type=int, default=16, help='size of mini batches for test data')
    
    parser.add_argument('--CNN_kernel_size', type=int, default=3, help='kernel size for CNN layers')
    
    parser.add_argument('--CNN_out_channels', type=int, default=32, help='number of output channels for CNN layers')
    
    parser.add_argument('--distal_fc_dropout', type=float, default=0.25, help='dropout rate for distal fc layer')
    
    parser.add_argument('--model_no', type=int, default=2, help='which NN model to be used')
    
    parser.add_argument('--pred_file', type=str, default='pred.tsv', help='Output file for saving predictions')
    
    parser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate for training')
    
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay (L2 regularization) for training')
    
    parser.add_argument('--LR_gamma', type=float, default=0.5, help='gamma for learning rate change during training')
    
    parser.add_argument('--epochs', type=int, default=10, help='numbe of epochs for training')
        
    parser.add_argument('--train_all', default=False, action='store_true')
    
    parser.add_argument('--init_fc_with_pretrained', default=False, action='store_true')

    parser.add_argument('--model_path', type=str, default='', help='model path')
    
    parser.add_argument('--calibrator_path', type=str, default='', help='calibrator path')
    
    args = parser.parse_args()

    return args
def main():
    parser = argparse.ArgumentParser(description='Mutation rate modeling using machine learning')
    args = parse_arguments(parser)

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
    emb_dropout = args.emb_dropout
    local_dropout = args.local_dropout
    batch_size = args.batch_size
    pred_batch_size = args.pred_batch_size
    CNN_kernel_size = args.CNN_kernel_size   
    CNN_out_channels = args.CNN_out_channels
    distal_fc_dropout = args.distal_fc_dropout
    model_no = args.model_no   
    pred_file = args.pred_file   
    
    learning_rate = args.learning_rate   
    weight_decay = args.weight_decay  
    LR_gamma = args.LR_gamma  
    epochs = args.epochs
    
    train_all = args.train_all
    init_fc_with_pretrained = args.init_fc_with_pretrained
    
    model_path = args.model_path
    calibrator_path = args.calibrator_path
    
    n_class = args.n_class
    
    start_time = time.time()
    print('Start time:', datetime.datetime.now())
    
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
        train_h5f_path = train_file + '.distal_' + str(distal_radius)
        if(distal_order >1):
            train_h5f_path = train_h5f_path + '_' + str(distal_order)
        if len(bw_names) > 0:
            train_h5f_path = train_h5f_path + '.' + '.'.join(list(bw_names))
        train_h5f_path = train_h5f_path + '.h5'
    
    if len(test_h5f_path) == 0:
        test_h5f_path = test_file + '.distal_' + str(distal_radius)
        if(distal_order >1):
            test_h5f_path = test_h5f_path + '_' + str(distal_order)
        if len(bw_names) > 0:
            test_h5f_path = test_h5f_path + '.' + '.'.join(list(bw_names))
        test_h5f_path = test_h5f_path + '.h5'   
    
    # Prepare the datasets for trainging
    dataset = prepare_dataset(train_bed, ref_genome, bw_files, bw_names, local_radius, local_order, distal_radius, distal_order, train_h5f_path)
    data_local = dataset.data_local
    
    train_size = len(dataset)

    # Dataloader for training
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)

    # Number of categorical features
    cat_dims = dataset.cat_dims

    #Embedding dimensions for categorical features
    emb_dims = [(x, min(16, int(x**0.25))) for x in cat_dims]  

    # Prepare testing data 
    dataset_test = prepare_dataset(test_bed, ref_genome, bw_files, bw_names, local_radius, local_order, distal_radius, distal_order, test_h5f_path, 1)
    data_local_test = dataset_test.data_local
    
    test_size = len(dataset_test)

    # Dataloader for testing data
    dataloader1 = DataLoader(dataset_test, batch_size=pred_batch_size, shuffle=False, num_workers=2)

    # Find a device with enough memory
    nvmlInit()
    cuda_id = '0'
    for i in range(nvmlDeviceGetCount()):
        h = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(h)
        if info.free > 1.5*(2**30): #reserve 1.5GB
            cuda_id = str(i)
            break
        
    print("CUDA: ", torch.cuda.is_available())
    print("using  ", "cuda:"+cuda_id)
    device = torch.device("cuda:"+cuda_id if torch.cuda.is_available() else "cpu")
    
    # Choose the network model
    if model_no == 0:
        model = Network0(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], n_class=n_class, emb_padding_idx=4**local_order).to(device)

    elif model_no == 1:
        model = Network1(in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=CNN_kernel_size, distal_radius=distal_radius, distal_order=distal_order, distal_fc_dropout=distal_fc_dropout, n_class=n_class, emb_padding_idx=4**local_order).to(device)

    elif model_no == 2:
        model = Network2(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=emb_dropout, lin_layer_dropouts=[local_dropout, local_dropout], in_channels=4**distal_order+n_cont, out_channels=CNN_out_channels, kernel_size=CNN_kernel_size, distal_radius=distal_radius, distal_order=distal_order, distal_fc_dropout=distal_fc_dropout, n_class=n_class, emb_padding_idx=4**local_order).to(device)

    else:
        print('Error: no model selected!')
        sys.exit() 
    
    count_parameters(model)
    print('model:')
    print(model)


    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)
    #model2.load_state_dict(torch.load(model2_path, map_location=device))

    #criterion = torch.nn.NLLLoss()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    if train_all:
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
    
    if not init_fc_with_pretrained:
        # Re-initialize fc layers
        model.local_fc[-1].apply(weights_init)
        model.distal_fc[-1].apply(weights_init)
    
    # Set Optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LR_gamma)

    print('optimizer:', optimizer)

    prob_names = ['prob'+str(i) for i in range(n_class)]
    
    # Training
    for epoch in range(epochs):
        model.train()
        total_loss = 0

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
            total_loss += loss.item()
        
        print ("Train Loss: ", total_loss/train_size)    
        print('optimizer learning rate:', optimizer.param_groups[0]['lr'])
        scheduler.step()

        sys.stdout.flush()
        
        model.eval()
        with torch.no_grad():   
            # Do predictions on testing data
            pred_y, test_total_loss = model_predict_m(model, dataloader1, criterion, device, n_class, distal=True)
            
            # Print some data for debugging
            print('pred_y:', F.softmax(pred_y[1:10], dim=1))
            for i in range(1, n_class):
                print('min and max of pred_y: type', i, np.min(to_np(F.softmax(pred_y, dim=1))[:,i]), np.max(to_np(F.softmax(pred_y, dim=1))[:,i]))
                
            # Get the predicted probabilities, as the returns of model are logits
            y_prob = pd.DataFrame(data=to_np(F.softmax(pred_y, dim=1)), columns=prob_names)
            data_and_prob = pd.concat([data_local_test, y_prob], axis=1) 
            y = data_and_prob['mut_type'].to_numpy().squeeze()
            
            # Fit a calibrator
            fdiri_cal, fdiri_nll = calibrate_prob(y_prob.to_numpy(), y, device, calibr_name='FullDiri')
            
            # Calculate k-mer correlations
            print('3mer correlation: ', freq_kmer_comp_multi(data_and_prob, 3, n_class))
            print('5mer correlation: ', freq_kmer_comp_multi(data_and_prob, 5, n_class))
            print('7mer correlation: ', freq_kmer_comp_multi(data_and_prob, 7, n_class))
            
            # Write the prediction
            test_pred_df = data_and_prob[['mut_type'] + prob_names]
            pred_df = pd.concat((test_bed.to_dataframe()[['chrom', 'start', 'end', 'strand']], test_pred_df), axis=1)
            pred_df.columns = ['chrom', 'start', 'end', 'strand', 'mut_type'] +  prob_names

            # Save the model and calibrator
            if epoch == epochs -1:
                pred_df.to_csv(pred_file, sep='\t', index=False)
                torch.save(model.state_dict(), pred_file+'.model')
                
                with open(pred_file + '.fdiri_cal.pkl', 'wb') as pkl_file:
                    pickle.dump(fdiri_cal, pkl_file)
            
            # Calculate regional correlations for a few window sizes
            for win_size in [10000, 50000, 200000]:
                corr1 = corr_calc_sub(pred_df, win_size, prob_names)
                print('regional corr:', str(win_size)+'bp', corr1)

            print ("Test Loss: ", test_total_loss/test_size)
            print ('Test Loss (after fdiri_cal): ', fdiri_nll) 

    
    print('Total time used: %s seconds' % (time.time() - start_time))
   
    
if __name__ == "__main__":
    main()


