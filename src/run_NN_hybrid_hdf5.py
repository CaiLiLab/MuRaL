from pybedtools import BedTool

import sys
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

import os

from sklearn import metrics, calibration

from NN_utils import *
from preprocessing import *
from evaluation import *

print("CUDA: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(' '.join(sys.argv))

# Set train file
train_file = sys.argv[1]

# Set test file
test_file = sys.argv[2]

ref_genome='/public/home/licai/DNMML/data/hg19/hg19_ucsc_ordered.fa'

# Read BED files
train_bed = BedTool(train_file)

test_bed = BedTool(test_file)

# Read bigWig file names
bw_files = []
bw_names = []
n_cont = 0
try:
    bw_list = pd.read_table('/public/home/licai/DNMML/analysis/test/bw_files.txt', sep='\s+', header=None, comment='#')

    bw_files = list(bw_list[0])
    bw_names = list(bw_list[1])
    n_cont = len(bw_names)
except pd.errors.EmptyDataError:
    print('Warnings: no bigWig files provided')
except FileNotFoundError:
    print('Error: bw_list file does not exist')
    
# The width to be considered for local signals
if len(sys.argv)>3:
    radius = int(sys.argv[3])
else:
    radius = 5
print('radius:', radius)

# The width to be considered for more distal signals
if len(sys.argv)>4:
    distal_radius = int(sys.argv[4])
else:
    distal_radius = 200
print('distal_radius:', distal_radius)

# The order of sequence when converting sequence to digital data
if len(sys.argv)>5:
    distal_order = int(sys.argv[5])
else:
    distal_order = 1
print('distal_order:', distal_order)

train_h5f_path = sys.argv[11] + '.train_distal.h5'
test_h5f_path = sys.argv[11] + '.test_distal.h5'

# Prepare the datasets for trainging
dataset, data_local, categorical_features = prepare_dataset2(train_bed, ref_genome, bw_files,bw_names, radius, distal_radius, distal_order, train_h5f_path)

#sys.exit()

# Batch size for training
if len(sys.argv)>6:
    batchsize = int(sys.argv[6])
else:
    batchsize = 200
print('batchsize:', batchsize)

# CNN kernel size
if len(sys.argv)>7:
    cnn_kernel_size = int(sys.argv[7])
else:
    cnn_kernel_size = 12
print('cnn_kernel_size:', cnn_kernel_size)

# CNN output channels
if len(sys.argv)>8:
    cnn_out_channels = int(sys.argv[8])
else:
    cnn_out_channels = 50
    
print('cnn_out_channels:', cnn_out_channels)

# RNN hidden neurons
if len(sys.argv)>9:
    RNN_hidden_size = int(sys.argv[9])
else:
    RNN_hidden_size = 0
    
print('RNN_hidden_size:', RNN_hidden_size)

# Dataloader for training
dataloader = DataLoader(dataset, batchsize, shuffle=True, num_workers=1) #shuffle=False for HybridLoss

# Dataloader for predicting
dataloader2 = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=1)

# Number of categorical features
cat_dims = [int(data_local[col].nunique()) for col in categorical_features]

#Embedding dimensions for categorical features
emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
#emb_dims

# Prepare testing data 
dataset_test, data_local_test, _ = prepare_dataset2(test_bed, ref_genome, bw_files, bw_names, radius, distal_radius, distal_order, test_h5f_path)

# Dataloader for testing data
dataloader1 = DataLoader(dataset_test, batch_size=batchsize, shuffle=False, num_workers=1)

# Choose the network model
if len(sys.argv) < 10:
    model = Network(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order+n_cont, out_channels=cnn_out_channels, kernel_size=cnn_kernel_size, RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=distal_radius, distal_order=distal_order).to(device)
    
elif int(sys.argv[10]) == 1:
    model = Network2(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order+n_cont, out_channels=cnn_out_channels, kernel_size=cnn_kernel_size, RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=distal_radius, distal_order=distal_order).to(device)
    
elif int(sys.argv[10]) == 2:
    model = Network3(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order+n_cont, out_channels=cnn_out_channels, kernel_size=cnn_kernel_size, RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=distal_radius, distal_order=distal_order).to(device)

elif int(sys.argv[10]) == 3:
    model = Network4(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[150, 80], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order+n_cont, out_channels=cnn_out_channels, kernel_size=cnn_kernel_size, RNN_hidden_size=RNN_hidden_size, RNN_layers=1, last_lin_size=35, distal_radius=distal_radius, distal_order=distal_order).to(device)
    
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
weights_init(model)
weights_init(model2)

# Loss function
criterion = torch.nn.BCELoss()
#criterion = HybridLoss(10)

# Output file name for saving predictions 
if len(sys.argv)>11:
    pred_outfile = sys.argv[11]
else:
    pred_outfile = test_file + '.csv'
    
if len(sys.argv)>12:
    learning_rate = float(sys.argv[12])
else:
    learning_rate = 0.005

if len(sys.argv)>13:
    weight_decay = float(sys.argv[13])
else:
    weight_decay = 1e-5

if len(sys.argv)>14:
    gamma = float(sys.argv[14])
else:
    gamma = 0.5

if len(sys.argv)>15:
    no_of_epochs = int(sys.argv[15])
else:
    no_of_epochs = 15
    
# Set Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=gamma)
print('optimizer, optimizer2:', optimizer, optimizer2)
#print('scheduler, scheduler2:', scheduler, scheduler2)


best_loss = 0
pred_df = None
last_pred_df = None

best_loss2 = 0
pred_df2 = None
last_pred_df2 = None


# Training
for epoch in range(no_of_epochs):
    
    model.train()
    model2.train()
    
    total_loss = 0
    total_loss2 = 0
    
    torch.cuda.empty_cache() 
    
    for y, cont_x, cat_x, distal_x in dataloader:
        cat_x = cat_x.to(device)
        cont_x = cont_x.to(device)
        distal_x = distal_x.to(device)
        y  = y.to(device)
        
        # Forward Pass
        #preds = model(cont_x, cat_x) #original
        preds = model.forward((cont_x, cat_x), distal_x)
        #print("preds:")
        #print(preds.shape)    
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
        
        #if len(sys.argv)>7 and int(sys.argv[7]) > 0:
        #    print('torch.sigmoid(model.w_ld):', torch.sigmoid(model.w_ld))
    
        # Do predictions for testing data
        pred_y, test_total_loss = model.batch_predict(dataloader1, criterion, device)
        y_prob = pd.Series(data=to_np(pred_y).T[0], name="prob")    
        data_and_prob = pd.concat([data_local_test, y_prob], axis=1)
    
        # Do predictions for training data
        all_pred_y, train_total_loss = model.batch_predict(dataloader2, criterion, device)      
        all_y_prob = pd.Series(data=to_np(all_pred_y).T[0], name="prob")
        all_data_and_prob = pd.concat([data_local, all_y_prob], axis=1)
    
        # Compare observed/predicted 3/5/7mer mutation frequencies
        print ('3mer correlation - test: ' + str(f3mer_comp(data_and_prob)))
        print ('3mer correlation - all: ' + str(f3mer_comp(all_data_and_prob)))
        print ('5mer correlation - test: ' + str(f5mer_comp(data_and_prob)))
        print ('5mer correlation - all: ' + str(f5mer_comp(all_data_and_prob)))
        print ('7mer correlation - test: ' + str(f7mer_comp(data_and_prob)))
        print ('7mer correlation - all: ' + str(f7mer_comp(all_data_and_prob)))
    
        # For FeedForward-only model
        pred_y2, test_total_loss2 = model2.batch_predict(dataloader1, criterion, device)
        y_prob2 = pd.Series(data=to_np(pred_y2).T[0], name="prob")    
        data_and_prob2 = pd.concat([data_local_test, y_prob2], axis=1)
    
        all_pred_y2, train_total_loss2 = model2.batch_predict(dataloader2, criterion, device)      
        all_y_prob2 = pd.Series(data=to_np(all_pred_y2).T[0], name="prob")
        all_data_and_prob2 = pd.concat([data_local, all_y_prob2], axis=1)

        print ('3mer correlation - test (FF only): ' + str(f3mer_comp(data_and_prob2)))
        print ('3mer correlation - all (FF only): ' + str(f3mer_comp(all_data_and_prob2)))
        print ('5mer correlation - test (FF only): ' + str(f5mer_comp(data_and_prob2)))
        print ('5mer correlation - all (FF only): ' + str(f5mer_comp(all_data_and_prob2)))
        print ('7mer correlation - test (FF only): ' + str(f7mer_comp(data_and_prob2)))
        print ('7mer correlation - all (FF only): ' + str(f7mer_comp(all_data_and_prob2)))
    
        # Save the predictions of the best model
        if epoch == 0:
            best_loss = test_total_loss
            best_loss2 = test_total_loss2
            pred_df = data_and_prob[['mut_type','prob']]
            pred_df2 = data_and_prob2[['mut_type','prob']]
    
        if test_total_loss < best_loss:
            best_loss = test_total_loss
            pred_df = data_and_prob[['mut_type','prob']]
    
        if test_total_loss2 < best_loss2:
            best_loss2 = test_total_loss2
            pred_df2 = data_and_prob2[['mut_type','prob']]
    
        if epoch == no_of_epochs-1:
            last_pred_df = data_and_prob[['mut_type','prob']]
            last_pred_df2 = data_and_prob2[['mut_type','prob']]
            
            torch.save(model.state_dict(), pred_outfile+'.model1')
            torch.save(model2.state_dict(), pred_outfile+'.model2')

        # Get the scores
        #auc_score = metrics.roc_auc_score(to_np(test_y), to_np(pred_y))
        test_y = data_local_test['mut_type']
        auc_score = metrics.roc_auc_score(test_y, to_np(pred_y))
        auc_score2 = metrics.roc_auc_score(test_y, to_np(pred_y2))
    
        # Print some data for debugging
        print("print test_y, pred_y:")
        print(test_y)
        print(to_np(pred_y))
        print('min and max of pred_y:', np.min(to_np(pred_y)), np.max(to_np(pred_y)))
        print('min and max of pred_y2:', np.min(to_np(pred_y2)), np.max(to_np(pred_y2)))
   
        brier_score = metrics.brier_score_loss(data_local_test['mut_type'], to_np(pred_y))
        brier_score2 = metrics.brier_score_loss(data_local_test['mut_type'], to_np(pred_y2))

    
        prob_true, prob_pred = calibration.calibration_curve(test_y, to_np(pred_y),n_bins=50)
    
        #print("calibration: ", np.column_stack((prob_pred,prob_true)))
    
        print ("AUC score: ", auc_score, auc_score2)
        print ("Brier score: ", brier_score, brier_score2)
    
        print ("Total Loss: ", train_total_loss, train_total_loss2, test_total_loss, test_total_loss2)
        #np.savetxt(sys.stdout, test_pred, fmt='%s', delimiter='\t')

# Write the prediction
print('best loss, best loss2:', best_loss, best_loss2)

pred_df = pd.concat((test_bed.to_dataframe()[['chrom', 'start', 'end']], pred_df, pred_df2['prob'], last_pred_df['prob'], last_pred_df2['prob']), axis=1)
pred_df.columns = ['chrom', 'start', 'end','mut_type','prob1', 'prob2', 'last_prob', 'last_prob2']
pred_df.to_csv(pred_outfile, sep='\t', index=False)

os.remove(train_h5f_path)
os.remove(test_h5f_path)

