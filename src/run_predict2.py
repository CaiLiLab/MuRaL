from pybedtools import BedTool

import sys
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from sklearn import metrics, calibration

from NN_utils import *
from preprocessing import *
from evaluation import *

from torchsummary import summary

print("CUDA: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(' '.join(sys.argv))

# Set input file
train_file = sys.argv[1]
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
radius = int(sys.argv[3])

# The width to be considered for more distal signals
distal_radius = int(sys.argv[4])
distal_order = int(sys.argv[5])

# The order of sequence when converting sequence to digital data
# Batch size
if len(sys.argv)>6:
    batchsize = int(sys.argv[6])
else:
    batchsize = 200
print('batchsize:', batchsize)

# Prepare the datasets for input
dataset_train, data_local_train, categorical_features = prepare_dataset(train_bed, ref_genome, bw_files,bw_names, radius, distal_radius, distal_order)

dataset, data_local, categorical_features = prepare_dataset(test_bed, ref_genome, bw_files,bw_names, radius, distal_radius, distal_order)

# Dataloader for input data
dataloader_train = DataLoader(dataset_train, batchsize, shuffle=False, num_workers=1) #shuffle=False for HybridLoss
dataloader = DataLoader(dataset, batchsize, shuffle=False, num_workers=1) 

cat_dims = [int(data_local[col].nunique()) for col in categorical_features]

#Embedding dimensions for categorical features
emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]




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

model1_path = sys.argv[11]

model2_path = sys.argv[12]

pred_outfile = sys.argv[13]

model.load_state_dict(torch.load(model1_path))
model2.load_state_dict(torch.load(model2_path))

#y, cont_x, cat_x, distal_x = next(iter(dataloader))
#print('Summary of models:')
#summary(model, (cont_x.shape, cat_x), distal_x)
#summary(model2, cont_x.shape, cat_x.shape)

# Loss function
criterion = torch.nn.BCELoss()
#criterion = HybridLoss(10)

total_loss = 0
pred_df = None

total_loss2 = 0
pred_df2 = None

   
# Do predictions for the data
pred_y, total_loss = model.batch_predict(dataloader, criterion, device)
y_prob = pd.Series(data=to_np(pred_y).T[0], name="prob")    
data_and_prob = pd.concat([data_local, y_prob], axis=1)

    
# Compare observed/predicted 3/5/7mer mutation frequencies
print ('3mer correlation - test: ' + str(f3mer_comp(data_and_prob)))
print ('5mer correlation - test: ' + str(f5mer_comp(data_and_prob)))
print ('7mer correlation - test: ' + str(f7mer_comp(data_and_prob)))
    
# For FeedForward-only model
pred_y2, total_loss2 = model2.batch_predict(dataloader, criterion, device)
y_prob2 = pd.Series(data=to_np(pred_y2).T[0], name="prob")    
data_and_prob2 = pd.concat([data_local, y_prob2], axis=1)
    
print ('3mer correlation - test (FF only): ' + str(f3mer_comp(data_and_prob2)))
print ('5mer correlation - test (FF only): ' + str(f5mer_comp(data_and_prob2)))
print ('7mer correlation - test (FF only): ' + str(f7mer_comp(data_and_prob2)))

##############################
scaled_model = ModelWithTemperature(model)
scaled_model.set_temperature(dataloader_train)
pred_y3, total_loss3 = scaled_model.batch_predict(dataloader, criterion, device)
y_prob3 = pd.Series(data=to_np(pred_y3).T[0], name="prob")    
data_and_prob3 = pd.concat([data_local, y_prob3], axis=1)

print ('3mer correlation - test (scaled): ' + str(f3mer_comp(data_and_prob3)))
print ('5mer correlation - test (scaled): ' + str(f5mer_comp(data_and_prob3)))
print ('7mer correlation - test (scaled): ' + str(f7mer_comp(data_and_prob3)))
###############################

pred_df = data_and_prob[['mut_type','prob']]
pred_df2 = data_and_prob2[['mut_type','prob']]

pred_df3 = data_and_prob3[['mut_type','prob']]
    
# Get the scores
#auc_score = metrics.roc_auc_score(to_np(true_y), to_np(pred_y))
true_y = data_local['mut_type']
auc_score = metrics.roc_auc_score(true_y, to_np(pred_y))
auc_score2 = metrics.roc_auc_score(true_y, to_np(pred_y2))
    
# Print some data for debugging
print("print true_y, pred_y:")
print(true_y)
print(to_np(pred_y))
print('min and max of pred_y:', np.min(to_np(pred_y)), np.max(to_np(pred_y)))
print('min and max of pred_y2:', np.min(to_np(pred_y2)), np.max(to_np(pred_y2)))
   
brier_score = metrics.brier_score_loss(data_local['mut_type'], to_np(pred_y))
brier_score2 = metrics.brier_score_loss(data_local['mut_type'], to_np(pred_y2))

    
#prob_true, prob_pred = calibration.calibration_curve(true_y, to_np(pred_y),n_bins=50)
    
#print("calibration: ", np.column_stack((prob_pred,prob_true)))
    
print ("AUC score: ", auc_score, auc_score2)
print ("Brier score: ", brier_score, brier_score2)
    
print ("Total Loss: ",  total_loss, total_loss2)
#np.savetxt(sys.stdout, test_pred, fmt='%s', delimiter='\t')

# Write the prediction
#pred_df = pd.concat((test_bed.to_dataframe()[['chrom', 'start', 'end']], pred_df, pred_df2['prob']), axis=1)
#pred_df.columns = ['chrom', 'start', 'end', 'mut_type','prob1', 'prob2']
pred_df = pd.concat((test_bed.to_dataframe()[['chrom', 'start', 'end']], pred_df, pred_df2['prob'], pred_df3['prob']), axis=1)
pred_df.columns = ['chrom', 'start', 'end', 'mut_type','prob1', 'prob2', 'prob3']

pred_df.to_csv(pred_outfile, sep='\t', index=False)

