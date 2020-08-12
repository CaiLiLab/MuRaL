from janggu.data import Bioseq, Cover
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
from evaluation import f3mer_comp, f5mer_comp, f7mer_comp

print("CUDA: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set train file
train_file = sys.argv[1]

#set test file
test_file = sys.argv[2]

ref_genome='/public/home/licai/DNMML/data/hg19/hg19_ucsc_ordered.fa'

train_bed = BedTool(train_file)

test_bed = BedTool(test_file)


n_cont = 1

bw_files= '/public/home/licai/DNMML/data/germ_cell/Guo_2016_CR/merge_replicates.PGC.RNA-seq.hg19.log2.bw'

radius = 5
distal_radius = 100
#############

distal_order = 2
dataset, data_local, categorical_features = prepare_dataset(train_bed, ref_genome, bw_files, radius, distal_radius, distal_order)

#############
batchsize = 500

dataloader = DataLoader(dataset, batchsize, shuffle=True, num_workers=1)

dataloader2 = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=1)

cat_dims = [int(data_local[col].nunique()) for col in categorical_features]

emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
#emb_dims

######test data #####
dataset_test, data_local_test, _ = prepare_dataset(test_bed, ref_genome, bw_files, radius, distal_radius, distal_order=2)

dataloader1 = DataLoader(dataset_test, batch_size=batchsize, shuffle=False, num_workers=1)

###################
model = Network(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[100, 50], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4**distal_order, out_channels=20, kernel_size=12, RNN_hidden_size=0, RNN_layers=1, last_lin_size=25).to(device)

model2 = FeedForwardNN(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[100, 50], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15]).to(device)

no_of_epochs = 5

# Loss function
criterion = torch.nn.BCELoss()
#criterion = torch.nn.NLLLoss()
#criterion = torch.nn.BCEWithLogitsLoss()

#set Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.05)

for epoch in range(no_of_epochs):
    
    model.train()
    
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
       
    model.eval()

    pred_y = model.batch_predict(dataloader1, device)
    y_prob = pd.Series(data=to_np(pred_y).T[0], name="prob")    
    data_and_prob = pd.concat([data_local_test, y_prob], axis=1)
    
    all_pred_y = model.batch_predict(dataloader2, device)      
    all_y_prob = pd.Series(data=to_np(all_pred_y).T[0], name="prob")
    all_data_and_prob = pd.concat([data_local, all_y_prob], axis=1)

    print ('3mer correlation - test: ' + str(f3mer_comp(data_and_prob)))
    print ('3mer correlation - all: ' + str(f3mer_comp(all_data_and_prob)))
    print ('5mer correlation - test: ' + str(f5mer_comp(data_and_prob)))
    print ('5mer correlation - all: ' + str(f5mer_comp(all_data_and_prob)))
    print ('7mer correlation - test: ' + str(f7mer_comp(data_and_prob)))
    print ('7mer correlation - all: ' + str(f7mer_comp(all_data_and_prob)))
    
    #########################
    pred_y2 = model2.batch_predict(dataloader1, device)
    y_prob2 = pd.Series(data=to_np(pred_y2).T[0], name="prob")    
    data_and_prob2 = pd.concat([data_local_test, y_prob2], axis=1)
    
    all_pred_y2 = model2.batch_predict(dataloader2, device)      
    all_y_prob2 = pd.Series(data=to_np(all_pred_y2).T[0], name="prob")
    all_data_and_prob2 = pd.concat([data_local, all_y_prob2], axis=1)

    print ('3mer correlation - test (FF only): ' + str(f3mer_comp(data_and_prob2)))
    print ('3mer correlation - all (FF only): ' + str(f3mer_comp(all_data_and_prob2)))
    print ('5mer correlation - test (FF only): ' + str(f5mer_comp(data_and_prob2)))
    print ('5mer correlation - all (FF only): ' + str(f5mer_comp(all_data_and_prob2)))
    print ('7mer correlation - test (FF only): ' + str(f7mer_comp(data_and_prob2)))
    print ('7mer correlation - all (FF only): ' + str(f7mer_comp(all_data_and_prob2)))
    #########################
    #get the scores
    #auc_score = metrics.roc_auc_score(to_np(test_y), to_np(pred_y))
    test_y = data_local_test['mut_type']
    auc_score = metrics.roc_auc_score(test_y, to_np(pred_y))
    print("print test_y, pred_y:")
    print(test_y)
    print(to_np(pred_y))
    
    brier_score = metrics.brier_score_loss(data_local_test['mut_type'], to_np(pred_y))
    #test_pred = to_np(torch.cat((test_y,pred_y),1))
    
    #logits = torch.cat((1-pred_y,pred_y),1)
    #logits = torch.log(logits/(1-logits))
    #ECE = to_np(ece_model.forward(logits, test_y.long()))
    
    prob_true, prob_pred = calibration.calibration_curve(test_y, to_np(pred_y),n_bins=50)
    
    #print("calibration: ", np.column_stack((prob_pred,prob_true)))
    
    print ("AUC score: ", auc_score)
    print ("Brier score: ", brier_score)
    #print ("ECE score: ", ECE.item())
    print ("Loss: ", loss.item())
    #np.savetxt(sys.stdout, test_pred, fmt='%s', delimiter='\t')

