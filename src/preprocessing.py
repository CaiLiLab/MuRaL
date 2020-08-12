from janggu.data import Bioseq, Cover, ReduceDim
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

def prepare_dataset(bed_regions, ref_genome,  bw_files, radius=5, distal_radius=50):

    local_seq = Bioseq.create_from_refgenome(name='local', refgenome=ref_genome, roi=bed_regions, flank=radius)

    local_seq_cat = local_seq.iseq4idx(list(range(local_seq.shape[0])))

    #radius = local_seq_cat.shape[1]//2

    categorical_features = ['us'+str(radius - i)for i in range(radius)] + ['mid'] + ['ds'+str(i+1)for i in range(radius)]

    local_seq_cat = pd.DataFrame(local_seq_cat, columns = seq_cat_features)

    #adj_seq.columns = ['us'+str(radius - i)for i in range(radius)] + ['mid'] + ['ds'+str(i+1)for i in range(radius)]

    #
    y = np.array([float(loc.score) for loc in bed_regions], ndmin=2).reshape((-1,1))
    y = pd.DataFrame(y, columns=['mut_type'])
    output_feature = 'mut_type'

    local_RNA = np.array(Cover.create_from_bigwig(name="", bigwigfiles=bw_files, roi=bed_regions, resolution=2*radius+1, flank=radius)).reshape(-1, 1)

    local_RNA = pd.DataFrame(local_RNA, columns=['local_RNA'])

    data_local = pd.concat([local_seq_cat, local_RNA, y], axis=1)

    dataset_local = TabularDataset(data=data_local, cat_cols=categorical_features, output_col=output_feature)

    #######

    distal_seq = Bioseq.create_from_refgenome(name='distal', refgenome=ref_genome, roi=bed_regions, flank=distal_radius)

    distal_seq = np.array(distal_seq).squeeze().transpose(0,2,1)
    dataset_distal = seqDataset([distal_seq, y])

    dataset = CombinedDataset(dataset_local, dataset_distal)
    
    return data_local, dataset



print("CUDA: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set train file
train_file = sys.argv[1]

#set test file
test_file = sys.argv[2]

ref_genome='/public/home/licai/DNMML/data/hg19/hg19_ucsc_ordered.fa'

train_bed = BedTool(train_file)

test_bed = BedTool(test_file)

train_local_seq = Bioseq.create_from_refgenome(name='local', refgenome=ref_genome, roi=train_bed, flank=5)

train_local_seq_cat = train_local_seq.iseq4idx(list(range(train_local_seq.shape[0])))

radius = train_local_seq_cat.shape[1]//2

categorical_features = ['us'+str(radius - i)for i in range(radius)] + ['mid'] + ['ds'+str(i+1)for i in range(radius)]

train_local_seq_cat = pd.DataFrame(train_local_seq_cat, columns = seq_cat_features)

#adj_seq.columns = ['us'+str(radius - i)for i in range(radius)] + ['mid'] + ['ds'+str(i+1)for i in range(radius)]

#
train_y = np.array([float(loc.score) for loc in train_bed], ndmin=2).reshape((-1,1))
train_y = pd.DataFrame(train_y, columns=['mut_type'])
output_feature = 'mut_type'

#embed = nn.Embedding(4,2)
#out = embed(torch.tensor(train_seq_local_cat, dtype=torch.long))
n_cont = 1

bw_files= '/public/home/licai/DNMML/data/germ_cell/Guo_2016_CR/merge_replicates.PGC.RNA-seq.hg19.log2.bw'

train_local_RNA = np.array(Cover.create_from_bigwig(name="", bigwigfiles=bw_files, roi=train_bed, resolution=11, flank=5)).reshape(-1, 1)

train_local_RNA = pd.DataFrame(train_local_RNA, columns=['local_RNA'])

data_local = pd.concat([train_local_seq_cat, train_local_RNA, train_y], axis=1)

dataset_local = TabularDataset(data=data_local, cat_cols=categorical_features, output_col=output_feature)

#######

train_distal_seq = Bioseq.create_from_refgenome(name='distal', refgenome=ref_genome, roi=train_bed, flank=50)

train_distal_seq = np.array(train_distal_seq).squeeze().transpose(0,2,1)
dataset_distal = seqDataset([train_distal_seq, train_y])

dataset = CombinedDataset(dataset_local, dataset_distal)

batchsize = 1000

dataloader = DataLoader(dataset, batchsize, shuffle=True, num_workers=1)

dataloader2 = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=1)

cat_dims = [int(data_local[col].nunique()) for col in categorical_features]

emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
#emb_dims

######test data #####
data_local_test, dataset_test = prepare_dataset(test_bed, ref_genome, bw_files, radius=5, distal_radius=50)

dataloader1 = DataLoader(dataset_test, batch_size=1000, shuffle=False, num_workers=1)

###################
model = Network(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[100, 50], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4, out_channels=50, kernel_size=12, RNN_hidden_size=0, RNN_layers=1, last_lin_size=25).to(device)

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


