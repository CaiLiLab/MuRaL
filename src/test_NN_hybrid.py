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
#from temperature_scaling import ModelWithTemperature, _ECELoss
from evaluation import f3mer_comp, f5mer_comp, f7mer_comp

def to_np(tensor):
    if torch.cuda.is_available():
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()

#device = torch.device("cpu")

#check whether GPU is available
print("CUDA: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set train file
train_file = sys.argv[1]

#set test file
test_file = sys.argv[2]

# set n_cont
n_cont = int(sys.argv[3])

# Using only a subset of the variables.
data = pd.read_csv(train_file).dropna()

data_test = pd.read_csv(test_file).dropna()

output_feature = "mut_type"

data_local, data_distal, categorical_features = separate_local_distal(data, radius=8)
data_local_test, data_distal_test, categorical_features = separate_local_distal(data_test, radius=8)

print('categorical_features:')
print(categorical_features)

dataset_local = TabularDataset(data=data_local, cat_cols=categorical_features, output_col=output_feature)
dataset_local_test = TabularDataset(data=data_local_test, cat_cols=categorical_features, output_col=output_feature)

dataset_distal = gen_ohe_dataset(data_distal)
dataset_distal_test = gen_ohe_dataset(data_distal_test)
print('dataset_distal.shape:', dataset_distal)

dataset = CombinedDataset(dataset_local, dataset_distal)
dataset_test = CombinedDataset(dataset_local_test, dataset_distal_test)

#DataLoader for testing data
dataloader1 = DataLoader(dataset_test, batch_size=1000, shuffle=False, num_workers=1)
#test_y, test_cont_x, test_cat_x, test_distal_x = dataset_test.y, dataset_test.local_cont_X, dataset_test.local_cat_X, dataset_test.distal_X
#test_y, test_cont_x, test_cat_x, test_distal_x = next(iter(dataloader1))


#####

#DataLoader for the train data
batchsize = 1000

dataloader = DataLoader(dataset, batchsize, shuffle=True, num_workers=1)

dataloader2 = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=1)
#all_y, all_cont_x, all_cat_x, all_distal_x = next(iter(dataloader2))
#all_y, all_cont_x, all_cat_x, all_distal_x = dataset.y, dataset.local_cont_X, dataset.local_cat_X, dataset.distal_X
#all_cont_x = all_cont_x.to(device)
#all_cat_x = all_cat_x.to(device)
#all_distal_x = all_distal_x.to(device)
#all_y = all_y.to(device)

cat_dims = [int(data_local[col].nunique()) for col in categorical_features]
#cat_dims
#[15, 5, 2, 4, 112]

emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
#emb_dims
#[(15, 8), (5, 3), (2, 1), (4, 2), (112, 50)]

# define the model 
#model = FeedForwardNN(emb_dims, no_of_cont=15, lin_layer_sizes=[50, 100], output_size=1, emb_dropout=0.04, lin_layer_dropouts=[0.001,0.01]).to(device)
#model = FeedForwardNN(emb_dims, no_of_cont=15, lin_layer_sizes=[50, 200], output_size=1, emb_dropout=0.001, lin_layer_dropouts=[0.001,0.001]).to(device) #bs=8000
#model = FeedForwardNN(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[200, 100], output_size=1, emb_dropout=0.3, lin_layer_dropouts=[0.1,0.1]).to(device)

model = Network(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[100, 50], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4, out_channels=50, kernel_size=12, RNN_hidden_size=0, RNN_layers=1, last_lin_size=25).to(device)

model2 = FeedForwardNN(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[100, 50], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15]).to(device)

no_of_epochs = 20

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

class seqDataset(Dataset):
    """ Diabetes dataset."""

    def __init__(self,xy=None):
        #self.x_data = np.asarray([el for el in xy[0]],dtype=np.float32)
        self.x_data = np.asarray(xy[0], dtype=np.float32)
        #self.y_data = np.asarray([el for el in xy[1]],dtype=np.float32)
        self.y_data = np.asarray(xy[1], dtype=np.float32)
        
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
        
        self.len=len(self.x_data)


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len