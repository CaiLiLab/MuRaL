import sys

from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import pandas as pd
import numpy as np

from sklearn import metrics

from pytorch_tabular import TabularDataset, FeedForwardNN

#device = torch.device("cpu")

print("CUDA: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_file = sys.argv[1]

test_file = sys.argv[2]

# Using only a subset of the variables.
data = pd.read_csv(train_file).dropna()

data_test = pd.read_csv(test_file).dropna()

#categorical_features = ["MSSubClass", "MSZoning", "Street", "LotShape", "YearBuilt"]
categorical_features = ["us5", "us4", "us3", "us2", "us1", "ds1", "ds2", "ds3", "ds4", "ds5", "tdist"]

output_feature = "mut_type"

from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for cat_col in categorical_features:
	label_encoders[cat_col] = LabelEncoder()
	data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])

dataset = TabularDataset(data=data, cat_cols=categorical_features,
									 output_col=output_feature)
#####
label_encoders = {}
for cat_col in categorical_features:
	label_encoders[cat_col] = LabelEncoder()
	data_test[cat_col] = label_encoders[cat_col].fit_transform(data_test[cat_col])

dataset_test = TabularDataset(data=data_test, cat_cols=categorical_features,
									 output_col=output_feature)

dataloader1 = DataLoader(dataset_test, batch_size=999, shuffle=False, num_workers=1)

test_y, test_cont_x, test_cat_x = next(iter(dataloader1))

test_cont_x = test_cont_x.to(device)
test_cat_x = test_cat_x.to(device)
test_y = test_y.to(device)

#####


batchsize = 10000
dataloader = DataLoader(dataset, batchsize, shuffle=True, num_workers=1)

cat_dims = [int(data[col].nunique()) for col in categorical_features]
#cat_dims
#[15, 5, 2, 4, 112]

emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
#emb_dims
#[(15, 8), (5, 3), (2, 1), (4, 2), (112, 50)]

#model = FeedForwardNN(emb_dims, no_of_cont=15, lin_layer_sizes=[50, 100], output_size=1, emb_dropout=0.04, lin_layer_dropouts=[0.001,0.01]).to(device)
#model = FeedForwardNN(emb_dims, no_of_cont=15, lin_layer_sizes=[50, 200], output_size=1, emb_dropout=0.001, lin_layer_dropouts=[0.001,0.001]).to(device) #bs=8000
model = FeedForwardNN(emb_dims, no_of_cont=15, lin_layer_sizes=[200, 100], output_size=1, emb_dropout=0.3, lin_layer_dropouts=[0.1,0.1]).to(device)

no_of_epochs = 100

#criterion = nn.MSELoss()
criterion = torch.nn.BCELoss()
#criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
for epoch in range(no_of_epochs):
	for y, cont_x, cat_x in dataloader:
		cat_x = cat_x.to(device)
		cont_x = cont_x.to(device)
		y  = y.to(device)
		# Forward Pass
		#preds = model(cont_x, cat_x) #original
		preds = model.forward(cont_x, cat_x)
		loss = criterion(preds, y)
		# Backward Pass and Optimization
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	#print([preds[0:20], y[0:20]])
	#fpr, tpr, thresholds = metrics.roc_curve(y, preds, pos_label=1)
	pred_y = model.forward(test_cont_x, test_cat_x)
	if torch.cuda.is_available():
		auc_score = metrics.roc_auc_score(test_y.cpu().detach().numpy(), pred_y.cpu().detach().numpy())
		brier_score = metrics.brier_score_loss(test_y.cpu().detach().numpy(), pred_y.cpu().detach().numpy()) 
	else:
		auc_score = metrics.roc_auc_score(test_y.detach().numpy(), pred_y.detach().numpy())
		brier_score = metrics.brier_score_loss(test_y.detach().numpy(), pred_y.detach().numpy())
	print ("AUC score: ", auc_score)
	print ("Brier score: ", brier_score)
	print (loss.item())
