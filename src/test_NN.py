import sys

from sklearn.preprocessing import LabelEncoder


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import pandas as pd
import numpy as np

from sklearn import metrics, calibration

from pytorch_tabular import TabularDataset, FeedForwardNN

from temperature_scaling import ModelWithTemperature, _ECELoss

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

#categorical_features = ["MSSubClass", "MSZoning", "Street", "LotShape", "YearBuilt"]
#categorical_features = ["us5", "us4", "us3", "us2", "us1", "ds1", "ds2", "ds3", "ds4", "ds5", "tdist"]
categorical_features = ["us5", "us4", "us3", "us2", "us1", "ds1", "ds2", "ds3", "ds4", "ds5"]

output_feature = "mut_type"

#Encode categorical features
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for cat_col in categorical_features:
	label_encoders[cat_col] = LabelEncoder()
	data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])

dataset = TabularDataset(data=data, cat_cols=categorical_features, output_col=output_feature)

#split data into train, validation and test data
#train_set, val_set, test_set = data.random_split(dataset, (800, 100, 100))

#Encode categorical features for testing data
label_encoders = {}
for cat_col in categorical_features:
	label_encoders[cat_col] = LabelEncoder()
	data_test[cat_col] = label_encoders[cat_col].fit_transform(data_test[cat_col])

#decoder
#label_encoders['us5'].inverse_transform([int(i) for i in data.iloc[0, 0:10]]).tolist()

dataset_test = TabularDataset(data=data_test, cat_cols=categorical_features, output_col=output_feature)

#DataLoader for testing data
dataloader1 = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False, num_workers=1)

test_y, test_cont_x, test_cat_x = next(iter(dataloader1))

test_cont_x = test_cont_x.to(device)
test_cat_x = test_cat_x.to(device)
test_y = test_y.to(device)

#####

#DataLoader for the train data
batchsize = 10000
dataloader = DataLoader(dataset, batchsize, shuffle=True, num_workers=1)

dataloader2 = DataLoader(dataset, len(dataset), shuffle=False, num_workers=1)
all_y, all_cont_x, all_cat_x = next(iter(dataloader2))
all_cat_x = all_cat_x.to(device)
all_cont_x = all_cont_x.to(device)
all_y  = all_y.to(device)

cat_dims = [int(data[col].nunique()) for col in categorical_features]
#cat_dims
#[15, 5, 2, 4, 112]

emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
#emb_dims
#[(15, 8), (5, 3), (2, 1), (4, 2), (112, 50)]

# define the model 
#model = FeedForwardNN(emb_dims, no_of_cont=15, lin_layer_sizes=[50, 100], output_size=1, emb_dropout=0.04, lin_layer_dropouts=[0.001,0.01]).to(device)
#model = FeedForwardNN(emb_dims, no_of_cont=15, lin_layer_sizes=[50, 200], output_size=1, emb_dropout=0.001, lin_layer_dropouts=[0.001,0.001]).to(device) #bs=8000
model = FeedForwardNN(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[200, 100], output_size=1, emb_dropout=0.3, lin_layer_dropouts=[0.1,0.1]).to(device)

no_of_epochs = 30

#criterion = nn.MSELoss()
criterion = torch.nn.BCELoss()
#criterion = torch.nn.NLLLoss()
#criterion = torch.nn.BCEWithLogitsLoss()

#set Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

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
	#scaled_model = ModelWithTemperature(model)
	#scaled_model.set_temperature(dataloader1)

	#pred_y = scaled_model.forward(test_cont_x, test_cat_x)
	pred_y = model.forward(test_cont_x, test_cat_x)
	
	ece_model = _ECELoss(10)
	ece_model.zero_grad()

	#check flanking k-mers
	#pd.Series
	#tmp = pred_y.cpu().detach().numpy()
	#print(tmp)
	y_prob = pd.Series(data=to_np(pred_y).T[0], name="prob")	
	data_and_prob = pd.concat([data_test, y_prob], axis=1)

	all_pred_y = model.forward(all_cont_x, all_cat_x)
	all_y_prob = pd.Series(data=to_np(all_pred_y).T[0], name="prob")
	all_data_and_prob = pd.concat([data, all_y_prob], axis=1)

	print ('3mer correlation - test: ' + str(f3mer_comp(data_and_prob)))
	print ('3mer correlation - all: ' + str(f3mer_comp(all_data_and_prob)))
	print ('5mer correlation - test: ' + str(f5mer_comp(data_and_prob)))
	print ('5mer correlation - all: ' + str(f5mer_comp(all_data_and_prob)))
	print ('7mer correlation - test: ' + str(f7mer_comp(data_and_prob)))
	print ('7mer correlation - all: ' + str(f7mer_comp(all_data_and_prob)))
	
	#get the scores
	auc_score = metrics.roc_auc_score(to_np(test_y), to_np(pred_y))
	brier_score = metrics.brier_score_loss(to_np(test_y), to_np(pred_y))
	test_pred = to_np(torch.cat((test_y,pred_y),1))
	logits = torch.cat((1-pred_y,pred_y),1)
	logits = torch.log(logits/(1-logits))
	#ECE = to_np(ece_model.forward(logits, test_y.long()))
	prob_true, prob_pred = calibration.calibration_curve(to_np(test_y), to_np(pred_y),n_bins=50)
	
	#print("calibration: ", np.column_stack((prob_pred,prob_true)))
	
	print ("AUC score: ", auc_score)
	print ("Brier score: ", brier_score)
	#print ("ECE score: ", ECE.item())
	print (loss.item())
	#np.savetxt(sys.stdout, test_pred, fmt='%s', delimiter='\t')
