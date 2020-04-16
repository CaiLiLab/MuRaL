from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import pandas as pd
import numpy as np

from pytorch_tabular import TabularDataset, FeedForwardNN

# Using only a subset of the variables.
data = pd.read_csv("test1.nonCpG.5bins.h10000.usds.csv").dropna()

#categorical_features = ["MSSubClass", "MSZoning", "Street", "LotShape", "YearBuilt"]
categorical_features = ["us5us4us3us2us1", "ds1ds2ds3ds4ds5", "tdist"]

output_feature = "mut_type"

from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for cat_col in categorical_features:
	label_encoders[cat_col] = LabelEncoder()
	data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])


dataset = TabularDataset(data=data, cat_cols=categorical_features,
									 output_col=output_feature)

batchsize = 64
dataloader = DataLoader(dataset, batchsize, shuffle=True, num_workers=1)

cat_dims = [int(data[col].nunique()) for col in categorical_features]
print (cat_dims)
#[15, 5, 2, 4, 112]

emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
print (emb_dims)
#[(15, 8), (5, 3), (2, 1), (4, 2), (112, 50)]

device = torch.device("cpu")
model = FeedForwardNN(emb_dims, no_of_cont=15, lin_layer_sizes=[50, 100], output_size=1, emb_dropout=0.04, lin_layer_dropouts=[0.001,0.01]).to(device)

no_of_epochs = 30
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
for epoch in range(no_of_epochs):
	for y, cont_x, cat_x in dataloader:
				
		cat_x = cat_x.to(device)
		cont_x = cont_x.to(device)
		y  = y.to(device)
		# Forward Pass
		preds = model(cont_x, cat_x)
		loss = criterion(preds, y)
		# Backward Pass and Optimization
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	print (loss.item())
