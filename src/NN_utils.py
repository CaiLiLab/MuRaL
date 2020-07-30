import sys
import math
import random
import gzip
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn import metrics, calibration

from evaluation import f3mer_comp, f5mer_comp, f7mer_comp

def to_np(tensor):
	if torch.cuda.is_available():
		return tensor.cpu().detach().numpy()
	else:
		return tensor.detach().numpy()

def seq2ohe(sequence,motlen):
	rows = len(sequence)+2*motlen-2
	S = np.empty([rows,4])
	base = 'ACGT'
	for i in range(rows):
		for j in range(4):
			if i-motlen+1<len(sequence) and sequence[i-motlen+1].upper() =='N' or i<motlen-1 or i>len(sequence)+motlen-2:
				S[i,j]=np.float32(0.25)
			elif sequence[i-motlen+1].upper() == base[j]:
				S[i,j]=np.float32(1)
			else:
				S[i,j]=np.float32(0)
	return np.transpose(S)

def seqs2ohe(sequences,motiflen=24):

	dataset=[]
	for row in sequences:			  
		dataset.append(seq2ohe(row,motiflen))
  
	return dataset

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

def load_data(data_file):
	data = pd.read_csv(data_file, sep='\t').dropna()
	seq_data = data['sequence']
	y_data = data['label'].astype(np.float32).values.reshape(-1, 1)

	seqs_ohe = seqs2ohe(seq_data, 6)

	dataset = seqDataset([seqs_ohe, y_data])
	#print(dataset[0:2][0][0][0:4,4:10])
	
	return dataset

class ConvNN(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, lin_layer_size):
		super(ConvNN, self).__init__()
		#self.cnn = nn.Conv1d(in_channels, out_channels, kernel_size)
		#self.maxpool =  nn.MaxPool1d(kernel_size, stride)
		
		self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool1d(2, 2), # kernel_size, stride
            nn.Conv1d(out_channels, out_channels*2, kernel_size),
            nn.ReLU()
        )
		
		self.fc = nn.Sequential(
            #nn.BatchNorm1d(out_channels*2)
			nn.Linear(out_channels*2, lin_layer_size), ####TO DO
            nn.ReLU(),
			nn.Dropout(0.1), #dropout prob
            nn.Linear(lin_layer_size, 1),
			nn.Sigmoid()
        )
		
		#nn.init.kaiming_normal_(self.fc.weight.data) #
	
	def forward(self, input_data):
		#input data shape: batch_size, in_channels, L_in (lenth of sequence)
		out = self.conv(input_data) #out_shape: batch_size, L_out; L_out = floor((L_in+2*padding-kernel_size)/stride + 1)
		out, _ = torch.max(out, dim=2)
		
		out = self.fc(out)
		
		return out

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv1d') != -1:
		nn.init.normal_(m.weight, 0.0, 0.1)
		print(m.weight.shape)
	elif classname.find('Linear') != -1:
		nn.init.normal_(m.weight, 0, 0.1)
		nn.init.zeros_(m.bias)
		print(m.weight.shape)

def train_network(model, no_of_epochs, dataloader):
	criterion = nn.BCELoss()
	#criterion = torch.nn.NLLLoss()
	#criterion = torch.nn.BCEWithLogitsLoss()

	#set Optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

	for epoch in range(no_of_epochs):
		for x_data, y in dataloader:
			x_data = x_data.to(device)
			y  = y.to(device)
		
			# Forward Pass
			#preds = model(cont_x, cat_x) #original
			preds = model.forward(x_data)
			#print("preds")
			#print(preds)
			#print(y)
			
			loss = criterion(preds, y)
		
			# Backward Pass and Optimization
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
	
	return model
	#pred_y = model.forward(test_cont_x, test_cat_x)

def eval_model(model, test_x, test_y, device):
	model.eval()
	
	test_x = test_x.to(device)
	test_y = test_y.to(device)
	
	pred_y = model.forward(test_x)
	y_prob = pd.Series(data=to_np(pred_y).T[0], name="prob")
	
	print("data_and_prob:")
	#print(to_np(test_x))
	#data_and_prob = pd.concat([pd.DataFrame(to_np(test_x)), y_prob], axis=1)

	auc_score = metrics.roc_auc_score(to_np(test_y), to_np(pred_y))
	brier_score = metrics.brier_score_loss(to_np(test_y), to_np(pred_y))
	print ("AUC score: ", auc_score)
	print ("Brier score: ", brier_score)

def main():
	
	print("CUDA: ", torch.cuda.is_available())
	
	global device
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#set train file
	train_file = sys.argv[1]
	#train_file = "/public/home/licai/DNMML/analysis/test/merge.95win.A.pos.c1000.train.30k.gz"
	
	#set test file
	test_file = sys.argv[2]
	
	dataset = load_data(train_file)
	train_x, train_y = dataset.x_data, dataset.y_data
	dataset_test = load_data(test_file)
	test_x, test_y = dataset_test.x_data, dataset_test.y_data,
	#test_x, test_y = torch.tensor(dataset_test[0].values), torch.tensor(dataset_test[1].values)

	dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
	
	model = ConvNN(4, 120, 10, 100).to(device)
	model.apply(weights_init)
	
	no_of_epochs = 10
	
	model.train()
	model = train_network(model, no_of_epochs, dataloader)

	#######
	model.to('cpu')
	device = 'cpu'
	print("for the tain data:")
	eval_model(model, train_x, train_y, device)
	
	print("for the test data:")
	eval_model(model, test_x, test_y, device)
	#print ('3mer correlation - test: ' + str(f3mer_comp(data_and_prob)))
	#print ('5mer correlation - test: ' + str(f5mer_comp(data_and_prob)))
	#print ('7mer correlation - test: ' + str(f7mer_comp(data_and_prob)))
		
if __name__ == "__main__":
	main()



