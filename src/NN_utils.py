import sys
import math
import random
import gzip
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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

def load_data(train_file):
	data = pd.read_csv(train_file, sep='\t').dropna()
	seq_data = data['sequence']
	y_data = data['label']

	seqs_ohe = seqs2ohe(seq_data, 6)

	dataset = seqDataset([seqs_ohe, y_data])
	print(dataset[0:2][0][0][0:4,4:10])

def main():
	#set train file
	train_file = sys.argv[1]

	#set test file
	test_file = sys.argv[2]
	load_data(train_file)	

if __name__ == "__main__":
	main()



