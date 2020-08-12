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

train_local_RNA = np.array(Cover.create_from_bigwig(name="", bigwigfiles='/public/home/licai/DNMML/data/germ_cell/Guo_2016_CR/merge_replicates.PGC.RNA-seq.hg19.log2.bw', roi=train_bed, resolution=11, flank=5)).reshape(-1, 1)

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

model = Network(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[100, 50], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15], in_channels=4, out_channels=50, kernel_size=12, RNN_hidden_size=0, RNN_layers=1, last_lin_size=25).to(device)

model2 = FeedForwardNN(emb_dims, no_of_cont=n_cont, lin_layer_sizes=[100, 50], emb_dropout=0.2, lin_layer_dropouts=[0.15, 0.15]).to(device)

############################
seq = Bioseq.create_from_refgenome(name='dna', refgenome='hg19_ucsc_ordered.fa', roi='test.bed')

#seqs in numbers
seq.iseq4idx([0])

#merge the functional signatures and the sequences
c = torch.cat((torch.tensor(seq, dtype=torch.float32).squeeze(1),torch.tensor(a, dtype=torch.float32).squeeze(1)), axis=3)

#
v = torch.tensor(range(16)).view(16,1)

#one-hot encoding to labels
label = torch.matmul(torch.tensor(seq, dtype=torch.long).squeeze(1), v).squeeze()
#label shape: sample_size * seq_len

embed = nn.Embedding(16,8)
out = embed(label)


bed = BedTool('test1.bed')

#get the score value
scores = [pos.score for pos in bed]

seq = Bioseq.create_from_refgenome(name='dna', refgenome='hg19_ucsc_ordered.fa', roi=bed, flank=0)

def flank(feature, size):
    feature.start -=size+1
    feature.stop +=size
    return feature

bed10 = bed.each(flank, size=10)


seq = Bioseq.create_from_refgenome(name='dna', refgenome='hg19_ucsc_ordered.fa', roi=bed1, flank=50)


