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
from evaluation import f3mer_comp, f5mer_comp, f7mer_comp, f3mer_comp_rand, f5mer_comp_rand, f7mer_comp_rand

print("CUDA: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(' '.join(sys.argv))
#set train file
train_file = sys.argv[1]

#set test file
test_file = sys.argv[2]

ref_genome='/public/home/licai/DNMML/data/hg19/hg19_ucsc_ordered.fa'

train_bed = BedTool(train_file)

test_bed = BedTool(test_file)


bw_files = []
bw_names = []

#the width to be considered for local signals
if len(sys.argv)>3:
    radius = int(sys.argv[3])
else:
    radius = 5
print('radius:', radius)

#the width to be considered for more distal signals
if len(sys.argv)>4:
    n_rows = int(sys.argv[4])
else:
    n_rows = 10000

dataset, data_local, categorical_features = prepare_dataset(train_bed, ref_genome, bw_files,bw_names, radius, 10, 1)

f3mer_comp_rand(data_local, n_rows)

if n_rows>=100000:
    f5mer_comp_rand(data_local, n_rows)
    f7mer_comp_rand(data_local, n_rows)

