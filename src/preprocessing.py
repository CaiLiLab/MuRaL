
import os, os.path
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from janggu.data import Bioseq, Cover
sys.stderr = stderr

from pybedtools import BedTool

from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import h5py

from sklearn import metrics, calibration

# Convert numpy arrays to Tensor
def to_np(tensor):
    if torch.cuda.is_available():
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()

# Deprecated.One-hot encoding for the sequence. 
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

# Deprecated. One-hot encoding for multple sequences.
def seqs2ohe(sequences,motiflen=24):

    dataset=[]
    for row in sequences:             
        dataset.append(seq2ohe(row,motiflen))
        
  
    return dataset

# Define a Dataset for handling distal data
class DistalDataset(Dataset):

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

class DistalDataset2(Dataset):

    def __init__(self, h5f_path):
        
        self.h5f_path = h5f_path
        self.dataset = None
        with h5py.File(self.h5f_path, 'r') as h5f:
            self.len = len(h5f["distal_X"])
        

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5f_path, 'r')["distal_X"]
            
            
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

# 
class HDF5Dataset(Dataset):
    def __init__(self, data, cat_cols, output_col, h5f_path):
        
        self.data_local = data
        
        # First, change labels to digits
        label_encoders = {}
        for cat_col in cat_cols:
            label_encoders[cat_col] = LabelEncoder()
            data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])
        
        self.n = data.shape[0]

        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))

        self.cat_cols = cat_cols
        self.cont_cols = [col for col in data.columns if col not in self.cat_cols + [output_col]]

        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))

        if len(self.cat_cols) >0:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            self.cat_X =    np.zeros((self.n, 1)) #this may not be needed!!
        
        #============================
        self.h5f_path = h5f_path
        self.distal_X = None
        

    def __len__(self):
        """
        Denote the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generate one sample of data.
        """
        if self.distal_X is None:
            #self.distal_X = h5py.File(self.h5f_path, 'r', rdcc_nbytes=50*1024**2)["distal_X"]
            self.distal_X = h5py.File(self.h5f_path, 'r')["distal_X"]
            #print('open h5f file:', self.h5f_path)
            #print('idx is:', idx)
        
        
        return self.y[idx], self.cont_X[idx], self.cat_X[idx], np.array(self.distal_X[idx])
    
# Define a Dataset with both local data and distal data
class CombinedDataset(Dataset):
    """ Combined dataset."""

    def __init__(self,local_dataset, distal_dataset):
        
        self.y = local_dataset.y
        self.local_cont_X = local_dataset.cont_X
        self.local_cat_X = local_dataset.cat_X
        self.distal_X = distal_dataset.x_data
        self.len=len(self.y)

    def __getitem__(self, index):
        return self.y[index], self.local_cont_X[index], self.local_cat_X[index], self.distal_X[index]

    def __len__(self):
        return self.len

# Deprecated.
def gen_ohe_dataset(data):
    seq_data = data['seq']
    y_data = data['mut_type'].astype(np.float32).values.reshape(-1, 1)

    seqs_ohe = seqs2ohe(seq_data, motiflen=6)

    dataset = DistalDataset([seqs_ohe, y_data])
    #print(dataset[0:2][0][0][0:4,4:10])
    
    return dataset

# Deprecated.
def separate_local_distal(data, radius = 5): 
    seq_len = len(data['seq'][0])
    mid_pos = int((seq_len+1)/2)

    adj_seq = pd.DataFrame([list(el[mid_pos-(radius+1):mid_pos+radius]) for el in data['seq']])
    adj_seq.columns = ['us'+str(radius - i)for i in range(radius)] + ['mid'] + ['ds'+str(i+1)for i in range(radius)]

    # Local sequences and functional genomic data
    data_local = pd.concat([adj_seq, data.drop(['pos','seq'], axis=1)], axis=1)

    # Consider more distal sequences
    data_distal = data[['seq', 'mut_type']]
    
    categorical_features = list(adj_seq.columns)
    #categorical_features = ["us5", "us4", "us3", "us2", "us1", "ds1", "ds2", "ds3", "ds4", "ds5"]
    
    return data_local, data_distal, categorical_features

# Define a Dataset class for handling local signals with categorical and continuous data  
class LocalDataset(Dataset):
    def __init__(self, data, cat_cols, output_col):
        """
        Characterizes a Dataset for PyTorch

        Parameters
        ----------

        data: pandas data frame
            The data frame object for the input data. It must
            contain all the continuous, categorical and the
            output columns to be used.

        cat_cols: List of strings
            The names of the categorical columns in the data.
            These columns will be passed through the embedding
            layers in the model. These columns must be
            label encoded beforehand. 

        output_col: string
            The name of the output variable column in the data
            provided.
        """
        # First, change labels to digits
        label_encoders = {}
        for cat_col in cat_cols:
            label_encoders[cat_col] = LabelEncoder()
            data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])
        
        self.n = data.shape[0]

        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))

        self.cat_cols = cat_cols
        self.cont_cols = [col for col in data.columns if col not in self.cat_cols + [output_col]]

        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))

        if len(self.cat_cols) >0:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            self.cat_X =    np.zeros((self.n, 1))

    def __len__(self):
        """
        Denote the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generate one sample of data.
        """
        return [self.y[idx], self.cont_X[idx], self.cat_X[idx]]

# Prepare the datasets for given regions
def prepare_dataset(bed_regions, ref_genome,  bw_files, bw_names, radius=5, distal_radius=50, distal_order=1):
    
    # Use janggu Bioseq to read the data
    local_seq = Bioseq.create_from_refgenome(name='', refgenome=ref_genome, roi=bed_regions, flank=radius)

    # Get the digitalized seq data
    local_seq_cat = local_seq.iseq4idx(list(range(local_seq.shape[0])))

    # TO DO: some other categorical data can be added here
    # Names of the categorical variables
    categorical_features = ['us'+str(radius - i)for i in range(radius)] + ['mid'] + ['ds'+str(i+1)for i in range(radius)]

    local_seq_cat = pd.DataFrame(local_seq_cat, columns = categorical_features)

    # The 'score' field in the BED file stores the label/class information
    y = np.array([float(loc.score) for loc in bed_regions], ndmin=2).reshape((-1,1))
    y = pd.DataFrame(y, columns=['mut_type'])
    output_feature = 'mut_type'

    if len(bw_files) > 0:
        # Use the mean value of the region of 2*radius+1 bp around the focal site
        bw_data = np.array(Cover.create_from_bigwig(name='', bigwigfiles=bw_files, roi=bed_regions, resolution=2*radius+1, flank=radius)).reshape(len(bed_regions), -1)

        bw_data = pd.DataFrame(bw_data, columns=bw_names)
        #print ('bw_data.shape', bw_data.shape, local_seq_cat.shape)

        data_local = pd.concat([local_seq_cat, bw_data, y], axis=1)
    else:
        data_local = pd.concat([local_seq_cat, y], axis=1)

    dataset_local = LocalDataset(data=data_local, cat_cols=categorical_features, output_col=output_feature)

    # For the distal data, first extract the sequences and convert them to one-hot-encoding data 
    distal_seq = Bioseq.create_from_refgenome(name='distal', refgenome=ref_genome, roi=bed_regions, flank=distal_radius, order=distal_order)
    # Note the shape of data
    distal_seq = np.array(distal_seq).squeeze().transpose(0,2,1)
    
    # Handle distal bigWig data
    if len(bw_files) > 0:
        bw_distal = Cover.create_from_bigwig(name='', bigwigfiles=bw_files, roi=bed_regions, resolution=1, flank=distal_radius)
        
        #print('bw_distal.shape:', np.array(bw_distal).shape)
        #bw_distal should have the same seq len as that for distal_seq
        bw_distal = np.array(bw_distal).squeeze(axis=(1,3)).transpose(0,2,1)[:,:,:(distal_radius*2-distal_order+2)]
        
        # Concatenate the sequence data and the bigWig data
        distal_seq = np.concatenate((distal_seq, bw_distal), axis=1)
    
    dataset_distal = DistalDataset([distal_seq, y])
    
    # Combine local Dataset and distal Dataset
    dataset = CombinedDataset(dataset_local, dataset_distal)
    
    return dataset, data_local, categorical_features

# Prepare the datasets for given regions
def prepare_dataset2(bed_regions, ref_genome,  bw_files, bw_names, radius=5, distal_radius=50, distal_order=1, h5f_path='distal_data.h5', h5_chunk_size=1):
    
    # Use janggu Bioseq to read the data
    local_seq = Bioseq.create_from_refgenome(name='local', refgenome=ref_genome, roi=bed_regions, flank=radius)
    
    # To get One-Hot encoded data, shape is [sample_size, 4*seq_len]
    #local_seq = np.array(local_seq).squeeze().reshape(local_seq.shape[0], -1)   
    
    # Get the digitalized seq data; values are one of 0,1,2,3
    local_seq_cat = local_seq.iseq4idx(list(range(local_seq.shape[0]))).astype(np.int8)

    # TO DO: some other categorical data can be added here
    # Names of the categorical variables
    categorical_features = ['us'+str(radius - i) for i in range(radius)] + ['mid'] + ['ds'+str(i+1) for i in range(radius)]

    local_seq_cat = pd.DataFrame(local_seq_cat, columns = categorical_features)

    # The 'score' field in the BED file stores the label/class information
    y = np.array([float(loc.score) for loc in bed_regions], ndmin=2).reshape((-1,1))
    y = pd.DataFrame(y, columns=['mut_type'])
    output_feature = 'mut_type'

    if len(bw_files) > 0:
        # Use the mean value of the region of 2*radius+1 bp around the focal site
        bw_data = np.array(Cover.create_from_bigwig(name='local', bigwigfiles=bw_files, roi=bed_regions, resolution=2*radius+1, flank=radius)).reshape(len(bed_regions), -1)

        bw_data = pd.DataFrame(bw_data, columns=bw_names)
        #print ('bw_data.shape', bw_data.shape, local_seq_cat.shape)

        data_local = pd.concat([local_seq_cat, bw_data, y], axis=1)
    else:
        data_local = pd.concat([local_seq_cat, y], axis=1)

    #dataset_local = LocalDataset(data=data_local, cat_cols=categorical_features, output_col=output_feature)

    ##=============================================
    # For the distal data, first extract the sequences and convert them to one-hot-encoding data 
    #distal_seq = np.empty((0, 4, distal_radius*2+1), dtype=np.float32)
    
    n_channels = 4 + len(bw_files)
    
    write_h5f = True
    if os.path.exists(h5f_path):
        try:
            with h5py.File(h5f_path, 'r') as hf:
                if len(y) == hf["distal_X"].shape[0] and n_channels == hf["distal_X"].shape[1]:
                    write_h5f = False
        except OSError:
            print('Warning: the file is empty or imcomplete:', h5f_path)
            
    
    if write_h5f:            
        with h5py.File(h5f_path, 'w') as hf:
            #hf.create_dataset("X_train", data=X_train_data, maxshape=(None, 512, 512, 9))
            #hf.create_dataset("X_test", data=X_test_data, maxshape=(None, 512, 512, 9))
            hf.create_dataset(name='distal_X', shape=(0, n_channels, distal_radius*2+1), compression="gzip", compression_opts=2, chunks=(h5_chunk_size,n_channels, distal_radius*2+1), maxshape=(None,n_channels, distal_radius*2+1)) 

            chunk_size = 50000
            for start in range(0, len(bed_regions), chunk_size):
                end = min(start+chunk_size, len(bed_regions))
                seqs = Bioseq.create_from_refgenome(name='distal', refgenome=ref_genome, roi=bed_regions.at(range(start, end)), flank=distal_radius, order=distal_order, verbose=True)
                #print('seqs:', seqs)
                seqs = np.array(seqs).squeeze().transpose(0,2,1)
                #print('np.array(seqs):', seqs)

                # Handle distal bigWig data
                if len(bw_files) > 0:
                    bw_distal = Cover.create_from_bigwig(name='', bigwigfiles=bw_files, roi=bed_regions.at(range(start, end)), resolution=1, flank=distal_radius, verbose=True)

                    #print('bw_distal.shape:', np.array(bw_distal).shape)
                    #bw_distal should have the same seq len as that for distal_seq
                    bw_distal = np.array(bw_distal).squeeze(axis=(1,3)).transpose(0,2,1)[:,:,:(distal_radius*2-distal_order+2)]

                    # Concatenate the sequence data and the bigWig data
                    seqs = np.concatenate((seqs, bw_distal), axis=1)       
                    #distal_seq = np.concatenate((distal_seq, seqs), axis=0)

                hf['distal_X'].resize((hf['distal_X'].shape[0] + seqs.shape[0]), axis = 0)
                hf['distal_X'][-seqs.shape[0]:] = seqs
    

    
    #dataset_distal = DistalDataset([distal_seq, y])
    #===========================================
    
    
    # Combine local Dataset and distal Dataset
    #dataset = CombinedDataset(dataset_local, dataset_distal)
    dataset = HDF5Dataset(data=data_local, cat_cols=categorical_features, output_col=output_feature, h5f_path=h5f_path)
    
    #return dataset, data_local, categorical_features
    return dataset

def prepare_dataset3(bed_regions, ref_genome,  bw_files, bw_names, radius=5, distal_radius=50, distal_order=1, h5f_path='distal_data.h5', h5_chunk_size=1):
    
    # Use janggu Bioseq to read the data
    local_seq = Bioseq.create_from_refgenome(name='local', refgenome=ref_genome, roi=bed_regions, flank=radius)
    
    # To get One-Hot encoded data, shape is [sample_size, 4*seq_len]
    #local_seq = np.array(local_seq).squeeze().reshape(local_seq.shape[0], -1)   
    
    # Get the digitalized seq data; values are one of 0,1,2,3
    local_seq_cat = local_seq.iseq4idx(list(range(local_seq.shape[0]))).astype(np.int8)

    # TO DO: some other categorical data can be added here
    # Names of the categorical variables
    categorical_features = ['us'+str(radius - i) for i in range(radius)] + ['mid'] + ['ds'+str(i+1) for i in range(radius)]

    local_seq_cat = pd.DataFrame(local_seq_cat, columns = categorical_features)

    # The 'score' field in the BED file stores the label/class information
    y = np.array([float(loc.score) for loc in bed_regions], ndmin=2).reshape((-1,1))
    y = pd.DataFrame(y, columns=['mut_type'])
    output_feature = 'mut_type'

    if len(bw_files) > 0:
        # Use the mean value of the region of 2*radius+1 bp around the focal site
        bw_data = np.array(Cover.create_from_bigwig(name='local', bigwigfiles=bw_files, roi=bed_regions, resolution=2*radius+1, flank=radius)).reshape(len(bed_regions), -1)

        bw_data = pd.DataFrame(bw_data, columns=bw_names)
        #print ('bw_data.shape', bw_data.shape, local_seq_cat.shape)

        data_local = pd.concat([local_seq_cat, bw_data, y], axis=1)
    else:
        data_local = pd.concat([local_seq_cat, y], axis=1)

    #dataset_local = LocalDataset(data=data_local, cat_cols=categorical_features, output_col=output_feature)

    ##=============================================
    # For the distal data, first extract the sequences and convert them to one-hot-encoding data 
    #distal_seq = np.empty((0, 4, distal_radius*2+1), dtype=np.float32)
    
    n_channels = 4 + len(bw_files)
    
    write_h5f = True
    if os.path.exists(h5f_path):
        try:
            with h5py.File(h5f_path, 'r') as hf:
                if len(y) == hf["distal_X"].shape[0] and n_channels == hf["distal_X"].shape[1]:
                    write_h5f = False
        except OSError:
            print('Warning: the file is empty or imcomplete:', h5f_path)
            
    
    if write_h5f:            
        with h5py.File(h5f_path, 'w') as hf:
            #hf.create_dataset("X_train", data=X_train_data, maxshape=(None, 512, 512, 9))
            #hf.create_dataset("X_test", data=X_test_data, maxshape=(None, 512, 512, 9))
            hf.create_dataset(name='distal_X', shape=(0, n_channels, distal_radius*2+1), compression="gzip", compression_opts=2, chunks=(h5_chunk_size,n_channels, distal_radius*2+1), maxshape=(None,n_channels, distal_radius*2+1)) 

            chunk_size = 50000
            for start in range(0, len(bed_regions), chunk_size):
                end = min(start+chunk_size, len(bed_regions))
                seqs = Bioseq.create_from_refgenome(name='distal', refgenome=ref_genome, roi=bed_regions.at(range(start, end)), flank=distal_radius, order=distal_order, verbose=True)
                #print('seqs:', seqs)
                seqs = np.array(seqs).squeeze().transpose(0,2,1)
                #print('np.array(seqs):', seqs)

                # Handle distal bigWig data
                if len(bw_files) > 0:
                    bw_distal = Cover.create_from_bigwig(name='', bigwigfiles=bw_files, roi=bed_regions, resolution=1, flank=distal_radius, verbose=True)

                    #print('bw_distal.shape:', np.array(bw_distal).shape)
                    #bw_distal should have the same seq len as that for distal_seq
                    bw_distal = np.array(bw_distal).squeeze(axis=(1,3)).transpose(0,2,1)[:,:,:(distal_radius*2-distal_order+2)]

                    # Concatenate the sequence data and the bigWig data
                    seqs = np.concatenate((seqs, bw_distal), axis=1)       
                    #distal_seq = np.concatenate((distal_seq, seqs), axis=0)

                hf["distal_X"].resize((hf["distal_X"].shape[0] + seqs.shape[0]), axis = 0)
                hf["distal_X"][-seqs.shape[0]:] = seqs
    

    
    #dataset_distal = DistalDataset([distal_seq, y])
    #===========================================
    
    
    # Combine local Dataset and distal Dataset
    #dataset = CombinedDataset(dataset_local, dataset_distal)
    dataset = HDF5Dataset(data=data_local, cat_cols=categorical_features, output_col=output_feature, h5f_path=h5f_path)
    
    #return dataset, data_local, categorical_features
    return dataset

# Deprecated. Old function
def load_data(data_file):
    
    data = pd.read_csv(data_file, sep='\t').dropna()
    seq_data = data['sequence']
    y_data = data['label'].astype(np.float32).values.reshape(-1, 1)

    seqs_ohe = seqs2ohe(seq_data, 6)

    dataset = DistalDataset([seqs_ohe, y_data])
    #print(dataset[0:2][0][0][0:4,4:10])
    
    return dataset




