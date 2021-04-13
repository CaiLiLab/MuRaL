import os, os.path
import sys

from janggu.data import Bioseq, Cover
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

def to_np(tensor):
    """Convert numpy arrays to Tensor"""
    if torch.cuda.is_available():
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()


class HDF5Dataset(Dataset):
    """Combine local data and distal into Dataset"""
    def __init__(self, data, seq_cols, cat_cols, output_col, h5f_path, n_channels):
        """  
        Args:
            data: DataFrame containing local seq data and categorical data
            seq_cols: names of local seq columns
            cat_cols: names of categorical columns used for training
            output_col: name of the label column
            h5f_path: H5 file storing the distal data
            n_channels: number of columns (channels) in distal data to be extracted
        """
        # Store the local seq data and label for later use
        self.data_local = data[seq_cols+[output_col]]
        
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
        
        self.cat_dims = [np.max(data[col]) +1 for col in cat_cols]
        
        self.cont_cols = [col for col in data.columns if col not in self.cat_cols + seq_cols + [output_col]]

        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1)) 

        if len(self.cat_cols) >0:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            #self.cat_X =    np.zeros((self.n, 1)) 
            print("Error: no categorical data, something is wrong!", file=sys.stderr)
            sys.exit()
        
        self.h5f_path = h5f_path
        self.distal_X = None
        self.n_channels = n_channels
        print('Number of channels to be used for distal data:', self.n_channels)
        

    def __len__(self):
        """ Denote the total number of samples. """
        return self.n

    def __getitem__(self, idx):
        """ Generate one sample of data. """
        if self.distal_X is None:

            self.distal_X = h5py.File(self.h5f_path, 'r')["distal_X"]
            #print('open h5f file:', self.h5f_path)
            #print('idx is:', idx)
        
        
        return self.y[idx], self.cont_X[idx], self.cat_X[idx], np.array(self.distal_X[idx, 0:self.n_channels, :])

def get_h5f_path(bed_file, bw_names, distal_radius, distal_order):
    """Get the H5 file path name based on input data"""
    
    h5f_path = bed_file + '.distal_' + str(distal_radius)
    if distal_order > 1:
        h5f_path = h5f_path + '_' + str(distal_order)
    if len(bw_names) > 0:
        h5f_path = h5f_path + '.' + '.'.join(list(bw_names))
    h5f_path = h5f_path + '.h5'
    
    return h5f_path

def generate_h5f(bed_regions, h5f_path, ref_genome, distal_radius, distal_order, bw_files, h5_chunk_size):
    n_channels = 4**distal_order + len(bw_files)
    
    write_h5f = True
    if os.path.exists(h5f_path):
        try:
            with h5py.File(h5f_path, 'r') as hf:
                if len(bed_regions) == hf["distal_X"].shape[0] and n_channels == hf["distal_X"].shape[1]:
                    write_h5f = False
        except OSError:
            print('Warning: the file is empty or imcomplete:', h5f_path)
            
    
    if write_h5f:            
        with h5py.File(h5f_path, 'w') as hf:
            #hf.create_dataset("X_train", data=X_train_data, maxshape=(None, 512, 512, 9))
            #hf.create_dataset("X_test", data=X_test_data, maxshape=(None, 512, 512, 9))
            seq_len =  distal_radius*2+1-(distal_order-1)
            hf.create_dataset(name='distal_X', shape=(0, n_channels, seq_len), compression="gzip", compression_opts=2, chunks=(h5_chunk_size,n_channels, seq_len), maxshape=(None,n_channels, seq_len)) 

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

    return None


def prepare_dataset(bed_regions, ref_genome, bw_files, bw_names, local_radius=5, local_order=1, distal_radius=50, distal_order=1, h5f_path='distal_data.h5', h5_chunk_size=1, seq_only=False):
    """Prepare the datasets for given regions"""
    
    # Use janggu Bioseq to read the data
    local_seq = Bioseq.create_from_refgenome(name='local', refgenome=ref_genome, roi=bed_regions, flank=local_radius, order=1)    
    
    # To get One-Hot encoded data, shape is [sample_size, 4*seq_len]
    #local_seq = np.array(local_seq).squeeze().reshape(local_seq.shape[0], -1)   
    
    # Get the digitalized seq data; values are one of [0,1,2,3]
    local_seq_cat = local_seq.iseq4idx(list(range(local_seq.shape[0]))).astype(np.int32)
    
    # NOTE: replace negatives with 0, meaning replacing 'N' with 'A'
    local_seq_cat = np.where(local_seq_cat>=0, local_seq_cat, 0)
    
    # Assign column names and convert to DataFrame
    seq_cols = ['us'+str(local_radius - i) for i in range(local_radius)] + ['mid'] + ['ds'+str(i+1) for i in range(local_radius)]
    local_seq_cat = pd.DataFrame(local_seq_cat, columns = seq_cols)
    
    if local_order > 1:
        local_seq2 = Bioseq.create_from_refgenome(name='local', refgenome=ref_genome, roi=bed_regions, flank=local_radius, order=local_order)
        
        # NOTE: use np.int64 because nn.Embedding needs a Long type
        local_seq_cat2 = local_seq2.iseq4idx(list(range(local_seq2.shape[0]))).astype(np.int64)
        
        # NOTE: replace k-mers with 'N' with a large number
        local_seq_cat2= np.where(local_seq_cat2>=0, local_seq_cat2, 4**local_order) 

        # TO DO: some other categorical data may be added here
        
        # Names of the categorical variables
        cat_n = local_radius*2 +1 - (local_order-1)
        categorical_features  = ['cat'+str(i+1) for i in range(cat_n)]
        
        local_seq_cat2 = pd.DataFrame(local_seq_cat2, columns = categorical_features)
        local_seq_cat2 = pd.concat([local_seq_cat, local_seq_cat2], axis=1)
    else:
        local_seq_cat2 = local_seq_cat
        categorical_features = seq_cols
    
    print('local_seq_cat2 shape and columns:', local_seq_cat2.shape, local_seq_cat2.columns)
    print('categorical_features:', categorical_features)
    
    # The 'score' field in the BED file stores the label/class information
    y = np.array([float(loc.score) for loc in bed_regions], ndmin=2).reshape((-1,1)) # shape: (n_row, 1)
    y = pd.DataFrame(y, columns=['mut_type'])
    output_feature = 'mut_type'
    
    # Add feature data in bigWig files
    if len(bw_files) > 0 and seq_only == False:
        # Use the mean value of the region of 2*radius+1 bp around the focal site
        bw_data = np.array(Cover.create_from_bigwig(name='local', bigwigfiles=bw_files, roi=bed_regions, resolution=2*local_radius+1, flank=local_radius)).reshape(len(bed_regions), -1)

        bw_data = pd.DataFrame(bw_data, columns=bw_names)
        #print ('bw_data.shape', bw_data.shape, local_seq_cat.shape)

        data_local = pd.concat([local_seq_cat2, bw_data, y], axis=1)
    else:
        data_local = pd.concat([local_seq_cat2, y], axis=1)
    
    # Generate H5 file for distal data
    generate_h5f(bed_regions, h5f_path, ref_genome, distal_radius, distal_order, bw_files, h5_chunk_size)

    # If seq_only flag was set, bigWig files will be ignored
    if seq_only:
        n_channels = 4**distal_order
        print('NOTE: seq_only flag was set, so will not use any bigWig track!')
    else:
        n_channels = 4**distal_order + len(bw_files)
    
    # Combine local data and distal into Dataset objects
    dataset = HDF5Dataset(data=data_local, seq_cols=seq_cols, cat_cols=categorical_features, output_col=output_feature, h5f_path=h5f_path, n_channels=n_channels)
    
    #return dataset, data_local, categorical_features
    return dataset





