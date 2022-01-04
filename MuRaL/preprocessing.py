import os
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
from itertools import product

def to_np(tensor):
    """Convert Tensor to numpy arrays"""
    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()

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
    """Generate the H5 file for storing distal data"""
    n_channels = 4**distal_order + len(bw_files)
    
    write_h5f = True
    if os.path.exists(h5f_path):
        try:
            with h5py.File(h5f_path, 'r') as hf:
                bed_path = bed_regions.fn
                
                # Check whether the existing H5 file is latest and complete
                #if os.path.getmtime(bed_path) < os.path.getmtime(h5f_path) and len(bed_regions) == hf["distal_X"].shape[0] and n_channels == hf["distal_X"].shape[1]:
                # Check whether the existing H5 file (not following the link) is latest and complete
                if os.lstat(bed_path).st_mtime < os.lstat(h5f_path).st_mtime and len(bed_regions) == hf["distal_X"].shape[0] and n_channels == hf["distal_X"].shape[1]:
                    write_h5f = False
        except OSError:
            print('Warning: re-genenerating the H5 file, because the file is empty or imcomplete:', h5f_path)
            
    # If the H5 file is unavailable or im complete, generate the file
    if write_h5f:            
        with h5py.File(h5f_path, 'w') as hf:
            
            print('Generating HDF5 file:', h5f_path)
            sys.stdout.flush()
            
            # Total seq len
            seq_len =  distal_radius*2+1-(distal_order-1)
            
            # Create distal_X dataset
            # Note, the default dtype for create_dataset is numpy.float32
            hf.create_dataset(name='distal_X', shape=(0, n_channels, seq_len), compression="gzip", compression_opts=2, chunks=(h5_chunk_size,n_channels, seq_len), maxshape=(None,n_channels, seq_len)) 
            
            # Write data in chunks
            chunk_size = 50000
            for start in range(0, len(bed_regions), chunk_size):
                end = min(start+chunk_size, len(bed_regions))
                
                # Extract sequence from the genome, which is in one-hot encoding format
                seqs = Bioseq.create_from_refgenome(name='distal', refgenome=ref_genome, roi=bed_regions.at(range(start, end)), flank=distal_radius, order=distal_order, verbose=True)
                
                #print('seqs.shape 1:', seqs.shape)
                # Get the correct shape (batch_size, channels, seq_len) for pytorch
                seqs = np.array(seqs).squeeze().transpose(0,2,1)
                
                # Handle distal bigWig data, return base-wise values
                if len(bw_files) > 0:
                    bw_distal = Cover.create_from_bigwig(name='', bigwigfiles=bw_files, roi=bed_regions.at(range(start, end)), resolution=1, flank=distal_radius, verbose=True)
                    #print('bw_distal.shape:', np.array(bw_distal).shape)
                    
                    #bw_distal should have the same seq len as that for distal_seq
                    bw_distal = np.array(bw_distal).squeeze(axis=(1,3)).transpose(0,2,1)[:,:,:(distal_radius*2-distal_order+2)]

                    # Concatenate the sequence data and the bigWig data
                    seqs = np.concatenate((seqs, bw_distal), axis=1)       
                # Write the numpy array into the H5 file
                hf['distal_X'].resize((hf['distal_X'].shape[0] + seqs.shape[0]), axis = 0)
                hf['distal_X'][-seqs.shape[0]:] = seqs

    return None

def prepare_local_data(bed_regions, ref_genome, bw_files, bw_names, local_radius, local_order, seq_only):
    """Prepare local data for given regions"""
    
    # Use janggu Bioseq to read the data
    local_seq = Bioseq.create_from_refgenome(name='local', refgenome=ref_genome, roi=bed_regions, flank=local_radius, order=1)    
    
    # Check whether the data is correctly extracted (e.g. not all sites are A/T; incorrect padding in the beginning of a chromosome)
    if np.unique(np.array(local_seq)[:,:,local_radius,:,:], axis=0).shape[0] != 1:
        print('ERROR: The positions in input BED file have multiple nucleotides! The ref_genome or input BED file could be wrong.', file=sys.stderr)
        sys.exit()
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
        
        # NOTE: replace k-mers with 'N' with a large number; the padding numbers at the two ends of the chromosomes are also large numbers
        #local_seq_cat2= np.where(local_seq_cat2>=0, local_seq_cat2, 4**local_order)     
        local_seq_cat2 = np.where(np.logical_and(local_seq_cat2>=0, local_seq_cat2<=4**local_order), local_seq_cat2, 4**local_order)

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

    return data_local, seq_cols, categorical_features, output_feature


class CombinedDatasetH5(Dataset):
    """Combine local data and distal into Dataset, with H5"""
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
        
        # First, change labels to digits (NOTE: the labels already digitalized)
        #label_encoders = {}
        #print("Not using LabelEncoder ...")
        #for cat_col in cat_cols:
            #label_encoders[cat_col] = LabelEncoder()
            #data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])
            #keywords = [''.join(i) for i in product(['A','C','G','T'], repeat = 3)]
            #a = LabelEncoder().fit(keywords)
        
        # Sample size
        self.n = data.shape[0]
        
        # Output column
        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))
        
        # Names of categorical columns
        self.cat_cols = cat_cols
        
        # Set biggest dimension for each categorical column
        self.cat_dims = [np.max(data[col]) + 1 for col in cat_cols]
        
        # Find the continuous columns
        self.cont_cols = [col for col in data.columns if col not in self.cat_cols + seq_cols + [output_col]]
        
        # Assign the continuous data to cont_X
        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1)) 
        
        # Assign the categorical data to cat_X
        if len(self.cat_cols) > 0:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            print("Error: no categorical data, something is wrong!", file=sys.stderr)
            sys.exit()
        
        # For distal data
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
            
            # Open the H5 file once
            self.distal_X = h5py.File(self.h5f_path, 'r')["distal_X"]
            #print('open h5f file:', self.h5f_path)     
        
        return self.y[idx], self.cont_X[idx], self.cat_X[idx], np.array(self.distal_X[idx, 0:self.n_channels, :])
    
    def get_labels(self): 
        return np.squeeze(self.y)
    
    def _get_labels(self, dataset, idx):
        return dataset.__getitem__(idx)[1]
    
        #return np.squeeze(self.y)



def prepare_dataset_h5(bed_regions, ref_genome, bw_files, bw_names, local_radius=5, local_order=1, distal_radius=50, distal_order=1, h5f_path='distal_data.h5', h5_chunk_size=1, seq_only=False):
    """Prepare the datasets for given regions, using H5 file"""
 
    # Generate H5 file for distal data
    generate_h5f(bed_regions, h5f_path, ref_genome, distal_radius, distal_order, bw_files, h5_chunk_size)
    
    # Prepare local data
    data_local, seq_cols, categorical_features, output_feature = prepare_local_data(bed_regions, ref_genome, bw_files, bw_names, local_radius, local_order, seq_only)

    # If seq_only flag was set, bigWig files will be ignored
    if seq_only:
        n_channels = 4**distal_order
        print('NOTE: seq_only flag was set, so will not use any bigWig track!')
    else:
        n_channels = 4**distal_order + len(bw_files)
    
    # Combine local data and distal into Dataset objects
    dataset = CombinedDatasetH5(data=data_local, seq_cols=seq_cols, cat_cols=categorical_features, output_col=output_feature, h5f_path=h5f_path, n_channels=n_channels)
    
    #return dataset, data_local, categorical_features
    return dataset


class CombinedDataset(Dataset):

    """Combine local data and distal into Dataset, without H5"""
    def __init__(self, data, seq_cols, cat_cols, output_col, distal_data):
        """  
        Args:
            data: DataFrame containing local seq data and categorical data
            seq_cols: names of local seq columns
            cat_cols: names of categorical columns used for training
            output_col: name of the label column
            distal_data: distal data
        """
        # Store the local seq data and label for later use
        self.data_local = data[seq_cols+[output_col]]
        
        # First, change labels to digits
        #label_encoders = {}
        #for cat_col in cat_cols:
        #    label_encoders[cat_col] = LabelEncoder()
        #    data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])
        
        # Sample size
        self.n = data.shape[0]
        
        # Output column
        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1), dtype=float32)
        
        # Names of categorical columns
        self.cat_cols = cat_cols
        
        # Set biggest dimension for each categorical column
        self.cat_dims = [np.max(data[col]) + 1 for col in cat_cols]
        
        # Find the continuous columns
        self.cont_cols = [col for col in data.columns if col not in self.cat_cols + seq_cols + [output_col]]
        
        # Assign the continuous data to cont_X
        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
            print ('using bigWig ...', self.cont_cols)
        else:
            self.cont_X = np.zeros((self.n, 1)) 
            print ('not using bigWig ...', self.cont_cols)
        
        # Assign the categorical data to cat_X
        if len(self.cat_cols) > 0:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            print("Error: no categorical data, something is wrong!", file=sys.stderr)
            sys.exit()
        
        # For distal data
        self.distal_X = distal_data.astype(np.float32)
        #print('Number of channels to be used for distal data:', self.n_channels)
        

    def __len__(self):
        """ Denote the total number of samples. """
        return self.n

    def __getitem__(self, idx):
        """ Generate one sample of data. """  
        
        return self.y[idx], self.cont_X[idx], self.cat_X[idx], self.distal_X[idx]

def prepare_dataset(bed_regions, ref_genome, bw_files, bw_names, local_radius=5, local_order=1, distal_radius=50, distal_order=1, seq_only=False):
    """Prepare the datasets for given regions, without H5"""
    
    data_local, seq_cols, categorical_features, output_feature = prepare_local_data(bed_regions, ref_genome, bw_files, bw_names, local_radius, local_order, seq_only)
    
    #dataset_local = LocalDataset(data=data_local, cat_cols=categorical_features, output_col=output_feature)

    # For the distal data, first extract the sequences and convert them to one-hot-encoding data 
    distal_seq = Bioseq.create_from_refgenome(name='distal', refgenome=ref_genome, roi=bed_regions, flank=distal_radius, order=distal_order)
    # Note the shape of data
    distal_seq = np.array(distal_seq).squeeze().transpose(0,2,1)
    
    # Handle distal bigWig data
    if len(bw_files) > 0 and seq_only == False:
        bw_distal = Cover.create_from_bigwig(name='', bigwigfiles=bw_files, roi=bed_regions, resolution=1, flank=distal_radius)
 
        #bw_distal should have the same seq len as that for distal_seq
        bw_distal = np.array(bw_distal).squeeze(axis=(1,3)).transpose(0,2,1)[:,:,:(distal_radius*2-distal_order+2)]
        
        # Concatenate the sequence data and the bigWig data
        distal_seq = np.concatenate((distal_seq, bw_distal), axis=1).astype(np.float32)
        
        print('distal_seq.dtype:', distal_seq.dtype)
    
    #dataset_distal = DistalDataset([distal_seq, y])
    
    # Combine local  and distal data into a Dataset
    dataset = CombinedDataset(data=data_local, seq_cols=seq_cols, cat_cols=categorical_features, output_col=output_feature, distal_data=distal_seq)
    
    return dataset