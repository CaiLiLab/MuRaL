import os
import sys

#from janggu.data import Bioseq, Cover
import pyBigWig
from pybedtools import BedTool
from Bio import SeqIO

from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import h5py
import time

from sklearn import metrics, calibration
from itertools import product

from functools import partial
from itertools import repeat
from multiprocessing import Pool
import re
import subprocess


def to_np(tensor):
    """Convert Tensor to numpy arrays"""
    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()

def get_h5f_path(bed_file, bw_names, distal_radius, distal_order, without_bw_distal):
    """Get the H5 file path name based on input data"""
    
    h5f_path = bed_file + '.distal_' + str(distal_radius)
    
    if distal_order > 1:
        h5f_path = h5f_path + '_' + str(distal_order)
        
    if len(bw_names) > 0 and (not without_bw_distal):
        h5f_path = h5f_path + '.' + '.'.join(list(bw_names))
    
    h5f_path = h5f_path + '.h5'
    
    return h5f_path

def generate_h5f(bed_regions, h5f_path, ref_genome, distal_radius, distal_order, bw_files, h5_chunk_size, chunk_size=50000, without_bw_distal=False):
    """Generate the H5 file for storing distal data"""
    
    if without_bw_distal:
        n_channels = 4**distal_order
    else:
        n_channels = 4**distal_order + len(bw_files)
    
    write_h5f = True
    if os.path.exists(h5f_path):
        try:
            with h5py.File(h5f_path, 'r', swmr=True) as hf:
                bed_path = bed_regions.fn
                
                # Check whether the existing H5 file is latest and complete
                #if os.path.getmtime(bed_path) < os.path.getmtime(h5f_path) and len(bed_regions) == hf["distal_X"].shape[0] and n_channels == hf["distal_X"].shape[1]:
                # Check whether the existing H5 file (not following the link) is latest and complete
                if os.lstat(bed_path).st_mtime < os.lstat(h5f_path).st_mtime and len(bed_regions) == hf["distal_X"].shape[0] and n_channels == hf["distal_X"].shape[1]:
                    write_h5f = False
        except OSError:
            print('Warning: re-genenerating the H5 file, because the file is empty or imcomplete:', h5f_path)
            
    # If the H5 file is unavailable or imcomplete, generate the file
    if write_h5f:            
        with h5py.File(h5f_path, 'w') as hf:
            
            print('Generating HDF5 file:', h5f_path)
            sys.stdout.flush()
            
            # Total seq len
            seq_len =  distal_radius*2+1-(distal_order-1)
            
            # Create distal_X dataset
            # Note, the default dtype for create_dataset is numpy.float32
            hf.create_dataset(name='distal_X', shape=(0, n_channels, seq_len), compression="gzip", compression_opts=4, chunks=(h5_chunk_size,n_channels, seq_len), maxshape=(None,n_channels, seq_len)) 
            
            # Write data in chunks
            # chunk_size = 50000
            seq_records = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))
            for start in range(0, len(bed_regions), chunk_size):
                end = min(start+chunk_size, len(bed_regions))
                
                # Extract sequence from the genome, which is in one-hot encoding format
                seqs = get_digitalized_seq_ohe(seq_records, bed_regions.at(range(start, end)), distal_radius)
                
                # Handle distal bigWig data, return base-wise values
                if len(bw_files) > 0 and (not without_bw_distal):
 
                    bw_distal = get_bw_for_bed(bw_files, bed_regions.at(range(start, end)), distal_radius)

                    # Concatenate the sequence data and the bigWig data
                    seqs = np.concatenate((seqs, bw_distal), axis=1)       
                # Write the numpy array into the H5 file
                hf['distal_X'].resize((hf['distal_X'].shape[0] + seqs.shape[0]), axis = 0)
                hf['distal_X'][-seqs.shape[0]:] = seqs

    return None


def generate_h5fv2(bed_regions, h5f_path, ref_genome, distal_radius, distal_order, bw_paths, bw_files, chunk_size=50000, n_h5_files=1, without_bw_distal=False):
    """Generate the H5 file for storing distal data"""
    if without_bw_distal:
        n_channels = 4**distal_order
    else:    
        n_channels = 4**distal_order + len(bw_files)
    
    write_h5f = True
    if os.path.exists(h5f_path):
        try:
            with h5py.File(h5f_path, 'r', swmr=True) as hf:
                bed_path = bed_regions.fn
                
                # Check whether the existing H5 file (not following the link) is latest and complete
                if len(hf.keys()) ==1 \
                and os.lstat(bed_path).st_mtime < os.lstat(h5f_path).st_mtime \
                and len(bed_regions) == hf["distal_X"].shape[0] \
                and n_channels == hf["distal_X"].shape[1]:
                    write_h5f = False
                
                if len(hf.keys()) > 1:
                    try:
                        h5_sample_size = sum([hf[key].shape[0] for key in hf.keys()])
                        
                        if os.lstat(bed_path).st_mtime < os.lstat(h5f_path).st_mtime \
                        and len(bed_regions) == h5_sample_size \
                        and n_channels == hf["distal_X1"].shape[1]:
                            write_h5f = False
                    except KeyError:
                        print('Warning: re-genenerating the H5 file, because the file is empty or imcomplete:', h5f_path)
                                       
        except OSError:
            print('Warning: re-genenerating the H5 file, because the file is empty or imcomplete:', h5f_path)

            
    # If the H5 file is unavailable or im complete, generate the file
    if write_h5f:            
        print('Genenerating the H5 file:', h5f_path)
        sys.stdout.flush()
        
        args = ['gen_distal_h5', 
                 '--ref_genome', ref_genome, 
                 '--bed_file', bed_regions.fn, 
                 '--distal_radius', str(distal_radius), 
                 '--distal_order', str(distal_order), 
                 '--n_files', str(n_h5_files), 
                 '--chunk_size', str(chunk_size)]
        if bw_paths != None:
            args.append('--bw_paths')
            args.append(bw_paths)
        if without_bw_distal:
            args.append('--without_bw_distal')
        p = subprocess.Popen(args)
        p.wait()
            
    return None



def generate_h5f_singlev1(bed_regions, h5f_path, ref_genome, distal_radius, distal_order, bw_files, chunk_size, without_bw_distal):
    """generate an HDF file for specific regions"""
    #bed_regions = BedTool(bed_file)
    if without_bw_distal:
        n_channels = 4**distal_order
    else:
        n_channels = 4**distal_order + len(bw_files)
    
    with h5py.File(h5f_path, 'w') as hf:

        print('Generating HDF5 file:', h5f_path)
        sys.stdout.flush()

        # Total seq len
        seq_len =  distal_radius*2+1-(distal_order-1)

        # Create distal_X dataset
        # Note, the default dtype for create_dataset is numpy.float32
        hf.create_dataset(name='distal_X', shape=(0, n_channels, seq_len), compression="gzip", compression_opts=4, chunks=(1,n_channels, seq_len), maxshape=(None,n_channels, seq_len)) 

        # Write data in chunks
        #chunk_size = 50000
        
        seq_records = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))
        for start in range(0, len(bed_regions), chunk_size):
            end = min(start+chunk_size, len(bed_regions))

            # Extract sequence from the genome, which is in one-hot encoding format
            seqs = get_digitalized_seq_ohe(seq_records, bed_regions.at(range(start, end)), distal_radius)
            
            # Handle distal bigWig data, return base-wise values
            if len(bw_files) > 0 and (not without_bw_distal):
                bw_distal = get_bw_for_bed(bw_files, bed_regions.at(range(start, end)), distal_radius)

                # Concatenate the sequence data and the bigWig data
                #seqs = np.concatenate((seqs, bw_distal), axis=1)
                seqs = np.concatenate((seqs, bw_distal), axis=1).round(decimals=2)       
            # Write the numpy array into the H5 file
            hf['distal_X'].resize((hf['distal_X'].shape[0] + seqs.shape[0]), axis = 0)
            hf['distal_X'][-seqs.shape[0]:] = seqs
    
    return h5f_path

def generate_h5f_singlev2(bed_regions, h5f_path, ref_genome, distal_radius, distal_order, binsize, bw_files, chunk_size, without_bw_distal):
    
    #bed_regions = BedTool(bed_file)
    if without_bw_distal:
        n_channels = 4**distal_order
    else:
        n_channels = 4**distal_order + len(bw_files)
    
    with h5py.File(h5f_path, 'w') as hf:

        print('Generating HDF5 file:', h5f_path)
        sys.stdout.flush()

        # Total seq len
        seq_len_orig =  distal_radius*2+1-(distal_order-1)
        seq_len = int(np.ceil(seq_len_orig/binsize))
        pad_left = 0
        pad_right = 0
        if seq_len_orig % binsize:
            pad_len = binsize - (seq_len_orig % binsize)
            pad_left = pad_len//2
            pad_right = pad_len - pad_left

        # Create distal_X dataset
        # Note, the default dtype for create_dataset is numpy.float32
        hf.create_dataset(name='distal_X', shape=(0, n_channels, seq_len), compression="gzip", compression_opts=4, chunks=(1,n_channels, seq_len), maxshape=(None,n_channels, seq_len)) 

        # Write data in chunks
        #chunk_size = 50000
        
        seq_records = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))
        for start in range(0, len(bed_regions), chunk_size):
            end = min(start+chunk_size, len(bed_regions))

            # Extract sequence from the genome, which is in one-hot encoding format
            seqs = get_digitalized_seq_ohe(seq_records, bed_regions.at(range(start, end)), distal_radius)
            
            # Handle distal bigWig data, return base-wise values
            if len(bw_files) > 0 and (not without_bw_distal):
                bw_distal = get_bw_for_bed(bw_files, bed_regions.at(range(start, end)), distal_radius)

                # Concatenate the sequence data and the bigWig data
                #seqs = np.concatenate((seqs, bw_distal), axis=1)
                seqs = np.concatenate((seqs, bw_distal), axis=1).round(decimals=2)        
            
            ######
            #print('seqs[0]:', seqs[0])
            #print('seqs before: ', seqs.shape)
            seqs = np.pad(seqs, ((0,0), (0,0), (pad_left,pad_right))).reshape(seqs.shape[0], seqs.shape[1],-1,binsize).mean(axis=3).round(decimals=2)
            #print('seqs after: ', seqs.shape)
            #print('seqs[0]:', seqs[0])
            
            assert seqs.shape[2] == seq_len
            ######
            # Write the numpy array into the H5 file
            hf['distal_X'].resize((hf['distal_X'].shape[0] + seqs.shape[0]), axis = 0)
            hf['distal_X'][-seqs.shape[0]:] = seqs
    
    return h5f_path


def get_digitalized_seq(ref_genome, bed_regions, radius, order):
    seq_records = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))

    digit_encoder = {'A':0,'C':1,'G':2,'T':3,
               'R':-1, #A,G
               'Y':-1, #C,T
               'M':-1, #A,C
               'S':-1, #C,G
               'W':-1, #A,T
               'K':-1, #G,T
               'B':-1, #not A
               'D':-1, #not C
               'H':-1, #not G
               'V':-1, #not T
               'N':-1}

    digit_encoder_rc = {'A':3, 'C':2, 'G':1, 'T':0,
               'R':-1, #A,G
               'Y':-1, #C,T
               'M':-1, #A,C
               'S':-1, #C,G
               'W':-1, #A,T
               'K':-1, #G,T
               'B':-1, #not A
               'D':-1, #not C
               'H':-1, #not G
               'V':-1, #not T
               'N':-1}

    #self.records = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))

    digit_seqs = []
    for region in bed_regions:
        chrom, start, stop, strand = str(region.chrom), region.start, region.stop, region.strand

        #long_seq_record = self.records[chrom]

        long_seq = str(seq_records[chrom].seq)
        long_seq_len = len(long_seq)

        start1 = np.max([int(start)-radius, 0])
        stop1 = np.min([int(stop)+radius, long_seq_len])
        short_seq = long_seq[start1:stop1].upper()

        seq_len = 2*radius + 1 

        if(len(short_seq) < seq_len):
            #print('warning:', chrom, start1, stop1, long_seq_len)
            if start1 == 0:
                short_seq = (seq_len - len(short_seq))*'N' + short_seq
                #print(short_seq)
            else:
                short_seq = short_seq + (seq_len - len(short_seq))*'N'
                #print(short_seq)
        #a = np.concatenate([digit_encoder[c] for c in short_seq], axis=1)
        if strand == '+':
            digit_seq = np.array([digit_encoder[c] for c in short_seq])
        else:
            #a = [digit_encoder_rc[c] for c in short_seq[::-1]]
            #a.reverse()
            digit_seq = np.array([digit_encoder_rc[c] for c in short_seq[::-1]])
            
        if order > 1:
            new_seq = []
            for i in range(seq_len - order +1):
                kmer = digit_seq[i:i+order]
                if min(kmer) < 0:
                    new_seq.append(-1)
                else:
                    digit = sum([kmer[d]*4**(order-d-1) for d in range(order)])
                    new_seq.append(digit)
            
            digit_seq = np.array(new_seq)
        
        if len(digit_seq) != seq_len - order +1:
            print('digit_seq.shape:', digit_seq.shape, chrom, start, stop)
            print('short_seq:', short_seq)
        digit_seqs.append(digit_seq)
    
    digit_seqs = np.array(digit_seqs, dtype=np.int32)
    
    return digit_seqs

def get_mean_bw_for_bed(bw_files, bw_names, bw_radii, bed_regions):

    bw_fh = []
    for file in bw_files:
        bw_fh.append(pyBigWig.open(file))
    
    bw_data = np.zeros((len(bed_regions), len(bw_fh)), dtype=float)
    
    if len(bw_fh) > 0:
        
        for i, region in enumerate(bed_regions):
            chrom, start, stop = str(region.chrom), region.start, region.stop
            #bw_values = []
            #seq_len = [bw.chroms(chrom) for bw in bw_fh]
            
            for j, bw in enumerate(bw_fh):
                            
                start1 = max([int(start)-bw_radii[j], 0])
                stop1 = min([int(stop)+bw_radii[j], bw.chroms(chrom)])
                bw_data[i,j] = np.nan_to_num(bw.values(chrom, start1, stop1, numpy=True)).mean()
                

        bw_data = pd.DataFrame(bw_data, columns=bw_names)
    
    return bw_data

def get_digitalized_seq_ohe(seq_records, bed_regions, distal_radius):

    one_hot_encoder = {'A':np.array([[1,0,0,0]], dtype=np.float32).T,
               'C':np.array([[0,1,0,0]], dtype=np.float32).T,
               'G':np.array([[0,0,1,0]], dtype=np.float32).T,
               'T':np.array([[0,0,0,1]], dtype=np.float32).T,
               'R':np.array([[0.5,0,0.5,0]], dtype=np.float32).T, #A,G
               'Y':np.array([[0,0.5,0,0.5]], dtype=np.float32).T, #C,T
               'M':np.array([[0.5,0.5,0,0]], dtype=np.float32).T, #A,C
               'S':np.array([[0,0.5,0.5,0]], dtype=np.float32).T, #C,G
               'W':np.array([[0.5,0,0,0.5]], dtype=np.float32).T, #A,T
               'K':np.array([[0,0,0.5,0.5]], dtype=np.float32).T, #G,T
               'B':np.array([[0,1/3,1/3,1/3]], dtype=np.float32).T, #not A
               'D':np.array([[1/3,0,1/3,1/3]], dtype=np.float32).T, #not C
               'H':np.array([[1/3,1/3,0,1/3]], dtype=np.float32).T, #not G
               'V':np.array([[1/3,1/3,1/3,0]], dtype=np.float32).T, #not T
               'N':np.array([[0.25,0.25,0.25,0.25]], dtype=np.float32).T}

    one_hot_encoder_rc = {'A':np.array([[0,0,0,1]], dtype=np.float32).T,
               'C':np.array([[0,0,1,0]], dtype=np.float32).T,
               'G':np.array([[0,1,0,0]], dtype=np.float32).T,
               'T':np.array([[1,0,0,0]], dtype=np.float32).T,
               'R':np.array([[0,0.5,0,0.5]], dtype=np.float32).T, #A,G
               'Y':np.array([[0.5,0,0.5,0]], dtype=np.float32).T, #C,T
               'M':np.array([[0,0,0.5,0.5]], dtype=np.float32).T, #A,C
               'S':np.array([[0,0.5,0.5,0]], dtype=np.float32).T, #C,G
               'W':np.array([[0.5,0,0,0.5]], dtype=np.float32).T, #A,T
               'K':np.array([[0.5,0.5,0,0]], dtype=np.float32).T, #G,T
               'B':np.array([[1/3,1/3,1/3,0]], dtype=np.float32).T, #not A
               'D':np.array([[1/3,1/3,0,1/3]], dtype=np.float32).T, #not C
               'H':np.array([[1/3,0,1/3,1/3]], dtype=np.float32).T, #not G
               'V':np.array([[0,1/3,1/3,1/3]], dtype=np.float32).T, #not T
               'N':np.array([[0.25,0.25,0.25,0.25]], dtype=np.float32).T}

    #self.records = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))

    distal_seqs = []
    for region in bed_regions:
        chrom, start, stop, strand = str(region.chrom), region.start, region.stop, region.strand

        #long_seq_record = self.records[chrom]

        long_seq = str(seq_records[chrom].seq)
        long_seq_len = len(long_seq)

        start1 = np.max([int(start)-distal_radius, 0])
        stop1 = np.min([int(stop)+distal_radius, long_seq_len])
        short_seq = long_seq[start1:stop1].upper()

        seq_len = 2*distal_radius + 1 

        if(len(short_seq) < seq_len):
            #print('warning:', chrom, start1, stop1, long_seq_len)
            if start1 == 0:
                short_seq = (seq_len - len(short_seq))*'N' + short_seq
                #print(short_seq)
            else:
                short_seq = short_seq + (seq_len - len(short_seq))*'N'
                #print(short_seq)
        #a = np.concatenate([one_hot_encoder[c] for c in short_seq], axis=1)
        if strand == '+':
            distal_seq = np.concatenate([one_hot_encoder[c] for c in short_seq], axis=1)
        else:
            #a = [one_hot_encoder_rc[c] for c in short_seq[::-1]]
            #a.reverse()
            distal_seq = np.concatenate([one_hot_encoder_rc[c] for c in short_seq[::-1]], axis=1)
        if distal_seq.shape[1] != seq_len:
            print('distal_seq.shape:', distal_seq.shape, chrom, start, stop)
            print('short_seq:', short_seq)
        distal_seqs.append(distal_seq)
    
    distal_seqs = np.array(distal_seqs)
    
    return distal_seqs
    

def get_bw_for_bed(bw_files, bed_regions, radius):

    bw_fh = []
    for file in bw_files:
        bw_fh.append(pyBigWig.open(file))
    
    bw_data = []
    
    seq_len = radius*2+1
    
    if len(bw_fh) > 0:
        
        for i, region in enumerate(bed_regions):
            chrom, start, stop, strand = str(region.chrom), region.start, region.stop, region.strand
            #bw_values = []
            #seq_len = [bw.chroms(chrom) for bw in bw_fh]
            
            bw_list = []
            for j, bw in enumerate(bw_fh):
                            
                start1 = max([int(start)-radius, 0])
                stop1 = min([int(stop)+radius, bw.chroms(chrom)])
                
                bw_values = np.nan_to_num(bw.values(chrom, start1, stop1, numpy=True))
                if(len(bw_values) < seq_len):
                    if start1 == 0:
                        bw_values = np.concatenate([(seq_len - len(bw_values))*[0], bw_values])
                    else:
                        bw_values = np.concatenate([bw_values, (seq_len - len(bw_values))*[0]])
                
                if strand == '-':
                    bw_values = np.flip(bw_values)
                
                bw_list.append(bw_values)
            
            bw_data.append(bw_list)     

        bw_data = np.array(bw_data).astype(np.float32)
    
    return bw_data

def prepare_local_data(bed_regions, ref_genome, bw_files, bw_names, bw_radii, local_radius, local_order, seq_only):
    """Prepare local data for given regions"""
    
    # Read the seq data
    local_seq_cat = get_digitalized_seq(ref_genome, bed_regions, local_radius, order=1)    
    
    # Check whether the data is correctly extracted (e.g. not all sites are A/T; incorrect padding in the beginning of a chromosome)
    if np.unique(local_seq_cat[:,local_radius], axis=0).shape[0] != 1:
        print('ERROR: The positions in input BED file have different bases (A/T and C/G mixed)! The ref_genome or input BED file could be wrong.', file=sys.stderr)
        sys.exit()
  
    # NOTE: replace negatives with 0, meaning replacing 'N' with 'A'
    local_seq_cat = np.where(local_seq_cat>=0, local_seq_cat, 0)
    
    # Assign column names and convert to DataFrame
    seq_cols = ['us'+str(local_radius - i) for i in range(local_radius)] + ['mid'] + ['ds'+str(i+1) for i in range(local_radius)]
    local_seq_cat = pd.DataFrame(local_seq_cat, columns = seq_cols)
    
    if local_order > 1:
        local_seq_cat2 = get_digitalized_seq(ref_genome, bed_regions, local_radius, order=local_order)
        
        # NOTE: use np.int64 because nn.Embedding needs a Long type
        local_seq_cat2 = local_seq_cat2.astype(np.int64)
        
        # NOTE: replace k-mers with 'N' with a large number; the padding numbers at the two ends of the chromosomes are also large numbers   
        local_seq_cat2 = np.where(np.logical_and(local_seq_cat2>=0, local_seq_cat2<=4**local_order), local_seq_cat2, 4**local_order)
        
        # Names of the categorical variables
        cat_n = local_radius*2 +1 - (local_order-1)
        categorical_features  = ['cat'+str(i+1) for i in range(cat_n)]
        
        local_seq_cat2 = pd.DataFrame(local_seq_cat2, columns = categorical_features)
        local_seq_cat2 = pd.concat([local_seq_cat, local_seq_cat2], axis=1)
    else:
        local_seq_cat2 = local_seq_cat.astype(np.int64)
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
        bw_data = get_mean_bw_for_bed(bw_files, bw_names, bw_radii, bed_regions)
 
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
        self.h5f = None
        self.single_h5_size = 0
        self.n_channels = n_channels
        print('Number of channels to be used for distal data:', self.n_channels)
        

    def __len__(self):
        """ Denote the total number of samples. """
        return self.n

    def __getitem__(self, idx):
        """ Generate one sample of data. """
        if self.h5f is None:
            
            # Open the H5 file once
            self.h5f = h5py.File(self.h5f_path, 'r', swmr=True)
            
            if len(self.h5f.keys()) > 1:
                self.single_h5_size = self.h5f['distal_X1'].shape[0]
            #print('open h5f file:', self.h5f_path)     
        if self.single_h5_size > 0:
            file_i = (idx // self.single_h5_size) + 1
            idx1 = idx % self.single_h5_size
            return self.y[idx], self.cont_X[idx], self.cat_X[idx], np.array(self.h5f['distal_X'+str(file_i)][idx1, 0:self.n_channels, :])
        
        else:
            return self.y[idx], self.cont_X[idx], self.cat_X[idx], np.array(self.h5f['distal_X'][idx, 0:self.n_channels, :])
    
    def get_labels(self): 
        return np.squeeze(self.y)
    
    def _get_labels(self, dataset, idx):
        return dataset.__getitem__(idx)[1]




class CombinedDatasetNP(Dataset):
    """Combine local data and distal into Dataset, using NumPy funcions"""
    def __init__(self, data, seq_cols, cat_cols, output_col, ref_genome, bed_regions, distal_radius, n_channels, bw_files, seq_only, without_bw_distal):
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
        #self.h5f_path = h5f_path
        self.distal_X = None
        self.n_channels = n_channels
        self.bw_files = bw_files
        
        ####
        self.bw_fh = []
        for file in self.bw_files:
            self.bw_fh.append(pyBigWig.open(file))
        ####
        self.seq_only = seq_only
        self.without_bw_distal = without_bw_distal
        print('Number of channels to be used for distal data:', self.n_channels)
        
        self.distal_radius = distal_radius
        self.seq_len = 2*distal_radius + 1 
        
        self.bed_regions = bed_regions
        self.bed_pd = pd.read_csv(bed_regions.fn, sep='\t', header=None, memory_map=True)
        self.bed_pd.columns = ['chrom', 'start', 'stop', 'name', 'score', 'strand']

        self.one_hot_encoder = {'A':np.array([[1,0,0,0]], dtype=np.float32).T,
               'C':np.array([[0,1,0,0]], dtype=np.float32).T,
               'G':np.array([[0,0,1,0]], dtype=np.float32).T,
               'T':np.array([[0,0,0,1]], dtype=np.float32).T,
               'R':np.array([[0.5,0,0.5,0]], dtype=np.float32).T, #A,G
               'Y':np.array([[0,0.5,0,0.5]], dtype=np.float32).T, #C,T
               'M':np.array([[0.5,0.5,0,0]], dtype=np.float32).T, #A,C
               'S':np.array([[0,0.5,0.5,0]], dtype=np.float32).T, #C,G
               'W':np.array([[0.5,0,0,0.5]], dtype=np.float32).T, #A,T
               'K':np.array([[0,0,0.5,0.5]], dtype=np.float32).T, #G,T
               'B':np.array([[0,1/3,1/3,1/3]], dtype=np.float32).T, #not A
               'D':np.array([[1/3,0,1/3,1/3]], dtype=np.float32).T, #not C
               'H':np.array([[1/3,1/3,0,1/3]], dtype=np.float32).T, #not G
               'V':np.array([[1/3,1/3,1/3,0]], dtype=np.float32).T, #not T
               'N':np.array([[0.25,0.25,0.25,0.25]], dtype=np.float32).T}

        self.one_hot_encoder_rc = {'A':np.array([[0,0,0,1]], dtype=np.float32).T,
               'C':np.array([[0,0,1,0]], dtype=np.float32).T,
               'G':np.array([[0,1,0,0]], dtype=np.float32).T,
               'T':np.array([[1,0,0,0]], dtype=np.float32).T,
               'R':np.array([[0,0.5,0,0.5]], dtype=np.float32).T, #A,G
               'Y':np.array([[0.5,0,0.5,0]], dtype=np.float32).T, #C,T
               'M':np.array([[0,0,0.5,0.5]], dtype=np.float32).T, #A,C
               'S':np.array([[0,0.5,0.5,0]], dtype=np.float32).T, #C,G
               'W':np.array([[0.5,0,0,0.5]], dtype=np.float32).T, #A,T
               'K':np.array([[0.5,0.5,0,0]], dtype=np.float32).T, #G,T
               'B':np.array([[1/3,1/3,1/3,0]], dtype=np.float32).T, #not A
               'D':np.array([[1/3,1/3,0,1/3]], dtype=np.float32).T, #not C
               'H':np.array([[1/3,0,1/3,1/3]], dtype=np.float32).T, #not G
               'V':np.array([[0,1/3,1/3,1/3]], dtype=np.float32).T, #not T
               'N':np.array([[0.25,0.25,0.25,0.25]], dtype=np.float32).T}
            
        self.records = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))
        

    def __len__(self):
        """ Denote the total number of samples. """
        return self.n

    def __getitem__(self, idx):
        """ Generate one sample of data. """

        region = self.bed_pd.iloc[idx]
        chrom, start, stop, strand = str(region.chrom), region.start, region.stop, region.strand
        
        #long_seq_record = self.records[chrom]

        long_seq = str(self.records[chrom].seq)
        long_seq_len = len(long_seq)

        start1 = np.max([int(start)-self.distal_radius, 0])
        stop1 = np.min([int(stop)+self.distal_radius, long_seq_len])
        short_seq = long_seq[start1:stop1].upper()

        if(len(short_seq) < self.seq_len):
            #print('warning:', chrom, start1, stop1, long_seq_len)
            if start1 == 0:
                short_seq = (self.seq_len - len(short_seq))*'N' + short_seq
                #print(short_seq)
            else:
                short_seq = short_seq + (self.seq_len - len(short_seq))*'N'
                #print(short_seq)
        #a = np.concatenate([one_hot_encoder[c] for c in short_seq], axis=1)
        if strand == '+':
            distal_seq = np.concatenate([self.one_hot_encoder[c] for c in short_seq], axis=1)
        else:
            #a = [one_hot_encoder_rc[c] for c in short_seq[::-1]]
            #a.reverse()
            distal_seq = np.concatenate([self.one_hot_encoder_rc[c] for c in short_seq[::-1]], axis=1)
        if distal_seq.shape[1] != self.seq_len:
            print('distal_seq.shape:', distal_seq.shape, chrom, start, stop)
            print('short_seq:', short_seq)

        # Handle distal bigWig data
        if len(self.bw_fh) > 0 and (not self.seq_only) and (not self.without_bw_distal):
            for bw in self.bw_fh:
                bw_values = np.nan_to_num(bw.values(chrom, start1, stop1, numpy=True))
                if(len(bw_values) < self.seq_len):
                    if start1 == 0:
                        bw_values = np.concatenate([(self.seq_len - len(bw_values))*[0], bw_values])
                    else:
                        bw_values = np.concatenate([bw_values, (self.seq_len - len(bw_values))*[0]])
                
                if strand == '-':
                    bw_values = np.flip(bw_values)
                
                distal_seq = np.concatenate((distal_seq, [bw_values]), axis=0).astype(np.float32)

        
        return self.y[idx], self.cont_X[idx], self.cat_X[idx], distal_seq
    
    def get_labels(self): 
        return np.squeeze(self.y)
    
    def _get_labels(self, dataset, idx):
        return dataset.__getitem__(idx)[1]

def prepare_dataset_h5(bed_regions, ref_genome, bw_paths, bw_files, bw_names, bw_radii, local_radius=5, local_order=1, distal_radius=50, distal_order=1, h5f_path='distal_data.h5', chunk_size=5000, seq_only=False, n_h5_files=1, without_bw_distal=False):
    """Prepare the datasets for given regions, using H5 file"""
 
    # Generate H5 file for distal data
    generate_h5fv2(bed_regions, h5f_path, ref_genome, distal_radius, distal_order, bw_paths, bw_files, chunk_size, n_h5_files, without_bw_distal)
    
    # Prepare local data
    data_local, seq_cols, categorical_features, output_feature = prepare_local_data(bed_regions, ref_genome, bw_files, bw_names, bw_radii, local_radius, local_order, seq_only)

    # If seq_only flag was set, bigWig files will be ignored
    if seq_only or without_bw_distal:
        n_channels = 4**distal_order
        print('NOTE: seq_only/without_bw_distal was set, so skip bigwig tracks for distal regions!')
    else:
        n_channels = 4**distal_order + len(bw_files)
    
    # Combine local data and distal into Dataset objects
    dataset = CombinedDatasetH5(data=data_local, seq_cols=seq_cols, cat_cols=categorical_features, output_col=output_feature, h5f_path=h5f_path, n_channels=n_channels)
    
    #return dataset, data_local, categorical_features
    return dataset


def prepare_dataset_np(bed_regions, ref_genome, bw_files, bw_names, bw_radii, local_radius=5, local_order=1, distal_radius=50, distal_order=1, seq_only=False, without_bw_distal=False):
    """Prepare the datasets for given regions, without an H5 file"""
    
    # Prepare local data
    data_local, seq_cols, categorical_features, output_feature = prepare_local_data(bed_regions, ref_genome, bw_files, bw_names, bw_radii, local_radius, local_order, seq_only)

    # If seq_only flag was set, bigWig files will be ignored
    if seq_only or without_bw_distal:
        n_channels = 4**distal_order
        print('NOTE: seq_only/without_bw_distal was set, so skip bigwig tracks for distal regions!')
    else:
        n_channels = 4**distal_order + len(bw_files)
    
    # Combine local data and distal into Dataset objects  
    dataset = CombinedDatasetNP(data=data_local, seq_cols=seq_cols, cat_cols=categorical_features, output_col=output_feature, ref_genome=ref_genome, bed_regions=bed_regions, distal_radius=distal_radius, n_channels=n_channels, bw_files=bw_files, seq_only=seq_only, without_bw_distal=without_bw_distal)
    #return dataset, data_local, categorical_features
    return dataset
