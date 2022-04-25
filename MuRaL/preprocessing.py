import os
import sys

from janggu.data import Bioseq, Cover
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

#from kipoiseq.extractors import FastaStringExtractor
#from kipoiseq import Interval
#from kipoiseq.transforms import ReorderedOneHot
#from kipoiseq.transforms.functional import pad
#from pyfaidx import Fasta

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

def get_h5f_path(bed_file, bw_names, distal_radius, distal_order):
    """Get the H5 file path name based on input data"""
    
    h5f_path = bed_file + '.distal_' + str(distal_radius)
    
    if distal_order > 1:
        h5f_path = h5f_path + '_' + str(distal_order)
        
    if len(bw_names) > 0:
        h5f_path = h5f_path + '.' + '.'.join(list(bw_names))
    
    h5f_path = h5f_path + '.h5'
    
    return h5f_path

def generate_h5f(bed_regions, h5f_path, ref_genome, distal_radius, distal_order, bw_files, h5_chunk_size, chunk_size=50000):
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
            hf.create_dataset(name='distal_X', shape=(0, n_channels, seq_len), compression="gzip", compression_opts=4, chunks=(h5_chunk_size,n_channels, seq_len), maxshape=(None,n_channels, seq_len)) 
            
            # Write data in chunks
            # chunk_size = 50000
            seq_records = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))
            for start in range(0, len(bed_regions), chunk_size):
                end = min(start+chunk_size, len(bed_regions))
                
                # Extract sequence from the genome, which is in one-hot encoding format
                seqs = get_digitalized_seq(seq_records, bed_regions.at(range(start, end)), distal_radius)
                #seqs = Bioseq.create_from_refgenome(name='distal', refgenome=ref_genome, roi=bed_regions.at(range(start, end)), flank=distal_radius, order=distal_order, verbose=True)
                
                #print('seqs.shape 1:', seqs.shape)
                # Get the correct shape (batch_size, channels, seq_len) for pytorch
                #seqs = np.array(seqs).squeeze((1,3)).transpose(0,2,1)
                
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

def get_digitalized_seq(seq_records, bed_regions, distal_radius):

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
    


def generate_h5fv2(bed_regions, h5f_path, ref_genome, distal_radius, distal_order, bw_paths, bw_files, chunk_size=50000, n_h5_files=1):
    """Generate the H5 file for storing distal data"""
    n_channels = 4**distal_order + len(bw_files)
    
    write_h5f = True
    if os.path.exists(h5f_path):
        try:
            with h5py.File(h5f_path, 'r') as hf:
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
        p = subprocess.Popen(args)
        p.wait()
            
    return None

def generate_h5f_single(bed_regions, h5f_path, ref_genome, distal_radius, distal_order, bw_files, h5_chunk_size, i_file, single_size):
    
    n_channels = 4**distal_order + len(bw_files)
    
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
        for start in range(i_file*single_size, np.min([(i_file+1)*single_size, len(bed_regions)]), chunk_size):
            end = min(start+chunk_size, len(bed_regions))

            # Extract sequence from the genome, which is in one-hot encoding format
            seqs = Bioseq.create_from_refgenome(name='distal', refgenome=ref_genome, roi=bed_regions.at(range(start, end)), flank=distal_radius, order=distal_order, verbose=True)

            #print('seqs.shape 1:', seqs.shape)
            # Get the correct shape (batch_size, channels, seq_len) for pytorch
            seqs = np.array(seqs).squeeze((1,3)).transpose(0,2,1)

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
    
    return h5f_path

def generate_h5f_single2(bed_regions, h5f_path, ref_genome, distal_radius, distal_order, bw_files, h5_chunk_size, i_file, single_size):
    
    #bed_regions = BedTool(bed_file)
    n_channels = 4**distal_order + len(bw_files)
    
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
            seqs = np.array(seqs).squeeze((1,3)).transpose(0,2,1)

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
    
    return h5f_path

def generate_h5f_singlev2(bed_regions, h5f_path, ref_genome, distal_radius, distal_order, bw_files, i_file, chunk_size):
    
    #bed_regions = BedTool(bed_file)
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
            seqs = get_digitalized_seq(seq_records, bed_regions.at(range(start, end)), distal_radius)
            
            # Handle distal bigWig data, return base-wise values
            if len(bw_files) > 0:
                bw_distal = Cover.create_from_bigwig(name='', bigwigfiles=bw_files, roi=bed_regions.at(range(start, end)), resolution=1, flank=distal_radius, verbose=True)
                #print('bw_distal.shape:', np.array(bw_distal).shape)

                #bw_distal should have the same seq len as that for distal_seq
                bw_distal = np.array(bw_distal).squeeze(axis=(1,3)).transpose(0,2,1)[:,:,:(distal_radius*2-distal_order+2)]

                # Concatenate the sequence data and the bigWig data
                #seqs = np.concatenate((seqs, bw_distal), axis=1)
                seqs = np.concatenate((seqs, bw_distal), axis=1).round(decimals=2)       
            # Write the numpy array into the H5 file
            hf['distal_X'].resize((hf['distal_X'].shape[0] + seqs.shape[0]), axis = 0)
            hf['distal_X'][-seqs.shape[0]:] = seqs
    
    return h5f_path

def generate_h5f_singlev3(bed_regions, h5f_path, ref_genome, distal_radius, distal_order, binsize, bw_files, i_file, chunk_size):
    
    #bed_regions = BedTool(bed_file)
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
            seqs = get_digitalized_seq(seq_records, bed_regions.at(range(start, end)), distal_radius)
            
            # Handle distal bigWig data, return base-wise values
            if len(bw_files) > 0:
                bw_distal = Cover.create_from_bigwig(name='', bigwigfiles=bw_files, roi=bed_regions.at(range(start, end)), resolution=1, flank=distal_radius, verbose=True)
                #print('bw_distal.shape:', np.array(bw_distal).shape)

                #bw_distal should have the same seq len as that for distal_seq
                bw_distal = np.array(bw_distal).squeeze(axis=(1,3)).transpose(0,2,1)[:,:,:(distal_radius*2-distal_order+2)]

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

def generate_h5f_mp(bed_regions, h5f_path, ref_genome, distal_radius, distal_order, bw_files, h5_chunk_size, n_h5_files):
    """Generate the H5 file for storing distal data"""
    n_channels = 4**distal_order + len(bw_files)
    
    write_h5f = True
    if os.path.exists(h5f_path):
        try:
            with h5py.File(h5f_path, 'r') as hf:
                bed_path = bed_regions.fn
                
                if os.lstat(bed_path).st_mtime < os.lstat(h5f_path).st_mtime and len(bed_regions) == hf["distal_X"].shape[0] and n_channels == hf["distal_X"].shape[1]:
                    write_h5f = False
        except OSError:
            print('Warning: re-genenerating the H5 file, because the file is empty or imcomplete:', h5f_path)
            
    # If the H5 file is unavailable or im complete, generate the file
    if write_h5f:            
        with h5py.File(h5f_path, 'w') as hf:
        
            with Pool(processes=5) as pool:

                single_size = int(np.ceil(len(bed_regions)/float(n_h5_files)))
                print('single_size:', single_size)

                #args = [(bed_regions, re.sub('h5$', str(i)+'.h5', h5f_path), ref_genome, distal_radius, distal_order, bw_files, h5_chunk_size, i, single_size) for i in range(n_h5_files)]
                #h5f_paths = pool.starmap(generate_h5f_single, args)
                args = [( BedTool(bed_regions.at(range(i*single_size,np.min([(i+1)*single_size, len(bed_regions)])))), re.sub('h5$', str(i)+'.h5', h5f_path), ref_genome, distal_radius, distal_order, bw_files, h5_chunk_size, i, single_size) for i in range(n_h5_files)]
                #h5f_paths = pool.starmap(generate_h5f_single2, args)
                h5f_paths = pool.starmap_async(generate_h5f_single2, args)
                

                #generate_h5f_single(bed_regions, h5f_path, ref_genome, distal_radius, distal_order, bw_files, h5_chunk_size, i_file, single_size)
                pool.close()
                pool.join()
                print('h5f_paths:', h5f_paths)

                for i in  range(n_h5_files):
                    #hf['data'+str(i)] = h5py.ExternalLink(h5f_paths[i], 'distal_X')
                    hf['data'+str(i)] = h5py.ExternalLink(re.sub('h5$', str(i)+'.h5', h5f_path), 'distal_X')
 

    return None

def get_seqs(ref_genome, bed_regions, distal_radius, start, size, end):
    
    end = min(start+size, end)
    
    seqs = Bioseq.create_from_refgenome(name='distal', refgenome=ref_genome, roi=bed_regions.at(range(start, end)), flank=distal_radius, order=1, cache=False, verbose=False)

    return np.array(seqs).squeeze((1,3)).transpose(0,2,1)




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


class CombinedDatasetH5v1(Dataset):
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
            self.h5f = h5py.File(self.h5f_path, 'r')
            
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

class CombinedDatasetKP(Dataset):
    """Combine local data and distal into Dataset, using Kipoi funcions"""
    def __init__(self, data, seq_cols, cat_cols, output_col, ref_genome, bed_regions, distal_radius, n_channels):
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
        #self.h5f_path = h5f_path
        self.distal_X = None
        self.n_channels = n_channels
        print('Number of channels to be used for distal data:', self.n_channels)
        
        self.fasta_extractor = FastaStringExtractor(fasta_file=ref_genome, use_strand=True, force_upper=True)
        self.bed_regions = bed_regions
        self.distal_radius = distal_radius
        self.seq_transform = ReorderedOneHot()
        self.fasta = Fasta(ref_genome)
        

    def __len__(self):
        """ Denote the total number of samples. """
        return self.n

    def __getitem__(self, idx):
        """ Generate one sample of data. """

        region = self.bed_regions[idx]
        
        #from kipoiseq.transforms.functional import pad
        #self.fasta['chr2L'].__len__()
        #pad(seq, 220, value="N", anchor="end")
        chr_len = self.fasta[region.chrom].__len__()
        start = np.max([0, region.start-self.distal_radius])
        end = np.min([chr_len, region.stop+self.distal_radius])
        
        interval = Interval(region.chrom, start=start, end=end, strand=region.strand)
        distal_seq = self.fasta_extractor.extract(interval)
        
        if region.start < self.distal_radius:
            distal_seq = pad(distal_seq, 2*self.distal_radius+1, value="N", anchor="end")
            print("Warning: padding at the front!")
        if region.stop+self.distal_radius > chr_len:
            distal_seq = pad(distal_seq, 2*self.distal_radius+1, value="N", anchor="start")
            
        distal_seq = self.seq_transform(distal_seq)      
        
        return self.y[idx], self.cont_X[idx], self.cat_X[idx], distal_seq.T.astype(np.float32)
    
    def get_labels(self): 
        return np.squeeze(self.y)
    
    def _get_labels(self, dataset, idx):
        return dataset.__getitem__(idx)[1]
    

class CombinedDatasetNP(Dataset):
    """Combine local data and distal into Dataset, using Kipoi funcions"""
    def __init__(self, data, seq_cols, cat_cols, output_col, ref_genome, bed_regions, distal_radius, n_channels):
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
        #    label_encoders[cat_col] = LabelEncoder()
        #    data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])
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
        #self.h5f_path = h5f_path
        self.distal_X = None
        self.n_channels = n_channels
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
    #seqs = np.array(seqs)
   
        
        return self.y[idx], self.cont_X[idx], self.cat_X[idx], distal_seq
    
    def get_labels(self): 
        return np.squeeze(self.y)
    
    def _get_labels(self, dataset, idx):
        return dataset.__getitem__(idx)[1]

class CombinedDatasetNP2(Dataset):
    """Combine local data and distal into Dataset, using Kipoi funcions"""
    def __init__(self, data, seq_cols, cat_cols, output_col, ref_genome, bed_regions, distal_radius, n_channels, bw_files, seq_only):
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
        #    label_encoders[cat_col] = LabelEncoder()
        #    data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])
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
    #seqs = np.array(seqs)
        #####################
        # Handle distal bigWig data
        if len(self.bw_fh) > 0 and self.seq_only == False:
            for bw in self.bw_fh:
                bw_values = np.nan_to_num(bw.values(chrom, start1, stop1, numpy=True))
                if(len(bw_values) < self.seq_len):
                    if start1 == 0:
                        bw_values = np.concatenate((self.seq_len - len(bw_values))*[0], bw_values)
                    else:
                        bw_values = np.concatenate(bw_values, (self.seq_len - len(bw_values))*[0])
                
                if strand == '-':
                    bw_values = np.flip(bw_values)
                
                distal_seq = np.concatenate((distal_seq, [bw_values]), axis=0).astype(np.float32)
                
        '''
        if len(self.bw_files) > 0 and self.seq_only == False:
            bw_distal = Cover.create_from_bigwig(name='', bigwigfiles=self.bw_files, roi=self.bed_regions.at(range(idx, idx+1)), resolution=1, flank=self.distal_radius)

            #bw_distal should have the same seq len as that for distal_seq
            bw_distal = np.array(bw_distal).squeeze(axis=(0,1,3)).transpose(1,0)[:,:(self.distal_radius*2+1)]

            # Concatenate the sequence data and the bigWig data
            distal_seq = np.concatenate((distal_seq, bw_distal), axis=0).astype(np.float32)
        '''
        
        #############

        
        return self.y[idx], self.cont_X[idx], self.cat_X[idx], distal_seq
    
    def get_labels(self): 
        return np.squeeze(self.y)
    
    def _get_labels(self, dataset, idx):
        return dataset.__getitem__(idx)[1]
    
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
def prepare_dataset_h5v2(bed_regions, ref_genome, bw_paths, bw_files, bw_names, local_radius=5, local_order=1, distal_radius=50, distal_order=1, h5f_path='distal_data.h5', chunk_size=5000, seq_only=False, n_h5_files=1):
    """Prepare the datasets for given regions, using H5 file"""
 
    # Generate H5 file for distal data
    generate_h5fv2(bed_regions, h5f_path, ref_genome, distal_radius, distal_order, bw_paths, bw_files, chunk_size, n_h5_files)
    
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

def prepare_dataset_kp(bed_regions, ref_genome, bw_files, bw_names, local_radius=5, local_order=1, distal_radius=50, distal_order=1,seq_only=False):
    """Prepare the datasets for given regions, using H5 file"""
    
    # Prepare local data
    data_local, seq_cols, categorical_features, output_feature = prepare_local_data(bed_regions, ref_genome, bw_files, bw_names, local_radius, local_order, seq_only)

    # If seq_only flag was set, bigWig files will be ignored
    if seq_only:
        n_channels = 4**distal_order
        print('NOTE: seq_only flag was set, so will not use any bigWig track!')
    else:
        n_channels = 4**distal_order + len(bw_files)
    
    # Combine local data and distal into Dataset objects
    dataset = CombinedDatasetKP(data=data_local, seq_cols=seq_cols, cat_cols=categorical_features, output_col=output_feature, ref_genome=ref_genome, bed_regions=bed_regions, distal_radius=distal_radius, n_channels=n_channels)
    
    #return dataset, data_local, categorical_features
    return dataset

def prepare_dataset_np(bed_regions, ref_genome, bw_files, bw_names, local_radius=5, local_order=1, distal_radius=50, distal_order=1,seq_only=False):
    """Prepare the datasets for given regions, using H5 file"""
    
    # Prepare local data
    data_local, seq_cols, categorical_features, output_feature = prepare_local_data(bed_regions, ref_genome, bw_files, bw_names, local_radius, local_order, seq_only)

    # If seq_only flag was set, bigWig files will be ignored
    if seq_only:
        n_channels = 4**distal_order
        print('NOTE: seq_only flag was set, so will not use any bigWig track!')
    else:
        n_channels = 4**distal_order + len(bw_files)
    
    # Combine local data and distal into Dataset objects
    if seq_only:
        dataset = CombinedDatasetNP(data=data_local, seq_cols=seq_cols, cat_cols=categorical_features, output_col=output_feature, ref_genome=ref_genome, bed_regions=bed_regions, distal_radius=distal_radius, n_channels=n_channels)
    else:
        dataset = CombinedDatasetNP2(data=data_local, seq_cols=seq_cols, cat_cols=categorical_features, output_col=output_feature, ref_genome=ref_genome, bed_regions=bed_regions, distal_radius=distal_radius, n_channels=n_channels, bw_files=bw_files, seq_only=seq_only)
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
    distal_seq = np.array(distal_seq).squeeze((1,3)).transpose(0,2,1)
    
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

from Bio import SeqIO
from Bio.Seq import Seq

def generate_h5f2(bed_regions, h5f_path, ref_genome, distal_radius, distal_order, bw_files, h5_chunk_size, chunk_size=5000):
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
            
            print('Generating HDF5 file using numpy/pandas:', h5f_path)
            sys.stdout.flush()
            
            # Total seq len
            seq_len =  distal_radius*2+1-(distal_order-1)
            
            # Create distal_X dataset
            # Note, the default dtype for create_dataset is numpy.float32
            hf.create_dataset(name='distal_X', shape=(0, n_channels, seq_len), compression="gzip", compression_opts=4, chunks=(h5_chunk_size,n_channels, seq_len), maxshape=(None,n_channels, seq_len))
            
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
            
            records = SeqIO.to_dict(SeqIO.parse(open(ref_genome), 'fasta'))
            
            fbed = open(bed_regions.fn)
            # Write data in chunks
            # chunk_size = 50000
            for start in range(0, len(bed_regions), chunk_size):
                end = min(start+chunk_size, len(bed_regions))
                #print("extracting sequence ...", start, end)
                sys.stdout.flush()
                
                seqs = []
                
                #start_time = time.time()
                for idx in range(start, end):
                    
                    #region = bed_regions[idx]
                    line = fbed.readline()
                    chrom, start1, stop1, seqid, score, strand = line.split()
                    
                    long_seq_record = records[chrom]
                    
                    long_seq = str(long_seq_record.seq)
                    long_seq_len = len(long_seq)

                    start1 = np.max([int(start1)-distal_radius, 0])
                    stop1 = np.min([int(stop1)+distal_radius, long_seq_len])
                    short_seq = long_seq[start1:stop1].upper()
                    
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
                        seqs.append(np.concatenate([one_hot_encoder[c] for c in short_seq], axis=1))
                    else:
                        #a = [one_hot_encoder_rc[c] for c in short_seq[::-1]]
                        #a.reverse()
                        seqs.append(np.concatenate([one_hot_encoder_rc[c] for c in short_seq[::-1]], axis=1))
                seqs = np.array(seqs)
                
                #print('seqs.shape:', seqs.shape)
                #print('Time used for writing get_seqs: %s seconds' % (time.time() - start_time))
                #sys.stdout.flush()
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
            
            fbed.close()
            
    return None