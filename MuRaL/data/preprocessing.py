import os
import sys
import multiprocessing

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

def bed_reader(bed_regions, central_bp):
    """
    Read a given BED region file and generate a new list of regions,
    and split the regions into two lists based on their strand information.

    Args:
    - bed_regions: BedTool object or bed file path representing the BED region file to read.
    - central_bp: Integer representing the length of the new regions to generate used encoding.

    Yields:
    Generator object that yields a list and a string:
    - The list contains regions used encoding.
    - The string represents the regions strand direction.
    """
    if isinstance(bed_regions, str):
        bed_regions = BedTool(bed_regions)
    else:
        if not isinstance(bed_regions, BedTool):
            print(f"Error: bed_regions should be <str> or <Bedtools>, but input is {bed_regions.__class__}!")
            sys.exit()
    init = 0
    for region in bed_regions:
        if not init:
            init += 1
            chrom, start0 = str(region.chrom), region.start
            end0 = start0 + central_bp 
            pos_strand_region = []
            neg_strand_region = []
            
        chrom2, start, stop, strand = str(region.chrom), region.start, region.stop, region.strand
            
        if chrom2 != chrom:
            if pos_strand_region:
                yield pos_strand_region, '+'
                pos_strand_region = []
            if neg_strand_region:
                yield neg_strand_region, '-'
                neg_strand_region = []    
            chrom = chrom2
            start0 = 1
            end0 = 1 + central_bp 
            
        if strand == '+':
            pos_strand_region.append(region)

        else:
            neg_strand_region.append(region)
            
        if start > end0:
            if pos_strand_region:
                yield pos_strand_region, '+'
                pos_strand_region = []

            if neg_strand_region:
                yield neg_strand_region, '-'
                neg_strand_region = []

            start0 = end0
            end0 += central_bp
            
    if pos_strand_region:
        yield pos_strand_region, '+'
    if neg_strand_region:
        yield neg_strand_region, '-'

def get_position_info(test_bed, central_radius):
    """
    Get validation position information

    Returns:
    pd.DataFrame: A DataFrame containing the chromosome, start, end, and strand 
                  information with columns ['chrom', 'start', 'end', 'strand'].
    """
    bed_generator = bed_reader(test_bed, central_radius)
    info = []
    for batch, stand in bed_generator:
        info.extend([[nucleo.chrom, nucleo.start, nucleo.end, stand] for nucleo in batch])
    info = pd.DataFrame(info, columns=['chrom', 'start', 'end', 'stand'])
    return info

def get_position_info_by_trainset(test_bed, central_radius):
    """
    Get validation position information from the test bed file.

    This function reads a BED file and processes its entries in batches.
    It returns a MultiIndex DataFrame where the first level index is the 
    batch number and the second level index is the sample number within each batch.

    Returns:
    pd.DataFrame: A DataFrame containing the chromosome, start, end, and strand 
                  information with a MultiIndex (batch_num, sample_num).
    """
    bed_generator = bed_reader(test_bed, central_radius)
    info = []

    for batch_num, (batch, stand) in enumerate(bed_generator):
        batch_info = [[batch_num, i, nucleo.chrom, nucleo.start, nucleo.end, stand] for i, nucleo in enumerate(batch)]
        info.extend(batch_info)
    
    info_df = pd.DataFrame(info, columns=['batch_num', 'sample_num', 'chrom', 'start', 'end', 'stand'])
    info_df.set_index(['batch_num', 'sample_num'], inplace=True)
    
    return info_df

def get_bw_for_bed(bw_files, bed_regions, radius):# to-do : start , length, adpatation for indel

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
#########################################################################
#                          gene HD5
#  use HD5 for saving distal Encoding(One-Hot)
#########################################################################
def get_h5f_path(bed_file, bw_names, central_radius, distal_radius, distal_order, without_bw_distal):
    """Get the H5 file path name based on input data"""
    
    h5f_path = bed_file + '.distal_' + str(distal_radius) + '.segment_' + str(central_radius) + '_segshare'
    
    if distal_order > 1:
        h5f_path = h5f_path + '_' + str(distal_order)
        
    if len(bw_names) > 0 and (not without_bw_distal):
        h5f_path = h5f_path + '.' + '.'.join(list(bw_names))
    
    h5f_path = h5f_path + '.h5'
    
    return h5f_path

def change_h5f_path(h5f_path, bed_file, bw_names, central_radius, distal_radius, distal_order, without_bw_distal):
    name = get_h5f_path(bed_file, bw_names, central_radius, distal_radius, distal_order, without_bw_distal)
    name = name.split('/')[-1]
    h5f_path_new = os.path.join(h5f_path, name) 

    if not os.path.isdir(h5f_path):
        print(f"Warming : input h5f_path not dir, h5f path generate to {h5f_path_new} !")
    return h5f_path_new

#def generate_h5f(bed_regions, h5f_path, ref_genome, central_radius, distal_radius, distal_order, bw_files, h5_chunk_size=1, chunk_size=50000, without_bw_distal=True, model_type='snv'):
def generate_h5f(bed_regions, h5f_path, ref_genome, central_radius, distal_radius, distal_order, bw_files, model_type='snv'):
    """Generate the H5 file for storing distal data"""

    print('Generating HDF5 file:', h5f_path)
    sys.stdout.flush()
    # recode overlap realtion ship of sample in each segment
    seq_records = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))
    seq_list, batch_shape = get_distal_seqs_by_region(bed_regions, seq_records, distal_radius, central_radius, model_type)
    
    with h5py.File(h5f_path, 'w') as hf:
        for idx in range(len(batch_shape)):
            # Creat group And Wirte in HD5 for each segment
            create_group_and_store(hf, idx, seq_list[idx], seq_records, distal_radius)
    
    return None

def create_group_and_store(hf, idx, seqs_info, seq_records, distal_radius):
    """Create a group for each segment and store encoded data"""
    group = hf.create_group(f'segment_{idx}')
    stand = seqs_info[0][3]
    group.attrs['stand'] = stand
    # Split segment into multi sample share sub-segment 
    iter_segment_info = get_segment_info(seqs_info, seq_records, distal_radius)
    
    store_encodings(group, iter_segment_info)

def store_encodings(group, iter_segment_info):
    """Store encoding sample share sub-segment into the HDF5 group"""
    for sample_num, (long_seq, start, stop, chrom, strand, radius, index, end) in enumerate(iter_segment_info):
        encoding = segment_ohe_encoder(long_seq, start, stop, strand, radius, end)
        sample_dset = group.create_dataset(f'sample_{sample_num}', data=encoding, compression="gzip", compression_opts=4)
        sample_dset.attrs['index'] = index # recode sample index in sub-segment

def get_segment_info(seqs, seq_records, radius):
    init = True
    for start0, stop0, chrom, strand, index, end in seqs:
        if init:
            init = False
            long_seq = str(seq_records[chrom].seq)
            c = chrom
        assert chrom == c
        yield long_seq, start0, stop0, chrom, strand, radius, index, end

def segment_ohe_encoder(long_seq, start, stop, strand, radius, end=False):

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
        
    #imput 
    short_seq = ['', '','']
    if start < 0:
        left_impute = 0 - start 
        start = 0
        short_seq[0] = left_impute * 'N'
        
    if end:
        long_seq_len = len(long_seq)
        right_impute = stop - long_seq_len
        short_seq[2] = right_impute * 'N'

    short_seq[1] = long_seq[start:stop].upper()

    short_seq = ''.join(short_seq)

    
   # return short_seq
    if strand == '+':
        distal_ecoding = np.concatenate([one_hot_encoder[c] for c in short_seq], axis=1)
        
    else:
        distal_ecoding = np.concatenate([one_hot_encoder_rc[c] for c in short_seq[::-1]], axis=1)
        
    return distal_ecoding

def generate_h5fv2(bed_regions, h5f_path, ref_genome, central_radius, distal_radius, distal_order, bw_paths, bw_files, chunk_size=50000, n_h5_files=1, without_bw_distal=False, model_type='snv'):
    """Generate the H5 file for storing distal data"""
    write_h5f = True
    if os.path.exists(h5f_path):
        try:
            with h5py.File(h5f_path, 'r', swmr=True) as hf:
                bed_path = bed_regions.fn
                h5_sample_size = check_h5f_sample_size(hf)
                try:
                    if os.lstat(bed_path).st_mtime < os.lstat(h5f_path).st_mtime \
                    and len(bed_regions) == h5_sample_size:
                        write_h5f = False
                except KeyError:
                    print('Warning: re-genenerating the H5 file, because the file is empty or imcomplete:', h5f_path)
                                       
        except OSError:
            print('Warning: re-genenerating the H5 file, because the file is empty or imcomplete:', h5f_path)

            
    # If the H5 file is unavailable or incomplete, generate the file
    if write_h5f:            
        p = multiprocessing.Process(target=generate_h5f,\
                                    args=(bed_regions,h5f_path,ref_genome,central_radius, \
                                          distal_radius,distal_order,bw_files, model_type)\
                                   )       
    
        return p
    return 0

def check_h5f_sample_size(h5f):
    h5_sample_size = 0
    for key in h5f.keys():
        segment = h5f[key]
        h5_sample_size += sum([len(segment[key].attrs['index']) for key in segment.keys()])
    return h5_sample_size

#########################################################################
#                           Local Embeding 
#########################################################################
def get_local_header(local_radius: int, local_order: int, model_type: str) -> list:
    """
    Args:
        model_type: 模型类型 ('snv' or 'indel')
    Returns:
        col_names, used to dataframe
    Examples:
        >>> get_local_header(3, 1, 'snv')
        ['us3', 'us2', 'us1', 'mid', 'ds1', 'ds2', 'ds3']
    """
    if local_order == 1:
        # 生成一阶特征的列名
        upstream = [f'us{local_radius - i}' for i in range(local_radius)]
        downstream = [f'ds{i+1}' for i in range(local_radius)]
        
        if model_type == 'snv':
            return upstream + ['mid'] + downstream
        return upstream + downstream
    
    # calc local order
    cat_n = calc_cat_n(local_radius, local_order, model_type)
    
    return [f'cat{i+1}' for i in range(cat_n)]

def calc_cat_n(local_radius: int, local_order: int, model_type: str) -> int:
    offset = local_order - 1
    cat_n = (local_radius * 2 + (1 if model_type == 'snv' else 0)) - offset
    return cat_n

def prepare_local_data(bed_regions, ref_genome, bw_files, bw_names, bw_radii, central_radius, local_radius, local_order, seq_only, model_type='snv'):
    """Prepare local data for given regions
    local_digitalized_seqs_by_region
        |---> get_seqs_to_digitalized ---> local_encoding_seqs--->seq_digit_encoder
    
    """

    """  
        Args:
            bed_regions: <Bedtools> 
            ref_genome:  <SeqRecord> ref genome
    """
    # Read the seq data
    local_seq_cat,y = local_digitalized_seqs_by_region(bed_regions, ref_genome, central_radius, local_radius, local_order=1, model_type=model_type)    
    #local_seq_cat = pd.concat([pd.DataFrame(x,columns = seq_cols) for x in local_seq_cat],keys=range(len(local_seq_cat)))
    # keys correspond to the segment number 
    local_seq_cat = pd.concat(local_seq_cat, keys=range(len(local_seq_cat)), copy=False)

    #seq_cols = ['us'+str(local_radius - i) for i in range(local_radius)] + ['mid'] + ['ds'+str(i+1) for i in range(local_radius)]
    seq_cols = get_local_header(local_radius, local_order=1, model_type=model_type)
    #local_seq_cat = pd.DataFrame(local_seq_cat, columns = seq_cols)
    assert list(local_seq_cat.columns) == seq_cols, f"Error: local_seq_cat columns is {local_seq_cat.columns}, but need {seq_cols} !"
    if local_order > 1:
        local_seq_cat2,y = local_digitalized_seqs_by_region(bed_regions, ref_genome, central_radius, local_radius, local_order=local_order, model_type=model_type)
        
        # NOTE: replace k-mers with 'N' with a large number; the padding numbers at the two ends of the chromosomes are also large numbers   
        #local_seq_cat2 = np.where(np.logical_and(local_seq_cat2>=0, local_seq_cat2<=4**local_order), local_seq_cat2, 4**local_order)
        # Names of the categorical variables
        #cat_n = local_radius*2 +1 - (local_order-1)
        #categorical_features  = ['cat'+str(i+1) for i in range(cat_n)]
        categorical_features  = get_local_header(local_radius, local_order=local_order, model_type=model_type)
        local_seq_cat2 = pd.concat(local_seq_cat2,keys=range(len(local_seq_cat2)))
        local_seq_cat2 = pd.concat([local_seq_cat, local_seq_cat2], axis=1)
    else:
        categorical_features = seq_cols
    
    print('local_seq_cat2 shape and columns:', local_seq_cat2.shape, local_seq_cat2.columns)
    print('categorical_features:', categorical_features)
    
    # The 'score' field in the BED file stores the label/class information
    y = pd.concat(y,keys=range(len(y)))
    output_feature = 'mut_type'
    
    # Add feature data in bigWig files
    if len(bw_files) > 0 and seq_only == False:
        # Use the mean value of the region of 2*radius+1 bp around the focal site
        bw_data = get_mean_bw_for_bed(bw_files, bw_names, bw_radii, bed_regions)
        data_local = pd.concat([local_seq_cat2, bw_data, y], axis=1)
    else:
        data_local = pd.concat([local_seq_cat2, y], axis=1)

    return data_local, seq_cols, categorical_features, output_feature

def local_digitalized_seqs_by_region(bed_regions, seq_records, central_bp, local_radius, local_order=1, model_type='snv'):

    if 'items' not in dir(seq_records):
        _ = type(seq_records)
        sys.exit(f'seq_records need be dict, but input is {_} !')
    
    #cat_n = local_radius*2 +1 - (local_order-1) 
    outlier_process = preocess_local_seq_outlier(local_order, local_radius, model_type=model_type)

    seq_cols = get_local_header(local_radius, local_order=local_order, model_type=model_type)
    
    bed_generator = bed_reader(bed_regions, central_bp)
    digit_dataset = []
    y = []
    init = False
    for batch,stand in bed_generator:
        if not init:
            chrom = batch[0].chrom
            long_seq = str(seq_records[chrom].seq)
            init = True
        else:
            if chrom != batch[0].chrom:
                chrom = batch[0].chrom
                long_seq = str(seq_records[chrom].seq)
        
        #batch_local_encoding = np.empty((len(batch),cat_n), dtype=np.int64)
        batch_local_encoding = np.empty((len(batch), len(seq_cols)), dtype=np.int64)
        seqs = get_seqs_to_digitalized(long_seq, batch, local_radius, stand, model_type=model_type)
        digit_seqs = local_encoding_seqs(long_seq, seqs, local_radius, batch_local_encoding, local_order=local_order, model_type=model_type)
        digit_seqs = outlier_process(digit_seqs)
        digit_dataset.append(pd.DataFrame(digit_seqs,columns=seq_cols))
        
        label = get_label(batch)
        y.append(pd.DataFrame(label.reshape((-1,1)),columns = ['mut_type']))

        
    #digit_dataset = np.concatenate(digit_dataset)
    return digit_dataset,y 

def process_local_seq_snv(local_seq_cat,local_radius):

    # check 
    if np.unique(local_seq_cat[:,local_radius], axis=0).shape[0] != 1:
        print('ERROR: The positions in input BED file have different bases (A/T and C/G mixed)! The ref_genome or input BED file could be wrong.', file=sys.stderr)
        sys.exit()
    # preprocess outlier
    return np.where(local_seq_cat>=0, local_seq_cat, 0)

def process_local_seq_indel(local_seq_cat,local_radius):
   # preprocess outlier
    return np.where(local_seq_cat>=0, local_seq_cat, 0)

def preocess_local_seq_outlier(local_order,local_radius, model_type='snv'):
    """
    Generate a function to process local sequence outliers based on the given local order and radius.

    Returns:
    A function that processes local sequence outliers.

    Raises:
    - SystemExit: If local_order is not greater than 0.
    """
    # diffrent check condition for indel and snv
    if local_order == 1:
        processor_map = {
        'snv': process_local_seq_snv,
        'indel': process_local_seq_indel
        }
        if model_type not in processor_map:
            sys.exit(f"Error: model_type {model_type} not supported!")
        return partial(processor_map[model_type], local_radius=local_radius)

    # same kmer encoder 
    elif local_order > 1:
        v = 4**local_order
        def process_local_seq(local_seq_cat,v=v):
            """
            Process the local sequence for local order greater than 1.
            """
            return np.where(np.logical_and(local_seq_cat >= 0, local_seq_cat <= v), local_seq_cat, v)
        return process_local_seq
    else:
        sys.exit("local_oreder need larger than 0!")

def get_expanded_region(start, stop, radius, model_type='snv'):
    """
    Calculate expanded genomic coordinates by radius 
    
    Args:
        start: Region start position (0-based inclusive)
        stop: Region end position (0-based inclusive)
        radius: Number of bases to expand around the region
        model_type: Either 'snv'  or 'indel' 
    
    Returns:
        Tuple of (expanded_start, expanded_stop)
    
    Coordinate Expansion Diagrams:
    
    SNV Model (expand according to the sampled site):
    segment: |-----|← radius →[sampled_site]← radius →|-----|
    index:    start-radius  --[    start   ]--    start+radius+1
    
    Indel Model (expand according to the sampled gap):
    segment: |----|← radius-1 →[start [   -----   ] stop]← radius-1 →|----|
    index:   start-(radius-1)  [start [sampled_gap] stop]        stop+radius
    
    Examples:
        >>> # SNV model, sampled site starts at 100.
        >>> get_expanded_region(100, 101, 10, 'snv')
        (90, 111)  # [90,130] contains original [100,120]
        
        >>> # Indel model, sampled gap between 100 and 120.
        >>> get_expanded_region(100, 120, 10, 'indel')
        (91, 130)  # [91,130] contains original [100,120]
    """
    if model_type == 'snv':
        start1 = start - radius
        stop1 = stop + radius
    if model_type == 'indel':
        start1 = start - radius + 1
        stop1 = stop + radius
    return start1, stop1


def get_seqs_to_digitalized(long_seq, regions, radius, seq_strand, model_type='snv'):
    """
    Check relationship of regions by radius and save the information for encoding.

    Yields:
    A tuple containing start position of encoding, stop position of encoding, chromosome,
             strand, index list, and a boolean indicating if imputation is needed
    - The index list used to find samples from encoding segment.

    """
    #seqs = []
    end=True
    init = False
    index = [0]
    for region in regions:
        chrom, start, stop, strand = str(region.chrom), region.start, region.stop, region.strand
        
        assert seq_strand == strand
        
        # init 
        if not init:
            init = True
            chrom_init = chrom
            start0, stop0 = get_expanded_region(start, stop, radius, model_type=model_type)
            leng_seq = len(long_seq)
            continue
        
        start1, stop1 = get_expanded_region(start, stop, radius, model_type=model_type)
        #start1 = int(start) - radius
        #stop1 = int(stop) + radius
        
        if start1 > stop0:
            # one-hot encoding
            impute = False
            yield (start0, stop0, chrom, strand, index, impute)
            start0, stop0 = start1, stop1
            index = [0]
        else:
            stop0 = stop1
            index.append(start1-start0)
    if stop0 > leng_seq:
        impute = True
    else:
        impute = False
    yield (start0, stop0, chrom, strand, index, impute)

def local_encoding_seqs(long_seq, seqs, radius, batch_local_encoding, local_order, model_type):
    if '__iter__' not in dir(seqs):
        sys.exit("Error : input seqs is not <generator>!")

    if not isinstance(batch_local_encoding, np.ndarray):
        sys.exit("Error: one-hot Encoding Need provided an array to save batch infomation!")
        
    batch_index = 0
    for start0, stop0, chrom, strand, index, end in seqs:
        sub_batch_num = len(index)
        sub_batch = seq_digit_encoder(long_seq, start0, stop0, chrom, strand, radius,index, local_order, end, model_type)
        batch_local_encoding[batch_index:batch_index+sub_batch_num] = sub_batch
        batch_index += sub_batch_num
    return batch_local_encoding 

def calc_window_size(radius, local_order, model_type):
    """same as local categorical number compute"""
    return calc_cat_n(radius, local_order, model_type)

def seq_digit_encoder(long_seq, start, stop, chrom, strand, radius, index, local_order, end=False, model_type='snv'): 
    """
    Convert genomic sequence to numerical representation with position handling
    
    Args:
        long_seq: Reference genome sequence (string)
        start: Segment start position (0-based)
        stop: Segment end position 
        chrom: Chromosome name (for reference)
        strand: Strand direction ('+' or '-')
        radius: Window radius around central position
        index: List of positions to extract windows
        local_order: K-mer order 
        end: Flag indicating if this is the end of sequence (for right padding)
    
    Returns:
        Numeric array of shape (n_windows, window_size) representing encoded sequences
    """

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
    
    #impute
    short_seq = ['', '','']
    if start < 0:
        #left_imput = 0 - start + 1
        left_impute = 0 - start
        start = 0
        short_seq[0] = left_impute * 'N'
        
    if end:
        long_seq_len = len(long_seq)
        right_impute = stop - long_seq_len
        short_seq[2] = right_impute * 'N'

    short_seq[1] = long_seq[start:stop].upper()
    short_seq = ''.join(short_seq)
  #  return short_seq
    if strand == '+':
        digit_seq = np.array([digit_encoder[c] for c in short_seq])
    else:
        digit_seq = np.array([digit_encoder_rc[c] for c in short_seq[::-1]])

    if local_order > 1:
        seq_len = len(digit_seq)
        new_seq = []
        for i in range(seq_len - local_order +1):
            kmer = digit_seq[i:i+local_order]
            if min(kmer) < 0:
                new_seq.append(-1)
            else:
                digit = sum([kmer[d]*4**(local_order-d-1) for d in range(local_order)])
                new_seq.append(digit)
        digit_seq = np.array(new_seq)

    # Compute window size for extracting distal sequences from segments 
    window_size = calc_window_size(radius, local_order, model_type)

    if strand == '+':
        digit_seq = np.asarray([digit_seq[start1:start1 + window_size] for start1 in index], dtype=np.int64)         
    else:
        digit_seq = np.asarray([digit_seq[-start1-window_size:-start1] if start1 else digit_seq[-start1-window_size:] for start1 in index], dtype=np.int64)
    
    digit_seq = np.where(np.logical_and(digit_seq>=0, digit_seq<=4**local_order), digit_seq, 4**local_order)
    return digit_seq

def get_mean_bw_for_bed(bw_files, bw_names, bw_radii, bed_regions, model_type='snv'):

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
                start, stop = get_expanded_region(start, stop, bw_radii[j], model_type=model_type)
                start1 = max([start, 0])
                stop1 = min([stop, bw.chroms(chrom)])

                bw_data[i,j] = np.nan_to_num(bw.values(chrom, start1, stop1, numpy=True)).mean()
                

        bw_data = pd.DataFrame(bw_data, columns=bw_names)
    
    return bw_data 

def get_label(bed_regions):
    y = np.array([float(loc.score) for loc in bed_regions])
    return y

def seq_ohe_encoder(long_seq, start, stop, chrom, strand, radius, index, end=False, model_type='snv'):

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
        
    #imput 
    short_seq = ['', '','']
    if start < 0:
        left_impute = 0 - start 
        start = 0
        short_seq[0] = left_impute * 'N'
        
    if end:
        long_seq_len = len(long_seq)
        right_impute = stop - long_seq_len
        short_seq[2] = right_impute * 'N'

    short_seq[1] = long_seq[start:stop].upper()

    short_seq = ''.join(short_seq)
   
    # return short_seq
    window_size = calc_window_size(radius, local_order=1, model_type=model_type)
    if strand == '+':
        distal_seq = np.concatenate([one_hot_encoder[c] for c in short_seq], axis=1)
        distal_seq = [distal_seq[:,start1:start1 + window_size] for start1 in index]
        #distal_seq = np.expand_dims(distal_seq, 0)
    else:
        distal_seq = np.concatenate([one_hot_encoder_rc[c] for c in short_seq[::-1]], axis=1)
        distal_seq = [distal_seq[:,-start1-window_size:-start1] if start1 else distal_seq[:,-start1-window_size:] for start1 in index]    
        #distal_seq = np.expand_dims(distal_seq, 0)
    return distal_seq

#########################################################################
#                          Construct Dataset Without HDF5 
# 
# Note: When sample redundancy is high, I/O in preprocessing can become a bottleneck
#       in model training. This method computes distal encoding dirctly from the reference  
#       genome, reducing the disk IO. 
#
# Suggestion: For human data, if the distal region is greater than 8k,
#             it is recommended to use this method.
#########################################################################
def prepare_dataset_np(bed_regions, ref_genome, bw_files, bw_names, bw_radii,central_radius=30000, local_radius=5, local_order=1, distal_radius=50, distal_order=1, seq_only=False, without_bw_distal=False, model_type='snv'):
    """Prepare the datasets for given regions, without an H5 file"""
    """  
        Args:
            bed_regions: <Bedtools> 
            ref_genome:  <str> path of ref genome
    """
    # Prepare local data
    ref_genome = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))
    data_local, seq_cols, categorical_features, output_feature = prepare_local_data(bed_regions, ref_genome, bw_files, bw_names, bw_radii, central_radius, local_radius, local_order, seq_only, model_type=model_type)

    # If seq_only flag was set, bigWig files will be ignored
    if seq_only or without_bw_distal:
        n_channels = 4**distal_order
        print('NOTE: seq_only/without_bw_distal was set, so skip bigwig tracks for distal regions!')
    else:
        n_channels = 4**distal_order + len(bw_files)
    
    # Combine local data and distal into Dataset objects  
    dataset = CombinedDatasetNP(data=data_local, seq_cols=seq_cols, cat_cols=categorical_features, output_col=output_feature, ref_genome=ref_genome, bed_regions=bed_regions, central_radius=central_radius, distal_radius=distal_radius, n_channels=n_channels, bw_files=bw_files, seq_only=seq_only, without_bw_distal=without_bw_distal, model_type=model_type)
    return dataset

class CombinedDatasetNP(Dataset):
    """Combine local data and distal into Dataset, using NumPy funcions"""
    def __init__(self, data, seq_cols, cat_cols, output_col, \
                 ref_genome, bed_regions, central_radius, distal_radius, \
                 n_channels, bw_files, seq_only, without_bw_distal, model_type='snv'):
        """  
        Args:
            data: DataFrame containing local seq data and categorical data
            seq_cols: names of local seq columns
            cat_cols: names of categorical columns used for training
            output_col: name of the label column
            n_channels: number of columns (channels) in distal data to be extracted
        """
        # check input
        if not isinstance(bed_regions, BedTool):
            print(f"Error: bed_regions should be  <Bedtools>, but input is {bed_regions.__class__}!")
            sys.exit()

        if not isinstance(ref_genome, dict):
            print(f"Error : ref_genome should be <dict>, but input is {ref_genome.__class__}!")
            sys.exit()

        # Store the local seq data and label for later use
        self.model_type = model_type
        self.data_local = data[seq_cols+[output_col]]
        
        # Sample size
        #self.n = data.shape[0]# sample size
        self.n = data.index[-1][0] + 1# batch number
        # Output column
        if output_col:
             self.y = data[output_col].astype(np.float32)
            #self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            sys.exit(f"Error: {output_col}")
            #self.y = np.zeros((self.n, 1))
        
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
            self.cat_X = data[cat_cols]
            #self.cat_X = data[cat_cols].astype(np.int64).values
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
        #self.seq_len = distal_radius*2+ central_radius - (distal_order-1) 

        self.central_radius = central_radius
        self.bed_regions = bed_regions
        self.records = ref_genome

        self.distal_info = False
    def __len__(self):
        """ Denote the total number of samples. """
        return self.n

    def __getitem__(self, index):
        """ Generate one batch of data. """
        assert index < self.n
        seqs = iter(self.seqs_list[index])
        batch_distal = distal_encoding_by_region(seqs, self.batch_shape[index], self.distal_radius,self.records, model_type=self.model_type)
        #assert np.sum(self.y.loc[index] == label) == len(label)
        
        return self.y.loc[index].values.reshape(-1, 1), self.cont_X[index], self.cat_X.loc[index].values, batch_distal

    def get_distal_encoding_infomation(self):
        self.seqs_list,self.batch_shape = get_distal_seqs_by_region(self.bed_regions, self.records, self.distal_radius, self.central_radius, self.model_type)
        self.distal_info = True
        
    def get_labels(self): 
        return np.squeeze(self.y)
    
    def _get_labels(self, dataset, idx):
        return dataset.__getitem__(idx)[1]
    
def get_distal_seqs_by_region(bed_regions, seq_records, radius, central_bp, model_type):
    seqs_list = []
    batch_shape = []
    bed_generator = bed_reader(bed_regions, central_bp)
    init = False
    for batch,stand in bed_generator:
        if not init:
            chrom = batch[0].chrom
            long_seq = str(seq_records[chrom].seq)
            init = True
        else:
            if chrom != batch[0].chrom:
                chrom = batch[0].chrom
                long_seq = str(seq_records[chrom].seq)

        # generate seq info used to encoding
        seqs = get_seqs_to_digitalized(long_seq, batch, radius, stand, model_type)
        seqs_list.append([i for i in seqs])
        batch_shape.append(len(batch))

    return seqs_list,batch_shape

def distal_encoding_by_region(seqs, batch_shape, radius,seq_records, model_type='snv'):
    if '__iter__' not in dir(seqs):
        sys.exit("Error : input seqs is not <generator>!")
        
    # Create an array to store batch after ohe encoding
    window_size = calc_window_size(radius, local_order=1, model_type=model_type)
    batch_ohe_encoding = np.empty((batch_shape,4,window_size), dtype='float32')
    batch_index = 0
    init = True
    for start0, stop0, chrom, strand, index, end in seqs:
        if init:
            init = False
            long_seq = str(seq_records[chrom].seq)
            c = chrom
        assert chrom == c
        sub_batch_num = len(index)
        sub_batch = seq_ohe_encoder(long_seq, start0, stop0, chrom, strand, radius, index, end, model_type)

        batch_ohe_encoding[batch_index:batch_index+sub_batch_num] = sub_batch
        batch_index += sub_batch_num
    
    return batch_ohe_encoding



#########################################################################
#                          Construct Dataset With HDF5 
# 
# Note: When sample redundancy is low, computation in preprocessing become 
#       a bottleneck im model training. This method used HD5 file to save 
#       non-redundancy distal encoding, enabling the reuse of encoding. 
#
# Suggestion: For human data, if the distal region is less than 4k,
#             it is recommended to use this method.
# 
# Dependency: h5py==3.10.0; h5py==2.10.0 can not run in multi process in this code.
#########################################################################
def prepare_dataset_h5(bed_regions, ref_genome, bw_paths, bw_files, bw_names, bw_radii,central_radius, local_radius=5, local_order=1, distal_radius=50, distal_order=1, h5f_path=None, chunk_size=5000, seq_only=True, n_h5_files=1, without_bw_distal=True, model_type='snv'):
    """Prepare the datasets for given regions, using H5 file"""
 
    # get h5f_path 
    bed_file = bed_regions.fn
    if not h5f_path:
        h5f_path = get_h5f_path(bed_file, bw_names, central_radius, distal_radius, distal_order, without_bw_distal)
    else:
        h5f_path = change_h5f_path(h5f_path, bed_file, bw_names, central_radius, distal_radius, distal_order, without_bw_distal)
    # Generate H5 file for distal data
    bed_file = bed_regions.fn
    process = generate_h5fv2(bed_regions, h5f_path, ref_genome, central_radius, distal_radius, distal_order, bw_paths, bw_files, chunk_size, n_h5_files, without_bw_distal, model_type=model_type)
    if process:
        process.start()
    
    # Prepare local data
    ref_genome = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))
    start_time = time.time()
    data_local, seq_cols, categorical_features, output_feature = prepare_local_data(bed_regions, ref_genome, bw_files, bw_names, bw_radii, central_radius, local_radius, local_order, seq_only, model_type=model_type)
    print(f"local preprocess used time: {time.time() -start_time}")
    # If seq_only flag was set, bigWig files will be ignored
    if seq_only or without_bw_distal:
        n_channels = 4**distal_order
        print('NOTE: seq_only/without_bw_distal was set, so skip bigwig tracks for distal regions!')
    else:
        n_channels = 4**distal_order + len(bw_files)
    
    if process:
        process.join()
    
    # Combine local data and distal into Dataset objects
    dataset = CombinedDatasetH5(data=data_local, seq_cols=seq_cols, cat_cols=categorical_features, output_col=output_feature, h5f_path=h5f_path, distal_radius=distal_radius, n_channels=n_channels, model_type=model_type)
    
    #return dataset, data_local, categorical_features
    return dataset

class CombinedDatasetH5(Dataset):
    """Combine local data and distal into Dataset, with H5"""
    def __init__(self, data, seq_cols, cat_cols, output_col, h5f_path, distal_radius, n_channels, model_type='snv'):
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
        self.model_type = model_type
        self.data_local = data[seq_cols+[output_col]]
        
        self.n = data.index[-1][0] + 1
        
        # Output column
        if output_col:
            self.y = data[output_col].astype(np.float32)
        else:
            sys.exit(f"Error: {output_col}")
        
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
            self.cat_X = data[cat_cols]
        else:
            print("Error: no categorical data, something is wrong!", file=sys.stderr)
            sys.exit()
        
        # For distal data
        self.h5f = h5py.File(h5f_path, 'r', swmr=True)
        self.distal_radius = distal_radius

    def __len__(self):
        """ Denote the total number of samples. """
        return self.n

    def __getitem__(self, index):
        """ Generate one sample of data. """
        y_values = self.y.loc[index].values.reshape(-1, 1)
        cont_X = self.cont_X[index]
        cat_X_values = self.cat_X.loc[index].values
        distal_encoding = self._read_distal(index)
        
        return y_values, cont_X, cat_X_values, distal_encoding

    def _read_distal(self, index):
        segment_encoding = self.h5f[f"segment_{index}"] # get segment
        stand = segment_encoding.attrs['stand']
        
        batch_ohe_encoding = []
        for sample_num in range(len(segment_encoding)):
            sample_dset = segment_encoding[f"sample_{sample_num}"]
            distal_seq = sample_dset[:]
            sub_index = sample_dset.attrs['index']
            sub_batch = get_sample_from_segment(distal_seq, sub_index, stand,self.distal_radius, self.model_type)
            batch_ohe_encoding.append(sub_batch)
        
        batch_ohe_encoding = np.concatenate(batch_ohe_encoding)

        return batch_ohe_encoding
    
def get_sample_from_segment(distal_seq, sub_index, stand,radius, model_type='snv'):
    # Validate model type
    if model_type not in ('snv', 'indel'):
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'snv' or 'indel'")

    # Calculate window size based on model type
    window_size = 2 * radius + (1 if model_type == 'snv' else 0)

    if stand == '+':
        distal_seq = [distal_seq[:,start1:start1 + window_size] for start1 in sub_index]
        #distal_seq = np.expand_dims(distal_seq, 0)
    else:
        distal_seq = [distal_seq[:,-start1-window_size:-start1] if start1 else distal_seq[:,-start1-window_size:] for start1 in sub_index]    
    return distal_seq

#########################################################################
#                          Construct DataLoader 
#########################################################################
def generate_data_batches(segmentLoader_train, batch_segment, batch_size, shuffle=True, sample_workers=0):
    iter_seg_share_dataset = get_seg_share_dataset(segmentLoader_train, batch_segment)
    # init
    seg_dataset = next(iter_seg_share_dataset)
    
    merge = False
    drop_last = False
    # gene batch to train
    while True:
        dataloader = DataLoader(seg_dataset, batch_size, shuffle=shuffle, num_workers=sample_workers, pin_memory=False)
        for y, cont_x, cat_x, distal_x in dataloader:
            # if sample less than batch number, merge to next segment
            if y.shape[0] < batch_size:
                merge = True
                break

            yield y, cont_x, cat_x, distal_x
        # check end and read next segment
        try:
            seg_dataset = next(iter_seg_share_dataset)
        except StopIteration:
            # merge=True, indicate last batch not output. 
            if merge and not drop_last:
                yield y, cont_x, cat_x, distal_x
            return
                
        # if merge, merge to next segment
        if merge:
            merge = False
            seg_dataset.merge(y, cont_x, cat_x, distal_x)

def get_seg_share_dataset(segmentLoader, batch_segment):
    count = 0
    segment_saver = []
    for segment in segmentLoader:
        segment_saver.append(segment)
        count += 1
    
        if count >= batch_segment:
            segment_dataset = Create_DatasetSegment(segment_saver)
            yield segment_dataset
            count = 0
            segment_saver = []

    if segment_saver:
        segment_dataset = Create_DatasetSegment(segment_saver)
        yield segment_dataset

class Create_DatasetSegment(Dataset):
    """     """
    def __init__(self, data_batch):
        """  
        Args:
          
        """
        
        self.y = torch.cat([batch[0].squeeze(0) for batch in data_batch])
        self.cat_X = torch.cat([batch[2].squeeze(0) for batch in data_batch])
        self.distal_x = torch.cat([batch[3].squeeze(0) for batch in data_batch])
        
        self.n = self.y.shape[0]
        self.cont_X = np.zeros((self.n, 1))  
    def __len__(self):
        """ Denote the total number of samples. """
        return self.n

    def __getitem__(self, index):
        """ Generate one batch of data. """
        
        return self.y[index], self.cont_X[index], self.cat_X[index], self.distal_x[index]
    
    def merge(self, y, cont_x, cat_x, distal_x):
        """ Add one batch of data"""
        self.y = torch.cat([y, self.y])
        self.cat_X = torch.cat([cat_x, self.cat_X])
        self.distal_x = torch.cat([distal_x, self.distal_x])

        self.n = self.y.shape[0]
        self.cont_X = np.zeros((self.n, 1)) 