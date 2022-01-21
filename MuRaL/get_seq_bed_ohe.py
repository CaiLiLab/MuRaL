from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
import numpy as np

import time
import datetime


'''
# read names and postions from bed file
positions = defaultdict(list)
with open('/public/home/licai/DNMML/analysis/training/drosophila/data/test_100k.bed') as f:
    for line in f:
        name, start, stop, seqid, score, strand = line.split()
        positions[name].append((int(start), int(stop)))
'''

# parse faste file and turn into dictionary
records = SeqIO.to_dict(SeqIO.parse(open('/public2/home/dengsy/other_species/fly/dm3.fa'), 'fasta'))

one_hot_encoder = {'A':np.array([[1,0,0,0]]).T,
                   'C':np.array([[0,1,0,0]]).T,
                   'G':np.array([[0,0,1,0]]).T,
                   'T':np.array([[0,0,0,1]]).T,
                   'N':np.array([[0.25,0.25,0.25,0.25]]).T}

# search for short sequences
#short_seq_records = []

start_time = time.time()
print('Start time:', datetime.datetime.now())

distal_radius = 100

seq_len = 2*distal_radius + 1

ohe_out = []
with open('/public/home/licai/DNMML/analysis/training/drosophila/data/test_500k.bed') as f:
    for line in f:
        name, start, stop, seqid, score, strand = line.split()
        long_seq_record = records[name]
        long_seq = str(long_seq_record.seq)
        long_seq_len = len(long_seq)

        start = np.max([int(start)-distal_radius, 0])
        stop = np.min([int(stop)+distal_radius, long_seq_len])
        
        short_seq = long_seq[start:stop].upper()
        
        if(len(short_seq) < seq_len):
            print('warning:', name, start, stop, long_seq_len)
            if start == 0:
                short_seq = (seq_len - len(short_seq))*'N' + short_seq
                print(short_seq)
            else:
                short_seq = short_seq + (seq_len - len(short_seq))*'N'
                print(short_seq)
            
        a = np.concatenate([one_hot_encoder[c] for c in short_seq], axis=1)
        ohe_out.append(a)
        
ohe_out = np.array(ohe_out)
print('ohe_out.shape:', ohe_out.shape)

print('Total time used: %s seconds' % (time.time() - start_time))

