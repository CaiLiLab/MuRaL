
from Bio import SeqIO
import sys


base = sys.argv[1]

genome_file = sys.argv[2]

output_file = sys.argv[3]

fout = open(output_file, 'w') 

for record in list(SeqIO.parse(genome_file, 'fasta')):
    chrom = record.id

    for pos in range(len(record)):
        if record.seq[pos].upper() == base.upper():
            print(chrom+'\t'+str(pos)+'\t'+str(pos+1), file=fout)
