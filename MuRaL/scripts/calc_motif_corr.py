#!/usr/bin/env python
"""
Calculate k-mer mutation rate correlations between observed and predicted values
"""
import argparse
import gzip
import sys
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from Bio import SeqIO
from Bio.Seq import reverse_complement
from scipy.stats import pearsonr
import numpy as np

from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import data.preprocessing 
from typing import Dict, Any

def parse_arguments():
    """Parse and validate command line parameters"""
    parser = argparse.ArgumentParser(
        description='Calculate k-mer mutation rate correlations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--pred_file', required=True, type=Path,
                       help='predicted file')
    parser.add_argument('--ref_genome', required=True, type=Path,
                       help='Reference genome FASTA file')
    parser.add_argument('--out_prefix', default='result', type=Path,
                       help='Output filename prefix')
    parser.add_argument('--motif_length', type=int, default=3,
                       help='k-mer length (positive odd integer)')
    parser.add_argument('--n_class', type=int, default=4,
                       help='snv is 4, indel is 8.')

    args = parser.parse_args()

    return args

# ----------------------
# Core Data Structures
# ----------------------
class KmerMutSaver:
    def __init__(self, n_class, merge_reverse=True):
        self.n_class = n_class
        self.kmer_mut_obs = {}
        self.kmer_mut_pred = {}
        self.check_and_init_kmer = self._check_and_init_kmer_merge_reverse \
            if merge_reverse else self._check_and_init_kmer
    
    def _check_and_init_kmer(self, kmer, kmer_saver):
        if kmer not in kmer_saver:
            kmer_saver[kmer] = np.zeros(self.n_class)
        return kmer

    def _check_and_init_kmer_merge_reverse(self, kmer, kmer_saver):
        if kmer in kmer_saver:
            return kmer
        # check reverse complement is in the dict
        rev_kmer = reverse_complement(kmer)
        if rev_kmer in kmer_saver:
            return rev_kmer
        # kmer and reverse not in dict, initialize kmer
        kmer_saver[kmer] = np.zeros(self.n_class)
        return kmer
    
    def add_obs(self, kmer, mut_type):
        kmer = self.check_and_init_kmer(kmer, self.kmer_mut_obs)
        self.kmer_mut_obs[kmer][mut_type] += 1
    
    def add_pred(self, kmer, probs):
        kmer = self.check_and_init_kmer(kmer, self.kmer_mut_pred)
        for i in range(self.n_class):
            self.kmer_mut_pred[kmer][i] += probs[i]

# ----------------------
# Argument Parsing
# ----------------------
def validate_motif_length(value: int) -> int:
    """Validate k-mer length is a positive odd integer"""
    if value <= 1 or value % 2 != 1:
        raise ValueError("--motif_length must be a positive odd integer >1")
    return value

# ----------------------
# Genome Processing
# ----------------------
def load_chromosome_sequence(genome_file: Path, chrom: str) -> str:
    """Load chromosome sequence from FASTA"""
    # if not chrom.startswith('chr'):
    #     chrom = f"chr{chrom}"
    for record in SeqIO.parse(genome_file, "fasta"):
        if record.id == chrom:
            return str(record.seq).upper()
    raise ValueError(f"Chromosome {chrom} not found in {genome_file}")

# ----------------------
# K-mer Processing
# ----------------------
def process_kmer_seq(strand: str, sequence: str, mut_type: int, probs: list, data_saver) -> None:
    """Process mutation site statistics for a single k-mer"""
    if not is_valid_sequence(sequence):
        return
    # get kmer sequence
    canonical_kmer = get_canonical_kmer(strand, sequence)
    # Update k-mer counts
    data_saver.add_obs(canonical_kmer, mut_type)
    data_saver.add_pred(canonical_kmer, probs)
    
def is_valid_sequence(sequence: str) -> bool:
    """Validate k-mer contains only standard bases"""
    return all(base in {'A', 'C', 'G', 'T'} for base in sequence)

def get_canonical_kmer(strand: str, sequence: str) -> str:
    """Get strand-corrected k-mer sequence"""
    if strand == '+':
        return sequence
    elif strand == '-':
        return reverse_complement(sequence)
    else:
        raise ValueError(f"Invalid strand: {strand}")

# ----------------------
# Data Analysis
# ----------------------
def calculate_mutation_rates(kmer_saver: Any) -> pd.DataFrame:
    """Calculate observed and predicted mutation rates for each k-mer.
    
    Returns:
        DataFrame with columns:
        - type: k-mer sequence
        - avg_obs_rate{1..n}: Average observed rates per class
        - avg_pred_rate{1..n}: Average predicted rates per class
        - number_of_mut{1..n}: Raw mutation counts per class (int)
        - number_of_all: Total counts (int)
    """
    # Generate column names using list comprehension
    mutation_classes = range(1, kmer_saver.n_class)
    rate_cols = [f"avg_obs_rate{i}" for i in mutation_classes]
    pred_cols = [f"avg_pred_rate{i}" for i in mutation_classes]
    count_cols = [f"number_of_mut{i}" for i in mutation_classes]
    
    # Process data using dictionary comprehension
    results = {
        kmer: np.concatenate([
            obs_counts[1:] / obs_counts.sum(),  # Normalized observed rates
            kmer_saver.kmer_mut_pred[kmer][1:] / obs_counts.sum(),  # Normalized predicted rates
            obs_counts[1:],  # Raw mutation counts
            [obs_counts.sum()]  # Total counts
        ])
        for kmer, obs_counts in kmer_saver.kmer_mut_obs.items()
    }
    
    # Create DataFrame with proper typing
    results = (
        pd.DataFrame.from_dict(results, orient='index', columns=rate_cols + pred_cols + count_cols + ['number_of_all'])
        .rename_axis('type')
        .reset_index()
    )
    
    # Convert count columns to integers
    results[count_cols] = results[count_cols].astype(int)
    results['number_of_all'] = results['number_of_all'].astype(int)
    
    return results

def calculate_correlations(df: pd.DataFrame, n_class) -> Dict[int, Tuple[float, float]]:
    """Calculate Pearson correlations for each mutation subtype"""
    return {
        subtype: pearsonr(df[f'avg_obs_rate{subtype}'], df[f'avg_pred_rate{subtype}'])
        for subtype in range(1, n_class)
    }

# ----------------------
# Output Handling
# ----------------------
def write_correlation(args, correlations: Dict[int, Tuple[float, float]]) -> None:
    """Write results to output files"""
    # Write correlation results
    with open(f"{args.out_prefix}.{args.motif_length}-motif.corr.txt", 'w') as f:
        for subtype, (corr, pval) in correlations.items():
            f.write(f"{args.motif_length}-moitf\t{subtype}\t{corr:.5f}\t{pval:.10e}\n")

# ----------------------
# Main Workflow
# ----------------------
def run_motif_corr_calc(args, model_type:'str') -> None:
    """Main processing pipeline"""

    n_class = args.n_class
    kmer_mut_saver = KmerMutSaver(n_class)
    current_chrom = None
    chromosome_seq = ""
    # radius = args.motif_length // 2
    # all motifs one site chould be located
    if model_type == 'indel':
        # for indel, mut in gap, gap should have one base at least in each side.
        motifs_coords = [(i, args.motif_length - i) for i in range(1, args.motif_length)]
    else:
        # for snv, mut in base, extend motif_legth - 1.
        motifs_coords = [(i, args.motif_length - 1 - i) for i in range(args.motif_length)]
    
    # Open prediction file
    opener = gzip.open if args.pred_file.endswith('.gz') else open
    # save kmer count of obs and predict
    with opener(args.pred_file, 'rt') as obs_file:
        # header process
        header = next(obs_file)
        if not header.startswith('chrom'):
            raise ValueError(f"Invalid file header: {header.strip()}, header should be continue with 'chrom'")
            # Parse line
        # check file is consistent with parameter
        header = header.strip().split('\t')
        if len(header) != n_class + 5:
            raise ValueError(
                f"Column count mismatch. Expected {n_class + 5} columns, "
                f"got {len(header)} in line: {header}")

        for line in obs_file:
            line = line.strip().split('\t')
            chrom, start, end, strand, mut = line[:5] 
            if model_type == 'indel':
                strand = args.strand
            probs = np.asarray(line[5:], dtype='float') # prob0 to prob n

            start, end, mut = int(start), int(end), int(mut)
            
            # Load chromosome sequence when needed
            if chrom != current_chrom:
                chromosome_seq = load_chromosome_sequence(args.ref_genome, chrom)
                current_chrom = chrom
            
            # Extract k-mer sequence, diff between indel and snv
            for radius_left, radius_right in motifs_coords:
                seq_start, seq_end = data.preprocessing.extend_interval(start, end, radius_left, radius_right, model_type)

                kmer_seq = chromosome_seq[seq_start:seq_end]

                # skip boundary kmer
                if len(kmer_seq) != args.motif_length:
                    continue
            
                # Process site statistics
                process_kmer_seq(
                    strand="+",# undistinguished seq in ref strand
                    sequence=kmer_seq,
                    mut_type=mut,
                    probs=probs,
                    data_saver=kmer_mut_saver
                )

    # Calculate and save results
    results_df = calculate_mutation_rates(kmer_mut_saver)
    results_df.to_csv(
        f"{args.out_prefix}.{args.motif_length}-motif.mut_rates.tsv", sep='\t', index=False)
    corr_results = calculate_correlations(results_df, n_class)
    write_correlation(args, corr_results)

# if __name__ == '__main__':
#     args = parse_arguments()
#     run_motif_corr_calc(args, model_type='snv')