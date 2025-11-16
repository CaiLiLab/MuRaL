from typing import Callable, Any
import argparse
import textwrap

def add_common_eval_parser(
    eval_parser: argparse.ArgumentParser,
) -> tuple:

    eval_parser.set_defaults(func='evaluate')

    eval_optional = eval_parser._action_groups.pop()
    eval_required = eval_parser.add_argument_group('commoned arguments')
    eval_kmer_parser = eval_parser.add_argument_group('kmer-related arguments')
    eval_regional_parser = eval_parser.add_argument_group('regional-related arguments')

    # Add required arguments
    eval_required.add_argument('--pred_file', required=True, type=str, help='Predicted file')

    eval_required.add_argument('--ref_genome', required=True, type=str, help='Reference genome FASTA file')

    eval_required.add_argument('--out_prefix', default='result', type=str, help='Output filename prefix')

    eval_required.add_argument('--kmer_only', default=False, action='store_true', help=textwrap.dedent("""
        Only evaluate kmer correlation for indel. Default: False.
    """).strip())

    eval_required.add_argument('--regional_only', default=False, action='store_true', help=textwrap.dedent("""
        Only evaluate regional correlation for indel. Default: False.
    """).strip())

    eval_required.add_argument('--motif_only', default=False, action='store_true', 
        help=argparse.SUPPRESS)

    # Add kmer-related arguments, PASS

    # Add regional-related arguments
    eval_regional_parser.add_argument('--window_size', type=int, default=100000, help=textwrap.dedent("""
        Window size (bp) for calculating regional rates. Default: 100000.
    """).strip())

    eval_regional_parser.add_argument('--ratio_cutoff', type=float, default=0.2, help=textwrap.dedent("""
        Ratio cutoff for filtering windows with few valid sites. 
        Default: 0.2, meaning that windows with fewer than 0.2*median(numbers of sites in surveyed windows) will be discarded.
    """).strip())

    eval_parser._action_groups.append(eval_optional)

    return eval_parser, eval_required, eval_kmer_parser, eval_regional_parser, eval_optional




def add_indel_eval_parser(subparsers: argparse._SubParsersAction) -> argparse._SubParsersAction:
    """
    Register INDEL specify parser for evaluation parser.

    This function is part of the logical flow:
        Parser Setup -> Common Parser regist -> Specify Parser regist
    """
    eval_parser = subparsers.add_parser(
      'evaluate',
      help='Evaluation for mural-snv model',
      formatter_class=argparse.RawDescriptionHelpFormatter,
      description=textwrap.dedent("""
      The command is used for calculating evaluation metrics of the model,
      including k-mer and regional correlation analyses.

      K-mer Correlation Analysis:
      ---------------------------
      Inputs required:
      - Reference genome (FASTA format)
      - Prediction file (TSV format)
      - K-mer length parameter

      Note for evaluation:
      The 5th column of the prediction TSV must contain observed mutation data
      (e.g., rare variants). These observed mutations are used to calculate
      observed mutation rates. You can modify this column to evaluate model
      performance against different observed mutation datasets.

      Output files:
       The outputs include a file ('\*-mer.mut\_rates.tsv') storing predicted and 
       observed k-mer rates of all possible mutation subtypes, and a file ('\*-mer.corr.txt')
       storing the k-mer correlations (Pearson's r and p-value) of three mutation
       types in a specific order (e.g., for A/T sites, prob1, prob2 and prob3 are
       for A>C, A>G and A>T, respectively).

       # example of '*-mer.mut_rates.tsv'
       type   avg_obs_rate1   avg_obs_rate2   avg_obs_rate3   avg_pred_prob1  avg_pred_prob2  avg_pred_prob3  number_of_mut1  number_of_mut2  number_of_mut3  number_of_all
       TAG    0.006806776385512125    0.010141979926438501    0.012039461380213204    0.012744358544122413    0.01817057941563919     0.021860978496512425    3494    5206    6180    513312
       TAA    0.007517292690907348    0.011278023120833133    0.01318808653952362     0.013600087566977897    0.019697007577734515    0.024266536859123104    7214    10823   12656   959654
       AAA    0.0068964404639771226   0.010705555691654661    0.009617493130148654    0.012599749576515839    0.020442895433664586    0.01646869397956817     11542   17917   16096   1673617
 
       # example of '*-mer.corr.txt'
       3-mer  prob1   0.9569216831654604      6.585788162834682e-09 # r and p for prob1
       3-mer  prob2   0.9326211281771537      1.4129640985193586e-07 # r and p for prob2
       3-mer  prob3   0.947146892265788       2.6848989196451608e-08 # r and p for prob3
    
      Command line examples
      ---------------------
       mural_indel evaluate --pred_file testing.ckpt9.fdiri.tsv.gz --ref_genome data/seq.fa --kmer_length 2 --kmer_only --out_prefix test
       mural_indel evaluate --pred_file testing.ckpt9.fdiri.tsv.gz --ref_genome data/seq.fa --kmer_length 4 --kmer_only --out_prefix test
       mural_indel evaluate --pred_file testing.ckpt9.fdiri.tsv.gz --ref_genome data/seq.fa --kmer_length 6 --kmer_only --out_prefix test

      Regional Correlation Analysis:
      -----------------------------
      Inputs required:
      - Prediction file (TSV format)
      - Window size parameter

      Similar to k-mer analysis, the 5th column should contain the specific set
      of observed mutations used for calculating regional mutation rates.

      Output data:
       There are multiple output files. The files storing regional rates 
       ('\*.regional\_rates.tsv') have seven columns: chromosome name, the end
       position of the window, number of valid sites in the window, number of 
       observed mutations in the window, average observed mutation rate, average 
       predicted mutation rate in the window and the 'used_or_deprecated' label. 
       The windows labeled 'deprecated' are not used in correlation analysis due 
       to too few valid sites. The regional correlation (Pearson's r and p-value)
       of the considered mutation type is given in the '\*.corr.txt'.

       # example of '*.regional_rates.tsv'
       chrom  end     sites_count     mut_type_total  mut_type_avg    avg_pred        used_or_deprecated
       chr3   100000  61492   576     0.009367072139465296    0.020374342255903233    used
       chr3   200000  60680   531     0.008750823994726434    0.02025859070533955     used
       chr3   300000  59005   499     0.00845691043131938     0.01882644280993153     used
 
       # example of '*.corr.txt'
       100Kb  prob3   0.4999  6.040983e-16 

      Command line examples
      ---------------------
       mural_indel evaluate --pred_file testing.ckpt9.fdiri.tsv.gz --window 100000 --model prob2 --regional_only --out_prefix test_region_corr

      """)
      )
    
    # Setup the common evaluation parser
    eval_parser, eval_required, eval_kmer_parser, eval_regional_parser, eval_optional = add_common_eval_parser(eval_parser)

    eval_required.add_argument('--n_class', type=int, default=8, help='Number of classes for indel (default: 8)')

    eval_kmer_parser.add_argument('--kmer_length', type=int, default=2, help=textwrap.dedent("""
        Length of k-mer used for evaluation (typically 2, 4, or 6). Default: 2.
    """).strip())

    eval_kmer_parser.add_argument('--motif_length', type=int, default=6, 
            help=argparse.SUPPRESS)

    eval_kmer_parser.add_argument(
    '--strand', type=str, default='pos', choices=['pos', 'neg', 'both'],  
    help=textwrap.dedent("""
        Read kmer from which strand:
        '+' : Forward strand only
        '-' : Reverse strand only
        'both' : Both strands (forward and reverse)
        Default: '+'
    """).strip()
    )

    return eval_parser


def add_snv_eval_parser(subparsers: argparse._SubParsersAction) -> argparse._SubParsersAction:
    """
    Register SNV specify parser for evaluation parser.

    This function is part of the logical flow:
        Parser Setup -> Common Parser regist -> Specify Parser regist
    """
    # Setup the common evaluation parser
    eval_parser = subparsers.add_parser(
      'evaluate',
      help='Evaluation for mural-snv model',
      formatter_class=argparse.RawDescriptionHelpFormatter,
      description=textwrap.dedent("""
      The command is used for calculating evaluation metrics of the model,
      including k-mer and regional correlation analyses.

      K-mer Correlation Analysis:
      ---------------------------
      Inputs required:
      - Reference genome (FASTA format)
      - Prediction file (TSV format)
      - K-mer length parameter

      Note for evaluation:
      The 5th column of the prediction TSV must contain observed mutation data
      (e.g., rare variants). These observed mutations are used to calculate
      observed mutation rates. You can modify this column to evaluate model
      performance against different observed mutation datasets.

      Output data:
       The outputs include a file ('\*-mer.mut\_rates.tsv') storing predicted and 
       observed k-mer rates of all possible mutation subtypes, and a file ('\*-mer.corr.txt')
       storing the k-mer correlations (Pearson's r and p-value) of three mutation
       types in a specific order (e.g., for A/T sites, prob1, prob2 and prob3 are
       for A>C, A>G and A>T, respectively).

       # example of '*-mer.mut_rates.tsv'
       type   avg_obs_rate1   avg_obs_rate2   avg_obs_rate3   avg_pred_prob1  avg_pred_prob2  avg_pred_prob3  number_of_mut1  number_of_mut2  number_of_mut3  number_of_all
       TAG    0.006806776385512125    0.010141979926438501    0.012039461380213204    0.012744358544122413    0.01817057941563919     0.021860978496512425    3494    5206    6180    513312
       TAA    0.007517292690907348    0.011278023120833133    0.01318808653952362     0.013600087566977897    0.019697007577734515    0.024266536859123104    7214    10823   12656   959654
       AAA    0.0068964404639771226   0.010705555691654661    0.009617493130148654    0.012599749576515839    0.020442895433664586    0.01646869397956817     11542   17917   16096   1673617
 
       # example of '*-mer.corr.txt'
       3-mer  prob1   0.9569216831654604      6.585788162834682e-09 # r and p for prob1
       3-mer  prob2   0.9326211281771537      1.4129640985193586e-07 # r and p for prob2
       3-mer  prob3   0.947146892265788       2.6848989196451608e-08 # r and p for prob3
    
      Command line examples
      ---------------------
       mural_snv evaluate --pred_file testing.ckpt4.fdiri.tsv.gz --ref_genome data/seq.fa --kmer_length 3 --kmer_only --out_prefix test
       mural_snv evaluate --pred_file testing.ckpt4.fdiri.tsv.gz --ref_genome data/seq.fa --kmer_length 5 --kmer_only --out_prefix test
       mural_snv evaluate --pred_file testing.ckpt4.fdiri.tsv.gz --ref_genome data/seq.fa --kmer_length 7 --kmer_only --out_prefix test

      Regional Correlation Analysis:
      -----------------------------
      Inputs required:
      - Prediction file (TSV format)
      - Window size parameter

      Similar to k-mer analysis, the 5th column should contain the specific set
      of observed mutations used for calculating regional mutation rates.

      Output data:
       There are multiple output files. The files storing regional rates 
       ('\*.regional\_rates.tsv') have seven columns: chromosome name, the end
       position of the window, number of valid sites in the window, number of 
       observed mutations in the window, average observed mutation rate, average 
       predicted mutation rate in the window and the 'used_or_deprecated' label. 
       The windows labeled 'deprecated' are not used in correlation analysis due 
       to too few valid sites. The regional correlation (Pearson's r and p-value)
       of the considered mutation type is given in the '\*.corr.txt'.

       # example of '*.regional_rates.tsv'
       chrom  end     sites_count     mut_type_total  mut_type_avg    avg_pred        used_or_deprecated
       chr3   100000  61492   576     0.009367072139465296    0.020374342255903233    used
       chr3   200000  60680   531     0.008750823994726434    0.02025859070533955     used
       chr3   300000  59005   499     0.00845691043131938     0.01882644280993153     used
 
       # example of '*.corr.txt'
       100Kb  prob3   0.4999  6.040983e-16 

      Command line examples
      ---------------------
       mural_snv evaluate --pred_file testing.ckpt4.fdiri.tsv.gz --window 100000 --model prob2 --regional_only --out_prefix test_region_corr

      """)
      )

    eval_parser, eval_required, eval_kmer_parser, eval_regional_parser, eval_optional = add_common_eval_parser(eval_parser)

    # Add snv-specific arguments
    eval_required.add_argument('--n_class', type=int, default=4, help='Number of classes for SNV (default: 4)')

    eval_kmer_parser.add_argument('--kmer_length', type=int, default=3, help=textwrap.dedent("""
        Length of k-mer used for evaluation (typically 3, 5, or 7). Default: 3.
    """).strip())

    eval_kmer_parser.add_argument('--motif_length', type=int, default=3, help=textwrap.dedent("""
        Length of k-mer used for evaluation (typically 3, 5, or 7). Default: 3.
    """).strip())

    return eval_parser