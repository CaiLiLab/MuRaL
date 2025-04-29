from typing import Callable, Any
import argparse
import textwrap

def setup_eval_parser(subparsers: argparse._SubParsersAction, parser_name: str, help_message: str) -> tuple:
    """
    Set evaluation parsers.

    This function is part of the logical flow:
        Parser Setup -> Common Parser regist -> Specify Parser regist
    """
    eval_parser = subparsers.add_parser(
      parser_name,
      help=help_message,
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

      Regional Correlation Analysis:
      -----------------------------
      Inputs required:
      - Prediction file (TSV format)
      - Window size parameter

      Similar to k-mer analysis, the 5th column should contain the specific set
      of observed mutations used for calculating regional mutation rates.
      """)
      )
    # Pop the optional arguments group
    eval_optional = eval_parser._action_groups.pop()

    # Define argument groups
    eval_required = eval_parser.add_argument_group('Required arguments')
    eval_kmer_parser = eval_parser.add_argument_group('kmer-related arguments')
    eval_regional_parser = eval_parser.add_argument_group('regional-related arguments')

    return eval_parser, eval_required, eval_kmer_parser, eval_regional_parser, eval_optional


def regist_common_eval_parser(
  eval_required: argparse.ArgumentParser,
  eval_kmer_parser: argparse.ArgumentParser,
  eval_regional_parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """
    Register common arguments for all evaluation parsers.

    This function is part of the logical flow:
        Parser Setup -> Common Parser regist -> Specify Parser regist
    """

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

    # Add kmer-related arguments, PASS

    # Add regional-related arguments
    eval_regional_parser.add_argument('--window_size', type=int, default=100000, help=textwrap.dedent("""
        Window size (bp) for calculating regional rates. Default: 100000.
    """).strip())

    eval_regional_parser.add_argument('--ratio_cutoff', type=float, default=0.2, help=textwrap.dedent("""
        Ratio cutoff for filtering windows with few valid sites. 
        Default: 0.2, meaning that windows with fewer than 0.2*median(numbers of sites in surveyed windows) will be discarded.
    """).strip())




def add_indel_eval_parser(subparsers: argparse._SubParsersAction) -> argparse._SubParsersAction:
    """
    Register INDEL specify parser for evaluation parser.

    This function is part of the logical flow:
        Parser Setup -> Common Parser regist -> Specify Parser regist
    """
    # Setup the common evaluation parser
    eval_parser, eval_required, eval_kmer_parser, eval_regional_parser, eval_optional = setup_eval_parser(
        subparsers=subparsers,
        parser_name='evaluate',
        help_message='Evaluate mural-indel model'
    )

    # Register the common parser
    regist_common_eval_parser(
        eval_required=eval_required,
        eval_kmer_parser=eval_kmer_parser,
        eval_regional_parser=eval_regional_parser
    )

    # Add indel-specific arguments
    eval_required.add_argument('--n_class', type=int, default=8, help='Number of classes for indel (default: 8)')

    eval_kmer_parser.add_argument('--kmer_length', type=int, default=2, help=textwrap.dedent("""
        Length of k-mer used for evaluation (typically 2, 4, or 6). Default: 2.
    """).strip())

    eval_parser._action_groups.append(eval_optional)
    return eval_parser


def add_snv_eval_parser(subparsers: argparse._SubParsersAction) -> argparse._SubParsersAction:
    """
    Register SNV specify parser for evaluation parser.

    This function is part of the logical flow:
        Parser Setup -> Common Parser regist -> Specify Parser regist
    """
    # Setup the common evaluation parser
    eval_parser, eval_required, eval_kmer_parser, eval_regional_parser, eval_optional = setup_eval_parser(
        subparsers=subparsers,
        parser_name='evaluate',
        help_message='Evaluate mural-SNV model'
    )

    # Register the common parser
    regist_common_eval_parser(
        eval_required=eval_required,
        eval_kmer_parser=eval_kmer_parser,
        eval_regional_parser=eval_regional_parser
    )

    # Add snv-specific arguments
    eval_required.add_argument('--n_class', type=int, default=4, help='Number of classes for SNV (default: 4)')

    eval_kmer_parser.add_argument('--kmer_length', type=int, default=3, help=textwrap.dedent("""
        Length of k-mer used for evaluation (typically 3, 5, or 7). Default: 3.
    """).strip())

    eval_parser._action_groups.append(eval_optional)
    return eval_parser