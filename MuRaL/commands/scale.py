from typing import Callable, Any
import argparse
import textwrap


def add_common_scale_parser(
    scale_parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:
    """
    Add common arguments for all scaling parsers.
    """
    # Define argument groups
    scale_optional = scale_parser._action_groups.pop()
    scale_required = scale_parser.add_argument_group("Required arguments")  
    scale_optional.title = "Other arguments"

    # Add required arguments
    scale_required.add_argument('--pred_files', required=True, type=str, metavar='FILE', nargs='+',
                                help='Prediction files (one or more) for calculating scaling factors.')

    scale_required.add_argument('--out_file', type=str, metavar='FILE', nargs='+', 
                                help='Output file with scaled rates. Default: pred_file_name.scaled.tsv.gz')

    scale_required.add_argument('--benchmark_regions', type=str, metavar='FILE', default='',
                                help='High-confidence regions used for calculating the scaling factor')

    scale_required.add_argument('--genomewide_mu', type=float, metavar='FLOAT', default=None, 
                                help='Mutation rate per base per generation.')

    scale_required.add_argument('--m_proportions', type=float, metavar='FLOAT', nargs='+', 
                                help='Proportions of specific mutation types.')

    scale_required.add_argument('--g_proportions', type=float, metavar='FLOAT', nargs='+', 
                                help='Proportions of specific sites in the genome.')

    scale_required.add_argument('--do_scaling', default=False, action='store_true', 
                                help='Save scaled mutation rates for input pred files. Default: False.')

    scale_required.add_argument('--scale_factors', type=float, metavar='FLOAT', nargs='+', 
                                help='Scaling factor for producing mutation rates per base per generation.')

    # Re-append the optional arguments group
    scale_parser._action_groups.append(scale_optional)

    return scale_required


def add_indel_scale_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """
    Add a parser for scaling indel mutation rates.
    """
    scale_parser = subparsers.add_parser(
        'scale', 
        help='Scale predicted INDEL rates to per base per generation rates', 
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        The raw MuRaL-predicted INDEL rates are not per bp per generation. 
        To obtain a INDEL rates per bp per generation for each nucleotide, 
        one can scale the MuRaL-predicted rates using reported genome-wide 
        INDEL rate and spectrum per generation.
        """)
    )

    # Register common arguments
    scale_required = add_common_scale_parser(scale_parser)

    # Add indel-specific arguments
    scale_required.add_argument('--n_class', type=int, default=8, 
                                help='Number of mutation classes (or types), including the non-mutated class. Default: 8.')

    return scale_parser


def add_snv_scale_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """
    Add a parser for scaling snv mutation rates.
    """
    scale_parser = subparsers.add_parser(
        'scale', 
        help='Scale predicted de-novo mutation rates to per base per generation rates', 
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        The raw MuRaL-predicted mutation rates are not  per bp per generation. 
        To obtain a mutation rate per bp per generation for each nucleotide, 
        one can scale the MuRaL-predicted rates using reported genome-wide 
        mutation rate and spectrum per generation.

        Example Scaling Command:
        --------------------------
        1. Calculate scaling factors:

            mural_snv scale --pred_files AT_validation.ckpt6.fdiri.tsv.gz --genomewide_mu 5e-9 
            --m_proportions 0.355 --g_proportions 0.475 > scaling_factor.out

        # Output file 'scaling_factor.out' may look like the following:
            pred_file: AT_validation.ckpt6.fdiri.tsv.gz
            genomewide_mu: 5e-09
            n_sites: 84000
            g_proportion: 0.475
            m_proportion: 0.355
            prob_sum: 4.000e+03
            scaling factor: 7.848e-08

        2. Apply scaing:

            mural_snv scale --pred_files AT_validation.ckpt6.fdiri.tsv.gz --scale_factors 7.848e-08
            --out_file AT_validation.ckpt6.fdiri.scaled.tsv.gz

        Similarly, you can generate the scaled mutation rates for non-CpG and CpG sites like 
        the above example. More details can be found in the MuRaL paper
        """)
    )

    # Register common arguments
    scale_required = add_common_scale_parser(scale_parser)

    # Add snv-specific arguments
    scale_required.add_argument('--n_class', type=int, default=4, 
                                help='Number of mutation classes (or types), including the non-mutated class. Default: 4.')

    return scale_parser