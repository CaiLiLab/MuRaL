from typing import Callable, Any
import argparse
import textwrap

def add_common_predict_parser(
    predict_parser: argparse.ArgumentParser,
) -> tuple:
    """
    Add common arguments for all prediction parsers.
    """

    predict_parser.set_defaults(func='predict')

    # Define argument groups
    predict_optional = predict_parser._action_groups.pop()
    predict_required = predict_parser.add_argument_group("Predict Required arguments")
    predict_optional.title = "Other arguments"

    # Add required arguments
    predict_required.add_argument('--ref_genome', type=str, metavar='FILE', default='',  
                                  required=True, help=textwrap.dedent("""
                                  File path of the reference genome in FASTA format.""").strip())
    
    predict_required.add_argument('--test_data', type=str, metavar='FILE', required=True,
                                  help=textwrap.dedent("""
                                  File path of the data to do prediction, in BED format.""").strip())
    
    predict_required.add_argument('--model_path', type=str, metavar='FILE', required=True,
                                  help=textwrap.dedent("""
                                  File path of the trained model.
                                  """ ).strip())
        
    predict_required.add_argument('--model_config_path', type=str, metavar='FILE', required=True,
                                  help=textwrap.dedent("""
                                  File path for the configurations of the trained model.
                                  """ ).strip()) 

    # Add optional arguments
    predict_optional.add_argument('--pred_file', type=str, metavar='FILE', default='pred.tsv.gz', 
                                 help=textwrap.dedent("""
                                 Name of the output file for prediction results.
                                 Default: 'pred.tsv.gz'.
                                 """ ).strip())
        
    predict_optional.add_argument('--calibrator_path', type=str, metavar='FILE', default='', 
                                  help=textwrap.dedent("""
                                  File path for the paired calibrator of the trained model.
                                  """ ).strip())

    predict_optional.add_argument('--poisson_calib', default=False, action='store_true', 
                                  help=textwrap.dedent("""
                                  Use Poisson calibration for the model. 
                                  Default: False.""").strip())
    
    predict_optional.add_argument('--bw_paths', type=str, metavar='FILE', default=None,
                                  help=textwrap.dedent("""
                                  File path for a list of BigWig files for non-sequence 
                                  features such as the coverage track. Default: None.""").strip())

    predict_optional.add_argument('--n_h5_files', metavar='INT', default=1, 
                                  help=argparse.SUPPRESS)
    
    predict_optional.add_argument('--pred_time_view', default=False, action='store_true',  
                                  help=textwrap.dedent("""
                                  Check pred time of each part. Default: False.
                                  """).strip())
    
    predict_optional.add_argument('--with_h5', default=False, action='store_true',  
                                  help=argparse.SUPPRESS)

    predict_optional.add_argument('--h5f_path', type=str, default=None,
                                  help=argparse.SUPPRESS)

    predict_optional.add_argument('--cpu_only', default=False, action='store_true',  
                                  help=textwrap.dedent("""
                                  Only use CPU computing. Default: False.
                                  """).strip())

    predict_optional.add_argument('--cuda_id', type=str, metavar='STR', default=None, 
                                  help=textwrap.dedent("""
                                  Which GPU device to be used. Default: '0'. 
                                  """ ).strip())
    
    predict_optional.add_argument('--segment_center', type=int, metavar='INT', default=300000,
                                  help=textwrap.dedent("""
                                  The maximum encoding unit of the sequence. It affects trade-off 
                                  between RAM memory and preprocessing speed. It is recommended to use 300k.
                                  Default: 300000.""" ).strip())

    predict_optional.add_argument('--pred_batch_size', type=int, metavar='INT', default=16, 
                                  help=textwrap.dedent("""
                                  Size of mini batches for prediction. Default: 16.
                                  """ ).strip())
    
    predict_optional.add_argument('--kmer_corr', type=int, metavar='INT', default=[], nargs='+',
                                  help=textwrap.dedent("""
                                  Calculate k-mer correlations with observed variants in 5th column.
                                  Accept one or more odd positive integers for k-mers, e.g., "3 5 7".
                                  Default: no value.
                                  """ ).strip())
    
    predict_optional.add_argument('--region_corr', type=int, metavar='INT', default=[], nargs='+',
                                  help=textwrap.dedent("""
                                  Calculate region correlations with observed variants in 5th column.
                                  Accept one or more positive integers for window size (bp), 
                                  e.g., "10000 50000". Default: no value.
                                  """ ).strip())
    
    predict_parser._action_groups.append(predict_optional)

    return predict_required, predict_optional


def add_indel_predict_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add a parser for predicting indel models.
    """
    predict_parser = subparsers.add_parser(
      'predict', 
      help='Predict mural-indel model', 
      formatter_class=argparse.RawTextHelpFormatter,
      description="""
    Overview
    -------- 
    This tool uses a trained MuRaL model to do prediction for the sites in the 
    input BED file.
    
    * Input data 
    Required input files for prediction include the reference FASTA file, 
    a BED-formatted data file and a trained model. The BED file is organized 
    in the same way as that for training. The 5th column can be set to '0' 
    if no observed mutations for the sites in the prediction BED. The 
    model-related files for input are 'model' and 'model.config.pkl', which 
    are generated at the training step. The file 'model.fdiri_cal.pkl', which 
    is for calibrating predicted INDEL rates, is optional. If the input BED
    file has many sites (e.g. many millions), it is recommended to split it
    into smaller files (e.g. 1 million each) for parallel processing.
   
    * Output data 
    The output of `mural_predict` is a tab-separated file containing the 
    sequence coordinates and the predicted probabilities for all possible 
    INDEL types. Usually, the 'prob0' column stores probabilities for the 
    non-indel class and other 'probX' columns for mutated classes. 
   
    Some example lines of a prediction output file are shown below:
    chrom   start   end    strand mut_type  prob0   prob1   prob2   prob3 ... prob7
    chr1    10006   10007   -       0       0.9597  0.003134 0.01444 0.002724 ... 0.003294
    chr1    10007   10008   +       0       0.9649  0.005517 0.00707 0.002520 ... 0.002123
    chr1    10008   10009   +       0       0.9617  0.004801 0.01006 0.003399 ... 0.001942
    chr1    10012   10013   -       0       0.9511  0.004898 0.02029 0.003746 ... 0.001764

    Command line examples
    ---------------------
    All files are located in: example/indel

    1. The following command will predict INDEL rates for all sites in 
    'testing.bed.gz' using model files under the 'checkpoint_6/' folder 
    and save prediction results into 'testing.ckpt6.fdiri.tsv.gz'. For most
    models, as prediction tasks usually won't take long, it is recommended to 
    set '--cpu_only' for using only CPUs and not generating HDF5 files.
    If the input BED file has many sites (e.g. many millions), it is recommended 
    to spilt it into smaller files (e.g. 1 million each) for parallel processing.
    
        mural_indel predict --ref_genome data/seq.fa --test_data data/testing.bed.gz \\
        --model_path models/checkpoint_9/model \\
        --model_config_path models/checkpoint_9/model.config.pkl \\
        --calibrator_path models/checkpoint_9/model.fdiri_cal.pkl \\
        --pred_file testing.ckpt9.fdiri.tsv.gz \\
        --cpu_only \\
        > test.out 2> test.err
    """)

    # Register common arguments
    add_common_predict_parser(predict_parser)


def add_snv_predict_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add a parser for predicting snv models.
    """
    predict_parser = subparsers.add_parser(
      'predict', 
      help='Predict mural-snv model', 
      formatter_class=argparse.RawTextHelpFormatter,
      description="""
    Overview
    -------- 
    This tool uses a trained MuRaL model to do prediction for the sites in the 
    input BED file.
    
    * Input data 
    Required input files for prediction include the reference FASTA file, 
    a BED-formatted data file and a trained model. The BED file is organized 
    in the same way as that for training. The 5th column can be set to '0' 
    if no observed mutations for the sites in the prediction BED. The 
    model-related files for input are 'model' and 'model.config.pkl', which 
    are generated at the training step. The file 'model.fdiri_cal.pkl', which 
    is for calibrating predicted mutation rates, is optional. If the input BED
    file has many sites (e.g. many millions), it is recommended to split it
    into smaller files (e.g. 1 million each) for parallel processing.
   
    * Output data 
    The output of `mural_predict` is a tab-separated file containing the 
    sequence coordinates and the predicted probabilities for all possible 
    mutation types. Usually, the 'prob0' column stores probabilities for the 
    non-mutated class and other 'probX' columns for mutated classes. 
   
    Some example lines of a prediction output file are shown below:
    chrom   start   end    strand mut_type  prob0   prob1   prob2   prob3
    chr1    10006   10007   -       0       0.9597  0.003134 0.01444 0.002724 
    chr1    10007   10008   +       0       0.9649  0.005517 0.00707 0.002520
    chr1    10008   10009   +       0       0.9617  0.004801 0.01006 0.003399
    chr1    10012   10013   -       0       0.9511  0.004898 0.02029 0.003746

    Command line examples
    ---------------------
    All files are located in: example/snv

    1. The following command will predict mutation rates for all sites in 
    'testing.bed.gz' using model files under the 'checkpoint_6/' folder 
    and save prediction results into 'testing.ckpt6.fdiri.tsv.gz'. For most
    models, as prediction tasks usually won't take long, it is recommended to 
    set '--cpu_only' for using only CPUs and not generating HDF5 files.
    If the input BED file has many sites (e.g. many millions), it is recommended 
    to spilt it into smaller files (e.g. 1 million each) for parallel processing.
    
        mural_snv predict --ref_genome data/seq.fa --test_data data/testing.bed.gz \\
        --model_path models/checkpoint_6/model \\
        --model_config_path models/checkpoint_6/model.config.pkl \\
        --calibrator_path models/checkpoint_6/model.fdiri_cal.pkl \\
        --pred_file models/testing.ckpt6.fdiri.tsv.gz \\
        --cpu_only \\
        > test.out 2> test.err
    """)
    # Register common arguments
    add_common_predict_parser(predict_parser)