# cli.py
import sys
import argparse
import textwrap

from _version import __version__

# import commands
from commands.train import add_snv_train_parser
from commands.predict import add_snv_predict_parser
from commands.transfer import add_snv_transfer_parser
from commands.evaluate import add_snv_eval_parser
from commands.scale import add_snv_scale_parser, add_snv_calc_scaling_factor_parser
from commands.get_best_model import add_snv_get_best_model_parser

# import scripts
from scripts.run_train_raytune import run_train_pipline
from scripts.run_predict import run_predict_pipline
from scripts.run_train_TL_raytune import run_transfer_pipline
from scripts.calc_kmer_corr import run_kmer_corr_calc
from scripts.calc_regional_corr import run_regional_corr_calc
from scripts.scaling import scaling_files, calc_mu_scaling_factor
from scripts.get_best_model import get_best_model

class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        """Override the default error handling method"""
        sys.stderr.write('\nError: %s\n\n' % message)
        self.print_help(sys.stderr)
        sys.exit(1)

banner = """
mural_snv is a generalizable framework to estimate single-nucleotide mutation rates based on deep learning.

Version:   {}
About:     Developed for genome editing research.
Docs:      https://mural-doc.readthedocs.io/en/latest/
Mail:      mural-project@outlook.com
""".format(__version__)

def create_parser():
#    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
#                                     description=banner)

    parser = ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=banner)

    optional = parser._action_groups.pop()
    subparsers = parser.add_subparsers(
        dest='command',
        title='[Available Commands]',
        description="Run specific operations using these sub-commands:",
        metavar='<command>',
        required=True
    )

    # register subparsers for each command
    train_parser = add_snv_train_parser(subparsers)
    predict_parsert = add_snv_predict_parser(subparsers)
    transfer_parser = add_snv_transfer_parser(subparsers)
    eval_parser = add_snv_eval_parser(subparsers)
    calc_scaling_factor_parser = add_snv_calc_scaling_factor_parser(subparsers)
    scale_parser = add_snv_scale_parser(subparsers)
    get_best_model_parser = add_snv_get_best_model_parser(subparsers)

    # global options
    optional.title = '[General help]' 
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(__version__))
    parser._action_groups.append(optional)

    if len(sys.argv) == 1:
        parser.parse_args(['--help'])
    
    subparsers = {
        'train': train_parser,
        'predict': predict_parsert,
        'transfer': transfer_parser,
        'evaluate': eval_parser,
        'calc_scaling_factor': calc_scaling_factor_parser,
        'scale': scale_parser,
        'get_best_model': get_best_model_parser
    }
    
    return parser, subparsers

def main():
    parser, subparsers = create_parser()

    try:
        args = parser.parse_args()
    except Exception as e:
        print(f"Error: {e}")
        subparsers[args.func].print_help()
        sys.exit(1)

    if args.func == 'train':
        run_train_pipline(args, model_type='snv')

    elif args.func == 'predict':
        run_predict_pipline(args, model_type='snv')
    
    elif args.func == 'transfer':
        if not args.train_all:
            print("Warming: --train_all need used in commend line for snv transfer learning! Set it to True.")
            args.train_all = True
        run_transfer_pipline(args, model_type='snv')

    elif args.func == 'evaluate':
        assert (args.kmer_only and args.regional_only) is False, "Please set one of --kmer_only or --regional_only to True."
        if args.kmer_only:
            run_kmer_corr_calc(args, model_type='snv')
            return
        elif args.regional_only:
            run_regional_corr_calc(args)
            return

        run_kmer_corr_calc(args, model_type='snv')
        run_regional_corr_calc(args)
    
    elif args.func == 'calc_scaling_factor':
        calc_mu_scaling_factor(args, model_type='snv')

    elif args.func == 'scale':
        scaling_files(args.pred_file, args.scale_factor, args.n_class, args.out_file)
    
    elif args.func == 'get_best_model':
        get_best_model(args.trial_path)

    else:
        parser.print_help()

if __name__ == '__main__':
    main()