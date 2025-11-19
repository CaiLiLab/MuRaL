from typing import Callable, Any
import argparse
import textwrap

def add_common_get_best_model_parser(
    get_best_model_parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:
    """
    Add common arguments for all scaling parsers.
    """

    get_best_model_parser.set_defaults(func='get_best_model')

    # Define argument groups
    get_best_model_optional = get_best_model_parser._action_groups.pop()
    get_best_model_commoned = get_best_model_parser.add_argument_group("Commoned arguments")  
    get_best_model_optional.title = "Other arguments"

    # Add required arguments
    get_best_model_commoned.add_argument('--trial_path', required=True, type=str, metavar='FILE', 
                                help='Path to the trial results')


    # Re-append the optional arguments group
    get_best_model_parser._action_groups.append(get_best_model_optional)

    return get_best_model_commoned

def add_indel_get_best_model_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """
    Add a parser for scaling indel mutation rates.
    """
    get_best_model_parser = subparsers.add_parser(
        'get_best_model', 
        help='Find the best indel rate model from trial results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        After training, Each training trial generates these key files:

        -checkpoint_*.pth: Model weights
        -progress.csv: Training metrics per epoch (validation loss, time, etc.)
        
        You can use the command ``mural_indel get_best_model`` to find the best model 
        per trial after training.

        Command line examples
        ---------------------
        # running trials without Ray (default)
        mural_indel get_best_model ./results/your_experiment_name

        # running trials using Ray 
        mural_indel get_best_model ./ray_results/your_experiment_name
        
        The output lists all trials sorted by validation loss (ascending). Use the top result for predictions. Example:

        ./results/your_experiment_name/Train_zoq97_00000/checkpoint_8   0.401637  # ← Best model (lowest loss)
        ./results/your_experiment_name/Train_zoq97_00002/checkpoint_8   0.407693
        ./results/your_experiment_name/Train_zoq97_00001/checkpoint_6   0.445094
        """)
    )

    # Register common arguments
    get_best_model_commoned = add_common_get_best_model_parser(get_best_model_parser)

    return get_best_model_parser


def add_snv_get_best_model_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """
    Add a parser for scaling snv mutation rates.
    """
    get_best_model_parser = subparsers.add_parser(
        'get_best_model', 
        help='Find the best indel rate model from trial results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        After training, Each training trial generates these key files:

        -checkpoint_*.pth: Model weights
        -progress.csv: Training metrics per epoch (validation loss, time, etc.)
        
        You can use the command ``mural_indel get_best_model`` to find the best model 
        per trial after training.

        Command line examples
        ---------------------
        # running trials without Ray (default)
        mural_snv get_best_model ./results/your_experiment_name

        # running trials using Ray 
        mural_snv get_best_model ./ray_results/your_experiment_name
        
        The output lists all trials sorted by validation loss (ascending). Use the top result for predictions. Example:

        ./results/your_experiment_name/Train_zoq97_00000/checkpoint_8   0.401637  # ← Best model (lowest loss)
        ./results/your_experiment_name/Train_zoq97_00002/checkpoint_8   0.407693
        ./results/your_experiment_name/Train_zoq97_00001/checkpoint_6   0.445094
        """)
    )

    # Register common arguments
    get_best_model_commoned = add_common_get_best_model_parser(get_best_model_parser)

    return get_best_model_parser