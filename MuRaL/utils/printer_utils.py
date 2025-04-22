from functools import partial

def get_printer(use_ray, file_path):
    if use_ray:
        return print
    else:
        return partial(print_to_both, file_path=file_path)

def print_to_both(*args, file_path, **kwargs):
    """
    Prints the given arguments to both the standard output and a specified file.

    Args:
        *args: Variable length argument list to print.
        file_path (str): Path to the file where the output will be written. Default is 'output.log'.
        **kwargs: Keyword arguments for the built-in print function.
    """
    # Convert arguments to a single string
    message = ' '.join(map(str, args))
    
    # Print to standard output
    print(message, **kwargs)
    
    # Append to the specified file
    with open(file_path, 'a') as file:
        print(message, file=file, **{k: v for k, v in kwargs.items() if k != 'file'})
