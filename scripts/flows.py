import os
from argparse import Namespace

'''
Helper function for reading flow inputs. If the script was not triggered by a flow, it will return command line arguments provided by the user.

All inputs are stored in a blob at /workflow/inputs/<NAME OF INPUT>.
For file input types, the blob is the file input itself.
For all other supported input types (str, int, bool, etc), they are stored as conents inside the blob.

Args:
    name (str): The name of the input
    args (argparse.Namespace): Command line arguments
    is_file (bool): Whether the input type is a file or not.

Returns:
    Any: Either the input file or value
'''
def read_input(name: str, args: Namespace, is_file: bool=False):
    if os.environ.get('DOMINO_IS_WORKFLOW_JOB') == 'false':
        return getattr(args, name) # Local execution, return the arguments in command line
    else:
        input_location = f'/workflow/inputs/{name}'
        if is_file:
            return input_location # Directly return the blob for file inputs
        else:
            with open(input_location, 'r') as file: # Read the contents of the blob for other inputs
                contents = file.read()
                return contents

'''
Helper function for getting a flow output location. If the script was not triggered by a flow, it will use the output folder provided through the command line argument.

All outputs must be stored in a blob at /workflow/outputs/<NAME OF INPUT>

Args:
    name (str): The name of the output
    args (argparse.Namespace): Command line arguments

Returns:
    Any: The path to where the output should be written to
'''
def get_output_location(name: str, args: Namespace):
    if os.environ.get('DOMINO_IS_WORKFLOW_JOB') == 'false': # Local execution, return a default output folder
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok=True)
        return f'{output_folder}/{name}'
    else:
        output_folder = '/workflow/outputs' # Flow execution, return the outputs folder
        return f'{output_folder}/{name}'
