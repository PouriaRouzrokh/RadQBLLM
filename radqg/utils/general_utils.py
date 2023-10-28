##########################################################################################
# Description: A script containing general functionalites for the project.
##########################################################################################

import os
import pathlib
import tiktoken

# ----------------------------------------------------------------------------------------
# redirect_path


def redirect_path(path, counter_limit=10):
    """Changes an input relative path which is being called from the root direcotry to a
    new path that is being called
    from the current directory.
    Assumes that the "main.py" file in the root directory.

    Args:
        path (_type_): The input path to be changed.
        counter_limit (int, optional): The maximum number of times to go up in the
        directory tree. Defaults to 10.
    """
    path = pathlib.Path(path)
    current_dir = pathlib.Path().absolute()
    counter = 0
    while "main.py" not in os.listdir(current_dir):
        current_dir = current_dir.parent
        counter += 1
        if counter > counter_limit:
            raise FileNotFoundError(
                f"Could not find the file {path} in the current directory or its parents."
            )
    new_path = current_dir / path
    return str(new_path)


# ----------------------------------------------------------------------------------------
# count_tokens


def count_tokens(string: str, encoding_name: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a string.

    Args:
        string (str): the string to count tokens in.
        encoding_name (str, optional): Name of the LLM model. Defaults to "gpt-3.5-turbo".

    Returns:
        int: number of tokens in the string.
    """

    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))

    return num_tokens
