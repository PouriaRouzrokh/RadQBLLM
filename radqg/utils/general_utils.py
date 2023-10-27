##########################################################################################
# Description: A script containing general functionalites for the project.
##########################################################################################

import os
import pathlib
import radqg.settings.configs as configs

# ----------------------------------------------------------------------------------------
# Logging utils
# ----------------------------------------------------------------------------------------


def talk(x):
    """Prints the given message if the VERBOSE flag is set to True.

    Args:
        x (str): The message to be printed.
    """
    if configs.VERBOSE:
        print(x)


# ----------------------------------------------------------------------------------------
# Path utils
# ----------------------------------------------------------------------------------------


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
