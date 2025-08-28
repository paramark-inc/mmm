import logging
import sys
import yaml

from typing import Tuple


def load_config(config_file: str) -> Tuple[bytes, dict]:
    """
    Config object loader

    :param config_file: full path of file to load
    :return: (bytes, dict) bytes of file contents, dict of config data
    """
    try:
        with open(config_file, "r") as f:
            # read the full file contents (to preserve comments in describe_config()),
            # then reset the file pointer to get parsed yaml (for general use)
            contents = f.read()
            f.seek(0)
            config = yaml.full_load(f)

        return contents, config

    except FileNotFoundError:
        logging.error(f"Couldn't load config file: {config_file}")
        raise
