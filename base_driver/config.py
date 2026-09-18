import logging
import sys
import yaml

from typing import Tuple

# Throw an error if we encounter non-standard whitespace (e.g. nbsp characters),
# which can confusingly look like indentation but don't count in YAML parsing
_ALLOWED_WHITESPACE = {" ", "\t", "\n", "\r"}


def _check_for_invisible_whitespace(contents: str, config_file: str) -> None:
    for lineno, line in enumerate(contents.splitlines(), start=1):
        for char in line:
            if char.isspace() and char not in _ALLOWED_WHITESPACE:
                raise ValueError(
                    f"Couldn't parse config {config_file} line {lineno}: "
                    "check for invalid whitespace characters"
                )


def load_config(config_file: str) -> Tuple[bytes, dict]:
    """
    Config object loader

    :param config_file: full path of file to load
    :return: (bytes, dict) bytes of file contents, dict of config data
    """
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            # read the full file contents (to preserve comments in describe_config()),
            # then reset the file pointer to get parsed yaml (for general use)
            contents = f.read()
            _check_for_invisible_whitespace(contents, config_file)
            f.seek(0)
            config = yaml.full_load(f)

        return contents, config

    except FileNotFoundError:
        logging.error(f"Couldn't load config file: {config_file}")
        raise
