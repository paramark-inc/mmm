import logging
import sys
import unicodedata
import yaml

from typing import Tuple

# Regular ASCII whitespace is fine anywhere in a YAML file; any other
# Unicode whitespace (e.g. a non-breaking space pasted from a rich-text
# editor) is structurally invisible to YAML's indentation rules but
# visually indistinguishable from a real space, so a corrupted block
# silently parses as something other than what was intended (e.g. a
# nested `priors:` block landing as sibling keys instead). Fail loudly
# instead of letting that through.
_ALLOWED_WHITESPACE = {" ", "\t", "\n", "\r"}


def _check_for_invisible_whitespace(contents: str, config_file: str) -> None:
    for lineno, line in enumerate(contents.splitlines(), start=1):
        for char in line:
            if char in _ALLOWED_WHITESPACE:
                continue
            if unicodedata.category(char) in ("Zs", "Zl", "Zp") or (
                char.isspace() and char not in _ALLOWED_WHITESPACE
            ):
                raise ValueError(
                    f"{config_file}:{lineno}: found non-standard whitespace "
                    f"character U+{ord(char):04X} ({unicodedata.name(char, 'UNKNOWN')}). "
                    f"YAML only treats plain ASCII spaces as indentation, so this "
                    f"character silently breaks the file's structure instead of "
                    f"raising a parse error (e.g. a pasted `priors:` block can land "
                    f"as sibling keys instead of nesting correctly). This usually "
                    f"comes from pasting YAML out of a rich-text source (Notion, "
                    f"Google Docs, a web form, etc.) -- replace it with a plain "
                    f"ASCII space and re-check the file's indentation."
                )


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
            _check_for_invisible_whitespace(contents, config_file)
            f.seek(0)
            config = yaml.full_load(f)

        return contents, config

    except FileNotFoundError:
        logging.error(f"Couldn't load config file: {config_file}")
        raise
