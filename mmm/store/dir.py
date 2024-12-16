import os
import secrets

import datetime


def generate_unique_date_key():
    now_date = datetime.datetime.now(datetime.UTC)
    yyyymmdd = now_date.date().strftime("%Y-%m-%d")
    seconds_since_midnight = now_date.hour * 3600 + now_date.minute * 60 + now_date.second
    hextoken = secrets.token_hex(4)
    return f"{yyyymmdd}-{seconds_since_midnight:05d}-{hextoken}"


def make_results_dir(data_dir: str, dirname_fixed: str = "", results_key=None) -> str:
    """
    create a directory with a unique name for this mmm run

    :param data_dir: directory prefix
    :param dirname_fixed: (optional) fixed path suffix, e.g. for a project name or customer name
    :return: directory name of form <data_dir>/results/<dirname_fixed>/<generated name>
    """
    if results_key is None:
        results_key = generate_unique_date_key()

    results_dir = os.path.join(data_dir, "results", dirname_fixed, results_key)
    print(f"Creating directory '{results_dir}' for output")
    os.makedirs(results_dir, exist_ok=False)

    return results_dir
