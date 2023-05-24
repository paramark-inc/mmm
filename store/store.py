import os
import secrets

from datetime import datetime


def generate_results_dirname(data_dir, customer):
    """
    generate a unique directory name for a given customer name

    :param data_dir: directory prefix
    :param customer: customer name suitable for including in a pathname
    :return: directory name of form <data_dir>/results/<customer>/<generated name>
    """
    now_date = datetime.now()
    yyyymmdd = now_date.date().strftime("%Y-%m-%d")
    seconds_since_midnight = now_date.hour * 3600 + now_date.minute * 60 + now_date.second
    hextoken = secrets.token_hex(4)
    return os.path.join(
        data_dir,
        "results",
        customer,
        f"{yyyymmdd}-{seconds_since_midnight}-{hextoken}"
    )
