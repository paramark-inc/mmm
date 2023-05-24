import os
import secrets

from datetime import datetime

from impl.lightweight_mmm.lightweight_mmm.utils import save_model as lwmmm_save_model


def make_results_dir(data_dir, dirname_fixed):
    """
    create a directory with a unique name for this mmm run

    :param data_dir: directory prefix
    :param dirname_fixed: customer name suitable for including in a pathname
    :return: directory name of form <data_dir>/results/<dirname_fixed>/<generated name>
    """
    now_date = datetime.now()
    yyyymmdd = now_date.date().strftime("%Y-%m-%d")
    seconds_since_midnight = now_date.hour * 3600 + now_date.minute * 60 + now_date.second
    hextoken = secrets.token_hex(4)
    results_dir = os.path.join(
        data_dir,
        "results",
        dirname_fixed,
        f"{yyyymmdd}-{seconds_since_midnight}-{hextoken}"
    )
    print(f"Creating directory '{results_dir}' for output")
    os.makedirs(results_dir, exist_ok=False)

    return results_dir


def save_model(mmm, results_dir):
    """
    save the lightweight mmm model file to the file system
    :param mmm: LightweightMMM instance
    :param results_dir: directory to write the model file to
    :return: fully qualified path to model file
    """
    output_fname = os.path.join(results_dir, "model.lwm")
    lwmmm_save_model(mmm, output_fname)
    print(f"wrote {output_fname}")

    return output_fname
