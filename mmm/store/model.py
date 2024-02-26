import os

from impl.lightweight_mmm.lightweight_mmm.utils import load_model as lwmmm_load_model
from impl.lightweight_mmm.lightweight_mmm.utils import save_model as lwmmm_save_model


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


def load_model(model_filename):
    """
    load a model that was created in a previous run
    :param model_filename: model filename
    :return: LightweightMMM instance
    """
    return lwmmm_load_model(file_path=model_filename)
