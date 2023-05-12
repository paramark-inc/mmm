import matplotlib.pyplot as plt
import numpy as np
import os

from constants import constants


def plot_all_metrics(data_dict, output_dir):
    """
    plots each metric in the input dict, generating an image file for each in the output directory

    :param data_dict: dict with format {
        KEY_GRANULARITY: granularity,
        KEY_OBSERVATIONS: observations,
        KEY_METRICS: { metric_name: [ metric values ], ... }
    }
    :param output_dir: directory to write image files to each
    :return: n/a
    """

    time_axis = np.arange(data_dict[constants.KEY_OBSERVATIONS])
    metric_data = data_dict[constants.KEY_METRICS]
    fig, axs = plt.subplots(len(metric_data), 1, sharex="all", figsize=(8, 12))

    i = 0

    for metric_name, metric_values in metric_data.items():
        axs[i].set_title(metric_name)
        axs[i].set_xlabel(f"Time({data_dict[constants.KEY_GRANULARITY]})")
        axs[i].plot(time_axis, metric_values)
        i = i + 1

    plt.savefig(os.path.join(output_dir, "all_metrics.png"))
