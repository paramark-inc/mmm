import matplotlib.pyplot as plt
import numpy as np
import os


def _plot_one_metric(axs, idx, chart_name, granularity, time_axis, values):
    axs[idx].set_title(chart_name)
    axs[idx].set_xlabel(f"Time({granularity})")
    axs[idx].plot(time_axis, values)


def plot_all_metrics(input_data, output_dir):
    """
    plots each metric in the input data object, generating an image file for each in the output directory.
    note that we are not plotting the costs at present.

    :param input_data: InputData model
    :param output_dir: directory to write image files to each
    :return: n/a
    """

    time_axis = np.arange(input_data.media_data.shape[0])

    # add 1 for the target metric
    num_metrics = input_data.media_data.shape[1] + input_data.extra_features_data.shape[1] + 1
    fig, axs = plt.subplots(num_metrics, 1, sharex="all", figsize=(8, 4 * num_metrics))

    charts = []

    for media_idx in range(input_data.media_data.shape[1]):
        values = input_data.media_data[:, media_idx]
        chart_name = f"{input_data.media_names[media_idx]} (volume)"
        charts.append((chart_name, values))

    for extra_features_idx in range(input_data.extra_features_data.shape[1]):
        values = input_data.extra_features_data[:, extra_features_idx]
        chart_name = input_data.extra_features_names[extra_features_idx]
        charts.append((chart_name, values))

    charts.append((input_data.target_name, input_data.target_data))

    for idx, (chart_name, values) in enumerate(charts):
        _plot_one_metric(axs=axs, idx=idx, chart_name=chart_name, granularity=input_data.time_granularity,
                         time_axis=time_axis, values=values)

    # tight_layout will space the charts out evenly, vertically
    fig.tight_layout()

    output_fname = os.path.join(output_dir, "all_metrics.png")
    plt.savefig(output_fname)
    print(f"wrote '{output_fname}'")
