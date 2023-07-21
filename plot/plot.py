import matplotlib.pyplot as plt
import numpy as np
import os


def _plot_one_metric(axs, idx, chart_name, granularity, time_axis, date_strs, values):
    # add a primary X axis with the observation indices
    ax = axs[idx]
    ax.set_title(chart_name)
    ax.set_xlabel(f"Time({granularity})")
    ax.plot(time_axis, values)

    xlim = ax.get_xlim()
    ax.tick_params("x", labelbottom=True)

    # add a second X axis with the date labels
    ax2 = ax.twiny()
    ax2.set_xlim(xlim)
    ticks = [
        int(tick) for tick in ax.get_xticks() if tick >= 0.0 and int(tick) < date_strs.shape[0]
    ]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([date_strs[tick] for tick in ticks])
    ax2.set_xlabel("Date")
    ax2.tick_params("x", labeltop=True, labelrotation=45)


def plot_all_metrics(input_data, output_dir, suffix):
    """
    plots each metric in the input data object, generating an image file for each in the output directory.
    note that we are not plotting the costs at present.

    :param input_data: InputData model
    :param output_dir: directory to write image files to each
    :param suffix: suffix to append to filename
    :return: n/a
    """

    time_axis = np.arange(input_data.media_data.shape[0])

    # add 1 for the target metric
    num_metrics = (
        input_data.media_data.shape[1]
        + input_data.media_costs_by_row.shape[1]
        + input_data.extra_features_data.shape[1]
        + 1
    )
    fig, axs = plt.subplots(num_metrics, 1, sharex="all", figsize=(8, 4 * num_metrics))

    charts = []

    for media_idx in range(input_data.media_data.shape[1]):
        values = input_data.media_data[:, media_idx]
        chart_name = f"{input_data.media_names[media_idx]} (volume)"
        charts.append((chart_name, values))

        values = input_data.media_costs_by_row[:, media_idx]
        chart_name = f"{input_data.media_names[media_idx]} (cost)"
        charts.append((chart_name, values))

    for extra_features_idx in range(input_data.extra_features_data.shape[1]):
        values = input_data.extra_features_data[:, extra_features_idx]
        chart_name = input_data.extra_features_names[extra_features_idx]
        charts.append((chart_name, values))

    charts.append((input_data.target_name, input_data.target_data))

    for idx, (chart_name, values) in enumerate(charts):
        _plot_one_metric(
            axs=axs,
            idx=idx,
            chart_name=chart_name,
            granularity=input_data.time_granularity,
            time_axis=time_axis,
            date_strs=input_data.date_strs,
            values=values,
        )

    # tight_layout will space the charts out evenly, vertically
    fig.tight_layout()

    output_fname = os.path.join(output_dir, f"metrics_{suffix}.png")
    plt.savefig(output_fname)
