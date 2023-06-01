import os

import numpy as np
import pandas as pd
import scipy.stats as spstats

from ..constants import constants
from ..model.model import InputData


def _print_outliers_for_data(name, data, output_dir, suffix):
    df = pd.DataFrame({'values': data})
    mean = df['values'].mean()
    stddev = df['values'].std()

    # mean and stddev are floats so this is a floating point operation
    outliers = df['values'][(df['values'] > (mean + 2 * stddev)) | (df['values'] < (mean - 2 * stddev))]

    with open(os.path.join(output_dir, f"outliers_{suffix}.txt"), "w") as outliers_file:
        outliers_file.write(
            f"outlier data points (stddev method) for {name} ({suffix}) (mean={mean:,.2f} stddev={stddev:,.2f})\n"
        )
        for i, v in outliers.items():
            stddevs_from_mean = np.absolute(v - mean) / stddev
            outliers_file.write(f"  index={i:4d} value={v:14,.0f} stddevs from mean={stddevs_from_mean:.2f}\n")

        outliers_file.write(f"outlier data points (top/bottom method) for {name} ({suffix}) (bottom values first)\n")
        sortedvalues = df['values'].sort_values(ascending=True)
        for i, v in sortedvalues.head(10).items():
            outliers_file.write(f"  index={i:4d} value={v:14,.0f}\n")
        for i, v in sortedvalues.tail(10).items():
            outliers_file.write(f"  index={i:4d} value={v:14,.0f}\n")


def print_outliers(input_data, output_dir, suffix):
    """
    print outliers (defined here as any data points more than 2 standard deviations from the mean)
    :param input_data: InputData object
    :param output_dir: path to directory to write to
    :param suffix: log suffix
    """

    metric_names_and_values = []

    for media_idx in range(input_data.media_data.shape[1]):
        name = f"{input_data.media_names[media_idx]} (volume)"
        metric_names_and_values.append((name, input_data.media_data[:, media_idx]))

    for extra_feature_idx in range(input_data.extra_features_data.shape[1]):
        metric_names_and_values.append(
            (input_data.extra_features_names[extra_feature_idx], input_data.extra_features_data[:, extra_feature_idx]))

    metric_names_and_values.append((input_data.target_name, input_data.target_data))

    for name, data in metric_names_and_values:
        _print_outliers_for_data(name=name, data=data, output_dir=output_dir, suffix=suffix)


# noinspection PyUnusedLocal
def _replace_with_trimmed_mean_editor_func(context, date_strs, media_data, extra_features_data, target_data):
    """
    See InputData.clone_with_data_edits
    :param context: client context
    :param date_strs: editable copy of date_strs
    :param media_data: editable copy of media_data
    :param extra_features_data: editable copy of extra_features_data
    :param target_data: editable copy of target_data
    :return: none
    """
    old_input_data = context["old_input_data"]
    media_data_outliers = context["media_data_outliers"]
    extra_features_outliers = context["extra_features_outliers"]
    target_outliers = context["target_outliers"]

    for media_name, outlier_indices in media_data_outliers.items():
        media_idx = old_input_data.media_names.index(media_name)
        assert media_idx >= 0, f"{media_idx} {media_name}"
        media_trimmed_mean = spstats.trim_mean(media_data[:, media_idx], 0.1)

        for outlier_idx in outlier_indices:
            media_data[outlier_idx][media_idx] = media_trimmed_mean

    for extra_features_name, outlier_indices in extra_features_outliers.items():
        extra_features_idx = old_input_data.extra_features_names.index(extra_features_name)
        assert extra_features_idx >= 0, f"{extra_features_idx} {extra_features_name}"
        extra_features_trimmed_mean = spstats.trim_mean(extra_features_data[:, extra_features_idx], 0.1)

        for outlier_idx in outlier_indices:
            extra_features_data[outlier_idx][extra_features_idx] = extra_features_trimmed_mean

    target_trimmed_mean = spstats.trim_mean(target_data, 0.1)
    for outlier_idx in target_outliers:
        target_data[outlier_idx] = target_trimmed_mean


def remove_outliers_from_input(input_data, media_data_outliers, extra_features_outliers, target_outliers, removal_type):
    """
    Generate a new input_data with outliers removed.

    :param input_data: InputData instance
    :param media_data_outliers: dict(media_name -> [indices]) of outlier indices to remove
    :param extra_features_outliers: dict(extra_feature_name -> [indices]) of outlier indices to remove
    :param target_outliers: [indices] of outlier indices to remove
    :param removal_type: constants.REMOVE_OUTLIERS_XXX constant describing how to perform the removal
    :return: new InputData instance
    """

    assert removal_type == constants.REMOVE_OUTLIERS_TYPE_REPLACE_WITH_TRIMMED_MEAN, f"{removal_type}"

    print(f"Removing outliers type={removal_type}")
    context = {
        "old_input_data": input_data,
        "media_data_outliers": media_data_outliers,
        "extra_features_outliers": extra_features_outliers,
        "target_outliers": target_outliers
    }

    return InputData.clone_with_data_edits(
        input_data=input_data,
        editor_func=_replace_with_trimmed_mean_editor_func,
        context=context
    )
