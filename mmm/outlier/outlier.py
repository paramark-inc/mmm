import os

import numpy as np
import pandas as pd
import scipy.stats as spstats

from mmm.constants import constants
from mmm.data import InputData


def print_outliers(input_data, output_dir, suffix):
    """
    print outliers (defined here as any data points more than 2 standard deviations from the mean)
    :param input_data: InputData object
    :param output_dir: path to directory to write to
    :param suffix: log suffix
    """

    metric_names_and_values = []

    for media_idx in range(input_data.media_data.shape[1]):
        name = f"{input_data.media_names[media_idx]}"
        metric_names_and_values.append((name, input_data.media_data[:, media_idx]))

    for extra_feature_idx in range(input_data.extra_features_data.shape[1]):
        metric_names_and_values.append(
            (
                input_data.extra_features_names[extra_feature_idx],
                input_data.extra_features_data[:, extra_feature_idx],
            )
        )

    metric_names_and_values.append((input_data.target_name, input_data.target_data))

    with open(os.path.join(output_dir, f"outliers_{suffix}.txt"), "w") as outliers_file:
        first = True
        for name, data in metric_names_and_values:
            if first:
                first = False
            else:
                outliers_file.write("\n")
            outliers_file.write(f"{name}:\n")
            df = pd.DataFrame({"values": data})
            mean = df["values"].mean()
            stddev = df["values"].std()
            # mean and stddev are floats so this is a floating point operation
            outliers = df["values"][
                (df["values"] > (mean + 2 * stddev)) | (df["values"] < (mean - 2 * stddev))
            ]

            outliers_file.write(f"stddev method (mean={mean:,.2f} stddev={stddev:,.2f}):\n")
            for i, v in outliers.items():
                stddevs_from_mean = np.absolute(v - mean) / stddev
                outliers_file.write(
                    f"index={i:4d} value={v:14,.0f} stddevs from mean={stddevs_from_mean:.2f}\n"
                )

            outliers_file.write(f"\nSmallest 10 values:\n")
            sortedvalues = df["values"].sort_values(ascending=True)
            for i, v in sortedvalues.head(10).items():
                outliers_file.write(f"index={i:4d} value={v:14,.0f}\n")
            outliers_file.write(f"\nLargest 10 values:\n")
            for i, v in sortedvalues.tail(10).items():
                outliers_file.write(f"index={i:4d} value={v:14,.0f}\n")


def _compute_outlier_replacement_value(removal_type, data):
    """
    Compute the value to replace outliers with, based on the removal type.
    :param removal_type: constants.REMOVE_OUTLIERS_XXX constant
    :param data: array of values, with outlier values present
    :return: single replacement value for outlier values
    """
    if removal_type == constants.REMOVE_OUTLIERS_TYPE_REPLACE_WITH_TRIMMED_MEAN:
        return spstats.trim_mean(data, 0.1)
    elif removal_type == constants.REMOVE_OUTLIERS_TYPE_REPLACE_WITH_P10_VALUE:
        return np.percentile(data, 10)
    else:
        assert False, removal_type


# noinspection PyUnusedLocal
def _replace_outlier_editor_func(context, date_strs, media_data, extra_features_data, target_data):
    """
    See InputData.clone_with_data_edits
    :param context: client context
    :param date_strs: editable copy of date_strs
    :param media_data: editable copy of media_data
    :param extra_features_data: editable copy of extra_features_data
    :param target_data: editable copy of target_data
    :return: none
    """
    removal_type = context["removal_type"]
    old_input_data = context["old_input_data"]
    media_data_outliers = context["media_data_outliers"]
    extra_features_outliers = context["extra_features_outliers"]
    target_outliers = context["target_outliers"]

    for media_name, outlier_indices in media_data_outliers.items():
        media_idx = old_input_data.media_names.index(media_name)
        assert media_idx >= 0, f"{media_idx} {media_name}"
        media_val = _compute_outlier_replacement_value(removal_type, media_data[:, media_idx])

        for outlier_idx in outlier_indices:
            media_data[outlier_idx][media_idx] = media_val

    for extra_features_name, outlier_indices in extra_features_outliers.items():
        extra_features_idx = old_input_data.extra_features_names.index(extra_features_name)
        assert extra_features_idx >= 0, f"{extra_features_idx} {extra_features_name}"
        extra_features_val = _compute_outlier_replacement_value(
            removal_type, extra_features_data[:, extra_features_idx]
        )

        for outlier_idx in outlier_indices:
            extra_features_data[outlier_idx][extra_features_idx] = extra_features_val

    target_val = _compute_outlier_replacement_value(removal_type, target_data)
    for outlier_idx in target_outliers:
        target_data[outlier_idx] = target_val


def remove_outliers_from_input(
    input_data, media_data_outliers, extra_features_outliers, target_outliers, removal_type
):
    """
    Generate a new input_data with outliers removed.

    :param input_data: InputData instance
    :param media_data_outliers: dict(media_name -> [indices]) of outlier indices to remove
    :param extra_features_outliers: dict(extra_feature_name -> [indices]) of outlier indices to remove
    :param target_outliers: [indices] of outlier indices to remove
    :param removal_type: constants.REMOVE_OUTLIERS_XXX constant describing how to perform the removal
    :return: new InputData instance
    """

    assert removal_type in (
        constants.REMOVE_OUTLIERS_TYPE_REPLACE_WITH_TRIMMED_MEAN,
        constants.REMOVE_OUTLIERS_TYPE_REPLACE_WITH_P10_VALUE,
    ), f"{removal_type}"

    context = {
        "removal_type": removal_type,
        "old_input_data": input_data,
        "media_data_outliers": media_data_outliers,
        "extra_features_outliers": extra_features_outliers,
        "target_outliers": target_outliers,
    }

    return InputData.clone_with_data_edits(
        input_data=input_data, editor_func=_replace_outlier_editor_func, context=context
    )
