import logging

# noinspection PyPackageRequirements
import numpy as np
import pandas as pd
import sys

from mmm.constants import constants
from mmm.data import InputData


def _copy_metric_values_to_media_data(metric_values, media_data, channel_index):
    """
    copy metric_values (from raw data format) to media data array (common InputData format)
    :param metric_values: metric values array from raw data format
    :param media_data: media data 2d array
    :param channel_index: index into media data array (second dimension) for this channel
    :return:
    """
    assert len(metric_values) == media_data.shape[0], f"{len(metric_values)} {media_data.shape[0]}"
    media_data[:, channel_index] = metric_values.astype(np.float64)


def _copy_cost_values_to_media_costs(
    metric_values_impressions, metric_values_spend, media_costs, media_costs_by_row, channel_index
):
    assert len(metric_values_impressions) == len(
        metric_values_spend
    ), f"{metric_values_impressions} {metric_values_spend}"

    media_costs_by_row[:, channel_index] = metric_values_spend.astype(np.float64)
    media_costs[channel_index] = media_costs_by_row[:, channel_index].sum()


def transform_input_generic(data_dict: dict, config: dict):
    """
    transform the raw input data into an InputData object

    :param data_dict: raw input data parsed from the csv
    :param config: config dictionary (from yaml file)
    :return: InputData object
    """

    metric_data = data_dict[constants.KEY_METRICS]
    column_names = set(metric_data.keys())
    column_names_already_seen = set()

    media_names = []
    num_media_channels = len(config.get("media", []))

    # initialize numpy arrays for each each of our features
    # (in the format that lightweight_mmm expects)
    media_data = np.ndarray(
        shape=(data_dict[constants.KEY_OBSERVATIONS], num_media_channels), dtype=np.float64
    )

    media_costs = np.zeros(shape=num_media_channels, dtype=np.float64)
    media_cost_priors = np.zeros(shape=num_media_channels, dtype=np.float64)
    learned_media_priors = np.zeros(shape=num_media_channels, dtype=np.float64)
    media_costs_by_row = np.zeros(
        shape=(data_dict[constants.KEY_OBSERVATIONS], num_media_channels), dtype=np.float64
    )

    extra_features_names = config.get("extra_features_cols", [])
    num_extra_features = len(extra_features_names)
    extra_features_data = np.ndarray(
        shape=(data_dict[constants.KEY_OBSERVATIONS], num_extra_features), dtype=np.float64
    )
    target_data = None

    # for each media channel going into our model, handle the impressions
    # and spend data, and build a list of display names (for charting)
    for i, media_config in enumerate(config.get("media", [])):
        display_name = media_config.get("display_name")
        impressions_col = media_config.get("impressions_col")
        spend_col = media_config.get("spend_col")

        _copy_metric_values_to_media_data(metric_data[impressions_col], media_data, i)

        _copy_cost_values_to_media_costs(
            metric_data[impressions_col],
            metric_data[spend_col],
            media_costs,
            media_costs_by_row,
            i,
        )

        # if the yaml specifies a learned prior, we set both the learned prior and the cost
        # prior.  The latter isn't necessary for fitting (since we ignore it in fit()), but
        # we keep it here for simplicity.
        learned_prior = media_config.get("learned_prior")
        if learned_prior:
            learned_media_priors[i] = learned_prior

        fixed_cost_prior = media_config.get("fixed_cost_prior")
        if fixed_cost_prior:
            media_cost_priors[i] = fixed_cost_prior
        else:
            media_cost_priors[i] = media_costs[i]

        media_names.append(display_name)

        # Allow reusing column names so only attempt to remove a column from the set if it
        # has not already been seen
        if impressions_col not in column_names_already_seen:
            column_names.remove(impressions_col)
            column_names_already_seen.add(impressions_col)

        # if we are using spend as the input metric, the spend column is the same as the
        # impressions column, so don't try to remove it twice.
        if impressions_col != spend_col and spend_col not in column_names_already_seen:
            column_names.remove(spend_col)
            column_names_already_seen.add(spend_col)

    # after copying spend/impression data for each media channel, loop over
    # the remaining columns to populate target and extra features
    matched_columns = set()
    for metric_name in column_names:
        if metric_name == config.get("target_col"):
            target_data = metric_data[metric_name].astype(np.float64)
            matched_columns.add(metric_name)

        elif metric_name in extra_features_names:
            extra_features_idx = extra_features_names.index(metric_name)
            extra_features_data[:, extra_features_idx] = metric_data[metric_name].astype(np.float64)
            matched_columns.add(metric_name)

    unaccounted = column_names - matched_columns
    if len(unaccounted) > 0:
        logging.error(
            f"Input data contains columns that weren't accounted for: {' '.join(unaccounted)}"
        )
        sys.exit(1)

    return InputData(
        date_strs=np.array(data_dict[constants.KEY_DATE_STRS]),
        time_granularity=data_dict[constants.KEY_GRANULARITY],
        media_data=media_data,
        media_costs=media_costs,
        media_costs_by_row=media_costs_by_row,
        media_cost_priors=media_cost_priors,
        learned_media_priors=learned_media_priors,
        media_names=media_names,
        extra_features_data=extra_features_data,
        extra_features_names=extra_features_names,
        target_data=target_data,
        target_is_log_scale=config.get("log_scale_target"),
        target_name=config.get("target_col"),
    )
