import logging
from multiprocessing.managers import Value

# noinspection PyPackageRequirements
import numpy as np
import pandas as pd
import sys

from mmm.constants import constants
from mmm.data import InputData


def _copy_metric_values_to_media_data(
    metric_values: np.ndarray,
    media_data: np.ndarray,
    channel_index: int,
    geo_index: int,
):
    """
    copy metric_values (from raw data format) to media data array (common InputData format)
    :param metric_values: metric values array from raw data format
    :param media_data: media data nd array
    :param channel_index: index into media data array (second dimension) for this channel
    :param geo_index: index into media data array (third dimension) for this geo, or -1 for a non-geo
        model
    :return:
    """
    if len(metric_values) != media_data.shape[0]:
        raise ValueError(
            f"Number of metric values ({len(metric_values)}) does not match array shape({media_data.shape[0]})"
        )

    if 3 == media_data.ndim and -1 == geo_index:
        raise ValueError("Expected geo_dim but non received")

    if 2 == media_data.ndim and -1 != geo_index:
        raise ValueError("Expected no geo_dim but received one")

    if -1 == geo_index:
        media_data[:, channel_index] = metric_values.astype(np.float64)
    else:
        media_data[:, channel_index, geo_index] = metric_values.astype(np.float64)


def _copy_cost_values_to_media_costs(
    metric_values_impressions: np.ndarray,
    metric_values_spend: np.ndarray,
    media_costs_by_row: np.ndarray,
    channel_index: int,
    geo_index: int,
):
    """
    copy cost values to media data array

    Args:
        metric_values_impressions: (in) impressions values
        metric_values_spend: (in) spend values
        media_costs: (out) spend array to copy to (dims: [channel])
        media_costs_by_row: (out) spend array by day to copy to
            (dims: [time, channel] / [time, channel, geo])
        channel_index: channel index for media costs / media costs by row
        geo_index: geo index for media costs / media costs by row, or -1 for national data

    """
    if len(metric_values_impressions) != len(metric_values_spend):
        raise ValueError("impressions / spend array length mismatch")

    if -1 == geo_index:
        media_costs_by_row[:, channel_index] = metric_values_spend.astype(np.float64)
    else:
        media_costs_by_row[:, channel_index, geo_index] = metric_values_spend.astype(np.float64)


def transform_input_generic(data_dict: dict, config: dict):
    """
    transform the raw input data into an InputData object

    :param data_dict: raw input data parsed from the csv.  See parse_csv_generic().
    :param config: config dictionary (from yaml file)
    :return: InputData object
    """

    # not existence of 'geo_col' in the yaml because there might be a geo filter, in which case
    # we do not have geo data in the output.
    has_geo_data = bool(data_dict[constants.KEY_GEO_NAMES])

    metric_data = data_dict[constants.KEY_METRICS]
    if has_geo_data:
        # all geos have the same column names, so we can get the list of column names from
        # an arbitrary geo.
        first_geo_metric_data = list(metric_data.values())[0]

        column_names = set(first_geo_metric_data.keys())
        geo_names = data_dict[constants.KEY_GEO_NAMES]
    else:
        column_names = set(metric_data.keys())
        geo_names = None
    column_names_already_seen = set()

    media_names = []
    num_media_channels = len(config.get("media", []))
    num_observations = data_dict[constants.KEY_OBSERVATIONS]
    num_geos = len(metric_data) if has_geo_data else 0

    # initialize numpy arrays for each each of our features
    # (in the format that lightweight_mmm expects)
    # media_data has dimensions [observations, channel] / [observations, channel, geo]
    if has_geo_data:
        media_data = np.ndarray(
            shape=(num_observations, num_media_channels, num_geos),
            dtype=np.float64,
        )
    else:
        media_data = np.ndarray(
            shape=(num_observations, num_media_channels),
            dtype=np.float64,
        )

    # For the case of a geo model, passing a 1-d array will cause LightweightMMM to use the same
    # priors for all geos.  We may want to consider setting geo-level priors but for now I figure
    # this is a feature and not a bug due to sparsity in some channels.  These arrays have
    # dimensions [channels] only.
    media_costs = np.zeros(shape=num_media_channels, dtype=np.float64)
    media_cost_priors = np.zeros(shape=num_media_channels, dtype=np.float64)
    learned_media_priors = np.zeros(shape=num_media_channels, dtype=np.float64)

    # media_costs_by_row has dimensions [observations, channels] / [observations, channels, geos]
    if has_geo_data:
        media_costs_by_row = np.zeros(
            shape=(num_observations, num_media_channels, num_geos),
            dtype=np.float64,
        )
    else:
        media_costs_by_row = np.zeros(
            shape=(num_observations, num_media_channels),
            dtype=np.float64,
        )

    extra_features_names = config.get("extra_features_cols", [])
    num_extra_features = len(extra_features_names)

    # extra_features_data has dimensions [observations, features] / [observations, features, geos]
    if has_geo_data:
        extra_features_data = np.ndarray(
            shape=(num_observations, num_extra_features, num_geos), dtype=np.float64
        )
    else:
        extra_features_data = np.ndarray(
            shape=(num_observations, num_extra_features), dtype=np.float64
        )

    # target_data has dimensions [observations] / [observations, geos]
    target_data = None

    # for each media channel going into our model, handle the impressions
    # and spend data, and build a list of display names (for charting)
    for channel_idx, media_config in enumerate(config.get("media", [])):
        display_name = media_config.get("display_name")
        impressions_col = media_config.get("impressions_col")
        spend_col = media_config.get("spend_col")

        if has_geo_data:
            for geo_idx, geo_name in enumerate(geo_names):
                _copy_metric_values_to_media_data(
                    metric_data[geo_name][impressions_col],
                    media_data,
                    channel_index=channel_idx,
                    geo_index=geo_idx,
                )
                _copy_cost_values_to_media_costs(
                    metric_data[geo_name][impressions_col],
                    metric_data[geo_name][spend_col],
                    media_costs_by_row,
                    channel_index=channel_idx,
                    geo_index=geo_idx,
                )
            media_costs[channel_idx] = media_costs_by_row[:, channel_idx, :].sum()
        else:
            _copy_metric_values_to_media_data(
                metric_data[impressions_col],
                media_data,
                channel_index=channel_idx,
                geo_index=-1,
            )
            _copy_cost_values_to_media_costs(
                metric_data[impressions_col],
                metric_data[spend_col],
                media_costs_by_row,
                channel_index=channel_idx,
                geo_index=-1,
            )
            media_costs[channel_idx] = media_costs_by_row[:, channel_idx].sum()

        # if the yaml specifies a learned prior, we set both the learned prior and the cost
        # prior.  The latter isn't necessary for fitting (since we ignore it in fit()), but
        # we keep it here for simplicity.
        learned_prior = media_config.get("learned_prior")
        if learned_prior:
            learned_media_priors[channel_idx] = learned_prior

        fixed_cost_prior = media_config.get("fixed_cost_prior")
        if fixed_cost_prior:
            media_cost_priors[channel_idx] = fixed_cost_prior
        else:
            media_cost_priors[channel_idx] = media_costs[channel_idx]

        # for geo models, we hardcode a media cost prior of 1000 when there is no spend data for
        # the column and no media cost prior.  This ensures a reasonable default for channels which
        # have spend for some geographies but not others, without taking on the complexity of
        # supporting geo-level media cost priors.  We do this based on existence of 'geo_col'
        # (i.e. regardless of whether there is a geo_filter or not) so that our notebooks will work
        # with the same yaml that we use for model fitting.
        has_geo_col = bool(config.get("geo_col", None))
        if has_geo_col and 0 == np.int64(media_costs[channel_idx]):
            print(
                f"Setting a prior of 1000 for '{display_name}' because its aggregated spend is zero."
            )
            media_cost_priors[channel_idx] = 1000

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
            if has_geo_data:
                target_data = np.ndarray(shape=(num_observations, num_geos), dtype=np.float64)
                for geo_idx, geo_name in enumerate(geo_names):
                    target_data[:, geo_idx] = metric_data[geo_name][metric_name].astype(np.float64)
            else:
                target_data = metric_data[metric_name].astype(np.float64)
            matched_columns.add(metric_name)

        elif metric_name in extra_features_names:
            extra_features_idx = extra_features_names.index(metric_name)
            if has_geo_data:
                for geo_idx, geo_name in enumerate(geo_names):
                    extra_features_data[:, extra_features_idx, geo_idx] = metric_data[geo_name][
                        metric_name
                    ].astype(np.float64)
            else:
                extra_features_data[:, extra_features_idx] = metric_data[metric_name].astype(
                    np.float64
                )
            matched_columns.add(metric_name)

    unaccounted = column_names - matched_columns
    if len(unaccounted) > 0:
        msg = f"Input data contains columns that weren't accounted for: {' '.join(unaccounted)}"
        # log but also raise exception since this an invariant we don't want to violate
        logging.error(msg)
        raise ValueError(msg)

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
        geo_names=geo_names,
    )
