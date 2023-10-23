from contextlib import redirect_stdout
import math
import numpy as np
import jax.numpy as jnp
import pandas as pd
import os
import lightweight_mmm.lightweight_mmm

from impl.lightweight_mmm.lightweight_mmm.plot import (
    plot_bars_media_metrics,
    plot_media_baseline_contribution_area_plot,
    plot_media_channel_posteriors,
    plot_model_fit,
    plot_out_of_sample_model_fit,
    plot_prior_and_posterior,
    plot_response_curves,
    create_media_baseline_contribution_df,
)

from impl.lightweight_mmm.lightweight_mmm.media_transforms import calculate_seasonality

from mmm.constants import constants
from mmm.data.input_data import InputData
from mmm.data.data_to_fit import DataToFit
from mmm.outlier.outlier import print_outliers
from mmm.plot.plot import plot_all_metrics


def describe_input_data(input_data, results_dir, suffix):
    """
    plot and print diagnostic analyses based on the input data
    :param input_data: InputData instance
    :param results_dir: directory to write plot files to
    :param suffix: suffix to append to filename
    :return:
    """
    plot_all_metrics(input_data=input_data, output_dir=results_dir, suffix=suffix)
    print_outliers(input_data=input_data, output_dir=results_dir, suffix=suffix)


def describe_config(output_dir, config, git_sha):
    """
    write text files with model configuration, so we can reproduce this run in future
    :param output_dir: directory to write plot files to
    :param config: raw contents (bytes) of the config file used for this run
    :param git_sha: current commit hash of the repo used to generate these results
    :return:
    """
    output_fname = os.path.join(output_dir, "git_sha.txt")
    with open(output_fname, "w") as f:
        f.write(git_sha)

    output_fname = os.path.join(output_dir, "config.yaml")
    with open(output_fname, "w") as f:
        f.write(config)


def _compute_blended_media_effect_hat(media_effect_hat: np.ndarray) -> np.ndarray:
    """
    Compute blended media effect hat.

    Args:
        media_effect_hat: array of media effect values, with axes:
           0: sample
           1: channel

    Returns:
        Array of blended media effect values, with axes:
          0: sample
    """
    # blended_media_effect_hat => array with one dimension (sample_count) representing the overall
    # media effect across the training period, as a percentage of the total target prediction.
    blended_media_effect_hat = media_effect_hat.sum(axis=1)

    return blended_media_effect_hat


def get_media_effect_df(data_to_fit: DataToFit, media_effect_hat: np.ndarray) -> pd.DataFrame:
    """
    Build and return a dataframe of media effect values.

    Args:
        data_to_fit: DataToFit instance
        media_effect_hat: array of media effect values, with axes:
           0: sample
           1: channel

    Returns:
        DataFrame of media effect values
    """
    blended_media_effect_hat = _compute_blended_media_effect_hat(media_effect_hat)
    blended_media_effect_quantiles = np.quantile(blended_media_effect_hat, [0.05, 0.95])

    media_effect_df = pd.DataFrame(
        index=["blended"] + data_to_fit.media_names, columns=["0.05", "0.95", "median", "mean"]
    )

    media_effect_df.loc["blended", "0.05"] = blended_media_effect_quantiles[0]
    media_effect_df.loc["blended", "0.95"] = blended_media_effect_quantiles[1]
    media_effect_df.loc["blended", "median"] = np.median(blended_media_effect_hat)
    media_effect_df.loc["blended", "mean"] = np.mean(blended_media_effect_hat)

    for media_idx in range(data_to_fit.media_data_train_scaled.shape[1]):
        quantiles = np.quantile(media_effect_hat[:, media_idx], [0.05, 0.95])
        mean = np.mean(media_effect_hat[:, media_idx])
        median = np.median(media_effect_hat[:, media_idx])
        media_name = data_to_fit.media_names[media_idx]
        media_effect_df.loc[media_name, "0.05"] = quantiles[0]
        media_effect_df.loc[media_name, "0.95"] = quantiles[1]
        media_effect_df.loc[media_name, "median"] = median
        media_effect_df.loc[media_name, "mean"] = mean

    return media_effect_df


def _compute_blended_roi_hat(
    data_to_fit: DataToFit, media_effect_hat: np.ndarray, roi_hat: np.ndarray
) -> np.ndarray:
    """
    Compute blended ROI hat.

    Args:
        data_to_fit: DataToFit instance
        media_effect_hat: array of media effect values, with axes:
           0: sample
           1: channel
        roi_hat: ndarray with axes
          0: samples
          1: channels

    Returns:
        ndarray with axes:
          0: samples
    """
    blended_media_effect_hat = _compute_blended_media_effect_hat(media_effect_hat)

    # target_sum_train_unscaled => scalar, unscaled sum of target values over the training period
    target_sum_train_unscaled = data_to_fit.target_scaler.inverse_transform(
        data_to_fit.target_train_scaled
    ).sum()
    # incremental_target_sum_hat => array with one dimension (sample count) representing the
    # unscaled total incremental target prediction over the training period
    incremental_target_sum_hat = blended_media_effect_hat * target_sum_train_unscaled
    # total_cost_train_unscaled => scalar, unscaled sum of media costs over the training period
    total_cost_train_unscaled = data_to_fit.media_costs_scaler.inverse_transform(
        data_to_fit.media_costs_by_row_train_scaled
    ).sum()
    # blended_roi_hat => array with one dimension (sample_count) representing the overall roi
    # across the training period (total target metric prediction attributable to media spend /
    # total media cost)
    blended_roi_hat = incremental_target_sum_hat / total_cost_train_unscaled
    return blended_roi_hat


def get_roi_df(
    data_to_fit: DataToFit, media_effect_hat: np.ndarray, roi_hat: np.ndarray
) -> pd.DataFrame:
    """
    Build and return a dataframe of ROI values.

    Args:
        data_to_fit: DataToFit instance
        media_effect_hat: array of media effect values, with axes:
           0: sample
           1: channel
        roi_hat: array of roi values, with axes:
          0: sample
          1: channel

    Returns:
        DataFrame of ROI values
    """
    blended_roi_hat = _compute_blended_roi_hat(data_to_fit, media_effect_hat, roi_hat)
    blended_roi_quantiles = np.quantile(blended_roi_hat, [0.05, 0.95])

    roi_df = pd.DataFrame(
        index=["blended"] + data_to_fit.media_names, columns=["0.05", "0.95", "median", "mean"]
    )

    roi_df.loc["blended", "0.05"] = blended_roi_quantiles[0]
    roi_df.loc["blended", "0.95"] = blended_roi_quantiles[1]
    roi_df.loc["blended", "median"] = np.median(blended_roi_hat)
    roi_df.loc["blended", "mean"] = np.mean(blended_roi_hat)

    for media_idx in range(data_to_fit.media_data_train_scaled.shape[1]):
        quantiles = np.quantile(roi_hat[:, media_idx], [0.05, 0.95])
        mean = np.mean(roi_hat[:, media_idx])
        median = np.median(roi_hat[:, media_idx])
        media_name = data_to_fit.media_names[media_idx]
        roi_df.loc[media_name, "0.05"] = quantiles[0]
        roi_df.loc[media_name, "0.95"] = quantiles[1]
        roi_df.loc[media_name, "median"] = median
        roi_df.loc[media_name, "mean"] = mean

    return roi_df


def _compute_blended_cost_per_target_hat(
    data_to_fit: DataToFit, media_effect_hat: np.ndarray, roi_hat: np.ndarray
) -> np.ndarray:
    """
    Compute blended cost per target hat

    Args:
        data_to_fit: DataToFit instance
        media_effect_hat: array of media effect values, with axes:
           0: sample
           1: channel
        roi_hat: array of roi values, with axes:
          0: sample
          1: channel

    Returns:
        ndarray of blended cost per target hat values with axes:
          0: samples

    """
    return 1.0 / _compute_blended_roi_hat(data_to_fit, media_effect_hat, roi_hat)


def get_cost_per_target_df(
    data_to_fit: DataToFit,
    media_effect_hat: np.ndarray,
    roi_hat: np.ndarray,
    cost_per_target_hat: np.ndarray,
) -> pd.DataFrame:
    """
    Build and return a dataframe of cost per target values.

    Args:
        data_to_fit: DataToFit instance
        media_effect_hat: array of media effect values, with axes:
           0: sample
           1: channel
        roi_hat: ndarray with axes
          0: samples
          1: channels
        cost_per_target_hat: cost per target metric values, with axes:
          0: sample
          1: channel

    Returns:
        DataFrame of cost per target values
    """
    blended_cost_per_target_hat = _compute_blended_cost_per_target_hat(
        data_to_fit, media_effect_hat, roi_hat
    )
    blended_cost_per_target_quantiles = np.quantile(blended_cost_per_target_hat, [0.05, 0.95])

    cost_per_target_df = pd.DataFrame(
        index=["blended"] + data_to_fit.media_names, columns=["0.05", "0.95", "median", "mean"]
    )

    cost_per_target_df.loc["blended", "0.05"] = blended_cost_per_target_quantiles[0]
    cost_per_target_df.loc["blended", "0.95"] = blended_cost_per_target_quantiles[1]
    cost_per_target_df.loc["blended", "median"] = np.median(blended_cost_per_target_hat)
    cost_per_target_df.loc["blended", "mean"] = np.mean(blended_cost_per_target_hat)

    for media_idx in range(data_to_fit.media_data_train_scaled.shape[1]):
        quantiles = np.quantile(cost_per_target_hat[:, media_idx], [0.05, 0.95])
        mean = np.mean(cost_per_target_hat[:, media_idx])
        median = np.median(cost_per_target_hat[:, media_idx])
        media_name = data_to_fit.media_names[media_idx]
        cost_per_target_df.loc[media_name, "0.05"] = quantiles[0]
        cost_per_target_df.loc[media_name, "0.95"] = quantiles[1]
        cost_per_target_df.loc[media_name, "median"] = median
        cost_per_target_df.loc[media_name, "mean"] = mean

    return cost_per_target_df


def _dump_posterior_metrics(
    results_dir: str,
    media_effect_df: pd.DataFrame,
    roi_df: pd.DataFrame,
    cost_per_target_df: pd.DataFrame,
):
    """
    Write posterior metrics to CSV files

    Args:
        results_dir: directory to write to
        media_effect_df: DataFrame of media effect values
        roi_df: DataFrame of ROI values
        cost_per_target_df: DataFrame of cost per target values

    Returns:
        None
    """
    media_effect_df.to_csv(os.path.join(results_dir, "media_performance_effect.csv"))
    roi_df.to_csv(os.path.join(results_dir, "media_performance_roi.csv"))
    cost_per_target_df.to_csv(os.path.join(results_dir, "media_performance_cost_per_target.csv"))


def get_baseline_breakdown_df(
    media_mix_model: lightweight_mmm.lightweight_mmm.LightweightMMM,
    input_data: InputData,
    data_to_fit: DataToFit,
    degrees_seasonality: int,
) -> pd.DataFrame:
    """
    Break down the baseline into its component pieces and return a DataFrame with the results.

    Args:
        media_mix_model: LightweightMMM instance
        input_data: InputData instance
        data_to_fit: DataToFit instance
        degrees_seasonality: Degrees of seasonality used to fit the model

    Returns:
        DataFrame with a breakdown of the baseline components for a given timestamp in each row.
    """
    num_observations = data_to_fit.media_data_train_scaled.shape[0]
    intercept = jnp.median(jnp.squeeze(media_mix_model.trace["intercept"]))
    coef_trend = jnp.median(jnp.squeeze(media_mix_model.trace["coef_trend"]))
    expo_trend = jnp.median(media_mix_model.trace["expo_trend"])
    if data_to_fit.extra_features_train_scaled.shape[1]:
        coef_extra_features = jnp.median(media_mix_model.trace["coef_extra_features"], axis=0)
    else:
        coef_extra_features = None
    gamma_seasonality = jnp.median(media_mix_model.trace["gamma_seasonality"], axis=0)

    columns = ["intercept", "trend", "seasonality"]

    if input_data.time_granularity == constants.GRANULARITY_DAILY:
        weekday = jnp.median(media_mix_model.trace["weekday"], axis=0)
        columns.append("weekday")
    else:
        weekday = None

    columns += data_to_fit.extra_features_names

    frequency = 365 if input_data.time_granularity == constants.GRANULARITY_DAILY else 52

    seasonality_by_obs = calculate_seasonality(
        number_periods=num_observations,
        degrees=degrees_seasonality,
        frequency=frequency,
        gamma_seasonality=gamma_seasonality,
    )

    data = np.zeros(shape=(num_observations, len(columns)))

    if data_to_fit.extra_features_train_scaled.shape[1]:
        extra_features_einsum = "tf, f -> tf"  # t = time, f = feature
        extra_features_mult = jnp.einsum(
            extra_features_einsum, data_to_fit.extra_features_train_scaled, coef_extra_features
        )
    else:
        extra_features_mult = None

    for i in range(num_observations):
        data[i, columns.index("intercept")] = intercept
        data[i, columns.index("trend")] = coef_trend * i**expo_trend
        data[i, columns.index("seasonality")] = seasonality_by_obs[i]
        if weekday is not None:
            data[i, columns.index("weekday")] = weekday[i % 7]
        for j in range(data_to_fit.extra_features_train_scaled.shape[1]):
            data[i, columns.index(data_to_fit.extra_features_names[j])] = extra_features_mult[i, j]

    data = data_to_fit.target_scaler.inverse_transform(data)

    baseline_breakdown_df = pd.DataFrame(data=data, columns=columns)
    baseline_breakdown_df["sum_of_medians"] = baseline_breakdown_df.sum(axis=1)

    # because the total baseline computed here in the 'sum' column is a less accurate estimate than
    # the one computed by create_media_baseline_contribution_df() for the weekly media and baseline
    # chart [1], we call the latter function here and include its estimate of the baseline
    # contribution in the results.
    #
    # [1] - this is because this function uses the median value of each model parameter, while
    # create_media_baseline_contribution_df allows the entire distribution to influence the result.
    plot_df = create_media_baseline_contribution_df(
        media_mix_model=media_mix_model,
        target_scaler=data_to_fit.target_scaler,
        channel_names=data_to_fit.media_names,
    )
    baseline_breakdown_df["chart_baseline_value"] = plot_df["baseline contribution"]

    baseline_breakdown_df["date"] = pd.to_datetime(data_to_fit.date_strs[:num_observations])
    baseline_breakdown_df = baseline_breakdown_df.set_index("date")

    return baseline_breakdown_df


def _dump_baseline_breakdown(
    results_dir: str,
    baseline_breakdown_df: pd.DataFrame,
):
    """
    Write the baseline breakdown to a file.

    Args:
        results_dir: Directory to write to
        baseline_breakdown_df: See get_baseline_breakdown_df

    Returns:
        None
    """
    baseline_breakdown_df.to_csv(os.path.join(results_dir, "baseline_breakdown.csv"))


def describe_mmm_training(
    mmm, input_data, data_to_fit, degrees_seasonality, results_dir, include_response_curves=False
):
    """
    Plot and print diagnostic analyses of the MMM training data.

    :param mmm: LightweightMMM instance
    :param input_data: InputData instance
    :param data_to_fit: DataToFit instance
    :param degrees_seasonality: degrees of seasonality used for fitting
    :param results_dir: directory to write plot files to
    :param include_response_curves: True to include response curves in the output, False otherwise.
        This is off by default because it is quite slow and appears to leak memory.

    :return: none
    """
    output_fname = os.path.join(results_dir, "model_coefficients.txt")
    with open(output_fname, "w") as f:
        with redirect_stdout(f):
            mmm.print_summary()

    fig = plot_model_fit(media_mix_model=mmm, target_scaler=data_to_fit.target_scaler)
    output_fname = os.path.join(results_dir, "model_fit_in_sample.png")
    fig.savefig(output_fname, bbox_inches="tight")

    fig = plot_media_channel_posteriors(media_mix_model=mmm, channel_names=data_to_fit.media_names)
    output_fname = os.path.join(results_dir, "model_media_posteriors.png")
    fig.savefig(output_fname, bbox_inches="tight")

    costs_per_day_unscaled = data_to_fit.media_costs_scaler.inverse_transform(
        data_to_fit.media_costs_by_row_train_scaled
    )
    if include_response_curves:
        num_pages = math.ceil(len(input_data.media_names) / 5)
        fig = plot_response_curves(
            media_mix_model=mmm,
            media_scaler=data_to_fit.media_scaler,
            target_scaler=data_to_fit.target_scaler,
            figure_size=(8, 10 * num_pages),
            costs_per_day=costs_per_day_unscaled,
            percentage_add=0.0,
            response_metric="target",
        )
        output_fname = os.path.join(results_dir, "response_curves_target.png")
        fig.savefig(output_fname, bbox_inches="tight")

        fig = plot_response_curves(
            media_mix_model=mmm,
            media_scaler=data_to_fit.media_scaler,
            target_scaler=data_to_fit.target_scaler,
            figure_size=(8, 10 * num_pages),
            costs_per_day=costs_per_day_unscaled,
            percentage_add=0.0,
            response_metric="cost_per_target",
        )
        output_fname = os.path.join(results_dir, "response_curves_cost_per_target.png")
        fig.savefig(output_fname, bbox_inches="tight")

    fig = plot_prior_and_posterior(media_mix_model=mmm)
    output_fname = os.path.join(results_dir, "model_priors_and_posteriors.png")
    fig.savefig(output_fname, bbox_inches="tight")

    media_effect_hat, roi_hat = mmm.get_posterior_metrics(
        unscaled_costs=costs_per_day_unscaled.sum(axis=0), target_scaler=data_to_fit.target_scaler
    )
    cost_per_target_hat = 1.0 / roi_hat

    _dump_posterior_metrics(
        results_dir,
        get_media_effect_df(data_to_fit, media_effect_hat),
        get_roi_df(data_to_fit, media_effect_hat, roi_hat),
        get_cost_per_target_df(data_to_fit, media_effect_hat, roi_hat, cost_per_target_hat),
    )

    fig = plot_bars_media_metrics(
        metric=media_effect_hat,
        metric_name="contribution percentage",
        channel_names=data_to_fit.media_names,
        bar_height="mean",
    )
    output_fname = os.path.join(results_dir, "media_contribution_mean.png")
    fig.savefig(output_fname, bbox_inches="tight")

    fig = plot_bars_media_metrics(
        metric=media_effect_hat,
        metric_name="contribution percentage",
        channel_names=data_to_fit.media_names,
        bar_height="median",
    )
    output_fname = os.path.join(results_dir, "media_contribution_median.png")
    fig.savefig(output_fname, bbox_inches="tight")

    fig = plot_bars_media_metrics(
        metric=roi_hat, metric_name="ROI", channel_names=data_to_fit.media_names, bar_height="mean"
    )
    output_fname = os.path.join(results_dir, "media_roi_mean.png")
    fig.savefig(output_fname, bbox_inches="tight")

    fig = plot_bars_media_metrics(
        metric=roi_hat,
        metric_name="ROI",
        channel_names=data_to_fit.media_names,
        bar_height="median",
    )
    output_fname = os.path.join(results_dir, "media_roi_median.png")
    fig.savefig(output_fname, bbox_inches="tight")

    fig = plot_bars_media_metrics(
        metric=cost_per_target_hat,
        metric_name="cost per target",
        channel_names=data_to_fit.media_names,
        bar_height="mean",
    )
    output_fname = os.path.join(results_dir, "media_cost_per_target_mean.png")
    fig.savefig(output_fname, bbox_inches="tight")

    fig = plot_bars_media_metrics(
        metric=cost_per_target_hat,
        metric_name="cost per target",
        channel_names=data_to_fit.media_names,
        bar_height="median",
    )
    output_fname = os.path.join(results_dir, "media_cost_per_target_median.png")
    fig.savefig(output_fname, bbox_inches="tight")

    _dump_baseline_breakdown(
        results_dir, get_baseline_breakdown_df(mmm, input_data, data_to_fit, degrees_seasonality)
    )

    fig = plot_media_baseline_contribution_area_plot(
        media_mix_model=mmm,
        target_scaler=data_to_fit.target_scaler,
        channel_names=data_to_fit.media_names,
    )
    output_fname = os.path.join(results_dir, "weekly_media_and_baseline_contribution.png")
    fig.savefig(output_fname, bbox_inches="tight")


def describe_mmm_prediction(mmm, data_to_fit, results_dir):
    """
    Plot and print diagnostic analyses of the mmm's predictive ability based on the test data

    :param mmm: LightweightMMM instance
    :param data_to_fit:  DataToFit instance
    :param results_dir: Directory to write plots to
    :return: None
    """
    if data_to_fit.extra_features_test_scaled.shape[1] == 0:
        extra_features = None
    else:
        extra_features = data_to_fit.extra_features_test_scaled

    prediction = mmm.predict(
        media=data_to_fit.media_data_test_scaled,
        extra_features=extra_features,
        target_scaler=data_to_fit.target_scaler,
    )

    if data_to_fit.has_test_dataset:
        target_test_unscaled = data_to_fit.target_scaler.inverse_transform(
            data_to_fit.target_test_scaled
        )
        fig = plot_out_of_sample_model_fit(
            out_of_sample_predictions=prediction,
            out_of_sample_target=target_test_unscaled,
            media_mix_model=mmm,
        )
        output_fname = os.path.join(results_dir, "model_fit_out_of_sample.png")
        fig.savefig(output_fname, bbox_inches="tight")
