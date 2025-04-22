from contextlib import redirect_stdout
import json
import math
from typing import TypedDict
from customer_reports.data.historical_predict import create_historical_predictions_daily_df
from lightweight_mmm.lightweight_mmm import LightweightMMM
import numpy as np
import jax.numpy as jnp
import pandas as pd
import os

# from numpyro.diagnostics import summary
import numpyro.diagnostics
from sklearn import metrics

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

# Type hints
Coefficients = TypedDict(
    "Coefficients", {"intercept": float, "coef_trend": float, "expo_trend": float}
)

MediaMedians = TypedDict("MediaMedians", {"blended_median": float, "top_medians": list[float]})
Media = TypedDict(
    "Media",
    {"effect": MediaMedians, "roi": MediaMedians, "cost_per_target": MediaMedians},
)


def describe_input_data(input_data, results_dir, suffix):
    """
    plot and print diagnostic analyses based on the input data
    :param input_data: InputData instance
    :param results_dir: directory to write plot files to
    :param suffix: suffix to append to filename
    :return:
    """
    # I'm being lazy and haven't made these functions support geos
    if input_data.geo_names is None:
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
           2: geo (only present if there is geo data in the input)

    Returns:
        Array of blended media effect values, with axes:
          0: sample
          1: geo (only present if there is geo data in the input)
    """
    # blended_media_effect_hat => array with dimensions
    #   0: sample
    #   1: geo (only present if there is geo data in the input)
    # representing the overall media effect across the training period, as a percentage of the
    # total target prediction.
    blended_media_effect_hat = media_effect_hat.sum(axis=1)

    return blended_media_effect_hat


def _get_summary_df(
    data_to_fit: DataToFit,
    media_distributions: np.ndarray,
    blended_distribution: np.ndarray,
    geo_name: str,
) -> pd.DataFrame:
    """
    Get a summary dataframe containing credibility interval quantiles, mean, and median values
    for each media as well as the blended media value.

    Args:
        data_to_fit: DataToFit instance
        media_distributions: array of media values (media effect, roi, or cost per target),
            with axes:
            0: sample
            1: channel
            2: geo (only present if there is geo data in the input)
        blended_distribution: array of blended media values (media effect, roi, or cost per target),
            with axes:
            0: sample
            1: geo (only present if there is geo data in the input)
        geo_name: geo name to filter on (for a geo model), or None for a global model.  Summarizing
            across geos in a geo model is not yet supported.

    Returns:
        Dataframe with credibility interval quantiles, mean, and median values
        Axes:
            0: channel
            1: <ci_lower_quantile>, <ci_upper_quantile>, median, mean
    """
    if geo_name is not None and data_to_fit.geo_names is None:
        raise ValueError("geo_name passed for a global model")

    if geo_name is None and data_to_fit.geo_names is not None:
        raise ValueError("summarizing across geos is not supported")

    expected_media_dims = 2 if data_to_fit.geo_names is None else 3
    assert expected_media_dims == media_distributions.ndim

    ci_quantiles = data_to_fit.get_ci_quantiles()

    columns = [str(x) for x in ci_quantiles]
    summary_df = pd.DataFrame(
        index=["blended"] + data_to_fit.media_names, columns=columns + ["median", "mean"]
    )

    if data_to_fit.geo_names is not None:
        geo_idx = data_to_fit.geo_names.index(geo_name)
        if geo_idx < 0:
            raise ValueError(f"invalid geo: {geo_name}")
        media_distributions_touse = media_distributions[:, :, geo_idx]
        blended_distribution_touse = blended_distribution[:, geo_idx]
    else:
        media_distributions_touse = media_distributions
        blended_distribution_touse = blended_distribution

    # at this point the geo dimension should be gone
    assert 2 == media_distributions_touse.ndim
    assert 1 == blended_distribution_touse.ndim

    quantiles = np.quantile(blended_distribution_touse, ci_quantiles)

    for idx, column in enumerate(columns):
        summary_df.loc["blended", column] = quantiles[idx]
    summary_df.loc["blended", "median"] = np.median(blended_distribution_touse)
    summary_df.loc["blended", "mean"] = np.mean(blended_distribution_touse)

    for media_idx in range(data_to_fit.media_data_train_scaled.shape[1]):
        quantiles = np.quantile(media_distributions_touse[:, media_idx], ci_quantiles)
        media_name = data_to_fit.media_names[media_idx]

        for idx, column in enumerate(columns):
            summary_df.loc[media_name, column] = quantiles[idx]
        summary_df.loc[media_name, "median"] = np.median(media_distributions_touse[:, media_idx])
        summary_df.loc[media_name, "mean"] = np.mean(media_distributions_touse[:, media_idx])

    return summary_df


def get_media_effect_df(
    data_to_fit: DataToFit,
    media_effect_hat: np.ndarray,
    geo_name: str = None,
) -> pd.DataFrame:
    """
    Build and return a dataframe of media effect values.

    Args:
        data_to_fit: DataToFit instance
        media_effect_hat: array of media effect values, with axes:
           0: sample
           1: channel
           2: geo (only present when there is geo data in the input)
        geo_name: geo name to filter on (for a geo model), or None for a global model.  Summarizing
            across geos in a geo model is not yet supported.

    Returns:
        DataFrame of media effect values
    """

    return _get_summary_df(
        data_to_fit,
        media_effect_hat,
        _compute_blended_media_effect_hat(media_effect_hat),
        geo_name,
    )


def _compute_blended_roi_hat(
    data_to_fit: DataToFit,
    media_effect_hat: np.ndarray,
    roi_hat: np.ndarray,
) -> np.ndarray:
    """
    Compute blended ROI hat.

    Args:
        data_to_fit: DataToFit instance
        media_effect_hat: array of media effect values, with axes:
           0: sample
           1: channel
           2: geo (geo models only)
        roi_hat: ndarray with axes
          0: samples
          1: channels
          2: geo (geo models only)

    Returns:
        ndarray with axes:
          0: samples
          1: geo (geo models only)
    """
    # blended_media_effect_hat has axes:
    #   0: samples
    #   1: geo (geo models only)
    blended_media_effect_hat = _compute_blended_media_effect_hat(media_effect_hat)

    # target_sum_train_unscaled => unscaled sum of target values over the training period.  For
    # global models this is a scalar value, and for geo models it is an array with dims [geo].
    target_sum_train_unscaled = data_to_fit.target_scaler.inverse_transform(
        data_to_fit.target_train_scaled
    ).sum(axis=0)

    # incremental_target_sum_hat => array with dimensions
    #   sample count
    #   geo (geo models only)
    #
    # representing the unscaled total incremental target prediction over the training period.
    incremental_target_sum_hat = blended_media_effect_hat * target_sum_train_unscaled

    # total_cost_train_unscaled => array with dimensions [geo] for geo models or a scalar for
    # global models.  The values are the unscaled sum of media costs over the training period.
    # media_costs_by_row_train_unscaled has dims [time, channel, geo] or [time, channel]
    media_costs_by_row_train_unscaled = data_to_fit.media_costs_scaler.inverse_transform(
        data_to_fit.media_costs_by_row_train_scaled
    )
    total_cost_train_unscaled = media_costs_by_row_train_unscaled.sum(axis=(0, 1))

    # blended_roi_hat => array with dimensions [sample_count, geo] or [sample_count] representing
    # the overall roi across the training period (total target metric prediction attributable
    # to media spend / total media cost)
    blended_roi_hat = incremental_target_sum_hat / total_cost_train_unscaled
    return blended_roi_hat


def get_roi_df(
    data_to_fit: DataToFit,
    media_effect_hat: np.ndarray,
    roi_hat: np.ndarray,
    geo_name: str = None,
) -> pd.DataFrame:
    """
    Build and return a dataframe of ROI values.

    Args:
        data_to_fit: DataToFit instance
        media_effect_hat: array of media effect values, with axes:
           0: sample
           1: channel
           2: geo (geo models only)
        roi_hat: array of roi values, with axes:
           0: sample
           1: channel
           2: geo (geo models only)
        geo_name: geo name to filter on (for a geo model), or None for a global model.  Summarizing
           across geos in a geo model is not yet supported.

    Returns:
        DataFrame of ROI values
    """

    return _get_summary_df(
        data_to_fit,
        roi_hat,
        _compute_blended_roi_hat(data_to_fit, media_effect_hat, roi_hat),
        geo_name,
    )


def _compute_blended_cost_per_target_hat(
    data_to_fit: DataToFit,
    media_effect_hat: np.ndarray,
    roi_hat: np.ndarray,
) -> np.ndarray:
    """
    Compute blended cost per target hat

    Args:
        data_to_fit: DataToFit instance
        media_effect_hat: array of media effect values, with axes:
           0: sample
           1: channel
           2: geo (geo models only)
        roi_hat: array of roi values, with axes:
           0: sample
           1: channel
           2: geo (geo models only)

    Returns:
        ndarray of blended cost per target hat values with axes:
           0: sample
           1: geo (geo models only)

    """
    return 1.0 / _compute_blended_roi_hat(data_to_fit, media_effect_hat, roi_hat)


def get_cost_per_target_df(
    data_to_fit: DataToFit,
    media_effect_hat: np.ndarray,
    roi_hat: np.ndarray,
    cost_per_target_hat: np.ndarray,
    geo_name: str = None,
) -> pd.DataFrame:
    """
    Build and return a dataframe of cost per target values.

    Args:
        data_to_fit: DataToFit instance
        media_effect_hat: array of media effect values, with axes:
           0: sample
           1: channel
           2: geo (geo models only)
        roi_hat: ndarray with axes
           0: samples
           1: channels
           2: geo (geo models only)
        cost_per_target_hat: cost per target metric values, with axes:
           0: sample
           1: channel
           2: geo (geo models only)
        geo_name: geo name to filter on (for a geo model), or None for a global model.  Summarizing
           across geos in a geo model is not yet supported.

    Returns:
        DataFrame of cost per target values
    """

    return _get_summary_df(
        data_to_fit,
        cost_per_target_hat,
        _compute_blended_cost_per_target_hat(data_to_fit, media_effect_hat, roi_hat),
        geo_name,
    )


def get_baseline_breakdown_df(
    media_mix_model: LightweightMMM,
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
    if input_data.geo_names is not None:
        # This function doesn't work for geo models because (in part at least) the target scaler
        # has one scaler per geo
        raise ValueError("unsupported operation")

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

    if data_to_fit.time_granularity == constants.GRANULARITY_DAILY:
        seasonality_frequency = 365
    elif data_to_fit.time_granularity == constants.GRANULARITY_WEEKLY:
        seasonality_frequency = 52
    elif data_to_fit.time_granularity == constants.GRANULARITY_TWO_WEEKS:
        seasonality_frequency = 26
    elif data_to_fit.time_granularity == constants.GRANULARITY_FOUR_WEEKS:
        seasonality_frequency = 13
    else:
        assert False

    seasonality_by_obs = calculate_seasonality(
        number_periods=num_observations,
        degrees=degrees_seasonality,
        frequency=seasonality_frequency,
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

    # TODO: this doesn't work for geo models because the scaler has one value per geo, and the
    # data array here has dimensions [time, baseline component].  Maybe adding a geo dimension to
    # the end would solve this?
    # e.g. for a 4 geo case I got the error ValueError: Incompatible shapes for broadcasting: shapes=[(4,), (71, 3)]
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


def _extract_and_dump_coefficients(
    mmm: LightweightMMM,
    data_to_fit: DataToFit,
    results_dir: str,
) -> Coefficients:
    """
    Extract and return a summary of coefficients data. Also dump the full data into a file.

    :param mmm: LightweightMMM instance
    :param data_to_fit: DataToFit instance
    :param results_dir: directory to write plot files to

    :return: Summary of coefficients data.
    """

    output_fname = os.path.join(results_dir, "model_coefficients.txt")
    with open(output_fname, "w") as f:
        with redirect_stdout(f):
            mmm._mcmc.print_summary(prob=data_to_fit.credibility_interval)

    # This summary calculation actually happens inside print_summary too, but print_summary does not
    # return the values so we have to run these ourselves.
    samples = mmm._mcmc.get_samples(group_by_chain=True)
    summary = numpyro.diagnostics.summary(
        samples, data_to_fit.credibility_interval, group_by_chain=True
    )

    # Convert type from float32 to float so they're compatible with json
    # TODO need to make this work with geo models
    if data_to_fit.geo_names is None:
        return {
            "intercept": float(summary["intercept"]["median"][0]),
            "coef_trend": float(summary["coef_trend"]["median"][0]),
            "expo_trend": float(summary["expo_trend"]["median"]),
        }
    else:
        return {}


def _extract_and_plot_fit(mmm: LightweightMMM, data_to_fit: DataToFit, results_dir: str) -> float:
    """
    Extract and return the fit mape. Also plot the fit on a chart file.

    :param mmm: LightweightMMM instance
    :param data_to_fit: DataToFit instance
    :param results_dir: directory to write plot files to

    :return: Fit mape.

    """
    fig = plot_model_fit(
        media_mix_model=mmm,
        target_scaler=data_to_fit.target_scaler,
        interval_mid_range=data_to_fit.credibility_interval,
    )
    output_fname = os.path.join(results_dir, "model_fit_in_sample.png")
    fig.savefig(output_fname, bbox_inches="tight")

    # Logic to calculate mape extracted from plot_model_fit
    actual = mmm._target
    prediction = mmm.trace["mu"]
    if data_to_fit.target_scaler:
        prediction = data_to_fit.target_scaler.inverse_transform(prediction)
        actual = data_to_fit.target_scaler.inverse_transform(actual)

    y_true = actual
    y_pred = prediction

    if mmm._target_is_log_scale:
        y_true = jnp.exp(y_true)
        y_pred = jnp.exp(y_pred)

    mape = 100 * metrics.mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred.mean(axis=0))

    return mape


def _plot_media(
    mmm: LightweightMMM,
    data_to_fit: DataToFit,
    results_dir: str,
    media_effect_hat: np.ndarray,
    roi_hat: np.ndarray,
    cost_per_target_hat: np.ndarray,
) -> None:
    # Posteriors
    ci_lower_quantile, ci_upper_quantile = data_to_fit.get_ci_quantiles()
    fig = plot_media_channel_posteriors(
        media_mix_model=mmm,
        channel_names=data_to_fit.media_names,
        quantiles=[ci_lower_quantile, 0.5, ci_upper_quantile],
    )
    output_fname = os.path.join(results_dir, "model_media_posteriors.png")
    fig.savefig(output_fname, bbox_inches="tight")

    # Priors and posteriors
    fig = plot_prior_and_posterior(media_mix_model=mmm)
    output_fname = os.path.join(results_dir, "model_priors_and_posteriors.png")
    fig.savefig(output_fname, bbox_inches="tight")

    # For geo models, plot_bars_media_metrics plots the mean across all geos.  We're not including
    # these at present because I'm considering these not useful for a small number of geos.
    if data_to_fit.geo_names is None:
        # Bar charts
        fig = plot_bars_media_metrics(
            metric=media_effect_hat,
            metric_name="contribution percentage",
            channel_names=data_to_fit.media_names,
            interval_mid_range=data_to_fit.credibility_interval,
            bar_height="mean",
        )
        output_fname = os.path.join(results_dir, "media_contribution_mean.png")
        fig.savefig(output_fname, bbox_inches="tight")

        fig = plot_bars_media_metrics(
            metric=media_effect_hat,
            metric_name="contribution percentage",
            channel_names=data_to_fit.media_names,
            interval_mid_range=data_to_fit.credibility_interval,
            bar_height="median",
        )
        output_fname = os.path.join(results_dir, "media_contribution_median.png")
        fig.savefig(output_fname, bbox_inches="tight")

        fig = plot_bars_media_metrics(
            metric=roi_hat,
            metric_name="ROI",
            channel_names=data_to_fit.media_names,
            interval_mid_range=data_to_fit.credibility_interval,
            bar_height="mean",
        )
        output_fname = os.path.join(results_dir, "media_roi_mean.png")
        fig.savefig(output_fname, bbox_inches="tight")

        fig = plot_bars_media_metrics(
            metric=roi_hat,
            metric_name="ROI",
            channel_names=data_to_fit.media_names,
            interval_mid_range=data_to_fit.credibility_interval,
            bar_height="median",
        )
        output_fname = os.path.join(results_dir, "media_roi_median.png")
        fig.savefig(output_fname, bbox_inches="tight")

        fig = plot_bars_media_metrics(
            metric=cost_per_target_hat,
            metric_name="cost per target",
            channel_names=data_to_fit.media_names,
            interval_mid_range=data_to_fit.credibility_interval,
            bar_height="mean",
        )
        output_fname = os.path.join(results_dir, "media_cost_per_target_mean.png")
        fig.savefig(output_fname, bbox_inches="tight")

        fig = plot_bars_media_metrics(
            metric=cost_per_target_hat,
            metric_name="cost per target",
            channel_names=data_to_fit.media_names,
            interval_mid_range=data_to_fit.credibility_interval,
            bar_height="median",
        )
        output_fname = os.path.join(results_dir, "media_cost_per_target_median.png")
        fig.savefig(output_fname, bbox_inches="tight")

    fig = plot_media_baseline_contribution_area_plot(
        media_mix_model=mmm,
        target_scaler=data_to_fit.target_scaler,
        channel_names=data_to_fit.media_names,
    )
    output_fname = os.path.join(results_dir, "weekly_media_and_baseline_contribution.png")
    fig.savefig(output_fname, bbox_inches="tight")


def _plot_response_curves(
    mmm: LightweightMMM,
    data_to_fit: DataToFit,
    results_dir: str,
    costs_per_day_unscaled: np.ndarray,
    input_data: InputData,
) -> None:
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


def _extract_and_dump_media(
    mmm: LightweightMMM,
    input_data: InputData,
    data_to_fit: DataToFit,
    results_dir: str,
    include_response_curves=False,
    include_plot_media=True,
) -> Media:
    """
    Extract and return a summary of the media effect, roi, and cost per target. Also dump the full media data
    into different files as well as plot various charts.

    :param mmm: LightweightMMM instance
    :param input_data: InputData instance
    :param data_to_fit: DataToFit instance
    :param results_dir: directory to write plot files to
    :param include_response_curves: True to include response curves in the output, False otherwise.
        This is off by default because it is quite slow and appears to leak memory.
    :param include_plot_media: True to plot media, False otherwise

    :return: Summary of media effect, roi, and cost per target across all geos
       as dict with keys -> values
       "effect" -> { "blended_median" -> n, "top_medians" -> [] }
       "roi" -> { "blended_median" -> n, "top_medians" -> [] }
       "cost_per_target" -> { "blended_median" -> n, "top_medians" -> [] }

    """

    # Calculate variables
    # costs_per_day_unscaled has dimensions:
    #   0: time
    #   1: channel
    #   2: geo (geo models only)
    costs_per_day_unscaled = data_to_fit.media_costs_scaler.inverse_transform(
        data_to_fit.media_costs_by_row_train_scaled
    )

    media_effect_hat, roi_hat = mmm.get_posterior_metrics(
        unscaled_costs=costs_per_day_unscaled.sum(axis=0),
        target_scaler=data_to_fit.target_scaler,
    )
    cost_per_target_hat = 1.0 / roi_hat

    # Summarize media effect, roi, cost per target across all geos
    if data_to_fit.geo_names is None:
        global_media_effect_df = get_media_effect_df(
            data_to_fit,
            media_effect_hat,
            geo_name=None,
        )
        global_roi_df = get_roi_df(
            data_to_fit,
            media_effect_hat,
            roi_hat,
            geo_name=None,
        )
        global_cost_per_target_df = get_cost_per_target_df(
            data_to_fit,
            media_effect_hat,
            roi_hat,
            cost_per_target_hat,
            geo_name=None,
        )

        # Write to CSVs
        global_media_effect_df.to_csv(os.path.join(results_dir, "media_performance_effect.csv"))
        global_roi_df.to_csv(os.path.join(results_dir, "media_performance_roi.csv"))
        global_cost_per_target_df.to_csv(
            os.path.join(results_dir, "media_performance_cost_per_target.csv")
        )
    else:
        # We don't yet support summarizing media effect, roi, cost per target across geos
        # because get_posterior_metrics can't do this.
        global_media_effect_df = None
        global_roi_df = None
        global_cost_per_target_df = None

        # Summarize media effect, roi, cost per target for each geo
        for geo_name in data_to_fit.geo_names:
            geo_media_effect_df = get_media_effect_df(
                data_to_fit,
                media_effect_hat,
                geo_name=geo_name,
            )
            geo_roi_df = get_roi_df(
                data_to_fit,
                media_effect_hat,
                roi_hat,
                geo_name=geo_name,
            )
            geo_cost_per_target_df = get_cost_per_target_df(
                data_to_fit,
                media_effect_hat,
                roi_hat,
                cost_per_target_hat,
                geo_name=geo_name,
            )

            geo_media_effect_df.to_csv(
                os.path.join(results_dir, f"media_performance_effect_{geo_name}.csv")
            )
            geo_roi_df.to_csv(os.path.join(results_dir, f"media_performance_roi_{geo_name}.csv"))
            geo_cost_per_target_df.to_csv(
                os.path.join(results_dir, f"media_performance_cost_per_target_{geo_name}.csv")
            )

    # Plot media graphs
    if include_plot_media:
        _plot_media(mmm, data_to_fit, results_dir, media_effect_hat, roi_hat, cost_per_target_hat)
    if include_response_curves:
        _plot_response_curves(mmm, data_to_fit, results_dir, costs_per_day_unscaled, input_data)

    # Summarise global media results
    # Convert type from float32 to float so they're compatible with json
    if data_to_fit.geo_names is None:
        media_effect_median = global_media_effect_df["median"].astype(float)
        roi_median = global_roi_df["median"].astype(float)
        cost_per_target_median = global_cost_per_target_df["median"].astype(float)
        media = {
            "effect": {
                "blended_median": media_effect_median.get("blended"),
                "top_medians": media_effect_median.astype(float)
                .drop("blended")
                .sort_values(ascending=False)
                .head(3)
                .to_dict(),
            },
            "roi": {
                "blended_median": roi_median.get("blended"),
                "top_medians": roi_median.drop("blended")
                # Remove inf
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .sort_values(ascending=False)
                .head(3)
                .to_dict(),
            },
            "cost_per_target": {
                "blended_median": cost_per_target_median.get("blended"),
                # Remove zero values
                "top_medians": cost_per_target_median[cost_per_target_median > 0]
                .drop("blended")
                .sort_values(ascending=True)
                .head(3)
                .to_dict(),
            },
        }
    else:
        # We don't yet support summarizing media effect, roi, cost per target across geos
        # because get_posterior_metrics can't do this.
        media = {}

    return media


def _extract_and_dump_baseline(
    mmm: LightweightMMM,
    input_data: InputData,
    data_to_fit: DataToFit,
    degrees_seasonality: int,
    results_dir: str,
) -> pd.DataFrame:
    """
    Extract and return baseline breakdown dataframe. Also write it to a file.

    :param mmm: LightweightMMM instance
    :param input_data: InputData instance
    :param data_to_fit: DataToFit instance
    :param degrees_seasonality: degrees of seasonality used for fitting
    :param results_dir: directory to write plot files to

    :return: Baseline breakdown dataframe.
    """
    baseline_breakdown_df = get_baseline_breakdown_df(
        mmm, input_data, data_to_fit, degrees_seasonality
    )
    baseline_breakdown_df.to_csv(os.path.join(results_dir, "baseline_breakdown.csv"))

    return baseline_breakdown_df


def describe_mmm_training(
    mmm: LightweightMMM,
    input_data: InputData,
    data_to_fit: DataToFit,
    degrees_seasonality: int,
    results_dir: str,
    include_response_curves=False,
    config: dict = None,
) -> dict:
    """
    Plot and print diagnostic analyses of the MMM training data.

    :param mmm: LightweightMMM instance
    :param input_data: InputData instance
    :param data_to_fit: DataToFit instance
    :param degrees_seasonality: degrees of seasonality used for fitting
    :param results_dir: directory to write plot files to
    :param include_response_curves: True to include response curves in the output, False otherwise.
        This is off by default because it is quite slow and appears to leak memory.
    :param config: Configuration dictionary containing model settings

    :return: Summary of mmm training, e.g. coefficients, fit mape, media effect, etc.
    """

    coefficients = _extract_and_dump_coefficients(mmm, data_to_fit, results_dir)
    fit_mape = _extract_and_plot_fit(mmm, data_to_fit, results_dir)
    media = _extract_and_dump_media(
        mmm, input_data, data_to_fit, results_dir, include_response_curves
    )
    if input_data.geo_names is None:
        # see comment in get_baseline_breakdown_df
        baseline = _extract_and_dump_baseline(
            mmm, input_data, data_to_fit, degrees_seasonality, results_dir
        )
    else:
        baseline = None

    # XXX for now do not generate fit data for hierarchical models
    if data_to_fit.geo_names is None:
        # Get and save fit data
        fit_df = get_fit_in_sample_df(
            media_mix_model=mmm,
            data_to_fit=data_to_fit,
            input_data=input_data,
            geo_name=None,  # For now, we don't support geo filtering in describe_mmm_training
            time_granularity=None,  # Use original time granularity
        )
        fit_df.to_csv(os.path.join(results_dir, "fit_data.csv"))

    # XXX for now do not generate fit data for hierarchical models
    if data_to_fit.geo_names is None:
        # Get and save media and baseline contribution data
        media_baseline_df = get_media_and_baseline_contribution_df(
            media_mix_model=mmm,
            data_to_fit=data_to_fit,
            config=config,
        )
        media_baseline_df.to_csv(os.path.join(results_dir, "media_baseline_contribution.csv"))

    summary = {
        "coefficients": coefficients,
        "fit_mape": fit_mape,
        "media": media,
    }
    if baseline is not None:
        summary["has_negative_baseline"] = bool((baseline["sum_of_medians"] < 0).any())

    # Write summary to a json file
    summary_file = open(os.path.join(results_dir, "summary.json"), "w")
    summary_file.write(json.dumps(summary, indent=2))
    summary_file.close()

    return summary


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


def get_fit_in_sample_df(
    media_mix_model: LightweightMMM,
    data_to_fit: DataToFit,
    input_data: InputData,
    geo_name: str = None,
    time_granularity: str = None,
) -> pd.DataFrame:
    """
    Create a DataFrame containing the actual and predicted values for a single model.

    Args:
        media_mix_model: LightweightMMM instance
        data_to_fit: DataToFit instance
        input_data: InputData instance containing the original data
        geo_name: Optional geo name to filter on. If None, uses all geos.
        time_granularity: Optional time granularity to resample the data to. One of:
            "week", "two_weeks", "four_weeks". If None, uses the original time granularity.

    Returns:
        DataFrame with columns:
            - date: datetime column
            - actual: actual target values
            - predicted:median: median predicted values
            - predicted:lower: lower bound of predicted values (5th percentile)
            - predicted:upper: upper bound of predicted values (95th percentile)
    """
    if not hasattr(media_mix_model, "trace"):
        raise lightweight_mmm.NotFittedModelError(
            "Model needs to be fit first before attempting to get fit data."
        )

    if data_to_fit.geo_names is not None and geo_name is None:
        raise NotImplementedError("Getting aggregate fit across geos is not supported")

    if geo_name is not None:
        geo_idx = data_to_fit.geo_names.index(geo_name)
    else:
        geo_idx = -1

    # Get posterior predictions
    posterior_pred = media_mix_model.trace["mu"]
    target_scaler = data_to_fit.target_scaler
    posterior_pred = target_scaler.inverse_transform(posterior_pred)

    if -1 != geo_idx:
        posterior_pred = posterior_pred[:, :, geo_idx]

    # Calculate summary statistics
    predictions_median = np.median(posterior_pred, axis=0)
    predictions_lower = np.percentile(posterior_pred, 5, axis=0)
    predictions_upper = np.percentile(posterior_pred, 95, axis=0)

    # Create DataFrame with datetime index
    df = pd.DataFrame()
    df.index = pd.to_datetime(data_to_fit.date_strs[: len(predictions_median)])
    df["date"] = df.index  # Add date as a column

    # Add actual values
    if data_to_fit.time_granularity == constants.GRANULARITY_DAILY:
        # For daily data, use the target data directly
        if geo_idx != -1:
            df["actual"] = input_data.target_data[:, geo_idx]
        else:
            df["actual"] = input_data.target_data
    else:
        # For non-daily data, we need to resample
        time_granularity_to_resample_rule = {
            constants.GRANULARITY_WEEKLY: "W",
            constants.GRANULARITY_TWO_WEEKS: "2W",
            constants.GRANULARITY_FOUR_WEEKS: "4W",
        }
        # Create a temporary daily DataFrame
        daily_df = pd.DataFrame(index=pd.to_datetime(input_data.date_strs))
        if geo_idx != -1:
            daily_df["target"] = input_data.target_data[:, geo_idx]
        else:
            daily_df["target"] = input_data.target_data
        # Resample to the desired granularity
        df["actual"] = daily_df.resample(
            time_granularity_to_resample_rule[data_to_fit.time_granularity],
            closed="left",
            label="left",
        ).sum()["target"]

    # Add predicted values
    df["predicted:median"] = predictions_median
    df["predicted:lower"] = predictions_lower
    df["predicted:upper"] = predictions_upper

    # Resample if requested
    if time_granularity is not None:
        data_groupby_to_resample_mode = {
            "week": "W",
            "two_weeks": "2W",
            "four_weeks": "4W",
        }
        mode = data_groupby_to_resample_mode[time_granularity]
        df = df.resample(mode, label="left", closed="left").sum()
        df["date"] = df.index  # Re-add date as column after resampling

    return df


def get_media_and_baseline_contribution_df(
    media_mix_model: LightweightMMM,
    data_to_fit: DataToFit,
    config: dict,
) -> pd.DataFrame:
    """
    Create a DataFrame containing the time series of media and baseline contributions.
    This version follows the same flow as plot_media_and_baseline but returns a DataFrame instead of plotting.

    Args:
        media_mix_model: LightweightMMM instance
        data_to_fit: DataToFit instance
        config: yaml config

    Returns:
        DataFrame with columns:
            - date: datetime column
            - baseline:median: baseline contribution
            - {channel_name}:median: contribution for each media channel
    """
    if not hasattr(media_mix_model, "trace"):
        raise lightweight_mmm.NotFittedModelError(
            "Model needs to be fit first before attempting to get media and baseline contributions."
        )

    daily_predictions_df = create_historical_predictions_daily_df(
        config, media_mix_model, data_to_fit
    )

    if "data_groupby" not in config or not config["data_groupby"]:
        period_predictions_df = daily_predictions_df
    else:
        data_groupby_to_resample_mode = {
            "week": "W",
            "two_weeks": "2W",
            "four_weeks": "4W",
        }
        mode = data_groupby_to_resample_mode[config["data_groupby"]]
        period_predictions_df = daily_predictions_df.resample(
            mode, label="left", closed="left"
        ).sum()
        period_predictions_df["date"] = (
            period_predictions_df.index
        )  # Add date as column after resampling

    channel_names = [m["display_name"] for m in config["media"]]
    # reverse the order of channel names to match lightweight mmm
    channel_names.reverse()
    channel_cols = [f"{c}:median" for c in channel_names]
    cols = ["date", "baseline:median"] + channel_cols  # Include date in the columns to select

    # Select only the columns we need and return the DataFrame
    return period_predictions_df[cols]
