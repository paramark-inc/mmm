from contextlib import redirect_stdout
import numpy as np
import jax.numpy as jnp
import pandas as pd
import os

from ..impl.lightweight_mmm.lightweight_mmm.plot import (
    plot_bars_media_metrics,
    plot_media_baseline_contribution_area_plot,
    plot_media_channel_posteriors,
    plot_model_fit,
    plot_out_of_sample_model_fit,
    plot_prior_and_posterior,
    plot_response_curves,
)

from ..impl.lightweight_mmm.lightweight_mmm.media_transforms import calculate_seasonality

from ..constants import constants
from ..outlier.outlier import print_outliers
from ..plot.plot import plot_all_metrics


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


def _dump_posterior_metrics(
    input_data, media_effect_hat, roi_hat, cost_per_target_hat, results_dir
):
    """
    write posterior metrics to a file

    :param input_data: InputData instance
    :param media_effect_hat: see LightweightMMM.get_posterior_metrics
    :param roi_hat: see LightweightMMM.get_posterior_metrics
    :param cost_per_target_hat: the inverse of ROI hat (cost per target)
    :param results_dir: results directory
    """
    output_fname = os.path.join(results_dir, "media_performance_breakdown.txt")
    with open(output_fname, "w") as f:
        for media_idx in range(input_data.media_data.shape[1]):
            f.write(f"{input_data.media_names[media_idx]} Media Effect:\n")
            f.write(f"mean={np.mean(media_effect_hat[:, media_idx]):,.6f}\n")
            f.write(f"median={np.median(media_effect_hat[:, media_idx]):,.6f}\n")
            quantiles = np.quantile(media_effect_hat[:, media_idx], [0.05, 0.95])
            f.write(f"[0.05, 0.95]=[{quantiles[0]:,.6f}, {quantiles[1]:,.6f}]\n\n")

        for media_idx in range(input_data.media_data.shape[1]):
            f.write(f"{input_data.media_names[media_idx]} ROI:\n")
            f.write(f"mean={np.mean(roi_hat[:, media_idx]):,.6f}\n")
            f.write(f"median={np.median(roi_hat[:, media_idx]):,.6f}\n")
            quantiles = np.quantile(roi_hat[:, media_idx], [0.05, 0.95])
            f.write(f"[0.05, 0.95]=[{quantiles[0]:,.6f}, {quantiles[1]:,.6f}]\n\n")

        for media_idx in range(input_data.media_data.shape[1]):
            f.write(f"{input_data.media_names[media_idx]} cost per target:\n")
            f.write(f"mean={np.mean(cost_per_target_hat[:, media_idx]):,.6f}\n")
            f.write(f"median={np.median(cost_per_target_hat[:, media_idx]):,.6f}\n")
            quantiles = np.quantile(cost_per_target_hat[:, media_idx], [0.05, 0.95])
            f.write(f"[0.05, 0.95]=[{quantiles[0]:,.6f}, {quantiles[1]:,.6f}]\n\n")


def _dump_baseline_breakdown(
    media_mix_model, input_data, data_to_fit, degrees_seasonality, results_dir
):
    """
    Break down the baseline into its component pieces and write the results to a text file.

    :param media_mix_model: LightweightMMM instance
    :param input_data: InputData instance
    :param data_to_fit: DataToFit instance
    :param degrees_seasonality: Degrees of seasonality used to fit the model
    :param results_dir: results directory to write to
    :return: None
    """

    # media_einsum = "tc, c -> t"  # t = time, c = channel
    # extra_features_einsum = "tf, f -> t"  # t = time, f = feature
    # target = intercept +
    #          coef_trend * trend ** expo_trend +
    #          seasonality +
    #          jnp.einsum(media_einsum, media_transformed, coef_media) +
    #          jnp.einsum(extra_features_einsum,
    #                     extra_features,
    #                     coef_extra_features)
    # with an extra term for weekday cases:
    #          weekday[jnp.arange(data_size) % 7]

    # Array shapes:
    #   intercept = (samples, 1)
    #   coef_trend = (samples, 1)
    #   expo_trend = (samples,)
    #   media_transformed = (samples, observations, channels)
    #   coef_media = (samples, channels)
    #   coef_extra_features = (samples, features)
    #   coef_weekday = (samples, 7)
    #

    mmm = media_mix_model
    num_observations = data_to_fit.media_data_train_scaled.shape[0]
    intercept = jnp.mean(jnp.squeeze(mmm.trace["intercept"]))
    coef_trend = jnp.mean(jnp.squeeze(mmm.trace["coef_trend"]))
    expo_trend = jnp.mean(mmm.trace["expo_trend"])
    if data_to_fit.extra_features_train_scaled.shape[1]:
        coef_extra_features = jnp.mean(mmm.trace["coef_extra_features"], axis=0)
    else:
        coef_extra_features = None
    gamma_seasonality = jnp.mean(mmm.trace["gamma_seasonality"], axis=0)

    columns = ["intercept", "trend", "seasonality"]

    if input_data.time_granularity == constants.GRANULARITY_DAILY:
        weekday = jnp.mean(mmm.trace["weekday"], axis=0)
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
    baseline_breakdown_df["baseline"] = baseline_breakdown_df.sum(axis=1)

    with open(os.path.join(results_dir, "baseline_breakdown.txt"), "w") as f:
        f.write("Mean value by component:\n\n")
        f.write(f"intercept={baseline_breakdown_df['intercept'].mean():,.4f}\n")
        f.write(f"trend={baseline_breakdown_df['trend'].mean():,.4f}\n")
        f.write(f"seasonality={baseline_breakdown_df['seasonality'].mean():,.4f}\n")
        for j in range(data_to_fit.extra_features_train_scaled.shape[1]):
            extra_feature_name = data_to_fit.extra_features_names[j]
            f.write(
                f"{extra_feature_name}={baseline_breakdown_df[extra_feature_name].mean():,.4f}\n"
            )
        if weekday is not None:
            f.write(f"weekday={baseline_breakdown_df['weekday'].mean():,.4f}\n")
        f.write(f"baseline={baseline_breakdown_df['baseline'].mean():,.4f}\n")

        f.write("\n")
        f.write("Row by Row breakdown:\n\n")
        f.write(baseline_breakdown_df.to_string(float_format=lambda x: f"{x:,.4f}"))

    return baseline_breakdown_df


def describe_mmm_training(mmm, input_data, data_to_fit, degrees_seasonality, results_dir):
    """
    Plot and print diagnostic analyses of the MMM training data.

    :param mmm: LightweightMMM instance
    :param input_data: InputData instance
    :param data_to_fit: DataToFit instance
    :param degrees_seasonality: degrees of seasonality used for fitting
    :param results_dir: directory to write plot files to
    :return:
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
    fig = plot_response_curves(
        media_mix_model=mmm,
        media_scaler=data_to_fit.media_scaler,
        target_scaler=data_to_fit.target_scaler,
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
        input_data=input_data,
        media_effect_hat=media_effect_hat,
        roi_hat=roi_hat,
        cost_per_target_hat=cost_per_target_hat,
        results_dir=results_dir,
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
        media_mix_model=mmm,
        input_data=input_data,
        data_to_fit=data_to_fit,
        degrees_seasonality=degrees_seasonality,
        results_dir=results_dir,
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
