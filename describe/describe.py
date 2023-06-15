from contextlib import redirect_stdout
import numpy as np
import os

from ..impl.lightweight_mmm.lightweight_mmm.plot import (
    plot_bars_media_metrics,
    plot_media_baseline_contribution_area_plot,
    plot_media_channel_posteriors,
    plot_model_fit,
    plot_out_of_sample_model_fit,
    plot_prior_and_posterior,
    plot_response_curves
)

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


def _dump_posterior_metrics(input_data, media_effect_hat, roi_hat, results_dir):
    """
    write posterior metrics to a file

    :param input_data: InputData instance
    :param media_effect_hat: see LightweightMMM.get_posterior_metrics
    :param roi_hat: see LightweightMMM.get_posterior_metrics
    :param results_dir: results directory
    """
    output_fname = os.path.join(results_dir, "media_effect_and_roi_results.txt")
    with open(output_fname, 'w') as f:
        for media_idx in range(input_data.media_data.shape[1]):
            f.write(f"{input_data.media_names[media_idx]} Media Effect:\n")
            f.write(f"mean={np.mean(media_effect_hat[:, media_idx]):,.3f}\n")
            f.write(f"median={np.median(media_effect_hat[:, media_idx]):,.3f}\n")
            quantiles = np.quantile(media_effect_hat[:, media_idx], [0.05, 0.95])
            f.write(f"[0.05, 0.95]=[{quantiles[0]:,.3f}, {quantiles[1]:,.3f}]\n\n")

        for media_idx in range(input_data.media_data.shape[1]):
            f.write(f"{input_data.media_names[media_idx]} ROI:\n")
            f.write(f"mean={np.mean(roi_hat[:, media_idx]):,.3f}\n")
            f.write(f"median={np.median(roi_hat[:, media_idx]):,.3f}\n")
            quantiles = np.quantile(roi_hat[:, media_idx], [0.05, 0.95])
            f.write(f"[0.05, 0.95]=[{quantiles[0]:,.3f}, {quantiles[1]:,.3f}]\n\n")


def describe_mmm_training(mmm, input_data, data_to_fit, results_dir):
    """
    Plot and print diagnostic analyses of the MMM training data.

    :param mmm: LightweightMMM instance
    :param input_data: InputData instance
    :param data_to_fit: DataToFit instance
    :param results_dir: directory to write plot files to
    :return:
    """
    output_fname = os.path.join(results_dir, "model_summary.txt")
    with open(output_fname, 'w') as f:
        with redirect_stdout(f):
            mmm.print_summary()

    fig = plot_model_fit(media_mix_model=mmm, target_scaler=data_to_fit.target_scaler)
    output_fname = os.path.join(results_dir, "model_fit.png")
    fig.savefig(output_fname)

    fig = plot_media_channel_posteriors(media_mix_model=mmm, channel_names=data_to_fit.media_names)
    output_fname = os.path.join(results_dir, "media_posteriors.png")
    fig.savefig(output_fname)

    media_cost_per_unscaled_unit = input_data.media_costs / np.sum(input_data.media_data, axis=0)
    fig = plot_response_curves(
        media_mix_model=mmm,
        media_scaler=data_to_fit.media_scaler,
        target_scaler=data_to_fit.target_scaler,
        prices=media_cost_per_unscaled_unit
    )
    output_fname = os.path.join(results_dir, "response_curves.png")
    fig.savefig(output_fname)

    fig = plot_prior_and_posterior(media_mix_model=mmm)
    output_fname = os.path.join(results_dir, "all_priors_and_posteriors.png")
    fig.savefig(output_fname)

    media_effect_hat, roi_hat = mmm.get_posterior_metrics(
        unscaled_costs=input_data.media_costs,
        target_scaler=data_to_fit.target_scaler
    )
    _dump_posterior_metrics(
        input_data=input_data,
        media_effect_hat=media_effect_hat,
        roi_hat=roi_hat,
        results_dir=results_dir
    )

    fig = plot_bars_media_metrics(metric=media_effect_hat, channel_names=data_to_fit.media_names)
    output_fname = os.path.join(results_dir, "media_effect_hat.png")
    fig.savefig(output_fname)

    fig = plot_bars_media_metrics(metric=roi_hat, channel_names=data_to_fit.media_names)
    output_fname = os.path.join(results_dir, "roi_hat.png")
    fig.savefig(output_fname)

    fig = plot_media_baseline_contribution_area_plot(
        media_mix_model=mmm, target_scaler=data_to_fit.target_scaler, channel_names=data_to_fit.media_names
    )
    output_fname = os.path.join(results_dir, "weekly_media_and_baseline_contribution.png")
    fig.savefig(output_fname)


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
        target_scaler=data_to_fit.target_scaler
    )

    target_test_unscaled = data_to_fit.target_scaler.inverse_transform(data_to_fit.target_test_scaled)
    fig = plot_out_of_sample_model_fit(
        out_of_sample_predictions=prediction, out_of_sample_target=target_test_unscaled
    )
    output_fname = os.path.join(results_dir, "out_of_sample_model_fit.png")
    fig.savefig(output_fname)
