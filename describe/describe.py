import os

from ..impl.lightweight_mmm.lightweight_mmm.plot import (
    plot_bars_media_metrics,
    plot_media_channel_posteriors,
    plot_model_fit,
    plot_out_of_sample_model_fit,
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
    print_outliers(input_data=input_data, suffix=suffix)


def describe_mmm_training(mmm, input_data, data_to_fit, results_dir):
    """
    Plot and print diagnostic analyses of the MMM training data.

    :param mmm: LightweightMMM instance
    :param input_data: InputData instance
    :param data_to_fit: DataToFit instance
    :param results_dir: directory to write plot files to
    :return:
    """
    mmm.print_summary()
    plt = plot_model_fit(media_mix_model=mmm, target_scaler=data_to_fit.target_scaler)
    output_fname = os.path.join(results_dir, "model_fit.png")
    plt.savefig(output_fname)
    print(f"wrote {output_fname}")

    plt = plot_media_channel_posteriors(media_mix_model=mmm, channel_names=input_data.media_names)
    output_fname = os.path.join(results_dir, "media_posteriors.png")
    plt.savefig(output_fname)
    print(f"wrote {output_fname}")

    plt = plot_response_curves(
        media_mix_model=mmm, media_scaler=data_to_fit.media_scaler, target_scaler=data_to_fit.target_scaler
    )
    output_fname = os.path.join(results_dir, "response_curves.png")
    plt.savefig(output_fname)
    print(f"wrote {output_fname}")

    media_effect_hat, roi_hat = mmm.get_posterior_metrics(
        cost_scaler=data_to_fit.media_costs_scaler,
        target_scaler=data_to_fit.target_scaler
    )

    plt = plot_bars_media_metrics(metric=media_effect_hat, channel_names=input_data.media_names)
    output_fname = os.path.join(results_dir, "media_effect_hat.png")
    plt.savefig(output_fname)
    print(f"wrote {output_fname}")

    plt = plot_bars_media_metrics(metric=roi_hat, channel_names=input_data.media_names)
    output_fname = os.path.join(results_dir, "roi_hat.png")
    plt.savefig(output_fname)
    print(f"wrote {output_fname}")


def describe_mmm_prediction(mmm, data_to_fit, results_dir):
    """
    Plot and print diagnostic analyses of the mmm's predictive ability based on the test data

    :param mmm: LightweightMMM instance
    :param data_to_fit:  DataToFit instance
    :param results_dir: Directory to write plots to
    :return: None
    """
    prediction = mmm.predict(
        media=data_to_fit.media_data_test_scaled,
        extra_features=data_to_fit.extra_features_test_scaled,
        target_scaler=data_to_fit.target_scaler
    )

    target_test_unscaled = data_to_fit.target_scaler.inverse_transform(data_to_fit.target_test_scaled)
    plt = plot_out_of_sample_model_fit(
        out_of_sample_predictions=prediction, out_of_sample_target=target_test_unscaled
    )
    output_fname = os.path.join(results_dir, "out_of_sample_model_fit.png")
    plt.savefig(output_fname)
    print(f"wrote {output_fname}")
