from ..impl.lightweight_mmm.lightweight_mmm import lightweight_mmm

from ..constants import constants

import os


# noinspection GrazieInspection
def fit_lightweight_mmm(
        data_to_fit,
        model_name,
        results_dir,
        degrees_seasonality=2,
        weekday_seasonality=None,
        number_warmup=2000,
        number_samples=2000,
):
    """
    fit a lightweight mmm model to input_data

    :param data_to_fit: DataToFit instance
    :param model_name: name of transform to perform (FIT_LIGHTWEIGHT_MMM_MODELNAME_XXX)
    :param results_dir: directory to write log output to
    :param degrees_seasonality: degrees of seasonality to pass through to lightweightMMM
    :param weekday_seasonality: if None, we will derive this parameter from the time_granularity;
                                otherwise, use the value provided.  For daily data, passing "False" will cause the
                                model to omit the daily coefficients, which can be preferable if you are seeing too
                                much day to day swing in the results.

                                Do not pass a value if you have weekly data.
    :param number_warmup to pass through to lightweightMMM
    :param number_samples to pass through to lightweightMMM
    :return: lightweightMMM instance
    """
    assert model_name in (
        constants.FIT_LIGHTWEIGHT_MMM_MODELNAME_ADSTOCK,
        constants.FIT_LIGHTWEIGHT_MMM_MODELNAME_HILL_ADSTOCK,
        constants.FIT_LIGHTWEIGHT_MMM_MODELNAME_CARRYOVER
    ), model_name

    # train the model
    mmm = lightweight_mmm.LightweightMMM(model_name=model_name)

    if data_to_fit.extra_features_train_scaled.shape[1] == 0:
        extra_features = None
    else:
        extra_features = data_to_fit.extra_features_train_scaled

    seasonality_frequency = 365 if data_to_fit.time_granularity == constants.GRANULARITY_DAILY else 52

    if weekday_seasonality is None:
        weekday_seasonality_touse = True if data_to_fit.time_granularity == constants.GRANULARITY_DAILY else False
    else:
        weekday_seasonality_touse = weekday_seasonality

    # number_chains=1 because my laptop has only one CPU (jax.local_device_count())
    number_chains = 1

    with open(os.path.join(results_dir, "fit_params.txt"), "w") as output_file:
        output_file.write(f"model_name={model_name}\n")
        output_file.write(f"degrees_seasonality={degrees_seasonality}\n")
        output_file.write(f"seasonality_frequency={seasonality_frequency}\n")
        output_file.write(f"weekday_seasonality_touse={weekday_seasonality_touse}\n")
        output_file.write(f"media_prior={data_to_fit.media_costs_scaled}\n")
        output_file.write(f"number_warmup={number_warmup}\n")
        output_file.write(f"number_samples={number_samples}\n")
        output_file.write(f"number_chains={number_chains}\n")

    mmm.fit(media=data_to_fit.media_data_train_scaled,
            degrees_seasonality=degrees_seasonality,
            seasonality_frequency=seasonality_frequency,
            weekday_seasonality=weekday_seasonality_touse,
            media_names=data_to_fit.media_names,
            extra_features=extra_features,
            media_prior=data_to_fit.media_costs_scaled,
            target=data_to_fit.target_train_scaled,
            target_is_log_scale=data_to_fit.target_is_log_scale,
            number_warmup=number_warmup,
            number_samples=number_samples,
            number_chains=number_chains)

    return mmm
