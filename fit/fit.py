from ..impl.lightweight_mmm.lightweight_mmm import lightweight_mmm

from ..constants import constants


# noinspection GrazieInspection
def fit_lightweight_mmm(data_to_fit, model_name, degrees_seasonality=2, number_warmup=2000, number_samples=2000):
    """
    fit a lightweight mmm model to input_data

    :param data_to_fit: DataToFit instance
    :param model_name: name of transform to perform (FIT_LIGHTWEIGHT_MMM_MODELNAME_XXX)
    :param degrees_seasonality: degrees of seasonality to pass through to lightweightMMM
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

    # number_chains=1 because my laptop has only one CPU (jax.local_device_count())
    mmm.fit(media=data_to_fit.media_data_train_scaled,
            degrees_seasonality=degrees_seasonality,
            seasonality_frequency=365 if data_to_fit.time_granularity == constants.GRANULARITY_DAILY else 52,
            weekday_seasonality=True if data_to_fit.time_granularity == constants.GRANULARITY_DAILY else False,
            media_names=data_to_fit.media_names,
            extra_features=extra_features,
            media_prior=data_to_fit.media_costs_scaled,
            target=data_to_fit.target_train_scaled,
            number_warmup=number_warmup,
            number_samples=number_samples,
            number_chains=1)

    return mmm
