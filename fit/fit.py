from ..impl.lightweight_mmm.lightweight_mmm import lightweight_mmm

from ..constants import constants


def fit_lightweight_mmm(input_data, data_to_fit, model_name, degrees_seasonality=2):
    """
    fit a lightweight mmm model to input_data

    :param input_data: InputData instance
    :param data_to_fit: DataToFit instance
    :param model_name: name of transform to perform (FIT_LIGHTWEIGHT_MMM_MODELNAME_XXX)
    :param degrees_seasonality: degrees of seasonality to pass through to lw MMM
    :return: lightweightMMM instance
    """
    assert model_name in (
        constants.FIT_LIGHTWEIGHT_MMM_MODELNAME_ADSTOCK,
        constants.FIT_LIGHTWEIGHT_MMM_MODELNAME_HILL_ADSTOCK,
        constants.FIT_LIGHTWEIGHT_MMM_MODELNAME_CARRYOVER
    ), model_name

    # train the model
    mmm = lightweight_mmm.LightweightMMM(model_name=model_name)

    # number_chains=1 because my laptop has only one CPU (jax.local_device_count())
    # TODO degrees_seasonality
    mmm.fit(media=data_to_fit.media_data_train_scaled,
            degrees_seasonality=degrees_seasonality,
            seasonality_frequency=365 if input_data.time_granularity == constants.GRANULARITY_DAILY else 52,
            weekday_seasonality=True if input_data.time_granularity == constants.GRANULARITY_DAILY else False,
            media_names=input_data.media_names,
            extra_features=data_to_fit.extra_features_train_scaled,
            media_prior=data_to_fit.media_costs_scaled,
            target=data_to_fit.target_train_scaled,
            number_warmup=2000,
            number_samples=2000,
            number_chains=1)

    return mmm
