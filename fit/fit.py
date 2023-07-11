from ..impl.lightweight_mmm.lightweight_mmm import lightweight_mmm

from ..constants import constants

import numpyro
import os
import yaml


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

    fit_params = {
        "model_name": model_name,
        "degrees_seasonality": degrees_seasonality,
        "number_warmup": number_warmup,
        "number_samples": number_warmup,
        "media_prior": data_to_fit.media_priors_scaled.tolist(), # TODO hourann validate that tolist() doesn't break the arg
        "target_is_log_scale": data_to_fit.target_is_log_scale,
    }

    fit_params["seasonality_frequency"] = 365 if data_to_fit.time_granularity == constants.GRANULARITY_DAILY else 52

    if weekday_seasonality is None:
        fit_params["weekday_seasonality"] = True if data_to_fit.time_granularity == constants.GRANULARITY_DAILY else False
    else:
        fit_params["weekday_seasonality"] = weekday_seasonality

    # number_chains=1 because my laptop has only one CPU (jax.local_device_count())
    fit_params["number_chains"] = 1

    # manually generate a seed in the same way as lightweight mmm's
    # fit(), and store it for future reproducibility
    fit_params["seed"] = lightweight_mmm.utils.get_time_seed()

    with open(os.path.join(results_dir, "fit_params.yaml"), "w") as output_file:
        yaml.dump(fit_params, output_file, default_flow_style=False)

    # remove parameter(s) that we want to write, but don't want in fit()
    del fit_params["model_name"]

    mmm.fit(media=data_to_fit.media_data_train_scaled,
            media_names=data_to_fit.media_names,
            extra_features=extra_features,
            target=data_to_fit.target_train_scaled,
            **fit_params)

    # from lightweight_mmm/lightweight_mmm.py
    _NAMES_TO_MODEL_TRANSFORMS = {
      "hill_adstock": lightweight_mmm.models.transform_hill_adstock,
      "adstock": lightweight_mmm.models.transform_adstock,
      "carryover": lightweight_mmm.models.transform_carryover
    }

    numpyro.render_model(
        mmm._model_function,
        model_args=(
            data_to_fit.media_data_train_scaled, # media_data: jnp.ndarray
            data_to_fit.target_train_scaled, # target_data
            data_to_fit.media_priors_scaled, # media_prior
            fit_params["degrees_seasonality"], # degrees_seasonality
            fit_params["seasonality_frequency"], # frequency
            _NAMES_TO_MODEL_TRANSFORMS[model_name], # transform_function: TransformFunction,
            {}, # custom_priors: MutableMapping[str, Prior]
            # transform_kwargs: Optional[MutableMapping[str, Any]] = None,
            # weekday_seasonality: bool = False,
            # extra_features: Optional[jnp.array] = None
        ),
        filename=os.path.join(results_dir, "hourann.png"),
    )

    return mmm
