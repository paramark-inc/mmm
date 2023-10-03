from impl.lightweight_mmm.lightweight_mmm import lightweight_mmm
from impl.lightweight_mmm.lightweight_mmm.utils import get_time_seed

from mmm.constants import constants

import jax.numpy as jnp
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
    number_chains=1,
    seed=None,
    custom_prior_config=None,
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
    :param number_warmup to pass through to lightweightMMM.
    :param number_samples to pass through to lightweightMMM.
    :param number_chains to pass through to lightweightMMM.  Cannot be more than the number of CPUs
           on the system.
    :return: lightweightMMM instance
    """
    assert model_name in (
        constants.FIT_LIGHTWEIGHT_MMM_MODELNAME_ADSTOCK,
        constants.FIT_LIGHTWEIGHT_MMM_MODELNAME_HILL_ADSTOCK,
        constants.FIT_LIGHTWEIGHT_MMM_MODELNAME_CARRYOVER,
    ), model_name

    # train the model
    mmm = lightweight_mmm.LightweightMMM(model_name=model_name)

    if data_to_fit.extra_features_train_scaled.shape[1] == 0:
        extra_features = None
    else:
        extra_features = data_to_fit.extra_features_train_scaled

    # when we have a learned_media_prior, use it.  Otherwise, use the media_cost_prior.
    media_priors = jnp.where(
        data_to_fit.learned_media_priors > 0.0,
        data_to_fit.learned_media_priors,
        data_to_fit.media_cost_priors_scaled,
    ).tolist()

    learned_media_priors_count = len(
        [p for p in data_to_fit.learned_media_priors.tolist() if p > 0.0]
    )
    if learned_media_priors_count > 0:
        print(f"setting learned media priors for {learned_media_priors_count} channels")

    fit_params = {
        "custom_priors": custom_prior_config,
        "degrees_seasonality": degrees_seasonality,
        "media_prior": media_priors,
        "model_name": model_name,
        "number_warmup": number_warmup,
        "number_samples": number_samples,
        "target_is_log_scale": data_to_fit.target_is_log_scale,
    }

    fit_params["seasonality_frequency"] = (
        365 if data_to_fit.time_granularity == constants.GRANULARITY_DAILY else 52
    )

    if weekday_seasonality is None:
        fit_params["weekday_seasonality"] = (
            True if data_to_fit.time_granularity == constants.GRANULARITY_DAILY else False
        )
    else:
        fit_params["weekday_seasonality"] = weekday_seasonality

    fit_params["number_chains"] = number_chains

    # manually generate a seed in the same way as lightweight mmm's
    # fit(), and store it for future reproducibility
    if seed is not None:
        fit_params["seed"] = seed
    else:
        fit_params["seed"] = get_time_seed()

    with open(os.path.join(results_dir, "fit_params.yaml"), "w") as output_file:
        yaml.dump(fit_params, output_file, default_flow_style=False)

    custom_priors = None
    if custom_prior_config is not None:
        custom_priors = {}
        print(f"setting custom_priors for {', '.join(custom_prior_config.keys())}")

        for name, definition in custom_prior_config.items():
            if definition["type"] == "halfnormal":
                custom_priors[name] = numpyro.distributions.HalfNormal(definition["scale"])
            elif definition["type"] == "normal":
                custom_priors[name] = numpyro.distributions.Normal(
                    definition["loc"], definition["scale"]
                )
            elif definition["type"] == "uniform":
                custom_priors[name] = numpyro.distributions.Uniform(
                    definition["low"],
                    definition["high"],
                )
            elif definition["type"] == "gamma":
                custom_priors[name] = numpyro.distributions.Gamma(
                    definition["concentration"],
                    definition["rate"],
                )

    # remove parameter(s) that we want to write, but don't pass directly to fit()
    del fit_params["model_name"], fit_params["custom_priors"]

    # If you hit "RuntimeError: Cannot find valid initial parameters. Please
    # check your model again." while fitting the model, and are running on x64, consider trying
    # the following jax option.  See https://github.com/google/lightweight_mmm/issues/77.
    #
    # jax.config.update("jax_enable_x64", True)

    mmm.fit(
        media=data_to_fit.media_data_train_scaled,
        media_names=data_to_fit.media_names,
        extra_features=extra_features,
        target=data_to_fit.target_train_scaled,
        custom_priors=custom_priors,
        **fit_params,
    )

    return mmm
