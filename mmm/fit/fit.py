from impl.lightweight_mmm.lightweight_mmm import lightweight_mmm
from impl.lightweight_mmm.lightweight_mmm.utils import get_time_seed

from mmm.constants import constants
from mmm.data import DataToFit

import jax.numpy as jnp
import numpyro
import os
import yaml


# noinspection GrazieInspection
def fit_lightweight_mmm(
    config: dict,
    data_to_fit: DataToFit,
    results_dir: str,
):
    """
    fit a lightweight mmm model to input_data

    :param config: config object loaded from YAML file
    :param data_to_fit: DataToFit instance
    :param results_dir: directory to write log output to
    :return: lightweightMMM instance
    """
    model_name = config.get("model_name")
    assert model_name in (  # this setting is not optional
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
    )

    learned_media_priors_count = len(
        [p for p in data_to_fit.learned_media_priors.tolist() if p > 0.0]
    )
    if learned_media_priors_count > 0:
        print(f"setting learned media priors for {learned_media_priors_count} channels")

    fit_params = {
        "baseline_positivity_constraint": config.get("force_positive_baseline", False),
        "custom_priors": config.get("custom_priors"),
        "degrees_seasonality": config.get("degrees_seasonality", 2),
        "media_prior": media_priors,
        "model_name": model_name,
        "number_chains": config.get("number_chains", 1),
        "number_warmup": config.get("number_warmup", 2000),
        "number_samples": config.get("number_samples", 2000),
        "progress_bar": config.get("progress_bar", True),
        "target_is_log_scale": data_to_fit.target_is_log_scale,
    }

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

    fit_params["seasonality_frequency"] = seasonality_frequency

    if config.get("weekday_seasonality") is None:
        fit_params["weekday_seasonality"] = (
            True if data_to_fit.time_granularity == constants.GRANULARITY_DAILY else False
        )
    else:
        fit_params["weekday_seasonality"] = config.get("weekday_seasonality")

    print(
        f"fitting a model for {fit_params['model_name']} degrees_seasonality={fit_params['degrees_seasonality']}"
    )

    # manually generate a seed in the same way as lightweight mmm's
    # fit(), and store it for future reproducibility
    if config.get("seed") is not None:
        fit_params["seed"] = config.get("seed")
    else:
        fit_params["seed"] = get_time_seed()

    with open(os.path.join(results_dir, "fit_params.yaml"), "w") as output_file:
        yaml.dump(fit_params, output_file, default_flow_style=False)

    custom_priors = None
    if fit_params["custom_priors"] is not None:
        custom_priors = {}
        print(f"setting custom_priors for {', '.join(fit_params['custom_priors'].keys())}")

        for name, definition in fit_params["custom_priors"].items():
            # Handle case where definition is a list of distributions
            # XXX only uniform is supported for now

            if "values" in definition:
                if definition["type"] != "uniform":
                    raise ValueError(
                        f"Custom prior '{name}' has 'values' key but type '{definition['type']}'. "
                        "Only 'uniform' type supports multiple values."
                    )

                highs = []
                lows = []
                for values in definition["values"]:
                    highs.append(values["high"])
                    lows.append(values["low"])

                # example output:
                # { "custom_priors": { "coef_trend": <Uniform object at 0x7fff843f3dd0>} }
                # this is slightly weird, but later the model samples from
                # parallel arrays of values, so we need to pass in two arrays
                custom_priors[name] = numpyro.distributions.Uniform(
                    jnp.array(lows),
                    jnp.array(highs),
                )

            # Handle case where definition is a single distribution
            else:
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
