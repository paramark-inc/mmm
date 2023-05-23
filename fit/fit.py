import jax.numpy as jnp

import impl.lightweight_mmm.lightweight_mmm.lightweight_mmm as lightweight_mmm

from impl.lightweight_mmm.lightweight_mmm import preprocessing


# from mmm.impl.lightweight_mmm.lightweight_mmm import plot as mmm_plot

def fit_lightweight_mmm(input_data):
    """
    fit a lightweight mmm model to input_data

    :param input_data: input_data of type InputData
    :return: (model, target_scaler) tuple
    """
    data_size = input_data.media_data.shape[0]

    split_point = data_size - data_size // 10
    media_data_train = input_data.media_data[:split_point, :]
    target_train = input_data.target_data[:split_point]
    extra_features_train = input_data.extra_features_data[:split_point, :]
    # extra_features_test = input_data.extra_features_data[split_point:, :]

    # Scale data (ignoring the zeroes in the media data)
    # media_scaler = preprocessing.CustomScaler(divide_operation=lambda x: jnp.mean(x[x > 0]))
    media_scaler = preprocessing.CustomScaler(divide_operation=lambda x: x.sum() / (x > 0).sum())
    extra_features_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)

    # scale cost up by N since fit() will divide it by number of time periods
    # cost_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)

    scaled_media_data_train = media_scaler.fit_transform(media_data_train)
    scaled_extra_features_train = extra_features_scaler.fit_transform(extra_features_train)
    scaled_target_train = target_scaler.fit_transform(target_train)
    # scaled_costs = cost_scaler.fit_transform(input_data.media_costs_per_unit)

    # lightweightMMM requires that media priors are > 0 by virtue of using HalfNormal which has a Positive constraint
    # on all values
    media_prior = jnp.where(
        input_data.media_costs_per_unit > 0.0,
        input_data.media_costs_per_unit,
        0.001
    )

    # train the model
    mmm = lightweight_mmm.LightweightMMM(model_name="hill_adstock")

    # number_chains=1 because my laptop has only one CPU (jax.local_device_count())
    # TODO think about seasonality options
    mmm.fit(media=scaled_media_data_train,
            seasonality_frequency=365,
            weekday_seasonality=True,
            extra_features=scaled_extra_features_train,
            media_prior=media_prior,
            target=scaled_target_train,
            number_warmup=2000,
            number_samples=2000,
            number_chains=1)

    return mmm, target_scaler
