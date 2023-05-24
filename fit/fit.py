import jax.numpy as jnp

import impl.lightweight_mmm.lightweight_mmm.lightweight_mmm as lightweight_mmm

from impl.lightweight_mmm.lightweight_mmm import preprocessing
from model.model import DataToFit


def _make_data_to_fit(input_data):
    data_size = input_data.media_data.shape[0]

    split_point = data_size - data_size // 10
    media_data_train = input_data.media_data[:split_point, :]
    media_data_test = input_data.media_data[split_point:, :]
    target_train = input_data.target_data[:split_point]
    target_test = input_data.target_data[split_point:]
    extra_features_train = input_data.extra_features_data[:split_point, :]
    extra_features_test = input_data.extra_features_data[split_point:, :]

    # Scale data (ignoring the zeroes in the media data)
    media_scaler = preprocessing.CustomScaler(divide_operation=lambda x: x.sum() / (x > 0).sum())
    extra_features_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)

    # scale cost up by N since fit() will divide it by number of time periods
    media_cost_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)

    media_data_train_scaled = media_scaler.fit_transform(media_data_train)
    media_data_test_scaled = media_scaler.fit_transform(media_data_test)
    extra_features_train_scaled = extra_features_scaler.fit_transform(extra_features_train)
    extra_features_test_scaled = extra_features_scaler.fit_transform(extra_features_test)
    target_train_scaled = target_scaler.fit_transform(target_train)
    target_test_scaled = target_scaler.fit_transform(target_test)
    # lightweightMMM requires that media priors are > 0 by virtue of using HalfNormal which has a Positive constraint
    # on all values
    costs_fixup = jnp.where(
        input_data.media_costs_per_unit > 0.0,
        input_data.media_costs_per_unit,
        0.001
    )
    media_costs_scaled = media_cost_scaler.fit_transform(costs_fixup)

    return DataToFit(media_data_train_scaled=media_data_train_scaled, media_data_test_scaled=media_data_test_scaled,
                     media_scaler=media_scaler, extra_features_train_scaled=extra_features_train_scaled,
                     extra_features_test_scaled=extra_features_test_scaled, extra_features_scaler=extra_features_scaler,
                     media_costs_scaled=media_costs_scaled, media_costs_scaler=media_cost_scaler,
                     target_train_scaled=target_train_scaled, target_test_scaled=target_test_scaled,
                     target_scaler=target_scaler)


def fit_lightweight_mmm(input_data):
    """
    fit a lightweight mmm model to input_data

    :param input_data: input_data of type InputData
    :return: (model, target_scaler) tuple
    """
    data_to_fit = _make_data_to_fit(input_data)

    # train the model
    mmm = lightweight_mmm.LightweightMMM(model_name="hill_adstock")

    # number_chains=1 because my laptop has only one CPU (jax.local_device_count())
    # TODO think about seasonality options
    mmm.fit(media=data_to_fit.media_data_train_scaled,
            seasonality_frequency=365,
            weekday_seasonality=True,
            extra_features=data_to_fit.extra_features_train_scaled,
            media_prior=data_to_fit.media_costs_scaled,
            target=data_to_fit.target_train_scaled,
            number_warmup=2000,
            number_samples=2000,
            number_chains=1)

    return mmm, data_to_fit
