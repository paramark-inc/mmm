import json
import os
import numpy as np
import jax
import jax.numpy as jnp

from lightweight_mmm.lightweight_mmm import LightweightMMM
from impl.lightweight_mmm.lightweight_mmm.plot import (
    _make_single_prediction,
    _train_cost_models,
    _predict_costs_for_media_units,
)
from impl.lightweight_mmm.lightweight_mmm.media_transforms import (
    adstock as adstock_transform,
)

from mmm.data.data_to_fit import DataToFit
from mmm.data.input_data import InputData


def _extract_cost_model_params(cost_model) -> dict | None:
    """Extract polynomial parameters from a cost model fitted by _train_cost_models.

    The polynomial represents: spend = c0 + c1 * impressions.
    """
    if cost_model is None:
        return None
    converted = cost_model.convert()
    return {
        "type": "polynomial_deg1",
        "coefficients": [float(c) for c in converted.coef],
        "domain": [float(d) for d in cost_model.domain],
        "window": [float(w) for w in cost_model.window],
    }


def _get_scaler_values(scaler, channel_idx: int | None, geo_idx: int | None):
    """Return (multiply_by, divide_by) as Python floats for a specific channel/geo.

    For media scalers, pass channel_idx (and geo_idx for geo models).
    For target scalers, pass channel_idx=None and (optionally) geo_idx.
    """
    if scaler is None:
        return 1.0, 1.0

    mult = np.asarray(scaler.multiply_by)
    div = np.asarray(scaler.divide_by)

    if mult.ndim == 0:
        return float(mult), float(div)
    if mult.ndim == 1:
        idx = channel_idx if channel_idx is not None else geo_idx
        if idx is not None:
            return float(mult[idx]), float(div[idx])
        return float(mult), float(div)
    if mult.ndim == 2 and channel_idx is not None and geo_idx is not None:
        return float(mult[channel_idx, geo_idx]), float(div[channel_idx, geo_idx])

    # Fallback: return full value
    return float(mult), float(div)


def _compute_adstock_carry_over(
    mmm: LightweightMMM,
) -> np.ndarray:
    """Compute raw (un-normalised) adstock state at the last training time step.

    Uses the median lag_weight from the MCMC trace.

    Returns:
        Array of shape (channels,) for non-hierarchical models or (channels, geos) for hierarchical models.
    """
    lag_weight_median = jnp.median(mmm.trace["lag_weight"], axis=0)  # (channels,)
    if mmm.media.ndim == 3:
        lag_weight_median = jnp.expand_dims(lag_weight_median, axis=-1)
    raw_adstock = adstock_transform(
        data=mmm.media, lag_weight=lag_weight_median, normalise=False
    )
    return np.asarray(raw_adstock[-1])  # (channels,) or (channels, geos)


def _compute_carryover_params(
    mmm: LightweightMMM,
    channel_idx: int,
    geo_idx: int | None,
) -> tuple[float, float]:
    """Compute the linear carryover coefficients (w0, B) for a single channel.

    The carryover at a new time step is: carryover(x) = w0 * x + B
    where x is the new (scaled) media value and B captures the trailing training
    history.

    Returns:
        (w0, B) tuple of floats.
    """
    retention_median = np.asarray(
        jnp.median(mmm.trace["ad_effect_retention_rate"], axis=0)
    )  # (channels,)
    peak_delay_median = np.asarray(
        jnp.median(mmm.trace["peak_effect_delay"], axis=0)
    )  # (channels,)

    number_lags = 13
    if mmm._weekday_seasonality:
        number_lags = 13 * 7

    lags = np.arange(number_lags, dtype=np.float64)
    retention = float(retention_median[channel_idx])
    peak_delay = float(peak_delay_median[channel_idx])

    weights = retention ** ((lags - peak_delay) ** 2)
    w_sum = float(weights.sum())
    w0 = float(weights[0]) / w_sum

    T = mmm.media.shape[0] - 1
    B = 0.0
    for lag in range(1, number_lags):
        t_idx = T - lag
        if t_idx >= 0:
            if geo_idx is not None:
                data_val = float(mmm.media[t_idx, channel_idx, geo_idx])
            else:
                data_val = float(mmm.media[t_idx, channel_idx])
            B += float(weights[lag]) * data_val / w_sum

    return w0, B


def _build_channel_transform_params(
    mmm: LightweightMMM,
    data_to_fit: DataToFit,
    channel_idx: int,
    geo_idx: int | None,
    adstock_carry_over: np.ndarray | None,
) -> dict:
    """Build the impressions_to_target parameter dict for one channel.

    Args:
        mmm: Fitted model.
        data_to_fit: DataToFit instance.
        channel_idx: Channel index.
        geo_idx: Geo index (None for non-hierarchical models).
        adstock_carry_over: Pre-computed adstock carry-over array (for
            adstock / hill_adstock models), or None for carryover models.
    """
    model_name = mmm.model_name

    coef_median = np.asarray(jnp.median(mmm.trace["coef_media"], axis=0))
    if geo_idx is not None:
        coef_media = float(coef_median[channel_idx, geo_idx])
    else:
        coef_media = float(coef_median[channel_idx])

    media_mult, media_div = _get_scaler_values(
        data_to_fit.media_scaler, channel_idx=channel_idx, geo_idx=geo_idx
    )
    target_mult, target_div = _get_scaler_values(
        data_to_fit.target_scaler, channel_idx=None, geo_idx=geo_idx
    )

    result = {
        "type": model_name,
        "coef_media": coef_media,
        "media_scaler": {"multiply_by": media_mult, "divide_by": media_div},
        "target_scaler": {"multiply_by": target_mult, "divide_by": target_div},
    }

    if model_name in ("adstock", "hill_adstock"):
        if adstock_carry_over is None:
            # without a none check here, mypy complains about adstock_carry_over[channel_idx] below
            raise ValueError(
                f"adstock_carry_over cannot be None for model type {model_name}"
            )

        lag_weight_median = np.asarray(jnp.median(mmm.trace["lag_weight"], axis=0))
        if geo_idx is not None:
            carry_val = float(adstock_carry_over[channel_idx, geo_idx])
        else:
            carry_val = float(adstock_carry_over[channel_idx])

        result["lag_weight"] = float(lag_weight_median[channel_idx])
        result["adstock_carry_over"] = carry_val

        if model_name == "adstock":
            exponent_median = np.asarray(jnp.median(mmm.trace["exponent"], axis=0))
            result["exponent"] = float(exponent_median[channel_idx])
        else:  # hill_adstock
            ec50_median = np.asarray(
                jnp.median(mmm.trace["half_max_effective_concentration"], axis=0)
            )
            slope_median = np.asarray(jnp.median(mmm.trace["slope"], axis=0))
            result["half_max_effective_concentration"] = float(ec50_median[channel_idx])
            result["slope"] = float(slope_median[channel_idx])

    elif model_name == "carryover":
        exponent_median = np.asarray(jnp.median(mmm.trace["exponent"], axis=0))
        retention_median = np.asarray(
            jnp.median(mmm.trace["ad_effect_retention_rate"], axis=0)
        )
        peak_delay_median = np.asarray(
            jnp.median(mmm.trace["peak_effect_delay"], axis=0)
        )
        w0, B = _compute_carryover_params(mmm, channel_idx, geo_idx)
        result["ad_effect_retention_rate"] = float(retention_median[channel_idx])
        result["peak_effect_delay"] = float(peak_delay_median[channel_idx])
        result["exponent"] = float(exponent_median[channel_idx])
        result["carryover_weight_coeff"] = w0
        result["carryover_offset"] = B

    return result


def _build_channels_json(
    mmm: LightweightMMM,
    data_to_fit: DataToFit,
    cost_models: list,
    should_skip_channel: list[bool],
    spend_ranges: np.ndarray,
    target_predictions: np.ndarray,
    geo_idx: int | None,
    adstock_carry_over: np.ndarray | None,
) -> list[dict]:
    """Build the list of per-channel JSON objects."""
    channels = []
    n_channels = mmm.n_media_channels

    for c in range(n_channels):
        channel = {
            "channel_index": c,
            "channel_name": mmm.media_names[c],
        }

        # spend_to_impressions
        if should_skip_channel[c]:
            channel["spend_to_impressions"] = None
        else:
            channel["spend_to_impressions"] = _extract_cost_model_params(cost_models[c])

        # impressions_to_target (closed-form parameters)
        channel["impressions_to_target"] = _build_channel_transform_params(
            mmm, data_to_fit, c, geo_idx, adstock_carry_over
        )

        # Piecewise linear curves
        if should_skip_channel[c]:
            channel["spend_to_target"] = None
            channel["spend_to_cost_per_target"] = None
        else:
            spend = spend_ranges[:, c]
            target = target_predictions[:, c]

            channel["spend_to_target"] = {
                "type": "piecewise_linear",
                "x_points": [float(x) for x in spend],
                "y_points": [float(y) for y in target],
            }

            cpt_y = []
            for s, t in zip(spend, target):
                if t > 0:
                    cpt_y.append(float(s / t))
                else:
                    cpt_y.append(None)

            channel["spend_to_cost_per_target"] = {
                "type": "piecewise_linear",
                "x_points": [float(x) for x in spend],
                "y_points": cpt_y,
            }

        channels.append(channel)

    return channels


def _write_response_curves_json(
    results_dir: str,
    channels_data: list[dict],
    model_type: str,
    geo_name: str | None = None,
) -> None:
    """Write a single response_curves JSON file."""
    data = {
        "model_type": model_type,
        "channels": channels_data,
    }
    filename = (
        f"response_curves_{geo_name}.json"
        if geo_name is not None
        else "response_curves.json"
    )
    with open(os.path.join(results_dir, filename), "w") as f:
        json.dump(data, f)


def _extract_and_export_response_curves(
    mmm: LightweightMMM,
    data_to_fit: DataToFit,
    results_dir: str,
    costs_per_day_unscaled: np.ndarray,
    input_data: InputData,
    steps: int = 50,
) -> None:
    """Extract response curve data and write to JSON files.

    Generates both:
    - Closed-form parameters (using median trace values) for optimiser use
    - Piecewise linear curves (50 points) for visualisation

    For single-geo models, writes response_curves.json.
    For hierarchical models, writes response_curves_{geo_name}.json per geo.
    """
    media = mmm.media
    media_scaler = data_to_fit.media_scaler
    target_scaler = data_to_fit.target_scaler
    n_channels = mmm.n_media_channels

    # -- Generate piecewise linear data (same core logic as plot_response_curves) --

    media_mins = media.min(axis=0)
    media_maxes = media.max(axis=0)  # percentage_add = 0.0

    extra_features = None
    if mmm._extra_features is not None:
        extra_features = jnp.expand_dims(mmm._extra_features.mean(axis=0), axis=0)

    media_ranges = jnp.expand_dims(
        jnp.linspace(start=media_mins, stop=media_maxes, num=steps), axis=0
    )

    make_predictions = jax.vmap(
        jax.vmap(
            _make_single_prediction,
            in_axes=(None, 0, None, None),
            out_axes=0,
        ),
        in_axes=(None, 0, None, None),
        out_axes=1,
    )

    diagonal = jnp.repeat(jnp.eye(n_channels), steps, axis=0).reshape(
        n_channels, steps, n_channels
    )

    prediction_offset = mmm.predict(
        media=jnp.zeros((1, *media.shape[1:])),
        extra_features=extra_features,
    ).mean(axis=0)

    if media.ndim == 3:
        diagonal = jnp.expand_dims(diagonal, axis=-1)
        prediction_offset = jnp.expand_dims(prediction_offset, axis=0)

    mock_media = media_ranges * diagonal
    predictions = jnp.squeeze(a=make_predictions(mmm, mock_media, extra_features, None))
    predictions = predictions - prediction_offset

    media_ranges_sq = jnp.squeeze(media_ranges)

    if target_scaler:
        predictions = target_scaler.inverse_transform(predictions)
    if media_scaler:
        media_ranges_unscaled = media_scaler.inverse_transform(media_ranges_sq)
    else:
        media_ranges_unscaled = media_ranges_sq

    # -- Compute carry-over state (once, for all channels) --

    adstock_carry_over = None
    if mmm.model_name in ("adstock", "hill_adstock"):
        adstock_carry_over = _compute_adstock_carry_over(mmm)

    # -- Build JSON per geo --

    if media.ndim == 3:
        # Geo model: per-geo cost models, per-geo predictions
        n_geos = media.shape[2]
        media_unscaled_full = (
            media_scaler.inverse_transform(media) if media_scaler else media
        )

        for geo_idx in range(n_geos):
            geo_name = data_to_fit.geo_names[geo_idx]

            # Per-geo cost models
            geo_cost_models = _train_cost_models(
                media=np.asarray(media_unscaled_full[:, :, geo_idx]),
                costs_per_day=np.asarray(costs_per_day_unscaled[:, :, geo_idx]),
                names=mmm.media_names,
            )
            geo_skip = [bool(m is None) for m in geo_cost_models]

            # Per-geo data
            geo_media = np.asarray(media_ranges_unscaled[:, :, geo_idx])
            geo_preds = np.asarray(predictions[:, :, geo_idx])

            geo_spend = _predict_costs_for_media_units(
                media=geo_media,
                channel_axis=1,
                cost_models=geo_cost_models,
            )

            channels_data = _build_channels_json(
                mmm,
                data_to_fit,
                geo_cost_models,
                geo_skip,
                geo_spend,
                geo_preds,
                geo_idx=geo_idx,
                adstock_carry_over=adstock_carry_over,
            )
            _write_response_curves_json(
                results_dir, channels_data, mmm.model_name, geo_name=geo_name
            )
    else:
        # Non-hierarchical model
        cost_models = _train_cost_models(
            media=(media_scaler.inverse_transform(media) if media_scaler else media),
            costs_per_day=costs_per_day_unscaled,
            names=mmm.media_names,
        )
        should_skip = [bool(m is None) for m in cost_models]

        spend_ranges = _predict_costs_for_media_units(
            media=np.asarray(media_ranges_unscaled),
            channel_axis=1,
            cost_models=cost_models,
        )

        channels_data = _build_channels_json(
            mmm,
            data_to_fit,
            cost_models,
            should_skip,
            np.asarray(spend_ranges),
            np.asarray(predictions),
            geo_idx=None,
            adstock_carry_over=adstock_carry_over,
        )
        _write_response_curves_json(results_dir, channels_data, mmm.model_name)
