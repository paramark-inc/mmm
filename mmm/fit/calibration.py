"""
Experiment calibration for lightweight_mmm.

An experiment (e.g. a geo lift test) measures a channel's cost per incremental target (CPIC) as
a range. Each entry under the config's `calibration` key turns that range into a measured lift
over the experiment window, which lightweight_mmm adds to the model as a likelihood term:

    calibration:
      - channels: [OTT]          # media display_names; contributions are summed
        start_date: 2026-05-03   # experiment window, inclusive
        end_date: 2026-06-27
        cpic_low: 1880
        cpic_high: 2700
        interval: 0.9            # probability that the true CPIC is in [cpic_low, cpic_high]

The measured lift is (spend in window) / CPIC, so lift_low = spend / cpic_high and
lift_high = spend / cpic_low.  We model it as Normal(mean=midpoint, sd=half-width / z).

Limitations:
  * The model's contribution in the window is its *average* effect over the window.  A pulse-up
    test measures the *marginal* CPIC of the added spend, which is higher when returns diminish.
  * Contribution in the window includes carryover from spend before the window and excludes
    carryover after it.
  * National models only, and not for log-scale targets (contributions aren't additive there).
"""

from dataclasses import dataclass
from datetime import date, timedelta
from statistics import NormalDist

import jax.numpy as jnp
import numpy as np
import pandas as pd

from mmm.constants import constants
from mmm.data import DataToFit

DEFAULT_INTERVAL = 0.9

_PERIOD_DAYS = {
    constants.GRANULARITY_DAILY: 1,
    constants.GRANULARITY_WEEKLY: 7,
    constants.GRANULARITY_TWO_WEEKS: 14,
    constants.GRANULARITY_FOUR_WEEKS: 28,
}


@dataclass
class CalibrationEntry:
    """One experiment, resolved against the data to fit (unscaled units)."""

    channels: list[str]
    start_date: date
    end_date: date
    cpic_low: float
    cpic_high: float
    interval: float
    row_mask: np.ndarray  # [time] 1.0 for train rows overlapping the window
    channel_mask: np.ndarray  # [channel] 1.0 for the experiment's channels
    window_spend: float
    lift_mean: float
    lift_sd: float

    def to_dict(self) -> dict:
        """Plain values for fit_params.yaml."""
        return {
            "channels": list(self.channels),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "cpic_low": float(self.cpic_low),
            "cpic_high": float(self.cpic_high),
            "interval": float(self.interval),
            "window_rows": int(self.row_mask.sum()),
            "window_spend": float(self.window_spend),
            "lift_mean": float(self.lift_mean),
            "lift_sd": float(self.lift_sd),
        }


def _to_date(value, field: str, idx: int) -> date:
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError) as e:
        raise ValueError(f"calibration[{idx}].{field} is not a date: {value!r}") from e


def build_calibration(config: dict, data_to_fit: DataToFit) -> list[CalibrationEntry]:
    """
    Resolve the config's `calibration` entries against data_to_fit.

    :param config: config object loaded from YAML file
    :param data_to_fit: DataToFit instance
    :return: list of CalibrationEntry, empty when the config has no calibration
    """
    entries_config = config.get("calibration") or []
    if not entries_config:
        return []

    if data_to_fit.geo_names is not None:
        raise ValueError("calibration is only supported for national models")
    if data_to_fit.target_is_log_scale:
        raise ValueError("calibration is not supported with log_scale_target")

    num_train_rows = data_to_fit.media_data_train_scaled.shape[0]
    period_starts = pd.to_datetime(data_to_fit.date_strs[:num_train_rows]).date
    period_days = _PERIOD_DAYS[data_to_fit.time_granularity]

    costs_by_row = np.asarray(
        data_to_fit.media_costs_scaler.inverse_transform(
            data_to_fit.media_costs_by_row_train_scaled
        )
    )

    entries = []
    for idx, entry in enumerate(entries_config):
        channels = entry.get("channels")
        if not channels or not isinstance(channels, list):
            raise ValueError(f"calibration[{idx}].channels must be a non-empty list")
        unknown = [c for c in channels if c not in data_to_fit.media_names]
        if unknown:
            raise ValueError(f"calibration[{idx}] has unknown channels: {unknown}")

        start_date = _to_date(entry.get("start_date"), "start_date", idx)
        end_date = _to_date(entry.get("end_date"), "end_date", idx)
        if end_date < start_date:
            raise ValueError(f"calibration[{idx}] end_date is before start_date")

        cpic_low = float(entry.get("cpic_low", 0))
        cpic_high = float(entry.get("cpic_high", 0))
        if not 0 < cpic_low < cpic_high:
            raise ValueError(f"calibration[{idx}] needs 0 < cpic_low < cpic_high")

        interval = float(entry.get("interval", DEFAULT_INTERVAL))
        if not 0 < interval < 1:
            raise ValueError(f"calibration[{idx}].interval must be between 0 and 1")

        # A row covers [period_start, period_start + period_days); use it if it overlaps the
        # window.  Spend and contribution use the same rows, so partial weeks stay consistent.
        row_mask = np.array(
            [
                1.0 if s <= end_date and s + timedelta(days=period_days - 1) >= start_date else 0.0
                for s in period_starts
            ]
        )
        if row_mask.sum() == 0:
            raise ValueError(
                f"calibration[{idx}] window {start_date}..{end_date} has no rows in the "
                "training data"
            )

        channel_mask = np.array([1.0 if n in channels else 0.0 for n in data_to_fit.media_names])

        window_spend = float((costs_by_row * row_mask[:, None] * channel_mask[None, :]).sum())
        if window_spend <= 0:
            raise ValueError(f"calibration[{idx}] channels {channels} have no spend in the window")

        lift_low = window_spend / cpic_high
        lift_high = window_spend / cpic_low
        z = NormalDist().inv_cdf(0.5 + interval / 2)

        entries.append(
            CalibrationEntry(
                channels=list(channels),
                start_date=start_date,
                end_date=end_date,
                cpic_low=cpic_low,
                cpic_high=cpic_high,
                interval=interval,
                row_mask=row_mask,
                channel_mask=channel_mask,
                window_spend=window_spend,
                lift_mean=(lift_low + lift_high) / 2,
                lift_sd=(lift_high - lift_low) / (2 * z),
            )
        )

    return entries


def to_model_calibration(entries: list[CalibrationEntry], data_to_fit: DataToFit) -> dict:
    """
    Stack entries into the arrays lightweight_mmm's model expects, in scaled target units.
    """
    target_scaler = data_to_fit.target_scaler
    return {
        "row_mask": jnp.array(np.stack([e.row_mask for e in entries])),
        "channel_mask": jnp.array(np.stack([e.channel_mask for e in entries])),
        "lift_mean": target_scaler.transform(jnp.array([e.lift_mean for e in entries])),
        "lift_sd": target_scaler.transform(jnp.array([e.lift_sd for e in entries])),
    }
