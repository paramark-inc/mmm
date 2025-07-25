# Historical prediction functions needed for plots

import jax.numpy as jnp
import numpy as np
import pandas as pd

from mmm.data.data_to_fit import DataToFit

from lightweight_mmm.lightweight_mmm import LightweightMMM


def _safe_div(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    Helper function to avoid divide by zero runtime warnings.  Replaces zeroes in the denominator
    with nan, resulting in nans in the output.

    Args:
        numerator: numerator array
        denominator: denominator array

    Returns:
        numerator / denominator

    """
    return numerator / np.where(np.isclose(denominator, 0.0), np.nan, denominator)


def _convert_model_historical_predictions_to_daily(
    numdays: int,
    numperiods: int,
    numsamples: int,
    numchannels: int,
    period_baseline_contribution_adjusted_t: np.ndarray,
    period_baseline_contribution_adjusted_st: np.ndarray,
    period_media_contribution_t: np.ndarray,
    period_media_contribution_st: np.ndarray,
    period_media_contribution_tc: np.ndarray,
    period_media_contribution_stc: np.ndarray,
    period_unscaled_prediction_st: np.ndarray,
) -> dict:
    """

    Args:
        numdays: total number of days
        numperiods: for weekly granularity, pass the number of weeks here.  For two week,
           granularity, this should be the number of 14 day periods. And so on for others.
           This matches the number of array elements in the model arrays.
        numsamples:
        numchannels:
        period_baseline_contribution_adjusted_t:
        period_baseline_contribution_adjusted_st:
        period_media_contribution_t:
        period_media_contribution_st:
        period_media_contribution_tc:
        period_media_contribution_stc:
        period_unscaled_prediction_st:

    Returns:

    """
    if numdays % numperiods != 0:
        raise ValueError("numperiods is not an even multiple of numdays")

    days_per_period = int(numdays / numperiods)

    daily_baseline_contribution_adjusted_t = np.zeros(shape=(numdays,))
    daily_baseline_contribution_adjusted_st = np.zeros(shape=(numsamples, numdays))
    daily_media_contribution_t = np.zeros(shape=(numdays,))
    daily_media_contribution_st = np.zeros(shape=(numsamples, numdays))
    daily_media_contribution_tc = np.zeros(shape=(numdays, numchannels))
    daily_media_contribution_stc = np.zeros(shape=(numsamples, numdays, numchannels))
    daily_unscaled_prediction_st = np.zeros(shape=(numsamples, numdays))

    # Computing the daily values via a loop may be slower than doing it via a single numpy
    # operation -- although I'm not sure how to do that in numpy.
    for period_num in range(numperiods):
        daystart = period_num * days_per_period
        dayend = daystart + days_per_period - 1

        # broadcast the values for this week across the columns mapping to the corresponding days
        # in the daily arrays.  When the source array has more than one dimension this requires
        # adding a 1-entry axis to the end so that the broadcast passes type checking.
        daily_baseline_contribution_adjusted_t[daystart : dayend + 1] = (
            period_baseline_contribution_adjusted_t[period_num] / days_per_period
        )
        perday_baseline_contribution_adjusted_s = (
            period_baseline_contribution_adjusted_st[:, period_num] / days_per_period
        )
        daily_baseline_contribution_adjusted_st[:, daystart : dayend + 1] = (
            perday_baseline_contribution_adjusted_s.reshape(numsamples, 1)
        )
        daily_media_contribution_t[daystart : dayend + 1] = (
            period_media_contribution_t[period_num] / days_per_period
        )
        perday_media_contribution_s = period_media_contribution_st[:, period_num] / days_per_period
        daily_media_contribution_st[:, daystart : dayend + 1] = perday_media_contribution_s.reshape(
            numsamples, 1
        )
        perday_media_contribution_c = period_media_contribution_tc[period_num, :] / days_per_period
        daily_media_contribution_tc[daystart : dayend + 1, :] = perday_media_contribution_c.reshape(
            1, numchannels
        )
        perday_media_contribution_sc = (
            period_media_contribution_stc[:, period_num, :] / days_per_period
        )
        daily_media_contribution_stc[:, daystart : dayend + 1, :] = (
            perday_media_contribution_sc.reshape(numsamples, 1, numchannels)
        )
        perday_unscaled_prediction_s = (
            period_unscaled_prediction_st[:, period_num] / days_per_period
        )
        daily_unscaled_prediction_st[:, daystart : dayend + 1] = (
            perday_unscaled_prediction_s.reshape(numsamples, 1)
        )

    return {
        "daily_unscaled_prediction_st": daily_unscaled_prediction_st,
        "daily_baseline_contribution_adjusted_t": daily_baseline_contribution_adjusted_t,
        "daily_baseline_contribution_adjusted_st": daily_baseline_contribution_adjusted_st,
        "daily_media_contribution_t": daily_media_contribution_t,
        "daily_media_contribution_st": daily_media_contribution_st,
        "daily_media_contribution_tc": daily_media_contribution_tc,
        "daily_media_contribution_stc": daily_media_contribution_stc,
    }


def _get_historical_predictions(
    config: dict,
    mmm: LightweightMMM,
    data_to_fit: DataToFit,
    geo_name: str,
) -> dict:
    """
    Gets the predictions needed to build the daily prediction dataframe.
    This code was pulled from LightweightMMM.plot.create_media_baseline_contribution_df and
    then modified for clarity and to translate the weekly value to daily ones.

    Args:
        config: yaml config
        mmm: model
        data_to_fit: DataToFit instance
        geo_name: geo name to filter on (geo models only)

    Returns:
        Dictionary of ndarrays with the different types of prediction data.
    """
    is_daily_model = ("data_groupby" not in config) or (not config["data_groupby"])

    # filter arrays down to the relevant geo:
    #   - media_transformed_stc has dimensions [samples, time, geos]
    #   - coef_media_sc has dimensions [samples, channels]
    #   - mu_st has dimensions [samples, time]
    #
    # We have to unscale the predictions in this block because we have a different scaler for each
    # geo, so we must unscale before we remove the geo dimension.
    if data_to_fit.geo_names is not None:
        geo_idx = data_to_fit.geo_names.index(geo_name)
        media_transformed_stc = mmm.trace["media_transformed"][:, :, :, geo_idx]
        coef_media_sc = mmm.trace["coef_media"][:, :, geo_idx]
        # prediction_stg is an ndarray with axes [samples, time, geos].  Values represent the
        # total target prediction for that observation (scaled).
        prediction_stg = mmm.trace["mu"]
        # unscaled_prediction_stg has axes [samples, time, geo].  Values represent the total
        # target prediction for that week (unscaled).
        unscaled_prediction_stg = data_to_fit.target_scaler.inverse_transform(prediction_stg)
        # unscaled_prediction_st has axes [samples, time].  Values represent the total
        # target prediction for that week (unscaled).
        unscaled_prediction_st = unscaled_prediction_stg[:, :, geo_idx]
        # prediction_st is an ndarray with axes [samples, time].  Values represent the
        # total target prediction for that observation (scaled).
        prediction_st = prediction_stg[:, :, geo_idx]
    else:
        media_transformed_stc = mmm.trace["media_transformed"]
        coef_media_sc = mmm.trace["coef_media"]
        # prediction_st is an ndarray with axes [samples, time].  Values represent the
        # total target prediction for that observation (scaled).
        prediction_st = mmm.trace["mu"]
        # unscaled_prediction_st has axes [samples, time].  Values represent the total
        # target prediction for that week (unscaled).
        unscaled_prediction_st = data_to_fit.target_scaler.inverse_transform(prediction_st)

    # s for samples, t for time, c for media channels
    einsum_str = "stc, sc->stc"

    # media_contribution_stc is an ndarray with axes [samples, time, channels].  Values
    # represent the number of unscaled target units attributable to that channel on that
    # observation
    media_contribution_stc = jnp.einsum(
        einsum_str,
        media_transformed_stc,
        coef_media_sc,
    )
    # media_contribution_tc is an ndarray with axes [time, channels].
    media_contribution_tc = np.median(media_contribution_stc, axis=0)
    # media_contribution_st is an ndarray with axes [samples, time].
    media_contribution_st = media_contribution_stc.sum(axis=2)
    # media_contribution_t has axes [time]
    media_contribution_t = np.median(media_contribution_st, axis=0)

    numsamples = media_contribution_st.shape[0]
    # numperiods is weeks / days depending on the model granularity
    numperiods = media_contribution_t.shape[0]
    numchannels = media_contribution_stc.shape[2]

    # baseline_contribution_st is an ndarray with axes [samples, time].  Values represent the
    # number of scaled target units for an observation attributable to baseline.
    baseline_contribution_st = prediction_st - media_contribution_st
    # baseline_contribution_t is an ndarray with axes [time].
    baseline_contribution_t = np.median(baseline_contribution_st, axis=0)

    # Adjust baseline contribution and prediction when there's any negative value.
    # In the case of a negative baseline, we ensure that the overall height of the curve matches the
    # daily prediction (mu), while dividing the area under the curve among the media channels
    # according to their relative contribution.  This results in a curve with the correct total
    # height, but where the media contribution will be decreased to compensate for the negative
    # baseline.  For this case, 'weekly_prediction_adjusted_t' will be higher
    # than the actual predictions.  By scaling the actual contribution numbers (below) based on
    # the daily prediction, we prevent this from impacting the overall height of the curve.

    # baseline_contribution_adjusted_t has axes [time] and values adjusted baseline
    # contribution in scaled target metric units
    baseline_contribution_adjusted_t = np.where(
        baseline_contribution_t < 0,
        0,
        baseline_contribution_t,
    )
    baseline_contribution_adjusted_st = np.where(
        baseline_contribution_st < 0,
        0,
        baseline_contribution_st,
    )

    if is_daily_model:
        numdays = numperiods
        daily_unscaled_prediction_st = unscaled_prediction_st
        daily_baseline_contribution_adjusted_t = baseline_contribution_adjusted_t
        daily_baseline_contribution_adjusted_st = baseline_contribution_adjusted_st
        daily_media_contribution_t = media_contribution_t
        daily_media_contribution_st = media_contribution_st
        daily_media_contribution_tc = media_contribution_tc
        daily_media_contribution_stc = media_contribution_stc
    else:
        data_groupby_to_days_per_period = {
            "week": 7,
            "two_weeks": 14,
            "four_weeks": 28,
        }
        days_per_period = data_groupby_to_days_per_period[config["data_groupby"]]
        numdays = numperiods * days_per_period
        dailyd = _convert_model_historical_predictions_to_daily(
            numdays,
            numperiods,
            numsamples,
            numchannels,
            baseline_contribution_adjusted_t,
            baseline_contribution_adjusted_st,
            media_contribution_t,
            media_contribution_st,
            media_contribution_tc,
            media_contribution_stc,
            unscaled_prediction_st,
        )
        daily_unscaled_prediction_st = dailyd["daily_unscaled_prediction_st"]
        daily_baseline_contribution_adjusted_t = dailyd["daily_baseline_contribution_adjusted_t"]
        daily_baseline_contribution_adjusted_st = dailyd["daily_baseline_contribution_adjusted_st"]
        daily_media_contribution_t = dailyd["daily_media_contribution_t"]
        daily_media_contribution_st = dailyd["daily_media_contribution_st"]
        daily_media_contribution_tc = dailyd["daily_media_contribution_tc"]
        daily_media_contribution_stc = dailyd["daily_media_contribution_stc"]

    # daily_unscaled_prediction_t has axes [time] and values as the total prediction for the day
    # in unscaled target metric units
    daily_unscaled_prediction_t = np.median(daily_unscaled_prediction_st, axis=0)

    # daily_adjusted_prediction_t has axes [time] and values adjusted prediction values in scaled
    # target metric units
    daily_unscaled_adjusted_prediction_t = (
        daily_baseline_contribution_adjusted_t + daily_media_contribution_t
    )
    daily_unscaled_adjusted_prediction_st = (
        daily_baseline_contribution_adjusted_st + daily_media_contribution_st
    )

    # daily_media_contribution_pct_tc has axes [time, channels].  Values represent the percentage of the
    # target output for that day attributable to that channel.
    daily_media_contribution_pct_tc = _safe_div(
        numerator=daily_media_contribution_tc,
        denominator=daily_unscaled_adjusted_prediction_t.reshape(numdays, 1),
    )

    # reshape the adjusted prediction array to an stc form so we can compute the ratio of media
    # contribution to adjusted prediction (below).
    daily_unscaled_adjusted_prediction_stc = daily_unscaled_adjusted_prediction_st.reshape(
        numsamples,
        numdays,
        1,
    )

    # we can have zeroes in the denominator where there are samples with bizarre estimates, so we
    # do a safe division which replaces zeroes in the denominator with nans, which produces nans
    # in the output.
    daily_media_contribution_pct_stc = _safe_div(
        numerator=daily_media_contribution_stc,
        denominator=daily_unscaled_adjusted_prediction_stc,
    )
    daily_media_contribution_pct_st = _safe_div(
        numerator=daily_media_contribution_st,
        denominator=daily_unscaled_adjusted_prediction_st,
    )

    # Adjust media pct contribution if the value is nan
    daily_media_contribution_pct_tc = np.nan_to_num(daily_media_contribution_pct_tc)
    daily_media_contribution_pct_stc = np.nan_to_num(daily_media_contribution_pct_stc)

    # daily_baseline_contribution_pct_t has axes [time] and values representing the daily
    # percentage of the target metric attributable to baseline (median).
    daily_baseline_contribution_pct_t = _safe_div(
        numerator=daily_baseline_contribution_adjusted_t,
        denominator=daily_unscaled_adjusted_prediction_t,
    )
    # Adjust baseline pct contribution if the value is nan
    daily_baseline_contribution_pct_t = np.nan_to_num(daily_baseline_contribution_pct_t)

    return {
        "daily_unscaled_prediction_t": daily_unscaled_prediction_t,
        "daily_baseline_contribution_pct_t": daily_baseline_contribution_pct_t,
        "daily_media_contribution_pct_stc": daily_media_contribution_pct_stc,
        "daily_media_contribution_pct_st": daily_media_contribution_pct_st,
    }


def create_historical_predictions_daily_df(
    config: dict,
    mmm: LightweightMMM,
    data_to_fit: DataToFit,
    geo_name: str = None,
) -> pd.DataFrame:
    """
    Create a data frame of predictions for the entire train plus test period

    Args:
        config: yaml config
        mmm: LightweightMMM model, already fitted and trained
        data_to_fit: DataToFit instance
        geo_name: geo_name to filter to, or None for global models.  Aggregating across geos in a
                  geo model is not supported so this parameter must be passed for geo models.

    Returns:
        DataFrame of predictions for the train + test period.  Values are total predicted
        target metric output (in unscaled predicted target metric units).

        The index has one row per day, regardless of the modelling period length.

        Columns are as follows
          baseline:median - amount attributable to baseline (median)
          incremental:<ci_lower_quantile> - amount attributed to incremental (i.e. 0.05 for 90% credibility interval)
          incremental:median - amount attributed to incremental (median)
          incremental:<ci_upper_quantile> - amount attributed to incremental (i.e. 0.95 for 90% credibility interval)
          total:median - total incremental output, baseline:median + incremental:median
          and then, for each channel
              <channel>:<ci_lower_quantile> - amount attributed to channel (i.e. 0.05 for 90% credibility interval)
              <channel>:median - amount attributed to channel (median)
              <channel>:<ci_upper_quantile> - amount attributed to channel (i.e. 0.95 for 90% credibility interval)


    """
    if data_to_fit.geo_names is not None:
        if not geo_name:
            raise ValueError("aggregating across geos is not supported")
    else:
        if geo_name:
            raise ValueError("geo_name passed for a global model")

    # This code doesn't yet support test data sets.  When we need that, we should add some code
    # here that calls predict() with the test data only, and a media_gap of zero.
    if data_to_fit.media_data_test_scaled.shape[0] > 0:
        raise ValueError("Feature not implemented")

    name_to_breakdown = _get_historical_predictions(
        config,
        mmm,
        data_to_fit,
        geo_name,
    )
    daily_unscaled_prediction_t = name_to_breakdown["daily_unscaled_prediction_t"]
    daily_baseline_contribution_pct_t = name_to_breakdown["daily_baseline_contribution_pct_t"]
    daily_media_contribution_pct_stc = name_to_breakdown["daily_media_contribution_pct_stc"]
    daily_media_contribution_pct_st = name_to_breakdown["daily_media_contribution_pct_st"]

    first_day = pd.to_datetime(data_to_fit.date_strs[0])
    groupby = config.get("data_groupby", None)
    if groupby == "week":
        last_day = pd.to_datetime(data_to_fit.date_strs[-1]) + pd.Timedelta(days=6)
    elif groupby == "two_weeks":
        last_day = pd.to_datetime(data_to_fit.date_strs[-1]) + pd.Timedelta(days=13)
    elif groupby == "four_weeks":
        last_day = pd.to_datetime(data_to_fit.date_strs[-1]) + pd.Timedelta(days=27)
    else:
        last_day = pd.to_datetime(data_to_fit.date_strs[-1])

    daily_index = pd.DatetimeIndex(
        pd.date_range(
            first_day,
            last_day,
            freq="D",
        ),
    )

    ci_lower_quantile, ci_upper_quantile = data_to_fit.get_ci_quantiles()

    totals_daily_df = pd.DataFrame(index=daily_index)
    # to compute the total target value attributable to baseline / incremental, we multiply by the
    # median value of the unscaled prediction.  This preserves uncertainty on the baseline /
    # incremental side but removes the uncertainty on the unscaled prediction side.  We lose
    # some statistical purity in the process in the name of keeping things simple.
    totals_daily_df["baseline:median"] = (
        daily_baseline_contribution_pct_t * daily_unscaled_prediction_t
    )
    incremental_cols = [
        f"incremental:{ci_lower_quantile}",
        "incremental:median",
        f"incremental:{ci_upper_quantile}",
    ]
    # daily_media_contribution_pct_qt has axes [quantile, time] and values as percentage of output
    # attributed as incremental for that quantile.
    daily_media_contribution_pct_qt = np.quantile(
        daily_media_contribution_pct_st, [ci_lower_quantile, 0.5, ci_upper_quantile], axis=0
    )

    totals_daily_df[incremental_cols] = (
        daily_media_contribution_pct_qt.T * daily_unscaled_prediction_t.reshape(-1, 1)
    )

    totals_daily_df["total:median"] = totals_daily_df[
        ["baseline:median", "incremental:median"]
    ].sum(axis=1)

    # Rather than writing this as a loop, we could probably compute it via a single numpy or pandas
    # operation that spans all of the channels.  This may be worth revisiting if we are trying
    # to speed this up.
    for idx, media_name in enumerate(data_to_fit.media_names):
        columns = [
            f"{media_name}:{ci_lower_quantile}",
            f"{media_name}:median",
            f"{media_name}:{ci_upper_quantile}",
        ]
        # daily_channel_contribution_pct_quantiles_qt has axes [quantile, time] and values of the
        # percent of target attributed as incremental to this particular channel
        daily_channel_contribution_pct_quantiles_qt = np.quantile(
            daily_media_contribution_pct_stc[:, :, idx],
            [ci_lower_quantile, 0.5, ci_upper_quantile],
            axis=0,
        )
        totals_daily_df[columns] = (
            daily_channel_contribution_pct_quantiles_qt.T
            * daily_unscaled_prediction_t.reshape(-1, 1)
        )

    # The sum of medians across all of the channels likely does not equal the blended
    # incremental:median.  Because the latter is more accurate (computed based on a proper blend),
    # and because we show each of these in our visualizations and customers will expect them to be
    # the same, we scale the medians for the individual channels so that the sums will match.
    # This is different than the scaling done by scale_historical_predictions_by_actuals() because
    # that function solves a different scaling problem -- the difference between predictions and
    # actuals.
    channel_median_columns = [f"{m['display_name']}:median" for m in config["media"]]
    daily_median_ratio = totals_daily_df["incremental:median"].div(
        totals_daily_df[channel_median_columns].sum(axis=1),
        axis=0,
    )
    totals_daily_df[channel_median_columns] = totals_daily_df[channel_median_columns].mul(
        daily_median_ratio,
        axis=0,
    )

    return totals_daily_df


def create_historical_predictions_yearly_df(
    config: dict,
    mmm: LightweightMMM,
    data_to_fit: DataToFit,
    geo_name: str = None,
) -> pd.DataFrame:
    totals_daily_df = create_historical_predictions_daily_df(config, mmm, data_to_fit, geo_name)

    # Adding daily values is not statistically pure, but we do it for simplicity.  A more correct
    # way to do this would be to preserve all of the sample-level resolution in the data, then roll
    # up to months, and then compute the <ci_lower_quantile> / median / <ci_upper_quantile>.
    totals_yearly_df = totals_daily_df.copy().resample("YS", closed="left", label="left").sum()

    return totals_yearly_df


def create_historical_predictions_monthly_df(
    config: dict,
    mmm: LightweightMMM,
    data_to_fit: DataToFit,
    geo_name: str = None,
) -> pd.DataFrame:
    totals_daily_df = create_historical_predictions_daily_df(config, mmm, data_to_fit, geo_name)

    # Adding daily values is not statistically pure, but we do it for simplicity.  A more correct
    # way to do this would be to preserve all of the sample-level resolution in the data, then roll
    # up to months, and then compute the <ci_lower_quantile> / median / <ci_upper_quantile>.
    totals_monthly_df = totals_daily_df.copy().resample("MS", closed="left", label="left").sum()

    return totals_monthly_df


def create_historical_predictions_weekly_df(
    config: dict,
    mmm: LightweightMMM,
    data_to_fit: DataToFit,
    geo_name: str = None,
) -> pd.DataFrame:
    totals_daily_df = create_historical_predictions_daily_df(config, mmm, data_to_fit, geo_name)

    # Adding daily values is not statistically pure, but we do it for simplicity.  A more correct
    # way to do this would be to preserve all of the sample-level resolution in the data, then roll
    # up to weeks, and then compute the <ci_lower_quantile> / median / <ci_upper_quantile>.
    totals_weekly_df = totals_daily_df.copy().resample("W", closed="left", label="left").sum()

    return totals_weekly_df


def scale_historical_predictions_by_actuals(
    config: dict,
    historical_predictions_monthly_df: pd.DataFrame,
    historical_actuals_monthly_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Re-scale historical monthly predictions so that the monthly totals will match the actual monthly
    totals.

    Args:
        config: yaml config
        historical_predictions_monthly_df: see create_historical_predictions_monthly_df
        historical_actuals_monthly_df: see create_monthly_rollup_df()

    Returns:
        Dataframe with the same index as the input dataframes.  Columns will be the same as the
        predictions dataframe.  Values are mapped to the scale of the actuals df.
    """

    # Match the index. If there are missing index/date in the predictions df, this will throw an error
    historical_predictions_monthly_df = historical_predictions_monthly_df.loc[
        historical_actuals_monthly_df.index
    ]

    scaled_historical_predictions_monthly_df = pd.DataFrame(
        index=historical_predictions_monthly_df.index
    )

    target_col = config["target_col"]
    # actual_per_predicted_monthly_series has the month and year as the index and the
    # actual total:predicted total ratio as the values
    actual_per_predicted_monthly_series = (
        historical_actuals_monthly_df[target_col]
        / historical_predictions_monthly_df["total:median"]
    )

    # Multiplying by a single ratio per month is not statistically pure.  A more correct way to do
    # this would be to scale the daily values and then roll up to monthly.  We do it this way for
    # simplicity.
    scaled_historical_predictions_monthly_df = historical_predictions_monthly_df.multiply(
        actual_per_predicted_monthly_series, axis=0
    )

    return scaled_historical_predictions_monthly_df
