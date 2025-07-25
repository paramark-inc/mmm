from multiprocessing.managers import Value
from mmm.constants import constants

import numpy as np
import pandas as pd


def _parse_csv_shared(
    data_fname: str,
    config: dict,
    keep_ignore_cols: bool = False,
    geo_filter: str = None,
) -> pd.DataFrame:
    """
    parse a CSV for MMM

    Args:
        data_fname: filename
        config: config dict
        keep_ignore_cols: True to preserve the ignore_cols
        geo_filter: include only rows with geo_col = this value

    Returns:
        Data Frame with

        index:
            [geo, date] for geo-level data, unless a geo_filter is specified
            date for aggregated data

        columns:
            columns in CSV minus ignore_cols (unless keep_ignore_cols), date, geo_col

        date range:
            as dictated by data_rows

        geos:
            all, unless geo_filter, in which case, only rows with geo_col = geo_filter
    """
    # allow config option for name of date column;
    # if not provided, use "date" (case sensitive)
    date_col = config.get("date_col", "date")

    geo_col = config.get("geo_col", None)

    data_df = pd.read_csv(data_fname)

    # Converting to datetime is necessary because we sort the index below, and makes the date range
    # slice below resilient to an unordered CSV.
    data_df[date_col] = pd.to_datetime(data_df[date_col])

    if geo_filter:
        data_df = data_df[data_df[config["geo_col"]] == geo_filter]
        data_df = data_df.drop(columns=[config["geo_col"]])

    # If there's a geo_filter, we just dropped the geo_col, making this equivalent to a non-geo
    # case.
    if not geo_col or geo_filter:
        has_geo = False
        data_df = data_df.set_index(date_col)
    else:
        has_geo = True
        data_df = data_df.set_index([geo_col, date_col])
        # sort_index is required to work around "MultiIndex slicing requires the index to be lexsorted"
        # errors.
        data_df = data_df.sort_index(level=[geo_col, date_col], ascending=[True, True])

    if "total" in config.get("data_rows", {}):
        total_rows_expected = config["data_rows"]["total"]
        assert (
            total_rows_expected == data_df.shape[0]
        ), f"lhs={total_rows_expected}, rhs={data_df.shape[0]}"

    # Lwmmm configs have start_date and end_date, but robyn configs have start_period_label and end_period_label.
    start_date_config = "start_date"
    end_date_config = "end_date"
    if "start_period_label" in config["data_rows"]:
        start_date_config = "start_period_label"
        end_date_config = "end_period_label"

    if start_date_config in config["data_rows"] and end_date_config in config["data_rows"]:
        # with yaml we may have a string or already a datetime object
        # if a string we assume it's already isoformat
        if isinstance(config["data_rows"][start_date_config], str):
           start = config["data_rows"][start_date_config]
           end = config["data_rows"][end_date_config]
        else:
            start = config["data_rows"][start_date_config].isoformat()
            end = config["data_rows"][end_date_config].isoformat()

        if has_geo:
            # Compute a slice that includes all geos but only dates between start and end
            data_df = data_df.loc[(slice(None), slice(start, end)), :]
        else:
            data_df = data_df.loc[start:end]
    elif "to_use" in config["data_rows"]:
        data_df = data_df.tail(config["data_rows"]["to_use"])

    if not keep_ignore_cols and "ignore_cols" in config:
        data_df = data_df.drop(columns=config["ignore_cols"])

    # Require that geo data is uniform (i.e. that all geos have the same dates)
    if has_geo:
        geo_values = data_df.index.levels[0].to_list()

        first_geo = geo_values[0]
        first_geo_dates = data_df.loc[first_geo].index.values

        remaining_geos = geo_values[1:]

        for this_geo in remaining_geos:
            this_geo_dates = data_df.loc[this_geo].index.values
            if not np.array_equal(first_geo_dates, this_geo_dates):
                raise ValueError(
                    f"geos '{first_geo}' and '{this_geo}' have data for different dates"
                )

    # Now that we have filtered down to the desired date range, we can enforce that the
    # dataset has all days between start and end.  Setting the frequency on the index
    # enforces this constraint.
    if data_df.index.nlevels > 1:
        # Geo dataset -> date is at level[1]
        # We need to call remove_unused_levels() because for MultiIndex, deleted keys
        # remain in the index until you refresh it with this call.
        data_df.index = data_df.index.remove_unused_levels()
        data_df.index.levels[1].freq = "D"
    else:
        # National dataset
        data_df.index.freq = "D"

    return data_df


def parse_csv_generic(
    data_fname: str,
    config: dict,
    geo_filter: str = None,
):
    """
    :param data_fname: full path to file for raw data
    :param config: config dictionary (from yaml file)
    :param geo_filter: geo to filter the data on, or None to parse data for all geos

    :return: dict with format {
            KEY_GRANULARITY: granularity,
            KEY_OBSERVATIONS: observations,
            KEY_METRICS:
              for a non geo model: { metric_name: [ metric values ], ... }
              for a geo model: { geo: { metric_name: [ metric values ], ... } }
        }
    """
    data_df = _parse_csv_shared(data_fname, config, geo_filter=geo_filter)

    has_geo_data = config.get("geo_col", None) and not geo_filter

    if has_geo_data:
        # remove_unused_levels() refreshes the index labels returned by index.levels[1].  Calling
        # it is necessary because otherwise index.levels[1] would return the labelled for
        # deleted rows.  The astype(str) is needed to remove the timestamp portion of the datetime.
        data_df.index = data_df.index.remove_unused_levels()
        date_strs = data_df.index.levels[1].astype(str).to_numpy()
        geo_names = data_df.index.levels[0].values.tolist()
    else:
        date_strs = data_df.index.astype(str).to_numpy()
        # note that this includes both national data and geo data with a geo filter
        geo_names = None

    metric_dict = {}
    for column in data_df.columns:
        if has_geo_data:
            for geo in geo_names:
                if geo not in metric_dict:
                    metric_dict[geo] = {}
                metric_dict[geo][column] = data_df.loc[geo][column].to_numpy(dtype=np.float64)
        else:
            metric_dict[column] = data_df[column].to_numpy(dtype=np.float64)

    data_dict = {
        constants.KEY_GRANULARITY: config.get("raw_data_granularity"),
        constants.KEY_OBSERVATIONS: date_strs.shape[0],
        constants.KEY_DATE_STRS: date_strs,
        constants.KEY_METRICS: metric_dict,
        # geo_names is None for national data and geo data with a geo filter
        constants.KEY_GEO_NAMES: geo_names,
    }

    return data_dict


def csv_to_df_generic(
    data_fname: str,
    config: dict,
    keep_ignore_cols: bool = False,
    geo_filter: str = None,
) -> pd.DataFrame:
    """
    Parse a CSV to a DataFrame (generic implementation).

    Args:
        data_fname: full path to file for raw data
        config: config dictionary (from yaml file)
        keep_ignore_cols: True to add the ignore_cols to the dataframe, False otherwise
        geo_filter: geo to filter the data on, or None to include data for all geos

    Returns:
        DataFrame
    """
    data_df = _parse_csv_shared(data_fname, config, keep_ignore_cols, geo_filter)

    # for val in data_df.index.values:
    #     print(val)

    return data_df
