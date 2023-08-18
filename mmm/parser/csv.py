from mmm.constants import constants

import numpy as np
import pandas as pd


def _parse_csv_shared(data_fname: str, config: dict) -> pd.DataFrame:
    # allow config option for name of date column;
    # if not provided, use "date" (case sensitive)
    index_col = config.get("date_col", "date")

    data_df = pd.read_csv(data_fname, index_col=index_col)

    if "total" in config.get("data_rows", {}):
        total_rows_expected = config["data_rows"]["total"]
        assert (
            total_rows_expected == data_df.shape[0]
        ), f"lhs={total_rows_expected}, rhs={data_df.shape[0]}"

    if "start_date" in config["data_rows"] and "end_date" in config["data_rows"]:
        start = config["data_rows"]["start_date"].isoformat()
        end = config["data_rows"]["end_date"].isoformat()
        data_df = data_df.loc[start:end]
    elif "to_use" in config["data_rows"]:
        data_df = data_df.tail(config["data_rows"]["to_use"])

    if "ignore_cols" in config:
        data_df = data_df.drop(columns=config["ignore_cols"])

    return data_df


def parse_csv_generic(data_fname: str, config: dict):
    """
    :param data_fname: full path to file for raw data
    :param config: config dictionary (from yaml file)
    :return: dict with format {
            KEY_GRANULARITY: granularity,
            KEY_OBSERVATIONS: observations,
            KEY_METRICS: { metric_name: [ metric values ], ... }
        }
    """
    data_df = _parse_csv_shared(data_fname, config)

    date_strs = data_df.index.to_numpy()

    metric_dict = {}
    for column in data_df.columns:
        metric_dict[column] = data_df[column].to_numpy(dtype=np.float64)

    data_dict = {
        constants.KEY_GRANULARITY: config.get("raw_data_granularity"),
        constants.KEY_OBSERVATIONS: data_df.shape[0],
        constants.KEY_DATE_STRS: date_strs,
        constants.KEY_METRICS: metric_dict,
    }

    return data_dict


def csv_to_df_generic(data_fname: str, config: dict) -> pd.DataFrame:
    """
    Parse a CSV to a DataFrame (generic implementation).

    Args:
        data_fname: full path to file for raw data
        config: config dictionary (from yaml file)

    Returns:
        DataFrame
    """
    data_df = _parse_csv_shared(data_fname, config)
    data_df.index = pd.DatetimeIndex(pd.to_datetime(data_df.index, format="%Y-%m-%d"), freq="D")

    # rename the columns in the dataframes from CSV column names to display name + "cost"/"volume"
    rename_cols = {}
    for i, media_config in enumerate(config.get("media", [])):
        display_name = media_config.get("display_name")
        impressions_col = media_config.get("impressions_col")
        spend_col = media_config.get("spend_col")
        rename_cols[spend_col] = f"{display_name} {constants.DATA_FRAME_COST_SUFFIX}"
        rename_cols[impressions_col] = f"{display_name} {constants.DATA_FRAME_IMPRESSIONS_SUFFIX}"

    return data_df.rename(columns=rename_cols)
