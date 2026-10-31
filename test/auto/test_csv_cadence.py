import io
from datetime import date, timedelta

import numpy as np
import pytest

from mmm.parser.csv import csv_to_df_generic, parse_csv_generic


# --- Helpers: synthetic CSVs and assertions shared by both native entrypoints ---
LAYOUTS = ["national", "geo", "filtered"]
READERS = [csv_to_df_generic, parse_csv_generic]


def read_csv(reader, layout, dates, granularity="daily", data_rows=None, other_dates=None):
    """Keep metric values tied to source rows so date alignment is observable."""
    has_geo = layout != "national"
    rows = ["date,metric,ignored" + (",geo" if has_geo else "")]
    for geo_index, geo in enumerate(["a", "b"] if has_geo else [None]):
        geo_dates = other_dates if geo == "b" and other_dates is not None else dates
        for index, day in enumerate(geo_dates):
            rows.append(f"{day},{100 * geo_index + index},999" + (f",{geo}" if has_geo else ""))
    config = {"data_rows": data_rows or {}, "ignore_cols": ["ignored"]}
    if granularity is not None:
        config["raw_data_granularity"] = granularity
    if has_geo:
        config["geo_col"] = "geo"
    return reader(
        io.StringIO("\n".join(rows)),
        config,
        geo_filter="a" if layout == "filtered" else None,
    )


def assert_rows(result, reader, layout, dates, values):
    expected_dates = [str(day) for day in dates]
    if reader is csv_to_df_generic:
        assert result.columns.tolist() == ["metric"]
        frames = (
            [(result.loc[geo], offset) for geo, offset in [("a", 0), ("b", 100)]]
            if layout == "geo"
            else [(result, 0)]
        )
        for frame, offset in frames:
            assert frame.index.strftime("%Y-%m-%d").tolist() == expected_dates
            np.testing.assert_array_equal(frame["metric"], np.asarray(values) + offset)
    else:
        assert result["observations"] == len(dates)
        np.testing.assert_array_equal(result["date_strs"], expected_dates)
        assert result["geo_names"] == (["a", "b"] if layout == "geo" else None)
        metrics = (
            [(result["metrics"][geo], offset) for geo, offset in [("a", 0), ("b", 100)]]
            if layout == "geo"
            else [(result["metrics"], 0)]
        )
        for metric, offset in metrics:
            assert set(metric) == {"metric"}
            np.testing.assert_array_equal(metric["metric"], np.asarray(values) + offset)


# --- End helpers ---


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("anchor", range(7))
def test_weekly_any_anchor(reader, layout, anchor):
    dates = [date(2026, 10, 5) + timedelta(days=anchor + 7 * i) for i in range(3)]
    result = read_csv(reader, layout, dates, "weekly")
    assert_rows(result, reader, layout, dates, [0, 1, 2])
    if reader is parse_csv_generic:
        assert result["granularity"] == "weekly"


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("granularity,step", [("daily", 1), (None, 1), ("weekly", 7)])
@pytest.mark.parametrize("count", [1, 2])
def test_short_series(reader, layout, granularity, step, count):
    dates = [date(2026, 10, 6) + timedelta(days=step * i) for i in range(count)]
    assert_rows(read_csv(reader, layout, dates, granularity), reader, layout, dates, range(count))


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("granularity,step", [("daily", 1), (None, 1), ("weekly", 7)])
@pytest.mark.parametrize("selection", ["dates", "labels", "tail"])
def test_inclusive_filtering(reader, layout, granularity, step, selection):
    dates = [date(2026, 10, 6) + timedelta(days=step * i) for i in range(5)]
    if selection == "tail":
        # Geo tail counts rows across the panel. Keeping all rows of the last
        # geo would change the existing shared-grid contract, so retain both.
        rows = {"to_use": 10 if layout == "geo" else 3}
        indices = range(5) if layout == "geo" else range(2, 5)
    else:
        start, end = (
            ("start_date", "end_date")
            if selection == "dates"
            else ("start_period_label", "end_period_label")
        )
        rows = {start: str(dates[1]), end: str(dates[3])}
        indices = range(1, 4)
    assert_rows(
        read_csv(reader, layout, dates, granularity, rows),
        reader,
        layout,
        [dates[i] for i in indices],
        list(indices),
    )


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "granularity,offsets",
    [
        ("daily", [0, 2]),
        (None, [0, 7, 14]),
        ("weekly", [0, 14]),
        ("weekly", [0, 7, 21]),
        ("weekly", [0, 7, 15]),
        ("two_weeks", [0, 14, 28]),
        ("four_weeks", [0, 28, 56]),
    ],
)
def test_invalid_spacing(reader, layout, granularity, offsets):
    dates = [date(2026, 10, 6) + timedelta(days=i) for i in offsets]
    with pytest.raises(ValueError):
        read_csv(reader, layout, dates, granularity)


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("granularity,step", [("daily", 1), ("weekly", 7)])
def test_shared_geo_grid_and_duplicates(reader, granularity, step):
    dates = [date(2026, 10, 6) + timedelta(days=step * i) for i in range(3)]
    for other_dates in [
        dates[:-1],
        [day + timedelta(days=1) for day in dates],
        [dates[0], dates[0], *dates[1:]],
    ]:
        with pytest.raises(ValueError, match="different dates"):
            read_csv(reader, "geo", dates, granularity, other_dates=other_dates)
    for layout in ["national", "filtered"]:
        with pytest.raises(ValueError):
            read_csv(reader, layout, [dates[0], dates[0], *dates[1:]], granularity)


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("granularity", ["daily", None, "weekly"])
def test_empty_series_existing_behavior(reader, layout, granularity):
    if layout == "geo":
        with pytest.raises(IndexError):
            read_csv(reader, layout, [], granularity)
    else:
        assert_rows(read_csv(reader, layout, [], granularity), reader, layout, [], [])
