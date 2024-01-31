import pandas as pd

EOD_PERIODS = [
    "morning_peak",
    "morning_valley",
    "lunch",
    "afternoon_valley",
    "afternoon_peak",
    "night_valley",
    "night",
]


def add_days_and_periods(
    df,
    time_column="tiemposubida",
    time_target_column="eod_period",
    merge_peaks=False,
    day_target_column="eod_day",
):
    triples = [
        ("6:01", "7:31", "morning_peak_1"),
        ("7:31", "9:01", "morning_peak_2"),
        ("9:01", "12:01", "morning_valley"),
        ("12:01", "14:01", "lunch"),
        ("14:01", "17:31", "afternoon_valley"),
        ("17:31", "20:31", "afternoon_peak"),
        ("20:31", "23:01", "night_valley"),
        ("23:01", "0:00", "night"),
        ("0:00", "6:01", "night"),
    ]

    index = pd.DatetimeIndex(df[time_column])
    with_periods = df.assign(**{time_target_column: "n/a", day_target_column: "n/a"})

    for start, end, name in triples:
        index_values = index.indexer_between_time(start, end, include_end=False)
        with_periods[time_target_column].iloc[index_values] = name

    if merge_peaks:
        with_periods[time_target_column] = (
            with_periods[time_target_column]
            .replace("morning_peak_2", "morning_peak")
            .replace("morning_peak_1", "morning_peak")
        )

    dow = with_periods[time_column].dt.dayofweek

    if not dow[dow < 4].empty:
        with_periods[day_target_column].loc[dow[dow < 4].index] = "monday-thursday"
    if not dow[dow == 4].empty:
        with_periods[day_target_column].loc[dow[dow == 4].index] = "friday"
    if not dow[dow == 5].empty:
        with_periods[day_target_column].loc[dow[dow == 5].index] = "saturday"
    if not dow[dow == 6].empty:
        with_periods[day_target_column].loc[dow[dow == 6].index] = "sunday"

    return with_periods
