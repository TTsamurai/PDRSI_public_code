import glob
from datetime import date
from datetime import datetime as dt
from datetime import timedelta

import pandas as pd
from calculate_technical_indicators import get_technical_indicators


def load_micro_data(micro_path: str) -> pd.DataFrame:
    micro_data = pd.read_csv(micro_path)
    ticker_code = micro_path[:-4].split("/")[-1].split("_")[0]
    if "-" in ticker_code:
        ticker_code = ticker_code.replace("-", ".")
    return micro_data, ticker_code


def turn_data_with_date_and_set_index(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = (
        df["created_at"]
        .apply(lambda x: x[:10])
        .apply(lambda x: dt.strptime(x, "%Y-%m-%d"))
        .apply(lambda x: date(x.year, x.month, x.day))
    )
    df = df.set_index("Date")
    return df


def fill_weekends_and_holidays_with_previous_values(
    data_technical: pd.DataFrame,
) -> pd.DataFrame:
    """
    Make sure to use this function after calculating the technical indicators
    """
    data_technical = data_technical.sort_values("Date")
    data_length_date = data_technical["Date"].iloc[-1] - data_technical["Date"].iloc[0]
    all_day = [
        data_technical["Date"].iloc[0] + timedelta(i)
        for i in range(data_length_date.days + 1)
    ]
    all_date_df = pd.DataFrame([all_day], index=["Date"]).T
    return (
        pd.merge(all_date_df, data_technical, on="Date", how="left")
        .fillna(method="ffill")
        .set_index("Date")
    )


def flatten_data_frame(data: pd.DataFrame, name):
    assert "Date" not in data.columns, "Set Date column as index"
    series = data.apply(lambda x: x.values, axis=1)
    return pd.DataFrame(series, columns=[name])


def get_micro_macro_aware_data(data_path: str = "data/"):

    df = (
        pd.read_csv(data_path + "filtered_data_more_than_10_smaller_than_100.csv")
        .sort_values(["author.id", "created_at"])
        .reset_index(drop=True)
    )
    df = turn_data_with_date_and_set_index(df=df)

    macro_data_path = data_path + "historical_data/SPIndex_historical.csv"
    micro_data_path = sorted(
        [
            path
            for path in glob.glob(data_path + "historical_data/*.csv")
            if path != macro_data_path
        ]
    )

    macro_data = pd.read_csv(macro_data_path)
    macro_technical = get_technical_indicators(macro_data)

    micro_data_list, ticker_code_list = zip(
        *[load_micro_data(micro_path=path) for path in micro_data_path]
    )
    micro_technical_list = [
        get_technical_indicators(micro_ind_data) for micro_ind_data in micro_data_list
    ]

    filled_macro_technical = fill_weekends_and_holidays_with_previous_values(
        macro_technical
    )
    macro_technical_compacted = flatten_data_frame(
        filled_macro_technical, "macro_technical"
    )

    filled_micro_technical_list = [
        fill_weekends_and_holidays_with_previous_values(micro_ind_technical)
        for micro_ind_technical in micro_technical_list
    ]
    micro_technical_compacted_list = [
        flatten_data_frame(filled_micro_ind_technical, "{}".format(ticker_ind_code))
        for filled_micro_ind_technical, ticker_ind_code in zip(
            filled_micro_technical_list, ticker_code_list
        )
    ]
    micro_all_df = pd.concat(micro_technical_compacted_list, axis=1)
    stacked_micro_all_df = (
        pd.DataFrame(micro_all_df.stack(), columns=["micro_technical"])
        .reset_index()
        .rename(columns={"level_1": "ticker"})
    )

    df_micro_aware = pd.merge(
        df,
        stacked_micro_all_df,
        how="left",
        left_on=["Date", "ticker"],
        right_on=["Date", "ticker"],
    )
    df_micro_macro_aware = pd.merge(
        df_micro_aware, macro_technical_compacted, on="Date", how="left"
    )
    return df_micro_macro_aware


if __name__ == "__main__":
    data = get_micro_macro_aware_data()
