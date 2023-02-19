import glob

import pandas as pd
from datetime import timedelta
from preprocess.calculate_technical_indicators import get_technical_indicators


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


def get_one_day_technical_discussion(use_path: list):
    technical_data = []
    for ticker_path in use_path:
        ind_ticker = ticker_path.split("/")[-1].split("_")[0]
        ind_data = pd.read_csv(ticker_path)
        ind_technical_indicators = fill_weekends_and_holidays_with_previous_values(
            get_technical_indicators(ind_data)
        ).reset_index()
        ind_technical_indicators = ind_technical_indicators.assign(ticker=ind_ticker)
        technical_data.append(ind_technical_indicators)
    return pd.concat(technical_data, axis=0).sort_values(["Date", "ticker"])


def get_n_days_technical_discussion(n: int, project_data_path: str):
    discussion = pd.read_csv(project_data_path + "hot_discussion.csv")
    data_path = glob.glob(project_data_path + "historical_data/*")
    use_path = []
    for i in data_path:
        if (
            i.split("/")[-1].split("_")[0].replace("-", ".")
            in discussion.ticker.unique()
        ):
            use_path.append(i)
    first_discussion = get_one_day_technical_discussion(use_path=use_path)
    if n == 1:
        return first_discussion.sort_values(["Date", "ticker"])
    sequential_data = []
    for i_len in range(n):
        sequential_data.append(
            first_discussion.set_index(["Date", "ticker"])
            .groupby(by="ticker")
            .shift(i_len)
            .groupby(by="ticker")
            .fillna(method="bfill")
        )
    return pd.concat(sequential_data, axis=1).reset_index()


if __name__ == "__main__":
    discussion = pd.read_csv("./data/hot_discussion.csv")
    data_path = glob.glob("../data/historical_data/*")
    use_path = []
    for i in data_path:
        if (
            i.split("/")[-1].split("_")[0].replace("-", ".")
            in discussion.ticker.unique()
        ):
            use_path.append(i)

    technical_data = get_n_days_technical_discussion(n=7, project_data_path="./data/")
