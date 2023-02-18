import pandas as pd
from pandarallel import pandarallel

# Setup
pandarallel.initialize(progress_bar=True)


def exclude_invalid_cashtags(data: pd.DataFrame) -> pd.DataFrame:

    valid_cashtags = set(data.ticker.unique())
    data = data[~data["entities.cashtags"].isna()]

    def func(x):
        return set(
            each_cashtag.upper()
            for each_cashtag in x
            if each_cashtag.upper() in valid_cashtags
        )

    cashtags_list = data["entities.cashtags"].map(turn_str_dict_into_list)
    valid_cashtags = cashtags_list.map(func)
    return data.assign(valid_cashtags=valid_cashtags).drop("entities.cashtags", axis=1)


def turn_str_dict_into_list(x):
    return list(map(lambda y: y.split('tag": "')[-1][0:-1], x.split("}")))[:-1]


def user_filter_out_tweets_with_more_than_two_valid_cashtags(
    data_with_valid_cashtag: pd.DataFrame,
) -> pd.DataFrame:
    assert "valid_cashtags" in data_with_valid_cashtag.columns
    authorid_id_data = data_with_valid_cashtag.groupby(["author.id", "id"]).size()
    authorid_all = set(data_with_valid_cashtag["author.id"].unique())
    author_id_with_more_than_two_castag = set(
        authorid_id_data[authorid_id_data != 1].reset_index()["author.id"].unique()
    )
    author_id_with_one_cashtag = authorid_all - author_id_with_more_than_two_castag
    return data_with_valid_cashtag[
        data_with_valid_cashtag["author.id"].isin(author_id_with_one_cashtag)
    ]


def user_filter_with_number_of_stocks(
    data: pd.DataFrame, low_bar: int = 20, high_bar: int = 100
) -> pd.DataFrame:
    number_of_stocks_per_investor = (
        data.groupby("author.id")["ticker"].apply(set).apply(len)
    )

    investors_bool = (
        ((number_of_stocks_per_investor < 100) & (number_of_stocks_per_investor >= 10))
        .reset_index()
        .set_axis(["author.id", "boolean_author"], axis=1)
        .reset_index(drop=True)
    )

    filtered_dataset = pd.merge(data, investors_bool, on="author.id", how="left").query(
        "boolean_author==True"
    )

    return filtered_dataset[
        ["id", "created_at", "text", "author.id", "ticker", "valid_cashtags"]
    ]


def get_filtered_data(data_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    # Concatenate all the data with ticker code
    all_tweets_with_ticker_code = pd.concat(
        [df.assign(ticker=ticker_code) for ticker_code, df in data.items()], axis=0
    ).reset_index(drop=True)
    data_with_valid_cashtag = exclude_invalid_cashtags(all_tweets_with_ticker_code)
    strict_filtered_data = user_filter_out_tweets_with_more_than_two_valid_cashtags(
        data_with_valid_cashtag=data_with_valid_cashtag
    )
    filtered_data = user_filter_with_number_of_stocks(strict_filtered_data)
    return filtered_data


if __name__ == "__main__":
    load_data_path = "../data/load_data"
    filtered_data = get_filtered_data(load_data_path)
