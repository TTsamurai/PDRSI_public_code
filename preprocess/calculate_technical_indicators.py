from datetime import date
from datetime import datetime as dt

import pandas as pd
import talib
from talib import MA_Type


def get_technical_indicators(
    data: pd.DataFrame,
    start_data_day: date = date(2022, 8, 1),
    end_data_day: date = date(2022, 12, 1),
) -> pd.DataFrame:
    talib_features = get_talib_features(data=data)
    talib_features["Date"] = (
        talib_features["Date"]
        .apply(lambda x: dt.strptime(x, "%Y-%m-%d"))
        .apply(lambda x: date(x.year, x.month, x.day))
    )
    talib_features = talib_features[
        (talib_features["Date"] >= start_data_day)
        & (talib_features["Date"] <= end_data_day)
    ]
    return talib_features


def get_talib_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Get technical features from TA-Lib
    """
    df = data.copy()
    hi = df["High"]
    lo = df["Low"]
    cl = df["Close"]

    # トレンド系
    df["BBANDS_upper"], df["BBANDS_middle"], df["BBANDS_lower"] = talib.BBANDS(
        cl, timeperiod=5, nbdevup=2, nbdevdn=2, matype=MA_Type.EMA
    )
    # the most common settings: timeperiod=14
    df["ADX"] = talib.ADX(hi, lo, cl, timeperiod=14)
    df["MACD_macd"], df["MACD_macdsignal"], MACD_macdhist = talib.MACD(
        cl, fastperiod=12, slowperiod=26, signalperiod=9
    )

    # オシレーター系
    # Relative Strength Index
    # the most common settings: timeperiod=14
    df["RSI"] = talib.RSI(cl, timeperiod=14)
    # Stochastic
    df["STOCH_slowk"], df["STOCH_slowd"] = talib.STOCH(
        hi,
        lo,
        cl,
        fastk_period=5,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )

    return df

