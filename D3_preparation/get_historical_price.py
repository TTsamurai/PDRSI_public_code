from datetime import date

import yfinance as yfin
from pandas_datareader import data as pdr

yfin.pdr_override()

endday = date(2022, 12, 1)
startday = date(2022, 8, 1)
with open("data_preparation/ticker_SP.txt", "r") as f:
    ticker_txt = f.read()

# In yahoo finacnce, BF.B and BRK.B are shown as BF-B and BRK-B. Therefore, we need to change period into hyphen
ticker_list = [
    ticker.replace("'", "").replace(".", "-")
    for ticker in ticker_txt.split()
    if ticker != ""
]

for ticker in ticker_list:
    data = pdr.get_data_yahoo(ticker, start=startday, end=endday)
    data.to_csv(f"./data/historical_data/{ticker}_historical.csv")
