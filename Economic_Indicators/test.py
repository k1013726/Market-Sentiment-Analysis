import numpy as np
import pandas as pd
import datetime as dt

#匯入Yahoo!Finance
#pip install 
import yfinance as yf

yf.pdr_override()
symbol = "BTC-USD"
start =dt.date(2018, 1, 1)
end = dt.date(2023, 1, 1)

# Read data
df = yf.download(symbol, start, end)
print(df)