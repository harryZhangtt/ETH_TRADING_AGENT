import datetime as dt
import yfinance as yf
import pandas as pd

# from 2015-01-01 to today
start = "2015-01-01"
end = dt.date.today().strftime("%Y-%m-%d")

# ticker list
tickers = {
    "sp500": "^GSPC",        # S&P 500
    "eurostoxx": "^STOXX50E",# Euro Stoxx 50
    "dow30": "^DJI",         # Dow Jones 30
    "nasdaq": "^IXIC",       # Nasdaq Composite
    "crude_oil": "CL=F",     # WTI Crude Oil Futures
    "sse": "000001.SS",      # Shanghai Composite Index
    "gold": "GC=F",          # Gold Futures
    "vix": "^VIX",           # VIX
    "nikkei225": "^N225",    # Nikkei 225
    "ftse100": "^FTSE"       # FTSE 100
}

for name, ticker in tickers.items():
    print(f"Downloading {name} ({ticker}) ...")
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)

    if df.empty:
        print(f"Warning: no data returned for {ticker}, skip.")
        continue

    #  MultiIndex Adj Close, Close
    if isinstance(df.columns, pd.MultiIndex):
        price_level = df.columns.get_level_values(0)

        target_name = None
        if "Adj Close" in price_level:
            target_name = "Adj Close"
        elif "Close" in price_level:
            target_name = "Close"

        if target_name is None:
            print(f"Warning: neither 'Adj Close' nor 'Close' found in columns for {ticker}, available columns: {df.columns}")
            continue

        price_cols = [c for c in df.columns if c[0] == target_name]
        col = price_cols[0]
        series = df[col].rename(target_name)
    else:
        if "Adj Close" in df.columns:
            series = df["Adj Close"]
        elif "Close" in df.columns:
            series = df["Close"]
        else:
            print(f"Warning: neither 'Adj Close' nor 'Close' found in columns for {ticker}, available columns: {df.columns}")
            continue

    out_df = series.reset_index()  # Date 

    #  <Adj Close / Close>
    filename = f"{name}_price_2015.csv"
    out_df.to_csv(filename, index=False)
    print(f"Saved to {filename}")